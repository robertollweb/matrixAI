# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""composite_model_hash — deterministic hash chain over imported component entry_hashes."""
from __future__ import annotations

import json
from typing import Any

from matrixai.registry.entry_hash import sha256_str
from matrixai.registry.model_registry import EntryNotFoundError, VerificationError


def compute_composite_model_hash(program: Any, registry: Any) -> str:
    """Return sha256 over the canonical program structure and all imports.

    The payload is:
        {
          "own_program_hash": sha256(<local program without imports>),
          "imports": [
              {"alias": ..., "mode": ..., "entry_hash": ...},  # sorted by alias
              ...
          ]
        }

    Changing the local GRAPH, NETWORK, or any local definition changes the hash.
    Changing an import's mode (FROZEN ↔ TRAINABLE) changes the hash.
    Changing an import's version changes its entry_hash, which changes the hash.
    Import declaration order does not affect the hash.

    Raises ImportResolutionError if any import is missing from the registry.
    """
    from matrixai.registry.resolver import ImportResolutionError

    # Canonical hash of the local program (everything except the imports list).
    own_dict = program.to_dict()
    own_dict.pop("imports", None)
    own_program_hash = sha256_str(
        json.dumps(own_dict, sort_keys=True, separators=(",", ":"))
    )

    # Per-import entries sorted by alias for determinism.
    imports_list: list[dict[str, str]] = []
    for imp in sorted(getattr(program, "imports", []), key=lambda x: x.alias):
        try:
            entry = registry.get(imp.registry_name, imp.version)
        except EntryNotFoundError as exc:
            raise ImportResolutionError(
                f"Cannot compute composite_model_hash: {imp.alias!r} "
                f"({imp.registry_name}@{imp.version}) not in registry"
            ) from exc
        imports_list.append({
            "alias": imp.alias,
            "mode": imp.mode,
            "entry_hash": entry.entry_hash,
        })

    payload = json.dumps(
        {"own_program_hash": own_program_hash, "imports": imports_list},
        sort_keys=True, separators=(",", ":"),
    )
    return sha256_str(payload)


def verify_composite_model(program: Any, registry: Any) -> bool:
    """Verify integrity of all imported components in the registry.

    1. Each imported component's entry_hash must match the recalculated hash.
    2. The composite_model_hash must be stable (no retroactive changes).

    Raises VerificationError if any component fails.
    """
    from matrixai.registry.entry_hash import compute_entry_hash

    for imp in getattr(program, "imports", []):
        try:
            entry = registry.get(imp.registry_name, imp.version)
        except EntryNotFoundError as exc:
            raise VerificationError(
                f"Imported component {imp.alias!r} ({imp.registry_name}@{imp.version}) not found"
            ) from exc

        expected = compute_entry_hash(
            name=entry.name,
            version=entry.version,
            model_hash=entry.model_hash,
            parameter_schema_hash=entry.parameter_schema_hash,
            parameter_set_id=entry.parameter_set_id,
            training_trace_hash=entry.training_trace_hash,
            evaluation_report_hash=entry.evaluation_report_hash,
            matrixai_version=entry.matrixai_version,
        )
        if entry.entry_hash != expected:
            raise VerificationError(
                f"Imported component {imp.alias!r} has tampered entry_hash: "
                f"stored {entry.entry_hash!r} != computed {expected!r}"
            )

    return True
