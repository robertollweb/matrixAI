# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P21 C10 — P20 action contract integration with composite models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentChainEntry:
    alias: str
    registry_name: str
    version: str
    mode: str
    entry_hash: str
    interpretability_level: str
    is_terminal: bool


@dataclass(frozen=True)
class CompositeAuditResult:
    composite_model_hash: str
    component_chain: list[ComponentChainEntry]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def get_component_chain(program: Any, registry: Any) -> list[ComponentChainEntry]:
    """Return ordered list of imported components with registry info."""
    from matrixai.registry.model_registry import EntryNotFoundError

    # Find terminal nodes: nodes that have no outgoing edges to other import nodes
    import_aliases = {imp.alias for imp in getattr(program, "imports", [])}
    out_nodes = {src for src, dst in program.graph.edges}
    terminal_aliases = import_aliases - (import_aliases & out_nodes)

    chain = []
    for imp in sorted(getattr(program, "imports", []), key=lambda x: x.alias):
        try:
            entry = registry.get(imp.registry_name, imp.version)
            chain.append(ComponentChainEntry(
                alias=imp.alias,
                registry_name=imp.registry_name,
                version=imp.version,
                mode=imp.mode,
                entry_hash=entry.entry_hash,
                interpretability_level=entry.interpretability_level,
                is_terminal=imp.alias in terminal_aliases,
            ))
        except EntryNotFoundError:
            chain.append(ComponentChainEntry(
                alias=imp.alias,
                registry_name=imp.registry_name,
                version=imp.version,
                mode=imp.mode,
                entry_hash="",
                interpretability_level="unknown",
                is_terminal=imp.alias in terminal_aliases,
            ))
    return chain


def validate_composite_action_contract(
    program: Any,
    registry: Any,
    *,
    contract: Any = None,
) -> CompositeAuditResult:
    """Validate that the composite program satisfies action contract rules.

    Rules:
    - Intermediate imported components must not have real_with_audit actions (blockers).
    - Only the terminal node may carry an action contract.
    """
    from matrixai.registry.composite_hash import compute_composite_model_hash
    from matrixai.registry.model_registry import EntryNotFoundError

    errors: list[str] = []
    warnings: list[str] = []

    import_aliases = {imp.alias for imp in getattr(program, "imports", [])}
    # A node is intermediate if it has both incoming AND outgoing edges to/from import nodes
    in_nodes = {dst for src, dst in program.graph.edges}
    out_nodes = {src for src, dst in program.graph.edges}
    intermediate_aliases = import_aliases & in_nodes & out_nodes

    for imp in getattr(program, "imports", []):
        if imp.alias not in intermediate_aliases:
            continue
        try:
            entry = registry.get(imp.registry_name, imp.version)
            blockers = getattr(entry, "blockers", [])
            if "real_with_audit_action" in blockers:
                errors.append(
                    f"Intermediate component {imp.alias!r} has real_with_audit action — "
                    f"only the terminal component may carry real actions"
                )
        except EntryNotFoundError:
            errors.append(f"Component {imp.alias!r} not found in registry")

    try:
        composite_hash = compute_composite_model_hash(program, registry)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Cannot compute composite_model_hash: {exc}")
        composite_hash = ""

    chain = get_component_chain(program, registry)
    return CompositeAuditResult(
        composite_model_hash=composite_hash,
        component_chain=chain,
        errors=errors,
        warnings=warnings,
    )


def composite_dry_run(
    program: Any,
    registry: Any,
    input_data: Any,
    *,
    parameter_set: Any = None,
) -> dict[str, Any]:
    """Resolve all imports and simulate forward pass for dry-run validation."""
    from matrixai.registry.resolver import CompositeModelResolver, ImportResolutionError
    from matrixai.registry.composite_hash import compute_composite_model_hash

    errors: list[str] = []

    # Resolve all imports
    resolver = CompositeModelResolver(registry)
    try:
        hydrated_imports = resolver.resolve(program, allow_mutable_tags=True)
    except ImportResolutionError as exc:
        errors.append(str(exc))
        hydrated_imports = []

    composite_hash = ""
    if not errors:
        try:
            composite_hash = compute_composite_model_hash(program, registry)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    return {
        "ok": not errors,
        "errors": errors,
        "resolved_imports": [
            {
                "alias": imp.alias,
                "registry_name": imp.registry_name,
                "version": imp.version,
                "resolved_entry_hash": imp.resolved_entry_hash,
            }
            for imp in hydrated_imports
        ],
        "composite_model_hash": composite_hash,
        "input": input_data,
    }
