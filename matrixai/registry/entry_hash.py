# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import json


def _canonical_json(obj: dict) -> str:
    """Deterministic JSON: sorted keys, no extra whitespace, ASCII-safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_str(text: str) -> str:
    """Return 'sha256:<hex>' of the UTF-8 encoding of text."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return 'sha256:<hex>' of raw bytes (for hashing file contents)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def compute_entry_hash(
    *,
    name: str,
    version: str,
    model_hash: str,
    parameter_schema_hash: str,
    parameter_set_id: str,
    training_trace_hash: str,
    evaluation_report_hash: str,
    matrixai_version: str,
    params_content_hash: str = "",
) -> str:
    """Return 'sha256:<hex>' covering all identity fields of a registry entry.

    Metrics, input/output types, and timestamps are excluded so they can be
    annotated without invalidating the entry_hash.

    params_content_hash is omitted from the payload when empty so that entries
    created before this field existed still verify correctly.
    """
    payload = {
        "name": name,
        "version": version,
        "model_hash": model_hash,
        "parameter_schema_hash": parameter_schema_hash,
        "parameter_set_id": parameter_set_id,
        "training_trace_hash": training_trace_hash,
        "evaluation_report_hash": evaluation_report_hash,
        "matrixai_version": matrixai_version,
    }
    if params_content_hash:
        payload["params_content_hash"] = params_content_hash
    return sha256_str(_canonical_json(payload))
