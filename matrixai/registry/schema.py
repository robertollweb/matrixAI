# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MATRIXAI_REGISTRY_SCHEMA_VERSION = "1.0.0"

# Backwards-compatible alias for callers that imported the original constant.
MATRIXAI_REGISTRY_VERSION = MATRIXAI_REGISTRY_SCHEMA_VERSION


@dataclass(frozen=True)
class RegistryEntry:
    """Immutable record for a single versioned model in the registry."""

    name: str
    version: str
    entry_hash: str                  # sha256:... covers all identity fields
    model_hash: str                  # sha256:... of the .mxai canonical content
    parameter_schema_hash: str       # sha256:... of trainable-params schema
    parameter_set_id: str
    input_type: dict                 # {name, kind, fields/shape/...}
    output_type: dict                # {name, kind, fields/shape/...}
    metrics: dict                    # {loss: float, accuracy: float, ...}
    matrixai_version: str
    created_at: str                  # ISO8601 UTC
    training_dataset_fingerprint: str
    interpretability_level: str      # "full" | "reduced" | "very_reduced"
    training_trace_hash: str = ""    # sha256:... stored to enable re-verification
    evaluation_report_hash: str = "" # sha256:... required for push
    params_content_hash: str = ""   # sha256:... of full params.json bytes, or of the
                                     # weights.mxw body (TRANSFORMER C6 / PESOS_GRANDES,
                                     # covers weights above the materialization threshold);
                                     # enables file-level tamper detection
    blockers: list[str] = field(default_factory=list)  # non-empty → propagated as errors by BackendContractAnalyzer

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "entry_hash": self.entry_hash,
            "model_hash": self.model_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
            "parameter_set_id": self.parameter_set_id,
            "input_type": dict(self.input_type),
            "output_type": dict(self.output_type),
            "metrics": dict(self.metrics),
            "matrixai_version": self.matrixai_version,
            "created_at": self.created_at,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "interpretability_level": self.interpretability_level,
            "training_trace_hash": self.training_trace_hash,
            "evaluation_report_hash": self.evaluation_report_hash,
            "params_content_hash": self.params_content_hash,
            "blockers": list(self.blockers),
        }

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> "RegistryEntry":
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            entry_hash=str(data["entry_hash"]),
            model_hash=str(data["model_hash"]),
            parameter_schema_hash=str(data["parameter_schema_hash"]),
            parameter_set_id=str(data["parameter_set_id"]),
            input_type=dict(data["input_type"]),
            output_type=dict(data["output_type"]),
            metrics=dict(data.get("metrics", {})),
            matrixai_version=str(data["matrixai_version"]),
            created_at=str(data["created_at"]),
            training_dataset_fingerprint=str(data.get("training_dataset_fingerprint", "")),
            interpretability_level=str(data.get("interpretability_level", "full")),
            training_trace_hash=str(data.get("training_trace_hash", "")),
            evaluation_report_hash=str(data.get("evaluation_report_hash", "")),
            params_content_hash=str(data.get("params_content_hash", "")),
            blockers=list(data.get("blockers", [])),
        )


class RegistryEntryError(Exception):
    """Raised when a registry entry is corrupt, missing, or has an invalid signature."""
