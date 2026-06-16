# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matrixai.compiler import BackendContractAnalyzer
from matrixai.ir import MatrixAIProgram


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(data: Any) -> str:
    return hashlib.sha256(_stable_json(data).encode("utf-8")).hexdigest()[:16]


def program_hash(program: MatrixAIProgram) -> str:
    return "mxai_" + _digest(program.to_dict())


def parameter_schema_hash(parameter_manifest: list[dict[str, Any]]) -> str:
    schema = []
    for parameter in parameter_manifest:
        schema.append(
            {
                "function": parameter.get("function"),
                "name": parameter.get("name"),
                "role": parameter.get("role"),
                "shape": parameter.get("shape", []),
                "dtype": parameter.get("dtype"),
                "initializer": parameter.get("initializer"),
            }
        )
    return "params_" + _digest(schema)


@dataclass(frozen=True)
class ParameterSet:
    parameter_set_id: str
    model_hash: str
    parameter_schema_hash: str
    parameters: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    source: str = "initial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_set_id": self.parameter_set_id,
            "model_hash": self.model_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
            "source": self.source,
            "parameters": deepcopy(self.parameters),
            "metrics": deepcopy(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSet:
        return cls(
            parameter_set_id=str(data["parameter_set_id"]),
            model_hash=str(data["model_hash"]),
            parameter_schema_hash=str(data["parameter_schema_hash"]),
            source=str(data.get("source", "unknown")),
            parameters={str(key): dict(value) for key, value in data.get("parameters", {}).items()},
            metrics=dict(data.get("metrics", {})),
        )

    def runtime_parameters(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, parameter in self.parameters.items():
            value = parameter.get("values")
            function = parameter.get("function")
            values[name] = value
            # For flat params (no dot in key), also expose as function.name
            if function and "." not in name:
                values[f"{function}.{name}"] = value
        return values


@dataclass(frozen=True)
class ParameterCompatibilityResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_hash: str = ""
    parameter_schema_hash: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "model_hash": self.model_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
        }

    def summary(self) -> str:
        status = "ok" if self.ok else "blocked"
        lines = [f"ParameterSet validation: {status}"]
        if self.model_hash:
            lines.append(f"Model hash: {self.model_hash}")
        if self.parameter_schema_hash:
            lines.append(f"Parameter schema hash: {self.parameter_schema_hash}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        for error in self.errors:
            lines.append(f"Error: {error}")
        return "\n".join(lines)


class ParameterStore:
    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)

    def write(self, name: str, parameter_set: ParameterSet) -> Path:
        self.base_path.mkdir(parents=True, exist_ok=True)
        path = self.base_path / name
        write_parameter_set(path, parameter_set)
        return path

    def load(self, name: str) -> ParameterSet:
        return load_parameter_set(self.base_path / name)


def build_initial_parameter_set(
    program: MatrixAIProgram,
    parameter_set_id: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> ParameterSet:
    report = BackendContractAnalyzer().analyze(program)
    if not report.ok:
        issues = [node.node for node in report.unsupported_nodes] + list(report.parameter_errors)
        blocked = ", ".join(issues)
        raise ValueError(f"Program {program.project} is not portable to differentiable backend: {blocked}")
    if not report.parameter_manifest:
        raise ValueError(f"Program {program.project} has no trainable parameters")

    model_digest = program_hash(program)
    schema_digest = parameter_schema_hash(report.parameter_manifest)
    parameters: dict[str, dict[str, Any]] = {}
    for manifest in report.parameter_manifest:
        manifest_path = str(manifest.get("path") or manifest["name"])
        flat_name = str(manifest["name"])
        # Use hierarchical path as key for layer params; flat name for top-level params
        key = manifest_path if manifest_path != flat_name else flat_name
        entry: dict[str, Any] = {
            "function": manifest["function"],
            "role": manifest["role"],
            "type": _type_for_shape(manifest.get("shape", [])),
            "shape": list(manifest.get("shape", [])),
            "dtype": manifest.get("dtype", "float32"),
            "initializer": manifest.get("initializer", "unknown"),
            "values": manifest.get("initial_value"),
        }
        if manifest_path != flat_name:
            entry["is_layer"] = True
        parameters[key] = entry

    return ParameterSet(
        parameter_set_id=parameter_set_id or f"{program.project}_initial",
        model_hash=model_digest,
        parameter_schema_hash=schema_digest,
        source="initial",
        parameters=parameters,
        metrics=metrics or {},
    )


def validate_parameter_set(
    program: MatrixAIProgram,
    parameter_set: ParameterSet,
) -> ParameterCompatibilityResult:
    report = BackendContractAnalyzer().analyze(program)
    model_digest = program_hash(program)
    schema_digest = parameter_schema_hash(report.parameter_manifest)
    errors: list[str] = []
    warnings: list[str] = []

    if not report.ok:
        # Action nodes have no trainable parameters — they are never a blocker
        # for parameter compatibility regardless of training/serving/export context.
        non_action_blockers = [
            node for node in report.unsupported_nodes
            if node.node_type != "action"
        ]
        blocker_names = [node.node for node in non_action_blockers] + list(report.parameter_errors)
        if blocker_names:
            blocked = ", ".join(blocker_names)
            errors.append(f"Program is not portable to differentiable backend: {blocked}")
    if parameter_set.model_hash != model_digest:
        errors.append(f"ParameterSet model_hash mismatch: expected {model_digest}, got {parameter_set.model_hash}")
    if parameter_set.parameter_schema_hash != schema_digest:
        errors.append(
            f"ParameterSet parameter_schema_hash mismatch: expected {schema_digest}, "
            f"got {parameter_set.parameter_schema_hash}"
        )

    actual = parameter_set.parameters
    expected: dict[str, dict[str, Any]] = {}
    for manifest in report.parameter_manifest:
        manifest_path = str(manifest.get("path") or manifest["name"])
        flat_name = str(manifest["name"])
        key = manifest_path if manifest_path != flat_name else flat_name
        expected[key] = manifest
    for name, manifest in expected.items():
        parameter = actual.get(name)
        if parameter is None:
            errors.append(f"ParameterSet missing parameter: {name}")
            continue
        expected_shape = list(manifest.get("shape", []))
        actual_shape = parameter.get("shape", [])
        if list(actual_shape) != expected_shape:
            errors.append(f"Parameter {name} expected shape {expected_shape}, got {actual_shape}")
        value_error = _value_shape_error(name, parameter.get("values"), expected_shape)
        if value_error:
            errors.append(value_error)
        expected_dtype = manifest.get("dtype", "float32")
        if parameter.get("dtype", "float32") != expected_dtype:
            errors.append(f"Parameter {name} expected dtype {expected_dtype}, got {parameter.get('dtype')}")

    extras = sorted(set(actual) - set(expected))
    for name in extras:
        errors.append(f"ParameterSet contains unknown parameter: {name}")

    return ParameterCompatibilityResult(
        errors=errors,
        warnings=warnings,
        model_hash=model_digest,
        parameter_schema_hash=schema_digest,
    )


def write_parameter_set(path: str | Path, parameter_set: ParameterSet) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(parameter_set.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_parameter_set(path: str | Path) -> ParameterSet:
    return ParameterSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _type_for_shape(shape: list[int]) -> str:
    if not shape:
        return "Scalar"
    if len(shape) == 1:
        return f"Vector[{shape[0]}]"
    return "Tensor[" + ",".join(str(item) for item in shape) + "]"


def _value_shape_error(name: str, value: Any, expected_shape: list[int]) -> str:
    try:
        actual_shape = _value_shape(value)
    except ValueError as exc:
        return f"Parameter {name} invalid: {exc}"
    if actual_shape != expected_shape:
        return f"Parameter {name} expected values shape {expected_shape}, got {actual_shape}"
    return ""


def _value_shape(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        first_shape = _value_shape(value[0])
        for item in value[1:]:
            if _value_shape(item) != first_shape:
                raise ValueError("Parameter contains ragged values")
        return [len(value)] + first_shape
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter contains non-numeric value {value!r}") from exc
    return []


# ---------------------------------------------------------------------------
# P21 C7 — Hierarchical / composite ParameterSet helpers
# ---------------------------------------------------------------------------

def composite_parameter_schema_hash(
    parameter_manifest: list[dict[str, Any]],
    frozen_aliases: frozenset[str] = frozenset(),
) -> str:
    """Like parameter_schema_hash but skips parameters belonging to frozen imports."""
    filtered = [
        p for p in parameter_manifest
        if p.get("function", "").split(".")[0] not in frozen_aliases
    ]
    return parameter_schema_hash(filtered)


def load_frozen_parameters_from_registry(
    program: Any,
    registry: Any,
) -> dict[str, dict[str, Any]]:
    """Return per-alias metadata for FROZEN imported components from the registry."""
    result: dict[str, dict[str, Any]] = {}
    for imp in getattr(program, "imports", []):
        if imp.mode != "FROZEN":
            continue
        try:
            entry = registry.get(imp.registry_name, imp.version)
            result[imp.alias] = {
                "alias": imp.alias,
                "registry_name": imp.registry_name,
                "version": imp.version,
                "parameter_set_id": entry.parameter_set_id,
                "parameter_schema_hash": entry.parameter_schema_hash,
                "mode": "FROZEN",
            }
        except Exception:  # noqa: BLE001
            pass
    return result


def separate_parameters(
    parameter_set: ParameterSet,
    program: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Return (trainable_params_dict, list_of_frozen_param_keys).

    Frozen keys are those whose prefix (before the first dot) matches a FROZEN import alias.
    """
    frozen_aliases = {
        imp.alias
        for imp in getattr(program, "imports", [])
        if imp.mode == "FROZEN"
    }
    trainable: dict[str, Any] = {}
    frozen_keys: list[str] = []
    for key, value in parameter_set.parameters.items():
        prefix = key.split(".")[0]
        if prefix in frozen_aliases:
            frozen_keys.append(key)
        else:
            trainable[key] = value
    return trainable, frozen_keys
