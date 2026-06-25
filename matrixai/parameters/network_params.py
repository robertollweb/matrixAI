# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18/P19 — Parameter manifest y ParameterSet para redes densas y compuestas."""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

from matrixai.parameters.store import (
    ParameterCompatibilityResult,
    ParameterSet,
    parameter_schema_hash,
)

_INITIALIZER_FOR_ACTIVATION: dict[str, str] = {
    "relu": "he_normal",
    "linear": "xavier_normal",
    "sigmoid": "xavier_normal",
    "softmax": "xavier_normal",
    "tanh": "xavier_normal",
}


def network_parameter_manifest(network_name: str, resolved_layers: list[Any]) -> list[dict[str, Any]]:
    """Build the parameter manifest for a dense network given shape-resolved layers (from C2)."""
    manifest: list[dict[str, Any]] = []
    for layer in resolved_layers:
        input_dim = layer.input_shape[0]
        units = layer.units
        initializer = _INITIALIZER_FOR_ACTIVATION.get(layer.activation, "xavier_normal")

        manifest.append({
            "function": network_name,
            "name": f"W{layer.index}",
            "path": f"{network_name}.W{layer.index}",
            "role": "weights",
            "shape": [units, input_dim],
            "dtype": "float32",
            "initializer": initializer,
        })
        manifest.append({
            "function": network_name,
            "name": f"b{layer.index}",
            "path": f"{network_name}.b{layer.index}",
            "role": "bias",
            "shape": [units],
            "dtype": "float32",
            "initializer": "zeros",
        })
    return manifest


def network_parameter_schema_hash(
    network_name: str,
    resolved_layers: list[Any],
    output_name: str = "",
) -> str:
    """Compute parameter_schema_hash for a dense network.

    output_name: when provided, the hash is sensitive to the output field name,
    so renaming the network output changes the schema hash.
    """
    manifest = network_parameter_manifest(network_name, resolved_layers)
    base = parameter_schema_hash(manifest)
    if not output_name:
        return base
    combined = json.dumps({"base": base, "output_name": output_name}, sort_keys=True)
    return "params_" + hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def build_network_parameter_set(
    network: Any,
    resolved_layers: list[Any],
    model_hash_str: str,
    parameter_set_id: str | None = None,
    seed: int = 42,
    output_name: str = "",
    with_values: bool = True,
) -> ParameterSet:
    """Build an initial ParameterSet for a dense network with He/Xavier initialization.

    M15(a): con ``with_values=False`` devuelve una **plantilla de estructura** (shapes,
    manifest) con ``values=None`` — sin generar pesos en Python. La usan los caminos torch,
    donde el módulo se inicializa con el init nativo de torch (Kaiming); los pesos
    construidos a mano serían redundantes (el entrenamiento los reemplaza). Evita el coste
    O(params) en Python para redes anchas.
    """
    manifest = network_parameter_manifest(network.name, resolved_layers)
    schema_digest = network_parameter_schema_hash(network.name, resolved_layers, output_name)
    rng = random.Random(seed)
    parameters: dict[str, dict[str, Any]] = {}

    for entry in manifest:
        path = entry["path"]
        shape = entry["shape"]
        initializer = entry["initializer"]

        if not with_values:
            values: Any = None
        elif initializer == "zeros":
            values = [0.0] * shape[0]
        else:
            values = _init_weights(shape, initializer, rng)

        parameters[path] = {
            "function": entry["function"],
            "role": entry["role"],
            "type": _type_for_shape(shape),
            "shape": shape,
            "dtype": "float32",
            "initializer": initializer,
            "values": values,
            "is_layer": True,
        }

    return ParameterSet(
        parameter_set_id=parameter_set_id or f"{network.name}_initial",
        model_hash=model_hash_str,
        parameter_schema_hash=schema_digest,
        source="initial",
        parameters=parameters,
    )


def validate_network_parameter_set(
    network: Any,
    resolved_layers: list[Any],
    parameter_set: ParameterSet,
    model_hash_str: str,
    output_name: str = "",
) -> ParameterCompatibilityResult:
    """Validate a ParameterSet against the network's expected manifest."""
    manifest = network_parameter_manifest(network.name, resolved_layers)
    schema_digest = network_parameter_schema_hash(network.name, resolved_layers, output_name)
    errors: list[str] = []
    warnings: list[str] = []

    if parameter_set.model_hash != model_hash_str:
        errors.append(
            f"model_hash mismatch: expected {model_hash_str}, got {parameter_set.model_hash}"
        )
    if parameter_set.parameter_schema_hash != schema_digest:
        errors.append(
            f"parameter_schema_hash mismatch: expected {schema_digest}, "
            f"got {parameter_set.parameter_schema_hash}"
        )

    actual = parameter_set.parameters
    for entry in manifest:
        path = entry["path"]
        param = actual.get(path)
        if param is None:
            errors.append(f"ParameterSet missing parameter: {path}")
            continue
        expected_shape = entry["shape"]
        if list(param.get("shape", [])) != expected_shape:
            errors.append(
                f"Parameter {path} expected shape {expected_shape}, "
                f"got {param.get('shape', [])}"
            )
        err = _validate_value_shape(path, param.get("values"), expected_shape)
        if err:
            errors.append(err)

    expected_paths = {e["path"] for e in manifest}
    for key in sorted(set(actual) - expected_paths):
        errors.append(f"ParameterSet contains unexpected parameter: {key}")

    return ParameterCompatibilityResult(
        errors=errors,
        warnings=warnings,
        model_hash=model_hash_str,
        parameter_schema_hash=schema_digest,
    )


# ---------------------------------------------------------------------------
# P19 — Composite network parameter manifest
# ---------------------------------------------------------------------------

def composite_network_parameter_manifest(
    network_name: str,
    network: Any,
    type_result: Any,
) -> list[dict[str, Any]]:
    """Build parameter manifest for a composite_network (P19).

    network: NetworkSpec (provides embeddings with vocab/dim/name)
    type_result: NetworkTypeResult from check_composite_network_types
                 (provides resolved_layers and resolved_blocks with shapes)
    """
    manifest: list[dict[str, Any]] = []

    # Embedding tables
    for emb in getattr(network, "embeddings", []):
        manifest.append({
            "function": network_name,
            "name": f"{emb.name}.table",
            "path": f"{network_name}.{emb.name}.table",
            "role": "embedding_table",
            "shape": [emb.vocab, emb.dim],
            "dtype": "float32",
            "initializer": "xavier_normal",
        })

    # Interleave top_layers and blocks by textual position (same key as C2)
    body_items: list[tuple[int, str, Any]] = []
    for layer in getattr(type_result, "resolved_layers", []):
        body_items.append((layer.index * 2, "layer", layer))
    for block in getattr(type_result, "resolved_blocks", []):
        pos = getattr(block, "position", 0)
        body_items.append((pos * 2 + 1, "block", block))
    body_items.sort(key=lambda x: x[0])

    for _, kind, spec in body_items:
        if kind == "layer":
            _append_composite_layer_params(manifest, network_name, network_name, spec)
        else:
            block = spec
            block_prefix = f"{network_name}.{block.name}"
            for layer in block.layers:
                _append_composite_layer_params(manifest, network_name, block_prefix, layer)

    return manifest


def _append_composite_layer_params(
    manifest: list[dict[str, Any]],
    function_name: str,
    prefix: str,
    layer: Any,
) -> None:
    """Append parameter entries for a single CompositeLayerSpec."""
    if layer.layer_type == "Dense":
        in_dim = layer.input_shape[-1] if layer.input_shape else 0
        units = layer.units
        initializer = _INITIALIZER_FOR_ACTIVATION.get(layer.activation, "xavier_normal")
        manifest.append({
            "function": function_name,
            "name": f"L{layer.index}.W",
            "path": f"{prefix}.L{layer.index}.W",
            "role": "weights",
            "shape": [units, in_dim],
            "dtype": "float32",
            "initializer": initializer,
        })
        manifest.append({
            "function": function_name,
            "name": f"L{layer.index}.b",
            "path": f"{prefix}.L{layer.index}.b",
            "role": "bias",
            "shape": [units],
            "dtype": "float32",
            "initializer": "zeros",
        })
    elif layer.layer_type == "LayerNorm":
        features = layer.input_shape[-1] if layer.input_shape else 0
        manifest.append({
            "function": function_name,
            "name": f"L{layer.index}.gamma",
            "path": f"{prefix}.L{layer.index}.gamma",
            "role": "gamma",
            "shape": [features],
            "dtype": "float32",
            "initializer": "ones",
        })
        manifest.append({
            "function": function_name,
            "name": f"L{layer.index}.beta",
            "path": f"{prefix}.L{layer.index}.beta",
            "role": "beta",
            "shape": [features],
            "dtype": "float32",
            "initializer": "zeros",
        })
    # Dropout, Activation, Pool, Reshape — no parameters


def _composite_architecture_summary(network: Any, type_result: Any) -> dict[str, Any]:
    """Canonical architecture structure for hash computation.

    Includes residual sources and embedding sources so that structural changes
    (not just parameter shape changes) alter the schema hash.
    """
    emb_summary = [
        {"name": e.name, "source": e.source, "vocab": e.vocab, "dim": e.dim}
        for e in getattr(network, "embeddings", [])
    ]
    block_summary = [
        {
            "name": b.name,
            "residual_from": b.residual_from,
            "layer_types": [l.layer_type for l in b.layers],
        }
        for b in getattr(type_result, "resolved_blocks", [])
    ]
    return {"embeddings": emb_summary, "blocks": block_summary}


def composite_network_parameter_schema_hash(
    network_name: str,
    network: Any,
    type_result: Any,
    output_name: str = "",
) -> str:
    """Compute parameter_schema_hash for a composite_network (P19)."""
    manifest = composite_network_parameter_manifest(network_name, network, type_result)
    arch = _composite_architecture_summary(network, type_result)
    combined = json.dumps(
        {"network": network_name, "manifest": manifest, "arch": arch, "output_name": output_name},
        sort_keys=True,
    )
    return "params_" + hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def build_composite_network_parameter_set(
    network: Any,
    type_result: Any,
    model_hash_str: str,
    parameter_set_id: str | None = None,
    seed: int = 42,
    output_name: str = "",
    with_values: bool = True,
) -> ParameterSet:
    """Build an initial ParameterSet for a composite_network (P19).

    M15(f): con `with_values=False` devuelve solo la plantilla de estructura (shapes, sin
    generar los pesos en Python). El camino torch usa el init nativo de los módulos
    (`composite_network_to_torch_module` lee `shape` y deja el init de nn.* cuando no hay
    valores), evitando el coste O(params) de materializar listas. Espejo de M15(a) del denso.
    El camino stdlib y el export siguen usando `with_values=True` (necesitan los valores)."""
    manifest = composite_network_parameter_manifest(network.name, network, type_result)
    schema_digest = composite_network_parameter_schema_hash(
        network.name, network, type_result, output_name
    )
    rng = random.Random(seed)
    parameters: dict[str, dict[str, Any]] = {}

    for entry in manifest:
        path = entry["path"]
        shape = entry["shape"]
        initializer = entry["initializer"]

        if not with_values:
            values: Any = None
        elif initializer == "zeros":
            values = [0.0] * shape[0]
        elif initializer == "ones":
            values = [1.0] * shape[0]
        elif len(shape) == 1:
            std = math.sqrt(1.0 / shape[0])
            values = [rng.gauss(0.0, std) for _ in range(shape[0])]
        else:
            values = _init_weights(shape, initializer, rng)

        parameters[path] = {
            "function": entry["function"],
            "role": entry["role"],
            "type": _type_for_shape(shape),
            "shape": shape,
            "dtype": "float32",
            "initializer": initializer,
            "values": values,
            "is_layer": True,
        }

    return ParameterSet(
        parameter_set_id=parameter_set_id or f"{network.name}_initial",
        model_hash=model_hash_str,
        parameter_schema_hash=schema_digest,
        source="initial",
        parameters=parameters,
    )


def validate_composite_network_parameter_set(
    network: Any,
    type_result: Any,
    parameter_set: ParameterSet,
    model_hash_str: str,
    output_name: str = "",
) -> ParameterCompatibilityResult:
    """Validate a ParameterSet against the composite network's expected manifest."""
    manifest = composite_network_parameter_manifest(network.name, network, type_result)
    schema_digest = composite_network_parameter_schema_hash(
        network.name, network, type_result, output_name
    )
    errors: list[str] = []
    warnings: list[str] = []

    if parameter_set.model_hash != model_hash_str:
        errors.append(
            f"model_hash mismatch: expected {model_hash_str}, got {parameter_set.model_hash}"
        )
    if parameter_set.parameter_schema_hash != schema_digest:
        errors.append(
            f"parameter_schema_hash mismatch: expected {schema_digest}, "
            f"got {parameter_set.parameter_schema_hash}"
        )

    actual = parameter_set.parameters
    for entry in manifest:
        path = entry["path"]
        param = actual.get(path)
        if param is None:
            errors.append(f"ParameterSet missing parameter: {path}")
            continue
        expected_shape = entry["shape"]
        if list(param.get("shape", [])) != expected_shape:
            errors.append(
                f"Parameter {path} expected shape {expected_shape}, "
                f"got {param.get('shape', [])}"
            )
        err = _validate_value_shape(path, param.get("values"), expected_shape)
        if err:
            errors.append(err)

    expected_paths = {e["path"] for e in manifest}
    for key in sorted(set(actual) - expected_paths):
        errors.append(f"ParameterSet contains unexpected parameter: {key}")

    return ParameterCompatibilityResult(
        errors=errors,
        warnings=warnings,
        model_hash=model_hash_str,
        parameter_schema_hash=schema_digest,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_weights(shape: list[int], initializer: str, rng: random.Random) -> list[list[float]]:
    rows, cols = shape[0], shape[1]
    if initializer == "he_normal":
        std = math.sqrt(2.0 / cols)
    else:  # xavier_normal
        std = math.sqrt(2.0 / (rows + cols))
    return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]


def _type_for_shape(shape: list[int]) -> str:
    if not shape:
        return "Scalar"
    if len(shape) == 1:
        return f"Vector[{shape[0]}]"
    return "Tensor[" + ",".join(str(d) for d in shape) + "]"


def _validate_value_shape(name: str, value: Any, expected_shape: list[int]) -> str:
    try:
        actual = _value_shape(value)
    except ValueError as exc:
        return f"Parameter {name} invalid values: {exc}"
    if actual != expected_shape:
        return f"Parameter {name} expected values shape {expected_shape}, got {actual}"
    return ""


def _value_shape(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        first = _value_shape(value[0])
        for item in value[1:]:
            if _value_shape(item) != first:
                raise ValueError("ragged values")
        return [len(value)] + first
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric value {value!r}") from exc
    return []
