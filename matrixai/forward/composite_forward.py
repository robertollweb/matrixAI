# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P19 C5 — Forward pass stdlib para redes compuestas (composite_network)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from matrixai.forward.dense_forward import _apply_activation as _dense_apply_activation
from matrixai.ir.schema import get_interleaved_body

EPS_LAYERNORM: float = 1e-5


class CompositeForwardError(ValueError):
    pass


@dataclass
class CompositeForwardTrace:
    """Intermediate values from a composite forward pass — used by backprop (C6)."""
    output: list[float]
    named_tensors: dict[str, list[float]] = field(default_factory=dict)
    # Keyed by path e.g. "Net.L1" or "Net.res1.L1"
    layer_traces: dict[str, dict[str, Any]] = field(default_factory=dict)
    dropout_masks: dict[str, list[float]] = field(default_factory=dict)
    layernorm_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Keyed by block name — the vector added in the residual connection
    residual_vectors: dict[str, list[float]] = field(default_factory=dict)
    # Keyed by block name — the input to the block before any of its layers run
    block_inputs: dict[str, list[float]] = field(default_factory=dict)


def composite_forward(
    network: Any,
    parameter_set: Any,
    input_data: dict[str, Any],
    training: bool = True,
    seed: int = 42,
) -> list[float]:
    """Run a composite network forward pass. Returns the final layer output."""
    return composite_forward_trace(
        network, parameter_set, input_data, training=training, seed=seed
    ).output


def composite_forward_trace(
    network: Any,
    parameter_set: Any,
    input_data: dict[str, Any],
    training: bool = True,
    seed: int = 42,
) -> CompositeForwardTrace:
    """Run a composite network forward pass and return full intermediate trace."""
    named_tensors: dict[str, list[float]] = {}
    layer_traces: dict[str, dict[str, Any]] = {}
    dropout_masks: dict[str, list[float]] = {}
    layernorm_cache: dict[str, dict[str, Any]] = {}
    residual_vectors: dict[str, list[float]] = {}
    block_inputs: dict[str, list[float]] = {}
    rng = random.Random(seed)

    # 1. Populate named_tensors from input fields
    for name, value in input_data.items():
        if isinstance(value, (int, float)):
            named_tensors[name] = [float(value)]
        else:
            named_tensors[name] = [float(v) for v in value]

    # 2. Embedding lookups
    for emb in getattr(network, "embeddings", []):
        table_key = f"{network.name}.{emb.name}.table"
        if table_key not in parameter_set.parameters:
            raise CompositeForwardError(f"Missing embedding table parameter: {table_key!r}")
        table = parameter_set.parameters[table_key]["values"]  # list[list[float]]
        src = named_tensors.get(emb.source)
        if src is None:
            raise CompositeForwardError(
                f"Embedding source field {emb.source!r} not found in input_data"
            )
        idx = int(round(src[0]))
        if idx < 0 or idx >= len(table):
            raise CompositeForwardError(
                f"Embedding {emb.name!r}: index {idx} out of range [0, {len(table)})"
            )
        named_tensors[emb.name] = list(table[idx])

    # 3. Concats (update named_tensors and track current_vec)
    concats = getattr(network, "concats", [])
    for concat in concats:
        parts: list[float] = []
        for src_name in concat.sources:
            src = named_tensors.get(src_name)
            if src is None:
                raise CompositeForwardError(
                    f"Concat {concat.name!r}: source {src_name!r} not in named_tensors"
                )
            parts.extend(src)
        named_tensors[concat.name] = parts

    # 4. Determine initial current_vec
    emb_names = {e.name for e in getattr(network, "embeddings", [])}
    if concats:
        current_vec: list[float] = list(named_tensors[concats[-1].name])
    else:
        # No concats: concatenate all input fields in order
        current_vec = []
        for name in input_data:
            if name not in emb_names:
                current_vec.extend(named_tensors[name])

    # 5. Interleave top_layers (by index) and blocks (by position) in textual order
    body_items = get_interleaved_body(network)

    for _, kind, spec in body_items:
        if kind == "layer":
            layer = spec
            prefix = network.name
            layer_key = f"{prefix}.L{layer.index}"
            current_vec, trace_entry = _forward_composite_layer(
                layer, current_vec, parameter_set, prefix, layer_key,
                training, rng, dropout_masks, layernorm_cache,
            )
            layer_traces[layer_key] = trace_entry
        else:
            block = spec
            block_prefix = f"{network.name}.{block.name}"
            residual_from = getattr(block, "residual_from", "")

            block_inputs[block.name] = list(current_vec)

            # Determine residual input vector
            if residual_from == "PREVIOUS":
                residual_vec: list[float] | None = list(current_vec)
            elif residual_from:
                rv = named_tensors.get(residual_from)
                if rv is None:
                    raise CompositeForwardError(
                        f"Block {block.name!r}: RESIDUAL FROM {residual_from!r} "
                        f"not in named_tensors"
                    )
                residual_vec = list(rv)
            else:
                residual_vec = None

            # Process block layers
            for layer in block.layers:
                layer_key = f"{block_prefix}.L{layer.index}"
                current_vec, trace_entry = _forward_composite_layer(
                    layer, current_vec, parameter_set, block_prefix, layer_key,
                    training, rng, dropout_masks, layernorm_cache,
                )
                layer_traces[layer_key] = trace_entry

            # Apply residual connection
            if residual_vec is not None:
                if len(residual_vec) != len(current_vec):
                    raise CompositeForwardError(
                        f"Block {block.name!r}: RESIDUAL shape mismatch — "
                        f"residual={len(residual_vec)}, block_output={len(current_vec)}"
                    )
                current_vec = [a + b for a, b in zip(residual_vec, current_vec)]
                residual_vectors[block.name] = residual_vec

    named_tensors["__output__"] = current_vec

    return CompositeForwardTrace(
        output=current_vec,
        named_tensors=named_tensors,
        layer_traces=layer_traces,
        dropout_masks=dropout_masks,
        layernorm_cache=layernorm_cache,
        residual_vectors=residual_vectors,
        block_inputs=block_inputs,
    )


# ---------------------------------------------------------------------------
# Per-layer forward dispatch
# ---------------------------------------------------------------------------

def _forward_composite_layer(
    layer: Any,
    x: list[float],
    parameter_set: Any,
    prefix: str,
    layer_key: str,
    training: bool,
    rng: random.Random,
    dropout_masks: dict[str, list[float]],
    layernorm_cache: dict[str, dict[str, Any]],
) -> tuple[list[float], dict[str, Any]]:
    """Forward pass for a single CompositeLayerSpec. Returns (output, trace_entry)."""
    lt = layer.layer_type

    if lt == "Dense":
        w_key = f"{prefix}.L{layer.index}.W"
        b_key = f"{prefix}.L{layer.index}.b"
        if w_key not in parameter_set.parameters:
            raise CompositeForwardError(f"Missing parameter {w_key!r}")
        if b_key not in parameter_set.parameters:
            raise CompositeForwardError(f"Missing parameter {b_key!r}")
        W = parameter_set.parameters[w_key]["values"]   # list[list[float]]
        b = parameter_set.parameters[b_key]["values"]   # list[float]
        if len(W[0]) != len(x):
            raise CompositeForwardError(
                f"{layer_key}: Dense weight cols={len(W[0])} but input dim={len(x)}"
            )
        pre_act = [
            sum(W[j][k] * x[k] for k in range(len(x))) + b[j]
            for j in range(len(W))
        ]
        act_name = getattr(layer, "activation", "linear")
        out = _apply_composite_activation(pre_act, act_name)
        return out, {"layer_type": "Dense", "input": list(x), "pre_act": pre_act, "output": out}

    elif lt == "LayerNorm":
        gamma_key = f"{prefix}.L{layer.index}.gamma"
        beta_key = f"{prefix}.L{layer.index}.beta"
        if gamma_key not in parameter_set.parameters:
            raise CompositeForwardError(f"Missing parameter {gamma_key!r}")
        if beta_key not in parameter_set.parameters:
            raise CompositeForwardError(f"Missing parameter {beta_key!r}")
        gamma = parameter_set.parameters[gamma_key]["values"]
        beta = parameter_set.parameters[beta_key]["values"]
        out, cache = _layer_norm(x, gamma, beta)
        layernorm_cache[layer_key] = cache
        return out, {"layer_type": "LayerNorm", "input": list(x), "output": out}

    elif lt == "Dropout":
        rate = getattr(layer, "rate", 0.0)
        if not training or rate <= 0.0:
            return list(x), {"layer_type": "Dropout", "input": list(x), "output": list(x), "mask": None}
        mask = [1.0 if rng.random() >= rate else 0.0 for _ in x]
        scale = 1.0 / (1.0 - rate)
        out = [mask[i] * x[i] * scale for i in range(len(x))]
        dropout_masks[layer_key] = mask
        return out, {"layer_type": "Dropout", "input": list(x), "output": out, "mask": mask}

    elif lt == "Activation":
        act_name = getattr(layer, "activation_kind", "relu")
        out = _apply_composite_activation(x, act_name)
        return out, {"layer_type": "Activation", "input": list(x), "output": out}

    elif lt == "Pool":
        # For P19 MVP, Pool on a flat 1D vector is identity (pass-through).
        # Sequence-aware pooling is handled in C8 (Evaluation).
        out = list(x)
        return out, {"layer_type": "Pool", "input": list(x), "output": out}

    elif lt == "Reshape":
        # Reshape is a shape annotation; the underlying flat vector is unchanged.
        out = list(x)
        return out, {"layer_type": "Reshape", "input": list(x), "output": out}

    else:
        raise CompositeForwardError(f"Unknown layer type {lt!r} at {layer_key}")


# ---------------------------------------------------------------------------
# Numeric primitives
# ---------------------------------------------------------------------------

def _layer_norm(
    x: list[float],
    gamma: list[float],
    beta: list[float],
) -> tuple[list[float], dict[str, Any]]:
    """LayerNorm: y = gamma * (x - mu) / sqrt(var + eps) + beta."""
    n = len(x)
    mu = sum(x) / n
    var = sum((xi - mu) ** 2 for xi in x) / n
    std_inv = 1.0 / math.sqrt(var + EPS_LAYERNORM)
    x_hat = [(xi - mu) * std_inv for xi in x]
    out = [gamma[i] * x_hat[i] + beta[i] for i in range(n)]
    cache = {"mu": mu, "var": var, "x_hat": x_hat, "std_inv": std_inv}
    return out, cache


def _gelu(z: float) -> float:
    return z * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _apply_composite_activation(z: list[float], name: str) -> list[float]:
    if name == "gelu":
        return [_gelu(v) for v in z]
    return _dense_apply_activation(z, name)
