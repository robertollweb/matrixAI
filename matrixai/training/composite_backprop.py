# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P19 C6 — Backprop stdlib para redes compuestas: gradientes + SGD."""
from __future__ import annotations

import math
from typing import Any

from matrixai.forward.composite_forward import composite_forward_trace
from matrixai.ir.schema import get_interleaved_body
from matrixai.parameters.store import ParameterSet
from matrixai.training.dense_backprop import (
    DenseBackpropError,
    compute_loss,
    mse_loss,
    binary_cross_entropy_loss,
    cross_entropy_loss,
    _loss_da,
    _sgd_update,
)


class CompositeBackpropError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Main gradient computation
# ---------------------------------------------------------------------------

def composite_compute_gradients(
    network: Any,
    parameter_set: ParameterSet,
    input_data: dict[str, Any],
    target: list[float],
    loss_fn: str,
    training: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute parameter gradients for one sample via backpropagation.

    Returns dict mapping parameter path → gradient (same shape as the parameter values).
    """
    trace = composite_forward_trace(
        network, parameter_set, input_data, training=training, seed=seed
    )

    body_items_fwd = get_interleaved_body(network)
    if not body_items_fwd:
        return {}

    gradients: dict[str, Any] = {}
    d_named: dict[str, list[float]] = {}  # gradient accumulators for named tensors

    # Find the last Dense layer (output layer)
    final_prefix, final_spec, final_key = _find_final_dense(body_items_fwd, network)
    final_is_top_layer = (final_prefix == network.name)

    # Compute initial dz for the final Dense (fused where possible)
    final_pre_act = trace.layer_traces[final_key].get("pre_act", [])
    dz_final = _output_dz(
        loss_fn, trace.output, target, final_spec.activation, final_pre_act
    )

    # Compute dW/db for final Dense and get d_current = da_prev
    a_prev_final = trace.layer_traces[final_key].get("input", [])
    W_key = f"{final_prefix}.L{final_spec.index}.W"
    b_key = f"{final_prefix}.L{final_spec.index}.b"
    W_final = parameter_set.parameters[W_key]["values"]
    gradients[W_key] = [
        [dz_final[j] * a_prev_final[k] for k in range(len(a_prev_final))]
        for j in range(len(dz_final))
    ]
    gradients[b_key] = list(dz_final)
    d_current: list[float] = [
        sum(W_final[j][k] * dz_final[j] for j in range(len(W_final)))
        for k in range(len(a_prev_final))
    ]

    # Process remaining body items in REVERSE (skipping the final Dense already handled)
    body_items_rev = list(reversed(body_items_fwd))
    skip_first = True

    for _, kind, spec in body_items_rev:
        if skip_first:
            skip_first = False
            continue  # final Dense already processed above

        if kind == "layer":
            layer = spec
            prefix = network.name
            layer_key = f"{prefix}.L{layer.index}"
            d_current = _backprop_composite_layer(
                layer, layer_key, d_current, prefix, parameter_set, trace, gradients
            )
        else:
            block = spec
            block_prefix = f"{network.name}.{block.name}"
            residual_from = block.residual_from

            if residual_from:
                d_skip = list(d_current)

            d_through = list(d_current)
            for layer in reversed(block.layers):
                layer_key = f"{block_prefix}.L{layer.index}"
                # Skip if this layer IS the final Dense (already handled)
                if layer_key == final_key:
                    continue
                d_through = _backprop_composite_layer(
                    layer, layer_key, d_through, block_prefix, parameter_set, trace, gradients
                )

            if residual_from:
                if residual_from == "PREVIOUS":
                    d_current = [a + b for a, b in zip(d_skip, d_through)]
                else:
                    d_named[residual_from] = _add_grad(d_named.get(residual_from), d_skip)
                    d_current = d_through
            else:
                d_current = d_through

    # After body backward: d_current is gradient w.r.t. the body's entry tensor
    concats = getattr(network, "concats", [])
    if concats:
        last_concat_name = concats[-1].name
        d_named[last_concat_name] = _add_grad(d_named.get(last_concat_name), d_current)

    # Backprop through concats (in reverse)
    for concat in reversed(concats):
        d_concat = d_named.get(concat.name)
        if d_concat:
            offset = 0
            for src_name in concat.sources:
                dim = len(trace.named_tensors[src_name])
                d_src = d_concat[offset:offset + dim]
                d_named[src_name] = _add_grad(d_named.get(src_name), d_src)
                offset += dim

    # Backprop through embeddings: accumulate gradient to the used row
    for emb in getattr(network, "embeddings", []):
        d_emb = d_named.get(emb.name)
        if d_emb:
            table_key = f"{network.name}.{emb.name}.table"
            table = parameter_set.parameters[table_key]["values"]
            d_table = [[0.0] * len(row) for row in table]
            idx = int(round(trace.named_tensors[emb.source][0]))
            if 0 <= idx < len(d_table):
                d_table[idx] = list(d_emb)
            gradients[table_key] = d_table

    return gradients


# ---------------------------------------------------------------------------
# SGD training step
# ---------------------------------------------------------------------------

def composite_train_step(
    network: Any,
    parameter_set: ParameterSet,
    input_data: dict[str, Any],
    target: list[float],
    loss_fn: str,
    learning_rate: float = 0.01,
    training: bool = True,
    seed: int = 42,
) -> tuple[ParameterSet, float]:
    """Run one SGD step. Returns (updated ParameterSet, loss value)."""
    trace = composite_forward_trace(
        network, parameter_set, input_data, training=training, seed=seed
    )
    loss = compute_loss(loss_fn, trace.output, target)
    gradients = composite_compute_gradients(
        network, parameter_set, input_data, target, loss_fn, training=training, seed=seed
    )
    updated = _sgd_update(parameter_set, gradients, learning_rate)
    return updated, loss


# ---------------------------------------------------------------------------
# Numerical gradient check
# ---------------------------------------------------------------------------

def numerical_gradient(
    network: Any,
    parameter_set: ParameterSet,
    input_data: dict[str, Any],
    target: list[float],
    loss_fn: str,
    param_path: str,
    param_idx: tuple[int, ...],
    eps: float = 1e-4,
) -> float:
    """Compute numerical gradient via central finite difference for one parameter element.

    param_idx: (row,) for 1D params, (row, col) for 2D params.
    Uses training=False to avoid stochastic dropout.
    """
    def _perturb(delta: float) -> float:
        params = dict(parameter_set.parameters)
        entry = dict(params[param_path])
        vals = entry["values"]
        if len(param_idx) == 1:
            vals_copy = list(vals)
            vals_copy[param_idx[0]] += delta
        else:
            vals_copy = [list(r) for r in vals]
            vals_copy[param_idx[0]][param_idx[1]] += delta
        new_entry = {**entry, "values": vals_copy}
        params[param_path] = new_entry
        ps = ParameterSet(
            parameter_set_id=parameter_set.parameter_set_id,
            model_hash=parameter_set.model_hash,
            parameter_schema_hash=parameter_set.parameter_schema_hash,
            source=parameter_set.source,
            parameters=params,
        )
        from matrixai.forward.composite_forward import composite_forward
        out = composite_forward(network, ps, input_data, training=False)
        return compute_loss(loss_fn, out, target)

    return (_perturb(eps) - _perturb(-eps)) / (2.0 * eps)


def gradient_check(
    network: Any,
    parameter_set: ParameterSet,
    input_data: dict[str, Any],
    target: list[float],
    loss_fn: str,
    param_path: str,
    param_idx: tuple[int, ...],
    eps: float = 1e-4,
    tol: float = 1e-3,
) -> tuple[float, float, bool]:
    """Compare analytical vs numerical gradient for a single parameter element.

    Returns (analytical, numerical, passed) where passed means |analytical - numerical| < tol.
    """
    grads = composite_compute_gradients(
        network, parameter_set, input_data, target, loss_fn, training=False, seed=0
    )
    grad = grads.get(param_path)
    if grad is None:
        return 0.0, 0.0, False
    if len(param_idx) == 1:
        analytical = grad[param_idx[0]]
    else:
        analytical = grad[param_idx[0]][param_idx[1]]
    num = numerical_gradient(
        network, parameter_set, input_data, target, loss_fn, param_path, param_idx, eps
    )
    return analytical, num, abs(analytical - num) < tol


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_final_dense(
    body_items_fwd: list[tuple[int, str, Any]], network: Any
) -> tuple[str, Any, str]:
    """Return (prefix, DenseLayerSpec, layer_key) for the last Dense in the body."""
    for _, kind, spec in reversed(body_items_fwd):
        if kind == "layer" and spec.layer_type == "Dense":
            prefix = network.name
            return prefix, spec, f"{prefix}.L{spec.index}"
        elif kind == "block":
            for layer in reversed(spec.layers):
                if layer.layer_type == "Dense":
                    prefix = f"{network.name}.{spec.name}"
                    return prefix, layer, f"{prefix}.L{layer.index}"
    raise CompositeBackpropError("No Dense output layer found in composite network body")


def _backprop_composite_layer(
    layer: Any,
    layer_key: str,
    d_out: list[float],
    prefix: str,
    parameter_set: ParameterSet,
    trace: Any,
    gradients: dict[str, Any],
) -> list[float]:
    """Backprop through a single composite layer.

    d_out: gradient w.r.t. the layer's OUTPUT (da).
    Returns: gradient w.r.t. the layer's INPUT (da_prev).
    """
    lt = layer.layer_type
    ltr = trace.layer_traces.get(layer_key, {})
    x_in = ltr.get("input", [])
    a_out = ltr.get("output", [])

    if lt == "Dense":
        W_key = f"{prefix}.L{layer.index}.W"
        b_key = f"{prefix}.L{layer.index}.b"
        W = parameter_set.parameters[W_key]["values"]
        pre_act = ltr.get("pre_act", [])
        act = layer.activation
        dz = _activation_gradient(d_out, a_out, pre_act, act)
        gradients[W_key] = [
            [dz[j] * x_in[k] for k in range(len(x_in))]
            for j in range(len(dz))
        ]
        gradients[b_key] = list(dz)
        return [
            sum(W[j][k] * dz[j] for j in range(len(W)))
            for k in range(len(x_in))
        ]

    elif lt == "LayerNorm":
        gamma_key = f"{prefix}.L{layer.index}.gamma"
        beta_key = f"{prefix}.L{layer.index}.beta"
        gamma = parameter_set.parameters[gamma_key]["values"]
        cache = trace.layernorm_cache.get(layer_key, {})
        x_hat = cache.get("x_hat", [0.0] * len(d_out))
        std_inv = cache.get("std_inv", 1.0)
        N = len(d_out)
        gradients[gamma_key] = [d_out[i] * x_hat[i] for i in range(N)]
        gradients[beta_key] = list(d_out)
        dgy = [d_out[i] * gamma[i] for i in range(N)]
        sum_dgy = sum(dgy)
        sum_dgy_xhat = sum(dgy[i] * x_hat[i] for i in range(N))
        return [
            std_inv * (dgy[i] - sum_dgy / N - x_hat[i] * sum_dgy_xhat / N)
            for i in range(N)
        ]

    elif lt == "Dropout":
        mask = trace.dropout_masks.get(layer_key)
        rate = getattr(layer, "rate", 0.0)
        if mask is None or rate <= 0.0:
            return list(d_out)
        scale = 1.0 / (1.0 - rate)
        return [mask[i] * d_out[i] * scale for i in range(len(d_out))]

    elif lt == "Activation":
        act = getattr(layer, "activation_kind", "relu")
        return _activation_gradient(d_out, a_out, x_in, act)

    elif lt in ("Pool", "Reshape"):
        return list(d_out)

    else:
        raise CompositeBackpropError(f"Unknown layer type {lt!r} at {layer_key}")


def _output_dz(
    loss_fn: str,
    output: list[float],
    target: list[float],
    activation: str,
    pre_act: list[float],
) -> list[float]:
    """Compute dL/dz for the output Dense layer (fused where applicable)."""
    if loss_fn == "cross_entropy" and activation == "softmax":
        return [p - t for p, t in zip(output, target)]
    if loss_fn == "binary_cross_entropy" and activation == "sigmoid":
        return [output[0] - target[0]]
    da = _loss_da(loss_fn, output, target)
    return _activation_gradient(da, output, pre_act, activation)


def _activation_gradient(
    da: list[float],
    a: list[float],
    z: list[float],
    activation: str,
) -> list[float]:
    """dL/dz = da ⊙ f'(z). Uses post-activation a for sigmoid/tanh efficiency."""
    if activation == "relu":
        return [da_i * (1.0 if z_i > 0.0 else 0.0) for da_i, z_i in zip(da, z)]
    if activation == "sigmoid":
        return [da_i * a_i * (1.0 - a_i) for da_i, a_i in zip(da, a)]
    if activation == "tanh":
        return [da_i * (1.0 - a_i ** 2) for da_i, a_i in zip(da, a)]
    if activation in ("linear", ""):
        return list(da)
    if activation == "softmax":
        dot = sum(d * a_j for d, a_j in zip(da, a))
        return [a_i * (da_i - dot) for a_i, da_i in zip(a, da)]
    if activation == "gelu":
        return [da_i * _gelu_prime(z_i) for da_i, z_i in zip(da, z)]
    raise CompositeBackpropError(f"Unknown activation: {activation!r}")


def _gelu_prime(z: float) -> float:
    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + math.erf(z / sqrt2)) + z * math.exp(-0.5 * z * z) / sqrt2pi


def _add_grad(
    a: list[float] | None,
    b: list[float] | None,
) -> list[float]:
    if a is None:
        return list(b) if b else []
    if b is None:
        return list(a)
    return [x + y for x, y in zip(a, b)]
