# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 C6 — Backprop stdlib para redes densas: mse, binary_cross_entropy, cross_entropy + SGD."""
from __future__ import annotations

import math
from typing import Any

from matrixai.forward.dense_forward import DenseForwardTrace, dense_forward_trace
from matrixai.parameters.store import ParameterSet

_SUPPORTED_LOSSES = frozenset({"mse", "binary_cross_entropy", "cross_entropy"})


class DenseBackpropError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(predictions: list[float], targets: list[float]) -> float:
    n = len(predictions)
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / n


def binary_cross_entropy_loss(predictions: list[float], targets: list[float]) -> float:
    eps = 1e-7
    p = max(eps, min(1.0 - eps, predictions[0]))
    t = targets[0]
    return -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))


def cross_entropy_loss(predictions: list[float], targets: list[float]) -> float:
    eps = 1e-7
    return -sum(t * math.log(max(eps, p)) for p, t in zip(predictions, targets))


def compute_loss(loss_fn: str, predictions: list[float], targets: list[float]) -> float:
    if loss_fn == "mse":
        return mse_loss(predictions, targets)
    if loss_fn == "binary_cross_entropy":
        return binary_cross_entropy_loss(predictions, targets)
    if loss_fn == "cross_entropy":
        return cross_entropy_loss(predictions, targets)
    raise DenseBackpropError(f"Unknown loss function {loss_fn!r}. Expected one of: {sorted(_SUPPORTED_LOSSES)}")


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def dense_compute_gradients(
    network: Any,
    parameter_set: ParameterSet,
    input_vector: list[float],
    target: list[float],
    loss_fn: str,
) -> dict[str, Any]:
    """Compute parameter gradients for one sample via backpropagation.

    Returns a dict mapping parameter path → gradient (same shape as the parameter values).
    """
    if loss_fn not in _SUPPORTED_LOSSES:
        raise DenseBackpropError(f"Unknown loss function {loss_fn!r}. Expected one of: {sorted(_SUPPORTED_LOSSES)}")

    trace = dense_forward_trace(network, parameter_set, input_vector)
    layers = network.layers
    n = len(layers)

    # dL/dz for output layer — fused where possible
    dz = _output_layer_dz(loss_fn, trace.output, target, layers[-1].activation, trace.pre_activations[-1])

    gradients: dict[str, Any] = {}

    for i in range(n - 1, -1, -1):
        layer = layers[i]
        w_key = f"{network.name}.W{layer.index}"
        b_key = f"{network.name}.b{layer.index}"
        W = parameter_set.parameters[w_key]["values"]
        a_prev = trace.activations[i]  # input to layer i (activations[0]=input vector)

        # dL/dW[j][k] = dz[j] * a_prev[k]
        dW = [[dz[j] * a_prev[k] for k in range(len(a_prev))] for j in range(len(dz))]
        gradients[w_key] = dW
        gradients[b_key] = list(dz)

        if i > 0:
            # dL/da_{i-1}[k] = sum_j W[j][k] * dz[j]
            da_prev = [
                sum(W[j][k] * dz[j] for j in range(len(dz)))
                for k in range(len(a_prev))
            ]
            # dL/dz_{i-1} = da_prev ⊙ f'(z_{i-1})
            # post-activation of layer i-1 = trace.activations[i]
            dz = _activation_gradient(
                da_prev,
                trace.activations[i],          # post-activation of layer i-1
                trace.pre_activations[i - 1],  # pre-activation of layer i-1
                layers[i - 1].activation,
            )

    return gradients


# ---------------------------------------------------------------------------
# SGD training step
# ---------------------------------------------------------------------------

def dense_train_step(
    network: Any,
    parameter_set: ParameterSet,
    input_vector: list[float],
    target: list[float],
    loss_fn: str,
    learning_rate: float = 0.01,
) -> tuple[ParameterSet, float]:
    """Run one SGD step. Returns (updated ParameterSet, loss value)."""
    trace = dense_forward_trace(network, parameter_set, input_vector)
    loss = compute_loss(loss_fn, trace.output, target)
    gradients = dense_compute_gradients(network, parameter_set, input_vector, target, loss_fn)
    updated = _sgd_update(parameter_set, gradients, learning_rate)
    return updated, loss


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _output_layer_dz(
    loss_fn: str,
    output: list[float],
    target: list[float],
    activation: str,
    pre_act: list[float],
) -> list[float]:
    """dL/dz for the output layer, using fused gradient when applicable."""
    # Fused: softmax + cross_entropy → dL/dz_i = p_i - y_i
    if loss_fn == "cross_entropy" and activation == "softmax":
        return [p - t for p, t in zip(output, target)]
    # Fused: sigmoid + binary_cross_entropy → dL/dz = ŷ - y
    if loss_fn == "binary_cross_entropy" and activation == "sigmoid":
        return [output[0] - target[0]]
    # General: dL/dz = dL/da ⊙ f'(z)
    da = _loss_da(loss_fn, output, target)
    return _activation_gradient(da, output, pre_act, activation)


def _loss_da(loss_fn: str, output: list[float], target: list[float]) -> list[float]:
    """dL/da (gradient of loss w.r.t. output activations)."""
    n = len(output)
    if loss_fn == "mse":
        return [2.0 * (p - t) / n for p, t in zip(output, target)]
    eps = 1e-7
    if loss_fn == "binary_cross_entropy":
        p = max(eps, min(1.0 - eps, output[0]))
        t = target[0]
        return [-(t / p - (1.0 - t) / (1.0 - p))]
    if loss_fn == "cross_entropy":
        return [-t / max(eps, p) for p, t in zip(output, target)]
    raise DenseBackpropError(f"Unknown loss function: {loss_fn!r}")


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
    if activation == "linear":
        return list(da)
    if activation == "softmax":
        # dL/dz_i = a_i * (da_i - sum_j(da_j * a_j))
        dot = sum(d * a_j for d, a_j in zip(da, a))
        return [a_i * (da_i - dot) for a_i, da_i in zip(a, da)]
    raise DenseBackpropError(f"Unknown activation: {activation!r}")


def _sgd_update(
    parameter_set: ParameterSet,
    gradients: dict[str, Any],
    learning_rate: float,
) -> ParameterSet:
    new_params: dict[str, Any] = {}
    for key, param in parameter_set.parameters.items():
        grad = gradients.get(key)
        if grad is None:
            new_params[key] = param
        else:
            new_params[key] = {**param, "values": _subtract_scaled(param["values"], grad, learning_rate)}
    return ParameterSet(
        parameter_set_id=parameter_set.parameter_set_id,
        model_hash=parameter_set.model_hash,
        parameter_schema_hash=parameter_set.parameter_schema_hash,
        source="trained",
        parameters=new_params,
        metrics=parameter_set.metrics,
    )


def _subtract_scaled(values: Any, grad: Any, lr: float) -> Any:
    if isinstance(values[0], list):
        return [[v - lr * g for v, g in zip(row_v, row_g)] for row_v, row_g in zip(values, grad)]
    return [v - lr * g for v, g in zip(values, grad)]
