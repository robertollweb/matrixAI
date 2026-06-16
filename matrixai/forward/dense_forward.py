# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 C5 — Forward pass stdlib para redes densas."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


class DenseForwardError(ValueError):
    pass


@dataclass(frozen=True)
class DenseForwardTrace:
    """Intermediate values from a forward pass — used by backprop (C6)."""
    output: list[float]
    # activations[0] = input vector; activations[i] = output of layer i
    activations: list[list[float]] = field(default_factory=list)
    # pre_activations[i] = W·a[i-1] + b before applying activation of layer i
    pre_activations: list[list[float]] = field(default_factory=list)


def dense_forward(network: Any, parameter_set: Any, input_vector: list[float]) -> list[float]:
    """Run a dense network forward pass. Returns the final layer output."""
    return dense_forward_trace(network, parameter_set, input_vector).output


def dense_forward_trace(network: Any, parameter_set: Any, input_vector: list[float]) -> DenseForwardTrace:
    """Run a dense network forward pass and return full intermediate trace."""
    activation = list(input_vector)
    all_activations: list[list[float]] = [activation]
    all_pre_activations: list[list[float]] = []

    for layer in network.layers:
        w_key = f"{network.name}.W{layer.index}"
        b_key = f"{network.name}.b{layer.index}"

        if w_key not in parameter_set.parameters:
            raise DenseForwardError(f"Missing parameter {w_key!r} in ParameterSet")
        if b_key not in parameter_set.parameters:
            raise DenseForwardError(f"Missing parameter {b_key!r} in ParameterSet")

        W = parameter_set.parameters[w_key]["values"]  # list[list[float]]
        b = parameter_set.parameters[b_key]["values"]  # list[float]
        x = activation

        if len(W[0]) != len(x):
            raise DenseForwardError(
                f"Layer {layer.index}: weight cols={len(W[0])} but input dim={len(x)}"
            )

        pre_act = [
            sum(W[j][k] * x[k] for k in range(len(x))) + b[j]
            for j in range(len(W))
        ]
        all_pre_activations.append(pre_act)
        activation = _apply_activation(pre_act, layer.activation)
        all_activations.append(activation)

    return DenseForwardTrace(
        output=activation,
        activations=all_activations,
        pre_activations=all_pre_activations,
    )


# ---------------------------------------------------------------------------
# Activation functions (stdlib only)
# ---------------------------------------------------------------------------

def _apply_activation(z: list[float], name: str) -> list[float]:
    if name == "relu":
        return [max(0.0, v) for v in z]
    if name == "sigmoid":
        return [_sigmoid(v) for v in z]
    if name == "tanh":
        return [math.tanh(v) for v in z]
    if name == "softmax":
        return _softmax(z)
    if name == "linear":
        return list(z)
    raise DenseForwardError(f"Unknown activation {name!r}")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _softmax(z: list[float]) -> list[float]:
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]
