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


def _rechazar_red_que_no_es_densa_plana(network: Any) -> None:
    """101-C3 (medido el 2026-09-12) — un forward que no entiende la red se
    PLANTA; nunca devuelve la entrada disfrazada de salida.

    Una red COMPUESTA (`kind="composite_network"`) trae `layers=[]` POR
    CONSTRUCCIÓN: el parser deja su cuerpo en `top_layers`/`blocks`
    (`parser/parser.py`, rama `is_composite`). Sin este rechazo el bucle de
    `dense_forward_trace` no daba NI UNA VUELTA y la función devolvía el
    VECTOR DE ENTRADA tal cual, haciéndolo pasar por la salida de la red.

    Medido sobre el dataset `pc4` de la pasada exploratoria de 101-C3
    (2026-09-07): la red tenía 2 unidades de salida y esto devolvía los 37
    números de la entrada normalizada. Quien llamaba (el motor denso de
    `matrixai-engines`, que no miraba `red.kind`) los tomaba por
    probabilidades y reventaba cien líneas después con un
    `IndexError: list index out of range` que no nombraba ni la red ni el
    forward — 15 intentos perdidos, el dataset entero fuera de la regla de
    cierre, y ningún mensaje que apuntara aquí.

    Es el ESPEJO EXACTO del guardián que `composite_forward` ya tenía para
    los bloques transformer ("this forward iterates top_layers/blocks only —
    it would silently SKIP a transformer block"): la misma clase de fallo, en
    la dirección contraria, y sin guardián hasta hoy.
    """
    if getattr(network, "transformer_blocks", None):
        raise DenseForwardError(
            f"dense_forward: NETWORK {getattr(network, 'name', '?')} contains a BLOCK "
            f"TRANSFORMER — use transformer_network_forward, not the dense forward"
        )
    compuesta = (
        getattr(network, "top_layers", None)
        or getattr(network, "blocks", None)
        or getattr(network, "embeddings", None)
        or getattr(network, "concats", None)
        or getattr(network, "kind", "dense_network") != "dense_network"
    )
    if compuesta:
        raise DenseForwardError(
            f"dense_forward: NETWORK {getattr(network, 'name', '?')} is a "
            f"{getattr(network, 'kind', '?')} (blocks/embeddings/top_layers live outside "
            f"`layers`) — use composite_forward; the dense forward would ignore its body"
        )
    if not getattr(network, "layers", None):
        raise DenseForwardError(
            f"dense_forward: NETWORK {getattr(network, 'name', '?')} has no layers — "
            f"returning the input unchanged would pass it off as a prediction"
        )


def dense_forward(network: Any, parameter_set: Any, input_vector: list[float]) -> list[float]:
    """Run a dense network forward pass. Returns the final layer output."""
    return dense_forward_trace(network, parameter_set, input_vector).output


def dense_forward_trace(network: Any, parameter_set: Any, input_vector: list[float]) -> DenseForwardTrace:
    """Run a dense network forward pass and return full intermediate trace."""
    _rechazar_red_que_no_es_densa_plana(network)
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
