# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""TRANSFORMER C2 — forward stdlib de REFERENCIA para el BLOCK TRANSFORMER.

Este forward existe como suelo determinista para shapes mini de test (paridad
C3, invariantes anti-transformer-falso). El camino de producto (entrenar/evaluar
con datos reales) es torch (C4) — regla de PESOS_GRANDES: nada O(params) en
Python puro fuera de tests.

Convenciones (las mismas del composite stdlib, `composite_forward.py`):
  - Pesos [out, in]: y = W @ x (+ b). LayerNorm eps 1e-5. GELU exacta (erf).
  - Softmax con resta de máximo. Posicional sinusoidal con la fórmula P10:
    angle = pos / 10000^((2*(i//2))/dim); sin en índices pares, cos en impares.
  - Máscara de padding: lista de bools (True = posición real). Las claves de
    posiciones de padding reciben score -inf (peso exactamente 0.0 tras el
    softmax); el POOL mean promedia SOLO posiciones reales. Así el contenido
    del padding no influye en la salida (invariante 1c) de forma exacta.
  - La máscara es EXPLÍCITA o derivada de pad_id: derivarla siempre del
    contenido haría imposible el test del invariante (cambiar el contenido
    des-enmascararía la posición).
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from matrixai.forward.composite_forward import (
    CompositeForwardError,
    EPS_LAYERNORM,
    _forward_composite_layer,
    _gelu,
)


class TransformerForwardError(CompositeForwardError):
    pass


@dataclass
class TransformerForwardTrace:
    """Salida + etapas intermedias del forward de referencia (para tests C2/C3)."""
    output: list[float]                       # salida de la cabeza densa
    embedded: list[list[float]]               # [L][dim] tras embedding + POS
    layer_outputs: list[list[list[float]]]    # salida [L][dim] de cada encoder layer
    block_output: list[list[float]]           # [L][dim] pre-pooling
    pooled: list[float]                       # [dim] tras POOL mean|cls
    attention_weights: list[list[list[list[float]]]]  # [layer][head][L][L]
    mask: list[bool]                          # máscara efectiva usada
    touched_params: set[str] = field(default_factory=set)  # liveness


class _RecordingParams(Mapping):
    """Envoltorio de parameter_set.parameters que registra cada path accedido
    (liveness: el forward debe tocar EXACTAMENTE los paths del manifest)."""

    def __init__(self, inner: dict, touched: set[str]):
        self._inner = inner
        self._touched = touched

    def __getitem__(self, key: str):
        self._touched.add(key)
        return self._inner[key]

    def __contains__(self, key: object) -> bool:
        return key in self._inner

    def __iter__(self):
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


class _RecordingParameterSet:
    """ParameterSet-like con .parameters de registro, para reusar
    _forward_composite_layer sin perder el tracking de liveness."""

    def __init__(self, parameter_set: Any, touched: set[str]):
        self.parameters = _RecordingParams(parameter_set.parameters, touched)


def sinusoidal_positional_table(length: int, dim: int) -> list[list[float]]:
    """Tabla [L][dim] con la fórmula P10 (determinista, no entrenable)."""
    table: list[list[float]] = []
    for pos in range(length):
        row: list[float] = []
        for i in range(dim):
            angle = float(pos) / (10000 ** ((2 * (i // 2)) / max(dim, 1)))
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        table.append(row)
    return table


def _matvec(w: list[list[float]], x: list[float]) -> list[float]:
    """y = W @ x con W [out][in] — misma convención que Dense composite."""
    return [sum(w[j][k] * x[k] for k in range(len(x))) for j in range(len(w))]


def _softmax(row: list[float]) -> list[float]:
    m = max(row)
    if m == float("-inf"):
        raise TransformerForwardError(
            "attention row fully masked — the mask must keep at least one real position"
        )
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def _layer_norm_vec(x: list[float], gain: list[float], bias: list[float]) -> list[float]:
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + EPS_LAYERNORM)
    return [gain[i] * ((x[i] - mu) * inv) + bias[i] for i in range(n)]


def _apply_ffn_activation(x: list[float], name: str) -> list[float]:
    if name == "gelu":
        return [_gelu(v) for v in x]
    if name == "relu":
        return [max(0.0, v) for v in x]
    raise TransformerForwardError(f"unsupported transformer FFN activation: {name!r}")


def _dropout_rows(
    rows: list[list[float]], rate: float, training: bool, rng: random.Random
) -> list[list[float]]:
    """Dropout invertido por posición (semántica P19: activo en train, identidad en eval)."""
    if not training or rate <= 0.0:
        return rows
    scale = 1.0 / (1.0 - rate)
    return [
        [(v * scale if rng.random() >= rate else 0.0) for v in row]
        for row in rows
    ]


def transformer_network_forward(
    network: Any,
    type_result: Any,
    parameter_set: Any,
    token_ids: list[int],
    mask: list[bool] | None = None,
    pad_id: int | None = None,
    training: bool = False,
    seed: int = 42,
) -> TransformerForwardTrace:
    """Forward de referencia de una red composite con BLOCK TRANSFORMER.

    token_ids: [L] ids enteros. mask: [L] bools (True = real); si None y pad_id
    dado, se deriva como token != pad_id; si ambos None, todo real.
    """
    # Fail closed on a dirty typecheck: e.g. HEADS-no-divisor still resolves
    # dim/ff, and running anyway would silently attend dim//heads truncated.
    if not getattr(type_result, "ok", False):
        raise TransformerForwardError(
            "transformer_network_forward requires a CLEAN type_result "
            f"(errors: {getattr(type_result, 'errors', ['unknown'])[:2]})"
        )
    tblocks = list(getattr(type_result, "resolved_transformer_blocks", []))
    if len(tblocks) != 1 or tblocks[0].resolved_dim <= 0:
        raise TransformerForwardError(
            "transformer_network_forward requires a type_result with exactly one "
            "RESOLVED transformer block (run check_composite_network_types first)"
        )
    tb = tblocks[0]
    seq_len, dim = tb.input_shape
    if len(token_ids) != seq_len:
        raise TransformerForwardError(
            f"expected {seq_len} token ids (SEQUENCE length), got {len(token_ids)}"
        )
    if mask is None:
        mask = [True] * seq_len if pad_id is None else [t != pad_id for t in token_ids]
    if len(mask) != seq_len:
        raise TransformerForwardError(f"mask length {len(mask)} != sequence length {seq_len}")
    if not any(mask):
        raise TransformerForwardError("mask must keep at least one real position")

    touched: set[str] = set()
    params = _RecordingParams(parameter_set.parameters, touched)
    rng = random.Random(seed)

    # 1. Embedding por posición: [L] ids → [L][dim]
    seq_embeddings = [
        e for e in getattr(type_result, "resolved_embeddings", [])
        if e.source == network.input
    ]
    if len(seq_embeddings) != 1:
        raise TransformerForwardError(
            f"expected exactly one EMBEDDING FROM the input SEQUENCE, "
            f"found {len(seq_embeddings)}"
        )
    emb = seq_embeddings[0]
    table_key = f"{network.name}.{emb.name}.table"
    if table_key not in params:
        raise TransformerForwardError(f"Missing embedding table parameter: {table_key!r}")
    table = params[table_key]["values"]
    x: list[list[float]] = []
    for t in token_ids:
        idx = int(t)
        if idx < 0 or idx >= len(table):
            raise TransformerForwardError(
                f"token id {idx} out of range [0, {len(table)}) for EMBEDDING {emb.name!r}"
            )
        x.append(list(table[idx]))

    # 2. Positional encoding
    block_prefix = f"{network.name}.{tb.name}"
    if tb.pos == "learned":
        pos_key = f"{block_prefix}.pos.table"
        if pos_key not in params:
            raise TransformerForwardError(f"Missing learned positional table: {pos_key!r}")
        pos_table = params[pos_key]["values"]
    else:
        pos_table = sinusoidal_positional_table(seq_len, dim)
    x = [[x[t][i] + pos_table[t][i] for i in range(dim)] for t in range(seq_len)]
    embedded = [list(row) for row in x]

    heads = tb.heads
    dh = dim // heads
    inv_sqrt_dh = 1.0 / math.sqrt(dh)
    layer_outputs: list[list[list[float]]] = []
    attention_weights: list[list[list[list[float]]]] = []

    def _p(name: str) -> Any:
        key = f"{block_prefix}.{name}"
        if key not in params:
            raise TransformerForwardError(f"Missing transformer parameter: {key!r}")
        return params[key]["values"]

    def _run_encoder(rows: list[list[float]]) -> list[list[float]]:
        """Encoder layers: {MHA multi-cabeza con máscara → dropout → residual+LN,
        FFN(act) → dropout → residual+LN} × tb.layers."""
        cur = rows
        for li in range(tb.layers):
            lp = f"layer_{li}"
            wq, wk, wv, wo = (_p(f"{lp}.attention.{w}") for w in ("Wq", "Wk", "Wv", "Wo"))
            q = [_matvec(wq, row) for row in cur]
            k = [_matvec(wk, row) for row in cur]
            v = [_matvec(wv, row) for row in cur]

            layer_attn: list[list[list[float]]] = []
            ctx: list[list[float]] = [[0.0] * dim for _ in range(seq_len)]
            for h in range(heads):
                lo = h * dh
                hi = lo + dh
                head_weights: list[list[float]] = []
                for t in range(seq_len):
                    scores = [
                        (
                            sum(q[t][i] * k[s][i] for i in range(lo, hi)) * inv_sqrt_dh
                            if mask[s] else float("-inf")
                        )
                        for s in range(seq_len)
                    ]
                    weights = _softmax(scores)
                    head_weights.append(weights)
                    for i in range(lo, hi):
                        ctx[t][i] = sum(weights[s] * v[s][i] for s in range(seq_len))
                layer_attn.append(head_weights)
            attention_weights.append(layer_attn)

            attn_out = [_matvec(wo, row) for row in ctx]
            attn_out = _dropout_rows(attn_out, tb.dropout, training, rng)
            n1_gain, n1_bias = _p(f"{lp}.norm1.gain"), _p(f"{lp}.norm1.bias")
            cur = [
                _layer_norm_vec(
                    [cur[t][i] + attn_out[t][i] for i in range(dim)], n1_gain, n1_bias
                )
                for t in range(seq_len)
            ]

            w1, b1 = _p(f"{lp}.ffn.W1"), _p(f"{lp}.ffn.b1")
            w2, b2 = _p(f"{lp}.ffn.W2"), _p(f"{lp}.ffn.b2")
            ffn_out: list[list[float]] = []
            for t in range(seq_len):
                h1 = [a + b for a, b in zip(_matvec(w1, cur[t]), b1)]
                h1 = _apply_ffn_activation(h1, tb.activation)
                h2 = [a + b for a, b in zip(_matvec(w2, h1), b2)]
                ffn_out.append(h2)
            ffn_out = _dropout_rows(ffn_out, tb.dropout, training, rng)
            n2_gain, n2_bias = _p(f"{lp}.norm2.gain"), _p(f"{lp}.norm2.bias")
            cur = [
                _layer_norm_vec(
                    [cur[t][i] + ffn_out[t][i] for i in range(dim)], n2_gain, n2_bias
                )
                for t in range(seq_len)
            ]
            layer_outputs.append([list(row) for row in cur])
        return cur

    # 3. Cuerpo en el MISMO orden intercalado que el typecheck: capas del stream
    #    (por posición) antes/después del encoder, POOL (mean enmascarado | cls)
    #    y cabeza densa (reusa el dispatcher del composite stdlib).
    body_items: list[tuple[int, str, Any]] = [
        (layer.index * 2, "layer", layer)
        for layer in getattr(type_result, "resolved_layers", [])
    ]
    body_items.append((tb.position * 2 + 1, "tblock", tb))
    body_items.sort(key=lambda item: item[0])

    recording_ps = _RecordingParameterSet(parameter_set, touched)
    current_rows: list[list[float]] | None = x
    current_vec: list[float] | None = None
    block_output: list[list[float]] | None = None
    pooled: list[float] | None = None
    dropout_masks: dict[str, list[float]] = {}
    layernorm_cache: dict[str, dict[str, Any]] = {}

    for _, kind, item in body_items:
        if kind == "tblock":
            current_rows = _run_encoder(current_rows)
            block_output = [list(row) for row in current_rows]
            continue
        layer = item
        layer_key = f"{network.name}.L{layer.index}"
        if current_rows is not None:
            # Stream [L][dim] (antes del POOL): solo capas que preservan shape
            if layer.layer_type == "Pool":
                if layer.pool_kind == "cls":
                    pooled = list(current_rows[0])
                elif layer.pool_kind == "mean":  # enmascarado: solo posiciones reales
                    real = [t for t in range(seq_len) if mask[t]]
                    pooled = [
                        sum(current_rows[t][i] for t in real) / len(real)
                        for i in range(dim)
                    ]
                else:
                    raise TransformerForwardError(
                        f"POOL {layer.pool_kind} over the sequence stream is not "
                        f"supported (typecheck should have rejected this)"
                    )
                current_vec = pooled
                current_rows = None
            elif layer.layer_type == "LayerNorm":
                g_key = f"{network.name}.L{layer.index}.gamma"
                b_key = f"{network.name}.L{layer.index}.beta"
                gain, bias = params[g_key]["values"], params[b_key]["values"]
                current_rows = [_layer_norm_vec(row, gain, bias) for row in current_rows]
            elif layer.layer_type == "Dropout":
                current_rows = _dropout_rows(current_rows, layer.rate, training, rng)
            elif layer.layer_type == "Activation":
                from matrixai.forward.composite_forward import _apply_composite_activation
                current_rows = [
                    _apply_composite_activation(row, layer.activation_kind)
                    for row in current_rows
                ]
            else:
                raise TransformerForwardError(
                    f"{layer_key}: LAYER {layer.layer_type} cannot run on the "
                    f"sequence stream (typecheck should have rejected this)"
                )
        else:
            # Ya en vector [dim]: cabeza clásica (Dense/Dropout/Activation/…)
            current_vec, _trace = _forward_composite_layer(
                layer, current_vec, recording_ps, network.name, layer_key,
                training, rng, dropout_masks, layernorm_cache,
            )

    if block_output is None or pooled is None or current_vec is None:
        raise TransformerForwardError(
            "the network body never pooled the stream (typecheck should require POOL)"
        )

    return TransformerForwardTrace(
        output=current_vec,
        embedded=embedded,
        layer_outputs=layer_outputs,
        block_output=block_output,
        pooled=pooled,
        attention_weights=attention_weights,
        mask=list(mask),
        touched_params=touched,
    )
