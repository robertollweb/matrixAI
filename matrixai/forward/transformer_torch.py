# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""TRANSFORMER C3 — módulo torch del BLOCK TRANSFORMER (techo de velocidad).

Composición por ops EXPLÍCITAS (decisión 2 del contrato: NUNCA
torch.nn.MultiheadAttention ni nn.TransformerEncoder en el forward de
producto — solo como referencia en los tests de paridad C3). La matemática
replica 1:1 el forward stdlib de referencia de C2
(`transformer_forward.transformer_network_forward`):

  - proyecciones q/k/v/o SIN bias (paths del contrato), scores [L,L]/√dh por
    cabeza, claves de padding a -inf → peso exactamente 0.0
  - residual + LayerNorm(eps 1e-5), FFN (gelu erf / relu), dropout invertido
    train-only (dos instancias independientes por capa, como los dos draws
    secuenciales del stdlib)
  - posicional sinusoidal con LA MISMA tabla P10 del stdlib (única fuente:
    `sinusoidal_positional_table`) o tabla learned [L, dim]
  - POOL mean ENMASCARADO (solo posiciones reales) | cls (exige posición 0
    real en TODA la fila del batch)
  - cabeza densa con las mismas activaciones del composite torch

Paridad documentada (tests C3): stdlib(float64) == torch(float64) con
atol 1e-9; en float32 (dtype de producto) la tolerancia documentada es
rtol/atol 1e-5. La carga de pesos sigue M15(f): `values=None`
(with_values=False) deja el init nativo de torch.
"""
from __future__ import annotations

from typing import Any

from matrixai.forward.composite_torch import (
    CompositeTorchError,
    _apply_torch_activation,
    torch_available,
)
from matrixai.forward.transformer_forward import sinusoidal_positional_table


class TransformerTorchError(CompositeTorchError):
    pass


def transformer_network_to_torch_module(
    network: Any,
    type_result: Any,
    parameter_set: Any,
    dtype: Any = None,
    output_name: str = "",
    expected_model_hash: str | None = None,
) -> Any:
    """Build the torch nn.Module for a composite network with a BLOCK TRANSFORMER.

    Requires a CLEAN type_result (same contract as the C2 reference forward).
    Weights load from parameter_set values; values=None keeps torch native init
    (with_values=False path, M15(f)). dtype: None → float32 (producto); los
    tests de paridad estricta pasan torch.float64 para cargar los valores
    Python SIN el redondeo float32 (module.double() a posteriori no lo
    recupera: convierte valores ya truncados).

    Auditoría C3 [ALTA]: el ParameterSet se valida ÍNTEGRO contra el manifest
    ANTES de construir — parameter_schema_hash (un set de HEADS=1 en una red
    HEADS=2 tiene shapes idénticas y solo el hash lo distingue), paths en ambos
    sentidos, shapes de metadata y shape REAL de los valores. La plantilla
    with_values=False se acepta solo cuando TODOS los values son None; un set
    parcialmente materializado falla para no mezclar pesos reales y aleatorios.
    output_name participa en el hash de esquema — debe ser el mismo con el que
    se construyó el set. expected_model_hash activa además la validación
    estricta de identidad del modelo. None conserva el caso deliberado de
    transferencia entre modelos con la misma arquitectura/schema.
    """
    if not torch_available():
        raise TransformerTorchError(
            "PyTorch is not installed — the transformer torch module requires torch"
        )
    if not getattr(type_result, "ok", False):
        raise TransformerTorchError(
            "transformer_network_to_torch_module requires a CLEAN type_result "
            f"(errors: {list(getattr(type_result, 'errors', ['unknown']))[:2]})"
        )
    tblocks = list(getattr(type_result, "resolved_transformer_blocks", []))
    if len(tblocks) != 1 or tblocks[0].resolved_dim <= 0:
        raise TransformerTorchError(
            "transformer_network_to_torch_module requires exactly one RESOLVED "
            "transformer block (run check_composite_network_types first)"
        )
    seq_embeddings = [
        e for e in getattr(type_result, "resolved_embeddings", [])
        if e.source == network.input
    ]
    if len(seq_embeddings) != 1:
        raise TransformerTorchError(
            f"expected exactly one EMBEDDING FROM the input SEQUENCE, "
            f"found {len(seq_embeddings)}"
        )
    # Validación íntegra pre-construcción (reusa la base de network_params;
    # Sin expected_model_hash no hay programa con el que validar identidad: se
    # permite transferencia compatible por schema. Los callers de carga estricta
    # deben pasar el program_hash esperado.
    from matrixai.parameters.network_params import (
        validate_composite_network_parameter_set,
    )
    compat = validate_composite_network_parameter_set(
        network, type_result, parameter_set,
        (
            expected_model_hash
            if expected_model_hash is not None
            else getattr(parameter_set, "model_hash", "")
        ),
        output_name,
        allow_missing_values=True,  # plantilla with_values=False (M15(f))
    )
    if not compat.ok:
        raise TransformerTorchError(
            "ParameterSet incompatible with this network: "
            + "; ".join(compat.errors[:4])
        )
    return _TransformerNetworkModule(network, type_result, parameter_set, dtype)


def transformer_torch_forward_batch(
    module: Any,
    token_ids: list[list[int]],
    masks: list[list[bool]] | None = None,
    pad_id: int | None = None,
    training: bool = False,
) -> list[list[float]]:
    """Batched forward [batch, L] → list of output rows (mirrors
    composite_torch_forward_batch: eval/train toggle + no_grad + tolist)."""
    import torch
    if not token_ids:
        return []
    module_mode = module.training
    module.train() if training else module.eval()
    # Auditoría C3 [MEDIA]: try/finally — una entrada inválida (longitud,
    # vocab, máscara) no debe dejar el módulo atascado en el modo temporal
    # (p.ej. eval permanente desactivando dropout en un train posterior).
    try:
        with torch.no_grad():
            result = module.forward_batch(token_ids, masks=masks, pad_id=pad_id)
        return result.detach().cpu().tolist()
    finally:
        if training != module_mode:
            module.train(module_mode)


def _build_torch_modules() -> tuple[Any, Any]:
    """Import torch lazily and build the nn.Module classes (module-level torch
    imports would break environments without torch — patrón composite_torch)."""
    import torch
    import torch.nn as nn

    class _TransformerEncoderLayer(nn.Module):
        """Una capa del encoder por ops explícitas: MHA (matmul/softmax) +
        residual/LN + FFN. Misma secuencia de operaciones que el stdlib C2."""

        def __init__(self, dim: int, heads: int, ff: int, dropout: float, activation: str):
            super().__init__()
            self.dim = dim
            self.heads = heads
            self.dh = dim // heads
            self.activation = activation
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim, bias=False)
            self.wv = nn.Linear(dim, dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)
            self.w1 = nn.Linear(dim, ff)
            self.w2 = nn.Linear(ff, dim)
            self.norm1 = nn.LayerNorm(dim, eps=1e-5)
            self.norm2 = nn.LayerNorm(dim, eps=1e-5)
            # Dos instancias (attn y ffn) — máscaras independientes, como los
            # dos draws secuenciales del rng del stdlib.
            self.drop_attn = nn.Dropout(p=dropout)
            self.drop_ffn = nn.Dropout(p=dropout)

        def forward(self, x: Any, real_mask: Any) -> Any:
            # x [B, L, D]; real_mask [B, L] bool (True = posición real)
            bsz, seq_len, dim = x.shape
            q = self.wq(x).view(bsz, seq_len, self.heads, self.dh).transpose(1, 2)
            k = self.wk(x).view(bsz, seq_len, self.heads, self.dh).transpose(1, 2)
            v = self.wv(x).view(bsz, seq_len, self.heads, self.dh).transpose(1, 2)
            # scores [B, H, L, L]; claves de padding → -inf (peso 0.0 exacto)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.dh ** -0.5)
            scores = scores.masked_fill(
                ~real_mask[:, None, None, :], float("-inf")
            )
            weights = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(weights, v).transpose(1, 2).reshape(bsz, seq_len, dim)
            attn_out = self.drop_attn(self.wo(ctx))
            x = self.norm1(x + attn_out)
            h = _apply_torch_activation(self.w1(x), self.activation)
            ffn_out = self.drop_ffn(self.w2(h))
            return self.norm2(x + ffn_out)

    class _TransformerNetworkModule(nn.Module):
        """Red composite completa con BLOCK TRANSFORMER: embedding + POS +
        encoder + capas del stream + POOL + cabeza densa, en el MISMO orden
        intercalado que el typecheck y el stdlib C2."""

        def __init__(
            self, network: Any, type_result: Any, parameter_set: Any, dtype: Any = None
        ):
            super().__init__()
            load_dtype = dtype if dtype is not None else torch.float32
            tb = type_result.resolved_transformer_blocks[0]
            emb = next(
                e for e in type_result.resolved_embeddings
                if e.source == network.input
            )
            self.seq_len, self.dim = tb.input_shape
            self.vocab = emb.vocab
            self.pool_kind = ""
            self.net_name = network.name

            params = parameter_set.parameters
            # Auditoría C3 [ALTA]: cargas DIFERIDAS con copy_ (no reemplazo de
            # .data): copy_ exige shape idéntica (defensa en profundidad tras la
            # validación del manifest) y conserva dtype/device del módulo. El
            # módulo se convierte a load_dtype ANTES de copiar para no truncar
            # float64 a float32.
            pending_loads: list[tuple[Any, str]] = []

            def _load(module_tensor: Any, path: str) -> None:
                entry = params.get(path)
                if entry is None:
                    raise TransformerTorchError(f"Missing parameter {path!r}")
                if entry.get("values") is not None:
                    pending_loads.append((module_tensor, path))

            # Embedding por posición
            self.embedding = nn.Embedding(self.vocab, self.dim)
            _load(self.embedding.weight, f"{network.name}.{emb.name}.table")

            # Posicional: buffer sinusoidal (fórmula P10, única fuente) o
            # parámetro learned [L, dim]
            self.pos_kind = tb.pos
            if tb.pos == "learned":
                self.pos_table = nn.Parameter(
                    torch.empty(self.seq_len, self.dim)
                )
                with torch.no_grad():
                    nn.init.xavier_normal_(self.pos_table)
                _load(self.pos_table, f"{network.name}.{tb.name}.pos.table")
            else:
                self.register_buffer(
                    "pos_table",
                    torch.tensor(
                        sinusoidal_positional_table(self.seq_len, self.dim),
                        dtype=load_dtype,
                    ),
                )

            # Encoder layers con pesos del ParameterSet
            self.encoder_layers = nn.ModuleList()
            bp = f"{network.name}.{tb.name}"
            for i in range(tb.layers):
                layer = _TransformerEncoderLayer(
                    self.dim, tb.heads, tb.resolved_ff, tb.dropout, tb.activation
                )
                lp = f"{bp}.layer_{i}"
                _load(layer.wq.weight, f"{lp}.attention.Wq")
                _load(layer.wk.weight, f"{lp}.attention.Wk")
                _load(layer.wv.weight, f"{lp}.attention.Wv")
                _load(layer.wo.weight, f"{lp}.attention.Wo")
                _load(layer.w1.weight, f"{lp}.ffn.W1")
                _load(layer.w1.bias, f"{lp}.ffn.b1")
                _load(layer.w2.weight, f"{lp}.ffn.W2")
                _load(layer.w2.bias, f"{lp}.ffn.b2")
                _load(layer.norm1.weight, f"{lp}.norm1.gain")
                _load(layer.norm1.bias, f"{lp}.norm1.bias")
                _load(layer.norm2.weight, f"{lp}.norm2.gain")
                _load(layer.norm2.bias, f"{lp}.norm2.bias")
                self.encoder_layers.append(layer)

            # Cuerpo intercalado: capas del stream (pre/post bloque) y cabeza
            # tras el POOL — mismo orden que el typecheck/stdlib.
            self.stream_ops = nn.ModuleDict()   # módulos con parámetros
            self._body_plan: list[tuple[str, Any]] = []  # (fase, spec) en orden
            block_key = tb.position * 2 + 1
            items: list[tuple[int, str, Any]] = [
                (layer.index * 2, "layer", layer)
                for layer in type_result.resolved_layers
            ]
            items.append((block_key, "tblock", tb))
            items.sort(key=lambda it: it[0])

            in_stream = True
            for _, kind, spec in items:
                if kind == "tblock":
                    self._body_plan.append(("encoder", None))
                    continue
                lt = spec.layer_type
                key = f"L{spec.index}"
                if in_stream:
                    if lt == "Pool":
                        self.pool_kind = spec.pool_kind
                        self._body_plan.append(("pool", spec))
                        in_stream = False
                    elif lt == "LayerNorm":
                        ln = nn.LayerNorm(self.dim, eps=1e-5)
                        _load(ln.weight, f"{network.name}.{key}.gamma")
                        _load(ln.bias, f"{network.name}.{key}.beta")
                        self.stream_ops[key] = ln
                        self._body_plan.append(("module", key))
                    elif lt == "Dropout":
                        self.stream_ops[key] = nn.Dropout(p=spec.rate)
                        self._body_plan.append(("module", key))
                    elif lt == "Activation":
                        self._body_plan.append(("activation", spec.activation_kind))
                    else:
                        raise TransformerTorchError(
                            f"LAYER {lt} cannot run on the sequence stream "
                            f"(typecheck should have rejected this)"
                        )
                else:
                    # Cabeza tras el POOL: Dense / Dropout / Activation / LayerNorm
                    if lt == "Dense":
                        linear = nn.Linear(
                            spec.input_shape[-1] if spec.input_shape else self.dim,
                            spec.units,
                        )
                        _load(linear.weight, f"{network.name}.{key}.W")
                        _load(linear.bias, f"{network.name}.{key}.b")
                        self.stream_ops[key] = linear
                        self._body_plan.append(("dense", (key, spec.activation)))
                    elif lt == "Dropout":
                        self.stream_ops[key] = nn.Dropout(p=spec.rate)
                        self._body_plan.append(("module", key))
                    elif lt == "Activation":
                        self._body_plan.append(("activation", spec.activation_kind))
                    elif lt == "LayerNorm":
                        ln = nn.LayerNorm(spec.input_shape[-1], eps=1e-5)
                        _load(ln.weight, f"{network.name}.{key}.gamma")
                        _load(ln.bias, f"{network.name}.{key}.beta")
                        self.stream_ops[key] = ln
                        self._body_plan.append(("module", key))
                    else:
                        raise TransformerTorchError(
                            f"LAYER {lt} is not supported in the post-POOL head"
                        )
            if self.pool_kind == "":
                raise TransformerTorchError(
                    "the network body never pooled the stream "
                    "(typecheck should require POOL)"
                )
            # Conversión de dtype ANTES de las cargas (copy_ conserva el dtype
            # destino; si convirtiéramos después, float64 quedaría truncado) —
            # también unifica los inits nativos de values=None.
            if dtype is not None:
                self.to(dtype)
            with torch.no_grad():
                for tensor, path in pending_loads:
                    source = torch.tensor(
                        params[path]["values"], dtype=tensor.dtype
                    )
                    if source.shape != tensor.shape:
                        raise TransformerTorchError(
                            f"parameter {path!r} values shape "
                            f"{list(source.shape)} != module tensor shape "
                            f"{list(tensor.shape)}"
                        )
                    tensor.copy_(source)

        def forward_batch(
            self,
            token_ids: Any,
            masks: Any = None,
            pad_id: int | None = None,
        ) -> Any:
            """[batch, L] ids (+ máscara) → [batch, units]."""
            device = self.embedding.weight.device
            dtype = self.embedding.weight.dtype
            ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
            if ids.dim() != 2 or ids.shape[1] != self.seq_len:
                raise TransformerTorchError(
                    f"expected token ids [batch, {self.seq_len}], got {list(ids.shape)}"
                )
            if int(ids.min()) < 0 or int(ids.max()) >= self.vocab:
                raise TransformerTorchError(
                    f"token id out of range [0, {self.vocab}) in batch"
                )
            if masks is not None:
                mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
                if mask.shape != ids.shape:
                    raise TransformerTorchError(
                        f"mask shape {list(mask.shape)} != ids shape {list(ids.shape)}"
                    )
            elif pad_id is not None:
                mask = ids != pad_id
            else:
                mask = torch.ones_like(ids, dtype=torch.bool)
            if not bool(mask.any(dim=1).all()):
                raise TransformerTorchError(
                    "each batch row must keep at least one real position"
                )

            x = self.embedding(ids) + self.pos_table.to(dtype)[None, :, :]

            pooled = False
            for phase, arg in self._body_plan:
                if phase == "encoder":
                    for layer in self.encoder_layers:
                        x = layer(x, mask)
                elif phase == "pool":
                    if self.pool_kind == "cls":
                        if not bool(mask[:, 0].all()):
                            raise TransformerTorchError(
                                "POOL cls requires position 0 to be a REAL token "
                                "in every batch row (invariante 1c)"
                            )
                        x = x[:, 0, :]
                    else:  # mean enmascarado
                        m = mask.to(dtype)[:, :, None]
                        x = (x * m).sum(dim=1) / m.sum(dim=1)
                    pooled = True
                elif phase == "module":
                    x = self.stream_ops[arg](x)
                elif phase == "activation":
                    x = _apply_torch_activation(x, arg)
                elif phase == "dense":
                    key, activation = arg
                    x = _apply_torch_activation(self.stream_ops[key](x), activation)
            if not pooled:
                raise TransformerTorchError("forward finished without pooling")
            return x

    return _TransformerEncoderLayer, _TransformerNetworkModule


_ENCODER_LAYER_CLS: Any = None
_NETWORK_MODULE_CLS: Any = None


def _module_classes() -> tuple[Any, Any]:
    global _ENCODER_LAYER_CLS, _NETWORK_MODULE_CLS
    if _NETWORK_MODULE_CLS is None:
        _ENCODER_LAYER_CLS, _NETWORK_MODULE_CLS = _build_torch_modules()
    return _ENCODER_LAYER_CLS, _NETWORK_MODULE_CLS


def _TransformerNetworkModule(
    network: Any, type_result: Any, parameter_set: Any, dtype: Any = None
) -> Any:
    _, cls = _module_classes()
    return cls(network, type_result, parameter_set, dtype)


def transformer_encoder_layer_class() -> Any:
    """La clase de capa del encoder (para los tests de paridad externa C3 —
    el producto SIEMPRE entra por transformer_network_to_torch_module)."""
    cls, _ = _module_classes()
    return cls
