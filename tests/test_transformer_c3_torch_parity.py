# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""TRANSFORMER_BLOQUE C3 — paridad torch (techo) + anti-transformer-falso.

Las tres paridades del contrato:
  1. stdlib (C2, float64 puro) == módulo torch (ops explícitas) con los MISMOS
     pesos — tolerancia documentada: atol 1e-9 en float64 (medida real ~1e-16);
     en float32 (dtype de producto) rtol/atol 1e-5.
  2. Paridad EXTERNA (invariante 1a): nuestro encoder == torch.nn.MultiheadAttention
     (bias=False) + FFN/LN de referencia con pesos idénticos. nn.MultiheadAttention
     SOLO aparece aquí como referencia de test (decisión 2: nunca en producto).
  3. stdlib == referencia externa (transitiva, apilando capas externas sobre el
     stream embebido del stdlib).

Más: batch>1 == batch=1 apilado, dropout=0 en eval, CUDA skip-if-unavailable.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from matrixai.forward.transformer_forward import transformer_network_forward
from matrixai.forward.transformer_torch import (
    TransformerTorchError,
    transformer_encoder_layer_class,
    transformer_network_to_torch_module,
    transformer_torch_forward_batch,
)
from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types

# Tolerancias documentadas (contrato C3)
ATOL_F64 = 1e-9   # ambos backends en float64 — medida real ~1e-16
ATOL_F32 = 1e-5   # dtype de producto


def _mxai(
    *, length: int = 6, vocab: int = 11, dim: int = 8, layers: int = 2,
    heads: int = 2, ff: int = 16, pos: str = "sinusoidal", pool: str = "mean",
    dropout: float = 0.0, activation: str = "gelu", pre_block: str = "",
) -> str:
    return f"""
PROJECT C3Test

SEQUENCE Texto
  length = {length}
  vocab_size = {vocab}
END

NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM {dim}
  {pre_block}
  BLOCK enc TRANSFORMER
    LAYERS {layers}
    HEADS {heads}
    FF {ff}
    DROPOUT {dropout}
    ACTIVATION {activation}
    POS {pos}
  END
  POOL {pool}
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END
"""


def _build(src: str, seed: int = 7, output_name: str = ""):
    prog = parse_text(src)
    net = prog.networks[0]
    res = check_composite_network_types(
        net, {v.name: v for v in prog.vectors}, {s.name: s for s in prog.sequences}
    )
    assert res.ok, res.errors
    ps = build_composite_network_parameter_set(
        net, res, "mxai_c3test", seed=seed, output_name=output_name
    )
    return prog, net, res, ps


def _module64(net, res, ps):
    return transformer_network_to_torch_module(net, res, ps, dtype=torch.float64)


def _assert_parity(net, res, ps, ids, mask=None, atol=ATOL_F64):
    std = transformer_network_forward(net, res, ps, ids, mask=mask)
    module = _module64(net, res, ps)
    tor = transformer_torch_forward_batch(
        module, [ids], masks=[mask] if mask is not None else None
    )
    assert max(abs(a - b) for a, b in zip(std.output, tor[0])) < atol


IDS = [1, 2, 3, 4, 5, 6]
MASK3 = [True, True, True, False, False, False]


# ---------------------------------------------------------------------------
# Paridad 1 — stdlib (C2) == torch, mismos pesos
# ---------------------------------------------------------------------------

class TestParidadStdlibTorch:
    def test_base_mean_sinusoidal_gelu(self):
        _, net, res, ps = _build(_mxai())
        _assert_parity(net, res, ps, IDS)

    def test_with_padding_mask(self):
        _, net, res, ps = _build(_mxai())
        _assert_parity(net, res, ps, [1, 2, 3, 0, 0, 0], mask=MASK3)

    def test_cls_pooling(self):
        _, net, res, ps = _build(_mxai(pool="cls"))
        _assert_parity(net, res, ps, [1, 2, 3, 0, 0, 0], mask=MASK3)

    def test_learned_pos(self):
        _, net, res, ps = _build(_mxai(pos="learned"))
        _assert_parity(net, res, ps, IDS)

    def test_relu_activation(self):
        _, net, res, ps = _build(_mxai(activation="relu"))
        _assert_parity(net, res, ps, IDS)

    @pytest.mark.parametrize("heads", [1, 4])
    def test_head_counts(self, heads):
        _, net, res, ps = _build(_mxai(heads=heads))
        _assert_parity(net, res, ps, IDS)

    def test_pre_block_layernorm(self):
        _, net, res, ps = _build(_mxai(pre_block="LAYER LayerNorm"))
        _assert_parity(net, res, ps, IDS)

    def test_dropout_identity_in_eval_parity(self):
        _, net, res, ps = _build(_mxai(dropout=0.4))
        _assert_parity(net, res, ps, IDS)  # el wrapper fuerza eval

    def test_float32_product_dtype_parity(self):
        _, net, res, ps = _build(_mxai())
        std = transformer_network_forward(net, res, ps, IDS)
        module = transformer_network_to_torch_module(net, res, ps)  # float32
        tor = transformer_torch_forward_batch(module, [IDS])
        assert max(abs(a - b) for a, b in zip(std.output, tor[0])) < ATOL_F32

    def test_intermediate_block_output_parity(self):
        """No solo la salida final: el stream pre-pooling también coincide."""
        _, net, res, ps = _build(_mxai())
        std = transformer_network_forward(net, res, ps, IDS)
        module = _module64(net, res, ps).eval()
        ids_t = torch.tensor([IDS])
        with torch.no_grad():
            x = module.embedding(ids_t) + module.pos_table[None, :, :]
            mask = torch.ones_like(ids_t, dtype=torch.bool)
            for layer in module.encoder_layers:
                x = layer(x, mask)
        mine = torch.tensor(std.block_output, dtype=torch.float64)
        assert (x[0] - mine).abs().max().item() < ATOL_F64


# ---------------------------------------------------------------------------
# Paridad 2 — EXTERNA: nn.MultiheadAttention + FFN de referencia (invariante 1a)
# ---------------------------------------------------------------------------

def _external_reference_layer(our_layer, x, real_mask=None):
    """Capa de referencia con torch.nn.MultiheadAttention (SOLO test) y los
    MISMOS pesos que nuestra capa de ops explícitas."""
    import torch.nn as nn
    import torch.nn.functional as F
    dim, heads = our_layer.dim, our_layer.heads
    mha = nn.MultiheadAttention(dim, heads, bias=False, batch_first=True).double()
    with torch.no_grad():
        mha.in_proj_weight.data = torch.cat(
            [our_layer.wq.weight, our_layer.wk.weight, our_layer.wv.weight], dim=0
        ).clone()
        mha.out_proj.weight.data = our_layer.wo.weight.clone()
    key_padding = None if real_mask is None else ~real_mask  # True = IGNORAR
    attn, _ = mha(x, x, x, key_padding_mask=key_padding, need_weights=False)
    y = F.layer_norm(
        x + attn, (dim,), our_layer.norm1.weight, our_layer.norm1.bias, eps=1e-5
    )
    h = F.gelu(y @ our_layer.w1.weight.T + our_layer.w1.bias)
    ffn = h @ our_layer.w2.weight.T + our_layer.w2.bias
    return F.layer_norm(
        y + ffn, (dim,), our_layer.norm2.weight, our_layer.norm2.bias, eps=1e-5
    )


class TestParidadExterna:
    def _our_layer(self, dim=8, heads=2, ff=16):
        cls = transformer_encoder_layer_class()
        torch.manual_seed(11)
        return cls(dim, heads, ff, dropout=0.0, activation="gelu").double().eval()

    def test_layer_vs_multihead_attention(self):
        layer = self._our_layer()
        torch.manual_seed(3)
        x = torch.randn(2, 6, 8, dtype=torch.float64)
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            ours = layer(x, mask)
            ref = _external_reference_layer(layer, x)
        assert (ours - ref).abs().max().item() < ATOL_F64

    def test_layer_vs_multihead_attention_with_mask(self):
        layer = self._our_layer()
        torch.manual_seed(4)
        x = torch.randn(2, 6, 8, dtype=torch.float64)
        mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
        with torch.no_grad():
            ours = layer(x, mask)
            ref = _external_reference_layer(layer, x, real_mask=mask)
        # Solo posiciones reales: la fila de un pad como QUERY difiere en la
        # referencia (nn.MultiheadAttention también la calcula) pero no entra
        # en ningún pooling — se comparan las posiciones reales.
        for b in range(2):
            for t in range(6):
                if mask[b, t]:
                    assert (ours[b, t] - ref[b, t]).abs().max().item() < ATOL_F64

    @pytest.mark.parametrize("heads", [1, 2, 4])
    def test_multiple_head_counts_vs_external(self, heads):
        layer = self._our_layer(heads=heads)
        torch.manual_seed(5)
        x = torch.randn(1, 6, 8, dtype=torch.float64)
        mask = torch.ones(1, 6, dtype=torch.bool)
        with torch.no_grad():
            ours = layer(x, mask)
            ref = _external_reference_layer(layer, x)
        assert (ours - ref).abs().max().item() < ATOL_F64

    def test_stdlib_vs_external_stack(self):
        """Paridad 3 (transitiva): el block_output del stdlib C2 == apilar las
        capas de REFERENCIA EXTERNA sobre el mismo stream embebido."""
        _, net, res, ps = _build(_mxai())
        std = transformer_network_forward(net, res, ps, IDS)
        module = _module64(net, res, ps).eval()
        x = torch.tensor(std.embedded, dtype=torch.float64)[None, :, :]
        with torch.no_grad():
            for layer in module.encoder_layers:
                x = _external_reference_layer(layer, x)
        mine = torch.tensor(std.block_output, dtype=torch.float64)
        assert (x[0] - mine).abs().max().item() < ATOL_F64


# ---------------------------------------------------------------------------
# Batch, dropout, CUDA
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_equals_stacked_singles(self):
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        rows = [IDS, [6, 5, 4, 3, 2, 1], [1, 1, 2, 2, 3, 3]]
        masks = [[True] * 6, MASK3, [True, True, True, True, False, False]]
        batched = transformer_torch_forward_batch(module, rows, masks=masks)
        singles = [
            transformer_torch_forward_batch(module, [r], masks=[m])[0]
            for r, m in zip(rows, masks)
        ]
        for b, s in zip(batched, singles):
            assert max(abs(x - y) for x, y in zip(b, s)) < 1e-12

    def test_batch_rows_match_stdlib_per_sample(self):
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        rows = [IDS, [2, 4, 6, 8, 10, 1]]
        batched = transformer_torch_forward_batch(module, rows)
        for row, out in zip(rows, batched):
            std = transformer_network_forward(net, res, ps, row)
            assert max(abs(a - b) for a, b in zip(std.output, out)) < ATOL_F64

    def test_pad_id_derivation_matches_explicit_mask(self):
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        by_pad = transformer_torch_forward_batch(
            module, [[1, 2, 3, 0, 0, 0]], pad_id=0
        )
        by_mask = transformer_torch_forward_batch(
            module, [[1, 2, 3, 0, 0, 0]], masks=[MASK3]
        )
        assert by_pad == by_mask


class TestDropoutYModo:
    def test_dropout_zero_in_eval(self):
        """dropout=0 en eval (contrato C3): con rate>0, eval es determinista e
        idéntico a la misma red con rate=0 (mismos pesos: dropout no aporta
        parámetros y el resto del manifest coincide con el mismo seed)."""
        _, net_d, res_d, ps_d = _build(_mxai(dropout=0.5), seed=7)
        _, net_0, res_0, ps_0 = _build(_mxai(dropout=0.0), seed=7)
        assert {k: v["values"] for k, v in ps_d.parameters.items()} == {
            k: v["values"] for k, v in ps_0.parameters.items()
        }
        m_d = _module64(net_d, res_d, ps_d)
        m_0 = _module64(net_0, res_0, ps_0)
        out_d1 = transformer_torch_forward_batch(m_d, [IDS])
        out_d2 = transformer_torch_forward_batch(m_d, [IDS])
        out_0 = transformer_torch_forward_batch(m_0, [IDS])
        assert out_d1 == out_d2 == out_0

    def test_dropout_active_in_train_mode(self):
        _, net, res, ps = _build(_mxai(dropout=0.5))
        module = _module64(net, res, ps)
        torch.manual_seed(1)
        a = transformer_torch_forward_batch(module, [IDS], training=True)
        torch.manual_seed(2)
        b = transformer_torch_forward_batch(module, [IDS], training=True)
        assert a != b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
class TestCUDA:
    def test_cpu_cuda_parity_float32(self):
        _, net, res, ps = _build(_mxai())
        module = transformer_network_to_torch_module(net, res, ps)
        cpu_out = transformer_torch_forward_batch(module, [IDS], masks=[MASK3])
        module_gpu = module.cuda()
        gpu_out = transformer_torch_forward_batch(module_gpu, [IDS], masks=[MASK3])
        assert max(abs(a - b) for a, b in zip(cpu_out[0], gpu_out[0])) < ATOL_F32


# ---------------------------------------------------------------------------
# Guards y anti-falso en el camino torch
# ---------------------------------------------------------------------------

class TestGuardsTorch:
    def test_dirty_type_result_rejected(self):
        prog = parse_text(_mxai(heads=3))  # 3 no divide 8
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {}, {s.name: s for s in prog.sequences}
        )
        _, _, _, ps = _build(_mxai(heads=2))
        with pytest.raises(TransformerTorchError, match="CLEAN type_result"):
            transformer_network_to_torch_module(net, res, ps)

    def test_wrong_length_rejected(self):
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        with pytest.raises(TransformerTorchError, match="expected token ids"):
            module.forward_batch([[1, 2, 3]])

    def test_vocab_out_of_range_rejected(self):
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        with pytest.raises(TransformerTorchError, match="out of range"):
            module.forward_batch([[1, 2, 3, 4, 5, 99]])

    def test_all_masked_row_rejected(self):
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        with pytest.raises(TransformerTorchError, match="at least one real"):
            module.forward_batch([IDS], masks=[[False] * 6])

    def test_cls_masked_position_0_rejected_per_batch_row(self):
        _, net, res, ps = _build(_mxai(pool="cls"))
        module = _module64(net, res, ps)
        masks = [[True] * 6, [False, True, True, True, False, False]]
        with pytest.raises(TransformerTorchError, match="POOL cls requires position 0"):
            module.forward_batch([IDS, [0, 2, 3, 4, 0, 0]], masks=masks)

    def test_composite_builder_redirects_to_transformer_module(self):
        from matrixai.forward.composite_torch import (
            CompositeTorchError,
            composite_network_to_torch_module,
        )
        _, net, res, ps = _build(_mxai())
        with pytest.raises(CompositeTorchError, match="transformer_network_to_torch_module"):
            composite_network_to_torch_module(net, ps)

    def test_with_values_false_native_init_runs(self):
        """M15(f): plantilla sin valores → init nativo de torch, forward corre."""
        _, net, res, _ = _build(_mxai())
        template = build_composite_network_parameter_set(
            net, res, "mxai_c3test", seed=7, with_values=False
        )
        torch.manual_seed(9)
        module = transformer_network_to_torch_module(net, res, template)
        out = transformer_torch_forward_batch(module, [IDS])
        assert len(out[0]) == 2
        assert abs(sum(out[0]) - 1.0) < 1e-5

    def test_permutation_changes_output_torch(self):
        """Anti-falso también en el camino torch: la atención VE posiciones."""
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        a = transformer_torch_forward_batch(module, [[1, 2, 3, 4, 5, 6]])
        b = transformer_torch_forward_batch(module, [[2, 1, 3, 4, 5, 6]])
        assert a != b

    def test_padding_content_irrelevant_torch(self):
        _, net, res, ps = _build(_mxai())
        module = _module64(net, res, ps)
        a = transformer_torch_forward_batch(module, [[1, 2, 3, 0, 0, 0]], masks=[MASK3])
        b = transformer_torch_forward_batch(module, [[1, 2, 3, 9, 7, 5]], masks=[MASK3])
        assert a == b


# ---------------------------------------------------------------------------
# Auditoría C3 (2026-07-11): validación íntegra pre-construcción, try/finally,
# delegación real del builder común
# ---------------------------------------------------------------------------

class TestAuditoriaC3:
    def test_schema_hash_mismatch_rejected(self):
        """[ALTA] Un ParameterSet de HEADS=1 tiene shapes IDÉNTICAS a HEADS=2 —
        solo el parameter_schema_hash los distingue; aceptarlo era corrupción
        semántica silenciosa."""
        _, net2, res2, _ = _build(_mxai(heads=2))
        _, _, _, ps1 = _build(_mxai(heads=1))
        with pytest.raises(TransformerTorchError, match="parameter_schema_hash mismatch"):
            transformer_network_to_torch_module(net2, res2, ps1)

    def test_malformed_values_rejected_at_build(self):
        """[ALTA] Wq [9,8] (fila extra) fallaba TARDE con RuntimeError interno
        de reshape — ahora falla en construcción con el error contractual."""
        _, net, res, ps = _build(_mxai())
        key = "N.enc.layer_0.attention.Wq"
        ps.parameters[key]["values"] = ps.parameters[key]["values"] + [[0.0] * 8]
        with pytest.raises(TransformerTorchError, match="incompatible"):
            transformer_network_to_torch_module(net, res, ps)

    def test_unexpected_parameter_rejected(self):
        _, net, res, ps = _build(_mxai())
        ps.parameters["N.enc.layer_9.attention.Wq"] = {
            "values": [[0.0] * 8] * 8, "shape": [8, 8],
        }
        with pytest.raises(TransformerTorchError, match="unexpected parameter"):
            transformer_network_to_torch_module(net, res, ps)

    def test_metadata_shape_mismatch_rejected(self):
        _, net, res, ps = _build(_mxai())
        ps.parameters["N.enc.layer_0.ffn.b1"]["shape"] = [17]
        with pytest.raises(TransformerTorchError, match="expected shape"):
            transformer_network_to_torch_module(net, res, ps)

    def test_mode_restored_after_exception(self):
        """[MEDIA] Sin try/finally, una llamada eval con entrada inválida
        dejaba el módulo permanentemente en eval (dropout desactivado)."""
        _, net, res, ps = _build(_mxai(dropout=0.5))
        module = transformer_network_to_torch_module(net, res, ps)
        module.train()
        with pytest.raises(TransformerTorchError):
            transformer_torch_forward_batch(module, [[1, 2, 3]])  # L inválida
        assert module.training is True

    def test_composite_builder_delegates_with_type_result(self):
        """[MEDIA] Delegación REAL: la entrada común construye el módulo
        transformer cuando recibe el type_result."""
        from matrixai.forward.composite_torch import composite_network_to_torch_module
        _, net, res, ps = _build(_mxai())
        module = composite_network_to_torch_module(net, ps, type_result=res)
        out = transformer_torch_forward_batch(module, [IDS])
        std = transformer_network_forward(net, res, ps, IDS)
        # float32 (dtype de producto de la entrada común)
        assert max(abs(a - b) for a, b in zip(std.output, out[0])) < ATOL_F32

    def test_composite_builder_without_type_result_asks_for_it(self):
        from matrixai.forward.composite_torch import (
            CompositeTorchError,
            composite_network_to_torch_module,
        )
        _, net, res, ps = _build(_mxai())
        with pytest.raises(CompositeTorchError, match="pass type_result"):
            composite_network_to_torch_module(net, ps)

    def test_template_without_values_still_builds(self):
        """allow_missing_values: la plantilla M15(f) valida estructura y
        omite solo la shape del valor ausente."""
        _, net, res, _ = _build(_mxai())
        template = build_composite_network_parameter_set(
            net, res, "mxai_c3test", seed=7, with_values=False
        )
        module = transformer_network_to_torch_module(net, res, template)
        out = transformer_torch_forward_batch(module, [IDS])
        assert len(out[0]) == 2

    def test_partially_missing_values_rejected(self):
        """Reauditoría: values=None solo representa la plantilla COMPLETA
        M15(f); mezclar pesos reales y ausentes no puede inicializar al azar en silencio."""
        _, net, res, ps = _build(_mxai())
        ps.parameters["N.enc.layer_0.attention.Wq"]["values"] = None
        with pytest.raises(TransformerTorchError, match="mixes materialized values"):
            transformer_network_to_torch_module(net, res, ps)

    def test_common_builder_propagates_output_name(self):
        """La entrada común debe aceptar todo set válido del builder
        especializado, incluido el output_name que forma parte del schema hash."""
        from matrixai.forward.composite_torch import composite_network_to_torch_module
        _, net, res, ps = _build(_mxai(), output_name="clase")
        module = composite_network_to_torch_module(
            net, ps, type_result=res, output_name="clase"
        )
        out = transformer_torch_forward_batch(module, [IDS])
        assert len(out[0]) == 2

    def test_expected_model_hash_strict_mode(self):
        """Sin expected_model_hash se permite transferencia por schema; cuando
        el caller lo aporta por la entrada común, una identidad distinta debe
        propagarse al builder especializado y fallar cerrada."""
        from matrixai.forward.composite_torch import composite_network_to_torch_module
        _, net, res, ps = _build(_mxai())
        with pytest.raises(TransformerTorchError, match="model_hash mismatch"):
            composite_network_to_torch_module(
                net,
                ps,
                type_result=res,
                expected_model_hash="another-model",
            )
