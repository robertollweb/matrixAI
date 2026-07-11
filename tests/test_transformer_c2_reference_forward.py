# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""TRANSFORMER_BLOQUE C2 — forward stdlib de referencia + manifest + param_count.

Cubre el contrato 51 §C2: shapes intermedios, invariante 1b (permutación) y 1c
(máscara de padding), multi-head ≠ single-head con mismos pesos, paths exactos
del manifest, param_count contra la fórmula cerrada, PAD determinista, liveness
(el forward toca EXACTAMENTE los paths del manifest) y el estado del backend
tras C2 (lowering auditable, entrenamiento cerrado hasta C4).
"""
from __future__ import annotations

import pytest

from matrixai.compiler.backend_contract import BackendContractAnalyzer
from matrixai.forward.transformer_forward import (
    TransformerForwardError,
    sinusoidal_positional_table,
    transformer_network_forward,
)
from matrixai.parameters.network_params import (
    build_composite_network_parameter_set,
    composite_network_parameter_manifest,
    composite_network_parameter_schema_hash,
    manifest_scalar_count,
    transformer_block_param_count,
)
from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mxai(
    *, length: int = 6, vocab: int = 11, dim: int = 8, layers: int = 2,
    heads: int = 2, ff: int = 16, pos: str = "sinusoidal", pool: str = "mean",
    dropout: float = 0.0, activation: str = "gelu",
) -> str:
    return f"""
PROJECT C2Test

SEQUENCE Texto
  length = {length}
  vocab_size = {vocab}
END

NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM {dim}
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

GRAPH
  Texto -> N
END
"""


def _build(src: str, seed: int = 7):
    prog = parse_text(src)
    net = prog.networks[0]
    res = check_composite_network_types(
        net, {v.name: v for v in prog.vectors}, {s.name: s for s in prog.sequences}
    )
    assert res.ok, res.errors
    ps = build_composite_network_parameter_set(net, res, "mxai_c2test", seed=seed)
    return prog, net, res, ps


IDS = [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Shapes intermedios
# ---------------------------------------------------------------------------

class TestShapes:
    def test_intermediate_shapes(self):
        _, net, res, ps = _build(_mxai())
        tr = transformer_network_forward(net, res, ps, IDS)
        assert len(tr.embedded) == 6 and all(len(r) == 8 for r in tr.embedded)
        assert len(tr.layer_outputs) == 2
        for lo in tr.layer_outputs:
            assert len(lo) == 6 and all(len(r) == 8 for r in lo)
        assert len(tr.block_output) == 6 and all(len(r) == 8 for r in tr.block_output)
        assert len(tr.pooled) == 8
        assert len(tr.output) == 2

    def test_attention_weights_shape_and_rows_sum_1(self):
        _, net, res, ps = _build(_mxai())
        tr = transformer_network_forward(net, res, ps, IDS)
        # [layer][head][L][L], cada fila del softmax suma 1
        assert len(tr.attention_weights) == 2
        for layer_attn in tr.attention_weights:
            assert len(layer_attn) == 2
            for head in layer_attn:
                assert len(head) == 6
                for row in head:
                    assert len(row) == 6
                    assert abs(sum(row) - 1.0) < 1e-9

    def test_softmax_head_output_sums_to_one(self):
        _, net, res, ps = _build(_mxai())
        tr = transformer_network_forward(net, res, ps, IDS)
        assert abs(sum(tr.output) - 1.0) < 1e-9

    def test_wrong_length_rejected(self):
        _, net, res, ps = _build(_mxai())
        with pytest.raises(TransformerForwardError, match="expected 6 token ids"):
            transformer_network_forward(net, res, ps, [1, 2, 3])

    def test_token_out_of_vocab_rejected(self):
        _, net, res, ps = _build(_mxai())
        with pytest.raises(TransformerForwardError, match="out of range"):
            transformer_network_forward(net, res, ps, [1, 2, 3, 4, 5, 99])


# ---------------------------------------------------------------------------
# Invariante 1b — permutación: la atención VE posiciones
# ---------------------------------------------------------------------------

class TestPermutacion:
    def test_swapping_positions_changes_prepool_and_pooled(self):
        _, net, res, ps = _build(_mxai())
        a = transformer_network_forward(net, res, ps, [1, 2, 3, 4, 5, 6])
        b = transformer_network_forward(net, res, ps, [2, 1, 3, 4, 5, 6])
        assert a.block_output != b.block_output
        # mean-pooling de un modelo ciego a posiciones daría el MISMO pooled
        # para el mismo multiconjunto de tokens — aquí debe cambiar (POS+atención)
        assert a.pooled != b.pooled
        assert a.output != b.output

    def test_same_input_same_output(self):
        _, net, res, ps = _build(_mxai())
        a = transformer_network_forward(net, res, ps, IDS)
        b = transformer_network_forward(net, res, ps, IDS)
        assert a.output == b.output
        assert a.block_output == b.block_output


# ---------------------------------------------------------------------------
# Invariante 1c — máscara: el padding no influye
# ---------------------------------------------------------------------------

class TestMascara:
    def test_padding_content_does_not_change_output(self):
        _, net, res, ps = _build(_mxai())
        mask = [True, True, True, False, False, False]
        a = transformer_network_forward(net, res, ps, [1, 2, 3, 0, 0, 0], mask=mask)
        b = transformer_network_forward(net, res, ps, [1, 2, 3, 9, 7, 5], mask=mask)
        # bit a bit: los pesos de atención de claves enmascaradas son exactamente
        # 0.0 (score -inf) y el mean-pooling excluye las posiciones de padding
        assert a.output == b.output
        assert a.pooled == b.pooled

    def test_padding_content_does_not_change_cls_output(self):
        _, net, res, ps = _build(_mxai(pool="cls"))
        mask = [True, True, True, False, False, False]
        a = transformer_network_forward(net, res, ps, [1, 2, 3, 0, 0, 0], mask=mask)
        b = transformer_network_forward(net, res, ps, [1, 2, 3, 9, 7, 5], mask=mask)
        assert a.output == b.output

    def test_mask_derived_from_pad_id(self):
        _, net, res, ps = _build(_mxai())
        explicit = transformer_network_forward(
            net, res, ps, [1, 2, 3, 0, 0, 0],
            mask=[True, True, True, False, False, False],
        )
        derived = transformer_network_forward(net, res, ps, [1, 2, 3, 0, 0, 0], pad_id=0)
        assert explicit.output == derived.output
        assert derived.mask == [True, True, True, False, False, False]

    def test_masked_mean_excludes_padding_positions(self):
        _, net, res, ps = _build(_mxai())
        mask = [True, True, True, False, False, False]
        tr = transformer_network_forward(net, res, ps, [1, 2, 3, 0, 0, 0], mask=mask)
        expected = [
            sum(tr.block_output[t][i] for t in range(3)) / 3.0
            for i in range(8)
        ]
        assert tr.pooled == expected

    def test_masking_real_tokens_changes_output(self):
        """La máscara importa: enmascarar una posición real cambia la salida."""
        _, net, res, ps = _build(_mxai())
        full = transformer_network_forward(net, res, ps, IDS)
        partial = transformer_network_forward(
            net, res, ps, IDS, mask=[True, True, True, True, True, False]
        )
        assert full.output != partial.output

    def test_all_masked_rejected(self):
        _, net, res, ps = _build(_mxai())
        with pytest.raises(TransformerForwardError, match="at least one real position"):
            transformer_network_forward(net, res, ps, IDS, mask=[False] * 6)


# ---------------------------------------------------------------------------
# Multi-head ≠ single-head con los mismos pesos base
# ---------------------------------------------------------------------------

class TestMultiHead:
    def test_heads_change_output_with_same_weights(self):
        # Mismas shapes (HEADS no cambia ningún parámetro) y mismo seed →
        # ParameterSets con valores idénticos; solo cambia el split de cabezas.
        _, net2, res2, ps2 = _build(_mxai(heads=2), seed=7)
        _, net1, res1, ps1 = _build(_mxai(heads=1), seed=7)
        assert {
            k: v["values"] for k, v in ps2.parameters.items()
        } == {k: v["values"] for k, v in ps1.parameters.items()}
        out2 = transformer_network_forward(net2, res2, ps2, IDS)
        out1 = transformer_network_forward(net1, res1, ps1, IDS)
        assert out2.block_output != out1.block_output
        assert out2.output != out1.output

    def test_heads_change_schema_hash(self):
        # heads no cambia shapes: el hash de esquema DEBE distinguirlo igualmente
        _, net2, res2, _ = _build(_mxai(heads=2))
        _, net1, res1, _ = _build(_mxai(heads=1))
        h2 = composite_network_parameter_schema_hash(net2.name, net2, res2)
        h1 = composite_network_parameter_schema_hash(net1.name, net1, res1)
        assert h2 != h1


# ---------------------------------------------------------------------------
# Manifest: paths exactos y liveness
# ---------------------------------------------------------------------------

class TestManifest:
    def test_exact_paths_sinusoidal(self):
        _, net, res, _ = _build(_mxai(layers=1))
        paths = [m["path"] for m in composite_network_parameter_manifest(net.name, net, res)]
        expected = [
            "N.tok.table",
            "N.enc.layer_0.attention.Wq",
            "N.enc.layer_0.attention.Wk",
            "N.enc.layer_0.attention.Wv",
            "N.enc.layer_0.attention.Wo",
            "N.enc.layer_0.ffn.W1",
            "N.enc.layer_0.ffn.b1",
            "N.enc.layer_0.ffn.W2",
            "N.enc.layer_0.ffn.b2",
            "N.enc.layer_0.norm1.gain",
            "N.enc.layer_0.norm1.bias",
            "N.enc.layer_0.norm2.gain",
            "N.enc.layer_0.norm2.bias",
            # POOL es L1 (sin parámetros); las Dense de la cabeza son L2 y L3
            "N.L2.W",
            "N.L2.b",
            "N.L3.W",
            "N.L3.b",
        ]
        assert paths == expected

    def test_learned_pos_adds_table(self):
        _, net, res, _ = _build(_mxai(layers=1, pos="learned"))
        manifest = composite_network_parameter_manifest(net.name, net, res)
        pos_entries = [m for m in manifest if m["path"] == "N.enc.pos.table"]
        assert len(pos_entries) == 1
        assert pos_entries[0]["shape"] == [6, 8]

    def test_shapes_in_manifest(self):
        _, net, res, _ = _build(_mxai(layers=1))
        by_path = {m["path"]: m for m in composite_network_parameter_manifest(net.name, net, res)}
        assert by_path["N.tok.table"]["shape"] == [11, 8]
        assert by_path["N.enc.layer_0.attention.Wq"]["shape"] == [8, 8]
        assert by_path["N.enc.layer_0.ffn.W1"]["shape"] == [16, 8]
        assert by_path["N.enc.layer_0.ffn.W2"]["shape"] == [8, 16]
        assert by_path["N.enc.layer_0.norm1.gain"]["shape"] == [8]

    def test_liveness_forward_touches_exactly_manifest(self):
        for pos in ("sinusoidal", "learned"):
            _, net, res, ps = _build(_mxai(pos=pos))
            manifest_paths = {
                m["path"] for m in composite_network_parameter_manifest(net.name, net, res)
            }
            tr = transformer_network_forward(net, res, ps, IDS)
            assert tr.touched_params == manifest_paths, (
                f"pos={pos}: dead or missing params — "
                f"manifest-touched: {manifest_paths - tr.touched_params}, "
                f"touched-not-manifest: {tr.touched_params - manifest_paths}"
            )


# ---------------------------------------------------------------------------
# param_count contra la fórmula cerrada
# ---------------------------------------------------------------------------

class TestParamCount:
    @pytest.mark.parametrize("layers,dim,ff,pos,length", [
        (2, 8, 16, "sinusoidal", 6),
        (1, 12, 48, "learned", 6),
        (3, 4, 8, "sinusoidal", 6),
    ])
    def test_formula_matches_manifest(self, layers, dim, ff, pos, length):
        heads = 2 if dim % 2 == 0 else 1
        src = _mxai(layers=layers, dim=dim, ff=ff, pos=pos, heads=heads)
        _, net, res, _ = _build(src)
        manifest = composite_network_parameter_manifest(net.name, net, res)
        block_total = sum(
            (m["shape"][0] * m["shape"][1] if len(m["shape"]) == 2 else m["shape"][0])
            for m in manifest if ".enc." in m["path"]
        )
        assert block_total == transformer_block_param_count(layers, dim, ff, length, pos)

    def test_ff_default_uses_4dim(self):
        # FF omitido → 4*dim también en la fórmula
        src = """
PROJECT FFD
SEQUENCE Texto
  length = 6
  vocab_size = 11
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 8
  BLOCK enc TRANSFORMER
    LAYERS 1
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        net = prog.networks[0]
        res = check_composite_network_types(net, {}, {s.name: s for s in prog.sequences})
        manifest = composite_network_parameter_manifest(net.name, net, res)
        block_total = sum(
            (m["shape"][0] * m["shape"][1] if len(m["shape"]) == 2 else m["shape"][0])
            for m in manifest if ".enc." in m["path"]
        )
        assert block_total == transformer_block_param_count(1, 8, 32)


# ---------------------------------------------------------------------------
# PAD determinista + dropout
# ---------------------------------------------------------------------------

class TestDeterminismo:
    def test_pad_deterministic_across_runs_and_fresh_params(self):
        _, net, res, ps = _build(_mxai(), seed=7)
        mask = [True, True, True, False, False, False]
        a = transformer_network_forward(net, res, ps, [1, 2, 3, 0, 0, 0], mask=mask)
        b = transformer_network_forward(net, res, ps, [1, 2, 3, 0, 0, 0], mask=mask)
        assert a.output == b.output
        # ParameterSet regenerado con el mismo seed → misma salida
        _, net2, res2, ps2 = _build(_mxai(), seed=7)
        c = transformer_network_forward(net2, res2, ps2, [1, 2, 3, 0, 0, 0], mask=mask)
        assert a.output == c.output

    def test_sinusoidal_table_is_deterministic(self):
        assert sinusoidal_positional_table(6, 8) == sinusoidal_positional_table(6, 8)

    def test_dropout_identity_in_eval(self):
        _, net, res, ps = _build(_mxai(dropout=0.5))
        a = transformer_network_forward(net, res, ps, IDS, training=False)
        b = transformer_network_forward(net, res, ps, IDS, training=False)
        assert a.output == b.output

    def test_dropout_active_in_training(self):
        _, net, res, ps = _build(_mxai(dropout=0.5))
        eval_out = transformer_network_forward(net, res, ps, IDS, training=False)
        train_out = transformer_network_forward(net, res, ps, IDS, training=True, seed=3)
        assert eval_out.output != train_out.output

    def test_relu_activation_variant(self):
        _, net, res, ps = _build(_mxai(activation="relu"))
        gelu_net = _build(_mxai(activation="gelu"), seed=7)
        tr_relu = transformer_network_forward(net, res, ps, IDS)
        tr_gelu = transformer_network_forward(*gelu_net[1:], IDS)
        assert tr_relu.output != tr_gelu.output


# ---------------------------------------------------------------------------
# Backend tras C2: lowering auditable, entrenamiento cerrado hasta C4
# ---------------------------------------------------------------------------

class TestBackendTrasC2:
    def test_differentiability_verifier_is_green(self):
        """Ronda 3: lowering_supported alimenta el verificador matemático;
        supported conserva el significado histórico de ejecución. El verifier
        queda VERDE sin abrir entrenamiento antes de C4."""
        from matrixai.training import parse_training_text
        from matrixai.training.differentiability import DifferentiabilityVerifier
        prog = parse_text(_mxai())
        report = BackendContractAnalyzer().analyze(prog)
        training = parse_training_text("""
MODEL dummy.mxai

DATASET D
  SOURCE csv("dummy.csv")
  INPUT Texto FROM COLUMNS [t0, t1, t2, t3, t4, t5]
  TARGET clase: ProbabilityMap[NEG, POS]
END

LOSS L
  TYPE cross_entropy
  PREDICTION clase
  TARGET clase
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE N.*
END

RUN
  EPOCHS 2
END
""")
        result = DifferentiabilityVerifier().verify(training, prog, report)
        # Traza los 29 parámetros (1 embedding + 24 del bloque + 4 de la cabeza)
        assert len(result.parameter_paths) == 29
        # VERDE: la matemática/lowering están completas (criterio C2)
        assert result.ok, result.errors

    def test_node_split_lowering_ready_execution_closed(self):
        """La matemática está soportada (verifier verde) pero la EJECUCIÓN no
        (report.ok False) — nada puede entrenar el bloque hasta C4."""
        prog = parse_text(_mxai())
        report = BackendContractAnalyzer().analyze(prog)
        node = next(n for n in report.nodes if n.node == "N")
        assert node.differentiable is True
        assert node.supported is False             # sin trainer hasta C4
        assert node.lowering_supported is True     # matemática/lowering C2
        assert node.lowering_ok is True
        assert report.ok is False
        assert node in report.unsupported_nodes
        assert "TRANSFORMER_BLOQUE C4" in node.reason

        # Contrato público coherente: unsupported_nodes nunca contiene un
        # payload que a la vez diga supported=true, y el resumen lo marca blocked.
        payload = report.to_dict()
        blocked = next(n for n in payload["unsupported_nodes"] if n["node"] == "N")
        assert blocked["supported"] is False
        assert blocked["lowering_supported"] is True
        assert "N (composite_network [composite_network]): blocked" in report.summary()

    def test_parameter_set_builds_and_validates(self):
        from matrixai.parameters.network_params import (
            validate_composite_network_parameter_set,
        )
        _, net, res, ps = _build(_mxai())
        result = validate_composite_network_parameter_set(
            net, res, ps, "mxai_c2test"
        )
        assert result.ok, result.errors


class TestReAuditoriaC2:
    """Regresiones de la re-auditoría de Roberto sobre C2 (2026-07-10)."""

    def _dirty(self):
        """DIM=8, HEADS=3 → typecheck sucio pero con resolved_dim=8 puesto."""
        prog = parse_text(_mxai(heads=3))
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {}, {s.name: s for s in prog.sequences}
        )
        assert not res.ok
        assert res.resolved_transformer_blocks[0].resolved_dim == 8
        return prog, net, res

    # [ALTA] POOL cls con padding en posición 0
    def test_cls_with_masked_position_0_rejected(self):
        _, net, res, ps = _build(_mxai(pool="cls"))
        mask = [False, True, True, True, False, False]
        with pytest.raises(TransformerForwardError, match="POOL cls requires position 0"):
            transformer_network_forward(net, res, ps, [0, 2, 3, 4, 0, 0], mask=mask)

    def test_cls_with_pad_id_at_position_0_rejected(self):
        _, net, res, ps = _build(_mxai(pool="cls"))
        with pytest.raises(TransformerForwardError, match="POOL cls requires position 0"):
            transformer_network_forward(net, res, ps, [0, 2, 3, 4, 0, 0], pad_id=0)

    def test_cls_with_real_position_0_still_ok(self):
        _, net, res, ps = _build(_mxai(pool="cls"))
        tr = transformer_network_forward(
            net, res, ps, [1, 2, 3, 0, 0, 0],
            mask=[True, True, True, False, False, False],
        )
        assert len(tr.output) == 2

    # [ALTA] artefactos con typecheck sucio
    def test_manifest_rejects_dirty_typecheck(self):
        _, net, res = self._dirty()
        with pytest.raises(ValueError, match="CLEAN"):
            composite_network_parameter_manifest(net.name, net, res)

    def test_parameter_set_build_rejects_dirty_typecheck(self):
        _, net, res = self._dirty()
        with pytest.raises(ValueError, match="CLEAN"):
            build_composite_network_parameter_set(net, res, "mxai_dirty", seed=7)

    def test_backend_reports_real_typecheck_error_not_differentiable(self):
        prog, net, res = self._dirty()
        report = BackendContractAnalyzer().analyze(prog)
        node = next(n for n in report.nodes if n.node == "N")
        assert node.supported is False
        assert node.differentiable is False
        assert "does not divide DIM" in node.reason
        assert report.trainable_parameters == []
        # y el layer manifest tampoco publica el lowering como válido
        block_entries = [
            e for e in report.layer_manifest if e.get("layer_type") == "TransformerBlock"
        ]
        assert len(block_entries) == 1
        assert block_entries[0]["differentiable"] is False
        assert "typecheck failed" in block_entries[0]["reason"]

    # [MEDIA] unidad de trainable_param_count restaurada
    def test_trainable_param_count_is_tensor_count(self):
        prog = parse_text(_mxai())
        report = BackendContractAnalyzer().analyze(prog)
        entry = next(
            e for e in report.layer_manifest if e.get("layer_type") == "TransformerBlock"
        )
        assert entry["trainable_param_count"] == len(entry["parameters"]) == 2 * 12
        assert entry["param_count"] == transformer_block_param_count(2, 8, 16)

    # [BAJA] matvec estricta
    def test_malformed_weight_matrix_rejected(self):
        _, net, res, ps = _build(_mxai())
        key = "N.enc.layer_0.attention.Wq"
        bad = [row + [999999.0] for row in ps.parameters[key]["values"]]
        ps.parameters[key]["values"] = bad
        with pytest.raises(TransformerForwardError, match="malformed transformer ParameterSet"):
            transformer_network_forward(net, res, ps, IDS)

    def test_weight_matrix_with_extra_row_rejected(self):
        """Variante de ronda 3: una fila extra tiene ancho correcto, pero Wq
        produciría una componente adicional que el split de cabezas ignoraba."""
        _, net, res, ps = _build(_mxai())
        key = "N.enc.layer_0.attention.Wq"
        ps.parameters[key]["values"].append([999999.0] * 8)
        with pytest.raises(TransformerForwardError, match="expected values shape"):
            transformer_network_forward(net, res, ps, IDS)

    def test_bias_with_extra_value_rejected(self):
        """Los zip de FFN también truncarían biases con elementos sobrantes."""
        _, net, res, ps = _build(_mxai())
        ps.parameters["N.enc.layer_0.ffn.b1"]["values"].append(999999.0)
        with pytest.raises(TransformerForwardError, match="expected values shape"):
            transformer_network_forward(net, res, ps, IDS)

    # Mejoras: pool kind en el hash de esquema + cap de tamaño
    def test_pool_kind_changes_schema_hash(self):
        _, net_m, res_m, _ = _build(_mxai(pool="mean"))
        _, net_c, res_c, _ = _build(_mxai(pool="cls"))
        h_mean = composite_network_parameter_schema_hash(net_m.name, net_m, res_m)
        h_cls = composite_network_parameter_schema_hash(net_c.name, net_c, res_c)
        assert h_mean != h_cls

    def test_reference_forward_size_cap(self):
        # Embedding dominante: el bloque solo tiene ~600 params, pero vocab·dim
        # ya supera 1M. La primera versión del cap contaba solo el bloque.
        src = _mxai(length=6, vocab=200_000, dim=8, layers=1, heads=2, ff=16)
        prog = parse_text(src)
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {}, {s.name: s for s in prog.sequences}
        )
        manifest = composite_network_parameter_manifest(net.name, net, res)
        assert manifest_scalar_count(manifest) > 1_000_000

        # Metadata-only sigue permitido para el futuro camino torch.
        ps_meta = build_composite_network_parameter_set(
            net, res, "mxai_large", seed=7, with_values=False
        )
        with pytest.raises(TransformerForwardError, match="mini test shapes"):
            transformer_network_forward(net, res, ps_meta, IDS)

    def test_parameter_builder_caps_before_materializing_values(self):
        src = _mxai(length=6, vocab=200_000, dim=8, layers=1, heads=2, ff=16)
        prog = parse_text(src)
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {}, {s.name: s for s in prog.sequences}
        )
        with pytest.raises(ValueError, match="with_values=False"):
            build_composite_network_parameter_set(net, res, "mxai_large", seed=7)


class TestGuardsDelForward:
    """Auditoría propia de C2: el forward falla cerrado ante estados sucios."""

    def test_dirty_type_result_rejected(self):
        """HEADS no divisor deja resolved_dim puesto — sin este guard el forward
        habría atendido dim//heads truncado silenciosamente."""
        src = _mxai(heads=3)  # 3 no divide 8
        prog = parse_text(src)
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {}, {s.name: s for s in prog.sequences}
        )
        assert not res.ok  # el typecheck ya lo rechaza…
        ps_src = _mxai(heads=2)
        _, _, res_ok, ps = _build(ps_src)
        # …y el forward NO acepta el type_result sucio aunque tenga resolved blocks
        with pytest.raises(TransformerForwardError, match="CLEAN type_result"):
            transformer_network_forward(net, res, ps, IDS)

    def test_unresolved_type_result_rejected(self):
        """Un type_result de OTRA red (sin bloques resueltos) también se rechaza."""
        _, net, res, ps = _build(_mxai())
        import dataclasses
        empty = dataclasses.replace(res, resolved_transformer_blocks=[])
        with pytest.raises(TransformerForwardError, match="exactly one"):
            transformer_network_forward(net, empty, ps, IDS)


class TestConsumidoresFallanCerrado:
    """Auditoría propia de C2: los consumidores que NO saben del bloque deben
    fallar cerrado, no construir artefactos que lo omitan silenciosamente."""

    def test_composite_forward_stdlib_rejects_transformer(self):
        from matrixai.forward.composite_forward import (
            CompositeForwardError,
            composite_forward,
        )
        _, net, res, ps = _build(_mxai())
        with pytest.raises(CompositeForwardError, match="BLOCK TRANSFORMER"):
            composite_forward(net, ps, {"Texto": IDS})

    def test_torch_module_builder_rejects_transformer(self):
        pytest.importorskip("torch")
        from matrixai.forward.composite_torch import (
            CompositeTorchError,
            composite_network_to_torch_module,
        )
        _, net, res, ps = _build(_mxai())
        with pytest.raises(CompositeTorchError, match="BLOCK TRANSFORMER"):
            composite_network_to_torch_module(net, ps)

    def test_onnx_export_rejects_transformer(self):
        pytest.importorskip("onnx")
        import tempfile
        from pathlib import Path
        from matrixai.export.onnx_exporter import OnnxExportError, export_onnx
        from matrixai.parameters.store import program_hash
        prog = parse_text(_mxai())
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {v.name: v for v in prog.vectors}, {s.name: s for s in prog.sequences}
        )
        ps = build_composite_network_parameter_set(net, res, program_hash(prog), seed=7)
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(OnnxExportError, match="BLOCK TRANSFORMER"):
                export_onnx(prog, ps, Path(td) / "out.onnx")

    def test_export_validator_sees_sequence_map(self):
        """validate_export_parameter_set llamaba al typecheck sin sequence_map
        (misma clase de bug que el ALTA-3 de C1): un programa de secuencia
        fallaba con 'INPUT is not a declared VECTOR' engañoso."""
        from matrixai.export.onnx_exporter import validate_export_parameter_set
        from matrixai.parameters.store import program_hash
        prog = parse_text(_mxai())
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {v.name: v for v in prog.vectors}, {s.name: s for s in prog.sequences}
        )
        ps = build_composite_network_parameter_set(net, res, program_hash(prog), seed=7)
        result = validate_export_parameter_set(prog, ps)
        # El ParameterSet es coherente con el manifest → el validador debe
        # aceptar (el bloqueo del EXPORT es del guard del exporter, no un
        # typecheck roto por el sequence_map ausente)
        assert result.ok, result.errors
