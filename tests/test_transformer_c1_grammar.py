# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""TRANSFORMER_BLOQUE C1 — gramática + IR + typecheck del BLOCK TRANSFORMER.

Cubre la especificación exacta del contrato 51 §C1: parse válido mínimo y
extendido, cada error de la gramática (LAYERS ausente, HEADS no divisor con
mensaje útil, POOL ausente, dos bloques, clave desconocida, EMBEDDING FROM
sequence con VOCAB explícito contradictorio) y los shapes por etapa
[L] → [L,dim] → [L,dim] → POOL → [dim].
"""
from __future__ import annotations

import pytest

from matrixai.parser.parser import MatrixAIParseError, parse_text
from matrixai.types import check_composite_network_types, check_program_types


# ---------------------------------------------------------------------------
# Fixtures — .mxai builders
# ---------------------------------------------------------------------------

def _mxai(block_body: str = "LAYERS 4", *, embedding: str = "EMBEDDING tok FROM Texto DIM 128",
          pool: str = "POOL mean", extra_after_block: str = "") -> str:
    """Minimal valid transformer program with replaceable pieces."""
    return f"""
PROJECT TransformerC1

SEQUENCE Texto
  length = 64
  vocab_size = 30000
END

NETWORK ClasificadorTexto
  INPUT Texto
  {embedding}
  BLOCK encoder TRANSFORMER
    {block_body}
  END
  {pool}
  LAYER Dense units=64 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END

GRAPH
  Texto -> ClasificadorTexto
END
"""


def _typecheck(src: str):
    prog = parse_text(src)
    net = prog.networks[0]
    return check_composite_network_types(
        net,
        {v.name: v for v in prog.vectors},
        {s.name: s for s in prog.sequences},
    )


# ---------------------------------------------------------------------------
# Parse válido — mínimo y extendido
# ---------------------------------------------------------------------------

class TestParseValid:
    def test_minimal_block_defaults(self):
        prog = parse_text(_mxai("LAYERS 2"))
        net = prog.networks[0]
        assert net.kind == "composite_network"
        assert len(net.transformer_blocks) == 1
        tb = net.transformer_blocks[0]
        assert tb.name == "encoder"
        assert tb.layers == 2
        assert tb.heads == 4              # default
        assert tb.ff == 0                 # 0 = 4*dim, se resuelve en typecheck
        assert tb.dropout == 0.0          # default
        assert tb.activation == "gelu"    # default
        assert tb.pos == "sinusoidal"     # default

    def test_extended_block_all_keys(self):
        body = "LAYERS 4\n    HEADS 8\n    FF 512\n    DROPOUT 0.1\n    ACTIVATION relu\n    POS learned"
        prog = parse_text(_mxai(body))
        tb = prog.networks[0].transformer_blocks[0]
        assert (tb.layers, tb.heads, tb.ff, tb.dropout) == (4, 8, 512, 0.1)
        assert tb.activation == "relu"
        assert tb.pos == "learned"

    def test_embedding_from_sequence_without_vocab(self):
        prog = parse_text(_mxai())
        emb = prog.networks[0].embeddings[0]
        assert emb.vocab == 0  # sentinel: heredar de la SEQUENCE
        assert emb.dim == 128
        assert emb.source == "Texto"

    def test_block_position_records_textual_order(self):
        prog = parse_text(_mxai())
        tb = prog.networks[0].transformer_blocks[0]
        assert tb.position == 0  # ningún top_layer antes del bloque

    def test_program_to_dict_serializes_transformer_block(self):
        prog = parse_text(_mxai("LAYERS 3\n    HEADS 2"))
        data = prog.to_dict()
        nets = data["networks"]
        tbs = nets[0]["transformer_blocks"]
        assert tbs[0]["name"] == "encoder"
        assert tbs[0]["layers"] == 3
        assert tbs[0]["heads"] == 2
        assert tbs[0]["pos"] == "sinusoidal"


# ---------------------------------------------------------------------------
# Errores de gramática (parse)
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_layers_missing(self):
        with pytest.raises(MatrixAIParseError, match="LAYERS is required"):
            parse_text(_mxai("HEADS 4"))

    def test_layers_below_one(self):
        with pytest.raises(MatrixAIParseError, match="LAYERS must be >= 1"):
            parse_text(_mxai("LAYERS 0"))

    def test_layers_not_integer(self):
        with pytest.raises(MatrixAIParseError, match="LAYERS must be an integer"):
            parse_text(_mxai("LAYERS dos"))

    def test_unknown_key_lists_valid_keys(self):
        with pytest.raises(MatrixAIParseError, match="unknown key 'WARMUP'.*LAYERS.*HEADS.*FF.*DROPOUT.*ACTIVATION.*POS"):
            parse_text(_mxai("LAYERS 2\n    WARMUP 100"))

    def test_duplicate_key(self):
        with pytest.raises(MatrixAIParseError, match="duplicate key 'LAYERS'"):
            parse_text(_mxai("LAYERS 2\n    LAYERS 4"))

    def test_key_without_value(self):
        with pytest.raises(MatrixAIParseError, match="HEADS requires a value"):
            parse_text(_mxai("LAYERS 2\n    HEADS"))

    def test_dropout_out_of_range(self):
        with pytest.raises(MatrixAIParseError, match=r"DROPOUT must be in \[0, 1\)"):
            parse_text(_mxai("LAYERS 2\n    DROPOUT 1.0"))

    def test_dropout_negative(self):
        with pytest.raises(MatrixAIParseError, match=r"DROPOUT must be in \[0, 1\)"):
            parse_text(_mxai("LAYERS 2\n    DROPOUT -0.1"))

    def test_activation_invalid(self):
        with pytest.raises(MatrixAIParseError, match="ACTIVATION must be one of"):
            parse_text(_mxai("LAYERS 2\n    ACTIVATION tanh"))

    def test_pos_invalid(self):
        with pytest.raises(MatrixAIParseError, match="POS must be one of"):
            parse_text(_mxai("LAYERS 2\n    POS rotary"))

    def test_missing_end(self):
        src = _mxai().replace("  END\n  POOL mean", "  POOL mean")
        with pytest.raises(MatrixAIParseError):
            parse_text(src)


# ---------------------------------------------------------------------------
# Errores de typecheck
# ---------------------------------------------------------------------------

class TestTypecheckErrors:
    def test_heads_not_divisor_lists_valid_divisors(self):
        res = _typecheck(_mxai("LAYERS 2\n    HEADS 5"))
        assert not res.ok
        msg = "; ".join(res.errors)
        assert "HEADS 5 does not divide DIM 128" in msg
        # el mensaje enumera los divisores válidos de 128
        assert "1, 2, 4, 8, 16, 32, 64, 128" in msg

    def test_pool_missing(self):
        res = _typecheck(_mxai(pool=""))
        assert not res.ok
        msg = "; ".join(res.errors)
        # Dense recibe [L, dim] → error accionable pidiendo POOL
        assert "POOL" in msg
        assert "mean" in msg and "cls" in msg

    def test_pool_missing_without_dense(self):
        src = _mxai(pool="").replace(
            "  LAYER Dense units=64 activation=relu\n  LAYER Dense units=2 activation=softmax\n", ""
        )
        res = _typecheck(src)
        assert not res.ok
        assert any("missing POOL after BLOCK TRANSFORMER" in e for e in res.errors)

    def test_pool_max_over_sequence_rejected(self):
        res = _typecheck(_mxai(pool="POOL max"))
        assert not res.ok
        assert any("POOL max over a sequence" in e and "mean" in e for e in res.errors)

    def test_two_transformer_blocks(self):
        src = _mxai().replace(
            "  POOL mean",
            "  BLOCK encoder2 TRANSFORMER\n    LAYERS 2\n  END\n  POOL mean",
        )
        res = _typecheck(src)
        assert not res.ok
        assert any("only one BLOCK TRANSFORMER" in e for e in res.errors)

    def test_transformer_mixed_with_classic_block(self):
        src = _mxai().replace(
            "  POOL mean",
            "  POOL mean\n  BLOCK mlp\n    LAYER Dense units=128 activation=relu\n  END",
        )
        res = _typecheck(src)
        assert not res.ok
        assert any("cannot be mixed with a classic BLOCK" in e for e in res.errors)

    def test_embedding_vocab_contradicts_sequence(self):
        res = _typecheck(_mxai(embedding="EMBEDDING tok FROM Texto VOCAB 5000 DIM 128"))
        assert not res.ok
        msg = "; ".join(res.errors)
        assert "VOCAB 5000" in msg and "vocab_size 30000" in msg
        assert "omit VOCAB" in msg

    def test_embedding_vocab_matching_sequence_ok(self):
        res = _typecheck(_mxai(embedding="EMBEDDING tok FROM Texto VOCAB 30000 DIM 128"))
        assert res.ok, res.errors

    def test_embedding_without_vocab_from_tabular_field_rejected(self):
        src = """
PROJECT Tab

VECTOR Product[2]
  category_id: Integer[0, 100]
  price: Scalar
END

NETWORK Net
  INPUT Product
  EMBEDDING cat FROM category_id DIM 8
  CONCAT [cat, price] -> features
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[a, b]
END

GRAPH
  Product -> Net
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(
            prog.networks[0], {v.name: v for v in prog.vectors}, {}
        )
        assert not res.ok
        assert any("requires VOCAB" in e for e in res.errors)

    def test_transformer_with_vector_input_rejected(self):
        src = """
PROJECT Vec

VECTOR Datos[2]
  a: Scalar
  b: Scalar
END

NETWORK Net
  INPUT Datos
  BLOCK encoder TRANSFORMER
    LAYERS 2
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[x, y]
END

GRAPH
  Datos -> Net
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(
            prog.networks[0], {v.name: v for v in prog.vectors}, {}
        )
        assert not res.ok
        assert any("requires INPUT to be a SEQUENCE" in e for e in res.errors)

    def test_transformer_without_embedding_before_block(self):
        res = _typecheck(_mxai(embedding=""))
        assert not res.ok
        assert any("EMBEDDING" in e and "before the block" in e for e in res.errors)

    def test_unknown_input_mentions_sequence_when_sequences_exist(self):
        src = _mxai().replace("INPUT Texto", "INPUT NoExiste")
        res = _typecheck(src)
        assert not res.ok
        assert "not a declared VECTOR or SEQUENCE" in res.errors[0]

    def test_unknown_input_keeps_vector_message_without_sequences(self):
        # Retro-compat: el mensaje clásico no cambia para redes tabulares
        src = """
PROJECT Vec

VECTOR Datos[1]
  a: Scalar
END

NETWORK Net
  INPUT Otro
  EMBEDDING e FROM a VOCAB 5 DIM 2
  CONCAT [e] -> f
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[x, y]
END

GRAPH
  Datos -> Net
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(
            prog.networks[0], {v.name: v for v in prog.vectors}, {}
        )
        assert not res.ok
        assert res.errors[0].endswith("is not a declared VECTOR")


# ---------------------------------------------------------------------------
# Shapes por etapa e IR resuelto
# ---------------------------------------------------------------------------

class TestShapes:
    def test_stage_shapes_L_Ld_Ld_pool_d(self):
        res = _typecheck(_mxai("LAYERS 4\n    HEADS 4\n    FF 512"))
        assert res.ok, res.errors
        tb = res.resolved_transformer_blocks[0]
        assert tb.input_shape == [64, 128]
        assert tb.output_shape == [64, 128]
        assert tb.resolved_dim == 128
        assert tb.resolved_ff == 512
        # POOL reduce [64, 128] → [128]; Dense siguen la cadena
        pool = res.resolved_layers[0]
        assert (pool.layer_type, pool.input_shape, pool.output_shape) == ("Pool", [64, 128], [128])
        d1, d2 = res.resolved_layers[1], res.resolved_layers[2]
        assert (d1.input_shape, d1.output_shape) == ([128], [64])
        assert (d2.input_shape, d2.output_shape) == ([64], [2])

    def test_ff_default_is_4_dim(self):
        res = _typecheck(_mxai("LAYERS 2"))
        assert res.ok, res.errors
        assert res.resolved_transformer_blocks[0].resolved_ff == 4 * 128

    def test_named_shapes_sequence_and_embedding(self):
        res = _typecheck(_mxai())
        assert res.named_shapes["Texto"] == [64]
        assert res.named_shapes["tok"] == [64, 128]

    def test_pool_cls_also_valid(self):
        res = _typecheck(_mxai(pool="POOL cls"))
        assert res.ok, res.errors

    def test_heads_one_valid(self):
        res = _typecheck(_mxai("LAYERS 2\n    HEADS 1"))
        assert res.ok, res.errors

    def test_program_level_typecheck_registers_output_type(self):
        prog = parse_text(_mxai())
        result = check_program_types(prog)
        assert result.ok, result.errors
        assert "clase" in result.symbols
        assert result.symbols["clase"].name == "ProbabilityMap"


# ---------------------------------------------------------------------------
# Retro-compat: el composite tabular clásico no cambia
# ---------------------------------------------------------------------------

class TestRetroCompat:
    _TABULAR = """
PROJECT Retro

VECTOR Product[3]
  category_id: Integer[0, 100]
  price: Scalar
  weight: Scalar
END

NETWORK CategoryNet
  INPUT Product
  EMBEDDING cat_emb FROM category_id VOCAB 100 DIM 8
  CONCAT [cat_emb, price, weight] -> features
  LAYER Dense units=16 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT label: ProbabilityMap[a, b, c]
END

GRAPH
  Product -> CategoryNet
END
"""

    def test_tabular_composite_still_ok(self):
        prog = parse_text(self._TABULAR)
        net = prog.networks[0]
        assert net.transformer_blocks == []
        res = check_composite_network_types(
            net, {v.name: v for v in prog.vectors}, {s.name: s for s in prog.sequences}
        )
        assert res.ok, res.errors
        assert res.resolved_transformer_blocks == []

    def test_tabular_composite_ok_without_sequences_arg(self):
        # Los call sites antiguos (sin sequences_by_name) siguen funcionando
        prog = parse_text(self._TABULAR)
        res = check_composite_network_types(
            prog.networks[0], {v.name: v for v in prog.vectors}
        )
        assert res.ok, res.errors
