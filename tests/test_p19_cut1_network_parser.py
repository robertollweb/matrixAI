"""P19 C1 — Parser e IR: EMBEDDING, CONCAT, BLOCK, RESIDUAL, POOL, Dropout, LayerNorm, Activation, Reshape."""
from __future__ import annotations

import pytest

from matrixai.parser.parser import MatrixAIParseError


# ---------------------------------------------------------------------------
# Test fixtures — minimal valid .mxai strings
# ---------------------------------------------------------------------------

_MXAI_EMBEDDING_BASIC = """
PROJECT EmbTest

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

_MXAI_BLOCK_RESIDUAL_PREVIOUS = """
PROJECT ResTest

VECTOR H[2]
  x1: Scalar
  x2: Scalar
END

NETWORK ResNet
  INPUT H
  LAYER Dense units=8 activation=relu
  BLOCK res1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  H -> ResNet
END
"""

_MXAI_BLOCK_RESIDUAL_NAMED = """
PROJECT ResNamedTest

VECTOR H[2]
  x1: Scalar
  x2: Scalar
END

NETWORK ResNet
  INPUT H
  EMBEDDING emb FROM x1 VOCAB 10 DIM 4
  CONCAT [emb, x2] -> features
  BLOCK res1
    LAYER Dense units=5 activation=relu
    LAYER Dropout rate=0.1
    RESIDUAL FROM features
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[pos, neg]
END

GRAPH
  H -> ResNet
END
"""

_MXAI_POOL = """
PROJECT PoolTest

VECTOR V[2]
  a: Scalar
  b: Scalar
END

NETWORK PoolNet
  INPUT V
  LAYER Dense units=8 activation=relu
  POOL mean
  LAYER Dense units=3 activation=softmax
  OUTPUT label: ProbabilityMap[x, y, z]
END

GRAPH
  V -> PoolNet
END
"""

_MXAI_ALL_LAYERS = """
PROJECT AllLayersTest

VECTOR V[2]
  a: Scalar
  b: Scalar
END

NETWORK AllNet
  INPUT V
  LAYER Dense units=16 activation=relu
  LAYER LayerNorm
  LAYER Dropout rate=0.2
  LAYER Activation kind=relu
  LAYER Dense units=4 activation=softmax
  OUTPUT label: ProbabilityMap[a, b, c, d]
END

GRAPH
  V -> AllNet
END
"""

_MXAI_RESHAPE = """
PROJECT ReshapeTest

VECTOR V[8]
  f1: Scalar
  f2: Scalar
  f3: Scalar
  f4: Scalar
  f5: Scalar
  f6: Scalar
  f7: Scalar
  f8: Scalar
END

NETWORK ReshapeNet
  INPUT V
  LAYER Dense units=16 activation=relu
  LAYER Reshape target=[4, 4]
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[pos, neg]
END

GRAPH
  V -> ReshapeNet
END
"""

_MXAI_P18_DENSE_ONLY = """
PROJECT P18Test

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK DenseNet
  INPUT V
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> DenseNet
END
"""


# ---------------------------------------------------------------------------
# TestNetworkP19Basics — EMBEDDING, CONCAT, BLOCK accepted
# ---------------------------------------------------------------------------

class TestNetworkP19Basics:
    def test_parser_accepts_embedding_declaration(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        net = program.networks[0]
        assert len(net.embeddings) == 1
        emb = net.embeddings[0]
        assert emb.name == "cat_emb"
        assert emb.source == "category_id"
        assert emb.vocab == 100
        assert emb.dim == 8

    def test_parser_accepts_concat_with_multiple_sources(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        net = program.networks[0]
        assert len(net.concats) == 1
        concat = net.concats[0]
        assert concat.name == "features"
        assert "cat_emb" in concat.sources
        assert "price" in concat.sources

    def test_parser_accepts_block_with_residual_from_previous(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        net = program.networks[0]
        assert len(net.blocks) == 1
        block = net.blocks[0]
        assert block.name == "res1"
        assert block.residual_from == "PREVIOUS"

    def test_parser_accepts_block_with_residual_from_named(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_BLOCK_RESIDUAL_NAMED)
        net = program.networks[0]
        block = net.blocks[0]
        assert block.residual_from == "features"

    def test_parser_accepts_pool_mean_at_top_level(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_POOL)
        net = program.networks[0]
        pool_layers = [l for l in net.top_layers if l.layer_type == "Pool"]
        assert len(pool_layers) == 1
        assert pool_layers[0].pool_kind == "mean"


# ---------------------------------------------------------------------------
# TestNetworkP19Layers — individual P19 layer types
# ---------------------------------------------------------------------------

class TestNetworkP19Layers:
    def test_parser_accepts_dropout_with_valid_rate(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_ALL_LAYERS)
        net = program.networks[0]
        dropout = next(l for l in net.top_layers if l.layer_type == "Dropout")
        assert dropout.rate == pytest.approx(0.2)

    def test_parser_accepts_layernorm(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_ALL_LAYERS)
        net = program.networks[0]
        ln = next(l for l in net.top_layers if l.layer_type == "LayerNorm")
        assert ln.layer_type == "LayerNorm"

    def test_parser_accepts_standalone_activation_relu(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_ALL_LAYERS)
        net = program.networks[0]
        act = next(l for l in net.top_layers if l.layer_type == "Activation")
        assert act.activation_kind == "relu"

    def test_parser_accepts_activation_gelu(self):
        from matrixai.parser.parser import parse_text
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu\n  LAYER Dense units=1 activation=linear",
            "LAYER Dense units=4 activation=relu\n  LAYER Activation kind=gelu\n  LAYER Dense units=1 activation=linear"
        )
        program = parse_text(src)
        net = program.networks[0]
        assert net.kind == "composite_network"
        act = next(l for l in net.top_layers if l.layer_type == "Activation")
        assert act.activation_kind == "gelu"

    def test_parser_accepts_reshape_with_valid_target(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_RESHAPE)
        net = program.networks[0]
        reshape = next(l for l in net.top_layers if l.layer_type == "Reshape")
        assert reshape.target_shape == [4, 4]


# ---------------------------------------------------------------------------
# TestNetworkP19KindDetection — kind field correctly set
# ---------------------------------------------------------------------------

class TestNetworkP19KindDetection:
    def test_p18_dense_only_is_kind_dense_network(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_P18_DENSE_ONLY)
        net = program.networks[0]
        assert net.kind == "dense_network"

    def test_p18_layers_populated_in_dense_network(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_P18_DENSE_ONLY)
        net = program.networks[0]
        assert len(net.layers) == 2
        assert net.top_layers == []

    def test_embedding_promotes_to_composite_network(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        net = program.networks[0]
        assert net.kind == "composite_network"

    def test_block_promotes_to_composite_network(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        net = program.networks[0]
        assert net.kind == "composite_network"

    def test_pool_promotes_to_composite_network(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_POOL)
        net = program.networks[0]
        assert net.kind == "composite_network"

    def test_composite_network_layers_field_is_empty(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        net = program.networks[0]
        assert net.layers == []

    def test_graph_node_type_is_composite_network(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        assert program.graph.node_types.get("CategoryNet") == "composite_network"


# ---------------------------------------------------------------------------
# TestNetworkP19Errors — validation errors
# ---------------------------------------------------------------------------

class TestNetworkP19Errors:
    def test_rejects_embedding_with_zero_vocab(self):
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu",
            "EMBEDDING emb FROM x1 VOCAB 0 DIM 8\n  LAYER Dense units=4 activation=relu"
        )
        with pytest.raises(MatrixAIParseError, match="VOCAB must be > 0"):
            from matrixai.parser.parser import parse_text
            parse_text(src)

    def test_rejects_embedding_with_zero_dim(self):
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu",
            "EMBEDDING emb FROM x1 VOCAB 10 DIM 0\n  LAYER Dense units=4 activation=relu"
        )
        with pytest.raises(MatrixAIParseError, match="DIM must be > 0"):
            from matrixai.parser.parser import parse_text
            parse_text(src)

    def test_rejects_dropout_rate_ge_1(self):
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu",
            "LAYER Dense units=4 activation=relu\n  LAYER Dropout rate=1.0"
        )
        with pytest.raises(MatrixAIParseError, match="rate must be in"):
            from matrixai.parser.parser import parse_text
            parse_text(src)

    def test_rejects_dropout_rate_le_0(self):
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu",
            "LAYER Dense units=4 activation=relu\n  LAYER Dropout rate=0.0"
        )
        with pytest.raises(MatrixAIParseError, match="rate must be in"):
            from matrixai.parser.parser import parse_text
            parse_text(src)

    def test_rejects_empty_block(self):
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu",
            "BLOCK empty_block\n  END\n  LAYER Dense units=4 activation=relu"
        )
        with pytest.raises(MatrixAIParseError, match="empty"):
            from matrixai.parser.parser import parse_text
            parse_text(src)

    def test_rejects_nested_blocks(self):
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu\n  LAYER Dense units=1 activation=linear",
            "BLOCK outer\n    BLOCK inner\n      LAYER Dense units=4 activation=relu\n    END\n  END\n  LAYER Dense units=1 activation=linear"
        )
        with pytest.raises(MatrixAIParseError, match="nested"):
            from matrixai.parser.parser import parse_text
            parse_text(src)

    def test_rejects_unknown_activation_for_standalone_activation_layer(self):
        src = _MXAI_P18_DENSE_ONLY.replace(
            "LAYER Dense units=4 activation=relu",
            "LAYER Dense units=4 activation=relu\n  LAYER Activation kind=swish"
        )
        with pytest.raises(MatrixAIParseError, match="kind 'swish' not allowed"):
            from matrixai.parser.parser import parse_text
            parse_text(src)


# ---------------------------------------------------------------------------
# TestNetworkP19IR — IR structure and to_dict
# ---------------------------------------------------------------------------

class TestNetworkP19IR:
    def test_embedding_spec_has_correct_fields(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        net = program.networks[0]
        emb = net.embeddings[0]
        assert emb.name == "cat_emb"
        assert emb.source == "category_id"
        assert emb.vocab == 100
        assert emb.dim == 8

    def test_block_spec_has_layers_and_residual_from(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        net = program.networks[0]
        block = net.blocks[0]
        assert len(block.layers) >= 2
        assert block.residual_from == "PREVIOUS"
        ln_layers = [l for l in block.layers if l.layer_type == "LayerNorm"]
        assert ln_layers, "LayerNorm should be in block"

    def test_composite_network_to_dict_includes_embeddings(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        d = program.to_dict()
        net_dict = d["networks"][0]
        assert net_dict["kind"] == "composite_network"
        assert "embeddings" in net_dict
        assert net_dict["embeddings"][0]["name"] == "cat_emb"
        assert net_dict["embeddings"][0]["vocab"] == 100

    def test_composite_network_to_dict_includes_blocks(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        d = program.to_dict()
        net_dict = d["networks"][0]
        assert "blocks" in net_dict
        assert net_dict["blocks"][0]["name"] == "res1"
        assert net_dict["blocks"][0]["residual_from"] == "PREVIOUS"

    def test_composite_layer_to_dict_dense_has_hierarchical_param_paths(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_EMBEDDING_BASIC)
        d = program.to_dict()
        net_dict = d["networks"][0]
        dense_layers = net_dict.get("layers", [])
        for layer in dense_layers:
            if layer["type"] == "Dense":
                assert "parameters" in layer
                assert ".L" in layer["parameters"]["weights"]

    def test_block_dense_layer_has_hierarchical_param_path(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        d = program.to_dict()
        net_dict = d["networks"][0]
        block_dict = net_dict["blocks"][0]
        dense_in_block = next(l for l in block_dict["layers"] if l["type"] == "Dense")
        assert "ResNet.res1" in dense_in_block["parameters"]["weights"]

    def test_p18_dense_network_to_dict_unchanged(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_P18_DENSE_ONLY)
        d = program.to_dict()
        net_dict = d["networks"][0]
        assert net_dict["kind"] == "dense_network"
        for layer in net_dict["layers"]:
            assert layer["parameters"]["weights"].startswith("DenseNet.W")
