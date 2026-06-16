"""P19 C2 — Type system and shape inference for composite networks."""
from __future__ import annotations

import pytest

from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types, check_network_types


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_net(src: str):
    program = parse_text(src)
    return program.networks[0], {v.name: v for v in program.vectors}


# ---------------------------------------------------------------------------
# Fixtures
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

# --- Error fixtures ---

_MXAI_RESIDUAL_SHAPE_MISMATCH_PREVIOUS = """
PROJECT RMTest

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK ResNet
  INPUT V
  LAYER Dense units=8 activation=relu
  BLOCK b1
    LAYER Dense units=16 activation=relu
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> ResNet
END
"""

_MXAI_RESIDUAL_SHAPE_MISMATCH_NAMED = """
PROJECT RMTest2

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK ResNet
  INPUT V
  EMBEDDING emb FROM x1 VOCAB 10 DIM 4
  CONCAT [emb, x2] -> features
  BLOCK b1
    LAYER Dense units=8 activation=relu
    RESIDUAL FROM features
  END
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> ResNet
END
"""

_MXAI_RESHAPE_BAD_PRODUCT = """
PROJECT RPTest

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
  LAYER Reshape target=[3, 4]
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[pos, neg]
END

GRAPH
  V -> ReshapeNet
END
"""


# ---------------------------------------------------------------------------
# TestCompositeNetworkShapeEmbeddingConcat
# ---------------------------------------------------------------------------

class TestCompositeNetworkShapeEmbeddingConcat:
    def test_composite_network_check_ok_for_valid_model(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        result = check_composite_network_types(net, vbn)
        assert result.ok, result.errors

    def test_embedding_named_shape_is_dim(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        result = check_composite_network_types(net, vbn)
        assert result.named_shapes.get("cat_emb") == [8]

    def test_scalar_field_named_shape_is_one(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        result = check_composite_network_types(net, vbn)
        assert result.named_shapes.get("price") == [1]
        assert result.named_shapes.get("weight") == [1]

    def test_concat_output_shape_is_sum_of_source_dims(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        result = check_composite_network_types(net, vbn)
        # cat_emb[8] + price[1] + weight[1] = 10
        assert result.named_shapes.get("features") == [10]

    def test_first_top_layer_input_shape_equals_concat_output(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        result = check_composite_network_types(net, vbn)
        dense = next(l for l in result.resolved_layers if l.layer_type == "Dense")
        assert dense.input_shape == [10]
        assert dense.output_shape == [16]


# ---------------------------------------------------------------------------
# TestCompositeNetworkLayerShapes
# ---------------------------------------------------------------------------

class TestCompositeNetworkLayerShapes:
    def test_layernorm_preserves_input_shape(self):
        net, vbn = _parse_net(_MXAI_ALL_LAYERS)
        result = check_composite_network_types(net, vbn)
        ln = next(l for l in result.resolved_layers if l.layer_type == "LayerNorm")
        assert ln.input_shape == ln.output_shape == [16]

    def test_dropout_preserves_input_shape(self):
        net, vbn = _parse_net(_MXAI_ALL_LAYERS)
        result = check_composite_network_types(net, vbn)
        do = next(l for l in result.resolved_layers if l.layer_type == "Dropout")
        assert do.input_shape == do.output_shape == [16]

    def test_activation_preserves_input_shape(self):
        net, vbn = _parse_net(_MXAI_ALL_LAYERS)
        result = check_composite_network_types(net, vbn)
        act = next(l for l in result.resolved_layers if l.layer_type == "Activation")
        assert act.input_shape == act.output_shape == [16]

    def test_reshape_output_equals_target_shape(self):
        net, vbn = _parse_net(_MXAI_RESHAPE)
        result = check_composite_network_types(net, vbn)
        assert result.ok, result.errors
        reshape = next(l for l in result.resolved_layers if l.layer_type == "Reshape")
        assert reshape.input_shape == [16]
        assert reshape.output_shape == [4, 4]

    def test_dense_after_reshape_uses_reshaped_shape(self):
        net, vbn = _parse_net(_MXAI_RESHAPE)
        result = check_composite_network_types(net, vbn)
        layers = result.resolved_layers
        dense_after_reshape = next(
            l for l in layers if l.layer_type == "Dense" and l.input_shape == [4, 4]
        )
        assert dense_after_reshape.output_shape == [2]

    def test_reshape_rejects_wrong_product_of_dims(self):
        net, vbn = _parse_net(_MXAI_RESHAPE_BAD_PRODUCT)
        result = check_composite_network_types(net, vbn)
        assert not result.ok
        assert any("product" in e for e in result.errors)

    def test_all_layers_check_returns_ok(self):
        net, vbn = _parse_net(_MXAI_ALL_LAYERS)
        result = check_composite_network_types(net, vbn)
        assert result.ok, result.errors


# ---------------------------------------------------------------------------
# TestCompositeNetworkBlockShapes
# ---------------------------------------------------------------------------

class TestCompositeNetworkBlockShapes:
    def test_block_layers_have_correct_shapes(self):
        net, vbn = _parse_net(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        result = check_composite_network_types(net, vbn)
        assert result.ok, result.errors
        block = result.resolved_blocks[0]
        # Block input shape = shape after Dense(8,relu) in top_layers = [8]
        assert block.input_shape == [8]
        assert block.layers[0].input_shape == [8]
        assert block.layers[0].output_shape == [8]

    def test_block_output_shape_equals_last_block_layer_output(self):
        net, vbn = _parse_net(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        result = check_composite_network_types(net, vbn)
        block = result.resolved_blocks[0]
        assert block.output_shape == block.layers[-1].output_shape

    def test_residual_from_previous_ok_when_shapes_match(self):
        net, vbn = _parse_net(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        result = check_composite_network_types(net, vbn)
        assert result.ok, result.errors

    def test_residual_from_named_ok_when_shapes_match(self):
        net, vbn = _parse_net(_MXAI_BLOCK_RESIDUAL_NAMED)
        result = check_composite_network_types(net, vbn)
        assert result.ok, result.errors

    def test_top_layer_after_block_uses_block_output_shape(self):
        net, vbn = _parse_net(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        result = check_composite_network_types(net, vbn)
        # Final Dense(1,linear) in top_layers should have input_shape = block output [8]
        final_dense = next(
            l for l in reversed(result.resolved_layers)
            if l.layer_type == "Dense"
        )
        assert final_dense.input_shape == [8]
        assert final_dense.output_shape == [1]


# ---------------------------------------------------------------------------
# TestCompositeNetworkShapeErrors
# ---------------------------------------------------------------------------

class TestCompositeNetworkShapeErrors:
    def test_residual_from_previous_errors_when_block_changes_shape(self):
        net, vbn = _parse_net(_MXAI_RESIDUAL_SHAPE_MISMATCH_PREVIOUS)
        result = check_composite_network_types(net, vbn)
        assert not result.ok
        assert any("RESIDUAL shape mismatch" in e for e in result.errors)

    def test_residual_from_named_errors_when_shape_mismatches(self):
        # features has shape [5] (emb[4]+x2[1]), block output is [8]
        net, vbn = _parse_net(_MXAI_RESIDUAL_SHAPE_MISMATCH_NAMED)
        result = check_composite_network_types(net, vbn)
        assert not result.ok
        assert any("RESIDUAL shape mismatch" in e for e in result.errors)

    def test_concat_errors_when_undeclared_source(self):
        src = _MXAI_EMBEDDING_BASIC.replace(
            "CONCAT [cat_emb, price, weight]",
            "CONCAT [cat_emb, price, unknown_tensor]"
        )
        net, vbn = _parse_net(src)
        result = check_composite_network_types(net, vbn)
        assert not result.ok
        assert any("undeclared source" in e for e in result.errors)


# ---------------------------------------------------------------------------
# TestCompositeNetworkDispatch
# ---------------------------------------------------------------------------

class TestCompositeNetworkDispatch:
    def test_check_network_types_dispatches_to_composite_for_composite_kind(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        assert net.kind == "composite_network"
        result = check_network_types(net, vbn)
        # composite result has named_shapes populated
        assert "cat_emb" in result.named_shapes

    def test_check_network_types_dispatches_to_dense_for_dense_kind(self):
        net, vbn = _parse_net(_MXAI_P18_DENSE_ONLY)
        assert net.kind == "dense_network"
        result = check_network_types(net, vbn)
        assert result.ok
        # dense result has no named_shapes
        assert result.named_shapes == {}

    def test_interpretability_warning_present_in_composite_result(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        result = check_composite_network_types(net, vbn)
        assert any("interpretability_level=reduced" in w for w in result.warnings)

    def test_composite_network_output_type_inferred_correctly(self):
        net, vbn = _parse_net(_MXAI_EMBEDDING_BASIC)
        result = check_composite_network_types(net, vbn)
        assert result.output_type is not None
        assert result.output_type.name == "ProbabilityMap"
