"""P19 C6 — Backprop stdlib + gradient check numérico para composite_network."""
from __future__ import annotations

import pytest

from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types
from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.training.composite_backprop import (
    CompositeBackpropError,
    composite_compute_gradients,
    composite_train_step,
    gradient_check,
    numerical_gradient,
)
from matrixai.training.dense_backprop import compute_loss


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
  LAYER Dropout rate=0.3
  LAYER Activation kind=relu
  LAYER Dense units=4 activation=softmax
  OUTPUT label: ProbabilityMap[a, b, c, d]
END

GRAPH
  V -> AllNet
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
    LAYER Dropout rate=0.5
    RESIDUAL FROM features
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[pos, neg]
END

GRAPH
  H -> ResNet
END
"""


def _setup(src: str, seed: int = 0):
    program = parse_text(src)
    net = program.networks[0]
    vbn = {v.name: v for v in program.vectors}
    result = check_composite_network_types(net, vbn)
    assert result.ok, result.errors
    ps = build_composite_network_parameter_set(net, result, model_hash_str="test_hash", seed=seed)
    return net, result, ps


# ---------------------------------------------------------------------------
# TestGradientShapes
# ---------------------------------------------------------------------------

class TestGradientShapes:
    def test_gradients_include_all_dense_keys(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC)
        grads = composite_compute_gradients(
            net, ps, {"category_id": 5, "price": 1.0, "weight": 0.5},
            [1.0, 0.0, 0.0], "cross_entropy", training=False
        )
        assert "CategoryNet.L1.W" in grads
        assert "CategoryNet.L1.b" in grads
        assert "CategoryNet.L2.W" in grads
        assert "CategoryNet.L2.b" in grads

    def test_gradients_include_embedding_table(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC)
        grads = composite_compute_gradients(
            net, ps, {"category_id": 3, "price": 0.5, "weight": 0.2},
            [0.0, 1.0, 0.0], "cross_entropy", training=False
        )
        assert "CategoryNet.cat_emb.table" in grads

    def test_embedding_table_gradient_shape(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC)
        grads = composite_compute_gradients(
            net, ps, {"category_id": 2, "price": 0.0, "weight": 0.0},
            [0.0, 0.0, 1.0], "cross_entropy", training=False
        )
        d_table = grads["CategoryNet.cat_emb.table"]
        assert len(d_table) == 100   # vocab size
        assert len(d_table[0]) == 8  # embedding dim

    def test_embedding_table_gradient_nonzero_at_used_index(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC)
        idx = 7
        grads = composite_compute_gradients(
            net, ps, {"category_id": idx, "price": 0.5, "weight": 0.5},
            [1.0, 0.0, 0.0], "cross_entropy", training=False
        )
        d_table = grads["CategoryNet.cat_emb.table"]
        # Row at used index should be non-zero; all other rows should be zero
        used_row = d_table[idx]
        other_rows = [d_table[i] for i in range(100) if i != idx]
        assert any(v != 0.0 for v in used_row)
        assert all(v == 0.0 for row in other_rows for v in row)

    def test_gradients_include_block_layernorm_params(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        grads = composite_compute_gradients(
            net, ps, {"x1": 0.5, "x2": -0.5}, [1.0], "mse", training=False
        )
        assert "ResNet.res1.L2.gamma" in grads
        assert "ResNet.res1.L2.beta" in grads

    def test_layernorm_gamma_gradient_shape(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        grads = composite_compute_gradients(
            net, ps, {"x1": 0.5, "x2": 0.3}, [0.5], "mse", training=False
        )
        assert len(grads["ResNet.res1.L2.gamma"]) == 8
        assert len(grads["ResNet.res1.L2.beta"]) == 8


# ---------------------------------------------------------------------------
# TestGradientCheck
# ---------------------------------------------------------------------------

class TestGradientCheck:
    def test_gradient_check_dense_weight(self):
        net, _, ps = _setup(_MXAI_ALL_LAYERS, seed=42)
        analytical, num, passed = gradient_check(
            net, ps, {"a": 0.5, "b": 0.5}, [1.0, 0.0, 0.0, 0.0],
            "cross_entropy", "AllNet.L1.W", (0, 0)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"

    def test_gradient_check_dense_bias(self):
        net, _, ps = _setup(_MXAI_ALL_LAYERS, seed=42)
        analytical, num, passed = gradient_check(
            net, ps, {"a": 0.3, "b": 0.7}, [0.0, 1.0, 0.0, 0.0],
            "cross_entropy", "AllNet.L5.W", (1, 3)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"

    def test_gradient_check_layernorm_gamma(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS, seed=7)
        analytical, num, passed = gradient_check(
            net, ps, {"x1": 0.8, "x2": -0.4}, [0.0],
            "mse", "ResNet.res1.L2.gamma", (0,)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"

    def test_gradient_check_layernorm_beta(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS, seed=7)
        analytical, num, passed = gradient_check(
            net, ps, {"x1": 0.8, "x2": -0.4}, [0.0],
            "mse", "ResNet.res1.L2.beta", (2,)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"

    def test_gradient_check_embedding_table_used_row(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC, seed=5)
        idx = 3
        analytical, num, passed = gradient_check(
            net, ps, {"category_id": idx, "price": 0.5, "weight": 0.5},
            [1.0, 0.0, 0.0], "cross_entropy",
            "CategoryNet.cat_emb.table", (idx, 0)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"

    def test_gradient_check_embedding_table_second_dim(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC, seed=5)
        idx = 3
        analytical, num, passed = gradient_check(
            net, ps, {"category_id": idx, "price": 0.5, "weight": 0.5},
            [1.0, 0.0, 0.0], "cross_entropy",
            "CategoryNet.cat_emb.table", (idx, 4)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"

    def test_gradient_check_block_dense_weight(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS, seed=3)
        analytical, num, passed = gradient_check(
            net, ps, {"x1": 0.6, "x2": -0.6}, [1.0],
            "mse", "ResNet.res1.L1.W", (0, 1)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"


# ---------------------------------------------------------------------------
# TestTrainingStep
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def test_train_step_returns_tuple(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC)
        result = composite_train_step(
            net, ps, {"category_id": 5, "price": 0.5, "weight": 0.5},
            [1.0, 0.0, 0.0], "cross_entropy"
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_train_step_returns_float_loss(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC)
        _, loss = composite_train_step(
            net, ps, {"category_id": 5, "price": 0.5, "weight": 0.5},
            [1.0, 0.0, 0.0], "cross_entropy"
        )
        assert isinstance(loss, float) and loss > 0

    def test_train_step_updates_parameters(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        ps2, _ = composite_train_step(
            net, ps, {"x1": 0.5, "x2": -0.5}, [1.0], "mse", learning_rate=0.1
        )
        # At least one parameter should have changed
        changed = any(
            ps.parameters[k]["values"] != ps2.parameters[k]["values"]
            for k in ps.parameters
        )
        assert changed


# ---------------------------------------------------------------------------
# TestLossReduction
# ---------------------------------------------------------------------------

class TestLossReduction:
    def test_multiple_steps_reduce_mse_loss(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        input_data = {"x1": 0.5, "x2": -0.5}
        target = [0.0]
        loss_prev = float("inf")
        for _ in range(5):
            ps, loss = composite_train_step(net, ps, input_data, target, "mse", learning_rate=0.01, training=False)
        assert loss < loss_prev

    def test_multiple_steps_reduce_cross_entropy(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC)
        input_data = {"category_id": 5, "price": 0.5, "weight": 0.5}
        target = [1.0, 0.0, 0.0]
        losses = []
        for _ in range(8):
            ps, loss = composite_train_step(
                net, ps, input_data, target, "cross_entropy", learning_rate=0.05, training=False
            )
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_embedding_table_updates_on_train(self):
        net, _, ps = _setup(_MXAI_EMBEDDING_BASIC, seed=42)
        table_before = list(ps.parameters["CategoryNet.cat_emb.table"]["values"][5])
        input_data = {"category_id": 5, "price": 0.5, "weight": 0.5}
        ps, _ = composite_train_step(
            net, ps, input_data, [1.0, 0.0, 0.0], "cross_entropy", learning_rate=0.1, training=False
        )
        table_after = ps.parameters["CategoryNet.cat_emb.table"]["values"][5]
        assert table_before != table_after

    def test_layernorm_gamma_updates_on_train(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS, seed=0)
        gamma_before = list(ps.parameters["ResNet.res1.L2.gamma"]["values"])
        ps, _ = composite_train_step(
            net, ps, {"x1": 0.5, "x2": 0.3}, [0.0], "mse", learning_rate=0.1, training=False
        )
        gamma_after = ps.parameters["ResNet.res1.L2.gamma"]["values"]
        assert gamma_before != gamma_after


# ---------------------------------------------------------------------------
# TestResidualGradient
# ---------------------------------------------------------------------------

class TestResidualGradient:
    def test_residual_block_gradient_flows_to_both_paths(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS, seed=1)
        grads = composite_compute_gradients(
            net, ps, {"x1": 0.5, "x2": -0.3}, [0.5], "mse", training=False
        )
        # Both top-level Dense layers and block Dense should have gradients
        assert "ResNet.L1.W" in grads        # top-level Dense before block
        assert "ResNet.res1.L1.W" in grads   # block Dense
        assert "ResNet.L2.W" in grads        # final Dense after block

    def test_gradient_check_passes_for_residual_network(self):
        net, _, ps = _setup(_MXAI_BLOCK_RESIDUAL_PREVIOUS, seed=2)
        analytical, num, passed = gradient_check(
            net, ps, {"x1": 0.4, "x2": 0.6}, [0.8],
            "mse", "ResNet.L2.W", (0, 0)
        )
        assert passed, f"analytical={analytical:.6f}, numerical={num:.6f}"
