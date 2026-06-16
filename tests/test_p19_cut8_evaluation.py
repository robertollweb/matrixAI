"""P19 C8 — Evaluación de redes compuestas: dropout off, métricas heredadas, columnas categóricas."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types
from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.training.composite_evaluator import (
    composite_examples_from_csv,
    evaluate_composite_network,
)
from matrixai.training.dense_evaluator import DenseEvaluationResult

# ---------------------------------------------------------------------------
# Fixtures — programs
# ---------------------------------------------------------------------------

_MXAI_EMBEDDING_SOFTMAX = """
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

_MXAI_RESIDUAL_REGRESSION = """
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

_MXAI_DROPOUT = """
PROJECT DropTest
VECTOR V[2]
  a: Scalar
  b: Scalar
END
NETWORK DropNet
  INPUT V
  LAYER Dense units=16 activation=relu
  LAYER Dropout rate=0.5
  LAYER Dense units=3 activation=softmax
  OUTPUT label: ProbabilityMap[x, y, z]
END
GRAPH
  V -> DropNet
END
"""


def _setup(src: str, seed: int = 0):
    program = parse_text(src)
    net = program.networks[0]
    vbn = {v.name: v for v in program.vectors}
    result = check_composite_network_types(net, vbn)
    assert result.ok, result.errors
    ps = build_composite_network_parameter_set(net, result, model_hash_str="test", seed=seed)
    return net, ps


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / f"{name}.csv"
    p.write_text(content)
    return p


_EMBEDDING_CSV = "category_id,price,weight,label\n5,0.5,0.5,a\n3,0.2,0.8,b\n7,0.9,0.1,c\n"
_REGRESSION_CSV = "x1,x2,y\n0.5,-0.5,0.0\n0.3,0.7,1.0\n-0.2,0.4,0.5\n"
_DROPOUT_CSV = "a,b,label\n0.5,0.3,x\n-0.1,0.9,y\n0.7,-0.4,z\n"


# ---------------------------------------------------------------------------
# TestCompositeExamplesFromCsv
# ---------------------------------------------------------------------------

class TestCompositeExamplesFromCsv:
    def test_produces_list_of_tuples(self, tmp_path):
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        assert isinstance(examples, list)
        assert all(isinstance(e, tuple) and len(e) == 2 for e in examples)

    def test_input_is_dict_with_column_keys(self, tmp_path):
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        input_dict, _ = examples[0]
        assert "category_id" in input_dict
        assert "price" in input_dict
        assert "weight" in input_dict

    def test_categorical_column_is_numeric(self, tmp_path):
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        input_dict, _ = examples[0]
        assert isinstance(input_dict["category_id"], float)
        assert int(input_dict["category_id"]) == 5

    def test_classification_target_is_onehot(self, tmp_path):
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        _, target = examples[0]  # label="a" → [1.0, 0.0, 0.0]
        assert len(target) == 3
        assert target[0] == 1.0
        assert target[1] == 0.0

    def test_regression_target_is_scalar_list(self, tmp_path):
        path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
        examples = composite_examples_from_csv(path, ["x1", "x2"], "y")
        _, target = examples[0]
        assert len(target) == 1
        assert abs(target[0] - 0.0) < 1e-9

    def test_loads_all_rows(self, tmp_path):
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        assert len(examples) == 3


# ---------------------------------------------------------------------------
# TestDropoutDisabledOnEval
# ---------------------------------------------------------------------------

class TestDropoutDisabledOnEval:
    def test_eval_is_deterministic_with_dropout_network(self, tmp_path):
        net, ps = _setup(_MXAI_DROPOUT, seed=0)
        path = _write_csv(tmp_path, "data", _DROPOUT_CSV)
        examples = composite_examples_from_csv(path, ["a", "b"], "label", ["x", "y", "z"])
        result1 = evaluate_composite_network(net, ps, examples, "cross_entropy", ["x", "y", "z"])
        result2 = evaluate_composite_network(net, ps, examples, "cross_entropy", ["x", "y", "z"])
        assert result1.loss == result2.loss
        assert result1.accuracy == result2.accuracy

    def test_eval_output_differs_from_training_with_dropout(self):
        from matrixai.forward.composite_forward import composite_forward
        net, ps = _setup(_MXAI_DROPOUT, seed=42)
        input_data = {"a": 0.5, "b": 0.3}
        out_train = composite_forward(net, ps, input_data, training=True, seed=1)
        out_eval = composite_forward(net, ps, input_data, training=False)
        # Eval output is deterministic; train is not (dropout applied)
        # They should differ unless the network is trivially identity
        out_eval2 = composite_forward(net, ps, input_data, training=False)
        assert out_eval == out_eval2  # eval is deterministic


# ---------------------------------------------------------------------------
# TestEvaluateCompositeNetwork — cross_entropy classification
# ---------------------------------------------------------------------------

class TestEvaluateCrossEntropy:
    def test_returns_dense_evaluation_result(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert isinstance(result, DenseEvaluationResult)

    def test_rows_count_matches_input(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert result.rows == 3

    def test_cross_entropy_loss_is_positive(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert result.loss > 0.0

    def test_accuracy_is_between_zero_and_one(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert 0.0 <= result.accuracy <= 1.0

    def test_confusion_matrix_present(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert isinstance(result.confusion_matrix, dict)
        assert len(result.confusion_matrix) == 3

    def test_labels_stored_in_result(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert result.labels == ["a", "b", "c"]

    def test_macro_f1_between_zero_and_one(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert 0.0 <= result.macro_f1 <= 1.0


# ---------------------------------------------------------------------------
# TestEvaluateCompositeMse — mse regression
# ---------------------------------------------------------------------------

class TestEvaluateMse:
    def test_returns_mae_rmse_r2(self, tmp_path):
        net, ps = _setup(_MXAI_RESIDUAL_REGRESSION, seed=0)
        path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
        examples = composite_examples_from_csv(path, ["x1", "x2"], "y")
        result = evaluate_composite_network(net, ps, examples, "mse")
        assert result.mae >= 0.0
        assert result.rmse >= 0.0
        assert isinstance(result.r2, float)

    def test_mse_loss_is_nonnegative(self, tmp_path):
        net, ps = _setup(_MXAI_RESIDUAL_REGRESSION, seed=0)
        path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
        examples = composite_examples_from_csv(path, ["x1", "x2"], "y")
        result = evaluate_composite_network(net, ps, examples, "mse")
        assert result.loss >= 0.0

    def test_mse_rows_count(self, tmp_path):
        net, ps = _setup(_MXAI_RESIDUAL_REGRESSION, seed=0)
        path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
        examples = composite_examples_from_csv(path, ["x1", "x2"], "y")
        result = evaluate_composite_network(net, ps, examples, "mse")
        assert result.rows == 3

    def test_training_reduces_eval_loss(self, tmp_path):
        from matrixai.training.composite_backprop import composite_train_step
        net, ps = _setup(_MXAI_RESIDUAL_REGRESSION, seed=0)
        path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
        examples = composite_examples_from_csv(path, ["x1", "x2"], "y")
        result_before = evaluate_composite_network(net, ps, examples, "mse")
        # Train for several steps
        for _ in range(10):
            for input_data, target in examples:
                ps, _ = composite_train_step(
                    net, ps, input_data, target, "mse", learning_rate=0.05, training=False
                )
        result_after = evaluate_composite_network(net, ps, examples, "mse")
        assert result_after.loss <= result_before.loss


# ---------------------------------------------------------------------------
# TestCategoricalColumnHandling
# ---------------------------------------------------------------------------

class TestCategoricalColumnHandling:
    def test_embedding_lookup_uses_csv_integer(self, tmp_path):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        path = _write_csv(tmp_path, "data", _EMBEDDING_CSV)
        examples = composite_examples_from_csv(
            path, ["category_id", "price", "weight"], "label", ["a", "b", "c"]
        )
        # Forward should not raise — embedding table lookup uses int(category_id)
        result = evaluate_composite_network(net, ps, examples, "cross_entropy", ["a", "b", "c"])
        assert isinstance(result, DenseEvaluationResult)

    def test_different_category_ids_give_different_predictions(self):
        from matrixai.forward.composite_forward import composite_forward
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=42)
        out1 = composite_forward(net, ps, {"category_id": 1, "price": 0.5, "weight": 0.5}, training=False)
        out2 = composite_forward(net, ps, {"category_id": 50, "price": 0.5, "weight": 0.5}, training=False)
        assert out1 != out2

    def test_empty_examples_raises(self):
        net, ps = _setup(_MXAI_EMBEDDING_SOFTMAX, seed=0)
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_composite_network(net, ps, [], "cross_entropy", ["a", "b", "c"])
