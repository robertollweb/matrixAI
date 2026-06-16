"""P18 C8 — Evaluación métricas: mae/rmse/r2 (regresión) y accuracy/confusion (clasificación)."""
from __future__ import annotations

import math

import pytest

from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai.parameters.network_params import build_network_parameter_set
from matrixai.parameters.store import ParameterSet
from matrixai.training.dense_backprop import dense_train_step
from matrixai.training.dense_evaluator import (
    DenseEvaluationResult,
    compute_accuracy,
    compute_mae,
    compute_r2,
    compute_rmse,
    evaluate_dense_network,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _net(*layer_specs: tuple[int, str], name: str = "Net", input_dim: int = 2) -> NetworkSpec:
    layers, dim = [], input_dim
    for i, (units, act) in enumerate(layer_specs, start=1):
        layers.append(DenseLayerSpec(index=i, units=units, activation=act,
                                     input_shape=[dim], output_shape=[units]))
        dim = units
    return NetworkSpec(name=name, input="X", layers=layers, output="y", output_type_str="Scalar")


def _ps(net: NetworkSpec, seed: int = 42) -> ParameterSet:
    return build_network_parameter_set(net, net.layers, "test_hash", seed=seed)


def _manual_ps_identity(net: NetworkSpec) -> ParameterSet:
    """Single linear layer that acts as identity: W=[[1,0],[0,1]], b=[0,0]."""
    params = {
        "Net.W1": {"function": "Net", "role": "weights", "type": "Tensor",
                   "shape": [2, 2], "dtype": "float32", "initializer": "xavier_normal",
                   "values": [[1.0, 0.0], [0.0, 1.0]], "is_layer": True},
        "Net.b1": {"function": "Net", "role": "bias", "type": "Vector",
                   "shape": [2], "dtype": "float32", "initializer": "zeros",
                   "values": [0.0, 0.0], "is_layer": True},
    }
    return ParameterSet(parameter_set_id="id", model_hash="h",
                        parameter_schema_hash="s", parameters=params)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def test_mae_perfect_prediction():
    assert compute_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_correct_value():
    # |2-1| + |3-1| = 1 + 2 → avg = 1.5
    assert abs(compute_mae([2.0, 3.0], [1.0, 1.0]) - 1.5) < 1e-9


def test_rmse_perfect_prediction():
    assert compute_rmse([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_rmse_correct_value():
    # errors=[1,2], mse=(1+4)/2=2.5, rmse=sqrt(2.5)
    assert abs(compute_rmse([2.0, 3.0], [1.0, 1.0]) - math.sqrt(2.5)) < 1e-9


def test_r2_perfect_prediction():
    assert abs(compute_r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_r2_constant_baseline():
    # predicting mean → r2 = 0.0
    targets = [1.0, 2.0, 3.0]
    mean = sum(targets) / len(targets)
    assert abs(compute_r2([mean, mean, mean], targets)) < 1e-9


def test_r2_worse_than_baseline_is_negative():
    # Very bad predictions → r2 < 0
    assert compute_r2([10.0, 10.0, 10.0], [1.0, 2.0, 3.0]) < 0.0


def test_accuracy_binary_all_correct():
    preds = [[0.9], [0.1], [0.8]]
    tgts = [[1.0], [0.0], [1.0]]
    assert compute_accuracy(preds, tgts) == 1.0


def test_accuracy_multiclass_all_correct():
    preds = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]
    tgts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert compute_accuracy(preds, tgts) == 1.0


def test_accuracy_all_wrong():
    preds = [[0.1], [0.9]]
    tgts = [[1.0], [0.0]]
    assert compute_accuracy(preds, tgts) == 0.0


# ---------------------------------------------------------------------------
# evaluate_dense_network — regression
# ---------------------------------------------------------------------------

def test_evaluate_regression_returns_result():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    examples = [([0.5, 0.5], [1.0]), ([1.0, 0.0], [0.5])]
    result = evaluate_dense_network(net, ps, examples, "mse")
    assert isinstance(result, DenseEvaluationResult)


def test_evaluate_regression_row_count():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    examples = [([0.5, 0.5], [1.0]), ([1.0, 0.0], [0.5]), ([0.0, 1.0], [0.8])]
    result = evaluate_dense_network(net, ps, examples, "mse")
    assert result.rows == 3


def test_evaluate_regression_loss_nonneg():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    examples = [([0.5, 0.5], [1.0]), ([1.0, 0.0], [2.0])]
    result = evaluate_dense_network(net, ps, examples, "mse")
    assert result.loss >= 0.0


def test_evaluate_regression_mae_nonneg():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    examples = [([0.5, 0.5], [1.0]), ([1.0, 0.0], [2.0])]
    result = evaluate_dense_network(net, ps, examples, "mse")
    assert result.mae >= 0.0


def test_evaluate_regression_is_regression_flag():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    result = evaluate_dense_network(net, ps, [([0.5, 0.5], [1.0])], "mse")
    assert result.is_regression()


def test_evaluate_regression_to_dict_has_mae():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    result = evaluate_dense_network(net, ps, [([0.5, 0.5], [1.0])], "mse")
    d = result.to_dict()
    assert "mae" in d and "rmse" in d and "r2" in d


# ---------------------------------------------------------------------------
# evaluate_dense_network — classification
# ---------------------------------------------------------------------------

def test_evaluate_crossentropy_returns_accuracy():
    net = _net((4, "relu"), (3, "softmax"))
    ps = _ps(net)
    examples = [([1.0, 0.0], [1.0, 0.0, 0.0]), ([0.0, 1.0], [0.0, 1.0, 0.0])]
    result = evaluate_dense_network(net, ps, examples, "cross_entropy",
                                    labels=["a", "b", "c"])
    assert 0.0 <= result.accuracy <= 1.0


def test_evaluate_crossentropy_not_regression():
    net = _net((4, "relu"), (3, "softmax"))
    ps = _ps(net)
    result = evaluate_dense_network(net, ps, [([1.0, 0.0], [1.0, 0.0, 0.0])],
                                    "cross_entropy", labels=["a", "b", "c"])
    assert not result.is_regression()


def test_evaluate_bce_returns_accuracy():
    net = _net((4, "relu"), (1, "sigmoid"))
    ps = _ps(net)
    examples = [([1.0, 0.0], [1.0]), ([0.0, 1.0], [0.0])]
    result = evaluate_dense_network(net, ps, examples, "binary_cross_entropy")
    assert 0.0 <= result.accuracy <= 1.0


def test_evaluate_crossentropy_perfect_accuracy_after_training():
    # Train until convergence on 2 distinguishable inputs → accuracy should reach 1.0
    net = _net((8, "relu"), (2, "softmax"))
    ps = _ps(net, seed=1)
    data = [([1.0, 0.0], [1.0, 0.0]), ([0.0, 1.0], [0.0, 1.0])]
    for _ in range(300):
        for x, t in data:
            ps, _ = dense_train_step(net, ps, x, t, "cross_entropy", learning_rate=0.05)
    result = evaluate_dense_network(net, ps, data, "cross_entropy", labels=["A", "B"])
    assert result.accuracy == 1.0


def test_evaluate_regression_mae_zero_after_convergence():
    # Train single sample until loss ≈ 0 → mae should also be ≈ 0
    net = _net((8, "relu"), (1, "linear"), input_dim=1)
    ps = _ps(net, seed=2)
    x, t = [1.0], [2.0]
    for _ in range(500):
        ps, _ = dense_train_step(net, ps, x, t, "mse", learning_rate=0.01)
    result = evaluate_dense_network(net, ps, [(x, t)], "mse")
    assert result.mae < 0.05


def test_evaluate_empty_examples_raises():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    with pytest.raises(ValueError):
        evaluate_dense_network(net, ps, [], "mse")
