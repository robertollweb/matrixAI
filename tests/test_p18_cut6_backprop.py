"""P18 C6 — Backprop stdlib: mse, binary_cross_entropy, cross_entropy + SGD."""
from __future__ import annotations

import math

import pytest

from matrixai.forward import dense_forward
from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai.parameters.network_params import build_network_parameter_set
from matrixai.parameters.store import ParameterSet
from matrixai.training.dense_backprop import (
    DenseBackpropError,
    binary_cross_entropy_loss,
    compute_loss,
    cross_entropy_loss,
    dense_compute_gradients,
    dense_train_step,
    mse_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _net(*layer_specs: tuple[int, str], name: str = "Net", input_dim: int = 2) -> NetworkSpec:
    layers = []
    dim = input_dim
    for i, (units, activation) in enumerate(layer_specs, start=1):
        layers.append(DenseLayerSpec(
            index=i, units=units, activation=activation,
            input_shape=[dim], output_shape=[units],
        ))
        dim = units
    return NetworkSpec(name=name, input="X", layers=layers, output="y", output_type_str="Scalar")


def _ps(net: NetworkSpec, seed: int = 42) -> ParameterSet:
    return build_network_parameter_set(net, net.layers, "test_hash", seed=seed)


def _manual_ps(net: NetworkSpec, W1: list, b1: list) -> ParameterSet:
    params = {
        f"{net.name}.W1": {"function": net.name, "role": "weights", "type": "Tensor",
                           "shape": [len(W1), len(W1[0])], "dtype": "float32",
                           "initializer": "he_normal", "values": W1, "is_layer": True},
        f"{net.name}.b1": {"function": net.name, "role": "bias", "type": "Vector",
                           "shape": [len(b1)], "dtype": "float32",
                           "initializer": "zeros", "values": b1, "is_layer": True},
    }
    return ParameterSet(
        parameter_set_id="manual", model_hash="test_hash",
        parameter_schema_hash="params_manual", parameters=params,
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def test_mse_loss_zero_for_perfect_prediction():
    assert mse_loss([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_mse_loss_correct_value():
    # ((3-1)^2 + (4-2)^2) / 2 = (4+4)/2 = 4.0
    assert abs(mse_loss([3.0, 4.0], [1.0, 2.0]) - 4.0) < 1e-9


def test_binary_cross_entropy_loss_perfect_prediction():
    # p=1.0, t=1.0 → -log(1.0) ≈ 0
    assert binary_cross_entropy_loss([1.0], [1.0]) < 1e-5


def test_binary_cross_entropy_loss_bad_prediction():
    # p≈0, t=1.0 → high loss
    assert binary_cross_entropy_loss([0.001], [1.0]) > 5.0


def test_cross_entropy_loss_perfect_onehot():
    # p=[1,0,0], t=[1,0,0] → -log(1.0) ≈ 0
    assert cross_entropy_loss([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) < 1e-5


def test_compute_loss_dispatches_mse():
    assert compute_loss("mse", [2.0], [1.0]) == mse_loss([2.0], [1.0])


def test_compute_loss_unknown_raises():
    with pytest.raises(DenseBackpropError):
        compute_loss("huber", [1.0], [0.0])


# ---------------------------------------------------------------------------
# Gradient shapes
# ---------------------------------------------------------------------------

def test_gradients_have_all_param_keys():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    grads = dense_compute_gradients(net, ps, [0.5, 0.5], [1.0], "mse")
    assert "Net.W1" in grads
    assert "Net.b1" in grads
    assert "Net.W2" in grads
    assert "Net.b2" in grads


def test_gradient_dW1_shape():
    net = _net((4, "relu"), input_dim=2)
    ps = _ps(net)
    grads = dense_compute_gradients(net, ps, [0.5, 0.5], [1.0, 0.0, 0.0, 0.0], "mse")
    assert len(grads["Net.W1"]) == 4
    assert len(grads["Net.W1"][0]) == 2


def test_gradient_db1_shape():
    net = _net((4, "relu"), input_dim=2)
    ps = _ps(net)
    grads = dense_compute_gradients(net, ps, [0.5, 0.5], [1.0, 0.0, 0.0, 0.0], "mse")
    assert len(grads["Net.b1"]) == 4


# ---------------------------------------------------------------------------
# Training step API
# ---------------------------------------------------------------------------

def test_train_step_returns_tuple():
    net = _net((4, "relu"), (1, "linear"), input_dim=2)
    ps = _ps(net)
    result = dense_train_step(net, ps, [0.5, 0.5], [1.0], "mse")
    assert isinstance(result, tuple) and len(result) == 2


def test_train_step_loss_is_float():
    net = _net((4, "relu"), (1, "linear"), input_dim=2)
    ps = _ps(net)
    _, loss = dense_train_step(net, ps, [0.5, 0.5], [1.0], "mse")
    assert isinstance(loss, float)


def test_train_step_returns_parameter_set():
    net = _net((4, "relu"), (1, "linear"), input_dim=2)
    ps = _ps(net)
    ps2, _ = dense_train_step(net, ps, [0.5, 0.5], [1.0], "mse")
    assert isinstance(ps2, ParameterSet)


def test_train_step_preserves_schema_hash():
    net = _net((4, "relu"), (1, "linear"), input_dim=2)
    ps = _ps(net)
    ps2, _ = dense_train_step(net, ps, [0.5, 0.5], [1.0], "mse", learning_rate=0.01)
    assert ps2.parameter_schema_hash == ps.parameter_schema_hash


def test_train_step_parameters_change():
    net = _net((4, "relu"), (1, "linear"), input_dim=2)
    ps = _ps(net, seed=1)
    ps2, _ = dense_train_step(net, ps, [1.0, 0.0], [2.0], "mse", learning_rate=0.1)
    assert ps2.parameters["Net.W1"]["values"] != ps.parameters["Net.W1"]["values"]


def test_train_step_source_is_trained():
    net = _net((4, "relu"), (1, "linear"), input_dim=2)
    ps = _ps(net)
    ps2, _ = dense_train_step(net, ps, [0.5, 0.5], [1.0], "mse")
    assert ps2.source == "trained"


# ---------------------------------------------------------------------------
# Loss decreases after one step
# ---------------------------------------------------------------------------

def test_loss_decreases_after_one_mse_step():
    net = _net((8, "relu"), (1, "linear"), input_dim=2)
    ps = _ps(net, seed=5)
    x, t = [1.0, 0.0], [3.0]
    ps2, loss1 = dense_train_step(net, ps, x, t, "mse", learning_rate=0.01)
    loss2 = compute_loss("mse", dense_forward(net, ps2, x), t)
    assert loss2 < loss1


def test_loss_decreases_after_one_cross_entropy_step():
    net = _net((8, "relu"), (3, "softmax"), input_dim=2)
    ps = _ps(net, seed=7)
    x, t = [1.0, 0.0], [1.0, 0.0, 0.0]
    ps2, loss1 = dense_train_step(net, ps, x, t, "cross_entropy", learning_rate=0.05)
    loss2 = compute_loss("cross_entropy", dense_forward(net, ps2, x), t)
    assert loss2 < loss1


def test_loss_decreases_after_one_bce_step():
    net = _net((4, "relu"), (1, "sigmoid"), input_dim=2)
    ps = _ps(net, seed=3)
    x, t = [1.0, 0.0], [1.0]
    ps2, loss1 = dense_train_step(net, ps, x, t, "binary_cross_entropy", learning_rate=0.05)
    loss2 = compute_loss("binary_cross_entropy", dense_forward(net, ps2, x), t)
    assert loss2 < loss1


# ---------------------------------------------------------------------------
# Gradient direction (known-value tests)
# ---------------------------------------------------------------------------

def test_mse_gradient_known_values_linear():
    # Single linear layer: W=[[1]], b=[0], x=[1], t=[0]
    # output = 1.0, target = 0.0, mse = 1.0
    # dL/dz = dL/da * f'(z) = 2*(1-0)/1 * 1 = 2.0
    # dW = dz * x = 2.0 * 1.0 = 2.0
    # db = dz = 2.0
    net = _net((1, "linear"), name="N", input_dim=1)
    ps = _manual_ps(net, [[1.0]], [0.0])
    grads = dense_compute_gradients(net, ps, [1.0], [0.0], "mse")
    assert abs(grads["N.W1"][0][0] - 2.0) < 1e-9
    assert abs(grads["N.b1"][0] - 2.0) < 1e-9


def test_sgd_moves_weight_toward_target():
    # W=[[2]], b=[0], x=[1], t=[0] → output=2, gradient pushes W down
    net = _net((1, "linear"), name="N", input_dim=1)
    ps = _manual_ps(net, [[2.0]], [0.0])
    ps2, _ = dense_train_step(net, ps, [1.0], [0.0], "mse", learning_rate=0.1)
    assert ps2.parameters["N.W1"]["values"][0][0] < 2.0


def test_softmax_cross_entropy_fused_gradient():
    # Fused gradient: dL/dz = p - y
    # With softmax: for equal pre-activations, p=[1/3,1/3,1/3], y=[1,0,0]
    # dz = [1/3-1, 1/3-0, 1/3-0] = [-2/3, 1/3, 1/3]
    net = _net((3, "softmax"), name="N", input_dim=1)
    W = [[0.0], [0.0], [0.0]]  # zero weights → pre_act = b
    b = [1.0, 1.0, 1.0]        # equal pre_act → uniform softmax
    params = {
        "N.W1": {"function": "N", "role": "weights", "type": "Tensor",
                 "shape": [3, 1], "dtype": "float32", "initializer": "xavier_normal",
                 "values": W, "is_layer": True},
        "N.b1": {"function": "N", "role": "bias", "type": "Vector",
                 "shape": [3], "dtype": "float32", "initializer": "zeros",
                 "values": b, "is_layer": True},
    }
    ps = ParameterSet(parameter_set_id="m", model_hash="h",
                      parameter_schema_hash="s", parameters=params)
    grads = dense_compute_gradients(net, ps, [1.0], [1.0, 0.0, 0.0], "cross_entropy")
    # db should be close to [-2/3, 1/3, 1/3]
    assert abs(grads["N.b1"][0] - (-2.0 / 3.0)) < 1e-6
    assert abs(grads["N.b1"][1] - (1.0 / 3.0)) < 1e-6
    assert abs(grads["N.b1"][2] - (1.0 / 3.0)) < 1e-6


def test_sigmoid_bce_fused_gradient():
    # sigmoid(0)=0.5, target=1.0 → fused dz = 0.5 - 1.0 = -0.5
    net = _net((1, "sigmoid"), name="N", input_dim=1)
    W = [[0.0]]
    b = [0.0]  # pre_act = 0 → sigmoid(0) = 0.5
    params = {
        "N.W1": {"function": "N", "role": "weights", "type": "Tensor",
                 "shape": [1, 1], "dtype": "float32", "initializer": "xavier_normal",
                 "values": W, "is_layer": True},
        "N.b1": {"function": "N", "role": "bias", "type": "Vector",
                 "shape": [1], "dtype": "float32", "initializer": "zeros",
                 "values": b, "is_layer": True},
    }
    ps = ParameterSet(parameter_set_id="m", model_hash="h",
                      parameter_schema_hash="s", parameters=params)
    grads = dense_compute_gradients(net, ps, [1.0], [1.0], "binary_cross_entropy")
    assert abs(grads["N.b1"][0] - (-0.5)) < 1e-9


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

def test_training_converges_mse_regression():
    # Simple regression: net learns to output target from input
    net = _net((4, "relu"), (1, "linear"), input_dim=1)
    ps = _ps(net, seed=1)
    x, t = [2.0], [4.0]
    for _ in range(300):
        ps, _ = dense_train_step(net, ps, x, t, "mse", learning_rate=0.01)
    final_loss = compute_loss("mse", dense_forward(net, ps, x), t)
    assert final_loss < 0.1


def test_training_converges_binary_classification():
    # OR gate: (1,0)→1, (0,1)→1, (0,0)→0 — learn on repeated (1,0)→1
    net = _net((4, "relu"), (1, "sigmoid"), input_dim=2)
    ps = _ps(net, seed=2)
    data = [([1.0, 0.0], [1.0]), ([0.1, 0.9], [1.0]), ([0.0, 0.0], [0.0])]
    for _ in range(200):
        for x, t in data:
            ps, _ = dense_train_step(net, ps, x, t, "binary_cross_entropy", learning_rate=0.05)
    # Check that (1,0) predicts closer to 1 than to 0
    pred = dense_forward(net, ps, [1.0, 0.0])
    assert pred[0] > 0.5


def test_training_converges_multiclass():
    # Learn [1,0]→[1,0,0] and [0,1]→[0,1,0]
    net = _net((8, "relu"), (3, "softmax"), input_dim=2)
    ps = _ps(net, seed=3)
    data = [([1.0, 0.0], [1.0, 0.0, 0.0]), ([0.0, 1.0], [0.0, 1.0, 0.0])]
    for _ in range(200):
        for x, t in data:
            ps, _ = dense_train_step(net, ps, x, t, "cross_entropy", learning_rate=0.05)
    pred1 = dense_forward(net, ps, [1.0, 0.0])
    pred2 = dense_forward(net, ps, [0.0, 1.0])
    # Class 0 should dominate for input [1,0]
    assert pred1[0] > pred1[1] and pred1[0] > pred1[2]
    # Class 1 should dominate for input [0,1]
    assert pred2[1] > pred2[0] and pred2[1] > pred2[2]
