"""P18 C5 — Forward stdlib para redes densas."""
from __future__ import annotations

import math

import pytest

from matrixai.forward import DenseForwardError, DenseForwardTrace, dense_forward, dense_forward_trace
from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai.parameters.network_params import build_network_parameter_set
from matrixai.parameters.store import ParameterSet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _net(*layer_specs: tuple[int, str], name: str = "Net", input_dim: int = 3) -> NetworkSpec:
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


def _manual_ps(net: NetworkSpec, W_values: list, b_values: list) -> ParameterSet:
    """Build a ParameterSet with manually specified W and b for a single-layer net."""
    params = {
        f"{net.name}.W1": {"function": net.name, "role": "weights", "type": "Tensor",
                           "shape": [len(W_values), len(W_values[0])], "dtype": "float32",
                           "initializer": "he_normal", "values": W_values, "is_layer": True},
        f"{net.name}.b1": {"function": net.name, "role": "bias", "type": "Vector",
                           "shape": [len(b_values)], "dtype": "float32",
                           "initializer": "zeros", "values": b_values, "is_layer": True},
    }
    return ParameterSet(
        parameter_set_id="manual",
        model_hash="test_hash",
        parameter_schema_hash="params_manual",
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Basic output shape and type
# ---------------------------------------------------------------------------

def test_dense_forward_output_is_list():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    result = dense_forward(net, ps, [0.1, 0.2, 0.3])
    assert isinstance(result, list)


def test_dense_forward_output_shape_matches_last_layer():
    net = _net((8, "relu"), (3, "softmax"))
    ps = _ps(net)
    result = dense_forward(net, ps, [0.1, 0.2, 0.3])
    assert len(result) == 3


def test_dense_forward_scalar_output_length_one():
    net = _net((16, "relu"), (1, "linear"))
    ps = _ps(net)
    result = dense_forward(net, ps, [1.0, 0.0, 0.5])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Activation correctness
# ---------------------------------------------------------------------------

def test_dense_forward_relu_non_negative():
    net = _net((8, "relu"), (2, "linear"))
    ps = _ps(net)
    # Capture internal relu layer activations via trace
    trace = dense_forward_trace(net, ps, [0.1, -0.5, 0.3])
    relu_out = trace.activations[1]  # after layer 1 (relu)
    assert all(v >= 0.0 for v in relu_out)


def test_dense_forward_sigmoid_in_range():
    net = _net((4, "sigmoid"), (1, "linear"))
    ps = _ps(net)
    trace = dense_forward_trace(net, ps, [1.0, 2.0, -1.0])
    sigmoid_out = trace.activations[1]
    assert all(0.0 < v < 1.0 for v in sigmoid_out)


def test_dense_forward_softmax_sums_to_one():
    net = _net((8, "relu"), (3, "softmax"))
    ps = _ps(net)
    result = dense_forward(net, ps, [0.1, 0.2, 0.3])
    assert abs(sum(result) - 1.0) < 1e-9


def test_dense_forward_softmax_all_positive():
    net = _net((4, "relu"), (4, "softmax"))
    ps = _ps(net)
    result = dense_forward(net, ps, [0.5, 0.5, 0.5])
    assert all(v > 0.0 for v in result)


def test_dense_forward_tanh_in_range():
    net = _net((4, "tanh"), (1, "linear"))
    ps = _ps(net)
    trace = dense_forward_trace(net, ps, [1.0, -1.0, 0.0])
    tanh_out = trace.activations[1]
    assert all(-1.0 < v < 1.0 for v in tanh_out)


def test_dense_forward_linear_is_identity_for_zero_weights():
    # W=identity-ish, b=0: linear activation leaves values unchanged
    net = _net((2, "linear"), name="N", input_dim=2)
    W = [[1.0, 0.0], [0.0, 1.0]]
    b = [0.0, 0.0]
    ps = _manual_ps(net, W, b)
    result = dense_forward(net, ps, [3.0, -2.0])
    assert abs(result[0] - 3.0) < 1e-9
    assert abs(result[1] - (-2.0)) < 1e-9


# ---------------------------------------------------------------------------
# Known-value computation
# ---------------------------------------------------------------------------

def test_dense_forward_known_values_single_layer():
    # W = [[2.0, 0.0], [0.0, 3.0]], b = [1.0, -1.0], relu
    # input = [1.0, 1.0]
    # pre_act = [2*1+0*1+1, 0*1+3*1-1] = [3, 2]
    # relu → [3, 2]
    net = _net((2, "relu"), name="N", input_dim=2)
    W = [[2.0, 0.0], [0.0, 3.0]]
    b = [1.0, -1.0]
    ps = _manual_ps(net, W, b)
    result = dense_forward(net, ps, [1.0, 1.0])
    assert abs(result[0] - 3.0) < 1e-9
    assert abs(result[1] - 2.0) < 1e-9


def test_dense_forward_relu_clips_negative():
    # W = [[-1.0, 0.0]], b = [0.0], relu
    # input = [2.0, 0.0] → pre_act = [-2.0] → relu → [0.0]
    net = _net((1, "relu"), name="N", input_dim=2)
    W = [[-1.0, 0.0]]
    b = [0.0]
    ps = _manual_ps(net, W, b)
    result = dense_forward(net, ps, [2.0, 0.0])
    assert result[0] == 0.0


# ---------------------------------------------------------------------------
# DenseForwardTrace
# ---------------------------------------------------------------------------

def test_dense_forward_trace_returns_trace_object():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    trace = dense_forward_trace(net, ps, [0.1, 0.2, 0.3])
    assert isinstance(trace, DenseForwardTrace)


def test_dense_forward_trace_activations_length():
    # activations[0]=input, [1]=after layer1, [2]=after layer2
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    trace = dense_forward_trace(net, ps, [0.1, 0.2, 0.3])
    assert len(trace.activations) == 3  # input + 2 layers


def test_dense_forward_trace_pre_activations_length():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    trace = dense_forward_trace(net, ps, [0.1, 0.2, 0.3])
    assert len(trace.pre_activations) == 2  # one per layer


def test_dense_forward_trace_input_preserved():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    x = [0.1, 0.2, 0.3]
    trace = dense_forward_trace(net, ps, x)
    assert trace.activations[0] == x


def test_dense_forward_trace_output_matches_dense_forward():
    net = _net((8, "relu"), (3, "softmax"))
    ps = _ps(net, seed=7)
    x = [0.1, 0.2, 0.3]
    trace = dense_forward_trace(net, ps, x)
    direct = dense_forward(net, ps, x)
    assert trace.output == direct


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_dense_forward_is_deterministic():
    net = _net((16, "relu"), (8, "relu"), (1, "linear"))
    ps = _ps(net, seed=99)
    x = [0.5, -0.3, 0.1]
    r1 = dense_forward(net, ps, x)
    r2 = dense_forward(net, ps, x)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_dense_forward_wrong_input_dim_raises():
    net = _net((4, "relu"), input_dim=3)
    ps = _ps(net)
    with pytest.raises(DenseForwardError):
        dense_forward(net, ps, [1.0, 2.0])  # expects dim=3


def test_dense_forward_missing_param_raises():
    net = _net((4, "relu"), input_dim=3)
    ps = _ps(net)
    params = dict(ps.parameters)
    del params["Net.W1"]
    ps2 = ParameterSet(
        parameter_set_id=ps.parameter_set_id,
        model_hash=ps.model_hash,
        parameter_schema_hash=ps.parameter_schema_hash,
        parameters=params,
    )
    with pytest.raises(DenseForwardError):
        dense_forward(net, ps2, [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_dense_forward_sigmoid_large_positive():
    # sigmoid(100) should be ~1.0 without overflow
    net = _net((1, "sigmoid"), name="N", input_dim=1)
    W = [[100.0]]
    b = [0.0]
    ps = _manual_ps(net, W, b)
    result = dense_forward(net, ps, [1.0])
    assert 0.999 < result[0] <= 1.0


def test_dense_forward_sigmoid_large_negative():
    net = _net((1, "sigmoid"), name="N", input_dim=1)
    W = [[-100.0]]
    b = [0.0]
    ps = _manual_ps(net, W, b)
    result = dense_forward(net, ps, [1.0])
    assert 0.0 <= result[0] < 0.001


def test_dense_forward_softmax_uniform_for_equal_inputs():
    # All pre_act equal → all softmax values equal → 1/n
    net = _net((3, "softmax"), name="N", input_dim=1)
    W = [[1.0], [1.0], [1.0]]
    b = [0.0, 0.0, 0.0]
    ps = _manual_ps(net, W, b)
    result = dense_forward(net, ps, [5.0])
    for v in result:
        assert abs(v - 1.0 / 3.0) < 1e-9
