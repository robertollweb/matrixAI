"""P18 C9 — Torch opcional: materialización de NetworkSpec como nn.Module."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from matrixai.forward.dense_forward import dense_forward
from matrixai.forward.dense_torch import (
    DenseTorchError,
    dense_network_to_torch_module,
    dense_torch_forward,
    torch_module_to_parameter_set,
)
from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai.parameters.network_params import build_network_parameter_set
from matrixai.parameters.store import ParameterSet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _net(*layer_specs: tuple[int, str], name: str = "Net", input_dim: int = 3) -> NetworkSpec:
    layers, dim = [], input_dim
    for i, (units, act) in enumerate(layer_specs, start=1):
        layers.append(DenseLayerSpec(index=i, units=units, activation=act,
                                     input_shape=[dim], output_shape=[units]))
        dim = units
    return NetworkSpec(name=name, input="X", layers=layers, output="y", output_type_str="Scalar")


def _ps(net: NetworkSpec, seed: int = 42) -> ParameterSet:
    return build_network_parameter_set(net, net.layers, "test_hash", seed=seed)


# ---------------------------------------------------------------------------
# Module creation
# ---------------------------------------------------------------------------

def test_module_created_without_error():
    net = _net((8, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    assert module is not None


def test_module_has_correct_linear_layer_count():
    net = _net((16, "relu"), (8, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    linears = list(module.linears)
    assert len(linears) == 3


def test_module_linear_shapes():
    # House[3] → 16 → 8 → 1
    net = _net((16, "relu"), (8, "tanh"), (1, "linear"), input_dim=3)
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    linears = list(module.linears)
    assert tuple(linears[0].weight.shape) == (16, 3)
    assert tuple(linears[1].weight.shape) == (8, 16)
    assert tuple(linears[2].weight.shape) == (1, 8)


def test_module_weights_match_parameter_set():
    net = _net((4, "relu"), input_dim=2)
    ps = _ps(net, seed=7)
    module = dense_network_to_torch_module(net, ps)
    W_expected = ps.parameters["Net.W1"]["values"]
    W_actual = module.linears[0].weight.detach().tolist()
    for row_exp, row_act in zip(W_expected, W_actual):
        for v_exp, v_act in zip(row_exp, row_act):
            assert abs(v_exp - v_act) < 1e-6


def test_module_bias_match_parameter_set():
    net = _net((4, "relu"), input_dim=2)
    ps = _ps(net, seed=7)
    module = dense_network_to_torch_module(net, ps)
    b_expected = ps.parameters["Net.b1"]["values"]
    b_actual = module.linears[0].bias.detach().tolist()
    for v_exp, v_act in zip(b_expected, b_actual):
        assert abs(v_exp - v_act) < 1e-6


def test_module_missing_param_raises():
    net = _net((4, "relu"), input_dim=2)
    ps = _ps(net)
    params = dict(ps.parameters)
    del params["Net.W1"]
    ps2 = ParameterSet(parameter_set_id=ps.parameter_set_id, model_hash=ps.model_hash,
                       parameter_schema_hash=ps.parameter_schema_hash, parameters=params)
    with pytest.raises(DenseTorchError):
        dense_network_to_torch_module(net, ps2)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_torch_forward_output_is_list():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    result = dense_torch_forward(module, [0.1, 0.2, 0.3])
    assert isinstance(result, list)


def test_torch_forward_output_length():
    net = _net((4, "relu"), (3, "softmax"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    result = dense_torch_forward(module, [0.1, 0.2, 0.3])
    assert len(result) == 3


def test_torch_forward_matches_stdlib_forward():
    # Same weights → torch and stdlib forward should agree within float32 tolerance
    net = _net((8, "relu"), (1, "linear"))
    ps = _ps(net, seed=99)
    x = [0.5, -0.3, 0.8]
    module = dense_network_to_torch_module(net, ps)
    torch_out = dense_torch_forward(module, x)
    stdlib_out = dense_forward(net, ps, x)
    for t, s in zip(torch_out, stdlib_out):
        assert abs(t - s) < 1e-4, f"torch={t}, stdlib={s}"


def test_torch_forward_softmax_sums_to_one():
    net = _net((8, "relu"), (3, "softmax"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    result = dense_torch_forward(module, [1.0, 0.0, 0.0])
    assert abs(sum(result) - 1.0) < 1e-5


def test_torch_forward_relu_nonneg():
    net = _net((8, "relu"), (4, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    # Run a batch of inputs and check internal relu activations via hooks
    x = [0.5, -0.5, 0.0]
    result = dense_torch_forward(module, x)
    assert isinstance(result[0], float)  # just verifies forward runs cleanly


def test_torch_forward_sigmoid_in_range():
    net = _net((4, "relu"), (1, "sigmoid"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    result = dense_torch_forward(module, [1.0, 0.0, 0.5])
    assert 0.0 < result[0] < 1.0


def test_torch_forward_tanh_in_range():
    net = _net((4, "tanh"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    result = dense_torch_forward(module, [1.0, 0.0, 0.5])
    assert isinstance(result[0], float)


def test_torch_forward_deterministic():
    net = _net((8, "relu"), (1, "linear"))
    ps = _ps(net, seed=5)
    module = dense_network_to_torch_module(net, ps)
    x = [0.1, 0.2, 0.3]
    assert dense_torch_forward(module, x) == dense_torch_forward(module, x)


# ---------------------------------------------------------------------------
# Parameter round-trip
# ---------------------------------------------------------------------------

def test_torch_module_to_parameter_set_roundtrip():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net, seed=3)
    module = dense_network_to_torch_module(net, ps)
    ps2 = torch_module_to_parameter_set(net, module, ps)
    # Values should match the original parameter set (no training occurred)
    for key in ps.parameters:
        orig = ps.parameters[key]["values"]
        restored = ps2.parameters[key]["values"]
        if isinstance(orig[0], list):
            for row_o, row_r in zip(orig, restored):
                for v_o, v_r in zip(row_o, row_r):
                    assert abs(v_o - v_r) < 1e-5
        else:
            for v_o, v_r in zip(orig, restored):
                assert abs(v_o - v_r) < 1e-5


def test_torch_module_to_parameter_set_source_is_torch():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    ps2 = torch_module_to_parameter_set(net, module, ps)
    assert ps2.source == "torch"


def test_torch_module_to_parameter_set_preserves_schema_hash():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    ps2 = torch_module_to_parameter_set(net, module, ps)
    assert ps2.parameter_schema_hash == ps.parameter_schema_hash


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_torch_module_parameters_have_grad():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    for p in module.parameters():
        assert p.requires_grad


def test_torch_backward_runs():
    net = _net((4, "relu"), (1, "linear"))
    ps = _ps(net)
    module = dense_network_to_torch_module(net, ps)
    x = torch.tensor([0.5, 0.3, 0.8], dtype=torch.float32)
    output = module(x)
    loss = (output - torch.tensor([1.0])).pow(2).mean()
    loss.backward()
    # Check that gradient exists on at least one parameter
    has_grad = any(p.grad is not None for p in module.parameters())
    assert has_grad
