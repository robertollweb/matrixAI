"""P18 C3 — Parameter manifest por capa y ParameterSet compatible."""
from __future__ import annotations

import json
import copy

import pytest

from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai.parameters.network_params import (
    build_network_parameter_set,
    network_parameter_manifest,
    network_parameter_schema_hash,
    validate_network_parameter_set,
)
from matrixai.parameters.store import ParameterSet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _layers(*specs: tuple[int, str]) -> list[DenseLayerSpec]:
    """Build shape-resolved DenseLayerSpec list from (units, activation) tuples."""
    result = []
    input_dim = 4  # simula vector de 4 campos
    for i, (units, activation) in enumerate(specs, start=1):
        result.append(DenseLayerSpec(
            index=i,
            units=units,
            activation=activation,
            input_shape=[input_dim],
            output_shape=[units],
        ))
        input_dim = units
    return result


def _net(name: str = "PriceRegressor") -> NetworkSpec:
    layers = [
        DenseLayerSpec(index=1, units=16, activation="relu", input_shape=[4], output_shape=[16]),
        DenseLayerSpec(index=2, units=8, activation="relu", input_shape=[16], output_shape=[8]),
        DenseLayerSpec(index=3, units=1, activation="linear", input_shape=[8], output_shape=[1]),
    ]
    return NetworkSpec(name=name, input="House", layers=layers, output="price", output_type_str="Scalar")


_MODEL_HASH = "mxai_test1234"

# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------

def test_network_parameter_manifest_has_correct_entry_count():
    net = _net()
    manifest = network_parameter_manifest(net.name, net.layers)
    # 3 layers × 2 params (W + b) = 6
    assert len(manifest) == 6


def test_network_parameter_manifest_contains_all_layers():
    net = _net()
    manifest = network_parameter_manifest(net.name, net.layers)
    paths = [e["path"] for e in manifest]
    assert "PriceRegressor.W1" in paths
    assert "PriceRegressor.b1" in paths
    assert "PriceRegressor.W2" in paths
    assert "PriceRegressor.b2" in paths
    assert "PriceRegressor.W3" in paths
    assert "PriceRegressor.b3" in paths


def test_network_parameter_manifest_weights_shape():
    # Layer 1: units=16, input_dim=4 → W1 shape=[16, 4]
    net = _net()
    manifest = network_parameter_manifest(net.name, net.layers)
    w1 = next(e for e in manifest if e["name"] == "W1")
    assert w1["shape"] == [16, 4]


def test_network_parameter_manifest_bias_shape():
    net = _net()
    manifest = network_parameter_manifest(net.name, net.layers)
    b1 = next(e for e in manifest if e["name"] == "b1")
    assert b1["shape"] == [16]


def test_network_parameter_manifest_qualified_names():
    net = _net()
    manifest = network_parameter_manifest(net.name, net.layers)
    for entry in manifest:
        assert entry["path"] == f"PriceRegressor.{entry['name']}"
        assert entry["function"] == "PriceRegressor"


def test_network_parameter_manifest_relu_uses_he_initializer():
    layers = _layers((16, "relu"), (1, "linear"))
    manifest = network_parameter_manifest("Net", layers)
    w1 = next(e for e in manifest if e["name"] == "W1")
    assert w1["initializer"] == "he_normal"


def test_network_parameter_manifest_sigmoid_uses_xavier_initializer():
    layers = _layers((8, "relu"), (1, "sigmoid"))
    manifest = network_parameter_manifest("Net", layers)
    w2 = next(e for e in manifest if e["name"] == "W2")
    assert w2["initializer"] == "xavier_normal"


def test_network_parameter_manifest_bias_always_zeros():
    net = _net()
    manifest = network_parameter_manifest(net.name, net.layers)
    for entry in manifest:
        if entry["role"] == "bias":
            assert entry["initializer"] == "zeros"


# ---------------------------------------------------------------------------
# Schema hash sensitivity
# ---------------------------------------------------------------------------

def test_network_parameter_schema_hash_changes_when_units_change():
    layers_a = _layers((16, "relu"), (1, "linear"))
    layers_b = _layers((32, "relu"), (1, "linear"))  # different units
    h_a = network_parameter_schema_hash("Net", layers_a)
    h_b = network_parameter_schema_hash("Net", layers_b)
    assert h_a != h_b


def test_network_parameter_schema_hash_changes_when_activation_changes():
    # sigmoid → xavier, relu → he: different initializer → different hash
    layers_relu = _layers((8, "relu"), (1, "linear"))
    layers_sigmoid = _layers((8, "sigmoid"), (1, "linear"))
    h_relu = network_parameter_schema_hash("Net", layers_relu)
    h_sigmoid = network_parameter_schema_hash("Net", layers_sigmoid)
    assert h_relu != h_sigmoid


def test_network_parameter_schema_hash_is_stable():
    net = _net()
    h1 = network_parameter_schema_hash(net.name, net.layers)
    h2 = network_parameter_schema_hash(net.name, net.layers)
    assert h1 == h2


# ---------------------------------------------------------------------------
# ParameterSet build
# ---------------------------------------------------------------------------

def test_build_network_parameter_set_structure():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    assert ps.model_hash == _MODEL_HASH
    assert ps.parameter_schema_hash.startswith("params_")
    assert ps.source == "initial"
    assert len(ps.parameters) == 6


def test_build_network_parameter_set_weights_not_zero_for_he():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH, seed=1)
    w1_values = ps.parameters["PriceRegressor.W1"]["values"]
    flat = [v for row in w1_values for v in row]
    assert any(abs(v) > 1e-6 for v in flat), "He weights should not all be zero"


def test_build_network_parameter_set_bias_is_zeros():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    for key, param in ps.parameters.items():
        if param["role"] == "bias":
            assert all(v == 0.0 for v in param["values"]), f"{key} bias not zero"


def test_build_network_parameter_set_weight_shape_correct():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    w1 = ps.parameters["PriceRegressor.W1"]
    assert w1["shape"] == [16, 4]
    assert len(w1["values"]) == 16
    assert len(w1["values"][0]) == 4


def test_build_network_parameter_set_is_deterministic():
    net = _net()
    ps1 = build_network_parameter_set(net, net.layers, _MODEL_HASH, seed=99)
    ps2 = build_network_parameter_set(net, net.layers, _MODEL_HASH, seed=99)
    assert ps1.parameters["PriceRegressor.W1"]["values"] == ps2.parameters["PriceRegressor.W1"]["values"]


def test_build_network_parameter_set_roundtrip():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    d = ps.to_dict()
    ps2 = ParameterSet.from_dict(d)
    assert ps2.parameter_schema_hash == ps.parameter_schema_hash
    assert ps2.parameters["PriceRegressor.W1"]["values"] == ps.parameters["PriceRegressor.W1"]["values"]
    assert ps2.parameters["PriceRegressor.b1"]["values"] == ps.parameters["PriceRegressor.b1"]["values"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_network_parameter_set_accepts_valid():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    result = validate_network_parameter_set(net, net.layers, ps, _MODEL_HASH)
    assert result.ok


def test_validate_network_parameter_set_rejects_missing_layer_weight():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    params = dict(ps.parameters)
    del params["PriceRegressor.W2"]
    ps2 = ParameterSet(
        parameter_set_id=ps.parameter_set_id,
        model_hash=ps.model_hash,
        parameter_schema_hash=ps.parameter_schema_hash,
        parameters=params,
    )
    result = validate_network_parameter_set(net, net.layers, ps2, _MODEL_HASH)
    assert not result.ok
    assert any("W2" in e for e in result.errors)


def test_validate_network_parameter_set_rejects_wrong_shape():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    params = dict(ps.parameters)
    bad_w1 = dict(params["PriceRegressor.W1"])
    bad_w1["shape"] = [8, 4]  # wrong units
    bad_w1["values"] = [[0.0] * 4 for _ in range(8)]
    params["PriceRegressor.W1"] = bad_w1
    ps2 = ParameterSet(
        parameter_set_id=ps.parameter_set_id,
        model_hash=ps.model_hash,
        parameter_schema_hash=ps.parameter_schema_hash,
        parameters=params,
    )
    result = validate_network_parameter_set(net, net.layers, ps2, _MODEL_HASH)
    assert not result.ok
    assert any("W1" in e for e in result.errors)


def test_validate_network_parameter_set_rejects_wrong_model_hash():
    net = _net()
    ps = build_network_parameter_set(net, net.layers, _MODEL_HASH)
    result = validate_network_parameter_set(net, net.layers, ps, "mxai_wrong")
    assert not result.ok
    assert any("model_hash" in e for e in result.errors)
