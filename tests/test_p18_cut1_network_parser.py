"""P18 C1 — Parser e IR para NETWORK, LAYER Dense, activaciones y output tipado."""
from __future__ import annotations

import pytest

from matrixai.ir import DenseLayerSpec, NetworkSpec
from matrixai.parser.parser import MatrixAIParseError, parse_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REGRESSION_MODEL = """
PROJECT HousePriceNeuralRegression

VECTOR House[4]
  size_m2: Scalar[0, 1000]
  rooms: Scalar[0, 20]
  age_years: Scalar[0, 200]
  distance_center_km: Scalar[0, 200]
END

NETWORK PriceRegressor
  INPUT House
  LAYER Dense units=16 activation=relu
  LAYER Dense units=8 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT price: Scalar
END

GRAPH
  House -> PriceRegressor
END

AUDIT
  EXPLAIN House -> PriceRegressor
END
"""

_MULTICLASS_MODEL = """
PROJECT EmailNeuralClassifier

VECTOR Email[3]
  urgency: Probability
  sender_trust: Score
  sentiment: Score
END

NETWORK EmailClassifier
  INPUT Email
  LAYER Dense units=12 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT category: ProbabilityMap[support, sales, operations]
END

GRAPH
  Email -> EmailClassifier
END

AUDIT
  EXPLAIN Email -> EmailClassifier
END
"""

_BINARY_MODEL = """
PROJECT FallRiskNeuralBinary

VECTOR Patient[3]
  age_score: Score
  mobility_score: Score
  medication_risk: Risk
END

NETWORK FallRiskModel
  INPUT Patient
  LAYER Dense units=8 activation=relu
  LAYER Dense units=1 activation=sigmoid
  OUTPUT fall_risk: Probability
END

GRAPH
  Patient -> FallRiskModel
END

AUDIT
  EXPLAIN Patient -> FallRiskModel
END
"""


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------

def test_network_parser_accepts_dense_regression_network():
    prog = parse_text(_REGRESSION_MODEL)
    assert len(prog.networks) == 1
    net = prog.networks[0]
    assert net.name == "PriceRegressor"
    assert net.kind == "dense_network"
    assert net.input == "House"
    assert net.output == "price"
    assert net.output_type_str == "Scalar"


def test_network_parser_accepts_dense_classification_network():
    prog = parse_text(_MULTICLASS_MODEL)
    assert len(prog.networks) == 1
    net = prog.networks[0]
    assert net.name == "EmailClassifier"
    assert net.input == "Email"
    assert "ProbabilityMap" in net.output_type_str


def test_network_parser_accepts_dense_binary_network():
    prog = parse_text(_BINARY_MODEL)
    net = prog.networks[0]
    assert net.name == "FallRiskModel"
    assert net.output == "fall_risk"
    assert net.output_type_str == "Probability"


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------

def test_network_parser_rejects_unknown_layer_type():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK Bad
  INPUT X
  LAYER CNN units=16 activation=relu
  OUTPUT y: Scalar
END

GRAPH
  X -> Bad
END
"""
    with pytest.raises(MatrixAIParseError, match="unsupported layer type"):
        parse_text(model)


def test_network_parser_rejects_unknown_activation():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK Bad
  INPUT X
  LAYER Dense units=16 activation=leaky_relu
  OUTPUT y: Scalar
END

GRAPH
  X -> Bad
END
"""
    with pytest.raises(MatrixAIParseError, match="unknown activation"):
        parse_text(model)


def test_network_parser_rejects_missing_input():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK Bad
  LAYER Dense units=8 activation=relu
  OUTPUT y: Scalar
END

GRAPH
  X -> Bad
END
"""
    with pytest.raises(MatrixAIParseError, match="missing INPUT"):
        parse_text(model)


def test_network_parser_rejects_missing_output():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK Bad
  INPUT X
  LAYER Dense units=8 activation=relu
END

GRAPH
  X -> Bad
END
"""
    with pytest.raises(MatrixAIParseError, match="missing OUTPUT"):
        parse_text(model)


def test_network_parser_rejects_missing_layers():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK Bad
  INPUT X
  OUTPUT y: Scalar
END

GRAPH
  X -> Bad
END
"""
    with pytest.raises(MatrixAIParseError, match="at least one LAYER"):
        parse_text(model)


def test_network_parser_rejects_zero_units():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK Bad
  INPUT X
  LAYER Dense units=0 activation=relu
  OUTPUT y: Scalar
END

GRAPH
  X -> Bad
END
"""
    with pytest.raises(MatrixAIParseError, match="units must be > 0"):
        parse_text(model)


def test_network_parser_rejects_multiple_inputs():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK Bad
  INPUT X
  INPUT X
  LAYER Dense units=8 activation=relu
  OUTPUT y: Scalar
END

GRAPH
  X -> Bad
END
"""
    with pytest.raises(MatrixAIParseError, match="exactly one INPUT"):
        parse_text(model)


# ---------------------------------------------------------------------------
# IR structure tests
# ---------------------------------------------------------------------------

def test_network_ir_has_correct_layer_count():
    prog = parse_text(_REGRESSION_MODEL)
    net = prog.networks[0]
    assert len(net.layers) == 3


def test_network_ir_layer_indices_are_sequential():
    prog = parse_text(_REGRESSION_MODEL)
    net = prog.networks[0]
    assert [layer.index for layer in net.layers] == [1, 2, 3]


def test_network_ir_layer_units_and_activations():
    prog = parse_text(_REGRESSION_MODEL)
    layers = prog.networks[0].layers
    assert layers[0].units == 16 and layers[0].activation == "relu"
    assert layers[1].units == 8 and layers[1].activation == "relu"
    assert layers[2].units == 1 and layers[2].activation == "linear"


def test_network_ir_output_name_and_type():
    prog = parse_text(_REGRESSION_MODEL)
    net = prog.networks[0]
    assert net.output == "price"
    assert net.output_type_str == "Scalar"


def test_network_program_contains_networks_list():
    prog = parse_text(_REGRESSION_MODEL)
    assert hasattr(prog, "networks")
    assert isinstance(prog.networks, list)
    assert all(isinstance(n, NetworkSpec) for n in prog.networks)


def test_network_graph_node_type_is_dense_network():
    prog = parse_text(_REGRESSION_MODEL)
    assert prog.graph.node_types.get("PriceRegressor") == "dense_network"


def test_network_to_dict_serializes_layers():
    prog = parse_text(_REGRESSION_MODEL)
    d = prog.to_dict()
    assert "networks" in d
    net_d = d["networks"][0]
    assert net_d["kind"] == "dense_network"
    assert net_d["name"] == "PriceRegressor"
    assert len(net_d["layers"]) == 3
    assert net_d["layers"][0]["type"] == "Dense"
    assert net_d["layers"][0]["units"] == 16
    assert net_d["layers"][0]["activation"] == "relu"


def test_network_parser_accepts_tanh_activation():
    model = """
PROJECT TanhNet

VECTOR X[2]
  a: Scalar
  b: Scalar
END

NETWORK TanhModel
  INPUT X
  LAYER Dense units=4 activation=tanh
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  X -> TanhModel
END
"""
    prog = parse_text(model)
    assert prog.networks[0].layers[0].activation == "tanh"


def test_network_dense_layer_spec_type():
    prog = parse_text(_REGRESSION_MODEL)
    for layer in prog.networks[0].layers:
        assert isinstance(layer, DenseLayerSpec)
