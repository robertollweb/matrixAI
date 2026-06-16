"""P18 C2 — Type system y shape inference para redes densas."""
from __future__ import annotations

import pytest

from matrixai.ir import NetworkTypeResult, check_network_types, check_program_types
from matrixai.ir.schema import DenseLayerSpec, NetworkSpec, VectorSpec
from matrixai.parser.parser import parse_text
from matrixai.types import PROBABILITY, PROBABILITY_MAP, SCALAR, TypeSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vector(name: str, fields: list[str], types: dict | None = None) -> VectorSpec:
    return VectorSpec(name=name, size=len(fields), fields=fields, field_types=types or {})


def _network(
    name: str,
    input_name: str,
    layers: list[tuple[int, str]],
    output: str,
    output_type_str: str,
) -> NetworkSpec:
    dense = [DenseLayerSpec(index=i + 1, units=u, activation=a) for i, (u, a) in enumerate(layers)]
    return NetworkSpec(name=name, input=input_name, layers=dense, output=output, output_type_str=output_type_str)


# ---------------------------------------------------------------------------
# Shape inference tests
# ---------------------------------------------------------------------------

def test_dense_network_shape_inference_regression():
    vec = _vector("House", ["a", "b", "c", "d"])
    net = _network("Reg", "House", [(16, "relu"), (8, "relu"), (1, "linear")], "price", "Scalar")
    result = check_network_types(net, {"House": vec})
    assert result.ok
    shapes = [(l.input_shape, l.output_shape) for l in result.resolved_layers]
    assert shapes[0] == ([4], [16])
    assert shapes[1] == ([16], [8])
    assert shapes[2] == ([8], [1])


def test_dense_network_shape_inference_multiclass():
    vec = _vector("Email", ["a", "b", "c"])
    net = _network("Cls", "Email", [(12, "relu"), (3, "softmax")], "category", "ProbabilityMap[a,b,c]")
    result = check_network_types(net, {"Email": vec})
    assert result.ok
    assert result.resolved_layers[0].input_shape == [3]
    assert result.resolved_layers[0].output_shape == [12]
    assert result.resolved_layers[1].input_shape == [12]
    assert result.resolved_layers[1].output_shape == [3]


def test_dense_network_shape_inference_single_hidden_layer():
    vec = _vector("X", ["x1", "x2"])
    net = _network("Net", "X", [(8, "relu"), (1, "sigmoid")], "y", "Probability")
    result = check_network_types(net, {"X": vec})
    assert result.ok
    assert result.resolved_layers[0].input_shape == [2]
    assert result.resolved_layers[1].input_shape == [8]


# ---------------------------------------------------------------------------
# Output type inference tests
# ---------------------------------------------------------------------------

def test_dense_network_output_type_scalar_from_linear():
    vec = _vector("X", ["a", "b"])
    net = _network("R", "X", [(4, "relu"), (1, "linear")], "y", "Scalar")
    result = check_network_types(net, {"X": vec})
    assert result.ok
    assert result.output_type is not None
    assert result.output_type.name == "Scalar"


def test_dense_network_output_type_probability_from_sigmoid():
    vec = _vector("X", ["a", "b"])
    net = _network("B", "X", [(4, "relu"), (1, "sigmoid")], "y", "Probability")
    result = check_network_types(net, {"X": vec})
    assert result.ok
    assert result.output_type.name == "Probability"


def test_dense_network_output_type_probability_map_from_softmax():
    vec = _vector("X", ["a", "b"])
    net = _network("C", "X", [(4, "relu"), (3, "softmax")], "y", "ProbabilityMap[x,y,z]")
    result = check_network_types(net, {"X": vec})
    assert result.ok
    assert result.output_type.name == "ProbabilityMap"


# ---------------------------------------------------------------------------
# Type rule violation tests
# ---------------------------------------------------------------------------

def test_dense_network_softmax_requires_probability_map():
    vec = _vector("X", ["a"])
    net = _network("Bad", "X", [(4, "relu"), (3, "softmax")], "y", "Scalar")
    result = check_network_types(net, {"X": vec})
    assert not result.ok
    assert any("Scalar" in e and "softmax" in e or "softmax" in e and "linear" in e for e in result.errors)


def test_dense_network_sigmoid_requires_probability_output():
    vec = _vector("X", ["a"])
    net = _network("Bad", "X", [(4, "relu"), (1, "sigmoid")], "y", "Scalar")
    result = check_network_types(net, {"X": vec})
    assert not result.ok
    assert any("Scalar" in e or "sigmoid" in e for e in result.errors)


def test_dense_network_linear_scalar_output_requires_one_unit():
    vec = _vector("X", ["a"])
    net = _network("Bad", "X", [(4, "relu"), (2, "linear")], "y", "Scalar")
    result = check_network_types(net, {"X": vec})
    assert not result.ok
    assert any("units=1" in e or "units=2" in e for e in result.errors)


def test_dense_network_softmax_requires_min_2_units():
    vec = _vector("X", ["a"])
    net = _network("Bad", "X", [(4, "relu"), (1, "softmax")], "y", "ProbabilityMap[a,b]")
    result = check_network_types(net, {"X": vec})
    assert not result.ok
    assert any("units >= 2" in e for e in result.errors)


def test_dense_network_relu_final_rejected_for_probability():
    vec = _vector("X", ["a"])
    net = _network("Bad", "X", [(4, "relu"), (1, "relu")], "y", "Probability")
    result = check_network_types(net, {"X": vec})
    assert not result.ok
    assert any("relu" in e for e in result.errors)


def test_dense_network_relu_final_rejected_for_probability_map():
    vec = _vector("X", ["a"])
    net = _network("Bad", "X", [(4, "relu"), (3, "relu")], "y", "ProbabilityMap[a,b,c]")
    result = check_network_types(net, {"X": vec})
    assert not result.ok
    assert any("relu" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Input vector validation
# ---------------------------------------------------------------------------

def test_dense_network_rejects_non_numeric_vector_fields():
    from matrixai.types import TypeSpec
    vec = _vector("X", ["label", "score"], types={"label": TypeSpec("String"), "score": TypeSpec("Score")})
    net = _network("Bad", "X", [(4, "relu"), (1, "linear")], "y", "Scalar")
    result = check_network_types(net, {"X": vec})
    assert not result.ok
    assert any("non-numeric" in e for e in result.errors)


def test_dense_network_rejects_unknown_input_vector():
    net = _network("Bad", "Missing", [(4, "relu"), (1, "linear")], "y", "Scalar")
    result = check_network_types(net, {})
    assert not result.ok
    assert any("Missing" in e for e in result.errors)


def test_dense_network_relu_hidden_layer_accepted():
    vec = _vector("X", ["a", "b", "c"])
    net = _network("Ok", "X", [(8, "relu"), (4, "relu"), (1, "linear")], "y", "Scalar")
    result = check_network_types(net, {"X": vec})
    assert result.ok


# ---------------------------------------------------------------------------
# Interpretability warning
# ---------------------------------------------------------------------------

def test_dense_network_valid_regression_produces_interpretability_warning():
    vec = _vector("House", ["a", "b", "c", "d"])
    net = _network("R", "House", [(16, "relu"), (1, "linear")], "price", "Scalar")
    result = check_network_types(net, {"House": vec})
    assert result.ok
    assert any("interpretability_level=reduced" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Integration with check_program_types
# ---------------------------------------------------------------------------

def test_dense_network_check_integrated_in_program_typecheck():
    model = """
PROJECT Bad

VECTOR X[1]
  x: Scalar
END

NETWORK BadNet
  INPUT X
  LAYER Dense units=3 activation=softmax
  OUTPUT y: Scalar
END

GRAPH
  X -> BadNet
END
"""
    prog = parse_text(model)
    result = check_program_types(prog)
    assert not result.ok
    assert any("Scalar" in e or "softmax" in e for e in result.errors)


def test_dense_network_valid_regression_has_no_errors():
    model = """
PROJECT Good

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
"""
    prog = parse_text(model)
    result = check_program_types(prog)
    assert result.ok
    assert result.symbols.get("PriceRegressor") is not None
    assert result.symbols["PriceRegressor"].name == "Scalar"


def test_dense_network_valid_multiclass_has_no_errors():
    model = """
PROJECT Good

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
"""
    prog = parse_text(model)
    result = check_program_types(prog)
    assert result.ok
    assert result.symbols.get("EmailClassifier").name == "ProbabilityMap"


def test_dense_network_valid_binary_has_no_errors():
    model = """
PROJECT Good

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
"""
    prog = parse_text(model)
    result = check_program_types(prog)
    assert result.ok
    assert result.symbols.get("FallRiskModel").name == "Probability"
