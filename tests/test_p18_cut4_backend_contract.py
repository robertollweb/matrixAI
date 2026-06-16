"""P18 C4 — BackendContractAnalyzer y DifferentiabilityVerifier para dense_network."""
from __future__ import annotations

import pytest

from matrixai.compiler import BackendContractAnalyzer
from matrixai.ir import DenseLayerSpec, GraphSpec, MatrixAIProgram, NetworkSpec, VectorSpec
from matrixai.parser.parser import parse_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REGRESSION_PROGRAM = """
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

_MULTICLASS_PROGRAM = """
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

_TWO_NETWORKS_PROGRAM = """
PROJECT MultiNetworkProject

VECTOR House[4]
  size_m2: Scalar[0, 1000]
  rooms: Scalar[0, 20]
  age_years: Scalar[0, 200]
  distance_center_km: Scalar[0, 200]
END

VECTOR Email[3]
  urgency: Probability
  sender_trust: Score
  sentiment: Score
END

NETWORK PriceRegressor
  INPUT House
  LAYER Dense units=16 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT price: Scalar
END

NETWORK EmailClassifier
  INPUT Email
  LAYER Dense units=12 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT category: ProbabilityMap[support, sales, operations]
END

GRAPH
  House -> PriceRegressor
  Email -> EmailClassifier
END

AUDIT
  EXPLAIN House -> PriceRegressor
END
"""


def _regression_program():
    return parse_text(_REGRESSION_PROGRAM)


def _multiclass_program():
    return parse_text(_MULTICLASS_PROGRAM)


# ---------------------------------------------------------------------------
# BackendNodeReport for dense_network
# ---------------------------------------------------------------------------

def test_dense_network_node_is_supported():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    node = next(n for n in report.nodes if n.node == "PriceRegressor")
    assert node.supported


def test_dense_network_node_is_differentiable():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    node = next(n for n in report.nodes if n.node == "PriceRegressor")
    assert node.differentiable


def test_dense_network_node_type():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    node = next(n for n in report.nodes if n.node == "PriceRegressor")
    assert node.node_type == "dense_network"


def test_dense_network_kind():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    node = next(n for n in report.nodes if n.node == "PriceRegressor")
    assert node.kind == "dense_network"


def test_dense_network_output_shape_scalar():
    # last layer units=1 → output_shape=(1,)
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    node = next(n for n in report.nodes if n.node == "PriceRegressor")
    assert node.output_shape == (1,)


def test_dense_network_output_shape_multiclass():
    # last layer units=3 → output_shape=(3,)
    program = _multiclass_program()
    report = BackendContractAnalyzer().analyze(program)
    node = next(n for n in report.nodes if n.node == "EmailClassifier")
    assert node.output_shape == (3,)


def test_dense_network_appears_in_differentiable_nodes():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    diff_names = [n.node for n in report.differentiable_nodes]
    assert "PriceRegressor" in diff_names


# ---------------------------------------------------------------------------
# Trainable parameters
# ---------------------------------------------------------------------------

def test_dense_network_generates_w_and_b_per_layer():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    param_names = {p.name for p in report.trainable_parameters}
    assert param_names == {"W1", "b1", "W2", "b2", "W3", "b3"}


def test_dense_network_param_count():
    # 3 layers × 2 = 6
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    net_params = [p for p in report.trainable_parameters if p.function == "PriceRegressor"]
    assert len(net_params) == 6


def test_dense_network_weight_roles():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    for p in report.trainable_parameters:
        if p.name.startswith("W"):
            assert p.role == "weights"
        else:
            assert p.role == "bias"


def test_dense_network_weight_shapes():
    # House[4]: W1=[16,4], W2=[8,16], W3=[1,8]
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    params = {p.name: p for p in report.trainable_parameters}
    assert params["W1"].shape == (16, 4)
    assert params["W2"].shape == (8, 16)
    assert params["W3"].shape == (1, 8)


def test_dense_network_bias_shapes():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    params = {p.name: p for p in report.trainable_parameters}
    assert params["b1"].shape == (16,)
    assert params["b2"].shape == (8,)
    assert params["b3"].shape == (1,)


def test_dense_network_param_paths():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    paths = {p.path for p in report.trainable_parameters}
    assert "PriceRegressor.W1" in paths
    assert "PriceRegressor.b1" in paths
    assert "PriceRegressor.W3" in paths
    assert "PriceRegressor.b3" in paths


def test_dense_network_param_function_field():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    for p in report.trainable_parameters:
        assert p.function == "PriceRegressor"


# ---------------------------------------------------------------------------
# Interpretability warning
# ---------------------------------------------------------------------------

def test_dense_network_emits_interpretability_warning():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    assert any("interpretability_level=reduced" in w for w in report.warnings)


def test_dense_network_interpretability_warning_mentions_network():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    assert any("dense network" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Layer manifest
# ---------------------------------------------------------------------------

def test_dense_network_in_layer_manifest():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    manifest_networks = {e["network"] for e in report.layer_manifest if "network" in e}
    assert "PriceRegressor" in manifest_networks


def test_dense_network_layer_manifest_entry_count():
    # 3 layers → 3 entries
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    net_entries = [e for e in report.layer_manifest if e.get("network") == "PriceRegressor"]
    assert len(net_entries) == 3


def test_dense_network_layer_manifest_param_shapes():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    net_entries = [e for e in report.layer_manifest if e.get("network") == "PriceRegressor"]
    layer1 = next(e for e in net_entries if e["layer_index"] == 1)
    w1 = next(p for p in layer1["parameters"] if p["name"] == "W1")
    assert w1["shape"] == [16, 4]


def test_dense_network_layer_manifest_relu_initializer():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    net_entries = [e for e in report.layer_manifest if e.get("network") == "PriceRegressor"]
    layer1 = next(e for e in net_entries if e["layer_index"] == 1)
    w1 = next(p for p in layer1["parameters"] if p["name"] == "W1")
    assert w1["initializer"] == "he_normal"


def test_dense_network_layer_manifest_bias_zeros():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    net_entries = [e for e in report.layer_manifest if e.get("network") == "PriceRegressor"]
    for entry in net_entries:
        b = next(p for p in entry["parameters"] if p["name"].startswith("b"))
        assert b["initializer"] == "zeros"


# ---------------------------------------------------------------------------
# Report-level integrity
# ---------------------------------------------------------------------------

def test_report_ok_with_dense_network_only():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    assert report.ok


def test_two_dense_networks_both_supported():
    program = parse_text(_TWO_NETWORKS_PROGRAM)
    report = BackendContractAnalyzer().analyze(program)
    nodes = {n.node: n for n in report.nodes if n.node_type == "dense_network"}
    assert "PriceRegressor" in nodes
    assert "EmailClassifier" in nodes
    assert nodes["PriceRegressor"].supported
    assert nodes["EmailClassifier"].supported


def test_two_dense_networks_param_count():
    # PriceRegressor: 2 layers × 2 = 4; EmailClassifier: 2 layers × 2 = 4 → total 8 + 2 vectors
    program = parse_text(_TWO_NETWORKS_PROGRAM)
    report = BackendContractAnalyzer().analyze(program)
    net_params = [p for p in report.trainable_parameters]
    # 4 params for PriceRegressor + 4 for EmailClassifier
    assert len(net_params) == 8


def test_dense_network_to_dict_structure():
    program = _regression_program()
    report = BackendContractAnalyzer().analyze(program)
    d = report.to_dict()
    assert "nodes" in d
    assert "trainable_parameters" in d
    assert "layer_manifest" in d
    net_node = next(n for n in d["nodes"] if n["node"] == "PriceRegressor")
    assert net_node["node_type"] == "dense_network"
    assert net_node["supported"] is True
    assert net_node["differentiable"] is True
