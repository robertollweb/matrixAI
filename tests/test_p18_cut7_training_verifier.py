"""P18 C7 — TrainingVerifier + DifferentiabilityVerifier para dense_network."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from matrixai.compiler import BackendContractAnalyzer
from matrixai.ir.schema import DenseLayerSpec, GraphSpec, MatrixAIProgram, NetworkSpec, VectorSpec
from matrixai.parser.parser import parse_text
from matrixai.training.differentiability import DifferentiabilityVerifier
from matrixai.training.parser import parse_training_text
from matrixai.training.verifier import TrainingVerifier

# ---------------------------------------------------------------------------
# Helpers — build minimal programs and specs in memory
# ---------------------------------------------------------------------------

_REGRESSION_MXAI = textwrap.dedent("""\
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
""")

_MULTICLASS_MXAI = textwrap.dedent("""\
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
""")

_BINARY_MXAI = textwrap.dedent("""\
    PROJECT SpamClassifier

    VECTOR Email[2]
      length: Scalar[0, 10000]
      link_count: Scalar[0, 100]
    END

    NETWORK SpamNet
      INPUT Email
      LAYER Dense units=8 activation=relu
      LAYER Dense units=1 activation=sigmoid
      OUTPUT spam_prob: Probability
    END

    GRAPH
      Email -> SpamNet
    END

    AUDIT
      EXPLAIN Email -> SpamNet
    END
""")

_REGRESSION_MXTRAIN = textwrap.dedent("""\
    MODEL {model_path}

    DATASET HouseDataset
      SOURCE csv("{dataset_path}")
      INPUT House FROM COLUMNS [size_m2, rooms, age_years, distance_center_km]
      TARGET price: Scalar
    END

    LOSS HouseLoss
      TYPE mse
      PREDICTION price
      TARGET price
    END

    OPTIMIZER HouseOptimizer
      TYPE sgd
      LEARNING_RATE 0.01
      UPDATE PriceRegressor.*
    END

    RUN
      EPOCHS 10
    END
""")

_MULTICLASS_MXTRAIN = textwrap.dedent("""\
    MODEL {model_path}

    DATASET EmailDataset
      SOURCE csv("{dataset_path}")
      INPUT Email FROM COLUMNS [urgency, sender_trust, sentiment]
      TARGET category: Label[support, sales, operations]
    END

    LOSS EmailLoss
      TYPE cross_entropy
      PREDICTION category
      TARGET category
    END

    OPTIMIZER EmailOptimizer
      TYPE sgd
      LEARNING_RATE 0.05
      UPDATE EmailClassifier.*
    END

    RUN
      EPOCHS 10
    END
""")

_BINARY_MXTRAIN = textwrap.dedent("""\
    MODEL {model_path}

    DATASET SpamDataset
      SOURCE csv("{dataset_path}")
      INPUT Email FROM COLUMNS [length, link_count]
      TARGET spam_prob: Probability
    END

    LOSS SpamLoss
      TYPE binary_cross_entropy
      PREDICTION spam_prob
      TARGET spam_prob
    END

    OPTIMIZER SpamOptimizer
      TYPE sgd
      LEARNING_RATE 0.05
      UPDATE SpamNet.*
    END

    RUN
      EPOCHS 10
    END
""")

_REGRESSION_CSV = "size_m2,rooms,age_years,distance_center_km,price\n500,3,10,5,250000\n300,2,20,10,150000\n"
_MULTICLASS_CSV = "urgency,sender_trust,sentiment,category\n0.9,0.1,0.2,support\n0.1,0.8,0.7,sales\n"
_BINARY_CSV = "length,link_count,spam_prob\n200,5,0.9\n50,0,0.1\n"


def _write_model(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / f"{name}.mxai"
    p.write_text(content)
    return p


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / f"{name}.csv"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# _symbol_type — looks up network output types
# ---------------------------------------------------------------------------

def test_symbol_type_finds_scalar_for_network_output():
    program = parse_text(_REGRESSION_MXAI)
    verifier = TrainingVerifier()
    t = verifier._symbol_type(program, "price")
    assert t is not None
    assert t.name == "Scalar"


def test_symbol_type_finds_probabilitymap_for_softmax_network():
    program = parse_text(_MULTICLASS_MXAI)
    verifier = TrainingVerifier()
    t = verifier._symbol_type(program, "category")
    assert t is not None
    assert t.name == "ProbabilityMap"


def test_symbol_type_finds_probability_for_sigmoid_network():
    program = parse_text(_BINARY_MXAI)
    verifier = TrainingVerifier()
    t = verifier._symbol_type(program, "spam_prob")
    assert t is not None
    assert t.name == "Probability"


def test_symbol_type_returns_none_for_unknown():
    program = parse_text(_REGRESSION_MXAI)
    verifier = TrainingVerifier()
    assert verifier._symbol_type(program, "nonexistent") is None


def test_symbol_type_also_matches_by_network_name():
    program = parse_text(_REGRESSION_MXAI)
    verifier = TrainingVerifier()
    t = verifier._symbol_type(program, "PriceRegressor")
    assert t is not None
    assert t.name == "Scalar"


# ---------------------------------------------------------------------------
# DifferentiabilityVerifier._prediction_node for dense_network
# ---------------------------------------------------------------------------

def test_prediction_node_finds_regression_network():
    program = parse_text(_REGRESSION_MXAI)
    verifier = DifferentiabilityVerifier()
    node = verifier._prediction_node(program, "price")
    assert node == "PriceRegressor"


def test_prediction_node_finds_multiclass_network():
    program = parse_text(_MULTICLASS_MXAI)
    verifier = DifferentiabilityVerifier()
    node = verifier._prediction_node(program, "category")
    assert node == "EmailClassifier"


def test_prediction_node_also_matches_network_name():
    program = parse_text(_REGRESSION_MXAI)
    verifier = DifferentiabilityVerifier()
    node = verifier._prediction_node(program, "PriceRegressor")
    assert node == "PriceRegressor"


def test_prediction_node_returns_empty_for_unknown():
    program = parse_text(_REGRESSION_MXAI)
    verifier = DifferentiabilityVerifier()
    assert verifier._prediction_node(program, "nonexistent") == ""


# ---------------------------------------------------------------------------
# DifferentiabilityVerifier.verify for dense_network
# ---------------------------------------------------------------------------

def test_differentiability_verifier_no_errors_for_regression_network():
    program = parse_text(_REGRESSION_MXAI)
    training = parse_training_text(
        _REGRESSION_MXTRAIN.format(model_path="dummy.mxai", dataset_path="dummy.csv")
    )
    backend_report = BackendContractAnalyzer().analyze(program)
    result = DifferentiabilityVerifier().verify(training, program, backend_report)
    assert result.ok, result.errors


def test_differentiability_verifier_prediction_node_is_network():
    program = parse_text(_REGRESSION_MXAI)
    training = parse_training_text(
        _REGRESSION_MXTRAIN.format(model_path="dummy.mxai", dataset_path="dummy.csv")
    )
    backend_report = BackendContractAnalyzer().analyze(program)
    result = DifferentiabilityVerifier().verify(training, program, backend_report)
    assert result.prediction_node == "PriceRegressor"


def test_differentiability_verifier_parameter_paths_populated():
    program = parse_text(_REGRESSION_MXAI)
    training = parse_training_text(
        _REGRESSION_MXTRAIN.format(model_path="dummy.mxai", dataset_path="dummy.csv")
    )
    backend_report = BackendContractAnalyzer().analyze(program)
    result = DifferentiabilityVerifier().verify(training, program, backend_report)
    assert len(result.parameter_paths) > 0


# ---------------------------------------------------------------------------
# TrainingVerifier.verify — full integration with files
# ---------------------------------------------------------------------------

def test_verifier_accepts_mse_dense_network(tmp_path):
    model_path = _write_model(tmp_path, "house", _REGRESSION_MXAI)
    dataset_path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
    training = parse_training_text(
        _REGRESSION_MXTRAIN.format(model_path=str(model_path), dataset_path=str(dataset_path))
    )
    result = TrainingVerifier().verify(training)
    assert result.ok, result.errors


def test_verifier_accepts_cross_entropy_dense_network(tmp_path):
    model_path = _write_model(tmp_path, "email", _MULTICLASS_MXAI)
    dataset_path = _write_csv(tmp_path, "data", _MULTICLASS_CSV)
    training = parse_training_text(
        _MULTICLASS_MXTRAIN.format(model_path=str(model_path), dataset_path=str(dataset_path))
    )
    result = TrainingVerifier().verify(training)
    assert result.ok, result.errors


def test_verifier_accepts_binary_cross_entropy_dense_network(tmp_path):
    model_path = _write_model(tmp_path, "spam", _BINARY_MXAI)
    dataset_path = _write_csv(tmp_path, "data", _BINARY_CSV)
    training = parse_training_text(
        _BINARY_MXTRAIN.format(model_path=str(model_path), dataset_path=str(dataset_path))
    )
    result = TrainingVerifier().verify(training)
    assert result.ok, result.errors


def test_verifier_reports_trainable_params_for_network(tmp_path):
    model_path = _write_model(tmp_path, "house", _REGRESSION_MXAI)
    dataset_path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
    training = parse_training_text(
        _REGRESSION_MXTRAIN.format(model_path=str(model_path), dataset_path=str(dataset_path))
    )
    result = TrainingVerifier().verify(training)
    # 3 layers × 2 params = 6
    assert len(result.trainable_parameters) == 6


def test_verifier_rejects_wrong_loss_for_scalar_network(tmp_path):
    # mse network but cross_entropy loss → error
    model_path = _write_model(tmp_path, "house", _REGRESSION_MXAI)
    dataset_path = _write_csv(tmp_path, "data", _MULTICLASS_CSV)
    bad_train = _REGRESSION_MXTRAIN.format(
        model_path=str(model_path), dataset_path=str(dataset_path)
    ).replace("TYPE mse", "TYPE cross_entropy").replace("TARGET price: Scalar", "TARGET category: Label[support, sales, operations]")
    training = parse_training_text(bad_train)
    result = TrainingVerifier().verify(training)
    assert not result.ok


def test_verifier_accepts_update_wildcard_for_network(tmp_path):
    model_path = _write_model(tmp_path, "house", _REGRESSION_MXAI)
    dataset_path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
    training = parse_training_text(
        _REGRESSION_MXTRAIN.format(model_path=str(model_path), dataset_path=str(dataset_path))
    )
    result = TrainingVerifier().verify(training)
    # UPDATE PriceRegressor.* should not produce update errors
    update_errors = [e for e in result.errors if "UPDATE" in e]
    assert not update_errors


def test_verifier_rejects_invalid_update_pattern(tmp_path):
    model_path = _write_model(tmp_path, "house", _REGRESSION_MXAI)
    dataset_path = _write_csv(tmp_path, "data", _REGRESSION_CSV)
    bad_train = _REGRESSION_MXTRAIN.format(
        model_path=str(model_path), dataset_path=str(dataset_path)
    ).replace("UPDATE PriceRegressor.*", "UPDATE NonExistentParam")
    training = parse_training_text(bad_train)
    result = TrainingVerifier().verify(training)
    assert not result.ok
    assert any("NonExistentParam" in e or "UPDATE" in e for e in result.errors)
