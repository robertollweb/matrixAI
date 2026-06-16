"""P19 C7 — TrainingVerifier extendido para composite_network.

Tests: _symbol_type, _prediction_node, DifferentiabilityVerifier, TrainingVerifier.verify
con redes compuestas (embeddings, LayerNorm, residuales, dropout).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from matrixai.compiler import BackendContractAnalyzer
from matrixai.parser.parser import parse_text
from matrixai.training.differentiability import DifferentiabilityVerifier
from matrixai.training.parser import parse_training_text
from matrixai.training.verifier import TrainingVerifier

# ---------------------------------------------------------------------------
# Fixtures — programs
# ---------------------------------------------------------------------------

_MXAI_EMBEDDING_SOFTMAX = textwrap.dedent("""\
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
""")

_MXAI_RESIDUAL_REGRESSION = textwrap.dedent("""\
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
""")

_MXAI_DROPOUT = textwrap.dedent("""\
    PROJECT DropTest
    VECTOR V[2]
      a: Scalar
      b: Scalar
    END
    NETWORK DropNet
      INPUT V
      LAYER Dense units=16 activation=relu
      LAYER Dropout rate=0.3
      LAYER Dense units=4 activation=softmax
      OUTPUT label: ProbabilityMap[w, x, y, z]
    END
    GRAPH
      V -> DropNet
    END
""")

_MXAI_SCALAR_COMPOSITE = textwrap.dedent("""\
    PROJECT ScalarTest
    VECTOR V[2]
      a: Scalar
      b: Scalar
    END
    NETWORK ScalarNet
      INPUT V
      LAYER Dense units=8 activation=relu
      LAYER Dense units=1 activation=linear
      OUTPUT y: Scalar
    END
    GRAPH
      V -> ScalarNet
    END
""")

# ---------------------------------------------------------------------------
# Training spec templates
# ---------------------------------------------------------------------------

_MXTRAIN_CROSS_ENTROPY = textwrap.dedent("""\
    MODEL {model_path}
    DATASET D
      SOURCE csv("{dataset_path}")
      INPUT Product FROM COLUMNS [category_id, price, weight]
      TARGET label: Label[a, b, c]
    END
    LOSS L
      TYPE cross_entropy
      PREDICTION label
      TARGET label
    END
    OPTIMIZER O
      TYPE sgd
      LEARNING_RATE 0.01
      UPDATE CategoryNet.*
    END
""")

_MXTRAIN_MSE = textwrap.dedent("""\
    MODEL {model_path}
    DATASET D
      SOURCE csv("{dataset_path}")
      INPUT H FROM COLUMNS [x1, x2]
      TARGET y: Scalar
    END
    LOSS L
      TYPE mse
      PREDICTION y
      TARGET y
    END
    OPTIMIZER O
      TYPE sgd
      LEARNING_RATE 0.01
      UPDATE ResNet.*
    END
""")

_MXTRAIN_DROPOUT_CE = textwrap.dedent("""\
    MODEL {model_path}
    DATASET D
      SOURCE csv("{dataset_path}")
      INPUT V FROM COLUMNS [a, b]
      TARGET label: Label[w, x, y, z]
    END
    LOSS L
      TYPE cross_entropy
      PREDICTION label
      TARGET label
    END
    OPTIMIZER O
      TYPE sgd
      LEARNING_RATE 0.01
      UPDATE DropNet.*
    END
""")

# ---------------------------------------------------------------------------
# CSV data
# ---------------------------------------------------------------------------

_CSV_EMBEDDING = "category_id,price,weight,label\n5,0.5,0.5,a\n3,0.2,0.8,b\n7,0.9,0.1,c\n"
_CSV_REGRESSION = "x1,x2,y\n0.5,-0.5,0.0\n0.3,0.7,1.0\n-0.2,0.4,0.5\n"
_CSV_DROPOUT = "a,b,label\n0.5,0.3,w\n-0.1,0.9,x\n0.7,-0.4,y\n0.2,0.2,z\n"


def _write_model(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / f"{name}.mxai"
    p.write_text(content)
    return p


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / f"{name}.csv"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# TestSymbolType — composite network output types
# ---------------------------------------------------------------------------

class TestSymbolType:
    def test_probabilitymap_for_softmax_composite_network(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        t = TrainingVerifier()._symbol_type(program, "label")
        assert t is not None
        assert t.name == "ProbabilityMap"

    def test_scalar_for_regression_composite_network(self):
        program = parse_text(_MXAI_RESIDUAL_REGRESSION)
        t = TrainingVerifier()._symbol_type(program, "y")
        assert t is not None
        assert t.name == "Scalar"

    def test_symbol_type_also_matches_composite_network_name(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        t = TrainingVerifier()._symbol_type(program, "CategoryNet")
        assert t is not None
        assert t.name == "ProbabilityMap"

    def test_prediction_function_is_none_for_composite_network(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        fn = TrainingVerifier()._prediction_function(program, "label")
        assert fn is None


# ---------------------------------------------------------------------------
# TestPredictionNode — differentiability verifier node lookup
# ---------------------------------------------------------------------------

class TestPredictionNode:
    def test_finds_composite_network_by_output_variable(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        node = DifferentiabilityVerifier()._prediction_node(program, "label")
        assert node == "CategoryNet"

    def test_finds_residual_composite_network_by_output(self):
        program = parse_text(_MXAI_RESIDUAL_REGRESSION)
        node = DifferentiabilityVerifier()._prediction_node(program, "y")
        assert node == "ResNet"

    def test_finds_composite_network_by_network_name(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        node = DifferentiabilityVerifier()._prediction_node(program, "CategoryNet")
        assert node == "CategoryNet"


# ---------------------------------------------------------------------------
# TestDifferentiabilityVerifier — composite network paths
# ---------------------------------------------------------------------------

class TestDifferentiabilityVerifier:
    def test_no_errors_for_cross_entropy_embedding_network(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        training = parse_training_text(
            _MXTRAIN_CROSS_ENTROPY.format(model_path="dummy.mxai", dataset_path="dummy.csv")
        )
        report = BackendContractAnalyzer().analyze(program)
        result = DifferentiabilityVerifier().verify(training, program, report)
        assert result.ok, result.errors

    def test_prediction_node_is_composite_network(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        training = parse_training_text(
            _MXTRAIN_CROSS_ENTROPY.format(model_path="dummy.mxai", dataset_path="dummy.csv")
        )
        report = BackendContractAnalyzer().analyze(program)
        result = DifferentiabilityVerifier().verify(training, program, report)
        assert result.prediction_node == "CategoryNet"

    def test_parameter_paths_include_embedding_table(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        training = parse_training_text(
            _MXTRAIN_CROSS_ENTROPY.format(model_path="dummy.mxai", dataset_path="dummy.csv")
        )
        report = BackendContractAnalyzer().analyze(program)
        result = DifferentiabilityVerifier().verify(training, program, report)
        assert any("cat_emb" in key for key in result.parameter_paths)

    def test_parameter_paths_include_dense_weights(self):
        program = parse_text(_MXAI_EMBEDDING_SOFTMAX)
        training = parse_training_text(
            _MXTRAIN_CROSS_ENTROPY.format(model_path="dummy.mxai", dataset_path="dummy.csv")
        )
        report = BackendContractAnalyzer().analyze(program)
        result = DifferentiabilityVerifier().verify(training, program, report)
        assert any("L1.W" in key or "L2.W" in key for key in result.parameter_paths)

    def test_no_errors_for_mse_residual_network(self):
        program = parse_text(_MXAI_RESIDUAL_REGRESSION)
        training = parse_training_text(
            _MXTRAIN_MSE.format(model_path="dummy.mxai", dataset_path="dummy.csv")
        )
        report = BackendContractAnalyzer().analyze(program)
        result = DifferentiabilityVerifier().verify(training, program, report)
        assert result.ok, result.errors

    def test_parameter_paths_include_layernorm_params(self):
        program = parse_text(_MXAI_RESIDUAL_REGRESSION)
        training = parse_training_text(
            _MXTRAIN_MSE.format(model_path="dummy.mxai", dataset_path="dummy.csv")
        )
        report = BackendContractAnalyzer().analyze(program)
        result = DifferentiabilityVerifier().verify(training, program, report)
        assert any("gamma" in key or "beta" in key for key in result.parameter_paths)

    def test_dropout_does_not_block_differentiability(self):
        program = parse_text(_MXAI_DROPOUT)
        training = parse_training_text(
            _MXTRAIN_DROPOUT_CE.format(model_path="dummy.mxai", dataset_path="dummy.csv")
        )
        report = BackendContractAnalyzer().analyze(program)
        result = DifferentiabilityVerifier().verify(training, program, report)
        assert result.ok, result.errors


# ---------------------------------------------------------------------------
# TestTrainingVerifierFull — full verify with real files
# ---------------------------------------------------------------------------

class TestTrainingVerifierFull:
    def test_accepts_cross_entropy_composite_with_embedding(self, tmp_path):
        model_path = _write_model(tmp_path, "emb", _MXAI_EMBEDDING_SOFTMAX)
        dataset_path = _write_csv(tmp_path, "data", _CSV_EMBEDDING)
        training = parse_training_text(
            _MXTRAIN_CROSS_ENTROPY.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        assert result.ok, result.errors

    def test_accepts_mse_composite_with_residual(self, tmp_path):
        model_path = _write_model(tmp_path, "res", _MXAI_RESIDUAL_REGRESSION)
        dataset_path = _write_csv(tmp_path, "data", _CSV_REGRESSION)
        training = parse_training_text(
            _MXTRAIN_MSE.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        assert result.ok, result.errors

    def test_accepts_dropout_composite_network(self, tmp_path):
        model_path = _write_model(tmp_path, "drop", _MXAI_DROPOUT)
        dataset_path = _write_csv(tmp_path, "data", _CSV_DROPOUT)
        training = parse_training_text(
            _MXTRAIN_DROPOUT_CE.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        assert result.ok, result.errors

    def test_trainable_params_include_embedding_table(self, tmp_path):
        model_path = _write_model(tmp_path, "emb", _MXAI_EMBEDDING_SOFTMAX)
        dataset_path = _write_csv(tmp_path, "data", _CSV_EMBEDDING)
        training = parse_training_text(
            _MXTRAIN_CROSS_ENTROPY.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        names = [p["name"] for p in result.trainable_parameters]
        assert any("cat_emb" in n for n in names)

    def test_trainable_params_include_layernorm_gamma_beta(self, tmp_path):
        model_path = _write_model(tmp_path, "res", _MXAI_RESIDUAL_REGRESSION)
        dataset_path = _write_csv(tmp_path, "data", _CSV_REGRESSION)
        training = parse_training_text(
            _MXTRAIN_MSE.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        names = [p["name"] for p in result.trainable_parameters]
        assert any("gamma" in n for n in names)
        assert any("beta" in n for n in names)

    def test_update_wildcard_covers_embedding_table(self, tmp_path):
        model_path = _write_model(tmp_path, "emb", _MXAI_EMBEDDING_SOFTMAX)
        dataset_path = _write_csv(tmp_path, "data", _CSV_EMBEDDING)
        training = parse_training_text(
            _MXTRAIN_CROSS_ENTROPY.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        update_errors = [e for e in result.errors if "UPDATE" in e]
        assert not update_errors

    def test_update_wildcard_covers_block_layernorm(self, tmp_path):
        model_path = _write_model(tmp_path, "res", _MXAI_RESIDUAL_REGRESSION)
        dataset_path = _write_csv(tmp_path, "data", _CSV_REGRESSION)
        training = parse_training_text(
            _MXTRAIN_MSE.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        update_errors = [e for e in result.errors if "UPDATE" in e]
        assert not update_errors

    def test_rejects_cross_entropy_for_scalar_output_composite(self, tmp_path):
        model_path = _write_model(tmp_path, "scalar", _MXAI_SCALAR_COMPOSITE)
        dataset_path = _write_csv(tmp_path, "data", "a,b,y\n0.5,0.3,a\n0.1,0.9,b\n")
        mxtrain = textwrap.dedent(f"""\
            MODEL {model_path}
            DATASET D
              SOURCE csv("{dataset_path}")
              INPUT V FROM COLUMNS [a, b]
              TARGET y: Label[a, b]
            END
            LOSS L
              TYPE cross_entropy
              PREDICTION y
              TARGET y
            END
            OPTIMIZER O
              TYPE sgd
              LEARNING_RATE 0.01
              UPDATE ScalarNet.*
            END
        """)
        result = TrainingVerifier().verify(parse_training_text(mxtrain))
        assert not result.ok
        assert any("cross_entropy" in e or "ProbabilityMap" in e for e in result.errors)

    def test_block_params_use_hierarchical_names(self, tmp_path):
        model_path = _write_model(tmp_path, "res", _MXAI_RESIDUAL_REGRESSION)
        dataset_path = _write_csv(tmp_path, "data", _CSV_REGRESSION)
        training = parse_training_text(
            _MXTRAIN_MSE.format(model_path=str(model_path), dataset_path=str(dataset_path))
        )
        result = TrainingVerifier().verify(training)
        names = [p["name"] for p in result.trainable_parameters]
        assert any("res1" in n for n in names)
