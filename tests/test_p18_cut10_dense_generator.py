"""P18 C10 — DenseNetworkGenerator: genera NetworkSpec y textos .mxai/.mxtrain desde intención humana."""
from __future__ import annotations

import pytest

from matrixai.training.dense_generator import (
    DenseNetworkGenerationResult,
    DenseNetworkGenerator,
    DenseNetworkGeneratorError,
)
from matrixai.parser.parser import parse_text
from matrixai.ir.schema import NetworkSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

gen = DenseNetworkGenerator()


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def test_regression_intent_loss():
    result = gen.generate("quiero predecir el precio de una casa")
    assert result.loss_type == "mse"


def test_regression_intent_activation():
    result = gen.generate("quiero predecir el precio de una casa")
    assert result.output_activation == "linear"


def test_binary_intent_loss():
    result = gen.generate("detectar si un correo es spam o no")
    assert result.loss_type == "binary_cross_entropy"


def test_binary_intent_activation():
    result = gen.generate("detectar si un correo es spam o no")
    assert result.output_activation == "sigmoid"


def test_multiclass_intent_loss():
    result = gen.generate(
        "clasificar tickets en categorias",
        labels=["soporte", "ventas", "operaciones"],
    )
    assert result.loss_type == "cross_entropy"


def test_multiclass_intent_activation():
    result = gen.generate(
        "clasificar tickets en categorias",
        labels=["soporte", "ventas", "operaciones"],
    )
    assert result.output_activation == "softmax"


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

def test_output_type_regression():
    result = gen.generate("predecir consumo energetico")
    assert result.output_type == "Scalar"


def test_output_type_binary():
    result = gen.generate("detectar fraude")
    assert result.output_type == "Probability"


def test_output_type_multiclass():
    result = gen.generate(
        "clasificar en clases",
        labels=["a", "b", "c"],
    )
    assert result.output_type.startswith("ProbabilityMap[")
    assert "a" in result.output_type
    assert "b" in result.output_type
    assert "c" in result.output_type


# ---------------------------------------------------------------------------
# Output units
# ---------------------------------------------------------------------------

def test_output_units_regression():
    result = gen.generate("predecir el precio de una casa")
    assert result.output_units == 1


def test_output_units_binary():
    result = gen.generate("detectar fraude")
    assert result.output_units == 1


def test_output_units_multiclass():
    labels = ["soporte", "ventas", "operaciones"]
    result = gen.generate("clasificar", labels=labels)
    assert result.output_units == len(labels)


# ---------------------------------------------------------------------------
# Mxai text parseable
# ---------------------------------------------------------------------------

def test_mxai_text_contains_network_block():
    result = gen.generate("predecir el precio de una casa")
    assert "NETWORK" in result.mxai_text


def test_mxai_text_contains_audit_explain():
    result = gen.generate("predecir el precio de una casa")
    assert "AUDIT" in result.mxai_text
    assert "EXPLAIN" in result.mxai_text


def test_mxai_text_contains_layer_dense():
    result = gen.generate("predecir el precio de una casa")
    assert "LAYER Dense" in result.mxai_text


def test_mxai_text_is_parseable():
    result = gen.generate("predecir el precio de una casa")
    program = parse_text(result.mxai_text)
    assert len(program.networks) == 1


def test_mxai_text_network_has_correct_activation():
    result = gen.generate("predecir el precio de una casa")
    program = parse_text(result.mxai_text)
    net = program.networks[0]
    assert net.layers[-1].activation == "linear"


def test_mxai_multiclass_parseable():
    result = gen.generate("clasificar", labels=["a", "b", "c"])
    program = parse_text(result.mxai_text)
    assert len(program.networks) == 1
    assert program.networks[0].layers[-1].activation == "softmax"


# ---------------------------------------------------------------------------
# Training text
# ---------------------------------------------------------------------------

def test_training_text_contains_loss_function():
    result = gen.generate("predecir el precio de una casa")
    assert "mse" in result.training_text


def test_training_text_contains_update_wildcard():
    result = gen.generate("predecir el precio de una casa")
    assert ".*" in result.training_text


def test_training_text_binary_contains_labels():
    # Labels stored in result.labels; binary target type is Probability (numeric)
    # so the dummy row must contain "0.0", not the label string.
    result = gen.generate("detectar fraude", labels=["no_fraud", "fraud"])
    assert result.labels == ["no_fraud", "fraud"]
    # Header must have the output column
    rows = result.dataset_template_text.strip().split("\n")
    assert len(rows) == 2
    # Dummy target value is numeric (Probability type)
    assert rows[1].split(",")[-1] == "0.0"


def test_binary_dummy_target_is_numeric():
    result = gen.generate("detectar fraude")
    rows = result.dataset_template_text.strip().split("\n")
    assert rows[1].split(",")[-1] == "0.0"


def test_multiclass_dummy_target_is_label():
    result = gen.generate("clasificar tickets", labels=["soporte", "ventas", "ops"])
    rows = result.dataset_template_text.strip().split("\n")
    assert rows[1].split(",")[-1] == "soporte"


# ---------------------------------------------------------------------------
# Labels extraction from prompt
# ---------------------------------------------------------------------------

def test_labels_extracted_from_prompt():
    result = gen.generate(
        "clasificar tickets, clases: soporte, ventas, operaciones"
    )
    assert "soporte" in result.labels
    assert "ventas" in result.labels
    assert "operaciones" in result.labels


# ---------------------------------------------------------------------------
# Custom fields
# ---------------------------------------------------------------------------

def test_input_dim_from_explicit_fields():
    fields = ["age", "income", "credit_score", "debt", "employment_years"]
    result = gen.generate("predecir riesgo de credito", input_fields=fields)
    assert result.input_dim == len(fields)


def test_hidden_layers_scale_with_input_dim():
    result_small = gen.generate("predecir algo", input_fields=["a", "b"])
    result_large = gen.generate("predecir algo", input_fields=[f"f{i}" for i in range(12)])
    assert len(result_large.hidden_layers) >= len(result_small.hidden_layers)


# ---------------------------------------------------------------------------
# network_spec() roundtrip
# ---------------------------------------------------------------------------

def test_network_spec_returns_networkspec():
    result = gen.generate("predecir el precio de una casa")
    ns = result.network_spec()
    assert isinstance(ns, NetworkSpec)


def test_network_spec_layer_count():
    result = gen.generate("predecir el precio de una casa")
    ns = result.network_spec()
    assert len(ns.layers) == len(result.hidden_layers) + 1


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_to_dict_contains_keys():
    result = gen.generate("predecir algo")
    d = result.to_dict()
    for key in ("prompt", "network_name", "loss_type", "mxai_text", "training_text"):
        assert key in d


def test_assumptions_not_empty():
    result = gen.generate("predecir el precio de una casa")
    assert len(result.assumptions) >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_empty_prompt_raises():
    with pytest.raises(DenseNetworkGeneratorError):
        gen.generate("")


def test_whitespace_only_prompt_raises():
    with pytest.raises(DenseNetworkGeneratorError):
        gen.generate("   ")


# ---------------------------------------------------------------------------
# Default network size — must be "serious" (not toy sizes)
# ---------------------------------------------------------------------------

def test_default_hidden_layers_small_input_min_units():
    result = gen.generate("predecir algo", input_fields=["a", "b"])
    assert result.hidden_layers[0][0] >= 16


def test_default_hidden_layers_medium_input_min_depth():
    fields = [f"f{i}" for i in range(6)]
    result = gen.generate("clasificar algo", input_fields=fields)
    assert len(result.hidden_layers) >= 2


def test_default_hidden_layers_large_input_min_units():
    fields = [f"f{i}" for i in range(12)]
    result = gen.generate("predecir algo", input_fields=fields)
    assert result.hidden_layers[0][0] >= 64


# ---------------------------------------------------------------------------
# Depth from prompt
# ---------------------------------------------------------------------------

def test_depth_from_prompt_capas_ocultas():
    result = gen.generate("quiero una red con 5 capas ocultas para clasificar riesgo")
    assert len(result.hidden_layers) == 5


def test_depth_from_prompt_hidden_layers():
    result = gen.generate("build a model with 4 hidden layers to detect fraud")
    assert len(result.hidden_layers) == 4


def test_depth_from_prompt_capped_at_max():
    result = gen.generate("quiero 30 capas ocultas")
    assert len(result.hidden_layers) <= 12


def test_depth_not_extracted_from_irrelevant_numbers():
    # "30 días" should not be parsed as 30 hidden layers
    result = gen.generate("detectar reingresos hospitalarios en 30 dias")
    assert len(result.hidden_layers) <= 5  # default, not 30
