"""P19 C10 — CompositeNetworkGenerator: genera composite networks desde prompt."""
from __future__ import annotations

import math
import pytest

from matrixai.training.composite_generator import (
    CompositeNetworkGenerationResult,
    CompositeNetworkGenerator,
    CompositeNetworkGeneratorError,
)
from matrixai.parser.parser import parse_text

gen = CompositeNetworkGenerator()


# ---------------------------------------------------------------------------
# TestCategoricalDetection
# ---------------------------------------------------------------------------

class TestCategoricalDetection:
    def test_generator_creates_embedding_for_categorical_field(self):
        result = gen.generate(
            "clasificar productos",
            categorical_fields={"category_id": 100},
            labels=["a", "b", "c"],
        )
        assert result.is_composite
        assert len(result.embeddings) == 1
        assert result.embeddings[0]["field"] == "category_id"

    def test_generator_embedding_vocab_and_dim(self):
        result = gen.generate(
            "clasificar productos",
            categorical_fields={"category_id": 100},
            labels=["a", "b", "c"],
        )
        emb = result.embeddings[0]
        assert emb["vocab"] == 100
        assert emb["dim"] == min(8, math.ceil(math.sqrt(100)))

    def test_generator_multiple_embeddings(self):
        result = gen.generate(
            "clasificar productos con categoria y tipo",
            categorical_fields={"category_id": 100, "tipo_id": 50},
            labels=["a", "b", "c"],
        )
        assert len(result.embeddings) == 2
        fields_found = {e["field"] for e in result.embeddings}
        assert "category_id" in fields_found
        assert "tipo_id" in fields_found

    def test_generator_small_vocab_no_embedding(self):
        result = gen.generate(
            "predecir algo",
            categorical_fields={"status": 3},
        )
        assert len(result.embeddings) == 0
        assert not result.is_composite

    def test_generator_categorical_keyword_in_field_name_auto_detected(self):
        result = gen.generate(
            "clasificar datos",
            input_fields=["categoria", "precio", "peso", "volumen"],
            labels=["x", "y", "z"],
        )
        assert result.is_composite
        cat_fields = {e["field"] for e in result.embeddings}
        assert "categoria" in cat_fields


# ---------------------------------------------------------------------------
# TestResidualDetection
# ---------------------------------------------------------------------------

class TestResidualDetection:
    def test_generator_creates_residual_block_for_complex_prompt(self):
        fields = ["f1", "f2", "f3", "f4", "f5", "f6"]
        result = gen.generate(
            "predecir un modelo complejo",
            input_fields=fields,
        )
        assert len(result.blocks) >= 1

    def test_generator_block_has_layernorm(self):
        fields = ["f1", "f2", "f3", "f4", "f5", "f6"]
        result = gen.generate(
            "predecir un sistema complejo",
            input_fields=fields,
        )
        assert result.blocks[0]["has_layernorm"] is True

    def test_generator_block_has_dropout(self):
        fields = ["f1", "f2", "f3", "f4", "f5", "f6"]
        result = gen.generate(
            "predecir un sistema complejo",
            input_fields=fields,
        )
        assert result.blocks[0]["dropout_rate"] == 0.2

    def test_generator_no_residual_without_complexity_keyword(self):
        fields = ["f1", "f2", "f3", "f4", "f5", "f6"]
        result = gen.generate(
            "predecir algo sencillo",
            input_fields=fields,
            force_dense=True,
        )
        assert len(result.blocks) == 0

    def test_generator_no_residual_with_few_features(self):
        result = gen.generate(
            "predecir algo complejo",
            input_fields=["f1", "f2", "f3"],
        )
        assert len(result.blocks) == 0


# ---------------------------------------------------------------------------
# TestSequenceInput
# ---------------------------------------------------------------------------

class TestSequenceInput:
    def test_generator_creates_pool_mean_for_sequence_input(self):
        result = gen.generate("clasificar una secuencia de tokens", labels=["a", "b", "c"])
        assert result.is_sequence
        assert "POOL mean" in result.mxai_text

    def test_generator_no_pool_for_non_sequence(self):
        result = gen.generate("predecir precio de casa", force_dense=True)
        assert not result.is_sequence
        assert "POOL mean" not in result.mxai_text


# ---------------------------------------------------------------------------
# TestFallback
# ---------------------------------------------------------------------------

class TestFallback:
    def test_generator_falls_back_to_p18_dense_for_simple_prompt(self):
        result = gen.generate("predecir el precio de una casa")
        assert not result.is_composite

    def test_generator_fallback_result_structure(self):
        result = gen.generate("predecir el precio de una casa")
        assert result.embeddings == []
        assert result.blocks == []
        assert not result.is_sequence

    def test_generator_fallback_emits_dense_network(self):
        result = gen.generate("predecir el precio de una casa")
        assert "LAYER Dense" in result.mxai_text
        assert "EMBEDDING" not in result.mxai_text
        assert "BLOCK" not in result.mxai_text


# ---------------------------------------------------------------------------
# TestMxaiText
# ---------------------------------------------------------------------------

class TestMxaiText:
    def test_generator_emits_valid_mxai_text(self):
        result = gen.generate(
            "clasificar productos",
            categorical_fields={"category_id": 100},
            labels=["a", "b", "c"],
        )
        program = parse_text(result.mxai_text)
        assert len(program.networks) == 1

    def test_generator_mxai_contains_embedding(self):
        result = gen.generate(
            "clasificar con categoria",
            categorical_fields={"category_id": 100},
            labels=["a", "b", "c"],
        )
        assert "EMBEDDING" in result.mxai_text

    def test_generator_mxai_contains_block(self):
        fields = ["f1", "f2", "f3", "f4", "f5", "f6"]
        result = gen.generate("predecir modelo no lineal", input_fields=fields)
        assert "BLOCK" in result.mxai_text

    def test_generator_emits_valid_mxtrain_text(self):
        result = gen.generate("predecir el precio")
        assert "MODEL" in result.training_text
        assert "LOSS" in result.training_text

    def test_generator_mxai_pool_in_text(self):
        result = gen.generate("clasificar sequence de eventos", labels=["a", "b", "c"])
        assert "POOL mean" in result.mxai_text

    def test_generator_dense_fallback_text_parseable(self):
        result = gen.generate("predecir el precio de una casa")
        program = parse_text(result.mxai_text)
        assert program.networks[0].layers[-1].activation == "linear"


# ---------------------------------------------------------------------------
# TestExplicitHints
# ---------------------------------------------------------------------------

class TestExplicitHints:
    def test_generator_explicit_vocab_respected(self):
        result = gen.generate(
            "clasificar productos",
            categorical_fields={"brand_id": 200},
            labels=["a", "b", "c"],
        )
        assert result.embeddings[0]["vocab"] == 200
        assert result.embeddings[0]["dim"] == min(8, math.ceil(math.sqrt(200)))

    def test_generator_explicit_force_residual(self):
        result = gen.generate(
            "predecir algo",
            input_fields=["f1", "f2", "f3", "f4", "f5", "f6"],
            force_residual=True,
        )
        assert len(result.blocks) == 1
        assert result.is_composite

    def test_generator_respects_explicit_architecture_hint(self):
        result = gen.generate(
            "predecir precio complejo",
            input_fields=["f1", "f2", "f3", "f4", "f5", "f6"],
            force_dense=True,
        )
        assert not result.is_composite


# ---------------------------------------------------------------------------
# TestResultStructure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_to_dict_contains_keys(self):
        result = gen.generate(
            "clasificar",
            categorical_fields={"cat_id": 100},
            labels=["a", "b", "c"],
        )
        d = result.to_dict()
        for key in ("prompt", "network_name", "mxai_text", "training_text",
                    "is_composite", "embeddings", "blocks", "is_sequence"):
            assert key in d

    def test_result_assumptions_not_empty(self):
        result = gen.generate("predecir algo")
        assert len(result.assumptions) >= 1

    def test_error_on_empty_prompt(self):
        with pytest.raises(CompositeNetworkGeneratorError):
            gen.generate("")

    def test_error_on_whitespace_prompt(self):
        with pytest.raises(CompositeNetworkGeneratorError):
            gen.generate("   ")
