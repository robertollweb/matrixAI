# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""S1 C15 — LLM-enhanced dataset generation.

Tests for _parse_field_ranges, _llm_field_ranges,
_generate_synthetic_dataset with use_llm, and SyntheticDataGenerator.field_ranges.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from matrixai.playground import (
    _DATASET_RANGES_SYSTEM,
    _llm_field_ranges,
    _parse_field_ranges,
)


# ---------------------------------------------------------------------------
# _parse_field_ranges — unit tests
# ---------------------------------------------------------------------------

class TestParseFieldRanges:
    def test_parses_integer_range(self):
        result = _parse_field_ranges("age: 18 90\n")
        assert result["age"] == (18.0, 90.0)

    def test_parses_float_range(self):
        result = _parse_field_ranges("debt_ratio: 0.0 1.5\n")
        assert result["debt_ratio"] == (0.0, 1.5)

    def test_multiple_fields(self):
        text = "credit_score: 300 850\nannual_income: 20000 200000\nage: 18 90\n"
        result = _parse_field_ranges(text)
        assert result["credit_score"] == (300.0, 850.0)
        assert result["annual_income"] == (20000.0, 200000.0)
        assert result["age"] == (18.0, 90.0)

    def test_ignores_inverted_range(self):
        result = _parse_field_ranges("bad_field: 100 10\n")
        assert "bad_field" not in result

    def test_ignores_non_matching_lines(self):
        text = "This is a model for loan risk.\nage: 18 90\nignore this line\n"
        result = _parse_field_ranges(text)
        assert list(result.keys()) == ["age"]

    def test_empty_text_returns_empty_dict(self):
        assert _parse_field_ranges("") == {}

    def test_negative_range(self):
        result = _parse_field_ranges("temperature: -10.0 40.0\n")
        assert result["temperature"] == (-10.0, 40.0)


# ---------------------------------------------------------------------------
# _llm_field_ranges — mocked LLM call
# ---------------------------------------------------------------------------

class TestLlmFieldRanges:
    def _make_provider(self, response_text: str):
        provider = MagicMock()
        provider.complete.return_value = response_text
        return provider

    def test_returns_parsed_ranges_on_success(self):
        fake = "credit_score: 300 850\nannual_income: 20000 200000\n"
        provider = self._make_provider(fake)
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            return_value=provider,
        ):
            result = _llm_field_ranges(["credit_score", "annual_income"], "LoanDefaultClassifier")

        assert result["credit_score"] == (300.0, 850.0)
        assert result["annual_income"] == (20000.0, 200000.0)

    def test_uses_dataset_ranges_system_prompt(self):
        provider = self._make_provider("age: 18 90\n")
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            return_value=provider,
        ):
            _llm_field_ranges(["age"], "SomeModel")

        call_args = provider.complete.call_args
        assert call_args[0][0] == _DATASET_RANGES_SYSTEM

    def test_returns_empty_dict_on_llm_error(self):
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            side_effect=RuntimeError("no key"),
        ):
            result = _llm_field_ranges(["age", "income"])

        assert result == {}

    def test_context_included_in_user_prompt(self):
        provider = self._make_provider("age: 18 90\n")
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            return_value=provider,
        ):
            _llm_field_ranges(["age"], "HospitalReadmission")

        user_prompt = provider.complete.call_args[0][1]
        assert "HospitalReadmission" in user_prompt
        assert "age" in user_prompt


# ---------------------------------------------------------------------------
# SyntheticDataGenerator.field_ranges — unit tests
# ---------------------------------------------------------------------------

class TestSyntheticGeneratorFieldRanges:
    """Test that field_ranges override the default Scalar range in the generator."""

    _MXAI = (
        "PROJECT LoanProject\n\n"
        "VECTOR LoanApplicant[2]\n"
        "  credit_score: Scalar\n"
        "  age: Scalar\n"
        "END\n\n"
        "NETWORK LoanDefaultClassifier\n"
        "  INPUT LoanApplicant\n"
        "  LAYER Dense units=32 activation=relu\n"
        "  LAYER Dense units=1 activation=sigmoid\n"
        "  OUTPUT predicted_prob: Probability\n"
        "END\n\n"
        "GRAPH\n  LoanApplicant -> LoanDefaultClassifier\nEND\n"
    )

    _MXTRAIN = (
        "MODEL LoanProject.mxai\n\n"
        "DATASET LoanDataset\n"
        "  SOURCE csv(\"loan.train.csv\")\n"
        "  INPUT LoanApplicant FROM COLUMNS [credit_score, age]\n"
        "  TARGET predicted_prob: Probability\n"
        "  SPLIT train=0.8 validation=0.2 seed=42\n"
        "  BATCH size=8\n"
        "END\n\n"
        "LOSS LoanLoss\n"
        "  TYPE binary_cross_entropy\n"
        "  PREDICTION LoanDefaultClassifier\n"
        "  TARGET predicted_prob\n"
        "END\n\n"
        "OPTIMIZER LoanOptimizer\n"
        "  TYPE sgd\n"
        "  LEARNING_RATE 0.01\n"
        "  UPDATE LoanDefaultClassifier.*\n"
        "END\n\n"
        "RUN\n  EPOCHS 10\nEND\n"
    )

    def _make_generator(self, field_ranges=None):
        from matrixai.parser import parse_text
        from matrixai.training.parser import parse_training_text
        from matrixai.training.synthetic import SyntheticDataGenerator

        program = parse_text(self._MXAI)
        training = parse_training_text(self._MXTRAIN)
        return SyntheticDataGenerator(
            program=program,
            training=training,
            seed=42,
            rows=10,
            mode="random",
            field_ranges=field_ranges,
        )

    def test_without_field_ranges_uses_default(self):
        gen = self._make_generator()
        adapter = gen.generate()
        rows = adapter.rows
        scores = [r["credit_score"] for r in rows]
        # Default untyped Scalar samples in [0, 1] — matches the inference slider range
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_field_ranges_normalized_to_slider_range(self):
        # LLM field_ranges shape the sampling domain but the stored value is
        # normalised to [0, 1] so training features match what the inference
        # sliders send (sliders are always 0-1, regardless of domain scale).
        gen = self._make_generator({"credit_score": (300.0, 850.0)})
        adapter = gen.generate()
        rows = adapter.rows
        scores = [r["credit_score"] for r in rows]
        assert all(0.0 <= s <= 1.0 for s in scores), f"Out-of-range scores: {scores}"

    def test_field_ranges_only_affect_named_fields(self):
        gen = self._make_generator({"credit_score": (300.0, 850.0)})
        adapter = gen.generate()
        rows = adapter.rows
        ages = [r["age"] for r in rows]
        # age not in field_ranges — default [0, 1]
        assert all(0.0 <= a <= 1.0 for a in ages)


# ---------------------------------------------------------------------------
# _generate_synthetic_dataset with use_llm
# ---------------------------------------------------------------------------

class TestGenerateSyntheticDatasetWithLlm:
    _MXAI = (
        "PROJECT LoanProject\n\n"
        "VECTOR LoanApplicant[2]\n"
        "  credit_score: Scalar\n"
        "  age: Scalar\n"
        "END\n\n"
        "NETWORK LoanDefaultClassifier\n"
        "  INPUT LoanApplicant\n"
        "  LAYER Dense units=32 activation=relu\n"
        "  LAYER Dense units=1 activation=sigmoid\n"
        "  OUTPUT predicted_prob: Probability\n"
        "END\n\n"
        "GRAPH\n  LoanApplicant -> LoanDefaultClassifier\nEND\n"
    )

    _MXTRAIN = (
        "MODEL LoanProject.mxai\n\n"
        "DATASET LoanDataset\n"
        "  SOURCE csv(\"loan.train.csv\")\n"
        "  INPUT LoanApplicant FROM COLUMNS [credit_score, age]\n"
        "  TARGET predicted_prob: Probability\n"
        "  SPLIT train=0.8 validation=0.2 seed=42\n"
        "  BATCH size=8\n"
        "END\n\n"
        "LOSS LoanLoss\n"
        "  TYPE binary_cross_entropy\n"
        "  PREDICTION LoanDefaultClassifier\n"
        "  TARGET predicted_prob\n"
        "END\n\n"
        "OPTIMIZER LoanOptimizer\n"
        "  TYPE sgd\n"
        "  LEARNING_RATE 0.01\n"
        "  UPDATE LoanDefaultClassifier.*\n"
        "END\n\n"
        "RUN\n  EPOCHS 10\nEND\n"
    )

    def test_use_llm_false_skips_llm(self):
        from matrixai.playground import _generate_synthetic_dataset
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}):
            with patch("matrixai.playground._llm_field_ranges") as mock_ranges:
                result = _generate_synthetic_dataset(
                    self._MXAI, self._MXTRAIN, 10, 42, "random", use_llm=False
                )
        assert result["ok"]
        mock_ranges.assert_not_called()
        assert result["llm_ranges_used"] is False

    def test_use_llm_true_calls_llm_when_active(self):
        from matrixai.playground import _generate_synthetic_dataset
        fake_ranges = {"credit_score": (300.0, 850.0), "age": (18.0, 90.0)}
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}):
            with patch("matrixai.playground._llm_field_ranges", return_value=fake_ranges) as mock_ranges:
                result = _generate_synthetic_dataset(
                    self._MXAI, self._MXTRAIN, 10, 42, "random", use_llm=True
                )
        assert result["ok"]
        mock_ranges.assert_called_once()
        assert result["llm_ranges_used"] is True

    def test_use_llm_true_but_llm_inactive_skips(self):
        from matrixai.playground import _generate_synthetic_dataset
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": False}):
            with patch("matrixai.playground._llm_field_ranges") as mock_ranges:
                result = _generate_synthetic_dataset(
                    self._MXAI, self._MXTRAIN, 10, 42, "random", use_llm=True
                )
        assert result["ok"]
        mock_ranges.assert_not_called()
        assert result["llm_ranges_used"] is False
