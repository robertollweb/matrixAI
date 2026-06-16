# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""S1 C14 — LLM schema extraction for DenseNetworkGenerator.

Tests for _parse_dense_schema, _dense_llm_schema, and the wired-up
analyze_playground_request neural path when use_llm=True.
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from matrixai.playground import (
    _parse_dense_schema,
    _dense_llm_schema,
    _DENSE_SCHEMA_SYSTEM,
    analyze_playground_request,
)


# ---------------------------------------------------------------------------
# _parse_dense_schema — unit tests
# ---------------------------------------------------------------------------

class TestParseDenseSchema:
    FULL_RESPONSE = (
        "FIELDS: credit_score, annual_income, debt_to_income, loan_amount, employment_years\n"
        "LABELS: low_risk, medium_risk, high_risk\n"
        "NAME: LoanDefaultClassifier\n"
        "ENTITY: LoanApplicant\n"
        "LAYERS: 64, 32, 16\n"
    )

    def test_parses_input_fields(self):
        result = _parse_dense_schema(self.FULL_RESPONSE)
        assert result["input_fields"] == [
            "credit_score", "annual_income", "debt_to_income", "loan_amount", "employment_years"
        ]

    def test_parses_labels(self):
        result = _parse_dense_schema(self.FULL_RESPONSE)
        assert result["labels"] == ["low_risk", "medium_risk", "high_risk"]

    def test_parses_network_name(self):
        result = _parse_dense_schema(self.FULL_RESPONSE)
        assert result["network_name"] == "LoanDefaultClassifier"

    def test_parses_entity_as_input_name(self):
        result = _parse_dense_schema(self.FULL_RESPONSE)
        assert result["input_name"] == "LoanApplicant"

    def test_parses_layers_as_relu_tuples(self):
        result = _parse_dense_schema(self.FULL_RESPONSE)
        assert result["hidden_layers"] == [(64, "relu"), (32, "relu"), (16, "relu")]

    def test_empty_text_returns_empty_dict(self):
        assert _parse_dense_schema("") == {}

    def test_missing_labels_line_omits_key(self):
        text = "FIELDS: age, income\nNAME: Foo\nENTITY: Person\nLAYERS: 32\n"
        result = _parse_dense_schema(text)
        assert "labels" not in result

    def test_invalid_layers_skipped_gracefully(self):
        text = "LAYERS: big, huge\n"
        result = _parse_dense_schema(text)
        assert "hidden_layers" not in result

    def test_extra_whitespace_trimmed(self):
        text = "FIELDS:  age ,  income  \nLABELS:  yes ,  no  \n"
        result = _parse_dense_schema(text)
        assert result["input_fields"] == ["age", "income"]
        assert result["labels"] == ["yes", "no"]

    def test_single_layer_works(self):
        text = "LAYERS: 128\n"
        result = _parse_dense_schema(text)
        assert result["hidden_layers"] == [(128, "relu")]


# ---------------------------------------------------------------------------
# _dense_llm_schema — mocked LLM call
# ---------------------------------------------------------------------------

class TestDenseLlmSchema:
    def _make_provider(self, response_text: str):
        provider = MagicMock()
        provider.complete.return_value = response_text
        return provider

    def test_returns_parsed_kwargs_on_success(self):
        fake_response = (
            "FIELDS: age, income\n"
            "LABELS: yes, no\n"
            "NAME: SpamClassifier\n"
            "ENTITY: Email\n"
            "LAYERS: 32, 16\n"
        )
        provider = self._make_provider(fake_response)
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            return_value=provider,
        ):
            result = _dense_llm_schema("classify emails as spam or not spam")

        assert result["input_fields"] == ["age", "income"]
        assert result["labels"] == ["yes", "no"]
        assert result["network_name"] == "SpamClassifier"
        assert result["hidden_layers"] == [(32, "relu"), (16, "relu")]

    def test_uses_dense_schema_system_prompt(self):
        provider = self._make_provider("FIELDS: x\nLAYERS: 8\n")
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            return_value=provider,
        ):
            _dense_llm_schema("some prompt")

        call_args = provider.complete.call_args
        assert call_args[0][0] == _DENSE_SCHEMA_SYSTEM

    def test_returns_warning_on_llm_error(self):
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            side_effect=RuntimeError("no API key"),
        ):
            result = _dense_llm_schema("classify fraud")

        assert "_llm_warning" in result
        assert "no API key" in result["_llm_warning"]


# ---------------------------------------------------------------------------
# analyze_playground_request — neural path with use_llm=True
# ---------------------------------------------------------------------------

class TestAnalyzePlaygroundRequestWithLLM:
    def _fake_complete(self, system, user):
        return (
            "FIELDS: sender_domain, subject_length, has_attachment, link_count\n"
            "LABELS: spam, not_spam\n"
            "NAME: SpamEmailClassifier\n"
            "ENTITY: EmailMessage\n"
            "LAYERS: 32, 16\n"
        )

    def test_neural_prompt_with_llm_uses_domain_fields(self):
        provider = MagicMock()
        provider.complete.side_effect = self._fake_complete
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            return_value=provider,
        ):
            result = analyze_playground_request({
                "mode": "prompt",
                "prompt": "classify emails as spam or not spam",
                "use_llm": True,
            })

        assert result.get("supervision_source") == "dense_generator"
        mxai = result.get("mxai", "")
        # Domain-specific field names should appear in the generated .mxai
        assert "sender_domain" in mxai or "SpamEmailClassifier" in mxai

    def test_neural_prompt_without_llm_uses_generic_fields(self):
        """When use_llm=False, complete() is never called (schema LLM skipped)."""
        provider = MagicMock()
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            return_value=provider,
        ):
            result = analyze_playground_request({
                "mode": "prompt",
                "prompt": "classify emails as spam or not spam",
                "use_llm": False,
            })
        provider.complete.assert_not_called()
        assert result.get("supervision_source") == "dense_generator"

    def test_llm_failure_falls_back_to_heuristic(self):
        """If LLM call fails, DenseNetworkGenerator still runs with heuristics."""
        with patch(
            "matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
            side_effect=RuntimeError("timeout"),
        ):
            result = analyze_playground_request({
                "mode": "prompt",
                "prompt": "predict customer churn from usage data",
                "use_llm": True,
            })

        assert result.get("supervision_source") == "dense_generator"
        assert result.get("mxai")
