"""M8 v2 — C2: esquema LLM de dominio (una llamada → reglas) + parser.

El LLM devuelve reglas umbral en escala de dominio; `_llm_domain_rules` las parsea a
DomainRules. Falla → None (el llamador cae a coherent)."""
from __future__ import annotations

from unittest.mock import patch

from matrixai.playground import _DOMAIN_RULES_SYSTEM, _llm_domain_rules


_RULES_TEXT = (
    "CRITICO: indice_charlson > 15 AND creatinina > 4\n"
    "ALTO: ingresos_previos > 5 OR num_diagnosticos > 10\n"
    "MEDIO: edad > 70\n"
    "DEFAULT: BAJO\n"
)


class _FakeProvider:
    def __init__(self, text):
        self._text = text

    def complete(self, system, user):  # noqa: ARG002
        return self._text


def test_system_prompt_mentions_format_and_default():
    assert "DEFAULT:" in _DOMAIN_RULES_SYSTEM
    assert "OP" in _DOMAIN_RULES_SYSTEM
    assert "verbatim" in _DOMAIN_RULES_SYSTEM


def test_llm_domain_rules_parses_provider_output():
    with patch("matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
               return_value=_FakeProvider(_RULES_TEXT)):
        dr = _llm_domain_rules("clasificar reingreso",
                               ["indice_charlson", "creatinina", "ingresos_previos",
                                "num_diagnosticos", "edad"],
                               ["BAJO", "MEDIO", "ALTO", "CRITICO"])
    assert dr is not None
    assert dr.default_label == "BAJO"
    assert len(dr.rules) == 3
    assert dr.validate(["indice_charlson", "creatinina", "ingresos_previos",
                        "num_diagnosticos", "edad"],
                       ["BAJO", "MEDIO", "ALTO", "CRITICO"]) == []


def test_llm_domain_rules_returns_none_on_provider_error():
    with patch("matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
               side_effect=RuntimeError("no key")):
        assert _llm_domain_rules("x", ["a"], ["A", "B"]) is None


def test_llm_domain_rules_garbage_text_is_invalid():
    with patch("matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
               return_value=_FakeProvider("I cannot help with that.")):
        dr = _llm_domain_rules("x", ["a"], ["A", "B"])
    # Parses but yields no usable rules → validate() reports it; caller falls back.
    assert dr is not None
    assert dr.validate(["a"], ["A", "B"]) != []
