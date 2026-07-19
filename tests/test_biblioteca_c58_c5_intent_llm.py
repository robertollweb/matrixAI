# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Contrato 58 (BIBLIOTECA_MEJORAS_USO_REAL) C5 — interpretación LLM
OPT-IN de la intención local: `intent_llm.py` (llamada + parser acotado +
saneo M8-A1) enhebrada por `generate_project_from_dataset`/
`generate_temporal_project_from_dataset` como un `architecture_hints` canal
COMPLETAMENTE separado de `use_llm=True` (que re-derivaría FIELDS/LABELS del
prompt — prohibido: el esquema ya está fijado). Cubre: propuesta válida,
saneo, límites de profundidad/ancho, fallos (sin LLM/propuesta inválida/
transporte) con la política de "nunca fallback silencioso", y el camino
temporal."""
from __future__ import annotations

import unittest.mock

import pytest

from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider
from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
    generate_temporal_project_from_dataset,
)
from matrixai.training.intent_llm import (
    IntentArchitectureError,
    _MAX_LAYERS,
    _MAX_WIDTH,
    build_llm_context,
    propose_intent_architecture,
)
from matrixai.training.dense_generator import validate_architecture_hints
from matrixai.playground import analyze_playground_request


def _tabular_csv(n: int = 20) -> str:
    lines = ["a,b,resultado"]
    for i in range(n):
        lines.append(f"{i},{i * 0.5:.2f},{'pos' if i % 2 == 0 else 'neg'}")
    return "\n".join(lines) + "\n"


def _mar_rows(n: int = 20) -> str:
    lines = ["fecha,altura_ola,temperatura"]
    for d in range(1, n + 1):
        lines.append(f"2024-01-{d:02d},{2.0 + d * 0.1:.2f},{15.0 + d * 0.05:.2f}")
    return "\n".join(lines) + "\n"


def _high_cardinality_csv(n_categories: int = 20, rows_per_cat: int = 3) -> str:
    """Auditoría C5 [ALTA]: fuerza `want_composite` en `playground.py`
    (`_prompt_highcard`) — una categórica declarada con más de `_ONEHOT_MAX`
    valores distintos enruta al generador COMPOSITE, la ruta que descarta
    `hidden_layers` propuestos por el LLM."""
    cats = [f"cat{i}" for i in range(n_categories)]
    lines = ["x,resultado"]
    for i in range(n_categories * rows_per_cat):
        lines.append(f"{cats[i % n_categories]},{i * 0.5:.2f}")
    return "\n".join(lines) + "\n"


def _mock_provider(response_text: str, *, provider_name="deepseek", model_name="deepseek-chat"):
    provider = unittest.mock.Mock()
    provider.provider_name = provider_name
    provider.model_name = model_name
    provider.complete.return_value = response_text
    return unittest.mock.patch.object(ChatCompletionsLLMProposalProvider, "from_env", return_value=provider)


# ---------------------------------------------------------------------------
# build_llm_context — nunca filas/CSV
# ---------------------------------------------------------------------------

class TestBuildLlmContext:
    def test_context_never_contains_row_values(self):
        ctx = build_llm_context(
            features=[{"name": "edad", "type": "number", "range": (18, 90)}],
            task="classification", target_column="resultado", user_intent="prioriza urgentes",
        )
        # Solo metadata — nombre, tipo, rango; nunca un valor de fila real.
        assert "edad" in ctx
        assert "[18, 90]" in ctx
        assert "prioriza urgentes" in ctx

    def test_context_includes_categories_when_present(self):
        ctx = build_llm_context(
            features=[{"name": "tipo", "type": "categorical", "categories": ["a", "b", "c"]}],
            task="classification", target_column="y", user_intent="x",
        )
        assert "a, b, c" in ctx


# ---------------------------------------------------------------------------
# propose_intent_architecture — unidad
# ---------------------------------------------------------------------------

class TestProposeIntentArchitecture:
    def test_valid_proposal_is_parsed_and_sanitized(self):
        with _mock_provider("LAYERS: 128, 64, 32\nRATIONALE: needs moderate capacity\n"):
            proposal = propose_intent_architecture("some context")
        assert proposal.hidden_layers == [(128, "relu"), (64, "relu"), (32, "relu")]
        assert proposal.rationale == "needs moderate capacity"
        assert proposal.sanitizer_adjusted is False
        assert proposal.provider == "deepseek"
        assert proposal.model == "deepseek-chat"

    def test_narrow_relu_layers_get_widened_by_the_sanitizer(self):
        with _mock_provider("LAYERS: 4, 8\nRATIONALE: tiny\n"):
            proposal = propose_intent_architecture("ctx")
        assert all(units >= 16 for units, _ in proposal.hidden_layers)
        assert proposal.sanitizer_adjusted is True

    def test_depth_is_capped_at_max_layers(self):
        sizes = ", ".join(str(64) for _ in range(_MAX_LAYERS + 10))
        with _mock_provider(f"LAYERS: {sizes}\n"):
            proposal = propose_intent_architecture("ctx")
        assert len(proposal.hidden_layers) == _MAX_LAYERS

    def test_width_is_capped_at_max_width(self):
        with _mock_provider(f"LAYERS: {_MAX_WIDTH * 10}\n"):
            proposal = propose_intent_architecture("ctx")
        assert proposal.hidden_layers[0][0] == _MAX_WIDTH

    def test_extraneous_lines_are_ignored_not_applied(self):
        """El LLM podría alucinar FIELDS/LABELS pese a las instrucciones —
        deben ignorarse en silencio, nunca aplicarse (el esquema no puede
        cambiar)."""
        text = "FIELDS: hacked_field\nLABELS: x,y\nLAYERS: 64, 32\n"
        with _mock_provider(text):
            proposal = propose_intent_architecture("ctx")
        assert proposal.hidden_layers == [(64, "relu"), (32, "relu")]

    def test_no_llm_configured_raises_non_retryable_error(self):
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider, "from_env", side_effect=ValueError("no api key"),
        ):
            with pytest.raises(IntentArchitectureError) as exc_info:
                propose_intent_architecture("ctx")
        assert exc_info.value.retryable is False

    def test_transport_failure_raises_retryable_error(self):
        provider = unittest.mock.Mock()
        provider.provider_name = "deepseek"
        provider.model_name = "deepseek-chat"
        provider.complete.side_effect = TimeoutError("timed out")
        with unittest.mock.patch.object(ChatCompletionsLLMProposalProvider, "from_env", return_value=provider):
            with pytest.raises(IntentArchitectureError) as exc_info:
                propose_intent_architecture("ctx")
        assert exc_info.value.retryable is True

    def test_unparseable_response_raises_retryable_error_not_a_silent_noop(self):
        """Decisión E: 'no hay fallback silencioso' — una propuesta
        ilegible debe fallar de forma visible, nunca proceder como si no
        hubiera pasado nada."""
        with _mock_provider("I cannot help with that."):
            with pytest.raises(IntentArchitectureError) as exc_info:
                propose_intent_architecture("ctx")
        assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# generate_project_from_dataset — enhebrado end-to-end
# ---------------------------------------------------------------------------

class TestGenerateProjectFromDatasetIntentLlm:
    def test_default_is_not_requested_and_not_used(self):
        res = generate_project_from_dataset(_tabular_csv(), "resultado", user_intent="algo")
        assert res["provenance"]["intent_llm"] == {
            "requested": False, "used": False, "provider": None, "model": None,
            "proposal_sha256": None, "sanitizer_result": None, "fallback": None,
        }

    def test_no_intent_means_no_intent_llm_block_at_all(self):
        res = generate_project_from_dataset(_tabular_csv(), "resultado")
        assert res["provenance"]["intent_llm"] is None

    def test_use_intent_llm_without_intent_is_rejected(self):
        with pytest.raises(DatasetProjectError, match="requiere una intención"):
            generate_project_from_dataset(_tabular_csv(), "resultado", use_intent_llm=True)

    def test_valid_proposal_changes_the_generated_layer_sizes(self):
        with _mock_provider("LAYERS: 256, 128\nRATIONALE: needs it\n"):
            res = generate_project_from_dataset(
                _tabular_csv(), "resultado", user_intent="prioriza urgentes", use_intent_llm=True,
            )
        assert res["ok"]
        assert "LAYER Dense units=256" in res["mxai"]
        assert "LAYER Dense units=128" in res["mxai"]

    def test_provenance_intent_llm_block_is_complete_on_success(self):
        with _mock_provider("LAYERS: 64, 32\nRATIONALE: r\n"):
            res = generate_project_from_dataset(
                _tabular_csv(), "resultado", user_intent="algo", use_intent_llm=True,
            )
        block = res["provenance"]["intent_llm"]
        assert block["requested"] is True
        assert block["used"] is True
        assert block["provider"] == "deepseek"
        assert block["model"] == "deepseek-chat"
        assert len(block["proposal_sha256"]) == 64
        assert block["sanitizer_result"] in ("accepted", "adjusted")
        assert block["fallback"] is None

    @pytest.mark.parametrize("intent", ["prioriza urgentes", "modelo simple y rápido"])
    def test_schema_and_metadata_never_change_regardless_of_llm_proposal(self, intent):
        """Invariante central de C5: el LLM SOLO puede tocar hidden_layers —
        features/tipos/rangos/categorías/target/csv preparado idénticos con
        o sin interpretación LLM."""
        baseline = generate_project_from_dataset(_tabular_csv(), "resultado")
        with _mock_provider("LAYERS: 512, 256, 128\nRATIONALE: r\n"):
            with_llm = generate_project_from_dataset(
                _tabular_csv(), "resultado", user_intent=intent, use_intent_llm=True,
            )
        assert with_llm["csv_text"] == baseline["csv_text"]
        assert with_llm.get("field_ranges") == baseline.get("field_ranges")
        assert with_llm.get("field_types") == baseline.get("field_types")
        assert with_llm.get("field_categories") == baseline.get("field_categories")
        assert with_llm["provenance"]["target_column"] == baseline["provenance"]["target_column"]
        assert with_llm["provenance"]["feature_name_map"] == baseline["provenance"]["feature_name_map"]
        assert with_llm["provenance"]["synthesized_prompt"] == baseline["provenance"]["synthesized_prompt"]
        # Lo único que SÍ debe diferir es la forma de la red.
        assert with_llm["mxai"] != baseline["mxai"]

    def test_no_llm_configured_is_a_dataset_project_error_with_retryable_false(self):
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider, "from_env", side_effect=ValueError("no key"),
        ):
            with pytest.raises(DatasetProjectError) as exc_info:
                generate_project_from_dataset(
                    _tabular_csv(), "resultado", user_intent="algo", use_intent_llm=True,
                )
        assert getattr(exc_info.value, "retryable", None) is False

    def test_transport_failure_is_a_dataset_project_error_with_retryable_true(self):
        provider = unittest.mock.Mock()
        provider.provider_name = "deepseek"
        provider.model_name = "deepseek-chat"
        provider.complete.side_effect = TimeoutError("timed out")
        with unittest.mock.patch.object(ChatCompletionsLLMProposalProvider, "from_env", return_value=provider):
            with pytest.raises(DatasetProjectError) as exc_info:
                generate_project_from_dataset(
                    _tabular_csv(), "resultado", user_intent="algo", use_intent_llm=True,
                )
        assert getattr(exc_info.value, "retryable", None) is True

    def test_invalid_proposal_never_falls_back_silently_to_default_architecture(self):
        """Si la propuesta es ilegible, la generación FALLA (no continúa
        con la arquitectura por defecto sin decírselo al usuario)."""
        with _mock_provider("no puedo ayudar con eso"):
            with pytest.raises(DatasetProjectError):
                generate_project_from_dataset(
                    _tabular_csv(), "resultado", user_intent="algo", use_intent_llm=True,
                )

    # -----------------------------------------------------------------
    # Auditoría C5 [ALTA]: `used=True` se marcaba en cuanto el LLM proponía
    # algo interpretable, sin comprobar si la RUTA de generación realmente
    # elegida (composite/transformer, decidida por el esquema, no por el
    # LLM) admite `hidden_layers`. Reproducido exactamente: una categórica
    # de alta cardinalidad fuerza `composite_generator`, que descarta el
    # hint — el `.mxai` resultante no contenía ninguno de los tamaños
    # propuestos, pero la procedencia seguía afirmando `used=True`.
    # -----------------------------------------------------------------

    def test_used_is_corrected_to_false_when_composite_routing_drops_the_proposal(self):
        with _mock_provider("LAYERS: 777, 333\nRATIONALE: big net\n"):
            res = generate_project_from_dataset(
                _high_cardinality_csv(), "resultado",
                user_intent="quiero una red grande", use_intent_llm=True,
            )
        assert res["ok"]
        assert res["supervision_source"] == "composite_generator"
        assert "777" not in res["mxai"]
        assert "333" not in res["mxai"]
        block = res["provenance"]["intent_llm"]
        assert block["requested"] is True
        # La corrección real del hallazgo: sin ella, esto era True.
        assert block["used"] is False
        assert block["fallback"] is not None
        assert "composite" in block["fallback"] or "denso" in block["fallback"]

    def test_used_stays_true_when_dense_routing_actually_applies_the_proposal(self):
        """Regresión de no sobre-corregir: la ruta DENSA (el caso normal)
        sigue marcando `used=True` como antes."""
        with _mock_provider("LAYERS: 256, 128\nRATIONALE: r\n"):
            res = generate_project_from_dataset(
                _tabular_csv(), "resultado", user_intent="prioriza urgentes", use_intent_llm=True,
            )
        assert res["supervision_source"] == "dense_generator"
        block = res["provenance"]["intent_llm"]
        assert block["used"] is True
        assert block["fallback"] is None
        assert "256" in res["mxai"]


# ---------------------------------------------------------------------------
# generate_temporal_project_from_dataset — el camino temporal enhebra use_intent_llm
# ---------------------------------------------------------------------------

class TestTemporalPathThreadsUseIntentLlm:
    def test_valid_proposal_reaches_the_temporal_path(self):
        with _mock_provider("LAYERS: 96, 48\nRATIONALE: r\n"):
            res = generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola", temporal_column="fecha", horizon=1,
                user_intent="quiero capturar tendencias sutiles", use_intent_llm=True,
            )
        assert res["ok"]
        assert res["provenance"]["intent_llm"]["used"] is True
        assert "LAYER Dense units=96" in res["mxai"]

    def test_temporal_path_rejects_use_intent_llm_without_intent(self):
        with pytest.raises(DatasetProjectError, match="requiere una intención"):
            generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola", temporal_column="fecha", horizon=1,
                use_intent_llm=True,
            )

    def test_temporal_path_propagates_retryable_failures(self):
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider, "from_env", side_effect=ValueError("no key"),
        ):
            with pytest.raises(DatasetProjectError) as exc_info:
                generate_temporal_project_from_dataset(
                    _mar_rows(), target_column="altura_ola", temporal_column="fecha", horizon=1,
                    user_intent="algo", use_intent_llm=True,
                )
        assert getattr(exc_info.value, "retryable", None) is False


# ---------------------------------------------------------------------------
# Auditoría C5 [MEDIA]: `architecture_hints` es un payload de la API PÚBLICA
# (`/api/analyze`, `playground.py`) — el saneador `sanitize_hidden_layers`
# (M8-A1) asume una forma ya correcta y solo ensancha ReLU estrechas; no
# validaba tipos ni acotaba profundidad/ancho, así que un payload arbitrario
# reventaba sin control (AttributeError/TypeError → HTTP 500) o se aceptaba
# muy por encima de los límites de 12 capas/16384 unidades que el propio C5
# exige. `validate_architecture_hints` es la primera línea de defensa.
# ---------------------------------------------------------------------------

_NEURAL_PROMPT = "predecir\nFEATURES:\n  a: Scalar[0,10]\n  b: Scalar[0,10]\nSALIDA: resultado"


class TestValidateArchitectureHintsUnit:
    def test_none_or_empty_is_valid_and_becomes_empty_dict(self):
        assert validate_architecture_hints(None) == ({}, None)
        assert validate_architecture_hints({}) == ({}, None)

    def test_non_dict_is_rejected(self):
        cleaned, error = validate_architecture_hints("bad")
        assert cleaned == {}
        assert error is not None and "objeto" in error

    def test_unknown_key_is_rejected(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [(64, "relu")], "extra": 1})
        assert cleaned == {}
        assert error is not None and "no reconocidas" in error

    def test_hidden_layers_must_be_a_non_empty_list(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": []})
        assert cleaned == {}
        assert error is not None

    def test_layer_must_be_a_pair(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [(64,)]})
        assert cleaned == {}
        assert error is not None

    def test_non_integer_units_is_rejected(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [("bad", "relu")]})
        assert cleaned == {}
        assert error is not None and "entero" in error

    def test_bool_units_is_rejected(self):
        """`bool` es subclase de `int` en Python — se rechaza explícitamente."""
        cleaned, error = validate_architecture_hints({"hidden_layers": [(True, "relu")]})
        assert cleaned == {}
        assert error is not None

    def test_units_out_of_range_is_rejected(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [(0, "relu")]})
        assert cleaned == {} and error is not None
        cleaned, error = validate_architecture_hints({"hidden_layers": [(16385, "relu")]})
        assert cleaned == {} and error is not None

    def test_disallowed_activation_is_rejected(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [(64, "sigmoid")]})
        assert cleaned == {}
        assert error is not None and "activación" in error

    def test_depth_over_max_is_rejected(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [(64, "relu")] * 13})
        assert cleaned == {}
        assert error is not None and "profundidad" in error

    def test_exactly_max_depth_and_max_width_are_accepted(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [(16384, "relu")] * 12})
        assert error is None
        assert cleaned == {"hidden_layers": [(16384, "relu")] * 12}

    def test_valid_hint_passes_through_unchanged(self):
        cleaned, error = validate_architecture_hints({"hidden_layers": [(256, "relu"), (128, "relu")]})
        assert error is None
        assert cleaned == {"hidden_layers": [(256, "relu"), (128, "relu")]}


class TestArchitectureHintsPublicApiIntegration:
    """Las 3 reproducciones exactas de la auditoría, a través de
    `analyze_playground_request` (la función que expone `/api/analyze`) —
    deben devolver un error CONTROLADO (`ok: False`), nunca dejar escapar
    una excepción sin capturar."""

    def test_string_instead_of_dict_is_a_controlled_error_not_an_attributeerror(self):
        res = analyze_playground_request({
            "mode": "prompt", "prompt": _NEURAL_PROMPT, "architecture_hints": "bad",
        })
        assert res["ok"] is False
        assert "architecture_hints" in res["error"]

    def test_non_integer_units_is_a_controlled_error_not_a_typeerror(self):
        res = analyze_playground_request({
            "mode": "prompt", "prompt": _NEURAL_PROMPT,
            "architecture_hints": {"hidden_layers": [("bad", "relu")]},
        })
        assert res["ok"] is False
        assert "architecture_hints" in res["error"]

    def test_101_layers_is_rejected_not_silently_generated(self):
        res = analyze_playground_request({
            "mode": "prompt", "prompt": _NEURAL_PROMPT,
            "architecture_hints": {"hidden_layers": [(64, "relu")] * 100},
        })
        assert res["ok"] is False
        assert "profundidad" in res["error"]

    def test_valid_hint_still_reaches_the_dense_generator(self):
        res = analyze_playground_request({
            "mode": "prompt", "prompt": _NEURAL_PROMPT,
            "architecture_hints": {"hidden_layers": [(256, "relu"), (128, "relu")]},
        })
        assert res["ok"] is True
        assert "LAYER Dense units=256" in res["mxai"]
        assert "LAYER Dense units=128" in res["mxai"]
