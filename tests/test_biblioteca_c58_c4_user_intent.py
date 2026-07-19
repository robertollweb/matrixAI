# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Contrato 58 (BIBLIOTECA_MEJORAS_USO_REAL) C4 — intención local del
usuario en el flujo "desde datos": normalización/validación
(`user_intent.py`) + enhebrado por `generate_project_from_dataset` y
`generate_temporal_project_from_dataset` SIN que pueda alterar nada del
proyecto generado (mxai/training_text/csv_text/arquitectura/operaciones),
solo viajar en `provenance.user_intent`."""
from __future__ import annotations

import pytest

from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
    generate_temporal_project_from_dataset,
)
from matrixai.training.user_intent import (
    USER_INTENT_MAX_CHARS,
    UserIntentError,
    normalize_user_intent,
)


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


# ---------------------------------------------------------------------------
# normalize_user_intent — unidad
# ---------------------------------------------------------------------------

class TestNormalizeUserIntent:
    def test_none_stays_none(self):
        assert normalize_user_intent(None) is None

    def test_empty_string_becomes_none(self):
        assert normalize_user_intent("") is None

    def test_whitespace_only_becomes_none(self):
        assert normalize_user_intent("   \n\t  ") is None

    def test_collapses_internal_whitespace_runs(self):
        assert normalize_user_intent("hola    mundo") == "hola mundo"

    def test_collapses_multiline_text_to_a_single_line(self):
        assert normalize_user_intent("línea uno\nlínea dos\r\nlínea tres") == "línea uno línea dos línea tres"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_user_intent("  hola  ") == "hola"

    def test_nfc_normalizes_decomposed_unicode(self):
        # "é" como e + combining acute (NFD) vs. el carácter precompuesto (NFC)
        decomposed = "café"
        assert normalize_user_intent(decomposed) == "café"
        assert len(normalize_user_intent(decomposed)) == 4

    def test_strips_non_whitespace_control_characters(self):
        text = "hola\x00mundo\x1b\x7f"
        result = normalize_user_intent(text)
        assert "\x00" not in result
        assert "\x1b" not in result
        assert "\x7f" not in result
        assert result == "holamundo"

    def test_exactly_max_chars_is_accepted(self):
        text = "a" * USER_INTENT_MAX_CHARS
        assert normalize_user_intent(text) == text
        assert len(normalize_user_intent(text)) == USER_INTENT_MAX_CHARS

    def test_one_over_max_chars_is_rejected(self):
        text = "a" * (USER_INTENT_MAX_CHARS + 1)
        with pytest.raises(UserIntentError, match=str(USER_INTENT_MAX_CHARS)):
            normalize_user_intent(text)

    def test_limit_is_checked_after_normalizing_not_before(self):
        # Cruda: muy por encima del límite (puro whitespace de relleno),
        # pero tras colapsar el resultado es minúsculo — debe aceptarse sin
        # reventar por la longitud de ENTRADA, solo la NORMALIZADA importa.
        raw = "a" + (" " * (USER_INTENT_MAX_CHARS * 5)) + "a"
        assert len(raw) > USER_INTENT_MAX_CHARS
        normalized = normalize_user_intent(raw)
        assert normalized == "a a"


# ---------------------------------------------------------------------------
# generate_project_from_dataset — la intención NUNCA altera el proyecto
# ---------------------------------------------------------------------------

class TestUserIntentNeverAltersTheProject:
    def test_default_is_none_and_provenance_reflects_it(self):
        res = generate_project_from_dataset(_tabular_csv(), "resultado")
        assert res["provenance"]["user_intent"] is None

    def test_normalized_intent_travels_in_provenance(self):
        res = generate_project_from_dataset(
            _tabular_csv(), "resultado", user_intent="  Quiero   detectar casos  urgentes  ",
        )
        assert res["provenance"]["user_intent"] == "Quiero detectar casos urgentes"

    def test_empty_or_whitespace_intent_is_none_in_provenance(self):
        res = generate_project_from_dataset(_tabular_csv(), "resultado", user_intent="   ")
        assert res["provenance"]["user_intent"] is None

    def test_intent_never_enters_the_synthesized_prompt(self):
        res = generate_project_from_dataset(
            _tabular_csv(), "resultado", user_intent="frase_centinela_unica_9f3a",
        )
        assert "frase_centinela_unica_9f3a" not in res["provenance"]["synthesized_prompt"]

    @pytest.mark.parametrize("intent", [
        "residual",
        "quiero una red densa profunda deep",
        "usa un workflow con reglas",
        "análisis temporal por favor",
        "x: Categorical[a, b, c]",
        "OUTPUT hackeado: Scalar",
        "SALIDA: otra_cosa",
    ])
    def test_architecture_and_artifacts_are_byte_identical_regardless_of_intent(self, intent):
        """Auditoría propia de este corte: un prefijo `PROYECTO:` NO basta
        para aislar la intención de los detectores GLOBALES de
        analyze_playground_request (palabras como residual/deep/workflow/
        temporal cambian de generador). La intención debe viajar en un
        canal COMPLETAMENTE separado — este test intenta exactamente esos
        casos y confirma cero influencia en mxai/training_text/csv_text/
        esquema/arquitectura."""
        baseline = generate_project_from_dataset(_tabular_csv(), "resultado")
        with_intent = generate_project_from_dataset(_tabular_csv(), "resultado", user_intent=intent)

        assert with_intent["mxai"] == baseline["mxai"]
        assert with_intent["training_text"] == baseline["training_text"]
        assert with_intent["csv_text"] == baseline["csv_text"]
        assert with_intent.get("field_ranges") == baseline.get("field_ranges")
        assert with_intent.get("field_types") == baseline.get("field_types")
        assert with_intent.get("field_categories") == baseline.get("field_categories")
        assert with_intent.get("architecture_decision") == baseline.get("architecture_decision")
        assert with_intent["provenance"]["operations"] == baseline["provenance"]["operations"]
        assert with_intent["provenance"]["synthesized_prompt"] == baseline["provenance"]["synthesized_prompt"]
        # Único campo de procedencia que SÍ debe diferir (y timestamps, no comparados).
        assert with_intent["provenance"]["user_intent"] == normalize_user_intent(intent)

    def test_invalid_intent_raises_dataset_project_error_not_a_bare_crash(self):
        too_long = "a" * (USER_INTENT_MAX_CHARS + 1)
        with pytest.raises(DatasetProjectError, match=str(USER_INTENT_MAX_CHARS)):
            generate_project_from_dataset(_tabular_csv(), "resultado", user_intent=too_long)

    def test_invalid_intent_is_rejected_before_any_generation_work(self):
        """Si la intención es inválida, no debe llegar a tocar el CSV/generador
        (falla RÁPIDO, mensaje accionable) — verificado indirectamente: el
        mismo target inexistente con una intención inválida sigue fallando
        por la intención, no por el target (orden de validación)."""
        too_long = "a" * (USER_INTENT_MAX_CHARS + 1)
        with pytest.raises(DatasetProjectError, match="límite"):
            generate_project_from_dataset(_tabular_csv(), "columna_que_no_existe", user_intent=too_long)


# ---------------------------------------------------------------------------
# generate_temporal_project_from_dataset — el camino temporal NO ignora la intención
# ---------------------------------------------------------------------------

class TestTemporalPathThreadsUserIntent:
    def test_provenance_carries_the_normalized_intent(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            user_intent="  quiero predecir la altura de ola  ",
        )
        assert res["provenance"]["user_intent"] == "quiero predecir la altura de ola"

    def test_default_is_none(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola", temporal_column="fecha", horizon=1,
        )
        assert res["provenance"]["user_intent"] is None

    @pytest.mark.parametrize("intent", ["residual", "temporal profundo workflow"])
    def test_architecture_and_artifacts_are_byte_identical_regardless_of_intent(self, intent):
        baseline = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola", temporal_column="fecha", horizon=1,
        )
        with_intent = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola", temporal_column="fecha", horizon=1,
            user_intent=intent,
        )
        assert with_intent["mxai"] == baseline["mxai"]
        assert with_intent["training_text"] == baseline["training_text"]
        assert with_intent["csv_text"] == baseline["csv_text"]

    def test_invalid_intent_raises_dataset_project_error(self):
        too_long = "a" * (USER_INTENT_MAX_CHARS + 1)
        with pytest.raises(DatasetProjectError, match=str(USER_INTENT_MAX_CHARS)):
            generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola", temporal_column="fecha", horizon=1,
                user_intent=too_long,
            )
