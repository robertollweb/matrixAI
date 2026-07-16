# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C1 — inferencia de esquema desde datos reales.

Cubre cada heurística de tipo (fecha, boolean, entero, decimal, identificador,
categórica), los dos ejemplos canónicos del contrato (semana santa: columna de
salida explícita; estado de la mar: serie temporal sin salida explícita, target
derivado por desplazamiento en C3), bordes (columna mixta, todo nulos, una sola
fila, dos formatos de fecha, cardinalidad justo en el umbral, numérica
disfrazada de texto), determinismo, y CSV ilegible → error accionable
(invariante 7 del contrato).
"""
from __future__ import annotations

import pytest

from matrixai.training.dataset_analysis import (
    DatasetAnalysisError,
    analyze_dataset_csv,
)
from matrixai.training.dense_generator import _ONEHOT_MAX


# ---------------------------------------------------------------------------
# Los dos ejemplos canónicos del contrato
# ---------------------------------------------------------------------------

class TestCanonicalExamples:
    def test_semana_santa_output_column_detected_as_top_target(self):
        """Salida explícita (lluvia/nublado/sol) — clasificación, nombre
        típico + última columna + poca cardinalidad: gana claro."""
        csv_text = (
            "fecha,temperatura,tipo_dia,es_festivo,resultado\n"
            "2024-01-01,12.5,laboral,no,lluvia\n"
            "2024-01-02,15.0,laboral,no,sol\n"
            "2024-01-03,9.2,festivo,si,nublado\n"
            "2024-01-04,18.1,laboral,no,sol\n"
            "2024-01-05,11.0,laboral,no,lluvia\n"
        )
        r = analyze_dataset_csv(csv_text)
        assert r["ok"]
        assert r["columns"]["resultado"]["type"] == "categorical"
        assert r["columns"]["resultado"]["vocabulary"] == ["lluvia", "nublado", "sol"]
        top = r["target_candidates"][0]
        assert top["column"] == "resultado"
        assert top["task"] == "classification"
        assert r["temporal_columns"] == ["fecha"]

    def test_mar_time_series_no_explicit_output_still_proposes_regression_targets(self):
        """Sin columna de salida — todas las numéricas quedan como candidatas
        de regresión (el target real se deriva por desplazamiento en C3);
        columna temporal detectada para ofrecer el modo serie temporal."""
        rows = ["fecha,altura_ola,periodo,temperatura_agua"]
        for i in range(20):
            rows.append(f"2024-01-{i + 1:02d},{2.0 + i * 0.1:.2f},{6 + i % 3},{15.0 + i * 0.05:.2f}")
        r = analyze_dataset_csv("\n".join(rows))
        assert r["ok"]
        assert r["temporal_columns"] == ["fecha"]
        candidate_cols = {c["column"] for c in r["target_candidates"]}
        assert "altura_ola" in candidate_cols
        assert "temperatura_agua" in candidate_cols
        assert "fecha" not in candidate_cols  # nunca se propone la fecha como target
        reg = next(c for c in r["target_candidates"] if c["column"] == "altura_ola")
        assert reg["task"] == "regression"


# ---------------------------------------------------------------------------
# Heurísticas de tipo, una por una
# ---------------------------------------------------------------------------

class TestTypeDetection:
    def test_integer_column(self):
        csv_text = "n\n1\n2\n3\n4\n5\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["n"]["type"] == "integer"
        assert r["columns"]["n"]["observed_range"] == [1, 5]

    def test_number_column_with_decimals(self):
        csv_text = "precio\n10.5\n22.3\n8.1\n45.0\n12.2\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["precio"]["type"] == "number"
        assert r["columns"]["precio"]["observed_range"] == [8.1, 45.0]

    def test_proposed_range_has_margin_over_observed(self):
        csv_text = "x\n0\n10\n"
        r = analyze_dataset_csv(csv_text)
        proposed = r["columns"]["x"]["proposed_range"]
        observed = r["columns"]["x"]["observed_range"]
        assert proposed[0] < observed[0]
        assert proposed[1] > observed[1]

    def test_proposed_range_with_zero_span_uses_absolute_margin(self):
        csv_text = "x\n5\n5\n5\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["x"]["observed_range"] == [5, 5]
        lo, hi = r["columns"]["x"]["proposed_range"]
        assert lo < 5 < hi

    @pytest.mark.parametrize("true_val,false_val", [
        ("true", "false"), ("si", "no"), ("sí", "no"), ("yes", "no"),
        ("1", "0"), ("verdadero", "falso"), ("Y", "N"),
    ])
    def test_boolean_columns_various_token_pairs(self, true_val, false_val):
        csv_text = f"flag\n{true_val}\n{false_val}\n{true_val}\n{false_val}\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["flag"]["type"] == "boolean"

    def test_boolean_case_insensitive(self):
        csv_text = "flag\nTRUE\nFalse\ntrue\nFALSE\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["flag"]["type"] == "boolean"

    def test_pure_zero_one_is_boolean_not_integer(self):
        """Ambigüedad documentada: 0/1 puro se lee como boolean (mismos
        tokens que predict_template.py), no como entero."""
        csv_text = "flag\n0\n1\n0\n1\n1\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["flag"]["type"] == "boolean"

    def test_zero_one_plus_other_integers_is_integer_not_boolean(self):
        csv_text = "n\n0\n1\n2\n0\n1\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["n"]["type"] == "integer"

    def test_iso_date_format_detected(self):
        csv_text = "fecha\n2024-01-01\n2024-06-15\n2024-12-31\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["fecha"]["type"] == "date"
        assert r["columns"]["fecha"]["date_format"] == "%Y-%m-%d"

    def test_dmy_date_format_detected(self):
        """Dos formatos de fecha (contrato): ISO en un test, D/M/Y en otro —
        cada columna en SU propio CSV se detecta con SU formato correcto."""
        csv_text = "fecha\n01/06/2024\n15/06/2024\n31/12/2024\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["fecha"]["type"] == "date"
        assert r["columns"]["fecha"]["date_format"] == "%d/%m/%Y"

    def test_mixed_inconsistent_date_formats_not_classified_as_date(self):
        """Columna con fechas en formatos DISTINTOS entre filas — ningún
        formato casa el 100%, cae a categórica (no se inventa una fecha)."""
        csv_text = "fecha\n2024-01-01\n15/06/2024\n2024-12-31\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["fecha"]["type"] != "date"

    def test_identifier_high_uniqueness_string(self):
        rows = ["id,valor"] + [f"P{1000 + i},{i}" for i in range(15)]
        r = analyze_dataset_csv("\n".join(rows))
        assert r["columns"]["id"]["type"] == "identifier"
        assert r["columns"]["id"]["unique_ratio"] == 1.0

    def test_identifier_sequential_integers(self):
        rows = ["id,valor"] + [f"{i},{i * 2}" for i in range(1, 16)]
        r = analyze_dataset_csv("\n".join(rows))
        assert r["columns"]["id"]["type"] == "identifier"

    def test_identifier_excluded_from_target_candidates(self):
        rows = ["id,resultado"] + [f"P{1000 + i},{'si' if i % 2 else 'no'}" for i in range(15)]
        r = analyze_dataset_csv("\n".join(rows))
        candidate_cols = {c["column"] for c in r["target_candidates"]}
        assert "id" not in candidate_cols

    def test_categorical_low_cardinality(self):
        csv_text = "color\nrojo\nverde\nazul\nrojo\nverde\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["color"]["type"] == "categorical"
        assert r["columns"]["color"]["vocabulary"] == ["azul", "rojo", "verde"]

    def test_numeric_looking_strings_are_numeric_not_categorical(self):
        """'numérica disfrazada de texto': valores que son texto por ser CSV,
        pero semánticamente números — deben tipar numérico, no categórico."""
        csv_text = "edad\n18\n25\n33\n41\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["edad"]["type"] == "integer"


# ---------------------------------------------------------------------------
# Cardinalidad justo en el umbral (_ONEHOT_MAX) para la sugerencia de tarea
# ---------------------------------------------------------------------------

class TestCardinalityThreshold:
    def _csv_with_cardinality(self, n: int) -> str:
        rows = ["cat,y"]
        for i in range(n * 3):
            rows.append(f"v{i % n},{'a' if i % 2 else 'b'}")
        return "\n".join(rows)

    def test_at_threshold_still_classification(self):
        r = analyze_dataset_csv(self._csv_with_cardinality(_ONEHOT_MAX))
        cat = next(c for c in r["target_candidates"] if c["column"] == "cat")
        assert cat["task"] == "classification"

    def test_one_above_threshold_falls_back_to_categorical_type_but_not_classification_by_cardinality(self):
        # El TIPO sigue siendo categórico (cardinalidad no decide el tipo),
        # pero como target dejaría de puntuar por "pocas categorías".
        n = _ONEHOT_MAX + 1
        r = analyze_dataset_csv(self._csv_with_cardinality(n))
        assert r["columns"]["cat"]["type"] == "categorical"
        cat = next((c for c in r["target_candidates"] if c["column"] == "cat"), None)
        assert cat is not None
        assert "pocas categorías" not in " ".join(cat["reasons"])


# ---------------------------------------------------------------------------
# Nulos y duplicados
# ---------------------------------------------------------------------------

class TestNullsAndDuplicates:
    def test_null_tokens_counted(self):
        # Nota: `csv.DictReader` descarta líneas EN BLANCO por completo (no
        # llegan como fila con celda vacía) — este CSV usa una segunda
        # columna para forzar celdas realmente vacías/NA sin depender de
        # líneas en blanco.
        csv_text = "x,y\n1,1\n,1\nNA,1\nN/A,1\nnull,1\n5,1\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["x"]["null_count"] == 4

    def test_all_null_column_returns_unknown_type_without_crashing(self):
        csv_text = "x,y\n,1\n,2\nNA,3\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["x"]["type"] == "unknown"
        assert r["columns"]["x"]["null_count"] == 3

    def test_duplicate_rows_counted(self):
        csv_text = "a,b\n1,2\n1,2\n3,4\n1,2\n"
        r = analyze_dataset_csv(csv_text)
        assert r["duplicate_rows"] == 2

    def test_no_duplicates_reports_zero(self):
        csv_text = "a,b\n1,2\n3,4\n5,6\n"
        r = analyze_dataset_csv(csv_text)
        assert r["duplicate_rows"] == 0


# ---------------------------------------------------------------------------
# Bordes explícitos del contrato
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_mixed_type_column_falls_back_to_categorical(self):
        """Columna mixta (números y texto entremezclados) — no numérica pura
        (no todos parsean), no identificador (poca unicidad relativa), cae a
        categórica sin reventar."""
        csv_text = "raro\n1\nabc\n2\ndef\n1\nabc\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["raro"]["type"] == "categorical"
        assert set(r["columns"]["raro"]["vocabulary"]) == {"1", "2", "abc", "def"}

    def test_single_row_dataset_does_not_crash(self):
        # '42' (no '1'/'0') evita el solape documentado boolean/entero de un
        # único valor — ver test_pure_zero_one_is_boolean_not_integer.
        csv_text = "a,b,c\n42,hola,2024-01-01\n"
        r = analyze_dataset_csv(csv_text)
        assert r["ok"]
        assert r["rows_total"] == 1
        assert r["columns"]["a"]["type"] == "integer"
        assert r["columns"]["b"]["type"] == "categorical"
        assert r["columns"]["c"]["type"] == "date"

    def test_low_row_count_high_uniqueness_not_misflagged_as_identifier(self):
        """Con pocas filas, 'todo distinto' no es señal fiable de
        identificador (umbral _IDENTIFIER_MIN_ROWS)."""
        csv_text = "nombre\nana\nbeatriz\ncarlos\n"
        r = analyze_dataset_csv(csv_text)
        assert r["columns"]["nombre"]["type"] != "identifier"

    def test_illegible_csv_raises_actionable_error(self):
        with pytest.raises(DatasetAnalysisError):
            analyze_dataset_csv("")

    def test_whitespace_only_csv_raises(self):
        with pytest.raises(DatasetAnalysisError):
            analyze_dataset_csv("   \n  \n")

    def test_header_only_no_data_rows_raises(self):
        with pytest.raises(DatasetAnalysisError):
            analyze_dataset_csv("a,b,c\n")

    def test_oversized_csv_raises_actionable_error(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_CSV_BYTES", "50")
        big_csv = "a,b\n" + "\n".join(f"{i},{i}" for i in range(100))
        with pytest.raises(DatasetAnalysisError):
            analyze_dataset_csv(big_csv)

    def test_rows_capped_at_max_rows_with_warning(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_ROWS", "5")
        csv_text = "x\n" + "\n".join(str(i) for i in range(20))
        r = analyze_dataset_csv(csv_text)
        assert r["rows_total"] == 20
        assert r["rows_analyzed"] == 5
        assert "rows_capped_warning" in r


# ---------------------------------------------------------------------------
# Autoauditoría C1 — 6 hallazgos empíricos, cada uno con su regresión
# ---------------------------------------------------------------------------

class TestAuditFindings:
    def test_hourly_timestamps_detected_as_date_not_identifier(self):
        """[MEDIA] Open-Meteo hourly emite '2024-01-01T00:00' — sin formatos
        con hora, el timestamp del ejemplo canónico del mar caía a
        'identifier' (unicidad) y el modo serie temporal nunca se ofrecía."""
        rows = ["time,wave_height"]
        for i in range(20):
            rows.append(f"2024-01-01T{i:02d}:00,{2.0 + i * 0.1:.2f}")
        r = analyze_dataset_csv("\n".join(rows))
        assert r["columns"]["time"]["type"] == "date"
        assert r["temporal_columns"] == ["time"]

    def test_datetime_with_seconds_detected_as_date(self):
        rows = ["ts,v"] + [f"2024-01-01 {i:02d}:30:00,{i}" for i in range(15)]
        r = analyze_dataset_csv("\n".join(rows))
        assert r["columns"]["ts"]["type"] == "date"

    def test_excel_bom_stripped_from_first_column_name(self):
        """[MEDIA] los CSV de Excel llevan BOM UTF-8 — sin quitarlo, la
        primera columna se llamaba '\\ufefffecha' y nada casaba después."""
        r = analyze_dataset_csv("﻿fecha,valor\n2024-01-01,5\n2024-01-02,6\n")
        assert r["column_order"] == ["fecha", "valor"]
        assert r["columns"]["fecha"]["type"] == "date"

    def test_duplicate_header_names_raise_actionable_error(self):
        """[MEDIA] DictReader se queda con el último valor por nombre — la
        columna duplicada se analizaba a medias y salía DOS veces en
        target_candidates. Ahora: error accionable con los nombres."""
        with pytest.raises(DatasetAnalysisError, match="repite nombres"):
            analyze_dataset_csv("a,b,a\n1,2,3\n4,5,6\n")

    def test_empty_header_name_raises_actionable_error(self):
        with pytest.raises(DatasetAnalysisError, match="sin nombre"):
            analyze_dataset_csv("a,b,\n1,2,3\n4,5,6\n")

    def test_constant_column_never_a_target_candidate(self):
        """[BAJA] una constante llamada 'y' llegaba a proponerse PRIMERA
        (última columna + nombre típico) como 'valores numéricos continuos'
        — siendo un valor que nunca cambia."""
        r = analyze_dataset_csv("x,y\n1,5.0\n2,5.0\n3,5.0\n")
        assert "y" not in {c["column"] for c in r["target_candidates"]}

    def test_constant_boolean_also_excluded(self):
        r = analyze_dataset_csv("x,activo\n1,si\n2,si\n3,si\n")
        assert "activo" not in {c["column"] for c in r["target_candidates"]}

    def test_high_cardinality_vocabulary_capped_with_sample(self):
        """[MEDIA] alineación con el contrato ('vocabulario si categórica de
        cardinalidad BAJA'): una categórica de 600 valores metía 600 entradas
        en la respuesta; ahora muestra de _ONEHOT_MAX + flag de truncado."""
        rows = ["ciudad,y"]
        for i in range(1000):
            rows.append(f"ciudad_{i % 600},{'a' if i % 2 else 'b'}")
        r = analyze_dataset_csv("\n".join(rows))
        info = r["columns"]["ciudad"]
        assert "vocabulary" not in info
        assert info["vocabulary_truncated"] is True
        assert len(info["vocabulary_sample"]) == _ONEHOT_MAX
        assert info["cardinality"] == 600

    def test_low_cardinality_keeps_full_vocabulary(self):
        r = analyze_dataset_csv("color,y\nrojo,a\nverde,b\nazul,a\nrojo,b\n")
        assert r["columns"]["color"]["vocabulary"] == ["azul", "rojo", "verde"]
        assert "vocabulary_truncated" not in r["columns"]["color"]


# ---------------------------------------------------------------------------
# Determinismo
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_csv_produces_identical_result(self):
        csv_text = (
            "fecha,temperatura,tipo_dia,resultado\n"
            "2024-01-01,12.5,laboral,lluvia\n"
            "2024-01-02,15.0,festivo,sol\n"
            "2024-01-03,9.2,laboral,nublado\n"
        )
        r1 = analyze_dataset_csv(csv_text)
        r2 = analyze_dataset_csv(csv_text)
        assert r1 == r2
