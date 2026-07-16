# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES — autoauditoría C1, sugerencias
implementadas: normalización de CSV real de entrada (BOM UTF-8, delimitador
';') en el punto ÚNICO de entrada (`matrixai.training.data.normalize_csv_text`),
enhebrado en los 3 sitios donde un CSV externo entra al producto
(`_validate_training_csv`, `_run_playground_training`, `_submit_training_job`
en playground.py) — no solo en `analyze_dataset_csv` (C1 propio). Antes de
este fix, un CSV de Excel (BOM y/o ';') subido por el flujo EXISTENTE
(validar/entrenar) fallaba con "faltan columnas" sobre un CSV que, a simple
vista, las tenía.
"""
from __future__ import annotations

from matrixai.training.data import (
    normalize_csv_delimiter,
    normalize_csv_text,
    strip_csv_bom,
)


# ---------------------------------------------------------------------------
# Helpers compartidos, unitarios
# ---------------------------------------------------------------------------

class TestStripCsvBom:
    def test_removes_leading_bom(self):
        assert strip_csv_bom("﻿fecha,valor\n1,2\n") == "fecha,valor\n1,2\n"

    def test_noop_without_bom(self):
        text = "fecha,valor\n1,2\n"
        assert strip_csv_bom(text) == text

    def test_only_strips_leading_bom_not_mid_text(self):
        text = "fecha,va﻿lor\n1,2\n"
        assert strip_csv_bom(text) == text


class TestNormalizeCsvDelimiter:
    def test_converts_semicolon_header_to_comma(self):
        out = normalize_csv_delimiter("a;b;c\n1;2;3\n")
        assert out == "a,b,c\n1,2,3\n"

    def test_noop_when_comma_already_present_in_header(self):
        text = "a,b;c\n1,2;3\n"  # cabecera con coma -> NO se toca (señal conservadora)
        assert normalize_csv_delimiter(text) == text

    def test_noop_for_plain_comma_csv(self):
        text = "a,b,c\n1,2,3\n"
        assert normalize_csv_delimiter(text) == text

    def test_noop_for_empty_text(self):
        assert normalize_csv_delimiter("") == ""

    def test_decimal_comma_value_stays_quoted_not_corrupted(self):
        """Límite documentado: la reescritura es estructuralmente segura
        (usa csv.reader/writer, no str.replace) — un valor con coma decimal
        queda entrecomillado, sin partirse en dos columnas."""
        out = normalize_csv_delimiter("a;b\n1;12,5\n")
        assert out == 'a,b\n1,"12,5"\n'
        # y ese valor se re-lee como UNA sola celda, no dos
        import csv
        import io
        rows = list(csv.reader(io.StringIO(out)))
        assert rows == [["a", "b"], ["1", "12,5"]]

    def test_single_column_semicolon_free_csv_untouched(self):
        # una sola columna, sin ';' en la cabecera -> nada que normalizar
        text = "a\n1\n2\n"
        assert normalize_csv_delimiter(text) == text


class TestNormalizeCsvText:
    def test_bom_and_semicolon_together(self):
        out = normalize_csv_text("﻿fecha;valor\n2024-01-01;5\n")
        assert out == "fecha,valor\n2024-01-01,5\n"

    def test_plain_csv_roundtrips_unchanged(self):
        text = "a,b\n1,2\n3,4\n"
        assert normalize_csv_text(text) == text


# ---------------------------------------------------------------------------
# Regresión: el flujo EXISTENTE (validar / entrenar), no solo analyze_dataset_csv
# ---------------------------------------------------------------------------

_PROMPT = (
    "clasificar\nFEATURES:\n  temp: Scalar en [0, 100]\n  humedad: Scalar en [0, 100]\n"
    "SALIDA: y: ProbabilityMap[OK, KO]"
)


def _generated():
    from matrixai.training.dense_generator import DenseNetworkGenerator
    return DenseNetworkGenerator().generate(_PROMPT)


class TestExistingUploadFlowAcceptsRealWorldCsv:
    """Antes del fix: un CSV de Excel (BOM y/o ';') fallaba aquí con 'faltan
    columnas' — el hueco que la autoauditoría de C1 encontró en el flujo
    EXISTENTE (no en el código nuevo de C1)."""

    def test_validate_training_csv_accepts_bom(self):
        from matrixai.playground import _validate_training_csv
        gen = _generated()
        csv_bom = "﻿temp,humedad,predicted_class\n50,60,ok\n30,40,ko\n70,20,ok\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_bom)
        assert r.get("ok"), r.get("error")

    def test_validate_training_csv_accepts_semicolon(self):
        from matrixai.playground import _validate_training_csv
        gen = _generated()
        csv_semi = "temp;humedad;predicted_class\n50;60;ok\n30;40;ko\n70;20;ok\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_semi)
        assert r.get("ok"), r.get("error")

    def test_validate_training_csv_accepts_bom_plus_semicolon(self):
        from matrixai.playground import _validate_training_csv
        gen = _generated()
        csv_text = "﻿temp;humedad;predicted_class\n50;60;ok\n30;40;ko\n70;20;ok\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_text)
        assert r.get("ok"), r.get("error")

    def test_sync_training_runs_end_to_end_with_bom_csv(self):
        """No solo 'la validación pasa' — el entrenamiento SÍ arranca y
        termina (prueba que la normalización llega hasta el trainer, no
        solo hasta `_validate_training_csv`)."""
        from matrixai.playground import _run_playground_training
        gen = _generated()
        csv_bom = (
            "﻿temp,humedad,predicted_class\n"
            "50,60,ok\n30,40,ko\n70,20,ok\n20,80,ko\n90,10,ok\n40,50,ko\n"
        )
        r = _run_playground_training(gen.mxai_text, gen.training_text, csv_bom, epochs_override=1)
        assert r.get("ok"), r.get("error")

    def test_sync_training_runs_end_to_end_with_semicolon_csv(self):
        from matrixai.playground import _run_playground_training
        gen = _generated()
        csv_semi = (
            "temp;humedad;predicted_class\n"
            "50;60;ok\n30;40;ko\n70;20;ok\n20;80;ko\n90;10;ok\n40;50;ko\n"
        )
        r = _run_playground_training(gen.mxai_text, gen.training_text, csv_semi, epochs_override=1)
        assert r.get("ok"), r.get("error")


# ---------------------------------------------------------------------------
# Auditoría de las sugerencias — 3 hallazgos, cada uno con su regresión
# ---------------------------------------------------------------------------

class TestAuditOfSuggestions:
    def test_multiline_quoted_value_survives_delimiter_rewrite(self):
        """[H-B, MEDIA] la versión inicial partía en líneas ANTES de parsear
        — un campo entrecomillado con salto de línea se corrompía EN SILENCIO
        ("linea1\\nlinea2" → "linea1linea2", salto perdido y celdas
        fusionadas). Con StringIO el valor sobrevive el round-trip entero."""
        import csv
        import io
        out = normalize_csv_delimiter('a;b\n"linea1\nlinea2";2\n')
        rows = list(csv.reader(io.StringIO(out)))
        assert rows == [["a", "b"], ["linea1\nlinea2", "2"]]

    def test_csv_data_adapter_reads_bom_file(self, tmp_path):
        """[H-A, MEDIA] el camino de FICHERO (CLI `mx train`, adapters) abría
        con encoding="utf-8" — un CSV de Excel EN DISCO seguía ensuciando la
        primera columna ("\\ufefftemp") aunque el flujo de subida ya
        estuviera arreglado. utf-8-sig lee igual un fichero sin BOM."""
        from matrixai.training.data import CSVDataAdapter
        p = tmp_path / "datos.csv"
        p.write_bytes("temp,humedad,y\n50,60,ok\n30,40,ko\n".encode("utf-8-sig"))
        adapter = CSVDataAdapter(p, "In", ["temp", "humedad"], "y", ["ok", "ko"])
        examples = adapter.examples()
        assert len(examples) == 2
        assert examples[0].vector == [50.0, 60.0]

    def test_csv_data_adapter_still_reads_plain_file(self, tmp_path):
        from matrixai.training.data import CSVDataAdapter
        p = tmp_path / "datos.csv"
        p.write_text("temp,humedad,y\n50,60,ok\n", encoding="utf-8")
        adapter = CSVDataAdapter(p, "In", ["temp", "humedad"], "y", ["ok", "ko"])
        assert adapter.examples()[0].vector == [50.0, 60.0]

    def test_training_verifier_accepts_bom_file(self, tmp_path):
        """El TrainingVerifier (valida el dataset desde FICHERO) también
        abre con utf-8-sig — el mismo CSV de Excel pasa la verificación."""
        gen = _generated()
        import re
        model_name = re.search(r"MODEL\s+(\S+)", gen.training_text).group(1)
        csv_name = re.search(r'csv\("([^"]+)"\)', gen.training_text).group(1)
        (tmp_path / model_name).write_text(gen.mxai_text, encoding="utf-8")
        rows = "temp,humedad,predicted_class\n" + "\n".join(
            f"0.{i},0.{9 - i},{'ok' if i % 2 else 'ko'}" for i in range(6)
        )
        (tmp_path / csv_name).write_bytes(rows.encode("utf-8-sig"))
        from matrixai.training.parser import parse_training_text
        from matrixai.training.verifier import TrainingVerifier
        training = parse_training_text(gen.training_text)
        report = TrainingVerifier().verify(training, base_path=tmp_path)
        assert report.ok, report.errors

    def test_size_limit_checked_before_normalization_rewrite(self, monkeypatch):
        """[orden, hosted anti-DoS] un payload sobredimensionado se rechaza
        ANTES de pagar la reescritura O(n) — normalize_csv_text NO llega a
        llamarse (verificado con spy), ni en validate ni en analyze."""
        import matrixai.training.data as data_mod
        monkeypatch.setenv("MATRIXAI_MAX_CSV_BYTES", "50")
        called = []
        orig = data_mod.normalize_csv_text
        monkeypatch.setattr(
            data_mod, "normalize_csv_text",
            lambda t: (called.append(1), orig(t))[1],
        )
        big = "a;b\n" + "\n".join(f"{i};{i}" for i in range(100))

        from matrixai.playground import _validate_training_csv
        r = _validate_training_csv("PROJECT X", "MODEL X", big)
        assert not r.get("ok")
        assert "límite" in (r.get("error") or "")
        assert not called

        from matrixai.training.dataset_analysis import (
            DatasetAnalysisError,
            analyze_dataset_csv,
        )
        import pytest as _pytest
        with _pytest.raises(DatasetAnalysisError, match="límite"):
            analyze_dataset_csv(big)
        assert not called
