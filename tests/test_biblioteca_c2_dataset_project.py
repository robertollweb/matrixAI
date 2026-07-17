# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C2 — modelo desde los datos.

`generate_project_from_dataset(csv_text, target_column)`: del esquema FINAL
(C1 + correcciones) + target elegido → prompt tipado canónico → generadores
EXISTENTES (`analyze_playground_request`, cero caminos paralelos) → mxai +
training_text + esquema S2 + CSV preparado + procedencia. Cubre los dos
ejemplos canónicos del contrato y el ciclo E2E hasta entrenar por el flujo
EXISTENTE (`_validate_training_csv`/`_submit_training_job`), sin ninguna
rama especial (invariante 4: "proyecto normal").
"""
from __future__ import annotations

import time
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None

from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
)


_SEMANA_SANTA_CSV = (
    "fecha,temperatura,tipo_dia,es_festivo,resultado\n"
    "2024-01-01,12.5,laboral,no,lluvia\n"
    "2024-01-02,15.0,laboral,no,sol\n"
    "2024-01-03,9.2,festivo,si,nublado\n"
    "2024-01-04,18.1,laboral,no,sol\n"
    "2024-01-05,11.0,laboral,no,lluvia\n"
    "2024-01-06,16.0,festivo,si,sol\n"
    "2024-01-07,10.0,laboral,no,nublado\n"
    "2024-01-08,17.5,laboral,no,sol\n"
    "2024-01-09,13.0,laboral,no,lluvia\n"
    "2024-01-10,14.5,festivo,si,nublado\n"
)


def _mar_csv(rows: int = 30) -> str:
    lines = ["fecha,altura_ola,periodo,temperatura_agua"]
    for i in range(rows):
        lines.append(
            f"2024-01-{(i % 28) + 1:02d},{2.0 + i * 0.1:.2f},{6 + i % 3},{15.0 + i * 0.05:.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Los dos ejemplos canónicos del contrato
# ---------------------------------------------------------------------------

class TestCanonicalExamples:
    def test_semana_santa_generates_classification_project(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        assert res["ok"]
        assert "ProbabilityMap[lluvia, nublado, sol]" in res["mxai"]
        assert "OUTPUT predicted_class" in res["mxai"]
        assert "TARGET predicted_class: Label[lluvia, nublado, sol]" in res["training_text"]
        # fecha (tipo date) nunca es feature; tipo_dia se expande a one-hot
        assert "fecha" not in res["mxai"]
        assert "tipo_dia__laboral" in res["mxai"] or "tipo_dia__festivo" in res["mxai"]

    def test_semana_santa_prepared_csv_matches_existing_validation_flow(self):
        """El flujo EXISTENTE (`_validate_training_csv`, sin ninguna rama
        especial) acepta el proyecto generado — invariante 4."""
        from matrixai.playground import _validate_training_csv
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        v = _validate_training_csv(
            res["mxai"], res["training_text"], res["csv_text"],
            field_ranges=res["field_ranges"],
        )
        assert v.get("ok"), v.get("error")

    def test_edited_category_vocabulary_controls_prompt_csv_and_provenance(self):
        res = generate_project_from_dataset(
            _SEMANA_SANTA_CSV, target_column="resultado",
            column_category_overrides={"tipo_dia": ["festivo", "laboral", "puente"]},
        )
        assert "Categorical[festivo, laboral, puente]" in res["provenance"]["synthesized_prompt"]
        assert "tipo_dia__puente" in res["csv_text"].splitlines()[0]
        assert res["provenance"]["column_category_overrides"]["tipo_dia"] == [
            "festivo", "laboral", "puente",
        ]

    def test_edited_category_vocabulary_cannot_omit_observed_values(self):
        with pytest.raises(DatasetProjectError, match="omite valores presentes"):
            generate_project_from_dataset(
                _SEMANA_SANTA_CSV, target_column="resultado",
                column_category_overrides={"tipo_dia": ["laboral", "puente"]},
            )

    def test_mar_time_series_regression_target_from_numeric_column(self):
        """Sin salida explícita (serie temporal pura): una columna numérica
        continua (no la fecha) sirve de target de regresión — el
        desplazamiento temporal real es C3, aún no construido; C2 cubre el
        caso tabular general de regresión."""
        res = generate_project_from_dataset(_mar_csv(), target_column="temperatura_agua")
        assert res["ok"]
        assert "OUTPUT predicted_value: Scalar" in res["mxai"]
        assert "fecha" not in res["mxai"]
        assert "altura_ola" in res["mxai"] and "periodo" in res["mxai"]


# ---------------------------------------------------------------------------
# Ciclo E2E real hasta entrenar (flujo existente, sin rama especial)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch required for training")
class TestEndToEndTraining:
    def test_classification_project_trains_and_learns_with_separable_signal(self):
        import random
        from matrixai.playground import _submit_training_job, _get_job_status

        rng = random.Random(7)
        rows = ["fecha,temperatura,tipo_dia,es_festivo,resultado"]
        for i in range(120):
            temp = rng.uniform(0, 30)
            label = "sol" if temp > 20 else ("lluvia" if temp < 10 else "nublado")
            tipo = rng.choice(["laboral", "festivo"])
            festivo = "si" if tipo == "festivo" else "no"
            rows.append(f"2024-01-{(i % 28) + 1:02d},{temp:.2f},{tipo},{festivo},{label}")
        csv_text = "\n".join(rows)

        res = generate_project_from_dataset(csv_text, target_column="resultado")
        job = _submit_training_job(
            res["mxai"], res["training_text"], res["csv_text"],
            epochs_override=60, field_ranges=res["field_ranges"],
        )
        assert job.get("ok"), job
        st = {}
        for _ in range(400):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.2)
        assert st["status"] == "done", st
        # señal separable de verdad -> debe aprender bien, no quedarse en azar (33%)
        assert st["accuracy"] > 0.8, st.get("accuracy")

    def test_regression_project_trains_end_to_end(self):
        from matrixai.playground import _run_playground_training
        res = generate_project_from_dataset(_mar_csv(rows=40), target_column="temperatura_agua")
        from matrixai.playground import _normalize_csv_with_ranges
        normalized = _normalize_csv_with_ranges(res["csv_text"], res["field_ranges"])
        tr = _run_playground_training(res["mxai"], res["training_text"], normalized, epochs_override=5)
        assert tr.get("ok"), tr.get("error")
        assert tr["task_kind"] == "regression"


# ---------------------------------------------------------------------------
# Exclusión de columnas no-feature (S2-C4 / invariante del contrato)
# ---------------------------------------------------------------------------

class TestColumnExclusion:
    def test_identifier_column_never_becomes_a_feature(self):
        rows = ["id,temp,resultado"] + [
            f"P{1000 + i},{10 + i % 5},{'a' if i % 2 else 'b'}" for i in range(15)
        ]
        res = generate_project_from_dataset("\n".join(rows), target_column="resultado")
        assert "id" not in res["mxai"].lower().split("vector")[1].split("end")[0]
        assert "id" not in res["csv_text"].splitlines()[0].split(",")

    def test_date_column_never_becomes_a_feature(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        header = res["csv_text"].splitlines()[0].split(",")
        assert "fecha" not in header

    def test_target_column_excluded_from_its_own_features(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        header = res["csv_text"].splitlines()[0].split(",")
        assert "resultado" not in header  # renombrada a predicted_class
        assert "predicted_class" in header


# ---------------------------------------------------------------------------
# Errores accionables (invariante 7)
# ---------------------------------------------------------------------------

class TestActionableErrors:
    def test_unknown_target_column_raises(self):
        with pytest.raises(DatasetProjectError, match="no existe en el CSV"):
            generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="no_existe")

    def test_identifier_as_target_raises(self):
        rows = ["id,temp,resultado"] + [
            f"P{1000 + i},{10 + i % 5},{'a' if i % 2 else 'b'}" for i in range(15)
        ]
        with pytest.raises(DatasetProjectError, match="no es un target válido"):
            generate_project_from_dataset("\n".join(rows), target_column="id")

    def test_date_as_target_raises(self):
        with pytest.raises(DatasetProjectError, match="no es un target válido"):
            generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="fecha")

    def test_colliding_labels_after_normalization_raise(self):
        rows = ["temp,y"] + [f"{10 + i % 5},{'Sí' if i % 2 else 'SI'}" for i in range(15)]
        with pytest.raises(DatasetProjectError, match="misma etiqueta tras"):
            generate_project_from_dataset("\n".join(rows), target_column="y")

    def test_constant_target_raises(self):
        # temp NO puede ser secuencial (15 valores únicos, >=10 filas caería
        # en identificador por unicidad, C1) — un feature con repeticiones
        # aísla de verdad el caso "target constante".
        rows = ["temp,y"] + [f"{i % 4},constante" for i in range(15)]
        with pytest.raises(DatasetProjectError, match="menos de 2 valores"):
            generate_project_from_dataset("\n".join(rows), target_column="y")

    def test_type_override_referencing_unknown_column_raises(self):
        with pytest.raises(DatasetProjectError, match="no existe en el CSV"):
            generate_project_from_dataset(
                _SEMANA_SANTA_CSV, target_column="resultado",
                column_type_overrides={"no_existe": "categorical"},
            )


# ---------------------------------------------------------------------------
# Correcciones del usuario ganan sobre lo inferido (invariante 8)
# ---------------------------------------------------------------------------

class TestUserOverridesWin:
    def test_type_override_changes_generated_schema(self):
        """Una columna que C1 tipó 'integer' se fuerza a 'categorical' — el
        prompt sintetizado debe reflejar la corrección, no lo inferido."""
        rows = ["codigo,temp,resultado"] + [
            f"{i % 4},{10 + i % 5},{'a' if i % 2 else 'b'}" for i in range(20)
        ]
        csv_text = "\n".join(rows)
        default = generate_project_from_dataset(csv_text, target_column="resultado")
        assert "codigo: Integer" in default["mxai"] or "codigo: Scalar" in default["mxai"]

        corrected = generate_project_from_dataset(
            csv_text, target_column="resultado",
            column_type_overrides={"codigo": "categorical"},
        )
        assert "codigo__" in corrected["mxai"]  # expandido a one-hot

    def test_range_override_changes_proposed_range(self):
        res = generate_project_from_dataset(
            _SEMANA_SANTA_CSV, target_column="resultado",
            column_range_overrides={"temperatura": (0.0, 50.0)},
        )
        assert res["field_ranges"]["temperatura"] == (0.0, 50.0)


# ---------------------------------------------------------------------------
# Reproducibilidad (invariante 5)
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_csv_and_target_produce_identical_project(self):
        res1 = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        res2 = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        assert res1["mxai"] == res2["mxai"]
        assert res1["training_text"] == res2["training_text"]
        assert res1["csv_text"] == res2["csv_text"]
        assert res1["provenance"]["raw_csv_sha256"] == res2["provenance"]["raw_csv_sha256"]
        assert res1["provenance"]["prepared_csv_sha256"] == res2["provenance"]["prepared_csv_sha256"]


# ---------------------------------------------------------------------------
# Procedencia (invariante 3)
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_has_required_fields(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        prov = res["provenance"]
        assert prov["source"] == "user_upload"
        assert prov["target_column"] == "resultado"
        assert prov["task"] == "classification"
        assert len(prov["raw_csv_sha256"]) == 64
        assert len(prov["prepared_csv_sha256"]) == 64
        assert prov["raw_csv_sha256"] != prov["prepared_csv_sha256"]
        assert "schema_inferred" in prov and "schema_final" in prov
        assert "synthesized_prompt" in prov
        assert "operations" in prov and prov["operations"]
        assert "created_at" in prov

    def test_provenance_records_user_overrides(self):
        res = generate_project_from_dataset(
            _SEMANA_SANTA_CSV, target_column="resultado",
            column_range_overrides={"temperatura": (0.0, 50.0)},
        )
        assert res["provenance"]["column_range_overrides"] == {"temperatura": [0.0, 50.0]}

    def test_raw_hash_is_of_the_original_bytes_not_normalized(self):
        """El hash CRUDO es del CSV tal cual llegó (antes de BOM/delimitador
        de C1) — reproducible incluso si el usuario re-sube el MISMO
        fichero con distinto salto de línea no debería mentir sobre qué
        bytes originales hasheó."""
        import hashlib
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        expected = hashlib.sha256(_SEMANA_SANTA_CSV.encode("utf-8")).hexdigest()
        assert res["provenance"]["raw_csv_sha256"] == expected
