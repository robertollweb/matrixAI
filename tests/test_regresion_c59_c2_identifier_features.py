# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 59 C2 — features enteras utilizables (decisión C + hallazgo 3):

1. Un entero casi-único (identifier para la heurística de C1, p.ej.
   "centigrados" 0..99) deja de vaciar el proyecto de features cuando es la
   ÚNICA columna disponible — se reconsidera como numérico ANTES de
   abortar, nunca después de que haya otras features reales.
2. Corregir el tipo de una columna a mano a number/integer
   (`column_type_overrides`) ya no exige ADEMÁS adivinar el rango en
   `column_range_overrides` — se recalcula desde los valores crudos.

Test de cierre del contrato (C2): `generate_project_from_dataset(csv_kelvin,
'prediccionKelvin')` funciona SIN overrides; un CSV con un id secuencial de
verdad + features reales sigue descartando el id."""
from __future__ import annotations

import time
import unittest

from matrixai.playground import _get_job_status, _submit_training_job
from matrixai.training.dataset_project import DatasetProjectError, generate_project_from_dataset


def _kelvin_csv(n: int = 100) -> str:
    lines = ["centigrados,prediccionKelvin"]
    for c in range(n):
        lines.append(f"{c},{c + 273.15}")
    return "\n".join(lines) + "\n"


def _id_plus_real_features_csv(n: int = 50) -> str:
    """`id` es un entero secuencial de verdad (0..n-1, único) — un
    identificador genuino que NUNCA debe colarse como feature mientras
    `edad` (baja cardinalidad, repetida) siga disponible."""
    lines = ["id,edad,compra"]
    for i in range(n):
        lines.append(f"{i},{20 + i % 5},{'si' if i % 2 == 0 else 'no'}")
    return "\n".join(lines) + "\n"


def _only_text_identifier_csv(n: int = 20) -> str:
    """Ningún entero, solo un identificador de TEXTO tipo UUID — no hay
    nada numérico que reconsiderar; el error original debe seguir
    disparándose (nunca se fuerza `float()` sobre texto)."""
    lines = ["uuid,resultado"]
    for i in range(n):
        lines.append(f"id-{i:08x}-abcd,{1.5 + i}")
    return "\n".join(lines) + "\n"


class TestKelvinCsvWorksWithoutAnyOverrides(unittest.TestCase):
    """El test de cierre literal del contrato."""

    def test_generates_without_overrides(self):
        proj = generate_project_from_dataset(_kelvin_csv(), "prediccionKelvin")
        self.assertIn("centigrados", proj["provenance"]["feature_name_map"])
        self.assertIsNotNone(proj.get("target_range"))

    def test_reconsideration_leaves_a_trace_in_provenance(self):
        """Hallazgo de auditoría C2: la reconsideración automática de
        'centigrados' (identifier -> numérico, sin ningún
        column_type_overrides del usuario) tiene que quedar registrada en
        `provenance["operations"]` — es la única otra vía por la que el
        tipo de una columna cambia, y las demás sí dejan rastro."""
        proj = generate_project_from_dataset(_kelvin_csv(), "prediccionKelvin")
        ops = proj["provenance"]["operations"]
        self.assertIn("reconsidered_identifier_as_feature:centigrados", ops)
        self.assertEqual(proj["provenance"]["column_type_overrides"], {})

    def test_reconsidered_feature_learns_end_to_end(self):
        """No basta con que genere — tiene que APRENDER (el objetivo del
        contrato entero, no solo 'no reventar')."""
        proj = generate_project_from_dataset(_kelvin_csv(), "prediccionKelvin")
        submitted = _submit_training_job(
            proj["mxai"], proj["training_text"], proj["csv_text"],
            field_ranges=proj.get("field_ranges"), target_range=tuple(proj["target_range"]),
        )
        self.assertTrue(submitted.get("ok"), submitted.get("error"))
        job_id = submitted["job_id"]
        status: dict = {}
        for _ in range(300):
            status = _get_job_status(job_id)
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.2)
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertFalse(status.get("model_collapsed"))
        self.assertGreaterEqual(status["r2"], 0.99)


class TestIdentifierStillExcludedWhenRealFeaturesExist(unittest.TestCase):
    """La otra mitad del test de cierre: reconsiderar es el ÚLTIMO recurso,
    nunca el comportamiento por defecto cuando hay features reales."""

    def test_id_column_excluded_when_other_features_available(self):
        proj = generate_project_from_dataset(_id_plus_real_features_csv(), "compra")
        prov = proj["provenance"]
        self.assertNotIn("id", prov["feature_name_map"])
        self.assertIn("id", prov["excluded_columns"])
        self.assertIn("edad", prov["feature_name_map"])

    def test_no_reconsideration_trace_when_it_never_fires(self):
        """Cuando la reconsideración no se dispara (hay features reales
        disponibles), `operations` no debe contener ninguna entrada
        `reconsidered_identifier_as_feature:*` — ni siquiera para 'id'."""
        proj = generate_project_from_dataset(_id_plus_real_features_csv(), "compra")
        ops = proj["provenance"]["operations"]
        self.assertFalse(any(op.startswith("reconsidered_identifier_as_feature:") for op in ops))


class TestNonNumericIdentifierStillFailsWithActionableError(unittest.TestCase):
    """Un identificador de TEXTO (UUID-like) nunca se reconsidera — forzar
    `float()` sobre él sería un error distinto y peor. El mensaje accionable
    original debe seguir aaplicando."""

    def test_uuid_like_identifier_alone_still_raises(self):
        with self.assertRaises(DatasetProjectError) as ctx:
            generate_project_from_dataset(_only_text_identifier_csv(), "resultado")
        self.assertIn("no hay nada con lo que entrenar", str(ctx.exception))


class TestTypeOverrideNoLongerNeedsARangeOverrideToo(unittest.TestCase):
    """Hallazgo 3: `column_type_overrides` solo (sin `column_range_
    overrides`) ya basta para una columna que C1 clasificó sin rango
    (identifier) — antes reventaba con 'no tiene un rango numérico
    calculable'."""

    def test_type_override_alone_is_enough_for_an_identifier_column(self):
        proj = generate_project_from_dataset(
            _kelvin_csv(), "prediccionKelvin",
            column_type_overrides={"centigrados": "number"},
        )
        self.assertIn("centigrados", proj["provenance"]["feature_name_map"])

    def test_explicit_range_override_still_wins_over_the_recomputed_one(self):
        """Invariante 8 (el usuario manda): si el usuario SÍ declara un
        rango explícito, ese gana sobre el recalculado automáticamente."""
        proj = generate_project_from_dataset(
            _kelvin_csv(), "prediccionKelvin",
            column_type_overrides={"centigrados": "number"},
            column_range_overrides={"centigrados": (-50.0, 200.0)},
        )
        lo, hi = proj["field_ranges"]["centigrados"]
        self.assertEqual((lo, hi), (-50.0, 200.0))


if __name__ == "__main__":
    unittest.main()
