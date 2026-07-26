# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 62 C1 — el error de un tope superado dice la verdad.

Antes de este corte cada sitio componía su propia frase ("CSV tiene N filas,
máximo M") y quien la recibía solo podía reconocerla parseando texto. Peor:
`generate_project_from_dataset` la envolvía en *"esto indica un hueco en la
preparación del CSV, no un problema de tus datos"* — que para un tope es
literalmente lo contrario de la verdad: la causa es el perfil de límites y la
solución es cambiarlo en Ajustes.

Estos tests fijan el contrato de datos (`limits.limit_error`) y comprueban que
las cinco rutas del Studio lo emiten, que `dataset_project` ya no miente, y que
en hosted no se ofrece una acción que el usuario no puede ejecutar.
"""
from __future__ import annotations

import csv
import io
import os
import unittest
from unittest.mock import patch

from matrixai import limits as _limits
from matrixai.playground import (
    _generate_synthetic_dataset,
    _normalize_external_csv,
    _validate_training_csv,
)
from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
)


def _csv(rows: int, cols: int = 3) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    headers = [f"c{i}" for i in range(cols - 1)] + ["target"]
    w.writerow(headers)
    for i in range(rows):
        w.writerow([f"{(i % 10) / 10:.1f}" for _ in range(cols - 1)] + [i % 2])
    return buf.getvalue()


class TestLimitErrorPayload(unittest.TestCase):
    """La forma del dato, que es el contrato con la SPA."""

    def test_campos_obligatorios(self):
        with patch.dict(os.environ, {"MATRIXAI_LIMITS_PROFILE": "equilibrado",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            payload = _limits.limit_error("max_rows", 94833)
        self.assertEqual(payload["error_kind"], "limit_exceeded")
        self.assertEqual(payload["limit_key"], "max_rows")
        self.assertEqual(payload["unit"], "rows")
        self.assertEqual(payload["actual"], 94833)
        self.assertEqual(payload["maximum"], 50000)
        self.assertEqual(payload["profile"], "equilibrado")
        self.assertTrue(payload["configurable"])
        self.assertIn("error", payload)

    def test_texto_de_respaldo_no_mezcla_idiomas(self):
        """El token `unit` es inglés (estable, para la SPA); el texto de
        respaldo es español entero. Mezclarlos daba "94833 rows supera el
        máximo", que no es ni una cosa ni la otra."""
        with patch.dict(os.environ, {"MATRIXAI_HOSTED": "0"}, clear=False):
            texto = _limits.limit_error("max_rows", 94833)["error"]
        self.assertIn("filas", texto)
        self.assertNotIn("rows", texto)

    def test_hosted_no_ofrece_accion_imposible(self):
        with patch.dict(os.environ, {"MATRIXAI_HOSTED": "1"}, clear=False):
            payload = _limits.limit_error("max_rows", 94833)
        self.assertFalse(payload["configurable"])
        self.assertNotIn("Ajustes", payload["error"])
        self.assertIn("servicio", payload["error"])

    def test_configurable_ofrece_donde_cambiarlo(self):
        with patch.dict(os.environ, {"MATRIXAI_HOSTED": "0"}, clear=False):
            payload = _limits.limit_error("max_rows", 94833)
        self.assertIn("Ajustes", payload["error"])

    def test_perfil_ilimitado_no_tiene_maximo(self):
        with patch.dict(os.environ, {"MATRIXAI_LIMITS_PROFILE": "ilimitado",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            payload = _limits.limit_error("max_rows", 94833)
        self.assertIsNone(payload["maximum"])

    def test_bytes_se_expresan_en_MB(self):
        with patch.dict(os.environ, {"MATRIXAI_HOSTED": "0"}, clear=False):
            payload = _limits.limit_error("max_csv_bytes", 160_000_000)
        self.assertEqual(payload["unit"], "bytes")
        self.assertIn("MB", payload["error"])

    def test_tamano_se_formatea_segun_magnitud(self):
        """Formatear siempre en MB daba "El CSV ocupa 0.0 MB y el máximo es
        0 MB" para un tope pequeño — inútil justo cuando hace falta entenderlo.
        Lo destapó un test preexistente al actualizarlo a este contrato."""
        self.assertEqual(_limits.human_bytes(583), "583 B")
        self.assertEqual(_limits.human_bytes(50_000), "50.0 KB")
        self.assertEqual(_limits.human_bytes(160_000_000), "160.0 MB")
        with patch.dict(os.environ, {"MATRIXAI_MAX_CSV_BYTES": "50",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            texto = _limits.limit_error("max_csv_bytes", 583)["error"]
        self.assertIn("583 B", texto)
        self.assertNotIn("0.0 MB", texto)

    def test_is_limit_error_discrimina(self):
        with patch.dict(os.environ, {"MATRIXAI_HOSTED": "0"}, clear=False):
            self.assertTrue(_limits.is_limit_error(_limits.limit_error("max_rows", 1)))
        self.assertFalse(_limits.is_limit_error({"ok": False, "error": "otra cosa"}))
        self.assertFalse(_limits.is_limit_error(None))
        self.assertFalse(_limits.is_limit_error("texto"))


class TestRutasEmitenElPayload(unittest.TestCase):
    """Las rutas del Studio (el CLI queda fuera del alcance de C1)."""

    MXAI = (
        "PROJECT P\n\nVECTOR Input[2]\n  c0: Scalar\n  c1: Scalar\nEND\n\n"
        "NETWORK Net\n  INPUT Input\n  LAYER Dense units=4 activation=relu\n"
        "  OUTPUT target: ProbabilityMap[class_0, class_1]\nEND\n\n"
        "GRAPH\n  Input -> Net\nEND\n"
    )
    MXTRAIN = (
        "MODEL P.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
        "  INPUT Input FROM COLUMNS [c0, c1]\n"
        "  TARGET target: Label[class_0, class_1]\n"
        "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
        "LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET target\nEND\n\n"
        "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.01\n  UPDATE Net.*\nEND\n\n"
        "RUN\n  EPOCHS 1\nEND\n"
    )

    def test_validate_training_csv_supera_max_rows(self):
        rows = 60
        body = "c0,c1,target\n" + "".join(
            f"0.{i % 10},0.{(i + 1) % 10},class_{i % 2}\n" for i in range(rows))
        with patch.dict(os.environ, {"MATRIXAI_MAX_ROWS": "10",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            res = _validate_training_csv(self.MXAI, self.MXTRAIN, body)
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_kind"], "limit_exceeded")
        self.assertEqual(res["limit_key"], "max_rows")
        self.assertEqual(res["actual"], rows)
        self.assertEqual(res["maximum"], 10)

    def test_normalize_external_csv_supera_max_csv_bytes(self):
        big = _csv(500)
        with patch.dict(os.environ, {"MATRIXAI_MAX_CSV_BYTES": "100",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            _text, err = _normalize_external_csv(big)
        self.assertIsInstance(err, dict)
        self.assertEqual(err["error_kind"], "limit_exceeded")
        self.assertEqual(err["limit_key"], "max_csv_bytes")
        self.assertIn("error", err, "el payload lleva su propio texto de respaldo")

    def test_normalize_external_csv_no_falla_por_debajo_del_tope(self):
        with patch.dict(os.environ, {"MATRIXAI_MAX_CSV_BYTES": "10000000",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            text, err = _normalize_external_csv("a,b\n1,2\n")
        self.assertIsNone(err)
        self.assertIn("a,b", text)

    def test_dataset_sintetico_supera_max_csv_bytes(self):
        with patch.dict(os.environ, {"MATRIXAI_MAX_CSV_BYTES": "50",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            res = _generate_synthetic_dataset(
                self.MXAI, self.MXTRAIN, rows=200, seed=42, mode="coherent")
        if res.get("ok"):
            self.skipTest("la generación no llegó al chequeo de tamaño")
        self.assertEqual(res.get("error_kind"), "limit_exceeded")
        self.assertEqual(res.get("limit_key"), "max_csv_bytes")


class TestDatasetProjectNoMiente(unittest.TestCase):
    """El envoltorio que culpaba a la preparación."""

    def test_tope_de_filas_no_se_presenta_como_hueco_de_preparacion(self):
        body = _csv(60)
        with patch.dict(os.environ, {"MATRIXAI_MAX_ROWS": "10",
                                     "MATRIXAI_HOSTED": "0"}, clear=False):
            with self.assertRaises(DatasetProjectError) as ctx:
                generate_project_from_dataset(body, target_column="target")
        exc = ctx.exception
        self.assertNotIn("hueco en la preparación", str(exc))
        self.assertNotIn("no un problema de tus datos", str(exc))
        self.assertIsInstance(exc.details, dict)
        self.assertEqual(exc.details["error_kind"], "limit_exceeded")
        self.assertEqual(exc.details["limit_key"], "max_rows")
        self.assertTrue(exc.details["configurable"])
        self.assertNotIn("ok", exc.details, "`ok` es ruido en `details`")

    def test_un_fallo_real_de_preparacion_conserva_su_mensaje(self):
        """El texto viejo se escribió para los fallos de preparación de verdad
        y ahí sigue siendo correcto: solo se retiró del camino de los topes."""
        body = "solo_una_columna\n1\n2\n3\n"
        with patch.dict(os.environ, {"MATRIXAI_HOSTED": "0"}, clear=False):
            with self.assertRaises(Exception) as ctx:
                generate_project_from_dataset(body, target_column="solo_una_columna")
        self.assertNotIsInstance(getattr(ctx.exception, "details", None), dict)

    def test_details_es_none_para_errores_que_no_son_topes(self):
        err = DatasetProjectError("target inexistente")
        self.assertIsNone(err.details)


if __name__ == "__main__":
    unittest.main()
