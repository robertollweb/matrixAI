"""LA MÉTRICA QUE SE IMPRIME ES LA DE LA TAREA (2026-08-25).

Medido montando la galería: al entrenar el Kelvin —una regresión que converge a
error 6,5e-17— el CLI imprimía **`Accuracy: 0.000000`**, que es lo PRIMERO que
lee quien acaba de entrenar el ejemplo canónico del repositorio. Se lee como «el
modelo no acierta nunca», y lo que pasaba es que la exactitud no aplica.

**Un valor que no aplica no es un cero.** Es la misma regla que ya obligó a que
un ausente no se escriba como dato, aquí en la primera línea que ve alguien.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.cli import _metricas_para_leer


class _Resultado:
    def __init__(self, output_dir: str, accuracy: float = 0.0):
        self.output_dir = output_dir
        self.accuracy = accuracy


def _run(tmp: Path, metricas: dict) -> _Resultado:
    (tmp / "metrics.json").write_text(json.dumps(metricas), encoding="utf-8")
    return _Resultado(str(tmp))


class LaMetricaQueAplicaTest(unittest.TestCase):
    def test_una_REGRESION_enseña_R2_y_MAE_y_NO_accuracy(self):
        with TemporaryDirectory() as d:
            lineas = _metricas_para_leer(
                _run(Path(d), {"r2": 1.0, "mae": 6.5e-17, "accuracy": 0.0}), None)
        self.assertEqual(lineas, ["R2: 1.000000", "MAE: 6.5e-17"])
        self.assertFalse(any("Accuracy" in x for x in lineas))

    def test_una_CLASIFICACION_sigue_enseñando_la_exactitud(self):
        with TemporaryDirectory() as d:
            lineas = _metricas_para_leer(
                _run(Path(d), {"accuracy": 0.96875, "macro_f1": 0.9}, ), None)
        self.assertEqual(lineas, ["Accuracy: 0.968750"])

    def test_sin_metrics_json_no_se_inventa_nada_y_cae_a_lo_de_siempre(self):
        with TemporaryDirectory() as d:
            lineas = _metricas_para_leer(_Resultado(d, accuracy=0.5), None)
        self.assertEqual(lineas, ["Accuracy: 0.500000"])

    def test_con_solo_MAE_tambien_es_regresion(self):
        with TemporaryDirectory() as d:
            lineas = _metricas_para_leer(_run(Path(d), {"mae": 0.25}), None)
        self.assertEqual(lineas, ["MAE: 0.25"])

    def test_un_metrics_json_ilegible_no_tumba_el_entrenamiento(self):
        with TemporaryDirectory() as d:
            (Path(d) / "metrics.json").write_text("{no es json", encoding="utf-8")
            lineas = _metricas_para_leer(_Resultado(d, accuracy=0.75), None)
        self.assertEqual(lineas, ["Accuracy: 0.750000"])


if __name__ == "__main__":
    unittest.main()
