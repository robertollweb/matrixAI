"""`binarize`: convertir una medida en una CLASE, con umbral explícito.

Roberto, probando la plantilla de lluvia (2026-08-13):

    «si subo el fichero por csv el resultado no tiene nada que ver con la
     plantilla; el modelo que yo quiero guardar como plantilla es el que
     se genera al subir por csv, tiene más parámetros, el resultado da
     clase 1 o clase 0 con un valor numérico»

Su CSV trae la clase ya hecha y sale un clasificador; la plantilla
descargaba milímetros y entrenaba una REGRESIÓN. La misma pregunta
—¿lloverá mañana?— daba dos modelos distintos según por dónde entraras.

El umbral es EXPLÍCITO a propósito: «llueve» no significa lo mismo con
0 mm que con 1 mm, y elegirlo por su cuenta sería decidir por quien monta
la plantilla.
"""
from __future__ import annotations

import unittest

from matrixai.training.dataset_pipeline import PipelineError, run_pipeline


class Binarize(unittest.TestCase):
    def _clases(self, valores, umbral=0.2):
        filas = [{"lluvia": v} for v in valores]
        r = run_pipeline(filas, [{"op": "binarize", "column": "lluvia", "threshold": umbral, "as": "llueve"}])
        return [f["llueve"] for f in r.rows]

    def test_por_encima_del_umbral_es_1(self) -> None:
        self.assertEqual(self._clases(["0", "0.1", "0.2", "0.21", "3.2"]), ["0", "0", "0", "1", "1"])

    def test_el_umbral_MANDA(self) -> None:
        """Con otro umbral el modelo es otro: no es un detalle."""
        self.assertEqual(self._clases(["0.5"], umbral=0.2), ["1"])
        self.assertEqual(self._clases(["0.5"], umbral=1.0), ["0"])

    def test_un_HUECO_no_es_un_cero(self) -> None:
        """Inventar un 0 diría «no llovió» de un día del que no se sabe
        nada. Se deja vacío para que `missing_values` decida, que es
        quien tiene esa política."""
        self.assertEqual(self._clases(["", "1.0"]), ["", "1"])

    def test_sin_umbral_NO_se_adivina(self) -> None:
        with self.assertRaises(PipelineError) as ctx:
            run_pipeline([{"a": "1"}], [{"op": "binarize", "column": "a", "as": "b"}])
        self.assertIn("threshold", str(ctx.exception))

    def test_un_valor_que_no_es_numero_se_DICE(self) -> None:
        with self.assertRaises(PipelineError) as ctx:
            run_pipeline([{"a": "mucha"}], [{"op": "binarize", "column": "a", "threshold": 1, "as": "b"}])
        self.assertIn("mucha", str(ctx.exception))

    def test_no_pisa_una_columna_que_ya_existe(self) -> None:
        with self.assertRaises(PipelineError):
            run_pipeline([{"a": "1", "b": "x"}], [{"op": "binarize", "column": "a", "threshold": 1, "as": "b"}])


class ElLinajeTemporalSeCONSERVA(unittest.TestCase):
    """Binarizar un objetivo YA DESPLAZADO sigue siendo del día
    siguiente. Si el linaje se perdiera, el verificador de fuga del
    objetivo no podría hacer su trabajo — y una fuga en una serie
    temporal es un modelo que parece perfecto y no vale nada."""

    def test_el_desfase_viaja_a_la_columna_nueva(self) -> None:
        filas = [{"t": str(i), "mm": str(i)} for i in range(5)]
        r = run_pipeline(filas, [
            {"op": "shift_target", "column": "mm", "horizon": 1, "as": "mm_manana"},
            {"op": "binarize", "column": "mm_manana", "threshold": 1.5, "as": "llueve_manana"},
        ])
        self.assertEqual(r.column_offsets["mm_manana"], 1)
        self.assertEqual(
            r.column_offsets["llueve_manana"], 1,
            "la clase hereda el desfase de la columna de la que sale",
        )


if __name__ == "__main__":
    unittest.main()
