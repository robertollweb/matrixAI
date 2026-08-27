"""UNA EXACTITUD QUE NO HA COMPARADO NADA NO SE PUBLICA (2026-08-25).

Medido montando el caso de LLUVIA de la galería, con datos REALES: el
entrenamiento decía

    Best epoch: 1 · Best validation loss: 0.000000 · Accuracy: 1.000000

sobre 2.189 días de observaciones meteorológicas. Un 100 % prediciendo si
llueve mañana es imposible, y lo que pasaba es esto:

* el contrato declaraba `Label[0, 1]` y el modelo `ProbabilityMap[0, 1]`, que
  **se leen como un RANGO** (`RangeSpec(0, 1)`) y no como dos clases;
* sin clases, el objetivo se carga como un vector de UN elemento;
* y `_argmax` de un vector de un elemento es **siempre 0**, así que todas las
  filas «aciertan».

El resultado era un **1,0 vacuo**: la matriz de confusión salía vacía y
`macro_f1: 0.0` al lado, pero lo que se imprime y lo que viaja al paquete es la
exactitud. Un número que parece un éxito perfecto y no ha comparado nada es peor
que no dar número.

Con las clases nombradas (`Label[no, si]`), el mismo modelo y los mismos datos
dan **0,7626** — que es un resultado creíble para predecir la lluvia del día
siguiente con seis variables.
"""
from __future__ import annotations

import unittest

from matrixai.training.dense_evaluator import _binary_metrics, _multiclass_metrics


class UnVectorDeUnElementoNoTieneClasesTest(unittest.TestCase):
    def test_el_camino_multiclase_se_NIEGA(self):
        with self.assertRaises(ValueError) as e:
            _multiclass_metrics([[0.9], [0.2]], [[1.0], [0.0]], [])
        self.assertIn("no hay clases que comparar", str(e.exception))
        # Y dice la causa probable, que es lo que permite arreglarlo.
        self.assertIn("Label[0, 1]", str(e.exception))

    def test_con_DOS_clases_de_verdad_mide_como_siempre(self):
        m = _multiclass_metrics(
            [[0.9, 0.1], [0.2, 0.8]], [[1.0, 0.0], [0.0, 1.0]], ["no", "si"])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["confusion_matrix"]["no"]["no"], 1)

    def test_y_una_exactitud_MALA_sigue_saliendo_mala(self):
        """El arreglo no puede consistir en que todo salga bien."""
        m = _multiclass_metrics(
            [[0.9, 0.1], [0.9, 0.1]], [[1.0, 0.0], [0.0, 1.0]], ["no", "si"])
        self.assertEqual(m["accuracy"], 0.5)


class FormasQueNoCasanTest(unittest.TestCase):
    def test_dos_salidas_contra_un_objetivo_escalar_se_niega(self):
        with self.assertRaises(ValueError) as e:
            _multiclass_metrics([[0.6, 0.4]], [[1.0]], ["no", "si"])
        self.assertIn("devuelve 2 valores y el objetivo trae 1", str(e.exception))

    def test_lo_mismo_en_el_camino_binario(self):
        with self.assertRaises(ValueError) as e:
            _binary_metrics([[0.6, 0.4]], [[1.0]], ["no", "si"])
        self.assertIn("clasificación binaria", str(e.exception))

    def test_el_binario_de_UNA_salida_contra_UN_objetivo_sigue_valiendo(self):
        """Ése es su caso legítimo: un escalar con umbral."""
        m = _binary_metrics([[0.9], [0.1]], [[1.0], [0.0]], ["no", "si"])
        self.assertEqual(m["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
