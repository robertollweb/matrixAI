# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Una muestra no puede DECLARAR una escala y ORDENAR por otra.

`Muestra.escala_de_decision` dice `calibrated_probability` en cuanto hay
`probabilities`. Hasta el 2026-09-17, `Muestra.puntuacion_del_positivo` —la que
ordenan el AUROC y la matriz de confusión— daba prioridad a `scores`. Con las
dos presentes, la muestra se declaraba en probabilidad y cortaba sobre logits:
`matriz_de_confusion` sin umbral aplicaba el 0,5 de PROBABILIDAD a un logit, y
la búsqueda de umbral de 104-C4 (que busca sobre la probabilidad) veía después
aplicar su umbral sobre otra escala. Sin error ni aviso.

Medido el 2026-09-16 y reproducido aquí con el mismo patrón: solo discrepan
las probabilidades entre 0,5 y 0,622, el único tramo donde `p > 0,5` y
`logit(p) > 0,5` no dicen lo mismo. Fuera de ese tramo el defecto no se ve,
y así se escapó la primera reproducción.

Los motores de la casa no lo pisaban: su `scores` es la columna positiva de
`predict_proba`, el mismo número (lo fija la última prueba). Muerde el día en
que un motor exponga su margen crudo.
"""
from __future__ import annotations

import math
import unittest

from matrixai.estudio.metricas import Muestra, calcular, matriz_de_confusion
from matrixai.estudio.umbral import elegir_umbral

_Y = ("si", "si", "si", "no", "no", "no")
#: Dos positivos en el tramo (0,5, 0,622), donde probabilidad y logit discrepan
#: frente a un corte de 0,5.
_P = (0.55, 0.60, 0.90, 0.20, 0.30, 0.10)
_LOGITS = tuple(math.log(p / (1 - p)) for p in _P)


def _muestra(**salidas) -> Muestra:
    return Muestra.binaria(_Y, classes=("no", "si"), positive_label="si", **salidas)


class TestConProbabilidadYLogitMandaLaProbabilidad(unittest.TestCase):
    def test_el_caso_medido_cae_donde_discrepan(self):
        """El control del instrumento: si ningún positivo cayera en el tramo,
        las dos lecturas darían la misma matriz y esta prueba no mediría nada."""
        self.assertTrue(any(0.5 < p < 1 / (1 + math.exp(-0.5)) for p in _P))

    def test_la_matriz_sin_umbral_corta_sobre_la_PROBABILIDAD(self):
        solo = matriz_de_confusion(_muestra(probabilidades=_P))
        ambas = matriz_de_confusion(_muestra(probabilidades=_P, puntuaciones=_LOGITS))
        self.assertEqual(solo.counts["si"]["si"], 3)
        self.assertEqual(ambas.counts, solo.counts)
        self.assertEqual(ambas.threshold, 0.5)

    def test_la_escala_declarada_y_la_que_ordena_son_la_misma(self):
        m = _muestra(probabilidades=_P, puntuaciones=_LOGITS)
        self.assertEqual(m.escala_de_decision, "calibrated_probability")
        self.assertEqual(m.puntuacion_del_positivo, m.probabilidad_del_positivo)

    def test_el_umbral_elegido_se_aplica_en_la_escala_en_que_se_busco(self):
        """104-C4 busca sobre la probabilidad; aplicar su umbral a la muestra
        con logits tiene que dar lo mismo que aplicarlo a la de solo
        probabilidades."""
        solo = _muestra(probabilidades=_P)
        ambas = _muestra(probabilidades=_P, puntuaciones=_LOGITS)
        politica = elegir_umbral(solo, cost_false_positive=1.0, cost_false_negative=1.0)
        self.assertEqual(elegir_umbral(ambas, cost_false_positive=1.0,
                                       cost_false_negative=1.0).threshold, politica.threshold)
        self.assertEqual(matriz_de_confusion(ambas, umbral=politica.threshold).counts,
                         matriz_de_confusion(solo, umbral=politica.threshold).counts)


class TestSoloPuntuacionesCrudasSiguenMandando(unittest.TestCase):
    """La otra mitad: sin probabilidades, las puntuaciones SON lo que hay, y su
    escala es `raw_score`."""

    def test_ordena_por_las_puntuaciones_y_lo_declara(self):
        m = _muestra(puntuaciones=_LOGITS)
        self.assertEqual(m.escala_de_decision, "raw_score")
        self.assertEqual(m.puntuacion_del_positivo, _LOGITS)
        self.assertEqual(matriz_de_confusion(m, umbral=0.5).counts["si"]["si"], 1)


class TestLosMotoresDeLaCasaNoCambianDeNumero(unittest.TestCase):
    def test_con_scores_IGUAL_a_la_probabilidad_el_AUROC_no_se_mueve(self):
        """Así construye el harness de la Fase 0 sus muestras binarias: las dos
        capacidades, y `scores` es la columna positiva de `predict_proba`. El
        veredicto de 101-C5 se midió así, y no puede cambiar por este arreglo."""
        como_el_harness = _muestra(probabilidades=_P, puntuaciones=_P)
        solo = _muestra(probabilidades=_P)
        self.assertEqual(calcular("auroc", como_el_harness).value, calcular("auroc", solo).value)


if __name__ == "__main__":
    unittest.main()
