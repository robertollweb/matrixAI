# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C4 — calibrador y umbral.

Criterio de terminado: «Umbral coherente con pipeline calibrado; caso de
coste sintético devuelve el óptimo conocido. Cambiar verdad de test no
cambia el umbral. Multiclase/regresión muestran no aplicable o capacidad
futura, no umbral binario inventado.»
"""
from __future__ import annotations

import random
import unittest

from matrixai.estudio.calibracion import _sigmoid, ajustar_recalibracion_logistica
from matrixai.estudio.esquemas import Restriccion
from matrixai.estudio.metricas import EntradaNoMedible, Muestra, calcular
from matrixai.estudio.umbral import elegir_umbral


def _muestra_oraculo(seed, n=8000):
    """Un modelo perfectamente calibrado: la probabilidad que reporta ES la
    probabilidad real de `y=1` — así el umbral óptimo teórico
    (`cost_fp/(cost_fp+cost_fn)`) es un número conocido de antemano, no
    supuesto."""
    rng = random.Random(seed)
    p_true = [rng.random() for _ in range(n)]
    y_true = ["si" if rng.random() < p else "no" for p in p_true]
    return Muestra.binaria(y_true=y_true, classes=("no", "si"), positive_label="si",
                           probabilidades=p_true)


class CriterioLiteralTest(unittest.TestCase):
    def test_caso_de_coste_sintetico_devuelve_el_optimo_conocido(self):
        m = _muestra_oraculo(seed=0)
        cost_fp, cost_fn = 1.0, 3.0
        esperado = cost_fp / (cost_fp + cost_fn)
        politica = elegir_umbral(m, cost_false_positive=cost_fp, cost_false_negative=cost_fn)
        self.assertAlmostEqual(politica.threshold, esperado, delta=0.05)

    def test_otro_coste_sintetico_desplaza_el_optimo_en_la_direccion_correcta(self):
        """Penalizar más el falso positivo que el falso negativo baja el
        umbral óptimo (más barato predecir positivo) -- la dirección
        importa, no solo un número aislado."""
        m = _muestra_oraculo(seed=1)
        barato_fn = elegir_umbral(m, cost_false_positive=3.0, cost_false_negative=1.0)
        barato_fp = elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=3.0)
        self.assertLess(barato_fp.threshold, barato_fn.threshold)

    def test_cambiar_verdad_de_test_no_cambia_el_umbral(self):
        """La función no recibe ningún dato de test -- estructuralmente no
        puede cambiar por él. Se demuestra llamando dos veces con la MISMA
        muestra de calibración mientras una muestra de "test" cambia por
        completo entre medias."""
        m_calibracion = _muestra_oraculo(seed=2, n=2000)
        politica_1 = elegir_umbral(m_calibracion, cost_false_positive=1.0, cost_false_negative=2.0)

        _test_que_cambia = _muestra_oraculo(seed=999, n=2000)  # no participa para nada
        politica_2 = elegir_umbral(m_calibracion, cost_false_positive=1.0, cost_false_negative=2.0)
        self.assertEqual(politica_1.threshold, politica_2.threshold)

    def test_multiclase_no_inventa_un_umbral_binario(self):
        m = Muestra(task="multiclass_classification", y_true=("a", "b", "c"), classes=("a", "b", "c"),
                   probabilities=((0.5, 0.3, 0.2), (0.2, 0.5, 0.3), (0.1, 0.2, 0.7)))
        with self.assertRaises(EntradaNoMedible):
            elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0)

    def test_regresion_no_inventa_un_umbral_binario(self):
        m = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        with self.assertRaises(EntradaNoMedible):
            elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0)


class EscalaYCalibradorTest(unittest.TestCase):
    def test_escala_es_probabilidad_calibrada(self):
        """El criterio de terminado exige "umbral coherente con pipeline
        calibrado": este corte solo busca sobre escala de probabilidad
        (`Muestra.escala_de_decision`, 104-C0) -- nunca sobre un score
        crudo sin interpretación de probabilidad."""
        m = _muestra_oraculo(seed=3, n=500)
        politica = elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0)
        self.assertEqual(politica.scale, "calibrated_probability")

    def test_sin_probabilidades_se_rechaza(self):
        m = Muestra(task="binary_classification", y_true=("si", "no"), classes=("no", "si"),
                   positive_label="si", scores=(0.8, 0.2))
        with self.assertRaises(EntradaNoMedible):
            elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0)

    def test_recalibracion_desplaza_el_umbral_buscado(self):
        """Aplicar el calibrador ANTES de buscar tiene que cambiar el
        resultado frente a buscar sobre la probabilidad cruda -- si no,
        `recalibracion` estaría aceptado pero ignorado en silencio."""
        rng = random.Random(4)
        zs = [rng.gauss(0, 1.5) for _ in range(3000)]
        ps_crudas = [_sigmoid(z) for z in zs]
        ys = ["si" if rng.random() < _sigmoid(0.8 + 1.6 * z) else "no" for z in zs]
        m_desarrollo = Muestra(task="binary_classification", y_true=tuple(ys), classes=("no", "si"),
                               positive_label="si", probabilities=tuple((1 - p, p) for p in ps_crudas))
        recal = ajustar_recalibracion_logistica(m_desarrollo)
        self.assertTrue(recal.convergio)

        rng2 = random.Random(5)
        zs2 = [rng2.gauss(0, 1.5) for _ in range(2000)]
        ps2 = [_sigmoid(z) for z in zs2]
        ys2 = ["si" if rng2.random() < _sigmoid(0.8 + 1.6 * z) else "no" for z in zs2]
        m_umbral = Muestra(task="binary_classification", y_true=tuple(ys2), classes=("no", "si"),
                           positive_label="si", probabilities=tuple((1 - p, p) for p in ps2))

        sin_recalibrar = elegir_umbral(m_umbral, cost_false_positive=1.0, cost_false_negative=1.0)
        con_recalibracion = elegir_umbral(m_umbral, cost_false_positive=1.0, cost_false_negative=1.0,
                                          recalibracion=recal)
        self.assertNotAlmostEqual(sin_recalibrar.threshold, con_recalibracion.threshold, places=3)


class RestriccionesTest(unittest.TestCase):
    def test_restriccion_obligatoria_descarta_umbrales_que_no_cumplen(self):
        m = _muestra_oraculo(seed=6)
        politica = elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0,
                                 restricciones=[Restriccion(clave="sensitivity", operador="min", valor=0.9)])
        sensibilidad = calcular("sensitivity", m, umbral=politica.threshold)
        self.assertGreaterEqual(sensibilidad.value, 0.9)

    def test_restriccion_no_obligatoria_no_descarta(self):
        m = _muestra_oraculo(seed=7)
        restriccion_imposible = Restriccion(clave="sensitivity", operador="min", valor=1.01,
                                            obligatoria=False)
        politica = elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0,
                                 restricciones=[restriccion_imposible])
        self.assertIsNotNone(politica.threshold)

    def test_ningun_umbral_cumple_todas_las_restricciones_se_rechaza(self):
        m = _muestra_oraculo(seed=8)
        restricciones = [Restriccion(clave="sensitivity", operador="min", valor=0.99),
                        Restriccion(clave="specificity", operador="min", valor=0.99)]
        with self.assertRaises(EntradaNoMedible):
            elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0, restricciones=restricciones)

    def test_restriccion_no_derivable_del_umbral_se_rechaza(self):
        """AUROC no depende de dónde se corte -- pedirla como restricción de
        umbral es un error de cableado, no un umbral "aproximado"."""
        m = _muestra_oraculo(seed=9)
        with self.assertRaises(EntradaNoMedible):
            elegir_umbral(m, cost_false_positive=1.0, cost_false_negative=1.0,
                         restricciones=[Restriccion(clave="auroc", operador="min", valor=0.8)])


if __name__ == "__main__":
    unittest.main()
