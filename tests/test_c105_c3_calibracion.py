# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C3 — calibración binaria: recalibrado logístico y curva de
fiabilidad.

Criterio de terminado: «fixtures de población sintética conocida y
tolerancias de muestreo justificadas; referencia numérica para a/b;
separación y probabilidades extremas no devuelven conclusiones falsas.
Datos usados para evaluar calibración no ajustan ese calibrador. No afirmar
calibración perfecta por una muestra finita.»
"""
from __future__ import annotations

import random
import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.calibracion import (
    _logit,
    _sigmoid,
    ajustar_recalibracion_logistica,
    aplicar_recalibracion,
    curva_de_fiabilidad,
)
from matrixai.estudio.metricas import Muestra


def _muestra_generada(a_real: float, b_real: float, n: int, *, seed: int = 0,
                      sigma_z: float = 1.5) -> tuple[Muestra, list[float]]:
    """Población SINTÉTICA con `a`/`b` REALES conocidos: `z ~ N(0, sigma_z)`,
    `p_crudo = sigmoid(z)` (lo que el motor "predice"), `y ~
    Bernoulli(sigmoid(a_real + b_real*z))` (la verdad, generada con la recta
    real que la recalibración tiene que recuperar)."""
    rng = random.Random(seed)
    zs = [rng.gauss(0, sigma_z) for _ in range(n)]
    ps = [_sigmoid(z) for z in zs]
    ys = ["si" if rng.random() < _sigmoid(a_real + b_real * z) else "no" for z in zs]
    m = Muestra(task="binary_classification", y_true=tuple(ys), classes=("no", "si"),
               positive_label="si", probabilities=tuple((1 - p, p) for p in ps))
    return m, ps


class SigmoidLogitTest(unittest.TestCase):
    def test_son_inversas(self):
        for p in (0.001, 0.1, 0.5, 0.9, 0.999):
            self.assertAlmostEqual(_sigmoid(_logit(p)), p, places=6)

    def test_logit_recorta_extremos_sin_dar_infinito(self):
        """«Probabilidades extremas no devuelven conclusiones falsas»: 0.0 y
        1.0 exactos no producen ±inf ni una excepción."""
        import math
        self.assertTrue(math.isfinite(_logit(0.0)))
        self.assertTrue(math.isfinite(_logit(1.0)))


class RecalibracionLogisticaTest(unittest.TestCase):
    def test_referencia_numerica_para_a_y_b(self):
        """El criterio de terminado literal: una población con a/b REALES
        conocidos, y el ajuste tiene que acercarse — con una tolerancia de
        muestreo justificada por el tamaño (n=6000, error estándar del
        orden de 0.05 para estos parámetros, medido por repetición)."""
        m, _ = _muestra_generada(a_real=0.3, b_real=1.2, n=6000, seed=0)
        r = ajustar_recalibracion_logistica(m)
        self.assertTrue(r.convergio)
        self.assertFalse(r.separacion_detectada)
        self.assertAlmostEqual(r.a, 0.3, delta=0.15)
        self.assertAlmostEqual(r.b, 1.2, delta=0.15)

    def test_bien_calibrado_da_a_cerca_de_0_y_b_cerca_de_1(self):
        m, _ = _muestra_generada(a_real=0.0, b_real=1.0, n=6000, seed=1)
        r = ajustar_recalibracion_logistica(m)
        self.assertAlmostEqual(r.a, 0.0, delta=0.1)
        self.assertAlmostEqual(r.b, 1.0, delta=0.1)

    def test_intercepto_en_grande_fija_b_a_uno(self):
        m, _ = _muestra_generada(a_real=0.5, b_real=1.0, n=4000, seed=2)
        r = ajustar_recalibracion_logistica(m, metodo="intercepto_en_grande")
        self.assertEqual(r.b, 1.0)
        self.assertAlmostEqual(r.a, 0.5, delta=0.15)

    def test_separacion_perfecta_no_da_un_numero_falso(self):
        """El criterio literal: «separación... no devuelve conclusiones
        falsas». y=1 exactamente cuando z>0 no tiene máximo finito de
        verosimilitud — declarado indefinido, no un b enorme."""
        rng = random.Random(3)
        zs = [rng.uniform(-5, 5) for _ in range(200)]
        ps = [_sigmoid(z) for z in zs]
        ys = ["si" if z > 0 else "no" for z in zs]
        m = Muestra(task="binary_classification", y_true=tuple(ys), classes=("no", "si"),
                   positive_label="si", probabilities=tuple((1 - p, p) for p in ps))
        r = ajustar_recalibracion_logistica(m)
        self.assertTrue(r.separacion_detectada)
        self.assertIsNone(r.a)
        self.assertIsNone(r.b)
        self.assertIsNotNone(r.undefined_reason)

    def test_una_sola_clase_se_declara_indefinida(self):
        """Una sola clase es un motivo DISTINTO de separación (no hay nada
        que separar, ni falta que hacer ni una sola iteración de Newton para
        saberlo) — comprobar el texto, no solo que exista alguno, porque el
        propio Newton también detectaría esto como separación (b diverge
        igual) y esa confusión sería un motivo menos preciso, no uno falso."""
        m = Muestra(task="binary_classification", y_true=("no", "no", "no", "no"),
                   classes=("no", "si"), positive_label="si",
                   probabilities=((0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)))
        r = ajustar_recalibracion_logistica(m)
        self.assertIsNone(r.a)
        self.assertIsNotNone(r.undefined_reason)
        self.assertIn("misma clase", r.undefined_reason["es"])
        self.assertFalse(r.separacion_detectada)

    def test_menos_de_dos_observaciones_se_declara_indefinida(self):
        """Con n=1 esto también sería «una sola clase» a ojos de ese otro
        chequeo (un solo valor no puede tener dos clases) — comprobar el
        texto para que este motivo específico (n<2) no quede sin cubrir."""
        m = Muestra(task="binary_classification", y_true=("si",), classes=("no", "si"),
                   positive_label="si", probabilities=((0.5, 0.5),))
        r = ajustar_recalibracion_logistica(m)
        self.assertIsNone(r.a)
        self.assertIsNotNone(r.undefined_reason)
        self.assertIn("filas suficientes", r.undefined_reason["es"])

    def test_no_se_puede_construir_con_valor_y_motivo_a_la_vez(self):
        from matrixai.estudio.calibracion import RecalibracionLogistica
        with self.assertRaises(EsquemaInvalido):
            RecalibracionLogistica(metodo="logistico_completo", n_observaciones=10,
                                   a=0.1, b=1.0, convergio=True,
                                   undefined_reason={"es": "x", "en": "x"})

    def test_no_se_puede_construir_sin_valor_ni_motivo(self):
        from matrixai.estudio.calibracion import RecalibracionLogistica
        with self.assertRaises(EsquemaInvalido):
            RecalibracionLogistica(metodo="logistico_completo", n_observaciones=10)


class AplicarRecalibracionTest(unittest.TestCase):
    def test_datos_de_evaluacion_no_ajustan_el_calibrador(self):
        """El criterio literal: «datos usados para evaluar calibración no
        ajustan ese calibrador». Se ajusta sobre una muestra de desarrollo,
        y se aplica sobre probabilidades NUEVAS — `aplicar_recalibracion`
        no tiene ninguna vía de volver a mirar `y_true`."""
        desarrollo, _ = _muestra_generada(a_real=0.3, b_real=1.2, n=4000, seed=4)
        r = ajustar_recalibracion_logistica(desarrollo)
        probabilidades_nuevas = [0.05, 0.25, 0.5, 0.75, 0.95]
        recalibradas = aplicar_recalibracion(r, probabilidades_nuevas)
        self.assertEqual(len(recalibradas), 5)
        self.assertTrue(all(0.0 < p < 1.0 for p in recalibradas))

    def test_aplicar_sin_ajuste_valido_se_rechaza(self):
        from matrixai.estudio.calibracion import RecalibracionLogistica
        indefinida = RecalibracionLogistica(metodo="logistico_completo", n_observaciones=1,
                                            undefined_reason={"es": "x", "en": "x"})
        with self.assertRaises(EsquemaInvalido):
            aplicar_recalibracion(indefinida, [0.5])

    def test_a_0_b_1_es_la_identidad(self):
        from matrixai.estudio.calibracion import RecalibracionLogistica
        identidad = RecalibracionLogistica(metodo="logistico_completo", n_observaciones=100,
                                           a=0.0, b=1.0, convergio=True, iteraciones=1)
        for p in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(aplicar_recalibracion(identidad, [p])[0], p, places=9)


class CurvaDeFiabilidadTest(unittest.TestCase):
    def test_todas_las_filas_caen_en_algun_bin(self):
        m, _ = _muestra_generada(a_real=0.0, b_real=1.0, n=537, seed=5)
        curva = curva_de_fiabilidad(m, n_bins=10)
        self.assertEqual(sum(b.n for b in curva.bins), 537)

    def test_ece_depende_del_metodo_de_bins_y_se_declara(self):
        """Texto literal: «ECE depende de bins... explicarlo» —
        `metodo_de_bins` viaja siempre en el resultado."""
        m, _ = _muestra_generada(a_real=0.5, b_real=0.7, n=800, seed=6)
        ancho = curva_de_fiabilidad(m, n_bins=8, metodo_de_bins="ancho_igual")
        frecuencia = curva_de_fiabilidad(m, n_bins=8, metodo_de_bins="frecuencia_igual")
        self.assertEqual(ancho.metodo_de_bins, "ancho_igual")
        self.assertEqual(frecuencia.metodo_de_bins, "frecuencia_igual")
        # frecuencia igual reparte EL MISMO numero de filas por bin (o casi)
        conteos_frecuencia = [b.n for b in frecuencia.bins]
        self.assertLessEqual(max(conteos_frecuencia) - min(conteos_frecuencia), 1)

    def test_bien_calibrado_tiene_ece_bajo(self):
        """No «calibración perfecta» (el criterio prohíbe afirmar eso de una
        muestra finita): solo que el ECE de un modelo bien calibrado es
        bajo, con su propio número, no una etiqueta binaria."""
        m, _ = _muestra_generada(a_real=0.0, b_real=1.0, n=6000, seed=7)
        curva = curva_de_fiabilidad(m, n_bins=10)
        self.assertLess(curva.ece, 0.03)

    def test_mal_calibrado_tiene_ece_mas_alto(self):
        bien, _ = _muestra_generada(a_real=0.0, b_real=1.0, n=4000, seed=8)
        mal, _ = _muestra_generada(a_real=1.5, b_real=0.4, n=4000, seed=8)
        c_bien = curva_de_fiabilidad(bien, n_bins=10)
        c_mal = curva_de_fiabilidad(mal, n_bins=10)
        self.assertGreater(c_mal.ece, c_bien.ece)

    def test_muestra_sin_probabilidades_se_declara_indefinida(self):
        m = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        curva = curva_de_fiabilidad(m)
        self.assertEqual(curva.bins, ())
        self.assertIsNone(curva.ece)
        self.assertIsNotNone(curva.undefined_reason)


if __name__ == "__main__":
    unittest.main()
