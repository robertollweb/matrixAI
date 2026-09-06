# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C5 — comparaciones y equivalencia.

Criterio de terminado: «Caso de intervalos individuales solapados pero
diferencia emparejada concluyente; caso inconcluso; caso de equivalencia
dentro del margen; caso AUROC superior con accuracy inferior a mayoritaria.
Cada uno produce el veredicto previsto.»
"""
from __future__ import annotations

import random
import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.comparaciones import ComparacionEmparejada, comparar_candidatos
from matrixai.estudio.incertidumbre import Intervalo, intervalo
from matrixai.estudio.metricas import EntradaNoMedible, Muestra, calcular


def _pareja_binaria(seed, n=300, n_desacuerdo=30, ventaja_candidato=25):
    """Dos modelos sobre las MISMAS filas: acuerdan en la mayoría (correlación
    de errores real) y el candidato acierta más que el baseline en las filas
    de desacuerdo -- la construcción exacta para demostrar que la diferencia
    EMPAREJADA puede ser concluyente aunque los intervalos individuales
    (que ignoran la correlación) se solapen."""
    rng = random.Random(seed)
    y_true = ["si" if i < n // 2 else "no" for i in range(n)]
    rng.shuffle(y_true)
    indices = list(range(n))
    rng.shuffle(indices)
    n_acuerdo_incorrecto = 20
    acuerdo_correcto = indices[:n - n_desacuerdo - n_acuerdo_incorrecto]
    acuerdo_incorrecto = indices[len(acuerdo_correcto):len(acuerdo_correcto) + n_acuerdo_incorrecto]
    desacuerdo = indices[len(acuerdo_correcto) + n_acuerdo_incorrecto:]
    cand_correcto = set(desacuerdo[:ventaja_candidato])

    otra = lambda y: "si" if y == "no" else "no"
    pred_cand, pred_base = [None] * n, [None] * n
    for i in acuerdo_correcto:
        pred_cand[i] = y_true[i]
        pred_base[i] = y_true[i]
    for i in acuerdo_incorrecto:
        pred_cand[i] = otra(y_true[i])
        pred_base[i] = otra(y_true[i])
    for i in desacuerdo:
        if i in cand_correcto:
            pred_cand[i], pred_base[i] = y_true[i], otra(y_true[i])
        else:
            pred_cand[i], pred_base[i] = otra(y_true[i]), y_true[i]

    m_cand = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                     positive_label="si", predictions=tuple(pred_cand))
    m_base = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                     positive_label="si", predictions=tuple(pred_base))
    return m_cand, m_base


class CriterioLiteralTest(unittest.TestCase):
    def test_intervalos_solapados_pero_diferencia_emparejada_concluyente(self):
        """El caso literal: individuales solapados, emparejada concluyente."""
        m_cand, m_base = _pareja_binaria(seed=0)
        ic_cand = intervalo("accuracy", m_cand, diseno="iid",
                            estimando="fixed_model_on_population", semilla=1, remuestras=2000)
        ic_base = intervalo("accuracy", m_base, diseno="iid",
                            estimando="fixed_model_on_population", semilla=1, remuestras=2000)
        solapan = ic_cand.ci_low <= ic_base.ci_high and ic_base.ci_low <= ic_cand.ci_high
        self.assertTrue(solapan, "el fixture tiene que producir intervalos solapados")

        resultado = comparar_candidatos("accuracy", m_cand, m_base, diseno="iid",
                                        estimando="fixed_model_on_population",
                                        semilla=1, remuestras=2000)
        self.assertEqual(resultado.veredicto, "mejora")
        self.assertGreater(resultado.intervalo.ci_low, 0.0)

    def test_caso_inconcluso(self):
        rng = random.Random(5)
        n = 200
        y_true = [rng.gauss(0, 1) for _ in range(n)]
        pred_cand = [y + rng.gauss(0, 1.0) for y in y_true]
        pred_base = [y + rng.gauss(0, 1.0) for y in y_true]
        m_cand = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_cand))
        m_base = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_base))
        resultado = comparar_candidatos("rmse", m_cand, m_base, diseno="iid",
                                        estimando="fixed_model_on_population",
                                        semilla=2, remuestras=1000)
        self.assertEqual(resultado.veredicto, "inconcluso")
        self.assertLess(resultado.intervalo.ci_low, 0.0)
        self.assertGreater(resultado.intervalo.ci_high, 0.0)

    def test_caso_equivalencia_practica(self):
        rng = random.Random(6)
        n = 2000
        y_true = [rng.gauss(0, 1) for _ in range(n)]
        pred_cand = [y + rng.gauss(0, 0.50) for y in y_true]
        pred_base = [y + rng.gauss(0, 0.505) for y in y_true]
        m_cand = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_cand))
        m_base = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_base))
        resultado = comparar_candidatos("rmse", m_cand, m_base, diseno="iid",
                                        estimando="fixed_model_on_population", semilla=3,
                                        remuestras=1000, margen_equivalencia=0.05)
        self.assertEqual(resultado.veredicto, "equivalencia_practica")
        self.assertLessEqual(abs(resultado.intervalo.ci_low), 0.05)
        self.assertLessEqual(abs(resultado.intervalo.ci_high), 0.05)

    def test_auroc_superior_con_accuracy_inferior_a_mayoritaria(self):
        """El baseline se compara en la métrica pertinente, no solo accuracy:
        un candidato puede ganar en AUROC y perder en accuracy frente a un
        baseline que predice siempre la clase mayoritaria."""
        rng = random.Random(7)
        n = 500
        y_true = ["si" if rng.random() < 0.10 else "no" for _ in range(n)]
        scores_base = [0.0] * n
        pred_base = ["no"] * n
        scores_cand = [rng.uniform(0.5, 1.0) if y == "si" else rng.uniform(0.0, 0.5) for y in y_true]
        pred_cand = ["si" if s > 0.08 else "no" for s in scores_cand]

        m_cand_auroc = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                               positive_label="si", scores=tuple(scores_cand))
        m_base_auroc = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                               positive_label="si", scores=tuple(scores_base))
        m_cand_acc = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                             positive_label="si", predictions=tuple(pred_cand))
        m_base_acc = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                             positive_label="si", predictions=tuple(pred_base))

        self.assertLess(calcular("accuracy", m_cand_acc).value, calcular("accuracy", m_base_acc).value)
        self.assertGreater(calcular("auroc", m_cand_auroc).value, calcular("auroc", m_base_auroc).value)

        r_auroc = comparar_candidatos("auroc", m_cand_auroc, m_base_auroc, diseno="iid",
                                      estimando="fixed_model_on_population", semilla=4, remuestras=1000)
        r_acc = comparar_candidatos("accuracy", m_cand_acc, m_base_acc, diseno="iid",
                                    estimando="fixed_model_on_population", semilla=4, remuestras=1000)
        self.assertEqual(r_auroc.veredicto, "mejora")
        self.assertEqual(r_acc.veredicto, "inferioridad")


class IncomparableTest(unittest.TestCase):
    def test_filas_distintas_por_n(self):
        m1 = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        m2 = Muestra(task="regression", y_true=(1.0, 2.0, 3.0, 4.0), predictions=(1.1, 2.1, 2.9, 4.1))
        r = comparar_candidatos("rmse", m1, m2, diseno="iid",
                                estimando="fixed_model_on_population", semilla=1)
        self.assertEqual(r.veredicto, "incomparable")
        self.assertIsNone(r.intervalo)
        self.assertIsNone(r.diferencia_puntual)

    def test_filas_distintas_por_verdad_observada(self):
        m1 = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        m2 = Muestra(task="regression", y_true=(1.0, 2.0, 3.5), predictions=(1.1, 2.1, 2.9))
        r = comparar_candidatos("rmse", m1, m2, diseno="iid",
                                estimando="fixed_model_on_population", semilla=1)
        self.assertEqual(r.veredicto, "incomparable")

    def test_protocolos_distintos(self):
        m1 = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        r = comparar_candidatos("rmse", m1, m1, diseno="iid", estimando="fixed_model_on_population",
                                semilla=1, protocolo_candidato="A", protocolo_baseline="B")
        self.assertEqual(r.veredicto, "incomparable")

    def test_mismo_protocolo_no_es_incomparable(self):
        m1 = Muestra(task="regression", y_true=tuple(range(40)),
                    predictions=tuple(float(v) + 0.1 for v in range(40)))
        r = comparar_candidatos("rmse", m1, m1, diseno="iid", estimando="fixed_model_on_population",
                                semilla=1, protocolo_candidato="A", protocolo_baseline="A")
        self.assertNotEqual(r.veredicto, "incomparable")

    def test_metrica_sin_direccion_se_rechaza(self):
        m = Muestra(task="binary_classification", y_true=("si", "no", "si", "no"),
                   classes=("no", "si"), positive_label="si",
                   probabilities=((0.2, 0.8), (0.7, 0.3), (0.3, 0.7), (0.6, 0.4)))
        with self.assertRaises(EntradaNoMedible):
            comparar_candidatos("calibration_in_the_large", m, m, diseno="iid",
                                estimando="fixed_model_on_population", semilla=1)


class MetricaIndefinidaTest(unittest.TestCase):
    def test_metrica_indefinida_en_una_mitad_da_inconcluso_con_motivo(self):
        """Una sola clase en el candidato: AUROC sale indefinido ahí, y la
        comparación entera se declara inconclusa CON motivo -- nunca se
        fabrica una diferencia con un lado que no se pudo medir."""
        y_mono = tuple(["no"] * 10)
        m_cand = Muestra(task="binary_classification", y_true=y_mono, classes=("no", "si"),
                         positive_label="si", scores=tuple(range(10)))
        m_base = Muestra(task="binary_classification", y_true=y_mono, classes=("no", "si"),
                         positive_label="si", scores=tuple(range(10)))
        r = comparar_candidatos("auroc", m_cand, m_base, diseno="iid",
                                estimando="fixed_model_on_population", semilla=1)
        self.assertEqual(r.veredicto, "inconcluso")
        self.assertIsNone(r.diferencia_puntual)
        self.assertIsNotNone(r.intervalo.undefined_reason)


class DisenosTest(unittest.TestCase):
    def test_diseno_groups_usa_la_unidad_declarada(self):
        rng = random.Random(11)
        y_true, pred_cand, pred_base, units = [], [], [], []
        for u in range(40):
            for _ in range(5):
                y = rng.gauss(0, 1)
                y_true.append(y)
                pred_cand.append(y + rng.gauss(0, 0.3))
                pred_base.append(y + rng.gauss(0, 0.8))
                units.append(f"u{u}")
        m_cand = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_cand), units=tuple(units))
        m_base = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_base), units=tuple(units))
        r = comparar_candidatos("rmse", m_cand, m_base, diseno="groups",
                                estimando="fixed_model_on_population", semilla=5, remuestras=500)
        self.assertEqual(r.intervalo.resampling_unit, "unit")
        self.assertEqual(r.veredicto, "mejora")

    def test_diseno_temporal_usa_bloques(self):
        rng = random.Random(12)
        n = 200
        y_true = [rng.gauss(0, 1) for _ in range(n)]
        pred_cand = [y + rng.gauss(0, 0.3) for y in y_true]
        pred_base = [y + rng.gauss(0, 0.8) for y in y_true]
        m_cand = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_cand))
        m_base = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(pred_base))
        r = comparar_candidatos("rmse", m_cand, m_base, diseno="temporal",
                                estimando="fixed_model_on_population", semilla=6, remuestras=500)
        self.assertEqual(r.intervalo.resampling_unit, "block")
        self.assertIsNotNone(r.intervalo.block_length)
        self.assertEqual(r.veredicto, "mejora")


class ComparacionEmparejadaEsquemaTest(unittest.TestCase):
    def test_incomparable_sin_motivo_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            ComparacionEmparejada(metric_id="rmse", diseno="iid", estimando="fixed_model_on_population",
                                  veredicto="incomparable", margen_equivalencia=None,
                                  diferencia_puntual=None, intervalo=None)

    def test_incomparable_con_intervalo_se_rechaza(self):
        ic = Intervalo(metric_id="rmse", design="iid", estimand="fixed_model_on_population",
                       resampling_unit="row", seed=1, n_resamples=10, level=0.95,
                       ci_low=-0.1, ci_high=0.1)
        with self.assertRaises(EsquemaInvalido):
            ComparacionEmparejada(metric_id="rmse", diseno="iid", estimando="fixed_model_on_population",
                                  veredicto="incomparable", margen_equivalencia=None,
                                  diferencia_puntual=0.0, intervalo=ic,
                                  undefined_reason={"es": "x", "en": "x"})

    def test_no_incomparable_sin_intervalo_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            ComparacionEmparejada(metric_id="rmse", diseno="iid", estimando="fixed_model_on_population",
                                  veredicto="mejora", margen_equivalencia=None,
                                  diferencia_puntual=0.1, intervalo=None)

    def test_intervalo_disponible_sin_diferencia_puntual_se_rechaza(self):
        ic = Intervalo(metric_id="rmse", design="iid", estimand="fixed_model_on_population",
                       resampling_unit="row", seed=1, n_resamples=10, level=0.95,
                       ci_low=-0.1, ci_high=0.1)
        with self.assertRaises(EsquemaInvalido):
            ComparacionEmparejada(metric_id="rmse", diseno="iid", estimando="fixed_model_on_population",
                                  veredicto="mejora", margen_equivalencia=None,
                                  diferencia_puntual=None, intervalo=ic)


if __name__ == "__main__":
    unittest.main()
