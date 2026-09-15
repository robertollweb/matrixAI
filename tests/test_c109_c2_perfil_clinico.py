# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""109-C2 — el perfil de la tarea, clínico: umbrales con IC, PPV/NPV **a
prevalencia**, calibración, curva de decisión (DCA) con sus dos referencias y
subgrupos.

Criterio de terminado, literal: «tabla de umbrales con IC, calibración, DCA con
sus referencias y subgrupos; prueba numérica a mano de DCA (beneficio neto =
TP/n − FP/n · pt/(1−pt)); invertir la probabilidad intercambia sens/esp y la
prueba lo caza».

LOS NÚMEROS DE LA DCA ESTÁN ESCRITOS AQUÍ, A MANO, sobre una cohorte inventada
de diez filas. No se comparan contra otra ejecución del mismo código —eso solo
demostraría que el código es consistente consigo mismo—: se comparan contra la
aritmética hecha con lápiz, que está en el docstring de cada prueba.

Y LA INVERSIÓN, que es el control conocido-bueno de todo lo demás: hay DOS
pruebas porque, MEDIDO, son dos hechos distintos. Cambiar `p` por `1−p`
dejando el mismo umbral y la misma clase positiva **COMPLEMENTA** sens y esp
(`1−sens`, `1−esp`), no las intercambia. El intercambio exacto aparece cuando
la inversión se escribe entera —`1−p` es la probabilidad de la OTRA clase, y el
umbral se refleja a `1−t`—: entonces sí, sens ↔ esp. Las dos están medidas
abajo con sus números.
"""
from __future__ import annotations

import random
import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.calibracion import curva_de_fiabilidad
from matrixai.estudio.incertidumbre import intervalo
from matrixai.estudio.metricas import EntradaNoMedible, Muestra, calcular, matriz_de_confusion
from matrixai.estudio.perfil_clinico import (
    _T,
    ALCANCES_DE_VALIDACION,
    BENEFICIO_NETO_NO_TRATAR,
    NPV_A_PREVALENCIA,
    PPV_A_PREVALENCIA,
    CurvaDeDecision,
    PerfilClinico,
    alcance_de_validacion,
    beneficio_neto,
    curva_de_decision,
    ficha_del_perfil,
    tabla_de_umbrales,
)
from matrixai.estudio.segmentos import analizar_segmento
from matrixai.estudio.textos import IDIOMAS, huecos_de

# ---------------------------------------------------------------------------
# La cohorte de diez filas sobre la que se hacen las cuentas a mano.
#
#   fila   1     2     3     4  |  5     6     7     8     9    10
#   y     si    si    si    si  | no    no    no    no    no    no
#   p    0.90  0.75  0.60  0.35 | 0.55  0.45  0.30  0.20  0.10  0.05
#
# n = 10, positivos = 4, negativos = 6, prevalencia observada = 0,4.
# Ningún `p` vale exactamente 0,5 ni 0,2... salvo la fila 8, que vale 0,20
# justo para que el umbral 0,20 caiga sobre ella y el `>=` de la matriz de
# confusión (105-C1) quede probado donde importa.
# ---------------------------------------------------------------------------
_Y = ("si", "si", "si", "si", "no", "no", "no", "no", "no", "no")
_P = (0.90, 0.75, 0.60, 0.35, 0.55, 0.45, 0.30, 0.20, 0.10, 0.05)


def _a_mano(y=_Y, p=_P, positiva: str = "si") -> Muestra:
    return Muestra.binaria(y, classes=("no", "si"), positive_label=positiva,
                           probabilidades=p if positiva == "si" else [1 - x for x in p])


def _cohorte_clinica_sintetica(n: int = 400, *, seed: int = 20260825
                               ) -> tuple[Muestra, list[dict]]:
    """La forma del caso clínico SINTÉTICO de `/casos`: reingreso hospitalario,
    con la receta que viaja en ese paquete —`alto: edad > 75 OR
    reingreso_previo > 0.5`, si no `bajo`— y un modelo que la aprende a medias.

    **Datos sintéticos: aquí no hay precisión clínica que atribuir a nadie.**
    Lo que se comprueba con esto es el CÁLCULO del perfil, no un modelo.
    """
    rng = random.Random(seed)
    filas = []
    ys, ps = [], []
    for _ in range(n):
        edad = rng.uniform(18, 100)
        dias = rng.uniform(0, 60)
        diagnosticos = rng.randint(0, 15)
        reingreso = rng.uniform(0, 1)
        y = "alto" if (edad > 75 or reingreso > 0.5) else "bajo"
        # Un "modelo" que ve las dos señales de la receta con ruido: ni
        # perfecto ni inútil, que es donde una curva de decisión dice algo.
        z = 0.06 * (edad - 70) + 3.0 * (reingreso - 0.5) + rng.gauss(0, 0.9)
        p = 1.0 / (1.0 + 2.718281828459045 ** (-z))
        filas.append({"edad": edad, "dias_ingresado": dias,
                      "num_diagnosticos": diagnosticos, "reingreso_previo": reingreso})
        ys.append(y)
        ps.append(p)
    muestra = Muestra.binaria(ys, classes=("bajo", "alto"), positive_label="alto",
                              probabilidades=ps)
    return muestra, filas


class BeneficioNetoAManoTest(unittest.TestCase):
    """`beneficio neto = TP/n − FP/n · pt/(1−pt)`, con lápiz."""

    def test_pt_0_20_da_0_30(self):
        """pt=0,20 → se trata a quien tenga p >= 0,20: las cuatro filas `si`
        (0,90 0,75 0,60 0,35) y cuatro `no` (0,55 0,45 0,30 0,20). TP=4, FP=4.
        w = 0,20/0,80 = 0,25.
        NB = 4/10 − (4/10)·0,25 = 0,4 − 0,1 = **0,30**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=4, fp=4, n=10, umbral_de_probabilidad=0.20), 0.30, places=12)

    def test_pt_0_50_da_0_20(self):
        """pt=0,50 → p >= 0,50: 0,90 0,75 0,60 (`si`) y 0,55 (`no`). TP=3, FP=1.
        w = 0,50/0,50 = 1. NB = 3/10 − (1/10)·1 = **0,20**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=3, fp=1, n=10, umbral_de_probabilidad=0.50), 0.20, places=12)

    def test_pt_0_80_da_0_10(self):
        """pt=0,80 → solo 0,90. TP=1, FP=0. w = 0,80/0,20 = 4.
        NB = 1/10 − 0·4 = **0,10**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=1, fp=0, n=10, umbral_de_probabilidad=0.80), 0.10, places=12)

    def test_pt_0_10_da_31_novenos_de_noventa(self):
        """pt=0,10 → todos menos 0,05. TP=4, FP=5. w = 0,10/0,90 = 1/9.
        NB = 2/5 − (1/2)·(1/9) = 36/90 − 5/90 = **31/90 = 0,344444…**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=4, fp=5, n=10, umbral_de_probabilidad=0.10),
            31 / 90, places=12)

    def test_el_CODIGO_da_lo_MISMO_que_el_lapiz(self):
        """La otra mitad: que la curva completa, contando ella sola TP y FP
        sobre la cohorte, reproduzca los cuatro números de arriba."""
        curva = curva_de_decision(_a_mano(),
                                  umbrales_de_probabilidad=(0.10, 0.20, 0.50, 0.80))
        esperados = {0.10: (4, 5, 31 / 90), 0.20: (4, 4, 0.30),
                     0.50: (3, 1, 0.20), 0.80: (1, 0, 0.10)}
        self.assertEqual(len(curva.puntos), 4)
        for punto in curva.puntos:
            tp, fp, nb = esperados[punto.umbral_de_probabilidad]
            with self.subTest(pt=punto.umbral_de_probabilidad):
                self.assertEqual((punto.tp, punto.fp), (tp, fp))
                self.assertAlmostEqual(punto.beneficio_neto, nb, places=12)

    def test_pt_igual_a_uno_no_es_un_beneficio_enorme_sino_ninguno(self):
        """`pt/(1−pt)` divide por cero. Se rechaza; no se devuelve un número."""
        with self.assertRaises(EntradaNoMedible) as e:
            beneficio_neto(tp=1, fp=0, n=10, umbral_de_probabilidad=1.0)
        self.assertEqual(e.exception.clave, "umbral_de_probabilidad_uno")

    def test_pt_cero_es_legitimo_y_no_penaliza_los_falsos_positivos(self):
        """w = 0/(1−0) = 0: a pt=0 tratar no cuesta nada, y NB = TP/n."""
        self.assertAlmostEqual(
            beneficio_neto(tp=4, fp=6, n=10, umbral_de_probabilidad=0.0), 0.4, places=12)


class LasDosReferenciasTest(unittest.TestCase):
    """«Tratar a todos» y «a nadie», que es lo que convierte un número suelto
    en una decisión."""

    def test_a_nadie_vale_cero_por_definicion_y_no_se_mide(self):
        curva = curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.10, 0.50))
        for punto in curva.puntos:
            self.assertEqual(punto.beneficio_neto_no_tratar, 0.0)
        self.assertEqual(BENEFICIO_NETO_NO_TRATAR, 0.0)

    def test_tratar_a_todos_a_mano(self):
        """Tratar a todos: TP = los 4 positivos, FP = los 6 negativos.
        pt=0,20 → 4/10 − (6/10)·0,25 = 0,4 − 0,15 = **0,25**.
        pt=0,50 → 4/10 − (6/10)·1    = 0,4 − 0,60 = **−0,20**.
        pt=0,10 → 2/5 − (3/5)·(1/9)  = 6/15 − 1/15 = **1/3**."""
        curva = curva_de_decision(_a_mano(),
                                  umbrales_de_probabilidad=(0.10, 0.20, 0.50))
        esperados = {0.10: 1 / 3, 0.20: 0.25, 0.50: -0.20}
        for punto in curva.puntos:
            with self.subTest(pt=punto.umbral_de_probabilidad):
                self.assertAlmostEqual(punto.beneficio_neto_tratar_a_todos,
                                       esperados[punto.umbral_de_probabilidad],
                                       places=12)

    def test_este_modelo_supera_a_las_dos_donde_se_esperaba(self):
        curva = curva_de_decision(_a_mano(),
                                  umbrales_de_probabilidad=(0.10, 0.20, 0.50, 0.80))
        self.assertTrue(all(p.supera_las_referencias for p in curva.puntos))
        self.assertEqual(curva.umbrales_sin_ventaja, ())
        self.assertTrue(curva.supera_en_algun_umbral)

    def test_empatar_con_tratar_a_todos_NO_es_superarlo(self):
        """pt=0,05: el modelo trata a los diez (todos tienen p >= 0,05), así
        que TP=4 y FP=6 son los MISMOS que los de «tratar a todos» y los dos
        beneficios netos valen 0,4 − 0,6·(1/19) = **0,368421…**. Aportar lo
        mismo que tratar a todo el mundo no es aportar nada."""
        curva = curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.05,))
        punto = curva.puntos[0]
        self.assertEqual((punto.tp, punto.fp), (4, 6))
        self.assertAlmostEqual(punto.beneficio_neto, 0.4 - 0.6 / 19, places=12)
        self.assertAlmostEqual(punto.beneficio_neto_tratar_a_todos, 0.4 - 0.6 / 19,
                               places=12)
        self.assertFalse(punto.supera_las_referencias)
        self.assertEqual(curva.umbrales_sin_ventaja, (0.05,))
        self.assertFalse(curva.supera_en_algun_umbral)

    def test_un_modelo_al_reves_no_supera_a_tratar_a_todos_y_la_curva_lo_dice(self):
        """La probabilidad invertida, a pt=0,20: se trata a quien tenga
        1−p >= 0,20, o sea p <= 0,80 → tres `si` (0,75 0,60 0,35) y los seis
        `no`. TP=3, FP=6. NB = 3/10 − (6/10)·0,25 = 0,30 − 0,15 = **0,15**,
        por debajo del 0,25 de tratar a todos."""
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                    probabilidades=[1 - p for p in _P])
        curva = curva_de_decision(invertida, umbrales_de_probabilidad=(0.20,))
        punto = curva.puntos[0]
        self.assertEqual((punto.tp, punto.fp), (3, 6))
        self.assertAlmostEqual(punto.beneficio_neto, 0.15, places=12)
        self.assertAlmostEqual(punto.beneficio_neto_tratar_a_todos, 0.25, places=12)
        self.assertGreater(punto.beneficio_neto, punto.beneficio_neto_no_tratar)
        self.assertFalse(punto.supera_las_referencias)


class InversionDeLaProbabilidadTest(unittest.TestCase):
    """El control conocido-bueno. Si esto no casa, ningún número de arriba
    vale nada."""

    def test_invertir_p_con_el_MISMO_umbral_COMPLEMENTA_sens_y_esp(self):
        """Original en t=0,5: TP=3 (0,90 0,75 0,60), FN=1 (0,35), TN=5, FP=1
        (0,55) → sens = 3/4 = **0,75**, esp = 5/6 = **0,8333…**.
        Invertida (p → 1−p, misma clase positiva, mismo t=0,5): se predice `si`
        cuando 1−p >= 0,5, o sea p <= 0,5 → 0,35 (`si`) y 0,45 0,30 0,20 0,10
        0,05 (`no`) → TP=1, FN=3, FP=5, TN=1 → sens = 1/4 = **0,25** = 1−0,75 y
        esp = 1/6 = **0,1666…** = 1−0,8333.
        Eso es el COMPLEMENTO, no el intercambio: medido, no supuesto."""
        directa = _a_mano()
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                    probabilidades=[1 - p for p in _P])
        sens = calcular("sensitivity", directa, umbral=0.5).value
        esp = calcular("specificity", directa, umbral=0.5).value
        self.assertAlmostEqual(sens, 0.75, places=12)
        self.assertAlmostEqual(esp, 5 / 6, places=12)

        sens_inv = calcular("sensitivity", invertida, umbral=0.5).value
        esp_inv = calcular("specificity", invertida, umbral=0.5).value
        self.assertAlmostEqual(sens_inv, 0.25, places=12)
        self.assertAlmostEqual(esp_inv, 1 / 6, places=12)
        self.assertAlmostEqual(sens_inv, 1 - sens, places=12)
        self.assertAlmostEqual(esp_inv, 1 - esp, places=12)
        # Y NO es un intercambio: decirlo así sería heredar una frase sin medirla.
        self.assertNotAlmostEqual(sens_inv, esp, places=6)
        self.assertNotAlmostEqual(esp_inv, sens, places=6)

    def test_invertir_la_probabilidad_ENTERA_INTERCAMBIA_sens_y_esp(self):
        """La inversión escrita entera: `1−p` es la probabilidad de la OTRA
        clase, así que la clase positiva pasa a ser `no` y el umbral se refleja
        a 1−t. Es el MISMO clasificador visto desde el otro lado, y por eso la
        sensibilidad de uno es la especificidad del otro.

        Con t=0,5: la invertida predice `no` cuando 1−p >= 0,5 ⟺ p <= 0,5 →
        0,35 (`si`) y los cinco `no` por debajo de 0,5. TP(no)=5, FN(no)=1
        (0,55), TN(no)=3 (0,90 0,75 0,60), FP(no)=1 (0,35) →
        sens_inv = 5/6 = esp_directa **0,8333…** y esp_inv = 3/4 =
        sens_directa **0,75**. INTERCAMBIO exacto."""
        directa = _a_mano()
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                                    probabilidades=[1 - p for p in _P])
        sens = calcular("sensitivity", directa, umbral=0.5).value
        esp = calcular("specificity", directa, umbral=0.5).value
        sens_inv = calcular("sensitivity", invertida, umbral=0.5).value
        esp_inv = calcular("specificity", invertida, umbral=0.5).value
        self.assertAlmostEqual(sens_inv, esp, places=12)
        self.assertAlmostEqual(esp_inv, sens, places=12)
        self.assertAlmostEqual(sens_inv, 5 / 6, places=12)
        self.assertAlmostEqual(esp_inv, 0.75, places=12)

    def test_el_intercambio_exacto_se_ROMPE_con_un_empate_en_el_umbral(self):
        """Por qué la cohorte de arriba no tiene ningún `p` igual a 0,5, y no
        es casualidad: la matriz de confusión decide con `>=`, así que la fila
        que cae JUSTO en el umbral se cuenta positiva en los dos lados de la
        inversión y el intercambio deja de ser exacto por esa fila.

        Cohorte igual pero con la fila 4 en 0,50 en vez de 0,35: la directa da
        sens = 4/4 = 1,0 y esp = 5/6; la invertida da sens = 5/6 (coincide) y
        esp = 3/4, que NO es 1,0. Está escrito para que nadie 'simplifique' la
        cohorte metiendo un 0,5."""
        p_con_empate = (0.90, 0.75, 0.60, 0.50, 0.55, 0.45, 0.30, 0.20, 0.10, 0.05)
        directa = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                  probabilidades=p_con_empate)
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                                    probabilidades=[1 - p for p in p_con_empate])
        self.assertAlmostEqual(calcular("sensitivity", directa, umbral=0.5).value, 1.0)
        self.assertAlmostEqual(calcular("specificity", directa, umbral=0.5).value, 5 / 6)
        self.assertAlmostEqual(calcular("sensitivity", invertida, umbral=0.5).value, 5 / 6)
        self.assertAlmostEqual(calcular("specificity", invertida, umbral=0.5).value, 0.75)

    def test_la_TABLA_del_perfil_tambien_intercambia_al_invertir(self):
        """No basta con que lo hagan las fórmulas de C1: lo que publica este
        corte es la tabla, y es la tabla la que tiene que cazarlo."""
        comun = dict(umbrales=(0.5,), diseno="iid",
                     estimando="fixed_model_on_population", semilla=1, remuestras=50)
        directa = tabla_de_umbrales(_a_mano(), **comun).filas[0]
        invertida = tabla_de_umbrales(
            Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                            probabilidades=[1 - p for p in _P]), **comun).filas[0]
        self.assertAlmostEqual(invertida.sensibilidad.value, directa.especificidad.value,
                               places=12)
        self.assertAlmostEqual(invertida.especificidad.value, directa.sensibilidad.value,
                               places=12)
        self.assertEqual((invertida.tp, invertida.fn), (directa.tn, directa.fp))


class PrevalenciaTest(unittest.TestCase):
    """Lo que más importa de este corte: PPV y NPV dependen de la prevalencia
    y sens/esp no."""

    def test_sin_prevalencia_declarada_NO_se_publican_ppv_ni_npv(self):
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  remuestras=50)
        fila = tabla.filas[0]
        self.assertFalse(tabla.publica_ppv_y_npv)
        self.assertIsNone(tabla.prevalencia_declarada)
        for valor in (fila.ppv, fila.npv):
            with self.subTest(metric_id=valor.metric_id):
                self.assertIsNone(valor.value)
                self.assertIsNotNone(valor.undefined_reason)
                self.assertEqual(set(valor.undefined_reason), set(IDIOMAS))

    def test_y_NO_se_callan_las_que_no_dependen_de_ella(self):
        """La otra mitad: un aserto negativo lo pasaría un perfil que no
        publicara nada. Sensibilidad y especificidad SÍ salen, con IC, porque
        no dependen de la prevalencia."""
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  remuestras=200)
        fila = tabla.filas[0]
        self.assertAlmostEqual(fila.sensibilidad.value, 0.75, places=12)
        self.assertAlmostEqual(fila.especificidad.value, 5 / 6, places=12)
        for valor in (fila.sensibilidad, fila.especificidad):
            with self.subTest(metric_id=valor.metric_id):
                self.assertIsNotNone(valor.uncertainty)
                self.assertIsNotNone(valor.uncertainty["ci_low"])
                self.assertLessEqual(valor.uncertainty["ci_low"], valor.value)
                self.assertGreaterEqual(valor.uncertainty["ci_high"], valor.value)

    def test_el_ppv_a_prevalencia_2_por_ciento_a_mano(self):
        """sens = 3/4, esp = 5/6 en t=0,5. A una prevalencia declarada del 2 %:

          PPV = sens·pr / (sens·pr + (1−esp)·(1−pr))
              = 0,75·0,02 / (0,75·0,02 + (1/6)·0,98)
              = 0,015 / 0,178333… = **0,0841121…**

          NPV = esp·(1−pr) / (esp·(1−pr) + (1−sens)·pr)
              = (5/6)·0,98 / ((5/6)·0,98 + 0,25·0,02)
              = 0,816666… / 0,821666… = **0,9939148…**

        Y el PPV OBSERVADO en esta muestra vale 3/4 = 0,75. Nueve veces el
        transportado, con la misma aritmética impecable: eso es lo que se
        estaría publicando si la prevalencia no viajara con el número."""
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  prevalencia=0.02, remuestras=50)
        fila = tabla.filas[0]
        self.assertAlmostEqual(fila.ppv.value, 0.015 / (0.015 + (1 / 6) * 0.98),
                               places=12)
        self.assertAlmostEqual(fila.ppv.value, 0.08411214953271028, places=12)
        self.assertAlmostEqual(fila.npv.value, 0.9939148073022313, places=12)
        self.assertAlmostEqual(calcular("ppv", _a_mano(), umbral=0.5).value, 0.75,
                               places=12)
        self.assertEqual(fila.ppv.metric_id, PPV_A_PREVALENCIA)
        self.assertEqual(fila.npv.metric_id, NPV_A_PREVALENCIA)

    def test_la_prevalencia_de_la_muestra_viaja_al_lado_de_la_declarada(self):
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  prevalencia=0.02, remuestras=50)
        self.assertAlmostEqual(tabla.prevalencia_declarada, 0.02)
        self.assertAlmostEqual(tabla.prevalencia_observada, 0.4)

    def test_el_transportado_trae_su_intervalo(self):
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=3,
                                  prevalencia=0.02, remuestras=300)
        ppv = tabla.filas[0].ppv
        self.assertIsNotNone(ppv.uncertainty)
        self.assertEqual(ppv.uncertainty["metric_id"], PPV_A_PREVALENCIA)
        self.assertIsNotNone(ppv.uncertainty["ci_low"])
        self.assertLessEqual(ppv.uncertainty["ci_low"], ppv.value)
        self.assertGreaterEqual(ppv.uncertainty["ci_high"], ppv.value)

    def test_a_prevalencia_igual_a_la_de_la_muestra_el_transporte_da_el_observado(self):
        """Control del instrumento: con pr = la prevalencia de la muestra, la
        fórmula de Bayes tiene que devolver el PPV observado (0,75) y el NPV
        observado (5/6). Si no lo hiciera, el transporte estaría mal escrito y
        los números de arriba tampoco valdrían."""
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  prevalencia=0.4, remuestras=50)
        fila = tabla.filas[0]
        self.assertAlmostEqual(fila.ppv.value,
                               calcular("ppv", _a_mano(), umbral=0.5).value, places=12)
        self.assertAlmostEqual(fila.npv.value,
                               calcular("npv", _a_mano(), umbral=0.5).value, places=12)

    def test_una_prevalencia_fuera_de_cero_uno_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                              estimando="fixed_model_on_population", semilla=1,
                              prevalencia=1.5, remuestras=10)


class UmbralesYCableadoTest(unittest.TestCase):
    def test_los_umbrales_los_declara_quien_decide_y_no_hay_omision(self):
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(_a_mano(), umbrales_de_probabilidad=())
        self.assertEqual(e.exception.clave, "umbrales_no_declarados")
        with self.assertRaises(EntradaNoMedible):
            tabla_de_umbrales(_a_mano(), umbrales=(), diseno="iid",
                              estimando="fixed_model_on_population", semilla=1)

    def test_un_umbral_repetido_no_son_dos_medidas(self):
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.2, 0.2))
        self.assertEqual(e.exception.clave, "umbral_de_probabilidad_repetido")

    def test_una_muestra_con_la_clase_YA_decidida_se_rechaza(self):
        """Sobre etiquetas fijas la curva saldría igual en todos los umbrales:
        una curva plana creíble y falsa. Es cableado, así que excepción."""
        con_etiquetas = Muestra.binaria(
            _Y, classes=("no", "si"), positive_label="si", probabilidades=_P,
            predicciones=["si" if p >= 0.5 else "no" for p in _P])
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(con_etiquetas, umbrales_de_probabilidad=(0.2, 0.5))
        self.assertEqual(e.exception.clave, "dca_con_etiquetas_declaradas")

    def test_una_muestra_de_puntuaciones_crudas_no_tiene_curva_de_decision(self):
        cruda = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                puntuaciones=[p * 10 - 5 for p in _P])
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(cruda, umbrales_de_probabilidad=(0.2,))
        self.assertEqual(e.exception.clave, "dca_sin_probabilidad_calibrada")

    def test_una_muestra_con_probabilidad_Y_puntuacion_cruda_se_rechaza(self):
        """Medido el 2026-09-14: `Muestra.escala_de_decision` dice
        `calibrated_probability` en cuanto hay `probabilities`, pero
        `puntuacion_del_positivo` —la que usa `matriz_de_confusion`— da
        prioridad a `scores`. Con las dos presentes, un `pt` de probabilidad
        acabaría comparándose contra un logit y la curva saldría creíble sobre
        un eje que no es el suyo."""
        ambigua = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                  probabilidades=_P,
                                  puntuaciones=[p * 10 - 5 for p in _P])
        self.assertEqual(ambigua.escala_de_decision, "calibrated_probability")
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(ambigua, umbrales_de_probabilidad=(0.2,))
        self.assertEqual(e.exception.clave, "dca_sin_probabilidad_calibrada")

    def test_la_tabla_sobre_puntuaciones_crudas_admite_umbrales_fuera_de_0_1(self):
        """La otra mitad: con puntuaciones crudas el umbral NO vive en [0,1], y
        exigirlo ahí rechazaría un umbral legítimo. Lo que manda es `scores`,
        que es lo que mira la matriz de confusión."""
        cruda = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                puntuaciones=[p * 10 - 5 for p in _P])
        tabla = tabla_de_umbrales(cruda, umbrales=(0.0,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  remuestras=20)
        self.assertAlmostEqual(tabla.filas[0].sensibilidad.value, 0.75, places=12)
        with self.assertRaises(EsquemaInvalido):
            tabla_de_umbrales(_a_mano(), umbrales=(3.0,), diseno="iid",
                              estimando="fixed_model_on_population", semilla=1,
                              remuestras=20)

    def test_cero_filas_no_dan_una_curva_en_cero(self):
        vacia = Muestra(task="binary_classification", y_true=(), classes=("no", "si"),
                        positive_label="si", probabilities=())
        curva = curva_de_decision(vacia, umbrales_de_probabilidad=(0.2,))
        self.assertEqual(curva.puntos, ())
        self.assertIsNotNone(curva.undefined_reason)
        self.assertIsNone(curva.prevalencia_observada)

    def test_un_intervalo_derivado_sin_su_punto_se_rechaza(self):
        """`formula=` sin `valor=` mandaría a `calcular()` a buscar en el
        catálogo un `metric_id` que por definición no está."""
        with self.assertRaises(EntradaNoMedible) as e:
            intervalo(PPV_A_PREVALENCIA, _a_mano(), diseno="iid",
                      estimando="fixed_model_on_population", semilla=1,
                      remuestras=10, formula=lambda m, u: 0.5)
        self.assertEqual(e.exception.clave, "intervalo_derivado_sin_punto")


class AlcanceDeValidacionTest(unittest.TestCase):
    def test_lo_normal_es_interna_unicamente(self):
        self.assertEqual(alcance_de_validacion(evidencia="independent_test",
                                               diseno="iid"), "internal_only")
        self.assertEqual(alcance_de_validacion(evidencia="development_estimate",
                                               diseno="iid"), "internal_only")
        self.assertEqual(alcance_de_validacion(evidencia="not_evaluated",
                                               diseno="temporal"), "internal_only")

    def test_temporal_y_externa_solo_cuando_existe_la_cohorte(self):
        self.assertEqual(alcance_de_validacion(evidencia="independent_test",
                                               diseno="temporal"), "temporal")
        self.assertEqual(alcance_de_validacion(evidencia="external_validation",
                                               diseno="iid"), "external")

    def test_todos_los_alcances_estan_en_el_vocabulario(self):
        for evidencia in ("independent_test", "external_validation",
                          "development_estimate", "repeated_test_use",
                          "test_used_for_development", "not_evaluated"):
            for diseno in ("iid", "temporal", "groups", "groups_and_time"):
                with self.subTest(evidencia=evidencia, diseno=diseno):
                    self.assertIn(alcance_de_validacion(evidencia=evidencia,
                                                        diseno=diseno),
                                  ALCANCES_DE_VALIDACION)


def _perfil(**cambios):
    muestra = _a_mano()
    base = dict(
        perfil_id="perfil-a-mano", evidencia="independent_test", diseno="iid",
        datos_sinteticos=True,
        tabla=tabla_de_umbrales(muestra, umbrales=(0.5,), diseno="iid",
                                estimando="fixed_model_on_population", semilla=1,
                                remuestras=50),
        curva=curva_de_decision(muestra, umbrales_de_probabilidad=(0.2, 0.5)))
    base.update(cambios)
    return PerfilClinico(**base)


class SegmentosPredefinidosTest(unittest.TestCase):
    """«Sin declaración no hay subgrupos confirmatorios» (decisión 5 del 109),
    hecho cumplir en la construcción y no en una nota."""

    def _segmento(self, predefinido: bool):
        return analizar_segmento("sensitivity", _a_mano(), [0, 1, 4, 5],
                                 segmento_id="mayores_de_75", predefinido=predefinido,
                                 umbral=0.5)

    def test_sin_equipo_un_segmento_no_puede_viajar_como_predefinido(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _perfil(segmentos=(self._segmento(True),))
        self.assertEqual(e.exception.clave, "segmento_confirmatorio_sin_equipo")

    def test_sin_equipo_los_exploratorios_SI_viajan_marcados(self):
        """La otra mitad: prohibirlo todo dejaría el perfil mudo. Lo que no se
        puede es presentarlo como predefinido."""
        perfil = _perfil(segmentos=(self._segmento(False),))
        self.assertEqual(perfil.segmentos_exploratorios, ("mayores_de_75",))
        self.assertIsNone(perfil.segmentos_predefinidos_por)

    def test_con_equipo_declarado_si_se_puede(self):
        perfil = _perfil(segmentos=(self._segmento(True),),
                         segmentos_predefinidos_por="comité de calidad del hospital X")
        self.assertEqual(perfil.segmentos_exploratorios, ())

    def test_la_regla_vale_TAMBIEN_para_las_curvas_por_segmento(self):
        curva = curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.2,),
                                  segmento_id="mayores_de_75", predefinido=True)
        with self.assertRaises(EsquemaInvalido) as e:
            _perfil(curvas_por_segmento=(curva,))
        self.assertEqual(e.exception.clave, "segmento_confirmatorio_sin_equipo")

    def test_una_curva_de_segmento_sin_id_no_es_un_segmento(self):
        with self.assertRaises(EsquemaInvalido):
            CurvaDeDecision(puntos=(), n=0, segmento_id="x", predefinido=None,
                            undefined_reason={"es": "a", "en": "b"})


class FichaDelPerfilTest(unittest.TestCase):
    """Lo que lee una persona. El alcance va ARRIBA, y lo que no se publica se
    dice con su motivo."""

    def _indice(self, ficha: str, trozo: str) -> int:
        self.assertIn(trozo, ficha, f"no está en la ficha: {trozo!r}")
        return ficha.index(trozo)

    def test_validacion_interna_unicamente_va_ANTES_que_el_primer_numero(self):
        for locale, aviso, umbrales in (("es", "VALIDACIÓN INTERNA ÚNICAMENTE", "## Umbrales"),
                                        ("en", "INTERNAL VALIDATION ONLY", "## Thresholds")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(_perfil(), locale=locale)
                sitio = self._indice(ficha, aviso)
                self.assertLess(sitio, self._indice(ficha, umbrales))
                self.assertLess(sitio, self._indice(ficha, "0.75"))
                # y no en una nota al pie: dentro de las cinco primeras líneas
                lineas = [l for l in ficha.splitlines() if l.strip()]
                self.assertIn(aviso, "\n".join(lineas[:5]))

    def test_una_validacion_externa_NO_dice_interna_unicamente(self):
        """La otra mitad: si la frase saliera siempre, no estaría diciendo nada."""
        ficha = ficha_del_perfil(_perfil(evidencia="external_validation"), locale="es")
        self.assertNotIn("VALIDACIÓN INTERNA ÚNICAMENTE", ficha)
        self.assertIn("VALIDACIÓN EXTERNA", ficha)

    def test_los_datos_sinteticos_se_avisan_y_no_se_les_atribuye_precision(self):
        self.assertIn("SINTÉTICOS", ficha_del_perfil(_perfil(), locale="es"))
        self.assertIn("SYNTHETIC", ficha_del_perfil(_perfil(), locale="en"))
        limpia = ficha_del_perfil(_perfil(datos_sinteticos=False), locale="es")
        self.assertNotIn("SINTÉTICOS", limpia)

    def test_sin_prevalencia_la_ficha_DICE_que_calla_ppv_y_npv(self):
        for locale, frase in (("es", "no se publican PPV"), ("en", "PPV and NPV are not")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(_perfil(), locale=locale)
                self.assertIn(frase, ficha)

    def test_la_casilla_vacia_no_queda_MUDA_y_el_motivo_va_debajo(self):
        """La casilla dice «no publicado» y el motivo entero, con las métricas
        que lo comparten, va una sola vez debajo de la tabla."""
        for locale, marca, motivo_ in (("es", "no publicado", "nadie ha declarado la prevalencia"),
                                       ("en", "not published", "nobody declared target")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(_perfil(), locale=locale)
                self.assertIn(f"| {marca} |", ficha)
                self.assertIn(motivo_, ficha)
                self.assertIn(f"**{PPV_A_PREVALENCIA}, {NPV_A_PREVALENCIA}**", ficha)
                # una sola vez, no una por umbral
                self.assertEqual(ficha.count(motivo_), 1)

    def test_con_prevalencia_no_hay_nota_de_ausencia(self):
        """La otra mitad: la nota no se imprime siempre."""
        perfil = _perfil(tabla=tabla_de_umbrales(
            _a_mano(), umbrales=(0.5,), diseno="iid",
            estimando="fixed_model_on_population", semilla=1, prevalencia=0.02,
            remuestras=50))
        self.assertNotIn("no publicado", ficha_del_perfil(perfil, locale="es"))
        self.assertNotIn("not published", ficha_del_perfil(perfil, locale="en"))

    def test_con_prevalencia_la_ficha_la_ENSEÑA_al_lado_del_numero(self):
        muestra = _a_mano()
        perfil = _perfil(tabla=tabla_de_umbrales(
            muestra, umbrales=(0.5,), diseno="iid",
            estimando="fixed_model_on_population", semilla=1, prevalencia=0.02,
            remuestras=50, declarados_por="el equipo de medicina interna"))
        ficha = ficha_del_perfil(perfil, locale="es")
        self.assertIn("0.0200", ficha)          # la declarada
        self.assertIn("0.4000", ficha)          # la de la muestra, al lado
        self.assertIn("0.0841", ficha)          # el PPV transportado
        self.assertIn("el equipo de medicina interna", ficha)
        self.assertIn("Bayes", ficha)           # el supuesto, escrito

    def test_cuando_el_modelo_no_supera_a_las_dos_la_ficha_lo_dice(self):
        perfil = _perfil(curva=curva_de_decision(_a_mano(),
                                                 umbrales_de_probabilidad=(0.05,)))
        self.assertIn("no supera a las dos referencias en ningún",
                      ficha_del_perfil(perfil, locale="es"))
        self.assertIn("beats both references at no declared",
                      ficha_del_perfil(perfil, locale="en"))

    def test_y_cuando_SI_supera_no_lo_dice(self):
        ficha = ficha_del_perfil(_perfil(), locale="es")
        self.assertNotIn("no supera a las dos referencias en ningún", ficha)

    def test_la_ficha_dice_que_el_BENEFICIO_NETO_tambien_es_de_esta_muestra(self):
        """`TP/n` y `FP/n` son proporciones de la muestra: el beneficio neto
        depende de la prevalencia igual que el PPV. Lo que NO se puede hacer es
        transportarlo con una fórmula, y la ficha lo dice cuando hay una
        prevalencia declarada distinta."""
        for locale, frase in (("es", "SOBRE ESTA MUESTRA"), ("en", "ON THIS SAMPLE")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))
        # la prevalencia observada, al lado
        self.assertIn("0.4000", ficha_del_perfil(_perfil(), locale="es"))

        con_prevalencia = _perfil(tabla=tabla_de_umbrales(
            _a_mano(), umbrales=(0.5,), diseno="iid",
            estimando="fixed_model_on_population", semilla=1, prevalencia=0.02,
            remuestras=50))
        for locale, frase in (("es", "no está transportado a la prevalencia"),
                              ("en", "it is not transported to declared")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(con_prevalencia, locale=locale))

    def test_y_sin_prevalencia_declarada_no_habla_de_transportar(self):
        """La otra mitad: el aviso del transporte solo aparece cuando hay algo
        a lo que no se transporta."""
        self.assertNotIn("no está transportado",
                         ficha_del_perfil(_perfil(), locale="es"))

    def test_sin_calibracion_medida_la_ficha_no_la_da_por_buena(self):
        for locale, frase in (("es", "No se ha medido la calibración"),
                              ("en", "Calibration was not measured")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))

    def test_con_calibracion_medida_sale_el_ece(self):
        perfil = _perfil(calibracion=curva_de_fiabilidad(_a_mano(), n_bins=4))
        self.assertIn("ECE", ficha_del_perfil(perfil, locale="es"))

    def test_sin_politica_de_faltantes_se_dice_que_no_consta(self):
        for locale, frase in (("es", "No se declara ninguna política"),
                              ("en", "No missing-data policy")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))

    def test_con_politica_de_faltantes_se_escribe(self):
        perfil = _perfil(politica_de_faltantes={"edad": "mediana de train + indicador"})
        self.assertIn("mediana de train + indicador",
                      ficha_del_perfil(perfil, locale="es"))

    def test_un_segmento_sin_cifra_no_sale_con_un_guion_mudo(self):
        """Un segmento de solo negativos no tiene sensibilidad: la ficha
        escribe POR QUÉ, no un guion."""
        solo_negativos = [i for i, y in enumerate(_Y) if y == "no"]
        segmento = analizar_segmento("sensitivity", _a_mano(), solo_negativos,
                                     segmento_id="solo_negativos", predefinido=False,
                                     umbral=0.5)
        self.assertIsNone(segmento.metrica_segmento.value)
        ficha = ficha_del_perfil(_perfil(segmentos=(segmento,)), locale="es")
        self.assertIn("solo_negativos", ficha)
        self.assertIn(segmento.metrica_segmento.undefined_reason["es"], ficha)

    def test_sin_equipo_la_ficha_dice_que_nadie_predefinio_subgrupos(self):
        for locale, frase in (("es", "Nadie ha predefinido subgrupos"),
                              ("en", "Nobody predefined any subgroup")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))

    def test_la_ficha_INGLESA_no_lleva_castellano_colado(self):
        """Un barrido con palabras FUNCIONALES sobre la ficha entera: no se
        pueden evitar escribiendo en castellano, y cazan lo que una lista de
        palabras escogida a mano no caza — por ejemplo un «IC 95 %» que se
        quedó sin traducir dentro de una casilla."""
        import re
        segmento = analizar_segmento("sensitivity", _a_mano(), [0, 1, 4, 5],
                                     segmento_id="segment_a", predefinido=False,
                                     umbral=0.5)
        perfil = _perfil(segmentos=(segmento,),
                         curva=curva_de_decision(_a_mano(),
                                                 umbrales_de_probabilidad=(0.05, 0.5)),
                         calibracion=curva_de_fiabilidad(_a_mano(), n_bins=4))
        ficha = ficha_del_perfil(perfil, locale="en")
        palabras = set(re.findall(r"[a-záéíóúñ]+", ficha.lower()))
        self.assertEqual(palabras.intersection(TextosDeLaFichaTest.FUNCIONALES), set())

    def test_y_la_ficha_CASTELLANA_no_esta_en_ingles(self):
        """La otra mitad: un barrido que solo mira la inglesa lo pasaría una
        ficha que saliera siempre en inglés."""
        ficha = ficha_del_perfil(_perfil(), locale="es")
        for marca in ("Thresholds", "not published", "INTERNAL VALIDATION"):
            self.assertNotIn(marca, ficha)

    def test_un_idioma_que_no_se_sabe_sale_en_ingles(self):
        self.assertEqual(ficha_del_perfil(_perfil(), locale="fr"),
                         ficha_del_perfil(_perfil(), locale="en"))


class TextosDeLaFichaTest(unittest.TestCase):
    """Lo que redacta el core se traduce en el core: las dos redacciones, con
    las mismas claves y sin castellano colado en la inglesa."""

    FUNCIONALES = ("el", "la", "los", "las", "del", "con", "que", "desde", "para",
                   "por", "una", "uno", "sin", "pero", "como", "cuando", "más",
                   "porque", "aunque", "sobre", "entre", "hay", "tiene", "puede")

    def test_los_dos_idiomas_tienen_las_mismas_claves(self):
        self.assertEqual(set(_T), set(IDIOMAS))
        self.assertEqual(set(_T["es"]), set(_T["en"]))

    def test_ninguna_redaccion_esta_vacia(self):
        for idioma in IDIOMAS:
            for clave, texto in _T[idioma].items():
                with self.subTest(idioma=idioma, clave=clave):
                    self.assertTrue(texto.strip())

    def test_los_huecos_son_los_mismos_en_los_dos_idiomas(self):
        for clave in _T["es"]:
            with self.subTest(clave=clave):
                self.assertEqual(huecos_de(_T["es"][clave]), huecos_de(_T["en"][clave]))

    def test_el_ingles_no_lleva_castellano_dentro(self):
        import re
        for clave, texto in _T["en"].items():
            palabras = set(re.findall(r"[a-záéíóúñ]+", texto.lower()))
            with self.subTest(clave=clave):
                self.assertEqual(palabras.intersection(self.FUNCIONALES), set())

    #: Las tres casillas que SÍ son iguales en los dos idiomas, y por qué:
    #: `col_recuentos` es «TP/FP/TN/FN» (siglas, no palabras), `col_pt` es el
    #: símbolo `pt` de la literatura de DCA, y «no» se escribe igual en
    #: castellano y en inglés. La lista es CERRADA: añadir una casilla aquí
    #: para que pase la prueba es justo lo que la prueba existe para impedir.
    IGUALES_A_PROPOSITO = ("col_recuentos", "col_pt", "no")

    def test_las_dos_redacciones_son_DISTINTAS(self):
        """Copiar el castellano en la casilla inglesa pasaría las tres pruebas
        de arriba en cuanto no llevara ninguna palabra funcional."""
        iguales = [c for c in _T["es"] if _T["es"][c] == _T["en"][c]]
        self.assertEqual(sorted(iguales), sorted(self.IGUALES_A_PROPOSITO))


class CasoClinicoSinteticoTest(unittest.TestCase):
    """El perfil entero sobre el caso clínico SINTÉTICO de `/casos` (reingreso
    hospitalario). **Datos sintéticos: se comprueba el cálculo, nunca la
    precisión clínica.**"""

    @classmethod
    def setUpClass(cls):
        cls.muestra, cls.filas = _cohorte_clinica_sintetica()

    def test_el_perfil_se_construye_entero_y_sella(self):
        umbrales_del_equipo = (0.3, 0.5, 0.7)
        tabla = tabla_de_umbrales(
            self.muestra, umbrales=umbrales_del_equipo, diseno="iid",
            estimando="fixed_model_on_population", semilla=20260914, remuestras=200)
        curva = curva_de_decision(self.muestra,
                                  umbrales_de_probabilidad=umbrales_del_equipo)
        mayores = [i for i, f in enumerate(self.filas) if f["edad"] > 75]
        perfil = PerfilClinico(
            perfil_id="reingreso-sintetico", evidencia="independent_test",
            diseno="iid", datos_sinteticos=True, tabla=tabla, curva=curva,
            calibracion=curva_de_fiabilidad(self.muestra, n_bins=10),
            segmentos=(analizar_segmento("sensitivity", self.muestra, mayores,
                                         segmento_id="edad_mayor_de_75",
                                         predefinido=False, umbral=0.5),),
            curvas_por_segmento=(curva_de_decision(
                _recorte(self.muestra, mayores),
                umbrales_de_probabilidad=umbrales_del_equipo,
                segmento_id="edad_mayor_de_75", predefinido=False),))
        self.assertEqual(perfil.alcance, "internal_only")
        self.assertEqual(len(perfil.tabla.filas), 3)
        self.assertEqual(len(perfil.curva.puntos), 3)
        self.assertEqual(perfil.segmentos_exploratorios, ("edad_mayor_de_75",))
        self.assertEqual(len(perfil.digest()), 64)
        # el documento se puede serializar entero
        cuerpo = perfil.a_json()
        self.assertEqual(cuerpo["schema"], PerfilClinico.ESQUEMA)
        self.assertEqual(cuerpo["alcance"], "internal_only")
        self.assertTrue(cuerpo["datos_sinteticos"])
        # y la ficha sale en los dos idiomas, con el aviso arriba
        for locale, aviso in (("es", "SINTÉTICOS"), ("en", "SYNTHETIC")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(perfil, locale=locale)
                self.assertIn(aviso, ficha)
                self.assertLess(ficha.index(aviso), len(ficha) // 3)

    def test_la_prevalencia_de_esta_cohorte_NO_es_la_de_un_hospital(self):
        """La receta del caso (`edad > 75 OR reingreso_previo > 0.5`) produce
        una prevalencia de ~65 %. Un PPV calculado ahí y enseñado a un equipo
        cuya población reingresa al 10 % sería un número que miente sin
        equivocarse en ninguna cuenta — y por eso, sin prevalencia declarada,
        no se publica."""
        tabla = tabla_de_umbrales(self.muestra, umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population",
                                  semilla=1, remuestras=100)
        self.assertGreater(tabla.prevalencia_observada, 0.55)
        self.assertIsNone(tabla.filas[0].ppv.value)

        # Medido sobre esta cohorte en t=0,5: prevalencia 0,6925, sens 0,6209,
        # esp 0,9594; PPV observado **0,9718** y NPV observado **0,5291**. A una
        # prevalencia declarada del 2 %, el mismo modelo da PPV **0,2377** y NPV
        # **0,9920**: el PPV cae a la CUARTA parte y el NPV sube. Los dos lados,
        # porque corregir un sesgo puede crear el contrario.
        al_dos = tabla_de_umbrales(self.muestra, umbrales=(0.5,), diseno="iid",
                                   estimando="fixed_model_on_population", semilla=1,
                                   prevalencia=0.02, remuestras=100)
        ppv_observado = calcular("ppv", self.muestra, umbral=0.5).value
        npv_observado = calcular("npv", self.muestra, umbral=0.5).value
        self.assertAlmostEqual(ppv_observado, 0.971751, places=5)
        self.assertAlmostEqual(al_dos.filas[0].ppv.value, 0.237652, places=5)
        self.assertAlmostEqual(npv_observado, 0.529148, places=5)
        self.assertAlmostEqual(al_dos.filas[0].npv.value, 0.992001, places=5)
        self.assertLess(al_dos.filas[0].ppv.value, ppv_observado / 3)
        self.assertGreater(al_dos.filas[0].npv.value, npv_observado)


def _recorte(muestra: Muestra, indices) -> Muestra:
    """Un recorte VALIDADO (no el atajo interno de 105-C2): esta muestra se
    construye una vez, no mil, y pasar por `__post_init__` la comprueba."""
    return Muestra.binaria(
        [muestra.y_true[i] for i in indices], classes=list(muestra.classes),
        positive_label=muestra.positive_label,
        probabilidades=[muestra.probabilidad_del_positivo[i] for i in indices])


class MatrizYCurvaCuadranTest(unittest.TestCase):
    """La curva cuenta TP y FP con la misma matriz de 105-C1: si un día dejaran
    de cuadrar, este corte estaría publicando recuentos propios."""

    def test_los_recuentos_de_la_curva_son_los_de_la_matriz(self):
        muestra = _a_mano()
        for pt in (0.1, 0.2, 0.5, 0.8):
            matriz = matriz_de_confusion(muestra, umbral=pt)
            punto = curva_de_decision(muestra, umbrales_de_probabilidad=(pt,)).puntos[0]
            with self.subTest(pt=pt):
                self.assertEqual((punto.tp, punto.fp), (matriz.tp, matriz.fp))


if __name__ == "__main__":
    unittest.main()
