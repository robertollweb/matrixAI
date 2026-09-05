# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C1 — EL REGISTRO MÉTRICO ÚNICO, Y LO QUE SE NIEGA A INVENTAR.

Lo que se prueba aquí no es que unas divisiones den lo que dan: es que **ningún
número salga de este core diciendo ser lo que no es**. Cada clase existe por un
defecto concreto que el contrato 105 nombra:

* publicar el área PR trapezoidal llamándola `average_precision` le atribuye al
  modelo un número que no es suyo — y con el fixture de aquí la diferencia es
  0,583333 frente a 0,416667, un 40 % más;
* calcular un Brier sobre unos logits eleva al cuadrado la distancia a 1 de algo
  que no está en [0,1] y devuelve una cifra con pinta de métrica;
* presentar una etiqueta dura como si ordenara da un AUROC que no ha ordenado
  nada dentro de cada clase;
* devolver 0,0 cuando el denominador está vacío hace que «ningún predicho
  positivo» se lea como «predijo mal todos los positivos»;
* y ordenar por «mayor mejor» un intercepto de calibración corona precisamente
  al modelo que más sobreestima.

Las referencias numéricas (scikit-learn 1.8, numpy) se usan SOLO aquí y con
`skipUnless`: el core es stdlib puro y `dependencies = []` no se toca. Los casos
pequeños van calculados A MANO en el propio test, con la cuenta escrita, para
que la prueba no dependa de que la referencia esté instalada.
"""
from __future__ import annotations

import math
import random
import re
import unittest
from dataclasses import replace
from importlib import util
from pathlib import Path

from matrixai.estudio import EsquemaInvalido, MetricSpec, PredictionRecord
from matrixai.estudio.metricas import (
    CLIP_LOG_LOSS,
    INFORME_VERSION,
    METRICAS_DIFERIDAS,
    REGISTRO,
    UMBRAL_POR_DEFECTO,
    EntradaNoMedible,
    MetricaAplazada,
    MetricaDesconocida,
    Muestra,
    aplazamiento,
    aptitud,
    calcular,
    catalogo,
    digest_del_catalogo,
    direccion_de,
    distancia_al_ideal,
    es_mejor,
    especificacion,
    evaluar,
    matriz_de_confusion,
    metricas_aplicables,
    ordenar_por,
)
from matrixai.estudio.textos import motivo
from matrixai.training.dense_evaluator import result_from_predictions

_HAY_REFERENCIA = util.find_spec("sklearn") is not None and util.find_spec("numpy") is not None
MODULO = Path(__file__).resolve().parents[1] / "matrixai" / "estudio" / "metricas.py"

CLASES = ("no", "si")
POSITIVA = "si"


def _binaria(y, probabilidades=None, puntuaciones=None, **extra) -> Muestra:
    return Muestra.binaria(y, classes=CLASES, positive_label=POSITIVA,
                           probabilidades=probabilidades, puntuaciones=puntuaciones,
                           **extra)


def _valor(metric_id: str, muestra: Muestra, **opciones) -> float | None:
    return calcular(metric_id, muestra, **opciones).value


# ---------------------------------------------------------------------------
# El invariante 3: AP no es el área trapezoidal
# ---------------------------------------------------------------------------

class ApNoEsElAreaPrTrapezoidalTest(unittest.TestCase):
    """Dos identidades distintas, dos referencias distintas y **dos IDs**.

    EL FIXTURE. Cuatro filas, `y = (no, si, si, no)` con puntuaciones
    `(0,9 · 0,8 · 0,7 · 0,6)`. Los cuatro umbrales, de mayor a menor:

        umbral   TP  FP   recall   precisión
        0,9       0   1     0,0        0,0
        0,8       1   1     0,5        0,5
        0,7       2   1     1,0        2/3
        0,6       2   2     1,0        0,5

    * **AP** suma escalones: `0·0 + 0,5·0,5 + 0,5·(2/3) + 0 = 7/12 = 0,583333`.
      Cada sumando es una precisión REALMENTE observada en un umbral.
    * **Trapecio** une los puntos con rectas desde el ancla (0, 1):
      `0 + 0,5·(0+0,5)/2 + 0,5·(0,5+2/3)/2 = 5/12 = 0,416667`. Entre el primer
      umbral y el segundo supone precisiones intermedias —0,1, 0,2, 0,3…— que
      NINGÚN umbral de este modelo alcanza.

    Si esta prueba dejara de distinguirlos, el fixture no probaría nada: por eso
    lo primero que se comprueba es que los dos números NO coinciden.
    """

    def setUp(self):
        self.muestra = _binaria(["no", "si", "si", "no"],
                                probabilidades=[0.9, 0.8, 0.7, 0.6])

    def test_los_dos_numeros_son_distintos_y_son_los_de_la_cuenta_a_mano(self):
        ap = _valor("average_precision", self.muestra)
        trapecio = _valor("pr_auc_trapezoidal", self.muestra)
        self.assertNotAlmostEqual(ap, trapecio, places=6)
        self.assertAlmostEqual(ap, 7 / 12, places=15)
        self.assertAlmostEqual(trapecio, 5 / 12, places=15)
        # Y la diferencia no es un decimal escondido: es un 40 % del trapecio.
        self.assertGreater(ap - trapecio, 0.16)

    def test_son_dos_metricas_con_dos_ids_y_dos_fichas(self):
        self.assertIn("average_precision", REGISTRO)
        self.assertIn("pr_auc_trapezoidal", REGISTRO)
        ficha_ap = especificacion("average_precision")
        ficha_trapecio = especificacion("pr_auc_trapezoidal")
        self.assertNotEqual(ficha_ap.metric_id, ficha_trapecio.metric_id)
        self.assertNotEqual(ficha_ap.normalization, ficha_trapecio.normalization)
        self.assertNotEqual(ficha_ap.digest(), ficha_trapecio.digest())

    def test_la_ficha_de_ap_DICE_que_no_es_el_trapecio(self):
        """Un ID distinto no basta si nadie sabe en qué se diferencian: quien lee
        la ficha tiene que enterarse sin abrir el código."""
        self.assertIn("escalones", especificacion("average_precision").normalization)
        self.assertIn("rectas", especificacion("pr_auc_trapezoidal").normalization)

    @unittest.skipUnless(_HAY_REFERENCIA, "sin scikit-learn/numpy en el entorno")
    def test_cada_uno_coincide_con_SU_referencia_de_version_fijada(self):
        """AP contra `average_precision_score` y el trapecio contra
        `auc(recall, precision)`. Cruzar las referencias es justo el error que
        este corte existe para impedir."""
        import numpy as np
        from sklearn.metrics import auc, average_precision_score, precision_recall_curve

        y = np.array([0, 1, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6])
        precision, recall, _ = precision_recall_curve(y, s)
        self.assertAlmostEqual(_valor("average_precision", self.muestra),
                               float(average_precision_score(y, s)), places=15)
        self.assertAlmostEqual(_valor("pr_auc_trapezoidal", self.muestra),
                               float(auc(recall, precision)), places=15)


# ---------------------------------------------------------------------------
# El invariante 2: qué admite una puntuación y qué exige una probabilidad
# ---------------------------------------------------------------------------

class UnaPuntuacionOrdenaYUnaProbabilidadAdemasMideTest(unittest.TestCase):
    """AUROC y AP se conforman con un ORDEN; log-loss y Brier no.

    El caso del criterio 3: unos logits `(2,2 · 1,4 · 0,3 · -1,7)` ordenan
    perfectamente y **no son probabilidades**. Elevar al cuadrado su distancia a
    1 daría un número con pinta de Brier que no mide nada, y un Brier de 3,24 en
    un informe no se distingue a simple vista de uno mal calibrado.
    """

    def setUp(self):
        self.logits = _binaria(["no", "si", "si", "no"],
                               puntuaciones=[2.2, 1.4, 0.3, -1.7])
        self.probabilidades = _binaria(["no", "si", "si", "no"],
                                       probabilidades=[0.9, 0.8, 0.7, 0.6])
        self.etiquetas = Muestra.binaria(["no", "si", "si", "no"], classes=CLASES,
                                         positive_label=POSITIVA,
                                         predicciones=["si", "si", "no", "no"])

    def test_el_auroc_sale_con_logits(self):
        self.assertAlmostEqual(_valor("auroc", self.logits), 0.5, places=15)

    def test_el_brier_sobre_logits_se_RECHAZA_no_se_calcula(self):
        with self.assertRaises(EntradaNoMedible) as e:
            calcular("brier_score", self.logits)
        self.assertEqual(e.exception.bilingue,
                         motivo("no_son_probabilidades", campo="brier_score"))

    def test_la_log_loss_sobre_logits_tambien(self):
        with self.assertRaises(EntradaNoMedible):
            calcular("log_loss", self.logits)

    def test_una_etiqueta_dura_no_se_presenta_como_probabilidad(self):
        """Con la clase predicha y nada más, AUROC no se calcula: dentro de cada
        clase no hay orden ninguno."""
        with self.assertRaises(EntradaNoMedible) as e:
            calcular("auroc", self.etiquetas)
        self.assertEqual(e.exception.bilingue,
                         motivo("etiquetas_duras_no_ordenan", campo="auroc"))

    def test_la_lista_de_aplicables_LO_DICE_antes_de_reventar(self):
        aplicables = metricas_aplicables(self.logits, umbral=1.0)
        self.assertIn("auroc", aplicables)
        self.assertIn("average_precision", aplicables)
        self.assertNotIn("brier_score", aplicables)
        self.assertNotIn("log_loss", aplicables)

    def test_una_probabilidad_SI_vale_como_puntuacion_ordenada(self):
        """Al revés que lo anterior: la muestra que solo trae la distribución
        completa sirve para AUROC. El grupo alternativo existe por esto."""
        aplicables = metricas_aplicables(self.probabilidades)
        for metric_id in ("auroc", "average_precision", "log_loss", "brier_score"):
            self.assertIn(metric_id, aplicables)

    def test_sin_umbral_y_sin_probabilidades_no_hay_matriz_ni_metricas_del_umbral(self):
        """0,5 sobre una escala que nadie ha acotado corta en un sitio arbitrario."""
        aplicables = metricas_aplicables(self.logits)
        for metric_id in ("sensitivity", "specificity", "ppv", "npv", "accuracy"):
            self.assertNotIn(metric_id, aplicables)
        with self.assertRaises(EntradaNoMedible) as e:
            calcular("ppv", self.logits)
        self.assertEqual(e.exception.bilingue, motivo("umbral_por_omision_en_puntuacion"))
        self.assertAlmostEqual(_valor("ppv", self.logits, umbral=1.0), 0.5, places=15)


# ---------------------------------------------------------------------------
# Empates y orden de clases
# ---------------------------------------------------------------------------

class LosEmpatesYElOrdenDeLasClasesNoCambianElNumeroTest(unittest.TestCase):
    """Los empates comparten punto y la clase positiva se declara, no se adivina.

    EL FIXTURE DE EMPATES. `y = (no, si, si, no, si)` con puntuaciones
    `(0,5 · 0,5 · 0,9 · 0,2 · 0,5)`: tres positivos, dos negativos, seis pares.
    A mano: el positivo de 0,9 gana los dos pares (2); cada positivo de 0,5
    empata con el negativo de 0,5 —medio punto— y gana al de 0,2 (1,5 + 1,5).
    Total 5 de 6 → **AUROC = 0,833333**. Contar el empate como victoria daría
    1,0 y contarlo como derrota 0,666667: los tres números son creíbles y solo
    uno es el correcto.
    """

    def setUp(self):
        self.empates = _binaria(["no", "si", "si", "no", "si"],
                                probabilidades=[0.5, 0.5, 0.9, 0.2, 0.5])

    def test_un_empate_vale_medio_par_ni_uno_ni_cero(self):
        self.assertAlmostEqual(_valor("auroc", self.empates), 5 / 6, places=15)

    def test_barajar_las_filas_no_mueve_el_auroc(self):
        """Si los empates se partieran fila a fila, el número dependería del
        orden en que llegaron los datos."""
        indices = [3, 0, 4, 2, 1]
        y = [self.empates.y_true[i] for i in indices]
        p = [self.empates.probabilidad_del_positivo[i] for i in indices]
        self.assertAlmostEqual(_valor("auroc", _binaria(y, probabilidades=p)),
                               _valor("auroc", self.empates), places=15)

    def test_declarar_las_clases_al_reves_da_el_MISMO_numero(self):
        """`classes=(si, no)` cambia el orden de las columnas, no el positivo. Un
        AUROC que cambiara aquí estaría leyendo la columna equivocada."""
        al_reves = Muestra.binaria(["no", "si", "si", "no", "si"], classes=("si", "no"),
                                   positive_label=POSITIVA,
                                   probabilidades=[0.5, 0.5, 0.9, 0.2, 0.5])
        self.assertEqual(al_reves.classes, ("si", "no"))
        self.assertEqual(al_reves.probabilities[0], (0.5, 0.5))
        for metric_id in ("auroc", "average_precision", "brier_score", "log_loss"):
            with self.subTest(metric_id=metric_id):
                self.assertAlmostEqual(_valor(metric_id, al_reves),
                                       _valor(metric_id, self.empates), places=15)

    def test_elegir_el_otro_positivo_da_otro_numero(self):
        """Y esto es lo que hace que la prueba anterior no sea vacía: cambiar la
        CLASE POSITIVA sí tiene que cambiar el resultado."""
        otro = Muestra.binaria(["no", "si", "si", "no", "si"], classes=CLASES,
                               positive_label="no",
                               probabilidades=[0.5, 0.5, 0.9, 0.2, 0.5])
        # A mano, con «no» de positiva: dos positivos (0,5 y 0,2) contra tres
        # negativos (0,5 · 0,9 · 0,5). El de 0,5 empata dos veces (1,0) y pierde
        # contra 0,9; el de 0,2 lo pierde todo. 1 de 6.
        self.assertAlmostEqual(_valor("auroc", otro), 1 / 6, places=15)
        self.assertNotAlmostEqual(_valor("auroc", otro),
                                  _valor("auroc", self.empates), places=6)

    @unittest.skipUnless(_HAY_REFERENCIA, "sin scikit-learn/numpy en el entorno")
    def test_los_empates_coinciden_con_la_referencia(self):
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = np.array([0, 1, 1, 0, 1])
        s = np.array([0.5, 0.5, 0.9, 0.2, 0.5])
        self.assertAlmostEqual(_valor("auroc", self.empates),
                               float(roc_auc_score(y, s)), places=15)
        self.assertAlmostEqual(_valor("average_precision", self.empates),
                               float(average_precision_score(y, s)), places=15)


# ---------------------------------------------------------------------------
# Ranking contra magnitud
# ---------------------------------------------------------------------------

class ElRankingMueveElAurocYLaMagnitudMueveLaLogLossTest(unittest.TestCase):
    """Las dos mitades del criterio 5, y las dos hacen falta.

    Base: `y = (si, si, no, no)` con `p = (0,9 · 0,6 · 0,4 · 0,1)` — los dos
    positivos por encima de los dos negativos, AUROC = 1,0.

    * **Cambiar el ORDEN** de la segunda y la tercera fila deja `(0,9 · 0,4 ·
      0,6 · 0,1)`: ahora un negativo adelanta a un positivo, se pierde uno de los
      cuatro pares y el AUROC baja a **0,75**. Cifra conocida de antemano.
    * **Cambiar solo la MAGNITUD** —comprimir a `(0,55 · 0,52 · 0,48 · 0,45)`—
      conserva el orden entero: el AUROC sigue siendo **1,0** y la log-loss sube
      de 0,2899 a 0,6444, porque el modelo acierta igual y afirma mucho menos.
      Un producto que solo mirara el AUROC no vería la diferencia.
    """

    def setUp(self):
        self.base = _binaria(["si", "si", "no", "no"],
                             probabilidades=[0.9, 0.6, 0.4, 0.1])
        self.reordenada = _binaria(["si", "si", "no", "no"],
                                   probabilidades=[0.9, 0.4, 0.6, 0.1])
        self.comprimida = _binaria(["si", "si", "no", "no"],
                                   probabilidades=[0.55, 0.52, 0.48, 0.45])

    def test_cambiar_el_ranking_baja_el_auroc_a_la_cifra_prevista(self):
        self.assertAlmostEqual(_valor("auroc", self.base), 1.0, places=15)
        self.assertAlmostEqual(_valor("auroc", self.reordenada), 0.75, places=15)

    def test_cambiar_solo_la_magnitud_deja_el_auroc_IGUAL(self):
        self.assertAlmostEqual(_valor("auroc", self.comprimida), 1.0, places=15)

    def test_y_esa_misma_magnitud_SI_mueve_la_log_loss_y_el_brier(self):
        a_mano = -(math.log(0.9) + math.log(0.6) + math.log(0.6)
                   + math.log(0.9)) / 4
        self.assertAlmostEqual(_valor("log_loss", self.base), a_mano, places=15)
        self.assertGreater(_valor("log_loss", self.comprimida),
                           _valor("log_loss", self.base))
        self.assertGreater(_valor("brier_score", self.comprimida),
                           _valor("brier_score", self.base))


# ---------------------------------------------------------------------------
# Lo imposible se rechaza; lo indefinido se dice
# ---------------------------------------------------------------------------

class LoQueNoSePuedeCalcularSeDiceYNoSeInventaUnCeroTest(unittest.TestCase):
    """La frontera del corte: **error** cuando alguien cableó mal, **indefinición
    motivada** cuando son los datos. Ninguna de las dos es un cero."""

    def test_un_nan_o_un_infinito_en_una_puntuacion_es_un_RECHAZO(self):
        for valor in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(valor=valor):
                with self.assertRaises(EsquemaInvalido):
                    _binaria(["no", "si"], puntuaciones=[0.5, valor])

    def test_una_probabilidad_fuera_de_rango_es_un_RECHAZO(self):
        with self.assertRaises(EsquemaInvalido):
            _binaria(["no", "si"], probabilidades=[0.5, 1.5])

    def test_una_distribucion_que_no_suma_uno_es_un_RECHAZO(self):
        with self.assertRaises(EsquemaInvalido) as e:
            Muestra.multiclase(["a", "b"], classes=("a", "b", "c"),
                               probabilidades=[[0.5, 0.3, 0.1], [0.2, 0.5, 0.3]])
        self.assertIn("0.9", e.exception.en)

    def test_columnas_desalineadas_con_las_clases_es_un_RECHAZO(self):
        with self.assertRaises(EsquemaInvalido):
            Muestra.multiclase(["a", "b"], classes=("a", "b", "c"),
                               probabilidades=[[0.5, 0.5], [0.5, 0.5]])

    def test_una_sola_clase_presente_da_INDEFINIDA_con_motivo_no_un_0_5(self):
        """El 0,5 de un AUROC indefinido es la trampa: se lee como «no
        discrimina» cuando lo que pasa es que no hay nada que discriminar."""
        una_clase = _binaria(["no", "no", "no"], probabilidades=[0.9, 0.4, 0.1])
        for metric_id in ("auroc", "average_precision", "pr_auc_trapezoidal"):
            with self.subTest(metric_id=metric_id):
                resultado = calcular(metric_id, una_clase)
                self.assertIsNone(resultado.value)
                self.assertEqual(resultado.undefined_reason,
                                 motivo("una_sola_clase", campo=metric_id, valor="no"))

    def test_un_denominador_vacio_da_INDEFINIDA_no_un_cero(self):
        """Ningún predicho positivo no es «acertó cero de los positivos»."""
        todo_negativo = _binaria(["no", "si", "no"], probabilidades=[0.1, 0.2, 0.3])
        ppv = calcular("ppv", todo_negativo)
        self.assertIsNone(ppv.value)
        self.assertEqual(ppv.undefined_reason,
                         motivo("division_por_cero_en_metrica", campo="ppv",
                                valor="el numero de predichos positivos"))
        # Y la que SÍ tiene denominador sigue dando su número.
        self.assertAlmostEqual(calcular("npv", todo_negativo).value, 2 / 3, places=15)

    def test_una_regresion_sin_varianza_da_INDEFINIDA_donde_el_historico_pone_cero(self):
        """La única divergencia deliberada con `dense_evaluator`, y está escrita.
        Un R² de 0,0 se lee como «no explica nada»; aquí no hay nada que explicar."""
        plana = Muestra.regresion([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])
        resultado = calcular("r2", plana)
        self.assertIsNone(resultado.value)
        self.assertEqual(resultado.undefined_reason,
                         motivo("division_por_cero_en_metrica", campo="r2",
                                valor="la varianza del objetivo"))
        historico = result_from_predictions([[4.0], [5.0], [6.0]],
                                            [[5.0], [5.0], [5.0]], "mse")
        self.assertEqual(historico.r2, 0.0)

    def test_cero_filas_no_dan_un_cero(self):
        vacia = _binaria([], probabilidades=[])
        self.assertEqual(vacia.n, 0)
        resultado = calcular("auroc", vacia)
        self.assertIsNone(resultado.value)
        self.assertEqual(resultado.undefined_reason,
                         motivo("sin_observaciones", campo="auroc"))
        self.assertEqual(resultado.n_observations, 0)

    def test_una_probabilidad_de_0_o_de_1_no_manda_la_log_loss_al_infinito(self):
        """Sin clipping, `ln(0)` se va a `-inf`, y un infinito no es JSON: el
        informe entero dejaría de poder canonicalizarse, firmarse y compararse.
        El recorte va DECLARADO en `CLIP_LOG_LOSS`, no escondido."""
        extrema = _binaria(["si", "no"], probabilidades=[0.0, 1.0])
        valor = _valor("log_loss", extrema)
        self.assertTrue(math.isfinite(valor))
        self.assertAlmostEqual(valor, -math.log(CLIP_LOG_LOSS), places=12)
        perfecta = _binaria(["si", "no"], probabilidades=[1.0, 0.0])
        self.assertAlmostEqual(_valor("log_loss", perfecta),
                               -math.log(1.0 - CLIP_LOG_LOSS), places=15)

    def test_una_metrica_indefinida_no_es_la_peor_es_que_no_se_sabe(self):
        with self.assertRaises(EntradaNoMedible):
            es_mejor("auroc", 0.8, None)

    def test_los_pesos_no_se_aplican_en_silencio_a_un_recuento(self):
        """Una matriz de confusión con pesos ya no cuenta filas. O se rechaza, o
        el `n` del informe deja de significar lo que dice."""
        con_pesos = _binaria(["no", "si"], probabilidades=[0.2, 0.8],
                             pesos=[1.0, 3.0])
        with self.assertRaises(EntradaNoMedible) as e:
            matriz_de_confusion(con_pesos)
        self.assertEqual(e.exception.bilingue,
                         motivo("pesos_no_admitidos", campo="confusion_matrix"))
        with self.assertRaises(EntradaNoMedible):
            calcular("accuracy", con_pesos)
        # Las que sí saben pesar siguen dando número.
        self.assertIsNotNone(calcular("brier_score", con_pesos).value)


# ---------------------------------------------------------------------------
# El invariante 4: no todo se ordena por «mayor mejor»
# ---------------------------------------------------------------------------

class NoTodoSeOrdenaPorMayorMejorTest(unittest.TestCase):
    """`calibration_in_the_large` vale idealmente 0 y desviarse a cualquiera de
    los dos lados es peor. Ordenarla por «mayor mejor» corona al modelo que más
    sobreestima el riesgo, que es exactamente el que menos conviene."""

    def test_la_ficha_declara_valor_ideal_y_NO_direccion(self):
        ficha = especificacion("calibration_in_the_large")
        self.assertIsNone(ficha.direction)
        self.assertEqual(ficha.ideal_value, 0.0)
        self.assertIsNone(direccion_de("calibration_in_the_large"))

    def test_el_esquema_se_NIEGA_a_llevar_direccion_e_ideal_a_la_vez(self):
        """La negativa es del 104-C0 y aquí se apoya en ella: sin esa negativa,
        alguien pondría las dos y el orden dependería de cuál mirara el llamante."""
        with self.assertRaises(EsquemaInvalido):
            MetricSpec(metric_id="inventada", formula_version="1.0.0",
                       estimand="fixed_model_on_population",
                       direction="higher_is_better", ideal_value=0.0)

    def test_desviarse_a_los_dos_lados_es_igual_de_malo(self):
        self.assertEqual(distancia_al_ideal("calibration_in_the_large", 0.2),
                         distancia_al_ideal("calibration_in_the_large", -0.2))
        self.assertFalse(es_mejor("calibration_in_the_large", 0.2, -0.2))
        self.assertFalse(es_mejor("calibration_in_the_large", -0.2, 0.2))
        self.assertTrue(es_mejor("calibration_in_the_large", 0.05, -0.2))

    def test_mayor_mejor_coronaria_al_que_mas_sobreestima(self):
        """La refutación: si `ordenar_por` usara «mayor mejor», el +0,30 saldría
        primero y el -0,01 último. Con el ideal, gana el -0,01."""
        valores = [0.30, -0.01, 0.12]
        self.assertEqual(ordenar_por("calibration_in_the_large", valores),
                         [-0.01, 0.12, 0.30])
        self.assertNotEqual(ordenar_por("calibration_in_the_large", valores),
                            sorted(valores, reverse=True))

    def test_las_de_calidad_si_tienen_direccion_y_cada_una_la_suya(self):
        self.assertEqual(direccion_de("auroc"), "higher_is_better")
        self.assertEqual(direccion_de("log_loss"), "lower_is_better")
        self.assertTrue(es_mejor("auroc", 0.9, 0.7))
        self.assertTrue(es_mejor("log_loss", 0.3, 0.7))
        self.assertEqual(ordenar_por("rmse", [3.0, 1.0, 2.0]), [1.0, 2.0, 3.0])

    def test_la_calibracion_en_grande_es_la_cuenta_a_mano(self):
        """Media de las probabilidades menos prevalencia: `(0,9+0,8+0,7+0,6)/4 -
        2/4 = 0,75 - 0,5 = 0,25`. El modelo dice 75 % y pasa el 50 %."""
        muestra = _binaria(["no", "si", "si", "no"],
                           probabilidades=[0.9, 0.8, 0.7, 0.6])
        self.assertAlmostEqual(_valor("calibration_in_the_large", muestra), 0.25,
                               places=15)

    def test_y_NO_se_presenta_como_un_diagnostico_de_calibracion(self):
        ficha = especificacion("calibration_in_the_large")
        self.assertIn("no es la pendiente", ficha.normalization)
        self.assertNotIn("brier_multiclass", REGISTRO)


# ---------------------------------------------------------------------------
# Los campos históricos
# ---------------------------------------------------------------------------

class LosCamposHistoricosNoCambianDeValorNiDeNombreTest(unittest.TestCase):
    """Mismos datos, mismos números que `dense_evaluator` publica hoy.

    Es el criterio 6 y no es una formalidad: `accuracy`, `macro_f1`, la matriz de
    confusión, `mae`, `rmse` y `r2` viajan en respuestas del API y en informes ya
    firmados. El macro-F1 histórico redondea a seis decimales cada F1 y también
    la media; reproducirlo es la diferencia entre el mismo campo y un campo con
    el mismo nombre y otro valor.
    """

    def test_multiclase_argmax_da_los_mismos_numeros_y_la_misma_matriz(self):
        clases = ["baja", "media", "alta"]
        probabilidades = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6],
                          [0.3, 0.4, 0.3], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]
        reales = ["baja", "media", "alta", "media", "alta", "alta"]
        objetivos = [[1.0 if c == y else 0.0 for c in clases] for y in reales]
        historico = result_from_predictions(probabilidades, objetivos,
                                            "cross_entropy", clases)
        muestra = Muestra.multiclase(reales, classes=tuple(clases),
                                     probabilidades=probabilidades)

        self.assertAlmostEqual(_valor("accuracy", muestra), historico.accuracy, places=15)
        self.assertEqual(_valor("macro_f1", muestra), historico.macro_f1)
        self.assertEqual(matriz_de_confusion(muestra).historica(),
                         historico.confusion_matrix)
        # Y el histórico no es trivialmente perfecto: si lo fuera, coincidir con
        # él no diría nada.
        self.assertLess(historico.accuracy, 1.0)
        self.assertGreater(historico.accuracy, 0.0)

    def test_la_log_loss_multiclase_es_la_cuenta_a_mano(self):
        clases = ("a", "b", "c")
        probabilidades = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        muestra = Muestra.multiclase(["a", "b", "c"], classes=clases,
                                     probabilidades=probabilidades)
        a_mano = -(math.log(0.7) + math.log(0.8) + math.log(0.6)) / 3
        self.assertAlmostEqual(_valor("log_loss", muestra), a_mano, places=15)

    def test_binaria_al_umbral_da_los_mismos_numeros_que_el_historico(self):
        probabilidades = [0.9, 0.8, 0.7, 0.6, 0.4, 0.1]
        reales = ["no", "si", "si", "no", "si", "no"]
        historico = result_from_predictions(
            [[p] for p in probabilidades],
            [[1.0 if y == "si" else 0.0] for y in reales],
            "binary_cross_entropy", ["no", "si"])
        muestra = _binaria(reales, probabilidades=probabilidades)
        self.assertAlmostEqual(_valor("accuracy", muestra), historico.accuracy,
                               places=15)
        self.assertEqual(_valor("macro_f1", muestra), historico.macro_f1)
        self.assertEqual(matriz_de_confusion(muestra).historica(),
                         historico.confusion_matrix)

    def test_el_umbral_binario_corta_con_MAYOR_O_IGUAL_como_el_historico(self):
        """La fila que cae justo en 0,5. Con `>` cambiaría de lado y con ella la
        matriz, la accuracy y el macro-F1 de un informe ya publicado."""
        muestra = _binaria(["si", "no"], probabilidades=[0.5, 0.5])
        matriz = matriz_de_confusion(muestra)
        self.assertEqual(matriz.counts["si"]["si"], 1)
        self.assertEqual(matriz.counts["no"]["si"], 1)
        historico = result_from_predictions([[0.5], [0.5]], [[1.0], [0.0]],
                                            "binary_cross_entropy", ["no", "si"])
        self.assertEqual(matriz.historica(), historico.confusion_matrix)

    def test_el_argmax_se_queda_con_el_PRIMERO_cuando_empatan(self):
        muestra = Muestra.multiclase(["a"], classes=("a", "b", "c"),
                                     probabilidades=[[0.4, 0.4, 0.2]])
        self.assertEqual(matriz_de_confusion(muestra).counts["a"]["a"], 1)

    def test_regresion_da_los_mismos_mae_rmse_y_r2(self):
        reales = [3.0, -0.5, 2.0, 7.0]
        predichos = [2.5, 0.0, 2.0, 8.0]
        historico = result_from_predictions([[p] for p in predichos],
                                            [[y] for y in reales], "mse")
        muestra = Muestra.regresion(reales, predichos)
        self.assertAlmostEqual(_valor("mae", muestra), historico.mae, places=15)
        self.assertAlmostEqual(_valor("rmse", muestra), historico.rmse, places=15)
        self.assertAlmostEqual(_valor("r2", muestra), historico.r2, places=15)
        self.assertAlmostEqual(historico.mae, 0.5, places=15)

    def test_los_ids_historicos_siguen_llamandose_igual(self):
        for metric_id in ("accuracy", "macro_f1", "mae", "rmse", "r2"):
            self.assertIn(metric_id, REGISTRO)

    def test_el_informe_ampliado_es_un_documento_NUEVO_con_version_y_digest(self):
        """No sustituye al histórico: lo enmarca. El viejo no tiene ni versión ni
        digest, y por eso el nuevo los lleva."""
        muestra = _binaria(["no", "si", "si", "no"],
                           probabilidades=[0.9, 0.8, 0.7, 0.6])
        informe = evaluar(muestra)
        self.assertEqual(informe.report_version, INFORME_VERSION)
        self.assertEqual(len(informe.digest()), 64)
        self.assertEqual(informe.a_json()["schema"], "matrixai.estudio.metric_report")
        historico = result_from_predictions(
            [[0.9], [0.8], [0.7], [0.6]], [[0.0], [1.0], [1.0], [0.0]],
            "binary_cross_entropy", ["no", "si"]).to_dict()
        self.assertNotIn("report_version", historico)
        self.assertNotIn("catalog_digest", historico)


# ---------------------------------------------------------------------------
# El informe: las reglas viajan con los números
# ---------------------------------------------------------------------------

class ElInformeDeclaraCONQUEReglasSeMidioTest(unittest.TestCase):
    """Un número sin su regla no se puede volver a calcular ni comparar. El
    contrato pide clase positiva, orden de clases, clipping, normalización del
    Brier, pesos y versión de fórmula: la mitad va en la ficha y la otra mitad en
    el informe, que es quien conoce la muestra."""

    def setUp(self):
        self.informe = evaluar(_binaria(["no", "si", "si", "no"],
                                        probabilidades=[0.9, 0.8, 0.7, 0.6]))

    def test_publica_el_orden_de_clases_la_positiva_el_umbral_y_la_escala(self):
        self.assertEqual(self.informe.classes, CLASES)
        self.assertEqual(self.informe.positive_label, POSITIVA)
        self.assertEqual(self.informe.threshold, UMBRAL_POR_DEFECTO)
        self.assertEqual(self.informe.decision_rule, "threshold")
        self.assertEqual(self.informe.decision_scale, "calibrated_probability")
        self.assertIs(self.informe.weighted, False)

    def test_publica_el_clipping_de_la_log_loss_y_es_el_de_la_referencia(self):
        self.assertEqual(self.informe.log_loss_clip, CLIP_LOG_LOSS)
        self.assertIn(repr(CLIP_LOG_LOSS), especificacion("log_loss").normalization)

    def test_publica_la_normalizacion_del_brier_y_dice_que_no_es_el_multicategoria(self):
        normalizacion = especificacion("brier_score").normalization
        self.assertIn("[0,1]", normalizacion)
        self.assertIn("doble", normalizacion)

    def test_cada_numero_viaja_con_SU_version_de_formula(self):
        for valor in self.informe.metrics:
            with self.subTest(metric_id=valor.metric_id):
                self.assertEqual(valor.formula_version,
                                 especificacion(valor.metric_id).formula_version)
                self.assertEqual(valor.n_observations, 4)

    def test_la_clase_positiva_del_informe_es_la_de_la_MUESTRA_no_la_de_la_ficha(self):
        """La ficha del catálogo NO declara clase positiva: `requires` ya dice
        que la trae la muestra, y «qué es el AUROC» no conoce ninguna. La que se
        publica es la de quien mide.

        Esta prueba exigía antes que la ficha llevara un **marcador de texto**
        (`"<positive_label of the sample>"`), puesto para satisfacer una regla
        del 104-C0 que se revisó el 2026-09-05: era un valor fabricado en un
        campo tipado como etiqueta de clase. La intención de la prueba —que el
        informe publique la clase REAL y no la de la ficha— no cambia.
        """
        self.assertIsNone(especificacion("auroc").positive_label)
        self.assertEqual(self.informe.positive_label, "si")
        self.assertIn(self.informe.positive_label, self.informe.classes)

    def test_una_metrica_que_no_se_midio_aparece_DICIENDO_por_que(self):
        informe = evaluar(_binaria(["no", "si", "si", "no"],
                                   puntuaciones=[2.2, 1.4, 0.3, -1.7]))
        no_medidas = dict(informe.not_applicable)
        self.assertIn("brier_score", no_medidas)
        self.assertEqual(no_medidas["brier_score"],
                         motivo("no_son_probabilidades", campo="brier_score"))
        self.assertNotIn("brier_score", [m.metric_id for m in informe.metrics])

    def test_pedir_una_metrica_imposible_POR_SU_NOMBRE_revienta(self):
        """En automático se listan las que no aplican; nombrada a dedo, no: quien
        la nombró se equivocó y tiene que enterarse."""
        logits = _binaria(["no", "si"], puntuaciones=[2.0, -1.0])
        with self.assertRaises(EntradaNoMedible):
            evaluar(logits, ["auroc", "brier_score"])

    def test_el_digest_del_informe_cambia_si_cambia_un_numero(self):
        otro = evaluar(_binaria(["no", "si", "si", "no"],
                                probabilidades=[0.9, 0.8, 0.7, 0.61]))
        self.assertNotEqual(self.informe.digest(), otro.digest())

    def test_el_digest_del_catalogo_cambia_si_cambia_una_formula(self):
        """Es lo que permite a `verify` decir «esto se midió con otro catálogo»
        en vez de comparar números medidos con reglas distintas."""
        antes = digest_del_catalogo()
        self.assertEqual(len(antes), 64)
        self.assertEqual(antes, digest_del_catalogo())
        from matrixai.estudio import metricas as modulo
        original = modulo._CATALOGO
        try:
            subida = replace(original[0].spec, formula_version="9.9.9")
            modulo._CATALOGO = (replace(original[0], spec=subida),) + original[1:]
            self.assertNotEqual(digest_del_catalogo(), antes)
        finally:
            modulo._CATALOGO = original
        self.assertEqual(digest_del_catalogo(), antes)

    def test_el_informe_de_una_muestra_con_pesos_lo_DICE(self):
        informe = evaluar(_binaria(["no", "si", "si", "no"],
                                   probabilidades=[0.9, 0.8, 0.7, 0.6],
                                   pesos=[1.0, 2.0, 1.0, 3.0]))
        self.assertIs(informe.weighted, True)
        self.assertIn("confusion_matrix", dict(informe.not_applicable))
        self.assertIn("accuracy", dict(informe.not_applicable))

    def test_el_informe_cuenta_unidades_de_remuestreo_cuando_las_hay(self):
        """Filas y pacientes no son intercambiables: si nadie declaró la unidad,
        el informe dice `None`, no el número de filas."""
        sin_unidad = evaluar(_binaria(["no", "si"], probabilidades=[0.2, 0.8]))
        self.assertIsNone(sin_unidad.n_units)
        con_unidad = evaluar(_binaria(["no", "si", "si"],
                                      probabilidades=[0.2, 0.8, 0.7],
                                      unidades=["p1", "p1", "p2"]))
        self.assertEqual(con_unidad.n_observations, 3)
        self.assertEqual(con_unidad.n_units, 2)


# ---------------------------------------------------------------------------
# El registro es único y cerrado
# ---------------------------------------------------------------------------

class ElRegistroEsUnicoYCerradoTest(unittest.TestCase):
    """Invariante 1: una sola fórmula por métrica, y lo que no está no se
    improvisa. Dos sitios calculando el mismo AUROC acaban divergiendo."""

    def test_una_metrica_que_no_esta_no_se_inventa(self):
        with self.assertRaises(MetricaDesconocida):
            calcular("f_beta_a_ojo", _binaria(["no", "si"], probabilidades=[0.1, 0.9]))

    def test_lo_diferido_se_distingue_de_lo_desconocido_y_trae_su_motivo(self):
        """`mape` no es que no exista: es que alguien la miró, decidió diferirla y
        escribió por qué. Devolver `MetricaDesconocida` perdería esa decisión."""
        for metric_id in ("mape", "brier_multiclass"):
            with self.subTest(metric_id=metric_id):
                with self.assertRaises(MetricaAplazada) as e:
                    calcular(metric_id, Muestra.regresion([1.0, 2.0], [1.5, 2.5]))
                self.assertEqual(e.exception.bilingue, aplazamiento(metric_id))
                self.assertNotIn(metric_id, REGISTRO)
        self.assertIn("cero", aplazamiento("mape")["es"])
        self.assertIn("calibración multiclase", aplazamiento("brier_multiclass")["es"])

    def test_cada_id_aparece_UNA_vez_y_con_una_sola_ficha(self):
        ids = [spec.metric_id for spec in catalogo()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(REGISTRO))

    def test_ninguna_ficha_se_queda_sin_dirección_ni_ideal(self):
        for spec in catalogo():
            with self.subTest(metric_id=spec.metric_id):
                self.assertTrue(spec.direction is not None
                                or spec.ideal_value is not None
                                or spec.ideal_range is not None)
                self.assertEqual(spec.estimand, "fixed_model_on_population")
                self.assertTrue(spec.normalization)

    def test_la_metrica_de_otra_tarea_se_rechaza_con_las_suyas_escritas(self):
        regresion = Muestra.regresion([1.0, 2.0], [1.5, 2.5])
        with self.assertRaises(EntradaNoMedible) as e:
            calcular("auroc", regresion)
        self.assertIn("binary_classification", e.exception.en)

    def test_el_modulo_es_stdlib_puro(self):
        """Un `import numpy` aquí rompería la instalación de quien no lo tenga, y
        no lo diría hasta ejecutarse."""
        prohibidos = re.compile(
            r"^\s*(?:import|from)\s+(numpy|scipy|sklearn|pandas|torch|onnx)\b",
            re.MULTILINE)
        self.assertIsNone(prohibidos.search(MODULO.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# El puente con el 104-C0
# ---------------------------------------------------------------------------

class ElPuenteConElRegistroDePrediccionesTest(unittest.TestCase):
    """Lo que sale del `PredictionRecord` del 104-C0 tiene que poder medirse sin
    que nadie rellene nada por el camino."""

    def _registro(self, row_id, y, p, **extra):
        return PredictionRecord(row_id=row_id, candidate="c1", y_true=y,
                                probabilities=(1.0 - p, p), classes=CLASES, **extra)

    def test_una_muestra_sale_de_los_registros_y_da_los_mismos_numeros(self):
        registros = [self._registro("r1", "no", 0.9), self._registro("r2", "si", 0.8),
                     self._registro("r3", "si", 0.7), self._registro("r4", "no", 0.6)]
        muestra = Muestra.desde_registros(registros, task="binary_classification",
                                          positive_label=POSITIVA)
        self.assertAlmostEqual(_valor("average_precision", muestra), 7 / 12, places=15)
        self.assertEqual(muestra.classes, CLASES)

    def test_dos_candidatos_en_la_misma_medicion_se_RECHAZAN(self):
        """Mezclar dos modelos no mide ninguno, y el número saldría con pinta de
        bueno."""
        registros = [self._registro("r1", "no", 0.9),
                     PredictionRecord(row_id="r2", candidate="c2", y_true="si",
                                      probabilities=(0.2, 0.8), classes=CLASES)]
        with self.assertRaises(EntradaNoMedible) as e:
            Muestra.desde_registros(registros, task="binary_classification",
                                    positive_label=POSITIVA)
        self.assertIn("c1", e.exception.en)
        self.assertIn("c2", e.exception.en)

    def test_un_registro_sin_verdad_no_entra_en_una_muestra_de_evaluacion(self):
        registros = [self._registro("r1", "no", 0.9),
                     PredictionRecord(row_id="r2", candidate="c1",
                                      probabilities=(0.2, 0.8), classes=CLASES)]
        with self.assertRaises(EntradaNoMedible) as e:
            Muestra.desde_registros(registros, task="binary_classification",
                                    positive_label=POSITIVA)
        self.assertIn("r2", e.exception.en)

    def test_media_columna_no_es_una_columna(self):
        registros = [self._registro("r1", "no", 0.9, weight=2.0),
                     self._registro("r2", "si", 0.8)]
        with self.assertRaises(EntradaNoMedible) as e:
            Muestra.desde_registros(registros, task="binary_classification",
                                    positive_label=POSITIVA)
        self.assertIn("weight", e.exception.en)

    def test_dos_ordenes_de_clases_distintos_no_se_mezclan(self):
        registros = [self._registro("r1", "no", 0.9),
                     PredictionRecord(row_id="r2", candidate="c1", y_true="si",
                                      probabilities=(0.8, 0.2), classes=("si", "no"))]
        with self.assertRaises(EntradaNoMedible):
            Muestra.desde_registros(registros, task="binary_classification",
                                    positive_label=POSITIVA)

    def test_dos_columnas_de_puntuacion_se_leen_como_positivo_menos_negativo(self):
        """Tomar solo la columna del positivo ordenaría MAL unos logits: la
        softmax es monótona en `z_pos - z_neg`, no en `z_pos`."""
        registros = [
            PredictionRecord(row_id="r1", candidate="c1", y_true="no",
                             scores=(0.0, 1.0), classes=CLASES),
            PredictionRecord(row_id="r2", candidate="c1", y_true="si",
                             scores=(-3.0, 0.5), classes=CLASES),
        ]
        muestra = Muestra.desde_registros(registros, task="binary_classification",
                                          positive_label=POSITIVA)
        self.assertEqual(muestra.scores, (1.0, 3.5))
        self.assertEqual(muestra.score_rule, "positivo_menos_negativo")
        # Con la columna suelta, `1.0 > 0.5` pondría al negativo por delante y el
        # AUROC saldría 0,0 en vez de 1,0.
        self.assertAlmostEqual(_valor("auroc", muestra), 1.0, places=15)

    def test_la_aptitud_del_104_se_ENSANCHA_no_se_contradice(self):
        """Un registro que solo trae la distribución completa SÍ vale para AUROC
        —una probabilidad es una puntuación ordenada— y NO vale para nada que
        necesite el rol o los pesos."""
        registro = self._registro("r1", "si", 0.8)
        # Sin decir cuál es la positiva, la respuesta correcta es que FALTA: no
        # se elige por orden alfabético.
        self.assertFalse(aptitud("auroc", registro).apto)
        self.assertEqual(aptitud("auroc", registro).faltan, ("positive_label",))
        self.assertTrue(aptitud("auroc", registro, positive_label=POSITIVA).apto)
        self.assertTrue(aptitud("brier_score", registro, positive_label=POSITIVA).apto)
        sin_verdad = PredictionRecord(row_id="r2", candidate="c1",
                                      probabilities=(0.2, 0.8), classes=CLASES)
        resultado = aptitud("auroc", sin_verdad, positive_label=POSITIVA)
        self.assertFalse(resultado.apto)
        self.assertEqual(resultado.faltan, ("y_true",))
        self.assertEqual(set(resultado.motivo), {"es", "en"})

    def test_un_registro_con_solo_etiqueta_no_vale_para_auroc_y_si_para_accuracy(self):
        registro = PredictionRecord(row_id="r1", candidate="c1", y_true="si",
                                    label="si", classes=CLASES)
        resultado = aptitud("auroc", registro, positive_label=POSITIVA)
        self.assertFalse(resultado.apto)
        self.assertEqual(resultado.faltan, ("scores",))
        self.assertTrue(aptitud("accuracy", registro).apto)


# ---------------------------------------------------------------------------
# Paridad con referencias de versión fijada
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAY_REFERENCIA, "sin scikit-learn/numpy en el entorno")
class ParidadConReferenciasDeVersionFijadaTest(unittest.TestCase):
    """Contra scikit-learn, sobre un fixture con empates, pesos y varias tallas.

    Las referencias NO entran en el core: `matrixai-core` declara
    `dependencies = []` y aquí solo se usan para comprobar que la aritmética de
    andar por casa da lo mismo que la de una biblioteca revisada.
    """

    def setUp(self):
        azar = random.Random(105)
        self.y = ["si" if azar.random() < 0.35 else "no" for _ in range(60)]
        # Rejilla gruesa a propósito: así hay empates de verdad.
        self.p = [round(azar.random(), 1) for _ in range(60)]
        self.w = [round(0.5 + azar.random() * 2.5, 2) for _ in range(60)]
        self.muestra = _binaria(self.y, probabilidades=self.p)
        self.pesada = _binaria(self.y, probabilidades=self.p, pesos=self.w)

    def _numpy(self):
        import numpy as np
        return (np.array([1 if y == "si" else 0 for y in self.y]),
                np.array(self.p), np.array(self.w))

    def test_hay_empates_en_el_fixture(self):
        self.assertLess(len(set(self.p)), len(self.p))

    def test_auroc_ap_trapecio_log_loss_y_brier(self):
        import numpy as np
        from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                                     log_loss, precision_recall_curve, roc_auc_score)
        y, p, _ = self._numpy()
        precision, recall, _u = precision_recall_curve(y, p)
        esperado = {
            "auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "pr_auc_trapezoidal": float(auc(recall, precision)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, p)),
        }
        for metric_id, referencia in esperado.items():
            with self.subTest(metric_id=metric_id):
                self.assertAlmostEqual(_valor(metric_id, self.muestra), referencia,
                                       places=12)

    def test_las_mismas_con_pesos(self):
        import numpy as np
        from sklearn.metrics import (average_precision_score, brier_score_loss,
                                     log_loss, roc_auc_score)
        y, p, w = self._numpy()
        esperado = {
            "auroc": float(roc_auc_score(y, p, sample_weight=w)),
            "average_precision": float(average_precision_score(y, p, sample_weight=w)),
            "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
            "brier_score": float(brier_score_loss(y, p, sample_weight=w)),
        }
        for metric_id, referencia in esperado.items():
            with self.subTest(metric_id=metric_id):
                self.assertAlmostEqual(_valor(metric_id, self.pesada), referencia,
                                       places=12)
        # Y los pesos cambian el número: si no, la prueba no probaría nada.
        self.assertNotAlmostEqual(_valor("auroc", self.pesada),
                                  _valor("auroc", self.muestra), places=6)

    def test_las_del_umbral_y_la_matriz(self):
        import numpy as np
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        y, p, _ = self._numpy()
        predicho = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicho, labels=[0, 1]).ravel()
        self.assertAlmostEqual(_valor("sensitivity", self.muestra),
                               tp / (tp + fn), places=15)
        self.assertAlmostEqual(_valor("specificity", self.muestra),
                               tn / (tn + fp), places=15)
        self.assertAlmostEqual(_valor("ppv", self.muestra), tp / (tp + fp), places=15)
        self.assertAlmostEqual(_valor("npv", self.muestra), tn / (tn + fn), places=15)
        self.assertAlmostEqual(_valor("accuracy", self.muestra),
                               float(accuracy_score(y, predicho)), places=15)
        self.assertAlmostEqual(_valor("macro_f1", self.muestra),
                               float(f1_score(y, predicho, average="macro")), places=5)
        matriz = matriz_de_confusion(self.muestra)
        self.assertEqual(matriz.tp, int(tp))
        self.assertEqual(matriz.fp, int(fp))
        self.assertEqual(matriz.fn, int(fn))
        self.assertEqual(matriz.tn, int(tn))

    def test_multiclase_log_loss_y_accuracy(self):
        import numpy as np
        from sklearn.metrics import accuracy_score, log_loss
        azar = random.Random(1050)
        clases = ("a", "b", "c")
        filas = []
        reales = []
        for _ in range(40):
            crudo = [azar.random() + 0.05 for _ in clases]
            total = sum(crudo)
            filas.append([v / total for v in crudo])
            reales.append(azar.choice(clases))
        muestra = Muestra.multiclase(reales, classes=clases, probabilidades=filas)
        y = np.array([clases.index(c) for c in reales])
        p = np.array(filas)
        self.assertAlmostEqual(_valor("log_loss", muestra),
                               float(log_loss(y, p, labels=[0, 1, 2])), places=12)
        predicho = p.argmax(axis=1)
        self.assertAlmostEqual(_valor("accuracy", muestra),
                               float(accuracy_score(y, predicho)), places=15)

    def test_regresion_con_y_sin_pesos(self):
        import numpy as np
        from sklearn.metrics import (mean_absolute_error, r2_score,
                                     root_mean_squared_error)
        azar = random.Random(10500)
        reales = [round(azar.uniform(-5, 25), 3) for _ in range(50)]
        predichos = [round(y + azar.gauss(0, 2), 3) for y in reales]
        pesos = [round(0.2 + azar.random() * 3, 2) for _ in reales]
        muestra = Muestra.regresion(reales, predichos)
        pesada = Muestra.regresion(reales, predichos, pesos=pesos)
        yt, yp, w = np.array(reales), np.array(predichos), np.array(pesos)
        self.assertAlmostEqual(_valor("mae", muestra),
                               float(mean_absolute_error(yt, yp)), places=12)
        self.assertAlmostEqual(_valor("rmse", muestra),
                               float(root_mean_squared_error(yt, yp)), places=12)
        self.assertAlmostEqual(_valor("r2", muestra), float(r2_score(yt, yp)), places=12)
        self.assertAlmostEqual(_valor("mae", pesada),
                               float(mean_absolute_error(yt, yp, sample_weight=w)),
                               places=12)
        self.assertAlmostEqual(_valor("rmse", pesada),
                               float(root_mean_squared_error(yt, yp, sample_weight=w)),
                               places=12)
        self.assertAlmostEqual(_valor("r2", pesada),
                               float(r2_score(yt, yp, sample_weight=w)), places=12)

    def test_el_clipping_de_la_log_loss_es_EL_MISMO_que_el_de_la_referencia(self):
        """Con probabilidades de 0 y 1 exactos, sin clipping el logaritmo se va a
        infinito y el informe deja de ser JSON. La referencia recorta con el
        épsilon de la máquina y aquí se recorta con el mismo."""
        import numpy as np
        from sklearn.metrics import log_loss
        extrema = _binaria(["si", "no"], probabilidades=[0.0, 1.0])
        y = np.array([1, 0])
        p = np.array([0.0, 1.0])
        self.assertAlmostEqual(_valor("log_loss", extrema),
                               float(log_loss(y, p, labels=[0, 1])), places=12)
        self.assertAlmostEqual(_valor("log_loss", extrema), -math.log(CLIP_LOG_LOSS),
                               places=12)
        self.assertTrue(math.isfinite(_valor("log_loss", extrema)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
