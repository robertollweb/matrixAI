# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C4 — segmentos, errores e importancia.

Criterio de terminado: «Segmento con ruido se identifica solo con soporte
suficiente; errores altos en regresión se orientan correctamente; categoría
con 30 filas y un evento recibe cautela. Columna ruido y variables
correlacionadas muestran alcance y límites. Análisis exploratorio queda
marcado.»
"""
from __future__ import annotations

import random
import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.metricas import EntradaNoMedible, Muestra
from matrixai.estudio.segmentos import (
    MINIMO_EVENTOS_POR_SEGMENTO,
    MINIMO_FILAS_POR_SEGMENTO,
    AnalisisDeSegmento,
    ImportanciaDeVariable,
    analizar_segmento,
    importancia_por_permutacion,
)


class AnalizarSegmentoTest(unittest.TestCase):
    def test_errores_altos_en_regresion_se_orientan_correctamente(self):
        """El criterio literal: RMSE mayor es peor. Un segmento con más
        error tiene que salir `peor_que_el_resto=True`, y uno con menos
        error `False` -- nunca "inferior" por estar numéricamente por
        debajo sin mirar la dirección de la métrica."""
        rng = random.Random(1)
        y_true = [rng.gauss(0, 1) for _ in range(200)]
        peor = [y + rng.gauss(0, 3.0) if i < 60 else y + rng.gauss(0, 0.2)
               for i, y in enumerate(y_true)]
        mejor = [y + rng.gauss(0, 0.05) if i < 60 else y + rng.gauss(0, 1.0)
                for i, y in enumerate(y_true)]

        m_peor = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(peor))
        r_peor = analizar_segmento("rmse", m_peor, list(range(60)),
                                   segmento_id="ruidoso", predefinido=True)
        self.assertTrue(r_peor.peor_que_el_resto)
        self.assertGreater(r_peor.metrica_segmento.value, r_peor.metrica_resto.value)

        m_mejor = Muestra(task="regression", y_true=tuple(y_true), predictions=tuple(mejor))
        r_mejor = analizar_segmento("rmse", m_mejor, list(range(60)),
                                    segmento_id="preciso", predefinido=True)
        self.assertFalse(r_mejor.peor_que_el_resto)
        self.assertLess(r_mejor.metrica_segmento.value, r_mejor.metrica_resto.value)

    def test_treinta_filas_un_evento_recibe_cautela_por_eventos(self):
        """El caso literal del criterio: 30 filas alcanza el mínimo de
        FILAS, pero un solo evento de la clase minoritaria no alcanza el
        mínimo de EVENTOS -- hacen falta los dos umbrales."""
        y_true = ["no"] * 29 + ["si"]
        probs = [(0.9, 0.1)] * 29 + [(0.4, 0.6)]
        m = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                   positive_label="si", probabilities=tuple(probs))
        r = analizar_segmento("accuracy", m, list(range(30)), segmento_id="rara", predefinido=False)
        self.assertEqual(r.n, MINIMO_FILAS_POR_SEGMENTO)
        self.assertFalse(r.soporte_suficiente)
        self.assertIn("evento", r.motivo_de_cautela["es"])

    def test_segmento_pequeno_recibe_cautela_por_filas_no_por_eventos(self):
        """Con menos de 30 filas, el motivo tiene que ser el de FILAS
        insuficientes -- no confundirlo con el de eventos aunque las dos
        cosas puedan fallar a la vez."""
        y_true = ["si"] * 10 + ["no"] * 10
        probs = [(0.1, 0.9)] * 10 + [(0.9, 0.1)] * 10
        m = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                   positive_label="si", probabilities=tuple(probs))
        r = analizar_segmento("accuracy", m, list(range(20)), segmento_id="chico", predefinido=False)
        self.assertFalse(r.soporte_suficiente)
        self.assertIn("fila", r.motivo_de_cautela["es"])
        self.assertNotIn("evento", r.motivo_de_cautela["es"])

    def test_segmento_con_soporte_suficiente_no_lleva_motivo(self):
        rng = random.Random(2)
        n = 200
        y_true = [("si" if rng.random() < 0.4 else "no") for _ in range(n)]
        probs = [(0.3, 0.7) if y == "si" else (0.7, 0.3) for y in y_true]
        m = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                   positive_label="si", probabilities=tuple(probs))
        r = analizar_segmento("accuracy", m, list(range(80)), segmento_id="grande", predefinido=True)
        self.assertGreaterEqual(r.n, MINIMO_FILAS_POR_SEGMENTO)
        self.assertGreaterEqual(min(r.eventos, r.n - r.eventos), MINIMO_EVENTOS_POR_SEGMENTO)
        self.assertTrue(r.soporte_suficiente)
        self.assertIsNone(r.motivo_de_cautela)

    def test_exploratorio_se_marca(self):
        """El criterio literal: «análisis exploratorio queda marcado»."""
        m = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        predefinido = analizar_segmento("rmse", m, [0], segmento_id="a", predefinido=True)
        exploratorio = analizar_segmento("rmse", m, [0], segmento_id="b", predefinido=False)
        self.assertFalse(predefinido.es_exploratorio)
        self.assertTrue(exploratorio.es_exploratorio)

    def test_segmento_vacio_no_fabrica_diferencia(self):
        """Un segmento de 0 filas no es un documento imposible (una `Muestra`
        vacía es indefinida en toda métrica) -- `diferencia` y
        `peor_que_el_resto` tienen que quedar en `None`, no en un 0/False
        de relleno."""
        m = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        r = analizar_segmento("rmse", m, [], segmento_id="vacio", predefinido=False)
        self.assertEqual(r.n, 0)
        self.assertIsNone(r.metrica_segmento.value)
        self.assertIsNone(r.diferencia)
        self.assertIsNone(r.peor_que_el_resto)

    def test_indices_repetidos_se_rechaza(self):
        m = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        with self.assertRaises(EntradaNoMedible):
            analizar_segmento("rmse", m, [0, 0, 1], segmento_id="x", predefinido=True)

    def test_indice_fuera_de_rango_se_rechaza(self):
        m = Muestra(task="regression", y_true=(1.0, 2.0, 3.0), predictions=(1.1, 2.1, 2.9))
        with self.assertRaises(EntradaNoMedible):
            analizar_segmento("rmse", m, [0, 99], segmento_id="x", predefinido=True)

    def test_construccion_incoherente_se_rechaza(self):
        m = Muestra(task="regression", y_true=(1.0,), predictions=(1.1,))
        base = analizar_segmento("rmse", m, [0], segmento_id="x", predefinido=True)
        with self.assertRaises(EsquemaInvalido):
            AnalisisDeSegmento(segmento_id="x", predefinido=True, metric_id="rmse", n=1,
                              eventos=None, metrica_segmento=base.metrica_segmento,
                              metrica_resto=base.metrica_resto, diferencia=None,
                              peor_que_el_resto=True,  # incoherente: uno None y el otro no
                              soporte_suficiente=False, motivo_de_cautela={"es": "x", "en": "x"})
        with self.assertRaises(EsquemaInvalido):
            AnalisisDeSegmento(segmento_id="x", predefinido=True, metric_id="rmse", n=1,
                              eventos=None, metrica_segmento=base.metrica_segmento,
                              metrica_resto=base.metrica_resto, diferencia=0.1,
                              peor_que_el_resto=True,
                              soporte_suficiente=True,  # incoherente: suficiente CON motivo
                              motivo_de_cautela={"es": "x", "en": "x"})


class ImportanciaPorPermutacionTest(unittest.TestCase):
    def _dataset_real_y_ruido(self, seed=7, n=400):
        rng = random.Random(seed)
        filas, y_true = [], []
        for _ in range(n):
            real, ruido = rng.gauss(0, 1), rng.gauss(0, 1)
            filas.append({"real": real, "ruido": ruido})
            y_true.append("si" if real > 0 else "no")
        return filas, tuple(y_true)

    def test_columna_real_pesa_mas_que_ruido(self):
        """El criterio literal: «columna ruido... muestra alcance y
        límites» -- una variable que el modelo nunca mira tiene que salir
        con caída prácticamente nula frente a la que sí decide la
        predicción."""
        filas, y_true = self._dataset_real_y_ruido()

        def puntuar(filas_):
            preds = ["si" if f["real"] > 0 else "no" for f in filas_]
            return Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                           positive_label="si", predictions=tuple(preds))

        resultado = {r.variable: r for r in importancia_por_permutacion(
            "accuracy", filas, puntuar, variables={"real": ["real"], "ruido": ["ruido"]},
            n_repeticiones=6, semilla=42)}
        self.assertGreater(resultado["real"].caida_media, 0.3)
        self.assertAlmostEqual(resultado["ruido"].caida_media, 0.0, places=6)

    def test_variacion_entre_repeticiones(self):
        """Cada repetición baraja con una semilla DISTINTA -- si todas
        dieran la misma permutación, `caidas` sería una tupla de valores
        idénticos, y el `desviacion` medido no significaría nada."""
        filas, y_true = self._dataset_real_y_ruido(seed=3)

        def puntuar(filas_):
            preds = ["si" if f["real"] > 0 else "no" for f in filas_]
            return Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                           positive_label="si", predictions=tuple(preds))

        resultado = importancia_por_permutacion(
            "accuracy", filas, puntuar, variables={"real": ["real"]},
            n_repeticiones=8, semilla=1)[0]
        self.assertGreater(len(set(resultado.caidas)), 1)

    def test_variables_agrupadas_se_permutan_juntas(self):
        """Una categórica de dos columnas one-hot complementarias (
        `color_rojo`/`color_azul`) declarada como UN grupo se baraja con la
        MISMA permutación de filas para las dos columnas a la vez -- si se
        barajaran por separado (o solo se permutara la primera del grupo),
        la complementariedad se rompería y la columna que de verdad decide
        la predicción (`color_rojo`, deliberadamente declarada SEGUNDA en
        el grupo) podría quedar intacta."""
        rng = random.Random(9)
        n = 300
        filas, y_true = [], []
        for _ in range(n):
            rojo = 1.0 if rng.random() > 0.5 else 0.0
            filas.append({"color_rojo": rojo, "color_azul": 1.0 - rojo})
            y_true.append("si" if rojo == 1.0 else "no")
        y_true = tuple(y_true)

        def puntuar(filas_):
            preds = ["si" if f["color_rojo"] == 1.0 else "no" for f in filas_]
            return Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                           positive_label="si", predictions=tuple(preds))

        resultado = importancia_por_permutacion(
            "accuracy", filas, puntuar, variables={"color": ["color_azul", "color_rojo"]},
            n_repeticiones=4, semilla=1)[0]
        self.assertGreater(resultado.caida_media, 0.3)

    def test_correlacion_diluye_la_importancia(self):
        """El criterio literal: «variables correlacionadas muestran
        alcance y límites» -- esto NO es un defecto a arreglar, es una
        propiedad conocida del método: permutar una variable con una
        gemela correlacionada no aísla su aportación, porque la gemela
        sigue llevando la misma señal."""
        filas, y_true = self._dataset_real_y_ruido(seed=11)
        filas_x1_solo = [{"x1": f["real"], "x2": f["ruido"]} for f in filas]
        filas_x1_con_gemela = [{"x1": f["real"], "x2": f["real"]} for f in filas]

        def puntuar_solo_x1(filas_):
            preds = ["si" if f["x1"] > 0 else "no" for f in filas_]
            return Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                           positive_label="si", predictions=tuple(preds))

        def puntuar_promedio(filas_):
            preds = ["si" if (f["x1"] + f["x2"]) / 2 > 0 else "no" for f in filas_]
            return Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                           positive_label="si", predictions=tuple(preds))

        solo = importancia_por_permutacion("accuracy", filas_x1_solo, puntuar_solo_x1,
                                           variables={"x1": ["x1"]}, n_repeticiones=6, semilla=42)[0]
        con_gemela = importancia_por_permutacion("accuracy", filas_x1_con_gemela, puntuar_promedio,
                                                 variables={"x1": ["x1"]}, n_repeticiones=6, semilla=42)[0]
        self.assertLess(con_gemela.caida_media, solo.caida_media)

    def test_metrica_base_indefinida_se_declara(self):
        y_mono = ["no"] * 20
        filas = [{"x": float(i)} for i in range(20)]

        def puntuar(filas_):
            return Muestra(task="binary_classification", y_true=tuple(y_mono), classes=("no", "si"),
                           positive_label="si", scores=tuple(f["x"] for f in filas_))

        resultado = importancia_por_permutacion("auroc", filas, puntuar,
                                                variables={"x": ["x"]}, n_repeticiones=3, semilla=1)[0]
        self.assertIsNone(resultado.caida_media)
        self.assertIsNotNone(resultado.undefined_reason)

    def test_metrica_permutada_indefinida_no_promedia_a_medias(self):
        """Si UNA repetición deja la métrica indefinida, la variable entera
        se declara indefinida -- promediar sobre menos repeticiones de las
        declaradas sería una cifra fabricada. Escenario forzado a propósito
        (no uno que surja solo): la función de puntuación declara una sola
        clase en cuanto detecta que la columna llegó permutada."""
        y_true_original = tuple(["si"] * 10 + ["no"] * 10)
        orden_original = list(range(20))
        filas = [{"x": float(i)} for i in orden_original]

        def puntuar(filas_):
            valores = [f["x"] for f in filas_]
            if valores == [float(i) for i in orden_original]:
                return Muestra(task="binary_classification", y_true=y_true_original,
                               classes=("no", "si"), positive_label="si",
                               scores=tuple(valores))
            return Muestra(task="binary_classification", y_true=tuple(["si"] * 20),
                           classes=("no", "si"), positive_label="si", scores=tuple(valores))

        resultado = importancia_por_permutacion("auroc", filas, puntuar,
                                                variables={"x": ["x"]}, n_repeticiones=3, semilla=5)[0]
        self.assertIsNone(resultado.caida_media)
        self.assertIsNotNone(resultado.undefined_reason)

    def test_variables_vacio_se_rechaza(self):
        m_filas = [{"x": 1.0}]

        def puntuar(filas_):
            return Muestra(task="regression", y_true=(1.0,), predictions=(1.0,))

        with self.assertRaises(EsquemaInvalido):
            importancia_por_permutacion("rmse", m_filas, puntuar, variables={}, n_repeticiones=1, semilla=1)

    def test_misma_semilla_reproduce_las_mismas_caidas(self):
        filas, y_true = self._dataset_real_y_ruido(seed=13)

        def puntuar(filas_):
            preds = ["si" if f["real"] > 0 else "no" for f in filas_]
            return Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                           positive_label="si", predictions=tuple(preds))

        r1 = importancia_por_permutacion("accuracy", filas, puntuar, variables={"real": ["real"]},
                                         n_repeticiones=4, semilla=99)[0]
        r2 = importancia_por_permutacion("accuracy", filas, puntuar, variables={"real": ["real"]},
                                         n_repeticiones=4, semilla=99)[0]
        self.assertEqual(r1.caidas, r2.caidas)


class ImportanciaDeVariableEsquemaTest(unittest.TestCase):
    def test_no_se_puede_construir_con_valor_y_motivo(self):
        with self.assertRaises(EsquemaInvalido):
            ImportanciaDeVariable(variable="x", n_repeticiones=3, caida_media=0.1,
                                  desviacion=0.0, caidas=(0.1, 0.1, 0.1),
                                  undefined_reason={"es": "x", "en": "x"})

    def test_no_se_puede_construir_sin_valor_ni_motivo(self):
        with self.assertRaises(EsquemaInvalido):
            ImportanciaDeVariable(variable="x", n_repeticiones=3)

    def test_caidas_desalineadas_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            ImportanciaDeVariable(variable="x", n_repeticiones=3, caida_media=0.1,
                                  desviacion=0.0, caidas=(0.1, 0.1))


if __name__ == "__main__":
    unittest.main()
