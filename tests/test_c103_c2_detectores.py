# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C2 — ERRORES ESTRUCTURALES, SOSPECHAS Y LÍMITES NO SON LA MISMA COSA.

El corte entero es que una correlación alta no se trate como prueba de fuga.
Los cuatro casos que este fichero mide son el criterio de terminado literal
del contrato:

1. un objetivo duplicado CONFIRMADO (coincidencia total, fila a fila) impide
   la recomendación — es `Bloqueo`, no se levanta aceptándolo;
2. un predictor legítimo muy correlacionado (el sueldo del año pasado, el
   caso del contrato 71) se CONSERVA con una `Sospecha` que pide contexto, no
   con un `Bloqueo`;
3. un identificador de alta cardinalidad, o una categoría espuria que por
   puro azar de muestra pequeña alcanza pureza alta IN-SAMPLE, no se califica
   como predictor útil solo por esa pureza — hace falta que se confirme en
   dos mitades FUERA de muestra;
4. la ausencia de información de linaje se muestra siempre como una
   limitación, nunca como ausencia de problema.

Los fixtures de pureza (2 y 3) no están inventados a ojo: se buscó por
semilla un caso real donde una categoría espuria SÍ alcanza el umbral
in-sample por azar (la trampa que el criterio de terminado #3 prohíbe caer
en) y un caso real donde una categoría de verdad predictiva SÍ se confirma en
las dos mitades — los dos con `random.seed` fijado y el resultado impreso
antes de escribir el aserto, no al revés.
"""
from __future__ import annotations

import random
import unittest
from importlib.util import find_spec

from matrixai.estudio import ProblemSpec
from matrixai.training.diagnostico import (
    Diagnostico,
    Limite,
    MUESTREO_MAXIMO_FILAS,
    N_MINIMO_ASOCIACION,
    RATIO_MINIMO_FILAS_POR_PREDICTOR,
    SOPORTE_MINIMO_PUREZA,
    Sospecha,
    UMBRAL_ASOCIACION_ALTA,
    UMBRAL_EVENTOS_MINIMOS,
    UMBRAL_NMI_ALTA,
    UMBRAL_PUREZA_ALTA,
    UMBRAL_PUREZA_ALTA_FUERA_DE_MUESTRA,
    asociacion,
    asociacion_muy_alta,
    ausencia_de_linaje,
    cruce_de_unidades,
    diagnosticar_csv,
    falta_de_eventos,
    identificador_probable,
    igualdad_de_valores,
    intervalo_wilson,
    nmi_categorica,
    objetivo_duplicado,
    pearson,
    precision_insuficiente,
    proxy_por_nmi,
    proxy_por_pureza,
    pureza_categorica,
    spearman,
    tamano_efectivo_insuficiente,
    variables_disponibles_tras_el_desenlace,
)
from matrixai.training.objetivo import Bloqueo

_TIENE_SCIPY = find_spec("scipy") is not None
_TIENE_SKLEARN = find_spec("sklearn") is not None


def _problema(**kwargs) -> ProblemSpec:
    base = dict(problem_id="p", task="regression", observation_unit="fila")
    base.update(kwargs)
    return ProblemSpec(**base)


def _problema_binaria(**kwargs) -> ProblemSpec:
    base = dict(problem_id="p", task="binary_classification", observation_unit="fila",
                classes=("si", "no"), positive_label="si")
    base.update(kwargs)
    return ProblemSpec(**base)


# ---------------------------------------------------------------------------
# Primitivas — Pearson, Spearman, con paridad contra scipy cuando existe
# ---------------------------------------------------------------------------

class PearsonSpearmanTest(unittest.TestCase):
    def test_pearson_None_con_menos_de_dos_puntos_o_serie_constante(self):
        self.assertIsNone(pearson([1.0], [2.0]))
        self.assertIsNone(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))

    def test_spearman_capta_monotona_no_solo_lineal(self):
        xs = [float(i) for i in range(1, 30)]
        ys = [x ** 3 for x in xs]
        # Monótona pura -> Spearman exacto en 1.0; Pearson bastante menos,
        # porque x**3 no es una recta.
        self.assertAlmostEqual(spearman(xs, ys), 1.0, places=9)
        self.assertLess(pearson(xs, ys), 0.95)

    @unittest.skipUnless(_TIENE_SCIPY, "requiere scipy instalado")
    def test_paridad_con_scipy(self):
        from scipy.stats import pearsonr, spearmanr
        random.seed(42)
        xs = [random.uniform(0, 100) for _ in range(50)]
        ys = [x * 2 + random.gauss(0, 10) for x in xs]
        self.assertAlmostEqual(pearson(xs, ys), pearsonr(xs, ys)[0], places=9)
        self.assertAlmostEqual(spearman(xs, ys), spearmanr(xs, ys)[0], places=9)

    def test_asociacion_None_por_debajo_del_minimo_de_pares(self):
        n = N_MINIMO_ASOCIACION - 1
        self.assertIsNone(asociacion(list(range(n)), list(range(n))))

    def test_asociacion_ignora_pares_con_lado_no_numerico(self):
        xs = list(range(40)) + [None, "texto"]
        ys = list(range(40)) + [1, 2]
        r = asociacion(xs, ys)
        self.assertEqual(r["n"], 40)


# ---------------------------------------------------------------------------
# NMI — sesgo por cardinalidad, y su corrección por soporte mínimo
# ---------------------------------------------------------------------------

class NmiCategoricaTest(unittest.TestCase):
    def test_identificador_infla_nmi_sin_colapsar_y_cae_a_cero_colapsando(self):
        random.seed(3)
        n = 300
        identificador = [f"id_{i}" for i in range(n)]
        objetivo = [random.choice(["si", "no"]) for _ in range(n)]
        sin_colapsar = nmi_categorica(identificador, objetivo, soporte_minimo=1)
        con_colapsar = nmi_categorica(identificador, objetivo, soporte_minimo=SOPORTE_MINIMO_PUREZA)
        # Medido: 0.2164 sin colapsar (ruido inflado por cardinalidad puro —
        # el objetivo es ALEATORIO, no hay señal real) frente a 0.0
        # colapsando (todas las categorías de soporte 1 se funden en una).
        # 0.2164 queda por debajo de `UMBRAL_NMI_ALTA` (0.3): el punto no es
        # que cruce el umbral de "alta", es que un identificador sin ninguna
        # relación real con el objetivo no debería dar NADA por encima de
        # cero, y sin corregir da un número visiblemente no-nulo.
        self.assertGreater(sin_colapsar["valor"], 0.1)
        self.assertEqual(con_colapsar["valor"], 0.0)
        self.assertTrue(con_colapsar["categorias_colapsadas"])
        self.assertGreater(sin_colapsar["valor"], con_colapsar["valor"])

    def test_proxy_por_nmi_no_acusa_al_identificador(self):
        random.seed(3)
        n = 300
        identificador = [f"id_{i}" for i in range(n)]
        objetivo = [random.choice(["si", "no"]) for _ in range(n)]
        self.assertIsNone(proxy_por_nmi(identificador, objetivo, columna="id"))

    @unittest.skipUnless(_TIENE_SKLEARN, "requiere scikit-learn instalado")
    def test_paridad_con_sklearn_average_arithmetic(self):
        from sklearn.metrics import normalized_mutual_info_score
        random.seed(7)
        cats = [random.choice(["a", "b", "c"]) for _ in range(300)]
        objetivo = ["si" if (c == "a" and random.random() < 0.8) else random.choice(["si", "no"])
                   for c in cats]
        mio = nmi_categorica(cats, objetivo, soporte_minimo=1)["valor"]
        suyo = normalized_mutual_info_score(cats, objetivo, average_method="arithmetic")
        self.assertAlmostEqual(mio, suyo, places=6)


# ---------------------------------------------------------------------------
# Wilson — centro y semiancho, y por qué el margen se mide en RELATIVO
# ---------------------------------------------------------------------------

class IntervaloWilsonTest(unittest.TestCase):
    def test_None_fuera_de_rango(self):
        self.assertIsNone(intervalo_wilson(-1, 10))
        self.assertIsNone(intervalo_wilson(11, 10))
        self.assertIsNone(intervalo_wilson(0, 0))

    def test_margen_relativo_cae_al_crecer_los_eventos(self):
        _, semi_10 = intervalo_wilson(10, 100)
        _, semi_100 = intervalo_wilson(100, 1000)
        margen_10 = semi_10 / (10 / 100)
        margen_100 = semi_100 / (100 / 1000)
        # Medido en el docstring del módulo: ~0.60-0.65 con 10 eventos,
        # ~0.19 con 100 — la SITUACIÓN de precisión, no solo el número, mejora.
        self.assertGreater(margen_10, 0.5)
        self.assertLess(margen_100, 0.25)
        self.assertGreater(margen_10, margen_100)


# ---------------------------------------------------------------------------
# Criterio de terminado #1 — objetivo duplicado CONFIRMADO bloquea
# ---------------------------------------------------------------------------

class ObjetivoDuplicadoTest(unittest.TestCase):
    def test_coincidencia_total_bloquea(self):
        precio = [str(x) for x in range(50)]
        b = objetivo_duplicado(precio, precio, objetivo="precio", predictor="precio_copiado")
        self.assertIsInstance(b, Bloqueo)
        self.assertEqual(b.clave, "objetivo_duplicado_confirmado")
        self.assertEqual(b.campo, "precio_copiado")

    def test_una_sola_fila_distinta_NO_bloquea(self):
        """Es la frontera literal con `asociacion_muy_alta`: una coincidencia
        PARCIAL, por pequeña que sea la diferencia, es una asociación alta —
        no una identidad."""
        objetivo = [str(x) for x in range(50)]
        predictor = list(objetivo)
        predictor[0] = "999999"
        self.assertIsNone(
            objetivo_duplicado(objetivo, predictor, objetivo="precio", predictor="casi_precio"))

    def test_menos_del_minimo_de_filas_comparables_no_bloquea(self):
        precio = [str(x) for x in range(5)]
        self.assertIsNone(
            objetivo_duplicado(precio, precio, objetivo="precio", predictor="precio2",
                               minimo_filas=20))

    def test_nulos_no_cuentan_como_coincidencia_ni_como_diferencia(self):
        objetivo = [str(x) for x in range(30)] + [""] * 5
        predictor = [str(x) for x in range(30)] + ["otra_cosa"] * 5
        b = objetivo_duplicado(objetivo, predictor, objetivo="precio", predictor="precio2")
        self.assertIsInstance(b, Bloqueo)


class VariablesTrasElDesenlaceTest(unittest.TestCase):
    def test_bloquea_solo_lo_declarado_after_outcome(self):
        problema = _problema(target="precio", predictors=("metros", "reclamacion_post_venta"),
                             predictor_availability={"metros": "at_prediction_time",
                                                     "reclamacion_post_venta": "after_outcome"})
        bloqueos = variables_disponibles_tras_el_desenlace(problema)
        self.assertEqual(len(bloqueos), 1)
        self.assertEqual(bloqueos[0].campo, "reclamacion_post_venta")

    def test_sin_declaracion_no_bloquea_nada(self):
        """Invariante 5: la AUSENCIA de declaración no es una declaración de
        disponibilidad — no se infiere `after_outcome` por omisión."""
        problema = _problema(target="precio", predictors=("metros", "barrio"))
        self.assertEqual(variables_disponibles_tras_el_desenlace(problema), ())


class CruceDeUnidadesTest(unittest.TestCase):
    def test_columna_unidad_como_entrada_bloquea(self):
        b = cruce_de_unidades(["edad", "paciente_id"], "paciente_id",
                              unidad_de_observacion="paciente")
        self.assertIsInstance(b, Bloqueo)
        self.assertEqual(b.campo, "paciente_id")

    def test_sin_columna_unidad_declarada_no_bloquea(self):
        self.assertIsNone(cruce_de_unidades(["edad", "peso"], None))

    def test_columna_unidad_que_no_es_entrada_no_bloquea(self):
        self.assertIsNone(cruce_de_unidades(["edad", "peso"], "paciente_id"))


# ---------------------------------------------------------------------------
# Criterio de terminado #2 — el sueldo del año pasado (contrato 71) NUNCA
# se convierte en Bloqueo por su correlación
# ---------------------------------------------------------------------------

class AsociacionMuyAltaSalarioDelSetentaYUnoTest(unittest.TestCase):
    """`last_year_salary` predice bien `salary`: r alto y legítimo. Medido con
    `random.seed(71)` sobre una relación lineal con ruido — no un dataset
    perfecto a propósito, para no probar solo el caso ideal."""

    def setUp(self):
        random.seed(71)
        n = 200
        self.last_year = [round(random.uniform(30000, 90000), 2) for _ in range(n)]
        self.salary = [round(ly * 1.03 + random.gauss(0, 3000), 2) for ly in self.last_year]
        self.r = pearson(self.last_year, self.salary)
        # Los detectores de coincidencia EXACTA (`objetivo_duplicado`,
        # `igualdad_de_valores`) reciben las columnas del CSV tal cual las
        # entrega `_read_rows`: texto, no floats — igual que en producción.
        self.last_year_str = [str(v) for v in self.last_year]
        self.salary_str = [str(v) for v in self.salary]

    def test_la_correlacion_medida_supera_el_umbral(self):
        # Medido: ~0.98 — igual de "sospechoso" en magnitud que una fuga real,
        # que es exactamente el punto del contrato 71.
        self.assertGreater(abs(self.r), UMBRAL_ASOCIACION_ALTA)

    def test_asociacion_muy_alta_devuelve_SOSPECHA_nunca_bloqueo(self):
        s = asociacion_muy_alta(self.last_year, self.salary, columna="last_year_salary")
        self.assertIsInstance(s, Sospecha)
        self.assertNotIsInstance(s, Bloqueo)
        self.assertEqual(s.clave, "asociacion_muy_alta")
        self.assertIn("last_year_salary", s.motivo["es"])

    def test_objetivo_duplicado_NO_confunde_correlacion_alta_con_identidad(self):
        """El mismo dataset, por la puerta del detector de bloqueo: al no
        coincidir fila a fila, `objetivo_duplicado` no lo toca — es
        `asociacion_muy_alta` quien lo mide, y como Sospecha."""
        self.assertIsNone(
            objetivo_duplicado(self.salary_str, self.last_year_str,
                               objetivo="salary", predictor="last_year_salary"))

    def test_correlacion_alta_pero_con_pocas_filas_no_se_acusa(self):
        """Sin significatividad estadística (`_r_critico`), un r extremo sobre
        una muestra minúscula es ruido, no señal."""
        self.assertIsNone(asociacion_muy_alta([1.0, 2.0], [1.0, 2.0], columna="x"))


class IgualdadDeValoresTest(unittest.TestCase):
    def test_dos_entradas_identicas_es_sospecha_no_bloqueo(self):
        celsius = [str(x) for x in range(30)]
        s = igualdad_de_valores(celsius, celsius, columna_a="temp_c", columna_b="temp_c_bis")
        self.assertIsInstance(s, Sospecha)
        self.assertEqual(s.clave, "igualdad_de_valores_sin_explicar")

    def test_entradas_distintas_no_genera_sospecha(self):
        a = [str(x) for x in range(30)]
        b = [str(x + 1) for x in range(30)]
        self.assertIsNone(igualdad_de_valores(a, b, columna_a="a", columna_b="b"))


# ---------------------------------------------------------------------------
# Criterio de terminado #3 — pureza: ni el identificador ni la categoría
# espuria se confirman como predictor útil solo por pureza IN-SAMPLE
# ---------------------------------------------------------------------------

class PurezaCategoricaTest(unittest.TestCase):
    def test_identificador_de_fila_nunca_alcanza_soporte_minimo(self):
        """Cada categoría del identificador tiene una sola fila: por debajo
        de `SOPORTE_MINIMO_PUREZA`, ninguna se evalúa siquiera — pureza 1.0
        trivial descartada antes de mirarla."""
        random.seed(1)
        n = 300
        identificador = [f"id_{i}" for i in range(n)]
        objetivo = [random.choice(["si", "no"]) for _ in range(n)]
        r = pureza_categorica(identificador, objetivo)
        self.assertEqual(r["categorias_evaluadas"], {})
        self.assertIsNone(proxy_por_pureza(identificador, objetivo, columna="id"))

    def test_categoria_ESPURIA_con_soporte_suficiente_pura_in_sample_NO_se_confirma(self):
        """El caso adversarial real del criterio de terminado #3: 40
        categorías de 20 filas cada una (soporte exactamente al mínimo), SIN
        relación real con el objetivo. Buscado por semilla y medido antes de
        escribir el aserto: con `random.seed(22)` una categoría (`cat_22`)
        alcanza pureza in-sample >= 0.85 por puro azar de muestra pequeña —
        la trampa exacta que el contrato prohíbe calificar como predictor."""
        random.seed(22)
        categorias = []
        for i in range(40):
            categorias += [f"cat_{i}"] * 20
        random.shuffle(categorias)
        objetivo = [random.choice(["si", "no"]) for _ in range(len(categorias))]

        r = pureza_categorica(categorias, objetivo)
        puras_in_sample = [c for c, info in r["categorias_evaluadas"].items()
                          if info["pureza"] >= UMBRAL_PUREZA_ALTA]
        self.assertIn("cat_22", puras_in_sample,
                      "fixture: se esperaba que cat_22 fuese pura in-sample por azar")
        self.assertNotIn("cat_22", r["confirmadas_fuera_de_muestra"])
        self.assertIsNone(proxy_por_pureza(categorias, objetivo, columna="cat_espuria"))

    def test_categoria_REALMENTE_predictiva_SI_se_confirma_y_es_sospecha(self):
        """Contraste positivo del caso anterior: la misma mecánica, pero con
        una categoría de verdad ligada al objetivo (98 % de probabilidad, 60
        filas), confirmada en las dos mitades — y aun así Sospecha, no
        Bloqueo, porque una categoría predictiva no es una fuga confirmada."""
        random.seed(5)
        categorias = []
        for i in range(10):
            categorias += [f"cat_{i}"] * 60
        random.shuffle(categorias)
        objetivo = ["si" if (c == "cat_0" and random.random() < 0.98)
                   else random.choice(["si", "no"]) for c in categorias]

        r = pureza_categorica(categorias, objetivo)
        self.assertIn("cat_0", r["confirmadas_fuera_de_muestra"])
        s = proxy_por_pureza(categorias, objetivo, columna="cat_real")
        self.assertIsInstance(s, Sospecha)
        self.assertNotIsInstance(s, Bloqueo)
        self.assertEqual(s.clave, "categoria_predictiva_por_pureza")


class IdentificadorProbableTest(unittest.TestCase):
    def test_por_tipo_de_columna(self):
        s = identificador_probable("cliente_id", tipo_columna="identifier")
        self.assertIsInstance(s, Sospecha)
        self.assertNotIsInstance(s, Bloqueo)

    def test_por_nombre_sin_tipo(self):
        s = identificador_probable("customer_id", tipo_columna=None)
        self.assertIsInstance(s, Sospecha)

    def test_columna_normal_no_es_identificador(self):
        self.assertIsNone(identificador_probable("edad", tipo_columna="number"))


# ---------------------------------------------------------------------------
# Límites de estimación — no son fugas, acotan la conclusión
# ---------------------------------------------------------------------------

class LimitesTest(unittest.TestCase):
    def test_falta_de_eventos_solo_para_clases_con_pocos_casos_pero_mas_de_cero(self):
        objetivo = ["si"] * 3 + ["no"] * 50
        limites = falta_de_eventos(objetivo, objetivo="y")
        self.assertEqual(len(limites), 1)
        self.assertEqual(limites[0].medida["clase"], "si")
        self.assertEqual(limites[0].medida["casos"], 3)

    def test_tamano_efectivo_insuficiente_por_ratio_filas_predictor(self):
        l = tamano_efectivo_insuficiente(50, ["a", "b", "c", "d", "e", "f"])
        self.assertIsInstance(l, Limite)
        l_ok = tamano_efectivo_insuficiente(1000, ["a", "b"])
        self.assertIsNone(l_ok)

    def test_precision_insuficiente_es_relativo_no_absoluto(self):
        """10 eventos en 1000 filas (proporción 1 %): el margen absoluto es
        pequeño pero el RELATIVO es enorme — declarado insuficiente."""
        l = precision_insuficiente(10, 1000, campo="y")
        self.assertIsInstance(l, Limite)
        l_ok = precision_insuficiente(500, 1000, campo="y")
        self.assertIsNone(l_ok)

    def test_ausencia_de_linaje_siempre_presente_con_las_columnas(self):
        l = ausencia_de_linaje(["a", "b", "c"])
        self.assertIsInstance(l, Limite)
        self.assertEqual(l.clave, "ausencia_de_linaje")
        self.assertEqual(l.medida["columnas"], ["a", "b", "c"])


# ---------------------------------------------------------------------------
# El orquestador — los cuatro criterios de terminado, sobre un CSV completo
# ---------------------------------------------------------------------------

class DiagnosticarCsvTest(unittest.TestCase):
    def test_caso_1_del_71_precio_duplicado_bloquea_e_impide_recomendacion(self):
        filas = [{"precio": str(i * 1000), "precio_de_venta": str(i * 1000),
                  "metros": str(50 + i)} for i in range(30)]
        analisis = {"columns": {"precio_de_venta": {"type": "number", "cardinality": 30},
                                "metros": {"type": "number", "cardinality": 30}}}
        problema = _problema(target="precio", predictors=("precio_de_venta", "metros"))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)
        self.assertIsInstance(d, Diagnostico)
        self.assertTrue(d.impide_recomendacion)
        self.assertEqual({b.clave for b in d.bloqueos}, {"objetivo_duplicado_confirmado"})
        self.assertEqual(d.bloqueos[0].campo, "precio_de_venta")

    def test_caso_2_del_71_salario_legitimo_no_bloquea_solo_avisa(self):
        random.seed(71)
        n = 200
        last_year = [round(random.uniform(30000, 90000), 2) for _ in range(n)]
        salary = [round(ly * 1.03 + random.gauss(0, 3000), 2) for ly in last_year]
        departamentos = [f"dept_{i % 5}" for i in range(n)]
        filas = [{"salary": str(salary[i]), "last_year_salary": str(last_year[i]),
                  "department": departamentos[i]} for i in range(n)]
        analisis = {"columns": {"last_year_salary": {"type": "number", "cardinality": n},
                                "department": {"type": "categorical", "cardinality": 5}}}
        problema = _problema(target="salary", predictors=("last_year_salary", "department"))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)

        self.assertFalse(d.impide_recomendacion, "un predictor legítimo no debe bloquear el estudio")
        self.assertEqual(d.bloqueos, ())
        claves_sospecha = {s.clave for s in d.sospechas}
        self.assertIn("asociacion_muy_alta", claves_sospecha)
        sospecha_salario = next(s for s in d.sospechas if s.clave == "asociacion_muy_alta")
        self.assertEqual(sospecha_salario.campo, "last_year_salary")

    def test_caso_3_identificador_de_alta_cardinalidad_no_es_predictor_util(self):
        random.seed(1)
        n = 300
        objetivo = [random.choice(["si", "no"]) for _ in range(n)]
        filas = [{"y": objetivo[i], "cliente_id": f"id_{i}"} for i in range(n)]
        analisis = {"columns": {"cliente_id": {"type": "identifier", "cardinality": n}}}
        problema = _problema_binaria(target="y", predictors=("cliente_id",))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)

        # SÍ se avisa que PARECE un identificador (Sospecha, por tipo)...
        self.assertTrue(any(s.clave == "identificador_probable" for s in d.sospechas))
        # ...pero NUNCA se le da valor de predictor confirmado por pureza:
        # con cardinalidad = n, ninguna categoría alcanza soporte mínimo.
        self.assertFalse(any(s.clave == "categoria_predictiva_por_pureza" for s in d.sospechas))
        self.assertFalse(d.impide_recomendacion)

    def test_criterio_4_ausencia_de_linaje_siempre_esta_pase_lo_que_pase(self):
        """Con datos limpios, sin correlaciones ni identificadores ni nada
        raro: el límite de linaje sigue presente. No es un aviso condicional
        a que algo más haya saltado — es SIEMPRE."""
        filas = [{"y": str(i % 2), "x": str(i)} for i in range(60)]
        analisis = {"columns": {"x": {"type": "number", "cardinality": 60}}}
        problema = _problema(target="y", predictors=("x",))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)
        self.assertTrue(any(l.clave == "ausencia_de_linaje" for l in d.limites))

    def test_columna_unidad_como_entrada_bloquea_a_traves_del_orquestador(self):
        filas = [{"y": str(i % 2), "paciente_id": str(i), "edad": str(20 + i)} for i in range(40)]
        analisis = {"columns": {"paciente_id": {"type": "identifier", "cardinality": 40},
                                "edad": {"type": "number", "cardinality": 40}}}
        problema = _problema_binaria(target="y", observation_unit="paciente",
                                     predictors=("paciente_id", "edad"))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas,
                             columna_unidad="paciente_id")
        self.assertTrue(d.impide_recomendacion)
        self.assertIn("cruce_de_unidades_prohibido", {b.clave for b in d.bloqueos})

    def test_variable_after_outcome_bloquea_a_traves_del_orquestador(self):
        filas = [{"y": str(i % 2), "edad": str(20 + i), "resultado_visita": str(i % 2)}
                for i in range(40)]
        analisis = {"columns": {"edad": {"type": "number", "cardinality": 40},
                                "resultado_visita": {"type": "categorical", "cardinality": 2}}}
        problema = _problema_binaria(
            target="y", predictors=("edad", "resultado_visita"),
            predictor_availability={"edad": "at_prediction_time",
                                    "resultado_visita": "after_outcome"})
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)
        self.assertTrue(d.impide_recomendacion)
        self.assertIn("variable_disponible_tras_el_desenlace", {b.clave for b in d.bloqueos})

    def test_muestreo_se_declara_nunca_en_silencio(self):
        n = 500
        filas = [{"y": str(i % 2), "x": str(i)} for i in range(n)]
        analisis = {"columns": {"x": {"type": "number", "cardinality": n}}}
        problema = _problema(target="y", predictors=("x",))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas,
                             muestreo_maximo=100)
        self.assertTrue(d.muestreado)
        self.assertEqual(d.filas_totales, n)
        self.assertEqual(d.filas_medidas, 100)

    def test_sin_muestreo_por_debajo_del_techo(self):
        n = 50
        filas = [{"y": str(i % 2), "x": str(i)} for i in range(n)]
        analisis = {"columns": {"x": {"type": "number", "cardinality": n}}}
        problema = _problema(target="y", predictors=("x",))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)
        self.assertFalse(d.muestreado)
        self.assertEqual(d.filas_medidas, n)

    def test_diagnostico_a_json_separa_las_tres_listas(self):
        filas = [{"precio": str(i), "precio2": str(i)} for i in range(30)]
        analisis = {"columns": {"precio2": {"type": "number", "cardinality": 30}}}
        problema = _problema(target="precio", predictors=("precio2",))
        d = diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)
        j = d.a_json()
        self.assertIn("bloqueos", j)
        self.assertIn("sospechas", j)
        self.assertIn("limites", j)
        self.assertTrue(j["impide_recomendacion"])

    def test_csv_real_end_to_end_objetivo_duplicado(self):
        """El mismo caso 1, pero por la puerta de entrada REAL: texto CSV
        crudo, sin `analisis`/`filas` precalculados — para probar el
        cableado con `analyze_dataset_csv`/`_read_rows`, no solo la función."""
        filas_csv = ["precio,precio_de_venta,metros"]
        for i in range(30):
            filas_csv.append(f"{i * 1000},{i * 1000},{50 + i}")
        csv_text = "\n".join(filas_csv)
        problema = _problema(target="precio", predictors=("precio_de_venta", "metros"))
        d = diagnosticar_csv(csv_text, problema)
        self.assertTrue(d.impide_recomendacion)
        self.assertIn("objetivo_duplicado_confirmado", {b.clave for b in d.bloqueos})


if __name__ == "__main__":
    unittest.main()
