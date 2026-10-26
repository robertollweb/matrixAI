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
from pathlib import Path
from importlib.util import find_spec

from matrixai.estudio import ProblemSpec
from matrixai.training.diagnostico import (
    ALFA_PUREZA,
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
    UMBRAL_REDUCCION_PUREZA,
    UMBRAL_REDUCCION_PUREZA_FUERA_DE_MUESTRA,
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
    _prob_por_azar,
    _reduccion,
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
        alcanza pureza in-sample >= 0,85 por puro azar de muestra pequeña —
        la trampa exacta que el contrato prohíbe calificar como predictor."""
        random.seed(22)
        categorias = []
        for i in range(40):
            categorias += [f"cat_{i}"] * 20
        random.shuffle(categorias)
        objetivo = [random.choice(["si", "no"]) for _ in range(len(categorias))]

        r = pureza_categorica(categorias, objetivo)
        # La trampa se buscó con la pureza absoluta de entonces (0,85): con un
        # objetivo ~50/50 (tasa base de 'no' 0,504) esa pureza es una reducción
        # de 0,698, a un pelo del umbral nuevo; el fixture se describe como se
        # encontró.
        puras_in_sample = [c for c, info in r["categorias_evaluadas"].items()
                          if info["pureza"] >= 0.85]
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


# ---------------------------------------------------------------------------
# 138 — la pureza se mide CONTRA LA TASA BASE de su clase (hallado el 26-10):
# con un objetivo desbalanceado, el umbral absoluto 0,85 acusaba a TODA
# categoría. Aquí se prueban LOS DOS LADOS: que ya no acuse a las que solo
# reflejan la tasa base, y que siga acusando a las que de verdad delatan.
# ---------------------------------------------------------------------------

def _escenario(n, k, p_evento, rng, *, especial=None):
    """`n` filas, `k` categorías al azar, objetivo '1' (evento) con
    probabilidad `p_evento`. `especial = (categoria, P(evento | categoria))`
    cambia la probabilidad de UNA categoría."""
    categorias = [f"c{rng.randrange(k)}" for _ in range(n)]
    objetivo = []
    for c in categorias:
        p = especial[1] if especial and c == especial[0] else p_evento
        objetivo.append("1" if rng.random() < p else "0")
    return categorias, objetivo


class PurezaRelativaALaTasaBaseTest(unittest.TestCase):
    def test_a_objetivo_al_7_5_por_ciento_categorias_SIN_relacion_NO_salen(self):
        """EL DEFECTO: con 7,5 % de eventos, la clase 'no evento' es el 92,5 %
        de TODO el conjunto, así que cualquier categoría tiene pureza ~0,92
        sin saber nada. El umbral absoluto de antes (0,85) acusaba a `sexo` y
        `servicio` en casi cualquier CSV clínico. Medido con 200 repeticiones
        por celda sobre datos SIN relación: 100 % de falsos positivos con el
        criterio viejo y 0–1 % con el nuevo (ver el comentario del módulo)."""
        for semilla in range(8):
            rng = random.Random(semilla)
            for n, k in ((600, 2), (1000, 4), (2000, 20)):
                categorias, objetivo = _escenario(n, k, 0.075, rng)
                r = pureza_categorica(categorias, objetivo)
                evaluadas = r["categorias_evaluadas"].values()
                self.assertTrue(evaluadas)
                # el fixture MUERDE: con la regla de antes casi todas esas
                # categorías (pureza ≥ 0,85, la tasa base misma) eran «altas»...
                self.assertGreaterEqual(sum(i["pureza"] >= 0.85 for i in evaluadas), len(evaluadas) - 1)
                # ...y con la nueva ninguna reduce lo bastante el error de la tasa base.
                self.assertIsNone(proxy_por_pureza(categorias, objetivo, columna="sexo"), (semilla, n, k))

    def test_b_objetivo_50_50_una_categoria_que_delata_SI_sale(self):
        """El otro lado, con clases equilibradas (donde el umbral nuevo
        equivale al absoluto de antes: 0,85 ↔ reducción 0,70). La categoría
        `c0` da evento el 97 % de las veces."""
        rng = random.Random(1)
        categorias, objetivo = _escenario(1000, 10, 0.5, rng, especial=("c0", 0.97))
        s = proxy_por_pureza(categorias, objetivo, columna="servicio")
        self.assertIsInstance(s, Sospecha)
        self.assertNotIsInstance(s, Bloqueo)
        self.assertEqual(s.medida["categoria"], "c0")
        self.assertEqual(s.medida["categorias_evaluadas"]["c0"]["clase"], "1")
        self.assertGreaterEqual(s.medida["categorias_evaluadas"]["c0"]["reduccion"], UMBRAL_REDUCCION_PUREZA)

    def test_c_objetivo_al_7_5_una_categoria_casi_toda_evento_SI_sale(self):
        """Lo sospechoso de verdad con eventos raros: una categoría donde el
        95 % de las filas SON evento cuando en el conjunto lo son el 7,5 %. Su
        clase dominante es la MINORITARIA ('1'), y se mide contra SU tasa base
        (~0,13 con esa categoría dentro), no contra la de la mayoritaria."""
        for semilla in range(5):
            rng = random.Random(semilla)
            categorias, objetivo = _escenario(1000, 10, 0.075, rng, especial=("c0", 0.95))
            s = proxy_por_pureza(categorias, objetivo, columna="servicio")
            self.assertIsInstance(s, Sospecha, semilla)
            info = s.medida["categorias_evaluadas"][s.medida["categoria"]]
            self.assertEqual(s.medida["categoria"], "c0")
            self.assertEqual(info["clase"], "1")
            self.assertLess(info["tasa_base"], 0.2)
            self.assertGreaterEqual(info["reduccion"], UMBRAL_REDUCCION_PUREZA)

    def test_d_categoria_100_por_cien_NO_evento_con_soporte_suficiente_SI_sale(self):
        """DECISIÓN: una categoría donde el evento NUNCA ocurre, con filas de
        sobra, SÍ sale. Reduce el 100 % del error de la tasa base y eso no es
        azar (0,925¹⁵⁰ ≈ 1e-5): es la forma de «los fallecidos no reingresan» —
        una fuga clínica de manual—, no un reflejo de la tasa base. Lo que NO
        sale es la misma categoría con pocas filas (abajo): 20 filas todas
        'no evento' son azar corriente (0,925²⁰ = 21 %)."""
        rng = random.Random(3)
        categorias, objetivo = _escenario(1500, 10, 0.075, rng, especial=("c0", 0.0))   # ~150 filas en c0
        s = proxy_por_pureza(categorias, objetivo, columna="destino")
        self.assertIsInstance(s, Sospecha)
        info = s.medida["categorias_evaluadas"]["c0"]
        self.assertEqual((s.medida["categoria"], info["clase"], info["pureza"], info["reduccion"]),
                         ("c0", "0", 1.0, 1.0))
        self.assertGreaterEqual(info["n"], 100)

    def test_d_la_misma_categoria_100_por_cien_NO_evento_con_20_filas_es_AZAR_y_no_sale(self):
        """La contraparte de la anterior, medida: con 7,5 % de eventos, de cada
        100 categorías de 20 filas AL AZAR, ~21 salen enteras 'no evento'. Si
        salieran, una columna con 40 categorías pequeñas acusaría siempre."""
        rng = random.Random(4)
        sale_pura = 0
        for _ in range(100):
            categorias, objetivo = _escenario(400, 20, 0.075, rng)   # ~20 filas por categoría
            r = pureza_categorica(categorias, objetivo)
            sale_pura += any(i["reduccion"] >= UMBRAL_REDUCCION_PUREZA for i in r["categorias_evaluadas"].values())
            self.assertIsNone(proxy_por_pureza(categorias, objetivo, columna="x"))
        self.assertGreater(sale_pura, 50, "el fixture debe MORDER: sin la prueba de azar, casi siempre habría una")

    def test_la_confirmacion_fuera_de_muestra_sigue_siendo_necesaria(self):
        """Una categoría fuerte en conjunto que NO se sostiene en una de las
        mitades internas no se confirma, aunque pase el umbral y la prueba de
        azar (premisas afirmadas aquí, para que el fixture no sea ciego):
        mitad par 100 % 'no'; mitad impar 76 %."""
        filas = []
        k = nf = 0
        for i in range(500):
            if i % 10 == 0:
                filas.append(("c", "no"))
            elif i % 10 == 1:
                filas.append(("c", "si" if k % 50 < 12 else "no"))
                k += 1
            else:
                filas.append(("f", "no" if nf % 2 == 0 else "si"))
                nf += 1
        categorias = [c for c, _ in filas]
        objetivo = [o for _, o in filas]
        r = pureza_categorica(categorias, objetivo)
        info = r["categorias_evaluadas"]["c"]
        self.assertGreaterEqual(info["reduccion"], UMBRAL_REDUCCION_PUREZA)
        self.assertLessEqual(info["prob_por_azar"], r["alfa_por_categoria"])
        self.assertNotIn("c", r["confirmadas_fuera_de_muestra"])
        self.assertIsNone(proxy_por_pureza(categorias, objetivo, columna="x"))

    def test_el_umbral_es_de_la_REDUCCION_no_de_la_pureza_absoluta_los_dos_lados(self):
        """Objetivo 65/35. Una categoría de 300 filas con 88 % de la clase
        mayoritaria tiene pureza ABSOLUTA 0,88 (≥ 0,85: la regla de antes la
        acusaba), se confirma en las dos mitades y NO es azar (premisas abajo),
        pero solo reduce el error en 0,66 < 0,70: no sale. Con 92 % (reducción
        0,77) sí. Sin esta prueba el umbral absoluto sobrevivía en el filtro
        final, porque las mitades y la prueba de azar ya cortaban todo lo demás."""
        def caso(ceros_en_c):
            categorias = ["c"] * 300 + ["f"] * 700
            objetivo = (["0"] * ceros_en_c + ["1"] * (300 - ceros_en_c)
                        + ["0"] * (650 - ceros_en_c) + ["1"] * (700 - (650 - ceros_en_c)))
            orden = list(range(1000))
            random.Random(1).shuffle(orden)
            return [categorias[i] for i in orden], [objetivo[i] for i in orden]

        categorias, objetivo = caso(264)
        r = pureza_categorica(categorias, objetivo)
        info = r["categorias_evaluadas"]["c"]
        self.assertGreaterEqual(info["pureza"], 0.85)
        self.assertLess(UMBRAL_REDUCCION_PUREZA_FUERA_DE_MUESTRA, info["reduccion"])
        self.assertLess(info["reduccion"], UMBRAL_REDUCCION_PUREZA)
        self.assertLessEqual(info["prob_por_azar"], r["alfa_por_categoria"])
        self.assertIn("c", r["confirmadas_fuera_de_muestra"])
        self.assertIsNone(proxy_por_pureza(categorias, objetivo, columna="x"))

        categorias, objetivo = caso(276)
        s = proxy_por_pureza(categorias, objetivo, columna="x")
        self.assertIsInstance(s, Sospecha)
        self.assertGreaterEqual(s.medida["categorias_evaluadas"]["c"]["reduccion"], UMBRAL_REDUCCION_PUREZA)

    def test_la_tasa_base_es_la_de_la_clase_de_la_categoria_no_la_otra(self):
        """Una categoría con 60 % de eventos en un conjunto al 7,5 % tiene la
        clase dominante '1': su reducción es (0,6 − 0,075) / 0,925 = 0,57, NO
        (0,6 − 0,925) / 0,075, que sería negativa. Se afirma el número."""
        categorias = ["a"] * 100 + ["b"] * 900
        objetivo = (["1"] * 60 + ["0"] * 40) + (["1"] * 15 + ["0"] * 885)
        r = pureza_categorica(categorias, objetivo)
        base = 75 / 1000
        self.assertAlmostEqual(r["categorias_evaluadas"]["a"]["tasa_base"], base, places=6)
        self.assertEqual(r["categorias_evaluadas"]["a"]["clase"], "1")
        self.assertAlmostEqual(r["categorias_evaluadas"]["a"]["reduccion"], (0.6 - base) / (1 - base), places=5)
        self.assertEqual(r["categorias_evaluadas"]["b"]["clase"], "0")

    def test_reduccion_y_prob_por_azar_numeros_conocidos(self):
        self.assertAlmostEqual(_reduccion(0.85, 0.5), 0.7)              # el 0,85 de antes con 50/50
        self.assertAlmostEqual(_reduccion(0.925, 0.925), 0.0)           # reflejar la tasa base
        self.assertAlmostEqual(_reduccion(1.0, 0.925), 1.0)
        self.assertEqual(_reduccion(1.0, 1.0), 0.0)                     # objetivo de una sola clase
        from math import comb
        exacta = sum(comb(30, i) * 0.2 ** i * 0.8 ** (30 - i) for i in range(12, 31))
        self.assertAlmostEqual(_prob_por_azar(12, 30, 0.2), exacta, places=12)
        self.assertAlmostEqual(_prob_por_azar(20, 20, 0.925), 0.925 ** 20, places=12)
        self.assertEqual(_prob_por_azar(5, 100, 0.5), 1.0)              # por debajo de lo esperado
        self.assertEqual(ALFA_PUREZA, 0.01)
        self.assertEqual(UMBRAL_REDUCCION_PUREZA_FUERA_DE_MUESTRA, 0.5)

    def test_el_motivo_dice_los_porcentajes_en_cada_idioma_sin_jerga(self):
        rng = random.Random(1)
        categorias, objetivo = _escenario(1000, 10, 0.075, rng, especial=("c0", 0.95))
        s = proxy_por_pureza(categorias, objetivo, columna="servicio")
        info = s.medida["categorias_evaluadas"]["c0"]
        es, en = s.motivo["es"], s.motivo["en"]
        base = f"{info['tasa_base'] * 100:.1f}"
        pureza = f"{info['pureza'] * 100:.1f}".removesuffix(".0")
        self.assertIn(base.replace(".", ",") + " %", es)
        self.assertIn(base + "%", en)
        self.assertIn(pureza.replace(".", ",") + " %", es)
        self.assertIn(pureza + "%", en)
        for texto in (es, en):
            self.assertNotIn("0.9", texto)          # nada de proporciones crudas
            self.assertNotIn("{", texto)
            for jerga in ("binomial", "Bonferroni", "reducción", "reduction"):
                self.assertNotIn(jerga, texto)
        self.assertIn("'c0'", es)
        self.assertIn("'1'", es)


_DATOS_STUDIO = (Path(__file__).resolve().parents[2] / "matrixaistudio" / "frontend" / "tests"
                 / "datos" / "auditor-fuga-plantada.csv")
_COLUMNAS_FUGA = ["id_paciente", "edad", "sexo", "glucosa", "servicio", "copia_del_desenlace",
                  "puntuacion_posterior"]


def _csv_fuga_plantada(*, intercepto: float, filas: int = 600, semilla: int = 138) -> str:
    """El CSV del auditor del Studio (`generar-fuga-plantada.py`) con el
    intercepto de la logística como parámetro: -1,2 son 26 % de reingresos
    (el CSV original); -2,75, ~7,3 %. Mismas columnas y misma semilla."""
    import csv
    import io
    import math
    azar = random.Random(semilla)
    salida = io.StringIO()
    w = csv.writer(salida, lineterminator="\n")
    w.writerow(_COLUMNAS_FUGA + ["reingreso_30d"])
    for i in range(filas):
        edad = azar.randint(18, 90)
        glucosa = round(azar.gauss(100, 22), 1)
        z = intercepto + 0.035 * (edad - 55) + 0.02 * (glucosa - 100)
        y = 1 if azar.random() < 1 / (1 + math.exp(-z)) else 0
        w.writerow([f"P{100000 + i * 7}", edad, azar.choice("HM"), glucosa,
                    azar.choice(["urgencias", "cardiologia", "medicina_interna", "traumatologia"]),
                    y, round(y * 10 + azar.gauss(0, 0.6), 2), y])
    return salida.getvalue()


def _campos_con_pureza(csv_text: str) -> tuple[set[str], float]:
    problema = _problema_binaria(observation_unit="paciente", classes=("0", "1"), positive_label="1",
                                 target="reingreso_30d", predictors=tuple(_COLUMNAS_FUGA))
    d = diagnosticar_csv(csv_text, problema)
    lineas = csv_text.splitlines()[1:]
    eventos = sum(1 for linea in lineas if linea.endswith(",1")) / len(lineas)
    return {s.campo for s in d.sospechas if s.clave == "categoria_predictiva_por_pureza"}, eventos


class AuditorFugaPlantadaTest(unittest.TestCase):
    """e) La fuga plantada del auditor (138-C5): `copia_del_desenlace` sigue
    saliendo y las columnas honestas (`sexo`, `servicio`, `edad`, `glucosa`) no
    — con el CSV original (26 % de eventos) Y con su variante al 7,5 %."""

    def _honestas_no(self, campos):
        for honesta in ("sexo", "servicio", "edad", "glucosa"):
            self.assertNotIn(honesta, campos)

    def test_e_el_CSV_del_Studio_copia_si_y_las_honestas_no(self):
        if not _DATOS_STUDIO.exists():
            self.skipTest("el CSV del auditor vive en el Studio")
        campos, eventos = _campos_con_pureza(_DATOS_STUDIO.read_text(encoding="utf-8"))
        self.assertGreater(eventos, 0.2)
        self.assertIn("copia_del_desenlace", campos)
        self._honestas_no(campos)

    def test_e_el_generador_de_aqui_reproduce_el_CSV_del_Studio(self):
        """La variante no deriva del original: con el intercepto del original
        salen las mismas filas."""
        if not _DATOS_STUDIO.exists():
            self.skipTest("el CSV del auditor vive en el Studio")
        self.assertEqual(_csv_fuga_plantada(intercepto=-1.2), _DATOS_STUDIO.read_text(encoding="utf-8"))

    def test_e_variante_al_7_5_por_ciento_sexo_y_servicio_dejan_de_salir_y_la_copia_sigue(self):
        csv_text = _csv_fuga_plantada(intercepto=-2.75)
        campos, eventos = _campos_con_pureza(csv_text)
        self.assertTrue(0.065 <= eventos <= 0.085, eventos)
        self.assertIn("copia_del_desenlace", campos)
        self._honestas_no(campos)
        # el fixture MUERDE: con la pureza absoluta de antes, TODAS las
        # categorías de `sexo` y `servicio` tienen pureza ~0,92 (≥ 0,85).
        filas = [l.split(",") for l in csv_text.splitlines()[1:]]
        for columna in ("sexo", "servicio"):
            idx = _COLUMNAS_FUGA.index(columna)
            r = pureza_categorica([f[idx] for f in filas], [f[-1] for f in filas])
            self.assertTrue(r["categorias_evaluadas"], columna)
            self.assertTrue(all(i["pureza"] >= 0.85 for i in r["categorias_evaluadas"].values()), columna)
            # (la más cercana al umbral es `medicina_interna`, 0,667 < 0,70: con
            # 164 filas y 4 eventos de ~12 esperados; el azar también llega lejos.)
            self.assertTrue(all(i["reduccion"] < UMBRAL_REDUCCION_PUREZA
                                for i in r["categorias_evaluadas"].values()), columna)


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



class ElCosteDelCruceTest(unittest.TestCase):
    """«EL COSTE DEL CRUCE» (2026-09-26). `diagnosticar_csv` cruza todas las parejas de
    predictores buscando dos columnas iguales. El comentario del módulo prometía desde 103-C2
    que cada columna se preparaba UNA vez y citaba una prueba «que lo contrasta» que no
    existía; el orquestador preparaba las dos columnas en CADA cruce, y confirmar un CSV de
    51.000 × 100 tardaba 476 s (el cliente del Studio corta a los 30). Estas pruebas atan las
    dos cosas que lo hacen asumible, por RECUENTO y no por reloj, y la semántica de la
    comparación rápida contra la documentada."""

    @staticmethod
    def _csv(n_filas: int, n_columnas: int) -> tuple[list[dict[str, str]], dict, ProblemSpec]:
        # Columnas que difieren ENTRE SÍ ya en la primera fila (x_j = i + j): ninguna pareja
        # es igual, así que una comparación que se para en la primera diferencia lee 1 fila.
        predictores = tuple(f"x{j}" for j in range(n_columnas))
        filas = [{**{f"x{j}": str(i + j) for j in range(n_columnas)}, "y": str(i % 2)}
                 for i in range(n_filas)]
        analisis = {"columns": {p: {"type": "integer", "cardinality": n_filas} for p in predictores}}
        return filas, analisis, _problema(target="y", predictors=predictores)

    def test_cada_columna_se_prepara_UNA_vez_no_una_vez_por_cruce(self):
        from unittest import mock
        import matrixai.training.diagnostico as d
        filas, analisis, problema = self._csv(60, 12)
        with mock.patch.object(d, "_preparar_columna", wraps=d._preparar_columna) as espia:
            diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)
        # 12 predictores + el objetivo. Antes: 2 por cada una de las 66 parejas y 2 por cada
        # predictor contra el objetivo = 156.
        self.assertEqual(espia.call_count, 13)

    def test_la_comparacion_se_para_en_la_primera_diferencia(self):
        from unittest import mock
        import matrixai.training.diagnostico as d

        leidos = [0]

        class ListaQueCuenta(list):
            def __iter__(self):
                for valor in super().__iter__():
                    leidos[0] += 1
                    yield valor

        original = d._preparar_columna
        filas, analisis, problema = self._csv(500, 10)
        with mock.patch.object(d, "_preparar_columna",
                               side_effect=lambda v: ListaQueCuenta(original(v))):
            diagnosticar_csv("no-usado", problema, analisis=analisis, filas=filas)
        # 45 parejas + 10 contra el objetivo, cada una se para en su primera fila: 2 valores
        # leídos por comparación (uno de cada lado). Recorriéndolas enteras serían 55.000.
        self.assertLessEqual(leidos[0], 2 * (45 + 10) + 10, leidos[0])

    def test_la_comparacion_rapida_dice_lo_mismo_que_la_semantica_documentada(self):
        """La prueba que el comentario citaba y no existía: `_iguales_en_lo_comparable` contra
        una implementación directa de lo documentado (nulo no cuenta; dos números a menos de
        1e-9; dos textos, iguales; número contra texto, nunca), contando TODO."""
        from matrixai.training.dataset_analysis import _is_null
        from matrixai.training.diagnostico import _iguales_en_lo_comparable, _preparar_columna

        def leer(v):
            s = str(v).strip()
            try:
                return float(s)
            except ValueError:
                return s

        def referencia(a, b):
            coincidencias = comparadas = 0
            for x, y in zip(a, b):
                if _is_null(x) or _is_null(y):
                    continue
                comparadas += 1
                vx, vy = leer(x), leer(y)
                if isinstance(vx, float) and isinstance(vy, float):
                    coincidencias += abs(vx - vy) < 1e-9
                elif isinstance(vx, str) and isinstance(vy, str):
                    coincidencias += vx == vy
            return comparadas if coincidencias == comparadas else None

        valores = ["", "NA", "?", "null", "1", "1.0", " 1 ", "1e0", "2", "0.3",
                   "0.30000000000000004", "1.0000000001", "1.00000001", "abc", "ABC", " abc",
                   "nan", "inf", "-inf", "1,5", "si", "Sí"]
        azar = random.Random(26092026)
        iguales_de_verdad = 0
        for _ in range(4000):
            n = azar.randint(0, 25)
            a = [azar.choice(valores) for _ in range(n)]
            # La mitad de las veces, b es a con algún cambio (o ninguno): sin eso casi nunca
            # saldría una coincidencia total y la prueba solo miraría el «no».
            b = list(a) if azar.random() < 0.5 else [azar.choice(valores) for _ in range(n)]
            for _ in range(azar.choice([0, 0, 1, 2])):
                if n:
                    b[azar.randrange(n)] = azar.choice(valores)
            esperado = referencia(a, b)
            iguales_de_verdad += esperado is not None and esperado > 0
            self.assertEqual(_iguales_en_lo_comparable(_preparar_columna(a), _preparar_columna(b)),
                             esperado, (a, b))
        # Que el banco tenga dientes: salen coincidencias totales de verdad, no solo «no».
        self.assertGreater(iguales_de_verdad, 500)


if __name__ == "__main__":
    unittest.main()
