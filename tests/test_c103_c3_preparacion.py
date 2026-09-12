# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C3 — preparación dentro del entrenamiento.

Criterio de terminado: «Alterar valores extremos solo en test no cambia
medianas, rangos ni vocabulario aprendido. Categorías desconocidas y
faltantes producen transformaciones definidas y equivalentes fuera de
Studio. Un ejemplo con 40 % y otro con 60 % valida la regla de aviso
elegida. La política queda en preparación ajustada y diagnóstico.»
"""
from __future__ import annotations

import unittest

from matrixai.training.preparacion import (
    CATEGORIA_DESCONOCIDA,
    CATEGORIA_FALTANTE,
    MINIMO_FILAS_SIN_AVISO,
    UMBRAL_AVISO_FALTANTES,
    PoliticaDePreparacion,
    ajustar_preparacion,
    tipar_columnas_numericas,
    transformar_fila,
)
from matrixai.training.preparacion_textos import IDIOMAS, MOTIVOS, huecos_de, motivo


class TestElCatalogoHablaLosDosIdiomas(unittest.TestCase):
    def test_toda_clave_tiene_los_dos_idiomas(self):
        for clave, textos in MOTIVOS.items():
            for idioma in IDIOMAS:
                self.assertIn(idioma, textos, f"{clave} sin {idioma}")

    def test_los_huecos_son_los_mismos_en_los_dos_idiomas(self):
        for clave, textos in MOTIVOS.items():
            self.assertEqual(huecos_de(textos["es"]), huecos_de(textos["en"]),
                             f"{clave}: los huecos no coinciden")

    def test_clave_desconocida_se_rechaza(self):
        with self.assertRaises(KeyError):
            motivo("no_existe")


def _filas(n_con_objetivo=120, incluir_sin_objetivo=0):
    filas = []
    for i in range(n_con_objetivo):
        filas.append({"objetivo": "si" if i % 3 == 0 else "no",
                      "edad": float(20 + (i % 50)), "barrio": "centro" if i % 2 == 0 else "sur"})
    for i in range(incluir_sin_objetivo):
        filas.append({"objetivo": None, "edad": 99.0, "barrio": "norte"})
    return filas


class CriterioLiteralTest(unittest.TestCase):
    def test_valores_extremos_en_test_no_cambian_mediana_ni_vocabulario(self):
        train = _filas()
        politica = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad", "barrio"],
                                       admite_categoricas=False, admite_faltantes=False)
        mediana_antes = politica.columna("edad").mediana
        categorias_antes = politica.columna("barrio").categorias_conocidas

        fila_extrema = {"edad": 10_000_000.0, "barrio": "categoria_jamas_vista_en_train"}
        transformar_fila(fila_extrema, politica)

        self.assertEqual(politica.columna("edad").mediana, mediana_antes)
        self.assertEqual(politica.columna("barrio").categorias_conocidas, categorias_antes)

    def test_categoria_desconocida_faltante_y_referencia_se_diferencian(self):
        """El texto literal: «diferenciar categoría desconocida, faltante y
        categoría de referencia» -- tres cosas, tres representaciones
        distintas, nunca colapsadas en un mismo "otro"."""
        train = _filas()
        politica = ajustar_preparacion(train, objetivo="objetivo", columnas=["barrio"],
                                       admite_categoricas=False, admite_faltantes=False)
        propuesta = politica.columna("barrio")
        self.assertIsNotNone(propuesta.categoria_de_referencia)
        self.assertIn(propuesta.categoria_de_referencia, propuesta.categorias_conocidas)

        conocida = transformar_fila({"barrio": "centro"}, politica)["barrio"]
        desconocida = transformar_fila({"barrio": "jamas_visto"}, politica)["barrio"]
        faltante = transformar_fila({"barrio": None}, politica)["barrio"]

        self.assertEqual(conocida, "centro")
        self.assertEqual(desconocida, CATEGORIA_DESCONOCIDA)
        self.assertEqual(faltante, CATEGORIA_FALTANTE)
        self.assertEqual(len({conocida, desconocida, faltante}), 3)

    def test_aviso_de_faltantes_40_no_dispara_60_si(self):
        base = {"objetivo": "si"}
        filas_40 = ([dict(base, x=1.0) for _ in range(60)]
                   + [dict(base, x=None) for _ in range(40)])
        filas_60 = ([dict(base, x=1.0) for _ in range(40)]
                   + [dict(base, x=None) for _ in range(60)])

        p40 = ajustar_preparacion(filas_40, objetivo="objetivo", columnas=["x"],
                                  admite_categoricas=False, admite_faltantes=False)
        p60 = ajustar_preparacion(filas_60, objetivo="objetivo", columnas=["x"],
                                  admite_categoricas=False, admite_faltantes=False)

        self.assertAlmostEqual(p40.columna("x").proporcion_faltante, 0.40)
        self.assertAlmostEqual(p60.columna("x").proporcion_faltante, 0.60)
        self.assertFalse(any(l.clave == "faltantes_por_encima_del_umbral" for l in p40.limites))
        self.assertTrue(any(l.clave == "faltantes_por_encima_del_umbral" for l in p60.limites))

    def test_no_bloquea_con_menos_de_100_filas(self):
        """Texto literal: «no bloquear automáticamente toda recomendación
        con menos de 100 filas» -- ni excepción ni política vacía, un
        `Limite` orientativo."""
        train_pequeno = _filas(n_con_objetivo=50)
        politica = ajustar_preparacion(train_pequeno, objetivo="objetivo", columnas=["edad"],
                                       admite_categoricas=False, admite_faltantes=False)
        self.assertEqual(politica.filas_de_train_efectivas, 50)
        self.assertLess(politica.filas_de_train_efectivas, MINIMO_FILAS_SIN_AVISO)
        self.assertTrue(any(l.clave == "pocas_filas_para_preparacion" for l in politica.limites))
        self.assertIsNotNone(politica.columna("edad"))


class ExclusionDelObjetivoTest(unittest.TestCase):
    def test_fila_sin_objetivo_se_excluye_con_recuento(self):
        train = _filas(n_con_objetivo=120, incluir_sin_objetivo=7)
        politica = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad"],
                                       admite_categoricas=False, admite_faltantes=False)
        self.assertEqual(politica.filas_excluidas_sin_objetivo, 7)
        self.assertEqual(politica.filas_de_train_efectivas, 120)

    def test_el_objetivo_nunca_se_imputa(self):
        """El objetivo no es una columna de `columnas`: no se le ajusta
        mediana ni vocabulario, se usa solo para filtrar filas."""
        train = _filas()
        politica = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad"],
                                       admite_categoricas=False, admite_faltantes=False)
        self.assertIsNone(politica.columna("objetivo"))


class CapacidadDelMotorTest(unittest.TestCase):
    def test_admite_faltantes_deja_marcador_nativo_en_vez_de_imputar(self):
        train = _filas()
        politica = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad"],
                                       admite_categoricas=False, admite_faltantes=True)
        transformada = transformar_fila({"edad": None}, politica)
        self.assertIsNone(transformada["edad"])
        self.assertEqual(transformada["edad__faltante"], 1.0)

    def test_sin_admite_faltantes_imputa_la_mediana(self):
        train = _filas()
        politica = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad"],
                                       admite_categoricas=False, admite_faltantes=False)
        transformada = transformar_fila({"edad": None}, politica)
        self.assertEqual(transformada["edad"], politica.columna("edad").mediana)
        self.assertEqual(transformada["edad__faltante"], 1.0)

    def test_capacidad_categorica_se_registra_en_la_propuesta(self):
        train = _filas()
        con_nativo = ajustar_preparacion(train, objetivo="objetivo", columnas=["barrio"],
                                         admite_categoricas=True, admite_faltantes=False)
        sin_nativo = ajustar_preparacion(train, objetivo="objetivo", columnas=["barrio"],
                                         admite_categoricas=False, admite_faltantes=False)
        self.assertTrue(con_nativo.columna("barrio").admite_nativo)
        self.assertFalse(sin_nativo.columna("barrio").admite_nativo)


class TransformNuncaReajustaTest(unittest.TestCase):
    def test_transformar_muchas_filas_no_cambia_la_politica(self):
        train = _filas()
        politica = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad", "barrio"],
                                       admite_categoricas=False, admite_faltantes=False)
        original = politica.a_json()
        for i in range(200):
            transformar_fila({"edad": float(i) * 1000, "barrio": f"nueva_{i}"}, politica)
        self.assertEqual(politica.a_json(), original)


class RoundTripJsonTest(unittest.TestCase):
    """108-C4: sin `desde_json()` no hay forma de recuperar la política de
    preparación del candidato ganador desde el sobre persistido en disco
    (`_persistir_seleccion`/`cargar_seleccion`, `estudio_job.py`) -- el
    checkpoint del `Motor` por sí solo no basta para transformar una fila
    nueva llegada por "Usar"."""

    def test_a_json_desde_json_reconstruye_una_politica_equivalente(self):
        train = _filas(n_con_objetivo=120, incluir_sin_objetivo=5)
        original = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad", "barrio"],
                                       admite_categoricas=True, admite_faltantes=True)
        reconstruida = PoliticaDePreparacion.desde_json(original.a_json())

        self.assertEqual(reconstruida.a_json(), original.a_json())

        for fila in ({"edad": 35.0, "barrio": "centro"},
                    {"edad": None, "barrio": "jamas_visto"},
                    {"edad": 10_000_000.0, "barrio": None}):
            self.assertEqual(transformar_fila(fila, reconstruida), transformar_fila(fila, original))

    def test_reconstruida_no_reajusta_con_filas_nuevas(self):
        """Mismo invariante que `TransformNuncaReajustaTest` de arriba,
        pero sobre la política QUE PASÓ por JSON -- una reconstrucción a
        medias (p. ej. `categorias_conocidas` como lista en vez de tupla)
        podría parecer correcta y romper en el primer uso real."""
        train = _filas()
        original = ajustar_preparacion(train, objetivo="objetivo", columnas=["edad", "barrio"],
                                       admite_categoricas=False, admite_faltantes=False)
        reconstruida = PoliticaDePreparacion.desde_json(original.a_json())
        antes = reconstruida.a_json()
        for i in range(200):
            transformar_fila({"edad": float(i) * 1000, "barrio": f"nueva_{i}"}, reconstruida)
        self.assertEqual(reconstruida.a_json(), antes)

    def test_categorias_conocidas_reconstruidas_son_tupla_no_lista(self):
        train = _filas()
        original = ajustar_preparacion(train, objetivo="objetivo", columnas=["barrio"],
                                       admite_categoricas=False, admite_faltantes=False)
        reconstruida = PoliticaDePreparacion.desde_json(original.a_json())
        self.assertIsInstance(reconstruida.columna("barrio").categorias_conocidas, tuple)
        self.assertIsInstance(reconstruida.columnas, tuple)
        self.assertIsInstance(reconstruida.limites, tuple)

    def test_limites_se_reconstruyen_tambien(self):
        filas_60 = ([{"objetivo": "si", "x": 1.0} for _ in range(40)]
                   + [{"objetivo": "si", "x": None} for _ in range(60)])
        original = ajustar_preparacion(filas_60, objetivo="objetivo", columnas=["x"],
                                       admite_categoricas=False, admite_faltantes=False)
        self.assertTrue(original.limites)
        reconstruida = PoliticaDePreparacion.desde_json(original.a_json())
        self.assertEqual([l.a_json() for l in reconstruida.limites],
                         [l.a_json() for l in original.limites])


if __name__ == "__main__":
    unittest.main()


class ColumnasDeSalidaTest(unittest.TestCase):
    """Auditoría propia 2026-09-11 — la política declara lo que produce.

    Sin esto, cada llamante reconstruía el sufijo `__faltante` por su cuenta
    para saber qué columnas le llegan de verdad a un motor, o —peor— no lo
    reconstruía y el motor las descartaba en silencio por no estar entre los
    `predictors` declarados del problema (102-C2).
    """

    def _filas(self):
        return [{"x": 1.0, "cat": "rojo", "y": "si"},
                {"x": None, "cat": "azul", "y": "no"},
                {"x": 3.0, "cat": "rojo", "y": "si"}]

    def test_declara_exactamente_lo_que_transformar_fila_emite(self):
        """El aserto que de verdad importa: no una lista escrita a mano, sino
        la igualdad con lo que la OTRA función produce. Si una de las dos
        cambia sin la otra, esto se pone rojo."""
        for admite in (True, False):
            with self.subTest(admite_faltantes=admite):
                politica = ajustar_preparacion(
                    self._filas(), objetivo="y", columnas=("x", "cat"),
                    admite_categoricas=admite, admite_faltantes=admite)
                emitidas = set(transformar_fila(self._filas()[0], politica))
                self.assertEqual(set(politica.columnas_de_salida()), emitidas)

    def test_el_indicador_de_faltantes_esta_incluido(self):
        """El caso concreto que motivó esto: el indicador se emite para los dos
        motores, con independencia de `admite_faltantes` (medido), así que
        quien declare predictores al motor tiene que llevarlo."""
        politica = ajustar_preparacion(
            self._filas(), objetivo="y", columnas=("x", "cat"),
            admite_categoricas=True, admite_faltantes=True)
        self.assertIn("x__faltante", politica.columnas_de_salida())
        # Y la categórica NO lleva indicador: un `None` categórico ya viaja
        # como `__faltante__`, su propia categoría.
        self.assertNotIn("cat__faltante", politica.columnas_de_salida())

    def test_no_incluye_el_objetivo(self):
        politica = ajustar_preparacion(
            self._filas(), objetivo="y", columnas=("x", "cat"),
            admite_categoricas=True, admite_faltantes=True)
        self.assertNotIn("y", politica.columnas_de_salida())


class ColisionDelIndicadorTest(unittest.TestCase):
    """REAUDITORÍA 2026-09-11 — una regresión introducida por el arreglo del
    mismo día, encontrada por un auditor independiente.

    Si el CSV ya trae una columna llamada `x__faltante` junto a una `x`
    numérica, el indicador derivado de `x` colisiona con ella:
    `transformar_fila` escribe en un `dict` y se queda con tres claves,
    mientras la primera versión de `columnas_de_salida()` declaraba cuatro con
    una repetida. `ProblemSpec` prohíbe predictores repetidos, así que el
    estudio del producto respondía **HTTP 500 sin motivo** en un caso que
    antes arrancaba.
    """

    def _filas(self):
        return [{"x": 1.0, "x__faltante": 5.0, "y": "si"},
                {"x": None, "x__faltante": 6.0, "y": "no"},
                {"x": 3.0, "x__faltante": 7.0, "y": "si"}]

    def _politica(self):
        return ajustar_preparacion(self._filas(), objetivo="y",
                                   columnas=("x", "x__faltante"),
                                   admite_categoricas=True, admite_faltantes=True)

    def test_no_declara_columnas_repetidas(self):
        salida = self._politica().columnas_de_salida()
        self.assertEqual(len(salida), len(set(salida)), salida)

    def test_y_sigue_coincidiendo_con_lo_que_se_emite(self):
        politica = self._politica()
        self.assertEqual(set(politica.columnas_de_salida()),
                         set(transformar_fila(self._filas()[0], politica)))

    def test_la_colision_en_si_queda_DECLARADA_no_arreglada(self):
        """Quién gana la colisión, MEDIDO — no supuesto.

        Mi primera versión de este test afirmaba lo contrario (que el
        indicador derivado pisaba la columna real del CSV) y se puso en rojo:
        el aserto estaba mal, no el producto. Lo que pasa de verdad es que las
        propuestas se recorren en orden, `x` escribe su indicador `x__faltante`
        = 0.0 y luego la propuesta de la columna REAL `x__faltante` lo
        sobreescribe con su valor (5.0). Es decir: **el dato del usuario
        sobrevive y lo que se pierde en silencio es el indicador derivado de
        `x`** — un motor que dependa de ese indicador recibe el valor de otra
        columna creyendo que es un 0/1.

        Sigue sin arreglarse a propósito (cambiar el nombre del indicador
        afectaría a todo lo ya ajustado); se fija aquí para que si alguien lo
        cambia, se entere de cuál de las dos cosas está cambiando."""
        transformada = transformar_fila(self._filas()[0], self._politica())
        self.assertEqual(transformada["x__faltante"], 5.0)
        # Y el indicador de la columna real sí se genera con su propio nombre.
        self.assertEqual(transformada["x__faltante__faltante"], 0.0)


class TiparColumnasNumericasTest(unittest.TestCase):
    """101-C3 / 103-C3, cableado — el choque de tipos que costó un dataset
    entero en la pasada exploratoria de Fase 0.

    `ajustar_preparacion` NO parsea texto, por diseño del núcleo: que una
    columna «parezca» numérica no es lo mismo que serlo, y adivinar ahí sería
    justo lo que este proyecto prohíbe. Pero los MOTORES sí parsean. Cuando la
    fuente entrega texto (un `csv.DictReader`, o un ARFF con columnas
    nominales de valores `"0"`/`"1"`), los dos lados tipan distinto: el núcleo
    ve categórica, mete el centinela `__desconocida__` para una categoría no
    vista en train, y el motor —que la ve numérica— revienta con
    `float("__desconocida__")`.

    Medido sobre `Internet-Advertisements` (2026-09-12): **1372 centinelas en
    101 columnas** sin esta función, **0** con ella. Eso costó 6 intentos, y
    con ellos el dataset entero para la regla de cierre del 101-C1, que cuenta
    un fallo como dataset perdido.

    La reparación existía en el Studio desde el 09 y el camino de benchmarks
    nunca la recibió. Vive aquí para que no haya una tercera copia.
    """

    def test_una_columna_de_texto_que_TODO_parsea_se_convierte(self):
        filas = [{"a": "0"}, {"a": "1"}, {"a": "2"}]
        self.assertEqual(tipar_columnas_numericas(filas, ("a",)), ("a",))
        self.assertEqual([f["a"] for f in filas], [0.0, 1.0, 2.0])

    def test_una_categorica_DE_VERDAD_se_deja_tal_cual(self):
        """La otra mitad. Sin ella, la función la pasaría una versión que
        convierte todo y rompe las categóricas de verdad, que es un daño mayor
        que el que repara."""
        filas = [{"a": "rojo"}, {"a": "azul"}]
        self.assertEqual(tipar_columnas_numericas(filas, ("a",)), ())
        self.assertEqual([f["a"] for f in filas], ["rojo", "azul"])

    def test_UN_solo_valor_no_numerico_deja_la_columna_ENTERA_como_esta(self):
        """«Todos o ninguno», no valor a valor: convertir la mitad dejaría una
        columna con floats y textos mezclados, que es peor que cualquiera de
        las dos cosas."""
        filas = [{"a": "0"}, {"a": "1"}, {"a": "N/A"}]
        self.assertEqual(tipar_columnas_numericas(filas, ("a",)), ())
        self.assertEqual([f["a"] for f in filas], ["0", "1", "N/A"])

    def test_los_huecos_no_cuentan_ni_se_rellenan(self):
        filas = [{"a": "0"}, {"a": ""}, {"a": None}, {"a": "2"}]
        self.assertEqual(tipar_columnas_numericas(filas, ("a",)), ("a",))
        self.assertEqual([f["a"] for f in filas], [0.0, "", None, 2.0])

    def test_una_columna_YA_tipada_no_se_declara_tocada(self):
        """Declarar que se tocó algo que no se tocó es declarar lo que se
        pidió y no lo que pasó: quien lea el retorno no puede distinguir un
        CSV de texto de un DataFrame ya tipado."""
        filas = [{"a": 1.5}, {"a": 2.5}]
        self.assertEqual(tipar_columnas_numericas(filas, ("a",)), ())

    def test_una_columna_vacia_entera_no_estalla(self):
        filas = [{"a": None}, {"a": ""}]
        self.assertEqual(tipar_columnas_numericas(filas, ("a",)), ())

    def test_solo_toca_las_columnas_QUE_SE_LE_PIDEN(self):
        """El desenlace no es un predictor y no se tipa por su cuenta: en una
        clasificación con clases `"0"`/`"1"` convertirlas a float cambiaría
        las etiquetas del problema."""
        filas = [{"a": "0", "y": "1"}, {"a": "1", "y": "0"}]
        tipar_columnas_numericas(filas, ("a",))
        self.assertEqual([f["y"] for f in filas], ["1", "0"])

    def test_el_CENTINELA_desaparece_del_camino_completo(self):
        """Por el producto y no solo por la función: se recorre
        `ajustar_preparacion` + `transformar_fila` de verdad, con una columna
        de texto cuyo valor nuevo solo aparece fuera de train. Es la forma
        exacta en que el fallo se manifestó."""
        train = [{"row_id": str(i), "ind": "0", "y": "si" if i % 2 else "no"}
                 for i in range(40)]
        fuera = [{"row_id": "99", "ind": "1", "y": "si"}]   # "1" nunca visto en train

        sin_tipar = [dict(f) for f in train + fuera]
        politica = ajustar_preparacion(sin_tipar[:40], objetivo="y", columnas=("ind",),
                                       admite_categoricas=True, admite_faltantes=True)
        centinelas = sum(1 for f in sin_tipar
                         for v in transformar_fila(f, politica).values()
                         if v == "__desconocida__")
        self.assertGreater(centinelas, 0, "el fallo original ya no se reproduce")

        con_tipar = [dict(f) for f in train + fuera]
        tipar_columnas_numericas(con_tipar, ("ind",))
        politica2 = ajustar_preparacion(con_tipar[:40], objetivo="y", columnas=("ind",),
                                        admite_categoricas=True, admite_faltantes=True)
        centinelas2 = sum(1 for f in con_tipar
                          for v in transformar_fila(f, politica2).values()
                          if v == "__desconocida__")
        self.assertEqual(centinelas2, 0)
