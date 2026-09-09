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
