# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""108-C4 — validar el esquema de un CSV nuevo contra columnas esperadas.

Criterio de terminado (parcial, la pieza pura): "un CSV con columnas
cambiadas se rechaza con el motivo por columna" -- una entrada por columna
faltante, no una excepción de fichero entero.
"""
from __future__ import annotations

import unittest

from matrixai.training.preparacion import ajustar_preparacion
from matrixai.training.validacion_de_esquema import validar_esquema


def _politica_de_ejemplo():
    filas = [
        {"objetivo": "si", "edad": 30, "ciudad": "madrid"},
        {"objetivo": "no", "edad": 40, "ciudad": "bilbao"},
        {"objetivo": "si", "edad": 25, "ciudad": "madrid"},
    ]
    return ajustar_preparacion(filas, objetivo="objetivo", columnas=("edad", "ciudad"),
                               admite_categoricas=True, admite_faltantes=True)


class ValidacionDeEsquemaTest(unittest.TestCase):
    def test_mismas_columnas_pasa_sin_rechazos(self):
        v = validar_esquema(["edad", "ciudad"], politica=_politica_de_ejemplo())
        self.assertTrue(v.ok)
        self.assertEqual(v.rechazadas, ())

    def test_columna_esperada_ausente_se_rechaza_con_motivo_bilingue(self):
        v = validar_esquema(["edad"], politica=_politica_de_ejemplo())
        self.assertFalse(v.ok)
        self.assertEqual(len(v.rechazadas), 1)
        self.assertEqual(v.rechazadas[0].columna, "ciudad")
        self.assertIn("es", v.rechazadas[0].motivo)
        self.assertIn("en", v.rechazadas[0].motivo)
        self.assertNotEqual(v.rechazadas[0].motivo["es"], v.rechazadas[0].motivo["en"])

    def test_dos_columnas_ausentes_dan_dos_entradas_no_una_excepcion_de_fichero(self):
        """El caso literal del criterio: el motivo va POR COLUMNA."""
        v = validar_esquema(["row_id"], politica=_politica_de_ejemplo())
        self.assertFalse(v.ok)
        columnas_rechazadas = {r.columna for r in v.rechazadas}
        self.assertEqual(columnas_rechazadas, {"edad", "ciudad"})

    def test_columnas_de_mas_no_se_rechazan(self):
        """`row_id` u otra columna que el estudio nunca usó como predictor
        no tumba el fichero -- mismo criterio que `exigir_columnas_de_entrada`
        (106-C5): ignorar lo que sobra, no exigir un fichero exacto."""
        v = validar_esquema(["edad", "ciudad", "row_id", "nota_interna"],
                            politica=_politica_de_ejemplo())
        self.assertTrue(v.ok)

    def test_a_json_conserva_orden_y_motivo(self):
        v = validar_esquema([], politica=_politica_de_ejemplo())
        payload = v.a_json()
        self.assertFalse(payload["ok"])
        self.assertEqual([r["columna"] for r in payload["rechazadas"]], ["edad", "ciudad"])


if __name__ == "__main__":
    unittest.main()
