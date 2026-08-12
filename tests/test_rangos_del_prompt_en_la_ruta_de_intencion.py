# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los rangos por campo en la ruta de PROMPT.

Qué había (medido contra el endpoint del Studio el 2026-08-11): un
prompt normal —«clasificar si un envío llegará tarde según la distancia
en km y el peso en kg»— salía con `field_ranges` VACÍO. La gramática de
tipos exige corchetes (`distancia_km: Scalar en [0, 3000]`) y nadie
escribe eso al pedir un modelo.

Consecuencias reales, las dos medidas antes de tocar nada:

· sin rango declarado el core NO normaliza — la familia del contrato 61,
  donde un modelo de regresión puede colapsar a la media;
· la fase Prueba se queda sin escala: cajas de texto en vez de
  deslizadores, que fue como Roberto se lo encontró.

Y una tercera cosa que costó una vuelta: **la ruta que sale de un prompt
en prosa NO es la densa sino la del supervisor**. El primer arreglo se
puso en la densa y no cambió nada; una sonda lo dijo en un minuto.

Se comprueba con el LLM SUSTITUIDO: lo que se prueba es el cableado —a
quién se le pregunta, por qué campos, y qué se hace con la respuesta—,
no lo que conteste un modelo de lenguaje, que ni es determinista ni está
disponible en la suite.
"""
from __future__ import annotations

import unittest
from unittest import mock

from matrixai import playground


class RangosQueElPromptNoDeclaro(unittest.TestCase):
    def _resultado(self, **extra):
        base = {
            "field_ranges": {},
            "field_types": {},
            "field_categories": {},
            "visual_model": {
                "inputs": [
                    {"vector": "Envio", "field": "distancia_km", "type": "Scalar", "range": ""},
                    {"vector": "Envio", "field": "peso_kg", "type": "Scalar", "range": ""},
                ]
            },
        }
        base.update(extra)
        return base

    def test_se_piden_para_los_campos_SIN_rango(self) -> None:
        pedidos: list[list[str]] = []

        def falso(campos, contexto=""):
            pedidos.append(list(campos))
            return {"distancia_km": (0.0, 3000.0), "peso_kg": (0.0, 50.0)}

        r = self._resultado()
        with mock.patch.object(playground, "_llm_field_ranges", falso):
            playground._completar_rangos_del_prompt(r, "clasificar envíos tardíos")

        self.assertEqual(pedidos, [["distancia_km", "peso_kg"]])
        self.assertEqual(r["field_ranges"], {"distancia_km": (0.0, 3000.0), "peso_kg": (0.0, 50.0)})

    def test_LO_DICHO_EN_EL_PROMPT_MANDA(self) -> None:
        """Un rango declarado no se toca, y ni siquiera se pregunta por él."""
        pedidos: list[list[str]] = []

        def falso(campos, contexto=""):
            pedidos.append(list(campos))
            return {"peso_kg": (0.0, 50.0)}

        r = self._resultado(field_ranges={"distancia_km": (0.0, 3000.0)})
        with mock.patch.object(playground, "_llm_field_ranges", falso):
            playground._completar_rangos_del_prompt(r, "x")

        self.assertEqual(pedidos, [["peso_kg"]], "no se pregunta por lo ya declarado")
        self.assertEqual(r["field_ranges"]["distancia_km"], (0.0, 3000.0))

    def test_se_declara_DE_DONDE_sale_cada_uno(self) -> None:
        with mock.patch.object(playground, "_llm_field_ranges", lambda c, ctx="": {"peso_kg": (0.0, 50.0)}):
            r = self._resultado(field_ranges={"distancia_km": (0.0, 3000.0)})
            playground._completar_rangos_del_prompt(r, "x")
        self.assertEqual(
            r["field_ranges_source"],
            {"distancia_km": "prompt", "peso_kg": "llm_proposed"},
        )

    def test_SIN_LLM_no_se_inventa_nada(self) -> None:
        # `_llm_field_ranges` ya devuelve {} cuando no hay proveedor; lo que
        # se comprueba aquí es que eso deja el resultado EXACTAMENTE como
        # estaba, sin claves nuevas que la interfaz leería como un dominio.
        r = self._resultado()
        with mock.patch.object(playground, "_llm_field_ranges", lambda c, ctx="": {}):
            playground._completar_rangos_del_prompt(r, "x")
        self.assertEqual(r["field_ranges"], {})
        self.assertNotIn("field_ranges_source", r)

    def test_un_nombre_GENERICO_no_tiene_dominio_que_deducir(self) -> None:
        """`feature_1` no nombra nada del mundo.

        Pedirle a un LLM el rango de `feature_1` es pedirle que se lo
        invente, y un rango inventado NORMALIZA de verdad: la predicción
        saldría mal en silencio. Mejor sin escala que con una falsa.
        """
        pedidos: list[list[str]] = []
        r = self._resultado(visual_model={"inputs": [
            {"vector": "Input", "field": "feature_1", "type": "Scalar"},
            {"vector": "Input", "field": "input_2", "type": "Scalar"},
            {"vector": "Input", "field": "x3", "type": "Scalar"},
        ]})
        with mock.patch.object(playground, "_llm_field_ranges",
                               lambda c, ctx="": pedidos.append(list(c)) or {}):
            playground._completar_rangos_del_prompt(r, "x")
        self.assertEqual(pedidos, [], "no se pregunta por nombres sin significado")

    def test_las_CATEGORICAS_y_los_BOOLEANOS_quedan_fuera(self) -> None:
        # No tienen rango: tienen vocabulario o dos valores. Pedir un
        # rango para ellos y guardarlo los normalizaría como si fueran
        # escalares.
        pedidos: list[list[str]] = []
        r = self._resultado(
            field_categories={"distancia_km": ["corta", "larga"]},
            field_types={"peso_kg": "boolean"},
        )
        with mock.patch.object(playground, "_llm_field_ranges",
                               lambda c, ctx="": pedidos.append(list(c)) or {}):
            playground._completar_rangos_del_prompt(r, "x")
        self.assertEqual(pedidos, [], pedidos)

    def test_lo_que_el_LLM_devuelva_de_MAS_se_ignora(self) -> None:
        # Un modelo de lenguaje puede contestar por un campo que no se le
        # preguntó. Guardarlo metería en el esquema una columna que el
        # modelo no tiene.
        r = self._resultado()
        with mock.patch.object(playground, "_llm_field_ranges",
                               lambda c, ctx="": {"distancia_km": (0.0, 3000.0), "inventado": (0.0, 1.0)}):
            playground._completar_rangos_del_prompt(r, "x")
        self.assertEqual(list(r["field_ranges"]), ["distancia_km"])

    def test_el_PROMPT_viaja_como_contexto(self) -> None:
        """Sin el prompt, el LLM propone rangos genéricos: es la
        diferencia entre «un peso» y «el peso de un envío»."""
        visto: dict[str, str] = {}

        def falso(campos, contexto=""):
            visto["contexto"] = contexto
            return {}

        with mock.patch.object(playground, "_llm_field_ranges", falso):
            playground._completar_rangos_del_prompt(self._resultado(), "clasificar envíos tardíos")
        self.assertEqual(visto["contexto"], "clasificar envíos tardíos")

class YLaRutaEnteraLoUsa(unittest.TestCase):
    """Que la función exista no basta: hay que LLAMARLA.

    Escrito porque el revert-restore lo cazó — borrando la llamada de
    `analyze_playground_request`, las ocho pruebas de arriba seguían
    verdes. Probar la función no es probar el producto.

    El prompt NOMBRA sus campos con la gramática de tipos (sin rango) por
    dos razones: así la ruta determinista los conserva con su nombre real
    —sin LLM los llamaría `feature_1`, y a esos no se les inventa dominio
    a propósito—, y así se comprueba justo el caso de Roberto: el prompt
    dice QUÉ campos hay y no dice entre qué valores se mueven.
    """

    PROMPT = (
        "Clasificar el riesgo de impago. "
        "Campos: ingresos_mensuales: Scalar, antiguedad_laboral: Scalar"
    )

    def test_un_modelo_pedido_por_prompt_SALE_con_sus_rangos(self) -> None:
        with mock.patch.object(
            playground, "_llm_field_ranges",
            lambda campos, contexto="": {c: (0.0, 100.0) for c in campos},
        ):
            r = playground.analyze_playground_request(
                {"mode": "prompt", "prompt": self.PROMPT, "locale": "es"}
            )
        self.assertEqual(
            r["field_ranges"],
            {"ingresos_mensuales": (0.0, 100.0), "antiguedad_laboral": (0.0, 100.0)},
        )
        self.assertEqual(
            r["field_ranges_source"],
            {"ingresos_mensuales": "llm_proposed", "antiguedad_laboral": "llm_proposed"},
        )

    def test_y_SIN_LLM_el_modelo_sale_igual_que_antes(self) -> None:
        # La red se construye lo mismo: esto rellena metadatos, no cambia
        # la arquitectura ni el .mxai.
        with mock.patch.object(playground, "_llm_field_ranges", lambda c, ctx="": {}):
            r = playground.analyze_playground_request(
                {"mode": "prompt", "prompt": self.PROMPT, "locale": "es"}
            )
        self.assertEqual(r["field_ranges"], {})
        self.assertTrue(r.get("mxai"))


if __name__ == "__main__":
    unittest.main()
