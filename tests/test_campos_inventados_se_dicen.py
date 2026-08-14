# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Si los campos del modelo no salieron de tu descripción, se dice.

POR QUÉ EXISTE, medido el 2026-08-14 con `use_llm=False` —que es lo que
tiene una instalación recién hecha, sin clave de LLM—: «clasifica reseñas
por sentimiento» devolvía

    VECTOR Item[5]
      priority
      trust
      topic_a
      topic_b
      confidence

cinco campos que no tienen nada que ver con lo pedido, `understanding`
vacío, y **ni una palabra de aviso**.

La causa no es el texto. `PromptAgent._extract_fields` solo reconoce la
sintaxis «campos: a, b», así que casi cualquier petición en prosa acaba
con los campos de relleno de una plantilla — «a partir de la
distancia_km y el peso_kg» tampoco extrae nada. El caso del texto es un
caso particular de este.

Y por eso el aviso dice «estos campos no los has dicho tú» y NO «tu
descripción menciona texto»: lo primero es un HECHO que el generador
conoce (usó la plantilla porque no extrajo nada) y no puede dar un falso
positivo; lo segundo exigiría una lista de palabras de dominio que
acusaría a «predecir el NÚMERO de reseñas», que es tabular.

Roberto, en su orden de prioridades: «no sabe qué dataset necesita,
recibe un modelo inesperado — ése es el roadmap».
"""
from __future__ import annotations

import json
import unittest

from matrixai.agents.prompt import PromptAgent
from matrixai.playground import analyze_playground_request, campos_inventados_warning

_FUNCIONALES_ES = (" el ", " la ", " los ", " de ", " del ", " con ", " que ",
                   " para ", " una ", " tu ", " tus ", " son ")


def _generar(prompt: str, locale: str = "es") -> str:
    """La respuesta ENTERA, serializada. Se barre todo y no una clave.

    El aviso viaja en `warnings` dentro de las etapas del pipeline, no en
    una clave de primer nivel; buscarlo por una clave concreta es cómo se
    da por ausente algo que está. Barrer entero no depende de acertar.
    """
    r = analyze_playground_request({"prompt": prompt, "use_llm": False, "locale": locale})
    return json.dumps(r, ensure_ascii=False, default=str)


class LosCamposQueNadieNombroSeDicen(unittest.TestCase):

    def test_el_agente_sigue_sin_extraer_campos_de_la_prosa(self):
        """UN FIXTURE DESCRIBE ALGO QUE EXISTE — y aquí, algo que PASA.

        Todo esto se sostiene sobre que «clasifica reseñas por
        sentimiento» no dé ningún campo. Si algún día `_extract_fields`
        aprendiera a leer prosa, este aserto se pondría rojo y avisaría
        de que las pruebas de abajo ya no miden lo que creen — en vez de
        seguir pasando en verde por la razón equivocada.
        """
        s = PromptAgent().synthesize("clasifica reseñas por sentimiento")
        self.assertTrue(s.fields_invented)
        self.assertEqual(s.selected_fields,
                         ["priority", "trust", "topic_a", "topic_b", "confidence"])

    def test_con_los_campos_inventados_lo_dice_y_los_NOMBRA(self):
        cuerpo = _generar("clasifica reseñas por sentimiento")
        self.assertIn("NO los has dicho", cuerpo)
        # Nombrarlos es lo que hace útil el aviso: sin la lista, «revisa
        # los campos» no dice cuáles.
        for campo in ("priority", "trust", "topic_a"):
            self.assertIn(campo, cuerpo)
        # Y la salida: la sintaxis que SÍ funciona, incluida la del texto.
        self.assertIn("campos:", cuerpo)
        self.assertIn(": Text", cuerpo)

    def test_si_los_campos_SON_tuyos_no_dice_nada(self):
        """La mitad que impide que esto sea ruido en cada generación.

        Es el aserto que cae si alguien invierte la condición: sin él, el
        aviso saldría también a quien nombró sus columnas, y un aviso que
        sale siempre enseña a ignorar los avisos.
        """
        cuerpo = _generar("campos: distancia_km, peso_kg")
        self.assertNotIn("NO los has dicho", cuerpo)

    def test_tambien_en_ingles(self):
        """El aviso EN INGLÉS, comprobado sobre la función que lo redacta.

        NO se comprueba de punta a punta con un prompt inglés, y el
        motivo es un hallazgo en sí: «classify reviews by sentiment»
        entra por OTRA RAMA —el generador denso, que devuelve un `VECTOR
        Input[4]` de `feature_1..feature_4`— y no pasa por el supervisor
        de plantillas, así que este aviso no la cubre todavía. Ese hueco
        queda escrito en TASKS.md en vez de disimularlo con un prompt
        elegido para que pase.
        """
        # Palabras FUNCIONALES del castellano: no se pueden evitar
        # escribiendo en español, así que si el aviso cayera al idioma por
        # defecto alguna aparecería. Se mira solo el aviso, no la
        # respuesta entera —que lleva el prompt del usuario y otros
        # textos— para no acusar al idioma equivocado.
        aviso = campos_inventados_warning(
            {"synthesis": {"fields_invented": True, "selected_fields": ["a", "b"]}}, "en")
        assert aviso is not None
        for palabra in _FUNCIONALES_ES:
            self.assertNotIn(palabra, f" {aviso.lower()} ")

    def test_sin_campos_que_nombrar_no_se_avisa(self):
        """Un aviso que no puede señalar qué revisar es ruido."""
        self.assertIsNone(campos_inventados_warning(
            {"synthesis": {"fields_invented": True, "selected_fields": []}}, "es"))

    def test_sin_sintesis_no_revienta_ni_afirma(self):
        """No saberlo no es un sí: sin síntesis no se acusa a nadie."""
        self.assertIsNone(campos_inventados_warning({}, "es"))
        self.assertIsNone(campos_inventados_warning({"synthesis": None}, "es"))


if __name__ == "__main__":
    unittest.main()
