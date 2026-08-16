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
from unittest import mock

from matrixai.agents.prompt import PromptAgent
from matrixai.playground import analyze_playground_request, campos_inventados_warning
from matrixai.training.dense_generator import (
    DenseNetworkGenerator,
    DenseNetworkGeneratorError,
)

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


def _aviso_de(prompt: str, locale: str = "es") -> str | None:
    """SOLO el aviso, sacado de donde vive: los `warnings` de las etapas.

    La respuesta entera lleva el prompt de quien pide y otros textos, así
    que un barrido de idioma sobre ella acusaría al idioma equivocado.
    """
    r = analyze_playground_request({"prompt": prompt, "use_llm": False, "locale": locale})
    for etapa in (r.get("pipeline_stages") or []):
        for aviso in (etapa.get("warnings") or []):
            if "did NOT name these fields" in aviso or "NO los has dicho" in aviso:
                return aviso
    return None


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
        """El aviso EN INGLÉS, y ahora SÍ de punta a punta.

        Esta prueba documentaba el hueco que hoy se cierra: decía que no
        se comprobaba con un prompt inglés porque «classify reviews by
        sentiment» entra por OTRA RAMA —el generador denso— que no
        avisaba. Conserva su intención (que el aviso no se caiga al
        idioma por defecto) y deja de elegir el camino que pasaba.
        """
        # Palabras FUNCIONALES del castellano: no se pueden evitar
        # escribiendo en español, así que si el aviso cayera al idioma por
        # defecto alguna aparecería. Se mira solo el aviso, no la
        # respuesta entera —que lleva el prompt del usuario y otros
        # textos— para no acusar al idioma equivocado.
        aviso = _aviso_de("classify reviews by sentiment", "en")
        assert aviso is not None, "el prompt inglés no llevó aviso de campos inventados"
        for palabra in _FUNCIONALES_ES:
            self.assertNotIn(palabra, f" {aviso.lower()} ")


class LaRamaDensaTambienLoDice(unittest.TestCase):
    """El agujero que dejó dicho `802ec71`, medido y cerrado.

    Un prompt en INGLÉS no entra por el supervisor de plantillas sino por
    el **generador denso**, que rellena con `feature_1..feature_4` cuando
    no extrae ningún campo del prompt. Igual de inventados que los de la
    plantilla, y allí el aviso no llegaba: `campos_inventados_warning`
    leía la síntesis del supervisor, que en esa rama no existe.

    La señal equivalente NO se ha inventado: es el mismo hecho que ya
    conocía `resolve_prompt_fields` —usó `_default_fields()` porque
    `_extract_fields` no devolvió nada— y viaja en `fields_invented` del
    resultado, con el mismo nombre que en `PromptSynthesis`. El texto del
    aviso sigue escrito UNA sola vez.
    """

    def test_el_generador_denso_sigue_rellenando_esa_prosa(self):
        """El fixture: si esto cambia, lo de abajo mide otra cosa.

        Igual que su gemelo del supervisor. El día que el generador
        aprenda a leer «classify reviews by sentiment», este aserto se
        pone rojo y avisa — en vez de dejar las pruebas de abajo en verde
        por la razón equivocada.
        """
        gen = DenseNetworkGenerator().generate("classify reviews by sentiment")
        self.assertTrue(gen.fields_invented)
        self.assertEqual(gen.input_fields,
                         ["feature_1", "feature_2", "feature_3", "feature_4"])

    def test_el_prompt_ingles_de_la_tarea_avisa_y_NOMBRA_el_relleno(self):
        cuerpo = _generar("classify reviews by sentiment", "en")
        self.assertIn("did NOT name these fields", cuerpo)
        # Nombrados: sin la lista, el aviso no dice qué revisar.
        for campo in ("feature_1", "feature_2", "feature_3", "feature_4"):
            self.assertIn(campo, cuerpo)
        # Y la salida, la misma que en la otra rama.
        self.assertIn("fields:", cuerpo)
        self.assertIn(": Text", cuerpo)

    def test_por_la_rama_densa_en_espanol_tambien(self):
        """No es «el aviso en inglés»: es el aviso de esa RAMA.

        `campos: a, b` es la única sintaxis que `_extract_fields`
        reconoce, así que un prompt español en prosa que llegue al
        generador denso rellena igual.
        """
        self.assertIn("NO los has dicho", _generar("predecir la rotacion de personal"))

    def test_si_los_campos_SON_tuyos_la_rama_densa_calla(self):
        """El otro lado del sesgo, medido por la rama densa.

        Arreglar un sesgo puede crear el contrario: si la condición se
        invirtiera, este aviso saldría en CADA generación con campos
        declarados y dejaría de significar algo.
        """
        cuerpo = _generar("classify customers fields: age, income", "en")
        self.assertIn("VECTOR", cuerpo)  # se generó un modelo, no un error
        self.assertNotIn("did NOT name these fields", cuerpo)

    def test_la_rama_COMPUESTA_comparte_resolutor_y_aviso(self):
        """Comparte `resolve_prompt_fields`, así que compartía el agujero."""
        r = analyze_playground_request(
            {"prompt": "modelo complejo con dropout para clasificar la rotacion de personal",
             "use_llm": False, "locale": "es"})
        self.assertEqual(r.get("supervision_source"), "composite_generator")
        self.assertIn("NO los has dicho", json.dumps(r, ensure_ascii=False, default=str))

    def test_cuando_el_generador_se_rinde_y_cae_al_supervisor_TAMBIEN(self):
        """El tercer hueco de la misma familia, que estaba al lado.

        Si el generador denso no sabe interpretar el prompt, se cae al
        supervisor — y ahí el informe no se guardaba, así que el aviso se
        lo saltaba justo cuando más probable es que los campos no sean
        los de quien pidió el modelo.

        EL PROMPT ES EL INGLÉS A PROPÓSITO, y la primera versión de esta
        prueba estaba mal por eso: con el español pasaba en VERDE sin
        tocar la rama —«clasifica reseñas por sentimiento» ni siquiera
        entra por el generador (`_is_neural_prompt` es False), así que el
        mock no pintaba nada y el aviso salía por el camino de siempre—.
        Revirtiendo el arreglo no se ponía roja: un banco de pruebas sin
        dientes.
        """
        with mock.patch.object(DenseNetworkGenerator, "generate",
                               side_effect=DenseNetworkGeneratorError("no se entiende")):
            r = analyze_playground_request(
                {"prompt": "classify reviews by sentiment", "use_llm": False, "locale": "en"})
        # Que la caída OCURRIÓ: el prompt va al generador denso, y solo
        # acaba en el supervisor porque este se ha rendido.
        self.assertEqual(r.get("supervision_source"), "deterministic")
        self.assertIn("did NOT name these fields",
                      json.dumps(r, ensure_ascii=False, default=str))

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
