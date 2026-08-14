# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El TEXTO descrito en PROSA: o sale un modelo de texto, o sale un aviso.

Medido contra el backend real el 2026-08-14, ANTES de esto: «Clasificar si
una reseña es positiva o negativa a partir del TEXTO de la reseña» generaba

    VECTOR Review[1]
      review_text: Scalar

sin un solo aviso — y `/api/train-start` lo rechazaba después fila por fila
(«field comentario_texto must be numeric, got 'el servicio fue genial'»).
El motor de transformers estaba sano: el hueco estaba en cómo se LLEGA.

Las dos mitades, y la segunda no es opcional:

1. El esquema que se le pide al LLM tiene una casilla `TEXT:`. Antes no
   había DÓNDE decirlo: el LLM escribía «text data» en su RATIONALE
   mientras ponía `Scalar` en el campo. Con la casilla, lo que ya sabía
   enruta al generador de transformers.
2. Cuando el campo se queda en `Scalar` habiendo texto por medio, se AVISA.
   Lo primero es una heurística —la rellena un LLM, que puede callarse,
   equivocarse o no estar— y una heurística que falla en silencio deja
   exactamente el defecto de partida.

Y lo que NO puede pasar: que un prompt tabular normal cambie o le salgan
avisos que no venían.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from matrixai.playground import (
    _DENSE_SCHEMA_SYSTEM,
    _parse_dense_schema,
    analyze_playground_request,
    text_left_scalar_warning,
)
from matrixai.training.transformer_generator import (
    TransformerNetworkGenerator,
    TransformerNetworkGeneratorError,
)

PROSA_ES = "Clasificar si una resena es positiva o negativa a partir del texto de la resena"
PROSA_EN = "Classify whether a customer review is positive or negative from the review text"
PROSA_DOS_TEXTOS = (
    "Clasificar correos en urgente o normal. El campo asunto es texto libre "
    "y el campo cuerpo es texto libre."
)
TABULAR = "Predecir si un envio llega tarde desde la distancia en km y el peso"


def _avisos(result) -> list[str]:
    return [w for s in result.get("pipeline_stages") or [] for w in (s.get("warnings") or [])]


def _aviso_de_texto(result) -> list[str]:
    """Los avisos que hablan de que el texto se quedó en `Scalar`."""
    return [w for w in _avisos(result)
            if "must be numeric" in w and ("Scalar" in w or "scalar" in w)]


# ---------------------------------------------------------------------------
# 1. La casilla del esquema
# ---------------------------------------------------------------------------

class TestCasillaTexto:
    def test_el_esquema_pide_los_campos_de_texto(self):
        """Sin la línea en el esquema, el LLM no tiene dónde decirlo: es la
        causa raíz medida, no un detalle de redacción."""
        assert "\nTEXT:" in _DENSE_SCHEMA_SYSTEM

    def test_el_esquema_prohibe_adivinar_por_el_nombre(self):
        """Lección del contrato 71: un detector por parecido de nombres
        acusaría a `last_year_salary`. Se decide por lo que DIJO el usuario,
        y el esquema tiene que decir las dos cosas: dónde está la evidencia
        (sus palabras) y que el nombre del campo NO lo es."""
        assert "the user's OWN WORDS" in _DENSE_SCHEMA_SYSTEM
        assert "Do NOT infer it from the field name alone" in _DENSE_SCHEMA_SYSTEM

    def test_parsea_la_linea_text(self):
        r = _parse_dense_schema("FIELDS: review_text\nTEXT: review_text\nLABELS: neg, pos\n")
        assert r["text_fields"] == {"review_text": None}

    def test_sin_linea_text_no_hay_campos_de_texto(self):
        assert "text_fields" not in _parse_dense_schema("FIELDS: a, b\nARCHITECTURE: dense\n")

    def test_none_no_es_un_campo(self):
        """«TEXT: none» es como un LLM dice que no hay ninguno; tomarlo por un
        campo crearía una columna llamada `none` de la nada."""
        assert "text_fields" not in _parse_dense_schema("FIELDS: a, b\nTEXT: none\n")


# ---------------------------------------------------------------------------
# 2. El enrutado: la prosa acaba en un transformer de verdad
# ---------------------------------------------------------------------------

_SOLO_TEXTO = {
    "input_fields": ["review_text"],
    "text_fields": {"review_text": None},
    "labels": ["negativa", "positiva"],
    "network_name": "ReviewClassifier",
    "input_name": "Review",
}


class TestEnrutado:
    def test_el_texto_de_la_prosa_genera_un_transformer(self):
        with patch("matrixai.playground._dense_llm_schema", return_value=dict(_SOLO_TEXTO)):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": PROSA_ES, "use_llm": True, "locale": "es"})
        assert r["ok"] is True
        assert r["supervision_source"] == "transformer_generator"
        assert r["architecture_decision"]["kind"] == "transformer"
        assert "BLOCK enc TRANSFORMER" in r["mxai"]
        assert "SEQUENCE" in r["mxai"]
        # El campo de texto es una SEQUENCE, ya no un `Scalar` del VECTOR.
        assert "review_text: Scalar" not in r["mxai"]
        assert r["field_types"] == {"review_text": "text"}
        assert r["field_seq"] == {"review_text": {"length": 64, "tokenizer": "byte_v1"}}

    def test_quien_decidio_se_declara_llm_no_prompt(self):
        """El prompt no declaró nada: atribuírselo sería mentir sobre quién
        decidió, y quien audita necesita distinguirlo."""
        with patch("matrixai.playground._dense_llm_schema", return_value=dict(_SOLO_TEXTO)):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": PROSA_ES, "use_llm": True, "locale": "es"})
        assert r["architecture_decision"]["source"] == "llm"

    def test_lo_que_paso_se_cuenta(self):
        """Un campo de texto que nadie declaró no puede aparecer en silencio."""
        with patch("matrixai.playground._dense_llm_schema", return_value=dict(_SOLO_TEXTO)):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": PROSA_ES, "use_llm": True, "locale": "es"})
        nota = [w for w in _avisos(r) if "review_text" in w and "propuesto por el LLM" in w]
        assert nota, _avisos(r)
        assert "Text[N]" in nota[0]  # cómo fijarlo uno mismo

    def test_lo_que_paso_se_cuenta_en_ingles(self):
        with patch("matrixai.playground._dense_llm_schema", return_value=dict(_SOLO_TEXTO)):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": PROSA_EN, "use_llm": True, "locale": "en"})
        nota = [w for w in _avisos(r) if "review_text" in w and "proposed by the LLM" in w]
        assert nota, _avisos(r)
        # Palabras FUNCIONALES del castellano: no se pueden evitar escribiendo
        # en español, así que delatan media traducción.
        assert not any(f" {p} " in nota[0] for p in ("el", "la", "del", "con", "que", "para"))

    def test_la_declaracion_del_prompt_gana_al_llm(self):
        """Invariante 1: `campo: Text` escrito por quien pide el modelo manda
        sobre lo que proponga el LLM."""
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={"input_fields": ["otro_campo"],
                                 "text_fields": {"otro_campo": None}}):
            r = analyze_playground_request({
                "mode": "prompt", "use_llm": True,
                "prompt": "Clasificar resenas\nresenas: Text\n"
                          "OUTPUT clase: ProbabilityMap[NEG, POS]",
            })
        assert r["field_types"] == {"resenas": "text"}
        assert r["architecture_decision"]["source"] == "prompt_types"

    def test_la_longitud_adivinada_no_se_honra(self):
        """La longitud NO está en las palabras de quien pide el modelo: la
        inventa el LLM, y la atención escala O(L²). Medido el 2026-08-14: el
        LLM propuso 2000 para una reseña. Sin longitud declarada se aplica el
        default del generador, la MISMA política que para un `campo: Text`."""
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_SOLO_TEXTO, "text_fields": {"review_text": 2000}}):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": PROSA_ES, "use_llm": True, "locale": "es"})
        assert r["field_seq"] == {"review_text": {"length": 64, "tokenizer": "byte_v1"}}
        assert "length = 2000" not in r["mxai"]


# ---------------------------------------------------------------------------
# 3. El aviso: lo que impide que la heurística falle en silencio
# ---------------------------------------------------------------------------

class TestAviso:
    def test_texto_mezclado_con_tabulares_avisa_y_no_calla(self):
        """v1 no admite texto + tabular en el mismo modelo. El modelo se queda
        tabular —nadie pidió un transformer— pero NO en silencio: sin aviso
        esto es exactamente el defecto de partida."""
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={"input_fields": ["review_text", "rating", "customer_id"],
                                 "text_fields": {"review_text": None},
                                 "labels": ["negativa", "positiva"]}):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": PROSA_ES, "use_llm": True, "locale": "es"})
        assert r["ok"] is True
        assert r["architecture_decision"]["kind"] != "transformer"
        avisos = _aviso_de_texto(r)
        assert avisos, _avisos(r)
        assert "review_text" in avisos[0]
        assert "Mezclar" in avisos[0]  # el motivo, no solo el síntoma

    def test_dos_campos_de_texto_avisan_con_su_motivo(self):
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={"input_fields": ["asunto", "cuerpo"],
                                 "text_fields": {"asunto": None, "cuerpo": None},
                                 "labels": ["urgente", "normal"]}):
            r = analyze_playground_request({"mode": "prompt", "prompt": PROSA_DOS_TEXTOS,
                                            "use_llm": True, "locale": "es"})
        avisos = _aviso_de_texto(r)
        assert avisos, _avisos(r)
        assert "asunto" in avisos[0] and "cuerpo" in avisos[0]
        assert "un campo Text por modelo" in avisos[0]

    @pytest.mark.parametrize("prompt", [PROSA_ES, PROSA_EN])
    def test_sin_llm_la_prosa_sigue_avisando(self, prompt):
        """LA RED DE SEGURIDAD. Sin proveedor LLM no hay casilla que rellenar,
        y el modelo sale igual de numérico: el aviso no puede depender de que
        el LLM esté y acierte."""
        r = analyze_playground_request(
            {"mode": "prompt", "prompt": prompt, "use_llm": False, "locale": "es"})
        assert r["ok"] is True
        assert _aviso_de_texto(r), _avisos(r)

    def test_el_aviso_tambien_sale_por_la_rama_del_supervisor(self):
        """Un prompt sin verbo de tarea no pasa por el generador: lo atiende
        PromptSupervisor, que también entrega un VECTOR numérico. El fallo era
        el silencio, no la rama."""
        r = analyze_playground_request({
            "mode": "prompt", "use_llm": False, "locale": "es",
            "prompt": "El campo asunto es texto libre y el campo cuerpo es texto libre",
        })
        assert r["ok"] is True
        assert r["supervision_source"] != "transformer_generator"
        assert _aviso_de_texto(r), _avisos(r)

    def test_el_aviso_se_traduce_en_el_core(self):
        r_es = analyze_playground_request(
            {"mode": "prompt", "prompt": PROSA_ES, "use_llm": False, "locale": "es"})
        r_en = analyze_playground_request(
            {"mode": "prompt", "prompt": PROSA_EN, "use_llm": False, "locale": "en"})
        es, en = _aviso_de_texto(r_es), _aviso_de_texto(r_en)
        assert es and en
        assert "Tu descripción" in es[0]
        assert "Your description" in en[0]
        assert not any(f" {p} " in en[0] for p in ("el", "la", "del", "con", "que", "para"))

    def test_un_modelo_de_texto_de_verdad_no_lleva_aviso(self):
        """Arreglar algo puede volver FALSO un aviso: cuando el texto SÍ se
        enrutó, decir que se quedó en Scalar sería mentira."""
        with patch("matrixai.playground._dense_llm_schema", return_value=dict(_SOLO_TEXTO)):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": PROSA_ES, "use_llm": True, "locale": "es"})
        assert not _aviso_de_texto(r), _avisos(r)

    def test_el_aviso_no_inventa_ningun_tipo(self):
        """Avisar no es decidir: el modelo entregado sigue siendo el que
        salió, sin campos convertidos a Text por la espalda."""
        r = analyze_playground_request(
            {"mode": "prompt", "prompt": PROSA_ES, "use_llm": False, "locale": "es"})
        assert r["field_seq"] == {}
        assert "text" not in (r.get("field_types") or {}).values()
        assert "TRANSFORMER" not in r["mxai"]


# ---------------------------------------------------------------------------
# 4. El caso común no se toca
# ---------------------------------------------------------------------------

class TestTabularIntacto:
    def test_prompt_tabular_sin_llm_no_gana_avisos(self):
        r = analyze_playground_request(
            {"mode": "prompt", "prompt": TABULAR, "use_llm": False, "locale": "es"})
        assert r["ok"] is True
        assert r["architecture_decision"]["kind"] == "dense"
        assert not _aviso_de_texto(r), _avisos(r)

    def test_prompt_tabular_con_llm_sin_casilla_text_no_cambia(self):
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={"input_fields": ["distancia_km", "peso_kg"],
                                 "labels": ["a_tiempo", "tarde"], "architecture": "dense"}):
            r = analyze_playground_request(
                {"mode": "prompt", "prompt": TABULAR, "use_llm": True, "locale": "es"})
        assert r["architecture_decision"]["kind"] == "dense"
        assert "distancia_km: Scalar" in r["mxai"]
        assert not _aviso_de_texto(r), _avisos(r)

    def test_una_palabra_que_solo_PARECE_texto_no_dispara_nada(self):
        """`contexto`/`pretexto` contienen «texto» y no lo son. Un aviso que
        sale siempre deja de ser un aviso."""
        r = analyze_playground_request({
            "mode": "prompt", "use_llm": False, "locale": "es",
            "prompt": "Predecir el riesgo de un cliente segun el contexto y el pretexto de la solicitud",
        })
        assert not _aviso_de_texto(r), _avisos(r)


# ---------------------------------------------------------------------------
# 5. Las piezas por dentro
# ---------------------------------------------------------------------------

class TestGeneradorCampoPropuesto:
    def test_genera_con_el_campo_propuesto_y_declara_su_origen(self):
        g = TransformerNetworkGenerator().generate(
            PROSA_ES, proposed_text_fields={"review_text": None})
        assert g.field_name == "review_text"
        assert g.text_field_source == "proposed"
        assert g.length == 64
        assert any("PROPOSED by the caller" in a for a in g.assumptions)

    def test_un_campo_declarado_gana_al_propuesto(self):
        g = TransformerNetworkGenerator().generate(
            "Clasificar resenas\nresenas: Text[128]",
            proposed_text_fields={"otro": None})
        assert g.field_name == "resenas"
        assert g.length == 128
        assert g.text_field_source == "prompt"

    def test_propuesto_mas_tabulares_es_un_error_accionable(self):
        with pytest.raises(TransformerNetworkGeneratorError) as exc:
            TransformerNetworkGenerator().generate(
                PROSA_ES,
                proposed_text_fields={"review_text": None},
                input_fields=["review_text", "rating"])
        assert "rating" in str(exc.value)

    def test_un_nombre_inservible_no_crea_ninguna_columna(self):
        """Sin nombre no hay columna que pedirle a nadie: mejor el error
        accionable que un campo llamado `_`."""
        with pytest.raises(TransformerNetworkGeneratorError):
            TransformerNetworkGenerator().generate(PROSA_ES, proposed_text_fields={"!!!": None})


class TestFuncionDelAviso:
    _MODELO_NUMERICO = {"ok": True, "mxai": "VECTOR X[1]\n  review_text: Scalar\nEND\n"}

    def test_calla_cuando_nada_habla_de_texto(self):
        assert text_left_scalar_warning(self._MODELO_NUMERICO, TABULAR, "es") is None

    def test_calla_cuando_la_generacion_fallo(self):
        """Ahí ya hay un error: dos mensajes contradictorios son peores."""
        assert text_left_scalar_warning({"ok": False}, PROSA_ES, "es") is None

    def test_calla_cuando_el_modelo_es_de_texto(self):
        assert text_left_scalar_warning(
            {"ok": True, "mxai": "", "field_seq": {"resenas": {"length": 64}}},
            PROSA_ES, "es") is None

    def test_habla_cuando_la_prosa_lo_dice(self):
        aviso = text_left_scalar_warning(self._MODELO_NUMERICO, PROSA_ES, "es")
        assert aviso and "must be numeric" in aviso

    def test_nombra_los_campos_y_el_motivo_cuando_se_saben(self):
        aviso = text_left_scalar_warning(
            self._MODELO_NUMERICO, TABULAR, "es",
            proposed_fields=("review_text",), reason="no cabe con los tabulares")
        assert "review_text" in aviso
        assert "no cabe con los tabulares" in aviso

    def test_un_idioma_desconocido_cae_al_castellano(self):
        aviso = text_left_scalar_warning(self._MODELO_NUMERICO, PROSA_ES, "klingon")
        assert aviso and "Tu descripción" in aviso
