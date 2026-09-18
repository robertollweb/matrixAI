"""82-C4 — la página del Space estático codifica IGUAL que `predict.py`, y lo dice.

`space/index.html` promete, en su propio comentario, que su `encodeRecord`
«mirrors predict.py's MatrixAIModel._encode exactly». Nada lo comprobaba: las
pruebas del Space leían el HTML como texto, y la auditoría de cecc160 condujo la
página con UN solo ejemplo. Medido el 2026-09-18 corriendo el `encodeRecord`
del HTML en Node contra `_encode` (`predict_template.py`, el `predict.py` del
paquete), con la misma especificación: 7 de 22 entradas salían distintas, y dos
de ellas se escriben desde el formulario:

  * un `embedding_index` SIN vocabulario (solo `vocab_size`) iba por
    `parseInt`: "3.5" → la página predecía con el índice 3 donde `predict.py`
    rechaza, y "1e1" → índice 1 donde `predict.py` lee 10. Un
    `<input type=number>` devuelve los dos tal cual en `.value`;
  * un valor fuera de rango se recorta en los dos, pero `predict.py` lo DECLARA
    (`meta["clipped"]`) y la página callaba: quien escribía 15 en [0, 10]
    recibía la predicción de 10 sin enterarse.

El `encodeRecord` se saca del HTML GENERADO (`space_index_html`), no del
fuente: es lo que se publica. Y se ejecuta de verdad en Node; leerlo como texto
no habría cazado ninguno de los dos.
"""
from __future__ import annotations

import functools
import json
import shutil
import subprocess

import pytest

from matrixai.export import predict_template as pt
from matrixai.export.space import space_index_html

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(
    NODE is None, reason="sin `node` en esta máquina: el codificador de la página es JavaScript "
                         "y solo se puede comparar ejecutándolo")

SPEC = {
    "input_order": ["s", "i", "b", "s01", "oh__a", "oh__b", "ev", "en"],
    "fields": {
        "s": {"encoding": "scalar", "range": [0, 10], "type": "number"},
        "i": {"encoding": "scalar", "range": [0, 100], "type": "integer"},
        "b": {"encoding": "scalar", "range": [0, 1], "type": "boolean"},
        "s01": {"encoding": "scalar01", "type": "number"},
        "oh": {"encoding": "one_hot", "values": [{"raw": "a", "column": "oh__a"},
                                                  {"raw": "b", "column": "oh__b"}]},
        "ev": {"encoding": "embedding_index", "column": "ev", "vocab": ["x", "y", "z"]},
        "en": {"encoding": "embedding_index", "column": "en", "vocab_size": 5},
    },
}
BASE = {"s": "5", "i": "50", "b": "true", "s01": "0.5", "oh": "a", "ev": "y", "en": "2"}
BORRAR = object()

#: Lo que el formulario de la página PUEDE producir: textos que un
#: `<input type=number>` deja pasar en `.value` (aunque `step`/`min`/`max` los
#: marquen inválidos) o una opción de un `<select>`.
CASOS_DEL_FORMULARIO = {
    "base": {},
    "indice_sin_vocab_con_decimales": {"en": "3.5"},
    "indice_sin_vocab_en_notacion_cientifica": {"en": "1e1"},
    "indice_sin_vocab_2.0": {"en": "2.0"},
    "indice_sin_vocab_menos_cero": {"en": "-0"},
    "escalar_por_encima_del_rango": {"s": "11"},
    "escalar_por_debajo_del_rango": {"s": "-3"},
    "escalar01_por_encima": {"s01": "1.5"},
    "entero_con_decimales": {"i": "3.5"},
    "entero_en_notacion_cientifica": {"i": "1e1"},
    "escalar_vacio": {"s": ""},
    "escalar_desbordado": {"s": "1e400"},
}

#: Solo por programa (un `<input type=number>` no los deja llegar).
CASOS_DE_PROGRAMA = {
    "escalar_solo_espacios": {"s": "  "},
    "escalar_null": {"s": None},
    "escalar_hexadecimal": {"s": "0x10"},
    "escalar_con_guion_bajo": {"s": "1_0"},
    "indice_sin_vocab_con_letras": {"en": "3abc"},
    "booleano_como_numero": {"b": 1},
    "booleano_2": {"b": "2"},
    "one_hot_con_un_numero": {"oh": 7.0},
    "falta_la_clave": {"s": BORRAR},
    "campo_desconocido": {"zz": "1"},
}

#: DEUDA DECLARADA (decisión de deployer-91, 2026-09-18): la gramática de
#: `float()` de Python no es la de `Number()` de JS, y en estas dos se nota.
#: No se arregla porque el formulario no las deja llegar; una prueba de abajo
#: exige que SIGAN siendo solo de programa.
DEUDA_DECLARADA = {"escalar_hexadecimal", "escalar_con_guion_bajo"}


def _registro(cambios):
    r = dict(BASE)
    for k, v in cambios.items():
        if v is BORRAR:
            r.pop(k, None)
        else:
            r[k] = v
    return r


def _js_de_la_pagina() -> str:
    html = space_index_html("prueba")
    return html[html.index("const TRUE_WORDS"):html.index("function decodeOutput")]


@functools.lru_cache(maxsize=1)
def _en_node() -> dict[str, dict]:
    nombres = list(CASOS_DEL_FORMULARIO) + list(CASOS_DE_PROGRAMA)
    todos = {**CASOS_DEL_FORMULARIO, **CASOS_DE_PROGRAMA}
    registros = [_registro(todos[n]) for n in nombres]
    programa = (_js_de_la_pagina() + "\nconst spec = " + json.dumps(SPEC) + ";\n"
                "const salida = [];\nfor (const r of " + json.dumps(registros) + ") {\n"
                "  try { salida.push({ok: encodeRecord(spec, r)}); }\n"
                "  catch (e) { salida.push({error: String(e && e.message)}); }\n}\n"
                "console.log(JSON.stringify(salida));\n")
    corrida = subprocess.run([NODE, "-e", programa], capture_output=True, text=True, timeout=60)
    assert corrida.returncode == 0, corrida.stderr[-2000:]
    return dict(zip(nombres, json.loads(corrida.stdout)))


def _en_python(cambios) -> dict:
    m = pt.MatrixAIModel.__new__(pt.MatrixAIModel)
    m.spec, m.input_order = SPEC, list(SPEC["input_order"])
    m.input_index = {c: i for i, c in enumerate(m.input_order)}
    m.fields = SPEC["fields"]
    try:
        vector, meta = m._encode(_registro(cambios))
    except pt.MatrixAIModelError as e:
        return {"error": str(e)}
    return {"vector": vector, "clipped": meta["clipped"]}


def _vector_de_la_pagina(ok):
    """Hoy `encodeRecord` devuelve el vector; con el aviso de recorte devolverá
    `{vector, clipped}`. Las dos formas se leen igual aquí."""
    return ok["vector"] if isinstance(ok, dict) else ok


def _misma_codificacion(nombre, cambios):
    p, j = _en_python(cambios), _en_node()[nombre]
    if "error" in p or "error" in j:
        assert "error" in p and "error" in j, (
            f"{nombre}: predict.py -> {p.get('error') or p.get('vector')} · "
            f"página -> {j.get('error') or _vector_de_la_pagina(j.get('ok'))}")
        return
    assert _vector_de_la_pagina(j["ok"]) == p["vector"], nombre


class TestElInstrumentoLeeLaPaginaDeVerdad:
    """Sin esto, un extractor que no sacara nada —o que sacara otra cosa— dejaría
    todas las comparaciones de abajo sin dientes."""

    def test_saca_la_funcion_del_html_generado(self):
        js = _js_de_la_pagina()
        assert "function encodeRecord(spec, record)" in js
        assert len(js) > 500

    def test_el_caso_base_codifica_de_verdad_y_coincide(self):
        j = _en_node()["base"]
        assert "ok" in j, j
        assert _vector_de_la_pagina(j["ok"]) == [0.5, 0.5, 1.0, 0.5, 1.0, 0.0, 1.0, 2.0]
        assert _en_python({})["vector"] == [0.5, 0.5, 1.0, 0.5, 1.0, 0.0, 1.0, 2.0]


class TestLaPaginaCodificaComoPredictPy:

    @pytest.mark.parametrize("nombre", sorted(CASOS_DEL_FORMULARIO))
    def test_lo_que_se_puede_escribir_en_el_formulario(self, nombre):
        _misma_codificacion(nombre, CASOS_DEL_FORMULARIO[nombre])

    @pytest.mark.parametrize("nombre", sorted(set(CASOS_DE_PROGRAMA) - DEUDA_DECLARADA))
    def test_lo_que_solo_llega_por_programa(self, nombre):
        _misma_codificacion(nombre, CASOS_DE_PROGRAMA[nombre])

    def test_la_deuda_declarada_es_solo_de_programa(self):
        """Si alguien mueve un caso de la deuda al formulario, o apunta en la
        deuda algo que el formulario sí produce, esto lo para."""
        assert DEUDA_DECLARADA <= set(CASOS_DE_PROGRAMA)
        assert not DEUDA_DECLARADA & set(CASOS_DEL_FORMULARIO)


class TestLaPaginaDeclaraElRecorte:
    """`predict.py` dice cuándo recorta un valor al rango; la página tiene que
    decirlo IGUAL: mismos campos, mismo valor crudo, mismo valor recortado."""

    @pytest.mark.parametrize("nombre", ["escalar_por_encima_del_rango",
                                        "escalar_por_debajo_del_rango", "escalar01_por_encima"])
    def test_fuera_de_rango_la_pagina_dice_lo_mismo_que_predict_py(self, nombre):
        esperado = _en_python(CASOS_DEL_FORMULARIO[nombre])["clipped"]
        assert esperado, "el caso tiene que recortar en predict.py o no prueba nada"
        ok = _en_node()[nombre].get("ok")
        assert isinstance(ok, dict) and "clipped" in ok, (
            f"{nombre}: la página recorta y no lo declara (devuelve {ok!r})")
        assert ok["clipped"] == esperado

    def test_dentro_de_rango_no_declara_ningun_recorte(self):
        ok = _en_node()["base"].get("ok")
        assert isinstance(ok, dict) and ok.get("clipped") == [], ok
