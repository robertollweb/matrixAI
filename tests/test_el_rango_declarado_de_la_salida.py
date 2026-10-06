# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""EL RANGO DECLARADO DE LA SALIDA DE REGRESIÓN DEL PROMPT SE USA — 05-10.

Decidido por Roberto («Ok si, pero commitea todo antes. Haz los test pertinentes y auditorias para asegurar que
va a funcionar bien»). `SALIDA: precio_eur: Scalar en [60000, 900000]` se leía y se tiraba: la ruta del prompt
no fijaba nunca un rango de objetivo, el generador inventaba el objetivo en [-1, 1] y el entrenamiento no lo
normalizaba. Ahora:

* `analyze_playground_request` (ruta del prompt) declara el rango en `rango_declarado_de_la_salida` —solo si la
  salida generada ES una regresión, y solo un rango válido: mal escrito, del revés, degenerado o infinito →
  `None` con su aviso—. CON CLAVE PROPIA y no `target_range`: la interfaz clásica toma `target_range` de esta
  respuesta para entrenar pero genera sin él; con la misma clave entrenaría normalizando con [a, b] unos datos en
  [-1, 1] (todo recortado a 0, un modelo constante, sin un error).
* `generate_synthetic_dataset(..., target_range=)` genera el objetivo DENTRO del rango; lo que no es un par finito
  y creciente se ignora.
* Entrenar ese CSV con `target_range` normaliza el objetivo con él (medido: sin él, el mismo CSV revienta con
  «Numerical result out of range»).
* Sin rango declarado, todo como antes: el `mxai`, el contrato y el CSV, byte a byte (medido contra `5bf8f8e`;
  aquí, que no pasar el parámetro y pasar uno inválido dan el MISMO CSV).

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest.
"""
from __future__ import annotations

import re

import pytest

from matrixai.generation.prompt_field_specs import rango_de_la_salida
from matrixai.playground import analyze_playground_request
from matrixai.playground_api import generate_synthetic_dataset, run_playground_training

FEATURES = ("predecir el precio de una vivienda\nFEATURES:\n  superficie_m2: Scalar en [30, 400]\n"
            "  habitaciones: Integer[1, 10]\n")
PROMPT_CON_RANGO = FEATURES + "SALIDA: precio_eur: Scalar en [60000, 900000]"
PROMPT_SIN_RANGO = FEATURES + "SALIDA: precio_eur: Scalar"


# ── el lector ──────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("salida,rango,con_aviso", [
    ("SALIDA: precio_eur: Scalar en [60000, 900000]", (60000.0, 900000.0), False),
    ("SALIDA: y: Integer[0, 10]", (0.0, 10.0), False),
    ("OUTPUT: y: Scalar in [0, 5]", (0.0, 5.0), False),
    ("SALIDA: y: Scalar en [9, 1]", None, True),          # del revés
    ("SALIDA: y: Scalar en [1, 1]", None, True),          # degenerado
    ("SALIDA: y: Scalar en [a, b]", None, True),          # no numérico
    ("SALIDA: y: Scalar en [0, inf]", None, True),        # no finito
    ("SALIDA: y: Scalar en [1, 2, 3]", None, True),       # mal formado
    ("SALIDA: y: Scalar", None, False),                   # sin rango: nada que avisar
    ("SALIDA: y: ProbabilityMap[a, b]", None, False),     # no es una regresión
])
def test_el_rango_de_la_salida_solo_si_es_valido(salida, rango, con_aviso):
    leido, avisos = rango_de_la_salida(FEATURES + salida)
    assert leido == rango
    assert bool(avisos) is con_aviso, avisos


# ── el camino real: la ruta del prompt ─────────────────────────────────────────────────────────────

def test_la_ruta_del_prompt_declara_el_rango_con_clave_propia_y_sin_tocar_lo_demas():
    con = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_CON_RANGO})
    sin = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_SIN_RANGO})
    assert con["ok"] and sin["ok"]
    assert con["rango_declarado_de_la_salida"] == [60000.0, 900000.0]
    assert sin["rango_declarado_de_la_salida"] is None
    # La clave que lee la clásica para ENTRENAR no aparece por esta ruta (ni antes ni ahora).
    assert "target_range" not in con and "target_range" not in sin
    # El rango no cambia la red ni el contrato: lo mismo con él que sin él.
    assert con["mxai"] == sin["mxai"] and con["training_text"] == sin["training_text"]
    # Y la salida sigue sin ser una entrada.
    assert "precio_eur" not in (con.get("field_ranges") or {})


def _notas(r) -> str:
    """Los avisos de las etapas, donde los cuelga `_anotar_avisos` (la del generador)."""
    return " ".join(str(n) for s in (r.get("pipeline_stages") or []) for n in (s.get("warnings") or []))


def test_un_rango_del_reves_no_viaja_y_se_dice_por_que():
    r = analyze_playground_request({"mode": "prompt", "prompt": FEATURES + "SALIDA: precio_eur: Scalar en [900000, 60000]"})
    assert r["ok"] and r["rango_declarado_de_la_salida"] is None
    assert "rango invertido o degenerado" in _notas(r), r.get("pipeline_stages")


def test_un_rango_no_convierte_en_regresion_una_clasificacion():
    r = analyze_playground_request({"mode": "prompt", "prompt":
        "clasificar averias\nFEATURES:\n  temperatura: Scalar en [0, 150]\nSALIDA: estado: ProbabilityMap[OK, FALLO]"})
    assert r["ok"] and r["rango_declarado_de_la_salida"] is None


# ── el generador ───────────────────────────────────────────────────────────────────────────────────

def _generar(r, filas=120, **kw) -> str:
    rangos = {k: tuple(v) for k, v in (r.get("field_ranges") or {}).items()}
    d = generate_synthetic_dataset(r["mxai"], r["training_text"], filas, 7, "coherent", False,
                                   field_ranges_override=rangos or None, field_types=r.get("field_types") or None,
                                   **kw)
    assert d["ok"], d
    return d["csv_text"]


def _objetivo(csv_text: str) -> list[float]:
    filas = csv_text.splitlines()
    j = filas[0].split(",").index("predicted_value")
    return [float(f.split(",")[j]) for f in filas[1:]]


def test_con_el_rango_el_objetivo_se_genera_dentro_de_el():
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_CON_RANGO})
    valores = _objetivo(_generar(r, target_range=tuple(r["rango_declarado_de_la_salida"])))
    assert all(60000.0 <= v <= 900000.0 for v in valores), (min(valores), max(valores))
    # Y lo ocupa: no se queda en una esquina (con 120 filas, más de la mitad del recorrido).
    assert max(valores) - min(valores) > 0.5 * 840000.0


@pytest.mark.parametrize("invalido", [(900000.0, 60000.0), (1.0, 1.0), (0.0, float("inf")), (float("nan"), 1.0),
                                      (1.0,), ("a", "b"), None])
def test_un_rango_invalido_no_viaja_y_el_csv_es_el_de_siempre(invalido):
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_CON_RANGO})
    de_siempre = _generar(r)
    assert _generar(r, target_range=invalido) == de_siempre
    # El de siempre: el objetivo en [-1, 1], como antes de esto.
    assert all(-1.0 <= v <= 1.0 for v in _objetivo(de_siempre))


# ── el entrenamiento ───────────────────────────────────────────────────────────────────────────────

def test_el_csv_con_el_rango_se_entrena_normalizando_con_el():
    """Lo generado DENTRO de [60000, 900000] se entrena con ese mismo rango: normaliza y responde en euros. Sin él,
    el mismo CSV revienta (medido: «Numerical result out of range»): por eso los dos viajan juntos."""
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_CON_RANGO})
    rango = tuple(r["rango_declarado_de_la_salida"])
    csv_text = _generar(r, filas=80, target_range=rango)
    rangos = {k: tuple(v) for k, v in (r.get("field_ranges") or {}).items()}
    t = run_playground_training(r["mxai"], r["training_text"], csv_text, epochs_override=2,
                                field_ranges=rangos, target_range=rango, seed=42)
    assert t["ok"], t.get("error")
    # El error, en la escala del dominio (euros), no en la normalizada.
    assert t["mae"] is not None and t["mae"] > 1000.0, t["mae"]


@pytest.mark.parametrize("prompt", [
    "clasificar el riesgo en BAJO, MEDIO, ALTO\nFEATURES:\n  edad: Scalar en [18, 90]\nSALIDA: riesgo: Scalar en [0, 2]",
    "clasificar si un cliente abandona\nFEATURES:\n  edad: Scalar en [18, 90]\nSALIDA: abandona: Integer[0, 1]",
    "clasificar averias en ok y fallo\nFEATURES:\n  temperatura: Scalar en [0, 150]\nSALIDA: estado: Scalar en [0, 1]",
])
def test_una_salida_numerica_de_una_red_de_clasificacion_no_declara_rango(prompt):
    """La salida que se GENERÓ manda: si la red es de clasificación (ProbabilityMap, Probability), un rango
    numérico escrito en la SALIDA no la convierte en regresión ni viaja como rango del objetivo."""
    r = analyze_playground_request({"mode": "prompt", "prompt": prompt})
    assert r["ok"] and "Scalar" not in r["mxai"].split("OUTPUT", 1)[1].splitlines()[0]
    assert r["rango_declarado_de_la_salida"] is None


# ── auditoría del rango, 1.ª pasada (06-10) ────────────────────────────────────────────────────────

# M2 — `analyze_playground_request` descarta el aviso del rango que no valió cuando la red que salió NO es una
# regresión (`if not _es_regresion: _avisos_rango_salida = []`). Ninguna prueba lo sostenía: el sabotaje que lo
# quitaba salía verde. Una clasificación no usa rango de salida, así que «se ignora el rango» no le dice nada.
_CLASIFICACIONES_CON_UN_RANGO_QUE_NO_VALE = [
    "clasificar averias en tres clases\nFEATURES:\n  temperatura: Scalar en [0, 150]\nSALIDA: clase: Integer[9, 1]",
    "clasificar averias en OK o FALLO\nFEATURES:\n  temperatura: Scalar en [0, 150]\nSALIDA: estado: Scalar en [a, b]",
    "clasificar averias en OK, AVISO o FALLO\nFEATURES:\n  temperatura: Scalar en [0, 150]\n"
    "OUTPUT: estado: Integer[0, inf]",
]


@pytest.mark.parametrize("prompt", _CLASIFICACIONES_CON_UN_RANGO_QUE_NO_VALE)
def test_una_clasificacion_no_lleva_el_aviso_del_rango_que_no_valio(prompt):
    _, avisos = rango_de_la_salida(prompt)
    assert avisos, "el prompt tiene que traer un rango que no vale, o la prueba no mide nada"
    r = analyze_playground_request({"mode": "prompt", "prompt": prompt})
    # La red que salió es una CLASIFICACIÓN (su OUTPUT no es `Scalar`): si no, la prueba no mide este caso.
    assert r["ok"] and re.search(r"^[ \t]*OUTPUT[ \t]+\S+[ \t]*:[ \t]*Scalar\b", r["mxai"], re.MULTILINE) is None
    assert r["rango_declarado_de_la_salida"] is None
    notas = _notas(r)
    assert not any(a in notas for a in avisos), notas


def test_la_misma_salida_en_una_regresion_si_lleva_el_aviso():
    """El otro lado: con la red de regresión, el aviso del rango que no valió SÍ llega (no se borra siempre)."""
    prompt = FEATURES + "SALIDA: precio_eur: Integer[9, 1]"
    _, avisos = rango_de_la_salida(prompt)
    r = analyze_playground_request({"mode": "prompt", "prompt": prompt})
    assert r["ok"] and r["rango_declarado_de_la_salida"] is None
    assert avisos and all(a in _notas(r) for a in avisos), (avisos, _notas(r))


# M3 — `_rango_de_objetivo_valido`: solo un PAR (lista o tupla de dos) de números reales, finitos y crecientes.
# Un dict daba `KeyError` sin capturar (y `/api/generate-dataset` cerraba la conexión sin responder), `"12"` daba
# (1.0, 2.0) y `[False, True]` daba (0.0, 1.0).
@pytest.mark.parametrize("rango", [
    {"a": 1, "b": 2}, {0: 60000, 1: 900000}, "12", "ab", b"12", [False, True], (True, 5), ["1", "2"],
    ["60000", 900000], [None, 1], [[1], [2]], 5, 5.0, [1, 2, 3], [], set(), {1, 2},
])
def test_lo_que_no_es_un_par_de_numeros_no_vale_como_rango(rango):
    from matrixai.playground import _rango_de_objetivo_valido
    assert _rango_de_objetivo_valido(rango) is None


@pytest.mark.parametrize("rango,esperado", [
    ([60000, 900000], (60000.0, 900000.0)), ((60000.0, 900000.0), (60000.0, 900000.0)), ([-1.5, 0], (-1.5, 0.0)),
])
def test_un_par_de_numeros_crecientes_si_vale(rango, esperado):
    from matrixai.playground import _rango_de_objetivo_valido
    assert _rango_de_objetivo_valido(rango) == esperado


@pytest.mark.parametrize("invalido", [{"a": 1, "b": 2}, "12", [False, True]])
def test_el_generador_ignora_lo_que_no_es_un_rango_y_da_el_csv_de_siempre(invalido):
    """Por el camino real: ni revienta (el dict) ni se inventa un rango (la cadena, los booleanos)."""
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_CON_RANGO})
    assert _generar(r, target_range=invalido) == _generar(r)


# M5 — `TARGET COLUMN: x: Scalar in [..]` se quedaba sin rango: en la alternancia de las palabras que declaran
# la salida, `TARGET` iba delante de `TARGET COLUMN` y ganaba (el nombre leído era «COLUMN», y el tipo «precio»).
_SALIDAS_DE_DOS_PALABRAS = [
    "TARGET COLUMN: precio: Scalar in [100, 200]",
    "COLUMNA OBJETIVO: precio: Scalar en [100, 200]",
    "VARIABLE OBJETIVO: precio: Scalar en [100, 200]",
    "target column: precio: Scalar in [100, 200]",
]


@pytest.mark.parametrize("salida", _SALIDAS_DE_DOS_PALABRAS)
def test_la_salida_de_dos_palabras_trae_su_rango(salida):
    assert rango_de_la_salida(FEATURES + salida) == ((100.0, 200.0), [])


@pytest.mark.parametrize("salida", _SALIDAS_DE_DOS_PALABRAS)
def test_la_salida_de_dos_palabras_se_lee_como_salida_y_no_como_entrada(salida):
    """El lector de la salida (`prompt_objetivo`, con la MISMA lista) lee «precio», no «column»; y `precio` no
    es una entrada. Por la ruta del prompt, el rango viaja."""
    from matrixai.generation.prompt_field_specs import parse_field_specs
    from matrixai.generation.prompt_objetivo import objetivo_declarado

    prompt = FEATURES + salida
    objetivo = objetivo_declarado(prompt)
    assert objetivo is not None and objetivo.nombre == "precio", objetivo
    assert [f.name for f in parse_field_specs(prompt).fields] == ["superficie_m2", "habitaciones"]
    r = analyze_playground_request({"mode": "prompt", "prompt": prompt})
    assert r["ok"] and r["rango_declarado_de_la_salida"] == [100.0, 200.0]
    assert "precio" not in (r.get("field_ranges") or {})
