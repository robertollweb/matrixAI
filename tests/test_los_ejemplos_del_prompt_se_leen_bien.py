# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""LOS EJEMPLOS DEL PROMPT SE LEEN COMO SE ESCRIBEN — 05-10 (decidido por Roberto).

Dos ejemplos de los que el Studio pone en el prompt (`ejemplosDePrompt.ts`, 2.1 y 7.1) los leía mal el lector
DETERMINISTA del núcleo (sin LLM; medido también así por deployer-f1):

* 2.1 — `SALIDA: precio_eur: Scalar en [60000, 900000]`, escrita después de las `FEATURES`, entraba en el VECTOR
  como una ENTRADA más (`Input[5]` con `precio_eur`), y el objetivo era otro (`predicted_value`). La causa:
  `_FIELD_ENTRY_RE` admite `:` como borde de campo, así que leía desde «SALIDA:» y se quedaba `precio_eur:
  Scalar`. Igual con `OUTPUT: y: Scalar` y `objetivo: y: Integer[...]`.
* 7.1 — la red se llamaba «NETWORK O» y el proyecto «OProject»: el campo «modelo: Categorical[...]» casaba con
  el patrón del nombre de la red como «model» + «o».

Y lo que se MIDIÓ y NO es un defecto, dicho para que nadie lo «arregle»: en el 7.1 la categórica de ENTRADA
`modelo` va como `Integer[0, 14]` con embedding (lo diseñado: el CSV lleva el índice), y el generador escribe el
índice como «2.0». Validar y entrenar con «2.0» o con «2» da lo mismo (la misma pérdida al último decimal,
medido el 05-10); aquí se fija que el CSV del propio generador VALIDA.

Por el camino real: `analyze_playground_request` con el prompt tal cual (sin LLM), lo mismo que devuelve
`generate` al Studio.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest.
"""
from __future__ import annotations

import re

import pytest

from matrixai.generation.prompt_field_specs import parse_field_specs
from matrixai.generation.prompt_objetivo import objetivo_declarado
from matrixai.playground import analyze_playground_request
from matrixai.training.dense_generator import DenseNetworkGenerator

#: Los dos ejemplos, COPIADOS de `studio/src/creacion/ejemplosDePrompt.ts` (ids 2.1 y 7.1).
PROMPT_2_1 = """predecir el precio de una vivienda
FEATURES:
  superficie_m2: Scalar en [30, 400]
  habitaciones: Integer[1, 10]
  antiguedad_anios: Integer[0, 100]
  distancia_centro_km: Scalar en [0, 40]
SALIDA: precio_eur: Scalar en [60000, 900000]"""

PROMPT_7_1 = """clasificar averias de maquinas
FEATURES:
  temperatura: Scalar en [0, 150]
  modelo: Categorical[M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14]
SALIDA: estado: ProbabilityMap[OK, FALLO]"""


def _vector(mxai: str) -> list[str]:
    bloque = re.search(r"VECTOR [^\n]*\n(.*?)\nEND", mxai, re.S)
    assert bloque, mxai
    return [linea.split(":")[0].strip() for linea in bloque.group(1).splitlines() if linea.strip()]


# ── 2.1: la SALIDA no es una entrada ─────────────────────────────────────────────────────────────

def test_2_1_por_el_camino_real_la_salida_no_entra_en_el_vector():
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_2_1})
    assert r["ok"], r.get("error")
    assert _vector(r["mxai"]) == ["superficie_m2", "habitaciones", "antiguedad_anios", "distancia_centro_km"]
    assert "precio_eur" not in (r.get("field_ranges") or {})
    # Lo que SÍ declaran las entradas sigue llegando (las enteras, sus rangos).
    assert r["field_types"] == {"habitaciones": "integer", "antiguedad_anios": "integer"}
    assert list(r["field_ranges"]["superficie_m2"]) == [30.0, 400.0]


@pytest.mark.parametrize("prompt,esperadas", [
    ("predecir\nFEATURES:\n  a: Scalar en [0, 1]\nSALIDA: y: Scalar en [5, 9]", ["a"]),
    ("predecir\nFEATURES:\n  a: Scalar\nOUTPUT: y: Scalar", ["a"]),           # con dos puntos: la guarda no lo veía
    ("predecir\nFEATURES:\n  a: Scalar\nOUTPUT y: Scalar", ["a"]),            # la forma que ya estaba cubierta
    ("predecir\nFEATURES:\n  a: Scalar\nobjetivo: y: Integer[0, 9]", ["a"]),
    ("predecir\nFEATURES:\n  a: Scalar\nTARGET: y: Boolean", ["a"]),
    ("clasificar\nFEATURES:\n  a: Scalar\nSALIDA: y: ProbabilityMap[si, no]", ["a"]),  # 3.1/5.1: ya salía bien
])
def test_ninguna_forma_de_declarar_la_salida_la_vuelve_una_entrada(prompt, esperadas):
    assert [f.name for f in parse_field_specs(prompt).fields] == esperadas


def test_en_un_prompt_de_una_sola_linea_lo_de_detras_de_la_salida_sigue_siendo_entrada():
    """Se quita SOLO la declaración de la salida, no el resto de la línea."""
    nombres = [f.name for f in parse_field_specs("a: Scalar en [0, 1], SALIDA: y: Scalar, b: Boolean").fields]
    assert nombres == ["a", "b"]


def test_el_objetivo_del_prompt_se_sigue_leyendo_de_la_salida():
    """La lista de palabras que declaran la salida es UNA (`PALABRAS_QUE_DECLARAN_LA_SALIDA`): el lector del
    objetivo la comparte con el de campos, y sigue leyendo `precio_eur`."""
    assert objetivo_declarado(PROMPT_2_1).nombre == "precio_eur"


# ── 7.1: el nombre de la red solo si el prompt lo dice ──────────────────────────────────────────

def test_7_1_por_el_camino_real_la_red_no_se_llama_o():
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_7_1})
    assert r["ok"], r.get("error")
    red = re.search(r"NETWORK (\S+)", r["mxai"]).group(1)
    proyecto = re.search(r"PROJECT (\S+)", r["mxai"]).group(1)
    assert red not in ("O", "") and proyecto == f"{red}Project", (red, proyecto)
    # Lo diseñado, sin cambios: `modelo` es una entrada de índice con embedding.
    assert re.search(r"modelo: Integer\[0, 14\]", r["mxai"]), r["mxai"]


@pytest.mark.parametrize("prompt,nombre,entidad", [
    ("FEATURES: modelo: Categorical[M0, M1]", "", ""),         # un CAMPO llamado modelo no nombra la red
    ("FEATURES: red: Scalar, input_x: Scalar", "", ""),       # ni «red:» ni «input_x:»
    ("redes neuronales para la demanda", "", ""),             # «red» dentro de otra palabra, tampoco
    ("crea una red llamada Ventas que prediga x", "Ventas", ""),
    ("modelo Riesgo con edad", "Riesgo", ""),
    ("network named Sales", "Sales", ""),
    ("entrada llamada Cliente", "", "Cliente"),
])
def test_el_nombre_de_la_red_y_de_la_entidad_solo_si_el_prompt_lo_dice(prompt, nombre, entidad):
    g = DenseNetworkGenerator()
    assert (g._extract_name(prompt), g._extract_entity(prompt)) == (nombre, entidad)


# ── 7.1: el índice escrito «2.0» NO es un defecto (medido), y el CSV del generador valida ───────

def test_7_1_el_csv_del_generador_con_el_indice_escrito_2_0_valida():
    from matrixai.playground_api import generate_synthetic_dataset, validate_training_csv
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT_7_1})
    datos = generate_synthetic_dataset(
        r["mxai"], r["training_text"], 40, 7, "coherent", False,
        field_ranges_override={"temperatura": (0.0, 150.0)}, field_types={"temperatura": "number"},
        field_categories=r["field_categories"])
    filas = datos["csv_text"].splitlines()
    j = filas[0].split(",").index("modelo")
    valores = {f.split(",")[j] for f in filas[1:]}
    # Lo medido: el índice, escrito como decimal y dentro de sus 15 valores…
    assert all(v.endswith(".0") and 0 <= float(v) < 15 for v in valores), valores
    # …y el núcleo lo acepta tal cual.
    v = validate_training_csv(r["mxai"], r["training_text"], datos["csv_text"], {"temperatura": (0.0, 150.0)})
    assert v["ok"], v
