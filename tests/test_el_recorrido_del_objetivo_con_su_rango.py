# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""EL RECORRIDO DEL OBJETIVO CON SU RANGO — auditoría del rango, 2.ª pasada (N1, N2, N5, N6, N7; 06-10).

La 1.ª reparación rechazaba con «≥ 90 % fuera del rango, tolerancia del 0,1 % del ANCHO», y la re-auditoría lo midió
por HTTP en los dos sentidos:

- N1: con un rango que contiene [-1, 1] —`[0, 900000]`, `[500, 900000]`, `[-1000, 1000]`—, la tolerancia (o el propio
  rango) metía «dentro» los datos que la clásica genera sin rango, y salía el modelo CONSTANTE de I1 (R² −1,23·10⁹ y la
  misma predicción para cualquier vivienda). `[0, X]` es seguramente el rango más corriente.
- N2: datos del MISMO dominio en otro tramo (1,1–2,7 M€ con `[60000, 900000]`) entrenaban con R² 0,994 sin la guarda,
  y se rechazaban diciendo que el modelo «contestaría siempre lo mismo».

El criterio es el RECORRIDO del objetivo normalizado con el rango, `(máx − mín) / (hi − lo)`. La 3.ª pasada lo midió
con datos APRENDIBLES y lo movió: por debajo del 4 % (no del 1 %) se rechaza, porque nada de lo medido ahí sirve (R²
−17,9 al 1,1 %, −1,6 al 3 %); con ≥ 90 % fuera, se rechaza si el recorrido no es comparable al del rango (fuera de
[0,1; 20]: 10,5 y 20 veces se aprenden con R² 0,997 y 0,984); y «más de 100 veces» con casi todo DENTRO ya no se
rechaza: es un extremo de validación (B-R3.1, que rompía A8). Los casos son los de las sondas de la 2.ª y la 3.ª pasada.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest.
"""
from __future__ import annotations

import csv
import io
import math
import re
import unittest.mock

import pytest

from matrixai.agents import ChatCompletionsLLMProposalProvider
from matrixai.playground import (
    _numero_legible,
    _objetivo_fuera_de_la_escala_del_rango as guarda,
    _run_playground_training,
    analyze_playground_request,
)
from matrixai.playground_api import generate_synthetic_dataset

MXAI = """PROJECT P

VECTOR Input[2]
  superficie_m2: Scalar
  habitaciones: Scalar
END

NETWORK R
  INPUT Input
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT predicted_value: Scalar
END
"""
RANGO = (60000.0, 900000.0)


def _csv(valores) -> str:
    return "superficie_m2,predicted_value\n" + "".join(f"100,{v}\n" for v in valores)


def _entre(a: float, b: float, n: int = 200) -> list[float]:
    return [a + (b - a) * i / (n - 1) for i in range(n)]


CLASICA = _entre(-1.0, 1.0)  # lo que genera la clásica SIN rango

# ── el criterio, caso a caso (la sonda del re-auditor) ─────────────────────────────────────────────

_SE_RECHAZA = [
    ("la clásica con [60000, 900000]", CLASICA, (60000, 900000)),
    ("la clásica con [1, 100]", CLASICA, (1, 100)),
    ("la clásica con [1, 1000]", CLASICA, (1, 1000)),
    ("la clásica con [10, 50000]", CLASICA, (10, 50000)),
    ("la clásica con [100, 200000]", CLASICA, (100, 200000)),
    ("la clásica con [500, 900000] (N1)", CLASICA, (500, 900000)),
    ("la clásica con [1000, 900000]", CLASICA, (1000, 900000)),
    ("la clásica con [0, 900000] (N1)", CLASICA, (0, 900000)),
    ("la clásica con [0.5, 1000]", CLASICA, (0.5, 1000)),
    ("la clásica con [-1000, 1000] (N1)", CLASICA, (-1000, 1000)),
    ("la clásica con [273.15, 373.15]", CLASICA, (273.15, 373.15)),
    ("k€ con el rango en €", _entre(60, 900), RANGO),
    ("€ con el rango en k€", _entre(60000, 900000), (60, 900)),
    ("recorrido del 0,5 % dentro del rango", _entre(100000, 104500), (0, 900000)),
    # 3.ª pasada (R3-1): lo que la 2.ª declaraba «deuda suave» no lo era con datos aprendibles (R² −4,8 con [0, 100]).
    ("la clásica con [0, 100] (antes deuda)", CLASICA, (0, 100)),
    ("la clásica con [-20, 45] (antes deuda)", CLASICA, (-20, 45)),
    ("la clásica con [-50, 50]", CLASICA, (-50, 50)),
    ("recorrido del 2 % dentro del rango (R3-1: R² −4,7)", _entre(100000, 118000), (0, 900000)),
    ("recorrido del 3 % dentro del rango (R3-1: R² −1,6)", _entre(100000, 127000), (0, 900000)),
    ("todo fuera, 50 veces el ancho (R² −0,08)", _entre(1_000_000, 1_000_000 + 50 * 840000), RANGO),
]

_SE_ENTRENA = [
    ("generado con el rango", _entre(60000, 900000), RANGO),
    ("bordes exactos", [60000.0] * 50 + [900000.0] * 50, RANGO),
    ("10 % de atípicos ×6", _entre(60000, 900000, 180) + [5_400_000.0] * 20, RANGO),
    ("×1,5 (una parte por encima del máximo)", _entre(90000, 1_350_000), RANGO),
    ("mismo dominio, otro mercado (N2)", _entre(1_100_000, 2_700_000), RANGO),
    ("un extremo de validación a 120 veces el ancho (B-R3.1: un 61,53 escrito 6153)",
     _entre(60000, 900000, 199) + [60000 + 120 * 840000.0], RANGO),
    ("todo fuera, 15 veces el ancho (R3-2: R² 0,98–0,997)", _entre(1_000_000, 1_000_000 + 15 * 840000), RANGO),
    ("SpO2 en [0, 100], 4,7 % (R² 0,88)", _entre(94.3, 99.0), (0, 100)),
]


@pytest.mark.parametrize("caso,valores,rango", _SE_RECHAZA, ids=[c[0] for c in _SE_RECHAZA])
def test_se_rechaza(caso, valores, rango):
    r = guarda(MXAI, _csv(valores), rango)
    assert r is not None, caso
    assert r["error_kind"] == "objetivo_fuera_del_rango"
    assert r["motivo_del_rechazo"] in ("otra_escala", "rango_mucho_mas_ancho_que_los_datos")


@pytest.mark.parametrize("caso,valores,rango", _SE_ENTRENA, ids=[c[0] for c in _SE_ENTRENA])
def test_se_entrena(caso, valores, rango):
    assert guarda(MXAI, _csv(valores), rango) is None, caso


@pytest.mark.parametrize("rango,parte,parte_en", [((0, 900000), "menos del 0,01 %", "less than 0.01 %"),
                                                  ((-1000, 1000), "solo el 0,1 %", "only 0.1 %")])
def test_dentro_del_rango_pero_en_un_trozo_infimo_el_motivo_es_ese_y_no_otra_escala(rango, parte, parte_en):
    """N1: los datos de la clásica caen DENTRO de estos rangos (casi todos): decir «otra escala» sería falso. El motivo
    es que ocupan una parte ínfima del rango, y la frase dice cuánta."""
    r = guarda(MXAI, _csv(CLASICA), rango)
    assert r["motivo_del_rechazo"] == "rango_mucho_mas_ancho_que_los_datos", r
    assert f"ocupa {parte} del rango" in r["error"] and f"covers {parte_en} of" in r["error_en"], r["error"]


def test_los_umbrales_del_recorrido_en_sus_bordes():
    """4 %: un recorrido del 4,1 % dentro del rango entrena y uno del 3,9 % no. La ventana «comparable», con TODO
    fuera: 0,11 y 19 veces el ancho entrenan; 0,09 y 21 veces, no. Y un recorrido de 1000 veces con casi todo DENTRO
    (un extremo de validación) entrena: «más de 100» ya no rechaza solo (B-R3.1)."""
    assert guarda(MXAI, _csv(_entre(100000, 100000 + 0.041 * 840000)), RANGO) is None
    assert guarda(MXAI, _csv(_entre(100000, 100000 + 0.039 * 840000)), RANGO)["motivo_del_rechazo"] == \
        "rango_mucho_mas_ancho_que_los_datos"
    for veces, entrena in ((0.11, True), (19, True), (0.09, False), (21, False)):
        r = guarda(MXAI, _csv(_entre(1_000_000, 1_000_000 + veces * 840000)), RANGO)
        assert (r is None) is entrena, (veces, r and r["motivo_del_rechazo"])
    dentro = _entre(60000, 900000, 99)
    assert guarda(MXAI, _csv(dentro + [60000 + 1000 * 840000.0]), RANGO) is None


# ── por el camino de verdad: entrenar ──────────────────────────────────────────────────────────────

FEATURES = ("predecir el precio de una vivienda\nFEATURES:\n  superficie_m2: Scalar en [30, 400]\n"
            "  habitaciones: Integer[1, 10]\n  antiguedad_anios: Integer[0, 100]\n"
            "  distancia_centro_km: Scalar en [0, 40]\n")
RECETA = ("precio_eur = 1500*superficie_m2 + 10000*habitaciones - 300*antiguedad_anios - 1000*distancia_centro_km "
          "+ 100000\nNOISE: 0.02")


def _red(salida: str) -> dict:
    with unittest.mock.patch.object(ChatCompletionsLLMProposalProvider, "from_env", side_effect=ValueError("x")):
        r = analyze_playground_request({"mode": "prompt", "prompt": FEATURES + salida})
    assert r["ok"] and r["rango_declarado_de_la_salida"], r
    return r


def _rangos(r: dict) -> dict:
    return {k: tuple(v) for k, v in r["field_ranges"].items()}


def _generar(r: dict, **kw) -> str:
    d = generate_synthetic_dataset(r["mxai"], r["training_text"], 200, 42, "coherent", False,
                                   field_ranges_override=_rangos(r), field_types=r.get("field_types"), **kw)
    assert d["ok"], d
    return d["csv_text"]


@pytest.mark.parametrize("salida", ["SALIDA: precio_eur: Scalar en [0, 900000]",
                                    "SALIDA: beneficio_eur: Scalar en [-1000, 1000]"])
def test_N1_los_datos_de_la_clasica_con_un_rango_que_contiene_menos_uno_uno_no_se_entrenan(salida):
    r = _red(salida)
    rango = tuple(r["rango_declarado_de_la_salida"])
    csv_clasica = _generar(r)  # la clásica: SIN rango → objetivo en [-1, 1]
    t = _run_playground_training(r["mxai"], r["training_text"], csv_clasica, epochs_override=5,
                                 field_ranges=_rangos(r), target_range=rango, seed=42, recortar_objetivo=False)
    assert t["ok"] is False and t.get("error_kind") == "objetivo_fuera_del_rango", \
        {k: t.get(k) for k in ("ok", "mae", "r2", "model_collapsed")}
    assert t["motivo_del_rechazo"] == "rango_mucho_mas_ancho_que_los_datos"


def test_N2_datos_del_mismo_dominio_fuera_del_rango_se_entrenan():
    """Otro mercado (1,1–2,7 M€) con `[60000, 900000]`: sin la guarda, R² 0,994 (medido). Se entrena, y bien."""
    r = _red("SALIDA: precio_eur: Scalar en [60000, 900000]")
    rango = tuple(r["rango_declarado_de_la_salida"])
    base = _generar(r, recipe_text=RECETA, target_range=rango)
    filas = list(csv.reader(io.StringIO(base)))
    j = filas[0].index("predicted_value")
    o = io.StringIO()
    w = csv.writer(o, lineterminator="\n")
    w.writerow(filas[0])
    for f in filas[1:]:
        f = list(f)
        f[j] = str(round(1_000_000 + (float(f[j]) - 60000) * 2.4, 2))
        w.writerow(f)
    t = _run_playground_training(r["mxai"], r["training_text"], o.getvalue(), epochs_override=40,
                                 field_ranges=_rangos(r), target_range=rango, seed=42, recortar_objetivo=False)
    assert t.get("ok"), t.get("error")
    assert t["r2"] > 0.9, t["r2"]


# ── N5: la columna que se mira es la de SALIDA, no la última ──────────────────────────────────────

def test_la_guarda_mira_la_salida_aunque_vaya_la_primera():
    csv_text = "predicted_value,habitaciones,superficie_m2\n" + "".join(
        f"{(i % 20) / 10 - 1},3,{500000 + i}\n" for i in range(50))
    r = guarda(MXAI, csv_text, RANGO)
    assert r is not None and r["columna_objetivo"] == "predicted_value", r
    assert (r["valores_fuera"], r["valores_con_dato"]) == (50, 50)


def test_y_no_rechaza_por_una_entrada_que_va_la_ultima():
    csv_text = "superficie_m2,predicted_value,habitaciones\n" + "".join(
        f"100,{300000 + 1000 * i},{i % 10 / 10}\n" for i in range(50))
    assert guarda(MXAI, csv_text, RANGO) is None


# ── los textos ─────────────────────────────────────────────────────────────────────────────────────

_FUNCIONALES = re.compile(r"\b(el|la|los|las|del|de|que|con|estos|datos|escala|rango|valores|quedan|modelo|"
                          r"menos|solo|vale|siempre|va)\b", re.IGNORECASE)


@pytest.mark.parametrize("valores,rango,motivo", [
    (CLASICA, RANGO, "otra_escala"),
    (CLASICA, (0, 900000), "rango_mucho_mas_ancho_que_los_datos"),
    ([500000.0] * 50, RANGO, "rango_mucho_mas_ancho_que_los_datos"),
    ([0.5] * 50, RANGO, "otra_escala"),
    (CLASICA, (5, 5), "rango_invalido"),
])
def test_el_motivo_en_ingles_no_lleva_castellano(valores, rango, motivo):
    r = guarda(MXAI, _csv(valores), rango)
    assert r["motivo_del_rechazo"] == motivo
    assert _FUNCIONALES.search(r["error_en"]) is None, r["error_en"]
    assert _FUNCIONALES.search(r["error"]) is not None  # el control: el castellano sí las tiene


def test_los_dos_motivos_no_dicen_lo_mismo():
    """N2: cada motivo, su frase. Y ninguno dice «contestaría siempre lo mismo»: la 3.ª pasada midió que el modelo de
    I1 SÍ cambia de respuesta (−23.338 € y −16.237 €), solo que sin parecerse a los datos."""
    infimo = guarda(MXAI, _csv(CLASICA), (0, 900000))
    otra = guarda(MXAI, _csv(_entre(60, 900)), RANGO)
    assert "apenas cambia de una fila a otra" in infimo["error"] and "barely changes" in infimo["error_en"]
    assert "apenas cambia" not in otra["error"] and "no está en la escala" in otra["error"]
    for r in (infimo, otra):
        assert "siempre lo mismo" not in r["error"] and "always answer the same" not in r["error_en"]
    # La otra escala ofrece también lo que sí sirve: un rango a la escala de los datos (3.ª pasada, R3-2).
    assert "declara un rango de salida a la escala de tus datos" in otra["error"]
    assert "declare an output range on the scale of your data" in otra["error_en"]


def test_un_objetivo_constante_se_dice_constante():
    r = guarda(MXAI, _csv([500000.0] * 50), RANGO)
    assert "vale siempre 500000" in r["error"] and "is always 500000" in r["error_en"], r["error"]
    assert "va de 500000 a 500000" not in r["error"]


def test_los_numeros_de_la_frase_sin_notacion_cientifica_N6_y_con_coma_en_castellano_R3_5():
    r = guarda(MXAI, _csv(_entre(60000.5, 1137473.26)), (60, 900))
    for texto in (r["error"], r["error_en"]):
        assert "e+" not in texto and "1137473" in texto, texto
    assert "60000,5" in r["error"] and "60000.5" in r["error_en"]
    # El caso de M6 (k€ con el rango en €): «117.279» se leía como ciento diecisiete MIL, dentro del rango.
    r = guarda(MXAI, _csv(_entre(117.279, 774.456)), RANGO)
    assert "117,279" in r["error"] and "117.279" not in r["error"] and "117.279" in r["error_en"]


def test_un_objetivo_constante_fuera_tambien_se_dice_constante():
    r = guarda(MXAI, _csv([0.5] * 50), RANGO)
    assert r["motivo_del_rechazo"] == "otra_escala"
    assert "valen siempre 0,5" in r["error"] and "they are always 0.5" in r["error_en"], r["error"]


def test_el_porcentaje_no_redondea_hasta_el_umbral():
    """3,96 % se rechaza por estar por debajo del 4 %: la frase no puede decir «4 %»."""
    r = guarda(MXAI, _csv(_entre(100000, 100000 + 0.0396 * 840000)), RANGO)
    assert "solo el 3,9 %" in r["error"] and "only 3.9 %" in r["error_en"], r["error"]


@pytest.mark.parametrize("v,texto", [(1000000.5, "1000000"), (2500000.25, "2500000"), (1137473.26, "1137473"),
                                     (100.5, "100.5"), (58320.0, "58320"), (60000.0, "60000"), (0.0, "0"),
                                     (-1000.0, "-1000"), (0.5, "0.5"), (273.15, "273.15")])
def test_numero_legible(v, texto):
    """Sin «.0», sin «e+06», y sin comerse los ceros de un entero («1000000.5» daba «1» en la 1.ª versión). En
    castellano, con coma decimal (R3-5)."""
    assert _numero_legible(v) == texto
    assert _numero_legible(v, ",") == texto.replace(".", ",")


# ── N7 / R3-3: un rango que no es un par finito y creciente se RECHAZA con su motivo ───────────────

@pytest.mark.parametrize("rango", [(900000, 60000), (5, 5), (math.nan, 1), (0, math.inf), (-math.inf, 0)])
def test_un_rango_imposible_se_rechaza_con_su_motivo(rango):
    """3.ª pasada, R3-3: la 2.ª lo apartaba de la guarda, y entonces `[5, 5]` reventaba la normalización
    (`ZeroDivisionError`, el servidor cerraba la conexión) y `[900000, 60000]` «entrenaba» con un MAE negativo."""
    r = guarda(MXAI, _csv(CLASICA), rango)
    assert r["motivo_del_rechazo"] == "rango_invalido" and r["error_kind"] == "objetivo_fuera_del_rango"
    assert "el mínimo tiene que ser un número menor que el máximo" in r["error"]
    assert "the minimum has to be a number below the maximum" in r["error_en"]


@pytest.mark.parametrize("rango", [(5.0, 5.0), (900000.0, 60000.0)])
def test_un_rango_imposible_no_llega_a_normalizar_ni_a_entrenar(rango):
    r = _red("SALIDA: precio_eur: Scalar en [60000, 900000]")
    csv_text = _generar(r, recipe_text=RECETA, target_range=(60000.0, 900000.0))
    t = _run_playground_training(r["mxai"], r["training_text"], csv_text, epochs_override=2, field_ranges=_rangos(r),
                                 target_range=rango, seed=42, recortar_objetivo=False)
    assert t["ok"] is False and t["motivo_del_rechazo"] == "rango_invalido", {k: t.get(k) for k in ("ok", "mae")}
