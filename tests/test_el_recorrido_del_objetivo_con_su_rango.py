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

El criterio de ahora es el RECORRIDO del objetivo normalizado con el rango, `(máx − mín) / (hi − lo)`: menos del 1 %
se rechaza (al normalizar queda casi igual en todas las filas), más de 100 veces también, y entre medias solo con
≥ 90 % fuera Y un recorrido que no se parece al del rango (fuera de [0,1; 10]). Los casos de abajo son los 24 de la
sonda del re-auditor (`sonda_criterio`), con los dos suaves que deja pasar declarados como deuda.

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
]

_SE_ENTRENA = [
    ("generado con el rango", _entre(60000, 900000), RANGO),
    ("bordes exactos", [60000.0] * 50 + [900000.0] * 50, RANGO),
    ("10 % de atípicos ×6", _entre(60000, 900000, 180) + [5_400_000.0] * 20, RANGO),
    ("×1,5 (una parte por encima del máximo)", _entre(90000, 1_350_000), RANGO),
    ("mismo dominio, otro mercado (N2)", _entre(1_100_000, 2_700_000), RANGO),
    # DEUDA declarada (2.ª pasada): dos rangos estrechos que contienen [-1, 1] dejan pasar a la clásica. Recorridos
    # del 2 y del 3 %, y suaves: MAE 0,64 y 0,54 frente a 0,48 sin rango (medido por el re-auditor).
    ("deuda: la clásica con [0, 100]", CLASICA, (0, 100)),
    ("deuda: la clásica con [-20, 45]", CLASICA, (-20, 45)),
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


@pytest.mark.parametrize("rango,parte,parte_en", [((0, 900000), "menos del 0.01 %", "less than 0.01 %"),
                                                  ((-1000, 1000), "solo el 0.1 %", "only 0.1 %")])
def test_dentro_del_rango_pero_en_un_trozo_infimo_el_motivo_es_ese_y_no_otra_escala(rango, parte, parte_en):
    """N1: los datos de la clásica caen DENTRO de estos rangos (casi todos): decir «otra escala» sería falso. El motivo
    es que ocupan una parte ínfima del rango, y la frase dice cuánta."""
    r = guarda(MXAI, _csv(CLASICA), rango)
    assert r["motivo_del_rechazo"] == "rango_mucho_mas_ancho_que_los_datos", r
    assert f"ocupa {parte} del rango" in r["error"] and f"covers {parte_en} of" in r["error_en"], r["error"]


def test_los_umbrales_del_recorrido_en_sus_bordes():
    """1 %: un recorrido del 1,1 % dentro del rango entrena y uno del 0,9 % no. 100 veces: con la mitad de los valores
    DENTRO (así no decide el 90 %), un recorrido de 90 veces el ancho entrena y uno de 110 no."""
    assert guarda(MXAI, _csv(_entre(100000, 100000 + 0.011 * 840000)), RANGO) is None
    assert guarda(MXAI, _csv(_entre(100000, 100000 + 0.009 * 840000)), RANGO) is not None
    dentro = _entre(60000, 900000, 100)
    assert guarda(MXAI, _csv(dentro + [60000 + 90 * 840000.0] * 100), RANGO) is None
    assert guarda(MXAI, _csv(dentro + [60000 + 110 * 840000.0] * 100), RANGO)["motivo_del_rechazo"] == "otra_escala"


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
])
def test_el_motivo_en_ingles_no_lleva_castellano(valores, rango, motivo):
    r = guarda(MXAI, _csv(valores), rango)
    assert r["motivo_del_rechazo"] == motivo
    assert _FUNCIONALES.search(r["error_en"]) is None, r["error_en"]
    assert _FUNCIONALES.search(r["error"]) is not None  # el control: el castellano sí las tiene


def test_los_dos_motivos_no_dicen_lo_mismo():
    """N2: «contestaría siempre lo mismo» es el motivo del trozo ínfimo; el de la otra escala no lo dice."""
    infimo = guarda(MXAI, _csv(CLASICA), (0, 900000))
    otra = guarda(MXAI, _csv(_entre(60, 900)), RANGO)
    assert "siempre lo mismo" in infimo["error"] and "always answer the same" in infimo["error_en"]
    assert "siempre lo mismo" not in otra["error"] and "always answer the same" not in otra["error_en"]


def test_un_objetivo_constante_se_dice_constante():
    r = guarda(MXAI, _csv([500000.0] * 50), RANGO)
    assert "vale siempre 500000" in r["error"] and "is always 500000" in r["error_en"], r["error"]
    assert "va de 500000 a 500000" not in r["error"]


def test_los_numeros_de_la_frase_sin_notacion_cientifica_N6():
    r = guarda(MXAI, _csv(_entre(60000.5, 1137473.26)), (60, 900))
    for texto in (r["error"], r["error_en"]):
        assert "e+" not in texto and "1137473" in texto and "60000.5" in texto, texto


@pytest.mark.parametrize("v,texto", [(1000000.5, "1000000"), (2500000.25, "2500000"), (1137473.26, "1137473"),
                                     (100.5, "100.5"), (58320.0, "58320"), (60000.0, "60000"), (0.0, "0"),
                                     (-1000.0, "-1000"), (0.5, "0.5"), (273.15, "273.15")])
def test_numero_legible(v, texto):
    """Sin «.0», sin «e+06», y sin comerse los ceros de un entero («1000000.5» daba «1» en la 1.ª versión)."""
    assert _numero_legible(v) == texto


# ── N7: un rango que no es un par finito y creciente no lo decide la guarda ────────────────────────

@pytest.mark.parametrize("rango", [(900000, 60000), (5, 5), (math.nan, 1), (0, math.inf), (-math.inf, 0)])
def test_un_rango_imposible_no_lo_decide_la_guarda(rango):
    assert guarda(MXAI, _csv(CLASICA), rango) is None
