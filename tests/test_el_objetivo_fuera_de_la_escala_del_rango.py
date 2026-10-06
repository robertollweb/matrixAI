# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""UN OBJETIVO QUE NO ESTÁ EN LA ESCALA DE SU RANGO NO SE ENTRENA — auditoría del rango, I1/M6 (06-10).

La auditoría independiente de «el rango declarado de la salida de regresión del prompt se usa» (1.ª pasada,
SUSPENDE) lo midió por HTTP: un modelo de regresión guardado con su rango (`SALIDA: precio_eur: Scalar en
[60000, 900000]`) y reabierto en la clásica GENERA sin el rango —objetivo en [-1, 1]— y ENTRENA con él. Salía
`done`, MAE «17.432 €», R² −1,16·10⁹ y la MISMA predicción para cualquier vivienda: un modelo constante con un
error en euros de buen aspecto. Lo mismo con un CSV propio en k€ y el rango en euros (M6).

Ahora el núcleo lo RECHAZA antes de normalizar, con su motivo (`error`/`error_en`, `error_kind:
objetivo_fuera_del_rango`), por las dos entradas que normalizan el objetivo con `target_range`
(`run_playground_training` y `submit_training_job`). El criterio, desde la 2.ª pasada (N1/N2): el RECORRIDO del
objetivo normalizado con el rango —menos del 1 %, o más de 100 veces— y, si no es comparable (fuera de [0,1; 10]),
el 90 % de los valores no nulos fuera de `[lo, hi]` con una tolerancia de REDONDEO. Sus casos, en
`test_el_recorrido_del_objetivo_con_su_rango.py`. Y lo que no es eso entrena IGUAL que antes: lo generado con el
rango, unos pocos atípicos, y todo lo que no trae rango.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest.
"""
from __future__ import annotations

import json
import unittest.mock

import pytest

import matrixai.playground as playground
from matrixai.playground import _objetivo_fuera_de_la_escala_del_rango, analyze_playground_request
from matrixai.playground_api import (
    generate_synthetic_dataset,
    run_playground_training,
    submit_training_job,
    training_jobs,
)

PROMPT = ("predecir el precio de una vivienda\nFEATURES:\n  superficie_m2: Scalar en [30, 400]\n"
          "  habitaciones: Integer[1, 10]\nSALIDA: precio_eur: Scalar en [60000, 900000]")
RANGO = (60000.0, 900000.0)

#: Lo que se MIDE de un entrenamiento, para comparar dos: todo menos lo que cambia en cada pasada por
#: construcción (`run_id`, y las rutas temporales que lleva dentro el informe de evaluación).
_LO_QUE_SE_MIDE = ("ok", "error", "mae", "rmse", "r2", "epochs", "final_train_loss", "best_validation_loss",
                   "best_epoch", "params_best", "target_range", "task_kind", "metrics")


@pytest.fixture(scope="module")
def red():
    r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT})
    assert r["ok"], r.get("error")
    assert tuple(r["rango_declarado_de_la_salida"]) == RANGO
    return r


def _rangos(r) -> dict:
    return {k: tuple(v) for k, v in (r.get("field_ranges") or {}).items()}


def _generar(r, filas: int = 80, **kw) -> str:
    d = generate_synthetic_dataset(r["mxai"], r["training_text"], filas, 7, "coherent", False,
                                   field_ranges_override=_rangos(r) or None,
                                   field_types=r.get("field_types") or None, **kw)
    assert d["ok"], d
    return d["csv_text"]


def _objetivo(csv_text: str) -> list[float]:
    filas = csv_text.splitlines()
    j = filas[0].split(",").index("predicted_value")
    return [float(f.split(",")[j]) for f in filas[1:]]


def _con_objetivo(csv_text: str, valores: list) -> str:
    """El mismo CSV con otra columna objetivo (el resto, intacto)."""
    filas = csv_text.splitlines()
    j = filas[0].split(",").index("predicted_value")
    assert len(valores) == len(filas) - 1
    salida = [filas[0]]
    for fila, v in zip(filas[1:], valores):
        celdas = fila.split(",")
        celdas[j] = str(v)
        salida.append(",".join(celdas))
    return "\n".join(salida) + "\n"


def _entrenar(r, csv_text: str, **kw) -> dict:
    return run_playground_training(r["mxai"], r["training_text"], csv_text, epochs_override=2,
                                   field_ranges=_rangos(r), seed=42, **kw)


def _medido(t: dict) -> str:
    return json.dumps({k: t.get(k) for k in _LO_QUE_SE_MIDE}, sort_keys=True, default=str)


def _sin_la_guarda():
    """El núcleo de antes de la guarda: para comparar que lo que debe entrenar entrena IGUAL."""
    return unittest.mock.patch.object(playground, "_objetivo_fuera_de_la_escala_del_rango",
                                      lambda *a, **k: None)


def _ni_normaliza_ni_entrena():
    """Si algo llega a normalizar o a entrenar, la prueba revienta: el rechazo tiene que ser ANTES."""
    no = AssertionError("se normalizó o se entrenó un objetivo fuera de la escala de su rango")
    return (unittest.mock.patch.object(playground, "_normalize_csv_with_ranges", side_effect=no),
            unittest.mock.patch.object(playground, "_run_playground_dense_training", side_effect=no))


# ── (a) el caso de I1: datos en [-1, 1] con el rango del modelo guardado ──────────────────────────────

def test_los_datos_de_la_clasica_con_el_rango_del_modelo_guardado_se_rechazan_sin_entrenar(red):
    """La clásica genera SIN el rango (objetivo en [-1, 1]) y entrena CON el del modelo guardado."""
    csv_text = _generar(red)
    assert all(-1.0 <= v <= 1.0 for v in _objetivo(csv_text))
    a, b = _ni_normaliza_ni_entrena()
    with a, b:
        t = _entrenar(red, csv_text, target_range=RANGO)
    assert t["ok"] is False
    assert t["error_kind"] == "objetivo_fuera_del_rango"
    assert t["columna_objetivo"] == "predicted_value"
    assert t["rango_declarado"] == [60000.0, 900000.0]
    assert t["valores_fuera"] == t["valores_con_dato"] == 80
    # El motivo, para personas y en los dos idiomas: qué escala dice el modelo y cuál traen los datos.
    for clave in ("error", "error_en"):
        assert "predicted_value" in t[clave] and "60000" in t[clave] and "900000" in t[clave], t[clave]
        assert "80" in t[clave], t[clave]
    assert "escala" in t["error"] and "scale" in t["error_en"]
    # Ni una métrica: no hay un MAE «en euros» que enseñar.
    assert t.get("mae") is None and t.get("r2") is None


def test_por_la_puerta_del_studio_tampoco_se_crea_el_trabajo(red):
    """`submit_training_job` (la de `/api/train-start`): el mismo rechazo, y ningún trabajo en el registro."""
    csv_text = _generar(red)
    antes = set(training_jobs)
    a, b = _ni_normaliza_ni_entrena()
    with a, b:
        t = submit_training_job(red["mxai"], red["training_text"], csv_text, 2, field_ranges=_rangos(red),
                                target_range=RANGO, recortar_objetivo=False)
    assert t["ok"] is False and t["error_kind"] == "objetivo_fuera_del_rango", t
    assert "job_id" not in t
    assert set(training_jobs) == antes


def test_un_csv_propio_en_miles_con_el_rango_en_euros_se_rechaza(red):
    """M6: el objetivo en k€ (60–900) y el rango declarado en euros."""
    csv_text = _generar(red, target_range=RANGO)
    en_miles = _con_objetivo(csv_text, [round(v / 1000.0, 4) for v in _objetivo(csv_text)])
    t = _entrenar(red, en_miles, target_range=RANGO)
    assert t["ok"] is False and t["error_kind"] == "objetivo_fuera_del_rango", t
    assert t["valores_fuera"] == 80


# ── (b) lo generado con el rango entrena IGUAL que antes de la guarda ─────────────────────────────────

def test_el_csv_generado_con_el_rango_entrena_igual_que_sin_la_guarda(red):
    csv_text = _generar(red, target_range=RANGO)
    con = _entrenar(red, csv_text, target_range=RANGO)
    with _sin_la_guarda():
        sin = _entrenar(red, csv_text, target_range=RANGO)
    assert con["ok"], con.get("error")
    # El error, en euros (la escala del dominio), no en la normalizada.
    assert con["mae"] is not None and con["mae"] > 1000.0, con["mae"]
    assert _medido(con) == _medido(sin)


# ── (c) unos pocos atípicos fuera del rango: se entrenan, como hoy ────────────────────────────────────

def test_unos_pocos_atipicos_fuera_del_rango_entrenan_como_hoy(red):
    csv_text = _generar(red, target_range=RANGO)
    valores = _objetivo(csv_text)
    # 10 de 80 (12,5 %) muy por encima del rango: datos de verdad con atípicos, no otra escala.
    valores[:10] = [5_000_000.0] * 10
    con_atipicos = _con_objetivo(csv_text, valores)
    assert _objetivo_fuera_de_la_escala_del_rango(red["mxai"], con_atipicos, RANGO) is None
    con = _entrenar(red, con_atipicos, target_range=RANGO)
    with _sin_la_guarda():
        sin = _entrenar(red, con_atipicos, target_range=RANGO)
    assert con["ok"], con.get("error")
    assert _medido(con) == _medido(sin)


# ── el criterio, en sus bordes ─────────────────────────────────────────────────────────────────────

def _csv_de_objetivo(valores: list) -> str:
    """Un CSV mínimo con la columna objetivo del modelo (la guarda solo mira esa)."""
    return "superficie_m2,predicted_value\n" + "".join(f"100,{v}\n" for v in valores)


@pytest.mark.parametrize("fuera,se_rechaza", [(100, True), (90, True), (89, False), (10, False), (0, False)])
def test_se_rechaza_desde_el_90_por_ciento_fuera(red, fuera, se_rechaza):
    """El umbral del 90 %, con un recorrido NO comparable (del 2 al 9 % del ancho): ni tan estrecho que lo rechace
    solo, ni parecido al del rango (eso es otro tramo del mismo dominio, y se entrena: N2)."""
    valores = [0.5] * fuera + [60000.0 + 200 * i for i in range(100 - fuera)]
    r = _objetivo_fuera_de_la_escala_del_rango(red["mxai"], _csv_de_objetivo(valores), RANGO)
    assert (r is not None) is se_rechaza, r
    if se_rechaza:
        assert (r["valores_fuera"], r["valores_con_dato"]) == (fuera, 100)
        assert r["motivo_del_rechazo"] == "otra_escala"


def test_un_valor_en_el_borde_por_redondeo_cuenta_como_dentro(red):
    """La tolerancia es de REDONDEO (1e-4 + 1e-6·|borde|: 0,9 € aquí), no del ancho (2.ª pasada, N1: el 0,1 % del
    ancho metía [-1, 1] «dentro» de [0, 900000]). Medio euro por debajo del mínimo cuenta dentro; dos, fuera."""
    assert playground._tolerancia_de_redondeo(*RANGO) == pytest.approx(0.9001)
    otra_escala = [0.5] * 90
    redondeo = _objetivo_fuera_de_la_escala_del_rango(
        red["mxai"], _csv_de_objetivo(otra_escala + [RANGO[0] - 0.5] * 10), RANGO)
    assert (redondeo["valores_fuera"], redondeo["valores_con_dato"]) == (90, 100)
    fuera = _objetivo_fuera_de_la_escala_del_rango(
        red["mxai"], _csv_de_objetivo(otra_escala + [RANGO[0] - 2.0] * 10), RANGO)
    assert (fuera["valores_fuera"], fuera["valores_con_dato"]) == (100, 100)


def test_lo_que_no_es_un_numero_no_cuenta_ni_dentro_ni_fuera(red):
    """Ausente no es cero: los vacíos, `nan` y el texto no entran en la cuenta."""
    nulos = ["", "nan", "NaN", "n/a"] * 5
    solo_fuera = _objetivo_fuera_de_la_escala_del_rango(red["mxai"], _csv_de_objetivo(nulos + [0.5]), RANGO)
    assert solo_fuera is not None and (solo_fuera["valores_fuera"], solo_fuera["valores_con_dato"]) == (1, 1)
    # 19 dentro y 1 fuera entre 20 nulos: el 5 %, se entrena.
    mezcla = nulos + [500000.0] * 19 + [0.5]
    assert _objetivo_fuera_de_la_escala_del_rango(red["mxai"], _csv_de_objetivo(mezcla), RANGO) is None
    # Y sin un solo número, no hay nada que decidir aquí.
    assert _objetivo_fuera_de_la_escala_del_rango(red["mxai"], _csv_de_objetivo(nulos), RANGO) is None


# ── (d) sin `target_range`, todo como hoy ──────────────────────────────────────────────────────────

def test_sin_rango_no_hay_guarda_y_se_entrena_como_hoy(red):
    """El CSV de la clásica (objetivo en [-1, 1]) SIN rango: el mismo entrenamiento que sin la guarda."""
    csv_text = _generar(red)
    assert _objetivo_fuera_de_la_escala_del_rango(red["mxai"], csv_text, None) is None
    con = _entrenar(red, csv_text)
    with _sin_la_guarda():
        sin = _entrenar(red, csv_text)
    assert con["ok"], con.get("error")
    assert con["target_range"] is None
    assert _medido(con) == _medido(sin)
