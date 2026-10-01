# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-WEB (campo) — la tabla de los 40 conjuntos en el campo de la red nueva sale de los registros
sellados, nunca de una mano, y el generador PARA si el control (los totales del recuento) no cuadra.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "datos_publicos_generar_campo_119", RAIZ / "benchmarks" / "datos_publicos" / "generar_campo_119.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _publicado():
    return json.loads(gen.RUTA_PUBLICADO.read_text(encoding="utf-8"))


def test_lo_publicado_es_BYTE_A_BYTE_lo_que_sale_de_los_registros():
    assert gen.RUTA_PUBLICADO.read_text(encoding="utf-8") == gen.generar(), (
        "campo_119_publico.json no es lo que sale de los registros: "
        "python3 benchmarks/datos_publicos/generar_campo_119.py --escribir, y mirar por qué cambió")


def test_los_totales_son_los_del_recuento_con_el_motor_nuevo_en_el_campo():
    t = _publicado()["totales"]
    assert {m: (x["cumplidos"], x["mejor_en"]) for m, x in t.items()} == {
        "baseline": (0, 0), "catboost": (25, 8), "lightgbm": (26, 2), gen.MOTOR_NUEVO: (38, 23),
        "sklearn.hgb": (25, 2), "sklearn.lineal": (13, 2), "xgboost": (24, 3)}
    assert all(x["de"] == 40 for x in t.values())


def test_la_cuenta_sin_la_red_nueva_es_la_de_la_v2():
    s = _publicado()["sin_la_red_nueva"]["por_motor"]
    assert {m: x["cumplidos"] for m, x in s.items()} == {
        "catboost": 29, "lightgbm": 33, gen.RED_ANTERIOR: 16, "sklearn.hgb": 33,
        "sklearn.lineal": 14, "xgboost": 29}


def test_el_detalle_por_conjunto_suma_los_totales():
    p = _publicado()
    cs = p["conjuntos"]
    assert len(cs) == 40 and len({c["nombre"] for c in cs}) == 40
    for m, t in p["totales"].items():
        if m == "baseline":
            continue
        assert sum(c["celdas"][m]["cumple"] for c in cs) == t["cumplidos"], m
        assert sum(c["celdas"][m]["es_el_mejor"] for c in cs) == t["mejor_en"], m
    assert sum(c["mejor"] == gen.MOTOR_NUEVO for c in cs) == 23
    assert sum(c["red_anterior"]["cumple"] for c in cs) == 16
    # Quién es el mejor es el que tiene la media más alta entre los que compiten.
    for c in cs:
        medias = {m: x["media"] for m, x in c["celdas"].items() if x["compite"] and x["media"] is not None}
        assert c["mejor"] == max(medias, key=medias.get), c["nombre"]


def test_la_red_nueva_no_cumple_en_dos_conjuntos_y_se_dicen():
    cs = _publicado()["conjuntos"]
    assert {c["nombre"] for c in cs if not c["celdas"][gen.MOTOR_NUEVO]["cumple"]} == {
        "climate-model-simulation-crashes", "pc3"}


def test_la_procedencia_dice_que_la_v2_no_es_anclable_y_c3_c4_si():
    pr = _publicado()["procedencia"]
    assert pr["v2"]["anclable_entera"] is False and pr["v2"]["intentos_con_arbol_sucio"] == 110
    assert pr["v2"]["digest_resultados_crudos"].startswith("600ebdaa")
    assert not pr["c3"]["arbol_sucio"] and not pr["c4"]["arbol_sucio"]
    assert pr["c3"]["digest_resultados_crudos"].startswith("640137")
    assert pr["c4"]["digest_resultados_crudos"].startswith("a44a38") and pr["sellos_verificados"]


# ───────────────────────── los dientes: el generador PARA ─────────────────────────

def _con_ficheros(monkeypatch, tmp_path, retoque_c3=None, retoque_c4=None, retoque_recuento=None):
    for nombre, ruta, retoque in (("RUTA_C3", gen.RUTA_C3, retoque_c3),
                                  ("RUTA_C4", gen.RUTA_C4, retoque_c4),
                                  ("RUTA_RECUENTO", gen.RUTA_RECUENTO, retoque_recuento)):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if retoque:
            retoque(datos)
        copia = tmp_path / ruta.name
        copia.write_text(json.dumps(datos), encoding="utf-8")
        monkeypatch.setattr(gen, nombre, copia)
        if nombre == "RUTA_C3":
            monkeypatch.setattr(gen.g_red, "RUTA_C3", copia)
        if nombre == "RUTA_C4":
            monkeypatch.setattr(gen.g_red, "RUTA_C4", copia)
        if nombre == "RUTA_RECUENTO":
            monkeypatch.setattr(gen.g_red, "RUTA_RECUENTO", copia)


def test_control_sin_retoque_el_generador_funciona_sobre_copias(monkeypatch, tmp_path):
    _con_ficheros(monkeypatch, tmp_path)
    assert gen.generar() == gen.RUTA_PUBLICADO.read_text(encoding="utf-8")


def test_un_sello_roto_de_c3_para(monkeypatch, tmp_path):
    def toca(d):
        d["resultados"] = copy.deepcopy(d["resultados"])
        d["resultados"].append({"manipulado": True})
    _con_ficheros(monkeypatch, tmp_path, retoque_c3=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="resultado_pasada_119_c3"):
        gen.componer()


def test_un_sello_roto_de_c4_para(monkeypatch, tmp_path):
    def toca(d):
        d["veredicto_x_de_40"]["x_de_40"] = 39
    _con_ficheros(monkeypatch, tmp_path, retoque_c4=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="resultado_pasada_119_c4"):
        gen.componer()


def test_un_control_distinto_en_los_totales_para(monkeypatch, tmp_path):
    def toca(d):
        d["tabla"]["lightgbm"]["cumplidos"] = 27
    _con_ficheros(monkeypatch, tmp_path, retoque_recuento=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="CONTROL"):
        gen.componer()


def test_un_mejor_en_distinto_en_el_control_para(monkeypatch, tmp_path):
    def toca(d):
        d["tabla"][gen.MOTOR_NUEVO]["mejor_en"] = 22
    _con_ficheros(monkeypatch, tmp_path, retoque_recuento=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="CONTROL"):
        gen.componer()


def test_una_v2_que_no_es_la_de_c4_para(monkeypatch, tmp_path):
    copia = tmp_path / gen.RUTA_V2.name
    copia.write_bytes(gen.RUTA_V2.read_bytes() + b" ")
    monkeypatch.setattr(gen, "RUTA_V2", copia)
    with pytest.raises(gen.DatosQueNoCuadran, match="v2 en disco"):
        gen.componer()
