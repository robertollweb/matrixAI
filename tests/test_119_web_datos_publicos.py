# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-WEB — los datos públicos de la red nueva salen de los registros sellados, nunca de una mano.

`benchmarks/datos_publicos/red_119_publico.json` es lo que lee la sección nueva de «Cómo medimos».
Si alguien retoca una cifra, o un registro cambia y nadie lo regenera, esto falla. Y el generador
PARA si un sello no cuadra (se prueba sobre una copia rota, no solo sobre lo bueno).
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "datos_publicos_generar_red_119", RAIZ / "benchmarks" / "datos_publicos" / "generar_red_119.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _publicado():
    return json.loads(gen.RUTA_PUBLICADO.read_text(encoding="utf-8"))


def test_lo_publicado_es_BYTE_A_BYTE_lo_que_sale_de_los_registros():
    assert gen.RUTA_PUBLICADO.read_text(encoding="utf-8") == gen.generar(), (
        "red_119_publico.json no es lo que sale de los registros: "
        "python3 benchmarks/datos_publicos/generar_red_119.py --escribir, y mirar por qué cambió")


def test_las_cifras_son_las_de_los_registros_sellados():
    p = _publicado()
    c4 = json.loads(gen.RUTA_C4.read_text(encoding="utf-8"))
    v = c4["veredicto_x_de_40"]
    assert p["titular"]["x_de_40"] == v["x_de_40"] == 38
    assert p["titular"]["n_conjuntos"] == 40
    assert p["titular"]["red_de_antes_x_de_40"] == v["densa_v2_en_los_mismos_conjuntos"]["cumplidos"] == 16
    assert (p["reparto"]["no_sellados"]["cumplidos"], p["reparto"]["sellados"]["cumplidos"]) == (31, 7)
    tabla = p["recuento_con_el_motor_nuevo_en_el_campo"]["tabla"]
    assert tabla[gen.MOTOR_NUEVO]["cumplidos"] == p["titular"]["x_de_40"]
    assert {m: t["cumplidos"] for m, t in tabla.items()} == {
        "baseline": 0, "catboost": 25, "lightgbm": 26, gen.MOTOR_NUEVO: 38,
        "sklearn.hgb": 25, "sklearn.lineal": 13, "xgboost": 24}


def test_la_cuenta_sin_la_red_nueva_es_la_de_la_v2_congelada():
    sin = _publicado()["recuento_con_el_motor_nuevo_en_el_campo"]["sin_la_red_nueva"]
    assert sin["digest_resultados_crudos"].startswith("600ebdaa")
    assert {m: x["cumplidos"] for m, x in sin["por_motor"].items()} == {
        "catboost": 29, "lightgbm": 33, "matrixai.dense.torch_cpu": 16,
        "sklearn.hgb": 33, "sklearn.lineal": 14, "xgboost": 29}


def test_el_detalle_por_conjunto_suma_el_titular_y_la_red_de_antes():
    p = _publicado()
    filas = p["detalle_por_conjunto"]
    assert len(filas) == 40 and len({f["conjunto"] for f in filas}) == 40
    assert sum(f["cumple"] for f in filas) == p["titular"]["x_de_40"]
    assert sum(f["red_de_antes"]["cumple"] for f in filas) == p["titular"]["red_de_antes_x_de_40"]
    assert sum(f["bloque"] == "sellado" for f in filas) == 8
    # Un conjunto que NO cumple lo dice el detalle, no se esconde.
    assert {f["conjunto"] for f in filas if not f["cumple"]} == {
        "climate-model-simulation-crashes", "pc3"}


def test_lo_declarado_sale_con_sus_numeros():
    d = _publicado()["declarado"]
    assert d["allstate"]["epocas_ejecutadas_min"] == d["allstate"]["epocas_ejecutadas_max"] == 5
    assert d["allstate"]["minimo_de_la_regla"] == 17 and d["allstate"]["todos_parados_por_el_plazo"]
    assert d["parada_temprana"]["metrica"] == ["validation_loss_del_ensamblado"]
    assert d["otras_redes"]["se_remidieron"] is False
    assert d["otras_redes"]["creado"].startswith("2026-09-22")
    assert d["sellados"]["medidos_una_sola_vez"] and d["sellados"]["n_intentos_reintentados"] == 0


def test_la_procedencia_cita_los_sellos_y_no_es_de_un_arbol_sucio():
    pr = _publicado()["procedencia"]
    assert pr["sellos_verificados"] is True
    assert pr["c3"]["digest_resultados_crudos"].startswith("640137")
    assert pr["c4"]["digest_resultados_crudos"].startswith("a44a38")
    assert not pr["c3"]["arbol_sucio"] and not pr["c4"]["arbol_sucio"]


# ───────────────────────── los dientes: el generador PARA ─────────────────────────

def _con_ficheros(monkeypatch, tmp_path, retoque_c3=None, retoque_c4=None, retoque_recuento=None,
                  retoque_c5a=None):
    for nombre, ruta, retoque in (("RUTA_C3", gen.RUTA_C3, retoque_c3),
                                  ("RUTA_C4", gen.RUTA_C4, retoque_c4),
                                  ("RUTA_RECUENTO", gen.RUTA_RECUENTO, retoque_recuento),
                                  ("RUTA_C5A", gen.RUTA_C5A, retoque_c5a)):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if retoque:
            retoque(datos)
        copia = tmp_path / ruta.name
        copia.write_text(json.dumps(datos), encoding="utf-8")
        monkeypatch.setattr(gen, nombre, copia)


def test_control_sin_retoque_el_generador_funciona_sobre_copias(monkeypatch, tmp_path):
    _con_ficheros(monkeypatch, tmp_path)
    assert gen.generar() == gen.RUTA_PUBLICADO.read_text(encoding="utf-8")


def test_un_sello_roto_de_c3_para(monkeypatch, tmp_path):
    def toca(d):
        d["veredicto"]["cumplidos_con_el_motor_nuevo"]["cumplidos"] = 32  # sin resellar
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


def test_un_recuento_que_cita_otro_sello_para(monkeypatch, tmp_path):
    def toca(d):
        d["fuentes"]["c4"] = "0" * 64
    _con_ficheros(monkeypatch, tmp_path, retoque_recuento=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="recuento cita"):
        gen.componer()


def test_un_control_del_recuento_que_no_da_el_x_de_c4_para(monkeypatch, tmp_path):
    def toca(d):
        d["tabla"][gen.MOTOR_NUEVO]["cumplidos"] = 37
    _con_ficheros(monkeypatch, tmp_path, retoque_recuento=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="control"):
        gen.componer()


# ───────────────────── «¿Y en el Studio?»: el bloque `en_el_studio` (119-C5a) ─────────────────────

def _c5a():
    return json.loads(gen.RUTA_C5A.read_text(encoding="utf-8"))


def _resellar(d):
    """Resella los dos sellos tras un retoque: lo que cuenta es lo que el generador exige DESPUÉS del sello."""
    d["digest_solo_de_resultados"] = gen.digest_canonico(d["resultados"])
    sin = {k: v for k, v in d.items() if k != "digest_resultados_crudos"}
    d["digest_resultados_crudos"] = gen.digest_canonico(sin)


def test_en_el_studio_sale_de_c5a_y_dice_que_no_sustituye():
    e = _publicado()["en_el_studio"]
    c5a = _c5a()
    v = c5a["veredicto"]
    assert e["decision"]["decision"] == "no_sustituye"
    assert "D5 re-decidida por Roberto el 2026-10-01, opción (a)" in e["decision"]["fuente"]
    assert e["cumple_las_cuatro"] is False and v["cumple_las_cuatro"] is False
    assert f'{e["completa"]["cumplidos"]}/{e["completa"]["de"]}' == v["completa"]["numero"] == "18/26"
    assert f'{e["compite"]["cumplidos_de_la_cartera"]}/{e["compite"]["de"]}' == v["compite"]["cumplidos_de_la_cartera"]
    assert e["compite"]["perdidos_contra_la_red_anterior"] == v["compite"]["conjuntos_perdidos_contra_la_densa_de_hoy"]
    assert e["compite"]["tope_de_perdidos"] == v["compite"]["tope_de_perdidos"]
    assert e["compite"]["puntos"] == v["compite"]["puntos"]
    assert e["n_conjuntos"] == v["n_conjuntos"] == 13
    assert e["condiciones"]["hilos_por_intento"] == c5a["condiciones_del_studio"]["hilos"] == 1
    assert e["condiciones"]["segundos_por_intento"] == c5a["condiciones_del_studio"]["segundos_por_intento"][gen.MOTOR_NUEVO]
    assert e["fase_0"]["hilos_por_intento"] == 4
    assert (e["fase_0"]["segundos_por_intento_min"], e["fase_0"]["segundos_por_intento_max"]) == (120.0, 600.0)
    assert e["medido"] == c5a["procedencia"]["medido"] and e["fuente"]["anclable"] is True
    assert e["fuente"]["digest_resultados_crudos"] == c5a["digest_resultados_crudos"]


def test_en_el_studio_control_sin_retoque(monkeypatch, tmp_path):
    _con_ficheros(monkeypatch, tmp_path, retoque_c5a=lambda d: None)
    assert gen.generar() == gen.RUTA_PUBLICADO.read_text(encoding="utf-8")


def test_una_cifra_de_c5a_cambiada_sin_resellar_para(monkeypatch, tmp_path):
    def toca(d):
        d["veredicto"]["compite"]["conjuntos_perdidos_contra_la_densa_de_hoy"] = 1
    _con_ficheros(monkeypatch, tmp_path, retoque_c5a=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="resultado_sonda_119_c5a"):
        gen.componer()


def test_un_veredicto_de_c5a_que_pasa_para_aunque_este_resellado(monkeypatch, tmp_path):
    def toca(d):
        v = d["veredicto"]
        v["cumple_las_cuatro"] = True
        for k in ("completa", "aprende", "compite", "cabe"):
            v[k]["cumple"] = True
        _resellar(d)
    _con_ficheros(monkeypatch, tmp_path, retoque_c5a=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="veredicto de C5a"):
        gen.componer()


@pytest.mark.parametrize("campo,valor,mensaje", [
    ("parcial", True, "pasada real completa"),
    ("es_humo", True, "pasada real completa"),
    ("es_subconjunto_de_prueba", True, "pasada real completa"),
    ("tipo_de_ejecucion", "humo", "pasada real completa"),
])
def test_una_c5a_parcial_o_de_humo_para(monkeypatch, tmp_path, campo, valor, mensaje):
    def toca(d):
        d[campo] = valor
        _resellar(d)
    _con_ficheros(monkeypatch, tmp_path, retoque_c5a=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match=mensaje):
        gen.componer()


def test_una_c5a_no_anclable_para(monkeypatch, tmp_path):
    def toca(d):
        d["procedencia"]["anclable"] = False
        _resellar(d)
    _con_ficheros(monkeypatch, tmp_path, retoque_c5a=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="anclable"):
        gen.componer()


def test_una_c5a_con_la_regla_distinta_de_la_registrada_para(monkeypatch, tmp_path):
    def toca(d):
        d["regla"]["coincide"] = False
        _resellar(d)
    _con_ficheros(monkeypatch, tmp_path, retoque_c5a=toca)
    with pytest.raises(gen.DatosQueNoCuadran, match="regla"):
        gen.componer()
