# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""160-C1 — el ensamblado diverso: la MEZCLA por fila, el analizador y la regla sellada (`benchmarks/c160/`). Nada entrena: todo
con muestras SINTÉTICAS y salidas fabricadas con los 23 nombres reales del congelado.

Fija: la mezcla por fila (binaria, multiclase con columnas en otro orden, regresión) emparejada por IDENTIFICADOR y no por posición;
que se compara contra el MEJOR de los dos miembros y no contra el peor ni contra la media; la reducción relativa del error y el IC;
los saturados (fuera de la media y del recuento, dentro de «ninguno empeora» y del coste); cada condición (a)–(d) con un caso que la
cumple y otro que no; SIN VEREDICTO si falta un conjunto, un pliegue o un volcado; el sello de la regla (en el analizador y en el guion
de medida); que la regla escrita y los umbrales del código dicen lo mismo; y el volcado (el motor que vuelca devuelve lo mismo y solo
guarda la prueba).
"""
from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "c160"))
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import analiza_c160 as an  # noqa: E402
import medir_c160 as med  # noqa: E402
import volcado_c160 as vc  # noqa: E402

L, T = an.LIGHTGBM, an.TABM
CONGELADO = an.conjuntos_y_pliegues_del_congelado()
NOMBRES = list(CONGELADO)


# ---------------------------------------------------------------------------
# 0. VOLCADOS SINTÉTICOS
# ---------------------------------------------------------------------------
def _cls(ids, y, probs, clases, positiva=None, t_proba=0.01, t_predict=0.01):
    probs = np.asarray(probs, dtype=float)
    return {"row_ids": list(ids), "y_true": list(y), "es_clasificacion": True, "clases": tuple(clases), "positiva": positiva,
            "probabilidades": probs, "predicciones": [clases[int(i)] for i in probs.argmax(1)],
            "t_predict_s": t_predict, "t_proba_s": t_proba}


def _reg(ids, y, pred, t_predict=0.01):
    return {"row_ids": list(ids), "y_true": [float(v) for v in y], "es_clasificacion": False, "clases": None, "positiva": None,
            "probabilidades": None, "predicciones": [float(v) for v in pred], "t_predict_s": t_predict, "t_proba_s": None}


# ---------------------------------------------------------------------------
# 1. LA MEZCLA POR FILA
# ---------------------------------------------------------------------------
def test_mezcla_binaria_es_la_media_de_las_dos_columnas_por_fila():
    a = _cls(["r1", "r2"], ["si", "no"], [[0.2, 0.8], [0.9, 0.1]], ("no", "si"), "si")
    b = _cls(["r1", "r2"], ["si", "no"], [[0.6, 0.4], [0.5, 0.5]], ("no", "si"), "si")
    m = an.mezclar(a, b)
    assert np.allclose(m["probabilidades"], [[0.4, 0.6], [0.7, 0.3]])
    assert m["predicciones"] == ["si", "no"]
    assert m["row_ids"] == ["r1", "r2"] and m["clases"] == ("no", "si") and m["positiva"] == "si"


def test_mezcla_multiclase_alinea_las_columnas_por_etiqueta_no_por_posicion():
    a = _cls(["1", "2"], ["a", "c"], [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], ("a", "b", "c"))
    # b declara las clases en OTRO orden: sus columnas son (c, a, b)
    b = _cls(["1", "2"], ["a", "c"], [[0.0, 0.5, 0.5], [0.6, 0.2, 0.2]], ("c", "a", "b"))
    m = an.mezclar(a, b)
    esperado = [[(0.7 + 0.5) / 2, (0.2 + 0.5) / 2, (0.1 + 0.0) / 2], [(0.1 + 0.2) / 2, (0.1 + 0.2) / 2, (0.8 + 0.6) / 2]]
    assert np.allclose(m["probabilidades"], esperado)
    assert m["predicciones"] == ["a", "c"]
    assert np.allclose(np.asarray(m["probabilidades"]).sum(1), 1.0)


def test_mezcla_de_regresion_es_la_media_de_las_predicciones():
    a = _reg(["x", "y", "z"], [1, 2, 3], [1.0, 2.5, 2.0])
    b = _reg(["x", "y", "z"], [1, 2, 3], [3.0, 1.5, 4.0])
    m = an.mezclar(a, b)
    assert m["predicciones"] == [2.0, 2.0, 3.0] and m["probabilidades"] is None


def test_la_mezcla_empareja_por_identificador_no_por_posicion():
    # SABOTAJE 4 («mezclar en el orden de filas equivocado»): b llega con las filas BARAJADAS. Por posición saldría otra cosa.
    a = _reg(["r1", "r2", "r3"], [10, 20, 30], [10, 20, 30])
    b = _reg(["r3", "r1", "r2"], [30, 10, 20], [34, 12, 26])      # r1->12, r2->26, r3->34
    m = an.mezclar(a, b)
    assert m["row_ids"] == ["r1", "r2", "r3"]
    assert m["predicciones"] == [11.0, 23.0, 32.0]
    # y la verdad sigue siendo la de cada fila
    assert m["y_true"] == [10.0, 20.0, 30.0]
    # clasificación barajada
    a = _cls(["1", "2"], ["x", "y"], [[0.9, 0.1], [0.2, 0.8]], ("x", "y"), "y")
    b = _cls(["2", "1"], ["y", "x"], [[0.0, 1.0], [1.0, 0.0]], ("x", "y"), "y")
    assert np.allclose(an.mezclar(a, b)["probabilidades"], [[0.95, 0.05], [0.1, 0.9]])


def test_no_se_mezclan_pruebas_distintas():
    a = _reg(["r1", "r2"], [1, 2], [1, 2])
    with pytest.raises(ValueError, match="mismas filas"):
        an.mezclar(a, _reg(["r1", "r9"], [1, 2], [1, 2]))
    with pytest.raises(ValueError, match="verdad"):
        an.mezclar(a, _reg(["r1", "r2"], [1, 99], [1, 2]))
    with pytest.raises(ValueError, match="repetidos"):
        an.mezclar(_reg(["r1", "r1"], [1, 1], [1, 1]), _reg(["r1", "r1"], [1, 1], [1, 1]))
    with pytest.raises(ValueError, match="clasificación y el otro de regresión"):
        an.mezclar(a, _cls(["r1", "r2"], ["x", "y"], [[1, 0], [0, 1]], ("x", "y"), "y"))


# ---------------------------------------------------------------------------
# 2. LA METRICA (misma cadena que la Fase 0), EL ERROR Y LA REDUCCIÓN RELATIVA
# ---------------------------------------------------------------------------
def test_la_metrica_de_cierre_sale_de_evaluar_y_es_la_conocida():
    ids = [f"r{i}" for i in range(8)]
    y = ["n", "n", "n", "n", "p", "p", "p", "p"]
    probs = [[.9, .1], [.8, .2], [.7, .3], [.4, .6], [.3, .7], [.2, .8], [.1, .9], [.6, .4]]  # un par mal ordenado: AUROC 15/16
    v = _cls(ids, y, probs, ("n", "p"), "p")
    assert an.metrica_de(v, "binary_classification", "auroc") == pytest.approx(15 / 16)
    r = _reg(ids, [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])
    assert an.metrica_de(r, "regression", "r2") == pytest.approx(1.0)
    m = _cls(["a", "b", "c", "d"], ["x", "y", "z", "x"], [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.1, .8, .1]], ("x", "y", "z"))
    assert an.metrica_de(m, "multiclass_classification", "accuracy") == pytest.approx(0.75)


def test_reduccion_relativa_del_error_y_sus_bordes():
    assert an.error_de(0.9) == pytest.approx(0.1)
    assert an.reduccion_relativa([0.10, 0.10], [0.08, 0.08]) == pytest.approx(0.20)
    assert an.reduccion_relativa([0.10, 0.10], [0.12, 0.12]) == pytest.approx(-0.20)
    assert an.reduccion_relativa([0.10, 0.30], [0.05, 0.15]) == pytest.approx(0.50)      # razón de MEDIAS, no media de razones
    assert an.reduccion_relativa([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert an.reduccion_relativa([0.0, 0.0], [0.01, 0.0]) == -1.0
    assert an.ic_de_r([0.1], [0.05]) is None                                              # un pliegue: sin IC
    lo, hi = an.ic_de_r([0.10] * 5, [0.08] * 5)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_el_ic_excluye_cero_solo_si_los_pliegues_lo_dicen():
    firme = an.ic_de_r([0.10, 0.11, 0.09, 0.10, 0.12], [0.07, 0.08, 0.06, 0.07, 0.09])
    ruidoso = an.ic_de_r([0.10, 0.10, 0.10, 0.10, 0.10], [0.05, 0.15, 0.06, 0.14, 0.10])
    assert firme[0] > 0
    assert ruidoso[0] < 0 < ruidoso[1]


def test_se_compara_contra_el_mejor_miembro_no_contra_el_peor_ni_la_media():
    # SABOTAJE 1 («comparar contra el peor miembro»). LightGBM casi perfecto, TabM malo: la mezcla gana al peor y pierde contra el mejor.
    ids = [f"r{i}" for i in range(10)]
    y = np.arange(10.0) * 3 + 1
    a = _reg(ids, y, y + 0.1)            # error pequeño
    b = _reg(ids, y, y + 3.0)            # error grande (sesgo constante)
    volc = {f: {L: a, T: b} for f in range(3)}
    e = an.evaluar_conjunto("ds", "regression", "r2", volc, [0, 1, 2], saturados=())
    assert e["mejor_miembro"] == L and e["completo"]
    ref = an.error_de(e["medias"][L])
    peor = an.error_de(e["medias"][T])
    assert e["error_mejor"] == pytest.approx(ref)
    assert an.reduccion_relativa([peor], [e["error_mezcla"]]) > 0                       # contra el PEOR saldría una «mejora»
    assert e["r"] < 0 and e["empeora_de_verdad"] is True                                 # contra el MEJOR, empeora de verdad
    assert e["error_mezcla"] > e["error_mejor"]
    # y el mejor miembro es el de MAYOR media (no el de menor): con los papeles cambiados cambia
    volc2 = {f: {L: b, T: a} for f in range(3)}
    assert an.evaluar_conjunto("ds", "regression", "r2", volc2, [0, 1, 2], saturados=())["mejor_miembro"] == T


def test_empate_exacto_de_medias_el_mejor_miembro_es_lightgbm():
    assert an.mejor_miembro({L: 0.9, T: 0.9}) == L
    assert an.mejor_miembro({L: 0.9, T: 0.9000001}) == T


def test_dos_miembros_independientes_la_mezcla_mejora_con_ic_claro():
    azar = np.random.RandomState(0)
    ids = [f"r{i}" for i in range(200)]
    y = azar.normal(size=200) * 5
    volc = {}
    for f in range(5):
        r = np.random.RandomState(f)
        volc[f] = {L: _reg(ids, y, y + r.normal(size=200)), T: _reg(ids, y, y + r.normal(size=200))}
    e = an.evaluar_conjunto("ds", "regression", "r2", volc, [0, 1, 2, 3, 4], saturados=())
    assert e["r"] > 0.3 and e["mejora_clara"] and not e["empeora_de_verdad"]


# ---------------------------------------------------------------------------
# 3. LA REGLA: (a)-(d), saturados, costes
# ---------------------------------------------------------------------------
def _ev(nombre, r, *, saturado=False, mejora=None, empeora=None, cl=1.0, ct=1.0, mejor=L):
    """Un conjunto ya evaluado (lo que `evaluar_conjunto` devuelve), para probar la regla sin la aritmética de debajo."""
    mejora = (r > 0.02) if mejora is None else mejora
    empeora = (r < -0.03) if empeora is None else empeora
    c = {L: cl, T: ct}
    return {"nombre": nombre, "saturado": saturado, "completo": True, "r": r, "mejora_clara": mejora, "empeora_de_verdad": empeora,
            "mejor_miembro": mejor, "coste_ms_por_fila": c, "coste_ratio": (cl + ct) / c[mejor]}


def _veintitres(r_no_sat, r_sat=0.0, **kw):
    sat = list(an.SATURADOS)
    otros = [n for n in NOMBRES if n not in sat]
    coste = {k: v for k, v in kw.items() if k in ("cl", "ct", "mejor")}
    return ([_ev(n, r_no_sat, **kw) for n in otros] + [_ev(n, r_sat, saturado=True, **coste) for n in sat])


def test_la_regla_cumple_si_se_dan_las_cuatro():
    g = an.aplicar_la_regla(_veintitres(0.05))
    assert len(NOMBRES) == 23 and g["a"]["n"] == 17
    assert g["a"]["cumple"] and g["b"]["cumple"] and g["c"]["cumple"] and g["d"]["cumple"] and g["cumple"]


def test_a_la_media_se_le_pide_3_por_ciento_exacto_y_solo_de_los_no_saturados():
    # SABOTAJE 2 («contar los saturados en la media»): con los 6 saturados a +50 %, contarlos subiría la media sobre el listón.
    evaluados = _veintitres(0.02, r_sat=0.50)
    contando_todos = sum(e["r"] for e in evaluados) / len(evaluados)
    g = an.aplicar_la_regla(evaluados)
    assert contando_todos > an.A_REDUCCION_MEDIA_MIN                                      # el defecto aprobaría
    assert g["a"]["media_r_no_saturados"] == pytest.approx(0.02) and not g["a"]["cumple"]  # la regla suspende
    assert g["a"]["n"] == 17
    # el borde: exactamente +3 % cumple, una pizca menos no
    assert an.aplicar_la_regla(_veintitres(0.03))["a"]["cumple"] is True
    assert an.aplicar_la_regla(_veintitres(0.0299999))["a"]["cumple"] is False


def test_b_cuenta_mejoras_claras_solo_entre_no_saturados():
    base = _veintitres(0.04, mejora=False)
    for i in range(4):
        base[i]["mejora_clara"] = True
    assert an.aplicar_la_regla(base)["b"]["n"] == 4 and not an.aplicar_la_regla(base)["b"]["cumple"]
    # una mejora clara en un saturado no suma
    for e in base:
        if e["saturado"]:
            e["mejora_clara"] = True
    g = an.aplicar_la_regla(base)
    assert g["b"]["n"] == 4 and not g["b"]["cumple"] and len(g["b"]["informativo_mejoras_claras_en_saturados"]) == 6
    base[4]["mejora_clara"] = True
    assert an.aplicar_la_regla(base)["b"]["n"] == 5 and an.aplicar_la_regla(base)["b"]["cumple"]


def test_c_ninguno_empeora_y_los_saturados_cuentan_aqui():
    ok = _veintitres(0.05)
    assert an.aplicar_la_regla(ok)["c"]["cumple"]
    mal = _veintitres(0.05)
    sat = next(e for e in mal if e["saturado"])
    sat.update(r=-0.05, empeora_de_verdad=True)                                           # un SATURADO que empeora de verdad
    g = an.aplicar_la_regla(mal)
    assert not g["c"]["cumple"] and not g["cumple"] and g["c"]["empeoran"] == [sat["nombre"]]
    assert g["a"]["cumple"] and g["b"]["cumple"]                                          # (a) y (b) no lo ven: por eso (c) lo incluye


def test_empeorar_exige_mas_del_3_por_ciento_y_un_ic_que_excluya_cero():
    # r = -0,10 pero IC que cruza 0: no es «de verdad»; r = -0,02 con IC < 0: tampoco (no pasa del 3 %)
    assert an.clasificar(-0.10, (-0.30, 0.05))["empeora_de_verdad"] is False
    assert an.clasificar(-0.02, (-0.04, -0.001))["empeora_de_verdad"] is False
    assert an.clasificar(-0.04, (-0.07, -0.001))["empeora_de_verdad"] is True
    assert an.clasificar(0.10, (0.001, 0.2))["mejora_clara"] is True
    assert an.clasificar(0.10, (-0.001, 0.2))["mejora_clara"] is False
    assert an.clasificar(0.10, None) == {"mejora_clara": False, "empeora_de_verdad": False}


def test_d_el_coste_es_la_razon_de_sumas_contra_el_mejor_miembro():
    assert an.aplicar_la_regla(_veintitres(0.05, cl=1.0, ct=2.0))["d"]["razon_de_costes"] == pytest.approx(3.0)
    assert an.aplicar_la_regla(_veintitres(0.05, cl=1.0, ct=2.0))["d"]["cumple"]                  # 3,0 exacto cumple
    g = an.aplicar_la_regla(_veintitres(0.05, cl=1.0, ct=2.01))
    assert not g["d"]["cumple"] and not g["cumple"]
    # el mejor miembro es el que se compara: si el mejor es el lento (TabM), la mezcla cuesta poco más que él
    g = an.aplicar_la_regla(_veintitres(0.05, cl=1.0, ct=10.0, mejor=T))
    assert g["d"]["razon_de_costes"] == pytest.approx(1.1) and g["d"]["cumple"]
    # sin coste medido, (d) no se da por buena
    sin = _veintitres(0.05)
    sin[0]["coste_ms_por_fila"] = {L: None, T: 1.0}
    sin[0]["coste_ratio"] = None
    assert not an.aplicar_la_regla(sin)["d"]["cumple"]


def test_un_conjunto_incompleto_hace_la_regla_inaplicable():
    ev = _veintitres(0.05)
    ev[3]["completo"] = False
    g = an.aplicar_la_regla(ev)
    assert not g["aplicable"] and g["incompletos"] == [ev[3]["nombre"]] and not g["cumple"]


# ---------------------------------------------------------------------------
# 4. LA MEDIDA ENTERA: completitud (SIN VEREDICTO) y sellos
# ---------------------------------------------------------------------------
def _volcado_sintetico(nombre, miembro, f, t_ms=1.0, ruido_l=1.0):
    ids = [f"r{i}" for i in range(40)]
    y = np.random.RandomState(7).normal(size=40) * 4
    ruido = np.random.RandomState(zlib.crc32(f"{nombre}|{miembro}|{f}".encode())).normal(size=40)
    return _reg(ids, y, y + ruido * (ruido_l if miembro == L else 1.0), t_predict=t_ms * 40 / 1000.0)


def _payload(*, quitar=None, estado=None, ct=1.0):
    conjuntos, resultados = [], []
    for n, info in CONGELADO.items():
        conjuntos.append({"nombre": n, "tarea": "regression", "metrica_de_cierre": "r2", "sellado": False,
                          "pliegues_esperados": info["pliegues"]})
        for m in (L, T):
            for f in info["pliegues"]:
                if quitar == (n, m, f):
                    continue
                resultados.append({"dataset": n, "motor": m, "repeticion": 0, "pliegue": f,
                                   "estado": estado if quitar is None and estado and (n, m, f) == ("kc2", T, 1) else "completed",
                                   "entorno_digest": "E", "motor_digest": "M" + m, "volcado": {"fichero": f"{n}_{m}_{f}", "sha256": "x"}})
    return {"es_sonda": False, "parcial": False, "conjuntos": conjuntos, "resultados": resultados, "_ct": ct}


def _cargador(ct=1.0, ruido_l=1.0):
    cache = {}

    def cargar(registro):
        k = (registro["dataset"], registro["motor"], registro["pliegue"])
        if k not in cache:
            cache[k] = _volcado_sintetico(k[0], k[1], k[2], t_ms=ct if k[1] == T else 1.0, ruido_l=ruido_l)
        return cache[k]
    return cargar


def test_una_medida_completa_con_mezcla_que_gana_cumple():
    r = an.analizar(_payload(), cargador=_cargador(), comprobar_sellos=False)
    assert r["problemas"] == [] and r["veredicto"] == "CUMPLE", an.informe_en_texto(r)
    assert r["regla"]["a"]["media_r_no_saturados"] > 0.3 and r["regla"]["b"]["n"] >= 5


def test_la_misma_medida_con_una_red_10_veces_mas_lenta_no_cumple_por_el_coste():
    r = an.analizar(_payload(), cargador=_cargador(ct=10.0, ruido_l=0.8), comprobar_sellos=False)
    assert r["veredicto"] == "NO CUMPLE" and r["regla"]["a"]["cumple"] and not r["regla"]["d"]["cumple"]


def test_si_falta_un_pliegue_no_hay_veredicto_ni_se_ignora():
    # SABOTAJE 3 («ignorar un pliegue que falta»): faltar el pliegue 4 de TabM en micro-mass.
    p = _payload(quitar=("micro-mass", T, 4))
    r = an.analizar(p, cargador=_cargador(), comprobar_sellos=False)
    assert r["veredicto"] == "SIN VEREDICTO"
    assert any("faltan" in x for x in r["problemas"])
    micro = next(e for e in r["conjuntos"] if e["nombre"] == "micro-mass")
    assert micro["completo"] is False and micro["pliegues_que_faltan"] == [4]
    assert r["regla"]["aplicable"] is False


def test_si_falta_un_conjunto_o_sobra_otro_no_hay_veredicto():
    p = _payload()
    p["conjuntos"] = [c for c in p["conjuntos"] if c["nombre"] != "wilt"]
    p["resultados"] = [x for x in p["resultados"] if x["dataset"] != "wilt"]
    r = an.analizar(p, cargador=_cargador(), comprobar_sellos=False)
    assert r["veredicto"] == "SIN VEREDICTO" and any("wilt" in x for x in r["problemas"])
    p = _payload()
    p["conjuntos"][0] = dict(p["conjuntos"][0], nombre="un_sellado_cualquiera")
    assert an.analizar(p, cargador=_cargador(), comprobar_sellos=False)["veredicto"] == "SIN VEREDICTO"


def test_un_intento_que_no_cuenta_como_medida_es_un_pliegue_que_falta():
    p = _payload(estado="failed")
    r = an.analizar(p, cargador=_cargador(), comprobar_sellos=False)
    assert r["veredicto"] == "SIN VEREDICTO" and any("no cuenta como medida" in x for x in r["problemas"])


def test_yeast_tiene_cuatro_pliegues_y_los_demas_cinco():
    assert CONGELADO["yeast"]["pliegues"] == [0, 1, 2, 3]
    assert all(v["pliegues"] == [0, 1, 2, 3, 4] for n, v in CONGELADO.items() if n != "yeast")
    assert len(CONGELADO) == 23


def test_parcial_o_registro_repetido_o_de_otra_repeticion_no_dan_veredicto():
    p = _payload()
    p["parcial"] = True
    assert an.analizar(p, cargador=_cargador(), comprobar_sellos=False)["veredicto"] == "SIN VEREDICTO"
    p = _payload()
    p["resultados"].append(dict(p["resultados"][0]))
    assert any("más de una vez" in x for x in an.analizar(p, cargador=_cargador(), comprobar_sellos=False)["problemas"])
    p = _payload()
    p["resultados"][0]["repeticion"] = 1
    assert any("repetición 1" in x for x in an.analizar(p, cargador=_cargador(), comprobar_sellos=False)["problemas"])


def test_codigo_distinto_en_un_mismo_motor_no_es_una_medida():
    p = _payload()
    p["resultados"][0]["motor_digest"] = "OTRO"
    r = an.analizar(p, cargador=_cargador(), comprobar_sellos=False)
    assert r["veredicto"] == "SIN VEREDICTO" and any("CÓDIGO DISTINTO" in x for x in r["problemas"])


def test_una_sonda_no_decide_nada():
    p = _payload()
    p["es_sonda"] = True
    p["conjuntos"], p["resultados"] = p["conjuntos"][:1], [x for x in p["resultados"] if x["dataset"] == p["conjuntos"][0]["nombre"]]
    assert an.analizar(p, cargador=_cargador(), comprobar_sellos=False)["veredicto"] == "SONDA"


def test_el_sello_de_la_regla_se_comprueba_en_el_analizador(monkeypatch):
    sha = an._sha256(an.RUTA_DE_LA_REGLA)
    p = _payload()
    p["regla_sha256_al_medir"] = sha
    p["analizador_sha256_al_medir"] = an._sha256(an.RUTA_DEL_ANALIZADOR)
    p["dependencias_sha256_al_medir"] = an.dependencias_de_ahora()
    # sin sellar: ninguna medida vale
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    r = an.analizar(p, cargador=_cargador())
    assert r["veredicto"] == "SIN VEREDICTO" and any("NO está sellada" in x for x in r["problemas"])
    # sellada con otra: la regla de ahora no es la sellada
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    assert any("no es la sellada" in x for x in an.analizar(p, cargador=_cargador())["problemas"])
    # sellada con la de ahora y la salida con los mismos sellos: veredicto
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", sha)
    r = an.analizar(p, cargador=_cargador())
    assert r["problemas"] == [] and r["veredicto"] == "CUMPLE"
    # la salida midió con OTRA regla / otro analizador / otras dependencias
    for campo, valor in (("regla_sha256_al_medir", "1" * 64), ("analizador_sha256_al_medir", "2" * 64)):
        q = dict(p, **{campo: valor})
        rr = an.analizar(q, cargador=_cargador())
        assert rr["veredicto"] == "SIN VEREDICTO" and any(campo in x for x in rr["problemas"])
    q = dict(p, dependencias_sha256_al_medir={**an.dependencias_de_ahora(), "matrixai/estudio/metricas.py": "3" * 64})
    assert any("metricas.py" in x for x in an.analizar(q, cargador=_cargador())["problemas"])
    q = {k: v for k, v in p.items() if k != "regla_sha256_al_medir"}
    assert any("no declara `regla_sha256_al_medir`" in x for x in an.analizar(q, cargador=_cargador())["problemas"])


def test_el_guion_de_medida_se_niega_sin_sello_o_con_la_regla_cambiada(monkeypatch, tmp_path):
    salida = tmp_path / "r.json"
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    with pytest.raises(SystemExit, match="NO está sellada"):
        med.main(["--salida", str(salida)])
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    with pytest.raises(SystemExit, match="no es la sellada"):
        med.main(["--salida", str(salida)])
    assert not salida.exists()


def test_el_sello_grabado_es_el_de_la_regla_escrita():
    """El supervisor sella: hasta entonces la constante es None. Si está puesta, ES el sha256 del fichero (si no, la medida no arranca)."""
    if an.REGLA_SHA256_SELLADA is not None:
        assert an.REGLA_SHA256_SELLADA == an._sha256(an.RUTA_DE_LA_REGLA)


# ---------------------------------------------------------------------------
# 5. LA REGLA ESCRITA Y EL CÓDIGO DICEN LO MISMO
# ---------------------------------------------------------------------------
def _texto_de_la_regla():
    return an.RUTA_DE_LA_REGLA.read_text(encoding="utf-8")


def test_la_lista_de_saturados_de_la_regla_es_la_del_codigo():
    m = re.search(r"^\s*SATURADOS:\s*(.+)$", _texto_de_la_regla(), re.M)
    assert m, "la regla tiene que declarar `SATURADOS: a, b, ...` en una línea"
    assert sorted(x.strip() for x in m.group(1).split(",")) == sorted(an.SATURADOS)
    assert set(an.SATURADOS) <= set(NOMBRES) and len(an.SATURADOS) == 6


def test_los_saturados_son_los_que_da_el_criterio_aplicado_a_los_artefactos_congelados():
    assert sorted(an.saturados_segun_el_criterio()) == sorted(an.SATURADOS)
    # y el criterio no es de adorno: con un listón de 99 % quedan menos
    assert set(an.saturados_segun_el_criterio(0.99)) < set(an.SATURADOS)


def test_los_umbrales_del_codigo_son_los_que_dice_la_regla():
    t = _texto_de_la_regla()
    assert an.A_REDUCCION_MEDIA_MIN == 0.03 and "≥ **+3 %**" in t
    assert an.B_MEJORAS_CLARAS_MIN == 5 and "≥ **5** de los 17" in t
    assert an.C_EMPEORA_MAS_DE == 0.03 and "R < −3 %" in t
    assert an.D_COSTE_MAXIMO == 3.0 and "≤ **3 ×**" in t
    assert an.N_CONJUNTOS == 23 and an.SEMILLA_DEL_IC == 0 and an.REMUESTRAS_DEL_IC == 1000 and an.NIVEL_DEL_IC == 0.95
    assert an.UMBRAL_DE_SATURACION == 0.98 and "≥ 98 %" in t
    assert "17 no saturados" in t and len(NOMBRES) - len(an.SATURADOS) == 17


def test_los_conjuntos_del_guion_son_los_23_del_congelado_y_no_hay_sellados():
    assert med.conjuntos_del_congelado() == NOMBRES
    assert len(NOMBRES) == 23


def test_las_dependencias_del_guion_y_del_analizador_llevan_las_mismas_etiquetas():
    assert med.dependencias_sha256().keys() == an.dependencias_de_ahora().keys()
    assert set(an.dependencias_de_ahora()) >= {"benchmarks/fase0/protocolo.py", "matrixai/estudio/metricas.py"}


def test_los_miembros_son_los_dos_motores_de_la_fase_0():
    m = med.motores()
    assert list(m) == [L, T] == list(an.MIEMBROS)
    assert m[L].nombre == L and m[T].nombre == T
    assert type(m[L]).__mro__[2].__name__ == "MotorArbolLightGBM"
    assert type(m[T]).__mro__[2].__name__ == "MotorDensaTabM"


# ---------------------------------------------------------------------------
# 6. EL VOLCADO
# ---------------------------------------------------------------------------
def test_volcado_ida_y_vuelta_sin_pickle(tmp_path):
    ruta = tmp_path / "a" / "v.npz"
    vc.guardar_volcado(ruta, row_ids=["a", "b"], y_true=["x", "y"], predicciones=["x", "y"], probabilidades=[[.7, .3], [.2, .8]],
                       clases=("x", "y"), positiva="y", t_predict_s=0.5, t_proba_s=0.25)
    v = vc.leer_volcado(ruta)
    assert v["row_ids"] == ["a", "b"] and v["y_true"] == ["x", "y"] and v["clases"] == ("x", "y") and v["positiva"] == "y"
    assert np.allclose(v["probabilidades"], [[.7, .3], [.2, .8]]) and v["t_proba_s"] == 0.25 and v["es_clasificacion"]
    assert not list(ruta.parent.glob("*.tmp*"))
    vc.guardar_volcado(tmp_path / "r.npz", row_ids=["a"], y_true=[1.5], predicciones=[1.25], probabilidades=None, clases=None,
                       positiva=None, t_predict_s=0.1, t_proba_s=None)
    r = vc.leer_volcado(tmp_path / "r.npz")
    assert r["predicciones"] == [1.25] and r["probabilidades"] is None and r["t_proba_s"] is None and not r["es_clasificacion"]


class _Particion:
    def __init__(self, ids, target):
        self.row_ids, self.target = tuple(ids), tuple(target)


class _Fitted:
    def __init__(self, clases=None, positiva=None):
        self.classes, self.positive_label = clases, positiva


class _MotorFalso:
    def predict(self, fitted, datos):
        return tuple("y" if i % 2 else "x" for i, _ in enumerate(datos.row_ids)) if fitted.classes else tuple(float(i) for i, _ in enumerate(datos.row_ids))

    def predict_proba(self, fitted, datos):
        return tuple((0.3, 0.7) for _ in datos.row_ids)


class _Volcando(vc._Volcador, _MotorFalso):
    pass


def test_el_motor_que_vuelca_devuelve_lo_mismo_y_solo_guarda_la_prueba(tmp_path):
    test, val = _Particion(["t1", "t2"], ["x", "y"]), _Particion(["v1", "v2"], ["x", "y"])
    m = _Volcando()
    ruta = tmp_path / "t.npz"
    vc.con_volcado(m, ruta, test.row_ids)
    f = _Fitted(("x", "y"), "y")
    assert m.predict(f, val) == _MotorFalso().predict(f, val) and m.predict_proba(f, val) == _MotorFalso().predict_proba(f, val)
    assert not ruta.exists()                                  # la validación no se vuelca
    assert m.predict(f, test) == _MotorFalso().predict(f, test)
    assert not ruta.exists()                                  # en clasificación se guarda al pedir las probabilidades
    assert m.predict_proba(f, test) == _MotorFalso().predict_proba(f, test)
    v = vc.leer_volcado(ruta)
    assert v["row_ids"] == ["t1", "t2"] and v["y_true"] == ["x", "y"] and v["predicciones"] == ["x", "y"]
    assert np.allclose(v["probabilidades"], [[.3, .7], [.3, .7]]) and v["t_predict_s"] >= 0 and v["t_proba_s"] >= 0


def test_el_motor_que_vuelca_en_regresion_guarda_al_predecir(tmp_path):
    test = _Particion(["t1", "t2", "t3"], [1.0, 2.0, 3.0])
    m = _Volcando()
    ruta = tmp_path / "t.npz"
    vc.con_volcado(m, ruta, test.row_ids)
    assert m.predict(_Fitted(), test) == (0.0, 1.0, 2.0)
    v = vc.leer_volcado(ruta)
    assert v["predicciones"] == [0.0, 1.0, 2.0] and v["y_true"] == [1.0, 2.0, 3.0] and not v["es_clasificacion"]


def test_las_probabilidades_antes_que_las_predicciones_paran_en_vez_de_guardar_a_medias(tmp_path):
    test = _Particion(["t1"], ["x"])
    m = _Volcando()
    vc.con_volcado(m, tmp_path / "t.npz", test.row_ids)
    with pytest.raises(RuntimeError, match="antes que las predicciones"):
        m.predict_proba(_Fitted(("x", "y"), "y"), test)


def test_el_coste_por_fila_sale_de_la_llamada_que_mide_el_arnes():
    c = _cls(["a", "b", "c", "d"], ["x"] * 4, [[1, 0]] * 4, ("x", "y"), "y", t_proba=0.04, t_predict=9.0)
    assert an.coste_por_fila_ms(c) == pytest.approx(10.0)
    r = _reg(["a", "b"], [1, 2], [1, 2], t_predict=0.002)
    assert an.coste_por_fila_ms(r) == pytest.approx(1.0)
    r["t_predict_s"] = None
    assert an.coste_por_fila_ms(r) is None
