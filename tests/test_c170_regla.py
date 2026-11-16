# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""170-C1 — el ANALIZADOR de `benchmarks/c170/analiza_c170.py` contra `REGLA.md`: una prueba por condición, con su nombre.

Todo con salidas SINTÉTICAS (ninguna prueba mide ni descarga). Dos referencias: la REAL (la v2 y TabM commiteadas: el régimen son 24, los rivales están completos, las particiones coinciden) y una
SINTÉTICA de 24 conjuntos de 5 pliegues donde se controla cada valor (los bordes del 80 % y de los 2 puntos, el mejor de tres, los fallidos). El sello de la regla se simula con monkeypatch de
`REGLA_SHA256_SELLADA` y el del CÓDIGO con un `SELLO_DEL_CODIGO.json` temporal (los reales los pone el supervisor).
"""
from __future__ import annotations

import copy
import json
import math
import re
import statistics
import sys
from pathlib import Path

import pytest

_C170 = Path(__file__).resolve().parents[1] / "benchmarks" / "c170"
if str(_C170) not in sys.path:
    sys.path.insert(0, str(_C170))

import analiza_c170 as an  # noqa: E402

REGLA = an.RUTA_DE_LA_REGLA.read_text(encoding="utf-8")
ENTORNO, MOTOR_DIGEST = "e170", "m170"
PLIEGUES = 5


# ---------------------------------------------------------------------------------------------------------------------------
# la referencia SINTÉTICA: n conjuntos, 1 repetición x 5 pliegues, 3 rivales con valores controlados
# ---------------------------------------------------------------------------------------------------------------------------
def ref_sintetica(n: int = 24, valores: dict | None = None, n_predictores: int = 10, falla: dict | None = None):
    """`valores[(i, rival)]` = valor constante (o lista de 5) del rival en el conjunto i; por omisión lightgbm 0,80, hgb 0,79, TabM 0,78. `falla[(i, rival)]` = pliegue que ese rival NO completó."""
    valores, falla = valores or {}, falla or {}
    nombres = [f"s{i:02d}" for i in range(n)]
    v2_res, tabm_res = [], []
    for i, nom in enumerate(nombres):
        for rival, base in zip(an.RIVALES, (0.80, 0.79, 0.78)):
            v = valores.get((i, rival), base)
            v = list(v) if isinstance(v, (list, tuple)) else [v] * PLIEGUES
            for f in range(PLIEGUES):
                reg = {"dataset": nom, "motor": rival, "repeticion": 0, "pliegue": f, "estado": "completed", "auroc": v[f], "semilla": 0,
                       "presupuesto_wall_s": 120.0, "tiempo_de_ajuste": 1.0, "metrica_de_cierre": "auroc"}
                if falla.get((i, rival)) == f:
                    reg["estado"] = "failed"
                (tabm_res if rival == an.RIVALES[2] else v2_res).append(reg)
        # el motor `lightgbm` es quien define las claves esperadas: siempre tiene los 5 registros (aunque uno haya fallado)
    v2 = {"datasets_declarados": [{"nombre": nom, "cubo": "pequeno", "tarea": "binary_classification", "sellado": False, "data_id": i} for i, nom in enumerate(nombres)],
          "particion_por_dataset": {nom: {"n_predictores": n_predictores, "n_test": 100, "plan_digest": f"plan-{nom}"} for nom in nombres},
          "procedencia": {"datos_de_entrada": {nom: {"sha256": f"sha-{nom}"} for nom in nombres}}, "resultados": v2_res}
    tabm = {"resultados": tabm_res, "particion_comparada_con_la_v2": {nom: {"plan_digest": f"plan-{nom}", "identica_a_la_v2": True} for nom in nombres}, "particion_por_dataset": {}}
    return an.Referencia(v2, tabm)


def _registro(ref, nombre, k, valor, *, coste=10.0):
    m = ref.meta[nombre]
    v2 = m["por_clave"][k]
    return {"dataset": nombre, "data_id": m["data_id"], "motor": an.MOTOR, "repeticion": k[0], "pliegue": k[1], "estado": "completed", "metrica_de_cierre": m["metrica"], m["metrica"]: valor,
            "semilla": v2["semilla"], "presupuesto_wall_s": v2["presupuesto_wall_s"], "hilos": an.HILOS_POR_INTENTO, "split_plan_digest": m["plan_digest"], "datos_sha256": m["datos_sha256"],
            "n_test": m["n_test"], "entorno_digest": ENTORNO, "motor_digest": MOTOR_DIGEST, "engine_version": an.VERSION_DEL_MOTOR,
            "hiperparametros": {"interpret_core": an.VERSION_DE_INTERPRET_CORE}, "tiempo_de_ajuste": coste, "wall_s": coste + 2.0, "tarea": m["tarea"], "cubo": m["cubo"], "sellado": False}


def construir(ref, delta: dict | float = 0.0, *, coste=10.0, conjuntos=None) -> dict:
    """Una salida completa y válida: el EBM = el mejor rival + `delta` (un número, o `{nombre: número}`), en todos los pliegues."""
    conjuntos_decl, resultados = [], []
    for nom in (conjuntos if conjuntos is not None else ref.regimen()):
        m = ref.meta[nom]
        mejor = an.mejor_rival(ref, nom)["rival"]
        rival = ref.valores_completos(nom, mejor) if mejor else {k: 0.5 for k in m["claves"]}
        d = delta.get(nom, 0.0) if isinstance(delta, dict) else delta
        for k in sorted(m["claves"]):
            resultados.append(_registro(ref, nom, k, rival[k] + d, coste=coste))
        conjuntos_decl.append({"nombre": nom, "data_id": m["data_id"], "cubo": m["cubo"], "tarea": m["tarea"], "sellado": False, "metrica_de_cierre": m["metrica"],
                               "pliegues_esperados": [list(k) for k in sorted(m["claves"])], "n_test": m["n_test"], "datos_sha256": m["datos_sha256"], "plan_digest": m["plan_digest"],
                               "n_predictores": m["n_predictores"]})
    s = an.sellos_de_ahora()
    return {"es_sonda": False, "parcial": False, "conjuntos": conjuntos_decl, "resultados": resultados, "digest_de_la_cache": {"entorno": ENTORNO, "motor": MOTOR_DIGEST},
            "regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "regla_sha256_al_medir": s["regla_sha256"], "analizador_sha256_al_medir": s["analizador_sha256"],
            "dependencias_sha256_al_medir": s["dependencias_sha256"], "sello_del_codigo_al_medir": s["sello_del_codigo"]}


def escribir_sello_del_codigo(ruta: Path, sha: dict | None = None, regla: str | None = "auto") -> Path:
    sello = {"regla_sha256_sellada": an.REGLA_SHA256_SELLADA if regla == "auto" else regla, "sha256": dict(an.codigo_de_ahora()) if sha is None else sha}
    ruta.write_text(json.dumps(sello), encoding="utf-8")
    return ruta


@pytest.fixture(autouse=True)
def sellada(monkeypatch, tmp_path):
    """La regla «sellada» = el REGLA.md de ahora y el código «sellado» = el de ahora (un SELLO_DEL_CODIGO.json temporal). Las pruebas de la negativa los vuelven a romper."""
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", an.sellos_de_ahora()["regla_sha256"])
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", escribir_sello_del_codigo(tmp_path / "SELLO_DEL_CODIGO.json"))


def _sellar(payload):
    s = an.sellos_de_ahora()
    payload.update(regla_sha256_sellada=an.REGLA_SHA256_SELLADA, regla_sha256_al_medir=s["regla_sha256"], analizador_sha256_al_medir=s["analizador_sha256"],
                   dependencias_sha256_al_medir=s["dependencias_sha256"], sello_del_codigo_al_medir=s["sello_del_codigo"])
    return payload


def veredicto(payload, ref):
    return an.analizar(_sellar(payload), referencia=ref)


def con_cumplen(ref, cumplen: set[int], *, malo=-0.05, bueno=0.0):
    """EBM = rival + `bueno` en los conjuntos `cumplen` (por índice) y rival + `malo` en el resto."""
    return construir(ref, {nom: (bueno if i in cumplen else malo) for i, nom in enumerate(ref.regimen())})


# ---------------------------------------------------------------------------------------------------------------------------
# 0. LOS NÚMEROS DE LA REGLA, fijados contra el TEXTO de REGLA.md (no contra el código)
# ---------------------------------------------------------------------------------------------------------------------------
def test_los_umbrales_son_los_de_REGLA_md():
    assert an.DISTANCIA_MINIMA == -0.02 and "`d ≥ −0,02`" in REGLA and "`d >= -0.02`" in REGLA
    from fractions import Fraction
    assert an.FRACCION_MINIMA == Fraction(4, 5) and "`C / M ≥ 0,8`" in REGLA and "`5·C ≥ 4·M`" in REGLA
    assert an.MEDIBLES_MINIMOS == 20 and "**M ≥ 20**" in REGLA
    assert an.COLUMNAS_MAXIMAS == 100 and "**≤ 100 predictores**" in REGLA
    assert an.HILOS_POR_INTENTO == 4 and "4 hilos" in REGLA
    assert an.VERSION_DE_INTERPRET_CORE == "0.7.8" and "`interpret-core==0.7.8`" in REGLA
    assert an.RIVALES == ("lightgbm", "sklearn.hgb", "matrixai.dense.tabm_cpu")
    assert "`lightgbm` y `sklearn.hgb`" in REGLA and "`matrixai.dense.tabm_cpu` (TabM)" in REGLA


def test_la_regla_dice_los_dos_puntos_en_la_escala_cruda_y_exacta_sin_redondear():
    assert "**cruda**" in REGLA and "**exacta y sin redondear**" in REGLA and "**Puntos = centésimas de la métrica, no relativos.**" in REGLA


def test_la_regla_dice_que_decide_la_media_y_no_el_intervalo_y_el_IC_solo_se_informa():
    assert "**Se exige sobre la MEDIA, no sobre un intervalo.**" in REGLA
    assert "se informa para cada conjunto y NO decide" in REGLA


def test_el_regimen_son_los_24_de_la_regla_con_sus_nombres_y_los_8_de_fuera_con_sus_columnas():
    ref = an.cargar_referencia()
    nombres = re.search(r"Son \*\*24\*\*:\s*(.+?)\.\s+Quedan", REGLA, re.S).group(1)
    en_regla = sorted(re.findall(r"`([^`]+)`", nombres))
    assert en_regla == ref.regimen() and len(en_regla) == an.N_DEL_REGIMEN == 24
    fuera = dict(re.findall(r"`([^`]+)` ([0-9.]+)", REGLA.split("Quedan **fuera del régimen**")[1].split("**Los 8 sellados")[0]))
    assert {k: int(v.replace(".", "")) for k, v in fuera.items()} == ref.fuera_del_regimen() and len(fuera) == 8
    sellados = [d["nombre"] for d in ref.v2["datasets_declarados"] if d["sellado"]]
    assert len(sellados) == 8 and not set(sellados) & set(ref.regimen())


def test_la_referencia_real_tiene_los_tres_rivales_completos_en_los_24_y_la_misma_particion():
    ref = an.cargar_referencia()
    assert ref.problemas() == []
    for n in ref.regimen():
        mr = an.mejor_rival(ref, n)
        assert mr["excluidos"] == {} and set(mr["medias"]) == set(an.RIVALES), n


def test_los_topes_de_la_regla_por_cubo_son_los_del_protocolo_v2_de_la_referencia():
    ref = an.cargar_referencia()
    topes = {m["por_clave"][k]["presupuesto_wall_s"] for m in ref.meta.values() for k in m["claves"]}
    assert topes == {120.0, 300.0, 600.0}
    assert "120 s pequeño, 300 s mediano, 600 s grande" in REGLA


# ---------------------------------------------------------------------------------------------------------------------------
# 1. LA DISTANCIA Y EL BORDE DE LOS 2 PUNTOS
# ---------------------------------------------------------------------------------------------------------------------------
def _uno(delta, *, valor=0.80, ref=None):
    ref = ref or ref_sintetica(24)
    p = construir(ref, {ref.regimen()[0]: delta})
    ev = an.evaluar_conjunto(ref, ref.regimen()[0], {(r["repeticion"], r["pliegue"]): r for r in p["resultados"] if r["dataset"] == ref.regimen()[0]})
    return ev


def test_un_EBM_a_menos_de_dos_puntos_cumple_y_a_mas_de_dos_puntos_no():
    assert _uno(-0.019)["cumple"] is True
    assert _uno(-0.0201)["cumple"] is False
    assert _uno(-0.05)["cumple"] is False


def test_un_EBM_que_SUPERA_al_mejor_cumple():
    ev = _uno(+0.05)
    assert ev["cumple"] is True and ev["distancia"] > 0.0


def test_la_distancia_se_mide_en_puntos_de_la_metrica_cruda_no_relativa():
    """Con el mejor rival en 0,50, perder 0,015 son 1,5 puntos (cumple) aunque sea un 3 % relativo; con el rival en 0,95, perder 0,025 son 2,5 puntos (no cumple) aunque sea un 2,6 % relativo."""
    ref = ref_sintetica(24, {(0, "lightgbm"): 0.50})
    assert _uno(-0.015, ref=ref)["cumple"] is True
    ref = ref_sintetica(24, {(0, "lightgbm"): 0.95})
    assert _uno(-0.025, ref=ref)["cumple"] is False and _uno(-0.015, ref=ref)["cumple"] is True


def test_la_comparacion_es_exacta_y_sin_redondear_el_borde_en_coma_flotante():
    """`0,73 − 0,75` es −0,020000000000000018 en coma flotante: NO cumple (no se redondea a −0,02). Y existe un par cuya resta da EXACTAMENTE −0,02 (0,01 y −0,01: el doble de 0,01 es exacto, y R² puede ser
    negativo): ese SÍ cumple, porque la comparación es `>=` y no `>`."""
    ref = ref_sintetica(24, {(0, r): 0.75 for r in an.RIVALES})
    ev = _uno(0.73 - 0.75, ref=ref)
    assert ev["distancia"] == (0.75 + (0.73 - 0.75)) - 0.75 and ev["distancia"] < an.DISTANCIA_MINIMA and ev["cumple"] is False
    refx = ref_sintetica(24, {(0, r): 0.01 for r in an.RIVALES})
    nom = refx.regimen()[0]
    for x, cumple in ((-0.01, True), (math.nextafter(-0.01, -1.0), False), (math.nextafter(-0.01, 1.0), True)):
        p = construir(refx)
        for r in p["resultados"]:
            if r["dataset"] == nom:
                r["auroc"] = x
        ev = an.evaluar_conjunto(refx, nom, {(r["repeticion"], r["pliegue"]): r for r in p["resultados"] if r["dataset"] == nom})
        if x == -0.01:
            assert ev["distancia"] == an.DISTANCIA_MINIMA == -0.02, "el par del borde tiene que dar EXACTAMENTE −0,02"
        assert ev["cumple"] is cumple, (x, ev["distancia"])


def test_decide_la_MEDIA_de_pliegues_no_el_intervalo():
    """Un conjunto con mucha dispersión entre pliegues cuya MEDIA está a 1 punto y cuyo IC95 cruza la línea de −0,02 CUMPLE (y el informe lo cuenta aparte); uno con la media a −2,5 puntos y un IC
    estrecho NO cumple."""
    ref = ref_sintetica(24)
    nom = ref.regimen()[0]
    ruidoso = [0.80 - 0.01 + e for e in (0.06, -0.06, 0.03, -0.03, 0.0)]       # media -0,01; pliegues de -7 a +5 puntos
    p = construir(ref)
    for r in p["resultados"]:
        if r["dataset"] == nom:
            r["auroc"] = ruidoso[r["pliegue"]]
    ev = an.evaluar_conjunto(ref, nom, {(r["repeticion"], r["pliegue"]): r for r in p["resultados"] if r["dataset"] == nom})
    assert ev["cumple"] is True and ev["el_ic_cruza_la_linea"] is True
    assert ev["ic95_de_la_distancia"][0] < an.DISTANCIA_MINIMA
    v = veredicto(p, ref)
    assert v["veredicto"] == "CUMPLE" and nom in v["regla"]["el_ic_cruza_la_linea"]
    assert _uno(-0.025)["cumple"] is False


def test_una_media_a_3_puntos_NO_cumple_aunque_el_limite_alto_del_intervalo_llegue_a_los_2():
    """Lo contrario del anterior: con la media a −3 puntos y pliegues muy dispersos, el límite SUPERIOR del IC95 pasa de −0,02, y si decidiera el intervalo ese conjunto cumpliría."""
    ref = ref_sintetica(24)
    nom = ref.regimen()[0]
    p = construir(ref)
    deltas = [0.04, -0.10, 0.02, -0.08, -0.03]            # media −0,03
    for r in p["resultados"]:
        if r["dataset"] == nom:
            r["auroc"] = 0.80 + deltas[r["pliegue"]]
    ev = an.evaluar_conjunto(ref, nom, {(r["repeticion"], r["pliegue"]): r for r in p["resultados"] if r["dataset"] == nom})
    assert ev["distancia"] < an.DISTANCIA_MINIMA and ev["ic95_de_la_distancia"][1] > an.DISTANCIA_MINIMA, ev["ic95_de_la_distancia"]
    assert ev["cumple"] is False


def test_las_dos_medias_son_sobre_los_MISMOS_pliegues():
    """La distancia es la media de las diferencias emparejadas: igual a media(EBM) − media(rival) sobre las mismas claves."""
    ref = ref_sintetica(24, {(0, "lightgbm"): [0.70, 0.80, 0.75, 0.85, 0.90]})
    nom = ref.regimen()[0]
    p = construir(ref, {nom: -0.01})
    ev = an.evaluar_conjunto(ref, nom, {(r["repeticion"], r["pliegue"]): r for r in p["resultados"] if r["dataset"] == nom})
    rival = ref.valores_completos(nom, "lightgbm")
    assert abs(ev["distancia"] - (-0.01)) < 1e-12
    assert math.isclose(ev["distancia"], statistics.fmean(ev["por_pliegue"][f"0.{f}"]["ebm"] - rival[(0, f)] for f in range(5)))


# ---------------------------------------------------------------------------------------------------------------------------
# 2. EL RIVAL: EL MEJOR DE LA CARTERA EN CADA CONJUNTO
# ---------------------------------------------------------------------------------------------------------------------------
def test_el_rival_es_el_MEJOR_de_los_tres_en_ese_conjunto_no_siempre_el_mismo():
    ref = ref_sintetica(24, {(0, "matrixai.dense.tabm_cpu"): 0.90, (1, "sklearn.hgb"): 0.88, (2, "lightgbm"): 0.70, (2, "sklearn.hgb"): 0.60, (2, "matrixai.dense.tabm_cpu"): 0.65})
    assert an.mejor_rival(ref, "s00")["rival"] == "matrixai.dense.tabm_cpu"
    assert an.mejor_rival(ref, "s01")["rival"] == "sklearn.hgb"
    assert an.mejor_rival(ref, "s02")["rival"] == "lightgbm"
    assert an.mejor_rival(ref, "s03")["rival"] == "lightgbm"          # la base: 0,80 > 0,79 > 0,78


def test_el_mejor_se_elige_por_la_MEDIA_de_pliegues_no_por_un_pliegue():
    ref = ref_sintetica(24, {(0, "lightgbm"): [0.99, 0.60, 0.60, 0.60, 0.60], (0, "sklearn.hgb"): 0.70, (0, "matrixai.dense.tabm_cpu"): 0.50})
    assert an.mejor_rival(ref, "s00")["rival"] == "sklearn.hgb"       # 0,70 de media > 0,68


def test_un_empate_exacto_lo_gana_el_primero_del_orden_lightgbm_hgb_tabm():
    ref = ref_sintetica(24, {(0, "lightgbm"): 0.8, (0, "sklearn.hgb"): 0.8, (0, "matrixai.dense.tabm_cpu"): 0.8, (1, "sklearn.hgb"): 0.85, (1, "matrixai.dense.tabm_cpu"): 0.85})
    assert an.mejor_rival(ref, "s00")["rival"] == "lightgbm"
    assert an.mejor_rival(ref, "s01")["rival"] == "sklearn.hgb"


def test_un_rival_con_un_pliegue_fallido_NO_compite_en_ese_conjunto_y_se_dice():
    ref = ref_sintetica(24, {(0, "lightgbm"): 0.95}, falla={(0, "lightgbm"): 2})
    mr = an.mejor_rival(ref, "s00")
    assert mr["rival"] == "sklearn.hgb" and "lightgbm" in mr["excluidos"]


def test_el_rival_no_es_xgboost_ni_catboost_ni_el_lineal():
    ref = an.cargar_referencia()
    assert {"xgboost", "catboost", "sklearn.lineal"} <= {r["motor"] for r in ref.v2["resultados"]}       # la v2 SÍ los trae...
    assert {m for (_, m) in ref._rivales} == set(an.RIVALES)       # ...y la regla no los lee


def test_los_rivales_se_leen_ya_medidos_sin_remedir_el_analizador_no_ejecuta_ningun_motor():
    texto = Path(an.__file__).read_text(encoding="utf-8")
    assert "ejecutar_intento" not in texto and "motor_para" not in texto


# ---------------------------------------------------------------------------------------------------------------------------
# 3. LA REGLA: (a) M ≥ 20, (b) C/M ≥ 0,8 con enteros
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cumplen,esperado", [(24, "CUMPLE"), (20, "CUMPLE"), (19, "NO CUMPLE"), (0, "NO CUMPLE")])
def test_el_80_por_ciento_de_24_son_20_conjuntos_y_19_no_basta(cumplen, esperado):
    ref = ref_sintetica(24)
    v = veredicto(con_cumplen(ref, set(range(cumplen))), ref)
    assert v["veredicto"] == esperado and v["regla"]["b"]["cumplen"] == cumplen and v["regla"]["b"]["minimo_de_cumplen_para_este_M"] == 20


@pytest.mark.parametrize("m,minimo", [(24, 20), (23, 19), (22, 18), (21, 17), (20, 16)])
def test_el_numero_de_conjuntos_que_hace_falta_para_cada_M_es_el_de_la_regla(m, minimo):
    ref = ref_sintetica(24, falla={(i, r): 0 for i in range(24 - m) for r in an.RIVALES})
    # los 24 - m primeros conjuntos NO son comparables (los tres rivales con un pliegue fallido); los m últimos, sí
    for cumplen in (minimo, minimo - 1):
        payload = construir(ref, {nom: (0.0 if (i - (24 - m)) < cumplen else -0.05) for i, nom in enumerate(ref.regimen())})
        v = veredicto(payload, ref)
        assert v["regla"]["n_medibles"] == m
        assert v["veredicto"] == ("CUMPLE" if cumplen == minimo else "NO CUMPLE"), (m, cumplen)
        assert v["regla"]["b"]["minimo_de_cumplen_para_este_M"] == minimo


def test_la_fraccion_se_compara_con_enteros_no_con_flotantes():
    texto = Path(an.__file__).read_text(encoding="utf-8")
    assert "Fraction" in texto and "FRACCION_MINIMA = Fraction(4, 5)" in texto
    assert not re.search(r"fraccion\s*>=\s*0\.8", texto)


def test_con_menos_de_20_comparables_es_NO_CONCLUYENTE_aunque_cumplan_todos():
    ref = ref_sintetica(24, falla={(i, r): 0 for i in range(5) for r in an.RIVALES})        # 5 no comparables -> M = 19
    v = veredicto(construir(ref), ref)
    assert v["regla"]["n_medibles"] == 19 and v["veredicto"] == "NO CONCLUYENTE"
    assert v["regla"]["a"]["cumple"] is False


def test_con_exactamente_20_comparables_la_potencia_se_cumple():
    ref = ref_sintetica(24, falla={(i, r): 0 for i in range(4) for r in an.RIVALES})
    v = veredicto(construir(ref), ref)
    assert v["regla"]["n_medibles"] == 20 and v["regla"]["a"]["cumple"] is True and v["veredicto"] == "CUMPLE"


def test_un_conjunto_sin_rival_comparable_sale_del_denominador_y_se_nombra():
    ref = ref_sintetica(24, falla={(0, r): 1 for r in an.RIVALES})
    v = veredicto(con_cumplen(ref, set(range(1, 21))), ref)
    assert v["regla"]["no_comparables"] == ["s00"] and v["regla"]["n_medibles"] == 23
    assert v["regla"]["b"]["cumplen"] == 20 and v["veredicto"] == "CUMPLE"
    e0 = next(e for e in v["conjuntos"] if e["nombre"] == "s00")
    assert e0["categoria"] == "no_comparable" and e0["medible"] is False


def test_SIN_VEREDICTO_gana_a_NO_CONCLUYENTE():
    ref = ref_sintetica(24, falla={(i, r): 0 for i in range(5) for r in an.RIVALES})
    p = construir(ref)
    p["parcial"] = True
    assert veredicto(p, ref)["veredicto"] == "SIN VEREDICTO"


# ---------------------------------------------------------------------------------------------------------------------------
# 4. UN EBM QUE NO COMPLETA UN CONJUNTO: NO CUMPLE, Y SIGUE EN EL DENOMINADOR
# ---------------------------------------------------------------------------------------------------------------------------
def _fallar(payload, nombres, estado="failed", pliegue=0, motivo="tope de pared agotado"):
    for r in payload["resultados"]:
        if r["dataset"] in nombres and r["pliegue"] == pliegue:
            r["estado"] = estado
            r["motivo"] = motivo
            r["wall_s"] = 301.0
    return payload


def test_un_fallo_del_EBM_cuenta_como_no_cumple_y_sigue_en_el_denominador_20_buenos_y_4_fallos_es_20_de_24():
    ref = ref_sintetica(24)
    nombres = set(ref.regimen()[20:])
    v = veredicto(_fallar(construir(ref), nombres), ref)
    assert v["regla"]["n_medibles"] == 24 and v["regla"]["b"]["cumplen"] == 20 and v["veredicto"] == "CUMPLE"
    assert sorted(v["regla"]["b"]["no_cumplen_por_fallo"]) == sorted(nombres) and v["regla"]["b"]["no_cumplen_por_distancia"] == []


def test_un_fallo_de_mas_y_el_veredicto_cambia_19_buenos_y_5_fallos_es_NO_CUMPLE():
    """Si los fallos se quitaran del denominador, 19/19 = 100 % cumpliría: el motor pasaría la regla fallando donde más cuesta."""
    ref = ref_sintetica(24)
    v = veredicto(_fallar(construir(ref), set(ref.regimen()[19:])), ref)
    assert v["regla"]["b"]["cumplen"] == 19 and v["regla"]["n_medibles"] == 24 and v["veredicto"] == "NO CUMPLE"


def test_un_solo_pliegue_fallido_deja_el_conjunto_entero_como_fallido_aunque_los_otros_cuatro_sean_buenos():
    ref = ref_sintetica(24)
    v = veredicto(_fallar(construir(ref, 0.0), {"s00"}, pliegue=3), ref)
    e0 = next(e for e in v["conjuntos"] if e["nombre"] == "s00")
    assert e0["categoria"] == "no_cumple_por_fallo" and e0["cumple"] is False and e0["completo"] is False and e0["distancia"] is None
    assert e0["fallos_del_ebm"][0]["pliegue"] == 3 and "tope de pared" in e0["fallos_del_ebm"][0]["motivo"]
    assert v["regla"]["b"]["cumplen"] == 23


@pytest.mark.parametrize("estado", ["failed", "cancelled", "unsupported", "timeout", "crashed"])
def test_cualquier_estado_que_no_es_de_medida_deja_el_conjunto_como_fallido(estado):
    ref = ref_sintetica(24)
    v = veredicto(_fallar(construir(ref), {"s05"}, estado=estado), ref)
    assert next(e for e in v["conjuntos"] if e["nombre"] == "s05")["categoria"] == "no_cumple_por_fallo"


def test_completed_budget_limited_cuenta_como_medida():
    ref = ref_sintetica(24)
    p = construir(ref)
    for r in p["resultados"]:
        if r["dataset"] == "s00":
            r["estado"] = "completed_budget_limited"
    assert veredicto(p, ref)["regla"]["b"]["cumplen"] == 24


def test_un_fallo_por_distancia_y_uno_por_fallo_se_cuentan_por_separado_en_el_informe():
    ref = ref_sintetica(24)
    p = _fallar(construir(ref, {"s01": -0.05}), {"s00"})
    v = veredicto(p, ref)
    assert v["regla"]["b"]["no_cumplen_por_distancia"] == ["s01"] and v["regla"]["b"]["no_cumplen_por_fallo"] == ["s00"]
    texto = an.informe_en_texto(v)
    assert "no cumplen por DISTANCIA: ['s01']" in texto and "no cumplen por FALLO del EBM: ['s00']" in texto and "FALLO rep=0 pliegue=0" in texto


def test_no_se_promedian_los_pliegues_buenos_de_un_conjunto_fallido():
    ref = ref_sintetica(24)
    p = _fallar(construir(ref, {"s00": +0.10}), {"s00"})
    e = next(e for e in veredicto(p, ref)["conjuntos"] if e["nombre"] == "s00")
    assert "media_del_ebm" not in e and e["distancia"] is None


# ---------------------------------------------------------------------------------------------------------------------------
# 5. SIN VEREDICTO: la completitud y lo declarado
# ---------------------------------------------------------------------------------------------------------------------------
def _sin_veredicto(payload, ref, trozo):
    v = veredicto(payload, ref)
    assert v["veredicto"] == "SIN VEREDICTO", v["problemas"]
    assert any(trozo in p for p in v["problemas"]), v["problemas"]


def test_sin_veredicto_si_falta_un_pliegue():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["resultados"] = [r for r in p["resultados"] if not (r["dataset"] == "s03" and r["pliegue"] == 2)]
    _sin_veredicto(p, ref, "faltan 1 intentos")


def test_sin_veredicto_si_falta_un_conjunto_del_regimen():
    ref = ref_sintetica(24)
    p = construir(ref, conjuntos=ref.regimen()[:-1])
    _sin_veredicto(p, ref, "los conjuntos no son los")


def test_sin_veredicto_si_sobra_un_conjunto_que_no_es_del_regimen():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["conjuntos"].append(dict(p["conjuntos"][0], nombre="micro-mass"))
    _sin_veredicto(p, ref, "los conjuntos no son los")


def test_sin_veredicto_si_se_midio_un_conjunto_con_mas_de_100_columnas():
    """Un conjunto fuera del régimen (aquí, uno de la v2 con 130 predictores) NO puede colarse en la salida."""
    ref = ref_sintetica(24)
    ref.meta["s23"]["n_predictores"] = 130
    p = construir(ref, conjuntos=ref.meta and sorted(ref.meta))
    _sin_veredicto(p, ref, "FUERA del régimen")


def test_sin_veredicto_si_un_registro_esta_repetido():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["resultados"].append(copy.deepcopy(p["resultados"][0]))
    _sin_veredicto(p, ref, "registrados más de una vez")


def test_sin_veredicto_si_hay_registros_de_mas():
    ref = ref_sintetica(24)
    p = construir(ref)
    extra = copy.deepcopy(p["resultados"][0])
    extra["pliegue"] = 9
    p["resultados"].append(extra)
    _sin_veredicto(p, ref, "fuera de lo declarado")


def test_sin_veredicto_si_la_salida_es_parcial():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["parcial"] = True
    _sin_veredicto(p, ref, "PARCIAL")


def test_sin_veredicto_si_el_conjunto_esta_marcado_sellado():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["conjuntos"][0]["sellado"] = True
    _sin_veredicto(p, ref, "SELLADO")


@pytest.mark.parametrize("campo,valor,trozo", [
    ("semilla", 7, "semilla distinta"), ("presupuesto_wall_s", 5.0, "presupuesto de pared"), ("hilos", 1, "no graba 4 hilos"),
    ("split_plan_digest", "otro", "split_plan_digest"), ("datos_sha256", "otro", "sha256 de los datos"), ("n_test", 7, "n_test"),
    ("entorno_digest", "otro", "entorno_digest"), ("motor_digest", "otro", "motor_digest"), ("metrica_de_cierre", "accuracy", "métrica de cierre distinta"),
    ("engine_version", "1.0.0+interpret-core0.7.7", "engine_version"),
])
def test_sin_veredicto_si_un_registro_declara_lo_que_no_es(campo, valor, trozo):
    ref = ref_sintetica(24)
    p = construir(ref)
    p["resultados"][7][campo] = valor
    _sin_veredicto(p, ref, trozo)


def test_sin_veredicto_si_midio_con_otra_version_de_interpret_core():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["resultados"][3]["hiperparametros"] = {"interpret_core": "0.7.7"}
    _sin_veredicto(p, ref, "interpret-core")


@pytest.mark.parametrize("campo,valor,trozo", [("n_predictores", 11, "predictores"), ("plan_digest", "otro", "plan_digest"), ("datos_sha256", "otro", "sha256"),
                                               ("n_test", 3, "n_test")])
def test_sin_veredicto_si_un_conjunto_declara_lo_que_no_es(campo, valor, trozo):
    ref = ref_sintetica(24)
    p = construir(ref)
    p["conjuntos"][2][campo] = valor
    _sin_veredicto(p, ref, trozo)


def test_sin_veredicto_si_el_codigo_fue_distinto_en_dos_intentos():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["resultados"][0]["entorno_digest"] = "otro"
    p["resultados"][1]["entorno_digest"] = ENTORNO
    v = veredicto(p, ref)
    assert v["veredicto"] == "SIN VEREDICTO"


def test_sin_veredicto_si_un_registro_de_medida_no_trae_metrica_finita_ni_coste_valido():
    ref = ref_sintetica(24)
    for valor in (float("nan"), None, float("inf")):
        p = construir(ref)
        p["resultados"][4]["auroc"] = valor
        v = veredicto(p, ref)
        assert v["veredicto"] == "SIN VEREDICTO" or v["regla"]["b"]["no_cumplen_por_fallo"], valor      # no finito: o sin veredicto o fallido, NUNCA un conjunto que cumple
    for coste in (None, float("nan"), -1.0):
        p = construir(ref)
        p["resultados"][4]["tiempo_de_ajuste"] = coste
        _sin_veredicto(p, ref, "tiempo_de_ajuste")


def test_sin_veredicto_si_la_referencia_de_TabM_no_es_la_misma_particion():
    ref = ref_sintetica(24)
    ref.tabm["particion_comparada_con_la_v2"]["s04"]["plan_digest"] = "otra"
    _sin_veredicto(construir(ref), ref, "plan_digest` de TabM")


def test_la_referencia_real_con_un_EBM_igual_al_mejor_da_CUMPLE():
    ref = an.cargar_referencia()
    v = veredicto(construir(ref, 0.0), ref)
    assert v["problemas"] == [] and v["veredicto"] == "CUMPLE" and v["regla"]["n_regimen"] == 24 == v["regla"]["n_medibles"]


# ---------------------------------------------------------------------------------------------------------------------------
# 6. LOS SELLOS
# ---------------------------------------------------------------------------------------------------------------------------
def test_sin_regla_sellada_no_hay_veredicto(monkeypatch):
    ref = ref_sintetica(24)
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    p = construir(ref)
    v = an.analizar(p, referencia=ref)
    assert v["veredicto"] == "SIN VEREDICTO" and any("NO está sellada" in x for x in v["problemas"])


def test_una_regla_editada_despues_de_sellar_no_da_veredicto(monkeypatch):
    ref = ref_sintetica(24)
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    v = an.analizar(construir(ref), referencia=ref)
    assert v["veredicto"] == "SIN VEREDICTO" and any("no es la sellada" in x for x in v["problemas"])


def test_sin_sello_del_codigo_no_hay_veredicto(monkeypatch, tmp_path):
    ref = ref_sintetica(24)
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", tmp_path / "no_existe.json")
    p = construir(ref)
    v = an.analizar(p, referencia=ref)
    assert v["veredicto"] == "SIN VEREDICTO" and any("falta el sello del código" in x for x in v["problemas"])


def test_un_sello_del_codigo_con_nulos_no_vale(monkeypatch, tmp_path):
    ref = ref_sintetica(24)
    sha = {k: None for k in an.codigo_de_ahora()}
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", escribir_sello_del_codigo(tmp_path / "s.json", sha=sha))
    v = an.analizar(construir(ref), referencia=ref)
    assert v["veredicto"] == "SIN VEREDICTO" and any("SIN SELLAR" in x for x in v["problemas"])


@pytest.mark.parametrize("etiqueta", ["benchmarks/c170/analiza_c170.py", "benchmarks/c170/medir_c170.py", "matrixai-engines/motores/interpret_ebm.py",
                                      "benchmarks/fase0/pasada_v2_113_resultado.json", "benchmarks/fase0/resultado_pasada_119_c3.json", "benchmarks/fase0/pasada_116c2_tabicl.py"])
def test_un_fichero_sellado_que_cambio_despues_no_da_veredicto(monkeypatch, tmp_path, etiqueta):
    ref = ref_sintetica(24)
    sha = dict(an.codigo_de_ahora())
    assert etiqueta in sha, "el sello no cubre este fichero"
    sha[etiqueta] = "f" * 64
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", escribir_sello_del_codigo(tmp_path / "s.json", sha=sha))
    v = an.analizar(construir(ref), referencia=ref)
    assert v["veredicto"] == "SIN VEREDICTO" and any(etiqueta in x for x in v["problemas"])


def test_el_sello_del_codigo_cubre_el_motor_el_medidor_el_analizador_y_los_rivales():
    cubre = set(an.codigo_de_ahora())
    assert {"benchmarks/c170/analiza_c170.py", "benchmarks/c170/medir_c170.py", "matrixai-engines/motores/interpret_ebm.py", "matrixai-engines/subproceso.py",
            "benchmarks/fase0/pasada_v2_113_resultado.json", "benchmarks/fase0/resultado_pasada_119_c3.json", "benchmarks/fase0/protocolo_exploratorio_v2.json"} <= cubre
    assert not any("SELLO_DEL_CODIGO" in k for k in cubre)


def test_un_sello_hecho_sobre_otra_regla_no_vale(monkeypatch, tmp_path):
    ref = ref_sintetica(24)
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", escribir_sello_del_codigo(tmp_path / "s.json", regla="otra"))
    v = an.analizar(construir(ref), referencia=ref)
    assert v["veredicto"] == "SIN VEREDICTO" and any("otra regla" in x for x in v["problemas"])


def test_una_salida_medida_con_otro_codigo_no_da_veredicto():
    ref = ref_sintetica(24)
    p = construir(ref)
    p["analizador_sha256_al_medir"] = "0" * 64
    v = an.analizar(p, referencia=ref)
    assert v["veredicto"] == "SIN VEREDICTO" and any("cambió DESPUÉS de medir" in x for x in v["problemas"])
    p = construir(ref)
    p["dependencias_sha256_al_medir"] = dict(p["dependencias_sha256_al_medir"], **{"matrixai-engines/motores/interpret_ebm.py": "0" * 64})
    assert an.analizar(p, referencia=ref)["veredicto"] == "SIN VEREDICTO"


def test_el_veredicto_cita_la_regla_por_su_sha256():
    ref = ref_sintetica(24)
    v = veredicto(construir(ref), ref)
    assert v["regla_sha256"] == an.REGLA_SHA256_SELLADA == v["regla_sha256_sellada"] and len(v["regla_sha256"]) == 64
    assert an.REGLA_SHA256_SELLADA[:16] in an.informe_en_texto(v)


# ---------------------------------------------------------------------------------------------------------------------------
# 7. SONDA, CÓDIGOS DE SALIDA Y COSTE
# ---------------------------------------------------------------------------------------------------------------------------
def test_una_sonda_no_decide_nada_ni_siquiera_con_los_sellos_rotos(monkeypatch):
    ref = ref_sintetica(24)
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    nom = ref.regimen()[0]
    p = construir(ref, conjuntos=[nom])
    p["es_sonda"] = True
    p["resultados"] = [r for r in p["resultados"] if r["pliegue"] == 0]
    p["conjuntos"][0]["pliegues_esperados"] = [[0, 0]]
    v = an.analizar(p, referencia=ref)
    assert v["veredicto"] == "SONDA" and v["regla"] is None


def test_los_codigos_de_salida_distinguen_los_cinco_veredictos(tmp_path, capsys, monkeypatch):
    assert an.CODIGOS_DE_SALIDA == {"CUMPLE": 0, "NO CUMPLE": 1, "SIN VEREDICTO": 2, "NO CONCLUYENTE": 3, "SONDA": 4}
    ref = ref_sintetica(24)
    monkeypatch.setattr(an, "cargar_referencia", lambda *a, **k: ref)
    for cumplen, codigo in ((24, 0), (0, 1)):
        ruta = tmp_path / f"r{cumplen}.json"
        ruta.write_text(json.dumps(_sellar(con_cumplen(ref, set(range(cumplen))))), encoding="utf-8")
        assert an.main([str(ruta)]) == codigo
    ruta = tmp_path / "malo.json"
    ruta.write_text("{esto no es json", encoding="utf-8")
    assert an.main([str(ruta)]) == 2                      # una salida malformada NO sale con el 1 de NO CUMPLE
    capsys.readouterr()


def test_el_coste_se_informa_y_NO_entra_en_el_veredicto():
    ref = ref_sintetica(24)
    barato = veredicto(construir(ref, coste=1.0), ref)
    carisimo = veredicto(construir(ref, coste=100000.0), ref)
    assert barato["veredicto"] == carisimo["veredicto"] == "CUMPLE"
    assert carisimo["coste"]["ebm_s_total"] == 24 * PLIEGUES * 100000.0
    assert carisimo["coste"]["razon_de_sumas_ebm_sobre_los_tres_rivales"] == pytest.approx(100000.0 / 3.0)


def test_el_coste_da_mediana_maxima_y_por_conjunto():
    ref = ref_sintetica(24)
    p = construir(ref)
    for r in p["resultados"]:
        if r["dataset"] == "s00":
            r["tiempo_de_ajuste"] = 90.0
    c = veredicto(p, ref)["coste"]
    assert c["ebm_s_por_pliegue_maxima_por_conjunto"] == 90.0 and c["ebm_s_por_pliegue_mediana_por_conjunto"] == 10.0
    assert c["por_conjunto_s_por_pliegue"]["s00"] == 90.0 and c["n_pliegues_fallidos"] == 0


def test_REGLA_dice_que_el_coste_no_decide():
    assert "NO entra en el veredicto" in REGLA
