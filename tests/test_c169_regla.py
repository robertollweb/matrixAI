# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169-C1 — el ANALIZADOR de `benchmarks/c169/analiza_c169.py` contra `REGLA.md`: una prueba por condición, con su nombre.

Todo con salidas SINTÉTICAS: ninguna prueba llama a `main()`, descarga ni mide. El sello se simula con monkeypatch de `REGLA_SHA256_SELLADA`
(el real lo pone el supervisor).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_C169 = Path(__file__).resolve().parents[1] / "benchmarks" / "c169"
if str(_C169) not in sys.path:
    sys.path.insert(0, str(_C169))

import analiza_c169 as an  # noqa: E402
import conjuntos_c169 as cj  # noqa: E402

NOMBRES = [d.nombre for d in cj.conjuntos()]
TAREA = {d.nombre: d.tarea for d in cj.conjuntos()}
METRICA = {"binary_classification": "auroc", "multiclass_classification": "accuracy", "regression": "r2"}
SHA = {d.nombre: d.sha256_arff for d in cj.conjuntos()}
ID = {d.nombre: d.data_id for d in cj.conjuntos()}
# Cinco pliegues de referencia con algo de dispersión (para que el IC no sea degenerado salvo que se pida).
HOY_BASE = [0.80, 0.82, 0.78, 0.81, 0.79]
RUIDO = [0.0, 0.01, -0.01, 0.005, -0.005]


def _registro(nombre, brazo, f, valor, coste, *, elegida=1):
    es_rejilla = brazo == an.REJILLA
    metric = METRICA[TAREA[nombre]]
    hip = {"politica_por_tamano": {"aplicada": True}}
    if es_rejilla:
        cfgs = [{"validacion_mejor": 0.5 + 0.01 * i} for i in range(8)]
        hip["politica_por_tamano"]["validacion_mejor"] = 0.5
        hip["rejilla_por_validacion"] = {"aplicada": True, "mira_la_prueba": False, "criterio": "perdida_en_validation_del_pliegue",
                                         "configuraciones": cfgs, "elegida": 0}
    return {"dataset": nombre, "data_id": ID[nombre], "motor": brazo, "repeticion": 0, "pliegue": f, "estado": "completed",
            "metrica_de_cierre": metric, metric: valor, "tiempo_de_ajuste": coste, "hiperparametros": hip,
            "entorno_digest": "e1", "motor_digest": f"m-{brazo}", "datos_sha256": SHA[nombre], "sellado": False}


def construir(especificacion: dict | None = None, *, coste_hoy=1.0, coste_rej=3.0) -> dict:
    """`especificacion[nombre] = (lista_hoy, lista_rejilla)`; lo no dicho: hoy = HOY_BASE y rejilla = hoy (R = 0)."""
    especificacion = especificacion or {}
    conjuntos, resultados = [], []
    for n in NOMBRES:
        hoy, rej = especificacion.get(n, (HOY_BASE, HOY_BASE))
        conjuntos.append({"nombre": n, "data_id": ID[n], "tarea": TAREA[n], "metrica_de_cierre": METRICA[TAREA[n]], "sellado": False,
                          "pliegues_esperados": [0, 1, 2, 3, 4],
                          "columnas_ignoradas_por_openml": cj.ignorados_registrados()[ID[n]]})
        for f in range(5):
            resultados.append(_registro(n, an.REFERENCIA, f, hoy[f], coste_hoy))
            resultados.append(_registro(n, an.REJILLA, f, rej[f], coste_rej))
    payload = {"es_sonda": False, "parcial": False, "conjuntos": conjuntos, "resultados": resultados}
    s = an.sellos_de_ahora()
    payload.update(regla_sha256_sellada=an.REGLA_SHA256_SELLADA, regla_sha256_al_medir=s["regla_sha256"],
                   analizador_sha256_al_medir=s["analizador_sha256"], dependencias_sha256_al_medir=s["dependencias_sha256"])
    return payload


@pytest.fixture(autouse=True)
def sellada(monkeypatch):
    """La regla «sellada» = el REGLA.md de ahora. Las pruebas de la negativa la vuelven a poner a None."""
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", an.sellos_de_ahora()["regla_sha256"])


def _con_sello(payload):
    payload["regla_sha256_sellada"] = an.REGLA_SHA256_SELLADA
    return payload


def veredicto(payload):
    return an.analizar(_con_sello(payload))


def mejor(valores, delta):
    return [v + d for v, d in zip(valores, delta)]


#: Rejilla que mejora CLARAMENTE (IC de R > 0): el error baja un 20 % en los cinco pliegues, sin dispersión.
def _mejora(base=HOY_BASE, frac=0.20):
    return base, [1 - (1 - v) * (1 - frac) for v in base]


def _empeora(base=HOY_BASE, frac=0.20):
    return base, [1 - (1 - v) * (1 + frac) for v in base]


def test_la_regla_sellada_y_dos_veces_sus_umbrales_son_los_de_REGLA_md():
    """Los números de la regla, fijados contra el texto de REGLA.md (no contra el código)."""
    txt = an.RUTA_DE_LA_REGLA.read_text(encoding="utf-8")
    assert an.UMBRAL_DE_SATURACION == 0.98 and "**≥ 0,98**" in txt
    assert an.N_MINIMO_NO_SATURADOS == 3 and "N ≥ **3**" in txt
    assert an.A_REDUCCION_MEDIA_MIN == 0.03 and "≥ **+3 %**" in txt
    assert an.C_SATURADO_EMPEORA_D == -0.01 and "**−0,01**" in txt
    assert an.D_COSTE_MAXIMO == 8.0 and "**8 ×**" in txt
    assert an.C_EMPEORA_MAS_DE == 0.03 and "R < −3 %" in txt


def test_confirmada_cuando_se_dan_las_cinco():
    esp = {n: _mejora() for n in NOMBRES}
    v = veredicto(construir(esp))
    assert v["problemas"] == [] and v["veredicto"] == "CONFIRMADA", v["regla"]
    assert v["regla"]["n_no_saturados"] == 6


# ---------------------------------------------------------------- (0) potencia mínima
def test_cero_menos_de_3_no_saturados_es_NO_CONCLUYENTE_aunque_todo_lo_demas_cumpla():
    sat = ([0.99] * 5, [0.995] * 5)                       # saturados (hoy >= 0,98) y la rejilla algo mejor
    esp = {n: sat for n in NOMBRES[:4]}
    esp.update({n: _mejora() for n in NOMBRES[4:]})       # N = 2
    v = veredicto(construir(esp))
    assert v["regla"]["n_no_saturados"] == 2
    assert v["veredicto"] == "NO CONCLUYENTE"


def test_cero_con_exactamente_3_no_saturados_ya_se_juzga():
    sat = ([0.99] * 5, [0.99] * 5)
    esp = {n: sat for n in NOMBRES[:3]}
    esp.update({n: _mejora() for n in NOMBRES[3:]})       # N = 3, las tres mejoran claro
    v = veredicto(construir(esp))
    assert v["regla"]["n_no_saturados"] == 3 and v["veredicto"] == "CONFIRMADA", v["regla"]


# ---------------------------------------------------------------- (a) media de R >= +3 %
def test_a_media_de_R_por_debajo_de_3_por_ciento_es_NO_CUMPLE():
    # Todos mejoran 2,5 % de error (R = +0,025), con IC > 0 (b cumple): solo falla (a).
    esp = {n: _mejora(frac=0.025) for n in NOMBRES}
    v = veredicto(construir(esp))
    assert v["regla"]["b"]["cumple"] and not v["regla"]["a"]["cumple"]
    assert v["regla"]["a"]["media_r_no_saturados"] == pytest.approx(0.025)
    assert v["veredicto"] == "NO CUMPLE"


def test_a_en_3_por_ciento_justo_cumple_y_se_compara_sin_redondear():
    esp = {n: _mejora(frac=0.0301) for n in NOMBRES}
    assert veredicto(construir(esp))["regla"]["a"]["cumple"]
    esp = {n: _mejora(frac=0.0299) for n in NOMBRES}
    assert not veredicto(construir(esp))["regla"]["a"]["cumple"]


# ---------------------------------------------------------------- (b) ceil(N/2) mejoras claras
def _con_k_mejoras(k, frac_resto=0.0):
    """k conjuntos con mejora clara fuerte; el resto, ruidosos y con R medio alto pero IC que cruza 0 (no «claros»)."""
    ruido_pos = ([0.80, 0.82, 0.78, 0.81, 0.79],
                 [1 - (1 - v) * m for v, m in zip([0.80, 0.82, 0.78, 0.81, 0.79], [0.50, 1.30, 0.50, 1.30, 0.50])])
    esp = {}
    for i, n in enumerate(NOMBRES):
        esp[n] = _mejora(frac=0.30) if i < k else ruido_pos
    return esp


def test_b_con_N_6_hacen_falta_3_mejoras_claras_y_con_2_falla():
    v3 = veredicto(construir(_con_k_mejoras(3)))
    assert v3["regla"]["b"]["umbral"] == 3 and v3["regla"]["b"]["n"] == 3 and v3["regla"]["b"]["cumple"]
    v2 = veredicto(construir(_con_k_mejoras(2)))
    assert v2["regla"]["b"]["n"] == 2 and not v2["regla"]["b"]["cumple"] and v2["veredicto"] == "NO CUMPLE"


def test_b_con_N_5_el_umbral_es_ceil_5_2_igual_a_3_no_2():
    sat = ([0.99] * 5, [0.99] * 5)
    esp = _con_k_mejoras(2)
    esp[NOMBRES[5]] = sat                                      # N = 5
    v = veredicto(construir(esp))
    assert v["regla"]["n_no_saturados"] == 5 and v["regla"]["b"]["umbral"] == 3 and not v["regla"]["b"]["cumple"]
    esp = _con_k_mejoras(3)
    esp[NOMBRES[5]] = sat
    assert veredicto(construir(esp))["regla"]["b"]["cumple"]


def test_b_una_mejora_clara_de_un_saturado_no_cuenta():
    sat_mejora = ([0.985, 0.986, 0.984, 0.987, 0.983], [0.995, 0.996, 0.994, 0.997, 0.993])
    esp = _con_k_mejoras(2)                                    # 2 claras entre no saturados
    esp[NOMBRES[5]] = sat_mejora                               # N = 5 -> hacen falta 3
    v = veredicto(construir(esp))
    assert v["regla"]["n_no_saturados"] == 5 and v["regla"]["b"]["n"] == 2 and not v["regla"]["b"]["cumple"]


# ---------------------------------------------------------------- (c) no saturado: R < -3 % y IC < 0
def test_c_un_no_saturado_que_empeora_de_verdad_tumba_aunque_el_resto_cumpla():
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    esp[NOMBRES[0]] = _empeora(frac=0.20)                      # R = -0,20, IC entero < 0
    v = veredicto(construir(esp))
    assert NOMBRES[0] in v["regla"]["c"]["empeoran"] and not v["regla"]["c"]["cumple"]
    assert v["regla"]["a"]["cumple"] and v["regla"]["b"]["cumple"]       # solo (c) falla
    assert v["veredicto"] == "NO CUMPLE"


def test_c_un_no_saturado_con_R_menor_que_menos_3_pero_IC_que_cruza_0_no_empeora_de_verdad():
    base = [0.80, 0.82, 0.78, 0.81, 0.79]
    rej = [1 - (1 - v) * m for v, m in zip(base, [0.60, 1.60, 0.60, 1.60, 1.30])]   # R medio < -3 %, IC amplio
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    esp[NOMBRES[0]] = (base, rej)
    v = veredicto(construir(esp))
    e = next(x for x in v["conjuntos"] if x["nombre"] == NOMBRES[0])
    assert e["r"] < -0.03 and e["ic"][1] >= 0.0 and not e["empeora_de_verdad"]
    assert v["regla"]["c"]["cumple"]


# ---------------------------------------------------------------- (c) saturado: D < -0,01 y IC de D < 0
def _saturado_con_d(d, ruido=None):
    hoy = [0.99, 0.991, 0.989, 0.992, 0.988]
    ruido = ruido or [0.0] * 5
    return hoy, [h + d + r for h, r in zip(hoy, ruido)]


def _con_un_saturado(d, ruido=None):
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    esp[NOMBRES[0]] = _saturado_con_d(d, ruido)
    return esp


def test_c_saturado_con_D_menos_0_0101_y_IC_bajo_cero_EMPEORA_DE_VERDAD():
    v = veredicto(construir(_con_un_saturado(-0.0101)))
    e = next(x for x in v["conjuntos"] if x["nombre"] == NOMBRES[0])
    assert e["saturado"] and e["d"] < -0.01 and e["ic_d"][1] < 0
    assert e["empeora_de_verdad"] and NOMBRES[0] in v["regla"]["c"]["empeoran"] and v["veredicto"] == "NO CUMPLE"


def test_c_saturado_con_D_menos_0_0099_NO_empeora_de_verdad():
    v = veredicto(construir(_con_un_saturado(-0.0099)))
    e = next(x for x in v["conjuntos"] if x["nombre"] == NOMBRES[0])
    assert e["saturado"] and e["d"] > -0.01 and not e["empeora_de_verdad"]
    assert v["regla"]["c"]["cumple"] and v["veredicto"] == "CONFIRMADA", v["regla"]


def test_c_saturado_con_D_bajo_el_umbral_pero_IC_que_cruza_0_NO_empeora_de_verdad():
    ruido = [0.05, -0.05, 0.05, -0.05, 0.0]                    # D medio ~ -0,0121 con IC amplio
    v = veredicto(construir(_con_un_saturado(-0.0121, ruido)))
    e = next(x for x in v["conjuntos"] if x["nombre"] == NOMBRES[0])
    assert e["d"] < -0.01 and e["ic_d"][1] >= 0.0 and not e["empeora_de_verdad"]


def test_c_el_umbral_de_un_saturado_es_ABSOLUTO_y_el_de_R_no_se_le_aplica():
    """sick en el 163-C2: R = -68 % sobre el error pero D = -0,42 puntos. Aquí D = -0,0042: no empeora de verdad aunque R sea muy negativo."""
    v = veredicto(construir(_con_un_saturado(-0.0042)))
    e = next(x for x in v["conjuntos"] if x["nombre"] == NOMBRES[0])
    assert e["saturado"] and e["r"] < -0.03 and e["ic"][1] < 0.0       # por R sería «empeora de verdad»…
    assert not e["empeora_de_verdad"] and v["regla"]["c"]["cumple"]    # …pero un saturado se juzga por D


# ---------------------------------------------------------------- saturado: lo decide la REFERENCIA, no la rejilla
def test_saturado_lo_decide_la_media_de_hgb_hoy_y_NO_la_rejilla():
    hoy = [0.975, 0.976, 0.974, 0.977, 0.973]                  # media 0,975 < 0,98
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    esp[NOMBRES[0]] = (hoy, [0.999] * 5)                       # la rejilla sube a 0,999 pero NO la hace saturada
    v = veredicto(construir(esp))
    e = next(x for x in v["conjuntos"] if x["nombre"] == NOMBRES[0])
    assert not e["saturado"] and NOMBRES[0] in v["regla"]["no_saturados"]
    hoy2 = [0.985, 0.986, 0.984, 0.987, 0.983]                 # media 0,985 >= 0,98
    esp[NOMBRES[0]] = (hoy2, [0.90] * 5)                       # y aunque la rejilla caiga a 0,90 SIGUE saturado
    e = next(x for x in veredicto(construir(esp))["conjuntos"] if x["nombre"] == NOMBRES[0])
    assert e["saturado"]


def test_saturado_en_el_umbral_exacto_0_98_ya_es_saturado(monkeypatch):
    # El supervisor (15-11): fmean([0.98]*5) es 0.9800000000000001, así que con el umbral 0.98 esta prueba no estaba en el
    # borde y un `>` en vez de `>=` salía VERDE. Se fija el umbral a la media EXACTA de los pliegues: «≥» la hace saturada, «>» no.
    import statistics
    hoy = [0.98] * 5
    monkeypatch.setattr(an, "UMBRAL_DE_SATURACION", statistics.fmean(hoy))
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    esp[NOMBRES[0]] = (hoy, hoy)
    assert next(x for x in veredicto(construir(esp))["conjuntos"] if x["nombre"] == NOMBRES[0])["saturado"]


# ---------------------------------------------------------------- (d) coste
def test_d_razon_de_sumas_mayor_que_8_es_NO_CUMPLE_y_8_justo_cumple():
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    v = veredicto(construir(esp, coste_hoy=1.0, coste_rej=8.5))
    assert v["regla"]["d"]["razon_de_sumas"] == pytest.approx(8.5) and not v["regla"]["d"]["cumple"] and v["veredicto"] == "NO CUMPLE"
    v = veredicto(construir(esp, coste_hoy=1.0, coste_rej=8.0))
    assert v["regla"]["d"]["cumple"] and v["veredicto"] == "CONFIRMADA"


def test_d_es_razon_de_SUMAS_no_media_de_razones_y_cuenta_los_6_conjuntos():
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    p = construir(esp)
    for r in p["resultados"]:                                   # un conjunto con ratio 20 pero casi sin peso en la suma
        if r["dataset"] == NOMBRES[0]:
            r["tiempo_de_ajuste"] = 0.01 if r["motor"] == an.REFERENCIA else 0.2
        else:
            r["tiempo_de_ajuste"] = 1.0 if r["motor"] == an.REFERENCIA else 2.0
    v = veredicto(p)
    assert v["regla"]["d"]["razon_de_sumas"] < 3 and v["regla"]["d"]["maxima_por_conjunto"] == pytest.approx(20.0)
    assert v["regla"]["d"]["cumple"]


# ---------------------------------------------------------------- completitud -> SIN VEREDICTO
def test_completitud_falta_un_pliegue_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    p["resultados"] = [r for r in p["resultados"] if not (r["dataset"] == NOMBRES[2] and r["motor"] == an.REJILLA and r["pliegue"] == 3)]
    v = veredicto(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("faltan" in x for x in v["problemas"])


def test_completitud_falta_un_conjunto_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    quitado = NOMBRES[5]
    p["conjuntos"] = [c for c in p["conjuntos"] if c["nombre"] != quitado]
    p["resultados"] = [r for r in p["resultados"] if r["dataset"] != quitado]
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_completitud_un_intento_que_no_cuenta_como_medida_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    p["resultados"][0]["estado"] = "failed"
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_completitud_un_intento_sin_tiempo_de_ajuste_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    p["resultados"][1]["tiempo_de_ajuste"] = None
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_completitud_un_registro_repetido_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    p["resultados"].append(copy.deepcopy(p["resultados"][0]))
    v = veredicto(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("más de una vez" in x for x in v["problemas"])


def test_completitud_un_brazo_que_declara_lo_que_no_es_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    for r in p["resultados"]:
        if r["motor"] == an.REJILLA:
            r["hiperparametros"].pop("rejilla_por_validacion")        # una «rejilla» sin rejilla
            break
    v = veredicto(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("brazos cruzados" in x for x in v["problemas"])


def test_completitud_una_referencia_que_declara_rejilla_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    rej = next(r for r in p["resultados"] if r["motor"] == an.REJILLA)
    ref = next(r for r in p["resultados"] if r["motor"] == an.REFERENCIA)
    ref["hiperparametros"]["rejilla_por_validacion"] = rej["hiperparametros"]["rejilla_por_validacion"]
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_completitud_la_salida_parcial_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    p["parcial"] = True
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_completitud_un_conjunto_que_no_es_de_los_6_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    p["conjuntos"][0]["nombre"] = "sick"
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_completitud_codigo_distinto_entre_intentos_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    p["resultados"][3]["entorno_digest"] = "otro"
    v = veredicto(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("CÓDIGO DISTINTO" in x for x in v["problemas"])


# ---------------------------------------------------------------- datos y sellos
def test_el_sha256_de_los_datos_que_no_cuadra_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    for r in p["resultados"]:
        if r["dataset"] == NOMBRES[1]:
            r["datos_sha256"] = "0" * 64
    v = veredicto(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("sha256 de los datos" in x for x in v["problemas"])


def test_negativa_sin_sello_el_analizador_no_da_veredicto(monkeypatch):
    p = construir({n: _mejora() for n in NOMBRES})
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    p["regla_sha256_sellada"] = None
    v = an.analizar(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("NO está sellada" in x for x in v["problemas"])


def test_negativa_regla_distinta_de_la_sellada_no_da_veredicto(monkeypatch):
    p = construir({n: _mejora() for n in NOMBRES})
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "ab" * 32)
    p["regla_sha256_sellada"] = "ab" * 32
    v = an.analizar(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("no es la sellada" in x for x in v["problemas"])


def test_negativa_el_analizador_cambiado_despues_de_medir_no_da_veredicto():
    p = construir({n: _mejora() for n in NOMBRES})
    p["analizador_sha256_al_medir"] = "cd" * 32
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_negativa_una_dependencia_cambiada_despues_de_medir_no_da_veredicto():
    p = construir({n: _mejora() for n in NOMBRES})
    k = next(iter(p["dependencias_sha256_al_medir"]))
    p["dependencias_sha256_al_medir"][k] = "ef" * 32
    assert veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_una_sonda_no_decide_nada():
    p = construir({n: _mejora() for n in NOMBRES})
    p["es_sonda"] = True
    assert veredicto(p)["veredicto"] == "SONDA"


def test_codigos_de_salida_de_main(tmp_path):
    import json
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(construir({n: _mejora() for n in NOMBRES})))
    assert an.main([str(ok)]) == 0
    mal = tmp_path / "mal.json"
    mal.write_text(json.dumps(construir({n: _mejora(frac=0.0) for n in NOMBRES})))
    assert an.main([str(mal)]) == 1
    p = construir({n: _mejora() for n in NOMBRES})
    p["resultados"].pop()
    inc = tmp_path / "inc.json"
    inc.write_text(json.dumps(p))
    assert an.main([str(inc)]) == 2


# ---------------------------------------------------------------- (a) solo sobre los NO saturados, y en el umbral exacto
def test_a_la_media_de_R_es_solo_de_los_no_saturados_un_saturado_con_R_muy_negativo_no_la_baja():
    esp = {n: _mejora(frac=0.05) for n in NOMBRES}                       # 5 no saturados con R = +0,05
    esp[NOMBRES[0]] = ([0.99] * 5, [0.985] * 5)                          # saturado: error 0,01 -> 0,015 (R = -0,5), D = -0,005
    v = veredicto(construir(esp))
    assert v["regla"]["n_no_saturados"] == 5
    assert v["regla"]["a"]["media_r_no_saturados"] == pytest.approx(0.05)
    assert v["veredicto"] == "CONFIRMADA", v["regla"]


def test_a_en_el_umbral_EXACTO_de_3_por_ciento_cumple_y_justo_por_debajo_no():
    def ev(nombre, r):
        return {"nombre": nombre, "saturado": False, "r": r, "mejora_clara": True, "empeora_de_verdad": False,
                "coste_s": {an.REFERENCIA: 1.0, an.REJILLA: 2.0}, "coste_ratio": 2.0}
    assert an.aplicar_la_regla([ev(f"c{i}", 0.03) for i in range(4)])["a"]["cumple"]
    assert not an.aplicar_la_regla([ev(f"c{i}", 0.0299999) for i in range(4)])["a"]["cumple"]


# ---------------------------------------------------------------- orden de prioridad, comparaciones exactas, coste nulo (REGLA.md «Orden y bordes»)
def test_prioridad_SIN_VEREDICTO_gana_a_NO_CONCLUYENTE():
    sat = ([0.99] * 5, [0.99] * 5)
    esp = {n: sat for n in NOMBRES[:4]}                       # N = 2 -> NO CONCLUYENTE si la medida fuera completa
    p = construir(esp)
    assert veredicto(copy.deepcopy(p))["veredicto"] == "NO CONCLUYENTE"
    p["resultados"][0]["datos_sha256"] = "0" * 64              # y ahora un sha256 de datos no cuadra (la regla SÍ se puede calcular)
    v = veredicto(p)
    assert v["regla"] is not None and v["regla"]["veredicto"] == "NO CONCLUYENTE" and v["veredicto"] == "SIN VEREDICTO"


def test_prioridad_NO_CONCLUYENTE_gana_a_NO_CUMPLE_aunque_falle_otra_condicion():
    sat = ([0.99] * 5, [0.99] * 5)
    esp = {n: sat for n in NOMBRES[:4]}
    esp[NOMBRES[4]] = _empeora()                              # (c) fallaría
    v = veredicto(construir(esp, coste_hoy=1.0, coste_rej=20.0))   # y (d) también
    assert not v["regla"]["c"]["cumple"] and not v["regla"]["d"]["cumple"]
    assert v["veredicto"] == "NO CONCLUYENTE"


def test_comparaciones_exactas_D_exactamente_en_el_umbral_NO_es_menor_y_una_milmillonesima_por_debajo_si(monkeypatch):
    monkeypatch.setattr(an, "C_SATURADO_EMPEORA_D", -0.25)    # un umbral representable: D = -0,25 EXACTO en coma flotante
    def v_con(rej):
        esp = {n: _mejora(frac=0.30) for n in NOMBRES}
        esp[NOMBRES[0]] = ([1.0] * 5, [rej] * 5)
        return next(x for x in veredicto(construir(esp))["conjuntos"] if x["nombre"] == NOMBRES[0])
    e = v_con(0.75)
    assert e["d"] == -0.25 and not e["empeora_de_verdad"]
    e = v_con(0.75 - 1e-9)
    assert e["d"] < -0.25 and e["empeora_de_verdad"]


def test_coste_nulo_de_la_referencia_d_no_se_cumple():
    esp = {n: _mejora(frac=0.30) for n in NOMBRES}
    v = veredicto(construir(esp, coste_hoy=0.0, coste_rej=1.0))
    assert v["regla"]["d"]["razon_de_sumas"] is None and not v["regla"]["d"]["cumple"] and v["veredicto"] == "NO CUMPLE"


# ---------------------------------------------------------------- ignore_attribute de OpenML
def test_ignore_attribute_un_resultado_que_no_declara_la_exclusion_de_cooling_load_es_SIN_VEREDICTO():
    p = construir({n: _mejora() for n in NOMBRES})
    assert veredicto(copy.deepcopy(p))["veredicto"] == "CONFIRMADA"
    for c in p["conjuntos"]:
        if c["nombre"] == "energy_efficiency":
            c["columnas_ignoradas_por_openml"] = []
    v = veredicto(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("cooling_load" in x for x in v["problemas"])
