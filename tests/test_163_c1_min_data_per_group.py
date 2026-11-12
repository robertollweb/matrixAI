# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""163-C1 — `min_data_per_group` por tamaño en LightGBM: el analizador, la regla y el guion de medida (`benchmarks/c163/`). Nada entrena:
todo con salidas FABRICADAS con los 23 nombres reales del congelado.

Fija: que la regla escrita y los umbrales del código dicen lo mismo; que los AFECTADOS / sin efecto / control son disjuntos, están entre los 23 y
no son saturados; cada condición (a)–(e) con un caso que la cumple y otro que no; SIN VEREDICTO si falta un pliegue, un brazo, un conjunto, si la
salida es parcial, si los brazos están cruzados, si en un afectado la palanca no actuó o en un no afectado sí, si el control (donde no actúa debe
dar lo mismo) se rompe, o si la regla no está sellada / no es la sellada; la confirmación (escalada); y que el guion monta los dos brazos
con la palanca apagada en la referencia y encendida en la otra, y se niega a medir sin sello.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
for _r in ("c163", "c160", "fase0"):
    sys.path.insert(0, str(_RAIZ / "benchmarks" / _r))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import analiza_c160 as a160  # noqa: E402
import analiza_c163_c1 as an  # noqa: E402
import medir_c163_c1 as med  # noqa: E402

R, P = an.REFERENCIA, an.PALANCA
CONGELADO = a160.conjuntos_y_pliegues_del_congelado()
NOMBRES = list(CONGELADO)
REGLA = (_RAIZ / "benchmarks" / "c163" / "REGLA_C1.md").read_text(encoding="utf-8")
AF = list(an.AFECTADOS)


# ---------------------------------------------------------------------------
# 0. SALIDAS FABRICADAS
# ---------------------------------------------------------------------------
def _registro(ds, brazo, pliegue, metrica, wall=10.0, estado="completed", mdpg=None, metric_id="auroc"):
    """`mdpg`: el valor que la palanca declara haber pasado (None = sin categóricas: pedida, no aplicada)."""
    hip = {"n_estimators": 100, "num_leaves": 15}
    if brazo == P:
        hip["min_data_per_group_por_tamano"] = (
            {"pedida": True, "aplicada": True, "filas_de_train": 300, "min_data_per_group": mdpg} if mdpg is not None
            else {"pedida": True, "aplicada": False, "filas_de_train": 300, "min_data_per_group": 100})
    return {"dataset": ds, "motor": brazo, "repeticion": 0, "pliegue": pliegue, "estado": estado, "metrica_de_cierre": metric_id,
            metric_id: metrica, "wall_s": wall, "hiperparametros": hip, "entorno_digest": "e" * 16, "motor_digest": "m" * 16}


def _payload(ref=0.90, pal=0.92, wall_pal=10.0, *, a_medida=None, es_sonda=False, mdpg_de=None):
    """Una salida completa de 23 conjuntos. Los AFECTADOS: la palanca (`pal`) cambia la métrica y declara min_data_per_group 15; los demás:
    la palanca da EXACTAMENTE `ref` (el control). `a_medida(ds, brazo, pliegue) -> métrica` sobreescribe casos."""
    conjuntos, resultados = [], []
    for n, info in CONGELADO.items():
        conjuntos.append({"nombre": n, "tarea": "binary_classification", "sellado": False, "metrica_de_cierre": "auroc",
                          "pliegues_esperados": list(info["pliegues"])})
        for f in info["pliegues"]:
            for brazo in (R, P):
                base = ref if (brazo == R or n not in AF) else pal
                m = a_medida(n, brazo, f) if a_medida else None
                m = base + 0.001 * f if m is None else m
                mdpg = (mdpg_de(n, f) if mdpg_de else (15 if n in AF else None)) if brazo == P else None
                resultados.append(_registro(n, brazo, f, m, wall=10.0 if brazo == R else wall_pal, mdpg=mdpg))
    return {"es_sonda": es_sonda, "parcial": False, "conjuntos": conjuntos, "resultados": resultados}


def _veredicto(payload, **kw):
    return an.analizar(payload, comprobar_sellos=False, **kw)


def _ev(r_afectados, claras, *, empeora=False, coste=1.0):
    """Evaluados sintéticos: 3 afectados + 2 de control."""
    out = []
    for i in range(3):
        out.append({"nombre": f"a{i}", "completo": True, "afectado": True, "saturado": False, "r": r_afectados[i],
                    "mejora_clara": claras[i], "empeora_de_verdad": empeora and i == 0, "coste_s": {R: 1.0, P: coste},
                    "coste_ratio": coste, "aplicada_en_pliegues": [0]})
    for i in range(2):
        out.append({"nombre": f"k{i}", "completo": True, "afectado": False, "saturado": False, "r": 0.0, "mejora_clara": False,
                    "empeora_de_verdad": False, "coste_s": {R: 1.0, P: 1.0}, "coste_ratio": 1.0, "aplicada_en_pliegues": []})
    return out


# ---------------------------------------------------------------------------
# 1. DECLARACIONES ANTES DE MEDIR
# ---------------------------------------------------------------------------
def test_los_grupos_son_disjuntos_estan_entre_los_23_y_ningun_afectado_es_saturado():
    todos = set(an.AFECTADOS) | set(an.CON_CATEGORICAS_SIN_EFECTO)
    assert len(todos) == len(an.AFECTADOS) + len(an.CON_CATEGORICAS_SIN_EFECTO) == 5
    assert todos <= set(NOMBRES) and len(NOMBRES) == an.N_CONJUNTOS == 23
    assert not set(an.AFECTADOS) & set(an.SATURADOS), "un afectado saturado no puede mejorar y distorsiona (a)"
    assert tuple(an.AFECTADOS) == ("dresses-sales", "Moneyball", "house_prices_nominal")


def test_los_umbrales_del_codigo_son_los_de_la_regla_escrita():
    assert an.A_REDUCCION_MEDIA_MIN == 0.03 and "≥ **+3 %**" in REGLA
    assert an.B_MEJORAS_CLARAS_MIN == 2 and "≥ **2 de los 3**" in REGLA
    assert an.C_EMPEORA_MAS_DE == 0.03 and "R < −3 %" in REGLA
    assert an.D_COSTE_MAXIMO == 1.5 and "≤ **1,5 ×**" in REGLA
    assert an.E_CONTROL_TOLERANCIA == 1e-9 and "1e-9" in REGLA
    assert an.CONFIRMACION_AFECTADOS_MIN == 2 and "al menos 2" in REGLA
    for n in an.AFECTADOS:
        assert n in REGLA
    assert "min(100, max(5, filas_de_train // 20))" in REGLA


def test_los_brazos_estan_en_el_orden_que_el_analizador_espera():
    assert an.BRAZOS == (an.REFERENCIA, an.PALANCA) == ("lightgbm_hoy", "lightgbm_min_data_per_group")


def test_la_formula_de_la_regla_es_la_del_motor():
    from matrixai_engines.motores.arbol_lightgbm import min_data_per_group_por_tamano as f
    assert [f(n) for n in (319, 788, 934, 2414)] == [15, 39, 46, 100]   # los 3 afectados y sick (>= 2.000: la omisión)


# ---------------------------------------------------------------------------
# 2. LA REGLA, condición a condición
# ---------------------------------------------------------------------------
def test_una_mejora_clara_en_los_afectados_cumple():
    r = _veredicto(_payload(0.90, 0.95))
    assert r["veredicto"] == "CUMPLE", r["problemas"]
    assert r["regla"]["afectados"] == AF and r["regla"]["a"]["n"] == 3 and r["regla"]["b"]["n"] == 3


def test_sin_diferencia_el_control_es_igual_y_no_cumple_a_ni_b():
    r = _veredicto(_payload(0.90, 0.90))
    assert r["veredicto"] == "NO CUMPLE" and not r["regla"]["a"]["cumple"] and not r["regla"]["b"]["cumple"]


def test_a_exige_3_por_ciento_exacto_sin_redondear():
    justo = an.aplicar_la_regla(_ev([0.03, 0.03, 0.03], [True] * 3))
    assert justo["a"]["cumple"] and justo["cumple"]
    casi = an.aplicar_la_regla(_ev([0.03 - 1e-9, 0.03, 0.03], [True] * 3))
    assert not casi["a"]["cumple"] and not casi["cumple"]


def test_a_solo_mira_a_los_afectados_no_a_los_de_control():
    # los de control tienen R = 0; si entraran en la media, 0,06 sobre 3 se diluiría bajo 3 %
    g = an.aplicar_la_regla(_ev([0.06, 0.06, 0.06], [True] * 3))
    assert g["a"]["media_r_afectados"] == pytest.approx(0.06) and g["a"]["n"] == 3


def test_b_exige_dos_de_los_tres_afectados_con_mejora_clara():
    assert an.aplicar_la_regla(_ev([0.05] * 3, [True, True, False]))["b"]["cumple"]
    g = an.aplicar_la_regla(_ev([0.05] * 3, [True, False, False]))
    assert not g["b"]["cumple"] and not g["cumple"]


def test_c_un_conjunto_que_empeora_de_verdad_mata_el_veredicto():
    g = an.aplicar_la_regla(_ev([0.05] * 3, [True] * 3, empeora=True))
    assert g["c"]["empeoran"] == ["a0"] and not g["cumple"]
    r = _veredicto(_payload(0.90, 0.95, a_medida=lambda ds, b, f: (0.99 if b == R else 0.90) if ds == "wine_quality" else None))
    # wine_quality es de control: la palanca no actuó, así que que difiera es un CONTROL ROTO (sin veredicto), no un empeora
    assert r["veredicto"] == "SIN VEREDICTO" and any("control roto" in p for p in r["problemas"])


def test_d_el_coste_pasa_de_1_5_veces_el_de_la_referencia_no_cumple():
    ok = _veredicto(_payload(0.90, 0.95, wall_pal=15.0))
    assert ok["regla"]["d"]["razon_de_costes"] == pytest.approx(1.5) and ok["veredicto"] == "CUMPLE"
    mal = _veredicto(_payload(0.90, 0.95, wall_pal=15.1))
    assert not mal["regla"]["d"]["cumple"] and mal["veredicto"] == "NO CUMPLE"


def test_e_el_control_exige_lo_mismo_donde_la_palanca_no_actuo():
    rota = _veredicto(_payload(0.90, 0.95, a_medida=lambda ds, b, f: (0.90 + 1e-6) if (ds == "kc2" and b == P) else None))
    assert rota["veredicto"] == "SIN VEREDICTO" and any("kc2" in p and "control roto" in p for p in rota["problemas"])
    dentro = _veredicto(_payload(0.90, 0.95, a_medida=lambda ds, b, f: (0.90 + 0.001 * f + 1e-12) if (ds == "kc2" and b == P) else None))
    assert dentro["veredicto"] == "CUMPLE", dentro["problemas"]


def test_e_sick_con_categoricas_pero_mas_de_2000_filas_tambien_es_control():
    # sick declara min_data_per_group 100 aplicada: la palanca no cambió el valor -> es control
    r = _veredicto(_payload(0.90, 0.95, mdpg_de=lambda n, f: 15 if n in AF else (100 if n == "sick" else None)))
    assert r["veredicto"] == "CUMPLE", r["problemas"]
    rota = _veredicto(_payload(0.90, 0.95, a_medida=lambda ds, b, f: 0.5 if (ds == "sick" and b == P) else None,
                               mdpg_de=lambda n, f: 15 if n in AF else (100 if n == "sick" else None)))
    assert rota["veredicto"] == "SIN VEREDICTO" and any("sick" in p for p in rota["problemas"])


# ---------------------------------------------------------------------------
# 3. SIN VEREDICTO
# ---------------------------------------------------------------------------
def test_falta_un_pliegue_sin_veredicto():
    p = _payload()
    p["resultados"] = [r for r in p["resultados"] if not (r["dataset"] == "Moneyball" and r["pliegue"] == 2 and r["motor"] == P)]
    r = _veredicto(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any("Moneyball" in x for x in r["problemas"])


def test_falta_un_conjunto_sin_veredicto():
    p = _payload()
    p["conjuntos"] = [c for c in p["conjuntos"] if c["nombre"] != "pc1"]
    p["resultados"] = [r for r in p["resultados"] if r["dataset"] != "pc1"]
    r = _veredicto(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any("pc1" in x for x in r["problemas"])


def test_la_salida_parcial_o_un_intento_cortado_sin_veredicto():
    p = _payload()
    p["parcial"] = True
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    q = _payload()
    q["resultados"][0]["estado"] = "failed"
    assert _veredicto(q)["veredicto"] == "SIN VEREDICTO"


def test_registros_repetidos_sin_veredicto():
    p = _payload()
    p["resultados"].append(dict(p["resultados"][0]))
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_brazos_cruzados_sin_veredicto():
    p = _payload()
    for r in p["resultados"]:
        if r["motor"] == R and r["dataset"] == "dresses-sales":
            r["hiperparametros"]["min_data_per_group_por_tamano"] = {"pedida": True, "aplicada": True, "min_data_per_group": 15}
    v = _veredicto(p)
    assert v["veredicto"] == "SIN VEREDICTO" and any("brazos cruzados" in x for x in v["problemas"])
    q = _payload()
    for r in q["resultados"]:
        if r["motor"] == P and r["dataset"] == "kc2":
            del r["hiperparametros"]["min_data_per_group_por_tamano"]
    assert any("brazos cruzados" in x for x in _veredicto(q)["problemas"])


def test_un_afectado_donde_la_palanca_no_actuo_sin_veredicto():
    r = _veredicto(_payload(mdpg_de=lambda n, f: None if n == "Moneyball" else (15 if n in AF else None)))
    assert r["veredicto"] == "SIN VEREDICTO" and any("AFECTADO" in x and "Moneyball" in x for x in r["problemas"])


def test_un_no_afectado_donde_la_palanca_actuo_sin_veredicto():
    r = _veredicto(_payload(mdpg_de=lambda n, f: 15 if (n in AF or n == "kc2") else None))
    assert r["veredicto"] == "SIN VEREDICTO" and any("NO es de un afectado" in x and "kc2" in x for x in r["problemas"])


def test_codigo_mezclado_sin_veredicto():
    p = _payload()
    p["resultados"][0]["motor_digest"] = "otro"
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_un_conjunto_sellado_en_la_medida_sin_veredicto():
    p = _payload()
    p["conjuntos"][0]["sellado"] = True
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_la_sonda_no_decide_nada():
    p = _payload(es_sonda=True)
    p["conjuntos"], p["resultados"] = p["conjuntos"][:1], [r for r in p["resultados"] if r["dataset"] == p["conjuntos"][0]["nombre"]]
    assert _veredicto(p)["veredicto"] in ("SONDA", "SIN VEREDICTO") and _veredicto(p)["veredicto"] != "CUMPLE"


# ---------------------------------------------------------------------------
# 4. EL SELLO
# ---------------------------------------------------------------------------
def _con_sellos(monkeypatch, sellada):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", sellada)
    p = _payload()
    p["regla_sha256_al_medir"] = an._sha256(an.RUTA_DE_LA_REGLA)
    p["analizador_sha256_al_medir"] = an._sha256(an.RUTA_DEL_ANALIZADOR)
    p["dependencias_sha256_al_medir"] = an.dependencias_de_ahora()
    return p


def test_sin_sellar_la_regla_no_hay_veredicto(monkeypatch):
    p = _con_sellos(monkeypatch, None)
    r = an.analizar(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any("NO está sellada" in x for x in r["problemas"])


def test_con_la_regla_sellada_y_todo_igual_cumple(monkeypatch):
    p = _con_sellos(monkeypatch, an._sha256(an.RUTA_DE_LA_REGLA))
    for r in p["resultados"]:
        if r["dataset"] in AF and r["motor"] == P:
            r["auroc"] = 0.95
    assert an.analizar(p)["veredicto"] == "CUMPLE"


def test_una_regla_distinta_de_la_sellada_sin_veredicto(monkeypatch):
    p = _con_sellos(monkeypatch, "0" * 64)
    r = an.analizar(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any("no es la sellada" in x for x in r["problemas"])


def test_el_analizador_o_la_regla_cambiados_despues_de_medir_sin_veredicto(monkeypatch):
    p = _con_sellos(monkeypatch, an._sha256(an.RUTA_DE_LA_REGLA))
    p["analizador_sha256_al_medir"] = "a" * 64
    assert any("DESPUÉS de medir" in x for x in an.analizar(p)["problemas"])
    q = _con_sellos(monkeypatch, an._sha256(an.RUTA_DE_LA_REGLA))
    q["dependencias_sha256_al_medir"]["matrixai/estudio/metricas.py"] = "b" * 64
    assert any("metricas.py" in x for x in an.analizar(q)["problemas"])


# ---------------------------------------------------------------------------
# 5. LA CONFIRMACIÓN
# ---------------------------------------------------------------------------
def _confirmacion(afectados_n, mejora=0.95):
    nombres = ["tic-tac-toe", "cylinder-bands", "analcatdata_dmft", "MiceProtein", "solar_flare", "student_performance_por"]
    cj, rs = [], []
    for i, n in enumerate(nombres):
        cj.append({"nombre": n, "tarea": "binary_classification", "sellado": False, "metrica_de_cierre": "auroc",
                   "pliegues_esperados": [0, 1, 2, 3, 4]})
        for f in range(5):
            afecta = i < afectados_n
            rs.append(_registro(n, R, f, 0.90 + 0.001 * f))
            rs.append(_registro(n, P, f, (mejora if afecta else 0.90) + 0.001 * f, mdpg=20 if afecta else None))
    return {"es_sonda": False, "parcial": False, "conjuntos": cj, "resultados": rs}


def test_confirmacion_los_afectados_son_los_que_la_medida_declara_y_la_regla_se_escala():
    r = _veredicto(_confirmacion(4), confirmacion=True)
    assert r["veredicto"] == "CUMPLE", r["problemas"]
    assert r["regla"]["afectados"] == nombres_4() and r["regla"]["b"]["umbral"] == 2   # ceil(4/2)


def nombres_4():
    return ["tic-tac-toe", "cylinder-bands", "analcatdata_dmft", "MiceProtein"]


def test_confirmacion_con_menos_de_dos_afectados_es_sin_veredicto_no_cumple():
    r = _veredicto(_confirmacion(1), confirmacion=True)
    assert r["veredicto"] == "SIN VEREDICTO" and any("donde la palanca actúa" in x for x in r["problemas"])


def test_confirmacion_que_no_mejora_no_cumple():
    r = _veredicto(_confirmacion(3, mejora=0.90), confirmacion=True)
    assert r["veredicto"] == "NO CUMPLE"


def test_los_conjuntos_de_confirmacion_no_son_los_del_160_ni_los_del_162():
    texto = (_RAIZ / "benchmarks" / "c163" / "CONFIRMACION.md").read_text(encoding="utf-8")
    tabla = [l for l in texto.splitlines() if l.startswith("| ") and not l.startswith("| tarea") and not l.startswith("|---")]
    assert len(tabla) == 6
    nombres = {re.split(r"\s*\(", l.split("|")[3].strip())[0] for l in tabla}
    for previo in ("c160", "c162"):
        t = (_RAIZ / "benchmarks" / previo / "CONFIRMACION.md").read_text(encoding="utf-8")
        ids = {l.split("|")[2].strip() for l in t.splitlines() if l.startswith("| ") and l.split("|")[2].strip().isdigit()}
        assert not ids & {l.split("|")[2].strip() for l in tabla}
    assert not nombres & set(NOMBRES)


# ---------------------------------------------------------------------------
# 6. EL GUION DE MEDIDA
# ---------------------------------------------------------------------------
def test_los_dos_brazos_son_el_mismo_motor_con_la_palanca_apagada_y_encendida():
    m = med.motores()
    assert tuple(m) == an.BRAZOS
    assert m[R].min_data_per_group_por_tamano is False and m[P].min_data_per_group_por_tamano is True
    assert m[R].nombre == m[P].nombre == "lightgbm" and m[R].politica is None and m[P].politica is None


def test_el_digest_cubre_el_motor_lightgbm_y_la_politica():
    c = med.componentes_del_digest()
    assert any(k.endswith("arbol_lightgbm.py") for k in c) and any(k.endswith("politica_de_arboles.py") for k in c)
    assert any("analiza_c163_c1.py" in k for k in c)


def test_las_dependencias_del_guion_y_del_analizador_llevan_las_mismas_etiquetas():
    assert med.dependencias_sha256().keys() == an.dependencias_de_ahora().keys()


def test_la_declaracion_coherente_de_cada_brazo():
    assert med.declaracion_de_la_palanca_coherente(R, _registro("x", R, 0, 0.9)) == []
    assert med.declaracion_de_la_palanca_coherente(P, _registro("x", P, 0, 0.9, mdpg=15)) == []
    assert med.declaracion_de_la_palanca_coherente(P, _registro("x", R, 0, 0.9))   # la palanca sin declarar
    assert med.declaracion_de_la_palanca_coherente(R, _registro("x", P, 0, 0.9, mdpg=15))   # la referencia que la declara


def test_una_medida_real_se_niega_a_empezar_sin_sello(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    with pytest.raises(SystemExit, match="NO está sellada"):
        med.main(["--salida", str(tmp_path / "s.json")])
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    with pytest.raises(SystemExit, match="no es la sellada"):
        med.main(["--salida", str(tmp_path / "s.json")])
