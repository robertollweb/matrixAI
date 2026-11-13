# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2 — el banco interno de «Filas a revisar» (`benchmarks/c165/`): el generador de anomalías sembradas, las métricas de recuperación, la regla
sellada y su analizador. Nada de medida real: el analizador se prueba con salidas FABRICADAS y las de «sin sello» fuerzan
`REGLA_SHA256_SELLADA = None` con monkeypatch para no lanzar jamás una medida entera.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "c165"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import analiza_c165_c2 as an  # noqa: E402
import banco_c165 as banco  # noqa: E402
import medir_c165_c2 as med  # noqa: E402

REGLA = (_RAIZ / "benchmarks" / "c165" / "REGLA_C2.md").read_text(encoding="utf-8")


def _sintetico(n=400, semilla=1):
    rng = random.Random(semilla)
    filas = []
    for _ in range(n):
        a = rng.gauss(0, 1)
        filas.append({"a": f"{a:.5f}", "b": f"{a * 0.9 + rng.gauss(0, 0.3):.5f}", "c": f"{rng.gauss(5, 2):.5f}",
                      "d": f"{rng.uniform(0, 100):.4f}", "cat": rng.choice(["x", "y", "z"]), "cat2": rng.choice(["p", "q"])})
    return filas


# ---------------------------------------------------------------- generador
@pytest.mark.parametrize("tipo", banco.TIPOS)
def test_la_siembra_es_determinista_sustituye_filas_y_graba_cuales(tipo):
    f = _sintetico()
    n1, s1, d1 = banco.sembrar(f, tipo, 0.05, 0, "x")
    n2, s2, d2 = banco.sembrar(f, tipo, 0.05, 0, "x")
    assert (n1, s1, d1) == (n2, s2, d2)
    assert len(n1) == len(f) and len(s1) == banco.n_sembradas(len(f), 0.05) == 20 and s1 == sorted(set(s1))
    assert banco.sembrar(f, tipo, 0.05, 1, "x")[1] != s1, "otra repetición, otras filas"
    cambiadas = [i for i in range(len(f)) if n1[i] != f[i]]
    assert cambiadas == s1, "solo cambian las sembradas, y todas"
    assert f == _sintetico(), "no se toca el original"


def test_extremo_toca_una_columna_por_fila_y_la_saca_del_rango():
    f = _sintetico()
    n, s, d = banco.sembrar(f, "extremo", 0.05, 0, "x")
    nums = banco._numericas(f)
    for i, c in zip(s, d["columnas"]):
        assert sum(n[i][k] != f[i][k] for k in f[i]) == 1 and n[i][c] != f[i][c]
        assert float(n[i][c]) > nums[c].max()


def test_combinacion_cada_columna_normal_y_la_pareja_imposible():
    f = _sintetico()
    n, s, d = banco.sembrar(f, "combinacion", 0.05, 0, "x")
    a, b = d["pareja"]
    assert {a, b} == {"a", "b"} and d["r"] > 0.5
    xa, xb = banco._numericas(f)[a], banco._numericas(f)[b]
    for i in s:
        va, vb = float(n[i][a]), float(n[i][b])
        assert np.percentile(xa, 10) <= va <= np.percentile(xa, 95) and np.percentile(xb, 5) <= vb <= np.percentile(xb, 90)
        assert va > np.percentile(xa, 80) and vb < np.percentile(xb, 20), "esquina que la correlación positiva prohíbe"


def test_rara_pone_un_valor_nuevo_distinto_por_fila():
    f = _sintetico()
    n, s, d = banco.sembrar(f, "rara", 0.05, 0, "x")
    c = d["columna"]
    nuevos = [n[i][c] for i in s]
    assert len(set(nuevos)) == len(s) and not set(nuevos) & {r[c] for r in f}


def test_aplicabilidad_declara_el_motivo_cuando_un_tipo_no_existe():
    sin_cat = [{k: v for k, v in r.items() if not k.startswith("cat")} for r in _sintetico()]
    ap = banco.aplicabilidad(sin_cat)
    assert ap["rara"] and ap["extremo"] is None and ap["combinacion"] is None
    solo_cat = [{k: v for k, v in r.items() if k.startswith("cat")} for r in _sintetico()]
    assert banco.aplicabilidad(solo_cat)["extremo"]


# ---------------------------------------------------------------- métricas y detector
def test_metricas_orden_perfecto_y_orden_al_reves():
    s = [0, 1, 2, 3]
    perfecto = banco.metricas(list(range(100)), s)
    assert perfecto["recuperacion_uno"] == 1.0 and perfecto["precision_uno"] == 1.0 and perfecto["ap"] == 1.0
    reves = banco.metricas(list(range(99, -1, -1)), s)
    assert reves["recuperacion_doble"] == 0.0 and reves["ap"] < 0.1
    medio = banco.metricas(list(range(100)), [0, 50, 51, 52])
    assert medio["recuperacion_doble"] == 0.25 and medio["k_doble"] == 8 and medio["k_medio"] == 2


def test_el_instrumento_recupera_lo_obvio_y_no_el_azar():
    """Control conocido-bueno: valores extremos y categorías nuevas se recuperan casi enteros; el azar no."""
    f = _sintetico(600)
    for tipo in ("extremo", "rara"):
        n, s, _ = banco.sembrar(f, tipo, 0.05, 0, "x")
        m = banco.metricas(banco.ordenar(n, banco.HOY), s)
        assert m["recuperacion_doble"] > 0.9, (tipo, m)
    azar = banco.metricas(np.random.default_rng(0).permutation(len(n)), s)
    assert azar["recuperacion_doble"] < 0.4
    # la combinación imposible (cada columna normal) es lo difícil para ejes alineados: con cuantiles se recupera MÁS que sin escalar
    c, sc, _ = banco.sembrar(f, "combinacion", 0.05, 0, "x")
    sin = banco.metricas(banco.ordenar(c, banco.HOY), sc)["recuperacion_doble"]
    con = banco.metricas(banco.ordenar(c, {**banco.HOY, "escalado": "cuantiles"}), sc)["recuperacion_doble"]
    assert con > sin + 0.2, (sin, con)
    azar = banco.metricas(np.random.default_rng(0).permutation(len(n)), s)
    assert azar["recuperacion_doble"] < 0.4


def test_las_12_variantes_son_la_rejilla_y_hoy_es_la_primera_de_cada_eje():
    assert len(banco.VARIANTES) == 12 and len({banco.nombre_de_variante(v) for v in banco.VARIANTES}) == 12
    assert banco.HOY == {"escalado": "ninguno", "categoricas": "frecuencia", "max_samples": "auto"}
    f = _sintetico(300)
    for v in banco.VARIANTES:
        o = banco.ordenar(f, v)
        assert sorted(o.tolist()) == list(range(300))


def test_la_variante_hoy_reproduce_el_producto_y_contamination_no_cambia_el_orden():
    """Desde el 13-11 el producto lleva la variante elegida por este banco, no la `HOY` de referencia."""
    assert banco.PRODUCTO_VARIANTE == {"escalado": "ninguno", "categoricas": "ordinal", "max_samples": "todas"}
    f = _sintetico(300)
    assert banco.control_hoy_igual_al_producto(f)
    assert banco.control_contamination_invariante(f)


def test_el_control_de_hoy_se_pone_rojo_si_la_variante_hoy_no_es_la_del_producto(monkeypatch):
    f = _sintetico(300)
    monkeypatch.setattr(banco, "PRODUCTO_VARIANTE", {"escalado": "cuantiles", "categoricas": "onehot", "max_samples": "todas"})
    assert not banco.control_hoy_igual_al_producto(f)


# ---------------------------------------------------------------- la regla y el sello
def test_los_umbrales_del_codigo_son_los_de_la_regla_escrita():
    assert "+0,03" in REGLA and an.UMBRAL_DELTA == 0.03
    assert "+0,05" in REGLA and an.UMBRAL_DELTA_CATEGORICAS == 0.05
    assert "60 %" in REGLA and an.FRACCION_NO_PEOR == 0.60
    assert "−0,03" in REGLA and an.CAIDA_MAXIMA_POR_TIPO == -0.03 and an.AVISO_DE_INTERACCION == 0.03
    assert "recuperación@(2·n_s)" in REGLA and an.METRICA == "recuperacion_doble"
    assert "1 % y 5 %" in REGLA and banco.TASAS == (0.01, 0.05) and banco.REPETICIONES == (0, 1, 2) and banco.MIN_SEMBRADAS == 3
    assert "0,5 × rango" in REGLA and "|r| de Pearson ≥ 0,5" in REGLA and banco.CORR_MIN == 0.5
    for n, _, _ in banco.CONJUNTOS:
        assert f"`{n}`" in REGLA


def test_la_regla_esta_sellada_con_su_sha256_vigente():
    assert an.REGLA_SHA256_SELLADA == an._sha256(an.RUTA_DE_LA_REGLA)
    assert re.fullmatch(r"[0-9a-f]{64}", an.REGLA_SHA256_SELLADA)


def test_el_guion_se_niega_a_medir_sin_sello_y_con_otra_regla(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    monkeypatch.setattr(med, "medir", lambda *a, **k: pytest.fail("no debe medir"))
    with pytest.raises(SystemExit, match="NO está sellada"):
        med.main(["--salida", "/nonexistent/x.json"])
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    with pytest.raises(SystemExit, match="no es la sellada"):
        med.main(["--salida", "/nonexistent/x.json"])


def test_la_sonda_no_necesita_sello(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    llamado = []
    monkeypatch.setattr(med, "medir", lambda *a, **k: llamado.append(a))
    med.main(["--sonda", "--salida", "/nonexistent/x.json"])
    assert llamado and llamado[0][0] == list(med.CONJUNTOS_DE_LA_SONDA)


# ---------------------------------------------------------------- analizador con salidas FABRICADAS
APLICABLE = {"breast-cancer-wisconsin-diagnostic": (None, None, "x"), "cardiotocography": (None, None, "x"),
             "climate-model-simulation-crashes": (None, "x", "x"), "dresses-sales": (None, "x", None), "pc1": (None, None, "x"),
             "us_crime": (None, None, "x")}


def _payload(valor=None, *, es_sonda=False):
    """Salida completa: `valor(dataset, tipo, tasa, variante_dict) -> M`; por omisión todo 0,5 (ninguna variante gana)."""
    valor = valor or (lambda d, t, ta, v: 0.5)
    conjuntos, res = [], []
    for nombre, (e, c, r) in APLICABLE.items():
        n = {"breast-cancer-wisconsin-diagnostic": 569}.get(nombre, 1000)
        ap = {"extremo": e, "combinacion": c, "rara": r}
        conjuntos.append({"nombre": nombre, "filas": n, "aplicabilidad": ap, "control_hoy_igual_al_producto": True,
                          "control_contamination_invariante": True})
        for tipo, motivo in ap.items():
            if motivo is not None:
                continue
            for tasa in banco.TASAS:
                ns = banco.n_sembradas(n, tasa)
                for rep in banco.REPETICIONES:
                    for v in banco.VARIANTES:
                        res.append({"dataset": nombre, "tipo": tipo, "tasa": tasa, "repeticion": rep, "variante": banco.nombre_de_variante(v),
                                    "sembradas": list(range(ns)), "metricas": {"n_sembradas": ns, "recuperacion_doble": valor(nombre, tipo, tasa, v)}})
    return {"es_sonda": es_sonda, "parcial": False, "conjuntos": conjuntos, "resultados": res,
            "regla_sha256_al_medir": an._sha256(an.RUTA_DE_LA_REGLA), "analizador_sha256_al_medir": an._sha256(an.RUTA_DEL_ANALIZADOR),
            "dependencias_sha256_al_medir": an.dependencias_de_ahora()}


def _eje(r, nombre):
    return next(e for e in r["ejes"] if e["eje"] == nombre)


def test_sin_diferencias_se_queda_el_defecto_de_hoy():
    r = an.analizar(_payload())
    assert r["veredicto"] == "DECIDIDO", r["problemas"]
    assert r["defectos_elegidos"] == {**banco.HOY, "contamination": None}
    assert not any(e["cambia"] for e in r["ejes"])


def test_un_candidato_que_gana_claramente_cambia_el_defecto():
    r = an.analizar(_payload(lambda d, t, ta, v: 0.5 + (0.1 if v["escalado"] == "cuantiles" else 0)))
    assert r["defectos_elegidos"]["escalado"] == "cuantiles" and _eje(r, "escalado")["cambia"]
    assert r["defectos_elegidos"]["max_samples"] == "auto"


def test_a_exige_el_umbral_exacto_sin_redondear():
    justo = lambda d, t, ta, v: 0.5 + (0.03 if v["max_samples"] == "todas" else 0)  # noqa: E731
    casi = lambda d, t, ta, v: 0.5 + (0.0299999 if v["max_samples"] == "todas" else 0)  # noqa: E731
    assert an.analizar(_payload(justo))["defectos_elegidos"]["max_samples"] == "todas"
    assert an.analizar(_payload(casi))["defectos_elegidos"]["max_samples"] == "auto"


def test_categoricas_exige_mas_y_solo_cuenta_los_conjuntos_con_categoricas():
    f = lambda gana: (lambda d, t, ta, v: 0.5 + (gana if v["categoricas"] == "ordinal" and d == "dresses-sales" else 0))  # noqa: E731
    assert an.analizar(_payload(f(0.04)))["defectos_elegidos"]["categoricas"] == "frecuencia"
    r = an.analizar(_payload(f(0.05)))
    assert r["defectos_elegidos"]["categoricas"] == "ordinal"
    assert _eje(r, "categoricas")["candidatos"][0]["n_celdas"] == 4 or _eje(r, "categoricas")["candidatos"][1]["n_celdas"] == 4


def test_b_si_gana_en_pocas_celdas_aunque_la_media_pase_no_cambia():
    # +0,5 en 1 de cada 10 celdas, -0,0001 en el resto: media alta, pero < 60 % de celdas no peores
    def f(d, t, ta, v):
        if v["escalado"] != "cuantiles":
            return 0.2
        return 0.9 if (d, t, ta) == ("pc1", "extremo", 0.05) or (d, t, ta) == ("us_crime", "extremo", 0.05) else 0.1999
    r = an.analizar(_payload(f))
    e = _eje(r, "escalado")["candidatos"][0]
    assert e["a"] and not e["b"] and r["defectos_elegidos"]["escalado"] == "ninguno"


def test_c_el_que_ayuda_a_un_tipo_pero_rompe_otro_no_vale():
    def f(d, t, ta, v):
        if v["escalado"] != "cuantiles":
            return 0.5
        return 0.5 + (0.2 if t == "extremo" else -0.05 if t == "combinacion" else 0.0)
    r = an.analizar(_payload(f))
    e = _eje(r, "escalado")["candidatos"][0]
    assert e["a"] and e["b"] and not e["c"] and r["defectos_elegidos"]["escalado"] == "ninguno"


def test_entre_varios_candidatos_gana_el_de_mayor_delta():
    r = an.analizar(_payload(lambda d, t, ta, v: 0.5 + {"frecuencia": 0, "ordinal": 0.1, "onehot": 0.2}[v["categoricas"]]
                             * (d == "dresses-sales")))
    assert r["defectos_elegidos"]["categoricas"] == "onehot"


def test_una_combinacion_elegida_peor_que_hoy_por_interaccion_no_se_adopta():
    def f(d, t, ta, v):
        if v["escalado"] == "cuantiles" and v["max_samples"] == "todas":
            return 0.2
        return 0.5 + (0.1 if v["escalado"] == "cuantiles" and v["max_samples"] == "auto" else 0) + \
            (0.1 if v["max_samples"] == "todas" and v["escalado"] == "ninguno" else 0)
    r = an.analizar(_payload(f))
    assert r["interaccion"]["peor_que_hoy"] and r["defectos_elegidos"] == {**banco.HOY, "contamination": None}


def test_el_texto_de_limites_sale_de_la_combinacion_elegida_por_tipo_y_tasa():
    r = an.analizar(_payload(lambda d, t, ta, v: 0.4 if t == "extremo" else 0.6))
    L = {(x["tipo"], x["tasa"]): x["recuperacion_doble_media"] for x in r["para_el_texto_de_limites"]}
    assert len(L) == 6 and L[("extremo", 0.01)] == pytest.approx(0.4) and L[("rara", 0.05)] == pytest.approx(0.6)
    assert "no son rendimiento de producto" in r["aviso"]


@pytest.mark.parametrize("estropear, texto", [
    (lambda p: p["resultados"].pop(), "faltan"),
    (lambda p: p["resultados"].append(dict(p["resultados"][0])), "repetidos"),
    (lambda p: p["conjuntos"].pop(), "conjuntos no son"),
    (lambda p: p.update(parcial=True), "PARCIAL"),
    (lambda p: p["conjuntos"][0].update(control_hoy_igual_al_producto=False), "NO reproduce"),
    (lambda p: p["conjuntos"][0].update(control_contamination_invariante=False), "contamination"),
    (lambda p: p["conjuntos"][0]["aplicabilidad"].update(extremo="no hay"), "extremo"),
    (lambda p: p["conjuntos"][3]["aplicabilidad"].update(rara="no hay"), "dresses-sales"),
    (lambda p: p["resultados"][0].update(sembradas=[1, 2]), "sembradas"),
    (lambda p: p["resultados"][0].update(sembradas=[1] * 28, metricas={"n_sembradas": 28, "recuperacion_doble": 1}), "sembradas"),
    (lambda p: p["resultados"][0].update(metricas=None), "recuperacion_doble"),
    (lambda p: p["dependencias_sha256_al_medir"].update({"matrixai_engines/anomalias.py": "0" * 64}), "al medir"),
    (lambda p: p.update(analizador_sha256_al_medir="0" * 64), "DESPUÉS de medir"),
    (lambda p: p.pop("regla_sha256_al_medir"), "regla_sha256_al_medir"),
])
def test_completitud_y_sellos_sin_veredicto(estropear, texto):
    p = _payload()
    estropear(p)
    r = an.analizar(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any(texto in x for x in r["problemas"]), r["problemas"]
    assert r["defectos_elegidos"] is None and r["para_el_texto_de_limites"] == []


def test_sin_sellar_o_con_otra_regla_no_hay_veredicto(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    assert any("NO está sellada" in x for x in an.analizar(_payload())["problemas"])
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    assert any("no es la sellada" in x for x in an.analizar(_payload())["problemas"])


def test_la_sonda_no_decide_nada():
    p = _payload(es_sonda=True)
    p["conjuntos"] = p["conjuntos"][:1]
    p["resultados"] = [r for r in p["resultados"] if r["dataset"] == p["conjuntos"][0]["nombre"]]
    r = an.analizar(p)
    assert r["veredicto"] == "SONDA" and r["defectos_elegidos"] is None and r["para_el_texto_de_limites"] == []


def test_una_celda_a_medias_sin_veredicto_tambien_en_la_sonda():
    p = _payload(es_sonda=True)
    p["resultados"] = p["resultados"][:5]
    assert an.analizar(p)["veredicto"] == "SIN VEREDICTO"


# ---------------------------------------------------------------- el guion de punta a punta, en pequeño
def test_medir_en_pequeno_graba_lo_sembrado_y_el_analizador_lo_lee(tmp_path, monkeypatch):
    monkeypatch.setattr(banco, "TASAS", (0.05,))
    monkeypatch.setattr(an.banco, "TASAS", (0.05,))
    monkeypatch.setattr(banco, "VARIANTES", banco.VARIANTES[:2])
    p = med.medir(["dresses-sales"], (0,), tmp_path / "s.json", True, escribir=None)
    assert p["parcial"] is False and p["conjuntos"][0]["control_hoy_igual_al_producto"] and p["conjuntos"][0]["control_contamination_invariante"]
    assert {r["tipo"] for r in p["resultados"]} == {"extremo", "rara"} and len(p["resultados"]) == 4
    for r in p["resultados"]:
        assert len(r["sembradas"]) == r["metricas"]["n_sembradas"] == 25
