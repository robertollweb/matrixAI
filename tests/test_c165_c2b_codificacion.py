# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2b — el banco de la CODIFICACIÓN DE CATEGÓRICAS (`benchmarks/c165/*_c2b.py`, auditoría 165 I4 e I5): las variantes, las siembras con
nombres neutros, la sensibilidad a renombrar, la regla sellada y su analizador. Nada de medida real: el analizador se prueba con salidas
FABRICADAS y las de «sin sello» fuerzan `REGLA_SHA256_SELLADA = None` con monkeypatch para no lanzar jamás una medida entera.
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

import analiza_c165_c2b as an  # noqa: E402
import banco_c165_c2b as b2  # noqa: E402
import medir_c165_c2b as med  # noqa: E402

REGLA = (_RAIZ / "benchmarks" / "c165" / "REGLA_C2B.md").read_text(encoding="utf-8")


def _con_categorias(n=300, semilla=1):
    rng = random.Random(semilla)
    filas = []
    for i in range(n):
        filas.append({"x": f"{rng.gauss(0, 1):.5f}", "y": f"{rng.gauss(5, 2):.5f}",
                      "ciudad": rng.choice(["Málaga", "Ávila", "Bilbao", "Zaragoza", "Cádiz"]) if i % 17 else "",
                      "tipo": rng.choice(["a", "b", "c"])})
    return filas


# ---------------------------------------------------------------- la codificación
def test_alfabetico_menos_uno_es_el_producto_y_el_control_lo_comprueba():
    assert b2.HOY == {"orden": "alfabetico", "vacia": "menos_uno"} and b2.VARIANTES[0] == b2.HOY and len(b2.VARIANTES) == 6
    assert b2.control_hoy_igual_al_producto(_con_categorias())


def test_los_codigos_de_cada_variante():
    txt = ["b", "a", None, "c", "a", "b", "a"]          # frecuencias a=3, b=2, c=1; «b» aparece antes que «a»
    cod = lambda **v: b2._codigos(list(txt), {**b2.HOY, **v}).tolist()  # noqa: E731
    assert cod() == [1.0, 0.0, -1.0, 2.0, 0.0, 1.0, 0.0]
    assert cod(orden="frecuencia") == [1.0, 0.0, -1.0, 2.0, 0.0, 1.0, 0.0]
    assert cod(vacia="nivel_propio") == [2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0]          # «» va el primero por el alfabeto
    assert cod(orden="frecuencia", vacia="nivel_propio") == [1.0, 0.0, 2.0, 3.0, 0.0, 1.0, 0.0]   # «» con 1 celda, aparece antes que «c»
    assert cod(vacia="centro") == [1.0, 0.0, 0.5, 2.0, 0.0, 1.0, 0.0]               # mediana de [1,0,2,0,1,0]


def test_frecuencia_no_depende_del_nombre_y_alfabetico_si():
    txt = ["b", "a", "c", "a", "b", "a", "d"]
    otro = [{"a": "zz", "b": "b", "c": "c", "d": "!d"}[t] for t in txt]       # renombrar mueve «a» al final y «d» al principio
    f = {**b2.HOY, "orden": "frecuencia"}
    assert b2._codigos(txt, f).tolist() == b2._codigos(otro, f).tolist()
    assert b2._codigos(txt, b2.HOY).tolist() != b2._codigos(otro, b2.HOY).tolist()


def test_a_igual_frecuencia_manda_la_primera_aparicion():
    assert b2._codigos(["z", "a", "z", "a"], {**b2.HOY, "orden": "frecuencia"}).tolist() == [0.0, 1.0, 0.0, 1.0]


# ---------------------------------------------------------------- siembras y renombres
def test_las_siembras_neutras_caen_en_medio_y_la_del_c2_al_final():
    filas = _con_categorias()
    pos = {t: b2.sembrar(filas, t, 0.05, 0, "x")[2]["posicion_media"] for t in b2.TIPOS_RARA}
    assert pos["rara_final"] > 0.8 and 0.3 <= pos["rara_medio"] <= 0.7
    # por punto de código, como `sorted` del producto: «Ávila» va DETRÁS de «Zaragoza» (el vicio que se mide), así que «Bilbao~…» es el 2.º de 6
    assert b2.posicion_alfabetica("Bilbao~00001", ["Ávila", "Bilbao", "Cádiz", "Málaga", "Zaragoza"]) == pytest.approx(1 / 5)


@pytest.mark.parametrize("tipo", b2.TIPOS_RARA)
def test_la_siembra_rara_es_determinista_sustituye_filas_y_pone_niveles_nuevos(tipo):
    filas = _con_categorias()
    a, sa, da = b2.sembrar(filas, tipo, 0.05, 1, "x")
    b, sb, _ = b2.sembrar(filas, tipo, 0.05, 1, "x")
    assert a == b and sa == sb and len(a) == len(filas) and len(sa) == 15
    c = da["columna"]
    nuevos = {a[i][c] for i in sa}
    assert len(nuevos) == len(sa) and not nuevos & {f[c] for f in filas}
    assert all(a[i] == filas[i] for i in range(len(filas)) if i not in sa)


def test_los_renombres_de_la_regla():
    filas = _con_categorias()
    r = b2.renombres(filas)
    assert {x["columna"] for x in r} == {"ciudad", "tipo"}
    assert {x["nuevo"][:1] for x in r} == {"!", "z"} and all(x["nuevo"].endswith(x["nivel"]) for x in r)
    ren = b2.renombrar(filas, r[0])
    assert sum(1 for f in ren if f[r[0]["columna"]] == r[0]["nuevo"]) == sum(1 for f in filas if f[r[0]["columna"]] == r[0]["nivel"])


def test_el_derivado_categoriza_tres_numericas_con_nombres_barajados_y_vacias():
    rng = random.Random(3)
    filas = [{f"n{j}": f"{rng.gauss(j, 1):.4f}" for j in range(6)} for _ in range(400)]
    nuevas, det = b2.derivar("prueba", filas)
    assert sorted(det["columnas"]) == ["n0", "n3", "n5"]
    for c, d in det["columnas"].items():
        vals = [f[c] for f in nuevas]
        assert vals.count("") == 12 and set(vals) - {""} <= set(d["niveles"]) and len(set(vals) - {""}) == 6
    assert all(nuevas[i]["n1"] == filas[i]["n1"] for i in range(400))


def test_jaccard_e_inflacion():
    assert b2.jaccard([1, 2, 3], [1, 2, 3]) == 1.0 and b2.jaccard([1, 2], [3, 4]) == 0.0
    filas = _con_categorias(300)
    con_vacia = [i for i, f in enumerate(filas) if f["ciudad"] == ""]
    orden = con_vacia + [i for i in range(300) if i not in con_vacia]
    assert b2.inflacion_por_la_vacia(filas, orden) == pytest.approx(1 / (len(con_vacia) / 300))


# ---------------------------------------------------------------- el sello
def test_la_regla_esta_sellada_con_su_sha256_vigente_y_no_toca_la_del_c2():
    assert an.REGLA_SHA256_SELLADA == an._sha256(an.RUTA_DE_LA_REGLA)
    assert re.fullmatch(r"[0-9a-f]{64}", an.REGLA_SHA256_SELLADA)
    import analiza_c165_c2 as an_c2
    assert an_c2.REGLA_SHA256_SELLADA == "bab0a2c71b634090a2dd3cb732eea0a5f422461b824f5dbdb291a464a0c17855"
    for n in b2.NOMBRES:
        assert f"`{n}`" in REGLA


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


# ---------------------------------------------------------------- el analizador con salidas FABRICADAS
def _payload(M=None, S=None, I=None, *, es_sonda=False, pos=0.5):
    """Salida completa. `M(dataset, tipo, tasa, variante) -> recuperación`; `S(variante)`; `I(variante)`. Por omisión nada gana."""
    M = M or (lambda d, t, ta, v: 0.5)
    S = S or (lambda v: 0.6)
    I = I or (lambda v: 2.0)
    conjuntos, res, ren, sem, inf = [], [], [], [], []
    for n in b2.NOMBRES:
        filas = 1000
        conjuntos.append({"nombre": n, "filas": filas, "categoricas": ["c"], "control_hoy_igual_al_producto": True})
        for v in b2.VARIANTES:
            nv = b2.nombre_de_variante(v)
            ren.append({"dataset": n, "variante": nv, "columna": "c", "nivel": "x", "nuevo": "!x", "jaccard": S(nv)})
            sem += [{"dataset": n, "variante": nv, "semilla": s, "jaccard": 0.7} for s in (1, 2, 3, 4)]
            inf.append({"dataset": n, "variante": nv, "inflacion": I(nv)})
            for t in b2.TIPOS:
                for ta in b2.TASAS:
                    ns = b2.banco.n_sembradas(filas, ta)
                    for r in b2.REPETICIONES:
                        res.append({"dataset": n, "tipo": t, "tasa": ta, "repeticion": r, "variante": nv, "sembradas": list(range(ns)),
                                    "detalle_siembra": {"posicion_media": pos},
                                    "metricas": {"n_sembradas": ns, "recuperacion_doble": M(n, t, ta, nv)}})
    return {"es_sonda": es_sonda, "parcial": False, "conjuntos": conjuntos, "resultados": res, "renombres": ren, "semillas": sem,
            "inflacion": inf, "tasas": list(b2.TASAS), "repeticiones": list(b2.REPETICIONES),
            "regla_sha256_al_medir": an._sha256(an.RUTA_DE_LA_REGLA), "analizador_sha256_al_medir": an._sha256(an.RUTA_DEL_ANALIZADOR),
            "dependencias_sha256_al_medir": an.dependencias_de_ahora()}


def _eje(r, nombre):
    return next(e for e in r["ejes"] if e["eje"] == nombre)


def test_sin_diferencias_se_queda_hoy():
    r = an.analizar(_payload())
    assert r["veredicto"] == "DECIDIDO" and r["codificacion_elegida"] == b2.HOY, r["problemas"]
    assert {x["tasa"] for x in r["para_el_texto_de_limites"]} == set(b2.TASAS)


def test_frecuencia_gana_por_robustez_a_renombrar_sin_perder_recuperacion():
    r = an.analizar(_payload(S=lambda v: 1.0 if v.startswith("frecuencia") else 0.6))
    assert _eje(r, "orden")["elegido"] == "frecuencia" and r["codificacion_elegida"]["orden"] == "frecuencia"


def test_c_exige_el_umbral_de_s_exacto():
    r = an.analizar(_payload(S=lambda v: 0.6 + an.UMBRAL_S - 1e-9 if v.startswith("frecuencia") else 0.6))
    assert _eje(r, "orden")["elegido"] == "alfabetico"


def test_frecuencia_robusta_pero_que_pierde_recuperacion_no_entra():
    pierde = lambda d, t, ta, v: 0.5 - (0.06 if v.startswith("frecuencia") and t == "rara_medio" else 0)  # noqa: E731
    r = an.analizar(_payload(M=pierde, S=lambda v: 1.0 if v.startswith("frecuencia") else 0.6))
    c = _eje(r, "orden")["candidatos"][0]
    assert not c["b"] and _eje(r, "orden")["elegido"] == "alfabetico"


def test_rara_final_no_decide():
    """La siembra viciada del C2 se informa pero no cuenta: un candidato que solo gana en `rara_final` no gana."""
    gana_en_final = lambda d, t, ta, v: 0.5 + (0.5 if v.startswith("frecuencia") and t == "rara_final" else 0)  # noqa: E731
    r = an.analizar(_payload(M=gana_en_final))
    assert _eje(r, "orden")["candidatos"][0]["media_delta"] == pytest.approx(0.0) and r["codificacion_elegida"] == b2.HOY


def test_la_vacia_cambia_si_deja_de_inflar_y_gana_la_de_menor_d():
    I = {"alfabetico|menos_uno": 3.0, "alfabetico|nivel_propio": 1.3, "alfabetico|centro": 1.1}
    r = an.analizar(_payload(I=lambda v: I.get(v, 3.0)))
    assert _eje(r, "vacia")["elegido"] == "centro"
    I["alfabetico|centro"] = 2.6                        # D baja 0,4 < 0,5: no
    r = an.analizar(_payload(I=lambda v: I.get(v, 3.0)))
    assert _eje(r, "vacia")["elegido"] == "nivel_propio"


def test_una_combinacion_peor_que_hoy_por_interaccion_no_se_adopta():
    S = lambda v: 1.0 if v.startswith("frecuencia") else 0.6   # noqa: E731
    I = lambda v: 1.0 if v.endswith("centro") else 3.0           # noqa: E731
    M = lambda d, t, ta, v: 0.5 - (0.2 if v == "frecuencia|centro" else 0)   # noqa: E731
    r = an.analizar(_payload(M=M, S=S, I=I))
    assert r["interaccion"]["peor_que_hoy"] and r["codificacion_elegida"] == b2.HOY


def test_el_texto_de_limites_es_la_media_de_las_dos_neutras_con_la_combinacion_elegida():
    M = lambda d, t, ta, v: {"rara_medio": 0.2, "rara_azar": 0.4, "rara_final": 0.9}.get(t, 0.5)   # noqa: E731
    r = an.analizar(_payload(M=M))
    L = {x["tasa"]: x for x in r["para_el_texto_de_limites"]}
    assert L[0.05]["rara_neutra_media"] == pytest.approx(0.3) and L[0.05]["rara_final_media_solo_informe"] == pytest.approx(0.9)
    assert "no son rendimiento de producto" in r["aviso"]


@pytest.mark.parametrize("estropear, texto", [
    (lambda p: p["resultados"].pop(), "faltan"),
    (lambda p: p["resultados"].append(dict(p["resultados"][0])), "repetidos"),
    (lambda p: p["conjuntos"].pop(), "conjuntos no son"),
    (lambda p: p.update(parcial=True), "PARCIAL"),
    (lambda p: p["conjuntos"][2].update(control_hoy_igual_al_producto=False), "NO reproduce"),
    (lambda p: p["conjuntos"][0].update(categoricas=[]), "sin categóricas"),
    (lambda p: p.update(renombres=[r for r in p["renombres"] if r["dataset"] != "pc1"]), "renombres"),
    (lambda p: p.update(semillas=p["semillas"][1:]), "semillas"),
    (lambda p: p.update(inflacion=p["inflacion"][1:]), "inflación"),
    (lambda p: p["resultados"][0].update(sembradas=[1, 2]), "sembradas"),
    (lambda p: p.update(tasas=[0.05]), "tasas"),
    (lambda p: p["dependencias_sha256_al_medir"].update({"benchmarks/c165/banco_c165_c2b.py": "0" * 64}), "al medir"),
    (lambda p: p.update(analizador_sha256_al_medir="0" * 64), "DESPUÉS de medir"),
    (lambda p: p.pop("regla_sha256_al_medir"), "regla_sha256_al_medir"),
])
def test_completitud_y_sellos_sin_veredicto(estropear, texto):
    p = _payload()
    estropear(p)
    r = an.analizar(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any(texto in x for x in r["problemas"]), r["problemas"]
    assert r["codificacion_elegida"] is None and r["para_el_texto_de_limites"] == []


def test_una_siembra_que_no_es_neutra_no_da_veredicto():
    r = an.analizar(_payload(pos=0.9))
    assert r["veredicto"] == "SIN VEREDICTO" and any("no es neutra" in x for x in r["problemas"])


def test_sin_sellar_o_con_otra_regla_no_hay_veredicto(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    assert any("NO está sellada" in x for x in an.analizar(_payload())["problemas"])
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    assert any("no es la sellada" in x for x in an.analizar(_payload())["problemas"])


def test_la_sonda_no_decide_nada():
    p = _payload(es_sonda=True)
    p["conjuntos"] = p["conjuntos"][:1]
    n = p["conjuntos"][0]["nombre"]
    for k in ("resultados", "renombres", "semillas", "inflacion"):
        p[k] = [r for r in p[k] if r["dataset"] == n]
    r = an.analizar(p)
    assert r["veredicto"] == "SONDA" and r["codificacion_elegida"] is None and r["para_el_texto_de_limites"] == [], r["problemas"]


# ---------------------------------------------------------------- el guion de punta a punta, en pequeño
def test_medir_en_pequeno_graba_todo_y_el_analizador_lo_lee(tmp_path, monkeypatch):
    filas = _con_categorias(240)
    monkeypatch.setattr(b2, "cargar", lambda nombre: (filas, {"derivado": False}))
    monkeypatch.setattr(b2, "VARIANTES", [b2.HOY, {"orden": "frecuencia", "vacia": "centro"}])
    p = med.medir(["dresses-sales"], (0,), (0.05,), tmp_path / "s.json", True, escribir=None)
    assert p["parcial"] is False and p["conjuntos"][0]["control_hoy_igual_al_producto"] is True
    assert {r["tipo"] for r in p["resultados"]} == set(b2.TIPOS) and len(p["resultados"]) == 8
    assert len(p["semillas"]) == 8 and len(p["inflacion"]) == 2 and p["renombres"]
    for r in p["resultados"]:
        assert len(r["sembradas"]) == r["metricas"]["n_sembradas"] == 12
    assert all(0 <= r["jaccard"] <= 1 for r in p["renombres"] + p["semillas"])
    # con la codificación por frecuencia, renombrar NO mueve nada (es la propiedad que se mide)
    assert all(r["jaccard"] == 1.0 for r in p["renombres"] if r["variante"] == "frecuencia|centro")
