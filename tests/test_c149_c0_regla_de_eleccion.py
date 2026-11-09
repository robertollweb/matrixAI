# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C0 — la REGLA de elección del guion de medición, con celdas escritas a mano.

El guion (`benchmarks/c149/medir_c0.py`) implementa lo que dice `REGLA_DE_ELECCION.md`; aquí se prueba solo esa
lógica —sin entrenar nada— contra números que se cuentan en el comentario. Lo que NO prueba: que el instrumento
mida bien (eso lo dice el sabotaje de la etiqueta invertida dentro de la propia medición).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_ESPEC = importlib.util.spec_from_file_location("medir_c0_149", RAIZ / "benchmarks" / "c149" / "medir_c0.py")
medir = importlib.util.module_from_spec(_ESPEC)
sys.modules["medir_c0_149"] = medir
_ESPEC.loader.exec_module(medir)

CFG = {"G": (15, 25), "m": (100, 200)}
GRUPOS = ["a|reg", "b|bin"]


def celda(mon_marcados, nomon_marcados, n=10):
    """Una celda con `n` casos monótonos y `n` no monótonos."""
    return {"monotono": [n, mon_marcados, 0], "no_monotono": [n, nomon_marcados, 0]}


def construir(f):
    """`f(G, m, eps) -> (monótonos marcados, no monótonos marcados)` aplicado a los dos grupos."""
    return {(g, G, m, e): celda(*f(G, m, e)) for g in GRUPOS for G in CFG["G"] for m in CFG["m"] for e in medir.EPSILONES}


def test_el_menor_epsilon_que_cumple_en_la_primera_celda_estable():
    # Con ε >= 1 % : 1 falsa alarma de 10 (10 %) y 8 detecciones de 10 (80 %) en TODAS las celdas -> cumple.
    celdas = construir(lambda G, m, e: (1, 8) if e >= 0.01 else (9, 10))
    r = medir.aplicar_regla(celdas, GRUPOS, CFG)
    assert r["hay_epsilon"] and r["epsilon"] == 0.01 and (r["G"], r["m"]) == (15, 100)   # coste G×m mínimo


def test_los_umbrales_son_inclusivos():
    """FA exactamente 10 % y POT exactamente 80 % cumplen; 11 % o 79 % no."""
    assert medir.cumple(medir.tasas(construir(lambda G, m, e: (1, 8)), GRUPOS, 15, 100, 0.01))
    assert not medir.cumple(medir.tasas(construir(lambda G, m, e: (2, 8)), GRUPOS, 15, 100, 0.01))      # FA 20 %
    assert not medir.cumple(medir.tasas(construir(lambda G, m, e: (1, 7)), GRUPOS, 15, 100, 0.01))      # POT 70 %


def test_un_grupo_que_falla_hace_fallar_la_celda():
    celdas = construir(lambda G, m, e: (1, 8))
    celdas[("b|bin", 15, 100, 0.01)] = celda(5, 8)                       # solo un grupo, 50 % de falsas alarmas
    assert not medir.cumple(medir.tasas(celdas, GRUPOS, 15, 100, 0.01))
    assert medir.cumple(medir.tasas(celdas, GRUPOS, 15, 100, 0.02))


def test_la_estabilidad_descarta_una_celda_barata_que_se_estropea_al_crecer():
    # (15,100) cumpliría con ε = 0,5 %, pero (25,200) no cumple con ese ε -> no es estable; la regla pasa a otra celda.
    def f(G, m, e):
        if (G, m) == (15, 100):
            return (1, 8) if e >= 0.005 else (9, 10)
        if (G, m) == (25, 200):
            return (1, 8) if e >= 0.02 else (9, 10)
        return (1, 8) if e >= 0.01 else (9, 10)
    r = medir.aplicar_regla(construir(f), GRUPOS, CFG)
    assert r["epsilon_estrella_por_celda"]["15x100"] == 0.005
    assert r["hay_epsilon"] and r["epsilon"] == 0.02 and (r["G"], r["m"]) == (25, 200)


def test_si_ninguna_celda_cumple_no_hay_epsilon():
    r = medir.aplicar_regla(construir(lambda G, m, e: (9, 10)), GRUPOS, CFG)
    assert r["hay_epsilon"] is False and r["epsilon"] is None and r["G"] is None
    assert set(r["epsilon_estrella_por_celda"].values()) == {None}


def test_sin_potencia_o_sin_casos_no_se_cumple():
    celdas = construir(lambda G, m, e: (0, 10))
    celdas[("a|reg", 15, 100, 0.0)] = {"monotono": [0, 0, 0], "no_monotono": [0, 0, 0]}                 # sin casos: FA indefinida
    t = medir.tasas(celdas, GRUPOS, 15, 100, 0.0)
    assert t["a|reg"]["FA"] is None and not medir.cumple(t)


def test_la_etiqueta_invertida_no_elige_lo_mismo_que_la_verdadera():
    celdas = construir(lambda G, m, e: (1, 9) if e >= 0.01 else (9, 10))
    real = medir.aplicar_regla(celdas, GRUPOS, CFG)
    falso = medir.aplicar_regla(celdas, GRUPOS, CFG, invertida=True)
    assert real["epsilon"] == 0.01
    assert falso["hay_epsilon"] is False                                  # invertidas: FA' = 90 %, no cumple con ningún ε
    assert medir.tasas(celdas, GRUPOS, 15, 100, 0.01, invertida=True)["a|reg"]["FA"] == 0.9


def test_la_verdad_conocida_es_monotona_o_no_donde_dice_la_regla():
    """La verdad de los datos sintéticos: 4 columnas monótonas con su dirección, 2 planas, 4 con U/senoidal/interacción."""
    v = [0.0] * 10
    base = list(v)
    for i, c in enumerate(medir.COLUMNAS):
        base[i] = 5.0
    def curva(k, tarea="regression"):
        out = []
        for x in [10.0 * j / 20 for j in range(21)]:
            w = list(base); w[k] = x
            out.append(medir.salida_verdadera(tarea, w))
        return out
    from matrixai.estudio.comportamiento import retroceso_de_una_curva
    for k, c in enumerate(medir.COLUMNAS):
        tipo, direccion = medir.VERDAD_DE_LA_COLUMNA[c]
        for tarea in ("regression", "binary_classification"):
            ret = {d: retroceso_de_una_curva(curva(k, tarea), d).valor for d in ("sube", "baja")}
            if tipo in ("lineal", "saturacion"):
                assert ret[direccion] == 0.0 and ret["baja" if direccion == "sube" else "sube"] > 0.1, (c, tarea)
            if tipo in ("U", "senoidal"):
                assert ret["sube"] > 0.1 and ret["baja"] > 0.1, (c, tarea)
            if tipo == "plana":
                assert ret == {"sube": 0.0, "baja": 0.0}, (c, tarea)
