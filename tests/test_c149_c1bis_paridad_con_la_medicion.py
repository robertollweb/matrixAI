# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C1-bis — el veredicto del NÚCLEO da el MISMO que el guion que lo midió (`benchmarks/c149/medir_c0ter.py`).

Los valores por omisión (regresión ε 5 % / q 5 %, binaria ε 20 % / q 20 %) salen de una medición hecha con `veredicto_ter`
del guion. Si el núcleo implementara otra regla —aunque fuera parecida—, esos valores medirían otra cosa. Aquí se compara,
sin entrenar nada, el veredicto de `MonotoniaDeUnaColumna` con el del guion sobre curvas de las dos clases (a mano y
generadas con semilla), sobre toda la rejilla de la medición (ε × q × dirección × tarea), y se comprueba que la tabla de
`TOLERANCIA_POR_OMISION` es la que ELIGIÓ el resultado archivado de la medición. La comparación con los 240 casos entrenados de
verdad (LightGBM y HGB, semillas 6-8) es cara y se corre aparte: `.agentes/149/c1bis/paridad_c0ter.py` (resultado en el
contrato, §9).
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

from matrixai.estudio.comportamiento import (
    TOLERANCIA_POR_OMISION,
    escala_de_la_salida,
    monotonia_desde_curvas,
)

RAIZ = Path(__file__).resolve().parents[1]
if "medir_c0ter_149" in sys.modules:
    ter = sys.modules["medir_c0ter_149"]
else:
    _ESPEC = importlib.util.spec_from_file_location("medir_c0ter_149", RAIZ / "benchmarks" / "c149" / "medir_c0ter.py")
    ter = importlib.util.module_from_spec(_ESPEC)
    sys.modules["medir_c0ter_149"] = ter
    _ESPEC.loader.exec_module(ter)

MALLA = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
ESCALAS = {"binary_classification": escala_de_la_salida("binary_classification"),
           "regression": escala_de_la_salida("regression", amplitud_p90_p10=7.5)}


def _veredicto_del_nucleo(curvas, direccion, escala, e_pct, q_pct):
    r = monotonia_desde_curvas("x", direccion, MALLA, curvas, list(range(len(curvas))), escala=escala,
                               medido_sobre="muestra_de_desarrollo", semilla=6, epsilon_relativo=e_pct / 100.0,
                               q_relativo=q_pct / 100.0)
    return r.veredicto, r


def _veredicto_del_guion(r, curvas, q_pct):
    return ter.veredicto_ter(r.retrocesos, ter.amplitud_de_curvas(curvas), r.epsilon_absoluto, q_pct)


def _curvas_con_semilla(semilla, escala):
    """Filas de todas las clases que miden los casos: crecientes, decrecientes, U, senoides, planas con ruido, escalones y huecos,
    a la escala de la tarea. Cuantizadas (a 1/16 de la escala) en parte de los casos, para que haya EMPATES exactos."""
    rng = random.Random(f"c1bis:{semilla}")
    s = escala.valor
    n = rng.choice([1, 2, 7, 19, 20, 21, 40, 100])
    clase = rng.choice(["creciente", "decreciente", "U", "senoide", "plana", "escalones", "mezcla"])
    cuantizar = rng.random() < 0.5
    fuera = []
    for _ in range(n):
        c = clase if clase != "mezcla" else rng.choice(["creciente", "decreciente", "U", "senoide", "plana", "escalones"])
        t = [j / (len(MALLA) - 1) for j in range(len(MALLA))]
        amp = rng.choice([0.01, 0.04, 0.05, 0.06, 0.2, 0.25, 0.5, 1.0]) * s
        ruido = rng.choice([0.0, 0.0, 0.01, 0.1]) * s
        base = {"creciente": [amp * x for x in t], "decreciente": [amp * (1 - x) for x in t],
                "U": [amp * (2 * x - 1) ** 2 for x in t], "senoide": [amp * (1 + math.sin(6.28 * x)) / 2 for x in t],
                "plana": [0.0] * len(t), "escalones": [amp * (0 if x < 0.4 else 1 if x < 0.7 else 0.4) for x in t]}[c]
        fila = [b + rng.uniform(-ruido, ruido) for b in base]
        if cuantizar:
            paso = s / 16.0
            fila = [round(v / paso) * paso for v in fila]
        if rng.random() < 0.05:
            fila[rng.randrange(len(fila))] = None                      # sin predicción: la fila no cuenta
        fuera.append(fila)
    return fuera


@pytest.mark.parametrize("tarea", ["binary_classification", "regression"])
def test_el_nucleo_da_el_mismo_veredicto_que_el_guion_en_toda_la_rejilla_medida(tarea):
    escala = ESCALAS[tarea]
    comparados, por_veredicto = 0, {}
    for semilla in range(250):
        curvas = _curvas_con_semilla(semilla, escala)
        for direccion in ("sube", "baja"):
            for e in ter.EPS_PCT:
                for q in ter.Q_PCT:
                    nucleo, r = _veredicto_del_nucleo(curvas, direccion, escala, e, q)
                    guion = _veredicto_del_guion(r, curvas, q)
                    assert nucleo == guion, (tarea, semilla, direccion, e, q, nucleo, guion)
                    comparados += 1
                    por_veredicto[nucleo] = por_veredicto.get(nucleo, 0) + 1
    # control del instrumento: la comparación vio las tres palabras (un generador que solo diera una no compararía nada)
    assert set(por_veredicto) == {"con_contraejemplos", "sin_contraejemplos", "no_se_pudo_comprobar"}, por_veredicto
    assert min(por_veredicto.values()) > 200 and comparados == 250 * 2 * 5 * 3


def test_toda_fraccion_exacta_de_filas_da_el_mismo_veredicto_que_el_guion():
    """n filas, v violadoras, cada q de la medición: el borde `>=` (y el 5 %, 10 %, 20 % exactos) en todas las combinaciones."""
    escala = ESCALAS["binary_classification"]
    buena, mala = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], [0.0, 1.0, 0.5, 0.6, 0.7, 0.8, 1.0]       # retroceso 0 / 0,5 > ε=0,2
    comparados = 0
    for n in range(1, 41):
        for v in range(n + 1):
            curvas = [mala] * v + [buena] * (n - v)
            for q in ter.Q_PCT:
                nucleo, r = _veredicto_del_nucleo(curvas, "sube", escala, 20, q)
                assert nucleo == _veredicto_del_guion(r, curvas, q), (n, v, q)
                assert (nucleo == "con_contraejemplos") == (v * 100 >= q * n), (n, v, q)
                comparados += 1
    assert comparados == sum((n + 1) * 3 for n in range(1, 41))


def test_la_amplitud_justo_en_epsilon_da_lo_mismo_en_el_guion_y_en_el_nucleo():
    for tarea, escala in ESCALAS.items():
        for e in ter.EPS_PCT:
            eps_abs = e / 100.0 * escala.valor
            for amplitud in (eps_abs, eps_abs * (1 + 1e-9), eps_abs * (1 - 1e-9)):
                curvas = [[0.0] * 3 + [amplitud] * 4] * 5                     # monótona, amplitud exacta
                nucleo, r = _veredicto_del_nucleo(curvas, "sube", escala, e, 5)
                assert nucleo == _veredicto_del_guion(r, curvas, 5), (tarea, e, amplitud)
            # y el valor exacto del borde NO pasa la puerta, en ninguno de los dos
            exacta = [[0.0] * 3 + [eps_abs] * 4] * 5
            if exacta[0][-1] - exacta[0][0] == eps_abs:
                assert _veredicto_del_nucleo(exacta, "sube", escala, e, 5)[0] == "no_se_pudo_comprobar"


def test_la_tabla_de_valores_por_omision_es_la_que_eligio_la_medicion_archivada():
    """`TOLERANCIA_POR_OMISION` no se escribe: se copia de lo que eligió la regla. Si alguien cambia la tabla (o se vuelve a medir y
    elige otra cosa), esta prueba lo dice."""
    medicion = json.loads((RAIZ / "benchmarks" / "c149" / "resultados" / "c0ter_resultado.json").read_text(encoding="utf-8"))
    for tarea in ("regression", "binary_classification"):
        eleccion = medicion["por_tarea"][tarea]["eleccion"]
        assert TOLERANCIA_POR_OMISION[tarea] == (eleccion["epsilon_pct"] / 100.0, eleccion["q_pct"] / 100.0), tarea
    assert set(TOLERANCIA_POR_OMISION) == set(medicion["por_tarea"])             # y nada más (multiclase, fuera)
    assert medicion["config"]["epsilon_pct"] == list(ter.EPS_PCT) and medicion["config"]["q_pct"] == list(ter.Q_PCT)
