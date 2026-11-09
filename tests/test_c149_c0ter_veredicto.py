# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C0-ter — el veredicto V'(ε, q) por AMPLITUD del guion, con curvas escritas a mano.

`benchmarks/c149/medir_c0ter.py` implementa `REGLA_DE_ELECCION_TER.md`. Aquí se prueba solo esa lógica —sin entrenar nada—
con curvas cuyas amplitudes y retrocesos se cuentan en el comentario de cada prueba.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_ESPEC = importlib.util.spec_from_file_location("medir_c0ter_149", RAIZ / "benchmarks" / "c149" / "medir_c0ter.py")
ter = importlib.util.module_from_spec(_ESPEC)
sys.modules["medir_c0ter_149"] = ter
_ESPEC.loader.exec_module(ter)

CON, SIN, NSC = "con_contraejemplos", "sin_contraejemplos", "no_se_pudo_comprobar"
EPS_REL = 0.05
MALLA = [0.0, 1.0, 2.0, 3.0, 4.0]


def veredicto_de_curvas(curvas, direccion, q=10, eps_rel=EPS_REL, escala_valor=1.0):
    """El camino entero del guion sobre curvas a mano: retrocesos de C1 (ε = 0) + amplitud + V'."""
    from matrixai.estudio.comportamiento import escala_de_la_salida, monotonia_desde_curvas
    escala = escala_de_la_salida("binary_classification")            # escala 1,0 (probabilidad)
    assert escala.valor == escala_valor
    r = monotonia_desde_curvas("x", direccion, MALLA, curvas, list(range(len(curvas))), epsilon_relativo=0.0, escala=escala,
                               medido_sobre="muestra_de_desarrollo", semilla=6).con_epsilon(eps_rel)
    return ter.veredicto_ter(r.retrocesos, ter.amplitud_de_curvas(curvas), r.epsilon_absoluto, q), r


def test_una_U_de_efecto_extremo_a_extremo_cero_y_amplitud_grande_es_con():
    # Cada fila: 0,5 -> 0,25 -> 0,0 -> 0,25 -> 0,5. Extremo a extremo 0 (efecto 0 <= ε), amplitud 0,5 > ε = 0,05.
    # Declarada «sube»: baja 0,5 de golpe antes de volver a subir -> retroceso 0,5 > ε en las 20 filas.
    curvas = [[0.5, 0.25, 0.0, 0.25, 0.5] for _ in range(20)]
    v, r = veredicto_de_curvas(curvas, "sube")
    assert abs(r.efecto_total) < 1e-12 and ter.amplitud_de_curvas(curvas) == 0.5
    assert v == CON


def test_una_monotona_declarada_al_reves_es_con_y_no_se_queda_en_no_se_pudo_comprobar():
    # Cada fila sube 0 -> 1 (amplitud 1,0). Declarada «baja»: el retroceso es 1,0 > ε y el efecto CON SIGNO es -1 (<= ε).
    curvas = [[0.0, 0.25, 0.5, 0.75, 1.0] for _ in range(20)]
    v, r = veredicto_de_curvas(curvas, "baja")
    assert r.efecto_total < 0
    assert v == CON
    # Y declarada «sube» (la verdad) no retrocede nunca: sin contraejemplos.
    assert veredicto_de_curvas(curvas, "sube")[0] == SIN


def test_una_plana_es_no_se_pudo_comprobar_aunque_el_ruido_retroceda():
    # Amplitud 0,02 <= ε = 0,05: se mira PRIMERO, y el retroceso de 0,02 tampoco superaría ε.
    curvas = [[0.50, 0.51, 0.50, 0.52, 0.50] for _ in range(20)]
    assert ter.amplitud_de_curvas(curvas) < 0.05
    for direccion in ("sube", "baja"):
        for q in ter.Q_PCT:
            assert veredicto_de_curvas(curvas, direccion, q)[0] == NSC


def test_la_amplitud_es_la_mediana_de_las_filas_no_el_maximo():
    # 20 filas: 11 planas (amplitud 0) y 9 con amplitud 1 -> mediana 0 (la 10.ª y 11.ª ordenadas son 0): no se pudo comprobar;
    # con 11 de amplitud 1 y 9 planas, la mediana es 1.
    plana, sube = [0.5] * 5, [0.0, 0.25, 0.5, 0.75, 1.0]
    assert ter.amplitud_de_curvas([plana] * 11 + [sube] * 9) == 0.0
    assert ter.amplitud_de_curvas([plana] * 9 + [sube] * 11) == 1.0
    assert veredicto_de_curvas([plana] * 11 + [sube] * 9, "baja")[0] == NSC
    assert veredicto_de_curvas([plana] * 9 + [sube] * 11, "baja")[0] == CON


def test_las_curvas_a_medias_no_cuentan_en_la_amplitud():
    # Una fila con un hueco (None) no entra en la mediana; sin filas completas no hay amplitud.
    assert ter.amplitud_de_curvas([[0.0, None, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0, 1.0]]) == 1.0
    assert ter.amplitud_de_curvas([[0.0, None, 1.0]]) is None
    assert ter.veredicto_ter([0.5], None, 0.05, 5) == NSC


def test_la_amplitud_igual_a_epsilon_no_distingue_y_una_pizca_mas_si():
    assert ter.veredicto_ter([0.5] * 20, 0.25, 0.25, 5) == NSC
    assert ter.veredicto_ter([0.5] * 20, 0.2500001, 0.25, 5) == CON


def test_la_fraccion_de_filas_decide_con_amplitud_clara():
    # Amplitud 1,0 > ε = 0,25. 20 filas: 1 violadora = 5 %, 2 = 10 %, 4 = 20 %; `>` y no `>=` en el retroceso.
    def f(n):
        return [0.5] * n + [0.0] * (20 - n)
    assert ter.veredicto_ter(f(1), 1.0, 0.25, 5) == CON
    assert ter.veredicto_ter(f(1), 1.0, 0.25, 10) == SIN
    assert ter.veredicto_ter(f(4), 1.0, 0.25, 20) == CON
    assert ter.veredicto_ter(f(3), 1.0, 0.25, 20) == SIN
    assert ter.veredicto_ter([0.25] * 20, 1.0, 0.25, 5) == SIN
    assert ter.veredicto_ter([0.5] + [0.0] * 9 + [None] * 10, 1.0, 0.25, 10) == CON      # None fuera del denominador
