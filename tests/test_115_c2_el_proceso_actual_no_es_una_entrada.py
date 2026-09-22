# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C2 — la columna del proceso actual NO es una entrada del modelo.

Medido el 2026-09-22: con `proceso_actual=columna("decision_hoy")`, la columna salía
entre los predictores sin declarar las entradas y declarándolas, sin un aviso."""
from __future__ import annotations

import csv
import io
import random

from matrixai.estudio import ProcesoActual
from matrixai.training.objetivo import confirmar_desde_csv, confirmar_desde_prompt

HOY = ProcesoActual(tipo="columna", columna="decision_hoy")


def _csv(columnas=("x1", "decision_hoy")):
    azar = random.Random(0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["row_id", *columnas, "y"])
    for i in range(60):
        x = azar.uniform(0, 10)
        valores = {"x1": round(x, 3), "decision_hoy": "si" if x > 4 else "no"}
        w.writerow([i, *(valores[c] for c in columnas), "si" if x > 5 else "no"])
    return buf.getvalue()


def _confirmar(**extra):
    base = dict(objetivo="y", tarea="binary_classification", clase_positiva="si",
                unidad_de_observacion="una fila", proceso_actual=HOY)
    base.update(extra)
    return confirmar_desde_csv(_csv(), **base)


def _la_exclusion(conf):
    (exclusion,) = [e for e in conf.excluidas if e.clave == "proceso_actual_excluido"]
    assert exclusion.campo == "decision_hoy"
    assert "'decision_hoy'" in exclusion.motivo["es"] and "'decision_hoy'" in exclusion.motivo["en"]
    return exclusion


def test_sin_declarar_las_entradas_queda_fuera_y_se_dice():
    conf = _confirmar()
    assert conf.confirmado
    assert conf.problema.predictors == ("x1",)
    _la_exclusion(conf)


def test_declarandola_entre_las_entradas_tambien_queda_fuera():
    conf = _confirmar(entradas=["x1", "decision_hoy"])
    assert conf.confirmado and conf.problema.predictors == ("x1",)
    _la_exclusion(conf)


def test_por_la_ruta_del_prompt_tambien():
    conf = confirmar_desde_prompt(
        "predecir si y a partir de x1", objetivo="y", tarea="binary_classification",
        clase_positiva="si", clases=["no", "si"], unidad_de_observacion="una fila",
        entradas=["x1", "decision_hoy"], proceso_actual=HOY)
    if conf.confirmado:
        assert conf.problema.predictors == ("x1",)
    _la_exclusion(conf)


def test_si_era_la_unica_entrada_se_bloquea():
    conf = _confirmar(entradas=["decision_hoy"])
    assert not conf.confirmado
    assert "sin_entradas_utilizables" in [b.clave for b in conf.bloqueos]
    _la_exclusion(conf)


def test_sin_proceso_por_columna_no_se_quita_nada():
    for proceso in (None, ProcesoActual(tipo="ninguno"),
                    ProcesoActual(tipo="regla", regla="si x1 > 4, sí")):
        conf = _confirmar(proceso_actual=proceso)
        assert conf.problema.predictors == ("x1", "decision_hoy"), proceso
        assert not [e for e in conf.excluidas if e.clave == "proceso_actual_excluido"]
