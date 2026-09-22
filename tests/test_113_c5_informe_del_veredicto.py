# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""113-C5 — el informe del veredicto (`benchmarks/veredicto_113c5/informe_113c5.py`)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "veredicto_113c5"))
import informe_113c5 as inf  # noqa: E402

DENSA = inf.DENSA


def _artefacto(cumple_densa: dict[str, bool], segundos: list[float], parcial=False):
    motores = {m: {"cumplidos": 1, "datasets": 2, "fraccion": 0.5, "cumple_la_regla": False,
                   "datasets_que_puede_perder_sin_incumplir": None,
                   "datasets_que_le_faltan_para_cumplir": 1,
                   "detalle": [{"dataset": "a", "cumple": True}, {"dataset": "b", "cumple": False}]}
               for m in inf.APROBADOS}
    motores[DENSA] = {"cumplidos": sum(cumple_densa.values()), "datasets": len(cumple_densa),
                      "fraccion": 0.0, "cumple_la_regla": False,
                      "detalle": [{"dataset": d, "cumple": c} for d, c in cumple_densa.items()]}
    atributos = {"a": ("binary_classification", "pequeno"), "b": ("regression", "grande"),
                 "c": ("regression", "pequeno"), "d": ("binary_classification", "grande"),
                 "e": ("regression", "pequeno")}
    resultados = [{"dataset": d, "tarea": atributos[d][0], "cubo": atributos[d][1], "sellado": False,
                   "motor": DENSA, "estado": "completed", "wall_s": s}
                  for d, s in zip(cumple_densa, segundos)]
    # Y filas de OTRO motor, mucho más rápidas: la mediana es de la densa, no de todos.
    resultados += [{"dataset": d, "tarea": atributos[d][0], "cubo": atributos[d][1],
                    "sellado": False, "motor": "lightgbm", "estado": "completed", "wall_s": 0.01}
                   for d in cumple_densa for _ in range(5)]
    return {"parcial": parcial, "alcance_y_veredicto": {"por_motor": motores},
            "resultados": resultados}


def test_gana_pierde_y_los_que_solo_estan_en_una():
    # «e» cumple en las DOS: no es ganar, es seguir cumpliendo.
    v1 = _artefacto({"a": False, "b": True, "c": False, "e": True}, [10, 20, 30, 40])
    v2 = _artefacto({"a": True, "b": False, "d": True, "e": True}, [1, 2, 3, 4])
    cambios = inf.cambios(v1, v2, DENSA)
    assert cambios == {"gana": ["a"], "pierde": ["b"], "solo_en_v1": ["c"], "solo_en_v2": ["d"]}


def test_el_desglose_por_tarea_y_por_tamano():
    v2 = _artefacto({"a": True, "b": False, "c": True}, [1, 2, 3])
    assert inf.desglose(v2, DENSA, "tarea") == {"binary_classification": "1/1", "regression": "1/2"}
    assert inf.desglose(v2, DENSA, "cubo") == {"grande": "0/1", "pequeno": "2/2"}


def test_la_mediana_de_segundos_y_el_informe_entero():
    v1 = _artefacto({"a": False, "b": True, "c": False}, [10, 20, 30])
    v2 = _artefacto({"a": True, "b": False, "c": True}, [1, 2, 30], parcial=True)
    resultado = inf.informe(v1, v2)
    assert resultado["v2_terminada"] is False
    assert resultado["densa"]["segundos_por_intento_mediana"] == {"v1": 20, "v2": 2}
    assert set(resultado["aprobados"]) == {"lightgbm", "sklearn.hgb", "catboost"}
    assert resultado["regla"]["v2"][DENSA]["cumplidos"] == 2
