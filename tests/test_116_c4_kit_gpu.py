# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C4 — el kit para medir TabICL en la GPU de Roberto: lo que se puede probar sin torch.

El kit corre en su PC, dentro de su imagen; aquí se prueba la LÓGICA que decide qué se mide y
qué se salta, porque un fallo ahí le costaría a Roberto una tarde de GPU: el orden de la
rejilla, la regla de 116-C0 (tras un punto que no termina, los MAYORES de su fila no se lanzan),
retomar sin volver a medir lo medido, y la verificación de los pesos por su huella.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent / "benchmarks" / "tabicl_116c4_gpu"


def _cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(f"kit_116c4_{nombre}", KIT / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


todo = _cargar("medir_todo")
pesos = _cargar("bajar_pesos")


def test_la_rejilla_va_de_menos_a_mas_filas_dentro_de_cada_tarea_y_anchura():
    lista = todo.puntos(tareas=("binaria",), columnas=(8, 32), filas=(4000, 1000, 2000))
    assert lista == [("binaria", 8, 1000), ("binaria", 8, 2000), ("binaria", 8, 4000),
                     ("binaria", 32, 1000), ("binaria", 32, 2000), ("binaria", 32, 4000)]
    assert len(todo.puntos()) == 3 * 4 * 6


def test_tras_un_punto_que_no_termina_los_mayores_de_su_fila_no_se_lanzan():
    lanzados = []

    def medir(tarea, columnas, filas):
        lanzados.append((tarea, columnas, filas))
        estado = "tope" if (columnas, filas) == (8, 2000) else "medido"
        return {"tarea": tarea, "columnas": columnas, "filas": filas, "estado": estado}

    lista = todo.puntos(tareas=("binaria",), columnas=(8, 32), filas=(1000, 2000, 4000))
    hechos = todo.recorrer(lista, medir)
    # La fila de 8 columnas se corta tras 2.000; la de 32 sigue ENTERA (no se contagia).
    assert ("binaria", 8, 4000) not in lanzados
    assert [h["estado"] for h in hechos] == ["medido", "tope", "no_lanzado",
                                            "medido", "medido", "medido"]
    assert hechos[2]["motivo"] == "no lanzado, por el anterior"


def test_al_relanzar_reaprovecha_lo_medido_y_repite_lo_que_no_termino():
    previos = [{"tarea": "binaria", "columnas": 8, "filas": 1000, "estado": "medido", "x": 1},
               {"tarea": "binaria", "columnas": 8, "filas": 2000, "estado": "fallo"}]
    lanzados = []

    def medir(tarea, columnas, filas):
        lanzados.append(filas)
        return {"tarea": tarea, "columnas": columnas, "filas": filas, "estado": "medido"}

    hechos = todo.recorrer(todo.puntos(tareas=("binaria",), columnas=(8,), filas=(1000, 2000)),
                           medir, previos)
    assert lanzados == [2000]                 # el medido NO se repite; el fallido SÍ
    assert hechos[0]["x"] == 1


def test_guarda_tras_cada_punto():
    guardados = []
    todo.recorrer(todo.puntos(tareas=("binaria",), columnas=(8,), filas=(1000, 2000)),
                  lambda t, c, f: {"tarea": t, "columnas": c, "filas": f, "estado": "medido"},
                  guardar=lambda hechos: guardados.append(len(hechos)))
    assert guardados == [1, 2]


def test_los_pesos_se_verifican_por_su_huella(tmp_path):
    assert len(pesos.verificar(tmp_path)) == 2            # los dos ausentes
    for nombre in pesos.PESOS:
        (tmp_path / nombre).write_bytes(b"otra cosa")
    fallos = pesos.verificar(tmp_path)
    assert len(fallos) == 2 and all("sha256" in f for f in fallos)
    # Y las huellas son las mismas que espera el adaptador de TabICL (motores): una sola verdad.
    adaptador = (Path(__file__).resolve().parents[2] / "matrixai-engines" / "src" / "matrixai_engines"
                 / "motores" / "tabicl.py").read_text(encoding="utf-8")
    for nombre, huella in pesos.PESOS.items():
        assert nombre in adaptador and huella in adaptador, nombre
    assert hashlib.sha256(b"").hexdigest() != next(iter(pesos.PESOS.values()))
