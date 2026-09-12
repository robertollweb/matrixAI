# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C3 — el caché de la pasada exploratoria tiene que VER las reparaciones.

`pasada_exploratoria_101_c3.py` reusa del caché los intentos cuyo digest de
entorno y de motor no han cambiado. Es lo que permite reparar un motor sin
relanzar 720 intentos de 1,25 h.

El riesgo es el que trajo este fichero: **un caché que no ve el arreglo es
peor que no tener caché**, porque da un número nuevo con datos viejos y nadie
lo nota. Medido el 2026-09-12: los tres defectos de cableado que costaron los
36 intentos fallidos de la pasada del 09-07 vivían en `dataset_project.py`,
`playground.py` y `forward/dense_forward.py`, y **ninguno de los tres
invalidaba el caché**. Una re-pasada sin `--forzar` habría reusado justo los
intentos que se acababan de reparar.
"""
from __future__ import annotations

import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import pasada_exploratoria_101_c3 as pasada  # noqa: E402


def test_los_tres_ficheros_del_cableado_invalidan_el_cache():
    """Los tres donde vivieron los defectos del 09-07. No están por simetría:
    cada uno tiene su intento perdido detrás."""
    nombres = {f.name for f in pasada._FICHEROS_COMPARTIDOS}
    for fichero in ("dataset_project.py", "playground.py", "dense_forward.py"):
        assert fichero in nombres, f"{fichero} no invalida el caché"


def test_la_preparacion_y_el_harness_siguen_invalidando():
    """La otra mitad: la lista no se ha reordenado perdiendo lo que ya tenía."""
    nombres = {f.name for f in pasada._FICHEROS_COMPARTIDOS}
    for fichero in ("preparacion.py", "harness.py", "subproceso.py", "motor.py"):
        assert fichero in nombres, f"{fichero} dejó de invalidar el caché"


def test_ningun_fichero_de_la_lista_ha_DESAPARECIDO():
    """El fallo más silencioso posible de esta lista: si alguien renombra o
    mueve un fichero, su entrada apunta a la nada.

    Hoy `_digest_fichero` hace `ruta.read_bytes()` y eso levantaría
    `FileNotFoundError`, que se ve. Pero este test existe para que se vea
    AQUÍ, con el nombre del fichero, y no cinco minutos después en mitad de
    una pasada de 1,25 h que ya había empezado a medir."""
    ausentes = [str(f) for f in pasada._FICHEROS_COMPARTIDOS if not f.exists()]
    assert ausentes == [], f"la lista apunta a ficheros que no existen: {ausentes}"


def test_cada_motor_declara_UN_fichero_que_existe():
    ausentes = {m: str(f) for m, f in pasada._FICHERO_POR_MOTOR.items() if not f.exists()}
    assert ausentes == {}, f"ficheros por motor inexistentes: {ausentes}"


def test_el_digest_del_entorno_CAMBIA_si_cambia_un_compartido(tmp_path):
    """Sin este test, los de arriba los pasaría una lista bien escrita que
    nadie usa: comprueba que el digest de verdad depende del contenido."""
    antes = pasada._digest_entorno()
    original = list(pasada._FICHEROS_COMPARTIDOS)
    señuelo = tmp_path / "senuelo.py"
    señuelo.write_text("# uno", encoding="utf-8")
    try:
        pasada._FICHEROS_COMPARTIDOS = tuple(original) + (señuelo,)
        con_señuelo = pasada._digest_entorno()
        assert con_señuelo != antes
        señuelo.write_text("# dos", encoding="utf-8")
        assert pasada._digest_entorno() != con_señuelo, (
            "el digest no cambió al cambiar el CONTENIDO de un compartido")
    finally:
        pasada._FICHEROS_COMPARTIDOS = tuple(original)
    assert pasada._digest_entorno() == antes
