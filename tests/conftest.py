# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que comparten todas las pruebas del núcleo."""
from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _ningun_entrenamiento_queda_vivo(request):
    """Una prueba que lanza un entrenamiento espera a que acabe, y lo afirma.

    El núcleo admite UN entrenamiento a la vez: `_submit_training_job`
    contesta «Ya hay un entrenamiento en curso» si hay alguno `running` en
    `_training_jobs`, y ese registro es del PROCESO. Lo que una prueba deja
    corriendo se lo encuentra la siguiente que entrene en el mismo proceso,
    sea del fichero que sea, y el rojo sale lejos del culpable. Medido el
    2026-10-01: `test_p11_cut7_playground` lanzaba uno sin esperarlo, vivía
    ~22 s y tumbaba `test_pesos_grandes_c3_job_tensors` con un `KeyError`
    (en serie y con la máquina quieta); y en la suite con `-n 4` y carga ~9,
    dos rojos en `test_regresion_c59_c3_defaults_ya_bastan`.

    Así que el rojo sale AQUÍ, en la prueba que lo dejó y con su nombre —una
    espera con tope que se agota sin afirmarlo también cae—, y el
    entrenamiento se cancela (lo que hace «Detener») para que no tumbe a la
    siguiente. Cancelar libera el registro al momento, aunque el hilo siga
    hasta acabar su época (medido: 14,5 s con el transformer de `p11`).
    """
    yield
    # No se importa si la prueba no lo hizo: la mayoría ni lo toca.
    playground = sys.modules.get("matrixai.playground")
    if playground is None:
        return
    vivos = {
        jid: job for jid, job in list(playground._training_jobs.items())
        if job.get("status") == "running"
    }
    if not vivos:
        return
    for jid, job in vivos.items():
        playground._cancel_job(jid, owner=job.get("owner"))
    pytest.fail(
        f"{request.node.nodeid} terminó con {len(vivos)} entrenamiento(s) "
        f"«running» ({', '.join(vivos)}): los lanzó sin esperar a que "
        "acabaran, o su espera agotó el tope sin afirmarlo. El siguiente que "
        "entrenara en este proceso habría recibido «Ya hay un entrenamiento "
        "en curso». Se han cancelado. Espera hasta que el estado deje de ser "
        "«running» y afírmalo (tests/conftest.py).",
        pytrace=False,
    )
