# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Un entrenamiento es de quien lo lanza.

Hallazgo de una auditoría externa de la demo pública (2026-08-12), con
dos sesiones y cookies distintas:

    La sesión A inició un entrenamiento.
    La sesión B consultó su estado correctamente.
    La sesión B lo canceló.
    La sesión A lo recibió como `cancelled`.

El registro de jobs es global y las rutas solo miraban el `job_id`. En
una demo pública eso significa que cualquiera puede tumbar el
entrenamiento de otro con solo acertar —o ver— un identificador.

El dueño es una cadena OPACA: al core no le importa qué representa (el
producto le pasa el identificador de sesión), así que el motor no
aprende nada de sesiones ni de cookies.

Y `None` significa «de nadie», que es lo que corre en el Studio
descargable: ahí la máquina es de quien la usa y no hay a quién separar.
Esa mitad se prueba también — cerrar de más rompería el producto para
todos los que no son la demo.
"""
from __future__ import annotations

import unittest

from matrixai import playground


def _job(owner):
    """Un job en el registro, sin entrenar nada: lo que se prueba es
    QUIÉN puede mirarlo, no el entrenamiento."""
    jid = f"job-{owner or 'anon'}"
    playground._training_jobs[jid] = {
        "status": "running", "epochs": [], "owner": owner,
        "result": None, "error": None,
    }
    return jid


class UnEntrenamientoEsDeQuienLoLanza(unittest.TestCase):
    def tearDown(self) -> None:
        playground._training_jobs.clear()

    def test_su_dueno_lo_ve(self) -> None:
        jid = _job("sesion-A")
        self.assertTrue(playground._get_job_status(jid, "sesion-A")["ok"])

    def test_OTRO_no_lo_ve(self) -> None:
        jid = _job("sesion-A")
        r = playground._get_job_status(jid, "sesion-B")
        self.assertFalse(r["ok"])


class ElMensajeNoConfirmaQueExista(unittest.TestCase):
    """Contestar «prohibido» confirmaría que ese job existe.

    Con identificadores ajenos eso ya es información: se puede sondear
    cuáles son válidos. Para quien pregunta por lo que no es suyo, no
    existe.
    """

    def tearDown(self) -> None:
        playground._training_jobs.clear()

    def test_el_mismo_mensaje_que_un_id_inventado(self) -> None:
        jid = _job("sesion-A")
        ajeno = playground._get_job_status(jid, "sesion-B")["error"]
        inventado = playground._get_job_status("no-existe", "sesion-B")["error"]
        self.assertIn("no encontrado", ajeno)
        self.assertIn("no encontrado", inventado)


class Cancelar(unittest.TestCase):
    def tearDown(self) -> None:
        playground._training_jobs.clear()

    def test_OTRO_no_puede_cancelarlo(self) -> None:
        jid = _job("sesion-A")
        r = playground._cancel_job(jid, "sesion-B")
        self.assertFalse(r["ok"])
        # Y no lo ha tocado: sigue corriendo para su dueño.
        self.assertEqual(playground._training_jobs[jid]["status"], "running")

    def test_su_dueno_sí(self) -> None:
        jid = _job("sesion-A")
        playground._training_jobs[jid]["cancel_event"] = __import__("threading").Event()
        r = playground._cancel_job(jid, "sesion-A")
        self.assertTrue(r.get("ok"), r)


class SinDuenoNoCambiaNada(unittest.TestCase):
    """El Studio DESCARGABLE: sin sesiones, los jobs nacen sin dueño.

    Es la mitad que impide que este arreglo rompa el producto para todo
    el que no sea la demo.
    """

    def tearDown(self) -> None:
        playground._training_jobs.clear()

    def test_cualquiera_lo_ve_y_lo_cancela(self) -> None:
        jid = _job(None)
        self.assertTrue(playground._get_job_status(jid, None)["ok"])
        self.assertTrue(playground._get_job_status(jid, "quien-sea")["ok"])
        playground._training_jobs[jid]["cancel_event"] = __import__("threading").Event()
        self.assertTrue(playground._cancel_job(jid, "quien-sea").get("ok"))


if __name__ == "__main__":
    unittest.main()
