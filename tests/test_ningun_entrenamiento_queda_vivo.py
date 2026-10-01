# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El guardián de `tests/conftest.py`: una prueba que deja un entrenamiento
vivo cae ELLA, con su nombre, y la siguiente no lo hereda.

Se prueba en un pytest HIJO sobre un fichero generado, con el conftest
cargado como plugin: el rojo que se busca es el de otra prueba, y aquí no
se puede provocar sin tumbar esta. Los entrenamientos son FALSOS (una
entrada `running` en el registro, como en `test_los_entrenamientos_tienen_
dueno`): lo que se prueba es el guardián, no el entrenador."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def _pytest_hijo(codigo: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        fichero = Path(tmp) / "test_generado.py"
        fichero.write_text(textwrap.dedent(codigo), encoding="utf-8")
        entorno = dict(os.environ)
        entorno["PYTHONPATH"] = os.pathsep.join(
            [str(_TESTS), str(_TESTS.parent), entorno.get("PYTHONPATH", "")])
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(fichero), "-q",
             "-p", "conftest", "-p", "no:cacheprovider", "-p", "no:xdist"],
            cwd=tmp, env=entorno, capture_output=True, text=True, timeout=120,
        )


class ElGuardianDeLosEntrenamientosVivos(unittest.TestCase):
    def test_cae_la_que_lo_deja_con_su_nombre_y_la_siguiente_no_lo_hereda(self):
        r = _pytest_hijo('''
            from matrixai import playground

            def test_deja_uno_vivo():
                # Con DUEÑO: `_cancel_job` sin el dueño contesta «no
                # encontrado» y no lo cancela.
                playground._training_jobs["job-fantasma"] = {
                    "status": "running", "epochs": [], "owner": "sesion-A",
                    "result": None, "error": None,
                }

            def test_la_siguiente_no_lo_hereda():
                vivos = [j for j, v in playground._training_jobs.items()
                         if v["status"] == "running"]
                assert vivos == [], vivos
                assert playground._training_jobs["job-fantasma"]["status"] == "cancelled"
        ''')
        salida = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, salida)
        # «2 passed»: la llamada de la que lo deja PASA; lo que cae es su teardown.
        self.assertIn("2 passed, 1 error", salida, salida)
        self.assertIn("ERROR at teardown of test_deja_uno_vivo", salida, salida)
        self.assertIn("test_generado.py::test_deja_uno_vivo terminó con 1 "
                      "entrenamiento(s) «running» (job-fantasma)", salida, salida)

    def test_una_prueba_que_espera_al_suyo_no_cae(self):
        """El control: sin nada vivo al terminar, ningún rojo."""
        r = _pytest_hijo('''
            from matrixai import playground

            def test_termina_el_suyo():
                playground._training_jobs["job-acabado"] = {
                    "status": "done", "epochs": [], "owner": None,
                    "result": None, "error": None,
                }
        ''')
        salida = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, salida)
        self.assertIn("1 passed", salida, salida)

    def test_no_importa_el_playground_si_la_prueba_no_lo_hizo(self):
        """El guardián corre tras CADA prueba del núcleo y la mayoría no
        entrena: importarlo él cargaría el playground (y torch) en todas."""
        r = _pytest_hijo('''
            import sys

            def test_no_entrena():
                pass

            def test_y_nadie_lo_ha_importado():
                assert "matrixai.playground" not in sys.modules
        ''')
        salida = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, salida)
        self.assertIn("2 passed", salida, salida)


if __name__ == "__main__":
    unittest.main()
