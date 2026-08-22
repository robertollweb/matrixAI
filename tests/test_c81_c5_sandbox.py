"""CONTRATO 81-C5 — `Replay & Verify` en entorno aislado.

Dos reglas de la revisión, y son las que este fichero defiende:

> **P23-R-0022.** El contrato **DEBE** declarar los backends de
> aislamiento soportados y su versión mínima.
>
> **P23-R-0023.** Cuando ninguno esté disponible, `Replay & Verify`
> **DEBE fallar cerrado**: no ejecutar y decir por qué. **NO DEBE**
> degradarse en silencio a ejecución sin aislamiento — sería prometer un
> sandbox y correr sin él, que es peor que no ofrecerlo.

La segunda es la que importa de verdad: un `Replay & Verify` que se
ejecuta sin sandbox y no lo dice le da a alguien una confianza que no ha
ganado.
"""

import unittest
from unittest.mock import patch

from matrixai.pipelines.sandbox import (
    BACKENDS_SOPORTADOS,
    SinAislamiento,
    describir_aislamiento,
    replay_and_verify,
)


class LosBackendsSeDECLARANTest(unittest.TestCase):
    def test_hay_una_lista_declarada_con_version_minima(self):
        """P23-R-0022. «Se ejecuta aislado» sin decir con qué no se puede
        auditar: quien lo lea no sabe qué está confiando."""
        self.assertTrue(BACKENDS_SOPORTADOS)
        for backend in BACKENDS_SOPORTADOS:
            self.assertTrue(backend["name"])
            self.assertTrue(backend["min_version"])

    def test_describir_dice_cual_hay_y_cual_falta(self):
        with patch("matrixai.pipelines.sandbox._version_de", return_value=None):
            d = describir_aislamiento()
        self.assertFalse(d["available"])
        # Y enseña los que buscó: sin eso, «no hay sandbox» no dice qué
        # instalar.
        self.assertEqual(len(d["checked"]), len(BACKENDS_SOPORTADOS))


class SinAislamientoNOSeEjecutaTest(unittest.TestCase):
    """P23-R-0023, la regla central del corte."""

    def test_falla_CERRADO_y_dice_por_que(self):
        with patch("matrixai.pipelines.sandbox._version_de", return_value=None):
            with self.assertRaises(SinAislamiento) as caja:
                replay_and_verify("/tmp/paquete")
        mensaje = str(caja.exception)
        self.assertIn("sin aislamiento", mensaje.lower())
        # Dice QUÉ instalar, no solo que no puede.
        self.assertTrue(any(b["name"] in mensaje for b in BACKENDS_SOPORTADOS))

    def test_NO_se_degrada_a_ejecucion_sin_sandbox(self):
        """Si alguna vez alguien añadiera un `if no hay sandbox: ejecutar
        igual`, esta prueba se pone roja. Es la diferencia entre prometer
        un sandbox y tenerlo."""
        import inspect
        from matrixai.pipelines import sandbox
        fuente = inspect.getsource(sandbox.replay_and_verify)
        self.assertIn("raise SinAislamiento", fuente)
        for peligroso in ("subprocess.run([sys.executable", "os.system", "eval("):
            self.assertNotIn(peligroso, fuente)

    def test_ni_siquiera_con_una_bandera_amable(self):
        """No hay `permitir_sin_sandbox=True`: una salida de emergencia
        que nadie vigila acaba siendo el camino normal."""
        import inspect
        from matrixai.pipelines import sandbox
        firma = inspect.signature(sandbox.replay_and_verify)
        # ANCLA POSITIVA (auditoría 2ª pasada): con una firma sin
        # parámetros el bucle no se ejecutaría y la prueba pasaría sola.
        self.assertIn("paquete", firma.parameters)
        for parametro in firma.parameters:
            self.assertNotIn("sin_sandbox", parametro)
            self.assertNotIn("unsafe", parametro)


class ConAislamientoSeDECLARALoQueSEAPLICATest(unittest.TestCase):
    def test_los_limites_viajan_en_el_informe(self):
        """Decir «aislado» sin decir con qué límites es medio dato: un
        sandbox sin tope de memoria no protege de lo que más pasa."""
        with patch("matrixai.pipelines.sandbox._version_de", return_value="24.0.0"), \
             patch("matrixai.pipelines.sandbox._ejecutar_en", return_value={"ok": True}):
            informe = replay_and_verify("/tmp/paquete")
        limites = informe["isolation"]["limits"]
        for clave in ("network", "cpu", "memory", "timeout_s", "filesystem"):
            self.assertIn(clave, limites)
        # La red, DESACTIVADA por defecto.
        self.assertEqual(limites["network"], "disabled")

    def test_declara_el_backend_y_su_version(self):
        with patch("matrixai.pipelines.sandbox._version_de", return_value="24.0.0"), \
             patch("matrixai.pipelines.sandbox._ejecutar_en", return_value={"ok": True}):
            informe = replay_and_verify("/tmp/paquete")
        self.assertTrue(informe["isolation"]["backend"])
        self.assertEqual(informe["isolation"]["version"], "24.0.0")


if __name__ == "__main__":
    unittest.main()
