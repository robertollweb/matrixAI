"""CONTRATO 82-C3 — la ficha que lo CUENTA.

El README del paquete gana una sección «Reproducing this» con los cinco
elementos, la huella, el entorno y las líneas de comando del C2.

**Criterio de cierre del contrato:** quien llega al repositorio sabe *en
un minuto* qué ejecutar y qué debería salir. Eso es lo que miden estas
pruebas: no que la sección exista, sino que lleve **el comando** y **lo
que debería salir**.

Y lo que NO puede hacer: prometer reproducibilidad a un paquete que no la
tiene. Un README que dice «ejecuta esto y saldrá igual» sobre un paquete
sin receta manda a alguien a perder una tarde.
"""

import unittest

from matrixai.export.bundle import readme_reproducing_section


class LaSeccionCuentaQueEjecutarTest(unittest.TestCase):
    def _manifiesto(self, **cambios):
        base = {
            "reproducible": True,
            "reproducible_reason": None,
            "artifacts": {
                "model": {"path": "model.mxai", "sha256": "a" * 64},
                "training": {"path": "model.mxtrain", "sha256": "b" * 64},
                "recipe": {"path": "data_recipe.txt", "sha256": "c" * 64},
                "dataset": {"sha256": "d" * 64, "rows": 300},
            },
            "generation": {"seeds": {"dataset": 42}, "mode": "coherent"},
            "environment": {"matrixai": "1.5.0", "python": "3.12.1"},
        }
        base.update(cambios)
        return base

    def test_lleva_el_comando_que_hay_que_ejecutar(self):
        texto = readme_reproducing_section(self._manifiesto())
        self.assertIn("matrixai verify", texto)

    def test_dice_lo_que_deberia_salir(self):
        """La otra mitad del criterio. Un comando sin resultado esperado
        deja a quien lo corre sin saber si lo que ve está bien."""
        texto = readme_reproducing_section(self._manifiesto())
        for etapa in ("manifest", "R1"):
            self.assertIn(etapa, texto)
        self.assertIn("PASS", texto)

    def test_enseña_los_cinco_elementos_con_su_huella(self):
        texto = readme_reproducing_section(self._manifiesto())
        for pieza in ("model.mxai", "model.mxtrain", "data_recipe.txt"):
            self.assertIn(pieza, texto)
        # La huella del dataset y las filas: son lo que se compara.
        self.assertIn("d" * 16, texto)
        self.assertIn("300", texto)
        self.assertIn("42", texto)  # la semilla

    def test_dice_en_que_entorno_se_entrenó(self):
        """Sin esto, una diferencia de versión parece una manipulación."""
        texto = readme_reproducing_section(self._manifiesto())
        self.assertIn("1.5.0", texto)


class NoPrometeLoQueElPaqueteNoPuedeCumplirTest(unittest.TestCase):
    def test_un_paquete_no_reproducible_lo_DICE_y_no_invita_a_ejecutar(self):
        texto = readme_reproducing_section(self._con_motivo())
        self.assertIn("not reproducible", texto.lower())
        # Y con el motivo del core, citado: sin él, quien lo lea no sabe
        # si le falta un dato suyo o si el paquete está roto.
        self.assertIn("the dataset generation seed is unknown", texto)

    def _con_motivo(self):
        return {
            "reproducible": False,
            "reproducible_reason": "Not reproducible: the dataset generation seed is unknown.",
            "artifacts": {"model": {"path": "model.mxai", "sha256": "a" * 64}},
            "generation": {"seeds": {}},
            "environment": {},
        }

    def test_sin_manifiesto_no_se_inventa_una_seccion(self):
        """Un ONNX suelto no tiene nada que reproducir: la sección se
        omite en vez de escribir una vacía que parece un hueco."""
        self.assertEqual(readme_reproducing_section(None), "")


if __name__ == "__main__":
    unittest.main()


class LaSeccionLLEGAAlREADMEDeVerdadTest(unittest.TestCase):
    """Escribir la función y no llamarla es el hueco de cableado que este
    proyecto lleva repitiendo dieciséis veces. Aquí se comprueba por el
    README que compone el bundler.

    Se llama a `_build_readme` directamente y no se empaqueta un ONNX
    entero: el bundle completo necesita onnxruntime y en un entorno sin él
    la prueba se SALTARÍA — y una prueba que se salta no prueba nada, que
    es exactamente lo que este fichero critica.
    """

    def _readme(self, reproduce):
        from matrixai.export.bundle import _build_readme
        from matrixai.parser.parser import parse_text

        programa = parse_text(
            "PROJECT P\n\nVECTOR V[1]\n  a: Score\nEND\n\n"
            "NETWORK N\n  INPUT V\n  LAYER Dense units=1 activation=sigmoid\n"
            "  OUTPUT y: Probability\nEND\n\nGRAPH\n  V -> N\nEND\n")

        class _Falso:
            model_hash = "sha256:" + "a" * 64
            parameter_schema_hash = "sha256:" + "b" * 64
            onnx_path = "model.onnx"
            opset_version = 17
            input_name, output_name = "input", "output"
            input_shape, output_shape = [1, 1], [1, 1]
            exported_functions: list = []
            skipped_functions: list = []
            files: list = []
            parameter_set_id = "params-1"
            parameter_count = 2
            equivalence_max_abs_diff = 0.0

        return _build_readme(programa, _Falso(), None, reproduce=reproduce)

    def test_con_manifiesto_la_seccion_ESTA_en_el_readme(self):
        reproduce = {
            "reproducible": True, "reproducible_reason": None,
            "artifacts": {"model": {"path": "model.mxai", "sha256": "a" * 64},
                          "dataset": {"sha256": "d" * 64, "rows": 300}},
            "generation": {"seeds": {"dataset": 42}},
            "environment": {"matrixai": "1.5.0"},
        }
        readme = self._readme(reproduce)
        self.assertIn("Reproducing this", readme)
        self.assertIn("matrixai verify", readme)

    def test_sin_manifiesto_el_readme_no_la_inventa(self):
        readme = self._readme(None)
        self.assertNotIn("Reproducing this", readme)

    def test_un_paquete_no_reproducible_no_invita_a_verificar(self):
        """Y esto es lo que de verdad protege: el README de un paquete que
        no se puede reproducir NO enseña el comando, porque mandar a
        alguien a ejecutarlo es mandarle a perder una tarde."""
        readme = self._readme({
            "reproducible": False,
            "reproducible_reason": "Not reproducible: no data recipe.",
            "artifacts": {}, "generation": {}, "environment": {}})
        self.assertIn("not reproducible", readme.lower())
        self.assertNotIn("matrixai verify .", readme)
