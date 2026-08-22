"""A2 — UNA CAPTURA DICE SI SU PROCEDENCIA SE COMPROBÓ (contrato 82, §6.11).

Decisión de Roberto, 2026-08-20: la captura sube a **esquema 1.2** con un
campo que declara si la receta y la semilla se comprobaron. Sin comprobar,
**no sostiene un `reproducible: true`**.

El agujero que cierra, medido el 2026-08-20: `recipe_text` y
`dataset_seed` llegan del cliente y se guardan «tal cual»; el Studio SÍ
regenera el CSV y lo compara byte a byte, pero ese veredicto vivía **solo
en la respuesta HTTP** de train-start y no acompañaba al paquete. Una
captura comprobada y una sin comprobar producían bloques `provenance`
IDÉNTICOS — el mismo patrón que el propio contrato reprocha en
`reproduce.py` («el aviso vivía solo en la respuesta HTTP del export, que
no acompaña al ZIP»).

El vocabulario del código NO lo inventa el core: es el que ya usa el
backend (`regenera_el_dataset`, `no_regenera_el_dataset`,
`no_se_ha_podido_comprobar`). El core decide con el booleano y publica el
código tal como se lo den: dos sitios declarando el mismo vocabulario
acabarían divergiendo.
"""

import hashlib
import json
import os
import tempfile
import unittest

from matrixai.export import ReproduceManifestError, build_reproduce_manifest

_CSV = b"a,b\n1,2\n3,4\n"
_SHA = hashlib.sha256(_CSV).hexdigest()
_RECETA = "genera 200 filas\n"


def _bundle():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "model.mxai"), "w") as fh:
        fh.write("NETWORK N\n  DENSE 4\n")
    with open(os.path.join(d, "training.mxtrain"), "w") as fh:
        fh.write(open("examples/celsius_to_kelvin.mxtrain").read())
    with open(os.path.join(d, "recipe.txt"), "w") as fh:
        fh.write(_RECETA)
    return d


class ProcedenciaVerificadaTest(unittest.TestCase):
    def setUp(self):
        self.d = _bundle()
        self.train = open(os.path.join(self.d, "training.mxtrain")).read()

    def _captura(self, **over):
        cap = {
            "schema_version": "1.2",
            "mxai_sha256": hashlib.sha256(
                open(os.path.join(self.d, "model.mxai"), "rb").read()).hexdigest(),
            "mxtrain_sha256": hashlib.sha256(self.train.encode()).hexdigest(),
            "mxtrain_text": self.train,
            "recipe_sha256": hashlib.sha256(_RECETA.encode()).hexdigest(),
            "recipe_text": _RECETA,
            "seeds": {"dataset": 42},
            "dataset_sha256_raw": _SHA, "dataset_sha256_prepared": _SHA,
            "dataset_rows": 2, "dataset_rows_used": 2,
            "epochs_effective": 10000, "epochs_ran": 10000,
            "warm_start": False,
            "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
        }
        for k, v in over.items():
            if v is ...:
                cap.pop(k, None)
            else:
                cap[k] = v
        return cap

    def _build(self, **over):
        return build_reproduce_manifest(
            self.d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
            dataset_sha256=_SHA, dataset_rows=2,
            generation={"seeds": {"dataset": 42}},
            run_provenance=self._captura(**over), weights_source="trained")

    # ── lo que se gana ────────────────────────────────────────────────

    def test_comprobada_sostiene_la_promesa(self):
        m = self._build()
        self.assertTrue(m["reproducible"], m.get("reproducible_reason"))

    def test_no_comprobada_no_la_sostiene(self):
        """El caso del core sirviendo su propio `/api/train-start`: la
        receta llega del cliente y nadie la contrasta."""
        m = self._build(recipe_verification={"verified": False,
                                             "code": "no_se_ha_podido_comprobar"})
        self.assertFalse(m["reproducible"])
        self.assertIn("recipe_verification", m["missing"] + [c.get("field") for c in m["conflicts"]])

    def test_desmentida_tampoco(self):
        m = self._build(recipe_verification={"verified": False,
                                             "code": "no_regenera_el_dataset"})
        self.assertFalse(m["reproducible"])

    def test_el_codigo_del_backend_viaja_tal_cual(self):
        """El core no traduce el vocabulario de quien comprobó: lo publica.
        Quien recibe el paquete tiene que poder leer POR QUÉ."""
        m = self._build()
        self.assertEqual(
            m["provenance"]["recipe_verification"],
            {"verified": True, "code": "regenera_el_dataset"})

    # ── lo que NO se rompe ────────────────────────────────────────────

    def test_un_paquete_sin_receta_no_necesita_verificacion(self):
        """Sin receta no hay nada que comprobar: exigir el campo aquí
        convertiría en no reproducible un modelo entrenado con datos
        propios, que es un caso honesto y frecuente.

        La condición mira el ARTEFACTO, no la captura, y es lo correcto:
        si el paquete lleva una receta, es sobre ella sobre la que promete
        R1. Por eso este caso construye un bundle SIN `recipe.txt` — mi
        primera versión solo la quitaba de la captura y seguía llevándola
        dentro, que es otro caso distinto.
        """
        m = build_reproduce_manifest(
            self.d, training_filename="training.mxtrain",
            dataset_sha256=_SHA, dataset_rows=2,
            generation={"seeds": {"dataset": 42}},
            run_provenance=self._captura(recipe_text=..., recipe_sha256=...,
                                         recipe_verification=...),
            weights_source="trained")
        self.assertNotIn("recipe_verification", m["missing"])

    def test_una_captura_vieja_lo_dice_y_no_lo_finge(self):
        """1.1 no tenía el campo. No se le inventa un `verified: true`."""
        m = self._build(schema_version="1.1", recipe_verification=...)
        self.assertFalse(m["reproducible"])
        self.assertIn("recipe_verification", m["missing"])

    # ── formas imposibles ─────────────────────────────────────────────

    def test_una_verificacion_sin_veredicto_se_rechaza(self):
        for basura in ({}, {"code": "x"}, {"verified": "si"}, {"verified": 1}, []):
            with self.subTest(v=basura):
                with self.assertRaises(ReproduceManifestError):
                    self._build(recipe_verification=basura)

    def test_verificada_sin_receta_es_una_contradiccion(self):
        """Decir «comprobé la receta» sin receta que comprobar."""
        with self.assertRaises(ReproduceManifestError):
            self._build(recipe_text=..., recipe_sha256=...)


if __name__ == "__main__":
    unittest.main()
