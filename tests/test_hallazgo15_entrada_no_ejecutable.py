"""HALLAZGO 15 · UNA ENTRADA QUE NO SE PUEDE EJECUTAR SE PUBLICA MARCADA.

Medido montando el E2E del contrato 87: `registry push` acepta un run **sin
`model.mxai`** —está documentado como opcional, y publicar solo métricas tiene
usos legítimos— y la entrada resultante **falla al ejecutarse** con un
`FileNotFoundError` crudo desde dentro del nodo.

El motor lo declaraba honestamente (`status: failed`, y el recibo con
`outcome: failed`), así que no había mentira. Lo que faltaba es que **en la
lista las dos entradas se ven igual**: no había forma de saber cuál se puede
correr sin intentarlo.

Decisión de Roberto (2026-08-26), por la recomendación: **opción (b)** —se
publica, y se marca. Ni se prohíbe publicar (rompería un uso legítimo) ni se
deja como estaba.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.registry.schema import HASH_SIN_MODELO, RegistryEntry

_MXAI = """PROJECT Uno

VECTOR Entrada[1]
  x: Scalar[0, 1]
END

NETWORK Red
  INPUT Entrada
  LAYER Dense units=2 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  Entrada -> Red
END
"""


def _entrada(model_hash: str) -> RegistryEntry:
    return RegistryEntry(
        name="x", version="v1", entry_hash="sha256:" + "a" * 64,
        model_hash=model_hash, parameter_schema_hash="sha256:" + "b" * 64,
        parameter_set_id="p", input_type={}, output_type={}, metrics={},
        matrixai_version="1.6.0", created_at="2026-08-26T00:00:00Z",
        training_dataset_fingerprint="f", interpretability_level="full")


class LaEntradaSABE_SI_SePuedeEjecutarTest(unittest.TestCase):

    def test_sin_modelo_NO_es_ejecutable(self):
        self.assertFalse(_entrada(HASH_SIN_MODELO).es_ejecutable())

    def test_con_modelo_SI(self):
        self.assertTrue(_entrada("sha256:" + "c" * 64).es_ejecutable())

    def test_el_dato_NO_se_duplica_en_un_campo_nuevo(self):
        """Se deduce del `model_hash`, que ya está.

        El `entry_hash` cubre los campos de identidad: añadir uno rompería la
        cadena de las entradas ya publicadas, y tendríamos dos sitios diciendo
        lo mismo — que es como empiezan a divergir."""
        manifiesto = _entrada(HASH_SIN_MODELO).to_manifest()
        self.assertNotIn("executable", manifiesto)
        self.assertNotIn("es_ejecutable", manifiesto)
        self.assertEqual(manifiesto["model_hash"], HASH_SIN_MODELO)


class PublicarSIN_MODELO_SIGUE_PERMITIDOTest(unittest.TestCase):
    """No se prohíbe: publicar solo métricas tiene usos legítimos y quitarlo
    sería arreglar un aviso rompiendo un uso."""

    def _run(self, d: Path, *, con_modelo: bool) -> Path:
        run = d / "run"
        run.mkdir()
        (run / "evaluation_report.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}}), encoding="utf-8")
        (run / "params.best.json").write_text(
            json.dumps({"parameters": {}, "parameter_schema_hash": "sha256:" + "d" * 64}),
            encoding="utf-8")
        if con_modelo:
            (run / "model.mxai").write_text(_MXAI, encoding="utf-8")
        return run

    def test_se_publica_y_queda_MARCADA(self):
        from matrixai.registry.model_registry import ModelRegistry

        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            reg = ModelRegistry(d / "registry")
            entrada = reg.push_run_dir(self._run(d, con_modelo=False), "solo_metricas", "v1")
        self.assertEqual(entrada.model_hash, HASH_SIN_MODELO)
        self.assertFalse(entrada.es_ejecutable())

    def test_y_una_con_modelo_sale_ejecutable(self):
        """Sin esta mitad, la de arriba la pasaría un registry que marca todo
        como no ejecutable."""
        from matrixai.registry.model_registry import ModelRegistry

        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            reg = ModelRegistry(d / "registry")
            entrada = reg.push_run_dir(self._run(d, con_modelo=True), "completa", "v1")
        self.assertNotEqual(entrada.model_hash, HASH_SIN_MODELO)
        self.assertTrue(entrada.es_ejecutable())


class ElEjecutorLO_DICE_EnVezDeReventarTest(unittest.TestCase):

    def test_el_motivo_nombra_lo_que_falta_y_a_quien_le_toca(self):
        fuente = (Path(__file__).resolve().parents[1] / "matrixai" / "pipelines"
                  / "executors.py").read_text(encoding="utf-8")
        self.assertIn("es_ejecutable()", fuente)
        self.assertIn("se publicó SIN su modelo", fuente)
        # Qué se puede hacer con ella igualmente, que es lo que convierte un
        # «no se puede» en una salida.
        self.assertIn("sus métricas y sus", fuente)
