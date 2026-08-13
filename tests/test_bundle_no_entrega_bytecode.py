# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Un modelo descargable no lleva el bytecode de su propia prueba de humo.

Medido el 2026-08-13 (auditoría pasada 5) sobre un export REAL del Studio
(`POST /api/studio/export {"format": "bundle"}`), listando el zip:

    DECLARADOS en el manifiesto (11): README.md, example_input.json,
      expected_output.json, export_manifest.json, inference_spec.json,
      model.mxai, model.onnx, model_manifest.json, params.best.json,
      predict.py, requirements.txt
    REALMENTE en el zip (12 + el dir): … y `__pycache__/predict.cpython-312.pyc`

De dónde salía: `_run_prediction` importa `predict.py` con importlib para
probarlo en el momento de empaquetar, y ese import escribe el `__pycache__`
dentro del staging, que después se copiaba entero al bundle.

Por qué se limpia en el core y no en quien empaqueta: las DOS rutas de export
del Studio recortaban distinto —el zip por `make_archive` lo metía, el zip
streaming de pesos grandes lo dejaba fuera por filtrar solo ficheros sueltos—,
así que el mismo modelo salía con ficheros distintos según su tamaño. Y la
lista `files` del resultado tampoco lo declaraba: el paquete llevaba algo que
su propio manifiesto no nombraba, con la versión del intérprete de la máquina
que lo empaquetó dentro.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    return util.find_spec("onnxruntime") is not None


@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestElBundleNoLlevaBytecode(unittest.TestCase):
    def setUp(self) -> None:
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set

        base = Path(__file__).parent.parent
        self.mxai = base / "examples" / "fall-risk.mxai"
        self.prog = parse_file(self.mxai)
        self.ps = build_initial_parameter_set(self.prog)
        self.td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.td, True)
        self.params = Path(self.td) / "params.json"
        write_parameter_set(str(self.params), self.ps)

    def _bundle(self):
        from matrixai.export import create_edge_bundle

        return create_edge_bundle(
            self.prog, self.ps, mxai_path=str(self.mxai),
            params_path=str(self.params),
            outdir=str(Path(self.td) / "b"), validate=False,
        )

    def test_la_prueba_de_humo_se_ejecuto_de_verdad(self) -> None:
        # Si no, este banco no tendría dientes: sin `expected_output.json` no
        # se ha importado `predict.py` y no habría bytecode que limpiar.
        result = self._bundle()
        self.assertIn("expected_output.json", result.files)

    def test_no_queda_ningun_pycache_en_el_bundle(self) -> None:
        result = self._bundle()
        bd = Path(result.bundle_dir)
        sobrantes = sorted(str(p.relative_to(bd)) for p in bd.rglob("__pycache__"))
        self.assertEqual(sobrantes, [], f"el bundle entrega bytecode: {sobrantes}")

    def test_no_se_entrega_ningun_pyc(self) -> None:
        result = self._bundle()
        bd = Path(result.bundle_dir)
        pycs = sorted(str(p.relative_to(bd)) for p in bd.rglob("*.pyc"))
        self.assertEqual(pycs, [], f"el bundle entrega bytecode: {pycs}")

    def test_lo_que_hay_en_disco_es_LO_QUE_DECLARA_el_resultado(self) -> None:
        """`files` nombraba solo los ficheros sueltos, así que un directorio
        colado dentro no aparecía por ningún lado. Ahora coinciden."""
        result = self._bundle()
        bd = Path(result.bundle_dir)
        en_disco = sorted(str(p.relative_to(bd)) for p in bd.rglob("*") if p.is_file())
        self.assertEqual(en_disco, sorted(result.files))


if __name__ == "__main__":
    unittest.main()
