# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 59 C5 — coherencia del rango en el export: desde el contrato 59
el target de regresión se entrena normalizado a [0,1] (ver
59_REGRESION_QUE_APRENDE_CONTRACT.md); el bundle descargable debe
desnormalizar la salida IGUAL que `_studio_infer` (invariante del contrato
42, "predice == Studio"), o un modelo descargado daría una fracción [0,1]
en vez del valor real en escala de dominio.

Test mecánico (no necesita entrenar de verdad — la desnormalización es una
transformación afín fija, no depende de si la red aprendió): con pesos
iniciales deterministas, `predict.py` debe devolver
`onnx_raw * (hi - lo) + lo` cuando el `inference_spec.json` trae `range`, y
el valor crudo sin tocar cuando no lo trae (retrocompat con un bundle
exportado antes de este contrato)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path

_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None

_MXAI = """PROJECT KelvinLike
VECTOR Input[1]
  centigrados: Scalar
END
NETWORK Regressor
  INPUT Input
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT predicted_value: Scalar
END
GRAPH
  Input -> Regressor
END
"""


def _load_predict(bundle_dir: Path):
    spec = util.spec_from_file_location("predict_c59_c5", str(bundle_dir / "predict.py"))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MatrixAIModel(str(bundle_dir / "inference_spec.json"))
    return model, module


@unittest.skipUnless(_HAS_ONNX and _HAS_ORT, "onnx + onnxruntime required")
class TestExportDenormalizesRegressionOutput(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)

    def _bundle(self, **meta):
        from matrixai.parser import parse_text
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        from matrixai.export import create_edge_bundle
        prog = parse_text(_MXAI)
        ps = build_initial_parameter_set(prog)
        params = self.td / "params.json"
        write_parameter_set(str(params), ps)
        mxai_path = self.td / "model.mxai"
        mxai_path.write_text(_MXAI, encoding="utf-8")
        result = create_edge_bundle(
            prog, ps, mxai_path=str(mxai_path), params_path=str(params),
            outdir=str(self.td / "bundle"), validate=False,
            field_ranges={"centigrados": (0.0, 99.0)},
            **meta,
        )
        self.assertIsNone(result.inference_spec_skipped_reason)
        self.bundle_dir = Path(result.bundle_dir)
        return _load_predict(self.bundle_dir)

    def test_inference_spec_carries_the_target_range(self):
        import json
        self._bundle(target_range=(263.25, 382.05))
        spec = json.loads((self.bundle_dir / "inference_spec.json").read_text())
        self.assertEqual(spec["output"]["kind"], "regression")
        self.assertEqual(spec["output"]["range"], [263.25, 382.05])

    def test_predict_denormalizes_when_range_is_present(self):
        model, _module = self._bundle(target_range=(263.25, 382.05))
        raw_onnx = model.session.run(None, {model.input_name: __import__("numpy").array(
            [model._encode({"centigrados": 50})[0]], dtype="float32",
        )})[0][0][0]
        predicted = model.predict({"centigrados": 50})
        lo, hi = 263.25, 382.05
        expected = float(raw_onnx) * (hi - lo) + lo
        self.assertAlmostEqual(predicted, expected, places=4)
        # Sanity: en escala de dominio, no en [0,1] como si fuera probabilidad.
        self.assertGreater(predicted, 1.0)

    def test_predict_stays_raw_without_a_target_range(self):
        """Retrocompat (decisión B del contrato): un bundle exportado SIN
        `target_range` (modelo de antes de este contrato) sigue devolviendo
        el valor crudo — nunca un error nuevo, nunca inventa un rango."""
        model, _module = self._bundle()  # sin target_range
        import json
        spec = json.loads((self.bundle_dir / "inference_spec.json").read_text())
        self.assertNotIn("range", spec["output"])

        import numpy as np
        raw_onnx = model.session.run(None, {model.input_name: np.array(
            [model._encode({"centigrados": 50})[0]], dtype="float32",
        )})[0][0][0]
        predicted = model.predict({"centigrados": 50})
        self.assertAlmostEqual(predicted, float(raw_onnx), places=6)


if __name__ == "__main__":
    unittest.main()
