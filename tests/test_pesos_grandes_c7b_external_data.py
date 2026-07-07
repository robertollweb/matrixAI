# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PESOS_GRANDES C7b — `OnnxExporter.export(..., state_dict=...)`: exportar un
modelo grande (`.mxw`) directo desde tensores torch crudos, sin pasar por una
`ParameterSet` con valores (nunca `.tolist()`). Verifica paridad con el
camino clásico y que el resultado con external-data forzado (`ONNX_PROTOBUF_
LIMIT_GIB` parcheado a 0) sigue siendo correcto para onnxruntime.
"""
from __future__ import annotations

import tempfile
import unittest
from importlib import util
from pathlib import Path
from unittest.mock import patch

_HAS_TORCH = util.find_spec("torch") is not None
_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None

MXAI = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""


def _setup():
    import torch
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(MXAI)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    ps = build_network_parameter_set(net, rl, program_hash(prog), seed=1)
    state = {
        key: torch.tensor(param["values"], dtype=torch.float32)
        for key, param in ps.parameters.items()
    }
    return prog, net, ps, state


@unittest.skipUnless(_HAS_TORCH and _HAS_ONNX, "torch/onnx not installed")
class TestStateDictExport(unittest.TestCase):
    def test_state_dict_export_never_calls_tolist(self):
        """La prueba directa de "nunca materializa": si el camino state_dict
        llamara a `.tolist()` en algún tensor (el `.tolist()` O(#params) que
        el resto de PESOS_GRANDES evita), esto lo pillaría."""
        import torch
        from matrixai.export import OnnxExporter
        prog, net, ps, state = _setup()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            with patch.object(torch.Tensor, "tolist",
                              side_effect=AssertionError("NO debe llamar a .tolist()")):
                result = OnnxExporter().export(
                    prog, None, out,
                    state_dict=state, model_hash=ps.model_hash,
                    parameter_schema_hash=ps.parameter_schema_hash,
                )
        self.assertIsNotNone(result)
        self.assertEqual(result.parameter_set_id, "torch_state")

    def test_state_dict_export_matches_classic_parameter_set_path(self):
        """Paridad: el MISMO estado exportado vía state_dict (tensores) o vía
        ParameterSet (valores, camino clásico) debe dar el mismo ONNX
        (mismo checksum de comportamiento — verificado con onnxruntime)."""
        if not _HAS_ORT:
            self.skipTest("onnxruntime not installed")
        import numpy as np
        import onnxruntime as ort
        from matrixai.export import OnnxExporter
        prog, net, ps, state = _setup()
        with tempfile.TemporaryDirectory() as tmp:
            out_state = Path(tmp) / "state.onnx"
            out_classic = Path(tmp) / "classic.onnx"
            OnnxExporter().export(
                prog, None, out_state,
                state_dict=state, model_hash=ps.model_hash,
                parameter_schema_hash=ps.parameter_schema_hash,
            )
            OnnxExporter().export(prog, ps, out_classic)

            x = np.array([[0.3, 0.7]], dtype=np.float32)
            sess_state = ort.InferenceSession(str(out_state))
            sess_classic = ort.InferenceSession(str(out_classic))
            out1 = sess_state.run(None, {sess_state.get_inputs()[0].name: x})[0]
            out2 = sess_classic.run(None, {sess_classic.get_inputs()[0].name: x})[0]
        self.assertTrue(np.allclose(out1, out2, atol=1e-5))

    def test_wrong_model_hash_rejected(self):
        from matrixai.export import OnnxExporter, OnnxExportError
        prog, net, ps, state = _setup()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            with self.assertRaises(OnnxExportError) as ctx:
                OnnxExporter().export(
                    prog, None, out,
                    state_dict=state, model_hash="mxai_wronghash0000",
                    parameter_schema_hash=ps.parameter_schema_hash,
                )
        self.assertIn("model_hash", str(ctx.exception))

    def test_missing_hashes_rejected(self):
        from matrixai.export import OnnxExporter, OnnxExportError
        prog, net, ps, state = _setup()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            with self.assertRaises(OnnxExportError):
                OnnxExporter().export(prog, None, out, state_dict=state)

    def test_composite_program_rejects_state_dict(self):
        """El camino state_dict solo soporta dense_network (lo único que
        PESOS_GRANDES guarda en .mxw) — un composite debe rechazarse claro."""
        from matrixai.playground import analyze_playground_request, _network_is_composite
        from matrixai.parser import parse_text
        from matrixai.parameters.store import program_hash
        from matrixai.export import OnnxExporter, OnnxExportError
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar riesgo en 3 niveles con red profunda de bloques "
                      "residuales con LayerNorm y Dropout, 8 features clínicas",
        })
        mxai = r["mxai"]
        self.assertTrue(_network_is_composite(mxai))
        prog = parse_text(mxai)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            with self.assertRaises(OnnxExportError) as ctx:
                OnnxExporter().export(
                    prog, None, out,
                    state_dict={"anything": None}, model_hash=program_hash(prog),
                    parameter_schema_hash="y",
                )
        self.assertIn("dense_network", str(ctx.exception))

    def test_forced_external_data_still_correct_with_ort(self):
        """Con `ONNX_PROTOBUF_LIMIT_GIB` parcheado a 0 (fuerza external-data
        incluso en un modelo mini), el .onnx + .onnx.data resultante debe
        seguir siendo válido y dar la MISMA predicción que sin external-data."""
        if not _HAS_ORT:
            self.skipTest("onnxruntime not installed")
        import numpy as np
        import onnxruntime as ort
        from matrixai.export import OnnxExporter
        prog, net, ps, state = _setup()
        with tempfile.TemporaryDirectory() as tmp:
            out_ext = Path(tmp) / "ext.onnx"
            with patch("matrixai.resources.ONNX_PROTOBUF_LIMIT_GIB", 0.0):
                result = OnnxExporter().export(
                    prog, None, out_ext,
                    state_dict=state, model_hash=ps.model_hash,
                    parameter_schema_hash=ps.parameter_schema_hash,
                )
            self.assertTrue(result.external_data)
            self.assertTrue((Path(tmp) / "ext.onnx.data").exists())

            out_normal = Path(tmp) / "normal.onnx"
            OnnxExporter().export(
                prog, None, out_normal,
                state_dict=state, model_hash=ps.model_hash,
                parameter_schema_hash=ps.parameter_schema_hash,
            )

            x = np.array([[0.3, 0.7]], dtype=np.float32)
            sess_ext = ort.InferenceSession(str(out_ext))
            sess_normal = ort.InferenceSession(str(out_normal))
            out1 = sess_ext.run(None, {sess_ext.get_inputs()[0].name: x})[0]
            out2 = sess_normal.run(None, {sess_normal.get_inputs()[0].name: x})[0]
        self.assertTrue(np.allclose(out1, out2, atol=1e-6))
