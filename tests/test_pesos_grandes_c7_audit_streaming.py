# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PESOS_GRANDES C7 auditoría — export ONNX external-data por STREAMING desde
el `.mxw`, sin traer los pesos a RAM ni copiarlos.

La auditoría de Roberto encontró que C7b, aunque evita `.tolist()`, seguía
leyendo el `.mxw` entero (`read_mxw`) y copiando a `numpy_helper.from_array`
(→ `TensorProto.raw_data`), duplicando pesos antes de externalizar: para un
4B (15 GiB) el pico rondaría 30+ GiB. `export_dense_onnx_graph_external` +
`stream_mxw_tensor` construyen el grafo desde SOLO la cabecera y streamean los
blobs — nunca hay más de un chunk en RAM.
"""
from __future__ import annotations

import tempfile
import unittest
import zipfile
from importlib import util
from pathlib import Path

_HAS_TORCH = util.find_spec("torch") is not None
_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None

MXAI = """PROJECT P
VECTOR In[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=6 activation=relu
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""


def _setup_mxw(tmp: Path):
    import torch
    from matrixai.parser import parse_text
    from matrixai.forward.dense_torch import build_parameter_template_for_state
    from matrixai.parameters.binary_store import write_mxw
    prog = parse_text(MXAI)
    net, template = build_parameter_template_for_state(prog)
    torch.manual_seed(5)
    state = {
        "Net.W1": torch.randn(6, 3), "Net.b1": torch.randn(6),
        "Net.W2": torch.randn(4, 6), "Net.b2": torch.randn(4),
        "Net.W3": torch.randn(2, 4), "Net.b3": torch.randn(2),
    }
    mxw = tmp / "m.mxw"
    write_mxw(mxw, state, model_hash=template.model_hash,
              parameter_schema_hash=template.parameter_schema_hash)
    return prog, net, template, state, mxw


@unittest.skipUnless(_HAS_TORCH and _HAS_ONNX, "torch/onnx not installed")
class TestStreamingExternalExport(unittest.TestCase):
    def test_graph_only_never_reads_body(self):
        """`export_dense_onnx_graph_external` NO debe leer el cuerpo del `.mxw`
        (patcheamos `read_mxw` para que reviente): solo la cabecera."""
        from unittest.mock import patch
        from matrixai.parameters import binary_store
        from matrixai.parameters.binary_store import read_mxw_header
        from matrixai.export import export_dense_onnx_graph_external
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prog, net, template, state, mxw = _setup_mxw(tmp)
            header = read_mxw_header(mxw)
            with patch.object(binary_store, "read_mxw",
                              side_effect=AssertionError("NO debe leer el cuerpo")):
                result, ordered = export_dense_onnx_graph_external(
                    prog, header, tmp / "model.onnx",
                    model_hash=template.model_hash,
                    parameter_schema_hash=template.parameter_schema_hash,
                )
            self.assertTrue(result.external_data)
            self.assertEqual([m["path"] for m in ordered],
                             ["Net.W1", "Net.b1", "Net.W2", "Net.b2", "Net.W3", "Net.b3"])
            # model.onnx es SOLO el grafo — minúsculo, sin los pesos embebidos.
            self.assertLess((tmp / "model.onnx").stat().st_size, 4096)

    def test_streamed_offsets_match_and_ort_equals_torch(self):
        """El zip streameado (grafo + `.onnx.data`) carga en onnxruntime y da
        la MISMA salida que el forward torch del mismo `.mxw`."""
        if not _HAS_ORT:
            self.skipTest("onnxruntime not installed")
        import numpy as np
        import onnxruntime as ort
        from matrixai.parameters.binary_store import read_mxw_header_and_body_start, stream_mxw_tensor
        from matrixai.export import export_dense_onnx_graph_external
        from matrixai.forward.dense_torch import (
            dense_network_to_torch_module_from_state, dense_torch_forward,
        )
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prog, net, template, state, mxw = _setup_mxw(tmp)
            header, body_start = read_mxw_header_and_body_start(mxw)
            onnx_path = tmp / "model.onnx"
            result, ordered = export_dense_onnx_graph_external(
                prog, header, onnx_path,
                model_hash=template.model_hash,
                parameter_schema_hash=template.parameter_schema_hash,
            )
            zip_path = tmp / "out.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                zf.write(onnx_path, "model.onnx")
                with zf.open("model.onnx.data", "w") as data_out, open(mxw, "rb") as mf:
                    for meta in ordered:
                        stream_mxw_tensor(mf, body_start, meta, data_out)
                # ZIP_STORED: sin compresión (comprobamos que ningún entry se comprimió).
                for info in zf.infolist():
                    self.assertEqual(info.compress_type, zipfile.ZIP_STORED)

            extract = tmp / "ex"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract)
            sess = ort.InferenceSession(str(extract / "model.onnx"))
            x = np.array([[0.3, -0.5, 0.8]], dtype=np.float32)
            onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]

            module = dense_network_to_torch_module_from_state(net, state, "cpu")
            torch_out = dense_torch_forward(module, [0.3, -0.5, 0.8], "cpu")
            self.assertTrue(np.allclose(onnx_out, torch_out, atol=1e-5))

    def test_wrong_model_hash_rejected(self):
        from matrixai.parameters.binary_store import read_mxw_header
        from matrixai.export import export_dense_onnx_graph_external, OnnxExportError
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prog, net, template, state, mxw = _setup_mxw(tmp)
            header = read_mxw_header(mxw)
            with self.assertRaises(OnnxExportError):
                export_dense_onnx_graph_external(
                    prog, header, tmp / "m.onnx",
                    model_hash="mxai_wrong", parameter_schema_hash=template.parameter_schema_hash,
                )


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestStreamMxwTensor(unittest.TestCase):
    def test_stream_matches_raw_bytes(self):
        import torch
        from matrixai.parameters.binary_store import (
            write_mxw, read_mxw_header_and_body_start, stream_mxw_tensor,
        )
        import io as _io
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            state = {"W": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
            mxw = tmp / "t.mxw"
            write_mxw(mxw, state, model_hash="mh", parameter_schema_hash="sh")
            header, body_start = read_mxw_header_and_body_start(mxw)
            meta = header["tensors"][0]
            out = _io.BytesIO()
            with open(mxw, "rb") as f:
                n = stream_mxw_tensor(f, body_start, meta, out, chunk_bytes=7)
            import numpy as np
            arr = np.frombuffer(out.getvalue(), dtype=np.float32)
            self.assertEqual(n, 48)  # 12 float32
            self.assertTrue(np.array_equal(arr, np.arange(12, dtype=np.float32)))
