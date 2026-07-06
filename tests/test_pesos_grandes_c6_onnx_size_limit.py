# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PESOS_GRANDES C6 — ONNX guarda los pesos en línea en el protobuf; por
encima de ~2 GiB, protobuf rechaza serializar el mensaje. `OnnxExporter.export`
frena ANTES de construir el grafo con un `OnnxExportError` claro y accionable,
en vez de dejar que protobuf falle a mitad de export con un error críptico (o
peor, produzca un fichero corrupto). El chequeo usa `estimate_model_resources`
(manifest, O(#tensores)) — nunca hace falta un modelo real de 2 GiB para
probarlo: se parchea la estimación.
"""
from __future__ import annotations

import tempfile
import unittest
from importlib import util
from pathlib import Path
from unittest.mock import patch

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.mxai"


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None


def _make_program_and_params():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_EMAIL_MXAI)
    ps = build_initial_parameter_set(prog)
    return prog, ps


@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestOnnxSizeLimit(unittest.TestCase):
    def test_export_blocked_when_estimate_exceeds_limit(self):
        from matrixai.export import OnnxExportError, OnnxExporter
        from matrixai.resources import ResourceEstimate, ONNX_PROTOBUF_LIMIT_GIB
        prog, ps = _make_program_and_params()
        oversized = ResourceEstimate(
            param_count=600_000_000, weights_gib=ONNX_PROTOBUF_LIMIT_GIB + 0.5,
            vram_train_gib=0.0, effective_batch=1, json_ram_gib=0.0,
            json_disk_gib=0.0, json_time_seconds=0.0, binary_ram_gib=0.0,
            binary_disk_gib=0.0, binary_time_seconds=0.0,
        )
        with patch("matrixai.resources.estimate_model_resources", return_value=oversized):
            with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
                with self.assertRaises(OnnxExportError) as ctx:
                    OnnxExporter().export(prog, ps, f.name)
        msg = str(ctx.exception)
        self.assertIn("GiB", msg)
        self.assertIn("600,000,000", msg)
        # Auditoría C6: el mensaje NO debe sugerir bundle/wasm como salida —
        # ambos empaquetan un ONNX interno y comparten exactamente este límite.
        self.assertNotIn("Exporta como bundle", msg)
        self.assertIn("comparten este límite", msg)

    def test_export_proceeds_when_estimate_is_under_limit(self):
        """Un modelo real y pequeño (email-agent.mxai) no se ve afectado —
        el guardrail no cambia nada para lo que ya cabía en ONNX."""
        from matrixai.export import OnnxExporter
        prog, ps = _make_program_and_params()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        result = OnnxExporter().export(prog, ps, out)
        self.assertIsNotNone(result)

    def test_limit_message_gives_the_actual_estimated_size(self):
        from matrixai.export import OnnxExportError, OnnxExporter
        from matrixai.resources import ResourceEstimate
        prog, ps = _make_program_and_params()
        oversized = ResourceEstimate(
            param_count=1_200_000_000, weights_gib=4.4,
            vram_train_gib=0.0, effective_batch=1, json_ram_gib=0.0,
            json_disk_gib=0.0, json_time_seconds=0.0, binary_ram_gib=0.0,
            binary_disk_gib=0.0, binary_time_seconds=0.0,
        )
        with patch("matrixai.resources.estimate_model_resources", return_value=oversized):
            with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
                with self.assertRaises(OnnxExportError) as ctx:
                    OnnxExporter().export(prog, ps, f.name)
        msg = str(ctx.exception)
        self.assertIn("4.4", msg)
        self.assertIn("1,200,000,000", msg)

    def test_estimator_failure_fails_open_and_export_proceeds(self):
        """Auditoría C6 (BAJA): si el ESTIMADOR falla (programa exótico), el
        guardrail no debe convertir un export válido en un error — y menos con
        una excepción sin tipo que se escaparía del `except OnnxExportError`
        de los callers (500 en el Studio). Fail-open: sin estimación, sin
        chequeo — un modelo realmente grande fallará más adelante igual, solo
        que con el error de protobuf (comportamiento pre-C6)."""
        from matrixai.export import OnnxExporter
        prog, ps = _make_program_and_params()
        with patch("matrixai.resources.estimate_model_resources",
                   side_effect=RuntimeError("estimador roto")):
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                out = f.name
            result = OnnxExporter().export(prog, ps, out)
        self.assertIsNotNone(result)

    def test_bundle_and_wasm_share_the_limit_with_clear_error(self):
        """Auditoría C6: bundle y wasm empaquetan un ONNX interno
        (`OnnxExporter().export` dentro de `bundle.py`/`wasm_exporter.py`) —
        un modelo por encima del límite debe fallar en AMBOS con el mismo
        mensaje claro (envuelto en su tipo de error), nunca un crash críptico
        de protobuf a mitad de empaquetado."""
        import tempfile as _tf
        from pathlib import Path
        from matrixai.export import (
            create_edge_bundle, EdgeBundleError, export_wasm, WasmExportError,
        )
        from matrixai.resources import ResourceEstimate
        prog, ps = _make_program_and_params()
        oversized = ResourceEstimate(
            param_count=600_000_000, weights_gib=3.0,
            vram_train_gib=0.0, effective_batch=1, json_ram_gib=0.0,
            json_disk_gib=0.0, json_time_seconds=0.0, binary_ram_gib=0.0,
            binary_disk_gib=0.0, binary_time_seconds=0.0,
        )
        with patch("matrixai.resources.estimate_model_resources", return_value=oversized):
            with _tf.TemporaryDirectory() as tmp:
                import json as _json
                mxai_path = Path(tmp) / "m.mxai"
                mxai_path.write_text(_EMAIL_MXAI.read_text(encoding="utf-8"), encoding="utf-8")
                params_path = Path(tmp) / "params.best.json"
                params_path.write_text(_json.dumps(ps.to_dict()), encoding="utf-8")
                with self.assertRaises(EdgeBundleError) as ctx_b:
                    create_edge_bundle(prog, ps, mxai_path, params_path,
                                       Path(tmp) / "bundle", validate=False)
                self.assertIn("GiB", str(ctx_b.exception))
                with self.assertRaises(WasmExportError) as ctx_w:
                    export_wasm(prog, ps, Path(tmp) / "wasm", validate=False)
                self.assertIn("GiB", str(ctx_w.exception))
