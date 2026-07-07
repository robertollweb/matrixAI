# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PESOS_GRANDES C6→C7b — ONNX guarda los pesos en línea en el protobuf; por
encima de ~2 GiB, protobuf rechaza serializar el mensaje. C6 bloqueaba el
export con un error claro; C7b lo resuelve de verdad con "external data" (los
pesos van a un `.onnx.data` aparte, formato estándar que onnxruntime ya sabe
leer) — solo WASM (un navegador no puede cargar un `.data` de varios GiB
aparte) mantiene el bloqueo de C6. El chequeo usa `estimate_model_resources`
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


def _oversized_estimate(param_count=600_000_000, weights_gib=3.0):
    from matrixai.resources import ResourceEstimate
    return ResourceEstimate(
        param_count=param_count, weights_gib=weights_gib,
        vram_train_gib=0.0, effective_batch=1, json_ram_gib=0.0,
        json_disk_gib=0.0, json_time_seconds=0.0, binary_ram_gib=0.0,
        binary_disk_gib=0.0, binary_time_seconds=0.0,
    )


@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestOnnxSizeLimit(unittest.TestCase):
    def test_oversized_export_uses_external_data_instead_of_blocking(self):
        """C7b: un modelo por encima del límite YA NO bloquea el export de
        onnx — usa external-data (`.onnx` pequeño + `.onnx.data` con los
        pesos, formato estándar de la industria)."""
        from matrixai.export import OnnxExporter
        prog, ps = _make_program_and_params()
        with patch("matrixai.resources.estimate_model_resources", return_value=_oversized_estimate()):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "model.onnx"
                result = OnnxExporter().export(prog, ps, out)
                self.assertTrue(result.external_data)
                self.assertTrue(out.exists())
                data_file = Path(tmp) / "model.onnx.data"
                self.assertTrue(data_file.exists(), "esperaba un model.onnx.data externo")
                self.assertGreater(data_file.stat().st_size, 0)

    def test_export_proceeds_when_estimate_is_under_limit(self):
        """Un modelo real y pequeño (email-agent.mxai) no se ve afectado —
        el guardrail no cambia nada para lo que ya cabía en ONNX: sin
        external-data, comportamiento byte-idéntico a antes de C6/C7b."""
        from matrixai.export import OnnxExporter
        prog, ps = _make_program_and_params()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            result = OnnxExporter().export(prog, ps, out)
            self.assertFalse(result.external_data)
            self.assertFalse((Path(tmp) / "model.onnx.data").exists())

    def test_estimator_failure_fails_open_no_external_data(self):
        """Auditoría C6 (BAJA), sigue aplicando en C7b: si el ESTIMADOR falla
        (programa exótico), el guardrail no debe convertir un export válido
        en un error — y menos con una excepción sin tipo que se escaparía del
        `except OnnxExportError` de los callers (500 en el Studio). Fail-open:
        sin estimación, sin external-data — comportamiento normal."""
        from matrixai.export import OnnxExporter
        prog, ps = _make_program_and_params()
        with patch("matrixai.resources.estimate_model_resources",
                   side_effect=RuntimeError("estimador roto")):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "model.onnx"
                result = OnnxExporter().export(prog, ps, out)
        self.assertFalse(result.external_data)

    def test_wasm_still_blocks_oversized_models_with_clear_error(self):
        """WASM es el ÚNICO formato que sigue bloqueando: un navegador no
        puede cargar un `.onnx.data` de varios GiB aparte del `.wasm`."""
        from matrixai.export import export_wasm, WasmExportError
        prog, ps = _make_program_and_params()
        with patch("matrixai.resources.estimate_model_resources", return_value=_oversized_estimate()):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(WasmExportError) as ctx:
                    export_wasm(prog, ps, Path(tmp) / "wasm", validate=False)
        msg = str(ctx.exception)
        self.assertIn("GiB", msg)
        self.assertIn("600,000,000", msg)
        # C7b: el mensaje ya no dice que onnx/bundle comparten el límite de
        # wasm — ya NO lo comparten (usan external-data automáticamente).
        self.assertIn("external", msg.lower())

    def test_bundle_no_longer_blocked_uses_external_data(self):
        """C7b: bundle (que empaqueta un ONNX interno) tampoco bloquea ya un
        modelo grande — el zip lleva model.onnx + model.onnx.data."""
        from matrixai.export import create_edge_bundle
        prog, ps = _make_program_and_params()
        with patch("matrixai.resources.estimate_model_resources", return_value=_oversized_estimate()):
            with tempfile.TemporaryDirectory() as tmp:
                import json as _json
                mxai_path = Path(tmp) / "m.mxai"
                mxai_path.write_text(_EMAIL_MXAI.read_text(encoding="utf-8"), encoding="utf-8")
                params_path = Path(tmp) / "params.best.json"
                params_path.write_text(_json.dumps(ps.to_dict()), encoding="utf-8")
                result = create_edge_bundle(
                    prog, ps, mxai_path, params_path, Path(tmp) / "bundle",
                    validate=False,
                )
        self.assertTrue(result.export_result.external_data)
        self.assertIn("model.onnx.data", result.files)
