# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""WASM bundle: packages model.onnx + wasm_manifest.json + predict.js for ONNX Runtime Web.

Route: ONNX → ONNX Runtime Web (ORT Web WASM backend).
The `.onnx` model is the entry point; ORT Web provides the WASM runtime.
No WASM compilation toolchain required.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrixai.ir import MatrixAIProgram
from matrixai.parameters.store import ParameterSet
from matrixai.export.onnx_exporter import (
    OnnxExporter,
    OnnxExportResult,
    OnnxExportError,
    validate_export_parameter_set,
    onnx_size_limit_error,
)
from matrixai.export.equivalence import (
    OnnxEquivalenceResult,
    OnnxEquivalenceValidator,
    ort_available,
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    _DEFAULT_N_SAMPLES,
)

WASM_RUNTIME = "onnxruntime-web"
ORT_WEB_MIN_VERSION = "1.14"
_BUNDLE_FILES = {"model.onnx", "wasm_manifest.json", "predict.js"}


class WasmExportError(ValueError):
    pass


@dataclass(frozen=True)
class WasmExportResult:
    bundle_dir: str
    files: list[str]
    model_hash: str
    parameter_set_id: str
    parameter_schema_hash: str
    input_name: str
    input_shape: list[int]
    output_name: str
    output_shape: list[int]
    wasm_runtime: str
    opset_version: int
    equivalence_result: OnnxEquivalenceResult | None = None

    @property
    def equivalence_passed(self) -> bool:
        return self.equivalence_result is not None and self.equivalence_result.passed

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bundle_dir": self.bundle_dir,
            "files": self.files,
            "model_hash": self.model_hash,
            "parameter_set_id": self.parameter_set_id,
            "parameter_schema_hash": self.parameter_schema_hash,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "output_name": self.output_name,
            "output_shape": self.output_shape,
            "wasm_runtime": self.wasm_runtime,
            "opset_version": self.opset_version,
            "equivalence_passed": self.equivalence_passed,
        }
        if self.equivalence_result is not None:
            d["equivalence_check"] = self.equivalence_result.to_dict()
        return d


class WasmExporter:
    """Exports a MatrixAI model as an ONNX Runtime Web (WASM) deployment bundle."""

    def export(
        self,
        program: MatrixAIProgram,
        parameter_set: ParameterSet,
        output_dir: str | Path,
        *,
        validate: bool = True,
        atol: float = _DEFAULT_ATOL,
        rtol: float = _DEFAULT_RTOL,
        n_samples: int = _DEFAULT_N_SAMPLES,
        force: bool = False,
    ) -> WasmExportResult:
        # PESOS_GRANDES C7b: `OnnxExporter.export` ya NO bloquea un modelo
        # grande (usa external-data, un .onnx.data aparte) — pero un
        # navegador SÍ sigue sin poder cargar varios GiB en un fichero aparte
        # del .wasm, así que WASM (a diferencia de onnx/bundle) mantiene el
        # bloqueo explícito de C6, aquí, antes de tocar disco.
        size_error = onnx_size_limit_error(program)
        if size_error is not None:
            raise WasmExportError(size_error)

        output_dir = Path(output_dir)
        if output_dir.exists() and not force:
            raise WasmExportError(
                f"Bundle directory {output_dir} already exists. "
                "Pass force=True to overwrite."
            )

        # Validate ParameterSet shapes/schema before touching disk
        val = validate_export_parameter_set(program, parameter_set)
        if not val.ok:
            raise WasmExportError(
                f"ParameterSet validation failed: {'; '.join(val.errors)}"
            )

        # Build in a temp dir; rename to output_dir only when everything succeeds.
        tmp_parent = output_dir.parent
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent, prefix=".wasm_tmp_") as _tmp:
            work = Path(_tmp)

            # 1. Export ONNX (reuses full OnnxExporter including hash validation)
            onnx_path = work / "model.onnx"
            try:
                onnx_result = OnnxExporter().export(program, parameter_set, onnx_path)
            except OnnxExportError as exc:
                raise WasmExportError(f"ONNX export failed: {exc}") from exc

            # 2. Optional equivalence check
            eq_result: OnnxEquivalenceResult | None = None
            if validate:
                if not ort_available():
                    raise WasmExportError(
                        "Equivalence validation requires 'onnxruntime'. "
                        "Install with: pip install matrixai-core[export]"
                    )
                eq_result = OnnxEquivalenceValidator().validate(
                    program, parameter_set, onnx_path,
                    atol=atol, rtol=rtol, n_samples=n_samples,
                )
                if not eq_result.passed:
                    raise WasmExportError(
                        f"Equivalence check FAILED: max_abs_diff={eq_result.max_abs_diff:.2e} "
                        f"exceeds tolerance atol={atol:.0e} + rtol={rtol:.0e}. "
                        "Bundle not created."
                    )

            # 3. wasm_manifest.json
            _write_wasm_manifest(onnx_result, eq_result, work / "wasm_manifest.json", atol, rtol)

            # 4. predict.js
            (work / "predict.js").write_text(
                _build_predict_js(program, onnx_result),
                encoding="utf-8",
            )

            # Atomic promotion
            if output_dir.exists():
                shutil.rmtree(str(output_dir))
            shutil.copytree(str(work), str(output_dir))

        files = sorted(str(p.relative_to(output_dir)) for p in output_dir.iterdir() if p.is_file())

        return WasmExportResult(
            bundle_dir=str(output_dir),
            files=files,
            model_hash=onnx_result.model_hash,
            parameter_set_id=onnx_result.parameter_set_id,
            parameter_schema_hash=onnx_result.parameter_schema_hash,
            input_name=onnx_result.input_name,
            input_shape=onnx_result.input_shape,
            output_name=onnx_result.output_name,
            output_shape=onnx_result.output_shape,
            wasm_runtime=WASM_RUNTIME,
            opset_version=onnx_result.opset_version,
            equivalence_result=eq_result,
        )


def export_wasm(
    program: MatrixAIProgram,
    parameter_set: ParameterSet,
    output_dir: str | Path,
    *,
    validate: bool = True,
    force: bool = False,
) -> WasmExportResult:
    return WasmExporter().export(
        program, parameter_set, output_dir, validate=validate, force=force,
    )


# ---------------------------------------------------------------------------
# Manifest and JS builders
# ---------------------------------------------------------------------------

def _write_wasm_manifest(
    onnx_result: OnnxExportResult,
    eq_result: OnnxEquivalenceResult | None,
    path: Path,
    atol: float,
    rtol: float,
) -> None:
    data = {
        "model_hash": onnx_result.model_hash,
        "parameter_schema_hash": onnx_result.parameter_schema_hash,
        "parameter_set_id": onnx_result.parameter_set_id,
        "format": "wasm-onnxruntime-web",
        "wasm_runtime": WASM_RUNTIME,
        "ort_web_min_version": ORT_WEB_MIN_VERSION,
        "onnx_opset": onnx_result.opset_version,
        "input_name": onnx_result.input_name,
        "input_shape": onnx_result.input_shape,
        "output_name": onnx_result.output_name,
        "output_shape": onnx_result.output_shape,
        "exported_functions": onnx_result.exported_functions,
        "tolerance": {"atol": atol, "rtol": rtol} if eq_result is not None else None,
        "equivalence_check": eq_result.to_dict() if eq_result is not None else None,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_predict_js(program: MatrixAIProgram, onnx_result: OnnxExportResult) -> str:
    input_name = onnx_result.input_name
    output_name = onnx_result.output_name
    in_shape = onnx_result.input_shape  # e.g. [-1, 8]
    out_shape = onnx_result.output_shape  # e.g. [-1, 2]

    # Determine input tensor dtype (int64 for SEQUENCE, float32 for VECTOR)
    use_int64 = bool(program.sequences)
    if use_int64:
        js_dtype = "'int64'"
        js_arr_type = "BigInt64Array.from(inputIds.map(BigInt))"
        input_comment = f"inputIds: integer token IDs, length {in_shape[-1]}"
    else:
        js_dtype = "'float32'"
        js_arr_type = "Float32Array.from(inputData)"
        input_comment = f"inputData: float32 array, length {in_shape[-1]}"

    input_arg = "inputIds" if use_int64 else "inputData"
    shape_str = f"[1, {in_shape[-1]}]"
    out_shape_str = str(out_shape).replace("-1", "N")

    return f"""// MatrixAI WASM inference — ONNX Runtime Web
// Requires: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_WEB_MIN_VERSION}/dist/ort.min.js"></script>
// Project: {program.project}
// Input:  '{input_name}' {shape_str} — {input_comment}
// Output: '{output_name}' {out_shape_str}

async function predict({input_arg}) {{
  const session = await ort.InferenceSession.create('./model.onnx');
  const inputTensor = new ort.Tensor({js_dtype}, {js_arr_type}, {shape_str});
  const feeds = {{ '{input_name}': inputTensor }};
  const results = await session.run(feeds);
  return Array.from(results['{output_name}'].data);
}}

// Integrity check
const MODEL_HASH = '{onnx_result.model_hash}';
const PARAMETER_SET_ID = '{onnx_result.parameter_set_id}';
"""
