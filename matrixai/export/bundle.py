# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Edge bundle: packages .mxai + params + model.onnx + manifests into a deployable directory."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrixai.ir import MatrixAIProgram
from matrixai.export.onnx_exporter import validate_export_parameter_set
from matrixai.parameters.store import ParameterSet, program_hash
from matrixai.export.onnx_exporter import OnnxExporter, OnnxExportResult, OnnxExportError
from matrixai.export.equivalence import (
    OnnxEquivalenceResult,
    OnnxEquivalenceValidator,
    write_export_manifest,
    ort_available,
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    _DEFAULT_N_SAMPLES,
)


class EdgeBundleError(ValueError):
    pass


@dataclass(frozen=True)
class EdgeBundleResult:
    bundle_dir: str
    files: list[str]
    model_hash: str
    parameter_set_id: str
    export_result: OnnxExportResult
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
            "equivalence_passed": self.equivalence_passed,
            "export": self.export_result.to_dict(),
        }
        if self.equivalence_result is not None:
            d["equivalence_check"] = self.equivalence_result.to_dict()
        return d


class EdgeBundler:
    """Creates a self-contained edge bundle directory from a .mxai + ParameterSet."""

    def bundle(
        self,
        program: MatrixAIProgram,
        parameter_set: ParameterSet,
        mxai_path: str | Path,
        params_path: str | Path,
        outdir: str | Path,
        *,
        validate: bool = True,
        atol: float = _DEFAULT_ATOL,
        rtol: float = _DEFAULT_RTOL,
        n_samples: int = _DEFAULT_N_SAMPLES,
        force: bool = False,
    ) -> EdgeBundleResult:
        outdir = Path(outdir)
        if outdir.exists() and not force:
            raise EdgeBundleError(
                f"Bundle directory {outdir} already exists. "
                "Pass force=True to overwrite."
            )

        # Validate ParameterSet shapes/schema before touching disk
        val = validate_export_parameter_set(program, parameter_set)
        if not val.ok:
            raise EdgeBundleError(
                f"ParameterSet validation failed: {'; '.join(val.errors)}"
            )

        # Build in a temp dir; rename to outdir only when everything succeeds.
        tmp_parent = outdir.parent
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent, prefix=".bundle_tmp_") as _tmp:
            work = Path(_tmp)

            # 1. Copy .mxai and params
            shutil.copy2(str(mxai_path), str(work / "model.mxai"))
            shutil.copy2(str(params_path), str(work / "params.best.json"))

            # 2. Export ONNX
            onnx_dest = work / "model.onnx"
            try:
                export_result = OnnxExporter().export(program, parameter_set, onnx_dest)
            except OnnxExportError as exc:
                raise EdgeBundleError(f"ONNX export failed: {exc}") from exc

            # 3. Equivalence validation
            eq_result: OnnxEquivalenceResult | None = None
            if validate:
                if not ort_available():
                    raise EdgeBundleError(
                        "Equivalence validation requires 'onnxruntime'. "
                        "Install with: pip install matrixai-core[export]"
                    )
                eq_result = OnnxEquivalenceValidator().validate(
                    program, parameter_set, onnx_dest,
                    atol=atol, rtol=rtol, n_samples=n_samples,
                )
                if not eq_result.passed:
                    raise EdgeBundleError(
                        f"Equivalence check FAILED: max_abs_diff={eq_result.max_abs_diff:.2e} "
                        f"exceeds tolerance atol={atol:.0e} + rtol={rtol:.0e}. "
                        "Bundle not created."
                    )

            # 4. model_manifest.json
            (work / "model_manifest.json").write_text(
                json.dumps(_build_model_manifest(program, parameter_set), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

            # 5. export_manifest.json
            if eq_result is not None:
                write_export_manifest(export_result, eq_result, work / "export_manifest.json")
            else:
                _write_export_manifest_no_eq(export_result, work / "export_manifest.json")

            # 6. README.md
            (work / "README.md").write_text(
                _build_readme(program, export_result, eq_result),
                encoding="utf-8",
            )

            # Atomic promotion: remove stale outdir then rename temp into place
            if outdir.exists():
                shutil.rmtree(str(outdir))
            shutil.copytree(str(work), str(outdir))

        files = sorted(str(p.relative_to(outdir)) for p in outdir.iterdir() if p.is_file())

        return EdgeBundleResult(
            bundle_dir=str(outdir),
            files=files,
            model_hash=parameter_set.model_hash,
            parameter_set_id=parameter_set.parameter_set_id,
            export_result=export_result,
            equivalence_result=eq_result,
        )


def create_edge_bundle(
    program: MatrixAIProgram,
    parameter_set: ParameterSet,
    mxai_path: str | Path,
    params_path: str | Path,
    outdir: str | Path,
    *,
    validate: bool = True,
    force: bool = False,
) -> EdgeBundleResult:
    return EdgeBundler().bundle(
        program, parameter_set, mxai_path, params_path, outdir,
        validate=validate, force=force,
    )


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

def _build_model_manifest(program: MatrixAIProgram, parameter_set: ParameterSet) -> dict:
    from matrixai.compiler import BackendContractAnalyzer
    report = BackendContractAnalyzer().analyze(program)
    inputs: list[dict] = []
    for v in program.vectors:
        inputs.append({"kind": "vector", "name": v.name, "size": v.size,
                       "dtype": "float32", "fields": list(v.fields)})
    for s in program.sequences:
        inputs.append({"kind": "sequence", "name": s.name, "length": s.length,
                       "vocab_size": s.vocab_size, "dtype": "int64"})
    return {
        "project": program.project,
        "model_hash": parameter_set.model_hash,
        "parameter_schema_hash": parameter_set.parameter_schema_hash,
        "parameter_set_id": parameter_set.parameter_set_id,
        "inputs": inputs,
        "vectors": [
            {"name": v.name, "size": v.size, "fields": list(v.fields)}
            for v in program.vectors
        ],
        "sequences": [
            {"name": s.name, "length": s.length, "vocab_size": s.vocab_size}
            for s in program.sequences
        ],
        "functions": [
            {"name": f.name, "kind": f.semantic.kind}
            for f in program.functions
        ],
        "backend_contract": {
            "target": report.target,
            "ok": report.ok,
            "unsupported_nodes": list(report.unsupported_nodes),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_export_manifest_no_eq(export_result: OnnxExportResult, path: Path) -> None:
    data = {
        "model_hash": export_result.model_hash,
        "parameter_schema_hash": export_result.parameter_schema_hash,
        "parameter_set_id": export_result.parameter_set_id,
        "format": "onnx",
        "format_version": export_result.opset_version,
        "input_name": export_result.input_name,
        "input_shape": export_result.input_shape,
        "output_name": export_result.output_name,
        "output_shape": export_result.output_shape,
        "exported_function": export_result.exported_functions[0] if export_result.exported_functions else None,
        "tolerance": None,
        "equivalence_check": None,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_readme(
    program: MatrixAIProgram,
    export_result: OnnxExportResult,
    eq_result: OnnxEquivalenceResult | None,
) -> str:
    project = program.project
    out_name = export_result.output_name
    out_shape = export_result.output_shape

    is_sequence = bool(program.sequences)
    if is_sequence:
        seq = program.sequences[0]
        input_name = seq.name
        input_desc = f"`{seq.name}` shape `{export_result.input_shape}` (int64 token IDs)"
        inference_snippet = (
            f"sess = ort.InferenceSession(\"model.onnx\")\n"
            f"ids = np.array([[...]], dtype=np.int64)  # shape [batch, {seq.length}]\n"
            f"result = sess.run(None, {{\"{seq.name}\": ids}})[0]  # {out_shape}"
        )
    else:
        vec = program.vectors[0] if program.vectors else None
        input_name = vec.name if vec else "input"
        vec_size = vec.size if vec else "?"
        input_desc = f"`{input_name}` shape `{export_result.input_shape}` (float32)"
        inference_snippet = (
            f"sess = ort.InferenceSession(\"model.onnx\")\n"
            f"x = np.array([[...]], dtype=np.float32)  # shape [batch, {vec_size}]\n"
            f"result = sess.run(None, {{\"{input_name}\": x}})[0]  # {out_shape}"
        )

    eq_line = ""
    if eq_result is not None:
        status = "PASS" if eq_result.passed else "FAIL"
        eq_line = (
            f"\nEquivalence check: {status} "
            f"(max_abs_diff={eq_result.max_abs_diff:.2e}, "
            f"atol={eq_result.atol:.0e}, n={eq_result.n_samples})\n"
        )

    skipped = ""
    if export_result.skipped_functions:
        skipped = (
            f"\nNote: the following functions were not exported (unsupported kind): "
            f"{', '.join(export_result.skipped_functions)}\n"
        )

    return f"""# {project} Edge Bundle

MatrixAI model exported for edge/production inference.
Actions remain `simulate_only`. This bundle only provides predictions.

## Files

| File | Description |
|------|-------------|
| `model.mxai` | MatrixAI model definition (source of truth) |
| `params.best.json` | Trained parameter weights |
| `model.onnx` | ONNX model, opset {export_result.opset_version} |
| `model_manifest.json` | Model metadata, hashes and backend contract |
| `export_manifest.json` | Export metadata, tolerance and equivalence check |
| `README.md` | This file |

## Model info

- Project: `{project}`
- Model hash: `{export_result.model_hash}`
- Parameter set: `{export_result.parameter_set_id}`
- Input: {input_desc}
- Output: `{out_name}` shape `{out_shape}`
{eq_line}{skipped}
## Running inference with onnxruntime

```python
import onnxruntime as ort
import numpy as np

{inference_snippet}
```

## Verifying integrity

```python
import json
with open("model_manifest.json") as f:
    manifest = json.load(f)
assert manifest["model_hash"] == "{export_result.model_hash}"
assert manifest["parameter_schema_hash"] == "{export_result.parameter_schema_hash}"
```
"""
