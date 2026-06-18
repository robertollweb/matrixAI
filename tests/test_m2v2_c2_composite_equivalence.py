"""M2 v2 — C2: equivalencia ONNX y export WASM de arquitecturas compuestas (P19).

El validador de equivalencia compara el ONNX (onnxruntime) contra composite_forward;
el export WASM reutiliza el ONNX por delegación y corre el mismo check.
"""
from __future__ import annotations

import json
import tempfile
from importlib import util
from pathlib import Path

import pytest

_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_ONNX and _HAS_ORT), reason="onnx + onnxruntime required"
)


COMPOSITE = """PROJECT CompProj
VECTOR Input[4]
  f1: Scalar
  f2: Scalar
  f3: Scalar
  f4: Scalar
END
NETWORK Net
  INPUT Input
  LAYER Dense units=8 activation=relu
  BLOCK res1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    LAYER Dropout rate=0.3
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=6 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT y: ProbabilityMap[A, B, C]
END
GRAPH
  Input -> Net
END
"""


def _build():
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash

    prog = parse_text(COMPOSITE)
    net = prog.networks[0]
    vmap = {v.name: v for v in prog.vectors}
    tr = check_composite_network_types(net, vmap)
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=program_hash(prog), seed=11)
    return prog, ps


def test_composite_equivalence_passes():
    from matrixai.export.onnx_exporter import export_onnx
    from matrixai.export.equivalence import validate_onnx_equivalence

    prog, ps = _build()
    path = Path(tempfile.mkdtemp()) / "m.onnx"
    export_onnx(prog, ps, path)
    eq = validate_onnx_equivalence(prog, ps, path, n_samples=20)
    assert eq.passed
    assert eq.n_outputs_per_sample == 3
    assert eq.max_abs_diff < 1e-5


def test_wasm_export_composite_by_delegation():
    from matrixai.export.wasm_exporter import WasmExporter

    prog, ps = _build()
    outdir = Path(tempfile.mkdtemp()) / "bundle"
    # validate=True runs the composite equivalence check internally; it must pass.
    res = WasmExporter().export(prog, ps, outdir, force=True)
    files = set(res.files)
    assert {"model.onnx", "predict.js", "wasm_manifest.json"} <= files
    manifest = json.loads((outdir / "wasm_manifest.json").read_text())
    assert manifest["model_hash"] == ps.model_hash
    assert manifest["equivalence_check"]["passed"] is True


def test_core_manifest_for_composite():
    from matrixai.export.onnx_exporter import export_onnx
    from matrixai.export.equivalence import validate_onnx_equivalence, write_export_manifest

    prog, ps = _build()
    work = Path(tempfile.mkdtemp())
    onnx_path = work / "m.onnx"
    export_result = export_onnx(prog, ps, onnx_path)
    eq = validate_onnx_equivalence(prog, ps, onnx_path)
    manifest_path = work / "export_manifest.json"
    write_export_manifest(export_result, eq, manifest_path)
    data = json.loads(manifest_path.read_text())
    assert data["model_hash"] == ps.model_hash
    assert data["exported_function"] == "Net"
    assert data["equivalence_check"]["passed"] is True
