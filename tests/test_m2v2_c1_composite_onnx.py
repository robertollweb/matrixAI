"""M2 v2 — C1: ONNX lowering de arquitecturas compuestas (P19).

El exportador ONNX baja un composite_network (bloques residuales, LayerNorm,
Dropout, capas Dense interleaved) a un grafo ONNX cuyas salidas son equivalentes
a composite_forward(training=False). Embeddings/concats se difieren a C4 y deben
fallar con un error claro.
"""
from __future__ import annotations

import tempfile
from importlib import util
from pathlib import Path

import pytest

_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None
_ATOL = 1e-5

pytestmark = pytest.mark.skipif(not _HAS_ONNX, reason="onnx not installed")


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
    LAYER Dropout rate=0.2
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

# Composite without blocks, but a top-level LayerNorm forces composite_network kind
# (a Dense-only network would parse as dense_network).
FLAT_COMPOSITE = """PROJECT FlatComp
VECTOR Input[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT Input
  LAYER Dense units=5 activation=relu
  LAYER LayerNorm
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[X, Y]
END
GRAPH
  Input -> Net
END
"""

EMBEDDING_COMPOSITE = """PROJECT EmbComp
VECTOR Input[2]
  cat: Integer[0, 10]
  num: Scalar
END
NETWORK Net
  INPUT Input
  EMBEDDING e1 FROM cat VOCAB 10 DIM 4
  CONCAT [e1, num] -> features
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[X, Y]
END
GRAPH
  Input -> Net
END
"""


def _build(mxai_text, seed=7):
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash

    prog = parse_text(mxai_text)
    net = prog.networks[0]
    assert net.kind == "composite_network"
    vmap = {v.name: v for v in prog.vectors}
    tr = check_composite_network_types(net, vmap)
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=program_hash(prog), seed=seed)
    return prog, net, ps


def _export(prog, ps):
    from matrixai.export.onnx_exporter import export_onnx
    path = Path(tempfile.mkdtemp()) / "model.onnx"
    return export_onnx(prog, ps, path), path


def test_composite_export_produces_valid_onnx():
    import onnx
    prog, net, ps = _build(COMPOSITE)
    res, path = _export(prog, ps)
    assert res.exported_functions == ["Net"]
    assert res.input_shape == [-1, 4]
    assert res.output_shape == [-1, 3]
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    meta = {m.key: m.value for m in model.metadata_props}
    assert meta["matrixai_kind"] == "composite_network"
    assert meta["matrixai_model_hash"] == ps.model_hash


@pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime not installed")
def test_composite_parity_with_forward():
    import numpy as np
    import onnxruntime as ort
    from matrixai.forward.composite_forward import composite_forward

    prog, net, ps = _build(COMPOSITE)
    _res, path = _export(prog, ps)
    sess = ort.InferenceSession(str(path))
    iname = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    for _ in range(8):
        x = rng.random(4).astype(np.float32)
        onnx_out = sess.run(None, {iname: x.reshape(1, 4)})[0].ravel()
        fwd = composite_forward(net, ps, {f"f{i+1}": float(x[i]) for i in range(4)}, training=False)
        assert np.max(np.abs(onnx_out - np.array(fwd))) < _ATOL


@pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime not installed")
def test_flat_composite_parity():
    """A composite network with no blocks (just top Dense layers) still exports."""
    import numpy as np
    import onnxruntime as ort
    from matrixai.forward.composite_forward import composite_forward

    prog, net, ps = _build(FLAT_COMPOSITE)
    _res, path = _export(prog, ps)
    sess = ort.InferenceSession(str(path))
    iname = sess.get_inputs()[0].name
    rng = np.random.default_rng(1)
    x = rng.random(3).astype(np.float32)
    onnx_out = sess.run(None, {iname: x.reshape(1, 3)})[0].ravel()
    fwd = composite_forward(net, ps, {"a": float(x[0]), "b": float(x[1]), "c": float(x[2])}, training=False)
    assert np.max(np.abs(onnx_out - np.array(fwd))) < _ATOL


@pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime not installed")
def test_composite_parity_with_trained_weights(tmp_path, monkeypatch):
    """Parity holds after real training (weights are no longer the seeded init).

    Uses the real Studio surface: generate a composite model + valid training_text,
    train through the composite path, then export the trained ParameterSet.
    """
    import re
    import time
    import numpy as np
    import onnxruntime as ort
    from matrixai.parser import parse_text
    from matrixai.parameters.store import ParameterSet, program_hash
    from matrixai.forward.composite_forward import composite_forward
    from matrixai.playground import _submit_training_job, _get_job_status
    from matrixai_studio.endpoints import _studio_generate

    monkeypatch.setenv("MATRIXAI_MODELS_DIR", str(tmp_path / "studio-models"))
    gen = _studio_generate({
        "prompt": ("Clasificar el riesgo en 3 niveles BAJO MEDIO ALTO con una red "
                   "profunda de bloques residuales con LayerNorm y Dropout, "
                   "features f1 f2 f3 f4"),
        "locale": "es",
    })
    assert gen["ok"], gen.get("error")
    mxai, training = gen["mxai"], gen["training_text"]
    prog = parse_text(mxai)
    net = prog.networks[0]
    assert net.kind == "composite_network"

    cols = [c.strip() for c in re.search(r"FROM COLUMNS \[([^\]]*)\]", training).group(1).split(",")]
    tm = re.search(r"TARGET\s+\w+\s*:\s*Label\[([^\]]*)\]", training)
    labels = [l.strip() for l in tm.group(1).split(",")]
    rows = [",".join(cols + [re.search(r"TARGET\s+(\w+)", training).group(1)])]
    for i in range(48):
        vals = [f"{((i + j) % 10) / 10.0:.3f}" for j in range(len(cols))]
        rows.append(",".join(vals + [labels[i % len(labels)]]))
    csv_text = "\n".join(rows) + "\n"

    job = _submit_training_job(mxai, training, csv_text, epochs_override=3)
    assert job.get("ok"), job
    st = {}
    for _ in range(240):
        st = _get_job_status(job["job_id"])
        if st["status"] != "running":
            break
        time.sleep(0.25)
    assert st["status"] == "done", st
    params_best = st["params_best"]
    assert params_best is not None
    # params_best is a serialized ParameterSet (parameters + hashes), trained on this mxai.
    trained = ParameterSet.from_dict(params_best)
    assert trained.model_hash == program_hash(prog)
    _res, path = _export(prog, trained)
    sess = ort.InferenceSession(str(path))
    iname = sess.get_inputs()[0].name
    x = np.full(len(cols), 0.4, dtype=np.float32)
    onnx_out = sess.run(None, {iname: x.reshape(1, len(cols))})[0].ravel()
    fwd = composite_forward(net, trained, {cols[i]: float(x[i]) for i in range(len(cols))}, training=False)
    assert np.max(np.abs(onnx_out - np.array(fwd))) < _ATOL


@pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime not installed")
def test_embedding_composite_exports_with_parity():
    """M2 v2 C4: EMBEDDING + CONCAT composites export and match composite_forward."""
    import numpy as np
    import onnxruntime as ort
    from matrixai.forward.composite_forward import composite_forward

    prog, net, ps = _build(EMBEDDING_COMPOSITE)
    _res, path = _export(prog, ps)
    sess = ort.InferenceSession(str(path))
    iname = sess.get_inputs()[0].name
    rng = np.random.default_rng(2)
    for _ in range(8):
        cat = float(rng.integers(0, 10))
        num = float(rng.random())
        onnx_out = sess.run(None, {iname: np.array([[cat, num]], dtype=np.float32)})[0].ravel()
        fwd = composite_forward(net, ps, {"cat": cat, "num": num}, training=False)
        assert np.max(np.abs(onnx_out - np.array(fwd))) < _ATOL
