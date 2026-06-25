"""GPU C6 / M14 — evaluación batched en torch (paridad de métricas con stdlib).

El cuello de M14 era que tras entrenar en GPU, la evaluación corría `dense_forward`
fila a fila en Python → con muchas filas el run se colgaba. `evaluate_dense_network_torch`
hace el forward batched y reusa `result_from_predictions`, así que las métricas deben ser
IDÉNTICAS a las del evaluador stdlib (dentro de tolerancia del forward torch vs stdlib).
"""
from __future__ import annotations

import random
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

CLS_MXAI = """PROJECT P
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

REG_MXAI = """PROJECT R
VECTOR In[1]
  x: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END
GRAPH
  In -> Net
END
"""


def _setup(mxai):
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(mxai)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    ps = build_network_parameter_set(net, rl, program_hash(prog), seed=1)
    return net, ps


def _cls_examples(n=150, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.random(), rng.random()
        rows.append(([a, b], [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))
    return rows


def _reg_examples(n=150, seed=0):
    rng = random.Random(seed)
    return [([v := rng.random()], [2.0 * v]) for _ in range(n)]


def test_torch_eval_matches_stdlib_classification():
    from matrixai.training.dense_torch_trainer import (
        train_dense_network_torch, evaluate_dense_network_torch,
    )
    from matrixai.training.dense_evaluator import evaluate_dense_network
    net, ps = _setup(CLS_MXAI)
    ex = _cls_examples()
    best = train_dense_network_torch(net, ps, ex, "cross_entropy",
                                     lr=0.5, epochs=60, device="cpu", seed=1)["best_params"]
    labels = ["A", "B"]
    r_t = evaluate_dense_network_torch(net, best, ex, "cross_entropy", labels=labels, device="cpu")
    r_s = evaluate_dense_network(net, best, ex, "cross_entropy", labels=labels)
    assert r_t.rows == r_s.rows
    assert r_t.accuracy == pytest.approx(r_s.accuracy, abs=1e-6)
    assert r_t.macro_f1 == pytest.approx(r_s.macro_f1, abs=1e-6)
    assert r_t.confusion_matrix == r_s.confusion_matrix
    assert r_t.loss == pytest.approx(r_s.loss, abs=1e-4)


def test_torch_eval_matches_stdlib_regression():
    from matrixai.training.dense_torch_trainer import evaluate_dense_network_torch
    from matrixai.training.dense_evaluator import evaluate_dense_network
    net, ps = _setup(REG_MXAI)
    ex = _reg_examples()
    r_t = evaluate_dense_network_torch(net, ps, ex, "mse", device="cpu")
    r_s = evaluate_dense_network(net, ps, ex, "mse")
    assert r_t.mae == pytest.approx(r_s.mae, abs=1e-5)
    assert r_t.rmse == pytest.approx(r_s.rmse, abs=1e-5)
    assert r_t.r2 == pytest.approx(r_s.r2, abs=1e-5)


def test_torch_eval_empty_raises():
    from matrixai.training.dense_torch_trainer import evaluate_dense_network_torch
    net, ps = _setup(CLS_MXAI)
    with pytest.raises(ValueError):
        evaluate_dense_network_torch(net, ps, [], "cross_entropy", labels=["A", "B"], device="cpu")


# ── composite (M14 camino composite) ────────────────────────────────────────────

COMPOSITE_MXAI = """PROJECT C
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  BLOCK r1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""


def _setup_composite():
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(COMPOSITE_MXAI)
    net = prog.networks[0]
    tr = check_composite_network_types(net, {v.name: v for v in prog.vectors})
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=program_hash(prog), seed=1)
    return net, ps


def test_composite_torch_eval_matches_stdlib():
    from matrixai.training.composite_torch_trainer import evaluate_composite_network_torch
    from matrixai.training.composite_evaluator import evaluate_composite_network
    net, ps = _setup_composite()
    # examples como (input_dict, target) — lo que consume el evaluador composite
    rng = random.Random(0)
    rng_ex = []
    for _ in range(80):
        a, b = rng.random(), rng.random()
        rng_ex.append(({"a": a, "b": b}, [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))
    labels = ["A", "B"]
    r_t = evaluate_composite_network_torch(net, ps, rng_ex, "cross_entropy", labels=labels, device="cpu")
    r_s = evaluate_composite_network(net, ps, rng_ex, "cross_entropy", labels=labels)
    assert r_t.rows == r_s.rows
    assert r_t.accuracy == pytest.approx(r_s.accuracy, abs=1e-6)
    assert r_t.macro_f1 == pytest.approx(r_s.macro_f1, abs=1e-6)
    assert r_t.confusion_matrix == r_s.confusion_matrix
    assert r_t.loss == pytest.approx(r_s.loss, abs=1e-4)


def test_composite_batched_forward_matches_per_sample():
    """M15(e): forward_batch produce EXACTAMENTE lo mismo que forward_with_dict
    por muestra (LayerNorm normaliza por fila igual; el batch solo agrupa kernels)."""
    import torch
    from matrixai.forward.composite_torch import (
        composite_network_to_torch_module,
        composite_torch_forward,
        composite_torch_forward_batch,
    )
    net, ps = _setup_composite()
    module = composite_network_to_torch_module(net, ps)
    rng = random.Random(3)
    batch = [{"a": rng.random(), "b": rng.random()} for _ in range(50)]

    per_sample = [composite_torch_forward(module, x, False) for x in batch]
    batched = composite_torch_forward_batch(module, batch, False)

    assert len(batched) == len(per_sample)
    for row_b, row_s in zip(batched, per_sample):
        assert row_b == pytest.approx(row_s, abs=1e-5)


def test_composite_torch_eval_chunked_large():
    """M15(e): eval composite con >4096 muestras (troceo) completa y da métricas."""
    from matrixai.training.composite_torch_trainer import evaluate_composite_network_torch
    net, ps = _setup_composite()
    rng = random.Random(1)
    ex = []
    for _ in range(5000):
        a, b = rng.random(), rng.random()
        ex.append(({"a": a, "b": b}, [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))
    r = evaluate_composite_network_torch(net, ps, ex, "cross_entropy", labels=["A", "B"], device="cpu")
    assert r.rows == 5000
    assert r.accuracy is not None


# ── M15(c): TF32 gating ──────────────────────────────────────────────────────────

def test_tf32_helper_noop_on_cpu():
    """En CPU el helper no hace nada (TF32 es sólo CUDA) → False."""
    from matrixai.parameters.tensor_bridge import enable_tf32_if_cuda
    assert enable_tf32_if_cuda("cpu") is False


def test_tf32_helper_respects_env_off(monkeypatch):
    """MATRIXAI_GPU_TF32=0 desactiva el helper aunque el device sea cuda."""
    from matrixai.parameters.tensor_bridge import enable_tf32_if_cuda
    monkeypatch.setenv("MATRIXAI_GPU_TF32", "0")
    assert enable_tf32_if_cuda("cuda") is False


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requiere CUDA")
def test_tf32_helper_enables_on_cuda(monkeypatch):
    """Con CUDA real y env por defecto, activa los flags TF32 y devuelve True."""
    import torch
    from matrixai.parameters.tensor_bridge import enable_tf32_if_cuda
    monkeypatch.delenv("MATRIXAI_GPU_TF32", raising=False)
    assert enable_tf32_if_cuda("cuda") is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


_TRAIN_TEXT = """MODEL D.mxai
DATASET DS
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [a, b]
  TARGET y: Label[A, B]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=16
END
LOSS L
  TYPE cross_entropy
  PREDICTION Net
  TARGET y
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.5
  UPDATE Net.*
END
"""


def _cls_csv(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    rows = ["a,b,y"]
    for _ in range(n):
        a, b = rng.random(), rng.random()
        rows.append(f"{a:.4f},{b:.4f},{'B' if a > 0.5 else 'A'}")
    return "\n".join(rows) + "\n"


def test_playground_torch_path_reports_metrics(monkeypatch):
    """El camino torch completa y devuelve métricas + evaluation_backend sin warning."""
    from matrixai.playground import _run_playground_dense_training
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
    mxai = CLS_MXAI.replace("PROJECT P", "PROJECT D")
    r = _run_playground_dense_training(mxai, _TRAIN_TEXT, _cls_csv(120), epochs_override=40)
    assert r["ok"], r.get("error")
    import torch
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert r["backend"] == expected_device
    assert r["evaluation_backend"] == expected_device
    assert r["evaluation_warning"] is None
    assert r["accuracy"] is not None
    assert r["macro_f1"] is not None
    assert r["evaluation_report"] is not None


def test_playground_torch_eval_chunked_large(monkeypatch):
    """Eval densa con >4096 filas no falla ni regresa al batch completo."""
    from matrixai.playground import _run_playground_dense_training
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
    mxai = CLS_MXAI.replace("PROJECT P", "PROJECT E")
    r = _run_playground_dense_training(mxai, _TRAIN_TEXT, _cls_csv(5000), epochs_override=5)
    assert r["ok"], r.get("error")
    assert r["evaluation_backend"] is not None
    assert r["evaluation_warning"] is None
    assert r["accuracy"] is not None


def test_playground_torch_eval_fallback_on_error(monkeypatch):
    """Si evaluate_dense_network_torch lanza, evaluation_backend='failed' y warning presente."""
    from matrixai.playground import _run_playground_dense_training
    import matrixai.training.dense_torch_trainer as _dtt
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")

    def _boom(*args, **kwargs):
        raise RuntimeError("eval forced failure")

    monkeypatch.setattr(_dtt, "evaluate_dense_network_torch", _boom)
    mxai = CLS_MXAI.replace("PROJECT P", "PROJECT F")
    r = _run_playground_dense_training(mxai, _TRAIN_TEXT, _cls_csv(120), epochs_override=5)
    assert r["ok"], r.get("error")
    assert r["evaluation_backend"] == "failed"
    assert r["evaluation_warning"] is not None
    assert "forced failure" in r["evaluation_warning"]
    assert r["accuracy"] is None
