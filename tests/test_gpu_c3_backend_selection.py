"""GPU C3 — Studio backend selection: auto-detect GPU, torch path, stdlib fallback.

The default (no GPU, no env) must stay stdlib with no behaviour change. Forcing
MATRIXAI_TRAIN_BACKEND=torch routes the Studio training through the torch trainers
(on CPU here) and reports the real backend, with the same result shape.
"""
from __future__ import annotations

from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None

DENSE_MXAI = """PROJECT D
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
DENSE_TRAIN = """MODEL D.mxai
DATASET DS
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [a, b]
  TARGET y: Label[A, B]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END
LOSS L
  TYPE cross_entropy
  PREDICTION Net
  TARGET y
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.05
  UPDATE Net.*
END
"""
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
COMPOSITE_TRAIN = DENSE_TRAIN.replace("MODEL D.mxai", "MODEL C.mxai")


def _csv():
    rows = ["a,b,y"]
    for i in range(40):
        a = (i % 10) / 10.0
        rows.append(f"{a:.3f},{(i % 7) / 7.0:.3f},{'B' if a > 0.5 else 'A'}")
    return "\n".join(rows) + "\n"


def test_selector_stdlib_when_no_gpu_or_forced(monkeypatch):
    from matrixai.playground import _select_train_backend
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")
    assert _select_train_backend() == (False, "cpu")


def test_selector_auto_falls_back_to_stdlib_without_cuda(monkeypatch):
    # The dev box has no CUDA → auto must pick stdlib (no regression).
    from matrixai.playground import _select_train_backend
    monkeypatch.delenv("MATRIXAI_TRAIN_BACKEND", raising=False)
    use_torch, _ = _select_train_backend()
    import torch  # noqa
    assert use_torch is torch.cuda.is_available()  # False on a CPU box


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_selector_torch_forced(monkeypatch):
    # CUDA-aware: en un box sin GPU → (True, "cpu"); con GPU (p. ej. Colab) → (True, "cuda").
    import torch
    from matrixai.playground import _select_train_backend
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert _select_train_backend() == (True, expected_device)


def test_dense_default_backend_is_stdlib(monkeypatch):
    from matrixai.playground import _run_playground_dense_training
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")
    r = _run_playground_dense_training(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=5)
    assert r["ok"], r.get("error")
    assert r["backend"] == "stdlib"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_dense_torch_path_through_studio(monkeypatch):
    from matrixai.playground import _run_playground_dense_training
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
    r = _run_playground_dense_training(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=10)
    assert r["ok"], r.get("error")
    import torch
    assert r["backend"] == ("cuda" if torch.cuda.is_available() else "cpu")
    assert r["params_best"] is not None
    assert r["epochs"]                          # epoch trace present


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_composite_torch_path_through_studio(monkeypatch):
    from matrixai.playground import _run_playground_composite_training
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
    r = _run_playground_composite_training(COMPOSITE_MXAI, COMPOSITE_TRAIN, _csv(), epochs_override=10)
    assert r["ok"], r.get("error")
    import torch
    assert r["backend"] == ("cuda" if torch.cuda.is_available() else "cpu")
    assert r["network_kind"] == "composite_network"
    assert r["params_best"] is not None
    expected_batch = 32 if torch.cuda.is_available() else 8
    assert r["effective_batch_size"] == expected_batch
    assert isinstance(r["peak_vram_gb"], float)
    backend_report = r["training_trace"]["backend_report"]
    assert backend_report["effective_batch_size"] == r["effective_batch_size"]
    assert backend_report["peak_vram_gb"] == r["peak_vram_gb"]


def test_composite_default_backend_is_stdlib(monkeypatch):
    from matrixai.playground import _run_playground_composite_training
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")
    r = _run_playground_composite_training(COMPOSITE_MXAI, COMPOSITE_TRAIN, _csv(), epochs_override=5)
    assert r["ok"], r.get("error")
    assert r["backend"] == "stdlib"


def test_async_job_status_reports_backend(monkeypatch):
    import time
    from matrixai.playground import _cancel_job, _get_job_status, _submit_training_job

    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")
    r = _submit_training_job(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=3)
    assert r["ok"], r.get("error")
    job_id = r["job_id"]
    deadline = time.time() + 35
    status = {}
    try:
        while time.time() < deadline:
            status = _get_job_status(job_id)
            if status["status"] in ("done", "error", "cancelled", "timeout"):
                break
            time.sleep(0.2)
    finally:
        _cancel_job(job_id)

    assert status["status"] == "done", status.get("error")
    assert status["backend"] == "stdlib"


def test_embedded_ui_reports_backend_in_success_message():
    from matrixai.playground import _INDEX_HTML

    assert "Entrenamiento OK" in _INDEX_HTML
    assert "s.backend || 'stdlib'" in _INDEX_HTML
