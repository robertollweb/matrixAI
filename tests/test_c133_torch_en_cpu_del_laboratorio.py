"""CONTRATO 133 — decisión de Roberto del 18-10 (la A): el LABORATORIO entrena con torch en CPU.

En un equipo sin GPU, la regla global `auto` entrena la red densa con stdlib aunque torch esté instalado (el paquete CPU y
la demo). El laboratorio (tu red en el modo experto) pide `preferir_torch_en_cpu=True` en cada entrenamiento: con torch
instalado entrena con torch también sin CUDA, porque sus controles (adam, adamw, weight_decay, schedule) y lo que solo mide
ese entrenador (gradientes, pesos por época) lo necesitan. Lo que NO cambia, y se prueba aquí:
  - sin la petición, `auto` sigue siendo «CUDA o stdlib» (la clásica, el estudio);
  - un `MATRIXAI_TRAIN_BACKEND=stdlib` EXPLÍCITO del operador manda sobre la petición;
  - sin torch instalado, stdlib (y el `backend` del resultado lo dice).
Y vale también para tu red COMPUESTA (residual, embedding): la re-auditoría del C1 (18-10, N-2) midió que el Studio decía
«torch · cpu» y aceptaba adam mientras esa red entrenaba sin torch.
"""
from __future__ import annotations

import time
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch no instalado")

MXAI = """PROJECT D
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
TRAIN = """MODEL D.mxai
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


def _csv() -> str:
    filas = ["a,b,y"]
    for i in range(40):
        a = (i % 10) / 10.0
        filas.append(f"{a:.3f},{(i % 7) / 7.0:.3f},{'B' if a > 0.5 else 'A'}")
    return "\n".join(filas) + "\n"


@pytest.fixture
def sin_cuda(monkeypatch):
    """`auto` sin GPU, aunque la máquina de pruebas tuviera una: es el caso del paquete CPU y de la demo."""
    monkeypatch.delenv("MATRIXAI_TRAIN_BACKEND", raising=False)
    if _HAS_TORCH:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@needs_torch
def test_auto_sin_gpu_sigue_en_stdlib_sin_la_peticion(sin_cuda):
    from matrixai.playground import _select_train_backend
    assert _select_train_backend() == (False, "cpu")


@needs_torch
def test_el_laboratorio_pide_torch_en_cpu_y_lo_obtiene(sin_cuda):
    from matrixai.playground import _select_train_backend
    assert _select_train_backend(preferir_torch=True) == (True, "cpu")


@needs_torch
def test_stdlib_explicito_del_operador_manda_sobre_la_peticion(sin_cuda, monkeypatch):
    from matrixai.playground import _select_train_backend
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")
    assert _select_train_backend(preferir_torch=True) == (False, "cpu")


def test_sin_torch_instalado_la_peticion_cae_en_stdlib(monkeypatch):
    from matrixai.parameters import tensor_bridge
    from matrixai.playground import _select_train_backend
    monkeypatch.delenv("MATRIXAI_TRAIN_BACKEND", raising=False)
    monkeypatch.setattr(tensor_bridge, "torch_available", lambda: False)
    assert _select_train_backend(preferir_torch=True) == (False, "cpu")


@needs_torch
@pytest.mark.parametrize("pedir", [False, True])
def test_entrenamiento_sincrono_usa_el_entrenador_pedido(sin_cuda, pedir):
    from matrixai.playground_api import run_playground_training
    r = run_playground_training(MXAI, TRAIN, _csv(), epochs_override=3, seed=1, preferir_torch_en_cpu=pedir)
    assert r["ok"], r.get("error")
    assert r["backend"] == ("torch" if pedir else "stdlib")


@needs_torch
@pytest.mark.parametrize("pedir", [False, True])
def test_trabajo_asincrono_usa_el_entrenador_pedido(sin_cuda, pedir):
    from matrixai.playground import _get_job_status
    from matrixai.playground_api import submit_training_job
    r = submit_training_job(MXAI, TRAIN, _csv(), epochs_override=3, seed=1, preferir_torch_en_cpu=pedir)
    assert r["ok"], r.get("error")
    fin = time.monotonic() + 120
    estado = _get_job_status(r["job_id"])
    while estado.get("status") not in ("done", "error", "cancelled", "timeout") and time.monotonic() < fin:
        time.sleep(0.2)
        estado = _get_job_status(r["job_id"])
    assert estado.get("status") == "done", estado
    assert estado["backend"] == ("torch" if pedir else "stdlib")


MXAI_COMPUESTA = """PROJECT C
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
TRAIN_COMPUESTA = TRAIN.replace("MODEL D.mxai", "MODEL C.mxai")


@needs_torch
@pytest.mark.parametrize("pedir", [False, True])
def test_tu_red_compuesta_tambien_usa_el_entrenador_pedido_sincrono(sin_cuda, pedir):
    from matrixai.playground_api import run_playground_training
    r = run_playground_training(MXAI_COMPUESTA, TRAIN_COMPUESTA, _csv(), epochs_override=2, seed=1,
                                preferir_torch_en_cpu=pedir)
    assert r["ok"], r.get("error")
    assert r["network_kind"] == "composite_network"
    assert r["backend"] == ("torch" if pedir else "stdlib")


@needs_torch
@pytest.mark.parametrize("pedir", [False, True])
def test_tu_red_compuesta_tambien_usa_el_entrenador_pedido_asincrono(sin_cuda, pedir):
    from matrixai.playground import _get_job_status
    from matrixai.playground_api import submit_training_job
    r = submit_training_job(MXAI_COMPUESTA, TRAIN_COMPUESTA, _csv(), epochs_override=2, seed=1,
                            preferir_torch_en_cpu=pedir)
    assert r["ok"], r.get("error")
    fin = time.monotonic() + 120
    estado = _get_job_status(r["job_id"])
    while estado.get("status") not in ("done", "error", "cancelled", "timeout") and time.monotonic() < fin:
        time.sleep(0.2)
        estado = _get_job_status(r["job_id"])
    assert estado.get("status") == "done", estado
    assert estado["backend"] == ("torch" if pedir else "stdlib")
