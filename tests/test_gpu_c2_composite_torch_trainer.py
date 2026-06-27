"""GPU C2 — torch training loop for composite_network (CPU correctness).

Same guarantees as C1 (learns, forward parity with stdlib, early stop) but for
composite architectures: embeddings, concat, residual blocks, LayerNorm, Dropout.
GPU is validated separately; here we validate correctness on CPU.
"""
from __future__ import annotations

import random
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

_ATOL = 1e-5

EMB_MXAI = """PROJECT P
VECTOR Input[2]
  cat: Integer[0, 8]
  x: Scalar
END
NETWORK Net
  INPUT Input
  EMBEDDING e1 FROM cat VOCAB 9 DIM 4
  CONCAT [e1, x] -> features
  LAYER Dense units=16 activation=relu
  BLOCK r1
    LAYER Dense units=16 activation=relu
    LAYER LayerNorm
    LAYER Dropout rate=0.1
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=3 activation=softmax
  OUTPUT y: ProbabilityMap[A, B, C]
END
GRAPH
  Input -> Net
END
"""

RESIDUAL_MXAI = """PROJECT P
VECTOR Input[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT Input
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
  Input -> Net
END
"""


def _setup(mxai):
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(mxai)
    net = prog.networks[0]
    tr = check_composite_network_types(net, {v.name: v for v in prog.vectors})
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=program_hash(prog), seed=1)
    return net, ps


def _emb_examples(n=120, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        c = rng.randint(0, 8)
        oh = [0.0, 0.0, 0.0]
        oh[c % 3] = 1.0  # label = cat % 3 (pure embedding signal)
        rows.append(({"cat": c, "x": rng.random()}, oh))
    return rows


def test_composite_torch_trainer_learns_embedding_signal():
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    from matrixai.parameters.store import ParameterSet
    net, ps = _setup(EMB_MXAI)
    res = train_composite_network_torch(net, ps, _emb_examples(), "cross_entropy",
                                        lr=0.05, epochs=60, device="cpu", seed=1)
    assert res["backend"] == "torch"
    assert res["best_val_loss"] < res["epochs"][0]["val_loss"]
    assert isinstance(res["best_params"], ParameterSet)


def test_composite_torch_forward_parity_with_stdlib():
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    from matrixai.forward.composite_forward import composite_forward
    from matrixai.forward.composite_torch import composite_network_to_torch_module, composite_torch_forward
    net, ps = _setup(EMB_MXAI)
    ex = _emb_examples()
    best = train_composite_network_torch(net, ps, ex, "cross_entropy",
                                         lr=0.05, epochs=60, device="cpu", seed=1)["best_params"]
    module = composite_network_to_torch_module(net, best)
    for d, _ in ex[:12]:
        t = composite_torch_forward(module, d, False)
        s = composite_forward(net, best, d, training=False)
        assert max(abs(p - q) for p, q in zip(t, s)) < _ATOL


def test_composite_torch_trainer_accuracy_above_prior():
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    from matrixai.forward.composite_forward import composite_forward
    net, ps = _setup(EMB_MXAI)
    ex = _emb_examples()
    best = train_composite_network_torch(net, ps, ex, "cross_entropy",
                                         lr=0.05, epochs=120, device="cpu", seed=1)["best_params"]
    correct = 0
    for d, _ in ex:
        p = composite_forward(net, best, d, training=False)
        correct += p.index(max(p)) == (d["cat"] % 3)
    assert correct / len(ex) >= 0.85  # learns the embedding signal (1/3 prior)


def test_residual_only_composite_trains():
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    net, ps = _setup(RESIDUAL_MXAI)
    rng = random.Random(0)
    ex = []
    for _ in range(120):
        a, b, c = rng.random(), rng.random(), rng.random()
        ex.append(({"a": a, "b": b, "c": c}, [1.0, 0.0] if a > 0.5 else [0.0, 1.0]))
    res = train_composite_network_torch(net, ps, ex, "cross_entropy",
                                        lr=0.05, epochs=60, device="cpu", seed=1)
    assert res["best_val_loss"] < res["epochs"][0]["val_loss"]


def test_early_stop_respected():
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    net, ps = _setup(EMB_MXAI)
    rng = random.Random(7)
    noise = [({"cat": rng.randint(0, 8), "x": rng.random()},
              [1.0, 0.0, 0.0] if rng.random() < 0.5 else [0.0, 1.0, 0.0]) for _ in range(120)]
    res = train_composite_network_torch(net, ps, noise, "cross_entropy",
                                        lr=0.05, epochs=300, early_stop=(5, "validation_loss"),
                                        device="cpu", seed=1)
    assert len(res["epochs"]) < 300


def test_batched_training_effective_batch_size_in_result():
    """Batched trainer devuelve effective_batch_size y aprende igual que antes."""
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    net, ps = _setup(EMB_MXAI)
    ex = _emb_examples()
    res = train_composite_network_torch(net, ps, ex, "cross_entropy",
                                        lr=0.05, epochs=60, device="cpu", seed=1,
                                        batch_size=32)
    assert "effective_batch_size" in res
    assert res["effective_batch_size"] == 32  # CPU respeta spec
    assert res["best_val_loss"] < res["epochs"][0]["val_loss"]


def test_batched_cancel_check_called():
    """cancel_check se llama al menos una vez por epoch (una vez por batch)."""
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    net, ps = _setup(RESIDUAL_MXAI)
    rng = random.Random(0)
    ex = [({"a": rng.random(), "b": rng.random(), "c": rng.random()},
           [1.0, 0.0] if rng.random() > 0.5 else [0.0, 1.0]) for _ in range(60)]
    calls = []
    train_composite_network_torch(net, ps, ex, "cross_entropy",
                                  lr=0.05, epochs=3, device="cpu", seed=1,
                                  cancel_check=lambda: calls.append(1))
    assert len(calls) >= 3  # al menos un call por epoch
