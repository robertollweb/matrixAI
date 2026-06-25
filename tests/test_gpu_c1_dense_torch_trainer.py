"""GPU C1 — torch training loop for dense_network (CPU correctness).

The torch trainer must (1) actually learn (loss decreases, accuracy above prior),
(2) produce weights whose forward matches the stdlib forward within tolerance, and
(3) honour early stopping. GPU acceleration is validated separately on a GPU machine;
here we validate correctness on CPU (the dev server has no GPU but torch-CPU is present).
"""
from __future__ import annotations

import random
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

_ATOL = 1e-5

MXAI = """PROJECT P
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


def _setup():
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(MXAI)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    ps = build_network_parameter_set(net, rl, program_hash(prog), seed=1)
    return net, ps


def _signal_examples(n=120, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.random(), rng.random()
        rows.append(([a, b], [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))  # B iff a>0.5
    return rows


def test_torch_dense_trainer_learns():
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    res = train_dense_network_torch(net, ps, _signal_examples(), "cross_entropy",
                                    lr=0.5, epochs=80, device="cpu", seed=1)
    assert res["backend"] == "torch" and res["device"] == "cpu"
    # loss decreased from the first epoch to the best
    assert res["best_val_loss"] < res["epochs"][0]["val_loss"]
    from matrixai.parameters.store import ParameterSet
    assert isinstance(res["best_params"], ParameterSet)


def test_trained_weights_have_forward_parity_with_stdlib():
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.forward.dense_forward import dense_forward
    from matrixai.forward.dense_torch import dense_network_to_torch_module, dense_torch_forward
    net, ps = _setup()
    ex = _signal_examples()
    best = train_dense_network_torch(net, ps, ex, "cross_entropy",
                                     lr=0.5, epochs=80, device="cpu", seed=1)["best_params"]
    module = dense_network_to_torch_module(net, best)
    for x, _ in ex[:15]:
        t = dense_torch_forward(module, x)
        s = dense_forward(net, best, x)
        assert max(abs(p - q) for p, q in zip(t, s)) < _ATOL


def test_torch_dense_trainer_accuracy_above_prior():
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.forward.dense_forward import dense_forward
    net, ps = _setup()
    ex = _signal_examples()
    best = train_dense_network_torch(net, ps, ex, "cross_entropy",
                                     lr=0.5, epochs=80, device="cpu", seed=1)["best_params"]
    correct = 0
    for x, y in ex:
        p = dense_forward(net, best, x)
        correct += (0 if p[0] > p[1] else 1) == (0 if y[0] > y[1] else 1)
    assert correct / len(ex) >= 0.85  # clearly above the 0.5 prior


def test_early_stop_respected():
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    # No-signal data (random labels): validation loss plateaus quickly, so early
    # stopping with small patience must cut the run well before the 500-epoch cap.
    rng = random.Random(7)
    noise = [([rng.random(), rng.random()],
              [1.0, 0.0] if rng.random() < 0.5 else [0.0, 1.0]) for _ in range(120)]
    res = train_dense_network_torch(net, ps, noise, "cross_entropy",
                                    lr=0.5, epochs=500, early_stop=(5, "validation_loss"),
                                    device="cpu", seed=1)
    assert len(res["epochs"]) < 500
    # stopped 5 epochs after the best (patience)
    assert len(res["epochs"]) <= res["best_epoch"] + 5


def test_epoch_callback_can_cancel():
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()

    class _Stop(Exception):
        pass

    def cb(entry):
        if entry["epoch"] >= 3:
            raise _Stop()

    with pytest.raises(_Stop):
        train_dense_network_torch(net, ps, _signal_examples(), "cross_entropy",
                                  lr=0.5, epochs=80, device="cpu", seed=1, epoch_callback=cb)


def test_regression_mse_trains():
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    reg = """PROJECT R
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
    prog = parse_text(reg)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    ps = build_network_parameter_set(net, tr.resolved_layers or net.layers, program_hash(prog), seed=1)
    rng = random.Random(0)
    ex = [([v := rng.random()], [2.0 * v]) for _ in range(120)]  # y = 2x
    res = train_dense_network_torch(net, ps, ex, "mse", lr=0.3, epochs=120, device="cpu", seed=1)
    assert res["best_val_loss"] < res["epochs"][0]["val_loss"]
