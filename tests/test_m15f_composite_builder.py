"""M15(f) — `with_values=False` en el builder composite (espejo de M15(a) del denso).

Con torch, el builder devuelve solo la plantilla de estructura (shapes, sin pesos en
Python) y `composite_network_to_torch_module` usa el init nativo de nn.* (sembrado por
torch.manual_seed). Evita el coste O(params) de materializar listas. El stdlib y el export
siguen con `with_values=True`.
"""
from __future__ import annotations

from importlib import util

import pytest

from matrixai.parser import parse_text
from matrixai.types import check_composite_network_types
from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.parameters.store import program_hash

_HAS_TORCH = util.find_spec("torch") is not None

MXAI = """PROJECT C
VECTOR In[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  BLOCK r1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=3 activation=softmax
  OUTPUT y: ProbabilityMap[A, B, C]
END
GRAPH
  In -> Net
END
"""


def _setup():
    prog = parse_text(MXAI)
    net = prog.networks[0]
    tr = check_composite_network_types(net, {v.name: v for v in prog.vectors})
    return net, tr, program_hash(prog)


def test_with_values_false_is_structure_only():
    net, tr, mh = _setup()
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=mh, seed=1, with_values=False)
    assert len(ps.parameters) > 0
    for p in ps.parameters.values():
        assert p["values"] is None
        assert p["shape"] and len(p["shape"]) >= 1  # shape conservado para el init torch


def test_with_values_true_still_materializes():
    net, tr, mh = _setup()
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=mh, seed=1)  # default
    assert all(p["values"] is not None for p in ps.parameters.values())


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_module_builds_from_valueless_template():
    import torch
    from matrixai.forward.composite_torch import (
        composite_network_to_torch_module, composite_torch_forward,
    )
    net, tr, mh = _setup()
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=mh, seed=1, with_values=False)
    torch.manual_seed(1)
    mod = composite_network_to_torch_module(net, ps)
    out = composite_torch_forward(mod, {"a": 0.1, "b": 0.2, "c": 0.3}, False)
    assert len(out) == 3
    assert abs(sum(out) - 1.0) < 1e-5  # softmax


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_composite_training_is_reproducible_and_has_real_weights():
    """El camino torch con plantilla sin pesos produce best_params CON valores reales
    (round-trip desde el módulo) y es reproducible con el mismo seed."""
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    net, tr, mh = _setup()

    examples = []
    import random
    rng = random.Random(0)
    for _ in range(40):
        a, b, c = rng.random(), rng.random(), rng.random()
        idx = 0 if a > 0.6 else (1 if b > 0.6 else 2)
        tgt = [0.0, 0.0, 0.0]; tgt[idx] = 1.0
        examples.append(({"a": a, "b": b, "c": c}, tgt))

    def _run():
        ps = build_composite_network_parameter_set(net, tr, model_hash_str=mh, seed=1, with_values=False)
        return train_composite_network_torch(net, ps, examples, "cross_entropy",
                                              lr=0.1, epochs=5, device="cpu", seed=1)

    r1 = _run()
    r2 = _run()
    # best_params tiene valores reales (no None) tras el entrenamiento
    bp = r1["best_params"]
    assert all(p["values"] is not None for p in bp.parameters.values())
    # reproducible: mismo seed → misma pérdida final
    assert r1["best_val_loss"] == pytest.approx(r2["best_val_loss"], abs=1e-9)
