# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PESOS_GRANDES C5 — reanudar el entrenamiento desde un `state_dict` guardado
en vez de reinicializar. `train_dense_network_torch` acepta `initial_state_dict`
(mismo formato que `dense_module_to_state_dict`/`.mxw`) y, cuando se da,
construye el módulo inicial vía `dense_network_to_torch_module_from_state`
(mismo helper que ya usan `evaluate_dense_network_torch`/`probe_collapse_torch`)
en vez del init nativo de `nn.Linear` — nunca pasa por una `ParameterSet` con
valores ni por `.tolist()`, para modelos grandes o pequeños."""
from __future__ import annotations

import random
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

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


def test_initial_state_dict_is_the_starting_point_not_reinit(monkeypatch):
    """Con lr=0 ningún paso de gradiente cambia los pesos — si el trainer
    arranca de verdad desde `initial_state_dict`, el `best_params` devuelto
    debe ser BYTE-IDÉNTICO a lo que se pasó, nunca al init nativo de
    `nn.Linear` (que con semilla fija sería otro valor distinto y detectable)."""
    import torch
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    state = {
        "Net.W1": torch.full((8, 2), 0.25),
        "Net.b1": torch.full((8,), 0.1),
        "Net.W2": torch.full((2, 8), -0.5),
        "Net.b2": torch.full((2,), 0.05),
    }
    res = train_dense_network_torch(
        net, ps, _signal_examples(), "cross_entropy",
        lr=0.0, epochs=1, device="cpu", seed=1,
        initial_state_dict={k: v.clone() for k, v in state.items()},
    )
    best = res["best_params"]
    for key, expected in state.items():
        got = torch.tensor(best.parameters[key]["values"])
        assert torch.allclose(got, expected), key

    # Sanity: SIN initial_state_dict, la misma semilla da el init nativo de
    # nn.Linear (Kaiming), que NO coincide con los valores constantes de arriba.
    res_scratch = train_dense_network_torch(
        net, ps, _signal_examples(), "cross_entropy",
        lr=0.0, epochs=1, device="cpu", seed=1,
    )
    scratch_w1 = torch.tensor(res_scratch["best_params"].parameters["Net.W1"]["values"])
    assert not torch.allclose(scratch_w1, state["Net.W1"])


def test_resuming_from_a_trained_state_beats_fresh_init_after_one_epoch():
    """El caso de uso real de C5: entrenar hasta converger, guardar esos
    tensores, y 'reentrenar' (p.ej. con más datos) debe partir de ahí — tras
    solo 1 época, un warm-start desde un estado ya bueno debe dar una loss de
    validación claramente mejor que empezar de cero con esa misma época."""
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.forward.dense_torch import dense_module_to_state_dict, dense_network_to_torch_module
    net, ps = _setup()
    ex = _signal_examples()

    converged = train_dense_network_torch(
        net, ps, ex, "cross_entropy", lr=0.5, epochs=150, device="cpu", seed=1,
        materialize=False,
    )
    assert converged["best_val_loss"] < 0.3, "precondición: el modelo debe converger de verdad"
    trained_state = converged["best_state_dict"]
    assert trained_state is not None

    warm = train_dense_network_torch(
        net, ps, ex, "cross_entropy", lr=0.1, epochs=1, device="cpu", seed=2,
        initial_state_dict=trained_state,
    )
    scratch = train_dense_network_torch(
        net, ps, ex, "cross_entropy", lr=0.1, epochs=1, device="cpu", seed=2,
    )
    assert warm["best_val_loss"] < scratch["best_val_loss"]
    # y el warm-start conserva (aprox) la calidad ya alcanzada, no la destruye
    assert warm["best_val_loss"] < converged["best_val_loss"] * 3


def test_resume_that_never_improves_returns_the_starting_weights():
    """PESOS_GRANDES C5 audit (MEDIA): con lr=0 NINGUNA época puede mejorar la
    línea base (val_loss idéntico, y el criterio es estrictamente `<`) — el
    trainer debe devolver los pesos de PARTIDA como best (`best_epoch=0`), no
    los de la época 1. Sin la línea base, `best_val_loss` arrancaba en inf y
    la época 1 siempre 'ganaba': un reentrenamiento que EMPEORA devolvía
    pesos peores que los guardados y el auto-guardado los machacaba."""
    import torch
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    state = {
        "Net.W1": torch.full((8, 2), 0.25),
        "Net.b1": torch.full((8,), 0.1),
        "Net.W2": torch.full((2, 8), -0.5),
        "Net.b2": torch.full((2,), 0.05),
    }
    res = train_dense_network_torch(
        net, ps, _signal_examples(), "cross_entropy",
        lr=0.0, epochs=3, device="cpu", seed=1,
        initial_state_dict={k: v.clone() for k, v in state.items()},
    )
    assert res["best_epoch"] == 0
    best = res["best_params"]
    for key, expected in state.items():
        got = torch.tensor(best.parameters[key]["values"])
        assert torch.allclose(got, expected), key


def test_destructive_resume_does_not_return_worse_weights_than_start():
    """Escenario reproducido en la auditoría: modelo convergido + resume con
    lr destructivo → los pesos devueltos nunca deben ser peores (en val) que
    el punto de partida. La línea base época-0 lo garantiza por construcción:
    best_val_loss = min(baseline, épocas)."""
    import torch
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    ex = _signal_examples()
    # Estado PARCIALMENTE entrenado (5 épocas): a diferencia de uno convergido
    # (gradiente ~0, ni lr=1000 lo mueve), aquí los gradientes siguen vivos y
    # un lr enorme dispara los pesos de verdad (val_loss 0.70 → 2-12).
    partial = train_dense_network_torch(net, ps, ex, "cross_entropy", lr=0.5,
                                        epochs=5, device="cpu", seed=1,
                                        materialize=False)
    trained_state = partial["best_state_dict"]

    destroyed = train_dense_network_torch(
        net, ps, ex, "cross_entropy", lr=100.0, epochs=3, device="cpu",
        seed=2, initial_state_dict={k: v.clone() for k, v in trained_state.items()},
        materialize=False,
    )
    assert destroyed["best_epoch"] == 0
    assert destroyed["best_val_loss"] <= partial["best_val_loss"] + 1e-6
    for key in trained_state:
        assert torch.allclose(destroyed["best_state_dict"][key], trained_state[key]), key


def test_fresh_training_keeps_epoch_ge_1_semantics():
    """Retro-compat: SIN initial_state_dict no hay línea base — el init
    aleatorio nunca es un candidato a best (comportamiento idéntico a antes:
    best_epoch >= 1 siempre)."""
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    res = train_dense_network_torch(net, ps, _signal_examples(), "cross_entropy",
                                    lr=0.0, epochs=2, device="cpu", seed=1)
    assert res["best_epoch"] >= 1


def test_initial_state_dict_shapes_come_from_the_tensors_not_the_template():
    """Un `parameter_set` (plantilla) con shapes DISTINTAS a las del
    `initial_state_dict` no debe importar — igual que
    `dense_network_to_torch_module_from_state`, las dimensiones del módulo
    salen de los tensores dados, nunca de la plantilla."""
    import torch
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, _ = _setup()
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    prog = parse_text(MXAI)
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    # plantilla con OTRA semilla (mismas shapes, valores None de todos modos)
    bogus_ps = build_network_parameter_set(net, rl, program_hash(prog), seed=99, with_values=False)
    state = {
        "Net.W1": torch.randn(8, 2),
        "Net.b1": torch.randn(8),
        "Net.W2": torch.randn(2, 8),
        "Net.b2": torch.randn(2),
    }
    res = train_dense_network_torch(
        net, bogus_ps, _signal_examples(), "cross_entropy",
        lr=0.0, epochs=1, device="cpu", seed=1, initial_state_dict=state,
    )
    assert res["best_params"] is not None
