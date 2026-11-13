# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""CONTRATO 167-C1 — `paso_de_un_peso`: la mini-ejecución SGD de UN peso.

- `test_paridad_con_optimizer_step_*`: peso_antes − tasa × gradiente_efectivo == lo que deja
  el `optimizer.step()` real (con y sin decaimiento, capa 1 y 2);
- `test_reproduce_la_ejecucion_real_*`: el peso tras cada paso coincide EXACTO con el que deja
  el entrenador de verdad con la misma semilla (se GRABA el `SGD.step` real, no se deduce);
- `test_gradiente_es_el_de_la_mano`: el gradiente de una neurona sigmoide coincide con la
  fórmula escrita a mano;
- topes, motivos (Adam, fuera de rango, no finito) y determinismo.
"""
from __future__ import annotations

import random
from importlib import util

import pytest

needs_torch = pytest.mark.skipif(util.find_spec("torch") is None, reason="torch not installed")

MXAI = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=%d activation=%s
  %s
  OUTPUT y: %s
END
GRAPH
  In -> Net
END
"""
DOS_CAPAS = MXAI % (8, "relu", "LAYER Dense units=2 activation=softmax", "ProbabilityMap[A, B]")
UNA_NEURONA = MXAI % (1, "sigmoid", "", "Probability")


def _setup(texto=DOS_CAPAS):
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(texto)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    return net, build_network_parameter_set(net, rl, program_hash(prog), seed=1)


def _ejemplos(n=200, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.random(), rng.random()
        out.append(([a, b], [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))
    return out


def _mini(capa=1, fila=0, columna=0, **kw):
    from matrixai.training.paso_de_un_peso import paso_de_un_peso
    net, ps = _setup()
    kw.setdefault("lr", 0.05)
    kw.setdefault("batch_size", 8)
    kw.setdefault("seed", 3)
    return paso_de_un_peso(net, ps, _ejemplos(), "cross_entropy", capa=capa, fila=fila,
                           columna=columna, **kw)


@needs_torch
@pytest.mark.parametrize("capa,fila,col,wd", [(1, 0, 0, 0.0), (1, 3, 1, 0.1), (2, 1, 5, 0.0), (2, 0, 2, 0.3)])
def test_paridad_con_optimizer_step(capa, fila, col, wd):
    r = _mini(capa, fila, col, weight_decay=wd)
    assert r["ok"] and r["motivo"] is None
    assert len(r["pasos"]) == 20 and r["pasos_aplicados"] == 20
    for p in r["pasos"]:
        assert p["gradiente_efectivo"] == pytest.approx(p["gradiente"] + wd * p["peso_antes"], abs=1e-7)
        assert abs((p["peso_antes"] - p["tasa"] * p["gradiente_efectivo"]) - p["peso_despues_torch"]) <= 1e-6
        assert p["cuadra"]
    # el peso de cada paso es el «después» del anterior
    for a, b in zip(r["pasos"], r["pasos"][1:]):
        assert b["peso_antes"] == a["peso_despues_torch"]
    assert r["paridad"]["error_maximo"] < 1e-6


@needs_torch
@pytest.mark.parametrize("con_validacion", [False, True])
def test_reproduce_la_ejecucion_real(con_validacion):
    """La ejecución normal (entrenador real, misma semilla) pasa por los mismos pesos."""
    import torch
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    ejemplos = _ejemplos()
    val = ejemplos[:40] if con_validacion else None
    net, ps = _setup()
    capa, fila, col = 2, 1, 4
    r = _mini(capa, fila, col, weight_decay=0.1, validation_examples=val)
    assert r["ok"]
    grabado: list[float] = []
    original = torch.optim.SGD.step

    class _Fin(Exception):
        pass

    def grabar(self, *a, **k):
        out = original(self, *a, **k)
        grabado.append(float(self.param_groups[0]["params"][2 * (capa - 1)][fila, col].detach()))
        return out

    def cortar():
        if len(grabado) >= 20:
            raise _Fin

    torch.optim.SGD.step = grabar
    try:
        with pytest.raises(_Fin):
            train_dense_network_torch(net, ps, ejemplos, "cross_entropy", lr=0.05, epochs=3, device="cpu",
                                      seed=3, batch_size=8, materialize=True, weight_decay=0.1,
                                      validation_examples=val, cancel_check=cortar)
    finally:
        torch.optim.SGD.step = original
    assert grabado == [p["peso_despues_torch"] for p in r["pasos"]]   # igualdad EXACTA


@needs_torch
def test_gradiente_es_el_de_la_mano():
    """Una neurona sigmoide + BCE, un lote con todo: g_j = mean((p − y)·x_j)."""
    import math
    from matrixai.training.paso_de_un_peso import paso_de_un_peso
    net, ps = _setup(UNA_NEURONA)
    ps.parameters["Net.W1"]["values"] = [[0.3, -0.2]]
    ps.parameters["Net.b1"]["values"] = [-0.1]
    filas = [([0.9, 0.1], [1.0]), ([0.2, 0.8], [0.0]), ([0.7, 0.6], [1.0]), ([0.1, 0.3], [0.0])]
    r = paso_de_un_peso(net, ps, filas, "binary_cross_entropy", capa=1, fila=0, columna=1, lr=0.5,
                        seed=1, batch_size=4, validation_examples=filas[:2], n_pasos=1)
    assert r["ok"] and len(r["pasos"]) == 1
    p0 = r["pasos"][0]
    sig = lambda z: 1 / (1 + math.exp(-z))
    g = sum((sig(0.3 * x[0] - 0.2 * x[1] - 0.1) - y[0]) * x[1] for x, y in filas) / 4
    assert p0["peso_antes"] == pytest.approx(-0.2, abs=1e-7)
    assert p0["gradiente"] == pytest.approx(g, abs=1e-6)
    assert p0["peso_despues_torch"] == pytest.approx(-0.2 - 0.5 * g, abs=1e-6)


@needs_torch
def test_determinista_y_declara_topes():
    a, b = _mini(), _mini()
    assert a == b
    c = _mini(n_pasos=500)
    assert c["pasos_pedidos"] == 500 and c["pasos_aplicados"] == 20 and c["tope_pasos"] == 20
    assert len(c["pasos"]) == 20
    d = _mini(n_pasos=3)
    assert len(d["pasos"]) == 3 and d["pasos"] == a["pasos"][:3]


@needs_torch
def test_pocos_lotes_no_inventa_pasos():
    from matrixai.training.paso_de_un_peso import paso_de_un_peso
    net, ps = _setup()
    r = paso_de_un_peso(net, ps, _ejemplos(30), "cross_entropy", capa=1, fila=0, columna=0, lr=0.05,
                        seed=3, batch_size=8, validation_examples=_ejemplos(5))
    assert r["lotes_por_epoca"] == 4 and len(r["pasos"]) == 4 and r["ok"]


@needs_torch
@pytest.mark.parametrize("opt", ["adam", "adamw"])
def test_adam_sale_apagado_con_su_motivo(opt):
    r = _mini(optimizer=opt)
    assert not r["ok"] and r["pasos"] == []
    assert set(r["motivo"]) == {"es", "en"} and "SGD" in r["motivo"]["es"]


@needs_torch
@pytest.mark.parametrize("capa,fila,col", [(0, 0, 0), (3, 0, 0), (1, 8, 0), (1, 0, 2), (2, -1, 0)])
def test_peso_inexistente_se_declara(capa, fila, col):
    r = _mini(capa, fila, col)
    assert not r["ok"] and r["pasos"] == [] and r["motivo"]["es"]


@needs_torch
def test_no_finito_se_declara_y_no_se_dibuja():
    from matrixai.training.paso_de_un_peso import paso_de_un_peso
    net, ps = _setup()
    ej = _ejemplos(20)
    ej[0] = ([float("nan"), 0.5], ej[0][1])
    r = paso_de_un_peso(net, ps, ej, "cross_entropy", capa=1, fila=0, columna=0, lr=0.05, seed=3,
                        batch_size=20, validation_examples=ej[:2])
    assert not r["ok"] and r["pasos"][-1].get("no_finito") and "finit" in r["motivo"]["es"]
    assert r["pasos"][-1]["gradiente"] is None


@needs_torch
@pytest.mark.parametrize("desvio", [1e-3, 1e-5])
def test_si_torch_no_cuadra_con_la_cuenta_no_se_da_por_buena(monkeypatch, desvio):
    """El invariante 2 (supervisor, 13-11): la cuenta a mano se COMPARA con el `optimizer.step()` real. Si el paso
    real se aparta de `w − tasa × gradiente`, el resultado no vale: `ok` False, el motivo de paridad y cada paso
    marcado. Sin esta prueba, `"cuadra": True` fijo o una tolerancia de 0,1 pasaban la suite entera (medido)."""
    import torch
    from matrixai.training import paso_de_un_peso as m

    original = torch.optim.SGD.step

    def paso_desviado(self, *a, **kw):
        salida = original(self, *a, **kw)
        with torch.no_grad():
            for grupo in self.param_groups:
                for p in grupo["params"]:
                    p.add_(desvio)
        return salida

    monkeypatch.setattr(torch.optim.SGD, "step", paso_desviado)
    r = _mini(capa=1, fila=0, columna=0)
    assert r["ok"] is False
    assert r["motivo"] == m.MOTIVO_PARIDAD
    assert r["pasos"] and not any(p["cuadra"] for p in r["pasos"])
    assert r["paridad"]["error_maximo"] >= desvio * 0.9
