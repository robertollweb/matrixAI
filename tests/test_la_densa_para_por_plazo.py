# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""LA DENSA PARA SOLA DENTRO DE SU PRESUPUESTO (2026-09-22) — el `plazo` del entrenador.

Medido en Fase 0 (pasadas 101-C5 y 113-C4): 12 intentos de la densa muertos a 643-646 s
de un tope de 630, en KDDCup09 y Allstate, mientras los otros seis motores terminaban
dentro. No era el tope ni la máquina: el bucle de épocas no miraba el reloj, así que la
mataban desde fuera y no devolvía ni la mejor época que ya tenía. En el producto eso es
«el motor falló» donde debía ser «entrené lo que cupo».

Lo que estas pruebas sostienen, cada una con su nombre:
- un plazo vencido para tras UN lote, lo declara, y entrega un modelo evaluado;
- un plazo que no se alcanza da EXACTAMENTE los mismos pesos que sin plazo — es la
  garantía de que los 487 intentos de la densa que sí terminaron no cambian;
- lo que se entrega es la MEJOR época vista, no la última a medias;
- el plazo llega de verdad desde la superficie pública (`run_playground_training`).
"""
from __future__ import annotations

import random
import time
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


def _ejemplos(n=200, seed=0):
    rng = random.Random(seed)
    filas = []
    for _ in range(n):
        a, b = rng.random(), rng.random()
        filas.append(([a, b], [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))
    return filas


def _entrenar(**kw):
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    return train_dense_network_torch(net, ps, _ejemplos(), "cross_entropy", lr=kw.pop("lr", 0.05),
                                     epochs=kw.pop("epochs", 6), device="cpu", seed=3,
                                     batch_size=8, materialize=True, **kw)


def _perdida_de_validacion(res):
    """La pérdida de validación de los pesos DEVUELTOS, recalculada fuera del
    entrenador: mismas filas de validación (el 20 % final, sin `validation_examples`)
    y la misma pérdida."""
    import torch
    from matrixai.forward.dense_torch import dense_network_to_torch_module
    from matrixai.training.dense_torch_trainer import _loss_on_probabilities
    net, _ = _setup()
    modulo = dense_network_to_torch_module(net, res["best_params"])
    val = _ejemplos()[160:]
    x = torch.tensor([f for f, _ in val], dtype=torch.float32)
    y = torch.tensor([t for _, t in val], dtype=torch.float32)
    modulo.eval()
    with torch.no_grad():
        return float(_loss_on_probabilities(modulo(x), y, "cross_entropy", torch))


def _pesos(res):
    return {k: v["values"] for k, v in res["best_params"].parameters.items()}


def test_un_plazo_vencido_para_tras_UN_lote_lo_declara_y_entrega_un_modelo():
    pasos = []
    res = _entrenar(plazo=time.monotonic() - 1.0, cancel_check=lambda: pasos.append(1))
    assert res["parado_por_plazo"] is True
    # UN lote: se mira después de cada uno, y el primero ya encuentra el plazo pasado.
    assert len(pasos) == 1, len(pasos)
    # Una sola época en la traza, marcada A MEDIAS, y con su pérdida de validación
    # medida: lo entregado está evaluado, no es un estado cualquiera.
    assert len(res["epochs"]) == 1 and res["epochs"][0].get("parcial") is True, res["epochs"]
    assert res["best_epoch"] == 1
    assert res["best_val_loss"] == res["epochs"][0]["val_loss"]
    assert res["best_params"] is not None


def test_un_plazo_que_no_se_alcanza_da_EXACTAMENTE_los_mismos_pesos_que_sin_plazo():
    """La garantía de que el arreglo no mueve lo ya medido: la densa que terminaba
    dentro de su presupuesto entrena igual, bit a bit, porque el plazo solo compara un
    reloj."""
    sin = _entrenar()
    con = _entrenar(plazo=time.monotonic() + 3600.0)
    assert sin["parado_por_plazo"] is False and con["parado_por_plazo"] is False
    assert _pesos(sin) == _pesos(con)
    assert [e["val_loss"] for e in sin["epochs"]] == [e["val_loss"] for e in con["epochs"]]
    assert not any(e.get("parcial") for e in con["epochs"])


LR_QUE_EMPEORA = 3.0


def test_lo_entregado_es_la_MEJOR_epoca_vista_no_la_ultima_a_medias(monkeypatch):
    """El plazo salta en la TERCERA época, a mitad. Si esa época a medias es peor que
    una anterior, se devuelve la anterior — es lo que dice `best_state`, y el plazo no
    puede saltárselo."""
    import matrixai.training.dense_torch_trainer as trainer

    reloj = {"t": 0.0}
    lotes = {"n": 0}
    lotes_por_epoca = 200 * 4 // 5 // 8  # 80 % de 200 filas en lotes de 8 = 20

    def monotonic():
        return reloj["t"]

    def contar():
        lotes["n"] += 1
        # A mitad de la tercera época, el reloj salta por encima del plazo.
        if lotes["n"] == 2 * lotes_por_epoca + lotes_por_epoca // 2:
            reloj["t"] = 100.0

    monkeypatch.setattr(trainer.time, "monotonic", monotonic)
    # lr alto a propósito: con él la época a medias sale PEOR que una anterior (se
    # comprueba abajo), que es lo que hace falta para distinguir «la mejor» de «la última».
    res = _entrenar(plazo=50.0, cancel_check=contar, epochs=10, lr=LR_QUE_EMPEORA)
    assert res["parado_por_plazo"] is True
    assert len(res["epochs"]) == 3 and res["epochs"][-1].get("parcial") is True
    perdidas = [e["val_loss"] for e in res["epochs"]]
    # El caso tiene dientes solo si la última NO es la mejor: si lo fuera, devolver la
    # última pasaría igual. Se exige, en vez de esperar que pase.
    assert perdidas[-1] > min(perdidas), perdidas
    assert res["best_val_loss"] == min(perdidas), perdidas
    assert res["best_epoch"] == perdidas.index(min(perdidas)) + 1
    # Y los PESOS devueltos son los de esa época, no solo las cifras: recalculada con
    # ellos, la pérdida de validación es la que se declara. Sin esto, quedarse con los
    # pesos de la época a medias y dejar bien `best_val_loss` pasaría la prueba.
    assert abs(_perdida_de_validacion(res) - res["best_val_loss"]) < 1e-6


def test_el_plazo_llega_desde_la_superficie_publica(monkeypatch):
    """`run_playground_training` es lo que llama el motor denso: si el plazo se
    quedara por el camino, el arreglo no existiría para quien lo usa. Con torch en CPU
    FORZADO, como lo pide el motor: en `auto` el núcleo podría ir por stdlib y esta
    prueba no probaría nada."""
    from matrixai.playground import MODO_TORCH_EN_CPU
    from matrixai.playground_api import generate_project_from_dataset, run_playground_training
    from matrixai import limits
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", MODO_TORCH_EN_CPU)

    rng = random.Random(0)
    lineas = ["a,b,y"] + [f"{rng.random():.4f},{rng.random():.4f},{'si' if i % 2 else 'no'}"
                          for i in range(120)]
    gen = generate_project_from_dataset("\n".join(lineas) + "\n", "y", locale="es",
                                        column_type_overrides={"y": "categorical"})
    with limits.sin_topes():
        vencido = run_playground_training(gen["mxai"], gen["training_text"], gen["csv_text"],
                                          epochs_override=5, field_ranges=gen.get("field_ranges"),
                                          seed=1, plazo=time.monotonic() - 1.0)
        libre = run_playground_training(gen["mxai"], gen["training_text"], gen["csv_text"],
                                        epochs_override=5, field_ranges=gen.get("field_ranges"),
                                        seed=1)
    assert vencido.get("backend") == "torch", vencido.get("backend")
    assert vencido["ok"] and vencido["parado_por_plazo"] is True, vencido.get("error")
    assert len(vencido["epochs"]) == 1
    assert libre["ok"] and libre["parado_por_plazo"] is False
    assert len(libre["epochs"]) > 1


# --- LA RED COMPUESTA (2026-09-23) ------------------------------------------------------
#
# Medido con espías sobre KDDCup09: el bucle DENSO ni se llamaba. Con una categórica de
# alta cardinalidad el generador produce una red COMPUESTA (EMBEDDING + CONCAT) y el
# entrenamiento va por `train_composite_network_torch`, que no recibía el plazo; y el
# motor declaraba un plazo de 450 s que nadie aplicaba. Estas pruebas van por la
# superficie pública con un CSV que produce esa red de verdad, y lo COMPRUEBAN.

def _proyecto_compuesto():
    from matrixai.playground import _network_is_composite
    from matrixai.playground_api import generate_project_from_dataset
    rng = random.Random(0)
    lineas = ["x,c,y"] + [f"{rng.random():.4f},v{rng.randrange(300)},{'si' if rng.random() > 0.5 else 'no'}"
                          for _ in range(600)]
    gen = generate_project_from_dataset("\n".join(lineas) + "\n", "y", locale="es",
                                        column_type_overrides={"y": "categorical"})
    assert _network_is_composite(gen["mxai"]), "el CSV tenía que producir una red COMPUESTA"
    return gen


def _entrenar_compuesta(gen, **kw):
    from matrixai import limits
    from matrixai.playground_api import run_playground_training
    with limits.sin_topes():
        return run_playground_training(gen["mxai"], gen["training_text"], gen["csv_text"],
                                       epochs_override=5, field_ranges=gen.get("field_ranges"),
                                       seed=1, **kw)


def test_la_compuesta_tambien_para_por_plazo_y_lo_declara(monkeypatch):
    from matrixai.playground import MODO_TORCH_EN_CPU
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", MODO_TORCH_EN_CPU)
    gen = _proyecto_compuesto()
    vencido = _entrenar_compuesta(gen, plazo=time.monotonic() - 1.0)
    assert vencido.get("backend") == "torch", vencido.get("backend")
    assert vencido["ok"] and vencido["network_kind"] == "composite_network"
    assert vencido["parado_por_plazo"] is True
    assert len(vencido["epochs"]) == 1


def test_la_compuesta_con_un_plazo_lejano_da_los_MISMOS_pesos_y_dice_que_no_paro(monkeypatch):
    from matrixai.playground import MODO_TORCH_EN_CPU
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", MODO_TORCH_EN_CPU)
    gen = _proyecto_compuesto()
    sin = _entrenar_compuesta(gen)
    con = _entrenar_compuesta(gen, plazo=time.monotonic() + 3600.0)
    assert sin["params_best"] == con["params_best"]
    assert [e["validation_loss"] for e in sin["epochs"]] == [e["validation_loss"] for e in con["epochs"]]
    # La clave VA siempre por el camino torch: su ausencia es la que dice «no se aplicó».
    assert sin["parado_por_plazo"] is False and con["parado_por_plazo"] is False
