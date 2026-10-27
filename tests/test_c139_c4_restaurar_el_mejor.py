# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""139-C4 (núcleo) — `restaurar_el_mejor=False`: ÉPOCAS FIJAS, se entrega la ÚLTIMA época y no la de menor pérdida de validación.

Una red ganadora del Studio se reentrena con todo el desarrollo y las épocas de su ajuste de selección (`matrixai_engines`,
`con_epocas_fijas`). Eso exige un entrenamiento que ejecute EXACTAMENTE esas épocas y entregue lo último que aprendió: hasta
ahora los entrenadores torch de la red densa y de la compuesta restauraban SIEMPRE el mejor estado por la validación, con o
sin parada temprana. Lo que sostienen estas pruebas:

- por omisión (`True`) nada cambia: la mejor época, la misma traza de siempre;
- con `False` el estado entregado es el de la última época y `best_epoch`/`best_val_loss` lo dicen (no una mejor que no se
  entregó), en la densa y en la compuesta, y por la superficie pública (`run_playground_training`);
- con `False`, entrenar `n` épocas y entregar la última da EXACTAMENTE los pesos de entrenar `m > n` épocas y entregar la mejor
  cuando esa mejor es la `n` (la traza de las primeras `n` no depende de lo que viene después): lo que hace que fijar las
  épocas de la selección reproduzca el modelo de la selección.
"""
from __future__ import annotations

import random
import re
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

DENSA_MXAI = """PROJECT P
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

COMPUESTA_MXAI = """PROJECT P
VECTOR Input[2]
  cat: Integer[0, 8]
  x: Scalar
END
NETWORK Net
  INPUT Input
  EMBEDDING e1 FROM cat VOCAB 9 DIM 4
  CONCAT [e1, x] -> features
  LAYER Dense units=16 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT y: ProbabilityMap[A, B, C]
END
GRAPH
  Input -> Net
END
"""


def _densa():
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    prog = parse_text(DENSA_MXAI)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    return net, build_network_parameter_set(net, tr.resolved_layers or net.layers, program_hash(prog), seed=1)


def _compuesta():
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    prog = parse_text(COMPUESTA_MXAI)
    net = prog.networks[0]
    tr = check_composite_network_types(net, {v.name: v for v in prog.vectors})
    return net, build_composite_network_parameter_set(net, tr, model_hash_str=program_hash(prog), seed=1), tr


def _ruido_densa(n=200, seed=0):
    """Etiquetas al azar: la pérdida de validación no mejora con las épocas, así que la MEJOR época no es la última."""
    rng = random.Random(seed)
    return [([rng.random(), rng.random()], rng.choice([[0.0, 1.0], [1.0, 0.0]])) for _ in range(n)]


def _ruido_compuesta(n=120, seed=0):
    rng = random.Random(seed)
    filas = []
    for _ in range(n):
        etiqueta = [0.0, 0.0, 0.0]
        etiqueta[rng.randint(0, 2)] = 1.0
        filas.append(({"cat": rng.randint(0, 8), "x": rng.random()}, etiqueta))
    return filas


def _entrenar_densa(epochs, **k):
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _densa()
    return train_dense_network_torch(net, ps, _ruido_densa(), "cross_entropy", lr=0.05, epochs=epochs, device="cpu", seed=3,
                                     batch_size=8, materialize=True, **k)


def _entrenar_compuesta(epochs, **k):
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    net, ps, tr = _compuesta()
    return train_composite_network_torch(net, ps, _ruido_compuesta(), "cross_entropy", lr=0.05, epochs=epochs, device="cpu",
                                         seed=3, batch_size=8, materialize=True, type_result=tr, **k)


def _pesos(res):
    return {k: v["values"] for k, v in res["best_params"].parameters.items()}


ENTRENADORES = pytest.mark.parametrize("entrenar", [_entrenar_densa, _entrenar_compuesta], ids=["densa", "compuesta"])


@ENTRENADORES
def test_por_omision_se_entrega_la_mejor_epoca_y_con_ruido_no_es_la_ultima(entrenar):
    res = entrenar(12)
    ultima = res["epochs"][-1]["epoch"]
    assert ultima == 12
    assert res["best_epoch"] < ultima, res["best_epoch"]            # el control: con ruido la mejor NO es la última
    assert res["best_val_loss"] == min(e["val_loss"] for e in res["epochs"])


@ENTRENADORES
def test_con_restaurar_el_mejor_falso_se_entrega_la_ultima_y_best_epoch_lo_dice(entrenar):
    res = entrenar(12, restaurar_el_mejor=False)
    assert len(res["epochs"]) == 12
    assert res["best_epoch"] == 12
    assert res["best_val_loss"] == res["epochs"][-1]["val_loss"]


@ENTRENADORES
def test_los_pesos_de_la_ultima_son_los_de_la_traza_y_distintos_de_los_de_la_mejor(entrenar):
    de_la_mejor = entrenar(12)
    de_la_ultima = entrenar(12, restaurar_el_mejor=False)
    # misma semilla, mismas épocas: la traza es IDÉNTICA (restaurar no cambia cómo se entrena)…
    assert [e["val_loss"] for e in de_la_mejor["epochs"]] == [e["val_loss"] for e in de_la_ultima["epochs"]]
    # …y lo entregado es otra cosa: la mejor época no es la última, así que los pesos difieren
    assert _pesos(de_la_mejor) != _pesos(de_la_ultima)


@ENTRENADORES
def test_n_epocas_y_la_ultima_son_los_pesos_de_m_epocas_y_la_mejor_cuando_la_mejor_es_la_n(entrenar):
    """LA PROPIEDAD QUE USA EL STUDIO: la mejor época de un entrenamiento largo, reentrenada con ese número de épocas y
    entregando la última, da EXACTAMENTE los mismos pesos."""
    largo = entrenar(12)
    n = largo["best_epoch"]
    assert 1 <= n < 12
    corto = entrenar(n, restaurar_el_mejor=False)
    assert _pesos(corto) == _pesos(largo)


@ENTRENADORES
def test_sin_parada_temprana_con_restaurar_falso_ejecuta_exactamente_las_epocas_pedidas(entrenar):
    res = entrenar(9, restaurar_el_mejor=False)         # `early_stop=None` por omisión
    assert [e["epoch"] for e in res["epochs"]] == list(range(1, 10))
    assert not res.get("parado_por_plazo")


def test_por_la_superficie_publica_run_playground_training_la_densa_entrega_la_ultima():
    """Del extremo al otro: `run_playground_training(..., restaurar_el_mejor=False)` (lo que llama el motor de `matrixai_engines`)
    llega al entrenador. Sin el argumento, la mejor época; con él, la última."""
    from matrixai.playground_api import generate_project_from_dataset, run_playground_training

    rng = random.Random(0)
    csv = "x1,x2,y\n" + "\n".join(f"{rng.random():.4f},{rng.random():.4f},{rng.choice(['si', 'no'])}" for _ in range(160))
    gen = generate_project_from_dataset(csv, "y", locale="es")
    # una tasa alta: la pérdida de validación SALTA de una época a otra y la mejor no es la última
    texto = re.sub(r"LEARNING_RATE\s+\S+", "LEARNING_RATE 2.0", gen["training_text"])
    assert "LEARNING_RATE 2.0" in texto

    def correr(**k):
        return run_playground_training(gen["mxai"], texto, gen["csv_text"], epochs_override=60,
                                       field_ranges=gen.get("field_ranges"), target_range=gen.get("target_range"), seed=5,
                                       preferir_torch_en_cpu=True, **k)

    mejor, ultima = correr(), correr(restaurar_el_mejor=False)
    assert mejor["backend"] == ultima["backend"] == "torch"
    assert len(ultima["epochs"]) == 60 and ultima["best_epoch"] == 60
    assert mejor["best_epoch"] < 60
