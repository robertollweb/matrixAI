# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Auditoría de Roberto del contrato 139 (P2-2) — CON ÉPOCAS FIJAS, UNA ÚLTIMA ÉPOCA NO FINITA NO SE ENTREGA.

Con `restaurar_el_mejor=False` (139-C4: la red ganadora se reentrena con las épocas de su ajuste de selección) se entrega la
ÚLTIMA época ejecutada. Roberto lo reprodujo con 2 filas, 5 épocas y `lr = 1e10`: pesos NaN y pérdida NaN, entregados sin
ningún fallo (`ok: True`). TabM ya lo comprobaba (`densa_tabm`, «Una pérdida no finita al final es una red rota»); los dos
entrenadores torch del núcleo (densa y compuesta) no. Lo que sostienen estas pruebas:

- la reproducción de Roberto, por la superficie pública (`run_playground_training`, lo que llama el motor): `ok: False`, con su
  motivo bilingüe y sin caer al entrenador stdlib (no es un fallo de la GPU);
- el entrenador denso y el compuesto lanzan `UltimaEpocaNoFinita` con la época y la pérdida;
- unos pesos no finitos con una pérdida finita, también;
- el control: con `restaurar_el_mejor=True` (lo de siempre) la misma divergencia no lanza (allí se entrega la mejor época, otra
  regla), y un entrenamiento normal con épocas fijas sigue entregando la última.
"""
from __future__ import annotations

import math
import random
import re
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _csv_de_regresion(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "x1,x2,y\n" + "\n".join(f"{rng.random():.4f},{rng.random():.4f},{rng.random() * 100:.3f}" for _ in range(n))


def _entrenar(csv: str, *, lr: str, epocas: int, **k):
    from matrixai.playground_api import generate_project_from_dataset, run_playground_training

    gen = generate_project_from_dataset(csv, "y", locale="es")
    texto = re.sub(r"LEARNING_RATE\s+\S+", f"LEARNING_RATE {lr}", gen["training_text"])
    assert f"LEARNING_RATE {lr}" in texto
    return run_playground_training(gen["mxai"], texto, gen["csv_text"], epochs_override=epocas,
                                   field_ranges=gen.get("field_ranges"), target_range=gen.get("target_range"), seed=5,
                                   preferir_torch_en_cpu=True, **k)


def _plano(x):
    if isinstance(x, dict):
        for y in x.values():
            yield from _plano(y)
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _plano(y)
    elif isinstance(x, float):
        yield x


class TestLaReproduccionDeRoberto:
    """2 filas, 5 épocas, `lr = 1e10`, épocas fijas: antes `ok: True` con 254 de 641 pesos NaN."""

    def test_con_epocas_fijas_falla_con_su_motivo_y_no_entrega_pesos(self):
        from matrixai.training.dense_torch_trainer import CLAVE_ULTIMA_EPOCA_NO_FINITA

        r = _entrenar(_csv_de_regresion(2), lr="10000000000.0", epocas=5, restaurar_el_mejor=False)
        assert r["ok"] is False, {k: r.get(k) for k in ("ok", "backend", "best_epoch", "best_val_loss")}
        assert "params_best" not in r                                     # nada que entregar
        assert r["error"].startswith(CLAVE_ULTIMA_EPOCA_NO_FINITA + ":"), r["error"]
        motivo = r["motivo"]
        assert set(motivo) == {"es", "en"} and motivo["es"] != motivo["en"]
        assert "la última ejecutada (la 5)" in motivo["es"] and "no se entrega" in motivo["es"]
        assert "the last one run (epoch 5)" in motivo["en"] and "not delivered" in motivo["en"]
        assert "nan" in motivo["es"] and "nan" in motivo["en"]

    def test_no_cae_al_entrenador_stdlib(self, caplog):
        # Si cayera, el registro diría «falling back to stdlib» y el resultado sería de OTRO entrenador con el mismo nombre.
        with caplog.at_level("WARNING"):
            r = _entrenar(_csv_de_regresion(2), lr="10000000000.0", epocas=5, restaurar_el_mejor=False)
        assert r["ok"] is False
        assert "falling back to stdlib" not in caplog.text

    def test_el_control_con_la_mejor_epoca_no_lanza(self):
        # Lo de siempre (restaurar el MEJOR) no cambia: allí ninguna época mejora y se entregan los pesos de partida.
        r = _entrenar(_csv_de_regresion(2), lr="10000000000.0", epocas=5)
        assert r["ok"] is not False and r["backend"] == "torch"

    def test_el_control_un_entrenamiento_normal_con_epocas_fijas_entrega_la_ultima(self):
        r = _entrenar(_csv_de_regresion(40), lr="0.01", epocas=5, restaurar_el_mejor=False)
        assert r["ok"] is not False and r["backend"] == "torch" and r["best_epoch"] == 5
        valores = list(_plano(r["params_best"].get("parameters", r["params_best"])))
        assert valores and all(math.isfinite(v) for v in valores)


class TestLosEntrenadores:

    def test_la_densa_lanza_con_la_epoca_y_la_perdida(self):
        from matrixai.training.dense_torch_trainer import UltimaEpocaNoFinita, train_dense_network_torch
        from tests.test_c139_c4_restaurar_el_mejor import _densa

        net, ps = _densa()
        ejemplos = [([0.1, 0.9], [0.0, 1.0]), ([0.8, 0.2], [1.0, 0.0])]
        # `mse` sobre la salida de probabilidades acota la pérdida; con `inf` como tasa los pesos sí se van a NaN.
        with pytest.raises(UltimaEpocaNoFinita) as exc:
            train_dense_network_torch(net, ps, ejemplos, "mse", lr=float("inf"), epochs=3, device="cpu", seed=3,
                                      batch_size=8, materialize=True, restaurar_el_mejor=False,
                                      validation_examples=ejemplos)
        assert exc.value.epoca == 3 and exc.value.pesos_finitos is False
        assert set(exc.value.motivo) == {"es", "en"}

    def test_la_compuesta_lanza_igual(self):
        from matrixai.training.composite_torch_trainer import train_composite_network_torch
        from matrixai.training.dense_torch_trainer import UltimaEpocaNoFinita
        from tests.test_c139_c4_restaurar_el_mejor import _compuesta, _ruido_compuesta

        net, ps, tr = _compuesta()
        with pytest.raises(UltimaEpocaNoFinita) as exc:
            train_composite_network_torch(net, ps, _ruido_compuesta(), "cross_entropy", lr=float("inf"), epochs=2,
                                          device="cpu", seed=3, batch_size=8, materialize=True, type_result=tr,
                                          restaurar_el_mejor=False)
        assert exc.value.epoca == 2

    def test_unos_pesos_no_finitos_con_la_perdida_finita_tampoco_se_entregan(self):
        import torch

        from matrixai.training.dense_torch_trainer import UltimaEpocaNoFinita, exigir_ultima_epoca_finita

        traza = [{"epoch": 1, "loss": 0.5, "val_loss": 0.4}, {"epoch": 2, "loss": 0.4, "val_loss": 0.3}]
        exigir_ultima_epoca_finita([torch.ones(3), torch.zeros(2)], traza, torch)          # finito: no lanza
        with pytest.raises(UltimaEpocaNoFinita) as exc:
            exigir_ultima_epoca_finita([torch.ones(3), torch.tensor([0.0, float("inf")])], traza, torch)
        assert exc.value.epoca == 2 and exc.value.pesos_finitos is False and exc.value.val_loss == 0.3
        assert "pesos no finitos" in exc.value.motivo["es"] and "pérdida" not in exc.value.motivo["es"]
        with pytest.raises(UltimaEpocaNoFinita) as exc:
            exigir_ultima_epoca_finita([torch.ones(3)], [*traza[:1], {"epoch": 2, "loss": 0.4, "val_loss": float("nan")}],
                                       torch)
        assert exc.value.pesos_finitos is True and "pérdida de validación no finita" in exc.value.motivo["es"]
        assert "pesos" not in exc.value.motivo["es"]
