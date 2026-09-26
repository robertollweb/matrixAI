# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""118-C3b — los dos caminos de entrenamiento que `47a0af4` no cubrió.

Cazados al preparar la release 1.11.0 (2026-09-26), leyendo quién llama a los entrenadores:

1. **El camino `layer_call` del playground** (`_run_playground_generic_training`,
   `GenericSupervisedTrainer`, diferencias finitas) es SGD plano y no miraba el optimizador:
   un `WEIGHT_DECAY`, un `SCHEDULE cosine` o un `TYPE adamw` se entrenaban IGNORADOS. Ahora se
   niega con su motivo, como los demás caminos sin torch.
2. **El transformer de la CLI** (`TransformerSupervisedTrainer`, que también usa
   `export/verify.py`) llamaba a `train_composite_network_torch` sin `weight_decay` ni
   `schedule`: la receta declarada se perdía. El playground ya las pasaba.

`TYPE adam` en el camino `layer_call` se sigue ignorando (deuda anterior, en TASKS): hay
generadores que lo escriben, y negarlo rompería modelos que hoy entrenan.
"""
from __future__ import annotations

import csv as _csv
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from matrixai.training.parser import parse_training_text

_RAIZ = Path(__file__).resolve().parents[1]
_MXAI = (_RAIZ / "examples" / "transformer-classifier-vector.mxai").read_text(encoding="utf-8")
_TRAINING = (_RAIZ / "examples" / "transformer-classifier-vector.mxtrain").read_text(encoding="utf-8")
_CSV = (_RAIZ / "examples" / "transformer-classifier-vector.train.csv").read_text(encoding="utf-8")


def _con_optimizador(extra: str = "", tipo: str = "sgd") -> str:
    texto = _TRAINING.replace("  TYPE sgd\n  LEARNING_RATE 0.01\n",
                              f"  TYPE {tipo}\n  LEARNING_RATE 0.01\n{extra}")
    assert texto != _TRAINING or (extra == "" and tipo == "sgd")
    return texto


# ---------------------------------------------------------------------------
# 1. EL CAMINO layer_call SE NIEGA, NO IGNORA
# ---------------------------------------------------------------------------

class TestLayerCallSeNiega:
    def _entrenar(self, texto: str) -> dict:
        from matrixai.playground import _run_playground_generic_training
        return _run_playground_generic_training(_MXAI, texto, _CSV, epochs_override=1)

    def test_el_ejemplo_es_de_verdad_un_modelo_layer_call(self):
        from matrixai.playground import _get_prediction_kind
        assert _get_prediction_kind(_MXAI, _TRAINING) == "layer_call"

    def test_se_niega_con_weight_decay(self):
        r = self._entrenar(_con_optimizador("  WEIGHT_DECAY 0.01\n"))
        assert r["ok"] is False
        assert "WEIGHT_DECAY" in r["error"]

    def test_se_niega_con_schedule_cosine(self):
        r = self._entrenar(_con_optimizador("  SCHEDULE cosine\n"))
        assert r["ok"] is False
        assert "SCHEDULE" in r["error"]

    def test_se_niega_con_type_adamw(self):
        r = self._entrenar(_con_optimizador(tipo="adamw"))
        assert r["ok"] is False
        assert "adamw" in r["error"]

    def test_sin_las_lineas_nuevas_entrena_como_siempre(self):
        r = self._entrenar(_TRAINING)
        assert r["ok"] is True, r.get("error")


# ---------------------------------------------------------------------------
# 2. EL TRANSFORMER DE LA CLI PASA LA RECETA AL ENTRENADOR DE TORCH
# ---------------------------------------------------------------------------

_L = 12

_TOY_MXAI = """
PROJECT Toy
SEQUENCE Texto
  length = 12
  vocab_size = 16
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 16
  BLOCK enc TRANSFORMER
    LAYERS 1
    HEADS 2
    FF 32
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[antes, despues]
END

GRAPH
  Texto -> N
END
"""


def _toy_mxtrain(optimizador: str) -> str:
    return f"""
MODEL toy.mxai

DATASET D
  SOURCE csv("toy.csv")
  INPUT Texto FROM COLUMNS [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]
  TARGET clase: Label[antes, despues]
  SPLIT train=0.7 validation=0.3 seed=1
END

LOSS L
  TYPE cross_entropy
  PREDICTION clase
  TARGET clase
END

OPTIMIZER O
{optimizador}  UPDATE N.*
END

RUN
  EPOCHS 2
END
"""


def _escribir_toy(d: Path) -> None:
    rng = random.Random(3)
    with (d / "toy.csv").open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow([f"t{i}" for i in range(_L)] + ["clase"])
        for _ in range(16):
            p7, p9 = rng.sample(range(_L), 2)
            fila = [rng.choice([1, 2, 3, 4, 5, 6]) for _ in range(_L)]
            fila[p7], fila[p9] = 7, 9
            w.writerow(fila + ["antes" if p7 < p9 else "despues"])
    (d / "toy.mxai").write_text(_TOY_MXAI, encoding="utf-8")


class TestTransformerDeLaCLI:
    def _kwargs_que_llegan(self, optimizador: str, monkeypatch) -> dict:
        pytest.importorskip("torch")
        import matrixai.training.composite_torch_trainer as ctt
        from matrixai.training.transformer_trainer import TransformerSupervisedTrainer

        visto: dict = {}
        real = ctt.train_composite_network_torch

        def _espia(net, ps, examples, loss_fn, **kwargs):
            visto.update(kwargs)
            return real(net, ps, examples, loss_fn, **kwargs)

        # `transformer_trainer` importa la función DENTRO de `.train()`: basta con el módulo origen.
        monkeypatch.setattr(ctt, "train_composite_network_torch", _espia)
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            _escribir_toy(d)
            spec = parse_training_text(_toy_mxtrain(optimizador))
            TransformerSupervisedTrainer().train(spec, output_dir=str(d / "out"), base_path=d)
        return visto

    def test_pasa_weight_decay_y_schedule_declarados(self, monkeypatch):
        visto = self._kwargs_que_llegan(
            "  TYPE adamw\n  LEARNING_RATE 0.01\n  WEIGHT_DECAY 0.001\n  SCHEDULE cosine\n",
            monkeypatch)
        assert visto["optimizer"] == "adamw"
        assert visto["weight_decay"] == 0.001
        assert visto["schedule"] == "cosine"

    def test_sin_declararlos_llegan_los_de_siempre(self, monkeypatch):
        visto = self._kwargs_que_llegan("  TYPE adam\n  LEARNING_RATE 0.01\n", monkeypatch)
        assert visto.get("weight_decay", 0.0) == 0.0
        assert visto.get("schedule") is None
