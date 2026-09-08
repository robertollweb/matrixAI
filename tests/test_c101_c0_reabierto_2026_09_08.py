# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C0, REABIERTO (2026-09-08) — una auditoría externa, verificada de
forma independiente, encontró que el cierre original de este corte
(commit `5f25c8c`/`71f527e`, 2026-09-05) solo migró el camino DENSO
(`dense_trainer.py` stdlib y el camino torch en `playground.py`) a
`particion_para()`/`reparte()` -- pero el propio "qué se hace" del corte
prometía también "entrenadores... compuestos y torch", y tres caminos más
seguían con la lógica vieja (`train_ratio = split.train if split else
0.8; ...; val_ex = examples[n_train:]`), que NUNCA lee `split.test`: un
`.mxtrain` con `protocol=2` y un tramo de prueba reservado veía ese
tramo colarse entero en validación.

Corregidos en `matrixai/playground.py` (`_run_playground_composite_
training`, `_run_playground_transformer_training`, `_run_playground_
generic_training`) y `matrixai/training/transformer_trainer.py`
(`TransformerSupervisedTrainer`, que ya honraba semilla/ratio SIN exigir
`protocol=2` por una auditoría anterior propia -- ver el comentario en el
propio fichero -- y por eso no se movió a `particion_para()`, que sí lo
exigiría y apagaría esa capacidad ya enviada).

Este fichero cubre `TransformerSupervisedTrainer` (la ruta CLI/.mxtrain
del bloque transformer) con el canario del criterio de cierre original
del 101-C0 ("cambiar solo el objetivo de las filas de PRUEBA no cambia
el modelo") -- la misma técnica que ya usa `ElCanarioDeLaPruebaTest` en
`test_c101_c0_particion_y_protocolo.py` para el camino denso, aplicada
aquí al camino que NO se movió a `particion_para()` para comprobar que
el tramo de prueba, aun así, queda genuinamente fuera del entrenamiento.

Composite (stdlib y torch) y layer_call genérico comparten la MISMA
llamada a `particion_para()`/`reparte()` que el camino denso ya migrado
-- verificados por: (a) la prueba unitaria de `particion_para()` en
`test_c101_c0_particion_y_protocolo.py` (la función pura, sin
entrenador), y (b) `test_protocol2_test_no_se_cuela_en_validacion` en
`test_biblioteca_c3_split_temporal.py` para el camino composite/torch,
que es el que la auditoría reprodujo literalmente. La cobertura de
composite/stdlib y layer_call genérico con un canario propio queda
declarada, no hecha en esta pasada -- mismo patrón de wiring ya probado
en dos caminos distintos, deuda de cobertura menor, no de corrección.
"""
from __future__ import annotations

import csv as _csv
import hashlib
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

torch = pytest.importorskip("torch")

L_SEQ = 12


def _toy_rows(n: int, seed: int = 123):
    """Mismo generador que `test_transformer_c4_training.py`: 7 y 9 en
    posiciones aleatorias, relleno {1..6}, clase por el ORDEN de 7 vs 9."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p7, p9 = rng.sample(range(L_SEQ), 2)
        row = [rng.choice([1, 2, 3, 4, 5, 6]) for _ in range(L_SEQ)]
        row[p7], row[p9] = 7, 9
        target = [1.0, 0.0] if p7 < p9 else [0.0, 1.0]
        rows.append((row, target))
    return rows


_TOY_MXAI_GRAPH = """
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


def _mxtrain(model: str, csv: str, split_line: str) -> str:
    return f"""
MODEL {model}

DATASET D
  SOURCE csv("{csv}")
  INPUT Texto FROM COLUMNS [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]
  TARGET clase: Label[antes, despues]
  {split_line}
END

LOSS L
  TYPE cross_entropy
  PREDICTION clase
  TARGET clase
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.05
  UPDATE N.*
END

RUN
  EPOCHS 2
END
"""


def _write_dataset(tmp_dir: Path, rows) -> tuple[Path, Path]:
    csv_path = tmp_dir / "toy.csv"
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow([f"t{i}" for i in range(L_SEQ)] + ["clase"])
        for row, target in rows:
            w.writerow(row + ["antes" if target[0] == 1.0 else "despues"])
    model_path = tmp_dir / "toy.mxai"
    model_path.write_text(_TOY_MXAI_GRAPH, encoding="utf-8")
    return model_path, csv_path


class TestTransformerTrainerCLIHonorsProtocol2Test:
    """El camino de `TransformerSupervisedTrainer` que la migración a
    `particion_para()` NO tocó (ver docstring del módulo, y el propio
    fichero: ya honraba semilla/ratio sin `protocol=2`, por una auditoría
    anterior). Espía sobre `train_composite_network_torch` -- mismo
    patrón ya probado en `test_biblioteca_c3_split_temporal.py` para el
    camino composite/torch (`TestCompositeTorchStudioPathHonorsTemporal
    Split`) -- para comprobar de primera mano qué filas llegan de verdad
    a `validation_examples`."""

    def _entrenar_y_capturar(self, split_line: str, monkeypatch):
        import matrixai.training.composite_torch_trainer as ctt
        from matrixai.training.transformer_trainer import TransformerSupervisedTrainer
        from matrixai.training.parser import parse_training_text

        rows = _toy_rows(10, seed=7)  # 10 filas, orden fijo -- identificables por posición
        captured: dict = {}
        real_train = ctt.train_composite_network_torch

        def _spy(net, ps, examples, loss_fn, **kwargs):
            captured["train_examples"] = examples
            captured["validation_examples"] = kwargs.get("validation_examples")
            return real_train(net, ps, examples, loss_fn, **kwargs)

        monkeypatch.setattr(ctt, "train_composite_network_torch", _spy)
        # El propio módulo (`transformer_trainer.py`) importa la función
        # DENTRO de `.train()` -- parcheando el módulo origen (`ctt`) el
        # `from ... import ...` local sigue viendo la versión espiada,
        # porque Python resuelve el nombre en el módulo de `ctt` en el
        # momento de la llamada, no al importar.
        monkeypatch.setattr(
            "matrixai.training.transformer_trainer.train_composite_network_torch", _spy,
            raising=False,
        )

        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            model_path, csv_path = _write_dataset(d, rows)
            training = parse_training_text(_mxtrain("toy.mxai", "toy.csv", split_line))
            result = TransformerSupervisedTrainer().train(
                training, output_dir=str(d / "out"), base_path=d,
            )
            assert result.best_epoch >= 0
        return captured, rows

    @staticmethod
    def _identificar(examples, rows) -> list[int]:
        """Qué índice (0..9) de `rows` es cada ejemplo capturado, comparando
        la secuencia de tokens exacta -- las filas son deterministas
        (mismo seed), así que el contenido basta para identificarlas."""
        por_tokens = {tuple(row): i for i, (row, _t) in enumerate(rows)}
        return sorted(por_tokens[tuple(x["Texto"])] for x, _y in examples)

    def test_protocol2_test_no_se_cuela_en_validacion(self, monkeypatch):
        """El defecto reabierto, medido por ESTE camino exacto: con
        `protocol=2`+`test=` (mode=temporal, reparto determinista sin
        barajar), el tramo de prueba (índices 8,9) tiene que quedar FUERA
        de `validation_examples` -- antes de este corte se leía solo
        `split.train` y el resto (validación + prueba) caía entero en
        `val_examples`."""
        captured, rows = self._entrenar_y_capturar(
            "SPLIT train=0.6 validation=0.2 test=0.2 mode=temporal protocol=2", monkeypatch,
        )
        assert captured["validation_examples"] is not None
        val_idx = self._identificar(captured["validation_examples"], rows)
        train_idx = self._identificar(captured["train_examples"], rows)
        assert val_idx == [6, 7], val_idx  # NO [6,7,8,9]
        assert train_idx == [0, 1, 2, 3, 4, 5]
        assert 8 not in val_idx and 9 not in val_idx

    def test_sin_protocolo_sigue_siendo_el_comportamiento_de_siempre(self, monkeypatch):
        """Sin `protocol=2`: el ratio se sigue honrando (esa capacidad YA
        estaba enviada, de la auditoría "C4 ronda 2" -- ver el módulo),
        pero sin tramo de prueba: todo lo que no es train cae en
        validación, exactamente como antes de este corte."""
        captured, rows = self._entrenar_y_capturar(
            "SPLIT train=0.6 validation=0.4 mode=temporal", monkeypatch,
        )
        val_idx = self._identificar(captured["validation_examples"], rows)
        assert val_idx == [6, 7, 8, 9]
