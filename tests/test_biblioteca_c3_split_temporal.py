# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C3 — `SPLIT train=X validation=Y
mode=temporal|random` (contrato 57): parser + `DatasetSplitSpec` + las 3
familias de trainer (stdlib/torch vía `SupervisedTrainer._split_examples`,
denso vía `DenseSupervisedTrainer`, transformer vía
`TransformerSupervisedTrainer`). mode ausente/"random" es el
comportamiento de SIEMPRE, byte-idéntico — mode=temporal nunca baraja
(aunque haya seed) y el último tramo, en el orden que llega, es SIEMPRE
la validación.
"""
from __future__ import annotations

import random
import types

import pytest

from matrixai.training.parser import parse_training_text, MatrixAITrainingParseError
from matrixai.training.spec import DatasetSplitSpec


def _mxtrain_text(source: str, split_line: str) -> str:
    return f"""MODEL model.mxai

DATASET D
  SOURCE csv("{source}")
  INPUT V FROM COLUMNS [x1, x2]
  TARGET y: Scalar
  {split_line}
END

LOSS L
  TYPE mse
  PREDICTION y
  TARGET y
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END

RUN
  EPOCHS 5
END
"""


# ---------------------------------------------------------------------------
# Parser + DatasetSplitSpec
# ---------------------------------------------------------------------------

class TestParserSplitMode:
    def test_mode_temporal_parses(self):
        spec = parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0.7 validation=0.3 mode=temporal"))
        assert spec.dataset.split.mode == "temporal"

    def test_mode_random_parses(self):
        spec = parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0.7 validation=0.3 mode=random"))
        assert spec.dataset.split.mode == "random"

    def test_mode_absent_defaults_to_random(self):
        spec = parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0.7 validation=0.3"))
        assert spec.dataset.split.mode == "random"

    def test_seed_and_mode_together(self):
        spec = parse_training_text(
            _mxtrain_text("d.csv", "SPLIT train=0.7 validation=0.3 seed=42 mode=temporal")
        )
        assert spec.dataset.split.seed == 42
        assert spec.dataset.split.mode == "temporal"

    def test_invalid_mode_value_rejected(self):
        with pytest.raises(MatrixAITrainingParseError):
            parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0.7 validation=0.3 mode=bogus"))

    def test_no_split_line_at_all(self):
        spec = parse_training_text(_mxtrain_text("d.csv", ""))
        assert spec.dataset.split is None

    def test_split_spec_to_dict_includes_mode(self):
        spec = DatasetSplitSpec(train=0.8, validation=0.2)
        assert spec.to_dict()["mode"] == "random"
        spec2 = DatasetSplitSpec(train=0.8, validation=0.2, mode="temporal")
        assert spec2.to_dict()["mode"] == "temporal"


# ---------------------------------------------------------------------------
# Familia 1: SupervisedTrainer/_split_examples (compartida con
# TorchSupervisedTrainer vía herencia)
# ---------------------------------------------------------------------------

def _examples(n: int):
    from matrixai.training.data import SupervisedExample
    return [SupervisedExample(vector=[float(i)], label="a", row_index=i, row_hash=f"h{i}") for i in range(n)]


def _training_with_split(split: DatasetSplitSpec | None):
    return types.SimpleNamespace(dataset=types.SimpleNamespace(split=split))


class TestSharedSplitExamples:
    def test_temporal_never_shuffles_even_with_seed(self):
        from matrixai.training.trainer import SupervisedTrainer
        examples = _examples(10)
        split = DatasetSplitSpec(train=0.7, validation=0.3, seed=42, mode="temporal")
        train, val = SupervisedTrainer()._split_examples(examples, _training_with_split(split))
        assert [e.row_index for e in train] == [0, 1, 2, 3, 4, 5, 6]
        assert [e.row_index for e in val] == [7, 8, 9]

    def test_temporal_validation_is_exact_final_tramo_different_ratio(self):
        from matrixai.training.trainer import SupervisedTrainer
        examples = _examples(20)
        split = DatasetSplitSpec(train=0.9, validation=0.1, seed=7, mode="temporal")
        train, val = SupervisedTrainer()._split_examples(examples, _training_with_split(split))
        assert [e.row_index for e in val] == [18, 19]
        assert [e.row_index for e in train] == list(range(18))

    def test_random_with_seed_shuffles_as_before(self):
        from matrixai.training.trainer import SupervisedTrainer
        examples = _examples(10)
        split = DatasetSplitSpec(train=0.7, validation=0.3, seed=42, mode="random")
        train, val = SupervisedTrainer()._split_examples(examples, _training_with_split(split))
        # Replica EXACTA del algoritmo pre-C3 (byte-idéntico): mismo shuffle,
        # mismo corte.
        indices = list(range(10))
        random.Random(42).shuffle(indices)
        train_count = max(1, min(9, int(10 * 0.7)))
        expected_train_idx = set(indices[:train_count])
        expected_train = [i for i in range(10) if i in expected_train_idx]
        expected_val = [i for i in range(10) if i not in expected_train_idx]
        assert [e.row_index for e in train] == expected_train
        assert [e.row_index for e in val] == expected_val
        assert [e.row_index for e in val] != [7, 8, 9]  # de verdad barajado, no el tramo final

    def test_random_without_seed_is_sequential_same_as_before(self):
        from matrixai.training.trainer import SupervisedTrainer
        examples = _examples(10)
        split = DatasetSplitSpec(train=0.7, validation=0.3, mode="random")
        train, val = SupervisedTrainer()._split_examples(examples, _training_with_split(split))
        assert [e.row_index for e in val] == [7, 8, 9]

    def test_no_split_declared_unaffected_by_c3(self):
        from matrixai.training.trainer import SupervisedTrainer
        examples = _examples(10)
        train, val = SupervisedTrainer()._split_examples(examples, _training_with_split(None))
        assert len(train) == 8 and len(val) == 2

    def test_torch_trainer_inherits_same_split_examples(self):
        from matrixai.training.torch_trainer import TorchSupervisedTrainer
        from matrixai.training.trainer import SupervisedTrainer
        assert TorchSupervisedTrainer._split_examples is SupervisedTrainer._split_examples


# ---------------------------------------------------------------------------
# Familia 2: DenseSupervisedTrainer (E2E real, stdlib, sin torch)
# ---------------------------------------------------------------------------

def _write_dense_fixture(tmp_path, split_line: str):
    mxai = tmp_path / "model.mxai"
    mxai.write_text("""
PROJECT Test

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK Net
  INPUT V
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> Net
END
""", encoding="utf-8")
    csv_path = tmp_path / "data.csv"
    # x1 = índice de fila (0..9) — permite identificar EXACTAMENTE qué filas
    # originales terminan en validación inspeccionando los x1 vistos.
    rows = ["x1,x2,y"] + [f"{i},0,{i}" for i in range(10)]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    mxtrain = tmp_path / "train.mxtrain"
    mxtrain.write_text(f"""MODEL model.mxai

DATASET TrainData
  SOURCE csv("data.csv")
  INPUT V FROM COLUMNS [x1, x2]
  TARGET y: Scalar
  {split_line}
END

LOSS NetLoss
  TYPE mse
  PREDICTION y
  TARGET y
END

OPTIMIZER NetOpt
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END

RUN
  EPOCHS 2
END
""", encoding="utf-8")
    return mxtrain


class TestDenseTrainerTemporalSplit:
    def _train_and_capture_val_indices(self, tmp_path, split_line, monkeypatch):
        import matrixai.training.dense_trainer as dt
        from matrixai.training.parser import parse_training_file

        captured: dict = {}
        real_evaluate = dt.evaluate_dense_network

        def _spy(net, ps, examples, loss_fn, labels=None):
            captured["val_x1"] = sorted(int(x[0]) for x, _y in examples)
            return real_evaluate(net, ps, examples, loss_fn, labels=labels)

        monkeypatch.setattr(dt, "evaluate_dense_network", _spy)

        mxtrain = _write_dense_fixture(tmp_path, split_line)
        training = parse_training_file(mxtrain)
        dt.DenseSupervisedTrainer().train(
            training, output_dir=str(tmp_path / "out"), base_path=tmp_path,
        )
        return captured["val_x1"]

    def test_temporal_uses_declared_ratio_last_tramo(self, tmp_path, monkeypatch):
        val_idx = self._train_and_capture_val_indices(
            tmp_path, "SPLIT train=0.6 validation=0.4 mode=temporal", monkeypatch,
        )
        assert val_idx == [6, 7, 8, 9]

    def test_temporal_different_ratio_different_boundary(self, tmp_path, monkeypatch):
        val_idx = self._train_and_capture_val_indices(
            tmp_path, "SPLIT train=0.9 validation=0.1 mode=temporal", monkeypatch,
        )
        assert val_idx == [9]

    def test_mode_absent_unchanged_hardcoded_80_20(self, tmp_path, monkeypatch):
        """mode ausente/"random": SIEMPRE 0.8 fijo, ignora el ratio
        declarado — comportamiento histórico, byte-idéntico a antes de C3."""
        val_idx = self._train_and_capture_val_indices(
            tmp_path, "SPLIT train=0.6 validation=0.4", monkeypatch,
        )
        assert val_idx == [8, 9]  # 0.8 fijo, NO 0.6

    def test_no_split_declared_unchanged(self, tmp_path, monkeypatch):
        val_idx = self._train_and_capture_val_indices(tmp_path, "", monkeypatch)
        assert val_idx == [8, 9]


# ---------------------------------------------------------------------------
# Familia 3: TransformerSupervisedTrainer (E2E real, requiere torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

_SEQ_LEN, _VOCAB = 6, 8

_TRANS_MXAI = """
PROJECT ToyC3
SEQUENCE Texto
  length = 6
  vocab_size = 8
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 8
  BLOCK enc TRANSFORMER
    LAYERS 1
    HEADS 2
    FF 16
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a, b]
END

GRAPH
  Texto -> N
END
"""


def _write_transformer_fixture(tmp_path, split_line: str):
    mxai = tmp_path / "toy.mxai"
    mxai.write_text(_TRANS_MXAI, encoding="utf-8")
    csv_path = tmp_path / "toy.csv"
    rows = ["t0,t1,t2,t3,t4,t5,clase"]
    for i in range(12):
        toks = [(i + j) % _VOCAB for j in range(_SEQ_LEN)]
        cls = "a" if i % 2 == 0 else "b"
        rows.append(",".join(str(t) for t in toks) + f",{cls}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    mxtrain = tmp_path / "toy.mxtrain"
    mxtrain.write_text(f"""MODEL toy.mxai

DATASET D
  SOURCE csv("toy.csv")
  INPUT Texto FROM COLUMNS [t0, t1, t2, t3, t4, t5]
  TARGET clase: Label[a, b]
  {split_line}
END

LOSS L
  TYPE cross_entropy
  PREDICTION clase
  TARGET clase
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE N.*
END

RUN
  EPOCHS 1
END
""", encoding="utf-8")
    return mxtrain


class TestTransformerTrainerTemporalSplit:
    def test_temporal_never_calls_shuffle(self, tmp_path, monkeypatch):
        """Confirma el código real (no una réplica): con mode=temporal y
        seed declarado, random.Random.shuffle NUNCA se invoca."""
        from matrixai.training.transformer_trainer import TransformerSupervisedTrainer
        from matrixai.training.parser import parse_training_file

        calls = []
        real_shuffle = random.Random.shuffle

        def _spy_shuffle(self, x, *a, **kw):
            calls.append(True)
            return real_shuffle(self, x, *a, **kw)

        monkeypatch.setattr(random.Random, "shuffle", _spy_shuffle)

        mxtrain = _write_transformer_fixture(
            tmp_path, "SPLIT train=0.7 validation=0.3 seed=42 mode=temporal",
        )
        training = parse_training_file(mxtrain)
        TransformerSupervisedTrainer().train(
            training, output_dir=str(tmp_path / "out"), base_path=tmp_path,
        )
        assert calls == []

    def test_random_with_seed_calls_shuffle(self, tmp_path, monkeypatch):
        from matrixai.training.transformer_trainer import TransformerSupervisedTrainer
        from matrixai.training.parser import parse_training_file

        calls = []
        real_shuffle = random.Random.shuffle

        def _spy_shuffle(self, x, *a, **kw):
            calls.append(True)
            return real_shuffle(self, x, *a, **kw)

        monkeypatch.setattr(random.Random, "shuffle", _spy_shuffle)

        mxtrain = _write_transformer_fixture(
            tmp_path, "SPLIT train=0.7 validation=0.3 seed=42 mode=random",
        )
        training = parse_training_file(mxtrain)
        TransformerSupervisedTrainer().train(
            training, output_dir=str(tmp_path / "out"), base_path=tmp_path,
        )
        assert len(calls) >= 1
