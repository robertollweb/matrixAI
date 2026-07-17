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

    def test_invalid_mode_value_rejected(self):
        with pytest.raises(MatrixAITrainingParseError):
            parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0.7 validation=0.3 mode=bogus"))

    def test_no_split_line_at_all(self):
        spec = parse_training_text(_mxtrain_text("d.csv", ""))
        assert spec.dataset.split is None

    def test_split_spec_to_dict_omits_mode_when_random_byte_identical(self):
        """Auditoría [MEDIA]: `to_dict()` NO debe añadir "mode" cuando es
        "random" (default) — antes lo hacía siempre, así que la
        serialización de un split declarado ANTES de C3 dejaba de ser
        byte-idéntica aunque `mode` nunca se usara."""
        spec = DatasetSplitSpec(train=0.8, validation=0.2)
        assert "mode" not in spec.to_dict()
        spec2 = DatasetSplitSpec(train=0.8, validation=0.2, mode="temporal")
        assert spec2.to_dict()["mode"] == "temporal"

    # -- Auditoría 2026-07-17 [MEDIA]: SPLIT acepta ratios incoherentes --

    def test_seed_with_mode_temporal_rejected(self):
        """mode=temporal nunca baraja — un seed ahí es casi siempre una
        confusión del usuario, se rechaza en vez de aceptarlo e ignorarlo."""
        with pytest.raises(MatrixAITrainingParseError, match="seed"):
            parse_training_text(
                _mxtrain_text("d.csv", "SPLIT train=0.7 validation=0.3 seed=42 mode=temporal")
            )

    def test_train_plus_validation_over_one_rejected(self):
        with pytest.raises(MatrixAITrainingParseError, match="sumar 1"):
            parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0.9 validation=0.9"))

    def test_train_zero_rejected(self):
        with pytest.raises(MatrixAITrainingParseError, match="entre 0 y 1"):
            parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0 validation=1"))

    def test_train_one_rejected(self):
        with pytest.raises(MatrixAITrainingParseError, match="entre 0 y 1"):
            parse_training_text(_mxtrain_text("d.csv", "SPLIT train=1 validation=0"))

    def test_ratios_that_sum_to_one_still_parse(self):
        spec = parse_training_text(_mxtrain_text("d.csv", "SPLIT train=0.85 validation=0.15"))
        assert spec.dataset.split.train == 0.85


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
        """Confirma el código real (no una réplica): con mode=temporal,
        random.Random.shuffle NUNCA se invoca (mode=temporal no admite
        seed — invariante "sin barajar", ver TestParserSplitMode)."""
        from matrixai.training.transformer_trainer import TransformerSupervisedTrainer
        from matrixai.training.parser import parse_training_file

        calls = []
        real_shuffle = random.Random.shuffle

        def _spy_shuffle(self, x, *a, **kw):
            calls.append(True)
            return real_shuffle(self, x, *a, **kw)

        monkeypatch.setattr(random.Random, "shuffle", _spy_shuffle)

        mxtrain = _write_transformer_fixture(
            tmp_path, "SPLIT train=0.7 validation=0.3 mode=temporal",
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


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 [ALTA]: el camino torch REAL del Studio
# (`_run_playground_dense_training`/`_run_playground_training` con
# MATRIXAI_TRAIN_BACKEND=torch) ignoraba `training.dataset.split` por
# completo — SPLIT mode=temporal se declaraba y el trainer torch hacía su
# propio 80/20 interno sin mirarlo. Reproducido y corregido en
# dense_torch_trainer.py (validation_examples explícito) + playground.py
# (partición calculada honrando split.mode antes de llamar al trainer).
# ---------------------------------------------------------------------------

_DENSE_TORCH_MXAI = """PROJECT DT
VECTOR In[2]
  x1: Scalar
  x2: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END
GRAPH
  In -> Net
END
"""


def _dense_torch_train_text(split_line: str) -> str:
    return f"""MODEL DT.mxai
DATASET DS
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [x1, x2]
  TARGET y: Scalar
  {split_line}
END
LOSS L
  TYPE mse
  PREDICTION Net
  TARGET y
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END
"""


def _dense_torch_csv() -> str:
    # x1 = índice de fila (0..9) — identifica EXACTAMENTE qué filas
    # originales terminan en validación mirando los x1 vistos.
    rows = ["x1,x2,y"] + [f"{i},0,{i}" for i in range(10)]
    return "\n".join(rows) + "\n"


class TestDenseTorchStudioPathHonorsTemporalSplit:
    def _run_and_capture_val_x1(self, split_line, monkeypatch):
        import matrixai.training.dense_torch_trainer as dtt
        from matrixai.playground import _run_playground_dense_training

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
        captured: dict = {}
        real_train = dtt.train_dense_network_torch

        def _spy(*args, **kwargs):
            captured["validation_examples"] = kwargs.get("validation_examples")
            return real_train(*args, **kwargs)

        monkeypatch.setattr(dtt, "train_dense_network_torch", _spy)

        r = _run_playground_dense_training(
            _DENSE_TORCH_MXAI, _dense_torch_train_text(split_line), _dense_torch_csv(),
            epochs_override=2,
        )
        assert r["ok"], r.get("error")
        return captured["validation_examples"]

    def test_temporal_passes_explicit_last_tramo(self, monkeypatch):
        val = self._run_and_capture_val_x1(
            "SPLIT train=0.6 validation=0.4 mode=temporal", monkeypatch,
        )
        assert val is not None
        val_x1 = sorted(int(x[0]) for x, _y in val)
        assert val_x1 == [6, 7, 8, 9]

    def test_temporal_different_ratio_different_boundary(self, monkeypatch):
        val = self._run_and_capture_val_x1(
            "SPLIT train=0.9 validation=0.1 mode=temporal", monkeypatch,
        )
        val_x1 = sorted(int(x[0]) for x, _y in val)
        assert val_x1 == [9]

    def test_mode_random_matches_old_hardcoded_80_20_ignoring_custom_ratio(self, monkeypatch):
        """mode ausente/"random": la partición explícita que ahora se pasa
        reproduce EXACTAMENTE el 80/20 fijo que el trainer torch ya hacía
        internamente (byte-idéntico a antes de C3) — el ratio 0.6/0.4
        declarado se IGNORA, igual que antes de C3."""
        val = self._run_and_capture_val_x1(
            "SPLIT train=0.6 validation=0.4", monkeypatch,
        )
        assert val is not None
        val_x1 = sorted(int(x[0]) for x, _y in val)
        assert val_x1 == [8, 9]  # 0.8 fijo, NO 0.6

    # -- Reauditoría 2026-07-17 [ALTA]: la EVALUACIÓN dense-torch usaba
    # `examples` completo (train+val), no `val_ex` — métricas parcialmente
    # in-sample. Espía sobre evaluate_dense_network_torch, no solo el
    # trainer (petición explícita de la auditoría). --

    def _run_and_capture_eval_x1(self, split_line, monkeypatch):
        import matrixai.training.dense_torch_trainer as dtt
        from matrixai.playground import _run_playground_dense_training

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
        captured: dict = {}
        real_eval = dtt.evaluate_dense_network_torch

        def _spy_eval(net, ps, examples, *args, **kwargs):
            captured["eval_examples"] = examples
            return real_eval(net, ps, examples, *args, **kwargs)

        monkeypatch.setattr(dtt, "evaluate_dense_network_torch", _spy_eval)

        r = _run_playground_dense_training(
            _DENSE_TORCH_MXAI, _dense_torch_train_text(split_line), _dense_torch_csv(),
            epochs_override=2,
        )
        assert r["ok"], r.get("error")
        return captured["eval_examples"]

    def test_temporal_evaluates_only_on_held_out_validation_tramo(self, monkeypatch):
        eval_x1 = sorted(int(x[0]) for x, _y in self._run_and_capture_eval_x1(
            "SPLIT train=0.6 validation=0.4 mode=temporal", monkeypatch,
        ))
        assert eval_x1 == [6, 7, 8, 9]  # NUNCA las filas de entrenamiento [0..5]

    def test_mode_random_also_evaluates_only_on_validation_not_full_dataset(self, monkeypatch):
        """Incluso en mode=random (80/20 fijo, byte-idéntico para el
        ENTRENAMIENTO) la evaluación debe restringirse a las filas de
        validación — nunca al dataset completo (train+val), o las
        métricas reportadas serían parcialmente in-sample."""
        eval_x1 = sorted(int(x[0]) for x, _y in self._run_and_capture_eval_x1(
            "SPLIT train=0.6 validation=0.4", monkeypatch,
        ))
        assert eval_x1 == [8, 9]
        assert len(eval_x1) != 10  # nunca el dataset completo


_COMPOSITE_TORCH_MXAI = """PROJECT CT
VECTOR In[2]
  x1: Scalar
  x2: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  BLOCK r1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END
GRAPH
  In -> Net
END
"""


def _composite_torch_train_text(split_line: str) -> str:
    return f"""MODEL CT.mxai
DATASET DS
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [x1, x2]
  TARGET y: Scalar
  {split_line}
END
LOSS L
  TYPE mse
  PREDICTION Net
  TARGET y
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END
"""


class TestCompositeTorchStudioPathHonorsTemporalSplit:
    def _run_and_capture(self, split_line, monkeypatch):
        import matrixai.training.composite_torch_trainer as ctt
        from matrixai.playground import _run_playground_composite_training

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
        captured: dict = {}
        real_train = ctt.train_composite_network_torch

        def _spy(net, ps, examples, loss_fn, **kwargs):
            captured["examples"] = examples
            captured["validation_examples"] = kwargs.get("validation_examples")
            return real_train(net, ps, examples, loss_fn, **kwargs)

        monkeypatch.setattr(ctt, "train_composite_network_torch", _spy)

        r = _run_playground_composite_training(
            _COMPOSITE_TORCH_MXAI, _composite_torch_train_text(split_line), _dense_torch_csv(),
            epochs_override=1,
        )
        assert r["ok"], r.get("error")
        return captured

    def test_temporal_passes_explicit_split(self, monkeypatch):
        captured = self._run_and_capture(
            "SPLIT train=0.6 validation=0.4 mode=temporal", monkeypatch,
        )
        assert captured["validation_examples"] is not None
        val_x1 = sorted(int(x["x1"]) for x, _y in captured["validation_examples"])
        assert val_x1 == [6, 7, 8, 9]
        train_x1 = sorted(int(x["x1"]) for x, _y in captured["examples"])
        assert train_x1 == [0, 1, 2, 3, 4, 5]

    def test_mode_random_preserves_old_full_examples_no_explicit_partition(self, monkeypatch):
        """mode ausente/"random": el trainer torch recibe TODOS los
        examples (no train_ex) y validation_examples=None — el ratio 0.6
        declarado se ignora, byte-idéntico al comportamiento torch de
        antes de C3 (el bug real: train_ex/val_ex ya se calculaban pero
        se tiraban)."""
        captured = self._run_and_capture(
            "SPLIT train=0.6 validation=0.4", monkeypatch,
        )
        assert captured["validation_examples"] is None
        assert len(captured["examples"]) == 10  # TODOS, no solo el 60%
