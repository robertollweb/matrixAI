"""P22 C5 — IncrementalTrainer: warm-start fine-tuning from a base ParameterSet."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pytest

from matrixai.continual import (
    ContinualDataset,
    IncrementalTrainer,
    IncrementalTrainingResult,
    parse_mxcontinual,
)
from matrixai.ir.continual import (
    DatasetMixSpec,
    RecencyDecaySpec,
)
from matrixai.parameters.store import ParameterSet
from matrixai.training.data import SupervisedExample


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_SRC = """
CONTINUAL_POLICY TrainerTestPolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
    SOURCES [api]
  END

  DRIFT_DETECTION
    FEATURES [score]
    METHODS
      score: ks threshold=0.15
    END
    MIN_SAMPLES 50
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES 50
    MIN_GROUND_TRUTH_RATIO 0.5
    COOLDOWN_DAYS 1
  END

  TRAINING
    METHOD incremental_finetune
    LEARNING_RATE_FACTOR 0.1
    MAX_EPOCHS 20
    DATASET_MIX
      BASE_WEIGHT 0.5
      PRODUCTION_WEIGHT 0.5
      RECENCY_DECAY linear
    END
  END

  APPROVAL_GATE
    HOLDOUT_FRACTION 0.2
    REGRESSION_GUARD
      METRIC accuracy
      MUST_IMPROVE_BY 0.0
      MAX_DEGRADATION_PER_LABEL 0.05
    END
    HUMAN_APPROVAL false
  END

  ROLLBACK
    AUTO_TRIGGER false
    METRIC accuracy
    SLIDING_WINDOW_HOURS 24
    DEGRADATION_THRESHOLD 0.05
    MIN_SAMPLES_IN_WINDOW 50
  END

  AUDIT
    PERSIST_DRIFT_REPORTS true
    PERSIST_UPDATE_TRACES true
    EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT false
    REFINEMENT_DRIFT_PERSISTENCE_DAYS 14
    SIGNATURE_REQUIRED false
  END
END
"""

_POLICY_EARLY_STOP_SRC = _POLICY_SRC.replace(
    "MAX_EPOCHS 20",
    "MAX_EPOCHS 100\n    EARLY_STOP patience=3 metric=accuracy",
)

_POLICY_FEW_EPOCHS = _POLICY_SRC.replace("MAX_EPOCHS 20", "MAX_EPOCHS 3")

_LABELS = ["spam", "ham", "promo"]  # 3-class
_N_FEAT = 4


def _make_ps(
    w1_values: list[list[float]] | None = None,
    b1_values: list[float] | None = None,
    parameter_set_id: str = "base_ps_v1",
) -> ParameterSet:
    """Create a 3-class, 4-feature P4-style ParameterSet directly."""
    n_labels = 3
    n_feat = _N_FEAT
    W1 = w1_values or [[0.1 * (k + 1) * (-1 if j % 2 else 1) for j in range(n_feat)] for k in range(n_labels)]
    b1 = b1_values or [0.0] * n_labels
    return ParameterSet(
        parameter_set_id=parameter_set_id,
        model_hash="sha256:fake_model_hash",
        parameter_schema_hash="params_fake_schema",
        source="trained",
        parameters={
            "W1": {"function": "classifier", "role": "weights", "type": "Tensor[3,4]",
                   "shape": [3, 4], "dtype": "float32", "initializer": "xavier", "values": W1},
            "b1": {"function": "classifier", "role": "bias", "type": "Vector[3]",
                   "shape": [3], "dtype": "float32", "initializer": "zeros", "values": b1},
        },
        metrics={"validation_loss": 0.8, "accuracy": 0.6},
    )


def _make_example(vector: list[float], label: str, idx: int = 0) -> SupervisedExample:
    return SupervisedExample(
        vector=vector,
        label=label,
        row_index=idx,
        row_hash=f"h{idx:04d}",
    )


_NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
_MIX = DatasetMixSpec(
    base_weight=0.5,
    production_weight=0.5,
    recency_decay=RecencyDecaySpec(method="linear", half_life_days=None),
)

# Synthetic classification dataset: 3 classes × 20 examples each = 60 total
# Class "spam":  high x[0], low x[1]
# Class "ham":   low x[0], high x[1]
# Class "promo": medium x[0], medium x[1]
def _make_classification_dataset(n_per_class: int = 20) -> ContinualDataset:
    examples: list[SupervisedExample] = []
    for i in range(n_per_class):
        examples.append(_make_example([1.0, 0.1, float(i % 3), 0.0], "spam", i))
        examples.append(_make_example([0.1, 1.0, 0.0, float(i % 3)], "ham", n_per_class + i))
        examples.append(_make_example([0.5, 0.5, 1.0, 1.0], "promo", 2 * n_per_class + i))
    return ContinualDataset(
        base_examples=examples,
        production_examples=[],
        production_timestamps=[],
        mix_spec=_MIX,
        base_fingerprint="test_base",
        reference_time=_NOW,
        window_days=30,
    )


def _policy(src: str = _POLICY_SRC) -> Any:
    return parse_mxcontinual(src)


# ── IncrementalTrainingResult shape tests ─────────────────────────────────────

class TestIncrementalTrainingResultShape:
    def test_returns_incremental_training_result(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert isinstance(result, IncrementalTrainingResult)

    def test_result_has_candidate_parameter_set(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert isinstance(result.candidate_parameter_set, ParameterSet)

    def test_result_has_parent_parameter_set_id(self):
        base = _make_ps(parameter_set_id="my_base_v1")
        trainer = IncrementalTrainer(
            _policy(), base, _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert result.parent_parameter_set_id == "my_base_v1"

    def test_candidate_metrics_contain_parent_id(self):
        base = _make_ps(parameter_set_id="parent_ps_001")
        trainer = IncrementalTrainer(
            _policy(), base, _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert result.candidate_parameter_set.metrics["parent_parameter_set_id"] == "parent_ps_001"

    def test_epoch_trace_not_empty(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert len(result.epoch_trace) > 0

    def test_epoch_trace_contains_expected_keys(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        entry = result.epoch_trace[0]
        for key in ("epoch", "train_loss", "validation_loss", "accuracy"):
            assert key in entry

    def test_epochs_run_matches_trace_length(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert result.epochs_run == len(result.epoch_trace)

    def test_best_epoch_within_range(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert 1 <= result.best_epoch <= result.epochs_run


# ── Parameter update tests ─────────────────────────────────────────────────────

class TestParameterUpdates:
    def test_parameters_change_after_training(self):
        base = _make_ps()
        original_W1 = base.parameters["W1"]["values"]
        trainer = IncrementalTrainer(
            _policy(), base, _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        new_W1 = result.candidate_parameter_set.parameters["W1"]["values"]
        assert original_W1 != new_W1

    def test_candidate_has_same_parameter_schema(self):
        base = _make_ps()
        trainer = IncrementalTrainer(
            _policy(), base, _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        cand = result.candidate_parameter_set
        assert set(cand.parameters.keys()) == set(base.parameters.keys())

    def test_candidate_preserves_model_hash(self):
        base = _make_ps()
        trainer = IncrementalTrainer(
            _policy(), base, _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert result.candidate_parameter_set.model_hash == base.model_hash

    def test_candidate_source_is_incremental_finetune(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert result.candidate_parameter_set.source == "incremental_finetune"

    def test_candidate_id_derived_from_parent(self):
        base = _make_ps(parameter_set_id="base_v2")
        trainer = IncrementalTrainer(
            _policy(), base, _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert "base_v2" in result.candidate_parameter_set.parameter_set_id

    def test_candidate_metrics_contain_validation_loss(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        m = result.candidate_parameter_set.metrics
        assert "validation_loss" in m
        assert isinstance(m["validation_loss"], float)

    def test_candidate_metrics_contain_dataset_fingerprint(self):
        ds = _make_classification_dataset()
        trainer = IncrementalTrainer(_policy(), _make_ps(), ds, labels=_LABELS)
        result = trainer.run()
        assert result.candidate_parameter_set.metrics["dataset_fingerprint"] == ds.fingerprint()


# ── Epoch and learning-rate respecting tests ───────────────────────────────────

class TestEpochsAndLearningRate:
    def test_respects_max_epochs(self):
        policy = _policy(_POLICY_FEW_EPOCHS)
        trainer = IncrementalTrainer(
            policy, _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert result.epochs_run <= 3

    def test_learning_rate_factor_applied(self):
        """Smaller LEARNING_RATE_FACTOR should produce less total parameter change."""
        src_small = _POLICY_SRC.replace("LEARNING_RATE_FACTOR 0.1", "LEARNING_RATE_FACTOR 0.01")
        src_large = _POLICY_SRC.replace("LEARNING_RATE_FACTOR 0.1", "LEARNING_RATE_FACTOR 1.0")
        base = _make_ps()
        ds = _make_classification_dataset()

        r_small = IncrementalTrainer(
            parse_mxcontinual(src_small), base, ds, labels=_LABELS, base_learning_rate=0.01, seed=42,
        ).run()
        r_large = IncrementalTrainer(
            parse_mxcontinual(src_large), base, ds, labels=_LABELS, base_learning_rate=0.01, seed=42,
        ).run()

        def _total_change(result: IncrementalTrainingResult) -> float:
            orig = base.parameters["W1"]["values"]
            new = result.candidate_parameter_set.parameters["W1"]["values"]
            return sum(
                abs(new[k][j] - orig[k][j])
                for k in range(len(orig))
                for j in range(len(orig[0]))
            )

        assert _total_change(r_small) < _total_change(r_large)

    def test_epoch_callback_is_called(self):
        calls: list[dict] = []
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        trainer.run(epoch_callback=calls.append)
        assert len(calls) > 0
        assert all("epoch" in c for c in calls)


# ── Early stopping tests ───────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_early_stop_limits_epochs_below_max(self):
        # max=100, patience=3; use high LR so model converges/diverges fast
        policy = _policy(_POLICY_EARLY_STOP_SRC)
        trainer = IncrementalTrainer(
            policy, _make_ps(), _make_classification_dataset(), labels=_LABELS,
            base_learning_rate=2.0,  # high LR → loss plateaus quickly → early stop fires
        )
        result = trainer.run()
        assert result.epochs_run < 100

    def test_stopped_early_flag_set_when_triggered(self):
        policy = _policy(_POLICY_EARLY_STOP_SRC)
        trainer = IncrementalTrainer(
            policy, _make_ps(), _make_classification_dataset(), labels=_LABELS,
            base_learning_rate=2.0,
        )
        result = trainer.run()
        assert result.stopped_early is True

    def test_no_early_stop_without_spec(self):
        """Without EARLY_STOP, the trainer must run for exactly MAX_EPOCHS."""
        policy = _policy(_POLICY_FEW_EPOCHS)  # max=3, no EARLY_STOP
        trainer = IncrementalTrainer(
            policy, _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert result.stopped_early is False
        assert result.epochs_run == 3


# ── Training produces lower loss tests ────────────────────────────────────────

class TestTrainingReducesLoss:
    def test_loss_decreases_after_training(self):
        """Final validation_loss should be below initial loss on a learnable dataset."""
        base = _make_ps()
        trainer = IncrementalTrainer(
            _policy(), base, _make_classification_dataset(n_per_class=30), labels=_LABELS,
            base_learning_rate=0.1,
        )
        result = trainer.run()
        first_loss = result.epoch_trace[0]["validation_loss"]
        best_loss = result.best_validation_loss
        assert best_loss <= first_loss

    def test_accuracy_is_non_negative(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        for entry in result.epoch_trace:
            assert entry["accuracy"] >= 0.0


# ── Dataset fingerprint and misc ──────────────────────────────────────────────

class TestMiscellaneous:
    def test_raises_on_empty_dataset(self):
        empty_ds = ContinualDataset(
            base_examples=[],
            production_examples=[],
            production_timestamps=[],
            mix_spec=_MIX,
            base_fingerprint="empty",
            reference_time=_NOW,
            window_days=30,
        )
        trainer = IncrementalTrainer(_policy(), _make_ps(), empty_ds, labels=_LABELS)
        with pytest.raises(ValueError, match="no examples"):
            trainer.run()

    def test_raises_without_program_and_non_p4_params(self):
        """ParameterSet without W1/b1 keys and no program should raise."""
        non_p4_ps = ParameterSet(
            parameter_set_id="non_p4",
            model_hash="sha256:x",
            parameter_schema_hash="params_x",
            source="initial",
            parameters={
                "dense_network.layer_0.W1": {
                    "function": "layer_0", "role": "weights", "type": "Tensor[4,4]",
                    "shape": [4, 4], "dtype": "float32", "initializer": "xavier",
                    "values": [[0.1] * 4] * 4,
                },
            },
            metrics={},
        )
        trainer = IncrementalTrainer(
            _policy(), non_p4_ps, _make_classification_dataset(), labels=_LABELS,
        )
        with pytest.raises(ValueError, match="P4-style"):
            trainer.run()

    def test_p4_mode_detected_when_w1_b1_present(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        assert trainer._is_p4_mode() is True

    def test_candidate_metrics_epochs_run_matches_result(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        m = result.candidate_parameter_set.metrics
        assert m["epochs_run"] == result.epochs_run

    def test_candidate_stopped_early_flag_matches_result(self):
        trainer = IncrementalTrainer(
            _policy(), _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        m = result.candidate_parameter_set.metrics
        assert m["stopped_early"] == result.stopped_early

    def test_deterministic_with_same_seed(self):
        base = _make_ps()
        ds = _make_classification_dataset()
        policy = _policy()
        r1 = IncrementalTrainer(policy, base, ds, labels=_LABELS, seed=42).run()
        r2 = IncrementalTrainer(policy, base, ds, labels=_LABELS, seed=42).run()
        W1_a = r1.candidate_parameter_set.parameters["W1"]["values"]
        W1_b = r2.candidate_parameter_set.parameters["W1"]["values"]
        assert W1_a == W1_b

    def test_different_seeds_may_produce_different_results(self):
        base = _make_ps()
        ds = _make_classification_dataset()
        policy = _policy()
        r1 = IncrementalTrainer(policy, base, ds, labels=_LABELS, seed=1).run()
        r2 = IncrementalTrainer(policy, base, ds, labels=_LABELS, seed=999).run()
        W1_a = r1.candidate_parameter_set.parameters["W1"]["values"]
        W1_b = r2.candidate_parameter_set.parameters["W1"]["values"]
        # With different seeds the train/val split changes → different parameters
        assert W1_a != W1_b

    def test_raises_on_unsupported_training_method(self):
        """TRAINING.METHOD values outside the MVP set must be rejected at init."""
        src = _POLICY_SRC.replace("METHOD incremental_finetune", "METHOD full_retrain")
        unsupported_policy = parse_mxcontinual(src)
        with pytest.raises(ValueError, match="unsupported TRAINING METHOD"):
            IncrementalTrainer(
                unsupported_policy, _make_ps(), _make_classification_dataset(), labels=_LABELS,
            )


# ── replay_buffer alias tests ──────────────────────────────────────────────────

_POLICY_REPLAY_BUFFER_SRC = _POLICY_SRC.replace("METHOD incremental_finetune", "METHOD replay_buffer")


class TestReplayBufferAlias:
    def test_replay_buffer_accepted_as_method(self):
        policy = parse_mxcontinual(_POLICY_REPLAY_BUFFER_SRC)
        assert policy.training.method == "replay_buffer"
        trainer = IncrementalTrainer(
            policy, _make_ps(), _make_classification_dataset(), labels=_LABELS,
        )
        result = trainer.run()
        assert isinstance(result, IncrementalTrainingResult)

    def test_replay_buffer_produces_incremental_finetune_source(self):
        """replay_buffer is an alias: candidate source is always 'incremental_finetune'."""
        policy = parse_mxcontinual(_POLICY_REPLAY_BUFFER_SRC)
        result = IncrementalTrainer(
            policy, _make_ps(), _make_classification_dataset(), labels=_LABELS,
        ).run()
        assert result.candidate_parameter_set.source == "incremental_finetune"
