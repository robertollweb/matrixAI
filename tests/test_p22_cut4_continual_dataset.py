"""P22 C4 — ContinualDataset: base+production mixing with recency decay."""
import math
from datetime import datetime, timedelta, timezone

import pytest

from matrixai.continual import ContinualDataset, parse_mxcontinual
from matrixai.training.data import SupervisedExample


# ── helpers ────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_POLICY_SRC = """
CONTINUAL_POLICY MixTestPolicy
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
    MAX_EPOCHS 10
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

_POLICY_EXP_SRC = _POLICY_SRC.replace("RECENCY_DECAY linear", "RECENCY_DECAY exponential half_life_days=7")


def _policy():
    return parse_mxcontinual(_POLICY_SRC)


def _policy_exp():
    return parse_mxcontinual(_POLICY_EXP_SRC)


def _ex(label: str, idx: int) -> SupervisedExample:
    return SupervisedExample(
        vector=[float(idx), float(idx % 3)],
        label=label,
        row_index=idx,
        row_hash=f"row_{idx:04d}",
    )


def _base(n: int = 5) -> list[SupervisedExample]:
    return [_ex("support", i) for i in range(n)]


def _prod(n: int = 3) -> list[SupervisedExample]:
    return [_ex("sales", 100 + i) for i in range(n)]


def _ts_days_ago(days: float) -> datetime:
    return _NOW - timedelta(days=days)


# ── construction ──────────────────────────────────────────────────────────────

class TestContinualDatasetConstruction:
    def test_constructs_with_valid_inputs(self):
        policy = _policy()
        ds = ContinualDataset(_base(), _prod(), [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        assert ds is not None

    def test_raises_on_timestamp_count_mismatch(self):
        policy = _policy()
        with pytest.raises(ValueError, match="same length"):
            ContinualDataset(_base(), _prod(3), [_ts_days_ago(1)] * 2,
                             policy.training.dataset_mix, "fp_base",
                             reference_time=_NOW)

    def test_base_count_and_production_count(self):
        policy = _policy()
        ds = ContinualDataset(_base(4), _prod(3), [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        assert ds.base_count() == 4
        assert ds.production_count() == 3


# ── examples() ────────────────────────────────────────────────────────────────

class TestExamples:
    def test_base_examples_always_included(self):
        policy = _policy()
        base = _base(5)
        ds = ContinualDataset(base, [], [], policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        result = ds.examples()
        assert set(e.row_hash for e in result) == set(e.row_hash for e in base)

    def test_all_examples_are_supervised_examples(self):
        policy = _policy()
        ds = ContinualDataset(_base(4), _prod(3), [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        for ex in ds.examples():
            assert isinstance(ex, SupervisedExample)

    def test_production_examples_with_nonzero_weight_included(self):
        policy = _policy()
        prod = _prod(3)
        ds = ContinualDataset(_base(5), prod, [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        result = ds.examples()
        prod_hashes = {e.row_hash for e in prod}
        result_hashes = {e.row_hash for e in result}
        assert prod_hashes.issubset(result_hashes)

    def test_linear_decay_excludes_examples_beyond_window(self):
        policy = _policy()
        prod = _prod(2)
        # One recent (1 day ago), one beyond window (40 days ago with window_days=30)
        ts = [_ts_days_ago(1), _ts_days_ago(40)]
        ds = ContinualDataset(_base(3), prod, ts,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        result = ds.examples()
        result_hashes = {e.row_hash for e in result}
        assert prod[0].row_hash in result_hashes
        assert prod[1].row_hash not in result_hashes

    def test_production_weight_zero_excludes_production(self):
        src = _POLICY_SRC.replace("BASE_WEIGHT 0.5\n      PRODUCTION_WEIGHT 0.5",
                                  "BASE_WEIGHT 1.0\n      PRODUCTION_WEIGHT 0.0")
        policy = parse_mxcontinual(src)
        prod = _prod(3)
        ds = ContinualDataset(_base(5), prod, [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        result = ds.examples()
        prod_hashes = {e.row_hash for e in prod}
        for ex in result:
            assert ex.row_hash not in prod_hashes

    def test_empty_production_returns_base_only(self):
        policy = _policy()
        base = _base(5)
        ds = ContinualDataset(base, [], [], policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        assert len(ds.examples()) == 5
        assert all(e in base for e in ds.examples())

    def test_total_count_with_recent_production(self):
        policy = _policy()
        ds = ContinualDataset(_base(5), _prod(3), [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        assert len(ds.examples()) == 8  # 5 base + 3 production (all recent)


# ── recency decay ─────────────────────────────────────────────────────────────

class TestRecencyDecay:
    def test_exponential_decay_formula(self):
        """w = exp(-ln(2) * age_days / half_life_days)"""
        policy = _policy_exp()  # HALF_LIFE_DAYS 7
        prod = _prod(1)
        # Age exactly = half_life → weight should be ≈ 0.5
        ts = [_ts_days_ago(7.0)]
        ds = ContinualDataset(_base(1), prod, ts,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        decay = ds._decay_weights()
        assert abs(decay[0] - 0.5) < 1e-6

    def test_exponential_zero_age_weight_is_one(self):
        policy = _policy_exp()
        ds = ContinualDataset(_base(1), _prod(1), [_NOW],
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        decay = ds._decay_weights()
        assert abs(decay[0] - 1.0) < 1e-6

    def test_exponential_weight_decreases_with_age(self):
        policy = _policy_exp()
        prod = _prod(3)
        ts = [_ts_days_ago(1), _ts_days_ago(7), _ts_days_ago(14)]
        ds = ContinualDataset(_base(1), prod, ts,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        decay = ds._decay_weights()
        assert decay[0] > decay[1] > decay[2]

    def test_linear_decay_formula(self):
        """w = max(0, 1 - age_days / window_days)"""
        policy = _policy()
        prod = _prod(1)
        ts = [_ts_days_ago(15.0)]
        ds = ContinualDataset(_base(1), prod, ts,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        decay = ds._decay_weights()
        expected = 1.0 - 15.0 / 30.0
        assert abs(decay[0] - expected) < 1e-6

    def test_linear_decay_zero_at_boundary(self):
        policy = _policy()
        prod = _prod(1)
        ts = [_ts_days_ago(30.0)]
        ds = ContinualDataset(_base(1), prod, ts,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        decay = ds._decay_weights()
        assert decay[0] == pytest.approx(0.0, abs=1e-6)

    def test_linear_decay_beyond_window_clamped_to_zero(self):
        policy = _policy()
        prod = _prod(1)
        ts = [_ts_days_ago(50.0)]
        ds = ContinualDataset(_base(1), prod, ts,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        decay = ds._decay_weights()
        assert decay[0] == 0.0


# ── weights() ─────────────────────────────────────────────────────────────────

class TestWeights:
    def test_weights_parallel_to_examples(self):
        policy = _policy()
        ds = ContinualDataset(_base(4), _prod(3), [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        examples = ds.examples()
        weights = ds.weights()
        assert len(weights) == len(examples)

    def test_all_weights_positive(self):
        policy = _policy()
        ds = ContinualDataset(_base(4), _prod(3), [_ts_days_ago(1)] * 3,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        assert all(w > 0.0 for w in ds.weights())

    def test_recent_production_has_higher_weight_than_old(self):
        policy = _policy()
        prod = _prod(2)
        ts = [_ts_days_ago(1), _ts_days_ago(20)]
        ds = ContinualDataset(_base(3), prod, ts,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW, window_days=30)
        weights = ds.weights()
        # production weights are the last 2 entries; recent should be > old
        prod_weights = weights[3:]  # base is 3 examples
        assert prod_weights[0] > prod_weights[1]


# ── fingerprint ───────────────────────────────────────────────────────────────

class TestFingerprint:
    def test_fingerprint_starts_with_continual(self):
        policy = _policy()
        ds = ContinualDataset(_base(3), _prod(2), [_ts_days_ago(1)] * 2,
                              policy.training.dataset_mix, "fp_base",
                              reference_time=_NOW)
        assert ds.fingerprint().startswith("continual_")

    def test_fingerprint_is_deterministic(self):
        policy = _policy()
        ds1 = ContinualDataset(_base(3), _prod(2), [_ts_days_ago(1)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW)
        ds2 = ContinualDataset(_base(3), _prod(2), [_ts_days_ago(1)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW)
        assert ds1.fingerprint() == ds2.fingerprint()

    def test_fingerprint_changes_with_different_base(self):
        policy = _policy()
        ds1 = ContinualDataset(_base(3), [], [], policy.training.dataset_mix,
                               "fp_base_A", reference_time=_NOW)
        ds2 = ContinualDataset(_base(3), [], [], policy.training.dataset_mix,
                               "fp_base_B", reference_time=_NOW)
        assert ds1.fingerprint() != ds2.fingerprint()

    def test_fingerprint_changes_when_production_added(self):
        policy = _policy()
        ds1 = ContinualDataset(_base(3), [], [], policy.training.dataset_mix,
                               "fp_base", reference_time=_NOW)
        ds2 = ContinualDataset(_base(3), _prod(1), [_ts_days_ago(1)],
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW)
        assert ds1.fingerprint() != ds2.fingerprint()

    def test_fingerprint_changes_with_different_window_days(self):
        """Different window_days changes the effective decay → different fingerprint."""
        policy = _policy()
        ds1 = ContinualDataset(_base(2), _prod(2), [_ts_days_ago(1)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW, window_days=30)
        ds2 = ContinualDataset(_base(2), _prod(2), [_ts_days_ago(1)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW, window_days=7)
        assert ds1.fingerprint() != ds2.fingerprint()

    def test_fingerprint_changes_with_different_reference_date(self):
        """Different reference date changes effective weights → different fingerprint."""
        policy = _policy()
        ds1 = ContinualDataset(_base(2), _prod(2), [_ts_days_ago(1)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW, window_days=30)
        other_now = _NOW - timedelta(days=5)
        ds2 = ContinualDataset(_base(2), _prod(2), [_ts_days_ago(1)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=other_now, window_days=30)
        assert ds1.fingerprint() != ds2.fingerprint()

    def test_fingerprint_changes_with_different_production_timestamps(self):
        """Same examples but different timestamps → different decay → different fingerprint."""
        policy = _policy()
        ds1 = ContinualDataset(_base(2), _prod(2), [_ts_days_ago(1)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW, window_days=30)
        ds2 = ContinualDataset(_base(2), _prod(2), [_ts_days_ago(10)] * 2,
                               policy.training.dataset_mix, "fp_base",
                               reference_time=_NOW, window_days=30)
        assert ds1.fingerprint() != ds2.fingerprint()
