"""P22 C8 — ProductionMonitor: sliding-window online metrics and degradation detection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from matrixai.continual import (
    OnlineObservation,
    ProductionMonitor,
    WindowMetrics,
    parse_mxcontinual,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_SRC = """
CONTINUAL_POLICY MonitorTestPolicy
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
      MAX_DEGRADATION_PER_LABEL 0.1
    END
    HUMAN_APPROVAL false
  END

  ROLLBACK
    AUTO_TRIGGER true
    METRIC accuracy
    SLIDING_WINDOW_HOURS 24
    DEGRADATION_THRESHOLD 0.05
    MIN_SAMPLES_IN_WINDOW 10
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

_LABELS = ["spam", "ham", "promo"]
_NOW = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
_RECENT = _NOW - timedelta(hours=1)    # within 24h window
_OLD = _NOW - timedelta(hours=30)      # outside 24h window


def _policy(src: str = _POLICY_SRC) -> Any:
    return parse_mxcontinual(src)


def _monitor(ref: float | None = 0.91) -> ProductionMonitor:
    return ProductionMonitor(_policy(), reference_accuracy=ref, labels=_LABELS)


def _ts(delta_hours: float = -1.0) -> datetime:
    return _NOW + timedelta(hours=delta_hours)


# ── OnlineObservation tests ────────────────────────────────────────────────────

class TestOnlineObservation:
    def test_record_returns_observation(self):
        m = _monitor()
        obs = m.record("spam", "spam", trace_id="t001",
                       observed_at=_RECENT, parameter_set_id="ps-v1")
        assert isinstance(obs, OnlineObservation)

    def test_correct_flag_true_when_prediction_matches(self):
        m = _monitor()
        obs = m.record("spam", "spam", observed_at=_RECENT)
        assert obs.correct is True

    def test_correct_flag_false_when_prediction_wrong(self):
        m = _monitor()
        obs = m.record("ham", "spam", observed_at=_RECENT)
        assert obs.correct is False

    def test_observation_stores_trace_id(self):
        m = _monitor()
        obs = m.record("spam", "spam", trace_id="trace-xyz", observed_at=_RECENT)
        assert obs.trace_id == "trace-xyz"

    def test_observation_stores_parameter_set_id(self):
        m = _monitor()
        obs = m.record("spam", "spam", parameter_set_id="ps-v1.1", observed_at=_RECENT)
        assert obs.parameter_set_id == "ps-v1.1"

    def test_all_observations_accumulates(self):
        m = _monitor()
        for i in range(5):
            m.record("spam", "spam", trace_id=f"t{i}", observed_at=_RECENT)
        assert len(m.all_observations()) == 5


# ── WindowMetrics shape tests ──────────────────────────────────────────────────

class TestWindowMetricsShape:
    def test_window_metrics_returns_window_metrics(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert isinstance(metrics, WindowMetrics)

    def test_window_hours_matches_policy(self):
        m = _monitor()
        metrics = m.window_metrics(now=_NOW)
        assert metrics.window_hours == 24

    def test_window_end_matches_now(self):
        m = _monitor()
        metrics = m.window_metrics(now=_NOW)
        assert metrics.window_end == _NOW.isoformat()

    def test_window_start_is_window_hours_before_now(self):
        m = _monitor()
        metrics = m.window_metrics(now=_NOW)
        expected_start = (_NOW - timedelta(hours=24)).isoformat()
        assert metrics.window_start == expected_start

    def test_empty_window_gives_zero_accuracy(self):
        m = _monitor()
        metrics = m.window_metrics(now=_NOW)
        assert metrics.accuracy == 0.0
        assert metrics.samples == 0

    def test_samples_counts_in_window_only(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_RECENT)
        m.record("ham", "ham", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.samples == 2


# ── Window filtering tests ─────────────────────────────────────────────────────

class TestWindowFiltering:
    def test_excludes_observations_outside_window(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_OLD)    # outside window
        metrics = m.window_metrics(now=_NOW)
        assert metrics.samples == 0

    def test_includes_observations_inside_window(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_RECENT)   # 1h ago, inside 24h
        metrics = m.window_metrics(now=_NOW)
        assert metrics.samples == 1

    def test_boundary_observation_included(self):
        m = _monitor()
        # exactly at the window start boundary
        boundary = _NOW - timedelta(hours=24)
        m.record("spam", "spam", observed_at=boundary)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.samples == 1

    def test_mixed_in_and_out_of_window(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_OLD)     # outside
        m.record("ham",  "ham",  observed_at=_RECENT)  # inside
        m.record("promo","promo",observed_at=_RECENT)  # inside
        metrics = m.window_metrics(now=_NOW)
        assert metrics.samples == 2

    def test_observations_in_window_helper_matches(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_OLD)
        m.record("ham",  "ham",  observed_at=_RECENT)
        in_window = m.observations_in_window(now=_NOW)
        assert len(in_window) == 1
        assert in_window[0].ground_truth == "ham"


# ── Accuracy computation tests ─────────────────────────────────────────────────

class TestAccuracyComputation:
    def test_perfect_accuracy(self):
        m = _monitor()
        for lbl in _LABELS:
            m.record(lbl, lbl, observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.accuracy == pytest.approx(1.0)

    def test_zero_accuracy_all_wrong(self):
        m = _monitor()
        m.record("ham",   "spam", observed_at=_RECENT)
        m.record("promo", "ham",  observed_at=_RECENT)
        m.record("spam",  "promo",observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.accuracy == pytest.approx(0.0)

    def test_partial_accuracy(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_RECENT)   # correct
        m.record("spam", "spam", observed_at=_RECENT)   # correct
        m.record("ham",  "spam", observed_at=_RECENT)   # wrong
        m.record("ham",  "spam", observed_at=_RECENT)   # wrong
        metrics = m.window_metrics(now=_NOW)
        assert metrics.accuracy == pytest.approx(0.5)


# ── Degradation detection tests ────────────────────────────────────────────────

class TestDegradationDetection:
    def test_degradation_detected_when_accuracy_drops(self):
        # reference=0.91, threshold=0.05 → any accuracy < 0.86 triggers
        m = _monitor(ref=0.91)
        # fill with mostly wrong predictions (accuracy ≈ 0.0)
        for i in range(15):
            m.record("ham", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.degradation_detected is True

    def test_no_degradation_when_accuracy_above_threshold(self):
        m = _monitor(ref=0.91)
        # accuracy = 1.0, well above 0.91 - 0.05 = 0.86
        for i in range(15):
            m.record("spam", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.degradation_detected is False

    def test_no_degradation_without_enough_samples(self):
        m = _monitor(ref=0.91)
        # Only 5 samples (< MIN_SAMPLES_IN_WINDOW=10), even with bad accuracy
        for i in range(5):
            m.record("ham", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.degradation_detected is False
        assert metrics.enough_samples is False

    def test_enough_samples_true_at_min_threshold(self):
        m = _monitor(ref=0.91)
        for i in range(10):
            m.record("spam", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.enough_samples is True

    def test_no_degradation_when_reference_not_set(self):
        m = ProductionMonitor(_policy(), reference_accuracy=None, labels=_LABELS)
        for i in range(15):
            m.record("ham", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.degradation_detected is False
        assert metrics.reference_accuracy is None

    def test_actual_degradation_computed_correctly(self):
        m = _monitor(ref=0.91)
        # 5 correct, 10 wrong → accuracy = 5/15 ≈ 0.333
        for _ in range(5):
            m.record("spam", "spam", observed_at=_RECENT)
        for _ in range(10):
            m.record("ham", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        expected_deg = 0.91 - (5 / 15)
        assert metrics.actual_degradation == pytest.approx(expected_deg, abs=1e-6)


# ── Per-label metrics tests ────────────────────────────────────────────────────

class TestPerLabelMetrics:
    def test_per_label_metrics_present_for_known_labels(self):
        m = _monitor()
        for lbl in _LABELS:
            m.record(lbl, lbl, observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        for lbl in _LABELS:
            assert lbl in metrics.per_label

    def test_per_label_precision_recall_f1(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_RECENT)
        m.record("spam", "spam", observed_at=_RECENT)
        m.record("ham",  "ham",  observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        spam = metrics.per_label.get("spam", {})
        assert spam["precision"] == pytest.approx(1.0)
        assert spam["recall"] == pytest.approx(1.0)
        assert spam["f1"] == pytest.approx(1.0)

    def test_per_label_empty_without_labels(self):
        m = ProductionMonitor(_policy(), reference_accuracy=0.91, labels=[])
        m.record("spam", "spam", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.per_label == {}


# ── parameter_set_ids tracking tests ──────────────────────────────────────────

class TestParameterSetTracking:
    def test_parameter_set_ids_collected(self):
        m = _monitor()
        m.record("spam", "spam", parameter_set_id="ps-v1.0", observed_at=_RECENT)
        m.record("ham",  "ham",  parameter_set_id="ps-v1.1", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert "ps-v1.0" in metrics.parameter_set_ids
        assert "ps-v1.1" in metrics.parameter_set_ids

    def test_parameter_set_ids_deduplicated(self):
        m = _monitor()
        for _ in range(5):
            m.record("spam", "spam", parameter_set_id="ps-v1.1", observed_at=_RECENT)
        metrics = m.window_metrics(now=_NOW)
        assert metrics.parameter_set_ids.count("ps-v1.1") == 1

    def test_empty_parameter_set_id_excluded(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_RECENT)   # no parameter_set_id
        metrics = m.window_metrics(now=_NOW)
        assert metrics.parameter_set_ids == []


# ── utility tests ──────────────────────────────────────────────────────────────

class TestUtility:
    def test_clear_removes_all_observations(self):
        m = _monitor()
        m.record("spam", "spam", observed_at=_RECENT)
        m.clear()
        assert m.all_observations() == []

    def test_set_reference_accuracy_updates_detection(self):
        m = ProductionMonitor(_policy(), reference_accuracy=None, labels=_LABELS)
        for _ in range(15):
            m.record("ham", "spam", observed_at=_RECENT)
        assert m.window_metrics(now=_NOW).degradation_detected is False
        m.set_reference_accuracy(0.91)
        assert m.window_metrics(now=_NOW).degradation_detected is True
