"""P22 C3 — DriftDetector: PSI, KS, chi-square, JS, Wasserstein, ConceptDriftDetector."""
import math

import pytest

from matrixai.continual import (
    ConceptDriftDetector,
    ConceptDriftReport,
    DriftDetector,
    DriftReport,
    FeatureDriftResult,
    MAX_CHI_SQUARE_CATEGORIES,
    compute_chi_square,
    compute_js_divergence,
    compute_ks_statistic,
    compute_psi,
    compute_wasserstein,
    parse_mxcontinual,
)
from matrixai.continual.drift import compute_chi_square_from_samples


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_SRC = """
CONTINUAL_POLICY DriftTestPolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
    SOURCES [api]
  END

  DRIFT_DETECTION
    FEATURES [severity, hour_of_day, sender_domain, score, size]
    METHODS
      severity: psi threshold=0.2
      hour_of_day: ks threshold=0.1
      sender_domain: chi_square threshold=5.0
      score: js threshold=0.1
      size: wasserstein threshold=0.5
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

# Reference data: stable distributions
_N = 200
_REF = {
    "severity":     [float(i % 5) for i in range(_N)],       # uniform 0-4
    "hour_of_day":  [float(i % 24) for i in range(_N)],      # uniform 0-23
    "sender_domain":[float(i % 4) for i in range(_N)],       # 4 categories
    "score":        [0.5 + 0.1 * math.sin(i) for i in range(_N)],
    "size":         [float(100 + i % 50) for i in range(_N)],
}

# Production data matching reference (no drift)
_PROD_SAME = {k: list(v) for k, v in _REF.items()}

# Production data with severe drift
_PROD_DRIFT = {
    "severity":     [4.0] * _N,            # all at max — big shift
    "hour_of_day":  [23.0] * _N,           # all at 23
    "sender_domain":[3.0] * _N,            # all in one category
    "score":        [0.9 + 0.05 * math.sin(i) for i in range(_N)],  # shifted
    "size":         [float(200 + i % 50) for i in range(_N)],        # shifted +100
}


def _policy():
    return parse_mxcontinual(_POLICY_SRC)


# ── PSI tests ─────────────────────────────────────────────────────────────────

class TestComputePSI:
    def test_no_drift_when_distributions_match(self):
        ref = [float(i % 10) for i in range(100)]
        psi = compute_psi(ref, ref)
        assert psi < 0.01

    def test_detects_drift_when_distribution_shifts(self):
        ref = [float(i) for i in range(100)]        # uniform 0-99
        obs = [float(90 + i % 10) for i in range(100)]  # clustered at 90-99
        psi = compute_psi(ref, obs)
        assert psi > 0.2

    def test_symmetric_is_not_guaranteed(self):
        # PSI is asymmetric by design (uses reference bins)
        ref = [float(i) for i in range(100)]
        obs = [50.0] * 100
        psi = compute_psi(ref, obs)
        assert psi > 0.0

    def test_identical_small_sample(self):
        ref = [1.0, 2.0, 3.0]
        psi = compute_psi(ref, ref)
        assert psi < 0.01

    def test_empty_observed_returns_zero(self):
        ref = [1.0, 2.0, 3.0]
        assert compute_psi(ref, []) == 0.0

    def test_constant_reference_returns_zero(self):
        ref = [1.0] * 50
        obs = [2.0] * 50
        assert compute_psi(ref, obs) == 0.0


# ── KS tests ──────────────────────────────────────────────────────────────────

class TestComputeKS:
    def test_identical_distributions_zero(self):
        ref = [float(i) for i in range(100)]
        ks = compute_ks_statistic(ref, ref)
        assert ks < 0.01

    def test_detects_continuous_distribution_change(self):
        ref = [float(i) for i in range(100)]     # uniform 0-99
        obs = [float(50 + i) for i in range(100)]  # shifted +50
        ks = compute_ks_statistic(ref, obs)
        assert ks > 0.3

    def test_bounded_zero_to_one(self):
        ref = [0.0] * 50
        obs = [1.0] * 50
        ks = compute_ks_statistic(ref, obs)
        assert 0.0 <= ks <= 1.0

    def test_empty_returns_zero(self):
        assert compute_ks_statistic([], [1.0, 2.0]) == 0.0

    def test_small_sample_handled(self):
        ref = [1.0, 2.0, 3.0]
        obs = [10.0, 20.0, 30.0]
        ks = compute_ks_statistic(ref, obs)
        assert ks > 0.5


# ── Chi-square tests ───────────────────────────────────────────────────────────

class TestComputeChiSquare:
    def test_identical_counts_zero(self):
        counts = {"a": 50, "b": 30, "c": 20}
        chi2 = compute_chi_square(counts, counts)
        assert chi2 < 0.01

    def test_detects_categorical_distribution_change(self):
        ref = {"a": 50, "b": 50}
        obs = {"a": 90, "b": 10}
        chi2 = compute_chi_square(ref, obs)
        assert chi2 > 5.0

    def test_empty_returns_zero(self):
        assert compute_chi_square({}, {"a": 10}) == 0.0

    def test_new_category_in_observed(self):
        ref = {"a": 50, "b": 50}
        obs = {"a": 40, "b": 40, "c": 20}  # new category c
        chi2 = compute_chi_square(ref, obs)
        # New category has no expected → contributes to chi2
        assert chi2 > 0.0


# ── JS divergence tests ───────────────────────────────────────────────────────

class TestComputeJS:
    def test_identical_distributions_near_zero(self):
        ref = [float(i % 10) for i in range(100)]
        js = compute_js_divergence(ref, ref)
        assert js < 0.01

    def test_js_divergence_bounded_zero_to_one(self):
        ref = [0.0] * 100
        obs = [1.0] * 100
        js = compute_js_divergence(ref, obs)
        assert 0.0 <= js <= 1.0

    def test_detects_distribution_shift(self):
        ref = [float(i) for i in range(100)]
        obs = [float(50 + i) for i in range(100)]
        js = compute_js_divergence(ref, obs)
        assert js > 0.05

    def test_empty_returns_zero(self):
        assert compute_js_divergence([], [1.0]) == 0.0


# ── Wasserstein tests ─────────────────────────────────────────────────────────

class TestComputeWasserstein:
    def test_identical_distributions_zero(self):
        ref = [float(i) for i in range(50)]
        w = compute_wasserstein(ref, ref)
        assert w < 0.01

    def test_detects_shift(self):
        ref = [float(i) for i in range(50)]
        obs = [float(100 + i) for i in range(50)]
        w = compute_wasserstein(ref, obs)
        assert w > 50.0

    def test_handles_small_samples(self):
        ref = [1.0, 2.0]
        obs = [3.0, 4.0]
        w = compute_wasserstein(ref, obs)
        assert w > 0.0

    def test_empty_returns_zero(self):
        assert compute_wasserstein([], [1.0]) == 0.0


# ── DriftDetector integration ─────────────────────────────────────────────────

class TestDriftDetector:
    def test_produces_drift_report(self):
        policy = _policy()
        detector = DriftDetector(policy)
        report = detector.run_check(_REF, _PROD_SAME)
        assert isinstance(report, DriftReport)
        assert report.policy_hash == policy.policy_hash
        assert isinstance(report.checked_at, str)

    def test_no_drift_when_distributions_match(self):
        policy = _policy()
        detector = DriftDetector(policy)
        report = detector.run_check(_REF, _PROD_SAME)
        assert report.drift_detected is False

    def test_drift_detected_when_distribution_shifts(self):
        policy = _policy()
        detector = DriftDetector(policy)
        report = detector.run_check(_REF, _PROD_DRIFT)
        assert report.drift_detected is True

    def test_drift_threshold_exceeded_sets_feature_flag(self):
        policy = _policy()
        detector = DriftDetector(policy)
        report = detector.run_check(_REF, _PROD_DRIFT)
        # At least one feature should have drift_detected=True
        drifted = [f for f, r in report.results.items() if r.drift_detected]
        assert len(drifted) > 0

    def test_report_contains_all_features(self):
        policy = _policy()
        detector = DriftDetector(policy)
        report = detector.run_check(_REF, _PROD_SAME)
        dd_features = set(policy.drift_detection.features)
        assert set(report.features_checked) == dd_features

    def test_respects_min_samples(self):
        policy = _policy()
        detector = DriftDetector(policy)
        # Only 5 samples — below MIN_SAMPLES=50
        small_prod = {k: v[:5] for k, v in _PROD_DRIFT.items()}
        report = detector.run_check(_REF, small_prod)
        assert report.enough_samples is False
        assert all(r.enough_samples is False for r in report.results.values())
        # Even with drifted data, overall drift_detected should be False
        assert report.drift_detected is False

    def test_enough_samples_flag_set(self):
        policy = _policy()
        detector = DriftDetector(policy)
        report = detector.run_check(_REF, _PROD_SAME)
        assert report.enough_samples is True
        assert report.total_production_samples == _N

    def test_handles_unknown_feature_gracefully(self):
        policy = _policy()
        detector = DriftDetector(policy)
        # production_data has no entry for 'severity' and 'hour_of_day'
        partial_prod = {
            "score": _PROD_SAME["score"],
            "size": _PROD_SAME["size"],
            "sender_domain": _PROD_SAME["sender_domain"],
        }
        report = detector.run_check(_REF, partial_prod)
        # Missing-data features should be skipped, not raise
        assert "severity" in report.results
        assert report.results["severity"].skipped is True

    def test_feature_result_has_observed_value(self):
        policy = _policy()
        detector = DriftDetector(policy)
        report = detector.run_check(_REF, _PROD_DRIFT)
        result = report.results["severity"]
        assert result.observed_value > 0.0
        assert result.threshold == pytest.approx(0.2)

    def test_feature_without_declared_method_is_skipped(self):
        # Add a feature with no method declaration
        src = _POLICY_SRC.replace(
            "FEATURES [severity, hour_of_day, sender_domain, score, size]",
            "FEATURES [severity, hour_of_day, sender_domain, score, size, extra_feat]",
        )
        policy = parse_mxcontinual(src)
        detector = DriftDetector(policy)
        ref = {**_REF, "extra_feat": [1.0] * _N}
        prod = {**_PROD_SAME, "extra_feat": [2.0] * _N}
        report = detector.run_check(ref, prod)
        assert "extra_feat" in report.results
        assert report.results["extra_feat"].skipped is True
        assert "no method" in report.results["extra_feat"].skip_reason

    def test_ks_feature_drift_detected_above_threshold(self):
        policy = _policy()
        detector = DriftDetector(policy)
        # hour_of_day: KS threshold=0.1 — use extreme shift
        ref = {k: list(v) for k, v in _REF.items()}
        prod = {k: list(v) for k, v in _PROD_SAME.items()}
        prod["hour_of_day"] = [23.0] * _N  # all at max → KS >> 0.1
        report = detector.run_check(ref, prod)
        assert report.results["hour_of_day"].drift_detected is True

    def test_per_feature_min_samples_not_global_max(self):
        """A feature with insufficient samples must NOT mark drift even if
        another feature has enough samples — guard is per-feature, not global."""
        policy = _policy()
        detector = DriftDetector(policy)
        # severity has only 3 samples (< MIN_SAMPLES=50), others have 200
        prod = {k: list(v) for k, v in _PROD_DRIFT.items()}
        prod["severity"] = [4.0] * 3   # extreme drift value, but too few samples
        report = detector.run_check(_REF, prod)
        # severity must NOT be flagged despite high drift value
        assert report.results["severity"].drift_detected is False
        assert report.results["severity"].samples_used == 3
        assert report.results["severity"].enough_samples is False
        assert report.results["hour_of_day"].enough_samples is True
        # Overall enough_samples should still be True (other features qualify)
        assert report.enough_samples is True

    def test_psi_handles_observed_values_below_reference_range(self):
        """PSI must not produce negative bin indices when observed values
        fall below the reference distribution's minimum."""
        ref = [float(i) for i in range(10, 100)]   # min=10, max=99
        obs = [0.0, 1.0, 2.0] + [float(i) for i in range(10, 20)]  # some below min
        psi = compute_psi(ref, obs)
        assert psi >= 0.0   # must not crash or return nonsense

    def test_psi_handles_observed_values_above_reference_range(self):
        """PSI must not overflow when observed values exceed the reference max."""
        ref = [float(i) for i in range(100)]    # max=99
        obs = [200.0, 300.0] + [float(i) for i in range(90, 100)]  # some above max
        psi = compute_psi(ref, obs)
        assert psi >= 0.0


# ── chi_square cardinality guard ──────────────────────────────────────────────

class TestChiSquareCardinality:
    def test_low_cardinality_reference_succeeds(self):
        # 4 unique categories — well below MAX_CHI_SQUARE_CATEGORIES
        ref = [float(i % 4) for i in range(200)]
        obs = [float(i % 4) for i in range(200)]
        result = compute_chi_square_from_samples(ref, obs)
        assert result >= 0.0

    def test_high_cardinality_raises_value_error(self):
        # 100 unique int-rounded values — continuous feature
        ref = [float(i) for i in range(100)]
        obs = [float(i) for i in range(100)]
        with pytest.raises(ValueError, match="chi_square"):
            compute_chi_square_from_samples(ref, obs)

    def test_error_message_suggests_alternative(self):
        ref = [float(i) for i in range(100)]
        obs = [float(i) for i in range(100)]
        with pytest.raises(ValueError, match="psi.*ks|ks.*psi"):
            compute_chi_square_from_samples(ref, obs)

    def test_exactly_at_limit_succeeds(self):
        # MAX_CHI_SQUARE_CATEGORIES unique values → allowed (limit is strict >)
        ref = [float(i) for i in range(MAX_CHI_SQUARE_CATEGORIES)]
        obs = [float(i) for i in range(MAX_CHI_SQUARE_CATEGORIES)]
        result = compute_chi_square_from_samples(ref, obs)
        assert result >= 0.0

    def test_one_over_limit_raises(self):
        ref = [float(i) for i in range(MAX_CHI_SQUARE_CATEGORIES + 1)]
        obs = [float(i) for i in range(MAX_CHI_SQUARE_CATEGORIES + 1)]
        with pytest.raises(ValueError):
            compute_chi_square_from_samples(ref, obs)

    def test_drift_detector_raises_on_continuous_chi_square(self):
        """DriftDetector.run_check propagates ValueError when chi_square is
        used on a feature whose reference data has high cardinality."""
        # sender_domain uses chi_square (threshold=5.0 in _POLICY_SRC).
        # Inject 200 unique integer-valued floats as reference — far above the
        # MAX_CHI_SQUARE_CATEGORIES=50 limit — to trigger the cardinality guard.
        policy = _policy()
        detector = DriftDetector(policy)
        ref = {**_REF, "sender_domain": [float(i) for i in range(200)]}
        prod = {**_PROD_SAME, "sender_domain": [float(i) for i in range(200)]}
        with pytest.raises(ValueError, match="chi_square"):
            detector.run_check(ref, prod)


# ── ConceptDriftDetector ──────────────────────────────────────────────────────

_CONCEPT_DRIFT_POLICY_SRC = """
CONTINUAL_POLICY ConceptDriftTestPolicy
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
      score: ks threshold=0.1
    END
    MIN_SAMPLES 50
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  CONCEPT_DRIFT
    PREDICTION_METRIC accuracy
    REFERENCE_VALUE 0.95
    THRESHOLD_DEGRADATION 0.05
    MIN_SAMPLES_WITH_LABEL 100
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
# alert_threshold = 0.95 - 0.05 = 0.90
# drift detected when current_value < 0.90 AND labeled_samples >= 100


def _concept_drift_policy():
    return parse_mxcontinual(_CONCEPT_DRIFT_POLICY_SRC)


class TestConceptDriftDetector:
    def test_no_drift_above_alert_threshold(self):
        policy = _concept_drift_policy()
        detector = ConceptDriftDetector(policy)
        report = detector.run_check(current_metric_value=0.93, labeled_sample_count=200)
        assert report.concept_drift_detected is False
        assert report.enough_labeled_samples is True

    def test_drift_detected_below_alert_threshold(self):
        policy = _concept_drift_policy()
        detector = ConceptDriftDetector(policy)
        report = detector.run_check(current_metric_value=0.85, labeled_sample_count=200)
        assert report.concept_drift_detected is True

    def test_insufficient_labeled_samples_suppresses_detection(self):
        # current_value is below threshold but sample count too low
        policy = _concept_drift_policy()
        detector = ConceptDriftDetector(policy)
        report = detector.run_check(current_metric_value=0.85, labeled_sample_count=50)
        assert report.concept_drift_detected is False
        assert report.enough_labeled_samples is False

    def test_report_fields_populated_correctly(self):
        policy = _concept_drift_policy()
        detector = ConceptDriftDetector(policy)
        report = detector.run_check(current_metric_value=0.88, labeled_sample_count=150)
        assert isinstance(report, ConceptDriftReport)
        assert report.prediction_metric == "accuracy"
        assert report.reference_value == pytest.approx(0.95)
        assert report.threshold_degradation == pytest.approx(0.05)
        assert report.alert_threshold == pytest.approx(0.90)
        assert report.current_value == pytest.approx(0.88)
        assert report.labeled_samples == 150
        assert isinstance(report.checked_at, str)

    def test_exact_boundary_not_flagged(self):
        # current == alert_threshold (0.90) — not strictly less than → no drift
        policy = _concept_drift_policy()
        detector = ConceptDriftDetector(policy)
        report = detector.run_check(current_metric_value=0.90, labeled_sample_count=200)
        assert report.concept_drift_detected is False

    def test_policy_without_concept_drift_raises(self):
        # _POLICY_SRC has no CONCEPT_DRIFT block → ValueError in constructor
        policy = parse_mxcontinual(_POLICY_SRC)
        with pytest.raises(ValueError, match="CONCEPT_DRIFT"):
            ConceptDriftDetector(policy)

    def test_checked_at_is_iso8601(self):
        from datetime import datetime, timezone
        policy = _concept_drift_policy()
        detector = ConceptDriftDetector(policy)
        fixed = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        report = detector.run_check(0.92, 200, now=fixed)
        assert "2026-05-23" in report.checked_at


# ── ConceptDrift parser range validation ──────────────────────────────────────

class TestConceptDriftParserValidation:
    """Parser must reject CONCEPT_DRIFT configs that make the detector always-silent."""

    _BASE = _CONCEPT_DRIFT_POLICY_SRC  # reference_value=0.95, threshold_degradation=0.05

    def _src(self, ref: float, deg: float) -> str:
        return self._BASE.replace(
            "    REFERENCE_VALUE 0.95", f"    REFERENCE_VALUE {ref}"
        ).replace(
            "    THRESHOLD_DEGRADATION 0.05", f"    THRESHOLD_DEGRADATION {deg}"
        )

    def test_threshold_equal_to_reference_raises(self):
        # degradation == reference → alert_threshold = 0.0, never fires for [0,1] metrics
        with pytest.raises(Exception, match="THRESHOLD_DEGRADATION"):
            parse_mxcontinual(self._src(0.5, 0.5))

    def test_threshold_greater_than_reference_raises(self):
        with pytest.raises(Exception, match="THRESHOLD_DEGRADATION"):
            parse_mxcontinual(self._src(0.5, 0.8))

    def test_threshold_zero_raises(self):
        with pytest.raises(Exception, match="THRESHOLD_DEGRADATION"):
            parse_mxcontinual(self._src(0.95, 0.0))

    def test_threshold_negative_raises(self):
        with pytest.raises(Exception, match="THRESHOLD_DEGRADATION"):
            parse_mxcontinual(self._src(0.95, -0.01))

    def test_valid_config_parses_ok(self):
        # threshold < reference → valid
        policy = parse_mxcontinual(self._src(0.95, 0.05))
        assert policy.concept_drift is not None
        assert policy.concept_drift.threshold_degradation == 0.05
