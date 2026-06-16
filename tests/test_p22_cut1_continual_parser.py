"""P22 C1 — Parser e IR para .mxcontinual."""
import pytest

from matrixai.continual import (
    ContinualPolicySpec,
    MxcontinualParseError,
    canonical_dict,
    compute_policy_hash,
    parse_mxcontinual,
)


# ── fixture: full valid policy ─────────────────────────────────────────────────

_FULL_POLICY = """
CONTINUAL_POLICY IncidentClassifierContinual
  TARGET_MODEL examples/incident-classifier.mxai
  BASE_PARAMETER_SET runs/incident_classifier_001/params.best.json
  REGISTRY_NAME incident_classifier
  BASE_VERSION v1.0

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
    LABEL_TYPE Label[support, sales, ops, other]
    SOURCES [api, cli, file_watch]
    FILE_WATCH_PATH /var/matrixai/feedback/incident/
  END

  DRIFT_DETECTION
    FEATURES [severity, source_trust, hour_of_day, sender_domain]
    METHODS
      severity: psi threshold=0.2
      source_trust: psi threshold=0.2
      hour_of_day: ks threshold=0.1
      sender_domain: chi_square threshold=0.05
    END
    MIN_SAMPLES 500
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  CONCEPT_DRIFT
    PREDICTION_METRIC accuracy
    REFERENCE_VALUE 0.91
    THRESHOLD_DEGRADATION 0.05
    MIN_SAMPLES_WITH_LABEL 200
  END

  UPDATE_TRIGGER
    ANY_OF
      DRIFT_THRESHOLD_EXCEEDED
      CONCEPT_DRIFT_DETECTED
      SCHEDULED weekly
    END
    MIN_NEW_SAMPLES 200
    MIN_GROUND_TRUTH_RATIO 0.6
    COOLDOWN_DAYS 3
  END

  TRAINING
    METHOD incremental_finetune
    LEARNING_RATE_FACTOR 0.1
    MAX_EPOCHS 20
    EARLY_STOP patience=3 metric=validation_loss
    DATASET_MIX
      BASE_WEIGHT 0.4
      PRODUCTION_WEIGHT 0.6
      RECENCY_DECAY exponential half_life_days=30
    END
    SEED_INHERITANCE deterministic_from_parent
  END

  APPROVAL_GATE
    HOLDOUT_FRACTION 0.2
    HOLDOUT_SOURCE production_recent
    REGRESSION_GUARD
      METRIC accuracy
      MUST_IMPROVE_BY 0.0
      MAX_DEGRADATION_PER_LABEL 0.03
    END
    HUMAN_APPROVAL true
    APPROVAL_CHANNEL cli
    APPROVAL_TIMEOUT_HOURS 48
  END

  ROLLBACK
    AUTO_TRIGGER true
    METRIC accuracy
    SLIDING_WINDOW_HOURS 24
    DEGRADATION_THRESHOLD 0.05
    MIN_SAMPLES_IN_WINDOW 100
    NOTIFY_CAPABILITY notification
    NOTIFY_SCOPE
      allowed_urls = ["https://hooks.example.com/ops"]
    END
  END

  AUDIT
    PERSIST_DRIFT_REPORTS true
    PERSIST_UPDATE_TRACES true
    EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT true
    REFINEMENT_DRIFT_PERSISTENCE_DAYS 14
    SIGNATURE_REQUIRED true
  END
END
"""

# Minimal valid policy (no CONCEPT_DRIFT, no REGISTRY, no NOTIFY_SCOPE)
_MINIMAL_POLICY = """
CONTINUAL_POLICY MinimalPolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json

  GROUND_TRUTH
    WINDOW_DAYS 3
    REQUIRED_FIELD label
  END

  DRIFT_DETECTION
    FEATURES [score, confidence]
    METHODS
      score: ks threshold=0.15
    END
    MIN_SAMPLES 100
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES 100
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


# ── parse success ──────────────────────────────────────────────────────────────

class TestContinualParserAccepts:
    def test_full_policy_parses_name(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        assert spec.name == "IncidentClassifierContinual"

    def test_full_policy_target_model(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        assert spec.target_model == "examples/incident-classifier.mxai"

    def test_full_policy_base_parameter_set(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        assert spec.base_parameter_set == "runs/incident_classifier_001/params.best.json"

    def test_full_policy_registry_and_version(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        assert spec.registry_name == "incident_classifier"
        assert spec.base_version == "v1.0"

    def test_ground_truth_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        gt = spec.ground_truth
        assert gt.window_days == 7
        assert gt.required_field == "label"
        assert gt.label_type == "Label[support, sales, ops, other]"
        assert "api" in gt.sources and "file_watch" in gt.sources
        assert gt.file_watch_path == "/var/matrixai/feedback/incident/"

    def test_drift_detection_features(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        dd = spec.drift_detection
        assert "severity" in dd.features
        assert "sender_domain" in dd.features

    def test_drift_detection_methods(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        dd = spec.drift_detection
        assert dd.methods["severity"].method == "psi"
        assert dd.methods["severity"].threshold == 0.2
        assert dd.methods["hour_of_day"].method == "ks"
        assert dd.methods["sender_domain"].method == "chi_square"

    def test_drift_detection_min_samples(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        assert spec.drift_detection.min_samples == 500

    def test_concept_drift_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        cd = spec.concept_drift
        assert cd is not None
        assert cd.prediction_metric == "accuracy"
        assert cd.reference_value == pytest.approx(0.91)
        assert cd.threshold_degradation == pytest.approx(0.05)
        assert cd.min_samples_with_label == 200

    def test_update_trigger_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        ut = spec.update_trigger
        assert "drift_threshold_exceeded" in ut.triggers
        assert "concept_drift_detected" in ut.triggers
        assert "scheduled_weekly" in ut.triggers
        assert ut.min_new_samples == 200
        assert ut.min_ground_truth_ratio == pytest.approx(0.6)
        assert ut.cooldown_days == 3

    def test_training_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        tr = spec.training
        assert tr.method == "incremental_finetune"
        assert tr.learning_rate_factor == pytest.approx(0.1)
        assert tr.max_epochs == 20
        assert tr.early_stop is not None
        assert tr.early_stop.patience == 3
        assert tr.early_stop.metric == "validation_loss"

    def test_dataset_mix_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        dm = spec.training.dataset_mix
        assert dm.base_weight == pytest.approx(0.4)
        assert dm.production_weight == pytest.approx(0.6)
        assert dm.recency_decay.method == "exponential"
        assert dm.recency_decay.half_life_days == 30

    def test_approval_gate_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        ag = spec.approval_gate
        assert ag.holdout_fraction == pytest.approx(0.2)
        assert ag.holdout_source == "production_recent"
        assert ag.human_approval is True
        assert ag.approval_channel == "cli"
        assert ag.approval_timeout_hours == 48

    def test_regression_guard_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        rg = spec.approval_gate.regression_guard
        assert rg.metric == "accuracy"
        assert rg.must_improve_by == pytest.approx(0.0)
        assert rg.max_degradation_per_label == pytest.approx(0.03)

    def test_rollback_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        rb = spec.rollback
        assert rb.auto_trigger is True
        assert rb.metric == "accuracy"
        assert rb.sliding_window_hours == 24
        assert rb.degradation_threshold == pytest.approx(0.05)
        assert rb.min_samples_in_window == 100
        assert rb.notify_capability == "notification"

    def test_rollback_notify_scope(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        assert "allowed_urls" in spec.rollback.notify_scope
        assert "hooks.example.com" in spec.rollback.notify_scope["allowed_urls"][0]

    def test_audit_parsed(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        au = spec.audit
        assert au.persist_drift_reports is True
        assert au.emit_refinement_hint_on_sustained_drift is True
        assert au.refinement_drift_persistence_days == 14
        assert au.signature_required is True

    def test_minimal_policy_no_concept_drift(self):
        spec = parse_mxcontinual(_MINIMAL_POLICY)
        assert spec.concept_drift is None

    def test_minimal_policy_no_registry(self):
        spec = parse_mxcontinual(_MINIMAL_POLICY)
        assert spec.registry_name is None
        assert spec.base_version is None

    def test_linear_recency_decay(self):
        spec = parse_mxcontinual(_MINIMAL_POLICY)
        rd = spec.training.dataset_mix.recency_decay
        assert rd.method == "linear"
        assert rd.half_life_days is None

    def test_comments_ignored(self):
        src = "# leading comment\n" + _MINIMAL_POLICY + "\n# trailing comment\n"
        spec = parse_mxcontinual(src)
        assert spec.name == "MinimalPolicy"


# ── policy_hash ───────────────────────────────────────────────────────────────

class TestPolicyHash:
    def test_hash_is_deterministic(self):
        spec1 = parse_mxcontinual(_FULL_POLICY)
        spec2 = parse_mxcontinual(_FULL_POLICY)
        assert spec1.policy_hash == spec2.policy_hash

    def test_hash_has_sha256_prefix(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        assert spec.policy_hash.startswith("sha256:")

    def test_hash_changes_when_threshold_changes(self):
        modified = _FULL_POLICY.replace(
            "DEGRADATION_THRESHOLD 0.05",
            "DEGRADATION_THRESHOLD 0.10",
        )
        spec1 = parse_mxcontinual(_FULL_POLICY)
        spec2 = parse_mxcontinual(modified)
        assert spec1.policy_hash != spec2.policy_hash

    def test_hash_changes_when_lr_factor_changes(self):
        modified = _FULL_POLICY.replace(
            "LEARNING_RATE_FACTOR 0.1",
            "LEARNING_RATE_FACTOR 0.05",
        )
        spec1 = parse_mxcontinual(_FULL_POLICY)
        spec2 = parse_mxcontinual(modified)
        assert spec1.policy_hash != spec2.policy_hash

    def test_canonical_dict_has_required_keys(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        cd = canonical_dict(spec)
        assert "name" in cd
        assert "target_model" in cd
        assert "drift_detection" in cd
        assert "approval_gate" in cd
        assert "rollback" in cd
        assert "training" in cd

    def test_compute_policy_hash_matches_spec(self):
        spec = parse_mxcontinual(_FULL_POLICY)
        # Recompute without the stored hash
        recomputed = compute_policy_hash(spec)
        assert recomputed == spec.policy_hash


# ── parse errors ──────────────────────────────────────────────────────────────

class TestContinualParserRejects:
    def test_rejects_missing_target_model(self):
        src = _FULL_POLICY.replace(
            "TARGET_MODEL examples/incident-classifier.mxai\n", ""
        )
        with pytest.raises(MxcontinualParseError, match="TARGET_MODEL"):
            parse_mxcontinual(src)

    def test_rejects_missing_base_parameter_set(self):
        src = _FULL_POLICY.replace(
            "BASE_PARAMETER_SET runs/incident_classifier_001/params.best.json\n", ""
        )
        with pytest.raises(MxcontinualParseError, match="BASE_PARAMETER_SET"):
            parse_mxcontinual(src)

    def test_rejects_registry_without_version(self):
        src = _FULL_POLICY.replace("BASE_VERSION v1.0\n", "")
        with pytest.raises(MxcontinualParseError, match="BASE_VERSION"):
            parse_mxcontinual(src)

    def test_rejects_invalid_drift_method(self):
        src = _FULL_POLICY.replace(
            "severity: psi threshold=0.2",
            "severity: neural_drift threshold=0.2",
        )
        with pytest.raises(MxcontinualParseError, match="drift method"):
            parse_mxcontinual(src)

    def test_rejects_mix_weights_not_summing_to_one(self):
        src = _FULL_POLICY.replace(
            "BASE_WEIGHT 0.4\n      PRODUCTION_WEIGHT 0.6",
            "BASE_WEIGHT 0.3\n      PRODUCTION_WEIGHT 0.3",
        )
        with pytest.raises(MxcontinualParseError, match="sum to 1.0"):
            parse_mxcontinual(src)

    def test_rejects_invalid_learning_rate_factor_too_low(self):
        src = _FULL_POLICY.replace("LEARNING_RATE_FACTOR 0.1", "LEARNING_RATE_FACTOR 0.001")
        with pytest.raises(MxcontinualParseError, match="LEARNING_RATE_FACTOR"):
            parse_mxcontinual(src)

    def test_rejects_invalid_learning_rate_factor_too_high(self):
        src = _FULL_POLICY.replace("LEARNING_RATE_FACTOR 0.1", "LEARNING_RATE_FACTOR 1.5")
        with pytest.raises(MxcontinualParseError, match="LEARNING_RATE_FACTOR"):
            parse_mxcontinual(src)

    def test_rejects_holdout_fraction_out_of_range_low(self):
        src = _FULL_POLICY.replace("HOLDOUT_FRACTION 0.2", "HOLDOUT_FRACTION 0.05")
        with pytest.raises(MxcontinualParseError, match="HOLDOUT_FRACTION"):
            parse_mxcontinual(src)

    def test_rejects_holdout_fraction_out_of_range_high(self):
        src = _FULL_POLICY.replace("HOLDOUT_FRACTION 0.2", "HOLDOUT_FRACTION 0.8")
        with pytest.raises(MxcontinualParseError, match="HOLDOUT_FRACTION"):
            parse_mxcontinual(src)

    def test_rejects_degradation_threshold_out_of_range(self):
        src = _FULL_POLICY.replace("DEGRADATION_THRESHOLD 0.05", "DEGRADATION_THRESHOLD 0.9")
        with pytest.raises(MxcontinualParseError, match="DEGRADATION_THRESHOLD"):
            parse_mxcontinual(src)

    def test_rejects_method_for_undeclared_feature(self):
        src = _FULL_POLICY.replace(
            "severity: psi threshold=0.2",
            "severity: psi threshold=0.2\n      unknown_feat: ks threshold=0.1",
        )
        with pytest.raises(MxcontinualParseError, match="unknown_feat"):
            parse_mxcontinual(src)

    def test_rejects_missing_ground_truth_block(self):
        # Remove the GROUND_TRUTH block entirely using exact line match
        lines = _FULL_POLICY.splitlines()
        result = []
        skip = False
        skip_depth = 0
        for line in lines:
            stripped = line.strip()
            if not skip and stripped == "GROUND_TRUTH":
                skip = True
                skip_depth = 1
                continue
            if skip:
                # track nested depth so we consume only the right END
                if stripped in ("METHODS", "DATASET_MIX", "REGRESSION_GUARD",
                                "ANY_OF", "NOTIFY_SCOPE"):
                    skip_depth += 1
                elif stripped == "END":
                    skip_depth -= 1
                    if skip_depth == 0:
                        skip = False
                continue
            result.append(line)
        src = "\n".join(result)
        with pytest.raises(MxcontinualParseError, match="GROUND_TRUTH"):
            parse_mxcontinual(src)

    def test_rejects_invalid_training_method(self):
        src = _FULL_POLICY.replace(
            "METHOD incremental_finetune",
            "METHOD online_sgd",
        )
        with pytest.raises(MxcontinualParseError, match="unknown method"):
            parse_mxcontinual(src)

    def test_rejects_empty_source(self):
        with pytest.raises(MxcontinualParseError, match="Empty"):
            parse_mxcontinual("")

    def test_rejects_missing_continual_policy_header(self):
        with pytest.raises(MxcontinualParseError, match="CONTINUAL_POLICY"):
            parse_mxcontinual("TARGET_MODEL foo.mxai\n")
