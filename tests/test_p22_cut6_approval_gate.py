"""P22 C6 — ApprovalGate: holdout evaluation, regression guard, human approval."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from matrixai.continual import (
    ApprovalGate,
    ApprovalGateReport,
    HoldoutMetrics,
    PendingApproval,
    RegressionGuardResult,
    parse_mxcontinual,
)
from matrixai.parameters.store import ParameterSet
from matrixai.training.data import SupervisedExample


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_AUTO = """
CONTINUAL_POLICY GateTestAuto
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

_POLICY_HUMAN = _POLICY_AUTO.replace("HUMAN_APPROVAL false", "HUMAN_APPROVAL true")
_POLICY_HUMAN_TIMEOUT = _POLICY_AUTO.replace(
    "HUMAN_APPROVAL false",
    "HUMAN_APPROVAL true\n    APPROVAL_CHANNEL cli\n    APPROVAL_TIMEOUT_HOURS 48",
)
_POLICY_STRICT_GUARD = _POLICY_AUTO.replace(
    "MUST_IMPROVE_BY 0.0", "MUST_IMPROVE_BY 0.5"
)
_POLICY_LOSS_METRIC = _POLICY_AUTO.replace(
    "METRIC accuracy", "METRIC loss"
).replace("MUST_IMPROVE_BY 0.0", "MUST_IMPROVE_BY 0.0")

_LABELS = ["spam", "ham", "promo"]
_N_FEAT = 4
_NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
_SIGNING_KEY = "a" * 64  # 32-byte hex key


def _make_ps(
    W1: list[list[float]] | None = None,
    b1: list[float] | None = None,
    ps_id: str = "ps_base",
    metrics: dict | None = None,
) -> ParameterSet:
    n_labels, n_feat = 3, _N_FEAT
    w = W1 or [[0.1 * (k + 1) for _ in range(n_feat)] for k in range(n_labels)]
    b = b1 or [0.0] * n_labels
    return ParameterSet(
        parameter_set_id=ps_id,
        model_hash="sha256:fake",
        parameter_schema_hash="params_fake",
        source="trained",
        parameters={
            "W1": {"function": "cls", "role": "weights", "type": "Tensor[3,4]",
                   "shape": [3, 4], "dtype": "float32", "initializer": "xavier", "values": w},
            "b1": {"function": "cls", "role": "bias", "type": "Vector[3]",
                   "shape": [3], "dtype": "float32", "initializer": "zeros", "values": b},
        },
        metrics=metrics or {},
    )


def _make_example(vector: list[float], label: str, idx: int = 0) -> SupervisedExample:
    return SupervisedExample(vector=vector, label=label, row_index=idx, row_hash=f"h{idx:04d}")


def _make_holdout(n_per_class: int = 10) -> list[SupervisedExample]:
    examples = []
    for i in range(n_per_class):
        examples.append(_make_example([2.0, 0.1, 0.0, 0.0], "spam", i))
        examples.append(_make_example([0.1, 2.0, 0.0, 0.0], "ham", n_per_class + i))
        examples.append(_make_example([0.5, 0.5, 2.0, 0.0], "promo", 2 * n_per_class + i))
    return examples


def _perfect_ps() -> ParameterSet:
    """W1 aligned to perfectly classify the holdout (high weight for discriminative feature)."""
    W1 = [
        [5.0, -5.0, -1.0, 0.0],   # spam: x[0] large
        [-5.0, 5.0, -1.0, 0.0],   # ham: x[1] large
        [-1.0, -1.0, 5.0, 0.0],   # promo: x[2] large
    ]
    return _make_ps(W1=W1, ps_id="ps_perfect", metrics={"parent_parameter_set_id": "ps_base"})


def _bad_ps() -> ParameterSet:
    """W1 that produces mostly wrong predictions."""
    W1 = [
        [-5.0, 5.0, -1.0, 0.0],   # spam classified as ham
        [-1.0, -1.0, 5.0, 0.0],   # ham classified as promo
        [5.0, -5.0, -1.0, 0.0],   # promo classified as spam
    ]
    return _make_ps(W1=W1, ps_id="ps_bad", metrics={"parent_parameter_set_id": "ps_base"})


def _policy(src: str = _POLICY_AUTO) -> Any:
    return parse_mxcontinual(src)


# ── report shape tests ─────────────────────────────────────────────────────────

class TestApprovalGateReportShape:
    def test_returns_approval_gate_report(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert isinstance(report, ApprovalGateReport)

    def test_report_has_correct_candidate_id(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.candidate_parameter_set_id == "ps_perfect"

    def test_report_has_correct_baseline_id(self):
        base = _make_ps(ps_id="baseline_v1")
        gate = ApprovalGate(
            _policy(), _perfect_ps(), base, _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.baseline_parameter_set_id == "baseline_v1"

    def test_report_has_holdout_samples_count(self):
        holdout = _make_holdout(n_per_class=5)
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), holdout,
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.holdout_samples == len(holdout)

    def test_report_has_evaluated_at(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.evaluated_at == _NOW.isoformat()

    def test_baseline_and_candidate_metrics_are_holdout_metrics(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert isinstance(report.baseline_metrics, HoldoutMetrics)
        assert isinstance(report.candidate_metrics, HoldoutMetrics)

    def test_regression_guard_result_in_report(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert isinstance(report.regression_guard, RegressionGuardResult)


# ── automatic pass tests ───────────────────────────────────────────────────────

class TestAutomaticPass:
    def test_automatic_pass_when_candidate_better_and_no_human_required(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.status == "automatic_pass"

    def test_passed_property_true_for_automatic_pass(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.passed is True

    def test_no_pending_approval_for_automatic_pass(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.pending_approval is None

    def test_candidate_accuracy_exceeds_baseline(self):
        holdout = _make_holdout()
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), holdout,
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.candidate_metrics.accuracy > report.baseline_metrics.accuracy


# ── rejection tests ────────────────────────────────────────────────────────────

class TestRejection:
    def test_rejects_when_candidate_worse_and_strict_guard(self):
        gate = ApprovalGate(
            _policy(_POLICY_STRICT_GUARD), _bad_ps(), _perfect_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.status == "rejected"

    def test_passed_false_when_rejected(self):
        gate = ApprovalGate(
            _policy(_POLICY_STRICT_GUARD), _bad_ps(), _perfect_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.passed is False

    def test_rejection_reasons_not_empty(self):
        gate = ApprovalGate(
            _policy(_POLICY_STRICT_GUARD), _bad_ps(), _perfect_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert len(report.rejection_reasons) > 0

    def test_regression_guard_passed_false_when_rejected(self):
        gate = ApprovalGate(
            _policy(_POLICY_STRICT_GUARD), _bad_ps(), _perfect_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.regression_guard.passed is False

    def test_no_pending_approval_when_rejected(self):
        gate = ApprovalGate(
            _policy(_POLICY_STRICT_GUARD), _bad_ps(), _perfect_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.pending_approval is None


# ── human approval tests ───────────────────────────────────────────────────────

class TestHumanApproval:
    def test_pending_human_when_guard_passes_and_human_required(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.status == "pending_human"

    def test_passed_true_for_pending_human(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.passed is True

    def test_pending_approval_not_none_when_human_required(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert isinstance(report.pending_approval, PendingApproval)

    def test_pending_approval_has_candidate_id(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.pending_approval.candidate_parameter_set_id == "ps_perfect"

    def test_pending_approval_has_approval_token(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        pa = report.pending_approval
        assert pa.approval_token.startswith("sha256:") or pa.approval_token.startswith("hmac-")

    def test_pending_approval_hmac_token_when_signing_key_provided(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW, signing_key=_SIGNING_KEY,
        )
        report = gate.evaluate()
        assert report.pending_approval.approval_token.startswith("hmac-sha256:")

    def test_pending_approval_status_is_pending_by_default(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.pending_approval.status == "pending"
        assert report.pending_approval.decided_at is None
        assert report.pending_approval.decided_by is None
        assert report.pending_approval.decision_token is None

    def test_approve_pending_approval_sets_decision_fields(self):
        from matrixai.continual import approve_pending_approval

        gate = ApprovalGate(
            _policy(_POLICY_HUMAN), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW, signing_key=_SIGNING_KEY,
        )
        report = gate.evaluate()
        approved = approve_pending_approval(
            report.pending_approval, decided_by="qa", signing_key=_SIGNING_KEY, decided_at=_NOW,
        )
        assert approved.status == "approved"
        assert approved.decided_by == "qa"
        assert approved.decided_at == _NOW.isoformat()
        assert approved.decision_token.startswith("hmac-sha256:")

    def test_pending_approval_expires_at_set_when_timeout_declared(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN_TIMEOUT), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.pending_approval.expires_at is not None

    def test_rejected_even_with_human_approval_when_guard_fails(self):
        gate = ApprovalGate(
            _policy(_POLICY_HUMAN.replace("MUST_IMPROVE_BY 0.0", "MUST_IMPROVE_BY 0.5")),
            _bad_ps(), _perfect_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.status == "rejected"
        assert report.pending_approval is None


# ── holdout metrics quality tests ─────────────────────────────────────────────

class TestHoldoutMetrics:
    def test_perfect_ps_achieves_high_accuracy(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.candidate_metrics.accuracy >= 0.9

    def test_metrics_loss_is_non_negative(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.candidate_metrics.loss >= 0.0
        assert report.baseline_metrics.loss >= 0.0

    def test_per_label_metrics_present_for_classification(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert len(report.candidate_metrics.per_label) == len(_LABELS)
        for lbl in _LABELS:
            assert lbl in report.candidate_metrics.per_label

    def test_holdout_metrics_samples_count_correct(self):
        holdout = _make_holdout(n_per_class=4)
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), holdout,
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.baseline_metrics.samples == len(holdout)
        assert report.candidate_metrics.samples == len(holdout)

    def test_loss_metric_guard(self):
        """With METRIC=loss, a lower candidate loss should pass the gate."""
        gate = ApprovalGate(
            _policy(_POLICY_LOSS_METRIC), _perfect_ps(), _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        # Perfect candidate has lower loss → guard should pass
        assert report.regression_guard.metric == "loss"
        assert report.candidate_metrics.loss < report.baseline_metrics.loss

    def test_empty_holdout_does_not_raise(self):
        gate = ApprovalGate(
            _policy(), _perfect_ps(), _make_ps(), [],
            labels=_LABELS, now=_NOW,
        )
        report = gate.evaluate()
        assert report.holdout_samples == 0


class TestNonP4Rejection:
    """ApprovalGate must reject ParameterSets without W1/b1 with a clear error."""

    def test_raises_for_non_p4_candidate(self):
        """A candidate without W1/b1 must raise ValueError, not return dummy metrics."""
        non_p4 = ParameterSet(
            parameter_set_id="non_p4_candidate",
            model_hash="sha256:x",
            parameter_schema_hash="params_x",
            source="initial",
            parameters={
                "dense.layer_0.W": {
                    "function": "layer_0", "role": "weights", "type": "Tensor[4,4]",
                    "shape": [4, 4], "dtype": "float32", "initializer": "xavier",
                    "values": [[0.1] * 4] * 4,
                },
            },
            metrics={},
        )
        gate = ApprovalGate(
            _policy(), non_p4, _make_ps(), _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        with pytest.raises(ValueError, match="W1/b1"):
            gate.evaluate()

    def test_raises_for_non_p4_baseline(self):
        """A baseline without W1/b1 must also raise ValueError."""
        non_p4 = ParameterSet(
            parameter_set_id="non_p4_baseline",
            model_hash="sha256:y",
            parameter_schema_hash="params_y",
            source="initial",
            parameters={"weights": {"role": "weights", "type": "Tensor[4,4]",
                                    "shape": [4, 4], "values": [[0.0] * 4] * 4}},
            metrics={},
        )
        gate = ApprovalGate(
            _policy(), _make_ps(), non_p4, _make_holdout(),
            labels=_LABELS, now=_NOW,
        )
        with pytest.raises(ValueError, match="W1/b1"):
            gate.evaluate()
