"""P22 C7 — ContinualVersioner: promote candidate ParameterSet into P21 registry."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from matrixai.continual import (
    ApprovalGate,
    ApprovalGateReport,
    ContinualVersioner,
    ContinualVersioningError,
    ContinualVersioningResult,
    HoldoutMetrics,
    RegressionGuardResult,
    parse_mxcontinual,
)
from matrixai.parameters.store import ParameterSet
from matrixai.registry.entry_hash import compute_entry_hash, sha256_str
from matrixai.registry.model_registry import ModelRegistry
from matrixai import __version__ as _MATRIXAI_VERSION
from matrixai.registry.schema import RegistryEntry
from matrixai.training.data import SupervisedExample


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_SRC = """
CONTINUAL_POLICY VersionerTestPolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json
  REGISTRY_NAME test_classifier
  BASE_VERSION v1.0

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

_POLICY_NO_REGISTRY = _POLICY_SRC.replace(
    "  REGISTRY_NAME test_classifier\n", ""
).replace(
    "  BASE_VERSION v1.0\n", ""
)

_POLICY_SIG_REQUIRED = _POLICY_SRC.replace(
    "    SIGNATURE_REQUIRED false",
    "    SIGNATURE_REQUIRED true",
)
_POLICY_HASH_SIG_REQUIRED = parse_mxcontinual(_POLICY_SIG_REQUIRED).policy_hash

_LABELS = ["spam", "ham", "promo"]
_NOW = datetime(2026, 5, 22, 14, 0, 0, tzinfo=timezone.utc)

# Real policy hash derived from the canonical policy source — must match reports.
_POLICY_HASH = parse_mxcontinual(_POLICY_SRC).policy_hash


def _policy(src: str = _POLICY_SRC) -> Any:
    return parse_mxcontinual(src)


def _make_ps(ps_id: str = "candidate_ps", parent: str = "base_ps") -> ParameterSet:
    W1 = [[5.0, -5.0, -1.0, 0.0], [-5.0, 5.0, -1.0, 0.0], [-1.0, -1.0, 5.0, 0.0]]
    b1 = [0.0] * 3
    return ParameterSet(
        parameter_set_id=ps_id,
        model_hash="sha256:fake_model",
        parameter_schema_hash="params_fake",
        source="incremental_finetune",
        parameters={
            "W1": {"function": "cls", "role": "weights", "type": "Tensor[3,4]",
                   "shape": [3, 4], "dtype": "float32", "initializer": "xavier", "values": W1},
            "b1": {"function": "cls", "role": "bias", "type": "Vector[3]",
                   "shape": [3], "dtype": "float32", "initializer": "zeros", "values": b1},
        },
        metrics={
            "parent_parameter_set_id": parent,
            "validation_loss": 0.21,
            "accuracy": 0.93,
            "dataset_fingerprint": "continual_abc12345",
        },
    )


def _make_passed_report(
    candidate_id: str = "candidate_ps",
    baseline_id: str = "base_ps",
    policy_hash: str | None = None,
) -> ApprovalGateReport:
    """Build a fake passed ApprovalGateReport without running ApprovalGate.

    ``policy_hash`` defaults to the real hash of ``_POLICY_SRC`` so that the
    binding check in ``ContinualVersioner.promote()`` passes.
    """
    holdout_metrics = HoldoutMetrics(
        loss=0.21, accuracy=0.93, macro_f1=0.92,
        macro_precision=0.91, macro_recall=0.92,
        per_label={lbl: {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10.0}
                   for lbl in _LABELS},
        samples=30,
    )
    baseline_metrics = HoldoutMetrics(
        loss=0.28, accuracy=0.88, macro_f1=0.87,
        macro_precision=0.86, macro_recall=0.87,
        per_label={lbl: {"precision": 0.85, "recall": 0.85, "f1": 0.85, "support": 10.0}
                   for lbl in _LABELS},
        samples=30,
    )
    guard = RegressionGuardResult(
        passed=True, metric="accuracy",
        baseline_value=0.88, candidate_value=0.93,
        must_improve_by=0.0, actual_delta=0.05,
    )
    return ApprovalGateReport(
        policy_hash=policy_hash if policy_hash is not None else _POLICY_HASH,
        status="automatic_pass",
        candidate_parameter_set_id=candidate_id,
        baseline_parameter_set_id=baseline_id,
        holdout_samples=30,
        baseline_metrics=baseline_metrics,
        candidate_metrics=holdout_metrics,
        regression_guard=guard,
        pending_approval=None,
        evaluated_at=_NOW.isoformat(),
    )


def _make_failed_report() -> ApprovalGateReport:
    holdout = HoldoutMetrics(loss=1.0, accuracy=0.3, macro_f1=0.3,
                              macro_precision=0.3, macro_recall=0.3, samples=10)
    guard = RegressionGuardResult(
        passed=False, metric="accuracy",
        baseline_value=0.88, candidate_value=0.3,
        must_improve_by=0.0, actual_delta=-0.58,
        reasons=["REGRESSION_GUARD: accuracy delta -0.58 < must_improve_by 0.0"],
    )
    return ApprovalGateReport(
        policy_hash="sha256:" + "a" * 64,
        status="rejected",
        candidate_parameter_set_id="bad_ps",
        baseline_parameter_set_id="base_ps",
        holdout_samples=10,
        baseline_metrics=holdout,
        candidate_metrics=holdout,
        regression_guard=guard,
        pending_approval=None,
        evaluated_at=_NOW.isoformat(),
        rejection_reasons=["guard failed"],
    )


def _make_base_entry(name: str = "test_classifier", version: str = "v1.0") -> RegistryEntry:
    """Build a minimal base entry for the registry."""
    eval_hash = sha256_str('{"base_eval":"true"}')
    eh = compute_entry_hash(
        name=name, version=version,
        model_hash="sha256:fake_model",
        parameter_schema_hash="params_fake",
        parameter_set_id="base_ps",
        training_trace_hash="",
        evaluation_report_hash=eval_hash,
        matrixai_version=_MATRIXAI_VERSION,
    )
    return RegistryEntry(
        name=name, version=version, entry_hash=eh,
        model_hash="sha256:fake_model",
        parameter_schema_hash="params_fake",
        parameter_set_id="base_ps",
        input_type={}, output_type={},
        metrics={"accuracy": 0.88},
        matrixai_version=_MATRIXAI_VERSION,
        created_at=_NOW.isoformat(),
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash="",
        evaluation_report_hash=eval_hash,
    )


def _registry_with_base(tmp_path: Path) -> ModelRegistry:
    """Create a fresh registry with a v1.0 base entry."""
    registry = ModelRegistry(tmp_path / "registry")
    registry.push(_make_base_entry())
    return registry


# ── basic promotion tests ──────────────────────────────────────────────────────

class TestPromoteBasics:
    def test_returns_versioning_result(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        result = v.promote()
        assert isinstance(result, ContinualVersioningResult)

    def test_new_version_is_v1_1(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        result = v.promote()
        assert result.new_version == "v1.1"

    def test_previous_version_is_base_version(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        result = v.promote()
        assert result.previous_version == "v1.0"

    def test_registry_name_in_result(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        result = v.promote()
        assert result.registry_name == "test_classifier"

    def test_entry_hash_is_non_empty(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        result = v.promote()
        assert result.entry_hash.startswith("sha256:")

    def test_pushed_at_matches_now(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        result = v.promote()
        assert result.pushed_at == _NOW.isoformat()

    def test_parent_parameter_set_id_in_result(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ps = _make_ps(ps_id="cand_001", parent="base_ps_original")
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(candidate_id="cand_001"), ps, now=_NOW,
        )
        result = v.promote()
        assert result.parent_parameter_set_id == "base_ps_original"

    def test_candidate_parameter_set_id_in_result(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ps = _make_ps(ps_id="my_candidate_v1")
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(candidate_id="my_candidate_v1"), ps, now=_NOW,
        )
        result = v.promote()
        assert result.candidate_parameter_set_id == "my_candidate_v1"


# ── registry state tests ───────────────────────────────────────────────────────

class TestRegistryState:
    def test_new_version_retrievable_from_registry(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        v.promote()
        entry = reg.get("test_classifier", "v1.1")
        assert entry.version == "v1.1"

    def test_current_tag_updated_to_new_version(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        v.promote()
        # "current" tag should now resolve to v1.1
        entry = reg.get("test_classifier", "current")
        assert entry.version == "v1.1"

    def test_model_hash_preserved_in_entry(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), ps, now=_NOW,
        )
        v.promote()
        entry = reg.get("test_classifier", "v1.1")
        assert entry.model_hash == ps.model_hash

    def test_parameter_schema_hash_preserved(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), ps, now=_NOW,
        )
        v.promote()
        entry = reg.get("test_classifier", "v1.1")
        assert entry.parameter_schema_hash == ps.parameter_schema_hash

    def test_entry_records_product_matrixai_version(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        v.promote()
        entry = reg.get("test_classifier", "v1.1")
        assert entry.matrixai_version == _MATRIXAI_VERSION

    def test_entry_metrics_include_parent_ps_id(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ps = _make_ps(parent="original_base_ps")
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), ps, now=_NOW,
        )
        v.promote()
        entry = reg.get("test_classifier", "v1.1")
        assert entry.metrics["parent_parameter_set_id"] == "original_base_ps"

    def test_entry_metrics_include_continual_update_id(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(),
            continual_update_id="cu-test-001", now=_NOW,
        )
        v.promote()
        entry = reg.get("test_classifier", "v1.1")
        assert entry.metrics["continual_update_id"] == "cu-test-001"

    def test_params_json_stored_in_entry_dir(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), ps, now=_NOW,
        )
        v.promote()
        params_file = reg.layout.entry_dir("test_classifier", "v1.1") / "params.json"
        assert params_file.exists()
        stored = json.loads(params_file.read_text())
        assert stored["parameter_set_id"] == ps.parameter_set_id

    def test_approval_gate_report_stored_in_entry_dir(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        v.promote()
        report_file = reg.layout.entry_dir("test_classifier", "v1.1") / "approval_gate_report.json"
        assert report_file.exists()


# ── sequential promotion tests ─────────────────────────────────────────────────

class TestSequentialPromotions:
    def test_second_promotion_gives_v1_2(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ContinualVersioner(
            _policy(), reg, _make_passed_report(candidate_id="ps_v1"), _make_ps(ps_id="ps_v1"), now=_NOW,
        ).promote()
        result2 = ContinualVersioner(
            _policy(), reg, _make_passed_report(candidate_id="ps_v2"), _make_ps(ps_id="ps_v2"), now=_NOW,
        ).promote()
        assert result2.new_version == "v1.2"

    def test_current_tag_follows_latest_promotion(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ContinualVersioner(
            _policy(), reg, _make_passed_report(candidate_id="ps_v1"), _make_ps(ps_id="ps_v1"), now=_NOW,
        ).promote()
        ContinualVersioner(
            _policy(), reg, _make_passed_report(candidate_id="ps_v2"), _make_ps(ps_id="ps_v2"), now=_NOW,
        ).promote()
        entry = reg.get("test_classifier", "current")
        assert entry.version == "v1.2"

    def test_base_version_untouched_after_promotion(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        ).promote()
        # v1.0 must still be accessible
        entry = reg.get("test_classifier", "v1.0")
        assert entry.parameter_set_id == "base_ps"


# ── error handling tests ───────────────────────────────────────────────────────

class TestErrors:
    def test_raises_on_failed_approval_report(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_failed_report(), _make_ps(), now=_NOW,
        )
        with pytest.raises(ContinualVersioningError, match="approval gate must pass"):
            v.promote()

    def test_raises_when_no_registry_name_in_policy(self, tmp_path):
        reg = ModelRegistry(tmp_path / "reg2")
        no_reg_policy = _policy(_POLICY_NO_REGISTRY)
        v = ContinualVersioner(
            no_reg_policy, reg,
            _make_passed_report(policy_hash=no_reg_policy.policy_hash), _make_ps(), now=_NOW,
        )
        with pytest.raises(ContinualVersioningError, match="REGISTRY_NAME"):
            v.promote()

    def test_continual_update_id_auto_generated(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(), now=_NOW,
        )
        result = v.promote()
        assert result.continual_update_id.startswith("cu-")

    def test_custom_update_id_preserved(self, tmp_path):
        reg = _registry_with_base(tmp_path)
        v = ContinualVersioner(
            _policy(), reg, _make_passed_report(), _make_ps(),
            continual_update_id="cu-custom-xyz", now=_NOW,
        )
        result = v.promote()
        assert result.continual_update_id == "cu-custom-xyz"


# ── security binding tests ─────────────────────────────────────────────────────

class TestSecurityBinding:
    """Verify cryptographic binding between report, candidate, and policy."""

    def test_raises_when_report_candidate_mismatch(self, tmp_path):
        """Report for a different candidate must not allow promotion."""
        reg = _registry_with_base(tmp_path)
        ps = _make_ps(ps_id="real_candidate")
        # Report was for a different candidate
        report = _make_passed_report(candidate_id="other_candidate")
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        with pytest.raises(ContinualVersioningError, match="candidate_parameter_set_id"):
            v.promote()

    def test_raises_when_report_policy_hash_mismatch(self, tmp_path):
        """Report generated under a different policy must not allow promotion."""
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        # Report has a wrong policy hash (different policy)
        report = _make_passed_report(policy_hash="sha256:" + "b" * 64)
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        with pytest.raises(ContinualVersioningError, match="policy_hash"):
            v.promote()

    def test_raises_on_pending_human_without_explicit_acknowledgement(self, tmp_path):
        """pending_human status must not promote without human_approved=True."""
        from matrixai.continual import PendingApproval
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        pending = PendingApproval(
            approval_id="apr-xyz",
            policy_hash=_POLICY_HASH,
            candidate_parameter_set_id=ps.parameter_set_id,
            parent_parameter_set_id="base_ps",
            created_at=_NOW.isoformat(),
            expires_at=None,
            approval_token="sha256:abc",
            channel=None,
        )
        guard = RegressionGuardResult(
            passed=True, metric="accuracy",
            baseline_value=0.88, candidate_value=0.93,
            must_improve_by=0.0, actual_delta=0.05,
        )
        m = HoldoutMetrics(loss=0.2, accuracy=0.93, macro_f1=0.92,
                           macro_precision=0.91, macro_recall=0.92, samples=10)
        report = ApprovalGateReport(
            policy_hash=_POLICY_HASH,
            status="pending_human",
            candidate_parameter_set_id=ps.parameter_set_id,
            baseline_parameter_set_id="base_ps",
            holdout_samples=10,
            baseline_metrics=m, candidate_metrics=m,
            regression_guard=guard,
            pending_approval=pending,
            evaluated_at=_NOW.isoformat(),
        )
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        with pytest.raises(ContinualVersioningError, match="HUMAN_APPROVAL"):
            v.promote()

    def test_raises_when_acknowledged_but_pending_not_approved(self, tmp_path):
        """human_approved=True is not enough without an approved PendingApproval state."""
        from matrixai.continual import PendingApproval
        from matrixai.continual.approval import _make_approval_token

        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        approval_token = _make_approval_token(
            _POLICY_HASH, ps.parameter_set_id, _NOW.isoformat(), None, expires_at=None,
        )
        pending = PendingApproval(
            approval_id="apr-xyz",
            policy_hash=_POLICY_HASH,
            candidate_parameter_set_id=ps.parameter_set_id,
            parent_parameter_set_id="base_ps",
            created_at=_NOW.isoformat(),
            expires_at=None,
            approval_token=approval_token,
            channel=None,
        )
        guard = RegressionGuardResult(
            passed=True, metric="accuracy",
            baseline_value=0.88, candidate_value=0.93,
            must_improve_by=0.0, actual_delta=0.05,
        )
        m = HoldoutMetrics(loss=0.2, accuracy=0.93, macro_f1=0.92,
                           macro_precision=0.91, macro_recall=0.92, samples=10)
        report = ApprovalGateReport(
            policy_hash=_POLICY_HASH,
            status="pending_human",
            candidate_parameter_set_id=ps.parameter_set_id,
            baseline_parameter_set_id="base_ps",
            holdout_samples=10,
            baseline_metrics=m, candidate_metrics=m,
            regression_guard=guard,
            pending_approval=pending,
            evaluated_at=_NOW.isoformat(),
        )
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        with pytest.raises(ContinualVersioningError, match="status"):
            v.promote(human_approved=True)

    def test_promotes_with_pending_human_when_acknowledged(self, tmp_path):
        """human_approved=True must allow promotion of pending_human reports."""
        from matrixai.continual import PendingApproval, approve_pending_approval
        from matrixai.continual.approval import _make_approval_token
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        approval_token = _make_approval_token(
            _POLICY_HASH, ps.parameter_set_id, _NOW.isoformat(), None, expires_at=None,
        )
        pending = PendingApproval(
            approval_id="apr-xyz",
            policy_hash=_POLICY_HASH,
            candidate_parameter_set_id=ps.parameter_set_id,
            parent_parameter_set_id="base_ps",
            created_at=_NOW.isoformat(),
            expires_at=None,
            approval_token=approval_token,
            channel=None,
        )
        pending = approve_pending_approval(pending, decided_by="qa", decided_at=_NOW)
        guard = RegressionGuardResult(
            passed=True, metric="accuracy",
            baseline_value=0.88, candidate_value=0.93,
            must_improve_by=0.0, actual_delta=0.05,
        )
        m = HoldoutMetrics(loss=0.2, accuracy=0.93, macro_f1=0.92,
                           macro_precision=0.91, macro_recall=0.92, samples=10)
        report = ApprovalGateReport(
            policy_hash=_POLICY_HASH,
            status="pending_human",
            candidate_parameter_set_id=ps.parameter_set_id,
            baseline_parameter_set_id="base_ps",
            holdout_samples=10,
            baseline_metrics=m, candidate_metrics=m,
            regression_guard=guard,
            pending_approval=pending,
            evaluated_at=_NOW.isoformat(),
        )
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        result = v.promote(human_approved=True)
        assert result.new_version == "v1.1"


class TestPendingApprovalExpiry:
    """Verify that expired PendingApproval tokens are rejected."""

    def _make_pending_report(self, expires_at: str | None) -> ApprovalGateReport:
        from matrixai.continual import PendingApproval
        from matrixai.continual import approve_pending_approval
        from matrixai.continual.approval import _make_approval_token
        approval_token = _make_approval_token(
            _POLICY_HASH, "candidate_ps", _NOW.isoformat(), None, expires_at=expires_at,
        )
        pending = PendingApproval(
            approval_id="apr-expiry-test",
            policy_hash=_POLICY_HASH,
            candidate_parameter_set_id="candidate_ps",
            parent_parameter_set_id="base_ps",
            created_at=_NOW.isoformat(),
            expires_at=expires_at,
            approval_token=approval_token,
            channel=None,
        )
        pending = approve_pending_approval(pending, decided_by="qa", decided_at=_NOW)
        guard = RegressionGuardResult(
            passed=True, metric="accuracy",
            baseline_value=0.88, candidate_value=0.93,
            must_improve_by=0.0, actual_delta=0.05,
        )
        m = HoldoutMetrics(loss=0.2, accuracy=0.93, macro_f1=0.92,
                           macro_precision=0.91, macro_recall=0.92, samples=10)
        return ApprovalGateReport(
            policy_hash=_POLICY_HASH,
            status="pending_human",
            candidate_parameter_set_id="candidate_ps",
            baseline_parameter_set_id="base_ps",
            holdout_samples=10,
            baseline_metrics=m, candidate_metrics=m,
            regression_guard=guard,
            pending_approval=pending,
            evaluated_at=_NOW.isoformat(),
        )

    def test_raises_when_token_expired(self, tmp_path):
        """Token expired before NOW must raise ContinualVersioningError."""
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        # expires_at is 1 hour BEFORE _NOW
        from datetime import timedelta
        expired_at = (_NOW - timedelta(hours=1)).isoformat()
        report = self._make_pending_report(expires_at=expired_at)
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        with pytest.raises(ContinualVersioningError, match="expired"):
            v.promote(human_approved=True)

    def test_allows_when_token_not_yet_expired(self, tmp_path):
        """Token with future expires_at must allow promotion."""
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        from datetime import timedelta
        future_at = (_NOW + timedelta(hours=24)).isoformat()
        report = self._make_pending_report(expires_at=future_at)
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        result = v.promote(human_approved=True)
        assert result.new_version == "v1.1"

    def test_allows_when_expires_at_is_none(self, tmp_path):
        """No expiry date means token never expires — promotion must succeed."""
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        report = self._make_pending_report(expires_at=None)
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        result = v.promote(human_approved=True)
        assert result.new_version == "v1.1"


class TestPendingApprovalTokenIntegrity:
    """Verify that expires_at is covered by the HMAC token and malformed values fail closed."""

    def _make_pending_report_with_expiry(self, expires_at: str | None) -> ApprovalGateReport:
        from matrixai.continual import PendingApproval
        from matrixai.continual.approval import _make_approval_token
        token = _make_approval_token(
            _POLICY_HASH, "candidate_ps", _NOW.isoformat(), None, expires_at=expires_at,
        )
        from matrixai.continual import approve_pending_approval
        pending = PendingApproval(
            approval_id="apr-integrity",
            policy_hash=_POLICY_HASH,
            candidate_parameter_set_id="candidate_ps",
            parent_parameter_set_id="base_ps",
            created_at=_NOW.isoformat(),
            expires_at=expires_at,
            approval_token=token,
            channel=None,
        )
        pending = approve_pending_approval(pending, decided_by="qa", decided_at=_NOW)
        guard = RegressionGuardResult(
            passed=True, metric="accuracy",
            baseline_value=0.88, candidate_value=0.93,
            must_improve_by=0.0, actual_delta=0.05,
        )
        m = HoldoutMetrics(loss=0.2, accuracy=0.93, macro_f1=0.92,
                           macro_precision=0.91, macro_recall=0.92, samples=10)
        return ApprovalGateReport(
            policy_hash=_POLICY_HASH,
            status="pending_human",
            candidate_parameter_set_id="candidate_ps",
            baseline_parameter_set_id="base_ps",
            holdout_samples=10,
            baseline_metrics=m, candidate_metrics=m,
            regression_guard=guard,
            pending_approval=pending,
            evaluated_at=_NOW.isoformat(),
        )

    def test_malformed_expires_at_fails_closed(self, tmp_path):
        """Unparseable expires_at must raise ContinualVersioningError, not allow through."""
        from datetime import timedelta
        reg = _registry_with_base(tmp_path)
        ps = _make_ps()
        report = self._make_pending_report_with_expiry(expires_at="not-a-date")
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        with pytest.raises(ContinualVersioningError, match="ISO-8601"):
            v.promote(human_approved=True)

    def test_tampered_expires_at_invalidates_token(self, tmp_path):
        """Changing expires_at after token generation must cause CLI HMAC mismatch."""
        from matrixai.continual import PendingApproval
        from matrixai.continual.approval import _make_approval_token
        signing_key = "a" * 64
        # Token signed with original expires_at
        original_expires = _NOW.isoformat()
        token = _make_approval_token(
            _POLICY_HASH, "candidate_ps", _NOW.isoformat(), signing_key,
            expires_at=original_expires,
        )
        # Attacker extends expiry in JSON — token now covers original_expires, not extended_expires
        from datetime import timedelta
        extended_expires = (_NOW + timedelta(days=30)).isoformat()
        assert token != _make_approval_token(
            _POLICY_HASH, "candidate_ps", _NOW.isoformat(), signing_key,
            expires_at=extended_expires,
        )


class TestSignatureRequired:
    """Verify that SIGNATURE_REQUIRED true forces HMAC-signed tokens in promote()."""

    def _make_approved_pending(self, policy_hash: str, signing_key: str | None) -> Any:
        """Build a fully approved PendingApproval signed consistently with signing_key."""
        from matrixai.continual import PendingApproval, approve_pending_approval
        from matrixai.continual.approval import _make_approval_token
        token = _make_approval_token(
            policy_hash, "candidate_ps", _NOW.isoformat(), signing_key, expires_at=None,
        )
        pending = PendingApproval(
            approval_id="apr-sigtest",
            policy_hash=policy_hash,
            candidate_parameter_set_id="candidate_ps",
            parent_parameter_set_id="base_ps",
            created_at=_NOW.isoformat(),
            expires_at=None,
            approval_token=token,
            channel=None,
        )
        # Pass signing_key so decision_token is also HMAC-signed when required
        return approve_pending_approval(pending, decided_by="qa", decided_at=_NOW, signing_key=signing_key)

    def _make_pending_report(self, pending: Any, policy_hash: str) -> ApprovalGateReport:
        guard = RegressionGuardResult(
            passed=True, metric="accuracy",
            baseline_value=0.88, candidate_value=0.93,
            must_improve_by=0.0, actual_delta=0.05,
        )
        m = HoldoutMetrics(loss=0.2, accuracy=0.93, macro_f1=0.92,
                           macro_precision=0.91, macro_recall=0.92, samples=10)
        return ApprovalGateReport(
            policy_hash=policy_hash,
            status="pending_human",
            candidate_parameter_set_id="candidate_ps",
            baseline_parameter_set_id="base_ps",
            holdout_samples=10,
            baseline_metrics=m, candidate_metrics=m,
            regression_guard=guard,
            pending_approval=pending,
            evaluated_at=_NOW.isoformat(),
        )

    def test_raises_on_unsigned_token_when_signature_required(self, tmp_path):
        """SIGNATURE_REQUIRED true + unsigned sha256: token must raise, not promote."""
        reg = _registry_with_base(tmp_path)
        sig_policy = parse_mxcontinual(_POLICY_SIG_REQUIRED)
        ps = _make_ps(ps_id="candidate_ps")
        # Build unsigned approval (signing_key=None → sha256: prefix)
        pending = self._make_approved_pending(_POLICY_HASH_SIG_REQUIRED, signing_key=None)
        report = self._make_pending_report(pending, _POLICY_HASH_SIG_REQUIRED)
        v = ContinualVersioner(sig_policy, reg, report, ps, now=_NOW)
        with pytest.raises(ContinualVersioningError, match="SIGNATURE_REQUIRED"):
            v.promote(human_approved=True)

    def test_allows_hmac_signed_token_when_signature_required(self, tmp_path):
        """SIGNATURE_REQUIRED true + HMAC-signed token + matching key must succeed."""
        reg = _registry_with_base(tmp_path)
        sig_policy = parse_mxcontinual(_POLICY_SIG_REQUIRED)
        ps = _make_ps(ps_id="candidate_ps")
        signing_key = "b" * 64
        pending = self._make_approved_pending(_POLICY_HASH_SIG_REQUIRED, signing_key=signing_key)
        report = self._make_pending_report(pending, _POLICY_HASH_SIG_REQUIRED)
        v = ContinualVersioner(
            sig_policy, reg, report, ps,
            now=_NOW, approval_signing_key=signing_key,
        )
        result = v.promote(human_approved=True)
        assert result.new_version == "v1.1"

    def test_unsigned_token_still_works_without_signature_required(self, tmp_path):
        """SIGNATURE_REQUIRED false (default) must still allow sha256: tokens."""
        reg = _registry_with_base(tmp_path)
        ps = _make_ps(ps_id="candidate_ps")
        pending = self._make_approved_pending(_POLICY_HASH, signing_key=None)
        report = self._make_pending_report(pending, _POLICY_HASH)
        v = ContinualVersioner(_policy(), reg, report, ps, now=_NOW)
        result = v.promote(human_approved=True)
        assert result.new_version == "v1.1"
