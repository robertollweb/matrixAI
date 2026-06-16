"""P22 C10 — DriftRefinementBridge + RefinementAgent drift_driven mode."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from matrixai.agents.refinement import RefinementAgent, RefinementProposal
from matrixai.continual import DriftRefinementBridge, DriftReport, FeatureDriftResult, parse_mxcontinual


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_EMIT_ON = """
CONTINUAL_POLICY DriftBridgePolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
    SOURCES [api]
  END

  DRIFT_DETECTION
    FEATURES [score, length]
    METHODS
      score: ks threshold=0.15
      length: psi threshold=0.20
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
    EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT true
    REFINEMENT_DRIFT_PERSISTENCE_DAYS 14
    SIGNATURE_REQUIRED false
  END
END
"""

_POLICY_EMIT_OFF = _POLICY_EMIT_ON.replace(
    "EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT true",
    "EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT false",
)

_NOW_ISO = "2026-05-22T15:00:00+00:00"
_PROMPT = "Classify the email as spam or ham."


def _policy_on():
    return parse_mxcontinual(_POLICY_EMIT_ON)


def _policy_off():
    return parse_mxcontinual(_POLICY_EMIT_OFF)


def _drift_result(*, drifted: bool, method: str = "ks", observed: float = 0.3, threshold: float = 0.15) -> FeatureDriftResult:
    return FeatureDriftResult(
        feature="score",
        method=method,
        observed_value=observed,
        threshold=threshold,
        drift_detected=drifted,
        samples_used=100,
        enough_samples=True,
    )


def _drift_report(*, drift_detected: bool = True) -> DriftReport:
    result = _drift_result(drifted=drift_detected)
    return DriftReport(
        policy_hash="sha256:abc123",
        checked_at=_NOW_ISO,
        features_checked=["score"],
        results={"score": result},
        drift_detected=drift_detected,
        enough_samples=True,
        total_production_samples=100,
    )


def _drift_report_multi() -> DriftReport:
    """Two features, both drifted."""
    r_score = FeatureDriftResult(
        feature="score", method="ks", observed_value=0.35, threshold=0.15,
        drift_detected=True, samples_used=100, enough_samples=True,
    )
    r_length = FeatureDriftResult(
        feature="length", method="psi", observed_value=0.25, threshold=0.20,
        drift_detected=True, samples_used=100, enough_samples=True,
    )
    return DriftReport(
        policy_hash="sha256:abc123",
        checked_at=_NOW_ISO,
        features_checked=["score", "length"],
        results={"score": r_score, "length": r_length},
        drift_detected=True,
        enough_samples=True,
        total_production_samples=100,
    )


def _drift_report_skipped() -> DriftReport:
    """Feature with skipped=True should not produce a hint."""
    r = FeatureDriftResult(
        feature="score", method="none", observed_value=0.0, threshold=0.0,
        drift_detected=False, samples_used=0, enough_samples=False,
        skipped=True, skip_reason="no method declared",
    )
    return DriftReport(
        policy_hash="sha256:abc123",
        checked_at=_NOW_ISO,
        features_checked=["score"],
        results={"score": r},
        drift_detected=False,
        enough_samples=False,
        total_production_samples=0,
    )


# ── RefinementAgent drift_driven mode tests ───────────────────────────────────

class TestRefinementAgentDriftDriven:
    def test_drift_driven_mode_accepted(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert isinstance(proposal, RefinementProposal)

    def test_drift_driven_mode_field_set(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert proposal.mode == "drift_driven"

    def test_drift_driven_requires_drift_report(self):
        agent = RefinementAgent()
        with pytest.raises(ValueError, match="drift_driven mode requires a drift report"):
            agent.refine(_PROMPT, mode="drift_driven")

    def test_unknown_mode_rejected(self):
        agent = RefinementAgent()
        with pytest.raises(ValueError, match="Unknown refinement mode"):
            agent.refine(_PROMPT, mode="drift_driven_x", drift_report={})

    def test_hints_applied_contains_drift_hints(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report(drift_detected=True))
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert len(proposal.hints_applied) > 0
        assert any("drift" in h.lower() or "score" in h.lower() for h in proposal.hints_applied)

    def test_hints_applied_references_feature_name(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert any("score" in h for h in proposal.hints_applied)

    def test_hints_applied_references_method(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert any("ks" in h for h in proposal.hints_applied)

    def test_multiple_drifted_features_generate_multiple_hints(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report_multi())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        # At least one hint per drifted feature
        assert len(proposal.hints_applied) >= 2

    def test_skipped_features_do_not_generate_hints(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report_skipped())
        # No drift detected, no features drifted → fallback hint
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert len(proposal.hints_applied) == 1
        assert "no indica" in proposal.hints_applied[0].lower() or "revisa" in proposal.hints_applied[0].lower()

    def test_extra_hints_appended_after_derived(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(
            _PROMPT, mode="drift_driven", drift_report=report_dict,
            hints=["custom hint for testing"],
        )
        assert any("custom hint" in h for h in proposal.hints_applied)

    def test_proposed_prompt_contains_feedback_section(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert "<SystemFeedback>" in proposal.proposed_prompt

    def test_explanation_mentions_drift(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert "drift" in proposal.explanation.lower()

    def test_refinement_id_contains_drift_driven(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert "drift" in proposal.refinement_id

    def test_refinement_chain_populated(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert len(proposal.refinement_chain) == 1

    def test_iteration_count_stored(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict, iteration_count=2)
        assert proposal.iteration_count == 2

    def test_original_prompt_preserved(self):
        agent = RefinementAgent()
        report_dict = dataclasses.asdict(_drift_report())
        proposal = agent.refine(_PROMPT, mode="drift_driven", drift_report=report_dict)
        assert proposal.original_prompt == _PROMPT


# ── DriftRefinementBridge tests ────────────────────────────────────────────────

class TestDriftRefinementBridge:
    def test_returns_proposal_when_emit_on_and_drift_detected(self):
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(_drift_report(drift_detected=True), drift_persistence_days=14)
        assert isinstance(proposal, RefinementProposal)

    def test_returns_none_when_emit_off(self):
        bridge = DriftRefinementBridge(_policy_off(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(_drift_report(drift_detected=True), drift_persistence_days=14)
        assert proposal is None

    def test_returns_none_when_no_drift_detected(self):
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(_drift_report(drift_detected=False), drift_persistence_days=14)
        assert proposal is None

    def test_returns_none_when_emit_off_even_with_drift(self):
        bridge = DriftRefinementBridge(_policy_off(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(_drift_report(drift_detected=True), drift_persistence_days=14)
        assert proposal is None

    def test_proposal_mode_is_drift_driven(self):
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(_drift_report(drift_detected=True), drift_persistence_days=14)
        assert proposal is not None
        assert proposal.mode == "drift_driven"

    def test_proposal_original_prompt_matches(self):
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(_drift_report(drift_detected=True), drift_persistence_days=14)
        assert proposal is not None
        assert proposal.original_prompt == _PROMPT

    def test_extra_hints_forwarded_to_agent(self):
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(
            _drift_report(drift_detected=True),
            drift_persistence_days=14,
            hints=["extra hint from bridge"],
        )
        assert proposal is not None
        assert any("extra hint" in h for h in proposal.hints_applied)

    def test_multi_feature_drift_report_handled(self):
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(_drift_report_multi(), drift_persistence_days=14)
        assert proposal is not None
        score_hints = [h for h in proposal.hints_applied if "score" in h]
        length_hints = [h for h in proposal.hints_applied if "length" in h]
        assert score_hints
        assert length_hints


class TestDriftPersistence:
    """Verify that REFINEMENT_DRIFT_PERSISTENCE_DAYS gates the bridge correctly."""

    def test_persistence_check_blocks_when_too_few_days(self):
        """When drift has not persisted long enough, maybe_refine returns None."""
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        # Policy requires 14 days; only 7 have elapsed
        proposal = bridge.maybe_refine(
            _drift_report(drift_detected=True),
            drift_persistence_days=7,
        )
        assert proposal is None

    def test_persistence_check_allows_when_enough_days(self):
        """When drift has persisted >= required days, a proposal is returned."""
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        # Policy requires 14 days; exactly 14 have elapsed
        proposal = bridge.maybe_refine(
            _drift_report(drift_detected=True),
            drift_persistence_days=14,
        )
        assert isinstance(proposal, RefinementProposal)

    def test_persistence_check_allows_when_more_than_required(self):
        """When drift has persisted well beyond the threshold, proposal is returned."""
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(
            _drift_report(drift_detected=True),
            drift_persistence_days=30,
        )
        assert isinstance(proposal, RefinementProposal)

    def test_persistence_omitted_treated_as_zero(self):
        """When drift_persistence_days is omitted and required > 0, None is treated as 0 → blocks."""
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        # required=14, None → treated as 0 → conservatively blocks
        proposal = bridge.maybe_refine(_drift_report(drift_detected=True))
        assert proposal is None

    def test_persistence_blocks_when_one_day_short(self):
        """Boundary: 13 days when 14 required must still block."""
        bridge = DriftRefinementBridge(_policy_on(), prompt=_PROMPT)
        proposal = bridge.maybe_refine(
            _drift_report(drift_detected=True),
            drift_persistence_days=13,
        )
        assert proposal is None
