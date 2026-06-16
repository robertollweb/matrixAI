"""P20 C4 — DryRunSimulator + DryRunReport con validación de contrato."""
from datetime import datetime, timedelta, timezone

import pytest

from matrixai.actions import (
    DryRunReport,
    DryRunSimulator,
    RateTracker,
    parse_mxact,
)
from matrixai.actions.schema import RateLimitSpec
from matrixai.parser.parser import parse_text


# ── fixtures ───────────────────────────────────────────────────────────────────

_EMAIL_MXACT = """
ACTION_CONTRACT SendNotification
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com", "alerts@example.com"]
    allowed_domains    = ["example.com"]
    max_subject_length = 200
  END
  DRY_RUN required
  ROLLBACK send_correction_email
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED true
END

ROLLBACK send_correction_email
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    template = "correction"
  END
END
"""

_EMAIL_MXAI = """
PROJECT AlertSystem

VECTOR Alert[2]
  risk_score: Probability
  severity: Score[0, 10]
END

NETWORK AlertClassifier
  INPUT Alert
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=sigmoid
  OUTPUT alert_prob: Probability
END

ACTION SendNotification
  TARGET email_send
  POLICY real_with_audit
  CONDITION alert_prob > 0.9
  INPUT recipient: String, subject: String, body: String
END

GRAPH
  Alert -> AlertClassifier
  AlertClassifier -> SendNotification
END

AUDIT
  EXPLAIN Alert -> AlertClassifier -> SendNotification
END
"""

_VALID_INPUT = {
    "recipient": "ops@example.com",
    "subject": "Critical alert",
    "body": "System anomaly detected",
}

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def contract():
    return parse_mxact(_EMAIL_MXACT)[0]


@pytest.fixture
def program():
    return parse_text(_EMAIL_MXAI)


@pytest.fixture
def simulator():
    return DryRunSimulator()


# ── DryRunReport structure ─────────────────────────────────────────────────────

class TestDryRunReportStructure:
    def test_dry_run_report_has_report_id(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "params-001", "sha256:abc", _VALID_INPUT, now=_T0)
        assert r.report_id.startswith("dry-")
        assert len(r.report_id) > 5

    def test_dry_run_report_has_valid_until_field(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "params-001", "sha256:abc", _VALID_INPUT, now=_T0)
        assert r.valid_until is not None
        # must be parseable ISO datetime
        dt = datetime.fromisoformat(r.valid_until)
        assert dt > _T0

    def test_dry_run_report_has_input_hash(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "params-001", "sha256:abc", _VALID_INPUT, now=_T0)
        assert r.input_hash.startswith("sha256:")

    def test_dry_run_report_has_action_contract_hash(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "params-001", "sha256:abc", _VALID_INPUT, now=_T0)
        assert r.action_contract_hash.startswith("sha256:")

    def test_dry_run_report_ok_when_all_valid(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "params-001", "sha256:abc", _VALID_INPUT, now=_T0)
        assert r.ok is True
        assert r.errors == []


# ── valid_until expiry ─────────────────────────────────────────────────────────

class TestDryRunExpiry:
    def test_dry_run_report_expires_after_five_minutes(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0)
        just_before = _T0 + timedelta(minutes=4, seconds=59)
        just_after = _T0 + timedelta(minutes=5, seconds=1)
        assert not r.is_expired(now=just_before)
        assert r.is_expired(now=just_after)

    def test_dry_run_valid_until_is_five_minutes_after_executed_at(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0)
        executed = datetime.fromisoformat(r.executed_at)
        valid = datetime.fromisoformat(r.valid_until)
        assert valid - executed == timedelta(minutes=5)

    def test_dry_run_validity_minutes_configurable(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0, validity_minutes=30)
        executed = datetime.fromisoformat(r.executed_at)
        valid = datetime.fromisoformat(r.valid_until)
        assert valid - executed == timedelta(minutes=30)

    def test_dry_run_validity_capped_at_one_hour(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0, validity_minutes=999)
        executed = datetime.fromisoformat(r.executed_at)
        valid = datetime.fromisoformat(r.valid_until)
        assert valid - executed == timedelta(hours=1)

    def test_dry_run_report_is_not_expired_before_valid_until(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0)
        assert not r.is_expired(now=_T0)


# ── scope validation ───────────────────────────────────────────────────────────

class TestDryRunScopeValidation:
    def test_dry_run_validates_scope_against_model_output(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0)
        assert r.scope_ok is True

    def test_dry_run_blocks_execution_with_out_of_scope_recipient(self, simulator, contract, program):
        bad_input = {**_VALID_INPUT, "recipient": "evil@attacker.com"}
        r = simulator.simulate(contract, program, "p", "h", bad_input, now=_T0)
        assert r.ok is False
        assert r.scope_ok is False
        assert any("allowed_recipients" in e for e in r.errors)

    def test_dry_run_report_not_ok_when_scope_violated(self, simulator, contract, program):
        bad_input = {**_VALID_INPUT, "recipient": "unknown@other.com"}
        r = simulator.simulate(contract, program, "p", "h", bad_input, now=_T0)
        assert not r.ok


# ── input type validation ──────────────────────────────────────────────────────

class TestDryRunInputTypes:
    def test_dry_run_validates_input_types(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0)
        assert r.input_types_ok is True

    def test_dry_run_blocks_missing_required_input_field(self, simulator, contract, program):
        incomplete = {"recipient": "ops@example.com", "subject": "Alert"}
        # missing "body"
        r = simulator.simulate(contract, program, "p", "h", incomplete, now=_T0)
        assert r.ok is False
        assert r.input_types_ok is False
        assert any("body" in e for e in r.errors)

    def test_dry_run_blocks_wrong_input_type(self, simulator, contract, program):
        wrong_type = {**_VALID_INPUT, "recipient": 12345}  # int instead of String
        r = simulator.simulate(contract, program, "p", "h", wrong_type, now=_T0)
        assert r.ok is False
        assert r.input_types_ok is False


# ── rate limit validation ──────────────────────────────────────────────────────

class TestDryRunRateLimit:
    def test_dry_run_validates_rate_limit(self, simulator, contract, program):
        tracker = RateTracker()
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT,
                               rate_tracker=tracker, now=_T0)
        assert r.rate_limit_ok is True

    def test_dry_run_blocks_execution_when_rate_limit_exceeded(self, simulator, contract, program):
        tracker = RateTracker()
        # per_minute=5, pre-fill 5 calls within last minute
        for i in range(5):
            tracker.record(now=_T0 - timedelta(seconds=30 - i))
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT,
                               rate_tracker=tracker, now=_T0)
        assert r.ok is False
        assert r.rate_limit_ok is False
        assert any("Rate limit" in e for e in r.errors)

    def test_dry_run_rate_tracker_records_call(self):
        tracker = RateTracker()
        limit = RateLimitSpec(per_minute=2, per_hour=10)
        assert not tracker.would_exceed(limit, now=_T0)
        tracker.record(now=_T0)
        tracker.record(now=_T0)
        assert tracker.would_exceed(limit, now=_T0)


# ── rollback validation ────────────────────────────────────────────────────────

class TestDryRunRollback:
    def test_dry_run_validates_rollback_invocability(self, simulator, contract, program):
        r = simulator.simulate(contract, program, "p", "h", _VALID_INPUT, now=_T0)
        assert r.rollback_ok is True
