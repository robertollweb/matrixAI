"""P20 C2 — ActionContract schema, validación y hash determinista."""
import copy
import os
import pytest

from matrixai.actions import (
    ActionContractValidationResult,
    canonical_dict,
    check_signing_key_available,
    compute_action_contract_hash,
    parse_mxact,
    require_signing_key,
    validate_action_contract,
    CAPABILITIES,
)
from matrixai.parser.parser import parse_text


# ── shared fixtures ────────────────────────────────────────────────────────────

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

_SIMULATE_MXAI = """
PROJECT Demo

VECTOR Input[1]
  x: Probability
END

ACTION Respond
  WHEN x > 0.5
  CALL simulated.respond
END

GRAPH
  Input -> Respond
END
"""


@pytest.fixture
def email_contract():
    return parse_mxact(_EMAIL_MXACT)[0]


@pytest.fixture
def email_program():
    return parse_text(_EMAIL_MXAI)


@pytest.fixture
def simulate_program():
    return parse_text(_SIMULATE_MXAI)


# ── hash determinism ───────────────────────────────────────────────────────────

class TestActionContractHash:
    def test_action_contract_hash_is_deterministic(self, email_contract):
        h1 = compute_action_contract_hash(email_contract)
        h2 = compute_action_contract_hash(email_contract)
        assert h1 == h2

    def test_action_contract_hash_has_sha256_prefix(self, email_contract):
        h = compute_action_contract_hash(email_contract)
        assert h.startswith("sha256:")

    def test_action_contract_hash_is_hex_string(self, email_contract):
        h = compute_action_contract_hash(email_contract)
        hex_part = h.removeprefix("sha256:")
        assert len(hex_part) == 64
        int(hex_part, 16)  # raises if not valid hex

    def test_action_contract_hash_changes_when_scope_changes(self, email_contract):
        original = compute_action_contract_hash(email_contract)
        # rebuild with modified scope
        from matrixai.actions.schema import ActionContractSpec
        modified_scope = dict(email_contract.scope)
        modified_scope["max_subject_length"] = 999
        modified = ActionContractSpec(
            name=email_contract.name,
            capability=email_contract.capability,
            scope=modified_scope,
            dry_run_required=email_contract.dry_run_required,
            rollback=email_contract.rollback,
            sandbox_required=email_contract.sandbox_required,
            sandbox_limits=email_contract.sandbox_limits,
            human_approval=email_contract.human_approval,
            approval_channel=email_contract.approval_channel,
            approval_timeout_seconds=email_contract.approval_timeout_seconds,
            rate_limit=email_contract.rate_limit,
            signature_required=email_contract.signature_required,
        )
        assert compute_action_contract_hash(modified) != original

    def test_action_contract_hash_changes_when_rollback_changes(self, email_contract):
        original = compute_action_contract_hash(email_contract)
        from matrixai.actions.schema import ActionContractSpec, RollbackSpec
        new_rb = RollbackSpec(
            name="different_rollback",
            capability="email_send",
            scope={"template": "other"},
        )
        modified = ActionContractSpec(
            name=email_contract.name,
            capability=email_contract.capability,
            scope=email_contract.scope,
            dry_run_required=email_contract.dry_run_required,
            rollback=new_rb,
            sandbox_required=email_contract.sandbox_required,
            sandbox_limits=email_contract.sandbox_limits,
            human_approval=email_contract.human_approval,
            approval_channel=email_contract.approval_channel,
            approval_timeout_seconds=email_contract.approval_timeout_seconds,
            rate_limit=email_contract.rate_limit,
            signature_required=email_contract.signature_required,
        )
        assert compute_action_contract_hash(modified) != original

    def test_action_contract_hash_changes_when_capability_changes(self):
        contracts_a = parse_mxact(_EMAIL_MXACT)
        src_b = _EMAIL_MXACT.replace("email_send", "notification").replace(
            "ROLLBACK send_correction_email\n", ""
        ).replace("  ROLLBACK send_correction_email\n", "")
        # notification does not require rollback — build a valid one
        src_notification = """
ACTION_CONTRACT SendNotification
  CAPABILITY notification
  SCOPE
    allowed_recipients = ["ops@example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED true
END
"""
        contracts_b = parse_mxact(src_notification)
        assert compute_action_contract_hash(contracts_a[0]) != compute_action_contract_hash(contracts_b[0])


# ── canonical dict ─────────────────────────────────────────────────────────────

class TestCanonicalDict:
    def test_canonical_dict_includes_risk_level(self, email_contract):
        d = canonical_dict(email_contract)
        assert d["risk_level"] == CAPABILITIES["email_send"]

    def test_canonical_dict_scope_keys_sorted(self, email_contract):
        d = canonical_dict(email_contract)
        keys = list(d["scope"].keys())
        assert keys == sorted(keys)

    def test_canonical_dict_includes_rollback(self, email_contract):
        d = canonical_dict(email_contract)
        assert d["rollback"] is not None
        assert d["rollback"]["name"] == "send_correction_email"

    def test_canonical_dict_rollback_scope_sorted(self, email_contract):
        d = canonical_dict(email_contract)
        rb_keys = list(d["rollback"]["scope"].keys())
        assert rb_keys == sorted(rb_keys)


# ── compatibility validation ───────────────────────────────────────────────────

class TestValidateActionContract:
    def test_action_contract_compatibility_with_mxai_action(self, email_contract, email_program):
        result = validate_action_contract(email_contract, email_program)
        assert isinstance(result, ActionContractValidationResult)
        assert result.ok is True
        assert result.errors == []

    def test_action_contract_validate_result_ok_when_compatible(self, email_contract, email_program):
        result = validate_action_contract(email_contract, email_program)
        assert result.ok

    def test_action_contract_rejects_action_not_found_in_program(self, email_contract, simulate_program):
        result = validate_action_contract(email_contract, simulate_program)
        assert not result.ok
        assert any("not found" in e for e in result.errors)

    def test_action_contract_rejects_capability_mismatch(self, email_program):
        src_mismatch = """
ACTION_CONTRACT SendNotification
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://example.com/notify"]
  END
  DRY_RUN required
  ROLLBACK undo_post
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED true
END

ROLLBACK undo_post
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://example.com/notify/undo"]
  END
END
"""
        contract = parse_mxact(src_mismatch)[0]
        result = validate_action_contract(contract, email_program)
        assert not result.ok
        assert any("does not match" in e for e in result.errors)

    def test_action_contract_rejects_simulate_only_policy(self, email_contract, simulate_program):
        # Respond action exists but has simulate_only policy
        from matrixai.actions.schema import ActionContractSpec
        # build a contract matching Respond but it has simulate_only
        src = """
ACTION_CONTRACT Respond
  CAPABILITY notification
  SCOPE
    allowed_recipients = ["ops@example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED false
END
"""
        contract = parse_mxact(src)[0]
        result = validate_action_contract(contract, simulate_program)
        assert not result.ok
        assert any("real_with_audit" in e for e in result.errors)


# ── signing key ────────────────────────────────────────────────────────────────

class TestSigningKey:
    def test_check_signing_key_available_returns_true_when_set(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_ACTION_SIGNING_KEY", "a" * 64)
        assert check_signing_key_available() is True

    def test_check_signing_key_available_returns_false_when_missing(self, monkeypatch):
        monkeypatch.delenv("MATRIXAI_ACTION_SIGNING_KEY", raising=False)
        assert check_signing_key_available() is False

    def test_action_contract_rejects_signature_without_key(self, email_contract, monkeypatch):
        monkeypatch.delenv("MATRIXAI_ACTION_SIGNING_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MATRIXAI_ACTION_SIGNING_KEY"):
            require_signing_key(email_contract)

    def test_action_contract_accepts_signature_with_key(self, email_contract, monkeypatch):
        monkeypatch.setenv("MATRIXAI_ACTION_SIGNING_KEY", "b" * 64)
        require_signing_key(email_contract)  # must not raise

    def test_action_contract_validate_accepts_no_signing_key_when_not_required(self, monkeypatch):
        monkeypatch.delenv("MATRIXAI_ACTION_SIGNING_KEY", raising=False)
        src = """
ACTION_CONTRACT SendNotification
  CAPABILITY notification
  SCOPE
    allowed_recipients = ["ops@example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED false
END
"""
        contract = parse_mxact(src)[0]
        require_signing_key(contract)  # must not raise when signature_required=False
