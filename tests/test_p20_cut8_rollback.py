"""P20 C8 — RollbackManager: ejecucion del rollback declarado en un contrato."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from matrixai.actions import (
    ActionExecutor,
    ActionResult,
    ActionTrace,
    DryRunSimulator,
    ExecutionContext,
    RollbackError,
    RollbackManager,
    RollbackResult,
    build_action_trace,
    parse_mxact,
)
from matrixai.actions.rollback import _rollback_spec_to_contract
from matrixai.parser.parser import parse_text

# ── fixtures ───────────────────────────────────────────────────────────────────

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_EMAIL_WITH_ROLLBACK = """
ACTION_CONTRACT SendAlert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    allowed_domains    = ["example.com"]
  END
  DRY_RUN required
  ROLLBACK send_correction
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED false
END

ROLLBACK send_correction
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    template = "correction"
  END
END
"""

_HTTP_POST_WITH_ROLLBACK = """
ACTION_CONTRACT CreateTicket
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://api.example.com/tickets"]
  END
  DRY_RUN required
  ROLLBACK delete_ticket
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=2 per_hour=20
  SIGNATURE_REQUIRED false
END

ROLLBACK delete_ticket
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://api.example.com/tickets"]
  END
END
"""

_NO_ROLLBACK_MXACT = """
ACTION_CONTRACT FetchData
  CAPABILITY http_get
  SCOPE
    allowed_urls = ["https://api.example.com/"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=10 per_hour=100
  SIGNATURE_REQUIRED false
END
"""

_PROG = """
PROJECT Test

VECTOR Input[1]
  x: Probability
END

ACTION SendAlert
  TARGET email_send
  POLICY real_with_audit
  CONDITION x > 0.5
  INPUT recipient: String
END

GRAPH
  Input -> SendAlert
END
"""


def _make_trace(contract, input_data, *, ok=False):
    prog = parse_text(_PROG)
    sim = DryRunSimulator()
    report = sim.simulate(contract, prog, "param_set_1", "model_hash_abc", input_data, now=_T0)
    result = ActionResult(ok=ok, response_summary="", latency_ms=5.0, error="timeout" if not ok else None)
    return build_action_trace(
        ExecutionContext(
            contract=contract,
            dry_run_report=report,
            input_data=input_data,
            model_hash="model_hash_abc",
            parameter_set_id="param_set_1",
            allow_real_actions=True,
        ),
        result,
        now=_T0,
    )


# ── 1. imports and structural ─────────────────────────────────────────────────

def test_rollback_manager_importable():
    assert RollbackManager is not None


def test_rollback_result_importable():
    assert RollbackResult is not None


def test_rollback_error_is_action_executor_error():
    from matrixai.actions import ActionExecutorError
    assert issubclass(RollbackError, ActionExecutorError)


# ── 2. no rollback declared ───────────────────────────────────────────────────

def test_rollback_not_attempted_when_no_rollback_declared():
    contracts = parse_mxact(_NO_ROLLBACK_MXACT)
    contract = contracts[0]
    trace = ActionTrace(
        report_id="r1", model_hash="mh", parameter_set_id="ps",
        action_contract_hash="h", input_hash="ih",
        executed_at=_T0.isoformat(), executor_kind="in_process",
        ok=False, response_summary="", error="fail", latency_ms=1.0,
        hmac_signature=None,
    )
    mgr = RollbackManager()
    result = mgr.execute_rollback(trace, contract, {})
    assert result.attempted is False
    assert result.ok is False
    assert "No rollback" in result.error


# ── 3. successful rollback ────────────────────────────────────────────────────

def test_rollback_executed_with_injected_handler():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    trace = _make_trace(contract, {"recipient": "ops@example.com"})

    called = {}

    def fake_email(smtp_host, smtp_port, smtp_user, smtp_pass, recipient, subject, body):
        called["invoked"] = True
        return "correction sent"

    executor = ActionExecutor(email_fn=fake_email)
    mgr = RollbackManager(executor=executor)
    result = mgr.execute_rollback(
        trace, contract,
        {"recipient": "ops@example.com", "template": "correction"},
        now=_T0,
    )

    assert result.attempted is True
    assert result.ok is True
    assert result.error is None
    assert called.get("invoked") is True


def test_rollback_result_names_rollback_contract():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    trace = _make_trace(contract, {"recipient": "ops@example.com"})

    executor = ActionExecutor(email_fn=lambda scope, inp: ActionResult(
        ok=True, response_summary="ok", latency_ms=1.0, error=None))
    mgr = RollbackManager(executor=executor)
    result = mgr.execute_rollback(trace, contract, {"recipient": "ops@example.com"}, now=_T0)

    assert result.rollback_contract_name == "send_correction"


# ── 4. failed rollback ────────────────────────────────────────────────────────

def test_rollback_result_not_ok_when_handler_fails():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    trace = _make_trace(contract, {"recipient": "ops@example.com"})

    def failing_email(smtp_host, smtp_port, smtp_user, smtp_pass, recipient, subject, body):
        raise RuntimeError("SMTP unavailable")

    executor = ActionExecutor(email_fn=failing_email)
    mgr = RollbackManager(executor=executor)
    result = mgr.execute_rollback(trace, contract, {"recipient": "ops@example.com"}, now=_T0)

    assert result.attempted is True
    assert result.ok is False
    assert result.error is not None and "SMTP" in result.error


# ── 5. _rollback_spec_to_contract ────────────────────────────────────────────

def test_rollback_spec_to_contract_inherits_capability():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    rb_contract = _rollback_spec_to_contract(contract.rollback, contract)
    assert rb_contract.capability == "email_send"


def test_rollback_spec_to_contract_dry_run_not_required():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    rb_contract = _rollback_spec_to_contract(contract.rollback, contract)
    assert rb_contract.dry_run_required is False


def test_rollback_spec_to_contract_no_nested_rollback():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    rb_contract = _rollback_spec_to_contract(contract.rollback, contract)
    assert rb_contract.rollback is None


def test_rollback_spec_to_contract_no_sandbox_required():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    rb_contract = _rollback_spec_to_contract(contract.rollback, contract)
    assert rb_contract.sandbox_required is False


def test_rollback_spec_to_contract_not_signature_required():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    rb_contract = _rollback_spec_to_contract(contract.rollback, contract)
    assert rb_contract.signature_required is False


def test_rollback_spec_to_contract_inherits_scope():
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    rb_contract = _rollback_spec_to_contract(contract.rollback, contract)
    assert "allowed_recipients" in rb_contract.scope


# ── 6. http_post rollback path ────────────────────────────────────────────────

def test_rollback_works_for_http_post_capability():
    contracts = parse_mxact(_HTTP_POST_WITH_ROLLBACK)
    contract = contracts[0]

    trace = ActionTrace(
        report_id="r2", model_hash="mh2", parameter_set_id="ps2",
        action_contract_hash="h2", input_hash="ih2",
        executed_at=_T0.isoformat(), executor_kind="in_process",
        ok=False, response_summary="", error="upstream error", latency_ms=2.0,
        hmac_signature=None,
    )

    called = {}

    def fake_post(url, headers, payload, timeout):
        called["url"] = url
        return "deleted"

    executor = ActionExecutor(http_post_fn=fake_post)
    mgr = RollbackManager(executor=executor)
    result = mgr.execute_rollback(
        trace, contract,
        {"url": "https://api.example.com/tickets/123", "method": "DELETE"},
        now=_T0,
    )

    assert result.attempted is True
    assert result.ok is True
    assert result.rollback_contract_name == "delete_ticket"


# ── 7. regression: rollback dry-run must not expire immediately ───────────────

def test_rollback_dry_run_does_not_expire_immediately_with_default_now():
    """Regression: _build_rollback_dry_run used to set valid_until == now,
    causing the rollback to always fail with 'DryRunReport ... is expired' when
    invoked without an explicit `now` (i.e. in real runbooks)."""
    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    trace = _make_trace(contract, {"recipient": "ops@example.com"})

    called: dict = {}

    def fake_email(smtp_host, smtp_port, smtp_user, smtp_pass, recipient, subject, body):
        called["invoked"] = True
        return "correction sent"

    executor = ActionExecutor(email_fn=fake_email)
    mgr = RollbackManager(executor=executor)
    # NOTE: no now= argument → uses datetime.now() internally.
    result = mgr.execute_rollback(
        trace, contract, {"recipient": "ops@example.com", "template": "correction"}
    )

    assert result.attempted is True
    assert result.ok is True, f"rollback should succeed but failed: {result.error}"
    assert called.get("invoked") is True


def test_rollback_dry_run_has_validity_window():
    """The synthesized rollback DryRunReport should have valid_until strictly
    greater than executed_at, so it can be consumed by ActionExecutor without
    immediately tripping the expiry check."""
    from matrixai.actions.rollback import _build_rollback_dry_run

    contracts = parse_mxact(_EMAIL_WITH_ROLLBACK)
    contract = contracts[0]
    rollback_contract = _rollback_spec_to_contract(contract.rollback, contract)
    trace = _make_trace(contract, {"recipient": "ops@example.com"})

    report = _build_rollback_dry_run(
        rollback_contract,
        {"recipient": "ops@example.com", "template": "correction"},
        trace,
        now=_T0,
    )

    assert report.executed_at == _T0.isoformat()
    assert datetime.fromisoformat(report.valid_until) > _T0
    assert not report.is_expired(now=_T0)
