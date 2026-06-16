"""P20 Audit Fixes — regression tests for post-audit corrections (rounds 1-4).

PA1: HMAC canonical covers all result fields
PA2a: signing_key via ExecutionContext (not only env var)
PA2b: HumanApprovalGate integrated in executor preflight
PA2c: validate_scope called in validate_action_contract
PA2d: RateTracker passed to DryRunSimulator in CLI/server path
PA3a: find_approved() binds to model_hash and parameter_set_id
PA3b: dry-run-action CLI calls validate_action_contract
PA4: P4 identity derived from certified ParameterSet, not CLI args / request body
PA4b: execute-action accepts ParameterSet file without explicit --model-hash
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from matrixai.actions import (
    ActionExecutor,
    ApprovalStore,
    DryRunSimulator,
    ExecutionContext,
    HumanApprovalGate,
    parse_mxact,
    validate_action_contract,
)
from matrixai.actions.dryrun import RateTracker
from matrixai.actions.trace import (
    ActionTrace,
    _canonical_message,
    build_action_trace,
    sign_action_trace,
    verify_action_trace,
)
from matrixai.parser.parser import parse_text

_T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
_KEY = "a" * 64

# ── shared fixtures ───────────────────────────────────────────────────────────

# Contract name matches the ACTION name in _MXAI ("DraftReply")
_EMAIL_MXACT = """
ACTION_CONTRACT DraftReply
  CAPABILITY notification
  SCOPE
    allowed_recipients = ["ops@example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED false
END
"""

_HUMAN_MXACT = """
ACTION_CONTRACT DraftReply
  CAPABILITY notification
  SCOPE
    allowed_recipients = ["ceo@example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL true
  APPROVAL_CHANNEL slack
  SIGNATURE_REQUIRED false
END
"""

_BAD_SCOPE_MXACT = """
ACTION_CONTRACT DraftReply
  CAPABILITY notification
  SCOPE
    typo_field = ["ops@example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED false
END
"""

_HTTP_MXACT = """
ACTION_CONTRACT FetchStatus
  CAPABILITY http_get
  SCOPE
    allowed_urls = ["http://127.0.0.1"]
    timeout_seconds = 2
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED false
END
"""

_MXAI = """
PROJECT AlertAgent

VECTOR Input[1]
  score
END

GRAPH
  Input -> DraftReply
END

ACTION DraftReply
  WHEN Input > 0.5
  POLICY real_with_audit
  CALL notification.send
  TARGET notification
END
"""

_TRAINABLE_MXAI = """
PROJECT CliIdentityModel

VECTOR Input[1]
  x: Probability
END

FUNCTION ScoreModel
  R: Probability = sigmoid(W1 * Input + b1)
END

GRAPH
  Input -> ScoreModel
END

ACTION FetchStatus
  TARGET http_get
  POLICY real_with_audit
  CONDITION R > 0.5
END
"""


def _make_trace(ok=True, response_summary="sent", error=None, latency_ms=10.0):
    return ActionTrace(
        report_id="rep-001",
        model_hash="model_abc",
        parameter_set_id="ps_1",
        action_contract_hash="sha256:" + "a" * 64,
        input_hash="sha256:" + "b" * 64,
        executed_at="2026-06-01T12:00:00+00:00",
        executor_kind="in_process",
        ok=ok,
        response_summary=response_summary,
        error=error,
        latency_ms=latency_ms,
        hmac_signature=None,
    )


def _make_context(mxact_text, input_data, allow_real_actions=True):
    contracts = parse_mxact(mxact_text)
    contract = contracts[0]
    program = parse_text(_MXAI)
    sim = DryRunSimulator()
    report = sim.simulate(contract, program, "ps_1", "model_abc", input_data)
    return ExecutionContext(
        contract=contract,
        dry_run_report=report,
        input_data=input_data,
        model_hash="model_abc",
        parameter_set_id="ps_1",
        allow_real_actions=allow_real_actions,
    )


# ── PA1: HMAC canonical covers all result fields ─────────────────────────────

def test_canonical_message_includes_report_id():
    t = _make_trace()
    msg = _canonical_message(t)
    assert t.report_id in msg


def test_canonical_message_includes_executor_kind():
    t = _make_trace()
    msg = _canonical_message(t)
    assert t.executor_kind in msg


def test_canonical_message_includes_ok():
    t = _make_trace(ok=True)
    assert "true" in _canonical_message(t)
    t2 = _make_trace(ok=False)
    assert "false" in _canonical_message(t2)


def test_canonical_message_includes_response_summary():
    t = _make_trace(response_summary="250 OK")
    assert "250 OK" in _canonical_message(t)


def test_canonical_message_includes_error():
    t = _make_trace(ok=False, error="timeout")
    assert "timeout" in _canonical_message(t)


def test_canonical_message_includes_latency_ms():
    t = _make_trace(latency_ms=42.5)
    assert "42.5" in _canonical_message(t)


def test_tamper_ok_invalidates_signature():
    t = _make_trace()
    sig_str = sign_action_trace(t, _KEY)
    signed = replace(t, hmac_signature=sig_str)
    tampered = replace(signed, ok=not t.ok)
    assert verify_action_trace(tampered, _KEY) is False


def test_tamper_response_summary_invalidates_signature():
    t = _make_trace()
    signed = replace(t, hmac_signature=sign_action_trace(t, _KEY))
    tampered = replace(signed, response_summary="tampered")
    assert verify_action_trace(tampered, _KEY) is False


def test_tamper_error_invalidates_signature():
    t = _make_trace(ok=False, error="original error")
    signed = replace(t, hmac_signature=sign_action_trace(t, _KEY))
    tampered = replace(signed, error=None)
    assert verify_action_trace(tampered, _KEY) is False


def test_tamper_latency_ms_invalidates_signature():
    t = _make_trace(latency_ms=10.0)
    signed = replace(t, hmac_signature=sign_action_trace(t, _KEY))
    tampered = replace(signed, latency_ms=9999.0)
    assert verify_action_trace(tampered, _KEY) is False


# ── PA2a: signing_key via ExecutionContext, not only env var ──────────────────

def test_executor_preflight_accepts_signing_key_from_context(monkeypatch):
    """When contract.signature_required, a key in context is enough (no env var)."""
    _SIGNED_MXACT = _EMAIL_MXACT.replace(
        "SIGNATURE_REQUIRED false", "SIGNATURE_REQUIRED true"
    )
    monkeypatch.delenv("MATRIXAI_ACTION_SIGNING_KEY", raising=False)
    contracts = parse_mxact(_SIGNED_MXACT)
    contract = contracts[0]
    program = parse_text(_MXAI)
    sim = DryRunSimulator()
    report = sim.simulate(contract, program, "ps_1", "model_abc",
                          {"recipient": "ops@example.com"})
    ctx = ExecutionContext(
        contract=contract,
        dry_run_report=report,
        input_data={"recipient": "ops@example.com"},
        model_hash="model_abc",
        parameter_set_id="ps_1",
        allow_real_actions=True,
        signing_key=_KEY,  # key passed here, not in env
    )
    calls = []
    executor = ActionExecutor(webhook_fn=lambda *a: calls.append(True) or "OK")
    executor.execute(ctx)  # must not raise
    assert calls


def test_executor_preflight_raises_when_signature_required_and_no_key(monkeypatch):
    """Without key in context or env var, signature_required must raise."""
    _SIGNED_MXACT = _EMAIL_MXACT.replace(
        "SIGNATURE_REQUIRED false", "SIGNATURE_REQUIRED true"
    )
    monkeypatch.delenv("MATRIXAI_ACTION_SIGNING_KEY", raising=False)
    contracts = parse_mxact(_SIGNED_MXACT)
    contract = contracts[0]
    program = parse_text(_MXAI)
    sim = DryRunSimulator()
    report = sim.simulate(contract, program, "ps_1", "model_abc",
                          {"recipient": "ops@example.com"})
    ctx = ExecutionContext(
        contract=contract,
        dry_run_report=report,
        input_data={"recipient": "ops@example.com"},
        model_hash="model_abc",
        parameter_set_id="ps_1",
        allow_real_actions=True,
        signing_key=None,  # no key
    )
    from matrixai.actions.executor import ActionExecutorError
    with pytest.raises(ActionExecutorError, match="signing key"):
        ActionExecutor().execute(ctx)


# ── PA2b: HumanApprovalGate integrated in executor preflight ─────────────────

def test_executor_preflight_blocks_human_approval_without_store():
    """human_approval=True and no approval_store → raises."""
    from matrixai.actions.executor import ActionExecutorError
    ctx = _make_context(_HUMAN_MXACT, {"recipient": "ceo@example.com"})
    # approval_store is None (default)
    with pytest.raises(ActionExecutorError, match="ApprovalStore"):
        ActionExecutor(webhook_fn=lambda *a: "OK").execute(ctx)


def test_executor_preflight_blocks_human_approval_without_approval():
    """human_approval=True, store provided but no approved record → raises."""
    from matrixai.actions.executor import ActionExecutorError
    ctx = _make_context(_HUMAN_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    ctx = replace(ctx, approval_store=store)
    with pytest.raises(ActionExecutorError, match="human approval"):
        ActionExecutor(webhook_fn=lambda *a: "OK").execute(ctx)


def test_executor_preflight_passes_human_approval_when_approved():
    """human_approval=True and valid approval → executes without raising."""
    ctx = _make_context(_HUMAN_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx)
    store.approve(pending.execution_id)
    ctx = replace(ctx, approval_store=store)
    calls = []
    ActionExecutor(webhook_fn=lambda *a: calls.append(True) or "OK").execute(ctx)
    assert calls


def test_executor_preflight_skips_approval_gate_when_not_required():
    """human_approval=False → no gate check even without store."""
    ctx = _make_context(_EMAIL_MXACT, {"recipient": "ops@example.com"})
    # approval_store is None — must not raise for contracts without human_approval
    calls = []
    ActionExecutor(webhook_fn=lambda *a: calls.append(True) or "OK").execute(ctx)
    assert calls


# ── PA2c: validate_scope called in validate_action_contract ──────────────────

def test_validate_action_contract_fails_on_missing_scope_field():
    """validate_action_contract must reject a contract with invalid scope."""
    contracts = parse_mxact(_BAD_SCOPE_MXACT)
    contract = contracts[0]
    program = parse_text(_MXAI)
    result = validate_action_contract(contract, program)
    assert not result.ok
    scope_errors = [e for e in result.errors if "scope" in e.lower() or "allowed_recipients" in e]
    assert scope_errors, f"Expected scope validation error, got: {result.errors}"


def test_validate_action_contract_passes_valid_scope():
    """validate_action_contract must accept a contract with valid email_send scope."""
    contracts = parse_mxact(_EMAIL_MXACT)
    contract = contracts[0]
    program = parse_text(_MXAI)
    result = validate_action_contract(contract, program)
    assert result.ok, result.errors


# ── PA2d: RateTracker passed to DryRunSimulator ───────────────────────────────

def test_rate_tracker_blocks_when_limit_exceeded():
    """DryRunSimulator with a saturated RateTracker must fail rate limit check."""
    contracts = parse_mxact(_EMAIL_MXACT.replace(
        "HUMAN_APPROVAL false",
        "HUMAN_APPROVAL false\n  RATE_LIMIT per_minute=1 per_hour=10"
    ))
    contract = contracts[0]
    program = parse_text(_MXAI)
    input_data = {"recipient": "ops@example.com"}

    tracker = RateTracker()
    sim = DryRunSimulator()

    # first call succeeds
    r1 = sim.simulate(contract, program, "ps_1", "model_abc", input_data,
                      rate_tracker=tracker)
    assert r1.ok
    tracker.record(datetime.now(tz=timezone.utc))
    tracker.record(datetime.now(tz=timezone.utc))

    # second call with saturated tracker fails rate limit
    r2 = sim.simulate(contract, program, "ps_1", "model_abc", input_data,
                      rate_tracker=tracker)
    assert not r2.rate_limit_ok


# ── PA3a: find_approved() binds to model_hash and parameter_set_id ────────────

def _submit_and_approve(store, ctx):
    """Helper: submit + approve a PendingExecution for the given context."""
    pending = store.submit(ctx)
    store.approve(pending.execution_id)


def test_find_approved_rejects_different_model_hash():
    """Approval for model_v1 must not satisfy a context with model_v2."""
    ctx = _make_context(_HUMAN_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    _submit_and_approve(store, ctx)

    gate = HumanApprovalGate(store)
    ctx_different_model = replace(ctx, model_hash="model_v2")
    assert gate.find_approved(ctx_different_model) is None


def test_find_approved_rejects_different_parameter_set_id():
    """Approval for ps_1 must not satisfy a context with ps_2."""
    ctx = _make_context(_HUMAN_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    _submit_and_approve(store, ctx)

    gate = HumanApprovalGate(store)
    ctx_different_ps = replace(ctx, parameter_set_id="ps_2")
    assert gate.find_approved(ctx_different_ps) is None


def test_find_approved_accepts_exact_match():
    """Approval must be found when all four binding fields match."""
    ctx = _make_context(_HUMAN_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    _submit_and_approve(store, ctx)

    gate = HumanApprovalGate(store)
    assert gate.find_approved(ctx) is not None


# ── PA3b: dry-run-action CLI calls validate_action_contract ──────────────────

def test_dry_run_action_rejects_contract_with_wrong_target():
    """dry-run-action must fail if validate_action_contract rejects the contract."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    _MXACT_WRONG_TARGET = """\
ACTION_CONTRACT DraftReply
  CAPABILITY http_get
  SCOPE
    allowed_urls = ["https://example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED false
END
"""
    # ACTION DraftReply in _MXAI has TARGET notification, so http_get mismatches
    with tempfile.TemporaryDirectory() as tmp:
        mxact_path = str(Path(tmp) / "test.mxact")
        mxai_path = str(Path(tmp) / "test.mxai")
        Path(mxact_path).write_text(_MXACT_WRONG_TARGET, encoding="utf-8")
        Path(mxai_path).write_text(_MXAI, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", "dry-run-action",
             mxact_path, mxai_path,
             "--contract-name", "DraftReply",
             "--model-hash", "model_abc",
             "--param-set", "ps_1",
             "--input", "{}"],
            capture_output=True, text=True,
            cwd="/home/deployer/matrixAI",
            env=os.environ.copy(),
        )
    assert result.returncode != 0
    assert "contract validation failed" in result.stderr


# ── PA4: P4 identity derived from certified ParameterSet ─────────────────────

def test_find_approved_binds_all_four_fields():
    """Verify find_approved requires action_contract_hash + input_hash + model_hash + ps_id."""
    ctx = _make_context(_HUMAN_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx)
    store.approve(pending.execution_id)
    gate = HumanApprovalGate(store)
    # All four must match
    assert gate.find_approved(ctx) is not None
    assert gate.find_approved(replace(ctx, model_hash="other")) is None
    assert gate.find_approved(replace(ctx, parameter_set_id="other_ps")) is None


def test_executor_uses_context_model_hash_in_trace():
    """ActionTrace.model_hash must come from ExecutionContext, not the ParameterSet directly."""
    ctx = _make_context(_EMAIL_MXACT, {"recipient": "ops@example.com"})
    calls = []

    def fake_webhook(*a):
        calls.append(True)
        return "OK"

    executor = ActionExecutor(webhook_fn=fake_webhook)
    result = executor.execute(ctx)
    trace = build_action_trace(ctx, result)
    assert trace.model_hash == ctx.model_hash
    assert trace.parameter_set_id == ctx.parameter_set_id


def test_execute_action_param_set_file_without_model_hash_uses_artifact_identity(tmp_path):
    """A certified ParameterSet file must not require an explicit --model-hash flag."""
    from matrixai.parameters import build_initial_parameter_set, write_parameter_set

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        mxai_path = tmp_path / "model.mxai"
        mxact_path = tmp_path / "contract.mxact"
        params_path = tmp_path / "params.json"
        mxai_path.write_text(_TRAINABLE_MXAI, encoding="utf-8")
        mxact_path.write_text(_HTTP_MXACT, encoding="utf-8")

        program = parse_text(_TRAINABLE_MXAI)
        parameter_set = build_initial_parameter_set(program, parameter_set_id="certified_ps")
        write_parameter_set(params_path, parameter_set)

        port = server.server_address[1]
        result = subprocess.run(
            [
                sys.executable, "-m", "matrixai", "execute-action",
                str(mxact_path), str(mxai_path),
                "--contract-name", "FetchStatus",
                "--input", json.dumps({"url": f"http://127.0.0.1:{port}/health"}),
                "--param-set", str(params_path),
                "--allow-real-actions",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd="/home/deployer/matrixAI",
            env=os.environ.copy(),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "mismatches ParameterSet model_hash" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["model_hash"] == parameter_set.model_hash
    assert payload["parameter_set_id"] == "certified_ps"
