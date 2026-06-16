"""P20 C7 — ActionTrace: construcción, firma HMAC-SHA256 y verificación de integridad."""
from __future__ import annotations

import hashlib
import hmac as _hmac
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from matrixai.actions import (
    ActionTrace,
    DryRunSimulator,
    ExecutionContext,
    build_action_trace,
    parse_mxact,
    sign_action_trace,
    verify_action_trace,
    ActionResult,
)
from matrixai.actions.trace import _canonical_message, _HMAC_PREFIX
from matrixai.parser.parser import parse_text


# ── fixtures ───────────────────────────────────────────────────────────────────

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_SIGNING_KEY = "a" * 64  # 64 hex chars = 32 bytes

_EMAIL_MXACT = """
ACTION_CONTRACT SendAlert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    allowed_domains    = ["example.com"]
  END
  DRY_RUN required
  ROLLBACK undo_alert
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_alert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    template = "correction"
  END
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
  INPUT recipient: String, subject: String
END

GRAPH
  Input -> SendAlert
END
"""

_INPUT = {"recipient": "ops@example.com", "subject": "Alert"}


def _make_context(input_data=None, *, now=_T0):
    contracts = parse_mxact(_EMAIL_MXACT)
    contract = contracts[0]
    prog = parse_text(_PROG)
    sim = DryRunSimulator()
    report = sim.simulate(
        contract, prog, "param_set_1", "model_hash_abc",
        input_data or _INPUT, now=now,
    )
    return ExecutionContext(
        contract=contract,
        dry_run_report=report,
        input_data=input_data or _INPUT,
        model_hash="model_hash_abc",
        parameter_set_id="param_set_1",
        allow_real_actions=True,
    )


def _make_result(ok=True, executor_kind="in_process"):
    return ActionResult(
        ok=ok,
        response_summary="sent" if ok else "",
        latency_ms=12.5,
        error=None if ok else "timeout",
        executor_kind=executor_kind,
    )


# ── 1. ActionTrace structure ──────────────────────────────────────────────────

def test_action_trace_importable():
    assert ActionTrace is not None


def test_action_trace_dataclass_fields():
    ctx = _make_context()
    result = _make_result()
    trace = build_action_trace(ctx, result, now=_T0)

    assert trace.report_id == ctx.dry_run_report.report_id
    assert trace.model_hash == "model_hash_abc"
    assert trace.parameter_set_id == "param_set_1"
    assert trace.action_contract_hash == ctx.dry_run_report.action_contract_hash
    assert trace.input_hash == ctx.dry_run_report.input_hash
    assert trace.executor_kind == "in_process"
    assert trace.ok is True
    assert trace.response_summary == "sent"
    assert trace.error is None
    assert trace.latency_ms == 12.5


def test_action_trace_executed_at_is_iso8601():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), now=_T0)
    # parseable back to datetime
    parsed = datetime.fromisoformat(trace.executed_at)
    assert parsed.tzinfo is not None


def test_action_trace_without_signing_key_has_none_signature():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), now=_T0)
    assert trace.hmac_signature is None


# ── 2. signing ────────────────────────────────────────────────────────────────

def test_sign_action_trace_returns_hmac_prefix():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), now=_T0)
    sig = sign_action_trace(trace, _SIGNING_KEY)
    assert sig.startswith(_HMAC_PREFIX)


def test_sign_action_trace_hex_is_64_chars():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), now=_T0)
    sig = sign_action_trace(trace, _SIGNING_KEY)
    hex_part = sig.removeprefix(_HMAC_PREFIX)
    assert len(hex_part) == 64


def test_sign_action_trace_is_deterministic():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), now=_T0)
    assert sign_action_trace(trace, _SIGNING_KEY) == sign_action_trace(trace, _SIGNING_KEY)


def test_build_action_trace_with_signing_key_sets_signature():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    assert trace.hmac_signature is not None
    assert trace.hmac_signature.startswith(_HMAC_PREFIX)


def test_canonical_message_contains_all_fields():
    ctx = _make_context()
    result = _make_result()
    trace = build_action_trace(ctx, result, now=_T0)
    msg = _canonical_message(trace)
    # identity / routing fields
    assert trace.report_id in msg
    assert trace.model_hash in msg
    assert trace.parameter_set_id in msg
    assert trace.action_contract_hash in msg
    assert trace.input_hash in msg
    assert trace.executed_at in msg
    # result fields — must be covered so tampering is detectable
    assert trace.executor_kind in msg
    assert ("true" if trace.ok else "false") in msg
    assert trace.response_summary in msg


def test_canonical_message_ok_field_is_tamper_detectable():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    tampered = replace(trace, ok=not trace.ok)
    assert verify_action_trace(tampered, _SIGNING_KEY) is False


def test_canonical_message_response_summary_is_tamper_detectable():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    tampered = replace(trace, response_summary="tampered response")
    assert verify_action_trace(tampered, _SIGNING_KEY) is False


def test_canonical_message_error_field_is_tamper_detectable():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(ok=False), signing_key=_SIGNING_KEY, now=_T0)
    tampered = replace(trace, error=None)
    assert verify_action_trace(tampered, _SIGNING_KEY) is False


def test_canonical_message_latency_ms_is_tamper_detectable():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    tampered = replace(trace, latency_ms=9999.0)
    assert verify_action_trace(tampered, _SIGNING_KEY) is False


# ── 3. verification ───────────────────────────────────────────────────────────

def test_verify_action_trace_returns_true_for_valid_signature():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    assert verify_action_trace(trace, _SIGNING_KEY) is True


def test_verify_action_trace_returns_false_for_wrong_key():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    wrong_key = "b" * 64
    assert verify_action_trace(trace, wrong_key) is False


def test_verify_action_trace_returns_false_when_signature_none():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), now=_T0)
    assert verify_action_trace(trace, _SIGNING_KEY) is False


def test_verify_action_trace_returns_false_for_tampered_model_hash():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    tampered = replace(trace, model_hash="tampered_hash")
    assert verify_action_trace(tampered, _SIGNING_KEY) is False


def test_verify_action_trace_returns_false_for_tampered_executed_at():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), signing_key=_SIGNING_KEY, now=_T0)
    tampered = replace(trace, executed_at="2020-01-01T00:00:00+00:00")
    assert verify_action_trace(tampered, _SIGNING_KEY) is False


# ── 4. integration with sandbox result ───────────────────────────────────────

def test_build_action_trace_from_sandbox_result():
    ctx = _make_context()
    result = ActionResult(
        ok=True,
        response_summary='{"rows_written": 1}',
        latency_ms=55.3,
        error=None,
        executor_kind="sandbox",
    )
    trace = build_action_trace(ctx, result, signing_key=_SIGNING_KEY, now=_T0)
    assert trace.executor_kind == "sandbox"
    assert verify_action_trace(trace, _SIGNING_KEY) is True


def test_build_action_trace_from_failed_result():
    ctx = _make_context()
    result = _make_result(ok=False)
    trace = build_action_trace(ctx, result, signing_key=_SIGNING_KEY, now=_T0)
    assert trace.ok is False
    assert trace.error == "timeout"
    assert verify_action_trace(trace, _SIGNING_KEY) is True


def test_sign_matches_manual_hmac():
    ctx = _make_context()
    trace = build_action_trace(ctx, _make_result(), now=_T0)
    msg = _canonical_message(trace).encode()
    key_bytes = bytes.fromhex(_SIGNING_KEY)
    expected = "hmac-sha256:" + _hmac.new(key_bytes, msg, hashlib.sha256).hexdigest()
    assert sign_action_trace(trace, _SIGNING_KEY) == expected
