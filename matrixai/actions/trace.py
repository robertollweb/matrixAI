# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import hmac as _hmac
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrixai.actions.executor import ActionResult, ExecutionContext

_HMAC_PREFIX = "hmac-sha256:"


@dataclass
class ActionTrace:
    report_id: str
    model_hash: str
    parameter_set_id: str
    action_contract_hash: str
    input_hash: str
    executed_at: str          # ISO8601 UTC
    executor_kind: str
    ok: bool
    response_summary: str
    error: str | None
    latency_ms: float
    hmac_signature: str | None  # None when signing key not provided


def _canonical_message(trace: ActionTrace) -> str:
    # All non-signature fields so that tampering with any result field invalidates the HMAC.
    ok_str = "true" if trace.ok else "false"
    error_str = trace.error if trace.error is not None else ""
    latency_str = f"{trace.latency_ms:.6f}"
    return (
        f"{trace.report_id}|{trace.model_hash}|{trace.parameter_set_id}|"
        f"{trace.action_contract_hash}|{trace.input_hash}|{trace.executed_at}|"
        f"{trace.executor_kind}|{ok_str}|{trace.response_summary}|{error_str}|{latency_str}"
    )


def sign_action_trace(trace: ActionTrace, signing_key: str) -> str:
    """Return 'hmac-sha256:<hex>' for the trace's canonical message."""
    key_bytes = bytes.fromhex(signing_key) if len(signing_key) == 64 else signing_key.encode()
    msg = _canonical_message(trace).encode()
    digest = _hmac.new(key_bytes, msg, hashlib.sha256).hexdigest()
    return f"{_HMAC_PREFIX}{digest}"


def verify_action_trace(trace: ActionTrace, signing_key: str) -> bool:
    """Return True iff trace.hmac_signature matches the canonical HMAC."""
    if not trace.hmac_signature:
        return False
    expected = sign_action_trace(trace, signing_key)
    return _hmac.compare_digest(expected, trace.hmac_signature)


def verify_action_trace_with_keystore(
    trace: ActionTrace,
    current_key: str | None,
    keystore: "KeyStore",
) -> bool:
    """Verify *trace* against *current_key* and all historical action keys.

    Use this after a key rotation: traces signed with the old key will still
    verify as long as the old key was recorded in the keystore before rotation.
    Returns True on first match, False if no key matches.
    """
    return keystore.try_verify_action(trace, current_key)


if TYPE_CHECKING:
    from matrixai.signing.keystore import KeyStore


def build_action_trace(
    context: "ExecutionContext",
    result: "ActionResult",
    *,
    signing_key: str | None = None,
    now: datetime | None = None,
) -> ActionTrace:
    """Construct an ActionTrace from an execution context and result."""
    executed_at = (now or datetime.now(tz=timezone.utc)).isoformat()
    trace = ActionTrace(
        report_id=context.dry_run_report.report_id,
        model_hash=context.model_hash,
        parameter_set_id=context.parameter_set_id,
        action_contract_hash=context.dry_run_report.action_contract_hash,
        input_hash=context.dry_run_report.input_hash,
        executed_at=executed_at,
        executor_kind=result.executor_kind,
        ok=result.ok,
        response_summary=result.response_summary,
        error=result.error,
        latency_ms=result.latency_ms,
        hmac_signature=None,
    )
    if signing_key:
        trace.hmac_signature = sign_action_trace(trace, signing_key)
    return trace
