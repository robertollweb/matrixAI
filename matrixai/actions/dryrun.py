# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from matrixai.actions.schema import ActionContractSpec, RateLimitSpec, MUTATING_CAPABILITIES
from matrixai.ir.schema import MatrixAIProgram

_DEFAULT_VALIDITY_MINUTES = 5
_MAX_VALIDITY_MINUTES = 60

_TYPE_CHECKS: dict[str, type | tuple] = {
    "String": str,
    "Integer": int,
    "Float": float,
    "Boolean": bool,
    "Probability": (int, float),
    "Score": (int, float),
    "Confidence": (int, float),
    "Risk": (int, float),
}


# ── rate tracker ──────────────────────────────────────────────────────────────

class RateTracker:
    """In-memory sliding-window rate tracker (per-instance, not persistent)."""

    def __init__(self) -> None:
        self._timestamps: list[datetime] = []

    def would_exceed(self, rate_limit: RateLimitSpec, now: datetime | None = None) -> bool:
        t = now or datetime.now(tz=timezone.utc)
        minute_ago = t - timedelta(minutes=1)
        hour_ago = t - timedelta(hours=1)
        per_minute = sum(1 for ts in self._timestamps if ts > minute_ago)
        per_hour = sum(1 for ts in self._timestamps if ts > hour_ago)
        return per_minute >= rate_limit.per_minute or per_hour >= rate_limit.per_hour

    def record(self, now: datetime | None = None) -> None:
        t = now or datetime.now(tz=timezone.utc)
        self._timestamps.append(t)


# ── dry-run report ────────────────────────────────────────────────────────────

@dataclass
class DryRunReport:
    report_id: str
    model_hash: str
    parameter_set_id: str
    action_contract_hash: str
    input_hash: str
    executed_at: str     # ISO8601
    valid_until: str     # ISO8601
    ok: bool
    errors: list[str]
    scope_ok: bool
    rate_limit_ok: bool
    input_types_ok: bool
    rollback_ok: bool

    def is_expired(self, now: datetime | None = None) -> bool:
        t = now or datetime.now(tz=timezone.utc)
        return t > datetime.fromisoformat(self.valid_until)


# ── internal helpers ──────────────────────────────────────────────────────────

def _input_hash(input_data: dict) -> str:
    payload = json.dumps(input_data, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _check_input_types(action_name: str, input_params: tuple, input_data: dict) -> list[str]:
    errors: list[str] = []
    for param in input_params:
        if param.name not in input_data:
            errors.append(f"Missing required input field {param.name!r} for action {action_name!r}")
            continue
        expected = _TYPE_CHECKS.get(param.type)
        if expected and not isinstance(input_data[param.name], expected):
            errors.append(
                f"Input {param.name!r} expected {param.type}, "
                f"got {type(input_data[param.name]).__name__}"
            )
    return errors


def _check_scope(capability: str, scope: dict, input_data: dict) -> list[str]:
    errors: list[str] = []
    if capability in ("email_send", "notification"):
        recipient = input_data.get("recipient")
        if recipient is not None and "allowed_recipients" in scope:
            if recipient not in scope["allowed_recipients"]:
                errors.append(
                    f"recipient {recipient!r} is not in scope allowed_recipients"
                )
    if capability in ("http_get", "http_post"):
        url = input_data.get("url")
        if url is not None and "allowed_urls" in scope:
            if not any(url.startswith(u) for u in scope["allowed_urls"]):
                errors.append(f"url {url!r} is not in scope allowed_urls")
    if capability in ("filesystem_read", "filesystem_write"):
        path = input_data.get("path")
        if path is not None and "allowed_paths" in scope:
            if not any(path.startswith(p) for p in scope["allowed_paths"]):
                errors.append(f"path {path!r} is not in scope allowed_paths")
    return errors


# ── simulator ─────────────────────────────────────────────────────────────────

class DryRunSimulator:

    def simulate(
        self,
        contract: ActionContractSpec,
        program: MatrixAIProgram,
        parameter_set_id: str,
        model_hash: str,
        input_data: dict[str, Any],
        rate_tracker: RateTracker | None = None,
        validity_minutes: int = _DEFAULT_VALIDITY_MINUTES,
        now: datetime | None = None,
    ) -> DryRunReport:
        t = now or datetime.now(tz=timezone.utc)
        validity_minutes = min(max(validity_minutes, 1), _MAX_VALIDITY_MINUTES)
        valid_until = t + timedelta(minutes=validity_minutes)

        errors: list[str] = []
        scope_ok = True
        rate_limit_ok = True
        input_types_ok = True
        rollback_ok = True

        action = next((a for a in program.actions if a.name == contract.name), None)
        if action is None:
            errors.append(f"Action {contract.name!r} not found in program")
        else:
            type_errors = _check_input_types(action.name, action.input_params, input_data)
            if type_errors:
                input_types_ok = False
                errors.extend(type_errors)

        scope_errors = _check_scope(contract.capability, contract.scope, input_data)
        if scope_errors:
            scope_ok = False
            errors.extend(scope_errors)

        if contract.rate_limit and rate_tracker:
            if rate_tracker.would_exceed(contract.rate_limit, now=t):
                rate_limit_ok = False
                errors.append(f"Rate limit would be exceeded for {contract.name!r}")
            else:
                rate_tracker.record(t)  # consume the slot when the limit is not exceeded

        if contract.capability in MUTATING_CAPABILITIES and contract.rollback is None:
            rollback_ok = False
            errors.append(
                f"Contract {contract.name!r} requires ROLLBACK for capability "
                f"{contract.capability!r} but none is declared"
            )

        from matrixai.actions.contract import compute_action_contract_hash
        return DryRunReport(
            report_id=f"dry-{uuid.uuid4().hex[:12]}",
            model_hash=model_hash,
            parameter_set_id=parameter_set_id,
            action_contract_hash=compute_action_contract_hash(contract),
            input_hash=_input_hash(input_data),
            executed_at=t.isoformat(),
            valid_until=valid_until.isoformat(),
            ok=len(errors) == 0,
            errors=errors,
            scope_ok=scope_ok,
            rate_limit_ok=rate_limit_ok,
            input_types_ok=input_types_ok,
            rollback_ok=rollback_ok,
        )
