# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from matrixai.actions.contract import compute_action_contract_hash
from matrixai.actions.dryrun import DryRunReport, _DEFAULT_VALIDITY_MINUTES, _input_hash
from matrixai.actions.executor import ActionExecutor, ActionExecutorError, ActionResult, ExecutionContext
from matrixai.actions.schema import ActionContractSpec, RollbackSpec, SandboxLimitsSpec

if TYPE_CHECKING:
    from matrixai.actions.trace import ActionTrace


class RollbackError(ActionExecutorError):
    pass


@dataclass
class RollbackResult:
    attempted: bool
    ok: bool
    error: str | None
    rollback_contract_name: str | None = None


def _rollback_spec_to_contract(spec: RollbackSpec, original: ActionContractSpec) -> ActionContractSpec:
    """Synthesize a minimal ActionContractSpec from a RollbackSpec for execution."""
    return ActionContractSpec(
        name=spec.name,
        capability=spec.capability,
        scope=spec.scope,
        dry_run_required=False,
        rollback=None,
        sandbox_required=False,
        sandbox_limits=None,
        human_approval=False,
        approval_channel=None,
        approval_timeout_seconds=None,
        rate_limit=None,
        signature_required=False,
    )


def _build_rollback_dry_run(
    rollback_contract: ActionContractSpec,
    rollback_input: dict[str, Any],
    original_trace: "ActionTrace",
    *,
    now: datetime | None = None,
    validity_minutes: int = _DEFAULT_VALIDITY_MINUTES,
) -> DryRunReport:
    t = now or datetime.now(tz=timezone.utc)
    ts = t.isoformat()
    # Rollback dry-runs are synthesized internally (pre-approved by the
    # RollbackManager) and must be valid long enough for ActionExecutor to
    # consume them in the same call. Use the same default validity window as
    # DryRunSimulator so behaviour matches a normal dry-run/execute cycle.
    valid_ts = (t + timedelta(minutes=validity_minutes)).isoformat()
    contract_hash = compute_action_contract_hash(rollback_contract)
    return DryRunReport(
        report_id=f"rollback-{original_trace.report_id}",
        model_hash=original_trace.model_hash,
        parameter_set_id=original_trace.parameter_set_id,
        action_contract_hash=contract_hash,
        input_hash=_input_hash(rollback_input),
        executed_at=ts,
        valid_until=valid_ts,
        ok=True,
        errors=[],
        scope_ok=True,
        rate_limit_ok=True,
        input_types_ok=True,
        rollback_ok=True,
    )


class RollbackManager:
    """Execute the rollback declared in an ActionContractSpec after a failed action."""

    def __init__(self, executor: ActionExecutor | None = None) -> None:
        self._executor = executor or ActionExecutor()

    def execute_rollback(
        self,
        original_trace: "ActionTrace",
        original_contract: ActionContractSpec,
        rollback_input: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> RollbackResult:
        if original_contract.rollback is None:
            return RollbackResult(
                attempted=False,
                ok=False,
                error="No rollback declared in contract",
                rollback_contract_name=None,
            )

        rollback_spec = original_contract.rollback
        rollback_contract = _rollback_spec_to_contract(rollback_spec, original_contract)
        dry_run = _build_rollback_dry_run(
            rollback_contract, rollback_input, original_trace, now=now
        )

        ctx = ExecutionContext(
            contract=rollback_contract,
            dry_run_report=dry_run,
            input_data=rollback_input,
            model_hash=original_trace.model_hash,
            parameter_set_id=original_trace.parameter_set_id,
            allow_real_actions=True,
            now=now,
        )

        try:
            result: ActionResult = self._executor.execute(ctx)
        except ActionExecutorError as exc:
            return RollbackResult(
                attempted=True,
                ok=False,
                error=str(exc),
                rollback_contract_name=rollback_spec.name,
            )

        return RollbackResult(
            attempted=True,
            ok=result.ok,
            error=result.error if not result.ok else None,
            rollback_contract_name=rollback_spec.name,
        )
