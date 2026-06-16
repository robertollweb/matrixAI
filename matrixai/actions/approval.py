# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from matrixai.actions.dryrun import _input_hash as _hash_input

if TYPE_CHECKING:
    from matrixai.actions.executor import ExecutionContext

_DEFAULT_TTL_SECONDS = 300  # 5 minutes


@dataclass
class PendingExecution:
    execution_id: str
    contract_name: str
    capability: str
    input_data: dict
    input_hash: str           # hash of input_data at submit time
    model_hash: str
    parameter_set_id: str
    action_contract_hash: str
    created_at: str          # ISO8601 UTC
    expires_at: str          # ISO8601 UTC
    status: str = "pending"  # pending | approved | rejected | expired
    channel: str | None = None
    decided_at: str | None = None
    decided_by: str | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        ts = now or datetime.now(tz=timezone.utc)
        return ts >= datetime.fromisoformat(self.expires_at)

    def is_approved(self, now: datetime | None = None) -> bool:
        return self.status == "approved" and not self.is_expired(now)


class ApprovalStore:
    """In-memory store for PendingExecution records."""

    def __init__(self) -> None:
        self._store: dict[str, PendingExecution] = {}

    def submit(
        self,
        context: "ExecutionContext",
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        channel: str | None = None,
        now: datetime | None = None,
    ) -> PendingExecution:
        ts = now or datetime.now(tz=timezone.utc)
        execution_id = str(uuid.uuid4())
        pending = PendingExecution(
            execution_id=execution_id,
            contract_name=context.contract.name,
            capability=context.contract.capability,
            input_data=context.input_data,
            input_hash=context.dry_run_report.input_hash,
            model_hash=context.model_hash,
            parameter_set_id=context.parameter_set_id,
            action_contract_hash=context.dry_run_report.action_contract_hash,
            created_at=ts.isoformat(),
            expires_at=(ts + timedelta(seconds=ttl_seconds)).isoformat(),
            status="pending",
            channel=channel,
        )
        self._store[execution_id] = pending
        return pending

    def get(self, execution_id: str) -> PendingExecution | None:
        return self._store.get(execution_id)

    def approve(
        self,
        execution_id: str,
        *,
        decided_by: str | None = None,
        now: datetime | None = None,
    ) -> PendingExecution:
        pending = self._get_or_raise(execution_id)
        self._check_not_decided(pending)
        if pending.is_expired(now):
            pending.status = "expired"
            raise ApprovalError(f"Execution {execution_id!r} has expired")
        ts = now or datetime.now(tz=timezone.utc)
        pending.status = "approved"
        pending.decided_at = ts.isoformat()
        pending.decided_by = decided_by
        return pending

    def reject(
        self,
        execution_id: str,
        *,
        decided_by: str | None = None,
        now: datetime | None = None,
    ) -> PendingExecution:
        pending = self._get_or_raise(execution_id)
        self._check_not_decided(pending)
        ts = now or datetime.now(tz=timezone.utc)
        pending.status = "rejected"
        pending.decided_at = ts.isoformat()
        pending.decided_by = decided_by
        return pending

    def list_pending(self, now: datetime | None = None) -> list[PendingExecution]:
        ts = now or datetime.now(tz=timezone.utc)
        result = []
        for p in self._store.values():
            if p.status == "pending":
                if p.is_expired(ts):
                    p.status = "expired"
                else:
                    result.append(p)
        return result

    def _get_or_raise(self, execution_id: str) -> PendingExecution:
        p = self._store.get(execution_id)
        if p is None:
            raise ApprovalError(f"No pending execution found with id {execution_id!r}")
        return p

    @staticmethod
    def _check_not_decided(pending: PendingExecution) -> None:
        if pending.status in ("approved", "rejected"):
            raise ApprovalError(
                f"Execution {pending.execution_id!r} is already {pending.status}"
            )


class ApprovalError(Exception):
    pass


class HumanApprovalGate:
    """Check whether a context requires human approval and whether it has been granted."""

    def __init__(self, store: ApprovalStore) -> None:
        self._store = store

    def requires_approval(self, context: "ExecutionContext") -> bool:
        return context.contract.human_approval

    def find_approved(
        self,
        context: "ExecutionContext",
        now: datetime | None = None,
    ) -> PendingExecution | None:
        """Return the first approved, non-expired PendingExecution matching this context."""
        for pending in self._store._store.values():
            if (
                pending.action_contract_hash == context.dry_run_report.action_contract_hash
                and pending.input_hash == context.dry_run_report.input_hash
                and pending.model_hash == context.model_hash
                and pending.parameter_set_id == context.parameter_set_id
                and pending.is_approved(now)
            ):
                return pending
        return None

    def check(
        self,
        context: "ExecutionContext",
        now: datetime | None = None,
    ) -> bool:
        """Return True if approval is not required, or if a valid approval exists."""
        if not self.requires_approval(context):
            return True
        return self.find_approved(context, now) is not None
