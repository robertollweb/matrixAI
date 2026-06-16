"""P20 C9 — Human-in-the-loop: PendingExecution, ApprovalStore, HumanApprovalGate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from matrixai.actions import (
    ApprovalError,
    ApprovalStore,
    DryRunSimulator,
    ExecutionContext,
    HumanApprovalGate,
    PendingExecution,
    parse_mxact,
)
from matrixai.parser.parser import parse_text

# ── fixtures ───────────────────────────────────────────────────────────────────

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_HUMAN_APPROVAL_MXACT = """
ACTION_CONTRACT SendCriticalAlert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ceo@example.com"]
    allowed_domains    = ["example.com"]
  END
  DRY_RUN required
  ROLLBACK undo_critical_alert
  SANDBOX not_required
  HUMAN_APPROVAL true
  APPROVAL_CHANNEL slack
  APPROVAL_TIMEOUT_SECONDS 300
  RATE_LIMIT per_minute=1 per_hour=5
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_critical_alert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ceo@example.com"]
    template = "retraction"
  END
END
"""

_NO_APPROVAL_MXACT = """
ACTION_CONTRACT SendRoutineAlert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    allowed_domains    = ["example.com"]
  END
  DRY_RUN required
  ROLLBACK undo_routine
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=10 per_hour=100
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_routine
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
  END
END
"""

_PROG = """
PROJECT Test

VECTOR Input[1]
  x: Probability
END

ACTION SendCriticalAlert
  TARGET email_send
  POLICY real_with_audit
  CONDITION x > 0.9
  INPUT recipient: String
END

GRAPH
  Input -> SendCriticalAlert
END
"""


def _make_context(mxact_src, input_data, *, now=_T0):
    contracts = parse_mxact(mxact_src)
    contract = contracts[0]
    prog = parse_text(_PROG)
    sim = DryRunSimulator()
    report = sim.simulate(contract, prog, "param_set_1", "model_hash_abc", input_data, now=now)
    return ExecutionContext(
        contract=contract,
        dry_run_report=report,
        input_data=input_data,
        model_hash="model_hash_abc",
        parameter_set_id="param_set_1",
        allow_real_actions=True,
    )


# ── 1. imports and structure ──────────────────────────────────────────────────

def test_pending_execution_importable():
    assert PendingExecution is not None


def test_approval_store_importable():
    assert ApprovalStore is not None


def test_human_approval_gate_importable():
    assert HumanApprovalGate is not None


# ── 2. ApprovalStore.submit ───────────────────────────────────────────────────

def test_submit_creates_pending_execution():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, now=_T0)

    assert isinstance(pending, PendingExecution)
    assert pending.status == "pending"
    assert pending.contract_name == "SendCriticalAlert"
    assert pending.capability == "email_send"


def test_submit_sets_expiry_from_ttl():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, ttl_seconds=60, now=_T0)

    expected_expires = (_T0 + timedelta(seconds=60)).isoformat()
    assert pending.expires_at == expected_expires


def test_submit_sets_channel():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, channel="slack", now=_T0)
    assert pending.channel == "slack"


def test_submit_stores_action_contract_hash():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, now=_T0)
    assert pending.action_contract_hash == ctx.dry_run_report.action_contract_hash


def test_get_returns_submitted_pending():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, now=_T0)
    retrieved = store.get(pending.execution_id)
    assert retrieved is pending


def test_get_returns_none_for_unknown_id():
    store = ApprovalStore()
    assert store.get("nonexistent-id") is None


# ── 3. approve / reject ───────────────────────────────────────────────────────

def test_approve_changes_status_to_approved():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, now=_T0)
    approved = store.approve(pending.execution_id, decided_by="admin", now=_T0)
    assert approved.status == "approved"
    assert approved.decided_by == "admin"


def test_reject_changes_status_to_rejected():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, now=_T0)
    rejected = store.reject(pending.execution_id, decided_by="admin", now=_T0)
    assert rejected.status == "rejected"


def test_approve_raises_on_already_decided():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, now=_T0)
    store.approve(pending.execution_id, now=_T0)
    with pytest.raises(ApprovalError, match="already approved"):
        store.approve(pending.execution_id, now=_T0)


def test_reject_raises_on_already_decided():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, now=_T0)
    store.reject(pending.execution_id, now=_T0)
    with pytest.raises(ApprovalError, match="already rejected"):
        store.reject(pending.execution_id, now=_T0)


def test_approve_raises_on_expired():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, ttl_seconds=10, now=_T0)
    future = _T0 + timedelta(seconds=60)
    with pytest.raises(ApprovalError, match="expired"):
        store.approve(pending.execution_id, now=future)


def test_approve_raises_on_unknown_id():
    store = ApprovalStore()
    with pytest.raises(ApprovalError, match="No pending execution"):
        store.approve("bad-id")


# ── 4. is_expired / is_approved ──────────────────────────────────────────────

def test_pending_not_expired_before_ttl():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, ttl_seconds=300, now=_T0)
    assert not pending.is_expired(_T0 + timedelta(seconds=100))


def test_pending_expired_after_ttl():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, ttl_seconds=60, now=_T0)
    assert pending.is_expired(_T0 + timedelta(seconds=120))


def test_is_approved_false_when_expired():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    pending = store.submit(ctx, ttl_seconds=60, now=_T0)
    store.approve(pending.execution_id, now=_T0)
    assert not pending.is_approved(_T0 + timedelta(seconds=120))


# ── 5. list_pending ───────────────────────────────────────────────────────────

def test_list_pending_returns_only_pending():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    p1 = store.submit(ctx, now=_T0)
    p2 = store.submit(ctx, now=_T0)
    store.approve(p1.execution_id, now=_T0)
    pending = store.list_pending(now=_T0)
    assert len(pending) == 1
    assert pending[0].execution_id == p2.execution_id


def test_list_pending_excludes_expired():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    store.submit(ctx, ttl_seconds=30, now=_T0)
    future = _T0 + timedelta(seconds=60)
    assert store.list_pending(now=future) == []


# ── 6. HumanApprovalGate ─────────────────────────────────────────────────────

def test_gate_requires_approval_when_human_approval_true():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    gate = HumanApprovalGate(store)
    assert gate.requires_approval(ctx) is True


def test_gate_does_not_require_approval_when_false():
    ctx = _make_context(_NO_APPROVAL_MXACT, {"recipient": "ops@example.com"})
    store = ApprovalStore()
    gate = HumanApprovalGate(store)
    assert gate.requires_approval(ctx) is False


def test_gate_check_true_when_no_approval_required():
    ctx = _make_context(_NO_APPROVAL_MXACT, {"recipient": "ops@example.com"})
    store = ApprovalStore()
    gate = HumanApprovalGate(store)
    assert gate.check(ctx, now=_T0) is True


def test_gate_check_false_when_approval_required_but_none_submitted():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    gate = HumanApprovalGate(store)
    assert gate.check(ctx, now=_T0) is False


def test_gate_check_true_after_approval():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    gate = HumanApprovalGate(store)
    pending = store.submit(ctx, now=_T0)
    store.approve(pending.execution_id, now=_T0)
    assert gate.check(ctx, now=_T0) is True


def test_gate_check_false_after_rejection():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    gate = HumanApprovalGate(store)
    pending = store.submit(ctx, now=_T0)
    store.reject(pending.execution_id, now=_T0)
    assert gate.check(ctx, now=_T0) is False


def test_gate_check_false_when_approval_expired():
    ctx = _make_context(_HUMAN_APPROVAL_MXACT, {"recipient": "ceo@example.com"})
    store = ApprovalStore()
    gate = HumanApprovalGate(store)
    pending = store.submit(ctx, ttl_seconds=60, now=_T0)
    store.approve(pending.execution_id, now=_T0)
    future = _T0 + timedelta(seconds=120)
    assert gate.check(ctx, now=future) is False
