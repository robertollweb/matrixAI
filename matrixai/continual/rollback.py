# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P22 C9 — RollbackManager: automatic rollback with signed RollbackEvent."""
from __future__ import annotations

import hashlib
import hmac as _hmac_module
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from matrixai.ir.continual import ContinualPolicySpec
from matrixai.registry.model_registry import ModelRegistry


# ── value objects ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RollbackEvent:
    """Auditable record of a rollback action."""
    rollback_id: str
    policy_hash: str
    trigger_reason: str        # "online_degradation" | "manual"
    metric: str
    sliding_window_value: float
    threshold: float
    from_parameter_set_id: str
    to_parameter_set_id: str
    from_version: str
    to_version: str
    rolled_back_at: str        # ISO8601 UTC
    samples_in_window: int
    notification_sent: bool
    signature: str | None      # hmac-sha256:... or sha256:... of canonical payload


@dataclass(frozen=True)
class RollbackCheckResult:
    """Outcome of a rollback check (does not execute the rollback)."""
    should_rollback: bool
    reason: str
    window_accuracy: float
    samples_in_window: int
    enough_samples: bool
    from_parameter_set_id: str
    to_parameter_set_id: str
    from_version: str
    to_version: str


# ── manager ───────────────────────────────────────────────────────────────────

class RollbackManager:
    """Checks for and executes automatic rollback against a P21 registry.

    ``check()`` inspects the current :class:`~matrixai.continual.monitor.ProductionMonitor`
    sliding-window metrics and determines whether a rollback is warranted:

    - ``ROLLBACK.AUTO_TRIGGER`` must be ``true``.
    - ``ROLLBACK.MIN_SAMPLES_IN_WINDOW`` must be satisfied.
    - The window accuracy must have degraded by more than
      ``ROLLBACK.DEGRADATION_THRESHOLD`` below the monitor's reference accuracy.

    ``execute()`` applies the rollback against the registry:

    1. Locates the entry currently tagged ``"current"`` for the policy's
       ``REGISTRY_NAME``.
    2. Reads ``entry.metrics["parent_parameter_set_id"]`` to find the previous
       parameter set.
    3. Finds the registry version whose ``parameter_set_id`` matches the parent.
    4. Updates the ``"current"`` tag to point to that version.
    5. Emits a :class:`RollbackEvent` (HMAC-signed when a signing key is
       provided, deterministic SHA-256 otherwise).

    ``run()`` is a convenience wrapper that calls ``check()`` and, if a
    rollback is warranted, ``execute()``.
    """

    def __init__(
        self,
        policy: ContinualPolicySpec,
        monitor: Any,           # ProductionMonitor — avoid circular import
        registry: ModelRegistry,
        *,
        signing_key: str | None = None,
        now: datetime | None = None,
        notification_fn: Callable[["RollbackEvent"], bool] | None = None,
    ) -> None:
        self._policy = policy
        self._monitor = monitor
        self._registry = registry
        self._signing_key = signing_key
        self._default_now = now
        self._notification_fn = notification_fn

    # ── public API ─────────────────────────────────────────────────────────────

    def check(self, now: datetime | None = None) -> RollbackCheckResult:
        """Evaluate whether a rollback should be triggered; does not modify state."""
        now_dt = _coerce_utc(now or self._default_now)
        rb = self._policy.rollback
        metrics = self._monitor.window_metrics(now=now_dt)

        _no = RollbackCheckResult(
            should_rollback=False,
            reason="",
            window_accuracy=metrics.accuracy,
            samples_in_window=metrics.samples,
            enough_samples=metrics.enough_samples,
            from_parameter_set_id="",
            to_parameter_set_id="",
            from_version="",
            to_version="",
        )

        if not rb.auto_trigger:
            return _no

        if not metrics.enough_samples:
            return _no

        if not metrics.degradation_detected:
            return _no

        # Determine from/to versions via registry
        registry_name = self._policy.registry_name
        if not registry_name:
            return _no

        from_entry, to_ps_id = self._find_rollback_targets(registry_name)
        if from_entry is None or not to_ps_id:
            return _no

        to_version = _find_version_by_ps_id(self._registry, registry_name, to_ps_id)
        if to_version is None:
            return _no

        return RollbackCheckResult(
            should_rollback=True,
            reason="online_degradation",
            window_accuracy=metrics.accuracy,
            samples_in_window=metrics.samples,
            enough_samples=True,
            from_parameter_set_id=from_entry.parameter_set_id,
            to_parameter_set_id=to_ps_id,
            from_version=from_entry.version,
            to_version=to_version,
        )

    def execute(
        self,
        from_parameter_set_id: str,
        to_parameter_set_id: str,
        from_version: str,
        to_version: str,
        *,
        window_value: float = 0.0,
        samples_in_window: int = 0,
        trigger_reason: str = "online_degradation",
        now: datetime | None = None,
    ) -> RollbackEvent:
        """Execute the rollback: update registry "current" tag and emit RollbackEvent."""
        now_dt = _coerce_utc(now or self._default_now)
        registry_name = self._policy.registry_name or ""
        rb = self._policy.rollback

        # Validate the target version exists
        self._registry.get(registry_name, to_version)   # raises EntryNotFoundError if missing

        # Update "current" tag
        self._registry.tag(registry_name, to_version, "current")

        notification_sent = False
        if self._policy.rollback.notify_capability and self._notification_fn is not None:
            event_draft = self._build_event(
                from_ps_id=from_parameter_set_id,
                to_ps_id=to_parameter_set_id,
                from_version=from_version,
                to_version=to_version,
                window_value=window_value,
                samples_in_window=samples_in_window,
                trigger_reason=trigger_reason,
                now_dt=now_dt,
                notification_sent=False,
            )
            try:
                notification_sent = bool(self._notification_fn(event_draft))
            except Exception:  # noqa: BLE001
                notification_sent = False

        return self._build_event(
            from_ps_id=from_parameter_set_id,
            to_ps_id=to_parameter_set_id,
            from_version=from_version,
            to_version=to_version,
            window_value=window_value,
            samples_in_window=samples_in_window,
            trigger_reason=trigger_reason,
            now_dt=now_dt,
            notification_sent=notification_sent,
        )

    def run(self, now: datetime | None = None) -> RollbackEvent | None:
        """Check and, if warranted, execute automatic rollback. Returns None if no action taken."""
        now_dt = _coerce_utc(now or self._default_now)
        result = self.check(now=now_dt)
        if not result.should_rollback:
            return None
        return self.execute(
            from_parameter_set_id=result.from_parameter_set_id,
            to_parameter_set_id=result.to_parameter_set_id,
            from_version=result.from_version,
            to_version=result.to_version,
            window_value=result.window_accuracy,
            samples_in_window=result.samples_in_window,
            trigger_reason=result.reason,
            now=now_dt,
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _find_rollback_targets(
        self, registry_name: str
    ) -> tuple[Any, str]:
        """Return (current_entry, parent_parameter_set_id) for the current version."""
        try:
            from_entry = self._registry.get(registry_name, "current")
        except Exception:  # noqa: BLE001
            return None, ""
        parent_ps_id = from_entry.metrics.get("parent_parameter_set_id", "")
        return from_entry, parent_ps_id

    def _build_event(
        self,
        *,
        from_ps_id: str,
        to_ps_id: str,
        from_version: str,
        to_version: str,
        window_value: float,
        samples_in_window: int,
        trigger_reason: str,
        now_dt: datetime,
        notification_sent: bool = False,
    ) -> RollbackEvent:
        rb = self._policy.rollback
        rolled_back_at = now_dt.isoformat()
        phash = self._policy.policy_hash
        ts = now_dt.strftime("%Y%m%d%H%M%S")
        phash_short = phash[7:15] if phash.startswith("sha256:") else phash[:8]
        rollback_id = f"rb-{ts}-{phash_short}"

        payload_for_sig = {
            "rollback_id": rollback_id,
            "policy_hash": phash,
            "trigger_reason": trigger_reason,
            "metric": rb.metric,
            "sliding_window_value": window_value,
            "threshold": rb.degradation_threshold,
            "from_parameter_set_id": from_ps_id,
            "to_parameter_set_id": to_ps_id,
            "from_version": from_version,
            "to_version": to_version,
            "rolled_back_at": rolled_back_at,
            "samples_in_window": samples_in_window,
            "notification_sent": notification_sent,
        }
        signature = _sign(payload_for_sig, self._signing_key)

        return RollbackEvent(
            rollback_id=rollback_id,
            policy_hash=phash,
            trigger_reason=trigger_reason,
            metric=rb.metric,
            sliding_window_value=window_value,
            threshold=rb.degradation_threshold,
            from_parameter_set_id=from_ps_id,
            to_parameter_set_id=to_ps_id,
            from_version=from_version,
            to_version=to_version,
            rolled_back_at=rolled_back_at,
            samples_in_window=samples_in_window,
            notification_sent=notification_sent,
            signature=signature,
        )


# ── registry helpers ──────────────────────────────────────────────────────────

def _find_version_by_ps_id(registry: ModelRegistry, name: str, ps_id: str) -> str | None:
    """Find the registry version whose parameter_set_id matches ps_id."""
    try:
        entries = registry.list({"name": name})
    except Exception:  # noqa: BLE001
        return None
    for entry in entries:
        if entry.parameter_set_id == ps_id:
            return entry.version
    return None


# ── signing helpers ───────────────────────────────────────────────────────────

def _sign(payload: dict[str, Any], signing_key: str | None) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if signing_key:
        key_bytes = bytes.fromhex(signing_key) if len(signing_key) == 64 else signing_key.encode()
        digest = _hmac_module.new(key_bytes, canon, hashlib.sha256).hexdigest()
        return f"hmac-sha256:{digest}"
    return "sha256:" + hashlib.sha256(canon).hexdigest()


# ── time helpers ──────────────────────────────────────────────────────────────

def _coerce_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
