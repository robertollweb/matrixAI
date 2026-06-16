# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P22 C11 — Studio view for a ContinualPolicySpec operational status."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matrixai.ir.continual import ContinualPolicySpec


@dataclass
class ContinualPolicyView:
    """Flat, Studio-renderable snapshot of a ContinualPolicySpec."""
    name: str
    target_model: str
    registry_name: str | None
    base_version: str | None
    policy_hash: str
    # rollback config
    rollback_auto_trigger: bool
    rollback_metric: str
    rollback_sliding_window_hours: int
    rollback_degradation_threshold: float
    rollback_min_samples_in_window: int
    # audit config
    audit_persist_drift_reports: bool
    audit_persist_update_traces: bool
    audit_emit_refinement_hint: bool
    audit_refinement_drift_persistence_days: int
    audit_signature_required: bool
    # optional live state (injected externally)
    current_version: str | None = None
    current_parameter_set_id: str | None = None
    current_promoted_at: str | None = None


def build_continual_policy_view(
    policy: ContinualPolicySpec,
    *,
    current_entry: Any | None = None,
) -> ContinualPolicyView:
    """Build a ContinualPolicyView from a parsed policy and optional registry entry.

    ``current_entry`` is an optional ``RegistryEntry`` (from P21 ModelRegistry).
    When supplied, the view includes the current live version information.
    """
    rb = policy.rollback
    au = policy.audit

    current_version = None
    current_ps_id = None
    current_promoted_at = None
    if current_entry is not None:
        current_version = getattr(current_entry, "version", None)
        current_ps_id = getattr(current_entry, "parameter_set_id", None)
        metrics = getattr(current_entry, "metrics", {}) or {}
        current_promoted_at = metrics.get("promoted_at")

    return ContinualPolicyView(
        name=policy.name,
        target_model=policy.target_model,
        registry_name=policy.registry_name,
        base_version=policy.base_version,
        policy_hash=policy.policy_hash,
        rollback_auto_trigger=rb.auto_trigger,
        rollback_metric=rb.metric,
        rollback_sliding_window_hours=rb.sliding_window_hours,
        rollback_degradation_threshold=rb.degradation_threshold,
        rollback_min_samples_in_window=rb.min_samples_in_window,
        audit_persist_drift_reports=au.persist_drift_reports,
        audit_persist_update_traces=au.persist_update_traces,
        audit_emit_refinement_hint=au.emit_refinement_hint_on_sustained_drift,
        audit_refinement_drift_persistence_days=au.refinement_drift_persistence_days,
        audit_signature_required=au.signature_required,
        current_version=current_version,
        current_parameter_set_id=current_ps_id,
        current_promoted_at=current_promoted_at,
    )
