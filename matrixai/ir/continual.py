# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ── sub-specs ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroundTruthSpec:
    window_days: int
    required_field: str
    label_type: str | None          # e.g. "Label[support, sales, ops]"
    sources: tuple[str, ...]        # api, cli, file_watch
    file_watch_path: str | None


@dataclass(frozen=True)
class FeatureDriftMethodSpec:
    method: str     # psi | ks | chi_square | js | wasserstein
    threshold: float


VALID_DRIFT_METHODS: frozenset[str] = frozenset(
    {"psi", "ks", "chi_square", "js", "wasserstein"}
)


@dataclass(frozen=True)
class DriftDetectionSpec:
    features: tuple[str, ...]
    methods: dict[str, FeatureDriftMethodSpec]   # feature → method spec
    min_samples: int
    check_frequency: str            # daily | weekly
    reference_dataset: str          # base_training


@dataclass(frozen=True)
class ConceptDriftSpec:
    prediction_metric: str
    reference_value: float
    threshold_degradation: float
    min_samples_with_label: int


@dataclass(frozen=True)
class UpdateTriggerSpec:
    triggers: tuple[str, ...]       # drift_threshold_exceeded | concept_drift_detected | scheduled_<freq>
    min_new_samples: int
    min_ground_truth_ratio: float
    cooldown_days: int


@dataclass(frozen=True)
class RecencyDecaySpec:
    method: str             # exponential | linear
    half_life_days: int | None   # only for exponential


@dataclass(frozen=True)
class DatasetMixSpec:
    base_weight: float
    production_weight: float
    recency_decay: RecencyDecaySpec


@dataclass(frozen=True)
class EarlyStopSpec:
    patience: int
    metric: str


@dataclass(frozen=True)
class ContinualTrainingSpec:
    method: str                     # incremental_finetune | replay_buffer | full_retrain
    learning_rate_factor: float
    max_epochs: int
    early_stop: EarlyStopSpec | None
    dataset_mix: DatasetMixSpec
    seed_inheritance: str           # deterministic_from_parent


@dataclass(frozen=True)
class RegressionGuardSpec:
    metric: str
    must_improve_by: float
    max_degradation_per_label: float


@dataclass(frozen=True)
class ApprovalGateSpec:
    holdout_fraction: float
    holdout_source: str             # production_recent | base_training
    regression_guard: RegressionGuardSpec
    human_approval: bool
    approval_channel: str | None
    approval_timeout_hours: int | None


@dataclass(frozen=True)
class ContinualRollbackSpec:
    auto_trigger: bool
    metric: str
    sliding_window_hours: int
    degradation_threshold: float
    min_samples_in_window: int
    notify_capability: str | None
    notify_scope: dict[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class ContinualAuditSpec:
    persist_drift_reports: bool
    persist_update_traces: bool
    emit_refinement_hint_on_sustained_drift: bool
    refinement_drift_persistence_days: int
    signature_required: bool


@dataclass(frozen=True)
class ContinualPolicySpec:
    name: str
    target_model: str
    base_parameter_set: str
    registry_name: str | None
    base_version: str | None
    ground_truth: GroundTruthSpec
    drift_detection: DriftDetectionSpec
    concept_drift: ConceptDriftSpec | None
    update_trigger: UpdateTriggerSpec
    training: ContinualTrainingSpec
    approval_gate: ApprovalGateSpec
    rollback: ContinualRollbackSpec
    audit: ContinualAuditSpec
    policy_hash: str = field(default="", compare=False, hash=False)


# ── canonical serialization + hash ───────────────────────────────────────────

def _ser_ground_truth(gt: GroundTruthSpec) -> dict:
    return {
        "file_watch_path": gt.file_watch_path,
        "label_type": gt.label_type,
        "required_field": gt.required_field,
        "sources": sorted(gt.sources),
        "window_days": gt.window_days,
    }


def _ser_drift_method(m: FeatureDriftMethodSpec) -> dict:
    return {"method": m.method, "threshold": m.threshold}


def _ser_drift_detection(dd: DriftDetectionSpec) -> dict:
    return {
        "check_frequency": dd.check_frequency,
        "features": sorted(dd.features),
        "methods": {k: _ser_drift_method(v) for k, v in sorted(dd.methods.items())},
        "min_samples": dd.min_samples,
        "reference_dataset": dd.reference_dataset,
    }


def _ser_concept_drift(cd: ConceptDriftSpec | None) -> dict | None:
    if cd is None:
        return None
    return {
        "min_samples_with_label": cd.min_samples_with_label,
        "prediction_metric": cd.prediction_metric,
        "reference_value": cd.reference_value,
        "threshold_degradation": cd.threshold_degradation,
    }


def _ser_update_trigger(ut: UpdateTriggerSpec) -> dict:
    return {
        "cooldown_days": ut.cooldown_days,
        "min_ground_truth_ratio": ut.min_ground_truth_ratio,
        "min_new_samples": ut.min_new_samples,
        "triggers": sorted(ut.triggers),
    }


def _ser_recency_decay(rd: RecencyDecaySpec) -> dict:
    return {"half_life_days": rd.half_life_days, "method": rd.method}


def _ser_dataset_mix(dm: DatasetMixSpec) -> dict:
    return {
        "base_weight": dm.base_weight,
        "production_weight": dm.production_weight,
        "recency_decay": _ser_recency_decay(dm.recency_decay),
    }


def _ser_early_stop(es: EarlyStopSpec | None) -> dict | None:
    if es is None:
        return None
    return {"metric": es.metric, "patience": es.patience}


def _ser_training(t: ContinualTrainingSpec) -> dict:
    return {
        "dataset_mix": _ser_dataset_mix(t.dataset_mix),
        "early_stop": _ser_early_stop(t.early_stop),
        "learning_rate_factor": t.learning_rate_factor,
        "max_epochs": t.max_epochs,
        "method": t.method,
        "seed_inheritance": t.seed_inheritance,
    }


def _ser_regression_guard(rg: RegressionGuardSpec) -> dict:
    return {
        "max_degradation_per_label": rg.max_degradation_per_label,
        "metric": rg.metric,
        "must_improve_by": rg.must_improve_by,
    }


def _ser_approval_gate(ag: ApprovalGateSpec) -> dict:
    return {
        "approval_channel": ag.approval_channel,
        "approval_timeout_hours": ag.approval_timeout_hours,
        "holdout_fraction": ag.holdout_fraction,
        "holdout_source": ag.holdout_source,
        "human_approval": ag.human_approval,
        "regression_guard": _ser_regression_guard(ag.regression_guard),
    }


def _ser_rollback(rb: ContinualRollbackSpec) -> dict:
    return {
        "auto_trigger": rb.auto_trigger,
        "degradation_threshold": rb.degradation_threshold,
        "metric": rb.metric,
        "min_samples_in_window": rb.min_samples_in_window,
        "notify_capability": rb.notify_capability,
        "notify_scope": dict(sorted(rb.notify_scope.items())),
        "sliding_window_hours": rb.sliding_window_hours,
    }


def _ser_audit(a: ContinualAuditSpec) -> dict:
    return {
        "emit_refinement_hint_on_sustained_drift": a.emit_refinement_hint_on_sustained_drift,
        "persist_drift_reports": a.persist_drift_reports,
        "persist_update_traces": a.persist_update_traces,
        "refinement_drift_persistence_days": a.refinement_drift_persistence_days,
        "signature_required": a.signature_required,
    }


def canonical_dict(spec: ContinualPolicySpec) -> dict:
    """Canonical (sorted-keys) dict for deterministic hashing."""
    return {
        "approval_gate": _ser_approval_gate(spec.approval_gate),
        "audit": _ser_audit(spec.audit),
        "base_parameter_set": spec.base_parameter_set,
        "base_version": spec.base_version,
        "concept_drift": _ser_concept_drift(spec.concept_drift),
        "drift_detection": _ser_drift_detection(spec.drift_detection),
        "ground_truth": _ser_ground_truth(spec.ground_truth),
        "name": spec.name,
        "registry_name": spec.registry_name,
        "rollback": _ser_rollback(spec.rollback),
        "target_model": spec.target_model,
        "training": _ser_training(spec.training),
        "update_trigger": _ser_update_trigger(spec.update_trigger),
    }


def compute_policy_hash(spec: ContinualPolicySpec) -> str:
    """SHA-256 of canonical JSON serialization, prefixed sha256:<hex>."""
    payload = json.dumps(canonical_dict(spec), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"sha256:{digest}"
