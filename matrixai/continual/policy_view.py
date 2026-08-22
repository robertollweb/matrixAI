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
    # ── DETECCIÓN DE DERIVA ─────────────────────────────────────────────
    #
    # No estaba, y el panel del Studio SÍ tiene fila para las tres cosas:
    # «Drift thresholds», «Window» y «Minimum samples» salían con un guion
    # **para cualquier modelo**, porque esta vista nunca las publicó.
    # Medido conduciendo el Workbench con una política real el
    # 2026-08-21: la política declara `MIN_SAMPLES 30`, `WINDOW_DAYS 7` y
    # tres umbrales, y la pantalla no enseñaba ninguno.
    #
    # Una fila que NUNCA puede tener valor es peor que no tenerla: se lee
    # como «este modelo no lo declara», que es falso.
    #
    # Van las tres cosas que decide la detección: sobre QUÉ mira, con qué
    # método y umbral cada una, y cuántas muestras exige antes de opinar.
    #: LISTA y no tupla: `dataclasses.asdict` conserva la tupla, así que el
    #: endpoint devolvía `('score',)` donde el CLI —que serializa a JSON—
    #: devuelve `['score']`. Coincidían solo DESPUÉS de pasar por HTTP, y
    #: la prueba de paridad entre el CLI y el endpoint lo cazó. Una
    #: diferencia que solo se ve en uno de los dos caminos es de las que
    #: muerden tarde.
    drift_features: list[str]
    #: `feature → "método threshold"`, ya redactado por el core: quien lo
    #: pinte no tiene que saber que `psi` y `ks` son métodos.
    drift_methods: dict[str, str]
    drift_min_samples: int
    drift_check_frequency: str
    drift_reference_dataset: str
    #: De la verdad-terreno, que es la otra mitad de «cuándo se puede
    #: opinar»: cuántos días de ventana y qué campo hace falta.
    ground_truth_window_days: int
    ground_truth_required_field: str
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
    dd = policy.drift_detection
    gt = policy.ground_truth
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
        drift_features=list(dd.features),
        # `psi threshold=0.20` en una sola cadena: el panel enseña lo que
        # el core decidió, sin recomponer nada.
        drift_methods={f: f"{m.method} threshold={m.threshold:g}"
                       for f, m in (dd.methods or {}).items()},
        drift_min_samples=dd.min_samples,
        drift_check_frequency=dd.check_frequency,
        drift_reference_dataset=dd.reference_dataset,
        ground_truth_window_days=gt.window_days,
        ground_truth_required_field=gt.required_field,
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
