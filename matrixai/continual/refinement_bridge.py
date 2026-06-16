# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P22 C10 — DriftRefinementBridge: connects DriftReport to RefinementAgent drift_driven mode."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from matrixai.ir.continual import ContinualPolicySpec

if TYPE_CHECKING:
    from matrixai.agents.refinement import RefinementProposal
    from matrixai.continual.drift import DriftReport


class DriftRefinementBridge:
    """Triggers a RefinementAgent proposal when sustained drift is detected.

    Respects ``AUDIT.EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT``.  When that
    flag is ``false``, :meth:`maybe_refine` always returns ``None``.

    When it is ``true`` and ``drift_report.drift_detected`` is ``True``,
    the bridge converts the report to a plain dict and calls
    :meth:`~matrixai.agents.refinement.RefinementAgent.refine` in
    ``"drift_driven"`` mode.

    Usage::

        bridge = DriftRefinementBridge(policy, prompt="Classify the email.")
        proposal = bridge.maybe_refine(drift_report)
        if proposal is not None:
            # inspect proposal.proposed_prompt, proposal.hints_applied, …
    """

    def __init__(self, policy: ContinualPolicySpec, *, prompt: str) -> None:
        self._policy = policy
        self._prompt = prompt

    def maybe_refine(
        self,
        drift_report: "DriftReport",
        *,
        drift_persistence_days: int | None = None,
        hints: list[str] | None = None,
        iteration_count: int = 1,
        refinement_chain: list[str] | None = None,
        parent_prompt_hash: str | None = None,
        max_iterations: int = 3,
    ) -> "RefinementProposal | None":
        """Return a RefinementProposal if drift conditions warrant it, else None.

        ``drift_persistence_days`` is the number of days drift has been continuously
        detected.  When provided, it is compared against
        ``AUDIT.REFINEMENT_DRIFT_PERSISTENCE_DAYS``; if the drift has not persisted
        long enough, the method returns ``None`` (no hint yet).
        """
        if not self._policy.audit.emit_refinement_hint_on_sustained_drift:
            return None
        if not drift_report.drift_detected:
            return None
        required = self._policy.audit.refinement_drift_persistence_days
        if required > 0:
            # When persistence is configured, None means "unknown" → block conservatively
            days = drift_persistence_days if drift_persistence_days is not None else 0
            if days < required:
                return None

        from matrixai.agents.refinement import RefinementAgent

        report_dict = _drift_report_to_dict(drift_report)
        return RefinementAgent().refine(
            self._prompt,
            mode="drift_driven",
            drift_report=report_dict,
            hints=hints,
            iteration_count=iteration_count,
            refinement_chain=refinement_chain,
            parent_prompt_hash=parent_prompt_hash,
            max_iterations=max_iterations,
        )


def _drift_report_to_dict(report: "DriftReport") -> dict[str, Any]:
    """Serialize a DriftReport dataclass to a plain dict for RefinementAgent."""
    return dataclasses.asdict(report)
