# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


class IterationLimitReached(RuntimeError):
    """Raised when iteration_count exceeds max_iterations in RefinementAgent.refine()."""


@dataclass(frozen=True)
class RefinementProposal:
    refinement_id: str
    mode: str
    original_prompt: str
    proposed_prompt: str
    explanation: str
    supervision_accepted: bool
    supervision_report: dict[str, Any]
    hints_applied: list[str] = field(default_factory=list)
    iteration_count: int = 1
    refinement_chain: list[str] = field(default_factory=list)
    parent_prompt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RefinementAgent:
    """Proposes a refined prompt based on an audit explanation or evaluation report.

    The proposal always passes through PromptSupervisor before being returned.
    The agent only proposes — it never applies a change without human or CLI acceptance.
    """

    DEFAULT_MAX_ITERATIONS: int = 3

    def refine(
        self,
        prompt: str,
        *,
        semantic: str | None = None,
        mxai: str | None = None,
        audit: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        drift_report: dict[str, Any] | None = None,
        hints: list[str] | None = None,
        mode: str = "audit_driven",
        iteration_count: int = 1,
        refinement_chain: list[str] | None = None,
        parent_prompt_hash: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> RefinementProposal:
        if not prompt.strip():
            raise ValueError("RefinementAgent requires a non-empty prompt")
        if mode not in ("audit_driven", "metric_driven", "drift_driven"):
            raise ValueError(
                f"Unknown refinement mode: {mode!r}. "
                "Use 'audit_driven', 'metric_driven', or 'drift_driven'."
            )
        if mode == "audit_driven" and audit is None:
            raise ValueError("audit_driven mode requires an audit explanation dict (audit=)")
        if mode == "metric_driven" and evaluation is None:
            raise ValueError("metric_driven mode requires an evaluation report dict (evaluation=)")
        if mode == "drift_driven" and drift_report is None:
            raise ValueError("drift_driven mode requires a drift report dict (drift_report=)")
        if iteration_count > max_iterations:
            raise IterationLimitReached(
                f"Limite de iteraciones alcanzado: iteracion {iteration_count} supera el maximo "
                f"de {max_iterations}. Revisa los artefactos de iteraciones anteriores o "
                f"ajusta el limite con max_iterations={max_iterations + 1} o superior."
            )

        effective_parent_hash = parent_prompt_hash if parent_prompt_hash else _hash_prompt(prompt)
        prior_chain = list(refinement_chain) if refinement_chain else []

        if mode == "audit_driven":
            return self._refine_audit_driven(
                prompt,
                audit=audit,  # type: ignore[arg-type]
                semantic=semantic,
                mxai=mxai,
                hints=hints or [],
                iteration_count=iteration_count,
                prior_chain=prior_chain,
                parent_prompt_hash=effective_parent_hash,
            )
        if mode == "metric_driven":
            return self._refine_metric_driven(
                prompt,
                evaluation=evaluation,  # type: ignore[arg-type]
                semantic=semantic,
                mxai=mxai,
                hints=hints or [],
                iteration_count=iteration_count,
                prior_chain=prior_chain,
                parent_prompt_hash=effective_parent_hash,
            )
        # drift_driven
        return self._refine_drift_driven(
            prompt,
            drift_report=drift_report,  # type: ignore[arg-type]
            semantic=semantic,
            mxai=mxai,
            hints=hints or [],
            iteration_count=iteration_count,
            prior_chain=prior_chain,
            parent_prompt_hash=effective_parent_hash,
        )

    # ------------------------------------------------------------------
    # audit_driven
    # ------------------------------------------------------------------

    def _refine_audit_driven(
        self,
        prompt: str,
        *,
        audit: dict[str, Any],
        semantic: str | None,
        mxai: str | None,
        hints: list[str],
        iteration_count: int,
        prior_chain: list[str],
        parent_prompt_hash: str,
    ) -> RefinementProposal:
        from matrixai.agents.prompt_supervisor import PromptSupervisor

        derived_hints = self._analyze_audit(audit, mxai=mxai)
        all_hints = derived_hints + hints

        proposed_prompt = _build_refined_prompt(prompt, all_hints, iteration_count)
        explanation = _build_explanation(audit, all_hints)
        refinement_id = _make_refinement_id(prompt, proposed_prompt, "audit_driven", iteration_count)
        new_chain = prior_chain + [refinement_id]

        report = PromptSupervisor().supervise_prompt(proposed_prompt)
        enriched_mxai = _embed_refinement_metadata(report.mxai, new_chain, parent_prompt_hash)
        report = dataclasses.replace(
            report,
            mxai=enriched_mxai,
            refinement_chain=new_chain,
            parent_prompt_hash=parent_prompt_hash,
        )

        return RefinementProposal(
            refinement_id=refinement_id,
            mode="audit_driven",
            original_prompt=prompt,
            proposed_prompt=proposed_prompt,
            explanation=explanation,
            supervision_accepted=report.accepted,
            supervision_report=report.to_dict(),
            hints_applied=all_hints,
            iteration_count=iteration_count,
            refinement_chain=new_chain,
            parent_prompt_hash=parent_prompt_hash,
        )

    # ------------------------------------------------------------------
    # metric_driven
    # ------------------------------------------------------------------

    def _refine_metric_driven(
        self,
        prompt: str,
        *,
        evaluation: dict[str, Any],
        semantic: str | None,
        mxai: str | None,
        hints: list[str],
        iteration_count: int,
        prior_chain: list[str],
        parent_prompt_hash: str,
    ) -> RefinementProposal:
        from matrixai.agents.prompt_supervisor import PromptSupervisor

        derived_hints = self._analyze_evaluation(evaluation)
        all_hints = derived_hints + hints

        proposed_prompt = _build_refined_prompt(prompt, all_hints, iteration_count)
        explanation = _build_metric_explanation(evaluation, all_hints)
        refinement_id = _make_refinement_id(prompt, proposed_prompt, "metric_driven", iteration_count)
        new_chain = prior_chain + [refinement_id]

        report = PromptSupervisor().supervise_prompt(proposed_prompt)
        enriched_mxai = _embed_refinement_metadata(report.mxai, new_chain, parent_prompt_hash)
        report = dataclasses.replace(
            report,
            mxai=enriched_mxai,
            refinement_chain=new_chain,
            parent_prompt_hash=parent_prompt_hash,
        )

        return RefinementProposal(
            refinement_id=refinement_id,
            mode="metric_driven",
            original_prompt=prompt,
            proposed_prompt=proposed_prompt,
            explanation=explanation,
            supervision_accepted=report.accepted,
            supervision_report=report.to_dict(),
            hints_applied=all_hints,
            iteration_count=iteration_count,
            refinement_chain=new_chain,
            parent_prompt_hash=parent_prompt_hash,
        )

    # ------------------------------------------------------------------
    # drift_driven
    # ------------------------------------------------------------------

    def _refine_drift_driven(
        self,
        prompt: str,
        *,
        drift_report: dict[str, Any],
        semantic: str | None,
        mxai: str | None,
        hints: list[str],
        iteration_count: int,
        prior_chain: list[str],
        parent_prompt_hash: str,
    ) -> RefinementProposal:
        from matrixai.agents.prompt_supervisor import PromptSupervisor

        derived_hints = self._analyze_drift(drift_report)
        all_hints = derived_hints + hints

        proposed_prompt = _build_refined_prompt(prompt, all_hints, iteration_count)
        explanation = _build_drift_explanation(drift_report, all_hints)
        refinement_id = _make_refinement_id(prompt, proposed_prompt, "drift_driven", iteration_count)
        new_chain = prior_chain + [refinement_id]

        report = PromptSupervisor().supervise_prompt(proposed_prompt)
        enriched_mxai = _embed_refinement_metadata(report.mxai, new_chain, parent_prompt_hash)
        report = dataclasses.replace(
            report,
            mxai=enriched_mxai,
            refinement_chain=new_chain,
            parent_prompt_hash=parent_prompt_hash,
        )

        return RefinementProposal(
            refinement_id=refinement_id,
            mode="drift_driven",
            original_prompt=prompt,
            proposed_prompt=proposed_prompt,
            explanation=explanation,
            supervision_accepted=report.accepted,
            supervision_report=report.to_dict(),
            hints_applied=all_hints,
            iteration_count=iteration_count,
            refinement_chain=new_chain,
            parent_prompt_hash=parent_prompt_hash,
        )

    def _analyze_drift(self, drift_report: dict[str, Any]) -> list[str]:
        """Extract actionable hints from a DriftReport dict."""
        hints: list[str] = []
        results = drift_report.get("results", {})
        for feature, result in results.items():
            if result.get("skipped"):
                continue
            if result.get("drift_detected"):
                method = result.get("method", "unknown")
                observed = result.get("observed_value", 0.0)
                threshold = result.get("threshold", 0.0)
                hints.append(
                    f"Feature '{feature}' muestra drift significativo "
                    f"(metodo {method}: valor observado {observed:.4f}, umbral {threshold:.4f}); "
                    "considera actualizar el dataset de referencia o revisar la distribucion de entrada."
                )
        if not hints:
            hints.append(
                "El reporte de drift no indica features con desviacion significativa; "
                "revisa si los umbrales y metodos de deteccion son apropiados para este modelo."
            )
        return hints

    def _analyze_evaluation(self, evaluation: dict[str, Any]) -> list[str]:
        """Extract actionable hints from an evaluation_report dict."""
        hints: list[str] = []
        thresholds = evaluation.get("thresholds", {})

        accuracy = evaluation.get("accuracy")
        if accuracy is not None:
            min_acc = thresholds.get("accuracy", 0.8)
            if accuracy < min_acc:
                hints.append(
                    f"La precision del modelo es {accuracy:.4f}, por debajo del umbral {min_acc:.4f}; "
                    "considera ampliar el dataset de entrenamiento o ajustar las condiciones de disparo."
                )

        loss = evaluation.get("loss")
        if loss is not None:
            max_loss = thresholds.get("loss", 0.5)
            if loss > max_loss:
                hints.append(
                    f"La perdida del modelo es {loss:.4f}, por encima del maximo aceptable {max_loss:.4f}; "
                    "considera reducir el umbral de decision o revisar el balanceo del dataset."
                )

        metrics_by_label = evaluation.get("metrics_by_label", {})
        for label, metrics in metrics_by_label.items():
            f1 = metrics.get("f1")
            if f1 is not None and f1 < thresholds.get("f1", 0.7):
                hints.append(
                    f"La metrica F1 para la etiqueta '{label}' es {f1:.4f}; "
                    "considera agregar mas ejemplos de esta clase al dataset sintetico."
                )

        if not hints:
            hints.append(
                "El reporte de evaluacion no muestra metricas por debajo de los umbrales; "
                "revisa si los umbrales declarados son los correctos para este modelo."
            )

        return hints

    def _analyze_audit(self, audit: dict[str, Any], mxai: str | None = None) -> list[str]:
        """Extract actionable hints from structured audit report."""
        hints: list[str] = []
        actions = audit.get("actions", [])

        if not actions:
            hints.append(
                "El grafo no activo ninguna accion discreta; "
                "considera revisar las condiciones de disparo o los umbrales declarados en el modelo."
            )
            return hints

        for action in actions:
            name = action.get("name", "Unknown")
            source = action.get("source", "Unknown")
            value = action.get("value", 0.0)
            threshold = action.get("threshold", 0.0)
            activated = action.get("activated", False)

            context_str = ""
            if mxai and source != "Unknown":
                # Find line containing the source definition
                for line in mxai.splitlines():
                    if line.strip().startswith(f"{source} =") or line.strip().startswith(f"{source}:"):
                        context_str = f" Contexto actual en el modelo: `{line.strip()}`."
                        break

            if not activated:
                hints.append(
                    f"La accion {name} no se activo (valor {value:.4f}, umbral {threshold:.4f}); "
                    f"considera reducir el umbral o reforzar la condicion de {source} en el modelo.{context_str}"
                )
            else:
                hints.append(
                    f"La accion {name} se activo correctamente "
                    f"(valor {value:.4f}, umbral {threshold:.4f}); mantener la logica de {source}."
                )

        return hints


# ------------------------------------------------------------------
# Module-level helpers (stateless)
# ------------------------------------------------------------------

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


_META_CHAIN_PREFIX = "# refinement_chain:"
_META_HASH_PREFIX = "# parent_prompt_hash:"


def _embed_refinement_metadata(mxai_text: str, chain: list[str], parent_hash: str) -> str:
    """Prepend refinement metadata as # comment lines to a .mxai text block.

    The parser skips # lines, so the result remains valid .mxai syntax.
    """
    if not mxai_text.strip():
        return mxai_text
    chain_line = f"{_META_CHAIN_PREFIX} {','.join(chain)}"
    hash_line = f"{_META_HASH_PREFIX} {parent_hash}"
    return f"{chain_line}\n{hash_line}\n{mxai_text}"


def _parse_refinement_metadata(mxai_text: str) -> tuple[list[str], str]:
    """Read back refinement_chain and parent_prompt_hash embedded in a .mxai text.

    Returns (chain, parent_prompt_hash). Both empty if no metadata present.
    """
    chain: list[str] = []
    parent_hash = ""
    for line in mxai_text.splitlines():
        if line.startswith(_META_CHAIN_PREFIX):
            raw = line[len(_META_CHAIN_PREFIX):].strip()
            chain = [x.strip() for x in raw.split(",") if x.strip()]
        elif line.startswith(_META_HASH_PREFIX):
            parent_hash = line[len(_META_HASH_PREFIX):].strip()
    return chain, parent_hash


def _build_refined_prompt(prompt: str, hints: list[str], iteration_count: int = 1) -> str:
    if not hints:
        return prompt.strip()
    
    lines = [prompt.strip(), "", "<SystemFeedback>"]
    for hint in hints:
        lines.append(f"- {hint}")
    if iteration_count > 3:
        lines.append("- ADVERTENCIA: Multiples iteraciones sin exito. Considera cambiar el enfoque arquitectonico o relajar significativamente las restricciones.")
    lines.append("</SystemFeedback>")
    return "\n".join(lines)


def _build_explanation(audit: dict[str, Any], hints: list[str]) -> str:
    lines = ["Refinamiento basado en auditoria del modelo."]
    actions = audit.get("actions", [])
    if not actions:
        lines.append("Auditoria: No se activo ninguna accion.")
    else:
        activated = sum(1 for a in actions if a.get("activated"))
        lines.append(f"Auditoria: {activated}/{len(actions)} acciones activadas.")
        
    if hints:
        lines.append("Hints aplicados:")
        for h in hints:
            lines.append(f"  - {h}")
    return "\n".join(lines)


def _build_drift_explanation(drift_report: dict[str, Any], hints: list[str]) -> str:
    lines = ["Refinamiento basado en reporte de drift en produccion."]
    drift_detected = drift_report.get("drift_detected", False)
    features = drift_report.get("features_checked", [])
    results = drift_report.get("results", {})
    drifted = [f for f, r in results.items() if r.get("drift_detected")]
    lines.append(f"Features chequeadas: {len(features)}. Features con drift: {len(drifted)}.")
    if drift_detected:
        lines.append(f"Drift detectado en: {', '.join(drifted)}.")
    if hints:
        lines.append("Hints aplicados:")
        for h in hints:
            lines.append(f"  - {h}")
    return "\n".join(lines)


def _build_metric_explanation(evaluation: dict[str, Any], hints: list[str]) -> str:
    lines = ["Refinamiento basado en reporte de evaluacion."]
    accuracy = evaluation.get("accuracy")
    loss = evaluation.get("loss")
    if accuracy is not None:
        lines.append(f"Precision: {accuracy:.4f}.")
    if loss is not None:
        lines.append(f"Perdida: {loss:.4f}.")
    if hints:
        lines.append("Hints aplicados:")
        for h in hints:
            lines.append(f"  - {h}")
    return "\n".join(lines)


def _make_refinement_id(original_prompt: str, proposed_prompt: str, mode: str, iteration_count: int = 1) -> str:
    payload = json.dumps(
        {"original": original_prompt, "proposed": proposed_prompt, "mode": mode, "iteration": iteration_count},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"refinement_{mode[:5]}_{digest}"
