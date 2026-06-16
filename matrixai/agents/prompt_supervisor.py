# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai.agents.architect import ArchitectAgent
from matrixai.agents.planner_verifier import PlannerVerifier
from matrixai.agents.prompt import PromptAgent, PromptSynthesis
from matrixai.agents.safety import SafetyAgent
from matrixai.agents.verifier import VerifierAgent
from matrixai.compiler import PythonBackendCompiler
from matrixai.parser import parse_text


def _summarize_trace(trace: Any) -> dict[str, Any]:
    return {
        "attempt": trace.attempt,
        "latency_ms": round(trace.latency_ms, 1),
        "http_status": trace.http_status,
        "token_usage": trace.token_usage,
        "error": trace.error,
    }


@dataclass(frozen=True)
class SupervisionCheck:
    name: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSupervisionReport:
    accepted: bool
    prompt: str
    source: str
    semantic_text: str
    checks: list[SupervisionCheck]
    synthesis: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    mxai: str = ""
    program: dict[str, Any] | None = None
    compiled_python: str = ""
    supervision_source: str = "deterministic"
    llm_provider: str = ""
    llm_model: str = ""
    call_traces_summary: list[dict] = field(default_factory=list)
    fallback_reason: str = ""
    refinement_chain: list[str] = field(default_factory=list)
    parent_prompt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [f"PromptSupervisor: {status}", f"Source: {self.source}"]
        if self.plan:
            lines.append(f"Project: {self.plan['project']}")
        for check in self.checks:
            mark = "ok" if check.ok else "fail"
            lines.append(f"- {check.name}: {mark}")
            for warning in check.warnings:
                lines.append(f"  Warning: {warning}")
            for error in check.errors:
                lines.append(f"  Error: {error}")
        return "\n".join(lines)


class PromptSupervisor:
    """Deterministic acceptance gate for prompt-generated MatrixAI artifacts."""

    def supervise_prompt(self, prompt: str, *, force_deterministic: bool = False) -> PromptSupervisionReport:
        if force_deterministic:
            return self._deterministic_prompt_report(
                PromptAgent().synthesize(prompt),
                fallback_reason="forced_deterministic",
            )
        # Lazy import avoids circular dependency (llm_proposal imports from this module)
        from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider

        fallback_reason = ""
        try:
            provider = ChatCompletionsLLMProposalProvider.from_env()
            proposals, call_traces = provider.propose(prompt)
            if proposals:
                report = self.supervise_semantic(
                    prompt=prompt,
                    semantic_text=proposals[0].semantic_text,
                    source=f"llm:{provider.provider_name}",
                )
                llm_report = dataclasses.replace(
                    report,
                    supervision_source="llm",
                    llm_provider=provider.provider_name,
                    llm_model=provider.model_name,
                    call_traces_summary=[_summarize_trace(t) for t in call_traces],
                )
                deterministic_synthesis = PromptAgent().synthesize(prompt)
                if not llm_report.accepted:
                    return self._deterministic_prompt_report(
                        deterministic_synthesis,
                        fallback_reason="llm_rejected",
                    )
                if (
                    deterministic_synthesis.inferred_mode == "regression"
                    and (llm_report.plan or {}).get("mode") != "regression"
                ):
                    return self._deterministic_prompt_report(
                        deterministic_synthesis,
                        fallback_reason="llm_non_regression_for_continuous_prompt",
                    )
                return llm_report
        except ValueError:
            fallback_reason = "no_api_key"
        except RuntimeError as exc:
            fallback_reason = f"llm_error: {str(exc)[:120]}"

        return self._deterministic_prompt_report(
            PromptAgent().synthesize(prompt),
            fallback_reason=fallback_reason,
        )

    def _deterministic_prompt_report(
        self,
        synthesis: PromptSynthesis,
        *,
        fallback_reason: str,
    ) -> PromptSupervisionReport:
        report = self.supervise_semantic(
            prompt=synthesis.prompt,
            semantic_text=synthesis.semantic_text,
            source="PromptAgent",
            synthesis=synthesis,
        )
        return dataclasses.replace(
            report,
            supervision_source="deterministic",
            fallback_reason=fallback_reason,
        )

    def supervise_semantic(
        self,
        *,
        prompt: str,
        semantic_text: str,
        source: str = "semantic_proposal",
        synthesis: PromptSynthesis | None = None,
    ) -> PromptSupervisionReport:
        checks: list[SupervisionCheck] = []

        try:
            plan = ArchitectAgent().plan_from_text(semantic_text)
            checks.append(SupervisionCheck(name="architect_plan", ok=True))
        except Exception as exc:
            checks.append(
                SupervisionCheck(name="architect_plan", ok=False, errors=[str(exc)])
            )
            return self._report(False, prompt, source, semantic_text, checks, synthesis)

        plan_result = PlannerVerifier().verify(plan)
        checks.append(
            SupervisionCheck(
                name="planner_verifier",
                ok=plan_result.ok,
                errors=plan_result.errors,
                warnings=plan_result.warnings,
            )
        )

        math_unresolved = list(plan.mathematical_unresolved)
        checks.append(
            SupervisionCheck(
                name="mathematical_rules_resolved",
                ok=not math_unresolved,
                errors=math_unresolved,
                detail={"translations": plan.mathematical_translations},
            )
        )

        if not plan_result.ok or math_unresolved:
            return self._report(
                False,
                prompt,
                source,
                semantic_text,
                checks,
                synthesis,
                plan=plan.to_dict(),
            )

        mxai = ArchitectAgent().to_mxai(plan)
        try:
            program = parse_text(mxai)
            checks.append(SupervisionCheck(name="parser", ok=True))
        except Exception as exc:
            checks.append(SupervisionCheck(name="parser", ok=False, errors=[str(exc)]))
            return self._report(
                False,
                prompt,
                source,
                semantic_text,
                checks,
                synthesis,
                plan=plan.to_dict(),
                mxai=mxai,
            )

        verifier_result = VerifierAgent().verify(program)
        checks.append(
            SupervisionCheck(
                name="verifier_agent",
                ok=verifier_result.ok,
                errors=verifier_result.errors,
                warnings=verifier_result.warnings,
            )
        )

        safety_warnings = SafetyAgent().review(program)
        checks.append(
            SupervisionCheck(
                name="safety_agent",
                ok=not safety_warnings,
                errors=safety_warnings,
            )
        )

        if not verifier_result.ok or safety_warnings:
            return self._report(
                False,
                prompt,
                source,
                semantic_text,
                checks,
                synthesis,
                plan=plan.to_dict(),
                mxai=mxai,
                program=program.to_dict(),
            )

        try:
            compiled_python = PythonBackendCompiler().compile(program)
            checks.append(
                SupervisionCheck(
                    name="python_compiler",
                    ok=True,
                    detail={"bytes": len(compiled_python.encode("utf-8"))},
                )
            )
        except Exception as exc:
            checks.append(
                SupervisionCheck(name="python_compiler", ok=False, errors=[str(exc)])
            )
            return self._report(
                False,
                prompt,
                source,
                semantic_text,
                checks,
                synthesis,
                plan=plan.to_dict(),
                mxai=mxai,
                program=program.to_dict(),
            )

        return self._report(
            True,
            prompt,
            source,
            semantic_text,
            checks,
            synthesis,
            plan=plan.to_dict(),
            mxai=mxai,
            program=program.to_dict(),
            compiled_python=compiled_python,
        )

    def _report(
        self,
        accepted: bool,
        prompt: str,
        source: str,
        semantic_text: str,
        checks: list[SupervisionCheck],
        synthesis: PromptSynthesis | None,
        *,
        plan: dict[str, Any] | None = None,
        mxai: str = "",
        program: dict[str, Any] | None = None,
        compiled_python: str = "",
    ) -> PromptSupervisionReport:
        return PromptSupervisionReport(
            accepted=accepted,
            prompt=prompt,
            source=source,
            semantic_text=semantic_text,
            checks=checks,
            synthesis=synthesis.to_dict() if synthesis else None,
            plan=plan,
            mxai=mxai,
            program=program,
            compiled_python=compiled_python,
        )
