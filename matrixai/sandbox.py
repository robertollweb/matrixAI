# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matrixai.ir import ActionSpec, MatrixAIProgram


@dataclass(frozen=True)
class PermissionDecision:
    action: str
    allowed: bool
    policy: str
    call: str
    capabilities: list[str]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "allowed": self.allowed,
            "policy": self.policy,
            "call": self.call,
            "capabilities": list(self.capabilities),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SandboxReport:
    policy_name: str
    decisions: list[PermissionDecision]

    @property
    def ok(self) -> bool:
        return all(decision.allowed for decision in self.decisions)

    def messages(self) -> list[str]:
        return [reason for decision in self.decisions for reason in decision.reasons]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "policy": self.policy_name,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def summary(self) -> str:
        lines = [f"Sandbox policy: {self.policy_name}"]
        if not self.decisions:
            lines.append("No actions declared")
            return "\n".join(lines)
        for decision in self.decisions:
            status = "allowed" if decision.allowed else "blocked"
            capabilities = ", ".join(decision.capabilities) or "none"
            lines.append(
                f"- ACTION {decision.action}: {status} "
                f"({decision.policy}, {decision.call}, capabilities: {capabilities})"
            )
            for reason in decision.reasons:
                lines.append(f"  {reason}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SandboxPolicy:
    name: str = "mvp_simulate_only"
    allowed_policies: frozenset[str] = frozenset({"simulate_only"})
    require_simulated_calls: bool = True

    def review(self, program: MatrixAIProgram) -> SandboxReport:
        return SandboxReport(
            policy_name=self.name,
            decisions=[self.review_action(action) for action in program.actions],
        )

    def review_action(self, action: ActionSpec) -> PermissionDecision:
        capabilities = capabilities_for_call(action.call)
        reasons: list[str] = []

        if action.policy not in self.allowed_policies:
            allowed = ", ".join(sorted(self.allowed_policies))
            reasons.append(
                f"ACTION {action.name} policy '{action.policy}' is not allowed in MVP; "
                f"allowed policies: {allowed}"
            )
        if self.require_simulated_calls and "simulated" not in capabilities:
            reasons.append(
                f"ACTION {action.name} calls '{action.call}'. MVP requires simulated actions."
            )

        return PermissionDecision(
            action=action.name,
            allowed=not reasons,
            policy=action.policy,
            call=action.call,
            capabilities=capabilities,
            reasons=reasons,
        )

    @classmethod
    def mvp_simulate_only(cls) -> SandboxPolicy:
        return cls()


def capabilities_for_call(call: str) -> list[str]:
    parts = [part for part in call.split(".") if part]
    capabilities: list[str] = []

    if call.startswith("simulated."):
        capabilities.append("simulated")
        domain_parts = parts[1:]
    else:
        capabilities.append("external")
        domain_parts = parts

    domain_text = ".".join(domain_parts).lower()
    if any(token in domain_text for token in {"email", "mail", "smtp"}):
        capabilities.append("email")
    if any(token in domain_text for token in {"notify", "notification", "alert", "nurse"}):
        capabilities.append("notification")
    if any(token in domain_text for token in {"pharmacy", "dispense", "medication"}):
        capabilities.append("pharmacy")
    if any(token in domain_text for token in {"http", "https", "api", "webhook", "url"}):
        capabilities.append("network")
    if any(token in domain_text for token in {"db", "sql", "postgres", "database"}):
        capabilities.append("database")
    if any(token in domain_text for token in {"file", "fs", "path", "disk"}):
        capabilities.append("filesystem")

    return list(dict.fromkeys(capabilities))
