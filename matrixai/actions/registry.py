# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matrixai.actions.schema import (
    CAPABILITIES,
    HIGH_RISK_CAPABILITIES,
    MUTATING_CAPABILITIES,
)

# Required scope fields per capability
REQUIRED_SCOPE_FIELDS: dict[str, list[str]] = {
    "http_get":         ["allowed_urls"],
    "http_post":        ["allowed_urls"],
    "notification":     ["allowed_recipients"],
    "email_send":       ["allowed_recipients", "allowed_domains"],
    "database_read":    ["allowed_tables"],
    "database_write":   ["allowed_tables", "allowed_operations"],
    "filesystem_read":  ["allowed_paths"],
    "filesystem_write": ["allowed_paths"],
    "subprocess_spawn": [],  # declared case-by-case per rollback
}


@dataclass
class ScopeValidationResult:
    ok: bool
    errors: list[str]
    effective_scope: dict[str, Any] = field(default_factory=dict)


class CapabilityRegistry:
    """Central registry for MatrixAI action capabilities."""

    def list_capabilities(self) -> list[str]:
        return sorted(CAPABILITIES.keys())

    def risk_level(self, capability: str) -> str:
        self._assert_known(capability)
        return CAPABILITIES[capability]

    def is_high_risk(self, capability: str) -> bool:
        self._assert_known(capability)
        return capability in HIGH_RISK_CAPABILITIES

    def requires_rollback(self, capability: str) -> bool:
        self._assert_known(capability)
        return capability in MUTATING_CAPABILITIES

    def required_scope_fields(self, capability: str) -> list[str]:
        self._assert_known(capability)
        return list(REQUIRED_SCOPE_FIELDS[capability])

    def validate_scope(
        self, capability: str, scope: dict[str, Any]
    ) -> ScopeValidationResult:
        if capability not in CAPABILITIES:
            return ScopeValidationResult(
                ok=False,
                errors=[f"Unknown capability {capability!r}"],
            )
        errors: list[str] = []
        for field_name in REQUIRED_SCOPE_FIELDS[capability]:
            if field_name not in scope:
                errors.append(
                    f"Capability {capability!r} requires scope field {field_name!r}"
                )
        return ScopeValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            effective_scope=dict(scope) if not errors else {},
        )

    def resolve_scope(
        self, capability: str, scope: dict[str, Any]
    ) -> dict[str, Any]:
        """Return effective scope after validation; raises on invalid scope."""
        result = self.validate_scope(capability, scope)
        if not result.ok:
            raise ValueError(
                f"Invalid scope for {capability!r}: {'; '.join(result.errors)}"
            )
        return result.effective_scope

    def _assert_known(self, capability: str) -> None:
        if capability not in CAPABILITIES:
            raise ValueError(
                f"Unknown capability {capability!r}; "
                f"supported: {self.list_capabilities()}"
            )


# module-level singleton
registry = CapabilityRegistry()
