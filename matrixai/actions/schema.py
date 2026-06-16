# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Capability registry: capability → risk level
CAPABILITIES: dict[str, str] = {
    "http_get": "low",
    "http_post": "medium",
    "notification": "low",
    "email_send": "medium",
    "database_read": "medium",
    "database_write": "high",
    "filesystem_read": "medium",
    "filesystem_write": "high",
    "subprocess_spawn": "high",
}

# Capabilities that require rollback when they mutate external state
MUTATING_CAPABILITIES: set[str] = {
    "http_post",
    "email_send",
    "database_write",
    "filesystem_write",
    "subprocess_spawn",
}

# Capabilities that require sandbox (risk = high)
HIGH_RISK_CAPABILITIES: set[str] = {
    k for k, v in CAPABILITIES.items() if v == "high"
}

# Scope keys whose list values must not contain bare wildcards
_WILDCARD_SCOPE_KEYS: set[str] = {"allowed_urls", "allowed_recipients", "allowed_paths"}
_FORBIDDEN_WILDCARD_VALUES: set[str] = {"*", "/", "**"}


@dataclass
class RateLimitSpec:
    per_minute: int
    per_hour: int


@dataclass
class SandboxLimitsSpec:
    max_memory_mb: int | None = None
    max_cpu_seconds: int | None = None
    max_wall_seconds: int | None = None
    no_network: bool = False
    allowed_env_vars: list[str] = field(default_factory=list)


@dataclass
class RollbackSpec:
    name: str
    capability: str
    scope: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionContractSpec:
    name: str
    capability: str
    scope: dict[str, Any]
    dry_run_required: bool
    rollback: RollbackSpec | None
    sandbox_required: bool
    sandbox_limits: SandboxLimitsSpec | None
    human_approval: bool
    approval_channel: str | None
    approval_timeout_seconds: int | None
    rate_limit: RateLimitSpec | None
    signature_required: bool


def _validate_scope_wildcards(scope: dict[str, Any], contract_name: str) -> None:
    for key in _WILDCARD_SCOPE_KEYS:
        val = scope.get(key)
        if val is None:
            continue
        items = val if isinstance(val, list) else [val]
        for item in items:
            if str(item) in _FORBIDDEN_WILDCARD_VALUES:
                raise ValueError(
                    f"ActionContract {contract_name}: scope.{key} contains forbidden "
                    f"wildcard {item!r}; use explicit values"
                )
