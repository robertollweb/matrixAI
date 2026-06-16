# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from matrixai.actions.registry import registry
from matrixai.actions.schema import (
    ActionContractSpec,
    CAPABILITIES,
    RateLimitSpec,
    RollbackSpec,
    SandboxLimitsSpec,
)
from matrixai.ir.schema import MatrixAIProgram


# ── canonical serialization ───────────────────────────────────────────────────

def _ser_rollback(rb: RollbackSpec | None) -> dict | None:
    if rb is None:
        return None
    return {
        "capability": rb.capability,
        "name": rb.name,
        "scope": dict(sorted(rb.scope.items())),
    }


def _ser_rate_limit(rl: RateLimitSpec | None) -> dict | None:
    if rl is None:
        return None
    return {"per_hour": rl.per_hour, "per_minute": rl.per_minute}


def _ser_sandbox_limits(sl: SandboxLimitsSpec | None) -> dict | None:
    if sl is None:
        return None
    return {
        "allowed_env_vars": sorted(sl.allowed_env_vars),
        "max_cpu_seconds": sl.max_cpu_seconds,
        "max_memory_mb": sl.max_memory_mb,
        "max_wall_seconds": sl.max_wall_seconds,
        "no_network": sl.no_network,
    }


def canonical_dict(spec: ActionContractSpec) -> dict:
    """Canonical dict for deterministic hashing — keys always sorted."""
    return {
        "approval_channel": spec.approval_channel,
        "approval_timeout_seconds": spec.approval_timeout_seconds,
        "capability": spec.capability,
        "dry_run_required": spec.dry_run_required,
        "human_approval": spec.human_approval,
        "name": spec.name,
        "rate_limit": _ser_rate_limit(spec.rate_limit),
        "risk_level": CAPABILITIES[spec.capability],
        "rollback": _ser_rollback(spec.rollback),
        "sandbox_limits": _ser_sandbox_limits(spec.sandbox_limits),
        "sandbox_required": spec.sandbox_required,
        "scope": dict(sorted(spec.scope.items())),
        "signature_required": spec.signature_required,
    }


def compute_action_contract_hash(spec: ActionContractSpec) -> str:
    """Return 'sha256:<hex>' of the canonical JSON of spec."""
    payload = json.dumps(canonical_dict(spec), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"sha256:{digest}"


# ── compatibility validation ──────────────────────────────────────────────────

@dataclass
class ActionContractValidationResult:
    ok: bool
    errors: list[str]


def validate_action_contract(
    spec: ActionContractSpec,
    program: MatrixAIProgram,
) -> ActionContractValidationResult:
    """Validate that spec is compatible with the .mxai program."""
    errors: list[str] = []

    matching = [a for a in program.actions if a.name == spec.name]
    if not matching:
        errors.append(f"Action {spec.name!r} not found in program actions")
        return ActionContractValidationResult(ok=False, errors=errors)

    action = matching[0]

    if action.target and action.target != spec.capability:
        errors.append(
            f"Action {spec.name!r} TARGET {action.target!r} does not match "
            f"contract CAPABILITY {spec.capability!r}"
        )

    if action.policy != "real_with_audit":
        errors.append(
            f"Action {spec.name!r} must have POLICY real_with_audit, got {action.policy!r}"
        )

    scope_result = registry.validate_scope(spec.capability, spec.scope)
    if not scope_result.ok:
        errors.extend(scope_result.errors)

    return ActionContractValidationResult(ok=len(errors) == 0, errors=errors)


# ── signing key check ─────────────────────────────────────────────────────────

def check_signing_key_available() -> bool:
    """Return True if MATRIXAI_ACTION_SIGNING_KEY is set in environment."""
    return bool(os.environ.get("MATRIXAI_ACTION_SIGNING_KEY", ""))


def require_signing_key(spec: ActionContractSpec) -> None:
    """Raise RuntimeError if spec requires a signing key that is not available."""
    if spec.signature_required and not check_signing_key_available():
        raise RuntimeError(
            f"ActionContract {spec.name!r} requires MATRIXAI_ACTION_SIGNING_KEY "
            f"but it is not set in the environment"
        )
