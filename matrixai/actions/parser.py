# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import json
import re
from typing import Any

from matrixai.actions.schema import (
    ActionContractSpec,
    RateLimitSpec,
    RollbackSpec,
    SandboxLimitsSpec,
    CAPABILITIES,
    HIGH_RISK_CAPABILITIES,
    MUTATING_CAPABILITIES,
    _validate_scope_wildcards,
)


class MxactParseError(ValueError):
    pass


# ── low-level helpers ─────────────────────────────────────────────────────────

def _clean_lines(source: str) -> list[str]:
    result = []
    for raw in source.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            result.append(line)
    return result


def _read_top_blocks(lines: list[str]) -> list[list[str]]:
    """Split cleaned lines into top-level ACTION_CONTRACT / ROLLBACK blocks."""
    _NESTED = {"SCOPE", "SANDBOX_LIMITS"}
    blocks: list[list[str]] = []
    current: list[str] = []
    depth = 0
    for line in lines:
        kw = line.split(maxsplit=1)[0]
        if depth == 0:
            if kw in ("ACTION_CONTRACT", "ROLLBACK"):
                depth = 1
                current = [line]
        else:
            current.append(line)
            if kw in _NESTED:
                depth += 1
            elif line == "END":
                depth -= 1
                if depth == 0:
                    blocks.append(current)
                    current = []
    return blocks


# ── scope value parser ────────────────────────────────────────────────────────

_SCOPE_KV_RE = re.compile(r"^(?P<key>[A-Za-z_]\w*)\s*=\s*(?P<val>.+)$")


def _parse_scope_value(raw: str) -> Any:
    raw = raw.strip()
    # JSON array or object
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    # bool
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    # quoted string
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    # int
    try:
        return int(raw)
    except ValueError:
        pass
    # float
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_scope_block(scope_lines: list[str]) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    for line in scope_lines:
        m = _SCOPE_KV_RE.match(line)
        if not m:
            raise MxactParseError(f"Invalid SCOPE line: {line!r}")
        scope[m.group("key")] = _parse_scope_value(m.group("val"))
    return scope


def _extract_sub_block(lines: list[str], keyword: str) -> tuple[list[str], list[str]]:
    """Return (sub_block_body_lines, remaining_lines) for a KEYWORD...END sub-block."""
    _NESTED = {"SCOPE", "SANDBOX_LIMITS"}
    body: list[str] = []
    remaining: list[str] = []
    i = 0
    # find keyword line
    while i < len(lines) and lines[i].split(maxsplit=1)[0] != keyword:
        remaining.append(lines[i])
        i += 1
    if i >= len(lines):
        return [], lines  # keyword not found
    depth = 1
    i += 1
    while i < len(lines) and depth > 0:
        line = lines[i]
        kw = line.split(maxsplit=1)[0]
        if kw in _NESTED:
            depth += 1
            body.append(line)
        elif line == "END":
            depth -= 1
            if depth > 0:
                body.append(line)
        else:
            body.append(line)
        i += 1
    remaining.extend(lines[i:])
    return body, remaining


# ── rollback block parser ─────────────────────────────────────────────────────

def _parse_rollback_block(block: list[str]) -> RollbackSpec:
    name = block[0].removeprefix("ROLLBACK ").strip()
    capability = ""
    scope: dict[str, Any] = {}
    inner = block[1:-1]  # strip ROLLBACK header and final END
    scope_body, flat = _extract_sub_block(inner, "SCOPE")
    if scope_body is not None and scope_body != []:
        scope = _parse_scope_block(scope_body)
    for line in flat:
        if line.startswith("CAPABILITY "):
            capability = line.removeprefix("CAPABILITY ").strip()
    if not capability:
        raise MxactParseError(f"ROLLBACK {name} requires CAPABILITY")
    return RollbackSpec(name=name, capability=capability, scope=scope)


# ── sandbox limits parser ─────────────────────────────────────────────────────

_RATE_PART_RE = re.compile(r"per_minute=(?P<pm>\d+)\s+per_hour=(?P<ph>\d+)")


def _parse_sandbox_limits(body: list[str]) -> SandboxLimitsSpec:
    spec = SandboxLimitsSpec()
    for line in body:
        kw = line.split(maxsplit=1)[0]
        if kw == "max_memory_mb":
            spec.max_memory_mb = int(line.split("=")[1].strip())
        elif kw == "max_cpu_seconds":
            spec.max_cpu_seconds = int(line.split("=")[1].strip())
        elif kw == "max_wall_seconds":
            spec.max_wall_seconds = int(line.split("=")[1].strip())
        elif kw == "no_network":
            spec.no_network = line.split("=")[1].strip().lower() == "true"
        elif kw == "allowed_env_vars":
            raw = line.split("=", 1)[1].strip()
            spec.allowed_env_vars = json.loads(raw) if raw.startswith("[") else [raw.strip('"').strip("'")]
    return spec


# ── contract block parser ─────────────────────────────────────────────────────

def _parse_contract_block(
    block: list[str], rollback_specs: dict[str, RollbackSpec]
) -> ActionContractSpec:
    name = block[0].removeprefix("ACTION_CONTRACT ").strip()
    inner = block[1:-1]

    # extract SCOPE sub-block
    scope_body, inner = _extract_sub_block(inner, "SCOPE")
    scope = _parse_scope_block(scope_body) if scope_body else {}

    # extract SANDBOX_LIMITS sub-block
    sb_body, inner = _extract_sub_block(inner, "SANDBOX_LIMITS")
    sandbox_limits = _parse_sandbox_limits(sb_body) if sb_body else None

    # parse flat fields
    capability = ""
    dry_run_required = True
    rollback_name: str | None = None
    sandbox_required = False
    human_approval = False
    approval_channel: str | None = None
    approval_timeout: int | None = None
    rate_limit: RateLimitSpec | None = None
    signature_required = True

    for line in inner:
        if line.startswith("CAPABILITY "):
            capability = line.removeprefix("CAPABILITY ").strip()
        elif line.startswith("DRY_RUN "):
            val = line.removeprefix("DRY_RUN ").strip()
            dry_run_required = val == "required"
        elif line.startswith("ROLLBACK "):
            rollback_name = line.removeprefix("ROLLBACK ").strip()
        elif line.startswith("SANDBOX "):
            val = line.removeprefix("SANDBOX ").strip()
            sandbox_required = val == "required"
        elif line.startswith("HUMAN_APPROVAL "):
            val = line.removeprefix("HUMAN_APPROVAL ").strip()
            human_approval = val.lower() == "true"
        elif line.startswith("APPROVAL_CHANNEL "):
            approval_channel = line.removeprefix("APPROVAL_CHANNEL ").strip()
        elif line.startswith("APPROVAL_TIMEOUT_SECONDS "):
            approval_timeout = int(line.removeprefix("APPROVAL_TIMEOUT_SECONDS ").strip())
        elif line.startswith("RATE_LIMIT "):
            m = _RATE_PART_RE.search(line)
            if m:
                rate_limit = RateLimitSpec(
                    per_minute=int(m.group("pm")),
                    per_hour=int(m.group("ph")),
                )
        elif line.startswith("SIGNATURE_REQUIRED "):
            val = line.removeprefix("SIGNATURE_REQUIRED ").strip()
            signature_required = val.lower() == "true"

    # semantic validation
    if not capability:
        raise MxactParseError(f"ACTION_CONTRACT {name} requires CAPABILITY")
    if capability not in CAPABILITIES:
        raise MxactParseError(
            f"ACTION_CONTRACT {name}: unknown capability {capability!r}; "
            f"supported: {sorted(CAPABILITIES)}"
        )
    if capability in HIGH_RISK_CAPABILITIES and not sandbox_required:
        raise MxactParseError(
            f"ACTION_CONTRACT {name}: capability {capability!r} is high-risk and requires "
            f"SANDBOX required"
        )
    if human_approval and not approval_channel:
        raise MxactParseError(
            f"ACTION_CONTRACT {name}: HUMAN_APPROVAL true requires APPROVAL_CHANNEL"
        )
    if capability in MUTATING_CAPABILITIES and rollback_name is None:
        raise MxactParseError(
            f"ACTION_CONTRACT {name}: capability {capability!r} mutates state and requires ROLLBACK"
        )
    try:
        _validate_scope_wildcards(scope, name)
    except ValueError as exc:
        raise MxactParseError(str(exc)) from exc

    # resolve rollback
    rollback: RollbackSpec | None = None
    if rollback_name and rollback_name.lower() != "none":
        if rollback_name not in rollback_specs:
            raise MxactParseError(
                f"ACTION_CONTRACT {name}: ROLLBACK {rollback_name!r} not defined in this file"
            )
        rollback = rollback_specs[rollback_name]

    return ActionContractSpec(
        name=name,
        capability=capability,
        scope=scope,
        dry_run_required=dry_run_required,
        rollback=rollback,
        sandbox_required=sandbox_required,
        sandbox_limits=sandbox_limits,
        human_approval=human_approval,
        approval_channel=approval_channel,
        approval_timeout_seconds=approval_timeout,
        rate_limit=rate_limit,
        signature_required=signature_required,
    )


# ── public entry point ────────────────────────────────────────────────────────

def parse_mxact(source: str) -> list[ActionContractSpec]:
    """Parse a .mxact source string and return a list of ActionContractSpec."""
    lines = _clean_lines(source)
    if not lines:
        raise MxactParseError("Empty .mxact document")
    blocks = _read_top_blocks(lines)
    if not blocks:
        raise MxactParseError("No ACTION_CONTRACT or ROLLBACK blocks found")

    # first pass: collect rollback definitions
    rollback_specs: dict[str, RollbackSpec] = {}
    for block in blocks:
        if block[0].startswith("ROLLBACK "):
            rb = _parse_rollback_block(block)
            rollback_specs[rb.name] = rb

    # second pass: parse contracts
    contracts: list[ActionContractSpec] = []
    for block in blocks:
        if block[0].startswith("ACTION_CONTRACT "):
            contracts.append(_parse_contract_block(block, rollback_specs))

    if not contracts:
        raise MxactParseError("No ACTION_CONTRACT found in .mxact document")
    return contracts
