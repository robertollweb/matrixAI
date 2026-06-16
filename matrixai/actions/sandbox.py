# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from matrixai.actions.contract import compute_action_contract_hash
from matrixai.actions.dryrun import DryRunReport, _input_hash
from matrixai.actions.executor import ActionExecutorError, ActionResult, ExecutionContext, common_preflight
from matrixai.actions.schema import ActionContractSpec, HIGH_RISK_CAPABILITIES

_MAX_OUTPUT_BYTES = 1024 * 1024  # 1 MB stdout/stderr cap


class SandboxedExecutorError(ActionExecutorError):
    pass


# ── sandbox params / result ───────────────────────────────────────────────────

@dataclass
class SandboxParams:
    capability: str
    scope: dict[str, Any]
    input_data: dict[str, Any]
    allowed_env_vars: list[str]
    max_memory_mb: int | None
    max_cpu_seconds: int | None
    max_wall_seconds: int | None
    no_network: bool
    work_dir: str = "/tmp"


@dataclass
class SandboxResult:
    output: str
    exit_code: int
    error: str | None
    timed_out: bool = False


# ── default POSIX runner ──────────────────────────────────────────────────────

def _default_sandbox_runner(params: SandboxParams) -> SandboxResult:
    if sys.platform == "win32":
        raise SandboxedExecutorError(
            "High-risk capabilities are not supported on Windows in P20; "
            "planned for P20.2 using Job Objects"
        )

    import resource

    filtered_env = {k: v for k, v in os.environ.items() if k in params.allowed_env_vars}
    if params.no_network:
        filtered_env["MATRIXAI_NO_NETWORK"] = "1"

    import json
    script = (
        "import sys, json\n"
        f"params = {json.dumps({'capability': params.capability, 'scope': params.scope, 'input': params.input_data})}\n"
        "print(json.dumps({'ok': True, 'params': params['capability']}))\n"
    )

    def _preexec() -> None:
        os.setsid()
        if params.max_memory_mb:
            mem = params.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        if params.max_cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU,
                               (params.max_cpu_seconds, params.max_cpu_seconds))

    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=filtered_env,
            shell=False,
            preexec_fn=_preexec,
            cwd=params.work_dir,
        )
    except OSError as exc:
        return SandboxResult(output="", exit_code=1, error=str(exc))

    wall = params.max_wall_seconds or 30
    try:
        stdout, stderr = proc.communicate(timeout=wall)
        out = stdout.decode(errors="replace")[:_MAX_OUTPUT_BYTES]
        err = stderr.decode(errors="replace")[:_MAX_OUTPUT_BYTES]
        return SandboxResult(output=out, exit_code=proc.returncode, error=err or None)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return SandboxResult(
            output="", exit_code=-9,
            error="Process exceeded max_wall_seconds and was terminated with SIGKILL",
            timed_out=True,
        )


# ── scope enforcement helpers ─────────────────────────────────────────────────

def _check_sandbox_scope(capability: str, scope: dict, input_data: dict) -> str | None:
    if capability in ("filesystem_write", "filesystem_read"):
        path = input_data.get("path", "")
        allowed = scope.get("allowed_paths", [])
        if allowed and not any(path.startswith(p) for p in allowed):
            return f"path {path!r} is not in scope allowed_paths"
    if capability in ("database_write", "database_read"):
        table = input_data.get("table", "")
        allowed = scope.get("allowed_tables", [])
        if allowed and table not in allowed:
            return f"table {table!r} is not in scope allowed_tables"
    return None


# ── sandboxed executor ────────────────────────────────────────────────────────

class SandboxedActionExecutor:
    """Subprocess-sandboxed executor for high-risk capabilities (POSIX)."""

    def __init__(self, runner: Callable[[SandboxParams], SandboxResult] | None = None) -> None:
        self._runner = runner or _default_sandbox_runner

    def execute(self, context: ExecutionContext) -> ActionResult:
        self._preflight(context)
        scope_error = _check_sandbox_scope(
            context.contract.capability, context.contract.scope, context.input_data
        )
        if scope_error:
            return ActionResult(ok=False, response_summary="", latency_ms=0,
                                error=scope_error, executor_kind="sandbox")
        return self._run(context)

    # ── pre-flight ────────────────────────────────────────────────────────────

    def _preflight(self, context: ExecutionContext) -> None:
        if sys.platform == "win32" and context.contract.capability in HIGH_RISK_CAPABILITIES:
            raise SandboxedExecutorError(
                "High-risk capabilities are not supported on Windows in P20"
            )
        # All standard checks (allow_real_actions, signing_key, human_approval, expiry, hashes)
        common_preflight(context)
        if context.contract.capability not in HIGH_RISK_CAPABILITIES:
            raise ActionExecutorError(
                f"Capability {context.contract.capability!r} is not high-risk; "
                f"use ActionExecutor instead"
            )

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _run(self, context: ExecutionContext) -> ActionResult:
        limits = context.contract.sandbox_limits
        params = SandboxParams(
            capability=context.contract.capability,
            scope=context.contract.scope,
            input_data=context.input_data,
            allowed_env_vars=limits.allowed_env_vars if limits else [],
            max_memory_mb=limits.max_memory_mb if limits else None,
            max_cpu_seconds=limits.max_cpu_seconds if limits else None,
            max_wall_seconds=limits.max_wall_seconds if limits else 30,
            no_network=limits.no_network if limits else False,
        )
        import time
        t0 = time.monotonic()
        try:
            result = self._runner(params)
        except SandboxedExecutorError:
            raise
        except Exception as exc:
            return ActionResult(ok=False, response_summary="", latency_ms=0,
                                error=str(exc), executor_kind="sandbox")
        latency = (time.monotonic() - t0) * 1000
        if result.timed_out:
            return ActionResult(ok=False, response_summary="", latency_ms=latency,
                                error=result.error, executor_kind="sandbox")
        ok = result.exit_code == 0
        return ActionResult(
            ok=ok,
            response_summary=result.output.strip(),
            latency_ms=latency,
            error=result.error if not ok else None,
            executor_kind="sandbox",
        )
