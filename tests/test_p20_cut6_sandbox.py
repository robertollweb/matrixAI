"""P20 C6 — SandboxedActionExecutor para capacidades de alto riesgo (POSIX)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from matrixai.actions import (
    SandboxedActionExecutor,
    SandboxedExecutorError,
    SandboxParams,
    SandboxResult,
    ActionExecutorError,
    ActionResult,
    DryRunSimulator,
    ExecutionContext,
    parse_mxact,
)
from matrixai.actions.contract import compute_action_contract_hash
from matrixai.actions.dryrun import _input_hash
from matrixai.actions.sandbox import _check_sandbox_scope
from matrixai.parser.parser import parse_text


# ── fixtures ───────────────────────────────────────────────────────────────────

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_FUTURE = _T0 + timedelta(minutes=3)

_DB_WRITE_MXACT = """
ACTION_CONTRACT WriteUserRecord
  CAPABILITY database_write
  SCOPE
    allowed_tables     = ["users", "audit_log"]
    allowed_operations = ["INSERT", "UPDATE"]
  END
  DRY_RUN required
  ROLLBACK undo_user_write
  SANDBOX required
  SANDBOX_LIMITS
    max_memory_mb    = 128
    max_cpu_seconds  = 10
    max_wall_seconds = 20
    no_network       = true
    allowed_env_vars = ["DATABASE_URL", "MATRIXAI_ENV"]
  END
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=50
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_user_write
  CAPABILITY database_write
  SCOPE
    allowed_tables     = ["audit_log"]
    allowed_operations = ["INSERT"]
  END
END
"""

_FS_WRITE_MXACT = """
ACTION_CONTRACT WriteReport
  CAPABILITY filesystem_write
  SCOPE
    allowed_paths = ["/tmp/reports/", "/var/matrixai/output/"]
  END
  DRY_RUN required
  ROLLBACK undo_fs_write
  SANDBOX required
  SANDBOX_LIMITS
    max_memory_mb    = 64
    max_cpu_seconds  = 5
    max_wall_seconds = 15
    no_network       = true
    allowed_env_vars = []
  END
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=2 per_hour=20
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_fs_write
  CAPABILITY filesystem_write
  SCOPE
    allowed_paths = ["/tmp/reports/"]
  END
END
"""

_PROG_SOURCE = """
PROJECT Test

VECTOR Input[1]
  x: Probability
END

ACTION WriteData
  TARGET database_write
  POLICY real_with_audit
  CONDITION x > 0.5
  INPUT table: String
END

GRAPH
  Input -> WriteData
END
"""


def _make_dry_run(contract, input_data, *, now=_T0):
    prog = parse_text(_PROG_SOURCE)
    sim = DryRunSimulator()
    return sim.simulate(
        contract, prog, "param_set_1", "model_hash_abc",
        input_data, now=now,
    )


def _make_context(contract, input_data, *, now=_T0, allow_real_actions=True):
    report = _make_dry_run(contract, input_data, now=now)
    return ExecutionContext(
        contract=contract,
        dry_run_report=report,
        input_data=input_data,
        model_hash="model_hash_abc",
        parameter_set_id="param_set_1",
        allow_real_actions=allow_real_actions,
        now=now,
    )


def _ok_runner(params: SandboxParams) -> SandboxResult:
    return SandboxResult(output='{"status": "ok"}', exit_code=0, error=None)


# ── 1. import and structural tests ────────────────────────────────────────────

def test_sandboxed_executor_importable():
    assert SandboxedActionExecutor is not None


def test_sandbox_params_dataclass():
    p = SandboxParams(
        capability="database_write",
        scope={"allowed_tables": ["users"]},
        input_data={"table": "users"},
        allowed_env_vars=["DATABASE_URL"],
        max_memory_mb=128,
        max_cpu_seconds=10,
        max_wall_seconds=20,
        no_network=True,
    )
    assert p.work_dir == "/tmp"
    assert p.no_network is True


def test_sandbox_result_dataclass():
    r = SandboxResult(output="hello", exit_code=0, error=None)
    assert r.timed_out is False


def test_sandboxed_executor_accepts_injectable_runner():
    executor = SandboxedActionExecutor(runner=_ok_runner)
    assert executor._runner is _ok_runner


# ── 2. injectable runner: correct SandboxParams are built ────────────────────

def test_sandbox_params_built_from_contract():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users", "data": {"name": "Alice"}}
    ctx = _make_context(contract, input_data)

    captured = {}
    def recording_runner(params: SandboxParams) -> SandboxResult:
        captured["params"] = params
        return SandboxResult(output="ok", exit_code=0, error=None)

    executor = SandboxedActionExecutor(runner=recording_runner)
    result = executor.execute(ctx)

    assert result.ok
    p = captured["params"]
    assert p.capability == "database_write"
    assert p.max_memory_mb == 128
    assert p.max_cpu_seconds == 10
    assert p.max_wall_seconds == 20
    assert p.no_network is True
    assert "DATABASE_URL" in p.allowed_env_vars


def test_sandbox_params_include_scope_and_input():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users", "data": {"id": 1}}
    ctx = _make_context(contract, input_data)

    captured = {}
    def recording_runner(params: SandboxParams) -> SandboxResult:
        captured["params"] = params
        return SandboxResult(output="ok", exit_code=0, error=None)

    SandboxedActionExecutor(runner=recording_runner).execute(ctx)
    p = captured["params"]
    assert "allowed_tables" in p.scope
    assert p.input_data["table"] == "users"


# ── 3. scope enforcement ──────────────────────────────────────────────────────

def test_scope_check_filesystem_blocks_disallowed_path():
    err = _check_sandbox_scope(
        "filesystem_write",
        {"allowed_paths": ["/tmp/reports/"]},
        {"path": "/etc/passwd"},
    )
    assert err is not None
    assert "/etc/passwd" in err


def test_scope_check_filesystem_allows_valid_path():
    err = _check_sandbox_scope(
        "filesystem_write",
        {"allowed_paths": ["/tmp/reports/"]},
        {"path": "/tmp/reports/output.txt"},
    )
    assert err is None


def test_scope_check_database_blocks_disallowed_table():
    err = _check_sandbox_scope(
        "database_write",
        {"allowed_tables": ["users", "audit_log"]},
        {"table": "secrets"},
    )
    assert err is not None
    assert "secrets" in err


def test_scope_check_database_allows_valid_table():
    err = _check_sandbox_scope(
        "database_write",
        {"allowed_tables": ["users"]},
        {"table": "users"},
    )
    assert err is None


def test_scope_check_unknown_capability_returns_none():
    err = _check_sandbox_scope("subprocess_spawn", {}, {"cmd": "ls"})
    assert err is None


def test_executor_returns_error_on_scope_violation():
    contracts = parse_mxact(_FS_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"path": "/etc/shadow", "content": "malicious"}
    ctx = _make_context(contract, input_data)

    executor = SandboxedActionExecutor(runner=_ok_runner)
    result = executor.execute(ctx)

    assert result.ok is False
    assert "/etc/shadow" in result.error


# ── 4. pre-flight checks ──────────────────────────────────────────────────────

def test_preflight_rejects_disabled_real_actions():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    ctx = _make_context(contract, input_data, allow_real_actions=False)

    with pytest.raises(ActionExecutorError, match="Real actions are disabled"):
        SandboxedActionExecutor(runner=_ok_runner).execute(ctx)


def test_preflight_rejects_expired_dry_run():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    # Dry-run created at old_t0; valid_until = old_t0 + 5 min.
    # Execution checked at old_t0 + 10 min → expired.
    old_t0 = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    ctx = _make_context(contract, input_data, now=old_t0)
    from dataclasses import replace as _replace
    ctx = _replace(ctx, now=old_t0 + timedelta(minutes=10))

    with pytest.raises(ActionExecutorError, match="expired"):
        SandboxedActionExecutor(runner=_ok_runner).execute(ctx)


def test_preflight_rejects_contract_hash_mismatch():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    ctx = _make_context(contract, input_data)

    other_contracts = parse_mxact(_FS_WRITE_MXACT)
    bad_ctx = replace(ctx, contract=other_contracts[0])

    with pytest.raises(ActionExecutorError, match="action_contract_hash"):
        SandboxedActionExecutor(runner=_ok_runner).execute(bad_ctx)


def test_preflight_rejects_input_hash_mismatch():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    ctx = _make_context(contract, input_data)

    tampered = replace(ctx, input_data={"table": "secrets"})

    with pytest.raises(ActionExecutorError, match="input_hash"):
        SandboxedActionExecutor(runner=_ok_runner).execute(tampered)


def test_preflight_rejects_non_high_risk_capability():
    """SandboxedActionExecutor must only accept HIGH_RISK capabilities."""
    low_risk_mxact = """
ACTION_CONTRACT FetchMetrics
  CAPABILITY http_get
  SCOPE
    allowed_urls = ["https://api.example.com/"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=10 per_hour=100
  SIGNATURE_REQUIRED false
END
"""
    contracts = parse_mxact(low_risk_mxact)
    contract = contracts[0]
    input_data = {"url": "https://api.example.com/metrics"}
    ctx = _make_context(contract, input_data)

    with pytest.raises(ActionExecutorError, match="not high-risk"):
        SandboxedActionExecutor(runner=_ok_runner).execute(ctx)


# ── 5. result mapping ─────────────────────────────────────────────────────────

def test_result_ok_on_exit_code_zero():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    ctx = _make_context(contract, input_data)

    executor = SandboxedActionExecutor(runner=_ok_runner)
    result = executor.execute(ctx)

    assert result.ok is True
    assert result.executor_kind == "sandbox"
    assert result.response_summary == '{"status": "ok"}'


def test_result_error_on_nonzero_exit():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    ctx = _make_context(contract, input_data)

    def failing_runner(params: SandboxParams) -> SandboxResult:
        return SandboxResult(output="", exit_code=1, error="permission denied")

    result = SandboxedActionExecutor(runner=failing_runner).execute(ctx)
    assert result.ok is False
    assert "permission denied" in result.error


def test_result_error_on_timeout():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    ctx = _make_context(contract, input_data)

    def timeout_runner(params: SandboxParams) -> SandboxResult:
        return SandboxResult(
            output="", exit_code=-9, error="Process exceeded max_wall_seconds and was terminated with SIGKILL",
            timed_out=True,
        )

    result = SandboxedActionExecutor(runner=timeout_runner).execute(ctx)
    assert result.ok is False
    assert "SIGKILL" in result.error


def test_result_has_positive_latency_ms():
    contracts = parse_mxact(_DB_WRITE_MXACT)
    contract = contracts[0]
    input_data = {"table": "users"}
    ctx = _make_context(contract, input_data)

    result = SandboxedActionExecutor(runner=_ok_runner).execute(ctx)
    assert result.latency_ms >= 0


# ── 6. default runner (POSIX-only) ────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
def test_default_runner_executes_simple_script():
    from matrixai.actions.sandbox import _default_sandbox_runner
    params = SandboxParams(
        capability="database_write",
        scope={"allowed_tables": ["users"]},
        input_data={"table": "users"},
        allowed_env_vars=[],
        max_memory_mb=None,
        max_cpu_seconds=None,
        max_wall_seconds=5,
        no_network=False,
    )
    result = _default_sandbox_runner(params)
    assert result.exit_code == 0
    assert "ok" in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
def test_default_runner_sets_network_env_when_no_network():
    from matrixai.actions.sandbox import _default_sandbox_runner
    import os

    captured_env = {}

    params = SandboxParams(
        capability="database_write",
        scope={},
        input_data={},
        allowed_env_vars=["MATRIXAI_NO_NETWORK"],
        max_memory_mb=None,
        max_cpu_seconds=None,
        max_wall_seconds=5,
        no_network=True,
    )
    result = _default_sandbox_runner(params)
    assert result.exit_code == 0
