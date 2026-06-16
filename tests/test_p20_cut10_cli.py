"""P20 C10 — CLI: validate-actions, dry-run-action, execute-action, audit-action."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_PYTHON = sys.executable
_CLI = [_PYTHON, "-m", "matrixai"]

# ── sample files ──────────────────────────────────────────────────────────────

_EMAIL_MXACT = """\
ACTION_CONTRACT SendAlert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    allowed_domains    = ["example.com"]
  END
  DRY_RUN required
  ROLLBACK undo_alert
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_alert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    template = "correction"
  END
END
"""

_MXAI = """\
PROJECT Test

VECTOR Input[1]
  x: Probability
END

ACTION SendAlert
  TARGET email_send
  POLICY real_with_audit
  CONDITION x > 0.5
  INPUT recipient: String
END

GRAPH
  Input -> SendAlert
END
"""

_BAD_MXACT = """\
ACTION_CONTRACT SendAlert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    allowed_domains    = ["example.com"]
  END
  DRY_RUN required
  ROLLBACK undo_alert
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_alert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
  END
END
"""

_MXAI_WRONG_TARGET = """\
PROJECT Test

VECTOR Input[1]
  x: Probability
END

ACTION SendAlert
  TARGET http_get
  POLICY real_with_audit
  CONDITION x > 0.5
  INPUT url: String
END

GRAPH
  Input -> SendAlert
END
"""


def _run(args, cwd=None, env=None, input_text=None):
    return subprocess.run(
        _CLI + args,
        capture_output=True, text=True,
        cwd=cwd or "/home/deployer/matrixAI",
        env=env or os.environ.copy(),
        input=input_text,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _write(tmp, name, content):
    p = Path(tmp) / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ── 1. validate-actions ───────────────────────────────────────────────────────

def test_validate_actions_ok():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        r = _run(["validate-actions", mxact, mxai])
        assert r.returncode == 0
        assert "OK" in r.stdout


def test_validate_actions_json_output():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        r = _run(["validate-actions", mxact, mxai, "--json"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["all_ok"] is True
        assert data["contracts"][0]["contract"] == "SendAlert"


def test_validate_actions_fail_on_wrong_target():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "wrong.mxai", _MXAI_WRONG_TARGET)
        r = _run(["validate-actions", mxact, mxai, "--json"])
        assert r.returncode == 1
        data = json.loads(r.stdout)
        assert data["all_ok"] is False


def test_validate_actions_fail_on_bad_contract_file():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "bad.mxact", "NOT A CONTRACT")
        mxai = _write(tmp, "alert.mxai", _MXAI)
        r = _run(["validate-actions", mxact, mxai])
        assert r.returncode == 1


# ── 2. dry-run-action ─────────────────────────────────────────────────────────

def test_dry_run_action_ok():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        inp = json.dumps({"recipient": "ops@example.com"})
        r = _run(["dry-run-action", mxact, mxai,
                  "--contract-name", "SendAlert",
                  "--input", inp])
        assert r.returncode == 0
        assert "OK" in r.stdout


def test_dry_run_action_json_output():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        inp = json.dumps({"recipient": "ops@example.com"})
        r = _run(["dry-run-action", mxact, mxai,
                  "--contract-name", "SendAlert",
                  "--input", inp, "--json"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["ok"] is True
        assert "report_id" in data
        assert "action_contract_hash" in data


def test_dry_run_action_missing_contract_name():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        r = _run(["dry-run-action", mxact, mxai,
                  "--contract-name", "NonExistent"])
        assert r.returncode == 1
        assert "not found" in r.stderr


def test_dry_run_action_scope_violation():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        inp = json.dumps({"recipient": "hacker@evil.com"})
        r = _run(["dry-run-action", mxact, mxai,
                  "--contract-name", "SendAlert",
                  "--input", inp, "--json"])
        assert r.returncode == 1
        data = json.loads(r.stdout)
        assert data["ok"] is False


# ── 3. execute-action ─────────────────────────────────────────────────────────

def test_execute_action_requires_allow_flag():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        inp = json.dumps({"recipient": "ops@example.com"})
        r = _run(["execute-action", mxact, mxai,
                  "--contract-name", "SendAlert",
                  "--input", inp])
        assert r.returncode == 1
        assert "Real actions are disabled" in r.stderr


def test_execute_action_json_output_with_flag():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        inp = json.dumps({"recipient": "ops@example.com",
                          "subject": "Test", "body": "Hello"})
        r = _run(["execute-action", mxact, mxai,
                  "--contract-name", "SendAlert",
                  "--input", inp,
                  "--allow-real-actions", "--json"])
        data = json.loads(r.stdout)
        # full auditable trace emitted at top level
        for field in ("report_id", "model_hash", "parameter_set_id",
                      "action_contract_hash", "input_hash", "executed_at",
                      "executor_kind", "ok", "response_summary", "error",
                      "latency_ms", "hmac_signature"):
            assert field in data, f"missing field: {field}"


def test_execute_action_missing_contract():
    with tempfile.TemporaryDirectory() as tmp:
        mxact = _write(tmp, "alert.mxact", _EMAIL_MXACT)
        mxai = _write(tmp, "alert.mxai", _MXAI)
        r = _run(["execute-action", mxact, mxai,
                  "--contract-name", "NoSuchContract",
                  "--input", "{}", "--allow-real-actions"])
        assert r.returncode == 1
        assert "not found" in r.stderr


# ── 4. audit-action ───────────────────────────────────────────────────────────

def _make_trace_file(tmp, signing_key=None):
    trace_data = {
        "report_id": "rep-001",
        "model_hash": "model_abc",
        "parameter_set_id": "ps_1",
        "action_contract_hash": "sha256:" + "a" * 64,
        "input_hash": "sha256:" + "b" * 64,
        "executed_at": "2026-06-01T12:00:00+00:00",
        "executor_kind": "in_process",
        "ok": True,
        "response_summary": "sent",
        "error": None,
        "latency_ms": 10.0,
        "hmac_signature": None,
    }
    if signing_key:
        from matrixai.actions.trace import ActionTrace, sign_action_trace
        trace_obj = ActionTrace(**trace_data)
        trace_data["hmac_signature"] = sign_action_trace(trace_obj, signing_key)
    p = Path(tmp) / "trace.json"
    p.write_text(json.dumps(trace_data), encoding="utf-8")
    return str(p)


def test_audit_action_no_key_reports_not_verified():
    with tempfile.TemporaryDirectory() as tmp:
        trace_file = _make_trace_file(tmp)
        env = os.environ.copy()
        env.pop("MATRIXAI_ACTION_SIGNING_KEY", None)
        r = _run(["audit-action", trace_file], env=env)
        assert r.returncode == 0
        assert "not verified" in r.stdout


def test_audit_action_valid_signature():
    key = "a" * 64
    with tempfile.TemporaryDirectory() as tmp:
        trace_file = _make_trace_file(tmp, signing_key=key)
        env = os.environ.copy()
        env["MATRIXAI_ACTION_SIGNING_KEY"] = key
        r = _run(["audit-action", trace_file], env=env)
        assert r.returncode == 0
        assert "valid" in r.stdout


def test_audit_action_invalid_signature():
    key = "a" * 64
    wrong_key = "b" * 64
    with tempfile.TemporaryDirectory() as tmp:
        trace_file = _make_trace_file(tmp, signing_key=key)
        env = os.environ.copy()
        env["MATRIXAI_ACTION_SIGNING_KEY"] = wrong_key
        r = _run(["audit-action", trace_file], env=env)
        assert r.returncode == 1
        assert "MISMATCH" in r.stdout


def test_audit_action_json_output():
    key = "a" * 64
    with tempfile.TemporaryDirectory() as tmp:
        trace_file = _make_trace_file(tmp, signing_key=key)
        env = os.environ.copy()
        env["MATRIXAI_ACTION_SIGNING_KEY"] = key
        r = _run(["audit-action", trace_file, "--json"], env=env)
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["signature_verified"] is True
        assert "report_id" in data
