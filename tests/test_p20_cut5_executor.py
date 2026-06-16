"""P20 C5 — ActionExecutor para acciones in-process de bajo y medio riesgo."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from matrixai.actions import (
    ActionExecutor,
    ActionExecutorError,
    ActionResult,
    DryRunSimulator,
    ExecutionContext,
    parse_mxact,
)
from matrixai.actions.dryrun import DryRunReport, _input_hash
from matrixai.actions.contract import compute_action_contract_hash
from matrixai.parser.parser import parse_text


# ── fixtures ───────────────────────────────────────────────────────────────────

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_FUTURE = _T0 + timedelta(minutes=3)  # within 5-min window

_EMAIL_MXACT = """
ACTION_CONTRACT SendNotification
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com", "alerts@example.com"]
    allowed_domains    = ["example.com"]
    max_subject_length = 200
  END
  DRY_RUN required
  ROLLBACK send_correction_email
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED false
END

ROLLBACK send_correction_email
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
    template = "correction"
  END
END
"""

_HTTP_GET_MXACT = """
ACTION_CONTRACT FetchData
  CAPABILITY http_get
  SCOPE
    allowed_urls = ["https://api.example.com/"]
    timeout_seconds = 5
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=10 per_hour=100
  SIGNATURE_REQUIRED false
END
"""

_HTTP_POST_MXACT = """
ACTION_CONTRACT CreateTicket
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://api.example.com/tickets"]
    required_headers = ["Authorization", "Content-Type"]
    timeout_seconds = 5
  END
  DRY_RUN required
  ROLLBACK undo_ticket
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=10 per_hour=100
  SIGNATURE_REQUIRED false
END

ROLLBACK undo_ticket
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://api.example.com/tickets/undo"]
  END
END
"""

_NOTIFICATION_MXACT = """
ACTION_CONTRACT SendWebhook
  CAPABILITY notification
  SCOPE
    allowed_recipients = ["ops@example.com"]
    timeout_seconds = 5
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=10 per_hour=100
  SIGNATURE_REQUIRED false
END
"""

_MXAI = """
PROJECT Test

VECTOR Input[1]
  x: Probability
END

ACTION SendNotification
  TARGET email_send
  POLICY real_with_audit
  CONDITION x > 0.5
  INPUT recipient: String, subject: String, body: String
END

GRAPH
  Input -> SendNotification
END
"""

_EMAIL_INPUT = {"recipient": "ops@example.com", "subject": "Alert", "body": "Test"}
_HTTP_GET_INPUT = {"url": "https://api.example.com/data"}
_HTTP_POST_INPUT = {
    "url": "https://api.example.com/tickets",
    "headers": {"Authorization": "Bearer tok", "Content-Type": "application/json"},
    "body": {"title": "bug"},
}
_NOTIFICATION_INPUT = {
    "recipient": "ops@example.com",
    "webhook_url": "https://hooks.example.com/notify",
}


def _make_dry_run(contract, input_data, now=_T0):
    sim = DryRunSimulator()
    prog = parse_text(_MXAI)
    # for non-email contracts, use a bare program
    return sim.simulate(contract, prog, "params-001", "sha256:modelabc",
                        input_data, now=now)


def _make_executor_email(ok_response="250 OK"):
    return ActionExecutor(email_fn=lambda *a, **kw: ok_response)


def _make_executor_http_get(ok_response="200 OK"):
    return ActionExecutor(http_get_fn=lambda url, timeout: ok_response)


def _make_executor_http_post(ok_response="201 Created"):
    return ActionExecutor(http_post_fn=lambda url, headers, payload, timeout: ok_response)


def _make_executor_notification(ok_response="200 OK"):
    return ActionExecutor(webhook_fn=lambda url, payload, timeout: ok_response)


# ── pre-flight: allow_real_actions ────────────────────────────────────────────

class TestExecutorPreFlight:
    def test_executor_blocks_without_allow_real_actions_flag(self, monkeypatch):
        monkeypatch.delenv("MATRIXAI_ACTION_SIGNING_KEY", raising=False)
        contract = parse_mxact(_EMAIL_MXACT)[0]
        dr = _make_dry_run(contract, _EMAIL_INPUT)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=_EMAIL_INPUT,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=False,
            now=_T0,
        )
        with pytest.raises(ActionExecutorError, match="allow-real-actions"):
            _make_executor_email().execute(ctx)

    def test_executor_blocks_without_signing_key(self, monkeypatch):
        monkeypatch.delenv("MATRIXAI_ACTION_SIGNING_KEY", raising=False)
        # build a contract that requires signature
        src = _EMAIL_MXACT.replace("SIGNATURE_REQUIRED false", "SIGNATURE_REQUIRED true")
        contract = parse_mxact(src)[0]
        dr = _make_dry_run(contract, _EMAIL_INPUT)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=_EMAIL_INPUT,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        with pytest.raises(ActionExecutorError, match="MATRIXAI_ACTION_SIGNING_KEY"):
            _make_executor_email().execute(ctx)

    def test_executor_blocks_without_valid_dry_run(self):
        contract = parse_mxact(_EMAIL_MXACT)[0]
        # create dry run anchored in 2020 so valid_until is genuinely in the past
        past = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dr = _make_dry_run(contract, _EMAIL_INPUT, now=past)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=_EMAIL_INPUT,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
        )
        with pytest.raises(ActionExecutorError, match="expired"):
            _make_executor_email().execute(ctx)

    def test_executor_blocks_expired_dry_run(self):
        contract = parse_mxact(_EMAIL_MXACT)[0]
        dr = _make_dry_run(contract, _EMAIL_INPUT, now=datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=_EMAIL_INPUT,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
        )
        with pytest.raises(ActionExecutorError, match="expired"):
            _make_executor_email().execute(ctx)

    def test_executor_blocks_dry_run_contract_hash_mismatch(self):
        contract = parse_mxact(_EMAIL_MXACT)[0]
        dr = _make_dry_run(contract, _EMAIL_INPUT, now=_T0)
        # tamper the contract hash in the report
        bad_dr = DryRunReport(
            report_id=dr.report_id, model_hash=dr.model_hash,
            parameter_set_id=dr.parameter_set_id,
            action_contract_hash="sha256:" + "0" * 64,
            input_hash=dr.input_hash, executed_at=dr.executed_at,
            valid_until=dr.valid_until, ok=dr.ok, errors=dr.errors,
            scope_ok=dr.scope_ok, rate_limit_ok=dr.rate_limit_ok,
            input_types_ok=dr.input_types_ok, rollback_ok=dr.rollback_ok,
        )
        ctx = ExecutionContext(
            contract=contract, dry_run_report=bad_dr, input_data=_EMAIL_INPUT,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        with pytest.raises(ActionExecutorError, match="action_contract_hash"):
            _make_executor_email().execute(ctx)

    def test_executor_blocks_dry_run_input_hash_mismatch(self):
        contract = parse_mxact(_EMAIL_MXACT)[0]
        dr = _make_dry_run(contract, _EMAIL_INPUT, now=_T0)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr,
            input_data={**_EMAIL_INPUT, "subject": "Changed after dry-run"},
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        with pytest.raises(ActionExecutorError, match="input_hash"):
            _make_executor_email().execute(ctx)


# ── http_get ───────────────────────────────────────────────────────────────────

class TestHttpGetExecutor:
    def _ctx(self, input_data=None):
        contract = parse_mxact(_HTTP_GET_MXACT)[0]
        data = input_data or _HTTP_GET_INPUT
        dr = _make_dry_run(contract, data, now=_T0)
        return ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=data,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )

    def test_http_get_executor_respects_scope_urls(self):
        ctx = self._ctx()
        result = _make_executor_http_get().execute(ctx)
        assert result.ok is True

    def test_http_get_executor_rejects_url_not_in_scope(self):
        bad_input = {"url": "https://evil.attacker.com/steal"}
        contract = parse_mxact(_HTTP_GET_MXACT)[0]
        dr = _make_dry_run(contract, bad_input, now=_T0)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=bad_input,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        result = _make_executor_http_get().execute(ctx)
        assert result.ok is False
        assert "allowed_urls" in result.error

    def test_http_get_executor_respects_timeout(self):
        calls = []
        def fake_get(url, timeout):
            calls.append(timeout)
            return "200 OK"
        ctx = self._ctx()
        ActionExecutor(http_get_fn=fake_get).execute(ctx)
        assert calls[0] == 5  # from scope timeout_seconds=5

    def test_action_result_ok_on_successful_http_get(self):
        ctx = self._ctx()
        result = _make_executor_http_get("200 OK").execute(ctx)
        assert result.ok is True
        assert result.response_summary == "200 OK"

    def test_action_result_has_latency_ms(self):
        ctx = self._ctx()
        result = _make_executor_http_get().execute(ctx)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0

    def test_action_result_has_executor_kind_in_process(self):
        ctx = self._ctx()
        result = _make_executor_http_get().execute(ctx)
        assert result.executor_kind == "in_process"


# ── http_post ──────────────────────────────────────────────────────────────────

class TestHttpPostExecutor:
    def _ctx(self, input_data=None):
        contract = parse_mxact(_HTTP_POST_MXACT)[0]
        data = input_data or _HTTP_POST_INPUT
        dr = _make_dry_run(contract, data, now=_T0)
        return ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=data,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )

    def test_http_post_executor_rejects_url_not_in_scope(self):
        bad = {**_HTTP_POST_INPUT, "url": "https://evil.com/post"}
        contract = parse_mxact(_HTTP_POST_MXACT)[0]
        dr = _make_dry_run(contract, bad, now=_T0)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=bad,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        result = _make_executor_http_post().execute(ctx)
        assert result.ok is False
        assert "allowed_urls" in result.error

    def test_http_post_executor_validates_required_headers(self):
        ctx = self._ctx()
        result = _make_executor_http_post().execute(ctx)
        assert result.ok is True

    def test_http_post_executor_rejects_missing_required_header(self):
        bad = {**_HTTP_POST_INPUT, "headers": {"Content-Type": "application/json"}}
        # missing Authorization
        contract = parse_mxact(_HTTP_POST_MXACT)[0]
        dr = _make_dry_run(contract, bad, now=_T0)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=bad,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        result = _make_executor_http_post().execute(ctx)
        assert result.ok is False
        assert "Authorization" in result.error


# ── email_send ─────────────────────────────────────────────────────────────────

class TestEmailExecutor:
    def _ctx(self, input_data=None):
        contract = parse_mxact(_EMAIL_MXACT)[0]
        data = input_data or _EMAIL_INPUT
        dr = _make_dry_run(contract, data, now=_T0)
        return ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=data,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )

    def test_email_executor_respects_allowed_recipients(self):
        ctx = self._ctx()
        result = _make_executor_email().execute(ctx)
        assert result.ok is True

    def test_email_executor_rejects_recipient_not_in_list(self):
        bad = {**_EMAIL_INPUT, "recipient": "unknown@example.com"}
        ctx = self._ctx(bad)
        result = _make_executor_email().execute(ctx)
        assert result.ok is False
        assert "allowed_recipients" in result.error

    def test_email_executor_rejects_domain_not_in_allowed_list(self):
        bad = {**_EMAIL_INPUT, "recipient": "ops@otherdomain.com"}
        # Note: otherdomain.com is not in allowed_recipients so hits that check first
        # Use a recipient that passes allowed_recipients but fails domain check
        # We need a contract where allowed_recipients is broader than allowed_domains
        src = _EMAIL_MXACT.replace(
            'allowed_recipients = ["ops@example.com", "alerts@example.com"]',
            'allowed_recipients = ["ops@example.com", "ops@forbidden.com"]',
        )
        contract = parse_mxact(src)[0]
        data = {**_EMAIL_INPUT, "recipient": "ops@forbidden.com"}
        dr = _make_dry_run(contract, data, now=_T0)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=data,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        result = _make_executor_email().execute(ctx)
        assert result.ok is False
        assert "allowed_domains" in result.error

    def test_executor_returns_action_result(self):
        ctx = self._ctx()
        result = _make_executor_email().execute(ctx)
        assert isinstance(result, ActionResult)

    def test_action_result_has_error_field_on_failure(self):
        bad = {**_EMAIL_INPUT, "recipient": "bad@evil.org"}
        ctx = self._ctx(bad)
        result = _make_executor_email().execute(ctx)
        assert result.ok is False
        assert result.error is not None


# ── notification ───────────────────────────────────────────────────────────────

class TestNotificationExecutor:
    def _ctx(self, input_data=None):
        contract = parse_mxact(_NOTIFICATION_MXACT)[0]
        data = input_data or _NOTIFICATION_INPUT
        dr = _make_dry_run(contract, data, now=_T0)
        return ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=data,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )

    def test_notification_executor_sends_to_webhook(self):
        calls = []
        def fake_webhook(url, payload, timeout):
            calls.append(url)
            return "200 OK"
        ctx = self._ctx()
        result = ActionExecutor(webhook_fn=fake_webhook).execute(ctx)
        assert result.ok is True
        assert len(calls) == 1

    def test_notification_executor_rejects_recipient_not_in_scope(self):
        bad = {**_NOTIFICATION_INPUT, "recipient": "hacker@evil.com"}
        ctx = self._ctx(bad)
        result = _make_executor_notification().execute(ctx)
        assert result.ok is False
        assert "allowed_recipients" in result.error


# ── high-risk blocked ─────────────────────────────────────────────────────────

class TestExecutorBlocksHighRisk:
    def test_executor_blocks_high_risk_capability(self):
        src = """
ACTION_CONTRACT WriteLog
  CAPABILITY filesystem_write
  SCOPE
    allowed_paths = ["/var/log/matrixai/"]
    allowed_extensions = [".log"]
  END
  DRY_RUN required
  ROLLBACK delete_log
  SANDBOX required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=60 per_hour=3600
  SIGNATURE_REQUIRED false
END

ROLLBACK delete_log
  CAPABILITY filesystem_write
  SCOPE
    allowed_paths = ["/var/log/matrixai/"]
    operation = "delete"
  END
END
"""
        contract = parse_mxact(src)[0]
        data = {"path": "/var/log/matrixai/test.log", "content": "log line"}
        dr = _make_dry_run(contract, data, now=_T0)
        ctx = ExecutionContext(
            contract=contract, dry_run_report=dr, input_data=data,
            model_hash="sha256:modelabc", parameter_set_id="params-001",
            allow_real_actions=True,
            now=_T0,
        )
        with pytest.raises(ActionExecutorError, match="SandboxedActionExecutor"):
            ActionExecutor().execute(ctx)
