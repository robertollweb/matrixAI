# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from matrixai.actions.contract import check_signing_key_available, compute_action_contract_hash
from matrixai.actions.dryrun import DryRunReport, _input_hash
from matrixai.actions.schema import ActionContractSpec, HIGH_RISK_CAPABILITIES

if TYPE_CHECKING:
    from matrixai.actions.approval import ApprovalStore


class ActionExecutorError(Exception):
    pass


@dataclass
class ActionResult:
    ok: bool
    response_summary: str
    latency_ms: float
    error: str | None
    executor_kind: str = "in_process"


@dataclass
class ExecutionContext:
    contract: ActionContractSpec
    dry_run_report: DryRunReport
    input_data: dict[str, Any]
    model_hash: str
    parameter_set_id: str
    allow_real_actions: bool = False
    signing_key: str | None = None          # passed by CLI/server, not read from env
    approval_store: "ApprovalStore | None" = field(default=None, repr=False)
    # Injected clock for testing — None means use datetime.now(utc)
    now: datetime | None = field(default=None, repr=False)


# ── capability dispatchers (overrideable for testing) ─────────────────────────

def _default_http_get(url: str, timeout: int) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return f"{resp.status} {resp.reason}"


def _default_http_post(url: str, headers: dict, payload: bytes, timeout: int) -> str:
    import urllib.request
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return f"{resp.status} {resp.reason}"


def _default_email_send(
    smtp_host: str, smtp_port: int,
    smtp_user: str, smtp_pass: str,
    recipient: str, subject: str, body: str,
) -> str:
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    with smtplib.SMTP(smtp_host, smtp_port) as s:
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)
    return "250 OK"


def _default_webhook_post(url: str, payload: bytes, timeout: int) -> str:
    import urllib.request
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return f"{resp.status} {resp.reason}"


# ── shared pre-flight (used by ActionExecutor AND SandboxedActionExecutor) ───

def common_preflight(context: ExecutionContext) -> None:
    """Validate context before any real execution. Raises ActionExecutorError on violation."""
    if not context.allow_real_actions:
        raise ActionExecutorError(
            "Real actions are disabled. Use --allow-real-actions flag."
        )
    if context.contract.signature_required:
        key_available = bool(context.signing_key) or check_signing_key_available()
        if not key_available:
            raise ActionExecutorError(
                f"Contract {context.contract.name!r} requires a signing key; "
                f"pass --signing-key or set MATRIXAI_ACTION_SIGNING_KEY"
            )
    if context.contract.human_approval:
        if context.approval_store is None:
            raise ActionExecutorError(
                f"Contract {context.contract.name!r} requires human approval "
                f"but no ApprovalStore is available in this context"
            )
        from matrixai.actions.approval import HumanApprovalGate
        gate = HumanApprovalGate(context.approval_store)
        if not gate.check(context):
            raise ActionExecutorError(
                f"Contract {context.contract.name!r} requires human approval; "
                f"submit execution for approval and wait for it to be approved"
            )
    if context.dry_run_report.is_expired(now=context.now):
        raise ActionExecutorError(
            f"DryRunReport {context.dry_run_report.report_id!r} is expired; "
            f"run a new dry-run before executing"
        )
    expected_ch = compute_action_contract_hash(context.contract)
    if context.dry_run_report.action_contract_hash != expected_ch:
        raise ActionExecutorError(
            "DryRunReport action_contract_hash does not match current contract"
        )
    expected_ih = _input_hash(context.input_data)
    if context.dry_run_report.input_hash != expected_ih:
        raise ActionExecutorError(
            "DryRunReport input_hash does not match input_data; "
            "input changed since dry-run"
        )


# ── executor ──────────────────────────────────────────────────────────────────

class ActionExecutor:
    """In-process executor for low and medium risk capabilities."""

    def __init__(
        self,
        http_get_fn: Callable | None = None,
        http_post_fn: Callable | None = None,
        email_fn: Callable | None = None,
        webhook_fn: Callable | None = None,
    ) -> None:
        self._http_get_fn = http_get_fn or _default_http_get
        self._http_post_fn = http_post_fn or _default_http_post
        self._email_fn = email_fn or _default_email_send
        self._webhook_fn = webhook_fn or _default_webhook_post

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: ExecutionContext) -> ActionResult:
        self._preflight(context)
        return self._dispatch(context)

    # ── pre-flight ────────────────────────────────────────────────────────────

    def _preflight(self, context: ExecutionContext) -> None:
        common_preflight(context)

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, context: ExecutionContext) -> ActionResult:
        cap = context.contract.capability
        if cap in HIGH_RISK_CAPABILITIES:
            raise ActionExecutorError(
                f"Capability {cap!r} is high-risk and must use SandboxedActionExecutor"
            )
        if cap == "http_get":
            return self._exec_http_get(context)
        if cap == "http_post":
            return self._exec_http_post(context)
        if cap == "email_send":
            return self._exec_email_send(context)
        if cap == "notification":
            return self._exec_notification(context)
        raise ActionExecutorError(f"Unsupported capability {cap!r}")

    # ── capability handlers ───────────────────────────────────────────────────

    def _exec_http_get(self, ctx: ExecutionContext) -> ActionResult:
        scope = ctx.contract.scope
        url = ctx.input_data.get("url", "")
        allowed = scope.get("allowed_urls", [])
        if not any(url.startswith(u) for u in allowed):
            return ActionResult(ok=False, response_summary="", latency_ms=0,
                                error=f"URL {url!r} not in scope allowed_urls")
        timeout = int(scope.get("timeout_seconds", 10))
        t0 = time.monotonic()
        try:
            summary = self._http_get_fn(url, timeout)
            return ActionResult(ok=True, response_summary=summary,
                                latency_ms=(time.monotonic() - t0) * 1000, error=None)
        except Exception as exc:
            return ActionResult(ok=False, response_summary="",
                                latency_ms=(time.monotonic() - t0) * 1000, error=str(exc))

    def _exec_http_post(self, ctx: ExecutionContext) -> ActionResult:
        scope = ctx.contract.scope
        url = ctx.input_data.get("url", "")
        allowed = scope.get("allowed_urls", [])
        if not any(url.startswith(u) for u in allowed):
            return ActionResult(ok=False, response_summary="", latency_ms=0,
                                error=f"URL {url!r} not in scope allowed_urls")
        required_headers = scope.get("required_headers", [])
        provided_headers = ctx.input_data.get("headers", {})
        missing = [h for h in required_headers if h not in provided_headers]
        if missing:
            return ActionResult(ok=False, response_summary="", latency_ms=0,
                                error=f"Missing required headers: {missing}")
        timeout = int(scope.get("timeout_seconds", 10))
        import json
        payload = json.dumps(ctx.input_data.get("body", {})).encode()
        t0 = time.monotonic()
        try:
            summary = self._http_post_fn(url, provided_headers, payload, timeout)
            return ActionResult(ok=True, response_summary=summary,
                                latency_ms=(time.monotonic() - t0) * 1000, error=None)
        except Exception as exc:
            return ActionResult(ok=False, response_summary="",
                                latency_ms=(time.monotonic() - t0) * 1000, error=str(exc))

    def _exec_email_send(self, ctx: ExecutionContext) -> ActionResult:
        scope = ctx.contract.scope
        recipient = ctx.input_data.get("recipient", "")
        allowed_recipients = scope.get("allowed_recipients", [])
        if allowed_recipients and recipient not in allowed_recipients:
            return ActionResult(ok=False, response_summary="", latency_ms=0,
                                error=f"recipient {recipient!r} not in scope allowed_recipients")
        allowed_domains = scope.get("allowed_domains", [])
        if allowed_domains:
            domain = recipient.split("@")[-1] if "@" in recipient else ""
            if domain not in allowed_domains:
                return ActionResult(ok=False, response_summary="", latency_ms=0,
                                    error=f"domain {domain!r} not in scope allowed_domains")
        import os
        smtp_host = os.environ.get(scope.get("smtp_host_env", "MATRIXAI_SMTP_HOST"), "localhost")
        smtp_port = int(os.environ.get(scope.get("smtp_port_env", "MATRIXAI_SMTP_PORT"), "25"))
        smtp_user = os.environ.get(scope.get("smtp_user_env", "MATRIXAI_SMTP_USER"), "")
        smtp_pass = os.environ.get(scope.get("smtp_pass_env", "MATRIXAI_SMTP_PASS"), "")
        subject = ctx.input_data.get("subject", "")
        body = ctx.input_data.get("body", "")
        t0 = time.monotonic()
        try:
            summary = self._email_fn(smtp_host, smtp_port, smtp_user, smtp_pass,
                                     recipient, subject, body)
            return ActionResult(ok=True, response_summary=summary,
                                latency_ms=(time.monotonic() - t0) * 1000, error=None)
        except Exception as exc:
            return ActionResult(ok=False, response_summary="",
                                latency_ms=(time.monotonic() - t0) * 1000, error=str(exc))

    def _exec_notification(self, ctx: ExecutionContext) -> ActionResult:
        scope = ctx.contract.scope
        recipient = ctx.input_data.get("recipient", "")
        allowed_recipients = scope.get("allowed_recipients", [])
        if allowed_recipients and recipient not in allowed_recipients:
            return ActionResult(ok=False, response_summary="", latency_ms=0,
                                error=f"recipient {recipient!r} not in scope allowed_recipients")
        webhook_url = ctx.input_data.get("webhook_url", scope.get("webhook_url", ""))
        timeout = int(scope.get("timeout_seconds", 10))
        import json
        payload = json.dumps(ctx.input_data).encode()
        t0 = time.monotonic()
        try:
            summary = self._webhook_fn(webhook_url, payload, timeout)
            return ActionResult(ok=True, response_summary=summary,
                                latency_ms=(time.monotonic() - t0) * 1000, error=None)
        except Exception as exc:
            return ActionResult(ok=False, response_summary="",
                                latency_ms=(time.monotonic() - t0) * 1000, error=str(exc))
