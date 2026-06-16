# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from matrixai.actions.trace import ActionTrace, verify_action_trace
from matrixai.ir.continual import ContinualPolicySpec


class CollectorError(ValueError):
    pass


# ── ProductionSample ──────────────────────────────────────────────────────────

@dataclass
class ProductionSample:
    sample_id: str
    trace_id: str           # ActionTrace.report_id
    ground_truth: str       # label or value provided after the fact
    ingested_at: str        # ISO8601 UTC of ingestion
    executed_at: str        # ISO8601 UTC from ActionTrace
    source: str             # api | cli | file_watch
    model_hash: str
    parameter_set_id: str
    input_hash: str
    signed: bool            # True if ActionTrace had a valid HMAC


# ── helpers ───────────────────────────────────────────────────────────────────

_LABEL_RE = re.compile(r"^Label\[([^\]]+)\]$")


def _parse_valid_labels(label_type: str | None) -> list[str] | None:
    """Extract label list from 'Label[a, b, c]', or None if unconstrained."""
    if not label_type:
        return None
    m = _LABEL_RE.match(label_type.strip())
    if not m:
        return None
    return [lbl.strip() for lbl in m.group(1).split(",")]


def _to_utc(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── ProductionDataCollector ───────────────────────────────────────────────────

class ProductionDataCollector:
    """
    Collects production samples (ActionTrace + delayed ground truth) and
    validates them against a ContinualPolicySpec before storing.

    Audited ingestion requires a verifiable HMAC signature. Policies with
    SIGNATURE_REQUIRED=false are treated as an explicit non-audited opt-out for
    tests/demo data; if a signing key is provided, signatures are still enforced.

    The `trace_store` is an injected dict[str, ActionTrace] used to look up
    traces by report_id.  In production this would be backed by persistent
    storage; in tests it is populated directly.
    """

    def __init__(
        self,
        policy: ContinualPolicySpec,
        trace_store: dict[str, ActionTrace] | None = None,
        signing_key: str | None = None,
    ) -> None:
        self._policy = policy
        self._signing_key = signing_key
        self._trace_store: dict[str, ActionTrace] = trace_store if trace_store is not None else {}
        self._samples: list[ProductionSample] = []
        self._valid_labels = _parse_valid_labels(policy.ground_truth.label_type)

    # ── public API ────────────────────────────────────────────────────────────

    def register_trace(self, trace: ActionTrace) -> None:
        """Register an ActionTrace so it can be looked up by trace_id later."""
        self._trace_store[trace.report_id] = trace

    def ingest(
        self,
        trace: ActionTrace,
        ground_truth: str,
        source: str = "api",
        now: datetime | None = None,
    ) -> ProductionSample:
        """
        Validate and ingest one (trace, ground_truth) pair.

        Raises CollectorError if:
        - ActionTrace signature is required but missing/invalid
        - trace.executed_at is outside WINDOW_DAYS
        - ground_truth value doesn't match LABEL_TYPE
        """
        now = now or datetime.now(tz=timezone.utc)
        self._validate_signature(trace)
        self._validate_window(trace, now)
        self._validate_ground_truth(ground_truth)

        sample = ProductionSample(
            sample_id=str(uuid.uuid4()),
            trace_id=trace.report_id,
            ground_truth=ground_truth,
            ingested_at=now.isoformat(),
            executed_at=trace.executed_at,
            source=source,
            model_hash=trace.model_hash,
            parameter_set_id=trace.parameter_set_id,
            input_hash=trace.input_hash,
            signed=bool(trace.hmac_signature),
        )
        self._samples.append(sample)
        return sample

    def ingest_by_id(
        self,
        trace_id: str,
        ground_truth: str,
        source: str = "cli",
        now: datetime | None = None,
    ) -> ProductionSample:
        """Look up a trace by id, then ingest."""
        trace = self._trace_store.get(trace_id)
        if trace is None:
            raise CollectorError(
                f"Unknown trace_id {trace_id!r}: not found in trace store"
            )
        return self.ingest(trace, ground_truth, source=source, now=now)

    def ingest_from_dict(
        self,
        trace_dict: dict[str, Any],
        ground_truth: str,
        source: str = "api",
        now: datetime | None = None,
    ) -> ProductionSample:
        """Reconstruct an ActionTrace from a dict (e.g. from JSON) and ingest."""
        try:
            trace = ActionTrace(
                report_id=trace_dict["report_id"],
                model_hash=trace_dict["model_hash"],
                parameter_set_id=trace_dict["parameter_set_id"],
                action_contract_hash=trace_dict["action_contract_hash"],
                input_hash=trace_dict["input_hash"],
                executed_at=trace_dict["executed_at"],
                executor_kind=trace_dict["executor_kind"],
                ok=trace_dict["ok"],
                response_summary=trace_dict["response_summary"],
                error=trace_dict.get("error"),
                latency_ms=float(trace_dict["latency_ms"]),
                hmac_signature=trace_dict.get("hmac_signature"),
            )
        except KeyError as exc:
            raise CollectorError(f"ActionTrace dict missing field {exc}") from exc
        return self.ingest(trace, ground_truth, source=source, now=now)

    def scan_directory(
        self,
        path: str,
        now: datetime | None = None,
    ) -> list[ProductionSample]:
        """
        Scan a directory for JSON feedback files.

        Each file must contain:
          {"trace_id": "...", "ground_truth": "...", "ingested_at": "..."}

        The trace is looked up from the internal trace_store.
        Files that fail validation are skipped (errors collected, not raised).
        Returns only the newly ingested samples from this scan.
        """
        now = now or datetime.now(tz=timezone.utc)
        ingested: list[ProductionSample] = []
        if not os.path.isdir(path):
            return ingested

        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(path, fname)
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
                trace_id = data["trace_id"]
                gt = str(data["ground_truth"])
                sample = self.ingest_by_id(
                    trace_id, gt, source="file_watch", now=now
                )
                ingested.append(sample)
            except (CollectorError, KeyError, json.JSONDecodeError, OSError):
                continue  # silently skip invalid files in batch scan

        return ingested

    def get_samples_in_window(
        self,
        now: datetime | None = None,
    ) -> list[ProductionSample]:
        """Return samples whose executed_at is within WINDOW_DAYS of now."""
        now = now or datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(days=self._policy.ground_truth.window_days)
        return [
            s for s in self._samples
            if _to_utc(s.executed_at) >= cutoff
        ]

    def all_samples(self) -> list[ProductionSample]:
        return list(self._samples)

    # ── private validation ────────────────────────────────────────────────────

    def _validate_signature(self, trace: ActionTrace) -> None:
        """Enforce signature rules.

        Decision matrix:
        - sig_required + no signature         → always reject
        - sig_required + signature + no key   → reject (cannot verify)
        - signing_key + no signature           → reject (key implies expectation)
        - signing_key + signature              → verify cryptographically
        """
        has_sig = bool(trace.hmac_signature)
        has_key = bool(self._signing_key)
        sig_required = self._policy.audit.signature_required

        if sig_required and not has_sig:
            raise CollectorError(
                f"ActionTrace {trace.report_id!r} has no HMAC signature "
                "but SIGNATURE_REQUIRED is true"
            )

        if sig_required and has_sig and not has_key:
            raise CollectorError(
                f"ActionTrace {trace.report_id!r} has an HMAC signature "
                "but no signing_key was provided; cannot verify "
                "(SIGNATURE_REQUIRED is true)"
            )

        if has_key and not has_sig:
            raise CollectorError(
                f"ActionTrace {trace.report_id!r} is unsigned; "
                "cannot verify without signature"
            )

        if has_key and has_sig:
            if not verify_action_trace(trace, self._signing_key):
                raise CollectorError(
                    f"ActionTrace {trace.report_id!r} HMAC signature is invalid"
                )

    def _validate_window(self, trace: ActionTrace, now: datetime) -> None:
        """Reject traces whose executed_at is outside the ground truth window."""
        executed = _to_utc(trace.executed_at)
        cutoff = now - timedelta(days=self._policy.ground_truth.window_days)
        if executed < cutoff:
            raise CollectorError(
                f"ActionTrace {trace.report_id!r} executed_at {trace.executed_at!r} "
                f"is outside the {self._policy.ground_truth.window_days}-day ground truth window"
            )

    def _validate_ground_truth(self, ground_truth: str) -> None:
        """Validate against LABEL_TYPE if declared."""
        if self._valid_labels is not None:
            if ground_truth not in self._valid_labels:
                raise CollectorError(
                    f"Ground truth {ground_truth!r} is not in declared LABEL_TYPE "
                    f"{self._policy.ground_truth.label_type!r}; "
                    f"valid: {self._valid_labels}"
                )
