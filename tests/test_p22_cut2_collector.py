"""P22 C2 — ProductionDataCollector: ingesta de ActionTrace + ground truth."""
import dataclasses
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from matrixai.actions.trace import ActionTrace, sign_action_trace
from matrixai.continual import parse_mxcontinual
from matrixai.continual.collector import (
    CollectorError,
    ProductionDataCollector,
    ProductionSample,
    _parse_valid_labels,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_SRC = """
CONTINUAL_POLICY TestCollectorPolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
    LABEL_TYPE Label[support, sales, ops]
    SOURCES [api, cli, file_watch]
  END

  DRIFT_DETECTION
    FEATURES [score, confidence]
    METHODS
      score: ks threshold=0.15
    END
    MIN_SAMPLES 100
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES 100
    MIN_GROUND_TRUTH_RATIO 0.5
    COOLDOWN_DAYS 1
  END

  TRAINING
    METHOD incremental_finetune
    LEARNING_RATE_FACTOR 0.1
    MAX_EPOCHS 10
    DATASET_MIX
      BASE_WEIGHT 0.5
      PRODUCTION_WEIGHT 0.5
      RECENCY_DECAY linear
    END
  END

  APPROVAL_GATE
    HOLDOUT_FRACTION 0.2
    REGRESSION_GUARD
      METRIC accuracy
      MUST_IMPROVE_BY 0.0
      MAX_DEGRADATION_PER_LABEL 0.05
    END
    HUMAN_APPROVAL false
  END

  ROLLBACK
    AUTO_TRIGGER false
    METRIC accuracy
    SLIDING_WINDOW_HOURS 24
    DEGRADATION_THRESHOLD 0.05
    MIN_SAMPLES_IN_WINDOW 50
  END

  AUDIT
    PERSIST_DRIFT_REPORTS true
    PERSIST_UPDATE_TRACES true
    EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT false
    REFINEMENT_DRIFT_PERSISTENCE_DAYS 14
    SIGNATURE_REQUIRED false
  END
END
"""

_POLICY_REQUIRES_SIG = _POLICY_SRC.replace(
    "SIGNATURE_REQUIRED false", "SIGNATURE_REQUIRED true"
)

_SIGNING_KEY = "a" * 64  # 32-byte hex key (64 hex chars)
_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
_RECENT = (_NOW - timedelta(days=2)).isoformat()   # within window
_OLD = (_NOW - timedelta(days=10)).isoformat()      # outside window


def _make_trace(
    report_id: str = "act-001",
    executed_at: str | None = None,
    signing_key: str | None = None,
) -> ActionTrace:
    trace = ActionTrace(
        report_id=report_id,
        model_hash="sha256:abc123",
        parameter_set_id="ps-v1.0",
        action_contract_hash="sha256:def456",
        input_hash="sha256:ghi789",
        executed_at=executed_at or _RECENT,
        executor_kind="in_process",
        ok=True,
        response_summary="ok",
        error=None,
        latency_ms=42.0,
        hmac_signature=None,
    )
    if signing_key:
        trace.hmac_signature = sign_action_trace(trace, signing_key)
    return trace


def _policy(src=_POLICY_SRC):
    return parse_mxcontinual(src)


# ── helper tests ───────────────────────────────────────────────────────────────

class TestParseValidLabels:
    def test_parses_label_type(self):
        labels = _parse_valid_labels("Label[support, sales, ops]")
        assert set(labels) == {"support", "sales", "ops"}

    def test_returns_none_for_no_label_type(self):
        assert _parse_valid_labels(None) is None

    def test_returns_none_for_unrecognized_format(self):
        assert _parse_valid_labels("Scalar[0, 1]") is None


# ── basic ingestion ────────────────────────────────────────────────────────────

class TestCollectorIngest:
    def test_accepts_valid_trace_with_ground_truth(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace()
        sample = collector.ingest(trace, "support", now=_NOW)
        assert isinstance(sample, ProductionSample)
        assert sample.trace_id == "act-001"
        assert sample.ground_truth == "support"

    def test_stores_sample_with_timestamp(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace()
        sample = collector.ingest(trace, "sales", now=_NOW)
        assert sample.ingested_at == _NOW.isoformat()
        assert sample.executed_at == _RECENT

    def test_sample_has_unique_id(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        t1 = _make_trace("act-001")
        t2 = _make_trace("act-002")
        s1 = collector.ingest(t1, "support", now=_NOW)
        s2 = collector.ingest(t2, "sales", now=_NOW)
        assert s1.sample_id != s2.sample_id

    def test_ingest_records_source(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace()
        sample = collector.ingest(trace, "ops", source="cli", now=_NOW)
        assert sample.source == "cli"

    def test_all_samples_returns_accumulated(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        for i in range(3):
            trace = _make_trace(f"act-{i:03d}")
            collector.ingest(trace, "support", now=_NOW)
        assert len(collector.all_samples()) == 3

    def test_ingest_from_dict_reconstructs_trace(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace()
        d = {
            "report_id": trace.report_id,
            "model_hash": trace.model_hash,
            "parameter_set_id": trace.parameter_set_id,
            "action_contract_hash": trace.action_contract_hash,
            "input_hash": trace.input_hash,
            "executed_at": trace.executed_at,
            "executor_kind": trace.executor_kind,
            "ok": trace.ok,
            "response_summary": trace.response_summary,
            "error": trace.error,
            "latency_ms": trace.latency_ms,
            "hmac_signature": trace.hmac_signature,
        }
        sample = collector.ingest_from_dict(d, "ops", now=_NOW)
        assert sample.ground_truth == "ops"


# ── ingest_by_id ──────────────────────────────────────────────────────────────

class TestIngestById:
    def test_ingests_registered_trace(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace("act-reg")
        collector.register_trace(trace)
        sample = collector.ingest_by_id("act-reg", "sales", now=_NOW)
        assert sample.trace_id == "act-reg"

    def test_rejects_unknown_trace_id(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        with pytest.raises(CollectorError, match="Unknown trace_id"):
            collector.ingest_by_id("nonexistent-id", "support", now=_NOW)


# ── ground truth window ────────────────────────────────────────────────────────

class TestGroundTruthWindow:
    def test_accepts_trace_within_window(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace(executed_at=_RECENT)
        sample = collector.ingest(trace, "ops", now=_NOW)
        assert sample is not None

    def test_rejects_trace_outside_window(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace(executed_at=_OLD)
        with pytest.raises(CollectorError, match="outside the .* ground truth window"):
            collector.ingest(trace, "support", now=_NOW)

    def test_get_samples_in_window_filters_by_executed_at(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        recent_trace = _make_trace("act-recent", executed_at=_RECENT)
        old_trace = _make_trace("act-old", executed_at=_OLD)
        collector._samples.append(ProductionSample(
            sample_id="s1", trace_id="act-old", ground_truth="support",
            ingested_at=_NOW.isoformat(), executed_at=_OLD,
            source="api", model_hash="x", parameter_set_id="ps1",
            input_hash="h", signed=False,
        ))
        collector.ingest(recent_trace, "sales", now=_NOW)
        in_window = collector.get_samples_in_window(now=_NOW)
        trace_ids = {s.trace_id for s in in_window}
        assert "act-recent" in trace_ids
        assert "act-old" not in trace_ids

    def test_respects_ground_truth_window_boundary(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        # Exactly at boundary (7 days ago) should be included
        boundary = (_NOW - timedelta(days=7)).isoformat()
        trace = _make_trace(executed_at=boundary)
        sample = collector.ingest(trace, "support", now=_NOW)
        assert sample is not None


# ── ground truth type validation ──────────────────────────────────────────────

class TestGroundTruthType:
    def test_accepts_valid_label(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace()
        sample = collector.ingest(trace, "support", now=_NOW)
        assert sample.ground_truth == "support"

    def test_rejects_invalid_label(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace()
        with pytest.raises(CollectorError, match="not in declared LABEL_TYPE"):
            collector.ingest(trace, "billing", now=_NOW)

    def test_accepts_any_label_when_type_not_declared(self):
        src = _POLICY_SRC.replace(
            "LABEL_TYPE Label[support, sales, ops]\n", ""
        )
        policy = parse_mxcontinual(src)
        collector = ProductionDataCollector(policy)
        trace = _make_trace()
        sample = collector.ingest(trace, "anything_at_all", now=_NOW)
        assert sample.ground_truth == "anything_at_all"


# ── signature validation ───────────────────────────────────────────────────────

class TestSignatureValidation:
    def test_accepts_unsigned_trace_when_not_required(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        trace = _make_trace()   # no signature
        sample = collector.ingest(trace, "support", now=_NOW)
        assert sample.signed is False

    def test_rejects_unsigned_when_signature_required(self):
        policy = parse_mxcontinual(_POLICY_REQUIRES_SIG)
        collector = ProductionDataCollector(policy)
        trace = _make_trace()   # no signature
        with pytest.raises(CollectorError, match="no HMAC signature"):
            collector.ingest(trace, "support", now=_NOW)

    def test_accepts_correctly_signed_trace(self):
        policy = parse_mxcontinual(_POLICY_REQUIRES_SIG)
        collector = ProductionDataCollector(policy, signing_key=_SIGNING_KEY)
        trace = _make_trace(signing_key=_SIGNING_KEY)
        sample = collector.ingest(trace, "sales", now=_NOW)
        assert sample.signed is True

    def test_rejects_tampered_signature(self):
        policy = _policy()
        collector = ProductionDataCollector(policy, signing_key=_SIGNING_KEY)
        trace = _make_trace(signing_key=_SIGNING_KEY)
        trace.hmac_signature = "hmac-sha256:" + "0" * 64
        with pytest.raises(CollectorError, match="HMAC signature is invalid"):
            collector.ingest(trace, "support", now=_NOW)

    def test_rejects_unsigned_when_signing_key_provided(self):
        policy = _policy()
        collector = ProductionDataCollector(policy, signing_key=_SIGNING_KEY)
        trace = _make_trace()   # no signature
        with pytest.raises(CollectorError, match="unsigned"):
            collector.ingest(trace, "ops", now=_NOW)

    def test_rejects_signed_trace_when_required_but_no_key_to_verify(self):
        # signature_required=True but no signing_key → can't verify → must reject
        policy = parse_mxcontinual(_POLICY_REQUIRES_SIG)
        collector = ProductionDataCollector(policy)   # no signing_key
        trace = _make_trace(signing_key=_SIGNING_KEY)  # trace IS signed
        with pytest.raises(CollectorError, match="cannot verify"):
            collector.ingest(trace, "support", now=_NOW)


# ── CLI ingest ────────────────────────────────────────────────────────────────

class TestContinualIngestCLI:
    def test_cli_ingest_accepts_signed_trace_file(self, tmp_path, monkeypatch, capsys):
        from matrixai.cli import main

        policy_path = tmp_path / "policy.mxcontinual"
        policy_path.write_text(_POLICY_REQUIRES_SIG)
        trace = _make_trace("act-cli-001", signing_key=_SIGNING_KEY)
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(dataclasses.asdict(trace)))

        monkeypatch.setattr(sys, "argv", [
            "matrixai", "continual", "ingest", str(policy_path),
            "--trace-id", "act-cli-001",
            "--label", "support",
            "--trace-file", str(trace_path),
            "--signing-key", _SIGNING_KEY,
            "--json",
        ])

        rc = main()
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert rc == 0
        assert payload["trace_id"] == "act-cli-001"
        assert payload["ground_truth"] == "support"
        assert payload["signed"] is True

    def test_cli_ingest_rejects_required_signature_without_key(self, tmp_path, monkeypatch, capsys):
        from matrixai.cli import main

        policy_path = tmp_path / "policy.mxcontinual"
        policy_path.write_text(_POLICY_REQUIRES_SIG)
        trace = _make_trace("act-cli-002", signing_key=_SIGNING_KEY)
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(dataclasses.asdict(trace)))

        monkeypatch.setattr(sys, "argv", [
            "matrixai", "continual", "ingest", str(policy_path),
            "--trace-id", "act-cli-002",
            "--label", "support",
            "--trace-file", str(trace_path),
        ])

        rc = main()
        err = capsys.readouterr().err
        assert rc == 1
        assert "cannot verify" in err


# ── file watch ────────────────────────────────────────────────────────────────

class TestFileWatch:
    def test_picks_up_new_json_files(self):
        policy = _policy()
        trace = _make_trace("act-fw-001", executed_at=_RECENT)
        collector = ProductionDataCollector(policy)
        collector.register_trace(trace)

        with tempfile.TemporaryDirectory() as tmpdir:
            fb = {"trace_id": "act-fw-001", "ground_truth": "support",
                  "ingested_at": _NOW.isoformat()}
            with open(os.path.join(tmpdir, "feedback_001.json"), "w") as fh:
                json.dump(fb, fh)
            samples = collector.scan_directory(tmpdir, now=_NOW)

        assert len(samples) == 1
        assert samples[0].ground_truth == "support"
        assert samples[0].source == "file_watch"

    def test_ignores_non_json_files(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "notes.txt"), "w") as fh:
                fh.write("not json")
            samples = collector.scan_directory(tmpdir, now=_NOW)

        assert samples == []

    def test_skips_files_with_unknown_trace_id(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)

        with tempfile.TemporaryDirectory() as tmpdir:
            fb = {"trace_id": "nonexistent", "ground_truth": "ops",
                  "ingested_at": _NOW.isoformat()}
            with open(os.path.join(tmpdir, "bad.json"), "w") as fh:
                json.dump(fb, fh)
            samples = collector.scan_directory(tmpdir, now=_NOW)

        assert samples == []

    def test_handles_missing_directory_gracefully(self):
        policy = _policy()
        collector = ProductionDataCollector(policy)
        samples = collector.scan_directory("/nonexistent/path/xyz", now=_NOW)
        assert samples == []
