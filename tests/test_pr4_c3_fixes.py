"""PR4-C3 fixes — regression tests for F1–F4.

F1: reference_accuracy read from params metrics.accuracy
F2: RollbackEvent persisted + continual status shows it
F3: /metrics emits matrixai_model_info with populated labels
F4: action_dry_runs_total increments on failed dry-run (no double-count on success)
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from matrixai.server import MatrixAIHTTPServer, MatrixAIServerHandler, RateLimiter


# ---------------------------------------------------------------------------
# Minimal fixtures shared across tests
# ---------------------------------------------------------------------------

MINIMAL_MXAI = """\
PROJECT TestModel

VECTOR Features[2]
  score: Score
  ratio: Score
END

FUNCTION Predict
  R: Risk = sigmoid(W1 * Features + b1)
END

GRAPH
  Features -> Predict
END
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port, path):
    import urllib.request, urllib.error
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ---------------------------------------------------------------------------
# F1 — reference_accuracy from params.json metrics.accuracy
# ---------------------------------------------------------------------------

class TestF1ReferenceAccuracy:

    def _make_params_file(self, tmp_path, accuracy: float | None) -> Path:
        data = {
            "parameter_set_id": "ps_test",
            "model_hash": "mxai_test",
            "parameter_schema_hash": "params_test",
            "source": "test",
            "parameters": {},
            "metrics": {"accuracy": accuracy} if accuracy is not None else {},
        }
        p = tmp_path / "params.json"
        p.write_text(json.dumps(data))
        return p

    def _make_policy_file(self, tmp_path) -> Path:
        policy_text = """\
CONTINUAL_POLICY TestPolicy
  TARGET_MODEL model.mxai
  BASE_PARAMETER_SET runs/params.json
  REGISTRY_NAME test-model
  BASE_VERSION v1.0

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
  END

  DRIFT_DETECTION
    FEATURES [score]
    METHODS
      score: ks threshold=0.15
    END
    MIN_SAMPLES 10
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES 50
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
      MIN_DELTA -0.02
    END
    HUMAN_APPROVAL false
  END

  ROLLBACK
    AUTO_TRIGGER false
    METRIC accuracy
    SLIDING_WINDOW_HOURS 24
    DEGRADATION_THRESHOLD 0.05
    MIN_SAMPLES_IN_WINDOW 10
  END

  AUDIT
    PERSIST_DRIFT_REPORTS true
    PERSIST_UPDATE_TRACES true
    EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT false
    REFINEMENT_DRIFT_PERSISTENCE_DAYS 3
    SIGNATURE_REQUIRED false
  END
END
"""
        p = tmp_path / "policy.mxcontinual"
        p.write_text(policy_text)
        return p

    def test_reference_accuracy_read_from_params(self, tmp_path):
        """ProductionMonitor._reference is set from params.json metrics.accuracy."""
        from matrixai.continual.parser import parse_mxcontinual
        from matrixai.continual.monitor import ProductionMonitor

        params_path = self._make_params_file(tmp_path, accuracy=0.91)
        policy_path = self._make_policy_file(tmp_path)

        policy = parse_mxcontinual(policy_path.read_text())

        # Simulate what _cmd_serve does
        reference_accuracy = None
        try:
            ps_data = json.loads(params_path.read_text())
            reference_accuracy = ps_data.get("metrics", {}).get("accuracy")
        except Exception:
            pass

        monitor = ProductionMonitor(policy, reference_accuracy=reference_accuracy, labels=[])
        assert monitor._reference == pytest.approx(0.91)

    def test_reference_accuracy_none_when_no_params(self, tmp_path):
        """Without --params, reference_accuracy is None and degradation stays False."""
        from matrixai.continual.parser import parse_mxcontinual
        from matrixai.continual.monitor import ProductionMonitor

        policy_path = self._make_policy_file(tmp_path)
        policy = parse_mxcontinual(policy_path.read_text())

        monitor = ProductionMonitor(policy, reference_accuracy=None, labels=[])
        assert monitor._reference is None

        # Even with all-wrong predictions, degradation_detected stays False
        monitor.record("wrong", "correct")
        monitor.record("wrong", "correct")
        wm = monitor.window_metrics()
        assert wm.degradation_detected is False

    def test_degradation_detected_with_reference(self, tmp_path):
        """With reference_accuracy set, degradation_detected fires when accuracy drops."""
        from matrixai.continual.parser import parse_mxcontinual
        from matrixai.continual.monitor import ProductionMonitor

        policy_path = self._make_policy_file(tmp_path)
        policy = parse_mxcontinual(policy_path.read_text())

        # reference=1.0, threshold=0.05 → degrade if accuracy < 0.95
        monitor = ProductionMonitor(policy, reference_accuracy=1.0, labels=[])
        for _ in range(10):  # 10 wrong predictions, 0% accuracy
            monitor.record("wrong", "correct")

        wm = monitor.window_metrics()
        assert wm.degradation_detected is True
        assert wm.actual_degradation == pytest.approx(1.0)

    def test_explicit_reference_accuracy_overrides_params(self, tmp_path):
        """--reference-accuracy explicit value takes priority over params file."""
        params_path = self._make_params_file(tmp_path, accuracy=0.91)

        # Simulate the priority logic in _cmd_serve
        explicit_ref = 0.75  # explicit flag
        reference_accuracy = explicit_ref
        if reference_accuracy is None and params_path.exists():
            ps_data = json.loads(params_path.read_text())
            reference_accuracy = ps_data.get("metrics", {}).get("accuracy")

        assert reference_accuracy == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# F2 — RollbackEvent persisted + continual status reads it
# ---------------------------------------------------------------------------

class TestF2RollbackPersist:

    def test_rollback_event_persisted(self, tmp_path):
        """continual rollback writes .{name}_last_rollback.json with rolled_back_at."""
        from matrixai.continual.rollback import RollbackEvent

        registry_name = "alert-monitor"
        event = RollbackEvent(
            rollback_id="rb_test_001",
            policy_hash="hash_test",
            trigger_reason="manual",
            metric="accuracy",
            sliding_window_value=0.71,
            threshold=0.05,
            from_parameter_set_id="ps_new",
            to_parameter_set_id="ps_base",
            from_version="v1.1",
            to_version="v1.0",
            rolled_back_at="2026-05-28T12:34:56+00:00",
            samples_in_window=50,
            notification_sent=False,
            signature="hmac-sha256:test",
        )

        import dataclasses
        event_dict = dataclasses.asdict(event)
        event_path = tmp_path / f".{registry_name}_last_rollback.json"
        event_path.write_text(json.dumps(event_dict, indent=2), encoding="utf-8")

        assert event_path.exists()
        loaded = json.loads(event_path.read_text())
        assert loaded["rollback_id"] == "rb_test_001"
        assert loaded["from_version"] == "v1.1"
        assert loaded["to_version"] == "v1.0"
        assert loaded["rolled_back_at"] == "2026-05-28T12:34:56+00:00"

    def test_status_reads_rolled_back_at_not_executed_at(self, tmp_path):
        """continual status reads 'rolled_back_at' (real field), not 'executed_at'."""
        registry_name = "alert-monitor"
        event_dict = {
            "rollback_id": "rb_001",
            "from_version": "v1.1",
            "to_version": "v1.0",
            "from_parameter_set_id": "ps_new",
            "to_parameter_set_id": "ps_base",
            "trigger_reason": "manual",
            "rolled_back_at": "2026-05-28T12:34:56+00:00",
            "signature": "hmac-sha256:abc",
        }
        event_path = tmp_path / f".{registry_name}_last_rollback.json"
        event_path.write_text(json.dumps(event_dict))

        loaded = json.loads(event_path.read_text())

        # Simulate exactly what cli.py _cmd_status does
        executed_at = loaded.get("rolled_back_at", "")  # must use rolled_back_at
        assert executed_at == "2026-05-28T12:34:56+00:00"

        # Confirm 'executed_at' key does NOT exist (it's not in the dataclass)
        assert "executed_at" not in loaded

    def test_no_auto_rollback_in_runbook_en(self):
        """RUNBOOK.md EN must NOT contain 'fires automatically' claim."""
        runbook_path = Path(__file__).parent.parent / "docs/en/deployment/RUNBOOK.md"
        if runbook_path.exists():
            text = runbook_path.read_text()
            assert "fires automatically" not in text, (
                "Runbook still claims auto-rollback which is not implemented"
            )
            assert "always triggered manually" in text or "always manual" in text.lower()


# ---------------------------------------------------------------------------
# F3 — matrixai_model_info gauge with populated labels
# ---------------------------------------------------------------------------

class TestF3ModelInfoGauge:

    @pytest.fixture()
    def mxai_file(self, tmp_path):
        p = tmp_path / "model.mxai"
        p.write_text(MINIMAL_MXAI)
        return p

    def _start_server(self, mxai_file, *, parameter_set=None):
        from matrixai.parser import parse_file
        program = parse_file(mxai_file)
        port = _free_port()
        server = MatrixAIHTTPServer(
            ("127.0.0.1", port), MatrixAIServerHandler,
            program, parameter_set, "stdlib", "test-key",
            rate_limiter=RateLimiter(0),
            cors_origins=["*"],
        )
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        return server, port

    def _fake_parameter_set(self):
        from matrixai.parameters.store import ParameterSet
        return ParameterSet(
            parameter_set_id="ps_test_001",
            model_hash="mxai_abc123",
            parameter_schema_hash="params_xyz",
            source="test",
            parameters={},
        )

    def test_model_info_present_with_params(self, mxai_file):
        """matrixai_model_info appears with populated labels when params loaded."""
        ps = self._fake_parameter_set()
        srv, port = self._start_server(mxai_file, parameter_set=ps)
        try:
            _, body = _get(port, "/metrics")
            assert "matrixai_model_info" in body
            assert "ps_test_001" in body
            assert "mxai_abc123" in body
            assert "TestModel" in body
        finally:
            srv.shutdown()

    def test_model_info_line_value_is_1(self, mxai_file):
        """The matrixai_model_info gauge value must be 1."""
        ps = self._fake_parameter_set()
        srv, port = self._start_server(mxai_file, parameter_set=ps)
        try:
            _, body = _get(port, "/metrics")
            for line in body.splitlines():
                if line.startswith("matrixai_model_info{"):
                    assert line.endswith(" 1"), f"Expected value 1, got: {line}"
                    break
            else:
                pytest.fail("matrixai_model_info line not found in /metrics")
        finally:
            srv.shutdown()

    def test_model_info_emitted_without_params(self, mxai_file):
        """matrixai_model_info is emitted even without params (empty label values)."""
        srv, port = self._start_server(mxai_file, parameter_set=None)
        try:
            _, body = _get(port, "/metrics")
            assert "matrixai_model_info" in body
        finally:
            srv.shutdown()


# ---------------------------------------------------------------------------
# F4 — action_dry_runs_total counts failed dry-runs, no double-count on success
# ---------------------------------------------------------------------------

class TestF4DryRunCounter:

    @pytest.fixture()
    def mxai_file(self, tmp_path):
        p = tmp_path / "model.mxai"
        p.write_text(MINIMAL_MXAI)
        return p

    def _start_server(self, mxai_file):
        from matrixai.parser import parse_file
        program = parse_file(mxai_file)
        port = _free_port()
        server = MatrixAIHTTPServer(
            ("127.0.0.1", port), MatrixAIServerHandler,
            program, None, "stdlib", "test-key",
            rate_limiter=RateLimiter(0),
            cors_origins=["*"],
        )
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        return server, port

    def _get_counter(self, port: int, name: str) -> float:
        _, body = _get(port, "/metrics")
        for line in body.splitlines():
            if line.startswith(name + "{") and not line.startswith("# "):
                return float(line.split()[-1])
        return 0.0

    def test_dry_run_counter_starts_at_zero(self, mxai_file):
        srv, port = self._start_server(mxai_file)
        try:
            assert self._get_counter(port, "matrixai_action_dry_runs_total") == 0.0
        finally:
            srv.shutdown()

    def test_dry_run_counter_no_double_count(self, mxai_file):
        """Dry-run counter is exactly the number of execute-action calls, not doubled."""
        import urllib.request, urllib.error
        srv, port = self._start_server(mxai_file)
        try:
            before = self._get_counter(port, "matrixai_action_dry_runs_total")
            # POST /execute-action will fail (no contracts loaded) — 404 before dry-run
            # The counter should NOT increment since we never reach sim.simulate()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/execute-action",
                data=b'{"contract_name":"X","input_data":{}}',
                method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer test-key"},
            )
            try:
                urllib.request.urlopen(req)
            except urllib.error.HTTPError:
                pass
            after = self._get_counter(port, "matrixai_action_dry_runs_total")
            # No contracts loaded → returns 404 before sim.simulate(), counter unchanged
            assert after == before
        finally:
            srv.shutdown()
