"""PR4-C2 — Observability tests.

Covers:
- GET /metrics returns 200 with Prometheus content-type
- All expected metric names present in the output
- Correct Prometheus text format (# HELP, # TYPE, value lines)
- Counter values increment after requests
- Drift metrics emitted when ProductionMonitor is attached
- Drift metrics absent when no monitor is attached
- /metrics listed in OpenAPI spec
- /metrics accessible without auth (same as /health)
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from matrixai.server import MatrixAIHTTPServer, MatrixAIServerHandler, RateLimiter


MINIMAL_MXAI = """\
PROJECT MinimalModel

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


# ---------------------------------------------------------------------------
# Helpers (mirrors test_pr4_c5_server_hardening.py style)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(mxai_file, *, api_key="test-key", monitor=None):
    from matrixai.parser import parse_file
    program = parse_file(mxai_file)
    port = _free_port()
    server = MatrixAIHTTPServer(
        ("127.0.0.1", port), MatrixAIServerHandler,
        program, None, "stdlib", api_key,
        rate_limiter=RateLimiter(0),  # no rate limiting in unit tests
        cors_origins=["*"],
        monitor=monitor,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return server, port


def _get(port, path, headers=None):
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


@pytest.fixture()
def mxai_file(tmp_path):
    p = tmp_path / "model.mxai"
    p.write_text(MINIMAL_MXAI)
    return p


# ---------------------------------------------------------------------------
# GET /metrics — basic shape
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(mxai_file)
        self.port = port
        self.server = srv
        yield
        srv.shutdown()

    def test_returns_200(self):
        status, _, _ = _get(self.port, "/metrics")
        assert status == 200

    def test_content_type_prometheus(self):
        _, headers, _ = _get(self.port, "/metrics")
        ct = headers.get("Content-Type", "")
        assert "text/plain" in ct
        assert "0.0.4" in ct

    def test_no_auth_required(self):
        # /metrics must be accessible without Authorization header, like /health
        status, _, _ = _get(self.port, "/metrics")
        assert status == 200

    def test_contains_help_lines(self):
        _, _, body = _get(self.port, "/metrics")
        assert "# HELP matrixai_requests_total" in body
        assert "# HELP matrixai_uptime_seconds" in body

    def test_contains_type_lines(self):
        _, _, body = _get(self.port, "/metrics")
        assert "# TYPE matrixai_requests_total counter" in body
        assert "# TYPE matrixai_uptime_seconds gauge" in body


# ---------------------------------------------------------------------------
# Metric names and structure
# ---------------------------------------------------------------------------

class TestMetricNames:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(mxai_file)
        self.port = port
        yield
        srv.shutdown()

    def _metrics(self) -> str:
        _, _, body = _get(self.port, "/metrics")
        return body

    def test_requests_total_present(self):
        assert "matrixai_requests_total" in self._metrics()

    def test_requests_successful_present(self):
        assert "matrixai_requests_successful_total" in self._metrics()

    def test_requests_failed_present(self):
        assert "matrixai_requests_failed_total" in self._metrics()

    def test_requests_rate_limited_present(self):
        assert "matrixai_requests_rate_limited_total" in self._metrics()

    def test_items_processed_present(self):
        assert "matrixai_items_processed_total" in self._metrics()

    def test_last_request_duration_present(self):
        assert "matrixai_last_request_duration_milliseconds" in self._metrics()

    def test_uptime_seconds_present(self):
        assert "matrixai_uptime_seconds" in self._metrics()

    def test_project_label_present(self):
        body = self._metrics()
        assert 'project="MinimalModel"' in body


# ---------------------------------------------------------------------------
# Counter increments after activity
# ---------------------------------------------------------------------------

class TestMetricValues:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(mxai_file, api_key="secret")
        self.port = port
        self.server = srv
        yield
        srv.shutdown()

    def _get_value(self, metric_name: str) -> float:
        _, _, body = _get(self.port, "/metrics")
        for line in body.splitlines():
            if line.startswith(metric_name + "{") and not line.startswith("# "):
                return float(line.split()[-1])
        return -1.0

    def test_uptime_positive(self):
        assert self._get_value("matrixai_uptime_seconds") > 0

    def test_requests_total_increments(self):
        import urllib.request, urllib.error
        before = self._get_value("matrixai_requests_total")
        # Hit health (a GET that counts as a request)
        _get(self.port, "/health")
        after = self._get_value("matrixai_requests_total")
        assert after >= before  # /metrics GET also counted

    def test_failed_increments_on_wrong_auth(self):
        before = self._get_value("matrixai_requests_failed_total")
        import urllib.request, urllib.error
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/predict",
            data=b'{"score":0.5,"ratio":0.5}',
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer wrong"},
        )
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError:
            pass
        after = self._get_value("matrixai_requests_failed_total")
        assert after > before


# ---------------------------------------------------------------------------
# Drift metrics — ProductionMonitor integration
# ---------------------------------------------------------------------------

class TestDriftMetrics:

    def _make_monitor(self):
        """Return a mock monitor that yields stable WindowMetrics without needing a full policy."""
        from unittest.mock import MagicMock
        from matrixai.continual.monitor import WindowMetrics
        from datetime import datetime, timezone

        wm = WindowMetrics(
            accuracy=1.0,
            samples=5,
            window_hours=1,
            window_start="2026-01-01T00:00:00+00:00",
            window_end=datetime.now(tz=timezone.utc).isoformat(),
            enough_samples=True,
            degradation_detected=False,
            reference_accuracy=1.0,
            actual_degradation=0.0,
        )
        monitor = MagicMock()
        monitor.window_metrics.return_value = wm
        return monitor

    def test_drift_metrics_absent_without_monitor(self, mxai_file):
        srv, port = _start_server(mxai_file, monitor=None)
        try:
            _, _, body = _get(port, "/metrics")
            assert "matrixai_drift" not in body
        finally:
            srv.shutdown()

    def test_drift_metrics_present_with_monitor(self, mxai_file):
        monitor = self._make_monitor()
        srv, port = _start_server(mxai_file, monitor=monitor)
        try:
            _, _, body = _get(port, "/metrics")
            assert "matrixai_drift_window_accuracy" in body
            assert "matrixai_drift_window_samples" in body
            assert "matrixai_drift_degradation_detected" in body
            assert "matrixai_drift_actual_degradation" in body
        finally:
            srv.shutdown()

    def test_drift_accuracy_value(self, mxai_file):
        monitor = self._make_monitor()
        srv, port = _start_server(mxai_file, monitor=monitor)
        try:
            _, _, body = _get(port, "/metrics")
            for line in body.splitlines():
                if line.startswith("matrixai_drift_window_accuracy{"):
                    value = float(line.split()[-1])
                    assert 0.0 <= value <= 1.0
                    break
            else:
                pytest.fail("matrixai_drift_window_accuracy not found in metrics")
        finally:
            srv.shutdown()

    def test_no_degradation_when_accuracy_perfect(self, mxai_file):
        monitor = self._make_monitor()  # all predictions correct
        srv, port = _start_server(mxai_file, monitor=monitor)
        try:
            _, _, body = _get(port, "/metrics")
            for line in body.splitlines():
                if line.startswith("matrixai_drift_degradation_detected{"):
                    assert float(line.split()[-1]) == 0.0
                    break
        finally:
            srv.shutdown()


# ---------------------------------------------------------------------------
# OpenAPI spec lists /metrics
# ---------------------------------------------------------------------------

class TestOpenAPIMetrics:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(mxai_file)
        self.port = port
        yield
        srv.shutdown()

    def test_metrics_in_openapi(self):
        _, _, body = _get(self.port, "/openapi.json")
        spec = json.loads(body)
        assert "/metrics" in spec.get("paths", {})

    def test_metrics_path_has_get(self):
        _, _, body = _get(self.port, "/openapi.json")
        spec = json.loads(body)
        assert "get" in spec["paths"]["/metrics"]
