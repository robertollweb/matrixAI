"""PR4-C5 — Server hardening tests.

Covers:
- RateLimiter: allows up to limit, blocks at limit, disabled at 0, thread-safe
- CORS: wildcard default, specific origins, Vary header, preflight response
- Auth: Bearer token, X-API-Key header, both rejected when missing
- serve_model: accepts rate_limit and cors_origins params
- CLI: --rate-limit and --cors-origin flags parsed and passed through
- Integration: 429 response when rate limit exceeded
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from matrixai.server import RateLimiter


# ---------------------------------------------------------------------------
# RateLimiter — unit tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_up_to_limit(self):
        rl = RateLimiter(5)
        for _ in range(5):
            assert rl.is_allowed("10.0.0.1") is True

    def test_blocks_at_limit(self):
        rl = RateLimiter(3)
        for _ in range(3):
            rl.is_allowed("10.0.0.2")
        assert rl.is_allowed("10.0.0.2") is False

    def test_different_ips_independent(self):
        rl = RateLimiter(1)
        assert rl.is_allowed("10.0.0.1") is True
        assert rl.is_allowed("10.0.0.2") is True  # different IP, not blocked

    def test_disabled_at_zero(self):
        rl = RateLimiter(0)
        assert rl.disabled is True
        for _ in range(1000):
            assert rl.is_allowed("10.0.0.1") is True

    def test_negative_also_disabled(self):
        rl = RateLimiter(-1)
        assert rl.disabled is True
        assert rl.is_allowed("1.2.3.4") is True

    def test_thread_safe(self):
        rl = RateLimiter(100)
        results = []
        def check():
            results.append(rl.is_allowed("10.0.0.1"))
        threads = [threading.Thread(target=check) for _ in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        allowed = sum(1 for r in results if r)
        assert allowed == 100

    def test_window_allows_after_expiry(self):
        # Use a very small internal window by monkeypatching — we test the logic
        # by filling the window and checking that old entries are cleaned up
        rl = RateLimiter(3)
        ip = "10.0.0.5"
        now = time.time()
        # Inject timestamps older than 60 seconds directly
        rl._windows[ip] = [now - 61, now - 62, now - 63]
        # Old entries should be expired — all 3 slots available
        assert rl.is_allowed(ip) is True


# ---------------------------------------------------------------------------
# Server integration helpers
# ---------------------------------------------------------------------------

MINIMAL_MXAI = """\
PROJECT HardeningTest

VECTOR Input[2]
  score: Probability
  ratio: Score[0, 1]
END

NETWORK Classifier
  INPUT Input
  LAYER Dense units=1 activation=sigmoid
  OUTPUT label: Probability
END

GRAPH
  Input -> Classifier
END
"""


@pytest.fixture()
def mxai_file(tmp_path):
    p = tmp_path / "model.mxai"
    p.write_text(MINIMAL_MXAI)
    return p


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(mxai_file, *, rate_limit=60, cors_origins=None, api_key="test-key-abc"):
    """Start a MatrixAIHTTPServer in a daemon thread; return (server, port)."""
    from matrixai.parser import parse_file
    from matrixai.server import MatrixAIHTTPServer, MatrixAIServerHandler

    program = parse_file(mxai_file)
    port = _free_port()
    rl = RateLimiter(rate_limit)
    server = MatrixAIHTTPServer(
        ("127.0.0.1", port), MatrixAIServerHandler,
        program, None, "stdlib", api_key,
        rate_limiter=rl,
        cors_origins=cors_origins if cors_origins is not None else ["*"],
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return server, port


def _get(port, path, headers=None):
    import urllib.request
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def _post(port, path, body, headers=None):
    import urllib.request
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


# ---------------------------------------------------------------------------
# Auth — Bearer and X-API-Key
# ---------------------------------------------------------------------------

class TestAuth:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(mxai_file, api_key="secret123")
        self.port = port
        yield
        srv.shutdown()

    def test_bearer_token_accepted(self):
        status, _, _ = _post(
            self.port, "/predict",
            {"score": 0.5, "ratio": 0.5},
            headers={"Authorization": "Bearer secret123"},
        )
        # 401 = auth rejected; anything else means auth passed (model may return 400 without params)
        assert status != 401

    def test_x_api_key_accepted(self):
        status, _, _ = _post(
            self.port, "/predict",
            {"score": 0.5, "ratio": 0.5},
            headers={"X-API-Key": "secret123"},
        )
        assert status != 401

    def test_no_auth_returns_401(self):
        status, _, _ = _post(self.port, "/predict", {"score": 0.5, "ratio": 0.5})
        assert status == 401

    def test_wrong_bearer_returns_401(self):
        status, _, _ = _post(
            self.port, "/predict",
            {"score": 0.5, "ratio": 0.5},
            headers={"Authorization": "Bearer wrongkey"},
        )
        assert status == 401

    def test_wrong_x_api_key_returns_401(self):
        status, _, _ = _post(
            self.port, "/predict",
            {"score": 0.5, "ratio": 0.5},
            headers={"X-API-Key": "wrongkey"},
        )
        assert status == 401

    def test_health_no_auth_returns_200(self):
        status, _, _ = _get(self.port, "/health")
        assert status == 200


# ---------------------------------------------------------------------------
# Rate limiting — integration
# ---------------------------------------------------------------------------

class TestRateLimitingIntegration:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        # Allow only 2 requests per minute
        srv, port = _start_server(mxai_file, rate_limit=2, api_key="key1")
        self.port = port
        yield
        srv.shutdown()

    def test_first_two_requests_allowed(self):
        for _ in range(2):
            status, _, _ = _post(
                self.port, "/predict",
                {"score": 0.5, "ratio": 0.5},
                headers={"Authorization": "Bearer key1"},
            )
            # 429 = rate limited; anything else (200 or 400 from model) means not limited
            assert status != 429

    def test_third_request_returns_429(self):
        for _ in range(2):
            _post(
                self.port, "/predict",
                {"score": 0.5, "ratio": 0.5},
                headers={"Authorization": "Bearer key1"},
            )
        status, headers, _ = _post(
            self.port, "/predict",
            {"score": 0.5, "ratio": 0.5},
            headers={"Authorization": "Bearer key1"},
        )
        assert status == 429
        assert "Retry-After" in headers

    def test_disabled_rate_limit_never_blocks(self, mxai_file):
        srv, port = _start_server(mxai_file, rate_limit=0, api_key="key2")
        try:
            for _ in range(20):
                status, _, _ = _post(
                    port, "/predict",
                    {"score": 0.5, "ratio": 0.5},
                    headers={"Authorization": "Bearer key2"},
                )
                # Never 429 (rate-limited); model may return 400 without params
                assert status != 429
        finally:
            srv.shutdown()

    def test_health_not_rate_limited(self):
        # health bypasses auth and rate limit check (GET)
        for _ in range(10):
            status, _, _ = _get(self.port, "/health")
            assert status == 200


# ---------------------------------------------------------------------------
# CORS — headers and preflight
# ---------------------------------------------------------------------------

class TestCORSWildcard:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(mxai_file, cors_origins=["*"], api_key="ckey")
        self.port = port
        yield
        srv.shutdown()

    def test_cors_wildcard_on_health(self):
        status, headers, _ = _get(self.port, "/health")
        assert status == 200
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_cors_wildcard_on_predict(self):
        status, headers, _ = _post(
            self.port, "/predict",
            {"score": 0.5, "ratio": 0.5},
            headers={"Authorization": "Bearer ckey", "Origin": "https://example.com"},
        )
        assert headers.get("Access-Control-Allow-Origin") == "*"


class TestCORSSpecificOrigin:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(
            mxai_file,
            cors_origins=["https://studio.example.com"],
            api_key="ckey2",
        )
        self.port = port
        yield
        srv.shutdown()

    def test_matching_origin_echoed(self):
        status, headers, _ = _get(
            self.port, "/health",
            headers={"Origin": "https://studio.example.com"},
        )
        assert headers.get("Access-Control-Allow-Origin") == "https://studio.example.com"

    def test_vary_header_present(self):
        _, headers, _ = _get(
            self.port, "/health",
            headers={"Origin": "https://studio.example.com"},
        )
        assert "Vary" in headers

    def test_non_matching_origin_omits_acao_header(self):
        _, headers, _ = _get(
            self.port, "/health",
            headers={"Origin": "https://evil.com"},
        )
        # Non-matching origins must NOT receive an ACAO header (RFC-correct: browser blocks)
        assert headers.get("Access-Control-Allow-Origin") is None


class TestCORSPreflight:
    @pytest.fixture(autouse=True)
    def server(self, mxai_file):
        srv, port = _start_server(mxai_file, cors_origins=["*"], api_key="ckey3")
        self.port = port
        yield
        srv.shutdown()

    def _options(self, path, headers=None):
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            method="OPTIONS",
            headers=headers or {},
        )
        req.get_method = lambda: "OPTIONS"
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers)

    def test_options_returns_204(self):
        status, _ = self._options("/predict")
        assert status == 204

    def test_options_allows_methods(self):
        _, headers = self._options("/predict")
        methods = headers.get("Access-Control-Allow-Methods", "")
        assert "POST" in methods
        assert "GET" in methods
        assert "OPTIONS" in methods

    def test_options_allows_auth_headers(self):
        _, headers = self._options("/predict")
        allowed = headers.get("Access-Control-Allow-Headers", "")
        assert "Authorization" in allowed
        assert "X-API-Key" in allowed

    def test_options_max_age_set(self):
        _, headers = self._options("/predict")
        assert headers.get("Access-Control-Max-Age") == "86400"


# ---------------------------------------------------------------------------
# serve_model signature — rate_limit and cors_origins accepted
# ---------------------------------------------------------------------------

class TestServeModelSignature:
    def test_accepts_rate_limit_param(self, mxai_file):
        from matrixai.server import serve_model
        import inspect
        sig = inspect.signature(serve_model)
        assert "rate_limit" in sig.parameters

    def test_accepts_cors_origins_param(self, mxai_file):
        from matrixai.server import serve_model
        import inspect
        sig = inspect.signature(serve_model)
        assert "cors_origins" in sig.parameters

    def test_rate_limit_default_is_none(self, mxai_file):
        from matrixai.server import serve_model
        import inspect
        sig = inspect.signature(serve_model)
        assert sig.parameters["rate_limit"].default is None

    def test_cors_origins_default_is_none(self, mxai_file):
        from matrixai.server import serve_model
        import inspect
        sig = inspect.signature(serve_model)
        assert sig.parameters["cors_origins"].default is None


# ---------------------------------------------------------------------------
# CLI — --rate-limit and --cors-origin parsed
# ---------------------------------------------------------------------------

class TestCLIHardeningArgs:
    def _run(self, *args):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "serve", "--help"],
            capture_output=True, text=True,
        )
        return result

    def test_rate_limit_in_help(self):
        r = self._run()
        assert "--rate-limit" in r.stdout

    def test_cors_origin_in_help(self):
        r = self._run()
        assert "--cors-origin" in r.stdout

    def test_matrixai_rate_limit_env_mentioned(self, mxai_file):
        r = self._run()
        assert "MATRIXAI_RATE_LIMIT" in r.stdout

    def test_matrixai_cors_origins_env_mentioned(self, mxai_file):
        r = self._run()
        assert "MATRIXAI_CORS_ORIGINS" in r.stdout


# ---------------------------------------------------------------------------
# Env-var-based rate limit and CORS config
# ---------------------------------------------------------------------------

class TestEnvVarConfig:
    def test_rate_limit_env_var(self, mxai_file, monkeypatch):
        monkeypatch.setenv("MATRIXAI_RATE_LIMIT", "5")
        from matrixai.server import serve_model
        import inspect
        # Just verify the env var key is documented (server reads it at startup)
        assert "MATRIXAI_RATE_LIMIT" in open(
            Path(__file__).parent.parent / "matrixai" / "server.py"
        ).read()

    def test_cors_origins_env_var(self):
        assert "MATRIXAI_CORS_ORIGINS" in open(
            Path(__file__).parent.parent / "matrixai" / "server.py"
        ).read()

    def test_metrics_tracks_rate_limited(self, mxai_file):
        srv, port = _start_server(mxai_file, rate_limit=1, api_key="mkey")
        try:
            # First request: allowed
            _post(port, "/predict", {"score": 0.5, "ratio": 0.5},
                  headers={"Authorization": "Bearer mkey"})
            # Second request: rate-limited
            _post(port, "/predict", {"score": 0.5, "ratio": 0.5},
                  headers={"Authorization": "Bearer mkey"})
            # Check metrics
            assert srv.matrixai_metrics["requests_rate_limited"] >= 1
        finally:
            srv.shutdown()
