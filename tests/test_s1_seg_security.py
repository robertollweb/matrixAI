"""SEG-1 / SEG-2 / PRD-1 — Security and configuration fixes tests.

SEG-1: docker-compose.yml must bind to 127.0.0.1 only.
SEG-2: playground _origin_is_local() must accept localhost origins and reject
       remote origins.
PRD-1: training timeout is configurable via MATRIXAI_TRAIN_TIMEOUT env var,
       default 300s (not 30s).
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── SEG-1 — docker-compose port binding ──────────────────────────────────────

class TestDockerComposePortBinding:
    _COMPOSE = Path(__file__).parent.parent.parent / "matrixaistudio" / "bundle" / "docker-compose.yml"

    def test_compose_file_exists(self):
        assert self._COMPOSE.exists(), f"docker-compose.yml not found at {self._COMPOSE}"

    def test_port_bound_to_loopback(self):
        text = self._COMPOSE.read_text(encoding="utf-8")
        assert '"127.0.0.1:8080:80"' in text or "127.0.0.1:8080:80" in text, (
            "docker-compose.yml must bind nginx port to 127.0.0.1:8080:80, "
            "not all interfaces (0.0.0.0:8080:80)"
        )

    def test_no_all_interfaces_binding(self):
        text = self._COMPOSE.read_text(encoding="utf-8")
        # Must not have bare "8080:80" (which binds to 0.0.0.0)
        import re
        bare = re.search(r'["\s]"?8080:80"?', text)
        assert bare is None or "127.0.0.1" in text[max(0, bare.start()-5):bare.end()], (
            "Found bare 8080:80 binding (exposes to all interfaces)"
        )


# ── SEG-2 — Origin validation ─────────────────────────────────────────────────

def _make_handler(origin: str | None):
    """Build a PlaygroundHandler-like object with a stubbed headers dict."""
    import matrixai.playground as pg

    HandlerClass = pg._handler_class(None)  # guard=None
    handler = object.__new__(HandlerClass)
    headers = {}
    if origin is not None:
        headers["Origin"] = origin
    handler.headers = headers
    return handler


class TestOriginIsLocal:
    def test_no_origin_allowed(self):
        h = _make_handler(None)
        assert h._origin_is_local() is True

    def test_localhost_no_port_allowed(self):
        h = _make_handler("http://localhost")
        assert h._origin_is_local() is True

    def test_localhost_with_port_allowed(self):
        h = _make_handler("http://localhost:5175")
        assert h._origin_is_local() is True

    def test_127_no_port_allowed(self):
        h = _make_handler("http://127.0.0.1")
        assert h._origin_is_local() is True

    def test_127_with_port_allowed(self):
        h = _make_handler("http://127.0.0.1:8080")
        assert h._origin_is_local() is True

    def test_remote_origin_rejected(self):
        h = _make_handler("https://evil.example.com")
        assert h._origin_is_local() is False

    def test_remote_origin_with_localhost_in_path_rejected(self):
        # Ensure subdomain tricks don't pass
        h = _make_handler("https://localhost.evil.com")
        assert h._origin_is_local() is False

    def test_null_origin_rejected(self):
        # Sandboxed iframes send 'null' — should not be allowed
        h = _make_handler("null")
        assert h._origin_is_local() is False

    def test_empty_string_allowed(self):
        # Empty string behaves like no Origin header
        h = _make_handler("")
        assert h._origin_is_local() is True


# ── PRD-1 — Training timeout configurable ────────────────────────────────────

class TestTrainingTimeout:
    def test_default_is_300(self):
        # Re-import with MATRIXAI_TRAIN_TIMEOUT unset to check the default
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATRIXAI_TRAIN_TIMEOUT", None)
            import importlib
            import matrixai.playground as pg
            importlib.reload(pg)
            assert pg._P9_TRAIN_TIMEOUT == 300

    def test_env_var_overrides_default(self):
        with patch.dict(os.environ, {"MATRIXAI_TRAIN_TIMEOUT": "60"}):
            import importlib
            import matrixai.playground as pg
            importlib.reload(pg)
            assert pg._P9_TRAIN_TIMEOUT == 60

    def test_timeout_zero_disables_wall_clock_limit(self):
        # Downloadable Studio: MATRIXAI_TRAIN_TIMEOUT=0 → no limit (train to
        # completion; Cancel is the user's control). join timeout becomes None.
        with patch.dict(os.environ, {"MATRIXAI_TRAIN_TIMEOUT": "0"}):
            import importlib
            import matrixai.playground as pg
            importlib.reload(pg)
            assert pg._P9_TRAIN_TIMEOUT == 0
            assert pg._train_join_timeout() is None
        # a positive budget still maps to the numeric join timeout
        with patch.dict(os.environ, {"MATRIXAI_TRAIN_TIMEOUT": "120"}):
            import importlib
            import matrixai.playground as pg
            importlib.reload(pg)
            assert pg._train_join_timeout() == 120

    def test_compose_passes_timeout_to_container(self):
        compose = Path(__file__).parent.parent.parent / "matrixaistudio" / "bundle" / "docker-compose.yml"
        text = compose.read_text(encoding="utf-8")
        assert "MATRIXAI_TRAIN_TIMEOUT" in text

    def test_env_template_documents_timeout(self):
        template = Path(__file__).parent.parent.parent / "matrixaistudio" / "bundle" / ".env.template"
        text = template.read_text(encoding="utf-8")
        assert "MATRIXAI_TRAIN_TIMEOUT" in text
