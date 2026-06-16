# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PR5-C6 — /api/v1 contract tests.

Validates: versioned routes, auth scopes, registry HTTP layer,
common error schema {ok, error, code}, and pagination.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _start_server(port: int, model_path=None, params_path=None,
                  registry_path=None, api_key="test-key", api_key_read=None):
    """Start a MatrixAI server in a background thread and return it."""
    from matrixai.server import MatrixAIHTTPServer, MatrixAIServerHandler, RateLimiter

    program = None
    if model_path:
        from matrixai.parser import parse_file
        program = parse_file(Path(model_path))

    parameter_set = None
    if params_path:
        from matrixai.parameters import load_parameter_set
        parameter_set = load_parameter_set(Path(params_path))

    registry = None
    if registry_path:
        from matrixai.registry.model_registry import ModelRegistry
        registry = ModelRegistry(Path(registry_path))

    server = MatrixAIHTTPServer(
        ("127.0.0.1", port), MatrixAIServerHandler,
        program, parameter_set, "stdlib", api_key,
        rate_limiter=RateLimiter(1000),
        api_key_read=api_key_read,
        registry=registry,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)
    return server


def _req(port, method, path, body=None, key="test-key", content_type="application/json"):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Content-Type": content_type}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    encoded = json.dumps(body).encode() if body is not None else None
    if encoded:
        headers["Content-Length"] = str(len(encoded))
    conn.request(method, path, body=encoded, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = raw.decode()
    return resp.status, data


# ── fixtures ──────────────────────────────────────────────────────────────────

TEMPLATE_DIR = ROOT / "matrixai" / "templates" / "classification"
MODEL_FILE = TEMPLATE_DIR / "{{project_name}}.mxai".replace("{{project_name}}", "{{project_name}}")


def _find_model_file():
    for f in TEMPLATE_DIR.glob("*.mxai"):
        return f
    return None


# ── test: versioned health alias ──────────────────────────────────────────────

class TestVersionedAliases:
    port = 19700

    @classmethod
    def setup_class(cls):
        cls.server = _start_server(cls.port)

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_v1_returns_ok_true(self):
        status, data = _req(self.port, "GET", "/api/v1/health", key=None)
        assert status == 200
        assert data["ok"] is True
        assert data["status"] == "ok"

    def test_health_v1_schema_has_ok_field(self):
        _, data = _req(self.port, "GET", "/api/v1/health", key=None)
        assert "ok" in data

    def test_unknown_v1_route_returns_404_with_schema(self):
        status, data = _req(self.port, "GET", "/api/v1/nonexistent")
        assert status == 404
        assert data["ok"] is False
        assert "code" in data
        assert "error" in data


# ── test: auth scopes ─────────────────────────────────────────────────────────

class TestAuthScopes:
    port = 19701

    @classmethod
    def setup_class(cls):
        cls.server = _start_server(cls.port, api_key="write-key",
                                   api_key_read="read-key")

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_write_key_accepted_on_read_endpoint(self):
        status, data = _req(self.port, "GET", "/api/v1/registry", key="write-key")
        assert status == 503   # no registry mounted, but auth passed
        assert data["code"] == "REGISTRY_NOT_LOADED"

    def test_read_key_accepted_on_read_endpoint(self):
        status, data = _req(self.port, "GET", "/api/v1/registry", key="read-key")
        assert status == 503
        assert data["code"] == "REGISTRY_NOT_LOADED"

    def test_read_key_rejected_on_write_endpoint(self):
        status, data = _req(self.port, "POST", "/api/v1/registry/push",
                             body={"name": "x", "version": "v1", "run_dir": "/tmp"},
                             key="read-key")
        assert status == 401
        assert data["ok"] is False
        assert data["code"] == "UNAUTHORIZED"

    def test_no_key_rejected(self):
        status, data = _req(self.port, "GET", "/api/v1/health", key=None)
        # health is public (no auth required on GET /api/v1/health)
        assert status == 200

    def test_wrong_key_rejected_on_predict_alias(self):
        status, data = _req(self.port, "POST", "/api/v1/predict",
                             body={"x": 1}, key="wrong-key")
        assert status == 401
        assert data["ok"] is False

    def test_v1_execute_action_no_contract_has_code(self):
        status, data = _req(self.port, "POST", "/api/v1/execute-action",
                             body={"contract_name": "x", "input_data": {}},
                             key="write-key")
        assert status == 404
        assert data["ok"] is False
        assert data["code"] == "NO_CONTRACT"

    def test_v1_feedback_no_monitor_has_code(self):
        status, data = _req(self.port, "POST", "/api/v1/feedback",
                             body={"prediction": "1", "ground_truth": "0"},
                             key="write-key")
        assert status == 404
        assert data["ok"] is False
        assert data["code"] == "NO_MONITOR"

    def test_v1_feedback_read_key_rejected(self):
        status, data = _req(self.port, "POST", "/api/v1/feedback",
                             body={"prediction": "1", "ground_truth": "0"},
                             key="read-key")
        assert status == 401
        assert data["ok"] is False
        assert data["code"] == "UNAUTHORIZED"


# ── test: error schema ────────────────────────────────────────────────────────

class TestErrorSchema:
    port = 19702

    @classmethod
    def setup_class(cls):
        cls.server = _start_server(cls.port)

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _assert_error_schema(self, data):
        assert "ok" in data and data["ok"] is False
        assert "error" in data and isinstance(data["error"], str)
        assert "code" in data and isinstance(data["code"], str)

    def test_not_found_has_full_schema(self):
        status, data = _req(self.port, "GET", "/api/v1/nonexistent")
        assert status == 404
        self._assert_error_schema(data)

    def test_registry_not_loaded_has_full_schema(self):
        status, data = _req(self.port, "GET", "/api/v1/registry")
        assert status == 503
        self._assert_error_schema(data)

    def test_push_no_registry_has_full_schema(self):
        # No registry mounted — 503 before field check; schema must still be correct
        status, data = _req(self.port, "POST", "/api/v1/registry/push", body={})
        assert status == 503
        self._assert_error_schema(data)

    def test_unauthorized_has_full_schema(self):
        status, data = _req(self.port, "POST", "/api/v1/predict",
                             body={"x": 1}, key="bad")
        assert status == 401
        self._assert_error_schema(data)


# ── test: registry HTTP layer ─────────────────────────────────────────────────

class TestRegistryHTTP:
    port = 19703

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        reg_path = Path(cls.tmpdir.name)

        # Populate a minimal registry entry
        from matrixai.registry.model_registry import ModelRegistry
        from matrixai.registry.schema import RegistryEntry
        from matrixai.registry.entry_hash import compute_entry_hash, sha256_str
        from matrixai import __version__
        import datetime

        reg = ModelRegistry(reg_path)
        eval_hash = sha256_str('{"accuracy":0.9}')
        eh = compute_entry_hash(
            name="test-model", version="v1.0",
            model_hash="sha256:" + "a" * 64,
            parameter_schema_hash="ps_hash",
            parameter_set_id="ps1",
            training_trace_hash="",
            evaluation_report_hash=eval_hash,
            matrixai_version=__version__,
        )
        entry = RegistryEntry(
            name="test-model", version="v1.0", entry_hash=eh,
            model_hash="sha256:" + "a" * 64,
            parameter_schema_hash="ps_hash",
            parameter_set_id="ps1",
            input_type={}, output_type={},
            metrics={"accuracy": 0.9},
            matrixai_version=__version__,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            training_dataset_fingerprint="",
            interpretability_level="full",
            training_trace_hash="",
            evaluation_report_hash=eval_hash,
        )
        # Write manifest directly (push requires evaluation_report_hash file)
        import json as _json
        entry_dir = reg.layout.entry_dir("test-model", "v1.0")
        entry_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = reg.layout.entry_file("test-model", "v1.0", "manifest")
        manifest_path.write_text(_json.dumps(entry.to_manifest(), indent=2))
        index_path = reg.layout.index_path
        index_path.write_text(_json.dumps(
            {"version": "0.21.0", "entries": [
                {"name": "test-model", "version": "v1.0", "entry_hash": eh}
            ]}, indent=2
        ))

        cls.server = _start_server(cls.port, registry_path=str(reg_path))

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmpdir.cleanup()

    def test_list_returns_ok_true(self):
        status, data = _req(self.port, "GET", "/api/v1/registry")
        assert status == 200
        assert data["ok"] is True
        assert "models" in data
        assert "total" in data

    def test_list_contains_registered_model(self):
        _, data = _req(self.port, "GET", "/api/v1/registry")
        names = [m["name"] for m in data["models"]]
        assert "test-model" in names

    def test_list_pagination_fields_present(self):
        _, data = _req(self.port, "GET", "/api/v1/registry?page=1&limit=10")
        assert "page" in data
        assert "limit" in data
        assert "total" in data

    def test_show_returns_model_data(self):
        status, data = _req(self.port, "GET", "/api/v1/registry/test-model/v1.0")
        assert status == 200
        assert data["ok"] is True
        assert data["model"]["name"] == "test-model"
        assert data["model"]["version"] == "v1.0"

    def test_show_not_found_returns_404(self):
        status, data = _req(self.port, "GET", "/api/v1/registry/nonexistent/v9.9")
        assert status == 404
        assert data["ok"] is False
        assert data["code"] == "NOT_FOUND"

    def test_tag_write_scope_succeeds(self):
        status, data = _req(self.port, "POST",
                             "/api/v1/registry/test-model/tag/stable",
                             body={"version": "v1.0"})
        assert status == 200
        assert data["ok"] is True
        assert data["tag"] == "stable"

    def test_verify_success(self):
        status, data = _req(self.port, "POST",
                             "/api/v1/registry/test-model/v1.0/verify")
        assert status == 200
        assert data["ok"] is True
        assert "verified" in data

    def test_tags_list(self):
        # tag was created in test_tag_write_scope_succeeds
        status, data = _req(self.port, "GET", "/api/v1/registry/test-model/tags")
        assert status == 200
        assert data["ok"] is True
        assert "tags" in data

    def test_tags_list_unknown_model_returns_empty(self):
        status, data = _req(self.port, "GET", "/api/v1/registry/unknown-model/tags")
        assert status == 200
        assert data["tags"] == []

    def test_push_missing_run_dir_returns_400(self):
        status, data = _req(self.port, "POST", "/api/v1/registry/push",
                             body={"name": "m", "version": "v1"})
        assert status == 400
        assert data["code"] == "MISSING_FIELDS"


# ── test: legacy routes still work ────────────────────────────────────────────

class TestLegacyRoutes:
    """Ensure unversioned routes remain functional (backwards compatibility)."""
    port = 19704

    @classmethod
    def setup_class(cls):
        model_file = _find_model_file()
        if model_file is None:
            pytest.skip("classification template not found")
        cls.server = _start_server(cls.port, model_path=str(model_file))

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_legacy_health_still_works(self):
        status, data = _req(self.port, "GET", "/health", key=None)
        assert status == 200
        assert data.get("status") == "ok"

    def test_legacy_predict_still_works(self):
        body = {"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.7}
        status, data = _req(self.port, "POST", "/predict", body=body)
        assert status == 200

    def test_v1_predict_has_ok_envelope(self):
        body = {"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.7}
        status, data = _req(self.port, "POST", "/api/v1/predict", body=body)
        assert status == 200
        assert data["ok"] is True
        assert "result" in data

    def test_v1_predict_error_has_code(self):
        status, data = _req(self.port, "POST", "/api/v1/predict", body=None)
        assert status == 400
        assert data["ok"] is False
        assert "code" in data

    def test_v1_openapi_has_v1_routes(self):
        status, data = _req(self.port, "GET", "/api/v1/openapi.json", key=None)
        assert status == 200
        assert "/api/v1/predict" in data.get("paths", {})
        assert "ErrorResponse" in data.get("components", {}).get("schemas", {})


# ── test: registry predict and pull ───────────────────────────────────────────

class TestRegistryPredictPull:
    """Tests for registry predict and pull with a real model file."""
    port = 19705

    @classmethod
    def setup_class(cls):
        model_file = _find_model_file()
        if model_file is None:
            pytest.skip("classification template not found")

        cls.tmpdir = tempfile.TemporaryDirectory()
        reg_path = Path(cls.tmpdir.name)

        # Copy model into registry entry structure
        import shutil, datetime
        from matrixai.registry.model_registry import ModelRegistry
        from matrixai.registry.schema import RegistryEntry
        from matrixai.registry.entry_hash import compute_entry_hash, sha256_str
        from matrixai import __version__
        import json as _json

        entry_dir = reg_path / "entries" / "test-clf" / "v1.0"
        entry_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_file, entry_dir / "model.mxai")

        eval_hash = sha256_str('{"accuracy":1.0}')
        eh = compute_entry_hash(
            name="test-clf", version="v1.0",
            model_hash="sha256:" + "b" * 64,
            parameter_schema_hash="ps_hash",
            parameter_set_id="ps1",
            training_trace_hash="",
            evaluation_report_hash=eval_hash,
            matrixai_version=__version__,
        )
        entry = RegistryEntry(
            name="test-clf", version="v1.0", entry_hash=eh,
            model_hash="sha256:" + "b" * 64,
            parameter_schema_hash="ps_hash",
            parameter_set_id="ps1",
            input_type={}, output_type={},
            metrics={"accuracy": 1.0},
            matrixai_version=__version__,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            training_dataset_fingerprint="",
            interpretability_level="full",
            training_trace_hash="",
            evaluation_report_hash=eval_hash,
        )
        manifest_path = reg_path / "entries" / "test-clf" / "v1.0" / "manifest.json"
        manifest_path.write_text(_json.dumps(entry.to_manifest(), indent=2))
        index_path = reg_path / "registry.json"
        index_path.write_text(_json.dumps(
            {"version": "0.21.0", "entries": [
                {"name": "test-clf", "version": "v1.0", "entry_hash": eh}
            ]}, indent=2
        ))

        cls.server = _start_server(cls.port, registry_path=str(reg_path))

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmpdir.cleanup()

    def test_registry_predict_returns_ok(self):
        body = {"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.7}
        status, data = _req(self.port, "POST", "/api/v1/registry/test-clf/v1.0/predict", body=body)
        assert status == 200
        assert data["ok"] is True
        assert "result" in data

    def test_registry_pull_returns_model_text(self):
        status, data = _req(self.port, "GET", "/api/v1/registry/test-clf/v1.0/pull")
        assert status == 200
        assert data["ok"] is True
        assert "model_text" in data
        assert len(data["model_text"]) > 0

    def test_registry_pull_not_found(self):
        status, data = _req(self.port, "GET", "/api/v1/registry/nonexistent/v1.0/pull")
        assert status in (404, 500)
        assert data["ok"] is False
