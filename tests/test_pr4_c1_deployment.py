"""PR4-C1 — Deployment packaging tests.

Covers:
- pack_model generates Dockerfile, docker-compose.yml, .env.example when --docker
- Dockerfile contains HEALTHCHECK, correct CMD, no hardcoded secrets
- docker-compose.yml references MATRIXAI_API_KEY as required, uses env_file
- .env.example documents all required and optional variables
- MATRIXAI_ALLOW_REAL_ACTIONS env var is read by serve_model
- Existing pack behavior (model copy, params copy, framework copy) is preserved
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from matrixai.pack import pack_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_MXAI = """\
PROJECT TestDeploy

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
def model_file(tmp_path: Path) -> Path:
    p = tmp_path / "deploy_model.mxai"
    p.write_text(MINIMAL_MXAI)
    return p


@pytest.fixture()
def outdir(tmp_path: Path) -> Path:
    return tmp_path / "dist"


# ---------------------------------------------------------------------------
# Non-docker pack — existing behavior preserved
# ---------------------------------------------------------------------------

class TestPackBasic:
    def test_copies_model_file(self, model_file, outdir):
        rc = pack_model(model_file, None, docker=False, outdir=outdir)
        assert rc == 0
        assert (outdir / model_file.name).exists()

    def test_copies_framework_directory(self, model_file, outdir):
        rc = pack_model(model_file, None, docker=False, outdir=outdir)
        assert rc == 0
        assert (outdir / "matrixai").is_dir()

    def test_no_docker_files_when_flag_absent(self, model_file, outdir):
        pack_model(model_file, None, docker=False, outdir=outdir)
        assert not (outdir / "Dockerfile").exists()
        assert not (outdir / "docker-compose.yml").exists()
        assert not (outdir / ".env.example").exists()

    def test_returns_1_for_missing_model(self, tmp_path, outdir):
        rc = pack_model(tmp_path / "nonexistent.mxai", None, docker=False, outdir=outdir)
        assert rc == 1


# ---------------------------------------------------------------------------
# Docker artifact generation
# ---------------------------------------------------------------------------

class TestDockerArtifacts:
    def test_generates_dockerfile(self, model_file, outdir):
        pack_model(model_file, None, docker=True, outdir=outdir)
        assert (outdir / "Dockerfile").exists()

    def test_generates_compose(self, model_file, outdir):
        pack_model(model_file, None, docker=True, outdir=outdir)
        assert (outdir / "docker-compose.yml").exists()

    def test_generates_env_example(self, model_file, outdir):
        pack_model(model_file, None, docker=True, outdir=outdir)
        assert (outdir / ".env.example").exists()

    def test_returns_0_on_success(self, model_file, outdir):
        rc = pack_model(model_file, None, docker=True, outdir=outdir)
        assert rc == 0


# ---------------------------------------------------------------------------
# Dockerfile content
# ---------------------------------------------------------------------------

class TestDockerfileContent:
    @pytest.fixture(autouse=True)
    def _pack(self, model_file, outdir):
        pack_model(model_file, None, docker=True, outdir=outdir)
        self.dockerfile = (outdir / "Dockerfile").read_text()

    def test_uses_python311_slim(self):
        assert "python:3.11-slim" in self.dockerfile

    def test_has_healthcheck(self):
        assert "HEALTHCHECK" in self.dockerfile
        assert "/health" in self.dockerfile

    def test_exposes_port_8000(self):
        assert "EXPOSE 8000" in self.dockerfile

    def test_cmd_includes_model_filename(self):
        assert "deploy_model.mxai" in self.dockerfile

    def test_cmd_includes_host_0000(self):
        assert "0.0.0.0" in self.dockerfile

    def test_no_hardcoded_api_key(self):
        # API key must come from env, never hardcoded in image
        assert "MATRIXAI_API_KEY=" not in self.dockerfile

    def test_copies_matrixai_framework(self):
        assert "COPY matrixai/" in self.dockerfile

    def test_copies_model_file(self):
        assert "COPY deploy_model.mxai" in self.dockerfile

    def test_params_not_copied_when_absent(self):
        # No params provided — no COPY params line
        assert "COPY params" not in self.dockerfile


class TestDockerfileWithParams:
    """Tests that --params wires correctly into the generated Dockerfile.

    validate_parameter_set is patched to bypass hash/weight checks — those
    are tested exhaustively in the parameter store tests.  Here we only care
    that pack_model plumbs the params path into the Docker artifacts.
    """

    @pytest.fixture()
    def params_file(self, tmp_path):
        import json
        p = tmp_path / "params.best.json"
        p.write_text(json.dumps({
            "parameter_set_id": "test-ps-001",
            "model_hash": "mxai_placeholder",
            "parameter_schema_hash": "params_placeholder",
            "source": "test",
            "parameters": {},
            "metrics": {},
        }))
        return p

    @pytest.fixture()
    def _skip_validation(self, monkeypatch):
        from matrixai.parameters.store import ParameterCompatibilityResult
        monkeypatch.setattr(
            "matrixai.pack.validate_parameter_set",
            lambda *_: ParameterCompatibilityResult(),
        )

    def test_copies_params_file(self, model_file, params_file, outdir, _skip_validation):
        pack_model(model_file, params_file, docker=True, outdir=outdir)
        dockerfile = (outdir / "Dockerfile").read_text()
        assert "COPY params.best.json" in dockerfile

    def test_cmd_includes_params_flag(self, model_file, params_file, outdir, _skip_validation):
        pack_model(model_file, params_file, docker=True, outdir=outdir)
        dockerfile = (outdir / "Dockerfile").read_text()
        assert "--params" in dockerfile


# ---------------------------------------------------------------------------
# docker-compose.yml content
# ---------------------------------------------------------------------------

class TestComposeContent:
    @pytest.fixture(autouse=True)
    def _pack(self, model_file, outdir):
        pack_model(model_file, None, docker=True, outdir=outdir)
        self.compose = (outdir / "docker-compose.yml").read_text()

    def test_has_service_definition(self):
        assert "matrixai-server" in self.compose

    def test_uses_env_file(self):
        # Must load variables from .env, not inline secrets
        assert "env_file" in self.compose

    def test_exposes_port_8000(self):
        assert "8000" in self.compose

    def test_has_healthcheck(self):
        assert "healthcheck" in self.compose

    def test_has_restart_policy(self):
        assert "restart" in self.compose

    def test_has_registry_volume(self):
        assert "matrixai-registry" in self.compose

    def test_has_build_directive(self):
        assert "build" in self.compose


# ---------------------------------------------------------------------------
# .env.example content
# ---------------------------------------------------------------------------

class TestEnvExample:
    @pytest.fixture(autouse=True)
    def _pack(self, model_file, outdir):
        pack_model(model_file, None, docker=True, outdir=outdir)
        self.env_example = (outdir / ".env.example").read_text()

    def test_documents_api_key(self):
        assert "MATRIXAI_API_KEY" in self.env_example

    def test_documents_signing_key(self):
        assert "MATRIXAI_ACTION_SIGNING_KEY" in self.env_example

    def test_documents_allow_real_actions(self):
        assert "MATRIXAI_ALLOW_REAL_ACTIONS" in self.env_example

    def test_documents_registry_signing_key(self):
        assert "MATRIXAI_REGISTRY_SIGNING_KEY" in self.env_example

    def test_allow_real_actions_defaults_false(self):
        assert "MATRIXAI_ALLOW_REAL_ACTIONS=false" in self.env_example

    def test_no_real_secret_values(self):
        # The example file must not contain any filled-in secret — just placeholders
        for line in self.env_example.splitlines():
            if line.startswith("MATRIXAI_ACTION_SIGNING_KEY="):
                value = line.split("=", 1)[1].strip()
                assert value == "", f"Signing key example should be empty, got: {value!r}"


# ---------------------------------------------------------------------------
# MATRIXAI_ALLOW_REAL_ACTIONS env var in server
# ---------------------------------------------------------------------------

class TestAllowRealActionsEnvVar:
    """serve_model must honour MATRIXAI_ALLOW_REAL_ACTIONS from the environment."""

    def test_env_var_true_enables_real_actions(self, model_file, monkeypatch):
        import threading
        from matrixai.server import serve_model

        monkeypatch.setenv("MATRIXAI_ALLOW_REAL_ACTIONS", "true")
        monkeypatch.setenv("MATRIXAI_API_KEY", "test-key")

        captured = {}

        original_MatrixAIHTTPServer = __import__(
            "matrixai.server", fromlist=["MatrixAIHTTPServer"]
        ).MatrixAIHTTPServer

        class CapturingServer(original_MatrixAIHTTPServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["allow_real_actions"] = self.matrixai_allow_real_actions
                # Raise to abort serve_forever immediately
                raise RuntimeError("abort")

        import matrixai.server as srv_module
        monkeypatch.setattr(srv_module, "MatrixAIHTTPServer", CapturingServer)

        try:
            serve_model(model_file, None, "127.0.0.1", 19999, "stdlib", allow_real_actions=False)
        except RuntimeError:
            pass

        assert captured.get("allow_real_actions") is True

    def test_env_var_false_leaves_flag_false(self, model_file, monkeypatch):
        from matrixai.server import serve_model

        monkeypatch.setenv("MATRIXAI_ALLOW_REAL_ACTIONS", "false")
        monkeypatch.setenv("MATRIXAI_API_KEY", "test-key")

        captured = {}

        original_MatrixAIHTTPServer = __import__(
            "matrixai.server", fromlist=["MatrixAIHTTPServer"]
        ).MatrixAIHTTPServer

        class CapturingServer(original_MatrixAIHTTPServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["allow_real_actions"] = self.matrixai_allow_real_actions
                raise RuntimeError("abort")

        import matrixai.server as srv_module
        monkeypatch.setattr(srv_module, "MatrixAIHTTPServer", CapturingServer)

        try:
            serve_model(model_file, None, "127.0.0.1", 19998, "stdlib", allow_real_actions=False)
        except RuntimeError:
            pass

        assert captured.get("allow_real_actions") is False

    def test_env_var_accepts_1_as_true(self, model_file, monkeypatch):
        from matrixai.server import serve_model

        monkeypatch.setenv("MATRIXAI_ALLOW_REAL_ACTIONS", "1")
        monkeypatch.setenv("MATRIXAI_API_KEY", "test-key")

        captured = {}

        original_MatrixAIHTTPServer = __import__(
            "matrixai.server", fromlist=["MatrixAIHTTPServer"]
        ).MatrixAIHTTPServer

        class CapturingServer(original_MatrixAIHTTPServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["allow_real_actions"] = self.matrixai_allow_real_actions
                raise RuntimeError("abort")

        import matrixai.server as srv_module
        monkeypatch.setattr(srv_module, "MatrixAIHTTPServer", CapturingServer)

        try:
            serve_model(model_file, None, "127.0.0.1", 19997, "stdlib", allow_real_actions=False)
        except RuntimeError:
            pass

        assert captured.get("allow_real_actions") is True

    def test_explicit_flag_true_overrides_env_false(self, model_file, monkeypatch):
        from matrixai.server import serve_model

        monkeypatch.setenv("MATRIXAI_ALLOW_REAL_ACTIONS", "false")
        monkeypatch.setenv("MATRIXAI_API_KEY", "test-key")

        captured = {}

        original_MatrixAIHTTPServer = __import__(
            "matrixai.server", fromlist=["MatrixAIHTTPServer"]
        ).MatrixAIHTTPServer

        class CapturingServer(original_MatrixAIHTTPServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["allow_real_actions"] = self.matrixai_allow_real_actions
                raise RuntimeError("abort")

        import matrixai.server as srv_module
        monkeypatch.setattr(srv_module, "MatrixAIHTTPServer", CapturingServer)

        try:
            # CLI flag=True is passed explicitly; env says false → flag wins
            serve_model(model_file, None, "127.0.0.1", 19996, "stdlib", allow_real_actions=True)
        except RuntimeError:
            pass

        assert captured.get("allow_real_actions") is True
