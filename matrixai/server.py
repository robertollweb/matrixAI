# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import sys


class RateLimiter:
    """Sliding-window per-IP rate limiter.

    Thread-safe. A limit of 0 disables rate limiting entirely.
    """

    def __init__(self, requests_per_minute: int):
        self._rpm = requests_per_minute
        self._windows: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def disabled(self) -> bool:
        return self._rpm <= 0

    def is_allowed(self, ip: str) -> bool:
        if self.disabled:
            return True
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            timestamps = [t for t in self._windows.get(ip, []) if t > cutoff]
            if len(timestamps) >= self._rpm:
                self._windows[ip] = timestamps
                return False
            timestamps.append(now)
            self._windows[ip] = timestamps
            # Evict IPs with no activity in the last 60 s when the dict grows large.
            # Runs at most once per request that pushes past the threshold.
            if len(self._windows) > 10_000:
                self._windows = {
                    k: v for k, v in self._windows.items()
                    if any(t > cutoff for t in v)
                }
            return True

from matrixai.parser import parse_file
from matrixai.runtime import MatrixAIRuntime
from matrixai.parameters import load_parameter_set, validate_parameter_set
from matrixai.cli import _json_safe

SWAGGER_UI_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>MatrixAI Swagger UI</title>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.0.0/swagger-ui.css">
  <style>html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; } *, *:before, *:after { box-sizing: inherit; } body { margin: 0; background: #fafafa; }</style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.0.0/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function() {
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
        layout: "BaseLayout"
      })
    }
  </script>
</body>
</html>
"""

class MatrixAIHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, program, parameter_set, backend, api_key,
                 contracts=None, allow_real_actions=False, signing_key=None,
                 rate_limiter: RateLimiter | None = None,
                 cors_origins: list[str] | None = None,
                 monitor=None,
                 api_key_read: str | None = None,
                 registry=None):
        super().__init__(server_address, RequestHandlerClass)
        self.matrixai_program = program
        self.matrixai_parameter_set = parameter_set
        self.matrixai_backend = backend
        self.matrixai_api_key = api_key
        self.matrixai_api_key_read = api_key_read      # read-only scope (PR5-C6)
        self.matrixai_registry = registry              # ModelRegistry | None (PR5-C6)
        self.matrixai_contracts = contracts or []   # list[ActionContractSpec]
        self.matrixai_allow_real_actions = allow_real_actions
        self.matrixai_signing_key = signing_key
        self.matrixai_rate_limiter = rate_limiter or RateLimiter(60)
        self.matrixai_cors_origins = cors_origins if cors_origins is not None else ["*"]
        self.matrixai_monitor = monitor  # ProductionMonitor | None (P22 drift metrics)
        from collections import defaultdict
        from matrixai.actions.approval import ApprovalStore
        from matrixai.actions.dryrun import RateTracker
        self.matrixai_approval_store = ApprovalStore()
        self.matrixai_rate_trackers: dict = defaultdict(RateTracker)
        self.matrixai_metrics = {
            "uptime_start": time.time(),
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "requests_rate_limited": 0,
            "items_processed": 0,
            "last_request_ms": 0.0,
            "action_executions_total": 0,
            "action_dry_runs_total": 0,
            "action_signed_total": 0,
        }

class MatrixAIServerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self):
        if not self._check_rate_limit():
            return
        self.send_response(204)
        self._write_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        # ── /api/v1/* versioned surface (PR5-C6) ──────────────────────────────
        if self.path.startswith("/api/v1/"):
            if not self._check_rate_limit(): return
            self._dispatch_v1_post(self.path[len("/api/v1"):])
            return
        # ── legacy unversioned routes (kept as backwards-compatible aliases) ──
        if self.path == "/predict":
            started_at = time.time()
            self._increment_metric("requests_total")
            if not self._check_rate_limit(): return
            if not self._check_auth(started_at): return
            self._handle_predict(started_at)
        elif self.path == "/execute-action":
            started_at = time.time()
            self._increment_metric("requests_total")
            if not self._check_rate_limit(): return
            if not self._check_auth(started_at): return
            self._handle_execute_action(started_at)
        elif self.path == "/feedback":
            self._increment_metric("requests_total")
            if not self._check_rate_limit(): return
            if not self._check_auth(): return
            self._handle_feedback()
        else:
            self._send_response(404, {"error": "Not Found. Try POST /predict or POST /execute-action."})

    def do_GET(self):
        self._increment_metric("requests_total")
        path = self.path.split("?", 1)[0]
        # ── /api/v1/* versioned surface (PR5-C6) ──────────────────────────────
        if path.startswith("/api/v1/") or path == "/api/v1":
            if not self._check_rate_limit(): return
            self._dispatch_v1_get(path[len("/api/v1"):] or "/", self.path)
            return
        # ── legacy unversioned routes ──────────────────────────────────────────
        if path == "/health":
            uptime = time.time() - self.server.matrixai_metrics["uptime_start"]
            metrics = dict(self.server.matrixai_metrics)
            metrics["uptime_seconds"] = round(uptime, 2)
            self._send_response(200, {
                "status": "ok",
                "service": "MatrixAI Server",
                "backend": self.server.matrixai_backend,
                "metrics": metrics
            })
        elif path == "/metrics":
            self._handle_metrics()
        elif path in ("/", "/docs"):
            self._send_html(200, SWAGGER_UI_HTML)
        elif path == "/openapi.json":
            self._send_response(200, self._generate_openapi())
        else:
            self._send_response(404, {"error": "Not Found. Try GET /docs or POST /predict."})

    # ── PR5-C6: /api/v1/* helpers ─────────────────────────────────────────────

    def _send_v1_error(self, status: int, code: str, message: str) -> None:
        self._send_response(status, {"ok": False, "error": message, "code": code})

    def _send_v1_success(self, data: dict, status: int = 200) -> None:
        self._send_response(status, {"ok": True, **data})

    def _check_auth_scope(self, write: bool = False) -> bool:
        """Return True if request carries a key with sufficient scope."""
        write_key = self.server.matrixai_api_key
        read_key = getattr(self.server, "matrixai_api_key_read", None)
        auth = self.headers.get("Authorization", "")
        xkey = self.headers.get("X-API-Key", "")
        if auth == f"Bearer {write_key}" or xkey == write_key:
            return True
        if not write and read_key and (auth == f"Bearer {read_key}" or xkey == read_key):
            return True
        self._send_v1_error(401, "UNAUTHORIZED",
                            "Provide Authorization: Bearer <key> or X-API-Key: <key>.")
        return False

    def _run_inference(self, program, parameter_set, backend: str, body_bytes: bytes):
        """Run model inference and return (result, error_response).

        Returns (result, None) on success or (None, error_dict) on failure.
        """
        try:
            input_data = json.loads(body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, {"status": 400, "code": "INVALID_JSON", "error": "Invalid JSON payload"}

        try:
            if backend == "torch":
                from matrixai.compiler.torch_forward import TorchForwardRunner
                runner = TorchForwardRunner()
                run_logic = lambda item: runner.run(program, item, parameter_set)
            else:
                runtime = MatrixAIRuntime()
                runtime_params = parameter_set.runtime_parameters() if parameter_set else None
                run_logic = lambda item: runtime.run(program, item, parameters=runtime_params)

            if isinstance(input_data, list):
                result = [run_logic(item) for item in input_data]
            else:
                result = run_logic(input_data)
            return result, None
        except ValueError as exc:
            return None, {"status": 400, "code": "INFERENCE_ERROR", "error": str(exc)}
        except Exception as exc:
            return None, {"status": 500, "code": "INTERNAL_ERROR", "error": f"Internal error: {exc}"}

    def _dispatch_v1_get(self, sub: str, full_path: str) -> None:
        """Route GET /api/v1{sub} requests."""
        import re
        qs = full_path[len("/api/v1") + len(sub):].lstrip("?") if "?" in full_path else ""

        # Aliases for legacy routes
        if sub == "/health" or sub == "/health/":
            uptime = time.time() - self.server.matrixai_metrics["uptime_start"]
            m = dict(self.server.matrixai_metrics)
            m["uptime_seconds"] = round(uptime, 2)
            self._send_v1_success({"service": "MatrixAI Server",
                                   "backend": self.server.matrixai_backend,
                                   "status": "ok", "metrics": m})
            return
        if sub in ("/metrics", "/metrics/"):
            self._handle_metrics(); return
        if sub in ("/openapi.json",):
            self._send_response(200, self._generate_v1_openapi()); return

        # Registry endpoints
        registry = getattr(self.server, "matrixai_registry", None)
        if sub in ("/registry", "/registry/"):
            if not self._check_auth_scope(write=False): return
            self._handle_v1_registry_list(qs, registry); return

        m = re.fullmatch(r"/registry/([^/]+)/tags", sub)
        if m:
            if not self._check_auth_scope(write=False): return
            self._handle_v1_registry_tags(m.group(1), registry); return

        m = re.fullmatch(r"/registry/([^/]+)/([^/]+)/pull", sub)
        if m:
            if not self._check_auth_scope(write=False): return
            self._handle_v1_registry_pull(m.group(1), m.group(2), registry); return

        m = re.fullmatch(r"/registry/([^/]+)/([^/]+)", sub)
        if m:
            if not self._check_auth_scope(write=False): return
            self._handle_v1_registry_show(m.group(1), m.group(2), registry); return

        self._send_v1_error(404, "NOT_FOUND", f"No GET route for /api/v1{sub}")

    def _dispatch_v1_post(self, sub: str) -> None:
        """Route POST /api/v1{sub} requests."""
        import re
        self._increment_metric("requests_total")

        # Versioned aliases — proper {ok,error,code} envelope
        if sub == "/predict":
            started_at = time.time()
            if not self._check_auth_scope(write=False): return
            self._handle_v1_predict(started_at); return
        if sub == "/execute-action":
            started_at = time.time()
            if not self._check_auth_scope(write=True): return
            self._handle_v1_execute_action(started_at); return
        if sub == "/feedback":
            if not self._check_auth_scope(write=True): return
            self._handle_v1_feedback(); return

        # Registry endpoints
        registry = getattr(self.server, "matrixai_registry", None)
        if sub in ("/registry/push", "/registry/push/"):
            if not self._check_auth_scope(write=True): return
            self._handle_v1_registry_push(registry); return

        m = re.fullmatch(r"/registry/([^/]+)/([^/]+)/verify", sub)
        if m:
            if not self._check_auth_scope(write=False): return
            self._handle_v1_registry_verify(m.group(1), m.group(2), registry); return

        m = re.fullmatch(r"/registry/([^/]+)/([^/]+)/predict", sub)
        if m:
            if not self._check_auth_scope(write=False): return
            self._handle_v1_registry_predict(m.group(1), m.group(2), registry); return

        m = re.fullmatch(r"/registry/([^/]+)/tag/([^/]+)", sub)
        if m:
            if not self._check_auth_scope(write=True): return
            self._handle_v1_registry_tag(m.group(1), m.group(2), registry); return

        self._send_v1_error(404, "NOT_FOUND", f"No POST route for /api/v1{sub}")

    def _handle_v1_registry_list(self, qs: str, registry) -> None:
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        try:
            from urllib.parse import parse_qs
            params = parse_qs(qs)
            page = max(1, int(params.get("page", ["1"])[0]))
            limit = max(1, min(100, int(params.get("limit", ["20"])[0])))
            name_filter = params.get("name", [None])[0]

            filters = {"name": name_filter} if name_filter else None
            all_entries = registry.list(filters=filters)
            total = len(all_entries)
            start = (page - 1) * limit
            page_entries = all_entries[start:start + limit]

            self._send_v1_success({
                "models": [
                    {"name": e.name, "version": e.version,
                     "matrixai_version": e.matrixai_version,
                     "metrics": e.metrics, "created_at": e.created_at}
                    for e in page_entries
                ],
                "page": page, "limit": limit, "total": total,
            })
        except Exception as exc:
            self._send_v1_error(500, "INTERNAL_ERROR", str(exc))

    def _handle_v1_registry_show(self, name: str, version: str, registry) -> None:
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        try:
            from matrixai.registry.model_registry import EntryNotFoundError
            entry = registry.get(name, version)
            self._send_v1_success({"model": {
                "name": entry.name, "version": entry.version,
                "matrixai_version": entry.matrixai_version,
                "entry_hash": entry.entry_hash,
                "model_hash": entry.model_hash,
                "parameter_set_id": entry.parameter_set_id,
                "metrics": entry.metrics,
                "created_at": entry.created_at,
                "interpretability_level": entry.interpretability_level,
            }})
        except Exception as exc:
            code = "NOT_FOUND" if "not found" in str(exc).lower() else "INTERNAL_ERROR"
            status = 404 if code == "NOT_FOUND" else 500
            self._send_v1_error(status, code, str(exc))

    def _handle_v1_registry_verify(self, name: str, version: str, registry) -> None:
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        try:
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ok = registry.verify(name, version)
            version_warnings = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self._send_v1_success({"verified": ok, "warnings": version_warnings})
        except Exception as exc:
            code = "NOT_FOUND" if "not found" in str(exc).lower() else "INTEGRITY_MISMATCH"
            status = 404 if code == "NOT_FOUND" else 409
            self._send_v1_error(status, code, str(exc))

    def _handle_v1_registry_predict(self, name: str, version: str, registry) -> None:
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else b""
        if not body:
            self._send_v1_error(400, "EMPTY_PAYLOAD", "Request body required.")
            return
        try:
            from matrixai.parser import parse_file as _parse_file
            from matrixai.parameters import load_parameter_set as _load_ps
            entry_dir = registry.layout.entry_dir(
                name, registry._resolve_version(name, version)
            )
            model_path = entry_dir / "model.mxai"
            params_path = entry_dir / "params.json"
            if not model_path.exists():
                self._send_v1_error(404, "MODEL_FILE_NOT_FOUND",
                                    f"model.mxai not found in registry for {name}@{version}.")
                return
            program = _parse_file(model_path)
            parameter_set = _load_ps(params_path) if params_path.exists() else None
            backend = self.server.matrixai_backend
        except Exception as exc:
            self._send_v1_error(500, "LOAD_ERROR", f"Failed to load model: {exc}")
            return

        result, err = self._run_inference(program, parameter_set, backend, body)
        if err:
            self._send_v1_error(err["status"], err["code"], err["error"])
        else:
            self._send_v1_success({"result": result})

    def _handle_v1_registry_push(self, registry) -> None:
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        cl = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(cl).decode("utf-8")) if cl else {}
        except json.JSONDecodeError:
            self._send_v1_error(400, "INVALID_JSON", "Invalid JSON body.")
            return

        run_dir = body.get("run_dir")
        name = body.get("name")
        version = body.get("version")
        if not all([run_dir, name, version]):
            self._send_v1_error(400, "MISSING_FIELDS",
                                "Required: name, version, run_dir.")
            return
        try:
            from pathlib import Path as _Path
            registry.push_run_dir(_Path(run_dir), name=name, version=version)
            self._send_v1_success({"name": name, "version": version}, status=201)
        except Exception as exc:
            code = "DUPLICATE_ENTRY" if "already exists" in str(exc) else "PUSH_ERROR"
            status = 409 if code == "DUPLICATE_ENTRY" else 500
            self._send_v1_error(status, code, str(exc))

    def _handle_v1_registry_tag(self, name: str, tag: str, registry) -> None:
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        cl = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(cl).decode("utf-8")) if cl else {}
        except json.JSONDecodeError:
            self._send_v1_error(400, "INVALID_JSON", "Invalid JSON body."); return
        version = body.get("version")
        if not version:
            self._send_v1_error(400, "MISSING_FIELDS", "Required: version."); return
        try:
            registry.tag(name, version, tag)
            self._send_v1_success({"name": name, "tag": tag, "version": version})
        except Exception as exc:
            code = "NOT_FOUND" if "not found" in str(exc).lower() else "TAG_ERROR"
            self._send_v1_error(404 if code == "NOT_FOUND" else 500, code, str(exc))

    def _handle_v1_execute_action(self, started_at: float) -> None:
        """POST /api/v1/execute-action — errors use {ok,error,code}; success returns ActionTrace."""
        contracts = self.server.matrixai_contracts
        if not contracts:
            self._increment_metric("requests_failed")
            self._send_v1_error(404, "NO_CONTRACT",
                                "No action contracts loaded. Start server with --contract.")
            return
        cl = int(self.headers.get("Content-Length", 0))
        if cl == 0:
            self._increment_metric("requests_failed")
            self._send_v1_error(400, "EMPTY_PAYLOAD", "Request body required.")
            return
        # Delegate to legacy handler for actual execution (returns ActionTrace with ok field)
        self._handle_execute_action(started_at)

    def _handle_v1_feedback(self) -> None:
        """POST /api/v1/feedback — full {ok,error,code} envelope on all paths."""
        monitor = getattr(self.server, "matrixai_monitor", None)
        if monitor is None:
            self._send_v1_error(404, "NO_MONITOR",
                                "No continual policy loaded. Start server with --continual-policy.")
            return
        cl = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(cl).decode("utf-8")) if cl else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_v1_error(400, "INVALID_JSON", "Invalid JSON body.")
            return
        prediction = body.get("prediction")
        ground_truth = body.get("ground_truth")
        if prediction is None or ground_truth is None:
            self._send_v1_error(400, "MISSING_FIELDS",
                                "Both 'prediction' and 'ground_truth' are required.")
            return
        try:
            obs = monitor.record(
                str(prediction), str(ground_truth),
                trace_id=str(body.get("trace_id", "")),
                observed_at=body.get("observed_at"),
                parameter_set_id=str(body.get("parameter_set_id", "")),
            )
            self._increment_metric("requests_successful")
            self._send_v1_success({"recorded": True, "correct": obs.correct,
                                   "trace_id": obs.trace_id})
        except Exception as exc:
            self._send_v1_error(500, "MONITOR_ERROR", str(exc))

    def _handle_v1_predict(self, started_at: float | None = None) -> None:
        """POST /api/v1/predict — wrapped in {ok, result} envelope."""
        started_at = started_at or time.time()
        cl = int(self.headers.get("Content-Length", 0))
        if cl == 0:
            self._increment_metric("requests_failed")
            self._send_v1_error(400, "EMPTY_PAYLOAD", "Request body required.")
            return
        body = self.rfile.read(cl)
        result, err = self._run_inference(
            self.server.matrixai_program,
            self.server.matrixai_parameter_set,
            self.server.matrixai_backend,
            body,
        )
        elapsed = round((time.time() - started_at) * 1000, 3)
        self._set_metric("last_request_ms", elapsed)
        if err:
            self._increment_metric("requests_failed")
            self._send_v1_error(err["status"], err["code"], err["error"])
        else:
            item_count = len(result) if isinstance(result, list) else 1
            self._increment_metric("items_processed", item_count)
            self._increment_metric("requests_successful")
            self._send_v1_success({"result": result})

    def _handle_v1_registry_tags(self, name: str, registry) -> None:
        """GET /api/v1/registry/{name}/tags — list tags for a model."""
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        try:
            tags_dir = registry.layout.tags_dir / name
            if not tags_dir.exists():
                self._send_v1_success({"name": name, "tags": []})
                return
            tags = []
            for tag_path in sorted(tags_dir.iterdir()):
                if tag_path.is_file():
                    data = json.loads(tag_path.read_text())
                    tags.append({"tag": tag_path.name, "version": data.get("version", "")})
            self._send_v1_success({"name": name, "tags": tags})
        except Exception as exc:
            self._send_v1_error(500, "INTERNAL_ERROR", str(exc))

    def _handle_v1_registry_pull(self, name: str, version: str, registry) -> None:
        """GET /api/v1/registry/{name}/{version}/pull — return model text and params."""
        if registry is None:
            self._send_v1_error(503, "REGISTRY_NOT_LOADED",
                                "Start server with --registry to enable registry endpoints.")
            return
        try:
            resolved = registry._resolve_version(name, version)
            entry_dir = registry.layout.entry_dir(name, resolved)
            model_path = entry_dir / "model.mxai"
            params_path = entry_dir / "params.json"
            if not model_path.exists():
                self._send_v1_error(404, "MODEL_FILE_NOT_FOUND",
                                    f"model.mxai not found in registry for {name}@{resolved}.")
                return
            model_text = model_path.read_text(encoding="utf-8")
            params = json.loads(params_path.read_text()) if params_path.exists() else None
            self._send_v1_success({
                "name": name,
                "version": resolved,
                "model_text": model_text,
                "params": params,
            })
        except Exception as exc:
            code = "NOT_FOUND" if "not found" in str(exc).lower() else "INTERNAL_ERROR"
            self._send_v1_error(404 if code == "NOT_FOUND" else 500, code, str(exc))

    # ── end PR5-C6 ────────────────────────────────────────────────────────────

    def _handle_feedback(self) -> None:
        """Handle POST /feedback — record ground truth for drift monitoring.

        Expected body:
          {"prediction": "label_a", "ground_truth": "label_b"}
          Optional fields: "trace_id", "observed_at", "parameter_set_id"
        """
        monitor = getattr(self.server, "matrixai_monitor", None)
        if monitor is None:
            self._send_response(404, {
                "error": "No continual policy loaded. Start server with --continual-policy."
            })
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, KeyError):
            self._send_response(400, {"error": "Invalid JSON body"})
            return

        prediction = body.get("prediction")
        ground_truth = body.get("ground_truth")
        if prediction is None or ground_truth is None:
            self._send_response(400, {
                "error": "Both 'prediction' and 'ground_truth' are required"
            })
            return

        obs = monitor.record(
            str(prediction),
            str(ground_truth),
            trace_id=str(body.get("trace_id", "")),
            observed_at=body.get("observed_at"),
            parameter_set_id=str(body.get("parameter_set_id", "")),
        )
        self._increment_metric("requests_successful")
        self._send_response(200, {
            "recorded": True,
            "correct": obs.correct,
            "trace_id": obs.trace_id,
        })

    def _handle_metrics(self) -> None:
        """Serve GET /metrics — Prometheus text format (text/plain; version=0.0.4)."""
        body = self._format_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._write_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _format_prometheus(self) -> str:
        """Render server metrics in Prometheus exposition format."""
        uptime = time.time() - self.server.matrixai_metrics["uptime_start"]
        m = self.server.matrixai_metrics
        project = getattr(getattr(self.server, "matrixai_program", None), "project", "unknown")
        label = f'{{project="{project}"}}'

        lines: list[str] = []

        def counter(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{label} {value}")

        def gauge(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{label} {value}")

        # Model identity (info metric — always 1, value is in labels)
        ps = getattr(self.server, "matrixai_parameter_set", None)
        ps_id = getattr(ps, "parameter_set_id", "") if ps else ""
        model_hash = getattr(ps, "model_hash", "") if ps else ""
        info_label = f'{{project="{project}",parameter_set_id="{ps_id}",model_hash="{model_hash}"}}'
        lines.append("# HELP matrixai_model_info Model identity: project, parameter_set_id, model_hash.")
        lines.append("# TYPE matrixai_model_info gauge")
        lines.append(f"matrixai_model_info{info_label} 1")

        counter("matrixai_requests_total",
                "Total HTTP requests across all endpoints.",
                m.get("requests_total", 0))
        counter("matrixai_requests_successful_total",
                "HTTP requests that returned 200.",
                m.get("requests_successful", 0))
        counter("matrixai_requests_failed_total",
                "HTTP requests that returned 4xx/5xx.",
                m.get("requests_failed", 0))
        counter("matrixai_requests_rate_limited_total",
                "HTTP requests rejected by rate limiting (429).",
                m.get("requests_rate_limited", 0))
        counter("matrixai_items_processed_total",
                "Individual predictions processed (batch items counted separately).",
                m.get("items_processed", 0))
        gauge("matrixai_last_request_duration_milliseconds",
              "Wall time of the most recent request in milliseconds.",
              m.get("last_request_ms", 0.0))
        gauge("matrixai_uptime_seconds",
              "Server uptime in seconds.",
              round(uptime, 3))

        # Action execution metrics (P20)
        counter("matrixai_action_executions_total",
                "Total POST /execute-action requests processed.",
                m.get("action_executions_total", 0))
        counter("matrixai_action_dry_runs_total",
                "Dry-run simulations executed (one per execute-action call).",
                m.get("action_dry_runs_total", 0))
        counter("matrixai_action_signed_total",
                "ActionTraces signed with MATRIXAI_ACTION_SIGNING_KEY.",
                m.get("action_signed_total", 0))

        # P22 drift metrics — emitted only when a ProductionMonitor is attached
        monitor = getattr(self.server, "matrixai_monitor", None)
        if monitor is not None:
            try:
                wm = monitor.window_metrics()
                gauge("matrixai_drift_window_accuracy",
                      "Sliding-window prediction accuracy (0–1).",
                      round(wm.accuracy, 6))
                gauge("matrixai_drift_window_samples",
                      "Number of labeled observations in the current window.",
                      wm.samples)
                gauge("matrixai_drift_degradation_detected",
                      "1 if accuracy degradation was detected in the current window, else 0.",
                      1.0 if wm.degradation_detected else 0.0)
                gauge("matrixai_drift_actual_degradation",
                      "Measured accuracy drop below reference (positive = worse than reference).",
                      round(wm.actual_degradation, 6))
            except Exception:
                pass  # monitor not yet warmed up — skip drift metrics silently

        return "\n".join(lines) + "\n"

    def _check_rate_limit(self) -> bool:
        ip = self.client_address[0]
        if not self.server.matrixai_rate_limiter.is_allowed(ip):
            self._increment_metric("requests_failed")
            self._increment_metric("requests_rate_limited")
            self._send_response(429, {"error": "rate limit exceeded"}, extra_headers={"Retry-After": "60"})
            return False
        return True

    def _check_auth(self, started_at: float | None = None) -> bool:
        expected = self.server.matrixai_api_key
        auth_header = self.headers.get("Authorization", "")
        api_key_header = self.headers.get("X-API-Key", "")
        bearer_ok = auth_header == f"Bearer {expected}"
        key_ok = api_key_header == expected
        if not bearer_ok and not key_ok:
            self._increment_metric("requests_failed")
            if started_at is not None:
                self._set_metric("last_request_ms", round((time.time() - started_at) * 1000, 3))
            self._send_response(401, {"error": "Unauthorized. Provide Authorization: Bearer <key> or X-API-Key: <key>."})
            return False
        return True

    def _write_cors_headers(self) -> None:
        origins = self.server.matrixai_cors_origins
        if "*" in origins:
            self.send_header("Access-Control-Allow-Origin", "*")
        else:
            request_origin = self.headers.get("Origin", "")
            if request_origin and request_origin in origins:
                self.send_header("Access-Control-Allow-Origin", request_origin)
                self.send_header("Vary", "Origin")

    def _increment_metric(self, name: str, amount: int | float = 1) -> None:
        metrics = getattr(self.server, "matrixai_metrics", None)
        if isinstance(metrics, dict):
            metrics[name] = metrics.get(name, 0) + amount

    def _set_metric(self, name: str, value: int | float) -> None:
        metrics = getattr(self.server, "matrixai_metrics", None)
        if isinstance(metrics, dict):
            metrics[name] = value

    def _generate_predict_example(self) -> dict:
        program = getattr(self.server, "matrixai_program", None)
        vectors = list(getattr(program, "vectors", []) or [])
        if not vectors:
            return {}
        example: dict[str, Any] = {}
        for vector in vectors:
            for field in vector.fields:
                type_spec = vector.field_types.get(field)
                type_name = getattr(type_spec, "name", None) if type_spec else None
                if type_name == "Integer":
                    example[field] = 1
                elif type_name == "Boolean":
                    example[field] = True
                elif type_name in {"String", "Label"}:
                    example[field] = "value"
                else:
                    example[field] = 0.5
        return example

    def _generate_v1_openapi(self) -> dict:
        """Return OpenAPI 3.0 spec covering all /api/v1/* routes."""
        from matrixai import __version__
        has_registry = getattr(self.server, "matrixai_registry", None) is not None
        has_model = getattr(self.server, "matrixai_program", None) is not None
        has_read_key = bool(getattr(self.server, "matrixai_api_key_read", None))

        error_schema = {
            "type": "object",
            "required": ["ok", "error", "code"],
            "properties": {
                "ok": {"type": "boolean", "example": False},
                "error": {"type": "string"},
                "code": {"type": "string"},
            },
        }
        security_schemes = {
            "BearerAuth": {"type": "http", "scheme": "bearer"},
            "ApiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
        }
        if has_read_key:
            security_schemes["BearerAuthRead"] = {"type": "http", "scheme": "bearer",
                                                   "description": "Read-only key (MATRIXAI_API_KEY_READ)"}

        paths: dict = {
            "/api/v1/health": {"get": {
                "summary": "Server health", "security": [],
                "responses": {"200": {"description": "ok",
                                       "content": {"application/json": {"schema": {
                                           "type": "object",
                                           "properties": {"ok": {"type": "boolean"}, "status": {"type": "string"}},
                                       }}}}}}},
            "/api/v1/metrics": {"get": {
                "summary": "Prometheus metrics", "security": [],
                "responses": {"200": {"description": "Prometheus text format",
                                       "content": {"text/plain": {"schema": {"type": "string"}}}}}}},
        }
        if has_model:
            model_schemas = self._generate_openapi_schemas()
            paths["/api/v1/predict"] = {"post": {
                "summary": "Run inference on the loaded model",
                "security": [{"BearerAuth": []}, {"ApiKeyHeader": []}],
                "requestBody": {"required": True, "content": {"application/json": {"schema": {
                    "oneOf": [{"$ref": "#/components/schemas/PredictionInput"},
                               {"$ref": "#/components/schemas/PredictionBatch"}],
                }}}},
                "responses": {
                    "200": {"description": "Inference result",
                             "content": {"application/json": {"schema": {
                                 "type": "object",
                                 "properties": {"ok": {"type": "boolean"}, "result": {}},
                             }}}},
                    "400": {"description": "Error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    "401": {"description": "Unauthorized", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                },
            }}
        if has_registry:
            paths.update({
                "/api/v1/registry": {"get": {
                    "summary": "List models (paginated)",
                    "security": [{"BearerAuth": []}, {"ApiKeyHeader": []}],
                    "parameters": [
                        {"in": "query", "name": "page", "schema": {"type": "integer", "default": 1}},
                        {"in": "query", "name": "limit", "schema": {"type": "integer", "default": 20}},
                        {"in": "query", "name": "name", "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "Model list"}, "401": {"description": "Unauthorized"}},
                }},
                "/api/v1/registry/push": {"post": {
                    "summary": "Push model from local run dir (write scope)",
                    "security": [{"BearerAuth": []}, {"ApiKeyHeader": []}],
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object", "required": ["name", "version", "run_dir"],
                        "properties": {"name": {"type": "string"}, "version": {"type": "string"}, "run_dir": {"type": "string"}},
                    }}}},
                    "responses": {"201": {"description": "Pushed"}, "401": {"description": "Unauthorized"}, "409": {"description": "Duplicate"}},
                }},
                "/api/v1/registry/{name}/tags": {"get": {
                    "summary": "List tags for a model",
                    "parameters": [{"in": "path", "name": "name", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Tags list"}},
                }},
                "/api/v1/registry/{name}/{version}": {"get": {
                    "summary": "Show model entry",
                    "parameters": [
                        {"in": "path", "name": "name", "required": True, "schema": {"type": "string"}},
                        {"in": "path", "name": "version", "required": True, "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "Entry"}, "404": {"description": "Not found"}},
                }},
                "/api/v1/registry/{name}/{version}/pull": {"get": {
                    "summary": "Pull model text and params",
                    "parameters": [
                        {"in": "path", "name": "name", "required": True, "schema": {"type": "string"}},
                        {"in": "path", "name": "version", "required": True, "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "Model text and params JSON"}},
                }},
                "/api/v1/registry/{name}/{version}/predict": {"post": {
                    "summary": "Predict from registry model",
                    "parameters": [
                        {"in": "path", "name": "name", "required": True, "schema": {"type": "string"}},
                        {"in": "path", "name": "version", "required": True, "schema": {"type": "string"}},
                    ],
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {"200": {"description": "Inference result"}},
                }},
                "/api/v1/registry/{name}/{version}/verify": {"post": {
                    "summary": "Verify integrity and signature",
                    "parameters": [
                        {"in": "path", "name": "name", "required": True, "schema": {"type": "string"}},
                        {"in": "path", "name": "version", "required": True, "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "Verified"}, "409": {"description": "Integrity mismatch"}},
                }},
                "/api/v1/registry/{name}/tag/{tag}": {"post": {
                    "summary": "Tag a version (write scope)",
                    "parameters": [
                        {"in": "path", "name": "name", "required": True, "schema": {"type": "string"}},
                        {"in": "path", "name": "tag", "required": True, "schema": {"type": "string"}},
                    ],
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object", "required": ["version"], "properties": {"version": {"type": "string"}},
                    }}}},
                    "responses": {"200": {"description": "Tagged"}},
                }},
            })

        schemas: dict = {"ErrorResponse": error_schema}
        if has_model:
            schemas.update(model_schemas)  # type: ignore[arg-type]

        return {
            "openapi": "3.0.0",
            "info": {
                "title": "MatrixAI API v1",
                "version": __version__,
                "description": "Versioned /api/v1/* surface. Error responses: {ok:false, error:str, code:str}.",
            },
            "components": {
                "securitySchemes": security_schemes,
                "schemas": schemas,
            },
            "security": [{"BearerAuth": []}, {"ApiKeyHeader": []}],
            "paths": paths,
        }

    def _generate_openapi(self) -> dict:
        schemas = self._generate_openapi_schemas()
        predict_example = self._generate_predict_example()
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "MatrixAI Prediction API",
                "version": "1.0.0",
                "description": "Auto-generated OpenAPI specification based on the deployed .mxai program."
            },
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer"
                    }
                },
                "schemas": schemas,
            },
            "security": [{"bearerAuth": []}],
            "paths": {
                "/predict": {
                    "post": {
                        "summary": "Run prediction on the model",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "oneOf": [
                                            {"$ref": "#/components/schemas/PredictionInput"},
                                            {"$ref": "#/components/schemas/PredictionBatch"},
                                        ],
                                        "description": "Single MatrixAI input payload or a batch of payloads."
                                    },
                                    **({"example": predict_example} if predict_example else {}),
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful Prediction",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "oneOf": [
                                                {"$ref": "#/components/schemas/PredictionResult"},
                                                {
                                                    "type": "array",
                                                    "items": {"$ref": "#/components/schemas/PredictionResult"},
                                                },
                                            ]
                                        }
                                    }
                                },
                            },
                            "400": {"description": "Invalid request payload"},
                            "401": {"description": "Unauthorized"},
                            "500": {"description": "Internal server error"},
                        }
                    }
                },
                "/execute-action": {
                    "post": {
                        "summary": "Execute a real action under a loaded .mxact contract",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["contract_name"],
                                        "properties": {
                                            "contract_name": {"type": "string", "description": "Name of the ACTION_CONTRACT to execute"},
                                            "input_data": {"type": "object", "description": "Input payload matching the action's INPUT declaration"},
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {"description": "Action executed successfully — returns ActionTrace fields"},
                            "400": {"description": "Validation or dry-run failure"},
                            "401": {"description": "Unauthorized"},
                            "404": {"description": "Contract not found or no contracts loaded"},
                            "422": {"description": "Action executed but reported failure"},
                            "500": {"description": "Execution error"},
                        }
                    }
                },
                "/docs": {
                    "get": {
                        "summary": "Swagger UI",
                        "security": [],
                        "responses": {"200": {"description": "HTML Interface"}}
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health Check",
                        "security": [],
                        "responses": {"200": {"description": "System OK"}}
                    }
                },
                "/metrics": {
                    "get": {
                        "summary": "Prometheus Metrics",
                        "description": "Exposes server metrics in Prometheus text format (text/plain; version=0.0.4). Includes request counters, uptime, last request latency, action counters, and P22 drift metrics when a ProductionMonitor is attached.",
                        "security": [],
                        "responses": {
                            "200": {
                                "description": "Prometheus exposition format",
                                "content": {"text/plain": {"schema": {"type": "string"}}}
                            }
                        }
                    }
                },
                "/feedback": {
                    "post": {
                        "summary": "Record ground truth for drift monitoring",
                        "description": "Feed a labeled observation to the ProductionMonitor (P22). Requires the server to be started with --continual-policy. Body: {prediction, ground_truth, trace_id?, observed_at?, parameter_set_id?}",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "required": ["prediction", "ground_truth"],
                                "properties": {
                                    "prediction":  {"type": "string"},
                                    "ground_truth": {"type": "string"},
                                    "trace_id":    {"type": "string"},
                                    "observed_at": {"type": "string"},
                                    "parameter_set_id": {"type": "string"},
                                }
                            }}}
                        },
                        "responses": {
                            "200": {"description": "Observation recorded"},
                            "404": {"description": "No continual policy loaded"}
                        }
                    }
                }
            }
        }

    def _generate_openapi_schemas(self) -> dict[str, Any]:
        program = getattr(self.server, "matrixai_program", None)
        vectors = list(getattr(program, "vectors", []) or [])
        schemas: dict[str, Any] = {}

        for vector in vectors:
            schemas[f"{vector.name}Input"] = self._vector_schema(vector)

        if len(vectors) == 1:
            vector = vectors[0]
            schemas["PredictionInput"] = {
                "oneOf": [
                    {"$ref": f"#/components/schemas/{vector.name}Input"},
                    {
                        "type": "object",
                        "required": [vector.name],
                        "properties": {
                            vector.name: {"$ref": f"#/components/schemas/{vector.name}Input"}
                        },
                        "additionalProperties": True,
                    },
                ]
            }
        elif vectors:
            schemas["PredictionInput"] = {
                "type": "object",
                "required": [vector.name for vector in vectors],
                "properties": {
                    vector.name: {"$ref": f"#/components/schemas/{vector.name}Input"}
                    for vector in vectors
                },
                "additionalProperties": True,
            }
        else:
            schemas["PredictionInput"] = {"type": "object", "additionalProperties": True}

        schemas["PredictionBatch"] = {
            "type": "array",
            "items": {"$ref": "#/components/schemas/PredictionInput"},
        }
        schemas["PredictionResult"] = {"type": "object", "additionalProperties": True}
        return schemas

    def _vector_schema(self, vector: Any) -> dict[str, Any]:
        return {
            "type": "object",
            "description": f"Input payload for MatrixAI VECTOR {vector.name}[{vector.size}]",
            "required": list(vector.fields),
            "properties": {
                field: self._type_schema(vector.field_types.get(field))
                for field in vector.fields
            },
            "additionalProperties": True,
        }

    def _type_schema(self, type_spec: Any | None) -> dict[str, Any]:
        if type_spec is None:
            return {"type": "number"}

        type_name = getattr(type_spec, "name", "Any")
        if type_name == "Integer":
            schema: dict[str, Any] = {"type": "integer"}
        elif type_name == "Boolean":
            schema = {"type": "boolean"}
        elif type_name in {"String", "Label"}:
            schema = {"type": "string"}
        elif type_name in {"Vector", "Embedding", "Tensor", "List"}:
            schema = {"type": "array"}
        elif type_name in {"Record", "Map", "ProbabilityMap", "Categorical", "Normal"}:
            schema = {"type": "object"}
        elif type_name == "Any":
            schema = {}
        else:
            schema = {"type": "number"}

        schema["x-matrixai-type"] = type_name
        parameters = getattr(type_spec, "parameters", {})
        if parameters:
            schema["x-matrixai-parameters"] = dict(parameters)

        range_spec = getattr(type_spec, "range", None)
        if range_spec is not None:
            if range_spec.minimum is not None:
                schema["minimum"] = range_spec.minimum
                if not range_spec.inclusive_min:
                    schema["exclusiveMinimum"] = True
            if range_spec.maximum is not None:
                schema["maximum"] = range_spec.maximum
                if not range_spec.inclusive_max:
                    schema["exclusiveMaximum"] = True
        return schema

    def _handle_predict(self, started_at: float | None = None):
        started_at = started_at or time.time()
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._increment_metric("requests_failed")
            self._set_metric("last_request_ms", round((time.time() - started_at) * 1000, 3))
            self._send_response(400, {"error": "Empty payload"})
            return

        body = self.rfile.read(content_length)
        result, err = self._run_inference(
            self.server.matrixai_program,
            self.server.matrixai_parameter_set,
            self.server.matrixai_backend,
            body,
        )
        elapsed = round((time.time() - started_at) * 1000, 3)
        self._set_metric("last_request_ms", elapsed)
        if err:
            self._increment_metric("requests_failed")
            self._send_response(err["status"], {"error": err["error"]})
            return
        item_count = len(result) if isinstance(result, list) else 1
        self._increment_metric("items_processed", item_count)
        self._increment_metric("requests_successful")
        self._send_response(200, result)

    def _handle_execute_action(self, started_at: float | None = None):
        started_at = started_at or time.time()
        from matrixai.actions import (
            ActionExecutor, DryRunSimulator, ExecutionContext,
            SandboxedActionExecutor, validate_action_contract,
        )
        from matrixai.actions.dryrun import RateTracker
        from matrixai.actions.schema import HIGH_RISK_CAPABILITIES
        from matrixai.actions.trace import build_action_trace

        contracts = self.server.matrixai_contracts
        if not contracts:
            self._increment_metric("requests_failed")
            self._send_response(404, {"error": "No action contracts loaded. Start server with --contract."})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._increment_metric("requests_failed")
            self._send_response(400, {"error": "Empty payload"})
            return
        try:
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except json.JSONDecodeError:
            self._increment_metric("requests_failed")
            self._send_response(400, {"error": "Invalid JSON"})
            return

        contract_name = body.get("contract_name")
        input_data = body.get("input_data", {})

        # Derive identity from the server's certified ParameterSet when available;
        # reject requests that supply mismatching values to prevent spoofed traces.
        ps = self.server.matrixai_parameter_set
        if ps is not None:
            req_mh = body.get("model_hash")
            req_ps = body.get("parameter_set_id")
            if req_mh and req_mh != ps.model_hash:
                self._increment_metric("requests_failed")
                self._send_response(400, {"error": (
                    f"model_hash mismatch: server has {ps.model_hash!r}, "
                    f"request sent {req_mh!r}"
                )})
                return
            if req_ps and req_ps != ps.parameter_set_id:
                self._increment_metric("requests_failed")
                self._send_response(400, {"error": (
                    f"parameter_set_id mismatch: server has {ps.parameter_set_id!r}, "
                    f"request sent {req_ps!r}"
                )})
                return
            model_hash = ps.model_hash
            param_set_id = ps.parameter_set_id
        else:
            model_hash = body.get("model_hash", "server")
            param_set_id = body.get("parameter_set_id", "default")

        matches = [c for c in contracts if c.name == contract_name]
        if not matches:
            self._increment_metric("requests_failed")
            names = [c.name for c in contracts]
            self._send_response(404, {"error": f"Contract {contract_name!r} not found", "available": names})
            return
        contract = matches[0]

        program = self.server.matrixai_program
        validation = validate_action_contract(contract, program)
        if not validation.ok:
            self._increment_metric("requests_failed")
            self._send_response(400, {"error": "Contract validation failed", "errors": validation.errors})
            return

        signing_key = self.server.matrixai_signing_key
        rate_tracker = self.server.matrixai_rate_trackers[contract_name]
        sim = DryRunSimulator()
        report = sim.simulate(contract, program, param_set_id, model_hash, input_data,
                              rate_tracker=rate_tracker)
        self._increment_metric("action_dry_runs_total")  # count every attempt, success or failure
        if not report.ok:
            self._increment_metric("requests_failed")
            self._send_response(400, {"error": "Dry-run failed", "errors": report.errors})
            return

        ctx = ExecutionContext(
            contract=contract,
            dry_run_report=report,
            input_data=input_data,
            model_hash=model_hash,
            parameter_set_id=param_set_id,
            allow_real_actions=self.server.matrixai_allow_real_actions,
            signing_key=signing_key,
            approval_store=self.server.matrixai_approval_store,
        )
        try:
            executor = (SandboxedActionExecutor() if contract.capability in HIGH_RISK_CAPABILITIES
                        else ActionExecutor())
            result = executor.execute(ctx)
        except Exception as exc:
            self._increment_metric("requests_failed")
            self._send_response(500, {"error": str(exc)})
            return

        trace = build_action_trace(ctx, result, signing_key=self.server.matrixai_signing_key)
        latency = round((time.time() - started_at) * 1000, 3)
        self._set_metric("last_request_ms", latency)
        self._increment_metric("action_executions_total")
        if self.server.matrixai_signing_key:
            self._increment_metric("action_signed_total")
        if result.ok:
            self._increment_metric("requests_successful")
        else:
            self._increment_metric("requests_failed")

        self._send_response(200 if result.ok else 422, {
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
        })

    def _send_response(self, code: int, payload: Any, extra_headers: dict[str, str] | None = None):
        response_body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self._write_cors_headers()
        self.end_headers()
        self.wfile.write(response_body)

    def _send_html(self, code: int, html: str):
        response_body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self._write_cors_headers()
        self.end_headers()
        self.wfile.write(response_body)

def serve_model(
    file_path: Path | None,
    params_path: Path | None,
    host: str,
    port: int,
    backend: str,
    api_key: str | None = None,
    contract_path: Path | None = None,
    allow_real_actions: bool = False,
    signing_key: str | None = None,
    rate_limit: int | None = None,
    cors_origins: list[str] | None = None,
    monitor=None,
    api_key_read: str | None = None,
    registry_path: Path | None = None,
) -> int:
    program = None
    if file_path is not None:
        program = parse_file(file_path)

    parameter_set = None
    if params_path:
        try:
            parameter_set = load_parameter_set(params_path)
            parameter_validation = validate_parameter_set(program, parameter_set)
            if not parameter_validation.ok:
                for error in parameter_validation.errors:
                    print(f"Parameter error: {error}", file=sys.stderr)
                return 1
        except (OSError, ValueError) as exc:
            print(f"Parameter error: {exc}", file=sys.stderr)
            return 2

    contracts = []
    if contract_path:
        from matrixai.actions import parse_mxact
        try:
            contracts = parse_mxact(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Contract error: {exc}", file=sys.stderr)
            return 3

    # Env var overrides: allow real actions if flag or env var says so
    if not allow_real_actions:
        allow_real_actions = os.environ.get("MATRIXAI_ALLOW_REAL_ACTIONS", "").lower() in ("1", "true", "yes")

    # Ensure an API key exists — track source to avoid logging operator-set keys
    _env_key = os.environ.get("MATRIXAI_API_KEY") or os.environ.get("API_KEY")
    if api_key:
        final_api_key, _key_source = api_key, "cli"
    elif _env_key:
        final_api_key, _key_source = _env_key, "env"
    else:
        final_api_key, _key_source = secrets.token_hex(16), "generated"

    # Rate limit: CLI arg → env var → default 60 rpm
    if rate_limit is None:
        env_rl = os.environ.get("MATRIXAI_RATE_LIMIT", "")
        try:
            rate_limit = int(env_rl) if env_rl else 60
        except ValueError:
            print(f"Warning: MATRIXAI_RATE_LIMIT={env_rl!r} is not a valid integer; using default 60 rpm.")
            rate_limit = 60

    # CORS origins: CLI arg → env var → wildcard default
    if cors_origins is None:
        env_cors = os.environ.get("MATRIXAI_CORS_ORIGINS", "")
        parsed = [o.strip() for o in env_cors.split(",") if o.strip()] if env_cors else []
        cors_origins = parsed if parsed else ["*"]

    # Load registry if path provided (PR5-C6)
    registry = None
    if registry_path is not None:
        from matrixai.registry.model_registry import ModelRegistry
        registry = ModelRegistry(registry_path)

    # Read-only API key: CLI arg → env var → None
    if api_key_read is None:
        api_key_read = os.environ.get("MATRIXAI_API_KEY_READ") or None

    server = MatrixAIHTTPServer(
        (host, port), MatrixAIServerHandler,
        program, parameter_set, backend, final_api_key,
        contracts=contracts,
        allow_real_actions=allow_real_actions,
        signing_key=signing_key,
        rate_limiter=RateLimiter(rate_limit),
        cors_origins=cors_origins,
        monitor=monitor,
        api_key_read=api_key_read,
        registry=registry,
    )
    
    print(f"MatrixAI server running at http://{host}:{port}/")
    print(f"Open docs in your browser: http://{host}:{port}/docs")
    print(f"--------------------------------------------------")
    if _key_source == "generated":
        print(f"🔒 API KEY (auto-generated — set MATRIXAI_API_KEY to persist): {final_api_key}")
        print(f"   Authorization: Bearer {final_api_key}")
    else:
        _src = "CLI --api-key" if _key_source == "cli" else "MATRIXAI_API_KEY env var"
        print(f"🔒 API KEY: active via {_src} (value not logged)")
    print(f"--------------------------------------------------")
    print(f"Backend: {backend}")
    print(f"Endpoints:")
    print(f"  GET  /api/v1/health")
    print(f"  GET  /api/v1/metrics      (Prometheus text format)")
    print(f"  GET  /docs                (Swagger UI)")
    print(f"  GET  /openapi.json        (OpenAPI Schema)")
    if program is not None:
        print(f"  POST /api/v1/predict      (Requires Bearer Token)")
    if contracts:
        print(f"  POST /api/v1/execute-action  (Real actions — requires --allow-real-actions)")
    if monitor is not None:
        print(f"  POST /api/v1/feedback        (Record ground truth for drift monitoring)")
    if registry is not None:
        print(f"  GET  /api/v1/registry         (List models; ?page=1&limit=20)")
        print(f"  GET  /api/v1/registry/{{name}}/{{version}}")
        print(f"  POST /api/v1/registry/{{name}}/{{version}}/predict")
        print(f"  POST /api/v1/registry/{{name}}/{{version}}/verify")
        print(f"  POST /api/v1/registry/push    (write scope)")
        print(f"  POST /api/v1/registry/{{name}}/tag/{{tag}}  (write scope)")
    if api_key_read:
        print(f"  Read-only key active (MATRIXAI_API_KEY_READ)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()
    return 0
