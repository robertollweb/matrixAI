# MatrixAI REST API Reference

MatrixAI exposes two independent HTTP servers, each with a distinct role:

- **Production Server** — serves a trained, registered model. Requires API key authentication. Intended for integration with external systems, the Studio app, and production deployments.
- **Studio Server** — the model development environment. Handles model generation from prompts, training, validation and refinement. No authentication required; not intended to be exposed publicly.

Both servers are started via the CLI (`matrixai serve`, `matrixai playground`) and can be run independently.

---

## Authentication

The Production Server requires an API key for prediction and action endpoints.

Two equivalent formats are accepted:

```
Authorization: Bearer <api_key>
```
```
X-API-Key: <api_key>
```

The write key is set at startup with `--api-key` or `MATRIXAI_API_KEY`.

**Read-only scope (PR5-C6):** a second key can be provided with `--api-key-read` or `MATRIXAI_API_KEY_READ`. It grants access to read endpoints (`GET /api/v1/registry`, `/api/v1/predict`, `/api/v1/registry/{name}/{version}/verify`, `/api/v1/registry/{name}/{version}/pull`) but is rejected on write endpoints (push, tag, execute-action, feedback).

Endpoints marked **public** do not require authentication.

---

## Rate Limiting

The Production Server enforces a per-IP sliding window rate limit (default: 60 requests/minute). Configurable with `--rate-limit` at startup or `MATRIXAI_RATE_LIMIT` env var. Exceeded requests return `429 Too Many Requests` with a `Retry-After: 60` header.

---

## Common Response Format

Versioned `/api/v1/*` endpoints use a consistent envelope:

```json
{ "ok": true, ...payload }
{ "ok": false, "error": "description", "code": "ERROR_CODE" }
```

The `code` field is a machine-readable string (e.g. `NOT_FOUND`, `UNAUTHORIZED`, `REGISTRY_NOT_LOADED`) for programmatic error handling.

Legacy unversioned routes (`/predict`, `/health`, etc.) preserve their existing response shape for backwards compatibility and do not include `code`.

HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Malformed request (invalid JSON) |
| 401 | Missing or invalid API key |
| 404 | Resource not found |
| 422 | Validation or processing failure |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Production Server

Default port: `8000`. Started with `matrixai serve <model.mxai> --params <params.json>`.

### GET /health

**Auth:** Public

Returns server status and basic metrics.

```json
{
  "status": "ok",
  "service": "MatrixAI Server",
  "backend": "numpy",
  "metrics": {
    "requests_total": 142,
    "requests_successful": 140,
    "requests_failed": 2,
    "uptime_seconds": 3600
  }
}
```

---

### GET /metrics

**Auth:** Public

Returns server metrics in Prometheus exposition format (`text/plain; version=0.0.4`). Suitable for scraping with Prometheus.

Core metrics (always present):

| Metric | Type | Description |
|--------|------|-------------|
| `matrixai_requests_total` | Counter | Total requests received |
| `matrixai_requests_successful` | Counter | Requests that returned 2xx |
| `matrixai_requests_failed` | Counter | Requests that returned 4xx/5xx |
| `matrixai_requests_rate_limited` | Counter | Requests rejected by rate limiter |
| `matrixai_items_processed` | Counter | Items processed (counts batch items individually) |
| `matrixai_last_request_duration_milliseconds` | Gauge | Latency of the last request |
| `matrixai_uptime_seconds` | Gauge | Server uptime in seconds |

Drift metrics (present only when continual monitoring is active):

| Metric | Type | Description |
|--------|------|-------------|
| `matrixai_drift_window_accuracy` | Gauge | Accuracy in the current monitoring window |
| `matrixai_drift_window_samples` | Gauge | Number of feedback samples in the current window |
| `matrixai_drift_degradation_detected` | Gauge | `1.0` if drift threshold exceeded, `0.0` otherwise |
| `matrixai_drift_actual_degradation` | Gauge | Magnitude of accuracy drop from the reference baseline |

All metrics include a `{project="<model_name>"}` label.

---

### GET /docs

**Auth:** Public

Returns the Swagger UI interface for interactive API exploration.

---

### GET /openapi.json

**Auth:** Public

Returns the OpenAPI 3.0.0 specification generated from the loaded `.mxai` program. Includes the input schema derived from the model's VECTOR definition.

---

### POST /predict

**Auth:** Required

Run inference on the loaded model. Accepts a single input object or a batch (array of objects). Input fields must match the model's VECTOR schema.

**Request — single:**
```json
{ "age": 35, "income": 52000, "credit_history": "good" }
```

**Request — batch:**
```json
[
  { "age": 35, "income": 52000, "credit_history": "good" },
  { "age": 28, "income": 31000, "credit_history": "fair" }
]
```

**Response:**
```json
{
  "ok": true,
  "result": 0.9998245440617306,
  "model": "CreditScoring",
  "parameter_set": "v1.0_best"
}
```

Batch response returns an array of results in the same order as the input.

---

### POST /execute-action

**Auth:** Required

Execute a real action under a loaded `.mxact` contract. The execution is signed, audited and recorded.

**Request:**
```json
{
  "contract_name": "CreditDecision",
  "input_data": { "age": 35, "income": 52000, "credit_history": "good" },
  "model_hash": "mxai_20d8ce3f...",
  "parameter_set_id": "v1.0_best"
}
```

**Response:**
```json
{
  "ok": true,
  "report_id": "rpt-20260528-abc123",
  "model_hash": "mxai_20d8ce3f...",
  "parameter_set_id": "v1.0_best",
  "action_contract_hash": "sha256:...",
  "input_hash": "sha256:...",
  "executed_at": "2026-05-28T14:00:00",
  "executor_kind": "real",
  "ok_action": true,
  "response_summary": "Credit approved",
  "error": null,
  "latency_ms": 3.2,
  "hmac_signature": "..."
}
```

Returns `422` if contract validation fails or the action is not permitted.

---

### POST /feedback

**Auth:** Required

Record ground truth for a previous prediction. Used by the continual learning monitor (P22) to track accuracy drift over time.

**Request:**
```json
{
  "prediction": "1",
  "ground_truth": "0",
  "trace_id": "trace-abc123",
  "observed_at": "2026-05-28T15:00:00",
  "parameter_set_id": "v1.0_best"
}
```

**Response:**
```json
{
  "ok": true,
  "recorded": true,
  "correct": false,
  "trace_id": "trace-abc123"
}
```

Returns `404` if no continual monitor is loaded on the server.

---

### OPTIONS *

**Auth:** Public

CORS preflight handler. Returns `204 No Content` with `Access-Control-Allow-*` headers. Controlled by `--cors-origin` at startup or `MATRIXAI_CORS_ORIGINS` env var.

---

## Versioned API — /api/v1/

All Production Server routes are also available under `/api/v1/` with a consistent error schema. The legacy unversioned routes are preserved as backwards-compatible aliases.

Enable the registry layer by starting the server with `--registry PATH`.

### GET /api/v1/health · GET /api/v1/metrics

**Auth:** Public. Same behaviour as unversioned equivalents; response wrapped in `{ok: true, ...}`.

---

### POST /api/v1/predict

**Auth:** Required (write or read key). Returns `{"ok": true, "result": {...}}` on success; `{"ok": false, "error": "...", "code": "..."}` on error.

### POST /api/v1/execute-action

**Auth:** Required (write key only). Returns the ActionTrace payload (includes `"ok"` representing action success, not HTTP envelope).

### POST /api/v1/feedback

**Auth:** Required (**write key only** — feedback mutates the drift monitor). Returns `{"ok": true, ...}` on success.

---

### GET /api/v1/registry

**Auth:** Required (read or write key). Requires `--registry PATH` at startup.

List all models. Supports pagination.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `page` | `1` | Page number (1-based) |
| `limit` | `20` | Items per page (max 100) |
| `name` | — | Filter by model name |

**Response:**
```json
{
  "ok": true,
  "models": [
    {
      "name": "credit-scoring",
      "version": "v1.0",
      "matrixai_version": "1.0.0",
      "metrics": { "accuracy": 0.933 },
      "created_at": "2026-05-30T10:00:00+00:00"
    }
  ],
  "page": 1,
  "limit": 20,
  "total": 1
}
```

---

### GET /api/v1/registry/{name}/{version}

**Auth:** Required (read or write key).

**Response:**
```json
{
  "ok": true,
  "model": {
    "name": "credit-scoring",
    "version": "v1.0",
    "matrixai_version": "1.0.0",
    "entry_hash": "sha256:...",
    "model_hash": "sha256:...",
    "parameter_set_id": "ps_v1",
    "metrics": { "accuracy": 0.933 },
    "created_at": "2026-05-30T10:00:00+00:00",
    "interpretability_level": "full"
  }
}
```

Returns `404 NOT_FOUND` if the entry does not exist.

---

### POST /api/v1/registry/{name}/{version}/predict

**Auth:** Required (read or write key).

Run inference on a model loaded from the registry. Same request format as `POST /predict`.

**Response:** `{ "ok": true, "result": { ... } }`

---

### POST /api/v1/registry/{name}/{version}/verify

**Auth:** Required (read or write key).

Verify integrity and signature of a registry entry. Warns if major version differs from running version.

**Response:** `{ "ok": true, "verified": true, "warnings": [] }`

Returns `409 INTEGRITY_MISMATCH` on failure.

---

### GET /api/v1/registry/{name}/tags

**Auth:** Required (read or write key).

List all tags for a model.

**Response:** `{ "ok": true, "name": "credit-scoring", "tags": [{"tag": "latest", "version": "v1.1"}] }`

---

### GET /api/v1/registry/{name}/{version}/pull

**Auth:** Required (read or write key).

Return the model program text and parameters JSON. Allows a client to retrieve the model without filesystem access.

**Response:**
```json
{
  "ok": true,
  "name": "credit-scoring",
  "version": "v1.0",
  "model_text": "MATRIX CreditScoring ...",
  "params": { "parameter_set_id": "...", "parameters": { ... } }
}
```

---

### POST /api/v1/registry/push

**Auth:** Required (**write key only**).

Register a model from a run directory on the server filesystem.

**Request:**
```json
{ "name": "credit-scoring", "version": "v1.1", "run_dir": "/path/to/runs/v1" }
```

**Response:** `{ "ok": true, "name": "credit-scoring", "version": "v1.1" }` — `201 Created`

Returns `409 DUPLICATE_ENTRY` if the version already exists (registry is append-only).

---

### POST /api/v1/registry/{name}/tag/{tag}

**Auth:** Required (**write key only**).

**Request:** `{ "version": "v1.1" }`

**Response:** `{ "ok": true, "name": "credit-scoring", "tag": "latest", "version": "v1.1" }`

---

## Technical Playground Server

Default port: `8080`. Started with `matrixai playground`. All endpoints are public — do not expose this server externally.

### GET /

**Returns the technical playground UI.** Browser-based model development environment.

---

### GET /expert

Returns the Expert Workbench UI for advanced `.mxai` editing.

---

### GET /api/defaults

Returns the default prompt and input examples shown on first load.

```json
{
  "ok": true,
  "prompt": "...",
  "input_json": "{ ... }",
  "examples": [
    { "id": "credit-scoring", "label": "Credit Scoring" },
    { "id": "fall-risk", "label": "Fall Risk Assessment" }
  ]
}
```

---

### GET /api/example/{example_id}

Load a full example package by ID. Returns all artifacts for the example: `.mxai` text, training contract, sample input, manifest, evaluation report.

```json
{
  "ok": true,
  "id": "credit-scoring",
  "label": "Credit Scoring",
  "mode": "prompt",
  "mxai_text": "...",
  "training_text": "...",
  "input_json": "...",
  "manifest_text": "...",
  "evaluation_report_text": "..."
}
```

Returns `404` if the example ID is unknown.

---

### POST /api/analyze

Parse and validate a `.mxai` program, or run the full pipeline `prompt → .semantic → .mxai`.

**Request:**
```json
{
  "mode": "prompt",
  "prompt": "A model that predicts credit approval based on age, income and credit history",
  "input_json": "{ \"age\": 35 }",
  "use_llm": false
}
```

Modes: `"prompt"` | `"semantic"` | `"mxai"`.

**Response:**
```json
{
  "ok": true,
  "mode": "prompt",
  "accepted": true,
  "mxai": "...",
  "semantic_text": "...",
  "checks": [...],
  "pipeline_stages": [...],
  "artifacts": { ... }
}
```

Returns `422` if validation fails, with `checks` listing the specific errors.

---

### POST /api/generate-training

Generate a `.mxtrain` contract and a dataset template from a `.mxai` program.

**Request:**
```json
{ "mxai_text": "PROGRAM CreditScoring ..." }
```

**Response:**
```json
{
  "ok": true,
  "training_text": "TRAINING CreditScoring ...",
  "dataset_template_text": "age,income,credit_history,label\n...",
  "warnings": [],
  "source": "generated"
}
```

---

### POST /api/generate-dataset

Generate a synthetic training dataset from a `.mxai` + `.mxtrain` spec. Maximum 5000 rows / 1 MB.

**Request:**
```json
{
  "mxai_text": "...",
  "training_text": "...",
  "rows": 500,
  "seed": 42,
  "mode": "coherent"
}
```

Modes: `"random"` | `"coherent"` (coherent generates data consistent with the model's semantics).

**Response:**
```json
{
  "ok": true,
  "csv_text": "age,income,...\n...",
  "rows": 500,
  "seed": 42,
  "mode": "coherent",
  "fingerprint": "sha256:...",
  "columns": ["age", "income", "credit_history"],
  "labels": ["approved", "rejected"]
}
```

---

### POST /api/validate-csv

Validate that a CSV file matches the training spec. Maximum 5000 rows / 1 MB.

**Request:**
```json
{
  "mxai_text": "...",
  "training_text": "...",
  "csv_text": "age,income,...\n..."
}
```

**Response:**
```json
{
  "ok": true,
  "rows": 450,
  "warnings": ["Column 'income' has 3 missing values"]
}
```

---

### POST /api/train

Synchronous training. Times out after 30 seconds, maximum 200 epochs. For longer training use `/api/train-start`.

**Request:**
```json
{
  "mxai_text": "...",
  "training_text": "...",
  "csv_text": "...",
  "epochs_override": 100
}
```

**Response:**
```json
{
  "ok": true,
  "task_kind": "classification",
  "run_id": "run-abc123",
  "best_epoch": 87,
  "best_validation_loss": 0.112,
  "final_train_loss": 0.098,
  "accuracy": 0.923,
  "mae": null,
  "rmse": null,
  "r2": null,
  "backend": "numpy",
  "epochs": [...],
  "params_best": { ... }
}
```

---

### POST /api/train-start

Start an async training job. Returns immediately with a `job_id` to poll.

**Request:** Same as `/api/train`.

**Response:**
```json
{ "ok": true, "job_id": "job-20260528-xyz789" }
```

---

### GET /api/train-status/{job_id}

Poll the status of an async training job.

```json
{
  "ok": true,
  "job_id": "job-20260528-xyz789",
  "status": "done",
  "epochs": [...],
  "accuracy": 0.923,
  "best_epoch": 87
}
```

`status` values: `"running"` | `"done"` | `"error"` | `"cancelled"` | `"timeout"`.

Returns `404` if the job ID is unknown.

---

### POST /api/train-cancel

Cancel a running async training job.

**Request:**
```json
{ "job_id": "job-20260528-xyz789" }
```

**Response:**
```json
{ "ok": true, "job_id": "job-20260528-xyz789", "status": "cancelled" }
```

---

### POST /api/run-with-params

Run the model with a specific set of loaded parameters without starting a server.

**Request:**
```json
{
  "mxai_text": "...",
  "params_json": "{ \"weights\": [...] }",
  "input_json": "{ \"age\": 35, \"income\": 52000 }"
}
```

**Response:**
```json
{ "ok": true, "result": { "approved": 0.9998 } }
```

---

### POST /api/refine

Refine a prompt based on an audit result. Implements the iterative refinement loop (P13).

**Request:**
```json
{
  "prompt": "A model that predicts credit approval...",
  "run_result": { ... },
  "mxai_text": "...",
  "hints": "Focus on improving recall for low-income applicants",
  "iteration_count": 1,
  "refinement_chain": [...],
  "parent_prompt_hash": "sha256:...",
  "max_iterations": 5
}
```

**Response:**
```json
{
  "ok": true,
  "refinement_id": "rfn-abc123",
  "mode": "supervised",
  "iteration": 2,
  "supervision_accepted": true,
  "chain": [...],
  "parent_hash": "sha256:...",
  "proposed_prompt": "A model that predicts credit approval...",
  "explanation": "Adjusted feature weighting for income sensitivity"
}
```

---

## Endpoint Summary

### Production Server (port 8000)

The paths below are the currently implemented unversioned aliases. PR5-C6 will
publish the versioned `/api/v1/` surface and keep these aliases for backwards
compatibility.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | Public | Server health and metrics |
| GET | `/metrics` | Public | Prometheus metrics |
| GET | `/docs` | Public | Swagger UI |
| GET | `/openapi.json` | Public | OpenAPI 3.0.0 spec |
| POST | `/predict` | Required | Run model inference |
| POST | `/execute-action` | Required | Execute audited real action |
| POST | `/feedback` | Required | Record ground truth for drift monitoring |
| OPTIONS | `*` | Public | CORS preflight |

### Technical Playground Server (port 8080)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | Public | Studio UI |
| GET | `/expert` | Public | Expert Workbench UI |
| GET | `/api/defaults` | Public | Default prompt and examples |
| GET | `/api/example/{id}` | Public | Load example package |
| GET | `/api/train-status/{job_id}` | Public | Poll async training job |
| POST | `/api/analyze` | Public | Parse/validate `.mxai` or run prompt pipeline |
| POST | `/api/generate-training` | Public | Generate training contract |
| POST | `/api/generate-dataset` | Public | Generate synthetic dataset |
| POST | `/api/validate-csv` | Public | Validate CSV against spec |
| POST | `/api/train` | Public | Synchronous training |
| POST | `/api/train-start` | Public | Start async training job |
| POST | `/api/train-cancel` | Public | Cancel async training job |
| POST | `/api/run-with-params` | Public | Run model with parameters |
| POST | `/api/refine` | Public | Iterative prompt refinement |

