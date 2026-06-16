# PR4-C2 — Observability Guide

> **Español:** [docs/es/deployment/OBSERVABILITY.md](../../es/deployment/OBSERVABILITY.md)

MatrixAI exposes a `GET /metrics` endpoint in [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) so any standard scraper (Prometheus, Grafana Agent, OpenTelemetry Collector) can pull server health and drift metrics without modifying the application.

---

## Endpoint

```
GET /metrics
```

- **Authentication:** not required (same as `/health`)
- **Content-Type:** `text/plain; version=0.0.4; charset=utf-8`
- **Always available** regardless of whether a model is loaded or actions are enabled

Quick smoke-test:

```bash
curl http://localhost:8000/metrics
```

Expected output (truncated):

```
# HELP matrixai_requests_total Total HTTP requests received.
# TYPE matrixai_requests_total counter
matrixai_requests_total{project="CreditScoring"} 42
# HELP matrixai_uptime_seconds Server uptime in seconds.
# TYPE matrixai_uptime_seconds gauge
matrixai_uptime_seconds{project="CreditScoring"} 317.5
...
```

---

## Metrics reference

### Server health (always emitted)

| Metric | Type | Description |
|---|---|---|
| `matrixai_requests_total` | counter | Total HTTP requests across all endpoints |
| `matrixai_requests_successful_total` | counter | Requests that returned 200 |
| `matrixai_requests_failed_total` | counter | Requests that returned 4xx/5xx |
| `matrixai_requests_rate_limited_total` | counter | Requests rejected by rate limiting (429) |
| `matrixai_items_processed_total` | counter | Individual predictions (batch items counted separately) |
| `matrixai_last_request_duration_milliseconds` | gauge | Wall time of the most recent request |
| `matrixai_uptime_seconds` | gauge | Server uptime |

All metrics carry a `project` label matching the `PROJECT` name in the `.mxai` file.

### P20 action metrics (always emitted)

| Metric | Type | Description |
|---|---|---|
| `matrixai_action_executions_total` | counter | Total `POST /execute-action` calls processed |
| `matrixai_action_dry_runs_total` | counter | Dry-run simulations executed (one per action call) |
| `matrixai_action_signed_total` | counter | ActionTraces signed with `MATRIXAI_ACTION_SIGNING_KEY` |

### P22 drift metrics (emitted when a ProductionMonitor is attached)

| Metric | Type | Description |
|---|---|---|
| `matrixai_drift_window_accuracy` | gauge | Sliding-window prediction accuracy (0–1) |
| `matrixai_drift_window_samples` | gauge | Labeled observations in the current window |
| `matrixai_drift_degradation_detected` | gauge | 1 if accuracy degradation detected, else 0 |
| `matrixai_drift_actual_degradation` | gauge | Accuracy drop below reference (positive = worse) |

Drift metrics require a `ProductionMonitor` to be wired to the server (see [Wiring drift metrics](#wiring-drift-metrics) below).

---

## Prometheus integration

### 1. `prometheus.yml` scrape config

```yaml
scrape_configs:
  - job_name: matrixai
    scrape_interval: 15s
    static_configs:
      - targets: ["matrixai-server:8000"]
    metrics_path: /metrics
```

### 2. Docker Compose with Prometheus

Add a Prometheus service to the generated `docker-compose.yml`:

```yaml
services:
  matrixai-server:
    build: .
    ports:
      - "${PORT:-8000}:8000"
    env_file: .env
    healthcheck:
      test: ["CMD", "python3", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
    restart: unless-stopped

volumes:
  matrixai-registry:
```

Save the scrape config from step 1 as `prometheus.yml` alongside the `docker-compose.yml`, then:

```bash
docker compose up --build -d
# Prometheus UI → http://localhost:9090
# Query: matrixai_requests_total
```

### 3. Grafana dashboard (quick start)

```bash
# Add Prometheus as a data source in Grafana, then query:
rate(matrixai_requests_total[1m])          # request rate
matrixai_uptime_seconds                    # uptime
matrixai_drift_degradation_detected == 1  # drift alert
```

---

## Wiring drift metrics

Drift metrics are emitted only when a `ProductionMonitor` (P22) is attached to the server. Two steps are needed:

### Step 1 — Start the server with a continual policy

```bash
matrixai serve model.mxai \
  --params params.json \
  --continual-policy policy/alert_monitor.mxcontinual
# Continual monitoring active (policy: AlertMonitorContinual)
# POST /feedback  (Record ground truth for drift monitoring)
```

### Step 2 — Send ground-truth feedback

After each prediction, send the actual label via `POST /feedback` (requires auth):

```bash
curl -s -X POST \
  -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prediction": "high_risk", "ground_truth": "low_risk", "trace_id": "abc123"}' \
  http://localhost:8000/feedback
# {"recorded": true, "correct": false, "trace_id": "abc123"}
```

Once enough observations accumulate (`MIN_SAMPLES_IN_WINDOW` in the policy), `/metrics` will include the drift gauges.

When no monitor is attached (the default), the drift metric lines are simply omitted from the output. Prometheus treats absent metrics as stale/unknown, which is the correct behavior.

---

## Security notes

- `/metrics` is unauthenticated by design — Prometheus scrapers typically run inside the same network segment as the server.
- In public-facing deployments, restrict access via a reverse proxy (nginx `location` block, firewall rule) rather than adding auth to the scrape endpoint.
- Metric labels include the model's `project` name but no user data, prediction values, or parameter values.

---

## Related guides

| Guide | Contents |
|---|---|
| [Deployment](DEPLOYMENT.md) | Pack and deploy a model with Docker |
| [Server Hardening](SERVER_HARDENING.md) | Rate limiting, CORS, API key auth |
| [Key Rotation](KEY_ROTATION.md) | Rotate signing keys without invalidating audit trails |
