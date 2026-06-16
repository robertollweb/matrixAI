# PR4-C1 — Production Deployment Guide

> **Español:** [docs/es/deployment/DEPLOYMENT.md](../../es/deployment/DEPLOYMENT.md)

This guide walks an operator through deploying a MatrixAI model on a production server using Docker and Docker Compose. The server runs on a clean machine with no prior MatrixAI installation; everything needed is bundled by `matrixai pack`.

**Prerequisites:** Docker ≥ 24, Docker Compose plugin (≥ v2), no other requirements.

---

## Step 1 — Pack the model

On the development machine (where MatrixAI is installed):

```bash
matrixai pack examples/credit-scoring/credit_scoring.mxai \
  --params examples/credit-scoring/registry/entries/credit-scoring/v1.0/params.json \
  --docker \
  --outdir dist/credit-scoring
```

To also bundle an action contract for real-action execution:

```bash
matrixai pack examples/agent-alert/alert_monitor.mxai \
  --params examples/agent-alert/registry/entries/alert-monitor/v1.0/params.json \
  --contract examples/agent-alert/alert_notifier.mxact \
  --docker \
  --outdir dist/agent-alert
```

This produces a self-contained directory:

```
dist/credit-scoring/
  credit_scoring.mxai        # model artifact
  params.json                # trained parameters
  matrixai/                  # framework source (no pip install needed)
  Dockerfile                 # production image
  docker-compose.yml         # compose stack
  .env.example               # documented environment variables
```

Transfer the directory to the production server (`scp -r`, `rsync`, or a container registry after `docker build`).

---

## Step 2 — Configure environment

```bash
cd dist/credit-scoring
cp .env.example .env
$EDITOR .env
```

Minimum required change — set a strong API key:

```
MATRIXAI_API_KEY=<strong-random-string>
```

Generate a secure value:

```bash
openssl rand -hex 32
```

**All environment variables:**

| Variable | Required | Description |
|---|---|---|
| `MATRIXAI_API_KEY` | **Yes** | Bearer token for `POST /predict` and `POST /execute-action` |
| `MATRIXAI_ACTION_SIGNING_KEY` | When real actions | HMAC-SHA256 key for `ActionTrace` signatures (min 32 bytes hex) |
| `MATRIXAI_ALLOW_REAL_ACTIONS` | No | Set to `true` to enable `POST /execute-action` with real side effects. Default: `false` |
| `MATRIXAI_REGISTRY_SIGNING_KEY` | Recommended | HMAC key for registry entry signatures. If unset, a per-container key is generated (not persistent across restarts) |
| `MATRIXAI_RATE_LIMIT` | No | Max requests/minute per IP. Default: `60`. Set to `0` to disable. |
| `MATRIXAI_CORS_ORIGINS` | No | Allowed CORS origins (comma-separated). Default: `*`. Restrict for browser clients. |
| `PORT` | No | Host port to expose. Default: `8000` |

**Security notes:**
- Never commit `.env` to version control. Add it to `.gitignore`.
- `MATRIXAI_ALLOW_REAL_ACTIONS=false` (the default) means all actions are simulated — safe for staging and CI.
- The signing keys must be at least 32 bytes hex (64 hex characters). Generate with `openssl rand -hex 32`.

---

## Step 3 — Build and start

```bash
docker compose up --build -d
```

Expected output:

```
[+] Building ...
[+] Running 1/1
 ✔ Container credit-scoring-matrixai-server-1  Started
```

Check the server is healthy:

```bash
docker compose ps
# Status should show: healthy

curl http://localhost:8000/health
# {"status": "ok", "service": "MatrixAI Server", ...}
```

---

## Step 4 — Send a prediction

```bash
curl -s \
  -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"income_score": 0.7, "credit_history": 0.8, "debt_ratio": 0.2, "employment_years": 0.6, "loan_amount_ratio": 0.3}' \
  http://localhost:8000/predict
```

Browse the auto-generated API docs:

```
http://localhost:8000/docs
```

---

## Step 5 — View logs and metrics

```bash
# Tail server logs
docker compose logs -f matrixai-server

# Check runtime metrics
curl http://localhost:8000/health | python3 -m json.tool
```

Metrics exposed at `/health`:

| Field | Description |
|---|---|
| `requests_total` | Total requests received |
| `requests_successful` | Requests that returned 200 |
| `requests_failed` | Requests that returned 4xx/5xx |
| `requests_rate_limited` | Requests rejected by rate limiting (429) |
| `items_processed` | Individual predictions (batches counted per item) |
| `last_request_ms` | Wall time of the last request |
| `uptime_seconds` | Server uptime |

---

## Step 6 — Stop and clean up

```bash
# Stop without removing data
docker compose stop

# Stop and remove containers (keeps the registry volume)
docker compose down

# Remove everything including the registry volume
docker compose down -v
```

---

## Enabling real actions (P20)

To serve a model with a `.mxact` contract and allow real side effects:

1. Copy the `.mxact` file into the dist directory.
2. Set in `.env`:
   ```
   MATRIXAI_ALLOW_REAL_ACTIONS=true
   MATRIXAI_ACTION_SIGNING_KEY=<openssl rand -hex 32>
   ```
3. Pass the contract path at pack time (or mount it at runtime and regenerate the image).
4. `docker compose up --build -d`

Without `MATRIXAI_ALLOW_REAL_ACTIONS=true`, `POST /execute-action` simulates all actions — no real side effects occur regardless of the `.mxact` contents.

---

## Conditions

- Docker: ≥ 24
- Docker Compose: plugin ≥ v2 (`docker compose` not `docker-compose`)
- Base image: `python:3.11-slim`
- MatrixAI backend: stdlib (no torch required)
- Registry volume: named volume `matrixai-registry` mounted at `/data/registry` inside the container

Run `matrixai pack --docker` on your own hardware to reproduce the build; image digest will differ but behavior is identical given the same model and parameters.

---

## Related guides

| Guide | Contents |
|---|---|
| [Server Hardening](SERVER_HARDENING.md) | Rate limiting, CORS, X-API-Key auth, JWT extension point, public deployment checklist |
| [Key Rotation](KEY_ROTATION.md) | Generate, rotate, and manage signing keys without invalidating historical traces |
