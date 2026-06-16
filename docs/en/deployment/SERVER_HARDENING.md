# PR4-C5 — Server Hardening

> **Español:** [docs/es/deployment/SERVER_HARDENING.md](../../es/deployment/SERVER_HARDENING.md)

MatrixAI's HTTP server ships with secure defaults and configurable hardening controls:
rate limiting per IP, API key authentication with a JWT-ready extension point, and
configurable CORS for cross-origin clients.

---

## Authentication

The server requires a secret API key on all write endpoints (`/predict`, `/execute-action`).

### Providing the API key

Set `MATRIXAI_API_KEY` in `.env` (see [DEPLOYMENT.md](DEPLOYMENT.md)):

```
MATRIXAI_API_KEY=<openssl rand -hex 32>
```

If the env var is not set and `--api-key` is not passed, the server generates a random key at
startup and prints it to stdout. This is safe for development but **not for production** — the
key changes every restart, invalidating existing client configuration.

### Sending requests

Two equivalent header forms are accepted:

```bash
# Bearer token (standard OAuth2 form)
curl -H "Authorization: Bearer $MATRIXAI_API_KEY" http://host:8000/predict -d '...'

# X-API-Key (simpler for non-OAuth clients)
curl -H "X-API-Key: $MATRIXAI_API_KEY" http://host:8000/predict -d '...'
```

### JWT extension point (for PR5-C6)

The auth check in `matrixai/server.py::MatrixAIServerHandler._check_auth` is the single
point where auth decisions are made. When PR5-C6 introduces the external Studio, replace the
bearer-token comparison with a JWT validation call:

```python
# Future extension — do NOT add now; document design intent
# if _validate_jwt(auth_header_value, public_key):
#     return True
```

The API surface (`Authorization: Bearer <token>`) is already JWT-compatible — no client changes
needed when switching from static key to JWT.

### Public endpoints (no auth required)

| Endpoint | Reason |
|---|---|
| `GET /health` | Liveness probe — must work without credentials for load balancers and Docker healthchecks |
| `GET /docs` | Swagger UI — informational only, no data access |
| `GET /openapi.json` | Schema — informational only |
| `OPTIONS *` | CORS preflight — browsers send without auth by spec |

---

## Rate limiting

Per-IP sliding-window rate limiting prevents abuse and protects the inference process.

### Default

60 requests/minute per IP address, applied to `POST /predict` and `POST /execute-action`.

### Configuration

**Environment variable:**
```
MATRIXAI_RATE_LIMIT=60
```

**CLI flag:**
```bash
matrixai serve model.mxai --rate-limit 30
```

Set to `0` to disable rate limiting entirely (not recommended for public endpoints):
```bash
matrixai serve model.mxai --rate-limit 0
```

### Response when rate-limited

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

The `Retry-After: 60` header tells compliant clients to wait 60 seconds before retrying.

### Per-endpoint tuning

Override limits per endpoint using environment variables:
```
MATRIXAI_RATE_LIMIT=60          # Default for all endpoints
```

The sliding window resets naturally — a client that sent 60 requests and then waits 60 seconds
can send 60 more. There is no explicit reset mechanism.

---

## CORS (Cross-Origin Resource Sharing)

Required when a browser-based client (such as the future MatrixAI Studio) calls the API from
a different origin.

### Default

```
Access-Control-Allow-Origin: *
```

The wildcard allows any origin. This is safe when authentication is enforced — unauthenticated
requests are rejected regardless of origin. However, it disables certain browser security
features (e.g., cookies with `SameSite=None` cannot be sent to wildcard CORS endpoints).

### Restricting to specific origins

**Environment variable (comma-separated):**
```
MATRIXAI_CORS_ORIGINS=https://studio.example.com,https://app.example.com
```

**CLI flag (repeatable):**
```bash
matrixai serve model.mxai \
  --cors-origin https://studio.example.com \
  --cors-origin https://app.example.com
```

When specific origins are listed, the server echoes the request's `Origin` header if it matches,
or returns the first listed origin otherwise. Non-matching origins receive a CORS response that
browsers will block.

### CORS preflight

The server handles `OPTIONS` requests automatically. Browsers send a preflight `OPTIONS` before
any cross-origin `POST` or `PUT`. The server returns:

```
HTTP/1.1 204 No Content
Access-Control-Allow-Origin: <origin>
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, X-API-Key
Access-Control-Max-Age: 86400
```

The `Access-Control-Max-Age: 86400` (24 hours) caches the preflight result in compliant browsers,
reducing preflight overhead for repeat calls.

---

## Secure defaults summary

| Feature | Default | Override |
|---|---|---|
| Auth | Required on `/predict`, `/execute-action` | Cannot disable |
| API key source | `MATRIXAI_API_KEY` env var → auto-generated | `--api-key` flag |
| Real actions | Disabled | `--allow-real-actions` or `MATRIXAI_ALLOW_REAL_ACTIONS=1` |
| Rate limit | 60 req/min per IP | `--rate-limit N` or `MATRIXAI_RATE_LIMIT=N` |
| CORS | `*` (all origins) | `--cors-origin ORIGIN` or `MATRIXAI_CORS_ORIGINS=...` |

---

## Checklist for public deployment

- [ ] `MATRIXAI_API_KEY` set explicitly in `.env` (not auto-generated)
- [ ] `MATRIXAI_ALLOW_REAL_ACTIONS` not set (real actions disabled by default)
- [ ] `MATRIXAI_RATE_LIMIT` reviewed for your expected traffic
- [ ] `MATRIXAI_CORS_ORIGINS` restricted to your actual client origins if browser-based clients are expected
- [ ] Server exposed via a TLS-terminating reverse proxy (nginx, Traefik, Caddy); the MatrixAI server itself does not terminate TLS
- [ ] Key history file (`matrixai_registry/.matrixai_key_history.json`) excluded from version control and backed up
