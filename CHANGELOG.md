# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-05-30

First stable public release.

### Language and runtime

- **`.mxai` language** — complete type system: scalars, vectors, matrices, embeddings,
  probability distributions and composite types. Node types: `VECTOR`, `FUNCTION`,
  `DENSE`, `TRANSFORMER`, `ACTION`, `EMBEDDING`. Explicit input/output declarations,
  named parameters, fully auditable computation graph.
- **`.mxtrain` training spec** — supervised training with SGD, Adam and L-BFGS optimisers;
  loss functions: binary cross-entropy, MSE, categorical cross-entropy. Configurable
  epochs, batch size and learning rate. Reproducible training with deterministic seeding.
- **`.mxact` action contracts** — HMAC-signed action traces with dry-run, approval gates,
  rollback and tamper detection. Real-action execution governed by
  `MATRIXAI_ALLOW_REAL_ACTIONS` environment variable.
- **`.mxcontinual` policies** — drift detection, automatic versioning, rollback triggers
  and configurable retraining windows.

### CLI

Complete command surface:

| Command | Description |
|---------|-------------|
| `matrixai init` | Scaffold a new project from a template |
| `matrixai train` | Train a model from a `.mxtrain` spec |
| `matrixai run` | Run inference on a trained model |
| `matrixai serve` | Serve a model over HTTP with auth and rate limiting |
| `matrixai prompt` | Generate a `.mxai` program from a natural-language prompt |
| `matrixai studio` | Launch the browser-based model development environment |
| `matrixai pack` | Package a model for Docker deployment |
| `matrixai export` | Export a model to ONNX or WASM |
| `matrixai registry` | Push, pull, verify and inspect model registry entries |
| `matrixai keys` | Rotate and list signing keys |
| `matrixai continual` | Manage continual learning policies |

### HTTP server

- REST API: `/predict`, `/execute-action`, `/feedback`, `/health`, `/metrics`
- Prometheus metrics endpoint: request counts, latency, drift gauges
- API key authentication (`X-API-Key` header and `Authorization: Bearer`)
- Configurable CORS origins and rate limiting (sliding window per IP)
- OpenAPI spec at `/docs`

### Model registry

- Versioned, signed entries with HMAC verification
- `matrixai_version` field on every entry for compatibility tracking
- Tamper detection across registry, traces and parameter files
- Signing key rotation with historical verification by fingerprint

### Training and learning

- Classification, risk scoring and regression pipelines
- Dense networks, transformer encoders, composite architectures
- GPU training via optional PyTorch backend (`pip install matrixai-core[torch]`)
- Synthetic data generation for rapid prototyping
- Continual learning: drift detection, rollback and automatic retraining

### Export and deployment

- ONNX export with equivalence validation (`pip install matrixai-core[export]`)
- WASM bundles for browser and edge deployment
- `matrixai pack --docker` generates a production-ready Dockerfile and Compose file
- Official Docker image: `ghcr.io/robertollweb/matrixai:1.0.0`

### Distribution

- PyPI package: `pip install matrixai-core`
- Docker image: `ghcr.io/robertollweb/matrixai:1.0.0` and `:latest`
- SHA-256 checksums published with each release
- PyPI Trusted Publishing — no manual credentials in CI

### Project templates

- `matrixai init --template classification` — binary classification with a ready-to-train
  dataset, model and training spec included

### License

AGPL-3.0-only. See `LICENSE`.

---

## [Unreleased]

Nothing yet.

[1.0.0]: https://github.com/robertollweb/matrixAI/releases/tag/v1.0.0
[Unreleased]: https://github.com/robertollweb/matrixAI/compare/v1.0.0...HEAD
