# Versioning and compatibility

MatrixAI uses [Semantic Versioning 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH`.

> **Español:** [docs/es/VERSIONING.md](../es/VERSIONING.md)

---

## What v1.0 means

v1.0.0 is the first stable public release. It marks the point where the language,
runtime, CLI, HTTP API and model registry are considered ready for production use
in critical environments.

From v1.0.0 onward this project follows a clear compatibility policy, described below.

---

## What is stable in v1.0

The following interfaces are stable and covered by the compatibility guarantee:

| Interface | Scope |
|-----------|-------|
| **`.mxai` language** | All node types, expressions and type constructs in the language spec |
| **`.mxtrain` format** | All training spec fields, optimisers and loss functions |
| **`.mxact` format** | All action contract fields and execution semantics |
| **`.mxcontinual` format** | All continual learning policy fields and trigger semantics |
| **CLI commands and flags** | All commands in `matrixai --help` and `CLI_REFERENCE.md` |
| **HTTP API** | Endpoints in `REST_API.md`; will be available under `/api/v1/` once versioned in this release cycle — backwards-compatible aliases will be maintained |
| **Model registry format** | Entry structure, signature scheme, `matrixai_version` field |
| **PyPI package name** | `matrixai-core`; entry point `matrixai` |
| **Docker image tags** | `ghcr.io/robertollweb/matrixai:MAJOR.MINOR.PATCH` and `:latest` |

---

## What is not stable

The following may change without a major version bump:

- **Internal Python module structure** — imports from `matrixai.compiler.*`,
  `matrixai.agents.*`, `matrixai.ir.*` are internal. Use the CLI or the HTTP API.
- **LLM bridge** — environment variable names and behaviour may change while
  this feature matures.
- **Playground output format** — subject to improvement.

---

## Version change rules

| Type of change | Bump | Examples |
|----------------|------|---------|
| Breaking change in language, CLI, HTTP API or registry | **Major** (2.0.0) | Remove a node type, rename a required CLI flag, change HTTP response shape |
| New backwards-compatible feature | **Minor** (1.1.0) | New node type, new CLI command, new HTTP endpoint |
| Bug fix, no API change | **Patch** (1.0.1) | Fix training convergence, fix error message |

A breaking change is one that requires a user to modify their `.mxai`, `.mxtrain`,
`.mxact` or `.mxcontinual` files, their CLI scripts, or their HTTP API integration.

---

## Model compatibility

Each entry in the model registry records the `matrixai_version` that created it.
This is the MatrixAI product version, separate from the registry index schema
version stored in `registry.json`.

- Models trained with **v1.0.x** load and run on any **v1.0.y**.
- Models trained with **v1.x.y** load on **v1.x.z** (z ≥ y) and on **v1.w.0** (w > x).
- A **major version bump** may require a migration step. Migration guides are
  published in `CHANGELOG.md` when this occurs.
- `matrixai registry verify` validates the entry hash, artifact checksums and
  signature. It emits a warning when the entry's major version differs from the
  running version.

---

## Deprecation policy

1. A feature is marked deprecated in `CHANGELOG.md` and emits a runtime warning when used.
2. A deprecated feature is not removed before the next minor release.
3. Features are never removed in a patch release.
4. Each feature goes through at most one deprecation cycle: deprecated in v1.x,
   removed no earlier than v1.(x+1).

---

## Support timeline

| Series | Status |
|--------|--------|
| 1.0.x | Active — bug fixes and security patches |
| < 1.0 | End of life — no fixes |

This project is maintained by a small team. Response targets are in `SECURITY.md`.
