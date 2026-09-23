# MatrixAI

**MatrixAI is a language for AI, not for humans.** Describe a model in a prompt, train it, audit every decision it makes, and deploy it where trust is not optional.

Models are not black boxes — they are auditable programs: explicit inputs, explicit transformations, explicit outputs, explicit audit trail. Every decision is traceable to a named node in the computation graph. That is the core value for critical environments: healthcare, finance, legal, industrial.

> **Website & Studio:** [matrixaistudio.org](https://matrixaistudio.org) — browser-based model development environment, downloads, documentation and member resources.

---

## Get started

```bash
pip install matrixai-core
matrixai --help
```

→ [Quickstart (5 min) 🇬🇧](docs/en/QUICKSTART.md) · [Quickstart (5 min) 🇪🇸](docs/es/QUICKSTART.md)

---

## What MatrixAI does

1. **Describe** — write a model in a natural-language prompt or in `.mxai` directly.
2. **Generate** — the system builds a verifiable computation graph and training contract.
3. **Train** — supervised training with versioned parameters, reproducible metrics and full trace.
4. **Audit** — every prediction is traceable; every action is signed and logged.
5. **Deploy** — serve over HTTP, export to ONNX/WASM, package as Docker, or register in the model registry.
6. **Monitor** — detect drift, trigger retraining, rollback automatically or manually.

---

## Key features

- **Prompt → model**: `matrixai prompt "..."` generates a runnable `.mxai` program
- **A data recipe you write yourself**: synthetic data used to have no relationship
  between inputs and target (measured correlation 0.049 against a 0.098 chance
  threshold), so a model trained on it could not learn. A recipe —
  `1: debt > 60000 OR income < 20000` / `DEFAULT: 0` / `NOISE: 0.1` /
  `BALANCE: 1=0.3` — is parsed and evaluated deterministically, with no LLM in the
  loop, works for continuous targets too (`usage = 0.05*sqm + 1.2*people + 5`),
  travels with the model and ships inside the exported bundle as `data_recipe.txt`
- **The task comes from the question, not the verb**: «predict which customers will
  churn» is a classification even though «predict» reads like regression. The inferred
  task reaches the pipeline and is stated in the trace, and the assumption is flagged
  only when it was a default — not when the prompt said it
- **Typed prompt fields**: declare feature types and ranges in the prompt itself
  (`edad: Scalar en [18, 95]`, `Integer[1, 10]`, `Boolean`, `Categorical[...]` → one-hot,
  `ProbabilityMap[NO, SI]` output) — honoured end-to-end by the generator, the LLM
  proposal, the synthetic data and the export metadata
- **Model generation from real data**: point at your own CSV instead of writing a prompt —
  schema inference (types, ranges, one-hot categoricals, temporal columns), target-column
  detection (classification or regression) and a trained model in one pass, with target
  normalization so regression targets in any scale (not just `[0, 1]`) actually converge
- **External data providers**: pluggable registry for pulling real datasets from
  third-party APIs instead of a manual CSV upload — license acceptance tracking and
  SSRF-hardened fetching (fixed host allowlist, redirect validation, DNS-rebinding
  protection). Ships an Open-Meteo provider (historical weather/marine data)
- **Sequence & Transformer models**: `SEQUENCE` inputs, `BLOCK <name> TRANSFORMER` (multi-head
  attention, feed-forward, layer norm, positional encoding) and a byte-level tokenizer for
  text classification — trained end-to-end on GPU (torch backend) and exported like any
  other network. See `examples/transformer-classifier.mxai`
- **Model studies with risk stated per decision** (`matrixai.estudio`): decision bands
  (act / review / act-not) chosen by declared costs and measured on held-out data with
  their interval, split-conformal prediction sets and intervals, and a per-row report of
  what the model never saw (out-of-range values, unseen categories). Dates enter as
  their own variables, and a `0/1` or `Yes/No` target works whatever way you write the
  positive class
- **Auditable graph**: computation graph with named nodes, explicit types and audit trail
- **Supervised training**: classification, risk scoring and regression with `.mxtrain` specs
- **Large models (billions of parameters)**: binary `.mxw` weights format with tamper
  detection, pre-training resource estimator (VRAM/RAM/disk/time), torch/GPU end-to-end
  (train, evaluate, infer, resume) and streamed ONNX external-data export — validated
  with a 2.95B-parameter dense model on an A100
- **Verifiable pipelines**: a pipeline is a deterministic, fail-closed JSON policy that
  **names the rule that fired**, plus an engine that resolves every component **by digest**
  and verifies it *before* running it. What did not start is said, not silently skipped
- **Decision receipts**: every run leaves a DSSE-signed `.mxreceipt` whose assurance level
  (A0–A4) is **deduced from what was actually checked**, never declared. `matrixai receipt
  inspect | verify | compare` — and `inspect` verifies nothing and says so
- **Reproducible packages**: `matrixai verify` runs four stages and reports each one, and
  `matrixai replay --compare-reference` **names the stages that differ** instead of saying
  «something changed». It always states whether both runs used the same environment:
  matching inside one environment proves repeatability, not reproducibility. A run
  elsewhere returns `INCOMPARABLE` — it neither accuses nor approves for free
- **A receipt for a model you did not train**: `matrixai attest model.onnx --data eval.csv`
  runs someone else's ONNX model on your data and ties the number to the digests of both.
  What makes it worth something is what it refuses to claim — the receipt carries
  `evidence.does_not_attest` and `provenance: external`, and says that a copied model
  produces exactly the same receipt as your own. Its ceiling is **A1**, written in the
  receipt and not only in the docs
- **Formats other tools already read**: the DSSE envelope is spec-conformant, `matrixai
  attest --in-toto` emits an in-toto Statement, `matrixai bom` emits a **CycloneDX 1.6
  ML-BOM** validated against the official schema and deterministic by construction (a BOM
  that changes on every run can be neither compared nor signed), and `--sigstore` signs
  with Sigstore — which does **not** raise the assurance level, because A0–A4 describe
  what was checked, not how strong the signature is
- **CI verification as a GitHub Action**: `matrixai-verify` (a composite
  action shipped in this repo, `.github/actions/matrixai-verify/`) runs
  `matrixai verify` against an exported package and fails the job with the
  verdict and the exact scopes that were **not** carried out — it never
  reports greener than `verify` itself, and `require` can only tighten
  which scopes are mandatory, never loosen them. `matrixai-engines` is
  installed only if the package's own manifest asks for it. Needs
  `matrixai-core >= 1.8.0` — the first release that ships `matrixai/ci/` —
  and says so out loud instead of crashing if an older version is pinned
- **PROBAST+AI gaps alongside TRIPOD+AI**: `matrixai report --probast`
  reports what an exported package supports for risk-of-bias and
  applicability review, what a person declared, and what's missing with
  the manifest field that would back it — same discipline as the existing
  `--tripod` report: it enumerates gaps, it does not score or rate risk of
  bias
- **No arbitrary ceiling on CSV size for your own machine**: the old fixed
  50 MB / 50,000-row limit is gone for local and self-hosted use; a
  measured memory guard (empirically ×12 the CSV's byte size) refuses a
  file only when it is actually likely to exhaust RAM, and can be turned
  off. The shared matrixaistudio.org demo keeps the old hard limits, since
  there the machine isn't the uploader's own
- **`matrixai attest` measures three more metrics**: `macro_f1`, `rmse`
  and `r2`, in addition to `accuracy`/`mae`, computed by the same metrics
  registry as the rest of the package instead of a hand-rolled formula —
  and it now refuses a metric/output combination that doesn't match the
  model's task (e.g. asking for MAE against a classification output)
  instead of silently reporting a meaningless number
- **TRIPOD+AI record**: `matrixai report <package> --tripod` writes the checklist a
  clinical-prediction journal asks for, from what the package already captured — and
  **enumerates every box it cannot fill** instead of filling it
- **Model registry**: versioned, signed, verifiable — `matrixai registry push/pull/verify`,
  with interface types derived from the model itself, so composing two components is
  **checked** instead of assumed (what cannot be determined is published without types
  and shown as such: an invented type is worse than none)
- **Real actions**: `.mxact` contracts with HMAC-signed traces, dry-run and rollback
- **Continual learning**: `.mxcontinual` policies with drift detection and automatic
  versioning — accepting a suggestion creates a **candidate**, never a deployment;
  promoting stays a separate, human act
- **HTTP server**: `/predict`, `/metrics` (Prometheus), `/execute-action`, `/feedback` with API key auth
- **ONNX / WASM export**: edge deployment bundles and browser-ready WASM packages — for
  dense **and composite** networks (residual blocks, LayerNorm, embeddings, concat), with
  output equivalence validated against the reference forward pass; the browser runtime is
  pinned to ONNX Runtime Web 1.20.1, and a test requires it to load the model IR version
  the exporter actually writes
- **Self-usable model bundles**: the exported bundle ships `model.onnx` + `predict.py` +
  `inference_spec.json` — it predicts from **raw human values** (same normalization and
  one-hot encoding as training) with no MatrixAI installation, only `onnxruntime` — plus a
  `space/` folder for a **static** Hugging Face Space that runs the model in the visitor's
  browser (no paid account needed; it states that it does not verify the package)
- **Studio**: browser-based model development environment — a separate product at [matrixaistudio.org](https://matrixaistudio.org), built on this core

---

## Quick example

```bash
# Create a project from a template
python -m matrixai init my-model --template classification

# Train
python -m matrixai train my-model/my-model.mxai \
  --training my-model/my-model.mxtrain \
  --output my-model/runs/v1

# Predict
python -m matrixai run my-model/my-model.mxai \
  --params my-model/runs/v1/params.best.json \
  --input my-model/input/sample.json

# Serve over HTTP
python -m matrixai serve my-model/my-model.mxai \
  --params my-model/runs/v1/params.best.json \
  --api-key my-secret
# → http://127.0.0.1:8000/docs
```

---

## Examples

| Example | Domain | Mode |
|---------|--------|------|
| `examples/credit-scoring/` | Credit approval | Risk scoring |
| `examples/clinical-risk/` | Fall risk assessment | Risk scoring |
| `examples/agent-alert/` | Alert monitoring with real action | Classification + action |
| `examples/text-routing/` | Support ticket routing | Multi-class classification |
| `examples/email-agent.typed.mxai` | Email classification | Classification |
| `examples/celsius_to_kelvin.mxai` | Temperature conversion | Regression |
| `examples/transformer-classifier.mxai` | Transformer encoder | Classification |

---

## Documentation

| Topic | English | Español |
|-------|---------|---------|
| Quickstart | [QUICKSTART.md](docs/en/QUICKSTART.md) | [QUICKSTART.md](docs/es/QUICKSTART.md) |
| Tutorial | [TUTORIAL.md](docs/en/TUTORIAL.md) | [TUTORIAL.md](docs/es/TUTORIAL.md) |
| Language spec | [LANGUAGE_SPEC.md](docs/en/api/LANGUAGE_SPEC.md) | [LANGUAGE_SPEC.md](docs/es/api/LANGUAGE_SPEC.md) |
| CLI reference | [CLI_REFERENCE.md](docs/en/api/CLI_REFERENCE.md) | [CLI_REFERENCE.md](docs/es/api/CLI_REFERENCE.md) |
| REST API | [REST_API.md](docs/en/api/REST_API.md) | [REST_API.md](docs/es/api/REST_API.md) |
| Use cases | [USE_CASES.md](docs/en/USE_CASES.md) | [CASOS_DE_USO.md](docs/es/CASOS_DE_USO.md) |
| Benchmarks | [INDEX.md](docs/en/benchmarks/INDEX.md) | [INDEX.md](docs/es/benchmarks/INDEX.md) |
| Deployment | [DEPLOYMENT.md](docs/en/deployment/DEPLOYMENT.md) | [DEPLOYMENT.md](docs/es/deployment/DEPLOYMENT.md) |
| Observability | [OBSERVABILITY.md](docs/en/deployment/OBSERVABILITY.md) | [OBSERVABILITY.md](docs/es/deployment/OBSERVABILITY.md) |
| Runbook | [RUNBOOK.md](docs/en/deployment/RUNBOOK.md) | [RUNBOOK.md](docs/es/deployment/RUNBOOK.md) |
| Key rotation | [KEY_ROTATION.md](docs/en/deployment/KEY_ROTATION.md) | [KEY_ROTATION.md](docs/es/deployment/KEY_ROTATION.md) |
| Server hardening | [SERVER_HARDENING.md](docs/en/deployment/SERVER_HARDENING.md) | [SERVER_HARDENING.md](docs/es/deployment/SERVER_HARDENING.md) |
| Versioning policy | [VERSIONING.md](docs/en/VERSIONING.md) | [VERSIONING.md](docs/es/VERSIONING.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) | [CHANGELOG.md](CHANGELOG.md) |
| Business model | [BUSINESS_MODEL.md](docs/en/BUSINESS_MODEL.md) | [MODELO_NEGOCIO.md](docs/es/MODELO_NEGOCIO.md) |

---

## Install

```bash
pip install matrixai-core
```

**With optional export dependencies (ONNX / WASM):**

```bash
pip install "matrixai-core[export]"
```

**With GPU training support (PyTorch):**

```bash
pip install "matrixai-core[torch]"
```

**All extras:**

```bash
pip install "matrixai-core[export,torch,dev]"
```

**From source:**

```bash
git clone https://github.com/robertollweb/matrixAI.git
cd matrixAI
pip install -e .
```

**Requirements:** Python 3.10+ must be installed on your system ([python.org/downloads](https://www.python.org/downloads/)).

> **Windows note:** use `python` instead of `python3` in all commands below.  
> If `matrixai` is not found after install, use `python -m matrixai` (or `python3 -m matrixai` on Linux/macOS).

---

## Running MatrixAI

After installing, you can call MatrixAI in two equivalent ways:

```bash
# Option A — direct command (works when pip scripts directory is in PATH)
matrixai --help

# Option B — via Python module (always works, recommended on Windows)
python -m matrixai --help       # Windows
python3 -m matrixai --help      # Linux / macOS
```

---

## LLM configuration (optional)

MatrixAI works without any LLM — it uses a built-in deterministic engine by default. To enable LLM-powered model generation, copy the example config and fill in your API key:

```bash
cp .env.example .env
```

Then edit `.env` and set your provider and key. Minimal example for **OpenAI**:

```
MATRIXAI_LLM_PROVIDER_NAME=openai
MATRIXAI_LLM_MODEL=gpt-4o-mini
MATRIXAI_LLM_API_KEY=sk-...your-key...
```

For **Anthropic (Claude)**:

```
MATRIXAI_LLM_PROVIDER_NAME=anthropic
MATRIXAI_LLM_MODEL=claude-opus-4-8
MATRIXAI_LLM_API_KEY=sk-ant-...your-key...
MATRIXAI_LLM_MAX_TOKENS=4096
```

For **Google Gemini** or **DeepSeek** — see the full list of providers and example configs in [`.env.example`](.env.example).

> Without a `.env` file (or with `MATRIXAI_LLM_API_KEY` empty), MatrixAI runs in **deterministic mode**: all features work except LLM-generated model suggestions.

---

## Studio

MatrixAI Studio is a browser-based model development environment — generate models from
prompts, train, evaluate and explore without writing code. It is distributed as a
separate product built on this core.

→ **[matrixaistudio.org](https://matrixaistudio.org)** — downloads, documentation and member resources.

The core itself ships a local technical playground (prompt → runtime):

```bash
python -m matrixai playground --open
# → http://127.0.0.1:8765
```

---

## Run the tests

```bash
python -m pytest tests/
# 8110 passed, 22 skipped
```

---

## LLM integration (optional)

MatrixAI can use an external LLM to generate model proposals from prompts. Without configuration it falls back to the deterministic local mode.

```bash
# .env (ignored by git)
MATRIXAI_LLM_API_KEY=your-key
MATRIXAI_LLM_MODEL=external-model-id
MATRIXAI_LLM_ENDPOINT=https://provider.example/v1/chat/completions
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MATRIXAI_LLM_API_KEY` | — | External provider key |
| `MATRIXAI_LLM_MODEL` | configured by you | Model identifier sent to the external provider |
| `MATRIXAI_LLM_ENDPOINT` | chat-completions-compatible endpoint | Provider endpoint |
| `MATRIXAI_LLM_CANDIDATES` | `1` | Number of candidates to generate |
| `MATRIXAI_LLM_TEMPERATURE` | `0` | Generation temperature |
| `MATRIXAI_LLM_TOKEN_BUDGET` | `0` (unlimited) | Max tokens per call |

Any chat-completions-compatible API can be used, including local model servers.

---

## License

See [LICENSE](LICENSE) — AGPL v3. License verification: [English](docs/en/LICENSE_VERIFICATION.md) · [Español](docs/es/VERIFICACION_LICENCIA.md).  
© Roberto Llamosas Conde — `robertollweb/matrixAI`
