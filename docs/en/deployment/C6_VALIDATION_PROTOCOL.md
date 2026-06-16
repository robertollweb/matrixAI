# PR4-C6 — External Operator Validation Protocol

> **Español:** [docs/es/deployment/C6_VALIDACION_PROTOCOLO.md](../../es/deployment/C6_VALIDACION_PROTOCOLO.md)

This document defines the formal protocol for closing PR4-C6. An external operator — not the author, not someone who built the system — must complete the full operational cycle on a clean machine using only the MatrixAI documentation. The author observes and records but does not help.

---

## What is MatrixAI?

MatrixAI is an open-source framework for building verifiable, auditable, and deployable AI models. It covers the full lifecycle: defining models in `.mxai` files, training with versioned parameters, serving via HTTP, and monitoring drift in production.

**This document is for operators.** You do not need to understand the internals of MatrixAI — you only need to follow the steps here and in the linked guides.

---

## Getting started — install MatrixAI (5 minutes)

**Requirements:** Python 3.8+, Git, Docker ≥ 24.

```bash
# 1. Clone the repository
git clone https://github.com/robertollweb/matrixAI.git
cd matrixAI

# 2. Verify the CLI works (no install needed — runs from the repo)
python3 -m matrixai --help
```

If you want to understand MatrixAI basics before starting the C6 cycle, read the quickstart first (optional — not required to complete C6):

- [🇬🇧 Quickstart (5 min)](../QUICKSTART.md) — first model, first prediction
- [🇬🇧 Deployment guide](DEPLOYMENT.md) — pack and deploy with Docker
- [🇬🇧 Observability guide](OBSERVABILITY.md) — `/metrics` and drift monitoring
- [🇬🇧 Operational runbook](RUNBOOK.md) — what to do when things fail

---

## Operator profile

- Has experience operating backend services (Docker, curl, shell).
- Has never used MatrixAI before.
- Has no access to the source code or the author's machine.
- Has internet access (to pull Docker images) and Docker ≥ 24 installed.

---

## Pre-conditions (author prepares, operator does not touch)

Before the session, the author must:

1. Run the smoke test on the operator's machine (or share the package):
   ```bash
   python3 scripts/smoke_test_c6.py
   # Expected: 28 OK | 0 FAIL
   ```
2. Pack the credit-scoring model with continual policy:
   ```bash
   matrixai pack examples/credit-scoring/credit_scoring.mxai \
     --params examples/credit-scoring/registry/entries/credit-scoring/v1.1/params.json \
     --docker \
     --outdir dist/credit-scoring-c6
   # Copy dist/credit-scoring-c6/ and examples/credit-scoring/ to the operator's machine
   ```
3. Transfer to the operator's machine:
   - `dist/credit-scoring-c6/` (the Docker package)
   - `examples/credit-scoring/credit_scoring.mxcontinual` (continual policy)
   - `examples/credit-scoring/registry/` (the model registry)
   - The guides: `DEPLOYMENT.md`, `OBSERVABILITY.md`, `RUNBOOK.md`

The operator receives only the above. **No Python installation, no source code.**

---

## Cycle to complete

The operator must complete all 6 phases in order. The observer records the time and any friction at each phase.

### Phase 1 — Deploy (est. 10 min)

**Objective:** server running and healthy.

**Documents:** [DEPLOYMENT.md](DEPLOYMENT.md)

```bash
cd dist/credit-scoring-c6
cp .env.example .env
# Edit .env: set MATRIXAI_API_KEY=<openssl rand -hex 32>
docker compose up --build -d
docker compose ps        # Status must show: healthy
curl http://localhost:8000/health
```

**Pass criterion:** `{"status": "ok"}` returned within 30 seconds of `docker compose up`.

---

### Phase 2 — Prediction (est. 5 min)

**Objective:** obtain a valid prediction from the model.

**Documents:** [DEPLOYMENT.md](DEPLOYMENT.md)

```bash
curl -s \
  -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"income_score":0.7,"credit_history":0.8,"debt_ratio":0.2,"employment_years":0.6,"loan_amount_ratio":0.3}' \
  http://localhost:8000/predict
```

**Pass criterion:** response contains a numeric prediction value.

---

### Phase 3 — Enable drift monitoring (est. 5 min)

**Objective:** restart the server with continual policy to enable `/feedback` and drift metrics.

**Documents:** [OBSERVABILITY.md](OBSERVABILITY.md)

The operator must stop the Docker container and start the server directly with `--continual-policy` (Docker image does not yet include the policy — this is an intentional manual step for C6):

```bash
docker compose stop

# From the machine with matrixai installed (or unpack matrixai/ from the dist dir):
python3 -m matrixai serve \
  examples/credit-scoring/credit_scoring.mxai \
  --params examples/credit-scoring/registry/entries/credit-scoring/v1.1/params.json \
  --continual-policy examples/credit-scoring/credit_scoring.mxcontinual \
  --api-key $MATRIXAI_API_KEY \
  --host 0.0.0.0 --port 8000
# Expected: "Continual monitoring active (policy: CreditScoringContinual, reference_accuracy: 0.8750)"
```

Verify drift metrics available:
```bash
curl http://localhost:8000/metrics | grep drift
# matrixai_drift_window_accuracy{...} 0.0
# matrixai_drift_degradation_detected{...} 0
```

**Pass criterion:** `/metrics` contains `matrixai_drift_degradation_detected`.

---

### Phase 4 — Induce drift (est. 5 min)

**Objective:** send enough wrong predictions to trigger degradation detection.

**Documents:** [OBSERVABILITY.md](OBSERVABILITY.md)

The operator sends deliberately wrong ground truth labels (predicting approved=1, actual=0) to simulate a model that stopped working:

```bash
for i in $(seq 1 6); do
  curl -s -X POST \
    -H "Authorization: Bearer $MATRIXAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"prediction\":\"1\",\"ground_truth\":\"0\",\"trace_id\":\"drift_$i\"}" \
    http://localhost:8000/feedback
done
```

Check that drift was detected:
```bash
curl -s http://localhost:8000/metrics | grep drift_degradation_detected
# matrixai_drift_degradation_detected{...} 1
```

**Pass criterion:** `matrixai_drift_degradation_detected` equals 1 in `/metrics`.

---

### Phase 5 — Execute rollback (est. 5 min)

**Objective:** roll back the model from v1.1 to v1.0 following the runbook.

**Documents:** [RUNBOOK.md](RUNBOOK.md) — Scenario 1

```bash
# Dry-run first
python3 -m matrixai continual rollback \
  examples/credit-scoring/credit_scoring.mxcontinual \
  --registry-dir examples/credit-scoring/registry \
  --dry-run
# Expected: [dry-run] Would rollback credit-scoring
#             from: v1.1 → to: v1.0

# Execute
python3 -m matrixai continual rollback \
  examples/credit-scoring/credit_scoring.mxcontinual \
  --registry-dir examples/credit-scoring/registry
# Expected: Rolled back credit-scoring
#             from: v1.1 (ps=v1.1_best) → to: v1.0 (ps=v1.0_best)
```

**Pass criterion:** command exits 0 and confirms `v1.1 → v1.0`.

---

### Phase 6 — Verify rollback and check status (est. 5 min)

**Objective:** confirm the rollback completed and the registry shows v1.0 as current.

**Documents:** [RUNBOOK.md](RUNBOOK.md) — Scenario 2

```bash
python3 -m matrixai continual status \
  examples/credit-scoring/credit_scoring.mxcontinual \
  --registry-dir examples/credit-scoring/registry
# Expected:
# Registry       : credit-scoring
# Current version: v1.0  (ps=v1.0_best)
# Last rollback  : v1.1 → v1.0  (manual)
#   executed_at  : 2026-...

python3 -m matrixai registry verify \
  --registry-path examples/credit-scoring/registry \
  credit-scoring@v1.0
# Expected: OK: credit-scoring@v1.0 integrity verified
```

**Pass criterion:** `Current version: v1.0` and `registry verify` passes.

---

## Recording sheet (observer fills in)

| Phase | Start | End | Passed | Friction / notes |
|---|---|---|---|---|
| 1 — Deploy | | | ☐ | |
| 2 — Prediction | | | ☐ | |
| 3 — Enable monitoring | | | ☐ | |
| 4 — Induce drift | | | ☐ | |
| 5 — Rollback | | | ☐ | |
| 6 — Verify | | | ☐ | |

**Operator asked for help:** yes / no  
**External source consulted:** yes / no (if yes: what?)  
**Total time:** _____ min

---

## Pass / fail criteria for C6 closure

PR4-C6 **passes** if:
- All 6 phases completed with ☐ checked.
- The operator did not ask the author for help at any point.
- The operator did not consult any source outside the MatrixAI documentation.

PR4-C6 **fails** if:
- Any phase is not completed.
- The operator asked the author for help to resolve a blocker.
- The operator needed to read source code.

**Recorded friction** (operator struggled but solved it alone with documentation) is tolerated and noted — it identifies documentation gaps to fix before PR5.

---

## Smoke test (operator can run before starting)

```bash
python3 scripts/smoke_test_c6.py
# Expected: 28 OK | 0 FAIL | Entorno listo para el ciclo C6.
```

---

## Related guides

| Guide | When to use |
|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Phases 1–2 |
| [OBSERVABILITY.md](OBSERVABILITY.md) | Phases 3–4 |
| [RUNBOOK.md](RUNBOOK.md) | Phases 5–6 |
| [KEY_ROTATION.md](KEY_ROTATION.md) | If signing key needs rotation during C6 |
