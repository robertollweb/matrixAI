# MatrixAI — Use Cases: Evaluator Guide

> **Español:** [docs/es/CASOS_DE_USO.md](../es/CASOS_DE_USO.md)

This guide is for someone evaluating whether MatrixAI solves a real problem.

Each case below is an industry problem that MatrixAI resolves end-to-end. You can run every case yourself in under 30 seconds with a single command. No API keys. No configuration. Data included.

---

## Before you start

From the project root, make sure MatrixAI is importable:

```bash
python3 -c "import matrixai; print('OK')"
```

If that fails, set up the path:

```bash
export PYTHONPATH=$(pwd)
```

---

## The four cases

| # | Industry | Problem | Key metric | Runtime |
|---|---|---|---|---|
| 1 | [Financial](#case-1--financial-credit-scoring) | Prove which exact model approved a loan | 93.3% vs 73.3% baseline | ~10 s |
| 2 | [SaaS / Support](#case-2--saas-text-routing-pipeline) | Route support tickets automatically, without losing traceability | 100% vs 33.3% baseline | ~20 s |
| 3 | [Healthcare](#case-3--healthcare-clinical-risk) | Risk decisions with an explanation that can be defended | 100% vs 58.3% baseline | ~10 s |
| 4 | [IT Operations](#case-4--it-operations-automated-agent) | Automated actions with audit trail, signing, and rollback | 83% vs 67% baseline | ~15 s |

---

## Case 1 — Financial: Credit Scoring

**The problem.** A fintech grants microloans and must prove to a regulator, for any historical decision, exactly which model version approved or rejected the application — and that the record was never altered.

**What MatrixAI does.** Trains a credit scoring model, registers it with a signed `entry_hash`, and links every scoring decision to that exact entry. Simulates a v1.0→v1.1 model upgrade and shows that old decisions still trace to v1.0. Demonstrates tamper detection.

```bash
python3 examples/credit-scoring/run_case.py
```

**What to look for:**
- `entry_hash` printed after registration: this is the cryptographic fingerprint of the model.
- The audit trail step: given any decision, the system recovers the exact model version that produced it.
- The tamper step at the end: modifying the registered `params.json` raises `VerificationError` in milliseconds.

**Full runbook:** [docs/en/cases/CREDIT_SCORING.md](cases/CREDIT_SCORING.md)

---

## Case 2 — SaaS: Text Routing Pipeline

**The problem.** A support team manually routes tickets into billing, technical, and sales queues. At 1000 tickets/day, manual routing doesn't scale. When a ticket is misrouted six months later, there's no way to know which model version caused it.

**What MatrixAI does.** Trains a two-stage composite pipeline: a TextEmbedder (BoW → Dense(8) → routing signal) and a RouteClassifier (signal → 3 categories). Both are registered independently. A single `composite_model_hash` links every routing decision to the exact version of *both* components simultaneously.

```bash
python3 examples/text-routing/run_case.py
```

**What to look for:**
- Step 3: 9 test tickets routed with 100% accuracy vs 33.3% baseline.
- Step 4: the audit trail for a single ticket — `composite_model_hash` expands to show both `feature_extractor@v1` and `route_classifier@v1`.
- Step 5: tampering with the TextEmbedder raises `VerificationError` even though the RouteClassifier was untouched.

**Full runbook:** [docs/en/cases/TEXT_ROUTING.md](cases/TEXT_ROUTING.md)

---

## Case 3 — Healthcare: Clinical Risk

**The problem.** A hospital decision-support system estimates fall risk. A model that only says "risk 0.87" is not acceptable in clinical practice: staff need to know *why*, and that explanation must be preserved for potential review.

**What MatrixAI does.** Trains a fall-risk classifier on 5 clinical features (age, mobility, medication load, previous falls, cognitive state). For every patient assessment, computes the exact linear contribution of each feature (`W1[i] × x_i`) — mathematically precise attribution for this model class. Every decision records the `entry_hash` of the exact model version that produced it.

```bash
python3 examples/clinical-risk/run_case.py
```

**What to look for:**
- Step 1: learned weights — `previous_falls W=+2.12` is the highest-impact feature.
- Step 2: each patient's risk level (BAJO/MEDIO/ALTO/ALERTA) with the primary contributing factor named.
- Step 3: full contribution breakdown for 3 representative patients, showing the `Σ contributions + bias` sum.
- Step 5: tamper detection — modifying the age weight (`W1[0]`) raises `VerificationError`.

> **Important:** this case demonstrates technical capability. It does not constitute clinical validation. A real clinical system requires medical validation and regulatory approval.

**Full runbook:** [docs/en/cases/CLINICAL_RISK.md](cases/CLINICAL_RISK.md)

---

## Case 4 — IT Operations: Automated Agent

**The problem.** An infrastructure monitoring system should automatically alert on critical events. But the organization requires proof of exactly what was sent, to whom, with which model, at what time — and the ability to reverse a wrong action.

**What MatrixAI does.** Trains an AlertModel, then runs every triggered action through the P20 framework: mandatory dry-run simulation, HMAC-signed execution, verifiable `ActionTrace`, and a declared rollback contract. Three guardrails are demonstrated: no signing key → blocked; unauthorized recipient → dry-run rejects; tampered trace → HMAC verification fails.

```bash
python3 examples/agent-alert/run_case.py
```

**What to look for:**
- Step 3: 4 events classified — 2 fire the alert, 2 are correctly ignored.
- Step 4.1: dry-run checks `scope_ok`, `rate_limit_ok`, `input_types_ok`, `rollback_ok` — all pass.
- Step 4.3: `ActionTrace` with HMAC signature; `verificación HMAC: OK`.
- Step 5: the three guardrails each block or reject as expected.
- Step 6: rollback executes a correction email — itself a traced action.

> **Note:** the email transport is mocked. No real email is sent. For production, pass real SMTP credentials via `MATRIXAI_SMTP_*` environment variables.

**Full runbook:** [docs/en/cases/AGENT_ALERT.md](cases/AGENT_ALERT.md)

---

## What is free and what is paid

The same boundary applies to all four cases:

| What is free | What is paid |
|---|---|
| Training and registering any model locally | Managed registry with retention, backup, access control |
| Signed `entry_hash` and tamper detection | Production API with SLA |
| Composite pipelines and `composite_model_hash` | Compliance audit report generation |
| Decision audit log per decision | Enterprise integrations (SMTP, HL7/FHIR, Slack, etc.) |
| Dry-run, HMAC-signed execution, rollback | Human-in-the-loop approval workflows and support |
| All four use cases above | Continual learning and drift monitoring in production |

The pattern: **the system is free to use locally. The paid tier is operating it at production scale, with the reliability, retention, and compliance guarantees that regulated industries require.**

---

## Next steps

- To understand the architecture behind a specific case: read its full runbook in `docs/en/cases/`.
- To understand the common format across cases: [docs/en/RUNBOOK_TEMPLATE.md](RUNBOOK_TEMPLATE.md).
- To see the full technical roadmap: [documentacion/OBJETIVO_FINAL_Y_ROADMAP.md](../../documentacion/OBJETIVO_FINAL_Y_ROADMAP.md).
- To get started with your own model: [docs/en/QUICKSTART.md](QUICKSTART.md).
