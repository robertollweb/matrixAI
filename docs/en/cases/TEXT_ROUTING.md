# Case 2 — Text Routing Pipeline: Composite Model Traceability

> **Español:** [docs/es/cases/TEXT_ROUTING.md](../../es/cases/TEXT_ROUTING.md)

**Industry:** SaaS / Customer Support  
**Difficulty:** Intermediate  
**Runtime:** ~20 seconds

---

## The problem

A SaaS company routes incoming support tickets to three queues — billing, technical, and sales — using an ML pipeline. The pipeline is composed of two independently trained models: a TextEmbedder (BoW → dense embedding → routing signal) and a RouteClassifier.

When a ticket is misrouted and a customer escalates six months later, the company needs to know:
- Which exact version of the TextEmbedder was active at that moment?
- Which exact version of the RouteClassifier was active?
- Has either been modified since?

With two separate models, the audit trail fragments across two model registries, two parameter stores, and two deployment logs. The risk: a model swap that was "minor" goes undocumented and becomes unverifiable.

---

## The solution

MatrixAI represents the two-stage pipeline as a single **composite model**. Each component is registered independently with its own `entry_hash`. The composite model produces a single `composite_model_hash` that is deterministically derived from both component hashes.

For every routing decision:
- The `composite_model_hash` links the decision to the exact version of **both** components simultaneously.
- Upgrading either component — even a "minor" parameter tweak — changes the composite hash automatically.
- `registry.verify()` proves neither component was altered after registration.

---

## Run it yourself

From the `matrixAI` root directory:

```bash
python3 examples/text-routing/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/text-routing/run_case.py
```

No external dependencies. No API keys. Dataset included.

### Expected output

```
MatrixAI — PR2-C2: Text Routing Pipeline with Composite Traceability
====================================================================

── Step 1 — Train & register TextEmbedder ──
  Registered: feature_extractor@v1
  entry_hash: sha256:18baae3b26e25519d...
  Architecture: TicketBOW[30] → Dense(8, relu) → Dense(1, sigmoid)

── Step 2 — Train & register RouteClassifier ──
  Registered: route_classifier@v1
  entry_hash: sha256:f628823298aefc2d6...
  RouteClassifier: routing_signal[1] → Dense(3, softmax)

── Step 3 — Route 9 test tickets ──
  Tickets routed: 9
  Pipeline accuracy:  100.0%
  Baseline (majority class): 33.3%
  Improvement: +66.7%
  Composite hash: sha256:9a0f529045084d72...

── Step 4 — Audit trail for TKT-0002 ──
  Decision: BILLING (confidence=0.9183)
  Ground truth: BILLING
  Composite hash: sha256:9a0f5290...
  TextEmbedder:    feature_extractor@v1 (sha256:18baae3b...)
  RouteClassifier: route_classifier@v1  (sha256:f6288232...)
  verify('feature_extractor', 'v1') → True  + intact
  verify('route_classifier', 'v1') → True   + intact

── Step 5 — Tamper detection ──
  Tamper detected — VerificationError: params.json content hash mismatch for feature_extractor@v1
  + Cryptographic chain caught modification of TextEmbedder
```

---

## The result

### Model metric

| Component | Role | Accuracy |
|---|---|---|
| TextEmbedder | TicketBOW[30] → Dense(8) → routing signal | N/A (regression head) |
| RouteClassifier | signal → billing/technical/sales | 100% on FE signal |
| **Composite pipeline** | raw text → routed category | **100% on 9 test tickets** |
| Baseline (majority class) | — | 33.3% |

On real production text with out-of-vocabulary jargon, expect 80–90% — the sustainable value is the signed traceability, not raw accuracy on a controlled corpus.

### Value metric

**Every routing decision is traceable to the exact version of both pipeline components.**

For any historical routing decision:
- The `composite_model_hash` identifies which TextEmbedder version AND which RouteClassifier version produced it.
- `registry.verify()` proves neither model was modified after registration.
- Upgrading the TextEmbedder to v2 changes the composite hash automatically — no manual tracking needed.
- Any retroactive tampering with either component is detected in milliseconds.

This directly eliminates the "which model caused this misroute?" ambiguity in multi-model pipelines — a problem that otherwise requires cross-referencing deployment logs, git history, and parameter backups.

---

## Architecture

```
Raw ticket text
       │
       ▼  vocabulary of 30 domain words (account, charge, crash, server, upgrade, pricing, …)
       ▼  TicketBOW[30] (binary presence per word)
┌─────────────────────┐
│  TextEmbedder       │  BoW[30] → Dense(8, relu) → Dense(1, sigmoid)
│  (FROZEN, registry) │  → routing_signal ∈ (0,1)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  RouteClassifier    │  signal → [p_billing, p_technical, p_sales]
│  (FROZEN, registry) │  argmax → category
└─────────────────────┘
       │
       ▼
  routing decision
```

The composite hash is:
```
composite_model_hash = SHA256({
  "own_program_hash": "<hash of text_routing_pipeline.mxai>",
  "imports": [
    {"alias": "TextEmbedder",    "entry_hash": "<fe_entry_hash>"},
    {"alias": "RouteClassifier", "entry_hash": "<rc_entry_hash>"}
  ]
})
```

---

## Limits

- The dataset is synthetic with a controlled 30-word vocabulary. Real accuracy on customer text depends on how much of the client's domain jargon is covered by the vocabulary; for new domains or non-English text, the vocabulary must be retrained on client data.
- BoW captures word presence only — not order, negation, or context. For those, swap Stage 1 for a more expressive encoder (transformer, fastText) keeping the same composite pattern.
- Both components are FROZEN in the composite. The `composite_training_step` API allows TRAINABLE components for fine-tuning the second stage without retraining the first.
- The registry is local-first. Production deployments would use managed registry infrastructure (paid tier).
- The pipeline does not include HTTP serving in this cut (HTTP is P6, available in the system but not demonstrated here).

---

## What is free and what is paid

| Layer | Status |
|---|---|
| FeatureExtractor training and registration | **Core — free** |
| RouteClassifier training and registration | **Core — free** |
| Composite pipeline with `composite_model_hash` | **Core — free** |
| Decision audit log with per-decision composite hash | **Core — free** |
| Tamper detection for all registered components | **Core — free** |
| Managed registry with retention and access control | Paid tier |
| Production routing API with SLA | Paid tier |
| Audit report generation for enterprise compliance | Paid tier |

---

## Files

```
examples/text-routing/
  feature_extractor.mxai       — Stage 1: TextEmbedder NETWORK on TicketBOW[30]
  route_classifier.mxai        — Stage 2: routing signal → 3 categories (softmax)
  ticket_router.mxai           — Standalone reference router
  text_routing_pipeline.mxai   — Composite: TextEmbedder + RouteClassifier (both FROZEN)
  ticket_router.mxtrain        — Training spec for standalone router
  data/
    train.csv                  — 45 synthetic tickets (15 per category), raw text + 30 bow_* columns
    test.csv                   — 9 held-out tickets (3 per category)
  run_case.py                  — End-to-end demo script (auto-generates CSVs on first run)
```
