# Case 2 — Text Routing Pipeline: Composite Model Traceability

> **Español:** [docs/es/cases/TEXT_ROUTING.md](../../es/cases/TEXT_ROUTING.md)

**Industry:** SaaS / Customer Support  
**Difficulty:** Intermediate  
**Runtime:** ~1 second (measured: 0.79s / 1.11s / 0.79s over three runs)

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
MatrixAI — PR2-C2: Text Routing with BoW Text Embedding + Composite Traceability
================================================================================

  Vocabulary (30 words): account, amount, api, cancel, charge, connection, contract, crash, demo, discount ...
  Architecture: ticket text → BoW[30] → Dense(8,relu) → Dense(1,sigmoid)
                                                      → Dense(3,softmax) → category

────────────────────────────────────────────────────────────────
  Step 1 — Train TextEmbedder: BoW[30] → Dense(8,relu) → Dense(1,sigmoid)
────────────────────────────────────────────────────────────────
  This is the learned text embedding: maps raw ticket words to a routing signal.
  Registered: feature_extractor@v1
  entry_hash: sha256:57ac18f47f1286f7bd3d7...
  Embedding signal range: billing~0.1  technical~0.5  sales~0.9

────────────────────────────────────────────────────────────────
  Step 2 — Text embedding demo: raw ticket → BoW → routing signal
────────────────────────────────────────────────────────────────

  Text:    "need help with my invoice the charge amount is wrong on my accoun"
  BoW:     [account, amount, charge, help, invoice, need]
  Signal:  0.0643  (expected category: billing)

  Text:    "api is crashing network connection is very slow please fix this i"
  BoW:     [api, connection, fix, issue, network, slow]
  Signal:  0.4597  (expected category: technical)

  Text:    "want to upgrade to enterprise plan please send pricing informatio"
  BoW:     [enterprise, plan, pricing, upgrade]
  Signal:  0.8859  (expected category: sales)

────────────────────────────────────────────────────────────────
  Step 3 — Train RouteClassifier: routing_signal → billing/technical/sales
────────────────────────────────────────────────────────────────
  Registered: route_classifier@v1
  entry_hash: sha256:f8c6c2abe6a9549608644...
  RouteClassifier accuracy on signals: 100.0%

────────────────────────────────────────────────────────────────
  Step 4 — Route 9 test tickets through composite pipeline (text → category)
────────────────────────────────────────────────────────────────
  Tickets routed:            9
  Pipeline accuracy:         100.0%
  Baseline (majority class): 33.3%
  Improvement:               +66.7%
  Composite hash: sha256:1a31cd2c438a01ff38762...
  (covers both fe_entry_hash and rc_entry_hash)

  [+] TKT-0001  BILLING     conf=0.62  [cancel my plan need a refund for the amount char]
  [+] TKT-0002  BILLING     conf=0.92  [my account was charged wrong amount please check]
  [+] TKT-0003  BILLING     conf=0.95  [please help me fix the charge on my account the ]
  [+] TKT-0004  TECHNICAL   conf=0.88  [app is crashing slow server connection and api e]
  [+] TKT-0005  TECHNICAL   conf=0.88  [connection timeout on server when login the api ]
  [+] TKT-0006  TECHNICAL   conf=0.88  [help me fix the server error the network connect]
  [+] TKT-0007  SALES       conf=0.92  [team wants to upgrade to enterprise plan pricing]
  [+] TKT-0008  SALES       conf=0.92  [enterprise demo and pricing for upgrade and cont]
  [+] TKT-0009  SALES       conf=0.93  [want features of enterprise plan for contract up]

────────────────────────────────────────────────────────────────
  Step 5 — Audit trail: exact components for TKT-0002
────────────────────────────────────────────────────────────────
  Ticket:         TKT-0002
  Text:           "my account was charged wrong amount please check my inv..."
  Decision:       BILLING (confidence=0.9183)
  Ground truth:   BILLING

  Composite hash:  sha256:1a31cd2c438a01ff3876204b2...
  TextEmbedder:    feature_extractor@v1  (sha256:57ac18f47f1286f7b...)
  RouteClassifier: route_classifier@v1   (sha256:f8c6c2abe6a954960...)

  verify('feature_extractor', 'v1') → True  + text embedding intact
  verify('route_classifier', 'v1')  → True  + route classifier intact

────────────────────────────────────────────────────────────────
  Step 6 — Tamper detection: modifying TextEmbedder breaks pipeline
────────────────────────────────────────────────────────────────
  Tamper detected — VerificationError: params.json content hash mismatch for feature_extractor@v1
  + Cryptographic chain caught modification of TextEmbedder
  + Restored: the entry verifies again

────────────────────────────────────────────────────────────────
  Summary
────────────────────────────────────────────────────────────────
  Registry entries: 2
    feature_extractor@v1  entry_hash=sha256:57ac18f47f1286f7b...
    route_classifier@v1  entry_hash=sha256:f8c6c2abe6a954960...

  Value delivered:
    + Raw ticket TEXT is the input — not hardcoded numeric features
    + 30-word vocabulary maps ticket words to a BoW vector
    + TextEmbedder (Dense 8x1) learns which words signal each routing queue
    + Composite pipeline routes text with accuracy 100.0% vs 33.3% baseline
    + Every routing decision traceable to exact TextEmbedder + RouteClassifier entry_hash
    + Upgrading either component changes composite_hash automatically
    + Tampering with any component is cryptographically detected

  Business value: 100.0% of tickets correctly routed without human triaging.
  At 1000 tickets/day: 1000 auto-routed, 0 need manual review.
```

> **The hashes above will NOT match yours, and that is correct.**
> Measured on 2026-08-20 over three consecutive runs: the only lines that
> change from one run to the next are the **sixteen** carrying a `sha256:`.
> The reason is in each entry's manifest, which stores a wall-clock
> `created_at`; `entry_hash` covers the manifest, so registering the same
> model twice yields two different hashes — and the composite hash, derived
> from both, changes with them.
>
> What **must** match yours, character for character, is everything else:
> the signal for each text (`0.0643`, `0.4597`, `0.8859`), the nine
> decisions with their confidence, the pipeline's 100.0% against the 33.3%
> baseline, and — inside the manifest — `model_hash` and
> `params_content_hash`, which are CONTENT hashes and carry no timestamp.
> Training is deterministic: if those figures come out different, that is
> worth looking into.


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
    train.csv                  — 36 synthetic tickets (12 per category), raw text + 30 bow_* columns
    test.csv                   — 9 held-out tickets (3 per category)
  run_case.py                  — End-to-end demo script
  registry/                    — The example's P21 registry, VERSIONED in the repo
    entries/feature_extractor/v1/   — manifest, params, signature, trace, report
    entries/route_classifier/v1/    — same for the second stage
    registry.json                   — registry index
```

**Careful with `registry/`:** it is committed, but `run_case.py` **wipes it
and rebuilds it** on every execution (`shutil.rmtree` before registering).
Running the case leaves those eighteen files dirty in your tree, and the
diff you see will be mostly `created_at` and hashes. The CSVs under `data/`
are versioned and are **not** regenerated: the code can generate them, but
only takes that path when they are missing, and in the repo they are not.

