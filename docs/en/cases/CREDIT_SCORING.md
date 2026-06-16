# Case 1 — Credit Scoring: Regulatory Traceability

> **Español:** [docs/es/cases/CREDIT_SCORING.md](../../es/cases/CREDIT_SCORING.md)

**Industry:** Financial / Microlending  
**Difficulty:** Beginner  
**Runtime:** ~15 seconds

---

## The problem

A microlending company must be able to prove — when audited by a regulator — exactly which model version approved or rejected each loan application, with cryptographic evidence that the record was not altered retroactively.

Today this is typically solved with spreadsheets, screenshots, and manual version notes. When an auditor asks "which model decided this application from six months ago?", the answer takes days of manual reconstruction and is impossible to verify.

The cost is real: regulatory fines for traceability failures, expensive manual audit preparation, and reputational risk from decisions that cannot be explained or attributed.

---

## The solution

MatrixAI keeps an immutable, cryptographically signed registry of every model version ever deployed. Each entry has:

- a deterministic `entry_hash` that covers the model weights, training trace, and evaluation report,
- a `parameter_set_id` that links each prediction to the exact parameters used,
- tamper detection: modifying any stored file breaks verification.

For every credit decision, you record which `entry_hash` produced it. Six months later, an auditor can verify that the model at that hash is byte-for-byte identical to what was in production — and that nobody changed it.

---

## Run it yourself

From the `matrixAI` root directory:

```bash
python3 examples/credit-scoring/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/credit-scoring/run_case.py
```

No external dependencies. No API keys. Dataset included.

### Expected output

```
MatrixAI — PR2-C1: Credit Scoring Regulatory Traceability
============================================================

── Step 1 — Train and register credit-scoring v1.0 ──
  Registered: credit-scoring@v1.0
  entry_hash: sha256:dfbfeffa978100669...
  Training accuracy: 87.5%

── Step 2 — Improved model: train and register v1.1 ──
  Registered: credit-scoring@v1.1
  entry_hash: sha256:ab21a5cb235f4e090...
  Training accuracy: 87.5%

── Step 3 — Score 30 test applications ──
  Model accuracy on test set:  93.3%
  Baseline (approve all):      73.3%
  Improvement over baseline:   +20.0%

── Step 4 — Regulatory audit: retrieve exact model for APP-0007 ──
  Decision: APPROVED (score=0.8098)
  Entry hash: sha256:ab21a5cb235f4e090...
  verify('credit-scoring', 'v1.1') → True  ✓ model and parameters intact

── Step 5 — Tamper detection ──
  Tamper detected — VerificationError: params.json content hash mismatch
  ✓ Cryptographic chain caught the modification
```

---

## The result

### Model metric

| Method | Test accuracy |
|---|---|
| Baseline (approve all) | 73.3% |
| MatrixAI credit-scoring v1.1 | **93.3%** |
| Improvement | **+20 percentage points** |

### Value metric

**100% of credit decisions are traceable and tamper-evident.**

For any historical decision:
- The exact model version (by `entry_hash`) that produced it is recoverable.
- Cryptographic verification proves neither model weights nor parameters were altered after the fact.
- Any retroactive modification is detected in milliseconds.

This directly satisfies the auditability requirement that regulators impose on automated credit decisions — a requirement that, without this system, costs days of manual reconstruction per audit.

---

## Limits

This case illustrates **technical traceability capability**, not regulatory certification.

- The dataset is synthetic. Real accuracy depends on the quality of your credit application data.
- MatrixAI provides the cryptographic audit trail; it does not constitute legal compliance or replace regulatory review in your jurisdiction.
- The model is a logistic regression (linear) over 5 features. For complex credit portfolios, a richer feature set and more expressive model may be needed.
- The registry is local-first. For production, you would operate it on infrastructure with access controls, backups, and retention policies — that operational layer is the paid tier.

---

## What is free and what is paid

| Layer | Status |
|---|---|
| Credit scoring model (training, inference, evaluation) | **Core — free** |
| Local registry with `entry_hash` chain and tamper detection | **Core — free** |
| Decision audit log with `entry_hash` per prediction | **Core — free** |
| Registry-as-a-service (managed, with retention and access control) | Paid tier |
| Audit support for regulators (report generation, expert guidance) | Paid tier |
| Integration with existing loan origination systems | Paid tier |

The value the free layer provides is real: any developer can build and verify a fully traceable credit scoring pipeline today, at zero cost. The paid layer is what makes that pipeline production-grade in a regulated environment.

---

## Files

```
examples/credit-scoring/
  credit_scoring.mxai       — model definition (5 features, sigmoid, binary classification)
  credit_scoring.mxtrain    — training specification
  data/
    train.csv               — 120 synthetic credit applications
    test.csv                — 30 held-out applications
  run_case.py               — end-to-end demo script
```
