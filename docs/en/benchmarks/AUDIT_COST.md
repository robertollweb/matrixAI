# C3 — Audit Cost: What the Audit Layer Costs

> [← Back to benchmarks index](INDEX.md) · **Español:** [docs/es/benchmarks/AUDIT_COST.md](../../es/benchmarks/AUDIT_COST.md)

**Reproduce:**
```bash
python3 benchmarks/audit_cost.py
```

**Environment:** Python 3.12.3 · x86_64 CPU-only · 200 repetitions per measurement

---

## Results

| Operation | Median | p95 | p99 | vs plain inference |
|---|---|---|---|---|
| Plain inference (no audit) | 0.005 ms | 0.009 ms | 0.030 ms | baseline |
| HMAC-SHA256 signing (`build_action_trace`) | 0.003 ms | 0.004 ms | 0.018 ms | ~60% of inference |
| Dry-run (`DryRunSimulator.simulate`) | 0.014 ms | 0.021 ms | 0.057 ms | ~280% of inference |
| Tamper detection (`registry.verify`) | 0.171 ms | 0.275 ms | 0.387 ms | ~3420% of inference |
| Versioning (`registry.push_run_dir`) | 0.991 ms | 2.015 ms | 2.015 ms | once per training |

---

## What each operation does

**Plain inference:** `MatrixAIRuntime.run()` on a 5-feature sigmoid model. Walks the computation graph, applies sigmoid, returns result. Sub-millisecond.

**HMAC signing:** Builds a canonical dict of all `ActionTrace` fields (report_id, model_hash, parameter_set_id, contract_hash, input_hash, executed_at, executor_kind, ok, response_summary, latency_ms), serializes to bytes, computes HMAC-SHA256. Result: cryptographic proof of what ran, when, and with which model version.

**Dry-run:** In-memory simulation before any real action. Validates: recipient in `allowed_recipients`, call count within `rate_limit`, input types match contract schema, rollback contract is declared. All four checks complete in 0.014 ms median.

**Tamper detection:** `registry.verify()` reads `params.json` from disk, computes its SHA256, and compares against the stored `params_content_hash` in the manifest. Filesystem I/O is the dominant cost. Typical range on local storage: 0.15–0.30 ms median (hardware-dependent; run the script for your machine's number).

**Versioning (`push_run_dir`):** Copies model snapshot + params + evaluation report to the registry entry directory, computes SHA256 of params, writes the manifest. One-time cost per training run.

---

## Full audited action cost

A single action with full audit (dry-run + execute + sign) costs:

```
dry-run:    0.014 ms
HMAC sign:  0.003 ms
────────────────────
total:      ~0.017 ms per audited action
```

For **1,000 actions/day**, the total audit overhead is ~17 ms/day — sub-second.

---

## What you get for 0.017 ms

Every audited action produces an `ActionTrace` that records:

- `report_id` — unique identifier for this execution
- `model_hash` — exact model version that produced the decision
- `parameter_set_id` — exact parameter version
- `action_contract_hash` — hash of the declared contract (scope, rate limits, rollback)
- `input_hash` — hash of the input that triggered the action
- `executed_at` — timestamp
- `hmac_signature` — HMAC-SHA256 over all of the above

Any modification of the trace — even flipping one bit — fails `verify_action_trace()`. Reproducible months or years later.

---

## The build-it-yourself comparison

To match this audit capability with a traditional stack (FastAPI + SQLAlchemy + custom HMAC):

| Component | Estimated LOC |
|---|---|
| ActionTrace schema + serialization | ~40 |
| HMAC signing + verification | ~30 |
| Dry-run simulation (scope, rate limit, type check, rollback) | ~80 |
| Audit log persistence (DB schema + ORM) | ~60 |
| Tamper detection (hash + manifest) | ~50 |
| Contract schema + validation | ~60 |
| Total | **~320 LOC** you write, test, and maintain |

MatrixAI provides all of this in one `push_run_dir()` + `build_action_trace()` + `verify_action_trace()`.

---

## Conditions

- Hardware: x86_64 CPU, no GPU
- Python: 3.12.3
- Repetitions: 200 per measurement
- Model: `clinical_risk` (5 features, sigmoid)
- Times reported: median over 200 runs (not single runs)
