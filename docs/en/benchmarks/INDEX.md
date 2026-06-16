# MatrixAI — Benchmarks

> **Español:** [docs/es/benchmarks/INDEX.md](../../es/benchmarks/INDEX.md)

These benchmarks measure where MatrixAI wins, where it loses, and what its audit layer costs — with scripts you can run yourself to verify every number.

**Reproduce any benchmark:**
```bash
python3 benchmarks/training.py       # C2: training speed
python3 benchmarks/audit_cost.py     # C3: audit overhead
python3 benchmarks/serving.py        # C1: HTTP throughput
```

**Environment:** Python 3.12.3 · scikit-learn 1.8.0 · x86_64 · CPU-only (no GPU)  
Numbers depend on hardware. Run on your own machine for comparable results.

---

## Benchmarks

| # | What is measured | Key finding |
|---|---|---|
| [C1 — Serving](SERVING.md) | HTTP throughput and latency | At concurrency=1: roughly comparable (within run noise). At concurrency≥10: clearly faster (baseline degrades to p99 > 1000ms under concurrent load; magnitude varies by run; root cause not instrumented) |
| [C2 — Training](TRAINING.md) | Training time vs scikit-learn | ~4x slower (dense network) to 25x or more (small dataset, high LBFGS variability). sklearn uses optimized C/Fortran solvers. MatrixAI adds typed IR, traces, and registry. |
| [C3 — Audit cost](AUDIT_COST.md) | Cost of each audit primitive | HMAC sign: 0.003 ms. Dry-run: 0.014 ms. Tamper detect: 0.15–0.30 ms (filesystem-dependent). All sub-millisecond. |
| [C4 — Functional comparison](FUNCTIONAL_COMPARISON.md) | What MatrixAI includes vs what you build | ~320 LOC for the audit layer alone; ~370 LOC for the full stack including ParameterSet versioning. |

---

## The honest summary

MatrixAI is **slower to train** and **slightly slower per request at low concurrency** than optimized traditional alternatives. This is not a bug — it is the cost of the audit layer.

What that overhead buys:

| Capability | MatrixAI | sklearn+FastAPI |
|---|---|---|
| Typed model IR (`.mxai`) | Included | Write your own schema |
| Signed `entry_hash` (model fingerprint) | Included | Write your own hashing |
| Training trace (dataset hash + split + epochs) | Included | Write your own logging |
| ParameterSet (versioned, inspectable) | Included | Write your own versioning |
| Registry with tamper detection | Included | MLflow or custom |
| Per-request execution trace | Included | Write your own |
| Dry-run before real actions | Included | Write your own |
| HMAC-signed ActionTrace | Included | Write your own |
| Rollback contract | Included | Write your own |
| OpenAPI / Swagger auto-generated | Included | FastAPI (explicit) |
| Bearer auth | Included | Write your own |

**Where MatrixAI loses** (declared explicitly, as required by PR3):
- Training speed: approximately 4x slower on dense networks, 10–25x or more on small tabular datasets (C/Fortran solvers vs Python SGD; exact ratio varies by run due to LBFGS convergence variability at small N)
- Single-request serving: roughly comparable at c=1 (within run noise); the per-request overhead of auth + trace + schema is real but small relative to network latency
- Algorithm variety: MatrixAI has ~10 model patterns; sklearn has 100+ estimators
- Ecosystem maturity: sklearn/FastAPI have larger communities, more integrations, more tutorials
- GPU-scale training: MatrixAI is not competitive with PyTorch for large-scale deep learning

**For whom the trade-off is worth it:**  
Regulated industries (financial, healthcare, operations) where you must prove — cryptographically, reproducibly — what model version produced which decision, and that it was not altered. The audit layer is not overhead in that context: it is the regulatory requirement.
