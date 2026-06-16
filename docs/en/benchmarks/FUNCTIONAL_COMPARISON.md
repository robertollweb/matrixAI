# C4 — Functional Comparison: What MatrixAI Includes vs What You Build

> [← Back to benchmarks index](INDEX.md) · **Español:** [docs/es/benchmarks/FUNCTIONAL_COMPARISON.md](../../es/benchmarks/FUNCTIONAL_COMPARISON.md)

This is not a speed benchmark. It answers a different question: **what would it cost to build the same audit and governance layer yourself?**

---

## The table

| Capability | MatrixAI | Traditional stack (FastAPI + sklearn + custom) |
|---|---|---|
| Typed model IR (`.mxai`) | Included | Write your own schema |
| Signed `entry_hash` (model fingerprint) | Included | Write your own hashing |
| Training trace (dataset hash + split + epochs) | Included | Write your own logging |
| ParameterSet (versioned, inspectable, diff-able) | Included | joblib pickle |
| Registry with tamper detection | Included | MLflow or custom |
| Per-request execution trace | Included | Write your own |
| Dry-run before real actions | Included | Write your own |
| HMAC-signed ActionTrace | Included | Write your own |
| Rollback contract | Included | Write your own |
| OpenAPI / Swagger auto-generated | Included | FastAPI (explicit routes) |
| Bearer auth | Included | Write your own middleware |

---

## LOC estimate to match the audit layer

| Component | Estimated LOC |
|---|---|
| ActionTrace schema + serialization | ~40 |
| HMAC signing + verification | ~30 |
| Dry-run simulation (scope, rate limit, type check, rollback) | ~80 |
| Audit log persistence (DB schema + ORM) | ~60 |
| Tamper detection (hash + manifest) | ~50 |
| Contract schema + validation | ~60 |
| ParameterSet versioning + diff | ~50 |
| **Total** | **~370 LOC you write, test, and maintain** |

MatrixAI delivers all of this through three calls: `push_run_dir()`, `build_action_trace()`, `verify_action_trace()`.

---

## What MatrixAI does NOT replace

This table is required. MatrixAI has real gaps:

| Dimension | Traditional stack | MatrixAI |
|---|---|---|
| Algorithm variety | sklearn: 100+ estimators, 20+ transformers | ~10 model patterns (linear, dense, composite) |
| Training speed (simple models) | LBFGS / liblinear (C/Fortran) | SGD in Python — ~4x slower (dense network) to 25x+ (small tabular, LBFGS variability) |
| Ecosystem integrations | Extensive (MLflow, DVC, Airflow, Ray, etc.) | Minimal |
| Community size | sklearn: 10+ years, millions of users | Early-stage |
| GPU-scale deep learning | PyTorch / TensorFlow | Not competitive |
| Arbitrary Python in models | Full Python, any library | Declarative IR only — no arbitrary code |
| Interactive exploration | Jupyter notebooks, rich visualization | CLI-only today |

---

## The honest conclusion

MatrixAI trades algorithm variety and ecosystem breadth for a built-in audit layer that regulated environments would otherwise build themselves.

**For whom the trade-off is worth it:**  
Industries where you must prove — cryptographically, reproducibly — what model version produced which decision, and that it was not altered. Financial services, healthcare, operations systems. In those contexts, the ~370 LOC audit infrastructure is not optional: it is the compliance requirement.

**For whom it is not worth it:**  
Research environments, data science exploration, teams that need sklearn's full estimator library, projects where model interpretability and governance are not regulatory requirements.

---

## Conditions

- LOC estimates: based on a FastAPI + SQLAlchemy + custom HMAC implementation that matches MatrixAI's audit layer feature by feature
- Estimates are order-of-magnitude. Actual implementation varies by team conventions and framework choices.
