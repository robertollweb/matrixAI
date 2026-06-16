# C1 — Serving Benchmark: MatrixAI HTTP vs ThreadingHTTPServer + sklearn

> [← Back to benchmarks index](INDEX.md) · **Español:** [docs/es/benchmarks/SERVING.md](../../es/benchmarks/SERVING.md)

**Reproduce:**
```bash
python3 benchmarks/serving.py
```

**Environment:** Python 3.12.3 · sklearn 1.8.0 · x86_64 · 15.6 GB RAM · 100 requests per concurrency level

---

## Results

| Concurrency | MatrixAI req/s | Baseline req/s | Winner | Ratio | MatrixAI p50 |
|---|---|---|---|---|---|
| 1 | 1,013 | 1,114 | ~comparable | within run noise | 0.7 ms |
| 5 | 1,989 | 1,829 | MatrixAI | **1.1x faster** | 2.5 ms |
| 10 | 3,065 | 666 | MatrixAI | **clearly faster** (varies by run) | 3.2 ms |
| 20 | 5,247 | 807 | MatrixAI | **clearly faster** (varies by run) | 3.5 ms |

Numbers are wall-time averages over 100 requests per level with 5-request warmup. Run on your own machine — absolute numbers vary by hardware.

---

## The baseline is a competent rival

This benchmark uses a **ThreadingHTTPServer + sklearn LogisticRegression** as the baseline — not a single-threaded stub. The baseline:

- Uses the same threading model as MatrixAI (`ThreadingHTTPServer`)
- Trains a real `LogisticRegression(LBFGS)` on the credit-scoring dataset at startup
- Serves real predictions via `predict_proba()`, not hardcoded weights

What it does **not** have:
- No Bearer token authentication
- No input type validation or schema
- No per-request execution trace or model hash
- No versioning, registry, or tamper detection
- No OpenAPI / Swagger documentation

At c=1 both servers are within run-to-run noise of each other. The per-request overhead of auth + trace + schema is real but small relative to network round-trip at low concurrency.

---

## What the numbers mean

**c=1 (roughly comparable):** Both servers are within run-to-run noise. Each MatrixAI request pays for auth, input validation, trace recording, and model hash — the baseline skips all of this. At single-request concurrency that overhead is small relative to the network round-trip cost.

**c=5 (MatrixAI 1.1x faster):** MatrixAI begins to pull ahead as its thread pool amortizes the per-request overhead across parallel workers.

**c=10 and c=20 (MatrixAI clearly faster):** The baseline's p99 spikes to ~1,000 ms under concurrent load, while MatrixAI maintains consistent latency (p99 ≤ 10 ms at c=20). The observed degradation in the baseline — throughput dropping from 1,829 req/s at c=5 to 666 req/s at c=10 — suggests thread-level serialization under concurrent Python workloads. The magnitude of the advantage varies by run and hardware; the script does not instrument the root cause.

---

## What you get for MatrixAI's request overhead

Every MatrixAI request:

1. Validates the Bearer token (O(1) string compare)
2. Parses and validates input against the typed model schema
3. Executes the computation graph through `MatrixAIRuntime.run()`
4. Records an execution trace (model hash, input hash, timestamp, latency)
5. Returns a typed response with the prediction

The baseline does step 3 only — with a sklearn model, not a versioned, registry-linked artifact.

---

## Conditions

- Hardware: x86_64 CPU, no GPU, 15.6 GB RAM
- Python: 3.12.3
- sklearn: 1.8.0
- Model: `credit_scoring` (5 features, sigmoid, stdlib backend)
- Baseline: `ThreadingHTTPServer` + `sklearn.linear_model.LogisticRegression(solver=lbfgs)` trained on the same dataset
- Concurrency levels: 1, 5, 10, 20
- Requests per level: 100 (with 5-request warmup)
