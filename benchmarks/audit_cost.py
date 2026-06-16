#!/usr/bin/env python3
"""PR3-C3 — Audit cost benchmark: overhead of MatrixAI's audit layer.

Measures the exact cost (in ms) of each audit primitive:
  - Plain inference (no audit)
  - registry.verify() — tamper detection
  - build_action_trace() + HMAC signing
  - DryRunSimulator.simulate()
  - registry.push_run_dir() — versioning + hash computation

Run from the matrixAI root:
    python3 benchmarks/audit_cost.py
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import timeit
from pathlib import Path

ROOT = Path(__file__).parent.parent
N_REPS = 200   # repetitions for stable median


def _sep(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print("─" * 64)


def _stats(times_s: list[float]) -> dict:
    import statistics
    s = sorted(times_s)
    return {
        "median_ms": round(statistics.median(s) * 1000, 3),
        "p95_ms":    round(s[int(len(s) * 0.95)] * 1000, 3),
        "p99_ms":    round(s[int(len(s) * 0.99)] * 1000, 3),
        "min_ms":    round(s[0] * 1000, 3),
    }


# ── Setup: train and register a model once ───────────────────────────────────

def _setup():
    """Returns (program, params, registry, entry, contract, action_program)."""
    from matrixai.training import parse_training_text, SupervisedTrainer
    from matrixai.parameters import load_parameter_set
    from matrixai.parser.parser import parse_file
    from matrixai.registry import ModelRegistry

    mxtrain = """MODEL clinical_risk.mxai

DATASET PatientData
  SOURCE csv("data/train.csv")
  INPUT Patient FROM COLUMNS [
    age, mobility, medication_load, previous_falls, cognitive_state
  ]
  TARGET risk_probability: Probability
  SPLIT train=0.8 validation=0.2 seed=7
  BATCH size=8 shuffle=true
END

LOSS ClinicalLoss
  TYPE binary_cross_entropy
  PREDICTION R
  TARGET risk_probability
END

OPTIMIZER ClinicalOpt
  TYPE sgd
  LEARNING_RATE 0.8
  UPDATE W1, b1
END

METRIC Accuracy
  TYPE accuracy
  PREDICTION R
  TARGET risk_probability
END

RUN
  EPOCHS 50
  SAVE_BEST true
END
"""
    spec = parse_training_text(mxtrain)
    tmp_dir = Path(tempfile.mkdtemp())
    run_dir = tmp_dir / "run"
    SupervisedTrainer().train(
        spec, output_dir=str(run_dir),
        base_path=ROOT / "examples" / "clinical-risk"
    )
    shutil.copy(ROOT / "examples/clinical-risk/clinical_risk.mxai",
                run_dir / "model_snapshot.mxai")

    import json as _json
    from datetime import datetime, timezone
    (run_dir / "evaluation_report.json").write_text(_json.dumps({
        "accuracy": 1.0, "best_epoch": 50,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }))
    registry_path = tmp_dir / "registry"
    registry = ModelRegistry(registry_path)
    entry = registry.push_run_dir(run_dir, "clinical-risk", "v1.0")

    program = parse_file(ROOT / "examples/clinical-risk/clinical_risk.mxai")
    ps_path = registry_path / "entries" / "clinical-risk" / "v1.0" / "params.json"
    ps = load_parameter_set(ps_path)

    # Load action contract for dry-run / execution tests
    from matrixai.actions import parse_mxact
    from matrixai.parser.parser import parse_file as pf
    contract_src = (ROOT / "examples/agent-alert/alert_notifier.mxact").read_text()
    contracts = parse_mxact(contract_src)
    contract = contracts[0]
    action_program = pf(str(ROOT / "examples/agent-alert/alert_monitor.mxai"))

    return program, ps, registry, entry, contract, action_program, tmp_dir


# ── Benchmark 1: Plain inference ─────────────────────────────────────────────

def bench_inference(program, ps):
    from matrixai.runtime import MatrixAIRuntime
    rt = MatrixAIRuntime()
    params = ps.runtime_parameters()
    features = {
        "age": 0.88, "mobility": 0.82, "medication_load": 0.78,
        "previous_falls": 1.00, "cognitive_state": 0.76
    }
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        rt.run(program, features, params)
        times.append(time.perf_counter() - t0)
    return _stats(times)


# ── Benchmark 2: registry.verify() ───────────────────────────────────────────

def bench_verify(registry):
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        registry.verify("clinical-risk", "v1.0")
        times.append(time.perf_counter() - t0)
    return _stats(times)


# ── Benchmark 3: HMAC signing (build_action_trace) ───────────────────────────

def bench_hmac_sign(entry):
    from matrixai.actions import (
        ActionExecutor, ExecutionContext, DryRunSimulator, build_action_trace
    )
    from matrixai.actions import parse_mxact
    from matrixai.parser.parser import parse_file

    SIGNING_KEY = "cafebabe" * 8
    contract_src = (ROOT / "examples/agent-alert/alert_notifier.mxact").read_text()
    contracts = parse_mxact(contract_src)
    contract = contracts[0]
    action_program = parse_file(str(ROOT / "examples/agent-alert/alert_monitor.mxai"))

    action_input = {
        "recipient": "ops@example.com",
        "subject": "[CRIT] servidor-db-01: CPU al 95%",
        "body": "Evento critico detectado.\nScore: 0.616\nHash: abc123...\n",
    }

    def _mock_email(*a, **kw):
        return "250 OK"

    dry_run = DryRunSimulator().simulate(
        contract, action_program, entry.parameter_set_id, entry.entry_hash, action_input
    )
    ctx = ExecutionContext(
        contract=contract, dry_run_report=dry_run, input_data=action_input,
        model_hash=entry.entry_hash, parameter_set_id=entry.parameter_set_id,
        allow_real_actions=True, signing_key=SIGNING_KEY,
    )
    result = ActionExecutor(email_fn=_mock_email).execute(ctx)

    # Now benchmark only the trace + sign step
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        build_action_trace(ctx, result, signing_key=SIGNING_KEY)
        times.append(time.perf_counter() - t0)
    return _stats(times)


# ── Benchmark 4: DryRunSimulator.simulate() ───────────────────────────────────

def bench_dry_run(entry, contract, action_program):
    from matrixai.actions import DryRunSimulator

    action_input = {
        "recipient": "ops@example.com",
        "subject": "[CRIT] servidor-db-01",
        "body": "Evento critico.\n",
    }
    sim = DryRunSimulator()
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        sim.simulate(contract, action_program, entry.parameter_set_id,
                     entry.entry_hash, action_input)
        times.append(time.perf_counter() - t0)
    return _stats(times)


# ── Benchmark 5: registry.push_run_dir() (versioning) ────────────────────────

def bench_push_run_dir(entry_run_dir: Path):
    from matrixai.registry import ModelRegistry

    times = []
    for i in range(20):   # fewer reps — writes to disk
        tmp = Path(tempfile.mkdtemp())
        registry = ModelRegistry(tmp / "reg")
        # copy a fresh run_dir each time
        rd = tmp / "run"
        shutil.copytree(entry_run_dir, rd)
        t0 = time.perf_counter()
        registry.push_run_dir(rd, "bench-model", f"v{i}")
        times.append(time.perf_counter() - t0)
        shutil.rmtree(tmp)
    return _stats(times)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("PR3-C3 — Audit Cost Benchmark")
    print("=" * 64)
    print(f"  Repetitions per measurement: {N_REPS}")
    print(f"  Model: clinical_risk (5 features, sigmoid)")
    print()
    print("  Setting up: training model and registering...")
    program, ps, registry, entry, contract, action_program, tmp_dir = _setup()
    entry_run_dir = (
        tmp_dir / "registry" / "entries" / "clinical-risk" / "v1.0"
    )
    print("  Setup complete.")

    results = {}

    # 1. Plain inference
    _sep("1 — Plain inference (no audit)")
    r = bench_inference(program, ps)
    results["inference_no_audit"] = r
    print(f"  median {r['median_ms']:.3f} ms  |  p95 {r['p95_ms']:.3f} ms  |  p99 {r['p99_ms']:.3f} ms")
    print(f"  → Baseline. MatrixAIRuntime.run() on 5-feature sigmoid model.")

    # 2. registry.verify()
    _sep("2 — registry.verify() — tamper detection")
    r = bench_verify(registry)
    results["registry_verify"] = r
    print(f"  median {r['median_ms']:.3f} ms  |  p95 {r['p95_ms']:.3f} ms  |  p99 {r['p99_ms']:.3f} ms")
    print(f"  → Re-hashes params.json, compares against manifest. File I/O + SHA256.")

    # 3. HMAC signing
    _sep("3 — build_action_trace() + HMAC-SHA256 signing")
    r = bench_hmac_sign(entry)
    results["hmac_sign"] = r
    print(f"  median {r['median_ms']:.3f} ms  |  p95 {r['p95_ms']:.3f} ms  |  p99 {r['p99_ms']:.3f} ms")
    print(f"  → Builds canonical trace dict, computes HMAC-SHA256 over all fields.")

    # 4. Dry-run
    _sep("4 — DryRunSimulator.simulate() — pre-action validation")
    r = bench_dry_run(entry, contract, action_program)
    results["dry_run"] = r
    print(f"  median {r['median_ms']:.3f} ms  |  p95 {r['p95_ms']:.3f} ms  |  p99 {r['p99_ms']:.3f} ms")
    print(f"  → Validates scope, rate limit, input types, rollback. In-memory.")

    # 5. push_run_dir (versioning)
    _sep("5 — registry.push_run_dir() — versioning + hash computation")
    r = bench_push_run_dir(entry_run_dir)
    results["push_run_dir"] = r
    print(f"  median {r['median_ms']:.3f} ms  |  p95 {r['p95_ms']:.3f} ms  |  p99 {r['p99_ms']:.3f} ms")
    print(f"  → Copies artifacts, computes SHA256 of params, writes manifest. Disk I/O.")

    # Summary with context
    _sep("Summary — overhead with context")
    inf = results["inference_no_audit"]["median_ms"]
    print(f"  {'Operation':<35}  {'Median':>9}  {'vs inference':>13}")
    print("  " + "─" * 62)
    ops = [
        ("Plain inference (no audit)",    "inference_no_audit", "baseline"),
        ("+ HMAC sign (per execution)",   "hmac_sign",          None),
        ("+ Dry-run (per execution)",     "dry_run",            None),
        ("registry.verify() (on demand)", "registry_verify",    None),
        ("registry.push_run_dir() (once per training)", "push_run_dir", None),
    ]
    for label, key, note in ops:
        m = results[key]["median_ms"]
        if note == "baseline":
            print(f"  {label:<35}  {m:>8.3f}ms  {'(baseline)':>13}")
        else:
            overhead_pct = (m / inf) * 100
            print(f"  {label:<35}  {m:>8.3f}ms  {overhead_pct:>12.0f}%")

    print()
    print("  Interpretation for regulated environments:")
    dry = results["dry_run"]["median_ms"]
    hmac = results["hmac_sign"]["median_ms"]
    total_per_action = dry + hmac
    print(f"    An audited action (dry-run + sign) costs ~{total_per_action:.3f} ms total.")
    print(f"    That's {total_per_action:.3f} ms for: scope validation, rate limit check,")
    print(f"    input type validation, rollback check, HMAC-SHA256 signing,")
    print(f"    and a verifiable ActionTrace that proves what ran, when, and with")
    print(f"    which model version — reproducible months later.")
    print()
    print("    Building equivalent audit infrastructure from scratch with")
    print("    FastAPI + SQLAlchemy + custom HMAC layer: est. 300-500 LOC,")
    print("    which you write once and maintain forever.")
    print()
    daily_ms = total_per_action * 1000
    print(f"    For a system processing 1000 actions/day, the total audit")
    print(f"    overhead is ~{daily_ms:.0f} ms (~{daily_ms/1000:.2f} s) of wall time per day.")

    # Save results
    out = ROOT / "benchmarks" / "results_audit_cost.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to benchmarks/results_audit_cost.json")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
