#!/usr/bin/env python3
"""PR2-C1 — Credit scoring: regulatory traceability demo.

Run from the matrixAI root:
    python3 examples/credit-scoring/run_case.py

Demonstrates:
  1. Training and registering two model versions (v1.0 and v1.1).
  2. Batch scoring of credit applications with a full decision audit trail.
  3. Regulatory traceability: for any historical decision, recover the exact
     model version, parameter set and entry hash that produced it.
  4. Tamper detection: modifying stored parameters breaks registry.verify().
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
REGISTRY_PATH = HERE / "registry"

# ── helpers ───────────────────────────────────────────────────────────────────

def _separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _train_version(
    version: str,
    learning_rate: float,
    seed: int,
    epochs: int,
    tmp: Path,
) -> Path:
    from matrixai.training import parse_training_text, SupervisedTrainer

    mxtrain = f"""
MODEL credit_scoring.mxai

DATASET CreditApplications
  SOURCE csv("data/train.csv")
  INPUT Application FROM COLUMNS [
    income_score,
    credit_history,
    debt_ratio,
    employment_years,
    loan_amount_ratio
  ]
  TARGET approved: Probability
  SPLIT train=0.8 validation=0.2 seed={seed}
  BATCH size=8 shuffle=true
END

LOSS CreditLoss
  TYPE binary_cross_entropy
  PREDICTION R
  TARGET approved
END

OPTIMIZER CreditOptimizer
  TYPE sgd
  LEARNING_RATE {learning_rate}
  UPDATE W1, b1
END

METRIC Accuracy
  TYPE accuracy
  PREDICTION R
  TARGET approved
END

RUN
  EPOCHS {epochs}
  SAVE_BEST true
END
"""
    spec = parse_training_text(mxtrain)
    trainer = SupervisedTrainer()
    run_dir = tmp / version
    result = trainer.train(spec, output_dir=str(run_dir), base_path=HERE)

    eval_report = {
        "version": version,
        "accuracy": round(result.accuracy, 4),
        "best_epoch": result.best_epoch,
        "best_validation_loss": round(result.best_validation_loss, 6),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "data/train.csv",
        "learning_rate": learning_rate,
        "seed": seed,
    }
    (run_dir / "evaluation_report.json").write_text(json.dumps(eval_report, indent=2))
    return run_dir, result.accuracy


def _predict(program, ps, row: dict) -> float:
    from matrixai.runtime import MatrixAIRuntime
    rt = MatrixAIRuntime()
    features = {
        "income_score": row["income_score"],
        "credit_history": row["credit_history"],
        "debt_ratio": row["debt_ratio"],
        "employment_years": row["employment_years"],
        "loan_amount_ratio": row["loan_amount_ratio"],
    }
    result = rt.run(program, features, ps.runtime_parameters())
    return result["state"]["R"]


def _load_test_rows() -> list[dict]:
    rows = []
    with open(HERE / "data" / "test.csv") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k != "approved" else int(v) for k, v in row.items()})
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from matrixai.parameters import load_parameter_set
    from matrixai.parser.parser import parse_file
    from matrixai.registry import ModelRegistry, VerificationError

    # Fresh registry for each run
    if REGISTRY_PATH.exists():
        shutil.rmtree(REGISTRY_PATH)
    registry = ModelRegistry(REGISTRY_PATH)

    print("MatrixAI — PR2-C1: Credit Scoring Regulatory Traceability")
    print("=" * 60)

    # ── Step 1: Train and register v1.0 ──────────────────────────────────────
    _separator("Step 1 — Train and register credit-scoring v1.0")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir_v1, acc_v1 = _train_version("v1.0", learning_rate=0.5, seed=42, epochs=40, tmp=Path(tmp))
        entry_v1 = registry.push_run_dir(run_dir_v1, "credit-scoring", "v1.0",
                                         interpretability_level="full")
    registry.tag("credit-scoring", "v1.0", "production")
    print(f"  Registered: credit-scoring@v1.0")
    print(f"  entry_hash: {entry_v1.entry_hash[:24]}...")
    print(f"  parameter_set_id: {entry_v1.parameter_set_id}")
    print(f"  Training accuracy: {acc_v1:.1%}")

    # ── Step 2: Train and register v1.1 ──────────────────────────────────────
    _separator("Step 2 — Improved model: train and register v1.1")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir_v2, acc_v2 = _train_version("v1.1", learning_rate=0.3, seed=99, epochs=50, tmp=Path(tmp))
        entry_v2 = registry.push_run_dir(run_dir_v2, "credit-scoring", "v1.1",
                                         interpretability_level="full")
    registry.tag("credit-scoring", "v1.1", "production")
    print(f"  Registered: credit-scoring@v1.1")
    print(f"  entry_hash: {entry_v2.entry_hash[:24]}...")
    print(f"  parameter_set_id: {entry_v2.parameter_set_id}")
    print(f"  Training accuracy: {acc_v2:.1%}")

    # ── Step 3: Score test applications and build decision audit log ──────────
    _separator("Step 3 — Score 30 test applications (decision audit log)")

    program = parse_file(HERE / "credit_scoring.mxai")
    test_rows = _load_test_rows()
    decision_log: list[dict] = []

    for version, entry in [("v1.0", entry_v1), ("v1.1", entry_v2)]:
        ps_path = REGISTRY_PATH / "entries" / "credit-scoring" / version / "params.json"
        if not ps_path.exists():
            continue
        ps = load_parameter_set(ps_path)
        for i, row in enumerate(test_rows):
            score = _predict(program, ps, row)
            decision = "APPROVED" if score > 0.5 else "REJECTED"
            decision_log.append({
                "application_id": f"APP-{i+1:04d}",
                "model_version": version,
                "parameter_set_id": entry.parameter_set_id,
                "entry_hash": entry.entry_hash,
                "score": round(score, 4),
                "decision": decision,
                "ground_truth": "APPROVED" if row["approved"] == 1 else "REJECTED",
                "decided_at": datetime.now(timezone.utc).isoformat(),
            })

    correct = sum(1 for d in decision_log if d["model_version"] == "v1.1"
                  and d["decision"] == d["ground_truth"])
    total_v11 = sum(1 for d in decision_log if d["model_version"] == "v1.1")
    baseline_acc = sum(1 for r in test_rows if r["approved"] == 1) / len(test_rows)

    print(f"  Applications scored by v1.1: {total_v11}")
    print(f"  Model accuracy on test set:  {correct/total_v11:.1%}")
    print(f"  Baseline (approve all):      {baseline_acc:.1%}")
    print(f"  Improvement over baseline:   +{(correct/total_v11 - baseline_acc):.1%}")

    # ── Step 4: Regulatory audit trail ───────────────────────────────────────
    _separator("Step 4 — Regulatory audit: retrieve exact model for APP-0007")

    audit_target = next(d for d in decision_log
                        if d["application_id"] == "APP-0007" and d["model_version"] == "v1.1")
    print(f"  Application:      {audit_target['application_id']}")
    print(f"  Decision:         {audit_target['decision']} (score={audit_target['score']})")
    print(f"  Model version:    {audit_target['model_version']}")
    print(f"  Parameter set:    {audit_target['parameter_set_id']}")
    print(f"  Entry hash:       {audit_target['entry_hash'][:32]}...")
    print(f"  Decided at:       {audit_target['decided_at']}")

    print(f"\n  Verifying registry entry integrity...")
    ok = registry.verify("credit-scoring", "v1.1")
    print(f"  verify('credit-scoring', 'v1.1') → {ok}  ✓ model and parameters intact")

    # ── Step 5: Tamper detection ──────────────────────────────────────────────
    _separator("Step 5 — Tamper detection: modifying params breaks verification")

    params_path = REGISTRY_PATH / "entries" / "credit-scoring" / "v1.1" / "params.json"
    original = json.loads(params_path.read_text())
    tampered = dict(original)
    if "parameters" in tampered and "W1" in tampered["parameters"]:
        w1 = tampered["parameters"]["W1"]
        if isinstance(w1, dict) and "values" in w1:
            vals = list(w1["values"])
            vals[0] = 9.999
            tampered["parameters"]["W1"]["values"] = vals
    params_path.write_text(json.dumps(tampered))

    try:
        registry.verify("credit-scoring", "v1.1")
        print("  ERROR: tamper was not detected!")
    except VerificationError as e:
        print(f"  Tamper detected — VerificationError: {e}")
        print("  ✓ Cryptographic chain caught the modification")

    # Restore
    params_path.write_text(json.dumps(original))

    # ── Summary ───────────────────────────────────────────────────────────────
    _separator("Summary")
    all_entries = registry.list()
    print(f"  Registry entries: {len(all_entries)}")
    for e in all_entries:
        print(f"    {e.name}@{e.version}  accuracy={e.metrics.get('accuracy','n/a')}")
    print()
    print("  Value delivered:")
    print("    For any historical credit decision, MatrixAI can prove:")
    print("    — which exact model version (entry_hash) produced the decision,")
    print("    — that neither the model nor its parameters were altered,")
    print("    — and any retroactive modification is cryptographically detected.")
    print()
    print("  Metric of value: 100% of decisions are traceable and tamper-evident.")
    print("  This fulfils a regulatory auditability requirement that otherwise")
    print("  requires manual paperwork per decision.")
    print()


if __name__ == "__main__":
    main()
