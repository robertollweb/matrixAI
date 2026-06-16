#!/usr/bin/env python3
"""PR3-C2 — Training benchmark: MatrixAI vs scikit-learn.

Measures training time (median + worst single run over N_REPS repetitions)
and convergence on the four PR2 pipelines.

Run from the matrixAI root:
    python3 benchmarks/training.py
"""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Pre-import sklearn so import overhead is not included in timing.
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np

ROOT = Path(__file__).parent.parent
N_REPS = 5  # repetitions per pipeline for stable median


# ── environment capture ────────────────────────────────────────────────────────

def _capture_env() -> dict:
    env: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
    except ImportError:
        env["sklearn"] = "not installed"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    env["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    return env


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _load_csv_mixed(path: Path, skip_cols: set) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if k not in skip_cols})
    return rows


def _time_reps(fn, n: int) -> dict:
    """Run fn n times; return median, worst, min wall times and the last result.

    With small N (e.g. 5), int(N*0.95) == N-1, so 'p95' would just be the
    maximum. We report max_s explicitly to avoid implying more precision than
    we have.
    """
    times: list[float] = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    s = sorted(times)
    med = statistics.median(s)
    return {
        "median_s": round(med, 3),
        "max_s": round(s[-1], 3),
        "min_s": round(s[0], 3),
        "reps": n,
        **(result or {}),
    }


def _sep(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print("─" * 64)


# ── Case 1: Credit Scoring ────────────────────────────────────────────────────

def bench_credit_matrixai():
    from matrixai.training import parse_training_text, SupervisedTrainer

    mxtrain = """MODEL credit_scoring.mxai

DATASET CreditData
  SOURCE csv("data/train.csv")
  INPUT Application FROM COLUMNS [
    income_score, credit_history, debt_ratio, employment_years, loan_amount_ratio
  ]
  TARGET approved: Probability
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=16 shuffle=true
END

LOSS CreditLoss
  TYPE binary_cross_entropy
  PREDICTION R
  TARGET approved
END

OPTIMIZER CreditOpt
  TYPE sgd
  LEARNING_RATE 0.5
  UPDATE W1, b1
END

METRIC Accuracy
  TYPE accuracy
  PREDICTION R
  TARGET approved
END

RUN
  EPOCHS 40
  SAVE_BEST true
END
"""
    spec = parse_training_text(mxtrain)
    with tempfile.TemporaryDirectory() as tmp:
        result = SupervisedTrainer().train(
            spec, output_dir=tmp, base_path=ROOT / "examples" / "credit-scoring"
        )
        return {"accuracy": round(result.accuracy, 4), "epochs": result.best_epoch, "train_rows": 120}


def bench_credit_sklearn():
    rows = _load_csv(ROOT / "examples/credit-scoring/data/train.csv")
    feature_cols = ["income_score", "credit_history", "debt_ratio",
                    "employment_years", "loan_amount_ratio"]
    X = np.array([[r[c] for c in feature_cols] for r in rows])
    y = (np.array([r["approved"] for r in rows]) >= 0.5).astype(int)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    accuracy = (clf.predict(X) == y).mean()
    return {"accuracy": round(float(accuracy), 4), "epochs": "N/A (LBFGS)", "train_rows": len(rows)}


# ── Case 2: Clinical Risk ─────────────────────────────────────────────────────

def bench_clinical_matrixai():
    from matrixai.training import parse_training_text, SupervisedTrainer

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
    with tempfile.TemporaryDirectory() as tmp:
        result = SupervisedTrainer().train(
            spec, output_dir=tmp, base_path=ROOT / "examples" / "clinical-risk"
        )
        return {"accuracy": round(result.accuracy, 4), "epochs": result.best_epoch, "train_rows": 60}


def bench_clinical_sklearn():
    rows = _load_csv(ROOT / "examples/clinical-risk/data/train.csv")
    feature_cols = ["age", "mobility", "medication_load", "previous_falls", "cognitive_state"]
    X = np.array([[r[c] for c in feature_cols] for r in rows])
    y = (np.array([r["risk_probability"] for r in rows]) >= 0.5).astype(int)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    accuracy = (clf.predict(X) == y).mean()
    return {"accuracy": round(float(accuracy), 4), "epochs": "N/A (LBFGS)", "train_rows": len(rows)}


# ── Case 3: Agent Alert ───────────────────────────────────────────────────────

def bench_alert_matrixai():
    from matrixai.training import parse_training_text, SupervisedTrainer

    mxtrain = """MODEL alert_model_train.mxai

DATASET AlertData
  SOURCE csv("data/train.csv")
  INPUT SystemMetrics FROM COLUMNS [
    severity, source_trust, is_business_hours
  ]
  TARGET alert_label: Probability
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8 shuffle=true
END

LOSS AlertLoss
  TYPE binary_cross_entropy
  PREDICTION R
  TARGET alert_label
END

OPTIMIZER AlertOpt
  TYPE sgd
  LEARNING_RATE 0.6
  UPDATE W1, b1
END

METRIC Accuracy
  TYPE accuracy
  PREDICTION R
  TARGET alert_label
END

RUN
  EPOCHS 80
  SAVE_BEST true
END
"""
    spec = parse_training_text(mxtrain)
    with tempfile.TemporaryDirectory() as tmp:
        result = SupervisedTrainer().train(
            spec, output_dir=tmp, base_path=ROOT / "examples" / "agent-alert"
        )
        return {"accuracy": round(result.accuracy, 4), "epochs": result.best_epoch, "train_rows": 30}


def bench_alert_sklearn():
    rows = _load_csv(ROOT / "examples/agent-alert/data/train.csv")
    feature_cols = ["severity", "source_trust", "is_business_hours"]
    X = np.array([[r[c] for c in feature_cols] for r in rows])
    y = np.array([r["alert_label"] for r in rows]).astype(int)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    accuracy = (clf.predict(X) == y).mean()
    return {"accuracy": round(float(accuracy), 4), "epochs": "N/A (LBFGS)", "train_rows": len(rows)}


# ── Case 4: Text Routing — TextEmbedder (Dense) ───────────────────────────────

def bench_text_routing_matrixai():
    """Times the TextEmbedder (DenseSupervisedTrainer) — the heavier stage."""
    from matrixai.training.dense_trainer import DenseSupervisedTrainer
    from matrixai.training.parser import parse_training_text

    rows = _load_csv_mixed(ROOT / "examples/text-routing/data/train.csv",
                           skip_cols={"text", "category"})
    bow_cols = sorted([k for k in rows[0].keys() if k.startswith("bow_")])
    bow_cols_str = ", ".join(bow_cols)

    mxtrain = f"""MODEL examples/text-routing/feature_extractor.mxai

DATASET TicketData
  SOURCE csv("examples/text-routing/data/train.csv")
  INPUT TicketBOW FROM COLUMNS [
    {bow_cols_str}
  ]
  TARGET routing_code: Probability
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=10 shuffle=true
END

LOSS EmbedLoss
  TYPE binary_cross_entropy
  PREDICTION routing_signal
  TARGET routing_code
END

OPTIMIZER EmbedOpt
  TYPE sgd
  LEARNING_RATE 0.3
  UPDATE W1, b1
END

RUN
  EPOCHS 60
  SAVE_BEST true
END
"""
    spec = parse_training_text(mxtrain)
    with tempfile.TemporaryDirectory() as tmp:
        result = DenseSupervisedTrainer().train(spec, output_dir=tmp)
    return {"accuracy": round(result.accuracy, 4), "epochs": result.best_epoch, "train_rows": len(rows)}


def bench_text_routing_sklearn():
    """Sklearn MLP equivalent of the 2-layer dense network."""
    rows = _load_csv_mixed(ROOT / "examples/text-routing/data/train.csv",
                           skip_cols={"text", "category"})
    bow_cols = [k for k in rows[0].keys() if k.startswith("bow_")]
    X = np.array([[r[c] for c in bow_cols] for r in rows])
    y_raw = np.array([r["routing_code"] for r in rows])

    def to_label(v):
        if v < 0.3: return "billing"
        if v < 0.7: return "technical"
        return "sales"
    y = np.array([to_label(v) for v in y_raw])

    clf = MLPClassifier(hidden_layer_sizes=(8,), activation="relu",
                        max_iter=200, random_state=42)
    clf.fit(X, y)
    accuracy = (clf.predict(X) == y).mean()
    return {"accuracy": round(float(accuracy), 4), "epochs": "N/A (adam)", "train_rows": len(rows)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    env = _capture_env()
    print("PR3-C2 — Training Benchmark: MatrixAI vs scikit-learn")
    print("=" * 64)
    print(f"  Python:      {env['python']}")
    print(f"  sklearn:     {env.get('sklearn', 'unknown')}")
    print(f"  Platform:    {env.get('machine', 'unknown')} — {env.get('platform', 'unknown')}")
    print(f"  Processor:   {env.get('processor', 'unknown')}")
    if "ram_gb" in env:
        print(f"  RAM:         {env['ram_gb']} GB")
    print(f"  Repetitions: {N_REPS} per pipeline (median reported)")
    print(f"  Note: sklearn pre-imported — import overhead excluded from timing.")
    print()

    results = []

    # ── Credit Scoring ────────────────────────────────────────────────────────
    _sep("Case 1 — Credit Scoring (120 train rows, 5 features, sigmoid+BCE)")
    print(f"  MatrixAI (SupervisedTrainer, SGD, 40 epochs) × {N_REPS} reps...")
    r_mx = _time_reps(bench_credit_matrixai, N_REPS)
    print(f"    median={r_mx['median_s']:.3f}s  max={r_mx['max_s']:.3f}s  "
          f"accuracy={r_mx['accuracy']:.1%}  best_epoch={r_mx['epochs']}")

    print(f"  scikit-learn (LogisticRegression, LBFGS) × {N_REPS} reps...")
    r_sk = _time_reps(bench_credit_sklearn, N_REPS)
    print(f"    median={r_sk['median_s']:.3f}s  max={r_sk['max_s']:.3f}s  "
          f"accuracy={r_sk['accuracy']:.1%}  solver=LBFGS")
    ratio = r_mx["median_s"] / max(r_sk["median_s"], 0.001)
    print(f"  → MatrixAI is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than sklearn (median)")
    results.append(("credit_scoring", r_mx, r_sk))

    # ── Clinical Risk ─────────────────────────────────────────────────────────
    _sep("Case 2 — Clinical Risk (60 train rows, 5 features, sigmoid+BCE)")
    print(f"  MatrixAI (SupervisedTrainer, SGD, 50 epochs) × {N_REPS} reps...")
    r_mx = _time_reps(bench_clinical_matrixai, N_REPS)
    print(f"    median={r_mx['median_s']:.3f}s  max={r_mx['max_s']:.3f}s  "
          f"accuracy={r_mx['accuracy']:.1%}  best_epoch={r_mx['epochs']}")

    print(f"  scikit-learn (LogisticRegression, LBFGS) × {N_REPS} reps...")
    r_sk = _time_reps(bench_clinical_sklearn, N_REPS)
    print(f"    median={r_sk['median_s']:.3f}s  max={r_sk['max_s']:.3f}s  "
          f"accuracy={r_sk['accuracy']:.1%}  solver=LBFGS")
    ratio = r_mx["median_s"] / max(r_sk["median_s"], 0.001)
    print(f"  → MatrixAI is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than sklearn (median)")
    results.append(("clinical_risk", r_mx, r_sk))

    # ── Agent Alert ───────────────────────────────────────────────────────────
    _sep("Case 3 — Agent Alert (30 train rows, 3 features, sigmoid+BCE)")
    print(f"  MatrixAI (SupervisedTrainer, SGD, 80 epochs) × {N_REPS} reps...")
    r_mx = _time_reps(bench_alert_matrixai, N_REPS)
    print(f"    median={r_mx['median_s']:.3f}s  max={r_mx['max_s']:.3f}s  "
          f"accuracy={r_mx['accuracy']:.1%}  best_epoch={r_mx['epochs']}")

    print(f"  scikit-learn (LogisticRegression, LBFGS) × {N_REPS} reps...")
    r_sk = _time_reps(bench_alert_sklearn, N_REPS)
    print(f"    median={r_sk['median_s']:.3f}s  max={r_sk['max_s']:.3f}s  "
          f"accuracy={r_sk['accuracy']:.1%}  solver=LBFGS")
    ratio = r_mx["median_s"] / max(r_sk["median_s"], 0.001)
    print(f"  → MatrixAI is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than sklearn (median)")
    results.append(("agent_alert", r_mx, r_sk))

    # ── Text Routing (Dense) ──────────────────────────────────────────────────
    _sep("Case 4 — Text Routing / TextEmbedder (45 train rows, BoW[30], Dense(8)+BCE)")
    print(f"  MatrixAI (DenseSupervisedTrainer, SGD, 60 epochs) × {N_REPS} reps...")
    r_mx = _time_reps(bench_text_routing_matrixai, N_REPS)
    print(f"    median={r_mx['median_s']:.3f}s  max={r_mx['max_s']:.3f}s  "
          f"accuracy={r_mx['accuracy']:.1%}  best_epoch={r_mx['epochs']}")

    print(f"  scikit-learn (MLPClassifier hidden=(8,), relu, adam) × {N_REPS} reps...")
    r_sk = _time_reps(bench_text_routing_sklearn, N_REPS)
    print(f"    median={r_sk['median_s']:.3f}s  max={r_sk['max_s']:.3f}s  "
          f"accuracy={r_sk['accuracy']:.1%}")
    ratio = r_mx["median_s"] / max(r_sk["median_s"], 0.001)
    print(f"  → MatrixAI is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than sklearn (median)")
    results.append(("text_routing_fe", r_mx, r_sk))

    # ── Summary ───────────────────────────────────────────────────────────────
    _sep("Summary (median over 5 reps, sklearn pre-imported; max_s = worst single run)")
    print(f"  {'Pipeline':<22}  {'MatrixAI':>12}  {'sklearn':>12}  {'Ratio':>8}  "
          f"{'MatrixAI acc':>13}  {'sklearn acc':>11}")
    print("  " + "─" * 88)
    for label, r_mx, r_sk in results:
        ratio = r_mx["median_s"] / max(r_sk["median_s"], 0.001)
        print(f"  {label:<22}  {r_mx['median_s']:>11.3f}s  {r_sk['median_s']:>11.3f}s  "
              f"{ratio:>7.1f}x  {r_mx['accuracy']:>13.1%}  {r_sk['accuracy']:>11.1%}")

    print()
    print("  Note: import overhead excluded — sklearn pre-loaded before timing.")
    print("  This reflects production use (one process, repeated calls), not cold start.")
    print()
    print("  What MatrixAI adds over sklearn:")
    print("    + Typed IR (.mxai) — model is a declarative artifact, not code")
    print("    + Signed entry_hash — cryptographic fingerprint of model+params")
    print("    + Training trace — reproducible hash of dataset + split + epochs")
    print("    + ParameterSet — versioned, inspectable, diff-able parameters")
    print("    + Registry integration — one push_run_dir() to version and audit")
    print()
    print("  What sklearn adds over MatrixAI:")
    print("    + Faster training (optimized C/Fortran solvers — LBFGS, liblinear)")
    print("    + Much larger algorithm catalogue (100+ estimators)")
    print("    + Larger community, more documentation, more integrations")
    print()
    print("  Honest reading: MatrixAI trains slower because it does more. The")
    print("  overhead is the audit layer — typed params, training traces, registry")
    print("  hashes. For regulated environments, that overhead is the product.")

    out = ROOT / "benchmarks" / "results_training.json"
    data = {
        "environment": env,
        "reps": N_REPS,
        "note": "sklearn pre-imported; import overhead excluded from timing",
        "results": [
            {
                "pipeline": label,
                "matrixai_median_s": r_mx["median_s"],
                "matrixai_max_s": r_mx["max_s"],
                "matrixai_accuracy": r_mx["accuracy"],
                "matrixai_best_epoch": r_mx.get("epochs"),
                "sklearn_median_s": r_sk["median_s"],
                "sklearn_max_s": r_sk["max_s"],
                "sklearn_accuracy": r_sk["accuracy"],
                "ratio_median": round(r_mx["median_s"] / max(r_sk["median_s"], 0.001), 1),
            }
            for label, r_mx, r_sk in results
        ],
    }
    out.write_text(json.dumps(data, indent=2))
    print(f"\n  Results saved to benchmarks/results_training.json")


if __name__ == "__main__":
    main()
