#!/usr/bin/env python3
"""PR2-C3 — Clinical risk: explainable decision support with audit trail.

Run from the matrixAI root:
    python3 examples/clinical-risk/run_case.py

Demonstrates:
  1. Training and registering a fall-risk classifier over 5 clinical features.
  2. Scoring patients with risk level interpretation (BAJO / MEDIO / ALTO / ALERTA).
  3. Linear contribution analysis per patient: for the sigmoid-of-linear model,
     contribution(feature_i) = W1[i] * x_i is the mathematically exact attribution.
  4. Decision audit trail: each clinical decision records the exact entry_hash of
     the model version that produced it, patient features, and risk level.
  5. Registry verification and tamper detection.
  6. Honest statement of limits: this is a support tool, not a clinical diagnosis.
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

FEATURE_NAMES = ["age", "mobility", "medication_load", "previous_falls", "cognitive_state"]
FEATURE_LABELS = {
    "age":             "Age (normalizado)",
    "mobility":        "Movilidad reducida",
    "medication_load": "Carga de medicación",
    "previous_falls":  "Caídas previas",
    "cognitive_state": "Estado cognitivo",
}


def _risk_level(score: float) -> str:
    if score >= 0.80:
        return "ALERTA"
    if score >= 0.60:
        return "ALTO"
    if score >= 0.35:
        return "MEDIO"
    return "BAJO"


# ── patient corpus ────────────────────────────────────────────────────────────

# Each row: (age, mobility, medication_load, previous_falls, cognitive_state, risk_probability)
# Values normalized 0-1. risk_probability is the supervision target.

_HIGH_RISK = [
    (0.88, 0.82, 0.78, 1.00, 0.76, 0.95),
    (0.79, 0.72, 0.69, 0.80, 0.70, 0.88),
    (0.93, 0.86, 0.82, 1.00, 0.83, 0.97),
    (0.74, 0.68, 0.71, 0.70, 0.64, 0.84),
    (0.85, 0.90, 0.85, 0.90, 0.88, 0.96),
    (0.77, 0.75, 0.73, 0.80, 0.72, 0.90),
    (0.91, 0.84, 0.80, 1.00, 0.78, 0.96),
    (0.82, 0.78, 0.76, 0.90, 0.80, 0.93),
    (0.70, 0.65, 0.68, 0.70, 0.66, 0.82),
    (0.87, 0.83, 0.79, 0.90, 0.81, 0.94),
    (0.76, 0.70, 0.74, 0.80, 0.68, 0.86),
    (0.94, 0.88, 0.84, 1.00, 0.85, 0.97),
    (0.80, 0.76, 0.72, 0.80, 0.74, 0.91),
    (0.83, 0.80, 0.77, 0.90, 0.79, 0.93),
    (0.78, 0.73, 0.70, 0.80, 0.71, 0.89),
]

_MEDIUM_RISK = [
    (0.55, 0.50, 0.52, 0.40, 0.48, 0.55),
    (0.62, 0.58, 0.55, 0.50, 0.54, 0.62),
    (0.48, 0.45, 0.50, 0.40, 0.43, 0.48),
    (0.58, 0.55, 0.48, 0.50, 0.52, 0.57),
    (0.65, 0.60, 0.57, 0.50, 0.58, 0.65),
    (0.50, 0.48, 0.53, 0.30, 0.46, 0.50),
    (0.60, 0.57, 0.54, 0.50, 0.55, 0.60),
    (0.45, 0.42, 0.46, 0.30, 0.40, 0.44),
    (0.68, 0.64, 0.60, 0.60, 0.62, 0.68),
    (0.52, 0.50, 0.47, 0.40, 0.48, 0.52),
    (0.63, 0.60, 0.56, 0.50, 0.57, 0.63),
    (0.47, 0.44, 0.49, 0.30, 0.42, 0.46),
    (0.57, 0.54, 0.51, 0.50, 0.53, 0.57),
    (0.70, 0.65, 0.62, 0.60, 0.64, 0.70),
    (0.53, 0.51, 0.48, 0.40, 0.49, 0.52),
    (0.61, 0.58, 0.55, 0.50, 0.56, 0.61),
    (0.46, 0.43, 0.47, 0.30, 0.41, 0.45),
    (0.64, 0.61, 0.57, 0.50, 0.59, 0.64),
    (0.49, 0.46, 0.50, 0.40, 0.44, 0.49),
    (0.56, 0.53, 0.51, 0.50, 0.52, 0.56),
    (0.67, 0.63, 0.59, 0.60, 0.62, 0.67),
    (0.43, 0.41, 0.45, 0.20, 0.38, 0.41),
    (0.59, 0.56, 0.53, 0.50, 0.55, 0.59),
    (0.51, 0.49, 0.46, 0.40, 0.47, 0.51),
    (0.66, 0.62, 0.58, 0.60, 0.61, 0.66),
    (0.44, 0.42, 0.46, 0.30, 0.39, 0.43),
    (0.69, 0.65, 0.61, 0.60, 0.63, 0.69),
    (0.54, 0.52, 0.49, 0.40, 0.50, 0.54),
    (0.62, 0.59, 0.55, 0.50, 0.58, 0.62),
    (0.48, 0.46, 0.49, 0.30, 0.43, 0.47),
]

_LOW_RISK = [
    (0.16, 0.22, 0.12, 0.00, 0.18, 0.08),
    (0.28, 0.26, 0.18, 0.10, 0.24, 0.14),
    (0.20, 0.18, 0.15, 0.00, 0.21, 0.10),
    (0.34, 0.31, 0.22, 0.20, 0.29, 0.18),
    (0.12, 0.15, 0.10, 0.00, 0.14, 0.06),
    (0.25, 0.28, 0.19, 0.10, 0.22, 0.12),
    (0.18, 0.20, 0.13, 0.00, 0.17, 0.08),
    (0.30, 0.27, 0.21, 0.10, 0.26, 0.15),
    (0.22, 0.24, 0.16, 0.10, 0.20, 0.11),
    (0.32, 0.29, 0.23, 0.20, 0.27, 0.17),
    (0.15, 0.17, 0.11, 0.00, 0.15, 0.07),
    (0.27, 0.25, 0.18, 0.10, 0.23, 0.13),
    (0.19, 0.21, 0.14, 0.00, 0.18, 0.09),
    (0.33, 0.30, 0.22, 0.20, 0.28, 0.17),
    (0.14, 0.16, 0.09, 0.00, 0.13, 0.06),
]


def _ensure_data() -> None:
    """Generate train.csv and test.csv if not present."""
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        return

    header = FEATURE_NAMES + ["risk_probability"]
    train_rows: list[tuple] = []
    test_rows: list[tuple] = []

    for i, row in enumerate(_HIGH_RISK):
        (test_rows if i >= 12 else train_rows).append(row)
    for i, row in enumerate(_MEDIUM_RISK):
        (test_rows if i >= 24 else train_rows).append(row)
    for i, row in enumerate(_LOW_RISK):
        (test_rows if i >= 12 else train_rows).append(row)

    for path, rows in [(train_path, train_rows), (test_path, test_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


# ── helpers ───────────────────────────────────────────────────────────────────

def _separator(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print("─" * 64)


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return [
            {k: float(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _train_model(learning_rate: float, seed: int, epochs: int, tmp: Path) -> tuple[Path, float]:
    from matrixai.training import parse_training_text, SupervisedTrainer

    mxtrain = f"""MODEL clinical_risk.mxai

DATASET PatientData
  SOURCE csv("data/train.csv")
  INPUT Patient FROM COLUMNS [
    age, mobility, medication_load, previous_falls, cognitive_state
  ]
  TARGET risk_probability: Probability
  SPLIT train=0.8 validation=0.2 seed={seed}
  BATCH size=8 shuffle=true
END

LOSS ClinicalLoss
  TYPE binary_cross_entropy
  PREDICTION R
  TARGET risk_probability
END

OPTIMIZER ClinicalOpt
  TYPE sgd
  LEARNING_RATE {learning_rate}
  UPDATE W1, b1
END

METRIC Accuracy
  TYPE accuracy
  PREDICTION R
  TARGET risk_probability
END

RUN
  EPOCHS {epochs}
  SAVE_BEST true
END
"""
    spec = parse_training_text(mxtrain)
    run_dir = tmp / "train_out"
    result = SupervisedTrainer().train(spec, output_dir=str(run_dir), base_path=HERE)

    (run_dir / "evaluation_report.json").write_text(json.dumps({
        "accuracy": round(result.accuracy, 4),
        "best_epoch": result.best_epoch,
        "best_validation_loss": round(result.best_validation_loss, 6),
        "learning_rate": learning_rate,
        "seed": seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    shutil.copy(HERE / "clinical_risk.mxai", run_dir / "model_snapshot.mxai")
    return run_dir, result.accuracy


def _predict(program: object, ps: object, row: dict) -> float:
    from matrixai.runtime import MatrixAIRuntime
    rt = MatrixAIRuntime()
    features = {f: row[f] for f in FEATURE_NAMES}
    result = rt.run(program, features, ps.runtime_parameters())
    return result["state"]["R"]


def _get_weights(ps: object) -> tuple[list[float], float]:
    """Extract W1 (shape [5]) and b1 (scalar) from parameter set."""
    params = ps.runtime_parameters()
    W1 = params.get("W1", params.get("RiskModel.W1", []))
    b1_raw = params.get("b1", params.get("RiskModel.b1", 0.0))
    # W1 can be [w0,..,w4] or [[w0,..,w4]] depending on initializer path
    if W1 and isinstance(W1[0], list):
        W1 = W1[0]
    b1 = b1_raw[0] if isinstance(b1_raw, list) else float(b1_raw)
    return list(W1), b1


def _explain(W1: list[float], row: dict) -> list[tuple[str, float, float]]:
    """
    Linear contribution analysis for sigmoid(W1·x + b1).

    For a linear (sigmoid-of-linear) model, the exact contribution of feature i is:
        contribution_i = W1[i] * x_i

    This is mathematically identical to a signed, linear SHAP value for this model class.
    Returns: [(feature_name, feature_value, contribution), ...] sorted by |contribution| desc.
    """
    contribs = []
    for i, name in enumerate(FEATURE_NAMES):
        val = row[name]
        contrib = W1[i] * val
        contribs.append((name, val, contrib))
    contribs.sort(key=lambda t: abs(t[2]), reverse=True)
    return contribs


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from matrixai.parameters import load_parameter_set
    from matrixai.parser.parser import parse_file
    from matrixai.registry import ModelRegistry, VerificationError

    _ensure_data()

    if REGISTRY_PATH.exists():
        shutil.rmtree(REGISTRY_PATH)
    registry = ModelRegistry(REGISTRY_PATH)

    print("MatrixAI — PR2-C3: Clinical Risk — Explainable Decision Support")
    print("=" * 66)
    print()
    print("  Modelo: riesgo de caída en pacientes hospitalizados")
    print("  Features: edad, movilidad, medicación, caídas previas, estado cognitivo")
    print("  Explicación: contribución lineal exacta W1[i]×x_i por feature")

    train_rows = _load_rows(HERE / "data" / "train.csv")
    test_rows = _load_rows(HERE / "data" / "test.csv")

    # ── Step 1: Train and register model ─────────────────────────────────────
    _separator("Step 1 — Entrenar y registrar modelo de riesgo clínico")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir, acc = _train_model(
            learning_rate=0.8, seed=7, epochs=50, tmp=Path(tmp)
        )
        entry = registry.push_run_dir(run_dir, "clinical-risk", "v1.0",
                                      interpretability_level="full")

    program = parse_file(HERE / "clinical_risk.mxai")
    ps_path = REGISTRY_PATH / "entries" / "clinical-risk" / "v1.0" / "params.json"
    ps = load_parameter_set(ps_path)
    W1, b1 = _get_weights(ps)

    print(f"  Registrado: clinical-risk@v1.0")
    print(f"  entry_hash: {entry.entry_hash[:28]}...")
    print(f"  Pesos aprendidos (W1): {[round(w, 3) for w in W1]}")
    print(f"  Bias (b1): {round(b1, 3)}")
    print(f"  Accuracy en entrenamiento: {acc:.1%}")
    print()
    print("  Interpretación de pesos (W1[i] > 0 → aumenta riesgo):")
    for name, w in zip(FEATURE_NAMES, W1):
        direction = "↑ riesgo" if w > 0 else "↓ riesgo"
        print(f"    {FEATURE_LABELS[name]:30s}  W={w:+.3f}  {direction}")

    # ── Step 2: Score test patients + explanation ─────────────────────────────
    _separator("Step 2 — Puntuar pacientes de test con explicación por decisión")

    decision_log: list[dict] = []
    for i, row in enumerate(test_rows):
        score = _predict(program, ps, row)
        level = _risk_level(score)
        contribs = _explain(W1, row)
        top_factor = contribs[0][0] if contribs else "—"
        decision_log.append({
            "patient_id": f"PAC-{i+1:04d}",
            "features": {f: round(row[f], 3) for f in FEATURE_NAMES},
            "risk_score": round(score, 4),
            "risk_level": level,
            "top_factor": top_factor,
            "contributions": [(n, round(v, 3), round(c, 4)) for n, v, c in contribs],
            "entry_hash": entry.entry_hash,
            "model_version": "v1.0",
            "decided_at": datetime.now(timezone.utc).isoformat(),
        })

    # accuracy vs baseline
    high_risk_actual = sum(1 for r in test_rows if r["risk_probability"] >= 0.6)
    low_risk_actual = sum(1 for r in test_rows if r["risk_probability"] < 0.35)
    correct = sum(
        1 for d, r in zip(decision_log, test_rows)
        if (d["risk_score"] >= 0.5) == (r["risk_probability"] >= 0.5)
    )
    total = len(test_rows)
    majority = max(
        sum(1 for r in test_rows if r["risk_probability"] >= 0.5),
        sum(1 for r in test_rows if r["risk_probability"] < 0.5),
    )

    print(f"  Pacientes evaluados: {total}")
    print(f"  Precisión del modelo (binaria 0.5):  {correct/total:.1%}")
    print(f"  Baseline (clase mayoritaria):        {majority/total:.1%}")
    print(f"  Mejora:                              +{(correct/total - majority/total):.1%}")
    print()

    # Print table
    header = f"  {'ID':8s}  {'Riesgo':7s}  {'Nivel':6s}  {'Factor principal':22s}  {'Contribución'}"
    print(header)
    print("  " + "─" * 70)
    for d in decision_log:
        top = d["contributions"][0]
        print(f"  {d['patient_id']}  {d['risk_score']:.4f}  {d['risk_level']:6s}  "
              f"{FEATURE_LABELS[top[0]]:22s}  {top[2]:+.4f}")

    # ── Step 3: Detailed explanation for 3 representative patients ────────────
    _separator("Step 3 — Análisis de contribución lineal: 3 pacientes representativos")
    print("  Para un modelo lineal sigmoid(W1·x + b1),")
    print("  la contribución exacta de feature_i es W1[i] × x_i.")
    print("  Esta es la atribución matemática precisa para este tipo de modelo.")
    print()

    # Pick one from each risk group
    alto = next((d for d in decision_log if d["risk_level"] in ("ALTO", "ALERTA")), None)
    medio = next((d for d in decision_log if d["risk_level"] == "MEDIO"), None)
    bajo = next((d for d in decision_log if d["risk_level"] == "BAJO"), None)

    for d in filter(None, [alto, medio, bajo]):
        print(f"  Paciente {d['patient_id']}  score={d['risk_score']:.4f}  NIVEL: {d['risk_level']}")
        print(f"  entry_hash: {d['entry_hash'][:32]}...")
        print(f"  Contribuciones (ordenadas por magnitud):")
        for name, val, contrib in d["contributions"]:
            bar = "█" * int(abs(contrib) * 30)
            sign = "+" if contrib >= 0 else "-"
            print(f"    {FEATURE_LABELS[name]:30s}  x={val:.3f}  contrib={contrib:+.4f}  {sign}{bar}")
        total_score_approx = sum(c for _, _, c in d["contributions"]) + b1
        print(f"    {'(suma = Σ contribuciones + bias)':30s}  Σ+b = {total_score_approx:.4f}")
        print()

    # ── Step 4: Audit trail ───────────────────────────────────────────────────
    _separator("Step 4 — Audit trail: trazabilidad de decisión PAC-0001")
    audit = decision_log[0]
    print(f"  Paciente:     {audit['patient_id']}")
    print(f"  Decisión:     nivel {audit['risk_level']}  (score={audit['risk_score']})")
    print(f"  Modelo:       clinical-risk@{audit['model_version']}")
    print(f"  entry_hash:   {audit['entry_hash'][:32]}...")
    print(f"  Decidido el:  {audit['decided_at']}")
    print()
    print("  Features registradas en el momento de la decisión:")
    for feat, val in audit["features"].items():
        print(f"    {FEATURE_LABELS[feat]:30s}  {val}")
    print()
    print("  Factor principal de riesgo:")
    top = audit["contributions"][0]
    print(f"    {FEATURE_LABELS[top[0]]}: valor={top[1]}, contribución={top[2]:+.4f}")
    print()
    ok = registry.verify("clinical-risk", "v1.0")
    print(f"  verify('clinical-risk', 'v1.0') → {ok}  + modelo íntegro")

    # ── Step 5: Tamper detection ──────────────────────────────────────────────
    _separator("Step 5 — Tamper detection: modificar pesos rompe la cadena criptográfica")
    params_path = REGISTRY_PATH / "entries" / "clinical-risk" / "v1.0" / "params.json"
    original = json.loads(params_path.read_text())
    tampered = json.loads(json.dumps(original))
    if "W1" in tampered.get("parameters", {}):
        w1_entry = tampered["parameters"]["W1"]
        if isinstance(w1_entry.get("values"), list):
            vals = list(w1_entry["values"])
            vals[0] = -99.0  # tamper: invert age weight
            tampered["parameters"]["W1"]["values"] = vals
    params_path.write_text(json.dumps(tampered))
    try:
        registry.verify("clinical-risk", "v1.0")
        print("  ERROR: tamper no detectado!")
    except VerificationError as e:
        print(f"  Tamper detectado — VerificationError: {e}")
        print("  + Cadena criptográfica detectó modificación de W1 (peso de edad)")
    params_path.write_text(json.dumps(original))

    # ── Límites honestos ──────────────────────────────────────────────────────
    _separator("Límites honestos del caso")
    print("  + Este caso ilustra capacidad técnica de explicación y trazabilidad.")
    print("  x NO constituye validación clínica. Un sistema de apoyo a decisión")
    print("    clínica real requiere validación médica, aprobación regulatoria")
    print("    y supervisión profesional.")
    print()
    print("  Sobre la explicación:")
    print("  + Contribución lineal W1[i]×x_i es matematicamente exacta para")
    print("    este modelo (sigmoid de función lineal).")
    print("  x No es SHAP ni LIME. Para modelos no-lineales (redes profundas)")
    print("    se necesita atribución más sofisticada — pendiente de hardening.")
    print("  x AUDIT EXPLAIN en el .mxai valida la estructura del grafo;")
    print("    la atribución en tiempo de ejecución la calcula este runbook.")

    # ── Summary ───────────────────────────────────────────────────────────────
    _separator("Summary")
    alerta = sum(1 for d in decision_log if d["risk_level"] == "ALERTA")
    alto_cnt = sum(1 for d in decision_log if d["risk_level"] == "ALTO")
    medio_cnt = sum(1 for d in decision_log if d["risk_level"] == "MEDIO")
    bajo_cnt = sum(1 for d in decision_log if d["risk_level"] == "BAJO")
    print(f"  Pacientes evaluados: {total}")
    print(f"    ALERTA: {alerta}  |  ALTO: {alto_cnt}  |  MEDIO: {medio_cnt}  |  BAJO: {bajo_cnt}")
    print(f"  Precisión del modelo: {correct/total:.1%} vs {majority/total:.1%} baseline")
    print()
    print("  Valor entregado:")
    print(f"    + Cada decisión registra el entry_hash exacto del modelo")
    print(f"    + Cada decisión registra los valores de features en ese instante")
    print(f"    + Explicación lineal muestra qué feature contribuyó más al riesgo")
    print(f"    + Tamper detection: modificar parámetros rompe verify()")
    print()
    print("  Valor de negocio:")
    print(f"    En una planta hospitalaria con {total} pacientes evaluados:")
    print(f"    - {alerta + alto_cnt} pacientes en riesgo ALTO/ALERTA identificados")
    print(f"    - Cada decisión es defendible ante una revisión clínica o legal")
    print(f"    - La explicación está disponible sin que el personal técnico intervenga")
    print()


if __name__ == "__main__":
    main()
