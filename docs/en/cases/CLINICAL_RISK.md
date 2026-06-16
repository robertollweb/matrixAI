# Case 3 — Clinical Risk: Explainable Decision Support

> **Español:** [docs/es/cases/CLINICAL_RISK.md](../../es/cases/CLINICAL_RISK.md)

**Industry:** Healthcare / Hospital Operations  
**Difficulty:** Intermediate  
**Runtime:** ~10 seconds

---

## The problem

A hospital's decision-support system estimates fall risk for admitted patients. Five clinical features are available: age, reduced mobility, medication load, previous falls, and cognitive state.

A model that outputs only "risk 0.87" is not enough. Clinical staff need to understand *why* — which factor drove the score — and the system must preserve that explanation so the decision can be defended under a clinical or legal review months later.

The two unsolved problems:
- Which exact version of the model produced this patient's risk classification?
- Why did the model assign that level, and can that reasoning be reproduced?

Without answers to both, the system cannot be used in regulated clinical practice.

---

## The solution

MatrixAI trains a fall-risk classifier and registers it in the model registry with a signed `entry_hash`. For every patient assessment:

- The decision is linked to the exact model version via `entry_hash`.
- Linear contribution analysis (`W1[i] × x_i`) provides the exact attribution for each feature — mathematically precise for a sigmoid-of-linear model, equivalent to linear SHAP.
- `registry.verify()` proves the model was not modified after registration.

Every decision is reproducible: same model version + same patient features → same risk score and explanation.

---

## Run it yourself

From the `matrixAI` root directory:

```bash
python3 examples/clinical-risk/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/clinical-risk/run_case.py
```

No external dependencies. No API keys. Dataset included.

### Expected output

```
MatrixAI — PR2-C3: Clinical Risk — Explainable Decision Support
==================================================================

  Modelo: riesgo de caída en pacientes hospitalizados
  Features: edad, movilidad, medicación, caídas previas, estado cognitivo
  Explicación: contribución lineal exacta W1[i]×x_i por feature

── Step 1 — Entrenar y registrar modelo de riesgo clínico ──
  Registrado: clinical-risk@v1.0
  entry_hash: sha256:a8b50f6e92bfc840826...
  Pesos aprendidos (W1): [1.035, 0.891, 1.264, 2.12, 0.993]
  Bias (b1): -2.878
  Accuracy en entrenamiento: 100.0%

  Interpretación de pesos (W1[i] > 0 → aumenta riesgo):
    Age (normalizado)               W=+1.035  ↑ riesgo
    Movilidad reducida              W=+0.891  ↑ riesgo
    Carga de medicación             W=+1.264  ↑ riesgo
    Caídas previas                  W=+2.120  ↑ riesgo
    Estado cognitivo                W=+0.993  ↑ riesgo

── Step 2 — Puntuar pacientes de test con explicación ──
  Pacientes evaluados: 12
  Precisión del modelo (binaria 0.5):  100.0%
  Baseline (clase mayoritaria):        58.3%
  Mejora:                              +41.7%

── Step 3 — Análisis de contribución lineal: 3 pacientes representativos ──
  Paciente PAC-0001  score=0.8773  NIVEL: ALERTA
  entry_hash: sha256:a8b50f6e92bfc84082651267e...
  Contribuciones (ordenadas por magnitud):
    Caídas previas    x=0.800  contrib=+1.6959  +██████████...
    Carga medicación  x=0.720  contrib=+0.9098  +███████████...
    ...

── Step 4 — Audit trail: trazabilidad de decisión PAC-0001 ──
  verify('clinical-risk', 'v1.0') → True  + modelo íntegro

── Step 5 — Tamper detection ──
  Tamper detectado — VerificationError: params.json content hash mismatch for clinical-risk@v1.0
  + Cadena criptográfica detectó modificación de W1 (peso de edad)
```

---

## The result

### Model metric

| Component | Role | Accuracy |
|---|---|---|
| FallRiskClassifier | 5 clinical features → sigmoid → risk score | 100% on 12 test patients |
| Baseline (majority class) | — | 58.3% |

On real clinical data with noisier distributions, expect 80–90%. The sustainable value is the signed explanation and traceability, not accuracy on a controlled synthetic corpus.

### Value metric

**Every clinical decision is explained and traceable to the exact model version that produced it.**

For any historical assessment:
- The `entry_hash` identifies which exact model version scored this patient.
- The linear contribution `W1[i] × x_i` shows which feature drove the risk level.
- `registry.verify()` proves the model was not altered after registration.
- Any retroactive tampering with model weights is detected in milliseconds.

This directly answers the two questions a clinical review requires: which model version decided, and why.

---

## Architecture

```
Patient features (5 normalized values)
    age, mobility, medication_load, previous_falls, cognitive_state
          │
          ▼
┌─────────────────────────────────┐
│  FallRiskClassifier             │
│  (registered, entry_hash)       │
│  VECTOR Patient[5] → sigmoid    │
│  OUTPUT R: Probability ∈ (0,1)  │
└─────────────────────────────────┘
          │
          ▼
  risk_score + threshold → risk_level (BAJO / MEDIO / ALTO / ALERTA)
          │
          ▼
  linear attribution: contribution[i] = W1[i] × x_i
  (mathematically exact for sigmoid-of-linear models)
```

Learned weights and their clinical interpretation:
```
previous_falls    W=+2.120   (highest impact — prior falls predict recurrence)
medication_load   W=+1.264   (polypharmacy increases fall risk)
age               W=+1.035
cognitive_state   W=+0.993
mobility          W=+0.891
```

---

## Limits

- **This case illustrates technical capability. It does NOT constitute clinical validation.** A real clinical decision-support system requires medical validation, regulatory approval, and professional supervision.
- The dataset is synthetic (60 training patients, 12 test). Real clinical deployment requires patient data with the actual distribution of the target population.
- Linear contribution `W1[i] × x_i` is mathematically exact for this model (sigmoid of a linear function). For non-linear models (deep networks), more sophisticated attribution is needed (SHAP/LIME) — pending hardening.
- `AUDIT EXPLAIN` in the `.mxai` file validates graph structure; runtime attribution is computed by the runbook directly over the registered parameters.
- The registry is local-first. Production deployments would use managed registry infrastructure (paid tier).

---

## What is free and what is paid

| Layer | Status |
|---|---|
| Model training and registration | **Core — free** |
| Linear contribution analysis per decision | **Core — free** |
| Decision audit log with `entry_hash` | **Core — free** |
| Tamper detection for all registered models | **Core — free** |
| Managed registry with retention and access control | Paid tier |
| Integration with clinical systems (HL7/FHIR) | Paid tier |
| Audit report generation for regulatory review | Paid tier |
| Advanced attribution for non-linear models (SHAP) | Paid tier |

---

## Files

```
examples/clinical-risk/
  clinical_risk.mxai     — Model: Patient[5] → sigmoid → risk score
  data/
    train.csv            — 60 synthetic patients (15 per risk group), auto-generated
    test.csv             — 12 held-out patients (3 per risk group)
  run_case.py            — End-to-end demo: train, score, explain, audit, tamper detect
```
