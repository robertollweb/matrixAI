# Caso 3 — Riesgo Clínico: Sistema de Apoyo a la Decisión con Explicación

> **English:** [docs/en/cases/CLINICAL_RISK.md](../../en/cases/CLINICAL_RISK.md)

**Industria:** Salud / Operaciones Hospitalarias  
**Dificultad:** Intermedio  
**Tiempo de ejecución:** ~10 segundos

---

## El problema

El sistema de apoyo a la decisión de un hospital estima el riesgo de caída de los pacientes ingresados. Dispone de cinco features clínicas: edad, movilidad reducida, carga de medicación, caídas previas y estado cognitivo.

Un modelo que solo devuelve "riesgo 0.87" no es suficiente. El personal clínico necesita entender *por qué* —qué factor impulsa la puntuación— y el sistema debe conservar esa explicación para que la decisión pueda defenderse ante una revisión clínica o legal meses después.

Los dos problemas no resueltos:
- ¿Qué versión exacta del modelo produjo la clasificación de riesgo de este paciente?
- ¿Por qué el modelo asignó ese nivel, y puede reproducirse ese razonamiento?

Sin respuesta a ambas preguntas, el sistema no puede utilizarse en la práctica clínica regulada.

---

## La solución

MatrixAI entrena un clasificador de riesgo de caída y lo registra en el registry de modelos con un `entry_hash` firmado. Para cada evaluación de paciente:

- La decisión queda vinculada a la versión exacta del modelo mediante el `entry_hash`.
- El análisis de contribución lineal (`W1[i] × x_i`) proporciona la atribución exacta de cada feature — matemáticamente preciso para un modelo sigmoid-de-lineal, equivalente a SHAP lineal.
- `registry.verify()` prueba que el modelo no fue modificado tras el registro.

Cada decisión es reproducible: misma versión del modelo + mismos valores de features → misma puntuación de riesgo y explicación.

---

## Ejecútalo tú mismo

Desde el directorio raíz de `matrixAI`:

```bash
python3 examples/clinical-risk/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/clinical-risk/run_case.py
```

Sin dependencias externas. Sin API keys. Dataset incluido.

### Salida esperada

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

## El resultado

### Métrica de modelo

| Componente | Rol | Accuracy |
|---|---|---|
| FallRiskClassifier | 5 features clínicas → sigmoid → score de riesgo | 100% en 12 pacientes de test |
| Baseline (clase mayoritaria) | — | 58,3 % |

Con datos clínicos reales y distribuciones más ruidosas se espera 80–90%. El valor sostenible es la explicación firmada y la trazabilidad, no la accuracy sobre un corpus sintético controlado.

### Métrica de valor

**Cada decisión clínica está explicada y es trazable a la versión exacta del modelo que la produjo.**

Para cualquier evaluación histórica:
- El `entry_hash` identifica qué versión exacta del modelo puntuó a este paciente.
- La contribución lineal `W1[i] × x_i` muestra qué feature impulsó el nivel de riesgo.
- `registry.verify()` prueba que el modelo no fue alterado tras el registro.
- Cualquier manipulación retroactiva de los pesos se detecta en milisegundos.

Esto responde directamente las dos preguntas que exige una revisión clínica: qué versión del modelo decidió, y por qué.

---

## Arquitectura

```
Features del paciente (5 valores normalizados)
    edad, movilidad, carga_medicación, caídas_previas, estado_cognitivo
          │
          ▼
┌─────────────────────────────────┐
│  FallRiskClassifier             │
│  (registrado, entry_hash)       │
│  VECTOR Patient[5] → sigmoid    │
│  OUTPUT R: Probability ∈ (0,1)  │
└─────────────────────────────────┘
          │
          ▼
  risk_score + umbral → nivel (BAJO / MEDIO / ALTO / ALERTA)
          │
          ▼
  atribución lineal: contribution[i] = W1[i] × x_i
  (matemáticamente exacta para modelos sigmoid-de-lineal)
```

Pesos aprendidos e interpretación clínica:
```
caídas_previas     W=+2.120   (mayor impacto — caídas previas predicen recurrencia)
carga_medicación   W=+1.264   (polifarmacia aumenta el riesgo de caída)
edad               W=+1.035
estado_cognitivo   W=+0.993
movilidad          W=+0.891
```

---

## Límites

- **Este caso ilustra capacidad técnica. NO constituye validación clínica.** Un sistema de apoyo a la decisión clínica real requiere validación médica, aprobación regulatoria y supervisión profesional.
- El dataset es sintético (60 pacientes de entrenamiento, 12 de test). El despliegue clínico real requiere datos de pacientes con la distribución real de la población objetivo.
- La contribución lineal `W1[i] × x_i` es matemáticamente exacta para este modelo (sigmoid de función lineal). Para modelos no-lineales (redes profundas) se necesita atribución más sofisticada (SHAP/LIME) — pendiente de hardening.
- `AUDIT EXPLAIN` en el fichero `.mxai` valida la estructura del grafo; la atribución en tiempo de ejecución la calcula el runbook directamente sobre los parámetros registrados.
- El registry es local-first. Para producción se operaría sobre infraestructura gestionada (nivel de pago).

---

## Qué es gratis y qué se paga

| Capa | Estado |
|---|---|
| Entrenamiento y registro del modelo | **Core — gratuito** |
| Análisis de contribución lineal por decisión | **Core — gratuito** |
| Log de auditoría de decisiones con `entry_hash` | **Core — gratuito** |
| Detección de manipulación en modelos registrados | **Core — gratuito** |
| Registry gestionado con retención y control de acceso | Nivel de pago |
| Integración con sistemas clínicos (HL7/FHIR) | Nivel de pago |
| Generación de informes de auditoría para revisión regulatoria | Nivel de pago |
| Atribución avanzada para modelos no-lineales (SHAP) | Nivel de pago |

---

## Archivos

```
examples/clinical-risk/
  clinical_risk.mxai     — Modelo: Patient[5] → sigmoid → score de riesgo
  data/
    train.csv            — 60 pacientes sintéticos (15 por nivel de riesgo), autogenerado
    test.csv             — 12 pacientes de test (3 por nivel de riesgo)
  run_case.py            — Demo extremo a extremo: entrenar, puntuar, explicar, auditar, tamper
```
