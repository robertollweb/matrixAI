# Caso 1 — Scoring de crédito: trazabilidad regulatoria

> **English:** [docs/en/cases/CREDIT_SCORING.md](../../en/cases/CREDIT_SCORING.md)

**Industria:** Financiero / Microcrédito  
**Dificultad:** Principiante  
**Tiempo de ejecución:** ~15 segundos

---

## El problema

Una entidad de microcréditos debe poder demostrar —cuando la audita un regulador— qué versión exacta del modelo aprobó o rechazó cada solicitud de préstamo, con prueba criptográfica de que el registro no fue alterado retroactivamente.

Hoy esto se resuelve habitualmente con hojas de cálculo, capturas de pantalla y notas manuales de versión. Cuando un auditor pregunta "¿qué modelo decidió esta solicitud de hace seis meses?", la respuesta requiere días de reconstrucción manual y es imposible de verificar.

El coste es real: multas regulatorias por fallos de trazabilidad, cara preparación manual de auditorías y riesgo reputacional por decisiones que no se pueden explicar ni atribuir.

---

## La solución

MatrixAI mantiene un registro inmutable y firmado criptográficamente de cada versión de modelo desplegada. Cada entrada tiene:

- un `entry_hash` determinista que cubre los pesos del modelo, la traza de entrenamiento y el informe de evaluación,
- un `parameter_set_id` que vincula cada predicción con los parámetros exactos utilizados,
- detección de manipulación: modificar cualquier archivo almacenado rompe la verificación.

Para cada decisión de crédito, se registra qué `entry_hash` la produjo. Seis meses después, un auditor puede verificar que el modelo en ese hash es byte a byte idéntico al que estaba en producción —y que nadie lo modificó.

---

## Ejecútalo tú mismo

Desde el directorio raíz de `matrixAI`:

```bash
python3 examples/credit-scoring/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/credit-scoring/run_case.py
```

Sin dependencias externas. Sin API keys. Dataset incluido.

### Salida esperada

```
MatrixAI — PR2-C1: Credit Scoring Regulatory Traceability
============================================================

── Step 1 — Train and register credit-scoring v1.0 ──
  Registered: credit-scoring@v1.0
  entry_hash: sha256:dfbfeffa978100669...
  Training accuracy: 87.5%

── Step 2 — Improved model: train and register v1.1 ──
  Registered: credit-scoring@v1.1
  entry_hash: sha256:ab21a5cb235f4e090...
  Training accuracy: 87.5%

── Step 3 — Score 30 test applications ──
  Model accuracy on test set:  93.3%
  Baseline (approve all):      73.3%
  Improvement over baseline:   +20.0%

── Step 4 — Regulatory audit: retrieve exact model for APP-0007 ──
  Decision: APPROVED (score=0.8098)
  Entry hash: sha256:ab21a5cb235f4e090...
  verify('credit-scoring', 'v1.1') → True  ✓ model and parameters intact

── Step 5 — Tamper detection ──
  Tamper detected — VerificationError: params.json content hash mismatch
  ✓ Cryptographic chain caught the modification
```

---

## El resultado

### Métrica de modelo

| Método | Accuracy en test |
|---|---|
| Baseline (aprobar todo) | 73,3 % |
| MatrixAI credit-scoring v1.1 | **93,3 %** |
| Mejora | **+20 puntos porcentuales** |

### Métrica de valor

**El 100 % de las decisiones de crédito son trazables y resistentes a manipulación.**

Para cualquier decisión histórica:
- La versión exacta del modelo (por `entry_hash`) que la produjo es recuperable.
- La verificación criptográfica prueba que ni los pesos del modelo ni los parámetros se alteraron después.
- Cualquier modificación retroactiva se detecta en milisegundos.

Esto satisface directamente el requisito de auditabilidad que los reguladores imponen sobre las decisiones automatizadas de crédito —un requisito que, sin este sistema, cuesta días de reconstrucción manual por auditoría.

---

## Límites

Este caso ilustra la **capacidad técnica de trazabilidad**, no la certificación regulatoria.

- El dataset es sintético. La accuracy real depende de la calidad de tus datos de solicitudes de crédito.
- MatrixAI proporciona la traza de auditoría criptográfica; no constituye cumplimiento legal ni sustituye la revisión regulatoria en tu jurisdicción.
- El modelo es una regresión logística (lineal) sobre 5 variables. Para carteras de crédito complejas, puede ser necesario un conjunto de variables más rico y un modelo más expresivo.
- El registry es local. Para producción, operarías sobre infraestructura con controles de acceso, copias de seguridad y políticas de retención —esa capa operativa es el nivel de pago.

---

## Qué es gratis y qué se paga

| Capa | Estado |
|---|---|
| Modelo de scoring (entrenamiento, inferencia, evaluación) | **Core — gratuito** |
| Registry local con cadena `entry_hash` y detección de manipulación | **Core — gratuito** |
| Log de decisiones con `entry_hash` por predicción | **Core — gratuito** |
| Registry como servicio (gestionado, con retención y control de acceso) | Nivel de pago |
| Soporte de auditoría para reguladores (generación de informes, orientación experta) | Nivel de pago |
| Integración con sistemas existentes de originación de crédito | Nivel de pago |

El valor que aporta la capa gratuita es real: cualquier desarrollador puede construir y verificar un pipeline de scoring de crédito completamente trazable hoy, sin coste. La capa de pago es lo que hace ese pipeline apto para producción en un entorno regulado.

---

## Archivos

```
examples/credit-scoring/
  credit_scoring.mxai       — definición del modelo (5 variables, sigmoid, clasificación binaria)
  credit_scoring.mxtrain    — especificación de entrenamiento
  data/
    train.csv               — 120 solicitudes de crédito sintéticas
    test.csv                — 30 solicitudes de test
  run_case.py               — script de demo extremo a extremo
```
