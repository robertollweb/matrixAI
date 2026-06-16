# Model: Best Parameters

**Status:** Production-ready  
**Model:** Binary classifier (3 features → probability)  
**Accuracy:** 1.0  
**Date trained:** 2026-05-23  
**Parent:** `runs/v1/`

## Contenido

- `params.json` — Parámetros entrenados del modelo (W1, b1)
  - Contiene: pesos, sesgos, métricas, hashes para auditoría
  - Formato: JSON estructurado compatible con MatrixAI

## Usar este modelo

### 1. Predicción local (CLI)

```bash
python3 -m matrixai run my-first-classifier.mxai \
  --params models/best/params.json \
  --input input/sample.json \
  --json
```

### 2. Predicción desde código Python

```python
from matrixai.runtime import MatrixAIProgram, ParameterSet

# Cargar modelo
program = MatrixAIProgram.parse_mxai_file("my-first-classifier.mxai")
params = ParameterSet.load_from_file("models/best/params.json")

# Predicción
input_data = {"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.85}
result = program.run(params, input_data)
print(f"Predicción: {result['state']['R']}")  # ≈ 0.9156
```

### 3. Exportar a ONNX (para otros frameworks)

```bash
python3 -m matrixai export-onnx my-first-classifier.mxai \
  --params models/best/params.json \
  --output models/best/classifier.onnx
```

Luego usar el ONNX en:
- TensorFlow/PyTorch/JAX
- ONNX Runtime (C++, Java, Node.js, etc.)
- Navegador (ONNX.js)
- Edge devices (ONNX Mobile)

### 4. Servir por HTTP

```bash
python3 -m matrixai serve my-first-classifier.mxai \
  --params models/best/params.json \
  --port 8000
```

Luego hacer requests:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.85}'
```

## Auditoría

Cada parámetro en `params.json` incluye:

```json
{
  "metrics": {
    "accuracy": 1.0,
    "validation_loss": 0.083059
  },
  "model_hash": "mxai_295d5ac53d5b70fd",
  "parameter_schema_hash": "params_7bdee2c45b56c62e",
  "parameter_set_id": "v1_best",
  "parameters": {
    "W1": { "shape": [3], "dtype": "float32", "role": "weights", ... },
    "b1": { "shape": [], "dtype": "float32", "role": "bias", ... }
  }
}
```

- `model_hash` — Identifica la arquitectura (.mxai)
- `parameter_schema_hash` — Identifica la estructura de parámetros
- `parameter_set_id` — Identifica esta versión de parámetros
- Métricas — Accuracy y loss en validación

## Comparar con entrenamiento

- **`models/best/params.json`** — Mejor modelo (usar en producción)
- **`runs/v1/params.best.json`** — Igual contenido (referencia)
- **`runs/v1/metrics.json`** — Métricas por epoch (cómo mejoró)
- **`runs/v1/training_trace.json`** — Trazabilidad completa

## Siguiente paso

Para mejorar este modelo:
1. Reemplaza `dataset/train.csv` con tus propios datos
2. Entrena: `python3 -m matrixai train ...`
3. Copia el mejor resultado: `cp runs/v1/params.best.json models/best/params.json`
4. Valida con usuarios
5. Despliega

Para explorar arquitecturas más complejas, ve a:
- [TUTORIAL_PRIMER_MODELO.md](../../TUTORIAL_PRIMER_MODELO.md) — hiperparámetros, evaluación, serving
- [documentacion/](../../documentacion/) — transformers, embeddings, architecturas avanzadas
