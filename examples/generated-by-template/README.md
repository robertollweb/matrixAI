# Ejemplos generados por templates

Este directorio contiene proyectos generados automáticamente por `matrixai init --template <name>`.

Sirven como referencia de:
- Estructura exacta que genera cada template
- Archivos y configuración de ejemplo
- Resultados del entrenamiento (parámetros, métricas)
- Reproducibilidad: cualquiera puede ejecutar estos proyectos sin cambios

## `my-classifier/`

**Generado con:** `python3 -m matrixai init my-classifier --template classification`  
**Fecha:** 2026-05-23  
**Estado:** ✅ Entrenable, predicción funcional

### Flujo de prueba

```bash
cd my-classifier

# Entrenar
python3 -m matrixai train my-classifier.mxai --training my-classifier.mxtrain --output runs/v1

# Predecir
python3 -m matrixai run my-classifier.mxai --params runs/v1/params.best.json --input input/sample.json --json
```

### Resultados esperados

- **Accuracy:** 1.0
- **Best epoch:** 30
- **Prediction (sample.json):** state.R ≈ 0.9156

### Archivos

- `my-classifier.mxai` — Arquitectura del modelo (Features → ClassifierModel)
- `my-classifier.mxtrain` — Configuración entrenamiento (SGD, 30 epochs, binary_cross_entropy)
- `dataset/train.csv` — 15 ejemplos de entrenamiento (3 features + label)
- `dataset/test.csv` — 5 ejemplos de prueba
- `input/sample.json` — Ejemplo de entrada para predicción
- `runs/v1/` — Artefactos de entrenamiento
  - `params.best.json` — Parámetros óptimos (W1, b1)
  - `metrics.json` — Métricas por epoch
  - `training_trace.json` — Historial completo
  - `model_snapshot.mxai` — Snapshot del modelo

## Uso

Para ver exactamente qué genera el template `classification`:

```bash
# Opción 1: Mirar este ejemplo
ls -la my-classifier/

# Opción 2: Generar uno nuevo
cd /tmp
python3 -m matrixai init another-project --template classification
```

## Nota

Estos proyectos están versionados en git para:
- Auditoría: cambios en templates quedan registrados
- Reproducibilidad: resultados entrenamiento verificables
- CI/CD: tests pueden validar que el template sigue siendo entrenable
