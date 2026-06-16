# Tutorial: Tu primer modelo en 30 minutos

> **English:** [docs/en/TUTORIAL.md](../en/TUTORIAL.md)

Este tutorial retoma donde acaba el [Quickstart](QUICKSTART.md). La meta es entender el proyecto generado por `matrixai init`, ajustar un hiperparámetro, evaluar el resultado y servir el modelo por HTTP.

**Tiempo**: ~30 minutos  
**Prerrequisito**: Haber completado el [Quickstart](QUICKSTART.md)

> **Windows:** usa `python` en lugar de `python3`. Los comandos de este tutorial están en una sola línea para que funcionen en PowerShell sin cambios.

---

## Parte 1: Qué es un `.mxai` (5 min)

Abre el modelo generado:

```bash
cat my-first-classifier/my-first-classifier.mxai
```

Verás algo como:

```mxai
PROJECT my-first-classifier

VECTOR Features[3]
  feature_1: Score
  feature_2: Score
  feature_3: Score
END

FUNCTION ClassifierModel
  R: Risk = sigmoid(W1 * Features + b1)
END

DISTRIBUTION Classification
  Classification ~ Normal(R, uncertainty(Features))
END

GRAPH
  Features -> ClassifierModel -> Classification
END

AUDIT
  EXPLAIN Features -> ClassifierModel -> Classification
END
```

Qué significa:

- `VECTOR Features[3]`: el modelo recibe tres señales numéricas.
- `FUNCTION ClassifierModel`: calcula una probabilidad `R` con una función sigmoide entrenable.
- `DISTRIBUTION Classification`: envuelve la predicción con incertidumbre auditable.
- `GRAPH`: declara el recorrido ejecutable.
- `AUDIT`: conserva una explicación humana del camino seguido.

---

## Parte 2: Qué es un `.mxtrain` (5 min)

Abre el contrato de entrenamiento:

```bash
cat my-first-classifier/my-first-classifier.mxtrain
```

Verás:

```mxtrain
MODEL my-first-classifier.mxai

DATASET TrainingSet
  SOURCE csv("dataset/train.csv")
  INPUT Features FROM COLUMNS [feature_1, feature_2, feature_3]
  TARGET label: Probability
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=4 shuffle=true
END

LOSS ClassificationLoss
  TYPE binary_cross_entropy
  PREDICTION R
  TARGET label
END

OPTIMIZER ClassificationOptimizer
  TYPE sgd
  LEARNING_RATE 0.5
  UPDATE W1, b1
END

METRIC Accuracy
  TYPE accuracy
  PREDICTION R
  TARGET label
END

RUN
  EPOCHS 30
  SAVE_BEST true
END
```

La decisión crítica aquí es `TARGET label: Probability`. Para este pipeline, `binary_cross_entropy` entrena una salida sigmoide (`R`) contra una probabilidad 0/1. Este es el patrón PR1 validado para que el scaffold entrene sin tocar el core.

---

## Parte 3: Mirar los datos (3 min)

```bash
head my-first-classifier/dataset/train.csv
cat my-first-classifier/input/sample.json
```

El CSV tiene tres columnas de entrada y una etiqueta:

```csv
feature_1,feature_2,feature_3,label
0.9,0.8,0.85,1
0.1,0.15,0.12,0
```

El JSON de predicción usa las mismas tres señales:

```json
{
  "feature_1": 0.9,
  "feature_2": 0.8,
  "feature_3": 0.85
}
```

---

## Parte 4: Reentrenar con más épocas (7 min)

Cambia `EPOCHS 30` a `EPOCHS 40` en el archivo `my-first-classifier/my-first-classifier.mxtrain`. Puedes editarlo con cualquier editor de texto (Notepad, VS Code, etc.), o desde la terminal:

**Linux/Mac:**
```bash
sed -i 's/EPOCHS 30/EPOCHS 40/' my-first-classifier/my-first-classifier.mxtrain
```

**Windows (PowerShell):**
```powershell
(Get-Content my-first-classifier\my-first-classifier.mxtrain) -replace 'EPOCHS 30', 'EPOCHS 40' | Set-Content my-first-classifier\my-first-classifier.mxtrain
```

Entrena de nuevo:

```bash
python3 -m matrixai train my-first-classifier/my-first-classifier.mxai --training my-first-classifier/my-first-classifier.mxtrain --output my-first-classifier/runs/model_v2
```

Verás algo como:

```text
Training OK: model_v2
Best epoch: 40
Accuracy: 1.000000
Artifacts: my-first-classifier/runs/model_v2
```

---

## Parte 5: Evaluar artefactos (5 min)

El entrenamiento deja artefactos reproducibles:

```bash
ls my-first-classifier/runs/model_v2
python3 -m json.tool my-first-classifier/runs/model_v2/metrics.json
python3 -m json.tool my-first-classifier/runs/model_v2/validation_report.json
```

Los archivos importantes son:

- `params.best.json`: parámetros entrenados que usarás para predecir.
- `metrics.json`: métricas agregadas del entrenamiento.
- `training_trace.json`: traza del proceso.
- `validation_report.json`: validación del contrato y artefactos.

---

## Parte 6: Hacer otra predicción (3 min)

```bash
python3 -m matrixai run my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/model_v2/params.best.json --input my-first-classifier/input/sample.json --json
```

Busca `state.R` en la salida. Es la probabilidad aprendida para la clase positiva.

---

## Parte 7: Servir el modelo por HTTP (5 min)

Este paso asume que has completado la Parte 4. Necesitas estos dos archivos (comprueba que existen en tu explorador de archivos o con `ls`):

- `my-first-classifier/my-first-classifier.mxai`
- `my-first-classifier/runs/model_v2/params.best.json`

Terminal 1:

```bash
python3 -m matrixai serve my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/model_v2/params.best.json --api-key dev-secret
```

> **Windows:** si el puerto 8000 está bloqueado, añade `--port 8080` al comando y accede a `http://127.0.0.1:8080/docs`.

**Opción A — Navegador (Swagger UI):**

1. Abre `http://127.0.0.1:8000/docs` (o puerto 8080 si usaste `--port 8080`)
2. Haz clic en **Authorize** (esquina superior derecha), escribe `dev-secret` y haz clic en **Authorize**
3. Expande **POST /predict** y haz clic en **Try it out**
4. El campo Request body mostrará un JSON de ejemplo con los campos de tu modelo. Edítalo si quieres y haz clic en **Execute**
5. La respuesta aparece más abajo — busca `state.R` para ver el resultado

> ¿No sabes qué significa lo que ves? Consulta la [Guía de la interfaz HTTP](HTTP_INTERFACE.md) — explica cada sección, cada campo de la respuesta y cómo interpretarlo.

**Opción B — Terminal (curl):**

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Authorization: Bearer dev-secret" -H "Content-Type: application/json" -d "{\"feature_1\": 0.9, \"feature_2\": 0.8, \"feature_3\": 0.85}"
```

La respuesta contiene el resultado del runtime para el mismo grafo MatrixAI que usaste por CLI.

---

## Qué aprendiste

- `.mxai` define el modelo ejecutable y auditable.
- `.mxtrain` define datos, loss, optimizador, métrica y épocas.
- `matrixai init` debe generar un proyecto que entrena sin edición previa.
- `binary_cross_entropy` usa `TARGET ...: Probability` en este camino PR1.
- `run --input` recibe una ruta a JSON, no JSON inline.
- El mismo modelo puede ejecutarse por CLI o servirse por HTTP.

---

Siguiente paso: probar el quickstart con usuarios externos y registrar dónde se atascan.
