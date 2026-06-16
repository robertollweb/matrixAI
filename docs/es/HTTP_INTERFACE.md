# Guía de la interfaz HTTP — `/docs`

> **English:** [docs/en/HTTP_INTERFACE.md](../en/HTTP_INTERFACE.md)

Cuando arrancas el servidor con `matrixai serve`, MatrixAI expone una interfaz web interactiva en `http://127.0.0.1:8000/docs`. Esta guía explica qué ves, qué hace cada cosa y cómo hacer tu primera predicción desde el navegador.

---

## Arrancar el servidor

```bash
python3 -m matrixai serve mi-proyecto/mi-proyecto.mxai --params mi-proyecto/runs/v1/params.best.json --api-key dev-secret
```

Abre el navegador en `http://127.0.0.1:8000/docs`.

> **Windows:** si el puerto 8000 está bloqueado, añade `--port 8080` y accede a `http://127.0.0.1:8080/docs`.

---

## Qué ves al abrir `/docs`

La página se llama **Swagger UI**. Es una interfaz estándar que describe y permite probar una API REST directamente desde el navegador, sin instalar nada más.

```
MatrixAI Prediction API  1.0.0  OAS 3.0
/openapi.json
Auto-generated OpenAPI specification based on the deployed .mxai program.

                                          [ Authorize 🔒 ]

default ∧
  POST  /predict          Run prediction on the model        🔒
  POST  /execute-action   Execute a real action              🔒
  GET   /docs             Swagger UI
  GET   /health           Health Check

Schemas ∧
  FeaturesInput { ... }
  PredictionInput { ... }
  ...
```

**OAS 3.0** significa que el esquema sigue el estándar OpenAPI 3.0 — el mismo formato que usan miles de APIs en producción.

El icono 🔒 en un endpoint indica que requiere autenticación.

---

## Paso 1 — Autenticarse

Haz clic en **Authorize** (esquina superior derecha).

Aparece un diálogo. En el campo **Value** escribe el API key que usaste al arrancar el servidor:

```
dev-secret
```

Haz clic en **Authorize** y luego en **Close**. El icono del candado queda cerrado — ahora tus peticiones incluyen el token automáticamente.

> Sin este paso, `/predict` y `/execute-action` devuelven **401 Unauthorized**.

---

## Paso 2 — Hacer una predicción (`POST /predict`)

Este es el endpoint principal. Recibe los datos de entrada de tu modelo y devuelve el resultado.

### Cómo usarlo

1. Haz clic en la barra verde **POST /predict** para expandirla
2. Haz clic en **Try it out** (esquina derecha)
3. El campo **Request body** se vuelve editable y muestra un JSON de ejemplo con los campos de tu modelo:

```json
{
  "feature_1": 0.5,
  "feature_2": 0.5,
  "feature_3": 0.5
}
```

4. Cambia los valores si quieres (deben estar entre 0.0 y 1.0 para este modelo)
5. Haz clic en **Execute**

### Qué devuelve

La respuesta aparece más abajo. Ejemplo real:

```json
{
  "state": {
    "feature_1": 0.9,
    "feature_2": 0.8,
    "feature_3": 0.85,
    "ClassifierModel": 0.9156,
    "R": 0.9156,
    "Classification": {
      "type": "Normal",
      "mean": 0.9156,
      "sigma": 0.05
    }
  },
  "trace": [
    { "step": 1, "status": "ok", "node": "Features", "value": [0.9, 0.8, 0.85] },
    { "step": 2, "status": "ok", "node": "ClassifierModel", "value": 0.9156 },
    { "step": 3, "status": "ok", "node": "Classification", "value": { "type": "Normal", "mean": 0.9156, "sigma": 0.05 } }
  ],
  "actions": []
}
```

### Qué significa cada campo

| Campo | Qué es |
|---|---|
| `state.R` | El resultado numérico del modelo. Para clasificación binaria: probabilidad de la clase positiva. `0.91` = 91% de probabilidad. |
| `state.Classification` | El resultado interpretado como distribución. `mean` es el mismo valor que `R`. `sigma` es la incertidumbre. |
| `trace` | Lista de pasos de ejecución del grafo. Cada nodo del `.mxai` aparece aquí con su valor y `status: ok`. Si algún paso fallara, aparecería `status: error`. |
| `actions` | Lista de acciones reales disparadas (envío de email, llamada a API externa, etc.). Vacío si el modelo no tiene acciones configuradas o no se activó ninguna. |

**`R` es el número que te importa.** El resto es auditoría — puedes ignorarlo hasta que necesites depurar algo.

### Interpretar `R`

`R` es una probabilidad entre 0.0 y 1.0. Indica qué tan seguro está el modelo de que el ejemplo pertenece a la **clase positiva** — es decir, la clase que tiene `label = 1` en tus datos de entrenamiento.

- **`R` cercano a 1.0** → el modelo predice clase positiva con alta confianza
- **`R` cercano a 0.0** → el modelo predice clase negativa con alta confianza
- **`R` ≈ 0.5** → el modelo no está seguro; el ejemplo está en la frontera

**¿Es bueno o malo `R = 1`?** Depende completamente de lo que represents con label=1 en tu entrenamiento:

| Caso de uso | label=1 significa | R=1 es... |
|---|---|---|
| Detector de spam | Es spam | Malo (es spam) |
| Aprobación de crédito | Aprobado | Bueno (aprobado) |
| Detección de enfermedad | Positivo en la prueba | Depende del contexto |
| Control de calidad | Pieza defectuosa | Malo (defecto detectado) |

El modelo del quickstart usa datos de ejemplo genéricos — label=1 no tiene significado de negocio. Cuando construyas tu propio modelo, tú defines qué significa label=1 en tus datos de entrenamiento, y eso determina cómo interpretar `R`.

El umbral de decisión habitual es 0.5: por encima → clase positiva, por debajo → clase negativa. Puedes ajustarlo según cuánto quieras priorizar falsos positivos vs. falsos negativos.

---

## `POST /execute-action`

Este endpoint solo es relevante si tu modelo tiene un contrato de acciones (`.mxact`) cargado. Permite disparar una acción real (enviar un email, llamar a un sistema externo, registrar una decisión) de forma auditada.

Si solo tienes un modelo de clasificación básico, **puedes ignorar este endpoint** — no afecta a `/predict`.

---

## `GET /health`

Devuelve el estado del servidor. Útil para saber si el servidor está vivo y cuántas peticiones ha procesado.

```json
{
  "status": "ok",
  "service": "MatrixAI Server",
  "metrics": {
    "requests_total": 5,
    "requests_successful": 5,
    "requests_failed": 0,
    "items_processed": 5,
    "last_request_ms": 0.12,
    "uptime_seconds": 342
  }
}
```

No requiere autenticación. Puedes abrirlo directamente en el navegador: `http://127.0.0.1:8000/health`.

---

## Sección "Schemas" (al final de la página)

Muestra la estructura de datos que espera y devuelve el modelo. Generada automáticamente a partir de tu `.mxai`.

- **FeaturesInput** — los campos que debes enviar y su tipo (`number`, rango `0.0–1.0`)
- **PredictionInput** — el formato completo del body (acepta `FeaturesInput` directamente o envuelto en un objeto)
- **PredictionResult** — el formato de la respuesta

No necesitas leer esto para usar el endpoint. Es útil si quieres integrar el modelo en otra aplicación y necesitas saber exactamente qué estructura esperar.

---

## Hacer la misma predicción desde la terminal

Si prefieres no usar la interfaz web, el equivalente exacto con `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.85}'
```

**Windows (PowerShell):**
```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Authorization: Bearer dev-secret" -H "Content-Type: application/json" -d "{\"feature_1\": 0.9, \"feature_2\": 0.8, \"feature_3\": 0.85}"
```

---

## Resumen

| Quiero... | Uso... |
|---|---|
| Ver el resultado de mi modelo | `POST /predict` → busco `state.R` |
| Saber si el servidor está vivo | `GET /health` |
| Disparar una acción real | `POST /execute-action` (requiere `.mxact`) |
| Ver el esquema de la API | `/openapi.json` o sección Schemas en `/docs` |
