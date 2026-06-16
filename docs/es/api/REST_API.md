# Referencia de la API REST de MatrixAI

MatrixAI expone dos servidores HTTP independientes, cada uno con un rol distinto:

- **Servidor de producción** — sirve un modelo entrenado y registrado. Requiere autenticación por API key. Destinado a la integración con sistemas externos, la app Studio y despliegues en producción.
- **Servidor Studio** — el entorno de desarrollo de modelos. Gestiona la generación de modelos desde prompts, el entrenamiento, la validación y el refinamiento. No requiere autenticación; no está diseñado para exponerse públicamente.

Ambos servidores se arrancan desde la CLI (`matrixai serve`, `matrixai playground`) y pueden ejecutarse de forma independiente.

---

## Autenticación

El servidor de producción requiere una API key para los endpoints de predicción y acción.

Se aceptan dos formatos equivalentes:

```
Authorization: Bearer <api_key>
```
```
X-API-Key: <api_key>
```

La clave de escritura se establece al arrancar con `--api-key` o `MATRIXAI_API_KEY`.

**Scope de solo lectura (PR5-C6):** se puede proporcionar una segunda clave con `--api-key-read` o `MATRIXAI_API_KEY_READ`. Da acceso a los endpoints de lectura (`GET /api/v1/registry`, `/api/v1/predict`, `/api/v1/registry/{name}/{version}/verify`, `/api/v1/registry/{name}/{version}/pull`) pero es rechazada en los endpoints de escritura (push, tag, execute-action, feedback).

Los endpoints marcados como **públicos** no requieren autenticación.

---

## Límite de peticiones

El servidor de producción aplica un límite de peticiones por IP con ventana deslizante (por defecto: 60 peticiones/minuto). Configurable con `--rate-limit` al arrancar o con la variable `MATRIXAI_RATE_LIMIT`. Las peticiones que superen el límite devuelven `429 Too Many Requests` con la cabecera `Retry-After: 60`.

---

## Formato de respuesta común

Los endpoints versionados `/api/v1/*` usan un envelope consistente:

```json
{ "ok": true, ...payload }
{ "ok": false, "error": "descripción del error", "code": "CODIGO_ERROR" }
```

El campo `code` es un string legible por máquina (ej. `NOT_FOUND`, `UNAUTHORIZED`, `REGISTRY_NOT_LOADED`) para manejo programático de errores.

Las rutas legacy sin versionar (`/predict`, `/health`, etc.) preservan su formato de respuesta original por compatibilidad y no incluyen `code`.

Códigos de estado HTTP:

| Código | Significado |
|--------|------------|
| 200 | Éxito |
| 400 | Petición malformada (JSON inválido) |
| 401 | API key ausente o inválida |
| 404 | Recurso no encontrado |
| 422 | Error de validación o procesamiento |
| 429 | Límite de peticiones superado |
| 500 | Error interno del servidor |

---

## Servidor de producción

Puerto por defecto: `8000`. Se arranca con `matrixai serve <modelo.mxai> --params <params.json>`.

### GET /health

**Auth:** Público

Devuelve el estado del servidor y métricas básicas.

```json
{
  "status": "ok",
  "service": "MatrixAI Server",
  "backend": "numpy",
  "metrics": {
    "requests_total": 142,
    "requests_successful": 140,
    "requests_failed": 2,
    "uptime_seconds": 3600
  }
}
```

---

### GET /metrics

**Auth:** Público

Devuelve las métricas del servidor en formato de exposición Prometheus (`text/plain; version=0.0.4`). Compatible con scraping de Prometheus.

Métricas del núcleo (siempre presentes):

| Métrica | Tipo | Descripción |
|---------|------|-------------|
| `matrixai_requests_total` | Counter | Total de peticiones recibidas |
| `matrixai_requests_successful` | Counter | Peticiones que devolvieron 2xx |
| `matrixai_requests_failed` | Counter | Peticiones que devolvieron 4xx/5xx |
| `matrixai_requests_rate_limited` | Counter | Peticiones rechazadas por el limitador |
| `matrixai_items_processed` | Counter | Ítems procesados (los lotes se cuentan individualmente) |
| `matrixai_last_request_duration_milliseconds` | Gauge | Latencia de la última petición |
| `matrixai_uptime_seconds` | Gauge | Tiempo de actividad del servidor en segundos |

Métricas de drift (presentes solo cuando la monitorización continua está activa):

| Métrica | Tipo | Descripción |
|---------|------|-------------|
| `matrixai_drift_window_accuracy` | Gauge | Precisión en la ventana de monitorización actual |
| `matrixai_drift_window_samples` | Gauge | Número de muestras de feedback en la ventana actual |
| `matrixai_drift_degradation_detected` | Gauge | `1.0` si se supera el umbral de drift, `0.0` en caso contrario |
| `matrixai_drift_actual_degradation` | Gauge | Magnitud de la caída de precisión respecto a la referencia |

Todas las métricas incluyen la etiqueta `{project="<nombre_del_modelo>"}`.

---

### GET /docs

**Auth:** Público

Devuelve la interfaz Swagger UI para exploración interactiva de la API.

---

### GET /openapi.json

**Auth:** Público

Devuelve la especificación OpenAPI 3.0.0 generada a partir del programa `.mxai` cargado. Incluye el esquema de entrada derivado de la definición VECTOR del modelo.

---

### POST /predict

**Auth:** Requerida

Ejecuta la inferencia sobre el modelo cargado. Acepta un único objeto de entrada o un lote (array de objetos). Los campos de entrada deben coincidir con el esquema VECTOR del modelo.

**Petición — objeto único:**
```json
{ "age": 35, "income": 52000, "credit_history": "good" }
```

**Petición — lote:**
```json
[
  { "age": 35, "income": 52000, "credit_history": "good" },
  { "age": 28, "income": 31000, "credit_history": "fair" }
]
```

**Respuesta:**
```json
{
  "ok": true,
  "result": 0.9998245440617306,
  "model": "CreditScoring",
  "parameter_set": "v1.0_best"
}
```

La respuesta en lote devuelve un array de resultados en el mismo orden que la entrada.

---

### POST /execute-action

**Auth:** Requerida

Ejecuta una acción real bajo un contrato `.mxact` cargado. La ejecución se firma, audita y registra.

**Petición:**
```json
{
  "contract_name": "CreditDecision",
  "input_data": { "age": 35, "income": 52000, "credit_history": "good" },
  "model_hash": "mxai_20d8ce3f...",
  "parameter_set_id": "v1.0_best"
}
```

**Respuesta:**
```json
{
  "ok": true,
  "report_id": "rpt-20260528-abc123",
  "model_hash": "mxai_20d8ce3f...",
  "parameter_set_id": "v1.0_best",
  "action_contract_hash": "sha256:...",
  "input_hash": "sha256:...",
  "executed_at": "2026-05-28T14:00:00",
  "executor_kind": "real",
  "ok_action": true,
  "response_summary": "Crédito aprobado",
  "error": null,
  "latency_ms": 3.2,
  "hmac_signature": "..."
}
```

Devuelve `422` si la validación del contrato falla o la acción no está permitida.

---

### POST /feedback

**Auth:** Requerida

Registra el valor real (ground truth) para una predicción anterior. Usado por el monitor de aprendizaje continuo (P22) para seguir la deriva de precisión a lo largo del tiempo.

**Petición:**
```json
{
  "prediction": "1",
  "ground_truth": "0",
  "trace_id": "trace-abc123",
  "observed_at": "2026-05-28T15:00:00",
  "parameter_set_id": "v1.0_best"
}
```

**Respuesta:**
```json
{
  "ok": true,
  "recorded": true,
  "correct": false,
  "trace_id": "trace-abc123"
}
```

Devuelve `404` si no hay ningún monitor continuo cargado en el servidor.

---

### OPTIONS *

**Auth:** Público

Gestor de preflight CORS. Devuelve `204 No Content` con las cabeceras `Access-Control-Allow-*`. Controlado por `--cors-origin` al arrancar o la variable `MATRIXAI_CORS_ORIGINS`.

---

## API versionada — /api/v1/

Todas las rutas del servidor de producción están disponibles también bajo `/api/v1/` con el schema de error consistente. Las rutas legacy sin versionar se mantienen como aliases compatibles.

Para habilitar el registry HTTP, arrancar el servidor con `--registry PATH`.

### GET /api/v1/health · GET /api/v1/metrics

**Auth:** Público. Mismo comportamiento que sus equivalentes sin versionar; respuesta envuelta en `{ok: true, ...}`.

---

### POST /api/v1/predict

**Auth:** Requerida (clave de escritura o lectura). Devuelve `{"ok": true, "result": {...}}` en éxito; `{"ok": false, "error": "...", "code": "..."}` en error.

### POST /api/v1/execute-action

**Auth:** Requerida (**solo clave de escritura**). Devuelve el payload ActionTrace (incluye campo `"ok"` que representa el resultado de la acción, no el envelope HTTP).

### POST /api/v1/feedback

**Auth:** Requerida (**solo clave de escritura** — feedback muta el monitor de drift). Devuelve `{"ok": true, ...}` en éxito; `{"ok": false, "error", "code"}` en error.

---

### GET /api/v1/registry

**Auth:** Requerida (clave de lectura o escritura). Requiere `--registry PATH` al arrancar.

Lista todos los modelos del registry. Soporta paginación.

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `page` | `1` | Número de página (base 1) |
| `limit` | `20` | Elementos por página (máx. 100) |
| `name` | — | Filtrar por nombre de modelo |

**Respuesta:**
```json
{
  "ok": true,
  "models": [
    {
      "name": "credit-scoring",
      "version": "v1.0",
      "matrixai_version": "1.0.0",
      "metrics": { "accuracy": 0.933 },
      "created_at": "2026-05-30T10:00:00+00:00"
    }
  ],
  "page": 1,
  "limit": 20,
  "total": 1
}
```

---

### GET /api/v1/registry/{name}/{version}

**Auth:** Requerida (clave de lectura o escritura).

**Respuesta:**
```json
{
  "ok": true,
  "model": {
    "name": "credit-scoring",
    "version": "v1.0",
    "matrixai_version": "1.0.0",
    "entry_hash": "sha256:...",
    "model_hash": "sha256:...",
    "parameter_set_id": "ps_v1",
    "metrics": { "accuracy": 0.933 },
    "created_at": "2026-05-30T10:00:00+00:00",
    "interpretability_level": "full"
  }
}
```

Devuelve `404 NOT_FOUND` si la entrada no existe.

---

### POST /api/v1/registry/{name}/{version}/predict

**Auth:** Requerida (clave de lectura o escritura).

Ejecuta inferencia sobre un modelo cargado del registry. Mismo formato de petición que `POST /predict`.

**Respuesta:** `{ "ok": true, "result": { ... } }`

---

### POST /api/v1/registry/{name}/{version}/verify

**Auth:** Requerida (clave de lectura o escritura).

Verifica la integridad y firma de una entrada del registry. Emite un aviso si la versión mayor difiere de la versión en ejecución.

**Respuesta:** `{ "ok": true, "verified": true, "warnings": [] }`

Devuelve `409 INTEGRITY_MISMATCH` si la verificación falla.

---

### POST /api/v1/registry/push

**Auth:** Requerida (**solo clave de escritura**).

Registra un modelo desde un directorio de entrenamiento en el sistema de ficheros del servidor.

**Petición:**
```json
{ "name": "credit-scoring", "version": "v1.1", "run_dir": "/ruta/al/runs/v1" }
```

**Respuesta:** `{ "ok": true, "name": "credit-scoring", "version": "v1.1" }` — `201 Created`

Devuelve `409 DUPLICATE_ENTRY` si la versión ya existe (el registry es append-only).

---

### POST /api/v1/registry/{name}/tag/{tag}

**Auth:** Requerida (**solo clave de escritura**).

**Petición:** `{ "version": "v1.1" }`

**Respuesta:** `{ "ok": true, "name": "credit-scoring", "tag": "latest", "version": "v1.1" }`

---

## Servidor Studio

Puerto por defecto: `8080`. Se arranca con `matrixai playground`. Todos los endpoints son públicos — no exponer este servidor externamente.

### GET /

**Devuelve la interfaz Studio.** Entorno de desarrollo de modelos en el navegador.

---

### GET /expert

Devuelve la interfaz del Workbench experto para la edición avanzada de `.mxai`.

---

### GET /api/defaults

Devuelve el prompt por defecto y los ejemplos de entrada que se muestran al cargar el Studio.

```json
{
  "ok": true,
  "prompt": "...",
  "input_json": "{ ... }",
  "examples": [
    { "id": "credit-scoring", "label": "Scoring de Crédito" },
    { "id": "fall-risk", "label": "Evaluación de Riesgo de Caídas" }
  ]
}
```

---

### GET /api/example/{id}

Carga un paquete de ejemplo completo por ID. Devuelve todos los artefactos del ejemplo: texto `.mxai`, contrato de entrenamiento, entrada de muestra, manifiesto e informe de evaluación.

```json
{
  "ok": true,
  "id": "credit-scoring",
  "label": "Scoring de Crédito",
  "mode": "prompt",
  "mxai_text": "...",
  "training_text": "...",
  "input_json": "...",
  "manifest_text": "...",
  "evaluation_report_text": "..."
}
```

Devuelve `404` si el ID del ejemplo es desconocido.

---

### POST /api/analyze

Analiza y valida un programa `.mxai`, o ejecuta el pipeline completo `prompt → .semantic → .mxai`.

**Petición:**
```json
{
  "mode": "prompt",
  "prompt": "Un modelo que predice la aprobación de crédito según edad, ingresos e historial crediticio",
  "input_json": "{ \"age\": 35 }",
  "use_llm": false
}
```

Modos: `"prompt"` | `"semantic"` | `"mxai"`.

**Respuesta:**
```json
{
  "ok": true,
  "mode": "prompt",
  "accepted": true,
  "mxai": "...",
  "semantic_text": "...",
  "checks": [...],
  "pipeline_stages": [...],
  "artifacts": { ... }
}
```

Devuelve `422` si la validación falla, con `checks` listando los errores específicos.

---

### POST /api/generate-training

Genera un contrato `.mxtrain` y una plantilla de dataset a partir de un programa `.mxai`.

**Petición:**
```json
{ "mxai_text": "PROGRAM CreditScoring ..." }
```

**Respuesta:**
```json
{
  "ok": true,
  "training_text": "TRAINING CreditScoring ...",
  "dataset_template_text": "age,income,credit_history,label\n...",
  "warnings": [],
  "source": "generated"
}
```

---

### POST /api/generate-dataset

Genera un dataset de entrenamiento sintético a partir de una especificación `.mxai` + `.mxtrain`. Máximo 5000 filas / 1 MB.

**Petición:**
```json
{
  "mxai_text": "...",
  "training_text": "...",
  "rows": 500,
  "seed": 42,
  "mode": "coherent"
}
```

Modos: `"random"` | `"coherent"` (coherente genera datos consistentes con la semántica del modelo).

**Respuesta:**
```json
{
  "ok": true,
  "csv_text": "age,income,...\n...",
  "rows": 500,
  "seed": 42,
  "mode": "coherent",
  "fingerprint": "sha256:...",
  "columns": ["age", "income", "credit_history"],
  "labels": ["approved", "rejected"]
}
```

---

### POST /api/validate-csv

Valida que un fichero CSV coincide con la especificación de entrenamiento. Máximo 5000 filas / 1 MB.

**Petición:**
```json
{
  "mxai_text": "...",
  "training_text": "...",
  "csv_text": "age,income,...\n..."
}
```

**Respuesta:**
```json
{
  "ok": true,
  "rows": 450,
  "warnings": ["La columna 'income' tiene 3 valores ausentes"]
}
```

---

### POST /api/train

Entrenamiento síncrono. Tiempo máximo de 30 segundos y 200 épocas. Para entrenamientos más largos usar `/api/train-start`.

**Petición:**
```json
{
  "mxai_text": "...",
  "training_text": "...",
  "csv_text": "...",
  "epochs_override": 100
}
```

**Respuesta:**
```json
{
  "ok": true,
  "task_kind": "classification",
  "run_id": "run-abc123",
  "best_epoch": 87,
  "best_validation_loss": 0.112,
  "final_train_loss": 0.098,
  "accuracy": 0.923,
  "mae": null,
  "rmse": null,
  "r2": null,
  "backend": "numpy",
  "epochs": [...],
  "params_best": { ... }
}
```

---

### POST /api/train-start

Inicia un trabajo de entrenamiento asíncrono. Devuelve inmediatamente un `job_id` para consultar el estado.

**Petición:** Igual que `/api/train`.

**Respuesta:**
```json
{ "ok": true, "job_id": "job-20260528-xyz789" }
```

---

### GET /api/train-status/{job_id}

Consulta el estado de un trabajo de entrenamiento asíncrono.

```json
{
  "ok": true,
  "job_id": "job-20260528-xyz789",
  "status": "done",
  "epochs": [...],
  "accuracy": 0.923,
  "best_epoch": 87
}
```

Valores de `status`: `"running"` | `"done"` | `"error"` | `"cancelled"` | `"timeout"`.

Devuelve `404` si el ID del trabajo es desconocido.

---

### POST /api/train-cancel

Cancela un trabajo de entrenamiento en ejecución.

**Petición:**
```json
{ "job_id": "job-20260528-xyz789" }
```

**Respuesta:**
```json
{ "ok": true, "job_id": "job-20260528-xyz789", "status": "cancelled" }
```

---

### POST /api/run-with-params

Ejecuta el modelo con un conjunto específico de parámetros cargados, sin arrancar un servidor.

**Petición:**
```json
{
  "mxai_text": "...",
  "params_json": "{ \"weights\": [...] }",
  "input_json": "{ \"age\": 35, \"income\": 52000 }"
}
```

**Respuesta:**
```json
{ "ok": true, "result": { "approved": 0.9998 } }
```

---

### POST /api/refine

Refina un prompt a partir de un resultado de auditoría. Implementa el bucle de refinamiento iterativo (P13).

**Petición:**
```json
{
  "prompt": "Un modelo que predice la aprobación de crédito...",
  "run_result": { ... },
  "mxai_text": "...",
  "hints": "Mejorar el recall para solicitantes con ingresos bajos",
  "iteration_count": 1,
  "refinement_chain": [...],
  "parent_prompt_hash": "sha256:...",
  "max_iterations": 5
}
```

**Respuesta:**
```json
{
  "ok": true,
  "refinement_id": "rfn-abc123",
  "mode": "supervised",
  "iteration": 2,
  "supervision_accepted": true,
  "chain": [...],
  "parent_hash": "sha256:...",
  "proposed_prompt": "Un modelo que predice la aprobación de crédito...",
  "explanation": "Ajuste de ponderación de características para sensibilidad a ingresos"
}
```

---


Devuelve las capacidades del Studio y el modo LLM actual.

```json
{
  "ok": true,
  "llm_mode": {
    "active": true,
    "provider": "proveedor-externo",
    "model": "modelo-chat-externo"
  },
  "capabilities": { ... },
  "degradation_messages": [],
  "optional_unavailable": [],
  "production_steps": [...]
}
```

---


Lista todos los casos guiados disponibles en el Studio.

```json
{
  "ok": true,
  "cases": [
    { "id": "credit-scoring", "title": "Scoring de Crédito", "sector": "finanzas", ... },
    { "id": "fall-risk", "title": "Evaluación de Riesgo de Caídas", "sector": "sanidad", ... }
  ]
}
```

---


Obtiene la definición completa de un caso guiado.

```json
{
  "ok": true,
  "case": {
    "id": "credit-scoring",
    "title": "Scoring de Crédito",
    "sector": "finanzas",
    "inputs": [...],
    "expected_outputs": [...],
    "description": "..."
  }
}
```

Devuelve `404` si el ID del caso es desconocido.

---


Simula un caso guiado con valores de entrada específicos.

**Petición:**
```json
{
  "case_id": "credit-scoring",
  "input_values": { "age": 35, "income": 52000, "credit_history": "good" },
  "free_text": "Énfasis en el riesgo de impago"
}
```

**Respuesta:**
```json
{
  "ok": true,
  "case_id": "credit-scoring",
  "executive_result": {
    "summary": "...",
    "recommendation": "...",
    "confidence": 0.92,
    "audit_trail": [...]
  }
}
```

---


Genera un modelo a partir de un prompt, opcionalmente guiado por una plantilla de caso.

**Petición:**
```json
{
  "prompt": "Un modelo que predice el riesgo de reingreso hospitalario en 30 días",
  "case_id": "readmission-risk",
  "input_json": "{ \"age\": 72, \"diagnosis\": \"ICC\" }"
}
```

**Respuesta:**
```json
{
  "ok": true,
  "model_origin": "prompt",
  "pipeline_ok": true,
  "understanding": "...",
  "executive_result": { ... },
  "mxai": "PROGRAM ReadmissionRisk ...",
  "pipeline_stages": [...],
  "semantic_text": "..."
}
```

---


Ejecuta un análisis ejecutivo sobre un programa `.mxai` existente con las entradas indicadas.

**Petición:**
```json
{
  "mxai": "PROGRAM CreditScoring ...",
  "input_json": "{ \"age\": 35, \"income\": 52000 }",
  "mxai_name": "CreditScoring"
}
```

**Respuesta:**
```json
{
  "ok": true,
  "model_origin": "existing",
  "pipeline_ok": true,
  "executive_result": { ... },
  "mxai_name": "CreditScoring",
  "pipeline_stages": [...],
  "error": null
}
```

---

## Resumen de endpoints

### Servidor de producción (puerto 8000)

Las rutas siguientes son los alias sin versión actualmente implementados. PR5-C6
publicará la superficie versionada `/api/v1/` y mantendrá estos alias por
compatibilidad hacia atrás.

| Método | Ruta | Auth | Descripción |
|--------|------|------|-------------|
| GET | `/health` | Público | Estado del servidor y métricas |
| GET | `/metrics` | Público | Métricas en formato Prometheus |
| GET | `/docs` | Público | Interfaz Swagger UI |
| GET | `/openapi.json` | Público | Especificación OpenAPI 3.0.0 |
| POST | `/predict` | Requerida | Ejecutar inferencia del modelo |
| POST | `/execute-action` | Requerida | Ejecutar acción real auditada |
| POST | `/feedback` | Requerida | Registrar ground truth para monitorización de drift |
| OPTIONS | `*` | Público | Preflight CORS |

### Servidor Studio (puerto 8080)

| Método | Ruta | Auth | Descripción |
|--------|------|------|-------------|
| GET | `/` | Público | Interfaz Studio |
| GET | `/expert` | Público | Interfaz Workbench experto |
| GET | `/api/defaults` | Público | Prompt y ejemplos por defecto |
| GET | `/api/example/{id}` | Público | Cargar paquete de ejemplo |
| GET | `/api/train-status/{job_id}` | Público | Consultar trabajo de entrenamiento asíncrono |
| POST | `/api/analyze` | Público | Analizar/validar `.mxai` o ejecutar pipeline de prompt |
| POST | `/api/generate-training` | Público | Generar contrato de entrenamiento |
| POST | `/api/generate-dataset` | Público | Generar dataset sintético |
| POST | `/api/validate-csv` | Público | Validar CSV contra la especificación |
| POST | `/api/train` | Público | Entrenamiento síncrono |
| POST | `/api/train-start` | Público | Iniciar trabajo de entrenamiento asíncrono |
| POST | `/api/train-cancel` | Público | Cancelar trabajo de entrenamiento |
| POST | `/api/run-with-params` | Público | Ejecutar modelo con parámetros |
| POST | `/api/refine` | Público | Refinamiento iterativo de prompt |

---

