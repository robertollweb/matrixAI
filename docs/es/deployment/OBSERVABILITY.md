# PR4-C2 — Guía de Observabilidad

> **English:** [docs/en/deployment/OBSERVABILITY.md](../../en/deployment/OBSERVABILITY.md)

MatrixAI expone un endpoint `GET /metrics` en [formato Prometheus](https://prometheus.io/docs/instrumenting/exposition_formats/) para que cualquier scraper estándar (Prometheus, Grafana Agent, OpenTelemetry Collector) pueda recoger métricas de salud del servidor y de drift sin modificar la aplicación.

---

## Endpoint

```
GET /metrics
```

- **Autenticación:** no requerida (igual que `/health`)
- **Content-Type:** `text/plain; version=0.0.4; charset=utf-8`
- **Siempre disponible** independientemente de si hay modelo cargado o acciones habilitadas

Smoke test rápido:

```bash
curl http://localhost:8000/metrics
```

Salida esperada (truncada):

```
# HELP matrixai_requests_total Total HTTP requests received.
# TYPE matrixai_requests_total counter
matrixai_requests_total{project="CreditScoring"} 42
# HELP matrixai_uptime_seconds Server uptime in seconds.
# TYPE matrixai_uptime_seconds gauge
matrixai_uptime_seconds{project="CreditScoring"} 317.5
...
```

---

## Referencia de métricas

### Identidad del modelo (siempre emitida)

| Métrica | Tipo | Descripción |
|---|---|---|
| `matrixai_model_info` | gauge | Siempre 1; las labels llevan `project`, `parameter_set_id` y `model_hash` |

### Salud del servidor (siempre emitidas)

| Métrica | Tipo | Descripción |
|---|---|---|
| `matrixai_requests_total` | counter | Total de peticiones HTTP en todos los endpoints |
| `matrixai_requests_successful_total` | counter | Peticiones que devolvieron 200 |
| `matrixai_requests_failed_total` | counter | Peticiones que devolvieron 4xx/5xx |
| `matrixai_requests_rate_limited_total` | counter | Peticiones rechazadas por rate limiting (429) |
| `matrixai_items_processed_total` | counter | Predicciones individuales (batches contados por ítem) |
| `matrixai_last_request_duration_milliseconds` | gauge | Tiempo de la última petición en milisegundos |
| `matrixai_uptime_seconds` | gauge | Tiempo de actividad del servidor |

Todas las métricas llevan una label `project` con el nombre del `PROJECT` del fichero `.mxai`.

### Métricas de acciones P20 (siempre emitidas)

| Métrica | Tipo | Descripción |
|---|---|---|
| `matrixai_action_executions_total` | counter | Total de llamadas a `POST /execute-action` procesadas |
| `matrixai_action_dry_runs_total` | counter | Simulaciones dry-run ejecutadas (una por llamada, incluyendo fallidas) |
| `matrixai_action_signed_total` | counter | ActionTraces firmadas con `MATRIXAI_ACTION_SIGNING_KEY` |

### Métricas de drift P22 (emitidas cuando hay ProductionMonitor activo)

| Métrica | Tipo | Descripción |
|---|---|---|
| `matrixai_drift_window_accuracy` | gauge | Accuracy en la ventana deslizante (0–1) |
| `matrixai_drift_window_samples` | gauge | Observaciones etiquetadas en la ventana actual |
| `matrixai_drift_degradation_detected` | gauge | 1 si se detectó degradación de accuracy, 0 si no |
| `matrixai_drift_actual_degradation` | gauge | Caída de accuracy respecto al valor de referencia (positivo = peor) |

Las métricas de drift requieren un `ProductionMonitor` conectado al servidor (ver [Conectar métricas de drift](#conectar-métricas-de-drift) más abajo).

---

## Integración con Prometheus

### 1. Configuración de scrape en `prometheus.yml`

```yaml
scrape_configs:
  - job_name: matrixai
    scrape_interval: 15s
    static_configs:
      - targets: ["matrixai-server:8000"]
    metrics_path: /metrics
```

### 2. Docker Compose con Prometheus

Añadir un servicio Prometheus al `docker-compose.yml` generado:

```yaml
services:
  matrixai-server:
    build: .
    ports:
      - "${PORT:-8000}:8000"
    env_file: .env
    healthcheck:
      test: ["CMD", "python3", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
    restart: unless-stopped

volumes:
  matrixai-registry:
```

Guardar la configuración del paso 1 como `prometheus.yml` junto al `docker-compose.yml` y luego:

```bash
docker compose up --build -d
# UI de Prometheus → http://localhost:9090
# Query: matrixai_requests_total
```

### 3. Dashboard de Grafana (inicio rápido)

```bash
# Añadir Prometheus como fuente de datos en Grafana y luego consultar:
rate(matrixai_requests_total[1m])          # tasa de peticiones
matrixai_uptime_seconds                    # tiempo de actividad
matrixai_drift_degradation_detected == 1  # alerta de drift
```

---

## Conectar métricas de drift

Las métricas de drift solo se emiten cuando hay un `ProductionMonitor` (P22) conectado al servidor. Esto requiere feedback de ground-truth por predicción.

Para registrar feedback y exponer métricas de drift en producción:

**Paso 1 — Arrancar el servidor con una política continual:**

```bash
matrixai serve model.mxai \
  --params params.json \
  --continual-policy policy/alert_monitor.mxcontinual
# Continual monitoring active (policy: AlertMonitorContinual, reference_accuracy: 0.9100)
# POST /feedback  (Record ground truth for drift monitoring)
```

La `reference_accuracy` se lee automáticamente de `--params metrics.accuracy`. Para forzar un valor concreto: `--reference-accuracy 0.91`. Sin referencia, `matrixai_drift_degradation_detected` siempre vale 0.

**Paso 2 — Enviar feedback de ground truth:**

```bash
curl -s -X POST \
  -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prediction": "high_risk", "ground_truth": "low_risk", "trace_id": "abc123"}' \
  http://localhost:8000/feedback
# {"recorded": true, "correct": false, "trace_id": "abc123"}
```

Una vez que haya suficientes observaciones (`MIN_SAMPLES_IN_WINDOW` en la política), `/metrics` incluirá los gauges de drift.

Cuando no hay monitor (comportamiento por defecto), las líneas de métricas de drift simplemente se omiten. Prometheus trata las métricas ausentes como stale/unknown, que es el comportamiento correcto.

---

## Notas de seguridad

- `/metrics` no requiere autenticación por diseño — los scrapers de Prometheus normalmente corren en el mismo segmento de red que el servidor.
- En despliegues expuestos a internet, restringir el acceso mediante un proxy inverso (bloque `location` de nginx, regla de firewall) en lugar de añadir auth al endpoint de scrape.
- Las labels de métricas incluyen el nombre `project` del modelo pero ningún dato de usuario, valor de predicción ni valor de parámetro.

---

## Guías relacionadas

| Guía | Contenido |
|---|---|
| [Deployment](DEPLOYMENT.md) | Empaquetar y desplegar un modelo con Docker |
| [Hardening del servidor](SERVER_HARDENING.md) | Rate limiting, CORS, autenticación por API key |
| [Rotación de claves](KEY_ROTATION.md) | Rotar signing keys sin invalidar trazas históricas |
