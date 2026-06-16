# PR4-C1 — Guía de Deployment Productivo

> **English:** [docs/en/deployment/DEPLOYMENT.md](../../en/deployment/DEPLOYMENT.md)

Esta guía lleva a un operador a través del despliegue de un modelo MatrixAI en un servidor de producción usando Docker y Docker Compose. El servidor corre en una máquina limpia sin instalación previa de MatrixAI; todo lo necesario queda incluido en el paquete generado por `matrixai pack`.

**Prerequisitos:** Docker ≥ 24, plugin Docker Compose (≥ v2), sin más requisitos.

---

## Paso 1 — Empaquetar el modelo

En la máquina de desarrollo (donde MatrixAI está instalado):

```bash
matrixai pack examples/credit-scoring/credit_scoring.mxai \
  --params examples/credit-scoring/registry/entries/credit-scoring/v1.0/params.json \
  --docker \
  --outdir dist/credit-scoring
```

Para incluir también un contrato de acción real:

```bash
matrixai pack examples/agent-alert/alert_monitor.mxai \
  --params examples/agent-alert/registry/entries/alert-monitor/v1.0/params.json \
  --contract examples/agent-alert/alert_notifier.mxact \
  --docker \
  --outdir dist/agent-alert
```

Esto produce un directorio autocontenido:

```
dist/credit-scoring/
  credit_scoring.mxai        # artefacto del modelo
  params.json                # parámetros entrenados
  matrixai/                  # fuente del framework (sin pip install)
  Dockerfile                 # imagen de producción
  docker-compose.yml         # stack compose
  .env.example               # variables de entorno documentadas
```

Transferir el directorio al servidor de producción (`scp -r`, `rsync`, o un container registry tras `docker build`).

---

## Paso 2 — Configurar el entorno

```bash
cd dist/credit-scoring
cp .env.example .env
$EDITOR .env
```

Cambio mínimo requerido — establecer una API key fuerte:

```
MATRIXAI_API_KEY=<string-aleatorio-fuerte>
```

Generar un valor seguro:

```bash
openssl rand -hex 32
```

**Todas las variables de entorno:**

| Variable | Requerida | Descripción |
|---|---|---|
| `MATRIXAI_API_KEY` | **Sí** | Bearer token para `POST /predict` y `POST /execute-action` |
| `MATRIXAI_ACTION_SIGNING_KEY` | Con acciones reales | Clave HMAC-SHA256 para firmas de `ActionTrace` (mínimo 32 bytes hex) |
| `MATRIXAI_ALLOW_REAL_ACTIONS` | No | `true` para habilitar `POST /execute-action` con efectos reales. Por defecto: `false` |
| `MATRIXAI_REGISTRY_SIGNING_KEY` | Recomendada | Clave HMAC para firmas de entradas del registry. Si no se establece, se genera una clave por contenedor (no persiste entre reinicios) |
| `MATRIXAI_RATE_LIMIT` | No | Máximo de peticiones/minuto por IP. Por defecto: `60`. Establecer a `0` para desactivar. |
| `MATRIXAI_CORS_ORIGINS` | No | Orígenes CORS permitidos (separados por comas). Por defecto: `*`. Restringir para clientes de navegador. |
| `PORT` | No | Puerto del host expuesto. Por defecto: `8000` |

**Notas de seguridad:**
- Nunca subir `.env` a control de versiones. Añadirlo a `.gitignore`.
- `MATRIXAI_ALLOW_REAL_ACTIONS=false` (por defecto) significa que todas las acciones son simuladas — seguro para staging y CI.
- Las signing keys deben tener mínimo 32 bytes hex (64 caracteres hex). Generar con `openssl rand -hex 32`.

---

## Paso 3 — Construir y arrancar

```bash
docker compose up --build -d
```

Salida esperada:

```
[+] Building ...
[+] Running 1/1
 ✔ Container credit-scoring-matrixai-server-1  Started
```

Verificar que el servidor está sano:

```bash
docker compose ps
# Estado debe mostrar: healthy

curl http://localhost:8000/health
# {"status": "ok", "service": "MatrixAI Server", ...}
```

---

## Paso 4 — Enviar una predicción

```bash
curl -s \
  -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"income_score": 0.7, "credit_history": 0.8, "debt_ratio": 0.2, "employment_years": 0.6, "loan_amount_ratio": 0.3}' \
  http://localhost:8000/predict
```

Ver la documentación de API autogenerada en el navegador:

```
http://localhost:8000/docs
```

---

## Paso 5 — Logs y métricas

```bash
# Ver logs en tiempo real
docker compose logs -f matrixai-server

# Consultar métricas de runtime
curl http://localhost:8000/health | python3 -m json.tool
```

Métricas expuestas en `/health`:

| Campo | Descripción |
|---|---|
| `requests_total` | Total de peticiones recibidas |
| `requests_successful` | Peticiones que devolvieron 200 |
| `requests_failed` | Peticiones que devolvieron 4xx/5xx |
| `requests_rate_limited` | Peticiones rechazadas por rate limiting (429) |
| `items_processed` | Predicciones individuales (batches contadas por ítem) |
| `last_request_ms` | Tiempo de pared de la última petición |
| `uptime_seconds` | Tiempo de actividad del servidor |

---

## Paso 6 — Parar y limpiar

```bash
# Parar sin eliminar datos
docker compose stop

# Parar y eliminar contenedores (conserva el volumen del registry)
docker compose down

# Eliminar todo incluyendo el volumen del registry
docker compose down -v
```

---

## Habilitar acciones reales (P20)

Para servir un modelo con contrato `.mxact` y permitir efectos reales:

1. Copiar el fichero `.mxact` al directorio dist.
2. Establecer en `.env`:
   ```
   MATRIXAI_ALLOW_REAL_ACTIONS=true
   MATRIXAI_ACTION_SIGNING_KEY=<openssl rand -hex 32>
   ```
3. Pasar la ruta del contrato al empaquetar (o montarlo en runtime y regenerar la imagen).
4. `docker compose up --build -d`

Sin `MATRIXAI_ALLOW_REAL_ACTIONS=true`, `POST /execute-action` simula todas las acciones — sin efectos reales independientemente del contenido del `.mxact`.

---

## Condiciones

- Docker: ≥ 24
- Docker Compose: plugin ≥ v2 (`docker compose` no `docker-compose`)
- Imagen base: `python:3.11-slim`
- Backend MatrixAI: stdlib (no requiere torch)
- Volumen del registry: volumen nombrado `matrixai-registry` montado en `/data/registry` dentro del contenedor

Ejecutar `matrixai pack --docker` en tu propio hardware para reproducir la build; el digest de la imagen diferirá pero el comportamiento es idéntico dado el mismo modelo y parámetros.

---

## Guías relacionadas

| Guía | Contenido |
|---|---|
| [Hardening del servidor](SERVER_HARDENING.md) | Rate limiting, CORS, auth X-API-Key, punto de extensión JWT, checklist de despliegue público |
| [Rotación de claves](KEY_ROTATION.md) | Generar, rotar y gestionar signing keys sin invalidar trazas históricas |
