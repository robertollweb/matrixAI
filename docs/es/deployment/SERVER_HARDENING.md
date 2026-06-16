# PR4-C5 — Hardening del Servidor

> **English:** [docs/en/deployment/SERVER_HARDENING.md](../../en/deployment/SERVER_HARDENING.md)

El servidor HTTP de MatrixAI viene con valores seguros por defecto y controles de hardening
configurables: rate limiting por IP, autenticación por API key con un punto de extensión
listo para JWT, y CORS configurable para clientes de origen cruzado.

---

## Autenticación

El servidor requiere una API key secreta en todos los endpoints de escritura
(`/predict`, `/execute-action`).

### Proveer la API key

Establece `MATRIXAI_API_KEY` en `.env` (ver [DEPLOYMENT.md](DEPLOYMENT.md)):

```
MATRIXAI_API_KEY=<openssl rand -hex 32>
```

Si la variable de entorno no está establecida y no se pasa `--api-key`, el servidor genera
una clave aleatoria al arrancar y la imprime en stdout. Esto es seguro para desarrollo pero
**no para producción** — la clave cambia en cada reinicio, invalidando la configuración de
los clientes existentes.

### Enviar peticiones

Se aceptan dos formas de cabecera equivalentes:

```bash
# Bearer token (forma estándar OAuth2)
curl -H "Authorization: Bearer $MATRIXAI_API_KEY" http://host:8000/predict -d '...'

# X-API-Key (más simple para clientes no-OAuth)
curl -H "X-API-Key: $MATRIXAI_API_KEY" http://host:8000/predict -d '...'
```

### Punto de extensión JWT (para PR5-C6)

La comprobación de auth en `matrixai/server.py::MatrixAIServerHandler._check_auth` es el
único punto donde se toman decisiones de autenticación. Cuando PR5-C6 introduzca el Studio
externo, sustituye la comparación del bearer token por una llamada de validación JWT:

```python
# Extensión futura — NO añadir ahora; documentar intención de diseño
# if _validate_jwt(auth_header_value, public_key):
#     return True
```

La interfaz de API (`Authorization: Bearer <token>`) ya es compatible con JWT — no se
necesitan cambios en el cliente al pasar de clave estática a JWT.

### Endpoints públicos (sin auth requerida)

| Endpoint | Motivo |
|---|---|
| `GET /health` | Sonda de liveness — debe funcionar sin credenciales para balanceadores de carga y healthchecks de Docker |
| `GET /docs` | Swagger UI — solo informativo, sin acceso a datos |
| `GET /openapi.json` | Schema — solo informativo |
| `OPTIONS *` | Preflight CORS — los navegadores lo envían sin auth por especificación |

---

## Rate limiting

El rate limiting por IP de ventana deslizante evita abusos y protege el proceso de inferencia.

### Por defecto

60 peticiones/minuto por dirección IP, aplicado a `POST /predict` y `POST /execute-action`.

### Configuración

**Variable de entorno:**
```
MATRIXAI_RATE_LIMIT=60
```

**Flag CLI:**
```bash
matrixai serve model.mxai --rate-limit 30
```

Establece `0` para desactivar el rate limiting completamente (no recomendado para endpoints
públicos):
```bash
matrixai serve model.mxai --rate-limit 0
```

### Respuesta cuando se supera el límite

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

La cabecera `Retry-After: 60` indica a los clientes compatibles que esperen 60 segundos antes
de reintentar.

### Ajuste por endpoint

Override de límites por endpoint mediante variables de entorno:
```
MATRIXAI_RATE_LIMIT=60          # Por defecto para todos los endpoints
```

La ventana deslizante se resetea de forma natural — un cliente que envió 60 peticiones y
espera 60 segundos puede enviar 60 más. No hay mecanismo de reset explícito.

---

## CORS (Cross-Origin Resource Sharing)

Necesario cuando un cliente basado en navegador (como el futuro MatrixAI Studio) llama a la
API desde un origen diferente.

### Por defecto

```
Access-Control-Allow-Origin: *
```

El comodín permite cualquier origen. Es seguro cuando se aplica autenticación — las peticiones
sin autenticar son rechazadas independientemente del origen. Sin embargo, desactiva ciertas
características de seguridad del navegador (p.ej., las cookies con `SameSite=None` no pueden
enviarse a endpoints CORS con comodín).

### Restringir a orígenes específicos

**Variable de entorno (separada por comas):**
```
MATRIXAI_CORS_ORIGINS=https://studio.example.com,https://app.example.com
```

**Flag CLI (repetible):**
```bash
matrixai serve model.mxai \
  --cors-origin https://studio.example.com \
  --cors-origin https://app.example.com
```

Cuando se listan orígenes específicos, el servidor devuelve la cabecera `Origin` de la
petición si coincide, o el primer origen listado en caso contrario. Los orígenes que no
coincidan reciben una respuesta CORS que los navegadores bloquearán.

### Preflight CORS

El servidor gestiona las peticiones `OPTIONS` automáticamente. Los navegadores envían un
preflight `OPTIONS` antes de cualquier `POST` o `PUT` de origen cruzado. El servidor
devuelve:

```
HTTP/1.1 204 No Content
Access-Control-Allow-Origin: <origen>
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, X-API-Key
Access-Control-Max-Age: 86400
```

`Access-Control-Max-Age: 86400` (24 horas) cachea el resultado del preflight en navegadores
compatibles, reduciendo el overhead del preflight en llamadas repetidas.

---

## Resumen de valores seguros por defecto

| Característica | Por defecto | Override |
|---|---|---|
| Auth | Requerida en `/predict`, `/execute-action` | No se puede desactivar |
| Fuente de API key | Variable de entorno `MATRIXAI_API_KEY` → auto-generada | Flag `--api-key` |
| Acciones reales | Desactivadas | `--allow-real-actions` o `MATRIXAI_ALLOW_REAL_ACTIONS=1` |
| Rate limit | 60 peticiones/min por IP | `--rate-limit N` o `MATRIXAI_RATE_LIMIT=N` |
| CORS | `*` (todos los orígenes) | `--cors-origin ORIGIN` o `MATRIXAI_CORS_ORIGINS=...` |

---

## Lista de comprobación para despliegue público

- [ ] `MATRIXAI_API_KEY` establecida explícitamente en `.env` (no auto-generada)
- [ ] `MATRIXAI_ALLOW_REAL_ACTIONS` no establecida (acciones reales desactivadas por defecto)
- [ ] `MATRIXAI_RATE_LIMIT` revisada para el tráfico esperado
- [ ] `MATRIXAI_CORS_ORIGINS` restringida a los orígenes reales de los clientes si se esperan clientes basados en navegador
- [ ] Servidor expuesto a través de un proxy inverso con terminación TLS (nginx, Traefik, Caddy); el servidor MatrixAI no termina TLS
- [ ] Fichero de historial de claves (`matrixai_registry/.matrixai_key_history.json`) excluido del control de versiones y con backup
