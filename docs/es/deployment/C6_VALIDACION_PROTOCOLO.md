# PR4-C6 — Protocolo de Validación con Operador Externo

> **English:** [docs/en/deployment/C6_VALIDATION_PROTOCOL.md](../../en/deployment/C6_VALIDATION_PROTOCOL.md)

Este documento define el protocolo formal para cerrar PR4-C6. Un operador externo — no el autor, no alguien que haya construido el sistema — debe completar el ciclo operativo completo en una máquina limpia usando solo la documentación de MatrixAI. El autor observa y registra pero no ayuda.

---

## ¿Qué es MatrixAI?

MatrixAI es un framework open source para construir modelos de IA verificables, auditables y desplegables. Cubre el ciclo completo: definir modelos en ficheros `.mxai`, entrenar con parámetros versionados, servir vía HTTP y monitorizar drift en producción.

**Este documento es para operadores.** No necesitas entender los internos de MatrixAI — solo seguir los pasos de aquí y de las guías enlazadas.

---

## Primeros pasos — instalar MatrixAI (5 minutos)

**Requisitos:** Python 3.8+, Git, Docker ≥ 24.

```bash
# 1. Clonar el repositorio
git clone https://github.com/robertollweb/matrixAI.git
cd matrixAI

# 2. Verificar que el CLI funciona (sin instalación — se ejecuta desde el repo)
python3 -m matrixai --help
```

Si quieres entender los conceptos básicos de MatrixAI antes de empezar el ciclo C6, lee el quickstart primero (opcional — no es necesario para completar C6):

- [🇪🇸 Quickstart (5 min)](../QUICKSTART.md) — primer modelo, primera predicción
- [🇪🇸 Guía de deployment](DEPLOYMENT.md) — empaquetar y desplegar con Docker
- [🇪🇸 Guía de observabilidad](OBSERVABILITY.md) — `/metrics` y monitorización de drift
- [🇪🇸 Runbook operativo](RUNBOOK.md) — qué hacer cuando algo falla

---

## Perfil del operador

- Tiene experiencia operando servicios backend (Docker, curl, shell).
- Nunca ha usado MatrixAI antes.
- No tiene acceso al código fuente ni a la máquina del autor.
- Tiene acceso a internet (para hacer pull de imágenes Docker) y Docker ≥ 24 instalado.

---

## Pre-condiciones (el autor prepara, el operador no toca)

Antes de la sesión, el autor debe:

1. Ejecutar el smoke test en la máquina del operador (o compartir el paquete):
   ```bash
   python3 scripts/smoke_test_c6.py
   # Esperado: 28 OK | 0 FAIL
   ```
2. Empaquetar el modelo credit-scoring con política continual:
   ```bash
   matrixai pack examples/credit-scoring/credit_scoring.mxai \
     --params examples/credit-scoring/registry/entries/credit-scoring/v1.1/params.json \
     --docker \
     --outdir dist/credit-scoring-c6
   # Copiar dist/credit-scoring-c6/ y examples/credit-scoring/ a la máquina del operador
   ```
3. Transferir a la máquina del operador:
   - `dist/credit-scoring-c6/` (el paquete Docker)
   - `examples/credit-scoring/credit_scoring.mxcontinual` (política continual)
   - `examples/credit-scoring/registry/` (el registry del modelo)
   - Las guías: `DEPLOYMENT.md`, `OBSERVABILITY.md`, `RUNBOOK.md`

El operador recibe solo lo anterior. **Sin instalación de Python, sin código fuente.**

---

## Ciclo a completar

El operador debe completar las 6 fases en orden. El observador registra el tiempo y la fricción en cada fase.

### Fase 1 — Desplegar (est. 10 min)

**Objetivo:** servidor en ejecución y sano.

**Documentos:** [DEPLOYMENT.md](DEPLOYMENT.md)

```bash
cd dist/credit-scoring-c6
cp .env.example .env
# Editar .env: establecer MATRIXAI_API_KEY=<openssl rand -hex 32>
docker compose up --build -d
docker compose ps        # Estado debe mostrar: healthy
curl http://localhost:8000/health
```

**Criterio de paso:** `{"status": "ok"}` devuelto en menos de 30 segundos desde `docker compose up`.

---

### Fase 2 — Predicción (est. 5 min)

**Objetivo:** obtener una predicción válida del modelo.

**Documentos:** [DEPLOYMENT.md](DEPLOYMENT.md)

```bash
curl -s \
  -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"income_score":0.7,"credit_history":0.8,"debt_ratio":0.2,"employment_years":0.6,"loan_amount_ratio":0.3}' \
  http://localhost:8000/predict
```

**Criterio de paso:** la respuesta contiene un valor de predicción numérico.

---

### Fase 3 — Activar monitorización de drift (est. 5 min)

**Objetivo:** reiniciar el servidor con política continual para activar `/feedback` y las métricas de drift.

**Documentos:** [OBSERVABILITY.md](OBSERVABILITY.md)

El operador para el contenedor Docker y arranca el servidor directamente con `--continual-policy`:

```bash
docker compose stop

python3 -m matrixai serve \
  examples/credit-scoring/credit_scoring.mxai \
  --params examples/credit-scoring/registry/entries/credit-scoring/v1.1/params.json \
  --continual-policy examples/credit-scoring/credit_scoring.mxcontinual \
  --api-key $MATRIXAI_API_KEY \
  --host 0.0.0.0 --port 8000
# Esperado: "Continual monitoring active (policy: CreditScoringContinual, reference_accuracy: 0.8750)"
```

Verificar que las métricas de drift están disponibles:
```bash
curl http://localhost:8000/metrics | grep drift
# matrixai_drift_window_accuracy{...} 0.0
# matrixai_drift_degradation_detected{...} 0
```

**Criterio de paso:** `/metrics` contiene `matrixai_drift_degradation_detected`.

---

### Fase 4 — Inducir drift (est. 5 min)

**Objetivo:** enviar suficientes predicciones incorrectas para disparar la detección de degradación.

**Documentos:** [OBSERVABILITY.md](OBSERVABILITY.md)

El operador envía etiquetas de ground truth deliberadamente incorrectas para simular un modelo que ha dejado de funcionar:

```bash
for i in $(seq 1 6); do
  curl -s -X POST \
    -H "Authorization: Bearer $MATRIXAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"prediction\":\"1\",\"ground_truth\":\"0\",\"trace_id\":\"drift_$i\"}" \
    http://localhost:8000/feedback
done
```

Verificar que el drift fue detectado:
```bash
curl -s http://localhost:8000/metrics | grep drift_degradation_detected
# matrixai_drift_degradation_detected{...} 1
```

**Criterio de paso:** `matrixai_drift_degradation_detected` igual a 1 en `/metrics`.

---

### Fase 5 — Ejecutar rollback (est. 5 min)

**Objetivo:** revertir el modelo de v1.1 a v1.0 siguiendo el runbook.

**Documentos:** [RUNBOOK.md](RUNBOOK.md) — Escenario 1

```bash
# Dry-run primero
python3 -m matrixai continual rollback \
  examples/credit-scoring/credit_scoring.mxcontinual \
  --registry-dir examples/credit-scoring/registry \
  --dry-run
# Esperado: [dry-run] Would rollback credit-scoring
#             from: v1.1 → to: v1.0

# Ejecutar
python3 -m matrixai continual rollback \
  examples/credit-scoring/credit_scoring.mxcontinual \
  --registry-dir examples/credit-scoring/registry
# Esperado: Rolled back credit-scoring
#             from: v1.1 (ps=v1.1_best) → to: v1.0 (ps=v1.0_best)
```

**Criterio de paso:** el comando termina con código 0 y confirma `v1.1 → v1.0`.

---

### Fase 6 — Verificar rollback y comprobar estado (est. 5 min)

**Objetivo:** confirmar que el rollback se completó y el registry muestra v1.0 como versión activa.

**Documentos:** [RUNBOOK.md](RUNBOOK.md) — Escenario 2

```bash
python3 -m matrixai continual status \
  examples/credit-scoring/credit_scoring.mxcontinual \
  --registry-dir examples/credit-scoring/registry
# Esperado:
# Registry       : credit-scoring
# Current version: v1.0  (ps=v1.0_best)
# Last rollback  : v1.1 → v1.0  (manual)
#   executed_at  : 2026-...

python3 -m matrixai registry verify \
  --registry-path examples/credit-scoring/registry \
  credit-scoring@v1.0
# Esperado: OK: credit-scoring@v1.0 integrity verified
```

**Criterio de paso:** `Current version: v1.0` y `registry verify` pasa.

---

## Ficha de registro (el observador la rellena)

| Fase | Inicio | Fin | Pasó | Fricción / notas |
|---|---|---|---|---|
| 1 — Desplegar | | | ☐ | |
| 2 — Predicción | | | ☐ | |
| 3 — Activar monitorización | | | ☐ | |
| 4 — Inducir drift | | | ☐ | |
| 5 — Rollback | | | ☐ | |
| 6 — Verificar | | | ☐ | |

**El operador pidió ayuda:** sí / no  
**Consultó fuente externa:** sí / no (si sí: ¿cuál?)  
**Tiempo total:** _____ min

---

## Criterios de paso / fallo para cerrar C6

PR4-C6 **pasa** si:
- Las 6 fases completadas con ☐ marcado.
- El operador no pidió ayuda al autor en ningún momento.
- El operador no consultó ninguna fuente fuera de la documentación de MatrixAI.

PR4-C6 **falla** si:
- Alguna fase no se completa.
- El operador pidió ayuda al autor para resolver un bloqueo.
- El operador necesitó leer código fuente.

La **fricción registrada** (el operador se atascó pero lo resolvió solo con la documentación) se tolera y se anota — identifica gaps de documentación a resolver antes de PR5.

---

## Smoke test (el operador puede ejecutarlo antes de empezar)

```bash
python3 scripts/smoke_test_c6.py
# Esperado: 28 OK | 0 FAIL | Entorno listo para el ciclo C6.
```

---

## Guías relacionadas

| Guía | Cuándo usarla |
|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Fases 1–2 |
| [OBSERVABILITY.md](OBSERVABILITY.md) | Fases 3–4 |
| [RUNBOOK.md](RUNBOOK.md) | Fases 5–6 |
| [KEY_ROTATION.md](KEY_ROTATION.md) | Si hay que rotar una clave durante C6 |
