# PR4-C3 — Runbook Operativo

> **English:** [docs/en/deployment/RUNBOOK.md](../../en/deployment/RUNBOOK.md)

Este runbook cubre los seis escenarios de fallo que un operador puede encontrar ejecutando MatrixAI en producción. Cada escenario sigue la misma estructura: **Síntoma → Diagnóstico → Acción → Verificación**. El operador debe poder resolver cualquier escenario usando solo este documento y el CLI de MatrixAI, sin contactar al autor.

**Requisitos:** CLI `matrixai` disponible; acceso al directorio del registry y al fichero de política `.mxcontinual` si aplica.

---

## Escenario 1 — Drift detectado

### Síntoma

Uno o más de:
- `/metrics` muestra `matrixai_drift_degradation_detected{...} 1`
- `/metrics` muestra `matrixai_drift_actual_degradation{...} > 0`
- El log de monitorización continual reporta un trigger de rollback
- La accuracy en la ventana deslizante ha caído por debajo del umbral configurado

```bash
curl -s http://localhost:8000/metrics | grep drift
# matrixai_drift_degradation_detected{project="AlertMonitor"} 1
# matrixai_drift_actual_degradation{project="AlertMonitor"} 0.12
```

### Diagnóstico

El `ProductionMonitor` detectó que la accuracy en la ventana actual ha caído más de `DEGRADATION_THRESHOLD` por debajo de la accuracy de referencia. Esto activa la condición de drift de la política.

Comprobar el estado de la política continual para confirmar las versiones actuales y de referencia:

```bash
matrixai continual status policy/alert_monitor.mxcontinual \
  --registry-dir registry/
# Registry       : alert-monitor
# Current version: v1.1  (ps=run_20260527_best)
# Base version   : v1.0  (ps=train_out_best)
# Rollback config: threshold=0.1  window=1h  min_samples=100
# Metrics (training):
#   accuracy: 0.91
# Drift status   : see GET /metrics (server must run with --continual-policy)
```

Para datos de drift en vivo (accuracy de ventana, degradación detectada), consultar el servidor en ejecución:

```bash
curl -s http://localhost:8000/metrics | grep matrixai_drift
# matrixai_drift_degradation_detected{project="AlertMonitor"} 1
# matrixai_drift_window_accuracy{project="AlertMonitor"} 0.71
# matrixai_drift_actual_degradation{project="AlertMonitor"} 0.12
```

### Acción

> **Nota:** El servidor de MatrixAI no ejecuta rollbacks automáticamente. El rollback siempre se dispara manualmente desde el CLI tras confirmar el drift en `/metrics`. Un pipeline de rollback automatizado se puede construir ejecutando este comando en respuesta a una alerta de Prometheus.

1. Dry-run primero para confirmar qué se va a revertir:

```bash
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/ \
  --dry-run
# Would roll back alert-monitor from v1.1 → v1.0
# from_ps: run_20260527_best  →  to_ps: train_out_best
```

2. Ejecutar el rollback (requiere `MATRIXAI_CONTINUAL_SIGNING_KEY` configurada, o pasar `--signing-key`):

```bash
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/ \
  --json
```

3. Reiniciar el servidor apuntando a la versión revertida:

```bash
# Parar el contenedor en ejecución
docker compose stop

# Actualizar .env para apuntar a los params revertidos si es necesario
# Luego reiniciar
docker compose up -d
```

### Verificación

```bash
# Confirmar que el evento de rollback se registró
matrixai continual status policy/alert_monitor.mxcontinual --registry-dir registry/
# Current version: v1.0  (ps=train_out_best)
# Last rollback  : v1.1 → v1.0  (manual)
#   executed_at  : 2026-05-28T12:34:56+00:00
#   event        : rollback_20260528T123456...

# Confirmar que /metrics ya no muestra degradación
curl -s http://localhost:8000/metrics | grep drift_degradation_detected
# matrixai_drift_degradation_detected{...} 0
```

---

## Escenario 2 — Rollback ejecutado: verificar que se completó correctamente

### Síntoma

Se ejecutó un rollback (automático o manual). El operador necesita confirmar que se completó sin errores y que el parameter set correcto está activo.

### Diagnóstico

Los eventos de rollback son objetos `RollbackEvent` firmados y persistidos en el log de auditoría continual. Un rollback incompleto deja el servidor en la versión antigua.

Comprobar que la versión actual coincide con el objetivo del rollback:

```bash
matrixai continual status policy/alert_monitor.mxcontinual \
  --registry-dir registry/
# Current version: v1.0  (ps=train_out_best)   ← revertido desde v1.1
```

### Acción

Si `Current version` coincide con el objetivo del rollback, no se requiere acción — ir a Verificación.

Si la versión no ha cambiado o el comando devuelve error, ver **Escenario 3**.

### Verificación

```bash
# 1. Confirmar que la versión activa del registry coincide con el objetivo del rollback
matrixai registry show alert-monitor@v1.0 --registry-path registry/
# parameter_set_id: train_out_best  ← coincide con el objetivo esperado

# 2. Verificar la firma del evento de rollback
matrixai continual status policy/alert_monitor.mxcontinual \
  --registry-dir registry/ --json | python3 -m json.tool | grep signature
# "signature": "hmac-sha256:..."  ← no vacío significa firmado

# 3. Confirmar que el servidor sirviendo cargó los params correctos
curl -s -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  http://localhost:8000/health | python3 -m json.tool
```

---

## Escenario 3 — Rollback fallido

### Síntoma

```
RollbackResult.ok = false
Error: rollback_failed — no previous version found
```
o
```
Error: rollback_failed — registry entry not found: alert-monitor@v0.9
```
o el comando rollback termina con código de salida distinto de cero.

### Diagnóstico

Causas posibles:

| Causa | Indicador |
|---|---|
| No hay versión anterior en el registry | `registry list` muestra solo una versión |
| La versión objetivo fue eliminada | `registry show alert-monitor@v0.9` falla |
| La versión base de la política no coincide con el registry | `base_version` en la política no se encuentra |
| Clave de firma no configurada | `Error: MATRIXAI_CONTINUAL_SIGNING_KEY not set` |

```bash
matrixai registry list --registry-path registry/ --json
# Comprobar: ¿existe la versión objetivo de rollback?
```

### Acción

**Caso A — No hay versión anterior disponible:**

El modelo no tiene una versión anterior entrenada en el registry. Opciones:

1. Re-entrenar con datos históricos estables y publicar una nueva versión:
```bash
matrixai train model.mxai --training train.mxtrain --output runs/recovery
matrixai registry push runs/recovery/ \
  --name alert-monitor --version v0.9 --registry-path registry/
```
2. A continuación, re-ejecutar el rollback.

**Caso B — Clave de firma no configurada:**

```bash
export MATRIXAI_CONTINUAL_SIGNING_KEY=$(openssl rand -hex 32)
# O cargar desde el gestor de secretos
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/
```

**Caso C — Inconsistencia de base_version en la política:**

Editar el fichero `.mxcontinual` para establecer `BASE_VERSION` a una versión existente del registry, luego reintentar.

### Verificación

```bash
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/ --dry-run
# Debe mostrar: "Would roll back ... → v<objetivo>" sin error
```

---

## Escenario 4 — Registry corrupto

### Síntoma

```bash
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# FAIL: params.json content hash mismatch for alert-monitor@v1.0
```
o
```
FAIL: entry_hash mismatch
FAIL: signature invalid
```

### Diagnóstico

El registry es append-only y verificado criptográficamente. Cualquier discrepancia significa uno de:

| Error | Causa |
|---|---|
| `params.json content hash mismatch` | `params.json` fue editado directamente en disco |
| `entry_hash mismatch` | `manifest.json` fue modificado tras la firma |
| `signature invalid` | Clave rotada sin re-firmar, o fichero manipulado |
| `model.mxai hash mismatch` | Fichero de modelo reemplazado tras el registro |

Ejecutar verificación completa de todas las entradas:

```bash
for entry in $(matrixai registry list --registry-path registry/ --json \
  | python3 -c "import sys,json; [print(e['name']+'@'+e['version']) for e in json.load(sys.stdin)]")
do
  matrixai registry verify --registry-path registry/ "$entry"
done
```

### Acción

**No modificar los ficheros del registry manualmente** — esto invalida la cadena de auditoría.

**Recuperación desde backup (recomendado):**

```bash
# Parar el servidor
docker compose stop

# Restaurar registry desde el último backup conocido como válido
cp -r /backup/registry-YYYYMMDD/ registry/

# Re-verificar
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# OK: alert-monitor@v1.0 integrity verified
```

**Recuperación re-registrando (si no hay backup):**

```bash
# Re-publicar la entrada desde los artefactos del run de entrenamiento original
matrixai registry push runs/v1.0/ \
  --name alert-monitor \
  --version v1.0 \
  --registry-path registry/

# La entrada corrupta debe eliminarse primero — parar el servidor,
# eliminar entries/alert-monitor/v1.0/, borrar su entrada del índice en
# registry/registry.json, luego re-publicar.
```

### Verificación

```bash
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# OK: alert-monitor@v1.0 integrity verified

# Reiniciar el servidor
docker compose up -d
curl -s http://localhost:8000/health | python3 -m json.tool
# "status": "ok"
```

---

## Escenario 5 — Clave de firma perdida o comprometida

### Síntoma

Uno de:
- El servidor falla al firmar ActionTraces (`MATRIXAI_ACTION_SIGNING_KEY not set`)
- `matrixai audit-action` devuelve `signature_valid: false` para trazas recientes
- La clave fue eliminada accidentalmente o el contenedor se reinició sin clave persistente

### Diagnóstico

**Clave perdida (eliminada o no persistida):**
Las ActionTraces recientes firmadas con la clave perdida no pueden re-verificarse. Las trazas firmadas antes de la pérdida no se ven afectadas si la clave fue registrada previamente con `matrixai keys rotate`.

**Clave comprometida (filtrada):**
Un atacante con la clave podría falsificar firmas de ActionTrace. Rotar inmediatamente.

Comprobar el historial de claves:

```bash
matrixai keys list --registry-path registry/
# fingerprint  purpose   added_at                     rotated_at
# sha256:ab12  action    2026-05-01T10:00:00+00:00    —         (activa)
# sha256:cd34  action    2026-04-01T09:00:00+00:00    2026-05-01T10:00:00+00:00
```

### Acción

**Paso 1 — Retirar la clave perdida/comprometida:**

```bash
# Si el valor de la clave es conocido (caso comprometida):
matrixai keys rotate --purpose action \
  --key <clave-hex-antigua> \
  --registry-path registry/
# Key sha256:<fp> retired at 2026-05-28T...

# Si la clave está realmente perdida (no se puede pasar --key), nunca fue registrada.
# No se puede retirar, pero las nuevas trazas usarán la nueva clave.
```

**Paso 2 — Generar y configurar la nueva clave** (NO llamar a `keys rotate` con la nueva clave):

```bash
NEW_KEY=$(openssl rand -hex 32)
echo "MATRIXAI_ACTION_SIGNING_KEY=$NEW_KEY" >> .env
# La nueva clave quedará registrada en el historial automáticamente cuando se rote en el futuro.
```

**Paso 3 — Reiniciar el servidor:**

```bash
docker compose stop
docker compose up -d
```

**Paso 4 — Evaluar el impacto:**

```bash
# Identificar trazas que pueden no ser verificables
# (las firmadas con la clave perdida, si nunca fue registrada)
matrixai audit-action trace_2026-05-27.json \
  --signing-key $NEW_KEY
# Si signature_valid: false, esta traza no puede re-verificarse con la nueva clave.
# Debe re-auditarse contra la clave antigua si se recupera.
```

### Verificación

```bash
# Confirmar que la nueva clave está activa
matrixai keys list --registry-path registry/
# sha256:<nueva_fp>  action  2026-05-28T...  — (activa)

# Confirmar que las nuevas trazas se firman correctamente
matrixai audit-action nueva_traza.json
# signature_valid: true
```

**Qué se preserva tras la pérdida de clave:**
- Todas las trazas firmadas antes de la pérdida permanecen en disco — su contenido está intacto.
- La verificación de firmas de esas trazas requiere la clave antigua. Si fue registrada en el historial antes de la pérdida, `matrixai audit-action` la encontrará automáticamente.
- La cadena de auditoría (qué se decidió, cuándo y en qué estado estaba el modelo) se preserva independientemente.

---

## Escenario 6 — Tamper detection en producción

### Síntoma

Uno de:
- `matrixai registry verify` devuelve `FAIL: entry_hash mismatch` o `FAIL: signature invalid`
- `matrixai audit-action` devuelve `signature_valid: false` para una traza no modificada
- El servidor registra un error inesperado de firma o verificación
- Un parameter set fue cargado pero su `model_hash` no coincide con el modelo actual

### Diagnóstico

La detección de manipulaciones de MatrixAI opera en tres niveles:

| Nivel | Qué protege | Cómo verificar |
|---|---|---|
| Entradas del registry | `params.json`, `model.mxai`, `manifest.json` | `matrixai registry verify` |
| ActionTraces | Contenido de la traza (predicción, acción, timestamp) | `matrixai audit-action <traza>` |
| ParameterSet | Identidad del modelo via `model_hash` | `matrixai validate-parameters` |

```bash
# Comprobar integridad del registry
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# FAIL: entry_hash mismatch ← el contenido fue modificado tras la firma

# Comprobar integridad del parameter set
matrixai validate-parameters model.mxai --params params.json
# model_hash mismatch: expected mxai_20d8ce..., got mxai_61d33f...
```

### Acción

**No servir un modelo manipulado.** Parar el servidor primero.

```bash
docker compose stop
```

**Identificar el alcance:**

```bash
# Verificar todas las entradas del registry
for entry in $(matrixai registry list --registry-path registry/ --json \
  | python3 -c "import sys,json; [print(e['name']+'@'+e['version']) for e in json.load(sys.stdin)]")
do
  matrixai registry verify --registry-path registry/ "$entry"
done

# Comprobar la cadena de ActionTrace para trazas recientes
matrixai audit-action traza_reciente.json
```

**Restaurar desde backup:**

```bash
cp -r /backup/registry-YYYYMMDD/ registry/
cp /backup/params-YYYYMMDD.json params.json
```

**Si no hay backup disponible:**
Re-entrenar el modelo desde el dataset original y re-registrar. Los artefactos manipulados deben descartarse — no intentar parchear campos individuales.

### Verificación

```bash
# Todas las entradas deben pasar
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# OK: alert-monitor@v1.0 integrity verified

matrixai validate-parameters model.mxai --params params.json
# ParameterSet validation: ok

# Reiniciar el servidor
docker compose up -d
curl -s http://localhost:8000/health | python3 -m json.tool
# "status": "ok"
```

**Registrar el incidente.** Documentar qué fue manipulado, cuándo se detectó y qué se restauró. Si es un sistema en producción, tratarlo como un incidente de seguridad y notificar al proceso de respuesta a incidentes.

---

## Referencia rápida

| Síntoma | Escenario | Primer comando |
|---|---|---|
| `drift_degradation_detected 1` | 1 — Drift | `matrixai continual status policy.mxcontinual` |
| Confirmar rollback completado | 2 — Verificar rollback | `matrixai continual status policy.mxcontinual --json` |
| `rollback_failed` en logs | 3 — Rollback fallido | `matrixai registry list --registry-path registry/` |
| `registry verify` FAIL | 4 — Registry corrupto | Bucle de verify completo + restaurar desde backup |
| Trazas sin firmar / `signature_valid: false` | 5 — Clave perdida | `matrixai keys list --registry-path registry/` |
| `entry_hash mismatch` / fallos inesperados de verify | 6 — Tamper | Parar servidor + `matrixai registry verify` todas las entradas |

---

## Guías relacionadas

| Guía | Contenido |
|---|---|
| [Deployment](DEPLOYMENT.md) | Empaquetar y desplegar con Docker |
| [Observabilidad](OBSERVABILITY.md) | Prometheus `/metrics`, métricas de drift |
| [Rotación de claves](KEY_ROTATION.md) | Rotar signing keys sin invalidar historial |
