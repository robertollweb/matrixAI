# C3 — Costo de Auditoría: Cuánto Cuesta la Capa de Auditoría

> [← Volver al índice de benchmarks](INDEX.md) · **English:** [docs/en/benchmarks/AUDIT_COST.md](../../en/benchmarks/AUDIT_COST.md)

**Reproducir:**
```bash
python3 benchmarks/audit_cost.py
```

**Entorno:** Python 3.12.3 · x86_64 Solo CPU · 200 repeticiones por medición

---

## Resultados

| Operación | Mediana | p95 | p99 | vs inferencia simple |
|---|---|---|---|---|
| Inferencia simple (sin auditoría) | 0.005 ms | 0.009 ms | 0.030 ms | baseline |
| Firma HMAC-SHA256 (`build_action_trace`) | 0.003 ms | 0.004 ms | 0.018 ms | ~60% de la inferencia |
| Dry-run (`DryRunSimulator.simulate`) | 0.014 ms | 0.021 ms | 0.057 ms | ~280% de la inferencia |
| Detección de tampering (`registry.verify`) | 0.171 ms | 0.275 ms | 0.387 ms | ~3420% de la inferencia |
| Versionado (`registry.push_run_dir`) | 0.991 ms | 2.015 ms | 2.015 ms | una vez por entrenamiento |

---

## Qué hace cada operación

**Inferencia simple:** `MatrixAIRuntime.run()` en un modelo sigmoid de 5 features. Recorre el grafo de cómputo, aplica sigmoid, devuelve el resultado. Sub-milisegundo.

**Firma HMAC:** Construye un dict canónico de todos los campos de `ActionTrace` (report_id, model_hash, parameter_set_id, contract_hash, input_hash, executed_at, executor_kind, ok, response_summary, latency_ms), serializa a bytes, computa HMAC-SHA256. Resultado: prueba criptográfica de qué ejecutó, cuándo, y con qué versión del modelo.

**Dry-run:** Simulación en memoria antes de cualquier acción real. Valida: destinatario en `allowed_recipients`, conteo de llamadas dentro de `rate_limit`, tipos de inputs coinciden con el esquema del contrato, contrato de rollback declarado. Las cuatro validaciones se completan en 0.014 ms de mediana.

**Detección de tampering:** `registry.verify()` lee `params.json` del disco, computa su SHA256, y compara contra el `params_content_hash` almacenado en el manifiesto. El I/O de filesystem es el costo dominante. Rango típico en almacenamiento local: 0.15–0.30 ms de mediana (depende del hardware; ejecuta el script para el número de tu máquina).

**Versionado (`push_run_dir`):** Copia el snapshot del modelo + params + reporte de evaluación al directorio de entrada del registro, computa SHA256 de params, escribe el manifiesto. Costo único por ejecución de entrenamiento.

---

## Costo total de una acción auditada

Una sola acción con auditoría completa (dry-run + ejecución + firma) cuesta:

```
dry-run:    0.014 ms
firma HMAC: 0.003 ms
────────────────────
total:      ~0.017 ms por acción auditada
```

Para **1,000 acciones/día**, el overhead total de auditoría es ~17 ms/día — sub-segundo.

---

## Qué obtienes por 0.017 ms

Cada acción auditada produce un `ActionTrace` que registra:

- `report_id` — identificador único de esta ejecución
- `model_hash` — versión exacta del modelo que produjo la decisión
- `parameter_set_id` — versión exacta de los parámetros
- `action_contract_hash` — hash del contrato declarado (scope, rate limits, rollback)
- `input_hash` — hash del input que disparó la acción
- `executed_at` — timestamp
- `hmac_signature` — HMAC-SHA256 sobre todos los campos anteriores

Cualquier modificación de la traza — incluso cambiar un bit — falla en `verify_action_trace()`. Reproducible meses o años después.

---

## La comparación de construirlo tú mismo

Para igualar esta capacidad de auditoría con un stack tradicional (FastAPI + SQLAlchemy + HMAC personalizado):

| Componente | LOC estimados |
|---|---|
| Esquema + serialización de ActionTrace | ~40 |
| Firma + verificación HMAC | ~30 |
| Simulación dry-run (scope, rate limit, tipos, rollback) | ~80 |
| Persistencia de audit log (esquema DB + ORM) | ~60 |
| Detección de tampering (hash + manifiesto) | ~50 |
| Esquema + validación de contratos | ~60 |
| Total | **~320 LOC que escribes, testeas y mantienes** |

MatrixAI provee todo esto en un `push_run_dir()` + `build_action_trace()` + `verify_action_trace()`.

---

## Condiciones

- Hardware: x86_64 CPU, sin GPU
- Python: 3.12.3
- Repeticiones: 200 por medición
- Modelo: `clinical_risk` (5 features, sigmoid)
- Tiempos reportados: mediana sobre 200 ejecuciones (no ejecuciones individuales)
