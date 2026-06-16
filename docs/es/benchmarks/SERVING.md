# C1 — Benchmark de Serving: MatrixAI HTTP vs ThreadingHTTPServer + sklearn

> [← Volver al índice de benchmarks](INDEX.md) · **English:** [docs/en/benchmarks/SERVING.md](../../en/benchmarks/SERVING.md)

**Reproducir:**
```bash
python3 benchmarks/serving.py
```

**Entorno:** Python 3.12.3 · sklearn 1.8.0 · x86_64 · 15.6 GB RAM · 100 peticiones por nivel de concurrencia

---

## Resultados

| Concurrencia | MatrixAI req/s | Baseline req/s | Ganador | Ratio | MatrixAI p50 |
|---|---|---|---|---|---|
| 1 | 1,013 | 1,114 | ~comparable | dentro del ruido de ejecución | 0.7 ms |
| 5 | 1,989 | 1,829 | MatrixAI | **1.1x más rápido** | 2.5 ms |
| 10 | 3,065 | 666 | MatrixAI | **claramente más rápido** (varía por ejecución) | 3.2 ms |
| 20 | 5,247 | 807 | MatrixAI | **claramente más rápido** (varía por ejecución) | 3.5 ms |

Los números son promedios de tiempo de pared sobre 100 peticiones por nivel con 5 peticiones de warmup. Ejecuta en tu propia máquina para resultados comparables — los números absolutos varían por hardware.

---

## El baseline es un rival competente

Este benchmark usa **ThreadingHTTPServer + sklearn LogisticRegression** como baseline — no un stub de un solo hilo. El baseline:

- Usa el mismo modelo de threading que MatrixAI (`ThreadingHTTPServer`)
- Entrena un `LogisticRegression(LBFGS)` real sobre el dataset de credit-scoring al arrancar
- Sirve predicciones reales via `predict_proba()`, no pesos hardcodeados

Lo que **no** tiene:
- Sin autenticación Bearer token
- Sin validación de tipos de input ni esquema
- Sin traza de ejecución por petición ni hash del modelo
- Sin versionado, registro ni detección de tampering
- Sin documentación OpenAPI / Swagger

A c=1 ambos servidores están dentro del ruido de ejecución a ejecución. El overhead por petición de auth + traza + esquema es real pero pequeño en relación al costo de ida y vuelta de red a baja concurrencia.

---

## Qué significan los números

**c=1 (roughly comparable):** Ambos servidores están dentro del ruido de ejecución a ejecución. Cada petición de MatrixAI paga por auth, validación de input, registro de traza y hash del modelo — el baseline omite todo esto. A concurrencia de una sola petición, ese overhead es pequeño en relación al costo de ida y vuelta de red.

**c=5 (MatrixAI 1.1x más rápido):** MatrixAI empieza a adelantarse mientras su thread pool amortiza el overhead por petición entre workers paralelos.

**c=10 y c=20 (MatrixAI claramente más rápido):** El p99 del baseline sube a ~1,000 ms bajo carga concurrente, mientras MatrixAI mantiene latencia consistente (p99 ≤ 10 ms a c=20). La degradación observada en el baseline — throughput cayendo de 1,829 req/s a c=5 a 666 req/s a c=10 — sugiere serialización a nivel de threads bajo cargas concurrentes de Python. La magnitud de la ventaja varía por ejecución y hardware; el script no instrumenta la causa raíz.

---

## Qué obtienes por el overhead de petición de MatrixAI

Cada petición de MatrixAI:

1. Valida el Bearer token (comparación de string O(1))
2. Parsea y valida el input contra el esquema tipado del modelo
3. Ejecuta el grafo de cómputo a través de `MatrixAIRuntime.run()`
4. Registra una traza de ejecución (hash del modelo, hash del input, timestamp, latencia)
5. Devuelve una respuesta tipada con la predicción

El baseline solo hace el paso 3 — con un modelo sklearn, no un artefacto versionado y vinculado al registro.

---

## Condiciones

- Hardware: x86_64 CPU, sin GPU, 15.6 GB RAM
- Python: 3.12.3
- sklearn: 1.8.0
- Modelo: `credit_scoring` (5 features, sigmoid, backend stdlib)
- Baseline: `ThreadingHTTPServer` + `sklearn.linear_model.LogisticRegression(solver=lbfgs)` entrenado sobre el mismo dataset
- Niveles de concurrencia: 1, 5, 10, 20
- Peticiones por nivel: 100 (con warmup de 5 peticiones)
