# MatrixAI — Benchmarks

> **English:** [docs/en/benchmarks/INDEX.md](../../en/benchmarks/INDEX.md)

Estos benchmarks miden dónde MatrixAI gana, dónde pierde, y cuánto cuesta su capa de auditoría — con scripts que puedes ejecutar tú mismo para verificar cada número.

**Reproducir cualquier benchmark:**
```bash
python3 benchmarks/training.py       # C2: velocidad de entrenamiento
python3 benchmarks/audit_cost.py     # C3: overhead de auditoría
python3 benchmarks/serving.py        # C1: throughput HTTP
```

**Entorno:** Python 3.12.3 · scikit-learn 1.8.0 · x86_64 · Solo CPU (sin GPU)  
Los números dependen del hardware. Ejecuta en tu propia máquina para resultados comparables.

---

## Benchmarks

| # | Qué se mide | Hallazgo clave |
|---|---|---|
| [C1 — Serving](SERVING.md) | Throughput HTTP y latencia | A concurrency=1: roughly comparable (dentro del ruido de ejecución). A concurrency≥10: claramente más rápido (baseline degrada a p99 > 1000ms bajo carga concurrente; magnitud varía por ejecución; causa raíz no instrumentada) |
| [C2 — Entrenamiento](TRAINING.md) | Tiempo de entrenamiento vs scikit-learn | ~4x más lento (red densa) hasta 25x o más (dataset pequeño, alta variabilidad LBFGS). sklearn usa solvers C/Fortran. MatrixAI agrega IR tipado, trazas y registro. |
| [C3 — Costo de auditoría](AUDIT_COST.md) | Costo de cada primitiva de auditoría | HMAC sign: 0.003 ms. Dry-run: 0.014 ms. Detección de tampering: 0.15–0.30 ms (depende del filesystem). Todo sub-milisegundo. |
| [C4 — Comparación funcional](FUNCTIONAL_COMPARISON.md) | Qué incluye MatrixAI vs qué construyes tú | ~320 LOC solo para la capa de auditoría; ~370 LOC para el stack completo incluyendo versionado de ParameterSet. |

---

## El resumen honesto

MatrixAI es **más lento para entrenar** y **ligeramente más lento por petición a baja concurrency** que las alternativas tradicionales optimizadas. Esto no es un bug — es el costo de la capa de auditoría.

Qué compra ese overhead:

| Capacidad | MatrixAI | sklearn+FastAPI |
|---|---|---|
| IR de modelo tipado (`.mxai`) | Incluido | Escribe tu propio esquema |
| `entry_hash` firmado (fingerprint del modelo) | Incluido | Escribe tu propio hashing |
| Traza de entrenamiento (hash de dataset + split + épocas) | Incluido | Escribe tu propio logging |
| ParameterSet (versionado, inspeccionable) | Incluido | Escribe tu propio versionado |
| Registro con detección de tampering | Incluido | MLflow o personalizado |
| Traza de ejecución por petición | Incluido | Escribe tu propia traza |
| Dry-run antes de acciones reales | Incluido | Escribe tu propio simulador |
| ActionTrace firmado con HMAC | Incluido | Escribe tu propio HMAC |
| Contrato de rollback | Incluido | Escribe tu propio contrato |
| OpenAPI / Swagger auto-generado | Incluido | FastAPI (rutas explícitas) |
| Auth Bearer | Incluido | Escribe tu propio middleware |

**Dónde MatrixAI pierde** (declarado explícitamente):
- Velocidad de entrenamiento: aproximadamente 4x más lento en redes densas, 10–25x o más en datasets tabulares pequeños (solvers C/Fortran vs SGD Python; el ratio exacto varía por ejecución debido a la variabilidad de convergencia LBFGS con N pequeño)
- Serving de petición única: roughly comparable a c=1 (dentro del ruido de ejecución); el overhead por petición de auth + traza + esquema es real pero pequeño en relación a la latencia de red
- Variedad de algoritmos: MatrixAI tiene ~10 patrones de modelos; sklearn tiene 100+ estimadores
- Madurez del ecosistema: sklearn/FastAPI tienen comunidades más grandes, más integraciones, más tutoriales
- Entrenamiento a escala GPU: MatrixAI no compite con PyTorch para deep learning a gran escala

**Para quiénes vale la pena el trade-off:**  
Industrias reguladas (financiera, salud, operaciones) donde debes probar — criptográficamente, reproduciblemente — qué versión de modelo produjo qué decisión, y que no fue alterada. La capa de auditoría no es overhead en ese contexto: es el requisito regulatorio.
