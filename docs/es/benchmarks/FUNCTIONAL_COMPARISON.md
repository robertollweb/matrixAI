# C4 — Comparación Funcional: Qué Incluye MatrixAI vs Qué Construyes Tú

> [← Volver al índice de benchmarks](INDEX.md) · **English:** [docs/en/benchmarks/FUNCTIONAL_COMPARISON.md](../../en/benchmarks/FUNCTIONAL_COMPARISON.md)

Esto no es un benchmark de velocidad. Responde una pregunta diferente: **¿cuánto costaría construir la misma capa de auditoría y gobernanza tú mismo?**

---

## La tabla

| Capacidad | MatrixAI | Stack tradicional (FastAPI + sklearn + personalizado) |
|---|---|---|
| IR de modelo tipado (`.mxai`) | Incluido | Escribe tu propio esquema |
| `entry_hash` firmado (fingerprint del modelo) | Incluido | Escribe tu propio hashing |
| Traza de entrenamiento (hash dataset + split + épocas) | Incluido | Escribe tu propio logging |
| ParameterSet (versionado, inspeccionable, comparable) | Incluido | pickle de joblib |
| Registro con detección de tampering | Incluido | MLflow o personalizado |
| Traza de ejecución por petición | Incluido | Escribe tu propia traza |
| Dry-run antes de acciones reales | Incluido | Escribe tu propio simulador |
| ActionTrace firmado con HMAC | Incluido | Escribe tu propio HMAC |
| Contrato de rollback | Incluido | Escribe tu propio contrato |
| OpenAPI / Swagger auto-generado | Incluido | FastAPI (rutas explícitas) |
| Auth Bearer | Incluido | Escribe tu propio middleware |

---

## Estimación de LOC para igualar la capa de auditoría

| Componente | LOC estimados |
|---|---|
| Esquema + serialización de ActionTrace | ~40 |
| Firma + verificación HMAC | ~30 |
| Simulación dry-run (scope, rate limit, tipos, rollback) | ~80 |
| Persistencia de audit log (esquema DB + ORM) | ~60 |
| Detección de tampering (hash + manifiesto) | ~50 |
| Esquema + validación de contratos | ~60 |
| Versionado de ParameterSet + diff | ~50 |
| **Total** | **~370 LOC que escribes, testeas y mantienes** |

MatrixAI entrega todo esto a través de tres llamadas: `push_run_dir()`, `build_action_trace()`, `verify_action_trace()`.

---

## Qué NO reemplaza MatrixAI

Esta tabla es obligatoria. MatrixAI tiene brechas reales:

| Dimensión | Stack tradicional | MatrixAI |
|---|---|---|
| Variedad de algoritmos | sklearn: 100+ estimadores, 20+ transformers | ~10 patrones de modelos (lineal, denso, compuesto) |
| Velocidad de entrenamiento (modelos simples) | LBFGS / liblinear (C/Fortran) | SGD en Python — ~4x más lento (red densa) hasta 25x+ (tabular pequeño, variabilidad LBFGS) |
| Integraciones del ecosistema | Extensas (MLflow, DVC, Airflow, Ray, etc.) | Mínimas |
| Tamaño de comunidad | sklearn: 10+ años, millones de usuarios | Etapa temprana |
| Entrenamiento deep learning a escala GPU | PyTorch / TensorFlow | No competitivo |
| Python arbitrario en modelos | Python completo, cualquier librería | Solo IR declarativo — sin código arbitrario |
| Exploración interactiva | Jupyter notebooks, visualización rica | Solo CLI hoy |

---

## La conclusión honesta

MatrixAI intercambia variedad de algoritmos y amplitud del ecosistema por una capa de auditoría integrada que los entornos regulados de otro modo construirían ellos mismos.

**Para quiénes vale la pena el trade-off:**  
Industrias donde debes probar — criptográficamente, reproduciblemente — qué versión de modelo produjo qué decisión, y que no fue alterada. Servicios financieros, salud, sistemas de operaciones. En esos contextos, la infraestructura de auditoría de ~370 LOC no es opcional: es el requisito de cumplimiento.

**Para quiénes no vale la pena:**  
Entornos de investigación, exploración de data science, equipos que necesitan la librería completa de estimadores de sklearn, proyectos donde la interpretabilidad y gobernanza de modelos no son requisitos regulatorios.

---

## Condiciones

- Estimaciones de LOC: basadas en una implementación FastAPI + SQLAlchemy + HMAC personalizado que iguala la capa de auditoría de MatrixAI feature por feature
- Las estimaciones son de orden de magnitud. La implementación real varía según las convenciones del equipo y las elecciones de framework.
