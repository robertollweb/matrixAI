# C2 — Benchmark de Entrenamiento: MatrixAI vs scikit-learn

> [← Volver al índice de benchmarks](INDEX.md) · **English:** [docs/en/benchmarks/TRAINING.md](../../en/benchmarks/TRAINING.md)

**Reproducir:**
```bash
python3 benchmarks/training.py
```

**Entorno:** Python 3.12.3 · scikit-learn 1.8.0 · x86_64 · 15.6 GB RAM · Solo CPU · Sin GPU · 5 repeticiones por pipeline · sklearn pre-importado (overhead de importación excluido)

---

## Resultados

| Pipeline | Filas | Features | MatrixAI (mediana) | sklearn (mediana) | Rango observado | Acc MatrixAI | Acc sklearn |
|---|---|---|---|---|---|---|---|
| Credit scoring | 120 | 5 | ~0.02 s | ~0.002 s | **~10–12x más lento** | 87.5% | 85.0% |
| Clinical risk | 60 | 5 | ~0.015 s | ~0.001 s | **~5–25x más lento** | 100.0% | 89.6% |
| Agent alert | 30 | 3 | ~0.018 s | ~0.001 s | **~15–20x más lento** | 100.0% | 83.3% |
| Text routing (Dense 8) | 45 | 30 (BoW) | ~0.10 s | ~0.024 s | **~4x más lento** | 100.0% | 100.0% |

Los números son medianas sobre 5 repeticiones, sklearn pre-importado. "Rango observado" refleja múltiples ejecuciones en este hardware — el tiempo de convergencia LBFGS de sklearn es muy sensible al tamaño del dataset con N pequeño, lo que hace que los ratios de clinical-risk y agent-alert sean los más volátiles. Ejecuta el script para ver los números específicos de tu hardware (`results_training.json`).

---

## Qué significan los números

**Todos los casos: sklearn entrena más rápido.** Sus solvers (LBFGS, liblinear) están escritos en Fortran/C optimizado y convergen en pocas iteraciones en estos datasets pequeños. MatrixAI ejecuta épocas SGD completas sobre mini-batches — más iteraciones para la misma convergencia.

**Por qué MatrixAI es más preciso en 3 de 4 casos:** LBFGS minimiza el loss de entrenamiento más agresivamente, pero con solo 30–120 filas y sin ajuste de regularización, puede sobreajustar más que un SGD calibrado. La diferencia de accuracy es específica del dataset; no es una afirmación general.

**Text routing (4.3x):** El MLP de sklearn usa Adam (optimizador adaptativo) que converge más rápido que SGD vanilla en redes densas. MatrixAI ejecuta 60 épocas SGD completas; Adam de sklearn converge antes. El MLP también alcanzó su límite de 200 iteraciones (ConvergenceWarning) pero igualmente convergió.

---

## Qué agrega MatrixAI sobre el entrenamiento de sklearn

| | MatrixAI | sklearn |
|---|---|---|
| IR tipado | Artefacto declarativo `.mxai` | Código Python |
| Fingerprint del modelo | `entry_hash` (SHA256 de modelo+params) | Ninguno |
| Traza de entrenamiento | hash de dataset + seed de split + número de épocas | Ninguno |
| ParameterSet | Versionado, inspeccionable, comparable con diff | pickle de joblib |
| Push al registro | Una llamada a `push_run_dir()` | MLflow o manual |
| Detección de tampering | `registry.verify()` | Ninguno |

---

## La conclusión honesta

**scikit-learn entrena más rápido en todos los casos.** Sus solvers C/Fortran (LBFGS, liblinear) convergen en menos iteraciones que SGD Python en datasets tabulares pequeños. La brecha es aproximadamente 4x en redes densas y va de 10x a 25x o más en datasets muy pequeños — los ratios exactos varían por ejecución debido a la variabilidad de convergencia de LBFGS con N pequeño.

**MatrixAI entrena más lento porque hace más.** Cada ejecución de entrenamiento produce un artefacto tipado, un ParameterSet versionado, una traza de entrenamiento con hashes reproducibles, y se registra en una llamada. En entornos regulados, este overhead es el valor — no el costo.

---

## Condiciones

- Hardware: x86_64 CPU, sin GPU, 15.6 GB RAM
- Python: 3.12.3
- scikit-learn: 1.8.0
- Backend MatrixAI: stdlib (sin torch)
- Repeticiones: 5 por pipeline; se reporta la mediana (peor run individual disponible como max_s en results_training.json)
- sklearn pre-importado antes del timing para excluir el overhead de importación
- Épocas: 40 (credit), 50 (clinical), 80 (alert), 60 (routing)
- Solver sklearn: LBFGS (LogisticRegression), Adam/max_iter=200 (MLPClassifier)
