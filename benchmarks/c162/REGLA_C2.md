# Contrato 162 · C2 — REGLA DE DECISIÓN (para SELLAR ANTES de medir): TabM con MÁS CAPACIDAD (k = 32 miembros) en el cubo PEQUEÑO

Fija, antes de que exista ninguna cifra de la medida real, qué se mide, cómo se compara y cuándo la palanca «más capacidad» de TabM (idea 39,
Ola 5; **D4: k = 32 SOLO en el cubo pequeño**) cumple. Es la regla del C1 (`REGLA_C1.md`, que a su vez es la del 160 con la decisión D3 de Roberto)
adaptada a los **13 conjuntos pequeños**. Su sha256 se graba en `analiza_c162_c2.REGLA_SHA256_SELLADA` y en la salida de la medida
(`regla_sha256_al_medir`); `medir_c162_c2.py` se niega a medir y `analiza_c162_c2.py` a dar veredicto si el fichero ya no es el sellado.
**Cambiar una coma después de sellar es cambiar la regla**: una enmienda nueva con su sello, no una edición.

## Lo que se decidió al adaptar la regla (dicho ANTES de sellar)

- **(b) = 4, no 5.** El C1 pide 5 mejoras claras de 17 no saturados (29,4 %). Con **12 no saturados** la misma proporción da 3,53; **redondeada hacia
  arriba, 4**. Una prueba (`test_la_proporcion_de_b_es_la_del_c1_redondeada_hacia_arriba`) recalcula el 4 desde el 5 y el 17.
- **(d) = 3,5 ×, no 1,5 ×.** k = 32 son 4 veces más miembros, pero el coste por época NO es 4 veces: la preparación, los embeddings PLR y los lotes
  son compartidos y solo escalan con k las capas del ensamblado. Sonda (`.agentes/ola5-39/c2/sonda/coste_por_epoca.log`, 3 conjuntos pequeños
  reales, 4 hilos, ajuste de 20 épocas, con la máquina en **carga 10–12**: los **tiempos son orientativos, ESTIMADOS, no medidos en limpio**; la
  memoria sí es fiable): k = 32 costó **×1,3** (kc2), **×2,7** (balance-scale) y **×2,7** (us_crime, el ancho: 100 columnas; por época ×3,1 con la
  pendiente 4→20 épocas) el tiempo del ajuste de k = 8; la memoria pico, **×1,21 / ×1,21 / ×1,42** (us_crime: 644 → 913 MB). Con un tope de **3,0 ×** un
  conjunto ancho y largo como us_crime ya rozaría el límite por sí solo, y el corte quedaría en NO CUMPLE por coste con una palanca buena; se fija
  **3,5 ×**: por encima de lo observado (3,1 por época en el peor) y por debajo de las 4 veces que serían «cuatro modelos». Esto es un umbral de COSTE
  fijado con la sonda, ANTES de sellar y sin ninguna cifra de calidad a la vista. **El plazo de pared de 120 s por intento es además un techo**: ningún
  intento puede costar más de ~120 s + predecir (micro-mass, 1.300 columnas, ya para por plazo con k = 8: su razón será ≈ 1). Se informan la mediana y
  el máximo por conjunto y la memoria pico de cada brazo (la memoria se declara, no decide).
- **`d_block` = 512 NO se mide.** La palanca del motor lo admite (`con_capacidad(d_block=512)`, apagada), pero este corte mide **solo k** (una palanca por
  medida, y D4 solo fija k = 32). Con lo medido no se justifica meterlo: en la sonda solo sube el coste (solo, ×1,06–1,09 de memoria y ×1,2–2,5 de tiempo
  del ajuste; junto a k = 32, ×1,5 de memoria) y no hay ninguna cifra de calidad que lo respalde. Si el 162 quiere probarlo, es una medida y una regla aparte.
- **La parada es la de siempre (por la PÉRDIDA) en los dos brazos.** La parada por métrica es la palanca del C1; mezclarla haría imposible saber qué
  mejora. Un intento que declare `parada_por_metrica` es SIN VEREDICTO.

## 1. Qué se mide

- **La palanca.** `MotorDensaTabM().con_capacidad(k=32)` (matrixai-engines, 162-C2): la red del ensamblado tiene 32 miembros en vez de 8; el
  `d_block` (256), `n_blocks` (2), el lote, la tasa, la paciencia y el plazo son los mismos. Apagada por omisión y bit a bit igual que el commit
  `72db6fc` cuando está apagada (pruebas con nombre). El predictor la declara (`arquitectura.k` = 32 y `entrenamiento_efectivo.capacidad`).
- **Los dos brazos** (cada uno, un subproceso aislado con su tope, UNA vez por conjunto y pliegue, la referencia PRIMERO y la palanca inmediatamente
  después): **`tabm_k8`** (TabM `ref` tal y como entrega hoy) y **`tabm_k32`** (la palanca). Ambos se miden en ESTA pasada con el MISMO código: no se
  reutiliza nada del 155, del 160 ni del C1. Mismos datos preparados, misma semilla del intento, mismo presupuesto (120 s de pared), 4 hilos.
  Con k = 32 la época cuesta más, así que **a igual plazo hace menos épocas**: es parte de lo que se mide (el coste (d) y los cortes por plazo se
  declaran), no un defecto del arnés.
- **Conjuntos:** los **13** conjuntos de cubo `pequeno` entre los 23 no sellados de la Fase 0 (los de `datasets_declarados` de
  `resultado_pasada_119_c3.json` con cubo `pequeno` y `sellado: false`; el guion los lee del congelado y para si no son 13):
  dresses-sales, kc2, climate-model-simulation-crashes, micro-mass, balance-scale, diabetes, pc1, Moneyball, pc4, house_prices_nominal, yeast,
  us_crime, mfeat-factors. Los medianos, los grandes y los sellados no se tocan.
- **Una repetición, todos los pliegues:** repetición 0 con su semilla; pliegues 0–4 (yeast, 0–3). MISMAS particiones, preparación y presupuesto
  que 119-C3/155/160/C1 (se comprueba la partición contra la v2 antes del primer intento).
- **Métrica de cierre:** `metrica_de_cierre_por_dataset` (AUROC / accuracy / R²), sobre la prueba fija del conjunto, con la cadena de la Fase 0.
  **error = 1 − métrica** (en regresión, 1 − R²).

## 2. Cantidades

- **R(c)** = 1 − Σ error_k32 / Σ error_k8, sobre los pliegues del conjunto (razón de medias, como en el 160 y el C1). Positivo = k = 32 es mejor.
  Si Σ error_k8 = 0: R = 0 si el de k = 32 también, −1 si no.
- **IC95 de R(c):** bootstrap de percentiles de PLIEGUES, 1.000 remuestras, semilla 0: las MISMAS funciones del 160 (`analiza_c160.reduccion_relativa` /
  `ic_de_r`, importadas por el analizador del C1 y el de este corte, no copiadas).
- **«Mejora clara»:** el límite inferior del IC95 de R > 0. **«Empeora de verdad»:** R < −3 % **y** el límite superior del IC95 < 0.
- **Saturados** (los del 160 que caen en el cubo, declarados ANTES de medir y por el mismo criterio —la mejor referencia de los miembros en la Fase 0
  congelada ≥ 98 %—; una prueba recalcula el criterio de los artefactos): de los 6 del 160 solo uno es pequeño, los otros cinco (optdigits, pendigits,
  sick, wilt, PhishingWebsites) son medianos y no entran en esta medida.
  SATURADOS: mfeat-factors
  El saturado NO entra en la media (a) ni en el recuento (b); SÍ entra en (c) y (d), y se lista.
- **Coste** de un intento: `wall_s` del subproceso (incluye ajustar, predecir train/validación/prueba y medir). Coste de un brazo en un conjunto =
  suma de `wall_s` de sus pliegues. Se informa también el `rss_pico_mb` de cada brazo.

## 3. La regla — CUMPLE si se dan las cuatro (y la medida es completa)

- **(a)** la media de R(c) sobre los **12 no saturados** ≥ **+3 %** (comparación exacta, sin redondear);
- **(b)** «mejora clara» en ≥ **4** de los 12 no saturados (el saturado con mejora clara se lista, no cuenta);
- **(c)** **ninguno** de los **13** (el saturado INCLUIDO) «empeora de verdad»;
- **(d)** Σ sobre los 13 del coste de `tabm_k32` ≤ **3,5 ×** Σ sobre los 13 del coste de `tabm_k8` (razón de sumas); se informan también la mediana y el
  máximo por conjunto y la memoria pico de cada brazo.

**Completitud → SIN VEREDICTO** (ni CUMPLE ni NO CUMPLE; lo comprueba el analizador antes de aplicar la regla): si falta algún conjunto de los 13,
algún pliegue esperado (yeast 0–3, el resto 0–4) de algún brazo; si un intento no cuenta como medida (`failed`, cortado…) —un pliegue sin medida es
un pliegue que falta, no un cero—; si hay registros de más o repetidos; si hay un conjunto mediano, grande o sellado; si la salida es parcial; si hay
intentos del mismo brazo con código distinto; si un brazo declara una red que no es la suya (la referencia con k ≠ 8 o con `capacidad`, la palanca
sin k = 32 declarado); si algún intento declara `parada_por_metrica`; si la regla, el analizador o lo que importa (`protocolo.py`,
`pasada_amplia_101_c5.py`, `matrixai/estudio/metricas.py`, `matrixai/estudio/incertidumbre.py`, `benchmarks/c160/analiza_c160.py`,
`benchmarks/c162/analiza_c162_c1.py`) no son los que midieron. Una sonda no decide nada.

**Se declara, no se decide con ello:** las épocas ejecutadas y la mejor época de cada brazo, cuántos intentos se pararon por plazo y cuántos por
paciencia en cada brazo (si k = 32 para por plazo mucho más que k = 8, parte de su resultado es «menos épocas», no «más capacidad»), y la memoria pico.

Si no cumple: la palanca no entra en el producto y el contrato 162 sigue con C3.

## 4. Tres palancas contra la misma referencia (invariante 3 del contrato)

C1, C2 y C3 son tres oportunidades de acertar por azar contra la misma referencia. El umbral de esta regla NO se endurece (D3 lo fija igual que el
160); lo que protege es la **confirmación**: nada cumple si no repite en conjuntos que no vio.

## 5. Confirmación (§4.3 del contrato)

Si y solo si los 13 CUMPLEN, la misma palanca y la misma regla se aplican UNA vez a los conjuntos de `CONFIRMACION_C2.md` (los 4 pequeños de
`CONFIRMACION.md` del C1, apartados antes de medir nada, distintos de los del 160 y de la Fase 0). Condiciones escaladas: (a) media de R ≥ +3 %
sobre los no saturados; (b) «mejora clara» en ≥ la mitad (redondeada hacia arriba) de los no saturados; (c) ninguno «empeora de verdad»; (d) igual
(≤ 3,5 ×). Se escribe su enmienda sellada ANTES de medirlos (incluida la declaración de saturados con el criterio ≥ 98 %). Cumplir en los 13 y
fallar en la confirmación = NO CUMPLE (sobreajuste a los 13).

## 6. Lo que NO se afirma

- Es la geometría de la Fase 0 (cada intento entrena con todo `entrena`, para con `valida`, plazo de 120 s), no la del estudio del Studio
  (`plazo_de_tabm`, 139-C4 reentrena con épocas fijas): el Studio estima el coste por época SIN k, así que **con k = 32 su plazo habría que
  recalcularlo** (trabajo de la integración, no de este corte; el coste por época de k = 32 sale de esta medida).
- 1 repetición y 5 pliegues: el IC dice cuánto se mueve el número al reentrenar, no cuánto se movería con otros datos; con 5 pliegues es grueso.
- Los dos brazos comparten semilla, no ruido de reloj: una máquina cargada puede cortar a un brazo antes que a otro. Por eso van pegados y se
  declaran los cortes por plazo.
- Es una sola palanca (k). `d_block` = 512 y la parada por métrica no se miden aquí.
