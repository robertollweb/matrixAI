# Contrato 160 — REGLA DE DECISIÓN (propuesta en C1, para SELLAR ANTES de medir)

Fija, antes de que exista ninguna cifra de la medida real, qué se mide, cómo se compara y cuándo el ensamblado «LightGBM + TabM»
(idea 37, Ola 5) cumple. Su sha256 se graba en `analiza_c160.REGLA_SHA256_SELLADA` (el supervisor, al sellar) y en la salida de la medida
(`regla_sha256_al_medir`); `medir_c160.py` se niega a medir y `analiza_c160.py` a dar veredicto si el fichero ya no es el sellado.
**Cambiar una coma después de sellar es cambiar la regla**: una enmienda nueva con su sello, no una edición.

Decisiones de Roberto (12-11, tarjeta `ola5-37-decisiones` = A): **D1** la mezcla se compara con el MEJOR de sus dos miembros en cada
conjunto; **D2** miembros LightGBM + TabM, media simple (de probabilidades; en regresión, de predicciones); **D3** esta regla, sellada
antes de medir, más conjuntos NUEVOS de confirmación apartados ahora (`CONFIRMACION.md`); **D4** la P3 del documento ya está hecha en el 129.

## 1. Qué se mide

- **Conjuntos:** los 23 no sellados pequeños y medianos de la Fase 0, los mismos del 155 (los de `datasets_declarados` de
  `resultado_pasada_119_c3.json` con cubo `pequeno`/`mediano` y `sellado: false`; el guion los lee del congelado y para si no son 23).
  Los 9 grandes y los 8 sellados no se tocan.
- **Una repetición, todos los pliegues:** repetición 0 con su semilla; pliegues 0–4 (yeast, 0–3: la partición solo da 4). MISMAS particiones,
  preparación por motor, presupuesto de pared por cubo (120 s pequeños, 300 s medianos) y 4 hilos que 119-C3/155 (se comprueba la
  partición contra la v2 antes del primer intento).
- **Miembros:** `lightgbm` (el de la Fase 0, política por omisión) y TabM `ref` (`matrixai.dense.tabm_cpu`, la receta de referencia del
  producto), cada uno entrenado en su propio subproceso aislado, UNA vez por (conjunto, pliegue). La mezcla NO reentrena: sale de las
  predicciones volcadas.
- **Métrica de cierre:** la de `metrica_de_cierre_por_dataset` (AUROC en binarias, accuracy en multiclase, R² en regresión), sobre la prueba
  fija del conjunto, calculada con la cadena de la Fase 0 (`matrixai.estudio.metricas.evaluar` + `pasada_amplia_101_c5.metricas_del_informe`).
  **error = 1 − métrica** (en regresión, 1 − R²).
- **La mezcla:** por fila, media simple de las probabilidades (distribución completa, columnas alineadas por etiqueta de clase; la clase
  predicha es la de mayor probabilidad media, empate → la primera) o de las predicciones (regresión). Las filas se emparejan por su
  IDENTIFICADOR, no por su posición.
- **Mejor miembro** (D1) de un conjunto: el de mayor media de la métrica de cierre sobre los pliegues del conjunto; empate exacto → LightGBM.
  Es una vara dura a propósito (el «oráculo», elegido con los mismos pliegues en que se mide).

## 2. Cantidades

- **R(c)** = 1 − Σ error_mezcla / Σ error_mejor_miembro, sobre los pliegues del conjunto (reducción relativa del error medio). Razón de
  medias y no media de razones por pliegue: con un error de referencia casi cero en un pliegue (conjuntos saturados) la razón por
  pliegue explota. Si Σ error_mejor = 0: R = 0 si la mezcla también es 0, −1 si no. (Cambio frente a la propuesta de C0, que decía «media de
  los pliegues»: declarado aquí, antes de medir.)
- **IC95 de R(c):** bootstrap de percentiles de PLIEGUES, 1.000 remuestras, semilla 0 (los de `protocolo.SEMILLA_DEL_INTERVALO` y
  `REMUESTRAS_DEL_INTERVALO`, y el `_percentil` de 105-C2). Con menos de dos pliegues no hay IC y no hay «mejora clara» ni «empeora».
- **«Mejora clara»:** el límite inferior del IC95 de R > 0. **«Empeora de verdad»:** R < −3 % **y** el límite superior del IC95 < 0.
- **Saturados** (declarados ahora, ANTES de medir):
  SATURADOS: mfeat-factors, optdigits, pendigits, sick, wilt, PhishingWebsites
  Criterio: la mejor de las dos referencias de los miembros en la Fase 0 congelada ≥ 98 % —la media de la métrica de cierre de LightGBM
  (`pasada_v2_113_resultado.json`) y la de TabM (119-C3, 119-C4, 120-C5), sobre todos sus intentos que cuentan como medida—.
  Comprobado contra los artefactos (23 conjuntos; mejor miembro): mfeat-factors 0,9815 · optdigits 0,9912 · pendigits 0,9919 · sick 0,9981 ·
  wilt 0,9991 · PhishingWebsites 0,9965. El siguiente es Internet-Advertisements, 0,9763 (LightGBM; TabM 0,9700): NO saturado. (APSFailure
  0,995 también lo sería, pero es grande y no está en la medida.) Una prueba recalcula el criterio de los artefactos y exige esta lista.
  Los saturados NO entran en la media (a) ni en el recuento (b); SÍ entran en (c) y (d).
- **Coste de predecir** de un miembro en un conjunto: ms por fila de la llamada que el arnés mide al predecir sobre la prueba
  (`predict_proba` en clasificación, `predict` en regresión), media de los pliegues. Coste de la mezcla = suma de los dos miembros
  (predice con ambos).

## 3. La regla — CUMPLE si se dan las cuatro (y la medida es completa)

- **(a)** la media de R(c) sobre los **17 no saturados** ≥ **+3 %** (comparación exacta, sin redondear);
- **(b)** «mejora clara» en ≥ **5** de los 17 no saturados (los saturados con mejora clara se listan, no cuentan);
- **(c)** **ninguno** de los 23 (saturados INCLUIDOS) «empeora de verdad»;
- **(d)** Σ sobre los 23 del coste de la mezcla ≤ **3 ×** Σ sobre los 23 del coste del mejor miembro de cada conjunto (razón de sumas); se
  informan también la mediana y el máximo por conjunto.

**Completitud → SIN VEREDICTO** (ni CUMPLE ni NO CUMPLE; lo comprueba el analizador antes de aplicar la regla): si falta algún conjunto de los
23, algún pliegue esperado (yeast 0–3, el resto 0–4) de algún miembro, o su volcado; si un intento no cuenta como medida (`failed`, cortado…) —un
pliegue sin medida es un pliegue que falta, no un cero—; si hay registros de más o repetidos; si la salida es parcial; si hay intentos del mismo
motor con código distinto; si la regla, el analizador o lo que importa (`protocolo.py`, `pasada_amplia_101_c5.py`, `matrixai/estudio/metricas.py`,
`matrixai/estudio/incertidumbre.py`, `volcado_c160.py`) no son los que midieron; si un `.npz` no tiene el sha256 registrado. Una sonda no decide
nada.

Si no cumple: la mezcla no entra en el estudio (el C3 no se hace) y el contrato 160 se cierra con su medida; el 114-C6 deja de ser el único
antecedente.

## 4. Confirmación (D3)

Si y solo si los 23 CUMPLEN, la misma mezcla y la misma regla se aplican UNA vez a los conjuntos de `CONFIRMACION.md` (nuevos, apartados antes de
mirar nada). Con 4–6 conjuntos las condiciones numéricas se escalan así: (a) media de R ≥ +3 % sobre los no saturados; (b) «mejora clara» en
≥ la mitad (redondeada hacia arriba) de los no saturados; (c) ninguno «empeora de verdad»; (d) igual. Se escribe su enmienda sellada ANTES de
medirlos. Cumplir en los 23 y fallar en confirmación = NO CUMPLE (sobreajuste a los 23).

## 5. Lo que NO se afirma

- Es la geometría de la Fase 0 (cada intento entrena con todo `entrena`, para con `valida`, plazo de su cubo), no la del estudio; TabM aquí no
  es TabM con el plazo de `plazo_de_tabm` del Studio.
- 1 repetición y 5 pliegues, con la prueba fija de cada conjunto: el IC dice cuánto se mueve el número al reentrenar, no cuánto se movería con otros
  datos. Con 5 pliegues el IC es grueso.
- El coste se mide dentro del arnés (datos ya preparados, 4 hilos, una sola llamada por intento): es una razón entre dos motores en las mismas
  condiciones, no el tiempo de «Usar» del producto. No incluye cargar dos modelos.
- Los empates de LightGBM/TabM entre pliegues en la elección del «mejor miembro» se resuelven por la media de los mismos pliegues en que se mide
  la mezcla: favorece al miembro (D1, deliberado).
