# Contrato 165 · C2 — REGLA DE DECISIÓN (para SELLAR ANTES de medir): cómo se elige cada defecto de «Filas a revisar»

Fija, antes de que exista ninguna cifra de la medida real, qué se siembra, qué se mide y cuándo un defecto del detector cambia. Su sha256 se graba en
`analiza_c165_c2.REGLA_SHA256_SELLADA` y en la salida (`regla_sha256_al_medir`); `medir_c165_c2.py` se niega a medir y `analiza_c165_c2.py` a dar
veredicto si el fichero ya no es el sellado. **Cambiar una coma después de sellar es cambiar la regla**: una enmienda nueva con su sello.

**LÍMITE DE LO QUE ESTO ES (invariantes 1 y 5 del contrato).** Esto mide cuánto recupera el detector de lo que SEMBRAMOS NOSOTROS, con tres tipos de
rareza que conocemos. NO es acierto sobre las anomalías reales de nadie, y ninguna cifra de aquí entra en el producto ni en la pantalla como
rendimiento. Sirve para dos cosas: decidir defectos y escribir el texto de límites, siempre dicho «sobre anomalías sembradas por MatrixAI, no sobre
las tuyas».

## 1. Qué se mide

- **Conjuntos (6, del Studio, `ejemplos_medidos/`):** `breast-cancer-wisconsin-diagnostic`, `cardiotocography`, `climate-model-simulation-crashes`,
  `dresses-sales`, `pc1`, `us_crime`. Se les QUITA la columna objetivo del catálogo (el flujo no tiene objetivo) y nada más. `parkinsons-telemonitoring`
  queda fuera (5.875 filas con repetidos por sujeto: filas no independientes).
- **Siembra (semilla propia por conjunto, tipo, tasa y repetición):** se SUSTITUYEN filas ya existentes (el número de filas no cambia) y se graba cuáles.
  Tasas **1 % y 5 %** de las filas (mínimo 3 sembradas). Tres repeticiones de siembra (semillas 0, 1, 2). Tres tipos:
  1. **`extremo`**: en cada fila sembrada, UNA columna numérica (al azar entre las numéricas con ≥ 10 valores distintos) pasa a `máximo + 0,5 × rango`.
  2. **`combinacion`**: la pareja de columnas numéricas más correlacionada (|r| de Pearson ≥ 0,5, sin ser copias, |r| < 0,999) recibe valores que, uno a
     uno, caen dentro de [percentil 10, percentil 90] pero **en la esquina que la correlación prohíbe** (A en el percentil 85–95 y B en el 5–15, o los dos
     altos si r < 0). Cada columna por separado es normal; la pareja, imposible. Un conjunto sin pareja así no tiene este tipo (se declara).
  3. **`rara`**: una columna categórica (≤ 30 niveles) recibe un valor NUEVO, distinto por fila sembrada (`valor_nuevo_<i>`). Un conjunto sin
     categóricas no tiene este tipo (se declara). Hoy solo `dresses-sales` las tiene: este eje es de UN conjunto y se dice.
- **Variantes (12 = rejilla completa de tres ejes; «hoy» = la primera de cada eje):**
  - `escalado`: `ninguno` (hoy) | `cuantiles` (cada columna por su rango medio, 0–1).
  - `categoricas`: `frecuencia` (hoy: frecuencia relativa) | `ordinal` (código por orden alfabético) | `onehot` (una columna 0/1 por nivel).
  - `max_samples` de IsolationForest: `auto` (hoy: min(256, n)) | `todas` (n).
  Siempre 200 árboles, `n_jobs=1`, semilla del detector 0 (la del producto). Preparación idéntica a `matrixai_engines.anomalias` (misma exclusión de
  columnas, mediana para faltantes numéricos).
- **`contamination` NO es una variante de la rejilla**, y se dice por qué: en IsolationForest solo fija el UMBRAL de «anomalía» y no cambia ni una
  puntuación ni el orden (se comprueba en cada conjunto: `score_samples` idéntico con y sin él). Como el producto da un orden, no un corte, no hay nada
  que elegir en el orden. Lo que sí se mide es cómo cae la recuperación según dónde la persona corte (ver `k`).
- **Cantidades por celda (conjunto × tipo × tasa × repetición × variante):** con `n_s` = filas sembradas y el orden de revisión del detector,
  `recuperación@k` = sembradas entre las k primeras / `n_s`, y `precisión@k` = sembradas entre las k primeras / k, para `k = n_s/2`, `n_s`, `2·n_s`
  (redondeado hacia arriba, mínimo 1); y la precisión media (AP) sobre todo el orden. Se promedian las 3 repeticiones por celda (conjunto × tipo ×
  tasa × variante).
- **La métrica que decide: `M` = recuperación@(2·n_s)** (una persona que revisa el doble de filas de las sembradas). Se informan las demás.

## 2. La regla — qué defecto se elige (cada eje por separado, los otros dos en «hoy»)

Para cada candidato de un eje se compara con «hoy» sobre las **celdas afectadas** (las del conjunto × tipo × tasa donde la variante cambia el orden de
algo: para `categoricas`, los conjuntos con categóricas; para `escalado` y `max_samples`, todas). `Δ(celda)` = M(candidato) − M(hoy).

El candidato **CAMBIA el defecto** si se dan las tres:
- **(a)** media de Δ sobre las celdas afectadas ≥ **+0,03** (en `categoricas`, ≥ **+0,05**: un solo conjunto, se exige más);
- **(b)** Δ ≥ 0 en ≥ **60 %** de las celdas afectadas;
- **(c)** ningún TIPO de siembra tiene media de Δ < **−0,03** (el que ayuda a un tipo no vale si rompe otro).
Si varios candidatos de un eje cumplen, el de mayor media de Δ (empate: el primero de la lista). Si ninguno: **se queda el
defecto de hoy** (el que no demuestra ganar no entra). Comparación exacta, sin redondear.

`contamination`: **se queda sin definir (`None`)** si el control de invariancia pasa en los 6 conjuntos; si falla en alguno, SIN VEREDICTO.

## 3. Texto de límites

Con la combinación ELEGIDA (que está en la rejilla, medida entera, no deducida de los ejes), el informe da la `recuperación@(2·n_s)` media por TIPO
de siembra y tasa. Esas cifras alimentan **solo** la frase de límites, escrita así: «sobre anomalías sembradas por MatrixAI (valor extremo,
combinación imposible, categoría rara), el detector pone en las primeras 2·n filas una media de X de las n sembradas; no dice nada de las tuyas».
Si la combinación elegida es peor que «hoy» en más de 0,03 de M media (una interacción entre ejes), se avisa y no se adopta: **queda «hoy»**.

## 4. Completitud → SIN VEREDICTO

Ni defectos elegidos ni cifras si: falta algún conjunto de los 6, alguna celda o repetición o variante de las que la propia salida declara aplicables;
`extremo` no es aplicable en los 6 o `rara` no lo es en `dresses-sales`; hay registros repetidos o de más; la salida es parcial; la regla, el analizador, el
módulo `matrixai_engines/anomalias.py` o los seis CSV no son los que midieron; la variante «hoy» no reproduce el orden de `puntuar_filas` del producto
(control por conjunto); falla la invariancia de `contamination`; una fila sembrada no coincide con la lista grabada. Una sonda no decide nada.
