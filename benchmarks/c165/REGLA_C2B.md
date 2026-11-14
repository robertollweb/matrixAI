# Contrato 165 · C2b — REGLA DE DECISIÓN (para SELLAR ANTES de medir): cómo se codifican las categóricas de «Filas a revisar»

Nace de la auditoría del 165 (aud1, I4 e I5): el ordinal ALFABÉTICO que eligió el C2 mete un orden que depende del NOMBRE de las categorías
(renombrar un solo nivel movía el top-25 de dresses-sales tanto como la semilla), la siembra `valor_nuevo_<i>` del C2 caía al FINAL del alfabeto —
el extremo del eje— e inflaba la cifra de «categoría rara», y una categórica vacía se codifica −1, en el borde del eje, y sube sus filas. Esta regla
fija, antes de que exista ninguna cifra, qué se mide y cuándo cambia la codificación. Su sha256 se graba en `analiza_c165_c2b.REGLA_SHA256_SELLADA` y en
la salida; `medir_c165_c2b.py` se niega a medir y `analiza_c165_c2b.py` a dar veredicto si el fichero ya no es el sellado. **Cambiar una coma después
de sellar es cambiar la regla**: una enmienda nueva con su sello. La regla del C2 (`REGLA_C2.md`), su analizador y su `banco.json` NO se tocan.

**LÍMITE DE LO QUE ESTO ES.** Mide cuánto recupera el detector de lo que SEMBRAMOS NOSOTROS y cuánto cambia la lista al renombrar una categoría. NO
es acierto sobre las anomalías reales de nadie. Sirve para decidir la codificación de las categóricas y para la frase de límites de la categoría rara.

## 1. Qué se mide

- **Conjuntos (6).** UNO real con categóricas: `dresses-sales` (sin su objetivo `Class`, como en el C2). Y CINCO DERIVADOS, declarados como tales:
  `breast-cancer-wisconsin-diagnostic`, `cardiotocography`, `climate-model-simulation-crashes`, `pc1` y `us_crime` (sin su objetivo), en los que TRES
  columnas numéricas completas con ≥ 10 valores distintos (las de las posiciones primera, central y última por orden de nombre) se convierten en
  categóricas: se cortan por los cuantiles 5, 20, 45, 75 y 93 % (seis niveles de frecuencias desiguales), cada nivel recibe un nombre de una lista
  fija de 40 municipios (con tildes, mayúsculas y minúsculas) barajada con semilla propia por conjunto y columna —así el orden alfabético NO tiene
  que ver con el orden de los valores— y el 3 % de sus celdas (con semilla) se deja VACÍO. El resto del conjunto no se toca.
- **Variantes (6 = 2 × 3; «hoy» = la primera de cada eje = el producto).**
  - `orden` de los niveles en el eje ordinal: `alfabetico` (hoy: por el nombre) | `frecuencia` (del más frecuente al menos; a igual frecuencia, por
    orden de primera aparición en el fichero: no depende del nombre).
  - `vacia`: `menos_uno` (hoy: −1, fuera del eje) | `nivel_propio` (la vacía es un nivel más, ordenado con la misma regla que los demás; en
    `alfabetico` su nombre es «» y va el primero) | `centro` (la mediana de los códigos de las celdas con dato de esa columna, como la mediana de las
    numéricas).
  Siempre: IsolationForest de 200 árboles con todas las filas (`max_samples = n`), `n_jobs = 1`, sin `contamination`; misma exclusión de columnas que el
  producto; numéricas tal cual con la mediana para el faltante. Semilla del detector 0 salvo en la estabilidad.
- **Siembras (tasas 1 % y 5 % de las filas, mínimo 3; repeticiones 0, 1 y 2; semilla propia por conjunto, tipo, tasa y repetición).** Se SUSTITUYEN
  filas y se graba cuáles:
  - `rara_medio`: en una categórica (2 a 30 niveles, elegida con la semilla), cada fila sembrada recibe un nivel NUEVO, distinto por fila, cuyo nombre
    es el del nivel MEDIANO por orden alfabético seguido de `~` y su número (cae junto a la mitad del alfabeto).
  - `rara_azar`: lo mismo con el nombre de un nivel existente elegido AL AZAR seguido de `~` y su número (cae en una posición al azar).
  - `rara_final`: el nombre del C2, `valor_nuevo_<i>` (cae al final). **Se informa, NO decide**: es el vicio que se quiere medir.
  - `extremo`: el del C2 (una numérica por fila a `máximo + 0,5 × rango`; `banco_c165.sembrar`).
  De cada siembra se graba la POSICIÓN media de sus nombres en el orden alfabético de su columna (0 = primero, 1 = último). Si la media de esas
  posiciones en todas las siembras de `rara_medio`, o en todas las de `rara_azar`, cae fuera de [0,3, 0,7], la siembra no es neutra: **SIN VEREDICTO**.
- **`M` (la métrica de recuperación) = recuperación@(2·n_s)**, la del C2: de las n_s sembradas, cuántas pone el detector en las 2·n_s primeras.
  Se promedian las 3 repeticiones por celda (conjunto × tipo × tasa × variante).
- **`S` (sensibilidad a RENOMBRAR)**, sin siembra: en cada conjunto, en sus dos primeras categóricas usadas (por nombre) de 2 a 30 niveles, el nivel más
  frecuente y el menos frecuente (empate: el primero en aparecer) se renombran de dos maneras, `!` + nombre (va el primero) y `zzzz_` + nombre (va
  detrás de las letras sin tilde); para cada renombre, el Jaccard de las k primeras filas frente a la lista sin renombrar, con k el del producto
  (5 % de las filas, mínimo 1, tope 50). `S` = la media de esos Jaccard (1 = renombrar no mueve nada).
- **`E` (la referencia de la semilla)**: el Jaccard de las k primeras entre la semilla 0 y las semillas 1, 2, 3 y 4, de media. Se informa al lado de `S`.
- **`I` (inflación por la VACÍA)**, sin siembra: con la semilla 0, la proporción de filas con alguna categórica usada vacía entre las k primeras,
  dividida por esa proporción en todo el conjunto (1 = las vacías no suben ni bajan). `D` = media sobre los conjuntos de |I − 1|.

## 2. La regla — qué se elige (cada eje por separado, el otro en «hoy»)

**Celdas de decisión:** las de (conjunto × tipo × tasa) con tipo `rara_medio`, `rara_azar` o `extremo`, en los 6 conjuntos.
`Δ(celda)` = M(candidato) − M(hoy). Comparación exacta, sin redondear.

- **Eje `orden`** (con `vacia = menos_uno`). `frecuencia` CAMBIA el defecto si se dan las tres:
  - **(a)** media de Δ sobre las celdas de decisión ≥ **−0,02** (no pierde recuperación);
  - **(b)** ningún tipo de decisión con media de Δ < **−0,05**;
  - **(c)** S(frecuencia) − S(alfabetico) ≥ **+0,10**; o media de Δ ≥ **+0,03** con S(frecuencia) ≥ S(alfabetico) − 0,02.
- **Eje `vacia`** (con `orden = alfabetico`). Un candidato (`nivel_propio`, `centro`) CAMBIA el defecto si cumple (a) y (b) y además
  - **(c')** D(candidato) ≤ D(menos_uno) − **0,5**.
  Si cumplen varios: el de menor D (empate: el primero de la lista).
- Si ninguno cumple, **se queda «hoy»**: el que no demuestra ganar no entra.
- **Combinación:** la pareja elegida está en la rejilla (medida entera, no deducida). Si es distinta de «hoy» y su media de Δ sobre las celdas de
  decisión frente a «hoy» es < **−0,03** (una interacción entre ejes), se avisa y **queda «hoy»**.

## 3. Texto de límites

Con la combinación ELEGIDA, la media de M de `rara_medio` y `rara_azar` juntas, por tasa, sobre los 6 conjuntos: solo eso alimenta la frase de la
categoría rara, escrita «sobre anomalías sembradas por MatrixAI, en un conjunto real y cinco derivados con categorías puestas por nosotros; no sobre
las tuyas». Se informan al lado, sin entrar en el texto, la misma cifra de `rara_final` (el vicio), `S`, `E` y `D` de cada variante.

## 4. Completitud → SIN VEREDICTO

Ni codificación elegida ni cifras si: falta algún conjunto de los 6, alguna celda, repetición o variante; algún conjunto no tiene `extremo` o no tiene
`rara_*`; faltan los renombres, las semillas o la inflación de alguna variante en algún conjunto; hay registros repetidos o de más; la salida es
parcial; la regla, el analizador, `matrixai_engines/anomalias.py`, `banco_c165.py`, `banco_c165_c2b.py` o los seis CSV no son los que midieron; la
variante «hoy» no reproduce el orden de `puntuar_filas` del producto en algún conjunto (control, también en los derivados); una lista de sembradas no
cuadra con su número; la siembra no es neutra (§1). Una sonda no decide nada.
