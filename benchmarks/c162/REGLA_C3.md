# Contrato 162 · C3 — REGLA DE DECISIÓN (para SELLAR ANTES de medir): TabM-mini con entrada lineal a trozos (PLE)

Fija, antes de que exista ninguna cifra de la medida real, qué se mide, cómo se compara y cuándo la palanca «TabM-mini + embeddings lineales a
trozos» (idea 39, Ola 5) cumple. Es la regla del C1 (`REGLA_C1.md`, que a su vez es la del 160 con la decisión **D3** de Roberto): reducción
relativa del error, media ≥ +3 % sobre los no saturados, mejora clara en ≥ 5, ninguno empeora, coste acotado (≤ 1,5 ×) y confirmación en
conjuntos nuevos. Su sha256 se graba en `analiza_c162_c3.REGLA_SHA256_SELLADA` y en la salida de la medida (`regla_sha256_al_medir`);
`medir_c162_c3.py` se niega a medir y `analiza_c162_c3.py` a dar veredicto si el fichero ya no es el sellado. **Cambiar una coma después de
sellar es cambiar la regla**: una enmienda nueva con su sello, no una edición.

## 1. Qué se mide

- **La palanca.** `MotorDensaTabM.con_variante("mini_ple")` (matrixai-engines, 162-C3, `redes/tabm_mini_ple.py`): la red pasa a ser **TabM-mini**
  (`arch_type='tabm-mini'` del repositorio oficial yandex-research/tabm, Apache-2.0: el ÚNICO adaptador por miembro es un escalado `(k, d_in)` a la
  ENTRADA; los bloques `Linear-ReLU-Dropout` son los de un MLP corriente, compartidos por los k miembros; la salida son k capas independientes) y
  cada numérica se codifica con **embeddings lineales a trozos** (`PiecewiseLinearEmbeddings(version='B')` de `rtdl_num_embeddings`, sin ReLU
  final) en vez de los periódicos PLR. Los **bordes de los trozos** salen de los **cuantiles de las filas de ENTRENAMIENTO** (solo train: ni
  validación ni prueba; una prueba con nombre lo vigila y un sabotaje de ella), `n_bins = 48` (el de `compute_bins`) y `d_embedding` numérico = 16.
  Las categóricas y las entradas simples, igual que hoy. **`n_bins` y `d_embedding` quedan FIJOS aquí, sin afinar**: el paper los afina
  (`n_bins` en [2, 128], `d_embedding` en [8, 32] de 4 en 4) y por eso esta palanca no es «el TabM-mini del paper afinado» sino «TabM-mini con una
  elección razonable y fija»: un resultado negativo no descarta la idea con otros valores (§6). Apagada por omisión y bit a bit igual que el
  commit anterior de motores cuando está apagada (pruebas con nombre).
- **Los dos brazos** (cada uno, un subproceso aislado con su tope, UNA vez por conjunto y pliegue, la referencia PRIMERO y la palanca
  inmediatamente después, para que la carga de la máquina los afecte igual): **`tabm_ref`** (TabM `ref` tal y como entrega hoy: PLR, k = 8) y
  **`tabm_mini_ple`** (la palanca, k = 8). Los dos paran por la **pérdida** de validación (las otras palancas del 162 —parada por métrica,
  k = 32— NO se mezclan y el analizador rechaza un intento que las declare). Ambos se miden en ESTA pasada con el MISMO código: no se reutiliza
  nada del 155, del 160 ni del C1 (otro código, otra carga, y el plazo es de reloj de pared). Mismos datos preparados, misma semilla del intento,
  mismo presupuesto, 4 hilos.
- **Conjuntos:** los 23 no sellados pequeños y medianos de la Fase 0, los mismos del 155, del 160 y del C1 (los de `datasets_declarados` de
  `resultado_pasada_119_c3.json` con cubo `pequeno`/`mediano` y `sellado: false`; el guion los lee del congelado y para si no son 23). Los 9
  grandes y los 8 sellados no se tocan.
- **Una repetición, todos los pliegues:** repetición 0 con su semilla; pliegues 0–4 (yeast, 0–3). MISMAS particiones, preparación, presupuesto
  de pared por cubo (120 s pequeños, 300 s medianos) que 119-C3/155/160/C1 (se comprueba la partición contra la v2 antes del primer intento).
- **Métrica de cierre:** la de cada conjunto (AUROC / accuracy / R²), sobre la prueba fija, con la cadena de la Fase 0. **error = 1 − métrica**.
  La prueba NO interviene en nada del ajuste; los bordes y la parada usan solo train y validación respectivamente.

## 2. Cantidades

- **R(c)** = 1 − Σ error_palanca / Σ error_referencia, sobre los pliegues del conjunto (razón de medias, como en el 160). Positivo = la palanca
  es mejor. Si Σ error_referencia = 0: R = 0 si el de la palanca también es 0, −1 si no.
- **IC95 de R(c):** bootstrap de percentiles de PLIEGUES, 1.000 remuestras, semilla 0: las MISMAS funciones que el 160
  (`analiza_c160.reduccion_relativa` / `ic_de_r`, importadas, no copiadas). Con menos de dos pliegues no hay IC y no hay «mejora clara» ni «empeora».
- **«Mejora clara»:** el límite inferior del IC95 de R > 0. **«Empeora de verdad»:** R < −3 % **y** el límite superior del IC95 < 0.
- **Saturados** (los 6 del 160, los del C1, importados de `analiza_c162_c1.SATURADOS`): mfeat-factors, optdigits, pendigits, sick, wilt,
  PhishingWebsites. NO entran en la media (a) ni en el recuento (b); SÍ entran en (c) y (d), y se listan.
- **Coste** de un intento: `wall_s` del subproceso. Coste de un brazo en un conjunto = suma de `wall_s` de sus pliegues.

## 3. La regla — CUMPLE si se dan las cuatro (y la medida es completa)

- **(a)** la media de R(c) sobre los **17 no saturados** ≥ **+3 %** (comparación exacta, sin redondear);
- **(b)** «mejora clara» en ≥ **5** de los 17 no saturados (los saturados con mejora clara se listan, no cuentan);
- **(c)** **ninguno** de los 23 (saturados INCLUIDOS) «empeora de verdad»;
- **(d)** Σ sobre los 23 del coste de `tabm_mini_ple` ≤ **1,5 ×** Σ sobre los 23 del coste de `tabm_ref` (razón de sumas); se informan también
  la mediana y el máximo por conjunto. Se espera MÁS BARATA por época (sin las 96 frecuencias por columna ni los `r`/`s` por capa; sonda en el
  INFORME), pero el plazo es de pared: la red mini + PLE puede tardar lo mismo si ambas agotan su plazo; (d) solo falla si es más cara.

**Completitud → SIN VEREDICTO** (ni CUMPLE ni NO CUMPLE; lo comprueba el analizador antes de aplicar la regla): si falta algún conjunto de los
23, algún pliegue esperado (yeast 0–3, el resto 0–4) de algún brazo; si un intento no cuenta como medida (`failed`, cortado…) —un pliegue sin
medida es un pliegue que falta, no un cero—; si hay registros de más o repetidos; si la salida es parcial; si hay intentos del mismo brazo con
código distinto; si un brazo declara otra red que la suya (la referencia con variante o sin PLR; la palanca sin la variante `mini_ple` con
bordes de train), una parada por métrica o una capacidad (k ≠ 8); si la regla, el analizador o lo que importa (`protocolo.py`,
`pasada_amplia_101_c5.py`, `matrixai/estudio/metricas.py`, `matrixai/estudio/incertidumbre.py`, `benchmarks/c160/analiza_c160.py`,
`benchmarks/c162/analiza_c162_c1.py`) no son los que midieron. Una sonda no decide nada.

**Se declara, no se decide con ello:** las épocas ejecutadas y la mejor época de cada brazo, y cuántos se pararon por plazo y cuántos por
paciencia (si casi todo para por plazo, la comparación es sobre todo de velocidad).

Si no cumple: la palanca no entra en el producto y el contrato 162 sigue con su cierre.

## 4. Tres palancas contra la misma referencia (invariante 3 del contrato)

C1, C2 y C3 son tres oportunidades de acertar por azar contra la misma referencia. El umbral de esta regla NO se endurece (D3 lo fija igual que
el 160); lo que protege es la **confirmación**: nada cumple si no repite en conjuntos que no vio.

## 5. Confirmación (§4.3 del contrato, D3)

Si y solo si los 23 CUMPLEN, la misma palanca y la misma regla se aplican UNA vez a los 6 conjuntos de `benchmarks/c162/CONFIRMACION.md` (nuevos,
apartados el 12-11 antes de medir nada, y distintos de los 6 del 160; son los mismos que usa la confirmación del C1: se miden por separado,
contra su propia referencia). Con 6 conjuntos las condiciones se escalan así: (a) media de R ≥ +3 % sobre los no saturados; (b) «mejora clara» en
≥ la mitad (redondeada hacia arriba) de los no saturados; (c) ninguno «empeora de verdad»; (d) igual (≤ 1,5 ×). Se escribe su enmienda sellada
ANTES de medirlos (incluida la declaración de saturados con el criterio ≥ 98 %). Cumplir en los 23 y fallar en la confirmación = NO CUMPLE.

## 6. Lo que NO se afirma

- Es la geometría de la Fase 0 (cada intento entrena con todo `entrena`, para con `valida`, plazo de su cubo), no la del estudio del Studio
  (`plazo_de_tabm`): integrar la variante en el producto sería trabajo aparte (el `esquema()` del motor no la sabe leer todavía y lo dice).
- `n_bins` = 48 y `d_embedding` = 16 están fijos sin afinar (§1): un NO CUMPLE dice que ESTA elección no gana, no que la idea no valga.
- 1 repetición y 5 pliegues, con la prueba fija de cada conjunto: el IC dice cuánto se mueve el número al reentrenar, no cuánto se movería con
  otros datos. Con 5 pliegues el IC es grueso.
- Los dos brazos comparten semilla, no ruido de reloj: el plazo es de reloj de pared. Por eso van pegados y se declaran los cortes por plazo.
- La palanca cambia DOS cosas a la vez (la entrada y la estructura del ensamblado: mini); no separa cuál aporta. Esa descomposición no es de
  este corte.
