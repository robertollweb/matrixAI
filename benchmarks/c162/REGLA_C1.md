# Contrato 162 · C1 — REGLA DE DECISIÓN (para SELLAR ANTES de medir): TabM parando y entregando por la MÉTRICA DE CIERRE

Fija, antes de que exista ninguna cifra de la medida real, qué se mide, cómo se compara y cuándo la palanca «parada por métrica» de TabM
(idea 39, Ola 5) cumple. Es la regla del 160 (`benchmarks/c160/REGLA_DE_DECISION.md`) adaptada por la decisión **D3** de Roberto (12-11,
tarjeta `ola5-39-decisiones` = A): reducción relativa del error, media ≥ +3 % sobre los no saturados, mejora clara en ≥ 5, ninguno empeora,
coste acotado, y confirmación en conjuntos nuevos. Su sha256 se graba en `analiza_c162_c1.REGLA_SHA256_SELLADA` y en la salida de la medida
(`regla_sha256_al_medir`); `medir_c162_c1.py` se niega a medir y `analiza_c162_c1.py` a dar veredicto si el fichero ya no es el sellado.
**Cambiar una coma después de sellar es cambiar la regla**: una enmienda nueva con su sello, no una edición.

## 1. Qué se mide

- **La palanca.** `MotorDensaTabM.con_parada_por_metrica()` (matrixai-engines, 162-C1): el motor para (paciencia 16, la semántica del notebook:
  tras 17 épocas seguidas sin mejora) y entrega la época de MEJOR métrica de cierre sobre las MISMAS filas de validación y la predicción
  promediada del ensamblado, en vez de la de menor pérdida. Mejorar es ESTRICTAMENTE mejor (un empate no es mejora). La métrica es la del
  núcleo (`matrixai.estudio.metricas.calcular`): AUROC en binarias, accuracy en multiclase, R² en regresión. Apagada por omisión y bit a bit
  igual que antes cuando está apagada (pruebas con nombre). Si la métrica no se puede calcular en esa validación (una sola clase, validación
  constante), el ajuste cae a la pérdida y lo DECLARA; ese pliegue cuenta como medido y se lista.
- **Los dos brazos** (cada uno, un subproceso aislado con su tope, UNA vez por conjunto y pliegue, el de pérdida PRIMERO y el de métrica
  inmediatamente después, para que la carga de la máquina los afecte igual):
  **`tabm_perdida`** (la REFERENCIA: TabM `ref` tal y como entrega hoy, parada por la pérdida) y **`tabm_metrica`** (la palanca).
  Ambos se miden en ESTA pasada con el MISMO código: no se reutiliza nada del 155 ni del 160 (otro código, otra carga, y el plazo es de reloj
  de pared). Mismos datos preparados, misma semilla del intento, mismo presupuesto, 4 hilos.
- **Conjuntos:** los 23 no sellados pequeños y medianos de la Fase 0, los mismos del 155 y del 160 (los de `datasets_declarados` de
  `resultado_pasada_119_c3.json` con cubo `pequeno`/`mediano` y `sellado: false`; el guion los lee del congelado y para si no son 23).
  Los 9 grandes y los 8 sellados no se tocan.
- **Una repetición, todos los pliegues:** repetición 0 con su semilla; pliegues 0–4 (yeast, 0–3). MISMAS particiones, preparación,
  presupuesto de pared por cubo (120 s pequeños, 300 s medianos) que 119-C3/155/160 (se comprueba la partición contra la v2 antes del primer
  intento).
- **Métrica de cierre:** `metrica_de_cierre_por_dataset` (AUROC / accuracy / R²), sobre la prueba fija del conjunto, con la cadena de la Fase 0
  (`matrixai.estudio.metricas.evaluar` + `pasada_amplia_101_c5.metricas_del_informe`, la que ya hace el arnés). **error = 1 − métrica**
  (en regresión, 1 − R²). La prueba NO interviene en la parada: la parada usa solo validación.

## 2. Cantidades

- **R(c)** = 1 − Σ error_métrica / Σ error_pérdida, sobre los pliegues del conjunto (reducción relativa del error medio; razón de medias,
  como en el 160). Positivo = parar por métrica es mejor. Si Σ error_pérdida = 0: R = 0 si el de métrica también es 0, −1 si no.
- **IC95 de R(c):** bootstrap de percentiles de PLIEGUES, 1.000 remuestras, semilla 0 (los de `protocolo.SEMILLA_DEL_INTERVALO` y
  `REMUESTRAS_DEL_INTERVALO`, y el `_percentil` de 105-C2): las MISMAS funciones que el 160 (`analiza_c160.reduccion_relativa` / `ic_de_r`,
  importadas, no copiadas). Con menos de dos pliegues no hay IC y no hay «mejora clara» ni «empeora».
- **«Mejora clara»:** el límite inferior del IC95 de R > 0. **«Empeora de verdad»:** R < −3 % **y** el límite superior del IC95 < 0.
- **Saturados** (los 6 del 160, declarados ANTES de medir y por el mismo criterio —la mejor referencia de los miembros en la Fase 0 congelada
  ≥ 98 %—; una prueba recalcula el criterio de los artefactos y exige la lista):
  SATURADOS: mfeat-factors, optdigits, pendigits, sick, wilt, PhishingWebsites
  Los saturados NO entran en la media (a) ni en el recuento (b); SÍ entran en (c) y (d), y se listan.
- **Coste** de un intento: `wall_s` del subproceso (preparación del motor aparte; incluye ajustar, predecir train/validación/prueba y medir).
  Coste de un brazo en un conjunto = suma de `wall_s` de sus pliegues.

## 3. La regla — CUMPLE si se dan las cuatro (y la medida es completa)

- **(a)** la media de R(c) sobre los **17 no saturados** ≥ **+3 %** (comparación exacta, sin redondear);
- **(b)** «mejora clara» en ≥ **5** de los 17 no saturados (los saturados con mejora clara se listan, no cuentan);
- **(c)** **ninguno** de los 23 (saturados INCLUIDOS) «empeora de verdad»;
- **(d)** Σ sobre los 23 del coste de `tabm_metrica` ≤ **1,5 ×** Σ sobre los 23 del coste de `tabm_perdida` (razón de sumas); se informan
  también la mediana y el máximo por conjunto.

**Completitud → SIN VEREDICTO** (ni CUMPLE ni NO CUMPLE; lo comprueba el analizador antes de aplicar la regla): si falta algún conjunto de los
23, algún pliegue esperado (yeast 0–3, el resto 0–4) de algún brazo; si un intento no cuenta como medida (`failed`, cortado…) —un pliegue sin
medida es un pliegue que falta, no un cero—; si hay registros de más o repetidos; si la salida es parcial; si hay intentos del mismo brazo con
código distinto; si un brazo declara una parada que no es la suya (el de pérdida con `parada_por_metrica`, el de métrica sin ella); si la
regla, el analizador o lo que importa (`protocolo.py`, `pasada_amplia_101_c5.py`, `matrixai/estudio/metricas.py`,
`matrixai/estudio/incertidumbre.py`, `benchmarks/c160/analiza_c160.py`) no son los que midieron. Una sonda no decide nada.

**Se declara, no se decide con ello:** cuántos pliegues del brazo de métrica cayeron a la pérdida (y por qué), las épocas ejecutadas y la mejor
época de cada brazo, y cuántos se pararon por plazo y cuántos por paciencia (si casi todo para por plazo, la palanca tiene poco que elegir).

Si no cumple: la palanca no entra en el producto y el contrato 162 sigue con C2 y C3.

## 4. Tres palancas contra la misma referencia (invariante 3 del contrato)

C1, C2 y C3 son tres oportunidades de acertar por azar contra la misma referencia. El umbral de esta regla NO se endurece (D3 lo fija
igual que el 160); lo que protege es la **confirmación**: nada cumple si no repite en conjuntos que no vio.

## 5. Confirmación (§4.3 del contrato, D3)

Si y solo si los 23 CUMPLEN, la misma palanca y la misma regla se aplican UNA vez a los conjuntos de `CONFIRMACION.md` (nuevos, apartados
antes de medir nada, y distintos de los 6 del 160). Con 6 conjuntos las condiciones se escalan así: (a) media de R ≥ +3 % sobre los no
saturados; (b) «mejora clara» en ≥ la mitad (redondeada hacia arriba) de los no saturados; (c) ninguno «empeora de verdad»; (d) igual (≤ 1,5×).
Se escribe su enmienda sellada ANTES de medirlos (incluida la declaración de saturados con el mismo criterio ≥ 98 %). Cumplir en los 23 y fallar
en la confirmación = NO CUMPLE (sobreajuste a los 23).

## 6. Lo que NO se afirma

- Es la geometría de la Fase 0 (cada intento entrena con todo `entrena`, para con `valida`, plazo de su cubo), no la del estudio del Studio
  (`plazo_de_tabm`, 139-C4 reentrena con épocas fijas y entrega la última: la palanca no actúa ahí sin más trabajo, que sería de C5).
- 1 repetición y 5 pliegues, con la prueba fija de cada conjunto: el IC dice cuánto se mueve el número al reentrenar, no cuánto se movería con
  otros datos. Con 5 pliegues el IC es grueso.
- Los dos brazos comparten semilla, no ruido de reloj: el plazo es de reloj de pared, así que una máquina cargada puede cortar a un brazo antes
  que a otro. Por eso van pegados (pérdida y luego métrica en el mismo pliegue) y se declaran los cortes por plazo.
- La palanca elige la época por la métrica de VALIDACIÓN del cierre (AUROC, accuracy, R²): puede ser más ruidosa que la pérdida con una
  validación pequeña. Medirlo es el objeto del corte.
- El coste es el del arnés (datos ya preparados, 4 hilos), una razón entre dos brazos en las mismas condiciones, no el tiempo de «Usar».
