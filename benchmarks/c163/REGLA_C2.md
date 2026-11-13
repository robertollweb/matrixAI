# Contrato 163 · C2 — REGLA DE DECISIÓN (para SELLAR ANTES de medir): la rejilla de 8 configuraciones de los ÁRBOLES por validación

Fija, antes de que exista ninguna cifra de la medida real, qué se mide, cómo se compara y cuándo la rejilla cumple. Es la regla del 160 (D3 del
contrato 163, aprobada por Roberto con A el 12-11, tarjeta `ola5-40-decisiones`): reducción relativa del error ≥ +3 % sobre los no saturados,
mejora clara en ≥ 5, ninguno empeora, y un coste acotado (≤ 8× el del árbol de hoy), con confirmación en conjuntos nuevos. Su sha256 se graba en
`analiza_c163_c2.REGLA_SHA256_SELLADA` (lo pone el supervisor al sellar) y en la salida de la medida (`regla_sha256_al_medir`); `medir_c163_c2.py`
se niega a medir y `analiza_c163_c2.py` a dar veredicto si el fichero ya no es el sellado. **Cambiar una coma después de sellar es cambiar la
regla**: una enmienda nueva con su sello, no una edición.

## 1. Qué se mide

- **La palanca.** `MotorArbolLightGBM(politica).con_rejilla_por_validacion()` y `MotorArbolHGB(politica).con_rejilla_por_validacion()`
  (matrixai-engines, 163-C2). Con ella el motor ajusta OCHO veces —una por configuración de `politica_de_arboles.REJILLA_DE_8`— con el MISMO
  `train` y la MISMA `validation` que recibe, y se queda con la de menor **pérdida de validación en su mejor ronda** (la misma pérdida que decide
  su parada temprana: logloss / multi_logloss / error cuadrático); el empate lo gana el orden (el perfil de hoy primero). **Nunca mira la prueba**:
  el motor no la recibe (pruebas con nombre). Entrega el modelo de la elegida tal como lo entrega hoy la parada temprana (el modelo recortado a su
  mejor ronda): **sin reentrenar**. El reloj (`wall_seconds`) se reparte a partes iguales entre las 8.
- **La rejilla, fijada ANTES de ver ninguna cifra de calidad** (factores sobre el perfil del 120-C2 de cada tamaño; hojas × tasa × hojas mínimas):

  | # | hojas | tasa | hojas mín. | idea |
  |---|---|---|---|---|
  | 0 | ×1 | ×1 | ×1 | el perfil de hoy (la elegida nunca es peor que hoy EN VALIDACIÓN) |
  | 1, 2 | ×½, ×2 | ×1 | ×1 | eje de las hojas |
  | 3, 4 | ×1 | ×½, ×2 | ×1 | eje de la tasa |
  | 5, 6 | ×1 | ×1 | ×½, ×2 | eje de las hojas mínimas |
  | 7 | ×2 | ×1 | ×2 | más capacidad con la regularización que la acompaña |

  Perfiles: pequeño 7 hojas/10 mín./tasa 0,05; mediano 31/20/0,05; (grande 63/40/0,1, fuera de esta medida). `arboles_max` (500 / 800) y paciencia (30)
  no cambian. Por qué estas 8 y no un cubo: tres ejes a dos niveles son 27 combinaciones con el centro; con 60–200 filas de validación, 27 serían
  buscar a ciegas (D2: «más es buscar a ciegas»). Un eje cada vez dice qué eje paga y el último junta los dos que van juntos en la práctica.
- **Los cuatro brazos** (cada uno, un subproceso aislado con su tope, UNA vez por conjunto y pliegue; por motor la referencia PRIMERO y la rejilla
  inmediatamente después): **`lightgbm_hoy`** = `MotorArbolLightGBM(PoliticaPorTamano(1.0))`, **`lightgbm_rejilla`** = el mismo con la rejilla,
  **`hgb_hoy`** = `MotorArbolHGB(PoliticaPorTamano(1.0))`, **`hgb_rejilla`**. Todos se miden en ESTA pasada con el MISMO código: no se reutiliza nada
  del 155, 160, 162 ni 163-C1. Mismos datos preparados, misma semilla del intento, mismo presupuesto, 4 hilos.
- **Por qué «hoy» es la política del 120-C2 en los TRES tamaños y no el motor de la Fase 0**: la rejilla gira alrededor del perfil por tamaño, y la
  referencia justa es ESE perfil; así los dos brazos de un motor se diferencian solo en la rejilla. En producción la política rige hoy solo en
  «mediano» (L2 = 1); en «pequeño» rige la configuración fija de la Fase 0. Esta medida **no** dice nada de «Fase 0 contra rejilla» (§6).
- **Conjuntos:** los 23 no sellados pequeños y medianos de la Fase 0 (los de `datasets_declarados` de `resultado_pasada_119_c3.json` con cubo
  `pequeno`/`mediano` y `sellado: false`; el guion los lee del congelado y para si no son 23). Una repetición, todos los pliegues (yeast 0–3, el resto
  0–4), MISMAS particiones, preparación y presupuesto de pared por cubo (120 s pequeños, 300 s medianos) que 119-C3/155/160/162/163-C1.
- **Métrica de cierre:** `metrica_de_cierre_por_dataset` (AUROC / accuracy / R²) sobre la prueba fija del conjunto; **error = 1 − métrica**
  (en regresión, 1 − R²).

## 2. Cantidades (las del 160, importadas)

- **R(c, motor)** = 1 − Σ error_rejilla / Σ error_hoy, sobre los pliegues del conjunto (razón de medias). Positivo = la rejilla es mejor. Si
  Σ error_hoy = 0: R = 0 si el de la rejilla también, −1 si no.
- **IC95 de R:** bootstrap de percentiles de PLIEGUES, 1.000 remuestras, semilla 0 (`analiza_c160.reduccion_relativa` / `ic_de_r`, importadas).
- **«Mejora clara»:** límite inferior del IC95 de R > 0. **«Empeora de verdad»:** R < −3 % **y** límite superior del IC95 < 0.
- **Saturados** (los del 160, declarados ANTES de medir, `analiza_c160.SATURADOS`): mfeat-factors, optdigits, pendigits, sick, wilt, PhishingWebsites.
  No entran en (a) ni (b); SÍ en (c) y (d).
- **Coste de un intento:** `tiempo_de_ajuste` (los segundos de pared del `fit` que el motor declara, la rejilla entera incluida; no el `wall_s` del
  subproceso, que lleva arrancar Python y vuelca el modelo: un fijo que achataría la razón). Coste de un brazo = suma de los pliegues y de los conjuntos.

## 3. Cómo se combinan los dos motores (decidido AQUÍ, antes de medir)

**Cada motor se juzga POR SEPARADO** con la misma regla (a)–(d), con su propia referencia: no hay un veredicto «de los árboles» que mezcle un motor
que cumple con uno que no. Salidas posibles: LightGBM CUMPLE y HGB no (se integra solo el de LightGBM), al revés, los dos, o ninguno. Si ninguno
cumple, el contrato 163 cierra C2 sin integración. Por qué: son bibliotecas distintas con perfiles de coste y de ruido distintos; exigir que ambos
cumplan haría que un motor bueno pagase por el otro, y promediar escondería el que empeora.

## 4. La regla — CUMPLE (un motor) si se dan las cuatro (y la medida es completa)

- **(a)** la media de R sobre los **17 no saturados** ≥ **+3 %** (comparación exacta, sin redondear);
- **(b)** «mejora clara» en ≥ **5** de los 17 no saturados (los saturados con mejora clara se listan, no cuentan);
- **(c)** **ninguno** de los 23 (saturados INCLUIDOS) «empeora de verdad»;
- **(d)** Σ sobre los 23 de `tiempo_de_ajuste` de la rejilla ≤ **8 ×** Σ de `tiempo_de_ajuste` de la referencia del mismo motor (razón de sumas;
  se informan también la mediana y el máximo por conjunto). Es el techo de D3. Ocho ajustes cuestan del orden de 8× el de la referencia (el perfil de hoy es uno de los 8); las
  configuraciones de menos hojas abaratan y las de tasa ×½ (más árboles hasta `arboles_max`) encarecen. No hay margen de sobra: si pasa de 8, NO CUMPLE.

**Completitud → SIN VEREDICTO** (ni CUMPLE ni NO CUMPLE; lo comprueba el analizador antes de aplicar la regla): si falta algún conjunto de los 23,
algún pliegue esperado de algún brazo; si un intento no cuenta como medida (`failed`, cortado…) o no trae `tiempo_de_ajuste`; si hay registros de más
o repetidos; si la salida es parcial; si hay intentos del mismo brazo con código distinto; si un brazo declara lo que no es (la referencia con
`rejilla_por_validacion` o con `validacion_mejor`; la rejilla sin ella, con `aplicada: false` —sin validación útil no hay con qué elegir—, sin
exactamente 8 configuraciones con su pérdida, sin declarar `mira_la_prueba: false`, o con una elegida que no es la de menor pérdida; cualquier
brazo sin la política por tamaño aplicada); si la regla, el analizador o lo que importa (`protocolo.py`, `pasada_amplia_101_c5.py`,
`matrixai/estudio/metricas.py`, `matrixai/estudio/incertidumbre.py`, `benchmarks/c160/analiza_c160.py`) no son los que midieron. Una sonda no decide
nada.

## 5. Confirmación (D3, invariante 3 del contrato)

Si y solo si los 23 CUMPLEN **para un motor**, la misma rejilla y la misma regla se aplican UNA vez, a ese motor, a los conjuntos de
`CONFIRMACION_C2.md` (nuevos: no son los de `benchmarks/c160/`, `benchmarks/c162/` ni `benchmarks/c163/CONFIRMACION.md`; apartados antes de medir
nada). Condiciones escaladas: (a) media de R sobre los no saturados ≥ +3 %; (b) «mejora clara» en ≥ la mitad (hacia arriba) de los no saturados;
(c) ninguno «empeora de verdad»; (d) igual (≤ 8×). Los saturados de la confirmación se declaran con el criterio del 160 (mejor referencia ≥ 98 %)
en una enmienda sellada ANTES de medirlos. Cumplir en los 23 y fallar en la confirmación = NO CUMPLE (sobreajuste a los 23).

## 6. Lo que NO se afirma

- Es la geometría de la Fase 0 (cada intento entrena con todo `entrena`, para con `valida`, plazo de su cubo), no la del estudio del Studio. Allí la
  `validation` de la búsqueda es otra partición y el reajuste final entrena con árboles fijados (`arboles_fijos`): con ellos la rejilla NO se aplica
  (no hay validación con que elegir), así que llevar la elegida al reajuste es trabajo del corte de integración (C3), no de este.
- «Hoy» es el perfil del 120-C2 con parada temprana en los tres tamaños (L2 = 1), no la configuración fija de la Fase 0 ni exactamente producción (que solo
  aplica la política en «mediano»). Una rejilla que cumpla aquí mejora el perfil por tamaño; no se afirma contra el árbol fijo de 100 árboles.
- La validación que elige entre 8 es la misma que decide la parada temprana de cada una: la pérdida de validación de la elegida es optimista (ganó
  entre 8). Por eso el veredicto se mide en la PRUEBA fija y no en la validación, y pide confirmación.
- 1 repetición y 5 pliegues: el IC dice cuánto se mueve el número al reentrenar, no cuánto se movería con otros datos.
- La rejilla es UNA elección de 8 puntos; si no cumple, no se prueba otra sin enmienda. `l2` no se busca (es el eje del 120-C2).
