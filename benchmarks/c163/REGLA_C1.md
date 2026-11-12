# Contrato 163 · C1 — REGLA DE DECISIÓN (para SELLAR ANTES de medir): LightGBM con `min_data_per_group` POR TAMAÑO

Fija, antes de que exista ninguna cifra de la medida real, qué se mide, cómo se compara y cuándo la palanca `min_data_per_group` por tamaño
cumple. Es la regla del 160/162 (`benchmarks/c162/REGLA_C1.md`) adaptada a una palanca que **solo puede cambiar nada en una minoría de los
conjuntos**: reducción relativa del error, media ≥ +3 %, mejora clara, ninguno empeora, coste acotado y confirmación en conjuntos nuevos. Decisión
de Roberto del 10-11 (tarjeta `lightgbm-min-data-per-group` = A: medirlo con 100 y con uno que dependa del tamaño) y D1–D4 del 12-11
(`ola5-40-decisiones` = A). Su sha256 se graba en `analiza_c163_c1.REGLA_SHA256_SELLADA` y en la salida de la medida (`regla_sha256_al_medir`);
`medir_c163_c1.py` se niega a medir y `analiza_c163_c1.py` a dar veredicto si el fichero ya no es el sellado. **Cambiar una coma después de sellar
es cambiar la regla**: una enmienda nueva con su sello, no una edición.

## 1. Qué se mide

- **La palanca.** `MotorArbolLightGBM().con_min_data_per_group_por_tamano()` (matrixai-engines, 163-C1): pasa a LightGBM
  `min_data_per_group = min(100, max(5, filas_de_train // 20))` (el 5 % de las filas de train; suelo 5; techo 100 = la omisión de LightGBM, que se
  alcanza desde 2.000 filas) en vez del 100 de la omisión. Solo toca columnas CATEGÓRICAS (las que LightGBM parte por grupos de categorías); sin
  ellas el parámetro ni se pasa. **La fórmula se escribió y se fijó el 12-11 antes de ver ninguna cifra de calidad de la medida** (solo el sintético
  del 10-11, 120 filas y 6 códigos: 1,325 con 100 y 0,516 con 5, que fija el suelo en 5 y el orden de magnitud del 5 %). Apagada por omisión y bit a
  bit igual que antes con ella apagada (pruebas con nombre). No se toca `cat_smooth` (una sola palanca). La palanca declara lo que pasó
  (`hiperparametros.min_data_per_group_por_tamano = {pedida, aplicada, filas_de_train, min_data_per_group}`).
- **Los dos brazos** (cada uno, un subproceso aislado con su tope, UNA vez por conjunto y pliegue, la referencia PRIMERO y la palanca inmediatamente
  después): **`lightgbm_hoy`** (REFERENCIA: LightGBM de la Fase 0, 100 árboles, 15 hojas, `min_data_per_group` = 100 de la omisión) y
  **`lightgbm_min_data_per_group`** (la palanca). Ambos se miden en ESTA pasada con el MISMO código: no se reutiliza nada del 155, 160 ni 162.
  Mismos datos preparados, misma semilla del intento, mismo presupuesto, 4 hilos.
- **Conjuntos:** los 23 no sellados pequeños y medianos de la Fase 0 (los de `datasets_declarados` de `resultado_pasada_119_c3.json` con cubo
  `pequeno`/`mediano` y `sellado: false`; el guion los lee del congelado y para si no son 23). **Los otros 17 de los 40 no se miden y por qué:** los
  8 SELLADOS no se tocan; los 9 GRANDES tienen > 2.000 filas de train, donde la fórmula da exactamente 100 = la omisión (la palanca no puede cambiar
  nada y medirlos solo gastaría una noche).
- **Una repetición, todos los pliegues:** repetición 0; pliegues 0–4 (yeast, 0–3). MISMAS particiones, preparación y presupuesto de pared por cubo
  (120 s pequeños, 300 s medianos) que 119-C3/155/160/162.
- **Métrica de cierre:** `metrica_de_cierre_por_dataset` (AUROC / accuracy / R²), sobre la prueba fija del conjunto, con la cadena de la Fase 0.
  **error = 1 − métrica** (en regresión, 1 − R²).

## 2. Quién cuenta: AFECTADOS y CONTROL (declarados ANTES de medir)

Medido el 12-11 (23:07) sobre la preparación de la Fase 0, pliegue 0, solo para saber quién tiene categóricas y cuántas filas de train
(`~/.agentes/163/c1/categoricas.py`, ninguna métrica de modelo):

| grupo | conjuntos | por qué |
|---|---|---|
| **AFECTADOS (3)** | `dresses-sales` (319 filas de train, 11 categóricas), `Moneyball` (788; 2), `house_prices_nominal` (934; 43) | tienen categóricas y < 2.000 filas: la fórmula da 15, 39 y 46 |
| con categóricas, **sin efecto por construcción (2)** | `sick` (2.414; 23), `SAT11-HAND-runtime-regression` (2.841; 1) | ≥ 2.000 filas: la fórmula da 100 |
| sin categóricas (18) | los demás | la palanca no pasa ni el parámetro |

Ningún AFECTADO está saturado (el único saturado con categóricas, `sick`, no es afectado). Los 6 saturados del 160 (mfeat-factors, optdigits, pendigits,
sick, wilt, PhishingWebsites) no cambian la regla: todos son del CONTROL.
Con **solo 3 afectados** la regla tiene poca potencia y se dice: un CUMPLE sobre 3 conjuntos es una indicación que la confirmación (§5) tiene que
repetir; no una prueba. Es lo que hay: con categóricas y pocas filas, el resto de la Fase 0 no tiene nada que la palanca pueda tocar.

## 3. Cantidades

- **R(c)** = 1 − Σ error_palanca / Σ error_referencia, sobre los pliegues del conjunto (reducción relativa del error medio, como en el 160). Positivo =
  la palanca es mejor. Si Σ error_referencia = 0: R = 0 si el de la palanca también es 0, −1 si no.
- **IC95 de R(c):** bootstrap de percentiles de PLIEGUES, 1.000 remuestras, semilla 0: las MISMAS funciones del 160 (`analiza_c160.reduccion_relativa` /
  `ic_de_r`, importadas). Con menos de dos pliegues no hay IC.
- **«Mejora clara»:** el límite inferior del IC95 de R > 0. **«Empeora de verdad»:** R < −3 % **y** el límite superior del IC95 < 0.
- **Coste** de un intento: `wall_s` del subproceso. Coste de un brazo en un conjunto = suma de los `wall_s` de sus pliegues.

## 4. La regla — CUMPLE si se dan las cinco (y la medida es completa)

- **(a)** la media de R sobre los **3 afectados** ≥ **+3 %** (comparación exacta, sin redondear);
- **(b)** «mejora clara» en ≥ **2 de los 3** afectados (la mitad redondeada hacia arriba; el 160 pedía 5 de 17 ≈ 30 %: con 3 conjuntos, 1 de 3 sería
  un azar de moneda);
- **(c)** **ninguno** de los 23 «empeora de verdad»;
- **(d)** Σ sobre los 23 del coste de la palanca ≤ **1,5 ×** Σ del coste de la referencia (razón de sumas; se informan mediana y máximo);
- **(e) EL CONTROL:** en cada conjunto y pliegue donde la palanca NO actuó (los 20 no afectados), la métrica de la palanca coincide con la de la
  referencia con una tolerancia de 1e-9. Si no: el instrumento no es determinista o la palanca actúa donde no debe, y eso es **SIN VEREDICTO**, no
  un empeora ni un mejora.

**Completitud → SIN VEREDICTO** (ni CUMPLE ni NO CUMPLE; lo comprueba el analizador antes de aplicar la regla): si falta algún conjunto de los 23, algún
pliegue esperado de algún brazo; si un intento no cuenta como medida; si hay registros de más o repetidos; si la salida es parcial; si hay intentos del
mismo brazo con código distinto; si un brazo declara una palanca que no es la suya (la referencia con `min_data_per_group_por_tamano`, la palanca sin
haberla pedido); si en un AFECTADO la palanca no actuó, o en un no afectado actuó (la lista de §2 no cuadra con lo medido); si la regla, el
analizador o lo que importa (`protocolo.py`, `pasada_amplia_101_c5.py`, `matrixai/estudio/metricas.py`, `matrixai/estudio/incertidumbre.py`,
`benchmarks/c160/analiza_c160.py`) no son los que midieron. Una sonda no decide nada.

Si no cumple: la palanca no entra en el producto y el contrato 163 sigue con C2 (la rejilla).

## 5. Confirmación (invariante 3 del contrato)

Si y solo si los 23 CUMPLEN, la misma palanca y la misma regla se aplican UNA vez a los conjuntos de `CONFIRMACION.md` (nuevos, apartados antes de
medir nada y distintos de los del 160 y del 162). Con ellos no se sabe de antemano quién es afectado, así que los AFECTADOS son los conjuntos donde la
propia medida declara la palanca actuando (valor ≠ 100); hacen falta **al menos 2** (si no, SIN VEREDICTO). Condiciones escaladas: (a) media de R sobre
los afectados ≥ +3 %; (b) «mejora clara» en ≥ la mitad (redondeada hacia arriba) de los afectados; (c) ninguno de los conjuntos «empeora de
verdad»; (d) igual (≤ 1,5×); (e) el control igual. Se escribe su enmienda sellada ANTES de medirlos. Cumplir en los 23 y fallar en la confirmación =
NO CUMPLE (sobreajuste a los 3).

## 6. Lo que NO se afirma

- Es la geometría de la Fase 0 (LightGBM de 100 árboles y 15 hojas, sin política por tamaño ni parada temprana), no la del estudio del Studio, que usa
  la política del 120-C2 (la palanca también actúa ahí —pruebas con nombre— pero no se mide aquí; sería del corte de integración, C3).
- 1 repetición y 5 pliegues: el IC dice cuánto se mueve el número al reentrenar, no cuánto se movería con otros datos; con 3 afectados, menos aún.
- La fórmula (5 % de train, suelo 5, techo 100) es UNA elección; no se explora una rejilla de valores (sería buscar a ciegas). Si no cumple, no se
  prueba otro valor sin enmienda.
- Un conjunto cuyas categóricas tienen ≤ 4 valores se parte en one-hot en LightGBM, donde `min_data_per_group` no cuenta: la palanca puede no cambiar
  nada en un afectado (entonces R = 0 ahí, y cuenta).
- Cambiar `min_data_per_group` mueve la evidencia de la Fase 0 de LightGBM en los conjuntos con categóricas y pocas filas: si cumple, la integración
  lo declara y re-mide lo que cite (invariante 4 del contrato).
