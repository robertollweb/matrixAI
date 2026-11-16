# Contrato 170 — REGLA DE DECISIÓN (SELLADA ANTES de medir): ¿entra el EBM (`interpret.ebm`) en el producto como «versión legible al lado»?

**Estado: SELLADA por el supervisor el 16-11, antes de medir** (revisada: se mantienen las decisiones del redactor de §2–§5). **Lo sella el supervisor** (pone el sha256 de este fichero en `analiza_c170.REGLA_SHA256_SELLADA` y escribe el sello del código con
`sellar_codigo_c170.py`) **antes de encolar la medida**. `medir_c170.py` se niega a medir, y `analiza_c170.py` a dar veredicto, si este fichero ya no es el
sellado. Cambiar una coma después de sellar es cambiar la regla. Esta regla se escribió antes de que exista **ninguna** cifra del EBM sobre los conjuntos de la
medida (la única cifra que hay es la del C0: UN pliegue de 5 conjuntos, «orientativa, NO la medida del C1»).

## 0. Qué decidió Roberto (contrato 170 §12, 16-11 13:50) — y lo que esta regla NO decide

> «**Versión legible al lado**»: el EBM NO compite por ganar el estudio. En cada estudio se entrena también y se enseña como «versión legible», con su distancia
> medida al ganador. **Entra en el producto si queda a ≤ 2 puntos del mejor de la cartera en ≥ 80 % de los conjuntos medibles** (régimen: no sellados con
> ≤ 100 columnas). La regla exacta se escribe y se sella antes del C1; el C1 mide también lo que añade al tiempo de un estudio.

Esta regla traduce esa frase a condiciones comprobables. Lo que **decide el redactor** (no Roberto) y por tanto se dice aquí con su motivo: (1) que «a ≤ 2 puntos» se
exige **sobre la media de pliegues emparejados** y no sobre un intervalo (§2); (2) cómo se tratan un EBM que no completa un conjunto y un conjunto sin rival comparable
(§3); (3) el mínimo de conjuntos comparables (§3); y (4) cómo se lee el coste (§5). Un CUMPLE **no integra** el EBM: abre el C2 con el visto bueno de Roberto.

## 1. Qué se mide

- **Motor**: `interpret.ebm` (`matrixai_engines.motores.interpret_ebm.MotorEBM`) con **`interpret-core==0.7.8`** y **sus valores por omisión** (`interactions="3x"`,
  `outer_bags=14`, `learning_rate=0.015`, `max_rounds=50000`, `early_stopping_rounds=100`, `validation_size=0.15` interna, `max_bins=1024`), `n_jobs` = 4 hilos del
  presupuesto, `random_state` derivado de la semilla del intento. Ajusta SOLO con las filas `entrena` del pliegue (la `valida` del protocolo no se usa) y se evalúa en la
  prueba fija. Sin búsqueda de hiperparámetros: la misma configuración que midió el C0.
- **Protocolo**: el **v2** (`protocolo_exploratorio_v2.json`, versión `113-C0.v2`, digest `1b4e902e61e06edcfda8e9f4ccef1232a233d46718089c5f0a2f814feb9dc878`):
  la **misma partición** (prueba fija, `entrena`/`valida`, `plan_digest` de cada conjunto), **misma preparación** (la del motor LightGBM: categóricas sin codificar a
  entero), **semillas, repeticiones y pliegues** de la v2 (3 repeticiones × 5 pliegues en pequeño y mediano, 1 × 5 en grande; `yeast` trae 12 por límite de su
  partición), **presupuesto de pared por cubo** (120 s pequeño, 300 s mediano, 600 s grande; el subproceso mata el intento al pasarse) y **métrica de cierre**
  (AUROC en binaria, accuracy en multiclase, R² en regresión), sobre la prueba fija. Cada registro graba y el analizador COMPRUEBA esos cinco datos contra los de la v2.
- **Régimen** (la decisión de Roberto, escrita exacta): los conjuntos **NO sellados** del protocolo v2 con **≤ 100 predictores** (`n_predictores` de
  `particion_por_dataset` de `pasada_v2_113_resultado.json`). Son **24**:
  `Amazon_employee_access`, `Moneyball`, `PhishingWebsites`, `balance-scale`, `climate-model-simulation-crashes`, `connect-4`, `diabetes`, `dresses-sales`, `house_16H`,
  `house_prices_nominal`, `house_sales`, `jm1`, `kc2`, `kick`, `okcupid-stem`, `optdigits`, `ozone-level-8hr`, `pc1`, `pc4`, `pendigits`, `sick`, `wilt`, `wine_quality`, `yeast`.
  Quedan **fuera del régimen** (más de 100 columnas, con su número): `APSFailure` 170, `Allstate_Claims_Severity` 130, `Internet-Advertisements` 1.558,
  `KDDCup09_appetency` 230, `SAT11-HAND-runtime-regression` 116, `mfeat-factors` 216, `micro-mass` 1.300 y `us_crime` 126. **Los 8 sellados no se tocan.**
  El régimen es una función de esa tabla, no una lista a ojo: el medidor comprueba, con los datos leídos, que cada conjunto tiene el número de predictores que la v2 registró.
- **El rival** (por conjunto): **el mejor de la cartera en ese conjunto**, con sus resultados **YA medidos, sin remedir**: `lightgbm` y `sklearn.hgb` de
  `pasada_v2_113_resultado.json` y `matrixai.dense.tabm_cpu` (TabM) de `resultado_pasada_119_c3.json` (mismo protocolo v2: sus `plan_digest` son idénticos, medido y
  comprobado por el analizador). «Mejor» = **la mayor media de la métrica de cierre sobre los pliegues esperados del conjunto**, entre los rivales que **completaron TODOS esos
  pliegues** (un rival con un pliegue fallido o ausente no compite en ese conjunto, y se dice). Empate exacto: el primero en el orden `lightgbm`, `sklearn.hgb`, TabM.
  `xgboost`, `catboost`, el lineal y la red densa vieja de la v2 NO son rivales de esta regla (Roberto nombró LightGBM, HGB y TabM).

## 2. La distancia y la condición «a ≤ 2 puntos»

- **Distancia** de un conjunto: `d = media(EBM) − media(mejor rival)` en la métrica de cierre **cruda** (AUROC, accuracy o R², todas en [−∞, 1] con **más es mejor**), las dos
  medias sobre **exactamente los mismos pliegues** (es la media de las diferencias emparejadas por repetición y pliegue). **Puntos = centésimas de la métrica, no relativos.**
- **«A ≤ 2 puntos del mejor» = `d ≥ −0,02`**, comparación **exacta y sin redondear** (en coma flotante: `d >= -0.02`; −0,0200000001 NO cumple, −0,02 exacto SÍ si la
  aritmética lo da; el informe enseña los diez decimales para que el borde se vea). Un EBM que supera al mejor (`d > 0`) cumple.
- **Se exige sobre la MEDIA, no sobre un intervalo.** Motivos: (i) es la misma cantidad con la que la regla de cierre de la cartera (`protocolo.aplicar_regla_de_cierre`,
  «a 2 puntos del mejor del campo») compara cada conjunto —medias por conjunto—, y la frase de Roberto es esa; (ii) exigir que el límite superior de un IC95 de pliegues
  llegue a −0,02 (o el inferior) castigaría a los conjuntos con pocos pliegues o ruidosos por el ruido, no por la distancia, y haría depender el veredicto de qué conjunto es
  pequeño. **El IC95 de `d` (bootstrap de percentiles de pliegues, 1.000 remuestras, semilla 0, `protocolo._intervalo_pareado`) se informa para cada conjunto y NO decide**; el
  informe cuenta además en cuántos conjuntos el IC cruza la línea de −0,02, para que quien lea vea cuántos cumplen «por poco».

## 3. La regla — CUMPLE si se dan las dos (y la medida es completa)

Sea **R = 24** (el régimen), **F** los conjuntos donde el EBM **no completa** todos sus pliegues, **N** los **no comparables** y **M = R − N** los **medibles**.

- **(a) potencia mínima**: **M ≥ 20**. Si M < 20 el veredicto es **NO CONCLUYENTE** (con menos de 20 conjuntos comparables la fracción del 80 % ya no dice lo que dice).
- **(b)** `C / M ≥ 0,8`, con **C** = el número de conjuntos medibles en que **el EBM completó todos sus pliegues Y `d ≥ −0,02`**. Comparación **exacta de enteros**
  (`5·C ≥ 4·M`): con M = 24, **C ≥ 20** (19/24 = 0,792 NO cumple); con M = 23, C ≥ 19; con M = 22, C ≥ 18; con M = 21, C ≥ 17; con M = 20, C ≥ 16.

Si (a) se cumple y falla (b), el veredicto es **NO CUMPLE**. Si las dos se cumplen, **CUMPLE**.

**Un EBM que NO completa un conjunto (fallido) cuenta como NO cumple, y SIGUE en el denominador.** «Completa» = **todos** los pliegues esperados con `estado` de medida
(`completed`/`completed_budget_limited`) y la métrica de cierre finita. Cualquier otro estado (`failed`, tope de pared agotado, memoria, `cancelled`…) deja el conjunto como
fallido ENTERO: no se promedian los pliegues buenos. Motivo: en un estudio del producto el EBM correría con su propio tope de tiempo; una «versión legible» que no llega al final
en un conjunto del régimen no está ahí cuando se la pide, y retirar los fallos del denominador dejaría que el motor pasara la regla fallando justo donde es más caro (lo mismo
que dice la regla de cierre de la cartera: «un fallo cuenta como dataset perdido»). **No se reintenta ni se alarga el tope después de ver el resultado.** El informe separa cuántos
conjuntos no cumplen por **distancia** y cuántos por **fallo**, y da el motivo de cada fallo, para que un NO CUMPLE causado por el coste no se lea como uno causado por el acierto.

**Un conjunto es NO COMPARABLE** si **ningún** rival de §1 completó todos los pliegues esperados. Sale del denominador (se cuenta en N, con su nombre y motivo) y no hay
ajuste posterior. Hoy los 24 tienen los tres rivales completos (comprobado contra los artefactos); la regla lo dice para que no se improvise si un artefacto cambia.

**Orden de prioridad**: primero **SIN VEREDICTO** (completitud, §4), después **NO CONCLUYENTE** (M < 20), después **CUMPLE** / **NO CUMPLE**.

## 4. Completitud → SIN VEREDICTO

La medida no da veredicto si pasa cualquiera de estas cosas:
- **falta un conjunto del régimen o un pliegue esperado**, o sobra uno (conjuntos o registros fuera de la regla), o hay registros **repetidos**; o la salida es **parcial**;
- un registro que cuenta como medida no trae su métrica de cierre finita ni `tiempo_de_ajuste` finito y ≥ 0;
- un registro declara lo que no es: otra métrica de cierre que la de su tarea, otra semilla / presupuesto de pared / `plan_digest` / `n_test` que los de la v2 para ese pliegue,
  distinto número de hilos que 4, `entorno_digest` o `motor_digest` distintos del de la salida (código distinto en el mismo conjunto de intentos), otra versión de `interpret-core`
  que la 0.7.8, o el sha256 de los datos distinto del que midió la v2;
- el conjunto declara un número de predictores distinto del registrado en la v2, o está marcado como sellado;
- las referencias de rival (v2 y TabM) no se pueden leer, o su `plan_digest` por conjunto no coincide con el de la v2;
- **la regla, el analizador, el medidor, el motor o sus dependencias no son los sellados** (`SELLO_DEL_CODIGO.json`), o no son los que midieron.

**Una sonda no decide nada** (veredicto SONDA, código de salida propio).

## 5. El coste (se mide y se publica; NO entra en el veredicto)

El C1 mide también lo que añade el EBM al tiempo de un estudio. Por pliegue se registra `tiempo_de_ajuste` (los segundos de pared del `fit` del motor, no el del subproceso) y,
aparte, `wall_s` del subproceso (que lleva arrancar Python y volcar el modelo: la diferencia es lo que cuesta serializar las tablas). El informe da, por conjunto, la media de
segundos de ajuste del EBM por pliegue, la de LightGBM y la **suma de los tres rivales** en esos mismos pliegues (lo que cuesta ajustar la cartera de rivales una vez); y en
total: Σ segundos del EBM, mediana y máximo por conjunto, la **razón de sumas EBM / (LightGBM + HGB + TabM)**, y cuántos pliegues pasaron del tope. «**Lo que añade a un estudio**» se
lee como **un ajuste más del EBM por estudio** (la versión legible se entrena una vez): su `tiempo_de_ajuste` medio por conjunto, y esa razón frente al coste de ajustar los tres
rivales. Es una medida de la geometría de la Fase 0 (un pliegue de entrenamiento), no del estudio del Studio, que reparte y remuestrea distinto.

## 6. Lo que se dirá con el veredicto, sea cual sea

- **No es la misma configuración «de fábrica» que el rival**: `lightgbm` de la v2 son 100 iteraciones fijas sin parada temprana; HGB es el de la política por tamaño; TabM es el de la
  receta 119; el EBM lleva su parada temprana con validación interna y 14 bolsas. Se compara lo que cada uno hace **por omisión en el protocolo**, que es lo que el protocolo dice.
- El «mejor de la cartera por conjunto» se elige **mirando las mismas pruebas con las que se compara**: favorece al rival (es un máximo de tres medias) y es lo que pidió Roberto.
- Es **un solo pliegue de entrenamiento de la Fase 0** (15 pliegues en pequeño y mediano, 5 en grande), no el estudio completo del Studio; y la medida es **por conjunto**, no por
  filas: el 80 % cuenta conjuntos, todos con el mismo peso (un conjunto de 500 filas pesa lo que uno de 70.000).
- Los 8 conjuntos con más de 100 columnas **no se miden**: el veredicto no dice nada de ellos (C0: con las parejas, el coste del EBM crece con las columnas y no está medido ahí).
- Las cifras del C0 (un pliegue de 5 conjuntos) eran orientativas y **no cuentan**.
- Un CUMPLE abre el C2 (integración, la explicación por columna y por fila, la exportación; el ONNX está apagado con su motivo); un NO CUMPLE cierra el contrato «descartada tras medir».
