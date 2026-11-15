# Contrato 169 — REGLA DE DECISIÓN (para SELLAR ANTES de medir): confirmar la rejilla de HGB en 6 conjuntos nuevos

Esta regla fija, antes de que exista ninguna cifra de estos 6 conjuntos, qué se mide, cómo se compara y cuándo la rejilla de HGB queda CONFIRMADA. Su sha256
se graba en `analiza_c169.REGLA_SHA256_SELLADA` (lo pone el supervisor al sellar) y en la salida de la medida (`regla_sha256_al_medir`).
`medir_c169.py` se niega a medir, y `analiza_c169.py` a dar veredicto, si este fichero ya no es el sellado. **Cambiar una coma después de sellar es
cambiar la regla.**

**Esta regla NO es la del 163-C2.** Aquella sigue diciendo NO CUMPLE sobre los 23. Esta cambia UNA cosa, el trato de los saturados en (c), y
**se decidió DESPUÉS de ver que HGB cayó por `sick`**, un saturado: −0,42 puntos de AUROC absolutos, que la razón relativa sobre un error de
~0,006 convierte en −68 %. Por eso la deciden **solo** los 6 conjuntos nuevos, y los 23 no se usan para nada.

## 1. Qué se mide
- **Los dos brazos**, los mismos del 163-C2 y con el mismo código: **`hgb_hoy`** = `MotorArbolHGB(PoliticaPorTamano(1.0))` y **`hgb_rejilla`** = el
  mismo con `.con_rejilla_por_validacion()`. Cada uno corre en un subproceso aislado con su tope, UNA vez por conjunto y pliegue, la referencia
  primero. LightGBM no se mide.
- **Los conjuntos**: los 6 de `benchmarks/c163/CONFIRMACION_C2.md`, y ninguno más: 458 analcatdata_authorship, 1067 kc1,
  1464 blood-transfusion-service-center, 1475 first-order-theorem-proving, 44960 energy_efficiency y 44962 forest_fires. Cada fichero se
  comprueba contra el sha256 de esa tabla. Si alguno no cuadra, no se mide. **Los atributos que OpenML marca como `ignore_attribute`
  no entran como predictores.** Se decidió ANTES de medir, al ver en la sonda que `energy_efficiency` traía `cooling_load`, su segundo
  objetivo: era una fuga. Cada exclusión se declara en la salida.
- **El protocolo**, el de los 23: 1 repetición y 5 pliegues (0–4). Se mantienen la partición de la Fase 0 (prueba fija, `entrena`/`valida`),
  su preparación y el presupuesto de pared por cubo (120 s pequeño, 300 s mediano), con 4 hilos.
- **Métrica de cierre**: `metrica_de_cierre_por_dataset` (AUROC en binaria, accuracy en multiclase, R² en regresión) sobre la prueba fija.
  **Error = 1 − métrica.**

## 2. Cantidades
- **R, su IC95, «mejora clara» y «empeora de verdad» (no saturados)**: son las del 160, importadas sin cambios (`analiza_c160.reduccion_relativa`,
  `ic_de_r`, `clasificar`). Mejora clara: límite inferior del IC95 de R > 0. Empeora de verdad: R < −3 % **y** límite superior del IC95 < 0.
- **Saturado**: un conjunto cuya media de la métrica de cierre de **`hgb_hoy`** sobre sus 5 pliegues es **≥ 0,98**. Es el criterio del 160
  («la mejor referencia ≥ 98 %») con la única referencia que hay aquí. Lo calcula el analizador, de forma mecánica, a partir del brazo de
  referencia; la rejilla no influye en quién es saturado.
- **D (solo saturados)**: media por pliegues de (métrica_rejilla − métrica_hoy), emparejada por pliegue y en puntos de la métrica, no relativa.
  **IC95 de D**: bootstrap de percentiles de pliegues, 1.000 remuestras, semilla 0 y 95 %, el mismo bootstrap que el de R.
- **«Empeora de verdad» en un SATURADO**: D < **−0,01** (más de un punto de la métrica perdido) **y** límite superior del IC95 de D < 0.
- **Coste**: `tiempo_de_ajuste` declarado por el motor, sumado sobre pliegues y conjuntos (no el `wall_s` del subproceso).

## 3. La regla — CONFIRMADA si se dan las cinco (y la medida es completa)
Sea N el número de conjuntos NO saturados de los 6.
- **(0) potencia mínima**: N ≥ **3**. Si N < 3, el veredicto es **NO CONCLUYENTE**: no confirma ni desmiente, y el contrato cierra sin
  integración.
- **(a)** la media de R sobre los N no saturados es ≥ **+3 %**, comparada sin redondear;
- **(b)** hay «mejora clara» en ≥ **⌈N/2⌉** de los N;
- **(c)** **ninguno** de los 6 «empeora de verdad»: los no saturados por R (§2) y los saturados por D (§2);
- **(d)** Σ `tiempo_de_ajuste` de la rejilla ≤ **8 ×** Σ el de `hgb_hoy`, como razón de sumas sobre los 6. Se informan también la mediana y el
  máximo por conjunto.

Si (0) se cumple y falla cualquiera de (a)–(d), el veredicto es **NO CUMPLE**.

**Orden y bordes.**
- **El orden de prioridad**: primero SIN VEREDICTO (completitud), después NO CONCLUYENTE (N < 3, aunque falle otra condición), y después
  CONFIRMADA o NO CUMPLE.
- **Comparaciones**: todas son exactas y sin redondear, incluida «D < −0,01».
- **Coste nulo de la referencia**: si Σ `tiempo_de_ajuste` de `hgb_hoy` es 0, (d) no se cumple.

**Completitud → SIN VEREDICTO.** La medida no da veredicto si pasa cualquiera de estas cosas:
- falta un conjunto o un pliegue de algún brazo;
- un intento no cuenta como medida o no trae `tiempo_de_ajuste`;
- hay registros de más o repetidos;
- un brazo declara lo que no es (las mismas comprobaciones de brazo que en el 163-C2 §4);
- un sha256 de los datos no cuadra;
- la regla, el analizador o sus dependencias no son los que midieron.

Una sonda no decide nada.

## 4. Lo que se dirá con el veredicto, sea cual sea
- El umbral de −0,01 de los saturados se fijó después de ver `sick`, y en `sick` habría quedado dentro (−0,42 puntos).
- Con 6 conjuntos pequeños, en su mayoría de < 850 filas, la confirmación tiene poca potencia. Un NO CUMPLE aquí no demuestra que la rejilla
  sea mala, y un CONFIRMADA no la integra sola: integrarla es una decisión nueva de Roberto.
- Se mide en la geometría de la Fase 0, no en la del estudio del Studio (163-C2 §6, sin cambios).
