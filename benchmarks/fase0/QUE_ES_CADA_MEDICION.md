# Qué es cada JSON de medición de la Fase 0

Escrito el **2026-09-15** porque hacía falta: aquí conviven **cinco** mediciones
de aparentemente lo mismo, y **ninguna de las cuatro primeras dice, dentro de
sí, cuál es**. Solo lo dice el nombre del fichero — y un nombre **no está
dentro del sello**: se puede renombrar sin que el digest deje de cuadrar.

Este índice no toca ni un byte de los artefactos. Sus digests siguen siendo los
suyos; lo que se añade es el contexto que les falta.

Todo lo de abajo está **medido leyendo los ficheros**, no copiado de notas.

---

## La que sostiene la cartera aprobada

**`pasada_exploratoria_101_c3_conforme_20260914.json`**
· 1.260 intentos · 12 conjuntos de datos · **los 7 motores del protocolo**
· **1.260 de 1.260 completados, cero fallos**
· digest de resultados crudos: **`3cb87513b80ebf74…`**

**Es la evidencia que cita la entrada de cartera** (`matrixai_engines/cartera.py`,
campo `evidencia_digest`), y la cita **cuadra**: comprobado el 2026-09-15 que el
digest de la entrada es exactamente el del artefacto. De aquí sale
**catboost 12/12 CUMPLE** y **lightgbm 9/12 NO**.

`anclable: false`, y **conviene saber por qué**: el único fichero sin seguir por
git al medir era **otro JSON de resultados** — un dato, no código. Ningún
fichero del camino de medición estaba modificado. El aviso es correcto y
conservador; la causa es inmaterial.

---

## Las otras tres, y qué las distingue

**`pasada_exploratoria_101_c3_resultado.json`** — 2026-09-07
· 720 intentos · 12 conjuntos · **4 motores** · **36 FALLIDOS**
· Sin bloque de procedencia: se midió antes de que existiera.
**Sus 36 fallos no eran de ningún algoritmo**: eran tres defectos de cableado
propios, uno de los cuales devolvía los datos de entrada como si fueran la
predicción, en silencio. Se conserva porque es el registro de lo que se midió
ese día, no porque sus números valgan.

**`pasada_exploratoria_101_c3_remedida_20260913.json`** — 2026-09-13
· 720 intentos · 12 conjuntos · 4 motores · **720 de 720, cero fallos**
· **`anclable: true` — es la ÚNICA de las cinco que lo es.**
La repetición limpia tras reparar los tres defectos. Con ella, la regla de
cierre daba 10/12 = 0,833 y parecía que el listón se alcanzaba. **Ese número
era estrecho, no falso**: compitieron cuatro motores de los siete.

**`pasada_exploratoria_101_c3_siete_motores_20260913.json`** — 2026-09-13
· 1.260 intentos · 12 conjuntos · **7 motores** · **1 fallido**
· `anclable: false`, y **aquí el aviso SÍ es material**: al medir estaban
modificados y sin commitear `pasada_exploratoria_101_c3.py` —el propio guion de
la medición— y su test. **El commit `e94f4169` no identifica el código que
produjo estos números, y no hay forma de volver a atarlos.**
Es donde apareció por primera vez el 12/12 de catboost. **No se cayó nada
porque la pasada conforme del día siguiente lo volvió a medir desde un árbol
limpio de código, y es ésa la que cita la cartera** — pero eso fue suerte de
secuencia, no método.

---

## La que manda ahora — TERMINADA

**`pasada_amplia_101_c5_resultado.json`** — lanzada el 2026-09-15, parada a las
2 horas, relanzada, **terminada el 2026-09-16**
· **40 conjuntos, con los 8 sellados dentro** · 7 motores · **3.479 intentos,
3.464 completados y 15 fallidos** · **61.836,8 s = 17,18 h** de reloj
· digest de entorno **`3646af61b155c132`** · commit `5c752ba`
· **Es la PRIMERA ANCLABLE de las cinco** (`anclable: true`, sin avisos), y la
única que se declara a sí misma: lleva `corte: "101-C5"` dentro.

**EL VEREDICTO**, con la regla de cierre que 101-C1 registró con hash antes de
medir (a menos de 2 puntos del mejor en ≥80 % de los conjuntos; un fallo cuenta
como conjunto perdido):

| motor | | | binaria | multi | regresión | SELLADOS | grande |
|---|---|---|---|---|---|---|---|
| lightgbm | 34/40 | **0,850 CUMPLE** | 16/20 | 8/10 | 10/10 | 8/8 | 9/10 |
| sklearn.hgb | 34/40 | **0,850 CUMPLE** | 16/20 | 8/10 | 10/10 | 8/8 | 9/10 |
| catboost | 33/40 | **0,825 CUMPLE** | 18/20 | 5/10 | 10/10 | 6/8 | 8/10 |
| xgboost | 30/40 | 0,750 no | 14/20 | 6/10 | 10/10 | 7/8 | 8/10 |
| sklearn.lineal | 15/40 | 0,375 no | 11/20 | 3/10 | 1/10 | 3/8 | 2/10 |
| matrixai.dense.torch_cpu | 15/40 | 0,375 no | 8/20 | 4/10 | 3/10 | 5/8 | 2/10 |

**QUÉ LE HACE A LA CARTERA.** El `12/12 = 1,000` de catboost que cita la entrada
sellada **no sobrevive**: 0,825, y deja de ser «el único que cumple» —cumplen
tres—. Y sus pérdidas caen **justo en el hueco que esa misma entrada declaró no
haber medido** («de esta pasada no se sigue NADA sobre multiclase, regresión ni
cubo grande»): **cinco de sus siete son multiclase** y **dos son sellados**
(`Satellite`, `letter`). En binaria sigue siendo el mejor de los seis.

**LO QUE ESTOS NÚMEROS NO DICEN, y hay que leerlo pegado a lo de arriba**: que
lightgbm sea mejor que catboost. Es **un** conjunto de diferencia, y leyendo los
intervalos al 95 % los rangos plausibles se solapan — lightgbm 33..38, catboost
30..35, sklearn.hgb 33..37. Lo sólido es que **el 1,000 era del alcance viejo,
no del motor**.

**LOS 15 FALLOS son todos del motor denso**, en tres conjuntos:
· **10 por RELOJ** (`KDDCup09_appetency`, `Allstate_Claims_Severity`): «superó
630,0 s … el proceso se ha matado desde fuera». Estaba **declarado por escrito
antes de lanzar** que probablemente no cabría en el cubo grande — es un
resultado sobre el motor, no un accidente.
· **5 por DEFECTO NUESTRO** (`okcupid-stem`): `DatasetProjectError` porque un
valor legítimo de `ethnicity` —«asian, pacific islander»— lleva una coma que
`Categorical[...]` no sabe representar (`args.split(",")`, sin escape). Misma
familia que el `None` de `house_prices_nominal`: **un dato real que nuestra
representación no expresa, con un mensaje que culpa a quien trae el CSV**.
Medido: afecta a **4 de las 18 columnas categóricas** de ese conjunto, y
`column_category_overrides` —la escapatoria del producto— **llama a la misma
guarda**, así que hoy ese CSV no se puede usar de ninguna forma.

**Y la densa gana DOS conjuntos por ser la mejor de los seis**: `diabetes` (por
0,017 puntos, o sea nada) y **`Satellite`, que es SELLADO**, por 0,96 puntos
sobre el segundo.

**CÓMO HAY QUE LEER «2 PUNTOS» EN CADA TAREA, porque no significa lo mismo.**
La regla registrada dice «a menos de 2 puntos del mejor» y el código multiplica
las tres métricas por 100 por igual. O sea que en las 10 tareas de regresión el
listón literal es **ΔR² ≤ 0,02**. Es la lectura literal de lo prerregistrado y
**no se toca** —cambiar una regla registrada después de ver los números es
justo lo que el pre-registro impide—, pero hay dos cosas que decir:

1. **Relativamente es más estricto**: 0,02 sobre el rango útil de R² (≈0..1)
   pide el doble de precisión que 0,02 sobre el rango útil de AUROC (≈0,5..1).
2. **Y aun así, regresión es donde MÁS se cumple**: 44 de 59 medidas (0,746),
   por encima de binaria (0,697) y de multiclase (0,576). Los tres motores de
   la cartera hacen 10/10, con sus peores distancias en 1,452 (lightgbm),
   1,355 (sklearn.hgb) y 1,880 (catboost). Ese 10/10 es un resultado **más**
   sólido por el listón, no menos.

**LO QUE SÍ ES ASIMÉTRICO DE VERDAD ES LA COLA.** R² no tiene suelo, así que un
modelo peor que predecir la media da distancias que en AUROC no pueden existir:
**nuestra red densa marca 8.147,2 puntos de distancia en
`house_prices_nominal`** —un R² unos 81 por debajo del mejor—, mientras que en
AUROC lo peor posible son 100 puntos. Al leer `distancia_en_puntos` en una fila
de regresión, un número de cuatro cifras no es un error del artefacto: es un
modelo que no sirve.

**POR QUÉ SE PARÓ A LAS 2 HORAS**, que es lo que hay que saber para leerla. A
los 1.904 intentos aparecieron sus primeros 15 fallos: todos de
`house_prices_nominal` y del motor nuestro, muriendo en **0,9 s de un tope de
120**. No era el reloj. La lista de marcadores de dato ausente del núcleo
incluía `"none"`, y ese conjunto **declara `None` en su cabecera ARFF como nivel
válido** —«sin revestimiento de mampostería»—: **864 de 1.460 filas** se leían
como ausentes. La guarda que hizo saltar todo **no era el defecto: detectó una
ambigüedad que existía de verdad.** Se reparó en la raíz (`b2fad46`, `83b7f70`)
y se relanzó con `--forzar`, re-ejecutando los 1.904. El artefacto parado se
conserva fuera del árbol: `c5_parcial_antes_de_parar.json`.

## Antes de relanzar cualquiera de ellas

Está en `CLAUDE.md`, sección «ANTES DE RELANZAR UNA PASADA», y son tres cosas
de las que **dos son trampas**: un intento fallido se reutiliza del caché, el
digest del caché cubre 20 de los 112 módulos del camino, y tocar el guion
invalida la caché entera.
