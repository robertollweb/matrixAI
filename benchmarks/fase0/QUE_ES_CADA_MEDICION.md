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

## La que está corriendo

**`pasada_amplia_101_c5_resultado.json`** — lanzada el 2026-09-15, **parada a
las 2 horas y relanzada**
· **40 conjuntos de datos, con los 8 sellados dentro** · 7 motores · 3.479
intentos previstos · un punto de control por repetición
· **Esta sí se declara a sí misma**: lleva `corte: "101-C5"` dentro. El guion
nuevo aprendió lo que a los cuatro de arriba les falta.
· **Y es la PRIMERA ANCLABLE de las cinco** (`anclable: true`, sin avisos). Las
cuatro de arriba no lo son. Salió así porque antes de relanzar se commiteó la
reparación y **se retiró el JSON parcial**, dejando los dos repos limpios: el
número que salga de aquí se podrá volver a atar a un commit.
· Mientras el fichero diga `parcial: true`, **está a medias**: no es evidencia
de nada todavía.

**POR QUÉ SE PARÓ, que es lo que hay que saber para leerla.** A los 1.904
intentos aparecieron sus primeros 15 fallos: todos del mismo conjunto
(`house_prices_nominal`) y del mismo motor —el nuestro—, muriendo en **0,9 s de
un tope de 120**. No era el reloj. La causa: la lista de marcadores de dato
ausente del núcleo incluye `"none"`, y ese conjunto **declara `None` en su
cabecera ARFF como nivel válido** —«sin revestimiento de mampostería»—: **864 de
1.460 filas** se leían como ausentes. La guarda que hizo saltar todo **no era el
defecto: detectó una ambigüedad que existía de verdad.**

Se reparó en la raíz (un nivel declarado es una categoría, no un ausente;
`b2fad46` y `83b7f70`) y se relanzó con `--forzar`, porque el fichero tocado
entra en la huella del caché. **Los 1.904 intentos se re-ejecutaron**: es el
coste de que el código cambiara, y fue una decisión, no un descuido.

**El artefacto parado se conserva** como registro de ese defecto, fuera del
árbol, en el scratchpad de la sesión: `c5_parcial_antes_de_parar.json`.

---

## Antes de relanzar cualquiera de ellas

Está en `CLAUDE.md`, sección «ANTES DE RELANZAR UNA PASADA», y son tres cosas
de las que **dos son trampas**: un intento fallido se reutiliza del caché, el
digest del caché cubre 20 de los 112 módulos del camino, y tocar el guion
invalida la caché entera.
