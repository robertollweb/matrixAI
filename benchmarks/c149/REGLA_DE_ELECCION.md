# 149-C0 — regla de elección, fijada ANTES de medir

Escrita el **2026-11-02 19:09** (`date`, hora local del servidor), antes de que exista ninguna cifra de
`medir_c0.py`: el guion todavía no se ha ejecutado ni una vez con datos cuando se commitea este fichero. La
medición escribe en su resultado el sha256 de ESTE fichero; si alguien lo edita después, el sha no cuadra.
Lo que no está escrito aquí no se decide con las cifras: se declara «no concluyente».

## Qué se mide

Datos sintéticos de **verdad conocida**: 2.000 filas × 10 columnas `x1…x10` uniformes en [0, 10], cuatro
tareas-motor (regresión y binaria × LightGBM y HGB), **3 semillas**. La salida verdadera es la suma de una
función por columna (`gen`), más ruido en regresión / un logit en binaria:

| columna | función verdadera | monótona | efecto |
|---|---|---|---|
| x1 | lineal creciente | sí, «sube» | sí |
| x2 | saturación creciente (`1 − e^{−x/2}`) | sí, «sube» | sí |
| x3 | lineal decreciente | sí, «baja» | sí |
| x4 | saturación decreciente | sí, «baja» | sí |
| x5 | U (`((x−5)/5)²`) | no | sí |
| x6 | senoidal (1,5 periodos) | no | sí |
| x7 | plana (no entra en la salida) | débilmente (ambas) | no |
| x8, x9 | interacción `((x8−5)/5)·((x9−5)/5)` | no (depende de la otra) | sí |
| x10 | plana (ruido, no entra) | débilmente (ambas) | no |

El modelo es el **motor del producto** (`lightgbm`, `sklearn.hgb`), entrenado con la preparación del estudio
(`_particiones_preparadas`); la escala de la salida es la de C1: binaria = 1,0 (probabilidad del modelo, sin
recalibrar: el estudio la recalibra con `b > 0`, que no cambia el orden), regresión = p90−p10 del objetivo
de ENTRENAMIENTO.

## Los casos (columna, dirección declarada) y su verdad

Para cada columna y cada dirección declarada se calcula el **retroceso VERDADERO**: el de la función
verdadera (sin ruido ni modelo) sobre la misma rejilla y las mismas filas base, y se divide entre la escala.

* **Caso monótono** (para la falsa alarma): retroceso verdadero **= 0** en esa dirección — x1/x2 «sube»,
  x3/x4 «baja», y las dos planas (x7, x10) en las dos direcciones. 8 casos × 3 semillas por (motor, tarea).
* **Caso no monótono con retroceso** (para la potencia): retroceso verdadero **≥ 10 % de la escala** —
  x5, x6, x8, x9 en las dos direcciones, y la dirección CONTRARIA de x1…x4. 12 casos × 3 semillas por (motor, tarea).
  Si en alguna tarea un caso no llega al 10 % de la escala, no cuenta como caso (se lista cuáles).
* Todo lo demás (retroceso verdadero entre 0 y el 10 %) no entra en ninguna de las dos tasas.

«Marcado» = el veredicto de C1 (`comportamiento.monotonia_desde_curvas`) es **`con_contraejemplos`**.
`no_se_pudo_comprobar` (p. ej. una plana: la prueba no distingue) **no es una falsa alarma ni una detección**:
se cuenta aparte.

* **Falsa alarma** FA(ε, G, m) = marcados / casos monótonos.
* **Potencia** POT(ε, G, m) = marcados / casos no monótonos con retroceso.

Se calculan por separado en cada grupo **(motor × tarea)** = 4 grupos, juntando las 3 semillas.

## La rejilla (recortada) y la regla

ε ∈ {0; 0,1; 0,5; 1; 2; 5} % de la escala · G ∈ {15; 25; 50} puntos · m ∈ {100; 200; 500} filas base
(m = 100 y 200 son submuestras con semilla de las 500 de cada columna).

1. Una celda (ε, G, m) **cumple** si, en **los cuatro grupos a la vez**, **FA ≤ 10 %** y **POT ≥ 80 %**.
2. Para cada (G, m), **ε\*(G, m)** = el **menor** ε de la rejilla que cumple. Si ninguno cumple: *ninguno*.
3. Se ordenan las (G, m) por coste `G × m` creciente (empate: menor G). **(G, m) por omisión** = la primera
   cuyo ε\* existe **y** tal que el MISMO ε cumple también en toda celda (G′, m′) con G′ ≥ G y m′ ≥ m
   (estabilidad: más puntos o más filas no lo estropean). **ε por omisión = ese ε\*.**
4. Si ninguna (G, m) tiene ε\* estable, **no hay ε por omisión**: la pantalla obliga a elegir y lo dice
   (decisión del contrato 149). Se publican igualmente las tablas FA/POT por celda.

## Sabotaje del instrumento (etiqueta invertida)

Se repite el **análisis** (no el entrenamiento: la etiqueta solo entra en el análisis) con los papeles de las
etiquetas intercambiados — los casos monótonos pasan a tratarse como «no monótonos» y al revés — y se aplica
la misma regla. Si la regla vuelve a elegir **el mismo ε** (o ambos «ninguno»), el instrumento **no distingue** y
la medición se declara **no concluyente** en esa dimensión; se escribe tal cual.

## n de la sensibilidad (B)

Modelos binarios (LightGBM y HGB, 3 semillas), fracciones δ ∈ {1 %, 5 %} del rango, signos ±. Para
n ∈ {100, 200, 500, 1.000, 2.000} filas de desarrollo (submuestra con semilla) se mide, por columna, la fracción de
filas cuya DECISIÓN (probabilidad ≥ 0,5) cambia. **n por omisión** = el menor n tal que el **percentil 95** de
|tasa(n) − tasa(2.000)| sobre todas las unidades (motor × semilla × columna × fracción × signo) es **≤ 0,02**
(2 puntos absolutos). Si ninguno lo cumple, se queda 2.000 y se dice.

## G por omisión de C

Es el G de la regla de arriba (paso 3), no un número aparte.

## Hipótesis «una red da curvas más suaves»

UNA red (`matrixai.dense.tabm_cpu`, épocas fijas = 30), regresión, semilla 0, mismos datos. Para los casos
monótonos con efecto (x1…x4 en su dirección) se mide el **retroceso mediano / escala** (G = 25, m = 200) y se
compara con LightGBM, misma semilla y tarea. **La hipótesis se acepta** si el de TabM es **< 50 %** del de
LightGBM; **se rechaza** si es ≥ 100 %; entre medias, *no concluyente*. Con una sola red y una sola semilla no
se generaliza a «las redes»: se escribe como esa red en ese conjunto.

## Tiempos y paridad (estudios reales rápidos)

* **Paridad.** En un estudio rápido de cáncer de mama (LightGBM) y otro de `us_crime`, para 3 columnas numéricas y
  50 filas base × G = 15: las salidas de **«perturbar la fila CRUDA y preparar»** y de **«preparar una vez y
  perturbar la fila ya preparada»** (valor en la columna y su indicador `__faltante` a 0) deben ser
  **idénticas (tolerancia 0)**. Una sola diferencia → «preparar una vez» **no se admite** en C2 y el tiempo que
  cuente es el del camino crudo.
* **Tiempo.** ms por fila perturbada de cada camino en `us_crime` (122 columnas, LightGBM) con 5 columnas × 200 filas
  × 25 puntos; se declara también el ahorro (crudo / preparado-una-vez).
* **Escala de regresión.** El sobre del estudio (`us_crime`) trae `objetivo_del_entrenamiento` con `p10` y `p90` finitos
  y `p90 > p10` (la escala sale de ahí, no de otro cálculo). Se publica al lado el p90−p10 de TODAS las filas de
  desarrollo, solo para ver la diferencia (el sobre no dice qué ids entrenaron, así que «igual a mano» no se puede
  comprobar desde fuera; ver la enmienda).
* **122 columnas, solo para tiempos:** ningún veredicto sale de `us_crime`.

## Lo que esta medición NO contesta

Datos reales con verdad desconocida; XGBoost y CatBoost; multiclase; la densa propia; estudios de 51.000×100.

## Enmienda (la hora está en el commit que la añade)

Hecha DESPUÉS de una prueba de humo del guion a tamaño mínimo (500 filas, 1 semilla, solo LightGBM, 2 épocas de red;
sin cifras que se usen aquí) y ANTES de cualquier medición de tamaño completo. Cambia dos cosas y ningún umbral:

1. La cláusula de la escala de regresión (arriba): el sobre no guarda los ids que entrenaron, y la comprobación «igual
   a mano» que se había escrito no se puede hacer; queda la que sí se puede (valores finitos, `p90 > p10`).
2. El guion añade, solo para el par (G, m) central, la **fracción de filas base que violan** a cada ε. Es **exploratoria**:
   NO entra en la regla ni en ninguna de las decisiones de arriba. Existe porque el veredicto de C1 es «con
   contraejemplos» en cuanto UNA fila supera la tolerancia, y con m = 100–500 filas eso puede dar «ningún ε cumple»;
   si ocurre, esa fracción es lo que permitiría al supervisor plantear otra regla en un corte posterior, declarada como
   post hoc.
