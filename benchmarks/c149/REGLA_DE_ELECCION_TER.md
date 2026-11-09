# 149-C0-ter — regla de elección, fijada ANTES de medir (la última)

Escrita por el supervisor el 03-11, antes de que exista `medir_c0ter.py` y antes de ninguna cifra con las semillas de abajo. La
medición escribe en su resultado el sha256 de ESTE fichero. Lo que no está escrito aquí se declara «no concluyente».

**Por qué hay una tercera.** La regla del C0-bis (`REGLA_DE_ELECCION_BIS.md`) tenía dos errores de lógica del supervisor, no de ajuste
fino. Los vio el agente al leer sus casos (C0-bis, commit `3fd16d3`):
1. Usaba el efecto CON SIGNO en la dirección declarada. Una columna monótona declarada al revés (el contraejemplo más claro) tiene un
   efecto negativo ≤ ε y salía «no se pudo comprobar».
2. Usaba el efecto de EXTREMO A EXTREMO. Una U o una senoide se mueven mucho y acaban casi donde empezaron, y también salían «no se
   pudo comprobar».
Con eso la potencia no podía llegar al 80 % con ningún (ε, q). Esta regla corrige SOLO esos dos puntos. **Es la última ronda**: si no
elige, no hay más C0 y se decide con las tres mediciones a la vista.

## Lo que se mide

Todo igual que en el C0-bis (mismos datos sintéticos, motores, tareas, preparación, escala, G = 25, m = 200, mismas planas aparte,
mismos casos de verdad), con **semillas 6, 7 y 8**.

## El veredicto V'(ε, q)

Por (columna, dirección), con los mismos retrocesos por fila y las mismas curvas por fila de C1:
1. **Amplitud** de la columna = la MEDIANA, sobre las m filas, de (máximo − mínimo) de la curva de esa fila, dividida entre la
   escala. Sin signo y sin depender de los extremos. Si la amplitud es ≤ ε → `no_se_pudo_comprobar`.
2. Si la **fracción de filas** con retroceso > ε es **≥ q** → `con_contraejemplos`.
3. Si no → `sin_contraejemplos`.

ε ∈ {2; 5; 10; 15; 20} % · q ∈ {5; 10; 20} %.

## La regla

La misma del C0-bis, punto por punto: por tarea, la celda cumple si en sus dos grupos FA ≤ 10 % y POT ≥ 80 %; el menor ε y, con
él, el menor q; si no hay ninguna, esa tarea no tiene veredicto por omisión; las planas tienen que salir `no_se_pudo_comprobar` en
≥ 90 % de sus casos o se declara; el sabotaje de etiqueta invertida no puede elegir el mismo (ε, q) en la misma tarea o se anula.

Además, como DATO (no decide): la potencia separada en «dirección contraria de `x1`–`x4`» y «no monótonas de `x5`, `x6`, `x8`, `x9`».

## Fuera

Igual que el C0-bis. Esto no decide el texto de la pantalla: lo decide Roberto.
