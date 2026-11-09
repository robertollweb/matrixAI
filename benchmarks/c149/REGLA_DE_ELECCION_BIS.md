# 149-C0-bis — regla de elección, fijada ANTES de medir

Escrita por el supervisor el 03-11, antes de que exista el guion `medir_c0bis.py` y antes de cualquier cifra con las semillas de
abajo. El C0 (`REGLA_DE_ELECCION.md`, semillas 0–2) dio «no hay ε por omisión» con el veredicto de C1 (una sola fila basta). El
diagnóstico exploratorio de `documentacion/149_C0_MEDICION.md` §2 sugirió esta variante; por eso se mide con **semillas nuevas** y
no sobre las cifras que la sugirieron. La medición escribe en su resultado el sha256 de ESTE fichero.
Lo que no está escrito aquí no se decide con las cifras: se declara «no concluyente».

## Lo que se mide

Los mismos datos sintéticos de verdad conocida que en el C0 (10 columnas × 2.000 filas, `gen` sin cambios), los mismos motores
(`lightgbm`, `sklearn.hgb`), las mismas tareas (regresión, binaria), la misma preparación y la misma escala. **Semillas 3, 4 y 5**.
**G = 25, m = 200** fijos (el C0 dio n = 200 y G no cambió nada).

Los casos y su verdad, igual que en el C0, con un cambio: **las planas (`x7`, `x10`) no son casos monótonos**. La regla nueva las
manda a «no se pudo comprobar», que no es ni falsa alarma ni detección, y se cuentan aparte: cuántas veces sale cada veredicto.

## El veredicto alternativo V(ε, q)

Por (columna, dirección), con los mismos retrocesos por fila que calcula C1:
1. Si el **efecto total** de la columna es ≤ ε → `no_se_pudo_comprobar` (se mira PRIMERO).
2. Si la **fracción de filas** con retroceso > ε es **≥ q** → `con_contraejemplos`.
3. Si no → `sin_contraejemplos`.

ε ∈ {2; 5; 10; 15; 20} % de la escala · q ∈ {5; 10; 20} %.

## La regla

Se decide **por tarea** (regresión y binaria por separado; cada una tiene dos grupos, LightGBM y HGB):
1. Una celda (ε, q) **cumple** en una tarea si, en sus **dos grupos a la vez**, FA ≤ 10 % y POT ≥ 80 % (las tasas del C0, con las
   3 semillas juntas).
2. **(ε, q) por omisión de la tarea** = el menor ε que cumple con algún q; entre los q que cumplen con ese ε, el **menor** q.
3. Si ninguna celda cumple en una tarea: esa tarea **no tiene veredicto por omisión**.
4. **Las planas:** en una tarea con (ε, q) elegido, cada plana tiene que salir `no_se_pudo_comprobar` en al menos el 90 % de sus
   casos. Si no, se declara «las planas no se distinguen» junto a la elección, sin anularla.
5. **Sabotaje** (obligatorio): la misma regla con la etiqueta de verdad invertida no puede elegir el mismo (ε, q) en la misma
   tarea. Si lo elige, el instrumento no distingue y la elección de esa tarea se anula.

## Fuera

Redes, multiclase, datos reales, tiempos. Esto no decide el texto de la pantalla: lo decide Roberto con estas cifras.
