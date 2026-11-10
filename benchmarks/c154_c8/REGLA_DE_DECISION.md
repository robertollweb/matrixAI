# 154-C8 · criterio 8 por el ESTUDIO REAL · protocolo y regla de decisión (escrita el 10-11-2026 ANTES de ver ninguna cifra)

## Pregunta
La medida original (idea 10, §1 del contrato) usó un arnés propio (`medir.py`, 5x3 sobre `train`, códigos como `str`). La auditoría C4
(I2) midió que con los códigos como número —lo que hace el estudio— `H_lin` pierde su ventaja en C. ¿Qué dicen C, D y ropa cuando los
mide el ESTUDIO entero, y sigue en pie la frase fija del §6 («de −0,3 puntos de AUROC (empate) a −8,6 % de error frente a la mejor de
las dos partes por separado, y no ganó siempre»)?

## Qué se mide (guion: `medir_c8.py`)
Código: núcleo `main` (con los ceros a la izquierda), motores `f82e8d7`, Studio `c03a0d14` (rama c154), fijados por `~/encolar.sh`.
Datos: los mismos CSV de la medida (sha256 en `procedencia`): C = `placsp/tarea_c.csv` (30.462 filas), D = `boe/tarea_d.csv` (5.000),
ropa = `ropa/ropa.csv` (22.636). Se dan al estudio TODAS las filas (train+dev+test del arnés) con el texto declarado y las columnas del
arnés; el estudio reparte como siempre (5 pliegues x 1, 20 % de prueba reservada, semilla 0). NO son las particiones de §1: las cifras se
comparan en signo y tamaño, no al milésimo. Vino NO se mide (licencia; el §1 queda para él).
Por conjunto: (1) un estudio COMPLETO (todos los candidatos, el reloj y el reparto de búsqueda reales); (2) un estudio por candidato de
interés SOLO (+ el baseline de reserva, que el estudio exige): `H_lin`, `H_sel`, `T` (`sklearn.texto_tfidf`) y `X*` (el mejor de solo tabla
por la media de selección del completo; candidatos: lightgbm, hgb, torch_cpu, tabm_cpu). Cada uno pasa por el camino entero (reajuste con
todo el desarrollo, recalibración, umbral) y se mide en la prueba reservada. Declarado: en los solos cada motor tiene más reserva de reloj
que en el completo; las medias de selección salen iguales (comprobado en la prueba pequeña) porque los ajustes son deterministas.
Se registra, por candidato: media de selección y valor por pliegue, prueba reservada, tiempo por intento (frío, un hijo por intento), RAM,
y, del completo, quién gana, quién quedó fuera y por qué, el reparto de la búsqueda y los tipos que el estudio dio a cada columna.

## Definiciones (como el PROTOCOLO de la medida original)
`H` = el mejor de los dos híbridos por la media de selección; `B` = el mejor entre `T` y `X*` por la media de selección.
mejora = AUROC(H) − AUROC(B) (absoluta) o (RMSE(B) − RMSE(H)) / RMSE(B) (relativa). Umbral: +0,010 AUROC o 2 % RMSE.
IC95: t de Student (4 g.l.) sobre las 5 diferencias por pliegue H − B (mismos pliegues). La prueba reservada de H y de B es la de sus
estudios solos.
- GANA: mejora de selección ≥ umbral Y límite inferior del IC > 0 Y mejora en prueba reservada > 0.
- PIERDE: mejora ≤ −umbral Y límite superior del IC < 0.
- EMPATE: lo demás.

## Regla de decisión sobre la frase del §6
1. **Cuenta lo que mide el estudio**, no §1 (decisión del supervisor del 09-11). Se calcula GANA/EMPATE/PIERDE en C, D y ropa.
2. Si H GANA en **menos de 2 de los 3**, o PIERDE en alguno: la frase NO puede decir que mejora; se aparca `[?]` para Roberto (qué hacer con el
   candidato) y mientras tanto la frase pasa a «no mejoró de forma fiable frente a la mejor de las dos partes por separado en todas las pruebas».
3. Si GANA en ≥ 2 de 3 y no PIERDE en ninguno: se compara el rango de la frase con el medido:
   - **MANTENER la frase tal cual** solo si (a) el extremo bueno medido en C está a ≤ 1 punto porcentual de −8,6 % (es decir, mejora relativa
     de RMSE ≥ 7,6 %) Y (b) el extremo malo medido (el peor de D y ropa) está dentro de ±0,5 puntos de AUROC de −0,3.
   - En otro caso: **CAMBIAR** la frase por los extremos medidos por el estudio (peor y mejor de C, D, ropa, con su métrica), y decir en el
     expediente que «en las pruebas internas, medidas por el propio estudio». `vino` solo cuenta si se cita su §1 aparte; no se mezcla.
   - Cualquiera que sea el resultado, se publica la diferencia con §1 (las dos cifras), como manda el criterio 1.
4. El campeón del estudio completo se registra; no decide la frase (un híbrido puede perder en el completo frente a otro motor y aun así
   mejorar a «la mejor de las dos partes»); pero si en algún conjunto el campeón es un candidato que no es ni híbrido, ni T, ni X*, se anota.
5. Si algún X* no puntúa (p. ej. los árboles caen con una categoría no vista, visto en la prueba pequeña de C), `B` = `T` y se dice; y se
   abre hallazgo de producto (los árboles de solo tabla deben puntuar en C).

## Peso de búsqueda (declarado 1,0 por híbrido)
Referencia: la mediana del tiempo por intento de `lightgbm`/`sklearn.hgb` (peso 1,0) en el estudio completo del MISMO conjunto.
- MANTENER 1,0 si, en C, D y ropa, la mediana por intento de cada híbrido es ≤ 2 x la de la referencia Y ningún intento de un híbrido
  cae por plazo (`fuera_por_plazo`, estado de intento ≠ completed por reloj).
- Si no: el peso propuesto es la razón redondeada (mediana híbrido / mediana referencia, el peor de los tres conjuntos), a corregir en
  `_PESO_DE_BUSQUEDA` en el C8.

## Estimación y techo (CLAUDE.md: por escrito antes de encolar)
Ver contrato §12.8. Resultados: `~/.agentes/154/c8/resultados-<fecha>/c8_{D,ropa,C}.json` y `analiza_c8.py` los resume contra esta regla.
