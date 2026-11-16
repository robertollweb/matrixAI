# REGLA de lectura del kit 116-C5 (acierto de TabICL en GPU)

Escrita el 2026-11-16 (10:45, `date`) **ANTES de que exista ningún dato de esta medición**. El analizador
(`analiza_kit_acierto.py`) cita este fichero por su sha256 y **se niega a correr si el fichero ha cambiado**:
retocar la regla después de ver números exige cambiar también el analizador, y eso queda en git.

## Qué se mide y con qué se compara

- **Candidato**: `tabicl.v2` con GPU (kit `benchmarks/tabicl_116c5_acierto_gpu/`), con EXACTAMENTE las particiones,
  semillas, repeticiones, presupuesto por cubo y métrica de cierre del protocolo `113-C0.v2`
  (digest `1b4e902e…`), la misma pasada que 116-C2 hizo en CPU.
- **Rivales, ya medidos en CPU con esas mismas particiones y sin remedir**:
  1. `lightgbm` de `benchmarks/fase0/pasada_v2_113_resultado.json` (113-C4).
  2. La red TabM, motor `matrixai.dense.tabm_cpu`, de `benchmarks/fase0/resultado_pasada_119_c3.json` (119-C3).
- **Métrica**: la de cierre de cada conjunto según el protocolo (`auroc`, `accuracy` o `r2`), en su escala cruda
  0–1; en las tres, más es mejor. Una métrica fuera de esas tres PARA el análisis.

## Cómo se compara (por conjunto y por rival)

1. **Emparejado por (repetición, pliegue)**: `d = métrica(TabICL GPU) − métrica(rival)` en los pliegues que
   ambos completaron. Se calculan la media de cada uno y la media de `d`.
2. **Intervalo**: bootstrap de percentiles de PLIEGUES al 95 %, 1.000 remuestras, semilla 0 — la función
   `protocolo._intervalo_pareado` de la 116-C2, sin reimplementar.
3. **Margen de equivalencia**: **±0,005** (0,5 puntos de la métrica cruda), el mismo de 114-C6 y 116-C2.
4. **Veredicto del conjunto frente a ese rival**, con `[bajo, alto]` el intervalo de `d`:
   - **GANA**: `bajo > 0` y el intervalo NO cabe entero en ±0,005. (Es el «mejora» de 116-C2.)
   - **PIERDE**: `alto < 0` y el intervalo NO cabe entero en ±0,005. (Su espejo.)
   - **EMPATE**: todo lo demás, con dos subclases que se cuentan juntas pero se ven por separado:
     *equivalente* (el intervalo cabe entero en ±0,005, aunque excluya el cero: no hay diferencia práctica) y
     *no concluyente* (el resto: el intervalo cruza el cero y es más ancho que el margen, no hay datos para decidir).
5. **NO COMPARABLE** (no entra en gana/pierde/empate y se lista aparte, con su motivo, nunca se maquilla):
   - TabICL no completó TODOS los pliegues del conjunto (un fallo, un tope o un pliegue no lanzado);
   - al rival le falta algún pliegue que TabICL sí completó;
   - menos de 2 pliegues comunes;
   - el digest de los pliegues del kit no coincide con el que registró el rival (ver abajo).
   Un fallo de TabICL NO se cuenta como «pierde»: se dice «no comparable por fallo» y el conjunto se lee a mano.

## Salvaguardas

- **Misma partición, demostrada**: el kit guarda `pliegues_digest_por_dataset` (el de `propuesta.pliegues.digest()`);
  el analizador lo compara con el de 119-C3 (`pliegues_digest_por_dataset`) y con el `plan_digest` de la v2
  (`particion_por_dataset`). Si no coincide, el conjunto es NO COMPARABLE.
- **Mismo protocolo**: la versión y el digest del protocolo del JSON del kit deben ser `113-C0.v2` / `1b4e902e…`.
- **Los sellados no entran** (no se miden ni se analizan; la confirmatoria los reserva).

## Qué NO se concluye

- Esta tabla describe **acierto en estos conjuntos y en esta tarjeta**; no decide por sí sola el peso de TabICL en el
  estudio ni su techo (eso se decide con el tiempo y la memoria, ya medidos en 116-C4).
- No hay un «veredicto global» ni umbral del tipo «≥ 60 %»: se cuentan gana / pierde / empate por rival y se
  ENSEÑAN los números por conjunto. Cualquier umbral se pondría después, por escrito, por Roberto.
- Diferencia GPU frente a CPU de TabICL (control): donde 116-C2 midió TabICL en CPU el analizador enseña la media
  de `GPU − CPU` como control informativo (la GPU no es bit a bit igual a la CPU); no tiene veredicto.
- Con 5 pliegues (conjuntos medianos y grandes) el intervalo por pliegues es muy grueso: un «empate no concluyente»
  ahí significa «no se puede decir», no «iguales».
