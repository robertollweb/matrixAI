# Contrato 163 (C2) — conjuntos NUEVOS de CONFIRMACIÓN de la rejilla, apartados el 13-11-2026 ANTES de medir nada

**No se ha medido ni se mide nada sobre ellos** (ningún modelo). Solo se han leído sus metadatos de OpenML (nombre, licencia, filas, columnas,
clases, valores faltantes) y descargado sus ficheros para fijar su sha256. Se aplican UNA vez, por motor, tras el veredicto de los 23 y solo si
ese motor CUMPLE (`REGLA_C2.md` §5).

**No son los de `benchmarks/c160/CONFIRMACION.md`, `benchmarks/c162/CONFIRMACION.md` ni `benchmarks/c163/CONFIRMACION.md`** (los 18: breast-w,
credit-approval, cmc, vehicle, abalone, airfoil_self_noise; credit-g, spambase, eucalyptus, vowel, auction_verification,
concrete_compressive_strength; tic-tac-toe, cylinder-bands, analcatdata_dmft, MiceProtein, solar_flare, student_performance_por).

## La regla de selección (la del 160, escrita antes de mirar nada; `seleccion_confirmacion_c2.py`)

Candidatos de **OpenML-CC18** (estudio 99) y **OpenML-CTR23** (estudio 353) en ORDEN ASCENDENTE de `data_id`; se queda el primero que cumple todo,
hasta 2 binarios, 2 multiclase (3–20 clases) y 2 de regresión: (1) no es de los 40 de la Fase 0 (por `data_id` ni por nombre; ni del catálogo del
protocolo registrado) ni de los 18 de arriba; (2) licencia permisiva (`Public`, `CC BY 4.0`, `CC BY`, `CC0`); (3) **entre 500 y 20.000 filas**
(pequeño-mediano, el cubo de los 23; la palanca de este corte no depende de que haya categóricas, así que no se exige ninguna, a diferencia del
C1); (4) como mucho 100 predictores; (5) tarea y clases según el estudio; (6) no es hermano de origen de uno de la Fase 0 (`mfeat-`, `satimage`).
El script mira 46 candidatos; su log completo está en `~/.agentes/163/c2/seleccion_log.json`.

## Los 6

| tarea | OpenML id | nombre (versión) | licencia (campo OpenML) | filas × cols | clases | faltantes | md5 OpenML = descargado | sha256 del fichero descargado |
|---|---|---|---|---|---|---|---|---|
| multiclase | 458 | analcatdata_authorship (1) | Public | 841 × 71 | 4 | 0 | sí | 8674a81705440cfcb458e481300185293c1fe75576c2d86d426b0be8148d4506 |
| binaria | 1067 | kc1 (1) | Public | 2.109 × 22 | 2 | 0 | sí | fe0ea9ab3977bf3d2411dc72eefed2462ef4bf750a77668bdcf2acbb351c521d |
| binaria | 1464 | blood-transfusion-service-center (1) | Public | 748 × 5 | 2 | 0 | sí | ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a |
| multiclase | 1475 | first-order-theorem-proving (1) | Public | 6.118 × 52 | 6 | 0 | sí | 8ada14230e4df0a3eb12aafd9af2bff99ee83bbb539c97d4fef169d981ca8c51 |
| regresión | 44960 | energy_efficiency (9) | CC BY 4.0 | 768 × 9 | — | 0 | sí | 7d52e49858ff8785d6d4cfe36a4202ffcb0c83df084f1582e666e17a6fa69b24 |
| regresión | 44962 | forest_fires (9) | CC BY 4.0 | 517 × 13 | — | 0 | sí | 25672c7dd97b7af4c00b5d5812c17f6fde8a9e422835e3610c730117c8d6e2b5 |

(Hay 6 y la regla no pide más: sobran para «la mitad hacia arriba» de los no saturados.)

## Cautelas (dichas antes de medir)

- **Tamaño:** cinco de los seis tienen < 2.200 filas y tres < 850 (`blood-transfusion`, `forest_fires`, `analcatdata_authorship`, `energy_efficiency`
  están entre 500 y 850): con la prueba del 20 % son ~100–170 filas y el IC de 5 pliegues será muy ancho; además la `validation` que elige entre 8
  configuraciones tiene muy pocas filas (el riesgo que el contrato 163 §6 ya nombra: elegir entre 8 con la validación de un pliegue pequeño
  sobreajusta). Es lo que la regla del 160 produjo; si el supervisor prefiere un mínimo de 1.000 filas, se cambia ahora, antes de medir (con 1.000+
  y el orden ascendente salen otros).
- **Solo UN conjunto es mediano** (`first-order-theorem-proving`): la confirmación apenas ve el perfil mediano (31 hojas), que es donde la política
  rige hoy en producción.
- **Saturados:** no se declara ninguno aquí (no se puede sin medir; `analcatdata_authorship` podría estar cerca). Se declararán con el criterio del
  160 (mejor referencia ≥ 98 %) en la enmienda sellada de confirmación, antes de medirlos.
- **Registrarlos en el catálogo de la Fase 0** antes de correrlos es trabajo del corte de confirmación (igual que en el C1). `medir_c163_c2.py` no
  trae modo de confirmación todavía (solo mide los 23): se construye con la enmienda, si algún motor cumple.
- **Licencia:** la que declara OpenML; el origen no se ha verificado. No se redistribuyen los ficheros, solo se citan id y sha256.
