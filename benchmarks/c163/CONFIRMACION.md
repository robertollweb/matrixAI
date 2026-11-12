# Contrato 163 (C1) — conjuntos NUEVOS de CONFIRMACIÓN, apartados el 12-11-2026 ANTES de medir nada

**No se ha medido ni se mide nada sobre ellos** (ningún modelo). Solo se han leído sus metadatos de OpenML (nombre, licencia, filas, columnas,
clases, nº de columnas nominales) y descargado sus ficheros para fijar su sha256. Se aplican UNA vez, tras el veredicto de los 23, y solo si
los 23 CUMPLEN (`REGLA_C1.md` §5). Para correrlos hará falta registrarlos antes en el catálogo de la Fase 0: trabajo del corte de confirmación.

**No son los de `benchmarks/c160/CONFIRMACION.md` ni los de `benchmarks/c162/CONFIRMACION.md`** (breast-w, credit-approval, cmc, vehicle, abalone,
airfoil_self_noise; credit-g, spambase, eucalyptus, vowel, auction_verification, concrete_compressive_strength).

## La regla de selección (la del 160/162 con DOS cambios que exige esta palanca, escritos antes de mirar nada)

La palanca solo actúa con columnas categóricas y menos de 2.000 filas de train, y la confirmación de los 6 del 160/162 casi no las tiene. Misma
base: candidatos de **OpenML-CC18** (estudio 99) y **OpenML-CTR23** (estudio 353) en ORDEN ASCENDENTE de `data_id`; se queda el primero que
cumple todo, hasta 2 binarios, 2 multiclase (3–20 clases) y 2 de regresión:
1. no es de los 40 de la Fase 0 (por `data_id` ni por nombre; ni del catálogo del protocolo registrado) ni de los 12 de las confirmaciones del
   160 y del 162;
2. licencia permisiva (`Public`, `CC BY 4.0`, `CC BY`, `CC0`/`CC 0: Public Domain`);
3. **entre 500 y 2.500 filas** (cambio 1: con el 80 % de train quedan < 2.000 y la fórmula da < 100; el 160/162 admitían hasta 20.000);
4. como mucho 100 predictores; 5. tarea y clases según el estudio;
6. no es hermano de origen de uno de la Fase 0 (prefijo `mfeat-` y `satimage`, como el 162);
7. **al menos 3 predictores nominales** (`NumberOfSymbolicFeatures`, menos 1 el objetivo en clasificación) (cambio 2).

El script (`seleccion_confirmacion.py`, derivado del del 162) mira 75 candidatos; su log completo está en `.agentes/163/c1/sel/seleccion_log.json`.

## Los 6

| tarea | OpenML id | nombre (versión) | licencia (campo OpenML) | filas × cols | clases | faltantes | md5 (OpenML = descargado) | sha256 del fichero descargado |
|---|---|---|---|---|---|---|---|---|
| binaria | 50 | tic-tac-toe (1) | Public | 958 × 10 | 2 | 0 | 34b2992c41e5e23c42769817e96305ac | 09cb4ceb7d410dc7663a5ae8a4d9da6d82270468bdfa5d73a4161204cbb90133 |
| binaria | 6332 | cylinder-bands (2) | Public | 540 × 40 | 2 | 999 | 59ec3d17d1fb9fa79c3108a7a5a9bc5e | 6e792f09b2d59d0500a2ba9a5975b31b5eef9534b98f5bf87ab8047452e4c45e |
| multiclase | 469 | analcatdata_dmft (1) | Public | 797 × 5 | 6 | 0 | ad86226880052117a3e95479cf61faf6 | d1fce1e52ef3e29c3cb8af6fc8215dd2982f829c0bfcd18a8be39269db4df501 |
| multiclase | 40966 | MiceProtein (4) | Public | 1.080 × 82 | 8 | 1.396 | 3c479a6885bfa0438971388283a1ce32 | f48ced98ecfe8ff056d8ea3507e067258c82f29199a1af2261b516010827cc4d |
| regresión | 44966 | solar_flare (7) | CC BY 4.0 | 1.066 × 11 | — | 0 | f8b0a7c961d3329e1bcea52c724bd724 | 34f377d0c949b0a35e7b6ce5fd9854f4154842dd03e46d24cfcebe2d5c267794 |
| regresión | 44967 | student_performance_por (8) | CC BY 4.0 | 649 × 31 | — | 0 | cee70f2a16e9c52694bf0bb77603b4e8 | ab4db935dca19424413e68129cab61609d6340d83194f185bb4e74e2268e4bd6 |

Descarga (OpenML, sin autenticación; el md5 coincide con el declarado en los seis):

```
curl -sSL -o tic-tac-toe.arff https://openml.org/data/v1/download/50/tic-tac-toe.arff
curl -sSL -o cylinder-bands.arff https://openml.org/data/v1/download/1854224/cylinder-bands.arff
curl -sSL -o analcatdata_dmft.arff https://openml.org/data/v1/download/52581/analcatdata_dmft.arff
curl -sSL -o MiceProtein.arff https://openml.org/data/v1/download/17928620/MiceProtein.arff
curl -sSL -o solar_flare.arff https://openml.org/data/v1/download/22111830/solar_flare.arff
curl -sSL -o student_performance_por.arff https://openml.org/data/v1/download/22111831/student_performance_por.arff
```

## Cautelas (dichas antes de medir)

- **Cuántos son AFECTADOS no se sabe sin prepararlos**: «nominal» en OpenML no es «categórica tras la preparación de la Fase 0» (que además puede
  tratar como numéricas o excluir columnas) ni «con más de 4 valores» (LightGBM parte una categórica con ≤ 4 valores en one-hot, donde
  `min_data_per_group` no cuenta). `tic-tac-toe` (9 nominales de 3 valores) y `MiceProtein` (3 nominales de 2 valores) son los candidatos a no
  cambiar. Por eso la regla de confirmación (`REGLA_C1.md` §5) cuenta como afectados los que la PROPIA medida declara con la palanca actuando y
  pide al menos 2: si no hay, SIN VEREDICTO, no CUMPLE.
- **Licencia:** `Public` / `CC BY 4.0` es lo que declara OpenML; el origen no se ha verificado. El supervisor lo comprueba antes de publicar nada.
  No se redistribuyen los ficheros, solo se citan id y sha256.
- **Tamaño:** todos tienen < 1.100 filas: con la prueba del 20 % el IC de 5 pliegues será muy ancho. Es lo que la regla produjo.
