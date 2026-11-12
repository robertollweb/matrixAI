# Contrato 162 (C1) — conjuntos NUEVOS de CONFIRMACIÓN (§4.3 del contrato), apartados el 12-11-2026 ANTES de medir nada

**No se ha medido ni se mide nada sobre ellos** (ningún modelo, de ningún tipo). Solo se han leído sus metadatos de OpenML (nombre, licencia,
filas, columnas, clases) y descargado sus ficheros para fijar su sha256. Se aplican UNA vez, tras el veredicto de los 23, y solo si los 23
CUMPLEN (`REGLA_C1.md` §5). Para correrlos hará falta registrarlos antes en el catálogo de la Fase 0: trabajo del corte de confirmación.

**No son los 6 de `benchmarks/c160/CONFIRMACION.md`** (breast-w, credit-approval, cmc, vehicle, abalone, airfoil_self_noise): esos son del 160
(contrato 162 §4.3). Cada palanca del 162 (C1, C2, C3) se confirma en conjuntos propios; C2 y C3 apartarán los suyos con la misma regla.

## La regla de selección (la del 160, sin cambios, tomando los SIGUIENTES)

Es la de `benchmarks/c160/CONFIRMACION.md`: candidatos de **OpenML-CC18** (estudio 99) y **OpenML-CTR23** (estudio 353) en ORDEN ASCENDENTE de
`data_id`; se queda el primero que cumple todo, hasta 2 binarios, 2 multiclase (3–20 clases) y 2 de regresión, descontando los 6 del 160:
1. no es de los 40 de la Fase 0 (por `data_id` ni por nombre; los 32 de `resultado_pasada_119_c3.json` + los 8 sellados del protocolo; y, por
   prudencia, tampoco ninguno del catálogo del protocolo registrado) ni es uno de los 6 del 160;
2. licencia permisiva (`Public`, `CC BY 4.0`, `CC BY`, `CC0`/`CC 0: Public Domain`);
3. entre 500 y 20.000 filas; 4. como mucho 100 predictores; 5. tarea y clases según el estudio;
6. no es hermano de origen de uno de la Fase 0: se descartan los de prefijo `mfeat-` (clausula del 160) **y `satimage` (182)**, que es el Statlog
   Landsat Satellite de UCI del que sale `Satellite` (40900, SELLADO, variante de detección de anomalías): clausula aplicada con solo metadatos
   a la vista, escrita aquí antes de medir.

**Comprobación de que es LA MISMA regla**: con el cupo del 160 vacío y la definición de «Fase 0» del 160 (32 + 8 sellados), el mismo script
reproduce EXACTAMENTE los 6 del 160 (breast-w, cmc, credit-approval, vehicle, abalone, airfoil). El script y su log completo (30 candidatos
mirados) están en `.agentes/ola5-39/c1/sel/` (`seleccion_log.json`) y una copia del script en `benchmarks/c162/seleccion_confirmacion.py` (con `SOLO_DEFINICION_DEL_160=1` y el cupo del 160 vacío es la comprobación).

## Los 6

| tarea | OpenML id | nombre (versión) | licencia (campo OpenML) | filas × cols | clases | faltantes | md5 (OpenML = descargado) | sha256 del fichero descargado |
|---|---|---|---|---|---|---|---|---|
| binaria | 31 | credit-g (1) | Public | 1.000 × 21 | 2 | 0 | 9a475053fed0c26ee95cd4525e50074c | 768be84ef749110d5fcac109a6541f2df190e8c6987e1dc432c0a4f29d0642f5 |
| binaria | 44 | spambase (1) | Public | 4.601 × 58 | 2 | 0 | d9ace01aeac3461e326a8e1b2d53fd84 | f0f6ccd8e8c6b1feeb09c0565ac691d7906cca5373959981512b9934ab4112a6 |
| multiclase | 188 | eucalyptus (1) | Public | 736 × 20 | 5 | 448 | adaf2938bd76922c010d3d32592bdb80 | 59a2fe2cb13540ea64a01faba34048209e096bb75a105b9c053950df4f6d678c |
| multiclase | 307 | vowel (2) | Public | 990 × 13 | 11 | 0 | 9c82432df30166fc62353fe4f8bb237a | a0526fad8f69a04a9b8844b756046ab0dca5a97954e14b5af08d0c85e74e7e72 |
| regresión | 44958 | auction_verification (7) | CC BY 4.0 | 2.043 × 8 | — | 0 | fee9a1bd5d59943c9a83508a401821c8 | f0dcaf7165b3235a6031395f2068bb7833bfbeb32c263713cf8834bd679a4d97 |
| regresión | 44959 | concrete_compressive_strength (7) | CC BY 4.0 | 1.030 × 9 | — | 0 | 1906eae71bd8b8142d079a4b966549ac | 510beb3f50cba7907815b611af40edf14eb75902995bb9d7bccb40db156cf8ec |

Descarga (OpenML, sin autenticación; el md5 coincide con el que declara OpenML en los seis):

```
curl -sSL -o credit-g.arff https://openml.org/data/v1/download/31/credit-g.arff
curl -sSL -o spambase.arff https://openml.org/data/v1/download/44/spambase.arff
curl -sSL -o eucalyptus.arff https://openml.org/data/v1/download/3625/eucalyptus.arff
curl -sSL -o vowel.arff https://openml.org/data/v1/download/52210/vowel.arff
curl -sSL -o auction_verification.arff https://openml.org/data/v1/download/22111822/auction_verification.arff
curl -sSL -o concrete_compressive_strength.arff https://openml.org/data/v1/download/22111823/concrete_compressive_strength.arff
```

## Cautelas (dichas antes de medir)

- **Licencia:** `Public` / `CC BY 4.0` es lo que declara OpenML; el origen (UCI, StatLib/R) NO se ha verificado en esta pasada. El supervisor lo
  comprueba antes de publicar nada que cite estos datos. No se redistribuyen los ficheros, solo se citan id y sha256.
- **Tamaño:** credit-g (1.000), vowel (990), eucalyptus (736) y concrete (1.030) son pequeños: con una prueba del 20 % el IC de 5 pliegues será muy
  ancho. Es lo que la regla produjo.
- **Saturación:** `spambase` y `vowel` podrían ser conjuntos casi saturados; no se puede declarar aquí sin medir. Se declarará, con el mismo
  criterio (referencia ≥ 98 %), en la enmienda de confirmación que se selle antes de medirlos.
- **eucalyptus** trae 448 faltantes y categóricas: es un buen ejercicio de la preparación, no un riesgo para la palanca.
- El hermano de origen de `satimage` es un juicio sobre metadatos (misma fuente UCI): si el supervisor prefiere admitirlo, entraría `satimage`
  como multiclase en lugar de `vowel`; se cambia ahora, antes de medir.
