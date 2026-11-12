# Contrato 160 — conjuntos NUEVOS de CONFIRMACIÓN (D3), apartados el 12-11-2026 ANTES de medir nada

**No se ha medido ni se mide nada sobre ellos** (ni LightGBM, ni TabM, ni la mezcla). Solo se han leído sus metadatos de OpenML
(nombre, licencia, filas, columnas, clases) y descargado sus ficheros para fijar su sha256. El análisis de confirmación se aplica UNA vez,
tras el veredicto de los 23 (`REGLA_DE_DECISION.md` §4), y solo si los 23 CUMPLEN. Para correrlos hará falta registrarlos antes en el
catálogo de la Fase 0 (positiva/negativa, cubo…): trabajo del corte de confirmación, no de este.

## La regla de selección (escrita antes de mirar ningún resultado de modelo)

Candidatos: los conjuntos de dos suites públicas curadas de OpenML, en ORDEN ASCENDENTE de `data_id`: **OpenML-CC18** (estudio 99, 72
conjuntos de clasificación) y **OpenML-CTR23** (estudio 353, 35 de regresión). Se recorren en ese orden y se queda el primero que cumple TODO,
hasta 2 binarios, 2 multiclase (3–20 clases) y 2 de regresión (6 en total, las tres tareas):

1. NO está entre los 40 de la Fase 0 (por `data_id` ni por nombre) ni es de los 8 sellados;
2. **licencia** (campo `licence` de OpenML) permisiva: `Public`, `public`, `CC BY 4.0`, `CC BY`, `CC0`/`CC 0: Public Domain`; se descarta
   cualquier otra (`GPL`, «Non-commercial…», vacía);
3. entre **500 y 20.000** filas (pequeño-mediano: el cubo de los 23);
4. como mucho **100 predictores** (`NumberOfFeatures` ≤ 101 contando el objetivo);
5. tarea y clases según el estudio: CC18 → binaria (2 clases) o multiclase (3–20); CTR23 → regresión;
6. **no es un hermano de origen de uno de la Fase 0** (cláusula añadida tras ver la PRIMERA salida de la regla, que eligió `mfeat-fourier` y
   `mfeat-karhunen`: otras dos representaciones de los mismos dígitos manuscritos que `mfeat-factors`). Solo metadatos a la vista, ningún
   resultado de modelo; se descartan los de prefijo `mfeat-`.

De los 107 candidatos: 24 ya están en la Fase 0, 2 tienen licencia no permisiva, ~25 pasan de 20.000 filas, ~14 de 100 columnas; los 6 de la
tabla son los primeros que sobreviven (el log completo de la selección está en el INFORME del corte).

## Los 6

| tarea | OpenML id | nombre (versión) | licencia (campo OpenML) | filas × cols | clases | faltantes | md5 (OpenML = descargado) | sha256 del fichero descargado |
|---|---|---|---|---|---|---|---|---|
| binaria | 15 | breast-w (1) | Public | 699 × 10 | 2 | 16 | 88633c065209e7a57323d3d7f2d00546 | cb44e7f62f4a962881147264e5f985eacc13256712e7a4b9d1731ced05f3ee9b |
| binaria | 29 | credit-approval (1) | Public | 690 × 16 | 2 | 67 | a948be968270a823fce432227c1aa36e | ae8e5eadb70e187885ddc6defb7f3e4afa726aa472ac63ebf5e387f0a8467891 |
| multiclase | 23 | cmc (1) | Public | 1.473 × 10 | 3 | 0 | 3149646ecff276abac3e892d1556655f | 9d8694d39e05d9d2963fccbe4f71e2af04c72c13de45f5dcbe775a586f214567 |
| multiclase | 54 | vehicle (1) | Public | 846 × 19 | 4 | 0 | fbba18157b188f309d772f9ca4e578f5 | 727ab4eb1bf76c0f01adccea696a07c690b5f1e25e15b3defc68c015360edf36 |
| regresión | 44956 | abalone (15) | CC BY 4.0 | 4.177 × 9 | — | 0 | 81f6d8b4dce14710077ab88389254d83 | 148af233543e257b41b2ad79b4c3a22116be68d8f7a177a1bd108ce94402c8f4 |
| regresión | 44957 | airfoil_self_noise (8) | CC BY 4.0 | 1.503 × 6 | — | 0 | 60e512ce65710079292e99a3acc71886 | 27c5c925f7f81911e9bd4664bba3656d2558d9ba301a6a0a617fac4b5797117a |

Descarga (OpenML, sin autenticación; el md5 coincide con el que declara OpenML en los seis):

```
curl -sSL -o breast-w.arff             https://openml.org/data/v1/download/52350/breast-w.arff
curl -sSL -o credit-approval.arff      https://openml.org/data/v1/download/29/credit-approval.arff
curl -sSL -o cmc.arff                  https://openml.org/data/v1/download/23/cmc.arff
curl -sSL -o vehicle.arff              https://openml.org/data/v1/download/54/vehicle.arff
curl -sSL -o abalone.arff              https://openml.org/data/v1/download/22111820/abalone.arff
curl -sSL -o airfoil_self_noise.arff   https://openml.org/data/v1/download/22111821/airfoil_self_noise.arff
```

## Cautelas (dichas antes de medir)

- **Licencia:** `Public` es lo que declara OpenML; el origen de cmc, credit-approval, breast-w, vehicle, abalone y airfoil es el repositorio UCI
  (que publica sus conjuntos bajo CC BY 4.0), pero eso es lo que recuerda quien escribe esto, NO se ha verificado en esta pasada: el supervisor lo
  comprueba antes de publicar nada que cite estos datos. `vehicle` pide en su cabecera reconocer su fuente (Turing Institute, Glasgow): hay
  que citarla. No se redistribuyen los ficheros, solo se citan id y sha256.
- **Tamaño:** cuatro de los seis tienen < 1.500 filas y dos < 700 (breast-w y credit-approval): una prueba del 20 % son ~140 filas y el IC de
  5 pliegues será muy ancho. Es lo que la regla produjo; si el supervisor prefiere un mínimo de 1.000 filas, se cambia ahora, antes de medir.
- **breast-w** es casi seguro un conjunto SATURADO (AUROC ≈ 0,99): se declarará con el mismo criterio (referencia ≥ 98 %) en la enmienda de
  confirmación, que se sella antes de medir; no se puede declarar aquí sin medirlo.
