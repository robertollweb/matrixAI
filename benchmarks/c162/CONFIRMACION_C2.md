# Contrato 162 (C2) — conjuntos de CONFIRMACIÓN de TabM con k = 32 (cubo PEQUEÑO), fijados el 12-11-2026 ANTES de medir nada

**No se ha medido ni se mide nada sobre ellos** (ningún modelo, de ningún tipo). Son, tal cual, los conjuntos PEQUEÑOS de
`benchmarks/c162/CONFIRMACION.md` (los que el 162-C1 apartó con la regla de selección del 160, sin mirar resultados), porque el cubo pequeño
del protocolo v2 es de **500 a 2.000 filas** (medido en `protocolo_exploratorio_v2.json`: pequeño 500–2.000, mediano 2.534–20.000):

| tarea | OpenML id | nombre | filas × cols | cubo | sha256 del fichero descargado (el de `CONFIRMACION.md`) |
|---|---|---|---|---|---|
| binaria | 31 | credit-g | 1.000 × 21 | pequeño | 768be84ef749110d5fcac109a6541f2df190e8c6987e1dc432c0a4f29d0642f5 |
| multiclase | 188 | eucalyptus | 736 × 20 | pequeño | 59a2fe2cb13540ea64a01faba34048209e096bb75a105b9c053950df4f6d678c |
| multiclase | 307 | vowel | 990 × 13 | pequeño | a0526fad8f69a04a9b8844b756046ab0dca5a97954e14b5af08d0c85e74e7e72 |
| regresión | 44959 | concrete_compressive_strength | 1.030 × 9 | pequeño | 510beb3f50cba7907815b611af40edf14eb75902995bb9d7bccb40db156cf8ec |

Quedan fuera, por ser **medianos** (> 2.000 filas): `spambase` (4.601 filas) y `auction_verification` (2.043). Son **4 ≥ 3**: no hay que apartar
ninguno más ni se ha mirado ningún resultado para decidirlo.

**Cautela dicha antes de medir (la de siempre en esta serie).** Estos mismos cuatro son también los de la confirmación del C1. Cada palanca
del 162 se confirma una vez; compartir los conjuntos no contamina a ninguna de las dos (se miden por separado, contra su propia referencia), pero
sí hace que las tres confirmaciones sean menos independientes entre sí. Si el supervisor prefiere conjuntos propios para el C2, se aparta
con `seleccion_confirmacion.py` (los SIGUIENTES a estos cuatro) ANTES de medir, y esta enmienda se cambia ahora.

**Saturados:** no se declara ninguno aquí (no se puede sin medir; `vowel` podría estar cerca). Se declararán con el criterio del 160 (mejor
referencia ≥ 98 %) en la enmienda sellada de confirmación, antes de medirlos (`REGLA_C2.md` §5). **Registrarlos en el catálogo de la Fase 0**
antes de correrlos es trabajo del corte de confirmación (igual que en el C1).
