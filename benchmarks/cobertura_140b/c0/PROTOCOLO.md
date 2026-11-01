# 140b-C0 — PROTOCOLO (sellado antes de correr) — 2026-11-01 01:4x

Núcleo importado: worktree `rev30/matrixAI` @ `ebadf76`; motores `rev30/matrixai-engines/src` @ `9b3936e`. Nada de los repos se edita.
Código de la medición (sha256, congelado al sellar):

```
3f82cce801846c742bf2ce126087166df5e3cdc4cc349543a19acf68252e201f  lib140b.py
02489bf17c152416ba5d6ef56a2dfe1d253856edb3d28a0b9bf492ff04fd9122  simcob.py
b766da338d84f1344cc420f46795d27dc5afe5d0e7fa74c019e96e59ed7dd427  e1_geometria.py
e6808e003c910c5c7b21d847c1ed68e0bacf8a776297346d46d7b0603f942e16  e2_pliegues.py
fe8d59a76c6a30336375a3e20b9372fb03e92c8b66d7c0b037b6ab7a43c8539f  e3_cobertura.py
db5ad1df8f62ab8c9013492368fb15c2cdd1a13f72b8ed2e0f2dd0290cd2fd0d  e4_nucleo_y_nota.py
```

Antes de sellar se corrieron los cuatro guiones en modo `--humo` (2–3 semillas, rejilla mínima) SOLO para ver que arrancan y cuadran los controles;
se vieron esas líneas de humo (Parkinson: M3 falla en 1 de 3 semillas con s=0,2; F2 sobre el calendario de 1000 días deja pliegues de validación casi
vacíos, lo que motivó añadir la variante F2q ANTES de sellar). Los umbrales de abajo se escriben con eso a la vista y sin ningún número de la corrida completa.

## 1. Las preguntas

- **P1 — Mínimos** de la prueba del reparto estricto (propuesta [S]: M1 ≥ 5 sujetos con filas de prueba, M2 ≥ 50 filas de prueba, M3 cada clase en ≥ 2 sujetos de prueba): con qué tamaños y formas se cumplen y cuándo no.
- **P2 — B y pliegues**: B0 (sujetos de prueba al azar entre todos; el prototipo del inventario) frente a B1 (al azar entre los que tienen filas posteriores al corte+hueco/2), con filas perdidas, sujetos en prueba, y prueba posterior al desarrollo; pliegues F1 (por sujeto), F2 (sujeto × último tramo = última fracción 0,3 del SPAN de tiempo del desarrollo) y F2q (último tramo = último 30 % de las FILAS del desarrollo), asignación al azar (`a`) o estratificada por positivos (`e`), con bajada de pliegues 5→3→2. Datos: Parkinson (tiempo relativo y dos calendarios sintéticos de entrada uniforme en 365 y 1000 días), Avocado, lluvia, y 75 cohortes sintéticas.
- **P3 — Cobertura (D4)** del IC95 del bootstrap por unidad sobre las filas de prueba, con verdad conocida, con y sin deriva temporal COMPARTIDA entre sujetos.
- **P4 — Sensibilidad de q (cuantil de filas del corte) y s (cuota de sujetos de prueba)**: filas perdidas, filas/sujetos de prueba, tasa de cumplimiento de mínimos, nota de un HGB sin afinar y anchura del IC.

## 2. Qué se mide (rejilla RECORTADA)

**E1 geometría** (`e1_geometria.py`, semillas reales 0–19, sintéticas 0–19): s ∈ {0,2; 0,3}, q ∈ {0,5; 0,6; 0,7; 0,8; 0,9}, hueco ∈ {0; 5 % del tramo de tiempo} (solo reales), B0 y B1. Reparto AUTOMÁTICO: escalera s = 0,2/0,3/0,4 (exterior, el primero que cumple) × q = 0,9…0,4 (interior, el mayor que cumple los mínimos); si ninguno, rechazo con los números del mejor intento. Sintéticos: n_suj ∈ {10,20,42,100,500} × visitas ∈ {2,5,20,100,'d20' (lognormal mediana 20)} × calendario ∈ {panel, entrada uniforme en 1× el seguimiento, en 3×}; objetivo binario con efecto-sujeto (sd 1,5): equilibrado (≈50 %) y raro (≈4 %).
**E2 pliegues** (`e2_pliegues.py`, semillas 0–19): sobre el desarrollo del reparto automático (B1, hueco 0): F1/F2/F2q × {azar, estratificado} × K ∈ {5,3,2}; viabilidad: valida ≥ 30 filas, entrena ≥ 100, y en clasificación `ok1` = cada clase ≥ 1 en valida y entrena, `ok5` = además ≥ 5 de cada clase en valida. Mismas cohortes (clase equilibrada y rara) y los reales (regresión y clasificación).
**E3 cobertura** (`e3_cobertura.py`): modelo fijo (ŷ = 0) sobre y = u_i + d(t) + e (u~N(0,ρ), e~N(0,1−ρ), d proceso gaussiano suave compartido, ventana 100, longitud 30); métrica RMSE (verdad analítica: condicional a la ventana realizada y marginal sobre la deriva); n sujetos de prueba ∈ {5,8,20,50,200} × visitas ∈ {3,10,50,'d10'} × ρ ∈ {0; 0,3; 0,8} × deriva (τ,calendario) ∈ {(0,panel),(0,esc),(0,5 panel),(0,5 esc),(1 panel),(1 esc)}; R = 500 réplicas, B = 500 remuestras, nivel 95 %. AUROC (verdad condicional por Monte Carlo con 6.000 sujetos nuevos y el mismo d; marginal = media de las condicionales): n ∈ {8,50} × sd_u ∈ {0; 1,5} × τ ∈ {0; 1} (deriva en el logit, panel), m = 10, R = 300, B = 300.
**E4** (`e4_nucleo_y_nota.py`): (a) el bootstrap numpy de E3 contra `intervalo(..., diseno='groups')` del núcleo (3 celdas RMSE + 1 AUROC, 40 réplicas, B = 1000, mismos datos); (b) HGB sin afinar en Parkinson y Avocado (regresión), B1, hueco 0, s × q de arriba, semillas 0–7: RMSE, referencia «media del desarrollo», dispersión entre semillas, ancho del IC95 por unidad.

## 3. CONTROLES (si uno falla, no se sacan conclusiones de su etapa)

- **E1-C1**: la publicada `groups_and_time` (generador REAL del núcleo) en Parkinson da dev 4.693/34 sujetos, prueba 1.182/8, 1.062 filas de prueba anteriores al fin del desarrollo (90 %); con calendario sintético 365: 4.586/33, 1.289/9, 1.002 (78 %) [inventario s02].
- **E1-C2**: mi B0 reproduce, redondeado, las cinco filas del inventario (Parkinson s=0,2 q=0,7 → 3.312/34 y 325/8; s=0,3 q=0,5 → 2.025/29 y 902/13; cal365 → 3.350/33 y 366/4; Avocado → 10.147/43 y 1.121/11; lluvia → 47.250/45 y 4.950/11), 5 semillas.
- **E1-C3**: en B1 la prueba es posterior al desarrollo (máx t desarrollo < mín t prueba − hueco) y ningún sujeto está a los dos lados, en los 5 conjuntos reales.
- **E3-Ca**: con unidades de 1 fila, ρ = 0, sin deriva (i.i.d.), n = 100 y 300, R = 1.000: la cobertura cae en [0,93; 0,97].
- **E3-Cb**: la verdad analítica coincide con Monte Carlo (400.000 sujetos nuevos) dentro de 4 errores estándar en tres configuraciones.
- **E4-Ca**: numpy y núcleo equivalentes en cada celda: media de |Δlímite|/ancho ≤ 0,08 en cada extremo, razón de anchos en [0,95; 1,05], acuerdo en «¿cubre la verdad?» ≥ 90 %. Si falla, la cobertura de E3 no se atribuye al método del núcleo.

## 4. CRITERIOS DE DECISIÓN (escritos antes de medir)

**P1 mínimos.**
1. *Sujetos.* `n*` = el menor n de la rejilla de E3 tal que, para ese n y todos los mayores, en las celdas SIN deriva (ρ ≤ 0,8, todas las visitas), la cobertura condicional es ≥ 0,925 en ≥ 90 % de las celdas y ninguna < 0,90. Recomendación: **rechazo duro** con M1 = 5 (prueba sin información: se rechaza con números) y **aviso blando** («intervalo poco fiable») por debajo de `n*` si `n*` > 5. Si `n*` ≤ 5, M1 = 5 queda tal cual.
2. *Filas (M2) y clase (M3)* se recomiendan si tienen dientes: hay celdas de E1 donde M1 pasa y M2 falla (resp. M1 y M2 pasan y M3 falla) con tasa > 5 %. Si no hay ninguna, se declara redundante en la rejilla medida (y se dice que se mantiene como guarda barata o se retira, según el caso).
3. *Reparto automático*: debe NO rechazar (rechazos ≤ 10 % de semillas) en cada uno de los conjuntos REALES con hueco 0 (Parkinson, Avocado, lluvia, calendarios 365/1000) y en las formas sintéticas de referencia (n_suj ∈ {42,100,500} × visitas ∈ {20,100,'d20'} × los tres calendarios, clase equilibrada). Si rechaza más, se dice qué mínimo lo causa y se propone relajarlo o la escalera.
4. *Formas donde se rechaza o se avisa*: se tabula por (n_suj, visitas, calendario) la tasa de rechazo del automático (regresión, clase equilibrada, clase rara); «no viable» = rechazo > 50 %.

**P2 B y pliegues.**
1. *B1 frente a B0*: se recomienda B1 si en ninguna celda su tasa de cumplimiento de M1&M2 es menor que la de B0 por más de 0,03 y en alguna es mayor por ≥ 0,10; si no, se declaran equivalentes. «Filas perdidas»: se informan; se marca como aviso fuerte toda celda con > 50 % de filas fuera.
2. *Prueba posterior*: cumplirlo por construcción (E1-C3). Se informa, para la publicada, el % de prueba anterior (E1-C1) y, para B1, 0 %.
3. *Pliegues viables*: una variante es **viable** en un caso si, con bajada de pliegues (algún K de 5,3,2), la tasa de semillas con `ok5` (condición completa) entre las que pasan el reparto automático es ≥ 0,90. **F2 (D3) se confirma** si F2 o F2q es viable en los cinco conjuntos reales (clasificación; y regresión) y en todas las formas sintéticas de referencia del punto 3 de P1 (clase equilibrada). En otro caso, se recomienda F1 DECLARADO en la zona donde F2/F2q no es viable, con la frontera (menor n_suj y mínimas filas de desarrollo con viabilidad ≥ 0,90). Se prefiere **F2q a F2** si F2 no es viable en algún caso donde F2q sí. La asignación **estratificada** se recomienda si, en alguna celda, mejora la viabilidad en ≥ 0,05 estando la aleatoria < 0,90; si no, no hace falta.

**P3 cobertura (D4).** Tolerancia: con R = 500 el error de Monte Carlo es ≈ 1 punto; **aceptable** = [0,925; 0,975], **infracubre** < 0,925 (grave < 0,90), **sobrecubre** > 0,975 (conservador).
- *Sin deriva*: PUBLICAR sin nota donde la cobertura condicional sea aceptable en todas las celdas de un n dado (y mayores); por debajo, aviso («intervalo aproximado») o NO DISPONIBLE si es grave en > 25 % de las celdas.
- *Con deriva*: si cond es aceptable pero **marginal** < 0,925 → PUBLICAR CON NOTA («condicional a esta ventana de tiempo, no al futuro en general»); si la cobertura **condicional** misma < 0,925 en celdas con n ≥ `n*` → el bootstrap por unidad NO basta con deriva compartida: no publicarlo como IC sin esa advertencia (la decisión de producto va a Roberto con los números). Se informa por separado el efecto del calendario (panel vs escalonado).
- AUROC: mismos criterios con su menor rejilla; sirve para decir si el patrón de RMSE se repite en una métrica no lineal.

**P4 q y s.** Parámetros por omisión = la celda (s,q) de {0,2; 0,3}×{0,5…0,9} que, cumpliendo (i) tasa de cumplimiento de mínimos (M1&M2&M3) ≥ 0,90 en CADA forma de referencia (los cinco reales con B1 y hueco 0, y las formas sintéticas de referencia de P1.3) y (ii) filas de prueba medias ≥ 150 en cada real, tenga el menor % de filas descartadas; si ninguna cumple, la de mayor cumplimiento mínimo, y se dice. Se informa el segundo mejor y el ancho de IC95 de E4. Se informa cómo mueve la nota q (si el cambio de RMSE entre q = 0,5 y 0,9 supera 1 desviación típica entre semillas, «la nota depende de q» y se declara).

## 5. Qué NO se mide (de antemano)

Ningún estudio real del pipeline entero; HGB sin afinar solo en Parkinson y Avocado (lluvia sin características en la caché: solo geometría y pliegues); modelo ENTRENADO dentro de la cobertura (el IC del núcleo es de modelo fijo: no incluye la variación del entrenamiento); métricas distintas de RMSE y AUROC; hueco ≠ 0 en pliegues y cobertura; calendarios reales con sujetos (no existen en la máquina); deriva en la media vs en la varianza (solo en la media); `reentrenar_con_todo`, conforme, simulador (cobertura relajada), tamaños > 50.000 filas en sintéticos.

## 6. Duración estimada y techo

Estimado por etapa (medido en humo y por cronometraje de una réplica): E1 ≈ 4–6 min, E2 ≈ 2–3, E3 ≈ 6–8, E4 ≈ 3–4: **≈ 15–20 min de CPU en total, < 30 min → de día**. Techo por etapa: 900 s, 2 GB, `CPUQuota=200%`, `MemorySwapMax=0`, `nice`. Etapas en serie, espera por carga ≤ 6.
