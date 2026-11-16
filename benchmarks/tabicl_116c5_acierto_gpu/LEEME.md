# Medir el ACIERTO de TabICL en tu GPU, en los conjuntos de la Fase 0 (116-C5)

**Para qué.** 116-C4 midió lo que cuesta TabICL en tu RTX 2000 Ada (tiempo y memoria). Esto mide **cuánto acierta**: corre
`tabicl.v2` con GPU en los conjuntos de la Fase 0 que caben en el techo del paquete GPU (≤ 16.000 filas de contexto y
≤ 512.000 celdas), con **exactamente** las particiones, semillas, repeticiones, presupuesto por cubo y métrica de cierre del
protocolo `113-C0.v2` que usó la pasada de CPU (116-C2). Aquí, en el servidor, lo emparejo pliegue a pliegue con lo que ya
medimos en CPU (LightGBM de la v2 y la red TabM de 119-C3) y sale la tabla «TabICL frente a TabM y frente a LightGBM:
gana / pierde / empate». **Un solo guion los corre todos, uno tras otro, sin que tengas que ir conjunto a conjunto.**

**Qué corre.** Los 24 conjuntos de abajo, que **van dentro del kit** (`datos/*.arff.gz`, con su sha256 en
`datos/manifiesto.json`; 5 MB) junto con el código del núcleo y de motores (`codigo/`, el que midió la pasada de CPU). El
contenedor corre **sin red** y no toca nada de tu PC: lo único que sale es un JSON y un log.

## Qué conjuntos entran y cuáles no

**Entran (24)** — no sellados y con TODOS sus pliegues dentro del techo de la GPU (calculado con la partición real, no
estimado):

| conjunto | cubo | métrica de cierre | pliegues | filas de contexto (máx.) | columnas | celdas (máx.) |
|---|---|---|---|---|---|---|
| balance-scale | pequeno | accuracy | 15 | 400 | 4 | 1.600 |
| dresses-sales | pequeno | auroc | 15 | 321 | 12 | 3.852 |
| diabetes | pequeno | auroc | 15 | 492 | 8 | 3.936 |
| climate-model-simulation-crashes | pequeno | auroc | 15 | 346 | 20 | 6.920 |
| kc2 | pequeno | auroc | 15 | 335 | 21 | 7.035 |
| yeast | pequeno | accuracy | 12 | 893 | 8 | 7.144 |
| Moneyball | pequeno | r2 | 15 | 789 | 14 | 11.046 |
| pc1 | pequeno | auroc | 15 | 711 | 21 | 14.931 |
| wilt | mediano | auroc | 15 | 3.098 | 5 | 15.490 |
| pc4 | pequeno | auroc | 15 | 934 | 37 | 34.558 |
| wine_quality | mediano | r2 | 15 | 4.159 | 11 | 45.749 |
| sick | mediano | auroc | 15 | 2.415 | 29 | 70.035 |
| house_prices_nominal | pequeno | r2 | 15 | 935 | 79 | 73.865 |
| pendigits | mediano | accuracy | 15 | 7.040 | 16 | 112.640 |
| ozone-level-8hr | mediano | auroc | 15 | 1.623 | 72 | 116.856 |
| jm1 | mediano | auroc | 15 | 6.967 | 21 | 146.307 |
| us_crime | pequeno | r2 | 15 | 1.276 | 126 | 160.776 |
| PhishingWebsites | mediano | auroc | 15 | 7.076 | 30 | 212.280 |
| optdigits | mediano | accuracy | 15 | 3.601 | 64 | 230.464 |
| house_16H | grande | r2 | 5 | 14.582 | 16 | 233.312 |
| mfeat-factors | pequeno | accuracy | 15 | 1.280 | 216 | 276.480 |
| house_sales | grande | r2 | 5 | 13.832 | 21 | 290.472 |
| SAT11-HAND-runtime-regression | mediano | r2 | 15 | 2.842 | 116 | 329.672 |
| micro-mass | pequeno | accuracy | 15 | 374 | 1300 | 486.200 |

**NO entran, por sellados (8):** Satellite, banknote-authentication, diamonds, kr-vs-kp, letter, pc3, pol, splice. Los sellados se reservan para la confirmatoria: no se miden ni se analizan, ni aquí ni
en ninguna prueba de este kit.

**NO entran, por no caber en el techo de la GPU (8)** (se estima con la partición que midió la v2; cada uno pasa de 16.000
filas o de 512.000 celdas, y ahí TabICL empezaría a descargar a RAM, que el portátil no aguanta):

| conjunto | por qué |
|---|---|
| Internet-Advertisements | ≈2099 filas de contexto y 1558 columnas = ≈3270242 celdas |
| Amazon_employee_access | ≈20973 filas de contexto y 9 columnas = ≈188757 celdas |
| KDDCup09_appetency | ≈32000 filas de contexto y 230 columnas = ≈7360000 celdas |
| okcupid-stem | ≈32506 filas de contexto y 19 columnas = ≈617614 celdas |
| connect-4 | ≈43236 filas de contexto y 42 columnas = ≈1815912 celdas |
| kick | ≈46710 filas de contexto y 32 columnas = ≈1494720 celdas |
| APSFailure | ≈48640 filas de contexto y 170 columnas = ≈8268800 celdas |
| Allstate_Claims_Severity | ≈120524 filas de contexto y 130 columnas = ≈15668120 celdas |

Un conjunto que **cabe en el techo pero justo** es `micro-mass` (486.200 de 512.000 celdas, 1.300 columnas): si tu tarjeta
informara menos VRAM que la medida (8.188 MB), el motor lo rechazaría pliegue a pliegue y quedaría anotado como fallo, no
como acierto.

## 1. Lo que necesita tu PC (una vez)

Lo mismo que 116-C4: driver de NVIDIA al día y **Docker Desktop con el motor WSL 2** (la GPU pasa a WSL con el driver
normal; no hay que instalar CUDA). Comprobarlo:

    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

Tiene que salir tu GPU.

## 2. Traer el kit (una vez)

La carpeta es `benchmarks/tabicl_116c5_acierto_gpu/` del núcleo (~15 MB). Desde tu PC:

    scp -r deployer@217.154.179.205:/home/deployer/matrixAI/benchmarks/tabicl_116c5_acierto_gpu .
    cd tabicl_116c5_acierto_gpu

## 3. Lanzar

**Windows (PowerShell o doble clic en `lanzar.bat`):**

    .\lanzar.ps1 -Prueba     # construye la imagen (1.a vez, 10-40 min) y mide 1 conjunto, 1 pliegue (~2 min)
    .\lanzar.ps1             # LA MEDICIÓN: todos los conjuntos

**Linux / WSL:** `./lanzar.sh --prueba` y después `./lanzar.sh`.

Lo que hace `lanzar`, por si prefieres los pasos a mano (como en 116-C4):

    docker build -t medicion-tabicl-acierto-gpu:2.2.0 .
    docker run --rm --gpus all --network none -v "${PWD}/salida:/salida" medicion-tabicl-acierto-gpu:2.2.0 python3 medir_acierto.py --prueba --salida /salida/prueba.json
    docker run --rm --gpus all --network none --memory=12g -v "${PWD}/salida:/salida" medicion-tabicl-acierto-gpu:2.2.0

(En Linux, `$PWD` en vez de `${PWD}`.) La construcción baja ~3 GB (torch 2.6.0 con CUDA 12.4, el del paquete GPU) y los pesos
de TabICL (215 MB), **verificados por su huella**: si Hugging Face sirviera otros, la imagen no se construye.

La primera línea del log trae `"hay_gpu": true` y el nombre de tu GPU, y cada pliegue dice `dispositivo=cuda`. **Sin GPU se
niega a empezar** (en CPU serían días): casi siempre es un `--gpus all` olvidado.

## 4. Cuánto tarda — ESTIMADO, NO MEDIDO

Sale de la tabla de tiempos de **tu** tarjeta del contrato 116 (§ «MEDIDO EN GPU», 12-11): un intento (ajuste + validación)
tarda ≤ 4,3 s hasta 64.000 celdas, ≤ 6,7 s a 128.000, ≤ 13,8 s a 256.000–512.000 (tomo el máximo de las tres tareas). Por
pliegue cuento **25 s de arranque** del proceso hijo (importar torch y pasar los pesos a la GPU; es un SUPUESTO, no está
medido) **+ 2 × ese tiempo** (el contrato mide ajuste + validación; aquí además se predice el test).

- **Total estimado: ≈ 3,8 h** (24 conjuntos, 337 pliegues). Los pequeños, ~8 min cada uno; los de 150.000–490.000 celdas,
  ~13 min cada uno (los dos de 5 pliegues, ~4 min).
- **Tope por conjunto**: 3 × su estimación con un mínimo de 10 min (25 min los pequeños, 29–39 los mayores). Si un conjunto lo
  supera en una ejecución, los pliegues que quedan se anotan «no lanzado» y se pasa al siguiente; al relanzar se reintentan.
  Peor caso con todos los topes agotados: ≈ 11,5 h (no debería pasar).
- **Tope por intento**: el del protocolo (2 min el cubo pequeño, 5 el mediano, 10 el grande); un intento que lo pasa se mata y
  queda como fallo (no se cuenta como acierto).
- `python3 medir_acierto.py --estimar` (o `docker run … medir_acierto.py --estimar`) imprime la tabla completa por conjunto.
- **Se puede cortar (Ctrl+C) y relanzar la misma orden**: guarda tras CADA pliegue y reusa los ya completados (los fallidos se
  reintentan). Con el portátil enchufado y sin suspensión.

## 5. Lo que me mandas

Los dos ficheros de `salida/`: **`resultado_116c5_acierto_gpu.json`** y **`log_acierto_gpu.txt`**.

## 6. Cómo lo leo (la regla, escrita ANTES de tener datos)

`REGLA.md` fija el margen de equivalencia (±0,005 de la métrica cruda), qué cuenta como gana / pierde / empate, cuándo un
conjunto es «no comparable» y qué NO se concluye. El analizador `analiza_kit_acierto.py` (corre aquí, no en tu PC) cita su
sha256 y se niega a correr si ha cambiado. Por conjunto: media de TabICL GPU, de TabM y de LightGBM; diferencia con intervalo
bootstrap al 95 % emparejado por (repetición, pliegue) — la misma `protocolo._intervalo_pareado` de 116-C2 —; y la tabla final.
Como control informativo, para los conjuntos donde 116-C2 midió TabICL en CPU, la diferencia GPU − CPU.

## Lo que este kit NO hace

No decide el peso de TabICL en el estudio ni su techo (eso es tiempo y memoria, ya medidos en 116-C4). No mide los sellados.
No compara con otros motores que no sean TabM y LightGBM. Y la GPU no es bit a bit igual que la CPU: el control GPU − CPU
enseña cuánto bailan las cifras por sí solas.
