# Medir TabICL en tu GPU (116-C4)

**Para qué.** TabICL cumple en su régimen (116-C2: mejora a lightgbm en 7 de 8 conjuntos
pequeños), y decidiste meterlo primero en el paquete GPU. Pero en CPU cada intento cuesta
~72 s, y la búsqueda entera de un estudio tiene 140 s para todos los motores: tal cual, no
competiría. Esto mide lo que cuesta en una GPU de verdad (la tuya) —tiempo de un intento,
memoria de la GPU y hasta cuántas filas y columnas llega— para decidir con números su peso en
el estudio y sus techos.

**Qué corre.** Datos sintéticos (binaria, multiclase de 10 clases y regresión), generados
dentro del contenedor. No lee ni envía nada de tu PC: el contenedor corre **sin red**. Lo único
que sale es un JSON con los tiempos, la memoria y los datos de la máquina (modelo de GPU, RAM,
versiones).

## 1. Lo que necesita tu PC (una vez)

- El driver de NVIDIA al día.
- Docker con acceso a la GPU:
  - **Windows**: Docker Desktop con el motor de WSL 2 (la GPU pasa a WSL con el driver normal
    de NVIDIA; no hay que instalar CUDA aparte).
  - **Linux**: Docker y el `nvidia-container-toolkit`.
- Comprobarlo:

      docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

  Tiene que salir tu GPU. Si no sale, lo de abajo no la verá tampoco.

## 2. Traer el kit y construir la imagen (una vez, con red)

La carpeta es `benchmarks/tabicl_116c4_gpu/` del núcleo. Desde tu PC:

    scp -r deployer@217.154.179.205:/home/deployer/matrixAI/benchmarks/tabicl_116c4_gpu .
    cd tabicl_116c4_gpu
    docker build -t medicion-tabicl-gpu:2.2.0 .

Baja ~3 GB (torch con CUDA 12.4, el mismo que la imagen GPU del Studio) y los pesos de TabICL
(215 MB), que se **verifican por su huella**: si Hugging Face sirviera otros, la construcción
falla con el motivo.

## 3. Probar que ve la GPU (un minuto)

Windows (PowerShell):

    docker run --rm --gpus all --network none -v "${PWD}/salida:/salida" medicion-tabicl-gpu:2.2.0 python3 medir_todo.py --prueba --salida /salida/prueba.json

Linux:

    docker run --rm --gpus all --network none -v "$PWD/salida:/salida" medicion-tabicl-gpu:2.2.0 python3 medir_todo.py --prueba --salida /salida/prueba.json

La primera línea que imprime dice `"hay_gpu": true` y el nombre de tu GPU, y cada punto dice
`"dispositivo": "cuda"` en `salida/prueba.json`.

## 4. Medir

Windows:

    docker run --rm --gpus all --network none --memory=32g -v "${PWD}/salida:/salida" medicion-tabicl-gpu:2.2.0

Linux:

    docker run --rm --gpus all --network none --memory=32g -v "$PWD/salida:/salida" medicion-tabicl-gpu:2.2.0

- **Duración: no la sé, es justo lo que se mide.** Son 72 puntos (3 tareas × 4 anchuras × 6
  tamaños, de 1.000 a 32.000 filas), cada uno con un tope de 15 minutos; si uno no termina, los
  más grandes de su fila no se lanzan. Va diciendo `[n/72]` con el tiempo de cada intento.
- **Se puede cortar** (Ctrl+C) y relanzar la misma orden: sigue donde iba.
- **Sin GPU se niega a empezar** (en CPU serían días): casi siempre es un `--gpus all` olvidado.

## 5. Lo que me mandas

`salida/resultado_116c4_gpu.json`. Con eso decido (y te lo digo con los números) cuánto peso
lleva TabICL en el reparto del estudio, hasta dónde subir sus techos y si cabe en la reserva
de siempre o necesita más.
