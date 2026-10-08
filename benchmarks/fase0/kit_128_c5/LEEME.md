# Kit 128-C5 — ¿entrena mejor TabM en tu GPU? (se mide en TU PC)

**Qué hace.** Mide los 13 conjuntos de referencia del Studio dos veces, en el mismo PC y con la misma imagen del
paquete GPU: **brazo A** con TabM en la CPU (`MATRIXAI_TABM_DISPOSITIVO=cpu`) y **brazo B** con TabM en la GPU
(`auto` y `--gpus all`). En cada brazo repite los 4 conjuntos pequeños de control para saber cuánto bailan las cifras
por sí solas (el *suelo de ruido*). La regla es la tuya: **la GPU entra solo si ningún conjunto baja más que ese
suelo, ninguno deja de terminar y alguno mejora**. Si no, TabM sigue en la CPU y no se pierde nada.

Los 13 CSV ya van dentro (`datos/`, con su sha256 en `datos/manifiesto.json`): el estudio no necesita red ni descarga
nada. No toca tu instalación normal del Studio: usa contenedores propios (`c5-128-…`) y una red propia que borra al
terminar.

## Qué necesitas
- Docker con GPU NVIDIA (Docker Desktop con WSL2 en Windows, o Docker + NVIDIA Container Toolkit en Linux).
- Internet la **primera vez**: el lanzador construye en TU PC la capa de PyTorch con CUDA 12.4 (descarga de ~3 GB,
  estimado 10–40 min según tu línea), igual que hace el instalador del paquete GPU. Se hace una vez; después se reusa.
- Espacio: unos 4–5 GB para las imágenes, ~250 MB para el kit y unos pocos cientos de MB de resultados.
- La carpeta del kit trae `imagen/matrixai-studio-128.tar` (imagen base de este corte, ~190 MB) y `imagen/Dockerfile.gpu`
  (el del paquete GPU de este corte). **No hay que pasar ninguna imagen**: el lanzador carga el tar, construye la capa
  GPU (`matrixai-studio-gpu-128:local`, no pisa tu instalación normal) y mide contra ella. Si el tar falta, para y lo dice.

## Cómo se lanza (UNA orden por sistema)
**1. Prueba corta (≈ 5–10 min más la construcción de la capa la primera vez, estimado).** Un conjunto pequeño por brazo;
debe acabar diciendo `la GPU SE USO`:

| Windows (PowerShell) | Linux / WSL |
|---|---|
| `.\lanzar.ps1 -Prueba` | `./lanzar.sh --prueba` |

(En Windows también vale `lanzar.bat -Prueba`.) Si dice que no usó la GPU, **para** y mándame `salida/` entero: no sigas.

**2. La medición de verdad (≈ 3,5–4,5 h desatendido, estimado, no medido).** La misma orden sin la prueba:

| Windows | Linux / WSL |
|---|---|
| `.\lanzar.ps1` | `./lanzar.sh` |

Cálculo de la estimación (no medida): brazo A ≈ 2–2,5 h, brazo B ≈ 1,3–2 h, más la repetición de los 4 de control en
cada uno. Cada conjunto tiene un tope de 1 h; si algún conjunto grande no termina, el brazo sigue con el siguiente.
**Se puede cortar** (Ctrl+C, apagar el PC) **y volver a lanzar la MISMA orden**: sigue donde estaba (reusa lo ya
medido con la misma imagen y los mismos ajustes; la capa GPU ya construida no se vuelve a bajar).
Si el brazo B baja o deja de terminar un conjunto, **se corta solo** (no hace falta esperar las horas que quedan) y lo dice.
Ese corte **es el resultado**: si vuelves a lanzar la misma orden, no repite el brazo B (compone el JSON con él). Nunca
se pisa una pasada: la que no se continúa (otra imagen u otros ajustes, o un corte del instrumento) se guarda al lado
como `salida/brazo_B.anterior-<fecha-hora>.json`.
También **para solo** si lo que mide no es lo que se quiere medir: el brazo A entrenó TabM en la GPU, o en el B TabM gana
y su modelo entregado se entrenó en la CPU sin decirlo. Sale con «código 6» y el motivo: mándame `salida/` entero.

Opciones: `-Memoria 6g` (por omisión; es la del servidor, para que sea comparable) y `-Cpus 2`;
`-Brazos A` o `-Brazos B` para medir solo uno (el B necesita el A ya hecho); `-Solo nombre1,nombre2` para un
subconjunto; `-Reconstruir` borra la imagen base y la capa y las vuelve a hacer (si te mando otro tar). (En Linux:
`--memoria`, `--cpus`, `--brazos`, `--solo`, `--reconstruir`.) `-Imagen <imagen>` (`--imagen`) salta todo lo anterior y usa
una imagen GPU ya hecha.

**3. Opcional (≈ 1 h, estimado): los 3 grandes con mucha memoria.** Con 6 GB, la guarda de RAM suele dejar a TabM
fuera de KDDCup09, APSFailure y Allstate. Para separar «no cabe» de «la GPU no gana», repítelos con memoria alta y el
suelo del resultado principal (`suelo_de_ruido` de `veredicto`):
`.\lanzar.ps1 -Memoria 24g -Solo KDDCup09_appetency,APSFailure,Allstate_Claims_Severity -Salida salida_grandes -Suelo <X>`.

## Qué me mandas
**`salida/resultado_128_c5.json`** (si hiciste el paso 3, también `salida_grandes/resultado_128_c5.json`). Lleva la
procedencia (imagen, GPU, driver, versiones de torch y CUDA, CPU y RAM), las cifras de cada conjunto en los dos
brazos, los picos de RAM y de memoria de la GPU, el suelo de ruido de cada brazo y el veredicto. Si algo falla, manda
la carpeta `salida/` entera (lleva los registros del Studio de cada brazo). Para releer el veredicto en el servidor:
`python3 benchmarks/fase0/veredicto_128_c5.py resultado_128_c5.json`.

## Si algo va mal
- *«Docker no ve la GPU»*: el paso 0 lo prueba antes de empezar. En Windows, Docker Desktop → Settings → Resources → WSL
  integration, y el driver NVIDIA al día. `docker run --rm --gpus all matrixai-studio-gpu-128:local nvidia-smi` tiene que funcionar.
- *«No se pudo construir la capa GPU»*: casi siempre la red (la descarga es de ~3 GB) o el disco lleno; el registro está en
  `salida/maquina/build_capa.log`. Vuelve a lanzar la misma orden: reintenta.
- *El Studio de un brazo deja de contestar*: el lanzador lo rearranca hasta 3 veces y sigue.
- *Quieres empezar de cero*: borra la carpeta `salida/`.
