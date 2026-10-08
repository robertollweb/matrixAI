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
- La imagen del **paquete GPU con el corte C5** (`docker images`; la que te indique Claude). Si le pasas una anterior,
  la prueba lo dice: «la GPU NO SE USO».
- Espacio: unos 100 MB para el kit y unos pocos cientos de MB de resultados.

## Cómo se lanza (UNA orden por sistema)
**1. Prueba corta (≈ 5–10 min, estimado).** Un conjunto pequeño por brazo; debe acabar diciendo `la GPU SE USO`:

| Windows (PowerShell) | Linux / WSL |
|---|---|
| `.\lanzar.ps1 -Imagen <imagen> -Prueba` | `./lanzar.sh --imagen <imagen> --prueba` |

(En Windows también vale `lanzar.bat -Imagen <imagen> -Prueba`.) Si dice que no usó la GPU, **para** y mándame
`salida/` entero: no sigas.

**2. La medición de verdad (≈ 3,5–4,5 h desatendido, estimado, no medido).** La misma orden sin la prueba:

| Windows | Linux / WSL |
|---|---|
| `.\lanzar.ps1 -Imagen <imagen>` | `./lanzar.sh --imagen <imagen>` |

Cálculo de la estimación (no medida): brazo A ≈ 2–2,5 h, brazo B ≈ 1,3–2 h, más la repetición de los 4 de control en
cada uno. Cada conjunto tiene un tope de 1 h; si algún conjunto grande no termina, el brazo sigue con el siguiente.
**Se puede cortar** (Ctrl+C, apagar el PC) **y volver a lanzar la MISMA orden**: sigue donde estaba (reusa lo ya
medido con la misma imagen y los mismos ajustes). Si el brazo B baja o deja de terminar un conjunto, **se corta solo**
(no hace falta esperar las horas que quedan) y lo dice.

Opciones: `-Memoria 6g` (por omisión; es la del servidor, para que sea comparable) y `-Cpus 2`;
`-Brazos A` o `-Brazos B` para medir solo uno (el B necesita el A ya hecho); `-Solo nombre1,nombre2` para un
subconjunto. (En Linux: `--memoria`, `--cpus`, `--brazos`, `--solo`.)

**3. Opcional (≈ 1 h, estimado): los 3 grandes con mucha memoria.** Con 6 GB, la guarda de RAM suele dejar a TabM
fuera de KDDCup09, APSFailure y Allstate. Para separar «no cabe» de «la GPU no gana», repítelos con memoria alta y el
suelo del resultado principal (`suelo_de_ruido` de `veredicto`):
`.\lanzar.ps1 -Imagen <imagen> -Memoria 24g -Solo KDDCup09_appetency,APSFailure,Allstate_Claims_Severity -Salida salida_grandes -Suelo <X>`.

## Qué me mandas
**`salida/resultado_128_c5.json`** (si hiciste el paso 3, también `salida_grandes/resultado_128_c5.json`). Lleva la
procedencia (imagen, GPU, driver, versiones de torch y CUDA, CPU y RAM), las cifras de cada conjunto en los dos
brazos, los picos de RAM y de memoria de la GPU, el suelo de ruido de cada brazo y el veredicto. Si algo falla, manda
la carpeta `salida/` entera (lleva los registros del Studio de cada brazo). Para releer el veredicto en el servidor:
`python3 benchmarks/fase0/veredicto_128_c5.py resultado_128_c5.json`.

## Si algo va mal
- *«Docker no ve la GPU»*: el paso 0 lo prueba antes de empezar. En Windows, Docker Desktop → Settings → Resources → WSL
  integration, y el driver NVIDIA al día. `docker run --rm --gpus all <imagen> nvidia-smi` tiene que funcionar.
- *El Studio de un brazo deja de contestar*: el lanzador lo rearranca hasta 3 veces y sigue.
- *Quieres empezar de cero*: borra la carpeta `salida/`.
