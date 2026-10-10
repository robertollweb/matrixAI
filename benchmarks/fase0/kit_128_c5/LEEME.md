# Kit 128-C5 — ¿entrena mejor TabM en tu GPU? (se mide en TU PC)

**Qué hace.** Mide los 13 conjuntos de referencia del Studio dos veces, en el mismo PC y con la misma imagen del
paquete GPU, los dos con la GPU a la vista (`--gpus all`): **brazo A** con TabM en la CPU (`MATRIXAI_TABM_DISPOSITIVO=cpu`)
y **brazo B** con TabM en la GPU (`auto`). Solo cambia dónde entrena TabM. En cada brazo repite los 4 conjuntos pequeños de control para saber cuánto bailan las cifras
por sí solas (el *suelo de ruido*). La regla es la tuya (C5′, «banda de 1 punto + tiempo»): por conjunto, la diferencia del test
(GPU − CPU) dentro de ±1 punto (o el suelo, si es mayor) es «igual»; por encima, **sube**; entre −1 y −2, **baja
tolerada**; −2 o peor, o dejar de terminar, **empeora**. Si el estudio de la GPU tarda la mitad o menos y queda «igual»,
es una **mejora de tiempo**. **La GPU entra si ninguno empeora y las mejoras (suben + tiempo) superan a las bajas
toleradas.** Si no, TabM sigue en la CPU y no se pierde nada. Con `-SinCortar` (`--sin-cortar` en Linux) no se corta
nunca: mide los 13 y solo informa; el veredicto es el mismo.

Los 13 CSV ya van dentro (`datos/`, con su sha256 en `datos/manifiesto.json`): el estudio no necesita red ni descarga
nada. No toca tu instalación normal del Studio: usa contenedores propios (`c5-128-…`) y una red propia que borra al
terminar.

## Qué necesitas
- Docker con GPU NVIDIA (Docker Desktop con WSL2 en Windows, o Docker + NVIDIA Container Toolkit en Linux).
- Internet la **primera vez**: el lanzador construye en TU PC la capa de PyTorch con CUDA 12.4 (descarga de ~3 GB,
  estimado 10–40 min según tu línea), igual que hace el instalador del paquete GPU. Se hace una vez; después se reusa.
- Espacio: unos 4–5 GB para las imágenes, ~250 MB para el kit y unos pocos cientos de MB de resultados.
- La carpeta del kit trae `imagen/matrixai-studio-128.tar` (imagen base de este corte, ~190 MB), su identidad
  (`imagen/matrixai-studio-128.ids`: el ID de esa imagen, leído del tar al empaquetar) e `imagen/Dockerfile.gpu` (el del
  paquete GPU de este corte). **No hay que pasar ninguna imagen**: el lanzador carga el tar, construye la capa GPU
  (`matrixai-studio-gpu-128:local`, no pisa tu instalación normal) y mide contra ella. Si el tar falta, para y lo dice.
- **Si tu Docker ya tiene la imagen de un kit anterior** (las etiquetas son las mismas), el lanzador lo ve: compara el ID
  de `matrixai-studio:128` con el de este kit y, si no es el mismo, carga el tar y rehace la capa GPU él solo (vuelve a
  bajar PyTorch con CUDA, ~3 GB). Lo dice en pantalla. No hace falta `-Reconstruir`.

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
Si el brazo B empeora o deja de terminar un conjunto (o las bajas toleradas ya no se pueden compensar), **se corta solo** (no hace falta esperar las horas que quedan) y lo dice.
Ese corte **es el resultado**: si vuelves a lanzar la misma orden, no repite el brazo B (compone el JSON con él). Nunca
se pisa una pasada: la que no se continúa (otra imagen u otros ajustes, o un corte del instrumento) se guarda al lado
como `salida/brazo_B.anterior-<fecha-hora>.json`.
También **para solo** si lo que mide no es lo que se quiere medir: el brazo A entrenó TabM en la GPU, o en el B TabM gana
y su modelo entregado se entrenó en la CPU sin decirlo. Sale con «código 6» y el motivo: mándame `salida/` entero.

Opciones: `-Memoria 6g` (por omisión; es la del servidor, para que sea comparable) y `-Cpus 2`;
`-Brazos A` o `-Brazos B` para medir solo uno (el B necesita el A ya hecho); `-Solo nombre1,nombre2` para un
subconjunto; `-Reconstruir` borra la imagen base y la capa y las vuelve a hacer (si te mando otro tar). (En Linux:
`--memoria`, `--cpus`, `--brazos`, `--solo`, `--reconstruir`.) `-Reconstruir` ya no hace falta para un kit nuevo: el
lanzador compara la imagen con la del kit y la rehace si no es la misma. `-Imagen <imagen>` (`--imagen`) salta todo lo anterior y usa
una imagen GPU ya hecha.

**Lo que esta comparación incluye y no está medido.** Los dos brazos ven la GPU y solo cambia la señal de TabM, así que
lo que se compara es **encender la GPU para TabM**, entero. Eso incluye una cosa más que TabM: con TabM en la CPU (brazo
A) la red densa esconde la GPU a su proceso mientras entrena, como siempre; con TabM en la GPU (brazo B) no la esconde
(si la escondiera, TabM dejaba de verla en el reajuste final), y verla le cuesta algo de RAM del PC y de tiempo. Si el B
gasta más memoria, tarda más o se queda sin RAM en un conjunto grande, puede ser eso y no TabM. No lo he podido medir
sin una GPU: el resultado recoge la RAM pico de cada brazo (`veredicto.ram_pico_mb_por_brazo`, y por conjunto en
`recursos.ram_pico_mb`) para verlo. Y si el modelo entregado de un conjunto **cayó a la CPU** en el B (lo declara), el
veredicto lo lista (`veredicto.caidas_a_cpu_en_b`) y su mejora **no cuenta** como mejora de la GPU.

**3. Opcional (≈ 1 h, estimado): los 3 grandes con mucha memoria.** Con 6 GB, la guarda de RAM suele dejar a TabM
fuera de KDDCup09, APSFailure y Allstate. Para separar «no cabe» de «la GPU no gana», repítelos con memoria alta y el
suelo del resultado principal (`suelo_de_ruido` de `veredicto`):
`.\lanzar.ps1 -Memoria 24g -Solo KDDCup09_appetency,APSFailure,Allstate_Claims_Severity -Salida salida_grandes -Suelo <X>`.

**4. Opcional (estimado 20 min – 1,5 h, no medido): TabM por fases en la GPU.** Para decidir el plazo de TabM en el paquete GPU:
en la CPU (2 núcleos) una época de TabM en los tres grandes tarda ~100 s y la preparación 27–56 s, y ni con 1.800 s de plazo
llega a parar por paciencia (11–13 épocas), así que allí TabM se queda fuera. Este paso mide **en tu GPU** dónde se va el
tiempo de UN intento de TabM (carga, preparación, red, épocas, validación, checkpoint, exportación) con el plazo del producto y
con uno de 1.800 s, en **APSFailure, KDDCup09_appetency, Allstate_Claims_Severity, house_16H y mfeat-factors** (los dos
últimos paran por paciencia en la CPU, en 126 y 202 s). Va **después** de los dos brazos (el resultado principal ya está escrito
cuando empieza), con la misma imagen GPU, `--gpus all` y la misma memoria que los brazos (`-Memoria` / `--memoria`: 6 GB
por omisión, o la que pases):

| Windows (PowerShell) | Linux / WSL |
|---|---|
| `.\lanzar.ps1 -Fases` (con los brazos, o después de ellos) | `./lanzar.sh --fases` |
| `.\lanzar.ps1 -SoloFases` (solo este paso, brazos ya hechos) | `./lanzar.sh --solo-fases` |

Una prueba corta del paso (`dresses-sales`, plazo amplio de 120 s; unos minutos) con `-Fases -Prueba` / `--fases --prueba`.
Opciones: `-ConjuntosFases a,b` (`--conjuntos-fases`) para otros conjuntos del kit, `-PlazoFases 1800` (`--plazo-fases`) y
`-Memoria` (`--memoria`). **No reusa lo medido**: si lo cortas y lo relanzas, repite el paso entero.

Cálculo de la estimación (no medida; en la CPU del servidor, los mismos cinco conjuntos tardaron 1.420–1.540 s cada grande y
180–270 s cada mediano, 85 min en total): si una época en tu GPU es ~10 veces más rápida, son **minutos por conjunto** (la
preparación de los datos, ~30–60 s, es CPU y no baja); si no lo es, hasta ~25 min cada grande. Cada conjunto tiene un tope de 1 h.

Deja `salida/tabm_por_fases_gpu.json` (y `salida/tabm_por_fases_gpu_series/`), con, por conjunto y por corrida: las fases con su
tiempo, el dispositivo que **declaró el motor** (no el que se pidió: `dispositivo_de_tabm`, y si dice `la GPU SE USO`), la VRAM
(pico asignado y reservado por torch, y `nvidia-smi`) y la RAM. Si el motor cae a la CPU, lo dice ahí. El paso **no** mide la
RAM por proceso de la forma en que lo hace el servidor (no hay `docker exec` desde dentro): el pico es el del contenedor.

## Qué me mandas
**`salida/resultado_128_c5.json`** (si hiciste el paso 3, también `salida_grandes/resultado_128_c5.json`; si hiciste el 4, también `salida/tabm_por_fases_gpu.json`). Lleva la
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
