#!/usr/bin/env bash
# 128-C5 -- UNA orden: mide TabM en la CPU (brazo A) y en la GPU (brazo B) en ESTE PC, con el paquete GPU, y deja
# salida/resultado_128_c5.json para mandarlo. Linux (o WSL2). Ver LEEME.md.
#
#   ./lanzar.sh [--prueba] [--memoria 6g] [--cpus 2] [--brazos A|B|A,B] [--solo a,b] [--suelo X] [--salida DIR]
#               [--sin-cortar] [--fases | --solo-fases] [--conjuntos-fases a,b] [--plazo-fases 1800]
#               [--reconstruir] [--imagen IMG --imagen-medidor IMG]
#
# SIN --imagen (lo normal): carga la imagen BASE de este corte (imagen/matrixai-studio-128.tar) y CONSTRUYE en este PC
# la capa GPU (imagen/Dockerfile.gpu: PyTorch con CUDA 12.4, ~3 GB de descarga), igual que el instalador del paquete GPU.
# Con --imagen se usa una ya hecha y no se construye nada.
# Antes de medir COMPRUEBA que la base de este Docker es la de ESTE kit (su ID contra imagen/matrixai-studio-128.ids) y
# que la capa se construyo sobre ella; si no, carga el tar y rehace la capa solo (128, I3 de la auditoria final: el kit 2
# reusaba en silencio la imagen del kit 1, con las mismas etiquetas).
#
# Los DOS brazos van con --gpus all; solo cambia donde entrena TabM (A: cpu, B: auto). 128, I2 de la auditoria final.
#
# --fases (OPCIONAL, DESPUES de los dos brazos): mide TabM POR FASES en la GPU (128-R4: carga, preparacion, red, epocas,
# validacion...) en 5 conjuntos y deja salida/tabm_por_fases_gpu.json. --solo-fases: solo ese paso (con los brazos ya hechos).
#
# Se puede cortar (Ctrl+C) y volver a lanzar con la MISMA orden: sigue donde estaba.
set -uo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
FASES=0 ; SOLO_FASES=0 ; CONJ_FASES="APSFailure,KDDCup09_appetency,Allstate_Claims_Severity,house_16H,mfeat-factors" ; PLAZO_FASES=1800
BRAZOS=AB ; IMAGEN="" ; RECONSTRUIR=0 ; PRUEBA=0 ; MEMORIA=6g ; CPUS=2 ; SOLO="" ; SUELO="" ; SALIDA="$AQUI/salida" ; MEDIDOR="" ; SIN_CORTAR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --imagen) IMAGEN="$2"; shift 2;;
    --imagen-medidor) MEDIDOR="$2"; shift 2;;
    --prueba) PRUEBA=1; shift;;
    --sin-cortar) SIN_CORTAR=1; shift;;
    --reconstruir) RECONSTRUIR=1; shift;;
    --fases) FASES=1; shift;;
    --solo-fases) FASES=1; SOLO_FASES=1; BRAZOS=""; shift;;
    --conjuntos-fases) CONJ_FASES="$(echo "$2" | tr -d ' ')"; shift 2;;
    --plazo-fases) PLAZO_FASES="$2"; shift 2;;
    --brazos) BRAZOS="$(echo "$2" | tr -d , | tr a-z A-Z)"; shift 2;;
    --memoria) MEMORIA="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --solo) SOLO="$2"; shift 2;;
    --suelo) SUELO="$2"; shift 2;;
    --salida) SALIDA="$2"; shift 2;;
    -h|--help) sed -n 2,10p "$0"; exit 0;;
    *) echo "Argumento desconocido: $1"; exit 2;;
  esac
done
command -v docker >/dev/null || { echo "No encuentro docker."; exit 2; }
BASE=matrixai-studio:128            # la etiqueta que lleva el tar de imagen/ (build-image.sh --image matrixai-studio:128)
CAPA=matrixai-studio-gpu-128:local  # la imagen GPU que se construye AQUI; no pisa ninguna de tu instalacion normal
mkdir -p "$SALIDA/maquina" ; chmod -R a+rwX "$SALIDA" 2>/dev/null || true
SALIDA="$(cd "$SALIDA" && pwd)"
RED=c5-128-red
log() { echo "[$(date '+%H:%M:%S')] $*"; }
limpiar() { docker rm -f c5-128-studio-A c5-128-studio-B c5-128-medidor c5-128-fases >/dev/null 2>&1; docker network rm "$RED" >/dev/null 2>&1; }
trap 'limpiar' EXIT
trap 'echo; echo "Cortado. Vuelve a lanzar la MISMA orden para seguir donde estaba."; exit 130' INT TERM
limpiar

# --- -1. la imagen: base (tar) + capa GPU construida en ESTE PC (salvo --imagen)
# 128, I3 de la auditoria final: NO basta con que las etiquetas existan (las del kit anterior son las mismas). La base
# tiene que ser la del tar de ESTE kit (su ID en imagen/matrixai-studio-128.ids, que apunta el empaquetador leyendo el
# tar) y la capa, construida sobre ESA base (la etiqueta $ETIQUETA_DE_LA_BASE que se le pone al construirla).
IDS="$AQUI/imagen/matrixai-studio-128.ids"
TAR="$AQUI/imagen/matrixai-studio-128.tar"
ETIQUETA_DE_LA_BASE=matrixai.kit.base
# 128, m7 de la re-auditoria final: la etiqueta de la base NO cubria el Dockerfile.gpu: un kit con la misma base y otro
# Dockerfile.gpu reusaba la capa vieja. Segunda etiqueta con el sha256 del Dockerfile.gpu del kit; la capa vale si
# lleva LAS DOS (si cambia el Dockerfile, se rehace). Una capa de antes (sin esta etiqueta) tambien se rehace.
ETIQUETA_DEL_DOCKERFILE=matrixai.kit.dockerfile
sha_del_fichero() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else openssl dgst -sha256 "$1" | sed 's/^.*= //'; fi
}
id_de() { docker image inspect "$1" --format '{{.Id}}' 2>/dev/null; }
es_la_base_del_kit() { local id; id="$(id_de "$BASE")" && [ -n "$id" ] && grep -Fxq -- "$id" "$IDS"; }
capa_sobre() {
  local etiquetas; etiquetas="$(docker image inspect "$CAPA" --format '{{json .Config.Labels}}' 2>/dev/null)" || return 1
  printf '%s' "$etiquetas" | grep -Fq "\"$ETIQUETA_DE_LA_BASE\":\"$1\"" && printf '%s' "$etiquetas" | grep -Fq "\"$ETIQUETA_DEL_DOCKERFILE\":\"$2\""
}
if [ -z "$IMAGEN" ]; then
  [ "$RECONSTRUIR" = 1 ] && docker rmi -f "$CAPA" "$BASE" >/dev/null 2>&1
  [ -f "$IDS" ] || { echo "PARO: falta $IDS (la identidad de la imagen base, que el empaquetador apunta junto a imagen/matrixai-studio-128.tar): sin ella no se sabe si la imagen de este Docker es la de este kit. Pide el kit entero o pasa --imagen."; exit 2; }
  if ! es_la_base_del_kit; then
    if docker image inspect "$BASE" >/dev/null 2>&1; then
      log "La imagen $BASE de este Docker NO es la de este kit ($(id_de "$BASE")): cargo la del kit y rehago la capa GPU."
    fi
    [ -f "$TAR" ] || { echo "PARO: falta $TAR (la imagen base de este corte). Pidela o pasa --imagen."; exit 2; }
    log "Cargando la imagen base (docker load, ~190 MB)..."
    docker load -i "$TAR" || { echo "PARO: docker load fallo."; exit 2; }
    docker image inspect "$BASE" >/dev/null 2>&1 || { echo "PARO: el tar no trae la etiqueta $BASE."; exit 2; }
    es_la_base_del_kit || { echo "PARO: cargado el tar, $BASE es $(id_de "$BASE") y el kit apunta otra ($IDS): el tar no es el de este kit."; exit 2; }
  fi
  BASE_ID="$(id_de "$BASE")"
  [ -f "$AQUI/imagen/Dockerfile.gpu" ] || { echo "PARO: falta $AQUI/imagen/Dockerfile.gpu."; exit 2; }
  DOCKERFILE_SHA="$(sha_del_fichero "$AQUI/imagen/Dockerfile.gpu")"
  if ! capa_sobre "$BASE_ID" "$DOCKERFILE_SHA"; then
    docker image inspect "$CAPA" >/dev/null 2>&1 && log "La capa GPU $CAPA de este Docker no se construyo sobre la base de este kit con su Dockerfile.gpu: la rehago."
    mkdir -p "$SALIDA/maquina/capa" && sed "s#^FROM .*#FROM $BASE#" "$AQUI/imagen/Dockerfile.gpu" > "$SALIDA/maquina/capa/Dockerfile"
    log "Construyendo la capa GPU (descarga PyTorch con CUDA, ~3 GB: tarda; se hace UNA vez por imagen base)..."
    docker build -t "$CAPA" --label "$ETIQUETA_DE_LA_BASE=$BASE_ID" --label "$ETIQUETA_DEL_DOCKERFILE=$DOCKERFILE_SHA" "$SALIDA/maquina/capa" 2>&1 | tee "$SALIDA/maquina/build_capa.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PARO: no se pudo construir la capa GPU (mira $SALIDA/maquina/build_capa.log; suele ser la red o el disco)."; exit 2; }
    capa_sobre "$BASE_ID" "$DOCKERFILE_SHA" || { echo "PARO: la capa GPU construida no dice que este sobre la base $BASE_ID y el Dockerfile.gpu $DOCKERFILE_SHA."; exit 2; }
  fi
  IMAGEN="$CAPA"
fi
[ -n "$MEDIDOR" ] || MEDIDOR="$IMAGEN"
docker image inspect "$IMAGEN" >/dev/null 2>&1 || { echo "No existe la imagen $IMAGEN en este Docker (docker images)."; exit 2; }
IMAGEN_ID="$(docker image inspect "$IMAGEN" --format '{{.Id}}')"

# --- 0. la GPU de verdad, dentro de Docker (si no, no se empieza). Para cualquier brazo: los DOS llevan --gpus all.
case "$BRAZOS$([ "$FASES" = 1 ] && echo B)" in *A*|*B*)
log "Paso 0: comprobando que Docker ve la GPU..."
docker run --rm --gpus all --entrypoint python3 "$IMAGEN" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0) if ok else None); sys.exit(0 if ok else 1)" \
  | tee "$SALIDA/maquina/torch_gpu.txt" ; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PARO: Docker no ve la GPU con esta imagen (revisa NVIDIA Container Toolkit / Docker Desktop con GPU)."; exit 3; }
docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGEN" -q > "$SALIDA/maquina/nvidia-smi.txt" 2>&1 || true
;; esac
{ docker version; echo ---; docker info; } > "$SALIDA/maquina/docker.txt" 2>&1 || true
{ lscpu; echo ---; free -m; echo ---; uname -a; } > "$SALIDA/maquina/sistema.txt" 2>&1 || true
echo "$IMAGEN $IMAGEN_ID" > "$SALIDA/maquina/imagen.txt"
docker network create "$RED" >/dev/null

# --- 1. verificar los datos del kit (contra el manifiesto, byte a byte)
docker run --rm --entrypoint python3 -v "$AQUI":/work/kit:ro "$MEDIDOR" /work/kit/medir_c5.py --verificar-datos --datos /work/kit/datos || exit 2

medidor() {  # medidor <brazo> [extra...]
  local brazo="$1"; shift
  # los dos con la GPU a la vista (I2): el entorno que apunta el medidor lo dice en los dos
  local ent='{}'; [ "$brazo" = A ] && ent='{"MATRIXAI_TABM_DISPOSITIVO":"cpu","gpus":"all"}' || ent='{"MATRIXAI_TABM_DISPOSITIVO":"auto","gpus":"all"}'
  docker run --rm --name c5-128-medidor --network "$RED" --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python3 -v "$AQUI":/work/kit:ro -v "$SALIDA":/salida "$MEDIDOR" \
    /work/kit/medir_c5.py --brazo "$brazo" --studio "http://c5-128-studio-$brazo:8765" --salida /salida --datos /work/kit/datos \
    --imagen "$IMAGEN" --imagen-id "$IMAGEN_ID" --memoria "$MEMORIA" --cpus "$CPUS" --entorno "$ent" \
    ${SOLO:+--solo "$SOLO"} ${SUELO:+--suelo "$SUELO"} $( [ "$PRUEBA" = 1 ] && echo --prueba ) $( [ "$SIN_CORTAR" = 1 ] && echo --sin-cortar ) "$@"
}

brazo() {  # brazo <A|B>
  local b="$1" intento=0 rc
  while [ $intento -lt 3 ]; do
    intento=$((intento+1))
    docker rm -f "c5-128-studio-$b" >/dev/null 2>&1
    # 128, I2 de la auditoria final: los DOS brazos ven la GPU (--gpus all) y solo cambia donde entrena TabM. Asi la
    # comparacion aisla a TabM: con la GPU a la vista y TabM en cpu, la densa la esconde como en la v2.12 (A); con
    # TabM en auto no la esconde y el coste de verla va con TabM en la GPU (B).
    local gpu=(--gpus all); local disp=cpu
    [ "$b" = B ] && disp=auto
    log "Brazo $b: arrancando el Studio (TabM en '$disp', memoria $MEMORIA, $CPUS CPU) ..."
    docker run -d --init --name "c5-128-studio-$b" --network "$RED" --memory="$MEMORIA" --memory-swap="$MEMORIA" --cpus="$CPUS" \
      -e MATRIXAI_LICENSE_ENABLED=false -e "MATRIXAI_TABM_DISPOSITIVO=$disp" "${gpu[@]}" \
      -v "$AQUI":/work/kit:ro -v "$SALIDA":/salida "$IMAGEN" >/dev/null || { echo "PARO: no arranca el Studio del brazo $b"; return 4; }
    docker exec -d "c5-128-studio-$b" python3 /work/kit/muestreador.py "/salida/muestras_$b.jsonl" >/dev/null 2>&1 || true
    medidor "$b"; rc=$?
    docker exec "c5-128-studio-$b" python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'disponible',torch.cuda.is_available())" > "$SALIDA/maquina/torch_$b.txt" 2>&1 || true
    docker logs "c5-128-studio-$b" > "$SALIDA/studio_$b.log" 2>&1 || true
    docker rm -f "c5-128-studio-$b" >/dev/null 2>&1
    [ $rc -eq 4 ] && { log "El Studio del brazo $b dejo de contestar: lo rearranco y sigo (intento $intento de 3)."; continue; }
    return $rc
  done
  return 4
}

if [ "$PRUEBA" = 1 ]; then
  log "PRUEBA: un conjunto pequeno por brazo (unos minutos, estimado). Debe decir 'la GPU SE USO'."
fi
for b in A B; do
  case "$BRAZOS" in *"$b"*) brazo "$b" ; rc=$? ; [ $rc -eq 0 ] || { echo "El brazo $b termino con codigo $rc."; exit $rc; };; esac
done
# --- 3 (OPCIONAL, --fases): TabM POR FASES en la GPU. La sonda del 128-R4 corre DENTRO de la imagen GPU (no hace falta Python en
# este PC) con --gpus all y el mismo techo de memoria; los CSV son los del kit (sha256 comprobado). Deja
# salida/tabm_por_fases_gpu.json. Va DESPUES de componer: si falla, el resultado principal ya esta escrito.
paso_fases() {
  local conj="$CONJ_FASES" plazo="$PLAZO_FASES"
  [ "$PRUEBA" = 1 ] && { conj="dresses-sales"; plazo=120; }
  log "Paso opcional: TabM por fases en la GPU ($conj; plazo amplio $plazo s; memoria $MEMORIA)..."
  docker rm -f c5-128-fases >/dev/null 2>&1
  docker run --rm --init --name c5-128-fases --gpus all --memory="$MEMORIA" --memory-swap="$MEMORIA" --cpus="$CPUS" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e MATRIXAI_LICENSE_ENABLED=false --entrypoint python3 \
    -v "$AQUI":/work/kit:ro -v "$SALIDA":/salida "$IMAGEN" \
    /work/kit/sonda_128_c0.py --tabm-por-fases --gpu --dentro-del-contenedor --datos-del-kit /work/kit/datos \
    --temporales /salida/fases_tmp --salida /salida/tabm_por_fases_gpu.json --imagen "$IMAGEN" --imagen-id "$IMAGEN_ID" \
    --memoria "$MEMORIA" --plazo-amplio "$plazo" --tope-estudio 3600 --solo "$conj"
  local rc=$?
  [ $rc -eq 0 ] && log "Hecho. Manda tambien: $SALIDA/tabm_por_fases_gpu.json (y la carpeta tabm_por_fases_gpu_series/)." \
                || echo "El paso de fases termino con codigo $rc (lo ya medido esta en $SALIDA/tabm_por_fases_gpu.json)."
  return $rc
}

COMPONER=1
if [ "$SOLO_FASES" = 1 ]; then COMPONER=0
elif [ "$PRUEBA" = 1 ]; then log "Prueba hecha. Ahora lanza la medicion de verdad (sin --prueba)."; COMPONER=0
else case "$BRAZOS" in AB) ;; *) log "Hecho el brazo $BRAZOS (falta el otro para componer el resultado)."; COMPONER=0; FASES=0;; esac
fi
if [ "$COMPONER" = 1 ]; then
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python3 -v "$AQUI":/work/kit:ro -v "$SALIDA":/salida "$MEDIDOR" /work/kit/medir_c5.py --componer --salida /salida --datos /work/kit/datos ${SUELO:+--suelo "$SUELO"}
  log "Hecho. Manda: $SALIDA/resultado_128_c5.json"
fi
[ "$FASES" = 1 ] && { paso_fases; exit $?; }
exit 0
