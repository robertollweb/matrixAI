#!/usr/bin/env bash
# 128-C5 -- UNA orden: mide TabM en la CPU (brazo A) y en la GPU (brazo B) en ESTE PC, con el paquete GPU, y deja
# salida/resultado_128_c5.json para mandarlo. Linux (o WSL2). Ver LEEME.md.
#
#   ./lanzar.sh [--prueba] [--memoria 6g] [--cpus 2] [--brazos A|B|A,B] [--solo a,b] [--suelo X] [--salida DIR]
#               [--sin-cortar]
#               [--reconstruir] [--imagen IMG --imagen-medidor IMG]
#
# SIN --imagen (lo normal): carga la imagen BASE de este corte (imagen/matrixai-studio-128.tar) y CONSTRUYE en este PC
# la capa GPU (imagen/Dockerfile.gpu: PyTorch con CUDA 12.4, ~3 GB de descarga), igual que el instalador del paquete GPU.
# Con --imagen se usa una ya hecha y no se construye nada.
#
# Se puede cortar (Ctrl+C) y volver a lanzar con la MISMA orden: sigue donde estaba.
set -uo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
BRAZOS=AB ; IMAGEN="" ; RECONSTRUIR=0 ; PRUEBA=0 ; MEMORIA=6g ; CPUS=2 ; SOLO="" ; SUELO="" ; SALIDA="$AQUI/salida" ; MEDIDOR="" ; SIN_CORTAR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --imagen) IMAGEN="$2"; shift 2;;
    --imagen-medidor) MEDIDOR="$2"; shift 2;;
    --prueba) PRUEBA=1; shift;;
    --sin-cortar) SIN_CORTAR=1; shift;;
    --reconstruir) RECONSTRUIR=1; shift;;
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
limpiar() { docker rm -f c5-128-studio-A c5-128-studio-B c5-128-medidor >/dev/null 2>&1; docker network rm "$RED" >/dev/null 2>&1; }
trap 'limpiar' EXIT
trap 'echo; echo "Cortado. Vuelve a lanzar la MISMA orden para seguir donde estaba."; exit 130' INT TERM
limpiar

# --- -1. la imagen: base (tar) + capa GPU construida en ESTE PC (salvo --imagen)
if [ -z "$IMAGEN" ]; then
  [ "$RECONSTRUIR" = 1 ] && docker rmi -f "$CAPA" "$BASE" >/dev/null 2>&1
  if ! docker image inspect "$CAPA" >/dev/null 2>&1; then
    if ! docker image inspect "$BASE" >/dev/null 2>&1; then
      [ -f "$AQUI/imagen/matrixai-studio-128.tar" ] || { echo "PARO: falta $AQUI/imagen/matrixai-studio-128.tar (la imagen base de este corte). Pidela o pasa --imagen."; exit 2; }
      log "Cargando la imagen base (docker load, ~190 MB)..."
      docker load -i "$AQUI/imagen/matrixai-studio-128.tar" || { echo "PARO: docker load fallo."; exit 2; }
      docker image inspect "$BASE" >/dev/null 2>&1 || { echo "PARO: el tar no trae la etiqueta $BASE."; exit 2; }
    fi
    [ -f "$AQUI/imagen/Dockerfile.gpu" ] || { echo "PARO: falta $AQUI/imagen/Dockerfile.gpu."; exit 2; }
    mkdir -p "$SALIDA/maquina/capa" && sed "s#^FROM .*#FROM $BASE#" "$AQUI/imagen/Dockerfile.gpu" > "$SALIDA/maquina/capa/Dockerfile"
    log "Construyendo la capa GPU (descarga PyTorch con CUDA, ~3 GB: tarda; se hace UNA vez)..."
    docker build -t "$CAPA" "$SALIDA/maquina/capa" 2>&1 | tee "$SALIDA/maquina/build_capa.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PARO: no se pudo construir la capa GPU (mira $SALIDA/maquina/build_capa.log; suele ser la red o el disco)."; exit 2; }
  fi
  IMAGEN="$CAPA"
fi
[ -n "$MEDIDOR" ] || MEDIDOR="$IMAGEN"
docker image inspect "$IMAGEN" >/dev/null 2>&1 || { echo "No existe la imagen $IMAGEN en este Docker (docker images)."; exit 2; }
IMAGEN_ID="$(docker image inspect "$IMAGEN" --format '{{.Id}}')"

# --- 0. la GPU de verdad, dentro de Docker (si no, no se empieza). Solo si se va a medir el brazo B.
case "$BRAZOS" in *B*)
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
  local ent='{}'; [ "$brazo" = A ] && ent='{"MATRIXAI_TABM_DISPOSITIVO":"cpu"}' || ent='{"MATRIXAI_TABM_DISPOSITIVO":"auto","gpus":"all"}'
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
    local gpu=(); local disp=cpu
    [ "$b" = B ] && { gpu=(--gpus all); disp=auto; }
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
if [ "$PRUEBA" = 1 ]; then log "Prueba hecha. Ahora lanza la medicion de verdad (sin --prueba)."; exit 0; fi
case "$BRAZOS" in AB) ;; *) log "Hecho el brazo $BRAZOS (falta el otro para componer el resultado)."; exit 0;; esac
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python3 -v "$AQUI":/work/kit:ro -v "$SALIDA":/salida "$MEDIDOR" /work/kit/medir_c5.py --componer --salida /salida --datos /work/kit/datos ${SUELO:+--suelo "$SUELO"}
log "Hecho. Manda: $SALIDA/resultado_128_c5.json"
