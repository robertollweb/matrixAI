#!/usr/bin/env bash
# 116-C5 -- lo mismo que lanzar.ps1, para Linux o dentro de WSL2.   ./lanzar.sh [--prueba] [--memoria 12g]
set -uo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"; cd "$AQUI"
IMAGEN=medicion-tabicl-acierto-gpu:2.2.0 ; MEMORIA=12g ; PRUEBA=0
while [ $# -gt 0 ]; do case "$1" in --prueba) PRUEBA=1; shift;; --memoria) MEMORIA="$2"; shift 2;; *) echo "argumento desconocido: $1"; exit 2;; esac; done
mkdir -p "$AQUI/salida"
if ! docker image inspect "$IMAGEN" >/dev/null 2>&1; then
  echo "Construyendo la imagen (la primera vez: ~3 GB de PyTorch con CUDA y 215 MB de pesos, 10-40 min)..."
  docker build -t "$IMAGEN" . || { echo "FALLO al construir la imagen"; exit 1; }
fi
echo "Comprobando que Docker ve la GPU..."
docker run --rm --gpus all "$IMAGEN" python3 -c "import torch,sys; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'); sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "Docker NO ve la GPU (LEEME.md, paso 1)"; exit 1; }
if [ "$PRUEBA" = 1 ]; then
  exec docker run --rm --gpus all --network none --memory="$MEMORIA" -v "$AQUI/salida:/salida" "$IMAGEN" python3 medir_acierto.py --prueba --salida /salida/prueba.json
fi
docker run --rm --gpus all --network none --memory="$MEMORIA" -v "$AQUI/salida:/salida" "$IMAGEN"
echo "Terminado. Mandame salida/resultado_116c5_acierto_gpu.json y salida/log_acierto_gpu.txt"
