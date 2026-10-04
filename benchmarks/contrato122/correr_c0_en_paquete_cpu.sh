#!/bin/bash
# 122-C0 — corre la medida de confirmación DENTRO de la imagen del paquete CPU del Studio
# (sus condiciones reales: torch de CPU, 2 CPU, 6 GB, sin swap).
#
#   correr_c0_en_paquete_cpu.sh <directorio de salida> [argumentos de c0_confirmacion.py…]
#
#   p. ej. (humo):  correr_c0_en_paquete_cpu.sh /tmp/c0 --humo --salida /salida/humo.json
#         (pasada): correr_c0_en_paquete_cpu.sh ~/cola-nocturna/resultados/c122-c0 \
#                       --salida /salida/resultado_122_c0.json
#
# Lo que decide, y por qué:
# * El núcleo que se mide es el del ÁRBOL (el worktree se monta en /work y el guion exige que
#   `matrixai` salga de ahí); los motores son los de la imagen. Su ruta va en la procedencia.
# * La imagen no trae git: el commit y la suciedad de los TRES repos los escribe el host
#   (`--escribir-git-del-host`) antes de arrancar, y el guion los lee (`fuente: host`).
# * Los ARFF de la Fase 0 van de solo lectura en /datos.
# * El contenedor lleva NOMBRE, `--init` y un trap: un `docker run` lanzado por un guion
#   sobrevive a su `timeout` (memoria «contenedor sobrevive al tope»).
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,16p "$0"; exit 2; }
SALIDA="$1"; shift
IMAGEN="${IMAGEN_DEL_PAQUETE_CPU:-matrixai-studio-cpu-local:v2.8}"
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
NOMBRE="c122c0-$$"
mkdir -p "$SALIDA"
python3 "$RAIZ/benchmarks/contrato122/c0_confirmacion.py" --escribir-git-del-host "$SALIDA/git_del_host.json"
trap 'docker rm -f "$NOMBRE" >/dev/null 2>&1 || true' EXIT INT TERM
docker run --rm --init --name "$NOMBRE" --cpus=2 --memory=6g --memory-swap=6g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$RAIZ":/work -v "${DATOS_ARFF:-$HOME/fase0_openml_datos/arff}":/datos:ro \
  -v "$SALIDA":/salida -w /work --entrypoint python3 "$IMAGEN" \
  benchmarks/contrato122/c0_confirmacion.py --arff /datos --git-del-host /salida/git_del_host.json "$@" &
wait $!
