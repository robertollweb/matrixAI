#!/bin/bash
# 170-C1 — lanza `medir_c170.py` DENTRO de la imagen de 116-C0 (numpy, pandas, scikit-learn, scipy, joblib) con `interpret-core==0.7.8` montado
# desde una instalación APARTE (`pip install --no-deps --target`, fuera de los árboles y del Python del host), con `--network none` y techo de recursos.
# Nada se instala en esta máquina (terreno). La imagen NO lleva `interpret-core`: se monta en `/ebm/site` y entra por `PYTHONPATH`.
#
# Uso:
#   correr_c170_en_contenedor.sh [--comprobar] [--salida-dir DIR] [--nombre FICHERO.json] [-- <argumentos de medir_c170, SIN --salida>]
#
# La RAÍZ de los repos (donde están `matrixAI/` y `matrixai-engines/` como hermanos) sale de la ubicación de ESTE guion: en el árbol de trabajo es /home/deployer;
# en la cola nocturna (`~/encolar.sh`), el worktree `/home/deployer/worktrees/cola-<nombre>/`. Así la medida corre el código FIJADO por la cola.
#
# Sin argumentos de medir, corre `--plan` (no lee datos). `--salida-dir` (por omisión ~/cola-nocturna/resultados/c170) se monta como /salida (rw); el resultado es
# <salida-dir>/<nombre> (resultado_c170.json). La medida real se niega a empezar sin la regla y el código SELLADOS (`medir_c170.py`); `--sonda` no.
#
# Ejemplos:
#   correr_c170_en_contenedor.sh --comprobar
#   correr_c170_en_contenedor.sh --salida-dir /home/deployer/.agentes/170/c1/out --nombre sonda_diabetes.json -- --sonda --solo diabetes
#   (en la cola)  ESTIMADA_S=… ~/encolar.sh c170 <tope> 6G matrixAI bash benchmarks/c170/correr_c170_en_contenedor.sh
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../../.." && pwd)"
SITE="${EBM_SITE:-/home/deployer/ebm_170_site}"
IMAGEN=medicion-tabicl-116c0-test:2.2.0
SALIDA_DIR="$HOME/cola-nocturna/resultados/c170"
NOMBRE=resultado_c170.json
COMPROBAR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --comprobar) COMPROBAR=1; shift ;;
    --salida-dir) SALIDA_DIR="$2"; shift 2 ;;
    --nombre) NOMBRE="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "argumento desconocido: $1" >&2; exit 2 ;;
  esac
done
ARGS=("$@")
[ "$COMPROBAR" -eq 0 ] && [ "${#ARGS[@]}" -eq 0 ] && ARGS=(--plan)

docker image inspect "$IMAGEN" >/dev/null 2>&1 || IMAGEN=medicion-tabicl-116c0:2.2.0
docker image inspect "$IMAGEN" >/dev/null 2>&1 || { echo "no existe la imagen de 116-C0 (medicion-tabicl-116c0[-test]:2.2.0)" >&2; exit 1; }
[ -d "$SITE/interpret" ] && ls -d "$SITE"/interpret_core-0.7.8.dist-info >/dev/null 2>&1 || { echo "falta $SITE con interpret-core 0.7.8 (pip install --no-deps --target)" >&2; exit 1; }
[ -d "$RAIZ/matrixAI" ] && [ -d "$RAIZ/matrixai-engines" ] || { echo "$RAIZ no tiene matrixAI/ y matrixai-engines/ como hermanos" >&2; exit 1; }
DATOS_HOST=/home/deployer/fase0_openml_datos
[ -d "$DATOS_HOST/arff" ] || { echo "no está $DATOS_HOST/arff (los ARFF de la Fase 0)" >&2; exit 1; }
mkdir -p "$SALIDA_DIR"

CARGA=$(cut -d' ' -f1 /proc/loadavg)
if [ "$COMPROBAR" -eq 0 ] && [ "${#ARGS[@]}" -gt 0 ] && [ "${ARGS[0]}" != "--plan" ]; then
  # una medición: un contenedor de medición/navegador a la vez, y la carga manda (el presupuesto de pared de cada intento es de reloj)
  if awk -v c="$CARGA" 'BEGIN{exit !(c>6)}'; then echo "carga $CARGA > 6: esperar antes de lanzar un contenedor (terreno)" >&2; exit 1; fi
  if docker ps --format '{{.Image}}' 2>/dev/null | grep -qiE 'medicion-tabicl|playwright'; then echo "ya hay un contenedor de medición o de navegador: uno a la vez (terreno)" >&2; exit 1; fi
fi

MONTAJES=(
  -v "$RAIZ/matrixAI:/home/deployer/matrixAI:ro"
  -v "$RAIZ/matrixai-engines:/home/deployer/matrixai-engines:ro"
  -v "$DATOS_HOST:/tmp/fase0_openml_datos:ro"
  -v "$SITE:/ebm/site:ro"
  -v "$SALIDA_DIR:/salida:rw"
)
# El contenedor NO trae `git`: sin esto la procedencia dice «SIN COMMIT» y la medida no se puede atar a una versión del código (medido en las sondas del 16-11). Se declara AQUÍ, en el host,
# y `medir_c170.py` lo copia a la salida (`commits_declarados_por_el_lanzador`). En la cola, el worktree está desacoplado en el commit fijado por `~/encolar.sh`.
declarar() { printf '"%s":{"commit":"%s","ficheros_modificados":%s}' "$1" "$(git -C "$RAIZ/$1" rev-parse HEAD 2>/dev/null || echo desconocido)" "$(git -C "$RAIZ/$1" status --short --untracked-files=no 2>/dev/null | wc -l)"; }
C170_COMMITS="{$(declarar matrixAI),$(declarar matrixai-engines)}"
echo "imagen: $IMAGEN | raíz: $RAIZ | interpret-core: $SITE | salida: $SALIDA_DIR/$NOMBRE | carga: $CARGA" >&2

NOMBRE_CONTENEDOR="medicion-170-$$"
trap 'docker rm -f "$NOMBRE_CONTENEDOR" >/dev/null 2>&1' EXIT INT TERM
COMUN=(--rm --init --name "$NOMBRE_CONTENEDOR" --network none --memory=5g --memory-swap=5g --cpus=4 --user "$(id -u):$(id -g)" -e HOME=/tmp
       -e C170_COMMITS="$C170_COMMITS" -e PYTHONPATH=/ebm/site -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=4 "${MONTAJES[@]}")

if [ "$COMPROBAR" -eq 1 ]; then
  docker run "${COMUN[@]}" -w /home/deployer/matrixAI/benchmarks/c170 "$IMAGEN" python3 -c '
import importlib, importlib.metadata as md
for m in ("numpy", "pandas", "scipy", "sklearn", "joblib", "threadpoolctl", "interpret", "interpret.glassbox", "analiza_c170", "medir_c170",
          "matrixai_engines.motores.interpret_ebm"):
    importlib.import_module(m); print("  OK   ", m)
print("interpret-core", md.version("interpret-core"))
# 172-C1-bis: por su NOMBRE, motor_para("interpret.ebm") da el perfil LIGERO (el del estudio); el 170 midio el COMPLETO (valores por
# omision de interpret-core). Se pide el completo de forma explicita: MotorEBM() sin perfil. NO volver a pedirlo por nombre.
from matrixai_engines.motores.interpret_ebm import MotorEBM
m = MotorEBM()
print("motor", m.nombre, m.version, "| no disponible:", m.motivo_si_no_puede_entrenar())
' &
  wait $!
  exit $?
fi

# `--init` y `& wait` NO son adorno (26-09): con `docker run` en primer plano bash aplaza el trap, y python como PID 1 ignora el SIGTERM; el contenedor sobreviviría al tope de la cola.
if [ "${ARGS[0]}" = "--plan" ]; then
  docker run "${COMUN[@]}" -w /home/deployer/matrixAI "$IMAGEN" python3 benchmarks/c170/medir_c170.py "${ARGS[@]}" &
else
  docker run "${COMUN[@]}" -w /home/deployer/matrixAI "$IMAGEN" python3 benchmarks/c170/medir_c170.py "${ARGS[@]}" --salida "/salida/$NOMBRE" &
fi
wait $!
