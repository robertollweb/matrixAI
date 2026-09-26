#!/bin/bash
# 107-C3.0, condición (3) — lanza `medir_c30_respondedor.py` DENTRO de la imagen de
# `respondedor/` (la única con `onnxruntime-genai`; el host no lo tiene a propósito, "no
# instales nada en el host", 26-09). Mismo esqueleto que
# `benchmarks/fase0/correr_116c2_en_contenedor.sh`: nombre propio + trap + `--init` +
# `& wait $!` (SIN esto el contenedor sobrevive a su propio tope -- medido el 26-09 con la
# cadena de la cola nocturna, ver ese guion), `--network none` (esta condición no necesita
# red: los pesos ya están descargados), techo medido y comprobación de carga/otro
# contenedor antes de arrancar.
#
# Uso:
#   correr_respondedor_en_contenedor.sh [RAIZ] [--comprobar] [-- <args de medir_c30_respondedor.py>]
#
# RAIZ contiene `matrixAI/` y `matrixai-engines/` como HERMANOS (matrixaistudio no hace
# falta: esta condición no lo toca). Por omisión `/home/deployer` (los árboles en vivo);
# MIENTRAS este corte viva SOLO en el worktree, se le pasa
# `/home/deployer/worktrees/107c3-respondedor` (que ya tiene `matrixai-engines` enlazado
# como hermano -- ver ese directorio).
#
# Sin argumentos de la pasada, corre `--estimar` (barato, no llama al modelo).
# Con `--comprobar`, en vez de medir importa las bibliotecas de la imagen y dice qué falta.
#
# Ejemplos:
#   correr_respondedor_en_contenedor.sh /home/deployer/worktrees/107c3-respondedor
#   correr_respondedor_en_contenedor.sh /home/deployer/worktrees/107c3-respondedor --comprobar
#   correr_respondedor_en_contenedor.sh /home/deployer/worktrees/107c3-respondedor -- --solo A
#
# Un directorio SIEMPRE disponible para salidas de prueba sin tocar el árbol de trabajo ni
# /tmp del host: /home/deployer/.agentes/107c3, montado dentro como /scratch (rw). La
# pasada real (sin --salida/--cache) escribe en benchmarks/texto_107c30/ del propio RAIZ,
# remontado aparte en lectura-escritura (el resto de matrixAI queda SOLO LECTURA).
set -euo pipefail

RAIZ="/home/deployer"
COMPROBAR=0
if [ "${1:-}" != "" ] && [ "${1:-}" != "--" ] && [ "${1:-}" != "--comprobar" ]; then
  RAIZ="$1"; shift
fi
if [ "${1:-}" == "--comprobar" ]; then
  COMPROBAR=1; shift
fi
if [ "${1:-}" == "--" ]; then
  shift
fi
ARGS=("$@")
if [ "$COMPROBAR" -eq 0 ] && [ "${#ARGS[@]}" -eq 0 ]; then
  ARGS=(--estimar)
fi

IMAGEN=medicion-respondedor-107c30:1
docker image inspect "$IMAGEN" >/dev/null 2>&1 || {
  echo "no existe la imagen $IMAGEN. Construirla es " \
       "benchmarks/texto_107c30/respondedor/preparar.sh (UNA vez, con red); este guion NO " \
       "la construye por su cuenta." >&2
  exit 1
}

CARGA=$(cut -d' ' -f1 /proc/loadavg)
if awk -v c="$CARGA" 'BEGIN{exit !(c>6)}'; then
  echo "carga $CARGA > 6: esperar a que baje antes de lanzar un contenedor (terreno)." >&2
  exit 1
fi
if docker ps --format '{{.Image}} {{.Names}}' 2>/dev/null | grep -qiE 'medicion-|playwright'; then
  echo "ya hay un contenedor de medición (respondedor, tabicl u otro) o de navegador " \
       "corriendo: un contenedor a la vez (terreno). Esperar a que termine." >&2
  exit 1
fi

PESOS_QWEN_HOST="$HOME/.cache/matrixai/text_responders/qwen2.5-0.5b-instruct-onnx-genai-cpu-int4"
[ -f "$PESOS_QWEN_HOST/genai_config.json" ] || {
  echo "no está $PESOS_QWEN_HOST/genai_config.json: faltan los pesos de Qwen " \
       "(sondear_respondedor.py los descarga la primera vez, con red)." >&2
  exit 1
}

EMBEDDINGS_C1_HOST="$HOME/.cache/matrixai/embeddings/paraphrase-multilingual-MiniLM-L12-v2-onnx-int8"
[ -f "$EMBEDDINGS_C1_HOST/tokenizer.json" ] || {
  echo "no está $EMBEDDINGS_C1_HOST/tokenizer.json: falta el tokenizador del embedding de " \
       "C1, que la condición (3) usa para truncar el texto igual que él (107-C1)." >&2
  exit 1
}

[ -d "$RAIZ/matrixAI" ] || { echo "$RAIZ/matrixAI no existe: no hay pasada que lanzar" >&2; exit 1; }
[ -d "$RAIZ/matrixai-engines" ] || { echo "$RAIZ/matrixai-engines no existe: no hay " \
                                          "MotorArbolLightGBM ni el proveedor de embeddings " \
                                          "que resolver" >&2; exit 1; }
[ -f "$RAIZ/matrixAI/benchmarks/texto_107c30/tareas/tarea_a.csv" ] || {
  echo "no está $RAIZ/matrixAI/benchmarks/texto_107c30/tareas/tarea_a.csv: las tareas no " \
       "están preparadas en ESTE árbol (preparar_tareas.py, o copiarlas desde donde ya se " \
       "midieron (0)/(1)/(2))." >&2
  exit 1
}

SCRATCH_HOST=/home/deployer/.agentes/107c3
mkdir -p "$SCRATCH_HOST"

MONTAJES=(
  -v "$RAIZ/matrixAI:/home/deployer/matrixAI:ro"
  -v "$RAIZ/matrixai-engines:/home/deployer/matrixai-engines:ro"
  -v "$RAIZ/matrixAI/benchmarks/texto_107c30:/home/deployer/matrixAI/benchmarks/texto_107c30:rw"
  -v "$PESOS_QWEN_HOST:/pesos_qwen/qwen2.5-0.5b-instruct-onnx-genai-cpu-int4:ro"
  -v "$EMBEDDINGS_C1_HOST:/pesos_embeddings/paraphrase-multilingual-MiniLM-L12-v2-onnx-int8:ro"
  -v "$SCRATCH_HOST:/scratch:rw"
)

echo "imagen: $IMAGEN"
echo "raiz de los repos: $RAIZ"
echo "pesos qwen: $PESOS_QWEN_HOST"
echo "tokenizador c1: $EMBEDDINGS_C1_HOST"
echo "carga: $CARGA"

if [ "$COMPROBAR" -eq 1 ]; then
  echo "comprobando bibliotecas y módulos dentro de la imagen..."
  exec docker run --rm --network none --memory=1g --memory-swap=1g --cpus=2 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    "${MONTAJES[@]}" \
    -w /home/deployer/matrixAI/benchmarks/texto_107c30 \
    "$IMAGEN" \
    python3 -c '
import importlib
MODULOS = [
    "numpy", "pandas", "sklearn", "lightgbm", "tokenizers", "onnxruntime", "onnxruntime_genai",
    "matrixai_engines", "matrixai_engines.motores.arbol_lightgbm",
    "matrixai_engines.embeddings.proveedor_de_texto", "matrixai_engines.embeddings.tokenizador_unigram",
    "matrixai.estudio", "matrixai.estudio.comparaciones", "matrixai.estudio.metricas",
    "matrixai.text.embeddings.descarga",
    "medir_c30", "sondear_respondedor", "medir_c30_respondedor",
]
faltan = []
for m in MODULOS:
    try:
        importlib.import_module(m)
        print(f"  OK    {m}")
    except Exception as exc:  # noqa: BLE001 -- se quiere ver TODAS, no parar en la primera
        faltan.append((m, f"{type(exc).__name__}: {exc}"))
        print(f"  FALTA {m}: {type(exc).__name__}: {exc}")
print()
if faltan:
    print(f"FALTAN {len(faltan)} de {len(MODULOS)}: {[m for m, _ in faltan]}")
else:
    print(f"TODOS los {len(MODULOS)} módulos importan dentro de la imagen.")
'
fi

echo "argumentos de la pasada: ${ARGS[*]}"
# EL CONTENEDOR SE PARA SI SE PARA ESTE GUION (mismo arreglo del 26-09 que
# correr_116c2_en_contenedor.sh, medido con una réplica de la cadena de la cola nocturna):
# con `docker run` en primer plano, bash APLAZA el trap hasta que vuelva, y python3 como
# PID 1 del contenedor ignora el SIGTERM que le reenvía el cliente sin `--init`. Con nombre
# propio + `wait`, la señal corta la espera y el trap lo quita.
NOMBRE_CONTENEDOR="medicion-respondedor-107c30-$$"
trap 'docker rm -f "$NOMBRE_CONTENEDOR" >/dev/null 2>&1' EXIT INT TERM
docker run --rm --init --name "$NOMBRE_CONTENEDOR" --network none \
  --memory=4g --memory-swap=4g --cpus=4 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e MATRIXAI_EMBEDDINGS_HOME=/pesos_embeddings \
  "${MONTAJES[@]}" \
  -w /home/deployer/matrixAI/benchmarks/texto_107c30 \
  "$IMAGEN" \
  python3 medir_c30_respondedor.py \
  --modelo-local /pesos_qwen/qwen2.5-0.5b-instruct-onnx-genai-cpu-int4 \
  "${ARGS[@]}" &
wait $!
