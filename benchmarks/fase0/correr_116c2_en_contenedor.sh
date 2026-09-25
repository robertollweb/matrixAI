#!/bin/bash
# 116-C2 — lanza `pasada_116c2_tabicl.py` DENTRO de la imagen de 116-C0 (la
# única que tiene `tabicl` instalado), nunca suelta: el host no tiene
# `tabicl` a propósito (registro de C2, «Dónde corre»; terreno: nada se
# instala a mano en esta máquina).
#
# Uso:
#   correr_116c2_en_contenedor.sh [RAIZ] [--comprobar] [-- <args de la pasada>]
#
# RAIZ es el directorio que contiene `matrixAI/`, `matrixai-engines/` y
# `matrixaistudio/` como HERMANOS — por omisión `/home/deployer` (los
# árboles de trabajo en vivo). La cola nocturna (`~/encolar.sh`) fija un
# worktree en `/home/deployer/worktrees/cola-<nombre>/` con esos tres
# hermanos dentro: ese directorio es lo que se le pasa aquí como RAIZ.
#
# Sin argumentos de la pasada, corre `--estimar` (barato, no mide nada).
# Con `--comprobar`, en vez de la pasada importa el árbol de módulos que
# necesita (núcleo + motores) y dice qué falta dentro de la imagen, SIN
# lanzar nada más — es la comprobación que pide el encargo de 116-C2 antes
# de dar la imagen por buena.
#
# Ejemplos:
#   correr_116c2_en_contenedor.sh                                    # --estimar
#   correr_116c2_en_contenedor.sh --comprobar
#   correr_116c2_en_contenedor.sh -- --solo dresses-sales --salida /scratch/prueba.json
#   correr_116c2_en_contenedor.sh /home/deployer/worktrees/cola-116c2 -- --forzar
#
# Un directorio SIEMPRE disponible para dejar salidas de prueba, sin tocar
# el árbol de trabajo ni /tmp del host: /home/deployer/.agentes/116c2,
# montado dentro como /scratch (rw). La pasada real (sin --salida) escribe
# en su sitio de siempre, benchmarks/fase0/resultado_116c2_tabicl.json,
# remontado aparte en lectura-escritura (ver más abajo): el resto de cada
# repo queda SOLO LECTURA.
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

IMAGEN=medicion-tabicl-116c0-test:2.2.0
if ! docker image inspect "$IMAGEN" >/dev/null 2>&1; then
  echo "AVISO: no existe la imagen $IMAGEN (base de 116-C0 + pytest); se usa " \
       "medicion-tabicl-116c0:2.2.0 sin pytest (no hace falta pytest para la pasada, " \
       "solo para correr aqui dentro la suite de tests, que este guion no hace)." >&2
  IMAGEN=medicion-tabicl-116c0:2.2.0
fi
docker image inspect "$IMAGEN" >/dev/null 2>&1 || {
  echo "no existe NINGUNA de las dos imagenes (medicion-tabicl-116c0-test:2.2.0 / " \
       "medicion-tabicl-116c0:2.2.0). Construirla es benchmarks/tabicl_116c0/preparar.sh " \
       "(UNA vez, con red); este guion NO la construye por su cuenta." >&2
  exit 1
}

CARGA=$(cut -d' ' -f1 /proc/loadavg)
if awk -v c="$CARGA" 'BEGIN{exit !(c>6)}'; then
  echo "carga $CARGA > 6: esperar a que baje antes de lanzar un contenedor (terreno)." >&2
  exit 1
fi
if docker ps --format '{{.Image}}' 2>/dev/null | grep -qiE 'medicion-tabicl|playwright'; then
  echo "ya hay un contenedor de medicion (tabicl o navegador) corriendo: un contenedor a la " \
       "vez (terreno). Esperar a que termine." >&2
  exit 1
fi

PESOS_HOST=/home/deployer/tabicl_116c0_pesos
SNAP_DIR="$PESOS_HOST/hub/models--jingang--TabICL/snapshots"
[ -d "$SNAP_DIR" ] || { echo "no esta $SNAP_DIR: faltan los pesos " \
                             "(benchmarks/tabicl_116c0/preparar.sh)" >&2; exit 1; }
REV=$(ls "$SNAP_DIR")
[ -n "$REV" ] || { echo "no hay ninguna revision en $SNAP_DIR" >&2; exit 1; }

DATOS_HOST=/home/deployer/fase0_openml_datos
[ -d "$DATOS_HOST" ] || { echo "no esta $DATOS_HOST: faltan los ARFF de Fase 0" >&2; exit 1; }

SCRATCH_HOST=/home/deployer/.agentes/116c2
mkdir -p "$SCRATCH_HOST"

[ -d "$RAIZ/matrixAI" ] || { echo "$RAIZ/matrixAI no existe: no hay pasada que lanzar" >&2; exit 1; }
[ -d "$RAIZ/matrixai-engines" ] || { echo "$RAIZ/matrixai-engines no existe: no hay motor " \
                                          "tabicl.v2 que resolver" >&2; exit 1; }

MONTAJES=(
  -v "$RAIZ/matrixAI:/home/deployer/matrixAI:ro"
  -v "$RAIZ/matrixai-engines:/home/deployer/matrixai-engines:ro"
  -v "$RAIZ/matrixAI/benchmarks/fase0:/home/deployer/matrixAI/benchmarks/fase0:rw"
  -v "$PESOS_HOST:/pesos_root:ro"
  -v "$DATOS_HOST:/tmp/fase0_openml_datos:ro"
  -v "$SCRATCH_HOST:/scratch:rw"
)
if [ -d "$RAIZ/matrixaistudio" ]; then
  MONTAJES+=(-v "$RAIZ/matrixaistudio:/home/deployer/matrixaistudio:ro")
else
  echo "AVISO: $RAIZ/matrixaistudio no existe -- se monta sin el tercer repo (esta pasada " \
       "no lo necesita: no toca studio-backend)." >&2
fi

echo "imagen: $IMAGEN"
echo "raiz de los repos: $RAIZ"
echo "pesos: $SNAP_DIR/$REV"
echo "datos: $DATOS_HOST"
echo "carga: $CARGA"

if [ "$COMPROBAR" -eq 1 ]; then
  echo "comprobando bibliotecas del nucleo y de motores dentro de la imagen..."
  # HOME=/tmp: mismo patron que el resto del terreno con --user puesto (sin el,
  # node_modules/pip escriben como root y todo lo de despues falla en silencio).
  exec docker run --rm --network none --memory=2g --memory-swap=2g --cpus=2 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e MATRIXAI_TABICL_PESOS_DIR="/pesos_root/hub/models--jingang--TabICL/snapshots/$REV" \
    "${MONTAJES[@]}" \
    -w /home/deployer/matrixAI/benchmarks/fase0 \
    "$IMAGEN" \
    python3 -c '
import importlib
# ORDEN A PROPOSITO: "pasada_exploratoria_101_c3" es quien inserta la raiz
# del nucleo y de matrixai-engines en sys.path (ver su docstring y
# _RAIZ_DEL_CORE/_RAIZ_DE_ENGINES) -- exactamente lo que hace
# pasada_116c2_tabicl.py al arrancar de verdad. Comprobar "matrixai" ANTES
# de importar cualquier guion de fase0 mediria una ruta que la pasada real
# nunca toma (import matrixai.estudio SUELTO, sin el bootstrap de sys.path)
# y daria un falso "FALTA". Se importa primero lo que hace ese bootstrap,
# igual que la pasada, y LUEGO se comprueba el resto de submodulos sueltos.
MODULOS = [
    "lector_arff", "protocolo", "pasada_exploratoria_101_c3", "pasada_amplia_101_c5",
    "pasada_v2_113", "pasada_116c2_tabicl",
    "matrixai", "matrixai.estudio", "matrixai.estudio.metricas",
    "matrixai.estudio.vocabulario", "matrixai.estudio.validacion",
    "matrixai.estudio.inferencia", "matrixai.training.particion_por_diseno",
    "matrixai.training.preparacion", "matrixai.training.dataset_analysis",
    "matrixai.playground", "matrixai.forward.dense_forward", "matrixai.limits",
    "matrixai_engines", "matrixai_engines.motor", "matrixai_engines.particiones",
    "matrixai_engines.capacidades", "matrixai_engines.errores", "matrixai_engines.textos",
    "matrixai_engines.tipos_de_columna", "matrixai_engines.procedencia",
    "matrixai_engines.subproceso", "matrixai_engines.harness",
    "matrixai_engines.registro_de_motores", "matrixai_engines.motores.tabicl",
    "matrixai_engines.motores.ensamblado", "matrixai_engines.cartera",
    "tabicl", "torch", "numpy", "pandas", "scipy", "sklearn", "threadpoolctl",
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
    print(f"TODAS las {len(MODULOS)} bibliotecas/modulos importan dentro de la imagen.")
'
fi

echo "argumentos de la pasada: ${ARGS[*]}"
docker run --rm --network none --memory=6g --memory-swap=6g --cpus=5 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e MATRIXAI_TABICL_PESOS_DIR="/pesos_root/hub/models--jingang--TabICL/snapshots/$REV" \
  "${MONTAJES[@]}" \
  -w /home/deployer/matrixAI/benchmarks/fase0 \
  "$IMAGEN" \
  python3 pasada_116c2_tabicl.py "${ARGS[@]}"
