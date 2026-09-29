#!/bin/bash
# 107-C3.0, condición (3) — UNA vez, con red y con la máquina quieta: construye la imagen
# del respondedor y deja escritas las versiones MEDIDAS (pip freeze DENTRO de la imagen
# construida, no las de este Dockerfile a mano) en `versiones_de_la_imagen.txt`, junto a
# este guion -- mismo patrón que `benchmarks/tabicl_116c0/preparar.sh`.
#
# Este guion NO lo corre el agente que preparó esto (26-09: "no instales nada en el host",
# máquina compartida con una release en curso): lo corre Roberto, cuando la máquina esté
# quieta para no competir con la release.
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
IMAGEN=medicion-respondedor-107c30:1

docker build -t "$IMAGEN" "$AQUI"

echo "comprobando que las bibliotecas importan dentro de la imagen..."
docker run --rm --network none "$IMAGEN" python3 -c '
import numpy, pandas, sklearn, lightgbm, tokenizers, onnxruntime, onnxruntime_genai
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lightgbm.__version__)
print("tokenizers", tokenizers.__version__)
print("onnxruntime", onnxruntime.__version__)
print("onnxruntime_genai", onnxruntime_genai.__version__)
'

docker run --rm --network none "$IMAGEN" python3 -m pip freeze | sort > "$AQUI/versiones_de_la_imagen.txt"

echo
echo "imagen construida: $IMAGEN"
echo "versiones escritas en $AQUI/versiones_de_la_imagen.txt"
echo
echo "Los PESOS (Qwen y el tokenizador del embedding de C1) NO van en la imagen: se montan"
echo "en tiempo de ejecución -- ver correr_respondedor_en_contenedor.sh."
