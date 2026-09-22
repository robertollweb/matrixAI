#!/bin/bash
# 116-C0 — UNA vez, con red y con la máquina quieta: la imagen y los pesos.
# Deja escrito lo que ocupa cada cosa (el protocolo pide medirlo en la imagen construida).
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
PESOS="$HOME/tabicl_116c0_pesos"
mkdir -p "$PESOS"
docker build -t medicion-tabicl-116c0:2.2.0 "$AQUI"
docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' medicion-tabicl-116c0:2.2.0 \
  > "$AQUI/capas_de_la_imagen.txt"
docker run --rm --memory=6g --memory-swap=6g --cpus=5 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e HF_HOME=/pesos -v "$PESOS":/pesos medicion-tabicl-116c0:2.2.0 python3 -c "
from tabicl import TabICLClassifier, TabICLRegressor
import numpy as np
X = np.random.default_rng(0).normal(size=(40, 4)); y = (X[:, 0] > 0).astype(int)
TabICLClassifier().fit(X, y); TabICLRegressor().fit(X, X[:, 1])
print('pesos descargados')"
du -sh "$PESOS" | tee "$AQUI/tamano_de_los_pesos.txt"
find "$PESOS" -name "*.ckpt" -exec sha256sum {} \; | tee "$AQUI/sha256_de_los_pesos.txt"
