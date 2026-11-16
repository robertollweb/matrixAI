# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C4 — los pesos de TabICL v2, UNA vez y con red (al construir la imagen), verificados
por su sha256. Los mismos que usó 116-C0 y que espera el adaptador (`motores/tabicl.py`,
`PESOS_TABICL`): si Hugging Face sirviera otros, la imagen NO se construye.

Quedan en un directorio PLANO (`/pesos/<fichero>`), que es como los busca el adaptador; la
medición corre después sin red.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = "jingang/TabICL"
REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
PESOS = {
    "tabicl-classifier-v2-20260212.ckpt":
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    "tabicl-regressor-v2-20260212.ckpt":
        "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
}


def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def verificar(directorio: Path) -> list[str]:
    """Los fallos: fichero ausente o huella que no casa. Vacía = todo en orden."""
    fallos = []
    for nombre, esperado in PESOS.items():
        ruta = directorio / nombre
        if not ruta.is_file():
            fallos.append(f"falta {ruta}")
        elif (calculado := sha256_de(ruta)) != esperado:
            fallos.append(f"{nombre}: sha256 {calculado} en vez de {esperado}")
    return fallos


def main() -> None:
    destino = Path(sys.argv[1] if len(sys.argv) > 1 else "/pesos")
    destino.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 — solo al construir
    for nombre in PESOS:
        hf_hub_download(repo_id=REPO, filename=nombre, revision=REVISION, local_dir=destino)
    fallos = verificar(destino)
    if fallos:
        raise SystemExit("PESOS QUE NO CASAN, la imagen no se construye: " + "; ".join(fallos))
    print(f"pesos verificados en {destino}: {', '.join(PESOS)}")


if __name__ == "__main__":
    main()
