#!/usr/bin/env python3
"""128-C5 — arma el KIT que se lleva al PC de Roberto: la carpeta `kit_128_c5/` (medidor, lanzadores, LEEME, datos) MÁS las
cuatro piezas del repo que el medidor usa por ruta —`referencia_120_r0.py`, `veredicto_120.py`, `veredicto_120_c3p.py` y
`veredicto_128_c5.py`—, COPIADAS tal cual (una sola fuente: no se reescriben), con un `SHA256SUMS` y un .zip.

    python3 empaquetar_kit_128_c5.py --destino /ruta/vacia [--zip kit_128_c5.zip] \
        [--tar-base /ruta/matrixai-studio-128.tar] [--dockerfile-gpu /ruta/Dockerfile.gpu]

Y la CARPETA `imagen/`: el `Dockerfile.gpu` de la integración (por omisión el de `matrixaistudio/bundle-gpu/` junto a este
repo) COPIADO tal cual, y el TAR de la imagen base (`docker save matrixai-studio:128`; `--tar-base`). El lanzador carga el
tar y construye la capa GPU (PyTorch con CUDA) en el PC de Roberto, como el instalador del paquete GPU. Sin `--tar-base` el
kit sale sin el tar y el lanzador pararía diciendo que falta: se avisa al empaquetar.
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
KIT = AQUI / "kit_128_c5"
PIEZAS_DEL_REPO = ("referencia_120_r0.py", "veredicto_120.py", "veredicto_120_c3p.py", "veredicto_128_c5.py")
DOCKERFILE_GPU_POR_OMISION = AQUI.parents[2] / "matrixaistudio" / "bundle-gpu" / "Dockerfile.gpu"
NOMBRE_DEL_TAR = "matrixai-studio-128.tar"      # el que busca el lanzador
FICHEROS_DEL_KIT = ("LEEME.md", "lanzar.sh", "lanzar.ps1", "lanzar.bat", "medir_c5.py", "muestreador.py")


def armar(destino: Path, *, dockerfile_gpu: Path | None = None, tar_base: Path | None = None) -> list[Path]:
    destino = Path(destino)
    if destino.exists() and any(destino.iterdir()):
        raise SystemExit(f"PARO: {destino} no está vacío")
    destino.mkdir(parents=True, exist_ok=True)
    escritos = []
    for nombre in FICHEROS_DEL_KIT:
        shutil.copy2(KIT / nombre, destino / nombre)
        escritos.append(destino / nombre)
    for nombre in PIEZAS_DEL_REPO:
        shutil.copy2(AQUI / nombre, destino / nombre)
        escritos.append(destino / nombre)
    dockerfile_gpu = Path(dockerfile_gpu) if dockerfile_gpu else DOCKERFILE_GPU_POR_OMISION
    if not dockerfile_gpu.is_file():
        raise SystemExit(f"PARO: no encuentro el Dockerfile.gpu ({dockerfile_gpu}); pásalo con --dockerfile-gpu")
    (destino / "imagen").mkdir()
    shutil.copy2(dockerfile_gpu, destino / "imagen" / "Dockerfile.gpu")
    escritos.append(destino / "imagen" / "Dockerfile.gpu")
    if tar_base:
        shutil.copy2(tar_base, destino / "imagen" / NOMBRE_DEL_TAR)
        escritos.append(destino / "imagen" / NOMBRE_DEL_TAR)
    else:
        print(f"AVISO: sin --tar-base el kit NO lleva imagen/{NOMBRE_DEL_TAR}: el lanzador pararía diciendo que falta.", file=sys.stderr)
    (destino / "datos").mkdir()
    for f in sorted((KIT / "datos").iterdir()):
        shutil.copy2(f, destino / "datos" / f.name)
        escritos.append(destino / "datos" / f.name)
    for f in escritos:
        if f.suffix in (".sh",):
            f.chmod(0o755)
    sumas = "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(destino).as_posix()}\n"
                    for f in sorted(escritos))
    (destino / "SHA256SUMS").write_text(sumas)
    return escritos + [destino / "SHA256SUMS"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", required=True)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--tar-base", default=None)
    ap.add_argument("--dockerfile-gpu", default=None)
    a = ap.parse_args(argv)
    escritos = armar(Path(a.destino), dockerfile_gpu=a.dockerfile_gpu, tar_base=a.tar_base)
    print(f"{len(escritos)} ficheros en {a.destino}")
    if a.zip:
        raiz = Path(a.destino)
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(escritos):
                info = zipfile.ZipInfo(f"kit_128_c5/{f.relative_to(raiz).as_posix()}", date_time=(2026, 10, 8, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o755 if f.suffix == ".sh" else 0o644) << 16
                z.writestr(info, f.read_bytes())
        print("zip:", a.zip, round(Path(a.zip).stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
