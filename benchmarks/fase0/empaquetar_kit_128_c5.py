#!/usr/bin/env python3
"""128-C5 — arma el KIT que se lleva al PC de Roberto: la carpeta `kit_128_c5/` (medidor, lanzadores, LEEME, datos) MÁS las
cuatro piezas del repo que el medidor usa por ruta —`referencia_120_r0.py`, `veredicto_120.py`, `veredicto_120_c3p.py` y
`veredicto_128_c5.py`—, COPIADAS tal cual (una sola fuente: no se reescriben), con un `SHA256SUMS` y un .zip.

    python3 empaquetar_kit_128_c5.py --destino /ruta/vacia [--zip kit_128_c5.zip]
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
FICHEROS_DEL_KIT = ("LEEME.md", "lanzar.sh", "lanzar.ps1", "lanzar.bat", "medir_c5.py", "muestreador.py")


def armar(destino: Path) -> list[Path]:
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
    a = ap.parse_args(argv)
    escritos = armar(Path(a.destino))
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
