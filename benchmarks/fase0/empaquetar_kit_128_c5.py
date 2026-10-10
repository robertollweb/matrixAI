#!/usr/bin/env python3
"""128-C5 — arma el KIT que se lleva al PC de Roberto: la carpeta `kit_128_c5/` (medidor, lanzadores, LEEME, datos) MÁS las
cinco piezas del repo que el kit usa por ruta —`referencia_120_r0.py`, `veredicto_120.py`, `veredicto_120_c3p.py`,
`veredicto_128_c5.py` y `sonda_128_c0.py` (el paso OPCIONAL «TabM por fases en la GPU», `lanzar.* --fases`; importa
`referencia_120_r0.py` por ruta, que ya va)—, COPIADAS tal cual (una sola fuente: no se reescriben), con un `SHA256SUMS` y un .zip.

    python3 empaquetar_kit_128_c5.py --destino /ruta/vacia [--zip kit_128_c5.zip] \
        [--tar-base /ruta/matrixai-studio-128.tar] [--dockerfile-gpu /ruta/Dockerfile.gpu]

Y la CARPETA `imagen/`: el `Dockerfile.gpu` de la integración (por omisión el de `matrixaistudio/bundle-gpu/` junto a este
repo) COPIADO tal cual, y el TAR de la imagen base (`docker save matrixai-studio:128`; `--tar-base`). El lanzador carga el
tar y construye la capa GPU (PyTorch con CUDA) en el PC de Roberto, como el instalador del paquete GPU. Sin `--tar-base` el
kit sale sin el tar y el lanzador pararía diciendo que falta: se avisa al empaquetar.

LA IDENTIDAD DE LA IMAGEN BASE (128, I3 de la auditoría final): `imagen/matrixai-studio-128.ids`, leída DEL TAR (no de este
Docker) y comprobada contra el contenido de sus blobs. Los dos kits del 128 usan la misma etiqueta (`matrixai-studio:128`) y
la misma capa (`matrixai-studio-gpu-128:local`), y el lanzador solo cargaba y construía si la capa NO existía: el kit 2,
seguido al pie de su LEEME, medía en silencio la imagen del kit 1. Ahora el lanzador compara el ID de la base cargada con
estos y, si no es uno de ellos, carga el tar y rehace la capa él solo. Un ID por línea, porque Docker da uno u otro según su
almacén de imágenes: con el de containerd (el de este servidor), el digest del índice OCI (`index.json`); con el clásico,
el de la configuración de la imagen (`manifest.json`, `Config`).
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
KIT = AQUI / "kit_128_c5"
#: `sonda_128_c0.py` (128-R4 en GPU) solo importa la biblioteca estándar y `referencia_120_r0.py` (por ruta, ya aquí): no necesita más.
PIEZAS_DEL_REPO = ("referencia_120_r0.py", "veredicto_120.py", "veredicto_120_c3p.py", "veredicto_128_c5.py", "sonda_128_c0.py")
DOCKERFILE_GPU_POR_OMISION = AQUI.parents[2] / "matrixaistudio" / "bundle-gpu" / "Dockerfile.gpu"
NOMBRE_DEL_TAR = "matrixai-studio-128.tar"      # el que busca el lanzador
#: 128 (I3 de la auditoría final): los IDs de la imagen del tar, uno por línea (los lee el lanzador).
NOMBRE_DE_LOS_IDS = "matrixai-studio-128.ids"
#: La etiqueta que el lanzador espera en el tar (`BASE` en `lanzar.sh`, `$Base` en `lanzar.ps1`).
ETIQUETA_DE_LA_BASE = "matrixai-studio:128"
FICHEROS_DEL_KIT = ("LEEME.md", "lanzar.sh", "lanzar.ps1", "lanzar.bat", "medir_c5.py", "muestreador.py")


def _blob(tar: tarfile.TarFile, digest: str) -> bytes:
    """El blob `sha256:<hex>` del tar, COMPROBADO: su sha256 tiene que ser su nombre (un tar a medias no da un ID)."""
    algoritmo, _, hexa = digest.partition(":")
    if algoritmo != "sha256" or len(hexa) != 64:
        raise SystemExit(f"PARO: digest raro en el tar: {digest!r}")
    contenido = tar.extractfile(f"blobs/sha256/{hexa}").read()
    if hashlib.sha256(contenido).hexdigest() != hexa:
        raise SystemExit(f"PARO: el blob {digest} del tar no es lo que dice su nombre (¿tar corrupto?)")
    return contenido


def identidad_de_la_imagen(tar_base: Path) -> list[str]:
    """Los IDs con que Docker puede llamar a la imagen de `tar_base` (`docker save`), leídos DEL TAR y comprobados.

    - `index.json` (OCI): el digest del manifiesto o índice de la etiqueta; es el `.Id` con el almacén de containerd.
    - `manifest.json` (`docker save`): `Config` es el blob de la configuración; su digest es el `.Id` con el clásico.
    Exige que el tar traiga la etiqueta que el lanzador busca (`ETIQUETA_DE_LA_BASE`). Sin ningún ID, PARA."""
    ids: list[str] = []
    with tarfile.open(tar_base) as tar:
        nombres = set(tar.getnames())
        if "manifest.json" not in nombres:
            raise SystemExit(f"PARO: {tar_base} no es un `docker save` (le falta manifest.json)")
        manifiesto = json.loads(tar.extractfile("manifest.json").read())
        de_la_base = [m for m in manifiesto if ETIQUETA_DE_LA_BASE in (m.get("RepoTags") or [])]
        if not de_la_base:
            raise SystemExit(f"PARO: el tar no trae la etiqueta {ETIQUETA_DE_LA_BASE} (trae "
                             f"{[t for m in manifiesto for t in (m.get('RepoTags') or [])]}): el lanzador no la encontraría")
        if "index.json" in nombres:
            indice = json.loads(tar.extractfile("index.json").read())
            nombre_ref = ETIQUETA_DE_LA_BASE.split(":", 1)[1]
            for m in indice.get("manifests", []):
                anotaciones = m.get("annotations") or {}
                if anotaciones.get("org.opencontainers.image.ref.name") in (nombre_ref, None) or \
                        anotaciones.get("io.containerd.image.name", "").endswith(ETIQUETA_DE_LA_BASE):
                    _blob(tar, m["digest"])
                    ids.append(m["digest"])
        for m in de_la_base:
            config = m["Config"]                  # «blobs/sha256/<hex>» (OCI) o «<hex>.json» (formato viejo)
            hexa = config.rsplit("/", 1)[-1].removesuffix(".json")
            digest = f"sha256:{hexa}"
            if config.startswith("blobs/"):
                _blob(tar, digest)
            elif hashlib.sha256(tar.extractfile(config).read()).hexdigest() != hexa:
                raise SystemExit(f"PARO: la configuración {config} del tar no es lo que dice su nombre")
            ids.append(digest)
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise SystemExit(f"PARO: no sé leer la identidad de la imagen de {tar_base}")
    return ids


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
        ids = identidad_de_la_imagen(Path(tar_base))         # antes de copiar: un tar que no vale no entra
        shutil.copy2(tar_base, destino / "imagen" / NOMBRE_DEL_TAR)
        escritos.append(destino / "imagen" / NOMBRE_DEL_TAR)
        (destino / "imagen" / NOMBRE_DE_LOS_IDS).write_text("".join(f"{i}\n" for i in ids))
        escritos.append(destino / "imagen" / NOMBRE_DE_LOS_IDS)
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
