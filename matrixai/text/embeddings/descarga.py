# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — descarga verificada de los pesos de un proveedor de embeddings.

Los pesos de terceros NO van en la imagen (102-C5) ni en este repositorio:
se descargan a una caché local del usuario, y cada fichero se verifica
contra el digest que su origen publica ANTES de dejarlo en su sitio.

Dos algoritmos de digest, y no por gusto: Hugging Face publica `sha256`
para los ficheros grandes (los que van por LFS) y, para los pequeños, solo
el `oid` de git — que es el **sha1 del blob**, `sha1(b"blob %d\\0" % n + datos)`.
Verificar solo los grandes dejaría el `tokenizer.json` y el `vocab.txt` sin
comprobar, y el tokenizer es parte del modelo (107, invariante 6): un
`vocab.txt` cambiado produce OTRO vector con el mismo `model.onnx`.

**Por qué esto no usa `matrixai.training.secure_fetch`**, que es el punto
único de salida a la red de los `DataProvider`: `secure_fetch` acumula el
cuerpo ENTERO en memoria (`max_bytes` corta, no vuelca), y aquí los ficheros
son de decenas a cientos de MB — un `model.onnx` de 512 MB en RAM en una
máquina sin holgura es justo lo que no se quiere. Lo que sí se reutiliza son
sus reglas: solo HTTPS, allowlist FIJO de hosts, y cada redirección validada
a mano contra el mismo allowlist antes de seguirla. Esa decisión la sostiene
`test_c107_c1_descarga.py::TestPorQueNoSeUsaSecureFetch`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

#: Hosts a los que este módulo puede pedir. Igual que en `secure_fetch`, es
#: FIJO: ninguna configuración de usuario puede añadir uno.
HOSTS_PERMITIDOS: tuple[str, ...] = ("huggingface.co", "hf.co")

#: Y además, los SUBDOMINIOS de estos dominios. Medido, no supuesto: pedir
#: `huggingface.co/<repo>/resolve/<rev>/<fichero>` redirige a un CDN cuyo
#: nombre depende de la región y del día — el 2026-09-14 respondió
#: `us.aws.cdn.hf.co`, y una lista de nombres exactos habría caducado ahí
#: mismo. Lo que garantiza que el fichero recibido es el bueno NO es el
#: host: es el digest, que se comprueba siempre. El allowlist solo existe
#: para no salir del dominio del origen.
#:
#: La comprobación es por sufijo CON el punto delante a propósito:
#: `evilhf.co` no termina en `.hf.co`. `test_c107_c1_descarga.py::
#: TestAllowlistDeHosts` lo fija.
SUFIJOS_PERMITIDOS: tuple[str, ...] = (".hf.co", ".huggingface.co")


def host_permitido(host: str) -> bool:
    h = (host or "").lower()
    if h in HOSTS_PERMITIDOS:
        return True
    return any(h.endswith(s) for s in SUFIJOS_PERMITIDOS)

MAX_REDIRECCIONES = 8
_TROZO = 1 << 20


class DescargaError(Exception):
    """Fallo de descarga o de verificación — mensaje siempre accionable."""


@dataclass(frozen=True)
class FicheroDelProveedor:
    """Un fichero del proveedor, fijado por su digest y su tamaño.

    `algoritmo` es el que publica el origen: `sha256` (LFS) o `git-blob-sha1`
    (el `oid` de git de un fichero versionado normal).
    """

    ruta: str
    tamano_bytes: int
    digest: str
    algoritmo: str

    def __post_init__(self) -> None:
        if self.algoritmo not in ("sha256", "git-blob-sha1"):
            raise ValueError(
                f"algoritmo de digest desconocido para {self.ruta!r}: {self.algoritmo!r}; "
                "se admiten 'sha256' y 'git-blob-sha1'"
            )
        if self.tamano_bytes <= 0:
            raise ValueError(f"{self.ruta!r}: tamano_bytes debe ser positivo, es {self.tamano_bytes!r}")
        if not self.digest:
            raise ValueError(f"{self.ruta!r}: sin digest — un fichero sin digest no se descarga")


def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(_TROZO), b""):
            h.update(trozo)
    return h.hexdigest()


def git_blob_sha1_de(ruta: Path) -> str:
    """El `oid` que publica git para un blob: `sha1("blob <n>\\0" + datos)`."""
    n = ruta.stat().st_size
    h = hashlib.sha1()
    h.update(b"blob %d\0" % n)
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(_TROZO), b""):
            h.update(trozo)
    return h.hexdigest()


def digest_de(ruta: Path, algoritmo: str) -> str:
    if algoritmo == "sha256":
        return sha256_de(ruta)
    if algoritmo == "git-blob-sha1":
        return git_blob_sha1_de(ruta)
    raise DescargaError(f"algoritmo de digest desconocido: {algoritmo!r}")


def verificar_fichero(ruta: Path, fichero: FicheroDelProveedor) -> None:
    """Lanza `DescargaError` si el fichero de disco no es EXACTAMENTE el
    declarado. Comprueba tamaño y digest: el tamaño da un mensaje útil
    cuando la descarga se cortó a medias, el digest es lo que decide."""
    if not ruta.is_file():
        raise DescargaError(f"falta {ruta}")
    real = ruta.stat().st_size
    if real != fichero.tamano_bytes:
        raise DescargaError(
            f"{fichero.ruta}: {real} bytes en disco, el catálogo declara "
            f"{fichero.tamano_bytes} — descarga incompleta o fichero cambiado en origen"
        )
    obtenido = digest_de(ruta, fichero.algoritmo)
    if obtenido != fichero.digest:
        raise DescargaError(
            f"{fichero.ruta}: {fichero.algoritmo} {obtenido} ≠ {fichero.digest} declarado — "
            "el fichero NO es el que fija el catálogo; no se usa"
        )


def raiz_cache() -> Path:
    """Dónde viven los pesos descargados. Fuera del repositorio y fuera de
    la imagen, siempre: `$MATRIXAI_EMBEDDINGS_HOME` o
    `~/.cache/matrixai/embeddings`."""
    env = os.environ.get("MATRIXAI_EMBEDDINGS_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "matrixai" / "embeddings"


def _validar_url(url: str) -> urllib.parse.ParseResult:
    partes = urllib.parse.urlparse(url)
    if partes.scheme != "https":
        raise DescargaError(f"solo HTTPS; {url!r} usa {partes.scheme!r}")
    host = (partes.hostname or "").lower()
    if not host_permitido(host):
        raise DescargaError(
            f"host no permitido: {host!r} (permitidos: {list(HOSTS_PERMITIDOS)} "
            f"y subdominios de {list(SUFIJOS_PERMITIDOS)})"
        )
    return partes


class _SinRedirecciones(urllib.request.HTTPRedirectHandler):
    """Apaga el seguimiento automático de `urllib`.

    Devolver `None` en `redirect_request` hace que `urllib` NO siga el salto
    y lance el `HTTPError` 30x con sus cabeceras — que es justo lo que
    `_abrir` necesita para validar el destino antes de ir. Sin esto el
    allowlist por salto sería decorativo: `urlopen` ya habría seguido la
    redirección antes de que nadie mirase el host.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _abridor_sin_redirecciones(req: urllib.request.Request):
    return urllib.request.build_opener(_SinRedirecciones()).open(req, timeout=120)


def _abrir(url: str, *, abridor: Callable[[urllib.request.Request], object] | None = None):
    """Sigue redirecciones A MANO, validando cada salto contra el allowlist.

    `urllib` las seguiría solo, y entonces un 302 hacia otro host se
    descargaría sin que nadie lo mirase.
    """
    abrir = abridor or _abridor_sin_redirecciones
    actual = url
    for _ in range(MAX_REDIRECCIONES):
        _validar_url(actual)
        req = urllib.request.Request(actual, headers={"User-Agent": "matrixai-core/107-C1"})
        try:
            resp = abrir(req)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                destino = e.headers.get("Location")
                if not destino:
                    raise DescargaError(f"redirección {e.code} sin Location desde {actual}") from None
                actual = urllib.parse.urljoin(actual, destino)
                continue
            raise DescargaError(f"HTTP {e.code} al pedir {actual}") from None
        except urllib.error.URLError as e:
            raise DescargaError(f"no se pudo pedir {actual}: {e.reason}") from None
        estado = getattr(resp, "status", None) or resp.getcode()
        if estado in (301, 302, 303, 307, 308):
            destino = resp.headers.get("Location")
            resp.close()
            if not destino:
                raise DescargaError(f"redirección {estado} sin Location desde {actual}")
            actual = urllib.parse.urljoin(actual, destino)
            continue
        if estado != 200:
            resp.close()
            raise DescargaError(f"HTTP {estado} al pedir {actual}")
        return resp
    raise DescargaError(f"más de {MAX_REDIRECCIONES} redirecciones desde {url}")


def descargar_fichero(
    url: str,
    destino: Path,
    fichero: FicheroDelProveedor,
    *,
    abridor: Callable[[urllib.request.Request], object] | None = None,
) -> Path:
    """Descarga `url` a `destino` y lo verifica ANTES de dejarlo en su sitio.

    Se escribe a `destino.parcial` y solo se mueve con `os.replace` —atómico—
    si el digest cuadra: así nunca existe en la caché un fichero a medias que
    la siguiente ejecución dé por bueno.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    resp = _abrir(url, abridor=abridor)
    try:
        with open(parcial, "wb") as salida:
            shutil.copyfileobj(resp, salida, _TROZO)
    finally:
        resp.close()
    try:
        verificar_fichero(parcial, fichero)
    except DescargaError:
        parcial.unlink(missing_ok=True)
        raise
    os.replace(parcial, destino)
    return destino


def url_de(repo: str, revision: str, ruta: str) -> str:
    """URL de un fichero en una revisión FIJA del repo — nunca `main`.

    102 invariante 7: la licencia se fija por código y por checkpoint
    concreto; apuntar a `main` es apuntar a un checkpoint que puede cambiar
    mañana sin que nadie lo note.
    """
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise DescargaError(
            f"revision debe ser el commit completo de 40 hex, no {revision!r} "
            "— una rama o un prefijo no fijan el checkpoint"
        )
    return f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(ruta)}"


def estado_local(directorio: Path, ficheros: Iterable[FicheroDelProveedor]) -> dict[str, str]:
    """Qué hay ya descargado y verificado, sin tocar la red.

    Devuelve `{ruta: "ok" | motivo}` — nunca un booleano suelto: cuando algo
    no está bien, quien llama necesita saber si falta o si no cuadra.
    """
    salida: dict[str, str] = {}
    for f in ficheros:
        ruta = directorio / f.ruta
        if not ruta.is_file():
            salida[f.ruta] = "falta"
            continue
        try:
            verificar_fichero(ruta, f)
        except DescargaError as e:
            salida[f.ruta] = str(e)
        else:
            salida[f.ruta] = "ok"
    return salida
