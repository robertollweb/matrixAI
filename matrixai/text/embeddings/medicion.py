# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — lo que MIDE el catálogo. Ningún número de aquí se teclea.

Tres bloques, cada uno con su método escrito al lado del número que produce:

1. **Fijación** (`fijar_candidato`): lee del origen el tamaño, el digest y la
   licencia de cada fichero, en la revisión exacta del candidato. La licencia
   se LEE de los metadatos del repositorio y se traduce a SPDX por una tabla
   explícita; una licencia que no esté en la tabla es un error, nunca un
   supuesto — una afirmación legal equivocada es peor que ninguna.
2. **Ejecución** (`matrixai.text.embeddings.proveedor`): dimensión y vectores.
3. **Cronometraje** (`cronometrar`): tiempo por 1.000 textos, con repeticiones,
   calentamiento y el reloj puesto SOLO alrededor de lo que se quiere medir.

`onnxruntime` y `numpy` NO se importan arriba: el núcleo se instala con
`dependencies = []` (102, invariante 1) y este módulo tiene que poder
importarse para leer el catálogo en una máquina que no los tenga.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from matrixai.text.embeddings.catalogo import Candidato
from matrixai.text.embeddings.descarga import (
    DescargaError,
    FicheroDelProveedor,
    descargar_fichero,
    url_de,
)

API = "https://huggingface.co/api/models"

#: Los identificadores que usa Hugging Face en su ficha NO son SPDX
#: (`apache-2.0` no es `Apache-2.0`). La traducción es una tabla explícita y
#: cerrada: si aparece una licencia que no está aquí, se para. Adivinar el
#: SPDX de una licencia desconocida es exactamente el fallo que este catálogo
#: existe para no cometer.
LICENCIAS_SPDX: dict[str, str] = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc0-1.0": "CC0-1.0",
    "gpl-3.0": "GPL-3.0-only",
    "agpl-3.0": "AGPL-3.0-only",
}

#: Licencias que NO pueden entrar en el producto, con el motivo. Un proveedor
#: con licencia incompatible no entra por bueno que sea.
INCOMPATIBLES: dict[str, str] = {
    "CC-BY-NC-4.0": "prohíbe el uso comercial",
    "GPL-3.0-only": "copyleft fuerte: contaminaría la distribución del producto",
    "CC-BY-SA-4.0": "share-alike: obliga a licenciar lo derivado con los mismos términos",
}


class MedicionError(Exception):
    """Algo que se iba a medir no se pudo medir. Nunca se sustituye por un
    valor por defecto: un valor ausente no es un cero."""


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_json(url: str, *, transporte: Callable[[str], bytes] | None = None) -> Any:
    if transporte is not None:
        return json.loads(transporte(url).decode("utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "matrixai-core/107-C1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise MedicionError(f"no se pudo leer {url}: {e}") from None


def spdx_de(licencia_del_origen: str | None, *, repo: str) -> str:
    """Traduce el identificador de licencia del origen a SPDX, o para.

    `None` no es «sin licencia»: es «el origen no la declara», y eso descalifica
    al candidato hasta que alguien lea la ficha y lo diga.
    """
    if licencia_del_origen is None:
        raise MedicionError(
            f"{repo}: su ficha no declara licencia. Un modelo sin licencia declarada "
            "no entra en el catálogo — no se supone ninguna."
        )
    clave = str(licencia_del_origen).strip().lower()
    if clave not in LICENCIAS_SPDX:
        raise MedicionError(
            f"{repo}: licencia {licencia_del_origen!r} sin equivalencia SPDX en la tabla. "
            "Añádela a LICENCIAS_SPDX a conciencia; no se adivina."
        )
    return LICENCIAS_SPDX[clave]


def fijar_candidato(c: Candidato, *, transporte: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """Lee del origen todo lo fijable del candidato, en SU revisión.

    Devuelve el bloque que va al artefacto: licencia SPDX con su fuente y la
    fecha en que se leyó, y cada fichero con su tamaño y su digest.
    """
    ficha = _get_json(f"{API}/{c.repo}/revision/{c.revision}", transporte=transporte)
    if ficha.get("sha") != c.revision:
        raise MedicionError(
            f"{c.id}: se pidió la revisión {c.revision} y el origen devolvió "
            f"{ficha.get('sha')!r}"
        )
    if ficha.get("gated"):
        raise MedicionError(
            f"{c.id}: el origen exige aceptar términos o una cuenta "
            f"(gated={ficha.get('gated')!r}). No se acepta nada en nombre de nadie."
        )
    licencia_origen = (ficha.get("cardData") or {}).get("license")
    spdx = spdx_de(licencia_origen, repo=c.repo)

    arbol = _get_json(f"{API}/{c.repo}/tree/{c.revision}?recursive=1", transporte=transporte)
    por_ruta = {f["path"]: f for f in arbol if f.get("type") == "file"}
    ficheros: list[FicheroDelProveedor] = []
    for ruta in c.ficheros:
        if ruta not in por_ruta:
            raise MedicionError(f"{c.id}: el origen no tiene {ruta!r} en {c.revision}")
        f = por_ruta[ruta]
        lfs = f.get("lfs") or {}
        if lfs.get("oid"):
            digest, algoritmo = lfs["oid"], "sha256"
        elif f.get("oid"):
            digest, algoritmo = f["oid"], "git-blob-sha1"
        else:
            raise MedicionError(f"{c.id}: el origen no publica digest de {ruta!r}")
        ficheros.append(
            FicheroDelProveedor(
                ruta=ruta,
                tamano_bytes=int(f["size"]),
                digest=digest,
                algoritmo=algoritmo,
            )
        )
    del_modelo = next(f for f in ficheros if f.ruta == c.modelo)
    return {
        "id": c.id,
        "familia": c.familia,
        "repo": c.repo,
        "revision": c.revision,
        "licencia": {
            "spdx": spdx,
            "tal_como_lo_declara_el_origen": licencia_origen,
            "fuente": f"{API}/{c.repo}/revision/{c.revision}",
            "leida_el": _ahora(),
            "compatible": spdx not in INCOMPATIBLES,
            "motivo_si_no": INCOMPATIBLES.get(spdx, ""),
        },
        "pesos_bytes": del_modelo.tamano_bytes,
        "descarga_total_bytes": sum(f.tamano_bytes for f in ficheros),
        "ficheros": [asdict(f) for f in ficheros],
    }


def ficheros_fijados(bloque: dict[str, Any]) -> list[FicheroDelProveedor]:
    return [FicheroDelProveedor(**f) for f in bloque["ficheros"]]


def descargar_candidato(c: Candidato, bloque: dict[str, Any], destino: Path) -> dict[str, str]:
    """Trae los ficheros que falten y verifica TODOS, incluidos los que ya
    estaban: un fichero de la caché puede haberse tocado desde la última vez."""
    if bloque["revision"] != c.revision:
        raise MedicionError(
            f"{c.id}: el bloque fijado es de la revisión {bloque['revision']}, "
            f"el candidato pide {c.revision} — no se mezclan"
        )
    resultado: dict[str, str] = {}
    for f in ficheros_fijados(bloque):
        ruta = destino / f.ruta
        if ruta.is_file():
            try:
                from matrixai.text.embeddings.descarga import verificar_fichero

                verificar_fichero(ruta, f)
                resultado[f.ruta] = "ya estaba, verificado"
                continue
            except DescargaError:
                ruta.unlink()
        descargar_fichero(url_de(c.repo, c.revision, f.ruta), ruta, f)
        resultado[f.ruta] = "descargado y verificado"
    return resultado


# ===========================================================================
# 3. Cronometraje y cobertura de idioma
# ===========================================================================


def cronometrar(proveedor: Any, textos: list[str], *, repeticiones: int = 5, lote: int = 32) -> dict[str, Any]:
    """Tiempo de codificar `textos`, con el reloj SOLO alrededor de codificar.

    Ni la carga del modelo ni el arranque del proceso entran: se miden aparte
    (`carga_s`), porque un cronómetro que mide el arranque da una tabla
    preciosa y falsa. Hay una pasada de calentamiento que se tira — la
    primera llamada de onnxruntime reserva memoria y elige núcleos, y contarla
    mezcla dos cosas distintas.

    Se devuelve la MEDIANA de las repeticiones y también el mínimo y el
    máximo: sin la dispersión, un número solo no dice si la máquina estaba
    tranquila.
    """
    import statistics
    import time

    if repeticiones < 1:
        raise MedicionError("repeticiones debe ser al menos 1")
    proveedor.codificar(textos[: min(len(textos), 64)], lote=lote)  # calentamiento, se tira
    tiempos: list[float] = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        proveedor.codificar(textos, lote=lote)
        tiempos.append(time.perf_counter() - t0)
    mediana = statistics.median(tiempos)
    return {
        "n_textos": len(textos),
        "repeticiones": repeticiones,
        "lote": lote,
        "segundos_mediana": round(mediana, 4),
        "segundos_min": round(min(tiempos), 4),
        "segundos_max": round(max(tiempos), 4),
        "segundos_por_1000_textos": round(mediana * 1000.0 / len(textos), 4),
        "textos_por_segundo": round(len(textos) / mediana, 1),
    }


def control_de_linealidad(proveedor: Any, textos: list[str], *, tamanos: tuple[int, ...] = (250, 500, 1000)) -> dict[str, Any]:
    """Comprueba el INSTRUMENTO antes de fiarse del número.

    Si el cronómetro estuviera midiendo el arranque (la carga del modelo, la
    reserva de memoria de onnxruntime) en vez del trabajo, el tiempo apenas
    cambiaría al doblar el número de textos. Se mide con varios tamaños y se
    mira el segundos-por-1.000 de cada uno: deben parecerse. Un instrumento
    que no pasa esto invalida la fila entera de la tabla, no la matiza.
    """
    filas = []
    for n in tamanos:
        if n > len(textos):
            continue
        filas.append({"n": n, **cronometrar(proveedor, textos[:n], repeticiones=3)})
    por_mil = [f["segundos_por_1000_textos"] for f in filas]
    dispersión = (max(por_mil) - min(por_mil)) / max(por_mil) if por_mil else 1.0
    return {
        "tamanos": filas,
        "dispersion_relativa": round(dispersión, 4),
        "lineal": dispersión < 0.35,
        "que_significa": "el segundos-por-1.000 se mantiene al cambiar el número de "
        "textos: el reloj mide el trabajo, no el arranque",
    }


def medir_tokenizacion(proveedor: Any, textos: list[str]) -> dict[str, Any]:
    """Cobertura del TOKENIZADOR sobre un idioma, medida, no citada.

    Un modelo que no conoce un idioma no falla: lo trocea en pedacitos. Estos
    dos números lo enseñan sin necesidad de etiquetas — cuántos tokens gasta
    por palabra, y cuántas palabras acaban en `[UNK]`.
    """
    tok = proveedor.tokenizador
    con = proveedor.candidato.tokens_especiales
    n_tokens = 0
    n_palabras = 0
    n_unk = 0
    n_truncados = 0
    for t in textos:
        ids = tok.codificar(t, con_especiales=con)
        utiles = len(ids.ids) - (2 if con else 0)
        n_tokens += utiles
        n_palabras += len(t.split())
        n_unk += sum(1 for x in ids.tokens if x == tok.unk_token)
        if len(ids.ids) >= tok.max_longitud:
            n_truncados += 1
    return {
        "n_textos": len(textos),
        "tokens_por_palabra": round(n_tokens / max(n_palabras, 1), 3),
        "porcentaje_unk": round(100.0 * n_unk / max(n_tokens, 1), 4),
        "textos_truncados": n_truncados,
    }


def recuperacion_translingue(vectores_a: Any, vectores_b: Any) -> dict[str, Any]:
    """¿Cada frase encuentra su traducción? P@1 sobre un corpus paralelo.

    Mide si los dos idiomas viven en el MISMO espacio. Un modelo monolingüe
    da un número cerca del azar aunque produzca vectores perfectamente
    válidos para cada idioma por separado — por eso esta medida sola no
    basta para decir «no cubre es», y va acompañada de la coherencia dentro
    de cada idioma.
    """
    import numpy as np

    a = np.asarray(vectores_a, dtype=np.float32)
    b = np.asarray(vectores_b, dtype=np.float32)
    if a.shape != b.shape:
        raise MedicionError(f"los dos lados del corpus paralelo no tienen la misma forma: {a.shape} vs {b.shape}")
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    sim = a @ b.T
    n = sim.shape[0]
    aciertos_ab = int((sim.argmax(axis=1) == np.arange(n)).sum())
    aciertos_ba = int((sim.argmax(axis=0) == np.arange(n)).sum())
    return {
        "n_pares": n,
        "p_at_1_a_b": round(aciertos_ab / n, 4),
        "p_at_1_b_a": round(aciertos_ba / n, 4),
        "azar": round(1.0 / n, 6),
    }


def coherencia_por_documento(vectores: Any, grupos: list[str]) -> dict[str, Any]:
    """AUC de «dos frases del mismo artículo se parecen más que dos de
    artículos distintos», DENTRO de un idioma.

    Es la medida que de verdad dice si el modelo entiende ese idioma, porque
    no depende de que los idiomas estén alineados entre sí. La verdad de
    terreno no la pone nadie a mano: sale de la URL del artículo del que
    procede cada frase en el corpus.
    """
    import numpy as np

    v = np.asarray(vectores, dtype=np.float32)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    sim = v @ v.T
    g = np.asarray(grupos)
    n = len(g)
    iu = np.triu_indices(n, k=1)
    puntuaciones = sim[iu]
    mismo = (g[iu[0]] == g[iu[1]])
    pos = puntuaciones[mismo]
    neg = puntuaciones[~mismo]
    if len(pos) == 0 or len(neg) == 0:
        raise MedicionError("hacen falta pares del mismo artículo y de artículos distintos")
    orden = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    rangos = np.empty(len(orden), dtype=np.float64)
    rangos[orden] = np.arange(1, len(orden) + 1)
    suma_pos = rangos[: len(pos)].sum()
    auc = (suma_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return {
        "auc": round(float(auc), 4),
        "pares_mismo_articulo": int(len(pos)),
        "pares_distinto_articulo": int(len(neg)),
        "azar": 0.5,
    }
