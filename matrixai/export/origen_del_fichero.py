# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El ORIGEN DEL FICHERO en el recibo y en el manifiesto de reproducción — contrato 150-C4.

POR QUÉ EXISTE. Un Excel entra al producto convertido a un CSV canónico (`matrixai.tablas`, 150-C1), y un CSV
`;` con decimal español entra con la coma pasada a punto (150-C2). En los dos casos **lo que se hashea y lo que
el modelo vio es el CSV derivado, no el fichero que la persona subió**. Sin este bloque, el recibo enseñaba la
huella de un CSV que nadie ha visto y no decía de qué libro salió, con qué hoja ni qué se hizo por el camino.

QUÉ ES `origen_del_fichero`. El `origen` que declara la conversión (`TablaConvertida.origen`: formato, nombre,
sha256 del ORIGINAL, hoja, rango, filas excluidas, conversor y declaraciones), tal cual, más —si la entrada
canónica del estudio convirtió algo al entrar— sus `declaraciones_de_entrada`. Para un CSV que solo se
convirtió al entrar (decimal español) se compone un origen con lo que el estudio sabe: la huella del texto tal
como llegó, su tamaño y lo que se hizo con él.

TRES REGLAS (las que el contrato fija y este módulo SOSTIENE, cada una con su prueba):

1. **El hash que se verifica sigue siendo el del CSV derivado.** `artifacts.dataset.sha256` es el del CSV que
   entró al entrenamiento; la huella del original es INFORMATIVA y no entra en ninguna comprobación (R1–R3 del
   verificador): no se puede prometer reproducir desde el Excel con otra versión de la biblioteca. El bloque
   del manifiesto lo dice él mismo (`informativo`, `nota`), y la línea del recibo también.
2. **Lo que no hay no se inventa.** Un CSV sin conversión ni declaración NO lleva `origen_del_fichero`: ni
   vacío, ni con la huella de sí mismo. La clave se OMITE.
3. **Forma cerrada.** Una captura del run que lo lleva es de la versión `1.3`; un núcleo que no la conoce la
   rechaza en vez de publicarlo como «parámetro de generación» (que es lo que hace con una clave que no tiene
   dada de alta en `_RUN_PROVENANCE_CORE_KEYS`).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from matrixai.export.reproduce import (
    ReproduceManifestError,
    _require_full_sha256,
    _require_text,
    canonical_json,
)

__all__ = [
    "CLAVE", "VERSION_DE_LA_CAPTURA_CON_ORIGEN", "TOPE_DE_BYTES_DEL_ORIGEN",
    "normalizar_origen_del_fichero", "numero_de_declaraciones", "origen_de_un_csv",
    "origen_para_el_estudio", "bloque_del_manifiesto", "texto_del_origen", "ROTULO",
]

#: El nombre de la clave: en la captura del run, en el sobre del estudio y en `artifacts.dataset`.
CLAVE = "origen_del_fichero"

#: La versión de la captura del run (`run_provenance.schema_version`) que puede llevarlo. Una captura que no lo lleva
#: sigue siendo `1.2`, con los mismos bytes de siempre: subir la versión de TODAS las capturas habría cambiado el
#: digest de cada paquete que no tiene nada que ver con un Excel.
VERSION_DE_LA_CAPTURA_CON_ORIGEN = "1.3"

#: Lo máximo que pesa el bloque (JSON canónico). Una hoja con miles de columnas renombradas declara miles de
#: cosas; sin tope, el recibo y la captura de un modelo guardado crecerían sin límite por lo que traiga un fichero.
TOPE_DE_BYTES_DEL_ORIGEN = 1_048_576

#: La versión del esquema de `origen` que ESTE núcleo sabe leer (`matrixai.tablas.ESQUEMA_DE_ORIGEN`). No se importa de
#: allí: `matrixai.tablas` no tiene por qué cargarse para escribir un recibo.
_ESQUEMA_QUE_SE_SABE_LEER = 1

_FORMATOS_DE_EXCEL = ("xlsx", "xls", "xlsm")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

#: Cómo se llama la línea en cada idioma (la ficha y el expediente la rotulan igual: los dos leen de aquí).
ROTULO = {"es": "Origen", "en": "Origin"}


def _lista_de_declaraciones(valor: Any, que: str) -> list[dict[str, Any]]:
    if not isinstance(valor, list):
        raise ReproduceManifestError(f"{que} must be a list, got {valor!r}")
    for i, d in enumerate(valor):
        if not isinstance(d, dict):
            raise ReproduceManifestError(f"{que}[{i}] must be an object, got {d!r}")
        _require_text(d.get("tipo"), f"{que}[{i}].tipo")
        cuantas = d.get("cuantas")
        if cuantas is not None and (type(cuantas) is not int or cuantas < 0):
            raise ReproduceManifestError(f"{que}[{i}].cuantas must be a non-negative integer or null, got {cuantas!r}")
    return valor


def normalizar_origen_del_fichero(crudo: Any, *, campo: str = CLAVE) -> dict[str, Any]:
    """Valida el origen y lo devuelve (copia). Lo imposible se corta aquí, con su motivo.

    Se exige lo que hace falta para que la línea del recibo diga la verdad: `formato` y la huella COMPLETA del
    original (`sha256`, 64 hex: la corta `data_…` no es una prueba). Todo lo demás es opcional —un valor ausente
    es una respuesta y no se rellena—, pero lo que venga tiene que tener su forma. Las claves que este módulo no
    conoce (`conversor`, `rango`, `filas_excluidas`…) se CONSERVAN: son del esquema de la conversión, no de aquí.
    """
    if not isinstance(crudo, Mapping):
        raise ReproduceManifestError(f"{campo} must be an object, got {crudo!r}")
    origen = dict(crudo)

    esquema = origen.get("esquema")
    if esquema is not None:
        # «No se interpreta a medias»: un esquema más nuevo que el que este núcleo conoce puede querer decir otra cosa
        # con los mismos nombres.
        if type(esquema) is not int or esquema < 1 or esquema > _ESQUEMA_QUE_SE_SABE_LEER:
            raise ReproduceManifestError(
                f"{campo}.esquema must be an integer between 1 and {_ESQUEMA_QUE_SE_SABE_LEER} (the versions of the origin "
                f"this core can read), got {esquema!r}: an origin it does not know how to read is not interpreted half-way")

    _require_text(origen.get("formato"), f"{campo}.formato")
    if len(origen["formato"]) > 32 or _CONTROL.search(origen["formato"]):
        raise ReproduceManifestError(f"{campo}.formato is not a format name, got {origen['formato']!r}")
    _require_full_sha256(origen.get("sha256"), f"{campo}.sha256")

    for clave in ("nombre", "hoja"):
        valor = origen.get(clave)
        if valor is None:
            continue
        _require_text(valor, f"{campo}.{clave}")
        if len(valor) > 255 or _CONTROL.search(valor):
            raise ReproduceManifestError(
                f"{campo}.{clave} must be a text of at most 255 characters with no control characters, got {valor!r}")
    octetos = origen.get("bytes")
    if octetos is not None and (type(octetos) is not int or octetos < 0):
        raise ReproduceManifestError(f"{campo}.bytes must be a non-negative integer or null, got {octetos!r}")

    for clave in ("declaraciones", "declaraciones_de_entrada"):
        if clave in origen and origen[clave] is not None:
            _lista_de_declaraciones(origen[clave], f"{campo}.{clave}")
        elif clave in origen:
            del origen[clave]

    try:
        pesa = len(canonical_json(origen).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ReproduceManifestError(f"{campo} must be JSON-serialisable: {exc}") from exc
    if pesa > TOPE_DE_BYTES_DEL_ORIGEN:
        raise ReproduceManifestError(
            f"{campo} is {pesa} bytes, over the {TOPE_DE_BYTES_DEL_ORIGEN} that a receipt carries: a file that "
            f"declares that much needs its own report, not a line in the receipt")
    return origen


def numero_de_declaraciones(origen: Mapping[str, Any]) -> int:
    """Cuántas cosas declaró el origen: las de la conversión del libro y las de la entrada canónica."""
    return sum(len(origen.get(c) or []) for c in ("declaraciones", "declaraciones_de_entrada"))


def origen_de_un_csv(texto_recibido: str, declaraciones: Sequence[Mapping[str, Any]], *,
                     nombre: str | None = None) -> dict[str, Any]:
    """El origen de un CSV que la entrada canónica CONVIRTIÓ o sobre el que declaró algo (150-C2).

    `texto_recibido` es el texto TAL COMO LLEGÓ al estudio, antes de normalizarlo: es lo único del original que el
    estudio ve (si el navegador decodificó o quitó un BOM, ese cambio ya estaba hecho), y `huella_de` lo dice.
    """
    octetos = texto_recibido.encode("utf-8")
    origen: dict[str, Any] = {
        "esquema": _ESQUEMA_QUE_SE_SABE_LEER,
        "formato": "csv",
        "sha256": hashlib.sha256(octetos).hexdigest(),
        "bytes": len(octetos),
        "huella_de": "texto_recibido",
        "declaraciones": [dict(d) for d in declaraciones],
    }
    if isinstance(nombre, str) and nombre.strip():
        origen["nombre"] = nombre.strip()[:255]
    return normalizar_origen_del_fichero(origen)


def origen_para_el_estudio(origen_recibido: Any, *, texto_recibido: str,
                           declaraciones_de_entrada: Sequence[Mapping[str, Any]] | None,
                           nombre: str | None = None) -> dict[str, Any] | None:
    """El `origen_del_fichero` de UN estudio, o `None` si no hay nada que decir.

    * Con `origen_recibido` (lo que devolvió la conversión de un libro): ese origen, validado, más las
      `declaraciones_de_entrada` si la entrada canónica también convirtió algo.
    * Sin él y con `declaraciones_de_entrada` (un CSV `;` con decimal español): el origen de ese CSV.
    * Sin ninguno de los dos: `None`. **No se inventa un origen para un CSV que no ha pasado por nada.**
    """
    entrada = [dict(d) for d in (declaraciones_de_entrada or [])]
    if origen_recibido is not None:
        origen = normalizar_origen_del_fichero(origen_recibido)
        if entrada:
            origen["declaraciones_de_entrada"] = _lista_de_declaraciones(entrada, "declaraciones_de_entrada")
            return normalizar_origen_del_fichero(origen)
        return origen
    if entrada:
        return origen_de_un_csv(texto_recibido, entrada, nombre=nombre)
    return None


#: Lo que se dice junto al origen en el manifiesto: la huella del original NO es lo que se verifica.
_NOTA = {
    "es": ("Informativo: lo que se verifica es la huella del CSV derivado (`artifacts.dataset.sha256`), que es lo que "
           "vio el modelo. La huella del original no entra en ninguna comprobación (R1–R3): no se puede prometer "
           "reproducir desde el fichero original con otra versión de la biblioteca que lo convierte."),
    "en": ("Informative: what is verified is the digest of the derived CSV (`artifacts.dataset.sha256`), which is what "
           "the model saw. The digest of the original file takes part in no check (R1–R3): reproducing from the "
           "original file cannot be promised with another version of the library that converts it."),
}


def bloque_del_manifiesto(origen: Mapping[str, Any]) -> dict[str, Any]:
    """Lo que `artifacts.dataset.origen_del_fichero` publica: el origen más su carácter informativo, dicho."""
    return {**origen, "informativo": True, "nota": dict(_NOTA)}


def _pl(n: int, idioma: str) -> str:
    if idioma == "es":
        return f"{n} declaración" if n == 1 else f"{n} declaraciones"
    return f"{n} declaration" if n == 1 else f"{n} declarations"


def texto_del_origen(origen: Mapping[str, Any], idioma: str) -> str:
    """«Excel `ventas.xlsx`, hoja `Datos`, huella del original `<sha256>`, 3 declaraciones» (+ lo que NO es).

    Una sola redacción para la ficha TRIPOD+AI y para el expediente: dos sitios declarando lo mismo acaban
    divergiendo. Lo que el origen no sabe (nombre, hoja) se omite, no se rellena."""
    es = idioma == "es"
    formato = str(origen.get("formato") or "")
    etiqueta = "Excel" if formato in _FORMATOS_DE_EXCEL else formato.upper()
    piezas = [etiqueta + (f" `{origen['nombre']}`" if origen.get("nombre") else "")]
    if origen.get("hoja"):
        piezas.append((f"hoja `{origen['hoja']}`" if es else f"sheet `{origen['hoja']}`"))
    piezas.append((f"huella del original `{origen.get('sha256')}`" if es
                   else f"digest of the original `{origen.get('sha256')}`"))
    piezas.append(_pl(numero_de_declaraciones(origen), "es" if es else "en"))
    cola = (" (informativo: lo que se verifica es la huella del dataset derivado, la de arriba; la del original no "
            "entra en la verificación)" if es else
            " (informative: what is verified is the digest of the derived dataset, the one above; the digest of "
            "the original takes no part in verification)")
    return ", ".join(piezas) + cola
