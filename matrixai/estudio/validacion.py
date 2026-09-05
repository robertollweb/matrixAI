# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Las comprobaciones de forma, fail-closed — 104-C0.

Están aquí y no en cada llamante por lo que ya costó un bloqueante en el 82-C1:
la validación estricta vivía en dos llamantes, cualquier tercero la esquivaba y
las dos ya estaban divergiendo. Un esquema que solo se valida cuando el llamante
se acuerda no valida nada.

Dos detalles que no son de estilo:

* `type(valor) is not int` y no `isinstance`: en Python `isinstance(True, int)`
  es `True`, y `True` no es una semilla ni un número de pliegues.
* Finitud obligatoria en los reales: `NaN` e `Infinity` no son JSON, y el
  canonicalizador JCS del 81 los rechaza — un documento con eso dentro no se
  puede digerir, así que no se puede ni firmar ni comparar.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping

from matrixai.estudio.errores import EsquemaInvalido
from matrixai.pipelines.canonical import jcs_bytes

__all__ = [
    "digest_canonico", "exigir_booleano", "exigir_entero", "exigir_entero_o_nulo",
    "exigir_mapa", "exigir_motivo_bilingue", "exigir_real", "exigir_real_o_nulo",
    "exigir_texto", "exigir_texto_o_nulo", "exigir_tupla_de_textos",
    "solo_estas_claves",
]


def exigir_texto(valor: Any, campo: str, *, opciones: tuple[str, ...] | None = None) -> str:
    if not isinstance(valor, str) or not valor.strip():
        raise EsquemaInvalido("no_es_texto", campo=campo, valor=repr(valor))
    if opciones is not None and valor not in opciones:
        raise EsquemaInvalido("no_es_del_vocabulario", campo=campo,
                              opciones=list(opciones), valor=repr(valor))
    return valor


def exigir_texto_o_nulo(valor: Any, campo: str,
                        *, opciones: tuple[str, ...] | None = None) -> str | None:
    """`None` pasa; la cadena vacía NO.

    No es lo mismo: `None` dice «no consta» —una respuesta legítima, como un uso
    previsto que nadie ha escrito— y `""` finge que consta y está vacío. Media
    verdad tranquilizadora.
    """
    if valor is None:
        return None
    return exigir_texto(valor, campo, opciones=opciones)


def exigir_entero(valor: Any, campo: str, *, minimo: int | None = None) -> int:
    if type(valor) is not int:
        raise EsquemaInvalido("no_es_entero", campo=campo, valor=repr(valor))
    if minimo is not None and valor < minimo:
        raise EsquemaInvalido("menor_que_el_minimo", campo=campo, minimo=minimo,
                              valor=repr(valor))
    return valor


def exigir_entero_o_nulo(valor: Any, campo: str, *, minimo: int | None = None) -> int | None:
    if valor is None:
        return None
    return exigir_entero(valor, campo, minimo=minimo)


def exigir_booleano(valor: Any, campo: str) -> bool:
    if type(valor) is not bool:
        raise EsquemaInvalido("no_es_booleano", campo=campo, valor=repr(valor))
    return valor


def exigir_real(valor: Any, campo: str, *, minimo: float | None = None,
                maximo: float | None = None) -> float:
    # `type(...) not in` y no `isinstance`: `True` pasaría por un `isinstance(_, int)`
    # y `True` no es una probabilidad ni un umbral.
    if type(valor) not in (int, float):
        raise EsquemaInvalido("no_es_numero_finito", campo=campo, valor=repr(valor))
    if not math.isfinite(valor):
        raise EsquemaInvalido("no_es_numero_finito", campo=campo, valor=repr(valor))
    if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
        raise EsquemaInvalido("fuera_de_rango", campo=campo, minimo=minimo,
                              maximo=maximo, valor=repr(valor))
    return float(valor)


def exigir_real_o_nulo(valor: Any, campo: str, *, minimo: float | None = None,
                       maximo: float | None = None) -> float | None:
    """`None` es una respuesta: una métrica indefinida no vale cero."""
    if valor is None:
        return None
    return exigir_real(valor, campo, minimo=minimo, maximo=maximo)


def exigir_tupla_de_textos(valor: Any, campo: str, *, minimo: int = 0,
                           sin_duplicados: bool = True,
                           opciones: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if isinstance(valor, (str, bytes)) or not isinstance(valor, (list, tuple)):
        raise EsquemaInvalido("no_es_lista_de_textos", campo=campo, valor=repr(valor))
    items = tuple(valor)
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise EsquemaInvalido("no_es_lista_de_textos", campo=campo, valor=repr(valor))
        if opciones is not None and item not in opciones:
            raise EsquemaInvalido("no_es_del_vocabulario", campo=campo,
                                  opciones=list(opciones), valor=repr(item))
    if sin_duplicados:
        vistos: set[str] = set()
        for item in items:
            if item in vistos:
                raise EsquemaInvalido("hay_duplicados", campo=campo, valor=repr(item))
            vistos.add(item)
    if len(items) < minimo:
        raise EsquemaInvalido("menor_que_el_minimo", campo=campo, minimo=minimo,
                              valor=f"{len(items)} elementos")
    return items


def exigir_mapa(valor: Any, campo: str) -> dict[str, Any]:
    if not isinstance(valor, Mapping):
        raise EsquemaInvalido("no_es_mapa", campo=campo, valor=repr(valor))
    for clave in valor:
        if not isinstance(clave, str) or not clave.strip():
            raise EsquemaInvalido("no_es_texto", campo=f"{campo}.<clave>",
                                  valor=repr(clave))
    return dict(valor)


def solo_estas_claves(payload: Mapping[str, Any], permitidas: Iterable[str],
                      campo: str) -> None:
    """Una clave que este esquema no conoce se RECHAZA.

    Es lo que hace `reproduce.py` con las versiones de `run_provenance` y por el
    mismo motivo: adivinar qué significa un campo nuevo es justo lo que el
    `schema_version` existe para evitar. Si el campo es de una versión posterior,
    el documento tiene que declararlo con su versión y pasar por la migración.
    """
    conocidas = set(permitidas)
    for clave in payload:
        if clave not in conocidas:
            raise EsquemaInvalido("clave_desconocida", campo=campo, valor=repr(clave))


def exigir_motivo_bilingue(valor: Any, campo: str) -> dict[str, str]:
    """Un motivo tiene que traer `es` y `en`.

    Lo que redacta el core se traduce en el core; un motivo con un solo idioma
    deja media pantalla sin traducir el día que alguien lo enseñe, y eso ya se ha
    visto en este producto.
    """
    from matrixai.estudio.textos import IDIOMAS  # noqa: PLC0415

    mapa = exigir_mapa(valor, campo)
    if set(mapa) != set(IDIOMAS):
        raise EsquemaInvalido("motivo_incompleto", campo=campo,
                              opciones=list(IDIOMAS), valor=sorted(mapa))
    for idioma in IDIOMAS:
        exigir_texto(mapa[idioma], f"{campo}.{idioma}")
    return {idioma: mapa[idioma] for idioma in IDIOMAS}


def digest_canonico(payload: Any) -> str:
    """El sha256 (64 hex) de la forma canónica JCS del documento.

    **Se reutiliza `jcs_bytes` del 81-C1 a propósito**: un segundo
    canonicalizador daría dos digests del mismo contenido, y ese es exactamente
    el fallo que el contrato 81 documenta. Va el sha256 COMPLETO: una huella
    corta es cómoda de leer y no es una prueba.
    """
    return hashlib.sha256(jcs_bytes(payload)).hexdigest()
