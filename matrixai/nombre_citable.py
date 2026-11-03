# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""La REGLA de los textos que el recibo CITA (el nombre de un fichero, el de una hoja) — 150-C4, remate.

Una sola función dice qué caracteres no puede llevar un texto que la ficha del paquete cita, y la usan los DOS
lados: el validador del origen (`matrixai.export.origen_del_fichero`), que RECHAZA, y la conversión de un libro
(`matrixai.tablas`), que los QUITA y lo declara. Si cada lado tuviera su lista (era el defecto: la conversión
devolvía un origen que `estudio/iniciar` rechazaba entero, con un tabulador o un U+202A en el nombre), dos sitios
declararían lo mismo y acabarían divergiendo.

Es una hoja suelta, sin importar nada: ni `matrixai.export` (arrastra el exportador entero) ni `matrixai.tablas`.

Qué no cabe:
* controles de C0 (`\\x00`–`\\x1f`), `\\x7f` y de C1 (`\\x80`–`\\x9f`, incluido U+0085, «siguiente línea»);
* los separadores de línea y de párrafo (U+2028, U+2029);
* los controles de DIRECCIÓN, que reordenan lo que se lee sin cambiar lo que se guarda: las marcas U+200E, U+200F y
  U+061C, las incrustaciones y anulaciones U+202A–U+202E y los aislamientos U+2066–U+2069.
"""
from __future__ import annotations

import re

__all__ = ["TOPE_DE_CARACTERES", "CONTROL", "DIRECCION", "saneado"]

#: Lo máximo que mide un nombre citado.
TOPE_DE_CARACTERES = 255

CONTROL = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029]")
DIRECCION = re.compile("[\u200e\u200f\u061c\u202a-\u202e\u2066-\u2069]")
_NO_CABE = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029\u200e\u200f\u061c\u202a-\u202e\u2066-\u2069]")


def saneado(texto: object) -> tuple[str | None, int]:
    """`(texto sin lo que no cabe, cuántos caracteres se quitaron)`; `None` si no queda nada o no era un texto.

    Quita, recorta los espacios de los extremos y corta a `TOPE_DE_CARACTERES`. Lo que no es texto, o queda vacío,
    es una AUSENCIA (`None`), no un `""`: un valor ausente es una respuesta, y la ficha lo omite."""
    if not isinstance(texto, str):
        return None, 0
    limpio, quitados = _NO_CABE.subn("", texto)
    limpio = limpio.strip()[:TOPE_DE_CARACTERES].strip()
    return (limpio or None), quitados
