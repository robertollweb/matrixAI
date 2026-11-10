# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""LA regla de «un código con ceros a la izquierda» («08001», «007», «0012»). Módulo hoja (sin imports del paquete) para que
`dataset_analysis`, `dataset_project` y `preparacion` la compartan sin duplicarla (decisión A de Roberto, 09-11).

Un valor es un CÓDIGO si empieza por `0`, tiene más de un carácter y TODO lo que sigue son cifras ASCII. El cero es información,
no relleno: `float("08001")` lo acepta igual (8001.0) y convertirlo lo perdería para siempre. Si ALGÚN valor de una columna lo es,
la columna entera es un código: no es numérica (el análisis la dice identificador o categórica; el estudio la deja como texto).

Los bordes, a propósito estrictos para no atrapar columnas numéricas de verdad: `0` (un cero), `0.5`, `0,5` (decimal español),
`0e3`, `-0.3`, `-08` y `+07` (con signo) NO son códigos; `00` sí (dos caracteres, todo cifras).
"""
from __future__ import annotations

from typing import Any, Iterable


def es_un_codigo_con_ceros(valor: Any) -> bool:
    """«08001»/«007»/«00» → True; «0»/«0.5»/«0,5»/«-08»/«8001»/«0e3» → False. Una cadena; lo demás nunca lo es."""
    if not isinstance(valor, str):
        return False
    v = valor.strip()
    return len(v) > 1 and v[0] == "0" and v[1:].isdigit() and v.isascii()


def codigos_con_ceros_de(valores: Iterable[Any]) -> list[str]:
    """Los valores (recortados, sin repetir, en orden de aparición) que son un código con ceros a la izquierda."""
    vistos: dict[str, None] = {}
    for v in valores:
        if es_un_codigo_con_ceros(v):
            vistos.setdefault(v.strip(), None)
    return list(vistos)
