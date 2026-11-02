# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""De un valor de celda al texto del CSV, y el escritor de CSV (150-C1, criterios 7 y 8).

El escritor es NUESTRO y determinista: UTF-8 sin BOM, `\\n`, `csv.QUOTE_MINIMAL`, números con su
representación más corta. Ningún escritor de CSV de una biblioteca de terceros (su salida cambia
entre versiones y el hash que todo verifica es el de estos bytes).

Ningún valor se inventa: lo que no es finito sale VACÍO (jamás `0`: «un valor ausente no es un
cero»), un booleano sale `true`/`false` (los que reconoce el analizador), un entero que el lector
devuelve como `10.0` sale `10`.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import math

__all__ = ["VACIO", "valor_a_texto", "tipo_de_valor", "escribir_csv", "numero_a_texto"]

VACIO = ""


def numero_a_texto(v: int | float) -> str:
    """La representación más corta que vuelve a dar el mismo número; `10.0` → `10`."""
    if isinstance(v, int):
        return str(v)
    if abs(v) < 1e15 and v == int(v):
        return str(int(v))
    return repr(v)


def tipo_de_valor(v) -> str:
    """`vacio`, `numero`, `texto`, `fecha`, `booleano`, `hora` o `duracion`."""
    if v is None or v == "":
        return "vacio"
    if isinstance(v, bool):
        return "booleano"
    if isinstance(v, (int, float)):
        return "numero"
    if isinstance(v, dt.datetime) or isinstance(v, dt.date):
        return "fecha"
    if isinstance(v, dt.timedelta):
        return "duracion"
    if isinstance(v, dt.time):
        return "hora"
    return "texto"


def valor_a_texto(v) -> tuple[str, str | None]:
    """`(texto, suceso)`. `suceso` dice qué se tocó y hay que DECLARAR: `fecha_truncada`
    (se descartaron fracciones de segundo), `no_finito` (NaN/inf → vacío), o `None`."""
    if v is None or v == "":
        return VACIO, None
    if isinstance(v, bool):
        return ("true" if v else "false"), None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return VACIO, "no_finito"
        return numero_a_texto(v), None
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S"), ("fecha_truncada" if v.microsecond else None)
    if isinstance(v, dt.date):
        return v.isoformat(), None
    if isinstance(v, dt.timedelta):
        total = v.days * 86400 + v.seconds
        signo = "-" if total < 0 else ""
        total = abs(total)
        h, resto = divmod(total, 3600)
        m, s = divmod(resto, 60)
        trunc = "fecha_truncada" if v.microseconds else None
        return f"{signo}{h:02d}:{m:02d}:{s:02d}", trunc
    if isinstance(v, dt.time):
        return v.strftime("%H:%M:%S"), ("fecha_truncada" if v.microsecond else None)
    return str(v), None


def escribir_csv(cabecera: list[str], filas) -> bytes:
    """Los bytes del CSV: mismas entradas → mismos bytes."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, dialect="excel", delimiter=",", quotechar='"',
                   quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(cabecera)
    w.writerows(filas)
    return buf.getvalue().encode("utf-8")
