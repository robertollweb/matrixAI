# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Cómo se hicieron los Parquet sintéticos de `tests/fixtures/tablas/` (150-C8).

A diferencia de los Excel (`generar.py`, solo biblioteca estándar), un Parquet NO se puede escribir sin una biblioteca:
aquí se usa `pyarrow` (el mismo que lee el producto; Apache-2.0), que solo hace falta para REGENERARLOS. Las pruebas leen
estos ficheros con el lector del producto (`matrixai.tablas`), nunca con `pyarrow` ni con `pandas`.

Regenerar (la versión de pyarrow queda en el pie del fichero: `created_by`):

    pip install --target ~/.agentes/150/c8/site pyarrow==26.0.1     # una vez
    PYTHONPATH=~/.agentes/150/c8/site python3 generar_parquet.py       # reescribe los .parquet de esta carpeta

**Determinista con la misma versión de pyarrow**: sin marcas de tiempo, grupos de filas de tamaño fijo, compresión
`snappy`, estadísticas y diccionarios fijos. `test_150_c8_parquet.py` regenera todo y compara byte a byte SOLO si la versión
instalada es la de `VERSION_DE_PYARROW`; con otra comprueba que el contenido leído por el producto es el mismo.

Qué lleva cada uno (todos de pocos KB, viven en git):

* `parquet_tipos.parquet`      una columna de cada tipo que trae la gente de pandas/Spark/BigQuery: enteros (con nulos, de más
                               de 2^53, `uint8`), `float64`/`float32` con `NaN`, `bool` con nulos, `decimal`, `date32`/`date64`,
                               `time64`, texto con diccionario, y `timestamp` en s/ms/us/ns, con y sin zona horaria.
* `parquet_anidados.parquet`   lista, estructura, mapa y binario junto a dos columnas que sí se convierten.
* `parquet_nombres.parquet`    nombres vacío, repetido y con espacios y U+FEFF.
* `parquet_enteros.parquet`    `int64` con nulos y valores por encima de 2^53 (lo que un paso por pandas redondearía).
* `parquet_grupos.parquet`     tres grupos de filas (la lectura por lotes cruza sus bordes).
* `parquet_solo_anidados.parquet`  todas las columnas son listas: no hay nada que convertir.
"""
from __future__ import annotations

import datetime as dt
import decimal
import io
import pathlib
import sys

AQUI = pathlib.Path(__file__).resolve().parent
VERSION_DE_PYARROW = "26.0.1"


def _ts(pa, unidad: str, tz, valores_en_segundos, fraccion_ns: int = 0):
    por_s = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[unidad]
    crudos = [None if v is None else v * por_s + (fraccion_ns * por_s // 10**9) for v in valores_en_segundos]
    return pa.array(crudos, pa.timestamp(unidad, tz=tz))


def _escribir(pa, pq, tabla, **kw) -> bytes:
    buf = io.BytesIO()
    pq.write_table(tabla, buf, compression="snappy", use_dictionary=True, write_statistics=True, **kw)
    return buf.getvalue()


def todos() -> dict[str, bytes]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = 12
    base = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp())      # 2024-01-01 00:00:00 UTC
    horas = [base + 3600 * h for h in range(n)]
    madrid = [int(dt.datetime(2024, 3, 31, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=1))).timestamp()) + 3600 * h
              for h in range(n)]          # cruza el cambio de hora de Madrid (+01:00 → +02:00)
    tipos = pa.table({
        "id": pa.array(range(n), pa.int64()),
        "ts_utc_us": _ts(pa, "us", "UTC", horas),
        "ts_madrid_ns": _ts(pa, "ns", "Europe/Madrid", madrid),
        "ts_naive_ns_con_fraccion": _ts(pa, "ns", None, [base + 86400 * h for h in range(n)], fraccion_ns=123_456_789),
        "ts_naive_ms": _ts(pa, "ms", None, [base + 60 * h for h in range(n)]),
        "ts_naive_s": _ts(pa, "s", None, [base + 60 * h if h != 3 else None for h in range(n)]),
        "dia": pa.array([dt.date(2024, 1, 1 + h) for h in range(n)], pa.date32()),
        "dia_ms": pa.array([dt.date(2024, 2, 1 + h) for h in range(n)], pa.date64()),
        "hora_us": pa.array([dt.time(8, 30, h) for h in range(n)], pa.time64("us")),
        "importe": pa.array([decimal.Decimal("1234.50") + h for h in range(n)], pa.decimal128(12, 2)),
        "cat": pa.array(["a", "b", None, "a"] * 3).dictionary_encode(),
        "entero_con_nulos": pa.array([1, None, 3] * 4, pa.int64()),
        "grande": pa.array([2**62 + h for h in range(n)], pa.int64()),
        "flotante_nan": pa.array([1.5, float("nan"), None] * 4, pa.float64()),
        "flotante_inf": pa.array([0.1, float("inf"), float("-inf"), 3.0] * 3, pa.float64()),
        "simple": pa.array([0.1, 0.25, 3.0, None] * 3, pa.float32()),
        "flag": pa.array([True, False, None] * 4, pa.bool_()),
        "uint8": pa.array([200 + h for h in range(n)], pa.uint8()),
        "texto": pa.array(["año", "ñandú, con coma", 'con "comillas"', "salto\nde línea"] * 3),
    })
    anidados = pa.table({
        "id": pa.array(range(n), pa.int64()),
        "lista": pa.array([[1, 2], [3], None] * 4),
        "estructura": pa.array([{"a": 1}, {"a": 2}, None] * 4),
        "mapa": pa.array([[("k", 1)], None, [("z", 2)]] * 4, pa.map_(pa.string(), pa.int64())),
        "binario": pa.array([b"\x00\xff", b"ab", None] * 4),
        "valor": pa.array([float(h) / 4 for h in range(n)], pa.float64()),
    })
    nombres = pa.table([pa.array(range(4)), pa.array(range(4)), pa.array(range(4)), pa.array(range(4)),
                        pa.array(range(4))],
                       names=["Edad", "Edad", "", "  Peso  ", "﻿altura"])
    enteros = pa.table({"x": pa.array([2**53 + 1, None, 2**63 - 1, -(2**63), 7, None], pa.int64()),
                        "y": pa.array([1, 2, 3, 4, 5, 6], pa.int64())})
    grupos = pa.table({"i": pa.array(range(25), pa.int64()), "v": pa.array([h * 0.5 for h in range(25)])})
    solo_anidados = pa.table({"a": pa.array([[1], [2]]), "b": pa.array([{"q": 1}, {"q": 2}])})
    return {
        "parquet_tipos.parquet": _escribir(pa, pq, tipos),
        "parquet_anidados.parquet": _escribir(pa, pq, anidados),
        "parquet_nombres.parquet": _escribir(pa, pq, nombres),
        "parquet_enteros.parquet": _escribir(pa, pq, enteros),
        "parquet_grupos.parquet": _escribir(pa, pq, grupos, row_group_size=10),
        "parquet_solo_anidados.parquet": _escribir(pa, pq, solo_anidados),
    }


def escribir(destino: pathlib.Path = AQUI) -> list[str]:
    nombres = []
    for nombre, b in todos().items():
        (destino / nombre).write_bytes(b)
        nombres.append(nombre)
    return nombres


if __name__ == "__main__":
    print("\n".join(escribir()))
