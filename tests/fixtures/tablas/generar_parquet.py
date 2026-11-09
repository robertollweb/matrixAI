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

Añadidos por la reparación de la auditoría de 150-C8 (los seis de arriba NO cambian ni un byte):

* `parquet_int96.parquet`      timestamps INT96 (lo que escriben Spark/Hive/Impala): los centinelas 9999-12-31 y 0001-01-01
                               (fuera de 1677-2262, lo que cabe en ns), una fracción de 500 ns dentro de ese rango y otra de
                               500 ns FUERA de él (puesta a mano en los bytes del valor: pyarrow no sabe escribirla).
* `parquet_horas.parquet`      `time32` en s y ms (TIME_MILLIS, el `time-millis` de Avro) y `time64` en us y ns, con
                               fracciones contadas a mano: 0, 2, 1 y 3.
* `parquet_nombres_chocan.parquet`  « Año » junto a «Año» (chocan solo al quitar los espacios) y el índice de filas que pandas
                               guarda como columna (`__index_level_0__`, con sus metadatos `pandas` escritos a mano).
"""
from __future__ import annotations

import datetime as dt
import decimal
import io
import json
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


#: Los bytes de un INT96: 8 de nanosegundos del día y 4 del día juliano (little endian). El 2440588 es el 1970-01-01.
def _int96(dia: dt.date, nanos_del_dia: int) -> bytes:
    return nanos_del_dia.to_bytes(8, "little") + (2_440_588 + (dia - dt.date(1970, 1, 1)).days).to_bytes(4, "little")


def _con_int96(pa, pq) -> bytes:
    us = pa.timestamp("us")
    tabla = pa.table({
        "alta": pa.array([dt.datetime(2020, 3, 1), dt.datetime(2021, 6, 15, 8, 30), dt.datetime(2022, 1, 1), None], us),
        "baja": pa.array([dt.datetime(9999, 12, 31), dt.datetime(1, 1, 1), dt.datetime(2262, 4, 11), None], us),
        # 2020-01-01 00:00:00.000000500: dentro del rango de ns; la fracción de 500 ns es MENOR que un microsegundo.
        "con_ns": pa.array([1_577_836_800_000_000_500, 1_577_836_801_000_000_000, None, 0], pa.timestamp("ns")),
        # 3000-01-01 00:00:00 se escribe sin fracción y luego se le ponen 500 ns en sus bytes (fuera del rango de ns).
        "lejos": pa.array([dt.datetime(3000, 1, 1), dt.datetime(3000, 1, 2), None, dt.datetime(1, 1, 1, 0, 0, 1)], us),
        "importe": pa.array([10.5, 20.0, None, 1.0], pa.float64()),
    })
    buf = io.BytesIO()
    # Sin diccionario ni compresión: el valor que se toca está tal cual en los bytes (y solo una vez: se comprueba).
    pq.write_table(tabla, buf, compression="none", use_dictionary=False, write_statistics=False,
                   use_deprecated_int96_timestamps=True)
    b = buf.getvalue()
    viejo, nuevo = _int96(dt.date(3000, 1, 1), 0), _int96(dt.date(3000, 1, 1), 500)
    assert b.count(viejo) == 1, "el valor INT96 a tocar no está una sola vez"
    return b.replace(viejo, nuevo)


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
    horas = pa.table({
        "hora_s": pa.array([0, 3_600, 86_399, None], pa.time32("s")),                               # 0 con fracción
        "hora_ms": pa.array([1_500, 3_600_000, None, 86_399_999], pa.time32("ms")),                 # 2
        "hora_us": pa.array([30_600_000_001, 30_601_000_000, None, 0], pa.time64("us")),            # 1
        "hora_ns": pa.array([1, 3_600_000_000_000, 86_399_999_999_999, 500], pa.time64("ns")),      # 3
    })
    chocan = pa.table([pa.array([1, 2, 3]), pa.array([4, 5, 6]), pa.array([7, 9, 12])],
                      names=["Año", " Año ", "__index_level_0__"])
    # Lo que escribe `DataFrame.to_parquet` con un índice sin nombre que no es un RangeIndex (filas filtradas).
    def _col(nombre, campo):
        return {"name": nombre, "field_name": campo, "pandas_type": "int64", "numpy_type": "int64", "metadata": None}
    chocan = chocan.replace_schema_metadata({b"pandas": json.dumps({
        "index_columns": ["__index_level_0__"], "column_indexes": [],
        "columns": [_col("Año", "Año"), _col(" Año ", " Año "), _col(None, "__index_level_0__")],
        "creator": {"library": "pyarrow", "version": VERSION_DE_PYARROW}, "pandas_version": "3.0.5"}).encode("ascii")})
    return {
        "parquet_tipos.parquet": _escribir(pa, pq, tipos),
        "parquet_anidados.parquet": _escribir(pa, pq, anidados),
        "parquet_nombres.parquet": _escribir(pa, pq, nombres),
        "parquet_enteros.parquet": _escribir(pa, pq, enteros),
        "parquet_grupos.parquet": _escribir(pa, pq, grupos, row_group_size=10),
        "parquet_solo_anidados.parquet": _escribir(pa, pq, solo_anidados),
        "parquet_int96.parquet": _con_int96(pa, pq),
        "parquet_horas.parquet": _escribir(pa, pq, horas),
        "parquet_nombres_chocan.parquet": _escribir(pa, pq, chocan),
    }


def escribir(destino: pathlib.Path = AQUI) -> list[str]:
    nombres = []
    for nombre, b in todos().items():
        (destino / nombre).write_bytes(b)
        nombres.append(nombre)
    return nombres


if __name__ == "__main__":
    print("\n".join(escribir()))
