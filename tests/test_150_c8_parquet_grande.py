# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C8 — forma y TAMAÑO reales de un Parquet (regla del terreno: probar con lo que trae la gente), fuera de la pasada rápida.

100.000 × 30 (floats, enteros y textos con pocas categorías), generado en la prueba con pyarrow y convertido en un proceso LIMPIO
que mide su pico de RSS (`VmHWM`). Comprueba tres cosas: que el CSV tiene todas las filas y columnas, que tarda menos que su tope, y
que la estimación de memoria del núcleo (`memoria_necesaria`, el MEJOR caso medido) no pasa de lo que de verdad se gastó: una
estimación por encima de lo gastado rechazaría ficheros que sí caben.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import subprocess
import sys
import time

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent
HAY_PYARROW = importlib.util.find_spec("pyarrow") is not None
pytestmark = pytest.mark.skipif(not HAY_PYARROW, reason="falta pyarrow (extra `parquet`)")

FILAS, COLUMNAS = 100_000, 30
TOPE_DE_SEGUNDOS = 240.0


def _fichero(ruta: pathlib.Path) -> None:
    import array
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = random.Random(150)
    cols = {}
    for j in range(COLUMNAS):
        k = j % 3
        cols[f"c{j}"] = (pa.array(array.array("d", (rng.random() for _ in range(FILAS)))) if k == 0 else
                         pa.array([rng.randrange(10**9) if rng.random() > 0.05 else None for _ in range(FILAS)], pa.int64()) if k == 1 else
                         pa.array([f"cat{rng.randrange(50)}" for _ in range(FILAS)]))
    pq.write_table(pa.table(cols), ruta, compression="zstd")


def test_100000_por_30_se_convierte_entero_dentro_de_su_tope_y_la_estimacion_de_memoria_no_pasa_de_lo_gastado(tmp_path):
    ruta = tmp_path / "grande.parquet"
    _fichero(ruta)
    codigo = f"""
import json, sys, time
sys.path[:0] = [{str(RAIZ)!r}]
def hwm():
    for l in open('/proc/self/status'):
        if l.startswith('VmHWM:'):
            return int(l.split()[1]) * 1024
import matrixai.tablas as t
from matrixai.tablas import parquet_lector as pl
datos = open({str(ruta)!r}, 'rb').read()
f = pl._Fichero(datos, 'g.parquet', t.TOPES_DE_FABRICA)
estimada = f.memoria_necesaria({COLUMNAS})
del f
base = hwm()
t0 = time.time()
r = t.convertir(datos, 'g.parquet')
print(json.dumps(dict(segundos=time.time() - t0, extra=hwm() - base, estimada=estimada, csv=len(r.csv_bytes),
                      filas=r.origen['filas'], columnas=r.origen['columnas'], lineas=r.csv_bytes.count(b'\\n'))))
"""
    t0 = time.time()
    out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=TOPE_DE_SEGUNDOS * 2)
    assert out.returncode == 0, out.stderr[-1500:]
    m = json.loads(out.stdout.strip().splitlines()[-1])
    assert m["filas"] == FILAS and m["columnas"] == COLUMNAS
    assert m["lineas"] == FILAS + 1, "una línea por fila más la cabecera (ningún texto lleva saltos de línea)"
    assert m["segundos"] < TOPE_DE_SEGUNDOS, m
    assert m["estimada"] <= m["extra"], f"la estimación pasa de lo gastado: rechazaría ficheros que caben: {m}"
    assert m["extra"] <= 8 * m["estimada"], f"la estimación se queda muy corta de lo gastado: {m}"
    assert time.time() - t0 < TOPE_DE_SEGUNDOS * 2
