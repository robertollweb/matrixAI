# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C1 — la forma y el tamaño que trae la gente (regla del terreno).

Una hoja de 50.000 × 30 (la forma medida en el inventario). CARA: generar el libro y convertirlo
cuesta unos segundos, así que vive en su propio fichero —la pasada rápida la aplaza a la nocturna—
y lleva su tope. Mide también que el pico de memoria no se sale de la guarda declarada.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import pathlib
import sys
import time

import pytest

from matrixai.tablas import convertir, inspeccionar
from matrixai.tablas.topes import FACTOR_RAM_POR_BYTE_DE_XML

pytest.importorskip("python_calamine", reason="falta python-calamine (extra `tablas`)")

_spec = importlib.util.spec_from_file_location(
    "generar_tablas_150", pathlib.Path(__file__).resolve().parent / "fixtures" / "tablas" / "generar.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["generar_tablas_150"] = G
_spec.loader.exec_module(G)

FILAS, COLUMNAS = 50_000, 30
TOPE_SEGUNDOS = 240   # medido ~6 s con la máquina quieta; la carga de otras sesiones lo multiplica


@pytest.fixture(scope="module")
def libro_grande() -> bytes:
    return G.grande(FILAS, COLUMNAS)


def test_50000_por_30_se_inspecciona_y_se_convierte_completo(libro_grande):
    t0 = time.monotonic()
    r = inspeccionar(libro_grande, "grande.xlsx")
    h = r["hojas"][0]
    assert (h["filas"], h["columnas"], h["fila_de_cabecera_propuesta"]) == (FILAS + 1, COLUMNAS, 1)
    t = convertir(libro_grande, "grande.xlsx", hoja="Datos", fila_de_cabecera=1)
    assert time.monotonic() - t0 < TOPE_SEGUNDOS
    filas = list(csv.reader(io.StringIO(t.csv_text)))
    assert len(filas) == FILAS + 1 and all(len(f) == COLUMNAS for f in filas)
    assert t.origen["declaraciones"] == [] and t.origen["rango"] == "A1:AD50001"
    # determinista también a este tamaño
    assert convertir(libro_grande, "grande.xlsx", hoja="Datos", fila_de_cabecera=1).csv_bytes == t.csv_bytes


def test_el_pico_de_memoria_no_baja_del_factor_que_la_guarda_declara(libro_grande, tmp_path):
    """El factor de la guarda (x3 el XML) es el MÍNIMO medido: si el pico real de una hoja densa
    bajara de ahí, la guarda rechazaría lo que sí cabe; si se disparara muy por encima, dejaría
    pasar lo que no. Se mide en un proceso LIMPIO y con `VmHWM` (el `ru_maxrss` de un hijo hereda el máximo del
    padre a través de fork+exec y diría 0)."""
    import subprocess
    import zipfile
    libro = tmp_path / "grande.xlsx"
    libro.write_bytes(libro_grande)
    xml = sum(i.file_size for i in zipfile.ZipFile(libro).infolist() if "worksheets" in i.filename)
    codigo = (
        "import sys\n"
        "def hwm():\n"
        "    for l in open('/proc/self/status'):\n"
        "        if l.startswith('VmHWM:'):\n"
        "            return int(l.split()[1]) * 1024\n"
        f"sys.path[:0] = {[p for p in sys.path if p]!r}\n"
        "from matrixai.tablas import convertir\n"
        f"b = open({str(libro)!r}, 'rb').read()\n"
        "base = hwm()\n"
        "convertir(b, 'g.xlsx', hoja='Datos', fila_de_cabecera=1)\n"
        "print(hwm() - base)\n")
    out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-500:]
    delta = int(out.stdout.strip().splitlines()[-1])
    assert FACTOR_RAM_POR_BYTE_DE_XML * xml <= delta <= 10 * xml, (delta / xml)
