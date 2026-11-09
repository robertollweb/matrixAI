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


def _estimada_y_picos(datos: bytes, tmp_path):
    """`(estimada, padre, hijo)`: la estimación del producto para este libro y los picos MEDIDOS
    de `convertir` en un proceso LIMPIO (`VmHWM`: el `ru_maxrss` de un hijo hereda el máximo del
    padre a través de fork+exec), restando el de arranque."""
    import subprocess
    import zipfile
    from matrixai.tablas import xlsx_xml
    from matrixai.tablas.topes import ram_estimada_de_una_hoja
    libro = tmp_path / "grande.xlsx"
    libro.write_bytes(datos)
    zf = zipfile.ZipFile(libro)
    ruta = xlsx_xml.mapa_de_hojas(zf)[0]["ruta"]
    xml = zf.getinfo(ruta).file_size
    g = xlsx_xml.explorar_hoja(zf, ruta, xlsx_xml.estilos_del_libro(zf))
    estimada = ram_estimada_de_una_hoja(g.celdas_del_rango, xml, xlsx_xml.cadenas_compartidas(zf))
    codigo = (
        "import sys\n"
        "def hwm():\n"
        "    for l in open('/proc/self/status'):\n"
        "        if l.startswith('VmHWM:'):\n"
        "            return int(l.split()[1]) * 1024\n"
        f"sys.path[:0] = {[p for p in sys.path if p]!r}\n"
        "from matrixai.tablas import convertir, aislado\n"
        "import python_calamine\n"
        f"b = open({str(libro)!r}, 'rb').read()\n"
        "base = hwm()\n"
        "convertir(b, 'g.xlsx', fila_de_cabecera=1)\n"
        "print(hwm() - base, aislado.ULTIMO_HIJO['hwm'])\n")
    out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-500:]
    padre, hijo = (int(x) for x in out.stdout.strip().splitlines()[-1].split())
    return estimada, padre, hijo


@pytest.mark.parametrize("filas,columnas", [(FILAS, COLUMNAS), (100_000, 10)])
def test_la_estimacion_de_memoria_cubre_padre_mas_hijo_en_dos_tamanos(filas, columnas, libro_grande, tmp_path):
    """I-R3 de la 2.ª re-auditoría: los factores se midieron con calamine EN el proceso, y desde B2
    lee un HIJO: gastan los dos (medido ×2,4–2,7 lo estimado con los factores viejos). La guarda
    tiene que cubrir la SUMA de los dos picos, medidos en un proceso LIMPIO. Y no pasarse de ×2,
    que rechazaría lo que sí cabe. El techo del hijo queda por ENCIMA de su pico."""
    from matrixai.tablas.topes import TOPES_DE_FABRICA, techo_del_lector
    datos = libro_grande if (filas, columnas) == (FILAS, COLUMNAS) else G.grande(filas, columnas)
    estimada, padre, hijo = _estimada_y_picos(datos, tmp_path)
    assert padre + hijo <= estimada <= 2 * (padre + hijo), (estimada >> 20, padre >> 20, hijo >> 20)
    assert hijo < techo_del_lector(estimada, len(datos), TOPES_DE_FABRICA)


@pytest.mark.parametrize("filas,columnas,largo", [(20_000, 30, 100), (50_000, 10, 200)])
def test_la_estimacion_de_memoria_cubre_padre_mas_hijo_con_textos_largos(filas, columnas, largo, tmp_path):
    """I2 de la 3.ª auditoría: en una hoja de TEXTOS LARGOS casi todos distintos el factor de los
    textos compartidos (×1) dejaba la estimación en ~1/3 de lo gastado (20.000×30 de 100
    caracteres: estimada 276 MB, gastado 800 MB; 50.000×10 de 200: 293 frente a 937). Con la
    misma regla que la prueba de arriba: padre + hijo ≤ estimada ≤ 2 × (padre + hijo)."""
    estimada, padre, hijo = _estimada_y_picos(G.grande_de_textos(filas, columnas, largo), tmp_path)
    assert padre + hijo <= estimada <= 2 * (padre + hijo), (estimada >> 20, padre >> 20, hijo >> 20)
