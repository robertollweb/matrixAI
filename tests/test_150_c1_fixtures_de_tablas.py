# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C1 — los Excel sintéticos de `tests/fixtures/tablas/` son los que su guion produce.

Un fichero binario commiteado cuyo origen nadie puede reproducir es una creencia, no una prueba.
`generar.py` los escribe sin ninguna biblioteca de terceros y de forma determinista; aquí se
regenera todo en una carpeta temporal y se compara byte a byte con lo commiteado.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import zipfile

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "tablas"

_spec = importlib.util.spec_from_file_location("generar_tablas_150", FIXTURES / "generar.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["generar_tablas_150"] = G
_spec.loader.exec_module(G)

#: Lo único que NO sale de `generar.py` (BIFF no se puede escribir sin una biblioteca): copiado de
#: una sonda del inventario, con su origen documentado en la cabecera del guion.
NO_GENERADOS = {"antiguo.xls"}


def test_los_fixtures_commiteados_son_los_que_produce_el_guion(tmp_path):
    esperados = G.todos()
    for nombre, b in esperados.items():
        assert (FIXTURES / nombre).read_bytes() == b, f"{nombre} no coincide con generar.py"
    G.escribir(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(esperados)


def test_no_hay_fixtures_de_mas_ni_sin_origen():
    binarios = {p.name for p in FIXTURES.iterdir()
                if p.suffix in {".xls", ".xlsx", ".xlsm"}}
    assert binarios == set(G.todos()) | NO_GENERADOS


def test_los_fixtures_son_pequenos():
    """Pocos KB cada uno: viven en git. Lo grande se genera dentro de la prueba."""
    for p in FIXTURES.iterdir():
        if p.suffix in {".xls", ".xlsx", ".xlsm"}:
            assert p.stat().st_size < 10_000, p.name


def test_el_xls_antiguo_es_un_ole_de_verdad():
    assert (FIXTURES / "antiguo.xls").read_bytes().startswith(bytes.fromhex("D0CF11E0A1B11AE1"))


def test_los_xlsx_generados_son_zips_validos_y_el_generador_es_determinista():
    for nombre, b in G.todos().items():
        if nombre in {"html_como.xls", "csv_renombrado.xlsx", "vacio.xlsx", "cifrado.xlsx",
                      "truncado.xlsx", "hostil_xls_aborta.xls"}:
            continue
        assert zipfile.ZipFile(__import__("io").BytesIO(b)).testzip() is None, nombre
    assert G.todos() == G.todos()


def test_generar_no_usa_bibliotecas_de_terceros():
    fuente = (FIXTURES / "generar.py").read_text(encoding="utf-8")
    for malo in ("import openpyxl", "from openpyxl", "import xlwt", "import xlrd", "import pandas"):
        assert malo not in fuente
