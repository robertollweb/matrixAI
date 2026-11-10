# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150, reparación de lo que encontró la CONDUCCIÓN en el navegador (núcleo): una prueba con nombre por arreglo.

* I1: ningún motivo de rechazo mete español en el texto inglés (el detalle que el motivo cita también va en el idioma);
* I3: con la cabecera de dos filas, el núcleo da los avisos que valen para ELLA (la pantalla no los compone);
* m5/m6: la predicción de la fracción de segundo no afirma una pérdida, y el aviso INT96 no cita una cifra de una zona.
"""
from __future__ import annotations

import os
import pathlib
import re

import pytest

from matrixai.tablas import TablaRechazada, convertir, inspeccionar
from matrixai.tablas import aislado
from matrixai.tablas import parquet_lector as pl

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "tablas"
_ESPANOL = re.compile(r"\b(el|la|los|las|del|con|que|no|fichero|pie|lecturas|marca|acaba|declara)\b", re.I)


def _rechazo(f, *a, **k) -> TablaRechazada:
    with pytest.raises(TablaRechazada) as e:
        f(*a, **k)
    return e.value


def test_i1_un_parquet_cortado_dice_su_detalle_en_ingles_en_el_motivo_ingles(monkeypatch):
    monkeypatch.setattr(pl, "_importar_pyarrow", lambda: (_ for _ in ()).throw(AssertionError("sin pyarrow")))
    d = (FIXTURES / "parquet_tipos.parquet").read_bytes()
    e = _rechazo(convertir, d[:-1], "c.parquet")
    assert "PAR1" in e.motivo_en and "cut off" in e.motivo_en
    assert not _ESPANOL.search(e.motivo_en.replace("«", " ")), e.motivo_en
    assert "cortado" in e.motivo_es and "cut off" not in e.motivo_es


def test_i1_el_detalle_en_espanol_de_todo_motivo_que_lo_cita_tiene_su_version_inglesa():
    """Barrido del código: cada `rechazar(..., tipo="<frase en español>")` lleva su `tipo_en` (las clases de excepción, no)."""
    raiz = pathlib.Path(pl.__file__).parent
    faltan = []
    for f in raiz.glob("*.py"):
        s = f.read_text(encoding="utf-8")
        for m in re.finditer(r"rechazar\([^()]*?tipo=f?\"([^\"]*)\"([^()]*(?:\([^()]*\)[^()]*)*)\)", s, re.S):
            if re.search(r"[a-záéíóúñ]+ [a-záéíóúñ]+", m.group(1)) and "tipo_en=" not in m.group(2):
                faltan.append((f.name, m.group(1)))
    assert faltan == []


def _requiere_calamine():
    if not aislado.calamine_disponible():
        (pytest.fail if os.environ.get("MATRIXAI_EXIGIR_TABLAS") == "1" else pytest.skip)("falta python-calamine")


def test_i3_con_la_cabecera_de_dos_filas_el_nucleo_da_los_avisos_que_valen_para_ella():
    _requiere_calamine()
    v = inspeccionar((FIXTURES / "cabecera_dos_filas.xlsx").read_bytes(), "x.xlsx")
    h = v["hojas"][0]
    una = [a["tipo"] for a in h["avisos"]]
    assert "nombres_repetidos" in una and una.count("columna_mixta") == 4   # con UNA fila de cabecera: lo de siempre
    dos = h["avisos_con_cabecera_de_dos_filas"]
    assert [a["tipo"] for a in dos] == []                                    # compuesta, no se repite nada ni mezcla tipos
    # y es lo mismo que luego hace convertir: ningún nombre renombrado
    t = convertir((FIXTURES / "cabecera_dos_filas.xlsx").read_bytes(), "x.xlsx", hoja="Ventas", fila_de_cabecera=1, filas_de_cabecera=2)
    assert not [d for d in t.origen["declaraciones"] if d["tipo"].startswith("nombres_")]


def test_m5_la_prediccion_de_la_fraccion_no_afirma_una_perdida_que_quiza_no_ocurre():
    from matrixai.tablas.declaraciones import declarar
    d = declarar("parquet_conversion_prevista", columna="ts", que="unidad_a_segundos", unidad="us")
    assert "se descartará" in d["texto"]["es"] and d["texto"]["es"].count("si algún valor") == 1
    assert "if any value" in d["texto"]["en"]


def test_m6_el_aviso_int96_no_cita_una_cifra_de_horas_de_una_zona():
    from matrixai.tablas.declaraciones import declarar
    d = declarar("parquet_int96_en_utc", columna="t")
    assert "1 o 2" not in d["texto"]["es"] and "1 or 2" not in d["texto"]["en"]
    assert "hora local" in d["texto"]["es"] and "local time" in d["texto"]["en"]
