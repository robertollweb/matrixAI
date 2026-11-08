# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C7 — los Excel REALES (de Roberto y públicos), contra su ficha escrita a mano (§6.2 del contrato).

Los libros NO están en el repo: los de Roberto viven en `~/datos_150/`, CTG.xls en `~/.agentes/146/ctg/xls/`, y
los dos públicos (python-calamine 0.9.0, MIT) en `~/.agentes/150/c7/dl_calamine_gh/…/tests/data/`. Todas las
rutas se pueden cambiar con `MATRIXAI_EXCEL_REALES_DIR` (un directorio con los cinco ficheros sueltos). La FICHA
(`tests/fixtures/tablas/reales/FICHAS.json`: sha256, hojas, filas×columnas, cabecera, totales, sumas) se
escribió contra el fichero abierto con openpyxl/xlrd (sonda `~/.agentes/150/c7/oraculo.py`), NO con la salida
del producto; aquí solo se lee con el lector del producto y el módulo `csv`.

`skipif` CON RAZÓN cuando falta el fichero o el lector; con `MATRIXAI_EXIGIR_REALES=1` (lo pone
`scripts/exigir_excel_reales.py`) faltar es un ROJO, no una saltada: una «saltada» silenciosa sería un verde
falso. Lo que el producto hace MAL con un libro real va como `xfail(strict=True)` con su reproducción: cuando se
arregle, la prueba pasa a ROJA y obliga a quitar la marca (no se queda tapado).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import pathlib

import pytest

from matrixai.tablas import TablaRechazada, convertir, inspeccionar

try:  # el extra opcional `tablas`
    import python_calamine  # noqa: F401
    HAY_CALAMINE = True
except ImportError:
    HAY_CALAMINE = False

EXIGIR = os.environ.get("MATRIXAI_EXIGIR_REALES") == "1"
FICHAS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "tablas" / "reales" / "FICHAS.json").read_text(encoding="utf-8"))["libros"]
HOME = pathlib.Path.home()
_CAL = HOME / ".agentes/150/c7/dl_calamine_gh/x/python-calamine-0.9.0/tests/data"
RUTAS = {
    "liga_espanola.xlsx": HOME / "datos_150/liga_espanola.xlsx",
    "lluvia_ENRIQUECIDO.xlsx": HOME / "datos_150/lluvia_ENRIQUECIDO.xlsx",
    "CTG.xls": HOME / ".agentes/146/ctg/xls/CTG.xls",
    "table-multiple.xlsx": _CAL / "table-multiple.xlsx",
    "any_sheets.xlsx": _CAL / "any_sheets.xlsx",
}


def _ruta(nombre: str) -> pathlib.Path:
    d = os.environ.get("MATRIXAI_EXCEL_REALES_DIR")
    return (pathlib.Path(d) / nombre) if d else RUTAS[nombre]


def _libro(nombre: str) -> bytes:
    """Los bytes del libro, o SALTA con razón (o FALLA si se exigen). Comprueba el sha256 de la ficha."""
    if not HAY_CALAMINE:
        msg = "falta python-calamine (extra `tablas`; PYTHONPATH a su rueda y MATRIXAI_EXIGIR_TABLAS=1)"
        (pytest.fail if EXIGIR else pytest.skip)(msg)
    ruta = _ruta(nombre)
    if not ruta.is_file():
        msg = (f"falta {ruta}: los libros reales NO están en el repo (son de Roberto o se descargaron); ver "
               f"el docstring. `scripts/exigir_excel_reales.py` los exige")
        (pytest.fail if EXIGIR else pytest.skip)(msg)
    datos = ruta.read_bytes()
    assert hashlib.sha256(datos).hexdigest() == FICHAS[nombre]["sha256"], (
        f"{nombre} no es el fichero de la ficha (cambió su sha256): la ficha se rehace contra el fichero abierto")
    return datos


def _stats(csv_text: str) -> dict:
    filas = list(csv.reader(io.StringIO(csv_text)))
    n_cols = len(filas[0]) if filas else 0
    no_vacios, suma = 0, 0.0
    for f in filas[1:]:
        for x in f:
            if x != "":
                no_vacios += 1
                try:
                    suma += float(x)
                except ValueError:
                    pass
    return {"filas_csv": len(filas), "columnas": n_cols, "no_vacios": no_vacios, "suma_numerica": suma}


def _conversiones():
    out = []
    for nombre, f in FICHAS.items():
        if "conversion" in f:
            c = dict(f["conversion"]); out.append((nombre, c["hoja"], c))
        for hoja, c in (f.get("conversiones") or {}).items():
            out.append((nombre, hoja, c))
    return out


CONV = _conversiones()
CONV_DECL = [x for x in CONV if x[2].get("declaraciones") is not None]


@pytest.mark.parametrize("nombre,hoja,c", CONV, ids=[f"{n}:{h}" for n, h, _ in CONV])
def test_la_conversion_cuadra_celda_a_celda_con_la_ficha(nombre, hoja, c):
    """Mismas filas, mismas columnas, mismo recuento de no vacíos y misma suma numérica que el oráculo."""
    r = convertir(_libro(nombre), nombre, hoja=hoja, fila_de_cabecera=c["fila_de_cabecera"],
                  filas_a_excluir=tuple(c["filas_a_excluir"]))
    s = _stats(r.csv_text)
    for k in ("filas_csv", "columnas", "no_vacios"):
        assert s[k] == c[k], (nombre, hoja, k, s[k], c[k])
    assert s["suma_numerica"] == pytest.approx(c["suma_numerica"], rel=1e-9, abs=1e-6)
    assert r.origen["sha256"] == FICHAS[nombre]["sha256"]
    assert r.origen["sha256_del_csv"] == hashlib.sha256(r.csv_bytes).hexdigest()


@pytest.mark.parametrize("nombre,hoja,c", CONV_DECL, ids=[f"{n}:{h}" for n, h, _ in CONV_DECL])
def test_las_declaraciones_son_las_que_el_fichero_justifica(nombre, hoja, c):
    """Ni falta ni sobra una declaración (contrastadas una a una con el fichero abierto con el oráculo)."""
    r = convertir(_libro(nombre), nombre, hoja=hoja, fila_de_cabecera=c["fila_de_cabecera"],
                  filas_a_excluir=tuple(c["filas_a_excluir"]))
    assert sorted({d["tipo"] for d in r.origen["declaraciones"]}) == sorted(set(c["declaraciones"]))


def test_ctg_las_filas_excluidas_se_declaran_con_su_numero():
    c = FICHAS["CTG.xls"]["conversiones"]["Data"]
    r = convertir(_libro("CTG.xls"), "CTG.xls", hoja="Data", fila_de_cabecera=2, filas_a_excluir=tuple(c["filas_a_excluir"]))
    assert [e["fila"] for e in r.origen["filas_excluidas"]] == [2130, 2131]


def test_any_sheets_las_ocultas_se_listan_marcadas_y_no_se_proponen():
    v = inspeccionar(_libro("any_sheets.xlsx"), "any_sheets.xlsx")
    estados = {h["nombre"]: h["estado"] for h in v["hojas"]}
    esperado = {k: x["estado"] for k, x in FICHAS["any_sheets.xlsx"]["hojas"].items()}
    assert {k: estados[k] for k in esperado} == esperado
    assert v["hoja_propuesta"] == "Visible"


@pytest.mark.parametrize("nombre", ["liga_espanola.xlsx", "lluvia_ENRIQUECIDO.xlsx", "CTG.xls", "table-multiple.xlsx"])
def test_inspeccionar_da_las_hojas_y_las_formas_de_la_ficha(nombre):
    v = inspeccionar(_libro(nombre), nombre)
    hojas = {h["nombre"]: h for h in v["hojas"]}
    for hoja, f in FICHAS[nombre]["hojas"].items():
        assert hojas[hoja]["filas"] == f["filas"] and hojas[hoja]["columnas"] == f["columnas"], (hoja, hojas[hoja]["rango"])


@pytest.mark.parametrize("nombre,hoja", [
    ("CTG.xls", "Data"), ("CTG.xls", "Description"), ("table-multiple.xlsx", "Sheet1"), ("table-multiple.xlsx", "Sheet3"),
])
def test_la_cabecera_propuesta_coincide_con_la_de_la_ficha(nombre, hoja):
    v = inspeccionar(_libro(nombre), nombre)
    h = {x["nombre"]: x for x in v["hojas"]}[hoja]
    assert h["fila_de_cabecera_propuesta"] == FICHAS[nombre]["cabecera_esperada"][hoja]


def test_ctg_raw_data_propone_la_cabecera_de_la_fila_1():
    v = inspeccionar(_libro("CTG.xls"), "CTG.xls")
    h = {x["nombre"]: x for x in v["hojas"]}["Raw Data"]
    assert h["fila_de_cabecera_propuesta"] == 1


@pytest.mark.parametrize("nombre", ["liga_espanola.xlsx", "lluvia_ENRIQUECIDO.xlsx"])
def test_un_csv_pegado_en_una_columna_se_avisa_o_se_declara(nombre):
    datos = _libro(nombre)
    hoja = next(iter(FICHAS[nombre]["hojas"]))
    avisos = [a["tipo"] for a in inspeccionar(datos, nombre)["hojas"][0]["avisos"]]
    r = convertir(datos, nombre, hoja=hoja, fila_de_cabecera=1)
    decl = [d["tipo"] for d in r.origen["declaraciones"]]
    assert set(FICHAS[nombre]["declaraciones_esperadas"]) & (set(avisos) | set(decl))


@pytest.mark.parametrize("nombre", ["liga_espanola.xlsx", "lluvia_ENRIQUECIDO.xlsx"])
def test_el_csv_pegado_en_una_columna_se_convierte_sin_perder_ni_inventar_una_celda(nombre):
    """Lo que SÍ hace bien con los de Roberto: cada celda sale tal cual (con sus comas, entrecomillada)."""
    datos = _libro(nombre)
    f = FICHAS[nombre]
    hoja = next(iter(f["hojas"]))
    r = convertir(datos, nombre, hoja=hoja, fila_de_cabecera=1)
    filas = list(csv.reader(io.StringIO(r.csv_text)))
    esp = f["forma_que_la_persona_espera"]
    assert len(filas) == esp["filas_de_datos"] + 1 and all(len(x) == 1 for x in filas)
    assert {len(next(csv.reader([x[0]]))) for x in filas} == {esp["columnas"]}


def test_un_libro_vacio_se_rechaza_con_motivo_propio():
    with pytest.raises(TablaRechazada) as e:
        convertir(b"", "vacio.xlsx", hoja="x", fila_de_cabecera=1)
    assert e.value.codigo == "vacio"


# ── 150-C7: los arreglos de lo que destapó la medida (H1-aviso, H2, H3, H4) ───────────────
class _Hoja:
    """Lo mínimo de `_HojaLeida` que mira la propuesta de cabecera."""
    def __init__(self, filas, fila0=1):
        self.filas, self.fila0 = filas, fila0
        self.ancho = max((len(f) for f in filas), default=0)


def test_h2_una_cabecera_con_filas_vacias_debajo_sigue_siendo_la_cabecera():
    from matrixai.tablas.api import _proponer_cabecera
    h = _Hoja([["id", "valor", "peso"], [], ["", "", ""], [1, 2.5, 3], [2, 3.5, 4]])
    assert _proponer_cabecera(h) == 1


def test_h2_pero_una_cabecera_sin_datos_detras_no_se_propone():
    from matrixai.tablas.api import _proponer_cabecera
    assert _proponer_cabecera(_Hoja([["id", "valor", "peso"], [], [], [], [], [], [9]])) is None


@pytest.mark.parametrize("nombre", ["liga_espanola.xlsx", "lluvia_ENRIQUECIDO.xlsx"])
def test_h1_la_vista_previa_dice_que_es_un_csv_pegado_con_su_separador_y_que_hacer(nombre):
    avisos = {a["tipo"]: a for a in inspeccionar(_libro(nombre), nombre)["hojas"][0]["avisos"]}
    a = avisos["columna_unica_con_separadores"]
    assert a["detalle"]["separador"] == "coma" and a["detalle"]["campos"] >= 14
    assert "comas" in a["texto"]["es"] and "Texto en columnas" in a["texto"]["es"] and "CSV" in a["texto"]["es"]
    assert "commas" in a["texto"]["en"] and "Text to Columns" in a["texto"]["en"]
    assert "comas" not in a["texto"]["en"] and "Datos" not in a["texto"]["en"]


@pytest.mark.parametrize("nombre", ["CTG.xls", "table-multiple.xlsx"])
def test_h1_un_libro_de_varias_columnas_no_lleva_ese_aviso(nombre):
    for h in inspeccionar(_libro(nombre), nombre)["hojas"]:
        assert "columna_unica_con_separadores" not in [a["tipo"] for a in h["avisos"]]


def test_h1_una_lista_de_apellido_coma_nombre_no_es_un_csv_pegado():
    from matrixai.tablas.api import _columna_unica_con_separadores
    assert _columna_unica_con_separadores(_Hoja([["Perez, Ana"]] * 50)) is None
    assert _columna_unica_con_separadores(_Hoja([[1.5]] * 50)) is None
    assert _columna_unica_con_separadores(_Hoja([["a;b;c"]] * 50)) == ("punto_y_coma", 3)
    assert _columna_unica_con_separadores(_Hoja([["a\tb\tc"]] * 50)) == ("tabulador", 3)
    assert _columna_unica_con_separadores(_Hoja([['"x,y",b,c']] * 50)) == ("coma", 3)   # la coma entrecomillada no cuenta


def test_h3_el_aviso_de_xls_en_ingles_no_lleva_elementos_en_espanol():
    from matrixai.tablas.api import _NO_COMPROBABLE_EN_XLS
    from matrixai.tablas.declaraciones import _NO_COMPROBABLE_EN_XLS_EN, declarar
    assert set(_NO_COMPROBABLE_EN_XLS) == set(_NO_COMPROBABLE_EN_XLS_EN)   # un elemento nuevo sin traducir lo caza
    d = declarar("no_comprobable_en_xls", que=_NO_COMPROBABLE_EN_XLS)
    assert d["detalle"]["que"] == _NO_COMPROBABLE_EN_XLS                   # la clave estable sigue en español
    for en in _NO_COMPROBABLE_EN_XLS_EN.values():
        assert en in d["texto"]["en"]
    for es in _NO_COMPROBABLE_EN_XLS:
        assert es not in d["texto"]["en"]


def _password_xls() -> bytes:
    ruta = _CAL / "password.xls"
    if not ruta.is_file():
        (pytest.fail if EXIGIR else pytest.skip)(f"falta {ruta}")
    return ruta.read_bytes()


def test_h4_un_xls_con_contrasena_se_dice_protegido_y_no_danado():
    if not HAY_CALAMINE:
        (pytest.fail if EXIGIR else pytest.skip)("falta python-calamine")
    for fn in (lambda d: inspeccionar(d, "password.xls"),
               lambda d: convertir(d, "password.xls", hoja="Sheet1", fila_de_cabecera=1)):
        with pytest.raises(TablaRechazada) as e:
            fn(_password_xls())
        assert e.value.codigo == "protegido"
        assert "contraseña" in e.value.motivo_es and "password" in e.value.motivo_en
        assert "dañado" not in e.value.motivo_es and "damaged" not in e.value.motivo_en


def test_h4_un_xls_sin_contrasena_no_se_toma_por_protegido():
    from matrixai.tablas.formato import xls_con_contrasena
    antiguo = pathlib.Path(__file__).parent / "fixtures" / "tablas" / "antiguo.xls"
    assert not xls_con_contrasena(antiguo.read_bytes())
    assert xls_con_contrasena(_password_xls())
