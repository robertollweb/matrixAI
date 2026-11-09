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
import datetime as _dt
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
_PUB = HOME / ".agentes/150/c7-publicos/datos"
RUTAS = {
    "liga_espanola.xlsx": HOME / "datos_150/liga_espanola.xlsx",
    "lluvia_ENRIQUECIDO.xlsx": HOME / "datos_150/lluvia_ENRIQUECIDO.xlsx",
    "CTG.xls": HOME / ".agentes/146/ctg/xls/CTG.xls",
    "table-multiple.xlsx": _CAL / "table-multiple.xlsx",
    "any_sheets.xlsx": _CAL / "any_sheets.xlsx",
    # Libros de oficina PÚBLICOS (150-C7, 09-11): URL y licencia de cada uno en su ficha («origen»).
    "ine_ipc_50902.xlsx": _PUB / "ine_ipc/t.xlsx",
    "ine_poblacion_56934.xlsx": _PUB / "ine_pob/t.xlsx",
    "uk_homeoffice_hosb0812.xls": _PUB / "uk_hosb/t.xls",
    "uk_dluhc_livetable208.xls": _PUB / "uk_live208/t.xls",
    "madrid_gastos_2026.xlsx": _PUB / "madrid_gastos/t.xlsx",
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


# ═══ 150-C7 (decidido por Roberto, 08-11): OFRECER partir un CSV pegado en una columna, con vista previa ═══════════════════
# El núcleo: `inspeccionar` enseña cómo quedaría (`partida_ofrecida`) SIN partir; `convertir(partir_columna_unica=",")` parte con un
# lector CSV de verdad y lo DECLARA (es/en). Sin esa opción, todo como antes.
from test_150_c1_auditoria import _hoja as _hoja_xml, _libro as _libro_xml  # noqa: E402


def _libro_de_una_columna(textos) -> bytes:
    """Un libro de UNA columna (A) con un texto por celda (inlineStr)."""
    from xml.sax.saxutils import escape
    filas = "".join(f'<row r="{i}"><c r="A{i}" t="inlineStr"><is><t xml:space="preserve">{escape(t)}</t></is></c></row>'
                    for i, t in enumerate(textos, 1))
    return _libro_xml(_hoja_xml(filas))


def _esperado_con_csv(ruta_libro_textos, sep=","):
    ancho = max(len(f) for f in ruta_libro_textos)
    return [f + [""] * (ancho - len(f)) for f in ruta_libro_textos]


@pytest.mark.parametrize("nombre", ["liga_espanola.xlsx", "lluvia_ENRIQUECIDO.xlsx"])
def test_partir_c7_los_libros_de_roberto_quedan_con_las_columnas_de_su_ficha_y_celda_a_celda_como_el_modulo_csv(nombre):
    datos = _libro(nombre)
    f = FICHAS[nombre]
    hoja = next(iter(f["hojas"]))
    esp = f["forma_que_la_persona_espera"]
    # el texto de la columna A de la conversión SIN partir es el oráculo (la hoja ya salió celda a celda en las pruebas de arriba)
    sin = list(csv.reader(io.StringIO(convertir(datos, nombre, hoja=hoja, fila_de_cabecera=1).csv_text)))
    textos = [x[0] for x in sin]
    oraculo = _esperado_con_csv([next(csv.reader([t])) for t in textos])
    r = convertir(datos, nombre, hoja=hoja, fila_de_cabecera=1, partir_columna_unica=",")
    filas = list(csv.reader(io.StringIO(r.csv_text)))
    assert len(filas[0]) == esp["columnas"] and len(filas) == esp["filas_de_datos"] + 1
    assert filas == oraculo
    assert r.origen["fila_de_cabecera"] == esp["cabecera"]


@pytest.mark.parametrize("nombre", ["liga_espanola.xlsx", "lluvia_ENRIQUECIDO.xlsx"])
def test_partir_c7_se_declara_en_los_dos_idiomas_y_el_origen_ata_el_csv_partido(nombre):
    datos = _libro(nombre)
    hoja = next(iter(FICHAS[nombre]["hojas"]))
    r = convertir(datos, nombre, hoja=hoja, fila_de_cabecera=1, partir_columna_unica=",")
    d = [x for x in r.origen["declaraciones"] if x["tipo"] == "columna_unica_partida"]
    assert len(d) == 1 and d[0]["detalle"]["separador"] == "coma"
    assert d[0]["detalle"]["campos"] == FICHAS[nombre]["forma_que_la_persona_espera"]["columnas"]
    assert "partido por «,» desde una columna" in d[0]["texto"]["es"] and "split by “,” from a single column" in d[0]["texto"]["en"]
    assert "partido" not in d[0]["texto"]["en"] and "split" not in d[0]["texto"]["es"]
    assert r.origen["sha256_del_csv"] == hashlib.sha256(r.csv_bytes).hexdigest()


@pytest.mark.parametrize("nombre", ["liga_espanola.xlsx", "lluvia_ENRIQUECIDO.xlsx"])
def test_partir_c7_la_vista_previa_enseña_como_quedaria_y_no_parte_nada(nombre):
    datos = _libro(nombre)
    hoja = inspeccionar(datos, nombre)["hojas"][0]
    p = hoja["partida_ofrecida"]
    esp = FICHAS[nombre]["forma_que_la_persona_espera"]
    assert p["separador"] == "coma" and p["columnas"] == esp["columnas"]
    assert p["primeras"][0]["fila"] == 1 and len(p["primeras"][0]["celdas"]) == p["columnas_mostradas"] <= 12
    assert hoja["columnas"] == 1 and all(len(f["celdas"]) == 1 for f in hoja["primeras"])   # la hoja sigue siendo UNA columna
    sin = convertir(datos, nombre, hoja=hoja["nombre"], fila_de_cabecera=1)
    assert "columna_unica_partida" not in [x["tipo"] for x in sin.origen["declaraciones"]]   # sin aceptar, todo como antes
    assert len(next(csv.reader(io.StringIO(sin.csv_text)))) == 1


@pytest.mark.skipif(not HAY_CALAMINE, reason="falta python-calamine")
def test_partir_c7_una_coma_entre_comillas_no_parte_el_campo_y_las_comillas_dobladas_se_leen():
    textos = ["id,nombre,nota", '1,"Perez, Ana",7', '2,"dice ""hola, mundo""",8', '3,"",9']
    r = convertir(_libro_de_una_columna(textos), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=",")
    filas = list(csv.reader(io.StringIO(r.csv_text)))
    assert filas == [["id", "nombre", "nota"], ["1", "Perez, Ana", "7"], ["2", 'dice "hola, mundo"', "8"], ["3", "", "9"]]


@pytest.mark.skipif(not HAY_CALAMINE, reason="falta python-calamine")
def test_partir_c7_un_bom_al_principio_no_se_cuela_en_el_nombre_de_la_primera_columna():
    r = convertir(_libro_de_una_columna(["﻿id,x,y", "1,2,3", "4,5,6"]), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=",")
    assert next(csv.reader(io.StringIO(r.csv_text)))[0] == "id"


@pytest.mark.skipif(not HAY_CALAMINE, reason="falta python-calamine")
def test_partir_c7_punto_y_coma_y_tabulador_con_su_separador():
    for sep in (";", "\t"):
        r = convertir(_libro_de_una_columna([f"a{sep}b{sep}c", f"1{sep}2{sep}3", f"4{sep}5{sep}6"]), "x.xlsx",
                      fila_de_cabecera=1, partir_columna_unica=sep)
        assert list(csv.reader(io.StringIO(r.csv_text))) == [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]]


@pytest.mark.skipif(not HAY_CALAMINE, reason="falta python-calamine")
def test_partir_c7_un_separador_que_la_columna_no_tiene_o_invalido_se_rechaza_no_se_inventa():
    libro = _libro_de_una_columna(["a,b,c", "1,2,3", "4,5,6"])
    with pytest.raises(TablaRechazada):
        convertir(libro, "x.xlsx", fila_de_cabecera=1, partir_columna_unica="|")
    with pytest.raises(TablaRechazada):   # la columna no lleva «;»: nada que partir
        convertir(libro, "x.xlsx", fila_de_cabecera=1, partir_columna_unica=";")


def test_partir_c7_el_ejemplo_de_roberto_con_otro_separador_no_cambia_el_aviso_de_la_vista():
    # la regla de partir es UNA (`_partir_filas`): la vista previa y convertir la comparten
    import inspect
    from matrixai.tablas import api
    assert inspect.getsource(api._vista_de_la_partida).count("_partir_filas(") == 1
    assert inspect.getsource(api.convertir).count("_partir_filas(") == 1
    assert "split(" not in inspect.getsource(api._partir_filas)


def test_el_aviso_del_csv_pegado_ofrece_partirlo_y_no_dice_que_no_se_puede():
    """08-11 (conducción del C7): el aviso decía «No se ha partido. Sube el CSV…» cuando la vista previa ya ofrece partirlo ahí mismo."""
    from matrixai.tablas.declaraciones import declarar
    d = declarar("columna_unica_con_separadores", campos=4, separador=",")
    es, en = d["texto"]["es"], d["texto"]["en"]
    assert "puedes partirlo aquí" in es and "No se ha partido." not in es
    assert "you can split it here" in en and "«" not in en and "»" not in en


# ── 150-C7 (libros de oficina públicos): la cabecera con una BANDA de segundo nivel debajo ───────────────
def test_la_banda_de_segundo_nivel_bajo_la_cabecera_no_descarta_la_cabecera():
    """«Table 2b» del Home Office: cabecera en la fila 4, debajo una banda combinada («Number of detections», UNA celda con
    contenido) y luego los datos. Se proponía la fila 6, el PRIMER DATO, como cabecera."""
    from matrixai.tablas.api import _proponer_cabecera
    h = _Hoja([["Table 2b"], [], ["England and Wales"], ["Grupo", "Total", "Cargos", "Avisos"],
               ["", "Number of detections", "", ""], ["Robo", 5, 3, 1], ["Hurto", 7, 4, 2]])
    assert _proponer_cabecera(h) == 4


def test_una_cabecera_seguida_solo_de_bandas_de_una_celda_sigue_sin_proponerse():
    from matrixai.tablas.api import _proponer_cabecera
    assert _proponer_cabecera(_Hoja([["a", "b", "c"], ["banda"], ["otra"], ["y otra"], ["mas"], [1, 2, 3]])) is None


@pytest.mark.parametrize("nombre,hoja,esperada", [
    ("uk_homeoffice_hosb0812.xls", "Table 2b", 4), ("uk_homeoffice_hosb0812.xls", "Table 2a", 4),
    ("uk_homeoffice_hosb0812.xls", "Table 4", 6), ("ine_poblacion_56934.xlsx", "tabla-56934", 7),
    ("ine_ipc_50902.xlsx", "tabla-50902", 8), ("uk_dluhc_livetable208.xls", "LiveTable208", 5),
    ("madrid_gastos_2026.xlsx", "Ppto Gastos 2026 V40", 1),
])
def test_publicos_la_cabecera_propuesta_es_la_de_la_ficha(nombre, hoja, esperada):
    h = {x["nombre"]: x for x in inspeccionar(_libro(nombre), nombre)["hojas"]}[hoja]
    assert h["fila_de_cabecera_propuesta"] == esperada == FICHAS[nombre]["cabecera_esperada"][hoja]


def test_publicos_la_fila_total_de_la_tabla_2a_se_ofrece_como_candidata_a_excluir():
    h = {x["nombre"]: x for x in inspeccionar(_libro("uk_homeoffice_hosb0812.xls"), "x.xls")["hojas"]}["Table 2a"]
    assert [t["fila"] for t in h["filas_candidatas_a_total"]] == [17]


def test_publicos_un_xls_con_contenido_xlsx_se_lee_por_su_contenido():
    v = inspeccionar(_libro("uk_dluhc_livetable208.xls"), "LiveTable208.xls")
    assert v["formato"] == "xlsx" and v["hoja_propuesta"] == "LiveTable208"


def test_publicos_los_codigos_de_texto_con_ceros_a_la_izquierda_no_pierden_el_cero():
    r = convertir(_libro("madrid_gastos_2026.xlsx"), "g.xlsx", hoja="Ppto Gastos 2026 V40", fila_de_cabecera=1)
    filas = list(csv.reader(io.StringIO(r.csv_text)))
    assert filas[1][0] == "001" and filas[1][2] == "100"


# ═══ Reparación de la auditoría del 150-C7 (09-11): I1, I2 y los menores ═══════════════════════════════════════════════════
# Informe y sondas en ~/.agentes/150/auditoria-c7/; notas de la reparación en ~/.agentes/150/reparacion-c7/.
from test_150_c1_tablas import G as _G  # noqa: E402

from matrixai.training.data import normalize_csv_text, normalize_csv_text_declarando  # noqa: E402

_N = None


def _requiere_calamine():
    if not HAY_CALAMINE:
        (pytest.fail if EXIGIR or os.environ.get("MATRIXAI_EXIGIR_TABLAS") == "1" else pytest.skip)("falta python-calamine")


def _decimales(declaraciones):
    """Las declaraciones del decimal (las del C2), sin el texto: tipo, columna, cuántas y detalle."""
    return [(d["tipo"], d["columna"], d["cuantas"], d["detalle"]) for d in declaraciones
            if d["tipo"] in ("decimal_coma_convertido", "decimal_coma_no_convertido")]


_CSV_PYC = {
    "un_decimal": ["fecha;precio;unidades;compra", "2024-01-01;18,72;2;no", "2024-01-02;3,50;1;si", "2024-01-03;73,58;4;no",
                   "2024-01-04;12;3;si"],
    "dos_decimales_y_miles": ["fecha;precio;peso;compra", "2024-01-01;1.234,5;2,25;no", "2024-01-02;13,75;0,5;si",
                              "2024-01-03;9,5;1,75;no"],
    "ambigua_mezcla_porcentaje_y_texto": ["id;miles;mezcla;pct;cod", "1;1.234;1,5;12,5 %;0,5", "2;2.345;2.5;3,5 %;n/d",
                                          "3;3.456;3,5;1,0 %;0,7"],
    "comillas_y_coma_dentro": ["nombre;importe;nota", '"Perez, Ana";18,72;"dice ""hola"""', "Luis;3,5;x", "Eva;7;y"],
}


@pytest.mark.parametrize("caso", list(_CSV_PYC))
def test_rep_c7_i1_un_csv_punto_y_coma_partido_llega_igual_que_su_csv_con_el_decimal_convertido_y_declarado(caso):
    """I1: partir por «;» un CSV pegado en la columna A da EL MISMO texto que `normalize_csv_text` sobre su `.csv`, con las MISMAS
    declaraciones del decimal (la regla del C2: por columna, todo o nada; lo ambiguo, lo mezclado y el % se dejan y se dicen)."""
    _requiere_calamine()
    lineas = _CSV_PYC[caso]
    texto = "\n".join(lineas) + "\n"
    r = convertir(_libro_de_una_columna(lineas), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=";")
    canon, decl = normalize_csv_text_declarando(texto)
    assert r.csv_text == canon == normalize_csv_text(texto)
    assert _decimales(r.origen["declaraciones"]) == _decimales(decl) != []
    for d in r.origen["declaraciones"]:
        if d["tipo"].startswith("decimal_coma"):
            assert d["texto"]["es"] and d["texto"]["en"]


def test_rep_c7_i1_el_analisis_del_partido_ve_el_precio_como_numero_igual_que_el_del_csv():
    """I1 por el análisis del núcleo (el que usa `analyze-dataset`): antes, el partido daba `precio: identifier`."""
    _requiere_calamine()
    from matrixai.training.dataset_analysis import analyze_dataset_csv
    lineas = _CSV_PYC["un_decimal"]
    r = convertir(_libro_de_una_columna(lineas), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=";")
    del_partido = {k: v["type"] for k, v in analyze_dataset_csv(r.csv_text)["columns"].items()}
    del_csv = {k: v["type"] for k, v in analyze_dataset_csv("\n".join(lineas) + "\n")["columns"].items()}
    assert del_partido == del_csv and del_partido["precio"] == "number"
    assert next(csv.reader(io.StringIO(r.csv_text.splitlines()[1]))) == ["2024-01-01", "18.72", "2", "no"]
    # y es un PUNTO FIJO de la entrada canónica (la conversión de un libro pasa por ella tal cual)
    assert normalize_csv_text(r.csv_text) == r.csv_text


def test_rep_c7_i1_partido_por_coma_o_tabulador_el_decimal_no_se_toca():
    """La regla del C2 solo vale en un CSV «;» (en un CSV de comas, `1,234` son miles): como el `.csv`, nada se convierte."""
    _requiere_calamine()
    r = convertir(_libro_de_una_columna(["a,precio,c", '1,"18,72",x', '2,"3,5",y', '3,"7,25",z']), "x.xlsx",
                  fila_de_cabecera=1, partir_columna_unica=",")
    assert [f[1] for f in csv.reader(io.StringIO(r.csv_text))][1:] == ["18,72", "3,5", "7,25"]
    assert _decimales(r.origen["declaraciones"]) == []
    r = convertir(_libro_de_una_columna(["a\tprecio\tc", "1\t18,72\tx", "2\t3,5\ty", "3\t7,25\tz"]), "x.xlsx",
                  fila_de_cabecera=1, partir_columna_unica="\t")
    assert [f[1] for f in csv.reader(io.StringIO(r.csv_text))][1:] == ["18,72", "3,5", "7,25"]
    assert _decimales(r.origen["declaraciones"]) == []


def test_rep_c7_i1_la_regla_del_decimal_es_la_del_c2_no_una_segunda():
    """La regla del decimal se RECORRE en un solo sitio (`decimal_es.decimal_de_un_csv_punto_y_coma`) y la llaman los dos caminos
    (re-auditoría, M-C: `data.py` tenía su propio bucle y el partido otro). Ningún otro módulo del núcleo construye el clasificador
    ni convierte celdas por su cuenta."""
    import inspect
    from matrixai.tablas import api
    from matrixai.training import data, decimal_es
    fuente = inspect.getsource(api.convertir)
    assert fuente.count("decimal_de_un_csv_punto_y_coma(") == 1
    assert inspect.getsource(data.normalize_csv_delimiter_declarando).count("decimal_de_un_csv_punto_y_coma(") == 1
    assert "replace(\",\", \".\")" not in fuente and "_RE_D" not in fuente
    raiz = pathlib.Path(decimal_es.__file__).parents[1]
    quien = {str(f.relative_to(raiz)): f.read_text(encoding="utf-8").count(t) for f in raiz.rglob("*.py")
             for t in ("ClasificadorDeColumnas(", "convertir_celda_de_una_columna_decidida(") if t in f.read_text(encoding="utf-8")}
    assert set(quien) == {"training/decimal_es.py"}, quien


def test_rep_c7_i1_un_csv_punto_y_coma_con_dos_columnas_de_decimal_coma_se_ofrece_partir_por_punto_y_coma():
    """Con DOS columnas de decimal coma, cada fila se parte por comas en 3 trozos y se ofrecía partir por COMAS. La cabecera manda
    (como en el `.csv`): sin comas y con «;», es un CSV «;»."""
    from matrixai.tablas.api import _columna_unica_con_separadores
    filas = [["fecha;precio;peso;compra"]] + [[f"2024-01-{i % 28 + 1:02d};{i},5;{i % 7},25;no"] for i in range(60)]
    assert _columna_unica_con_separadores(_Hoja(filas)) == ("punto_y_coma", 4)
    # y lo de antes sigue: una cabecera con comas se parte por comas aunque los datos lleven «;» entre comillas
    filas = [["a,b,c"]] + [['1,"x;y;z",2']] * 20
    assert _columna_unica_con_separadores(_Hoja(filas)) == ("coma", 3)


# ── I2: la banda de una celda no convierte en cabecera lo que va ENCIMA de un título ───────────────────────────────────────
_FORMAS_DE_CABECERA = {
    # nombre: (filas, cabecera correcta)
    "kpi_2textos_titulo_cabecera_estrecha": ([["Empresa", "ACME", _N], ["Ventas por región", _N, _N], ["región", "mes", "importe"],
                                              ["norte", "ene", 10], ["sur", "ene", 12]], 3),
    "kpi_2textos_titulo_cabecera_ancha": ([["Empresa", "ACME", _N, _N], ["Ventas por región", _N, _N, _N],
                                           ["región", "mes", "importe", "unidades"], ["norte", "ene", 10, 1], ["sur", "ene", 12, 2]], 3),
    "kpi_4textos_titulo_cabecera": ([["Empresa", "ACME", "Periodo", "2024", _N, _N], ["Ventas", _N, _N, _N, _N, _N],
                                     ["a", "b", "c", "d", "e", "f"], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], 3),
    "fuente_unidad_titulo_cabecera": ([["Fuente: INE", "Unidad: personas", _N], [_N, _N, _N], ["Total nacional", _N, _N],
                                       ["provincia", "sexo", "valor"], ["Madrid", "H", 1], ["Madrid", "M", 2]], 4),
    "nota_texto2_titulo_cabecera": ([["Nota:", "ver anexo", _N], ["Tabla 1", _N, _N], ["x", "y", "z"], [1, 2, 3], [4, 5, 6]], 3),
    "cuadro_kpi_texto": ([["Cuadro de mando", _N, _N], [_N, _N, _N], ["Responsable", "Ana García", _N], [_N, _N, _N],
                          ["Detalle por región", _N, _N], ["región", "mes", "importe"], ["norte", "ene", 10], ["sur", "ene", 12]], 6),
    "datos_texto_con_secciones_sin_cabecera": ([["NORTE", _N, _N], ["Madrid", "norte", 10], ["SUR", _N, _N], ["Sevilla", "sur", 9]],
                                               None),
    # el otro lado (lo que arregló e0978df y tiene que seguir): la banda de segundo nivel bajo la cabecera
    "banda_2nivel_INE": ([["provincia", "2023", "2022"], ["Todas las edades", _N, _N], ["Madrid", 1, 2], ["Bilbao", 3, 4]], 1),
    "cabecera_y_seccion": ([["ciudad", "zona", "v"], ["NORTE", _N, _N], ["Madrid", "norte", 10], ["Bilbao", "norte", 12]], 1),
    "banda_combinada_table_2b": ([["Table 2b"], [], ["England and Wales"], ["Grupo", "Total", "Cargos", "Avisos"],
                                  ["", "Number of detections", "", ""], ["Robo", 5, 3, 1], ["Hurto", 7, 4, 2]], 4),
    "cabecera_y_un_dato_suelto_ceros_y_vacios": ([["a", "b"], [1.0, _N], [_N, _N], [3.0, 4.0], [10.0, 2.5]], 1),
}


@pytest.mark.parametrize("nombre", list(_FORMAS_DE_CABECERA))
def test_rep_c7_i2_la_cabecera_propuesta_acierta_los_dos_lados_de_la_banda(nombre):
    """I2: `e0978df` saltaba una fila de UNA celda tras la candidata y proponía como cabecera unos metadatos de dos textos que van
    encima de un título (7 formas de oficina que antes acertaba). Tras un salto, la candidata solo es cabecera si lo de detrás
    tiene MENOS textos que ella. Las bandas de 2.º nivel (INE, Table 2b, una sección) siguen acertando."""
    from matrixai.tablas.api import _proponer_cabecera
    filas, correcta = _FORMAS_DE_CABECERA[nombre]
    assert _proponer_cabecera(_Hoja(filas)) == correcta


@pytest.mark.parametrize("nombre", ["kpi_2textos_titulo_cabecera_estrecha", "fuente_unidad_titulo_cabecera", "cuadro_kpi_texto",
                                    "datos_texto_con_secciones_sin_cabecera", "banda_2nivel_INE"])
def test_rep_c7_i2_y_por_inspeccionar_con_un_libro_de_verdad(nombre):
    _requiere_calamine()
    filas, correcta = _FORMAS_DE_CABECERA[nombre]
    datos = _G.xlsx([_G.Hoja("H", [list(f) for f in filas])])
    assert inspeccionar(datos, "x.xlsx")["hojas"][0]["fila_de_cabecera_propuesta"] == correcta


# ── M2: la vista previa de «;» y tabulador es lo que parte `convertir` ─────────────────────────────────────────────────────
@pytest.mark.parametrize("sep,clave,lineas", [
    (";", "punto_y_coma", ["nombre;ciudad;nota", '"Perez, Ana";"Lyon, FR";7', "Luis;Bilbao, ES;8", "Eva;Roma, IT;9"]),
    ("\t", "tabulador", ["nombre\tciudad\tnota", "Perez, Ana\tLyon; FR\t7", "Luis\tBilbao, ES\t8", "Eva\tRoma; IT\t9"]),
])
def test_rep_c7_m2_la_vista_previa_de_punto_y_coma_y_tabulador_es_lo_que_parte_convertir(sep, clave, lineas):
    _requiere_calamine()
    datos = _libro_de_una_columna(lineas)
    p = inspeccionar(datos, "x.xlsx")["hojas"][0]["partida_ofrecida"]
    assert p["separador"] == clave and p["columnas"] == 3
    r = convertir(datos, "x.xlsx", fila_de_cabecera=1, partir_columna_unica=sep)
    assert [f["celdas"] for f in p["primeras"]] == list(csv.reader(io.StringIO(r.csv_text)))
    assert p["primeras"][1]["celdas"][0] == "Perez, Ana"


# ── M3: el umbral del 90 % de la oferta ────────────────────────────────────────────────────────────────────────────────────
def test_rep_c7_m3_se_ofrece_partir_con_el_90_por_ciento_de_las_celdas_y_no_con_menos():
    from matrixai.tablas.api import _columna_unica_con_separadores
    def hoja(buenas, malas):
        return _Hoja([["a,b,c"]] + [["1,2,3"]] * (buenas - 1) + [["una nota suelta"]] * malas)
    assert _columna_unica_con_separadores(hoja(9, 1)) == ("coma", 3)       # 90 %: se ofrece
    assert _columna_unica_con_separadores(hoja(18, 2)) == ("coma", 3)
    assert _columna_unica_con_separadores(hoja(8, 2)) is None               # 80 %: no
    assert _columna_unica_con_separadores(hoja(17, 3)) is None              # 85 %: no
    assert _columna_unica_con_separadores(hoja(6, 4)) is None               # 60 %: no


# ── M4: el BOM del CSV pegado se quita sin declararlo, como en el `.csv` ─────────────────────────────────────────────────────
def test_rep_c7_m4_el_bom_del_csv_pegado_se_quita_como_en_el_csv_sin_declarar_un_renombre():
    """`strip_csv_bom` quita el U+FEFF de un `.csv` sin decir nada: es la marca de la codificación, no un nombre. Al partir, igual
    (sin la línea de `_partir_filas`, `nombre_normalizado` lo quitaba y declaraba un renombre que la persona no hizo)."""
    _requiere_calamine()
    lineas = ["﻿id,x,y", "1,2,3", "4,5,6"]
    r = convertir(_libro_de_una_columna(lineas), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=",")
    assert r.csv_text == normalize_csv_text("\n".join(lineas) + "\n")
    assert [d["tipo"] for d in r.origen["declaraciones"]] == ["columna_unica_partida"]


# ── M5: las filas con otro número de campos se dicen ──────────────────────────────────────────────────────────────────────
def test_rep_c7_m5_las_filas_partidas_con_otro_numero_de_campos_se_declaran_con_su_numero_de_fila():
    _requiere_calamine()
    lineas = ["id,txt,v", '1,"hola', 'mundo",2'] + [f"{i},x{i},{i}" for i in range(2, 50)] + ["99,corta"]
    r = convertir(_libro_de_una_columna(lineas), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=",")
    d = [x for x in r.origen["declaraciones"] if x["tipo"] == "columna_unica_filas_desiguales"]
    assert len(d) == 1 and d[0]["cuantas"] == 3 and d[0]["detalle"]["filas"] == [2, 3, 52] and d[0]["detalle"]["campos"] == 3
    assert "salto de línea" in d[0]["texto"]["es"] and "line break" in d[0]["texto"]["en"]
    assert "filas" not in d[0]["texto"]["en"] and "«" not in d[0]["texto"]["en"]
    # una partición sin filas desiguales no lo dice
    r = convertir(_libro_de_una_columna(["a,b,c", "1,2,3", "4,5,6"]), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=",")
    assert "columna_unica_filas_desiguales" not in [x["tipo"] for x in r.origen["declaraciones"]]


# ── M6: solo se parte lo que se ofrece ────────────────────────────────────────────────────────────────────────────────────
def test_rep_c7_m6_convertir_no_parte_lo_que_inspeccionar_no_ofrece():
    _requiere_calamine()
    irregular = _libro_de_una_columna(["a,b,c", "1,2", "2,3,4,5", "3,4,5"])   # 2 de 4 con 3 campos: no se ofrece
    assert "partida_ofrecida" not in inspeccionar(irregular, "x.xlsx")["hojas"][0]
    with pytest.raises(TablaRechazada) as e:
        convertir(irregular, "x.xlsx", fila_de_cabecera=1, partir_columna_unica=",")
    assert e.value.codigo == "parametro_invalido" and "no se parte" in e.value.motivo_es and "not split" in e.value.motivo_en
    # ofrecido por «;», pedido por «,»: tampoco
    pyc = _libro_de_una_columna(["fecha;precio;peso", "2024-01-01;1,5;2,5", "2024-01-02;3,5;4,5", "2024-01-03;5,5;6,5"])
    assert inspeccionar(pyc, "x.xlsx")["hojas"][0]["partida_ofrecida"]["separador"] == "punto_y_coma"
    with pytest.raises(TablaRechazada):
        convertir(pyc, "x.xlsx", fila_de_cabecera=1, partir_columna_unica=",")


# ── M7: textos ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_rep_c7_m7_los_textos_del_aviso_del_parametro_y_de_columna_mixta():
    from matrixai.tablas.declaraciones import declarar
    es = declarar("columna_unica_con_separadores", campos=4, separador="coma")["texto"]["es"]
    assert "y guardar el libro de nuevo" in es and "y guarda el libro" not in es
    mixta = declarar("columna_mixta", columna="fecha", cuantas=3, tipos={"fecha": 1, "numero": 1, "texto": 1})
    assert mixta["texto"]["en"] == "«fecha» mixes value types (date: 1, number: 1, text: 1)."
    assert mixta["texto"]["es"] == "«fecha» mezcla tipos de valor (fecha: 1, número: 1, texto: 1)."
    assert mixta["detalle"]["tipos"] == {"fecha": 1, "numero": 1, "texto": 1}   # la clave estable no cambia
    from matrixai.tablas.valores import tipo_de_valor
    import datetime as dt
    from matrixai.tablas.declaraciones import _TIPOS_DE_VALOR
    for v in (1.5, "x", dt.date(2024, 1, 1), dt.time(1), dt.timedelta(1), True):
        assert tipo_de_valor(v) in _TIPOS_DE_VALOR       # un tipo nuevo sin traducir lo caza
    _requiere_calamine()
    with pytest.raises(TablaRechazada) as e:
        convertir(_libro_de_una_columna(["a,b,c", "1,2,3", "4,5,6"]), "x.xlsx", fila_de_cabecera=1, partir_columna_unica="|")
    assert "«" not in e.value.motivo_en and "»" not in e.value.motivo_en


# ═══ 150-C7, decisión de Roberto del 09-11 (tarjeta 150-c7-cabecera-de-dos-niveles = A) ═══════════════════════════════════
# (a) una banda combinada encima o debajo de los nombres se AVISA y se OFRECE componer «grupo · nombre» o quitarla; solo se hace
# si la persona lo acepta (`componer_cabecera_de_dos_niveles`, o la fila de la banda en `filas_a_excluir`), y se declara.
# (b) «Notas:», «Fuente:» y el organismo se ofrecen con su propio motivo, `pie`; (c) los totales, como estaban.
def _libro_banda_encima() -> bytes:
    return _G.xlsx([_G.Hoja("H", [["", "Índice", None, "Variación", None],
                                  ["grupo", "2025M02", "2025M01", "2025M02", "2025M01"],
                                  ["general", 1.5, 1.4, 0.1, 0.2], ["alimentos", 2.5, 2.4, 0.3, 0.4], ["vivienda", 3.5, 3.4, 0.5, 0.6]],
                            combinadas=("B1:C1", "D1:E1"))])


def _libro_banda_debajo() -> bytes:
    return _G.xlsx([_G.Hoja("H", [["Grupo", "Total", "Cargos", "Avisos", "Otras"], ["", "Number of detections", None, None, None],
                                  ["Robo", 5, 3, 1, 9], ["Hurto", 7, 4, 2, 8], ["Fraude", 9, 5, 3, 7]],
                            combinadas=("B2:D2",))])


def _tipos_de(r):
    return [d["tipo"] for d in r.origen["declaraciones"]]


def test_dos_niveles_a_una_banda_encima_de_los_nombres_se_avisa_y_se_ofrece_con_los_nombres_compuestos():
    _requiere_calamine()
    h = inspeccionar(_libro_banda_encima(), "x.xlsx")["hojas"][0]
    assert h["fila_de_cabecera_propuesta"] == 2
    o = h["cabecera_de_dos_niveles_ofrecida"]
    assert o == {"fila_de_cabecera": 2, "fila_de_la_banda": 1, "posicion": "encima", "grupos": ["Índice", "Variación"],
                 "columnas_compuestas": 4, "se_puede_quitar_la_banda": False,
                 "nombres_compuestos": ["Índice · 2025M02", "Índice · 2025M01", "Variación · 2025M02", "Variación · 2025M01"]}
    a = next(x for x in h["avisos"] if x["tipo"] == "posible_cabecera_de_dos_niveles")
    assert "dos niveles" in a["texto"]["es"] and "Índice" in a["texto"]["es"]
    assert "two-level header" in a["texto"]["en"] and "«" not in a["texto"]["en"] and "encima" not in a["texto"]["en"]


def test_dos_niveles_a_sin_aceptar_nada_se_compone_y_aceptado_se_compone_y_se_declara():
    _requiere_calamine()
    datos = _libro_banda_encima()
    sin = convertir(datos, "x.xlsx", fila_de_cabecera=2)
    assert sin.csv_text.splitlines()[0] == "grupo,2025M02,2025M01,2025M02_2,2025M01_2"   # como antes: repetidos renombrados
    assert "cabecera_de_dos_niveles_compuesta" not in _tipos_de(sin)
    con = convertir(datos, "x.xlsx", fila_de_cabecera=2, componer_cabecera_de_dos_niveles=True)
    assert con.csv_text.splitlines()[0] == "grupo,Índice · 2025M02,Índice · 2025M01,Variación · 2025M02,Variación · 2025M01"
    assert con.csv_text.splitlines()[1:] == sin.csv_text.splitlines()[1:]                 # los datos, los mismos
    d = next(x for x in con.origen["declaraciones"] if x["tipo"] == "cabecera_de_dos_niveles_compuesta")
    assert d["cuantas"] == 4 and d["detalle"]["fila"] == 1 and d["detalle"]["posicion"] == "encima"
    assert "grupo · nombre" in d["texto"]["es"] and "group · name" in d["texto"]["en"] and "aceptó" not in d["texto"]["en"]
    tipos = _tipos_de(con)
    assert "nombre_repetido_renombrado" not in tipos and "filas_previas_omitidas" not in tipos and "nombre_normalizado" not in tipos


def test_dos_niveles_a_una_banda_debajo_se_compone_y_deja_de_ser_un_dato_o_se_quita_como_fila_excluida():
    _requiere_calamine()
    datos = _libro_banda_debajo()
    h = inspeccionar(datos, "x.xlsx")["hojas"][0]
    o = h["cabecera_de_dos_niveles_ofrecida"]
    assert (o["fila_de_la_banda"], o["posicion"], o["se_puede_quitar_la_banda"], o["columnas_compuestas"]) == (2, "debajo", True, 3)
    sin = list(csv.reader(io.StringIO(convertir(datos, "x.xlsx", fila_de_cabecera=1).csv_text)))
    assert sin[1][1] == "Number of detections"                                             # sin aceptar, como antes
    con = convertir(datos, "x.xlsx", fila_de_cabecera=1, componer_cabecera_de_dos_niveles=True)
    filas = list(csv.reader(io.StringIO(con.csv_text)))
    assert filas[0] == ["Grupo", "Number of detections · Total", "Number of detections · Cargos", "Number of detections · Avisos", "Otras"]
    assert [f[0] for f in filas[1:]] == ["Robo", "Hurto", "Fraude"]
    assert "celdas_combinadas_vacias" not in _tipos_de(con)
    d = next(x for x in con.origen["declaraciones"] if x["tipo"] == "cabecera_de_dos_niveles_compuesta")
    assert d["cuantas"] == 3 and "ya no es un dato" in d["texto"]["es"] and "no longer data" in d["texto"]["en"]
    quitada = convertir(datos, "x.xlsx", fila_de_cabecera=1, filas_a_excluir=[2])
    assert [f[0] for f in csv.reader(io.StringIO(quitada.csv_text))][1:] == ["Robo", "Hurto", "Fraude"]
    assert quitada.origen["filas_excluidas"] == [{"fila": 2, "texto": "Number of detections"}]
    # M-B (re-auditoría): al componer, la fila de la banda sale de los datos Y está en `filas_excluidas`, con su motivo
    assert con.origen["filas_excluidas"] == [{"fila": 2, "motivo": "banda_de_dos_niveles", "texto": "Number of detections"}]


def test_dos_niveles_a_solo_se_compone_lo_que_se_ofrece():
    _requiere_calamine()
    with pytest.raises(TablaRechazada) as e:          # sin banda
        convertir(_G.xlsx([_G.Hoja("H", [["a", "b"], [1, 2], [3, 4]])]), "x.xlsx", fila_de_cabecera=1,
                  componer_cabecera_de_dos_niveles=True)
    assert e.value.codigo == "parametro_invalido" and "no se compone" in e.value.motivo_es and "nothing is composed" in e.value.motivo_en
    with pytest.raises(TablaRechazada):               # otra cabecera que la de la banda
        convertir(_libro_banda_encima(), "x.xlsx", fila_de_cabecera=3, componer_cabecera_de_dos_niveles=True)
    with pytest.raises(TablaRechazada):               # con una cabecera de dos filas
        convertir(_libro_banda_encima(), "x.xlsx", fila_de_cabecera=1, filas_de_cabecera=2, componer_cabecera_de_dos_niveles=True)
    with pytest.raises(TablaRechazada) as e:          # no es un booleano
        convertir(_libro_banda_encima(), "x.xlsx", fila_de_cabecera=2, componer_cabecera_de_dos_niveles="true")
    assert "«" not in e.value.motivo_en


def test_dos_niveles_a_un_titulo_combinado_sobre_todas_las_columnas_no_es_una_banda():
    from matrixai.tablas.api import _banda_de_dos_niveles
    class H(_Hoja):
        combinadas = [(1, 1, 1, 3)]
        col0 = 1
    assert _banda_de_dos_niveles(H([["Ventas de 2024", None, None], ["a", "b", "c"], [1, 2, 3]]), 2) is None
    H.combinadas = [(1, 2, 1, 3)]
    assert _banda_de_dos_niveles(H([[None, "Ventas", None], ["a", "b", "c"], [1, 2, 3]]), 2)["grupos"] == [(2, 3, "Ventas")]


def test_dos_niveles_a_una_tabla_sin_banda_no_lo_ofrece_ni_lo_avisa():
    """Un rango combinado VERTICAL (oficina_con_valor) o una cabecera de dos filas (la banda ES la cabecera) no son esto."""
    _requiere_calamine()
    for nombre in ("erp_totales.xlsx", "oficina_con_valor.xlsx", "cabecera_dos_filas.xlsx", "cabecera_en_fila_15.xlsx", "nombres.xlsx"):
        for h in inspeccionar((pathlib.Path(__file__).parent / "fixtures" / "tablas" / nombre).read_bytes(), nombre)["hojas"]:
            assert "cabecera_de_dos_niveles_ofrecida" not in h, (nombre, h["nombre"])
            assert "posible_cabecera_de_dos_niveles" not in [a["tipo"] for a in h["avisos"]]


@pytest.mark.parametrize("nombre,hoja,cab,banda,posicion,primero", [
    ("ine_ipc_50902.xlsx", "tabla-50902", 8, 7, "encima", "Índice · 2025M12"),
    ("uk_homeoffice_hosb0812.xls", "Table 2b", 4, 5, "debajo", "Number of detections · Total recorded crime"),
    ("uk_homeoffice_hosb0812.xls", "Table 4", 6, 7, "debajo", "Number of detections · 2006/07"),
])
def test_dos_niveles_a_los_libros_publicos_con_banda(nombre, hoja, cab, banda, posicion, primero):
    datos = _libro(nombre)
    h = {x["nombre"]: x for x in inspeccionar(datos, nombre)["hojas"]}[hoja]
    o = h["cabecera_de_dos_niveles_ofrecida"]
    assert (o["fila_de_cabecera"], o["fila_de_la_banda"], o["posicion"]) == (cab, banda, posicion)
    assert o["nombres_compuestos"][0] == primero
    con = convertir(datos, nombre, hoja=hoja, fila_de_cabecera=cab, componer_cabecera_de_dos_niveles=True)
    assert next(csv.reader(io.StringIO(con.csv_text)))[1] == primero
    assert "nombre_repetido_renombrado" not in _tipos_de(con)   # INE IPC: sin componer, los meses se repetían 4 veces


# ── (b) las notas al pie, con su motivo; (c) los totales, como estaban ────────────────────────────────────────────────────
def test_pie_b_notas_fuente_y_organismo_se_ofrecen_como_pie_y_no_como_total():
    _requiere_calamine()
    filas = ([["k", "v", "w"]] + [[f"x{i}", i, i] for i in range(12)] + [["Total", 66, 66], [None, None, None], ["Notas:"],
             [None], ["Fuente: INE"], ["Instituto Nacional de Estadística"]])
    h = inspeccionar(_G.xlsx([_G.Hoja("H", filas)]), "x.xlsx")["hojas"][0]
    assert [(c["fila"], c["motivo"]) for c in h["filas_candidatas_a_total"]] == [(14, "palabra"), (16, "pie"), (18, "pie"), (19, "pie")]
    # nunca se quitan solas: sin `filas_a_excluir`, siguen en el CSV
    r = convertir(_G.xlsx([_G.Hoja("H", filas)]), "x.xlsx", fila_de_cabecera=1)
    assert "Notas:" in r.csv_text and "Instituto Nacional de Estadística" in r.csv_text and "Total,66,66" in r.csv_text


def test_pie_b_una_ultima_fila_corta_que_no_es_una_nota_sigue_siendo_mitad_vacia():
    from matrixai.tablas.api import _candidatas_a_total
    filas = [["a", "b", "c", "d"]] + [[i, i, i, i] for i in range(10)] + [[None] * 4, [564, None, None, 23], ["sin datos"]]
    assert [(c["fila"], c["motivo"]) for c in _candidatas_a_total(_Hoja(filas), 2)] == [(13, "mitad_vacia"), (14, "mitad_vacia")]


@pytest.mark.parametrize("nombre,hoja,pie,palabras", [
    ("ine_ipc_50902.xlsx", "tabla-50902", [24, 26, 27], []),
    ("ine_poblacion_56934.xlsx", "tabla-56934", [446, 448], None),
    ("uk_homeoffice_hosb0812.xls", "Table 2b", [30, 31, 32, 33, 34], [16, 28]),
    ("uk_homeoffice_hosb0812.xls", "Table 1", [252, 253, 254, 255, 256, 257, 258], None),   # 252-253: notas que antes no se veían
])
def test_pie_b_c_los_libros_publicos_notas_como_pie_y_totales_como_palabra(nombre, hoja, pie, palabras):
    h = {x["nombre"]: x for x in inspeccionar(_libro(nombre), nombre)["hojas"]}[hoja]
    c = h["filas_candidatas_a_total"]
    assert [x["fila"] for x in c if x["motivo"] == "pie"] == pie
    if palabras is not None:
        assert [x["fila"] for x in c if x["motivo"] == "palabra"] == palabras
    else:
        assert len([x for x in c if x["motivo"] == "palabra"]) >= 10   # los «Total» siguen ofreciéndose como total (c)


def test_pie_c_un_total_dentro_del_pie_sigue_siendo_un_total():
    from matrixai.tablas.api import _candidatas_a_total
    filas = [["a", "b", "c"]] + [[f"x{i}", i, i] for i in range(10)] + [["Notas: provisional"], ["Total"]]
    assert [(c["fila"], c["motivo"]) for c in _candidatas_a_total(_Hoja(filas), 2)] == [(12, "pie"), (13, "palabra")]


# ═══ 2.ª reparación del 150-C7 (re-auditoría del 09-11: I-R1, I-R2, M-B, M-C) ═════════════════════════════════════════════
# Informe y sondas en ~/.agentes/150/reauditoria-c7/; notas, medidas y sabotajes en ~/.agentes/150/reparacion2-c7/.

# ── I-R1: la cabecera acierta por los DOS lados (lo de detrás del salto, por sus TIPOS) ───────────────────────────────────────
_FORMAS_DE_CABECERA_2 = {
    # las de la re-auditoría (`formas_cabecera.py`); F1 y F2 fallaban con la regla de «menos textos»
    "F1_metadatos_3_textos_titulo_cabecera_de_2": ([["Empresa", "ACME", "T1 2024"], ["Ventas mensuales", _N, _N], ["mes", "importe", _N],
                                                   ["ene", 10, _N], ["feb", 12, _N], ["mar", 9, _N]], 3),
    "F2_cabecera_de_3_banda_y_datos_todo_texto": ([["nombre", "ciudad", "categoria"], ["Todas las categorías", _N, _N],
                                                  ["Ana", "Madrid", "A"], ["Luis", "Bilbao", "B"], ["Eva", "Sevilla", "A"]], 1),
    "F3_cabecera_de_2_banda_y_datos_mixtos": ([["provincia", "habitantes"], ["Ambos sexos", _N], ["Madrid", 100], ["Bilbao", 50],
                                              ["Soria", 9]], 1),
    "F4_metadatos_2_titulo_cabecera_de_5": ([["Fuente: SEPE", "Fecha: 2024-05"], ["Paro registrado", _N],
                                            ["provincia", "sexo", "edad", "sector", "parados"], ["Madrid", "H", "<25", "servicios", 10],
                                            ["Madrid", "M", "<25", "servicios", 12]], 3),
    "F5_cabecera_de_4_vacia_banda_datos_de_3_textos": ([["pais", "region", "sexo", "valor"], [_N] * 4, ["Total", _N, _N, _N],
                                                       ["ES", "Norte", "H", 1], ["ES", "Sur", "M", 2], ["FR", "Norte", "H", 3]], 1),
    "F7_metadatos_2_titulo_cabecera_repetida": ([["Hospital", "Valdecilla", _N], ["Consultas", _N, _N], ["servicio", "servicio", "total"],
                                                ["cardio", "c1", 10], ["neuro", "n1", 5]], 3),
    # 4 formas NUEVAS de esta reparación (no estaban en ninguna prueba ni sonda); N1, N2 y N4 fallaban antes
    "N1_metadatos_3_vacia_titulo_cabecera_de_2_con_fechas": ([["Hospital", "Valdecilla", "2024"], [_N] * 3, ["Consultas por servicio", _N, _N],
                                                             ["fecha", "servicio", _N], [_dt.date(2024, 1, 1), "cardio", _N],
                                                             [_dt.date(2024, 1, 2), "neuro", _N]], 4),
    "N2_cabecera_de_3_banda_y_un_guion_de_ausencia_en_el_primer_dato": ([["ciudad", "zona", "habitantes"], ["Todas las zonas", _N, _N],
                                                                        ["Madrid", "centro", "-"], ["Bilbao", "norte", 50],
                                                                        ["Soria", "centro", 9]], 1),
    "N3_metadatos_2_titulo_cabecera_de_4_con_el_numero_al_final": ([["Proyecto", "Alfa", _N, _N], ["Horas por persona", _N, _N, _N],
                                                                   ["nombre", "equipo", "rol", "horas"], ["Ana", "A", "dev", 10],
                                                                   ["Luis", "B", "qa", 12]], 3),
    "N4_cabecera_de_2_banda_y_datos_todo_texto": ([["producto", "categoria"], ["Catálogo completo", _N], ["Tornillo", "ferretería"],
                                                  ["Martillo", "herramienta"], ["Tuerca", "ferretería"]], 1),
}


@pytest.mark.parametrize("nombre", list(_FORMAS_DE_CABECERA_2))
def test_rep2_c7_ir1_la_cabecera_propuesta_por_los_tipos_de_lo_que_hay_detras_del_salto(nombre):
    """I-R1: tras saltar una fila de una celda, «lo de detrás tiene menos textos» fallaba por los dos lados (F1: metadatos de 3
    textos sobre una cabecera de 2; F2: cabecera con banda y datos todo texto). Ahora decide el TIPO: lo de detrás es otra cabecera
    si es todo nombres con un número o una fecha debajo de uno de ellos, o si mezcla tipos con al menos tantos textos."""
    from matrixai.tablas.api import _proponer_cabecera
    filas, correcta = _FORMAS_DE_CABECERA_2[nombre]
    assert _proponer_cabecera(_Hoja(filas)) == correcta


@pytest.mark.parametrize("nombre", ["F1_metadatos_3_textos_titulo_cabecera_de_2", "F2_cabecera_de_3_banda_y_datos_todo_texto",
                                    "N1_metadatos_3_vacia_titulo_cabecera_de_2_con_fechas",
                                    "N2_cabecera_de_3_banda_y_un_guion_de_ausencia_en_el_primer_dato",
                                    "N4_cabecera_de_2_banda_y_datos_todo_texto"])
def test_rep2_c7_ir1_y_por_inspeccionar_con_un_libro_de_verdad(nombre):
    _requiere_calamine()
    filas, correcta = _FORMAS_DE_CABECERA_2[nombre]
    datos = _G.xlsx([_G.Hoja("H", [list(f) for f in filas])])
    assert inspeccionar(datos, "x.xlsx")["hojas"][0]["fila_de_cabecera_propuesta"] == correcta


def test_rep2_c7_ir1_una_marca_de_ausencia_no_cuenta_como_un_nombre():
    """N2 sin la guarda de las marcas: «-» bajo «habitantes» en la 1.ª fila de datos haría de esa fila «todo nombres con un número
    debajo» y se propondría como cabecera. Las marcas son las del análisis (`_NULL_TOKENS`), no una lista nueva."""
    from matrixai.tablas.api import _es_un_nombre
    for marca in ("-", "n/a", "NA", "null", "?", " nan "):
        assert not _es_un_nombre(marca), marca
    assert _es_un_nombre("habitantes") and not _es_un_nombre(10) and not _es_un_nombre("  ")


# ── Remate de la 3.ª auditoría (IMPORTANTE-1, MENOR-1): años como números y la siguiente fila con ≥ 2 datos ───────────────────
_FORMAS_PERIODOS = {
    # la regla 2 proponía el primer DATO: «lo de detrás» (2 textos, números) tenía tantos textos como la cabecera de años numéricos
    "A1_anios_numericos_y_una_banda": ([["Comunidad", "Sexo", 2023, 2024], ["Todas las edades", _N, _N, _N],
                                       ["Andalucía", "Ambos sexos", 8500, 8600], ["Aragón", "Ambos sexos", 1300, 1310],
                                       ["Asturias", "Hombres", 490, 488]], 1),
    "A12_numeracion_correlativa_y_una_seccion": ([["servicio", "turno", 1, 2, 3], ["Enero", _N, _N, _N, _N],
                                                 ["Urgencias", "Mañana", 10, 11, 12], ["UCI", "Noche", 3, 4, 5]], 1),
    "A16_anios_numericos_y_una_unidad": ([["Provincia", "Sexo", 2022, 2023], ["(en miles)", _N, _N, _N],
                                         ["Cantabria", "Mujeres", 300, 301], ["Cantabria", "Hombres", 280, 281]], 1),
    # el otro lado: la hoja SIN cabecera, con números de 4 cifras que no son años seguidos, sigue sin proponer la fila de datos
    "S1_sin_cabecera_numeros_de_4_cifras_que_no_son_anios": ([["NORTE", _N, _N, _N], ["Madrid", "norte", 1234, 1500],
                                                             ["SUR", _N, _N, _N], ["Sevilla", "sur", 2900, 3100]], None),
    "S2_sin_cabecera_un_solo_numero_de_anio": ([["NORTE", _N, _N], ["Madrid", "norte", 2023], ["SUR", _N, _N],
                                               ["Sevilla", "sur", 2019]], None),
    "S3_sin_cabecera_numeros_pequenos_no_correlativos": ([["NORTE", _N, _N, _N, _N], ["Madrid", "norte", 10, 12, 14],
                                                         ["SUR", _N, _N, _N, _N], ["Sevilla", "sur", 9, 8, 7]], None),
}


@pytest.mark.parametrize("nombre", list(_FORMAS_PERIODOS))
def test_rem3_c7_ir1_los_numeros_de_periodos_en_la_cabecera_no_la_descartan_y_los_datos_sin_cabecera_siguen_protegidos(nombre):
    """IMPORTANTE-1: una cabecera con los años (o 1, 2, 3) como NÚMEROS, una fila de una celda debajo y datos de 2 textos se leía
    como «datos sobre una sección» (regla 2) y se proponía el primer dato. Los números de periodos no la activan; los números
    corrientes, sí (la hoja sin cabecera, de la que protegía, sigue igual)."""
    from matrixai.tablas.api import _proponer_cabecera
    filas, correcta = _FORMAS_PERIODOS[nombre]
    assert _proponer_cabecera(_Hoja(filas)) == correcta


def test_rem3_c7_ir1_que_son_numeros_de_periodos():
    from matrixai.tablas.api import _numeros_de_periodos as p
    assert p(["a", "b", 2022, 2023]) and p(["a", 1, 2, 3]) and p(["a", 2022.0, 2023.0, _N])
    assert not p(["a", "b", 300, 301]) and not p(["a", "b", 2023]) and not p(["a", "b", 1234, 1500])
    assert not p(["a", "b", 2023, 2022]) and not p(["a", 1, 2]) and not p(["a", "b", 2022, 1.5])


_F1_BASE = [["Empresa", "ACME", "T1 2024"], ["Ventas mensuales", _N, _N], ["mes", "importe", _N]]
_F1_DATOS = [["ene", 10, _N], ["feb", 12, _N], ["mar", 9, _N]]


@pytest.mark.parametrize("nombre,entre", [
    ("fila_vacia", [[_N, _N, _N]]),
    ("fila_de_unidad", [["(euros)", _N, _N]]),
    ("vacia_y_unidad", [[_N, _N, _N], ["(euros)", _N, _N]]),
])
def test_rem3_c7_ir1_la_siguiente_fila_con_dos_datos_no_la_inmediata(nombre, entre):
    """MENOR-1: la cabecera «mes | importe» con metadatos de 3 textos y un título encima, y una fila vacía o de una celda ENTRE
    ella y los datos. Lo de detrás del salto se busca en la siguiente fila con ≥ 2 datos (la de los datos), no en la inmediata
    (vacía: «todo nombres» sin nada tipado debajo, y se proponían los metadatos). Correcta: la fila 3."""
    from matrixai.tablas.api import _proponer_cabecera
    assert _proponer_cabecera(_Hoja(_F1_BASE + entre + _F1_DATOS)) == 3


# ── I-R2: un grupo que cubre la PRIMERA columna no es un primer nivel ─────────────────────────────────────────────────────────
_BANDAS_2 = {
    # SIN dos niveles (`bandas.py` de la re-auditoría): no se avisa ni se ofrece
    "T1_titulo_combinado_sobre_parte": ([["Ventas 2024", _N, _N, _N, _N], ["region", "mes", "importe", "unidades", "iva"],
                                         ["n", "ene", 1, 2, 3], ["s", "feb", 4, 5, 6], ["e", "mar", 7, 8, 9]], ("A1:C1",), None),
    "T2_metadatos_combinados": ([["Empresa", _N, "ACME", _N, _N, _N], ["region", "mes", "importe", "unidades", "iva", "total"],
                                 ["n", "ene", 1, 2, 3, 4], ["s", "feb", 4, 5, 6, 7], ["e", "mar", 7, 8, 9, 1]], ("A1:B1", "C1:D1"), None),
    "T3_seccion_combinada_bajo_la_cabecera": ([["ciudad", "zona", "v", "w", "x", "y"], ["Zona norte", _N, _N, _N, _N, _N],
                                               ["Bilbao", "n", 1, 2, 3, 4], ["Santander", "n", 5, 6, 7, 8], ["Oviedo", "n", 1, 1, 1, 1]],
                                              ("A2:E2",), None),
    "T5_fecha_y_autor_combinados": ([["Fecha: 2024-05-01", _N, _N, "Autor: Ana", _N, _N], ["a", "b", "c", "d", "e", "f"],
                                     [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], ("A1:C1", "D1:F1"), None),
    # CON dos niveles: se sigue ofreciendo
    "V2_banda_encima_desde_la_segunda_columna": ([[_N, "Ventas", _N, "Costes", _N], ["region", "ene", "feb", "ene", "feb"],
                                                  ["n", 1, 2, 3, 4], ["s", 5, 6, 7, 8], ["e", 1, 1, 1, 1]], ("B1:C1", "D1:E1"),
                                                 ("encima", "Ventas · ene")),
    "titulo_parcial_encima_y_banda_de_verdad_debajo": ([["Delitos 2024", _N, _N, _N, _N], ["Grupo", "Total", "Cargos", "Avisos", "Otras"],
                                                        ["", "Number of detections", _N, _N, _N], ["Robo", 5, 3, 1, 9],
                                                        ["Hurto", 7, 4, 2, 8]], ("A1:C1", "B3:D3"),
                                                       ("debajo", "Number of detections · Total")),
}


@pytest.mark.parametrize("nombre", list(_BANDAS_2))
def test_rep2_c7_ir2_un_grupo_sobre_la_primera_columna_no_es_un_primer_nivel(nombre):
    """I-R2: un título, unos metadatos o una sección combinados DESDE la primera columna se avisaban como «cabecera de dos niveles»
    y se ofrecía componer «Ventas 2024 · region». Un primer nivel agrupa columnas de datos a la derecha de la etiqueta."""
    _requiere_calamine()
    filas, combinadas, esperada = _BANDAS_2[nombre]
    h = inspeccionar(_G.xlsx([_G.Hoja("H", [list(f) for f in filas], combinadas=combinadas)]), "x.xlsx")["hojas"][0]
    tipos = [a["tipo"] for a in h["avisos"]]
    if esperada is None:
        assert "cabecera_de_dos_niveles_ofrecida" not in h and "posible_cabecera_de_dos_niveles" not in tipos
    else:
        o = h["cabecera_de_dos_niveles_ofrecida"]
        assert (o["posicion"], o["nombres_compuestos"][0]) == esperada and "posible_cabecera_de_dos_niveles" in tipos


def test_rep2_c7_ir2_sin_oferta_componer_se_rechaza_en_convertir():
    """La MISMA función decide la oferta y lo que `convertir` acepta: en T1 ya no hay nada que componer."""
    _requiere_calamine()
    filas, combinadas, _ = _BANDAS_2["T1_titulo_combinado_sobre_parte"]
    with pytest.raises(TablaRechazada) as e:
        convertir(_G.xlsx([_G.Hoja("H", [list(f) for f in filas], combinadas=combinadas)]), "x.xlsx", fila_de_cabecera=2,
                  componer_cabecera_de_dos_niveles=True)
    assert e.value.codigo == "parametro_invalido"


# ── M-B: la banda de debajo, en `filas_excluidas` con su motivo ───────────────────────────────────────────────────────────────
def test_rep2_c7_mb_la_banda_compuesta_esta_en_filas_excluidas_junto_a_las_que_elige_la_persona_sin_repetirse():
    _requiere_calamine()
    datos = _libro_banda_debajo()
    con = convertir(datos, "x.xlsx", fila_de_cabecera=1, componer_cabecera_de_dos_niveles=True, filas_a_excluir=[5])
    assert con.origen["filas_excluidas"] == [{"fila": 2, "motivo": "banda_de_dos_niveles", "texto": "Number of detections"},
                                             {"fila": 5, "texto": "Fraude"}]
    assert [f[0] for f in csv.reader(io.StringIO(con.csv_text))][1:] == ["Robo", "Hurto"]
    # la que eligió la persona sigue declarada como suya («Se quitó 1 fila que elegiste»), la banda la declara lo compuesto
    d = next(x for x in con.origen["declaraciones"] if x["tipo"] == "filas_excluidas")
    assert d["cuantas"] == 1 and d["detalle"]["filas"] == [5]
    # si la persona ADEMÁS marca la banda como fila a quitar, sale una vez (como suya)
    doble = convertir(datos, "x.xlsx", fila_de_cabecera=1, componer_cabecera_de_dos_niveles=True, filas_a_excluir=[2])
    assert doble.origen["filas_excluidas"] == [{"fila": 2, "texto": "Number of detections"}]
    # con la banda ENCIMA no hay fila de datos que quitar: nada en `filas_excluidas`
    assert convertir(_libro_banda_encima(), "x.xlsx", fila_de_cabecera=2,
                     componer_cabecera_de_dos_niveles=True).origen["filas_excluidas"] == []


# ── M-C: UNA señal y UN bucle para el decimal del CSV «;», por el `.csv` y por el partido ────────────────────────────────────
_PARIDAD_PYC = {
    **_CSV_PYC,
    "cabecera_con_una_coma_en_un_nombre": ["nombre;importe, en euros;fecha"] + [f"x{i};{i},5;2024-01-{i % 28 + 1:02d}" for i in range(1, 12)],
    "codigos_con_ceros": ["id;cod;v"] + [f"{i};0{i % 9},0{i % 7};{i}" for i in range(1, 12)],
    "ingles_con_miles": ["item;price;qty"] + [f"a{i};{i},{i * 13 % 1000:03d}.50;{i}" for i in range(1, 12)],
}


@pytest.mark.parametrize("caso", list(_PARIDAD_PYC))
def test_rep2_c7_mc_el_partido_y_el_csv_deciden_el_decimal_igual_con_la_misma_senal(caso):
    """M-C: el `.csv` exigía la cabecera sin `,`; el partido, solo la oferta por `;`. Con `nombre;importe, en euros;fecha` el partido
    convertía `1,5` y el `.csv` no. Ahora los dos usan `es_la_cabecera_de_un_csv_punto_y_coma` y `decimal_de_un_csv_punto_y_coma`:
    las MISMAS declaraciones del decimal, y donde el `.csv` es un CSV «;», el mismo texto."""
    _requiere_calamine()
    from matrixai.training.decimal_es import es_la_cabecera_de_un_csv_punto_y_coma
    lineas = _PARIDAD_PYC[caso]
    texto = "\n".join(lineas) + "\n"
    r = convertir(_libro_de_una_columna(lineas), "x.xlsx", fila_de_cabecera=1, partir_columna_unica=";")
    canon, decl = normalize_csv_text_declarando(texto)
    assert _decimales(r.origen["declaraciones"]) == _decimales(decl)
    if es_la_cabecera_de_un_csv_punto_y_coma(lineas[0]):
        assert r.csv_text == canon
    else:
        # el `.csv` no es un CSV «;» para la entrada canónica (sale tal cual); el partido se parte, pero su decimal queda como en él
        assert canon == texto and _decimales(decl) == []
        assert [f[1] for f in csv.reader(io.StringIO(r.csv_text))][1:3] == ["1,5", "2,5"]


def test_rep2_c7_mc_la_senal_es_la_primera_linea_sin_comas_y_con_punto_y_coma():
    from matrixai.training.decimal_es import es_la_cabecera_de_un_csv_punto_y_coma as es
    assert es("a;b;c") and not es("a,b,c") and not es("a;b, c;d") and not es("abc") and not es("")
