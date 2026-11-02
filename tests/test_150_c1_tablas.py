# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C1 — abrir un Excel: de un libro a un CSV canónico, declarando lo que se hizo.

Una prueba por criterio de aceptación del contrato 150 (§3, C1) y por invariante, con el nombre de
lo que defiende. Los libros se generan con `tests/fixtures/tablas/generar.py` (solo biblioteca
estándar): LEER usa únicamente el lector del producto, nunca `openpyxl`.

Las pruebas que necesitan `python-calamine` (extra opcional `tablas`) se saltan SI FALTA, y eso
no puede pasar en silencio: `test_el_entorno_de_medicion_tiene_calamine` FALLA cuando falta y
`MATRIXAI_EXIGIR_TABLAS=1` (lo ponen los guiones de medición y C7). Las de «sin calamine el módulo
se importa y dice su motivo» corren siempre.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import zipfile

import pytest

from matrixai.tablas import (CODIGOS, Topes, TablaRechazada, convertir, detectar_formato,
                             inspeccionar)
from matrixai.tablas import api as api_tablas
from matrixai.tablas import formato as formato_tablas
from matrixai.tablas import valores as valores_tablas
from matrixai.tablas import xlsx_xml

RAIZ = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "tablas"

_spec = importlib.util.spec_from_file_location("generar_tablas_150", FIXTURES / "generar.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["generar_tablas_150"] = G
_spec.loader.exec_module(G)

try:
    import python_calamine  # noqa: F401
    HAY_CALAMINE = True
except ImportError:
    HAY_CALAMINE = False

requiere_calamine = pytest.mark.skipif(
    not HAY_CALAMINE,
    reason="falta python-calamine (extra `tablas`): pip install 'matrixai-core[tablas]'")


def leer(nombre: str) -> bytes:
    return (FIXTURES / nombre).read_bytes()


def csv_de(t) -> list[list[str]]:
    import csv
    return list(csv.reader(io.StringIO(t.csv_text)))


def tipos(t) -> list[str]:
    return [d["tipo"] for d in t.origen["declaraciones"]]


def decl(t, tipo, columna=None) -> dict:
    hay = [d for d in t.origen["declaraciones"] if d["tipo"] == tipo
           and (columna is None or d["columna"] == columna)]
    assert hay, f"falta la declaración {tipo!r}/{columna!r}: {tipos(t)}"
    return hay[0]


def rechazo(f, *a, **k) -> TablaRechazada:
    with pytest.raises(TablaRechazada) as e:
        f(*a, **k)
    return e.value


# ── El entorno ──────────────────────────────────────────────────────────────────────────
def test_el_entorno_de_medicion_tiene_calamine():
    """Que las pruebas con calamine no se salten TODAS en silencio: donde se mide, falla si falta."""
    if os.environ.get("MATRIXAI_EXIGIR_TABLAS") == "1":
        assert HAY_CALAMINE, ("MATRIXAI_EXIGIR_TABLAS=1 y python-calamine no está: las pruebas de "
                              "lectura se habrían saltado en silencio")
    elif not HAY_CALAMINE:
        pytest.skip("sin MATRIXAI_EXIGIR_TABLAS=1: este entorno no mide el extra `tablas`")


def test_sin_calamine_el_modulo_se_importa_y_dice_su_motivo(tmp_path):
    """Invariante 7: el núcleo sigue sin dependencias duras. Sin el extra, `matrixai.tablas` se
    importa, NO importa calamine, rechaza lo que no es un libro con su motivo propio y para un
    libro de verdad dice «falta el lector» — no un ImportError."""
    ejemplo = FIXTURES / "oficina_con_valor.xlsx"
    html = FIXTURES / "html_como.xls"
    codigo = f"""
import sys
sys.path[:0] = [{str(RAIZ)!r}]
sys.modules['python_calamine'] = None     # «no está instalado»
import matrixai.tablas as t
assert 'python_calamine' not in [k for k, v in sys.modules.items() if v is not None], 'importó calamine'
r = []
for f, kw in [({str(ejemplo)!r}, dict(hoja='Datos', fila_de_cabecera=3)), ({str(html)!r}, dict(fila_de_cabecera=1))]:
    try:
        t.convertir(open(f, 'rb').read(), f, **kw)
    except t.TablaRechazada as e:
        r.append(e.codigo)
print(','.join(r))
"""
    out = subprocess.run([sys.executable, "-s", "-E", "-c", codigo], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-600:]
    assert out.stdout.strip() == "falta_lector,html_no_excel"
    motivo = CODIGOS["falta_lector"]
    assert "matrixai-core[tablas]" in motivo[0] and "matrixai-core[tablas]" in motivo[1]


def test_importar_el_modulo_no_arrastra_lo_pesado():
    """`__init__` sin imports pesados: ni calamine, ni numpy, ni nada de terceros."""
    codigo = (f"import sys; sys.path[:0] = [{str(RAIZ)!r}]\n"
              "antes = set(sys.modules)\nimport matrixai.tablas\n"
              "nuevos = sorted(m.split('.')[0] for m in set(sys.modules) - antes)\n"
              "print(','.join(sorted(set(nuevos))))\n")
    out = subprocess.run([sys.executable, "-s", "-E", "-c", codigo], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-600:]
    for pesado in ("python_calamine", "numpy", "openpyxl", "pandas", "pyarrow", "torch"):
        assert pesado not in out.stdout.split(","), out.stdout


def test_el_producto_no_usa_bibliotecas_de_excel_ni_de_csv_de_terceros():
    """Criterio 8: el escritor es nuestro. Y `openpyxl`/`pandas`/`pyarrow` no están en el producto."""
    for f in (RAIZ / "matrixai" / "tablas").glob("*.py"):
        fuente = f.read_text(encoding="utf-8")
        for malo in ("import openpyxl", "from openpyxl", "import pandas", "from pandas",
                     "import pyarrow", "from pyarrow", "import xlrd", "import xlwt"):
            assert malo not in fuente, f"{f.name}: {malo}"


# ── Criterio 1: formato por FIRMA ───────────────────────────────────────────────────────
def test_firma_zip_ole_y_parquet_sin_mirar_la_extension():
    assert detectar_formato(b"PK\x03\x04rest") == "xlsx"
    assert detectar_formato(bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 20) == "ole"
    assert detectar_formato(b"PAR1\x15\x00") == "parquet"
    assert detectar_formato(b"") == "vacio"
    assert detectar_formato(b"\x89PNG\r\n\x1a\n\0\0") == "desconocido"
    assert detectar_formato(b"a,b\n1,2\n") == "texto"
    assert detectar_formato(b"\xef\xbb\xbf<HTML><body>") == "html"


@requiere_calamine
def test_un_xlsx_se_abre_aunque_se_llame_csv_o_xls():
    """La extensión no manda: la firma sí."""
    d = leer("oficina_con_valor.xlsx")
    for nombre in ("datos.csv", "datos.xls", "datos", "DATOS.XLSX"):
        t = convertir(d, nombre, hoja="Datos", fila_de_cabecera=3, filas_a_excluir=[10])
        assert t.origen["formato"] == "xlsx"
        assert t.origen["nombre"] == nombre


@pytest.mark.parametrize("fichero,codigo", [
    ("html_como.xls", "html_no_excel"),
    ("csv_renombrado.xlsx", "texto_no_excel"),
    ("truncado.xlsx", "zip_danado"),
    ("vacio.xlsx", "vacio"),
    ("cifrado.xlsx", "protegido"),
])
def test_cada_fichero_defectuoso_se_rechaza_con_su_motivo_propio(fichero, codigo):
    """Criterio 1. Ninguno llega a la persona como `BadZipFile` ni como traceback, y cada uno con
    SU motivo (no un «no se puede abrir» para todos). Corre sin calamine: se rechaza ANTES."""
    e = rechazo(inspeccionar, leer(fichero), fichero)
    assert e.codigo == codigo
    assert e.motivo_es and e.motivo_en and e.motivo_es != e.motivo_en
    assert e.__cause__ is None and (e.__context__ is None or e.__suppress_context__), \
        "la excepción de la biblioteca no se encadena"
    # convertir da lo mismo
    assert rechazo(convertir, leer(fichero), fichero, hoja="x", fila_de_cabecera=1).codigo == codigo


def test_los_motivos_de_los_defectuosos_dicen_que_hacer():
    assert "página web" in rechazo(inspeccionar, leer("html_como.xls"), "a.xls").motivo_es
    assert "web page" in rechazo(inspeccionar, leer("html_como.xls"), "a.xls").motivo_en
    assert ".csv" in rechazo(inspeccionar, leer("csv_renombrado.xlsx"), "a.xlsx").motivo_es
    assert "contraseña" in rechazo(inspeccionar, leer("cifrado.xlsx"), "a.xlsx").motivo_es
    assert "password" in rechazo(inspeccionar, leer("cifrado.xlsx"), "a.xlsx").motivo_en


def test_parquet_se_reconoce_y_dice_que_aun_no_se_abre():
    e = rechazo(inspeccionar, b"PAR1" + b"\0" * 50 + b"PAR1", "datos.parquet")
    assert e.codigo == "parquet_no_soportado"
    assert "todavía no" in e.motivo_es and "yet" in e.motivo_en


def test_un_zip_con_contrasena_se_rechaza_como_protegido():
    cifrado = G.marcar_cifrado(leer("oficina_con_valor.xlsx"))
    assert any(i.flag_bits & 1 for i in zipfile.ZipFile(io.BytesIO(cifrado)).infolist())
    assert rechazo(inspeccionar, cifrado, "a.xlsx").codigo == "protegido"


def test_un_zip_que_no_es_excel_se_rechaza_con_su_motivo():
    docx = G.zip_determinista({"word/document.xml": b"<w/>", "[Content_Types].xml": b"<T/>"})
    assert rechazo(inspeccionar, docx, "informe.xlsx").codigo == "no_es_excel"
    ods = G.zip_determinista({"mimetype": b"application/vnd.oasis.opendocument.spreadsheet",
                              "content.xml": b"<c/>"})
    e = rechazo(inspeccionar, ods, "hoja.xlsx")
    assert e.codigo == "formato_no_soportado" and ".ods" in e.motivo_es
    xlsb = G.zip_determinista({"xl/workbook.bin": b"\0\0", "[Content_Types].xml": b"<T/>"})
    assert ".xlsb" in rechazo(inspeccionar, xlsb, "b.xlsx").motivo_es


def test_el_xml_de_excel_2003_no_se_confunde_con_una_pagina_web():
    xml = (b'<?xml version="1.0"?><Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet">'
           b"</Workbook>")
    e = rechazo(inspeccionar, xml, "a.xls")
    assert e.codigo == "formato_no_soportado" and "2003" in e.motivo_es


def test_un_doc_antiguo_de_ole_no_se_toma_por_un_xls():
    ole_sin_libro = bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 600
    assert rechazo(inspeccionar, ole_sin_libro, "a.xls").codigo == "no_es_excel"


@requiere_calamine
def test_ninguna_excepcion_de_biblioteca_llega_a_la_persona():
    """Invariante 10: truncar o estropear un libro bueno en muchos sitios da SIEMPRE un libro
    bueno o una `TablaRechazada`, nunca otra excepción."""
    import random
    base = leer("oficina_con_valor.xlsx")
    r = random.Random(150)
    casos = [base[:n] for n in range(0, len(base), max(1, len(base) // 40))]
    for _ in range(60):
        b = bytearray(base)
        for _ in range(r.randint(1, 6)):
            b[r.randrange(len(b))] = r.randrange(256)
        casos.append(bytes(b))
    for c in casos:
        try:
            convertir(c, "a.xlsx", hoja="Datos", fila_de_cabecera=3)
        except TablaRechazada:
            pass


# ── Criterio 2: topes sobre el ORIGINAL, antes de abrir ─────────────────────────────────
class _NoAbrir(AssertionError):
    pass


def _prohibir_abrir(monkeypatch):
    """Si el producto llega a abrir el libro o a leer una hoja, la prueba falla."""
    def boom(*a, **k):
        raise _NoAbrir("se abrió el libro antes de comprobar los topes")
    monkeypatch.setattr(api_tablas, "_importar_calamine", boom)
    monkeypatch.setattr(xlsx_xml, "explorar_hoja", boom)
    monkeypatch.setattr(xlsx_xml, "hallazgos", boom)


def test_bomba_de_kilobytes_que_declaran_gigabytes_se_rechaza_sin_leerla(monkeypatch):
    """Criterio 2: el directorio central dice 5 GB; se rechaza con `zipfile` sin extraer ni abrir."""
    bomba = G.declarar_tamano(leer("oficina_con_valor.xlsx"), "xl/worksheets/sheet2.xml", 4_000_000_000)
    assert len(bomba) < 10_000
    _prohibir_abrir(monkeypatch)
    for f in (lambda: inspeccionar(bomba, "bomba.xlsx"),
              lambda: convertir(bomba, "bomba.xlsx", hoja="Datos", fila_de_cabecera=3)):
        e = rechazo(f)
        assert e.codigo == "bomba"
        assert e.detalle["bytes_descomprimidos"] > 4_000_000_000 and "GB" in e.motivo_es
        assert "limit" in e.motivo_en


def test_la_bomba_real_comprimida_se_rechaza_con_el_tope_del_llamante(monkeypatch):
    """Una hoja de verdad enorme que comprime a casi nada: con un tope bajo se rechaza antes de
    abrirla. (Los gigabytes de verdad no hacen falta: es el mismo tope.)"""
    relleno = b"<row r=\"1\"><c r=\"A1\"><v>1</v></c></row>" * 200_000
    cuerpo = (b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              b"<dimension ref=\"A1\"/><sheetData>" + relleno + b"</sheetData></worksheet>")
    libro = G.xlsx([G.Hoja("H", [["a"], [1]])], sobreescribir={"xl/worksheets/sheet1.xml": cuerpo})
    assert len(libro) < 200_000 < len(cuerpo)
    _prohibir_abrir(monkeypatch)
    e = rechazo(inspeccionar, libro, "b.xlsx", topes=Topes(max_descomprimido=1 << 20))
    assert e.codigo == "bomba"


def test_una_bomba_que_miente_en_su_tamano_se_rechaza_sin_abrirla(monkeypatch):
    """Medido: calamine NO se fía del tamaño declarado (declarando 1.000 B lee los 147 KB), así
    que la suma de tamaños declarados no basta. El flujo se descomprime con tope por miembro."""
    mentirosa = G.declarar_tamano(G.xlsx([G.Hoja("H", [["a", "b"]] + [[i, i] for i in range(3000)])]),
                                  "xl/worksheets/sheet1.xml", 1000)
    _prohibir_abrir(monkeypatch)
    e = rechazo(inspeccionar, mentirosa, "m.xlsx")
    assert e.codigo == "zip_danado"


def test_el_directorio_con_mas_filas_que_excel_se_rechaza_sin_leer_las_celdas(monkeypatch):
    """La bomba del inventario (20,7 MB → 309 MB, 2,5 M de filas): declara más filas de las que
    Excel admite. Se rechaza por la cabecera de la hoja, sin recorrer las celdas."""
    cuerpo = (b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              b'<dimension ref="A1:AD2500000"/><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>x</t>'
              b"</is></c></row></sheetData></worksheet>")
    libro = G.xlsx([G.Hoja("H", [["a"]])], sobreescribir={"xl/worksheets/sheet1.xml": cuerpo})
    vistas = []
    original = xlsx_xml._procesar_trozo
    monkeypatch.setattr(xlsx_xml, "_procesar_trozo", lambda *a, **k: vistas.append(1) or original(*a, **k))
    # inspeccionar lista el error en su hoja; convertir lo da entero
    hoja = inspeccionar(libro, "grande.xlsx")["hojas"][0]
    assert hoja["error"]["codigo"] == "demasiadas_filas" and hoja["filas"] is None
    e = rechazo(convertir, libro, "grande.xlsx", hoja="H", fila_de_cabecera=1)
    assert e.codigo == "demasiadas_filas" and e.detalle["filas"] == 2_500_000
    assert not vistas, "se recorrieron las celdas antes de rechazar"


@requiere_calamine
def test_tope_de_filas_y_de_columnas_del_llamante():
    d = leer("oficina_con_valor.xlsx")
    e = rechazo(convertir, d, "o.xlsx", hoja="Datos", fila_de_cabecera=3, topes=Topes(max_filas=5))
    assert e.codigo == "demasiadas_filas" and e.detalle["tope"] == 5
    e = rechazo(convertir, d, "o.xlsx", hoja="Datos", fila_de_cabecera=3, topes=Topes(max_columnas=3))
    assert e.codigo == "demasiadas_columnas"
    # y en inspeccionar, una hoja que no cabe no tumba a las demás
    r = inspeccionar(d, "o.xlsx", topes=Topes(max_filas=5))
    por_nombre = {h["nombre"]: h for h in r["hojas"]}
    assert por_nombre["Datos"]["error"]["codigo"] == "demasiadas_filas"
    assert "error" not in por_nombre["Portada"]


@requiere_calamine
def test_la_guarda_de_memoria_es_la_que_ya_existe_no_una_segunda(monkeypatch):
    """Invariante 6: se le pregunta a `limits.memoria_insuficiente` (la de los CSV), y su respuesta
    (poca RAM) rechaza ANTES de que el lector reserve el rectángulo."""
    from matrixai import limits
    llamadas = []
    real = limits.memoria_insuficiente
    monkeypatch.setattr(limits, "_memoria_disponible_del_sistema", lambda: 100 * 1024 * 1024)
    monkeypatch.setattr(limits, "memoria_insuficiente",
                        lambda n: llamadas.append(n) or real(n))
    _prohibir_calamine = lambda *a, **k: (_ for _ in ()).throw(_NoAbrir("se cargó la hoja"))
    # la celda suelta en Z1048576 reserva 27 M de celdas: 1,1 GB > los 100 MB «disponibles»
    monkeypatch.setattr(api_tablas._Libro, "calamine", _prohibir_calamine)
    e = rechazo(convertir, leer("celda_perdida.xlsx"), "v.xlsx", hoja="Enero", fila_de_cabecera=1)
    assert e.codigo == "memoria_insuficiente"
    assert llamadas, "no se consultó limits.memoria_insuficiente"
    assert "GB" in e.motivo_es and "MB" in e.motivo_es
    assert e.motivo_en != e.motivo_es


@requiere_calamine
def test_la_guarda_de_memoria_se_puede_apagar_como_la_de_los_csv(monkeypatch):
    from matrixai import limits
    monkeypatch.setattr(limits, "_memoria_disponible_del_sistema", lambda: 1)
    monkeypatch.setenv("MATRIXAI_GUARDIA_MEMORIA", "0")
    t = convertir(leer("celda_perdida.xlsx"), "v.xlsx", hoja="Enero", fila_de_cabecera=1)
    assert t.csv_text.startswith("x,y,columna_26\n")


# ── Criterio 3: rango real ──────────────────────────────────────────────────────────────
@requiere_calamine
def test_una_celda_perdida_lejos_no_produce_un_millon_de_filas_vacias():
    """Criterio 3: `Z1048576` no llena el CSV de filas vacías: se quitan y se DECLARAN."""
    t = convertir(leer("celda_perdida.xlsx"), "v.xlsx", hoja="Enero", fila_de_cabecera=1)
    filas = csv_de(t)
    assert len(filas) == 1 + 5 + 1, "cabecera + 5 filas + la fila de la celda suelta"
    assert filas[-1][-1] == "fin"
    d = decl(t, "filas_vacias_quitadas")
    assert d["cuantas"] == 1_048_576 - 7           # cada fila sin una sola celda con valor
    assert decl(t, "columnas_vacias_quitadas")["cuantas"] == 23


@requiere_calamine
def test_la_dimension_inflada_y_las_celdas_con_solo_estilo_no_cuentan():
    """Una `<dimension>` que dice A1:XFD1048576 y celdas vacías con formato lejos no ensanchan nada."""
    base = G.xlsx([G.Hoja("H", [["a", "b"], [1, 2], [3, 4]], dimension="A1:XFD1048576")])
    libro = G.reescribir_hoja(base, 1, lambda x: x.replace(
        "</sheetData>", '<row r="1048576" s="1" customFormat="1"><c r="Z1048576" s="1"/></row></sheetData>'))
    t = convertir(libro, "i.xlsx", hoja="H", fila_de_cabecera=1)
    assert t.csv_text == "a,b\n1,2\n3,4\n"
    assert t.origen["rango"] == "A1:B3"
    assert tipos(t) == []


def test_la_geometria_cuenta_solo_celdas_con_valor():
    """Lo que la guarda de memoria mide: el rectángulo de las celdas CON valor. Una `<dimension>`
    inflada o una celda vacía con formato lejos no cuentan; una celda con dato lejos, sí."""
    def geo(libro, n=1):
        zf = zipfile.ZipFile(io.BytesIO(libro))
        ruta = xlsx_xml.mapa_de_hojas(zf)[n - 1]["ruta"]
        return xlsx_xml.explorar_hoja(zf, ruta, frozenset(), 10**7, 10**5)
    base = G.xlsx([G.Hoja("H", [["a", "b"], [1, 2], [3, 4]], dimension="A1:XFD1048576")])
    inflada = G.reescribir_hoja(base, 1, lambda x: x.replace(
        "</sheetData>", '<row r="1048576"><c r="Z1048576" s="1"/></row></sheetData>'))
    g = geo(inflada)
    assert (g.min_fila, g.max_fila, g.min_col, g.max_col, g.celdas_del_rango) == (1, 3, 1, 2, 6)
    assert g.exacta
    g = geo(leer("celda_perdida.xlsx"))
    assert (g.filas, g.columnas, g.celdas_con_valor) == (1_048_576, 26, 13)
    g = geo(leer("varias_hojas.xlsx"))
    assert g.hay_formulas is False and g.n_filas_ocultas == 1 and g.columnas_ocultas == [2]
    g = geo(leer("tipos_raros.xlsx"))
    assert g.hay_errores and not g.hay_formulas
    g = geo(leer("oficina_con_valor.xlsx"), 2)
    assert g.hay_formulas and (g.min_fila, g.max_fila, g.min_col, g.max_col) == (1, 10, 1, 6)
    assert geo(G.xlsx([G.Hoja("H", [])])).vacia


@requiere_calamine
def test_una_hoja_sin_datos_se_dice_y_no_revienta():
    d = G.xlsx([G.Hoja("Vacía", []), G.Hoja("Con", [["a"], [1]])])
    r = inspeccionar(d, "v.xlsx")
    vacia = r["hojas"][0]
    assert vacia["filas"] == 0 and any(a["tipo"] == "hoja_sin_datos" for a in vacia["avisos"])
    assert rechazo(convertir, d, "v.xlsx", hoja="Vacía", fila_de_cabecera=1).codigo == "hoja_vacia"
    assert convertir(d, "v.xlsx", hoja=1, fila_de_cabecera=1).csv_text == "a\n1\n"   # por posición


# ── Criterio 4: inspeccionar ────────────────────────────────────────────────────────────
@requiere_calamine
def test_inspeccionar_lista_las_hojas_y_marca_las_ocultas():
    r = inspeccionar(leer("varias_hojas.xlsx"), "v.xlsx")
    estados = {h["nombre"]: (h["estado"], h["visible"]) for h in r["hojas"]}
    assert estados == {"Enero": ("visible", True), "Febrero": ("visible", True),
                       "Calculos": ("oculta", False), "Secreta": ("muy_oculta", False)}
    assert r["hoja_propuesta"] is None, "con dos hojas visibles no se elige por la persona"
    oculta = next(h for h in r["hojas"] if h["nombre"] == "Calculos")
    assert oculta["primeras"] == [] and oculta["filas"] == 2 and oculta["columnas"] == 1
    assert any(a["tipo"] == "hoja_oculta" for a in oculta["avisos"])
    assert r["formato"] == "xlsx" and r["bytes"] == len(leer("varias_hojas.xlsx"))


@requiere_calamine
def test_inspeccionar_con_una_sola_hoja_visible_la_propone():
    r = inspeccionar(leer("cabecera_dos_filas.xlsx"), "c.xlsx")
    assert r["hoja_propuesta"] == "Ventas"


@requiere_calamine
def test_inspeccionar_da_primeras_8_y_ultimas_5_filas_con_su_numero():
    d = G.xlsx([G.Hoja("H", [["a", "b"]] + [[i, i * 2] for i in range(1, 31)])])
    h = inspeccionar(d, "h.xlsx")["hojas"][0]
    assert [f["fila"] for f in h["primeras"]] == list(range(1, 9))
    assert [f["fila"] for f in h["ultimas"]] == [27, 28, 29, 30, 31]
    assert h["ultimas"][-1]["celdas"] == ["30", "60"]
    assert h["filas"] == 31 and h["columnas"] == 2 and h["rango"] == "A1:B31"
    corta = inspeccionar(G.xlsx([G.Hoja("H", [["a"], [1], [2]])]), "c.xlsx")["hojas"][0]
    assert [f["fila"] for f in corta["primeras"]] == [1, 2, 3] and corta["ultimas"] == []


@requiere_calamine
@pytest.mark.parametrize("fichero,hoja,esperada", [
    ("oficina_con_valor.xlsx", "Datos", 3),
    ("cabecera_en_fila_15.xlsx", "Cuadro", 15),
    ("erp_totales.xlsx", "Pedidos", 1),
    ("nombres.xlsx", "Año", 1),
])
def test_inspeccionar_propone_la_fila_de_cabecera(fichero, hoja, esperada):
    """Primera fila con ≥2 textos distintos y bastante llena, seguida de datos. Propone."""
    h = next(h for h in inspeccionar(leer(fichero), fichero)["hojas"] if h["nombre"] == hoja)
    assert h["fila_de_cabecera_propuesta"] == esperada


@requiere_calamine
def test_inspeccionar_propone_las_filas_de_total_en_espanol_e_ingles_sin_tildes():
    r = inspeccionar(leer("erp_totales.xlsx"), "e.xlsx")["hojas"][0]
    por_fila = {c["fila"]: c for c in r["filas_candidatas_a_total"]}
    assert set(por_fila) == {4, 7, 9, 10}, "Subtotal ×2, Total general y Promedio; la fila vacía no"
    assert por_fila[4]["texto"] == "Subtotal Ana" and por_fila[4]["motivo"] == "palabra"
    assert por_fila[9]["texto"] == "Total general"
    # inglés, mayúsculas y tildes: «Résumé» no, pero «TOTAL», «Grand Total» y «Média» sí
    d = G.xlsx([G.Hoja("H", [["k", "v"], ["a", 1], ["b", 2], ["Grand Total", 3], ["MEDIA", 2],
                              ["Média", 1], ["Totalmente ajeno", 4], ["Summary", 9]])])
    cands = {c["fila"] for c in inspeccionar(d, "h.xlsx")["hojas"][0]["filas_candidatas_a_total"]}
    assert cands == {4, 5, 6, 8}


@requiere_calamine
def test_inspeccionar_propone_las_ultimas_con_la_mitad_de_celdas_vacias():
    filas = [["a", "b", "c", "d"]] + [[i, i, i, i] for i in range(10)] + [[None, None, None, None],
                                                                          [564, None, None, 23]]
    h = inspeccionar(G.xlsx([G.Hoja("H", filas)]), "h.xlsx")["hojas"][0]
    assert [(c["fila"], c["motivo"]) for c in h["filas_candidatas_a_total"]] == [(13, "mitad_vacia")]


@requiere_calamine
def test_inspeccionar_avisa_de_combinadas_ocultas_mixtas_serie_y_formulas():
    r = inspeccionar(leer("oficina_con_valor.xlsx"), "o.xlsx")
    datos = next(h for h in r["hojas"] if h["nombre"] == "Datos")
    av = {a["tipo"]: a for a in datos["avisos"]}
    assert av["hay_celdas_combinadas"]["cuantas"] == 2
    assert "B4:B5" in av["hay_celdas_combinadas"]["detalle"]["rangos"]
    tr = inspeccionar(leer("tipos_raros.xlsx"), "t.xlsx")["hojas"][0]
    av = {(a["tipo"], a["columna"]): a for a in tr["avisos"]}
    assert ("posible_fecha_en_serie", "B") in av
    assert ("columna_mixta", "I") in av
    assert ("errores_de_celda", "H") in av and av[("errores_de_celda", "H")]["cuantas"] == 2
    assert ("formato_porcentaje", "D") in av
    oc = inspeccionar(leer("varias_hojas.xlsx"), "v.xlsx")["hojas"][0]
    tipos_oc = {a["tipo"] for a in oc["avisos"]}
    assert {"filas_ocultas", "columnas_ocultas"} <= tipos_oc
    sin = inspeccionar(leer("oficina_sin_valor.xlsx"), "s.xlsx")
    f = next(a for a in next(h for h in sin["hojas"] if h["nombre"] == "Datos")["avisos"]
             if a["tipo"] == "formulas_sin_valor")
    assert f["cuantas"] == 8 and f["detalle"]["celdas"][:2] == ["E4", "E5"]


@requiere_calamine
def test_inspeccionar_avisa_de_nombres_repetidos_y_de_cabecera_de_dos_filas():
    n = inspeccionar(leer("nombres.xlsx"), "n.xlsx")["hojas"][0]
    a = next(a for a in n["avisos"] if a["tipo"] == "nombres_repetidos")
    assert a["cuantas"] == 3     # AC repetido y dos columnas con datos y sin nombre
    d = inspeccionar(leer("cabecera_dos_filas.xlsx"), "c.xlsx")["hojas"][0]
    assert any(a["tipo"] == "posible_cabecera_de_dos_filas" for a in d["avisos"])


@requiere_calamine
def test_inspeccionar_propone_y_no_decide_no_devuelve_csv_ni_quita_nada():
    r = inspeccionar(leer("oficina_con_valor.xlsx"), "o.xlsx")
    assert "csv_text" not in json.dumps(r) and "csv_bytes" not in r
    datos = next(h for h in r["hojas"] if h["nombre"] == "Datos")
    assert datos["filas_candidatas_a_total"][0]["fila"] == 10
    assert datos["ultimas"][-1]["celdas"][0] == "TOTAL", "el total se ENSEÑA; no se quita"
    json.dumps(r, ensure_ascii=False)       # y todo es serializable (viaja por JSON)


# ── Criterio 5: convertir no quita nada que la persona no haya listado ──────────────────
@requiere_calamine
def test_convertir_no_quita_una_fila_total_por_su_cuenta():
    """Invariante 4: la fila TOTAL llega al CSV si la persona no la lista, por muy total que parezca."""
    t = convertir(leer("oficina_con_valor.xlsx"), "o.xlsx", hoja="Datos", fila_de_cabecera=3)
    filas = csv_de(t)
    assert filas[-1][0] == "TOTAL" and len(filas) == 1 + 6 + 1
    assert t.origen["filas_excluidas"] == []
    assert "filas_excluidas" not in tipos(t)


@requiere_calamine
def test_filas_excluidas_se_declaran_con_su_numero_y_el_texto_de_su_primera_celda():
    t = convertir(leer("erp_totales.xlsx"), "e.xlsx", hoja="Pedidos", fila_de_cabecera=1,
                  filas_a_excluir=[9, 4, 7, 10])
    assert t.origen["filas_excluidas"] == [{"fila": 4, "texto": "Subtotal Ana"},
                                          {"fila": 7, "texto": "Subtotal Luis"},
                                          {"fila": 9, "texto": "Total general"},
                                          {"fila": 10, "texto": "Promedio"}]
    assert csv_de(t)[1:] == [["Ana", "A", "1", "10"], ["Ana", "B", "2", "20"],
                             ["Luis", "A", "4", "40"], ["Luis", "B", "5", "50"]]
    d = decl(t, "filas_excluidas")
    assert d["cuantas"] == 4 and d["detalle"]["filas"] == [4, 7, 9, 10]


@requiere_calamine
def test_solo_se_quitan_las_filas_listadas_y_las_vacias_quedan_declaradas():
    t = convertir(leer("erp_totales.xlsx"), "e.xlsx", hoja="Pedidos", fila_de_cabecera=1,
                  filas_a_excluir=[4])
    assert len(csv_de(t)) == 1 + 4 + 1 + 2     # los 4 pedidos, el otro subtotal, total y promedio
    v = decl(t, "filas_vacias_quitadas")
    assert v["cuantas"] == 1 and v["detalle"]["filas"] == [8]


@requiere_calamine
@pytest.mark.parametrize("kw", [
    dict(filas_a_excluir=[3]),                    # la cabecera
    dict(filas_a_excluir=[1]),                    # por encima de la cabecera
    dict(filas_a_excluir=["x"]),
    dict(filas_a_excluir=[0]),
    dict(fila_de_cabecera=0),
    dict(filas_de_cabecera=3),
    dict(fila_de_cabecera=99),
    dict(fila_de_cabecera=10),                    # sin filas de datos debajo
])
def test_parametros_invalidos_se_rechazan_con_motivo(kw):
    base = dict(hoja="Datos", fila_de_cabecera=3)
    base.update(kw)
    e = rechazo(convertir, leer("oficina_con_valor.xlsx"), "o.xlsx", **base)
    assert e.codigo in {"parametro_invalido", "cabecera_invalida", "sin_filas_de_datos"}


@requiere_calamine
def test_la_hoja_hay_que_elegirla_y_no_se_unen_varias():
    d = leer("varias_hojas.xlsx")
    e = rechazo(convertir, d, "v.xlsx", fila_de_cabecera=1)
    assert e.codigo == "hoja_requerida" and "Enero" in e.motivo_es and "Febrero" in e.motivo_es
    assert "Calculos" not in e.motivo_es, "las ocultas no se ofrecen"
    assert rechazo(convertir, d, "v.xlsx", hoja="Nope", fila_de_cabecera=1).codigo == "hoja_no_encontrada"
    oculta = convertir(d, "v.xlsx", hoja="Calculos", fila_de_cabecera=1)   # pedirla a mano se puede…
    assert any(x["tipo"] == "hoja_oculta" for x in oculta.origen["declaraciones"])  # …y se declara


# ── Criterio 6: fórmulas ────────────────────────────────────────────────────────────────
@requiere_calamine
def test_formula_sin_valor_rechaza_con_recuento_y_celdas():
    """DA3: nunca se calcula. Sin valor guardado en lo que se convierte → rechazo, no vacío."""
    e = rechazo(convertir, leer("oficina_sin_valor.xlsx"), "o.xlsx", hoja="Datos", fila_de_cabecera=3)
    assert e.codigo == "formula_sin_valor"
    assert e.detalle["cuantas"] == 8 and e.detalle["celdas"][:3] == ["E4", "E5", "E6"]
    assert "Excel" in e.motivo_es and "Excel" in e.motivo_en and "guárdalo" in e.motivo_es


@requiere_calamine
def test_formula_sin_valor_lista_como_mucho_diez_celdas():
    filas = [["a", "b"]] + [[i, G.Formula(f"A{i + 2}*2")] for i in range(25)]
    e = rechazo(convertir, G.xlsx([G.Hoja("H", filas)]), "f.xlsx", hoja="H", fila_de_cabecera=1)
    assert e.detalle["cuantas"] == 25 and len(e.detalle["celdas"]) == 10
    assert e.detalle["celdas"][0] == "B2"


@requiere_calamine
def test_una_formula_sin_valor_en_una_fila_excluida_no_rechaza():
    """El `=SUMA` de la fila TOTAL que la persona quita no impide convertir…"""
    d = leer("oficina_total_sin_valor.xlsx")
    t = convertir(d, "o.xlsx", hoja="Datos", fila_de_cabecera=3, filas_a_excluir=[10])
    assert csv_de(t)[-1][0] == "2024-01-10"
    # …pero si no la quita, rechaza y nombra las celdas de ESA fila
    e = rechazo(convertir, d, "o.xlsx", hoja="Datos", fila_de_cabecera=3)
    assert e.detalle["celdas"] == ["C10", "E10"]


@requiere_calamine
def test_se_lee_siempre_el_valor_guardado_nunca_se_calcula_la_formula():
    """La fórmula dice 1+1 y el valor guardado 5: sale 5."""
    d = G.xlsx([G.Hoja("H", [["x", "y"], [1, G.Formula("1+1", 5)], [2, G.Formula("A3*2", 4.5)]])])
    assert csv_de(convertir(d, "f.xlsx", hoja="H", fila_de_cabecera=1))[1:] == [["1", "5"], ["2", "4.5"]]


@requiere_calamine
def test_una_formula_con_cadena_vacia_guardada_no_es_una_formula_sin_valor():
    d = G.xlsx([G.Hoja("H", [["x", "y"], [1, G.Formula('""', "")], [2, G.Formula("A3", 2)]])])
    t = convertir(d, "f.xlsx", hoja="H", fila_de_cabecera=1)
    assert csv_de(t)[1:] == [["1", ""], ["2", "2"]]


@requiere_calamine
def test_en_xls_se_declara_que_no_se_puede_comprobar():
    """Criterio 6: «no comprobable en .xls», no se calla."""
    t = convertir(leer("antiguo.xls"), "a.xls", hoja="Datos", fila_de_cabecera=1)
    d = decl(t, "no_comprobable_en_xls")
    assert "fórmulas sin valor guardado" in d["detalle"]["que"]
    assert "old .xls" in d["texto"]["en"] and "xls" in d["texto"]["es"]
    r = inspeccionar(leer("antiguo.xls"), "a.xls")
    assert r["formato"] == "xls" and any(a["tipo"] == "no_comprobable_en_xls" for a in r["avisos"])


# ── Criterio 7: valores ─────────────────────────────────────────────────────────────────
def test_valor_a_texto_enteros_no_finitos_booleanos_y_vacios():
    v = valores_tablas.valor_a_texto
    assert v(10.0) == ("10", None) and v(-0.0) == ("0", None) and v(2.5) == ("2.5", None)
    assert v(32.400000000000006) == ("32.400000000000006", None)
    assert v(1e20)[0] == "1e+20" and v(7) == ("7", None)
    assert v(float("nan")) == ("", "no_finito") and v(float("inf")) == ("", "no_finito")
    assert v(True) == ("true", None) and v(False) == ("false", None)
    assert v(None) == ("", None) and v("") == ("", None)
    import datetime as dt
    assert v(dt.date(2024, 1, 5)) == ("2024-01-05", None)
    assert v(dt.datetime(2024, 1, 5, 13, 30, 15)) == ("2024-01-05 13:30:15", None)
    assert v(dt.datetime(2024, 1, 5, 1, 0, 0, 123000)) == ("2024-01-05 01:00:00", "fecha_truncada")
    assert v(dt.time(23, 59, 59)) == ("23:59:59", None)
    assert v(dt.timedelta(days=1, hours=2, minutes=24)) == ("26:24:00", None)


@requiere_calamine
def test_los_enteros_salen_sin_decimales_y_el_vacio_no_es_un_cero():
    t = convertir(leer("ceros_y_vacios.xlsx"), "c.xlsx", hoja="Datos", fila_de_cabecera=1)
    assert csv_de(t) == [["a", "b"], ["1", ""], ["3", "4"], ["10", "2.5"]]
    assert "10.0" not in t.csv_text and ",0\n" not in t.csv_text


@requiere_calamine
def test_ceros_a_la_izquierda_y_1e3_se_conservan_como_texto():
    t = convertir(leer("tipos_raros.xlsx"), "t.xlsx", hoja="Hoja1", fila_de_cabecera=1)
    fila = {k: v for k, v in zip(csv_de(t)[0], csv_de(t)[1])}
    assert (fila["num_texto"], csv_de(t)[2][5], csv_de(t)[3][5]) == ("007", "012", "1E3")
    assert fila["coma_texto"] == "1,5" and csv_de(t)[3][4] == "1.234,56"
    assert '"1,5"' in t.csv_text, "el texto con coma va entrecomillado, no partido"


@requiere_calamine
def test_booleanos_y_errores_de_celda_contados_por_columna():
    t = convertir(leer("booleanos_y_errores.xlsx"), "b.xlsx", hoja="Datos", fila_de_cabecera=1)
    assert csv_de(t) == [["a", "b", "c"], ["1", "true", ""], ["2", "false", ""], ["3", "true", "5"]]
    d = decl(t, "errores_de_celda", "c")
    assert d["cuantas"] == 2 and d["detalle"]["tipos"] == ["#DIV/0!", "#N/A"]
    t2 = convertir(leer("tipos_raros.xlsx"), "t.xlsx", hoja="Hoja1", fila_de_cabecera=1)
    assert decl(t2, "errores_de_celda", "error")["cuantas"] == 2


@requiere_calamine
def test_porcentaje_es_el_valor_guardado_y_la_columna_se_declara():
    t = convertir(leer("tipos_raros.xlsx"), "t.xlsx", hoja="Hoja1", fila_de_cabecera=1)
    assert [f[3] for f in csv_de(t)[1:]] == ["0.153", "0.2", "1"]
    assert decl(t, "formato_porcentaje", "pct")["cuantas"] == 3


@requiere_calamine
def test_hora_del_dia_y_duracion_en_hhmmss_declaradas():
    t = convertir(leer("fechas_1900.xlsx"), "f.xlsx", hoja="Fechas", fila_de_cabecera=1)
    filas = csv_de(t)
    assert [f[2] for f in filas[1:3]] == ["13:30:00", "23:59:59"]
    assert [f[3] for f in filas[1:3]] == ["26:24:00", "06:00:00"]
    assert decl(t, "hora_del_dia", "hora")["cuantas"] == 2
    assert decl(t, "duracion", "duracion")["cuantas"] == 2


@requiere_calamine
def test_fechas_iso_sin_fracciones_y_truncado_declarado():
    t = convertir(leer("fechas_1900.xlsx"), "f.xlsx", hoja="Fechas", fila_de_cabecera=1)
    fechas = [f[0] for f in csv_de(t)[1:]]
    # una columna mezcla fechas sin hora y una con hora (y fracción): UN formato, y se declara
    assert fechas == ["2024-01-05 00:00:00", "2024-03-01 00:00:00", "2024-03-01 00:00:00"]
    assert decl(t, "fecha_truncada", "fecha")["cuantas"] == 1
    assert decl(t, "fechas_a_medianoche", "fecha")["cuantas"] == 2
    assert [f[1] for f in csv_de(t)[1:3]] == ["2024-01-05 13:30:15", "2024-03-01 08:00:00"]


@requiere_calamine
def test_las_epocas_1900_y_1904_dan_las_mismas_fechas():
    """Criterio 7: épocas correctas. Mismas fechas escritas en las dos épocas → el mismo CSV."""
    a = convertir(leer("fechas_1900.xlsx"), "a.xlsx", hoja="Fechas", fila_de_cabecera=1)
    b = convertir(leer("fechas_1904.xlsx"), "b.xlsx", hoja="Fechas", fila_de_cabecera=1)
    assert a.csv_bytes == b.csv_bytes
    assert "2024-01-05" in a.csv_text
    # y el fichero 1904 de verdad lleva el calendario 1904 (si no, la prueba no mide nada)
    wb = zipfile.ZipFile(io.BytesIO(leer("fechas_1904.xlsx"))).read("xl/workbook.xml")
    assert b'date1904="1"' in wb
    serie = zipfile.ZipFile(io.BytesIO(leer("fechas_1904.xlsx"))).read("xl/worksheets/sheet1.xml")
    assert b"<v>43834.0</v>" in serie, "2024-01-05 en la época 1904 es el 43834, no el 45296"


@requiere_calamine
def test_serie_sin_formato_de_fecha_no_se_convierte_solo_avisa():
    """DS7: `45292` sin formato de fecha sigue siendo un número; el aviso solo salta si TODA la
    columna cae entre 20.000 y 60.000."""
    t = convertir(leer("tipos_raros.xlsx"), "t.xlsx", hoja="Hoja1", fila_de_cabecera=1)
    assert [f[1] for f in csv_de(t)[1:]] == ["45292", "45293.75", ""]
    assert decl(t, "posible_fecha_en_serie", "serie_sin_formato")["cuantas"] == 2
    d = G.xlsx([G.Hoja("H", [["a", "b"], [45292, 5], [10, 6]])])
    t2 = convertir(d, "s.xlsx", hoja="H", fila_de_cabecera=1)
    assert "posible_fecha_en_serie" not in tipos(t2)
    d = G.xlsx([G.Hoja("H", [["a", "b"], [45292, 5], [45293, 6]])])
    assert "posible_fecha_en_serie" in tipos(convertir(d, "s.xlsx", hoja="H", fila_de_cabecera=1))


@requiere_calamine
def test_columna_mixta_se_declara_y_no_se_arregla():
    t = convertir(leer("tipos_raros.xlsx"), "t.xlsx", hoja="Hoja1", fila_de_cabecera=1)
    d = decl(t, "columna_mixta", "mixta")
    assert d["detalle"]["tipos"] == {"fecha": 1, "numero": 1, "texto": 1}
    assert [f[8] for f in csv_de(t)[1:]] == ["3", "tres", "2024-01-01"]


# ── DS4–DS6: combinadas, nombres, cabecera de dos filas ─────────────────────────────────
@requiere_calamine
def test_combinadas_llegan_vacias_y_se_declaran_con_su_rango():
    t = convertir(leer("oficina_con_valor.xlsx"), "o.xlsx", hoja="Datos", fila_de_cabecera=3,
                  filas_a_excluir=[10])
    assert [f[1] for f in csv_de(t)[1:]] == ["Madrid", "", "", "Bilbao", "Bilbao", "Sevilla"]
    d = decl(t, "celdas_combinadas_vacias", "Tienda")
    assert d["cuantas"] == 1 and d["detalle"]["rangos"] == ["B4:B5"]
    assert "combinadas_rellenadas" not in tipos(t), "rellenar solo si la persona lo marca"


@requiere_calamine
def test_rellenar_combinadas_solo_si_la_persona_lo_marca_y_se_declara():
    t = convertir(leer("oficina_con_valor.xlsx"), "o.xlsx", hoja="Datos", fila_de_cabecera=3,
                  filas_a_excluir=[10], rellenar_combinadas=True)
    assert [f[1] for f in csv_de(t)[1:]] == ["Madrid", "Madrid", "", "Bilbao", "Bilbao", "Sevilla"]
    assert decl(t, "combinadas_rellenadas", "Tienda")["cuantas"] == 1
    assert "celdas_combinadas_vacias" not in tipos(t)


@requiere_calamine
def test_columnas_vacias_se_quitan_repetidos_se_renombran_y_vacios_con_datos_se_nombran():
    t = convertir(leer("nombres.xlsx"), "n.xlsx", hoja="Año", fila_de_cabecera=1)
    assert csv_de(t)[0] == ["AC", "AC_2", "columna_3", "Niño", "vacía", "Precio €", "columna_7"]
    assert "señor" in t.csv_text and "pingüino" in t.csv_text and "ñandú" in t.csv_text
    d = decl(t, "nombre_repetido_renombrado", "AC_2")
    assert d["detalle"]["original"] == "AC" and d["detalle"]["letra"] == "B"
    assert decl(t, "nombre_vacio_con_datos", "columna_3")["detalle"]["letra"] == "C"
    # la columna con nombre pero sin datos se QUEDA (nada de lo que tiene nombre se tira)
    assert "vacía" in csv_de(t)[0]
    # una columna sin nombre ni datos se quita, declarada
    d2 = G.xlsx([G.Hoja("H", [["a", None, "c"], [1, None, 3]])])
    t2 = convertir(d2, "n.xlsx", hoja="H", fila_de_cabecera=1)
    assert csv_de(t2) == [["a", "c"], ["1", "3"]]
    assert decl(t2, "columnas_vacias_quitadas")["detalle"]["letras"] == ["B"]


@requiere_calamine
def test_un_nombre_nuevo_no_choca_con_uno_que_ya_existe():
    d = G.xlsx([G.Hoja("H", [["AC", "AC", "AC_2"], [1, 2, 3]])])
    assert csv_de(convertir(d, "n.xlsx", hoja="H", fila_de_cabecera=1))[0] == ["AC", "AC_3", "AC_2"]


@requiere_calamine
def test_cabecera_de_dos_filas_une_grupo_y_subcolumna_y_lo_declara():
    t = convertir(leer("cabecera_dos_filas.xlsx"), "c.xlsx", hoja="Ventas", fila_de_cabecera=1,
                  filas_de_cabecera=2)
    assert csv_de(t) == [["Producto", "Ventas Enero", "Ventas Febrero", "Coste Enero", "Coste Febrero"],
                         ["Tornillo", "10", "12", "4", "5"], ["Tuerca", "7", "9", "3", "4"],
                         ["Arandela", "5", "6", "2", "2"]]
    d = decl(t, "cabecera_compuesta", "Ventas Febrero")
    assert d["detalle"] == {"grupo": "Ventas", "sub": "Febrero"}, "el grupo se arrastra por la combinada"
    assert t.origen["filas_de_cabecera"] == 2
    assert "Producto" not in [x["columna"] for x in t.origen["declaraciones"] if x["tipo"] == "cabecera_compuesta"]


@requiere_calamine
def test_lo_que_hay_encima_de_la_cabecera_se_declara_como_omitido():
    t = convertir(leer("cabecera_en_fila_15.xlsx"), "c.xlsx", hoja="Cuadro", fila_de_cabecera=15)
    d = decl(t, "filas_previas_omitidas")
    assert d["cuantas"] == 4 and d["detalle"]["textos"][0] == "Cuadro de mando"
    assert t.origen["rango"] == "A15:D19"


@requiere_calamine
def test_filas_y_columnas_ocultas_se_incluyen_y_se_declaran():
    t = convertir(leer("varias_hojas.xlsx"), "v.xlsx", hoja="Enero", fila_de_cabecera=1)
    assert decl(t, "filas_ocultas")["detalle"]["filas"] == [3]
    assert decl(t, "columnas_ocultas")["detalle"]["letras"] == ["B"]
    assert csv_de(t)[2] == ["1", "2"], "la fila 3 y la columna B ocultas están en el CSV"


@requiere_calamine
def test_macros_no_se_ejecutan_y_se_declaran():
    t = convertir(leer("macro.xlsm"), "m.xlsm", hoja="Datos", fila_de_cabecera=1)
    assert t.origen["formato"] == "xlsm" and "macros_ignoradas" in tipos(t)
    assert t.csv_text == "a,b\n1,2\n"


# ── Criterio 8: escritor determinista ───────────────────────────────────────────────────
GOLDEN = {
    # sha256 del CSV de cada libro con python-calamine 0.9.0 (medido el 2026-11-02). Si cambia, o
    # cambió el lector (re-medir el Paso 0 al actualizarlo, y revisar) o cambió el escritor — y
    # entonces cambia el hash que todo verifica.
    "oficina_con_valor.xlsx": (dict(hoja="Datos", fila_de_cabecera=3, filas_a_excluir=[10]),
                               "bd6d8c0422be1cc451491170ee15c61c4961dca10dd288e958d1f7d5ef9bd82c"),
    "tipos_raros.xlsx": (dict(hoja="Hoja1", fila_de_cabecera=1),
                         "d7392b9c10a044c1e0b299bbff2475d0c6bc71904ed5b82d7f6b4717b25953d6"),
    "cabecera_dos_filas.xlsx": (dict(hoja="Ventas", fila_de_cabecera=1, filas_de_cabecera=2),
                                "4d919ae43f024221787d20fe0dbc4853ef6affbb8ee20decbe9b6eac6d3eb007"),
    "antiguo.xls": (dict(hoja="Datos", fila_de_cabecera=1),
                    "2c28441ee503d1c62400b3ab57df3376d69ef95a365aed6bdf4753d61b9cf083"),
}


@requiere_calamine
@pytest.mark.parametrize("fichero", sorted(GOLDEN))
def test_el_csv_de_cada_libro_tiene_el_hash_medido(fichero):
    """Los bytes del CSV son lo que todo hashea y verifica: no pueden moverse sin que se note."""
    kw, esperado = GOLDEN[fichero]
    assert hashlib.sha256(convertir(leer(fichero), fichero, **kw).csv_bytes).hexdigest() == esperado


@requiere_calamine
def test_mismo_fichero_mismos_parametros_mismos_bytes():
    for f, kw in (("oficina_con_valor.xlsx", dict(hoja="Datos", fila_de_cabecera=3, filas_a_excluir=[10])),
                  ("tipos_raros.xlsx", dict(hoja="Hoja1", fila_de_cabecera=1)),
                  ("nombres.xlsx", dict(hoja="Año", fila_de_cabecera=1))):
        a, b = convertir(leer(f), f, **kw), convertir(leer(f), f, **kw)
        assert a.csv_bytes == b.csv_bytes
        assert json.dumps(a.origen, sort_keys=True) == json.dumps(b.origen, sort_keys=True)


@requiere_calamine
def test_el_csv_es_utf8_sin_bom_con_lf_y_comillas_minimas():
    t = convertir(leer("nombres.xlsx"), "n.xlsx", hoja="Año", fila_de_cabecera=1)
    assert not t.csv_bytes.startswith(b"\xef\xbb\xbf") and b"\r" not in t.csv_bytes
    assert t.csv_bytes.endswith(b"\n") and "Niño".encode() in t.csv_bytes
    assert '"' not in t.csv_text.replace('"1,5"', "").replace('"2,5"', "").replace('"3,5"', "")
    d = G.xlsx([G.Hoja("H", [["a", "b"], ['dice "hola"', "x,y"], ["salto\nde línea", "ok"]])])
    t2 = convertir(d, "q.xlsx", hoja="H", fila_de_cabecera=1)
    assert t2.csv_text == 'a,b\n"dice ""hola""","x,y"\n"salto\nde línea",ok\n'
    assert csv_de(t2)[1] == ['dice "hola"', "x,y"]


def test_escribir_csv_es_determinista_y_una_sola_columna_vacia_va_entre_comillas():
    w = valores_tablas.escribir_csv
    assert w(["a", "b"], [["1", ""], ["", ""]]) == b"a,b\n1,\n,\n"
    assert w(["a"], [[""]]) == b'a\n""\n'
    assert w(["x"], [["é"]]) == "x\né\n".encode("utf-8")


@requiere_calamine
def test_la_version_del_conversor_va_en_el_origen():
    from importlib import metadata
    import matrixai
    t = convertir(leer("tipos_raros.xlsx"), "t.xlsx", hoja="Hoja1", fila_de_cabecera=1)
    c = t.origen["conversor"]
    assert c["nombre"] == "matrixai.tablas"
    assert c["versiones"]["python-calamine"] == metadata.version("python-calamine")
    assert c["versiones"]["matrixai-core"] == matrix_version() == matrixai.__version__


def matrix_version() -> str:
    import matrixai
    return matrixai.__version__


# ── Criterio 9: origen ──────────────────────────────────────────────────────────────────
@requiere_calamine
def test_el_origen_lleva_la_huella_del_original_y_lo_que_se_hizo():
    d = leer("oficina_con_valor.xlsx")
    t = convertir(d, "C:\\Users\\ana\\ventas\\oficina.xlsx", hoja="Datos", fila_de_cabecera=3,
                  filas_a_excluir=[10])
    o = t.origen
    assert o["sha256"] == hashlib.sha256(d).hexdigest() != hashlib.sha256(t.csv_bytes).hexdigest()
    assert o["bytes"] == len(d) and o["formato"] == "xlsx"
    assert o["nombre"] == "oficina.xlsx", "solo el nombre, nunca la ruta de la persona"
    assert (o["hoja"], o["fila_de_cabecera"], o["filas_de_cabecera"]) == ("Datos", 3, 1)
    assert o["rango"] == "A3:F10"
    assert o["filas_excluidas"] == [{"fila": 10, "texto": "TOTAL"}]
    assert {"esquema", "conversor", "declaraciones"} <= set(o)
    for x in o["declaraciones"]:
        assert set(x) == {"tipo", "columna", "cuantas", "detalle", "texto"}
        assert x["texto"]["es"] and x["texto"]["en"] and x["texto"]["es"] != x["texto"]["en"]
        assert "{" not in x["texto"]["es"] + x["texto"]["en"]
    json.dumps(o, ensure_ascii=False)


@requiere_calamine
def test_un_libro_limpio_no_inventa_declaraciones_ni_origen_de_mas():
    """«Lo que no se sabe se omite o va null, nunca inventado»; y sin nada que declarar, no se
    inventan avisos."""
    d = G.xlsx([G.Hoja("H", [["a", "b"], [1, "x"], [2, "y"]])])
    t = convertir(d, "limpio.xlsx", hoja="H", fila_de_cabecera=1)
    assert t.origen["declaraciones"] == [] and t.origen["filas_excluidas"] == []
    assert t.csv_text == "a,b\n1,x\n2,y\n"


@requiere_calamine
def test_el_nombre_del_fichero_no_se_cuela_en_rutas_ni_rompe_el_json():
    for feo in ("../../etc/passwd", "a/b\\c.xlsx", "\"x\"\n.xlsx", "ñandú\u0000.xlsx", "x" * 1000):
        t = convertir(leer("ceros_y_vacios.xlsx"), feo, hoja="Datos", fila_de_cabecera=1)
        assert "/" not in t.origen["nombre"] and "\\" not in t.origen["nombre"]
        assert len(t.origen["nombre"]) <= 255
        json.loads(json.dumps(t.origen))


# ── Criterio 10: el CSV pasa por el producto sin tocarlo ────────────────────────────────
@requiere_calamine
def test_el_csv_resultante_pasa_por_el_analizador_sin_tocarlo():
    from matrixai.training.data import normalize_csv_text
    from matrixai.training.dataset_analysis import analyze_dataset_csv
    t = convertir(leer("oficina_con_valor.xlsx"), "o.xlsx", hoja="Datos", fila_de_cabecera=3,
                  filas_a_excluir=[10])
    assert normalize_csv_text(t.csv_text) == t.csv_text, "el CSV canónico ya es el que el producto espera"
    a = analyze_dataset_csv(t.csv_text)
    tipo = {c: v["type"] for c, v in a["columns"].items()}
    assert tipo["Fecha"] == "date" and a["temporal_columns"] == ["Fecha"]
    assert tipo["Unidades"] == "integer" and tipo["Precio"] == "number" and tipo["Importe"] == "number"


@requiere_calamine
def test_la_fila_total_sin_quitar_estropea_la_fecha_y_quitada_no():
    """El motivo de DA2: con el TOTAL dentro, `Fecha` deja de reconocerse como fecha."""
    from matrixai.training.dataset_analysis import analyze_dataset_csv
    sin = convertir(leer("oficina_con_valor.xlsx"), "o.xlsx", hoja="Datos", fila_de_cabecera=3)
    con = convertir(leer("oficina_con_valor.xlsx"), "o.xlsx", hoja="Datos", fila_de_cabecera=3,
                    filas_a_excluir=[10])
    assert analyze_dataset_csv(sin.csv_text)["columns"]["Fecha"]["type"] != "date"
    assert analyze_dataset_csv(con.csv_text)["columns"]["Fecha"]["type"] == "date"


@requiere_calamine
def test_una_columna_de_fechas_con_y_sin_hora_sigue_siendo_una_fecha_para_el_analizador():
    from matrixai.training.dataset_analysis import analyze_dataset_csv
    t = convertir(leer("fechas_1900.xlsx"), "f.xlsx", hoja="Fechas", fila_de_cabecera=1)
    a = analyze_dataset_csv(t.csv_text)
    assert a["columns"]["fecha"]["type"] == "date" and a["columns"]["momento"]["type"] == "date"


@requiere_calamine
def test_la_cabecera_de_ctg_se_propone_en_la_fila_2_y_el_csv_lo_analiza():
    """Un `.xls` de verdad (CTG de UCI): hoja de descripción delante, cabecera en la 2, nombres
    repetidos. Si el fichero no está en esta máquina se salta CON razón."""
    ctg = pathlib.Path.home() / ".agentes" / "146" / "ctg" / "xls" / "CTG.xls"
    if not ctg.exists():
        pytest.skip(f"{ctg} no está en esta máquina")
    from matrixai.training.dataset_analysis import analyze_dataset_csv
    d = ctg.read_bytes()
    r = inspeccionar(d, "CTG.xls")
    assert [h["nombre"] for h in r["hojas"]] == ["Description", "Data", "Raw Data"]
    data = next(h for h in r["hojas"] if h["nombre"] == "Data")
    assert data["fila_de_cabecera_propuesta"] == 2 and (data["filas"], data["columnas"]) == (2131, 46)
    t = convertir(d, "CTG.xls", hoja="Data", fila_de_cabecera=2, filas_a_excluir=[2130, 2131])
    cab = csv_de(t)[0]
    assert len(cab) == len(set(cab)) == 42 and "AC_2" in cab
    assert len(csv_de(t)) == 1 + 2126
    assert analyze_dataset_csv(t.csv_text)["columns"]["NSP"]["type"] in {"integer", "categorical"}


# ── Los motivos, en los dos idiomas ─────────────────────────────────────────────────────
def test_todo_codigo_de_rechazo_tiene_motivo_en_es_y_en():
    from matrixai.tablas.errores import rechazar
    detalle = dict(extension=".xlsx", formato=".ods", descomprimido="1 GB", original="1 KB",
                   tope="1 GB", hoja="H", filas=1, columnas=1, necesario="1 GB", disponible="1 MB",
                   hojas=["A", "B"], fila=1, primera=1, ultima=2, parametro="p", cuantas=2,
                   celdas=["A1"], tipo="X")
    for codigo, (es, en) in CODIGOS.items():
        e = rechazar(codigo, **detalle)
        assert e.motivo_es and e.motivo_en and e.motivo_es != e.motivo_en, codigo
        assert "{" not in e.motivo_es + e.motivo_en, codigo
        assert e.motivo("en") == e.motivo_en and e.motivo("es") == e.motivo_es
        assert e.como_dict()["codigo"] == codigo


def test_los_textos_de_las_declaraciones_estan_en_los_dos_idiomas():
    from matrixai.tablas.declaraciones import TIPOS, declarar
    for tipo in TIPOS:
        d = declarar(tipo, columna="C", cuantas=3, letras=["A"], letra="A", original="o",
                     grupo="g", sub="s", filas=[1], textos=["t"], rangos=["A1:A2"], tipos=["#N/A"],
                     hoja="H", que=["q"], celdas=["A1"], nombres=["n"], fila=1)
        assert d["texto"]["es"] and d["texto"]["en"] and d["texto"]["es"] != d["texto"]["en"], tipo
        assert "{" not in d["texto"]["es"] + d["texto"]["en"], tipo


# ── Singular y plural ───────────────────────────────────────────────────────────────────
_SUSTANTIVOS_ES = "celdas|filas|columnas|fechas|valores|rangos|hojas|horas|duraciones"
_SUSTANTIVOS_EN = "cells|rows|columns|dates|values|ranges|sheets|times|durations"
_SINGULARES_ES = "celda|fila|columna|fecha|valor|rango|hoja"
_SINGULARES_EN = "cell|row|column|date|value|range|sheet"
_DETALLE_PLURAL = dict(extension=".xlsx", formato=".ods", descomprimido="1 GB", original="1 KB",
                       tope="1 GB", hoja="H", necesario="1 GB", disponible="1 MB",
                       hojas=["A", "B"], fila=1, primera=1, ultima=2, parametro="p",
                       celdas=["A1"], tipo="X")
_DETALLE_DECLARACION = dict(letras=["A"], letra="A", original="o", grupo="g", sub="s", filas=[1],
                            textos=["t"], rangos=["A1:A2"], tipos=["#N/A"], hoja="H", que=["q"],
                            celdas=["A1"], nombres=["n"], fila=1)


def _textos_con_n(n: int):
    """Todos los textos que el módulo redacta con un recuento `n`: (origen, idioma, texto)."""
    from matrixai.tablas.declaraciones import TIPOS, declarar
    from matrixai.tablas.errores import rechazar
    for tipo in TIPOS:
        d = declarar(tipo, columna="C", cuantas=n, **_DETALLE_DECLARACION)
        for idioma in ("es", "en"):
            yield f"declaración {tipo}", idioma, d["texto"][idioma]
    for codigo in CODIGOS:
        e = rechazar(codigo, filas=n, columnas=n, cuantas=n, **_DETALLE_PLURAL)
        yield f"rechazo {codigo}", "es", e.motivo_es
        yield f"rechazo {codigo}", "en", e.motivo_en


def test_ningun_texto_dice_1_celdas_ni_2_celda_en_es_ni_en_en():
    """Recorre TODAS las declaraciones y TODOS los códigos de rechazo con n=1 y n=2: el singular con
    el 1 y el plural con lo demás, en español y en inglés."""
    import re
    mal_uno = {"es": re.compile(rf"(?<![\d.,])1 ({_SUSTANTIVOS_ES})\b"),
               "en": re.compile(rf"(?<![\d.,])1 ({_SUSTANTIVOS_EN})\b")}
    mal_dos = {"es": re.compile(rf"(?<![\d.,])2 ({_SINGULARES_ES})\b"),
               "en": re.compile(rf"(?<![\d.,])2 ({_SINGULARES_EN})\b")}
    vistos = 0
    for n, mal in ((1, mal_uno), (2, mal_dos)):
        for origen, idioma, texto in _textos_con_n(n):
            vistos += 1
            assert not mal[idioma].search(texto), f"{origen} [{idioma}] n={n}: {texto!r}"
    assert vistos >= 2 * 2 * (len(CODIGOS) + 28)


def test_los_textos_con_recuento_cambian_de_forma_entre_1_y_2():
    """El singular no es el plural con otro número: si ningún texto cambiara de forma, la prueba de
    arriba no estaría mirando nada."""
    import re
    uno = {(o, i): t for o, i, t in _textos_con_n(1)}
    dos = {(o, i): t for o, i, t in _textos_con_n(2)}
    cambian = [k for k in uno if re.sub(r"\d+", "#", uno[k]) != re.sub(r"\d+", "#", dos[k])]
    esperados = {"declaración filas_excluidas", "declaración celdas_combinadas_vacias",
                 "declaración formulas_sin_valor", "rechazo formula_sin_valor",
                 "rechazo demasiadas_filas", "rechazo demasiadas_columnas",
                 "declaración columnas_vacias_quitadas", "declaración hay_celdas_combinadas"}
    assert esperados <= {o for o, _ in cambian}
    assert all((o, "es") in cambian and (o, "en") in cambian for o in esperados)


# ── Seguridad del XML ───────────────────────────────────────────────────────────────────
@requiere_calamine
def test_un_xml_con_entidades_se_rechaza_en_vez_de_expandirse():
    """Billion laughs: el DOCTYPE/ENTITY de una hoja se rechaza antes de pasar por el parser."""
    bomba = (b'<?xml version="1.0"?><!DOCTYPE w [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;">'
             b'<!ENTITY c "&b;&b;&b;&b;&b;">]><worksheet xmlns="http://schemas.openxmlformats.org/'
             b'spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="1"><c r="A1" '
             b't="inlineStr"><is><t>&c;</t></is></c></row><row r="2"><c r="A2"><f>1</f></c></row>'
             b"</sheetData></worksheet>")
    libro = G.xlsx([G.Hoja("H", [["a"]])], sobreescribir={"xl/worksheets/sheet1.xml": bomba})
    e = rechazo(convertir, libro, "e.xlsx", hoja="H", fila_de_cabecera=1)
    assert e.codigo == "lector_fallo" and "entidades" in e.detalle["tipo"]


@requiere_calamine
def test_texto_que_empieza_por_igual_mas_o_arroba_se_conserva_como_dato():
    d = G.xlsx([G.Hoja("H", [["a", "b"], ["=1+1", "+34 600"], ["@usuario", "-5x"]])])
    t = convertir(d, "f.xlsx", hoja="H", fila_de_cabecera=1)
    assert csv_de(t)[1:] == [["=1+1", "+34 600"], ["@usuario", "-5x"]]
