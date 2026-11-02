# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C1 — lo que cazó la auditoría Opus de C1 (2026-11-02), una prueba con nombre por hallazgo.

B1 la geometría previa era una expresión regular y daba por VACÍAS hojas legítimas (prefijo `x:`,
   atributos reordenados, celdas sin `r`); B2 un fichero de 1–6 KB hacía ABORTAR el proceso desde
Rust; I1 tres bombas esquivaban las guardas (columnas ocultas, textos compartidos, combinadas);
I2 excepciones de biblioteca fuera; I3 recuentos falsos; I4 dos sabotajes sin dientes (A5, A8);
I5 el formato `00000`; y los menores (enteros grandes, series de fecha imposibles, motivos de
formato). Los libros se escriben aquí a mano (`_libro`): se controla cada byte del XML.

El ORÁCULO de la geometría es calamine mismo, en este proceso, sobre libros pequeños y legítimos:
la regla es «las mismas celdas que calamine», así que se compara con él, no con lo que uno cree.
"""
from __future__ import annotations

import csv
import io
import os
import pickle
import time
import tracemalloc
import zipfile

import pytest

from matrixai.tablas import Topes, TablaRechazada, convertir, detectar_formato, inspeccionar
from matrixai.tablas import aislado
from matrixai.tablas import api as api_tablas
from matrixai.tablas import xlsx_xml

try:
    import python_calamine  # noqa: F401
    HAY_CALAMINE = True
except ImportError:
    HAY_CALAMINE = False

requiere_calamine = pytest.mark.skipif(
    not HAY_CALAMINE,
    reason="falta python-calamine (extra `tablas`): pip install 'matrixai-core[tablas]'")

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "tablas")

# ── Un .xlsx a mano ─────────────────────────────────────────────────────────────────────
_NS = ('xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
_CT = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
       '<Default Extension="xml" ContentType="application/xml"/>'
       '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
       '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
       '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
       '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
       '</Types>')
_RELS = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
_WBRELS = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
           '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
           '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
           '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
           '</Relationships>')
# estilos: 0 general, 1 fecha (14), 2 fecha-hora (22), 3 hora (21), 4 porcentaje (10),
# 5 [h]:mm:ss (46), 6 "00000" (164), 7 "000-00-0000" (165)
_STYLES = ('<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
           '<numFmts count="2"><numFmt numFmtId="164" formatCode="00000"/><numFmt numFmtId="165" formatCode="000-00-0000"/></numFmts>'
           '<fonts count="1"><font/></fonts><fills count="1"><fill/></fills><borders count="1"><border/></borders>'
           '<cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="8">'
           '<xf numFmtId="0"/><xf numFmtId="14" applyNumberFormat="1"/><xf numFmtId="22" applyNumberFormat="1"/>'
           '<xf numFmtId="21" applyNumberFormat="1"/><xf numFmtId="10" applyNumberFormat="1"/>'
           '<xf numFmtId="46" applyNumberFormat="1"/><xf numFmtId="164" applyNumberFormat="1"/>'
           '<xf numFmtId="165" applyNumberFormat="1"/></cellXfs></styleSheet>')
_SST_VACIA = ('<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
              'count="1" uniqueCount="1"><si><t></t></si></sst>')


def _workbook(fecha1904=False) -> str:
    pr = '<workbookPr date1904="1"/>' if fecha1904 else "<workbookPr/>"
    return (f'<?xml version="1.0" encoding="UTF-8"?><workbook {_NS}>{pr}<sheets>'
            '<sheet name="Hoja1" sheetId="1" r:id="rId1"/></sheets></workbook>')


def _hoja(datos: str, antes: str = "", despues: str = "") -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?><worksheet {_NS}>{antes}<sheetData>{datos}'
            f'</sheetData>{despues}</worksheet>')


def _libro(hoja_xml, *, fecha1904=False, sst=_SST_VACIA, wb=None, extra=None) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("xl/workbook.xml", wb or _workbook(fecha1904))
        z.writestr("xl/_rels/workbook.xml.rels", _WBRELS)
        z.writestr("xl/styles.xml", _STYLES)
        z.writestr("xl/sharedStrings.xml", sst)
        z.writestr("xl/worksheets/sheet1.xml", hoja_xml)
        for k, v in (extra or {}).items():
            z.writestr(k, v)
    return b.getvalue()


_BASE = ('<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>b</t></is></c></row>'
         '<row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c></row>'
         '<row r="3"><c r="A3"><v>3</v></c><c r="B3"><v>4</v></c></row>')


def _geo(libro: bytes):
    zf = zipfile.ZipFile(io.BytesIO(libro))
    ruta = xlsx_xml.mapa_de_hojas(zf)[0]["ruta"]
    return xlsx_xml.explorar_hoja(zf, ruta, xlsx_xml.estilos_del_libro(zf))


def _rect(g):
    return None if g.vacia else (g.min_fila, g.min_col, g.max_fila, g.max_col)


def _rect_de_calamine(libro: bytes):
    import python_calamine as pc
    s = pc.load_workbook(io.BytesIO(libro)).get_sheet_by_index(0)
    if s.start is None:
        return None
    return (s.start[0] + 1, s.start[1] + 1, s.end[0] + 1, s.end[1] + 1)


def _csv(t) -> list[list[str]]:
    return list(csv.reader(io.StringIO(t.csv_text)))


def _decl(t, tipo, columna=None):
    hay = [d for d in t.origen["declaraciones"] if d["tipo"] == tipo
           and (columna is None or d["columna"] == columna)]
    return hay


def _rechazo(f, *a, **k) -> TablaRechazada:
    with pytest.raises(TablaRechazada) as e:
        f(*a, **k)
    return e.value


def _prefijo_x(xml: str) -> str:
    for et in ("row", "c", "v", "is", "t"):
        xml = xml.replace(f"<{et}>", f"<x:{et}>").replace(f"<{et} ", f"<x:{et} ").replace(f"</{et}>", f"</x:{et}>")
    return xml


_NSX = ('xmlns:x="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')


# ── B1: la geometría da las mismas celdas que calamine ──────────────────────────────────
_FORMAS = {
    # los cuatro del auditor
    "reordenado": _BASE.replace('<c r="', '<c s="0" r="'),
    "sin_r": ('<row r="1"><c t="inlineStr"><is><t>a</t></is></c><c t="inlineStr"><is><t>b</t></is></c></row>'
              '<row r="2"><c><v>1</v></c><c><v>2</v></c></row><row r="3"><c><v>3</v></c><c><v>4</v></c></row>'),
    "lejos_reordenado": _BASE + '<row r="5000"><c s="0" r="Z5000"><v>9</v></c></row>',
    # medidas en `~/.agentes/150/rep-c1/sondas/cal_formas*.py`
    "estilo_autocerrada": _BASE + '<row r="100"><c r="Z100" s="1"/></row>',
    "v_vacio": _BASE + '<row r="100"><c r="Z100"><v></v></c></row>',
    "formula_sin_v": _BASE + '<row r="100"><c r="Z100"><f>1+1</f></c></row>',
    "str_vacio": _BASE + '<row r="100"><c r="Z100" t="str"><v></v></c></row>',
    "str_v_autocerrado": _BASE + '<row r="100"><c r="Z100" t="str"><v/></c></row>',
    "inline_vacio": _BASE + '<row r="100"><c r="Z100" t="inlineStr"><is><t></t></is></c></row>',
    "inline_con_v": _BASE + '<row r="100"><c r="Z100" t="inlineStr"><v>q</v></c></row>',
    "is_sin_t": _BASE + '<row r="100"><c r="Z100"><is><t>x</t></is></c></row>',
    "sst_vacio": _BASE + '<row r="100"><c r="Z100" t="s"><v>0</v></c></row>',
    "s_v_vacio": _BASE + '<row r="100"><c r="Z100" t="s"><v></v></c></row>',
    "error": _BASE + '<row r="100"><c r="Z100" t="e"><v>#N/A</v></c></row>',
    "error_vacio": _BASE + '<row r="100"><c r="Z100" t="e"><v></v></c></row>',
    "d_vacio": _BASE + '<row r="100"><c r="Z100" t="d"><v></v></c></row>',
    "v_espacios": _BASE + '<row r="100"><c r="Z100"><v>  </v></c></row>',
    "sin_r_tras_Y": _BASE + '<row r="100"><c r="Y100"><v>1</v></c><c><v>2</v></c></row>',
    "fila_sin_r": _BASE + '<row><c r="Z100"><v>3</v></c></row>',
    "fila_y_celdas_sin_r": _BASE + '<row><c><v>3</v></c><c><v>4</v></c></row>',
    "row_r_y_celdas_sin_r": _BASE + '<row r="100"><c><v>3</v></c><c><v>4</v></c></row>',
    "minusculas": _BASE + '<row r="100"><c r="z100"><v>5</v></c></row>',
    "otro_espacio_de_nombres": _BASE + '<row r="100"><c xmlns="urn:otro" r="Z100"><v>7</v></c></row>',
    "r_con_prefijo_no_es_r": _BASE + '<row r="100"><c xmlns:q="urn:q" q:r="Z100"><v>1</v></c></row>',
    "filas_desordenadas": _BASE + '<row r="100"><c r="Z100"><v>1</v></c></row><row r="50"><c r="A50"><v>2</v></c></row>',
    "celda_en_otra_fila": _BASE + '<row r="100"><c r="Z7"><v>1</v></c></row>',
    "fila_sin_r_tras_celda_lejos": _BASE + '<row r="100"><c r="Z7"><v>1</v></c></row><row><c><v>3</v></c></row>',
    "v_repetido": _BASE + '<row r="100"><c r="Z100"><v>1</v><v>2</v></c></row>',
    "con_saltos_de_linea": _BASE.replace("<v>", "\n  <v>").replace("</c>", "\n</c>"),
    "v_vacio_con_saltos": _BASE + '<row r="100">\n<c r="Z100">\n  <v></v>\n</c></row>',
}


@requiere_calamine
@pytest.mark.parametrize("forma", sorted(_FORMAS))
def test_la_geometria_da_el_mismo_rectangulo_que_calamine(forma):
    """B1: el rectángulo que la guarda mide es el de calamine, con SUS reglas (nombres locales,
    atributos en cualquier orden, `r` opcional, qué cuenta como celda con valor)."""
    libro = _libro(_hoja(_FORMAS[forma]))
    assert _rect(_geo(libro)) == _rect_de_calamine(libro), forma


@requiere_calamine
def test_la_geometria_con_prefijo_x_es_la_de_calamine_y_la_hoja_se_convierte():
    """B1, el caso del Open XML SDK: `x:worksheet`, `x:c`… La regex la daba por VACÍA."""
    hx = f'<?xml version="1.0" encoding="UTF-8"?><x:worksheet {_NSX}><x:sheetData>{_prefijo_x(_BASE)}</x:sheetData></x:worksheet>'
    wbx = (f'<?xml version="1.0" encoding="UTF-8"?><x:workbook {_NSX}><x:sheets>'
           '<x:sheet name="Hoja1" sheetId="1" r:id="rId1"/></x:sheets></x:workbook>')
    libro = _libro(hx, wb=wbx)
    assert _rect(_geo(libro)) == _rect_de_calamine(libro) == (1, 1, 3, 2)
    t = convertir(libro, "x.xlsx", fila_de_cabecera=1)
    assert t.csv_text == "a,b\n1,2\n3,4\n"
    r = inspeccionar(libro, "x.xlsx")["hojas"][0]
    assert (r["filas"], r["columnas"], r["rango"]) == (3, 2, "A1:B3")


@requiere_calamine
@pytest.mark.parametrize("forma", ["reordenado", "sin_r"])
def test_una_hoja_legitima_con_otra_forma_de_xml_no_se_declara_vacia(forma):
    """B1: los dos casos del auditor que salían «hoja vacía» se convierten enteros."""
    t = convertir(_libro(_hoja(_FORMAS[forma])), "f.xlsx", fila_de_cabecera=1)
    assert t.csv_text == "a,b\n1,2\n3,4\n"


@requiere_calamine
def test_la_celda_lejana_con_atributos_reordenados_entra_en_la_guarda_de_memoria(monkeypatch):
    """B1: `<c s="0" r="XFD1048576">` (atributos reordenados) esquivaba la regex y calamine
    pedía 549 GB y abortaba. Ahora la guarda la ve ANTES de abrir: memoria insuficiente."""
    libro = _libro(_hoja(_BASE + '<row r="1048576"><c s="0" r="XFD1048576"><v>9</v></c></row>'))
    llamadas = []
    monkeypatch.setattr(aislado, "Lectura", lambda *a, **k: llamadas.append(1) or (_ for _ in ()).throw(
        AssertionError("se abrió con calamine")))
    monkeypatch.setattr(aislado, "leer_hoja", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("se abrió con calamine")))
    monkeypatch.setattr(api_tablas.aislado, "empezar_a_leer", lambda *a, **k: _LecturaInerte())
    e = _rechazo(convertir, libro, "l.xlsx", fila_de_cabecera=1)
    assert e.codigo == "memoria_insuficiente"


class _LecturaInerte:
    def cancelar(self):
        pass

    def esperar(self):
        raise AssertionError("se esperó a calamine en vez de rechazar por la geometría")


@requiere_calamine
def test_una_hoja_nunca_se_declara_vacia_sin_que_calamine_lo_confirme(monkeypatch):
    """B1: aunque la geometría (fallando) diga «vacía», quien decide es calamine."""
    vacia = xlsx_xml.Geometria()
    monkeypatch.setattr(api_tablas._Libro, "geometria", lambda self, meta: vacia)
    libro = _libro(_hoja(_BASE))
    t = convertir(libro, "v.xlsx", fila_de_cabecera=1)
    assert t.csv_text == "a,b\n1,2\n3,4\n"
    h = inspeccionar(libro, "v.xlsx")["hojas"][0]
    assert h["filas"] == 3 and not [a for a in h["avisos"] if a["tipo"] == "hoja_sin_datos"]


@requiere_calamine
def test_una_hoja_vacia_de_verdad_la_confirma_calamine():
    libro = _libro(_hoja('<row r="1"><c r="A1" s="1"/></row>'))
    assert _rechazo(convertir, libro, "v.xlsx", fila_de_cabecera=1).codigo == "hoja_vacia"


@pytest.mark.parametrize("codificacion", ["utf-8", "utf-16"])
def test_un_doctype_partido_en_el_borde_de_un_trozo_o_en_utf16_se_rechaza(codificacion):
    """B1 (defensas): el `<!DOCTYPE` con entidades lo rechaza el ANALIZADOR, caiga donde caiga
    el borde del trozo de 1 MiB y en cualquier codificación (una búsqueda de bytes no lo ve en
    UTF-16)."""
    relleno = "<!--" + "x" * ((1 << 20) - 50) + "-->"
    dt = ('<!DOCTYPE worksheet [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>')
    xml = _hoja(_BASE.replace("<t>a</t>", "<t>&b;</t>")).replace("?>", "?>" + relleno + dt, 1)
    if codificacion == "utf-16":
        xml = xml.replace('encoding="UTF-8"', 'encoding="UTF-16"')
    zf = zipfile.ZipFile(io.BytesIO(_libro(xml.encode(codificacion))))
    with pytest.raises(TablaRechazada) as e:
        xlsx_xml.explorar_hoja(zf, "xl/worksheets/sheet1.xml")
    assert e.value.codigo == "lector_fallo" and "DTD" in e.value.motivo_es


# ── B2: un fichero hostil no tumba el proceso ───────────────────────────────────────────
@requiere_calamine
def test_un_xls_que_hace_abortar_al_lector_no_tumba_el_proceso():
    """B2: con calamine en ESTE proceso, `hostil_xls_aborta.xls` (un byte cambiado de
    `antiguo.xls`) lo abortaba con SIGABRT pidiendo 181 GB. Ahora aborta el HIJO, con techo, y
    aquí llega una `TablaRechazada` con su motivo en los dos idiomas."""
    datos = open(os.path.join(FIXTURES, "hostil_xls_aborta.xls"), "rb").read()
    for f in (lambda: inspeccionar(datos, "a.xls"), lambda: convertir(datos, "a.xls", hoja=0, fila_de_cabecera=1)):
        e = _rechazo(f)
        assert e.codigo == "lector_sin_memoria", e.codigo
        assert "memoria" in e.motivo_es and "memory" in e.motivo_en
    assert os.getpid()   # seguimos vivos


@requiere_calamine
def test_un_xlsx_de_2_kb_con_un_recuento_de_textos_enorme_no_tumba_el_proceso():
    """B2 en `.xlsx`: `uniqueCount="1000000000"` en `sharedStrings.xml` hace que calamine reserve
    24 GB al ABRIR el libro (antes de cualquier hoja): la geometría no lo ve; el techo del hijo, sí."""
    sst = ('<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
           'count="1000000000" uniqueCount="1000000000"><si><t>a</t></si><si><t>b</t></si></sst>')
    libro = _libro(_hoja('<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                         '<row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c></row>'), sst=sst)
    assert len(libro) < 4000
    e = _rechazo(convertir, libro, "u.xlsx", fila_de_cabecera=1)
    assert e.codigo == "lector_sin_memoria"
    assert aislado.ULTIMO_HIJO.get("codigo") == "lector_sin_memoria"


@requiere_calamine
def test_el_techo_de_memoria_del_hijo_se_aplica(monkeypatch):
    """B2: el `RLIMIT_AS` llega al hijo. La celda suelta en Z1048576 necesita ~1,1 GB: con un
    techo de 300 MB, el hijo se queda sin memoria y aquí se dice."""
    datos = open(os.path.join(FIXTURES, "celda_perdida.xlsx"), "rb").read()
    e = _rechazo(convertir, datos, "c.xlsx", hoja="Enero", fila_de_cabecera=1,
                 topes=Topes(max_memoria_lector=300 << 20))
    assert e.codigo == "lector_sin_memoria" and "314.6 MB" in e.motivo_es


@requiere_calamine
def test_un_lector_que_se_pasa_de_tiempo_se_mata_y_no_queda_huerfano(monkeypatch):
    """B2: el tope de tiempo de pared. El hijo se MATA y se ESPERA (no queda ni zombi)."""
    procesos = []
    real = aislado.Lectura.__init__

    def espia(self, *a, **k):
        real(self, *a, **k)
        procesos.append(self.proc)
    monkeypatch.setattr(aislado.Lectura, "__init__", espia)
    monkeypatch.setattr(aislado, "_FUENTE_DEL_HIJO", "import time; time.sleep(60)")
    t0 = time.monotonic()
    e = _rechazo(convertir, _libro(_hoja(_BASE)), "t.xlsx", fila_de_cabecera=1,
                 topes=Topes(max_segundos_lector=1.0))
    assert e.codigo == "lector_tiempo" and "1 s" in e.motivo_es
    assert time.monotonic() - t0 < 30
    assert procesos and all(p.returncode is not None for p in procesos)
    for p in procesos:
        assert not os.path.exists(f"/proc/{p.pid}") or "Z" not in open(f"/proc/{p.pid}/stat").read().split()[2]


@requiere_calamine
@pytest.mark.parametrize("fuente,codigo", [
    ("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)", "lector_abortado"),
    ("import os; os.abort()", "lector_abortado"),
    ("import sys; sys.stderr.write('memory allocation of 181193933040 bytes failed\\n'); import os; os.abort()", "lector_sin_memoria"),
    ("import sys; sys.stdout.buffer.write(b'basura'); sys.exit(0)", "lector_abortado"),
])
def test_cada_final_del_hijo_tiene_su_codigo(monkeypatch, fuente, codigo):
    """B2: cómo acabe el hijo decide el código; ninguno llega como excepción ajena."""
    monkeypatch.setattr(aislado, "_FUENTE_DEL_HIJO", fuente)
    e = _rechazo(convertir, _libro(_hoja(_BASE)), "s.xlsx", fila_de_cabecera=1)
    assert e.codigo == codigo
    assert e.motivo_es != e.motivo_en


def test_lo_que_devuelve_el_hijo_no_puede_pedir_que_el_padre_ejecute_nada():
    """B2: el padre lee la respuesta con un `Unpickler` que solo admite fechas."""
    class Malo:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    with pytest.raises(pickle.UnpicklingError):
        aislado._UnpicklerDeValores(io.BytesIO(pickle.dumps({"ok": True, "x": Malo()}))).load()
    import datetime as dt
    bueno = {"ok": True, "filas": [[dt.datetime(2024, 1, 2, 3, 4, 5), dt.date(2024, 1, 2), dt.time(1),
                                    dt.timedelta(hours=36), 1.5, "x", True, None]]}
    assert aislado._UnpicklerDeValores(io.BytesIO(pickle.dumps(bueno, protocol=4))).load() == bueno


@requiere_calamine
def test_el_proceso_del_studio_nunca_importa_calamine():
    """B2: calamine vive SOLO en el hijo. Si se importara aquí, un aborto volvería a ser nuestro."""
    import subprocess
    import sys
    codigo = ("import sys; from matrixai.tablas import convertir\n"
              f"convertir(open({os.path.join(FIXTURES, 'nombres.xlsx')!r}, 'rb').read(), 'n.xlsx', fila_de_cabecera=1)\n"
              "print('python_calamine' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=120,
                         env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "False"


# ── I1: las bombas que esquivaban las guardas ───────────────────────────────────────────
def _pico_de_memoria(f):
    tracemalloc.start()
    try:
        t0 = time.monotonic()
        r = f()
        return r, tracemalloc.get_traced_memory()[1], time.monotonic() - t0
    finally:
        tracemalloc.stop()


@requiere_calamine
def test_6000_columnas_ocultas_de_16384_no_se_expanden():
    """I1: 6.000 `<col min=1 max=16384 hidden>` eran 98 M de enteros en una lista (3,7 GB, 20 s).
    Ahora son RANGOS: se funden y se cuentan sin expandirlos."""
    cols = "<cols>" + '<col min="1" max="16384" hidden="1"/>' * 6000 + "</cols>"
    libro = _libro(_hoja(_BASE, antes=cols))
    t, pico, segundos = _pico_de_memoria(lambda: convertir(libro, "c.xlsx", fila_de_cabecera=1))
    assert pico < 50 << 20, pico
    assert segundos < 30
    assert _decl(t, "columnas_ocultas")[0]["cuantas"] == 2     # A y B, una vez cada una


def test_mas_definiciones_de_columna_que_columnas_tiene_excel_se_rechaza():
    cols = "<cols>" + '<col min="1" max="1" hidden="1"/>' * (xlsx_xml.MAX_DEFINICIONES_DE_COLUMNA + 1) + "</cols>"
    zf = zipfile.ZipFile(io.BytesIO(_libro(_hoja(_BASE, antes=cols))))
    with pytest.raises(TablaRechazada) as e:
        xlsx_xml.explorar_hoja(zf, "xl/worksheets/sheet1.xml")
    assert e.value.codigo == "lector_fallo" and "16.384" in e.value.motivo_es


@requiere_calamine
def test_6000_combinadas_de_16384_columnas_se_cuentan_por_rangos():
    """I1: 6.000 `A2:XFD2` sobre una hoja de 2 columnas costaban 53 s y 857 MB (se recorrían las
    16.384 columnas de cada rango). Recortadas a la hoja y sin repetidas: B2 es la única cubierta."""
    merges = '<mergeCells count="6000">' + '<mergeCell ref="A2:XFD2"/>' * 6000 + "</mergeCells>"
    libro = _libro(_hoja(_BASE, despues=merges))
    t, pico, segundos = _pico_de_memoria(lambda: convertir(libro, "m.xlsx", fila_de_cabecera=1))
    assert pico < 50 << 20, pico
    assert segundos < 30
    d = _decl(t, "celdas_combinadas_vacias")
    assert [(x["columna"], x["cuantas"]) for x in d] == [("b", 1)]


@requiere_calamine
def test_combinadas_que_se_solapan_se_rechazan():
    """I1: Excel no solapa combinaciones; si lo recortado suma más que la propia zona, el libro
    está fabricado (y rellenarlas costaría rangos × celdas)."""
    merges = '<mergeCells count="3"><mergeCell ref="A2:B3"/><mergeCell ref="A2:B3"/><mergeCell ref="A2:A3"/><mergeCell ref="B2:B3"/><mergeCell ref="A3:B3"/></mergeCells>'
    libro = _libro(_hoja(_BASE, despues=merges))
    e = _rechazo(convertir, libro, "m.xlsx", fila_de_cabecera=1)
    assert e.codigo == "lector_fallo" and "solapan" in e.motivo_es


def _sst_de(n_bytes: int) -> str:
    item = "<si><t>x</t></si>"
    cab = '<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    return cab + "<si><t>a</t></si><si><t>b</t></si>" + item * (n_bytes // len(item)) + "</sst>"


def test_los_textos_compartidos_tienen_tope_propio_y_se_rechazan_sin_abrir(monkeypatch):
    """I1: calamine carga `sharedStrings.xml` ENTERO al abrir (900 MB → 3,1 GB y 141 s, medido).
    Tope propio, comprobado ANTES de arrancar el lector."""
    monkeypatch.setattr(aislado, "Lectura", lambda *a, **k: (_ for _ in ()).throw(AssertionError("se abrió")))
    libro = _libro(_hoja(_BASE), sst=_sst_de(2 << 20))
    e = _rechazo(inspeccionar, libro, "s.xlsx", topes=Topes(max_cadenas_compartidas=1 << 20))
    assert e.codigo == "demasiados_textos" and e.detalle["bytes"] > 2 << 20
    assert "MB" in e.motivo_es and e.motivo_en != e.motivo_es


def test_el_tope_de_fabrica_de_textos_compartidos_para_la_bomba_de_900_mb():
    from matrixai.tablas import topes
    assert topes.MAX_CADENAS_COMPARTIDAS_DE_FABRICA < 900 << 20


@requiere_calamine
def test_los_textos_compartidos_entran_en_la_estimacion_de_memoria(monkeypatch):
    """I1: la memoria que se pide a la guarda incluye los textos compartidos."""
    from matrixai import limits
    pedidos = []
    real = limits.memoria_insuficiente
    monkeypatch.setattr(limits, "memoria_insuficiente", lambda n: pedidos.append(n) or real(n))
    monkeypatch.setattr(limits, "_memoria_disponible_del_sistema", lambda: 3 << 20)
    monkeypatch.setattr(aislado, "Lectura", lambda *a, **k: (_ for _ in ()).throw(AssertionError("se abrió")))
    libro = _libro(_hoja(_BASE), sst=_sst_de(4 << 20))
    e = _rechazo(convertir, libro, "s.xlsx", fila_de_cabecera=1)
    assert e.codigo == "memoria_insuficiente"
    from matrixai.tablas.topes import ram_estimada_de_una_hoja
    assert ram_estimada_de_una_hoja(0, 0, 4 << 20) >= 4 << 20


# ── I2: ninguna excepción de biblioteca fuera ───────────────────────────────────────────
def _con_version_de_zip(libro: bytes, version: int) -> bytes:
    """Cambia «versión necesaria para extraer» de todos los miembros en el directorio central
    (lo que hizo el fuzz del auditor: `NotImplementedError: zip file version 14.2`)."""
    b = bytearray(libro)
    i = b.find(b"PK\x01\x02")
    while i != -1:
        b[i + 6:i + 8] = version.to_bytes(2, "little")
        i = b.find(b"PK\x01\x02", i + 4)
    return bytes(b)


@pytest.mark.parametrize("version", [142, 159, 241, 0xFFFF])
def test_una_version_de_zip_desconocida_no_saca_notimplementederror(version):
    libro = _con_version_de_zip(_libro(_hoja(_BASE)), version)
    for f in (lambda: inspeccionar(libro, "z.xlsx"), lambda: convertir(libro, "z.xlsx", fila_de_cabecera=1)):
        assert _rechazo(f).codigo == "zip_danado"


# workbook.xml se lee ANTES de llegar a calamine; la hoja y los estilos, en el camino que la
# necesita (sin el extra `tablas` se rechaza antes con «falta el lector»: I-R2 de 150-C1).
@pytest.mark.parametrize("parte", ["xl/workbook.xml",
                                   pytest.param("xl/worksheets/sheet1.xml", marks=requiere_calamine),
                                   pytest.param("xl/styles.xml", marks=requiere_calamine)])
def test_una_codificacion_xml_que_no_existe_es_xml_mal_formado(parte):
    """I2, cazado por el fuzz de la reparación: `encoding="yes"` hacía que expat lanzara
    `LookupError` (salía como `error_interno`)."""
    zf = zipfile.ZipFile(io.BytesIO(_libro(_hoja(_BASE))))
    miembros = {i.filename: zf.read(i) for i in zf.infolist()}
    miembros[parte] = miembros[parte].replace(b'encoding="UTF-8"', b'encoding="yes"', 1)
    assert b'encoding="yes"' in miembros[parte]
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in miembros.items():
            z.writestr(k, v)
    for f in (lambda: inspeccionar(b.getvalue(), "e.xlsx"), lambda: convertir(b.getvalue(), "e.xlsx", fila_de_cabecera=1)):
        try:
            f()
        except TablaRechazada as e:
            assert e.codigo == "lector_fallo", e.codigo


def test_un_nombre_repetido_en_el_zip_se_rechaza():
    """El directorio con la hoja DOS veces: `zipfile` y calamine podrían leer miembros distintos."""
    import warnings
    b = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(b, "w") as z:
            src = zipfile.ZipFile(io.BytesIO(_libro(_hoja(_BASE))))
            for i in src.infolist():
                z.writestr(i.filename, src.read(i))
            z.writestr("xl/worksheets/sheet1.xml", _hoja(_BASE))
    assert _rechazo(inspeccionar, b.getvalue(), "d.xlsx").codigo == "zip_danado"


def test_un_fallo_nuestro_sale_como_error_interno_no_como_fichero_danado(monkeypatch):
    """I2 (red final): lo que se escape a las guardas no llega como excepción ajena NI se disfraza
    de «fichero dañado»: `error_interno` dice que el fallo es del conversor."""
    monkeypatch.setattr(api_tablas._Libro, "_clasificar", lambda self: {}["x"])
    e = _rechazo(inspeccionar, _libro(_hoja(_BASE)), "i.xlsx")
    assert e.codigo == "error_interno" and "KeyError" in e.motivo_es
    assert "conversor" in e.motivo_es and "converter" in e.motivo_en
    assert isinstance(e.__cause__, KeyError)


# ── I3: recuentos verdaderos ────────────────────────────────────────────────────────────
@requiere_calamine
def test_1500_filas_ocultas_se_declaran_1500_no_1000():
    """I3: la lista se cortaba en 1.000 y el recuento salía de la lista cortada."""
    filas = ['<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>b</t></is></c></row>']
    for r in range(2, 3002):
        oculta = ' hidden="1"' if r % 2 == 0 else ""
        filas.append(f'<row r="{r}"{oculta}><c r="A{r}"><v>{r}</v></c><c r="B{r}"><v>1</v></c></row>')
    libro = _libro(_hoja("".join(filas)))
    t = convertir(libro, "o.xlsx", fila_de_cabecera=1)
    d = _decl(t, "filas_ocultas")[0]
    assert d["cuantas"] == 1500 and d["detalle"]["filas"][:3] == [2, 4, 6]
    h = inspeccionar(libro, "o.xlsx")["hojas"][0]
    assert [a for a in h["avisos"] if a["tipo"] == "filas_ocultas"][0]["cuantas"] == 1500
    # las excluidas y las que se quitan por vacías no cuentan como «incluidas igualmente»
    t = convertir(libro, "o.xlsx", fila_de_cabecera=1, filas_a_excluir=[2, 4, 3])
    assert _decl(t, "filas_ocultas")[0]["cuantas"] == 1498


@requiere_calamine
def test_columnas_ocultas_solapadas_se_cuentan_una_vez():
    """I3: `<col 1-1 hidden>` y `<col 1-2 hidden>` son A y B, no A, A y B."""
    cols = '<cols><col min="1" max="1" hidden="1"/><col min="1" max="2" hidden="1"/></cols>'
    libro = _libro(_hoja(_BASE, antes=cols))
    t = convertir(libro, "c.xlsx", fila_de_cabecera=1)
    d = _decl(t, "columnas_ocultas")[0]
    assert d["cuantas"] == 2 and d["detalle"]["letras"] == ["A", "B"]
    aviso = [a for a in inspeccionar(libro, "c.xlsx")["hojas"][0]["avisos"] if a["tipo"] == "columnas_ocultas"]
    assert aviso[0]["cuantas"] == 2 and aviso[0]["detalle"]["letras"] == ["A", "B"]


@requiere_calamine
def test_un_error_en_la_cabecera_no_se_declara_como_error_de_los_datos():
    """I3 (y el sabotaje A9 del auditor, al revés): un `#REF!` en la CABECERA no es un error de la
    columna de datos. Que la cabecera se quede sin nombre ya lo declara `nombre_vacio_con_datos`."""
    filas = ('<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="e"><v>#REF!</v></c></row>'
             '<row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c></row>'
             '<row r="3"><c r="A3"><v>1</v></c><c r="B3" t="e"><v>#N/A</v></c></row>')
    t = convertir(_libro(_hoja(filas)), "e.xlsx", fila_de_cabecera=1)
    d = _decl(t, "errores_de_celda")
    assert [(x["columna"], x["cuantas"], x["detalle"]["tipos"]) for x in d] == [("columna_2", 1, ["#N/A"])]
    assert _decl(t, "nombre_vacio_con_datos")


@requiere_calamine
def test_un_error_en_una_fila_quitada_por_vacia_no_se_cuenta():
    filas = ('<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>b</t></is></c></row>'
             '<row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c></row>'
             '<row r="3"><c r="A3" t="e"><v>#N/A</v></c><c r="B3" t="e"><v>#N/A</v></c></row>'
             '<row r="4"><c r="A4"><v>1</v></c><c r="B4"><v>2</v></c></row>')
    t = convertir(_libro(_hoja(filas)), "e.xlsx", fila_de_cabecera=1)
    assert _decl(t, "filas_vacias_quitadas")[0]["detalle"]["filas"] == [3]
    assert not _decl(t, "errores_de_celda")


# ── I4: dientes para A5 y A8 ────────────────────────────────────────────────────────────
@requiere_calamine
def test_nan_e_infinito_en_la_hoja_salen_vacios_y_declarados_en_convertir():
    """I4/A5: `<v>NaN</v>` e `inf` los da calamine como `float` no finito: en el CSV, VACÍO
    (nunca `nan`, nunca `0`) y declarados. El sabotaje `repr(v)` del auditor salía verde."""
    filas = ['<row r="1"><c r="A1" t="inlineStr"><is><t>k</t></is></c><c r="B1" t="inlineStr"><is><t>x</t></is></c></row>']
    for i, v in enumerate(["NaN", "inf", "-inf", "1.5"], start=2):
        filas.append(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c><c r="B{i}"><v>{v}</v></c></row>')
    t = convertir(_libro(_hoja("".join(filas))), "n.xlsx", fila_de_cabecera=1)
    assert [r[1] for r in _csv(t)[1:]] == ["", "", "", "1.5"]
    assert "nan" not in t.csv_text.lower() and "inf" not in t.csv_text.lower()
    assert _decl(t, "no_finito", "x")[0]["cuantas"] == 3


@requiere_calamine
def test_el_tope_de_columnas_del_llamante_rechaza_antes_de_leer(monkeypatch):
    """I4/A8: con `max_columnas=3`, una hoja de 6 columnas se rechaza en la GEOMETRÍA, antes de
    que calamine lea nada (el auditor quitó esa comprobación y la de después de leer la tapaba)."""
    fila = lambda r: f'<row r="{r}">' + "".join(f'<c r="{chr(65 + j)}{r}"><v>{j}</v></c>' for j in range(6)) + "</row>"
    libro = _libro(_hoja("".join(fila(r) for r in range(1, 4))))
    monkeypatch.setattr(api_tablas.aislado, "empezar_a_leer", lambda *a, **k: _LecturaInerte())
    monkeypatch.setattr(api_tablas.aislado, "leer_hoja", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("se leyó con calamine")))
    e = _rechazo(convertir, libro, "c.xlsx", fila_de_cabecera=1, topes=Topes(max_columnas=3))
    # deja de mirar en cuanto pasa del tope: el número es una cota, y el motivo lo dice
    assert e.codigo == "demasiadas_columnas" and 3 < e.detalle["columnas"] <= 6
    assert "al menos" in e.motivo_es and "at least" in e.motivo_en
    e = _rechazo(convertir, libro, "c.xlsx", fila_de_cabecera=1, topes=Topes(max_filas=2))
    assert e.codigo == "demasiadas_filas" and e.detalle["filas"] == 3


# ── I5: el formato 00000 ────────────────────────────────────────────────────────────────
@requiere_calamine
def test_un_numero_con_formato_00000_conserva_su_valor_y_se_declara_lo_que_enseña_excel():
    """I5: Excel enseña `01001` para 1001 con formato `00000`. El CSV lleva el VALOR GUARDADO
    (1001: no se aplica el formato, que cambiaría el dato) y se DECLARA que Excel lo enseña
    distinto. 12345 se ve igual y no cuenta."""
    filas = ('<row r="1"><c r="A1" t="inlineStr"><is><t>cp</t></is></c><c r="B1" t="inlineStr"><is><t>ssn</t></is></c></row>'
             '<row r="2"><c r="A2" s="6"><v>1001</v></c><c r="B2" s="7"><v>12345678</v></c></row>'
             '<row r="3"><c r="A3" s="6"><v>12345</v></c><c r="B3" s="7"><v>123456789</v></c></row>'
             '<row r="4"><c r="A4" s="6"><v>7</v></c><c r="B4"><v>1</v></c></row>')
    libro = _libro(_hoja(filas))
    t = convertir(libro, "z.xlsx", fila_de_cabecera=1)
    assert [r[0] for r in _csv(t)[1:]] == ["1001", "12345", "7"]
    d = _decl(t, "formato_con_ceros", "cp")[0]
    assert d["cuantas"] == 2 and d["detalle"]["codigo"] == "00000"
    assert d["detalle"]["ejemplo"] == "1001" and d["detalle"]["visible"] == "01001"
    assert "01001" in d["texto"]["es"] and "01001" in d["texto"]["en"]
    d = _decl(t, "formato_con_ceros", "ssn")[0]
    assert d["cuantas"] == 1 and d["detalle"]["visible"] == "012-34-5678"
    avisos = inspeccionar(libro, "z.xlsx")["hojas"][0]["avisos"]
    assert [a["cuantas"] for a in avisos if a["tipo"] == "formato_con_ceros"] == [2, 1]


# ── Menores ─────────────────────────────────────────────────────────────────────────────
@requiere_calamine
def test_enteros_de_16_cifras_sin_punto_cero_y_los_de_mas_de_2_53_declarados():
    filas = ['<row r="1"><c r="A1" t="inlineStr"><is><t>k</t></is></c><c r="B1" t="inlineStr"><is><t>n</t></is></c></row>']
    for i, v in enumerate(["1234567890123456", "1000000000000000", "9007199254740993", "1E+20", "-0"], start=2):
        filas.append(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c><c r="B{i}"><v>{v}</v></c></row>')
    t = convertir(_libro(_hoja("".join(filas))), "e.xlsx", fila_de_cabecera=1)
    assert [r[1] for r in _csv(t)[1:]] == ["1234567890123456", "1000000000000000", "9007199254740992",
                                          "1e+20", "0"]
    d = _decl(t, "entero_no_exacto", "n")[0]
    assert d["cuantas"] == 2 and d["detalle"]["ejemplos"] == ["9007199254740992", "1e+20"]
    assert ".0" not in t.csv_text


@requiere_calamine
@pytest.mark.parametrize("fecha1904", [False, True])
def test_series_de_fecha_que_no_son_una_fecha_quedan_vacias_y_declaradas(fecha1904):
    """Menores: con formato de fecha, la serie 60 (29-02-1900, que no existió; calamine da el
    28-02, otra fecha), una serie ≤ 0 (calamine da una HORA) y una > 9999-12-31 (calamine da el
    número) NO son una fecha: vacías, y declaradas celda a celda. 59 y 61 son fechas normales."""
    series = ["59", "60", "61", "0", "-5", "3000000", "45000"]
    filas = ['<row r="1"><c r="A1" t="inlineStr"><is><t>k</t></is></c><c r="B1" t="inlineStr"><is><t>f</t></is></c></row>']
    for i, v in enumerate(series, start=2):
        filas.append(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c><c r="B{i}" s="1"><v>{v}</v></c></row>')
    t = convertir(_libro(_hoja("".join(filas)), fecha1904=fecha1904), "f.xlsx", fila_de_cabecera=1)
    salida = {r[0]: r[1] for r in _csv(t)[1:]}
    d = _decl(t, "fecha_no_valida", "f")[0]
    if not fecha1904:
        assert (salida["2"], salida["3"], salida["4"]) == ("1900-02-28", "", "1900-03-01")
        assert salida["5"] == salida["6"] == salida["7"] == "" and salida["8"] == "2023-03-15"
        assert d["cuantas"] == 4
        # 0 es el «día 0» (Excel enseña 00/01/1900), -5 es anterior (####): motivos distintos
        assert d["detalle"]["motivos"] == {"29_02_1900": 1, "anterior_a_la_epoca": 1,
                                           "dia_cero_1900": 1, "posterior_a_9999": 1}
        assert d["detalle"]["celdas"] == ["B3", "B5", "B6", "B7"]
    else:
        # en 1904 la serie 0 ES una fecha (1904-01-01): ya no se anula (2.ª re-auditoría)
        assert salida["3"] == "1904-03-01" and salida["5"] == "1904-01-01"
        assert salida["6"] == salida["7"] == ""
        assert d["detalle"]["motivos"] == {"anterior_a_la_epoca": 1, "posterior_a_9999": 1}
    assert "29-02-1900" in d["texto"]["es"] or fecha1904
    assert ":" not in "".join(salida.values()).replace("00:00:00", "")


@pytest.mark.parametrize("datos,codigo", [
    (b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n", "pdf_no_excel"),
    (b"%PDF-1.5\n%\xd0\xd4\xc5\xd8\n1 0 obj", "pdf_no_excel"),
    (b"<div><span>hola</span></div>", "html_no_excel"),
    (b'<?xml version="1.0"?><datos><a>1</a></datos>', "xml_no_excel"),
    (b'<?xml version="1.0"?><!DOCTYPE html><html><body/></html>', "html_no_excel"),
])
def test_los_motivos_de_formato_dicen_lo_que_es(datos, codigo):
    """Menores: un PDF no es «un fichero de texto (parece un CSV)», un XML no es «una página web»."""
    e = _rechazo(inspeccionar, datos, "x.xlsx")
    assert e.codigo == codigo
    assert ".xlsx" in e.motivo_es or codigo == "html_no_excel"
    assert detectar_formato(datos) in ("pdf", "html", "xml")


# ── B-R1 (2.ª re-auditoría): un `<v>` autocerrado NO tiene texto ─────────────────────────
# El explorador decidía «vacío o no» por la posición en bytes del cierre del `<v>`; en una
# etiqueta autocerrada `expat` da la de DESPUÉS de la etiqueta, y `<v/>` contaba como «con
# texto»: la fórmula sin valor salía como una celda vacía, sin rechazo. openpyxl sin lxml (su
# modo por omisión) escribe justo `<f>A2*2</f><v />`.
_V_VACIOS = ["<v/>", "<v />", '<v xml:space="preserve"></v>', "<v></v>"]


def _con_formula(v: str) -> str:
    return ('<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>b</t></is></c></row>'
            f'<row r="2"><c r="A2"><v>1</v></c><c r="B2"><f>A2*2</f>{v}</c></row>'
            '<row r="3"><c r="A3"><v>3</v></c><c r="B3"><v>4</v></c></row>')


@requiere_calamine
@pytest.mark.parametrize("v", _V_VACIOS)
def test_una_formula_con_v_autocerrado_o_vacio_se_rechaza_sin_valor(v):
    e = _rechazo(convertir, _libro(_hoja(_con_formula(v))), "f.xlsx", fila_de_cabecera=1)
    assert e.codigo == "formula_sin_valor" and e.detalle["celdas"] == ["B2"], v
    # control: con su valor guardado, la misma hoja se convierte
    t = convertir(_libro(_hoja(_con_formula("<v>2</v>"))), "f.xlsx", fila_de_cabecera=1)
    assert _csv(t)[1] == ["1", "2"]


@pytest.mark.parametrize("v", _V_VACIOS)
def test_la_geometria_ve_sin_valor_el_v_vacio_tambien_en_una_hoja_utf16(v):
    """En UTF-16 los bytes no son caracteres: `<v></v>` ocupa 2 bytes por carácter y la cuenta
    por posiciones lo daba «con texto». Se mira en la geometría (el oráculo es la marca)."""
    for codif, enc in (("UTF-8", "utf-8"), ("UTF-16", "utf-16")):
        xml = _hoja(_con_formula(v)).replace('encoding="UTF-8"', f'encoding="{codif}"')
        zf = zipfile.ZipFile(io.BytesIO(_libro(xml.encode(enc))))
        g = xlsx_xml.explorar_hoja(zf, "xl/worksheets/sheet1.xml", xlsx_xml.estilos_del_libro(zf))
        assert g.hallazgos().celdas_sin_valor == ["B2"], (v, codif)
        xml = _hoja(_con_formula("<v>2</v>")).replace('encoding="UTF-8"', f'encoding="{codif}"')
        zf = zipfile.ZipFile(io.BytesIO(_libro(xml.encode(enc))))
        g = xlsx_xml.explorar_hoja(zf, "xl/worksheets/sheet1.xml", xlsx_xml.estilos_del_libro(zf))
        assert g.hallazgos().formulas_sin_valor == 0 and g.n_formulas == 1, codif


@requiere_calamine
@pytest.mark.parametrize("v", ["<v/>", "<v />", "<v xml:space=\"preserve\"/>"])
def test_la_geometria_con_v_autocerrado_da_el_rectangulo_de_calamine(v):
    """Una celda cuyo único contenido es `<v/>` no la cuenta calamine: tampoco la geometría
    (la re-auditoría vio 777 formas distintas de 3.000 con `<v/>` en la mezcla)."""
    datos = ('<row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row>'
             f'<row r="2"><c r="A2"><v>3</v></c><c r="D5">{v}</c></row>')
    libro = _libro(_hoja(datos))
    assert _rect(_geo(libro)) == _rect_de_calamine(libro) == (1, 1, 2, 2)


@requiere_calamine
def test_un_libro_de_openpyxl_sin_lxml_con_formulas_sin_calcular_se_rechaza(tmp_path):
    """Un libro REAL de openpyxl en su modo por omisión (sin lxml: `<v />`). En un proceso
    aparte para fijar `OPENPYXL_LXML=False` antes de importarlo."""
    pytest.importorskip("openpyxl")
    import subprocess
    import sys
    ruta = tmp_path / "o.xlsx"
    codigo = ("import openpyxl,sys\nassert not openpyxl.LXML\nwb=openpyxl.Workbook();ws=wb.active\n"
              "ws.append(['importe','doble'])\n"
              "for i in range(1,4): ws.append([i*10, f'=A{i+1}*2'])\nwb.save(sys.argv[1])\n")
    subprocess.run([sys.executable, "-c", codigo, str(ruta)], check=True,
                   env={**os.environ, "OPENPYXL_LXML": "False"}, timeout=60)
    datos = ruta.read_bytes()
    assert b"<v />" in zipfile.ZipFile(io.BytesIO(datos)).read("xl/worksheets/sheet1.xml")
    e = _rechazo(convertir, datos, "o.xlsx", fila_de_cabecera=1)
    assert e.codigo == "formula_sin_valor" and e.detalle["cuantas"] == 3


# ── I-R1 (2.ª re-auditoría): un libro que pide recalcular al abrir ─────────────────────────
# DECISIÓN DEL SUPERVISOR (revocable por Roberto, escrita en el contrato): XlsxWriter guarda
# `<f>…</f><v>0</v>` con `<calcPr fullCalcOnLoad="1"/>`; el 0 es un relleno, no el resultado.
def _wb_con_calcpr(valor: str) -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?><workbook {_NS}><workbookPr/><sheets>'
            '<sheet name="Hoja1" sheetId="1" r:id="rId1"/></sheets>'
            f'<calcPr calcId="124519" fullCalcOnLoad="{valor}"/></workbook>')


_CON_CERO = ('<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>b</t></is></c></row>'
             '<row r="2"><c r="A2"><v>10</v></c><c r="B2"><f>A2*2</f><v>0</v></c></row>'
             '<row r="3"><c r="A3"><v>20</v></c><c r="B3"><v>7</v></c></row>'
             '<row r="4"><c r="A4" t="inlineStr"><is><t>TOTAL</t></is></c><c r="B4"><f>SUM(B2:B3)</f><v>0</v></c></row>')


@requiere_calamine
@pytest.mark.parametrize("valor", ["1", "true"])
def test_un_libro_que_pide_recalcular_al_abrir_con_formulas_se_rechaza(valor):
    libro = _libro(_hoja(_CON_CERO), wb=_wb_con_calcpr(valor))
    e = _rechazo(convertir, libro, "xw.xlsx", fila_de_cabecera=1)
    assert e.codigo == "formula_sin_valor"
    assert e.detalle["cuantas"] == 2 and e.detalle["celdas"] == ["B2", "B4"]
    assert "recalcular" in e.motivo_es and "LibreOffice" in e.motivo_es
    assert "recalculated" in e.motivo_en and "LibreOffice" in e.motivo_en
    avisos = [a for a in inspeccionar(libro, "xw.xlsx")["hojas"][0]["avisos"]
              if a["tipo"] == "formulas_a_recalcular"]
    assert len(avisos) == 1 and avisos[0]["cuantas"] == 2
    # la fila TOTAL fuera: queda B2, y sigue rechazándose
    assert _rechazo(convertir, libro, "xw.xlsx", fila_de_cabecera=1,
                    filas_a_excluir=[4]).detalle["celdas"] == ["B2"]


@requiere_calamine
def test_pedir_recalcular_sin_formulas_en_lo_convertido_no_rechaza():
    sin_f = _CON_CERO.replace("<f>A2*2</f>", "")
    t = convertir(_libro(_hoja(sin_f), wb=_wb_con_calcpr("1")), "xw.xlsx", fila_de_cabecera=1,
                  filas_a_excluir=[4])
    assert _csv(t)[1] == ["10", "0"]
    # sin calcPr el 0 guardado es el valor (control: la regla es la del atributo)
    t = convertir(_libro(_hoja(_CON_CERO)), "x.xlsx", fila_de_cabecera=1)
    assert _csv(t)[1] == ["10", "0"]
    t = convertir(_libro(_hoja(_CON_CERO), wb=_wb_con_calcpr("0")), "x.xlsx", fila_de_cabecera=1)
    assert _csv(t)[1] == ["10", "0"]


def test_la_variante_recalcular_tiene_singular_y_plural_en_los_dos_idiomas():
    from matrixai.tablas.errores import VARIANTES, rechazar
    assert ("formula_sin_valor", "recalcular") in VARIANTES
    uno = rechazar("formula_sin_valor", variante="recalcular", cuantas=1, celdas=["B2"])
    dos = rechazar("formula_sin_valor", variante="recalcular", cuantas=2, celdas=["B2", "B3"])
    assert uno.codigo == dos.codigo == "formula_sin_valor"
    assert "Hay 1 celda con" in uno.motivo_es and "There is 1 cell with" in uno.motivo_en
    assert "Hay 2 celdas con" in dos.motivo_es and "There are 2 cells with" in dos.motivo_en


# ── Menores de la 2.ª re-auditoría ─────────────────────────────────────────────────────
@requiere_calamine
@pytest.mark.parametrize("excluir", [[2.5], ["3"], [3.0], [99], [3, 40]])
def test_filas_a_excluir_que_no_son_enteros_o_no_existen_se_rechazan(excluir):
    """`int(2.5)` quitaba la fila 2 y declaraba «se quitó 1 fila»; una fila que la hoja no tiene
    se declaraba quitada sin quitar nada."""
    e = _rechazo(convertir, _libro(_hoja(_BASE)), "x.xlsx", fila_de_cabecera=1, filas_a_excluir=excluir)
    assert e.codigo == "parametro_invalido", excluir
    if excluir in ([99], [3, 40]):
        assert "no existe" in e.motivo_es and "3" in e.motivo_es
    t = convertir(_libro(_hoja(_BASE)), "x.xlsx", fila_de_cabecera=1, filas_a_excluir=[3])
    assert _csv(t)[1:] == [["1", "2"]] and _decl(t, "filas_excluidas")[0]["cuantas"] == 1


def test_el_texto_de_entero_no_exacto_dice_desde_donde_salta():
    """Salta desde 2^53 (16 cifras): decir «más de 15 cifras» describía otro umbral."""
    from matrixai.tablas.declaraciones import declarar
    from matrixai.tablas.valores import DOS_A_LA_53, entero_no_exacto
    assert entero_no_exacto(float(DOS_A_LA_53)) and not entero_no_exacto(float(DOS_A_LA_53 - 1))
    d = declarar("entero_no_exacto", columna="id", cuantas=1, ejemplos=["9007199254740992"])
    assert "2^53" in d["texto"]["es"] and "2^53" in d["texto"]["en"]
    assert "más de 15 cifras" not in d["texto"]["es"] and "more than 15 digits" not in d["texto"]["en"]


def test_un_tope_de_tiempo_por_debajo_de_un_segundo_no_dice_mas_de_0_s():
    import signal
    with pytest.raises(TablaRechazada) as e:
        aislado._interpretar(b"", b"", -signal.SIGXCPU, 0.6, None, 0.5)
    assert e.value.codigo == "lector_tiempo"
    assert "más de 0,5 s" in e.value.motivo_es and "0 s" not in e.value.motivo_es.replace("0,5 s", "")
    with pytest.raises(TablaRechazada) as e:
        aislado._interpretar(b"", b"", -signal.SIGXCPU, 121, None, 120.0)
    assert "más de 120 s" in e.value.motivo_es


def test_un_hijo_matado_desde_fuera_no_se_da_por_fichero_danado():
    import signal
    for senal in (signal.SIGKILL, signal.SIGTERM):
        with pytest.raises(TablaRechazada) as e:
            aislado._interpretar(b"", b"", -senal, 1.0, None, 120.0)
        assert e.value.codigo == "lector_abortado"
        assert "dañado" in e.value.motivo_es and "No indica" in e.value.motivo_es, senal
        assert "does not mean" in e.value.motivo_en and senal.name in e.value.motivo_es
    with pytest.raises(TablaRechazada) as e:      # control: un aborto de Rust SÍ apunta al fichero
        aislado._interpretar(b"", b"", -signal.SIGABRT, 1.0, None, 120.0)
    assert "puede estar dañado" in e.value.motivo_es and "No indica" not in e.value.motivo_es


def test_el_hijo_arranca_aislado_con_menos_I(monkeypatch):
    """`-I`: sin variables de entorno ni `site` del usuario ni el directorio actual en `sys.path`
    (un `json.py` plantado al lado no se importa). Si alguien lo quita, esto se pone rojo."""
    vistos = []

    def popen(args, **k):
        vistos.append(list(args))
        raise OSError("no arranca: es la prueba")
    monkeypatch.setattr(aislado, "_rutas_de_calamine", lambda: ["/nada"])
    monkeypatch.setattr(aislado.subprocess, "Popen", popen)
    with pytest.raises(TablaRechazada):
        aislado.Lectura(b"x", {"op": "hojas"}, memoria=None, segundos=1)
    assert vistos and vistos[0][1] == "-I" and vistos[0][2] == "-c", vistos


@requiere_calamine
def test_el_hijo_muere_con_su_padre_pdeathsig():
    """El hijo pone `PR_SET_PDEATHSIG = SIGKILL` y lo comprueba con `PR_GET_PDEATHSIG` (lo que
    quedó puesto, medido en el hijo, no la intención): si el padre muere, el kernel lo mata."""
    import signal
    aislado.listar_hojas(_libro(_hoja(_BASE)), memoria=None, segundos=30)
    assert aislado.ULTIMO_HIJO.get("pdeathsig") == signal.SIGKILL, aislado.ULTIMO_HIJO


@requiere_calamine
def test_un_rango_combinado_repetido_cuenta_una_vez_al_inspeccionar():
    comb = '<mergeCells count="3"><mergeCell ref="A2:A3"/><mergeCell ref="A2:A3"/><mergeCell ref="B2:B3"/></mergeCells>'
    libro = _libro(_hoja(_BASE, despues=comb))
    avisos = [a for a in inspeccionar(libro, "c.xlsx")["hojas"][0]["avisos"] if a["tipo"] == "hay_celdas_combinadas"]
    assert len(avisos) == 1 and avisos[0]["cuantas"] == 2, avisos
    assert avisos[0]["detalle"]["rangos"] == ["A2:A3", "B2:B3"]


def _libro_con_grafico() -> bytes:
    """Un libro con una hoja de datos y una hoja de GRÁFICO (`chartsheet`), las dos visibles."""
    wb = (f'<?xml version="1.0" encoding="UTF-8"?><workbook {_NS}><workbookPr/><sheets>'
          '<sheet name="Datos" sheetId="1" r:id="rId1"/><sheet name="Grafico1" sheetId="2" r:id="rId9"/>'
          '</sheets></workbook>')
    rels = _WBRELS.replace("</Relationships>",
                           '<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chartsheet" '
                           'Target="chartsheets/sheet1.xml"/></Relationships>')
    graf = (f'<?xml version="1.0" encoding="UTF-8"?><chartsheet {_NS}><sheetViews><sheetView workbookViewId="0"/>'
            '</sheetViews></chartsheet>')
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT.replace("</Types>",
                   '<Override PartName="/xl/chartsheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.chartsheet+xml"/></Types>'))
        z.writestr("_rels/.rels", _RELS)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/styles.xml", _STYLES)
        z.writestr("xl/sharedStrings.xml", _SST_VACIA)
        z.writestr("xl/worksheets/sheet1.xml", _hoja(_BASE))
        z.writestr("xl/chartsheets/sheet1.xml", graf)
    return b.getvalue()


@requiere_calamine
def test_una_hoja_de_grafico_no_cuenta_como_hoja_de_datos():
    """Con una hoja de datos y una de gráfico, sin decir la hoja: se elige la de datos (antes,
    «hay varias hojas, di cuál»)."""
    libro = _libro_con_grafico()
    zf = zipfile.ZipFile(io.BytesIO(libro))
    assert [(h["nombre"], h["grafico"]) for h in xlsx_xml.mapa_de_hojas(zf)] == [("Datos", False), ("Grafico1", True)]
    assert inspeccionar(libro, "g.xlsx")["hoja_propuesta"] == "Datos"
    t = convertir(libro, "g.xlsx", fila_de_cabecera=1)
    assert _csv(t) == [["a", "b"], ["1", "2"], ["3", "4"]]


@requiere_calamine
def test_en_la_epoca_1904_la_serie_0_es_el_1_de_enero_de_1904_con_su_hora():
    """calamine da solo la HORA para 0 ≤ serie < 1, y en 1904 eso es el 1904-01-01: Excel lo
    enseña así. Antes se anulaba («anterior al primer día de Excel; Excel enseña ####», falso)."""
    filas = ['<row r="1"><c r="A1" t="inlineStr"><is><t>k</t></is></c><c r="B1" t="inlineStr"><is><t>f</t></is></c></row>']
    for i, (v, s_) in enumerate([("0", 1), ("0.25", 2), ("1", 1), ("-1", 1)], start=2):
        filas.append(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c><c r="B{i}" s="{s_}"><v>{v}</v></c></row>')
    t = convertir(_libro(_hoja("".join(filas)), fecha1904=True), "f.xlsx", fila_de_cabecera=1)
    salida = {r[0]: r[1] for r in _csv(t)[1:]}
    # la columna lleva una fecha con hora: todas se escriben con hora (regla de la columna)
    assert salida["2"] == "1904-01-01 00:00:00" and salida["3"] == "1904-01-01 06:00:00"
    assert salida["4"] == "1904-01-02 00:00:00" and salida["5"] == ""
    solo_cero = convertir(_libro(_hoja(filas[0] + filas[1]), fecha1904=True), "f.xlsx", fila_de_cabecera=1)
    assert _csv(solo_cero)[1] == ["2", "1904-01-01"]
    assert _decl(t, "fecha_no_valida", "f")[0]["detalle"]["motivos"] == {"anterior_a_la_epoca": 1}


@requiere_calamine
def test_en_la_epoca_1900_una_serie_entre_0_y_1_dice_dia_cero_no_almohadillas():
    filas = ('<row r="1"><c r="A1" t="inlineStr"><is><t>k</t></is></c><c r="B1" t="inlineStr"><is><t>f</t></is></c></row>'
             '<row r="2"><c r="A2"><v>1</v></c><c r="B2" s="2"><v>0.5</v></c></row>'
             '<row r="3"><c r="A3"><v>2</v></c><c r="B3" s="1"><v>45000</v></c></row>')
    t = convertir(_libro(_hoja(filas)), "f.xlsx", fila_de_cabecera=1)
    d = _decl(t, "fecha_no_valida", "f")[0]
    assert d["detalle"]["motivos"] == {"dia_cero_1900": 1} and _csv(t)[1] == ["1", ""]
    assert "00/01/1900" in d["texto"]["es"] and "####" not in d["texto"]["es"]
    assert "00/01/1900" in d["texto"]["en"] and "####" not in d["texto"]["en"]
