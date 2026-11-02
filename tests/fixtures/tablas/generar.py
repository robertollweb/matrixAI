# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Cómo se hicieron los Excel sintéticos de `tests/fixtures/tablas/` (150-C1).

SOLO biblioteca estándar: ni `openpyxl` ni `xlwt` (no están declaradas). Un `.xlsx` es un zip
de XML y escribirlo a mano da control de lo que NINGUNA biblioteca deja hacer —una fórmula
SIN valor guardado, una dimensión inflada, una hoja `veryHidden`, el calendario 1904—, que es
justo lo que hay que probar.

**Determinista**: fechas del zip fijas (1980-01-01), orden de miembros fijo, sin marcas de
tiempo. `python3 generar.py` reescribe los binarios de esta carpeta y la prueba
`test_150_c1_fixtures_de_tablas.py` comprueba que lo commiteado == lo que sale de aquí.

Lo que NO se genera aquí y viaja como binario copiado de las sondas del inventario
(`~/.agentes/ola3-20/inventario/ficheros/m1_antiguo.xls`, hecho con `xlwt` 1.3 en una sonda):
el `.xls` BIFF. No hay forma estándar de escribir BIFF sin una biblioteca; es un fichero de
5,6 KB con una hoja `Datos` (cabecera `fecha`,`v`; una fecha con formato `DD/MM/YYYY`, un
número de serie SIN formato 45292.0 y dos decimales).

Uso desde una prueba: `cargar_generador()` devuelve este módulo.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import io
import pathlib
import struct
import sys
import zipfile
from dataclasses import dataclass
from xml.sax.saxutils import escape

AQUI = pathlib.Path(__file__).resolve().parent

# ── Estilos: el índice es el `s=` de la celda ───────────────────────────────────────────
#   0 general · 1 fecha dd/mm/yyyy · 2 fecha y hora · 3 porcentaje 0,0 % · 4 hora h:mm:ss
#   5 duración [h]:mm:ss · 6 texto (@) · 7 fecha ISO yyyy-mm-dd · 8 fecha con fracción
ESTILO = {"general": 0, "fecha": 1, "fecha_hora": 2, "pct": 3, "hora": 4,
          "duracion": 5, "texto": 6, "iso": 7, "fecha_frac": 8}
_NUMFMTS = {164: "dd/mm/yyyy", 165: "yyyy-mm-dd h:mm:ss", 166: "0.0%", 167: "h:mm:ss",
            168: "[h]:mm:ss", 169: "yyyy-mm-dd", 170: "yyyy-mm-dd h:mm:ss.000"}
_XFS = [0, 164, 165, 166, 167, 168, 49, 169, 170]   # numFmtId por índice de estilo


@dataclass(frozen=True)
class Formula:
    """`<f>` con (o sin) valor guardado. `valor=None` → el fichero «de otra herramienta»."""
    f: str
    valor: object = None
    estilo: str = "general"


@dataclass(frozen=True)
class Err:
    """Error de celda: `#N/A`, `#DIV/0!`…"""
    texto: str


@dataclass(frozen=True)
class N:
    """Un número con un estilo (porcentaje, fecha serie…)."""
    v: float
    estilo: str = "general"


@dataclass(frozen=True)
class T:
    """Texto, opcionalmente con el estilo `@` (número guardado como texto)."""
    s: str
    estilo: str = "general"


@dataclass
class Hoja:
    nombre: str
    filas: list                       # lista de listas; None = vacía
    combinadas: tuple = ()            # ("A1:F1", …)
    estado: str = "visible"           # visible | hidden | veryHidden
    filas_ocultas: tuple = ()         # números de fila (1-based)
    columnas_ocultas: tuple = ()      # letras
    dimension: str | None = None      # forzar <dimension ref=…> (inflada)
    celdas_sueltas: tuple = ()        # (("Z1048576", "fin"), …) celdas lejanas


def _col(j: int) -> str:
    s = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        s = chr(65 + r) + s
    return s


def serie_excel(d: dt.datetime | dt.date, epoca_1904: bool = False) -> float:
    if isinstance(d, dt.datetime):
        base = dt.datetime(1904, 1, 1) if epoca_1904 else dt.datetime(1899, 12, 30)
        delta = d - base
        return delta.days + (delta.seconds + delta.microseconds / 1e6) / 86400
    base = dt.date(1904, 1, 1) if epoca_1904 else dt.date(1899, 12, 30)
    return float((d - base).days)


def _num(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def _celda(ref: str, v, compartidas: dict, epoca_1904: bool) -> str:
    if v is None:
        return ""
    if isinstance(v, Formula):
        s = f' s="{ESTILO[v.estilo]}"' if v.estilo != "general" else ""
        f = f"<f>{escape(v.f)}</f>"
        if v.valor is None:
            return f'<c r="{ref}"{s}>{f}</c>'
        if isinstance(v.valor, str):
            return f'<c r="{ref}"{s} t="str">{f}<v>{escape(v.valor)}</v></c>'
        return f'<c r="{ref}"{s}>{f}<v>{_num(v.valor)}</v></c>'
    if isinstance(v, Err):
        return f'<c r="{ref}" t="e"><v>{escape(v.texto)}</v></c>'
    if isinstance(v, bool):
        return f'<c r="{ref}" t="b"><v>{int(v)}</v></c>'
    if isinstance(v, N):
        return f'<c r="{ref}" s="{ESTILO[v.estilo]}"><v>{_num(v.v)}</v></c>'
    if isinstance(v, T):
        i = compartidas.setdefault(v.s, len(compartidas))
        s = f' s="{ESTILO[v.estilo]}"' if v.estilo != "general" else ""
        return f'<c r="{ref}"{s} t="s"><v>{i}</v></c>'
    if isinstance(v, dt.datetime):
        est = "fecha_hora" if (v.hour or v.minute or v.second or v.microsecond) else "fecha"
        if v.microsecond:
            est = "fecha_frac"
        return (f'<c r="{ref}" s="{ESTILO[est]}"><v>{_num(serie_excel(v, epoca_1904))}'
                f"</v></c>")
    if isinstance(v, dt.date):
        return f'<c r="{ref}" s="{ESTILO["fecha"]}"><v>{_num(serie_excel(v, epoca_1904))}</v></c>'
    if isinstance(v, (int, float)):
        return f'<c r="{ref}"><v>{_num(v)}</v></c>'
    if isinstance(v, str):
        i = compartidas.setdefault(v, len(compartidas))
        return f'<c r="{ref}" t="s"><v>{i}</v></c>'
    raise TypeError(f"celda no soportada: {v!r}")


def _xml_hoja(h: Hoja, compartidas: dict, epoca_1904: bool) -> str:
    filas_xml = []
    max_r, max_c = 0, 0
    por_fila: dict[int, list[str]] = {}
    for i, fila in enumerate(h.filas, 1):
        celdas = []
        for j, v in enumerate(fila):
            c = _celda(f"{_col(j)}{i}", v, compartidas, epoca_1904)
            if c:
                celdas.append(c)
                max_c = max(max_c, j + 1)
        if celdas:
            max_r = max(max_r, i)
        por_fila[i] = celdas
    for ref, v in h.celdas_sueltas:
        letras = "".join(ch for ch in ref if ch.isalpha())
        num = int("".join(ch for ch in ref if ch.isdigit()))
        por_fila.setdefault(num, []).append(_celda(ref, v, compartidas, epoca_1904))
    for i in sorted(por_fila):
        attr = ' hidden="1"' if i in h.filas_ocultas else ""
        if por_fila[i] or attr:
            filas_xml.append(f'<row r="{i}"{attr}>{"".join(por_fila[i])}</row>')
    dim = h.dimension or (f"A1:{_col(max_c - 1)}{max_r}" if max_r else "A1")
    cols = ""
    if h.columnas_ocultas:
        def idx(letras):
            n = 0
            for ch in letras:
                n = n * 26 + ord(ch) - 64
            return n
        cols = "<cols>" + "".join(
            f'<col min="{idx(c)}" max="{idx(c)}" width="9" hidden="1" customWidth="1"/>'
            for c in h.columnas_ocultas) + "</cols>"
    merges = ""
    if h.combinadas:
        merges = (f'<mergeCells count="{len(h.combinadas)}">'
                  + "".join(f'<mergeCell ref="{m}"/>' for m in h.combinadas) + "</mergeCells>")
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="{dim}"/>{cols}<sheetData>{"".join(filas_xml)}</sheetData>'
            f"{merges}</worksheet>")


def _estilos_xml() -> str:
    fmts = "".join(f'<numFmt numFmtId="{k}" formatCode="{escape(v)}"/>' for k, v in _NUMFMTS.items())
    xfs = "".join(f'<xf numFmtId="{n}" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
                  for n in _XFS)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<numFmts count="{len(_NUMFMTS)}">{fmts}</numFmts>'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            f'<cellXfs count="{len(_XFS)}">{xfs}</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            "</styleSheet>")


def xlsx(hojas: list[Hoja], *, epoca_1904: bool = False, macro: bool = False,
         sobreescribir: dict[str, bytes] | None = None) -> bytes:
    """Un `.xlsx` válido y determinista. `sobreescribir`: ruta del zip → bytes (para reescribir
    a mano el XML de una hoja, como hace quien exporta desde otra herramienta)."""
    compartidas: dict[str, int] = {}
    hojas_xml = [_xml_hoja(h, compartidas, epoca_1904) for h in hojas]
    ss = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
          '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          f'count="{len(compartidas)}" uniqueCount="{len(compartidas)}">'
          + "".join(f'<si><t xml:space="preserve">{escape(s)}</t></si>'
                    for s, _ in sorted(compartidas.items(), key=lambda kv: kv[1])) + "</sst>")
    wb_pr = '<workbookPr date1904="1"/>' if epoca_1904 else "<workbookPr/>"
    sheets = "".join(
        f'<sheet name="{escape(h.nombre, {chr(34): "&quot;"})}" sheetId="{i}"'
        + (f' state="{h.estado}"' if h.estado != "visible" else "")
        + f' r:id="rId{i}"/>' for i, h in enumerate(hojas, 1))
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f"{wb_pr}<sheets>{sheets}</sheets></workbook>")
    n = len(hojas)
    rels = ("".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        f'relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, n + 1))
        + f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/officeDocument/'
          f'2006/relationships/styles" Target="styles.xml"/>'
          f'<Relationship Id="rId{n + 2}" Type="http://schemas.openxmlformats.org/officeDocument/'
          f'2006/relationships/sharedStrings" Target="sharedStrings.xml"/>')
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               f"{rels}</Relationships>")
    ct_wb = ("application/vnd.ms-excel.sheet.macroEnabled.main+xml" if macro else
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     f'<Override PartName="/xl/workbook.xml" ContentType="{ct_wb}"/>'
                     + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/'
                               f'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                               for i in range(1, n + 1))
                     + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                     '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
                     "</Types>")
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                 'relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    miembros = {"[Content_Types].xml": content_types, "_rels/.rels": root_rels,
                "xl/workbook.xml": workbook, "xl/_rels/workbook.xml.rels": wb_rels,
                "xl/styles.xml": _estilos_xml(), "xl/sharedStrings.xml": ss}
    for i, x in enumerate(hojas_xml, 1):
        miembros[f"xl/worksheets/sheet{i}.xml"] = x
    if macro:
        miembros["xl/vbaProject.bin"] = b"\xd0\xcf\x11\xe0 macro de mentira (nunca se ejecuta)"
    datos = {k: (v.encode("utf-8") if isinstance(v, str) else v) for k, v in miembros.items()}
    for k, v in (sobreescribir or {}).items():
        datos[k] = v
    return zip_determinista(datos)


def zip_determinista(miembros: dict[str, bytes], *, nivel: int = 6) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=nivel) as z:
        for nombre, b in miembros.items():
            zi = zipfile.ZipInfo(nombre, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            z.writestr(zi, b)
    return buf.getvalue()


def reescribir_hoja(xlsx_bytes: bytes, n_hoja: int, funcion) -> bytes:
    """Aplica `funcion(xml:str) -> str` al XML de la hoja `n_hoja` (1-based), como cuando alguien
    «arregla a mano» lo que su herramienta guardó."""
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as z:
        datos = {i.filename: z.read(i.filename) for i in z.infolist()}
    ruta = f"xl/worksheets/sheet{n_hoja}.xml"
    datos[ruta] = funcion(datos[ruta].decode("utf-8")).encode("utf-8")
    return zip_determinista(datos)


def declarar_tamano(zip_bytes: bytes, miembro: str, tamano: int) -> bytes:
    """Reescribe el tamaño DESCOMPRIMIDO que el directorio central declara para `miembro`, sin
    tocar su contenido: kilobytes que dicen ser gigabytes (la bomba) o gigabytes que dicen ser
    kilobytes (la bomba que miente). Un lector que se fíe del directorio se equivoca."""
    ba = bytearray(zip_bytes)
    pos, cambiados = 0, 0
    while True:
        pos = ba.find(b"PK\x01\x02", pos)
        if pos < 0:
            break
        n = struct.unpack_from("<H", ba, pos + 28)[0]
        if bytes(ba[pos + 46:pos + 46 + n]).decode("utf-8") == miembro:
            struct.pack_into("<I", ba, pos + 24, tamano)
            cambiados += 1
        pos += 4
    assert cambiados == 1, f"no encontré {miembro!r} (o está repetido)"
    return bytes(ba)


def marcar_cifrado(zip_bytes: bytes) -> bytes:
    """Pone a 1 el bit de «cifrado» (bit 0) de las cabeceras locales y del directorio central, como
    un zip protegido con contraseña (`zipfile` lo borra al escribir, de ahí los bytes a mano)."""
    ba = bytearray(zip_bytes)
    for firma, desplazamiento in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        pos = 0
        while True:
            pos = ba.find(firma, pos)
            if pos < 0:
                break
            ba[pos + desplazamiento] |= 0x01
            pos += 4
    return bytes(ba)


def grande(filas: int, columnas: int, semilla: int = 1) -> bytes:
    """Una hoja densa de la forma que trae la gente: cabecera, números, enteros y textos."""
    import random
    r = random.Random(semilla)
    datos = [[f"c{j}" for j in range(columnas)]]
    for _ in range(filas):
        datos.append([f"t{r.randint(0, 50)}" if j % 3 == 0 else (r.randint(0, 1000) if j % 3 == 1
                      else round(r.random() * 100, 4)) for j in range(columnas)])
    return xlsx([Hoja("Datos", datos)])


def ole_cifrado() -> bytes:
    """Un contenedor OLE2 (CFB v3) con los dos flujos que Excel crea al proteger un libro con
    contraseña: `EncryptionInfo` y `EncryptedPackage`. Vacíos: solo importa su NOMBRE en el
    directorio, que es lo que mira el producto (y lo que distingue un `.xlsx` cifrado de un
    `.xls` antiguo, que también empieza por D0CF11E0)."""
    FIN, FAT_, LIBRE = 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF
    cab = bytearray(512)
    cab[0:8] = bytes.fromhex("D0CF11E0A1B11AE1")
    struct.pack_into("<HHHHHH", cab, 24, 0x3E, 3, 0xFFFE, 9, 6, 0)   # menor, mayor, orden, shift
    struct.pack_into("<IIIIIIIII", cab, 40, 0, 1, 1, 0, 4096, FIN, 0, FIN, 0)
    # ↑ dir sectores(v3=0), nº FAT=1, 1.º sector dir=1, (transacción)0, cutoff 4096,
    #   1.º miniFAT=FIN, nº miniFAT=0, 1.º DIFAT=FIN, nº DIFAT=0
    difat = [0] + [LIBRE] * 108
    struct.pack_into("<109I", cab, 76, *difat)
    fat = [FAT_, FIN] + [LIBRE] * 126
    sec_fat = struct.pack("<128I", *fat)

    def entrada(nombre, tipo, izq=LIBRE, der=LIBRE, hijo=LIBRE, inicio=FIN, tam=0):
        b = bytearray(128)
        u = nombre.encode("utf-16-le")
        b[0:len(u)] = u
        struct.pack_into("<H", b, 64, len(u) + 2)
        b[66] = tipo
        b[67] = 1
        struct.pack_into("<III", b, 68, izq, der, hijo)
        struct.pack_into("<I", b, 116, inicio)
        struct.pack_into("<Q", b, 120, tam)
        return bytes(b)
    d = (entrada("Root Entry", 5, hijo=1) + entrada("EncryptionInfo", 2, der=2)
         + entrada("EncryptedPackage", 2) + bytes(128))
    return bytes(cab) + sec_fat + d


# ── Los ficheros ────────────────────────────────────────────────────────────────────────
def _oficina(*, importe_con_valor: bool = True, total_con_valor: bool = True) -> bytes:
    """Portada + hoja `Datos`: título combinado, cabecera en la fila 3, fechas con formato,
    una combinada vertical (B4:B5), fórmulas y una fila TOTAL con fórmulas."""
    tiendas = ["Madrid", None, None, "Bilbao", "Bilbao", "Sevilla"]
    filas: list = [["VENTAS POR TIENDA"], ["(cifras en euros)"],
                   ["Fecha", "Tienda", "Unidades", "Precio", "Importe", "Abandona"]]
    importes = []
    for i in range(6):
        r = 4 + i
        u, p = 10 + i, 2.5 + i / 10
        importes.append(u * p)
        filas.append([dt.date(2024, 1, 5 + i), tiendas[i], u, p,
                      Formula(f"C{r}*D{r}", (u * p) if importe_con_valor else None),
                      ["Sí", "No", "Sí", "No", "No", "Sí"][i]])
    filas.append(["TOTAL", None, Formula("SUM(C4:C9)", 75 if total_con_valor else None), None,
                  Formula("SUM(E4:E9)", sum(importes) if total_con_valor else None), None])
    return xlsx([Hoja("Portada", [["Informe de ventas 2024 - NO TOCAR"], [None], ["Ver hoja Datos"]]),
                 Hoja("Datos", filas, combinadas=("A1:F1", "B4:B5"))])


def _tipos_raros() -> bytes:
    return xlsx([Hoja("Hoja1", [
        ["id", "serie_sin_formato", "fecha_texto", "pct", "coma_texto", "num_texto", "flag",
         "error", "mixta"],
        [1, 45292, "05/01/2024", N(0.153, "pct"), "1,5", T("007", "texto"), True,
         Err("#DIV/0!"), 3],
        [2, 45293.75, "13/01/2024", N(0.2, "pct"), "2,25", T("012", "texto"), False,
         Err("#N/A"), "tres"],
        [3, None, "2024-02-30", N(1.0, "pct"), "1.234,56", T("1E3", "texto"), None, None,
         dt.datetime(2024, 1, 1)],
    ])])


def _fechas(epoca_1904: bool) -> bytes:
    return xlsx([Hoja("Fechas", [
        ["fecha", "momento", "hora", "duracion", "v"],
        [dt.date(2024, 1, 5), dt.datetime(2024, 1, 5, 13, 30, 15), N(0.5625, "hora"),
         N(1.1, "duracion"), 1],
        [dt.date(2024, 3, 1), dt.datetime(2024, 3, 1, 8, 0, 0), N(0.999988425925926, "hora"),
         N(0.25, "duracion"), 2],
        [dt.datetime(2024, 3, 1, 0, 0, 0, 123000), None, None, None, 3],
    ])], epoca_1904=epoca_1904)


def _varias_hojas() -> bytes:
    def mes(n):
        return [["x", "y"]] + [[i, i * 2] for i in range(5)]
    return xlsx([
        Hoja("Enero", mes("Enero"), filas_ocultas=(3,), columnas_ocultas=("B",)),
        Hoja("Febrero", mes("Febrero")),
        Hoja("Calculos", [["aux"], [1]], estado="hidden"),
        Hoja("Secreta", [["aux"], [2]], estado="veryHidden"),
    ])


def _celda_perdida() -> bytes:
    """Una tabla de 5 filas y una celda con dato perdida en `Z1048576` (la «dimensión inflada»
    del inventario): el rectángulo que el lector reservaría es de 27 millones de celdas."""
    return xlsx([Hoja("Enero", [["x", "y"]] + [[i, i * 2] for i in range(5)],
                      celdas_sueltas=(("Z1048576", "fin"),), dimension="A1:Z1048576")])


def _cabecera_dos_filas() -> bytes:
    return xlsx([Hoja("Ventas", [
        ["Producto", "Ventas", None, "Coste", None],
        [None, "Enero", "Febrero", "Enero", "Febrero"],
        ["Tornillo", 10, 12, 4, 5],
        ["Tuerca", 7, 9, 3, 4],
        ["Arandela", 5, 6, 2, 2],
    ], combinadas=("A1:A2", "B1:C1", "D1:E1"))])


def _cabecera_en_fila_15() -> bytes:
    filas: list = [["Cuadro de mando"], [], ["Ventas del año", 12345.6], ["Margen", N(0.231, "pct")],
                   ["Clientes", 321], []]
    while len(filas) < 14:
        filas.append([])
    filas.append(["Region", "Mes", "Unidades", "Importe"])
    datos = [("Norte", "Enero", 10, 100.5), ("Norte", "Febrero", 12, 120.0),
             ("Sur", "Enero", 8, 80.25), ("Sur", "Febrero", 9, 90.0)]
    for d in datos:
        filas.append(list(d))
    return xlsx([Hoja("Cuadro", filas, combinadas=("A1:D1",))])


def _erp_totales() -> bytes:
    """Subtotales en medio (ERP), TOTAL GENERAL al final, fila vacía y un promedio."""
    filas: list = [["Cliente", "Producto", "Unidades", "Importe"]]
    filas += [["Ana", "A", 1, 10.0], ["Ana", "B", 2, 20.0],
              ["Subtotal Ana", None, Formula("SUM(C2:C3)", 3), Formula("SUM(D2:D3)", 30.0)],
              ["Luis", "A", 4, 40.0], ["Luis", "B", 5, 50.0],
              ["Subtotal Luis", None, Formula("SUM(C5:C6)", 9), Formula("SUM(D5:D6)", 90.0)],
              [None, None, None, None],
              ["Total general", None, Formula("C4+C7", 12), Formula("D4+D7", 120.0)],
              ["Promedio", None, 2.4, 24.0]]
    return xlsx([Hoja("Pedidos", filas)])


def _nombres() -> bytes:
    return xlsx([Hoja("Año", [
        ["AC", "AC", None, "Niño", "vacía", "Precio €", None],
        [1, 2, "x", "señor", None, "1,5", 7],
        [3, 4, "y", "pingüino", None, "2,5", 8],
        [5, 6, "z", "ñandú", None, "3,5", 9],
    ])])


def _formulas_sin_valor_en_total() -> bytes:
    return _oficina(importe_con_valor=True, total_con_valor=False)


def _booleanos_y_errores() -> bytes:
    return xlsx([Hoja("Datos", [
        ["a", "b", "c"],
        [1, True, Err("#N/A")],
        [2, False, Err("#DIV/0!")],
        [3, True, 5],
    ])])


def _ceros_y_vacios() -> bytes:
    return xlsx([Hoja("Datos", [
        ["a", "b"],
        [1, None],
        [None, None],
        [3, 4],
        [10.0, 2.5],
    ])])


def todos() -> dict[str, bytes]:
    d = {
        "oficina_con_valor.xlsx": _oficina(),
        "oficina_sin_valor.xlsx": _oficina(importe_con_valor=False, total_con_valor=False),
        "oficina_total_sin_valor.xlsx": _formulas_sin_valor_en_total(),
        "tipos_raros.xlsx": _tipos_raros(),
        "fechas_1900.xlsx": _fechas(False),
        "fechas_1904.xlsx": _fechas(True),
        "varias_hojas.xlsx": _varias_hojas(),
        "celda_perdida.xlsx": _celda_perdida(),
        "cabecera_dos_filas.xlsx": _cabecera_dos_filas(),
        "cabecera_en_fila_15.xlsx": _cabecera_en_fila_15(),
        "erp_totales.xlsx": _erp_totales(),
        "nombres.xlsx": _nombres(),
        "booleanos_y_errores.xlsx": _booleanos_y_errores(),
        "ceros_y_vacios.xlsx": _ceros_y_vacios(),
        "macro.xlsm": xlsx([Hoja("Datos", [["a", "b"], [1, 2]])], macro=True),
        # ── defectuosos ──
        "html_como.xls": (b"<html><table><tr><th>a</th><th>b</th></tr>"
                          b"<tr><td>1</td><td>2,5</td></tr></table></html>"),
        "csv_renombrado.xlsx": b"a,b\n1,2\n",
        "vacio.xlsx": b"",
        "cifrado.xlsx": ole_cifrado(),
    }
    d["truncado.xlsx"] = d["oficina_con_valor.xlsx"][: len(d["oficina_con_valor.xlsx"]) // 2]
    return d


def escribir(destino: pathlib.Path = AQUI) -> list[str]:
    nombres = []
    for nombre, b in todos().items():
        (destino / nombre).write_bytes(b)
        nombres.append(nombre)
    return nombres


def cargar_generador():
    """Para las pruebas: este mismo módulo, cargado por ruta (la carpeta no es un paquete)."""
    if "generar_tablas_150" in sys.modules:
        return sys.modules["generar_tablas_150"]
    spec = importlib.util.spec_from_file_location("generar_tablas_150", AQUI / "generar.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generar_tablas_150"] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    print("\n".join(escribir()))
