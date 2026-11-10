# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que calamine no da, leído del XML del zip con la biblioteca estándar (150-C1, DS2).

Medido en el Paso 0 con `python-calamine 0.9.0`: SÍ da la visibilidad de las hojas
(`sheets_metadata`) y las celdas combinadas (`merged_cell_ranges`); NO distingue una celda de
fórmula SIN valor guardado de una vacía, devuelve `''` para un error de celda (`#N/A`) sin decir
cuál, no dice qué celdas son porcentajes, fechas o códigos con ceros ni qué filas/columnas están
ocultas, y reserva en Rust el rectángulo ENTERO del rango (una celda suelta en `Z1048576` cuesta
832 MB). Eso es lo que hace este módulo, **sin `openpyxl`**, en UNA pasada de `expat` (streaming:
el XML nunca está entero en memoria) por hoja:

* la **geometría**: el rectángulo de las celdas que calamine cuenta, con SUS reglas (medidas el
  2026-11-02, `~/.agentes/150/rep-c1/sondas/cal_formas*.py`): nombres LOCALES (`x:c` es `c`; el
  espacio de nombres no importa), atributos en cualquier orden, la referencia `r` en mayúsculas o
  minúsculas y OPCIONAL (sin ella, la celda va a la columna siguiente a la anterior de su fila, y
  una fila sin `r` es la siguiente a la anterior). Una celda cuenta si tiene `<is>`, o un `<v>`
  con texto (salvo `t="inlineStr"`, que ignora su `<v>`), o un `<v>` vacío con `t="str"`/`"d"`.
  Sirve para la guarda de memoria ANTES de que calamine reserve nada; **no decide que una hoja
  está vacía** (eso solo lo dice calamine, `api._Libro.leer`).
* las **marcas** de cada celda que importa: fórmulas sin valor, errores, porcentajes, números
  con formato de ceros a la izquierda, series de fecha que no son una fecha; filas ocultas (todas,
  en un `array`) y columnas ocultas (por RANGOS: `<col min max>`, nunca expandidas).

Defensas: un `<!DOCTYPE>` o una declaración de entidad se rechazan en el propio analizador (sea
cual sea la codificación del XML y caiga donde caiga el borde de un trozo); ningún trozo se lee más
allá de lo que declara el directorio central; las listas que crecen con el fichero son `array` de
enteros, con un tope de definiciones de columna.
"""
from __future__ import annotations

import array
import bisect
import re
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
import zipfile
from collections import Counter
from dataclasses import dataclass, field

from .errores import TablaRechazada, rechazar

__all__ = ["mapa_de_hojas", "estilos_del_libro", "Estilos", "explorar_hoja", "Geometria",
           "Hallazgos", "indice_de_columna", "letras_de_columna", "referencia",
           "cadenas_compartidas", "es_fecha_1904", "pide_recalcular_al_abrir"]

_TROZO = 1 << 20
_MAX_XML_PEQUENO = 64 << 20
#: Una hoja de Excel tiene como mucho 16.384 columnas: más definiciones `<col>` que eso no las
#: escribe Excel (cada una cubre un rango que no se solapa con otro).
MAX_DEFINICIONES_DE_COLUMNA = 16_384
_MAX_FILAS_FISICAS = 1_048_576
_MAX_TIPOS_DE_ERROR = 64


def indice_de_columna(letras: str) -> int:
    """`A` → 1, `Z` → 26, `AA` → 27 (1-based). Mayúsculas o minúsculas, como calamine."""
    n = 0
    for ch in letras.upper():
        n = n * 26 + ord(ch) - 64
    return n


def letras_de_columna(n: int) -> str:
    """1 → `A`, 27 → `AA`."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def referencia(fila: int, columna: int) -> str:
    """Fila y columna (1-based) → `D7`."""
    return f"{letras_de_columna(columna)}{fila}"


class _LectorSeguro:
    """Lee un miembro del zip por trozos, sin pasar de lo que declara el directorio central."""

    def __init__(self, zf: zipfile.ZipFile, ruta: str):
        try:
            self._info = zf.getinfo(ruta)
            self._f = zf.open(self._info)
        except (KeyError, zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError,
                ValueError, EOFError):
            raise rechazar("zip_danado") from None
        self._leidos = 0

    def read(self, n: int = -1) -> bytes:
        try:
            b = self._f.read(n if n and n > 0 else _TROZO)
        except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, ValueError,
                NotImplementedError):
            raise rechazar("zip_danado") from None
        self._leidos += len(b)
        if self._leidos > self._info.file_size:
            raise rechazar("zip_danado")
        return b

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:      # cerrar un flujo dañado no añade nada que decir
            pass


def _prohibido(*_a):
    raise rechazar("lector_fallo", tipo="xml con DTD o entidades", tipo_en="XML with a DTD or entities")


def _expat_seguro() -> expat.XMLParserType:
    """Un analizador `expat` SIN espacios de nombres (como calamine: compara nombres locales) y
    que rechaza cualquier DTD o declaración de entidad (ahí viven las que se expanden)."""
    p = expat.ParserCreate()
    p.StartDoctypeDeclHandler = _prohibido
    p.EntityDeclHandler = _prohibido
    p.UnparsedEntityDeclHandler = _prohibido
    p.ExternalEntityRefHandler = _prohibido
    p.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    p.buffer_text = True
    return p


def _leer_pequeno(zf: zipfile.ZipFile, ruta: str) -> bytes:
    try:
        tam = zf.getinfo(ruta).file_size
    except KeyError:
        raise rechazar("zip_danado") from None
    if tam > _MAX_XML_PEQUENO:
        raise rechazar("lector_fallo", tipo="parte demasiado grande", tipo_en="part too large")
    lector = _LectorSeguro(zf, ruta)
    try:
        partes = []
        while True:
            b = lector.read(_TROZO)
            if not b:
                break
            partes.append(b)
        return b"".join(partes)
    finally:
        lector.close()


def _local(nombre: str) -> str:
    """El nombre local: `x:sheet` → `sheet`, `{ns}sheet` → `sheet`, `r:id` → `id`."""
    return nombre.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _parsear(b: bytes) -> ET.Element:
    """Un árbol de `ElementTree` construido con el analizador seguro (sin DTD ni entidades)."""
    p = _expat_seguro()
    tb = ET.TreeBuilder()
    p.StartElementHandler = tb.start
    p.EndElementHandler = tb.end
    p.CharacterDataHandler = tb.data
    try:
        p.Parse(b, True)
        return tb.close()
    except (expat.ExpatError, LookupError, ValueError, UnicodeError):
        # LookupError: una declaración `encoding="…"` que no existe (cazado por el fuzz).
        raise rechazar("lector_fallo", tipo="xml mal formado", tipo_en="malformed XML") from None


def _relaciones(zf: zipfile.ZipFile, nombres: set) -> list[tuple[str, str, str]]:
    """`(Id, Type, ruta)` de `xl/_rels/workbook.xml.rels` (ruta normalizada dentro del zip)."""
    if "xl/_rels/workbook.xml.rels" not in nombres:
        return []
    rels = _parsear(_leer_pequeno(zf, "xl/_rels/workbook.xml.rels"))
    salida = []
    for r in rels.iter():
        if _local(r.tag) != "Relationship":
            continue
        rid, destino, tipo = r.get("Id"), r.get("Target"), r.get("Type") or ""
        if rid and destino:
            destino = destino.lstrip("/") if destino.startswith("/") else "xl/" + destino
            salida.append((rid, tipo, _normalizar_ruta(destino)))
    return salida


def mapa_de_hojas(zf: zipfile.ZipFile) -> list[dict]:
    """Las hojas de `xl/workbook.xml` en su orden: `{nombre, estado, ruta}`.

    `estado` es `visible`, `hidden` o `veryHidden`. `ruta` es el miembro del zip con su XML (o
    `None` si el libro apunta a algo que no existe: una hoja de gráfico, un enlace roto).
    """
    nombres = set(zf.namelist())
    wb = _parsear(_leer_pequeno(zf, "xl/workbook.xml"))
    rels = _relaciones(zf, nombres)
    objetivos = {rid: ruta for rid, _t, ruta in rels}
    graficos = {rid for rid, t, _r in rels if (t or "").rsplit("/", 1)[-1].lower() == "chartsheet"}
    hojas = []
    for el in wb.iter():
        if _local(el.tag) != "sheet":
            continue
        rid = None
        for k, v in el.attrib.items():
            if k != "id" and _local(k) == "id":
                rid = v
        ruta = objetivos.get(rid)
        hojas.append({"nombre": el.get("name") or "", "estado": el.get("state") or "visible",
                      "ruta": ruta if ruta in nombres else None,
                      # una hoja de GRÁFICO no es una hoja de datos: no cuenta al elegir
                      "grafico": rid in graficos})
    return hojas


def es_fecha_1904(zf: zipfile.ZipFile) -> bool:
    wb = _parsear(_leer_pequeno(zf, "xl/workbook.xml"))
    for el in wb.iter():
        if _local(el.tag) == "workbookPr":
            return (el.get("date1904") or "").lower() in ("1", "true")
    return False


def pide_recalcular_al_abrir(zf: zipfile.ZipFile) -> bool:
    """`<calcPr fullCalcOnLoad="1"/>` en `workbook.xml`: el libro pide recalcular TODAS sus
    fórmulas al abrirse, y lo que guarda en cada `<v>` no es el resultado (XlsxWriter guarda 0).
    I-R1 de 150-C1."""
    wb = _parsear(_leer_pequeno(zf, "xl/workbook.xml"))
    for el in wb.iter():
        if _local(el.tag) == "calcPr":
            return (el.get("fullCalcOnLoad") or "").strip().lower() in ("1", "true")
    return False


def cadenas_compartidas(zf: zipfile.ZipFile) -> int:
    """Bytes descomprimidos de la tabla de textos compartidos (`sharedStrings.xml`) que calamine
    carga ENTERA en memoria al abrir el libro, la use una hoja o no."""
    nombres = set(zf.namelist())
    rutas = {ruta for _rid, tipo, ruta in _relaciones(zf, nombres)
             if tipo.rsplit("/", 1)[-1] == "sharedStrings"}
    rutas.add("xl/sharedStrings.xml")
    return sum(zf.getinfo(r).file_size for r in rutas if r in nombres)


def _normalizar_ruta(ruta: str) -> str:
    partes: list[str] = []
    for p in ruta.split("/"):
        if p == "..":
            if partes:
                partes.pop()
        elif p and p != ".":
            partes.append(p)
    return "/".join(partes)


# ── Estilos: qué índice `s=` es porcentaje, fecha o un código con ceros ────────────────
_FORMATOS_PORCENTAJE = {9, 10}
#: Formatos de fábrica de Excel con DÍA (no solo hora): 14–17, 22 y los de Asia oriental.
_FORMATOS_FECHA = {14, 15, 16, 17, 22, *range(27, 37), *range(50, 59)}
_RE_LITERAL = re.compile(r'"[^"]*"|\\.|\[[^\]]*\]')

PORCENTAJE, FECHA, CEROS = 1, 2, 4


@dataclass(frozen=True)
class Estilos:
    """Por índice de estilo (`s=`): su clase (bits `PORCENTAJE|FECHA|CEROS`), y para CEROS el
    número de cifras que enseña Excel y el código de formato."""
    clase: tuple = ()
    ceros: dict = field(default_factory=dict)   # índice → (cifras, código)

    def de(self, s) -> int:
        try:
            return self.clase[int(s)]
        except (TypeError, ValueError, IndexError):
            return 0


def _clase_de_codigo(codigo: str) -> tuple[int, int]:
    """`(clase, cifras)` de un código de formato personalizado."""
    seccion = _RE_LITERAL.sub("", codigo.split(";", 1)[0])
    if "%" in seccion:
        return PORCENTAJE, 0
    bajo = seccion.lower()
    if "d" in bajo or "y" in bajo:
        return FECHA, 0
    entera = seccion.split(".", 1)[0]
    cifras = entera.count("0")
    if cifras >= 2 and "e" not in bajo:
        return CEROS, cifras
    return 0, 0


def estilos_del_libro(zf: zipfile.ZipFile) -> Estilos:
    if "xl/styles.xml" not in zf.namelist():
        return Estilos()
    raiz = _parsear(_leer_pequeno(zf, "xl/styles.xml"))
    personalizados: dict[int, str] = {}
    xfs: list[int] = []
    for el in raiz.iter():
        t = _local(el.tag)
        if t == "numFmt" and (el.get("numFmtId") or "").isdigit():
            personalizados[int(el.get("numFmtId"))] = el.get("formatCode") or ""
        elif t == "cellXfs":
            for xf in el:
                if _local(xf.tag) == "xf":
                    n = xf.get("numFmtId") or "0"
                    xfs.append(int(n) if n.isdigit() else 0)
    clase, ceros = [], {}
    for i, nf in enumerate(xfs):
        if nf in personalizados:
            c, cifras = _clase_de_codigo(personalizados[nf])
            if c == CEROS:
                ceros[i] = (cifras, personalizados[nf])
        elif nf in _FORMATOS_PORCENTAJE:
            c = PORCENTAJE
        elif nf in _FORMATOS_FECHA:
            c = FECHA
        else:
            c = 0
        clase.append(c)
    return Estilos(tuple(clase), ceros)


# ── La pasada ───────────────────────────────────────────────────────────────────────────
_RE_REF = re.compile(r"([A-Za-z]+)([0-9]+)")
_RE_DIMENSION = re.compile(r"[A-Za-z]+([0-9]+)$")

#: Clases de serie de fecha que NO son una fecha (lo que calamine escribiría es otra cosa).
FECHA_ANTES_DE_1900 = "anterior_a_la_epoca"     # serie < 0: Excel enseña ####
FECHA_DIA_CERO_1900 = "dia_cero_1900"            # 0 ≤ serie < 1 en 1900: Excel enseña 00/01/1900
#: 0 ≤ serie < 1 en la época 1904 SÍ es una fecha (1904-01-01, con su hora), pero calamine da
#: solo la HORA: no se anula, se corrige al convertir (menor de la 2.ª re-auditoría de 150-C1).
FECHA_DIA_CERO_1904 = "dia_cero_1904"
FECHA_29_FEBRERO_1900 = "29_02_1900"              # serie 60 en 1900: un día que no existió
FECHA_DESPUES_DE_9999 = "posterior_a_9999"        # Excel enseña ####
_FIN_DE_EXCEL_1900 = 2_958_466      # 10000-01-01
_FIN_DE_EXCEL_1904 = 2_958_466 - 1_462


def _clase_de_fecha(x: float, epoca_1904: bool) -> str | None:
    if epoca_1904:
        if x < 0:
            return FECHA_ANTES_DE_1900
        if x < 1:        # calamine escribe una HORA donde Excel enseña 1904-01-01
            return FECHA_DIA_CERO_1904
        if x >= _FIN_DE_EXCEL_1904:
            return FECHA_DESPUES_DE_9999
        return None
    if x < 0:
        return FECHA_ANTES_DE_1900
    if x < 1:
        return FECHA_DIA_CERO_1900
    if 60 <= x < 61:
        return FECHA_29_FEBRERO_1900
    if x >= _FIN_DE_EXCEL_1900:
        return FECHA_DESPUES_DE_9999
    return None


class _Marcas:
    """Lo que se apunta celda a celda, en `array` de enteros (filas y columnas 1-based)."""

    def __init__(self):
        self.fsv_f, self.fsv_c = array.array("l"), array.array("l")       # fórmula sin valor
        self.frm_f, self.frm_c = array.array("l"), array.array("l")       # toda fórmula
        self.err_f, self.err_c, self.err_t = array.array("l"), array.array("l"), array.array("l")
        self.tipos_error: list[str] = []
        self.pct_f, self.pct_c = array.array("l"), array.array("l")
        self.cer_f, self.cer_c, self.cer_s = array.array("l"), array.array("l"), array.array("l")
        self.cer_v: dict[int, str] = {}       # columna → el primer valor (para el ejemplo)
        self.fec_f, self.fec_c, self.fec_k = array.array("l"), array.array("l"), array.array("l")
        self.d04_f, self.d04_c = array.array("l"), array.array("l")       # día 0 de 1904

    def tipo_de_error(self, texto: str) -> int:
        texto = texto.strip() or "#ERROR"
        try:
            return self.tipos_error.index(texto)
        except ValueError:
            if len(self.tipos_error) >= _MAX_TIPOS_DE_ERROR:
                texto = "#ERROR"
                if texto in self.tipos_error:
                    return self.tipos_error.index(texto)
            self.tipos_error.append(texto)
            return len(self.tipos_error) - 1


_CLASES_FECHA = (FECHA_ANTES_DE_1900, FECHA_29_FEBRERO_1900, FECHA_DESPUES_DE_9999,
                 FECHA_DIA_CERO_1900)


@dataclass
class Hallazgos:
    """Las marcas de las filas que se piden, ya contadas."""
    formulas_sin_valor: int = 0
    celdas_sin_valor: list[str] = field(default_factory=list)       # hasta 10, `D7`
    formulas: int = 0                                               # con o sin valor guardado
    celdas_con_formula: list[str] = field(default_factory=list)     # hasta 10
    errores_por_columna: dict[int, Counter] = field(default_factory=dict)  # col → texto → n
    porcentajes_por_columna: Counter = field(default_factory=Counter)      # col → n
    ceros_por_columna: dict[int, dict] = field(default_factory=dict)       # col → {n, cifras, codigo, ejemplo}
    fechas_no_validas_por_columna: dict[int, Counter] = field(default_factory=dict)  # col → clase → n
    celdas_de_fecha_no_valida: dict[int, list[str]] = field(default_factory=dict)    # col → hasta 10


@dataclass
class Geometria:
    """El rectángulo de las celdas que calamine cuenta (1-based) y las marcas de la hoja."""
    min_fila: int | None = None
    max_fila: int | None = None
    min_col: int | None = None
    max_col: int | None = None
    celdas_con_valor: int = 0
    n_formulas: int = 0      # celdas con fórmula (con o sin valor guardado)
    filas_ocultas: array.array = field(default_factory=lambda: array.array("l"))  # ordenadas, sin repetir
    columnas_ocultas: list = field(default_factory=list)   # [(desde, hasta)], fundidos y ordenados
    exacta: bool = True      # False si alguna celda tenía una referencia que no se entiende
    marcas: _Marcas = field(default_factory=_Marcas)
    estilos: Estilos = field(default_factory=Estilos)

    @property
    def vacia(self) -> bool:
        return self.min_fila is None

    @property
    def filas(self) -> int:
        return 0 if self.vacia else self.max_fila - self.min_fila + 1

    @property
    def columnas(self) -> int:
        return 0 if self.vacia else self.max_col - self.min_col + 1

    @property
    def celdas_del_rango(self) -> int:
        return self.filas * self.columnas

    @property
    def n_filas_ocultas(self) -> int:
        return len(self.filas_ocultas)

    @property
    def hay_formulas(self) -> bool:
        return self.n_formulas > 0

    @property
    def hay_errores(self) -> bool:
        return len(self.marcas.err_f) > 0

    @property
    def hay_porcentajes(self) -> bool:
        return len(self.marcas.pct_f) > 0

    def filas_ocultas_en(self, desde: int, hasta: int, fuera=frozenset()) -> list[int]:
        """Las filas ocultas en `[desde, hasta]` que no están en `fuera`, en orden."""
        a = bisect.bisect_left(self.filas_ocultas, desde)
        b = bisect.bisect_right(self.filas_ocultas, hasta)
        return [r for r in self.filas_ocultas[a:b] if r not in fuera]

    def columna_oculta(self, c: int) -> bool:
        i = bisect.bisect_right(self.columnas_ocultas, (c, float("inf"))) - 1
        return i >= 0 and self.columnas_ocultas[i][0] <= c <= self.columnas_ocultas[i][1]

    def columnas_ocultas_entre(self, desde: int, hasta: int) -> list[int]:
        """Las columnas ocultas en `[desde, hasta]` (cada una UNA vez, aunque la cubran varios
        `<col>` solapados)."""
        salida = []
        for lo, hi in self.columnas_ocultas:
            if hi < desde or lo > hasta:
                continue
            salida.extend(range(max(lo, desde), min(hi, hasta) + 1))
        return salida

    def hallazgos(self, formula_incluida=None, dato_incluido=None, recuadro=None) -> Hallazgos:
        """Cuenta las marcas. Las fórmulas sin valor, en las filas para las que
        `formula_incluida(fila)` Y dentro de `recuadro` = `(fila_desde, fila_hasta, col_desde,
        col_hasta)` (todo el libro si es `None`); errores, porcentajes, ceros y fechas no
        válidas, en las de `dato_incluido(fila)` (todas si es `None`).

        El recuadro es el rectángulo que SE CONVIERTE: una fórmula sin valor fuera de él (una
        fila de total suelta tras una fila vacía, una columna a la derecha) no es de lo
        convertido, y la guarda de fórmulas (DA3) no la mira (3.ª auditoría de C1, I1)."""
        m, h = self.marcas, Hallazgos()
        if recuadro is None:
            def dentro(_f, _c):
                return True
        else:
            f0, f1, c0, c1 = recuadro

            def dentro(f, c):
                return f0 <= f <= f1 and c0 <= c <= c1
        for f, c in zip(m.fsv_f, m.fsv_c):
            if (formula_incluida is None or formula_incluida(f)) and dentro(f, c):
                h.formulas_sin_valor += 1
                if len(h.celdas_sin_valor) < 10:
                    h.celdas_sin_valor.append(referencia(f, c))
        for f, c in zip(m.frm_f, m.frm_c):
            if (formula_incluida is None or formula_incluida(f)) and dentro(f, c):
                h.formulas += 1
                if len(h.celdas_con_formula) < 10:
                    h.celdas_con_formula.append(referencia(f, c))
        ok = dato_incluido or (lambda _f: True)
        for f, c, t in zip(m.err_f, m.err_c, m.err_t):
            if ok(f):
                h.errores_por_columna.setdefault(c, Counter())[m.tipos_error[t]] += 1
        for f, c in zip(m.pct_f, m.pct_c):
            if ok(f):
                h.porcentajes_por_columna[c] += 1
        for f, c, s in zip(m.cer_f, m.cer_c, m.cer_s):
            if ok(f):
                d = h.ceros_por_columna.get(c)
                if d is None:
                    cifras, codigo = self.estilos.ceros.get(s, (0, ""))
                    d = h.ceros_por_columna[c] = {"cuantas": 0, "cifras": cifras, "codigo": codigo,
                                                  "ejemplo": m.cer_v.get(c, "")}
                d["cuantas"] += 1
        for f, c, k in zip(m.fec_f, m.fec_c, m.fec_k):
            if ok(f):
                h.fechas_no_validas_por_columna.setdefault(c, Counter())[_CLASES_FECHA[k]] += 1
                lista = h.celdas_de_fecha_no_valida.setdefault(c, [])
                if len(lista) < 10:
                    lista.append(referencia(f, c))
        return h

    def formulas_sin_valor_fuera(self, recuadro) -> tuple[int, list[str]]:
        """Cuántas fórmulas sin valor caen FUERA de `recuadro` (`fila_desde, fila_hasta,
        col_desde, col_hasta`), y hasta 10 de sus celdas: lo que no se convierte pero tampoco
        debe callarse."""
        f0, f1, c0, c1 = recuadro
        m, n, celdas = self.marcas, 0, []
        for f, c in zip(m.fsv_f, m.fsv_c):
            if not (f0 <= f <= f1 and c0 <= c <= c1):
                n += 1
                if len(celdas) < 10:
                    celdas.append(referencia(f, c))
        return n, celdas

    def dias_cero_1904(self) -> set[tuple[int, int]]:
        """`(fila, columna)` de cada serie 0 ≤ x < 1 con formato de fecha en la época 1904."""
        m = self.marcas
        return set(zip(m.d04_f, m.d04_c))

    def fechas_no_validas(self, incluida=None) -> set[tuple[int, int]]:
        """`(fila, columna)` de cada serie con formato de fecha que no es una fecha."""
        m = self.marcas
        return {(f, c) for f, c in zip(m.fec_f, m.fec_c) if incluida is None or incluida(f)}


_NO_VACIO = ["x"]


def _local_nuevo(locales: dict, nombre: str) -> str:
    loc = locales[nombre] = nombre.rpartition(":")[2]
    return loc


class _Explorador:
    """Los manejadores de `expat` para UNA hoja. Las reglas de posición son las de calamine.

    Los tres manejadores son CIERRES sobre variables locales (`manejadores`): con 3 M de llamadas
    por cada 1,5 M de celdas, leer y escribir atributos de un objeto costaba el doble (medido)."""

    def __init__(self, hoja: str, estilos: Estilos, max_filas: int, max_columnas: int,
                 epoca_1904: bool):
        self.g = Geometria(estilos=estilos)
        self.hoja, self.estilos = hoja, estilos
        self.max_filas, self.max_columnas = max_filas, max_columnas
        self.epoca_1904 = epoca_1904
        self.n_col = 0
        self._ocultas_cols: list[tuple[int, int]] = []

    def manejadores(self, p):
        """`(inicio, fin)` para el analizador `p`. El texto de un `<v>` solo se recoge en las
        celdas que lo necesitan (errores y estilos de fecha, porcentaje o ceros: se activa el
        manejador de texto para ESE `<v>`); en las demás basta saber si el `<v>` está vacío, y eso
        lo dice si `expat` entrega ALGÚN dato de caracteres entre su inicio y su fin (`hay`). NO
        la posición en bytes de la etiqueta de cierre: en una autocerrada (`<v/>`, `<v />`, que
        escribe openpyxl sin lxml) `expat` da en el cierre la posición de DESPUÉS de la etiqueta y
        un `<v/>` parecía tener texto; y en una hoja UTF-16 los bytes no son caracteres (150-C1,
        B-R1). El rectángulo vive en variables locales y se escribe en `Geometria` al cerrar
        (`fijar`)."""
        g, hoja = self.g, self.hoja
        clase_de = self.estilos.de
        necesitan = FECHA | CEROS | PORCENTAJE
        max_filas, max_columnas = self.max_filas, self.max_columnas
        locales: dict[str, str] = {}
        columnas: dict[str, int] = {}
        fila = 0            # la fila del <row> en curso (1-based; 0 = ninguna aún)
        col = 0             # la última columna vista en la fila en curso
        en_celda = en_v = tiene_f = tiene_is = texto_v = False
        vbuf: list[str] | None = None
        t = st = None
        cf = cc = 0
        f0 = f1 = c0 = c1 = None
        n = 0
        marcar = self._marcar

        def datos(texto):
            vbuf.append(texto)

        def hay(_texto):
            # `expat` no llama con cadenas vacías: cualquier llamada es texto dentro del `<v>`.
            nonlocal vbuf
            vbuf = _NO_VACIO

        def inicio(nombre, attrs):
            nonlocal fila, col, en_celda, en_v, tiene_f, tiene_is, texto_v, vbuf, t, st, cf, cc
            loc = locales.get(nombre) or _local_nuevo(locales, nombre)
            if loc == "c":
                en_celda = True
                tiene_f = tiene_is = False
                vbuf = None
                t = attrs.get("t")
                st = attrs.get("s")
                ref = attrs.get("r")
                if ref is None:
                    if not fila:
                        g.exacta = False
                    cf, cc = (fila or 1), col + 1
                else:
                    letras = ref.rstrip("0123456789")
                    cifras = ref[len(letras):]
                    c_ = columnas.get(letras)
                    if c_ is None:
                        c_ = (indice_de_columna(letras)
                              if letras and letras.isascii() and letras.isalpha() else 0)
                        columnas[letras] = c_
                    r_ = int(cifras) if cifras else 0
                    if c_ and r_ > 0:
                        cf, cc = r_, c_
                    else:                 # `$A$1`, `A0`…: calamine no la acepta
                        g.exacta = False
                        cf, cc = (fila or 1), col + 1
                col = cc
            elif loc == "v":
                if en_celda:
                    en_v = True
                    vbuf = []
                    texto_v = t == "e" or (st is not None and bool(clase_de(st) & necesitan))
                    p.CharacterDataHandler = datos if texto_v else hay
            elif loc == "f":
                if en_celda:
                    tiene_f = True
            elif loc == "is":
                if en_celda:
                    tiene_is = True
            elif loc == "row":
                r = attrs.get("r")
                if r is not None and r.isdigit() and int(r) > 0:
                    r = int(r)
                else:
                    if r is not None:
                        g.exacta = False      # calamine no la acepta: lo dirá al leer
                    r = fila + 1
                fila, col = r, 0
                if attrs.get("hidden") in ("1", "true"):
                    g.filas_ocultas.append(r)
            elif loc == "col":
                self._una_columna(attrs)
            elif loc == "dimension":
                m = _RE_DIMENSION.search(attrs.get("ref") or "")
                if m and int(m.group(1)) > _MAX_FILAS_FISICAS:
                    # Un libro de Excel no pasa de 1.048.576 filas: declarar más es una hoja
                    # fabricada, y se rechaza SIN leer las celdas.
                    raise rechazar("demasiadas_filas", hoja=hoja, filas=int(m.group(1)),
                                   tope=max_filas)

        def fin(nombre):
            nonlocal en_celda, en_v, vbuf, f0, f1, c0, c1, n
            loc = locales.get(nombre) or _local_nuevo(locales, nombre)
            if loc == "v":
                if en_v:
                    en_v = False
                    p.CharacterDataHandler = None
                return
            if loc != "c" or not en_celda:
                return
            en_celda = False
            if en_v:                      # un <v> sin cerrar dentro de la celda: XML roto
                en_v = False
                p.CharacterDataHandler = None
            v = None if vbuf is None else ("x" if vbuf is _NO_VACIO else "".join(vbuf))
            # ¿La cuenta calamine? (medido: `cal_formas*.py`)
            if tiene_is or (v is not None and t != "inlineStr" and (v != "" or t in ("str", "d"))):
                n += 1
                if f0 is None:
                    f0 = f1 = cf
                    c0 = c1 = cc
                else:
                    if cf < f0:
                        f0 = cf
                    elif cf > f1:
                        f1 = cf
                    if cc < c0:
                        c0 = cc
                    elif cc > c1:
                        c1 = cc
                if f1 - f0 >= max_filas:
                    raise rechazar("demasiadas_filas", hoja=hoja, filas=f1 - f0 + 1, tope=max_filas)
                if c1 - c0 >= max_columnas:
                    raise rechazar("demasiadas_columnas", hoja=hoja, columnas=c1 - c0 + 1,
                                   tope=max_columnas)
            if tiene_f or st is not None or t == "e":
                marcar(t, st, cf, cc, v, tiene_f, tiene_is)

        def fijar():
            g.min_fila, g.max_fila, g.min_col, g.max_col, g.celdas_con_valor = f0, f1, c0, c1, n

        return inicio, fin, fijar

    def _una_columna(self, attrs: dict) -> None:
        self.n_col += 1
        if self.n_col > MAX_DEFINICIONES_DE_COLUMNA:
            raise rechazar("lector_fallo", tipo="más de 16.384 definiciones de columna", tipo_en="more than 16,384 column definitions")
        if attrs.get("hidden") in ("1", "true"):
            lo, hi = attrs.get("min") or "", attrs.get("max") or ""
            if lo.isdigit() and hi.isdigit() and 0 < int(lo) <= int(hi):
                self._ocultas_cols.append((int(lo), int(hi)))

    def _marcar(self, t, st, f, c, v, tiene_f, tiene_is) -> None:
        """Las marcas de una celda con fórmula, estilo o error (el camino lento)."""
        g = self.g
        m = g.marcas
        tiene_valor = tiene_is or (v is not None and (v != "" or t == "str"))
        if tiene_f:
            g.n_formulas += 1
            m.frm_f.append(f)
            m.frm_c.append(c)
            if not tiene_valor:
                m.fsv_f.append(f)
                m.fsv_c.append(c)
        if t == "e":
            if v:
                m.err_f.append(f)
                m.err_c.append(c)
                m.err_t.append(m.tipo_de_error(v))
            return
        if st is None or not tiene_valor or t not in (None, "n"):
            return
        clase = self.estilos.de(st)
        if not clase:
            return
        if clase & PORCENTAJE:
            m.pct_f.append(f)
            m.pct_c.append(c)
        if clase & (FECHA | CEROS):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return
            if x != x or x in (float("inf"), float("-inf")):
                return
            if clase & FECHA:
                k = _clase_de_fecha(x, self.epoca_1904)
                if k is FECHA_DIA_CERO_1904:
                    m.d04_f.append(f)
                    m.d04_c.append(c)
                elif k is not None:
                    m.fec_f.append(f)
                    m.fec_c.append(c)
                    m.fec_k.append(_CLASES_FECHA.index(k))
            if clase & CEROS:
                cifras = self.estilos.ceros.get(int(st), (0, ""))[0]
                if not x.is_integer() or len(str(int(abs(x)))) < cifras:
                    m.cer_f.append(f)
                    m.cer_c.append(c)
                    m.cer_s.append(int(st))
                    if c not in m.cer_v:
                        m.cer_v[c] = v.strip()

    def cerrar(self) -> Geometria:
        g = self.g
        if g.filas_ocultas:
            g.filas_ocultas = array.array("l", sorted(set(g.filas_ocultas)))
        fundidos: list[list[int]] = []
        for lo, hi in sorted(self._ocultas_cols):
            if fundidos and lo <= fundidos[-1][1] + 1:
                fundidos[-1][1] = max(fundidos[-1][1], hi)
            else:
                fundidos.append([lo, hi])
        g.columnas_ocultas = [tuple(x) for x in fundidos]
        return g


def explorar_hoja(zf: zipfile.ZipFile, ruta: str, estilos: Estilos | None = None,
                  max_filas: int = _MAX_FILAS_FISICAS, max_columnas: int = 16_384, *,
                  hoja: str | None = None, epoca_1904: bool = False) -> Geometria:
    """Una pasada de `expat` por el XML de la hoja, por trozos. Aborta en cuanto el rectángulo
    pasa de `max_filas`/`max_columnas` (no hace falta leer el resto para saber que no cabe)."""
    if estilos is None or isinstance(estilos, (frozenset, set)):
        estilos = Estilos()
    ex = _Explorador(hoja or ruta, estilos, max_filas, max_columnas, epoca_1904)
    p = _expat_seguro()
    p.StartElementHandler, p.EndElementHandler, fijar = ex.manejadores(p)
    lector = _LectorSeguro(zf, ruta)
    try:
        while True:
            trozo = lector.read(_TROZO)
            p.Parse(trozo, not trozo)
            if not trozo:
                break
    except (expat.ExpatError, LookupError, ValueError, UnicodeError):
        # LookupError: una declaración `encoding="…"` que no existe (cazado por el fuzz).
        raise rechazar("lector_fallo", tipo="xml mal formado", tipo_en="malformed XML") from None
    except TablaRechazada:
        raise
    except (RecursionError, MemoryError) as e:
        raise rechazar("lector_fallo", tipo=type(e).__name__) from None
    finally:
        lector.close()
    fijar()
    return ex.cerrar()
