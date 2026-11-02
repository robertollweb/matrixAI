# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que calamine no da, leído del XML del zip con la biblioteca estándar (150-C1, DS2).

Medido en el Paso 0 con `python-calamine 0.9.0`: SÍ da la visibilidad de las hojas
(`sheets_metadata`: Visible/Hidden/VeryHidden) y las celdas combinadas (`merged_cell_ranges`);
NO distingue una celda de fórmula SIN valor guardado de una vacía, devuelve `''` para un error de
celda (`#N/A`) sin decir cuál, no dice qué celdas son porcentajes ni qué filas/columnas están
ocultas, y reserva en Rust el rectángulo ENTERO del rango (30,5 B/celda: una celda suelta en
`Z1048576` cuesta 832 MB). Eso es lo que hace este módulo, **sin `openpyxl`**:

* `explorar_hoja`: una pasada barata por expresiones regulares sobre el XML descomprimido por
  trozos. Da el rectángulo de las celdas CON valor (para la guarda de memoria ANTES de que
  calamine reserve nada), las filas/columnas ocultas y si la hoja tiene fórmulas, errores o
  porcentajes.
* `hallazgos`: una pasada con `xml.etree.iterparse` (más lenta, 4 s por 1,5 M de celdas) que SOLO
  se hace si la anterior vio fórmulas, errores o porcentajes.

El XML llega de un zip que ya pasó la guarda de la bomba; ningún trozo se lee más allá de lo que el
directorio central declara, y un `<!DOCTYPE`/`<!ENTITY` se rechaza (ahí viven las entidades que
se expanden).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass, field

from .errores import rechazar

__all__ = ["mapa_de_hojas", "estilos_de_porcentaje", "explorar_hoja", "hallazgos",
           "Geometria", "Hallazgos", "indice_de_columna", "letras_de_columna", "referencia"]

_TROZO = 1 << 20
_MAX_XML_PEQUENO = 64 << 20


def indice_de_columna(letras: str) -> int:
    """`A` → 1, `Z` → 26, `AA` → 27 (1-based)."""
    n = 0
    for ch in letras:
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
    """Lee un miembro del zip por trozos, sin pasar de lo declarado y rechazando entidades XML."""

    def __init__(self, zf: zipfile.ZipFile, ruta: str):
        try:
            self._info = zf.getinfo(ruta)
            self._f = zf.open(self._info)
        except (KeyError, zipfile.BadZipFile, RuntimeError, OSError):
            raise rechazar("zip_danado") from None
        self._leidos = 0

    def read(self, n: int = -1) -> bytes:
        try:
            b = self._f.read(n if n and n > 0 else _TROZO)
        except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, ValueError):
            raise rechazar("zip_danado") from None
        self._leidos += len(b)
        if self._leidos > self._info.file_size:
            raise rechazar("zip_danado")
        if b"<!DOCTYPE" in b or b"<!ENTITY" in b:
            raise rechazar("lector_fallo", tipo="xml con entidades")
        return b

    def close(self) -> None:
        self._f.close()


def _leer_pequeno(zf: zipfile.ZipFile, ruta: str) -> bytes:
    if zf.getinfo(ruta).file_size > _MAX_XML_PEQUENO:
        raise rechazar("lector_fallo", tipo="parte demasiado grande")
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


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parsear(b: bytes) -> ET.Element:
    try:
        return ET.fromstring(b)
    except ET.ParseError:
        raise rechazar("lector_fallo", tipo="xml mal formado") from None


def mapa_de_hojas(zf: zipfile.ZipFile) -> list[dict]:
    """Las hojas de `xl/workbook.xml` en su orden: `{nombre, estado, ruta}`.

    `estado` es `visible`, `hidden` o `veryHidden`. `ruta` es el miembro del zip con su XML (o
    `None` si el libro apunta a algo que no existe: una hoja de gráfico, un enlace roto).
    """
    wb = _parsear(_leer_pequeno(zf, "xl/workbook.xml"))
    objetivos: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in zf.namelist():
        rels = _parsear(_leer_pequeno(zf, "xl/_rels/workbook.xml.rels"))
        for r in rels:
            rid, destino = r.get("Id"), r.get("Target")
            if rid and destino:
                if destino.startswith("/"):
                    destino = destino.lstrip("/")
                else:
                    destino = "xl/" + destino
                objetivos[rid] = _normalizar_ruta(destino)
    nombres = set(zf.namelist())
    hojas = []
    for el in wb.iter():
        if _local(el.tag) != "sheet":
            continue
        rid = None
        for k, v in el.attrib.items():
            if _local(k) == "id":
                rid = v
        ruta = objetivos.get(rid)
        hojas.append({"nombre": el.get("name") or "", "estado": el.get("state") or "visible",
                      "ruta": ruta if ruta in nombres else None})
    return hojas


def _normalizar_ruta(ruta: str) -> str:
    partes: list[str] = []
    for p in ruta.split("/"):
        if p == "..":
            if partes:
                partes.pop()
        elif p and p != ".":
            partes.append(p)
    return "/".join(partes)


_FORMATOS_PORCENTAJE = {9, 10}


def estilos_de_porcentaje(zf: zipfile.ZipFile) -> frozenset[int]:
    """Los índices de estilo (`s=`) cuyo formato de número es un porcentaje."""
    if "xl/styles.xml" not in zf.namelist():
        return frozenset()
    raiz = _parsear(_leer_pequeno(zf, "xl/styles.xml"))
    personalizados: dict[int, str] = {}
    xfs: list[int] = []
    for el in raiz.iter():
        t = _local(el.tag)
        if t == "numFmt" and el.get("numFmtId", "").isdigit():
            personalizados[int(el.get("numFmtId"))] = el.get("formatCode") or ""
        elif t == "cellXfs":
            for xf in el:
                if _local(xf.tag) == "xf":
                    xfs.append(int(xf.get("numFmtId") or 0) if (xf.get("numFmtId") or "0").isdigit() else 0)
    resultado = set()
    for i, nf in enumerate(xfs):
        if nf in _FORMATOS_PORCENTAJE:
            resultado.add(i)
        elif nf in personalizados and "%" in re.sub(r'"[^"]*"|\\.', "", personalizados[nf]):
            resultado.add(i)
    return frozenset(resultado)


# ── Pasada barata ───────────────────────────────────────────────────────────────────────
# Celdas con contenido (no autocerradas) y celdas vacías con estilo (`<c r="A1" s="3"/>`): el
# valor de ningún atributo de una celda lleva `/`, así que `[^>/]*` distingue una de otra.
_RE_CELDA = re.compile(rb'<c r="([A-Z]{1,3})(\d{1,7})"[^>/]*>')
_RE_CELDA_VACIA = re.compile(rb'<c r="[A-Z]{1,3}\d{1,7}"[^>/]*/>')
_RE_FORMULA = re.compile(rb"<(?:[A-Za-z]+:)?f[ />]")
_RE_FILA_OCULTA = re.compile(rb'<row\b[^>]*?\br="(\d+)"[^>]*?\bhidden="(?:1|true)"')
_RE_FILA_OCULTA_2 = re.compile(rb'<row\b[^>]*?\bhidden="(?:1|true)"[^>]*?\br="(\d+)"')
_RE_COL = re.compile(rb"<col\b[^>]*>")
_RE_ESTILO = re.compile(rb' s="(\d+)"')
_RE_DIMENSION = re.compile(rb'<dimension ref="[A-Z]+\d+:[A-Z]+(\d+)"')
_MAX_FILAS_FISICAS = 1_048_576


@dataclass
class Geometria:
    """El rectángulo de las celdas CON valor de una hoja (1-based) y sus rarezas."""
    min_fila: int | None = None
    max_fila: int | None = None
    min_col: int | None = None
    max_col: int | None = None
    celdas_con_valor: int = 0
    filas_ocultas: list[int] = field(default_factory=list)
    n_filas_ocultas: int = 0
    columnas_ocultas: list[int] = field(default_factory=list)
    hay_formulas: bool = False
    hay_errores: bool = False
    hay_porcentajes: bool = False
    exacta: bool = True      # False si el XML no tenía la forma canónica y hubo que estimar

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


def explorar_hoja(zf: zipfile.ZipFile, ruta: str, estilos_pct: frozenset[int],
                  max_filas: int, max_columnas: int) -> Geometria:
    """Una pasada por trozos sobre el XML de la hoja. Aborta en cuanto el rectángulo pasa de
    `max_filas`/`max_columnas` (no hace falta leer el resto para saber que no cabe)."""
    g = Geometria()
    lector = _LectorSeguro(zf, ruta)
    bufer = b""
    vistos_estilos: set[int] = set()
    primera = True
    try:
        while True:
            nuevo = lector.read(_TROZO)
            if not nuevo and not bufer:
                break
            bufer += nuevo
            corte = bufer.rfind(b">") + 1 if nuevo else len(bufer)
            trozo, bufer = bufer[:corte], bufer[corte:]
            if primera:
                primera = False
                m = _RE_DIMENSION.search(trozo)
                if m and int(m.group(1)) > _MAX_FILAS_FISICAS:
                    # Un libro de Excel no pasa de 1.048.576 filas: declarar más es una hoja
                    # fabricada, y se rechaza SIN leer las celdas.
                    raise rechazar("demasiadas_filas", hoja=ruta, filas=int(m.group(1)),
                                   tope=max_filas)
            _procesar_trozo(trozo, g, estilos_pct, vistos_estilos)
            if not g.vacia and (g.filas > max_filas or g.columnas > max_columnas):
                if g.filas > max_filas:
                    raise rechazar("demasiadas_filas", hoja=ruta, filas=g.filas, tope=max_filas)
                raise rechazar("demasiadas_columnas", hoja=ruta, columnas=g.columnas,
                               tope=max_columnas)
            if not nuevo:
                break
    finally:
        lector.close()
    g.hay_porcentajes = bool(vistos_estilos & estilos_pct) if estilos_pct else False
    return g


def _procesar_trozo(trozo: bytes, g: Geometria, estilos_pct, vistos_estilos: set) -> None:
    pares = _RE_CELDA.findall(trozo)
    esperadas = trozo.count(b"<c ") + trozo.count(b"<c>")
    if len(pares) + len(_RE_CELDA_VACIA.findall(trozo)) != esperadas:
        g.exacta = False   # prefijos de espacio de nombres, atributos en otro orden…
    if pares:
        columnas_vistas, filas_vistas = zip(*pares)           # a nivel de C, no celda a celda
        filas = [int(r) for r in set(filas_vistas)]
        columnas = {indice_de_columna(c.decode("ascii")) for c in set(columnas_vistas)}
        g.min_fila = min(filas) if g.min_fila is None else min(g.min_fila, min(filas))
        g.max_fila = max(filas) if g.max_fila is None else max(g.max_fila, max(filas))
        g.min_col = min(columnas) if g.min_col is None else min(g.min_col, min(columnas))
        g.max_col = max(columnas) if g.max_col is None else max(g.max_col, max(columnas))
        g.celdas_con_valor += len(pares)
    if not g.hay_formulas and _RE_FORMULA.search(trozo):
        g.hay_formulas = True
    if not g.hay_errores and b' t="e"' in trozo:
        g.hay_errores = True
    if estilos_pct:
        vistos_estilos.update(int(s) for s in set(_RE_ESTILO.findall(trozo)))
    for m in _RE_FILA_OCULTA.finditer(trozo):
        g.n_filas_ocultas += 1
        if len(g.filas_ocultas) < 1000:
            g.filas_ocultas.append(int(m.group(1)))
    for m in _RE_FILA_OCULTA_2.finditer(trozo):
        g.n_filas_ocultas += 1
        if len(g.filas_ocultas) < 1000:
            g.filas_ocultas.append(int(m.group(1)))
    for m in _RE_COL.finditer(trozo):
        etiqueta = m.group(0)
        if re.search(rb'\bhidden="(?:1|true)"', etiqueta):
            lo = re.search(rb'\bmin="(\d+)"', etiqueta)
            hi = re.search(rb'\bmax="(\d+)"', etiqueta)
            if lo and hi:
                for c in range(int(lo.group(1)), min(int(hi.group(1)), int(lo.group(1)) + 16384) + 1):
                    g.columnas_ocultas.append(c)


# ── Pasada lenta ────────────────────────────────────────────────────────────────────────
@dataclass
class Hallazgos:
    formulas_sin_valor: int = 0
    celdas_sin_valor: list[str] = field(default_factory=list)       # hasta 10, `D7`
    filas_con_formula_sin_valor: list[int] = field(default_factory=list)  # hasta 1000
    errores_por_columna: dict[int, Counter] = field(default_factory=dict)  # col 1-based → texto→n
    porcentajes_por_columna: Counter = field(default_factory=Counter)      # col 1-based → n


def hallazgos(zf: zipfile.ZipFile, ruta: str, estilos_pct: frozenset[int], incluida=None) -> Hallazgos:
    """Cuenta, en las filas para las que `incluida(fila)` es verdadera (todas si es `None`):
    celdas con fórmula pero SIN valor guardado, errores de celda por columna y celdas con formato
    de porcentaje por columna."""
    h = Hallazgos()
    lector = _LectorSeguro(zf, ruta)
    try:
        for _evento, el in ET.iterparse(lector, events=("end",)):
            nombre = _local(el.tag)
            if nombre == "c":
                _una_celda(el, h, estilos_pct, incluida)
            elif nombre == "row":
                el.clear()
    except ET.ParseError:
        raise rechazar("lector_fallo", tipo="xml mal formado") from None
    finally:
        lector.close()
    return h


_RE_REF = re.compile(r"^([A-Z]{1,3})(\d+)$")


def _una_celda(el: ET.Element, h: Hallazgos, estilos_pct, incluida) -> None:
    m = _RE_REF.match(el.get("r") or "")
    if not m:
        return
    fila, col = int(m.group(2)), indice_de_columna(m.group(1))
    if incluida is not None and not incluida(fila):
        return
    tipo = el.get("t")
    f = v = None
    es_texto = False
    for hijo in el:
        n = _local(hijo.tag)
        if n == "f":
            f = hijo
        elif n == "v":
            v = hijo
        elif n == "is":
            es_texto = True
    tiene_valor = es_texto or (v is not None and ((v.text or "") != "" or tipo == "str"))
    if f is not None and not tiene_valor:
        h.formulas_sin_valor += 1
        if len(h.celdas_sin_valor) < 10:
            h.celdas_sin_valor.append(f"{m.group(1)}{fila}")
        if len(h.filas_con_formula_sin_valor) < 1000 and (
                not h.filas_con_formula_sin_valor or h.filas_con_formula_sin_valor[-1] != fila):
            h.filas_con_formula_sin_valor.append(fila)
    if tipo == "e" and v is not None:
        h.errores_por_columna.setdefault(col, Counter())[(v.text or "#ERROR").strip()] += 1
    if estilos_pct and tiene_valor and (el.get("s") or "").isdigit() and int(el.get("s")) in estilos_pct:
        h.porcentajes_por_columna[col] += 1
