# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""`inspeccionar` y `convertir`: de un libro de Excel al CSV canónico (150-C1).

**Propone; no decide.** `inspeccionar` enseña las hojas, la cabecera que cree ver, las filas que
parecen un total y los avisos; `convertir` hace SOLO lo que la persona confirmó (`hoja`,
`fila_de_cabecera`, `filas_a_excluir`…) y declara todo lo demás que hizo.

Reglas que no se negocian (invariantes del contrato 150):

* los topes se comprueban sobre el ORIGINAL antes de abrirlo (`formato`, `topes`);
* una fórmula sin valor guardado RECHAZA, y nunca se calcula una fórmula;
* `convertir` jamás excluye una fila por su cuenta;
* un valor ausente no es un cero;
* mismo libro y mismos parámetros → los mismos bytes.
"""
from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass, field

from .declaraciones import declarar
from .errores import TablaRechazada, rechazar
from .formato import abrir_zip_de_excel, detectar_formato, es_ole_de_excel
from .topes import (TOPES_DE_FABRICA, Topes, comprobar_memoria, ram_estimada_de_una_hoja)
from .valores import escribir_csv, tipo_de_valor, valor_a_texto
from . import xlsx_xml
from .xlsx_xml import letras_de_columna, referencia

__all__ = ["inspeccionar", "convertir", "TablaConvertida", "ESQUEMA_DE_ORIGEN"]

ESQUEMA_DE_ORIGEN = 1
_MAX_FILAS_FISICAS = 1_048_576
#: Peor caso de un `.xls` (BIFF8): 65.536 filas × 256 columnas, a lo que reserva el lector.
_CELDAS_MAXIMAS_DE_UN_XLS = 65_536 * 256
_PREVISTA_PRIMERAS = 8
_PREVISTA_ULTIMAS = 5
_MAX_COLUMNAS_PREVISTA = 60
_ventana_cabecera = 60


@dataclass
class TablaConvertida:
    """El CSV canónico y su `origen`. `csv_bytes` es lo que se hashea; no se reescribe."""
    csv_bytes: bytes
    origen: dict

    @property
    def csv_text(self) -> str:
        return self.csv_bytes.decode("utf-8")


# ── El libro ────────────────────────────────────────────────────────────────────────────
@dataclass
class _HojaLeida:
    nombre: str
    fila0: int = 1                 # fila (1-based) de la primera fila de `filas`
    col0: int = 1
    filas: list = field(default_factory=list)
    combinadas: list = field(default_factory=list)   # (r0, c0, r1, c1) 1-based inclusivo
    geom: object = None

    @property
    def vacia(self) -> bool:
        return not self.filas

    @property
    def ultima_fila(self) -> int:
        return self.fila0 + len(self.filas) - 1

    @property
    def ancho(self) -> int:
        return max((len(f) for f in self.filas), default=0)


def _importar_calamine():
    try:
        import python_calamine
    except ImportError:
        raise rechazar("falta_lector") from None
    return python_calamine


def _version_de_calamine() -> str | None:
    try:
        from importlib import metadata
        return metadata.version("python-calamine")
    except Exception:
        try:
            import python_calamine
            return getattr(python_calamine, "__version__", None)
        except Exception:
            return None


class _Libro:
    def __init__(self, datos: bytes, nombre: str, topes: Topes):
        self.datos = bytes(datos)
        self.nombre = _nombre_corto(nombre)
        self.topes = topes
        self.zf = None
        self.macros = False
        self._wb = None
        self._estilos_pct = None
        self.formato = self._clasificar()
        self.hojas = self._listar_hojas()

    # -- clasificación por FIRMA ----------------------------------------------------------
    def _clasificar(self) -> str:
        familia = detectar_formato(self.datos[:4096])
        extension = ("." + self.nombre.rsplit(".", 1)[-1].lower()) if "." in self.nombre else "(sin extensión)"
        if familia == "vacio":
            raise rechazar("vacio")
        if familia == "html":
            raise rechazar("html_no_excel")
        if familia == "texto":
            raise rechazar("texto_no_excel", extension=extension)
        if familia == "xml_excel":
            raise rechazar("formato_no_soportado", formato="XML de Excel 2003")
        if familia == "parquet":
            raise rechazar("parquet_no_soportado")
        if familia == "desconocido":
            raise rechazar("formato_desconocido")
        if familia == "ole":
            if any(f in self.datos for f in (b"E\x00n\x00c\x00r\x00y\x00p\x00t\x00e\x00d\x00P\x00a\x00c\x00k\x00a\x00g\x00e\x00",)):
                raise rechazar("protegido")
            if not es_ole_de_excel(self.datos):
                raise rechazar("no_es_excel")
            return "xls"
        self.zf, info = abrir_zip_de_excel(self.datos, self.topes)
        self.macros = info["macros"]
        return "xlsm" if self.macros else "xlsx"

    # -- hojas ------------------------------------------------------------------------------
    def _listar_hojas(self) -> list[dict]:
        if self.zf is not None:
            hojas = xlsx_xml.mapa_de_hojas(self.zf)
        else:
            wb = self.calamine()
            hojas = []
            for m in wb.sheets_metadata:
                estado = {"Visible": "visible", "Hidden": "hidden", "VeryHidden": "veryHidden"}.get(
                    str(m.visible).rsplit(".", 1)[-1], "visible")
                hojas.append({"nombre": m.name, "estado": estado, "ruta": None})
        return hojas

    def calamine(self):
        if self._wb is None:
            cal = _importar_calamine()
            try:
                self._wb = cal.load_workbook(io.BytesIO(self.datos))
            except cal.PasswordError:
                raise rechazar("protegido") from None
            except TablaRechazada:
                raise
            except BaseException as e:   # ninguna excepción de la biblioteca llega a la persona
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                raise rechazar("lector_fallo", tipo=type(e).__name__) from None
        return self._wb

    def estilos_pct(self) -> frozenset:
        if self._estilos_pct is None:
            self._estilos_pct = (xlsx_xml.estilos_de_porcentaje(self.zf)
                                 if self.zf is not None else frozenset())
        return self._estilos_pct

    def meta(self, nombre: str) -> dict:
        for h in self.hojas:
            if h["nombre"] == nombre:
                return h
        raise rechazar("hoja_no_encontrada", hoja=nombre, hojas=[h["nombre"] for h in self.hojas])

    def elegir(self, hoja) -> dict:
        if not self.hojas:
            raise rechazar("sin_hojas")
        if hoja is None:
            visibles = [h for h in self.hojas if h["estado"] == "visible"]
            if len(visibles) == 1:
                return visibles[0]
            raise rechazar("hoja_requerida", hojas=[h["nombre"] for h in (visibles or self.hojas)])
        if isinstance(hoja, int) and not isinstance(hoja, bool):
            if 0 <= hoja < len(self.hojas):
                return self.hojas[hoja]
            raise rechazar("hoja_no_encontrada", hoja=hoja, hojas=[h["nombre"] for h in self.hojas])
        return self.meta(str(hoja))

    def geometria(self, meta: dict):
        if self.zf is None or meta["ruta"] is None:
            return None
        return xlsx_xml.explorar_hoja(self.zf, meta["ruta"], self.estilos_pct(),
                                      self.topes.max_filas, self.topes.max_columnas)

    # -- leer una hoja con las guardas ANTES de que el lector reserve nada ------------------
    def leer(self, meta: dict, geom=None) -> _HojaLeida:
        nombre = meta["nombre"]
        if self.zf is not None:
            if geom is None:
                geom = self.geometria(meta)
            if geom is None or geom.vacia:
                return _HojaLeida(nombre, geom=geom)
            tam = self.zf.getinfo(meta["ruta"]).file_size
            celdas = geom.celdas_del_rango
            if not geom.exacta:   # el XML no tenía la forma canónica: el rectángulo es una cota
                celdas = max(celdas, tam // 20)
            comprobar_memoria(ram_estimada_de_una_hoja(celdas, tam), nombre)
        else:
            comprobar_memoria(ram_estimada_de_una_hoja(_CELDAS_MAXIMAS_DE_UN_XLS), nombre)
        wb = self.calamine()
        cal = _importar_calamine()
        try:
            hoja = wb.get_sheet_by_name(nombre)
            inicio = hoja.start
            if inicio is None:
                return _HojaLeida(nombre, geom=geom)
            filas = hoja.to_python(skip_empty_area=True)
            combinadas = hoja.merged_cell_ranges or []
        except cal.PasswordError:
            raise rechazar("protegido") from None
        except TablaRechazada:
            raise
        except MemoryError:
            raise rechazar("lector_fallo", tipo="MemoryError") from None
        except Exception as e:
            raise rechazar("lector_fallo", tipo=type(e).__name__) from None
        h = _HojaLeida(nombre, inicio[0] + 1, inicio[1] + 1, filas,
                       [(a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1) for a, b in combinadas], geom)
        if len(filas) > self.topes.max_filas:
            raise rechazar("demasiadas_filas", hoja=nombre, filas=len(filas), tope=self.topes.max_filas)
        if h.ancho > self.topes.max_columnas:
            raise rechazar("demasiadas_columnas", hoja=nombre, columnas=h.ancho,
                           tope=self.topes.max_columnas)
        return h

    def hallazgos(self, meta: dict, incluida=None):
        if self.zf is None or meta["ruta"] is None:
            return None
        return xlsx_xml.hallazgos(self.zf, meta["ruta"], self.estilos_pct(), incluida)

    def conversor(self) -> dict:
        import matrixai
        return {"nombre": "matrixai.tablas",
                "versiones": {"matrixai-core": getattr(matrixai, "__version__", None),
                              "python-calamine": _version_de_calamine()}}


def _nombre_corto(nombre) -> str:
    n = str(nombre or "").replace("\\", "/").rsplit("/", 1)[-1]
    return n[:255]


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _texto(v) -> str:
    """El texto de una celda para un nombre o una vista previa: sin espacios sobrantes."""
    t = valor_a_texto(v)[0]
    return " ".join(t.split())


def _vacia(v) -> bool:
    return v is None or v == "" or (isinstance(v, str) and not v.strip())


# ── Lo que se propone ───────────────────────────────────────────────────────────────────
_RE_TOTAL = re.compile(
    r"^(totales|total|subtotales|subtotal|suma|promedio|media|resumen|grand total|"
    r"totals|sum|average|mean|summary)\b")


def _primera_celda_de_texto(fila: list) -> str | None:
    for v in fila:
        if _vacia(v):
            continue
        return v if isinstance(v, str) else None
    return None


def _proponer_cabecera(h: _HojaLeida) -> int | None:
    """Primera fila con ≥2 textos distintos, bastante llena y seguida de filas con datos."""
    ventana = h.filas[:_ventana_cabecera]
    llenas = [sum(1 for v in f if not _vacia(v)) for f in ventana]
    if not llenas:
        return None
    referencia_ = max(llenas)
    for i, fila in enumerate(ventana):
        textos = {v.strip() for v in fila if isinstance(v, str) and v.strip()}
        if len(textos) < 2 or llenas[i] < max(2, 0.5 * referencia_):
            continue
        if i + 1 < len(h.filas) and sum(1 for v in h.filas[i + 1] if not _vacia(v)) >= 2:
            return h.fila0 + i
    return None


def _candidatas_a_total(h: _HojaLeida, desde: int) -> list[dict]:
    """Filas que PARECEN un total: la palabra, o ser de las últimas con la mitad de celdas vacías.
    Propone; la persona confirma."""
    candidatas: dict[int, str] = {}
    inicio = max(desde - h.fila0, 0)
    for i in range(inicio, len(h.filas)):
        t = _primera_celda_de_texto(h.filas[i])
        if t and _RE_TOTAL.match(_sin_tildes(t).strip().lower()):
            candidatas[h.fila0 + i] = "palabra"
    region = [(i, f) for i, f in enumerate(h.filas) if i >= inicio]
    if region:
        ancho = max(sum(1 for v in f if not _vacia(v)) for _, f in region)
        for i, f in region[-_PREVISTA_ULTIMAS:]:
            llenas = sum(1 for v in f if not _vacia(v))
            if 0 < llenas <= ancho / 2 and h.fila0 + i not in candidatas:
                candidatas[h.fila0 + i] = "mitad_vacia"
    resultado = []
    for fila in sorted(candidatas):
        celdas = h.filas[fila - h.fila0]
        primera = next((_texto(v) for v in celdas if not _vacia(v)), "")
        resultado.append({"fila": fila, "texto": primera, "motivo": candidatas[fila]})
    return resultado


def _vista(h: _HojaLeida, i: int) -> dict:
    celdas = [valor_a_texto(v)[0][:200] for v in h.filas[i][:_MAX_COLUMNAS_PREVISTA]]
    return {"fila": h.fila0 + i, "celdas": celdas}


def _nombres_propuestos(h: _HojaLeida, fila_cab: int) -> list[str]:
    i = fila_cab - h.fila0
    return [_texto(v) for v in h.filas[i]]


# ── inspeccionar ────────────────────────────────────────────────────────────────────────
def inspeccionar(datos: bytes, nombre: str = "", *, topes: Topes | None = None) -> dict:
    """Qué hay en el libro, sin convertir nada. Propone la hoja, la cabecera y los totales."""
    libro = _Libro(datos, nombre, topes or TOPES_DE_FABRICA)
    visibles = [h for h in libro.hojas if h["estado"] == "visible"]
    hojas = [_inspeccionar_hoja(libro, m) for m in libro.hojas]
    avisos_globales = []
    if libro.macros:
        avisos_globales.append(declarar("macros_ignoradas"))
    if libro.formato == "xls":
        avisos_globales.append(declarar("no_comprobable_en_xls",
                                        que=["fórmulas sin valor guardado", "errores de celda",
                                             "formato de porcentaje", "filas y columnas ocultas"]))
    return {
        "formato": libro.formato,
        "nombre": libro.nombre,
        "bytes": len(libro.datos),
        "sha256": hashlib.sha256(libro.datos).hexdigest(),
        "hojas": hojas,
        "hoja_propuesta": visibles[0]["nombre"] if len(visibles) == 1 else None,
        "avisos": avisos_globales,
        "conversor": libro.conversor(),
    }


def _inspeccionar_hoja(libro: _Libro, meta: dict) -> dict:
    estado = {"visible": "visible", "hidden": "oculta", "veryHidden": "muy_oculta"}.get(
        meta["estado"], "visible")
    entrada = {"nombre": meta["nombre"], "estado": estado, "visible": estado == "visible",
               "filas": None, "columnas": None, "rango": None, "primeras": [], "ultimas": [],
               "fila_de_cabecera_propuesta": None, "filas_candidatas_a_total": [], "avisos": []}
    if estado != "visible":
        entrada["avisos"].append(declarar("hoja_oculta", hoja=meta["nombre"]))
    try:
        geom = libro.geometria(meta)
        if estado != "visible":
            # Una hoja oculta se LISTA con sus medidas, pero no se carga: no se ofrece.
            if geom is not None and not geom.vacia:
                entrada.update(filas=geom.filas, columnas=geom.columnas,
                               rango=_rango_a1(geom.min_fila, geom.min_col, geom.max_fila, geom.max_col))
            return entrada
        h = libro.leer(meta, geom)
    except TablaRechazada as e:
        if e.codigo in ("memoria_insuficiente", "demasiadas_filas", "demasiadas_columnas",
                        "lector_fallo"):
            entrada["error"] = e.como_dict()
            return entrada
        raise
    if h.vacia:
        entrada["filas"] = entrada["columnas"] = 0
        entrada["avisos"].append(declarar("hoja_sin_datos"))
        return entrada
    n = len(h.filas)
    entrada.update(filas=n, columnas=h.ancho,
                   rango=_rango_a1(h.fila0, h.col0, h.ultima_fila, h.col0 + h.ancho - 1))
    entrada["primeras"] = [_vista(h, i) for i in range(min(_PREVISTA_PRIMERAS, n))]
    entrada["ultimas"] = [_vista(h, i) for i in range(max(_PREVISTA_PRIMERAS, n - _PREVISTA_ULTIMAS), n)]
    cab = _proponer_cabecera(h)
    entrada["fila_de_cabecera_propuesta"] = cab
    entrada["filas_candidatas_a_total"] = _candidatas_a_total(h, (cab + 1) if cab else h.fila0)
    avisos = entrada["avisos"]
    if h.combinadas:
        avisos.append(declarar("hay_celdas_combinadas", cuantas=len(h.combinadas),
                               rangos=[_rango_a1(*m[:2], *m[2:]) for m in h.combinadas[:10]]))
    geom = h.geom
    if geom is not None and geom.n_filas_ocultas:
        avisos.append(declarar("filas_ocultas", cuantas=geom.n_filas_ocultas,
                               filas=geom.filas_ocultas[:10]))
    if geom is not None and geom.columnas_ocultas:
        avisos.append(declarar("columnas_ocultas", cuantas=len(geom.columnas_ocultas),
                               letras=[letras_de_columna(c) for c in geom.columnas_ocultas[:10]]))
    if geom is not None and (geom.hay_formulas or geom.hay_errores or geom.hay_porcentajes):
        hz = libro.hallazgos(meta)
        if hz.formulas_sin_valor:
            avisos.append(declarar("formulas_sin_valor", cuantas=hz.formulas_sin_valor,
                                   celdas=hz.celdas_sin_valor))
        for col in sorted(hz.errores_por_columna):
            tipos = hz.errores_por_columna[col]
            avisos.append(declarar("errores_de_celda", columna=letras_de_columna(col),
                                   cuantas=sum(tipos.values()), tipos=sorted(tipos)))
        for col in sorted(hz.porcentajes_por_columna):
            avisos.append(declarar("formato_porcentaje", columna=letras_de_columna(col),
                                   cuantas=hz.porcentajes_por_columna[col]))
    if cab:
        nombres = _nombres_propuestos(h, cab)
        vistos, problemas = set(), []
        for j, t in enumerate(nombres):
            if not t:
                if any(not _vacia(f[j]) for f in h.filas[cab - h.fila0 + 1:] if j < len(f)):
                    problemas.append(f"({letras_de_columna(h.col0 + j)})")
            elif t in vistos:
                problemas.append(t)
            vistos.add(t)
        if problemas:
            avisos.append(declarar("nombres_repetidos", cuantas=len(problemas), nombres=problemas))
        i = cab - h.fila0
        if i + 1 < len(h.filas):
            sig = h.filas[i + 1]
            no_vacias = [v for v in sig if not _vacia(v)]
            if len({v for v in no_vacias if isinstance(v, str)}) >= 2 and all(
                    isinstance(v, str) for v in no_vacias) and any(
                    c for c in h.combinadas if c[0] == cab and c[3] > c[1]):
                avisos.append(declarar("posible_cabecera_de_dos_filas", fila=cab))
        _avisos_de_columnas(h, cab + 1, avisos)
    return entrada


def _avisos_de_columnas(h: _HojaLeida, desde: int, avisos: list) -> None:
    ancho = h.ancho
    for j in range(ancho):
        tipos: dict[str, int] = {}
        valores_numero = []
        for fila in h.filas[max(desde - h.fila0, 0):]:
            if j >= len(fila) or _vacia(fila[j]):
                continue
            t = tipo_de_valor(fila[j])
            tipos[t] = tipos.get(t, 0) + 1
            if t == "numero":
                valores_numero.append(fila[j])
        letra = letras_de_columna(h.col0 + j)
        if len(tipos) > 1:
            avisos.append(declarar("columna_mixta", columna=letra, cuantas=sum(tipos.values()),
                                   tipos=dict(sorted(tipos.items()))))
        elif set(tipos) == {"numero"} and valores_numero and all(
                20_000 <= v <= 60_000 for v in valores_numero):
            avisos.append(declarar("posible_fecha_en_serie", columna=letra,
                                   cuantas=len(valores_numero)))


def _rango_a1(f0: int, c0: int, f1: int, c1: int) -> str:
    return f"{referencia(f0, c0)}:{referencia(f1, c1)}"


# ── convertir ───────────────────────────────────────────────────────────────────────────
def convertir(datos: bytes, nombre: str = "", *, hoja=None, fila_de_cabecera: int,
              filas_de_cabecera: int = 1, filas_a_excluir=(), rellenar_combinadas: bool = False,
              topes: Topes | None = None) -> TablaConvertida:
    """El CSV canónico de una hoja, con lo que la persona confirmó y el `origen` que lo declara.

    `fila_de_cabecera` es el número de fila de la hoja (el que enseña Excel). `filas_a_excluir`
    son números de fila de la hoja, y SOLO se quitan esas.
    """
    if isinstance(fila_de_cabecera, bool) or not isinstance(fila_de_cabecera, int) or fila_de_cabecera < 1:
        raise rechazar("parametro_invalido", parametro="fila_de_cabecera debe ser un entero ≥ 1")
    if filas_de_cabecera not in (1, 2) or isinstance(filas_de_cabecera, bool):
        raise rechazar("parametro_invalido", parametro="filas_de_cabecera debe ser 1 o 2")
    try:
        excluir = sorted({int(f) for f in filas_a_excluir if not isinstance(f, bool)})
    except (TypeError, ValueError):
        raise rechazar("parametro_invalido", parametro="filas_a_excluir debe ser una lista de números de fila") from None
    if len(excluir) != len(set(filas_a_excluir)) or any(f < 1 or f > _MAX_FILAS_FISICAS for f in excluir):
        raise rechazar("parametro_invalido", parametro="filas_a_excluir debe ser una lista de números de fila")

    libro = _Libro(datos, nombre, topes or TOPES_DE_FABRICA)
    meta = libro.elegir(hoja)
    h = libro.leer(meta)
    if h.vacia:
        raise rechazar("hoja_vacia", hoja=meta["nombre"])

    fh, nh = fila_de_cabecera, filas_de_cabecera
    fin = h.ultima_fila
    if fh < h.fila0 or fh + nh - 1 > fin:
        raise rechazar("cabecera_invalida", fila=fh, hoja=meta["nombre"], primera=h.fila0, ultima=fin)
    fd = fh + nh
    if fd > fin:
        raise rechazar("sin_filas_de_datos", fila=fh)
    if any(f < fd for f in excluir):
        raise rechazar("parametro_invalido",
                       parametro=f"filas_a_excluir {[f for f in excluir if f < fd]} está en la cabecera o por encima")
    excluidas = set(excluir)

    # Fórmulas sin valor y errores: se comprueban sobre lo que SE CONVIERTE (la fila TOTAL que la
    # persona quita no rechaza). Nunca se calcula una fórmula.
    hz = None
    geom = h.geom
    if geom is not None and (geom.hay_formulas or geom.hay_errores or geom.hay_porcentajes):
        hz = libro.hallazgos(meta, lambda r: r >= fh and r not in excluidas)
        if hz.formulas_sin_valor:
            raise rechazar("formula_sin_valor", cuantas=hz.formulas_sin_valor,
                           celdas=hz.celdas_sin_valor)

    ancho = h.ancho
    col0 = h.col0
    # ── combinadas dentro de lo que se convierte ──
    # Las celdas cubiertas llegan VACÍAS del lector. Se cuentan por columna (y por rango) y se
    # declaran; solo si la persona lo marcó (`rellenar_combinadas`) se rellenan con la celda
    # combinada, y entonces se declara ESO.
    vacias_por_columna: dict[int, int] = {}
    rellenadas_por_columna: dict[int, int] = {}
    rangos_vacias: dict[int, list[str]] = {}
    rangos_rellenadas: dict[int, list[str]] = {}
    for (r0, c0, r1, c1) in h.combinadas:
        filas_dentro = [r for r in range(max(r0, fd), min(r1, fin) + 1) if r not in excluidas]
        if not filas_dentro:
            continue
        valor_origen = ""
        if rellenar_combinadas and 0 <= r0 - h.fila0 < len(h.filas):
            fila_origen = h.filas[r0 - h.fila0]
            if 0 <= c0 - col0 < len(fila_origen):
                valor_origen = fila_origen[c0 - col0]
        rango = _rango_a1(r0, c0, r1, c1)
        for c in range(c0, c1 + 1):
            cubiertas = [r for r in filas_dentro if not (r == r0 and c == c0)]
            if not cubiertas:
                continue
            if rellenar_combinadas and not _vacia(valor_origen) and 0 <= c - col0 < ancho:
                for r in cubiertas:
                    h.filas[r - h.fila0][c - col0] = valor_origen
                rellenadas_por_columna[c] = rellenadas_por_columna.get(c, 0) + len(cubiertas)
                rangos_rellenadas.setdefault(c, []).append(rango)
            else:
                vacias_por_columna[c] = vacias_por_columna.get(c, 0) + len(cubiertas)
                rangos_vacias.setdefault(c, []).append(rango)

    # ── cabecera ──
    def celda(r: int, j: int):
        fila = h.filas[r - h.fila0]
        return fila[j] if j < len(fila) else ""
    nombres_crudos: list[str] = []
    compuestos: dict[int, tuple[str, str]] = {}
    grupo_actual = None
    for j in range(ancho):
        if nh == 1:
            nombres_crudos.append(_texto(celda(fh, j)))
            continue
        g, s = _texto(celda(fh, j)), _texto(celda(fh + 1, j))
        if g and s:
            grupo_actual, nombre_col = g, f"{g} {s}"
            compuestos[j] = (g, s)
        elif g:
            grupo_actual, nombre_col = None, g
        elif s:
            nombre_col = f"{grupo_actual} {s}" if grupo_actual else s
            if grupo_actual:
                compuestos[j] = (grupo_actual, s)
        else:
            grupo_actual, nombre_col = None, ""
        nombres_crudos.append(nombre_col)

    # ── datos ──
    filas_csv: list[list[str]] = []
    usada = [False] * ancho
    filas_vacias: list[int] = []
    sucesos: dict[tuple[int, str], int] = {}
    tipos_col: list[dict[str, int]] = [dict() for _ in range(ancho)]
    solo_fecha: dict[int, list[int]] = {}      # col → índices de filas_csv con fecha SIN hora
    con_hora: dict[int, int] = {}              # col → cuántas fechas llevan hora
    series_ok = [True] * ancho
    for r in range(fd, fin + 1):
        if r in excluidas:
            continue
        fuente = h.filas[r - h.fila0]
        salida = [""] * ancho
        alguna = False
        for j in range(min(ancho, len(fuente))):
            v = fuente[j]
            if v == "" or v is None:
                continue
            t = type(v)
            if t is str:
                texto, suceso = v, None
                tipo = "texto"
                series_ok[j] = False
            elif t is float:
                if v - v == 0.0:   # finito (NaN e infinito dan NaN)
                    texto = str(int(v)) if (v.is_integer() and -1e15 < v < 1e15) else repr(v)
                    suceso = None
                else:
                    texto, suceso = "", "no_finito"
                tipo = "numero"
                if not (20_000 <= v <= 60_000):
                    series_ok[j] = False
            else:
                texto, suceso = valor_a_texto(v)
                tipo = tipo_de_valor(v)
                if tipo != "numero" or not (20_000 <= v <= 60_000):
                    series_ok[j] = False
            if suceso:
                sucesos[(j, suceso)] = sucesos.get((j, suceso), 0) + 1
            if texto == "":
                continue
            if tipo in ("hora", "duracion"):
                sucesos[(j, tipo)] = sucesos.get((j, tipo), 0) + 1
            tipos_col[j][tipo] = tipos_col[j].get(tipo, 0) + 1
            if tipo == "fecha":
                if len(texto) == 10:
                    solo_fecha.setdefault(j, []).append(len(filas_csv))
                else:
                    con_hora[j] = con_hora.get(j, 0) + 1
            salida[j] = texto
            usada[j] = True
            alguna = True
        if alguna:
            filas_csv.append(salida)
        else:
            filas_vacias.append(r)
    if not filas_csv:
        raise rechazar("sin_filas_de_datos", fila=fh)

    # El analizador exige UN formato de fecha por columna: si una mezcla fechas sin hora y con
    # hora, las primeras se escriben a medianoche (que es lo que son) y se declara.
    fechas_a_medianoche: dict[int, int] = {}
    for j, indices in solo_fecha.items():
        if con_hora.get(j):
            for i in indices:
                filas_csv[i][j] += " 00:00:00"
            fechas_a_medianoche[j] = len(indices)

    # ── columnas que se quedan ──
    conserva = [j for j in range(ancho) if usada[j] or nombres_crudos[j]]
    quitadas = [j for j in range(ancho) if j not in set(conserva)]
    if not conserva:
        raise rechazar("hoja_vacia", hoja=meta["nombre"])
    declaraciones: list[dict] = []
    if quitadas:
        declaraciones.append(declarar("columnas_vacias_quitadas", cuantas=len(quitadas),
                                      letras=[letras_de_columna(col0 + j) for j in quitadas]))

    # ── nombres: vacíos con datos → columna_N; repetidos → X, X_2 ──
    finales: dict[int, str] = {}
    usados: set[str] = set()
    pendientes_repetidos = []
    for j in conserva:
        t = nombres_crudos[j]
        letra = letras_de_columna(col0 + j)
        if not t:
            n = f"columna_{col0 + j}"
            finales[j] = n
            declaraciones.append(declarar("nombre_vacio_con_datos", columna=n, letra=letra))
        else:
            finales[j] = t
    # los nombres «reales» ocupan su sitio primero; luego los renombrados buscan hueco
    for j in conserva:
        n = finales[j]
        if n in usados:
            pendientes_repetidos.append(j)
        else:
            usados.add(n)
    for j in pendientes_repetidos:
        base = finales[j]
        k = 2
        while f"{base}_{k}" in usados:
            k += 1
        nuevo = f"{base}_{k}"
        usados.add(nuevo)
        finales[j] = nuevo
        declaraciones.append(declarar("nombre_repetido_renombrado", columna=nuevo,
                                      original=base, letra=letras_de_columna(col0 + j)))
    for j in conserva:
        crudo_texto = valor_a_texto(celda(fh, j))[0] if nh == 1 else None
        if nh == 1 and crudo_texto and crudo_texto != nombres_crudos[j]:
            declaraciones.append(declarar("nombre_normalizado", columna=finales[j], original=crudo_texto))
        if nh == 2 and j in compuestos:
            declaraciones.append(declarar("cabecera_compuesta", columna=finales[j],
                                          grupo=compuestos[j][0], sub=compuestos[j][1]))

    # ── lo demás que se hizo ──
    for j in conserva:
        c_abs = col0 + j
        if rellenadas_por_columna.get(c_abs):
            declaraciones.append(declarar("combinadas_rellenadas", columna=finales[j],
                                          cuantas=rellenadas_por_columna[c_abs],
                                          rangos=rangos_rellenadas[c_abs][:10]))
        if vacias_por_columna.get(c_abs):
            declaraciones.append(declarar("celdas_combinadas_vacias", columna=finales[j],
                                          cuantas=vacias_por_columna[c_abs],
                                          rangos=rangos_vacias[c_abs][:10]))
    for j in conserva:
        if fechas_a_medianoche.get(j):
            declaraciones.append(declarar("fechas_a_medianoche", columna=finales[j],
                                          cuantas=fechas_a_medianoche[j]))
        for suceso, tipo_decl in (("no_finito", "no_finito"), ("fecha_truncada", "fecha_truncada"),
                                  ("hora", "hora_del_dia"), ("duracion", "duracion")):
            n = sucesos.get((j, suceso), 0)
            if n:
                declaraciones.append(declarar(tipo_decl, columna=finales[j], cuantas=n))
        tipos = tipos_col[j]
        if len(tipos) > 1:
            declaraciones.append(declarar("columna_mixta", columna=finales[j],
                                          cuantas=sum(tipos.values()), tipos=dict(sorted(tipos.items()))))
        elif set(tipos) == {"numero"} and series_ok[j]:
            declaraciones.append(declarar("posible_fecha_en_serie", columna=finales[j],
                                          cuantas=tipos["numero"]))
    if hz is not None:
        for c_abs in sorted(hz.errores_por_columna):
            j = c_abs - col0
            if j in finales:
                tipos = hz.errores_por_columna[c_abs]
                declaraciones.append(declarar("errores_de_celda", columna=finales[j],
                                              cuantas=sum(tipos.values()), tipos=sorted(tipos)))
        for c_abs in sorted(hz.porcentajes_por_columna):
            j = c_abs - col0
            if j in finales:
                declaraciones.append(declarar("formato_porcentaje", columna=finales[j],
                                              cuantas=hz.porcentajes_por_columna[c_abs]))
    if geom is not None:
        ocultas = [r for r in geom.filas_ocultas if r >= fd and r not in excluidas]
        if ocultas:
            declaraciones.append(declarar("filas_ocultas", cuantas=len(ocultas), filas=ocultas[:10]))
        col_ocultas = [c for c in geom.columnas_ocultas if (c - col0) in finales]
        if col_ocultas:
            declaraciones.append(declarar("columnas_ocultas", cuantas=len(col_ocultas),
                                          letras=[letras_de_columna(c) for c in col_ocultas[:10]]))
    if libro.formato == "xls":
        declaraciones.append(declarar("no_comprobable_en_xls",
                                      que=["fórmulas sin valor guardado", "errores de celda",
                                           "formato de porcentaje", "filas y columnas ocultas"]))
    if libro.macros:
        declaraciones.append(declarar("macros_ignoradas"))
    if meta["estado"] != "visible":
        declaraciones.append(declarar("hoja_oculta", hoja=meta["nombre"]))
    previas = [r for r in range(h.fila0, fh) if any(not _vacia(v) for v in h.filas[r - h.fila0])]
    if previas:
        textos = [next((_texto(v) for v in h.filas[r - h.fila0] if not _vacia(v)), "") for r in previas[:3]]
        declaraciones.append(declarar("filas_previas_omitidas", cuantas=len(previas),
                                      filas=previas[:10], textos=textos))
    filas_excluidas = []
    for f in excluir:
        texto = ""
        if h.fila0 <= f <= fin:
            texto = next((_texto(v) for v in h.filas[f - h.fila0] if not _vacia(v)), "")
        filas_excluidas.append({"fila": f, "texto": texto})
    if excluir:
        declaraciones.append(declarar("filas_excluidas", cuantas=len(excluir), filas=excluir))
    if filas_vacias:
        declaraciones.append(declarar("filas_vacias_quitadas", cuantas=len(filas_vacias),
                                      filas=filas_vacias[:10]))

    # ── el CSV ──
    if len(conserva) < ancho:
        filas_csv = [[f[j] for j in conserva] for f in filas_csv]
    csv_bytes = escribir_csv([finales[j] for j in conserva], filas_csv)

    primera_col, ultima_col = col0 + conserva[0], col0 + conserva[-1]
    origen = {
        "esquema": ESQUEMA_DE_ORIGEN,
        "formato": libro.formato,
        "nombre": libro.nombre,
        "sha256": hashlib.sha256(libro.datos).hexdigest(),
        "bytes": len(libro.datos),
        "hoja": meta["nombre"],
        "fila_de_cabecera": fh,
        "filas_de_cabecera": nh,
        "rango": _rango_a1(fh, primera_col, fin, ultima_col),
        "filas_excluidas": filas_excluidas,
        "conversor": libro.conversor(),
        "declaraciones": declaraciones,
    }
    return TablaConvertida(csv_bytes=csv_bytes, origen=origen)
