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

import bisect
import functools
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from . import aislado
from .declaraciones import declarar
from .errores import TablaRechazada, rechazar
from .formato import abrir_zip_de_excel, detectar_formato, es_ole_de_excel
from .topes import (TOPES_DE_FABRICA, Topes, comprobar_cadenas_compartidas, comprobar_memoria,
                    ram_estimada_de_una_hoja, techo_del_lector)
from .valores import DOS_A_LA_53, escribir_csv, numero_a_texto, tipo_de_valor, valor_a_texto
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
    #: El ancho del rango de calamine. Las filas llegan del lector SIN las celdas vacías del final
    #: (una fila vacía es `[]`), así que el ancho no se puede deducir de ellas.
    ancho_del_rango: int | None = None

    @property
    def vacia(self) -> bool:
        return not self.filas

    @property
    def ultima_fila(self) -> int:
        return self.fila0 + len(self.filas) - 1

    @property
    def ancho(self) -> int:
        if self.ancho_del_rango is not None:
            return self.ancho_del_rango
        return max((len(f) for f in self.filas), default=0)


def _importar_calamine() -> None:
    """Comprueba que el lector existe SIN importarlo en este proceso (se importa en el hijo)."""
    if not aislado.calamine_disponible():
        raise rechazar("falta_lector")


def _version_de_calamine() -> str | None:
    try:
        from importlib import metadata
        return metadata.version("python-calamine")
    except Exception:
        return None


_NO_COMPROBABLE_EN_XLS = ["fórmulas sin valor guardado", "errores de celda", "formato de porcentaje",
                          "formato con ceros a la izquierda", "fechas que no son fechas",
                          "filas y columnas ocultas"]


class _Libro:
    def __init__(self, datos: bytes, nombre: str, topes: Topes):
        self.datos = bytes(datos)
        self.nombre = _nombre_corto(nombre)
        self.topes = topes
        self.zf = None
        self.macros = False
        self.descomprimido = 0
        self.bytes_de_cadenas = 0
        self.epoca_1904 = False
        self.recalcular_al_abrir = False
        self._estilos = None
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
        if familia == "pdf":
            raise rechazar("pdf_no_excel", extension=extension)
        if familia == "xml":
            raise rechazar("xml_no_excel", extension=extension)
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
        self.descomprimido = info["descomprimido"]
        # calamine carga la tabla de textos compartidos ENTERA al abrir el libro: tope propio y
        # entra en la estimación de memoria de cada hoja.
        self.bytes_de_cadenas = xlsx_xml.cadenas_compartidas(self.zf)
        comprobar_cadenas_compartidas(self.bytes_de_cadenas, self.topes)
        self.epoca_1904 = xlsx_xml.es_fecha_1904(self.zf)
        self.recalcular_al_abrir = xlsx_xml.pide_recalcular_al_abrir(self.zf)
        return "xlsm" if self.macros else "xlsx"

    # -- hojas ------------------------------------------------------------------------------
    def _listar_hojas(self) -> list[dict]:
        if self.zf is not None:
            return xlsx_xml.mapa_de_hojas(self.zf)
        _importar_calamine()
        necesario = ram_estimada_de_una_hoja(_CELDAS_MAXIMAS_DE_UN_XLS)
        hojas = []
        for nombre, visible in aislado.listar_hojas(
                self.datos, memoria=techo_del_lector(necesario, len(self.datos), self.topes),
                segundos=self.topes.max_segundos_lector):
            estado = {"Visible": "visible", "Hidden": "hidden", "VeryHidden": "veryHidden"}.get(
                visible, "visible")
            hojas.append({"nombre": nombre, "estado": estado, "ruta": None})
        return hojas

    def estilos(self) -> xlsx_xml.Estilos:
        if self._estilos is None:
            self._estilos = (xlsx_xml.estilos_del_libro(self.zf) if self.zf is not None
                             else xlsx_xml.Estilos())
        return self._estilos

    def meta(self, nombre: str) -> dict:
        for h in self.hojas:
            if h["nombre"] == nombre:
                return h
        raise rechazar("hoja_no_encontrada", hoja=nombre, hojas=[h["nombre"] for h in self.hojas])

    def elegir(self, hoja) -> dict:
        if not self.hojas:
            raise rechazar("sin_hojas")
        if hoja is None:
            visibles = [h for h in self.hojas if h["estado"] == "visible" and not h.get("grafico")]
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
        return xlsx_xml.explorar_hoja(self.zf, meta["ruta"], self.estilos(),
                                      self.topes.max_filas, self.topes.max_columnas,
                                      hoja=meta["nombre"], epoca_1904=self.epoca_1904)

    # -- leer una hoja con las guardas ANTES de que el lector reserve nada ------------------
    def leer(self, meta: dict, geom=None) -> _HojaLeida:
        """Lee la hoja con calamine en un proceso hijo con techo (`aislado`). La geometría solo
        sirve para la guarda de memoria: que la hoja está VACÍA lo dice calamine, nunca ella."""
        nombre = meta["nombre"]
        _importar_calamine()
        segundos = self.topes.max_segundos_lector

        def techo(necesario: int) -> int:
            return techo_del_lector(necesario, len(self.datos), self.topes)

        if self.zf is None:
            necesario = ram_estimada_de_una_hoja(_CELDAS_MAXIMAS_DE_UN_XLS)
            comprobar_memoria(necesario, nombre)
            r = aislado.leer_hoja(self.datos, nombre, memoria=techo(necesario), segundos=segundos)
        else:
            ruta = meta["ruta"]
            tam = self.zf.getinfo(ruta).file_size if ruta is not None else self.descomprimido
            # La cota que se sabe SIN recorrer la hoja (su XML y los textos compartidos): si ni
            # eso cabe, no se abre nada.
            provisional = ram_estimada_de_una_hoja(0, tam, self.bytes_de_cadenas)
            comprobar_memoria(provisional, nombre)
            lectura = None
            if geom is None and ruta is not None:
                # calamine (en su hijo, con el techo provisional) y la geometría (aquí) A LA VEZ:
                # el tiempo es el mayor de los dos, no la suma. Si la geometría rechaza, se mata
                # al hijo antes de que acabe.
                lectura = aislado.empezar_a_leer(self.datos, nombre, memoria=techo(provisional),
                                                 segundos=segundos)
            try:
                if geom is None:
                    geom = self.geometria(meta)
                celdas = 0
                if geom is not None:
                    celdas = geom.celdas_del_rango
                    if not geom.exacta:   # alguna referencia no se entiende: el rectángulo es una cota
                        celdas = max(celdas, tam // 20)
                necesario = ram_estimada_de_una_hoja(celdas, tam, self.bytes_de_cadenas)
                comprobar_memoria(necesario, nombre)
            except BaseException:
                if lectura is not None:
                    lectura.cancelar()
                raise
            if lectura is None:
                r = aislado.leer_hoja(self.datos, nombre, memoria=techo(necesario), segundos=segundos)
            else:
                try:
                    r = lectura.esperar()
                except TablaRechazada as e:
                    # El techo provisional no sabía del rectángulo (una celda suelta en Z1048576
                    # reserva 1 GB con un XML de 2 KB): con el de verdad, se repite.
                    if e.codigo != "lector_sin_memoria" or techo(necesario) <= techo(provisional):
                        raise
                    r = aislado.leer_hoja(self.datos, nombre, memoria=techo(necesario),
                                          segundos=segundos)
        if r["inicio"] is None:
            return _HojaLeida(nombre, geom=geom)
        inicio, filas = r["inicio"], r["filas"]
        h = _HojaLeida(nombre, inicio[0] + 1, inicio[1] + 1, filas,
                       [(a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1) for a, b in r["combinadas"]], geom,
                       ancho_del_rango=r.get("ancho"))
        if len(filas) > self.topes.max_filas:
            raise rechazar("demasiadas_filas", hoja=nombre, filas=len(filas), tope=self.topes.max_filas)
        if h.ancho > self.topes.max_columnas:
            raise rechazar("demasiadas_columnas", hoja=nombre, columnas=h.ancho,
                           tope=self.topes.max_columnas)
        return h

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
    fila = h.filas[i][:_MAX_COLUMNAS_PREVISTA]
    celdas = [valor_a_texto(v)[0][:200] for v in fila]
    celdas += [""] * (min(h.ancho, _MAX_COLUMNAS_PREVISTA) - len(celdas))
    return {"fila": h.fila0 + i, "celdas": celdas}


def _nombres_propuestos(h: _HojaLeida, fila_cab: int) -> list[str]:
    i = fila_cab - h.fila0
    nombres = [_texto(v) for v in h.filas[i]]
    return nombres + [""] * (h.ancho - len(nombres))


def _sin_excepciones_ajenas(f):
    """Invariante 10: ninguna excepción de una biblioteca llega a la persona. Lo que no es ya una
    `TablaRechazada` (un fallo NUESTRO que se escapó a las guardas) sale como `error_interno`, que
    dice que el fallo es del conversor y no del fichero —no se disfraza de «fichero dañado»—, y
    conserva la causa encadenada para quien depura. `AssertionError` no se toca (pruebas)."""
    @functools.wraps(f)
    def envoltorio(*a, **k):
        try:
            return f(*a, **k)
        except (TablaRechazada, AssertionError):
            raise
        except Exception as e:
            raise rechazar("error_interno", tipo=type(e).__name__) from e
    return envoltorio


# ── inspeccionar ────────────────────────────────────────────────────────────────────────
@_sin_excepciones_ajenas
def inspeccionar(datos: bytes, nombre: str = "", *, topes: Topes | None = None) -> dict:
    """Qué hay en el libro, sin convertir nada. Propone la hoja, la cabecera y los totales."""
    libro = _Libro(datos, nombre, topes or TOPES_DE_FABRICA)
    visibles = [h for h in libro.hojas if h["estado"] == "visible" and not h.get("grafico")]
    hojas = [_inspeccionar_hoja(libro, m) for m in libro.hojas]
    avisos_globales = []
    if libro.macros:
        avisos_globales.append(declarar("macros_ignoradas"))
    if libro.formato == "xls":
        avisos_globales.append(declarar("no_comprobable_en_xls", que=_NO_COMPROBABLE_EN_XLS))
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
        if estado != "visible":
            # Una hoja oculta se LISTA con sus medidas, pero no se carga: no se ofrece.
            geom = libro.geometria(meta)
            if geom is not None and not geom.vacia:
                entrada.update(filas=geom.filas, columnas=geom.columnas,
                               rango=_rango_a1(geom.min_fila, geom.min_col, geom.max_fila, geom.max_col))
            return entrada
        h = libro.leer(meta)
    except TablaRechazada as e:
        if e.codigo in _ERRORES_DE_UNA_HOJA:
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
        # Un rango repetido en `<mergeCells>` es UN rango (como al convertir: `set`).
        distintas = sorted(set(h.combinadas))
        avisos.append(declarar("hay_celdas_combinadas", cuantas=len(distintas),
                               rangos=[_rango_a1(*m[:2], *m[2:]) for m in distintas[:10]]))
    geom = h.geom
    if geom is not None:
        ocultas = geom.filas_ocultas_en(h.fila0, h.ultima_fila)
        if ocultas:
            avisos.append(declarar("filas_ocultas", cuantas=len(ocultas), filas=ocultas[:10]))
        col_ocultas = geom.columnas_ocultas_entre(h.col0, h.col0 + h.ancho - 1)
        if col_ocultas:
            avisos.append(declarar("columnas_ocultas", cuantas=len(col_ocultas),
                                   letras=[letras_de_columna(c) for c in col_ocultas[:10]]))
        hz = geom.hallazgos()
        if hz.formulas_sin_valor:
            avisos.append(declarar("formulas_sin_valor", cuantas=hz.formulas_sin_valor,
                                   celdas=hz.celdas_sin_valor))
        if libro.recalcular_al_abrir and hz.formulas:
            avisos.append(declarar("formulas_a_recalcular", cuantas=hz.formulas,
                                   celdas=hz.celdas_con_formula))
        _avisos_de_hallazgos(hz, letras_de_columna, avisos)
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


def _avisos_de_hallazgos(hz, nombre_de, avisos: list) -> None:
    """Errores, porcentajes, ceros a la izquierda y fechas que no son fechas, por columna.
    `nombre_de(columna_absoluta)` da el nombre con que se declara (o `None` si no se declara)."""
    def col(c):
        return nombre_de(c)
    for c in sorted(hz.errores_por_columna):
        if col(c) is not None:
            tipos = hz.errores_por_columna[c]
            avisos.append(declarar("errores_de_celda", columna=col(c),
                                   cuantas=sum(tipos.values()), tipos=sorted(tipos)))
    for c in sorted(hz.porcentajes_por_columna):
        if col(c) is not None:
            avisos.append(declarar("formato_porcentaje", columna=col(c),
                                   cuantas=hz.porcentajes_por_columna[c]))
    for c in sorted(hz.ceros_por_columna):
        if col(c) is not None:
            d = hz.ceros_por_columna[c]
            ejemplo = _ejemplo_numerico(d["ejemplo"])
            avisos.append(declarar("formato_con_ceros", columna=col(c), cuantas=d["cuantas"],
                                   codigo=d["codigo"], cifras=d["cifras"], ejemplo=ejemplo,
                                   visible=_como_lo_ensena_excel(d["ejemplo"], d["codigo"])))
    for c in sorted(hz.fechas_no_validas_por_columna):
        if col(c) is not None:
            motivos = hz.fechas_no_validas_por_columna[c]
            avisos.append(declarar("fecha_no_valida", columna=col(c), cuantas=sum(motivos.values()),
                                   motivos=dict(sorted(motivos.items())),
                                   celdas=hz.celdas_de_fecha_no_valida.get(c, [])))


def _ejemplo_numerico(v: str) -> str:
    try:
        return numero_a_texto(float(v))
    except (TypeError, ValueError):
        return str(v)


def _como_lo_ensena_excel(v: str, codigo: str) -> str:
    """Lo que Excel enseña para el número `v` con un formato de ceros (`00000`, `000-00-0000`):
    las cifras, redondeadas como Excel (la mitad, hacia fuera del cero), rellenan los `0` del
    código de derecha a izquierda. Si el código lleva más que ceros, almohadillas y separadores,
    no se finge: se dice «con ceros delante»."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "con ceros delante"
    plano = xlsx_xml._RE_LITERAL.sub("", codigo.split(";", 1)[0])
    if not plano or any(ch not in "0#-/ ()" for ch in plano):
        return "con ceros delante"
    n = int(abs(x) + 0.5)
    cifras = list(str(n).zfill(plano.count("0")))
    salida = []
    for ch in reversed(plano):
        if ch in "0#":
            salida.append(cifras.pop() if cifras else "")
        else:
            salida.append(ch)
    texto = "".join(cifras) + "".join(reversed(salida))
    return ("-" if x < 0 and n else "") + texto


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


_ERRORES_DE_UNA_HOJA = ("memoria_insuficiente", "demasiadas_filas", "demasiadas_columnas",
                        "lector_fallo", "lector_sin_memoria", "lector_tiempo", "lector_abortado")


def _rango_a1(f0: int, c0: int, f1: int, c1: int) -> str:
    return f"{referencia(f0, c0)}:{referencia(f1, c1)}"


# ── convertir ───────────────────────────────────────────────────────────────────────────
def _combinadas_en_lo_convertido(h: _HojaLeida, fd: int, fin: int, excluir: list[int],
                                 rellenar: bool):
    """Las celdas cubiertas por una combinación dentro de lo que se convierte, por columna.

    Llegan VACÍAS del lector. Se cuentan por columna (y por rango) y se declaran; solo si la
    persona lo marcó (`rellenar`) se rellenan con la celda combinada, y entonces se declara ESO.

    Se trabaja por RANGOS recortados a lo que hay (filas `fd..fin`, columnas de la hoja leída):
    un rango `A2:XFD2` sobre una hoja de 2 columnas cuesta 2 columnas, no 16.384 (auditoría I1:
    6.000 de esos costaban 53 s y 857 MB). Las filas cubiertas se cuentan con aritmética (y las
    excluidas con `bisect`), sin recorrerlas. Excel no solapa combinaciones: si lo recortado
    suma más celdas que la propia zona, el libro está fabricado y se rechaza."""
    col0, ancho = h.col0, h.ancho
    vacias: dict[int, int] = {}
    rellenadas: dict[int, int] = {}
    rangos_v: dict[int, list[str]] = {}
    rangos_r: dict[int, list[str]] = {}
    if not h.combinadas or fd > fin or ancho == 0:
        return vacias, rangos_v, rellenadas, rangos_r
    zona = (fin - fd + 1) * ancho
    total = 0
    excl = set(excluir)
    for (r0, c0, r1, c1) in sorted(set(h.combinadas)):
        lo, hi = max(r0, fd), min(r1, fin)
        a, b = max(c0, col0), min(c1, col0 + ancho - 1)
        if lo > hi or a > b:
            continue
        n_filas = (hi - lo + 1) - (bisect.bisect_right(excluir, hi) - bisect.bisect_left(excluir, lo))
        if n_filas <= 0:
            continue
        total += n_filas * (b - a + 1)
        if total > zona:
            raise rechazar("lector_fallo", tipo="celdas combinadas que se solapan")
        origen_dentro = lo <= r0 <= hi and r0 not in excl
        valor = ""
        if rellenar and 0 <= r0 - h.fila0 < len(h.filas):
            fila_origen = h.filas[r0 - h.fila0]
            if 0 <= c0 - col0 < len(fila_origen):
                valor = fila_origen[c0 - col0]
        rango = _rango_a1(r0, c0, r1, c1)
        for c in range(a, b + 1):
            cubiertas = n_filas - (1 if (c == c0 and origen_dentro) else 0)
            if cubiertas <= 0:
                continue
            if rellenar and not _vacia(valor):
                j = c - col0
                for r in range(lo, hi + 1):
                    if r in excl or (r == r0 and c == c0):
                        continue
                    fila = h.filas[r - h.fila0]
                    if j >= len(fila):
                        fila.extend([""] * (j + 1 - len(fila)))
                    fila[j] = valor
                rellenadas[c] = rellenadas.get(c, 0) + cubiertas
                lista = rangos_r.setdefault(c, [])
            else:
                vacias[c] = vacias.get(c, 0) + cubiertas
                lista = rangos_v.setdefault(c, [])
            if len(lista) < 10:
                lista.append(rango)
    return vacias, rangos_v, rellenadas, rangos_r


@_sin_excepciones_ajenas
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
        lista_excluir = list(filas_a_excluir)
    except TypeError:
        raise rechazar("parametro_invalido", parametro="filas_a_excluir debe ser una lista de números de fila") from None
    # Solo enteros: `2.5` o `"3"` se convertían con `int()` y se quitaba OTRA fila (o una que no
    # se pidió así), declarando «se quitó 1 fila» (menor de la 2.ª re-auditoría).
    if any(isinstance(f, bool) or not isinstance(f, int) for f in lista_excluir):
        raise rechazar("parametro_invalido", parametro="filas_a_excluir debe ser una lista de números de fila enteros")
    excluir = sorted(set(lista_excluir))
    if any(f < 1 or f > _MAX_FILAS_FISICAS for f in excluir):
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
    if any(f > fin for f in excluir):
        # Una fila que la hoja no tiene no se «quita»: declararla quitada sería mentir.
        raise rechazar("parametro_invalido",
                       parametro=f"filas_a_excluir {[f for f in excluir if f > fin]} no existe en la hoja (la última fila es la {fin})")
    excluidas = set(excluir)

    # Fórmulas sin valor: se comprueban sobre lo que SE CONVIERTE, cabecera incluida (la fila
    # TOTAL que la persona quita no rechaza). Nunca se calcula una fórmula.
    geom = h.geom
    if geom is not None and geom.hay_formulas:
        hzf = geom.hallazgos(lambda r: r >= fh and r not in excluidas, lambda r: False)
        if hzf.formulas_sin_valor:
            raise rechazar("formula_sin_valor", cuantas=hzf.formulas_sin_valor,
                           celdas=hzf.celdas_sin_valor)
        # I-R1: el libro pide recalcular al abrir (XlsxWriter): el valor guardado es un relleno.
        if libro.recalcular_al_abrir and hzf.formulas:
            raise rechazar("formula_sin_valor", variante="recalcular", cuantas=hzf.formulas,
                           celdas=hzf.celdas_con_formula, recalcular_al_abrir=True)
    # Series con formato de fecha que no son una fecha (29-02-1900, ≤ 0, > 9999-12-31): lo que
    # calamine da para ellas es OTRA cosa (una hora, el 28-02, el número): quedan vacías y se
    # declaran, celda a celda.
    anuladas = (geom.fechas_no_validas(lambda r: r >= fd and r not in excluidas)
                if geom is not None else set())

    ancho = h.ancho
    col0 = h.col0
    vacias_por_columna, rangos_vacias, rellenadas_por_columna, rangos_rellenadas = \
        _combinadas_en_lo_convertido(h, fd, fin, excluir, rellenar_combinadas)

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
    ejemplos_no_exactos: dict[int, list[str]] = {}
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
            if anuladas and (r, col0 + j) in anuladas:
                continue
            t = type(v)
            if t is str:
                texto, suceso = v, None
                tipo = "texto"
                series_ok[j] = False
            elif t is float:
                suceso = None
                if v - v == 0.0:   # finito (NaN e infinito dan NaN)
                    if v.is_integer():
                        texto = str(int(v)) if -DOS_A_LA_53 < v < DOS_A_LA_53 else numero_a_texto(v)
                        if not -DOS_A_LA_53 < v < DOS_A_LA_53:
                            suceso = "entero_no_exacto"
                    else:
                        texto = repr(v)
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
                if suceso == "entero_no_exacto" and len(ejemplos_no_exactos.setdefault(j, [])) < 3:
                    ejemplos_no_exactos[j].append(texto)
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
        n = sucesos.get((j, "entero_no_exacto"), 0)
        if n:
            declaraciones.append(declarar("entero_no_exacto", columna=finales[j], cuantas=n,
                                          ejemplos=ejemplos_no_exactos.get(j, [])))
        tipos = tipos_col[j]
        if len(tipos) > 1:
            declaraciones.append(declarar("columna_mixta", columna=finales[j],
                                          cuantas=sum(tipos.values()), tipos=dict(sorted(tipos.items()))))
        elif set(tipos) == {"numero"} and series_ok[j]:
            declaraciones.append(declarar("posible_fecha_en_serie", columna=finales[j],
                                          cuantas=tipos["numero"]))
    if geom is not None:
        # Errores, porcentajes, ceros y fechas anuladas: solo los de las filas que SALEN en el CSV
        # (ni la cabecera, ni las excluidas, ni las que se quitaron por vacías).
        quitadas_vacias = set(filas_vacias)
        hz = geom.hallazgos(None, lambda r: r >= fd and r not in excluidas and r not in quitadas_vacias)
        _avisos_de_hallazgos(hz, lambda c: finales.get(c - col0), declaraciones)
        ocultas = geom.filas_ocultas_en(fd, fin, excluidas | quitadas_vacias)
        if ocultas:
            declaraciones.append(declarar("filas_ocultas", cuantas=len(ocultas), filas=ocultas[:10]))
        col_ocultas = [col0 + j for j in conserva if geom.columna_oculta(col0 + j)]
        if col_ocultas:
            declaraciones.append(declarar("columnas_ocultas", cuantas=len(col_ocultas),
                                          letras=[letras_de_columna(c) for c in col_ocultas[:10]]))
    if libro.formato == "xls":
        declaraciones.append(declarar("no_comprobable_en_xls", que=_NO_COMPROBABLE_EN_XLS))
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
