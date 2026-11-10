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
import datetime as _dt
import functools
import hashlib
import csv
import re
import unicodedata
from dataclasses import dataclass, field

from matrixai import nombre_citable

from . import aislado
from .declaraciones import declarar
from .errores import TablaRechazada, lista_para_texto, rechazar
from .formato import abrir_zip_de_excel, detectar_formato, es_ole_de_excel, xls_con_contrasena
from .parquet_lector import es_parquet
from .topes import (TOPES_DE_FABRICA, Topes, comprobar_cadenas_compartidas, comprobar_memoria,
                    memoria_virtual_del_proceso,
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
        sin_ext = {"variante": "sin_extension"} if "." not in self.nombre else {}
        if familia == "vacio":
            raise rechazar("vacio")
        if familia == "html":
            raise rechazar("html_no_excel")
        if familia == "pdf":
            raise rechazar("pdf_no_excel", extension=extension, **sin_ext)
        if familia == "xml":
            raise rechazar("xml_no_excel", extension=extension, **sin_ext)
        if familia == "texto":
            raise rechazar("texto_no_excel", extension=extension, **sin_ext)
        if familia == "xml_excel":
            raise rechazar("formato_no_soportado", formato="XML de Excel 2003")
        if familia == "parquet":
            # Un Parquet no llega aquí: `inspeccionar` y `convertir` lo desvían a `parquet_lector` antes de abrir un libro.
            raise rechazar("formato_desconocido")
        if familia == "desconocido":
            raise rechazar("formato_desconocido")
        if familia == "ole":
            if any(f in self.datos for f in (b"E\x00n\x00c\x00r\x00y\x00p\x00t\x00e\x00d\x00P\x00a\x00c\x00k\x00a\x00g\x00e\x00",)):
                raise rechazar("protegido")
            if not es_ole_de_excel(self.datos):
                raise rechazar("no_es_excel")
            if xls_con_contrasena(self.datos):
                raise rechazar("protegido")
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
        # Lo que este proceso ocupa ANTES de leer la hoja: la guarda compara lo que la lectura va a pedir con el techo
        # del proceso menos esto (no menos lo de cada momento, que ya incluye parte de lo pedido).
        al_empezar = memoria_virtual_del_proceso()

        def techo(necesario: int) -> int:
            return techo_del_lector(necesario, len(self.datos), self.topes)

        if self.zf is None:
            necesario = ram_estimada_de_una_hoja(_CELDAS_MAXIMAS_DE_UN_XLS)
            comprobar_memoria(necesario, nombre, usado_al_empezar=al_empezar)
            r = aislado.leer_hoja(self.datos, nombre, memoria=techo(necesario), segundos=segundos)
        else:
            ruta = meta["ruta"]
            tam = self.zf.getinfo(ruta).file_size if ruta is not None else self.descomprimido
            # La cota que se sabe SIN recorrer la hoja (su XML y los textos compartidos): si ni
            # eso cabe, no se abre nada.
            provisional = ram_estimada_de_una_hoja(0, tam, self.bytes_de_cadenas)
            comprobar_memoria(provisional, nombre, usado_al_empezar=al_empezar)
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
                comprobar_memoria(necesario, nombre, usado_al_empezar=al_empezar)
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
    """El texto de una celda para un nombre o una vista previa: sin espacios sobrantes ni la marca U+FEFF del principio.

    Remate de 150-C4: un nombre de columna que empezaba por U+FEFF (un BOM pegado en la celda) salía así en el CSV, y
    la entrada canónica del análisis lo quita al principio del texto: la pantalla veía `paciente` y el estudio, que
    recibe el CSV de la conversión tal cual, `\ufeffpaciente`. El CSV de la conversión es un punto fijo de esa entrada
    canónica; si cambia el nombre, `nombre_normalizado` lo declara."""
    t = valor_a_texto(v)[0]
    return " ".join(t.lstrip("\ufeff").split())


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


#: Cuántas filas vacías seguidas se toleran entre la cabecera propuesta y sus datos.
_VACIAS_TRAS_CABECERA = 3


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
        # 150-C7 H2: una cabecera con filas VACÍAS debajo sigue siendo cabecera si lo siguiente con contenido son
        # datos (CTG «Raw Data»: cabecera en la fila 1, fila 2 vacía): saltarla proponía un dato como cabecera.
        # 150-C7 (libros públicos): una banda de segundo nivel («Number of detections», combinada sobre varias columnas, o
        # «Todas las edades» del INE) deja UNA sola celda con contenido entre la cabecera y los datos: no es un dato ni
        # cierra la búsqueda; se salta como las vacías. Sin esto la cabecera real se descartaba y se proponía el primer
        # dato (UK Home Office «Table 2b»: fila 6) o ninguna (INE población, «Table 4»).
        siguiente, salto_de_una_celda, k_sig = None, False, None
        for k in range(i + 1, min(len(h.filas), i + 1 + _VACIAS_TRAS_CABECERA + 1)):
            con_dato = sum(1 for v in h.filas[k] if not _vacia(v))
            if con_dato >= 2:
                siguiente, k_sig = h.filas[k], k
                break
            salto_de_una_celda = salto_de_una_celda or con_dato == 1
        if siguiente is None:
            continue
        # Sin salto, la regla de siempre (lo que ya acertaba no cambia). Tras saltar una fila de UNA celda, ver
        # `_lo_de_detras_del_salto_es_otra_cabecera`.
        if salto_de_una_celda and _lo_de_detras_del_salto_es_otra_cabecera(h, len(textos), siguiente, k_sig, fila):
            continue
        return h.fila0 + i
    return None


def _es_un_nombre(v) -> bool:
    """Un texto que podría ser un NOMBRE de columna: no vacío y que no es una marca de ausencia de las que el análisis lee como
    vacío (`-`, `n/a`, `NA`…): una celda «-» en una fila de datos no la vuelve una fila de nombres."""
    if not isinstance(v, str) or not v.strip():
        return False
    from matrixai.training.dataset_analysis import _NULL_TOKENS  # noqa: PLC0415 (las mismas marcas que el análisis)
    return v.strip().lower() not in _NULL_TOKENS


def _lo_de_detras_del_salto_es_otra_cabecera(h: _HojaLeida, textos_de_la_candidata: int, siguiente: list, k_sig: int,
                                             candidata: list | None = None) -> bool:
    """¿La fila con datos que hay DETRÁS de una fila de una celda (un título, una banda) es la cabecera de verdad, y no los datos de
    la candidata? Entonces la candidata no es la cabecera.

    Re-auditoría del 150-C7 (I-R1). La regla del 1.er arreglo («lo de detrás tiene menos textos») fallaba por los dos lados:
      * unos metadatos de TRES textos encima de un título y una cabecera de DOS («Empresa | ACME | T1 2024», «Ventas», «mes |
        importe», «ene | 10»): lo de detrás tenía menos textos y se proponían los metadatos;
      * una cabecera con su banda de 2.º nivel y datos que son TODO texto («nombre | ciudad | categoría», «Todas», «Ana | Madrid |
        A»): lo de detrás tenía tantos textos y se proponía un dato.
    La señal que separa los dos es la de los TIPOS, no la de contar textos:
      1. Lo de detrás es TODO nombres y en la fila siguiente hay un número, una fecha o un booleano BAJO uno de esos nombres: es
         una cabecera sobre sus datos (los metadatos y el título van encima). Un dato de texto no tiene un número debajo de él
         en su misma columna porque sí; un nombre, sí.
      2. Lo de detrás mezcla textos con datos tipados y tiene AL MENOS tantos textos como la candidata: la candidata es una fila de
         datos de texto sobre una sección («Madrid | norte | 10», «SUR», «Sevilla | sur | 9»), no una cabecera.
         NO se aplica si los números de la candidata son PERIODOS (`_numeros_de_periodos`): «Provincia | Sexo | 2022 | 2023» con
         «(en miles)» debajo y «Cantabria | Mujeres | 300 | 301» tiene la forma de datos sobre una sección, pero es la cabecera
         con los años tecleados como números (3.ª auditoría, IMPORTANTE-1).
    En lo demás, la candidata es la cabecera: la banda del INE o de «Table 2b» va seguida de una etiqueta y números (pocos
    textos), y una cabecera sobre datos todo texto lleva detrás más texto (sin nada tipado que la delate)."""
    con_dato = [v for v in siguiente if not _vacia(v)]
    if all(isinstance(v, str) for v in con_dato):
        detras = next((f for f in h.filas[k_sig + 1:k_sig + 1 + _VACIAS_TRAS_CABECERA + 1]
                       if sum(1 for v in f if not _vacia(v)) >= 2), [])
        return any(j < len(siguiente) and _es_un_nombre(siguiente[j]) and not _vacia(v) and not isinstance(v, str)
                   for j, v in enumerate(detras))
    if candidata is not None and _numeros_de_periodos(candidata):
        return False
    return _textos_distintos(siguiente) >= textos_de_la_candidata


def _numeros_de_periodos(fila: list) -> bool:
    """¿Los NÚMEROS de la fila son nombres de periodo, y no datos? Dos o más enteros que son años seguidos y crecientes
    (entre 1900 y 2100: 2022, 2023) o, a partir de tres, una numeración correlativa (1, 2, 3: meses, semanas, trimestres).
    Un único número, o uno que no encaja («10», 1234), sigue siendo un dato."""
    nums = []
    for v in fila:
        if _vacia(v) or isinstance(v, (str, bool)):
            continue
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if not isinstance(v, int):
            return False
        nums.append(v)
    if len(nums) < 2 or any(b <= a for a, b in zip(nums, nums[1:])):
        return False
    if all(1900 <= n <= 2100 for n in nums):
        return True
    return len(nums) >= 3 and all(b - a == 1 for a, b in zip(nums, nums[1:]))


def _textos_distintos(fila: list) -> int:
    return len({v.strip() for v in fila if isinstance(v, str) and v.strip()})


_SEPARADORES_PEGADOS = (("coma", ","), ("punto_y_coma", ";"), ("tabulador", "\t"))


def _columna_unica_con_separadores(h: _HojaLeida) -> tuple[str, int] | None:
    """150-C7 H1: ¿es la hoja un CSV pegado entero en una columna? -> `(separador, campos)` o `None`.

    UNA columna de texto en la que ≥ 90 % de las celdas (de las primeras 500 con contenido, al menos 3) se parten con el
    mismo separador en el MISMO número (≥ 2) de campos, leyendo las comillas como un CSV. Solo avisa: no parte nada."""
    if h.ancho != 1:
        return None
    celdas = []
    for f in h.filas:
        v = f[0] if f else None
        if _vacia(v):
            continue
        if not isinstance(v, str):
            return None   # un número o una fecha en la columna: es una columna de datos, no un CSV pegado
        celdas.append(v)
        if len(celdas) >= 500:
            break
    if len(celdas) < 3:
        return None
    # Reparación de la auditoría del 150-C7 (I1): primero los separadores que lleva la PRIMERA celda (la cabecera del CSV pegado),
    # como hace la entrada canónica con un `.csv` (la cabecera decide: sin comas y con `;` es un CSV `;`). Sin esto, un CSV `;`
    # español con DOS columnas de decimal coma («2024-01-01;18,5;2,25;no») se partía por COMAS en 3 campos que no son nada.
    primera = celdas[0]
    orden = sorted(_SEPARADORES_PEGADOS, key=lambda cs: cs[1] not in primera)
    for clave, sep in orden:
        cuentas: dict[int, int] = {}
        for c in celdas:
            if sep not in c:
                continue
            try:
                n = len(next(csv.reader([c], delimiter=sep)))
            except (csv.Error, StopIteration):
                continue
            cuentas[n] = cuentas.get(n, 0) + 1
        if cuentas:
            n, veces = max(cuentas.items(), key=lambda kv: kv[1])
            if n >= 3 and veces >= 0.9 * len(celdas):
                return clave, n
    return None


_CARACTER_DE = {clave: sep for clave, sep in _SEPARADORES_PEGADOS}
_CLAVE_DE = {sep: clave for clave, sep in _SEPARADORES_PEGADOS}
_COLUMNAS_DE_LA_PARTIDA_PREVISTA = 12


def _partir_filas(h: _HojaLeida, sep: str) -> list[list[str]]:
    """150-C7 (H1, decidido por Roberto): LA regla de partir una columna única, en UN sitio (la vista previa y `convertir` la
    comparten). Cada celda se lee con un lector CSV de verdad (comillas, separador dentro de comillas, comillas dobladas), no con
    un `split`; el BOM del principio del texto se quita. Una celda vacía da una fila vacía; no se inventa nada."""
    salida: list[list[str]] = []
    primera = True
    for f in h.filas:
        v = f[0] if f else None
        if _vacia(v):
            salida.append([])
            continue
        texto = v if isinstance(v, str) else valor_a_texto(v)[0]
        if primera:
            texto = texto.lstrip("\ufeff")
            primera = False
        try:
            campos = next(csv.reader([texto], delimiter=sep), [])
        except csv.Error:
            raise rechazar("parametro_invalido",
                           parametro="una celda de la columna no se puede leer como CSV; no se parte",
                           parametro_en="a cell in the column cannot be read as CSV; it is not split") from None
        salida.append(campos)
    return salida


def _vista_de_la_partida(h: _HojaLeida, clave: str) -> dict:
    """Cómo QUEDARÍA la hoja partida (sin partir nada): las primeras filas con sus campos, para que la persona lo vea antes de aceptar."""
    partidas = _partir_filas(h, _CARACTER_DE[clave])
    ancho = max((len(f) for f in partidas), default=0)
    n = _COLUMNAS_DE_LA_PARTIDA_PREVISTA
    filas = []
    for i, f in enumerate(partidas):
        if len(filas) >= _PREVISTA_PRIMERAS:
            break
        if f:
            filas.append({"fila": h.fila0 + i, "celdas": f[:n]})
    return {"separador": clave, "columnas": ancho, "columnas_mostradas": min(ancho, n), "primeras": filas}


#: 150-C7 (decisión de Roberto, 09-11): lo que abre una NOTA AL PIE de una tabla de oficina («Notas:», «Fuente:», «1. …», «n/a …»,
#: «- Indica …», «* …»). No es un total: se ofrece con su propio motivo, `pie`.
_RE_PIE = re.compile(r"^(notas?|notes?|nota bene|fuentes?|sources?|elaboracion|footnotes?)\b|^\(?\d{1,2}[.)]\s|^n/a\b|^[-*\u2013\u2014]\s")


def _filas_de_pie(h: _HojaLeida, inicio: int) -> set[int]:
    """Las filas (índices) del PIE de la tabla: tras la última fila con ≥ 2 celdas, las de UNA celda de texto desde la primera que
    abre una nota (`_RE_PIE`) — y las que la siguen, como el nombre del organismo bajo «Fuente:»."""
    ultima_de_datos = max((i for i in range(inicio, len(h.filas)) if sum(1 for v in h.filas[i] if not _vacia(v)) >= 2),
                          default=inicio - 1)
    pie: set[int] = set()
    abierto = False
    for i in range(max(ultima_de_datos + 1, inicio), len(h.filas)):
        con_dato = [v for v in h.filas[i] if not _vacia(v)]
        if not con_dato:
            continue
        if len(con_dato) != 1 or not isinstance(con_dato[0], str):
            abierto = False
            continue
        if _RE_PIE.match(_sin_tildes(con_dato[0]).strip().lower()):
            abierto = True
        if abierto:
            pie.add(i)
    return pie


def _candidatas_a_total(h: _HojaLeida, desde: int) -> list[dict]:
    """Filas que PARECEN un total (`palabra`, o ser de las últimas con la mitad de celdas vacías: `mitad_vacia`) o una nota al pie
    (`pie`: «Notas:», «Fuente:», el organismo debajo, «1. …»). Propone; la persona confirma: nada se quita solo."""
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
    for i in _filas_de_pie(h, inicio):
        if candidatas.get(h.fila0 + i) != "palabra":
            candidatas[h.fila0 + i] = "pie"
    resultado = []
    for fila in sorted(candidatas):
        celdas = h.filas[fila - h.fila0]
        primera = next((_texto(v) for v in celdas if not _vacia(v)), "")
        resultado.append({"fila": fila, "texto": primera, "motivo": candidatas[fila]})
    return resultado


#: 150-C7 (decisión de Roberto, 09-11): cómo se componen los dos niveles de una cabecera con banda («Índice · 2025M12»).
SEPARADOR_DE_NIVELES = " · "
_NOMBRES_COMPUESTOS_PREVISTOS = 12


def _banda_de_dos_niveles(h: _HojaLeida, cab: int) -> dict | None:
    """150-C7 (decisión de Roberto, 09-11): ¿la fila de ENCIMA o de DEBAJO de la cabecera `cab` es una BANDA de primer nivel?

    Lo es si cada celda con dato de esa fila es un TEXTO que abre un rango combinado HORIZONTAL de esa misma fila (≥ 2 columnas) y
    tiene menos celdas con dato que la cabecera (INE IPC: «Índice» sobre los meses; Home Office «Table 2b»: «Number of detections»
    bajo los nombres). -> `{fila, posicion: "encima"|"debajo", grupos: [(col_ini, col_fin, texto)]}` o `None`. Solo DICE; componer
    o quitar la banda lo decide la persona en `convertir`."""
    i = cab - h.fila0
    if not 0 <= i < len(h.filas) or _textos_distintos(h.filas[i]) < 2:
        return None
    n_cab = sum(1 for v in h.filas[i] if not _vacia(v))
    for posicion, fila in (("encima", cab - 1), ("debajo", cab + 1)):
        k = fila - h.fila0
        if not 0 <= k < len(h.filas):
            continue
        celdas = [(j, v) for j, v in enumerate(h.filas[k]) if not _vacia(v)]
        if not celdas or len(celdas) >= n_cab:
            continue
        rangos = {c[1]: c for c in set(h.combinadas) if c[0] == fila and c[2] == fila and c[3] > c[1]}
        grupos = []
        for j, v in celdas:
            c = rangos.get(h.col0 + j)
            if not isinstance(v, str) or c is None:
                break
            grupos.append((c[1], c[3], _texto(v)))
        else:
            # Un rango que cubre TODAS las columnas con nombre no agrupa nada: es un título («England and Wales, …» encima de
            # «Table 2b»), no un primer nivel.
            con_nombre = {h.col0 + j for j, v in enumerate(h.filas[i]) if not _vacia(v)}
            if len(grupos) == 1 and con_nombre <= set(range(grupos[0][0], grupos[0][1] + 1)):
                continue
            # Re-auditoría del 150-C7 (I-R2): un grupo que cubre la PRIMERA columna de la tabla (la de las etiquetas: provincia,
            # región, grupo) no es un primer nivel: un primer nivel agrupa las columnas de DATOS que hay a la derecha de la
            # etiqueta (INE IPC: «Índice» sobre los meses; «Table 2b»: «Number of detections» desde la 2.ª). Lo que empieza en
            # la primera columna es un título («Ventas 2024» sobre A:C), unos metadatos («Empresa» A:B, «ACME» C:D; «Fecha: …»
            # A:C, «Autor: …» D:F) o una sección bajo la cabecera («Zona norte» A:E): se avisaba de una cabecera de dos niveles
            # que no lo es y se ofrecía componer «Ventas 2024 · region».
            if any(c_ini <= h.col0 <= c_fin for c_ini, c_fin, _ in grupos):
                continue
            return {"fila": fila, "posicion": posicion, "grupos": grupos}
    return None


def _componer_con_la_banda(nombres: list[str], banda: dict, col0: int) -> dict[int, str]:
    """Los nombres que se COMPONEN «nivel1 · nivel2» (índice de columna -> nombre): solo los de las columnas bajo un grupo de la
    banda y con nombre propio. Una columna sin grupo encima, o sin nombre, se queda como está."""
    compuestos: dict[int, str] = {}
    for c_ini, c_fin, grupo in banda["grupos"]:
        for c in range(c_ini, c_fin + 1):
            j = c - col0
            if 0 <= j < len(nombres) and nombres[j] and grupo:
                compuestos[j] = f"{grupo}{SEPARADOR_DE_NIVELES}{nombres[j]}"
    return compuestos


def _vista_de_la_banda(h: _HojaLeida, cab: int, banda: dict) -> dict:
    """Lo que se OFRECE (sin hacer nada): componer los nombres, con cómo quedarían los primeros; o, si la banda va DEBAJO de la
    cabecera (entre los nombres y los datos), quitar esa fila."""
    nombres = _nombres_propuestos(h, cab)
    compuestos = _componer_con_la_banda(nombres, banda, h.col0)
    primeros = [compuestos[j] for j in sorted(compuestos)][:_NOMBRES_COMPUESTOS_PREVISTOS]
    return {"fila_de_cabecera": cab, "fila_de_la_banda": banda["fila"], "posicion": banda["posicion"],
            "grupos": [g for _, _, g in banda["grupos"]][:_NOMBRES_COMPUESTOS_PREVISTOS],
            "columnas_compuestas": len(compuestos), "nombres_compuestos": primeros,
            "se_puede_quitar_la_banda": banda["posicion"] == "debajo"}


def _vista(h: _HojaLeida, i: int) -> dict:
    fila = h.filas[i][:_MAX_COLUMNAS_PREVISTA]
    celdas = [valor_a_texto(v)[0][:200] for v in fila]
    celdas += [""] * (min(h.ancho, _MAX_COLUMNAS_PREVISTA) - len(celdas))
    return {"fila": h.fila0 + i, "celdas": celdas}


def _nombres_propuestos(h: _HojaLeida, fila_cab: int) -> list[str]:
    i = fila_cab - h.fila0
    nombres = [_texto(v) for v in h.filas[i]]
    return nombres + [""] * (h.ancho - len(nombres))


def _nombres_de_dos_filas(h: _HojaLeida, fh: int) -> tuple[list[str], dict[int, tuple[str, str]]]:
    """Los nombres de una cabecera que ocupa DOS filas (`fh` y la siguiente): «grupo subnombre». Los usan `convertir` y los avisos
    de `inspeccionar` para esa cabecera (la misma función, para que lo que se avisa sea lo que se hace)."""
    def celda(r: int, j: int):
        fila = h.filas[r - h.fila0]
        return fila[j] if j < len(fila) else ""
    nombres: list[str] = []
    compuestos: dict[int, tuple[str, str]] = {}
    grupo_actual = None
    for j in range(h.ancho):
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
        nombres.append(nombre_col)
    return nombres, compuestos


def _avisos_de_nombres(h: _HojaLeida, nombres: list[str], primera_de_datos: int, avisos: list) -> None:
    """Nombres repetidos o vacíos con datos debajo (de la fila `primera_de_datos` en adelante)."""
    vistos, problemas = set(), []
    for j, t in enumerate(nombres):
        if not t:
            if any(not _vacia(f[j]) for f in h.filas[max(primera_de_datos - h.fila0, 0):] if j < len(f)):
                problemas.append(f"({letras_de_columna(h.col0 + j)})")
        elif t in vistos:
            problemas.append(t)
        vistos.add(t)
    if problemas:
        avisos.append(declarar("nombres_repetidos", cuantas=len(problemas), nombres=problemas))


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
    """Qué hay en el libro, sin convertir nada. Propone la hoja, la cabecera y los totales.

    Un Parquet (por su FIRMA, no por su extensión) va a `parquet_lector`: no tiene hojas ni cabecera que proponer, y
    enseña sus columnas, las que no entran en el CSV y lo que se hará con las fechas (150-C8)."""
    if es_parquet(datos):
        from .parquet_lector import inspeccionar_parquet  # noqa: PLC0415 — pyarrow, solo si hay un Parquet
        return inspeccionar_parquet(datos, nombre, topes=topes)
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
    pegado = _columna_unica_con_separadores(h)
    if pegado:
        avisos.append(declarar("columna_unica_con_separadores", separador=pegado[0], campos=pegado[1]))
        # Se OFRECE partirla: aquí solo se enseña cómo quedaría. Nada se parte sin que la persona lo acepte en `convertir`.
        entrada["partida_ofrecida"] = _vista_de_la_partida(h, pegado[0])
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
        # Lo mismo que `convertir`: solo las del rango de la hoja leída cuentan como «de la
        # hoja»; las de fuera (no se pueden convertir ni excluir) se declaran aparte.
        recuadro = (h.fila0, h.ultima_fila, h.col0, h.col0 + h.ancho - 1)
        hz = geom.hallazgos(recuadro=recuadro)
        fuera_n, fuera_celdas = geom.formulas_sin_valor_fuera(recuadro)
        if fuera_n:
            avisos.append(declarar("formulas_sin_valor_fuera", cuantas=fuera_n,
                                   celdas=fuera_celdas, rango=entrada["rango"]))
        if hz.formulas_sin_valor:
            avisos.append(declarar("formulas_sin_valor", cuantas=hz.formulas_sin_valor,
                                   celdas=hz.celdas_sin_valor))
        if libro.recalcular_al_abrir and hz.formulas:
            avisos.append(declarar("formulas_a_recalcular", cuantas=hz.formulas,
                                   celdas=hz.celdas_con_formula))
        _avisos_de_hallazgos(hz, letras_de_columna, avisos)
    if cab:
        nombres = _nombres_propuestos(h, cab)
        _avisos_de_nombres(h, nombres, cab + 1, avisos)
        i = cab - h.fila0
        if i + 1 < len(h.filas):
            sig = h.filas[i + 1]
            no_vacias = [v for v in sig if not _vacia(v)]
            if len({v for v in no_vacias if isinstance(v, str)}) >= 2 and all(
                    isinstance(v, str) for v in no_vacias) and any(
                    c for c in h.combinadas if c[0] == cab and c[3] > c[1]):
                avisos.append(declarar("posible_cabecera_de_dos_filas", fila=cab))
        banda = _banda_de_dos_niveles(h, cab)
        if banda is not None:
            # Se OFRECE (componer o quitar la banda); nada se hace sin que la persona lo acepte en `convertir`.
            ofrecida = _vista_de_la_banda(h, cab, banda)
            if ofrecida["columnas_compuestas"]:
                avisos.append(declarar("posible_cabecera_de_dos_niveles", fila=banda["fila"], posicion=banda["posicion"],
                                       cabecera=cab, grupos=ofrecida["grupos"][:3]))
                entrada["cabecera_de_dos_niveles_ofrecida"] = ofrecida
        _avisos_de_columnas(h, cab + 1, avisos)
        # Si la persona marca que la cabecera ocupa DOS filas, los avisos que dependen de la cabecera y de dónde empiezan los
        # datos (nombres repetidos, tipos mezclados, serie de fechas) son otros: los dice el núcleo, la pantalla no los compone.
        if cab + 1 < h.ultima_fila:
            dos: list = []
            nombres2, _c = _nombres_de_dos_filas(h, cab)
            _avisos_de_nombres(h, nombres2, cab + 2, dos)
            _avisos_de_columnas(h, cab + 2, dos)
            entrada["avisos_con_cabecera_de_dos_filas"] = dos
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
            raise rechazar("lector_fallo", tipo="celdas combinadas que se solapan", tipo_en="overlapping merged cells")
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
def convertir(datos: bytes, nombre: str = "", *, hoja=None, fila_de_cabecera: int | None = None,
              filas_de_cabecera: int = 1, filas_a_excluir=(), rellenar_combinadas: bool = False,
              partir_columna_unica: str | None = None,
              componer_cabecera_de_dos_niveles: bool = False,
              topes: Topes | None = None) -> TablaConvertida:
    """El CSV canónico de una hoja, con lo que la persona confirmó y el `origen` que lo declara.

    `fila_de_cabecera` es el número de fila de la hoja (el que enseña Excel). `filas_a_excluir`
    son números de fila de la hoja, y SOLO se quitan esas.

    `partir_columna_unica` (150-C7): `","`, `";"` o `"\\t"`. Solo lo manda quien ACEPTÓ la oferta que hace `inspeccionar` cuando la
    hoja es un CSV pegado en una columna; parte cada celda con un lector CSV y lo declara. `None` (lo normal): nada se parte.

    `componer_cabecera_de_dos_niveles` (150-C7, decisión de Roberto del 09-11): solo lo manda quien ACEPTÓ la oferta de `inspeccionar`
    (`cabecera_de_dos_niveles_ofrecida`): los nombres bajo cada grupo de la banda se escriben «grupo · nombre» y, si la banda va
    debajo de la cabecera, esa fila deja de ser un dato. Se declara. `False` (lo normal): nada se compone.

    Un Parquet (150-C8) va a `parquet_lector.convertir_parquet`: sin hojas ni cabecera, así que `fila_de_cabecera` puede omitirse
    (con un libro de Excel es obligatoria) y lo demás de Excel se rechaza si se pide.
    """
    if es_parquet(datos):
        from .parquet_lector import convertir_parquet  # noqa: PLC0415
        return convertir_parquet(datos, nombre, hoja=hoja, fila_de_cabecera=fila_de_cabecera, filas_de_cabecera=filas_de_cabecera,
                                 filas_a_excluir=filas_a_excluir, rellenar_combinadas=rellenar_combinadas,
                                 partir_columna_unica=partir_columna_unica,
                                 componer_cabecera_de_dos_niveles=componer_cabecera_de_dos_niveles, topes=topes)
    if not isinstance(componer_cabecera_de_dos_niveles, bool):
        raise rechazar("parametro_invalido", parametro="componer_cabecera_de_dos_niveles debe ser verdadero o falso",
                       parametro_en="componer_cabecera_de_dos_niveles must be true or false")
    if partir_columna_unica is not None and (not isinstance(partir_columna_unica, str)
                                              or partir_columna_unica not in _CLAVE_DE):
        raise rechazar("parametro_invalido", parametro="partir_columna_unica debe ser «,», «;» o un tabulador",
                       parametro_en="partir_columna_unica must be “,”, “;” or a tab")
    if isinstance(fila_de_cabecera, bool) or not isinstance(fila_de_cabecera, int) or fila_de_cabecera < 1:
        raise rechazar("parametro_invalido", parametro="fila_de_cabecera debe ser un entero ≥ 1",
                       parametro_en="fila_de_cabecera must be an integer ≥ 1")
    if filas_de_cabecera not in (1, 2) or isinstance(filas_de_cabecera, bool):
        raise rechazar("parametro_invalido", parametro="filas_de_cabecera debe ser 1 o 2",
                       parametro_en="filas_de_cabecera must be 1 or 2")
    try:
        lista_excluir = list(filas_a_excluir)
    except TypeError:
        raise rechazar("parametro_invalido", parametro="filas_a_excluir debe ser una lista de números de fila",
                       parametro_en="filas_a_excluir must be a list of row numbers") from None
    # Solo enteros: `2.5` o `"3"` se convertían con `int()` y se quitaba OTRA fila (o una que no
    # se pidió así), declarando «se quitó 1 fila» (menor de la 2.ª re-auditoría).
    if any(isinstance(f, bool) or not isinstance(f, int) for f in lista_excluir):
        raise rechazar("parametro_invalido", parametro="filas_a_excluir debe ser una lista de números de fila enteros",
                       parametro_en="filas_a_excluir must be a list of integer row numbers")
    excluir = sorted(set(lista_excluir))
    if any(f < 1 or f > _MAX_FILAS_FISICAS for f in excluir):
        raise rechazar("parametro_invalido", parametro="filas_a_excluir debe ser una lista de números de fila",
                       parametro_en="filas_a_excluir must be a list of row numbers")

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
                       parametro=f"filas_a_excluir [{lista_para_texto([f for f in excluir if f < fd])}] está en la cabecera o por encima",
                       parametro_en=f"filas_a_excluir [{lista_para_texto([f for f in excluir if f < fd], 'en')}] is in the header or above it")
    if any(f > fin for f in excluir):
        # Una fila que la hoja no tiene no se «quita»: declararla quitada sería mentir.
        raise rechazar("parametro_invalido",
                       parametro=f"filas_a_excluir [{lista_para_texto([f for f in excluir if f > fin])}] no existe en la hoja (la última fila es la {fin})",
                       parametro_en=f"filas_a_excluir [{lista_para_texto([f for f in excluir if f > fin], 'en')}] does not exist in the sheet (the last row is {fin})")
    excluidas = set(excluir)

    # Fórmulas sin valor: se comprueban sobre lo que SE CONVIERTE, cabecera incluida (la fila
    # TOTAL que la persona quita no rechaza). Nunca se calcula una fórmula.
    geom = h.geom
    # SOLO las del rectángulo que se convierte (filas `fh..fin` y las columnas leídas): una
    # fórmula sin valor fuera (una fila de total suelta tras una fila vacía, una columna a la
    # derecha) no se puede excluir —no está en la hoja que se lee— y rechazar el libro por ella
    # iría contra «celda del rango convertido» (DA3). Se DECLARA, con su celda (3.ª auditoría, I1).
    recuadro = (fh, fin, h.col0, h.col0 + h.ancho - 1)
    formulas_fuera: tuple[int, list[str]] = (0, [])
    if geom is not None and geom.hay_formulas:
        formulas_fuera = geom.formulas_sin_valor_fuera(recuadro)
        hzf = geom.hallazgos(lambda r: r not in excluidas, lambda r: False, recuadro)
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
    dias_cero = geom.dias_cero_1904() if geom is not None else set()

    partida = None
    if partir_columna_unica is not None:
        if h.ancho != 1:
            raise rechazar("parametro_invalido",
                           parametro="partir_columna_unica solo vale para una hoja de UNA columna",
                           parametro_en="partir_columna_unica only applies to a single-column sheet")
        # Auditoría del 150-C7 (M6): solo se parte lo que `inspeccionar` OFRECE, con su separador (la misma función decide).
        # Sin esto, la API partía una columna que no es un CSV pegado (filas con distinto número de campos, sin oferta).
        ofrecida = _columna_unica_con_separadores(h)
        if ofrecida is None or _CARACTER_DE[ofrecida[0]] != partir_columna_unica:
            raise rechazar("parametro_invalido",
                           parametro="la hoja no es un CSV pegado con ese separador (no se ofrece partirla así): no se parte",
                           parametro_en="the sheet is not a CSV pasted with that separator (splitting it that way is not offered): "
                                        "it is not split")
        # Se parte DESPUÉS de las comprobaciones de fórmulas y fechas (son de la columna A original). Las celdas partidas ya
        # no son celdas de Excel: sus hallazgos por columna (errores, %, columnas ocultas) no se miran; los de FILA, sí.
        sep = partir_columna_unica
        # La primera línea del CSV pegado (la celda de su cabecera, tal cual): con ella se decide, como con un `.csv`, si es un CSV
        # `;` al que se aplica la regla del decimal (re-auditoría del 150-C7, M-C: LA señal del C2, no «se ofreció por `;`»).
        celda_de_la_cabecera = h.filas[fh - h.fila0][0] if h.filas[fh - h.fila0] else ""
        texto_de_la_cabecera = celda_de_la_cabecera if isinstance(celda_de_la_cabecera, str) else valor_a_texto(celda_de_la_cabecera)[0]
        filas_partidas = _partir_filas(h, sep)
        partida = {"separador": _CLAVE_DE[sep], "campos": max((len(f) for f in filas_partidas), default=0),
                   "primera_linea": (texto_de_la_cabecera.splitlines() or [""])[0]}
        anuladas = set()
        dias_cero = set()
        h = _HojaLeida(nombre=h.nombre, fila0=h.fila0, col0=h.col0, filas=filas_partidas, combinadas=[],
                       geom=h.geom, ancho_del_rango=partida["campos"])
        if h.ancho < 2:
            raise rechazar("parametro_invalido",
                           parametro="la columna no tiene el separador pedido: no hay nada que partir",
                           parametro_en="the column does not contain the requested separator: nothing to split")
    banda = None
    if componer_cabecera_de_dos_niveles:
        if partida is not None or nh != 1:
            raise rechazar("parametro_invalido",
                           parametro="componer_cabecera_de_dos_niveles no se junta con partir la columna ni con una cabecera de dos filas",
                           parametro_en="componer_cabecera_de_dos_niveles cannot be combined with splitting the column or a two-row header")
        # La MISMA función que hace la oferta en `inspeccionar`: solo se compone lo que se ofrece para ESTA cabecera.
        banda = _banda_de_dos_niveles(h, fh)
        if banda is None or not _componer_con_la_banda(_nombres_propuestos(h, fh), banda, h.col0):
            raise rechazar("parametro_invalido",
                           parametro=f"no hay una banda de dos niveles junto a la cabecera de la fila {fh}: no se compone nada",
                           parametro_en=f"there is no two-level band next to the header in row {fh}: nothing is composed")
        if banda["posicion"] == "debajo":
            # La banda va entre los nombres y los datos: es parte de la cabecera, no un dato (se declara con lo compuesto).
            excluidas = excluidas | {banda["fila"]}
    excluir_combinadas = sorted(excluidas)
    ancho = h.ancho
    col0 = h.col0
    vacias_por_columna, rangos_vacias, rellenadas_por_columna, rangos_rellenadas = \
        _combinadas_en_lo_convertido(h, fd, fin, excluir_combinadas, rellenar_combinadas)

    # ── cabecera ──
    def celda(r: int, j: int):
        fila = h.filas[r - h.fila0]
        return fila[j] if j < len(fila) else ""
    nombres_crudos: list[str] = []
    compuestos: dict[int, tuple[str, str]] = {}
    if nh == 1:
        nombres_crudos = [_texto(celda(fh, j)) for j in range(ancho)]
    else:
        nombres_crudos, compuestos = _nombres_de_dos_filas(h, fh)
    nombres_sin_banda = list(nombres_crudos)
    compuestos_por_la_banda: dict[int, str] = {}
    if banda is not None:
        compuestos_por_la_banda = _componer_con_la_banda(nombres_crudos, banda, col0)
        for j, n in compuestos_por_la_banda.items():
            nombres_crudos[j] = n

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
            if dias_cero and type(v) is _dt.time and (r, col0 + j) in dias_cero:
                # 1904-01-01 (con su hora): calamine dio solo la hora
                v = (_dt.date(1904, 1, 1) if v == _dt.time(0) else
                     _dt.datetime.combine(_dt.date(1904, 1, 1), v))
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
        elif len(conserva) == 1 and ";" in t and "," not in t:
            # Remate de 150-C4: con UNA sola columna, un `;` en su nombre deja la cabecera del CSV sin comas y con un `;`, que es
            # justo la señal con la que la entrada canónica (150-C2) reconoce un CSV `;` y le reescribe la cabecera: el análisis
            # y el estudio leerían «peso,altura» sobre filas de una sola columna. El CSV de la conversión tiene que ser un punto
            # fijo de esa entrada, así que el `;` se escribe `_` y se declara. (Con varias columnas hay comas: no pasa.)
            finales[j] = t.replace(";", "_")
            declaraciones.append(declarar("punto_y_coma_en_el_nombre", columna=finales[j], original=t))
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
        if nh == 1 and crudo_texto and crudo_texto != nombres_sin_banda[j]:
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
        if partida is None:
            hz = geom.hallazgos(None, lambda r: r >= fd and r not in excluidas and r not in quitadas_vacias)
            _avisos_de_hallazgos(hz, lambda c: finales.get(c - col0), declaraciones)
        ocultas = geom.filas_ocultas_en(fd, fin, excluidas | quitadas_vacias)
        if ocultas:
            declaraciones.append(declarar("filas_ocultas", cuantas=len(ocultas), filas=ocultas[:10]))
        col_ocultas = [col0 + j for j in conserva if geom.columna_oculta(col0 + j)] if partida is None else []
        if col_ocultas:
            declaraciones.append(declarar("columnas_ocultas", cuantas=len(col_ocultas),
                                          letras=[letras_de_columna(c) for c in col_ocultas[:10]]))
    if formulas_fuera[0]:
        declaraciones.append(declarar("formulas_sin_valor_fuera", cuantas=formulas_fuera[0],
                                      celdas=formulas_fuera[1], rango=_rango_a1(fh, h.col0, fin, h.col0 + h.ancho - 1)))
    if banda is not None:
        ejemplos = [finales[j] for j in conserva if j in compuestos_por_la_banda][:3]
        declaraciones.append(declarar("cabecera_de_dos_niveles_compuesta",
                                      cuantas=sum(1 for j in conserva if j in compuestos_por_la_banda),
                                      fila=banda["fila"], posicion=banda["posicion"], ejemplos=ejemplos))
    if partida is not None:
        declaraciones.append(declarar("columna_unica_partida", cuantas=len(filas_csv) + 1,
                                      separador=partida["separador"], campos=partida["campos"]))
    if libro.formato == "xls":
        declaraciones.append(declarar("no_comprobable_en_xls", que=_NO_COMPROBABLE_EN_XLS))
    if libro.macros:
        declaraciones.append(declarar("macros_ignoradas"))
    if meta["estado"] != "visible":
        declaraciones.append(declarar("hoja_oculta", hoja=meta["nombre"]))
    previas = [r for r in range(h.fila0, fh) if any(not _vacia(v) for v in h.filas[r - h.fila0])
              and not (banda is not None and r == banda["fila"])]
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
    if banda is not None and banda["posicion"] == "debajo" and banda["fila"] not in excluir:
        # Re-auditoría del 150-C7 (M-B): al componer con la banda DEBAJO, su fila sale de los datos; quien lea `filas_excluidas` (el
        # informe de «¿Funciona aquí?», el recibo) tiene que verla, con su MOTIVO (no la eligió la persona como fila a quitar: es
        # parte de la cabecera compuesta). Las que eligió la persona siguen siendo `{fila, texto}`, como en el contrato.
        filas_excluidas.append({"fila": banda["fila"], "motivo": "banda_de_dos_niveles",
                                "texto": next((_texto(v) for v in h.filas[banda["fila"] - h.fila0] if not _vacia(v)), "")})
        filas_excluidas.sort(key=lambda e: e["fila"])
    if excluir:
        declaraciones.append(declarar("filas_excluidas", cuantas=len(excluir), filas=excluir))
    if filas_vacias:
        declaraciones.append(declarar("filas_vacias_quitadas", cuantas=len(filas_vacias),
                                      filas=filas_vacias[:10]))
    if partida is not None:
        # Auditoría del 150-C7 (M5): una fila partida con OTRO número de campos que la cabecera (un campo entre comillas con un
        # salto de línea queda en dos filas de Excel; una fila corta se completa con vacías; una larga añade columnas) se DICE.
        campos_cab = len(h.filas[fh - h.fila0])
        desiguales = [r for r in range(fd, fin + 1) if r not in excluidas and h.filas[r - h.fila0]
                      and len(h.filas[r - h.fila0]) != campos_cab]
        if desiguales:
            declaraciones.append(declarar("columna_unica_filas_desiguales", cuantas=len(desiguales),
                                          filas=desiguales[:10], campos=campos_cab))

    # ── el CSV ──
    if len(conserva) < ancho:
        filas_csv = [[f[j] for j in conserva] for f in filas_csv]
    if partida is not None and partida["separador"] == "punto_y_coma":
        # Auditoría del 150-C7 (I1): un CSV `;` pegado y partido llega IGUAL que su `.csv`: el decimal coma (`18,72`) pasa a punto
        # con LA regla del C2 (`decimal_es`: por columna, todo o nada, y declarado), la misma que aplica `normalize_csv_text` a un
        # `.csv` `;`, recorrida por el MISMO código (`decimal_de_un_csv_punto_y_coma`) y con la MISMA señal (la primera línea sin
        # `,` y con `;`; re-auditoría, M-C). Aquí, y no en la entrada canónica: el CSV de una conversión pasa por ella tal cual (es su
        # punto fijo). Con `,` o tabulador no se toca nada, como en el `.csv` (en inglés `1,234` son miles).
        from matrixai.training.decimal_es import (  # noqa: PLC0415
            decimal_de_un_csv_punto_y_coma, es_la_cabecera_de_un_csv_punto_y_coma)
        if es_la_cabecera_de_un_csv_punto_y_coma(partida["primera_linea"]):
            convertidas, del_decimal = decimal_de_un_csv_punto_y_coma([finales[j] for j in conserva], lambda: filas_csv)
            filas_csv = list(convertidas)
            declaraciones.extend(del_decimal)
    csv_bytes = escribir_csv([finales[j] for j in conserva], filas_csv)

    primera_col, ultima_col = col0 + conserva[0], col0 + conserva[-1]
    # Remate de 150-C4 (menor 2): el nombre y la hoja que el origen CITA pasan por la MISMA regla que el estudio aplica al
    # recibirlo (`matrixai.nombre_citable`): sin ella, un nombre con un tabulador o un U+202A, o sin nombre, daba un
    # origen que `estudio/iniciar` rechazaba entero. Se QUITA lo que no cabe y se declara; un nombre vacío es una
    # ausencia (no hay clave), no un `""`.
    nombre_citado, quitados_nombre = nombre_citable.saneado(libro.nombre)
    hoja_citada, quitados_hoja = nombre_citable.saneado(meta["nombre"])
    for que, quitados in (("nombre", quitados_nombre), ("hoja", quitados_hoja)):
        if quitados:
            declaraciones.append(declarar("texto_del_origen_saneado", cuantas=quitados, que=que))
    origen = {
        "esquema": ESQUEMA_DE_ORIGEN,
        "formato": libro.formato,
        **({"nombre": nombre_citado} if nombre_citado is not None else {}),
        "sha256": hashlib.sha256(libro.datos).hexdigest(),
        "bytes": len(libro.datos),
        **({"hoja": hoja_citada} if hoja_citada is not None else {}),
        "fila_de_cabecera": fh,
        "filas_de_cabecera": nh,
        # Partida: el rango es el de la columna de la HOJA que se leyó (las columnas del CSV ya no son columnas de Excel).
        "rango": _rango_a1(fh, col0, fin, col0) if partida is not None else _rango_a1(fh, primera_col, fin, ultima_col),
        "filas_excluidas": filas_excluidas,
        "conversor": libro.conversor(),
        "declaraciones": declaraciones,
        # 150-C4 (I1): la huella del CSV que ESTA conversión devuelve. Ata el origen a su CSV: el estudio la compara
        # con el texto que recibe y rechaza un origen que no es de él (sin esto, el origen de un libro se podía
        # adjuntar a cualquier CSV y la ficha lo afirmaba).
        "sha256_del_csv": hashlib.sha256(csv_bytes).hexdigest(),
    }
    return TablaConvertida(csv_bytes=csv_bytes, origen=origen)
