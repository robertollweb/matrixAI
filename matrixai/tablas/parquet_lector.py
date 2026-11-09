# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Abrir un fichero Parquet como tabla (150-C8): de sus bytes al CSV canónico + `origen`.

Sigue el patrón de Excel (C1/C3), con lo que cambia en un formato de COLUMNAS TIPADAS:

* **`pyarrow` es un extra opcional** (`matrixai-core[parquet]`), importado DENTRO de la función. Sin él, este módulo se
  importa y un Parquet dice «instala pyarrow» (`falta_pyarrow`); Excel y CSV siguen funcionando (invariante 7).
* **Las guardas corren ANTES de leer un dato** (invariante 6) y con los METADATOS del pie del fichero: filas, columnas y
  bytes descomprimidos. La memoria se pregunta a la guarda que ya existe (`limits.memoria_insuficiente`), no a una segunda.
* **Sin pandas.** Un `int64` con nulos pasaría por `float64` y perdería precisión por encima de 2^53; aquí cada valor llega
  como `int` de Python. La lectura es por lotes (`ParquetFile.iter_batches`), y solo de las columnas que se convierten.
* **El mismo escritor de CSV propio** (`valores.escribir_csv`): mismo fichero → mismos bytes.
* **Nada se quita, se convierte ni se redondea sin declararlo** (invariante 3), con el texto es/en redactado aquí:
  zona horaria → UTC, `ns`/`us`/`ms` → segundos (con cuántos valores perdieron fracción), `NaN`/infinito → vacío (contados),
  y listas, estructuras, mapas y binarios → EXCLUIDOS por nombre y tipo, con la lista visible en `inspeccionar` antes de seguir.
* Un valor ausente no es un cero: `null` y `NaN` salen vacíos, jamás `0`.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import struct

from matrixai import nombre_citable

from .. import limits
from .declaraciones import declarar
from .errores import TablaRechazada, rechazar
from .formato import FIRMA_PARQUET, FIRMA_PARQUET_CIFRADO
from . import topes as _topes
from .topes import TOPES_DE_FABRICA, Topes

__all__ = ["es_parquet", "pyarrow_disponible", "inspeccionar_parquet", "convertir_parquet",
           "FACTOR_RAM_POR_BYTE_DESCOMPRIMIDO_DE_PARQUET", "BYTES_POR_CELDA_DE_PARQUET", "TAMANO_DE_LOTE"]

#: Filas por lote de `iter_batches`: el pico de memoria de la lectura no depende del tamaño del fichero.
TAMANO_DE_LOTE = 65_536

# LOS FACTORES ESTÁN MEDIDOS, no supuestos (2026-11-09, esta máquina, Python 3.12, pyarrow 26.0.1, pico de RSS de `convertir`
# en un proceso limpio menos el RSS al empezar; sondas en `~/.agentes/150/c8/sondas/memoria.py`, zstd, 12 filas de lote 65.536):
#
#   300.000 x 30 float64     80,2 MB descomprimidos  CSV 173 MB  -> +546 MB   (x6,8 los descomprimidos; 60,7 B/celda)
#   300.000 x 30 int64       80,2 MB                 CSV  89 MB  -> +451 MB   (x5,6; 50,1 B/celda)
#   300.000 x 30 mixto       55,7 MB                 CSV 105 MB  -> +427 MB   (x7,7; 47,5 B/celda)
#   200.000 x 10 textos      3,4 MB (!)              CSV  53 MB  -> +225 MB   (x67; 112 B/celda)
#
# Los metadatos NO bastan por sí solos: una columna de textos con diccionario declara muy pocos bytes descomprimidos y su
# CSV es 16 veces mayor (x67). Como en la guarda de Excel (`topes.py`), hay dos cotas y se toma la MAYOR, cada una el MÍNIMO
# medido redondeado hacia abajo (solo se rechaza lo que no cabría NI EN EL MEJOR CASO): por celda (las filas y columnas que
# declara el pie) y por byte descomprimido.
BYTES_POR_CELDA_DE_PARQUET = 40
FACTOR_RAM_POR_BYTE_DESCOMPRIMIDO_DE_PARQUET = 5

_EPOCA = _dt.datetime(1970, 1, 1)
#: La unidad en que se LEE un INT96 (ver `_Fichero.__init__`); lo que GUARDA es nanosegundos.
_UNIDAD_DE_INT96 = "us"
_POR_SEGUNDO = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
_UNIDAD_LEGIBLE = {"s": "segundos", "ms": "milisegundos", "us": "microsegundos", "ns": "nanosegundos"}


# ── Detección y disponibilidad ──────────────────────────────────────────────────────────
def es_parquet(datos: bytes) -> bool:
    """¿Empieza como un Parquet (`PAR1`) o como uno de pie cifrado (`PARE`, y acaba igual)?"""
    cabeza = bytes(datos[:4])
    return cabeza == FIRMA_PARQUET or (cabeza == FIRMA_PARQUET_CIFRADO and bytes(datos[-4:]) == FIRMA_PARQUET_CIFRADO)


def pyarrow_disponible() -> bool:
    """¿Se podría importar `pyarrow`? Sin importarlo (pesa +66 MB de RSS)."""
    try:
        return importlib.util.find_spec("pyarrow") is not None
    except (ImportError, ValueError):
        return False


def _importar_pyarrow():
    if not pyarrow_disponible():
        raise rechazar("falta_pyarrow")
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except (ImportError, OSError, MemoryError) as e:
        # Está en el disco (`find_spec` lo encontró) y no carga: decir «instala pyarrow» sería falso y no arreglaría nada.
        raise rechazar("pyarrow_no_carga", tipo=type(e).__name__) from None
    return pa, pq


def _version_de_pyarrow() -> str | None:
    try:
        from importlib import metadata
        return metadata.version("pyarrow")
    except Exception:
        return None


def _conversor() -> dict:
    import matrixai
    return {"nombre": "matrixai.tablas",
            "versiones": {"matrixai-core": getattr(matrixai, "__version__", None), "pyarrow": _version_de_pyarrow()}}


# ── El tipo de cada columna ─────────────────────────────────────────────────────────────
def _categoria_no_convertible(pa, t) -> str | None:
    """`None` si el tipo cabe en un CSV; si no, su categoría (clave estable del detalle de la declaración)."""
    ty = pa.types
    if ty.is_dictionary(t):
        return _categoria_no_convertible(pa, t.value_type)
    storage = getattr(t, "storage_type", None)          # un tipo de extensión se juzga por su almacenamiento
    if storage is not None and storage != t:
        return _categoria_no_convertible(pa, storage)
    if ty.is_map(t):
        return "mapa"
    if ty.is_struct(t):
        return "estructura"
    for nombre in ("is_list", "is_large_list", "is_fixed_size_list", "is_list_view", "is_large_list_view"):
        f = getattr(ty, nombre, None)
        if f is not None and f(t):
            return "lista"
    for nombre in ("is_binary", "is_large_binary", "is_fixed_size_binary", "is_binary_view"):
        f = getattr(ty, nombre, None)
        if f is not None and f(t):
            return "binario"
    if ty.is_duration(t):
        return "duracion"
    if ty.is_union(t):
        return "union"
    if getattr(ty, "is_interval", None) is not None and ty.is_interval(t):
        return "intervalo"
    escalar = (ty.is_null(t) or ty.is_boolean(t) or ty.is_integer(t) or ty.is_floating(t) or ty.is_decimal(t)
               or ty.is_string(t) or ty.is_large_string(t) or ty.is_date(t) or ty.is_time(t) or ty.is_timestamp(t)
               or (getattr(ty, "is_string_view", None) is not None and ty.is_string_view(t)))
    return None if escalar else "otro"


def _valor_del_tipo(pa, t):
    return _valor_del_tipo(pa, t.value_type) if pa.types.is_dictionary(t) else t


# ── Abrir con las guardas ANTES de leer un dato ─────────────────────────────────────────
class _Fichero:
    def __init__(self, datos: bytes, nombre: str, topes: Topes):
        from .api import _nombre_corto  # noqa: PLC0415 — api importa este módulo dentro de sus funciones
        self.datos = bytes(datos)
        self.nombre = _nombre_corto(nombre)
        self.topes = topes
        if self.datos[:4] == FIRMA_PARQUET_CIFRADO:
            raise rechazar("parquet_cifrado")
        if len(self.datos) < 12 or self.datos[-4:] != FIRMA_PARQUET:
            # Empieza por PAR1 y no acaba por PAR1: una descarga cortada (el pie, que es lo que dice dónde está todo, va al final).
            raise rechazar("parquet_danado", tipo="no acaba con la marca PAR1: el fichero está cortado")
        largo_pie = int.from_bytes(self.datos[-8:-4], "little")
        if largo_pie <= 0 or largo_pie > len(self.datos) - 12:
            raise rechazar("parquet_danado", tipo="el pie del fichero declara un tamaño imposible")
        self.pa, self.pq = _importar_pyarrow()
        try:
            # INT96 (lo que escriben Spark, Hive e Impala) se lee en MICROSEGUNDOS, no en ns (lo que pyarrow hace por omisión):
            # en ns un `int64` solo llega de 1677 a 2262 y lo que se sale DA LA VUELTA sin error (BLOQUEANTE-1 de la auditoría
            # de 150-C8: 9999-12-31, el «sin fin» de una tabla de almacén, salía 1816-03-29). Con `us` cubre los años 1-9999.
            self.pf = self.pq.ParquetFile(self.pa.BufferReader(self.pa.py_buffer(self.datos)),
                                          coerce_int96_timestamp_unit=_UNIDAD_DE_INT96)
            self.meta = self.pf.metadata
            self.esquema = self.pf.schema_arrow
            # Pero INT96 GUARDA nanosegundos: leído en `us`, una fracción de menos de 1 us se perdería sin contarse. Las
            # columnas INT96 son las que cambian de tipo al abrir el MISMO fichero sin forzar la unidad; de esas, un segundo
            # lector en ns da solo el resto de nanosegundos (ver `_Columna._instantes`). Solo si hay un `timestamp[us]`.
            self.pf_ns = None
            self.int96: frozenset[int] = frozenset()
            if any(self.pa.types.is_timestamp(c.type) and c.type.unit == _UNIDAD_DE_INT96 for c in self.esquema):
                pf_ns = self.pq.ParquetFile(self.pa.BufferReader(self.pa.py_buffer(self.datos)))
                self.int96 = frozenset(i for i, (a, b) in enumerate(zip(self.esquema, pf_ns.schema_arrow))
                                       if a.type != b.type)
                if self.int96:
                    self.pf_ns = pf_ns
            self.filas = int(self.meta.num_rows)
            self.grupos = int(self.meta.num_row_groups)
            # Lo que DECLARAN los metadatos: la suma de lo que ocupa cada grupo descomprimido y, por columna, lo que dice cada
            # fragmento; se toma lo mayor (un pie que miente en uno de los dos no esconde la bomba).
            por_grupo = 0
            por_columna = 0
            for g in range(self.grupos):
                rg = self.meta.row_group(g)
                por_grupo += int(rg.total_byte_size)
                por_columna += sum(int(rg.column(c).total_uncompressed_size) for c in range(rg.num_columns))
            self.descomprimido = max(por_grupo, por_columna)
            self.creado_por = str(self.meta.created_by or "") or None
        except TablaRechazada:
            raise
        except MemoryError:
            raise rechazar("lector_sin_memoria", variante="parquet") from None
        except Exception as e:  # noqa: BLE001 — ninguna excepción de la biblioteca llega a la persona (invariante 10)
            raise self._rechazo_de(e) from None
        self.comprobar_topes()

    def _rechazo_de(self, e: BaseException) -> TablaRechazada:
        pa = self.pa
        texto = str(e).lower()
        if isinstance(e, pa.ArrowMemoryError) or isinstance(e, MemoryError):
            return rechazar("lector_sin_memoria", variante="parquet")
        if "encrypt" in texto or "decrypt" in texto or "footer" in texto and "key" in texto:
            return rechazar("parquet_cifrado")
        if isinstance(e, (pa.ArrowNotImplementedError, NotImplementedError)):
            return rechazar("parquet_no_legible", tipo=type(e).__name__)
        return rechazar("parquet_danado", tipo=type(e).__name__)

    def comprobar_topes(self) -> None:
        t = self.topes
        if self.filas > t.max_filas_parquet:
            raise rechazar("demasiadas_filas", variante="parquet", filas=self.filas, tope=t.max_filas_parquet)
        n_cols = len(self.esquema)
        if n_cols > t.max_columnas:
            raise rechazar("demasiadas_columnas", variante="parquet", columnas=n_cols, tope=t.max_columnas)
        if self.descomprimido > t.max_descomprimido:
            raise rechazar("bomba", variante="parquet", descomprimido=limits.human_bytes(self.descomprimido),
                           original=limits.human_bytes(len(self.datos)), tope=limits.human_bytes(t.max_descomprimido),
                           bytes_descomprimidos=self.descomprimido)
        if self.filas == 0 or n_cols == 0:
            raise rechazar("parquet_vacio")

    def memoria_necesaria(self, columnas_que_se_convierten: int) -> int:
        return max(self.filas * int(columnas_que_se_convierten) * BYTES_POR_CELDA_DE_PARQUET,
                   self.descomprimido * FACTOR_RAM_POR_BYTE_DESCOMPRIMIDO_DE_PARQUET)

    def comprobar_memoria(self, columnas_que_se_convierten: int) -> None:
        """La guarda de memoria de Excel (`topes.comprobar_memoria`: la de los CSV con los bytes «equivalentes» y el techo
        propio del proceso), no una copia de su rama (M7 de la auditoría de 150-C8). Con dos diferencias: los motivos son
        los de Parquet, y el techo se compara con TODO lo necesario, porque aquí lee y escribe el MISMO proceso (no hay un
        hijo de lectura aparte que se lleve su parte)."""
        _topes.comprobar_memoria(self.memoria_necesaria(columnas_que_se_convierten), "", variante="parquet",
                                fraccion_del_proceso=1)

    def columnas(self) -> list[dict]:
        """Por columna: nombre, tipo (texto de Arrow) y, si no cabe en un CSV, su categoría."""
        pa = self.pa
        salida = []
        for i, campo in enumerate(self.esquema):
            salida.append({"posicion": i + 1, "nombre": campo.name, "tipo": str(campo.type),
                           "categoria": _categoria_no_convertible(pa, campo.type), "_tipo": campo.type,
                           "_int96": i in self.int96})
        return salida


def _indices_de_pandas(esquema) -> frozenset[str]:
    """Los nombres de las columnas que son el ÍNDICE de filas de pandas (`index_columns` de sus metadatos, los que son
    columna: un `RangeIndex` va como diccionario y no ocupa ninguna). Metadatos ilegibles: ninguno (no se adivina)."""
    import json  # noqa: PLC0415
    try:
        crudo = (esquema.metadata or {}).get(b"pandas")
        indices = json.loads(crudo)["index_columns"] if crudo else []
        return frozenset(x for x in indices if isinstance(x, str))
    except (ValueError, TypeError, KeyError, AttributeError):
        return frozenset()


# ── inspeccionar: sin leer un dato ──────────────────────────────────────────────────────
def _zona_y_unidad(pa, t, int96: bool = False):
    """La zona y la unidad que GUARDA la columna. Un `time` también se escribe en segundos (IMPORTANTE-1 / M4 de la
    auditoría de 150-C8: `convertir` lo declaraba y `inspeccionar` no lo avisaba), y un INT96 guarda ns aunque se lea en us."""
    t = _valor_del_tipo(pa, t)
    if pa.types.is_timestamp(t):
        return t.tz, ("ns" if int96 else t.unit)
    if pa.types.is_time(t):
        return None, t.unit
    return None, None


def inspeccionar_parquet(datos: bytes, nombre: str = "", *, topes: Topes | None = None) -> dict:
    """Qué hay en el fichero, SIN leer ni un dato (solo el pie): filas, columnas con su tipo, las que NO entran en el CSV
    (por nombre y tipo, para que la persona las vea ANTES de seguir) y lo que la conversión hará con las fechas."""
    f = _Fichero(datos, nombre, topes or TOPES_DE_FABRICA)
    columnas = f.columnas()
    avisos: list[dict] = []
    vista_columnas = []
    for c in columnas:
        entra = c["categoria"] is None
        vista_columnas.append({"posicion": c["posicion"], "nombre": c["nombre"], "tipo": c["tipo"], "se_convierte": entra,
                               **({"categoria": c["categoria"]} if c["categoria"] else {})})
        if not entra:
            avisos.append(declarar("parquet_columna_excluida", columna=c["nombre"], tipo_arrow=c["tipo"], categoria=c["categoria"]))
            continue
        zona, unidad = _zona_y_unidad(f.pa, c["_tipo"], c["_int96"])
        if zona:
            avisos.append(declarar("parquet_conversion_prevista", columna=c["nombre"], que="zona_horaria_a_utc", zona=zona))
        if unidad in ("ms", "us", "ns"):
            avisos.append(declarar("parquet_conversion_prevista", columna=c["nombre"], que="unidad_a_segundos",
                                   unidad=unidad))
    excluidas = [c for c in vista_columnas if not c["se_convierte"]]
    if len(excluidas) == len(vista_columnas):
        raise rechazar("parquet_sin_columnas_convertibles", columnas=len(vista_columnas))
    return {
        "formato": "parquet",
        "nombre": f.nombre,
        "bytes": len(f.datos),
        "sha256": hashlib.sha256(f.datos).hexdigest(),
        "filas": f.filas,
        "grupos_de_filas": f.grupos,
        "bytes_descomprimidos": f.descomprimido,
        "creado_por": f.creado_por,
        "columnas": vista_columnas,
        "columnas_excluidas": excluidas,
        # Parquet no tiene hojas ni cabecera que elegir: estos campos existen para que quien pinta el libro de Excel no
        # se rompa, y dicen «nada que elegir».
        "hojas": [],
        "hoja_propuesta": None,
        "avisos": avisos,
        "conversor": _conversor(),
    }


# ── convertir ───────────────────────────────────────────────────────────────────────────
def _texto_de_float(v: float, formato: str) -> str:
    """La representación más corta que vuelve a dar el MISMO número en su ancho (`f` = float32, `e` = float16); sin numpy."""
    for precision in range(1, 18):
        s = f"{v:.{precision}g}"
        try:
            if struct.unpack(formato, struct.pack(formato, float(s)))[0] == v:
                return s[:-2] if s.endswith(".0") else s
        except (OverflowError, struct.error):
            break
    return repr(v)


class _Columna:
    """Cómo se escribe una columna y lo que hay que declarar de ella."""

    def __init__(self, pa, nombre: str, tipo, int96: bool = False):
        self.nombre = nombre
        self.tipo = tipo
        self.pa = pa
        self.no_finitos = 0
        self.con_fraccion = 0
        self.fuera_de_rango = 0
        self.hora_fuera = 0
        self.instantes = 0
        t = _valor_del_tipo(pa, tipo)
        ty = pa.types
        self.zona = None
        self.unidad = None
        if ty.is_timestamp(t):
            self.clase, self.zona, self.unidad = "timestamp", t.tz, t.unit
        elif ty.is_date(t):
            self.clase = "date64" if ty.is_date64(t) else "date32"
        elif ty.is_time(t):
            self.clase, self.unidad = "time", t.unit
        elif ty.is_boolean(t):
            self.clase = "bool"
        elif ty.is_integer(t):
            self.clase = "int"
        elif ty.is_floating(t):
            self.clase = {16: "f16", 32: "f32"}.get(t.bit_width, "f64")
        elif ty.is_decimal(t):
            self.clase = "decimal"
        elif ty.is_null(t):
            self.clase = "null"
        else:
            self.clase = "texto"
        #: INT96 se lee en `us` y GUARDA ns: la unidad que se declara es la guardada, y su resto de ns lo da `arr_ns`.
        self.int96 = bool(int96) and self.clase == "timestamp"
        self.unidad_guardada = "ns" if self.int96 else self.unidad

    def textos(self, arr, arr_ns=None) -> list[str]:
        pa = self.pa
        if pa.types.is_dictionary(arr.type):
            arr = arr.dictionary_decode()
        c = self.clase
        if c == "texto":
            return ["" if v is None else v for v in arr.to_pylist()]
        if c == "int":
            return ["" if v is None else str(v) for v in arr.to_pylist()]
        if c == "bool":
            return ["" if v is None else ("true" if v else "false") for v in arr.to_pylist()]
        if c == "null":
            return [""] * len(arr)
        if c == "decimal":
            return ["" if v is None else format(v, "f") for v in arr.to_pylist()]
        if c in ("f64", "f32", "f16"):
            return self._flotantes(arr, c)
        if c == "timestamp":
            return self._instantes(arr, arr_ns)
        if c in ("date32", "date64"):
            return self._fechas(arr, c)
        return self._horas(arr)

    def _flotantes(self, arr, c: str) -> list[str]:
        salida = []
        for v in arr.to_pylist():
            if v is None:
                salida.append("")
                continue
            v = float(v)
            if v - v != 0.0:                       # NaN e infinito: vacío y contado, jamás 0
                self.no_finitos += 1
                salida.append("")
            elif c == "f64":
                salida.append(str(int(v)) if v.is_integer() and abs(v) < 2 ** 53 else repr(v))
            else:
                s = _texto_de_float(v, "f" if c == "f32" else "e")
                salida.append(s)
        return salida

    def _instantes(self, arr, arr_ns=None) -> list[str]:
        pa = self.pa
        por_s = _POR_SEGUNDO[self.unidad]
        crudos = arr.cast(pa.int64()).to_pylist()           # los instantes tal cual están guardados (UTC si lleva zona)
        # INT96: el MISMO valor leído en ns. Dentro de 1677-2262 es exacto; fuera, pyarrow lo calcula en `int64` y da la
        # vuelta (módulo 2^64), pero el resto de nanosegundos sobre el valor en us sigue siendo `(ns - us*1000) mod 2^64`.
        en_ns = arr_ns.cast(pa.int64()).to_pylist() if (self.int96 and arr_ns is not None) else None
        salida = []
        for k, v in enumerate(crudos):
            if v is None:
                salida.append("")
                continue
            s, resto = divmod(v, por_s)
            if en_ns is not None and en_ns[k] is not None and (en_ns[k] - v * 1_000) % (1 << 64) < 1_000:
                resto = resto or (en_ns[k] - v * 1_000) % (1 << 64)
            try:
                d = _EPOCA + _dt.timedelta(seconds=s)
            except OverflowError:
                self.fuera_de_rango += 1
                salida.append("")
                continue
            if resto:
                self.con_fraccion += 1
            if self.zona:
                self.instantes += 1
            salida.append(f"{d.year:04d}-{d.month:02d}-{d.day:02d} {d.hour:02d}:{d.minute:02d}:{d.second:02d}")
        return salida

    def _fechas(self, arr, c: str) -> list[str]:
        pa = self.pa
        crudos = arr.cast(pa.int32() if c == "date32" else pa.int64()).to_pylist()
        salida = []
        for v in crudos:
            if v is None:
                salida.append("")
                continue
            dias = v if c == "date32" else v // 86_400_000
            try:
                d = _dt.date.fromordinal(719_163 + dias)
            except (OverflowError, ValueError):
                self.fuera_de_rango += 1
                salida.append("")
                continue
            salida.append(f"{d.year:04d}-{d.month:02d}-{d.day:02d}")
        return salida

    def _horas(self, arr) -> list[str]:
        pa = self.pa
        por_s = _POR_SEGUNDO[self.unidad]
        # `time32` (s, ms: TIME_MILLIS de Parquet, el `time-millis` de Avro) no admite el cast a int64 en pyarrow: se pasa por
        # int32, que es lo que guarda (IMPORTANTE-1 de la auditoría de 150-C8: tumbaba el fichero entero con un motivo falso).
        crudos = arr.cast(pa.int32() if arr.type.bit_width == 32 else pa.int64()).to_pylist()
        salida = []
        for v in crudos:
            if v is None:
                salida.append("")
                continue
            s, resto = divmod(v, por_s)
            # m2 de la re-auditoría de 150-C8: fuera del día (24:00:00, negativa) sale vacía con su PROPIO motivo, y se mira ANTES
            # de contar la fracción: un valor que se deja vacío no perdió una fracción de nada.
            if not 0 <= s < 86_400:
                self.hora_fuera += 1
                salida.append("")
                continue
            if resto:
                self.con_fraccion += 1
            salida.append(f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}")
        return salida


def _filas_de_los_lotes(f: _Fichero, columnas: list[_Columna], indices: list[int], proyeccion: list[str] | None, contador: list[int]):
    """Las filas del CSV, lote a lote: nunca están en memoria todos los valores de Python a la vez."""
    lotes = f.pf.iter_batches(batch_size=TAMANO_DE_LOTE, columns=proyeccion, use_threads=False)
    # Las INT96 que se convierten, otra vez y en ns (solo ellas), con los MISMOS lotes: es lo que cuenta su fracción.
    de_int96 = [k for k, c in enumerate(columnas) if c.int96]
    lotes_ns = None
    if de_int96 and f.pf_ns is not None:
        proy_ns = [proyeccion[k] for k in de_int96] if proyeccion is not None else None
        lotes_ns = f.pf_ns.iter_batches(batch_size=TAMANO_DE_LOTE, columns=proy_ns, use_threads=False)
    for lote in lotes:
        n = lote.num_rows
        contador[0] += n
        ns_de: dict[int, object] = {}
        if lotes_ns is not None:
            lote_ns = next(lotes_ns, None)
            if lote_ns is None or lote_ns.num_rows != n:
                raise rechazar("parquet_danado", tipo="las dos lecturas de una columna INT96 no dan las mismas filas")
            for j, k in enumerate(de_int96):
                ns_de[k] = lote_ns.column(j if proyeccion is not None else indices[k])
        cols = [c.textos(lote.column(k if proyeccion is not None else i), ns_de.get(k))
                for k, (c, i) in enumerate(zip(columnas, indices))]
        yield from zip(*cols)


def convertir_parquet(datos: bytes, nombre: str = "", *, hoja=None, fila_de_cabecera=None, filas_de_cabecera: int = 1,
                      filas_a_excluir=(), rellenar_combinadas: bool = False, partir_columna_unica=None,
                      componer_cabecera_de_dos_niveles: bool = False, topes: Topes | None = None):
    """El CSV canónico de un Parquet y su `origen` (el mismo `TablaConvertida` que Excel).

    Un Parquet no tiene hojas, ni cabecera que elegir, ni filas de total que confirmar: lo que sea de Excel y se pida aquí
    se RECHAZA (`parametro_invalido`) en vez de ignorarse, porque ignorar `filas_a_excluir` sería dejar en el CSV una fila
    que la persona pidió quitar y decirle que no.
    """
    from .api import TablaConvertida, ESQUEMA_DE_ORIGEN, _texto  # noqa: PLC0415
    from .valores import escribir_csv  # noqa: PLC0415
    _rechazar_lo_de_excel(hoja, fila_de_cabecera, filas_de_cabecera, filas_a_excluir, rellenar_combinadas,
                          partir_columna_unica, componer_cabecera_de_dos_niveles)
    f = _Fichero(datos, nombre, topes or TOPES_DE_FABRICA)
    info = f.columnas()
    pa = f.pa
    f.comprobar_memoria(sum(1 for c in info if not c["categoria"]))
    declaraciones: list[dict] = []
    for c in info:
        if c["categoria"]:
            declaraciones.append(declarar("parquet_columna_excluida", columna=c["nombre"], tipo_arrow=c["tipo"],
                                          categoria=c["categoria"]))
    quedan = [c for c in info if not c["categoria"]]
    if not quedan:
        raise rechazar("parquet_sin_columnas_convertibles", columnas=len(info))

    # ── nombres: normalizados como los de Excel, vacíos → columna_N, repetidos → X, X_2 ──
    crudos: list[str] = []
    for c in quedan:
        original = c["nombre"] if isinstance(c["nombre"], str) else str(c["nombre"])
        crudos.append(_texto(original))
    finales: list[str] = []
    # m1 de la re-auditoría de 150-C8: un nombre vacío se llama `columna_N`, pero si ya había una `columna_N` queda `columna_N_2`.
    # Su declaración se escribe DESPUÉS de resolver los choques (en el mismo sitio de la lista), con el nombre que de verdad queda.
    vacios: dict[int, int] = {}          # índice en `quedan` → posición de la declaración puesta de reserva
    for c, t, original in zip(quedan, crudos, (c["nombre"] for c in quedan)):
        if not t:
            n = f"columna_{c['posicion']}"
            vacios[len(finales)] = len(declaraciones)
            finales.append(n)
            declaraciones.append(None)
        elif len(quedan) == 1 and ";" in t and "," not in t:
            finales.append(t.replace(";", "_"))
            declaraciones.append(declarar("punto_y_coma_en_el_nombre", columna=finales[-1], original=t))
        else:
            finales.append(t)
    usados: set[str] = set()
    pendientes = []
    for i, n in enumerate(finales):
        if n in usados:
            pendientes.append(i)
        else:
            usados.add(n)
    for i in pendientes:
        base, k = finales[i], 2
        while f"{base}_{k}" in usados:
            k += 1
        usados.add(f"{base}_{k}")
        crudo = quedan[i]["nombre"] if isinstance(quedan[i]["nombre"], str) else base
        # Un nombre vacío no «se llamaba» nada: ya lo dice su declaración de nombre vacío, con el nombre final (m1).
        if i not in vacios:
            declaraciones.append(declarar("parquet_nombre_repetido", columna=f"{base}_{k}", original=crudo,
                                          posicion=quedan[i]["posicion"], **({"normalizado": base} if crudo != base else {})))
        finales[i] = f"{base}_{k}"
    for i, donde in vacios.items():
        declaraciones[donde] = declarar("parquet_nombre_vacio", columna=finales[i], posicion=quedan[i]["posicion"])
    for c, t, fin in zip(quedan, crudos, finales):
        if t and isinstance(c["nombre"], str) and t != c["nombre"] and fin == t:
            declaraciones.append(declarar("nombre_normalizado", columna=fin, original=c["nombre"]))

    de_pandas = _indices_de_pandas(f.esquema)
    for c, fin in zip(quedan, finales):
        if c["nombre"] in de_pandas:
            declaraciones.append(declarar("parquet_indice_de_pandas", columna=fin, original=c["nombre"]))

    # ── lectura por lotes, solo de las columnas que se convierten ──
    columnas = [_Columna(pa, c["nombre"], c["_tipo"], c["_int96"]) for c in quedan]
    indices = [c["posicion"] - 1 for c in quedan]
    nombres_arrow = [c["nombre"] for c in info]
    unicos = len(set(nombres_arrow)) == len(nombres_arrow)
    # Con nombres de columna repetidos no se puede pedir una proyección por nombre: se lee entero y se elige por posición.
    proyeccion = [c["nombre"] for c in quedan] if unicos else None
    contador = [0]
    try:
        csv_bytes = escribir_csv(finales, _filas_de_los_lotes(f, columnas, indices, proyeccion, contador))
    except TablaRechazada:
        raise
    except MemoryError:
        raise rechazar("lector_sin_memoria", variante="parquet") from None
    except Exception as e:  # noqa: BLE001 — ninguna excepción de la biblioteca llega a la persona
        raise f._rechazo_de(e) from e
    if contador[0] != f.filas:
        raise rechazar("parquet_danado", tipo=f"el pie declara {f.filas} filas y se leyeron {contador[0]}")

    # ── lo que se hizo, dicho ──
    for col, fin in zip(columnas, finales):
        if col.zona:
            declaraciones.append(declarar("parquet_zona_horaria_a_utc", columna=fin, cuantas=col.instantes, zona=col.zona))
        if col.unidad_guardada in ("ms", "us", "ns"):
            declaraciones.append(declarar("parquet_unidad_a_segundos", columna=fin, cuantas=col.con_fraccion,
                                          unidad=col.unidad_guardada))
        if col.no_finitos:
            declaraciones.append(declarar("no_finito", columna=fin, cuantas=col.no_finitos))
        if col.fuera_de_rango:
            declaraciones.append(declarar("parquet_fecha_fuera_de_rango", columna=fin, cuantas=col.fuera_de_rango))
        if col.hora_fuera:
            declaraciones.append(declarar("parquet_hora_fuera_del_dia", columna=fin, cuantas=col.hora_fuera))
        if col.int96:
            declaraciones.append(declarar("parquet_int96_en_utc", columna=fin))

    nombre_citado, quitados = nombre_citable.saneado(f.nombre)
    if quitados:
        declaraciones.append(declarar("texto_del_origen_saneado", cuantas=quitados, que="nombre"))
    origen = {
        "esquema": ESQUEMA_DE_ORIGEN,
        "formato": "parquet",
        **({"nombre": nombre_citado} if nombre_citado is not None else {}),
        "sha256": hashlib.sha256(f.datos).hexdigest(),
        "bytes": len(f.datos),
        "filas": f.filas,
        "columnas": len(quedan),
        "columnas_del_fichero": len(info),
        "filas_excluidas": [],
        "conversor": _conversor(),
        "declaraciones": declaraciones,
        "sha256_del_csv": hashlib.sha256(csv_bytes).hexdigest(),
    }
    return TablaConvertida(csv_bytes=csv_bytes, origen=origen)


def _rechazar_lo_de_excel(hoja, fila_de_cabecera, filas_de_cabecera, filas_a_excluir, rellenar, partir, componer) -> None:
    def no(que_es: str, que_en: str):
        return rechazar("parametro_invalido", parametro=f"{que_es}: un fichero Parquet no tiene hojas, cabecera ni filas de total",
                        parametro_en=f"{que_en}: a Parquet file has no sheets, header rows or total rows")
    if hoja is not None:
        raise no("`hoja` no vale", "`hoja` does not apply")
    if fila_de_cabecera not in (None, 1) or isinstance(fila_de_cabecera, bool):
        raise no("`fila_de_cabecera` no vale", "`fila_de_cabecera` does not apply")
    if filas_de_cabecera != 1 or isinstance(filas_de_cabecera, bool):
        raise no("`filas_de_cabecera` no vale", "`filas_de_cabecera` does not apply")
    try:
        quitar = list(filas_a_excluir or ())
    except TypeError:
        quitar = [None]
    if quitar:
        raise no("`filas_a_excluir` no vale", "`filas_a_excluir` does not apply")
    if rellenar is not False or partir is not None or componer is not False:
        raise no("`rellenar_combinadas`, `partir_columna_unica` y `componer_cabecera_de_dos_niveles` no valen",
                 "`rellenar_combinadas`, `partir_columna_unica` and `componer_cabecera_de_dos_niveles` do not apply")
