# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Qué es un fichero — por su FIRMA, nunca por su extensión (150-C1, criterios 1 y 2).

Cada rechazo sale con su motivo propio y ninguno llega como `BadZipFile` ni como traceback.
Lo único que lee del zip es su directorio central (`ZipInfo`): **no se extrae ni se descomprime
nada** antes de comprobar la suma de los tamaños descomprimidos contra el tope.
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib

from .. import limits
from .errores import rechazar
from .topes import Topes

__all__ = ["detectar_formato", "abrir_zip_de_excel", "es_ole_de_excel", "FIRMA_ZIP",
           "FIRMA_OLE", "FIRMA_PARQUET"]

FIRMA_ZIP = b"PK\x03\x04"
FIRMA_OLE = bytes.fromhex("D0CF11E0A1B11AE1")
FIRMA_PARQUET = b"PAR1"

#: Los nombres de los dos flujos que Excel crea al proteger un .xlsx con contraseña (OLE2 los
#: guarda en UTF-16LE en su directorio).
_FLUJOS_CIFRADO = ("EncryptedPackage".encode("utf-16-le"), "EncryptionInfo".encode("utf-16-le"))
_FLUJOS_DE_LIBRO = ("Workbook".encode("utf-16-le"), "Book".encode("utf-16-le"))


def detectar_formato(bytes_iniciales: bytes) -> str:
    """Familia del fichero según sus primeros bytes.

    Devuelve `"vacio"`, `"xlsx"` (cualquier zip: xlsx/xlsm; si de verdad es un libro lo dice
    `abrir_zip_de_excel`), `"ole"` (`.xls` antiguo u OLE cifrado), `"parquet"`, `"html"`, `"xml_excel"` (SpreadsheetML 2003),
    `"texto"` (un CSV con la extensión cambiada) o `"desconocido"`.
    """
    b = bytes(bytes_iniciales[:4096])
    if not b:
        return "vacio"
    if b.startswith(FIRMA_ZIP):
        return "xlsx"
    if b.startswith(FIRMA_OLE):
        return "ole"
    if b.startswith(FIRMA_PARQUET):
        return "parquet"
    texto = _como_texto(b)
    if texto is None:
        return "desconocido"
    bajo = texto.lstrip("﻿ \t\r\n").lower()
    if bajo.startswith("<"):
        if "urn:schemas-microsoft-com:office:spreadsheet" in bajo[:4096]:
            return "xml_excel"
        if any(m in bajo[:2048] for m in ("<html", "<!doctype html", "<table", "<head", "<body",
                                           "<meta", "<?xml")):
            return "html"
    return "texto"


def _como_texto(b: bytes) -> str | None:
    """`b` como texto si lo parece (UTF-8 con o sin BOM, o UTF-16 con BOM) y NO tiene binarios."""
    if b.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return b.decode("utf-16")
        except UnicodeError:
            return None
    if b"\x00" in b:
        return None
    try:
        t = b.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Un corte en mitad de un carácter multibyte no hace binario al fichero.
        try:
            t = b[:-3].decode("utf-8-sig")
        except UnicodeDecodeError:
            return None
    raros = sum(1 for c in t if ord(c) < 32 and c not in "\t\r\n\f")
    if raros > max(2, len(t) // 50):
        return None
    return t


def es_ole_de_excel(datos: bytes) -> bool:
    return any(f in datos for f in _FLUJOS_DE_LIBRO)


def abrir_zip_de_excel(datos: bytes, topes: Topes) -> tuple[zipfile.ZipFile, dict]:
    """Abre el zip SIN extraer nada y lo valida: es un libro, no está cifrado, no es una bomba.

    Devuelve `(zip, info)` con `info = {"descomprimido": suma, "macros": bool, "miembros": set}`.
    La comprobación de la bomba es sobre los tamaños que DECLARA el directorio central, que es
    lo único que se puede saber sin descomprimir. Los lectores posteriores (`zipfile` y el de
    calamine) no devuelven de un miembro más bytes que los que declara.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(datos))
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError):
        raise rechazar("zip_danado") from None
    infos = zf.infolist()
    if any(i.flag_bits & 0x1 for i in infos):
        raise rechazar("protegido")
    total = sum(i.file_size for i in infos)
    if total > topes.max_descomprimido:
        raise rechazar("bomba", descomprimido=limits.human_bytes(total),
                       original=limits.human_bytes(len(datos)),
                       tope=limits.human_bytes(topes.max_descomprimido),
                       bytes_descomprimidos=total)
    _verificar_flujos(datos, infos)
    nombres = {i.filename for i in infos}
    if "xl/workbook.xml" not in nombres:
        if "xl/workbook.bin" in nombres:
            raise rechazar("formato_no_soportado", formato=".xlsb")
        if "content.xml" in nombres and "mimetype" in nombres:
            raise rechazar("formato_no_soportado", formato=".ods")
        raise rechazar("no_es_excel")
    return zf, {"descomprimido": total, "macros": "xl/vbaProject.bin" in nombres,
                "miembros": nombres}


def _verificar_flujos(datos: bytes, infos: list[zipfile.ZipInfo]) -> None:
    """Comprueba que cada miembro DESCOMPRIME a lo que el directorio central declara — ni más ni
    menos —, con un tope duro de salida por miembro.

    Medido el 2026-11-02 con `python-calamine 0.9.0`: si el directorio central declara 1.000 bytes
    y el flujo trae 147.291, calamine lee los 147.291 (no se fía del tamaño declarado), mientras
    `zipfile` solo falla con «Bad CRC». O sea que la suma de tamaños declarados NO basta contra una
    bomba que miente: este paso la cierra, descomprimiendo cada flujo con `zlib` y cortando en
    cuanto pasa de lo declarado. Cuesta una descompresión (~0,3 s por 58 MB).
    """
    vista = memoryview(datos)
    for i in infos:
        if i.is_dir() or i.file_size == 0 and i.compress_size == 0:
            continue
        try:
            fin_local = i.header_offset + 30
            firma, = struct.unpack_from("<I", datos, i.header_offset)
            n, m = struct.unpack_from("<HH", datos, i.header_offset + 26)
        except struct.error:
            raise rechazar("zip_danado") from None
        if firma != 0x04034B50:
            raise rechazar("zip_danado")
        inicio = fin_local + n + m
        crudo = vista[inicio:inicio + i.compress_size]
        if len(crudo) != i.compress_size:
            raise rechazar("zip_danado")
        if i.compress_type == zipfile.ZIP_STORED:
            if i.compress_size != i.file_size:
                raise rechazar("zip_danado")
            continue
        if i.compress_type != zipfile.ZIP_DEFLATED:
            raise rechazar("zip_danado")
        d = zlib.decompressobj(-15)
        salida = 0
        try:
            for a in range(0, len(crudo), 1 << 16):
                trozo = crudo[a:a + (1 << 16)]
                while True:
                    # Nunca pide más de lo que queda por declarar (+1, para ver que se pasa).
                    out = d.decompress(trozo, i.file_size - salida + 1)
                    salida += len(out)
                    if salida > i.file_size:
                        raise rechazar("zip_danado")
                    trozo = d.unconsumed_tail
                    if not trozo:
                        break
        except zlib.error:
            raise rechazar("zip_danado") from None
        if not d.eof or salida != i.file_size:
            raise rechazar("zip_danado")
