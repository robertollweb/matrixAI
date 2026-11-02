# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los topes que se comprueban sobre el ORIGINAL, antes de abrirlo (150-C1, invariante 6).

Dos clases de cosas, que no se confunden:

* **De seguridad, no de producto** (bomba de compresión, memoria): no se eligen ni se suben en
  Ajustes. La memoria NO es una segunda guarda: se pregunta a la que ya existe
  (`matrixai.limits.memoria_insuficiente`, la de los CSV) con los bytes «equivalentes» a lo que
  leer una hoja necesita, y se redacta el motivo aquí.
* **Parámetros** de quien llama (`Topes`): el backend de la demo los baja; el perfil completo
  deja los de fábrica (los límites físicos de Excel), porque «el Studio que se descarga no tiene
  tope de filas».
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import limits
from .errores import rechazar

__all__ = ["Topes", "TOPES_DE_FABRICA", "BYTES_POR_CELDA_DEL_RANGO", "FACTOR_RAM_POR_BYTE_DE_XML",
           "FACTOR_RAM_POR_BYTE_DE_CADENAS", "comprobar_memoria", "ram_estimada_de_una_hoja",
           "techo_del_lector", "comprobar_cadenas_compartidas"]

#: Un libro cuyo contenido descomprimido pasa de esto no se abre. 1 GiB: una hoja de 1.048.576
#: filas × 30 columnas de números ocupa ~0,9 GB de XML, así que no recorta un caso legítimo; la
#: bomba medida en el inventario (20,7 MB → 309 MB) sí la pilla la guarda de memoria / filas.
MAX_DESCOMPRIMIDO_DE_FABRICA = 1 << 30

#: Los límites físicos de una hoja de Excel (.xlsx): 1.048.576 filas × 16.384 columnas.
MAX_FILAS_DE_FABRICA = 1_048_576
MAX_COLUMNAS_DE_FABRICA = 16_384

#: La tabla de textos compartidos (`sharedStrings.xml`) la carga calamine ENTERA al abrir el libro,
#: la use la hoja o no. Medido el 2026-11-02 (casos del auditor, 17 B por texto de una letra): 200 MB
#: de XML → 682 MB de pico y 20 s; 900 MB → 3.094 MB y 141 s, con una hoja de 2×2. 256 MiB son ~1,5 M
#: de textos DISTINTOS de 150 caracteres: más de lo que guarda un libro de verdad.
MAX_CADENAS_COMPARTIDAS_DE_FABRICA = 256 << 20

#: Tiempo de pared máximo del proceso que lee con calamine. La hoja de 50.000 × 30 tarda ~2 s.
MAX_SEGUNDOS_DEL_LECTOR_DE_FABRICA = 120.0


@dataclass(frozen=True)
class Topes:
    max_descomprimido: int = MAX_DESCOMPRIMIDO_DE_FABRICA
    max_filas: int = MAX_FILAS_DE_FABRICA
    max_columnas: int = MAX_COLUMNAS_DE_FABRICA
    max_cadenas_compartidas: int = MAX_CADENAS_COMPARTIDAS_DE_FABRICA
    max_segundos_lector: float = MAX_SEGUNDOS_DEL_LECTOR_DE_FABRICA
    #: Techo de memoria del proceso lector. `None` = el calculado (`techo_del_lector`).
    max_memoria_lector: int | None = None


TOPES_DE_FABRICA = Topes()

# LOS FACTORES ESTÁN MEDIDOS, no supuestos (2026-11-02, esta máquina, Python 3.12, python-calamine
# 0.9.0, pico de RSS de `convertir` entera en un proceso limpio, restando el RSS de arranque):
#
#   hoja densa   50.000 x 30 mixtas   58,3 MB de XML -> +256 MB   (x4,4 el XML; 170 B/celda)
#   hoja densa  200.000 x  5 números  46,9 MB de XML -> +213 MB   (x4,5 el XML; 213 B/celda)
#   hoja densa   20.000 x 100 textos  77,9 MB de XML -> +392 MB   (x5,0 el XML; 196 B/celda)
#   hoja densa   50.000 x 30 ENTEROS, cortos y textos (`generar.grande`, la de la prueba)
#                                     47,2 MB de XML -> +178 MB   (x3,8 el XML; 119 B/celda)
#   UNA celda en Z1048576 (27,3 M de celdas de rango, 7 con valor) -> +1.106 MB (40,5 B/celda)
#
# Como en la guarda de los CSV, se usa el MÍNIMO medido redondeado hacia abajo (solo se rechaza lo
# que no cabría NI EN EL MEJOR CASO), y cada cota cubre un caso distinto: el rectángulo que
# calamine reserva aunque esté vacío (40 B/celda) y el contenido que de verdad hay (x3 el XML).
BYTES_POR_CELDA_DEL_RANGO = 40
FACTOR_RAM_POR_BYTE_DE_XML = 3
#: Los textos compartidos se SUMAN (calamine los tiene todos a la vez que la hoja). Cota inferior:
#: cada byte de XML de un texto es al menos un byte en memoria (medido 3,4 B/B con textos de una
#: letra, el peor caso; con textos largos tiende a 1).
FACTOR_RAM_POR_BYTE_DE_CADENAS = 1

#: El techo del proceso lector: lo estimado ×4 (la estimación es el MEJOR caso medido; el peor
#: medido es ×1,7) más un margen fijo para el intérprete, calamine y la copia de los bytes del
#: fichero; nunca menos de 1 GiB.
_TECHO_FACTOR = 4
_TECHO_MARGEN = 512 << 20
_TECHO_MINIMO = 1 << 30


def ram_estimada_de_una_hoja(celdas_del_rango: int, bytes_de_xml: int = 0,
                             bytes_de_cadenas: int = 0) -> int:
    """Memoria que hace falta para leer y convertir una hoja: lo mayor de las dos cotas de la hoja,
    más los textos compartidos."""
    return (max(int(celdas_del_rango) * BYTES_POR_CELDA_DEL_RANGO,
                int(bytes_de_xml) * FACTOR_RAM_POR_BYTE_DE_XML)
            + int(bytes_de_cadenas) * FACTOR_RAM_POR_BYTE_DE_CADENAS)


def techo_del_lector(necesario: int, bytes_del_fichero: int, topes: "Topes") -> int:
    """`RLIMIT_AS` del proceso que lee con calamine. Si la guarda de memoria está encendida, nunca
    más de lo que esta máquina tiene disponible (no hay swap: pasarse es tumbar la máquina)."""
    if topes.max_memoria_lector:
        return int(topes.max_memoria_lector)
    techo = max(_TECHO_MINIMO, _TECHO_FACTOR * int(necesario) + 2 * int(bytes_del_fichero)
                + _TECHO_MARGEN)
    if limits.guarda_de_memoria_activa():
        disponible = limits._memoria_disponible_del_sistema()
        if disponible is not None:
            techo = min(techo, max(disponible, 256 << 20))
    return techo


def comprobar_cadenas_compartidas(n_bytes: int, topes: "Topes") -> None:
    if n_bytes > topes.max_cadenas_compartidas:
        raise rechazar("demasiados_textos", tamano=limits.human_bytes(n_bytes),
                       tope=limits.human_bytes(topes.max_cadenas_compartidas), bytes=n_bytes)


def comprobar_memoria(necesario: int, hoja: str) -> None:
    """Rechaza si esta máquina no tiene `necesario` bytes libres. Reutiliza la guarda de los CSV.

    `limits.memoria_insuficiente(csv_bytes)` multiplica por `FACTOR_RAM_POR_BYTE_DE_CSV`; se le
    pasan los bytes de CSV que EQUIVALDRÍAN a `necesario`, y de su respuesta solo se usa la
    decisión y los números (el texto es de un CSV y aquí hablamos de una hoja). Honra
    `MATRIXAI_GUARDIA_MEMORIA` y devuelve `None` (no opina) donde no se puede medir la RAM.
    """
    equivalente = math.ceil(necesario / limits.FACTOR_RAM_POR_BYTE_DE_CSV)
    p = limits.memoria_insuficiente(equivalente)
    if p is None:
        return
    raise rechazar("memoria_insuficiente", hoja=hoja,
                   necesario=limits.human_bytes(p["necesario_estimado"]),
                   disponible=limits.human_bytes(p["disponible"]))
