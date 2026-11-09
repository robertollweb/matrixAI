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

__all__ = ["Topes", "TOPES_DE_FABRICA", "MAX_FILAS_PARQUET_DE_FABRICA", "BYTES_POR_CELDA_DEL_RANGO", "FACTOR_RAM_POR_BYTE_DE_XML",
           "FACTOR_RAM_POR_BYTE_DE_CADENAS", "comprobar_memoria", "ram_estimada_de_una_hoja",
           "techo_del_lector", "comprobar_cadenas_compartidas", "hueco_bajo_el_techo_del_proceso",
           "memoria_virtual_del_proceso"]

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


#: Parquet (150-C8): un tope de SEGURIDAD sobre las filas que DECLARAN los metadatos, no de producto («el Studio que se
#: descarga no tiene tope»). Lo que de verdad acota es la guarda de memoria sobre los bytes descomprimidos; esto corta un
#: pie de fichero que miente (miles de millones de filas en unos pocos KB) antes de planificar nada. El backend de la demo
#: lo baja.
MAX_FILAS_PARQUET_DE_FABRICA = 200_000_000


@dataclass(frozen=True)
class Topes:
    max_descomprimido: int = MAX_DESCOMPRIMIDO_DE_FABRICA
    max_filas: int = MAX_FILAS_DE_FABRICA
    max_columnas: int = MAX_COLUMNAS_DE_FABRICA
    max_cadenas_compartidas: int = MAX_CADENAS_COMPARTIDAS_DE_FABRICA
    max_segundos_lector: float = MAX_SEGUNDOS_DEL_LECTOR_DE_FABRICA
    #: Techo de memoria del proceso lector. `None` = el calculado (`techo_del_lector`).
    max_memoria_lector: int | None = None
    #: Parquet: filas que declaran los metadatos (ver `MAX_FILAS_PARQUET_DE_FABRICA`).
    max_filas_parquet: int = MAX_FILAS_PARQUET_DE_FABRICA


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
#
# Y RE-MEDIDO el 2026-11-02 (23:10, 2.ª re-auditoría, I-R3): desde B2 calamine lee en un proceso
# HIJO y la tabla viaja al padre, así que gastan LOS DOS. Pico del padre (`VmHWM` desde después de
# importar) + pico del hijo, procesos limpios, con techo `systemd-run -p MemoryMax=3G`:
#
#   generar.grande 50.000 x 30   45,1 MB de XML -> padre +158..165 + hijo 159..190 = 324..348 MB (x7,2..7,7)
#   generar.grande 100.000 x 10  31,4 MB de XML -> padre +118 + hijo 117 = 235 MB                 (x7,5)
#   generar.grande 200.000 x 10  64,3 MB de XML -> padre +228 + hijo 286 = 514 MB                 (x8,0)
#
# Con x3 la estimación se quedaba en 0,37–0,42 de lo gastado. La cota del XML pasa a x9: es el
# máximo medido de la suma (x7,2..8,0) para hojas de NÚMEROS y textos CORTOS (`generar.grande`),
# redondeado hacia arriba. NO es el máximo de toda hoja: con textos largos el texto no está en el
# XML de la hoja sino en `sharedStrings.xml`, y por byte de XML de la hoja se gasta mucho más (x36
# en el primer caso de abajo); esos se cubren con el factor de los textos compartidos.
# La cota del rectángulo (40 B/celda) es del hijo y no se re-midió (deuda en el contrato).
BYTES_POR_CELDA_DEL_RANGO = 40
FACTOR_RAM_POR_BYTE_DE_XML = 9
# Y RE-MEDIDO el 2026-11-03 (3.ª auditoría, I2) con HOJAS DE TEXTOS CASI TODOS DISTINTOS
# (`~/.agentes/150/aud3-c1/mem_textos.py`, mismo método: padre + hijo, `systemd-run MemoryMax=3G`
# sin swap). Con el factor de los textos compartidos en x1 la estimación era 0,16–0,38 de lo
# gastado (los textos pasan de la tabla de calamine a la del hijo, a la del padre y al CSV):
#
#   20.000 x 30 de 100 car.  hoja 22,5 MB, textos 73,3 MB -> padre +331..396 + hijo 404 = 734..800 MB (x10,0..10,9 los textos)
#   50.000 x 10 de 200 car.  hoja 20,0 MB, textos 113,7 MB -> padre +365 + hijo 571 = 936 MB          (x8,2)
#   10.000 x 20 de 500 car.  hoja  7,4 MB, textos 108,7 MB -> padre +550 + hijo 521 = 1.071 MB        (x9,9)
#   100.000 x 10 de 20 car.  hoja 40,2 MB, textos  37,8 MB -> padre +141 + hijo 265 = 406 MB          (x10,7 con la hoja incluida)
#
# Con x10 la estimada queda entre x1,08 y x1,82 de lo gastado en los cuatro. Con x9 el de 500
# caracteres se quedaba corto (0,97), y con x12 el de 20 caracteres pasaba de x2: x10 es el hueco.
#: Los textos compartidos se SUMAN (calamine los tiene todos a la vez que la hoja), y cada byte de
#: texto se paga en calamine, en el hijo, en el padre y en el CSV (x10 medido, arriba).
FACTOR_RAM_POR_BYTE_DE_CADENAS = 10

#: Qué parte de `ram_estimada_de_una_hoja` (la SUMA de este proceso y el hijo de calamine) gasta, como poco, el proceso
#: que llama: es la que cuenta contra SU techo (`hueco_bajo_el_techo_del_proceso`). El padre medido arriba se lleva
#: entre el 35 % (100.000 x 10 de 20 car.: 141 de 406) y el 51 % de lo GASTADO, y la estimación pasa de lo gastado
#: (x1,08..x1,82): un tercio de la estimación es el mejor caso, redondeado hacia abajo. Es RSS y el techo cuenta memoria
#: virtual, que es mayor: sigue siendo una cota por abajo.
FRACCION_DEL_PROCESO_QUE_LLAMA = 1 / 3

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


def comprobar_memoria(necesario: int, hoja: str, *, usado_al_empezar: int | None = None, variante: str | None = None,
                      fraccion_del_proceso: float | None = None) -> None:
    """Rechaza si esta máquina no tiene `necesario` bytes libres. Reutiliza la guarda de los CSV.

    `limits.memoria_insuficiente(csv_bytes)` multiplica por `FACTOR_RAM_POR_BYTE_DE_CSV`; se le
    pasan los bytes de CSV que EQUIVALDRÍAN a `necesario`, y de su respuesta solo se usa la
    decisión y los números (el texto es de un CSV y aquí hablamos de una hoja). Honra
    `MATRIXAI_GUARDIA_MEMORIA` y devuelve `None` (no opina) donde no se puede medir la RAM.

    `usado_al_empezar`: la memoria virtual de este proceso ANTES de empezar a leer la hoja (`memoria_virtual_del_proceso`).
    Lo que la lectura ya ha gastado entre una comprobación y la siguiente es parte de `necesario`: medir el hueco con lo
    de ahora lo contaría dos veces.

    `variante` (Parquet, 150-C8): los motivos son los de ESE formato (`(memoria_insuficiente, variante)` y
    `(memoria_insuficiente, f"{variante}_techo_del_proceso")`). `fraccion_del_proceso`: la parte de `necesario` que gasta
    ESTE proceso (por omisión `FRACCION_DEL_PROCESO_QUE_LLAMA`, un tercio, en Excel, que reparte con el hijo de calamine; toda en Parquet, que no tiene hijo).
    """
    equivalente = math.ceil(necesario / limits.FACTOR_RAM_POR_BYTE_DE_CSV)
    p = limits.memoria_insuficiente(equivalente)
    if p is not None:
        raise rechazar("memoria_insuficiente", variante=variante, hoja=hoja,
                       necesario=limits.human_bytes(p["necesario_estimado"]),
                       disponible=limits.human_bytes(p["disponible"]))
    # Y lo que ESTE proceso puede pedir: con un techo propio (`RLIMIT_AS`, el que pone el Studio a su proceso de
    # conversión o un `ulimit -v`) la memoria del equipo no es la que cuenta. Sin esto, un libro sano que no cabe en el
    # techo empezaba a leerse y acababa en un `MemoryError` que el lector dice como «fichero dañado» (I1 de la
    # auditoría de 150-C3). Es `min(disponible, techo)`: la de arriba ya rechazó si el equipo tiene menos.
    #
    # Pero el techo es de ESTE proceso y `necesario` es la suma de los dos (este y el hijo de calamine, que tiene el
    # suyo): se compara SU parte en el mejor caso medido, para no rechazar lo que cabe. Medido el 03-11 con la guarda
    # apagada: `generar.grande(200_000, 10)` (606,5 MB estimados) SÍ se convierte con 300 MiB de techo y no con 180.
    if not limits.guarda_de_memoria_activa():
        return
    techo = hueco_bajo_el_techo_del_proceso(usado_al_empezar)
    if techo is None:
        return
    tope, hueco = techo
    propio = int(necesario * (FRACCION_DEL_PROCESO_QUE_LLAMA if fraccion_del_proceso is None else fraccion_del_proceso))
    if propio > hueco:
        raise rechazar("memoria_insuficiente", variante=f"{variante}_techo_del_proceso" if variante else "techo_del_proceso",
                       hoja=hoja,
                       necesario=limits.human_bytes(propio), disponible=limits.human_bytes(hueco),
                       techo=limits.human_bytes(tope))


def memoria_virtual_del_proceso() -> int | None:
    """`VmSize` de este proceso (lo que cuenta `RLIMIT_AS`), o `None` si no se puede medir."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for linea in fh:
                if linea.startswith("VmSize:"):
                    return int(linea.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def hueco_bajo_el_techo_del_proceso(usado: int | None = None) -> tuple[int, int] | None:
    """`(techo, hueco)`: el `RLIMIT_AS` blando de este proceso y lo que le queda por debajo (el techo menos `usado`, o
    su `VmSize` de ahora). `None` si no tiene techo o no se puede medir (no opina)."""
    try:
        import resource  # noqa: PLC0415 — no existe en Windows
        blando = resource.getrlimit(resource.RLIMIT_AS)[0]
    except (ImportError, ValueError, OSError):
        return None
    if blando == resource.RLIM_INFINITY or blando <= 0:
        return None
    if usado is None:
        usado = memoria_virtual_del_proceso() or 0
    return int(blando), max(0, int(blando) - int(usado))
