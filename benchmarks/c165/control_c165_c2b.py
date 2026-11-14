# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2b — EL CONTROL DESPUÉS DEL VEREDICTO: la variante que eligió el banco (`frecuencia|centro`, regla sellada `845586e7…`,
DECIDIDO el 14-11 sobre `~/cola-nocturna/resultados/c165-c2b/banco_c2b.json`) da EL MISMO orden que `puntuar_filas` del producto.

Fichero aparte a propósito: `banco_c165_c2b.py` es una dependencia con sello del analizador (su sha256 va en la salida medida) y tocarlo
dejaría SIN VEREDICTO la medida que decidió. Como en el C2 (`banco_c165.PRODUCTO_VARIANTE`), «hoy» sigue siendo la REFERENCIA con la que
se midió, y esto comprueba que el producto lleva lo elegido.
"""
from __future__ import annotations

import banco_c165_c2b as b2

#: Lo que eligió el C2b y lleva el producto (`matrixai_engines.anomalias._codigos_ordinales`).
ELEGIDA = {"orden": "frecuencia", "vacia": "centro"}


def control_elegida_igual_al_producto(filas: list[dict]) -> bool:
    res = b2._an.puntuar_filas(filas, None, 0, n_semillas=2)
    return [int(i) for i in b2.ordenar(filas, ELEGIDA)] == [e["fila"] - 1 for e in res["filas_a_revisar"]]
