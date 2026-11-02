# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Abrir tablas que no son CSV (150-C1: Excel; Parquet vendrá en un corte aparte).

Una sola regla para el Studio, la CLI y el lote: bytes de un libro → CSV canónico + `origen`
(la huella del ORIGINAL y todo lo que la conversión hizo). Sin imports pesados aquí: el lector
de Excel (`python-calamine`, extra opcional `tablas`) se importa dentro de la función, y sin él
el módulo se importa y dice su motivo (`TablaRechazada("falta_lector")`).

    from matrixai.tablas import inspeccionar, convertir
    vista = inspeccionar(datos, "ventas.xlsx")          # hojas, cabecera propuesta, totales…
    tabla = convertir(datos, "ventas.xlsx", hoja="Datos", fila_de_cabecera=3,
                      filas_a_excluir=[10])              # lo que la persona confirmó
    tabla.csv_text, tabla.origen
"""
from __future__ import annotations

from .api import ESQUEMA_DE_ORIGEN, TablaConvertida, convertir, inspeccionar
from .errores import CODIGOS, TablaRechazada
from .formato import detectar_formato
from .topes import TOPES_DE_FABRICA, Topes

__all__ = ["inspeccionar", "convertir", "detectar_formato", "TablaConvertida", "TablaRechazada",
           "Topes", "TOPES_DE_FABRICA", "CODIGOS", "ESQUEMA_DE_ORIGEN"]
