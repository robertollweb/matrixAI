# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""108-C4 — comparar las columnas de un CSV nuevo contra lo que
`ajustar_preparacion()` (103-C3) ya fijó en el desarrollo del estudio.

Vacío real, no una decisión de diseño: ninguna función del core comparaba
columnas de un fichero nuevo contra un dataset ya confirmado (grep
exhaustivo sobre `matrixAI`, `matrixai-engines` y `studio-backend`: cero
resultados). Lo más cercano,
`matrixai_engines.prediccion_por_lote.exigir_columnas_de_entrada()`, solo
lanza UN error de fichero entero con la lista de faltantes -- el criterio
literal de 108-C4 pide "el motivo por columna", así que este módulo
devuelve una entrada por columna, no una excepción con una lista dentro.

**Las columnas de MÁS no se rechazan, a propósito** -- mismo criterio que
ya aplica `exigir_columnas_de_entrada()`: un CSV real trae a menudo
columnas de vuelta que el estudio nunca usó como predictor (un `row_id`,
una nota, una fecha de exportación). Rechazar el fichero entero por una
columna sobrante que de todas formas se va a ignorar sería más estricto
que el propio 106-C5, y divergiría de él sin motivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from matrixai.training.preparacion import PoliticaDePreparacion
from matrixai.training.validacion_de_esquema_textos import motivo

__all__ = ["ColumnaRechazada", "ValidacionDeEsquema", "validar_esquema"]


@dataclass(frozen=True)
class ColumnaRechazada:
    """Una columna esperada que no llegó, con su motivo bilingüe -- nunca
    una cadena ya compuesta en un idioma (el mismo criterio que `Limite`,
    103-C2/C3: quien pinta elige el idioma, quien mide guarda los dos)."""

    columna: str
    motivo: dict[str, str]

    def a_json(self) -> dict[str, Any]:
        return {"columna": self.columna, "motivo": self.motivo}


@dataclass(frozen=True)
class ValidacionDeEsquema:
    ok: bool
    rechazadas: tuple[ColumnaRechazada, ...]

    def a_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "rechazadas": [r.a_json() for r in self.rechazadas]}


def validar_esquema(columnas_presentes: Sequence[str], *,
                    politica: PoliticaDePreparacion) -> ValidacionDeEsquema:
    """`columnas_presentes` son las cabeceras del CSV nuevo; `politica` es
    la que `ajustar_preparacion()` fijó sobre el train del estudio -- sus
    `columnas` son exactamente las que `transformar_fila()` necesita para
    tratar cada fila (103-C3), así que son la fuente real de "esperadas",
    no una lista mantenida aparte que pudiera divergir."""
    presentes = set(columnas_presentes)
    rechazadas = tuple(
        ColumnaRechazada(columna=esperada.columna,
                         motivo=motivo("columna_esperada_ausente", campo=esperada.columna))
        for esperada in politica.columnas
        if esperada.columna not in presentes
    )
    return ValidacionDeEsquema(ok=not rechazadas, rechazadas=rechazadas)
