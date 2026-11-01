# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b, auditoría I4 — un intervalo de Wilson cuenta FILAS: con varias filas del mismo sujeto, se DICE.

La cobertura conforme medida en la prueba (`conforme.medir_cobertura`) y el acierto de cada banda (`bandas.medir_bandas`)
llevan un intervalo de Wilson sobre las filas, y la pantalla lo llamaba «intervalo del 95 %» y, con la cobertura, decía
«Compatible / NO compatible con la cobertura prometida». Eso supone filas independientes. Con un reparto por sujeto (`groups`,
`groups_and_time`) las filas de la prueba son de unos pocos sujetos, cada uno con su propia tasa: medido por la auditoría
(`sonda_wilson.py`, R = 2.000), con 13 sujetos de 40 filas y una desviación del logit entre sujetos de 0,5 o 1, el «95 %» cubre
de verdad 0,84 y 0,61; sin efecto del sujeto, 0,950 (el instrumento está bien).

**El método NO cambia aquí** (es compartido y lo decide Roberto, tarjeta del 140b): el intervalo sigue siendo el de Wilson por
filas. Lo que cambia es lo que se AFIRMA de él: si la muestra trae `units` y hay menos sujetos que filas, el resultado lleva
`unidades` y este aviso, en los dos idiomas, y el veredicto de compatibilidad no se emite. Sin `units` (iid, temporal) o con un
sujeto por fila, nada cambia: ni una clave más en el JSON.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["AVISO_SIN_RECUENTO_DE_SUJETOS", "aviso_del_intervalo_por_filas", "unidades_distintas"]


def unidades_distintas(unidades: Sequence[Any] | None, indices: Sequence[int] | None = None) -> int | None:
    """Cuántos sujetos distintos hay entre las filas (`indices`, o todas). `None` si la muestra no trae `units`."""
    if unidades is None:
        return None
    if indices is None:
        return len(set(unidades))
    return len({unidades[i] for i in indices})


def aviso_del_intervalo_por_filas(filas: int, unidades: int | None, *,
                                  frente_a_una_promesa: bool = False) -> dict[str, str] | None:
    """La frase que acompaña a un intervalo de Wilson de `filas` filas que son de `unidades` sujetos, o `None` si no hace
    falta (sin sujetos declarados, o uno por fila). `frente_a_una_promesa` (la cobertura conforme): dice además que no sirve
    para el veredicto de compatibilidad, que por eso no se emite."""
    if unidades is None or filas <= 0 or unidades >= filas:
        return None
    es = (f"intervalo de Wilson por FILAS: cuenta las {filas} filas, de {unidades} sujetos, como si fueran independientes. "
          "Con varias filas del mismo sujeto es más estrecho de lo que debería, y su confianza real por sujeto es menor que la "
          "nominal")
    en = (f"Wilson interval by ROWS: it counts the {filas} rows, from {unidades} subjects, as if they were independent. With "
          "several rows from the same subject it is narrower than it should be, and its real confidence by subject is lower "
          "than the nominal one")
    if frente_a_una_promesa:
        es += ": no sirve para decir si la cobertura es compatible con la prometida"
        en += ": it cannot say whether the coverage is compatible with the promised one"
    return {"es": es, "en": en}


#: La misma advertencia cuando quien enseña el intervalo NO tiene el sujeto de cada fila (el simulador de automatización guarda
#: probabilidades y desenlaces, no las filas): se sabe que el reparto fue por sujeto, no cuántos sujetos hay en cada cifra.
AVISO_SIN_RECUENTO_DE_SUJETOS: dict[str, str] = {
    "es": "intervalos de Wilson por FILAS: este estudio se repartió por sujeto y varias filas son del mismo sujeto, así que "
          "cuentan como independientes filas que no lo son. Son más estrechos de lo que deberían, y su confianza real por "
          "sujeto es menor que la nominal",
    "en": "Wilson intervals by ROWS: this study was split by subject and several rows come from the same subject, so they count "
          "as independent rows that are not. They are narrower than they should be, and their real confidence by subject is "
          "lower than the nominal one",
}
