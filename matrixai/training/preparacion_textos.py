# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice la preparación dentro del entrenamiento, EN LOS DOS IDIOMAS —
103-C3.

Mismo patrón que `objetivo_textos.py` y `diagnostico_textos.py`, y por la
misma razón: lo que redacta el core se traduce en el core.

POR QUÉ UN CATÁLOGO PROPIO Y NO EL DEL 103-C2. Aquél es el catálogo de los
DETECTORES — lo que se mide sobre los datos para encontrar fugas y
sospechas. Éste es el de la PREPARACIÓN — lo que se AJUSTA sobre esos
mismos datos, dentro de train, para poder entrenar. Son momentos y dueños
distintos del flujo (detectar no es preparar), y mezclarlos obligaría a
los dos cortes a editar el mismo fichero.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

IDIOMAS = ("es", "en")

MOTIVOS: dict[str, dict[str, str]] = {
    "faltantes_por_encima_del_umbral": {
        "es": "{campo} tiene {valor} de valores faltantes en train: por encima "
              "de la mitad de la columna, cualquier imputación —mediana, "
              "indicador o nativa— descansa en menos de la mitad de los datos "
              "reales. Es un aviso orientativo, no un bloqueo: la columna "
              "puede seguir siendo útil con esa limitación declarada",
        "en": "{campo} has {valor} missing values in train: above half the "
              "column, so any imputation — median, indicator or native — "
              "rests on less than half real data. This is an orientative "
              "warning, not a block: the column can still be useful with "
              "this limitation declared",
    },
    "pocas_filas_para_preparacion": {
        "es": "el train efectivo tiene {valor} filas: no es motivo para "
              "bloquear la recomendación por sí solo, pero las estadísticas "
              "ajustadas aquí (mediana, vocabulario de categorías) tienen "
              "menos soporte del habitual",
        "en": "the effective train set has {valor} rows: not a reason to "
              "block the recommendation on its own, but the statistics "
              "fitted here (median, category vocabulary) have less support "
              "than usual",
    },
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla — para comprobar que las dos
    redacciones piden LOS MISMOS campos."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def motivo(clave: str, **campos: object) -> dict[str, str]:
    """El motivo, compuesto en los dos idiomas — nunca una cadena ya
    compuesta en un solo idioma, que al cambiar de idioma no cambiaría."""
    if clave not in MOTIVOS:
        raise KeyError(
            f"{clave!r} no está en el catálogo de motivos del 103-C3: escribir aquí "
            "una frase suelta dejaría media aplicación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(**campos) for idioma in IDIOMAS}
