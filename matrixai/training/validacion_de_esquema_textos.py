# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice la validación de esquema de un CSV nuevo, EN LOS DOS IDIOMAS —
108-C4.

Mismo patrón que `preparacion_textos.py` y por la misma razón: catálogo
propio porque es un momento y un dueño distintos del flujo — 103-C3 AJUSTA
sobre train; esto COMPARA un fichero nuevo, al predecir, contra lo ya
ajustado. Mezclarlos obligaría a los dos cortes a editar el mismo fichero.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

IDIOMAS = ("es", "en")

MOTIVOS: dict[str, dict[str, str]] = {
    "columna_esperada_ausente": {
        "es": "{campo} se usó para entrenar y no está en este fichero: sin "
              "ella no hay con qué predecir, no se aproxima con un valor "
              "inventado",
        "en": "{campo} was used to train and is missing from this file: "
              "without it there is nothing to predict with, it is not "
              "approximated with an invented value",
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
            f"{clave!r} no está en el catálogo de motivos del 108-C4: escribir aquí "
            "una frase suelta dejaría media aplicación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(**campos) for idioma in IDIOMAS}
