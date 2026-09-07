# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice el esquema de riesgo compuesto, EN LOS DOS IDIOMAS — 103-C5.

Mismo patrón que `objetivo_textos.py`, `diagnostico_textos.py` y
`preparacion_textos.py`, y por la misma razón: lo que redacta el core se
traduce en el core. Catálogo propio y no el de esos otros tres: éste es
el momento de ACEPTAR (o rechazar) una sospecha ya detectada, no el de
detectarla ni el de prepararla.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

IDIOMAS = ("es", "en")

MOTIVOS: dict[str, dict[str, str]] = {
    "aceptacion_sin_actor": {
        "es": "aceptar una sospecha exige declarar quién la acepta — una "
             "aceptación anónima no es un recibo auditable",
        "en": "accepting a suspicion requires declaring who accepts it — an "
             "anonymous acceptance is not an auditable receipt",
    },
    "aceptacion_sin_contexto": {
        "es": "{campo} no se puede aceptar sin contexto: la invariante 3 del "
             "103 lo exige explícitamente — una correlación alta por sí sola "
             "no basta como motivo",
        "en": "{campo} cannot be accepted without context: invariant 3 of 103 "
             "requires it explicitly — a high correlation alone is not "
             "sufficient reason",
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
            f"{clave!r} no está en el catálogo de motivos del 103-C5: escribir aquí "
            "una frase suelta dejaría media aplicación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(**campos) for idioma in IDIOMAS}
