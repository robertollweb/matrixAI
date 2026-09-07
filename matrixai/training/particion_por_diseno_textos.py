# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice la propuesta de partición, EN LOS DOS IDIOMAS — 103-C4.

Mismo patrón que `objetivo_textos.py`, `diagnostico_textos.py` y
`preparacion_textos.py`: cada corte de este contrato tiene su propio
catálogo porque cada uno redacta sobre un momento distinto del flujo —
aquí, el de decidir CÓMO se van a repartir las filas, no qué se mide ni
qué se prepara.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

IDIOMAS = ("es", "en")

MOTIVOS: dict[str, dict[str, str]] = {
    "grupos_insuficientes": {
        "es": "{campo} tiene {valor} grupo(s) distintos: hacen falta al menos "
              "2 para separar desarrollo de una prueba que generalice a "
              "unidades nuevas. No se rompe un grupo entre roles para "
              "conseguir dos partes artificialmente",
        "en": "{campo} has {valor} distinct group(s): at least 2 are needed "
              "to separate development from a test that generalizes to new "
              "units. A group is never split across roles just to force two parts",
    },
    "pliegues_reducidos_por_grupos": {
        "es": "se pidieron {opciones} pliegues y solo hay {valor} grupos en "
              "desarrollo: se reduce el número de pliegues a {valor} en vez "
              "de romper un grupo entre dos pliegues para completar el resto",
        "en": "{opciones} folds were requested and development only has "
              "{valor} groups: the fold count is reduced to {valor} instead "
              "of splitting a group across two folds to fill the rest",
    },
    "pliegues_reducidos_por_eventos": {
        "es": "se pidieron {opciones} pliegues y la clase minoritaria de "
              "desarrollo solo tiene {valor} evento(s): se reduce el número "
              "de pliegues a {valor} para que cada pliegue de validación "
              "pueda tener al menos uno",
        "en": "{opciones} folds were requested and development's minority "
              "class only has {valor} event(s): the fold count is reduced to "
              "{valor} so each validation fold can have at least one",
    },
    "una_sola_clase_no_estratifica": {
        "es": "{campo} tiene una sola clase en desarrollo: no hay nada que "
              "estratificar, los pliegues se reparten por filas sin más",
        "en": "{campo} has a single class in development: there is nothing "
              "to stratify, folds are assigned by row alone",
    },
    "cronologia_no_se_pudo_separar": {
        "es": "con la fracción de prueba pedida y el hueco declarado, no "
              "queda ninguna fila del lado de desarrollo, o ninguna del lado "
              "de prueba: el diseño temporal pedido no es viable con estos datos",
        "en": "with the requested test fraction and the declared gap, no rows "
              "remain on the development side, or none on the test side: the "
              "requested temporal design is not viable with this data",
    },
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla — para comprobar que las dos
    redacciones piden LOS MISMOS campos."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def motivo(clave: str, **campos: object) -> dict[str, str]:
    """El motivo, compuesto en los dos idiomas."""
    if clave not in MOTIVOS:
        raise KeyError(
            f"{clave!r} no está en el catálogo de motivos del 103-C4: escribir aquí "
            "una frase suelta dejaría media aplicación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(**campos) for idioma in IDIOMAS}
