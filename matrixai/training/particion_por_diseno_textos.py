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

    # -- 140-C1: guardas de grupos y de tiempo ------------------------------
    "falta_columna_de_tiempo": {
        "es": "el diseño {valor} necesita la columna de tiempo (time_column) y no se "
              "ha indicado ninguna: sin ella no hay un orden hacia delante que respetar",
        "en": "the {valor} design needs the time column (time_column) and none was "
              "given: without it there is no forward order to respect",
    },
    "tiempo_ilegible": {
        "es": "{campo}: {valor} valor(es) no se pueden leer como fecha ni como número "
              "(por ejemplo {ejemplos}). No se ordena por una columna que no se entiende: "
              "una fila sin fecha no se puede colocar antes ni después del corte",
        "en": "{campo}: {valor} value(s) cannot be read as a date or as a number (for "
              "example {ejemplos}). Rows are not ordered by a column that is not "
              "understood: a row without a date cannot be placed before or after the cut",
    },
    "fecha_ambigua": {
        "es": "{campo} se lee igual de bien como día/mes/año que como mes/día/año (por "
              "ejemplo {ejemplos}): no se elige por orden, porque la mitad de las fechas "
              "cambiaría de mes y el corte caería en otro sitio. Escríbelas sin "
              "ambigüedad (AAAA-MM-DD) o confirma el orden",
        "en": "{campo} reads equally well as day/month/year and as month/day/year (for "
              "example {ejemplos}): the order is not chosen by default, because half of "
              "the dates would change month and the cut would fall elsewhere. Write them "
              "unambiguously (YYYY-MM-DD) or confirm the order",
    },
    "hueco_ilegible": {
        "es": "el hueco (gap) {valor} no se entiende: se espera una magnitud no negativa "
              "y una unidad de {opciones} (por ejemplo 30 days)",
        "en": "the gap {valor} is not understood: expected a non-negative magnitude and "
              "a unit among {opciones} (for example 30 days)",
    },
    "una_sola_clase_sin_guardas": {
        "es": "{campo} tiene una sola clase en los datos: no hay clase que repartir "
              "entre pliegues ni prueba, y las guardas de clase no se aplican",
        "en": "{campo} has a single class in the data: there is no class to spread "
              "across folds or the test, and the class guards do not apply",
    },
    "repeticiones_fijadas_a_1": {
        "es": "se pidieron {opciones} repeticiones y el reparto temporal solo tiene "
              "una: no se puede barajar sin mezclar pasado y futuro, y repetirlo daría "
              "los mismos pliegues; se usa {valor}",
        "en": "{opciones} repeats were requested and the temporal split has only one: "
              "it cannot be shuffled without mixing past and future, and repeating it "
              "would give the same folds; {valor} is used",
    },
    "tiempo_numerico_sin_unidad": {
        "es": "{campo} es numérico y no dice en qué unidad está: el hueco declarado "
              "se aplica tomándolo como segundos",
        "en": "{campo} is numeric and does not say its unit: the declared gap is "
              "applied taking it as seconds",
    },
    "instantes_insuficientes": {
        "es": "desarrollo solo tiene {valor} instante(s) distintos en {campo}: la "
              "validación hacia delante necesita al menos {minimo} (un bloque que solo "
              "entrena y uno o más que se validan), y las filas del mismo instante no "
              "se separan nunca",
        "en": "development has only {valor} distinct instant(s) in {campo}: forward "
              "validation needs at least {minimo} (a block that only trains and one or "
              "more that are validated), and rows with the same instant are never separated",
    },
    "pliegues_reducidos_por_instantes": {
        "es": "se pidieron {opciones} pliegues y desarrollo solo tiene {n} instantes "
              "distintos: se reducen a {valor} (las filas del mismo instante no se separan)",
        "en": "{opciones} folds were requested and development only has {n} distinct "
              "instants: reduced to {valor} (rows with the same instant are not separated)",
    },
    "pliegues_reducidos_por_hueco": {
        "es": "se pidieron {opciones} pliegues y con el hueco declarado el pliegue más "
              "antiguo se quedaría sin ninguna fila para entrenar: se reducen a {valor}",
        "en": "{opciones} folds were requested and with the declared gap the oldest fold "
              "would be left with no row to train on: reduced to {valor}",
    },
    "pliegues_reducidos_por_clase_en_bloques": {
        "es": "se pidieron {opciones} pliegues y con ellos algún bloque de validación o "
              "de entrenamiento se quedaría sin ninguna fila de la clase {clase}: se "
              "reducen a {valor} para que todos la tengan",
        "en": "{opciones} folds were requested and with them some validation or training "
              "block would have no row of class {clase}: reduced to {valor} so every "
              "block has it",
    },
    "bloques_sin_clase": {
        "es": "la clase {clase} solo tiene {valor} fila(s) en desarrollo y, repartidas en "
              "el tiempo, ni con {opciones} pliegues (el mínimo) están en todos los "
              "bloques (filas de la clase por bloque, del más antiguo al más reciente: "
              "{detalle}). Un pliegue sin esa clase no la mide: el estudio no arranca",
        "en": "class {clase} has only {valor} row(s) in development and, spread over "
              "time, not even with {opciones} folds (the minimum) is it in every block "
              "(rows of the class per block, oldest to newest: {detalle}). A fold "
              "without that class does not measure it: the study does not start",
    },
    "prueba_sin_una_clase": {
        "es": "la prueba reservada ({filas} filas, la parte más reciente) no tiene "
              "ninguna fila de la clase {clase}: una nota sin esa clase no se puede "
              "medir. Cambia la fracción de prueba o el momento del corte",
        "en": "the reserved test ({filas} rows, the most recent part) has no row of "
              "class {clase}: a score without that class cannot be measured. Change the "
              "test fraction or the cut",
    },
    "pliegues_reducidos_por_clase_en_grupos": {
        "es": "se pidieron {opciones} pliegues y la clase {clase} solo aparece en {n} "
              "grupo(s) de {campo}: para que cada pliegue y la prueba reservada tengan "
              "alguna fila de la clase hacen falta {partes} grupos con ella; se reducen "
              "los pliegues a {valor}",
        "en": "{opciones} folds were requested and class {clase} only appears in {n} "
              "group(s) of {campo}: for every fold and the reserved test to have some "
              "row of the class, {partes} groups with it are needed; folds reduced to {valor}",
    },
    "clase_sin_grupos_suficientes": {
        "es": "la clase {clase} solo aparece en {valor} grupo(s) de {campo}; con {opciones} "
              "pliegues y la prueba reservada hacen falta {partes}, y aun bajando a 2 "
              "pliegues hacen falta {minimo}. No se puede repartir la clase entre grupos "
              "sin partir un grupo, y un examen sin esa clase no la mide: el estudio no arranca",
        "en": "class {clase} only appears in {valor} group(s) of {campo}; with {opciones} "
              "folds and the reserved test, {partes} are needed, and even dropping to 2 "
              "folds {minimo} are needed. The class cannot be spread across groups "
              "without splitting a group, and an exam without that class does not "
              "measure it: the study does not start",
    },
    "clases_sin_reparto_por_grupos": {
        "es": "la clase {clase} aparece en {valor} grupo(s) de {campo}, los que hacen "
              "falta, pero los grupos que traen las clases coinciden y no se ha encontrado un "
              "reparto en el que la prueba reservada y cada pliegue tengan todas las clases, "
              "ni con 2 pliegues: el estudio no arranca",
        "en": "class {clase} appears in {valor} group(s) of {campo}, as many as needed, "
              "but the groups carrying the classes overlap and no split was found in which "
              "the reserved test and every fold have all classes, not even with 2 folds: "
              "the study does not start",
    },
    "valida_con_una_sola_unidad": {
        "es": "la validación solo tiene un {campo}: no se puede partir en dos mitades sin "
              "partir esa unidad, así que las mitades se cortan por posición y comparten "
              "unidad",
        "en": "the validation fold has a single {campo}: it cannot be halved without "
              "splitting that unit, so the halves are cut by position and share a unit",
    },
    "valida_con_un_solo_instante": {
        "es": "todas las filas de la validación tienen el mismo instante en {campo}: no "
              "hay un antes y un después, y las mitades se cortan por posición",
        "en": "all validation rows share the same instant in {campo}: there is no before "
              "and after, and the halves are cut by position",
    },
    "mitad_sin_una_clase": {
        "es": "la clase {clase} solo está en una de las dos mitades de la validación "
              "({detalle}): no hay manera de repartirla sin partir una unidad",
        "en": "class {clase} is only in one of the two validation halves ({detalle}): "
              "there is no way to spread it without splitting a unit",
    },
    "hueco_sin_filas_para_entrenar": {
        "es": "con el hueco declarado ({valor}) no queda ninguna fila para entrenar "
              "antes del primer bloque de validación, ni con 2 pliegues: reduce el hueco "
              "o aporta más historia",
        "en": "with the declared gap ({valor}) no row is left to train on before the "
              "first validation block, not even with 2 folds: reduce the gap or provide "
              "more history",
    },
    "pliegues_reducidos_por_reparto_de_clases_en_grupos": {
        "es": "se pidieron {opciones} pliegues y, aunque la clase {clase} aparece en {n} "
              "grupo(s) de {campo}, los grupos que traen las clases coinciden y con ese "
              "número de pliegues algún pliegue o la prueba se quedarían sin una clase: "
              "se reducen los pliegues a {valor}",
        "en": "{opciones} folds were requested and, although class {clase} appears in "
              "{n} group(s) of {campo}, the groups carrying the classes overlap and with "
              "that many folds some fold or the test would lack a class: folds reduced "
              "to {valor}",
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
