# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dicen los detectores del diagnóstico, EN LOS DOS IDIOMAS — 103-C2.

Mismo patrón que `matrixai/training/objetivo_textos.py`, y por la misma razón:
**lo que redacta el core se traduce en el core**. «Esta columna reproduce el
objetivo fila a fila» lo lee una persona y decide con ello — bloquear una
recomendación o aceptar una sospecha con su motivo—, y dejar media frase en
inglés para que la pantalla la remate es el defecto que este producto ya ha
pagado.

POR QUÉ UN CATÁLOGO PROPIO Y NO EL DE `objetivo_textos.py`. Aquél es el
catálogo de las PREGUNTAS del C1 —lo que falta por confirmar del problema—;
éste es el de los DETECTORES del C2 —lo que se mide sobre los datos una vez el
problema ya está confirmado—. Son vocabularios con dueños distintos y momentos
distintos del flujo: mezclarlos obligaría a los dos cortes a editar el mismo
fichero cada vez que cualquiera de los dos cambie una frase.

La forma es la misma a propósito (clave → {es, en}, `motivo()` que compone),
para que quien ya sabe leer un catálogo de este programa sepa leer los dos sin
que se lo tengan que explicar otra vez.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

#: Los idiomas que este catálogo sabe escribir. No es una lista abierta.
IDIOMAS = ("es", "en")

#: Clave -> {idioma: plantilla}. Los huecos van con `str.format`.
MOTIVOS: dict[str, dict[str, str]] = {
    # ---- errores estructurales (Bloqueo): impiden la recomendación --------
    "objetivo_duplicado_confirmado": {
        "es": "la entrada {campo} reproduce el objetivo {valor} fila a fila en las "
              "{opciones} filas comparadas: no es una correlación, es la MISMA "
              "variable con otro nombre. Un modelo entrenado con ella copia la "
              "respuesta que ya se le da y ese acierto no significa nada. Quítala de "
              "las entradas, o corrige el objetivo si la columna que había que "
              "predecir era otra",
        "en": "input {campo} reproduces target {valor} row by row across the "
              "{opciones} rows compared: this is not a correlation, it is the SAME "
              "variable under another name. A model trained on it copies the answer "
              "already handed to it, and that score means nothing. Remove it from "
              "the inputs, or correct the target if the column to predict was "
              "another one",
    },
    "variable_disponible_tras_el_desenlace": {
        "es": "{campo} se ha declarado disponible «after_outcome»: solo se conoce "
              "DESPUÉS de que el desenlace ya ha ocurrido. Usarla como entrada es "
              "predecir con información del futuro, y ningún resultado medido así "
              "se sostiene cuando el modelo se use de verdad —en ese momento la "
              "variable todavía no existe. Corrige su disponibilidad si el análisis "
              "se equivocó, o quítala de las entradas",
        "en": "{campo} has been declared available «after_outcome»: it is only "
              "known AFTER the outcome has already happened. Using it as an input "
              "predicts with information from the future, and no result measured "
              "that way holds up once the model is actually used — at that point "
              "the variable does not exist yet. Correct its availability if the "
              "analysis got it wrong, or remove it from the inputs",
    },
    "cruce_de_unidades_prohibido": {
        "es": "{campo} identifica la UNIDAD de observación ({valor}) y a la vez se "
              "declara como entrada del modelo: cruzar los dos papeles deja que el "
              "modelo memorice de qué sujeto es cada fila en vez de aprender de sus "
              "rasgos, y además impide separar sujetos entre desarrollo y prueba sin "
              "que la identidad se cuele por la puerta de atrás. Quítala de las "
              "entradas: sigue viajando como metadato de partición",
        "en": "{campo} identifies the observation UNIT ({valor}) and is also "
              "declared as a model input: crossing the two roles lets the model "
              "memorise which subject each row belongs to instead of learning from "
              "its traits, and it also prevents keeping subjects apart between "
              "development and test without identity leaking through the back "
              "door. Remove it from the inputs: it still travels as partition "
              "metadata",
    },
    # ---- sospechas (Sospecha): exigen contexto, no bloquean ----------------
    "igualdad_de_valores_sin_explicar": {
        "es": "{campo} y {valor} coinciden fila a fila en las {opciones} filas "
              "comparadas, y no se sabe por qué: pueden ser dos nombres de la misma "
              "medida, un cálculo derivado, o una fuga por una tercera variable. Es "
              "una SOSPECHA, no un veredicto — se acepta con motivo, autor y aviso "
              "persistente, o se investiga antes de entrenar",
        "en": "{campo} and {valor} match row by row across the {opciones} rows "
              "compared, and it is not known why: they may be two names for the "
              "same measurement, a derived calculation, or a leak through a third "
              "variable. This is a SUSPICION, not a verdict — it is accepted with a "
              "reason, an author and a persistent notice, or investigated before "
              "training",
    },
    "asociacion_muy_alta": {
        "es": "{campo} tiene una asociación muy alta con el objetivo ({valor} por "
              "{opciones}, con {n} filas): una correlación alta por sí sola NO "
              "demuestra fuga —el sueldo del año pasado predice bien el siguiente "
              "y es legítimo— pero tampoco la descarta. Documenta CUÁNDO está "
              "disponible esta entrada respecto al momento de predicción, o "
              "investiga si es la misma variable con otro nombre",
        "en": "{campo} has a very high association with the target ({valor} by "
              "{opciones}, with {n} rows): a high correlation alone does NOT prove a "
              "leak — last year's salary predicts this year's well and is "
              "legitimate — but it does not rule one out either. Document WHEN this "
              "input is available relative to the prediction time, or check "
              "whether it is the same variable under another name",
    },
    "proxy_probable": {
        "es": "{campo} tiene una información mutua normalizada alta con el objetivo "
              "({valor}, método {opciones}, {n} filas tras agrupar categorías con "
              "menos de {minimo} casos): puede ser un predictor legítimo o un proxy "
              "de algo que no debería usarse. Documenta qué representa esta "
              "categoría antes de confirmarla",
        "en": "{campo} has a high normalised mutual information with the target "
              "({valor}, method {opciones}, {n} rows after grouping categories "
              "with fewer than {minimo} cases): it may be a legitimate predictor or "
              "a proxy for something that should not be used. Document what this "
              "category represents before confirming it",
    },
    "categoria_predictiva_por_pureza": {
        "es": "en la entrada {campo}, la categoría {valor} predice el objetivo con "
              "pureza {opciones} sobre {n} filas, y esa pureza SE SOSTIENE en las "
              "dos mitades internas de los datos (no es ruido de una muestra "
              "pequeña): puede ser un predictor legítimo o un proxy de algo que no "
              "debería usarse. Documenta qué representa esa categoría antes de "
              "confirmarla",
        "en": "in input {campo}, category {valor} predicts the target with purity "
              "{opciones} over {n} rows, and that purity HOLDS UP across the two "
              "internal halves of the data (it is not noise from a small sample): "
              "it may be a legitimate predictor or a proxy for something that "
              "should not be used. Document what this category represents before "
              "confirming it",
    },
    "identificador_probable": {
        "es": "{campo} tiene forma de identificador ({valor}) y se declara como "
              "entrada: un identificador puede memorizar la fila en vez de aportar "
              "un rasgo, aunque también puede llevar información real (un código "
              "que agrupa familias). No se quita solo: confírmalo con lo que sabes "
              "de la columna",
        "en": "{campo} looks like an identifier ({valor}) and is declared as an "
              "input: an identifier can let the model memorise the row instead of "
              "contributing a trait, though it can also carry real information (a "
              "code that groups families). It is not removed on its own: confirm "
              "it with what you know about the column",
    },
    # ---- límites de estimación (Limite): no bloquean, acotan la conclusión -
    "falta_de_eventos": {
        "es": "el objetivo {campo} tiene solo {valor} caso(s) de la clase {opciones} "
              "en los datos disponibles: con tan pocos eventos, cualquier métrica "
              "que dependa de ellos (sensibilidad, VPP, calibración) tiene un error "
              "de muestreo del orden de 1/√{valor} y no se puede leer como una cifra "
              "precisa. No es que el estudio esté mal diseñado: es que hay que decir "
              "que la conclusión sobre esa clase queda limitada",
        "en": "target {campo} has only {valor} case(s) of class {opciones} in the "
              "available data: with so few events, any metric that depends on them "
              "(sensitivity, PPV, calibration) has a sampling error on the order of "
              "1/√{valor} and cannot be read as a precise figure. This does not mean "
              "the study is badly designed: it means the conclusion about that class "
              "has to be declared limited",
    },
    "tamano_efectivo_insuficiente": {
        "es": "hay {campo} fila(s) útil(es) para {valor} predictor(es) declarado(s) "
              "({opciones} filas por predictor): con tan pocas observaciones por "
              "grado de libertad, la estimación queda dominada por el ruido de la "
              "muestra concreta y no generaliza. La conclusión sobre este diseño "
              "queda limitada por el tamaño, no rechazada",
        "en": "there are {campo} usable row(s) for {valor} declared predictor(s) "
              "({opciones} rows per predictor): with so few observations per degree "
              "of freedom, the estimate is dominated by the noise of this particular "
              "sample and does not generalise. The conclusion about this design is "
              "limited by size, not rejected",
    },
    "precision_insuficiente": {
        "es": "la proporción estimada de {campo} es {valor} sobre {n} filas, con un "
              "margen de {opciones} sobre el propio valor (intervalo de Wilson al "
              "95 %): la cifra puede publicarse, pero su precisión es demasiado "
              "baja para apoyar una decisión fina. Se declara el límite en vez de "
              "enseñar un número más seguro de lo que es",
        "en": "the estimated proportion of {campo} is {valor} over {n} rows, with a "
              "margin of {opciones} relative to the value itself (95 % Wilson "
              "interval): the figure can be published, but its precision is too low "
              "to support a fine-grained decision. The limit is declared instead of "
              "showing a number more certain than it is",
    },
    "ausencia_de_linaje": {
        "es": "estas {valor} columna(s) de {campo} llegan sin información de cómo "
              "se calcularon: ningún detector de este core puede ver si una columna "
              "externa se derivó del objetivo, de otra unidad, o de un momento "
              "posterior al que aquí se declara. Es una LIMITACIÓN de lo que se "
              "puede afirmar, no una ausencia de problema — declarar una decisión "
              "sobre ellas no la vuelve estadísticamente válida",
        "en": "these {valor} column(s) of {campo} arrive with no information about "
              "how they were computed: no detector in this core can see whether an "
              "external column was derived from the target, from another unit, or "
              "from a moment later than the one declared here. This is a LIMIT on "
              "what can be claimed, not the absence of a problem — recording a "
              "decision about them does not make it statistically valid",
    },
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla. Se usa para comprobar que las dos
    redacciones piden LOS MISMOS campos: si el castellano usa `{opciones}` y el
    inglés no, el fallo solo aparece cuando alguien pide inglés."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def motivo(clave: str, **campos: object) -> dict[str, str]:
    """El motivo, compuesto en los dos idiomas.

    Devuelve un `dict` y no una cadena a propósito: **no se guarda en el estado
    una cadena ya compuesta en un idioma**, porque al cambiar de idioma no
    cambiaría. Quien pinte elige; quien registra guarda los dos.
    """
    if clave not in MOTIVOS:
        raise KeyError(
            f"{clave!r} no está en el catálogo de motivos del 103-C2: escribir aquí "
            "una frase suelta dejaría media aplicación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(**campos) for idioma in IDIOMAS}
