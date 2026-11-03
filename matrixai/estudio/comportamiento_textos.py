# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice el banco de pruebas de comportamiento, EN LOS DOS IDIOMAS — 149-C1.

Mismo patrón que `curvas_de_la_evaluacion_textos.py` y `evaluacion_de_puntuacion_textos.py`, y
por la misma razón: lo que redacta el core se traduce en el core, y se compone el HECHO (una clave
y sus huecos), no la frase. Catálogo propio y no el compartido de `textos.py` (el contrato dice
«catálogo de motivos en textos.py»; se hizo aparte porque `textos.py` lo lee todo el paquete y las
dos veces anteriores que un contrato de resultados añadió claves allí se acabó rompiendo el
vocabulario de otro; la API es la misma: `motivo(clave, **huecos)` -> `{es, en}`).

VOCABULARIO CERRADO (I3 del contrato). El banco DESCRIBE: cifras, peor caso, contraejemplos. Solo
la monotonía declarada lleva un veredicto, y es de TRES palabras. Quedan fuera de cualquier texto
del banco, en los dos idiomas: garantiza/garantía, demuestra, cumple, seguro, aprobado, suspenso,
pasa/falla, robusto y «monótono» sin matices (guarantee, prove, comply, safe, pass/fail, robust).
La ÚNICA excepción es la palabra «garantía»/«guarantee» DENTRO de la frase fija de límites, que
sirve justo para decir que esto NO lo es. `palabras_prohibidas` es el instrumento del barrido y lo
reutiliza quien pinte (el Studio barre su DOM con ella): una lista en cada sitio acabaría divergiendo.
"""

from __future__ import annotations

import re
import unicodedata

from matrixai.estudio.calibracion import _cifra
from matrixai.estudio.textos import PorIdioma

__all__ = [
    "CLAVES_DE_VEREDICTO", "CLAVE_DE_LOS_LIMITES", "IDIOMAS", "MOTIVOS", "PorIdioma",
    "cifra", "huecos_de", "motivo", "palabras_prohibidas", "porcentaje",
]

IDIOMAS = ("es", "en")

#: Las tres palabras del veredicto de la monotonía y la clave de su texto, en el mismo orden que
#: `comportamiento.VEREDICTOS`. Una prueba exige que coincidan: dos listas del mismo vocabulario.
CLAVES_DE_VEREDICTO = {
    "sin_contraejemplos": "veredicto_sin_contraejemplos",
    "con_contraejemplos": "veredicto_con_contraejemplos",
    "no_se_pudo_comprobar": "veredicto_no_se_pudo_comprobar",
}

#: La frase fija de límites (I4). Va PEGADA a cada veredicto, no en un pie.
CLAVE_DE_LOS_LIMITES = "limites_del_banco"

MOTIVOS: dict[str, dict[str, str]] = {
    # ---- el veredicto: tres palabras -------------------------------------------------------------
    "veredicto_sin_contraejemplos": {
        "es": "sin contraejemplos en esta rejilla",
        "en": "no counterexamples on this grid",
    },
    "veredicto_con_contraejemplos": {
        "es": "con contraejemplos",
        "en": "with counterexamples",
    },
    "veredicto_no_se_pudo_comprobar": {
        "es": "no se pudo comprobar",
        "en": "could not be checked",
    },
    # ---- la frase fija de límites (texto definitivo del 149-C1) ----------------------------------
    "limites_del_banco": {
        "es": "Esto se ha comprobado en {puntos} puntos y {filas} filas con semilla {semilla}. No es una "
              "garantía: con otros valores, otras combinaciones o otras columnas a la vez puede "
              "comportarse de otra manera. La monotonía declarada no se impuso al entrenar.",
        "en": "This was checked at {puntos} points and {filas} rows with seed {semilla}. It is not a "
              "guarantee: with other values, other combinations or other columns at once it may "
              "behave differently. The declared monotonicity was not imposed during training.",
    },
    # ---- la lectura que acompaña al veredicto: SIEMPRE con la amplitud al lado (I5) ---------------
    # El veredicto es por FRACCIÓN de filas (149-C1-bis): «con contraejemplos» = al menos el q de las filas
    # comprobadas retroceden más que la tolerancia. La frase dice la fracción y el mínimo que se exigía.
    "lectura_con_contraejemplos": {
        "es": "El {fraccion} de las filas ({violadoras} de {filas}) retroceden más de {tolerancia}, y el "
              "veredicto pide al menos el {minimo}; el mayor retroceso es {retroceso}. La curva típica de "
              "una fila se mueve {amplitud} entre su valor más bajo y el más alto. Dirección declarada: "
              "{direccion}.",
        "en": "{fraccion} of the rows ({violadoras} of {filas}) go back by more than {tolerancia}, and the "
              "verdict asks for at least {minimo}; the largest step back is {retroceso}. The typical "
              "curve of a row moves {amplitud} between its lowest and its highest value. Declared "
              "direction: {direccion}.",
    },
    "lectura_sin_contraejemplos": {
        "es": "El {fraccion} de las filas ({violadoras} de {filas}) retroceden más de {tolerancia}, por "
              "debajo del {minimo} que pide el veredicto; el mayor retroceso es {retroceso}. La curva "
              "típica de una fila se mueve {amplitud} entre su valor más bajo y el más alto. Dirección "
              "declarada: {direccion}.",
        "en": "{fraccion} of the rows ({violadoras} of {filas}) go back by more than {tolerancia}, below "
              "the {minimo} the verdict asks for; the largest step back is {retroceso}. The typical "
              "curve of a row moves {amplitud} between its lowest and its highest value. Declared "
              "direction: {direccion}.",
    },
    # En binaria la tolerancia de partida es ALTA, y por qué (149-C0-ter). Sin «garantía» ni «seguro» (I3).
    "tolerancia_alta_en_binaria": {
        "es": "La tolerancia usada es alta: {tolerancia} de la probabilidad. Es la de partida en "
              "clasificación binaria, a propósito: las curvas de los modelos de árboles avanzan a "
              "escalones de entre el 10 % y el 15 % de la probabilidad con unas 2.000 filas, y con una "
              "tolerancia menor esos escalones se contarían como retrocesos.",
        "en": "The tolerance in use is high: {tolerancia} of the probability. It is the starting one for "
              "binary classification, on purpose: the curves of tree models move in steps of between "
              "10% and 15% of the probability with about 2,000 rows, and with a lower tolerance those "
              "steps would be counted as steps back.",
    },
    "lectura_sin_comprobar": {
        "es": "No hay lectura de esta columna: ver el motivo.",
        "en": "There is no reading for this column: see the reason.",
    },
    "direccion_sube": {"es": "sube", "en": "goes up"},
    "direccion_baja": {"es": "baja", "en": "goes down"},
    # ---- por qué no se pudo comprobar: un valor ausente NO es un cero (I8) -----------------------
    "amplitud_no_distingue": {
        "es": "la curva típica de una fila se mueve {amplitud} entre su valor más bajo y el más alto, y eso "
              "no supera la tolerancia ({tolerancia}): la prueba no distingue una dirección de la "
              "contraria, y «sin contraejemplos» aquí no diría nada",
        "en": "the typical curve of a row moves {amplitud} between its lowest and its highest value, and "
              "that does not exceed the tolerance ({tolerancia}): the check cannot tell one direction "
              "from the opposite, and \"no counterexamples\" would say nothing here",
    },
    "sin_prediccion": {
        "es": "el modelo no dio predicción en ninguna de las {filas} filas de esta columna",
        "en": "the model gave no prediction on any of the {filas} rows of this column",
    },
    "sin_filas": {
        "es": "no hay filas sobre las que comprobar esta columna",
        "en": "there are no rows to check this column on",
    },
    "rejilla_sin_rango": {
        "es": "de esta columna no se vio un rango (mínimo y máximo distintos) y no hay una rejilla que recorrer",
        "en": "no range (distinct minimum and maximum) was seen for this column, so there is no grid to walk",
    },
    "escala_de_regresion_no_valida": {
        "es": "la amplitud p90−p10 del objetivo de entrenamiento es {valor}: sin una escala positiva no "
              "hay tolerancia en unidades del objetivo",
        "en": "the p90−p10 spread of the training target is {valor}: without a positive scale there is "
              "no tolerance in units of the target",
    },
    "recalibracion_no_conserva_el_orden": {
        "es": "la recalibración del estudio tiene pendiente b = {valor}, que no es positiva: la "
              "probabilidad que ve la persona no conserva el orden de la puntuación, y comprobar "
              "la monotonía sobre ella no tendría sentido",
        "en": "the study's recalibration has slope b = {valor}, which is not positive: the probability "
              "the person sees does not keep the order of the score, and checking monotonicity on it "
              "would make no sense",
    },
    "multiclase_sin_clase_declarada": {
        "es": "en multiclase la monotonía solo se comprueba para una clase declarada (al subir la "
              "columna, sube la probabilidad de esa clase): la probabilidad máxima no vale",
        "en": "in multiclass, monotonicity is only checked for a declared class (as the column goes up, "
              "the probability of that class goes up): the maximum probability will not do",
    },
    "salida_no_numerica": {
        "es": "el modelo devolvió una salida que no es un único número por fila: la monotonía se "
              "comprueba sobre una sola salida (una probabilidad o un valor)",
        "en": "the model returned an output that is not a single number per row: monotonicity is "
              "checked on a single output (a probability or a value)",
    },
    # ---- avisos y escalas ---------------------------------------------------------------------
    "filas_sin_prediccion": {
        "es": "{sin} de {total} filas no tuvieron predicción y no cuentan en esta columna",
        "en": "{sin} of {total} rows had no prediction and do not count for this column",
    },
    "escala_de_probabilidad": {
        "es": "probabilidad calibrada tal como la ve la persona (la tolerancia son puntos de probabilidad)",
        "en": "calibrated probability as the person sees it (the tolerance is in probability points)",
    },
    "escala_de_regresion": {
        "es": "unidades del objetivo (la tolerancia es una fracción de la amplitud p90−p10 del objetivo de entrenamiento)",
        "en": "units of the target (the tolerance is a fraction of the p90−p10 spread of the training target)",
    },
    # ---- «si falta una columna» (A): describe, no aprueba ------------------------------------------
    "falta_nunca_visto": {
        "es": "esta columna nunca faltó al entrenar: la caída mide cómo se comporta el modelo ante un hueco "
              "que no vio, no cuánto depende de ella",
        "en": "this column never went missing during training: the drop measures how the model behaves "
              "with a gap it never saw, not how much it depends on the column",
    },
    "falta_sin_pliegues_con_valor": {
        "es": "ningún pliegue dio una caída para esta columna: no hay cifra que dar",
        "en": "no fold gave a drop for this column: there is no figure to give",
    },
    # ---- sensibilidad (B): describe, no aprueba ----------------------------------------------------
    "sensibilidad_sin_decision": {
        "es": "no hay decisión que cambie (regresión o sin función de decisión): se da cuánto se mueve la "
              "salida, no una tasa de cambios de decisión",
        "en": "there is no decision to change (regression or no decision function): how much the output "
              "moves is given, not a rate of decision changes",
    },
    "sensibilidad_sin_rango": {
        "es": "de esta columna no se vio un rango (mínimo y máximo distintos): sin él no hay un cambio "
              "pequeño que medir como fracción del rango",
        "en": "no range (distinct minimum and maximum) was seen for this column: without it there is no "
              "small change to measure as a fraction of the range",
    },
    "sensibilidad_sin_filas": {
        "es": "ninguna fila tiene un valor numérico en esta columna: no hay nada que mover",
        "en": "no row has a numeric value in this column: there is nothing to move",
    },
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla — para comprobar que las dos redacciones piden LOS MISMOS
    campos: un hueco que existe en castellano y no en inglés solo revienta cuando alguien pide inglés."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def motivo(clave: str, **campos: object) -> dict[str, str]:
    """El motivo, compuesto en los dos idiomas — nunca una cadena ya compuesta en uno solo, que al
    cambiar de idioma no cambiaría."""
    if clave not in MOTIVOS:
        raise KeyError(
            f"{clave!r} no está en el catálogo de textos del 149: escribir aquí una frase suelta "
            "dejaría media pantalla sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(
        **{k: (v[idioma] if isinstance(v, PorIdioma) else v) for k, v in campos.items()})
        for idioma in IDIOMAS}


def cifra(valor: float, *, decimales: int = 3) -> PorIdioma:
    """Un número para una frase, con coma decimal en castellano y punto en inglés. Misma regla que
    las curvas del 141 (`calibracion._cifra`): dos sitios escribiendo números acaban divergiendo."""
    return _cifra(float(valor), decimales=decimales)


def porcentaje(fraccion: float) -> PorIdioma:
    """`0,01` -> «1 %» / «1%». La tolerancia se enseña como porcentaje de la escala, que es como se
    declara y se edita."""
    texto = _cifra(float(fraccion) * 100.0, decimales=3)
    return PorIdioma(es=f"{texto['es']} %", en=f"{texto['en']}%")


# ---------------------------------------------------------------------------
# El barrido del vocabulario (I3)
# ---------------------------------------------------------------------------

def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


#: Raíces prohibidas por idioma, SIN tildes (el texto se normaliza antes de buscar) y con límite de
#: palabra por delante: «pasa» no casa con «pasada», pero «aprobado» y «aprobada» sí. La lista no es
#: de contenido escogido a mano para el barrido de IDIOMA (eso usa palabras funcionales); es el
#: vocabulario que el contrato prohíbe, y por eso es cerrada y está aquí, una sola vez.
_PROHIBIDAS = {
    "es": (
        r"garantiz\w*", r"garanti[ao]s?", r"demuestr\w*", r"demostr\w*", r"cumpl\w*",
        r"segur[oa]s?", r"aprob\w*", r"suspens\w*", r"suspend\w*", r"pasan?", r"fallan?",
        r"robust\w*", r"monoton[oa]s?",
    ),
    "en": (
        r"guarantee\w*", r"prove[sdn]?", r"proving", r"proof", r"compl(?:y|ies|ied|iant|iance)",
        r"safe\w*", r"approv\w*", r"pass(?:es|ed|ing)?", r"fail(?:s|ed|ing|ure\w*)?",
        r"robust\w*", r"monoton(?:e|ic)",
    ),
}
_PATRON = {idioma: re.compile(r"\b(?:" + "|".join(raices) + r")\b", re.IGNORECASE)
           for idioma, raices in _PROHIBIDAS.items()}
_GARANTIA = {"es": re.compile(r"^garanti[ao]s?$"), "en": re.compile(r"^guarantee\w*$")}


def palabras_prohibidas(texto: str, idioma: str, *, es_la_frase_de_limites: bool = False) -> list[str]:
    """Las palabras del vocabulario prohibido (I3) que aparecen en `texto`, en minúsculas y sin tildes.
    Lista vacía = limpio. Con `es_la_frase_de_limites=True` se admite «garantía»/«guarantee» (la
    frase fija existe para decir que esto NO lo es) y NADA más: «aprobado» en esa frase sigue siendo
    un hallazgo."""
    if idioma not in IDIOMAS:
        raise KeyError(idioma)
    hallazgos = [m.group(0) for m in _PATRON[idioma].finditer(_sin_tildes(texto))]
    if es_la_frase_de_limites:
        hallazgos = [h for h in hallazgos if not _GARANTIA[idioma].match(h)]
    return hallazgos
