# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dicen las curvas de la evaluación cuando algo NO se dibuja, EN LOS DOS IDIOMAS — 141-C1.

Mismo patrón que `evaluacion_de_puntuacion_textos.py` y por la misma razón: lo que redacta el
core se traduce en el core, y se compone el HECHO (una clave y sus huecos), no la frase. Catálogo
propio y no el compartido de `textos.py`: meter aquí claves en aquél cambiaría un fichero que lee
todo el paquete.

Sin adjetivos: ninguna de estas frases dice «bueno», «malo», «suficiente». Dicen qué falta, qué no
se mide y por qué no hay cifra. Un valor ausente NO es un cero: cada hueco de una curva lleva una
de estas claves en vez de un número.
"""

from __future__ import annotations

from matrixai.estudio.textos import PorIdioma

__all__ = ["IDIOMAS", "MOTIVOS", "PorIdioma", "huecos_de", "motivo"]

IDIOMAS = ("es", "en")

MOTIVOS: dict[str, dict[str, str]] = {
    "prueba_vacia": {
        "es": "la prueba reservada no tiene filas: no hay nada que dibujar",
        "en": "the held-out test has no rows: there is nothing to draw",
    },
    "no_aplica_a_la_tarea": {
        "es": "{curva} no se dibuja en una tarea de {tarea}",
        "en": "{curva} is not drawn for a {tarea} task",
    },
    "roc_sin_las_dos_clases": {
        "es": "la curva ROC necesita positivos y negativos en la prueba y hay {positivos} positivos "
              "y {negativos} negativos: con una sola clase no hay nada que ordenar",
        "en": "the ROC curve needs positives and negatives in the test and there are {positivos} "
              "positives and {negativos} negatives: with a single class there is nothing to rank",
    },
    "pr_sin_positivos": {
        "es": "la curva de precisión-exhaustividad necesita positivos en la prueba y hay {positivos}",
        "en": "the precision-recall curve needs positives in the test and there are {positivos}",
    },
    "sin_puntuacion": {
        "es": "el modelo no dio una puntuación ni una probabilidad por fila, solo la clase: "
              "una etiqueta dura no ordena nada y no se presenta como curva",
        "en": "the model gave no score or probability per row, only the class: a hard label "
              "ranks nothing and is not presented as a curve",
    },
    "sin_probabilidades": {
        "es": "el modelo no dio probabilidades por fila: sin ellas no hay diagrama de fiabilidad",
        "en": "the model gave no probabilities per row: without them there is no reliability diagram",
    },
    "tramo_con_pocas_filas": {
        "es": "tramo con {valor} filas: con menos de {opciones} no se da su cifra "
              "(evidencia insuficiente)",
        "en": "bin with {valor} rows: with fewer than {opciones} its figure is not given "
              "(insufficient evidence)",
    },
    "tramo_vacio": {
        "es": "tramo sin filas: no hay cifra que dar",
        "en": "bin with no rows: there is no figure to give",
    },
    "ece_con_pocas_filas": {
        "es": "la prueba tiene {valor} filas: con menos de {opciones} no se da el ECE "
              "(evidencia insuficiente)",
        "en": "the test has {valor} rows: with fewer than {opciones} the ECE is not given "
              "(insufficient evidence)",
    },
    "ece_con_pocos_tramos": {
        "es": "el ECE no se publica: hacen falta al menos {opciones} tramos con cifra ({minimo} filas o más "
              "cada uno) y con las {filas} filas de la prueba hay {valor}; con menos tramos, el ECE mide "
              "sobre todo el nivel general de la probabilidad, no si se puede creer tramo a tramo",
        "en": "the ECE is not published: it needs at least {opciones} bins with a figure ({minimo} rows or "
              "more each) and the {filas} rows of the test give {valor}; with fewer bins, the ECE mostly "
              "measures the overall level of the probability, not whether it can be believed bin by bin",
    },
    "sin_recalibracion": {
        "es": "el estudio no recalibró la probabilidad del modelo: la fiabilidad es la de esa probabilidad "
              "tal cual, y no hay un «antes» que comparar",
        "en": "the study did not recalibrate the model's probability: reliability is that of that "
              "probability as it comes, and there is no \"before\" to compare",
    },
    # La recalibración se INTENTÓ y no se pudo ajustar. La frase es de ESTE módulo, para pantalla, según la causa estructurada del
    # calibrador (`curvas_de_la_evaluacion._causa_de_la_no_ajustada`); el diagnóstico técnico del calibrador (que lleva el nombre
    # interno del método y números con punto) viaja aparte, en `recalibracion.detalle`. Decisión del supervisor tras la auditoría.
    "recalibracion_no_ajustada": {
        "es": "la recalibración no se pudo ajustar con las filas de calibración; la decisión usa las probabilidades del modelo tal cual, y no hay un «antes» que comparar",
        "en": "the recalibration could not be fitted on the calibration rows; the decision uses the model's probabilities as they come, and there is no \"before\" to compare",
    },
    "recalibracion_no_ajustada_por_separacion": {
        "es": "en las filas de calibración, las puntuaciones del modelo ya separan del todo las dos clases, así que no se "
              "puede ajustar una recalibración; la decisión usa las probabilidades del modelo tal cual, y no hay un «antes» que comparar",
        "en": "on the calibration rows, the model's scores already separate the two classes completely, so no "
              "recalibration can be fitted; the decision uses the model's probabilities as they come, and there is no \"before\" to compare",
    },
    "recalibracion_no_ajustada_por_no_converger": {
        "es": "el ajuste de la recalibración no llegó a estabilizarse con las filas de calibración, así que no se usa; "
              "la decisión usa las probabilidades del modelo tal cual, y no hay un «antes» que comparar",
        "en": "the recalibration fit did not settle on the calibration rows, so it is not used; the decision uses the model's probabilities as they come, and there is no \"before\" to compare",
    },
    "recalibracion_no_ajustada_por_una_sola_clase": {
        "es": "las filas de calibración tenían una sola de las dos clases, y con una sola no se puede ajustar una "
              "recalibración; la decisión usa las probabilidades del modelo tal cual, y no hay un «antes» que comparar",
        "en": "the calibration rows had only one of the two classes, and with only one no recalibration can be "
              "fitted; the decision uses the model's probabilities as they come, and there is no \"before\" to compare",
    },
    "recalibracion_no_ajustada_por_pocas_filas": {
        "es": "había demasiado pocas filas de calibración para ajustar una recalibración; la decisión usa las probabilidades del modelo tal cual, y no hay un «antes» que comparar",
        "en": "there were too few calibration rows to fit a recalibration; the decision uses the model's probabilities as they come, and there is no \"before\" to compare",
    },
    "multiclase_sin_recalibrar": {
        "es": "en multiclase el estudio no recalibra (decisión del 23-09): la fiabilidad es la de la "
              "confianza en la clase predicha tal cual la da el modelo",
        "en": "in multiclass the study does not recalibrate (decision of 23-09): reliability is that "
              "of the confidence in the predicted class as the model gives it",
    },
    "recalibracion_sin_probabilidades": {
        "es": "hay un calibrador pero el modelo no dio probabilidades por fila: no se puede "
              "aplicar, y las curvas usan la puntuación tal cual",
        "en": "there is a calibrator but the model gave no probabilities per row: it cannot be "
              "applied, and the curves use the score as it comes",
    },
    "matriz_sin_umbral_en_puntuacion": {
        "es": "la muestra trae una puntuación sin escala de probabilidad y no se indicó umbral: "
              "0,5 no tiene sentido sobre ella, así que no hay matriz de confusión",
        "en": "the sample carries a score with no probability scale and no threshold was given: "
              "0.5 means nothing on it, so there is no confusion matrix",
    },
    "umbral_por_omision": {
        "es": "no se indicó el umbral elegido: la matriz usa {valor}",
        "en": "no chosen threshold was given: the matrix uses {valor}",
    },
    "prueba_con_pocas_filas": {
        "es": "la prueba reservada tiene {valor} filas: con menos de {opciones} las curvas son "
              "escalones de pocas filas y cada punto pesa mucho",
        "en": "the held-out test has {valor} rows: with fewer than {opciones} the curves are "
              "steps of a few rows and each point weighs a lot",
    },
    "prueba_con_pocos_eventos": {
        "es": "la clase minoritaria tiene {valor} filas en la prueba: con menos de {opciones} "
              "ninguna conclusión se sostiene sola",
        "en": "the minority class has {valor} rows in the test: with fewer than {opciones} "
              "no conclusion stands on its own",
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
            f"{clave!r} no está en el catálogo de textos del 141: escribir aquí una frase suelta "
            "dejaría media pantalla sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(
        **{k: (v[idioma] if isinstance(v, PorIdioma) else v) for k, v in campos.items()})
        for idioma in IDIOMAS}
