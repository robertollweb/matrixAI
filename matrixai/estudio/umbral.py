# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C4 — calibrador y umbral: elegir DÓNDE cortar sobre la escala
calibrada minimizando el coste declarado del error, sujeto a las
restricciones obligatorias del problema — nunca al revés (un umbral que
minimiza coste ignorando una restricción de sensibilidad mínima no sirve
aunque sea barato).

BINARIA, DE ENTRADA. El contrato lo dice literal: "binaria inicial". Pedir
un umbral sobre multiclase o regresión rechaza con `EntradaNoMedible` — no
se inventa un corte binario donde no lo hay, ni se aproxima con un umbral
"probable".

LA ESCALA YA LA DECLARA `Muestra`, NO SE RECALCULA AQUÍ. `Muestra.
escala_de_decision` (104-C0) ya distingue `calibrated_probability` (trae
`probabilities`) de `raw_score` (solo `scores`) — la `PoliticaDeDecision`
que este módulo produce usa ese mismo valor, sin una segunda regla que
pudiera divergir de la primera con el tiempo. Aplicar un `recalibracion`
(105-C3, ya AJUSTADO en una partición distinta a la que aquí se busca el
umbral — esa disciplina de separar ajuste y evaluación es del llamante,
igual que ya lo era en 105-C3) reconstruye la muestra con la probabilidad
recalibrada antes de buscar, así que el umbral vive exactamente en la
escala que `FittedPipelineSpec` declarará junto al calibrador.

EL COSTE ES EL REALIZADO SOBRE ESTA MUESTRA, NO UNA FÓRMULA POBLACIONAL
APARTE. El coste esperado de cada umbral candidato es
`(fp*cost_fp + fn*cost_fn) / n` sobre sus recuentos — el mismo criterio con
el que se evaluaría después, no una aproximación teórica distinta que
podría no coincidir con lo medido.

BARRIDO EN `O(n log n)`, NO `O(n²)`. La primera versión de este módulo
recalculaba `matriz_de_confusion` entera (`O(n)`) para cada umbral
candidato — con hasta `n` candidatos distintos, `O(n²)`: medido, 8000 filas
tardaban 11-22 s por llamada, y una tabla de 100 000 filas habría tardado
del orden de media hora. Se sustituyó por un barrido: ordenar una vez por
probabilidad y acumular TP/FP/TN/FN según el umbral baja, cruzando cada
fila UNA sola vez. Verificado contra `matriz_de_confusion` (la fuente de
verdad de 105-C1) en varios umbrales de una muestra pequeña antes de
confiar en el barrido — la técnica es la misma del barrido de una curva
ROC, no una invención de este módulo.

RESTRICCIONES SOBRE SENSIBILIDAD/ESPECIFICIDAD/PPV/NPV SE DERIVAN DEL
MISMO BARRIDO — sin llamar a `calcular()` por candidato, que reintroduciría
el coste `O(n)` que el barrido existe para evitar. Una restricción sobre
CUALQUIER OTRA métrica (declarada pero no derivable de TP/FP/TN/FN) se
rechaza como error de cableado: este corte no promete optimizar bajo una
restricción que no depende del umbral en absoluto.

QUÉ NO HACE ESTE MÓDULO. No genera una versión nueva de política ni marca
`test_used_for_development` cuando un umbral se cambia tras mirar el test
— ese mecanismo YA EXISTE (`version_tras_aprender_del_test`, 104-C0/104-C5)
y es de quien orquesta el ciclo de vida del pipeline, no de esta función
pura que solo busca un número dado un conjunto de filas.
"""

from __future__ import annotations

from typing import Sequence

from matrixai.estudio.calibracion import RecalibracionLogistica, aplicar_recalibracion
from matrixai.estudio.esquemas import PoliticaDeDecision, Restriccion
from matrixai.estudio.metricas import EntradaNoMedible, Muestra

__all__ = ["elegir_umbral"]

#: Las cuatro métricas que se derivan directamente de TP/FP/TN/FN — las
#: únicas que el barrido puede comprobar SIN volver a recorrer la muestra
#: por candidato. Cualquier otra restricción se rechaza (error de
#: cableado): no depende del umbral, así que "optimizarla" no significa nada.
_METRICAS_DE_CONFUSION = ("sensitivity", "specificity", "ppv", "npv")


def _valor_de_confusion(clave: str, tp: int, fp: int, tn: int, fn: int) -> float | None:
    if clave == "sensitivity":
        return tp / (tp + fn) if (tp + fn) > 0 else None
    if clave == "specificity":
        return tn / (tn + fp) if (tn + fp) > 0 else None
    if clave == "ppv":
        return tp / (tp + fp) if (tp + fp) > 0 else None
    return tn / (tn + fn) if (tn + fn) > 0 else None  # npv


def _cumple(restriccion: Restriccion, tp: int, fp: int, tn: int, fn: int) -> bool:
    valor = _valor_de_confusion(restriccion.clave, tp, fp, tn, fn)
    if valor is None:
        return False
    if restriccion.operador == "min":
        return valor >= restriccion.valor
    if restriccion.operador == "max":
        return valor <= restriccion.valor
    if restriccion.operador == "equal":
        return valor == restriccion.valor
    return False  # "boolean" no es coherente con una métrica continua: falla cerrado


def elegir_umbral(muestra: Muestra, *, cost_false_positive: float, cost_false_negative: float,
                  restricciones: Sequence[Restriccion] = (),
                  recalibracion: RecalibracionLogistica | None = None) -> PoliticaDeDecision:
    """`muestra` es la partición de CALIBRACIÓN/desarrollo — nunca test, y
    estructuralmente no podría serlo: esta función no recibe ningún dato de
    test, así que cambiarlo no puede cambiar el umbral elegido aquí.

    Si `recalibracion` viene dada, tiene que estar ya AJUSTADA sobre una
    partición distinta (el llamante lo garantiza, igual que en 105-C3)."""
    if muestra.task != "binary_classification":
        raise EntradaNoMedible("metrica_de_otra_tarea", campo="umbral",
                               opciones=["binary_classification"], valor=muestra.task)

    probabilidades = muestra.probabilidad_del_positivo
    if probabilidades is None:
        raise EntradaNoMedible("umbral_sin_probabilidades", campo="umbral")

    restricciones_obligatorias = [r for r in restricciones if r.obligatoria]
    for restriccion in restricciones_obligatorias:
        if restriccion.clave not in _METRICAS_DE_CONFUSION:
            raise EntradaNoMedible("restriccion_no_derivable_del_umbral", campo=restriccion.clave)

    if recalibracion is not None:
        probabilidades = aplicar_recalibracion(recalibracion, probabilidades)

    muestra_de_busqueda = Muestra.binaria(
        y_true=muestra.y_true, classes=muestra.classes, positive_label=muestra.positive_label,
        probabilidades=probabilidades)

    n = muestra_de_busqueda.n
    es_positivo = tuple(y == muestra.positive_label for y in muestra_de_busqueda.y_true)
    probs = muestra_de_busqueda.probabilidad_del_positivo
    pares = sorted(zip(probs, es_positivo))  # ascendente por probabilidad
    total_positivos = sum(es_positivo)
    total_negativos = n - total_positivos

    # Descendente: en cada paso se añaden a "predicho positivo" las filas
    # cuya probabilidad ya alcanza el nuevo umbral, sin revisar las de antes.
    candidatos = sorted(set(probs) | {0.0, 1.0}, reverse=True)
    tp = fp = 0
    indice = n - 1
    mejor_umbral: float | None = None
    mejor_coste: float | None = None
    for t in candidatos:
        while indice >= 0 and pares[indice][0] >= t:
            if pares[indice][1]:
                tp += 1
            else:
                fp += 1
            indice -= 1
        fn = total_positivos - tp
        tn = total_negativos - fp
        if any(not _cumple(r, tp, fp, tn, fn) for r in restricciones_obligatorias):
            continue
        coste = (fp * cost_false_positive + fn * cost_false_negative) / n
        if mejor_coste is None or coste < mejor_coste:
            mejor_coste = coste
            mejor_umbral = t

    if mejor_umbral is None:
        raise EntradaNoMedible("no_hay_umbral_que_cumpla", valor=len(candidatos))

    return PoliticaDeDecision(
        threshold=mejor_umbral, scale=muestra_de_busqueda.escala_de_decision,
        positive_label=muestra.positive_label,
        cost_false_positive=cost_false_positive, cost_false_negative=cost_false_negative)
