# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""141-C1 — las curvas de una evaluación: ROC, PR, fiabilidad, matriz de confusión y, en regresión,
el predicho frente al real por tramos. NÚCLEO: una función pura sobre una `Muestra`, sin disco.

QUÉ HACE Y QUÉ NO. `curvas_de_la_evaluacion(muestra, ...)` recibe la `Muestra` de la PRUEBA RESERVADA
(la de la evaluación final, medida una sola vez con el modelo entregado) y devuelve un `dict` JSON
serializable y versionado. Todo lo que contiene es un AGREGADO: puntos de curva, recuentos por tramo,
medias. Ninguna fila, ningún id (D1 del 141). No lee el test, no lo registra, no decide nada: quien
la llama ya tiene la muestra delante, y lo que devuelve solo informa.

LAS DECISIONES DE ROBERTO (28-10) QUE ESTO IMPLEMENTA
  * D2 — matriz de confusión AL UMBRAL ELEGIDO en binaria (el de la política de decisión, sobre la
    escala de decisión) y AL ARGMAX en multiclase.
  * D3 — multiclase: fiabilidad de la clase PREDICHA (confianza frente a acierto) y su ECE, SIN
    recalibrar. El bloque lo dice (`recalibracion.clave = multiclase_sin_recalibrar`).
  * D4 — regresión: predicho frente a real POR TRAMOS (deciles del predicho: media real y
    dispersión por tramo).
  * D5 — un tramo/bin con menos de `MINIMO_FILAS_POR_SEGMENTO` (30) filas se da SIN cifra y con su
    motivo. Vale para los bins de la fiabilidad y los tramos de regresión. El tamaño del tramo (`n`)
    sí viaja: es un recuento, no una estimación.

LA ESCALA. En binaria, ROC, PR, matriz y fiabilidad «después» viven en la ESCALA DE LA DECISIÓN: la
probabilidad RECALIBRADA si el estudio ajustó una recalibración que se pudo aplicar, la del modelo
tal cual si no. La fiabilidad CRUDA solo se enseña como «antes» y solo cuando hubo recalibración:
no se mezclan en una misma curva. Si el calibrador es `None` (o no se pudo ajustar) el bloque lo
dice con su clave; jamás se dibuja como si se hubiera hecho.

UN VALOR AUSENTE NO ES UN CERO. Cada bloque es `{"disponible", "clave", "motivo", ...}`: o trae
datos y `motivo = None`, o trae `disponible = False`, la CLAVE del motivo y su redacción en los dos
idiomas (`curvas_de_la_evaluacion_textos.py`: lo que redacta el core se traduce en el core).

QUÉ SE REUTILIZA (para que no haya dos sitios contando lo mismo): los puntos de la curva exacta
salen de `metricas._curva_binaria`/`_puntos_pr`, el adelgazado de vértices de
`evaluacion_de_puntuacion._vertices_elegidos` con el mismo tope que la ROC del informe de
puntuación, los bins y el ECE de `calibracion.curva_de_fiabilidad`, la matriz de
`metricas.matriz_de_confusion` y el mínimo de `segmentos.MINIMO_FILAS_POR_SEGMENTO`. La ROC del 137
(`evaluacion_de_puntuacion._curva_roc`) tiene la clase positiva fija en «evento» y no sirve para una
clase cualquiera, así que aquí se recalcula con la misma lógica y una prueba exige que coincidan.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Sequence

from matrixai.estudio.calibracion import (
    METODOS_DE_BINS,
    CurvaDeFiabilidad,
    RecalibracionLogistica,
    aplicar_recalibracion,
    curva_de_fiabilidad,
)
from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.curvas_de_la_evaluacion_textos import PorIdioma, motivo
from matrixai.estudio.evaluacion_de_puntuacion import PUNTOS_DE_LA_CURVA_ROC, _vertices_elegidos
from matrixai.estudio.metricas import (
    UMBRAL_POR_DEFECTO,
    EntradaNoMedible,
    MatrizDeConfusion,
    Muestra,
    _argmax,
    _auroc,
    _average_precision,
    _curva_binaria,
    _puntos_pr,
    matriz_de_confusion,
)
from matrixai.estudio.segmentos import MINIMO_EVENTOS_POR_SEGMENTO, MINIMO_FILAS_POR_SEGMENTO
from matrixai.estudio.validacion import exigir_entero, exigir_real
from matrixai.estudio.vocabulario import exigir_opcion

__all__ = [
    "PUNTOS_DE_LA_CURVA_PR",
    "VERSION_DE_LAS_CURVAS",
    "curva_pr",
    "curva_roc",
    "curvas_de_la_evaluacion",
]

#: La versión del documento. Cambia con la FORMA (claves) o con la regla de una cifra.
VERSION_DE_LAS_CURVAS = "1.0.0"

#: Cuántos puntos de la curva PR viajan: los extremos y, como mucho, 38 vértices reales más. El mismo
#: tope que la ROC (`PUNTOS_DE_LA_CURVA_ROC`), a propósito: dos curvas del mismo informe, mismo peso.
PUNTOS_DE_LA_CURVA_PR = 40

_TAREA_EN_TEXTO = {
    "binary_classification": PorIdioma({"es": "clasificación binaria", "en": "binary classification"}),
    "multiclass_classification": PorIdioma({"es": "clasificación multiclase", "en": "multiclass classification"}),
    "regression": PorIdioma({"es": "regresión", "en": "regression"}),
}
_CURVA_EN_TEXTO = {
    "roc": PorIdioma({"es": "la curva ROC", "en": "the ROC curve"}),
    "pr": PorIdioma({"es": "la curva de precisión-exhaustividad", "en": "the precision-recall curve"}),
    "fiabilidad": PorIdioma({"es": "el diagrama de fiabilidad", "en": "the reliability diagram"}),
    "matriz": PorIdioma({"es": "la matriz de confusión", "en": "the confusion matrix"}),
    "tramos": PorIdioma({"es": "el predicho frente al real por tramos",
                         "en": "predicted against actual by bins"}),
}

_ACIERTO, _FALLO = "acierto", "fallo"


# ---------------------------------------------------------------------------
# Bloques: o hay datos, o hay una clave que dice por qué no
# ---------------------------------------------------------------------------

def _no(clave: str, **campos: object) -> dict[str, Any]:
    return {"disponible": False, "clave": clave, "motivo": motivo(clave, **campos)}


def _si(**datos: Any) -> dict[str, Any]:
    return {"disponible": True, "clave": None, "motivo": None, **datos}


def _no_aplica(curva: str, tarea: str) -> dict[str, Any]:
    return _no("no_aplica_a_la_tarea", curva=_CURVA_EN_TEXTO[curva], tarea=_TAREA_EN_TEXTO[tarea])


# ---------------------------------------------------------------------------
# ROC y PR: los vértices REALES de la curva exacta
# ---------------------------------------------------------------------------

def curva_roc(muestra: Muestra, *, max_puntos: int = PUNTOS_DE_LA_CURVA_ROC) -> list[dict[str, float]] | None:
    """La curva ROC de una muestra binaria, adelgazada a `max_puntos` vértices REALES.

    Un vértice por valor DISTINTO de puntuación (los empates se cierran juntos), de mayor a menor
    puntuación; (0,0) y (1,1) siempre están. Con más vértices que el tope se eligen los más cercanos
    a repartos iguales de la longitud de la curva (`_vertices_elegidos`): un subconjunto de la curva
    exacta, nunca un punto interpolado. `None` si falta una de las dos clases (lo mismo que dice el
    AUROC) o si la muestra no trae puntuación."""
    exigir_entero(max_puntos, "max_puntos", minimo=2)
    if muestra.task != "binary_classification" or muestra.puntuacion_del_positivo is None:
        return None
    puntos, tp_total, fp_total = _curva_binaria(muestra)
    if tp_total == 0.0 or fp_total == 0.0:
        return None
    vertices = [(0.0, 0.0)] + [(fp / fp_total, tp / tp_total) for tp, fp in puntos]
    if len(vertices) > max_puntos:
        vertices = [vertices[i] for i in _vertices_elegidos(vertices, max_puntos)]
    return [{"fpr": x, "tpr": y} for x, y in vertices]


def curva_pr(muestra: Muestra, *, max_puntos: int = PUNTOS_DE_LA_CURVA_PR) -> list[dict[str, float]] | None:
    """La curva de precisión-exhaustividad de una muestra binaria, adelgazada a `max_puntos` vértices.

    Un punto `(recall, precision)` por valor DISTINTO de puntuación, de mayor a menor puntuación
    (recall creciente): son exactamente los de `precision_recall_curve` de sklearn salvo el ancla
    `(0, 1)` que sklearn añade, y que aquí NO se inventa (ningún umbral la alcanza). Mismo adelgazado
    que la ROC. `None` si no hay positivos o la muestra no trae puntuación."""
    exigir_entero(max_puntos, "max_puntos", minimo=2)
    if muestra.task != "binary_classification" or muestra.puntuacion_del_positivo is None:
        return None
    puntos, tp_total, _ = _curva_binaria(muestra)
    if tp_total == 0.0:
        return None
    vertices = _puntos_pr(puntos, tp_total)
    if len(vertices) > max_puntos:
        vertices = [vertices[i] for i in _vertices_elegidos(vertices, max_puntos)]
    return [{"recall": r, "precision": p} for r, p in vertices]


# ---------------------------------------------------------------------------
# La matriz de confusión, con el punto de operación que deja en las curvas
# ---------------------------------------------------------------------------

def _razon(numerador: int, denominador: int) -> float | None:
    return numerador / denominador if denominador else None


def _matriz_a_json(matriz: MatrizDeConfusion, n: int, umbral_dado: bool) -> dict[str, Any]:
    datos = matriz.a_json()
    binaria = matriz.positive_label is not None and len(matriz.classes) == 2
    datos["filas"] = n
    datos["casillas"] = None
    datos["umbral_declarado"] = None
    datos["aviso"] = None
    if binaria:
        negativa = next(c for c in matriz.classes if c != matriz.positive_label)
        datos["casillas"] = {"tp": matriz.tp, "fn": matriz.fn, "fp": matriz.fp, "tn": matriz.tn,
                             "clase_positiva": matriz.positive_label, "clase_negativa": negativa}
        if matriz.decision_rule == "threshold":
            datos["umbral_declarado"] = umbral_dado
            if not umbral_dado:
                datos["aviso"] = {"clave": "umbral_por_omision",
                                  "motivo": motivo("umbral_por_omision", valor=PorIdioma({
                                      "es": f"{matriz.threshold:g}".replace(".", ","),
                                      "en": f"{matriz.threshold:g}"}))}
    return _si(**datos)


def _punto_de_operacion(matriz: MatrizDeConfusion) -> dict[str, Any] | None:
    """Dónde cae el umbral elegido sobre la ROC y la PR. `None` en cada cifra que no se puede
    calcular (sin positivos no hay exhaustividad; sin positivos predichos no hay precisión)."""
    if matriz.positive_label is None or len(matriz.classes) != 2:
        return None
    tp, fn, fp, tn = matriz.tp, matriz.fn, matriz.fp, matriz.tn
    return {"umbral": matriz.threshold, "regla": matriz.decision_rule,
            "fpr": _razon(fp, fp + tn), "tpr": _razon(tp, tp + fn),
            "recall": _razon(tp, tp + fn), "precision": _razon(tp, tp + fp)}


# ---------------------------------------------------------------------------
# Fiabilidad: bins con el mínimo de filas
# ---------------------------------------------------------------------------

def _bin_a_json(b, minimo: int) -> dict[str, Any]:
    base = {"desde": b.limite_inferior, "hasta": b.limite_superior, "n": b.n}
    if b.n == 0:
        return {**base, "eventos": None, "probabilidad_media": None, "tasa_observada": None,
                "clave": "tramo_vacio", "motivo": motivo("tramo_vacio")}
    if b.n < minimo:
        return {**base, "eventos": None, "probabilidad_media": None, "tasa_observada": None,
                "clave": "tramo_con_pocas_filas",
                "motivo": motivo("tramo_con_pocas_filas", valor=b.n, opciones=minimo)}
    return {**base, "eventos": b.eventos, "probabilidad_media": b.probabilidad_media_predicha,
            "tasa_observada": b.tasa_observada, "clave": None, "motivo": None}


def _curva_a_json(curva: CurvaDeFiabilidad, n: int, minimo: int) -> dict[str, Any]:
    """Los bins con la regla de D5 y el ECE. El ECE se calcula con TODAS las filas (un bin pequeño
    pesa poco en él, justamente), pero con una prueba entera por debajo del mínimo no se da."""
    ece, ece_clave = curva.ece, None
    if curva.ece is not None and n < minimo:
        ece, ece_clave = None, "ece_con_pocas_filas"
    return {"metodo_de_bins": curva.metodo_de_bins, "n_bins": curva.n_bins_solicitados,
            "ece": ece,
            "ece_clave": ece_clave,
            "ece_motivo": motivo(ece_clave, valor=n, opciones=minimo) if ece_clave else None,
            "no_definido": curva.undefined_reason,
            "bins": [_bin_a_json(b, minimo) for b in curva.bins]}


# ---------------------------------------------------------------------------
# Regresión: deciles del predicho
# ---------------------------------------------------------------------------

def _desviacion(valores: Sequence[float], media: float) -> float | None:
    """La desviación típica MUESTRAL (n-1) en dos pasadas; `None` con una sola fila."""
    if len(valores) < 2:
        return None
    return math.sqrt(math.fsum((v - media) ** 2 for v in valores) / (len(valores) - 1))


def _tramos_del_predicho(muestra: Muestra, n_tramos: int, minimo: int) -> dict[str, Any]:
    """Tramos por cuantiles del PREDICHO. Los cortes son valores predichos reales (el que está en la
    posición `round(i·n/k)` de los ordenados) y un valor repetido va entero a un solo tramo, así que
    con predicciones muy repetidas salen MENOS tramos que los pedidos (`n_tramos` lo dice) en vez de
    tramos que se solapan o quedan vacíos. Tramo `[desde, hasta)`; el último incluye el máximo."""
    predichos = [float(p) for p in muestra.predictions]
    reales = [float(y) for y in muestra.y_true]
    n = len(predichos)
    ordenados = sorted(predichos)
    cortes: list[float] = []
    for i in range(1, n_tramos):
        c = ordenados[min(n - 1, round(i * n / n_tramos))]
        if c > ordenados[0] and (not cortes or c > cortes[-1]):
            cortes.append(c)
    grupos: list[list[int]] = [[] for _ in range(len(cortes) + 1)]
    for i, p in enumerate(predichos):
        grupos[bisect.bisect_right(cortes, p)].append(i)
    limites = [ordenados[0], *cortes, ordenados[-1]]
    tramos = []
    for j, idx in enumerate(grupos):
        base = {"desde": limites[j], "hasta": limites[j + 1], "n": len(idx)}
        if len(idx) < minimo:
            tramos.append({**base, "predicho_medio": None, "real_medio": None, "real_desviacion": None,
                           "clave": "tramo_con_pocas_filas",
                           "motivo": motivo("tramo_con_pocas_filas", valor=len(idx), opciones=minimo)})
            continue
        real_medio = math.fsum(reales[i] for i in idx) / len(idx)
        tramos.append({**base,
                       "predicho_medio": math.fsum(predichos[i] for i in idx) / len(idx),
                       "real_medio": real_medio,
                       "real_desviacion": _desviacion([reales[i] for i in idx], real_medio),
                       "clave": None, "motivo": None})
    return _si(n_tramos_pedidos=n_tramos, n_tramos=len(tramos), desviacion="muestral_n_menos_1",
               tramos=tramos)


# ---------------------------------------------------------------------------
# La prueba al lado de cada curva
# ---------------------------------------------------------------------------

def _la_prueba(muestra: Muestra, minimo: int) -> dict[str, Any]:
    n = muestra.n
    positivos = negativos = prevalencia = None
    por_clase = None
    cautela: dict[str, Any] | None = None
    if muestra.task in ("binary_classification", "multiclass_classification"):
        por_clase = {c: sum(1 for y in muestra.y_true if y == c) for c in muestra.classes}
    if muestra.task == "binary_classification":
        positivos = por_clase[muestra.positive_label]
        negativos = n - positivos
        prevalencia = _razon(positivos, n)
    if n < minimo:
        cautela = {"clave": "prueba_con_pocas_filas", "motivo": motivo(
            "prueba_con_pocas_filas", valor=n, opciones=minimo)}
    elif positivos is not None and min(positivos, negativos) < MINIMO_EVENTOS_POR_SEGMENTO:
        cautela = {"clave": "prueba_con_pocos_eventos", "motivo": motivo(
            "prueba_con_pocos_eventos", valor=min(positivos, negativos),
            opciones=MINIMO_EVENTOS_POR_SEGMENTO)}
    return {"filas": n, "positivos": positivos, "negativos": negativos, "prevalencia": prevalencia,
            "clase_positiva": muestra.positive_label, "por_clase": por_clase,
            "minimo_de_filas": minimo, "cautela": cautela}


# ---------------------------------------------------------------------------
# La función pública
# ---------------------------------------------------------------------------

def curvas_de_la_evaluacion(muestra: Muestra, *, recalibracion: RecalibracionLogistica | None = None,
                            umbral: float | None = None, n_bins: int = 10,
                            metodo_de_bins: str = "ancho_igual", n_tramos: int = 10,
                            minimo_de_filas: int = MINIMO_FILAS_POR_SEGMENTO) -> dict[str, Any]:
    """Las curvas de UNA evaluación, como un `dict` JSON serializable.

    `muestra` es la de la PRUEBA RESERVADA con la salida CRUDA del modelo (en binaria, la
    `probabilidad_del_positivo` sin recalibrar; en multiclase, la distribución; en regresión, el
    predicho). `recalibracion` es el calibrador del estudio, sin aplicar (`None`, o uno que no se
    pudo ajustar, son dos claves distintas); esta función lo aplica y es quien separa el «antes» del
    «después». `umbral` es el elegido, en la escala de decisión (la recalibrada si la hay); sin él la
    matriz binaria usa 0,5 y LO DICE (`matriz.aviso`). Sin pesos: los recuentos son filas enteras.
    """
    if muestra.weights is not None:
        raise EntradaNoMedible("pesos_no_admitidos", campo="curvas_de_la_evaluacion")
    exigir_entero(n_bins, "n_bins", minimo=1)
    exigir_opcion(metodo_de_bins, "metodo_de_bins", METODOS_DE_BINS)
    exigir_entero(n_tramos, "n_tramos", minimo=1)
    exigir_entero(minimo_de_filas, "minimo_de_filas", minimo=1)
    if umbral is not None:
        exigir_real(umbral, "umbral")
    if recalibracion is not None and not isinstance(recalibracion, RecalibracionLogistica):
        raise EsquemaInvalido("no_es_del_vocabulario", campo="recalibracion",
                              opciones=["RecalibracionLogistica", "None"], valor=repr(type(recalibracion)))

    salida: dict[str, Any] = {"version": VERSION_DE_LAS_CURVAS, "tarea": muestra.task,
                              "prueba": _la_prueba(muestra, minimo_de_filas)}
    if muestra.n == 0:
        for bloque in ("roc", "pr", "fiabilidad", "matriz", "tramos"):
            salida[bloque] = _no("prueba_vacia")
        salida["recalibracion"] = {**_no("prueba_vacia"), "aplicada": False, "metodo": None}
        salida["escala"] = None
        return salida

    if muestra.task == "binary_classification":
        _binaria(muestra, salida, recalibracion, umbral, n_bins, metodo_de_bins, minimo_de_filas)
    elif muestra.task == "multiclass_classification":
        _multiclase(muestra, salida, n_bins, metodo_de_bins, minimo_de_filas)
    else:
        _regresion(muestra, salida, n_tramos, minimo_de_filas)
    return salida


def _binaria(muestra: Muestra, salida: dict[str, Any], recalibracion: RecalibracionLogistica | None,
             umbral: float | None, n_bins: int, metodo_de_bins: str, minimo: int) -> None:
    n = muestra.n
    prueba = salida["prueba"]
    hay_probabilidades = muestra.probabilidad_del_positivo
    aplicable = (recalibracion is not None and recalibracion.a is not None
                 and recalibracion.b is not None and hay_probabilidades is not None)

    # -- la muestra EN LA ESCALA DE LA DECISIÓN ------------------------------------------------------
    decision = muestra
    if aplicable:
        decision = Muestra.binaria(
            muestra.y_true, classes=muestra.classes, positive_label=muestra.positive_label,
            probabilidades=aplicar_recalibracion(recalibracion, hay_probabilidades),
            predicciones=muestra.predictions, unidades=muestra.units)
    if hay_probabilidades is not None:
        salida["escala"] = "probabilidad_recalibrada" if aplicable else "probabilidad_del_modelo"
    else:
        salida["escala"] = "puntuacion_cruda" if decision.puntuacion_del_positivo is not None else None

    # -- la recalibración, dicha ---------------------------------------------------------------------
    if aplicable:
        salida["recalibracion"] = _si(aplicada=True, metodo=recalibracion.metodo)
    elif recalibracion is None:
        salida["recalibracion"] = {**_no("sin_recalibracion"), "aplicada": False, "metodo": None}
    elif recalibracion.a is None or recalibracion.b is None:
        razon = PorIdioma(recalibracion.undefined_reason or {"es": "motivo no declarado",
                                                              "en": "reason not declared"})
        salida["recalibracion"] = {**_no("recalibracion_no_ajustada", valor=razon), "aplicada": False,
                                   "metodo": recalibracion.metodo}
    else:
        salida["recalibracion"] = {**_no("recalibracion_sin_probabilidades"), "aplicada": False,
                                   "metodo": recalibracion.metodo}

    # -- la matriz primero: el punto de operación cuelga de ella -------------------------------------
    matriz = None
    if decision.predictions is None and decision.probabilities is None and umbral is None:
        salida["matriz"] = _no("matriz_sin_umbral_en_puntuacion")
    else:
        matriz = matriz_de_confusion(decision, umbral=umbral)
        salida["matriz"] = _matriz_a_json(matriz, n, umbral is not None)
    operacion = _punto_de_operacion(matriz) if matriz is not None else None

    # -- ROC y PR ------------------------------------------------------------------------------------
    positivos, negativos = prueba["positivos"], prueba["negativos"]
    if decision.puntuacion_del_positivo is None:
        salida["roc"] = _no("sin_puntuacion")
        salida["pr"] = _no("sin_puntuacion")
    else:
        if not positivos or not negativos:
            salida["roc"] = _no("roc_sin_las_dos_clases", positivos=positivos, negativos=negativos)
        else:
            puntos = curva_roc(decision)
            salida["roc"] = _si(puntos=puntos, tope_de_puntos=PUNTOS_DE_LA_CURVA_ROC,
                                area=_auroc(decision, None),
                                punto_de_operacion=(
                                    {"umbral": operacion["umbral"], "regla": operacion["regla"],
                                     "fpr": operacion["fpr"], "tpr": operacion["tpr"]}
                                    if operacion else None))
        if not positivos:
            salida["pr"] = _no("pr_sin_positivos", positivos=positivos)
        else:
            salida["pr"] = _si(puntos=curva_pr(decision), tope_de_puntos=PUNTOS_DE_LA_CURVA_PR,
                               area=_average_precision(decision, None),
                               area_metodo="average_precision", prevalencia=prueba["prevalencia"],
                               punto_de_operacion=(
                                   {"umbral": operacion["umbral"], "regla": operacion["regla"],
                                    "recall": operacion["recall"], "precision": operacion["precision"]}
                                   if operacion else None))

    # -- fiabilidad: «después» en la escala de la decisión, «antes» en la cruda ---------------------
    if decision.probabilidad_del_positivo is None:
        salida["fiabilidad"] = _no("sin_probabilidades")
    else:
        despues = curva_de_fiabilidad(decision, n_bins=n_bins, metodo_de_bins=metodo_de_bins)
        if aplicable:
            antes = _si(escala="probabilidad_del_modelo",
                        curva=_curva_a_json(curva_de_fiabilidad(
                            muestra, n_bins=n_bins, metodo_de_bins=metodo_de_bins), n, minimo))
        else:
            antes = {k: salida["recalibracion"][k] for k in ("disponible", "clave", "motivo")}
        salida["fiabilidad"] = _si(tipo="clase_positiva", escala=salida["escala"],
                                   curva=_curva_a_json(despues, n, minimo), antes=antes, resumen=None)
    salida["tramos"] = _no_aplica("tramos", muestra.task)


def _multiclase(muestra: Muestra, salida: dict[str, Any], n_bins: int, metodo_de_bins: str,
                minimo: int) -> None:
    n = muestra.n
    salida["roc"] = _no_aplica("roc", muestra.task)
    salida["pr"] = _no_aplica("pr", muestra.task)
    salida["tramos"] = _no_aplica("tramos", muestra.task)
    salida["recalibracion"] = {**_no("multiclase_sin_recalibrar"), "aplicada": False, "metodo": None}
    salida["matriz"] = _matriz_a_json(matriz_de_confusion(muestra), n, False)
    if muestra.probabilities is None:
        salida["escala"] = None
        salida["fiabilidad"] = _no("sin_probabilidades")
        return
    salida["escala"] = "confianza_de_la_clase_predicha"
    clases = muestra.classes
    predichas = (list(muestra.predictions) if muestra.predictions is not None
                 else [clases[_argmax(fila)] for fila in muestra.probabilities])
    confianza = [fila[clases.index(p)] for fila, p in zip(muestra.probabilities, predichas)]
    acierto = [_ACIERTO if p == y else _FALLO for p, y in zip(predichas, muestra.y_true)]
    como_binaria = Muestra.binaria(acierto, classes=(_FALLO, _ACIERTO), positive_label=_ACIERTO,
                                   probabilidades=confianza)
    curva = curva_de_fiabilidad(como_binaria, n_bins=n_bins, metodo_de_bins=metodo_de_bins)
    aciertos = sum(1 for a in acierto if a == _ACIERTO)
    salida["fiabilidad"] = _si(
        tipo="clase_predicha", escala=salida["escala"],
        curva=_curva_a_json(curva, n, minimo),
        antes={k: salida["recalibracion"][k] for k in ("disponible", "clave", "motivo")},
        resumen={"confianza_media": math.fsum(confianza) / n, "tasa_de_acierto": aciertos / n,
                 "aciertos": aciertos})


def _regresion(muestra: Muestra, salida: dict[str, Any], n_tramos: int, minimo: int) -> None:
    for bloque in ("roc", "pr", "fiabilidad", "matriz"):
        salida[bloque] = _no_aplica(bloque, muestra.task)
    salida["recalibracion"] = {**_no_aplica("fiabilidad", muestra.task), "aplicada": False, "metodo": None}
    # `Muestra` rechaza una regresión con filas y sin `predictions`, así que aquí siempre hay predicho.
    salida["escala"] = "valor_predicho"
    salida["tramos"] = _tramos_del_predicho(muestra, n_tramos, minimo)
