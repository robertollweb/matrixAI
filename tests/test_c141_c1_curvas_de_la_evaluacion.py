# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""141-C1 — las curvas de una evaluación (núcleo).

Oráculo: `sklearn` (`roc_curve`, `roc_auc_score`, `precision_recall_curve`,
`average_precision_score`, `calibration_curve`, `confusion_matrix`) y `numpy` para los tramos de
regresión. Los datos tienen la forma que trae la gente: clases como texto con tildes y mayúsculas
(`Sí/No`, `ENFERMO`, `Árbol`), `0/1`, la clase positiva primera o última en `classes`, empates de
probabilidad, multiclase de 3 a 5 clases con una AUSENTE de la prueba, regresión con valores
grandes y negativos. Cada prueba afirma la CIFRA (contra el oráculo), no que «no falla».
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sk = pytest.importorskip("sklearn.metrics")
from sklearn.calibration import calibration_curve  # noqa: E402

from matrixai.estudio import curvas_de_la_evaluacion as modulo  # noqa: E402
from matrixai.estudio.calibracion import (  # noqa: E402
    RecalibracionLogistica,
    ajustar_recalibracion_logistica,
)
from matrixai.estudio.curvas_de_la_evaluacion import (  # noqa: E402
    PUNTOS_DE_LA_CURVA_PR,
    VERSION_DE_LAS_CURVAS,
    curva_pr,
    curva_roc,
    curvas_de_la_evaluacion,
)
from matrixai.estudio.curvas_de_la_evaluacion_textos import (  # noqa: E402
    IDIOMAS,
    MOTIVOS,
    huecos_de,
    motivo,
)
from matrixai.estudio.errores import EsquemaInvalido  # noqa: E402
from matrixai.estudio.evaluacion_de_puntuacion import (  # noqa: E402
    PUNTOS_DE_LA_CURVA_ROC,
    _curva_roc,
)
from matrixai.estudio.metricas import EntradaNoMedible, Muestra  # noqa: E402
from matrixai.estudio.segmentos import MINIMO_FILAS_POR_SEGMENTO  # noqa: E402

#: (negativa, positiva): la forma de las etiquetas que trae la gente.
ETIQUETAS = [("No", "Sí"), ("0", "1"), ("sano", "ENFERMO"), ("Negativo", "Positivo"), ("ñoño", "Árbol")]
CLASES_MULTI = [
    ("Árbol", "Ñandú", "Cañón"),
    ("A", "b", "ÉXITO", "ño"),
    ("Rojo", "verde", "AZUL", "Ámbar", "0"),
]


def _binaria(n, semilla, neg="No", pos="Sí", *, decimales=None, positiva_primero=False, senal=1.0):
    rng = random.Random(semilla)
    p = [min(max(rng.random() ** senal, 1e-6), 1 - 1e-6) for _ in range(n)]
    if decimales is not None:
        p = [min(max(round(q, decimales), 0.0), 1.0) for q in p]
    y = [pos if rng.random() < q else neg for q in p]
    clases = (pos, neg) if positiva_primero else (neg, pos)
    return Muestra.binaria(y, classes=clases, positive_label=pos, probabilidades=p)


def _multi(n, semilla, clases, *, ausente=None):
    rng = random.Random(semilla)
    k = len(clases)
    filas, y = [], []
    candidatas = [c for c in clases if c != ausente]
    for _ in range(n):
        crudo = [rng.random() ** 2 + 0.01 for _ in range(k)]
        s = sum(crudo)
        fila = tuple(v / s for v in crudo)
        verdad = rng.choice(candidatas)
        if rng.random() < 0.6:                        # el modelo acierta bastante
            j = clases.index(verdad)
            fila = tuple(0.5 * v + (0.5 if i == j else 0.0) for i, v in enumerate(fila))
        filas.append(fila)
        y.append(verdad)
    return Muestra.multiclase(y, classes=clases, probabilidades=filas)


def _regresion(n, semilla, *, escala=1.0, desplazamiento=0.0):
    rng = random.Random(semilla)
    pred = [desplazamiento + escala * rng.gauss(0, 1) for _ in range(n)]
    real = [v + escala * rng.gauss(0, 0.5) for v in pred]
    return Muestra(task="regression", y_true=tuple(real), predictions=tuple(pred))


def _y01(m):
    return [1 if y == m.positive_label else 0 for y in m.y_true]


def _sigmoide_oraculo(a, b, p):
    p = np.asarray(p, dtype=float)
    return 1.0 / (1.0 + np.exp(-(a + b * np.log(p / (1.0 - p)))))


def _con_recalibracion(m):
    rec = ajustar_recalibracion_logistica(m)
    assert rec.a is not None and rec.b is not None
    return rec


# ============================================================================
# ROC
# ============================================================================

@pytest.mark.parametrize("neg,pos", ETIQUETAS)
@pytest.mark.parametrize("positiva_primero", [False, True])
def test_roc_completa_es_la_de_sklearn(neg, pos, positiva_primero):
    m = _binaria(400, 3, neg, pos, positiva_primero=positiva_primero)
    puntos = curva_roc(m, max_puntos=10 ** 6)
    fpr, tpr, _ = sk.roc_curve(_y01(m), m.probabilidad_del_positivo, drop_intermediate=False)
    assert [p["fpr"] for p in puntos] == pytest.approx(list(fpr), abs=1e-12)
    assert [p["tpr"] for p in puntos] == pytest.approx(list(tpr), abs=1e-12)


def test_roc_con_empates_es_la_de_sklearn():
    m = _binaria(500, 5, decimales=1)                  # 11 valores distintos, cientos de empates
    puntos = curva_roc(m, max_puntos=10 ** 6)
    fpr, tpr, _ = sk.roc_curve(_y01(m), m.probabilidad_del_positivo, drop_intermediate=False)
    assert len(puntos) == len(fpr) <= 12
    assert [p["fpr"] for p in puntos] == pytest.approx(list(fpr), abs=1e-12)
    assert [p["tpr"] for p in puntos] == pytest.approx(list(tpr), abs=1e-12)


def test_roc_adelgazada_es_un_subconjunto_de_los_vertices_reales_con_extremos():
    m = _binaria(2000, 7)
    exacta = curva_roc(m, max_puntos=10 ** 6)
    delgada = curva_roc(m)
    assert len(exacta) > PUNTOS_DE_LA_CURVA_ROC
    assert len(delgada) == PUNTOS_DE_LA_CURVA_ROC
    assert delgada[0] == {"fpr": 0.0, "tpr": 0.0} and delgada[-1] == {"fpr": 1.0, "tpr": 1.0}
    assert all(p in exacta for p in delgada)           # vértices reales, ninguno interpolado
    assert [p["fpr"] for p in delgada] == sorted(p["fpr"] for p in delgada)
    assert [p["tpr"] for p in delgada] == sorted(p["tpr"] for p in delgada)


def test_roc_coincide_con_la_del_137_cuando_la_clase_positiva_es_evento():
    """`_curva_roc` del 137 tiene `evento` fija: si divergieran, dos ROC del mismo informe no cuadrarían."""
    for semilla, decimales in [(1, None), (2, 2), (3, 1)]:
        m = _binaria(900, semilla, "sano", "evento", decimales=decimales)
        assert curva_roc(m) == _curva_roc(m)


def test_roc_none_si_falta_una_clase():
    m = Muestra.binaria(["No"] * 5, classes=("No", "Sí"), positive_label="Sí", probabilidades=[0.1] * 5)
    assert curva_roc(m) is None and curva_pr(m) is None
    solo_pos = Muestra.binaria(["Sí"] * 5, classes=("No", "Sí"), positive_label="Sí", probabilidades=[0.1] * 5)
    assert curva_roc(solo_pos) is None
    assert curva_pr(solo_pos) is not None              # PR solo necesita positivos


@pytest.mark.parametrize("neg,pos", ETIQUETAS)
def test_bloque_roc_area_y_punto_de_operacion(neg, pos):
    m = _binaria(300, 11, neg, pos)
    c = curvas_de_la_evaluacion(m, umbral=0.4)
    roc = c["roc"]
    assert roc["disponible"] and roc["clave"] is None and roc["motivo"] is None
    assert roc["area"] == pytest.approx(sk.roc_auc_score(_y01(m), m.probabilidad_del_positivo), abs=1e-12)
    pred = [1 if p >= 0.4 else 0 for p in m.probabilidad_del_positivo]
    tn, fp, fn, tp = sk.confusion_matrix(_y01(m), pred, labels=[0, 1]).ravel()
    assert roc["punto_de_operacion"]["fpr"] == pytest.approx(fp / (fp + tn))
    assert roc["punto_de_operacion"]["tpr"] == pytest.approx(tp / (tp + fn))
    assert roc["punto_de_operacion"]["umbral"] == 0.4


# ============================================================================
# PR
# ============================================================================

@pytest.mark.parametrize("neg,pos", ETIQUETAS)
@pytest.mark.parametrize("decimales", [None, 1])
def test_pr_completa_es_la_de_sklearn_sin_el_ancla(neg, pos, decimales):
    m = _binaria(400, 13, neg, pos, decimales=decimales)
    puntos = curva_pr(m, max_puntos=10 ** 6)
    precision, recall, umbrales = sk.precision_recall_curve(_y01(m), m.probabilidad_del_positivo)
    # sklearn va de menor a mayor umbral y termina en el ancla (recall 0, precisión 1): se invierte y se quita.
    esperado_p = list(precision[:-1])[::-1]
    esperado_r = list(recall[:-1])[::-1]
    assert len(puntos) == len(umbrales)
    assert [p["precision"] for p in puntos] == pytest.approx(esperado_p, abs=1e-12)
    assert [p["recall"] for p in puntos] == pytest.approx(esperado_r, abs=1e-12)


def test_pr_adelgazada_subconjunto_de_vertices_reales_y_su_tope():
    m = _binaria(3000, 17)
    exacta = curva_pr(m, max_puntos=10 ** 6)
    delgada = curva_pr(m)
    assert len(exacta) > PUNTOS_DE_LA_CURVA_PR and len(delgada) == PUNTOS_DE_LA_CURVA_PR
    assert delgada[0] == exacta[0] and delgada[-1] == exacta[-1]
    assert all(p in exacta for p in delgada)
    assert [p["recall"] for p in delgada] == sorted(p["recall"] for p in delgada)


def test_bloque_pr_area_es_average_precision_prevalencia_y_operacion():
    m = _binaria(350, 19, "sano", "ENFERMO")
    c = curvas_de_la_evaluacion(m, umbral=0.5)
    pr = c["pr"]
    assert pr["area_metodo"] == "average_precision"
    assert pr["area"] == pytest.approx(sk.average_precision_score(_y01(m), m.probabilidad_del_positivo), abs=1e-12)
    assert pr["prevalencia"] == pytest.approx(sum(_y01(m)) / 350)
    pred = [1 if p >= 0.5 else 0 for p in m.probabilidad_del_positivo]
    assert pr["punto_de_operacion"]["precision"] == pytest.approx(sk.precision_score(_y01(m), pred))
    assert pr["punto_de_operacion"]["recall"] == pytest.approx(sk.recall_score(_y01(m), pred))


def test_precision_sin_positivos_predichos_es_none_no_cero():
    """Un umbral que no deja a nadie como positivo: la precisión no existe, y la exhaustividad es 0."""
    m = _binaria(200, 23)
    c = curvas_de_la_evaluacion(m, umbral=0.999999)
    op = c["pr"]["punto_de_operacion"]
    assert op["precision"] is None
    assert op["recall"] == 0.0
    assert c["roc"]["punto_de_operacion"]["fpr"] == 0.0


# ============================================================================
# Fiabilidad
# ============================================================================

@pytest.mark.parametrize("neg,pos", ETIQUETAS)
def test_fiabilidad_binaria_es_la_de_sklearn(neg, pos):
    m = _binaria(2000, 29, neg, pos)
    c = curvas_de_la_evaluacion(m, minimo_de_filas=1)
    f = c["fiabilidad"]
    assert f["disponible"] and f["tipo"] == "clase_positiva"
    prob_true, prob_pred = calibration_curve(_y01(m), m.probabilidad_del_positivo, n_bins=10, strategy="uniform")
    llenos = [b for b in f["curva"]["bins"] if b["n"] > 0]
    assert [b["tasa_observada"] for b in llenos] == pytest.approx(list(prob_true), abs=1e-12)
    assert [b["probabilidad_media"] for b in llenos] == pytest.approx(list(prob_pred), abs=1e-12)
    cuentas = np.histogram(m.probabilidad_del_positivo, bins=np.linspace(0, 1, 11))[0]
    assert [b["n"] for b in f["curva"]["bins"]] == list(cuentas)
    ece = sum(b["n"] / 2000 * abs(b["tasa_observada"] - b["probabilidad_media"]) for b in llenos)
    assert f["curva"]["ece"] == pytest.approx(ece, abs=1e-12)


def test_tramos_de_fiabilidad_con_menos_de_30_filas_van_sin_cifra_y_con_clave():
    """29 filas: sin cifra. 30: con cifra. El límite es EXACTO, `n` viaja siempre."""
    p = [0.05] * 29 + [0.15] * 30 + [0.95] * 100 + [0.55] * 1
    y = ["Sí" if i % 2 else "No" for i in range(len(p))]
    m = Muestra.binaria(y, classes=("No", "Sí"), positive_label="Sí", probabilidades=p)
    bins = curvas_de_la_evaluacion(m)["fiabilidad"]["curva"]["bins"]
    assert [b["n"] for b in bins] == [29, 30, 0, 0, 0, 1, 0, 0, 0, 100]
    assert bins[0]["clave"] == "tramo_con_pocas_filas"
    assert bins[0]["tasa_observada"] is None and bins[0]["probabilidad_media"] is None
    assert bins[0]["eventos"] is None
    assert bins[0]["motivo"] == motivo("tramo_con_pocas_filas", valor=29, opciones=MINIMO_FILAS_POR_SEGMENTO)
    assert bins[1]["clave"] is None and bins[1]["motivo"] is None
    assert bins[1]["probabilidad_media"] == pytest.approx(0.15)
    assert bins[1]["tasa_observada"] is not None and bins[1]["eventos"] is not None
    assert bins[2]["clave"] == "tramo_vacio" and bins[2]["tasa_observada"] is None
    assert bins[5]["clave"] == "tramo_con_pocas_filas" and bins[5]["n"] == 1
    assert bins[9]["clave"] is None and bins[9]["n"] == 100


def test_ninguna_cifra_de_un_tramo_pequeno_llega_ni_con_valor_cero():
    m = _binaria(120, 31)
    for b in curvas_de_la_evaluacion(m)["fiabilidad"]["curva"]["bins"]:
        if b["n"] < MINIMO_FILAS_POR_SEGMENTO:
            assert b["tasa_observada"] is None and b["probabilidad_media"] is None and b["eventos"] is None
            assert b["clave"] in ("tramo_vacio", "tramo_con_pocas_filas") and b["motivo"]
        else:
            assert b["tasa_observada"] is not None and b["clave"] is None


def test_ece_no_se_da_con_una_prueba_por_debajo_del_minimo_y_si_con_30():
    chica = curvas_de_la_evaluacion(_binaria(29, 37))["fiabilidad"]["curva"]
    assert chica["ece"] is None and chica["ece_clave"] == "ece_con_pocas_filas"
    assert chica["ece_motivo"] == motivo("ece_con_pocas_filas", valor=29, opciones=30)
    justa = curvas_de_la_evaluacion(_binaria(30, 37))["fiabilidad"]["curva"]
    assert justa["ece"] is not None and justa["ece_clave"] is None and justa["ece_motivo"] is None


def test_frecuencia_igual_viaja_declarada_y_cambia_los_bins():
    m = _binaria(1000, 41)
    c = curvas_de_la_evaluacion(m, metodo_de_bins="frecuencia_igual", n_bins=5)["fiabilidad"]["curva"]
    assert c["metodo_de_bins"] == "frecuencia_igual" and c["n_bins"] == 5 and len(c["bins"]) == 5
    assert sum(b["n"] for b in c["bins"]) == 1000


# ---- recalibrada = la escala de la decisión; cruda = «antes» ------------------------------------------

def test_con_recalibracion_la_fiabilidad_es_la_recalibrada_y_la_cruda_es_el_antes():
    m = _binaria(3000, 43, senal=0.6)                  # sobreconfianza fabricada: la recalibración mueve
    rec = _con_recalibracion(m)
    c = curvas_de_la_evaluacion(m, recalibracion=rec, minimo_de_filas=1)
    assert c["escala"] == "probabilidad_recalibrada"
    assert c["recalibracion"]["aplicada"] is True and c["recalibracion"]["metodo"] == rec.metodo
    f = c["fiabilidad"]
    nueva = _sigmoide_oraculo(rec.a, rec.b, m.probabilidad_del_positivo)
    pt, pp = calibration_curve(_y01(m), nueva, n_bins=10, strategy="uniform")
    llenos = [b for b in f["curva"]["bins"] if b["n"] > 0]
    assert [b["tasa_observada"] for b in llenos] == pytest.approx(list(pt), abs=1e-9)
    assert [b["probabilidad_media"] for b in llenos] == pytest.approx(list(pp), abs=1e-9)
    # el «antes» es la probabilidad CRUDA, otra curva
    assert f["antes"]["disponible"] and f["antes"]["escala"] == "probabilidad_del_modelo"
    pt0, pp0 = calibration_curve(_y01(m), m.probabilidad_del_positivo, n_bins=10, strategy="uniform")
    antes = [b for b in f["antes"]["curva"]["bins"] if b["n"] > 0]
    assert [b["tasa_observada"] for b in antes] == pytest.approx(list(pt0), abs=1e-12)
    assert [b["probabilidad_media"] for b in antes] == pytest.approx(list(pp0), abs=1e-12)
    assert [b["probabilidad_media"] for b in antes] != pytest.approx([b["probabilidad_media"] for b in llenos])


def test_con_recalibracion_roc_pr_y_matriz_viven_en_la_escala_recalibrada():
    m = _binaria(1500, 47, senal=0.6)
    rec = _con_recalibracion(m)
    nueva = _sigmoide_oraculo(rec.a, rec.b, m.probabilidad_del_positivo)
    c = curvas_de_la_evaluacion(m, recalibracion=rec, umbral=0.3)
    assert c["roc"]["area"] == pytest.approx(sk.roc_auc_score(_y01(m), nueva), abs=1e-9)
    assert c["pr"]["area"] == pytest.approx(sk.average_precision_score(_y01(m), nueva), abs=1e-9)
    pred = (nueva >= 0.3).astype(int)
    tn, fp, fn, tp = sk.confusion_matrix(_y01(m), pred, labels=[0, 1]).ravel()
    assert c["matriz"]["casillas"]["tp"] == tp and c["matriz"]["casillas"]["fp"] == fp
    assert c["matriz"]["casillas"]["fn"] == fn and c["matriz"]["casillas"]["tn"] == tn
    # y sobre la CRUDA daría otra matriz: el umbral 0,3 no es de esa escala
    pred_cruda = (np.asarray(m.probabilidad_del_positivo) >= 0.3).astype(int)
    assert (pred != pred_cruda).any()


def test_calibrador_nulo_se_dice_y_no_hay_antes():
    m = _binaria(400, 53)
    c = curvas_de_la_evaluacion(m, recalibracion=None)
    assert c["escala"] == "probabilidad_del_modelo"
    assert c["recalibracion"]["aplicada"] is False and c["recalibracion"]["clave"] == "sin_recalibracion"
    assert c["recalibracion"]["motivo"] == motivo("sin_recalibracion")
    f = c["fiabilidad"]
    assert f["antes"]["disponible"] is False and f["antes"]["clave"] == "sin_recalibracion"
    # la curva «después» es entonces la cruda, tal cual
    pt, pp = calibration_curve(_y01(m), m.probabilidad_del_positivo, n_bins=10, strategy="uniform")
    llenos = [b for b in curvas_de_la_evaluacion(m, minimo_de_filas=1)["fiabilidad"]["curva"]["bins"] if b["n"]]
    assert [b["tasa_observada"] for b in llenos] == pytest.approx(list(pt), abs=1e-12)


def test_recalibracion_que_no_se_pudo_ajustar_dice_su_motivo_y_no_se_aplica():
    m = _binaria(300, 59)
    una_clase = Muestra.binaria(["No"] * 10, classes=("No", "Sí"), positive_label="Sí", probabilidades=[0.3] * 10)
    fallida = ajustar_recalibracion_logistica(una_clase)
    assert fallida.a is None and fallida.undefined_reason
    c = curvas_de_la_evaluacion(m, recalibracion=fallida)
    r = c["recalibracion"]
    assert r["aplicada"] is False and r["clave"] == "recalibracion_no_ajustada"
    for idioma in IDIOMAS:
        assert fallida.undefined_reason[idioma] in r["motivo"][idioma]
    assert c["escala"] == "probabilidad_del_modelo"
    assert c["fiabilidad"]["antes"]["clave"] == "recalibracion_no_ajustada"


def test_calibrador_con_solo_puntuacion_no_se_aplica():
    p = [i / 100 for i in range(1, 100)]
    y = ["Sí" if i % 3 == 0 else "No" for i in range(99)]
    m = Muestra.binaria(y, classes=("No", "Sí"), positive_label="Sí", puntuaciones=p)
    rec = RecalibracionLogistica(metodo="logistico_completo", n_observaciones=99, a=0.1, b=1.2, convergio=True)
    c = curvas_de_la_evaluacion(m, recalibracion=rec, umbral=2 * 0.25)
    assert c["recalibracion"]["clave"] == "recalibracion_sin_probabilidades"
    assert c["escala"] == "puntuacion_cruda"
    assert c["fiabilidad"]["disponible"] is False and c["fiabilidad"]["clave"] == "sin_probabilidades"
    assert c["roc"]["disponible"]                      # la ROC solo pide ordenar


# ============================================================================
# Matriz de confusión
# ============================================================================

@pytest.mark.parametrize("neg,pos", ETIQUETAS)
@pytest.mark.parametrize("positiva_primero", [False, True])
def test_matriz_binaria_al_umbral_es_la_de_sklearn(neg, pos, positiva_primero):
    m = _binaria(500, 61, neg, pos, positiva_primero=positiva_primero)
    c = curvas_de_la_evaluacion(m, umbral=0.35)
    mat = c["matriz"]
    pred = [pos if p >= 0.35 else neg for p in m.probabilidad_del_positivo]
    esperado = sk.confusion_matrix(list(m.y_true), pred, labels=list(m.classes))
    for i, real in enumerate(m.classes):
        assert [mat["counts"][real][pr] for pr in m.classes] == list(esperado[i])
    assert mat["decision_rule"] == "threshold" and mat["threshold"] == 0.35
    assert mat["umbral_declarado"] is True and mat["aviso"] is None
    assert mat["casillas"]["clase_positiva"] == pos and mat["casillas"]["clase_negativa"] == neg
    i_pos, i_neg = m.classes.index(pos), m.classes.index(neg)
    assert mat["casillas"]["tp"] == esperado[i_pos][i_pos] and mat["casillas"]["fn"] == esperado[i_pos][i_neg]
    assert mat["casillas"]["fp"] == esperado[i_neg][i_pos] and mat["casillas"]["tn"] == esperado[i_neg][i_neg]
    assert mat["filas"] == 500 and sum(sum(f.values()) for f in mat["counts"].values()) == 500


def test_umbral_justo_en_el_valor_alcanza_se_cuenta_como_positivo():
    """`>=`, como el evaluador histórico: la fila que cae justo en el umbral es positiva."""
    m = Muestra.binaria(["Sí", "No", "Sí", "No"], classes=("No", "Sí"), positive_label="Sí",
                        probabilidades=[0.5, 0.5, 0.9, 0.1])
    mat = curvas_de_la_evaluacion(m, umbral=0.5)["matriz"]["casillas"]
    assert (mat["tp"], mat["fp"], mat["fn"], mat["tn"]) == (2, 1, 0, 1)


def test_sin_umbral_la_matriz_usa_05_y_lo_dice():
    m = _binaria(300, 67)
    mat = curvas_de_la_evaluacion(m)["matriz"]
    assert mat["threshold"] == 0.5 and mat["umbral_declarado"] is False
    assert mat["aviso"]["clave"] == "umbral_por_omision"
    assert mat["aviso"]["motivo"]["es"] == "no se indicó el umbral elegido: la matriz usa 0,5"
    assert mat["aviso"]["motivo"]["en"] == "no chosen threshold was given: the matrix uses 0.5"


def test_matriz_multiclase_al_argmax_es_la_de_sklearn_con_una_clase_ausente():
    for clases in CLASES_MULTI:
        ausente = clases[1]
        m = _multi(400, 71, clases, ausente=ausente)
        c = curvas_de_la_evaluacion(m)
        mat = c["matriz"]
        pred = [clases[max(range(len(clases)), key=lambda j: fila[j])] for fila in m.probabilities]
        esperado = sk.confusion_matrix(list(m.y_true), pred, labels=list(clases))
        for i, real in enumerate(clases):
            assert [mat["counts"][real][p] for p in clases] == list(esperado[i])
        assert mat["decision_rule"] == "argmax" and mat["threshold"] is None
        assert mat["casillas"] is None and mat["aviso"] is None
        assert sum(mat["counts"][ausente].values()) == 0       # la clase ausente sigue en la matriz, en cero
        assert c["prueba"]["por_clase"][ausente] == 0


def test_empate_exacto_en_multiclase_gana_la_primera_clase():
    m = Muestra.multiclase(["a", "b", "c"], classes=("a", "b", "c"),
                           probabilidades=[(0.5, 0.5, 0.0), (0.25, 0.5, 0.25), (0.0, 0.5, 0.5)])
    mat = curvas_de_la_evaluacion(m)["matriz"]["counts"]
    assert mat["a"]["a"] == 1 and mat["b"]["b"] == 1 and mat["c"]["b"] == 1


# ============================================================================
# Multiclase: fiabilidad de la clase predicha, sin recalibrar
# ============================================================================

@pytest.mark.parametrize("clases", CLASES_MULTI)
def test_fiabilidad_multiclase_es_confianza_frente_a_acierto_de_sklearn(clases):
    m = _multi(1500, 73, clases, ausente=clases[-1])
    c = curvas_de_la_evaluacion(m, minimo_de_filas=1)
    f = c["fiabilidad"]
    assert f["disponible"] and f["tipo"] == "clase_predicha"
    assert c["escala"] == f["escala"] == "confianza_de_la_clase_predicha"
    conf = np.array([max(fila) for fila in m.probabilities])
    pred = [clases[int(np.argmax(fila))] for fila in m.probabilities]
    acierto = np.array([1 if p == y else 0 for p, y in zip(pred, m.y_true)])
    pt, pp = calibration_curve(acierto, conf, n_bins=10, strategy="uniform")
    llenos = [b for b in f["curva"]["bins"] if b["n"] > 0]
    assert [b["tasa_observada"] for b in llenos] == pytest.approx(list(pt), abs=1e-12)
    assert [b["probabilidad_media"] for b in llenos] == pytest.approx(list(pp), abs=1e-12)
    ece = sum(b["n"] / 1500 * abs(b["tasa_observada"] - b["probabilidad_media"]) for b in llenos)
    assert f["curva"]["ece"] == pytest.approx(ece, abs=1e-12)
    assert f["resumen"]["tasa_de_acierto"] == pytest.approx(acierto.mean())
    assert f["resumen"]["confianza_media"] == pytest.approx(conf.mean())
    assert f["resumen"]["aciertos"] == int(acierto.sum())


def test_multiclase_no_recalibra_y_lo_dice_sin_antes():
    m = _multi(300, 79, CLASES_MULTI[0])
    c = curvas_de_la_evaluacion(m)
    assert c["recalibracion"]["aplicada"] is False and c["recalibracion"]["clave"] == "multiclase_sin_recalibrar"
    assert "23-09" in c["recalibracion"]["motivo"]["es"] and "23-09" in c["recalibracion"]["motivo"]["en"]
    assert c["fiabilidad"]["antes"]["disponible"] is False
    assert c["fiabilidad"]["antes"]["clave"] == "multiclase_sin_recalibrar"
    assert c["roc"]["clave"] == c["pr"]["clave"] == "no_aplica_a_la_tarea"
    assert c["tramos"]["clave"] == "no_aplica_a_la_tarea"


def test_multiclase_con_clase_declarada_mide_la_confianza_de_ESA_clase():
    clases = ("a", "b", "c")
    m = Muestra(task="multiclass_classification", y_true=("a", "b", "c", "a"), classes=clases,
                probabilities=((0.7, 0.2, 0.1), (0.7, 0.2, 0.1), (0.1, 0.2, 0.7), (0.2, 0.5, 0.3)),
                predictions=("a", "a", "c", "c"))
    f = curvas_de_la_evaluacion(m, minimo_de_filas=1)["fiabilidad"]
    # confianzas en la clase declarada: .7 (acierta), .7 (falla), .7 (acierta), .3 (falla)
    assert f["resumen"]["confianza_media"] == pytest.approx((0.7 + 0.7 + 0.7 + 0.3) / 4)
    assert f["resumen"]["aciertos"] == 2


def test_multiclase_solo_con_etiquetas_tiene_matriz_pero_no_fiabilidad():
    m = Muestra(task="multiclass_classification", y_true=("a", "b", "c", "a"), classes=("a", "b", "c"),
                predictions=("a", "b", "b", "a"))
    c = curvas_de_la_evaluacion(m)
    assert c["matriz"]["decision_rule"] == "declared_label" and c["matriz"]["counts"]["c"]["b"] == 1
    assert c["fiabilidad"]["clave"] == "sin_probabilidades" and c["escala"] is None


# ============================================================================
# Regresión: deciles del predicho
# ============================================================================

def test_tramos_son_los_deciles_del_predicho_de_numpy_con_n_divisible():
    m = _regresion(1000, 83, escala=1e6, desplazamiento=-2.5e6)     # valores grandes y negativos
    c = curvas_de_la_evaluacion(m)
    t = c["tramos"]
    assert t["disponible"] and t["n_tramos"] == t["n_tramos_pedidos"] == 10
    pred = np.array(m.predictions)
    real = np.array(m.y_true)
    orden = np.argsort(pred, kind="stable")
    for j, trozo in enumerate(np.array_split(orden, 10)):
        tr = t["tramos"][j]
        assert tr["n"] == 100 and tr["clave"] is None and tr["motivo"] is None
        assert tr["predicho_medio"] == pytest.approx(pred[trozo].mean(), rel=1e-12)
        assert tr["real_medio"] == pytest.approx(real[trozo].mean(), rel=1e-12)
        assert tr["real_desviacion"] == pytest.approx(real[trozo].std(ddof=1), rel=1e-9)
        assert tr["desde"] == pred[trozo].min()
        assert tr["hasta"] == (pred[trozo].max() if j == 9 else pred[orden[(j + 1) * 100]])


@pytest.mark.parametrize("n", [333, 457, 1009])
def test_tramos_con_n_no_divisible_cubren_todas_las_filas_por_su_corte(n):
    m = _regresion(n, 89, escala=3.0)
    t = curvas_de_la_evaluacion(m, minimo_de_filas=1)["tramos"]["tramos"]
    pred = np.array(m.predictions)
    real = np.array(m.y_true)
    assert sum(tr["n"] for tr in t) == n
    cortes = [tr["desde"] for tr in t[1:]]
    indice = np.digitize(pred, cortes)                  # [corte_j-1, corte_j) → tramo j
    for j, tr in enumerate(t):
        sel = indice == j
        assert tr["n"] == sel.sum()
        assert tr["predicho_medio"] == pytest.approx(pred[sel].mean(), rel=1e-12)
        assert tr["real_medio"] == pytest.approx(real[sel].mean(), rel=1e-12)
    # tamaños casi iguales: ningún tramo se aparta de n/10 más de 1 fila
    assert max(abs(tr["n"] - n / 10) for tr in t) <= 1.0


def test_tamanos_de_los_tramos_con_n_457_son_los_de_la_regla_round_i_n_k():
    """La regla del corte, fijada con números: la posición del corte i es `round(i·n/k)` (n=457, k=10 →
    46, 91, 137, 183, 228, 274, 320, 366, 411; el 228,5 redondea al par). Con `int()` saldrían otros tamaños."""
    m = _regresion(457, 89, escala=3.0)
    t = curvas_de_la_evaluacion(m, minimo_de_filas=1)["tramos"]["tramos"]
    assert [tr["n"] for tr in t] == [46, 45, 46, 46, 45, 46, 46, 46, 45, 46]
    orden = sorted(m.predictions)
    assert [tr["desde"] for tr in t[1:]] == [orden[i] for i in (46, 91, 137, 183, 228, 274, 320, 366, 411)]


def test_tramos_con_predicciones_repetidas_no_parten_un_valor_en_dos():
    rng = random.Random(97)
    pred = [float(rng.choice([1, 1, 1, 2, 2, 3, 5, 5, 8, 13])) for _ in range(600)]
    real = [v + rng.gauss(0, 1) for v in pred]
    m = Muestra(task="regression", y_true=tuple(real), predictions=tuple(pred))
    t = curvas_de_la_evaluacion(m, minimo_de_filas=1)["tramos"]
    assert t["n_tramos"] < t["n_tramos_pedidos"]
    assert sum(tr["n"] for tr in t["tramos"]) == 600
    for tr in t["tramos"]:                              # un valor repetido vive en UN solo tramo
        dentro = [p for p in pred if tr["desde"] <= p < tr["hasta"] or (tr is t["tramos"][-1] and p == tr["hasta"])]
        assert len(dentro) == tr["n"]
    assert [tr["desde"] for tr in t["tramos"]] == sorted(set(tr["desde"] for tr in t["tramos"]))


def test_regresion_con_un_solo_valor_predicho_es_un_tramo():
    m = Muestra(task="regression", y_true=tuple(float(i) for i in range(50)), predictions=(7.0,) * 50)
    t = curvas_de_la_evaluacion(m)["tramos"]
    assert t["n_tramos"] == 1 and t["tramos"][0]["n"] == 50
    assert t["tramos"][0]["real_medio"] == pytest.approx(24.5)
    assert t["tramos"][0]["real_desviacion"] == pytest.approx(np.std(np.arange(50), ddof=1))


def test_tramo_de_regresion_con_menos_de_30_filas_va_sin_cifra_y_con_su_n():
    # 29 + 30 + 41 filas en tres valores predichos; n_tramos=3 corta justo entre ellos
    pred = [1.0] * 29 + [2.0] * 30 + [3.0] * 41
    real = [p + 0.5 * ((i % 3) - 1) for i, p in enumerate(pred)]
    m = Muestra(task="regression", y_true=tuple(real), predictions=tuple(pred))
    t = curvas_de_la_evaluacion(m, n_tramos=3)["tramos"]["tramos"]
    assert [tr["n"] for tr in t] == [29, 30, 41]
    assert t[0]["clave"] == "tramo_con_pocas_filas"
    assert t[0]["predicho_medio"] is None and t[0]["real_medio"] is None and t[0]["real_desviacion"] is None
    assert t[0]["motivo"] == motivo("tramo_con_pocas_filas", valor=29, opciones=30)
    assert t[1]["clave"] is None and t[1]["predicho_medio"] == 2.0 and t[1]["real_medio"] is not None
    assert t[2]["clave"] is None


def test_regresion_de_200_filas_en_deciles_son_todos_de_20_y_van_sin_cifra():
    t = curvas_de_la_evaluacion(_regresion(200, 101))["tramos"]["tramos"]
    assert [tr["n"] for tr in t] == [20] * 10
    assert all(tr["real_medio"] is None and tr["clave"] == "tramo_con_pocas_filas" for tr in t)
    # y con menos tramos, los de 40 filas sí tienen cifra
    t5 = curvas_de_la_evaluacion(_regresion(200, 101), n_tramos=5)["tramos"]["tramos"]
    assert [tr["n"] for tr in t5] == [40] * 5 and all(tr["real_medio"] is not None for tr in t5)


def test_regresion_no_tiene_roc_pr_fiabilidad_ni_matriz_y_lo_dice():
    c = curvas_de_la_evaluacion(_regresion(100, 103))
    for bloque in ("roc", "pr", "fiabilidad", "matriz"):
        assert c[bloque]["disponible"] is False and c[bloque]["clave"] == "no_aplica_a_la_tarea"
        assert "regresión" in c[bloque]["motivo"]["es"] and "regression" in c[bloque]["motivo"]["en"]
    assert c["escala"] == "valor_predicho" and c["recalibracion"]["aplicada"] is False
    assert c["prueba"]["filas"] == 100 and c["prueba"]["positivos"] is None


# ============================================================================
# La prueba al lado, lo ausente y las entradas
# ============================================================================

def test_la_prueba_trae_filas_positivos_y_negativos():
    m = _binaria(250, 107, "sano", "ENFERMO")
    p = curvas_de_la_evaluacion(m)["prueba"]
    pos = sum(_y01(m))
    assert (p["filas"], p["positivos"], p["negativos"]) == (250, pos, 250 - pos)
    assert p["prevalencia"] == pytest.approx(pos / 250) and p["clase_positiva"] == "ENFERMO"
    assert p["por_clase"] == {"sano": 250 - pos, "ENFERMO": pos} and p["minimo_de_filas"] == 30
    assert p["cautela"] is None


def test_cautela_por_prueba_pequena_y_por_pocos_eventos():
    chica = curvas_de_la_evaluacion(_binaria(29, 109))["prueba"]["cautela"]
    assert chica["clave"] == "prueba_con_pocas_filas" and chica["motivo"] == motivo(
        "prueba_con_pocas_filas", valor=29, opciones=30)
    y = ["Sí"] * 5 + ["No"] * 95
    m = Muestra.binaria(y, classes=("No", "Sí"), positive_label="Sí", probabilidades=[i / 100 for i in range(100)])
    c = curvas_de_la_evaluacion(m)["prueba"]["cautela"]
    assert c["clave"] == "prueba_con_pocos_eventos" and c["motivo"] == motivo(
        "prueba_con_pocos_eventos", valor=5, opciones=10)


def test_una_sola_clase_en_la_prueba_dice_por_que_no_hay_roc():
    m = Muestra.binaria(["No"] * 40, classes=("No", "Sí"), positive_label="Sí",
                        probabilidades=[i / 40 for i in range(40)])
    c = curvas_de_la_evaluacion(m)
    assert c["roc"]["disponible"] is False and c["roc"]["clave"] == "roc_sin_las_dos_clases"
    assert "0 positivos" in c["roc"]["motivo"]["es"] and "40 negatives" in c["roc"]["motivo"]["en"]
    assert c["pr"]["clave"] == "pr_sin_positivos"
    assert c["matriz"]["disponible"] and c["fiabilidad"]["disponible"]


def test_binaria_con_solo_etiquetas_no_inventa_curvas():
    m = Muestra.binaria(["No", "Sí", "Sí", "No"], classes=("No", "Sí"), positive_label="Sí",
                        predicciones=["No", "Sí", "No", "No"])
    c = curvas_de_la_evaluacion(m)
    assert c["roc"]["clave"] == c["pr"]["clave"] == "sin_puntuacion"
    assert c["fiabilidad"]["clave"] == "sin_probabilidades"
    assert c["matriz"]["decision_rule"] == "declared_label"
    assert c["matriz"]["casillas"]["tp"] == 1 and c["matriz"]["casillas"]["fn"] == 1


def test_puntuacion_sin_escala_de_probabilidad_pide_umbral():
    m = Muestra.binaria(["No", "Sí", "Sí", "No"], classes=("No", "Sí"), positive_label="Sí",
                        puntuaciones=[-2.0, 3.0, 0.5, -0.1])
    sin = curvas_de_la_evaluacion(m)
    assert sin["matriz"]["clave"] == "matriz_sin_umbral_en_puntuacion" and sin["roc"]["disponible"]
    con = curvas_de_la_evaluacion(m, umbral=0.0)
    assert con["matriz"]["casillas"]["tp"] == 2 and con["matriz"]["casillas"]["fp"] == 0
    assert con["roc"]["punto_de_operacion"]["umbral"] == 0.0


def test_prueba_vacia_dice_prueba_vacia_en_todos_los_bloques():
    for m in (Muestra.binaria([], classes=("No", "Sí"), positive_label="Sí", probabilidades=[]),
              Muestra(task="regression", y_true=())):
        c = curvas_de_la_evaluacion(m)
        for bloque in ("roc", "pr", "fiabilidad", "matriz", "tramos", "recalibracion"):
            assert c[bloque]["disponible"] is False and c[bloque]["clave"] == "prueba_vacia"
        assert c["prueba"]["filas"] == 0 and c["prueba"]["cautela"]["clave"] == "prueba_con_pocas_filas"


def test_pesos_se_rechazan_en_vez_de_contar_filas_a_medias():
    m = Muestra.binaria(["No", "Sí"], classes=("No", "Sí"), positive_label="Sí", probabilidades=[0.2, 0.8],
                        pesos=[1.0, 2.0])
    with pytest.raises(EntradaNoMedible):
        curvas_de_la_evaluacion(m)


def test_parametros_invalidos_se_rechazan():
    m = _binaria(50, 113)
    for malo in ({"n_bins": 0}, {"n_tramos": 0}, {"minimo_de_filas": 0}, {"metodo_de_bins": "otro"},
                 {"umbral": float("nan")}, {"umbral": True}, {"recalibracion": {"a": 1}}):
        with pytest.raises(EsquemaInvalido):
            curvas_de_la_evaluacion(m, **malo)


# ============================================================================
# El documento: JSON, determinismo, solo agregados
# ============================================================================

def _todas():
    rec = None
    out = []
    m = _binaria(500, 127, "sano", "ENFERMO", senal=0.7)
    rec = _con_recalibracion(m)
    out.append(curvas_de_la_evaluacion(m, recalibracion=rec, umbral=0.45))
    out.append(curvas_de_la_evaluacion(_binaria(80, 131, decimales=1), umbral=0.5))
    out.append(curvas_de_la_evaluacion(_multi(400, 137, CLASES_MULTI[2], ausente="verde")))
    out.append(curvas_de_la_evaluacion(_regresion(400, 139, escala=1e9, desplazamiento=-3e9)))
    return out


def test_el_documento_es_json_estricto_y_viaja_idem():
    for c in _todas():
        texto = json.dumps(c, ensure_ascii=False, allow_nan=False)    # ni NaN ni Infinity
        assert json.loads(texto) == c
        assert c["version"] == VERSION_DE_LAS_CURVAS


def test_determinista_y_no_depende_del_orden_de_las_filas():
    m = _binaria(700, 149, "sano", "ENFERMO")
    a = curvas_de_la_evaluacion(m, umbral=0.4)
    assert a == curvas_de_la_evaluacion(m, umbral=0.4)
    idx = list(range(700))
    random.Random(1).shuffle(idx)
    barajada = Muestra.binaria([m.y_true[i] for i in idx], classes=m.classes, positive_label=m.positive_label,
                               probabilidades=[m.probabilidad_del_positivo[i] for i in idx])
    assert curvas_de_la_evaluacion(barajada, umbral=0.4) == a
    r = _regresion(500, 151)
    ordenada = curvas_de_la_evaluacion(r)
    idx = list(range(500))
    random.Random(2).shuffle(idx)
    rb = Muestra(task="regression", y_true=tuple(r.y_true[i] for i in idx),
                 predictions=tuple(r.predictions[i] for i in idx))
    assert curvas_de_la_evaluacion(rb) == ordenada


def test_solo_agregados_ninguna_lista_crece_con_las_filas():
    """Con 20.000 filas y puntuaciones todas distintas, ninguna lista del documento pasa de 40 elementos
    y el documento pesa lo mismo que con 2.000: no viaja ninguna fila."""
    tamanos = []
    for n in (2000, 20000):
        c = curvas_de_la_evaluacion(_binaria(n, 157), umbral=0.5)

        def recorrer(x):
            if isinstance(x, list):
                assert len(x) <= max(PUNTOS_DE_LA_CURVA_ROC, PUNTOS_DE_LA_CURVA_PR)
                for v in x:
                    recorrer(v)
            elif isinstance(x, dict):
                for v in x.values():
                    recorrer(v)
        recorrer(c)
        tamanos.append(len(json.dumps(c)))
    assert abs(tamanos[1] - tamanos[0]) < 0.2 * tamanos[0]
    r = curvas_de_la_evaluacion(_regresion(20000, 163))
    assert len(json.dumps(r)) < 6000


def test_todo_bloque_o_trae_datos_o_trae_clave_y_motivo_en_los_dos_idiomas():
    for c in _todas():
        for nombre in ("roc", "pr", "fiabilidad", "matriz", "tramos", "recalibracion"):
            b = c[nombre]
            if b["disponible"]:
                assert b["clave"] is None and b["motivo"] is None
            else:
                assert b["clave"] in MOTIVOS
                assert set(b["motivo"]) == set(IDIOMAS) and all(b["motivo"][i] for i in IDIOMAS)


# ============================================================================
# Los textos, en los dos idiomas
# ============================================================================

def test_cada_motivo_tiene_las_dos_redacciones_y_los_mismos_huecos():
    assert MOTIVOS
    for clave, redacciones in MOTIVOS.items():
        assert set(redacciones) == set(IDIOMAS), clave
        assert huecos_de(redacciones["es"]) == huecos_de(redacciones["en"]), clave
        assert redacciones["es"].strip() and redacciones["en"].strip(), clave
        assert redacciones["es"] != redacciones["en"], clave


def test_toda_clave_que_el_codigo_usa_esta_en_el_catalogo_y_toda_del_catalogo_se_usa():
    fuente = Path(modulo.__file__).read_text(encoding="utf-8")
    usadas = set(re.findall(r'(?:motivo|_no)\(\s*"([a-z_0-9]+)"', fuente))
    usadas |= set(re.findall(r'"(ece_con_pocas_filas|tramo_vacio|tramo_con_pocas_filas)"', fuente))
    assert usadas <= set(MOTIVOS)
    assert set(MOTIVOS) <= usadas | {"no_aplica_a_la_tarea"}, set(MOTIVOS) - usadas


def test_motivo_con_clave_desconocida_revienta_en_vez_de_escribir_una_frase_suelta():
    with pytest.raises(KeyError):
        motivo("no_existe")
