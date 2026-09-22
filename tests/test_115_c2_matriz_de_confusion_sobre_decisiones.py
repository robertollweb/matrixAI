# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C2 — sensibilidad, especificidad, VPP y VPN también sobre DECISIONES ya tomadas."""
from __future__ import annotations

import pytest

from matrixai.estudio.metricas import REGISTRO, EntradaNoMedible, Muestra, calcular
from matrixai.export.attest import METRICAS

#: 4 positivos reales (3 acertados) y 6 negativos (4 acertados): TP 3, FN 1, TN 4, FP 2.
DECISIONES = Muestra(task="binary_classification",
                     y_true=("si", "si", "si", "si", "no", "no", "no", "no", "no", "no"),
                     classes=("no", "si"), positive_label="si",
                     predictions=("si", "si", "si", "no", "no", "no", "no", "no", "si", "si"))

_LAS_CUATRO = ("sensitivity", "specificity", "ppv", "npv")


@pytest.mark.parametrize("metric_id, esperado", [
    ("sensitivity", 3 / 4), ("specificity", 4 / 6), ("ppv", 3 / 5), ("npv", 4 / 5),
    ("accuracy", 7 / 10)])
def test_las_de_la_matriz_de_confusion_salen_sobre_decisiones(metric_id, esperado):
    assert calcular(metric_id, DECISIONES).value == pytest.approx(esperado)


@pytest.mark.parametrize("metric_id", ["auroc", "average_precision", "brier_score"])
def test_las_que_ordenan_o_piden_probabilidad_SIGUEN_negandose(metric_id):
    """Lo de «una etiqueta dura no ordena nada» sigue valiendo donde es verdad."""
    with pytest.raises(EntradaNoMedible):
        calcular(metric_id, DECISIONES)


@pytest.mark.parametrize("metric_id", _LAS_CUATRO)
def test_el_catalogo_dice_que_piden_la_etiqueta_como_accuracy(metric_id):
    """Auditoría del 2026-09-22 (M8): el grupo se abrió y la ficha publicada seguía
    diciendo `scores`."""
    requires = REGISTRO[metric_id].spec.requires
    assert "labels" in requires and "scores" not in requires


def test_atestiguar_NO_ofrece_lo_que_depende_de_la_clase_positiva():
    """`attest` declara positiva la última clase del vocabulario, arbitraria con las
    clases observadas: la misma atestiguación daba sensibilidad 0,75 o 0,50 según el
    orden de las filas. Abrir el registro a las etiquetas no puede abrir esto."""
    assert not [m for m in METRICAS if "positive_label" in REGISTRO[m].spec.requires]
    assert METRICAS == ("accuracy", "macro_f1", "mae", "rmse", "r2")
