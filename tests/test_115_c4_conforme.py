# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C4 — predicción conforme por partición (`matrixai.estudio.conforme`)."""
from __future__ import annotations

import random

import pytest

from matrixai.estudio.conforme import (
    GARANTIA,
    CalibracionConforme,
    ConformeNoAplicable,
    calibrar,
    conjunto_de_clases,
    intervalo_de_prediccion,
    medir_cobertura,
)
from matrixai.estudio.metricas import Muestra

CLASES = ("no", "si")


def _binaria(n, semilla):
    """Probabilidades CALIBRADAS de verdad: la etiqueta se sortea con la propia p."""
    azar = random.Random(semilla)
    ps = [azar.random() for _ in range(n)]
    y = tuple("si" if azar.random() < p else "no" for p in ps)
    return Muestra(task="binary_classification", y_true=y, classes=CLASES, positive_label="si",
                   probabilities=tuple((1 - p, p) for p in ps))


def _regresion(n, semilla):
    azar = random.Random(semilla)
    xs = [azar.uniform(0, 10) for _ in range(n)]
    return Muestra(task="regression", y_true=tuple(x + azar.gauss(0, 1) for x in xs),
                   predictions=tuple(xs))


def test_el_cuantil_es_el_k_esimo_con_el_mas_uno_de_las_muestras_finitas():
    """Nueve puntuaciones 0,1…0,9 (y = «no» con p(no) = 1 − s).
    alfa 0,1 → k = ⌈10·0,9⌉ = 9 → 0,9; alfa 0,2 → k = ⌈10·0,8⌉ = 8 → 0,8.
    En esos dos el `+ 1` no cambia nada; en alfa 0,25 SÍ: k = ⌈10·0,75⌉ = 8
    (0,8) frente a ⌈9·0,75⌉ = 7 (0,7) sin él."""
    puntuaciones = [i / 10 for i in range(1, 10)]
    muestra = Muestra(task="binary_classification", y_true=("no",) * 9, classes=CLASES,
                      positive_label="si", probabilities=tuple((1 - s, s) for s in puntuaciones))
    assert calibrar(muestra, alfa=0.1).cuantil == pytest.approx(0.9)
    assert calibrar(muestra, alfa=0.2).cuantil == pytest.approx(0.8)
    assert calibrar(muestra, alfa=0.25).cuantil == pytest.approx(0.8)


def test_con_pocas_filas_no_se_garantiza_y_se_dice_cuantas_harian_falta():
    cal = calibrar(_binaria(5, 0), alfa=0.1)
    assert cal.cuantil is None
    # ⌈(9+1)·0,9⌉ = 9 ≤ 9. La cifra EN SU FRASE: un «9» suelto lo contiene «90 %» siempre
    # (auditoría del 2026-09-22: el sabotaje «harían falta 12» salía verde).
    assert "harían falta al menos 9" in cal.motivo["es"], cal.motivo
    assert "at least 9 would be needed" in cal.motivo["en"], cal.motivo
    with pytest.raises(ConformeNoAplicable):
        conjunto_de_clases((0.5, 0.5), cal)
    assert calibrar(_binaria(9, 0), alfa=0.1).cuantil is not None


def test_la_cobertura_en_test_cumple_la_nominal_en_binaria():
    cal = calibrar(_binaria(2000, 1), alfa=0.1)
    medida = medir_cobertura(_binaria(5000, 2), cal)
    assert medida.n == 5000
    assert medida.compatible_con_la_nominal is True, medida
    # Y el tamaño se declara: en binaria, entre 1 y 2 clases de media.
    assert 1.0 <= medida.tamano_medio <= 2.0


def test_la_cobertura_en_test_cumple_la_nominal_en_regresion():
    cal = calibrar(_regresion(2000, 3), alfa=0.1)
    # Con ruido N(0, 1), el cuantil del 90 % de |ruido| es ~1,645.
    assert 1.5 < cal.cuantil < 1.8
    medida = medir_cobertura(_regresion(5000, 4), cal)
    assert medida.compatible_con_la_nominal is True, medida
    assert medida.tamano_medio == pytest.approx(2 * cal.cuantil)
    assert intervalo_de_prediccion(10.0, cal) == pytest.approx((10.0 - cal.cuantil, 10.0 + cal.cuantil))


def test_con_un_modelo_mal_calibrado_la_medicion_lo_detecta():
    """La garantía depende de que test se parezca a calibración. Si en test las
    probabilidades mienten (se invierten), la cobertura cae y se VE."""
    cal = calibrar(_binaria(2000, 5), alfa=0.1)
    test = _binaria(3000, 6)
    invertida = Muestra(task=test.task, y_true=test.y_true, classes=CLASES, positive_label="si",
                        probabilities=tuple((p, q) for q, p in test.probabilities))
    medida = medir_cobertura(invertida, cal)
    assert medida.compatible_con_la_nominal is False
    assert medida.cobertura_observada < 0.85


def test_un_conjunto_vacio_se_cuenta_no_se_rellena():
    """Tres clases a partes iguales: con un cuantil pequeño ninguna llega."""
    cal = CalibracionConforme(tarea="multiclass_classification", alfa=0.1, n_calibracion=100,
                              cuantil=0.5, clases=("a", "b", "c"))
    assert conjunto_de_clases((1 / 3, 1 / 3, 1 / 3), cal) == ()
    # El borde ENTRA: 1 − 0,5 = 0,5 ≤ 0,5. Con `<` la garantía se pierde en los empates.
    assert conjunto_de_clases((0.5, 0.5, 0.0), cal) == ("a", "b")
    test = Muestra(task="multiclass_classification", y_true=("a", "b"), classes=("a", "b", "c"),
                   probabilities=((1 / 3, 1 / 3, 1 / 3), (0.1, 0.8, 0.1)))
    medida = medir_cobertura(test, cal)
    assert (medida.vacios, medida.cubiertas) == (1, 1)
    assert medida.tamano_medio == pytest.approx(0.5)  # 0 y 1


def test_una_etiqueta_dura_no_se_calibra():
    muestra = Muestra(task="binary_classification", y_true=("si", "no"), classes=CLASES,
                      positive_label="si", predictions=("si", "no"))
    with pytest.raises(ConformeNoAplicable) as error:
        calibrar(muestra, alfa=0.1)
    assert set(error.value.motivo) == {"es", "en"}


def test_viaja_en_json_con_su_garantia_dicha():
    cal = calibrar(_binaria(500, 7), alfa=0.2)
    datos = cal.a_json()
    assert datos["garantia"] == GARANTIA
    assert CalibracionConforme.desde_json(datos) == cal
    medida = medir_cobertura(_binaria(300, 8), cal).a_json()
    assert medida["garantia"] == GARANTIA and medida["tamano_medio"] is not None
