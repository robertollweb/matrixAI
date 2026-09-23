# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C3 — bandas de decisión (`matrixai.estudio.bandas`)."""
from __future__ import annotations

import itertools
import random

import pytest

from matrixai.estudio.bandas import (
    SEMIANCHO_MAXIMO,
    BandasNoAplicables,
    PoliticaDeBandas,
    elegir_bandas,
    medir_bandas,
)
from matrixai.estudio.esquemas import Restriccion
from matrixai.estudio.metricas import Muestra
from matrixai.estudio.umbral import elegir_umbral

CLASES = ("no", "si")


def _muestra(n, semilla, decimales=None):
    azar = random.Random(semilla)
    ps = [azar.random() for _ in range(n)]
    if decimales is not None:
        ps = [round(p, decimales) for p in ps]
    y = tuple("si" if azar.random() < p else "no" for p in ps)
    return Muestra(task="binary_classification", y_true=y, classes=CLASES, positive_label="si",
                   probabilities=tuple((1 - p, p) for p in ps))


def _coste(muestra, bajo, alto, c_fp, c_fn, c_rev):
    total = 0.0
    for p, y in zip(muestra.probabilidad_del_positivo, muestra.y_true):
        if p >= alto:
            total += c_fp if y == "no" else 0.0
        elif p < bajo:
            total += c_fn if y == "si" else 0.0
        else:
            total += c_rev
    return total / muestra.n


@pytest.mark.parametrize("semilla", range(6))
@pytest.mark.parametrize("costes", [(1.0, 1.0, 0.2), (1.0, 5.0, 0.3), (3.0, 1.0, 0.1), (1.0, 1.0, 2.0)])
def test_el_minimo_es_el_de_la_fuerza_bruta(semilla, costes):
    """Todos los pares `bajo ≤ alto` de candidatos, a mano: el barrido tiene que
    dar el mismo coste mínimo. Con probabilidades redondeadas a 1 decimal, para
    que haya empates."""
    c_fp, c_fn, c_rev = costes
    muestra = _muestra(40, semilla, decimales=1)
    politica = elegir_bandas(muestra, coste_falso_positivo=c_fp, coste_falso_negativo=c_fn,
                             coste_de_revision=c_rev)
    candidatos = sorted(set(muestra.probabilidad_del_positivo) | {0.0, 1.0})
    minimo = min(_coste(muestra, b, a, c_fp, c_fn, c_rev)
                 for b, a in itertools.combinations_with_replacement(candidatos, 2))
    assert _coste(muestra, politica.umbral_bajo, politica.umbral_alto, c_fp, c_fn, c_rev) \
        == pytest.approx(minimo)


def test_si_revisar_sale_caro_no_hay_revision_y_cuesta_lo_del_umbral_de_siempre():
    muestra = _muestra(500, 1)
    politica = elegir_bandas(muestra, coste_falso_positivo=1.0, coste_falso_negativo=1.0,
                             coste_de_revision=5.0)
    assert politica.umbral_bajo == politica.umbral_alto
    de_siempre = elegir_umbral(muestra, cost_false_positive=1.0, cost_false_negative=1.0)
    t = de_siempre.threshold
    assert _coste(muestra, politica.umbral_bajo, politica.umbral_alto, 1, 1, 5) == \
        pytest.approx(_coste(muestra, t, t, 1, 1, 5))


def test_si_revisar_es_barato_se_abre_una_banda_de_revision():
    muestra = _muestra(2000, 2)
    politica = elegir_bandas(muestra, coste_falso_positivo=1.0, coste_falso_negativo=1.0,
                             coste_de_revision=0.2)
    # Con probabilidades calibradas y c_rev = 0,2, conviene revisar cuando el
    # error esperado min(p, 1 − p) pasa de 0,2: de p ≈ 0,2 a p ≈ 0,8.
    assert 0.1 < politica.umbral_bajo < 0.3 and 0.7 < politica.umbral_alto < 0.9, politica


def test_las_fronteras_son_las_del_umbral_de_siempre():
    politica = PoliticaDeBandas(umbral_bajo=0.3, umbral_alto=0.7, positive_label="si",
                                negative_label="no", coste_falso_positivo=1,
                                coste_falso_negativo=1, coste_de_revision=0.2)
    assert [politica.banda_de(p) for p in (0.7, 0.69, 0.3, 0.29)] == \
        ["positiva", "revision", "revision", "negativa"]
    sin_revision = PoliticaDeBandas(umbral_bajo=0.5, umbral_alto=0.5, positive_label="si",
                                    negative_label="no", coste_falso_positivo=1,
                                    coste_falso_negativo=1, coste_de_revision=9)
    assert [sin_revision.banda_de(p) for p in (0.5, 0.49)] == ["positiva", "negativa"]


def test_cada_banda_se_mide_en_test_con_su_intervalo():
    cal, test = _muestra(2000, 3), _muestra(4000, 4)
    politica = elegir_bandas(cal, coste_falso_positivo=1.0, coste_falso_negativo=1.0,
                             coste_de_revision=0.2)
    medidas = {m.banda: m for m in medir_bandas(test, politica)}
    assert sum(m.n for m in medidas.values()) == 4000
    assert sum(m.cobertura for m in medidas.values()) == pytest.approx(1.0)
    # Con p ≥ ~0,8 calibrada, el VPP de la positiva ronda el 90 %; y se da por medido.
    positiva = medidas["positiva"]
    assert positiva.medida is True and positiva.motivo is None
    assert 0.85 < positiva.proporcion < 0.95
    assert positiva.intervalo_95[0] < positiva.proporcion < positiva.intervalo_95[1]
    # La negativa cuenta NEGATIVOS de verdad (el VPN), no positivos.
    negativa = medidas["negativa"]
    assert 0.85 < negativa.proporcion < 0.95
    # En la de revisión no se decidió nada: se dice cuántos positivos había (~la mitad).
    assert 0.35 < medidas["revision"].proporcion < 0.65


def test_una_banda_con_pocas_filas_no_se_pinta_y_se_dice():
    politica = PoliticaDeBandas(umbral_bajo=0.3, umbral_alto=0.7, positive_label="si",
                                negative_label="no", coste_falso_positivo=1,
                                coste_falso_negativo=1, coste_de_revision=0.2)
    test = Muestra(task="binary_classification", y_true=("si",) * 5 + ("no",) * 200,
                   classes=CLASES, positive_label="si",
                   probabilities=((0.1, 0.9),) * 5 + ((0.9, 0.1),) * 200)
    medidas = {m.banda: m for m in medir_bandas(test, politica)}
    positiva = medidas["positiva"]
    assert positiva.n == 5 and positiva.proporcion == 1.0
    assert positiva.medida is False
    assert "5" in positiva.motivo["es"] and "5" in positiva.motivo["en"]
    assert medidas["negativa"].medida is True
    # Una banda vacía tampoco se pinta.
    assert medidas["revision"].n == 0 and medidas["revision"].medida is False
    assert SEMIANCHO_MAXIMO == 0.10


def test_con_restricciones_obligatorias_se_niega_y_lo_dice():
    with pytest.raises(BandasNoAplicables) as error:
        elegir_bandas(_muestra(100, 5), coste_falso_positivo=1, coste_falso_negativo=1,
                      coste_de_revision=0.2,
                      restricciones=(Restriccion(clave="sensitivity", operador="min", valor=0.9),))
    assert set(error.value.motivo) == {"es", "en"}


def test_sin_probabilidades_o_fuera_de_binaria_no_se_eligen():
    etiquetas = Muestra(task="binary_classification", y_true=("si", "no"), classes=CLASES,
                        positive_label="si", predictions=("si", "no"))
    with pytest.raises(BandasNoAplicables):
        elegir_bandas(etiquetas, coste_falso_positivo=1, coste_falso_negativo=1,
                      coste_de_revision=0.2)
    regresion = Muestra(task="regression", y_true=(1.0, 2.0), predictions=(1.0, 2.0))
    with pytest.raises(BandasNoAplicables):
        medir_bandas(regresion, PoliticaDeBandas(0.3, 0.7, "si", "no", 1, 1, 0.2))


def test_viaja_en_json():
    politica = elegir_bandas(_muestra(300, 6), coste_falso_positivo=2.0, coste_falso_negativo=1.0,
                             coste_de_revision=0.3)
    assert PoliticaDeBandas.desde_json(politica.a_json()) == politica


# -- LAS BANDAS NO CONTRADICEN AL UMBRAL DE SIEMPRE (auditoría del 2026-09-23) -------------
#
# El umbral y las bandas se elegían por separado y desempataban distinto: con los mismos
# costes salía umbral 0,737 y bandas [0,717, 0,717], y una fila con p = 0,727 decía «no» por
# el umbral y «positiva» por la banda. Con `umbral=` dado: bajo ≤ umbral ≤ alto, y sin zona de
# revisión la política ES el umbral.

def _muestra_sintetica(semilla, n=120):
    rng = random.Random(semilla)
    y, p = [], []
    for _ in range(n):
        pos = rng.random() < 0.35
        y.append("si" if pos else "no")
        p.append(round(min(1, max(0, rng.gauss(0.62 if pos else 0.4, 0.2))), 3))
    return Muestra.binaria(y_true=tuple(y), classes=("no", "si"), positive_label="si",
                           probabilidades=tuple(p)), p


@pytest.mark.parametrize("coste_de_revision", [0.1, 0.3, 0.5, 0.9, 1.5])
def test_con_el_umbral_dado_ninguna_fila_contradice_al_umbral(coste_de_revision):
    contradicciones_sin_umbral = 0
    for semilla in range(60):
        muestra, p = _muestra_sintetica(semilla)
        t = elegir_umbral(muestra, cost_false_positive=1, cost_false_negative=1).threshold
        b = elegir_bandas(muestra, coste_falso_positivo=1, coste_falso_negativo=1,
                          coste_de_revision=coste_de_revision, umbral=t)
        assert b.umbral_bajo <= t <= b.umbral_alto, (semilla, t, b)
        for pi in p:
            banda = b.banda_de(pi)
            if banda != "revision":
                assert (banda == "positiva") == (pi >= t), (semilla, pi, t, b)
        if b.umbral_bajo == b.umbral_alto:
            assert b.umbral_bajo == t, "sin zona de revisión, la política tiene que SER el umbral"
        viejo = elegir_bandas(muestra, coste_falso_positivo=1, coste_falso_negativo=1,
                              coste_de_revision=coste_de_revision)
        contradicciones_sin_umbral += any(
            viejo.banda_de(pi) != "revision" and (viejo.banda_de(pi) == "positiva") != (pi >= t)
            for pi in p)
    if coste_de_revision in (0.5, 0.9):
        # Control: sin `umbral=` el defecto está ahí; si no lo estuviera, esta prueba no miraría nada.
        assert contradicciones_sin_umbral > 0


def test_el_par_de_la_auditoria_umbral_0737_y_bandas_0717():
    """El caso con nombre: la semilla y los costes con los que la auditoría lo encontró."""
    encontrado = None
    rng = random.Random(5)
    for _ in range(5000):
        y, p = [], []
        for _ in range(120):
            pos = rng.random() < 0.35
            y.append("si" if pos else "no")
            p.append(round(min(1, max(0, rng.gauss(0.62 if pos else 0.4, 0.2))), 3))
        m = Muestra.binaria(y_true=tuple(y), classes=("no", "si"), positive_label="si",
                            probabilidades=tuple(p))
        t = elegir_umbral(m, cost_false_positive=1, cost_false_negative=1).threshold
        viejo = elegir_bandas(m, coste_falso_positivo=1, coste_falso_negativo=1, coste_de_revision=0.9)
        if not (viejo.umbral_bajo <= t <= viejo.umbral_alto):
            encontrado = (m, t, viejo)
            break
    assert encontrado is not None
    m, t, viejo = encontrado
    assert (t, viejo.umbral_bajo, viejo.umbral_alto) == (0.737, 0.717, 0.717)
    nuevo = elegir_bandas(m, coste_falso_positivo=1, coste_falso_negativo=1, coste_de_revision=0.9,
                          umbral=t)
    assert nuevo.umbral_bajo == nuevo.umbral_alto == t
    assert nuevo.banda_de(0.727) == "negativa"
