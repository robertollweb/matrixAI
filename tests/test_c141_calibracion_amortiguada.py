# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""141 (28-10) — la recalibración logística con logits extremos.

El defecto, hallado conduciendo el 141: `_newton_logistico` era un Newton SIN amortiguar que partía de `a=0, b=1`.
Con logits de −12 a +7 (los de un buen clasificador) su primer paso salía disparado, un coeficiente pasaba de 50 en
2–3 iteraciones y se declaraba «separación» con datos que NO son separables. En el estudio eso entregaba sin
recalibrar (climate y pc1 de la demo: «las puntuaciones ya separan del todo las dos clases» con AUROC 0,964 y 0,793).

El oráculo es `sklearn.linear_model.LogisticRegression` con `C` enorme (sin regularización efectiva) y `tol=1e-12`; para
los datos reales de la reproducción, además, sus cifras van escritas aquí, para que la prueba muerda sin sklearn.
Y el otro lado: la separación REAL (completa o con empate en el corte) se sigue declarando, sin coeficientes.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from matrixai.estudio.calibracion import (
    _evaluar,
    _logit,
    _newton_logistico,
    _separacion_en_los_datos,
    _sigmoid,
    ajustar_recalibracion_logistica,
)
from matrixai.estudio.metricas import Muestra

_FIXTURE = Path(__file__).parent / "fixtures_141_calibracion" / "climate_fuera_de_pliegue.json"

#: `LogisticRegression(C=1e12, max_iter=100000, tol=1e-12)` sobre cada subconjunto del fixture (medido el 28-10).
_SKLEARN_CLIMATE = [(0.379945, 0.555332), (-0.113248, 0.501334), (-0.142552, 0.557476),
                    (0.478639, 0.577168), (0.352196, 0.631248), (0.093565, 0.564217)]


def _muestra(ps, ys) -> Muestra:
    return Muestra.binaria(["si" if y else "no" for y in ys], classes=("no", "si"), positive_label="si",
                           probabilidades=list(ps))


def _sintetica(seed: int, n: int = 215, a: float = 0.4, b: float = 0.56) -> tuple[list[float], list[float]]:
    """Logits extremos NO separables: `z ~ U(−12, 7)`, `y ~ Bernoulli(sigmoid(a + b·z))` — la forma de la reproducción."""
    rng = random.Random(seed)
    zs = [rng.uniform(-12.0, 7.0) for _ in range(n)]
    ys = [1.0 if rng.random() < _sigmoid(a + b * z) else 0.0 for z in zs]
    return zs, ys


def _ll(zs, ys, a, b) -> float:
    return _evaluar(zs, ys, a, b, False)[0]


def _sklearn(zs, ys):
    lineal = pytest.importorskip("sklearn.linear_model")
    np = pytest.importorskip("numpy")
    lr = lineal.LogisticRegression(C=1e12, max_iter=100000, tol=1e-12).fit(np.array(zs).reshape(-1, 1), ys)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


# -- logits extremos que NO separan: se ajustan, y como sklearn ---------------------------------------------------------

@pytest.mark.parametrize("k", range(6))
def test_los_logits_de_climate_que_daban_separacion_se_ajustan_como_sklearn(k):
    """Los datos REALES de la reproducción (4 de estos 6 subconjuntos salían «separación» con el Newton sin amortiguar)."""
    sub = json.loads(_FIXTURE.read_text())["subconjuntos"][k]
    r = ajustar_recalibracion_logistica(_muestra(sub["p"], sub["y"]))
    assert r.convergio and not r.separacion_detectada and r.undefined_reason is None, r.a_json()
    a_ref, b_ref = _SKLEARN_CLIMATE[k]
    assert r.a == pytest.approx(a_ref, abs=1e-5) and r.b == pytest.approx(b_ref, abs=1e-5)


@pytest.mark.parametrize("seed", range(8))
def test_logits_extremos_sinteticos_no_separables_se_ajustan_como_sklearn(seed):
    zs, ys = _sintetica(seed)
    assert _separacion_en_los_datos(zs, ys) is None
    a, b, _it, convergio, separacion = _newton_logistico(zs, ys, fijar_b=None)
    assert convergio and not separacion
    a_ref, b_ref = _sklearn(zs, ys)
    assert a == pytest.approx(a_ref, abs=1e-5) and b == pytest.approx(b_ref, abs=1e-5)
    # y no se queda por debajo del máximo que encuentra sklearn
    assert _ll(zs, ys, a, b) >= _ll(zs, ys, a_ref, b_ref) - 1e-9


def test_el_amortiguado_nunca_empeora_la_verosimilitud_entre_iteraciones():
    """Lo que garantiza el amortiguado, medido: la log-verosimilitud de los puntos ACEPTADOS no baja nunca (con la
    holgura de redondeo declarada). Se graba con `max_iter` creciente: cada corte devuelve el punto aceptado en esa
    iteración."""
    zs, ys = _sintetica(0)
    lls = []
    for max_iter in range(1, 7):
        a, b, _it, _conv, _sep = _newton_logistico(zs, ys, fijar_b=None, max_iter=max_iter)
        lls.append(_ll(zs, ys, a, b))
    inicio = _ll(zs, ys, 0.0, 1.0)
    for antes, despues in zip([inicio] + lls, lls):
        assert despues >= antes - 1e-13 * (1 + abs(antes)), (antes, despues)


# -- la separación REAL se sigue declarando -----------------------------------------------------------------------------

def test_separacion_completa_con_logits_extremos_se_declara_sin_coeficientes():
    """El otro lado: un caso separable de verdad no sale «ajustado» con un b absurdo, por mucho que el Newton amortiguado
    pudiera seguir subiendo la verosimilitud hacia el infinito."""
    zs, _ = _sintetica(1)
    ps = [_sigmoid(z) for z in zs]
    ys = [1 if z > -2.0 else 0 for z in zs]
    r = ajustar_recalibracion_logistica(_muestra(ps, ys))
    assert r.separacion_detectada and r.a is None and r.b is None and not r.convergio
    assert "completa" in r.undefined_reason["es"] and "complete" in r.undefined_reason["en"]
    assert "cuasi" not in r.undefined_reason["es"]


def test_separacion_al_reves_tambien_se_declara():
    """Una puntuación que ordena AL REVÉS (las positivas abajo) también separa: `b → −∞`."""
    zs = [i / 10 for i in range(-50, 50)]
    ys = [1 if z < 0 else 0 for z in zs]
    r = ajustar_recalibracion_logistica(_muestra([_sigmoid(z) for z in zs], ys))
    assert r.separacion_detectada and r.a is None


def test_cuasi_separacion_con_empate_en_el_corte_se_declara_como_cuasi():
    """Empate en el corte: las dos clases comparten SOLO el valor frontera. El máximo tampoco existe (Albert y Anderson)."""
    zs = [-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    ys = [0, 0, 0, 0, 1, 1, 1, 1]
    assert _separacion_en_los_datos(zs, [float(y) for y in ys]) == "cuasi_completa"
    r = ajustar_recalibracion_logistica(_muestra([_sigmoid(z) for z in zs], ys))
    assert r.separacion_detectada and r.a is None
    assert "cuasi" in r.undefined_reason["es"] and "quasi" in r.undefined_reason["en"]


def test_casi_separable_con_un_solo_solape_se_ajusta_con_el_maximo_finito():
    """UN punto en el lado contrario basta para que el máximo exista: se ajusta, con b grande y el MISMO que sklearn —
    no se declara separación por tener un coeficiente grande."""
    zs = [i / 4 for i in range(-20, 21)]
    ys = [1.0 if z > 0 else 0.0 for z in zs]
    ys[zs.index(-0.25)] = 1.0       # un solo positivo por debajo del corte
    assert _separacion_en_los_datos(zs, ys) is None
    a, b, _it, convergio, separacion = _newton_logistico(zs, ys, fijar_b=None)
    assert convergio and not separacion and b > 3
    a_ref, b_ref = _sklearn(zs, ys)
    assert a == pytest.approx(a_ref, rel=1e-4, abs=1e-4) and b == pytest.approx(b_ref, rel=1e-4)


# -- los bordes ---------------------------------------------------------------------------------------------------------

def test_todo_una_clase_con_logits_extremos_es_una_sola_clase_no_separacion():
    zs, _ = _sintetica(2)
    r = ajustar_recalibracion_logistica(_muestra([_sigmoid(z) for z in zs], [0] * len(zs)))
    assert r.a is None and not r.separacion_detectada and "misma clase" in r.undefined_reason["es"]


def test_pocas_filas_que_se_solapan_se_ajustan_como_sklearn():
    """Tres filas: positivas a los dos lados de la negativa. Por simetría, b = 0 y a = log 2."""
    zs, ys = [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]
    a, b, _it, convergio, separacion = _newton_logistico(zs, ys, fijar_b=None)
    assert convergio and not separacion
    assert a == pytest.approx(math.log(2), abs=1e-8) and b == pytest.approx(0.0, abs=1e-8)


def test_dos_filas_de_clases_distintas_son_separacion():
    r = ajustar_recalibracion_logistica(_muestra([0.2, 0.7], [0, 1]))
    assert r.separacion_detectada and r.a is None


def test_puntuacion_constante_no_es_separacion_y_lleva_su_motivo():
    """Todas las puntuaciones iguales (un modelo que predice la prevalencia): el viejo lo declaraba separación por el
    hessiano singular. No hay corte que deje una clase a cada lado; lo que no se identifica es la pendiente."""
    r = ajustar_recalibracion_logistica(_muestra([0.3] * 20, [1] * 6 + [0] * 14))
    assert r.a is None and not r.separacion_detectada
    assert "la misma" in r.undefined_reason["es"] and "the same" in r.undefined_reason["en"]
    # con b fijo SÍ se ajusta: a = logit(6/20) − logit(0,3) = 0
    r1 = ajustar_recalibracion_logistica(_muestra([0.3] * 20, [1] * 6 + [0] * 14), metodo="intercepto_en_grande")
    assert r1.convergio and r1.a == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("seed", range(4))
def test_fijar_b_con_logits_extremos_resuelve_la_ecuacion_del_intercepto(seed):
    """Con b fijo a 1 el máximo de `a` existe siempre que haya las dos clases — también con datos separables. Se
    comprueba contra la raíz de `Σ(y − sigmoid(a + z)) = 0` por bisección."""
    zs, ys = _sintetica(seed)
    if seed % 2:
        ys = [1.0 if z > 0 else 0.0 for z in zs]       # separable: con b fijo no importa
    a, b, _it, convergio, separacion = _newton_logistico(zs, ys, fijar_b=1.0)
    assert convergio and not separacion and b == 1.0

    def g(x):
        return sum(y - _sigmoid(x + z) for z, y in zip(zs, ys))

    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
    assert a == pytest.approx((lo + hi) / 2, abs=1e-7)


def test_no_converger_sigue_existiendo_con_su_motivo():
    zs, ys = _sintetica(0)
    r = ajustar_recalibracion_logistica(_muestra([_sigmoid(z) for z in zs], [int(y) for y in ys]), max_iter=2)
    assert r.a is None and not r.convergio and not r.separacion_detectada and r.iteraciones == 2
    assert "no convergió en 2 iteraciones" in r.undefined_reason["es"]


def test_probabilidades_exactas_de_cero_y_uno_no_rompen_el_ajuste():
    """0 y 1 exactos se recortan a ±23 en logit; con solapamiento sigue habiendo máximo finito."""
    ps = [0.0, 0.0, 1e-6, 0.3, 0.6, 0.999999, 1.0, 1.0, 0.0, 1.0]
    ys = [0, 0, 0, 1, 0, 1, 1, 1, 1, 0]
    r = ajustar_recalibracion_logistica(_muestra(ps, ys))
    assert r.convergio and r.a is not None and math.isfinite(r.a) and math.isfinite(r.b)
    zs = [_logit(p) for p in ps]
    a_ref, b_ref = _sklearn(zs, [float(y) for y in ys])
    assert r.a == pytest.approx(a_ref, abs=1e-4) and r.b == pytest.approx(b_ref, abs=1e-4)


# Decisión de Roberto (28-10 22:49, A): una recalibración con pendiente NEGATIVA no se acepta. El modelo ordena al revés en
# las filas de calibración; darle la vuelta en silencio escondería que predice peor que el azar (y la guarda del campeón,
# `test_128_r3_*`, dejaba de apartarlo). Se declara sin ajustar, con su motivo, y las curvas dicen esa causa.
def test_una_pendiente_negativa_se_declara_sin_ajustar_con_su_motivo_y_las_curvas_dicen_por_que():
    from matrixai.estudio.curvas_de_la_evaluacion import _causa_de_la_no_ajustada
    rng = random.Random(7)
    zs = [rng.gauss(0.0, 1.5) for _ in range(400)]
    ys = [1.0 if rng.random() < _sigmoid(-1.2 * z) else 0.0 for z in zs]   # cuanto más alta la puntuación, MENOS positivos
    r = ajustar_recalibracion_logistica(_muestra([_sigmoid(z) for z in zs], ys))
    assert r.a is None and r.b is None
    assert r.undefined_reason is not None and "pendiente negativa" in r.undefined_reason["es"]
    assert "negative slope" in r.undefined_reason["en"]
    assert _causa_de_la_no_ajustada(r) == "pendiente_negativa"
    # Y el lado contrario: la misma muestra con la puntuación al derecho se ajusta (b > 0).
    ok = ajustar_recalibracion_logistica(_muestra([_sigmoid(-z) for z in zs], ys))
    assert ok.b is not None and ok.b > 0
