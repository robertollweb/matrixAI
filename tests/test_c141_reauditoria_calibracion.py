# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""141 (29-10) — lo que suspendió la reauditoría de la recalibración: B1, I1, M1–M4.

B1. La regla «b < 0 no se acepta» (decisión de Roberto del 28-10, A) estaba DENTRO de `ajustar_recalibracion_logistica`,
que es también el instrumento con el que `evaluacion_de_puntuacion` MIDE la pendiente de calibración y su IC por
remuestreo. Ahí cada remuestra con b < 0 salía «degenerada» y el IC quedaba truncado en 0 (medido por la reauditoría:
[0,0093; 0,3246] con la regla, [−0,1547; 0,3184] sin ella, sobre los mismos datos). Ahora el ajuste devuelve la pendiente
medida, con su signo, y la regla es `aplicar_la_regla_de_la_pendiente`, que llama el estudio.

I1. «Pendiente negativa» no es «ordena al revés»: se dice lo medido (b y el AUROC de esas filas), y un |b| por debajo de
`TOLERANCIA_DE_LA_PENDIENTE` es «sin pendiente», no un signo.

M1/M2. El Newton acota el paso y cae al gradiente antes de rendirse; las dos ramas de rescate tienen prueba con nombre.

M3/M4. La puntuación constante y la separación al revés tienen su propia causa en las curvas.
"""
from __future__ import annotations

import math
import random

import pytest

from matrixai.estudio import calibracion as cal
from matrixai.estudio.calibracion import (
    PASO_MAXIMO,
    TOLERANCIA_DE_LA_PENDIENTE,
    _logit,
    _newton_logistico,
    _sigmoid,
    ajustar_recalibracion_logistica,
    aplicar_la_regla_de_la_pendiente,
)
from matrixai.estudio.curvas_de_la_evaluacion import _causa_de_la_no_ajustada, curvas_de_la_evaluacion
from matrixai.estudio.evaluacion_de_puntuacion import DeclaracionDePuntuacion, evaluar_puntuacion
from matrixai.estudio.metricas import Muestra


def _muestra(ps, ys) -> Muestra:
    return Muestra.binaria(["si" if y else "no" for y in ys], classes=("no", "si"), positive_label="si",
                           probabilidades=list(ps))


def _sklearn(ps, ys):
    lineal = pytest.importorskip("sklearn.linear_model")
    np = pytest.importorskip("numpy")
    z = np.array([_logit(p) for p in ps]).reshape(-1, 1)
    lr = lineal.LogisticRegression(C=1e12, max_iter=100000, tol=1e-12).fit(z, list(ys))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def _busca(o, clave):
    if isinstance(o, dict):
        if clave in o:
            return o[clave]
        for v in o.values():
            if (x := _busca(v, clave)) is not None:
                return x
    if isinstance(o, list):
        for v in o:
            if (x := _busca(v, clave)) is not None:
                return x
    return None


def _csv_de_una_puntuacion(pendiente_verdadera: float, n: int, semilla: int) -> str:
    """La forma de `sonda_evaluar_puntuacion.py` de la reauditoría: una probabilidad declarada `p = sigmoid(z)` y un
    desenlace con `P(y) = sigmoid(pendiente_verdadera · z)` (una puntuación débil si la pendiente es pequeña)."""
    rng = random.Random(semilla)
    lineas = ["riesgo,muerte"]
    for _ in range(n):
        z = rng.gauss(0, 1.5)
        p = 1 / (1 + math.exp(-z))
        lineas.append(f"{p:.6f},{'si' if rng.random() < 1 / (1 + math.exp(-pendiente_verdadera * z)) else 'no'}")
    return "\n".join(lineas)


def _pendiente_publicada(pendiente_verdadera: float, n: int, semilla: int, remuestras: int = 300) -> dict:
    r = evaluar_puntuacion(
        _csv_de_una_puntuacion(pendiente_verdadera, n, semilla), columna_desenlace="muerte", valor_del_evento="si",
        puntuaciones=[DeclaracionDePuntuacion(columna="riesgo", tipo="probabilidad", direccion="mas_alto_mas_riesgo",
                                              casos_en_su_desarrollo="no")],
        muestra_representativa=True, horizonte_del_desenlace="ingreso", momento_de_la_puntuacion="al ingreso",
        donde_se_calcula="local", remuestras=remuestras, semilla=137)
    informe = r.informe if hasattr(r, "informe") else r
    js = informe if isinstance(informe, dict) else informe.a_json()
    pendiente = _busca(js, "pendiente")
    assert pendiente is not None, "el informe no trae la pendiente de calibración"
    return pendiente


# ======================================================================================================================
# B1 (a) — el INSTRUMENTO: la evaluación de una puntuación publica la pendiente medida y su IC, con los dos signos
# ======================================================================================================================

def test_el_ic_de_la_pendiente_de_evaluar_puntuacion_incluye_pendientes_negativas():
    """Pendiente verdadera 0,1 con 150 filas: el IC tiene que poder bajar de cero, y ninguna remuestra es «degenerada»
    por tener b < 0. Con la regla dentro del ajuste salía [0,0093; 0,3246] con 400 de 500 remuestras válidas."""
    pendiente = _pendiente_publicada(0.1, 150, 4)
    ic = pendiente["ic"]
    assert ic is not None, pendiente
    assert ic["bajo"] < 0.0 < ic["alto"], ic
    assert ic["remuestras_validas"] == ic["remuestras"], ic


def test_con_pendiente_verdadera_cero_el_ic_existe_y_contiene_el_cero():
    """b verdadera = 0 con 300 filas: con la regla dentro del ajuste el IC DESAPARECÍA con un motivo falso («207 de 500
    remuestras salieron degeneradas (una sola clase…)»)."""
    pendiente = _pendiente_publicada(0.0, 300, 7)
    assert pendiente["ic"] is not None and pendiente.get("ic_no_disponible") in (None, {}), pendiente
    assert pendiente["ic"]["bajo"] < 0.0 < pendiente["ic"]["alto"], pendiente["ic"]


def test_una_pendiente_puntual_negativa_se_publica_con_su_valor():
    """Una puntuación que va al revés tiene pendiente negativa: es una MEDIDA, no un «no definido»."""
    pendiente = _pendiente_publicada(-0.8, 300, 11)
    assert pendiente["valor"] is not None and pendiente["valor"] < 0.0, pendiente
    assert not pendiente.get("no_definido"), pendiente


def test_el_ajuste_devuelve_la_pendiente_negativa_tal_como_se_mide():
    rng = random.Random(7)
    zs = [rng.gauss(0.0, 1.5) for _ in range(400)]
    ys = [1 if rng.random() < _sigmoid(-1.2 * z) else 0 for z in zs]
    ps = [_sigmoid(z) for z in zs]
    r = ajustar_recalibracion_logistica(_muestra(ps, ys))
    assert r.convergio and r.undefined_reason is None and r.b < 0
    a_ref, b_ref = _sklearn(ps, ys)
    assert r.a == pytest.approx(a_ref, abs=1e-6) and r.b == pytest.approx(b_ref, abs=1e-6)


# ======================================================================================================================
# B1 (b) e I1 — la DECISIÓN del estudio, y lo que dice
# ======================================================================================================================

def _con_atipicos(semilla: int = 3, n: int = 50, k: int = 3):
    """`sonda_atipicos.py` de la reauditoría: n filas bien ordenadas (y ~ Bernoulli(p)) y k con p = 1 e y = 0 y k con p = 0
    e y = 1 (recortadas a logit ±23). Arrastran b por debajo de cero con un modelo que ordena bien la mayoría."""
    rng = random.Random(semilla)
    ps, ys = [], []
    for _ in range(n):
        p = _sigmoid(rng.gauss(0, 3))
        ps.append(p)
        ys.append(1 if rng.random() < p else 0)
    return ps + [1.0] * k + [0.0] * k, ys + [0] * k + [1] * k


def test_una_pendiente_negativa_con_auroc_por_encima_del_azar_no_dice_que_ordena_al_reves():
    ps, ys = _con_atipicos()
    m = _muestra(ps, ys)
    medida = ajustar_recalibracion_logistica(m)
    assert medida.b is not None and medida.b < -TOLERANCIA_DE_LA_PENDIENTE, medida
    r = aplicar_la_regla_de_la_pendiente(medida, m)
    assert r.a is None and r.b is None and r.pendiente_descartada == medida.b
    assert r.auroc_de_calibracion is not None and r.auroc_de_calibracion > 0.5
    es, en = r.undefined_reason["es"], r.undefined_reason["en"]
    assert "pendiente negativa" in es and "negative slope" in en
    assert "no ordena al revés" in es and "does not rank the wrong way round" in en
    assert "por debajo del 0,5" not in es and "below chance" not in en
    assert _causa_de_la_no_ajustada(r) == "pendiente_negativa"
    # la frase de las curvas lleva lo medido, y tampoco afirma la inversión
    curvas = curvas_de_la_evaluacion(m, recalibracion=r)
    frase = curvas["recalibracion"]["motivo"]
    assert curvas["recalibracion"]["causa"] == "pendiente_negativa"
    assert "no ordena al revés" in frase["es"] and "does not rank the wrong way round" in frase["en"]
    assert f"{r.auroc_de_calibracion:.3f}".rstrip("0").replace(".", ",") in frase["es"]


def test_una_pendiente_negativa_con_auroc_por_debajo_del_azar_dice_que_ordena_al_reves():
    rng = random.Random(7)
    zs = [rng.gauss(0.0, 1.5) for _ in range(400)]
    ys = [1 if rng.random() < _sigmoid(-1.2 * z) else 0 for z in zs]
    m = _muestra([_sigmoid(z) for z in zs], ys)
    r = aplicar_la_regla_de_la_pendiente(ajustar_recalibracion_logistica(m), m)
    assert r.a is None and r.auroc_de_calibracion < 0.5
    assert "ordena al revés (su AUROC es" in r.undefined_reason["es"]
    assert "ranks the wrong way round (its AUROC is" in r.undefined_reason["en"]
    assert _causa_de_la_no_ajustada(r) == "pendiente_negativa_al_reves"
    frase = curvas_de_la_evaluacion(m, recalibracion=r)["recalibracion"]["motivo"]
    assert "ordena al revés" in frase["es"] and "por debajo del 0,5" in frase["es"]
    assert "below chance's 0.5" in frase["en"]


@pytest.mark.parametrize("signo", [-1.0, 1.0])
def test_una_pendiente_que_no_se_distingue_de_cero_es_sin_pendiente_sea_cual_sea_su_signo(signo):
    """p = (0,2; 0,5; 0,8), y = (0, 1, 0): el máximo exacto es b = 0 y el ajuste da b ≈ −1,9e-16 (medido por la
    reauditoría). Ese signo es redondeo: ni «pendiente negativa» ni, con +1e-16, una recalibración PLANA aceptada."""
    m = _muestra([0.2, 0.5, 0.8], [0, 1, 0])
    medida = ajustar_recalibracion_logistica(m)
    assert medida.b is not None and abs(medida.b) <= TOLERANCIA_DE_LA_PENDIENTE
    casi_cero = cal.RecalibracionLogistica(metodo="logistico_completo", n_observaciones=3, a=medida.a,
                                           b=signo * 1e-16, convergio=True, iteraciones=1)
    r = aplicar_la_regla_de_la_pendiente(casi_cero, m)
    assert r.a is None and r.pendiente_descartada == signo * 1e-16
    assert "sin pendiente" in r.undefined_reason["es"] and "no slope" in r.undefined_reason["en"]
    assert "negativa" not in r.undefined_reason["es"]
    assert _causa_de_la_no_ajustada(r) == "sin_pendiente"


def test_justo_por_encima_de_la_tolerancia_una_pendiente_positiva_se_usa():
    m = _muestra([0.2, 0.5, 0.8, 0.9], [0, 1, 0, 1])
    r = cal.RecalibracionLogistica(metodo="logistico_completo", n_observaciones=4, a=0.1,
                                   b=2 * TOLERANCIA_DE_LA_PENDIENTE, convergio=True, iteraciones=3)
    assert aplicar_la_regla_de_la_pendiente(r, m) is r


def test_la_regla_no_toca_el_intercepto_en_grande_ni_lo_que_no_se_ajusto():
    rng = random.Random(7)
    zs = [rng.gauss(0.0, 1.5) for _ in range(200)]
    ys = [1 if rng.random() < _sigmoid(-1.2 * z) else 0 for z in zs]
    m = _muestra([_sigmoid(z) for z in zs], ys)
    en_grande = ajustar_recalibracion_logistica(m, metodo="intercepto_en_grande")
    assert aplicar_la_regla_de_la_pendiente(en_grande, m) is en_grande
    sin_ajustar = ajustar_recalibracion_logistica(_muestra([0.3, 0.4], [1, 1]))
    assert aplicar_la_regla_de_la_pendiente(sin_ajustar, m) is sin_ajustar
    assert aplicar_la_regla_de_la_pendiente(None, m) is None


def test_lo_que_descarta_la_regla_viaja_en_el_json_y_se_reconstruye():
    """El calibrador va al sobre y al checkpoint (`a_json`) y la inferencia y el simulador lo reconstruyen con
    `RecalibracionLogistica(**json)`: la pendiente descartada y el AUROC tienen que sobrevivir el viaje, y un calibrador
    de siempre no gana claves nuevas (su digest no cambia)."""
    ps, ys = _con_atipicos()
    m = _muestra(ps, ys)
    r = aplicar_la_regla_de_la_pendiente(ajustar_recalibracion_logistica(m), m)
    vuelta = cal.RecalibracionLogistica(**r.a_json())
    assert vuelta == r and _causa_de_la_no_ajustada(vuelta) == "pendiente_negativa"
    normal = ajustar_recalibracion_logistica(_muestra([0.1, 0.4, 0.6, 0.9, 0.3, 0.7], [0, 1, 0, 1, 0, 1]))
    assert "pendiente_descartada" not in normal.a_json() and "auroc_de_calibracion" not in normal.a_json()


# ======================================================================================================================
# M1 y M2 — el Newton: paso acotado, caída al gradiente, y «el máximo cabe en el ruido»
# ======================================================================================================================

#: `sonda_newton.py 1 800` de la reauditoría, caso 555: el Newton sin acotar salía «no convergió» en 2 iteraciones (el
#: segundo paso, del orden de 1e24). sklearn (C = 1e12) da a = −1,89923, b = −2,25882.
_CASO_555_PS = [0.9995546588902962, 0.32541657244358035, 0.9999999769731557, 0.9729112577178554, 0.29821712765566344,
                2.0239329469826615e-16, 0.012865479289283542, 0.00028494828411814826, 3.03366386403997e-06, 1.0,
                0.999308740541033, 0.9999999956881815, 5.322349990607557e-12, 0.0030610937298287087, 0.6037024351408794]
_CASO_555_YS = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]


def test_con_logits_recortados_el_newton_acotado_converge_donde_antes_no_convergia():
    r = ajustar_recalibracion_logistica(_muestra(_CASO_555_PS, _CASO_555_YS))
    assert r.convergio and r.undefined_reason is None, r
    assert r.a == pytest.approx(-1.8992303355609303, abs=1e-4)
    assert r.b == pytest.approx(-2.258816672531316, abs=1e-4)


def test_ningun_paso_de_newton_pasa_del_paso_maximo_y_sin_la_cota_alguno_pasaria(monkeypatch):
    """La cota ACTÚA en este caso: algún paso de Newton sin acotar sale mucho más largo que `PASO_MAXIMO` (el del orden de 1e24 que
    medía la reauditoría) y ninguno de los que se dan lo pasa. Y la cota es un número, no un infinito que no acota nada."""
    assert math.isfinite(PASO_MAXIMO) and 0 < PASO_MAXIMO <= 100
    sin_acotar: list[float] = []
    dados: list[float] = []
    original_paso, original_linea = cal._paso, cal._buscar_en_la_linea

    def espia_paso(ga, gb, haa, hab, hbb, libre_b):
        det = haa * hbb - hab * hab
        if libre_b and haa < 0.0 and det > 0.0:
            sin_acotar.append(math.hypot(-(hbb * ga - hab * gb) / det, -(-hab * ga + haa * gb) / det))
        return original_paso(ga, gb, haa, hab, hbb, libre_b)

    def espia_linea(zs, ys, a, b, da, db, *resto):
        dados.append(math.hypot(da, db))
        return original_linea(zs, ys, a, b, da, db, *resto)

    monkeypatch.setattr(cal, "_paso", espia_paso)
    monkeypatch.setattr(cal, "_buscar_en_la_linea", espia_linea)
    ajustar_recalibracion_logistica(_muestra(_CASO_555_PS, _CASO_555_YS))
    assert max(sin_acotar) > 10 * PASO_MAXIMO, max(sin_acotar)
    assert dados and max(dados) <= PASO_MAXIMO * (1 + 1e-12), max(dados)


def test_la_cota_sola_basta_sin_la_caida_al_gradiente(monkeypatch):
    """Las dos defensas del M1 son independientes: con la caída al gradiente apagada, la cota sola ajusta el caso 555."""
    monkeypatch.setattr(cal, "_paso_del_gradiente", lambda ga, gb: (_ for _ in ()).throw(AssertionError("no debía hacer falta")))
    r = ajustar_recalibracion_logistica(_muestra(_CASO_555_PS, _CASO_555_YS))
    assert r.convergio and r.a == pytest.approx(-1.8992303355609303, abs=1e-4)


#: `sonda_newton.py 3 3000` (generador de la reauditoría), caso 2167: en el camino, el hessiano deja de ser definido
#: negativo y el paso es el del gradiente. Sin esa rama (siempre Newton) el ajuste divide por un determinante nulo.
_CASO_2167_PS = [0.9999937853423684, 1.0, 0.9999978463529969, 0.9999999999999971, 0.0003552630643514497]
_CASO_2167_YS = [0, 0, 1, 0, 1]


def test_con_el_hessiano_no_definido_el_paso_es_el_del_gradiente_y_llega_al_maximo(monkeypatch):
    pasos: list[bool] = []
    original = cal._paso

    def espia(*args):
        r = original(*args)
        pasos.append(r[2])
        return r

    monkeypatch.setattr(cal, "_paso", espia)
    r = ajustar_recalibracion_logistica(_muestra(_CASO_2167_PS, _CASO_2167_YS))
    assert False in pasos, "el caso ya no pasa por el paso del gradiente: la prueba no mide la rama"
    assert r.convergio and r.a is not None
    assert r.a == pytest.approx(4.077084308931938, abs=1e-4) and r.b == pytest.approx(-0.33396088514511796, abs=1e-4)


def test_si_ninguna_mitad_del_paso_de_newton_mejora_se_prueba_el_gradiente_antes_de_rendirse(monkeypatch):
    """La caída al gradiente tras agotar las mitades: se fuerza que la búsqueda en línea del PRIMER paso de Newton no
    encuentre mejora. El ajuste tiene que intentar la dirección del gradiente y seguir hasta el máximo de sklearn; sin
    esa caída, se rendía en la primera iteración («no convergió»)."""
    ps, ys = _CASO_555_PS, _CASO_555_YS
    llamadas: list[tuple[float, float]] = []
    original = cal._buscar_en_la_linea

    def falla_la_primera(zs, ys_, a, b, da, db, *resto):
        llamadas.append((da, db))
        if len(llamadas) == 1:
            return None
        return original(zs, ys_, a, b, da, db, *resto)

    monkeypatch.setattr(cal, "_buscar_en_la_linea", falla_la_primera)
    r = ajustar_recalibracion_logistica(_muestra(ps, ys))
    assert len(llamadas) >= 2
    # el segundo intento es la dirección del gradiente en el mismo punto (norma ≤ 1)
    assert math.hypot(*llamadas[1]) <= 1.0 + 1e-12
    assert r.convergio and r.a == pytest.approx(-1.8992303355609303, abs=1e-4), r


def _casi_en_el_maximo(n: int = 400):
    """`y` fraccionario = sigmoid(z) salvo una fila movida 1e-6: el máximo está a ~1e-9 de (0, 1), el arranque."""
    rng = random.Random(5)
    zs = [rng.gauss(0.0, 2.0) for _ in range(n)]
    ys = [_sigmoid(z) for z in zs]
    ys[0] += 1e-6
    return zs, ys


@pytest.mark.parametrize("fijar_b", [None, 1.0])
def test_si_ningun_trozo_mejora_y_lo_prometido_cabe_en_el_ruido_es_el_maximo(monkeypatch, fijar_b):
    """«El máximo cabe en el ruido»: se fuerza que ninguna búsqueda en línea mejore, arrancando a ~1e-9 del máximo (el
    paso es mayor que la tolerancia, pero lo que promete subir la verosimilitud es ~1e-15, muy por debajo de la holgura de
    redondeo). Eso es el máximo, y se declara convergido; declararlo «no convergió» descartaría un ajuste bueno.
    Con `y` fraccionario no hay «clases» que separar: la comprobación de separación (que exige 0/1) se aparta."""
    zs, ys = _casi_en_el_maximo()
    monkeypatch.setattr(cal, "_buscar_en_la_linea", lambda *a, **k: None)
    monkeypatch.setattr(cal, "_separacion_en_los_datos", lambda *a, **k: None)
    paso = cal._paso(*cal._evaluar(zs, ys, 0.0, 1.0, fijar_b is None)[1:], fijar_b is None)
    assert max(abs(paso[0]), abs(paso[1])) > 1e-10, "el paso cabe en la tolerancia: la prueba no llegaría a la rama"
    a, b, it, convergio, separacion = _newton_logistico(zs, ys, fijar_b=fijar_b)
    assert convergio and not separacion and it == 1
    assert a == 0.0 and b == 1.0


def test_si_ningun_trozo_mejora_lejos_del_maximo_no_convergio(monkeypatch):
    """El lado contrario: lejos del máximo, lo prometido NO cabe en el ruido y se dice «no convergió»."""
    zs, ys = _casi_en_el_maximo()
    ys = [1.0 if y > 0.5 else 0.0 for y in ys]
    ys[0], ys[1] = 1.0 - ys[0], 1.0 - ys[1]          # que no separen
    monkeypatch.setattr(cal, "_buscar_en_la_linea", lambda *a, **k: None)
    _a, _b, _it, convergio, separacion = _newton_logistico(zs, ys, fijar_b=None)
    assert not convergio and not separacion


# ======================================================================================================================
# M3 y M4 — causas propias en las curvas
# ======================================================================================================================

def test_la_puntuacion_constante_tiene_su_causa_y_su_frase_en_las_curvas():
    r = ajustar_recalibracion_logistica(_muestra([0.4] * 6, [0, 1, 0, 1, 1, 0]))
    assert r.a is None
    assert _causa_de_la_no_ajustada(r) == "puntuacion_constante"
    frase = curvas_de_la_evaluacion(_muestra([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1]), recalibracion=r)["recalibracion"]
    assert frase["causa"] == "puntuacion_constante"
    assert "misma puntuación" in frase["motivo"]["es"] and "same score" in frase["motivo"]["en"]


@pytest.mark.parametrize("ys,cuasi", [([1, 1, 1, 0, 0, 0], False), ([1, 1, 1, 1, 0, 0], True)])
def test_una_separacion_al_reves_se_dice_como_inversion(ys, cuasi):
    """Todas las de la clase positiva abajo (el modelo perfectamente invertido en esas filas): no es solo «separación»."""
    ps = [0.1, 0.2, 0.3, 0.4, 0.4, 0.9] if cuasi else [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    if cuasi:
        ys = [1, 1, 1, 1, 0, 0]          # empate en 0,4 entre una positiva y una negativa
    r = ajustar_recalibracion_logistica(_muestra(ps, ys))
    assert r.separacion_detectada and r.a is None
    assert "al revés" in r.undefined_reason["es"] and "wrong way round" in r.undefined_reason["en"]
    assert ("cuasi" in r.undefined_reason["es"]) == cuasi
    assert _causa_de_la_no_ajustada(r) == "separacion_al_reves"
    frase = curvas_de_la_evaluacion(_muestra(ps, ys), recalibracion=r)["recalibracion"]["motivo"]
    assert "al revés" in frase["es"] and "invertido" in frase["es"] and "inverted" in frase["en"]


def test_una_separacion_al_derecho_sigue_siendo_separacion():
    r = ajustar_recalibracion_logistica(_muestra([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0, 0, 0, 1, 1, 1]))
    assert r.separacion_detectada
    assert "al revés" not in r.undefined_reason["es"]
    assert _causa_de_la_no_ajustada(r) == "separacion"
