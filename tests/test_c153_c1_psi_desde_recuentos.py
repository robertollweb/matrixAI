"""153-C1 — el PSI desde RECUENTOS es el PSI de los datos (criterio 2 del contrato 153) y «ausente no es cero» (criterio 7).

El servicio exportado no guarda un valor suelto: guarda, hora a hora, cuántos valores cayeron en cada una de las 10 casillas
de la referencia de entrenamiento. `psi_desde_recuentos` tiene que dar, desde esos recuentos, EXACTAMENTE lo que
`compute_psi` da desde los datos (diferencia 0,0: no una tolerancia, la igualdad de bits), porque si no la traza y el CSV
subido a mano dirían cosas distintas de la misma tabla.

Medido en C0 (3 columnas × 3 escenarios, diferencia 0,0) y re-medido aquí con el código que va al núcleo.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import pytest

from matrixai.continual.drift import compute_psi, psi_desde_recuentos, recuentos_en_casillas

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLOS = RAIZ.parent / "matrixaistudio" / "studio-backend" / "matrixai_studio" / "ejemplos_medidos"
COLUMNAS_REALES = [
    ("climate-model-simulation-crashes.csv", "vconst_corr"),
    ("parkinsons-telemonitoring.csv", "Jitter(%)"),    # muy sesgada a la derecha
    ("cardiotocography.csv", "frecuencia_base"),       # enteros, otra tabla
]


def _columna(fichero, nombre):
    with open(EJEMPLOS / fichero, newline="", encoding="utf-8") as fh:
        valores = []
        for fila in csv.DictReader(fh):
            try:
                valores.append(float(fila[nombre]))
            except ValueError:
                pass  # un hueco no es un valor
    return valores


def _escenarios(validos):
    corte = int(len(validos) * 0.7)
    ref, natural = validos[:corte], validos[corte:]
    span = max(ref) - min(ref)
    return ref, {
        "natural": natural,
        "desplazado_con_fuera_de_rango": [v + 0.30 * span for v in natural],
        "cola_extrema": sorted(validos)[int(len(validos) * 0.95):],
        "todo_en_una_casilla": [min(ref)] * 50,
    }


@pytest.mark.skipif(not EJEMPLOS.is_dir(), reason="sin ../matrixaistudio al lado: el catálogo de ejemplos reales no está (la prueba sintética de abajo corre igual)")
@pytest.mark.parametrize("fichero,columna", COLUMNAS_REALES)
def test_psi_desde_recuentos_es_igual_bit_a_bit_a_compute_psi_en_columnas_reales_del_catalogo(fichero, columna):
    validos = _columna(fichero, columna)
    assert len(validos) > 100
    ref, escenarios = _escenarios(validos)
    minimo, maximo = min(ref), max(ref)
    ref_cnt, ref_falt, _ = recuentos_en_casillas(ref, minimo, maximo)
    assert ref_falt == 0 and sum(ref_cnt) == len(ref)
    psis = []
    for nombre, obs in escenarios.items():
        obs_cnt, falt, fuera = recuentos_en_casillas(obs, minimo, maximo)
        assert falt == 0 and sum(obs_cnt) == len(obs)
        desde_datos = compute_psi(ref, obs)
        desde_recuentos = psi_desde_recuentos(ref_cnt, obs_cnt)
        assert desde_recuentos == desde_datos, (nombre, desde_recuentos, desde_datos)   # igualdad de bits, no tolerancia
        assert abs(desde_recuentos - desde_datos) == 0.0
        psis.append(desde_recuentos)
        if nombre == "desplazado_con_fuera_de_rango":
            assert fuera > 0  # el escenario ejerce la ruta de los valores fuera de la referencia
    assert len(set(psis)) > 1 and max(psis) > 0.25  # el banco ve deriva de verdad: no son cuatro ceros


def test_psi_desde_recuentos_bit_a_bit_en_datos_sinteticos_de_todas_las_formas():
    """Sin el catálogo: 300 pares referencia/observado al azar (también con valores fuera y casillas vacías)."""
    rng = random.Random(153)
    for _ in range(300):
        n_ref, n_obs = rng.randint(20, 400), rng.randint(1, 400)
        ref = [rng.gauss(0, 1) ** 3 for _ in range(n_ref)]
        despl = rng.choice([0.0, 0.5, 3.0])
        obs = [rng.gauss(despl, rng.choice([0.5, 1, 4])) ** 3 for _ in range(n_obs)]
        c_ref, _, _ = recuentos_en_casillas(ref, min(ref), max(ref))
        c_obs, _, _ = recuentos_en_casillas(obs, min(ref), max(ref))
        assert psi_desde_recuentos(c_ref, c_obs) == compute_psi(ref, obs)


def test_recuentos_en_casillas_usa_la_casilla_de_compute_psi_y_cuenta_fuera_y_faltantes():
    cuenta, falt, fuera = recuentos_en_casillas([0.0, 0.05, 0.1, 0.99, 1.0, -3.0, 7.0, None, float("nan"), float("inf"), float("-inf")], 0.0, 1.0)
    assert cuenta == [4, 1, 0, 0, 0, 0, 0, 0, 0, 4]  # 0.0, 0.05, -3.0 y -inf en la 0; 0.1 en la 1; 0.99, 1.0, 7.0 e inf en la 9
    assert sum(cuenta) == 9 and falt == 2 and fuera == 4
    # el máximo cae en la última casilla (int(10.0) = 10 apretado a 9), igual que en compute_psi
    assert recuentos_en_casillas([1.0], 0.0, 1.0)[0][9] == 1
    # y la misma casilla que compute_psi, valor a valor, en todo el rango
    for v in [i / 997 for i in range(998)]:
        c, _, _ = recuentos_en_casillas([v], 0.0, 1.0)
        c_psi_ref = [i for i, x in enumerate(c) if x][0]
        assert c_psi_ref == max(0, min(int((v - 0.0) / 1.0 * 10), 9))


def test_recuentos_en_casillas_una_referencia_sin_amplitud_no_define_casillas():
    for mn, mx in ((1.0, 1.0), (2.0, 1.0), (0.0, float("inf")), (float("nan"), 1.0)):
        with pytest.raises(ValueError):
            recuentos_en_casillas([0.5], mn, mx)


# ───────────────────────────────────────────────── ausente no es cero

def test_psi_con_un_lado_vacio_es_None_no_cero_y_compute_psi_sigue_dando_su_cero():
    lleno = [5, 3, 0, 2, 4, 6, 1, 2, 3, 4]
    vacio = [0] * 10
    assert psi_desde_recuentos(lleno, vacio) is None            # sin observaciones: no se puede medir
    assert psi_desde_recuentos(vacio, lleno) is None            # sin referencia
    assert psi_desde_recuentos(vacio, vacio) is None
    # el cero fabricado de compute_psi es justo lo que esto NO debe copiar
    assert compute_psi([], [1.0, 2.0]) == 0.0 and compute_psi([1.0, 2.0], []) == 0.0
    assert psi_desde_recuentos(lleno, lleno) == 0.0 and psi_desde_recuentos(lleno, lleno) is not None  # 0.0 MEDIDO sí es un cero


def test_psi_con_la_referencia_constante_es_None():
    """Una columna constante en entrenamiento: `compute_psi` da 0.0 (cero fabricado); desde recuentos, una sola casilla ocupada."""
    constante = [3.0] * 80
    assert compute_psi(constante, [1.0, 2.0, 3.0, 4.0]) == 0.0
    for una_casilla in ([80, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 80, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 80]):
        assert psi_desde_recuentos(una_casilla, [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]) is None
    # y una referencia con amplitud (mínimo y máximo siempre ocupan la casilla 0 y la última) sí mide
    assert psi_desde_recuentos([40, 0, 0, 0, 0, 0, 0, 0, 0, 40], [5] * 10) is not None


def test_psi_con_una_casilla_vacia_pesa_el_suelo_de_1e_menos_10_igual_que_compute_psi():
    ref = [i / 99 for i in range(100)]                       # 10 por casilla
    obs = [i / 99 * 0.8 for i in range(100)]                 # nada llega a la última casilla
    c_ref, _, _ = recuentos_en_casillas(ref, 0.0, 1.0)
    c_obs, _, _ = recuentos_en_casillas(obs, 0.0, 1.0)
    assert c_obs[9] == 0 and c_ref[9] == 10
    assert psi_desde_recuentos(c_ref, c_obs) == compute_psi(ref, obs)
    aporte = (1e-10 - 0.1) * math.log(1e-10 / 0.1)
    assert abs(aporte - 2.0723) < 1e-3  # una casilla vacía pesa ~2,07 de PSI (medido en C0)


def test_psi_desde_recuentos_rechaza_lo_que_no_son_recuentos():
    for ref, obs in (([1] * 10, [1] * 9), ([], []), ([1.5] * 10, [1] * 10), ([1] * 10, [-1] + [1] * 9),
                     ([True] * 10, [1] * 10), (["1"] * 10, [1] * 10)):
        with pytest.raises(ValueError):
            psi_desde_recuentos(ref, obs)


def test_recuentos_en_casillas_un_nan_de_numpy_es_un_faltante_no_un_error():
    """rep-c1 (menor 6): `np.float32('nan')` no es un `float` de Python; antes llegaba a `int(nan)` y lanzaba. Ahora es un
    faltante, como el NaN de Python y el `np.float64`, y no entra en ninguna casilla."""
    np = pytest.importorskip("numpy")
    casillas, faltantes, fuera = recuentos_en_casillas(
        [np.float32("nan"), np.float64("nan"), float("nan"), None, np.float32(0.55), np.int64(0)], 0.0, 1.0)
    assert (faltantes, fuera, sum(casillas)) == (4, 0, 2)
    assert casillas[5] == 1 and casillas[0] == 1
