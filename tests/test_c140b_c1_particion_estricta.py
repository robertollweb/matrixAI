# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b-C1 — «por sujeto y por el tiempo» ESTRICTO (núcleo).

Cada prueba afirma DÓNDE caen las filas (qué ids en cada lado), no solo que «no falla». Los datos tienen la forma que
trae la gente: ids de sujeto de texto con ceros a la izquierda, fechas `DD/MM/AAAA`, sujetos de una sola fila, un sujeto
con el 40 % de las filas, el CSV en cualquier orden y la clase positiva en pocos sujetos.

Oráculos: el `particion_por_diseno.py` de la rama 139 (congelado en `tests/data`) para la `groups_and_time` publicada
(BIT A BIT), `np.quantile` para el corte, y el `intervalo()` real de `incertidumbre.py` para el método declarado.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_c140_c1_particion_grupos_y_tiempo import ANTES, _firma   # noqa: E402  (el oráculo de la 139)

from matrixai.estudio import Horizonte, SplitPlan   # noqa: E402
from matrixai.training.particion_por_diseno import proponer_particion   # noqa: E402
from matrixai.training.particion_por_diseno_grupos_y_tiempo import (   # noqa: E402
    FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO, FRACCION_DE_LA_VENTANA, MINIMO_FILAS_DE_PRUEBA,
    MINIMO_UNIDADES_DE_PRUEBA, MINIMO_UNIDADES_POR_CLASE, escalera_del_corte, proponer_particion_estricta)
from matrixai.training.particion_por_diseno_textos import MOTIVOS, huecos_de   # noqa: E402

BASE = date(2024, 1, 13)
_PARKINSON = (Path(__file__).resolve().parents[2] / "matrixaistudio" / "studio-backend" / "matrixai_studio"
              / "ejemplos_medidos" / "parkinsons-telemonitoring.csv")


def _cohorte(n=30, visitas=20, *, entrada=0, seguimiento=100, tiempo="iso", semilla=1, prefijo="", ancho=3,
             desiguales=None, barajar=True):
    """Una cohorte de `n` sujetos. `entrada` > 0: cada sujeto entra en un día al azar de [0, entrada) (calendario
    escalonado); 0: panel (todos a la vez). Los sujetos pares tienen casi todo «si»; los impares, casi todo «no»."""
    rng = random.Random(semilla)
    filas = []
    for i in range(n):
        inicio = rng.randrange(entrada) if entrada else 0
        m = desiguales[i] if desiguales else visitas
        for j in range(m):
            dia = inicio + (j * seguimiento) // max(m, 1)
            valor = {"iso": (BASE + timedelta(days=dia)).isoformat(),
                     "dmy": (BASE + timedelta(days=dia)).strftime("%d/%m/%Y"),
                     "numero": str(dia)}[tiempo]
            y = "si" if rng.random() < (0.8 if i % 2 == 0 else 0.2) else "no"
            filas.append({"id": f"r{len(filas):05d}", "sujeto": f"{prefijo}{i:0{ancho}d}", "t": valor, "y": y,
                          "x": rng.random()})
    if barajar:
        rng.shuffle(filas)
    return filas


def _pide(**kw):
    base = dict(plan_id="p", observation_id_field="id", seed=3, unit_id_field="sujeto", time_column="t", objetivo="y")
    base.update(kw)
    return base


def _estricta(filas, **kw):
    return proponer_particion_estricta(filas, **_pide(**kw))


def _dia(valor):
    """Los días de una columna `iso`, `dmy` o numérica, para comparar sin pasar por el código probado."""
    if "/" in valor:
        d, m, a = valor.split("/")
        return (date(int(a), int(m), int(d)) - BASE).days
    if "-" in valor:
        return (date.fromisoformat(valor) - BASE).days
    return int(valor)


def _lados(filas, r):
    por_id = {f["id"]: f for f in filas}
    ids = lambda rol: sorted(o for o, x in r.plan.assignments.items() if x == rol)   # noqa: E731
    return por_id, ids("development"), ids("test")


def _invariantes(filas, r, hueco=0):
    """Lo que TODO reparto estricto cumple: ningún sujeto a los dos lados, la prueba posterior, y cuentas que cuadran."""
    por_id, dev, test = _lados(filas, r)
    unidades_dev = {por_id[o]["sujeto"] for o in dev}
    unidades_test = {por_id[o]["sujeto"] for o in test}
    assert unidades_dev and unidades_test and not (unidades_dev & unidades_test)
    assert max(_dia(por_id[o]["t"]) for o in dev) + hueco <= min(_dia(por_id[o]["t"]) for o in test)
    rep = r.reparto
    assert tuple(dev) == tuple(sorted(rep.desarrollo)) and tuple(test) == tuple(sorted(rep.prueba))
    assert len(rep.desarrollo) + len(rep.prueba) + rep.filas_descartadas == len(filas)
    assert rep.porcentaje_descartado == round(100 * rep.filas_descartadas / len(filas), 1)
    assert set(rep.descartadas) == {f["id"] for f in filas} - set(dev) - set(test)
    assert sorted(rep.unidades_de_prueba) == sorted(unidades_test)
    return por_id, dev, test


# ---------------------------------------------------------------------------
# (a) D1: la publicada sigue BIT A BIT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tiempo", ["fecha", "numero"])
@pytest.mark.parametrize("hueco", [None, Horizonte(magnitud=5, unidad="days")])
@pytest.mark.parametrize("calendario", [0, 300])
def test_la_groups_and_time_publicada_sigue_bit_a_bit(tiempo, hueco, calendario):
    """El oráculo de la 139 solo lee `date`/`datetime`/números (no texto): se le dan esos."""
    filas = _cohorte(30, 15, entrada=calendario, tiempo="iso" if tiempo == "fecha" else "numero")
    if tiempo == "fecha":
        filas = [dict(f, t=date.fromisoformat(f["t"])) for f in filas]
    else:
        filas = [dict(f, t=float(f["t"])) for f in filas]
        hueco = Horizonte(magnitud=5, unidad="seconds") if hueco is not None else None
    kw = dict(plan_id="p", observation_id_field="id", split_type="groups_and_time", seed=3, unit_id_field="sujeto",
              time_column="t", gap=hueco)
    ahora, antes = proponer_particion(filas, **kw), ANTES.proponer_particion(filas, **kw)
    assert _firma(ahora) == _firma(antes)
    assert ahora.pliegues is None       # la publicada NO da pliegues, y no los gana por existir la estricta


@pytest.mark.skipif(not _PARKINSON.exists(), reason="el CSV de Parkinson del Studio hermano no está")
def test_parkinson_la_publicada_da_lo_medido_en_el_c0_y_la_estricta_otra_cosa():
    """Control del C0: con `test_fraction` 0,2 la publicada deja 4.693 filas de 34 pacientes en desarrollo y 1.182 de 8 en
    la prueba, y el 90 % de esa prueba es ANTERIOR al final del desarrollo. La estricta, 0 %."""
    with open(_PARKINSON, newline="") as f:
        filas = list(csv.DictReader(f))
    for i, f in enumerate(filas):
        f["rid"] = str(i)
        f["y"] = "1" if float(f["total_UPDRS"]) >= 30 else "0"
    kw = dict(plan_id="p", observation_id_field="rid", unit_id_field="subject#", time_column="test_time", seed=0)
    publicada = proponer_particion(filas, split_type="groups_and_time", test_fraction=0.2, **kw)
    numericas = [dict(f, test_time=float(f["test_time"])) for f in filas]      # el oráculo de la 139 no lee texto
    assert _firma(publicada) == _firma(ANTES.proponer_particion(numericas, split_type="groups_and_time",
                                                                 test_fraction=0.2, **kw))
    t = {f["rid"]: float(f["test_time"]) for f in filas}
    dev = [o for o, x in publicada.plan.assignments.items() if x == "development"]
    test = [o for o, x in publicada.plan.assignments.items() if x == "test"]
    assert (len(dev), len(test)) == (4693, 1182)
    assert sum(t[o] < max(t[d] for d in dev) for o in test) == 1062          # el 90 %

    estricta = proponer_particion_estricta(filas, objetivo="y", **kw)
    assert estricta.es_viable and estricta.reparto.tipo_de_pliegues == "por_sujeto_y_ultimo_tramo"
    rep = estricta.reparto
    assert (len(rep.desarrollo), len(rep.prueba), len(rep.unidades_de_desarrollo), len(rep.unidades_de_prueba)) \
        == (2857, 524, 29, 13)
    assert (rep.fraccion_de_sujetos, rep.cuantil, rep.cortes_probados) == (0.3, 0.7, 1)
    assert rep.filas_descartadas == 2494 and rep.porcentaje_descartado == 42.5
    assert max(t[o] for o in rep.desarrollo) < min(t[o] for o in rep.prueba)       # 0 % anterior
    assert len(rep.pliegues) == 5 and all(len(v) >= 150 for _, v in rep.pliegues)


# ---------------------------------------------------------------------------
# (b) La regla: ningún sujeto a los dos lados, la prueba posterior, y lo descartado se DECLARA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calendario", [0, 150, 600])
@pytest.mark.parametrize("tiempo", ["iso", "dmy", "numero"])
def test_ningun_sujeto_a_los_dos_lados_y_la_prueba_es_posterior(calendario, tiempo):
    filas = _cohorte(40, 20, entrada=calendario, tiempo=tiempo, semilla=calendario + 1)
    r = _estricta(filas)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    _invariantes(filas, r)
    assert r.plan.split_type == "groups_and_time" and r.plan.unit_id_field == "sujeto" and r.plan.time_column == "t"
    # el SplitPlan ya comprueba el aislamiento de unidades; aquí además cada fila con rol lleva SU unidad
    por_id = {f["id"]: f for f in filas}
    assert all(r.plan.units[o] == por_id[o]["sujeto"] for o in r.plan.assignments)
    assert SplitPlan.desde_json(r.plan.a_json()).digest() == r.plan.digest()


def test_con_hueco_la_prueba_empieza_a_medio_hueco_del_corte_y_el_desarrollo_acaba_a_medio_hueco():
    filas = _cohorte(30, 30, entrada=0, tiempo="iso", seguimiento=100)
    hueco = 10
    r = _estricta(filas, gap=Horizonte(magnitud=hueco, unidad="days"), automatico=False)
    assert r.es_viable
    por_id, dev, test = _invariantes(filas, r, hueco=hueco)
    corte = r.reparto.corte / 86400.0 - (BASE - date(1970, 1, 1)).days            # el corte, en días desde BASE
    assert all(_dia(por_id[o]["t"]) < corte - hueco / 2 for o in dev)
    assert all(_dia(por_id[o]["t"]) >= corte + hueco / 2 for o in test)
    # y lo que cae en el hueco es lo descartado: ni desarrollo ni prueba
    en_el_hueco = [f["id"] for f in filas if corte - hueco / 2 <= _dia(f["t"]) < corte + hueco / 2]
    assert en_el_hueco and set(en_el_hueco) <= set(r.reparto.descartadas)


def test_el_corte_es_el_cuantil_de_las_filas_como_np_quantile():
    np = pytest.importorskip("numpy")
    filas = _cohorte(37, 11, entrada=200, semilla=5)
    r = _estricta(filas, cuantil_del_corte=0.65, automatico=False)
    esperado = float(np.quantile([_dia(f["t"]) for f in filas], 0.65))
    assert r.reparto.corte / 86400.0 - (BASE - date(1970, 1, 1)).days == pytest.approx(esperado, abs=1e-6)


def test_el_resultado_declara_el_corte_las_filas_descartadas_y_su_porcentaje():
    filas = _cohorte(30, 20, entrada=100)
    r = _estricta(filas)
    rep = r.reparto
    dia_del_corte = BASE + timedelta(days=int(rep.corte // 86400.0 - (BASE - date(1970, 1, 1)).days))
    assert rep.corte_legible == dia_del_corte.isoformat()          # una columna de fechas: el corte se lee como fecha
    declarado = rep.a_json()
    assert declarado["filas_descartadas"] == rep.filas_descartadas > 0
    assert declarado["porcentaje_descartado"] == rep.porcentaje_descartado
    assert declarado["corte"] == rep.corte and declarado["modo"] == "grupos_y_tiempo_estricto"
    [limite] = [x for x in r.limites if x.clave == "estricto_filas_descartadas"]
    assert limite.medida["filas_descartadas"] == rep.filas_descartadas and limite.medida["filas"] == len(filas)
    assert limite.medida["porcentaje_descartado"] == rep.porcentaje_descartado
    # los dos idiomas llevan los números
    for idioma in ("es", "en"):
        assert str(rep.filas_descartadas) in limite.motivo[idioma] and f"{rep.porcentaje_descartado:g}" in limite.motivo[idioma]


# ---------------------------------------------------------------------------
# (c) Los sujetos de prueba se escogen entre los que tienen filas posteriores
# ---------------------------------------------------------------------------

def _con_la_mitad_solo_antigua(n=40, visitas=20):
    """La mitad de los sujetos (los 20 primeros) solo tiene visitas ANTIGUAS (días 0-39); la otra mitad, de 0 a 100."""
    filas = []
    for i in range(n):
        for j in range(visitas):
            dia = (j * 40) // visitas if i < n // 2 else (j * 100) // visitas
            filas.append({"id": f"r{len(filas):05d}", "sujeto": f"{i:03d}", "t": str(dia),
                          "y": "si" if (i + j) % 3 == 0 else "no"})
    return filas


@pytest.mark.parametrize("semilla", range(8))
def test_ningun_sujeto_de_prueba_es_uno_que_solo_tiene_visitas_antiguas(semilla):
    """B0 del C0 escogía los sujetos de prueba al azar entre TODOS y sacaba «solo antiguos» con cero filas posteriores."""
    filas = _con_la_mitad_solo_antigua()
    r = _estricta(filas, seed=semilla)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    por_id, dev, test = _invariantes(filas, r)
    solo_antiguos = {f"{i:03d}" for i in range(20)}
    assert not (set(r.reparto.unidades_de_prueba) & solo_antiguos)
    # y todos los sujetos de prueba aportan filas a la prueba: ninguno «de adorno»
    assert {por_id[o]["sujeto"] for o in test} == set(r.reparto.unidades_de_prueba)
    # la cuota de sujetos se pide sobre TODOS (40 · 0,3 = 12), no sobre los elegibles
    assert len(r.reparto.unidades_de_prueba) == 12


# ---------------------------------------------------------------------------
# (d) D2: los mínimos, el rechazo con números y la escalera
# ---------------------------------------------------------------------------

def test_los_minimos_son_los_del_c0():
    assert (MINIMO_UNIDADES_DE_PRUEBA, MINIMO_FILAS_DE_PRUEBA, MINIMO_UNIDADES_POR_CLASE) == (5, 50, 2)


def _bloqueo(r):
    assert not r.es_viable and r.plan is None and r.pliegues is None and r.reparto is None
    [b] = r.bloqueos
    return b


def _en_los_dos_idiomas(b, *trozos):
    for idioma in ("es", "en"):
        assert b.motivo[idioma]
        for t in trozos:
            assert str(t) in b.motivo[idioma], (idioma, t, b.motivo[idioma])
    assert b.motivo["es"] != b.motivo["en"]


def test_m1_menos_de_5_sujetos_de_prueba_se_rechaza_con_los_numeros():
    filas = _cohorte(14, 40, entrada=0)            # 14 · 0,3 = 4,2 → 4 sujetos de prueba
    b = _bloqueo(_estricta(filas, automatico=False))
    assert b.clave == "estricto_prueba_con_pocas_unidades"
    assert b.medida["unidades_de_prueba"] == 4 and b.medida["minimo"] == 5 and b.medida["sujetos_elegibles"] == 14
    assert b.medida["cortes_probados"] == 1 and b.campo == "test_fraction"
    _en_los_dos_idiomas(b, 4, 5, "sujeto")


def test_m2_menos_de_50_filas_de_prueba_se_rechaza_con_los_numeros():
    filas = _cohorte(20, 4, entrada=0, seguimiento=100)       # 6 sujetos de prueba, 1 fila posterior cada uno
    b = _bloqueo(_estricta(filas, automatico=False))
    assert b.clave == "estricto_prueba_con_pocas_filas"
    assert b.medida["minimo"] == 50 and b.medida["unidades_de_prueba"] >= 5 and b.medida["filas_de_prueba"] < 50
    _en_los_dos_idiomas(b, b.medida["filas_de_prueba"], 50)


def _con_una_clase_en_pocos_sujetos(n_con_la_clase, n=30, visitas=20, en_las_posteriores=True):
    """Los sujetos 0..n_con_la_clase-1 llevan la clase «rara» en sus visitas posteriores (o solo en las antiguas)."""
    filas = _cohorte(n, visitas, entrada=0, barajar=False)
    for f in filas:
        i = int(f["sujeto"])
        tarde = _dia(f["t"]) >= 70
        f["y"] = "rara" if (i < n_con_la_clase and tarde == en_las_posteriores) else ("si" if i % 2 else "no")
    random.Random(2).shuffle(filas)
    return filas


@pytest.mark.parametrize("cuantos", [1, 0])
def test_m3_una_clase_en_menos_de_2_sujetos_elegibles_se_rechaza_y_dice_cuantos_la_traen(cuantos):
    """Un solo sujeto con la clase (1) o ninguno que la tenga DESPUÉS del corte (0: la clase ausente de la prueba)."""
    filas = _con_una_clase_en_pocos_sujetos(cuantos if cuantos else 3, en_las_posteriores=bool(cuantos))
    if cuantos == 1:
        filas = _con_una_clase_en_pocos_sujetos(1)
    b = _bloqueo(_estricta(filas, automatico=False))
    assert b.clave == "estricto_clase_en_pocos_sujetos_de_prueba"
    assert b.medida["clase"] == "rara" and b.medida["minimo"] == 2
    assert b.medida["sujetos_de_prueba_por_clase"]["rara"] == cuantos
    _en_los_dos_idiomas(b, "rara", cuantos, 2)


def test_m3_con_la_clase_en_cuatro_sujetos_hay_dos_sujetos_de_prueba_con_ella_y_dos_en_desarrollo():
    filas = _cohorte(30, 20, entrada=0, barajar=False)
    for f in filas:
        f["y"] = "rara" if int(f["sujeto"]) < 4 else ("si" if int(f["sujeto"]) % 2 else "no")
    random.Random(2).shuffle(filas)
    r = _estricta(filas, automatico=False)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    por_id, dev, test = _invariantes(filas, r)
    assert len({por_id[o]["sujeto"] for o in test if por_id[o]["y"] == "rara"}) == 2
    assert len({por_id[o]["sujeto"] for o in dev if por_id[o]["y"] == "rara"}) == 2


def test_m3_cada_clase_en_2_sujetos_de_prueba_en_cuatro_clases_con_clase_rara_concentrada():
    """Multiclase, con la clase rara en 6 de 40 sujetos: la prueba lleva las 4 clases y cada una en ≥ 2 sujetos."""
    rng = random.Random(9)
    filas = _cohorte(40, 25, entrada=120, barajar=False)
    for f in filas:
        i = int(f["sujeto"])
        f["y"] = "rara" if (i % 7 == 0 and rng.random() < 0.9) else rng.choice(["a", "b", "c"])
    rng.shuffle(filas)
    for semilla in range(12):
        r = _estricta(filas, seed=semilla)
        assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
        por_id, dev, test = _invariantes(filas, r)
        for clase in ("a", "b", "c", "rara"):
            assert len({por_id[o]["sujeto"] for o in test if por_id[o]["y"] == clase}) >= 2, (semilla, clase)
            assert any(por_id[o]["y"] == clase for o in dev)


def test_m3_tambien_cuando_todos_los_elegibles_son_de_prueba_y_la_clase_esta_en_uno():
    """20 sujetos, solo 8 con filas posteriores y la cuota (10) los cubre a todos: no hay reparto que hacer, pero la clase
    «rara» sigue estando en un solo sujeto de prueba."""
    filas = []
    for i in range(20):
        for j in range(30):
            dia = (j * 100) // 30 if i < 8 else j // 3                      # los 8 primeros llegan al día 99; el resto, al 9
            filas.append({"id": f"r{len(filas):05d}", "sujeto": f"{i:03d}", "t": str(dia),
                          "y": "rara" if (i == 0) else ("si" if (i + j) % 2 else "no")})
    random.Random(1).shuffle(filas)
    b = _bloqueo(_estricta(filas, automatico=False, fraccion_de_sujetos=0.5, cuantil_del_corte=0.7))
    assert b.clave == "estricto_clase_en_pocos_sujetos_de_prueba" and b.medida["clase"] == "rara"
    assert b.medida["sujetos_de_prueba_por_clase"]["rara"] == 1 and b.medida["unidades_de_prueba"] == 8


def test_las_cuentas_de_clase_de_la_prueba_son_las_de_las_filas_posteriores_no_las_de_todas():
    """La clase «rara» esta en TODAS las filas de los sujetos 0-14, pero DESPUÉS del corte solo en los sujetos 0, 1 y 2:
    el asignador tiene que contar por mitad de tiempo, o escoge sujetos que la traen solo de antes."""
    filas = _cohorte(30, 20, entrada=0, barajar=False)
    for f in filas:
        i, tarde = int(f["sujeto"]), _dia(f["t"]) >= 70
        f["y"] = "rara" if (i < 3 or (i < 15 and not tarde)) else ("si" if i % 2 else "no")
    random.Random(7).shuffle(filas)
    for semilla in range(10):
        r = _estricta(filas, seed=semilla, automatico=False)
        assert r.es_viable, (semilla, [b.motivo["es"] for b in r.bloqueos])
        por_id, dev, test = _invariantes(filas, r)
        assert len({por_id[o]["sujeto"] for o in test if por_id[o]["y"] == "rara"}) >= 2
        assert any(por_id[o]["y"] == "rara" for o in dev)


def test_prueba_vacia_sin_sujetos_con_filas_posteriores_se_rechaza_con_el_corte_y_el_ultimo_tiempo():
    filas = _cohorte(30, 20, entrada=0, seguimiento=100)
    b = _bloqueo(_estricta(filas, gap=Horizonte(magnitud=400, unidad="days")))      # el hueco se come toda la prueba
    assert b.clave == "estricto_sin_sujetos_con_filas_posteriores" and b.campo == "time_column"
    ultimo = max(f["t"] for f in filas)
    assert b.medida["ultimo_tiempo"] == ultimo and b.medida["sujetos_elegibles"] == 0
    _en_los_dos_idiomas(b, ultimo, "t")


def test_prueba_vacia_sin_cuota_de_sujetos_es_el_bloqueo_de_siempre():
    b = _bloqueo(_estricta(_cohorte(30, 20), fraccion_de_sujetos=0.0))
    assert b.clave == "sin_prueba_reservada"


def test_el_desarrollo_sin_una_clase_se_rechaza():
    """La clase «nueva» solo aparece DESPUÉS del corte, en ≥ 4 sujetos: la prueba la tiene, el desarrollo no."""
    filas = _cohorte(30, 20, entrada=0, barajar=False)
    for f in filas:
        if _dia(f["t"]) >= 80 and int(f["sujeto"]) < 20:
            f["y"] = "nueva"
    random.Random(3).shuffle(filas)
    b = _bloqueo(_estricta(filas, automatico=False))
    assert b.clave == "estricto_desarrollo_sin_clase" and b.medida["clase"] == "nueva"
    assert b.medida["filas_de_desarrollo"] > 0
    _en_los_dos_idiomas(b, "nueva")


def test_el_desarrollo_con_menos_de_2_sujetos_se_rechaza():
    """Casi todos los sujetos solo tienen filas posteriores al corte: no queda con qué entrenar y validar por sujeto."""
    filas = []
    for i in range(12):
        for j in range(40):
            dia = j if i == 0 else 100 + j
            filas.append({"id": f"r{len(filas):05d}", "sujeto": f"{i:03d}", "t": str(dia), "y": "si" if j % 2 else "no"})
    random.Random(1).shuffle(filas)
    b = _bloqueo(_estricta(filas, automatico=False, cuantil_del_corte=0.05, fraccion_de_sujetos=0.5))
    assert b.clave == "estricto_desarrollo_vacio" and b.medida["unidades_de_desarrollo"] < 2


def test_la_escalera_parte_de_lo_pedido_y_relaja_de_lo_menos_a_lo_mas():
    e = escalera_del_corte(0.3, 0.7)
    assert e[0] == (0.3, 0.7) and len(set(e)) == len(e)
    assert all(s >= 0.3 and q <= 0.7 for s, q in e)                  # nunca MENOS sujetos ni un corte más tardío
    assert (0.4, 0.7) in e and (0.3, 0.6) in e and max(s for s, _ in e) <= 0.5 and min(q for _, q in e) >= 0.4
    relajo = [(round((s - 0.3) / 0.1) + round((0.7 - q) / 0.1)) for s, q in e]
    assert relajo == sorted(relajo)                                  # de lo menos a lo más relajado
    assert escalera_del_corte(0.3, 0.7, automatico=False) == [(0.3, 0.7)]


def test_la_escalera_no_busca_el_mayor_desarrollo_se_queda_con_lo_pedido_si_cumple():
    """El C0 vio que «la mayor q que cumple los mínimos» dejaba pruebas de 8 pacientes (s=0,2 q=0,9). Aquí, si lo
    pedido cumple, es lo que se usa aunque un corte más tardío también cumpliera."""
    filas = _cohorte(42, 100, entrada=0)
    r = _estricta(filas)
    rep = r.reparto
    assert (rep.fraccion_de_sujetos, rep.cuantil, rep.cortes_probados) == (0.3, 0.7, 1)
    assert len(rep.unidades_de_prueba) >= 12 and not any(x.clave == "estricto_corte_relajado" for x in r.limites)
    # y el corte tardío (s=0,2, q=0,9) SÍ cumpliría los mínimos pero dejaría una prueba más pequeña
    tardio = _estricta(filas, fraccion_de_sujetos=0.2, cuantil_del_corte=0.9, automatico=False)
    assert tardio.es_viable and len(tardio.reparto.prueba) < len(rep.prueba)


def test_si_lo_pedido_no_cumple_la_escalera_relaja_lo_justo_y_lo_dice():
    filas = _cohorte(14, 40, entrada=0)                      # 4 sujetos con 0,3; 6 con 0,4
    r = _estricta(filas)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    rep = r.reparto
    assert (rep.fraccion_de_sujetos, rep.cuantil) == (0.4, 0.7)       # primero MÁS sujetos; el corte no se mueve
    assert rep.cortes_probados == 2 and (rep.fraccion_de_sujetos_pedida, rep.cuantil_pedido) == (0.3, 0.7)
    [limite] = [x for x in r.limites if x.clave == "estricto_corte_relajado"]
    assert limite.medida["fraccion_usada"] == 0.4 and limite.medida["fraccion_pedida"] == 0.3
    for idioma in ("es", "en"):
        assert "30" in limite.motivo[idioma] and "40" in limite.motivo[idioma]


def test_si_ningun_peldano_cumple_el_rechazo_trae_los_numeros_del_mejor_intento():
    filas = _cohorte(10, 40, entrada=0)                      # ni con el 50 % de 10 sujetos (5)... pero 5 SÍ llega
    r = _estricta(filas)
    assert r.es_viable and len(r.reparto.unidades_de_prueba) >= 5
    filas = _cohorte(8, 40, entrada=0)                       # 8 sujetos: el máximo (50 %) son 4
    b = _bloqueo(_estricta(filas))
    assert b.clave == "estricto_prueba_con_pocas_unidades" and b.medida["unidades_de_prueba"] == 4
    assert b.medida["cortes_probados"] == len(escalera_del_corte(0.3, 0.7))
    assert b.medida["fraccion_de_sujetos"] == 0.5          # el mejor intento: el que más sujetos de prueba da


# ---------------------------------------------------------------------------
# (e) D3: pliegues por sujeto × último 30 % de las filas (F2q), y su caída a F1
# ---------------------------------------------------------------------------

def _comprobar_f2q(filas, r, hueco=0):
    por_id = {f["id"]: f for f in filas}
    rep = r.reparto
    assert rep.tipo_de_pliegues == "por_sujeto_y_ultimo_tramo" and rep.inicio_del_tramo is not None
    dev_t = sorted(_dia(por_id[o]["t"]) for o in rep.desarrollo)
    inicio = rep.inicio_del_tramo / 86400.0 - (BASE - date(1970, 1, 1)).days
    # el tramo empieza en el cuantil 0,7 de las FILAS del desarrollo
    pos = (len(dev_t) - 1) * (1 - FRACCION_DE_LA_VENTANA)
    esperado = dev_t[int(pos)] + (dev_t[min(int(pos) + 1, len(dev_t) - 1)] - dev_t[int(pos)]) * (pos - int(pos))
    assert inicio == pytest.approx(esperado, abs=1e-6)
    unidades_validadas: list[set] = []
    for p in r.pliegues.pliegues:
        val_u = {por_id[o]["sujeto"] for o in p.valida}
        ent_u = {por_id[o]["sujeto"] for o in p.entrena}
        assert not (val_u & ent_u)                                        # ningún sujeto en los dos lados del pliegue
        assert all(_dia(por_id[o]["t"]) >= inicio - 1e-9 for o in p.valida)               # solo el último tramo
        assert all(inicio - _dia(por_id[o]["t"]) > hueco for o in p.entrena)             # antes del tramo y del hueco
        assert set(p.valida) <= set(rep.desarrollo) and set(p.entrena) <= set(rep.desarrollo)
        # la validación lleva TODAS las filas del tramo de sus sujetos
        assert set(p.valida) == {o for o in rep.desarrollo if por_id[o]["sujeto"] in val_u
                                 and _dia(por_id[o]["t"]) >= inicio - 1e-9}
        unidades_validadas.append(val_u)
    for i, a in enumerate(unidades_validadas):                            # un sujeto se valida en UN solo pliegue
        for b in unidades_validadas[i + 1:]:
            assert not (a & b)
    con_validacion = {o for p in r.pliegues.pliegues for o in p.valida}
    assert set(r.cobertura.ids_sin_validacion) == set(rep.desarrollo) - con_validacion
    assert r.cobertura.filas_de_desarrollo == len(rep.desarrollo)
    return por_id, unidades_validadas


@pytest.mark.parametrize("calendario", [0, 200])
def test_los_pliegues_son_por_sujeto_y_validan_solo_el_ultimo_30_por_ciento_de_las_filas(calendario):
    filas = _cohorte(40, 25, entrada=calendario, semilla=11)
    r = _estricta(filas)
    assert r.es_viable and r.reparto.tipo_de_pliegues == "por_sujeto_y_ultimo_tramo"
    assert len(r.pliegues.pliegues) == 5 == r.pliegues.folds and r.pliegues.repeats == 1
    _comprobar_f2q(filas, r)
    # cada pliegue valida y entrena con TODAS las clases (la clase se reparte con el asignador real)
    por_id = {f["id"]: f for f in filas}
    for p in r.pliegues.pliegues:
        assert {por_id[o]["y"] for o in p.valida} == {por_id[o]["y"] for o in p.entrena} == {"si", "no"}


def test_el_hueco_quita_de_entrena_lo_que_esta_a_menos_de_un_hueco_del_tramo():
    """Una visita al día durante 100 días: el inicio del tramo cae en un día entero y HAY filas justo a un hueco de él.
    La frontera es estricta (la de `temporal`): una fila a `hueco` justo mira el primer instante de la validación."""
    filas = _cohorte(30, 100, entrada=0, seguimiento=100, semilla=4)
    hueco = 5
    r = _estricta(filas, gap=Horizonte(magnitud=hueco, unidad="days"), automatico=False)
    assert r.es_viable
    _comprobar_f2q(filas, r, hueco=hueco)
    por_id = {f["id"]: f for f in filas}
    inicio = r.reparto.inicio_del_tramo / 86400.0 - (BASE - date(1970, 1, 1)).days
    assert inicio == int(inicio) and any(inicio - _dia(por_id[o]["t"]) == hueco for o in r.reparto.desarrollo)
    en_entrena = {o for p in r.pliegues.pliegues for o in p.entrena}
    assert not any(inicio - _dia(por_id[o]["t"]) == hueco for o in en_entrena)       # la fila JUSTO a un hueco, fuera
    assert all(q > 0 for q in r.cobertura.quitadas_por_el_hueco) and len(r.cobertura.quitadas_por_el_hueco) == 5


def test_el_tramo_se_mide_en_filas_no_en_tiempo_F2_por_span_dejaba_validaciones_vacias():
    """Calendario largo y escalonado (entrada en 1.000 días, seguimiento de 100): el último 30 % del SPAN de tiempo
    del desarrollo casi no tiene filas (C0: validaciones de 0 filas). El último 30 % de las FILAS, sí."""
    filas = _cohorte(42, 40, entrada=1000, seguimiento=100, semilla=2)
    r = _estricta(filas)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    assert r.reparto.tipo_de_pliegues == "por_sujeto_y_ultimo_tramo"
    assert min(len(p.valida) for p in r.pliegues.pliegues) >= 30
    dev_t = [_dia({f["id"]: f for f in filas}[o]["t"]) for o in r.reparto.desarrollo]
    span_30 = max(dev_t) - 0.3 * (max(dev_t) - min(dev_t))
    # con el criterio de span, las filas por encima de ese tramo serían muy pocas: las del tramo de filas son ≥ 30 %
    assert sum(t >= span_30 for t in dev_t) < 0.3 * len(dev_t)
    assert sum(t >= r.reparto.inicio_del_tramo / 86400.0 - (BASE - date(1970, 1, 1)).days for t in dev_t) \
        >= 0.3 * len(dev_t)


def test_con_menos_de_300_filas_de_desarrollo_se_cae_a_F1_y_se_dice():
    filas = _cohorte(24, 14, entrada=0, semilla=6)
    r = _estricta(filas, cuantil_del_corte=0.8)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    assert len(r.reparto.desarrollo) < FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO
    assert r.reparto.tipo_de_pliegues == "por_sujeto" and r.reparto.inicio_del_tramo is None
    [limite] = [x for x in r.limites if x.clave == "estricto_pliegues_por_sujeto_por_pocas_filas"]
    assert limite.medida["filas_de_desarrollo"] == len(r.reparto.desarrollo)
    for idioma in ("es", "en"):
        assert str(len(r.reparto.desarrollo)) in limite.motivo[idioma] and "300" in limite.motivo[idioma]
    # F1: por sujeto, las `valida` cubren TODO el desarrollo, y un sujeto en un solo pliegue
    por_id = {f["id"]: f for f in filas}
    assert r.cobertura.filas_sin_validacion == 0
    assert sorted(o for p in r.pliegues.pliegues for o in p.valida) == sorted(r.reparto.desarrollo)
    for p in r.pliegues.pliegues:
        assert not ({por_id[o]["sujeto"] for o in p.valida} & {por_id[o]["sujeto"] for o in p.entrena})
        assert {por_id[o]["y"] for o in p.valida} == {"si", "no"}


def test_si_el_ultimo_tramo_no_deja_validaciones_suficientes_se_cae_a_F1_y_se_dice():
    """Un solo sujeto («G») tiene todas las filas del último 30 % del desarrollo: no hay con qué hacer ni 2 pliegues."""
    filas = []

    def anade(sujeto, dias):
        for dia in dias:
            filas.append({"id": f"r{len(filas):05d}", "sujeto": sujeto, "t": str(dia), "y": "si" if len(filas) % 2 else "no"})
    for i in range(20):
        anade(f"{i:03d}", range(10))                          # 200 filas muy antiguas
    anade("G", range(100, 250))                               # el sujeto largo: 150 filas, lo último del desarrollo
    for i in range(10):
        anade(f"L{i:02d}", range(300, 320))                   # 10 sujetos con filas tardías (la prueba)
    random.Random(8).shuffle(filas)
    r = _estricta(filas, automatico=False)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    assert len(r.reparto.desarrollo) >= FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO
    assert r.reparto.tipo_de_pliegues == "por_sujeto"
    assert [x.clave for x in r.limites if x.clave.startswith("estricto_pliegues_por_sujeto")] \
        == ["estricto_pliegues_por_sujeto_por_ventana_vacia"]


def test_si_una_clase_no_esta_en_el_ultimo_tramo_se_cae_a_F1_y_se_dice():
    filas = _cohorte(40, 25, entrada=0, barajar=False)
    for f in filas:
        if 55 <= _dia(f["t"]) < 78:
            f["y"] = "no"             # el último tramo del desarrollo (antes del corte, ≈ día 78) no tiene positivos
    random.Random(2).shuffle(filas)
    r = _estricta(filas, automatico=False, cuantil_del_corte=0.8)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    assert r.reparto.tipo_de_pliegues == "por_sujeto"
    assert any(x.clave == "estricto_pliegues_por_sujeto_por_ventana_vacia" for x in r.limites)


def test_F1_reparte_la_clase_con_el_asignador_de_groups_y_baja_pliegues_si_no_cabe():
    """Los positivos viven en 5 de 24 sujetos (2 acaban en la prueba, 3 en desarrollo): 5 pliegues al azar dejarían
    validaciones sin positivos; con el asignador real, cada `valida` tiene las dos clases y se baja a 3 pliegues."""
    filas = _cohorte(24, 20, entrada=0, barajar=False, semilla=6)
    for f in filas:
        f["y"] = "si" if int(f["sujeto"]) < 5 else "no"
    for semilla in range(8):
        r = _estricta(filas, seed=semilla, cuantil_del_corte=0.6, automatico=False)
        assert r.es_viable, (semilla, [b.motivo["es"] for b in r.bloqueos])
        assert r.reparto.tipo_de_pliegues == "por_sujeto"
        por_id = {f["id"]: f for f in filas}
        for p in r.pliegues.pliegues:
            assert {por_id[o]["y"] for o in p.valida} == {"si", "no"}
        dev_positivos = {por_id[o]["sujeto"] for o in r.reparto.desarrollo if por_id[o]["y"] == "si"}
        assert len(r.pliegues.pliegues) <= max(2, len(dev_positivos))


def test_la_viabilidad_de_los_pliegues_con_tiempo_regla_a_regla():
    """`valida` ≥ 30 filas, `entrena` ≥ 100 y todas las clases en las dos: cada una se comprueba SOLA, con la mano."""
    from matrixai.training.particion_por_diseno_grupos_y_tiempo import _son_viables
    cl = lambda o: "a" if o % 2 else "b"                                           # noqa: E731
    bien = (tuple(range(100)), tuple(range(100, 130)))
    assert _son_viables([bien], cl, ["a", "b"])
    assert not _son_viables([(tuple(range(99)), bien[1])], cl, ["a", "b"])            # entrena de 99
    assert not _son_viables([(bien[0], tuple(range(100, 129)))], cl, ["a", "b"])      # valida de 29
    assert not _son_viables([(tuple(range(0, 200, 2)), bien[1])], cl, ["a", "b"])      # entrena sin la clase «a»
    assert not _son_viables([(bien[0], tuple(range(100, 160, 2)))], cl, ["a", "b"])    # valida sin la clase «a»
    assert _son_viables([(tuple(range(0, 200, 2)), tuple(range(100, 160, 2)))], lambda o: None, [None])   # sin clases
    assert not _son_viables([bien, (tuple(range(99)), bien[1])], cl, ["a", "b"])       # basta UN pliegue malo


def test_si_una_clase_solo_esta_en_el_ultimo_tramo_no_hay_entrenamiento_con_ella_y_se_cae_a_F1():
    """Los positivos solo aparecen en el tramo de validación (y en la prueba): el entrenamiento de F2q no los vería."""
    filas = _cohorte(40, 25, entrada=0, barajar=False)
    for f in filas:
        if _dia(f["t"]) < 55:
            f["y"] = "no"
    random.Random(2).shuffle(filas)
    r = _estricta(filas, automatico=False, cuantil_del_corte=0.8)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    assert r.reparto.tipo_de_pliegues == "por_sujeto"
    assert any(x.clave == "estricto_pliegues_por_sujeto_por_ventana_vacia" for x in r.limites)


def test_las_repeticiones_se_fijan_a_1_y_se_dice():
    r = _estricta(_cohorte(40, 25, entrada=100), repeats=3)
    assert r.es_viable and r.pliegues.repeats == 1 == r.plan.repeats
    [limite] = [x for x in r.limites if x.clave == "repeticiones_fijadas_a_1"]
    assert limite.medida == {"repeticiones_pedidas": 3, "repeticiones_usadas": 1}
    assert not any(x.clave == "repeticiones_fijadas_a_1" for x in _estricta(_cohorte(40, 25, entrada=100)).limites)


def test_los_pliegues_se_reducen_declarandolo_si_el_tramo_no_da_para_cinco():
    """20 sujetos × 40 visitas, corte en el cuantil 0,7: el desarrollo (392 filas) tiene 118 en el tramo, y con 5 pliegues
    algún pliegue valida menos de 30 filas; con 3, todos llegan."""
    filas = _cohorte(20, 40, entrada=0, semilla=3)
    r = _estricta(filas, cuantil_del_corte=0.7, automatico=False, folds=5)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    assert r.reparto.tipo_de_pliegues == "por_sujeto_y_ultimo_tramo" and len(r.pliegues.pliegues) == 3 == r.pliegues.folds
    _comprobar_f2q(filas, r)
    assert min(len(p.valida) for p in r.pliegues.pliegues) >= 30
    [limite] = [x for x in r.limites if x.clave == "estricto_pliegues_reducidos"]
    assert (limite.medida["pliegues_pedidos"], limite.medida["pliegues_usados"]) == (5, 3)
    for idioma in ("es", "en"):
        assert "5" in limite.motivo[idioma] and "3" in limite.motivo[idioma] and "30" in limite.motivo[idioma]
    assert r.plan.folds == 3


def test_regresion_sin_objetivo_tambien_da_pliegues_por_sujeto_y_tramo():
    filas = _cohorte(40, 25, entrada=100)
    r = _estricta(filas, objetivo=None)
    assert r.es_viable and r.reparto.tipo_de_pliegues == "por_sujeto_y_ultimo_tramo"
    _comprobar_f2q(filas, r)
    _invariantes(filas, r)


def test_una_sola_clase_se_declara_y_el_reparto_sigue_sin_guardas_de_clase():
    filas = _cohorte(40, 25, entrada=100)
    for f in filas:
        f["y"] = "si"
    r = _estricta(filas)
    assert r.es_viable and any(x.clave == "una_sola_clase_sin_guardas" for x in r.limites)


# ---------------------------------------------------------------------------
# (f) D4: el intervalo
# ---------------------------------------------------------------------------

def test_el_intervalo_es_el_bootstrap_por_unidad_existente_y_condicional_a_la_ventana():
    from matrixai.estudio import incertidumbre
    from matrixai.estudio.metricas import Muestra
    filas = _cohorte(40, 25, entrada=100)
    r = _estricta(filas)
    iv = r.reparto.intervalo
    assert (iv.metodo, iv.diseno_de_remuestreo, iv.unidad_de_remuestreo) == ("cluster_bootstrap_by_unit", "groups", "unit")
    assert iv.condicional_a_la_ventana is True and iv.unidades_de_prueba == len(r.reparto.unidades_de_prueba) > 0
    por_id = {f["id"]: f for f in filas}
    prueba = r.reparto.prueba
    assert iv.ventana_de_prueba == (r.tiempo_por_id[prueba[0]], r.tiempo_por_id[prueba[-1]])
    # el `intervalo()` REAL, con el diseño declarado, hace lo que se declara
    rng = random.Random(1)
    verdad = [rng.random() for _ in prueba]
    muestra = Muestra(task="regression", y_true=verdad, predictions=[v + rng.gauss(0, 0.1) for v in verdad],
                      units=[por_id[o]["sujeto"] for o in prueba])
    real = incertidumbre.intervalo("rmse", muestra, diseno=iv.diseno_de_remuestreo,
                                   estimando="fixed_model_on_population", semilla=1, remuestras=200)
    assert real.disponible and real.method == iv.metodo and real.resampling_unit == iv.unidad_de_remuestreo
    # y `incertidumbre.py` NO se ha tocado: `groups_and_time` sigue fuera de sus diseños soportados
    assert "groups_and_time" not in incertidumbre.DISENOS_SOPORTADOS
    assert iv.a_json()["unidades_de_prueba"] == iv.unidades_de_prueba


def test_ningun_texto_redacta_una_cobertura():
    """Cómo se enseña la cobertura lo decide Roberto: el núcleo deja el número de sujetos de prueba y NADA más."""
    claves = [k for k in MOTIVOS if k.startswith("estricto_")]
    assert claves
    for k in claves:
        for idioma in ("es", "en"):
            texto = MOTIVOS[k][idioma].lower()
            assert "cobertura" not in texto and "coverage" not in texto and "95" not in texto, (k, idioma)


# ---------------------------------------------------------------------------
# (g) Las guardas heredadas: hueco numérico, ids de texto, fechas DD/MM, una fila, un sujeto enorme, CSV barajado
# ---------------------------------------------------------------------------

def test_hueco_en_tiempo_numerico_tambien_en_este_diseno_y_es_el_mismo_bloqueo_que_en_temporal():
    filas = _cohorte(30, 20, tiempo="numero")
    hueco = Horizonte(magnitud=7, unidad="days")
    estricta = _estricta(filas, gap=hueco)
    temporal = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="temporal", seed=3,
                                  time_column="t", gap=hueco)
    [b], [t] = estricta.bloqueos, temporal.bloqueos
    assert b.clave == "hueco_en_tiempo_numerico" and not estricta.es_viable
    assert (b.clave, b.motivo, b.medida, b.campo) == (t.clave, t.motivo, t.medida, t.campo)
    # un hueco en SEGUNDOS sobre un número sí se aplica, declarando que no se sabe la unidad
    ok = _estricta(filas, gap=Horizonte(magnitud=3, unidad="seconds"))
    assert ok.es_viable and any(x.clave == "tiempo_numerico_sin_unidad" for x in ok.limites)


def test_hueco_ilegible_y_tiempo_ilegible_son_bloqueos_con_motivo_no_excepciones():
    filas = _cohorte(30, 20)
    assert _bloqueo(_estricta(filas, gap={"magnitud": "mucho", "unidad": "days"})).clave == "hueco_ilegible"
    malas = [dict(f) for f in filas]
    malas[0]["t"] = "N/D"
    assert _bloqueo(_estricta(malas)).clave == "tiempo_ilegible"
    assert _bloqueo(_estricta(filas, time_column=None)).clave == "falta_columna_de_tiempo"


def test_los_ids_de_sujeto_de_texto_con_ceros_a_la_izquierda_no_se_juntan():
    """«007» y «7» son DOS sujetos; los ids de fila de texto llegan tal cual a los roles."""
    filas = _cohorte(20, 25, entrada=60, prefijo="P-", ancho=3)
    extra = [{"id": f"x{j}", "sujeto": s, "t": (BASE + timedelta(days=90 + j)).isoformat(), "y": "si"}
             for s in ("007", "7", "07") for j in range(30)]
    todas = filas + extra
    r = _estricta(todas)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    por_id = {f["id"]: f for f in todas}
    lados: dict[str, set] = {}
    for o, rol in r.plan.assignments.items():
        lados.setdefault(por_id[o]["sujeto"], set()).add(rol)
    assert all(len(v) == 1 for v in lados.values())                      # nadie a los dos lados
    assert len({por_id[o]["sujeto"] for o in r.plan.assignments} & {"007", "7", "07"}) >= 1
    assert all(isinstance(o, str) for o in r.plan.assignments) and all(isinstance(u, str) for u in r.plan.units.values())


def test_fechas_DD_MM_AAAA_dan_el_mismo_reparto_que_las_ISO_y_las_ambiguas_se_rechazan():
    iso = _cohorte(30, 20, entrada=120, tiempo="iso", barajar=False)
    dmy = _cohorte(30, 20, entrada=120, tiempo="dmy", barajar=False)
    a, b = _estricta(iso), _estricta(dmy)
    assert a.es_viable and b.es_viable
    assert dict(a.plan.assignments) == dict(b.plan.assignments)
    assert [p.valida for p in a.pliegues.pliegues] == [p.valida for p in b.pliegues.pliegues]
    assert a.reparto.corte == b.reparto.corte and a.reparto.corte_legible == b.reparto.corte_legible
    assert len(b.reparto.corte_legible) == 10 and b.reparto.corte_legible[4] == "-"
    ambiguas = [dict(f, t=f"{1 + i % 12:02d}/{1 + (i // 12) % 12:02d}/2024") for i, f in enumerate(dmy)]
    assert _bloqueo(_estricta(ambiguas)).clave == "fecha_ambigua"


def test_sujetos_de_una_sola_fila_juegan_en_el_lado_de_esa_fila():
    base = _cohorte(30, 20, entrada=100)
    sueltos = [{"id": f"s{i}", "sujeto": f"solo{i:02d}", "t": (BASE + timedelta(days=(i * 7) % 130)).isoformat(),
                "y": "si" if i % 2 else "no"} for i in range(40)]
    filas = base + sueltos
    for semilla in range(4):
        r = _estricta(filas, seed=semilla)
        assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
        por_id, dev, test = _invariantes(filas, r)
        assert len(r.reparto.descartadas) > 0
        for o in test:                                       # un sujeto de una fila en la prueba aporta SU fila posterior
            if por_id[o]["sujeto"].startswith("solo"):
                assert _dia(por_id[o]["t"]) * 86400.0 + (BASE - date(1970, 1, 1)).days * 86400.0 >= r.reparto.corte
    todos_de_una = [{"id": f"u{i}", "sujeto": f"s{i:03d}", "t": (BASE + timedelta(days=i % 90)).isoformat(),
                     "y": "si" if i % 3 else "no"} for i in range(400)]
    r = _estricta(todos_de_una)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    _invariantes(todos_de_una, r)


def test_un_paciente_enorme_no_rompe_los_invariantes_y_la_prueba_no_depende_de_el():
    pequenos = _cohorte(30, 20, entrada=0, barajar=False)
    enorme = [{"id": f"g{j}", "sujeto": "ENORME", "t": (BASE + timedelta(days=(j * 100) // 500)).isoformat(),
               "y": "si" if j % 2 else "no"} for j in range(500)]                                           # 500 de 1.100 filas: el 45 %
    filas = pequenos + enorme
    random.Random(4).shuffle(filas)
    for semilla in range(10):
        r = _estricta(filas, seed=semilla)
        assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
        _invariantes(filas, r)
        en_prueba = "ENORME" in r.reparto.unidades_de_prueba
        if en_prueba:
            filas_enorme = sum(1 for o in r.reparto.prueba if o.startswith("g"))
            domina = filas_enorme / len(r.reparto.prueba) > 0.5
            assert domina == any(x.clave == "estricto_unidad_dominante_en_la_prueba" for x in r.limites)
        else:
            assert not any(x.clave == "estricto_unidad_dominante_en_la_prueba" for x in r.limites)
        assert len(r.reparto.unidades_de_prueba) >= 5


def test_un_sujeto_que_domina_la_prueba_se_declara_con_su_porcentaje():
    """Solo 6 sujetos tienen filas posteriores y uno de ellos tiene 300 de las 350: la prueba es casi él."""
    filas = []
    for i in range(30):
        late = i < 6
        m = 300 if i == 0 else 10
        for j in range(m + (20 if late else 0)):
            dia = j if not late or j < 20 else 60 + (j - 20) * 40 // m
            filas.append({"id": f"r{len(filas):05d}", "sujeto": f"{i:03d}", "t": str(dia if dia < 55 or late else 54),
                          "y": "si" if (i + j) % 2 else "no"})
    random.Random(5).shuffle(filas)
    r = _estricta(filas, fraccion_de_sujetos=0.2, cuantil_del_corte=0.5, automatico=False)
    if r.es_viable and "000" in r.reparto.unidades_de_prueba:
        [limite] = [x for x in r.limites if x.clave == "estricto_unidad_dominante_en_la_prueba"]
        assert limite.medida["unidad"] == "000" and limite.medida["filas_de_la_unidad"] > 0.5 * limite.medida["filas_de_prueba"]
        for idioma in ("es", "en"):
            assert "000" in limite.motivo[idioma] and "sujeto" in limite.motivo["es"]
    else:
        pytest.skip("el reparto de esta semilla dejó al sujeto grande fuera de la prueba")


@pytest.mark.parametrize("calendario", [0, 150])
def test_el_csv_barajado_da_el_mismo_reparto(calendario):
    filas = _cohorte(40, 20, entrada=calendario, barajar=False)
    a = _estricta(filas)
    for semilla in (1, 2, 3):
        barajadas = list(filas)
        random.Random(semilla).shuffle(barajadas)
        b = _estricta(barajadas)
        assert a.es_viable and b.es_viable
        assert a.plan.digest() == b.plan.digest()
        assert a.pliegues.digest() == b.pliegues.digest()
        assert (a.reparto.unidades_de_prueba, a.reparto.corte, a.reparto.cortes_probados) \
            == (b.reparto.unidades_de_prueba, b.reparto.corte, b.reparto.cortes_probados)
        assert [x.a_json() for x in a.limites] == [x.a_json() for x in b.limites]


def test_la_semilla_decide_que_sujetos_se_apartan_y_la_misma_semilla_repite():
    filas = _cohorte(40, 20, entrada=100)
    pruebas = {tuple(_estricta(filas, seed=s).reparto.unidades_de_prueba) for s in range(6)}
    assert len(pruebas) > 1
    assert _estricta(filas, seed=4).plan.digest() == _estricta(filas, seed=4).plan.digest()


def test_en_un_panel_la_publicada_no_es_viable_y_la_estricta_si():
    """Todos los sujetos acaban el mismo día: la publicada ordena por la última observación y no puede cortar."""
    filas = _cohorte(40, 30, entrada=0)
    publicada = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="groups_and_time", seed=3,
                                   unit_id_field="sujeto", time_column="t", test_fraction=0.2)
    assert not publicada.es_viable and publicada.bloqueos[0].clave == "cronologia_no_se_pudo_separar"
    estricta = _estricta(filas)
    assert estricta.es_viable
    _invariantes(filas, estricta)


def test_la_publicada_con_calendario_escalonado_pone_prueba_anterior_al_desarrollo_la_estricta_no():
    filas = _cohorte(42, 40, entrada=60, seguimiento=170, semilla=3)
    publicada = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="groups_and_time", seed=3,
                                   unit_id_field="sujeto", time_column="t", test_fraction=0.2)
    por_id = {f["id"]: f for f in filas}
    dev = [o for o, x in publicada.plan.assignments.items() if x == "development"]
    test = [o for o, x in publicada.plan.assignments.items() if x == "test"]
    fin = max(_dia(por_id[o]["t"]) for o in dev)
    assert sum(_dia(por_id[o]["t"]) < fin for o in test) > 0.5 * len(test)
    estricta = _estricta(filas)
    _, dev2, test2 = _invariantes(filas, estricta)
    assert sum(_dia(por_id[o]["t"]) < max(_dia(por_id[d]["t"]) for d in dev2) for o in test2) == 0


# ---------------------------------------------------------------------------
# (h) La cohorte externa y las guardas de unidad (como en la publicada)
# ---------------------------------------------------------------------------

def test_la_cohorte_externa_es_external_test_con_su_unidad_y_no_entra_en_ningun_pliegue():
    filas = _cohorte(36, 20, entrada=80, semilla=2)
    externos = [f["id"] for f in filas if int(f["sujeto"]) >= 30]
    r = _estricta(filas, ids_cohorte_externa=externos)
    assert r.es_viable
    assert {o for o, x in r.plan.assignments.items() if x == "external_test"} == set(externos)
    assert all(o in r.plan.units for o in externos)
    en_pliegues = {o for p in r.pliegues.pliegues for o in (*p.entrena, *p.valida)}
    assert not (set(externos) & en_pliegues) and not (set(externos) & set(r.reparto.desarrollo + r.reparto.prueba))
    comparte = [next(f["id"] for f in filas if int(f["sujeto"]) == 3)]       # UNA fila del sujeto 3; el resto, dentro
    assert _bloqueo(_estricta(filas, ids_cohorte_externa=comparte)).clave == "cohorte_externa_comparte_unidad"


def test_unidad_vacia_y_pocos_grupos_son_bloqueos_con_numeros():
    filas = _cohorte(30, 20)
    vacias = [dict(f) for f in filas]
    vacias[0]["sujeto"] = "N/A"
    b = _bloqueo(_estricta(vacias))
    assert b.clave == "unidad_vacia" and b.medida["filas_sin_unidad"] == 1
    uno = [dict(f, sujeto="unico") for f in filas]
    b = _bloqueo(_estricta(uno))
    assert b.clave == "grupos_insuficientes" and b.medida["grupos"] == 1
    assert _bloqueo(_estricta(filas, unit_id_field=None)).clave == "grupos_insuficientes"


# ---------------------------------------------------------------------------
# (i) Los textos
# ---------------------------------------------------------------------------

def test_los_textos_nuevos_estan_en_los_dos_idiomas_con_los_mismos_huecos():
    claves = [k for k in MOTIVOS if k.startswith("estricto_")]
    assert len(claves) == 16   # 12 del C1 + 4 de la reparación de la auditoría (I3, M4 y M5)
    for k in claves:
        assert set(MOTIVOS[k]) == {"es", "en"}
        assert huecos_de(MOTIVOS[k]["es"]) == huecos_de(MOTIVOS[k]["en"]), k
        assert MOTIVOS[k]["es"] != MOTIVOS[k]["en"]


def test_cada_bloqueo_y_limite_estricto_que_el_reparto_emite_usa_una_clave_del_catalogo():
    emitidas: set[str] = set()
    casos = [_estricta(_cohorte(14, 40), automatico=False), _estricta(_cohorte(20, 4), automatico=False),
             _estricta(_cohorte(14, 40)), _estricta(_cohorte(24, 14), cuantil_del_corte=0.8),
             _estricta(_cohorte(30, 20), gap=Horizonte(magnitud=400, unidad="days"))]
    for r in casos:
        emitidas |= {b.clave for b in r.bloqueos} | {x.clave for x in r.limites}
    for clave in emitidas:
        if clave.startswith("estricto_"):
            assert clave in MOTIVOS
