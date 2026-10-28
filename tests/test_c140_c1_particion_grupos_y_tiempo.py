# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140-C1 — grupos estratificados y pliegues temporales (núcleo).

Cada prueba afirma DÓNDE caen las filas (qué ids en cada lado), no solo que
«no falla». Los datos tienen la forma que trae la gente: ids de sujeto de texto
con ceros a la izquierda, fechas `DD/MM/AAAA` ambiguas y no ambiguas, fechas
repetidas, sujetos de una sola fila, un sujeto con el 40 % de las filas y la
clase positiva en pocos sujetos.

Oráculos: el fichero de HEAD (139) congelado en `tests/data` para `iid` (bit a
bit) y para `groups_and_time`; `sklearn` (`TimeSeriesSplit`,
`StratifiedGroupKFold`) con `importorskip`.
"""
from __future__ import annotations

import random
import sys
import types
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from matrixai.estudio import Horizonte
from matrixai.training.mitades_de_validacion import partir_valida_en_mitades
from matrixai.training.particion_por_diseno import proponer_particion
from matrixai.training.particion_por_diseno_textos import MOTIVOS, huecos_de, motivo

_ORACULO = Path(__file__).parent / "data" / "c140_particion_por_diseno_139.txt"


def _oraculo_139() -> types.ModuleType:
    """El `particion_por_diseno.py` de la rama 139, tal cual, cargado con otro nombre."""
    modulo = types.ModuleType("particion_por_diseno_139")
    modulo.__dict__["__file__"] = str(_ORACULO)
    sys.modules["particion_por_diseno_139"] = modulo       # `dataclasses` busca el módulo de cada clase
    exec(compile(_ORACULO.read_text(encoding="utf-8"), str(_ORACULO), "exec"), modulo.__dict__)
    return modulo


ANTES = _oraculo_139()


def _pide(**kw):
    base = dict(plan_id="p", observation_id_field="id")
    base.update(kw)
    return base


def _rol(r, obs):
    return r.plan.assignments.get(obs)


def _ids_de(r, rol):
    return sorted(o for o, x in r.plan.assignments.items() if x == rol)


def _clases(filas, ids, objetivo="y"):
    por_id = {f["id"]: f for f in filas}
    return Counter(por_id[o][objetivo] for o in ids)


# ---------------------------------------------------------------------------
# (f) iid y groups_and_time, bit a bit contra el código de la rama 139
# ---------------------------------------------------------------------------

def _filas_iid(n, semilla, clases):
    rng = random.Random(semilla)
    return [{"id": f"o{i}", "y": rng.choice(clases), "x": rng.random()} for i in range(n)]


def _firma(r):
    """Todo lo que la propuesta de HEAD ya decía (se ignora lo nuevo: `cobertura`)."""
    return (r.plan.digest() if r.plan else None, dict(r.plan.assignments) if r.plan else None,
            r.pliegues.a_json() if r.pliegues else None,
            [x.a_json() for x in r.limites], [b.a_json() for b in r.bloqueos], r.es_viable)


def _fraccion_de_prueba(r) -> float:
    roles = list(r.plan.assignments.values())
    internos = [x for x in roles if x != "external_test"]
    return sum(x == "test" for x in internos) / len(internos)


def _desborda_la_prueba(r, test_fraction: float) -> bool:
    """Supervisor, 27-10: la 139 dejaba que un sujeto enorme se llevara la prueba (74 % en vez de 20 % en regresión).
    Eso ya no se conserva bit a bit: es el margen de `MARGEN_RELATIVO_DE_LA_PRUEBA`."""
    from matrixai.training.particion_por_diseno_grupos import MARGEN_RELATIVO_DE_LA_PRUEBA
    return abs(_fraccion_de_prueba(r) - test_fraction) > MARGEN_RELATIVO_DE_LA_PRUEBA * test_fraction


@pytest.mark.parametrize("semilla", [0, 1, 7, 42])
@pytest.mark.parametrize("n", [30, 200, 1000])
@pytest.mark.parametrize("clases", [["si"] * 9 + ["no"], ["a", "b", "c"], [None]])   # desbalanceada, multiclase, regresión
@pytest.mark.parametrize("folds,repeats,fraccion", [(5, 1, 0.2), (2, 2, 0.3), (10, 3, 0.1)])
def test_iid_es_bit_a_bit_igual_que_en_la_rama_139(semilla, n, clases, folds, repeats, fraccion):
    filas = _filas_iid(n, semilla, clases)
    objetivo = None if clases == [None] else "y"
    externos = [f"o{i}" for i in range(0, n, 25)]
    kw = _pide(split_type="iid", seed=semilla, test_fraction=fraccion, folds=folds, repeats=repeats,
               objetivo=objetivo, ids_cohorte_externa=externos)
    ahora = proponer_particion(filas, **kw)
    antes = ANTES.proponer_particion(filas, **kw)
    assert ahora.es_viable and antes.es_viable
    assert _firma(ahora) == _firma(antes)
    assert ahora.pliegues.digest() == antes.pliegues.digest()
    # lo nuevo, en iid: las `valida` cubren todo el desarrollo
    assert ahora.cobertura.filas_sin_validacion == 0
    assert ahora.cobertura.filas_de_desarrollo == len(_ids_de(ahora, "development"))


def test_iid_con_pocos_eventos_reduce_pliegues_igual_que_en_la_139():
    filas = [{"id": f"o{i}", "y": "si" if i < 3 else "no"} for i in range(100)]
    kw = _pide(split_type="iid", seed=1, objetivo="y", folds=5, test_fraction=0.1)
    ahora, antes = proponer_particion(filas, **kw), ANTES.proponer_particion(filas, **kw)
    assert _firma(ahora) == _firma(antes)
    assert [x.clave for x in ahora.limites] == ["pliegues_reducidos_por_eventos"]


def _filas_con_tiempo(valor_de):
    return [{"id": f"o{i}", "unidad": f"u{i % 17}", "t": valor_de(i), "y": "a" if i % 3 else "b"} for i in range(200)]


@pytest.mark.parametrize("nombre,valor_de", [
    ("numero", lambda i: float((i * 37) % 200)),
    ("datetime", lambda i: datetime(2020, 1, 1) + timedelta(days=(i * 37) % 200)),
    ("date", lambda i: date(2020, 1, 1) + timedelta(days=(i * 37) % 200)),
])
@pytest.mark.parametrize("hueco", [None, Horizonte(magnitud=5, unidad="days"), Horizonte(magnitud=3600, unidad="seconds")])
def test_groups_and_time_es_igual_que_en_la_rama_139(nombre, valor_de, hueco):
    filas = _filas_con_tiempo(valor_de)
    kw = _pide(split_type="groups_and_time", seed=3, unit_id_field="unidad", time_column="t", gap=hueco)
    ahora, antes = proponer_particion(filas, **kw), ANTES.proponer_particion(filas, **kw)
    assert _firma(ahora) == _firma(antes)
    assert ahora.pliegues is None            # sigue sin pliegues: 140b


@pytest.mark.parametrize("hueco", [None, Horizonte(magnitud=4, unidad="days")])
def test_temporal_conserva_el_corte_desarrollo_prueba_de_la_139(hueco):
    """Lo nuevo de `temporal` son los pliegues DENTRO de desarrollo: quién es desarrollo y quién es prueba no cambia."""
    filas = _filas_con_tiempo(lambda i: datetime(2020, 1, 1) + timedelta(days=(i * 37) % 200))
    kw = _pide(split_type="temporal", seed=3, time_column="t", gap=hueco)
    ahora, antes = proponer_particion(filas, **kw), ANTES.proponer_particion(filas, **kw)
    assert dict(ahora.plan.assignments) == dict(antes.plan.assignments)
    assert antes.pliegues is None and ahora.pliegues is not None


def test_groups_sin_objetivo_es_igual_que_en_la_139():
    """Regresión (sin objetivo): el reparto por grupos barajados no cambia."""
    rng = random.Random(4)
    filas = [{"id": f"o{i}", "s": f"s{rng.randint(0, 40)}"} for i in range(300)]
    kw = _pide(split_type="groups", seed=5, unit_id_field="s", folds=5, repeats=2)
    ahora, antes = proponer_particion(filas, **kw), ANTES.proponer_particion(filas, **kw)
    assert _firma(ahora) == _firma(antes)


# ---------------------------------------------------------------------------
# (a) temporal con pliegues: ventana que se expande, con hueco
# ---------------------------------------------------------------------------

BASE = date(2024, 1, 1)


def _dias(n, *, formato="%Y-%m-%d", por_dia=1, etiqueta=None, barajar=True, semilla=1):
    """`n` días consecutivos; el id lleva el número de orden cronológico (`r000`…)."""
    filas = []
    for i in range(n * por_dia):
        dia = i // por_dia
        filas.append({"id": f"r{i:03d}", "fecha": (BASE + timedelta(days=dia)).strftime(formato),
                      "y": etiqueta(i) if etiqueta else ("pos" if i % 4 == 0 else "neg")})
    if barajar:
        random.Random(semilla).shuffle(filas)
    return filas


def _temporal(filas, **kw):
    base = _pide(split_type="temporal", seed=3, time_column="fecha", objetivo="y")
    base.update(kw)
    return proponer_particion(filas, **base)


def _id(i):
    return f"r{i:03d}"


def test_la_ventana_se_expande_y_el_pliegue_0_es_el_mas_reciente():
    """60 días, 1 fila por día, prueba 20 % (días 48-59), desarrollo 0-47 = 6 bloques de 8."""
    r = _temporal(_dias(60), folds=5)
    assert r.es_viable and r.limites == ()
    assert _ids_de(r, "development") == [_id(i) for i in range(48)]
    assert _ids_de(r, "test") == [_id(i) for i in range(48, 60)]
    p = {x.pliegue: x for x in r.pliegues.pliegues}
    assert sorted(p) == [0, 1, 2, 3, 4] and r.pliegues.folds == 5 and r.pliegues.repeats == 1
    # pliegue 0: el MÁS reciente, el de más entrenamiento
    assert p[0].valida == tuple(_id(i) for i in range(40, 48))
    assert p[0].entrena == tuple(_id(i) for i in range(0, 40))
    assert p[1].valida == tuple(_id(i) for i in range(32, 40)) and p[1].entrena == tuple(_id(i) for i in range(0, 32))
    assert p[4].valida == tuple(_id(i) for i in range(8, 16)) and p[4].entrena == tuple(_id(i) for i in range(0, 8))
    assert [len(p[k].entrena) for k in range(5)] == [40, 32, 24, 16, 8]


def test_entrena_y_valida_salen_ordenados_por_tiempo_aunque_el_csv_venga_desordenado():
    filas = _dias(60, semilla=9)
    assert [f["id"] for f in filas] != sorted(f["id"] for f in filas)            # el CSV llega desordenado
    r = _temporal(filas, folds=5)
    for pl in r.pliegues.pliegues:
        assert list(pl.entrena) == sorted(pl.entrena)       # el id lleva el orden cronológico
        assert list(pl.valida) == sorted(pl.valida)
        assert pl.entrena[-1] < pl.valida[0]                # todo el entrenamiento es anterior a lo que se valida


def test_sklearn_TimeSeriesSplit_como_oraculo_sin_hueco():
    from sklearn.model_selection import TimeSeriesSplit    # noqa: PLC0415
    np = pytest.importorskip("numpy")
    r = _temporal(_dias(60), folds=5)
    desarrollo = _ids_de(r, "development")
    oraculo = list(TimeSeriesSplit(n_splits=5).split(np.arange(len(desarrollo))))
    # el oráculo va del más antiguo al más reciente; el nuestro, al revés
    for k, (entrena, valida) in enumerate(oraculo):
        ours = r.pliegues.pliegue_de(0, 4 - k)
        assert ours.entrena == tuple(desarrollo[i] for i in entrena)
        assert ours.valida == tuple(desarrollo[i] for i in valida)


def test_sklearn_TimeSeriesSplit_como_oraculo_con_hueco():
    """`gap=3` de sklearn quita las 3 filas anteriores; nuestro hueco quita las que están a 3 días O MENOS (frontera
    estricta, auditoría M2): con una fila por día, las mismas tres."""
    from sklearn.model_selection import TimeSeriesSplit    # noqa: PLC0415
    np = pytest.importorskip("numpy")
    r = _temporal(_dias(61), folds=5, gap={"magnitud": 3, "unidad": "days"})
    desarrollo = _ids_de(r, "development")
    # con hueco el corte desarrollo/prueba también deja 3 días: se compara sobre el desarrollo que salió
    # (61 días → 48 filas de desarrollo, divisible por 6 como pide la partición de sklearn)
    assert len(desarrollo) == 48
    oraculo = list(TimeSeriesSplit(n_splits=5, gap=3).split(np.arange(len(desarrollo))))
    for k, (entrena, valida) in enumerate(oraculo):
        ours = r.pliegues.pliegue_de(0, 4 - k)
        assert set(ours.valida) == {desarrollo[i] for i in valida}
        assert set(ours.entrena) == {desarrollo[i] for i in entrena}


def test_filas_con_el_mismo_instante_nunca_quedan_a_los_dos_lados_de_un_corte():
    """3 filas por día: los 8 del reparto en partes iguales no caen en frontera de día; los cortes tienen que moverse."""
    filas = _dias(30, por_dia=3)
    fecha = {f["id"]: f["fecha"] for f in filas}
    r = _temporal(filas, folds=5)
    assert r.es_viable
    fechas_dev, fechas_prueba = ({fecha[o] for o in _ids_de(r, rol)} for rol in ("development", "test"))
    assert fechas_dev.isdisjoint(fechas_prueba)
    for pl in r.pliegues.pliegues:
        assert {fecha[o] for o in pl.entrena}.isdisjoint({fecha[o] for o in pl.valida})
    # y cada día de desarrollo está ENTERO en un solo bloque de validación (o en ninguno: el primero)
    for dia in fechas_dev:
        en = {pl.pliegue for pl in r.pliegues.pliegues for o in pl.valida if fecha[o] == dia}
        assert len(en) <= 1
        sus_filas = [o for o in fecha if fecha[o] == dia]
        if en:
            (unico,) = en
            assert all(o in r.pliegues.pliegue_de(0, unico).valida for o in sus_filas)
    # los tamaños siguen siendo parecidos (no se ha degenerado a un bloque enorme)
    tamanos = [len(pl.valida) for pl in r.pliegues.pliegues]
    assert max(tamanos) - min(tamanos) <= 3


def test_el_hueco_quita_de_entrena_las_filas_a_gap_o_menos_del_inicio_de_valida():
    filas = _dias(100)
    r = _temporal(filas, folds=5, gap={"magnitud": 5, "unidad": "days"})
    assert r.es_viable
    dia = {f["id"]: BASE + timedelta(days=int(f["id"][1:])) for f in filas}      # el id r0NN = día NN
    for pl in r.pliegues.pliegues:
        inicio = min(dia[o] for o in pl.valida)
        # lo que entrena está a MÁS de 5 días del inicio de la validación (frontera estricta, auditoría M2)…
        assert max(dia[o] for o in pl.entrena) < inicio - timedelta(days=5)
        # …y las 5 filas de los 5 días justo anteriores (distancia 1-5; la de 5 justos también) se quitaron
        quitadas = [o for o in dia if inicio - timedelta(days=5) <= dia[o] < inicio and o in set(r.plan.assignments)
                    and r.plan.assignments[o] == "development"]
        assert len(quitadas) == 5 and not (set(quitadas) & set(pl.entrena))
    assert r.cobertura.quitadas_por_el_hueco == (5, 5, 5, 5, 5)
    # entre desarrollo y prueba, también
    ultimo_dev = max(dia[o] for o in _ids_de(r, "development"))
    primero_prueba = min(dia[o] for o in _ids_de(r, "test"))
    assert (primero_prueba - ultimo_dev).days >= 5


def test_sin_hueco_no_se_quita_nada():
    r = _temporal(_dias(60), folds=5)
    assert r.cobertura.quitadas_por_el_hueco == (0, 0, 0, 0, 0)


def test_repeats_se_fija_a_1_y_se_declara():
    r = _temporal(_dias(60), folds=5, repeats=3)
    assert r.pliegues.repeats == 1 and r.plan.repeats == 1
    assert {pl.repeticion for pl in r.pliegues.pliegues} == {0}
    limite = next(x for x in r.limites if x.clave == "repeticiones_fijadas_a_1")
    assert limite.medida == {"repeticiones_pedidas": 3, "repeticiones_usadas": 1}
    assert "3" in limite.motivo["es"] and "3" in limite.motivo["en"]
    # y con repeats=1 no hay nada que declarar
    assert not [x for x in _temporal(_dias(60), folds=5, repeats=1).limites if x.clave == "repeticiones_fijadas_a_1"]


def test_la_propuesta_dice_cuantas_filas_de_desarrollo_no_tienen_validacion_fuera_de_pliegue():
    r = _temporal(_dias(60), folds=5)
    c = r.cobertura
    assert c.filas_de_desarrollo == 48 and c.filas_sin_validacion == 8 and c.filas_con_validacion == 40
    assert c.ids_sin_validacion == tuple(_id(i) for i in range(8))          # el primer bloque, y solo él
    con_valida = {o for pl in r.pliegues.pliegues for o in pl.valida}
    assert set(c.ids_sin_validacion) == set(_ids_de(r, "development")) - con_valida
    assert c.a_json()["filas_sin_validacion"] == 8


def test_el_plan_temporal_lleva_los_pliegues_usados_y_la_semilla():
    """`estudio_job` lee `int(plan.seed)`: no puede ser `None` en temporal."""
    r = _temporal(_dias(60), folds=4, seed=11)
    assert (r.plan.folds, r.plan.repeats, r.plan.seed) == (4, 1, 11)
    assert r.pliegues.folds == 4 and r.pliegues.seed == 11


def test_misma_entrada_misma_ventana_y_la_semilla_no_decide():
    a, b = _temporal(_dias(60), seed=1), _temporal(_dias(60), seed=99)
    assert [(p.entrena, p.valida) for p in a.pliegues.pliegues] == [(p.entrena, p.valida) for p in b.pliegues.pliegues]


# --- (d) el tiempo como lo trae la gente -----------------------------------

def test_fechas_DD_MM_AAAA_no_ambiguas_dan_el_mismo_reparto_que_en_ISO():
    """Hay días > 12, así que la lectura es única."""
    iso = _temporal(_dias(60, formato="%Y-%m-%d"), folds=5)
    espanol = _temporal(_dias(60, formato="%d/%m/%Y"), folds=5)
    assert espanol.es_viable
    assert {o: x for o, x in espanol.plan.assignments.items()} == {o: x for o, x in iso.plan.assignments.items()}
    assert [(p.entrena, p.valida) for p in espanol.pliegues.pliegues] == [(p.entrena, p.valida) for p in iso.pliegues.pliegues]
    # y el orden es el del calendario, no el del texto ('31/01/2024' > '01/02/2024' como texto)
    assert _ids_de(espanol, "test") == [_id(i) for i in range(48, 60)]


def test_fechas_mm_dd_aaaa_con_un_dia_mayor_de_12_se_leen_como_mes_dia():
    filas = _dias(60, formato="%m/%d/%Y")       # 01/13/2024… hay días > 12 en la 2.ª posición
    r = _temporal(filas, folds=5)
    assert r.es_viable
    assert _ids_de(r, "test") == [_id(i) for i in range(48, 60)]


def test_fechas_ambiguas_se_rechazan_con_motivo_y_no_se_eligen_por_orden():
    filas = [{"id": f"r{i:03d}", "fecha": f"{(i % 12) + 1:02d}/{(i % 12) + 1:02d}/2024", "y": "a" if i % 2 else "b"}
             for i in range(60)]                  # 01/01, 02/02… se leen igual de día/mes que de mes/día
    r = _temporal(filas, folds=5)
    assert not r.es_viable and r.plan is None and r.pliegues is None
    (b,) = r.bloqueos
    assert b.clave == "fecha_ambigua" and b.campo == "time_column"
    assert "fecha" in b.motivo["es"] and "'01/01/2024'" in b.motivo["es"] and "day/month/year" in b.motivo["en"]
    assert b.medida["formatos_posibles"] == ["%d/%m/%Y", "%m/%d/%Y"]
    assert "01/01/2024" in b.medida["ejemplos"]


def test_el_mismo_error_en_groups_and_time_es_un_bloqueo_no_una_excepcion():
    filas = [{"id": f"r{i}", "u": f"u{i % 5}", "fecha": "05/06/2024"} for i in range(20)]
    r = proponer_particion(filas, **_pide(split_type="groups_and_time", seed=1, unit_id_field="u", time_column="fecha"))
    assert [b.clave for b in r.bloqueos] == ["fecha_ambigua"]


def test_un_valor_que_no_es_fecha_ni_numero_se_rechaza_con_cuantos_y_ejemplos():
    filas = _dias(60)
    filas[3]["fecha"] = "mañana"
    filas[9]["fecha"] = "N/D"
    r = _temporal(filas, folds=5)
    (b,) = r.bloqueos
    assert b.clave == "tiempo_ilegible" and b.medida["valores_ilegibles"] == 2 and b.medida["filas"] == 60
    assert "'mañana'" in b.motivo["es"] and "'N/D'" in b.motivo["en"]


def test_un_valor_vacio_en_la_columna_de_tiempo_se_rechaza():
    filas = _dias(60)
    filas[5]["fecha"] = ""
    filas[6]["fecha"] = None
    (b,) = _temporal(filas, folds=5).bloqueos
    assert b.clave == "tiempo_ilegible" and b.medida["valores_ilegibles"] == 2 and "(vacío)" in b.medida["ejemplos"]


def test_el_tiempo_numerico_escrito_como_texto_se_lee_y_el_hueco_se_declara_en_segundos():
    filas = [{"id": _id(i), "fecha": str(2000 + i), "y": "pos" if i % 4 == 0 else "neg"} for i in range(60)]
    r = _temporal(filas, folds=5, gap={"magnitud": 1, "unidad": "seconds"})
    assert r.es_viable
    # el hueco de 1 s alrededor del corte (2048) se lleva la fila del propio corte: ni desarrollo ni prueba
    assert _ids_de(r, "development") == [_id(i) for i in range(0, 48)]
    assert _ids_de(r, "test") == [_id(i) for i in range(49, 60)]
    assert [x.clave for x in r.limites] == ["tiempo_numerico_sin_unidad"]
    # sin hueco no hay nada que declarar
    assert _temporal(filas, folds=5).limites == ()


def test_datetime_y_date_siguen_funcionando_como_antes():
    for conv in (lambda d: datetime(d.year, d.month, d.day), lambda d: d):
        filas = _dias(60)
        for f in filas:
            f["fecha"] = conv(date.fromisoformat(f["fecha"]))
        r = _temporal(filas, folds=5)
        assert r.es_viable and _ids_de(r, "test") == [_id(i) for i in range(48, 60)]


def test_gap_como_dict_del_json_es_lo_mismo_que_como_horizonte():
    filas = _dias(100)
    a = _temporal(filas, folds=5, gap={"magnitud": 5, "unidad": "days"})
    b = _temporal(filas, folds=5, gap=Horizonte(magnitud=5, unidad="days"))
    assert a.plan.digest() == b.plan.digest() and a.pliegues.digest() == b.pliegues.digest()
    assert a.plan.gap == Horizonte(magnitud=5, unidad="days")


@pytest.mark.parametrize("gap", [
    {"magnitud": 5}, {"unidad": "days"}, {"magnitud": -1, "unidad": "days"}, {"magnitud": 5, "unidad": "fortnights"},
    {"magnitud": "cinco", "unidad": "days"}, {"magnitud": 5, "unidad": "days", "extra": 1}, "5 days", 5, [5, "days"],
    {"magnitud": float("nan"), "unidad": "days"}, {},
])
def test_un_gap_que_no_se_entiende_es_un_bloqueo_con_motivo_no_una_excepcion(gap):
    r = _temporal(_dias(60), folds=5, gap=gap)
    assert not r.es_viable
    (b,) = r.bloqueos
    assert b.clave == "hueco_ilegible" and b.campo == "gap"
    assert "days" in b.motivo["es"] and "days" in b.motivo["en"]


def test_temporal_sin_columna_de_tiempo_es_un_bloqueo():
    r = proponer_particion(_dias(60), **_pide(split_type="temporal", seed=1, time_column=None))
    assert [b.clave for b in r.bloqueos] == ["falta_columna_de_tiempo"]


# --- fechas repetidas, pocos instantes, hueco que no cabe ------------------------

def test_pocos_instantes_distintos_reducen_los_pliegues_y_lo_dicen():
    """4 días distintos en desarrollo (de 5 posibles): K=5 pide 6 instantes, K=4 pide 5, K=3 pide 4."""
    filas = []
    for i in range(100):
        dia = min(i // 20, 4)                 # 5 días: 0..4, 20 filas cada uno
        filas.append({"id": _id(i), "fecha": (BASE + timedelta(days=dia)).isoformat(), "y": "a" if i % 2 else "b"})
    r = _temporal(filas, folds=5, test_fraction=0.2)
    assert r.es_viable
    assert _ids_de(r, "test") == [_id(i) for i in range(80, 100)]                      # el último día
    assert r.pliegues.folds == 3
    (limite,) = [x for x in r.limites if x.clave == "pliegues_reducidos_por_instantes"]
    assert limite.medida == {"pliegues_pedidos": 5, "pliegues_usados": 3, "instantes_distintos": 4}
    # cada día entero en un solo sitio: el primer bloque es el día 0 y cada validación es un día
    assert [p.valida for p in r.pliegues.pliegues][::-1] == [tuple(_id(i) for i in range(20 * d, 20 * d + 20)) for d in (1, 2, 3)]


def test_menos_de_3_instantes_en_desarrollo_se_rechaza_con_los_numeros():
    # 3 días de 50 filas y la prueba (20 % = 30 filas) se lleva parte del último: desarrollo = 2 días
    filas = [{"id": _id(i), "fecha": (BASE + timedelta(days=min(i // 50, 2))).isoformat(), "y": "a" if i % 2 else "b"}
             for i in range(150)]                 # 3 días de 50 filas; prueba 20 % = parte del último día
    r = _temporal(filas, folds=5)
    (b,) = r.bloqueos
    assert b.clave == "instantes_insuficientes" and b.medida["instantes_distintos"] == 2
    filas2 = [{"id": _id(i), "fecha": (BASE + timedelta(days=min(i // 40, 3))).isoformat(), "y": "a" if i % 2 else "b"}
              for i in range(160)]                # 4 días; prueba = el último, desarrollo = 3 días -> cabe K=2
    ok = _temporal(filas2, folds=5)
    assert ok.es_viable and ok.pliegues.folds == 2
    filas3 = [{"id": _id(i), "fecha": (BASE + timedelta(days=min(i // 60, 2))).isoformat(), "y": "a" if i % 2 else "b"}
              for i in range(180)]                # 3 días; prueba = el último; desarrollo = 2 días -> no caben ni 2 pliegues
    (b3,) = _temporal(filas3, folds=5).bloqueos
    assert b3.clave == "instantes_insuficientes"
    assert b3.medida == {"columna": "fecha", "instantes_distintos": 2, "minimo": 3}
    assert "2 instante(s)" in b3.motivo["es"] and "3" in b3.motivo["es"]


def test_un_hueco_que_deja_sin_entrenamiento_al_pliegue_mas_antiguo_reduce_pliegues():
    # 13 días con la frontera estricta (auditoría M2) es el mismo caso que 14 con la de antes: con 14, el pliegue más
    # antiguo de 4 entrenaría con un solo día (un «pos») y la guarda de clase lo bajaría a 3 por otra causa.
    r = _temporal(_dias(100), folds=5, gap={"magnitud": 13, "unidad": "days"})
    assert r.es_viable
    (limite,) = [x for x in r.limites if x.clave == "pliegues_reducidos_por_hueco"]
    assert limite.medida["pliegues_pedidos"] == 5 and limite.medida["pliegues_usados"] == r.pliegues.folds == 4
    assert all(len(p.entrena) > 0 for p in r.pliegues.pliegues)
    assert min(len(p.entrena) for p in r.pliegues.pliegues) == 2        # el pliegue más antiguo: los días 0 y 1
    # y con un hueco mayor que todo el pasado de un bloque, ni con 2 pliegues hay entrenamiento
    r2 = _temporal(_dias(100), folds=5, gap={"magnitud": 40, "unidad": "days"})
    assert [b.clave for b in r2.bloqueos] == ["cronologia_no_se_pudo_separar"]      # no queda ni prueba


def test_un_hueco_imposible_entre_desarrollo_y_prueba_sigue_siendo_cronologia_no_se_pudo_separar():
    r = _temporal(_dias(60), folds=5, gap={"magnitud": 1000, "unidad": "days"})
    assert [b.clave for b in r.bloqueos] == ["cronologia_no_se_pudo_separar"]


def test_hueco_sin_filas_para_entrenar_con_los_numeros():
    # 100 días, prueba 20 %: desarrollo ≈ días 0..~79; con 2 pliegues el bloque 0 acaba en el día ~26 y el hueco lo tapa entero
    r = _temporal(_dias(100), folds=5, gap={"magnitud": 35, "unidad": "days"}, objetivo=None)
    assert [b.clave for b in r.bloqueos] == ["hueco_sin_filas_para_entrenar"]
    assert "35" in r.bloqueos[0].motivo["es"]


# --- (c) guardas de clase en temporal ---------------------------------------------

def test_clase_repartida_en_el_tiempo_no_necesita_ninguna_guarda():
    r = _temporal(_dias(120, etiqueta=lambda i: "pos" if i % 6 == 0 else "neg"), folds=5)
    assert r.es_viable and r.limites == ()
    por_id = {f["id"]: f for f in _dias(120, etiqueta=lambda i: "pos" if i % 6 == 0 else "neg")}
    assert set(_clases(list(por_id.values()), _ids_de(r, "test"))) == {"pos", "neg"}
    for pl in r.pliegues.pliegues:
        assert set(_clases(list(por_id.values()), pl.valida)) == {"pos", "neg"}
        assert set(_clases(list(por_id.values()), pl.entrena)) == {"pos", "neg"}


def test_bloques_sin_positivos_bajan_los_pliegues_y_lo_dicen():
    """Positivos solo en los días 0-9, 30-39, 60-69 y 90-99 de 100: con 5 pliegues (6 bloques de ~13) el 2.º no tiene
    ninguno (días 14-26); con 4 (5 bloques de 16) todos sí."""
    positivos = set(range(0, 10)) | set(range(30, 40)) | set(range(60, 70)) | set(range(90, 100))
    filas = _dias(100, etiqueta=lambda i: "pos" if i in positivos else "neg")
    por_id = {f["id"]: f for f in filas}
    r = _temporal(filas, folds=5)
    assert r.es_viable
    assert r.pliegues.folds == 4
    (limite,) = [x for x in r.limites if x.clave == "pliegues_reducidos_por_clase_en_bloques"]
    assert limite.medida["pliegues_pedidos"] == 5 and limite.medida["pliegues_usados"] == r.pliegues.folds
    assert limite.medida["clase_mas_rara"] == "pos"
    for pl in r.pliegues.pliegues:
        assert "pos" in _clases(filas, pl.valida) and "neg" in _clases(filas, pl.valida)
        assert "pos" in _clases(filas, pl.entrena)
    assert "pos" in _clases(filas, _ids_de(r, "test"))


def test_si_ni_con_2_pliegues_cada_bloque_tiene_la_clase_se_rechaza_con_los_numeros():
    """Los positivos están todos en el primer tercio: el bloque que solo entrena los tiene, ninguno de validación."""
    filas = _dias(120, etiqueta=lambda i: "pos" if i < 30 or i >= 110 else "neg")      # la prueba (días ≥ 96) sí tiene
    por_id = {f["id"]: f for f in filas}
    r = _temporal(filas, folds=5)
    (b,) = r.bloqueos
    assert b.clave == "bloques_sin_clase" and b.campo == "objetivo"
    assert b.medida["clase"] == "pos" and b.medida["pliegues"] == 2
    assert b.medida["filas_de_la_clase_en_desarrollo"] == 30
    assert b.medida["filas_de_la_clase_por_bloque"] == [30, 0, 0]        # 3 bloques de 32: solo el primero
    assert b.medida["filas_por_bloque"] == [32, 32, 32]
    assert "30" in b.motivo["es"] and "30, 0, 0" in b.motivo["es"] and "30, 0, 0" in b.motivo["en"]


def test_la_prueba_reservada_sin_una_clase_se_rechaza_no_se_calla():
    filas = _dias(100, etiqueta=lambda i: "pos" if i < 70 and i % 3 == 0 else "neg")     # prueba = días 80-99
    r = _temporal(filas, folds=5)
    (b,) = r.bloqueos
    assert b.clave == "prueba_sin_una_clase"
    assert b.medida["clase"] == "pos" and b.medida["filas_de_prueba"] == 20
    assert b.medida["filas_de_prueba_por_clase"] == {"neg": 20, "pos": 0}
    assert "20 filas" in b.motivo["es"] and "20 rows" in b.motivo["en"]


def test_en_regresion_no_hay_guardas_de_clase():
    """Sin objetivo (regresión) los mismos datos que en el rechazo de arriba arrancan."""
    filas = _dias(120, etiqueta=lambda i: "pos" if i < 30 else "neg")
    r = _temporal(filas, folds=5, objetivo=None)
    assert r.es_viable and r.pliegues.folds == 5 and r.limites == ()


def test_con_una_sola_clase_se_declara_que_no_hay_guardas():
    r = _temporal(_dias(60, etiqueta=lambda i: "neg"), folds=5, objetivo="y")
    assert r.es_viable and [x.clave for x in r.limites] == ["una_sola_clase_sin_guardas"]


# ---------------------------------------------------------------------------
# (b) groups estratificado por clase respetando la unidad
# ---------------------------------------------------------------------------

def _sujetos(n_sujetos, filas_de, positivo, prefijo=""):
    filas = []
    for s in range(n_sujetos):
        for i in range(filas_de(s)):
            filas.append({"id": f"o{s:03d}_{i}", "suj": f"{prefijo}{s:03d}", "y": "pos" if positivo(s, i) else "neg"})
    return filas


def _groups(filas, **kw):
    base = _pide(split_type="groups", seed=0, unit_id_field="suj", objetivo="y")
    base.update(kw)
    return proponer_particion(filas, **base)


def _unidad_de(filas):
    return {f["id"]: f["suj"] for f in filas}


def _ninguna_unidad_partida(filas, r):
    """Ninguna unidad en dos roles ni en dos pliegues de una repetición, ni a los dos lados de un pliegue."""
    u = _unidad_de(filas)
    roles: dict[str, set] = {}
    for o, rol in r.plan.assignments.items():
        roles.setdefault(u[o], set()).add(rol)
    assert all(len(v) == 1 for v in roles.values()), {k: v for k, v in roles.items() if len(v) > 1}
    for rep in {p.repeticion for p in r.pliegues.pliegues}:
        en_pliegue: dict[str, set] = {}
        for p in (p for p in r.pliegues.pliegues if p.repeticion == rep):
            assert {u[o] for o in p.entrena}.isdisjoint({u[o] for o in p.valida})
            for o in p.valida:
                en_pliegue.setdefault(u[o], set()).add(p.pliegue)
        assert all(len(v) == 1 for v in en_pliegue.values())


def _todas_las_clases_en_todas_partes(filas, r, clases=("pos", "neg")):
    assert set(_clases(filas, _ids_de(r, "test"))) == set(clases)
    for p in r.pliegues.pliegues:
        assert set(_clases(filas, p.valida)) == set(clases), (p.repeticion, p.pliegue)
        assert set(_clases(filas, p.entrena)) == set(clases)


def test_el_caso_medido_8_sujetos_con_positivos_en_2_hoy_deja_la_prueba_con_una_sola_clase():
    """Lo que midió la sonda del 27-10 (4 semillas), con el código de la rama 139."""
    filas = _sujetos(8, lambda s: 10, lambda s, i: s in (0, 1) and i % 2 == 0)
    una_sola = 0
    for semilla in range(4):
        antes = ANTES.proponer_particion(filas, **_pide(split_type="groups", seed=semilla, unit_id_field="suj",
                                                         objetivo="y", folds=5))
        prueba = [f["id"] for f in filas if antes.plan.assignments[f["id"]] == "test"]
        una_sola += len(set(_clases(filas, prueba))) == 1
        assert antes.limites == ()               # …y sin declarar nada
    assert una_sola == 4


def test_el_caso_medido_ahora_se_rechaza_con_los_numeros_en_las_4_semillas():
    filas = _sujetos(8, lambda s: 10, lambda s, i: s in (0, 1) and i % 2 == 0)
    for semilla in range(4):
        r = _groups(filas, seed=semilla, folds=5)
        assert not r.es_viable and r.plan is None and r.pliegues is None
        (b,) = r.bloqueos
        assert b.clave == "clase_sin_grupos_suficientes" and b.campo == "unit_id_field"
        assert b.medida["clase"] == "pos" and b.medida["grupos_con_la_clase"] == 2
        assert b.medida["grupos_necesarios_con_los_pedidos"] == 6      # 5 pliegues + la prueba
        assert b.medida["grupos_necesarios_con_2_pliegues"] == 3       # 2 pliegues + la prueba
        assert b.medida["grupos_por_clase"] == {"neg": 8, "pos": 2}
        for texto in (b.motivo["es"], b.motivo["en"]):
            assert "pos" in texto and "2" in texto and "6" in texto and "3" in texto and "suj" in texto


def test_clase_positiva_en_solo_2_sujetos_se_rechaza_tambien_pidiendo_2_pliegues():
    filas = _sujetos(20, lambda s: 5, lambda s, i: s in (3, 14) and i < 3)
    r = _groups(filas, folds=2)
    (b,) = r.bloqueos
    assert b.clave == "clase_sin_grupos_suficientes"
    assert (b.medida["grupos_con_la_clase"], b.medida["grupos_necesarios_con_los_pedidos"]) == (2, 3)


def test_con_3_sujetos_positivos_cabe_con_2_pliegues_y_se_declara_la_reduccion():
    filas = _sujetos(8, lambda s: 10, lambda s, i: s in (0, 1, 2) and i % 2 == 0)
    for semilla in range(4):
        r = _groups(filas, seed=semilla, folds=5)
        assert r.es_viable and r.pliegues.folds == 2 and r.plan.folds == 2
        (limite,) = r.limites
        assert limite.clave == "pliegues_reducidos_por_clase_en_grupos" and limite.campo == "unit_id_field"
        assert limite.medida["pliegues_pedidos"] == 5 and limite.medida["pliegues_usados"] == 2
        assert limite.medida["grupos_con_la_clase"] == 3 and limite.medida["grupos_necesarios"] == 6
        assert "3 grupo(s) de suj" in limite.motivo["es"]
        _todas_las_clases_en_todas_partes(filas, r)
        _ninguna_unidad_partida(filas, r)
        # cada sujeto positivo está en un sitio distinto: uno en la prueba y uno en cada pliegue
        u = _unidad_de(filas)
        sitios = [r.plan.assignments[f"o{s:03d}_0"] for s in (0, 1, 2)]
        assert sorted(sitios) == ["development", "development", "test"]


def test_con_5_sujetos_positivos_caben_4_pliegues_y_la_prueba():
    filas = _sujetos(20, lambda s: 6, lambda s, i: s in (1, 4, 9, 12, 17) and i < 3)
    r = _groups(filas, folds=4)
    assert r.es_viable and r.pliegues.folds == 4 and r.limites == ()
    _todas_las_clases_en_todas_partes(filas, r)
    # …pero 5 pliegues + la prueba piden 6 y solo hay 5: baja a 4 declarándolo
    r5 = _groups(filas, folds=5)
    assert r5.pliegues.folds == 4
    assert r5.limites[0].medida["grupos_con_la_clase"] == 5 and r5.limites[0].medida["grupos_necesarios"] == 6
    _todas_las_clases_en_todas_partes(filas, r5)


def test_sin_prueba_reservada_solo_hacen_falta_tantos_grupos_como_pliegues():
    """`SplitPlan` exige una prueba, así que la pieza que reparte se llama directa."""
    from matrixai.training.particion_por_diseno_grupos import repartir_grupos_estratificado    # noqa: PLC0415
    filas = _sujetos(20, lambda s: 6, lambda s, i: s in (1, 4, 9) and i < 3)
    por_unidad: dict[str, list[str]] = {}
    for f in filas:
        por_unidad.setdefault(f["suj"], []).append(f["id"])
    clase = {f["id"]: f["y"] for f in filas}
    r = repartir_grupos_estratificado(por_unidad, clase, folds=3, test_fraction=0.0, seed=0, repeats=1)
    assert r.pliegues_usados == 3 and r.limites == () and r.grupos_de_prueba == ()
    assert sorted(g for cubo in r.cubos_por_repeticion[0] for g in cubo) == sorted(por_unidad)
    for cubo in r.cubos_por_repeticion[0]:
        assert {clase[o] for g in cubo for o in por_unidad[g]} == {"pos", "neg"}
    # con 2 sujetos positivos ya no caben 3 pliegues (sin prueba bastan tantos grupos como pliegues): baja a 2
    dos = _sujetos(20, lambda s: 6, lambda s, i: s in (1, 4) and i < 3)
    por2 = {}
    for f in dos:
        por2.setdefault(f["suj"], []).append(f["id"])
    r2 = repartir_grupos_estratificado(por2, {f["id"]: f["y"] for f in dos}, folds=3, test_fraction=0.0, seed=0, repeats=1)
    assert r2.pliegues_usados == 2 and r2.limites[0].medida["grupos_necesarios"] == 3


def test_ids_de_sujeto_de_texto_con_ceros_a_la_izquierda_no_se_confunden():
    """'007' y '7' son sujetos distintos (y '0123' también): ninguno se parte ni se funde con otro."""
    filas = []
    nombres = ["007", "7", "0123", "123", "00", "0", "010", "10", "05", "5", "0099", "99"]
    for k, nombre in enumerate(nombres):
        for i in range(6):
            filas.append({"id": f"o{k}_{i}", "suj": nombre, "y": "pos" if k in (0, 2, 4, 6, 8, 10) and i < 3 else "neg"})
    r = _groups(filas, folds=3, seed=2)
    assert r.es_viable
    assert set(r.plan.units.values()) == set(nombres)           # los 12, tal como vinieron
    assert r.plan.units["o0_0"] == "007" and r.plan.units["o1_0"] == "7"
    _ninguna_unidad_partida(filas, r)
    _todas_las_clases_en_todas_partes(filas, r)
    # 12 sujetos distintos repartidos: prueba + desarrollo cubren los 12
    assert len({f["suj"] for f in filas if r.plan.assignments[f["id"]] in ("test", "development")}) == 12


def test_sujetos_de_una_sola_fila():
    """30 sujetos, una fila cada uno (20 neg, 10 pos): es un iid con otro nombre; todo se reparte parejo."""
    filas = _sujetos(30, lambda s: 1, lambda s, i: s % 3 == 0)
    r = _groups(filas, folds=5, test_fraction=0.2)
    assert r.es_viable and r.limites == ()
    assert len(_ids_de(r, "test")) == 6
    assert _clases(filas, _ids_de(r, "test")) == {"neg": 4, "pos": 2}
    tamanos = sorted(len(p.valida) for p in r.pliegues.pliegues)
    assert tamanos == [4, 4, 4, 6, 6] or max(tamanos) - min(tamanos) <= 2
    assert sum(tamanos) == 24
    _todas_las_clases_en_todas_partes(filas, r)
    _ninguna_unidad_partida(filas, r)


def test_un_sujeto_con_el_40_por_ciento_de_las_filas():
    """El sujeto 0 trae 40 de 100 filas, con las dos clases. Va entero a un sitio, y los demás no se quedan sin clase."""
    filas = _sujetos(11, lambda s: 40 if s == 0 else 6, lambda s, i: (s == 0 and i % 2 == 0) or (s in (3, 5, 8, 10) and i < 2))
    assert len(filas) == 100 and sum(f["suj"] == "000" for f in filas) == 40
    for semilla in range(5):
        r = _groups(filas, seed=semilla, folds=3, test_fraction=0.2)
        assert r.es_viable
        _ninguna_unidad_partida(filas, r)
        _todas_las_clases_en_todas_partes(filas, r)
        grande = [f["id"] for f in filas if f["suj"] == "000"]
        donde = {r.plan.assignments[o] for o in grande}
        assert len(donde) == 1
        if donde == {"development"}:
            assert sum(set(grande) <= set(p.valida) for p in r.pliegues.pliegues) == 1     # en UN pliegue, entero
        assert r.cobertura.filas_sin_validacion == 0


def test_ninguna_unidad_en_dos_sitios_con_muchas_semillas_y_repeticiones():
    rng = random.Random(8)
    filas = _sujetos(60, lambda s: rng.randint(1, 9), lambda s, i: s % 4 == 0 and i % 2 == 0)
    for semilla in range(10):
        r = _groups(filas, seed=semilla, folds=5, repeats=3)
        assert r.es_viable and r.limites == ()
        _ninguna_unidad_partida(filas, r)
        _todas_las_clases_en_todas_partes(filas, r)
        for rep in range(3):
            en_valida = [o for p in r.pliegues.pliegues if p.repeticion == rep for o in p.valida]
            assert sorted(en_valida) == _ids_de(r, "development")        # las `valida` cubren el desarrollo, una vez
        assert r.cobertura.filas_sin_validacion == 0


def test_la_prueba_es_la_misma_en_todas_las_repeticiones_y_los_pliegues_cambian():
    rng = random.Random(2)
    filas = _sujetos(40, lambda s: rng.randint(2, 7), lambda s, i: s % 3 == 0 and i == 0)
    r = _groups(filas, folds=4, repeats=2, seed=5)
    valida = {rep: sorted(tuple(sorted(p.valida)) for p in r.pliegues.pliegues if p.repeticion == rep) for rep in (0, 1)}
    assert valida[0] != valida[1]


def test_la_semilla_cambia_la_prueba_y_la_misma_semilla_la_repite():
    rng = random.Random(3)
    filas = _sujetos(50, lambda s: rng.randint(1, 8), lambda s, i: s % 4 == 0 and i % 2 == 0)
    pruebas = {tuple(_ids_de(_groups(filas, seed=s), "test")) for s in range(8)}
    assert len(pruebas) >= 6
    assert _ids_de(_groups(filas, seed=4), "test") == _ids_de(_groups(filas, seed=4), "test")
    assert _groups(filas, seed=4).plan.digest() == _groups(filas, seed=4).plan.digest()


def test_la_fraccion_de_prueba_se_respeta_y_la_clase_se_reparte_como_en_el_total():
    rng = random.Random(6)
    filas = _sujetos(80, lambda s: rng.randint(3, 8), lambda s, i: s % 5 == 0 and i % 2 == 0)
    total_pos = _clases(filas, [f["id"] for f in filas])["pos"] / len(filas)
    for semilla in range(5):
        r = _groups(filas, seed=semilla, folds=5)
        prueba = _ids_de(r, "test")
        assert abs(len(prueba) / len(filas) - 0.2) <= 0.04
        assert abs(_clases(filas, prueba)["pos"] / len(prueba) - total_pos) <= 0.06
        for p in r.pliegues.pliegues:
            assert abs(_clases(filas, p.valida)["pos"] / len(p.valida) - total_pos) <= 0.08


def test_sklearn_StratifiedGroupKFold_como_oraculo_de_calidad():
    """Donde sklearn tiene con qué (muchos sujetos con la clase), el reparto no es peor que el suyo."""
    pytest.importorskip("sklearn")
    from sklearn.model_selection import StratifiedGroupKFold    # noqa: PLC0415
    rng = random.Random(10)
    filas = _sujetos(60, lambda s: rng.randint(2, 9), lambda s, i: s % 4 == 0 and i % 2 == 0)
    total = _clases(filas, [f["id"] for f in filas])["pos"] / len(filas)
    for semilla in range(3):
        r = _groups(filas, seed=semilla, folds=5, test_fraction=0.2)
        dev = [f for f in filas if r.plan.assignments[f["id"]] == "development"]
        ours = max(abs(_clases(filas, p.valida)["pos"] / len(p.valida) - total) for p in r.pliegues.pliegues)
        y = [f["y"] == "pos" for f in dev]
        g = [f["suj"] for f in dev]
        theirs = max(abs(sum(y[i] for i in v) / len(v) - total)
                     for _, v in StratifiedGroupKFold(5, shuffle=True, random_state=semilla).split(dev, y, g))
        assert ours <= theirs + 0.03, (semilla, ours, theirs)


def test_sklearn_StratifiedGroupKFold_no_avisa_donde_nosotros_si():
    """El caso medido: sklearn deja pliegues sin positivos sin decir nada; aquí se rechaza."""
    pytest.importorskip("sklearn")
    import warnings    # noqa: PLC0415
    from sklearn.model_selection import StratifiedGroupKFold    # noqa: PLC0415
    filas = _sujetos(8, lambda s: 10, lambda s, i: s in (0, 1) and i % 2 == 0)
    y = [f["y"] == "pos" for f in filas]
    g = [f["suj"] for f in filas]
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        sin_positivos = sum(sum(y[i] for i in v) == 0 for _, v in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(filas, y, g))
    assert sin_positivos >= 1 and not avisos
    assert not _groups(filas, folds=5).es_viable


def test_multiclase_todas_las_clases_en_la_prueba_y_en_cada_pliegue():
    rng = random.Random(1)
    filas = []
    for s in range(30):
        for i in range(rng.randint(2, 6)):
            filas.append({"id": f"o{s}_{i}", "suj": f"{s:03d}", "y": ("a", "b", "c")[(s + i) % 3 if s % 5 else 0]})
    r = _groups(filas, folds=4, seed=3)
    assert r.es_viable
    _todas_las_clases_en_todas_partes(filas, r, clases=("a", "b", "c"))
    _ninguna_unidad_partida(filas, r)


def test_tres_clases_en_grupos_que_se_solapan_el_reparto_exacto_las_cubre():
    """Un voraz falla aquí aunque el reparto existe (medido el 28-10: 125 de 137 casos); la búsqueda con vuelta atrás, no."""
    # clases por grupo (7 grupos, 4 clases): cubre 3 partes (prueba + 2 pliegues) con cada clase en 3+ grupos
    clases_de_grupo = {0: [0, 2], 1: [1, 2], 2: [0, 1], 3: [1, 2], 4: [0, 3], 5: [3], 6: [3, 0]}
    filas = [{"id": f"g{g}_{i}", "suj": f"g{g}", "y": f"c{c}"} for g, cs in clases_de_grupo.items()
             for i, c in enumerate(cs * 2)]
    r = _groups(filas, folds=2, test_fraction=0.3, seed=0)
    assert r.es_viable, [b.motivo["es"] for b in r.bloqueos]
    _todas_las_clases_en_todas_partes(filas, r, clases=("c0", "c1", "c2", "c3"))


# Dos repartos en los que, SIN buscar la cobertura de clases, el voraz deja la prueba o un pliegue sin una clase
# (hallados por búsqueda aleatoria el 28-10; cada grupo es la lista de las clases de sus filas).
_GRUPOS_QUE_EXIGEN_COBERTURA = {
    "dos_clases": {"g0": "aaa", "g1": "a", "g2": "aa", "g3": "ba", "g4": "b", "g5": "bbbb", "g6": "aa", "g7": "bbb",
                   "g8": "a", "g9": "aaaaa"},
    "tres_clases": {"g0": "aa", "g1": "cc", "g2": "b", "g3": "aaaaa", "g4": "cbc", "g5": "ccc", "g6": "aaa", "g7": "aaaaa",
                    "g8": "c", "g9": "bbb", "g10": "aaa", "g11": "bbb", "g12": "acaca", "g13": "cc"},
    # sin buscar la cobertura de los PLIEGUES, uno se queda sin una clase
    "pliegue_sin_clase": {'g0': 'aa', 'g1': 'aa', 'g2': 'aca', 'g3': 'cbc', 'g4': 'bc', 'g5': 'bbb', 'g6': 'bbbb', 'g7': 'cccc', 'g8': 'baba'},
    # con una cobertura a medias (sin volver atrás), también
    "cobertura_con_vuelta_atras": {'g0': 'bbb', 'g1': 'a', 'g2': 'bb', 'g3': 'cdc', 'g4': 'cdc', 'g5': 'cccc', 'g6': 'bcbc', 'g7': 'a', 'g8': 'cdc', 'g9': 'aaaa', 'g10': 'aba', 'g11': 'aa', 'g12': 'bcbc', 'g13': 'bdbd', 'g14': 'd'},
}


@pytest.mark.parametrize("caso", sorted(_GRUPOS_QUE_EXIGEN_COBERTURA))
def test_la_cobertura_de_clases_es_lo_que_deja_todas_en_la_prueba_y_en_cada_pliegue(caso):
    grupos = _GRUPOS_QUE_EXIGEN_COBERTURA[caso]
    filas = [{"id": f"{g}_{i}", "suj": g, "y": c} for g, clases in grupos.items() for i, c in enumerate(clases)]
    todas = tuple(sorted(set("".join(grupos.values()))))
    r = _groups(filas, folds=3, test_fraction=0.2, seed=0)
    assert r.es_viable and r.pliegues.folds == 3 and r.limites == ()
    _todas_las_clases_en_todas_partes(filas, r, clases=todas)
    _ninguna_unidad_partida(filas, r)



def test_clases_que_no_tienen_reparto_aunque_cada_una_aparezca_en_los_grupos_necesarios():
    """Cada clase está en exactamente 3 grupos (= prueba + 2 pliegues) pero los grupos se pisan: comprobado por
    fuerza bruta que no hay asignación de los 7 grupos a 3 partes con las 4 clases en cada una."""
    clases_de_grupo = {0: [0, 2], 1: [1, 2], 2: [0, 1], 3: [1, 2], 4: [0, 3], 5: [3], 6: [3]}
    filas = [{"id": f"g{g}_{i}", "suj": f"g{g}", "y": f"c{c}"} for g, cs in clases_de_grupo.items()
             for i, c in enumerate(cs * 2)]
    import itertools    # noqa: PLC0415
    assert not any(all(all(any(c in clases_de_grupo[g] for g in clases_de_grupo if asg[g] == parte) for c in range(4))
                       for parte in range(3))
                   for asg in itertools.product(range(3), repeat=7))
    r = _groups(filas, folds=2, test_fraction=0.3, seed=0)
    (b,) = r.bloqueos
    assert b.clave == "clases_sin_reparto_por_grupos" and not r.es_viable
    assert b.medida["grupos_por_clase"] == {"c0": 3, "c1": 3, "c2": 3, "c3": 3}


def test_con_una_sola_clase_groups_reparte_como_siempre_y_lo_declara():
    filas = _sujetos(10, lambda s: 4, lambda s, i: False)
    r = _groups(filas, folds=3)
    antes = ANTES.proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="suj", objetivo="y", folds=3))
    assert r.plan.assignments == antes.plan.assignments and r.pliegues.a_json() == antes.pliegues.a_json()
    assert [x.clave for x in r.limites] == ["una_sola_clase_sin_guardas"]


def test_groups_con_menos_de_2_grupos_sigue_siendo_grupos_insuficientes():
    filas = [{"id": f"o{i}", "suj": "solo", "y": "a" if i % 2 else "b"} for i in range(10)]
    assert [b.clave for b in _groups(filas).bloqueos] == ["grupos_insuficientes"]


def test_regresion_no_tiene_guardas_aunque_los_valores_sean_de_un_solo_sujeto():
    rng = random.Random(0)
    filas = [{"id": f"o{s}_{i}", "suj": f"s{s}", "v": rng.random()} for s in range(6) for i in range(4)]
    r = proponer_particion(filas, **_pide(split_type="groups", seed=1, unit_id_field="suj", objetivo=None, folds=5))
    # La regresión no tiene guardas de clase: ningún límite ni bloqueo habla de clases. (Antes afirmaba además
    # `pliegues_reducidos_por_grupos`: con la 139 la prueba se llevaba 2 de 6 sujetos, el 33 % frente al 20 %
    # pedido; con el margen se lleva 1 y a desarrollo le quedan 5, que dan los 5 pliegues.)
    assert r.es_viable and not any("clase" in x.clave for x in (*r.limites, *r.bloqueos))
    assert not _desborda_la_prueba(r, 0.2)


def test_cohorte_externa_con_groups_estratificado_no_entra_en_ningun_pliegue():
    filas = _sujetos(30, lambda s: 4, lambda s, i: s % 3 == 0 and i < 2)
    externos = [f["id"] for f in filas if f["suj"] in ("028", "029")]
    r = _groups(filas, folds=4, ids_cohorte_externa=externos)
    assert r.es_viable
    assert _ids_de(r, "external_test") == sorted(externos)
    en_pliegues = {o for p in r.pliegues.pliegues for o in p.entrena + p.valida}
    assert en_pliegues.isdisjoint(externos)


# ---------------------------------------------------------------------------
# (e) las dos mitades de una `valida`
# ---------------------------------------------------------------------------

def test_mitades_iid_son_exactamente_las_posicionales_de_hoy():
    for n in (0, 1, 2, 7, 10, 101):
        ids = [f"o{i}" for i in range(n)]
        m = partir_valida_en_mitades(ids, diseno="iid")
        assert list(m.para_calibrar) == ids[: n // 2] and list(m.para_el_umbral) == ids[n // 2:]
        assert list(m.indices_para_calibrar) == list(range(n // 2))
        assert list(m.indices_para_el_umbral) == list(range(n // 2, n))
        assert m.limites == ()


def test_mitades_groups_no_parten_una_unidad_donde_las_posicionales_si():
    # 5 sujetos de 4 filas seguidas: el corte posicional (10 de 20) cae a mitad del sujeto 2
    ids = [f"o{s}_{i}" for s in range(5) for i in range(4)]
    unidad = {o: o.split("_")[0] for o in ids}
    mitad = len(ids) // 2
    assert {unidad[o] for o in ids[:mitad]} & {unidad[o] for o in ids[mitad:]} == {"o2"}      # hoy: partido
    m = partir_valida_en_mitades(ids, diseno="groups", unidad_por_id=unidad, semilla=1)
    assert {unidad[o] for o in m.para_calibrar}.isdisjoint({unidad[o] for o in m.para_el_umbral})
    assert sorted(m.para_calibrar + m.para_el_umbral) == sorted(ids)
    assert [ids[i] for i in m.indices_para_calibrar] == list(m.para_calibrar)
    assert [ids[i] for i in m.indices_para_el_umbral] == list(m.para_el_umbral)
    assert 6 <= len(m.para_calibrar) <= 14        # parecidas (4 filas por sujeto)


def test_mitades_groups_reparten_la_clase_entre_las_dos():
    rng = random.Random(3)
    filas = _sujetos(24, lambda s: rng.randint(1, 5), lambda s, i: s % 3 == 0)
    ids = [f["id"] for f in filas]
    clase = {f["id"]: f["y"] for f in filas}
    m = partir_valida_en_mitades(ids, diseno="groups", unidad_por_id=_unidad_de(filas), clase_por_id=clase, semilla=2)
    assert set(_clases(filas, m.para_calibrar)) == {"pos", "neg"} == set(_clases(filas, m.para_el_umbral))
    assert m.limites == ()
    u = _unidad_de(filas)
    assert {u[o] for o in m.para_calibrar}.isdisjoint({u[o] for o in m.para_el_umbral})


def test_mitades_groups_con_3_clases_cubren_las_tres_en_las_dos_mitades():
    """Sin buscar la cobertura (solo por coste) una mitad se queda sin una clase aunque el reparto existe."""
    grupos = {"g0": "bc", "g1": "cc", "g2": "acc", "g3": "c", "g4": "ab"}
    ids = [f"{g}_{i}" for g, cs in grupos.items() for i in range(len(cs))]
    clase = {f"{g}_{i}": c for g, cs in grupos.items() for i, c in enumerate(cs)}
    unidad = {o: o.split("_")[0] for o in ids}
    m = partir_valida_en_mitades(ids, diseno="groups", unidad_por_id=unidad, clase_por_id=clase)
    assert m.limites == ()
    assert {clase[o] for o in m.para_calibrar} == {"a", "b", "c"} == {clase[o] for o in m.para_el_umbral}
    assert {unidad[o] for o in m.para_calibrar}.isdisjoint({unidad[o] for o in m.para_el_umbral})


def test_mitades_groups_con_una_clase_que_solo_trae_un_sujeto_lo_declaran():
    filas = _sujetos(6, lambda s: 3, lambda s, i: s == 0)
    clase = {f["id"]: f["y"] for f in filas}
    m = partir_valida_en_mitades([f["id"] for f in filas], diseno="groups", unidad_por_id=_unidad_de(filas),
                                 clase_por_id=clase)
    (limite,) = m.limites
    assert limite.clave == "mitad_sin_una_clase" and limite.medida["clase"] == "pos"
    assert sorted((limite.medida["filas_para_calibrar"], limite.medida["filas_para_el_umbral"])) == [0, 3]


def test_mitades_groups_con_una_sola_unidad_cortan_por_posicion_y_lo_declaran():
    ids = [f"o{i}" for i in range(9)]
    m = partir_valida_en_mitades(ids, diseno="groups", unidad_por_id={o: "unica" for o in ids})
    assert list(m.para_calibrar) == ids[:4] and list(m.para_el_umbral) == ids[4:]
    assert [x.clave for x in m.limites] == ["valida_con_una_sola_unidad"]


def test_mitades_groups_nunca_dejan_una_mitad_vacia_con_2_unidades_o_mas():
    for tamanos in ([1, 9], [9, 1], [5, 5], [1, 1, 30], [2, 2, 2, 2]):
        ids = [f"s{k}_{i}" for k, t in enumerate(tamanos) for i in range(t)]
        unidad = {o: o.split("_")[0] for o in ids}
        m = partir_valida_en_mitades(ids, diseno="groups", unidad_por_id=unidad)
        assert m.para_calibrar and m.para_el_umbral
        assert {unidad[o] for o in m.para_calibrar}.isdisjoint({unidad[o] for o in m.para_el_umbral})


def test_mitades_temporal_la_anterior_calibra_y_la_posterior_elige_el_umbral():
    ids = [_id(i) for i in range(20)]
    tiempo = {o: float(i) for i, o in enumerate(ids)}
    barajados = ids[:]
    random.Random(1).shuffle(barajados)           # la `valida` NO viene ordenada
    m = partir_valida_en_mitades(barajados, diseno="temporal", tiempo_por_id=tiempo)
    assert list(m.para_calibrar) == ids[:10] and list(m.para_el_umbral) == ids[10:]      # en orden cronológico
    assert [barajados[i] for i in m.indices_para_calibrar] == ids[:10]
    assert [barajados[i] for i in m.indices_para_el_umbral] == ids[10:]


def test_mitades_temporal_no_corta_un_instante():
    ids = [f"o{i}" for i in range(12)]
    tiempo = {o: float(i // 4) for i, o in enumerate(ids)}          # 3 instantes de 4 filas
    m = partir_valida_en_mitades(ids, diseno="temporal", tiempo_por_id=tiempo)
    assert {tiempo[o] for o in m.para_calibrar}.isdisjoint({tiempo[o] for o in m.para_el_umbral})
    assert max(tiempo[o] for o in m.para_calibrar) < min(tiempo[o] for o in m.para_el_umbral)
    assert len(m.para_calibrar) in (4, 8)


def test_mitades_temporal_con_un_solo_instante_cortan_por_posicion_y_lo_declaran():
    ids = [f"o{i}" for i in range(6)]
    m = partir_valida_en_mitades(ids, diseno="temporal", tiempo_por_id={o: 5.0 for o in ids})
    assert list(m.para_calibrar) == ids[:3] and list(m.para_el_umbral) == ids[3:]
    assert [x.clave for x in m.limites] == ["valida_con_un_solo_instante"]


def test_mitades_de_las_valida_reales_de_cada_diseno():
    rng = random.Random(4)
    filas = _sujetos(40, lambda s: rng.randint(2, 6), lambda s, i: s % 3 == 0 and i == 0)
    r = _groups(filas, folds=4)
    u = _unidad_de(filas)
    clase = {f["id"]: f["y"] for f in filas}
    for p in r.pliegues.pliegues:
        m = partir_valida_en_mitades(list(p.valida), diseno="groups", unidad_por_id=u, clase_por_id=clase)
        assert {u[o] for o in m.para_calibrar}.isdisjoint({u[o] for o in m.para_el_umbral})
    t = _temporal(_dias(120), folds=5)
    tiempo = {f["id"]: float(int(f["id"][1:])) for f in _dias(120)}
    for p in t.pliegues.pliegues:
        m = partir_valida_en_mitades(list(p.valida), diseno="temporal", tiempo_por_id=tiempo)
        assert max(tiempo[o] for o in m.para_calibrar) < min(tiempo[o] for o in m.para_el_umbral)


def test_mitades_de_un_diseno_sin_soporte_se_niegan():
    with pytest.raises(ValueError, match="groups_and_time"):
        partir_valida_en_mitades(["a", "b"], diseno="groups_and_time")
    with pytest.raises(ValueError):
        partir_valida_en_mitades(["a", "b"], diseno="temporal")
    with pytest.raises(ValueError):
        partir_valida_en_mitades(["a", "b"], diseno="groups")


# ---------------------------------------------------------------------------
# Los textos
# ---------------------------------------------------------------------------

_CLAVES_NUEVAS = (
    "falta_columna_de_tiempo", "tiempo_ilegible", "fecha_ambigua", "hueco_ilegible", "una_sola_clase_sin_guardas",
    "repeticiones_fijadas_a_1", "tiempo_numerico_sin_unidad", "instantes_insuficientes",
    "pliegues_reducidos_por_instantes", "pliegues_reducidos_por_hueco", "pliegues_reducidos_por_clase_en_bloques",
    "bloques_sin_clase", "prueba_sin_una_clase", "pliegues_reducidos_por_clase_en_grupos", "clase_sin_grupos_suficientes",
    "clases_sin_reparto_por_grupos", "valida_con_una_sola_unidad", "valida_con_un_solo_instante", "mitad_sin_una_clase",
    "hueco_sin_filas_para_entrenar", "pliegues_reducidos_por_reparto_de_clases_en_grupos",
    # reparación de la auditoría (27-10)
    "fraccion_de_prueba_desviada", "entrena_sin_clase", "hueco_en_tiempo_numerico", "unidad_vacia",
    "cohorte_externa_comparte_unidad", "sin_prueba_reservada", "grupos_insuficientes_para_prueba_y_pliegues",
    "prueba_recortada_para_dejar_grupos",
    # reparación de la re-auditoría (28-10)
    "entrena_sin_clase_por_el_hueco",
)


@pytest.mark.parametrize("clave", _CLAVES_NUEVAS)
def test_cada_motivo_nuevo_se_compone_en_los_dos_idiomas_sin_huecos_sueltos(clave):
    assert clave in MOTIVOS and set(MOTIVOS[clave]) == {"es", "en"}
    assert huecos_de(MOTIVOS[clave]["es"]) == huecos_de(MOTIVOS[clave]["en"])
    textos = motivo(clave, **{h: "XX" for h in huecos_de(MOTIVOS[clave]["es"])})
    for idioma in ("es", "en"):
        assert "XX" in textos[idioma] and "{" not in textos[idioma]


# ---------------------------------------------------------------------------
# Reparación de la auditoría independiente del C1 (27-10): cada prueba salió ROJA antes de su arreglo
# ---------------------------------------------------------------------------

def _gigante(filas_del_gigante):
    """1000 filas: un sujeto `gigante` con las dos clases y 40 sujetos de una sola clase cada uno (8 «pos», 32 «neg»)."""
    return [{"id": f"o{i:04d}", "suj": ("gigante" if i < filas_del_gigante else f"s{i % 40:02d}"),
             "y": "pos" if i % 5 == 0 else "neg"} for i in range(1000)]


@pytest.mark.parametrize("filas_del_gigante", [600, 400])
def test_I1_un_sujeto_enorme_con_las_dos_clases_no_se_lleva_la_prueba(filas_del_gigante):
    """Auditoría I1: la cobertura de clases ponía primero en la prueba el grupo que trae las dos clases —el más
    grande—, y la prueba salía del 60 % (o del 40 %) en 10 semillas de 10, sin decir nada. Existe un reparto
    cercano al 20 % (un sujeto «pos» entero y grupos «neg»): tiene que salir ese."""
    filas = _gigante(filas_del_gigante)
    gigante = {f["id"] for f in filas if f["suj"] == "gigante"}
    for semilla in range(10):
        r = _groups(filas, seed=semilla, folds=5, test_fraction=0.2)
        assert r.es_viable
        fraccion = len(_ids_de(r, "test")) / len(filas)
        assert 0.15 <= fraccion <= 0.25, (semilla, fraccion)
        assert {r.plan.assignments[o] for o in gigante} == {"development"}
        assert not [x for x in r.limites if x.clave == "fraccion_de_prueba_desviada"]
        _ninguna_unidad_partida(filas, r)
        _todas_las_clases_en_todas_partes(filas, r)


def test_I1_el_sujeto_del_40_por_ciento_del_constructor_deja_la_prueba_cerca_de_lo_pedido():
    filas = _sujetos(11, lambda s: 40 if s == 0 else 6, lambda s, i: (s == 0 and i % 2 == 0) or (s in (3, 5, 8, 10) and i < 2))
    for semilla in range(5):
        r = _groups(filas, seed=semilla, folds=3, test_fraction=0.2)
        assert 0.1 <= len(_ids_de(r, "test")) / len(filas) <= 0.3


def test_I1_si_la_prueba_no_puede_acercarse_a_lo_pedido_se_declara_con_los_numeros():
    """Tres sujetos de ~333 filas, todos con las dos clases, y 2 pliegues: la prueba se lleva uno entero (33 %
    frente al 20 % pedido, más de un 50 % de desvío relativo). No hay reparto mejor: se dice."""
    filas = [{"id": f"o{i:04d}", "suj": f"S{i % 3}", "y": "pos" if i % 4 == 0 else "neg"} for i in range(1000)]
    r = _groups(filas, seed=0, folds=2, test_fraction=0.2)
    assert r.es_viable
    (limite,) = [x for x in r.limites if x.clave == "fraccion_de_prueba_desviada"]
    filas_de_prueba = len(_ids_de(r, "test"))
    assert limite.campo == "unit_id_field"
    assert limite.medida["fraccion_pedida"] == 0.2
    assert limite.medida["fraccion_obtenida"] == round(filas_de_prueba / 1000, 4)
    assert limite.medida["filas_de_prueba"] == filas_de_prueba and limite.medida["filas"] == 1000
    assert limite.medida["margen_relativo"] == 0.5
    assert "20" in limite.motivo["es"] and "suj" in limite.motivo["es"] and "suj" in limite.motivo["en"]
    # y dentro del margen no se declara nada
    r2 = _groups(_sujetos(40, lambda s: 5, lambda s, i: i % 2 == 0), seed=0, folds=3, test_fraction=0.2)
    assert not [x for x in r2.limites if x.clave == "fraccion_de_prueba_desviada"]


def test_I2_una_clase_que_solo_esta_en_la_prueba_tambien_cuenta_para_los_pliegues():
    """Auditoría I2 (sabotaje: no pasar `clases` a `elegir_ventana`, 223 verdes). 100 días; la clase «c» solo en los
    últimos, todos dentro de la prueba: ningún pliegue la tiene, y el estudio no arranca con pliegues que no la miden."""
    filas = [{"id": f"o{i:03d}", "t": float(i), "y": ("c" if i >= 85 and i % 2 else ("a" if i % 3 else "b"))}
             for i in range(100)]
    r = proponer_particion(filas, **_pide(split_type="temporal", seed=0, time_column="t", objetivo="y"))
    assert not r.es_viable
    (b,) = r.bloqueos
    assert b.clave == "bloques_sin_clase" and b.medida["clase"] == "c"
    assert b.medida["filas_de_la_clase_en_desarrollo"] == 0
    assert b.medida["filas_de_la_clase_por_bloque"] == [0, 0, 0]


def test_I3_cada_entrena_temporal_tiene_todas_las_clases():
    """Auditoría I3 (sabotaje: comprobar solo `valida`, 223 verdes). 120 días; «si» solo desde el día 40: el pliegue
    más antiguo entrenaría sin ningún «si». La regla (más estricta que la letra de D2) se le pregunta a Roberto; aquí
    se afirma que existe y que el motivo dice qué pliegue, sin qué clase y desde cuándo aparece."""
    filas = [{"id": f"o{i:03d}", "t": float(i), "y": ("si" if i >= 40 and i % 4 == 0 else "no")} for i in range(120)]
    r = proponer_particion(filas, **_pide(split_type="temporal", seed=0, time_column="t", objetivo="y"))
    assert not r.es_viable
    (b,) = r.bloqueos
    assert b.clave == "entrena_sin_clase" and b.campo == "objetivo"
    assert b.medida["clase"] == "si" and b.medida["pliegues"] == 2
    assert b.medida["pliegue"] == 1                    # con 2 pliegues, el 1 es el más antiguo
    assert b.medida["primera_aparicion"] == "40.0"     # el valor de la columna, como venía
    assert b.medida["filas_de_entrena"] == 32 and b.medida["filas_de_la_clase_en_entrena"] == 0
    for idioma in ("es", "en"):
        assert "si" in b.motivo[idioma] and "40.0" in b.motivo[idioma] and "t" in b.motivo[idioma]
    # con fechas, el instante se enseña como lo escribió la persona
    con_fecha = [dict(f, t=(BASE + timedelta(days=i)).strftime("%d/%m/%Y")) for i, f in enumerate(filas)]
    (b2,) = proponer_particion(con_fecha, **_pide(split_type="temporal", seed=0, time_column="t", objetivo="y")).bloqueos
    assert b2.clave == "entrena_sin_clase" and b2.medida["primera_aparicion"] == "10/02/2024"
    assert "10/02/2024" in b2.motivo["es"]


@pytest.mark.parametrize("valor_de,gap", [
    (lambda i: str(2000 + i), {"magnitud": 1, "unidad": "years"}),                                   # un año escrito como número
    (lambda i: (BASE + timedelta(days=i)).strftime("%Y%m%d"), {"magnitud": 7, "unidad": "days"}),   # AAAAMMDD en texto
    (lambda i: float(i), {"magnitud": 2, "unidad": "hours"}),                                         # un número de verdad
])
def test_I4_un_hueco_con_unidad_sobre_un_tiempo_numerico_se_rechaza_diciendo_que_es_un_numero(valor_de, gap):
    """Auditoría I4: el hueco se aplicaba como segundos y salía `cronologia_no_se_pudo_separar`, genérico; el aviso
    de tiempo numérico nunca llegaba (se calculaba después)."""
    filas = [{"id": _id(i), "fecha": valor_de(i), "y": "pos" if i % 4 == 0 else "neg"} for i in range(400)]
    r = _temporal(filas, folds=5, gap=gap)
    (b,) = r.bloqueos
    assert b.clave == "hueco_en_tiempo_numerico" and b.campo == "gap"
    assert b.medida["columna"] == "fecha" and b.medida["unidad_del_hueco"] == gap["unidad"]
    assert "fecha" in b.motivo["es"] and "número" in b.motivo["es"] and "AAAA-MM-DD" in b.motivo["es"]
    assert "number" in b.motivo["en"] and "YYYY-MM-DD" in b.motivo["en"]
    # sin hueco, el mismo tiempo numérico sigue funcionando como hoy: por orden
    sin = _temporal(filas, folds=5)
    assert sin.es_viable and _ids_de(sin, "test") == [_id(i) for i in range(320, 400)]


def _con_dos_grupos(objetivo):
    filas = [{"id": str(i), "u": "A" if i < 3 else "B", "y": "no", "v": float(i)} for i in range(30)]
    return proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="u", objetivo=objetivo))


@pytest.mark.parametrize("objetivo", [None, "y"])          # regresión, y clasificación con una sola clase
def test_I5_groups_sin_guardas_con_dos_grupos_no_da_un_desarrollo_vacio(objetivo):
    """Auditoría I5 (heredado de la 139): grupo A con 3 filas y B con 27 → las 30 a la prueba y pliegues 0/0, viable."""
    r = _con_dos_grupos(objetivo)
    assert not r.es_viable
    (b,) = r.bloqueos
    assert b.clave == "grupos_insuficientes_para_prueba_y_pliegues" and b.campo == "unit_id_field"
    assert b.medida == {"grupos": 2, "pliegues_pedidos": 5, "grupos_necesarios_con_los_pedidos": 6,
                        "grupos_necesarios_con_2_pliegues": 3}
    assert "u" in b.motivo["es"] and "2" in b.motivo["es"] and "3" in b.motivo["en"]


def test_I5_groups_sin_guardas_no_deja_menos_de_2_grupos_en_desarrollo_y_lo_dice():
    """Tres grupos de 2, 2 y 26 filas: barajados, la prueba se llevaba los pequeños y después el grande (el 100 %)."""
    filas = [{"id": f"o{i:02d}", "u": ("p1" if i < 2 else "p2" if i < 4 else "G")} for i in range(30)]
    vistos = 0
    for semilla in range(30):
        kw = _pide(split_type="groups", seed=semilla, unit_id_field="u", folds=3)
        antes, r = ANTES.proponer_particion(filas, **kw), proponer_particion(filas, **kw)
        dev_antes = {f["u"] for f in filas if antes.plan.assignments[f["id"]] == "development"}
        if len(dev_antes) >= 2 and not _desborda_la_prueba(antes, 0.2):
            assert _firma(r) == _firma(antes)           # lo que funcionaba, bit a bit
            continue
        if len(dev_antes) >= 2:                         # la 139 daba la prueba casi entera al grupo grande
            assert r.es_viable and (not _desborda_la_prueba(r, 0.2)
                                    or any(x.clave == "fraccion_de_prueba_desviada" for x in r.limites))
            continue
        vistos += 1
        assert r.es_viable
        assert len({f["u"] for f in filas if r.plan.assignments[f["id"]] == "development"}) == 2
        assert all(p.entrena and p.valida for p in r.pliegues.pliegues)
        (limite,) = [x for x in r.limites if x.clave == "prueba_recortada_para_dejar_grupos"]
        assert limite.medida["grupos"] == 3 and limite.medida["grupos_en_desarrollo"] == 2
    assert vistos > 0


def test_I5_groups_sin_objetivo_bit_a_bit_con_la_139_salvo_donde_la_139_era_inservible():
    """Oráculo congelado: en 600 formas aleatorias, lo que la 139 repartía bien sale igual; lo que cambia es solo
    lo que la 139 dejaba sin desarrollo, sin pliegues o sin prueba (o lo que lanzaba una excepción)."""
    cambiados = 0
    for c in range(600):
        rng = random.Random(c)
        n = rng.choice([3, 5, 10, 31, 100])
        filas = [{"id": f"{i:04d}", "u": f"u{rng.randrange(max(1, rng.choice([2, 3, n // 3 or 1])))}"} for i in range(n)]
        kw = _pide(split_type="groups", seed=rng.randrange(100), unit_id_field="u",
                   test_fraction=rng.choice([0.1, 0.2, 0.33, 0.5]), folds=rng.choice([2, 3, 5]), repeats=rng.choice([1, 2]))
        try:
            antes = ANTES.proponer_particion(filas, **kw)
        except Exception:                                      # noqa: BLE001 — la 139 lanzaba: ahora es un Bloqueo
            antes = None
        ahora = proponer_particion(filas, **kw)
        servible = (antes is not None and antes.es_viable and any(x == "test" for x in antes.plan.assignments.values())
                    and all(p.entrena and p.valida for p in antes.pliegues.pliegues))
        if servible and not _desborda_la_prueba(antes, kw["test_fraction"]):
            assert _firma(ahora) == _firma(antes), c
        elif servible:                                         # la 139 desbordaba la prueba: ahora cabe o se dice
            assert ahora.es_viable, c
            assert (not _desborda_la_prueba(ahora, kw["test_fraction"])
                    or any(x.clave == "fraccion_de_prueba_desviada" for x in ahora.limites)), c
        else:
            cambiados += 1
            assert not ahora.es_viable or all(p.entrena and p.valida for p in ahora.pliegues.pliegues), c
    assert cambiados > 0


def test_M1_una_unidad_vacia_es_un_bloqueo_con_motivo_no_una_excepcion():
    filas = [{"id": str(i), "u": ("" if i < 5 else " " if i < 7 else f"s{i % 6}"), "y": "a" if i % 2 else "b"}
             for i in range(60)]
    for objetivo in ("y", None):
        (b,) = proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="u", objetivo=objetivo)).bloqueos
        assert b.clave == "unidad_vacia" and b.medida == {"columna": "u", "filas_sin_unidad": 7, "filas": 60}
        assert "u" in b.motivo["es"] and "7" in b.motivo["en"] and "units." not in b.motivo["en"]


@pytest.mark.parametrize("objetivo", ["y", None])
def test_M1_groups_sin_prueba_reservada_es_un_bloqueo_con_motivo(objetivo):
    filas = [{"id": str(i), "u": f"s{i % 10}", "y": "a" if i % 2 else "b"} for i in range(60)]
    (b,) = proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="u", objetivo=objetivo,
                                             test_fraction=0)).bloqueos
    assert b.clave == "sin_prueba_reservada" and b.medida == {"fraccion_pedida": 0, "filas": 60}
    assert "evaluacion_anidada" not in b.motivo["es"] + b.motivo["en"]


def test_M1_iid_sin_prueba_reservada_tambien():
    filas = [{"id": str(i), "y": "a" if i % 2 else "b"} for i in range(60)]
    (b,) = proponer_particion(filas, **_pide(split_type="iid", seed=0, objetivo="y", test_fraction=0)).bloqueos
    assert b.clave == "sin_prueba_reservada"


@pytest.mark.parametrize("objetivo", ["y", None])
def test_M1_cohorte_externa_que_comparte_unidad_con_las_internas_es_un_bloqueo(objetivo):
    filas = [{"id": str(i), "u": f"s{i % 10}", "y": "a" if i % 2 else "b"} for i in range(60)]
    (b,) = proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="u", objetivo=objetivo,
                                             ids_cohorte_externa=["0", "1", "10"])).bloqueos
    assert b.clave == "cohorte_externa_comparte_unidad"
    assert b.medida == {"columna": "u", "unidades_compartidas": 2, "ejemplos": ["s0", "s1"]}
    assert "s0" in b.motivo["es"] and "s0" in b.motivo["en"]


def test_M2_la_frontera_del_hueco_es_la_misma_en_los_pliegues_y_en_la_prueba():
    """Una fila a EXACTAMENTE `gap` del inicio de lo que se valida no entrena: su etiqueta, que mira `gap` hacia
    delante, caería en ese inicio. La prueba reservada ya era estricta; los pliegues, no (`>=`)."""
    filas = _dias(100)
    r = _temporal(filas, folds=5, gap={"magnitud": 5, "unidad": "days"})
    dia = {f["id"]: BASE + timedelta(days=int(f["id"][1:])) for f in filas}
    for pl in r.pliegues.pliegues:
        inicio = min(dia[o] for o in pl.valida)
        assert inicio - max(dia[o] for o in pl.entrena) > timedelta(days=5)
        assert inicio - max(dia[o] for o in pl.entrena) == timedelta(days=6)    # el día a 5 justos, fuera
    ultimo_dev = max(dia[o] for o in _ids_de(r, "development"))
    primero_prueba = min(dia[o] for o in _ids_de(r, "test"))
    assert primero_prueba - ultimo_dev > timedelta(days=5)


def test_M3_los_mensajes_nombran_la_columna_no_el_rol():
    filas = _sujetos(10, lambda s: 4, lambda s, i: False)
    (limite,) = _groups(filas, folds=3).limites[:1]
    assert limite.clave == "una_sola_clase_sin_guardas" and "y" in limite.motivo["es"].split()[0]
    assert "objetivo" not in limite.motivo["es"] and "objetivo" not in limite.motivo["en"]
    t = _temporal(_dias(60, etiqueta=lambda i: "neg"), folds=5)
    assert t.limites[0].motivo["es"].startswith("y ")
    ids = [f"o{i}" for i in range(10)]
    m = partir_valida_en_mitades(ids, diseno="groups", unidad_por_id={o: "unica" for o in ids}, columna_de_unidad="paciente")
    assert "paciente" in m.limites[0].motivo["es"] and "unit_id_field" not in m.limites[0].motivo["es"]
    m2 = partir_valida_en_mitades(ids, diseno="temporal", tiempo_por_id={o: 5.0 for o in ids}, columna_de_tiempo="fecha")
    assert "fecha" in m2.limites[0].motivo["en"] and "time_column" not in m2.limites[0].motivo["en"]


def test_M1_groups_sin_guardas_con_una_fraccion_que_no_llega_a_una_fila_tambien_es_sin_prueba_reservada():
    """`test_fraction` > 0 pero tan pequeña que redondea a 0 filas (4 filas × 0,1): la 139 lanzaba `EsquemaInvalido`."""
    filas = [{"id": str(i), "u": f"s{i}"} for i in range(4)]
    (b,) = proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="u", test_fraction=0.1)).bloqueos
    assert b.clave == "sin_prueba_reservada" and b.medida == {"fraccion_pedida": 0.1, "filas": 4}
    assert "0.1" in b.motivo["es"] and "4" in b.motivo["en"]


@pytest.mark.parametrize("semilla", range(8))
def test_en_regresion_un_sujeto_enorme_no_se_lleva_la_prueba(semilla):
    """Supervisor, 27-10 (la ruta sin estratificar, que es la de la REGRESIÓN y la de Parkinson): un sujeto con el 60 %
    de las filas entraba en la prueba en 2 de 5 semillas y la dejaba en el 65–74 %, sin decir nada."""
    filas = [{"id": f"g{i}", "suj": "gigante"} for i in range(600)]
    filas += [{"id": f"r{i}", "suj": f"s{i % 40}"} for i in range(400)]
    r = proponer_particion(filas, **_pide(split_type="groups", seed=semilla, unit_id_field="suj", objetivo=None))
    assert r.es_viable
    assert r.plan.assignments["g0"] == "development"
    assert 0.1 <= _fraccion_de_prueba(r) <= 0.3


def test_en_regresion_si_ningun_sujeto_cabe_en_el_margen_se_dice():
    """Cinco sujetos de 20 filas y una prueba del 10 % (10 filas): ninguno cabe; va el más pequeño y se declara."""
    filas = [{"id": f"{s}_{i}", "suj": f"s{s}"} for s in range(5) for i in range(20)]
    r = proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="suj", objetivo=None,
                                          test_fraction=0.1, folds=2))
    assert r.es_viable and _fraccion_de_prueba(r) == 0.2
    (limite,) = [x for x in r.limites if x.clave == "fraccion_de_prueba_desviada"]
    assert limite.medida["fraccion_pedida"] == 0.1 and limite.medida["fraccion_obtenida"] == 0.2


# ---------------------------------------------------------------------------
# Reparación de la re-auditoría del C1 (28-10): cada prueba salió ROJA antes de su arreglo
# ---------------------------------------------------------------------------

def _forma_del_fuzz_estratificado(caso):
    """La forma `caso` del fuzz estratificado de la re-auditoría (`caso_regresion_fraccion.py`), tal cual."""
    rng = random.Random(50_000 + caso)
    n_suj = rng.choice([4, 6, 10, 20, 40, 80, 150])
    clases = rng.choice([["pos", "neg"], ["a", "b", "c"], ["raro", "comun"]])
    filas, k = [], 0
    for s in range(n_suj):
        t = rng.choice([1, 2, 3, 8, 20]) if rng.random() > 0.05 else rng.choice([100, 300])
        p_raro = rng.choice([0.0, 0.0, 0.05, 0.3, 1.0])
        for _ in range(t):
            if clases[0] == "raro":
                y = "raro" if rng.random() < p_raro else "comun"
            else:
                y = rng.choice(clases) if rng.random() < 0.7 else clases[s % len(clases)]
            filas.append({"id": f"o{k}", "suj": f"s{s}", "y": y})
            k += 1
    kw = dict(seed=rng.randrange(1000), test_fraction=rng.choice([0.1, 0.2, 0.3]),
              folds=rng.choice([2, 3, 5]), repeats=rng.choice([1, 2]))
    return filas, clases, kw


@pytest.mark.parametrize("caso,filas_de_prueba", [(90, 21), (591, 7), (408, 6)])
def test_reaudit_IMP1_la_ruta_estratificada_aplica_el_mismo_margen_que_la_sin_estratificar(caso, filas_de_prueba):
    """Re-auditoría, IMPORTANTE 1: `_cubrir` trataba cualquier desborde de la capacidad PELADA como peor que quedarse
    corto, y la prueba encogía donde el C1 acertaba: el caso 90 (90 filas, 20 % pedido) pasaba de 21 filas a 4, el 591
    de 7 a 3 y el 408 de 6 a 2, con `fraccion_de_prueba_desviada`. Con el margen (`MARGEN_RELATIVO_DE_LA_PRUEBA`,
    como en la otra ruta) salen dentro, sin límite, y con todas las clases en todas partes."""
    from matrixai.training.particion_por_diseno_grupos import fuera_del_margen
    filas, clases, kw = _forma_del_fuzz_estratificado(caso)
    r = _groups(filas, **kw)
    assert r.es_viable
    assert len(_ids_de(r, "test")) == filas_de_prueba
    assert not fuera_del_margen(filas_de_prueba, len(filas), kw["test_fraction"])
    assert not [x for x in r.limites if x.clave == "fraccion_de_prueba_desviada"]
    _ninguna_unidad_partida(filas, r)
    _todas_las_clases_en_todas_partes(filas, r, clases=tuple(sorted({f["y"] for f in filas})))


def _clase_que_el_hueco_quita():
    """120 días; «si» en los días 28–31 y desde el 40 cada 3 (`sonda_i3_con_hueco.py` de la re-auditoría)."""
    base = datetime(2024, 1, 1)
    return [{"id": f"o{i:03d}", "t": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
             "y": "si" if (28 <= i <= 31 or (i >= 40 and i % 3 == 0)) else "no"} for i in range(120)]


def test_reaudit_IMP2_si_es_el_hueco_el_que_deja_sin_la_clase_el_motivo_dice_el_hueco():
    """Re-auditoría, IMPORTANTE 2: con un hueco de 5 días, el pliegue más antiguo se queda sin «si» porque el HUECO le
    quita los días 28–30; el motivo decía «la primera fila con si está en 2024-01-29… empieza los datos más cerca»,
    una causa falsa y un remedio que no arregla nada. Sin hueco, el mismo reparto es viable."""
    filas = _clase_que_el_hueco_quita()
    kw = _pide(split_type="temporal", seed=0, time_column="t", objetivo="y", folds=2)
    sin = proponer_particion(filas, **kw)
    assert sin.es_viable
    (b,) = proponer_particion(filas, **kw, gap={"magnitud": 5, "unidad": "days"}).bloqueos
    assert b.clave == "entrena_sin_clase_por_el_hueco" and b.campo == "gap"
    assert b.medida["clase"] == "si" and b.medida["pliegue"] == 1 and b.medida["filas_de_la_clase_en_entrena"] == 0
    assert b.medida["filas_de_la_clase_quitadas_por_el_hueco"] == 3          # los días 28, 29 y 30 (29–31 de enero)
    assert b.medida["inicio_de_la_validacion"] == "2024-02-01"               # el día 31
    assert b.medida["ultima_fila_de_la_clase_antes_de_la_validacion"] == "2024-01-31"   # el día 30
    assert b.medida["hueco"] == "5 days"
    for idioma, hueco in (("es", "hueco"), ("en", "gap")):
        texto = b.motivo[idioma]
        assert hueco in texto and "5 days" in texto and "2024-02-01" in texto and "si" in texto
        assert "más cerca de" not in texto and "closer to" not in texto   # el remedio que no lo arregla, fuera
    # sin hueco en juego (la clase empieza tarde de verdad) sigue siendo `entrena_sin_clase` (I3)
    tarde = [{"id": f"o{i:03d}", "t": float(i), "y": ("si" if i >= 40 and i % 4 == 0 else "no")} for i in range(120)]
    (b2,) = proponer_particion(tarde, **_pide(split_type="temporal", seed=0, time_column="t", objetivo="y",
                                              gap={"magnitud": 1, "unidad": "seconds"})).bloqueos
    assert b2.clave == "entrena_sin_clase"


_FALTA = object()


@pytest.mark.parametrize("valor", ["NA", "N/A", "nan", "NULL", "None", "?", "-", "na", None, float("nan"), _FALTA],
                         ids=["NA", "N/A", "nan", "NULL", "None-texto", "?", "-", "na", "None", "NaN", "sin-columna"])
@pytest.mark.parametrize("diseno", ["groups", "groups_and_time"])
def test_reaudit_IMP3_una_unidad_que_el_nucleo_llama_faltante_es_unidad_vacia(valor, diseno):
    """Re-auditoría, IMPORTANTE 3: solo `""` contaba como vacía; «N/A», «nan» o `None` eran un sujeto más, sin aviso
    (y una fila SIN la columna, un `KeyError` del núcleo). Ahora es lo que el núcleo ya llama faltante (`_is_null`)."""
    filas = []
    for i in range(120):
        f = {"id": f"o{i}", "y": "a" if i % 2 else "b", "t": float(i)}
        if i >= 10:
            f["u"] = f"p{i % 12}"
        elif valor is not _FALTA:
            f["u"] = valor
        filas.append(f)
    for objetivo in ("y", None):
        kw = _pide(split_type=diseno, seed=0, unit_id_field="u", objetivo=objetivo, time_column="t")
        (b,) = proponer_particion(filas, **kw).bloqueos
        assert b.clave == "unidad_vacia" and b.medida == {"columna": "u", "filas_sin_unidad": 10, "filas": 120}


def test_reaudit_IMP3_la_lista_de_faltantes_es_la_del_nucleo_no_una_copia(monkeypatch):
    """Lo que decide es `dataset_analysis._is_null` (la misma con la que el núcleo da por faltante el objetivo): si
    el núcleo aprende una marca nueva, la unidad la aprende con él. Una copia de la lista no lo haría."""
    from matrixai.training import dataset_analysis
    filas = [{"id": f"o{i}", "u": ("desconocido" if i < 4 else f"p{i % 12}")} for i in range(60)]
    kw = _pide(split_type="groups", seed=0, unit_id_field="u")
    assert proponer_particion(filas, **kw).es_viable                  # «desconocido» no es faltante para el núcleo
    monkeypatch.setattr(dataset_analysis, "_NULL_TOKENS", dataset_analysis._NULL_TOKENS | {"desconocido"})
    (b,) = proponer_particion(filas, **kw).bloqueos
    assert b.clave == "unidad_vacia" and b.medida["filas_sin_unidad"] == 4


def test_reaudit_IMP3_una_unidad_de_verdad_que_se_parece_no_es_faltante():
    """`NA` es faltante; `NA1`, `0` o `Nadia` no: se decide por la celda entera, como en el objetivo."""
    filas = [{"id": f"o{i}", "u": ["NA1", "0", "Nadia", "nada", "x-1"][i % 5]} for i in range(60)]
    assert proponer_particion(filas, **_pide(split_type="groups", seed=0, unit_id_field="u")).es_viable


def _rejilla_en_el_borde_del_tope(tf):
    """Formas cuya prueba de la 139 cae justo entre el tope redondeado (`round(n·f)·1,5`) y el exacto (`n·f·1,5`):
    dos grupos de `floor(n·f·1,5)` filas y el resto de 1 a 3."""
    import math
    for n in range(10, 260):
        m = math.floor(n * tf * 1.5)
        if round(n * tf) == 0:
            continue
        for semilla in range(3):
            tam = [m, m]
            rng = random.Random(f"{tf}:{n}:{semilla}")
            while sum(tam) < n:
                tam.append(min(rng.choice([1, 2, 3]), n - sum(tam)))
            yield [{"id": f"o{s}_{i}", "u": f"S{s}"} for s, t in enumerate(tam) for i in range(t)], semilla


@pytest.mark.parametrize("tf", [0.1, 0.2, 0.25, 0.3])
def test_reaudit_Ma_groups_sin_objetivo_en_una_rejilla_solo_cambia_donde_la_139_se_salia_del_margen(tf):
    """Re-auditoría, M-a: el tope redondeado cambiaba el reparto donde la 139 estaba DENTRO del margen (n=1254, f=0,1:
    de 188 filas a 146). La prueba de 600 formas del I5 no lo cazaba; esta rejilla va a por el borde."""
    comparados = 0
    for filas, semilla in _rejilla_en_el_borde_del_tope(tf):
        kw = _pide(split_type="groups", seed=semilla, unit_id_field="u", test_fraction=tf, folds=3)
        antes = ANTES.proponer_particion(filas, **kw)
        if not (antes.es_viable and all(p.entrena and p.valida for p in antes.pliegues.pliegues)):
            continue
        if _desborda_la_prueba(antes, tf):
            continue
        comparados += 1
        assert _firma(proponer_particion(filas, **kw)) == _firma(antes), (len(filas), semilla)
    assert comparados > 30                     # que la rejilla compare de verdad (41 con 0,25; 46 con 0,3)


@pytest.mark.parametrize("caso", [498, 1056, 1351, 1582, 2190, 2194])
def test_reaudit_Ma_los_seis_casos_del_fuzz_de_la_reauditoria_salen_como_en_la_139(caso):
    """Los seis de `fuzz_groups_regresion_139.py` (entre ellos el de 1254 filas) donde la 139 estaba dentro."""
    rng = random.Random(7_000 + caso)
    n_suj = rng.choice([2, 3, 4, 5, 8, 20, 42, 100])
    tam = [rng.choice([1, 3, 10, 30, 140]) for _ in range(n_suj)]
    if rng.random() < 0.3:
        tam[0] = sum(tam) * rng.choice([0.5, 1, 2])
    filas = [{"id": f"o{s}_{i}", "u": (s if rng.random() < 0.3 else f"S{s}")} for s, t in enumerate(tam) for i in range(int(t))]
    kw = _pide(split_type="groups", seed=rng.randrange(1000), unit_id_field="u", test_fraction=rng.choice([0.1, 0.2, 0.25, 0.3]),
               folds=rng.choice([2, 3, 5]), repeats=rng.choice([1, 2]))
    antes = ANTES.proponer_particion(filas, **kw)
    assert antes.es_viable and not _desborda_la_prueba(antes, kw["test_fraction"])
    assert _firma(proponer_particion(filas, **kw)) == _firma(antes)


@pytest.mark.parametrize("diseno", ["groups", "groups_and_time"])
def test_reaudit_Mb_grupos_insuficientes_nombra_la_columna_y_lleva_sus_numeros(diseno):
    """Re-auditoría, M-b: decía «unit_id_field tiene 1 grupo(s)» y no llevaba números. (El llamante del backend de
    `partir_valida_en_mitades` aún no le pasa `columna_de_unidad`: es cableado del C2.)"""
    filas = [{"id": f"o{i}", "paciente": "P-07", "t": float(i), "y": "a" if i % 2 else "b"} for i in range(10)]
    (b,) = proponer_particion(filas, **_pide(split_type=diseno, seed=0, unit_id_field="paciente", time_column="t",
                                             objetivo="y")).bloqueos
    assert b.clave == "grupos_insuficientes" and b.campo == "unit_id_field"
    assert b.medida == {"columna": "paciente", "grupos": 1, "grupos_necesarios": 2, "filas": 10}
    for idioma in ("es", "en"):
        assert b.motivo[idioma].startswith("paciente ") and "unit_id_field" not in b.motivo[idioma]


def test_reaudit_Mc_las_clases_1_y_texto_1_no_se_escriben_igual():
    """Re-auditoría, M-c: `1` y `"1"` son dos clases, pero se nombraban igual: en `groups` las medidas perdían una
    clave (`{'0': 9, '1': 2}`) y en `temporal` el motivo decía «la clase 1 solo tiene 0 filas en desarrollo» con filas
    de `1` en desarrollo. No se juntan (son dos clases); solo se escriben distinto."""
    filas = [{"id": f"o{i}", "suj": f"p{i % 12}", "y": (1 if i % 3 == 0 else "1") if i % 4 == 0 else 0} for i in range(120)]
    (b,) = _groups(filas).bloqueos
    assert set(b.medida["grupos_por_clase"]) == {"0", "1", "'1'"}            # tres clases, tres claves
    t = [{"id": f"o{i}", "t": float(i), "y": ("1" if i > 100 else 1) if i % 4 == 0 else 0} for i in range(120)]
    (b2,) = proponer_particion(t, **_pide(split_type="temporal", seed=0, time_column="t", objetivo="y")).bloqueos
    assert b2.clave == "bloques_sin_clase" and b2.medida["clase"] == "'1'"   # la de texto, que es la que no está
    assert "la clase '1' solo tiene 0" in b2.motivo["es"]
    # sin choque de nombres, nada cambia: la clase se escribe como siempre
    from matrixai.training.particion_por_diseno_comun import nombres_de_clases
    assert nombres_de_clases([0, 1]) == {0: "0", 1: "1"} and nombres_de_clases(["si", "no"]) == {"si": "si", "no": "no"}
