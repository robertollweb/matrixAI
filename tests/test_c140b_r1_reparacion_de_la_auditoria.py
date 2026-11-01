# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b — la reparación de la auditoría Opus del reparto estricto (`~/.agentes/140b/auditoria/INFORME.md`), en el núcleo.

Cada prueba lleva el nombre del hallazgo que fija y salió ROJA con el código anterior a la reparación (comprobado revirtiendo
solo el arreglo). Los datos reproducen las sondas del auditor (`sondas/sonda_*.py`): las formas que trae la gente.

- I1: el alcance de `groups_and_time` y el diseño con que se remuestrea (el estudio real, en el backend).
- I2: la cuota de sujetos que se declara es la que HUBO (8 de 40), no la pedida (30 %).
- I3 y M4: el rechazo de la clase rara dice de qué población habla, y la escalera prueba otro corte si fallan los pliegues.
- I5: «7» y «007» se avisan con sus números, en `groups` y en el estricto (sin unificarlos).
- M1: el desarrollo tiene los mismos mínimos que la prueba. M3: el rechazo no aconseja un control que no existe. M5: la misma
  columna como sujeto y como tiempo se rechaza.
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_c140b_c1_particion_estricta import _bloqueo, _cohorte, _dia, _en_los_dos_idiomas   # noqa: E402
from test_c140b_c2b_manifiesto_estricto import _linea, _manifiesto, _paquete, _sobre_de   # noqa: E402

from matrixai.estudio.perfil_clinico import alcance_de_validacion, diseno_de_remuestreo   # noqa: E402
from matrixai.export.expediente_clinico import REPARTO_POR_DISENO   # noqa: E402
from matrixai.export.tripod import ficha_tripod   # noqa: E402
from matrixai.training.particion_por_diseno import proponer_particion   # noqa: E402
from matrixai.training.particion_por_diseno_comun import escritura_normalizada   # noqa: E402
from matrixai.training.particion_por_diseno_grupos_y_tiempo import (   # noqa: E402
    MINIMO_FILAS_DE_DESARROLLO, MINIMO_UNIDADES_DE_DESARROLLO, proponer_particion_estricta)


def _pide(filas, **kw):
    base = dict(plan_id="x", observation_id_field="rid", seed=0, unit_id_field="pac", time_column="t", objetivo="y")
    base.update(kw)
    return proponer_particion_estricta(filas, **base)


def _con_rid(filas):
    for i, f in enumerate(filas):
        f["rid"] = f"R{i}"
    return filas


# ---------------------------------------------------------------------------------------------------------------------------
# I1 · la PARTICIÓN que se declara y el diseño con que se REMUESTREA
# ---------------------------------------------------------------------------------------------------------------------------

def test_I1_groups_and_time_se_remuestrea_como_groups_y_su_alcance_es_validacion_temporal():
    """El perfil declara la partición (`groups_and_time`) y remuestrea con `diseno_de_remuestreo` (`groups`). Los otros tres
    diseños, el mismo. Con la partición declarada como `groups` (lo que hacía el estudio) el alcance salía `internal_only`."""
    assert {d: diseno_de_remuestreo(d) for d in ("iid", "groups", "temporal", "groups_and_time")} == {
        "iid": "iid", "groups": "groups", "temporal": "temporal", "groups_and_time": "groups"}
    assert alcance_de_validacion(evidencia="independent_test", diseno="groups_and_time") == "temporal"
    assert alcance_de_validacion(evidencia="independent_test", diseno="groups") == "internal_only"
    from matrixai.estudio.errores import ErrorDeEstudio
    with pytest.raises(ErrorDeEstudio):
        diseno_de_remuestreo("otro")


# ---------------------------------------------------------------------------------------------------------------------------
# I2 · la cuota de sujetos que HUBO
# ---------------------------------------------------------------------------------------------------------------------------

def _cohorte_con_pocos_elegibles():
    """La del auditor (`sonda_fraccion.py`, caso A): 40 sujetos, de los que solo 8 siguen hasta el final."""
    rng = random.Random(2)
    filas = []
    for k in range(40):
        largo = k < 8
        inicio = datetime(2023, 1, 1) + timedelta(days=rng.randrange(0, 30))
        for v in range(60 if largo else 12):
            filas.append({"pac": f"P{k}", "t": (inicio + timedelta(days=(5 if largo else 3) * v)).strftime("%Y-%m-%d"),
                          "y": "Sí" if rng.random() < .4 else "No"})
    return _con_rid(filas)


def test_I2_la_ficha_dice_los_sujetos_de_prueba_que_hubo_y_no_la_cuota_pedida_como_si_fuera_la_real(tmp_path):
    """Con 8 sujetos elegibles de 40 van todos a la prueba: el 20 %, no el 30 % pedido. La ficha decía «el 30 % de los sujetos»
    y, en la misma frase, «con 8 sujetos de prueba». (Mata el sabotaje VERDE M6: «la cuota pedida escrita como usada».)"""
    filas = _cohorte_con_pocos_elegibles()
    p = _pide(filas)
    r = p.reparto
    assert (len(r.unidades_de_prueba), r.sujetos, r.fraccion_de_sujetos) == (8, 40, 0.3)
    sobre = _sobre_de(p)
    s = _manifiesto(filas, p, sobre)["study"]["split"]["strict"]
    assert s["subject_fraction"] == {"requested": 0.3, "used": 0.3, "realized": 0.2} and s["subjects"] == 40
    paquete = _paquete(filas, p, sobre, tmp_path)
    es = _linea(ficha_tripod(paquete, locale="es"), REPARTO_POR_DISENO.rotulo["es"])
    en = _linea(ficha_tripod(paquete, locale="en"), REPARTO_POR_DISENO.rotulo["en"])
    assert "sujetos de prueba: 8 de 40 (el 20 %; se pidió el 30 %), con el corte en el cuantil 0,7 del tiempo" in es
    assert "test subjects: 8 of 40 (20%; 30% was requested), with the cut at the 0.7 quantile of time" in en
    assert "el 30 % de los sujetos" not in es and "30% of the subjects" not in en


def test_I2_el_limite_del_corte_relajado_dice_los_sujetos_que_tiene_la_prueba():
    filas = _cohorte(40, 7, entrada=30, tiempo="iso")
    for f in filas:
        f["pac"], f["rid"] = f["sujeto"], f["id"]
    p = _pide(filas)
    r = p.reparto
    [relajado] = [l for l in p.limites if l.clave == "estricto_corte_relajado"]
    n, m = len(r.unidades_de_prueba), r.sujetos
    assert f"la prueba tiene {n} de los {m} sujetos" in relajado.motivo["es"]
    assert f"the test has {n} of the {m} subjects" in relajado.motivo["en"]
    assert (relajado.medida["sujetos_de_prueba"], relajado.medida["sujetos"]) == (n, m)


# ---------------------------------------------------------------------------------------------------------------------------
# I3 y M4 · la clase rara
# ---------------------------------------------------------------------------------------------------------------------------

def _clase_en(n_con_clase, seed=7, n=40):
    """La del auditor (`sonda_clase.py`): la clase «Sí» en TODAS las visitas de los `n_con_clase` primeros pacientes."""
    rng = random.Random(seed)
    filas = []
    for k in range(n):
        ini = datetime(2023, 1, 1) + timedelta(days=rng.randrange(0, 200))
        for v in range(rng.randint(5, 20)):
            filas.append({"pac": f"P{k}", "t": (ini + timedelta(days=7 * v)).strftime("%d/%m/%Y"),
                          "y": "Sí" if k < n_con_clase else "No"})
    rng.shuffle(filas)
    return _con_rid(filas)


def test_I3_el_rechazo_de_los_pliegues_dice_desarrollo_conjunto_y_prueba_sin_hablar_de_la_prueba_reservada():
    """3 de 40 pacientes traen la clase: la prueba se lleva 2 (cumple su mínimo) y al desarrollo le queda 1, con el que no se
    pueden hacer pliegues. El rechazo salía del asignador de `groups` tal cual: «la clase Sí solo aparece en 1 grupo(s)… con 5
    pliegues y la prueba reservada…», sin decir que el 1 es del DESARROLLO."""
    b = _bloqueo(_pide(_clase_en(3)))
    assert b.clave == "estricto_pliegues_sin_la_clase"
    m = b.medida
    assert (m["sujetos_de_desarrollo_con_la_clase"], m["sujetos_del_conjunto_con_la_clase"],
            m["sujetos_de_prueba_con_la_clase"], m["minimo_en_el_desarrollo"]) == (1, 3, 2, 2)
    assert m["rechazo_de_los_pliegues"] == "clase_sin_grupos_suficientes" and m["cortes_probados"] > 1
    assert "en el desarrollo la clase Sí queda en 1 valor(es) de pac" in b.motivo["es"]
    assert "en todo el conjunto la traen 3 y la prueba se lleva 2" in b.motivo["es"]
    assert "in development class Sí is left in 1 value(s) of pac" in b.motivo["en"]
    assert "the whole dataset has it in 3 and the test takes 2" in b.motivo["en"]
    for idioma, frase in (("es", "prueba reservada"), ("en", "reserved test"), ("es", "grupo(s)"), ("en", "group(s)")):
        assert frase not in b.motivo[idioma]


def test_I3_si_fallan_los_pliegues_la_escalera_prueba_el_corte_siguiente():
    """Con la clase en 4 pacientes (semilla 33), el corte pedido deja al desarrollo sin la clase repartible y el tercero sí
    puede. Antes la escalera solo bajaba un peldaño por los mínimos de la PRUEBA: con los pliegues, rechazaba en el primero."""
    filas = _clase_en(4, seed=33)
    solo_lo_pedido = _bloqueo(_pide(filas, automatico=False))
    assert solo_lo_pedido.clave == "estricto_pliegues_sin_la_clase"
    p = _pide(filas)
    assert p.es_viable, [b.motivo["es"] for b in p.bloqueos]
    assert p.reparto.cortes_probados == 3 and (p.reparto.fraccion_de_sujetos, p.reparto.cuantil) == (0.3, 0.6)
    assert any(l.clave == "estricto_corte_relajado" for l in p.limites)


def test_M4_el_rechazo_por_la_clase_nombra_cada_poblacion_con_su_cifra():
    """La clase «rara» la traen 3 sujetos en el conjunto, pero DESPUÉS del corte solo 1: la prueba no puede tenerla en 2.
    La frase decía «estaría en 1 … de la prueba» y «entre los sujetos con filas posteriores, 1 la traen» con LA MISMA cuenta
    (los elegibles), y no decía cuántos la traen en total."""
    filas = _cohorte(30, 20, entrada=0, barajar=False)
    for f in filas:
        i, tarde = int(f["sujeto"]), _dia(f["t"]) >= 70
        f["y"] = "rara" if (i == 0 or (i in (1, 2) and not tarde)) else ("si" if i % 2 else "no")
        f["pac"], f["rid"] = f["sujeto"], f["id"]
    b = _bloqueo(_pide(filas, automatico=False))
    assert b.clave == "estricto_clase_en_pocos_sujetos_de_prueba"
    m = b.medida
    assert m["sujetos_de_prueba_por_clase"]["rara"] == 1 and m["sujetos_elegibles_por_clase"]["rara"] == 1
    assert m["sujetos_del_conjunto_por_clase"]["rara"] == 3
    assert ("De los sujetos con filas posteriores al corte (los únicos que pueden ser de prueba), 1 la traen en esas filas; en "
            "todo el conjunto, 3") in b.motivo["es"]
    assert "Of the subjects with rows after the cut (the only ones that can be in the test), 1 have it in those rows; in the " \
           "whole dataset, 3" in b.motivo["en"]


# ---------------------------------------------------------------------------------------------------------------------------
# I5 · «7» y «007»
# ---------------------------------------------------------------------------------------------------------------------------

def test_I5_la_escritura_normalizada_quita_ceros_a_la_izquierda_espacios_y_mayusculas():
    assert escritura_normalizada("007") == escritura_normalizada("7") == "7"
    # (re-auditoría N6: los guiones también se quitan, así que «P-007» se escribe como «P7»)
    assert escritura_normalizada("P-007") == escritura_normalizada("p-7") == escritura_normalizada(" P - 7 ") == "p7"
    assert escritura_normalizada("0") == "0" and escritura_normalizada("70") == "70"
    assert escritura_normalizada("P7") != escritura_normalizada("P70")


def _con_el_7_escrito_de_dos_maneras(seed=1):
    """La del auditor (`sonda_ceros.py`): el paciente 7 con las visitas alternadas como «7» y «007» (Excel al juntar hojas)."""
    rng = random.Random(seed)
    filas = []
    for k in range(40):
        ini = datetime(2023, 1, 1) + timedelta(days=rng.randrange(0, 200))
        for v in range(rng.randint(5, 20)):
            pid = f"P{k}" if k != 7 else ("007" if v % 2 else "7")
            filas.append({"pac": pid, "t": (ini + timedelta(days=7 * v)).strftime("%d/%m/%Y"),
                          "y": "Sí" if rng.random() < .4 else "No"})
    return _con_rid(filas)


@pytest.mark.parametrize("diseno", ["groups", "groups_and_time"])
def test_I5_dos_ids_que_se_escriben_igual_se_avisan_con_sus_numeros_y_no_se_unifican(diseno):
    filas = _con_el_7_escrito_de_dos_maneras()
    filas_del_7 = sum(1 for f in filas if f["pac"] in ("7", "007"))
    if diseno == "groups":
        p = proponer_particion(filas, plan_id="x", observation_id_field="rid", split_type="groups", seed=0,
                               unit_id_field="pac", objetivo="y")
    else:
        p = _pide(filas)
    assert p.es_viable
    [aviso] = [l for l in p.limites if l.clave == "sujetos_que_se_escriben_igual"]
    assert aviso.medida["grupos"] == 1 and aviso.medida["valores"] == 2 and aviso.medida["filas"] == filas_del_7
    assert aviso.medida["ejemplos"] == [{"valores": ["007", "7"], "filas": filas_del_7}]
    assert f"«007» = «7» ({filas_del_7} filas)" in aviso.motivo["es"] and "No se unifican solos" in aviso.motivo["es"]
    assert f"“007” = “7” ({filas_del_7} rows)" in aviso.motivo["en"] and "not merged automatically" in aviso.motivo["en"]
    # NO se unifican: siguen siendo dos sujetos del plan
    assert {"7", "007"} <= set(p.plan.units.values())


@pytest.mark.parametrize("diseno", ["groups", "groups_and_time"])
def test_I5_sin_ids_que_se_escriban_igual_no_hay_aviso(diseno):
    """El control: los mismos datos con el 7 escrito siempre igual (y todos con ceros, `P07`…): ningún aviso."""
    filas = _con_el_7_escrito_de_dos_maneras()
    for f in filas:
        f["pac"] = "P07" if f["pac"] in ("7", "007") else f["pac"]
    if diseno == "groups":
        p = proponer_particion(filas, plan_id="x", observation_id_field="rid", split_type="groups", seed=0,
                               unit_id_field="pac", objetivo="y")
    else:
        p = _pide(filas)
    assert p.es_viable and not [l for l in p.limites if l.clave == "sujetos_que_se_escriben_igual"]


# ---------------------------------------------------------------------------------------------------------------------------
# M1 · el desarrollo, con los mínimos de la prueba
# ---------------------------------------------------------------------------------------------------------------------------

def _cohorte_de_regresion(n, seed, visitas=(15, 30)):
    """La del auditor (`sonda_f1_regresion.py`)."""
    rng = random.Random(seed)
    filas = []
    for k in range(n):
        ini = datetime(2023, 1, 1) + timedelta(days=rng.randrange(0, 60))
        for v in range(rng.randint(*visitas)):
            filas.append({"pac": f"P{k}", "t": (ini + timedelta(days=3 * v)).strftime("%Y-%m-%d"), "y": str(rng.random())})
    return _con_rid(filas)


def test_M1_un_desarrollo_mas_pequeno_que_los_minimos_de_la_prueba_se_rechaza_con_sus_numeros():
    """8 sujetos: la escalera daba un desarrollo de 3 sujetos y 38 filas frente a una prueba de 5 y 60, con pliegues de un solo
    sujeto. Ahora el desarrollo pide lo mismo que la prueba (≥ 5 sujetos y ≥ 50 filas) y, si ningún corte llega, se rechaza."""
    assert (MINIMO_UNIDADES_DE_DESARROLLO, MINIMO_FILAS_DE_DESARROLLO) == (5, 50)
    b = _bloqueo(_pide(_cohorte_de_regresion(8, 2), objetivo=None))
    assert b.clave == "estricto_desarrollo_vacio"
    assert (b.medida["minimo"], b.medida["minimo_de_filas"]) == (5, 50) and b.medida["unidades_de_desarrollo"] < 5
    _en_los_dos_idiomas(b, "5", "50", b.medida["filas_de_desarrollo"])
    assert "los mismos mínimos que la prueba" in b.motivo["es"] and "the same minimums as the test" in b.motivo["en"]


@pytest.mark.parametrize("n,seed", [(10, 1), (12, 3), (20, 4), (40, 5)])
def test_M1_todo_reparto_viable_tiene_un_desarrollo_con_los_minimos(n, seed):
    p = _pide(_cohorte_de_regresion(n, seed, visitas=(3, 30)), objetivo=None)
    if p.es_viable:
        assert len(p.reparto.unidades_de_desarrollo) >= 5 and len(p.reparto.desarrollo) >= 50
    else:
        assert p.bloqueos[0].clave.startswith("estricto_")


# ---------------------------------------------------------------------------------------------------------------------------
# M3 y M5
# ---------------------------------------------------------------------------------------------------------------------------

def test_M3_el_rechazo_sin_filas_posteriores_aconseja_lo_que_la_pantalla_permite():
    """No hay control del corte (lo elige el cuantil): la frase decía «Mueve el corte hacia atrás». Lo que sí se puede es el
    hueco."""
    filas = _cohorte(30, 20, entrada=0)
    for f in filas:
        f["pac"], f["rid"] = f["sujeto"], f["id"]
    from matrixai.estudio import Horizonte
    b = _bloqueo(_pide(filas, gap=Horizonte(magnitud=400, unidad="days")))
    assert b.clave == "estricto_sin_sujetos_con_filas_posteriores"
    assert "Mueve el corte" not in b.motivo["es"] and "Move the cut" not in b.motivo["en"]
    assert "lo que se puede cambiar es el hueco, más pequeño o ninguno" in b.motivo["es"]
    assert "what can be changed is the gap, smaller or none" in b.motivo["en"]


def test_M5_la_misma_columna_como_sujeto_y_como_tiempo_se_rechaza_con_su_motivo():
    """Con «fecha» como sujeto y como tiempo, cada instante era un sujeto: un reparto puramente temporal, viable, con 0 % de
    filas sin rol, y una ficha que decía «ningún sujeto estuvo a los dos lados»."""
    filas = _cohorte(30, 20, entrada=300)
    for f in filas:
        f["rid"] = f["id"]
    b = _bloqueo(_pide(filas, unit_id_field="t", time_column="t"))
    assert b.clave == "estricto_sujeto_y_tiempo_la_misma_columna" and b.campo == "unit_id_field"
    assert b.medida == {"columna_del_sujeto": "t", "columna_de_tiempo": "t"}
    _en_los_dos_idiomas(b, "t")


# ---------------------------------------------------------------------------------------------------------------------------
# I4 · el intervalo de Wilson cuenta FILAS: con sujetos repetidos se dice, y no se emite el veredicto
# ---------------------------------------------------------------------------------------------------------------------------

def _regresion_con_unidades(unidades):
    """50 filas de regresión con residuos conocidos; `unidades` = el sujeto de cada fila (o `None`)."""
    from matrixai.estudio.metricas import Muestra
    rng = random.Random(4)
    y = [rng.gauss(0, 1) for _ in range(50)]
    return Muestra.regresion(y, [v + rng.gauss(0, 1) for v in y], unidades=unidades)


def test_I4_la_cobertura_de_un_reparto_por_sujeto_corrige_su_intervalo_por_sujeto_y_da_veredicto():
    """13 sujetos para 50 filas: el «intervalo del 95 %» por filas cubre de verdad 0,84 (o 0,61) por sujeto (sonda del auditor).
    Desde la decisión de Roberto del 01-11 (Kish) el intervalo se CORRIGE por sujeto, se dice cuánto acierta así y el
    veredicto vuelve; sin `units`, y con un sujeto por fila, el JSON de siempre (re-auditoría N5: ni una clave más)."""
    from matrixai.estudio.conforme import calibrar, medir_cobertura
    calibracion = calibrar(_regresion_con_unidades(None), alfa=0.1)
    sin = medir_cobertura(_regresion_con_unidades(None), calibracion)
    con = medir_cobertura(_regresion_con_unidades([f"P{i % 13}" for i in range(50)]), calibracion)
    assert (con.cubiertas, con.n) == (sin.cubiertas, sin.n)
    assert con.por_sujetos is not None and con.por_sujetos.intervalo_por_filas == sin.intervalo_95
    assert con.intervalo_95 == con.por_sujetos.intervalo_95
    assert sin.compatible_con_la_nominal is not None and con.compatible_con_la_nominal is not None
    j = con.a_json()
    assert j["unidades"] == 13 and j["por_sujetos"]["unidades"] == 13 and "intervalo_por_filas" not in j
    assert "intervalo del 95 % corregido por sujeto: las 50 filas son de 13 sujetos" in j["por_sujetos"]["texto"]["es"]
    assert "95% interval corrected by subject: the 50 rows come from 13 subjects" in j["por_sujetos"]["texto"]["en"]
    claves = {"cobertura_nominal", "n", "cubiertas", "cobertura_observada", "intervalo_95", "compatible_con_la_nominal",
              "tamano_medio", "vacios", "garantia"}
    assert set(sin.a_json()) == claves
    uno_por_fila = medir_cobertura(_regresion_con_unidades([f"P{i}" for i in range(50)]), calibracion)
    assert uno_por_fila.por_sujetos is None and uno_por_fila.a_json() == sin.a_json()
