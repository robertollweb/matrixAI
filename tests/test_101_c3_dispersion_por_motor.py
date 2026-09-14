# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""La DISPERSIÓN por motor, que el protocolo exigía desde el primer día y no
calculaba nadie — y lo que cambia al leer el veredicto con ella delante.

**EL HUECO.** `protocolo_exploratorio.json` registra, en
`metricas_por_tarea.siempre`, que en toda pasada se mide
`dispersion_entre_semillas`. Medido el 2026-09-14: no había rastro de
`dispersion` **ni en `pasada_exploratoria_101_c3.py` ni en el artefacto**. El
veredicto se leía solo con medias, y dos motores con la misma media y
dispersiones muy distintas no son el mismo resultado. Un campo prerregistrado
que nadie calcula es una promesa, no una medida.

**LO MEDIDO sobre la evidencia real** (`..._conforme_20260914.json`, 1.260
intentos, 12 datasets x 7 motores x 5 pliegues x 3 repeticiones), en puntos
porcentuales de AUROC:

| motor | sd mediana | sd máxima | rango máximo | datasets con rango > 2,0 |
|---|---|---|---|---|
| catboost | 0,531 | 2,728 | 8,204 (kc2) | 6 de 12 |
| sklearn.lineal | 0,543 | 4,193 | 12,679 (kc2) | 5 de 12 |
| lightgbm | 0,738 | 3,361 | 9,935 (pc1) | 8 de 12 |
| xgboost | 0,754 | 3,005 | 9,612 (pc1) | 7 de 12 |
| matrixai.dense.torch_cpu | 0,770 | 3,888 | 13,244 (climate) | 7 de 12 |
| sklearn.hgb | 0,858 | 4,100 | 11,521 (pc1) | 8 de 12 |

**QUÉ CAMBIA AL LEER EL VEREDICTO CON ESTO DELANTE.** En **6 de 12** datasets
la ordenación primero/segundo se discute, y en **4** el mejor POR MEDIA pierde
la mayoría de los pliegues emparejados contra el segundo:

* `wilt`: catboost gana a xgboost por **0,007** puntos y pierde **9 de 15**.
* `sick`: catboost gana a lightgbm por **0,014** y pierde **9 de 15**.
* `PhishingWebsites`: lightgbm gana a sklearn.hgb por **0,003** y pierde 8/15.
* `diabetes`: la densa gana a catboost por **0,017** y pierde 8/15.

Dos de esos cuatro son victorias de catboost, que es **el único motor que
cumple la regla** (12/12). No cambian su veredicto —la distancia sigue siendo
0,0 y el listón son 2,0 puntos— pero sí cambian qué significa su recuento de
`aciertos_por_ser_el_mejor`: 2 de sus 5 primeros puestos los decide una
diferencia de milésimas de punto entre motores que se reparten los pliegues.

**Y la ordenación que de verdad se discute, medida re-aplicando la regla
registrada a cada semilla por separado** (5 pliegues en vez de 15, o sea una
comprobación de sensibilidad, no un segundo veredicto): `lightgbm`,
`sklearn.hgb` y `xgboost` **cambian de lado del listón según la semilla**
(cumplen con la semilla 0, no cumplen con la 2), y solo `catboost` cumple con
las tres. Está en `test_el_lado_del_liston_de_TRES_motores_depende_de_la_semilla`.

**QUÉ NO SE TOCA.** La evidencia commiteada no se reescribe: los JSON
anteriores siguen sin el bloque y siguen cuadrando con su propio digest. Lo
traerán las pasadas siguientes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
_FASE0 = _RAIZ / "benchmarks" / "fase0"
if str(_FASE0) not in sys.path:
    sys.path.insert(0, str(_FASE0))

from benchmarks.fase0.protocolo import (  # noqa: E402
    ProtocoloExploratorio, aplicar_regla_de_cierre, dispersion_de_un_motor,
    estabilidad_del_ganador)

RUTA_EVIDENCIA = _FASE0 / "pasada_exploratoria_101_c3_conforme_20260914.json"
RUTA_PROTOCOLO = _FASE0 / "protocolo_exploratorio.json"

EVIDENCIA = json.loads(RUTA_EVIDENCIA.read_text(encoding="utf-8"))
RESULTADOS = EVIDENCIA["resultados"]
PROTOCOLO = ProtocoloExploratorio.cargar(RUTA_PROTOCOLO)


def test_el_protocolo_EXIGE_la_dispersion_entre_semillas():
    """De dónde sale la obligación: no es una idea mía, está registrada con
    hash desde el 2026-09-06 y llevaba ocho días sin calcularse."""
    assert "dispersion_entre_semillas" in PROTOCOLO.metricas_por_tarea["siempre"]


# --------------------------------------------------------------------------
# La dispersión, medida sobre la evidencia real
# --------------------------------------------------------------------------

@pytest.mark.parametrize("motor, sd_mediana, rango_maximo, peor, sobre_el_liston", [
    ("catboost", 0.531, 8.204, "kc2", 6),
    ("sklearn.lineal", 0.543, 12.679, "kc2", 5),
    ("lightgbm", 0.738, 9.935, "pc1", 8),
    ("xgboost", 0.754, 9.612, "pc1", 7),
    ("matrixai.dense.torch_cpu", 0.770, 13.244, "climate-model-simulation-crashes", 7),
    ("sklearn.hgb", 0.858, 11.521, "pc1", 8),
])
def test_la_dispersion_de_cada_motor_sobre_la_evidencia(
        motor, sd_mediana, rango_maximo, peor, sobre_el_liston):
    """Los números de la tabla del encabezado, contra la evidencia de verdad.
    Si un cambio en el cálculo los mueve, esto lo dice por su nombre."""
    d = dispersion_de_un_motor(RESULTADOS, motor=motor, liston_en_puntos=2.0)
    assert d["n_datasets"] == 12
    assert d["sd_mediana"] == pytest.approx(sd_mediana, abs=5e-4)
    assert d["rango_maximo"] == pytest.approx(rango_maximo, abs=5e-4)
    assert d["dataset_mas_disperso"] == peor
    assert d["datasets_con_rango_mayor_que_el_liston"] == sobre_el_liston


def test_la_dispersion_ENTRE_SEMILLAS_senala_a_la_densa():
    """La que el protocolo pide por su nombre, y el motor que describe: la
    densa mueve 4,3 puntos de AUROC entre semillas en `climate-model`, y los
    otros cinco no llegan a 2,1 en ninguno de sus doce."""
    densa = dispersion_de_un_motor(RESULTADOS, motor="matrixai.dense.torch_cpu")
    assert densa["sd_entre_semillas_maxima"] == pytest.approx(4.335, abs=5e-4)
    otros = [dispersion_de_un_motor(RESULTADOS, motor=m)["sd_entre_semillas_maxima"]
             for m in ("catboost", "lightgbm", "xgboost", "sklearn.hgb", "sklearn.lineal")]
    assert max(otros) < densa["sd_entre_semillas_maxima"]
    assert max(otros) == pytest.approx(2.062, abs=5e-4)


def test_con_UNA_sola_medida_la_dispersion_es_AUSENTE_y_no_cero():
    """Un valor ausente no es un cero. Devolver `0.0` con una sola medida
    diría «este motor no varía», que es lo contrario de «no se sabe»."""
    uno = [r for r in RESULTADOS
           if r["motor"] == "catboost" and r["dataset"] == "kc2"
           and (r["repeticion"], r["pliegue"]) == (0, 0)]
    assert len(uno) == 1
    d = dispersion_de_un_motor(uno, motor="catboost")
    assert d["por_dataset"]["kc2"]["n_medidas"] == 1
    assert d["por_dataset"]["kc2"]["sd"] is None
    assert d["por_dataset"]["kc2"]["media"] is not None
    # Y la frase que lo cuenta tampoco puede escribir un 0,000 tranquilizador
    # ni reventar al formatear el ausente (que es lo que hacía: lo cazó esta
    # misma prueba).
    assert "no medible" in d["como_hay_que_leer_este_numero"]


def test_sin_ninguna_medida_lo_DICE_en_vez_de_dar_una_tabla_vacia():
    """La otra mitad del aserto negativo: un bloque vacío tiene que decir que
    está vacío y por qué, no parecer una medición que salió plana."""
    d = dispersion_de_un_motor(RESULTADOS, motor="motor-que-no-existe")
    assert d["n_datasets"] == 0
    assert d["por_dataset"] == {}
    assert "NO es dispersion cero" in d["como_hay_que_leer_este_numero"]


# --------------------------------------------------------------------------
# Lo que la dispersión dice del VEREDICTO
# --------------------------------------------------------------------------

def test_en_CUATRO_datasets_el_mejor_por_media_PIERDE_la_mayoria_de_pliegues():
    """El hallazgo. «Ganó» puede significar «ganó el promedio»: en `wilt` y
    `sick` catboost gana por milésimas de punto y pierde 9 de los 15 pliegues
    emparejados contra el segundo."""
    e = estabilidad_del_ganador(RESULTADOS)
    assert e["datasets_en_los_que_el_mejor_por_media_PIERDE_la_mayoria_de_pliegues"] == [
        "PhishingWebsites", "diabetes", "sick", "wilt"]
    assert e["datasets_en_los_que_la_ordenacion_se_discute"] == [
        "PhishingWebsites", "diabetes", "kc2", "ozone-level-8hr", "sick", "wilt"]


@pytest.mark.parametrize("dataset, primero, segundo, ventaja, gana", [
    ("wilt", "catboost", "xgboost", 0.007, 6),
    ("sick", "catboost", "lightgbm", 0.014, 6),
    ("PhishingWebsites", "lightgbm", "sklearn.hgb", 0.003, 7),
    ("diabetes", "matrixai.dense.torch_cpu", "catboost", 0.017, 7),
])
def test_las_cuatro_victorias_que_decide_el_promedio(dataset, primero, segundo,
                                                     ventaja, gana):
    e = {d["dataset"]: d for d in estabilidad_del_ganador(RESULTADOS)["detalle"]}
    d = e[dataset]
    assert (d["mejor_por_media"], d["segundo"]) == (primero, segundo)
    assert d["ventaja_en_puntos"] == pytest.approx(ventaja, abs=5e-4)
    assert d["pliegues_emparejados"] == 15
    assert d["pliegues_que_le_gana_al_segundo"] == gana
    assert d["se_discute"] is True


def test_una_victoria_que_NO_se_discute_para_que_el_aserto_de_arriba_valga():
    """La otra mitad: si `se_discute` saliera verdad siempre, la prueba de
    arriba pasaría sin medir nada. `Internet-Advertisements` lo gana
    sklearn.lineal en los 15 pliegues, y el ganador no cambia con la semilla."""
    e = {d["dataset"]: d for d in estabilidad_del_ganador(RESULTADOS)["detalle"]}
    d = e["Internet-Advertisements"]
    assert d["mejor_por_media"] == "sklearn.lineal"
    assert d["pliegues_que_le_gana_al_segundo"] == 15
    assert d["el_ganador_cambia_con_la_semilla"] is False
    assert d["se_discute"] is False


def test_el_lado_del_liston_de_TRES_motores_depende_de_la_semilla():
    """**La ordenación que de verdad se discute.** Re-aplicando la MISMA regla
    registrada a cada semilla por separado —5 pliegues en vez de 15, así que
    es una comprobación de sensibilidad y no un segundo veredicto—,
    `lightgbm`, `sklearn.hgb` y `xgboost` cambian de lado del listón según con
    qué semilla se mire. `catboost` cumple con las tres.

    Leído con la media sola, el veredicto publicado dice «catboost cumple, los
    otros cinco no» y los cinco se leen igual. No son iguales: tres de ellos
    están en el borde y dos (la densa, 4/12, y sklearn.lineal, 7/12) no.
    """
    por_semilla = {}
    for motor in ("catboost", "lightgbm", "sklearn.hgb", "xgboost",
                  "sklearn.lineal", "matrixai.dense.torch_cpu"):
        lados = []
        for semilla in (0, 1, 2):
            subconjunto = [r for r in RESULTADOS if r["repeticion"] == semilla]
            lados.append(aplicar_regla_de_cierre(
                subconjunto, PROTOCOLO.regla_de_cierre, motor=motor)["cumple_la_regla"])
        por_semilla[motor] = lados

    cambian = sorted(m for m, l in por_semilla.items() if len(set(l)) > 1)
    assert cambian == ["lightgbm", "sklearn.hgb", "xgboost"]
    assert por_semilla["catboost"] == [True, True, True]
    assert por_semilla["matrixai.dense.torch_cpu"] == [False, False, False]
    # Y con las tres semillas juntas —el veredicto publicado— solo cumple una.
    completo = {m: aplicar_regla_de_cierre(
        RESULTADOS, PROTOCOLO.regla_de_cierre, motor=m)["cumple_la_regla"]
        for m in por_semilla}
    assert [m for m, v in completo.items() if v] == ["catboost"]


# --------------------------------------------------------------------------
# Que la pasada lo ESCRIBA, y que la evidencia vieja no se toque
# --------------------------------------------------------------------------

def test_la_pasada_escribe_la_dispersion_y_la_estabilidad():
    """Probar la función no es probar el producto: esto comprueba que el
    bloque sale del camino que corre de verdad, no solo de llamar a mano."""
    from benchmarks.fase0 import pasada_exploratoria_101_c3 as pasada

    bloque = pasada._alcance_y_veredicto(RESULTADOS)
    assert "estabilidad_del_ganador" in bloque
    assert bloque["estabilidad_del_ganador"]["n_datasets"] == 12
    for motor, veredicto in bloque["por_motor"].items():
        assert "dispersion" in veredicto, motor
        assert veredicto["dispersion"]["n_datasets"] == 12, motor
        assert veredicto["dispersion"]["sd_mediana"] is not None, motor
        # Y en la frase que lee el que pasa por encima, no solo en un campo
        # que hay que ir a buscar.
        assert "desviacion tipica" in veredicto["como_hay_que_leer_este_numero"], motor


def test_la_evidencia_YA_COMMITEADA_no_se_reescribe():
    """El bloque nuevo es para las pasadas siguientes. La evidencia del
    2026-09-14 no lo trae, y sigue cuadrando con su propio digest: añadir una
    medida no puede cambiar lo que ya se midió."""
    from matrixai.estudio.validacion import digest_canonico

    payload = dict(EVIDENCIA)
    guardado = payload.pop("digest_resultados_crudos")
    assert digest_canonico(payload) == guardado
    assert "estabilidad_del_ganador" not in EVIDENCIA["alcance_y_veredicto"]
    assert all("dispersion" not in v
               for v in EVIDENCIA["alcance_y_veredicto"]["por_motor"].values())
