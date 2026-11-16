# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""164 · reparación de la auditoría 3 (rep3), en el núcleo.

- M-3: una serie de OTRA FRECUENCIA (una diaria entre tres horarias) se pierde entera: se cuenta entre las series sin filas
  completas y su corte se marca `por_su_paso`, para que quien lo redacte no lo llame «un corte o un salto». Un corte de verdad
  (mismo paso, un hueco de días) NO se marca.
- M-2: con un hueco PEDIDO que quita muchas filas, la declaración trae el más ancho que las conservaría, con sus números.
- I-3: el contexto de predicción dice el paso de la ventana de cada serie si es UNO solo (lo usa quien pone la fecha del lote).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from matrixai.training.pronostico_temporal import construir_contexto_de_prediccion, construir_ventanas

HORA, DIA = 3600.0, 86400.0
BASE = datetime(2025, 1, 6)        # lunes


def _f(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _tres_horarias_y_una_diaria() -> list[dict[str, str]]:
    """La forma de aud3 M-3: 3 series horarias de 60 días y 1 diaria (D0) de 60 días."""
    rng = random.Random(7)
    return ([{"serie": f"H{s}", "fecha": _f(BASE + timedelta(hours=k)), "y": f"{rng.gauss(10, 1):.2f}"}
             for s in range(3) for k in range(24 * 60)]
            + [{"serie": "D0", "fecha": _f(BASE + timedelta(days=d)), "y": f"{rng.gauss(10, 1):.2f}"} for d in range(60)])


def _laborable_con_festivos_comunes() -> list[dict[str, str]]:
    """La forma de aud3 M-2: 4 tiendas de lunes a viernes, con Semana Santa y Navidad comunes."""
    rng = random.Random(7)
    comunes = {88, 89, 90, 91, 352, 353}
    return [{"serie": f"T{s}", "fecha": _f(BASE + timedelta(days=d)), "y": f"{rng.gauss(10, 1):.2f}"}
            for s in range(4) for d in range(400) if (BASE + timedelta(days=d)).weekday() < 5 and d not in comunes]


# ----------------------------------------------------------------------------------------------------------------------
# M-3 — una serie de otra frecuencia
# ----------------------------------------------------------------------------------------------------------------------

def test_una_serie_de_otra_frecuencia_que_se_pierde_entera_cuenta_entre_las_series_sin_filas_y_se_marca_por_su_paso():
    """aud3 M-3: antes se quitaban las 56 filas de D0, `series_sin_filas_completas` valía 0 y el corte se leía como «un salto»."""
    v = construir_ventanas(_tres_horarias_y_una_diaria(), tiempo="fecha", serie="serie", objetivo="y", horizonte=1, retardos=3)
    d = v.declaracion
    assert v.hueco == HORA
    assert d["series_sin_filas_por_su_paso"] == [{"serie": "D0", "paso_mediano": DIA, "filas": 56}]
    assert d["series_sin_filas_completas"] == 1                     # contada: el modelo no la ve
    (corte,) = d["cortes_del_calendario"]
    assert corte["serie"] == "D0" and corte["filas"] == 56 and corte["por_su_paso"] is True
    assert d["filas_quitadas_por_cortes_del_calendario"] == 56       # la cuenta de filas no cambia: suman lo mismo
    assert {f["serie"] for f in v.filas} == {"H0", "H1", "H2"}


def test_un_corte_de_verdad_con_el_mismo_paso_no_se_marca_como_otra_frecuencia():
    """El otro lado: una serie diaria con un hueco de 10 días entre otras diarias pierde unas filas por un CORTE, no por su paso."""
    rng = random.Random(3)
    filas = [{"serie": f"S{s}", "fecha": _f(BASE + timedelta(days=d)), "y": f"{rng.gauss(10, 1):.2f}"}
             for s in range(4) for d in range(200) if not (s == 0 and 100 <= d < 110)]
    v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=1, retardos=3)
    d = v.declaracion
    assert d["series_sin_filas_por_su_paso"] == []
    assert d["filas_quitadas_por_cortes_del_calendario"] >= 1
    assert all(not c.get("por_su_paso") for c in d["cortes_del_calendario"])
    assert d["series_sin_filas_completas"] == 0


# ----------------------------------------------------------------------------------------------------------------------
# M-2 — el hueco pedido que quita filas
# ----------------------------------------------------------------------------------------------------------------------

def test_un_hueco_pedido_que_quita_filas_declara_el_mas_ancho_que_las_conservaria():
    """aud3 M-2: laborable, h = 5, hueco pedido de 5 días: se quitaban 884 de 1.128 filas sin decir que con 8 días se conservan."""
    v = construir_ventanas(_laborable_con_festivos_comunes(), tiempo="fecha", serie="serie", objetivo="y", horizonte=5,
                           retardos=3, hueco=5 * DIA)
    h = v.declaracion["hueco"]
    assert h["eleccion"] == "pedido" and v.hueco == 5 * DIA                     # el pedido se RESPETA
    assert v.declaracion["filas_quitadas_por_cortes_del_calendario"] == 884
    assert h["otro_conservaria"] == {"segundos_o_unidades": 8 * DIA, "filas_que_quitaria": 4, "filas_que_quita_el_pedido": 884}


def test_un_hueco_pedido_que_ya_es_el_bueno_no_propone_otro_y_sin_pedirlo_tampoco():
    filas = _laborable_con_festivos_comunes()
    v8 = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=5, retardos=3, hueco=8 * DIA)
    assert v8.declaracion["hueco"]["otro_conservaria"] is None
    v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=5, retardos=3)
    assert v.declaracion["hueco"]["eleccion"] == "calendario" and v.declaracion["hueco"]["otro_conservaria"] is None


# ----------------------------------------------------------------------------------------------------------------------
# I-3 — el paso de la ventana del lote
# ----------------------------------------------------------------------------------------------------------------------

def test_el_contexto_de_prediccion_dice_el_paso_de_la_ventana_solo_si_es_uno():
    diaria = [{"serie": "A", "fecha": _f(BASE + timedelta(days=d)), "y": "1"} for d in range(4)]              # lun-jue
    laborable = [{"serie": "B", "fecha": _f(BASE + timedelta(days=d)), "y": "1"} for d in (2, 3, 4, 7)]     # mié-vie, lun
    ctx = construir_contexto_de_prediccion([*diaria, *laborable], tiempo="fecha", objetivo="y", retardos=3, serie="serie")
    por_serie = {p.serie: p for p in ctx.predecibles}
    assert por_serie["A"].paso_de_la_ventana == DIA and por_serie["A"].calendario_irregular is False
    assert por_serie["B"].paso_de_la_ventana is None and por_serie["B"].calendario_irregular is True
