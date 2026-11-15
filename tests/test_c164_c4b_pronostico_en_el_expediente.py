# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""164-C4b — `study.forecast` en el manifiesto de un estudio y su línea en el expediente (ficha TRIPOD / PROBAST).

Lo que un estudio de pronóstico declara (horizonte, retardos, serie, hueco, filas quitadas, contexto para predecir, límites) y la nota
de la prueba junto a sus dos referencias. Ausente es ausente: una referencia sin calcular no es un cero, y un estudio que no es de
pronóstico no lleva ni el bloque ni la línea (el manifiesto de siempre, byte a byte).
"""
from __future__ import annotations

import json

import pytest

from matrixai.export.expediente_clinico import ExpedienteClinico, PRONOSTICO_DEL_ESTUDIO, texto_del_valor
from matrixai.export.manifiesto_de_estudio import ReproduceManifestError, escribir_manifiesto_de_estudio, manifiesto_de_estudio
from matrixai.export.probast import huecos_probast
from matrixai.export.tripod import ficha_tripod

PRONOSTICO = {
    "horizonte_pasos": 7, "retardos": 14, "columna_de_tiempo": "fecha", "objetivo": "lluvia_mm", "columna_de_serie": "ciudad",
    "columnas_con_retardo": ["lluvia_mm", "temp"], "hueco": {"magnitud": 7, "unidad": "days"}, "estacion": 7,
    "filas": {"de_entrada": 1000, "resultantes": 800, "quitadas_por_el_borde_de_la_serie": 196, "quitadas_por_valores_ausentes": 4, "series": 2},
    "contexto_para_predecir": {"filas_por_serie": 15, "columnas": ["fecha", "ciudad", "lluvia_mm", "temp"]},
    "prueba": {"modelo": {"mae": 1.5, "n": 90}, "persistencia": {"mae": 2.0, "n": 90}, "estacional": {"mae": 1.2, "n": 90},
               "mejora_mae_sobre_persistencia": 0.25, "mejora_mae_sobre_estacional": -0.25, "mase": 1.1},
    "veredicto_por_pliegue": {"persistencia": {"gana": 2, "pierde": 1, "empata": 0, "medidos": 3}},
    "limites": [{"clave": "intervalos_con_filas_solapadas", "motivo": {"es": "los intervalos pueden ser estrechos", "en": "intervals may be narrow"}}],
}


def _paquete(tmp_path, **extra):
    escribir_manifiesto_de_estudio(tmp_path, filas=1000, pronostico=PRONOSTICO, **extra)
    return tmp_path


def test_el_manifiesto_lleva_el_planteamiento_y_la_nota_con_sus_referencias():
    f = manifiesto_de_estudio(pronostico=PRONOSTICO)["study"]["forecast"]
    assert (f["horizon_steps"], f["lags"], f["series_column"], f["time_column"], f["target"]) == (7, 14, "ciudad", "fecha", "lluvia_mm")
    assert f["gap"] == {"magnitude": 7, "unit": "days"} and f["seasonal_period"] == 7
    assert f["rows"] == {"input": 1000, "resulting": 800, "dropped_at_series_edges": 196, "dropped_for_missing_values": 4, "series": 2}
    assert f["context_to_predict"] == {"rows_per_series": 15, "columns": ["fecha", "ciudad", "lluvia_mm", "temp"], "single_row_allowed": False}
    assert f["reference_on_test"]["persistence"] == {"mae": 2.0, "n": 90} and f["reference_on_test"]["improvement_over_seasonal"] == -0.25
    assert f["fold_verdict"] == {"persistence": {"wins": 2, "loses": 1, "ties": 0, "measured": 3}}
    assert f["limits"][0]["reason"]["en"] == "intervals may be narrow"


def test_sin_pronostico_el_manifiesto_es_el_de_siempre():
    assert "forecast" not in manifiesto_de_estudio(filas=10)["study"]


def test_una_referencia_que_no_se_calculo_no_se_escribe_como_cero():
    f = manifiesto_de_estudio(pronostico={**PRONOSTICO, "prueba": {"modelo": {"mae": 1.5}, "persistencia": {"mae": 2.0},
                                                                   "estacional": None, "mejora_mae_sobre_estacional": None}})["study"]["forecast"]
    ref = f["reference_on_test"]
    assert "seasonal" not in ref and "improvement_over_seasonal" not in ref and "mase" not in ref


@pytest.mark.parametrize("mal", [{"horizonte_pasos": 0}, {"retardos": 1.5}, {"hueco": {"magnitud": -2, "unidad": "days"}},
                                 {"prueba": {"persistencia": {"mae": float("inf")}}}, {"limites": [{"clave": "x", "motivo": {"es": "a"}}]}])
def test_un_pronostico_mal_formado_se_rechaza(mal):
    with pytest.raises(ReproduceManifestError):
        manifiesto_de_estudio(pronostico={**PRONOSTICO, **mal})


def test_la_ficha_y_probast_escriben_la_linea_en_los_dos_idiomas(tmp_path):
    paquete = _paquete(tmp_path)
    es, en = ficha_tripod(paquete, locale="es"), ficha_tripod(paquete, locale="en")
    linea_es = next(l for l in es.splitlines() if "study.forecast" in l)
    linea_en = next(l for l in en.splitlines() if "study.forecast" in l)
    assert "horizonte de 7 paso(s), 14 retardo(s)" in linea_es and "hueco de 7 días" in linea_es and "serie «ciudad»" in linea_es
    assert "persistencia (repetir el último valor): mejora 25,0 %" in linea_es and "estacional ingenua: pierde 25,0 %" in linea_es
    assert "MASE 1,1" in linea_es and "gana en 2 de 3 pliegues" in linea_es and "los intervalos pueden ser estrechos" in linea_es
    assert "horizon of 7 step(s), 14 lag(s)" in linea_en and "gap of 7 days" in linea_en and "seasonal naive: loses 25.0 %" in linea_en
    assert "wins on 2 of 3 folds" in linea_en and "intervals may be narrow" in linea_en
    assert "study.forecast" in huecos_probast(paquete, locale="es")


def test_un_estudio_sin_pronostico_no_lleva_la_linea(tmp_path):
    escribir_manifiesto_de_estudio(tmp_path, filas=1000)
    assert "study.forecast" not in ficha_tripod(tmp_path, locale="es")
    assert "study.forecast" not in huecos_probast(tmp_path, locale="es")


def test_el_texto_no_inventa_lo_que_el_bloque_no_trae():
    campo = type("C", (), {"valor": {"horizon_steps": 1, "lags": 3, "time_column": "t", "target": "y"}})()
    texto = texto_del_valor(PRONOSTICO_DEL_ESTUDIO, campo, "es")
    assert texto == "horizonte de 1 paso(s), 3 retardo(s), objetivo «y», columna de tiempo «t»"


def test_el_hueco_conserva_su_unidad_y_la_ficha_la_escribe_en_cada_idioma(tmp_path):
    """Un pronóstico por HORAS no es uno por días: la unidad viaja al manifiesto y la ficha la nombra (singular y plural)."""
    for magnitud, unidad, es, en in ((36, "hours", "hueco de 36 horas", "gap of 36 hours"), (1, "hours", "hueco de 1 hora", "gap of 1 hour"),
                                     (1.5, "weeks", "hueco de 1,5 semanas", "gap of 1.5 weeks")):
        paquete = tmp_path / unidad / str(magnitud)
        paquete.mkdir(parents=True)
        escribir_manifiesto_de_estudio(paquete, filas=1000, pronostico={**PRONOSTICO, "hueco": {"magnitud": magnitud, "unidad": unidad}})
        assert json.loads((paquete / "reproduce.json").read_text())["study"]["forecast"]["gap"] == {"magnitude": magnitud, "unit": unidad}
        assert es in next(l for l in ficha_tripod(paquete, locale="es").splitlines() if "study.forecast" in l)
        assert en in next(l for l in ficha_tripod(paquete, locale="en").splitlines() if "study.forecast" in l)
