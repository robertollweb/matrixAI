# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""139-C2 — el expediente de un estudio dice QUÉ MOTOR es el modelo entregado, con cuántas filas entrenó y cómo se
calibró y se eligió su umbral (la ficha TRIPOD+AI y el mapa/los huecos PROBAST+AI), y NO lo inventa cuando el manifiesto
no lo trae (un estudio guardado antes del 139).

Se prueba lo que sale en la ficha, en los dos idiomas, y que `ficha` y PROBAST dicen lo mismo (un solo inventario).
"""
from __future__ import annotations

import json

import pytest

from matrixai.export.manifiesto_de_estudio import (
    METODOS_DE_CALIBRACION,
    escribir_manifiesto_de_estudio,
    manifiesto_de_estudio,
)
from matrixai.export.probast import huecos_probast, mapa_probast
from matrixai.export.reproduce import ReproduceManifestError, verify_manifest_digest
from matrixai.export.tripod import ficha_tripod

REPARTO = dict(filas=360, filas_de_desarrollo=288, filas_de_prueba_reservada=72)
# Auditoría de Roberto del 139 (P2-1): una binaria lleva `usos` (para qué sirvió cada mitad), como lo escribe hoy el estudio.
FUERA = {"metodo": "fuera_de_pliegue", "motivo": None, "pliegues": 2, "filas_fuera_de_pliegue": 288,
         "filas_para_calibrar": 144, "filas_para_el_umbral": 144,
         "usos": {"para_calibrar": ["recalibracion", "conforme"], "para_el_umbral": ["umbral", "bandas"]}}
MOTIVO = {"es": "El estudio lee una columna de texto, que no se reentrena por pliegue.",
          "en": "The study reads a text column, which is not retrained per fold."}
PLIEGUE_0 = {"metodo": "pliegue_0", "motivo": MOTIVO, "pliegues": 1, "filas_fuera_de_pliegue": 0,
             "filas_para_calibrar": 20, "filas_para_el_umbral": 21,
             "usos": {"para_calibrar": ["recalibracion"], "para_el_umbral": ["umbral", "evaluacion_de_seleccion"]}}


def _paquete(tmp_path, **kw):
    escribir_manifiesto_de_estudio(tmp_path, **kw)
    return tmp_path


def _entregado(**extra):
    return dict(REPARTO, filas_que_entrenaron_el_modelo=260, filas_de_cola_de_parada_temprana=28,
                motor_del_modelo_entregado="sklearn.hgb", calibracion_y_umbral=FUERA, **extra)


def test_el_bloque_study_lleva_motor_cola_y_metodo_y_el_digest_los_cubre():
    m = manifiesto_de_estudio(**_entregado())
    e = m["study"]
    assert e["delivered_engine"] == "sklearn.hgb"
    assert e["rows"] == {"development": 288, "holdout_test": 72, "trained_the_delivered_model": 260,
                         "early_stopping_tail": 28}
    assert e["calibration_and_threshold"] == {
        "method": "fuera_de_pliegue", "folds": 2, "out_of_fold_rows": 288,
        "rows_for_calibration": 144, "rows_for_threshold": 144,
        "uses": {"calibration_rows": ["recalibration", "conformal"], "threshold_rows": ["threshold", "bands"]}}
    assert "reason" not in e["calibration_and_threshold"]
    assert verify_manifest_digest(m)
    otro = manifiesto_de_estudio(**{**_entregado(), "motor_del_modelo_entregado": "xgboost"})
    assert otro["manifest_sha256"] != m["manifest_sha256"]
    assert verify_manifest_digest(otro)


def test_un_estudio_de_antes_no_lleva_nada_de_esto_ni_a_null():
    m = manifiesto_de_estudio(**REPARTO, filas_que_entrenaron_el_modelo=162)
    assert set(m["study"]) == {"rows"}
    assert set(m["study"]["rows"]) == {"development", "holdout_test", "trained_the_delivered_model"}


def test_el_pliegue_0_lleva_su_motivo_en_los_dos_idiomas():
    m = manifiesto_de_estudio(**REPARTO, calibracion_y_umbral=PLIEGUE_0)
    assert m["study"]["calibration_and_threshold"]["reason"] == MOTIVO
    assert m["study"]["calibration_and_threshold"]["method"] == "pliegue_0"


@pytest.mark.parametrize("malo", [
    dict(calibracion_y_umbral={"metodo": "magia"}),
    dict(calibracion_y_umbral={"metodo": "fuera_de_pliegue", "motivo": {"es": "solo"}}),
    dict(calibracion_y_umbral={"metodo": "fuera_de_pliegue", "pliegues": -1}),
    dict(calibracion_y_umbral={"metodo": "fuera_de_pliegue", "filas_para_calibrar": True}),
    dict(calibracion_y_umbral="fuera_de_pliegue"),
    dict(motor_del_modelo_entregado="  "),
    dict(filas_que_entrenaron_el_modelo=260, filas_de_cola_de_parada_temprana=30),  # 290 > 288
])
def test_un_dato_imposible_se_rechaza(malo):
    with pytest.raises(ReproduceManifestError):
        manifiesto_de_estudio(**REPARTO, **malo)


def test_el_vocabulario_de_metodos_es_el_del_estudio():
    assert METODOS_DE_CALIBRACION == ("fuera_de_pliegue", "pliegue_0")


def test_la_ficha_dice_motor_cola_y_metodo_en_castellano(tmp_path):
    ficha = ficha_tripod(_paquete(tmp_path, **_entregado()), locale="es")
    assert "- **Motor del modelo entregado**: sklearn.hgb" in ficha
    assert "`reproduce.json#study.delivered_engine`" in ficha
    assert "**Filas de la cola de parada temprana" in ficha and ": 28 ·" in ficha
    assert ("- **Calibración y umbral del modelo entregado**: con predicciones fuera de pliegue (cada fila puntuada por un "
            "modelo que no la vio); 288 filas de desarrollo en 2 pliegues; 144 para ajustar la recalibración y calibrar la "
            "predicción conforme; 144 para elegir el umbral y fijar las bandas de decisión") in ficha
    assert "`reproduce.json#study.calibration_and_threshold`" in ficha


def test_la_ficha_en_ingles_no_lleva_castellano(tmp_path):
    ficha = ficha_tripod(_paquete(tmp_path, **_entregado()), locale="en")
    assert "- **Engine of the delivered model**: sklearn.hgb" in ficha
    assert ("- **Calibration and threshold of the delivered model**: with out-of-fold predictions (each row scored by a model "
            "that had not seen it); 288 development rows across 2 folds; 144 to fit the recalibration and calibrate the "
            "conformal prediction; 144 to choose the threshold and set the decision bands") in ficha
    assert "Early-stopping tail rows" in ficha
    for es in ("Motor del modelo", "pliegues;", "fuera de pliegue", "parada temprana"):
        assert es not in ficha


def test_el_pliegue_0_dice_su_motivo_en_el_idioma_pedido(tmp_path):
    paquete = _paquete(tmp_path, **dict(_entregado(), calibracion_y_umbral=PLIEGUE_0))
    es = ficha_tripod(paquete, locale="es")
    en = ficha_tripod(paquete, locale="en")
    assert ("método de antes: con la parte de validación del pliegue 0; 20 para ajustar la recalibración; 21 para elegir "
            "el umbral y medir al modelo en la selección") in es
    assert f"Motivo: {MOTIVO['es']}" in es and MOTIVO["en"] not in es
    assert "the previous method: with the validation part of fold 0" in en
    assert f"Reason: {MOTIVO['en']}" in en and MOTIVO["es"] not in en


def test_un_estudio_de_antes_dice_no_disponible_y_no_inventa_pliegue_0_ni_motor(tmp_path):
    paquete = _paquete(tmp_path, **REPARTO, filas_que_entrenaron_el_modelo=162)
    for idioma, falta in (("es", "_no disponible_"), ("en", "_not available_")):
        ficha = ficha_tripod(paquete, locale=idioma)
        motor = next(l for l in ficha.splitlines() if "study.delivered_engine" in l)
        cal = next(l for l in ficha.splitlines() if "study.calibration_and_threshold" in l)
        assert falta in motor and falta in cal
        assert "pliegue 0" not in ficha and "fold 0" not in ficha
        # la cola no se anuncia como hueco: un estudio de antes no la contó
        assert "early_stopping_tail" not in ficha


def test_probast_dice_lo_mismo_que_la_ficha(tmp_path):
    paquete = _paquete(tmp_path, **_entregado())
    mapa = mapa_probast(paquete, locale="es")
    todas = {c["clave"]: c for dom in mapa["por_dominio"].values() for c in dom}
    assert todas["motor_del_modelo_entregado"]["estado"] == "medido"
    assert todas["motor_del_modelo_entregado"]["valor"] == "sklearn.hgb"
    assert todas["calibracion_y_umbral"]["valor"]["method"] == "fuera_de_pliegue"
    assert todas["filas_de_cola_de_parada_temprana"]["valor"] == 28
    md = huecos_probast(paquete, locale="es")
    assert "Motor del modelo entregado**: sklearn.hgb" in md
    assert "con predicciones fuera de pliegue" in md


def test_probast_de_un_estudio_de_antes_marca_el_hueco_sin_valor(tmp_path):
    paquete = _paquete(tmp_path, **REPARTO)
    mapa = mapa_probast(paquete, locale="es")
    todas = {c["clave"]: c for dom in mapa["por_dominio"].values() for c in dom}
    for clave in ("motor_del_modelo_entregado", "calibracion_y_umbral"):
        assert todas[clave]["estado"] == "falta" and todas[clave]["valor"] is None
    assert "filas_de_cola_de_parada_temprana" not in todas


def test_la_ficha_nunca_califica_el_riesgo_con_las_lineas_nuevas(tmp_path):
    # `ficha_tripod` pasa por la última puerta (sin_veredicto_de_riesgo): si una redacción nueva calificase, reventaría.
    for idioma in ("es", "en"):
        ficha_tripod(_paquete(tmp_path, **_entregado()), locale=idioma)
        ficha_tripod(_paquete(tmp_path, **dict(_entregado(), calibracion_y_umbral=PLIEGUE_0)), locale=idioma)
