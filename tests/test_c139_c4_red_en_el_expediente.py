# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""139-C4, reparación de la re-auditoría (I1, I2) — el expediente de un estudio cuya red entregada se reentrenó con ÉPOCAS FIJAS.

I1: la ficha decía «Épocas declaradas / efectivas / ejecutadas: _no disponible_» de un estudio que guardó 20 épocas fijadas
(medido por la re-auditoría). Ahora el manifiesto lleva `study.network_training` y la ficha y PROBAST lo escriben: cuántas
épocas, de dónde salen y si el plazo cortó el ajuste de selección (el número lo puso ese reloj).
I2: con épocas fijas la cola de parada temprana ya no decide nada, y el rótulo lo dice; con un árbol (o un estudio de antes)
sigue el rótulo de siempre, que allí es verdad.
"""
from __future__ import annotations

import pytest

from matrixai.export.manifiesto_de_estudio import escribir_manifiesto_de_estudio, manifiesto_de_estudio
from matrixai.export.probast import huecos_probast, mapa_probast
from matrixai.export.reproduce import ReproduceManifestError, verify_manifest_digest
from matrixai.export.tripod import ficha_tripod

RED = {"epocas_fijadas": 20, "mejor_epoca_de_la_seleccion": 21, "epocas_completas_de_la_seleccion": 20,
       "parado_por_plazo_en_la_seleccion": True, "epocas_ejecutadas_de_la_seleccion": 21,
       "plazo_de_la_seleccion_segundos": 4.0, "epocas_ejecutadas_en_el_final": 20, "se_entrega": "ultima_epoca",
       "techo_de_seguridad_segundos": 22.5, "la_cola_decide_algo": False}
FUERA = {"metodo": "fuera_de_pliegue", "motivo": None, "pliegues": 2, "filas_fuera_de_pliegue": 1200,
         "filas_para_calibrar": 600, "filas_para_el_umbral": 600}
BASE = dict(filas=1500, filas_de_desarrollo=1200, filas_de_prueba_reservada=300, filas_que_entrenaron_el_modelo=1020,
            filas_de_cola_de_parada_temprana=180, motor_del_modelo_entregado="matrixai.dense.torch_cpu")


def _paquete(tmp_path, calibracion):
    escribir_manifiesto_de_estudio(tmp_path, **BASE, calibracion_y_umbral=calibracion)
    return tmp_path


def test_el_manifiesto_lleva_el_bloque_de_la_red_y_el_digest_lo_cubre():
    m = manifiesto_de_estudio(**BASE, calibracion_y_umbral={**FUERA, "entrenamiento_de_la_red": RED})
    assert m["study"]["network_training"] == {
        "fixed_epochs": 20, "best_epoch_of_selection": 21, "completed_epochs_of_selection": 20,
        "selection_stopped_by_time_limit": True, "epochs_run_in_selection": 21, "selection_time_limit_seconds": 4.0,
        "epochs_run_in_final": 20, "delivers": "last_epoch", "safety_ceiling_seconds": 22.5,
        "early_stopping_tail_decides": False}
    assert verify_manifest_digest(m)
    otro = manifiesto_de_estudio(**BASE, calibracion_y_umbral={**FUERA, "entrenamiento_de_la_red": {**RED, "epocas_fijadas": 19}})
    assert otro["manifest_sha256"] != m["manifest_sha256"]


def test_lo_que_el_sobre_no_dijo_se_omite_y_un_arbol_no_lleva_el_bloque():
    sin_plazo = {k: v for k, v in RED.items() if k != "plazo_de_la_seleccion_segundos"}
    sin_plazo["techo_de_seguridad_segundos"] = None
    m = manifiesto_de_estudio(**BASE, calibracion_y_umbral={**FUERA, "entrenamiento_de_la_red": sin_plazo})
    assert "selection_time_limit_seconds" not in m["study"]["network_training"]
    assert "safety_ceiling_seconds" not in m["study"]["network_training"]
    assert "network_training" not in manifiesto_de_estudio(**BASE, calibracion_y_umbral=FUERA)["study"]


@pytest.mark.parametrize("malo", [{**RED, "epocas_fijadas": 0}, {**RED, "epocas_fijadas": None},
                                  {**RED, "parado_por_plazo_en_la_seleccion": "sí"}, {**RED, "se_entrega": "la_mejor"},
                                  {**RED, "plazo_de_la_seleccion_segundos": -1}, {**RED, "mejor_epoca_de_la_seleccion": True},
                                  "20"])
def test_un_bloque_imposible_se_rechaza(malo):
    with pytest.raises(ReproduceManifestError):
        manifiesto_de_estudio(**BASE, calibracion_y_umbral={**FUERA, "entrenamiento_de_la_red": malo})


@pytest.mark.parametrize("locale", ["es", "en"])
def test_la_ficha_y_probast_dicen_las_epocas_y_que_la_cola_no_decide(tmp_path, locale):
    paquete = _paquete(tmp_path, {**FUERA, "entrenamiento_de_la_red": RED})
    ficha, huecos = ficha_tripod(paquete, locale=locale), huecos_probast(paquete, locale=locale)
    if locale == "es":
        linea = ("Épocas de la red entregada**: 20 épocas fijadas, sin parada temprana; se entrega la última; salen del ajuste "
                 "de selección: su mejor época (21), acotada a sus épocas completas (20); ese ajuste de selección lo cortó su "
                 "tope de tiempo (unos 4 s): el número de épocas lo fijó ese reloj; el modelo entregado hizo 20; la cola de "
                 "parada temprana no decide nada")
        cola = ("Filas de la cola de parada temprana (del desarrollo; se apartan y no entrenan; con las épocas fijadas de la "
                "red no deciden nada)**: 180")
        assert "Épocas declaradas / efectivas / ejecutadas" not in ficha and "deciden cuándo parar" not in ficha + huecos
    else:
        linea = ("Epochs of the delivered network**: 20 fixed epochs, without early stopping; the last one is delivered; they "
                 "come from the selection fit: its best epoch (21), capped at its completed epochs (20); that selection fit "
                 "was cut short by its time limit (about 4 s): the number of epochs was set by that clock; the delivered "
                 "model ran 20; the early-stopping tail decides nothing")
        cola = ("Early-stopping tail rows (from development; set aside, they do not train; with the network's fixed epochs "
                "they decide nothing)**: 180")
        assert "Epochs declared / effective / ran" not in ficha and "decide when to stop" not in ficha + huecos
    assert linea in ficha and linea in huecos
    assert cola in ficha and cola in huecos
    # el mapa PROBAST lo cuenta como MEDIDO, con su ruta y el valor del manifiesto (un solo inventario para los dos)
    evidencias = [e for d in mapa_probast(paquete)["por_dominio"].values() for e in d
                  if isinstance(e, dict) and e.get("clave") == "entrenamiento_de_la_red"]
    assert len(evidencias) == 1
    assert evidencias[0]["estado"] == "medido" and evidencias[0]["ruta"] == "reproduce.json#study.network_training"
    assert evidencias[0]["valor"]["fixed_epochs"] == 20


def test_sin_plazo_en_la_seleccion_lo_dice_sin_inventar_el_reloj(tmp_path):
    paquete = _paquete(tmp_path, {**FUERA, "entrenamiento_de_la_red": {
        **RED, "parado_por_plazo_en_la_seleccion": False, "mejor_epoca_de_la_seleccion": 12,
        "epocas_completas_de_la_seleccion": 30, "epocas_fijadas": 12, "epocas_ejecutadas_en_el_final": 12}})
    ficha = ficha_tripod(paquete, locale="es")
    assert "su mejor época (12), acotada a sus épocas completas (30); ese ajuste de selección no lo cortó su tope de tiempo" in ficha
    assert "lo fijó ese reloj" not in ficha


@pytest.mark.parametrize("locale", ["es", "en"])
def test_un_arbol_o_un_estudio_de_antes_conserva_la_cola_de_siempre_y_no_inventa_epocas(tmp_path, locale):
    paquete = _paquete(tmp_path, FUERA)
    ficha = ficha_tripod(paquete, locale=locale)
    if locale == "es":
        assert "Filas de la cola de parada temprana (del desarrollo; deciden cuándo parar, no entrenan)**: 180" in ficha
        assert "Épocas de la red entregada" not in ficha
    else:
        assert "Early-stopping tail rows (from development; they decide when to stop, they do not train)**: 180" in ficha
        assert "Epochs of the delivered network" not in ficha
