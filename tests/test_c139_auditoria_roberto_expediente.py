# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Auditoría de Roberto del contrato 139 — lo que el EXPEDIENTE dice del modelo entregado (el núcleo: manifiesto y ficha).

P2-1 · «N para ajustar la recalibración; N para elegir el umbral y las bandas» se escribía SIEMPRE, también en una regresión
       y en una multiclase, que no tienen ni recalibración ni umbral. Roberto lo vio en estudios completados: una regresión
       declaraba 80 filas para recalibrar y 80 para el umbral y no tenía ninguna de las dos cosas. Ahora el sobre dice PARA
       QUÉ SIRVIÓ cada mitad (`usos`), el manifiesto lo lleva (`study.calibration_and_threshold.uses`) y la ficha lo escribe;
       un manifiesto de antes, sin `uses`, da las cuentas SIN afirmar para qué sirvieron.
P3-5 · una red cortada que CAE AL PLIEGUE 0 decía «Épocas: no disponible» sin una palabra del plazo. El manifiesto lleva ahora
       cómo terminó el entrenamiento de la red entregada (`study.delivered_network_training`) y la línea de las épocas lo
       lee cuando no hay `network_training`.

Los bloques de abajo son los que escribe el Studio (copiados de estudios reales del 29-10; la prueba con el estudio de
verdad, en es y en, está en el backend: `test_c139_auditoria_roberto.py`).
"""
from __future__ import annotations

import pytest

from matrixai.export.manifiesto_de_estudio import escribir_manifiesto_de_estudio, manifiesto_de_estudio
from matrixai.export.probast import huecos_probast, mapa_probast
from matrixai.export.reproduce import ReproduceManifestError
from matrixai.export.tripod import ficha_tripod

REPARTO = dict(filas=300, filas_de_desarrollo=240, filas_de_prueba_reservada=60, filas_que_entrenaron_el_modelo=204,
               filas_de_cola_de_parada_temprana=36, motor_del_modelo_entregado="lightgbm")


def _fuera(usos):
    return {"metodo": "fuera_de_pliegue", "motivo": None, "pliegues": 2, "filas_fuera_de_pliegue": 240,
            "filas_para_calibrar": 120, "filas_para_el_umbral": 120, "usos": usos}


#: Lo que escribe el Studio en cada tarea (medido con estudios reales: ver el backend).
REGRESION = _fuera({"para_calibrar": ["conforme"], "para_el_umbral": ["guarda_del_campeon"]})
MULTICLASE = _fuera({"para_calibrar": ["conforme"], "para_el_umbral": ["guarda_del_campeon"]})
BINARIA = _fuera({"para_calibrar": ["recalibracion", "conforme"], "para_el_umbral": ["umbral", "guarda_del_campeon"]})
#: Una regresión cuya guarda no se aplicó y sin cuantil conforme: ninguna de las dos mitades sirvió para nada.
REGRESION_SIN_NADA = _fuera({"para_calibrar": [], "para_el_umbral": []})
#: Un estudio guardado ANTES de la reparación: las cuentas, sin `usos`.
DE_ANTES = {k: v for k, v in REGRESION.items() if k != "usos"}


def _linea(ficha: str, ruta: str) -> str:
    return next(l for l in ficha.splitlines() if ruta in l)


def _ficha(tmp_path, idioma, **kw):
    escribir_manifiesto_de_estudio(tmp_path, **kw)
    return ficha_tripod(tmp_path, locale=idioma)


class TestParaQueSirvioCadaMitad:

    def test_el_manifiesto_lleva_los_usos_en_ingles(self):
        m = manifiesto_de_estudio(**REPARTO, calibracion_y_umbral=REGRESION)
        assert m["study"]["calibration_and_threshold"]["uses"] == {
            "calibration_rows": ["conformal"], "threshold_rows": ["champion_guard"]}
        b = manifiesto_de_estudio(**REPARTO, calibracion_y_umbral=BINARIA)["study"]["calibration_and_threshold"]["uses"]
        assert b == {"calibration_rows": ["recalibration", "conformal"], "threshold_rows": ["threshold", "champion_guard"]}
        # el digest cubre los usos: no se puede cambiar lo que se hizo sin que cambie
        assert m["manifest_sha256"] != manifiesto_de_estudio(**REPARTO, calibracion_y_umbral=REGRESION_SIN_NADA)[
            "manifest_sha256"]

    def test_un_manifiesto_de_antes_no_trae_usos(self):
        assert "uses" not in manifiesto_de_estudio(**REPARTO, calibracion_y_umbral=DE_ANTES)["study"][
            "calibration_and_threshold"]

    @pytest.mark.parametrize("usos", [
        {"para_calibrar": ["recalibracion"]},                                        # falta una mitad
        {"para_calibrar": ["magia"], "para_el_umbral": []},                          # fuera del vocabulario
        {"para_calibrar": ["umbral"], "para_el_umbral": []},                         # el umbral no sale de la mitad que calibra
        {"para_calibrar": ["conforme", "conforme"], "para_el_umbral": []},           # repetido
        {"para_calibrar": "conforme", "para_el_umbral": []},                         # no es una lista
    ])
    def test_unos_usos_imposibles_se_rechazan(self, usos):
        with pytest.raises(ReproduceManifestError):
            manifiesto_de_estudio(**REPARTO, calibracion_y_umbral=_fuera(usos))

    @pytest.mark.parametrize("bloque", [REGRESION, MULTICLASE], ids=["regresion", "multiclase"])
    def test_regresion_y_multiclase_no_dicen_recalibracion_ni_umbral_es(self, tmp_path, bloque):
        linea = _linea(_ficha(tmp_path, "es", **REPARTO, calibracion_y_umbral=bloque), "study.calibration_and_threshold")
        assert ("Calibración y umbral del modelo entregado**: sin recalibración ni umbral; con predicciones fuera de pliegue "
                "(cada fila puntuada por un modelo que no la vio); 240 filas de desarrollo en 2 pliegues; 120 para calibrar "
                "la predicción conforme; 120 para comparar al ganador con la referencia") in linea
        for falso in ("ajustar la recalibración", "elegir el umbral", "bandas"):
            assert falso not in linea

    @pytest.mark.parametrize("bloque", [REGRESION, MULTICLASE], ids=["regresion", "multiclase"])
    def test_regresion_y_multiclase_no_dicen_recalibracion_ni_umbral_en(self, tmp_path, bloque):
        linea = _linea(_ficha(tmp_path, "en", **REPARTO, calibracion_y_umbral=bloque), "study.calibration_and_threshold")
        assert ("Calibration and threshold of the delivered model**: no recalibration and no threshold; with out-of-fold "
                "predictions (each row scored by a model that had not seen it); 240 development rows across 2 folds; 120 to "
                "calibrate the conformal prediction; 120 to compare the winner with the reference") in linea
        for falso in ("fit the recalibration", "choose the threshold", "bands", "recalibración", "umbral"):
            assert falso not in linea

    def test_una_binaria_sigue_diciendo_su_recalibracion_y_su_umbral(self, tmp_path):
        es = _linea(_ficha(tmp_path, "es", **REPARTO, calibracion_y_umbral=BINARIA), "study.calibration_and_threshold")
        assert "120 para ajustar la recalibración y calibrar la predicción conforme" in es
        assert "120 para elegir el umbral y comparar al ganador con la referencia" in es
        assert "sin recalibración" not in es and "sin umbral" not in es
        assert "bandas" not in es                                     # sin coste de revisión no hubo bandas: no se dicen

    def test_una_mitad_que_no_sirvio_para_nada_lo_dice(self, tmp_path):
        es = _linea(_ficha(tmp_path, "es", **REPARTO, calibracion_y_umbral=REGRESION_SIN_NADA),
                    "study.calibration_and_threshold")
        assert "120 que no se usaron para nada; 120 que no se usaron para nada" in es
        en = _linea(_ficha(tmp_path, "en", **REPARTO, calibracion_y_umbral=REGRESION_SIN_NADA),
                    "study.calibration_and_threshold")
        assert "120 that were not used for anything; 120 that were not used for anything" in en

    def test_un_manifiesto_de_antes_da_las_cuentas_sin_afirmar_para_que_sirvieron(self, tmp_path):
        es = _linea(_ficha(tmp_path, "es", **REPARTO, calibracion_y_umbral=DE_ANTES), "study.calibration_and_threshold")
        assert "dos mitades de 120 y 120 filas (el estudio no registró para qué sirvió cada una)" in es
        assert "recalibración" not in es and "umbral" not in es.split("**:", 1)[1]
        en = _linea(_ficha(tmp_path, "en", **REPARTO, calibracion_y_umbral=DE_ANTES), "study.calibration_and_threshold")
        assert "two halves of 120 and 120 rows (the study did not record what each was used for)" in en


# ----------------------------------------------------------------------------------------------------------------------
# P3-5 — la red que cayó al pliegue 0 dice cómo terminó su entrenamiento
# ----------------------------------------------------------------------------------------------------------------------

#: Lo que declaró la red ENTREGADA de `142_densa_cortada_techo2` (tercera pasada del 139: selección cortada por su plazo de 4 s,
#: el reentrenamiento agotó su techo de seguridad y se cayó al pliegue 0; `out/142_densa_cortada_techo2.json`).
CORTADA = {"parado_por_plazo": True, "epocas_ejecutadas": 18, "epocas_completas": 17, "mejor_epoca": 18,
           "ultima_epoca_parcial": True, "plazo_antes_de_la_primera_epoca_completa": False,
           "plazo_de_entrenamiento_segundos": 4.0, "dispositivo": "cpu", "cayo_a_cpu": False, "por_que_cayo": None}
PLIEGUE_0_DE_LA_RED = {"metodo": "pliegue_0", "pliegues": 1, "filas_fuera_de_pliegue": 0, "filas_para_calibrar": 120,
                       "filas_para_el_umbral": 120,
                       "motivo": {"es": "La red reentrenada con 17 épocas fijas agotó su techo de seguridad.",
                                  "en": "The network retrained with 17 fixed epochs ran out of its safety ceiling."}}


class TestLaRedQueCayoAlPliegue0:

    def test_el_manifiesto_lleva_como_termino_el_entrenamiento_de_la_red_entregada(self):
        m = manifiesto_de_estudio(**REPARTO, calibracion_y_umbral=PLIEGUE_0_DE_LA_RED, entrenamiento_del_entregado=CORTADA)
        assert m["study"]["delivered_network_training"] == {
            "stopped_by_time_limit": True, "epochs_run": 18, "completed_epochs": 17, "best_epoch": 18,
            "last_epoch_partial": True, "time_limit_before_first_complete_epoch": False, "time_limit_seconds": 4.0}
        assert "network_training" not in m["study"]
        # un árbol no lo trae, y el manifiesto de siempre no cambia de forma
        assert "delivered_network_training" not in manifiesto_de_estudio(**REPARTO)["study"]

    @pytest.mark.parametrize("malo", [{"epocas_ejecutadas": 3}, {**CORTADA, "epocas_completas": -1},
                                      {**CORTADA, "plazo_de_entrenamiento_segundos": 0}, "cortada"])
    def test_lo_imposible_se_rechaza(self, malo):
        with pytest.raises(ReproduceManifestError):
            manifiesto_de_estudio(**REPARTO, entrenamiento_del_entregado=malo)

    def test_la_ficha_dice_las_epocas_y_que_el_plazo_la_corto_es(self, tmp_path):
        ficha = _ficha(tmp_path, "es", **REPARTO, calibracion_y_umbral=PLIEGUE_0_DE_LA_RED, entrenamiento_del_entregado=CORTADA)
        linea = _linea(ficha, "study.delivered_network_training")
        assert linea.startswith("- **Épocas de la red entregada**: ")
        assert ("se entrega su época 18; de 18 ejecutadas, 17 completas; su entrenamiento lo cortó su tope de tiempo (unos "
                "4 s): el número de épocas lo fijó ese reloj; la última época quedó a medias; y la época entregada es esa, "
                "la que quedó a medias") in linea
        assert "no disponible" not in linea
        assert "generation.epochs_declared" not in ficha                   # ya no la línea de siempre, vacía

    def test_la_ficha_en_ingles(self, tmp_path):
        ficha = _ficha(tmp_path, "en", **REPARTO, calibracion_y_umbral=PLIEGUE_0_DE_LA_RED, entrenamiento_del_entregado=CORTADA)
        linea = _linea(ficha, "study.delivered_network_training")
        assert ("- **Epochs of the delivered network**: its epoch 18 is delivered; out of 18 run, 17 completed; its training "
                "was cut short by its time limit (about 4 s): the number of epochs was set by that clock; the last epoch was "
                "left half done; and the delivered epoch is that one, the half-done one") in linea
        for es in ("época", "plazo", "reloj", "no disponible"):
            assert es not in linea

    def test_sin_plazo_lo_dice_y_no_inventa_el_corte(self, tmp_path):
        normal = {"parado_por_plazo": False, "epocas_ejecutadas": 40, "epocas_completas": 40, "mejor_epoca": 23}
        linea = _linea(_ficha(tmp_path, "es", **REPARTO, calibracion_y_umbral=PLIEGUE_0_DE_LA_RED,
                              entrenamiento_del_entregado=normal), "study.delivered_network_training")
        assert "se entrega su época 23; de 40 ejecutadas, 40 completas; no lo cortó su tope de tiempo" in linea
        assert "cortó su tope" not in linea.replace("no lo cortó su tope", "")

    def test_con_network_training_manda_ese_bloque(self, tmp_path):
        red = {"epocas_fijadas": 20, "mejor_epoca_de_la_seleccion": 20, "epocas_completas_de_la_seleccion": 20,
               "se_entrega": "ultima_epoca", "la_cola_decide_algo": False}
        fuera = {**BINARIA, "entrenamiento_de_la_red": red}
        ficha = _ficha(tmp_path, "es", **REPARTO, calibracion_y_umbral=fuera,
                       entrenamiento_del_entregado={"parado_por_plazo": False, "epocas_ejecutadas": 20, "mejor_epoca": 20})
        assert "study.network_training" in ficha and "study.delivered_network_training" not in ficha

    def test_probast_lo_mapea_igual(self, tmp_path):
        escribir_manifiesto_de_estudio(tmp_path, **REPARTO, calibracion_y_umbral=PLIEGUE_0_DE_LA_RED,
                                       entrenamiento_del_entregado=CORTADA)
        for idioma, frase in (("es", "su entrenamiento lo cortó su tope de tiempo (unos 4 s)"),
                              ("en", "its training was cut short by its time limit (about 4 s)")):
            assert frase in huecos_probast(tmp_path, locale=idioma)
        evidencias = [e for d in mapa_probast(tmp_path)["por_dominio"].values() for e in d
                      if isinstance(e, dict) and e.get("clave") == "entrenamiento_del_entregado"]
        assert len(evidencias) == 1
        assert evidencias[0]["estado"] == "medido"
        assert evidencias[0]["ruta"] == "reproduce.json#study.delivered_network_training"
        assert evidencias[0]["valor"]["stopped_by_time_limit"] is True
