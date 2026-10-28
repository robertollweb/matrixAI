# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Reparación de la auditoría independiente de 140-C2/C3 (28-10), la parte del NÚCLEO:

* I2 — lo que el sobre declara del tiempo llega a la ficha: `study.calibration_and_threshold` lleva `temporal_assumption`,
  `drift_between_halves` y `conformal_without_exchangeability`, y la ficha los escribe diciendo QUÉ mide la deriva (dentro del
  desarrollo) y qué NO mide (del desarrollo al futuro).
* Decisión de Roberto del 28-10 (140-deriva-hacia-el-futuro, A): `study.split.target_in_development_and_test`.
* M5 — «1 día», no «1 días», en la ficha.
"""
from __future__ import annotations

import pytest

from matrixai.export.expediente_clinico import CALIBRACION_Y_UMBRAL, REPARTO_POR_DISENO, Campo, texto_del_valor
from matrixai.export.manifiesto_de_estudio import manifiesto_de_estudio
from matrixai.export.reproduce import ReproduceManifestError

SUPUESTO = {"es": "El modelo final predice un futuro que no ha visto.", "en": "The final model predicts an unseen future."}
NOTA = {"es": "La cobertura nominal no es una garantía.", "en": "The nominal coverage is not a guarantee."}
DERIVA = {"medida": "prevalencia", "en_la_mitad_que_calibra": 0.359, "en_la_mitad_del_umbral": 0.45, "z": 1.17,
          "umbral_de_z": 1.96, "filas_que_calibran": 78, "filas_del_umbral": 80, "hay_deriva": False}
BASE = {"metodo": "fuera_de_pliegue", "pliegues": 2, "filas_fuera_de_pliegue": 158}


def _calibracion(**extra):
    return manifiesto_de_estudio(calibracion_y_umbral={**BASE, **extra})["study"]["calibration_and_threshold"]


def _texto(evidencia, ruta, bloque, idioma):
    return texto_del_valor(evidencia, Campo(ruta, "medido", valor=bloque), idioma)


class TestI2ElTiempoEnLaFicha:

    def test_el_supuesto_la_deriva_y_la_nota_conforme_llegan_al_manifiesto(self):
        c = _calibracion(supuesto_temporal=SUPUESTO, deriva=DERIVA, conforme_sin_intercambiabilidad=NOTA)
        assert c["temporal_assumption"] == SUPUESTO and c["conformal_without_exchangeability"] == NOTA
        assert c["drift_between_halves"] == {
            "measure": "prevalence", "in_the_calibration_half": 0.359, "in_the_threshold_half": 0.45, "z": 1.17,
            "z_threshold": 1.96, "calibration_rows": 78, "threshold_rows": 80, "drift": False}

    def test_sin_ellos_el_bloque_de_siempre(self):
        assert set(_calibracion()) == {"method", "folds", "out_of_fold_rows"}

    @pytest.mark.parametrize("malo", [
        {"supuesto_temporal": {"es": "solo es"}}, {"conforme_sin_intercambiabilidad": "texto"},
        {"deriva": {**DERIVA, "medida": "otra"}}, {"deriva": {**DERIVA, "z": float("nan")}},
        {"deriva": {**DERIVA, "hay_deriva": "no"}}, {"deriva": {**DERIVA, "filas_del_umbral": -1}}])
    def test_mal_formado_se_rechaza(self, malo):
        with pytest.raises(ReproduceManifestError):
            _calibracion(**malo)

    def test_la_ficha_dice_lo_que_mide_la_deriva_y_lo_que_no_mide(self):
        c = _calibracion(supuesto_temporal=SUPUESTO, deriva=DERIVA, conforme_sin_intercambiabilidad=NOTA)
        es = _texto(CALIBRACION_Y_UMBRAL, "reproduce.json#study.calibration_and_threshold", c, "es")
        en = _texto(CALIBRACION_Y_UMBRAL, "reproduce.json#study.calibration_and_threshold", c, "en")
        assert "El modelo final predice un futuro que no ha visto" in es and "nominal coverage is not a guarantee" in en
        assert "DENTRO del desarrollo" in es and "35.9 %" in es and "45.0 %" in es
        assert "No mide la deriva del desarrollo al futuro" in es
        assert "WITHIN development" in en and "does not measure drift from development to the future" in en
        assert "sin deriva apreciable" in es
        hay = _calibracion(deriva={**DERIVA, "z": 2.5, "hay_deriva": True})
        assert "pueden haber derivado" in _texto(CALIBRACION_Y_UMBRAL, "reproduce.json#study.calibration_and_threshold",
                                                 hay, "es")


OBJETIVO = {"tarea": "clasificacion", "cambia_mucho": True,
            "desarrollo": {"filas": 237, "proporciones": {"0": 0.646, "1": 0.354}},
            "prueba": {"filas": 56, "proporciones": {"0": 0.321, "1": 0.679}},
            "mayor_cambio": {"clase": "1", "puntos": 32.5, "z": 4.4}, "criterio": {"puntos": 10.0, "z": 1.96}}


class TestElObjetivoEnDesarrolloYPrueba:

    def _split(self, objetivo):
        return manifiesto_de_estudio(reparto_del_estudio={"diseno": "temporal", "time_column": "fecha",
                                                          "objetivo_en_desarrollo_y_prueba": objetivo})["study"]["split"]

    def test_clasificacion_en_el_manifiesto_y_en_la_ficha(self):
        o = self._split(OBJETIVO)["target_in_development_and_test"]
        assert o["task"] == "classification" and o["changes_a_lot"] is True
        assert o["development"] == {"rows": 237, "shares": {"0": 0.646, "1": 0.354}}
        assert o["largest_change"] == {"class": "1", "points": 32.5, "z": 4.4}
        assert o["criterion"] == {"points": 10.0, "z": 1.96}
        es = _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", self._split(OBJETIVO), "es")
        en = _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", self._split(OBJETIVO), "en")
        assert "«1» es el 35.4 % del desarrollo y el 67.9 % de la prueba reservada: CAMBIA MUCHO" in es
        assert "no cambia ninguna decisión" in es and "IT CHANGES A LOT" in en

    def test_regresion_con_la_media_y_la_desviacion_del_desarrollo(self):
        o = {"tarea": "regresion", "cambia_mucho": False, "desarrollo": {"filas": 240, "media": 24.85, "desviacion_tipica": 8.0},
             "prueba": {"filas": 60, "media": 22.37, "desviacion_tipica": 11.9}, "cambio": {"desviaciones": -0.31, "z": -1.53},
             "criterio": {"desviaciones": 0.5, "z": 1.96}}
        s = self._split(o)
        assert s["target_in_development_and_test"]["change"] == {"standard_deviations": -0.31, "z": -1.53}
        es = _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", s, "es")
        assert "media del objetivo es 24.85 en el desarrollo y 22.37" in es and "(no cambia mucho)" in es

    def test_sin_el_dato_no_hay_bloque_y_mal_formado_se_rechaza(self):
        assert "target_in_development_and_test" not in self._split(None)
        with pytest.raises(ReproduceManifestError):
            self._split({**OBJETIVO, "cambia_mucho": "sí"})
        with pytest.raises(ReproduceManifestError):
            self._split({**OBJETIVO, "prueba": {"filas": 0, "proporciones": {"1": 1.0}}})


class TestR5ElObjetivoSinVariacionEnElDesarrollo:
    """R5 de la re-auditoría del C2/C3 (28-10): con el objetivo CONSTANTE en el desarrollo no hay desviación con que medir el
    cambio. El sobre lo dice (`desviaciones: null`, `desarrollo_sin_variacion: true`) y el manifiesto lo admite SOLO con la
    desviación del desarrollo en 0 (un `null` cualquiera sigue rechazándose)."""

    SIN_VARIACION = {"tarea": "regresion", "cambia_mucho": True,
                     "desarrollo": {"filas": 20, "media": 5.0, "desviacion_tipica": 0.0},
                     "prueba": {"filas": 20, "media": 9.25, "desviacion_tipica": 0.2565},
                     "cambio": {"desviaciones": None, "z": 74.1, "desarrollo_sin_variacion": True},
                     "criterio": {"desviaciones": 0.5, "z": 1.96}}

    def _split(self, objetivo):
        return manifiesto_de_estudio(reparto_del_estudio={"diseno": "temporal", "time_column": "fecha",
                                                          "objetivo_en_desarrollo_y_prueba": objetivo})["study"]["split"]

    def test_el_manifiesto_lo_admite_y_la_ficha_lo_dice(self):
        s = self._split(self.SIN_VARIACION)
        assert s["target_in_development_and_test"]["change"] == {
            "standard_deviations": None, "z": 74.1, "development_without_variation": True}
        es = _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", s, "es")
        en = _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", s, "en")
        assert "en el desarrollo el objetivo no varía" in es and ": CAMBIA MUCHO" in es
        assert "the target does not vary in development" in en and "IT CHANGES A LOT" in en
        # con la prueba también constante, `z` es `null`
        o = {**self.SIN_VARIACION, "cambio": {"desviaciones": None, "z": None, "desarrollo_sin_variacion": True}}
        assert self._split(o)["target_in_development_and_test"]["change"]["z"] is None

    def test_un_null_sin_la_marca_o_con_desviacion_se_rechaza(self):
        with pytest.raises(ReproduceManifestError):
            self._split({**self.SIN_VARIACION, "cambio": {"desviaciones": None, "z": 74.1}})
        with pytest.raises(ReproduceManifestError):
            self._split({**self.SIN_VARIACION, "desarrollo": {"filas": 20, "media": 5.0, "desviacion_tipica": 1.2}})
        with pytest.raises(ReproduceManifestError):
            self._split({**self.SIN_VARIACION, "cambio": {"desviaciones": 3.0, "z": 74.1, "desarrollo_sin_variacion": True}})


class TestM5ElSingularEnLaFicha:

    def test_un_dia_y_un_day(self):
        bloque = manifiesto_de_estudio(reparto_del_estudio={
            "diseno": "temporal", "time_column": "fecha", "gap": {"magnitud": 1, "unidad": "days"}})["study"]["split"]
        assert "hueco de 1 día entre" in _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", bloque, "es")
        assert "gap of 1 day between" in _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", bloque, "en")
        dos = manifiesto_de_estudio(reparto_del_estudio={
            "diseno": "temporal", "time_column": "fecha", "gap": {"magnitud": 2, "unidad": "weeks"}})["study"]["split"]
        assert "hueco de 2 semanas" in _texto(REPARTO_POR_DISENO, "reproduce.json#study.split", dos, "es")
