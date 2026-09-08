# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""108-C4 — aplicar el pipeline calibrado a una predicción nueva.

Criterio de terminado (parcial, la pieza pura): el umbral cambiado en el
JSON cambia la etiqueta en pantalla; predicciones dentro y fuera equivalentes
(aquí: con y sin calibrador reconstruido desde su forma `a_json()`, nunca un
umbral fijo 0,5).
"""
from __future__ import annotations

import math
import unittest

from matrixai.estudio.calibracion import _sigmoid, _logit
from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import PoliticaDeDecision
from matrixai.estudio.inferencia import decidir_desde_probabilidad


def _politica(threshold, positive_label="si", scale="calibrated_probability"):
    return PoliticaDeDecision(threshold=threshold, scale=scale, positive_label=positive_label)


class SinCalibradorTest(unittest.TestCase):
    def test_probabilidad_final_es_la_cruda_sin_calibrador(self):
        d = decidir_desde_probabilidad(0.7, calibrator=None, decision_policy=_politica(0.5),
                                       negative_label="no")
        self.assertEqual(d.probabilidad_final, 0.7)
        self.assertEqual(d.probabilidad_cruda, 0.7)

    def test_umbral_gobierna_la_etiqueta_no_un_0_5_fijo(self):
        """El caso literal del criterio: cambiar el umbral cambia la
        etiqueta, con la MISMA probabilidad de entrada."""
        d_bajo = decidir_desde_probabilidad(0.4, calibrator=None, decision_policy=_politica(0.3),
                                            negative_label="no")
        d_alto = decidir_desde_probabilidad(0.4, calibrator=None, decision_policy=_politica(0.6),
                                            negative_label="no")
        self.assertEqual(d_bajo.etiqueta, "si")
        self.assertEqual(d_alto.etiqueta, "no")

    def test_frontera_exacta_es_positiva(self):
        d = decidir_desde_probabilidad(0.5, calibrator=None, decision_policy=_politica(0.5),
                                       negative_label="no")
        self.assertEqual(d.etiqueta, "si")


class ConCalibradorTest(unittest.TestCase):
    def _calibrador(self, a, b):
        # Misma forma que `RecalibracionLogistica.a_json()` -- lo que trae
        # `FittedPipelineSpec.calibrator` de verdad, no un objeto reconstruido
        # a mano con métodos que el esquema no expone (no hay `desde_json`).
        return {"metodo": "logistico_completo", "n_observaciones": 500, "a": a, "b": b,
                "convergio": True, "iteraciones": 4, "separacion_detectada": False,
                "undefined_reason": None}

    def test_recalibra_antes_de_comparar_contra_el_umbral(self):
        """`p' = sigmoid(a + b*logit(p))` -- valor conocido, no supuesto."""
        p_cruda = 0.8
        a, b = -0.5, 1.2
        esperado = _sigmoid(a + b * _logit(p_cruda))
        d = decidir_desde_probabilidad(p_cruda, calibrator=self._calibrador(a, b),
                                       decision_policy=_politica(0.5), negative_label="no")
        self.assertAlmostEqual(d.probabilidad_final, esperado, places=12)
        self.assertEqual(d.probabilidad_cruda, p_cruda)

    def test_calibrador_que_hunde_la_probabilidad_cambia_la_etiqueta(self):
        """Sin calibrar, 0.9 pasaría cualquier umbral razonable; un
        calibrador con `a` muy negativo la hunde por debajo del umbral --
        la etiqueta tiene que seguir a la probabilidad CALIBRADA, no a la cruda."""
        d = decidir_desde_probabilidad(0.9, calibrator=self._calibrador(a=-6.0, b=1.0),
                                       decision_policy=_politica(0.5), negative_label="no")
        self.assertLess(d.probabilidad_final, 0.5)
        self.assertEqual(d.etiqueta, "no")

    def test_calibrador_sin_ajuste_valido_propaga_el_rechazo_de_calibracion_py(self):
        """`aplicar_recalibracion` ya rechaza un `a`/`b` ausentes (105-C3) --
        este módulo no lo vuelve a comprobar por su cuenta, confía en el
        rechazo existente en vez de duplicarlo."""
        sin_ajuste = self._calibrador(a=None, b=None)
        sin_ajuste["undefined_reason"] = {"es": "separación", "en": "separation"}
        with self.assertRaises(EsquemaInvalido):
            decidir_desde_probabilidad(0.5, calibrator=sin_ajuste, decision_policy=_politica(0.5),
                                       negative_label="no")


class RechazosTest(unittest.TestCase):
    def test_probabilidad_fuera_de_rango_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            decidir_desde_probabilidad(1.5, calibrator=None, decision_policy=_politica(0.5),
                                       negative_label="no")

    def test_probabilidad_negativa_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            decidir_desde_probabilidad(-0.01, calibrator=None, decision_policy=_politica(0.5),
                                       negative_label="no")

    def test_escala_raw_score_se_rechaza_en_vez_de_comparar_en_el_sitio_equivocado(self):
        politica_cruda = _politica(0.0, scale="raw_score")
        with self.assertRaises(EsquemaInvalido):
            decidir_desde_probabilidad(0.5, calibrator=None, decision_policy=politica_cruda,
                                       negative_label="no")


if __name__ == "__main__":
    unittest.main()
