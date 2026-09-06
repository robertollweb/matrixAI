# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C5 — ajuste final, test y alternativas.

Criterio de terminado: «Digest del pipeline probado es el exportado a 106;
la etiqueta de evaluación identifica artefacto y partición. Reentrenar con
test crea versión nueva sin evaluación independiente atribuida. Promoción
de candidato conserva historial y requiere evidencia comparable.»
"""
from __future__ import annotations

import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.accesos import RegistroDeAccesos
from matrixai.estudio.esquemas import (
    EvaluationResult,
    Restriccion,
    SplitPlan,
    version_tras_aprender_del_test,
)
from matrixai.estudio.evaluacion_final import ValidacionFinal, evaluar_en_test, promover_candidato
from matrixai.estudio.metricas import Muestra


def _plan(n_dev=100, n_test=40):
    assignments = {f"d{i}": "development" for i in range(n_dev)}
    assignments.update({f"t{i}": "test" for i in range(n_test)})
    return SplitPlan(plan_id="p1", split_type="iid", assignments=assignments,
                     observation_id_field="id")


def _muestra_binaria(n, aciertos):
    y_true = tuple("si" if i % 3 == 0 else "no" for i in range(n))
    if aciertos:
        preds = y_true
    else:
        preds = tuple("no" for _ in range(n))
    return Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                   positive_label="si", predictions=preds)


class CriterioLiteralTest(unittest.TestCase):
    def test_digest_del_pipeline_probado_es_el_exportado(self):
        plan = _plan()
        registro = RegistroDeAccesos(plan)
        with registro.fase("final_evaluation"):
            registro.congelar("pipeline-A")
            resultado = evaluar_en_test(registro, "pipeline-A", _muestra_binaria(40, True))
        self.assertEqual(resultado.evaluacion.pipeline_digest, "pipeline-A")

    def test_etiqueta_identifica_artefacto_y_particion(self):
        plan = _plan()
        registro = RegistroDeAccesos(plan)
        with registro.fase("final_evaluation"):
            registro.congelar("pipeline-A")
            resultado = evaluar_en_test(registro, "pipeline-A", _muestra_binaria(40, True))
        self.assertEqual(resultado.evaluacion.pipeline_digest, "pipeline-A")
        self.assertEqual(resultado.evaluacion.split_plan_digest, plan.digest())

    def test_confirmado_cuando_supera_restricciones_en_test(self):
        plan = _plan()
        registro = RegistroDeAccesos(plan)
        with registro.fase("final_evaluation"):
            registro.congelar("pipeline-A")
            resultado = evaluar_en_test(
                registro, "pipeline-A", _muestra_binaria(40, True),
                restricciones=[Restriccion(clave="accuracy", operador="min", valor=0.9)])
        self.assertEqual(resultado.resultado, "confirmado")
        self.assertEqual(resultado.evaluacion.evidence, "independent_test")
        self.assertIsNone(resultado.motivo_del_fallo)

    def test_fallo_de_validacion_cuando_no_supera_restricciones_en_test(self):
        """El pipeline pasó las restricciones en desarrollo (por eso llegó
        aquí) pero falla en el test reservado -- se declara fallo de
        validación de la RECOMENDACIÓN, no se intenta otro candidato."""
        plan = _plan()
        registro = RegistroDeAccesos(plan)
        with registro.fase("final_evaluation"):
            registro.congelar("pipeline-B")
            resultado = evaluar_en_test(
                registro, "pipeline-B", _muestra_binaria(40, False),
                restricciones=[Restriccion(clave="accuracy", operador="min", valor=0.9)])
        self.assertEqual(resultado.resultado, "fallo_de_validacion")
        self.assertIsNotNone(resultado.motivo_del_fallo)
        self.assertIsNotNone(resultado.evaluacion)  # la medida se guarda igual

    def test_no_probar_candidatos_sin_reconocer_el_nuevo_uso(self):
        """El criterio literal: evaluar un SEGUNDO candidato en el mismo test
        nunca vuelve a dar independent_test -- lo pida o no quien llama."""
        plan = _plan()
        registro = RegistroDeAccesos(plan)
        with registro.fase("final_evaluation"):
            registro.congelar("pipeline-B")
            primero = evaluar_en_test(
                registro, "pipeline-B", _muestra_binaria(40, False),
                restricciones=[Restriccion(clave="accuracy", operador="min", valor=0.9)])
            self.assertEqual(primero.resultado, "fallo_de_validacion")

            registro.congelar("pipeline-C")
            segundo = evaluar_en_test(registro, "pipeline-C", _muestra_binaria(40, True))
        self.assertEqual(segundo.evaluacion.evidence, "repeated_test_use")

    def test_reentrenar_con_test_crea_version_sin_evaluacion_independiente(self):
        """Esta función YA EXISTÍA en 104-C0 (`version_tras_aprender_del_
        test`) -- este corte no la reescribe, verifica que satisface su
        propio criterio literal."""
        plan = _plan()
        registro = RegistroDeAccesos(plan)
        with registro.fase("final_evaluation"):
            registro.congelar("pipeline-A")
            resultado = evaluar_en_test(registro, "pipeline-A", _muestra_binaria(40, True))
        nueva = version_tras_aprender_del_test(
            resultado.evaluacion, "pipeline-A-v2", motivo_del_cambio={"es": "x", "en": "x"})
        self.assertEqual(nueva.evidence, "test_used_for_development")
        self.assertEqual(nueva.metrics, ())
        self.assertEqual(nueva.derives_from, resultado.evaluacion.evaluation_id)


class PromocionTest(unittest.TestCase):
    def test_promocion_conserva_historial_con_evidencia_comparable(self):
        plan = _plan()
        vigente = EvaluationResult(evaluation_id="eval-vigente", pipeline_digest="A",
                                   split_plan_digest=plan.digest(), evaluated_role="test",
                                   evidence="independent_test", metrics=())
        alternativo = EvaluationResult(evaluation_id="eval-alt", pipeline_digest="D",
                                       split_plan_digest=plan.digest(), evaluated_role="test",
                                       evidence="independent_test", metrics=())
        promovido = promover_candidato(vigente, alternativo)
        self.assertEqual(promovido.derives_from, "eval-vigente")
        self.assertEqual(promovido.pipeline_digest, "D")

    def test_promocion_rechaza_evidencia_no_comparable(self):
        """No se puede promocionar sustituyendo una evidencia independiente
        por una reutilizada (o al revés) como si fueran la misma categoría."""
        plan = _plan()
        vigente = EvaluationResult(evaluation_id="eval-vigente", pipeline_digest="A",
                                   split_plan_digest=plan.digest(), evaluated_role="test",
                                   evidence="independent_test", metrics=())
        alternativo_debil = EvaluationResult(evaluation_id="eval-debil", pipeline_digest="E",
                                             split_plan_digest=plan.digest(), evaluated_role="test",
                                             evidence="repeated_test_use", metrics=())
        with self.assertRaises(EsquemaInvalido):
            promover_candidato(vigente, alternativo_debil)


class ValidacionFinalEsquemaTest(unittest.TestCase):
    def _evaluacion(self):
        plan = _plan()
        return EvaluationResult(evaluation_id="eval-x", pipeline_digest="A",
                                split_plan_digest=plan.digest(), evaluated_role="test",
                                evidence="independent_test", metrics=())

    def test_fallo_sin_motivo_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            ValidacionFinal(resultado="fallo_de_validacion", evaluacion=self._evaluacion())

    def test_confirmado_con_motivo_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            ValidacionFinal(resultado="confirmado", evaluacion=self._evaluacion(),
                           motivo_del_fallo={"es": "x", "en": "x"})


if __name__ == "__main__":
    unittest.main()
