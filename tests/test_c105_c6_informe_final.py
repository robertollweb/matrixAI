# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C6 — evaluación final y reportes.

Criterio de terminado: «Modelos propio/nativo/exportado producen esquema
común. Una versión entrenada después con test no hereda evaluación
independiente. Reporte antiguo sigue legible con campos no disponibles.
Textos es/en y API reflejan estados inconclusos/indefinidos.»
"""
from __future__ import annotations

import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.esquemas import (
    EvaluationResult,
    SplitPlan,
    ValorDeMetrica,
    version_tras_aprender_del_test,
)
from matrixai.estudio.incertidumbre import Intervalo
from matrixai.estudio.informe_final import InformeDeEvaluacion


def _plan():
    assignments = {f"d{i}": "development" for i in range(50)}
    assignments.update({f"t{i}": "test" for i in range(20)})
    assignments.update({f"e{i}": "external_test" for i in range(10)})
    return SplitPlan(plan_id="p1", split_type="iid", assignments=assignments,
                     observation_id_field="id")


def _evaluacion(plan, *, role="test", evidence="independent_test", pipeline_digest="pipe-A"):
    return EvaluationResult(
        evaluation_id=f"eval-{pipeline_digest}", pipeline_digest=pipeline_digest,
        split_plan_digest=plan.digest(), evaluated_role=role, evidence=evidence,
        metrics=(ValorDeMetrica(metric_id="accuracy", formula_version="v1", value=0.9),))


def _intervalo():
    return Intervalo(metric_id="accuracy", design="iid", estimand="fixed_model_on_population",
                     resampling_unit="row", seed=1, n_resamples=100, level=0.95,
                     ci_low=0.85, ci_high=0.95)


class CriterioLiteralTest(unittest.TestCase):
    def test_incorpora_formulas_ic_calibracion_y_comparaciones(self):
        plan = _plan()
        informe = InformeDeEvaluacion(
            informe_id="informe-1", evaluacion=_evaluacion(plan), intervalos=(_intervalo(),),
            curva_de_fiabilidad={"ece": 0.02, "metodo_de_bins": "ancho_igual"},
            recalibracion={"a": 0.1, "b": 1.1},
            comparaciones=({"metric_id": "accuracy", "veredicto": "mejora"},))
        payload = informe.a_json()
        self.assertEqual(payload["evaluacion"]["metrics"][0]["metric_id"], "accuracy")
        self.assertEqual(payload["intervalos"][0]["ci_low"], 0.85)
        self.assertEqual(payload["curva_de_fiabilidad"]["ece"], 0.02)
        self.assertEqual(payload["recalibracion"]["a"], 0.1)
        self.assertEqual(payload["comparaciones"][0]["veredicto"], "mejora")

    def test_ningun_campo_de_equidad_o_aplicabilidad_clinica(self):
        """El criterio literal: «no afirmar equidad o aplicabilidad clínica
        por disponer de subgrupos» -- comprobado de la forma estructural:
        el esquema no tiene NINGÚN campo para escribir esa afirmación."""
        for clave in InformeDeEvaluacion.CLAVES:
            self.assertNotIn("equi", clave.lower())
            self.assertNotIn("fair", clave.lower())
            self.assertNotIn("clinic", clave.lower())

    def test_distingue_seleccion_interna_prueba_final_cohorte_externa(self):
        plan = _plan()
        interna = InformeDeEvaluacion(
            informe_id="i1", evaluacion=_evaluacion(plan, role="development",
                                                    evidence="development_estimate"))
        final = InformeDeEvaluacion(informe_id="i2", evaluacion=_evaluacion(plan, role="test"))
        externa = InformeDeEvaluacion(
            informe_id="i3", evaluacion=_evaluacion(plan, role="external_test",
                                                     evidence="external_validation"))
        self.assertEqual(interna.evaluacion.evaluated_role, "development")
        self.assertEqual(final.evaluacion.evaluated_role, "test")
        self.assertEqual(externa.evaluacion.evaluated_role, "external_test")

    def test_modelos_propio_nativo_exportado_producen_esquema_comun(self):
        plan = _plan()
        for motor in ("matrixai.dense.torch_cpu", "sklearn.lineal", "onnx.exportado"):
            informe = InformeDeEvaluacion(
                informe_id=f"informe-{motor}",
                evaluacion=_evaluacion(plan, pipeline_digest=motor))
            payload = informe.a_json()
            self.assertEqual(set(payload) - {"schema", "schema_version"}, set(InformeDeEvaluacion.CLAVES))

    def test_reentrenar_con_test_no_hereda_evaluacion_independiente(self):
        """`version_tras_aprender_del_test` (104-C0) ya lo hace -- este
        informe tiene que reflejarlo fielmente, no reescribir la etiqueta."""
        plan = _plan()
        original = _evaluacion(plan)
        nueva = version_tras_aprender_del_test(
            original, "pipe-A-v2", motivo_del_cambio={"es": "x", "en": "x"})
        informe = InformeDeEvaluacion(informe_id="informe-nuevo", evaluacion=nueva)
        self.assertEqual(informe.evaluacion.evidence, "test_used_for_development")
        self.assertNotEqual(informe.evaluacion.evidence, "independent_test")

    def test_reporte_antiguo_sigue_legible_con_campos_no_disponibles(self):
        plan = _plan()
        informe = InformeDeEvaluacion(informe_id="informe-1", evaluacion=_evaluacion(plan),
                                      intervalos=(_intervalo(),))
        payload_antiguo = dict(informe.a_json())
        del payload_antiguo["intervalos"]
        del payload_antiguo["curva_de_fiabilidad"]
        del payload_antiguo["recalibracion"]
        del payload_antiguo["comparaciones"]
        leido = InformeDeEvaluacion.desde_json(payload_antiguo)
        self.assertEqual(leido.intervalos, ())
        self.assertIsNone(leido.curva_de_fiabilidad)
        self.assertEqual(leido.evaluacion.pipeline_digest, "pipe-A")


class RoundTripTest(unittest.TestCase):
    def test_digest_estable_en_el_round_trip(self):
        plan = _plan()
        informe = InformeDeEvaluacion(
            informe_id="informe-1", evaluacion=_evaluacion(plan), intervalos=(_intervalo(),))
        recuperado = InformeDeEvaluacion.desde_json(informe.a_json())
        self.assertEqual(informe.digest(), recuperado.digest())

    def test_intervalo_se_reconstruye_tipado(self):
        plan = _plan()
        informe = InformeDeEvaluacion(
            informe_id="informe-1", evaluacion=_evaluacion(plan), intervalos=(_intervalo(),))
        recuperado = InformeDeEvaluacion.desde_json(informe.a_json())
        self.assertIsInstance(recuperado.intervalos[0], Intervalo)
        self.assertEqual(recuperado.intervalos[0].ci_low, 0.85)


class EsquemaTest(unittest.TestCase):
    def test_evaluacion_que_no_es_evaluationresult_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            InformeDeEvaluacion(informe_id="x", evaluacion={"no": "es un EvaluationResult"})

    def test_intervalo_que_no_es_intervalo_se_rechaza(self):
        plan = _plan()
        with self.assertRaises(EsquemaInvalido):
            InformeDeEvaluacion(informe_id="x", evaluacion=_evaluacion(plan),
                               intervalos=({"ci_low": 0.1},))

    def test_clave_extra_en_desde_json_se_rechaza(self):
        plan = _plan()
        informe = InformeDeEvaluacion(informe_id="x", evaluacion=_evaluacion(plan))
        payload = dict(informe.a_json())
        payload["campo_inventado"] = "algo"
        with self.assertRaises(EsquemaInvalido):
            InformeDeEvaluacion.desde_json(payload)


if __name__ == "__main__":
    unittest.main()
