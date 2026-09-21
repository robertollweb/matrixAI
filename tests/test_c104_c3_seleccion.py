# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C3 — requisitos y selección.

Criterio de terminado: «Siempre-positivo no gana por sensibilidad sin
superar las restricciones; siempre-negativo y baseline veloz tampoco eluden
mínimos. Un modelo descargado que opera offline pasa la restricción local.
El motivo distingue mejora demostrada de elección operativa ante
incertidumbre.»
"""
from __future__ import annotations

import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.comparaciones import ComparacionEmparejada
from matrixai.estudio.esquemas import EvaluationResult, Restriccion
from matrixai.estudio.incertidumbre import Intervalo
from matrixai.estudio.metricas import ValorDeMetrica
from matrixai.estudio.seleccion import POLITICA_UTILIDAD_MAXIMA, seleccionar


def _ev(pipeline_digest, role="development", evidence="development_estimate", **metricas):
    return EvaluationResult(
        evaluation_id=f"eval-{pipeline_digest}", pipeline_digest=pipeline_digest,
        split_plan_digest="split-1", evaluated_role=role, evidence=evidence,
        metrics=tuple(ValorDeMetrica(metric_id=k, formula_version="v1", value=v)
                     for k, v in metricas.items()))


def _comparacion(veredicto, ci_low=-0.02, ci_high=0.12):
    return ComparacionEmparejada(
        metric_id="sensitivity", diseno="iid", estimando="fixed_model_on_population",
        veredicto=veredicto, margen_equivalencia=None, diferencia_puntual=0.05,
        intervalo=Intervalo(metric_id="sensitivity", design="iid",
                            estimand="fixed_model_on_population", resampling_unit="row",
                            seed=1, n_resamples=100, level=0.95, ci_low=ci_low, ci_high=ci_high))


class CriterioLiteralTest(unittest.TestCase):
    def test_siempre_positivo_no_gana_por_sensibilidad_sin_superar_restricciones(self):
        evaluaciones = {
            "siempre_positivo": _ev("p1", sensitivity=1.0, specificity=0.0),
            "real": _ev("p2", sensitivity=0.85, specificity=0.85),
        }
        r = seleccionar(evaluaciones, restricciones=[Restriccion(clave="specificity", operador="min", valor=0.5)],
                        metric_id_calidad="sensitivity", decision_id="d1", split_plan_digest="split-1")
        self.assertEqual(r.outcome, "selected")
        self.assertEqual(r.chosen_candidate, "real")
        self.assertEqual({x["candidate"] for x in r.rejected}, {"siempre_positivo"})

    def test_siempre_negativo_y_baseline_veloz_tampoco_eluden_minimos(self):
        """"Veloz" no compra una excepción a los mínimos de calidad: un
        candidato rápido que no alcanza sensibilidad/especificidad se
        descarta igual que uno lento."""
        evaluaciones = {
            "siempre_negativo": _ev("p3", sensitivity=0.0, specificity=1.0),
            "baseline_veloz": _ev("p4", sensitivity=0.3, specificity=0.9),
            "real": _ev("p5", sensitivity=0.85, specificity=0.85),
        }
        restricciones = [Restriccion(clave="sensitivity", operador="min", valor=0.5),
                        Restriccion(clave="specificity", operador="min", valor=0.5)]
        r = seleccionar(evaluaciones, restricciones=restricciones, metric_id_calidad="sensitivity",
                        decision_id="d2", split_plan_digest="split-1")
        self.assertEqual(r.outcome, "selected")
        self.assertEqual(r.chosen_candidate, "real")
        self.assertEqual({x["candidate"] for x in r.rejected}, {"siempre_negativo", "baseline_veloz"})

    def test_modelo_offline_pasa_la_restriccion_local(self):
        evaluaciones = {
            "local_ok": _ev("p6", sensitivity=0.8, specificity=0.8),
            "necesita_internet": _ev("p7", sensitivity=0.9, specificity=0.9),
        }
        r = seleccionar(evaluaciones, restricciones=[Restriccion(clave="solo_local", operador="boolean", valor=True)],
                        metric_id_calidad="sensitivity", decision_id="d3", split_plan_digest="split-1",
                        necesita_red={"local_ok": False, "necesita_internet": True})
        self.assertEqual(r.outcome, "selected")
        self.assertEqual(r.chosen_candidate, "local_ok")

    def test_motivo_distingue_mejora_demostrada_de_eleccion_operativa(self):
        evaluaciones = {
            "lider": _ev("p8", sensitivity=0.9, specificity=0.9),
            "segundo": _ev("p9", sensitivity=0.85, specificity=0.85),
        }
        r_mejora = seleccionar(evaluaciones, restricciones=[], metric_id_calidad="sensitivity",
                               decision_id="d4a", split_plan_digest="split-1",
                               comparacion_lider=_comparacion("mejora", ci_low=0.01, ci_high=0.09))
        r_inconcluso = seleccionar(evaluaciones, restricciones=[], metric_id_calidad="sensitivity",
                                   decision_id="d4b", split_plan_digest="split-1",
                                   comparacion_lider=_comparacion("inconcluso"))
        self.assertIn("mejora demostrada", r_mejora.reason["es"])
        self.assertNotIn("mejora demostrada", r_inconcluso.reason["es"])
        # Y la mitad positiva: cuando NO hay mejora demostrada, la frase tiene
        # que decirlo, no limitarse a callar.
        self.assertIn("NO demuestra una mejora", r_inconcluso.reason["es"])
        self.assertIn("incertidumbre", r_inconcluso.reason["es"])

    def test_la_eleccion_bajo_incertidumbre_NO_INVENTA_un_criterio_operativo(self):
        """DECLARAR LO QUE PASÓ, NO LO QUE SE PIDIÓ — 2026-09-14.

        Esta frase decía que el ganador «se elige por criterio operativo
        (sencillez o recursos, DECLARADO)». **No existe ningún criterio de
        sencillez ni de recursos en toda la selección**: el ganador sale de un
        `functools.reduce` sobre la métrica de calidad y no hay ni un
        desempate por otra cosa. La frase inventaba el motivo, y lo inventaba
        **justo cuando la evidencia es más débil** — hacía sonar a decisión de
        ingeniería deliberada lo que es «salió un número más alto y la
        diferencia no se sostiene».

        LAS DOS MITADES, y la segunda es la que tiene dientes de verdad:
        la frase no lo afirma, **y el código sigue sin tener ese criterio**.
        Se mide sobre el FUENTE porque si algún día se añade un desempate por
        recursos, esta prueba se pone roja y entonces la frase vieja sería la
        correcta — que es exactamente cuando hay que volver a mirarla.
        """
        evaluaciones = {
            "lider": _ev("p8", sensitivity=0.9, specificity=0.9),
            "segundo": _ev("p9", sensitivity=0.85, specificity=0.85),
        }
        r = seleccionar(evaluaciones, restricciones=[], metric_id_calidad="sensitivity",
                        decision_id="d4c", split_plan_digest="split-1",
                        comparacion_lider=_comparacion("inconcluso"))
        for inventado in ("criterio operativo", "sencillez", "recursos",
                          "operational grounds", "simplicity"):
            self.assertNotIn(inventado, r.reason["es"] + " " + r.reason["en"],
                             f"la frase afirma «{inventado}», y eso no es lo que decidió")

        import inspect
        from matrixai.estudio import seleccion as modulo
        fuente = inspect.getsource(modulo)
        for palabra in ("sencillez", "simplicidad", "recursos_del_candidato", "desempate"):
            self.assertNotIn(palabra, fuente,
                             f"«{palabra}» aparece en seleccion.py: si ahora SÍ hay un "
                             "criterio operativo, la frase de la decisión tiene que "
                             "volver a contarlo")


class LaFraseNoNombraAlCandidatoTest(unittest.TestCase):
    """`nombrar_candidatos=False` — decisión de Roberto del 2026-09-21: «el
    motor en grande, el id pequeño».

    En el Studio el ganador es el reajuste final, `lightgbm-seleccion`, y la
    pantalla decía «Campeón: lightgbm-seleccion» y «lightgbm-seleccion tiene una
    mejora demostrada…». Ahora la pantalla pone el nombre del motor de su propio
    catálogo, UNA vez, y la frase dice solo el porqué con «el campeón» de sujeto
    —la misma palabra del rótulo—. Si la frase siguiera nombrándolo, habría dos
    grafías del mismo nombre en la misma sección.

    Tres cosas que no pueden cambiar con esto, y cada una con su prueba: que
    por omisión todo siga igual (Fase 0 y los benchmarks llaman a esta función),
    que `chosen_candidate` siga siendo el id (es lo que se audita), y que la
    RAMA que corresponde siga siendo la misma (una mejora demostrada no puede
    volverse otra frase por no nombrar a nadie)."""

    EVALUACIONES = {
        "lightgbm-seleccion": _ev("p8", sensitivity=0.9, specificity=0.9),
        "baseline-seleccion": _ev("p9", sensitivity=0.85, specificity=0.85),
    }

    def _decidir(self, **kw):
        return seleccionar(self.EVALUACIONES, restricciones=[], metric_id_calidad="sensitivity",
                           decision_id="d-nombre", split_plan_digest="split-1", **kw)

    def test_la_razon_y_los_rechazados_no_llevan_el_id_en_ningun_idioma(self):
        for veredicto in ("mejora", "inconcluso", None):
            kw = {"comparacion_lider": _comparacion(veredicto, ci_low=0.01, ci_high=0.09)} if veredicto else {}
            d = self._decidir(nombrar_candidatos=False, **kw)
            for idioma in ("es", "en"):
                self.assertNotIn("seleccion", d.reason[idioma], (veredicto, idioma, d.reason[idioma]))
                self.assertNotIn("lightgbm", d.reason[idioma], (veredicto, idioma))
                for r in d.rejected:
                    self.assertNotIn("lightgbm", r["reason"][idioma], (veredicto, idioma, r))
            # Con sujeto, y el MISMO que el rótulo: un «Tiene…» suelto se leería
            # con el sujeto de la frase de encima.
            self.assertTrue(d.reason["es"].startswith("El campeón"), d.reason["es"])
            self.assertTrue(d.reason["en"].startswith("The champion"), d.reason["en"])

    def test_el_id_SIGUE_en_chosen_candidate_que_es_lo_que_se_audita(self):
        d = self._decidir(nombrar_candidatos=False,
                          comparacion_lider=_comparacion("mejora", ci_low=0.01, ci_high=0.09))
        self.assertEqual(d.chosen_candidate, "lightgbm-seleccion")
        self.assertEqual({r["candidate"] for r in d.rejected}, {"baseline-seleccion"})

    def test_la_rama_no_cambia_por_no_nombrar(self):
        """Callar el nombre no puede convertir una elección bajo incertidumbre
        en una mejora demostrada, ni al revés: es la parte que más importa de
        esa frase y la que menos se ve."""
        mejora = self._decidir(nombrar_candidatos=False,
                               comparacion_lider=_comparacion("mejora", ci_low=0.01, ci_high=0.09))
        dudosa = self._decidir(nombrar_candidatos=False, comparacion_lider=_comparacion("inconcluso"))
        self.assertIn("mejora demostrada", mejora.reason["es"])
        self.assertIn("NO demuestra una mejora", dudosa.reason["es"])
        self.assertIn("incertidumbre", dudosa.reason["es"])

    def test_por_omision_todo_sigue_como_antes(self):
        """Fase 0 y los benchmarks llaman a `seleccionar()` sin la bandera: para
        ellos no puede cambiar ni una palabra."""
        d = self._decidir(comparacion_lider=_comparacion("mejora", ci_low=0.01, ci_high=0.09))
        self.assertTrue(d.reason["es"].startswith("lightgbm-seleccion tiene una mejora"), d.reason["es"])


class ResultadosNoSeleccionadosTest(unittest.TestCase):
    def test_no_feasible_model_con_evidencia_completa(self):
        """Todos medidos, ninguno supera las restricciones -- veredicto
        real, no ausencia de veredicto."""
        evaluaciones = {
            "a": _ev("p1", sensitivity=0.2, specificity=0.9),
            "b": _ev("p2", sensitivity=0.9, specificity=0.2),
        }
        restricciones = [Restriccion(clave="sensitivity", operador="min", valor=0.5),
                        Restriccion(clave="specificity", operador="min", valor=0.5)]
        r = seleccionar(evaluaciones, restricciones=restricciones, metric_id_calidad="sensitivity",
                        decision_id="d5", split_plan_digest="split-1")
        self.assertEqual(r.outcome, "no_feasible_model")
        self.assertIsNone(r.chosen_candidate)

    def test_insufficient_evidence_por_restriccion_no_medida(self):
        """Un candidato sin la métrica de la restricción medida no puede
        juzgarse: la ausencia de dato NO es lo mismo que "no cumple"."""
        evaluaciones = {
            "sin_specificity_medida": _ev("p1", sensitivity=0.9),
            "falla_restriccion": _ev("p2", sensitivity=0.9, specificity=0.1),
        }
        restricciones = [Restriccion(clave="specificity", operador="min", valor=0.5)]
        r = seleccionar(evaluaciones, restricciones=restricciones, metric_id_calidad="sensitivity",
                        decision_id="d6", split_plan_digest="split-1")
        self.assertEqual(r.outcome, "insufficient_evidence")
        self.assertIsNone(r.chosen_candidate)

    def test_insufficient_evidence_por_calidad_no_medida(self):
        """Los dos superan las restricciones pero ninguno tiene la métrica
        de calidad medida -- viable no es lo mismo que ordenable."""
        evaluaciones = {
            "a": _ev("p1", specificity=0.9),
            "b": _ev("p2", specificity=0.8),
        }
        r = seleccionar(evaluaciones, restricciones=[Restriccion(clave="specificity", operador="min", valor=0.5)],
                        metric_id_calidad="sensitivity", decision_id="d7", split_plan_digest="split-1")
        self.assertEqual(r.outcome, "insufficient_evidence")


class WiringTest(unittest.TestCase):
    def test_seleccionar_con_evaluacion_de_test_se_rechaza(self):
        evaluaciones = {"a": _ev("p1", role="test", evidence="independent_test",
                                sensitivity=0.9, specificity=0.9)}
        with self.assertRaises(EsquemaInvalido):
            seleccionar(evaluaciones, restricciones=[], metric_id_calidad="sensitivity",
                       decision_id="d8", split_plan_digest="split-1")

    def test_un_solo_candidato_viable_no_necesita_comparacion(self):
        evaluaciones = {"unico": _ev("p1", sensitivity=0.9, specificity=0.9)}
        r = seleccionar(evaluaciones, restricciones=[], metric_id_calidad="sensitivity",
                        decision_id="d9", split_plan_digest="split-1")
        self.assertEqual(r.outcome, "selected")
        self.assertIn("mayor utilidad medida", r.reason["es"])

    def test_rechazados_por_utilidad_traen_al_ganador_en_el_motivo(self):
        evaluaciones = {
            "lider": _ev("p1", sensitivity=0.9, specificity=0.9),
            "segundo": _ev("p2", sensitivity=0.85, specificity=0.85),
        }
        r = seleccionar(evaluaciones, restricciones=[], metric_id_calidad="sensitivity",
                        decision_id="d10", split_plan_digest="split-1")
        rechazo = next(x for x in r.rejected if x["candidate"] == "segundo")
        self.assertIn("lider", rechazo["reason"]["es"])

    def test_politica_por_omision_es_utilidad_maxima(self):
        evaluaciones = {"unico": _ev("p1", sensitivity=0.9)}
        r = seleccionar(evaluaciones, restricciones=[], metric_id_calidad="sensitivity",
                        decision_id="d11", split_plan_digest="split-1")
        self.assertEqual(r.policy, POLITICA_UTILIDAD_MAXIMA)


if __name__ == "__main__":
    unittest.main()
