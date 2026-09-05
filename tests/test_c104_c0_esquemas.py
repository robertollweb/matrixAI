# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C0 — LOS ESQUEMAS COMUNES Y EL PROTOCOLO DE EVALUACIÓN.

Lo que se prueba aquí no es que unos `dataclass` guarden campos: es que **no se
pueda escribir un estudio deshonesto**. Cada clase de este fichero existe por un
defecto concreto que el contrato nombra:

* una partición que mete al mismo paciente en desarrollo y en prueba invalida la
  prueba, y comprobar que los hashes son distintos NO lo detecta;
* un registro de predicciones sin `y_true` no vale para recomputar una métrica
  que lo necesita — y el esquema tiene que DECIRLO, no rellenarlo;
* un estado `completed` sin candidato disfraza un aborto de éxito;
* una decisión de selección apoyada en el test convierte la prueba reservada en
  parte del entrenamiento;
* y lo que de verdad sostiene todo lo anterior: un registro de accesos que
  DENIEGA leer el test durante desarrollo, selección o calibración, y que anota
  el intento denegado en vez de contar una historia limpia.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from matrixai.estudio import (
    ESTUDIO_SCHEMA_VERSION,
    EsquemaInvalido,
    EvaluationResult,
    FitResult,
    FittedPipelineSpec,
    FugaDeTest,
    Horizonte,
    MetricSpec,
    MigracionImposible,
    PoliticaDeDecision,
    PredictionRecord,
    ProblemSpec,
    ProtocoloRoto,
    Recursos,
    RegistroDeAccesos,
    RegistroDeMigraciones,
    Restriccion,
    SelectionDecision,
    SplitPlan,
    ValorDeMetrica,
    version_tras_aprender_del_test,
)
from matrixai.estudio.accesos import LectorDeParticiones
from matrixai.estudio.migracion import MIGRACIONES
from matrixai.estudio.textos import IDIOMAS, MOTIVOS, huecos_de, motivo

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE = RAIZ / "matrixai" / "estudio"


# ---------------------------------------------------------------------------
# Fixtures válidos: se construyen una vez y se ensucian a propósito en cada caso
# ---------------------------------------------------------------------------

def _problema(**cambios) -> ProblemSpec:
    base = dict(
        problem_id="ingreso-uci",
        target="ingreso_uci",
        task="binary_classification",
        observation_unit="un episodio de urgencias",
        classes=("no", "si"),
        positive_label="si",
        predictors=("edad", "lactato", "centro"),
        predictor_availability={"edad": "at_prediction_time",
                                "lactato": "at_prediction_time"},
        prediction_time="al triaje",
        horizon=Horizonte(magnitud=48.0, unidad="hours"),
        intended_use="apoyo a la decisión de ingreso; no sustituye al criterio clínico",
        constraints=(Restriccion(clave="min_especificidad", operador="min", valor=0.8),
                     Restriccion(clave="solo_local", operador="boolean", valor=True)),
    )
    base.update(cambios)
    return ProblemSpec(**base)


def _plan(**cambios) -> SplitPlan:
    """Seis observaciones de tres pacientes: dos filas por paciente."""
    base = dict(
        plan_id="plan-1",
        split_type="groups",
        observation_id_field="episodio_id",
        unit_id_field="paciente_id",
        assignments={"o1": "development", "o2": "development",
                     "o3": "selection", "o4": "calibration",
                     "o5": "test", "o6": "test"},
        units={"o1": "p1", "o2": "p1", "o3": "p2", "o4": "p3",
               "o5": "p4", "o6": "p4"},
        folds=5, repeats=2, seed=20260905,
    )
    base.update(cambios)
    return SplitPlan(**base)


def _plan_simple(**cambios) -> SplitPlan:
    base = dict(plan_id="plan-iid", split_type="iid", observation_id_field="id",
                assignments={"o1": "development", "o2": "selection",
                             "o3": "calibration", "o4": "test"})
    base.update(cambios)
    return SplitPlan(**base)


def _metrica(**cambios) -> MetricSpec:
    base = dict(metric_id="log_loss", formula_version="1.0",
                estimand="fixed_model_on_population",
                requires=("y_true", "probabilities", "classes"),
                direction="lower_is_better")
    base.update(cambios)
    return MetricSpec(**base)


def _prediccion(**cambios) -> PredictionRecord:
    base = dict(row_id="o5", candidate="cand-1", resampling_unit="p4",
                y_true="si", label="si", probabilities=(0.25, 0.75),
                classes=("no", "si"), fold_model="f3", repeat=1, fold=3, role="test")
    base.update(cambios)
    return PredictionRecord(**base)


def _pipeline(**cambios) -> FittedPipelineSpec:
    base = dict(pipeline_id="pl-1", problem_id="ingreso-uci", candidate="cand-1",
                engine="densa-propia", predictor={"kind": "dense", "digest": "sha256:aa"},
                trained_on_roles=("development", "selection"),
                split_plan_digest="a" * 64,
                preparation={"imputacion": "mediana+indicador"},
                calibrator={"kind": "logistic", "digest": "sha256:bb"},
                decision_policy=PoliticaDeDecision(threshold=0.42,
                                                   scale="calibrated_probability",
                                                   positive_label="si",
                                                   cost_false_negative=10.0,
                                                   cost_false_positive=1.0))
    base.update(cambios)
    return FittedPipelineSpec(**base)


def _motivo(texto_es: str, texto_en: str) -> dict[str, str]:
    return {"es": texto_es, "en": texto_en}


def _evaluacion(**cambios) -> EvaluationResult:
    base = dict(evaluation_id="ev-1", pipeline_digest="b" * 64,
                split_plan_digest="a" * 64, evaluated_role="test",
                evidence="independent_test",
                metrics=(ValorDeMetrica(metric_id="auroc", formula_version="1.0",
                                        value=0.81, n_observations=200, n_units=180),))
    base.update(cambios)
    return EvaluationResult(**base)


# ---------------------------------------------------------------------------
# ProblemSpec
# ---------------------------------------------------------------------------

class ElProblemaSeDeclaraEnteroTest(unittest.TestCase):
    """Un objetivo a medias no arranca un estudio: sin clase positiva no hay
    sensibilidad ni umbral, y elegirla por orden alfabético sería inventarse la
    mitad del problema."""

    def test_el_problema_completo_vale_y_conserva_lo_declarado(self):
        problema = _problema()
        self.assertEqual(problema.task, "binary_classification")
        self.assertEqual(problema.classes, ("no", "si"))
        self.assertEqual(problema.positive_label, "si")
        self.assertEqual(problema.horizon.unidad, "hours")
        self.assertEqual(len(problema.restricciones_obligatorias()), 2)

    def test_un_predictor_sin_disponibilidad_declarada_se_ENUMERA_no_se_rellena(self):
        """103 invariante 5: no se infiere disponibilidad por el nombre ni por el
        orden del CSV. Rellenar aquí con `unknown` sería inferirla por omisión."""
        problema = _problema()
        self.assertEqual(problema.sin_disponibilidad_declarada(), ("centro",))
        self.assertNotIn("centro", problema.predictor_availability)

    def test_unknown_declarado_NO_es_lo_mismo_que_no_declarado(self):
        problema = _problema(predictor_availability={"edad": "at_prediction_time",
                                                     "lactato": "at_prediction_time",
                                                     "centro": "unknown"})
        self.assertEqual(problema.sin_disponibilidad_declarada(), ())
        self.assertEqual(problema.predictor_availability["centro"], "unknown")

    def test_una_binaria_sin_clase_positiva_no_se_construye(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(positive_label=None)
        self.assertEqual(e.exception.clave, "binaria_sin_clase_positiva")

    def test_una_clase_positiva_que_no_esta_entre_las_clases_no_se_construye(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(positive_label="quizas")
        self.assertEqual(e.exception.clave, "clase_positiva_fuera_de_clases")

    def test_una_regresion_no_tiene_clases_ni_clase_positiva(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(task="regression", classes=("a", "b"), positive_label=None)
        self.assertEqual(e.exception.clave, "clases_en_regresion")
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(task="regression", classes=None, positive_label="si")
        self.assertEqual(e.exception.clave, "clase_positiva_en_regresion")
        # Y la regresión bien declarada SÍ vale.
        regresion = _problema(task="regression", classes=None, positive_label=None)
        self.assertIsNone(regresion.classes)

    def test_una_binaria_con_tres_clases_no_es_binaria(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(classes=("a", "b", "c"), positive_label="a")
        self.assertEqual(e.exception.clave, "binaria_con_otro_numero_de_clases")

    def test_una_tarea_que_el_producto_no_soporta_se_rechaza_con_la_lista(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(task="ranking")
        self.assertEqual(e.exception.clave, "no_es_del_vocabulario")
        self.assertIn("binary_classification", e.exception.en)

    def test_un_horizonte_sin_momento_de_prediccion_no_dice_desde_cuando(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(prediction_time=None)
        self.assertEqual(e.exception.clave, "horizonte_sin_momento")

    def test_disponibilidad_de_una_columna_que_no_es_predictor(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(predictor_availability={"tension": "at_prediction_time"})
        self.assertEqual(e.exception.clave, "disponibilidad_de_desconocido")

    def test_un_uso_previsto_vacio_no_es_un_uso_previsto(self):
        """`None` dice «nadie lo ha escrito» y es una respuesta; `""` finge que
        está y está vacío."""
        self.assertIsNone(_problema(intended_use=None).intended_use)
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(intended_use="   ")
        self.assertEqual(e.exception.clave, "no_es_texto")


# ---------------------------------------------------------------------------
# SplitPlan
# ---------------------------------------------------------------------------

class LaParticionAislaLasUnidadesTest(unittest.TestCase):
    """El fallo que el 103-C4 persigue: dos filas del mismo paciente, una en
    desarrollo y otra en prueba. Los hashes de los dos conjuntos son distintos y
    todo parece correcto; lo que hay que comprobar es la INTERSECCIÓN."""

    def test_el_plan_bueno_reparte_y_se_puede_consultar(self):
        plan = _plan()
        self.assertEqual(plan.observaciones_del_rol("test"), ("o5", "o6"))
        self.assertEqual(plan.unidades_del_rol("test"), ("p4",))
        self.assertEqual(plan.roles_presentes(),
                         ("development", "selection", "calibration", "test"))

    def test_el_mismo_paciente_en_desarrollo_y_en_prueba_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _plan(units={"o1": "p1", "o2": "p1", "o3": "p2", "o4": "p3",
                         "o5": "p1", "o6": "p4"})
        self.assertEqual(e.exception.clave, "unidad_en_dos_roles")
        self.assertIn("p1", e.exception.es)
        self.assertIn("development", e.exception.en)

    def test_los_hashes_distintos_NO_bastan_y_por_eso_se_miran_los_conjuntos(self):
        """Prueba de que la comprobación es de conjuntos: el plan contaminado y el
        limpio tienen digests distintos, así que comparar hashes no habría
        detectado nada."""
        limpio = _plan()
        contaminado_asignaciones = dict(limpio.assignments)
        contaminado_unidades = dict(limpio.units)
        contaminado_unidades["o5"] = "p1"
        with self.assertRaises(EsquemaInvalido):
            _plan(assignments=contaminado_asignaciones, units=contaminado_unidades)
        self.assertNotEqual(limpio.digest(), _plan(seed=1).digest())

    def test_un_plan_sin_test_no_se_construye_salvo_anidada_declarada(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _plan_simple(assignments={"o1": "development", "o2": "selection"})
        self.assertEqual(e.exception.clave, "particion_sin_test")
        anidado = _plan_simple(assignments={"o1": "development", "o2": "selection"},
                               nested_evaluation=True)
        self.assertTrue(anidado.nested_evaluation)

    def test_una_particion_por_grupos_sin_unidades_no_aisla_nada(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _plan(units=None)
        self.assertEqual(e.exception.clave, "grupos_sin_unidad")

    def test_media_declaracion_de_unidades_tampoco_aisla(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _plan(units={"o1": "p1", "o2": "p1", "o3": "p2"})
        self.assertEqual(e.exception.clave, "observacion_sin_unidad")

    def test_una_particion_temporal_sin_columna_de_tiempo_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _plan_simple(split_type="temporal")
        self.assertEqual(e.exception.clave, "temporal_sin_tiempo")
        temporal = _plan_simple(split_type="temporal", time_column="fecha_ingreso",
                                gap=Horizonte(magnitud=7.0, unidad="days"))
        self.assertEqual(temporal.gap.unidad, "days")

    def test_un_pliegue_no_es_validacion_cruzada(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _plan(folds=1)
        self.assertEqual(e.exception.clave, "pliegues_insuficientes")

    def test_una_semilla_booleana_no_es_una_semilla(self):
        """`isinstance(True, int)` es `True` en Python, y `True` no es una semilla."""
        with self.assertRaises(EsquemaInvalido) as e:
            _plan(seed=True)
        self.assertEqual(e.exception.clave, "no_es_entero")

    def test_un_rol_inventado_se_rechaza_con_la_lista_de_roles(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _plan_simple(assignments={"o1": "entrenamiento", "o2": "test"})
        self.assertEqual(e.exception.clave, "no_es_del_vocabulario")
        self.assertIn("external_test", e.exception.en)

    def test_el_orden_de_construccion_no_cambia_el_digest(self):
        uno = _plan_simple(assignments={"o1": "development", "o2": "selection",
                                        "o3": "calibration", "o4": "test"})
        otro = _plan_simple(assignments={"o4": "test", "o3": "calibration",
                                         "o2": "selection", "o1": "development"})
        self.assertEqual(uno.digest(), otro.digest())
        self.assertEqual(uno.digest_de_asignaciones(), otro.digest_de_asignaciones())
        self.assertEqual(len(uno.digest()), 64)


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------

class UnaPrediccionDiceLoQueLeFaltaTest(unittest.TestCase):
    """Criterio 2 del corte: un registro sin `y_true`, sin clases o sin partición
    NO vale para recomputar la métrica que los necesita — y el esquema lo dice,
    no lo rellena."""

    def test_el_registro_completo_vale_para_su_metrica(self):
        aptitud = _prediccion().apto_para_recomputar(_metrica())
        self.assertTrue(aptitud.apto)
        self.assertEqual(aptitud.faltan, ())
        self.assertIsNone(aptitud.motivo)

    def test_sin_y_true_no_vale_Y_NO_SE_RELLENA(self):
        registro = _prediccion(y_true=None)
        aptitud = registro.apto_para_recomputar(_metrica())
        self.assertFalse(aptitud.apto)
        self.assertEqual(aptitud.faltan, ("y_true",))
        self.assertIn("y_true", aptitud.motivo["es"])
        self.assertIn("log_loss", aptitud.motivo["en"])
        # Lo que de verdad se prueba: preguntar no ha rellenado nada.
        self.assertIsNone(registro.y_true)
        self.assertIsNone(registro.a_json()["y_true"])

    def test_sin_clases_ni_particion_enumera_TODO_lo_que_falta(self):
        """Y enumera lo que falta DE VERDAD: `positive_label` no está en la lista
        porque la métrica sí la declara. Lo que impide la correspondencia son las
        clases del registro, y eso ya está dicho una vez — repetirlo como dos
        carencias haría creer que hay dos cosas que arreglar."""
        registro = PredictionRecord(row_id="o9", candidate="cand-1", label="si")
        metrica = _metrica(metric_id="auroc", direction="higher_is_better",
                           requires=("y_true", "scores", "classes", "positive_label",
                                     "split_role", "resampling_unit"),
                           positive_label="si")
        aptitud = registro.apto_para_recomputar(metrica)
        self.assertFalse(aptitud.apto)
        self.assertEqual(set(aptitud.faltan),
                         {"y_true", "scores", "classes", "split_role",
                          "resampling_unit"})
        self.assertNotIn("positive_label", aptitud.faltan)
        for idioma in IDIOMAS:
            self.assertTrue(aptitud.motivo[idioma].strip())

    def test_una_clase_positiva_que_no_esta_en_las_clases_del_registro_falta(self):
        registro = _prediccion()
        metrica = _metrica(metric_id="ppv", direction="higher_is_better",
                           requires=("y_true", "labels", "positive_label"),
                           positive_label="POSITIVO")
        self.assertEqual(registro.apto_para_recomputar(metrica).faltan,
                         ("positive_label",))

    def test_un_registro_sin_ninguna_salida_no_es_una_prediccion(self):
        with self.assertRaises(EsquemaInvalido) as e:
            PredictionRecord(row_id="o1", candidate="c1")
        self.assertEqual(e.exception.clave, "prediccion_sin_salida")

    def test_probabilidades_sin_clases_no_se_saben_de_quien_son(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _prediccion(classes=None)
        self.assertEqual(e.exception.clave, "columnas_sin_clases")

    def test_probabilidades_desalineadas_con_las_clases(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _prediccion(probabilities=(0.2, 0.3, 0.5), classes=("no", "si"))
        self.assertEqual(e.exception.clave, "columnas_desalineadas")

    def test_media_distribucion_no_se_recompone(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _prediccion(probabilities=(0.3, 0.3))
        self.assertEqual(e.exception.clave, "probabilidades_no_suman_uno")

    def test_una_probabilidad_fuera_de_cero_uno_no_es_una_probabilidad(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _prediccion(probabilities=(-0.2, 1.2))
        self.assertEqual(e.exception.clave, "fuera_de_rango")

    def test_una_puntuacion_unica_solo_vale_en_binaria(self):
        binaria = _prediccion(probabilities=None, scores=(1.7,))
        self.assertEqual(binaria.scores, (1.7,))
        with self.assertRaises(EsquemaInvalido) as e:
            _prediccion(probabilities=None, scores=(1.7,), classes=("a", "b", "c"),
                        y_true="a")
        self.assertEqual(e.exception.clave, "columnas_desalineadas")

    def test_un_y_true_que_no_es_ninguna_de_las_clases(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _prediccion(y_true="tal_vez")
        self.assertEqual(e.exception.clave, "clase_positiva_fuera_de_clases")

    def test_una_prediccion_de_datos_nuevos_no_tiene_ni_y_true_ni_rol(self):
        """Y eso es legítimo: se guarda tal cual, y quien quiera recomputar se
        encuentra el `no vale` con su motivo, no un cero."""
        nueva = PredictionRecord(row_id="nuevo-1", candidate="cand-1", label="no",
                                 probabilities=(0.9, 0.1), classes=("no", "si"))
        self.assertIsNone(nueva.y_true)
        self.assertIsNone(nueva.role)
        self.assertFalse(nueva.apto_para_recomputar(_metrica()).apto)


# ---------------------------------------------------------------------------
# MetricSpec
# ---------------------------------------------------------------------------

class UnaMetricaDeclaraComoSeOrdenaTest(unittest.TestCase):
    """105 invariante 4: las métricas de calidad tienen dirección; la pendiente de
    calibración vale 1 y desviarse en cualquier sentido es peor. Ordenar todo por
    «mayor mejor» corona al modelo descalibrado por exceso."""

    def test_una_metrica_de_calidad_declara_direccion(self):
        self.assertEqual(_metrica().direction, "lower_is_better")
        self.assertIsNone(_metrica().ideal_value)

    def test_la_calibracion_declara_valor_ideal_y_no_direccion(self):
        pendiente = MetricSpec(metric_id="calibration_slope", formula_version="1.0",
                               estimand="fixed_model_on_population",
                               requires=("y_true", "probabilities", "classes"),
                               ideal_value=1.0, ideal_range=(0.9, 1.1))
        self.assertIsNone(pendiente.direction)
        self.assertEqual(pendiente.ideal_range, (0.9, 1.1))

    def test_las_dos_cosas_a_la_vez_se_rechazan(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _metrica(ideal_value=1.0)
        self.assertEqual(e.exception.clave, "metrica_con_direccion_e_ideal")

    def test_ninguna_de_las_dos_tambien(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _metrica(direction=None)
        self.assertEqual(e.exception.clave, "metrica_sin_direccion_ni_ideal")

    def test_un_rango_ideal_del_reves(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _metrica(direction=None, ideal_range=(1.1, 0.9))
        self.assertEqual(e.exception.clave, "rango_ideal_al_reves")

    def test_una_ficha_GENERICA_declara_que_necesita_clase_positiva_sin_traerla(self):
        """La intención de esta prueba no ha cambiado; el sitio donde se hace
        cumplir, sí (revisado el 2026-09-05 al auditar el 105-C1).

        Antes se exigía que una métrica que nombra `positive_label` en
        `requires` trajera una clase concreta. Pero `requires` significa **lo
        que la muestra tiene que traer**, y la ficha genérica del catálogo
        —«qué es la sensibilidad»— no conoce ninguna clase. Cumplir aquella
        regla obligó al 105-C1 a inventarse un marcador de texto en un campo
        tipado como etiqueta: un valor fabricado.

        Que una métrica de clase positiva no se calcule sin una **sigue siendo
        cierto**, y se comprueba donde vive el dato: una `Muestra` binaria sin
        clase positiva no se construye.
        """
        spec = _metrica(metric_id="sensibilidad",
                        requires=("y_true", "labels", "positive_label"))
        self.assertIn("positive_label", spec.requires)
        self.assertIsNone(spec.positive_label)

    def test_un_requisito_inventado_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _metrica(requires=("y_true", "telepatia"))
        self.assertEqual(e.exception.clave, "no_es_del_vocabulario")

    def test_el_estimando_se_declara_y_es_del_vocabulario(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _metrica(estimand="lo_que_sea")
        self.assertEqual(e.exception.clave, "no_es_del_vocabulario")
        self.assertIn("training_variability", e.exception.es)


# ---------------------------------------------------------------------------
# FittedPipelineSpec y FitResult
# ---------------------------------------------------------------------------

class ElPipelineNoSeAjustaConElTestTest(unittest.TestCase):
    """Invariante 2 del 104 metido en el propio documento: un pipeline que
    declara haberse ajustado con el test no se puede ni escribir."""

    def test_el_pipeline_completo_vale_y_declara_con_que_se_ajusto(self):
        pipeline = _pipeline()
        self.assertEqual(pipeline.trained_on_roles, ("development", "selection"))
        self.assertFalse(pipeline.frozen)
        self.assertEqual(pipeline.decision_policy.scale, "calibrated_probability")

    def test_entrenado_con_test_se_rechaza(self):
        for rol in ("test", "external_test"):
            with self.subTest(rol=rol):
                with self.assertRaises(EsquemaInvalido) as e:
                    _pipeline(trained_on_roles=("development", rol))
                self.assertEqual(e.exception.clave, "entrenar_con_test")

    def test_un_umbral_sobre_el_score_crudo_detras_de_un_calibrador(self):
        """104-C4: el umbral se elige sobre la MISMA escala calibrada. Uno elegido
        sobre el score crudo corta en otro sitio."""
        with self.assertRaises(EsquemaInvalido) as e:
            _pipeline(decision_policy=PoliticaDeDecision(threshold=1.7, scale="raw_score",
                                                         positive_label="si"))
        self.assertEqual(e.exception.clave, "umbral_fuera_de_la_escala_calibrada")
        # Sin calibrador, el mismo umbral crudo es legítimo.
        sin_calibrador = _pipeline(calibrator=None,
                                   decision_policy=PoliticaDeDecision(
                                       threshold=1.7, scale="raw_score",
                                       positive_label="si"))
        self.assertEqual(sin_calibrador.decision_policy.threshold, 1.7)

    def test_una_probabilidad_calibrada_fuera_de_cero_uno_no_es_un_umbral(self):
        with self.assertRaises(EsquemaInvalido) as e:
            PoliticaDeDecision(threshold=1.7, scale="calibrated_probability",
                               positive_label="si")
        self.assertEqual(e.exception.clave, "fuera_de_rango")

    def test_un_pipeline_sin_predictor_no_es_un_pipeline(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _pipeline(predictor={})
        self.assertEqual(e.exception.clave, "falta_campo")

    def test_congelar_produce_un_documento_nuevo_con_otro_digest(self):
        pipeline = _pipeline()
        congelado = pipeline.congelado()
        self.assertTrue(congelado.frozen)
        self.assertFalse(pipeline.frozen)
        self.assertNotEqual(pipeline.digest(), congelado.digest())


class UnAbortoNoSeDisfrazaDeCompletadoTest(unittest.TestCase):
    """Invariante 3 del 104. Y los estados son los del contrato, compartidos con
    102-C1: si cada uno escribe su lista, acaban divergiendo."""

    def test_completado_con_su_pipeline(self):
        resultado = FitResult(candidate="cand-1", state="completed", pipeline=_pipeline(),
                              resources=Recursos(cpu_seconds=42.0, wall_seconds=51.0))
        self.assertEqual(resultado.state, "completed")
        self.assertIsNone(resultado.resources.peak_ram_mb)  # no medido, no cero

    def test_completado_sin_pipeline_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            FitResult(candidate="cand-1", state="completed")
        self.assertEqual(e.exception.clave, "completado_sin_pipeline")

    def test_limitado_por_presupuesto_conserva_candidato_Y_dice_que_limite(self):
        con_motivo = FitResult(candidate="cand-1", state="completed_budget_limited",
                               pipeline=_pipeline(),
                               reason=_motivo("se agotó el reloj del estudio a los 600 s",
                                              "the study clock ran out at 600 s"))
        self.assertEqual(con_motivo.state, "completed_budget_limited")
        with self.assertRaises(EsquemaInvalido) as e:
            FitResult(candidate="cand-1", state="completed_budget_limited",
                      pipeline=_pipeline())
        self.assertEqual(e.exception.clave, "sin_motivo_escrito")

    def test_un_fallo_no_puede_traer_pipeline_ni_callarse_el_motivo(self):
        with self.assertRaises(EsquemaInvalido) as e:
            FitResult(candidate="cand-1", state="failed", pipeline=_pipeline(),
                      reason=_motivo("el motor lanzó una excepción",
                                     "the engine raised"))
        self.assertEqual(e.exception.clave, "fallido_con_pipeline")
        with self.assertRaises(EsquemaInvalido) as e:
            FitResult(candidate="cand-1", state="failed")
        self.assertEqual(e.exception.clave, "sin_motivo_escrito")

    def test_un_motivo_a_medio_traducir_no_vale(self):
        with self.assertRaises(EsquemaInvalido) as e:
            FitResult(candidate="cand-1", state="cancelled",
                      reason={"es": "lo canceló el usuario"})
        self.assertEqual(e.exception.clave, "motivo_incompleto")

    def test_un_estado_inventado_se_rechaza_con_la_lista(self):
        with self.assertRaises(EsquemaInvalido) as e:
            FitResult(candidate="cand-1", state="completed_partially")
        self.assertEqual(e.exception.clave, "no_es_del_vocabulario")
        for estado in ("queued", "running", "completed", "completed_budget_limited",
                       "failed", "cancelled", "unsupported"):
            self.assertIn(estado, e.exception.en)


# ---------------------------------------------------------------------------
# EvaluationResult y SelectionDecision
# ---------------------------------------------------------------------------

class UnNumeroDiceDeQueArtefactoEsTest(unittest.TestCase):
    """El vínculo artefacto-evaluación del 104-C0 y del 106-C2: sin él, dos
    versiones de un modelo comparten cifras que no son suyas."""

    def test_la_evaluacion_ata_el_numero_al_pipeline_y_a_la_particion(self):
        evaluacion = _evaluacion()
        self.assertEqual(evaluacion.pipeline_digest, "b" * 64)
        self.assertEqual(evaluacion.split_plan_digest, "a" * 64)
        self.assertEqual(evaluacion.evaluated_role, "test")

    def test_una_metrica_indefinida_vale_None_y_dice_por_que(self):
        indefinida = ValorDeMetrica(metric_id="auroc", formula_version="1.0",
                                    undefined_reason=_motivo(
                                        "solo hay una clase en la partición",
                                        "the split has a single class"))
        self.assertIsNone(indefinida.value)
        self.assertNotEqual(indefinida.value, 0.0)
        self.assertIn("clase", indefinida.undefined_reason["es"])

    def test_ni_valor_ni_motivo_es_un_None_mudo(self):
        with self.assertRaises(EsquemaInvalido) as e:
            ValorDeMetrica(metric_id="auroc", formula_version="1.0")
        self.assertEqual(e.exception.clave, "metrica_sin_valor_ni_motivo")

    def test_valor_y_motivo_a_la_vez_tampoco(self):
        with self.assertRaises(EsquemaInvalido) as e:
            ValorDeMetrica(metric_id="auroc", formula_version="1.0", value=0.8,
                           undefined_reason=_motivo("indefinida", "undefined"))
        self.assertEqual(e.exception.clave, "metrica_con_valor_y_motivo")

    def test_no_se_puede_llamar_independiente_a_lo_medido_en_desarrollo(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _evaluacion(evaluated_role="development")
        self.assertEqual(e.exception.clave, "evidencia_independiente_fuera_del_test")

    def test_la_version_que_aprendio_del_test_NO_hereda_los_numeros(self):
        """104-C0 literal: se registra nueva versión y el estado
        `test_used_for_development`, y el artefacto nuevo no conserva la etiqueta
        de evaluación independiente."""
        previa = _evaluacion()
        nueva = version_tras_aprender_del_test(
            previa, "c" * 64,
            _motivo("se reentrenó incluyendo el test",
                    "retrained including the test"))
        self.assertEqual(nueva.evidence, "test_used_for_development")
        self.assertEqual(nueva.metrics, ())
        self.assertEqual(nueva.derives_from, "ev-1")
        self.assertEqual(nueva.pipeline_digest, "c" * 64)
        # Y la evaluación anterior sigue intacta con sus números.
        self.assertEqual(previa.evidence, "independent_test")
        self.assertEqual(previa.metrics[0].value, 0.81)

    def test_la_nueva_version_tiene_que_ser_OTRO_artefacto(self):
        with self.assertRaises(EsquemaInvalido) as e:
            version_tras_aprender_del_test(_evaluacion(), "b" * 64,
                                           _motivo("igual", "same"))
        self.assertEqual(e.exception.clave, "hay_duplicados")


class LaSeleccionNoSeApoyaEnElTestTest(unittest.TestCase):
    """«Ningún modelo cumple» es una respuesta y «no hay evidencia» es otra:
    confundirlas convierte una ausencia de veredicto en un veredicto."""

    def _evaluacion_de_seleccion(self) -> EvaluationResult:
        return _evaluacion(evaluation_id="ev-sel", evaluated_role="selection",
                           evidence="development_estimate")

    def test_una_decision_con_ganador_necesita_candidato_y_evidencia(self):
        decision = SelectionDecision(
            decision_id="dec-1", outcome="selected", chosen_candidate="cand-1",
            reason=_motivo("supera los mínimos y gana en utilidad",
                           "clears the minimums and wins on utility"),
            policy="utilidad_con_minimos", split_plan_digest="a" * 64,
            evidence=(self._evaluacion_de_seleccion(),),
            constraints_checked=("min_especificidad",),
            rejected=({"candidate": "cand-2",
                       "reason": _motivo("no llega a la especificidad mínima",
                                         "below the minimum specificity")},))
        self.assertEqual(decision.chosen_candidate, "cand-1")
        self.assertEqual(decision.rejected[0]["candidate"], "cand-2")

    def test_apoyarse_en_una_evaluacion_del_test_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            SelectionDecision(decision_id="dec-2", outcome="selected",
                              chosen_candidate="cand-1",
                              reason=_motivo("gana en el test", "wins on the test"),
                              policy="utilidad", split_plan_digest="a" * 64,
                              evidence=(_evaluacion(),))
        self.assertEqual(e.exception.clave, "seleccionar_con_el_test")

    def test_sin_ganador_no_se_señala_candidato(self):
        sin_ganador = SelectionDecision(
            decision_id="dec-3", outcome="no_feasible_model",
            reason=_motivo("ninguno supera la especificidad mínima",
                           "none clears the minimum specificity"),
            policy="utilidad_con_minimos", split_plan_digest="a" * 64)
        self.assertIsNone(sin_ganador.chosen_candidate)
        with self.assertRaises(EsquemaInvalido) as e:
            SelectionDecision(decision_id="dec-4", outcome="insufficient_evidence",
                              chosen_candidate="cand-1",
                              reason=_motivo("no hay datos", "no data"),
                              policy="utilidad", split_plan_digest="a" * 64)
        self.assertEqual(e.exception.clave, "sin_ganador_con_candidato")

    def test_un_ganador_sin_una_sola_evaluacion_no_se_elige(self):
        with self.assertRaises(EsquemaInvalido) as e:
            SelectionDecision(decision_id="dec-5", outcome="selected",
                              chosen_candidate="cand-1",
                              reason=_motivo("me gusta", "I like it"),
                              policy="utilidad", split_plan_digest="a" * 64)
        self.assertEqual(e.exception.clave, "ganador_sin_evidencia")

    def test_un_descarte_sin_motivo_traducido_no_vale(self):
        with self.assertRaises(EsquemaInvalido) as e:
            SelectionDecision(decision_id="dec-6", outcome="selected",
                              chosen_candidate="cand-1",
                              reason=_motivo("gana", "wins"), policy="utilidad",
                              split_plan_digest="a" * 64,
                              evidence=(self._evaluacion_de_seleccion(),),
                              rejected=({"candidate": "cand-2", "reason": "lento"},))
        self.assertEqual(e.exception.clave, "no_es_mapa")


# ---------------------------------------------------------------------------
# El registro de accesos: la pieza central del corte
# ---------------------------------------------------------------------------

def _pipeline_final(plan: SplitPlan) -> FittedPipelineSpec:
    return _pipeline(trained_on_roles=("development", "selection", "calibration"),
                     split_plan_digest=plan.digest()).congelado()


def _estudio_honrado(plan: SplitPlan, registro: RegistroDeAccesos,
                     lector: LectorDeParticiones, *,
                     candidato: str = "cand-1") -> FittedPipelineSpec:
    """El recorrido del protocolo mínimo, entero y por el orden que manda.

    Es una simulación DELIBERADAMENTE completa: si solo llegara hasta la
    selección, la prueba diría poco — el test nunca se leería y todo saldría
    verde aunque el guardia no existiera.
    """
    with registro.fase("development"):
        lector.filas("development", proposito="fit", candidato=candidato)
        lector.filas("development", proposito="tune", candidato=candidato)
    with registro.fase("selection"):
        lector.filas("selection", proposito="select", candidato=candidato)
    with registro.fase("calibration"):
        lector.filas("calibration", proposito="calibrate", candidato=candidato)
        lector.filas("calibration", proposito="tune_threshold", candidato=candidato)
    with registro.fase("final_fit"):
        for rol in ("development", "selection", "calibration"):
            lector.filas(rol, proposito="fit", candidato=candidato)
    pipeline = _pipeline_final(plan)
    registro.congelar(pipeline.digest())
    with registro.fase("final_evaluation"):
        lector.filas("test", proposito="evaluate", candidato=candidato,
                     artefacto=pipeline.digest())
    return pipeline


class ElRegistroDeAccesosImpideLaFugaTest(unittest.TestCase):
    """Criterio 3 del corte, y sin esto el resto es papel: una simulación de un
    pipeline completo que FALLA si alguien lee el rol `test` durante desarrollo,
    selección o calibración."""

    def setUp(self):
        self.plan = _plan()
        self.registro = RegistroDeAccesos(self.plan, estudio="uci-2026")
        self.lector = LectorDeParticiones(self.plan, self.registro)

    def test_el_estudio_honrado_llega_hasta_el_final_sin_un_solo_denegado(self):
        pipeline = _estudio_honrado(self.plan, self.registro, self.lector)
        self.assertEqual(self.registro.denegados(), ())
        # Y lo positivo, que un aserto negativo lo pasaría un estudio en blanco:
        lecturas = [a for a in self.registro.accesos if a.tipo == "read"]
        self.assertEqual(len(lecturas), 9)
        self.assertEqual({a.fase for a in lecturas},
                         {"development", "selection", "calibration", "final_fit",
                          "final_evaluation"})
        del_test = self.registro.lecturas_del_test()
        self.assertEqual(len(del_test), 1)
        self.assertEqual(del_test[0].fase, "final_evaluation")
        self.assertEqual(del_test[0].proposito, "evaluate")
        self.assertEqual(del_test[0].observaciones, 2)
        self.assertEqual(del_test[0].artefacto, pipeline.digest())
        self.assertEqual(self.registro.etiqueta_de_evidencia(pipeline.digest()),
                         "independent_test")

    def test_mirar_el_test_DURANTE_LA_SELECCION_se_deniega_y_queda_anotado(self):
        with self.registro.fase("development"):
            self.lector.filas("development", proposito="fit")
        with self.registro.fase("selection"):
            self.lector.filas("selection", proposito="select")
            with self.assertRaises(FugaDeTest) as e:
                self.lector.filas("test", proposito="select")
        self.assertEqual(e.exception.clave, "fuga_de_test")
        denegados = self.registro.denegados()
        self.assertEqual(len(denegados), 1)
        self.assertEqual(denegados[0].rol, "test")
        self.assertEqual(denegados[0].fase, "selection")
        self.assertFalse(denegados[0].concedido)
        for idioma in IDIOMAS:
            self.assertIn("test", denegados[0].motivo[idioma])
        # Ninguna fila del test se ha entregado.
        concedidos_del_test = [a for a in self.registro.lecturas_del_test() if a.concedido]
        self.assertEqual(concedidos_del_test, [])

    def test_lo_mismo_en_desarrollo_en_calibracion_y_en_el_ajuste_final(self):
        for fase in ("development", "calibration", "final_fit"):
            with self.subTest(fase=fase):
                registro = RegistroDeAccesos(self.plan)
                lector = LectorDeParticiones(self.plan, registro)
                registro.congelar("da igual: la fase ya lo impide")
                with registro.fase(fase):
                    with self.assertRaises(FugaDeTest) as e:
                        lector.filas("test", proposito="fit", artefacto="x")
                self.assertEqual(e.exception.clave, "fuga_de_test")
                self.assertEqual(len(registro.denegados()), 1)

    def test_en_su_fase_pero_para_APRENDER_tambien_se_deniega(self):
        """Mirar el test para elegir el umbral es usarlo para decidir aunque se
        haga al final."""
        pipeline = _pipeline_final(self.plan)
        self.registro.congelar(pipeline.digest())
        with self.registro.fase("final_evaluation"):
            for proposito in ("fit", "tune", "select", "calibrate", "tune_threshold"):
                with self.subTest(proposito=proposito):
                    with self.assertRaises(FugaDeTest) as e:
                        self.lector.filas("test", proposito=proposito,
                                          artefacto=pipeline.digest())
                    self.assertEqual(e.exception.clave, "proposito_indebido_en_el_test")

    def test_evaluar_sin_congelar_no_se_permite(self):
        with self.registro.fase("final_evaluation"):
            with self.assertRaises(ProtocoloRoto) as e:
                self.lector.filas("test", proposito="evaluate", artefacto="pl-1")
        self.assertEqual(e.exception.clave, "test_sin_congelar")

    def test_evaluar_OTRO_artefacto_distinto_del_congelado_no_se_permite(self):
        pipeline = _pipeline_final(self.plan)
        self.registro.congelar(pipeline.digest())
        with self.registro.fase("final_evaluation"):
            with self.assertRaises(ProtocoloRoto) as e:
                self.lector.filas("test", proposito="evaluate", artefacto="otro")
        self.assertEqual(e.exception.clave, "artefacto_distinto_del_congelado")

    def test_un_acceso_sin_fase_abierta_no_se_puede_juzgar(self):
        with self.assertRaises(ProtocoloRoto) as e:
            self.lector.filas("development", proposito="fit")
        self.assertEqual(e.exception.clave, "fase_desconocida")

    def test_la_cohorte_externa_tiene_su_propia_fase(self):
        plan = _plan(assignments={"o1": "development", "o2": "development",
                                  "o3": "selection", "o4": "calibration",
                                  "o5": "test", "o6": "external_test"},
                     units={"o1": "p1", "o2": "p1", "o3": "p2", "o4": "p3",
                            "o5": "p4", "o6": "p5"})
        registro = RegistroDeAccesos(plan)
        lector = LectorDeParticiones(plan, registro)
        registro.congelar("artefacto-1")
        with registro.fase("final_evaluation"):
            with self.assertRaises(FugaDeTest):
                lector.filas("external_test", proposito="evaluate",
                             artefacto="artefacto-1")
        with registro.fase("external_validation"):
            self.assertEqual(lector.filas("external_test", proposito="evaluate",
                                          artefacto="artefacto-1"), ("o6",))
        self.assertEqual(registro.etiqueta_de_evidencia("artefacto-1"),
                         "external_validation")

    def test_un_rol_que_el_plan_no_reparte_no_se_puede_leer(self):
        plan = _plan_simple(assignments={"o1": "development", "o2": "test"})
        registro = RegistroDeAccesos(plan)
        lector = LectorDeParticiones(plan, registro)
        with registro.fase("selection"):
            with self.assertRaises(EsquemaInvalido) as e:
                lector.filas("selection", proposito="select")
        self.assertEqual(e.exception.clave, "rol_no_esta_en_el_plan")

    def test_la_evaluacion_anidada_se_declara_FUERA_en_vez_de_correrse_a_medias(self):
        anidado = _plan_simple(assignments={"o1": "development", "o2": "selection"},
                               nested_evaluation=True)
        with self.assertRaises(ProtocoloRoto) as e:
            RegistroDeAccesos(anidado)
        self.assertEqual(e.exception.clave, "anidada_no_implementada")

    def test_el_registro_se_publica_entero_para_el_expediente(self):
        pipeline = _estudio_honrado(self.plan, self.registro, self.lector)
        publicado = self.registro.a_json()
        self.assertEqual(publicado["schema_version"], ESTUDIO_SCHEMA_VERSION)
        self.assertEqual(publicado["split_plan_digest"], self.plan.digest())
        self.assertEqual(publicado["frozen_artifact"], pipeline.digest())
        self.assertFalse(publicado["test_used_for_development"])
        self.assertEqual(len(publicado["accesses"]), len(self.registro.accesos))
        self.assertEqual(len(self.registro.digest()), 64)

    def test_el_registro_dice_QUE_conto_y_no_solo_cuantos(self):
        """Seis filas de tres pacientes son `6 observations` o `3 units`: un
        registro que no lo distinguiera enseñaría un 3 donde hubo 6."""
        pipeline = _pipeline_final(self.plan)
        self.registro.congelar(pipeline.digest())
        with self.registro.fase("final_evaluation"):
            filas = self.lector.filas("test", proposito="evaluate",
                                      artefacto=pipeline.digest())
            unidades = self.lector.unidades("test", proposito="evaluate",
                                            artefacto=pipeline.digest())
        self.assertEqual(filas, ("o5", "o6"))
        self.assertEqual(unidades, ("p4",))
        self.assertEqual([(a.observaciones, a.granularidad)
                          for a in self.registro.lecturas_del_test()],
                         [(2, "observations"), (1, "units")])

    def test_pedir_las_UNIDADES_del_test_fuera_de_su_fase_tampoco_cuela(self):
        """El remuestreo por unidad del 105-C2 pasa por el mismo guardia: si no,
        habría una puerta de atrás para contar los positivos del test."""
        self.registro.congelar("artefacto-1")
        with self.registro.fase("selection"):
            with self.assertRaises(FugaDeTest) as e:
                self.lector.unidades("test", proposito="select", artefacto="artefacto-1")
        self.assertEqual(e.exception.clave, "fuga_de_test")
        self.assertEqual(self.registro.denegados()[0].granularidad, "units")


class LaEvidenciaLaDecideElRegistroTest(unittest.TestCase):
    """Quién puede decir `independent_test` no lo decide quien redacta el
    informe: lo decide el único que sabe si alguien había mirado antes."""

    def setUp(self):
        self.plan = _plan()
        self.registro = RegistroDeAccesos(self.plan)
        self.lector = LectorDeParticiones(self.plan, self.registro)

    def test_el_segundo_candidato_evaluado_en_el_MISMO_test_ya_no_es_independiente(self):
        """104-C5: no se prueban candidatos hasta encontrar uno que gane sobre el
        mismo test sin reconocer ese nuevo uso."""
        primero = _estudio_honrado(self.plan, self.registro, self.lector,
                                   candidato="cand-1")
        segundo = _pipeline(pipeline_id="pl-2", candidate="cand-2",
                            trained_on_roles=("development",),
                            split_plan_digest=self.plan.digest()).congelado()
        self.registro.congelar(segundo.digest())
        with self.registro.fase("final_evaluation"):
            self.lector.filas("test", proposito="evaluate", candidato="cand-2",
                              artefacto=segundo.digest())
        self.assertEqual(self.registro.etiqueta_de_evidencia(primero.digest()),
                         "independent_test")
        self.assertEqual(self.registro.etiqueta_de_evidencia(segundo.digest()),
                         "repeated_test_use")

    def test_tras_aprender_del_test_no_se_vuelve_a_conceder_independencia(self):
        primero = _estudio_honrado(self.plan, self.registro, self.lector)
        self.registro.aprender_del_test(_motivo(
            "se reentrena incluyendo el test para exprimir los datos",
            "retraining on the test to squeeze the data"))
        self.assertTrue(self.registro.test_usado_para_desarrollar)
        self.assertEqual(self.registro.etiqueta_de_evidencia("artefacto-nuevo"),
                         "test_used_for_development")
        # Lo medido ANTES sigue siendo lo que era: no se reescribe la historia.
        self.assertEqual(self.registro.etiqueta_de_evidencia(primero.digest()),
                         "independent_test")
        self.assertTrue(self.registro.a_json()["test_used_for_development"])

    def test_un_artefacto_que_nadie_ha_medido_no_esta_aprobado(self):
        self.assertEqual(self.registro.etiqueta_de_evidencia("nunca-medido"),
                         "not_evaluated")

    def test_la_etiqueta_del_registro_encaja_con_el_esquema_de_evaluacion(self):
        pipeline = _estudio_honrado(self.plan, self.registro, self.lector)
        evaluacion = EvaluationResult(
            evaluation_id="ev-final", pipeline_digest=pipeline.digest(),
            split_plan_digest=self.plan.digest(), evaluated_role="test",
            evidence=self.registro.etiqueta_de_evidencia(pipeline.digest()),
            metrics=(ValorDeMetrica(metric_id="auroc", formula_version="1.0",
                                    value=0.77, n_observations=2, n_units=1),))
        self.assertEqual(evaluacion.evidence, "independent_test")
        self.assertEqual(evaluacion.pipeline_digest, pipeline.digest())


# ---------------------------------------------------------------------------
# Versiones y migración
# ---------------------------------------------------------------------------

class UnProyectoAntiguoSeAbreYSeMarcaTest(unittest.TestCase):
    """Criterio 4 del corte: esquemas versionados con política de migración, sin
    romper proyectos antiguos. Un documento de una versión anterior **se lee y se
    marca**; uno de una versión desconocida se rechaza — que no es lo mismo que
    borrarlo."""

    def _registro_con_una_migracion(self) -> RegistroDeMigraciones:
        migraciones = RegistroDeMigraciones()

        def de_09_a_10(documento):
            """En `0.9` el campo se llamaba `id_field`."""
            documento["observation_id_field"] = documento.pop("id_field")
            return documento

        migraciones.registrar(SplitPlan.ESQUEMA, "0.9", "1.0", de_09_a_10)
        return migraciones

    def _plan_antiguo(self) -> dict:
        return {"schema": SplitPlan.ESQUEMA, "schema_version": "0.9",
                "plan_id": "proyecto-de-marzo", "split_type": "iid",
                "id_field": "id",
                "assignments": {"o1": "development", "o2": "test"}}

    def test_un_plan_de_la_version_anterior_se_ABRE(self):
        plan = SplitPlan.desde_json(self._plan_antiguo(),
                                    migraciones=self._registro_con_una_migracion())
        self.assertEqual(plan.plan_id, "proyecto-de-marzo")
        self.assertEqual(plan.observation_id_field, "id")
        self.assertEqual(plan.observaciones_del_rol("test"), ("o2",))

    def test_y_SE_MARCA_de_donde_venia(self):
        plan = SplitPlan.desde_json(self._plan_antiguo(),
                                    migraciones=self._registro_con_una_migracion())
        self.assertEqual(plan.migracion,
                         {"desde": "0.9", "hasta": "1.0", "pasos": ["0.9->1.0"]})
        self.assertEqual(plan.a_json()["schema_migration"]["desde"], "0.9")
        self.assertEqual(plan.a_json()["schema_version"], ESTUDIO_SCHEMA_VERSION)

    def test_la_marca_NO_cambia_el_digest_del_contenido(self):
        """Si entrara en el digest, el mismo plan tendría dos huellas según se
        hubiera cargado de un fichero antiguo o de uno nuevo, y los vínculos
        artefacto-evaluación dejarían de casar por un motivo que no tiene nada
        que ver con lo que el plan dice."""
        migrado = SplitPlan.desde_json(self._plan_antiguo(),
                                       migraciones=self._registro_con_una_migracion())
        moderno = SplitPlan(plan_id="proyecto-de-marzo", split_type="iid",
                            observation_id_field="id",
                            assignments={"o1": "development", "o2": "test"})
        self.assertIsNone(moderno.migracion)
        self.assertEqual(migrado.digest(), moderno.digest())

    def test_una_version_desconocida_se_rechaza_diciendo_cuales_se_leen(self):
        futuro = dict(self._plan_antiguo(), schema_version="9.9")
        with self.assertRaises(MigracionImposible) as e:
            SplitPlan.desde_json(futuro, migraciones=self._registro_con_una_migracion())
        self.assertEqual(e.exception.clave, "version_no_legible")
        for idioma in IDIOMAS:
            self.assertIn("9.9", e.exception.bilingue[idioma])
            self.assertIn("0.9", e.exception.bilingue[idioma])
            self.assertIn("1.0", e.exception.bilingue[idioma])

    def test_un_documento_sin_version_no_se_interpreta(self):
        sin_version = {k: v for k, v in self._plan_antiguo().items()
                       if k != "schema_version"}
        with self.assertRaises(EsquemaInvalido) as e:
            SplitPlan.desde_json(sin_version)
        self.assertEqual(e.exception.clave, "falta_campo")

    def test_las_versiones_legibles_se_pueden_consultar(self):
        migraciones = self._registro_con_una_migracion()
        self.assertEqual(migraciones.versiones_legibles(SplitPlan.ESQUEMA),
                         ("0.9", "1.0"))
        self.assertEqual(migraciones.versiones_legibles(ProblemSpec.ESQUEMA), ("1.0",))

    def test_hoy_el_producto_no_tiene_ninguna_migracion_registrada_y_lo_dice(self):
        """`1.0` es la primera versión. Es un hecho, no un olvido: el mecanismo
        existe desde el primer día porque añadirlo después obligaría a decidir
        qué hacer con los documentos ya escritos."""
        for esquema in (ProblemSpec, SplitPlan, PredictionRecord, MetricSpec,
                        FittedPipelineSpec, FitResult, EvaluationResult,
                        SelectionDecision):
            with self.subTest(esquema=esquema.ESQUEMA):
                self.assertEqual(MIGRACIONES.versiones_legibles(esquema.ESQUEMA),
                                 (ESTUDIO_SCHEMA_VERSION,))

    def test_dos_caminos_desde_la_misma_version_es_un_fallo_de_cableado(self):
        migraciones = self._registro_con_una_migracion()
        with self.assertRaises(EsquemaInvalido) as e:
            migraciones.registrar(SplitPlan.ESQUEMA, "0.9", "1.0", lambda d: d)
        self.assertEqual(e.exception.clave, "migracion_ya_registrada")


# ---------------------------------------------------------------------------
# Ida y vuelta a JSON
# ---------------------------------------------------------------------------

class LoQueSeEscribeSeVuelveALeerTest(unittest.TestCase):
    """Un esquema que no se puede releer no sirve para guardar un proyecto. Y una
    clave que este core no conoce se RECHAZA: interpretarla a medias es justo lo
    que la versión de esquema existe para evitar."""

    def _documentos(self):
        plan = _plan()
        return [
            _problema(),
            plan,
            _prediccion(),
            _metrica(),
            _pipeline(),
            FitResult(candidate="cand-1", state="completed", pipeline=_pipeline(),
                      resources=Recursos(cpu_seconds=1.5)),
            _evaluacion(),
            SelectionDecision(
                decision_id="dec-1", outcome="selected", chosen_candidate="cand-1",
                reason=_motivo("gana en utilidad", "wins on utility"),
                policy="utilidad_con_minimos", split_plan_digest=plan.digest(),
                evidence=(_evaluacion(evaluation_id="ev-sel",
                                      evaluated_role="selection",
                                      evidence="development_estimate"),),
                rejected=({"candidate": "cand-2",
                           "reason": _motivo("demasiado lento", "too slow")},),
                constraints_checked=("max_latencia_ms",)),
        ]

    def test_ida_y_vuelta_conserva_el_documento_y_el_digest(self):
        for documento in self._documentos():
            with self.subTest(esquema=type(documento).ESQUEMA):
                vuelta = type(documento).desde_json(documento.a_json())
                self.assertEqual(vuelta.a_json(), documento.a_json())
                self.assertEqual(vuelta.digest(), documento.digest())
                self.assertIsNone(vuelta.migracion)

    def test_una_clave_desconocida_se_rechaza_en_todos(self):
        for documento in self._documentos():
            with self.subTest(esquema=type(documento).ESQUEMA):
                sucio = dict(documento.a_json())
                sucio["campo_del_futuro"] = 1
                with self.assertRaises(EsquemaInvalido) as e:
                    type(documento).desde_json(sucio)
                self.assertEqual(e.exception.clave, "clave_desconocida")

    def test_un_documento_de_otro_esquema_no_se_lee_como_este(self):
        payload = dict(_plan().a_json(), schema=ProblemSpec.ESQUEMA)
        with self.assertRaises(EsquemaInvalido) as e:
            SplitPlan.desde_json(payload)
        self.assertEqual(e.exception.clave, "esquema_equivocado")

    def test_un_digest_de_asignaciones_que_no_cuadra_se_rechaza(self):
        payload = _plan().a_json()
        payload["digests"] = {"assignments": "0" * 64}
        with self.assertRaises(EsquemaInvalido) as e:
            SplitPlan.desde_json(payload)
        self.assertEqual(e.exception.clave, "hay_duplicados")

    def test_un_texto_donde_iba_una_lista_no_se_lee_como_sus_letras(self):
        """`tuple("edad")` son cuatro cadenas no vacías, y una validación por
        elemento las daría por buenas: cuatro predictores donde había uno."""
        payload = _problema().a_json()
        payload["predictors"] = "edad"
        with self.assertRaises(EsquemaInvalido) as e:
            ProblemSpec.desde_json(payload)
        self.assertEqual(e.exception.clave, "no_es_lista_de_textos")
        decision = SelectionDecision(
            decision_id="dec-1", outcome="no_feasible_model",
            reason=_motivo("ninguno cumple", "none qualifies"), policy="minimos",
            split_plan_digest="a" * 64).a_json()
        decision["constraints_checked"] = "min_especificidad"
        with self.assertRaises(EsquemaInvalido) as e:
            SelectionDecision.desde_json(decision)
        self.assertEqual(e.exception.clave, "no_es_lista_de_textos")

    def test_un_campo_obligatorio_que_falta_se_dice_que_FALTA(self):
        payload = _problema().a_json()
        del payload["target"]
        with self.assertRaises(EsquemaInvalido) as e:
            ProblemSpec.desde_json(payload)
        self.assertEqual(e.exception.clave, "falta_campo")
        self.assertIn("target", e.exception.es)


# ---------------------------------------------------------------------------
# Lo que lee una persona
# ---------------------------------------------------------------------------

class LoQueRedactaElCoreSeTraduceEnElCoreTest(unittest.TestCase):
    """Media aplicación traducida se ve así: un rechazo con la frase en inglés
    dentro de una pantalla en castellano. Y guardar la cadena ya compuesta hace
    que al cambiar de idioma no cambie."""

    #: Palabras FUNCIONALES: no se pueden evitar escribiendo en castellano, y por
    #: eso sirven para detectarlo. Una lista de palabras de contenido escogida a
    #: mano no barre.
    FUNCIONALES = ("el", "la", "los", "las", "del", "con", "que", "desde", "para",
                   "por", "una", "uno", "sin", "pero", "como", "cuando", "más",
                   "porque", "aunque", "sobre", "entre", "hay", "tiene", "puede")

    def test_todas_las_claves_tienen_los_dos_idiomas(self):
        self.assertTrue(MOTIVOS)
        for clave, redacciones in MOTIVOS.items():
            with self.subTest(clave=clave):
                self.assertEqual(set(redacciones), set(IDIOMAS))
                for idioma in IDIOMAS:
                    self.assertTrue(redacciones[idioma].strip())

    def test_los_huecos_son_LOS_MISMOS_en_los_dos_idiomas(self):
        """Un `{opciones}` que existe en castellano y no en inglés solo revienta
        cuando alguien pide inglés."""
        for clave, redacciones in MOTIVOS.items():
            with self.subTest(clave=clave):
                self.assertEqual(huecos_de(redacciones["es"]),
                                 huecos_de(redacciones["en"]))

    def test_el_ingles_no_lleva_castellano_dentro(self):
        for clave, redacciones in MOTIVOS.items():
            palabras = set(re.findall(r"[a-záéíóúñ]+", redacciones["en"].lower()))
            colados = palabras.intersection(self.FUNCIONALES)
            with self.subTest(clave=clave):
                self.assertEqual(colados, set(), f"{clave}: {sorted(colados)}")

    def test_un_motivo_que_no_esta_en_el_catalogo_no_se_improvisa(self):
        with self.assertRaises(KeyError):
            motivo("una_clave_que_no_existe")

    def test_el_rechazo_llega_en_los_dos_idiomas_y_se_puede_elegir(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _problema(positive_label="quizas")
        error = e.exception
        self.assertNotEqual(error.es, error.en)
        self.assertEqual(error.motivo("es"), error.es)
        self.assertEqual(error.motivo("en"), error.en)
        self.assertEqual(error.motivo("fr"), error.en)  # idioma que no se sabe
        self.assertEqual(str(error), error.en)
        self.assertEqual(set(error.bilingue), set(IDIOMAS))


class ElCoreSigueSiendoStdlibPuroTest(unittest.TestCase):
    """`matrixai-core` declara `dependencies = []`. Un `import numpy` aquí
    rompería la instalación de quien no lo tenga, y no lo diría hasta ejecutarse."""

    PROHIBIDOS = ("numpy", "pandas", "pydantic", "scipy", "sklearn", "torch",
                  "onnx", "yaml", "requests")

    def test_ningun_modulo_del_paquete_importa_nada_de_fuera(self):
        ficheros = sorted(PAQUETE.glob("*.py"))
        self.assertGreaterEqual(len(ficheros), 7)
        patron = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(self.PROHIBIDOS) + r")\b",
                            re.MULTILINE)
        for fichero in ficheros:
            with self.subTest(fichero=fichero.name):
                self.assertIsNone(patron.search(fichero.read_text(encoding="utf-8")))

    def test_los_digests_salen_del_canonicalizador_del_81_y_no_de_otro(self):
        """Dos canonicalizadores darían dos huellas del mismo contenido."""
        from matrixai.pipelines.canonical import jcs_bytes  # noqa: PLC0415
        import hashlib  # noqa: PLC0415

        plan = _plan()
        esperado = hashlib.sha256(jcs_bytes(
            {k: v for k, v in plan.a_json().items() if k != "schema_migration"}
        )).hexdigest()
        self.assertEqual(plan.digest(), esperado)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
