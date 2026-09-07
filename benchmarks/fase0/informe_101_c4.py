#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C4 — informe y decisión de cartera: «rendimiento de motores» (ya
producido por 101-C3) y «rendimiento del selector COMPLETO... dentro de
desarrollo», más «una repetición independiente comprueba un caso y sus
tolerancias» (texto literal del criterio de terminado).

UN CASO, NO LOS DOCE. El criterio pide un caso, no repetir la batería
entera — "sick" (categóricas de alta cardinalidad, faltantes reales,
desbalanceo 6,1%) es el más completo de los doce de 101-C3, y el que
más EXIGE al pipeline entero (búsqueda → calibración → umbral →
selección → evaluación final), no el más fácil de aprobar.

CUATRO PARTICIONES DE DESARROLLO, NO DOS. El fold de "sick" de 101-C3
da train/validation/test — para ejercer 105-C3 (recalibración) y 104-C4
(umbral) sin que uno contamine al otro («ajustar sobre datos NUEVOS,
nunca los que ajustaron la recalibración», texto literal de
`aplicar_recalibracion`), el train original (2414 filas) se reparte en
`fit_train`/`fit_val` (para `Motor.fit`) y la validation original (604
filas) en `calibracion`/`seleccion` (para `ajustar_recalibracion_
logistica` y `elegir_umbral`/`comparar_candidatos`/`seleccionar`,
respectivamente). El test original (754 filas) quedó SIN TOCAR por
esas cuatro particiones — se lee una sola vez, al final, vía
`evaluar_en_test`.

SIN AISLAMIENTO POR SUBPROCESO AQUÍ, A PROPÓSITO. `ejecutar_intento_
aislado` (104-C2 ext.) existe para una BATERÍA de cientos de intentos
donde un motor colgado no puede tirar la pasada entera — un solo caso,
en el mismo proceso, no necesita esa protección; medirlo así habría
sido ceremonia sin beneficio real.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
_RAIZ_DE_ENGINES = _RAIZ_DEL_CORE.parent / "matrixai-engines" / "src"
for ruta in (_RAIZ_DEL_CORE, _RAIZ_DE_ENGINES):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from benchmarks.fase0.pasada_exploratoria_101_c3 import (  # noqa: E402
    particiones_base, preparar_para_motor)

from matrixai.estudio import ProblemSpec  # noqa: E402
from matrixai.estudio.accesos import RegistroDeAccesos  # noqa: E402
from matrixai.estudio.calibracion import ajustar_recalibracion_logistica, aplicar_recalibracion  # noqa: E402
from matrixai.estudio.comparaciones import comparar_candidatos  # noqa: E402
from matrixai.estudio.esquemas import EvaluationResult, Restriccion  # noqa: E402
from matrixai.estudio.evaluacion_final import evaluar_en_test  # noqa: E402
from matrixai.estudio.informe_final import InformeDeEvaluacion  # noqa: E402
from matrixai.estudio.metricas import Muestra, evaluar  # noqa: E402
from matrixai.estudio.seleccion import seleccionar  # noqa: E402
from matrixai.estudio.umbral import elegir_umbral  # noqa: E402
from matrixai.estudio.validacion import digest_canonico  # noqa: E402

from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.motores.baseline import MotorBaseline  # noqa: E402
from matrixai_engines.motores.lineal import MotorLineal  # noqa: E402
from matrixai_engines.motores.arbol_lightgbm import MotorArbolLightGBM  # noqa: E402
from matrixai_engines.motores.densa import MotorDensaPropia  # noqa: E402

DATASET = (38, "sick", "mediano", "sick", "negative")
COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 4.0  # perder un caso "sick" real cuesta más que una falsa alarma


def _particion_de(filas: list[dict], objetivo: str) -> Particion:
    return Particion.desde_filas(filas, row_id_field="row_id", target_field=objetivo)


def _muestra_desde(motor, fitted, filas: list[dict], objetivo: str, spec) -> Muestra:
    datos = _particion_de(filas, objetivo)
    predicciones = motor.predict(fitted, datos)
    campos = dict(task=spec.task, y_true=tuple(datos.target), predictions=tuple(predicciones),
                 classes=fitted.classes, positive_label=fitted.positive_label)
    if fitted.capacidades.admite_probabilidades:
        campos["probabilities"] = tuple(motor.predict_proba(fitted, datos))
    return Muestra(**campos)


def _particiones_de_desarrollo(por_id: dict, pliegue, seed: int):
    """`fit_train`/`fit_val` del train original; `calibracion`/`seleccion`
    de la validation original -- cuatro slices que no se pisan entre sí."""
    rng_ids_train = list(pliegue.entrena)
    corte_fit = int(len(rng_ids_train) * 0.85)
    fit_train_ids = rng_ids_train[:corte_fit]
    fit_val_ids = rng_ids_train[corte_fit:]

    val_ids = list(pliegue.valida)
    mitad = len(val_ids) // 2
    calibracion_ids = val_ids[:mitad]
    seleccion_ids = val_ids[mitad:]

    return fit_train_ids, fit_val_ids, calibracion_ids, seleccion_ids


def main() -> None:
    data_id, nombre_ds, cubo, positiva, negativa = DATASET
    por_id, propuesta, spec, objetivo, predictores = particiones_base(
        data_id, nombre_ds, cubo, positiva, negativa)
    pliegue = propuesta.pliegues.pliegue_de(repeticion=0, pliegue=0)
    test_ids = propuesta.plan.observaciones_del_rol("test")
    fit_train_ids, fit_val_ids, calibracion_ids, seleccion_ids = _particiones_de_desarrollo(
        por_id, pliegue, seed=0)

    print(f"sick: fit_train={len(fit_train_ids)} fit_val={len(fit_val_ids)} "
         f"calibracion={len(calibracion_ids)} seleccion={len(seleccion_ids)} "
         f"test={len(test_ids)}", flush=True)

    registro = RegistroDeAccesos(propuesta.plan, estudio="101-c4-caso-sick")
    registro.abrir_fase("development")

    motores = [MotorBaseline(), MotorLineal(), MotorArbolLightGBM(), MotorDensaPropia()]
    presupuesto = Presupuesto(wall_seconds=120.0, hilos=4, seed=0)

    evaluaciones_seleccion: dict[str, EvaluationResult] = {}
    muestras_seleccion: dict[str, Muestra] = {}
    recalibraciones: dict[str, object] = {}
    umbrales: dict[str, object] = {}
    pipeline_digest_de: dict[str, str] = {}
    necesita_red: dict[str, bool] = {}

    for motor in motores:
        candidato = f"{motor.nombre}-default"
        todas = ([por_id[i] for i in fit_train_ids] + [por_id[i] for i in fit_val_ids]
                + [por_id[i] for i in calibracion_ids] + [por_id[i] for i in seleccion_ids])
        transformadas = preparar_para_motor([por_id[i] for i in fit_train_ids], todas,
                                            objetivo, predictores, motor)
        n1, n2, n3 = len(fit_train_ids), len(fit_val_ids), len(calibracion_ids)
        filas_fit_train = transformadas[:n1]
        filas_fit_val = transformadas[n1:n1 + n2]
        filas_calibracion = transformadas[n1 + n2:n1 + n2 + n3]
        filas_seleccion = transformadas[n1 + n2 + n3:]

        registro.anotar("development", proposito="fit", candidato=candidato,
                        observaciones=len(filas_fit_train) + len(filas_fit_val))
        fit_result, fitted = motor.fit(_particion_de(filas_fit_train, objetivo),
                                       _particion_de(filas_fit_val, objetivo), spec, presupuesto,
                                       candidate=candidato, split_plan_digest=propuesta.plan.digest())
        if fit_result.state != "completed" or fitted is None:
            print(f"  {motor.nombre}: fit no completado ({fit_result.state}), fuera de la comparación")
            continue
        pipeline_digest_de[candidato] = fitted.digest()
        necesita_red[candidato] = motor.capabilities().necesita_red

        registro.anotar("development", proposito="calibrate", candidato=candidato,
                        observaciones=len(filas_calibracion))
        muestra_calibracion = _muestra_desde(motor, fitted, filas_calibracion, objetivo, spec)
        recalibracion = None
        if muestra_calibracion.probabilities is not None:
            recalibracion = ajustar_recalibracion_logistica(muestra_calibracion)
        recalibraciones[candidato] = recalibracion

        registro.anotar("development", proposito="tune_threshold", candidato=candidato,
                        observaciones=len(filas_seleccion))
        muestra_seleccion = _muestra_desde(motor, fitted, filas_seleccion, objetivo, spec)
        if recalibracion is not None and recalibracion.a is not None:
            p_positiva = tuple(p[muestra_seleccion.classes.index(spec.positive_label)]
                              for p in muestra_seleccion.probabilities)
            p_recalibrada = aplicar_recalibracion(recalibracion, p_positiva)
            muestra_seleccion = Muestra(
                task=muestra_seleccion.task, y_true=muestra_seleccion.y_true,
                classes=muestra_seleccion.classes, positive_label=muestra_seleccion.positive_label,
                scores=p_recalibrada, score_rule="105-c3.recalibrada", predictions=muestra_seleccion.predictions)
        umbral = elegir_umbral(muestra_seleccion, cost_false_positive=COST_FALSE_POSITIVE,
                               cost_false_negative=COST_FALSE_NEGATIVE)
        umbrales[candidato] = umbral

        registro.anotar("development", proposito="select", candidato=candidato,
                        observaciones=len(filas_seleccion))
        informe = evaluar(muestra_seleccion, ["auroc", "accuracy", "specificity"], umbral=umbral.threshold)
        evaluaciones_seleccion[candidato] = EvaluationResult(
            evaluation_id=f"sel-{candidato}", pipeline_digest=fitted.digest(),
            split_plan_digest=propuesta.plan.digest(), evaluated_role="selection",
            evidence="development_estimate", metrics=informe.metrics)
        muestras_seleccion[candidato] = muestra_seleccion
        auroc = next((m.value for m in informe.metrics if m.metric_id == "auroc"), None)
        print(f"  {motor.nombre}: fit ok, auroc(seleccion)={auroc}, "
             f"umbral={umbral.threshold}")

    # comparación emparejada: cada candidato contra el baseline
    baseline_cand = "baseline-default"
    ganador_actual = max(evaluaciones_seleccion, key=lambda c: next(
        (m.value for m in evaluaciones_seleccion[c].metrics if m.metric_id == "auroc"), -1.0))
    comparacion_lider = None
    if baseline_cand in muestras_seleccion and ganador_actual != baseline_cand:
        comparacion_lider = comparar_candidatos(
            "auroc", muestras_seleccion[ganador_actual], muestras_seleccion[baseline_cand],
            diseno="iid", estimando="fixed_model_on_population", semilla=0)
        print(f"  comparación {ganador_actual} vs {baseline_cand}: {comparacion_lider.veredicto}")

    restricciones = (Restriccion(clave="specificity", operador="min", valor=0.3),)
    decision = seleccionar(evaluaciones_seleccion, restricciones=restricciones,
                           metric_id_calidad="auroc", decision_id="101-c4-sick",
                           split_plan_digest=propuesta.plan.digest(),
                           necesita_red=necesita_red, comparacion_lider=comparacion_lider)
    print(f"\nDecisión de selección: {decision.outcome}, ganador={decision.chosen_candidate}")

    if decision.outcome != "selected" or decision.chosen_candidate is None:
        print("Sin candidato viable -- no hay evaluación final que hacer.")
        return

    ganador = decision.chosen_candidate
    motor_ganador = next(m for m in motores if f"{m.nombre}-default" == ganador)
    pipeline_digest = pipeline_digest_de[ganador]
    registro.congelar(pipeline_digest)
    registro.abrir_fase("final_evaluation")

    # re-ajustar el ganador sobre fit_train/fit_val (mismo digest, mismo motor)
    todas_test = [por_id[i] for i in fit_train_ids] + [por_id[i] for i in fit_val_ids] + [por_id[i] for i in test_ids]
    transformadas_test = preparar_para_motor([por_id[i] for i in fit_train_ids], todas_test,
                                             objetivo, predictores, motor_ganador)
    n1, n2 = len(fit_train_ids), len(fit_val_ids)
    _, fitted_ganador = motor_ganador.fit(
        _particion_de(transformadas_test[:n1], objetivo),
        _particion_de(transformadas_test[n1:n1 + n2], objetivo), spec, presupuesto,
        candidate=ganador, split_plan_digest=propuesta.plan.digest())
    assert fitted_ganador.digest() == pipeline_digest, "el reajuste no reprodujo el mismo digest"

    muestra_test = _muestra_desde(motor_ganador, fitted_ganador, transformadas_test[n1 + n2:], objetivo, spec)
    umbral_ganador = umbrales[ganador].threshold
    validacion = evaluar_en_test(registro, pipeline_digest, muestra_test,
                                 metricas=["auroc", "accuracy", "sensitivity", "specificity"],
                                 restricciones=restricciones, umbral=umbral_ganador)
    print(f"Validación final: {validacion.resultado}, evidencia={validacion.evaluacion.evidence}")
    for m in validacion.evaluacion.metrics:
        print(f"  {m.metric_id} = {m.value}")

    informe_final = InformeDeEvaluacion(
        informe_id=f"informe-101-c4-{ganador}", evaluacion=validacion.evaluacion,
        recalibracion=(recalibraciones[ganador].a_json() if recalibraciones.get(ganador) else None),
        comparaciones=((comparacion_lider.a_json(),) if comparacion_lider else ()))

    # --- repetición independiente: mismo caso, misma semilla, de cero ---
    _, fitted_repeticion = motor_ganador.fit(
        _particion_de(transformadas_test[:n1], objetivo),
        _particion_de(transformadas_test[n1:n1 + n2], objetivo), spec, presupuesto,
        candidate=ganador, split_plan_digest=propuesta.plan.digest())
    reproduce = fitted_repeticion.digest() == pipeline_digest
    print(f"\nRepetición independiente (mismo caso, misma semilla): "
         f"{'reproduce el mismo digest' if reproduce else 'NO REPRODUCE -- fallo real'}")

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caso": nombre_ds, "ganador": ganador, "decision": decision.a_json(),
        "informe": informe_final.a_json(), "repeticion_independiente_reproduce": reproduce,
        "cost_false_positive": COST_FALSE_POSITIVE, "cost_false_negative": COST_FALSE_NEGATIVE,
    }
    salida["digest"] = digest_canonico(salida)
    ruta = Path(__file__).resolve().parent / "informe_101_c4_caso_sick.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Guardado en {ruta}, digest={salida['digest'][:16]}")


if __name__ == "__main__":
    main()
