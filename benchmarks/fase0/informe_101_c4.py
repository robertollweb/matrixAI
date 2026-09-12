#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C4 — informe y decisión de cartera: «rendimiento de motores» (ya
producido por 101-C3) y «rendimiento del selector COMPLETO... dentro de
desarrollo», más «una repetición con la misma semilla comprueba el
determinismo de un caso» (texto literal del criterio de terminado, ver
nota de nombre más abajo).

REABIERTO Y CORREGIDO (2026-09-08), tras una auditoría externa verificada
de forma independiente que encontró defectos reales en la primera versión
de este script (detalle completo en `101_FASE0_LA_VERDAD_MEDIDA_
CONTRACT.md`, sección C4). Los cinco cambios de fondo:

1. **El caso ya NO es "sick".** La versión original reutilizaba
   `particiones_base()` de `pasada_exploratoria_101_c3.py` con el MISMO
   `seed=0` que 101-C3 ya había usado 60 veces sobre "sick" (4 motores ×
   3 repeticiones × 5 pliegues) -- el test que este script etiquetaba
   `independent_test` no lo era. Como el diseño de partición de C3 fija
   el rol "test" de cada dataset (no varía por pliegue/repetición), no
   hay forma de obtener un test de "sick" genuinamente no tocado sin
   reabrir C3 entero. Se sustituye por "adult" (data_id 1590, censo de
   EEUU, 48.842 filas, `grande`), NO sellado, con faltantes reales
   (marcados `?`) y categóricas de cardinalidad moderada (hasta 41
   niveles en `native-country`) -- de los 4 datasets `grande` no
   sellados y nunca tocados por 101-C3 (los otros: KDDCup09_appetency,
   Amazon_employee_access, APSFailure), el más tratable y menos
   propenso a los problemas de cardinalidad extrema que ya rompieron
   otros motores en 101-C3 (ver ese corte). Minoría = positiva (mismo
   criterio que 101-C3): `>50K` (24% de las filas).

2. **El umbral ya gobierna la decisión de verdad.** `matriz_de_confusion`
   (104-C0/105-C1) da prioridad a `Muestra.predictions` ("declared_label")
   sobre cualquier `umbral` explícito -- documentado y correcto AHÍ (si la
   muestra trae la decisión del modelo, esa manda), pero la primera
   versión de este script poblaba `predictions` con la etiqueta CRUDA del
   motor (`motor.predict()`, sin calibrar, sin el umbral coste-óptimo) en
   la MISMA `Muestra` que además llevaba el `umbral` elegido -- el umbral
   nunca se aplicaba, silenciosamente. `_muestra_desde()` ya no puebla
   `predictions`; accuracy/sensibilidad/especificidad salen SIEMPRE de la
   regla de umbral, nunca de la etiqueta cruda del motor.

3. **El umbral se aplica en la MISMA escala en la que se eligió.**
   `PoliticaDeDecision` declara que, con calibrador, "el umbral vive en la
   probabilidad calibrada" -- pero la evaluación final (`evaluar_en_test`)
   aplicaba ese umbral sobre las probabilidades CRUDAS del motor ganador,
   nunca recalibradas. Corregido: la muestra de test se recalibra con el
   MISMO calibrador del ganador antes de evaluar, igual que ya se hacía
   (a medias) en la fase de selección. Y `elegir_umbral()` ya admite un
   parámetro `recalibracion=` que hace esto internamente -- se usa ese en
   vez de reconstruir la recalibración a mano.

4. **La comparación es contra el segundo mejor candidato real, no contra
   el baseline.** `comparacion_lider` comparaba el ganador contra
   `baseline-default` (AUROC 0,5 exacto por construcción, ningún
   candidato real) mientras el texto de la decisión (`seleccion.py`/
   `textos.py`) dice "frente al siguiente mejor candidato". Corregido:
   se calcula el segundo mejor motor REAL (excluyendo baseline) por
   AUROC de selección, y la comparación emparejada es contra ESE.

5. **El umbral elegido queda persistido en la salida.** La primera
   versión no guardaba `decision_policy` en ningún sitio del JSON --
   ahora se guarda explícito (`PoliticaDeDecision.a_json()`), igual que
   ya se guardaba la recalibración.

UN CASO, NO LOS DOCE. El criterio pide un caso, no repetir la batería
entera -- "adult" (categóricas moderadas, faltantes reales, tamaño
`grande`) es el candidato disponible más exigente entre los no tocados
por 101-C3, no el más fácil de aprobar.

CUATRO PARTICIONES DE DESARROLLO, NO DOS. El fold de "adult" da
train/validation/test -- para ejercer 105-C3 (recalibración) y 104-C4
(umbral) sin que uno contamine al otro («ajustar sobre datos NUEVOS,
nunca los que ajustaron la recalibración», texto literal de
`aplicar_recalibracion`), el train original se reparte en
`fit_train`/`fit_val` (para `Motor.fit`) y la validation original en
`calibracion`/`seleccion` (para `ajustar_recalibracion_
logistica` y `elegir_umbral`/`comparar_candidatos`/`seleccionar`,
respectivamente). El test original quedó SIN TOCAR por
esas cuatro particiones -- se lee una sola vez, al final, vía
`evaluar_en_test`.

SIN AISLAMIENTO POR SUBPROCESO AQUÍ, A PROPÓSITO. `ejecutar_intento_
aislado` (104-C2 ext.) existe para una BATERÍA de cientos de intentos
donde un motor colgado no puede tirar la pasada entera -- un solo caso,
en el mismo proceso, no necesita esa protección; medirlo así habría
sido ceremonia sin beneficio real.

PROCEDENCIA EN LA SALIDA (2026-09-12). El `pipeline_digest` que este JSON
publica como prueba ya NO se reproduce: medido el 09-12 reajustando
lightgbm sobre el mismo `adult`, el guardado es `b888d0b1…` y hoy sale
`eeea0cdf…` (el `split_plan_digest`, en cambio, SÍ reproduce:
`bdef5eb8…`, así que la deriva está en el motor, no en la partición).
`matrixai-engines` lleva 33 commits desde el 09-07 —entre ellos
`9eddaf3`, que metió sklearn/lightgbm/torch/onnx en `library_versions`, y
`library_versions` entra en el `component_id` del predictor y de ahí en el
digest del pipeline—, pero el JSON no guardaba NI el commit NI las
versiones, así que la explicación hay que reconstruirla a mano cada vez.
Ahora la salida lleva el mismo bloque `procedencia` que 101-C3 (importado
de allí, no copiado: dos sitios declarando lo mismo acaban divergiendo),
con los commits de los dos repositorios —marcados si el árbol estaba
sucio—, las versiones cargadas, los digests de código y el sha256 del ARFF
de entrada.

"REPETICIÓN INDEPENDIENTE", RENOMBRADA A LO QUE MIDE DE VERDAD. La
auditoría señaló, con razón, que reajustar el MISMO motor sobre los
MISMOS datos con la MISMA semilla demuestra determinismo, no una
réplica independiente (que exigiría datos, entorno o al menos semilla
distintos). Se mantiene la medición -- sigue siendo real y útil -- pero
la salida y los mensajes ya no la llaman "independiente".
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
    ARFF_DIR, _FICHERO_POR_MOTOR, _digest_entorno, _digest_fichero,
    particiones_base, preparar_para_motor, procedencia_de_la_medicion)

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

DATASET = (1590, "adult", "grande", ">50K", "<=50K")
COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 4.0  # perder un caso positivo real cuesta más que una falsa alarma


def _particion_de(filas: list[dict], objetivo: str) -> Particion:
    return Particion.desde_filas(filas, row_id_field="row_id", target_field=objetivo)


def _muestra_desde(motor, fitted, filas: list[dict], objetivo: str, spec) -> Muestra:
    """SIN `predictions`, a propósito: `matriz_de_confusion()` da prioridad
    a la etiqueta declarada de la muestra sobre cualquier `umbral` que se le
    pase (comportamiento correcto y documentado AHÍ), así que una `Muestra`
    que quiere que el umbral decida no puede llevar la etiqueta cruda del
    motor. La AUROC no necesita `predictions` -- usa `probabilities`."""
    datos = _particion_de(filas, objetivo)
    campos = dict(task=spec.task, y_true=tuple(datos.target),
                 classes=fitted.classes, positive_label=fitted.positive_label)
    if fitted.capacidades.admite_probabilidades:
        campos["probabilities"] = tuple(motor.predict_proba(fitted, datos))
    return Muestra(**campos)


def _recalibracion_aplicable(recalibracion):
    """`aplicar_recalibracion()` (y `elegir_umbral(recalibracion=...)`, que
    la llama por dentro) EXIGE `a`/`b` reales -- levanta si la
    recalibración falló (`convergio=False`, `a`/`b` en `None`). Tratar
    "falló" igual que "no hay calibrador" (probabilidades crudas) es la
    única salida sensata; nunca crashear el caso entero por una
    recalibración que no convergió."""
    if recalibracion is None or recalibracion.a is None or recalibracion.b is None:
        return None
    return recalibracion


def _muestra_recalibrada(muestra: Muestra, recalibracion) -> Muestra:
    """Misma muestra, con `scores` = la probabilidad del positivo YA
    recalibrada -- para evaluar en la MISMA escala en la que se eligió el
    umbral (`PoliticaDeDecision`: con calibrador, el umbral vive en la
    probabilidad calibrada, nunca en la cruda). Sin calibrador (o si la
    recalibración no convergió), se devuelve la muestra tal cual: no hay
    una segunda escala que igualar."""
    recalibracion = _recalibracion_aplicable(recalibracion)
    if recalibracion is None:
        return muestra
    p_recalibrada = aplicar_recalibracion(recalibracion, muestra.probabilidad_del_positivo)
    return Muestra(task=muestra.task, y_true=muestra.y_true, classes=muestra.classes,
                   positive_label=muestra.positive_label, scores=p_recalibrada,
                   score_rule="105-c3.recalibrada")


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

    # Se sella ANTES de medir: describe el árbol con el que se va a medir, no
    # el que quede al acabar. `_digest_entorno()` cubre los ficheros
    # COMPARTIDOS (incluido `pasada_exploratoria_101_c3.py`, de donde salen
    # `particiones_base`/`preparar_para_motor`), pero no este fichero, que no
    # está en esa lista -- va aparte y con su nombre.
    procedencia = procedencia_de_la_medicion(
        digests_de_codigo={
            "entorno": _digest_entorno(),
            "informe_101_c4.py": _digest_fichero(Path(__file__).resolve()),
            "por_motor": {nombre: _digest_fichero(ruta)
                          for nombre, ruta in _FICHERO_POR_MOTOR.items()}},
        datos_de_entrada={nombre_ds: ARFF_DIR / f"{data_id}.arff"})
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)

    por_id, propuesta, spec, objetivo, predictores = particiones_base(
        data_id, nombre_ds, cubo, positiva, negativa)
    pliegue = propuesta.pliegues.pliegue_de(repeticion=0, pliegue=0)
    test_ids = propuesta.plan.observaciones_del_rol("test")
    fit_train_ids, fit_val_ids, calibracion_ids, seleccion_ids = _particiones_de_desarrollo(
        por_id, pliegue, seed=0)

    print(f"{nombre_ds}: fit_train={len(fit_train_ids)} fit_val={len(fit_val_ids)} "
         f"calibracion={len(calibracion_ids)} seleccion={len(seleccion_ids)} "
         f"test={len(test_ids)}", flush=True)

    registro = RegistroDeAccesos(propuesta.plan, estudio=f"101-c4-caso-{nombre_ds}")
    registro.abrir_fase("development")

    motores = [MotorBaseline(), MotorLineal(), MotorArbolLightGBM(), MotorDensaPropia()]
    # "grande" en el protocolo registrado presupuesta 10 min/intento (vs 2
    # min de "sick", mediano) -- "adult" tiene ~13x las filas de "sick".
    presupuesto = Presupuesto(wall_seconds=600.0, hilos=4, seed=0)

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
        try:
            fit_result, fitted = motor.fit(_particion_de(filas_fit_train, objetivo),
                                           _particion_de(filas_fit_val, objetivo), spec, presupuesto,
                                           candidate=candidato, split_plan_digest=propuesta.plan.digest())
        except Exception as exc:  # noqa: BLE001
            # Límite REAL del core, ya documentado en 101-C3 para
            # PhishingWebsites (mismo mecanismo: dos etiquetas distintas que
            # colisionan al mismo nombre saneado, p.ej. '>50K'/'<=50K' ->
            # 'class_50k') -- declarado, no una excepción que se traga en
            # silencio: un motor que no puede con este dataset queda fuera
            # de la comparación, el resto del caso sigue midiéndose.
            print(f"  {motor.nombre}: fit LEVANTÓ una excepción ({type(exc).__name__}: {exc}) "
                 f"-- fuera de la comparación, no es un fallo de este script.")
            continue
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

        if recalibracion is not None and not recalibracion.convergio:
            print(f"  {motor.nombre}: recalibración NO convergió "
                 f"({recalibracion.undefined_reason}) -- se sigue sin recalibrar, "
                 f"con las probabilidades crudas del motor.")

        registro.anotar("development", proposito="tune_threshold", candidato=candidato,
                        observaciones=len(filas_seleccion))
        muestra_seleccion = _muestra_desde(motor, fitted, filas_seleccion, objetivo, spec)
        # `elegir_umbral(recalibracion=...)` recalibra internamente ANTES de
        # buscar el umbral óptimo -- no hace falta reconstruir la
        # recalibración a mano aquí (y así el umbral sale ya en la escala
        # correcta, calibrada si hay recalibración que aplicar de verdad).
        umbral = elegir_umbral(muestra_seleccion, cost_false_positive=COST_FALSE_POSITIVE,
                               cost_false_negative=COST_FALSE_NEGATIVE,
                               recalibracion=_recalibracion_aplicable(recalibracion))
        umbrales[candidato] = umbral

        # Para EVALUAR con ese umbral, la muestra tiene que estar en la
        # MISMA escala en la que se eligió -- recalibrada si hubo
        # calibrador, cruda si no. Nunca con `predictions`: el umbral
        # decide, no la etiqueta cruda del motor (ver `_muestra_desde`).
        muestra_seleccion_evaluada = _muestra_recalibrada(muestra_seleccion, recalibracion)
        registro.anotar("development", proposito="select", candidato=candidato,
                        observaciones=len(filas_seleccion))
        informe = evaluar(muestra_seleccion_evaluada, ["auroc", "accuracy", "specificity"],
                          umbral=umbral.threshold)
        evaluaciones_seleccion[candidato] = EvaluationResult(
            evaluation_id=f"sel-{candidato}", pipeline_digest=fitted.digest(),
            split_plan_digest=propuesta.plan.digest(), evaluated_role="selection",
            evidence="development_estimate", metrics=informe.metrics)
        muestras_seleccion[candidato] = muestra_seleccion_evaluada
        auroc = next((m.value for m in informe.metrics if m.metric_id == "auroc"), None)
        print(f"  {motor.nombre}: fit ok, auroc(seleccion)={auroc}, "
             f"umbral={umbral.threshold} (escala={umbral.scale})")

    # comparación emparejada: el ganador contra el SEGUNDO MEJOR candidato
    # real -- nunca el baseline, que por construcción da AUROC 0,5 exacto
    # y no es un candidato desplegable. "ganador" y "segundo mejor" se
    # calculan los dos excluyendo baseline del ranking.
    baseline_cand = "baseline-default"
    candidatos_reales = [c for c in evaluaciones_seleccion if c != baseline_cand]
    ganador_actual = max(candidatos_reales, key=lambda c: next(
        (m.value for m in evaluaciones_seleccion[c].metrics if m.metric_id == "auroc"), -1.0))
    otros_candidatos_reales = [c for c in candidatos_reales if c != ganador_actual]
    segundo_mejor = None
    if otros_candidatos_reales:
        segundo_mejor = max(otros_candidatos_reales, key=lambda c: next(
            (m.value for m in evaluaciones_seleccion[c].metrics if m.metric_id == "auroc"), -1.0))

    comparacion_lider = None
    if segundo_mejor is not None:
        comparacion_lider = comparar_candidatos(
            "auroc", muestras_seleccion[ganador_actual], muestras_seleccion[segundo_mejor],
            diseno="iid", estimando="fixed_model_on_population", semilla=0)
        print(f"  comparación {ganador_actual} vs {segundo_mejor} (segundo mejor real, "
             f"NO el baseline): {comparacion_lider.veredicto}")
    else:
        print("  un solo candidato real (aparte del baseline) -- sin comparación con el segundo mejor.")

    restricciones = (Restriccion(clave="specificity", operador="min", valor=0.3),)
    decision = seleccionar(evaluaciones_seleccion, restricciones=restricciones,
                           metric_id_calidad="auroc", decision_id="101-c4-adult",
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

    # Test recalibrado con el MISMO calibrador del ganador -- el umbral se
    # eligió en escala calibrada (si hubo calibrador), y evaluarlo sobre
    # probabilidades crudas cortaría en el sitio equivocado.
    muestra_test_cruda = _muestra_desde(motor_ganador, fitted_ganador, transformadas_test[n1 + n2:], objetivo, spec)
    recalibracion_ganador = recalibraciones.get(ganador)
    muestra_test = _muestra_recalibrada(muestra_test_cruda, recalibracion_ganador)
    umbral_ganador = umbrales[ganador]
    validacion = evaluar_en_test(registro, pipeline_digest, muestra_test,
                                 metricas=["auroc", "accuracy", "sensitivity", "specificity"],
                                 restricciones=restricciones, umbral=umbral_ganador.threshold)
    print(f"Validación final: {validacion.resultado}, evidencia={validacion.evaluacion.evidence}")
    for m in validacion.evaluacion.metrics:
        print(f"  {m.metric_id} = {m.value}")

    informe_final = InformeDeEvaluacion(
        informe_id=f"informe-101-c4-{ganador}", evaluacion=validacion.evaluacion,
        recalibracion=(recalibracion_ganador.a_json() if recalibracion_ganador else None),
        comparaciones=((comparacion_lider.a_json(),) if comparacion_lider else ()))

    # --- repetición con la misma semilla: mide DETERMINISMO, no réplica
    # independiente (mismo motor, mismos datos, mismo proceso -- ver nota
    # del módulo). Útil y real, pero no es lo que su nombre decía antes. ---
    _, fitted_repeticion = motor_ganador.fit(
        _particion_de(transformadas_test[:n1], objetivo),
        _particion_de(transformadas_test[n1:n1 + n2], objetivo), spec, presupuesto,
        candidate=ganador, split_plan_digest=propuesta.plan.digest())
    reproduce = fitted_repeticion.digest() == pipeline_digest
    print(f"\nRepetición con la misma semilla (determinismo, NO réplica independiente): "
         f"{'reproduce el mismo digest' if reproduce else 'NO REPRODUCE -- fallo real'}")

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "procedencia": procedencia,
        "caso": nombre_ds, "ganador": ganador, "decision": decision.a_json(),
        "informe": informe_final.a_json(),
        "repeticion_misma_semilla_reproduce": reproduce,
        "decision_policy": umbral_ganador.a_json(),
        "segundo_mejor_candidato": segundo_mejor,
        "cost_false_positive": COST_FALSE_POSITIVE, "cost_false_negative": COST_FALSE_NEGATIVE,
    }
    salida["digest"] = digest_canonico(salida)
    ruta = Path(__file__).resolve().parent / f"informe_101_c4_caso_{nombre_ds}.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Guardado en {ruta}, digest={salida['digest'][:16]}")


if __name__ == "__main__":
    main()
