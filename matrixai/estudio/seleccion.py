# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C3 — requisitos y selección: filtrar candidatos por restricciones
OBLIGATORIAS antes de ordenar por utilidad, nunca al revés — un modelo que
siempre predice positivo puede ganar en sensibilidad y perder por
especificidad, y ordenar primero escondería ese fallo detrás de un número
bueno.

FORMA YA EXISTENTE, REUTILIZADA. `SelectionDecision`, `Restriccion`,
`EvaluationResult` son del 104-C0 — este corte no inventa un esquema nuevo,
ESCRIBE LA FUNCIÓN que produce un `SelectionDecision` a partir de un
`Leaderboard`/evaluaciones de desarrollo y las restricciones del problema.
El invariante «una decisión no se puede apoyar en una evaluación del test»
ya lo hace cumplir `SelectionDecision.__post_init__` (rechaza cualquier
`EvaluationResult` con `evaluated_role` reservado) — no se repite aquí.

TRES SALIDAS, NUNCA CONFUNDIDAS (texto literal del 104-C0). `selected`: hay
un candidato que supera TODAS las restricciones obligatorias con evidencia
completa. `no_feasible_model`: hay evidencia completa para TODOS los
candidatos y NINGUNO supera las restricciones — un veredicto real, medido.
`insufficient_evidence`: al menos un candidato tiene una restricción sin
medir — no se puede afirmar «ninguno sirve» con un dato que falta, así que
no se declara `no_feasible_model` por descuido.

CADA RESTRICCIÓN, TRES ESTADOS. `cumple` / `no_cumple` / `sin_evidencia` —
nunca se colapsa `sin_evidencia` en `no_cumple`: un candidato sin la
métrica medida no "falla" la restricción, es que no se puede juzgar. Las
restricciones sobre una MÉTRICA se resuelven contra el registro de 105-C1
(`min`/`max`/`equal` comparan `ValorDeMetrica.value`); la restricción
`solo_local` es la única con semántica propia (compara con la ausencia de
`necesita_red` de 102-C1 — un dato de CAPACIDAD, no de métrica, que este
módulo no calcula, solo recibe).

LA UTILIDAD DE ESTA PASADA: UNA MÉTRICA DE CALIDAD DECLARADA, NO UN COSTE
INVENTADO. El contrato menciona "coste del error" como entrada posible de
la UI (108), pero el criterio de terminado de ESTE corte no exige una
fórmula de coste esperado — exige que los mínimos filtren antes de
ordenar, y que el motivo distinga una mejora demostrada de una elección
operativa. Ordenar por coste esperado (combinando
`PoliticaDeDecision.cost_false_positive/negative` con sensibilidad/
especificidad) queda declarado como una política de ranking FUTURA,
alternativa a `POLITICA_UTILIDAD_MAXIMA` — no construida a medias aquí.

MEJORA DEMOSTRADA, NO SUPUESTA. Distinguir "el ganador es mejor de verdad"
de "el ganador se prefirió por ser más simple ante evidencia inconclusa"
exige la diferencia EMPAREJADA de 105-C5 entre el ganador y el segundo
candidato — este módulo no la calcula (no tiene las muestras por fila, solo
`ValorDeMetrica` agregados), la RECIBE ya calculada (`comparacion_lider`,
opcional) y solo interpreta su veredicto para redactar el motivo.
"""

from __future__ import annotations

import functools
from typing import Mapping, Sequence

from matrixai.estudio.comparaciones import ComparacionEmparejada
from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import EvaluationResult, Restriccion, SelectionDecision
from matrixai.estudio.metricas import ValorDeMetrica, direccion_de, es_mejor
from matrixai.estudio.textos import motivo
from matrixai.estudio.vocabulario import ROLES_RESERVADOS

__all__ = ["POLITICA_UTILIDAD_MAXIMA", "seleccionar"]

#: La ÚNICA política que este corte implementa: ordenar los candidatos que
#: superan las restricciones por una sola métrica de calidad declarada. Otra
#: política (coste esperado, Pareto sobre varias métricas) es un nombre
#: distinto y una implementación aparte — no el mismo código con otro rótulo.
POLITICA_UTILIDAD_MAXIMA = "utilidad_maxima_bajo_restricciones"


def _valor_de(evaluacion: EvaluationResult, clave: str) -> ValorDeMetrica | None:
    for metrica in evaluacion.metrics:
        if metrica.metric_id == clave:
            return metrica
    return None


def _evalua_restriccion(restriccion: Restriccion, evaluacion: EvaluationResult,
                        necesita_red: bool | None) -> tuple[str, dict[str, str] | None]:
    """`(estado, motivo)` con `estado` en `("cumple", "no_cumple", "sin_evidencia")`."""
    if restriccion.clave == "solo_local":
        if necesita_red is None:
            return "sin_evidencia", motivo("restriccion_sin_evidencia", campo=restriccion.clave)
        cumple = (not necesita_red) if restriccion.valor else necesita_red
        if cumple:
            return "cumple", None
        return "no_cumple", motivo("restriccion_no_cumplida", campo=restriccion.clave,
                                   valor=f"necesita_red={necesita_red}")

    valor = _valor_de(evaluacion, restriccion.clave)
    if valor is None or valor.value is None:
        return "sin_evidencia", motivo("restriccion_sin_evidencia", campo=restriccion.clave)
    if restriccion.operador == "min":
        cumple = valor.value >= restriccion.valor
    elif restriccion.operador == "max":
        cumple = valor.value <= restriccion.valor
    elif restriccion.operador == "equal":
        cumple = valor.value == restriccion.valor
    else:  # boolean
        cumple = bool(valor.value) == restriccion.valor
    if cumple:
        return "cumple", None
    return "no_cumple", motivo("restriccion_no_cumplida", campo=restriccion.clave, valor=valor.value)


def seleccionar(evaluaciones: Mapping[str, EvaluationResult], *, restricciones: Sequence[Restriccion],
                metric_id_calidad: str, decision_id: str, split_plan_digest: str,
                politica: str = POLITICA_UTILIDAD_MAXIMA,
                necesita_red: Mapping[str, bool] | None = None,
                comparacion_lider: ComparacionEmparejada | None = None) -> SelectionDecision:
    """`evaluaciones` mapea `candidate -> EvaluationResult` (evidencia de
    DESARROLLO, nunca de test — `SelectionDecision` lo rechaza si se intenta).
    `comparacion_lider`, si se da, es la `ComparacionEmparejada` (105-C5) del
    candidato ganador contra el segundo mejor por `metric_id_calidad` — la
    calcula el llamante, que sí tiene las muestras por fila."""
    necesita_red = necesita_red or {}
    for evaluacion in evaluaciones.values():
        if evaluacion.evaluated_role in ROLES_RESERVADOS:
            raise EsquemaInvalido("seleccionar_con_el_test", valor=evaluacion.evaluated_role)

    constraints_checked = tuple(r.clave for r in restricciones)
    feasibles: list[str] = []
    rechazados: list[dict] = []
    hubo_evidencia_insuficiente = False

    for candidato, evaluacion in evaluaciones.items():
        motivo_no_cumple = None
        motivo_sin_evidencia = None
        for restriccion in restricciones:
            if not restriccion.obligatoria:
                continue
            estado, razon = _evalua_restriccion(restriccion, evaluacion, necesita_red.get(candidato))
            if estado == "sin_evidencia":
                motivo_sin_evidencia = razon
                break
            if estado == "no_cumple":
                motivo_no_cumple = razon
                break
        if motivo_sin_evidencia is not None:
            hubo_evidencia_insuficiente = True
            rechazados.append({"candidate": candidato, "reason": motivo_sin_evidencia})
        elif motivo_no_cumple is not None:
            rechazados.append({"candidate": candidato, "reason": motivo_no_cumple})
        else:
            feasibles.append(candidato)

    if not feasibles:
        outcome = "insufficient_evidence" if hubo_evidencia_insuficiente else "no_feasible_model"
        return SelectionDecision(
            decision_id=decision_id, outcome=outcome,
            reason=motivo("evidencia_insuficiente_para_decidir" if outcome == "insufficient_evidence"
                         else "ningun_candidato_cumple"),
            policy=politica, split_plan_digest=split_plan_digest,
            rejected=tuple(rechazados), constraints_checked=constraints_checked)

    def _calidad(candidato: str) -> float | None:
        valor = _valor_de(evaluaciones[candidato], metric_id_calidad)
        return valor.value if valor is not None else None

    feasibles_medibles = [c for c in feasibles if _calidad(c) is not None]
    if not feasibles_medibles:
        rechazados.extend({"candidate": c, "reason": motivo("calidad_no_medida", campo=metric_id_calidad)}
                          for c in feasibles)
        return SelectionDecision(
            decision_id=decision_id, outcome="insufficient_evidence",
            reason=motivo("evidencia_insuficiente_para_decidir"),
            policy=politica, split_plan_digest=split_plan_digest,
            rejected=tuple(rechazados), constraints_checked=constraints_checked)

    direccion_de(metric_id_calidad)  # valida que la métrica exista y tenga registro
    ganador = functools.reduce(
        lambda a, b: a if es_mejor(metric_id_calidad, _calidad(a), _calidad(b)) else b,
        feasibles_medibles)

    if comparacion_lider is not None and comparacion_lider.veredicto == "mejora":
        razon_ganador = motivo("seleccion_mejora_demostrada", campo=ganador)
    elif comparacion_lider is not None and comparacion_lider.veredicto in ("inconcluso", "equivalencia_practica"):
        razon_ganador = motivo("seleccion_eleccion_operativa", campo=ganador)
    else:
        razon_ganador = motivo("seleccion_mayor_utilidad_medida", campo=ganador)

    for candidato in feasibles_medibles:
        if candidato != ganador:
            rechazados.append({"candidate": candidato,
                               "reason": motivo("seleccion_otro_con_mas_utilidad", opciones=ganador)})

    return SelectionDecision(
        decision_id=decision_id, outcome="selected", chosen_candidate=ganador,
        reason=razon_ganador, policy=politica, split_plan_digest=split_plan_digest,
        evidence=(evaluaciones[ganador],), rejected=tuple(rechazados),
        constraints_checked=constraints_checked)
