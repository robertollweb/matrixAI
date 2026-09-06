# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C5 — ajuste final, test y alternativas: evaluar el pipeline congelado
UNA vez en el test reservado, declarar si la recomendación se confirma o
falla su validación, y promocionar un candidato alternativo sin perder el
historial ni fingir una evidencia que no tiene.

CASI TODO EL PROTOCOLO YA LO HACE CUMPLIR EL 104-C0. `RegistroDeAccesos.
anotar("test", ...)` ya deniega leer test sin congelar, con un propósito que
aprende, o para un artefacto distinto del congelado; `etiqueta_de_evidencia()`
ya decide sola `independent_test` (primera vez) vs. `repeated_test_use`
(cualquier evaluación de test posterior, del MISMO candidato o de OTRO) vs.
`test_used_for_development` (en cuanto se llamó a `aprender_del_test`). El
criterio literal «no probar candidatos hasta encontrar uno que gane... sin
reconocer ese nuevo uso» es exactamente esa democión automática: evaluar un
segundo candidato en el mismo test nunca vuelve a dar `independent_test`,
lo diga o no quien llama. Este corte no reimplementa nada de eso — ESCRIBE
la función que ata esa vigilancia con 105 (evaluar) y con las restricciones
de 104-C3/C4, y expone `evaluar_en_test()`.

REENTRENAR CON TEST YA TIENE SU FUNCIÓN: `version_tras_aprender_del_test()`
(104-C0), que da la versión nueva SIN la etiqueta de evaluación independiente
heredada — el criterio literal de este corte («reentrenar con test crea
versión nueva sin evaluación independiente atribuida») ya estaba satisfecho
antes de escribir una sola línea aquí; verificarlo con una prueba PROPIA de
este corte, no reescribir la función.

FALLO DE VALIDACIÓN, NO UN SEGUNDO INTENTO SILENCIOSO. Si el pipeline
congelado no supera en test una restricción que sí superaba en desarrollo,
`evaluar_en_test()` declara `fallo_de_validacion` con la evaluación
COMPLETA adjunta (el número se guarda igual, la conclusión es que la
recomendación no se sostiene) — nunca lanza una excepción que invite a
"probar otra cosa" con el mismo test.

PROMOCIÓN: HISTORIAL CONSERVADO, EVIDENCIA COMPARABLE. Promocionar un
candidato alternativo no puede mezclar una evaluación `independent_test`
saliente con una `repeated_test_use` entrante como si fueran la misma
categoría de prueba — eso maquillaría una evidencia más débil de recambio.
`promover_candidato()` exige la MISMA categoría de evidencia en los dos
lados y conserva el vínculo (`derives_from`, ya un campo de
`EvaluationResult`) hacia lo que reemplaza.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from matrixai.estudio.accesos import RegistroDeAccesos
from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import EvaluationResult, Restriccion
from matrixai.estudio.metricas import Muestra, ValorDeMetrica, evaluar
from matrixai.estudio.textos import motivo
from matrixai.estudio.validacion import exigir_texto
from matrixai.estudio.vocabulario import exigir_opcion

__all__ = ["RESULTADOS_DE_VALIDACION_FINAL", "ValidacionFinal",
          "evaluar_en_test", "promover_candidato"]

#: `confirmado`: el pipeline congelado supera en test las mismas
#: restricciones que lo hicieron ganar en desarrollo. `fallo_de_validacion`:
#: no las supera — la evaluación se guarda igual, la recomendación no.
RESULTADOS_DE_VALIDACION_FINAL = ("confirmado", "fallo_de_validacion")


@dataclass(frozen=True)
class ValidacionFinal:
    resultado: str
    evaluacion: EvaluationResult
    motivo_del_fallo: dict[str, str] | None = None

    def __post_init__(self) -> None:
        exigir_opcion(self.resultado, "resultado", RESULTADOS_DE_VALIDACION_FINAL)
        if self.resultado == "fallo_de_validacion" and self.motivo_del_fallo is None:
            raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo="validacion_final")
        if self.resultado == "confirmado" and self.motivo_del_fallo is not None:
            raise EsquemaInvalido("metrica_con_valor_y_motivo", campo="validacion_final")


def _valor_de(evaluacion: EvaluationResult, clave: str) -> ValorDeMetrica | None:
    for metrica in evaluacion.metrics:
        if metrica.metric_id == clave:
            return metrica
    return None


def _incumple_en_test(restriccion: Restriccion, evaluacion: EvaluationResult) -> bool:
    valor = _valor_de(evaluacion, restriccion.clave)
    if valor is None or valor.value is None:
        return True
    if restriccion.operador == "min":
        return valor.value < restriccion.valor
    if restriccion.operador == "max":
        return valor.value > restriccion.valor
    if restriccion.operador == "equal":
        return valor.value != restriccion.valor
    return True  # "boolean" sobre una metrica de este registro: no hay como comprobarlo, falla cerrado


def evaluar_en_test(registro: RegistroDeAccesos, pipeline_digest: str, muestra_test: Muestra, *,
                    metricas: Sequence[str] | None = None,
                    restricciones: Sequence[Restriccion] = (),
                    umbral: float | None = None) -> ValidacionFinal:
    """UNA lectura de test por llamada — `registro.anotar()` es quien la
    vigila; si ya hay una evaluación previa de OTRO candidato en este mismo
    test, `registro.etiqueta_de_evidencia` demueve esta a `repeated_test_use`
    sin que esta función tenga que saberlo ni decidirlo."""
    exigir_texto(pipeline_digest, "pipeline_digest")
    registro.anotar("test", proposito="evaluate", artefacto=pipeline_digest,
                    observaciones=muestra_test.n)
    informe = evaluar(muestra_test, metricas, umbral=umbral)
    evidencia = registro.etiqueta_de_evidencia(pipeline_digest)

    evaluacion = EvaluationResult(
        evaluation_id=f"eval-{pipeline_digest}-test", pipeline_digest=pipeline_digest,
        split_plan_digest=registro.plan.digest(), evaluated_role="test",
        evidence=evidencia, metrics=informe.metrics)

    restricciones_obligatorias = [r for r in restricciones if r.obligatoria]
    incumplida = next((r for r in restricciones_obligatorias if _incumple_en_test(r, evaluacion)), None)
    if incumplida is not None:
        return ValidacionFinal(resultado="fallo_de_validacion", evaluacion=evaluacion,
                               motivo_del_fallo=motivo("recomendacion_falla_en_test", campo=incumplida.clave))
    return ValidacionFinal(resultado="confirmado", evaluacion=evaluacion)


def promover_candidato(vigente: EvaluationResult, alternativo: EvaluationResult) -> EvaluationResult:
    """El candidato `alternativo` releva a `vigente` — exige la MISMA
    categoría de evidencia (nunca sustituir un `independent_test` por un
    `repeated_test_use` como si fueran intercambiables) y conserva el
    vínculo hacia lo que reemplaza."""
    if alternativo.evidence != vigente.evidence:
        raise EsquemaInvalido("evidencia_no_comparable_para_promocion",
                              valor=alternativo.evidence, opciones=vigente.evidence)
    return replace(alternativo, derives_from=vigente.evaluation_id)
