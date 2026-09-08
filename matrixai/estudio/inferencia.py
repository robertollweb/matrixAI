# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""108-C4 — aplicar el pipeline calibrado a una predicción NUEVA.

Vacío real, no una decisión de diseño: `matrixai_engines.prediccion_por_lote`
(106-C5) llama al motor en crudo (`motor.predict_proba()`) y nunca toca
`FittedPipelineSpec.calibrator`/`.decision_policy` (104-C4/105-C3) -- los dos
campos existen y se guardan, pero hasta este módulo ningún sitio los
APLICABA a una probabilidad nueva, solo los reportaban (`tarjeta_de_uso.py`,
`recibo_de_estudio.py`). Sin esto, "usar el modelo" (108-C4) predeciría con
un umbral 0,5 fijo -- exactamente lo que 104-C4 invariante 1 prohíbe.

**El umbral vive en la escala calibrada cuando hay calibrador** (mismo
invariante que ya declara `FittedPipelineSpec.__post_init__`): la
probabilidad cruda se recalibra ANTES de compararla contra el umbral, nunca
al revés. Y `elegir_umbral()` (104-C4) hoy solo produce umbrales sobre
`calibrated_probability` -- nunca `raw_score`, porque exige
`muestra.probabilidad_del_positivo` o rechaza -- así que una `decision_policy`
en escala `raw_score` no tiene, todavía, ningún productor real; se rechaza
aquí en vez de comparar una probabilidad contra un umbral que vive en otra
escala, que cortaría en el sitio equivocado sin que nadie lo notara.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matrixai.estudio.calibracion import RecalibracionLogistica, aplicar_recalibracion
from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import PoliticaDeDecision

__all__ = ["PrediccionDecidida", "decidir_desde_probabilidad"]


@dataclass(frozen=True)
class PrediccionDecidida:
    """Las tres cifras que una fila predicha necesita para pintarse: lo que
    dio el motor, lo que queda tras calibrar (igual a lo primero si no hay
    calibrador) y la etiqueta que resulta de compararlo con el umbral del
    estudio -- nunca un 0,5 fijo."""

    probabilidad_cruda: float
    probabilidad_final: float
    etiqueta: str

    def a_json(self) -> dict[str, Any]:
        return {
            "probabilidad_cruda": self.probabilidad_cruda,
            "probabilidad_final": self.probabilidad_final,
            "etiqueta": self.etiqueta,
        }


def decidir_desde_probabilidad(probabilidad_cruda: float, *, calibrator: dict[str, Any] | None,
                               decision_policy: PoliticaDeDecision,
                               negative_label: str) -> PrediccionDecidida:
    """`probabilidad_cruda` es P(clase positiva) tal como la da el motor,
    SIN calibrar (`motor.predict_proba()`, 106-C5).

    `calibrator` llega en la forma exacta de `FittedPipelineSpec.calibrator`
    -- un dict con las claves de `RecalibracionLogistica.a_json()` (105-C3);
    `None` cuando el pipeline no lleva calibrador, y entonces la probabilidad
    final es la cruda sin tocar. `negative_label` la resuelve quien llama
    (`ProblemSpec.classes` menos `decision_policy.positive_label`): esta
    función no conoce el vocabulario de clases del problema, solo decide
    entre las dos que se le dan.
    """
    if not isinstance(probabilidad_cruda, (int, float)) or not (0.0 <= probabilidad_cruda <= 1.0):
        raise EsquemaInvalido("probabilidad_fuera_de_rango", campo="probabilidad_cruda",
                              valor=probabilidad_cruda)

    if calibrator is not None:
        recalibracion = RecalibracionLogistica(**calibrator)
        (probabilidad_final,) = aplicar_recalibracion(recalibracion, [probabilidad_cruda])
    else:
        probabilidad_final = float(probabilidad_cruda)

    if decision_policy.scale != "calibrated_probability":
        raise EsquemaInvalido("umbral_en_escala_no_soportada", campo="decision_policy.scale",
                              valor=decision_policy.scale, opciones="calibrated_probability")

    etiqueta = (decision_policy.positive_label if probabilidad_final >= decision_policy.threshold
                else negative_label)
    return PrediccionDecidida(probabilidad_cruda=float(probabilidad_cruda),
                              probabilidad_final=probabilidad_final, etiqueta=etiqueta)
