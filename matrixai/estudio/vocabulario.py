# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los vocabularios CERRADOS que comparte el programa 101-106 — 104-C0.

Están todos en un fichero porque el defecto que se evita es el de siempre: **dos
sitios declarando lo mismo acaban divergiendo**. Los estados de ejecución los
usan 104-C2 y 102-C1; los roles los escribe 103-C4 y los lee 105; las etiquetas
de evidencia las publica 106. Si cada contrato escribe su lista, el día que
alguien añada `paused` habrá dos productos con estados distintos.

Cerrado significa cerrado: un valor que no esté aquí se rechaza con motivo. No
es rigidez — aceptar `completed_partially` sería prometer un comportamiento que
nadie ha implementado, y quien lea el documento no tiene forma de saberlo.

Los tokens van **en inglés** porque son datos que viajan en JSON y los fija el
contrato con esas palabras (`completed_budget_limited`, `external_test`). Lo que
se escribe en castellano es el código y lo que lee una persona, que sale del
catálogo bilingüe de `textos.py`.
"""

from __future__ import annotations

from matrixai.estudio.errores import EsquemaInvalido

__all__ = [
    "DIRECCIONES", "DISPONIBILIDAD", "ESCALAS_DE_DECISION", "ESTADOS",
    "ESTADOS_CON_CANDIDATO", "ESTADOS_QUE_EXIGEN_MOTIVO", "ESTADOS_TERMINALES",
    "ESTIMANDOS", "ETIQUETAS_DE_EVIDENCIA", "FASES", "FASES_DE_DESARROLLO",
    "OPERADORES_DE_RESTRICCION", "PROPOSITOS", "PROPOSITOS_QUE_APRENDEN",
    "REQUISITOS_DE_METRICA",
    "RESULTADOS_DE_SELECCION", "ROLES", "ROLES_DE_DESARROLLO", "ROLES_RESERVADOS",
    "TAREAS", "TAREAS_DE_CLASIFICACION", "TIPOS_DE_PARTICION", "UNIDADES_DE_TIEMPO",
    "conserva_candidato", "es_terminal", "exige_motivo", "exigir_opcion",
    "fase_del_rol_reservado",
]

# ---------------------------------------------------------------------------
# Particiones y protocolo
# ---------------------------------------------------------------------------

#: Los cinco roles del 104-C0. `selection` y `calibration` son PARTES del
#: desarrollo, no excepciones a él: se separan porque quien compara candidatos y
#: quien ajusta el calibrador no pueden ser los mismos datos que entrenaron.
ROLES = ("development", "selection", "calibration", "test", "external_test")

#: Donde SÍ se aprende y se decide.
ROLES_DE_DESARROLLO = ("development", "selection", "calibration")

#: Lo reservado. No entra en ajuste, ni en búsqueda, ni en selección, ni en
#: umbral. Se mira UNA vez, con el pipeline ya congelado.
ROLES_RESERVADOS = ("test", "external_test")

#: Las fases del protocolo mínimo, en el orden en que ocurren.
FASES = (
    "development",        # preparación, ajuste y búsqueda
    "selection",          # comparar candidatos
    "calibration",        # calibrador y umbral
    "final_fit",          # el ajuste final sobre desarrollo
    "final_evaluation",   # la prueba reservada, una vez, congelado
    "external_validation",  # la cohorte externa, si existe
)

#: Las fases en las que se aprende o se decide algo. Son exactamente aquellas en
#: las que la prueba reservada no se puede leer.
FASES_DE_DESARROLLO = ("development", "selection", "calibration", "final_fit")

#: Para qué se leen unos datos. Solo `evaluate` e `inspect` no aprenden nada;
#: todos los demás modifican lo que el producto va a entregar.
PROPOSITOS = ("fit", "tune", "select", "calibrate", "tune_threshold",
              "evaluate", "inspect")

#: Los propósitos que no se pueden ejercer sobre un rol reservado NUNCA, ni
#: siquiera en su fase: mirar el test para elegir el umbral es usarlo para
#: decidir aunque se haga al final.
PROPOSITOS_QUE_APRENDEN = ("fit", "tune", "select", "calibrate", "tune_threshold")

#: Diseños de partición del 103-C4. `iid` no es el valor por omisión de nada: si
#: el problema es temporal o agrupado y no se soporta, se rechaza con motivo en
#: vez de usar `iid` en silencio.
TIPOS_DE_PARTICION = ("iid", "temporal", "groups", "groups_and_time")

# ---------------------------------------------------------------------------
# Estados compartidos (104-C2 y 102-C1)
# ---------------------------------------------------------------------------

ESTADOS = ("queued", "running", "completed", "completed_budget_limited",
           "failed", "cancelled", "unsupported")

#: Los estados que PROMETEN un candidato válido. `completed_budget_limited` es
#: uno de ellos: el reloj se acabó y quedó un artefacto usable. Que un estado
#: prometa candidato y no lo traiga es el disfraz que el invariante 3 del 104
#: prohíbe.
ESTADOS_CON_CANDIDATO = ("completed", "completed_budget_limited")

#: Los estados que EXIGEN motivo escrito. `completed_budget_limited` también:
#: sin decir qué límite se agotó, «completado con límite» no se distingue de
#: «completado».
ESTADOS_QUE_EXIGEN_MOTIVO = ("completed_budget_limited", "failed", "cancelled",
                             "unsupported")

ESTADOS_TERMINALES = ("completed", "completed_budget_limited", "failed",
                      "cancelled", "unsupported")

# ---------------------------------------------------------------------------
# Problema y métricas
# ---------------------------------------------------------------------------

TAREAS = ("binary_classification", "multiclass_classification", "regression")
TAREAS_DE_CLASIFICACION = ("binary_classification", "multiclass_classification")

#: Cuándo está disponible un predictor (103, invariante 5). `unknown` NO es lo
#: mismo que no declararlo: `unknown` dice que alguien lo miró y no lo sabe; la
#: ausencia dice que nadie lo miró, y el esquema conserva la diferencia.
DISPONIBILIDAD = ("at_prediction_time", "after_outcome", "unknown")

UNIDADES_DE_TIEMPO = ("seconds", "minutes", "hours", "days", "weeks", "months",
                      "years")

#: Solo para las métricas que SÍ son monótonas. La calibración no lo es, y por
#: eso una `MetricSpec` puede declarar valor/rango ideal en vez de dirección
#: (105, invariante 4).
DIRECCIONES = ("higher_is_better", "lower_is_better")

#: Qué estima el número (105, invariante 6): no es lo mismo el intervalo de un
#: modelo fijo sobre una población, la variación de volver a entrenar y la
#: tolerancia numérica entre entornos. Mezclarlos es lo que hace que un
#: intervalo diga menos de lo que parece.
ESTIMANDOS = ("fixed_model_on_population", "training_variability",
              "numeric_tolerance")

#: Qué necesita una métrica de un `PredictionRecord` para poder recomputarse.
#: El catálogo de métricas —qué exige cada una— es del 105-C1; aquí solo se fija
#: el vocabulario con el que se declara, para que las dos mitades encajen.
REQUISITOS_DE_METRICA = ("y_true", "labels", "scores", "probabilities", "classes",
                         "positive_label", "split_role", "resampling_unit",
                         "weights")

OPERADORES_DE_RESTRICCION = ("min", "max", "equal", "boolean")

#: Sobre qué escala se declara un umbral. Si hay calibrador, el umbral vive en
#: la probabilidad calibrada: un umbral elegido sobre el score crudo no es el
#: umbral de ese pipeline (104-C4).
ESCALAS_DE_DECISION = ("calibrated_probability", "raw_score")

# ---------------------------------------------------------------------------
# Evidencia y selección
# ---------------------------------------------------------------------------

#: Qué vale un número, que no es lo mismo que cuánto vale.
#:
#: * `independent_test`: medido una vez, sobre el test reservado, con el
#:   pipeline congelado y sin que nadie hubiera mirado antes ese test.
#: * `external_validation`: lo mismo sobre una cohorte externa.
#: * `repeated_test_use`: el número es correcto y la independencia ya no: ese
#:   test se había usado antes. El 104-C5 describe el caso («no probar
#:   candidatos hasta encontrar uno que gane sobre el mismo test sin reconocer
#:   ese nuevo uso») y no le pone nombre; se le pone aquí.
#: * `test_used_for_development`: se aprendió CON el test. El estado que el
#:   contrato nombra literalmente; el artefacto nuevo no hereda evaluación
#:   independiente.
#: * `development_estimate`: medido en desarrollo (OOF o holdout interno).
#: * `not_evaluated`: no se ha medido. No es un cero ni un aprobado.
ETIQUETAS_DE_EVIDENCIA = ("independent_test", "external_validation",
                          "repeated_test_use", "test_used_for_development",
                          "development_estimate", "not_evaluated")

#: El 104-C3 exige que «ningún modelo cumple» sea una respuesta y que se
#: distinga de «no hay evidencia para afirmarlo». Son cosas distintas: la
#: primera es un veredicto, la segunda es la ausencia de uno.
RESULTADOS_DE_SELECCION = ("selected", "no_feasible_model", "insufficient_evidence")


# ---------------------------------------------------------------------------
# Ayudas
# ---------------------------------------------------------------------------

def exigir_opcion(valor: object, campo: str, opciones: tuple[str, ...]) -> str:
    """El valor, si está en el vocabulario; si no, un rechazo con las opciones
    escritas —porque un «valor inválido» sin la lista obliga a leer el código."""
    if not isinstance(valor, str) or valor not in opciones:
        raise EsquemaInvalido("no_es_del_vocabulario", campo=campo,
                              opciones=list(opciones), valor=repr(valor))
    return valor


def es_terminal(estado: str) -> bool:
    """Si de ese estado ya no se sale solo."""
    return exigir_opcion(estado, "estado", ESTADOS) in ESTADOS_TERMINALES


def conserva_candidato(estado: str) -> bool:
    """Si ese estado promete un artefacto utilizable."""
    return exigir_opcion(estado, "estado", ESTADOS) in ESTADOS_CON_CANDIDATO


def exige_motivo(estado: str) -> bool:
    """Si ese estado no se puede publicar sin decir qué pasó."""
    return exigir_opcion(estado, "estado", ESTADOS) in ESTADOS_QUE_EXIGEN_MOTIVO


def fase_del_rol_reservado(rol: str) -> str:
    """La ÚNICA fase en la que se puede leer un rol reservado."""
    return "final_evaluation" if rol == "test" else "external_validation"
