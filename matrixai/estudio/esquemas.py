# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los esquemas que comparte el programa 101-106 — 104-C0.

Aquí no hay motores, ni métricas, ni búsqueda: hay **formas de documento** con
sus reglas, para que 101 (la verdad medida), 102 (los motores), 103 (el
diagnóstico), 105 (la evaluación) y 106 (la entrega) hablen del mismo objeto.
Ese es el orden que fija el contrato: los esquemas van primero y nadie más
puede empezar sin ellos.

CUATRO DECISIONES QUE NO SON DE ESTILO

* **Fail-closed con motivo escrito.** Un documento imposible no se construye:
  una clase positiva que no está entre las clases, un pipeline que declara
  haberse ajustado con el test, un estado `completed` sin candidato. Se rechaza
  en el constructor, con motivo en los dos idiomas, por lo mismo que el 82-C1
  decidió excepción y no `valido: false`: el fallo es de quien cablea, y hay que
  enterarse donde todavía se puede arreglar.
* **Ausente no es cero, y `None` es una respuesta.** Una métrica indefinida vale
  `None` y trae su motivo; un `y_true` que no existe es `None` y el registro
  DICE que no vale para recomputar en vez de inventarse un valor; un predictor
  sin disponibilidad declarada no se convierte en «disponible».
* **El vocabulario es cerrado** (`vocabulario.py`) y **se comparte**. Dos sitios
  declarando los mismos estados acaban divergiendo.
* **Un solo canonicalizador.** Los digests salen de `jcs_bytes` del 81-C1. Un
  segundo canonicalizador daría dos huellas del mismo contenido.

LO QUE ESTE MÓDULO NO HACE, y está dicho aquí para que no se dé por hecho: no
calcula ninguna métrica (105-C1), no orquesta ninguna búsqueda (104-C2), no
decide qué candidato gana (104-C3) y no ejecuta la evaluación anidada (declarada
en el esquema, fuera de este MVP; ver `SplitPlan.nested_evaluation`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.migracion import (
    CLAVE_DE_MARCA,
    ESTUDIO_SCHEMA_VERSION,
    MIGRACIONES,
    RegistroDeMigraciones,
)
from matrixai.estudio.textos import motivo
from matrixai.estudio.validacion import (
    digest_canonico,
    exigir_booleano,
    exigir_entero,
    exigir_entero_o_nulo,
    exigir_mapa,
    exigir_motivo_bilingue,
    exigir_real,
    exigir_real_o_nulo,
    exigir_texto,
    exigir_texto_o_nulo,
    exigir_tupla_de_textos,
    solo_estas_claves,
)
from matrixai.estudio.vocabulario import (
    DIRECCIONES,
    DISPONIBILIDAD,
    ESCALAS_DE_DECISION,
    ESTADOS,
    ESTIMANDOS,
    ETIQUETAS_DE_EVIDENCIA,
    OPERADORES_DE_RESTRICCION,
    REQUISITOS_DE_METRICA,
    RESULTADOS_DE_SELECCION,
    ROLES,
    ROLES_DE_DESARROLLO,
    ROLES_RESERVADOS,
    TAREAS,
    TIPOS_DE_PARTICION,
    UNIDADES_DE_TIEMPO,
    conserva_candidato,
    exige_motivo,
    exigir_opcion,
)

__all__ = [
    "Aptitud", "EvaluationResult", "FitResult", "FittedPipelineSpec", "Horizonte",
    "MetricSpec", "PoliticaDeDecision", "PredictionRecord", "ProblemSpec",
    "Recursos", "Restriccion", "SelectionDecision", "SplitPlan", "ValorDeMetrica",
    "digest_canonico", "version_tras_aprender_del_test",
]

#: Cuánto se le tolera a una distribución de probabilidad al sumar. No es una
#: cifra de estilo: los ficheros vienen de JSON con seis o siete decimales, y
#: exigir igualdad exacta rechazaría distribuciones correctas.
TOLERANCIA_SUMA = 1e-6


# ---------------------------------------------------------------------------
# Sobre y apertura de documentos
# ---------------------------------------------------------------------------

def _sobre(esquema: str, cuerpo: dict[str, Any],
           marca: dict[str, Any] | None) -> dict[str, Any]:
    documento: dict[str, Any] = {"schema": esquema,
                                 "schema_version": ESTUDIO_SCHEMA_VERSION}
    documento.update(cuerpo)
    if marca is not None:
        documento[CLAVE_DE_MARCA] = dict(marca)
    return documento


def _abrir(esquema: str, payload: Any,
           migraciones: RegistroDeMigraciones | None
           ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Comprueba versión y esquema, migra si hace falta y devuelve el cuerpo."""
    registro = migraciones if migraciones is not None else MIGRACIONES
    cuerpo, marca = registro.preparar(esquema, exigir_mapa(payload, esquema))
    declarado = cuerpo.pop("schema", None)
    if declarado != esquema:
        raise EsquemaInvalido("esquema_equivocado", campo=esquema,
                              valor=repr(declarado))
    cuerpo.pop("schema_version", None)
    cuerpo.pop(CLAVE_DE_MARCA, None)
    return cuerpo, marca


def _secuencia(valor: Any, campo: str) -> tuple:
    """Una lista de JSON, tal cual.

    **Un texto NO se convierte en una lista de letras.** `tuple("edad")` da
    cuatro elementos —`('e','d','a','d')`— y los cuatro son cadenas no vacías,
    así que una validación por elemento los daría por buenos: un `predictors:
    "edad"` se leería como cuatro predictores y nadie lo vería.
    """
    if valor is None:
        return ()
    if isinstance(valor, (str, bytes)) or not isinstance(valor, (list, tuple)):
        raise EsquemaInvalido("no_es_lista_de_textos", campo=campo, valor=repr(valor))
    return tuple(valor)


def _saca(cuerpo: Mapping[str, Any], clave: str, esquema: str) -> Any:
    """El valor de una clave OBLIGATORIA. Ausente se dice «falta», no «es None»:
    son cosas distintas y el motivo tiene que distinguirlas."""
    if clave not in cuerpo:
        raise EsquemaInvalido("falta_campo", campo=f"{esquema}.{clave}")
    return cuerpo[clave]


# ---------------------------------------------------------------------------
# Piezas pequeñas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Horizonte:
    """Cuánto tiempo hacia delante. Sirve para el horizonte del desenlace y para
    la separación (gap) de una partición temporal: son la misma forma."""

    magnitud: float
    unidad: str

    def __post_init__(self) -> None:
        exigir_real(self.magnitud, "horizonte.magnitud", minimo=0.0)
        exigir_opcion(self.unidad, "horizonte.unidad", UNIDADES_DE_TIEMPO)

    def a_json(self) -> dict[str, Any]:
        return {"magnitud": self.magnitud, "unidad": self.unidad}

    @classmethod
    def desde_json(cls, payload: Any, campo: str = "horizonte") -> "Horizonte":
        mapa = exigir_mapa(payload, campo)
        solo_estas_claves(mapa, ("magnitud", "unidad"), campo)
        return cls(magnitud=exigir_real(_saca(mapa, "magnitud", campo), f"{campo}.magnitud",
                                        minimo=0.0),
                   unidad=exigir_texto(_saca(mapa, "unidad", campo), f"{campo}.unidad"))


@dataclass(frozen=True)
class Restriccion:
    """Una condición del problema. **Obligatoria por omisión**: el 104-C3 filtra
    los mínimos ANTES de ordenar por utilidad, y una restricción que por defecto
    fuera un deseo se colaría en el orden en vez de filtrar.

    La CLAVE es texto libre a propósito (`max_latencia_ms`, `min_especificidad`,
    `solo_local`): el catálogo de restricciones es del 104-C3 y del problema de
    cada usuario. Lo cerrado es el OPERADOR, que es lo que decide cómo se
    comprueba.
    """

    clave: str
    operador: str
    valor: Any
    unidad: str | None = None
    obligatoria: bool = True

    def __post_init__(self) -> None:
        exigir_texto(self.clave, "restriccion.clave")
        exigir_opcion(self.operador, "restriccion.operador", OPERADORES_DE_RESTRICCION)
        exigir_texto_o_nulo(self.unidad, "restriccion.unidad")
        exigir_booleano(self.obligatoria, "restriccion.obligatoria")
        if self.operador == "boolean":
            exigir_booleano(self.valor, "restriccion.valor")
        elif self.operador in ("min", "max"):
            exigir_real(self.valor, "restriccion.valor")
        elif not isinstance(self.valor, (str, int, float, bool)):
            raise EsquemaInvalido("no_es_texto", campo="restriccion.valor",
                                  valor=repr(self.valor))

    def a_json(self) -> dict[str, Any]:
        return {"clave": self.clave, "operador": self.operador, "valor": self.valor,
                "unidad": self.unidad, "obligatoria": self.obligatoria}

    @classmethod
    def desde_json(cls, payload: Any, campo: str = "restriccion") -> "Restriccion":
        mapa = exigir_mapa(payload, campo)
        solo_estas_claves(mapa, ("clave", "operador", "valor", "unidad", "obligatoria"), campo)
        return cls(clave=_saca(mapa, "clave", campo), operador=_saca(mapa, "operador", campo),
                   valor=_saca(mapa, "valor", campo), unidad=mapa.get("unidad"),
                   obligatoria=mapa.get("obligatoria", True))


@dataclass(frozen=True)
class Recursos:
    """Lo que costó. Todos los campos son opcionales y `None` significa **no
    medido**, no cero: publicar un `0` de RAM porque nadie la midió convertiría
    una ausencia en una afirmación falsa, y el 104-C1 compara candidatos con
    estos números."""

    cpu_seconds: float | None = None
    wall_seconds: float | None = None
    peak_ram_mb: float | None = None
    latency_ms_per_row: float | None = None

    def __post_init__(self) -> None:
        for nombre in ("cpu_seconds", "wall_seconds", "peak_ram_mb", "latency_ms_per_row"):
            exigir_real_o_nulo(getattr(self, nombre), f"recursos.{nombre}", minimo=0.0)

    def a_json(self) -> dict[str, Any]:
        return {"cpu_seconds": self.cpu_seconds, "wall_seconds": self.wall_seconds,
                "peak_ram_mb": self.peak_ram_mb,
                "latency_ms_per_row": self.latency_ms_per_row}

    @classmethod
    def desde_json(cls, payload: Any, campo: str = "recursos") -> "Recursos":
        mapa = exigir_mapa(payload, campo)
        solo_estas_claves(mapa, ("cpu_seconds", "wall_seconds", "peak_ram_mb",
                                 "latency_ms_per_row"), campo)
        return cls(**{k: mapa.get(k) for k in ("cpu_seconds", "wall_seconds",
                                               "peak_ram_mb", "latency_ms_per_row")})


@dataclass(frozen=True)
class PoliticaDeDecision:
    """El umbral y de qué escala es.

    **Si el pipeline lleva calibrador, el umbral vive en la probabilidad
    calibrada** (104-C4). Un umbral elegido sobre el score crudo y aplicado
    detrás de un calibrador no es el umbral de ese pipeline: corta en otro sitio.
    Esa comprobación está en `FittedPipelineSpec`, que es quien sabe si hay
    calibrador.
    """

    threshold: float
    scale: str
    positive_label: str
    cost_false_positive: float | None = None
    cost_false_negative: float | None = None

    def __post_init__(self) -> None:
        exigir_opcion(self.scale, "decision_policy.scale", ESCALAS_DE_DECISION)
        if self.scale == "calibrated_probability":
            exigir_real(self.threshold, "decision_policy.threshold", minimo=0.0, maximo=1.0)
        else:
            exigir_real(self.threshold, "decision_policy.threshold")
        if not isinstance(self.positive_label, str) or not self.positive_label.strip():
            raise EsquemaInvalido("umbral_sin_clase_positiva")
        exigir_real_o_nulo(self.cost_false_positive, "decision_policy.cost_false_positive",
                           minimo=0.0)
        exigir_real_o_nulo(self.cost_false_negative, "decision_policy.cost_false_negative",
                           minimo=0.0)

    def a_json(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "scale": self.scale,
                "positive_label": self.positive_label,
                "cost_false_positive": self.cost_false_positive,
                "cost_false_negative": self.cost_false_negative}

    @classmethod
    def desde_json(cls, payload: Any, campo: str = "decision_policy") -> "PoliticaDeDecision":
        mapa = exigir_mapa(payload, campo)
        solo_estas_claves(mapa, ("threshold", "scale", "positive_label",
                                 "cost_false_positive", "cost_false_negative"), campo)
        return cls(threshold=_saca(mapa, "threshold", campo),
                   scale=_saca(mapa, "scale", campo),
                   positive_label=_saca(mapa, "positive_label", campo),
                   cost_false_positive=mapa.get("cost_false_positive"),
                   cost_false_negative=mapa.get("cost_false_negative"))


@dataclass(frozen=True)
class Aptitud:
    """Si un `PredictionRecord` sirve para recomputar una métrica, y si no, QUÉ
    falta. No rellena nada: enumera."""

    apto: bool
    faltan: tuple[str, ...]
    motivo: dict[str, str] | None

    def a_json(self) -> dict[str, Any]:
        return {"apto": self.apto, "faltan": list(self.faltan), "motivo": self.motivo}


# ---------------------------------------------------------------------------
# ProblemSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProblemSpec:
    """Qué se predice, para quién y con qué restricciones.

    El 103-C1 lo confirma con el usuario y el 106-C4 lo publica en la tarjeta.
    Dos reglas que valen un fallo cada una:

    * una binaria **sin clase positiva** deja sin definir sensibilidad, PPV y el
      umbral, y elegirla por orden alfabético es inventarse la mitad del
      problema;
    * un predictor **sin disponibilidad declarada** no es un predictor
      disponible. `unknown` (alguien lo miró y no lo sabe) y la ausencia (nadie
      lo miró) son datos distintos y el esquema los conserva: `sin_disponibilidad_declarada()`
      los enumera en vez de rellenarlos.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.problem_spec"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "problem_id", "target", "task", "classes", "positive_label",
        "observation_unit", "predictors", "predictor_availability",
        "prediction_time", "horizon", "intended_use", "constraints")

    problem_id: str
    target: str
    task: str
    observation_unit: str
    classes: tuple[str, ...] | None = None
    positive_label: str | None = None
    predictors: tuple[str, ...] = ()
    predictor_availability: dict[str, str] = field(default_factory=dict)
    prediction_time: str | None = None
    horizon: Horizonte | None = None
    intended_use: str | None = None
    constraints: tuple[Restriccion, ...] = ()
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.problem_id, "problem_id")
        exigir_texto(self.target, "target")
        exigir_opcion(self.task, "task", TAREAS)
        exigir_texto(self.observation_unit, "observation_unit")
        exigir_texto_o_nulo(self.intended_use, "intended_use")
        exigir_texto_o_nulo(self.prediction_time, "prediction_time")
        object.__setattr__(self, "predictors",
                           exigir_tupla_de_textos(self.predictors, "predictors"))

        if self.task == "regression":
            if self.classes is not None:
                raise EsquemaInvalido("clases_en_regresion", valor=repr(self.classes))
            if self.positive_label is not None:
                raise EsquemaInvalido("clase_positiva_en_regresion",
                                      valor=repr(self.positive_label))
        else:
            if self.classes is None:
                raise EsquemaInvalido("clasificacion_sin_clases")
            clases = exigir_tupla_de_textos(self.classes, "classes", minimo=2)
            object.__setattr__(self, "classes", clases)
            if self.task == "binary_classification" and len(clases) != 2:
                raise EsquemaInvalido("binaria_con_otro_numero_de_clases",
                                      valor=len(clases))
            if self.task == "multiclass_classification" and len(clases) < 3:
                raise EsquemaInvalido("multiclase_con_menos_de_tres", valor=len(clases))
            if self.positive_label is None:
                if self.task == "binary_classification":
                    raise EsquemaInvalido("binaria_sin_clase_positiva")
            else:
                exigir_texto(self.positive_label, "positive_label")
                if self.positive_label not in clases:
                    raise EsquemaInvalido("clase_positiva_fuera_de_clases",
                                          valor=repr(self.positive_label),
                                          opciones=list(clases))

        disponibilidad = exigir_mapa(self.predictor_availability, "predictor_availability")
        for nombre, cuando in disponibilidad.items():
            if nombre not in self.predictors:
                raise EsquemaInvalido("disponibilidad_de_desconocido", valor=repr(nombre))
            exigir_opcion(cuando, f"predictor_availability.{nombre}", DISPONIBILIDAD)
        object.__setattr__(self, "predictor_availability", dict(disponibilidad))

        if self.horizon is not None and not isinstance(self.horizon, Horizonte):
            raise EsquemaInvalido("no_es_mapa", campo="horizon", valor=repr(self.horizon))
        if self.horizon is not None and self.prediction_time is None:
            raise EsquemaInvalido("horizonte_sin_momento",
                                  valor=repr(self.horizon.a_json()))
        for restriccion in self.constraints:
            if not isinstance(restriccion, Restriccion):
                raise EsquemaInvalido("no_es_mapa", campo="constraints",
                                      valor=repr(restriccion))

    # -- consultas -------------------------------------------------------
    def sin_disponibilidad_declarada(self) -> tuple[str, ...]:
        """Los predictores de los que NADIE ha dicho cuándo están disponibles.

        No se rellenan con `unknown`: 103 invariante 5 prohíbe inferir
        disponibilidad por el nombre o el orden del CSV, y rellenarla aquí sería
        inferirla por omisión.
        """
        return tuple(sorted(p for p in self.predictors
                            if p not in self.predictor_availability))

    def restricciones_obligatorias(self) -> tuple[Restriccion, ...]:
        return tuple(r for r in self.constraints if r.obligatoria)

    # -- serialización ---------------------------------------------------
    def _cuerpo(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id, "target": self.target, "task": self.task,
            "observation_unit": self.observation_unit,
            "classes": list(self.classes) if self.classes is not None else None,
            "positive_label": self.positive_label,
            "predictors": list(self.predictors),
            "predictor_availability": dict(self.predictor_availability),
            "prediction_time": self.prediction_time,
            "horizon": self.horizon.a_json() if self.horizon else None,
            "intended_use": self.intended_use,
            "constraints": [r.a_json() for r in self.constraints],
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "ProblemSpec":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        clases = cuerpo.get("classes")
        horizonte = cuerpo.get("horizon")
        return cls(
            problem_id=_saca(cuerpo, "problem_id", cls.ESQUEMA),
            target=_saca(cuerpo, "target", cls.ESQUEMA),
            task=_saca(cuerpo, "task", cls.ESQUEMA),
            observation_unit=_saca(cuerpo, "observation_unit", cls.ESQUEMA),
            classes=tuple(clases) if isinstance(clases, (list, tuple)) else clases,
            positive_label=cuerpo.get("positive_label"),
            predictors=_secuencia(cuerpo.get("predictors"), "predictors"),
            predictor_availability=dict(cuerpo.get("predictor_availability") or {}),
            prediction_time=cuerpo.get("prediction_time"),
            horizon=Horizonte.desde_json(horizonte) if horizonte is not None else None,
            intended_use=cuerpo.get("intended_use"),
            constraints=tuple(Restriccion.desde_json(r) for r in
                              _secuencia(cuerpo.get("constraints"), "constraints")),
            migracion=marca,
        )


# ---------------------------------------------------------------------------
# SplitPlan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitPlan:
    """Qué observación juega en qué papel, y con qué diseño.

    LO QUE DE VERDAD PROTEGE ESTA CLASE es la comprobación de unidades: si dos
    filas del mismo paciente caen una en desarrollo y otra en prueba, la prueba
    ya no mide generalización a pacientes nuevos. **Se comprueba la intersección
    de conjuntos, no que los hashes sean distintos** (103-C4).

    `nested_evaluation` existe en el esquema porque el 104-C0 lo contempla para
    datasets pequeños, y **este MVP no lo implementa**: se declara fuera. Un plan
    que lo pida se puede escribir y guardar —para no perder la intención— pero el
    guardia del protocolo se niega a ejecutarlo (`accesos.py`), en vez de correr
    media evaluación anidada y llamarla completa.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.split_plan"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "plan_id", "split_type", "assignments", "units", "observation_id_field",
        "unit_id_field", "time_column", "gap", "folds", "repeats", "seed",
        "nested_evaluation", "digests")

    plan_id: str
    split_type: str
    assignments: dict[str, str]
    observation_id_field: str
    units: dict[str, str] | None = None
    unit_id_field: str | None = None
    time_column: str | None = None
    gap: Horizonte | None = None
    folds: int | None = None
    repeats: int | None = None
    seed: int | None = None
    nested_evaluation: bool = False
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.plan_id, "plan_id")
        exigir_opcion(self.split_type, "split_type", TIPOS_DE_PARTICION)
        exigir_texto(self.observation_id_field, "observation_id_field")
        exigir_texto_o_nulo(self.unit_id_field, "unit_id_field")
        exigir_texto_o_nulo(self.time_column, "time_column")
        exigir_booleano(self.nested_evaluation, "nested_evaluation")
        exigir_entero_o_nulo(self.seed, "seed")
        if self.folds is not None:
            exigir_entero(self.folds, "folds")
            if self.folds < 2:
                raise EsquemaInvalido("pliegues_insuficientes", campo="folds",
                                      valor=self.folds)
        exigir_entero_o_nulo(self.repeats, "repeats", minimo=1)

        asignaciones = exigir_mapa(self.assignments, "assignments")
        if not asignaciones:
            raise EsquemaInvalido("plan_vacio")
        for observacion, rol in asignaciones.items():
            exigir_opcion(rol, f"assignments.{observacion}", ROLES)
        object.__setattr__(self, "assignments", dict(asignaciones))

        if self.units is not None:
            unidades = exigir_mapa(self.units, "units")
            for observacion, unidad in unidades.items():
                if observacion not in asignaciones:
                    raise EsquemaInvalido("unidad_de_observacion_desconocida",
                                          valor=repr(observacion))
                exigir_texto(unidad, f"units.{observacion}")
            object.__setattr__(self, "units", dict(unidades))
            self._comprobar_aislamiento_de_unidades()

        if self.split_type in ("groups", "groups_and_time"):
            if self.units is None:
                raise EsquemaInvalido("grupos_sin_unidad", valor=self.split_type)
            faltan = [o for o in asignaciones if o not in self.units]
            if faltan:
                raise EsquemaInvalido("observacion_sin_unidad", valor=repr(sorted(faltan)[0]))
            if self.unit_id_field is None:
                raise EsquemaInvalido("falta_campo", campo="unit_id_field")
        if self.split_type in ("temporal", "groups_and_time") and self.time_column is None:
            raise EsquemaInvalido("temporal_sin_tiempo", valor=self.split_type)
        if self.gap is not None and not isinstance(self.gap, Horizonte):
            raise EsquemaInvalido("no_es_mapa", campo="gap", valor=repr(self.gap))

        if "test" not in set(asignaciones.values()) and not self.nested_evaluation:
            raise EsquemaInvalido("particion_sin_test")

    def _comprobar_aislamiento_de_unidades(self) -> None:
        roles_por_unidad: dict[str, set[str]] = {}
        for observacion, unidad in (self.units or {}).items():
            roles_por_unidad.setdefault(unidad, set()).add(self.assignments[observacion])
        for unidad, roles in sorted(roles_por_unidad.items()):
            if len(roles) > 1:
                raise EsquemaInvalido("unidad_en_dos_roles", valor=repr(unidad),
                                      opciones=sorted(roles))

    # -- consultas -------------------------------------------------------
    def observaciones_del_rol(self, rol: str) -> tuple[str, ...]:
        """Los ids de ese rol, ordenados. Ordenados a propósito: un orden que
        dependa del diccionario haría que el mismo plan diera dos digests."""
        exigir_opcion(rol, "rol", ROLES)
        return tuple(sorted(o for o, r in self.assignments.items() if r == rol))

    def unidades_del_rol(self, rol: str) -> tuple[str, ...]:
        if self.units is None:
            return ()
        return tuple(sorted({self.units[o] for o in self.observaciones_del_rol(rol)
                             if o in self.units}))

    def roles_presentes(self) -> tuple[str, ...]:
        presentes = set(self.assignments.values())
        return tuple(r for r in ROLES if r in presentes)

    def digest_de_asignaciones(self) -> str:
        """La huella de QUIÉN juega en QUÉ papel, aparte de la del plan entero:
        dos planes con distinta semilla y las mismas asignaciones reparten igual,
        y eso hay que poder verlo."""
        return digest_canonico({"assignments": dict(sorted(self.assignments.items())),
                                "units": dict(sorted((self.units or {}).items()))})

    # -- serialización ---------------------------------------------------
    def _cuerpo(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id, "split_type": self.split_type,
            "assignments": dict(sorted(self.assignments.items())),
            "units": dict(sorted(self.units.items())) if self.units is not None else None,
            "observation_id_field": self.observation_id_field,
            "unit_id_field": self.unit_id_field, "time_column": self.time_column,
            "gap": self.gap.a_json() if self.gap else None,
            "folds": self.folds, "repeats": self.repeats, "seed": self.seed,
            "nested_evaluation": self.nested_evaluation,
            "digests": {"assignments": self.digest_de_asignaciones()},
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "SplitPlan":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        gap = cuerpo.get("gap")
        plan = cls(
            plan_id=_saca(cuerpo, "plan_id", cls.ESQUEMA),
            split_type=_saca(cuerpo, "split_type", cls.ESQUEMA),
            assignments=dict(exigir_mapa(_saca(cuerpo, "assignments", cls.ESQUEMA),
                                         "assignments")),
            observation_id_field=_saca(cuerpo, "observation_id_field", cls.ESQUEMA),
            units=dict(cuerpo["units"]) if cuerpo.get("units") is not None else None,
            unit_id_field=cuerpo.get("unit_id_field"),
            time_column=cuerpo.get("time_column"),
            gap=Horizonte.desde_json(gap, "gap") if gap is not None else None,
            folds=cuerpo.get("folds"), repeats=cuerpo.get("repeats"),
            seed=cuerpo.get("seed"),
            nested_evaluation=bool(cuerpo.get("nested_evaluation", False)),
            migracion=marca,
        )
        declarados = (cuerpo.get("digests") or {}).get("assignments")
        if declarados is not None and declarados != plan.digest_de_asignaciones():
            raise EsquemaInvalido("hay_duplicados", campo="digests.assignments",
                                  valor=repr(declarados))
        return plan


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionRecord:
    """Una predicción, con todo lo que hace falta para volver a medirla.

    **Todo lo que puede faltar de verdad, falta como `None`**: hay predicciones
    sin `y_true` (datos nuevos), sin rol (no pertenecen a ninguna partición) y
    sin unidad de remuestreo (nadie la declaró). Lo que NO se admite es un
    registro que no sea una predicción —sin etiqueta, sin puntuaciones y sin
    probabilidades— ni columnas que no se sepa de qué clase son.

    Las probabilidades van como **distribución completa alineada con `classes`**:
    media distribución («la del positivo») no se recompone sin saber qué clase
    falta, y ese es exactamente el error de correspondencia de clases que el
    102-C0 documenta. Para la puntuación de decisión binaria sí se admite un solo
    valor, que es el del positivo, y entonces `classes` tiene que traer las dos.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.prediction_record"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "row_id", "candidate", "resampling_unit", "y_true", "label", "scores",
        "probabilities", "classes", "fold_model", "repeat", "fold", "role", "weight")

    row_id: str
    candidate: str
    resampling_unit: str | None = None
    y_true: Any | None = None
    label: Any | None = None
    scores: tuple[float, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    classes: tuple[str, ...] | None = None
    fold_model: str | None = None
    repeat: int | None = None
    fold: int | None = None
    role: str | None = None
    weight: float | None = None
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.row_id, "row_id")
        exigir_texto(self.candidate, "candidate")
        exigir_texto_o_nulo(self.resampling_unit, "resampling_unit")
        exigir_texto_o_nulo(self.fold_model, "fold_model")
        exigir_entero_o_nulo(self.repeat, "repeat", minimo=1)
        exigir_entero_o_nulo(self.fold, "fold", minimo=0)
        exigir_real_o_nulo(self.weight, "weight", minimo=0.0)
        if self.role is not None:
            exigir_opcion(self.role, "role", ROLES)

        if self.classes is not None:
            object.__setattr__(self, "classes",
                               exigir_tupla_de_textos(self.classes, "classes", minimo=2))

        if self.label is None and self.scores is None and self.probabilities is None:
            raise EsquemaInvalido("prediccion_sin_salida")

        if self.probabilities is not None:
            probabilidades = self._reales(self.probabilities, "probabilities",
                                          minimo=0.0, maximo=1.0)
            object.__setattr__(self, "probabilities", probabilidades)
            if self.classes is None:
                raise EsquemaInvalido("columnas_sin_clases", campo="probabilities")
            if len(probabilidades) != len(self.classes):
                raise EsquemaInvalido("columnas_desalineadas", campo="probabilities",
                                      valor=len(probabilidades), opciones=len(self.classes))
            suma = sum(probabilidades)
            if abs(suma - 1.0) > TOLERANCIA_SUMA:
                raise EsquemaInvalido("probabilidades_no_suman_uno", valor=repr(suma))

        if self.scores is not None:
            puntuaciones = self._reales(self.scores, "scores")
            object.__setattr__(self, "scores", puntuaciones)
            if self.classes is None:
                raise EsquemaInvalido("columnas_sin_clases", campo="scores")
            if len(puntuaciones) not in (1, len(self.classes)):
                raise EsquemaInvalido("columnas_desalineadas", campo="scores",
                                      valor=len(puntuaciones), opciones=len(self.classes))
            if len(puntuaciones) == 1 and len(self.classes) != 2:
                raise EsquemaInvalido("columnas_desalineadas", campo="scores",
                                      valor=1, opciones=len(self.classes))

        if self.classes is not None and self.y_true is not None:
            if self.y_true not in self.classes:
                raise EsquemaInvalido("clase_positiva_fuera_de_clases",
                                      valor=repr(self.y_true), opciones=list(self.classes))

    @staticmethod
    def _reales(valores: Any, campo: str, *, minimo: float | None = None,
                maximo: float | None = None) -> tuple[float, ...]:
        if isinstance(valores, (str, bytes)) or not isinstance(valores, (list, tuple)):
            raise EsquemaInvalido("no_es_lista_de_textos", campo=campo, valor=repr(valores))
        if not valores:
            raise EsquemaInvalido("menor_que_el_minimo", campo=campo, minimo=1,
                                  valor="0 elementos")
        return tuple(exigir_real(v, f"{campo}[{i}]", minimo=minimo, maximo=maximo)
                     for i, v in enumerate(valores))

    # -- la pregunta del contrato ---------------------------------------
    def apto_para_recomputar(self, metrica: "MetricSpec") -> Aptitud:
        """¿Sirve este registro para volver a calcular esa métrica?

        Es el criterio 2 del corte: **un registro sin `y_true`, sin clases o sin
        partición, cuando la métrica los necesita, NO vale — y el esquema lo
        DICE, no lo rellena**. Devuelve qué falta, con motivo escrito; nunca
        inventa un cero, una clase ni un rol.
        """
        faltan: list[str] = []
        for requisito in metrica.requires:
            if requisito == "y_true" and self.y_true is None:
                faltan.append("y_true")
            elif requisito == "labels" and self.label is None:
                faltan.append("labels")
            elif requisito == "scores" and self.scores is None:
                faltan.append("scores")
            elif requisito == "probabilities" and self.probabilities is None:
                faltan.append("probabilities")
            elif requisito == "classes" and self.classes is None:
                faltan.append("classes")
            elif requisito == "positive_label":
                if metrica.positive_label is None:
                    faltan.append("positive_label")
                elif self.classes is not None and metrica.positive_label not in self.classes:
                    faltan.append("positive_label")
            elif requisito == "split_role" and self.role is None:
                faltan.append("split_role")
            elif requisito == "resampling_unit" and self.resampling_unit is None:
                faltan.append("resampling_unit")
            elif requisito == "weights" and self.weight is None:
                faltan.append("weights")
        if not faltan:
            return Aptitud(apto=True, faltan=(), motivo=None)
        return Aptitud(apto=False, faltan=tuple(faltan),
                       motivo=motivo("no_vale_para_recomputar",
                                     campo=metrica.metric_id, valor=", ".join(faltan)))

    # -- serialización ---------------------------------------------------
    def _cuerpo(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id, "candidate": self.candidate,
            "resampling_unit": self.resampling_unit, "y_true": self.y_true,
            "label": self.label,
            "scores": list(self.scores) if self.scores is not None else None,
            "probabilities": (list(self.probabilities)
                              if self.probabilities is not None else None),
            "classes": list(self.classes) if self.classes is not None else None,
            "fold_model": self.fold_model, "repeat": self.repeat, "fold": self.fold,
            "role": self.role, "weight": self.weight,
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "PredictionRecord":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        def _tupla(clave: str) -> tuple | None:
            valor = cuerpo.get(clave)
            return tuple(valor) if isinstance(valor, (list, tuple)) else valor
        return cls(
            row_id=_saca(cuerpo, "row_id", cls.ESQUEMA),
            candidate=_saca(cuerpo, "candidate", cls.ESQUEMA),
            resampling_unit=cuerpo.get("resampling_unit"), y_true=cuerpo.get("y_true"),
            label=cuerpo.get("label"), scores=_tupla("scores"),
            probabilities=_tupla("probabilities"), classes=_tupla("classes"),
            fold_model=cuerpo.get("fold_model"), repeat=cuerpo.get("repeat"),
            fold=cuerpo.get("fold"), role=cuerpo.get("role"), weight=cuerpo.get("weight"),
            migracion=marca,
        )


# ---------------------------------------------------------------------------
# MetricSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """La identidad de una métrica: qué mide, con qué fórmula y qué necesita.

    **Dirección O valor/rango ideal, nunca las dos** (105 invariante 4): las
    métricas de calidad tienen dirección; la pendiente de calibración vale 1 y
    desviarse en cualquier sentido es peor. Ordenar todo por «mayor mejor»
    convierte un modelo descalibrado por exceso en el ganador.

    El CATÁLOGO de métricas —qué exige cada una, cómo se calcula— es del 105-C1:
    aquí solo está la forma con la que se declara, para que las dos mitades
    encajen sin duplicar fórmulas.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.metric_spec"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "metric_id", "formula_version", "direction", "ideal_value", "ideal_range",
        "normalization", "positive_label", "weights", "estimand", "requires")

    metric_id: str
    formula_version: str
    estimand: str
    requires: tuple[str, ...] = ()
    direction: str | None = None
    ideal_value: float | None = None
    ideal_range: tuple[float, float] | None = None
    normalization: str | None = None
    positive_label: str | None = None
    weights: str | None = None
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.metric_id, "metric_id")
        exigir_texto(self.formula_version, "formula_version")
        exigir_opcion(self.estimand, "estimand", ESTIMANDOS)
        exigir_texto_o_nulo(self.normalization, "normalization")
        exigir_texto_o_nulo(self.positive_label, "positive_label")
        exigir_texto_o_nulo(self.weights, "weights")
        object.__setattr__(self, "requires",
                           exigir_tupla_de_textos(self.requires, "requires",
                                                  opciones=REQUISITOS_DE_METRICA))
        if self.direction is not None:
            exigir_opcion(self.direction, "direction", DIRECCIONES)

        tiene_ideal = self.ideal_value is not None or self.ideal_range is not None
        if self.direction is None and not tiene_ideal:
            raise EsquemaInvalido("metrica_sin_direccion_ni_ideal", campo=self.metric_id)
        if self.direction is not None and tiene_ideal:
            raise EsquemaInvalido("metrica_con_direccion_e_ideal", campo=self.metric_id)

        exigir_real_o_nulo(self.ideal_value, "ideal_value")
        if self.ideal_range is not None:
            rango = self.ideal_range
            if (isinstance(rango, (str, bytes)) or not isinstance(rango, (list, tuple))
                    or len(rango) != 2):
                raise EsquemaInvalido("no_es_lista_de_textos", campo="ideal_range",
                                      valor=repr(rango))
            bajo = exigir_real(rango[0], "ideal_range[0]")
            alto = exigir_real(rango[1], "ideal_range[1]")
            if bajo > alto:
                raise EsquemaInvalido("rango_ideal_al_reves", valor=repr(list(rango)))
            object.__setattr__(self, "ideal_range", (bajo, alto))

        if "positive_label" in self.requires and self.positive_label is None:
            raise EsquemaInvalido("metrica_de_clase_positiva_sin_clase",
                                  campo=self.metric_id)

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id, "formula_version": self.formula_version,
            "estimand": self.estimand, "requires": list(self.requires),
            "direction": self.direction, "ideal_value": self.ideal_value,
            "ideal_range": list(self.ideal_range) if self.ideal_range else None,
            "normalization": self.normalization, "positive_label": self.positive_label,
            "weights": self.weights,
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "MetricSpec":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        rango = cuerpo.get("ideal_range")
        return cls(
            metric_id=_saca(cuerpo, "metric_id", cls.ESQUEMA),
            formula_version=_saca(cuerpo, "formula_version", cls.ESQUEMA),
            estimand=_saca(cuerpo, "estimand", cls.ESQUEMA),
            requires=_secuencia(cuerpo.get("requires"), "requires"),
            direction=cuerpo.get("direction"), ideal_value=cuerpo.get("ideal_value"),
            ideal_range=tuple(rango) if isinstance(rango, (list, tuple)) else rango,
            normalization=cuerpo.get("normalization"),
            positive_label=cuerpo.get("positive_label"), weights=cuerpo.get("weights"),
            migracion=marca,
        )


# ---------------------------------------------------------------------------
# FittedPipelineSpec y FitResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FittedPipelineSpec:
    """El pipeline entero: preparación + predictor + calibrador + decisión.

    Es inseparable (106 invariante 1): lo que se evalúa es lo que se exporta. Y
    **declara con qué roles se ajustó**, que es donde se caza la contaminación
    en el propio documento: un pipeline que dice haberse entrenado con `test` no
    se puede construir.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.fitted_pipeline_spec"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "pipeline_id", "problem_id", "candidate", "engine", "preparation", "predictor",
        "calibrator", "decision_policy", "trained_on_roles", "split_plan_digest",
        "frozen")

    pipeline_id: str
    problem_id: str
    candidate: str
    engine: str
    predictor: dict[str, Any]
    trained_on_roles: tuple[str, ...]
    split_plan_digest: str
    preparation: dict[str, Any] = field(default_factory=dict)
    calibrator: dict[str, Any] | None = None
    decision_policy: PoliticaDeDecision | None = None
    frozen: bool = False
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for nombre in ("pipeline_id", "problem_id", "candidate", "engine"):
            exigir_texto(getattr(self, nombre), nombre)
        exigir_texto(self.split_plan_digest, "split_plan_digest")
        exigir_booleano(self.frozen, "frozen")
        predictor = exigir_mapa(self.predictor, "predictor")
        if not predictor:
            raise EsquemaInvalido("falta_campo", campo="predictor")
        object.__setattr__(self, "predictor", dict(predictor))
        object.__setattr__(self, "preparation",
                           dict(exigir_mapa(self.preparation, "preparation")))
        if self.calibrator is not None:
            object.__setattr__(self, "calibrator",
                               dict(exigir_mapa(self.calibrator, "calibrator")))

        roles = exigir_tupla_de_textos(self.trained_on_roles, "trained_on_roles",
                                       minimo=1, opciones=ROLES)
        for rol in roles:
            if rol in ROLES_RESERVADOS:
                raise EsquemaInvalido("entrenar_con_test", valor=rol)
        object.__setattr__(self, "trained_on_roles", roles)

        if self.decision_policy is not None:
            if not isinstance(self.decision_policy, PoliticaDeDecision):
                raise EsquemaInvalido("no_es_mapa", campo="decision_policy",
                                      valor=repr(self.decision_policy))
            if (self.calibrator is not None
                    and self.decision_policy.scale != "calibrated_probability"):
                raise EsquemaInvalido("umbral_fuera_de_la_escala_calibrada",
                                      valor=self.decision_policy.scale)

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id, "problem_id": self.problem_id,
            "candidate": self.candidate, "engine": self.engine,
            "preparation": dict(self.preparation), "predictor": dict(self.predictor),
            "calibrator": dict(self.calibrator) if self.calibrator is not None else None,
            "decision_policy": (self.decision_policy.a_json()
                                if self.decision_policy else None),
            "trained_on_roles": list(self.trained_on_roles),
            "split_plan_digest": self.split_plan_digest, "frozen": self.frozen,
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        """La huella del artefacto. **Incluye `frozen`**: congelar es un cambio de
        estado del pipeline, y el 104-C5 exige que el digest de lo evaluado sea
        el de lo exportado — si congelar no cambiara la huella, no se podría
        distinguir lo que se evaluó de lo que todavía se estaba tocando."""
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    def congelado(self) -> "FittedPipelineSpec":
        """El mismo pipeline, congelado. Documento NUEVO: no se muta el anterior."""
        return FittedPipelineSpec(
            pipeline_id=self.pipeline_id, problem_id=self.problem_id,
            candidate=self.candidate, engine=self.engine, predictor=dict(self.predictor),
            trained_on_roles=self.trained_on_roles,
            split_plan_digest=self.split_plan_digest, preparation=dict(self.preparation),
            calibrator=dict(self.calibrator) if self.calibrator is not None else None,
            decision_policy=self.decision_policy, frozen=True, migracion=self.migracion)

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "FittedPipelineSpec":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        politica = cuerpo.get("decision_policy")
        return cls(
            pipeline_id=_saca(cuerpo, "pipeline_id", cls.ESQUEMA),
            problem_id=_saca(cuerpo, "problem_id", cls.ESQUEMA),
            candidate=_saca(cuerpo, "candidate", cls.ESQUEMA),
            engine=_saca(cuerpo, "engine", cls.ESQUEMA),
            predictor=dict(exigir_mapa(_saca(cuerpo, "predictor", cls.ESQUEMA), "predictor")),
            trained_on_roles=_secuencia(_saca(cuerpo, "trained_on_roles", cls.ESQUEMA),
                                        "trained_on_roles"),
            split_plan_digest=_saca(cuerpo, "split_plan_digest", cls.ESQUEMA),
            preparation=dict(cuerpo.get("preparation") or {}),
            calibrator=(dict(cuerpo["calibrator"])
                        if cuerpo.get("calibrator") is not None else None),
            decision_policy=(PoliticaDeDecision.desde_json(politica)
                             if politica is not None else None),
            frozen=bool(cuerpo.get("frozen", False)), migracion=marca)


@dataclass(frozen=True)
class FitResult:
    """Cómo acabó el ajuste de un candidato.

    El invariante 3 del 104 escrito como regla: **un proceso abortado sin
    candidato no se disfraza de completado**. `completed` y
    `completed_budget_limited` prometen un pipeline y tienen que traerlo;
    `failed`, `cancelled` y `unsupported` no pueden traerlo y tienen que decir
    qué pasó, en los dos idiomas.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.fit_result"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "candidate", "state", "pipeline", "reason", "resources", "checkpoint")

    candidate: str
    state: str
    pipeline: FittedPipelineSpec | None = None
    reason: dict[str, str] | None = None
    resources: Recursos | None = None
    checkpoint: str | None = None
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.candidate, "candidate")
        exigir_opcion(self.state, "state", ESTADOS)
        exigir_texto_o_nulo(self.checkpoint, "checkpoint")
        if self.reason is not None:
            object.__setattr__(self, "reason",
                               exigir_motivo_bilingue(self.reason, "reason"))
        if self.resources is not None and not isinstance(self.resources, Recursos):
            raise EsquemaInvalido("no_es_mapa", campo="resources", valor=repr(self.resources))

        if conserva_candidato(self.state):
            if self.pipeline is None:
                raise EsquemaInvalido("completado_sin_pipeline", valor=self.state)
        elif self.pipeline is not None:
            raise EsquemaInvalido("fallido_con_pipeline", valor=self.state)
        if self.pipeline is not None and not isinstance(self.pipeline, FittedPipelineSpec):
            raise EsquemaInvalido("no_es_mapa", campo="pipeline", valor=repr(self.pipeline))
        if exige_motivo(self.state) and self.reason is None:
            raise EsquemaInvalido("sin_motivo_escrito", valor=self.state)

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate, "state": self.state,
            "pipeline": self.pipeline.a_json() if self.pipeline else None,
            "reason": dict(self.reason) if self.reason else None,
            "resources": self.resources.a_json() if self.resources else None,
            "checkpoint": self.checkpoint,
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "FitResult":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        pipeline = cuerpo.get("pipeline")
        recursos = cuerpo.get("resources")
        return cls(
            candidate=_saca(cuerpo, "candidate", cls.ESQUEMA),
            state=_saca(cuerpo, "state", cls.ESQUEMA),
            pipeline=(FittedPipelineSpec.desde_json(pipeline, migraciones=migraciones)
                      if pipeline is not None else None),
            reason=cuerpo.get("reason"),
            resources=Recursos.desde_json(recursos) if recursos is not None else None,
            checkpoint=cuerpo.get("checkpoint"), migracion=marca)


# ---------------------------------------------------------------------------
# EvaluationResult y SelectionDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValorDeMetrica:
    """Un número medido, o la razón de que no lo haya.

    **Valor o motivo, exactamente uno.** Una métrica indefinida —una sola clase
    en la partición, una división por cero, cero eventos— no vale cero: vale
    `None` y explica por qué. Y un `None` mudo es igual de malo: quien lo lea no
    sabe si es que no se pudo o que a nadie se le ocurrió medirlo.
    """

    metric_id: str
    formula_version: str
    value: float | None = None
    undefined_reason: dict[str, str] | None = None
    n_observations: int | None = None
    n_units: int | None = None
    uncertainty: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.metric_id, "metric_id")
        exigir_texto(self.formula_version, "formula_version")
        exigir_real_o_nulo(self.value, f"{self.metric_id}.value")
        exigir_entero_o_nulo(self.n_observations, "n_observations", minimo=0)
        exigir_entero_o_nulo(self.n_units, "n_units", minimo=0)
        if self.undefined_reason is not None:
            object.__setattr__(self, "undefined_reason",
                               exigir_motivo_bilingue(self.undefined_reason,
                                                      "undefined_reason"))
        if self.value is None and self.undefined_reason is None:
            raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo=self.metric_id)
        if self.value is not None and self.undefined_reason is not None:
            raise EsquemaInvalido("metrica_con_valor_y_motivo", campo=self.metric_id)
        if self.uncertainty is not None:
            object.__setattr__(self, "uncertainty",
                               dict(exigir_mapa(self.uncertainty, "uncertainty")))

    def a_json(self) -> dict[str, Any]:
        return {"metric_id": self.metric_id, "formula_version": self.formula_version,
                "value": self.value,
                "undefined_reason": dict(self.undefined_reason) if self.undefined_reason else None,
                "n_observations": self.n_observations, "n_units": self.n_units,
                "uncertainty": dict(self.uncertainty) if self.uncertainty else None}

    @classmethod
    def desde_json(cls, payload: Any, campo: str = "metric") -> "ValorDeMetrica":
        mapa = exigir_mapa(payload, campo)
        solo_estas_claves(mapa, ("metric_id", "formula_version", "value",
                                 "undefined_reason", "n_observations", "n_units",
                                 "uncertainty"), campo)
        return cls(metric_id=_saca(mapa, "metric_id", campo),
                   formula_version=_saca(mapa, "formula_version", campo),
                   value=mapa.get("value"), undefined_reason=mapa.get("undefined_reason"),
                   n_observations=mapa.get("n_observations"), n_units=mapa.get("n_units"),
                   uncertainty=mapa.get("uncertainty"))


@dataclass(frozen=True)
class EvaluationResult:
    """Los números de UN artefacto sobre UNA partición, con lo que valen.

    **El vínculo artefacto-evaluación es obligatorio** (`pipeline_digest` +
    `split_plan_digest`): es lo que el 106-C2 necesita para que la etiqueta de
    evaluación identifique artefacto y partición, y sin él dos versiones de un
    modelo comparten cifras que no son suyas.

    `evidence` no es decorativa: `independent_test` solo se puede decir de algo
    medido sobre el rol `test`. Quién puede otorgarla de verdad lo decide el
    registro de accesos, que es el único que sabe si alguien había mirado antes.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.evaluation_result"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "evaluation_id", "pipeline_digest", "split_plan_digest", "evaluated_role",
        "metrics", "evidence", "derives_from")

    evaluation_id: str
    pipeline_digest: str
    split_plan_digest: str
    evaluated_role: str
    evidence: str
    metrics: tuple[ValorDeMetrica, ...] = ()
    derives_from: str | None = None
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for nombre in ("evaluation_id", "pipeline_digest", "split_plan_digest"):
            exigir_texto(getattr(self, nombre), nombre)
        exigir_opcion(self.evaluated_role, "evaluated_role", ROLES)
        exigir_opcion(self.evidence, "evidence", ETIQUETAS_DE_EVIDENCIA)
        exigir_texto_o_nulo(self.derives_from, "derives_from")
        for metrica in self.metrics:
            if not isinstance(metrica, ValorDeMetrica):
                raise EsquemaInvalido("no_es_mapa", campo="metrics", valor=repr(metrica))

        coherencia = {
            "independent_test": ("test",),
            "external_validation": ("external_test",),
            "repeated_test_use": ROLES_RESERVADOS,
            "development_estimate": ROLES_DE_DESARROLLO,
        }
        permitidos = coherencia.get(self.evidence)
        if permitidos is not None and self.evaluated_role not in permitidos:
            raise EsquemaInvalido("evidencia_independiente_fuera_del_test",
                                  valor=self.evidence, opciones=self.evaluated_role)
        if self.evidence == "not_evaluated" and self.metrics:
            raise EsquemaInvalido("metrica_con_valor_y_motivo", campo="not_evaluated")

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id, "pipeline_digest": self.pipeline_digest,
            "split_plan_digest": self.split_plan_digest,
            "evaluated_role": self.evaluated_role, "evidence": self.evidence,
            "metrics": [m.a_json() for m in self.metrics],
            "derives_from": self.derives_from,
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "EvaluationResult":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        return cls(
            evaluation_id=_saca(cuerpo, "evaluation_id", cls.ESQUEMA),
            pipeline_digest=_saca(cuerpo, "pipeline_digest", cls.ESQUEMA),
            split_plan_digest=_saca(cuerpo, "split_plan_digest", cls.ESQUEMA),
            evaluated_role=_saca(cuerpo, "evaluated_role", cls.ESQUEMA),
            evidence=_saca(cuerpo, "evidence", cls.ESQUEMA),
            metrics=tuple(ValorDeMetrica.desde_json(m) for m in
                          _secuencia(cuerpo.get("metrics"), "metrics")),
            derives_from=cuerpo.get("derives_from"), migracion=marca)


def version_tras_aprender_del_test(evaluacion: EvaluationResult,
                                   nuevo_pipeline_digest: str,
                                   motivo_del_cambio: dict[str, str]
                                   ) -> EvaluationResult:
    """El artefacto que aprendió del test, con su evaluación puesta en su sitio.

    El contrato lo dice literal: si después se aprende con el test, se registra
    **nueva versión** y el estado `test_used_for_development`, y el artefacto
    nuevo **no conserva la etiqueta de evaluación independiente**.

    Por eso esto devuelve una evaluación **sin métricas**: las cifras eran del
    artefacto anterior y atribuírselas al nuevo sería justo lo que el 104-C5
    prohíbe. Lo que sí se conserva es el vínculo (`derives_from`), para que el
    historial no se pierda — no se borra nada, se recoloca.
    """
    exigir_texto(nuevo_pipeline_digest, "nuevo_pipeline_digest")
    exigir_motivo_bilingue(motivo_del_cambio, "motivo_del_cambio")
    if nuevo_pipeline_digest == evaluacion.pipeline_digest:
        raise EsquemaInvalido("hay_duplicados", campo="pipeline_digest",
                              valor=repr(nuevo_pipeline_digest))
    return EvaluationResult(
        evaluation_id=f"{evaluacion.evaluation_id}+test_used_for_development",
        pipeline_digest=nuevo_pipeline_digest,
        split_plan_digest=evaluacion.split_plan_digest,
        evaluated_role=evaluacion.evaluated_role,
        evidence="test_used_for_development",
        metrics=(),
        derives_from=evaluacion.evaluation_id)


@dataclass(frozen=True)
class SelectionDecision:
    """Qué se eligió, por qué, y qué se descartó.

    Tres cosas que el 104-C3 separa y aquí no se pueden confundir: `selected`
    (hay ganador), `no_feasible_model` (ninguno cumple los mínimos: es un
    veredicto) e `insufficient_evidence` (no hay datos para afirmarlo: es la
    ausencia de un veredicto).

    Y la regla que da nombre a este corte: **una decisión no se puede apoyar en
    una evaluación del test**. No es una comprobación de cortesía: es el
    invariante 2 del contrato metido en el propio documento, para que ni siquiera
    se pueda escribir un estudio que lo viole.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.selection_decision"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "decision_id", "outcome", "chosen_candidate", "reason", "policy",
        "evidence", "rejected", "constraints_checked", "split_plan_digest")

    decision_id: str
    outcome: str
    reason: dict[str, str]
    policy: str
    split_plan_digest: str
    chosen_candidate: str | None = None
    evidence: tuple[EvaluationResult, ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    constraints_checked: tuple[str, ...] = ()
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.decision_id, "decision_id")
        exigir_texto(self.policy, "policy")
        exigir_texto(self.split_plan_digest, "split_plan_digest")
        exigir_opcion(self.outcome, "outcome", RESULTADOS_DE_SELECCION)
        object.__setattr__(self, "reason", exigir_motivo_bilingue(self.reason, "reason"))
        object.__setattr__(self, "constraints_checked",
                           exigir_tupla_de_textos(self.constraints_checked,
                                                  "constraints_checked"))
        for evaluacion in self.evidence:
            if not isinstance(evaluacion, EvaluationResult):
                raise EsquemaInvalido("no_es_mapa", campo="evidence", valor=repr(evaluacion))
            if evaluacion.evaluated_role in ROLES_RESERVADOS:
                raise EsquemaInvalido("seleccionar_con_el_test",
                                      valor=evaluacion.evaluated_role)
        descartados = []
        for descartado in self.rejected:
            mapa = exigir_mapa(descartado, "rejected")
            solo_estas_claves(mapa, ("candidate", "reason"), "rejected")
            exigir_texto(_saca(mapa, "candidate", "rejected"), "rejected.candidate")
            descartados.append({"candidate": mapa["candidate"],
                                "reason": exigir_motivo_bilingue(
                                    _saca(mapa, "reason", "rejected"), "rejected.reason")})
        object.__setattr__(self, "rejected", tuple(descartados))

        if self.outcome == "selected":
            if self.chosen_candidate is None:
                raise EsquemaInvalido("ganador_sin_candidato", valor=self.outcome)
            exigir_texto(self.chosen_candidate, "chosen_candidate")
            if not self.evidence:
                raise EsquemaInvalido("ganador_sin_evidencia")
        elif self.chosen_candidate is not None:
            raise EsquemaInvalido("sin_ganador_con_candidato", valor=self.outcome)

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id, "outcome": self.outcome,
            "chosen_candidate": self.chosen_candidate, "reason": dict(self.reason),
            "policy": self.policy, "evidence": [e.a_json() for e in self.evidence],
            "rejected": [dict(r) for r in self.rejected],
            "constraints_checked": list(self.constraints_checked),
            "split_plan_digest": self.split_plan_digest,
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: RegistroDeMigraciones | None = None) -> "SelectionDecision":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        return cls(
            decision_id=_saca(cuerpo, "decision_id", cls.ESQUEMA),
            outcome=_saca(cuerpo, "outcome", cls.ESQUEMA),
            reason=_saca(cuerpo, "reason", cls.ESQUEMA),
            policy=_saca(cuerpo, "policy", cls.ESQUEMA),
            split_plan_digest=_saca(cuerpo, "split_plan_digest", cls.ESQUEMA),
            chosen_candidate=cuerpo.get("chosen_candidate"),
            evidence=tuple(EvaluationResult.desde_json(e, migraciones=migraciones)
                           for e in _secuencia(cuerpo.get("evidence"), "evidence")),
            rejected=_secuencia(cuerpo.get("rejected"), "rejected"),
            constraints_checked=_secuencia(cuerpo.get("constraints_checked"),
                                           "constraints_checked"),
            migracion=marca)
