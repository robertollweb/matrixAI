# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El registro métrico ÚNICO — 105-C1.

Aquí viven las FÓRMULAS, una sola vez, para que entrenamiento, `attest`,
benchmark, selección y `verify` den el mismo número al mismo dato. El invariante
1 del 105 lo dice sin rodeos: registro único, y no duplicar fórmulas en 102-C0.
Dos sitios calculando el mismo AUROC acaban divergiendo, y el día que diverjan
nadie sabrá cuál de los dos informes firmó el recibo.

LO QUE ESTE MÓDULO SOSTIENE, que son los invariantes del contrato:

* **AP no es el área PR trapezoidal** (invariante 3). Son dos identidades
  distintas con dos referencias distintas y por eso llevan **dos IDs**:
  `average_precision` suma escalones —`sum((R_k - R_{k-1}) * P_k)`, exactamente
  `sklearn.metrics.average_precision_score`— y `pr_auc_trapezoidal` traza rectas
  entre los puntos —exactamente `sklearn.metrics.auc(recall, precision)`—.
  El trapecio INVENTA precisiones intermedias que ningún umbral alcanza: con
  `y = (no, si, si, no)` y puntuaciones `(0.9, 0.8, 0.7, 0.6)`, AP vale
  0,583333 y el trapecio 0,416667. Publicar el segundo llamándolo AP es
  atribuirle al modelo un número que no es suyo.
* **AUROC y AP admiten puntuaciones continuas ORDENADAS; log-loss y Brier
  necesitan PROBABILIDADES** (invariante 2). Unos logits ordenan igual de bien
  y no son probabilidades: pedirle el Brier a la muestra que solo trae
  puntuaciones se RECHAZA, no se calcula sobre un número que no está en [0,1].
  Y **una etiqueta dura no se presenta como probabilidad**: con la clase
  predicha y nada más, AUROC tampoco se calcula.
* **No todo se ordena por «mayor mejor»** (invariante 4). `calibration_in_the_
  large` vale idealmente 0 y desviarse en cualquiera de los dos sentidos es
  peor; `MetricSpec` del 104-C0 ya se niega a llevar dirección Y valor ideal a
  la vez, y `es_mejor()` de aquí ordena por distancia al ideal cuando no hay
  dirección. Ordenar el intercepto por «mayor mejor» corona al modelo que más
  sobreestima.

DÓNDE ESTÁ LA FRONTERA ENTRE **ERROR** E **INDEFINICIÓN**, que es la otra mitad
del corte y no es una preferencia de estilo:

* Es **error** —excepción, fail-closed— lo que no puede pasar sin que alguien
  haya cableado mal: un NaN o un infinito en una puntuación, una probabilidad
  fuera de [0,1], una distribución que no suma uno, pedir Brier sobre logits,
  pedir AUROC sobre etiquetas duras, mezclar registros de dos candidatos.
  Devolver «indefinida» ahí dejaría el fallo dentro de un campo que ya nadie
  mira, que es justo lo que el 82-C1 decidió no hacer.
* Es **indefinición motivada** —`ValorDeMetrica` con `undefined_reason` y sin
  valor— lo que son los datos y no el cableado: una sola clase presente, cero
  filas, un denominador a cero (ningún predicho positivo, varianza nula).
  **Nunca un cero inventado**: un AUROC indefinido no vale 0,5, un PPV sin
  predichos positivos no vale 0 y un R² sin varianza no vale 0.

QUÉ SE REGISTRA DE CADA MÉTRICA, que es lo que pide el corte: la clase positiva,
el orden de clases, el clipping, la normalización del Brier, los pesos y la
versión de fórmula. Los cinco primeros viajan en el `MetricSpec` del 104-C0 —que
se REUTILIZA, no se duplica— y el orden de clases, el umbral y la regla de
decisión viajan en el `InformeMetrico`, que es quien los conoce.

CAMPOS HISTÓRICOS. `accuracy`, `macro_f1`, la matriz de confusión, `mae`, `rmse`
y `r2` valen exactamente lo que valen hoy en `matrixai/training/dense_evaluator.py`
—redondeo a seis decimales del macro-F1 incluido, que no es cosmético: cambiarlo
cambiaría cifras ya publicadas—. El informe ampliado es un documento NUEVO, con
su `report_version` y su digest, y no toca al viejo. La única divergencia
deliberada está escrita abajo, en `_r2`: donde el evaluador histórico devuelve
`0.0` porque el objetivo no tiene varianza, aquí se devuelve indefinida con su
motivo, porque un valor ausente no es un cero.

Y el core sigue siendo **stdlib puro**: `math`, `sys`, `dataclasses`. Las
referencias numéricas (scikit-learn, numpy) viven en las PRUEBAS y con
`skipUnless`; aquí dentro no entra ninguna.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, ClassVar, Iterable, Sequence

from matrixai.estudio.errores import ErrorDeEstudio, EsquemaInvalido
from matrixai.estudio.esquemas import (
    TOLERANCIA_SUMA,
    Aptitud,
    MetricSpec,
    PredictionRecord,
    ValorDeMetrica,
    _sobre,  # el MISMO sobre que el resto del paquete: dos sobres, dos digests
)
from matrixai.estudio.textos import motivo
from matrixai.estudio.validacion import (
    digest_canonico,
    exigir_real,
    exigir_texto_o_nulo,
    exigir_tupla_de_textos,
)
from matrixai.estudio.vocabulario import (
    TAREAS,
    TAREAS_DE_CLASIFICACION,
    exigir_opcion,
)

__all__ = [
    "CLIP_LOG_LOSS", "INFORME_VERSION", "METRICAS_DIFERIDAS", "REGISTRO",
    "UMBRAL_POR_DEFECTO", "VERSION_DE_LA_MATRIZ",
    "EntradaNoMedible", "Indefinida", "InformeMetrico", "MatrizDeConfusion",
    "MetricaAplazada", "MetricaDesconocida", "MetricaRegistrada", "Muestra",
    "aplazamiento", "aptitud", "calcular", "catalogo", "digest_del_catalogo",
    "direccion_de", "distancia_al_ideal", "es_mejor", "especificacion",
    "evaluar", "matriz_de_confusion", "metricas_aplicables", "nombre_de_la_metrica",
    "ordenar_por",
]


# ---------------------------------------------------------------------------
# Rechazos propios del registro
# ---------------------------------------------------------------------------

class MetricaDesconocida(ErrorDeEstudio):
    """Se ha pedido una métrica que no está en el catálogo.

    El catálogo es cerrado por lo mismo que el vocabulario del 104-C0: una
    métrica improvisada no tiene fórmula, ni versión, ni dirección, y el número
    que saliera no se podría comparar con nada ni volver a calcular.
    """


class MetricaAplazada(ErrorDeEstudio):
    """La métrica existe como decisión y NO está implementada, con su motivo.

    No es lo mismo que `MetricaDesconocida`: aquí alguien la miró, decidió
    diferirla y escribió por qué. `METRICAS_DIFERIDAS` las lista.
    """


class EntradaNoMedible(EsquemaInvalido):
    """La muestra no puede sostener esa métrica, y es un fallo de cableado.

    Se distingue a propósito de la indefinición motivada: aquí no es que los
    datos salgan degenerados, es que se está pidiendo un Brier sobre logits, un
    AUROC sobre etiquetas duras o una métrica de otra tarea.
    """


# ---------------------------------------------------------------------------
# Constantes que el informe declara (y por las que se puede preguntar)
# ---------------------------------------------------------------------------

#: El clipping de log-loss. Es **el mismo** que usa `sklearn.metrics.log_loss`
#: (`np.finfo(float64).eps`), y por eso la paridad se puede exigir incluso con
#: probabilidades de 0 y 1 exactos, donde sin clipping el logaritmo se va a
#: infinito y el informe entero deja de ser JSON.
CLIP_LOG_LOSS: float = sys.float_info.epsilon

#: El umbral por omisión, y solo sobre la ESCALA DE PROBABILIDAD. Sobre
#: puntuaciones sin normalizar no hay umbral por omisión que valga: 0,5 cortaría
#: en un sitio arbitrario de una escala que nadie ha acotado.
UMBRAL_POR_DEFECTO: float = 0.5

#: La versión del documento `metric_report`. El informe ampliado es un documento
#: nuevo, con versión y digest propios; los campos históricos no se tocan.
INFORME_VERSION = "1.0.0"

#: La versión de la fórmula de la matriz de confusión. No es un `MetricSpec` a
#: propósito: una matriz no tiene dirección ni valor ideal, y `MetricSpec` se
#: NIEGA —con razón— a llevar ninguna de las dos. Se publica como bloque propio.
VERSION_DE_LA_MATRIZ = "1.0.0"

#: Lo que se difiere, con la CLAVE de su motivo bilingüe. Pedirlas levanta
#: `MetricaAplazada` con la explicación escrita, que es distinto de no conocerlas.
METRICAS_DIFERIDAS: dict[str, str] = {
    "mape": "mape_aplazada",
    "brier_multiclass": "brier_multiclase_aplazado",
}


@dataclass(frozen=True)
class Indefinida:
    """El resultado de una métrica que no se puede calcular CON ESTOS DATOS.

    Lo devuelven las fórmulas y lo convierte `calcular()` en un `ValorDeMetrica`
    sin valor y con motivo. No es un cero disfrazado: es la ausencia, dicha.
    """

    motivo: dict[str, str]


# ---------------------------------------------------------------------------
# La muestra: lo que se mide, ya alineado y ya validado
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Muestra:
    """Las predicciones y la verdad de UNA partición, alineadas y comprobadas.

    Es la entrada de todas las fórmulas. Todo lo IMPOSIBLE se rechaza aquí, una
    vez, en vez de dejar que cada fórmula se defienda por su cuenta: NaN, ±inf,
    probabilidades fuera de [0,1], distribuciones que no suman uno, columnas
    desalineadas, una `y_true` que no está entre las clases.

    Cero filas SÍ se admite: una muestra vacía —un segmento sin nadie— no es un
    documento imposible, es una muestra sobre la que toda métrica sale
    indefinida con su motivo. Ahí está la diferencia entre las dos mitades del
    corte.
    """

    task: str
    y_true: tuple[Any, ...]
    classes: tuple[str, ...] | None = None
    positive_label: str | None = None
    #: Distribución COMPLETA por fila, alineada con `classes`. Media distribución
    #: no se recompone sin saber qué clase falta (104-C0).
    probabilities: tuple[tuple[float, ...], ...] | None = None
    #: La puntuación de decisión del positivo, en cualquier escala. Solo binaria.
    scores: tuple[float, ...] | None = None
    #: La clase predicha (clasificación) o el valor predicho (regresión).
    predictions: tuple[Any, ...] | None = None
    weights: tuple[float, ...] | None = None
    units: tuple[str, ...] | None = None
    #: De dónde salió `scores`, para que el informe lo declare en vez de que
    #: alguien tenga que adivinarlo leyendo este fichero.
    score_rule: str | None = None

    def __post_init__(self) -> None:
        exigir_opcion(self.task, "task", TAREAS)
        object.__setattr__(self, "y_true", tuple(self.y_true))
        n = len(self.y_true)

        if self.task in TAREAS_DE_CLASIFICACION:
            if self.classes is None:
                raise EsquemaInvalido("clasificacion_sin_clases")
            clases = exigir_tupla_de_textos(self.classes, "classes", minimo=2)
            object.__setattr__(self, "classes", clases)
            if self.task == "binary_classification" and len(clases) != 2:
                raise EsquemaInvalido("binaria_con_otro_numero_de_clases",
                                      valor=len(clases))
            if self.task == "multiclass_classification" and len(clases) < 3:
                raise EsquemaInvalido("multiclase_con_menos_de_tres", valor=len(clases))
            if self.task == "binary_classification" and self.positive_label is None:
                raise EsquemaInvalido("binaria_sin_clase_positiva")
            if self.positive_label is not None and self.positive_label not in clases:
                raise EsquemaInvalido("clase_positiva_fuera_de_clases",
                                      valor=repr(self.positive_label),
                                      opciones=list(clases))
            for i, y in enumerate(self.y_true):
                if y not in clases:
                    raise EsquemaInvalido("no_es_del_vocabulario", campo=f"y_true[{i}]",
                                          opciones=list(clases), valor=repr(y))
        else:
            if self.classes is not None:
                raise EsquemaInvalido("clases_en_regresion", valor=list(self.classes))
            if self.positive_label is not None:
                raise EsquemaInvalido("clase_positiva_en_regresion",
                                      valor=repr(self.positive_label))
            object.__setattr__(self, "y_true", tuple(
                exigir_real(y, f"y_true[{i}]") for i, y in enumerate(self.y_true)))

        if self.probabilities is not None:
            if self.classes is None:
                raise EsquemaInvalido("columnas_sin_clases", campo="probabilities")
            filas = self._filas(self.probabilities, "probabilities", n)
            normalizadas = []
            for i, fila in enumerate(filas):
                if isinstance(fila, (str, bytes)) or not isinstance(fila, (list, tuple)):
                    raise EsquemaInvalido("no_es_lista_de_textos",
                                          campo=f"probabilities[{i}]", valor=repr(fila))
                if len(fila) != len(self.classes):
                    raise EsquemaInvalido("columnas_desalineadas",
                                          campo=f"probabilities[{i}]", valor=len(fila),
                                          opciones=len(self.classes))
                valores = tuple(
                    exigir_real(p, f"probabilities[{i}][{j}]", minimo=0.0, maximo=1.0)
                    for j, p in enumerate(fila))
                suma = sum(valores)
                if abs(suma - 1.0) > TOLERANCIA_SUMA:
                    raise EsquemaInvalido("probabilidades_no_suman_uno", valor=repr(suma))
                normalizadas.append(valores)
            object.__setattr__(self, "probabilities", tuple(normalizadas))

        if self.scores is not None:
            if self.task != "binary_classification":
                raise EntradaNoMedible("metrica_de_otra_tarea", campo="scores",
                                       opciones="binary_classification", valor=self.task)
            filas = self._filas(self.scores, "scores", n)
            object.__setattr__(self, "scores", tuple(
                exigir_real(s, f"scores[{i}]") for i, s in enumerate(filas)))

        if self.predictions is not None:
            filas = self._filas(self.predictions, "predictions", n)
            if self.classes is not None:
                for i, p in enumerate(filas):
                    if p not in self.classes:
                        raise EsquemaInvalido("no_es_del_vocabulario",
                                              campo=f"predictions[{i}]",
                                              opciones=list(self.classes), valor=repr(p))
                object.__setattr__(self, "predictions", tuple(filas))
            else:
                object.__setattr__(self, "predictions", tuple(
                    exigir_real(p, f"predictions[{i}]") for i, p in enumerate(filas)))

        if self.weights is not None:
            filas = self._filas(self.weights, "weights", n)
            object.__setattr__(self, "weights", tuple(
                exigir_real(w, f"weights[{i}]", minimo=0.0) for i, w in enumerate(filas)))

        if self.units is not None:
            unidades = exigir_tupla_de_textos(self.units, "units", sin_duplicados=False)
            if len(unidades) != n:
                raise EntradaNoMedible("filas_desalineadas", campo="units",
                                       valor=len(unidades), opciones=n)
            object.__setattr__(self, "units", unidades)

        exigir_texto_o_nulo(self.score_rule, "score_rule")

        if n and self.probabilities is None and self.scores is None \
                and self.predictions is None:
            raise EsquemaInvalido("muestra_sin_salida")

    @staticmethod
    def _filas(valor: Any, campo: str, n: int) -> tuple:
        if isinstance(valor, (str, bytes)) or not isinstance(valor, (list, tuple)):
            raise EsquemaInvalido("no_es_lista_de_textos", campo=campo, valor=repr(valor))
        filas = tuple(valor)
        if len(filas) != n:
            raise EntradaNoMedible("filas_desalineadas", campo=campo,
                                   valor=len(filas), opciones=n)
        return filas

    # -- lo que se pregunta de una muestra --------------------------------
    @property
    def n(self) -> int:
        return len(self.y_true)

    @property
    def n_unidades(self) -> int | None:
        """Cuántas unidades de remuestreo distintas hay. `None` si nadie las
        declaró: filas y pacientes no son intercambiables (invariante 5)."""
        return len(set(self.units)) if self.units is not None else None

    @property
    def pesos(self) -> tuple[float, ...]:
        return self.weights if self.weights is not None else (1.0,) * self.n

    @property
    def puntuacion_del_positivo(self) -> tuple[float, ...] | None:
        """La puntuación que ORDENA, en la MISMA escala que declara
        `escala_de_decision`. Una probabilidad ES una puntuación ordenada; al
        revés no (invariante 2).

        CON LAS DOS PRESENTES MANDA LA PROBABILIDAD. Hasta el 2026-09-17 mandaba
        `scores`, mientras `escala_de_decision` dice `calibrated_probability` en
        cuanto hay `probabilities`: la muestra declaraba una escala y ordenaba
        por otra, y `matriz_de_confusion` sin umbral aplicaba el 0,5 —un umbral de
        PROBABILIDAD— sobre logits. Medido el 2026-09-16, mismo modelo y datos:
        solo probabilidades, VP=3 FN=0; con las dos, VP=1 FN=2, declarando
        «escala de probabilidad, umbral 0,5» y sin error. Los motores de la casa
        no lo pisaban porque su `scores` es la columna positiva de
        `predict_proba`, el mismo número; uno que expusiera su margen crudo, sí.
        Si hay probabilidades pero no se puede saber cuál es la del positivo,
        `None`: volver a `scores` sería ordenar otra vez en otra escala."""
        if self.probabilities is not None:
            return self.probabilidad_del_positivo
        return self.scores

    @property
    def probabilidad_del_positivo(self) -> tuple[float, ...] | None:
        if self.probabilities is None or self.positive_label is None \
                or self.classes is None:
            return None
        j = self.classes.index(self.positive_label)
        return tuple(fila[j] for fila in self.probabilities)

    @property
    def escala_de_decision(self) -> str | None:
        """El vocabulario del 104 (`ESCALAS_DE_DECISION`). `calibrated_probability`
        nombra la ESCALA de probabilidad; no promete que esté calibrada —eso lo
        mide el 105-C3—."""
        if self.probabilities is not None:
            return "calibrated_probability"
        if self.scores is not None:
            return "raw_score"
        return None

    # -- constructores ----------------------------------------------------
    @classmethod
    def binaria(cls, y_true: Sequence[Any], *, classes: Sequence[str],
                positive_label: str, probabilidades: Sequence[float] | None = None,
                puntuaciones: Sequence[float] | None = None,
                predicciones: Sequence[Any] | None = None,
                pesos: Sequence[float] | None = None,
                unidades: Sequence[str] | None = None) -> "Muestra":
        """Una muestra binaria. `probabilidades` es la del POSITIVO —la forma en
        que la escribe todo el mundo— y aquí se expande a la distribución
        completa alineada con `classes`, que es como la guarda el 104-C0."""
        clases = tuple(classes)
        distribuciones = None
        if probabilidades is not None:
            if positive_label not in clases:
                raise EsquemaInvalido("clase_positiva_fuera_de_clases",
                                      valor=repr(positive_label), opciones=list(clases))
            j = clases.index(positive_label)
            filas = []
            for i, p in enumerate(probabilidades):
                v = exigir_real(p, f"probabilidades[{i}]", minimo=0.0, maximo=1.0)
                fila = [0.0, 0.0]
                fila[j] = v
                fila[1 - j] = 1.0 - v
                filas.append(tuple(fila))
            distribuciones = tuple(filas)
        regla = None
        if puntuaciones is not None:
            regla = "puntuacion_declarada"
        elif distribuciones is not None:
            regla = "probabilidad_del_positivo"
        return cls(task="binary_classification", y_true=tuple(y_true), classes=clases,
                   positive_label=positive_label, probabilities=distribuciones,
                   scores=tuple(puntuaciones) if puntuaciones is not None else None,
                   predictions=tuple(predicciones) if predicciones is not None else None,
                   weights=tuple(pesos) if pesos is not None else None,
                   units=tuple(unidades) if unidades is not None else None,
                   score_rule=regla)

    @classmethod
    def multiclase(cls, y_true: Sequence[Any], *, classes: Sequence[str],
                   probabilidades: Sequence[Sequence[float]] | None = None,
                   predicciones: Sequence[Any] | None = None,
                   pesos: Sequence[float] | None = None,
                   unidades: Sequence[str] | None = None) -> "Muestra":
        """Una muestra multiclase. Las probabilidades van como distribución
        completa por fila, en el orden de `classes`."""
        return cls(task="multiclass_classification", y_true=tuple(y_true),
                   classes=tuple(classes),
                   probabilities=(tuple(tuple(f) for f in probabilidades)
                                  if probabilidades is not None else None),
                   predictions=tuple(predicciones) if predicciones is not None else None,
                   weights=tuple(pesos) if pesos is not None else None,
                   units=tuple(unidades) if unidades is not None else None)

    @classmethod
    def regresion(cls, y_true: Sequence[float], predicciones: Sequence[float], *,
                  pesos: Sequence[float] | None = None,
                  unidades: Sequence[str] | None = None) -> "Muestra":
        return cls(task="regression", y_true=tuple(y_true),
                   predictions=tuple(predicciones),
                   weights=tuple(pesos) if pesos is not None else None,
                   units=tuple(unidades) if unidades is not None else None)

    @classmethod
    def desde_registros(cls, registros: Iterable[PredictionRecord], *, task: str,
                        positive_label: str | None = None,
                        classes: Sequence[str] | None = None,
                        rol: str | None = None) -> "Muestra":
        """El puente con el `PredictionRecord` del 104-C0.

        Lo que se comprueba aquí no es forma, es SENTIDO: que todos los registros
        sean del mismo candidato (mezclar dos modelos no mide ninguno), que no se
        repita un `row_id`, que las clases declaradas sean las mismas en todos
        —dos órdenes distintos no se mezclan— y que ninguno venga sin `y_true`,
        porque una muestra de evaluación mide CONTRA algo.

        La puntuación binaria sale de `scores`: una columna es la del positivo;
        dos columnas se leen como `positivo - negativo`, que es la función de
        decisión de siempre y la única lectura que conserva el orden tanto si son
        probabilidades como si son logits (tomar solo la columna del positivo
        ordenaría mal unos logits). La regla se declara en `score_rule`.
        """
        filas = [r for r in registros if rol is None or r.role == rol]
        candidatos = sorted({r.candidate for r in filas})
        if len(candidatos) > 1:
            raise EntradaNoMedible("registros_de_varios_candidatos",
                                   valor=len(candidatos), opciones=candidatos)
        vistos: set[str] = set()
        for r in filas:
            if r.row_id in vistos:
                raise EntradaNoMedible("hay_duplicados", campo="row_id",
                                       valor=repr(r.row_id))
            vistos.add(r.row_id)
            if r.y_true is None:
                raise EntradaNoMedible("registro_sin_verdad", valor=repr(r.row_id))

        declaradas = {r.classes for r in filas if r.classes is not None}
        if len(declaradas) > 1:
            dos = sorted(declaradas, key=lambda c: list(c))
            raise EntradaNoMedible("registros_con_clases_distintas",
                                   valor=list(dos[0]), opciones=list(dos[1]))
        clases = tuple(classes) if classes is not None else (
            tuple(next(iter(declaradas))) if declaradas else None)

        def _todos_o_ninguno(campo: str, tiene: Callable[[PredictionRecord], bool]) -> bool:
            cuantos = sum(1 for r in filas if tiene(r))
            if cuantos and cuantos != len(filas):
                raise EntradaNoMedible("registros_desiguales", campo=campo)
            return bool(cuantos)

        hay_prob = _todos_o_ninguno("probabilities", lambda r: r.probabilities is not None)
        hay_score = _todos_o_ninguno("scores", lambda r: r.scores is not None)
        hay_label = _todos_o_ninguno("label", lambda r: r.label is not None)
        hay_peso = _todos_o_ninguno("weight", lambda r: r.weight is not None)
        hay_unidad = _todos_o_ninguno("resampling_unit",
                                      lambda r: r.resampling_unit is not None)

        probabilidades = tuple(r.probabilities for r in filas) if hay_prob else None
        puntuaciones = None
        regla = None
        if task == "binary_classification" and hay_score and clases is not None:
            j = clases.index(positive_label) if positive_label in clases else None
            if j is None:
                raise EsquemaInvalido("clase_positiva_fuera_de_clases",
                                      valor=repr(positive_label), opciones=list(clases))
            if all(len(r.scores) == 1 for r in filas):
                puntuaciones = tuple(r.scores[0] for r in filas)
                regla = "columna_unica_del_positivo"
            else:
                puntuaciones = tuple(r.scores[j] - r.scores[1 - j] for r in filas)
                regla = "positivo_menos_negativo"
        elif hay_prob:
            regla = "probabilidad_del_positivo"

        return cls(task=task, y_true=tuple(r.y_true for r in filas), classes=clases,
                   positive_label=positive_label, probabilities=probabilidades,
                   scores=puntuaciones,
                   predictions=tuple(r.label for r in filas) if hay_label else None,
                   weights=tuple(r.weight for r in filas) if hay_peso else None,
                   units=tuple(r.resampling_unit for r in filas) if hay_unidad else None,
                   score_rule=regla)


# ---------------------------------------------------------------------------
# La matriz de confusión: bloque propio, NO un MetricSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatrizDeConfusion:
    """Los recuentos `{real: {predicha: n}}`, con la regla que decidió la clase.

    **No es un `MetricSpec` y no puede serlo**: una matriz no tiene dirección ni
    valor ideal, y el `MetricSpec` del 104-C0 se niega a construirse sin una de
    las dos. Esa negativa es correcta —ordenar una matriz por «mayor mejor» no
    significa nada—, así que la matriz viaja como bloque con su propia versión
    de fórmula.

    `historica()` devuelve exactamente la forma que publica hoy
    `dense_evaluator.to_dict()["confusion_matrix"]`, para que el campo histórico
    no cambie ni de nombre ni de valor.
    """

    classes: tuple[str, ...]
    counts: dict[str, dict[str, int]]
    decision_rule: str
    positive_label: str | None = None
    threshold: float | None = None
    formula_version: str = VERSION_DE_LA_MATRIZ

    def historica(self) -> dict[str, dict[str, int]]:
        return {real: dict(fila) for real, fila in self.counts.items()}

    def a_json(self) -> dict[str, Any]:
        return {"classes": list(self.classes), "counts": self.historica(),
                "decision_rule": self.decision_rule,
                "positive_label": self.positive_label, "threshold": self.threshold,
                "formula_version": self.formula_version}

    def _binaria(self) -> tuple[str, str]:
        if self.positive_label is None or len(self.classes) != 2:
            raise EntradaNoMedible("metrica_de_otra_tarea", campo="confusion_matrix",
                                   opciones="binary_classification",
                                   valor="multiclass_classification")
        negativa = next(c for c in self.classes if c != self.positive_label)
        return self.positive_label, negativa

    @property
    def tp(self) -> int:
        pos, _ = self._binaria()
        return self.counts[pos][pos]

    @property
    def fn(self) -> int:
        pos, neg = self._binaria()
        return self.counts[pos][neg]

    @property
    def fp(self) -> int:
        pos, neg = self._binaria()
        return self.counts[neg][pos]

    @property
    def tn(self) -> int:
        _, neg = self._binaria()
        return self.counts[neg][neg]


def _argmax(valores: Sequence[float]) -> int:
    """El índice del máximo, y **el PRIMERO** cuando empatan.

    Es la misma regla que `dense_evaluator._argmax`. No es un detalle: en un
    empate exacto, elegir el último cambiaría la matriz de confusión y con ella
    accuracy y macro-F1 respecto de lo que el producto publica hoy.
    """
    return max(range(len(valores)), key=lambda i: valores[i])


def matriz_de_confusion(muestra: Muestra, *,
                        umbral: float | None = None) -> MatrizDeConfusion:
    """Los recuentos, con la regla de decisión declarada.

    Tres reglas, en este orden de prioridad:

    * `declared_label` — si la muestra trae la clase predicha, esa es la decisión
      del modelo y manda. Recalcularla desde las probabilidades podría
      contradecir lo que el propio modelo dijo.
    * `threshold` — binaria: positiva si su puntuación **alcanza** el umbral
      (`>=`, como el evaluador histórico; con `>` la fila que cae justo en 0,5
      cambiaría de lado).
    * `argmax` — multiclase: la clase de mayor probabilidad, primera en caso de
      empate.

    Los recuentos son observaciones ENTERAS: una muestra con pesos se rechaza en
    vez de devolver una matriz que ya no cuenta filas.
    """
    if muestra.task not in TAREAS_DE_CLASIFICACION:
        raise EntradaNoMedible("metrica_de_otra_tarea", campo="confusion_matrix",
                               opciones=list(TAREAS_DE_CLASIFICACION), valor=muestra.task)
    if muestra.weights is not None:
        raise EntradaNoMedible("pesos_no_admitidos", campo="confusion_matrix")
    clases = muestra.classes or ()

    regla: str
    umbral_usado: float | None = None
    if muestra.predictions is not None:
        regla = "declared_label"
        predichas = list(muestra.predictions)
    elif muestra.task == "binary_classification":
        puntuaciones = muestra.puntuacion_del_positivo
        if puntuaciones is None:
            raise EsquemaInvalido("muestra_sin_salida")
        if umbral is None:
            if muestra.probabilities is None:
                raise EntradaNoMedible("umbral_por_omision_en_puntuacion")
            umbral_usado = UMBRAL_POR_DEFECTO
        else:
            umbral_usado = exigir_real(umbral, "umbral")
        regla = "threshold"
        negativa = next(c for c in clases if c != muestra.positive_label)
        predichas = [muestra.positive_label if s >= umbral_usado else negativa
                     for s in puntuaciones]
    else:
        if muestra.probabilities is None:
            raise EsquemaInvalido("muestra_sin_salida")
        regla = "argmax"
        predichas = [clases[_argmax(fila)] for fila in muestra.probabilities]

    counts = {real: {pred: 0 for pred in clases} for real in clases}
    for real, pred in zip(muestra.y_true, predichas):
        counts[real][pred] += 1
    return MatrizDeConfusion(classes=tuple(clases), counts=counts, decision_rule=regla,
                             positive_label=muestra.positive_label,
                             threshold=umbral_usado)


# ---------------------------------------------------------------------------
# Las fórmulas
# ---------------------------------------------------------------------------

def _curva_binaria(muestra: Muestra) -> tuple[list[tuple[float, float]], float, float]:
    """Los `(tp, fp)` acumulados por valor DISTINTO de puntuación, de mayor a menor.

    Los empates van en el MISMO punto: partirlos daría un área que depende del
    orden en que llegaron las filas, y dos ejecuciones del mismo modelo sobre las
    mismas filas darían dos AUROC.
    """
    puntuaciones = muestra.puntuacion_del_positivo or ()
    pesos = muestra.pesos
    positivo = muestra.positive_label
    orden = sorted(range(len(puntuaciones)), key=lambda i: puntuaciones[i], reverse=True)
    puntos: list[tuple[float, float]] = []
    tp = fp = 0.0
    i = 0
    while i < len(orden):
        actual = puntuaciones[orden[i]]
        j = i
        while j < len(orden) and puntuaciones[orden[j]] == actual:
            k = orden[j]
            if muestra.y_true[k] == positivo:
                tp += pesos[k]
            else:
                fp += pesos[k]
            j += 1
        puntos.append((tp, fp))
        i = j
    return puntos, tp, fp


def _sin_dos_clases(muestra: Muestra, metric_id: str) -> Indefinida:
    presentes = sorted({str(y) for y in muestra.y_true})
    if len(presentes) < 2:
        return Indefinida(motivo("una_sola_clase", campo=metric_id,
                                 valor=presentes[0] if presentes else "ninguna"))
    return Indefinida(motivo("division_por_cero_en_metrica", campo=metric_id,
                             valor="el peso total de una de las dos clases"))


def _auroc(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    puntos, tp_total, fp_total = _curva_binaria(muestra)
    if tp_total == 0.0 or fp_total == 0.0:
        return _sin_dos_clases(muestra, "auroc")
    area = 0.0
    tp_prev = fp_prev = 0.0
    for tp, fp in puntos:
        area += (fp - fp_prev) * (tp + tp_prev) / 2.0
        tp_prev, fp_prev = tp, fp
    return area / (tp_total * fp_total)


def _puntos_pr(puntos: list[tuple[float, float]],
               tp_total: float) -> list[tuple[float, float]]:
    """`(recall, precision)` en cada umbral distinto, de menor a mayor recall."""
    salida = []
    for tp, fp in puntos:
        positivos_predichos = tp + fp
        precision = tp / positivos_predichos if positivos_predichos > 0.0 else 0.0
        salida.append((tp / tp_total, precision))
    return salida


def _average_precision(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    """AP: la suma de ESCALONES, `sum((R_k - R_{k-1}) * P_k)`.

    Cada sumando es precisión REALMENTE observada en un umbral. No interpola
    nada, y por eso no es el área trapezoidal.
    """
    puntos, tp_total, fp_total = _curva_binaria(muestra)
    if tp_total == 0.0:
        return _sin_dos_clases(muestra, "average_precision")
    ap = 0.0
    recall_previo = 0.0
    for recall, precision in _puntos_pr(puntos, tp_total):
        ap += (recall - recall_previo) * precision
        recall_previo = recall
    return ap


def _pr_auc_trapezoidal(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    """El área bajo la curva PR uniendo los puntos con RECTAS.

    Es otra identidad: entre dos umbrales consecutivos supone precisiones
    intermedias que ningún umbral alcanza. Se ofrece con ID propio porque a
    veces se pide, y NUNCA se publica como `average_precision`.
    """
    puntos, tp_total, fp_total = _curva_binaria(muestra)
    if tp_total == 0.0:
        return _sin_dos_clases(muestra, "pr_auc_trapezoidal")
    # El punto ancla (recall 0, precisión 1) es el mismo que devuelve
    # `precision_recall_curve`; sin él el trapecio empezaría en el aire.
    curva = [(0.0, 1.0)] + _puntos_pr(puntos, tp_total)
    area = 0.0
    for (r1, p1), (r2, p2) in zip(curva, curva[1:]):
        area += (r2 - r1) * (p1 + p2) / 2.0
    return area


def _log_loss(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    """Log-loss media, con el clipping DECLARADO en `CLIP_LOG_LOSS`.

    Sirve para binaria y para multiclase con la misma fórmula, que es la razón
    de que haya un solo ID: `-sum(w * ln p[y]) / sum(w)`.
    """
    indices = {c: j for j, c in enumerate(muestra.classes or ())}
    total = 0.0
    peso_total = 0.0
    for i, y in enumerate(muestra.y_true):
        p = muestra.probabilities[i][indices[y]]
        p = min(max(p, CLIP_LOG_LOSS), 1.0 - CLIP_LOG_LOSS)
        total += -muestra.pesos[i] * math.log(p)
        peso_total += muestra.pesos[i]
    if peso_total == 0.0:
        return Indefinida(motivo("division_por_cero_en_metrica", campo="log_loss",
                                 valor="la suma de los pesos"))
    return total / peso_total


def _brier(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    """Brier binario clásico: media de `(p - 1[y = positiva])^2`, escala [0, 1]."""
    probabilidades = muestra.probabilidad_del_positivo or ()
    total = 0.0
    peso_total = 0.0
    for i, y in enumerate(muestra.y_true):
        real = 1.0 if y == muestra.positive_label else 0.0
        total += muestra.pesos[i] * (probabilidades[i] - real) ** 2
        peso_total += muestra.pesos[i]
    if peso_total == 0.0:
        return Indefinida(motivo("division_por_cero_en_metrica", campo="brier_score",
                                 valor="la suma de los pesos"))
    return total / peso_total


def _calibracion_en_grande(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    """Probabilidad media predicha menos prevalencia observada. Ideal: 0.

    Es «calibración en grande» y NADA MÁS: no es la pendiente ni el intercepto
    del recalibrado logístico —eso es el 105-C3, que ajusta un modelo—, y valer 0
    no certifica que el modelo esté calibrado. Está aquí porque es la métrica que
    obliga a que el registro sepa ordenar por VALOR IDEAL: +0,2 y -0,2 son
    igual de malos, y «mayor mejor» coronaría al que más sobreestima.
    """
    probabilidades = muestra.probabilidad_del_positivo or ()
    peso_total = sum(muestra.pesos)
    if peso_total == 0.0:
        return Indefinida(motivo("division_por_cero_en_metrica",
                                 campo="calibration_in_the_large",
                                 valor="la suma de los pesos"))
    media_p = sum(w * p for w, p in zip(muestra.pesos, probabilidades)) / peso_total
    prevalencia = sum(w for w, y in zip(muestra.pesos, muestra.y_true)
                      if y == muestra.positive_label) / peso_total
    return media_p - prevalencia


def _desde_la_matriz(metric_id: str, numerador: Callable[[MatrizDeConfusion], int],
                     denominador: Callable[[MatrizDeConfusion], int],
                     nombre_del_denominador: str
                     ) -> Callable[[Muestra, float | None], float | Indefinida]:
    """Las cuatro del umbral salen de la MISMA matriz: escribirlas por separado
    sería el defecto de las dos declaraciones que acaban divergiendo."""

    def calcular_una(muestra: Muestra, umbral: float | None) -> float | Indefinida:
        matriz = matriz_de_confusion(muestra, umbral=umbral)
        den = denominador(matriz)
        if den == 0:
            return Indefinida(motivo("division_por_cero_en_metrica", campo=metric_id,
                                     valor=nombre_del_denominador))
        return numerador(matriz) / den

    return calcular_una


def _accuracy(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    matriz = matriz_de_confusion(muestra, umbral=umbral)
    aciertos = sum(matriz.counts[c][c] for c in matriz.classes)
    return aciertos / muestra.n


def _macro_f1(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    """Macro-F1 **con el redondeo del evaluador histórico**.

    `dense_evaluator._precision_recall_f1` redondea cada F1 a seis decimales y
    luego redondea la media. Reproducirlo no es cosmética: sin él, este registro
    daría un macro-F1 distinto del que el producto lleva publicando, y el campo
    histórico habría cambiado de valor sin que nadie lo dijera. Y donde el
    histórico pone 0,0 por un denominador vacío, aquí se pone 0,0 TAMBIÉN: es
    una decisión heredada, no una indefinición nueva.
    """
    matriz = matriz_de_confusion(muestra, umbral=umbral)
    clases = matriz.classes
    f1s = []
    for clase in clases:
        tp = matriz.counts[clase][clase]
        fp = sum(matriz.counts[real][clase] for real in clases if real != clase)
        fn = sum(matriz.counts[clase][pred] for pred in clases if pred != clase)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 6))
    return round(sum(f1s) / len(f1s), 6) if f1s else 0.0


def _errores(muestra: Muestra) -> tuple[list[float], list[float], list[float]]:
    reales = [float(y) for y in muestra.y_true]
    predichos = [float(p) for p in (muestra.predictions or ())]
    return reales, predichos, list(muestra.pesos)


def _mae(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    reales, predichos, pesos = _errores(muestra)
    peso_total = sum(pesos)
    if peso_total == 0.0:
        return Indefinida(motivo("division_por_cero_en_metrica", campo="mae",
                                 valor="la suma de los pesos"))
    return sum(w * abs(p - y) for w, p, y in zip(pesos, predichos, reales)) / peso_total


def _rmse(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    reales, predichos, pesos = _errores(muestra)
    peso_total = sum(pesos)
    if peso_total == 0.0:
        return Indefinida(motivo("division_por_cero_en_metrica", campo="rmse",
                                 valor="la suma de los pesos"))
    return math.sqrt(
        sum(w * (p - y) ** 2 for w, p, y in zip(pesos, predichos, reales)) / peso_total)


def _r2(muestra: Muestra, umbral: float | None) -> float | Indefinida:
    """R². **Aquí está la única divergencia deliberada con el histórico.**

    `dense_evaluator.compute_r2` devuelve `0.0` cuando el objetivo no tiene
    varianza (`ss_tot == 0`). Un 0,0 ahí se lee como «el modelo no explica nada»
    cuando lo que pasa es que no hay nada que explicar, y esa es exactamente la
    regla que este contrato prohíbe: un valor ausente no es un cero. Sobre datos
    con varianza —los de cualquier fixture real— los dos números coinciden al
    último bit; la divergencia solo aparece en el caso degenerado y está probada.
    """
    reales, predichos, pesos = _errores(muestra)
    peso_total = sum(pesos)
    if peso_total == 0.0:
        return Indefinida(motivo("division_por_cero_en_metrica", campo="r2",
                                 valor="la suma de los pesos"))
    media = sum(w * y for w, y in zip(pesos, reales)) / peso_total
    ss_tot = sum(w * (y - media) ** 2 for w, y in zip(pesos, reales))
    ss_res = sum(w * (p - y) ** 2 for w, p, y in zip(pesos, predichos, reales))
    if ss_tot == 0.0:
        return Indefinida(motivo("division_por_cero_en_metrica", campo="r2",
                                 valor="la varianza del objetivo"))
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# El catálogo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricaRegistrada:
    """Una métrica del registro: su identidad (`MetricSpec`) y su fórmula.

    `alternativas` es la única cosa que este módulo añade encima del 104-C0, y
    tiene un motivo escrito: `MetricSpec.requires` enumera requisitos sueltos, y
    AUROC no necesita `scores` NI `probabilities`, necesita **una de las dos**
    —una probabilidad es una puntuación ordenada—. Sin los grupos,
    `apto_para_recomputar` daría un falso «no vale» a un registro que trae la
    distribución completa. Los grupos ENSANCHAN lo que el esquema ya dice; no
    contradicen ninguna negativa suya.
    """

    spec: MetricSpec
    tareas: tuple[str, ...]
    formula: Callable[[Muestra, float | None], "float | Indefinida"]
    alternativas: tuple[tuple[str, ...], ...] = ()
    admite_pesos: bool = False
    #: Si sale de la matriz de confusión: entonces necesita saber DÓNDE se corta,
    #: y sobre una puntuación sin normalizar eso no lo decide un 0,5 por omisión.
    necesita_umbral: bool = False
    #: Cómo se llama en pantalla, `(es, en)`. Presentación: por eso vive aquí y
    #: no en `spec`, que entra en la huella del catálogo.
    nombre: tuple[str, str] = ("", "")

    @property
    def metric_id(self) -> str:
        return self.spec.metric_id

    def grupo_de(self, requisito: str) -> tuple[str, ...]:
        for grupo in self.alternativas:
            if requisito in grupo:
                return grupo
        return (requisito,)


_BINARIA = ("binary_classification",)
_CLASIFICACION = TAREAS_DE_CLASIFICACION
_REGRESION = ("regression",)
_PUNTUACION = (("scores", "probabilities"),)
_CLASE_PREDICHA = (("labels", "probabilities", "scores"),)
#: Las cuatro de la matriz de confusión (sensibilidad, especificidad, VPP y VPN)
#: usan el MISMO grupo que `accuracy` desde el 115-C2, y piden `labels` como ella:
#: salen de los recuentos al umbral, y `matriz_de_confusion` ya da prioridad a la
#: etiqueta declarada, así que una decisión ya tomada —la del proceso actual— las
#: sostiene. Pedían `_PUNTUACION`, y eso dejaba comparar con el proceso actual
#: solo en `accuracy`.


#: EL NOMBRE DE CADA MÉTRICA, `(es, en)`. `_metrica` lo exige: una métrica sin
#: nombre no llega a registrarse, y así no puede salir en crudo en una pantalla.
_NOMBRES: dict[str, tuple[str, str]] = {
    "auroc": ("AUROC", "AUROC"),
    "average_precision": ("Precisión media (AP)", "Average precision (AP)"),
    "pr_auc_trapezoidal": ("Área bajo la curva PR (trapecios)", "PR AUC (trapezoidal)"),
    "log_loss": ("Pérdida logarítmica", "Log loss"),
    "brier_score": ("Puntuación de Brier", "Brier score"),
    "calibration_in_the_large": ("Calibración global", "Calibration-in-the-large"),
    "sensitivity": ("Sensibilidad", "Sensitivity"),
    "specificity": ("Especificidad", "Specificity"),
    "ppv": ("Valor predictivo positivo (VPP)", "Positive predictive value (PPV)"),
    "npv": ("Valor predictivo negativo (VPN)", "Negative predictive value (NPV)"),
    "accuracy": ("Exactitud", "Accuracy"),
    "macro_f1": ("F1 macro", "Macro F1"),
    "mae": ("MAE", "MAE"),
    "rmse": ("RMSE", "RMSE"),
    "r2": ("R²", "R²"),
}


def _metrica(metric_id: str, *, tareas: tuple[str, ...], formula, requires: tuple[str, ...],
             direction: str | None = None, ideal_value: float | None = None,
             normalization: str | None = None, weights: str | None = None,
             alternativas: tuple[tuple[str, ...], ...] = (),
             positive_label: str | None = None,
             necesita_umbral: bool = False) -> MetricaRegistrada:
    spec = MetricSpec(metric_id=metric_id, formula_version="1.0.0",
                      estimand="fixed_model_on_population", requires=requires,
                      direction=direction, ideal_value=ideal_value,
                      normalization=normalization, weights=weights,
                      positive_label=positive_label)
    return MetricaRegistrada(spec=spec, tareas=tareas, formula=formula,
                             alternativas=alternativas, admite_pesos=weights is not None,
                             necesita_umbral=necesita_umbral, nombre=_NOMBRES[metric_id])


#: La clase positiva NO se declara en el catálogo: `requires` ya dice que la
#: trae la muestra, y la ficha genérica —«qué es el AUROC»— no conoce ninguna.
#: Esto era un marcador de texto (`"<positive_label of the sample>"`) para
#: cumplir una regla del 104-C0 que se revisó el 2026-09-05: un valor fabricado
#: en un campo tipado como etiqueta de clase. Quien exige la clase de verdad es
#: `Muestra`, que sin ella no se construye.
CLASE_POSITIVA_DE_LA_MUESTRA = None

_CATALOGO: tuple[MetricaRegistrada, ...] = (
    _metrica("auroc", tareas=_BINARIA, formula=_auroc,
             requires=("y_true", "classes", "positive_label", "scores"),
             direction="higher_is_better", alternativas=_PUNTUACION,
             weights="sample_weight", positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="area bajo la curva ROC por trapecios sobre los umbrales "
                           "distintos; los empates comparten punto (equivale a "
                           "Mann-Whitney con 0,5 en los empates)"),
    _metrica("average_precision", tareas=_BINARIA, formula=_average_precision,
             requires=("y_true", "classes", "positive_label", "scores"),
             direction="higher_is_better", alternativas=_PUNTUACION,
             weights="sample_weight", positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="suma de escalones sum((R_k - R_k-1) * P_k); NO es el "
                           "area trapezoidal de la curva PR"),
    _metrica("pr_auc_trapezoidal", tareas=_BINARIA, formula=_pr_auc_trapezoidal,
             requires=("y_true", "classes", "positive_label", "scores"),
             direction="higher_is_better", alternativas=_PUNTUACION,
             weights="sample_weight", positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="area PR uniendo los puntos con rectas, ancladas en "
                           "(recall 0, precision 1); ID distinto del de AP a "
                           "proposito"),
    _metrica("log_loss", tareas=_CLASIFICACION, formula=_log_loss,
             requires=("y_true", "classes", "probabilities"),
             direction="lower_is_better", weights="sample_weight",
             normalization=f"media de -ln p[y]; clipping a "
                           f"[{CLIP_LOG_LOSS!r}, 1-{CLIP_LOG_LOSS!r}]; base e"),
    _metrica("brier_score", tareas=_BINARIA, formula=_brier,
             requires=("y_true", "classes", "positive_label", "probabilities"),
             direction="lower_is_better", weights="sample_weight",
             positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="Brier binario clasico: media de (p - 1[y=positiva])^2, "
                           "escala [0,1]; el multicategoria vale el doble y NO se "
                           "publica aqui"),
    _metrica("calibration_in_the_large", tareas=_BINARIA, formula=_calibracion_en_grande,
             requires=("y_true", "classes", "positive_label", "probabilities"),
             ideal_value=0.0, weights="sample_weight",
             positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="probabilidad media predicha menos prevalencia observada; "
                           "valor ideal 0 y desviarse a cualquiera de los dos lados "
                           "es peor; no es la pendiente ni el intercepto del 105-C3"),
    _metrica("sensitivity", tareas=_BINARIA, necesita_umbral=True,
             formula=_desde_la_matriz("sensitivity", lambda m: m.tp,
                                      lambda m: m.tp + m.fn, "el numero de positivos reales"),
             requires=("y_true", "classes", "positive_label", "labels"),
             direction="higher_is_better", alternativas=_CLASE_PREDICHA,
             positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="al umbral, sobre recuentos enteros: TP / (TP + FN)"),
    _metrica("specificity", tareas=_BINARIA, necesita_umbral=True,
             formula=_desde_la_matriz("specificity", lambda m: m.tn,
                                      lambda m: m.tn + m.fp, "el numero de negativos reales"),
             requires=("y_true", "classes", "positive_label", "labels"),
             direction="higher_is_better", alternativas=_CLASE_PREDICHA,
             positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="al umbral, sobre recuentos enteros: TN / (TN + FP)"),
    _metrica("ppv", tareas=_BINARIA, necesita_umbral=True,
             formula=_desde_la_matriz("ppv", lambda m: m.tp, lambda m: m.tp + m.fp,
                                      "el numero de predichos positivos"),
             requires=("y_true", "classes", "positive_label", "labels"),
             direction="higher_is_better", alternativas=_CLASE_PREDICHA,
             positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="al umbral, sobre recuentos enteros: TP / (TP + FP)"),
    _metrica("npv", tareas=_BINARIA, necesita_umbral=True,
             formula=_desde_la_matriz("npv", lambda m: m.tn, lambda m: m.tn + m.fn,
                                      "el numero de predichos negativos"),
             requires=("y_true", "classes", "positive_label", "labels"),
             direction="higher_is_better", alternativas=_CLASE_PREDICHA,
             positive_label=CLASE_POSITIVA_DE_LA_MUESTRA,
             normalization="al umbral, sobre recuentos enteros: TN / (TN + FN)"),
    _metrica("accuracy", tareas=_CLASIFICACION, formula=_accuracy,
             necesita_umbral=True,
             requires=("y_true", "classes", "labels"),
             direction="higher_is_better", alternativas=_CLASE_PREDICHA,
             normalization="aciertos / n sobre la matriz de confusion; en binaria la "
                           "clase predicha sale del umbral, y el umbral se publica "
                           "en el informe"),
    _metrica("macro_f1", tareas=_CLASIFICACION, formula=_macro_f1,
             necesita_umbral=True,
             requires=("y_true", "classes", "labels"),
             direction="higher_is_better", alternativas=_CLASE_PREDICHA,
             normalization="media de los F1 por clase, cada uno redondeado a 6 "
                           "decimales y la media tambien (compatibilidad exacta con "
                           "dense_evaluator)"),
    _metrica("mae", tareas=_REGRESION, formula=_mae,
             requires=("y_true", "labels"), direction="lower_is_better",
             weights="sample_weight",
             normalization="media de |prediccion - real|, en la unidad del objetivo"),
    _metrica("rmse", tareas=_REGRESION, formula=_rmse,
             requires=("y_true", "labels"), direction="lower_is_better",
             weights="sample_weight",
             normalization="raiz de la media de (prediccion - real)^2"),
    _metrica("r2", tareas=_REGRESION, formula=_r2,
             requires=("y_true", "labels"), direction="higher_is_better",
             weights="sample_weight",
             normalization="1 - SS_res / SS_tot; indefinida si SS_tot vale 0, donde "
                           "el evaluador historico devuelve 0.0"),
)

#: El registro, por ID. Es el único sitio donde vive cada fórmula.
REGISTRO: dict[str, MetricaRegistrada] = {m.metric_id: m for m in _CATALOGO}


def catalogo() -> tuple[MetricSpec, ...]:
    """Los `MetricSpec` del registro, en orden estable."""
    return tuple(m.spec for m in _CATALOGO)


def especificacion(metric_id: str) -> MetricSpec:
    return _buscar(metric_id).spec


def nombre_de_la_metrica(metric_id: str) -> dict[str, str]:
    """Cómo se llama en pantalla, en los dos idiomas. Mismos rechazos que
    `especificacion`: una métrica desconocida o aplazada no se nombra."""
    es, en = _buscar(metric_id).nombre
    return {"es": es, "en": en}


def aplazamiento(metric_id: str) -> dict[str, str] | None:
    """El motivo bilingüe de que una métrica esté diferida, o `None`."""
    clave = METRICAS_DIFERIDAS.get(metric_id)
    return motivo(clave) if clave else None


def digest_del_catalogo() -> str:
    """La huella del catálogo ENTERO: fórmulas, versiones, clipping y umbral.

    Es lo que permite que `verify` diga «este informe se midió con otro
    catálogo» en vez de comparar números medidos con reglas distintas.
    """
    return digest_canonico({
        "report_version": INFORME_VERSION,
        "confusion_formula_version": VERSION_DE_LA_MATRIZ,
        "log_loss_clip": CLIP_LOG_LOSS,
        "default_threshold": UMBRAL_POR_DEFECTO,
        "deferred": {k: motivo(v) for k, v in sorted(METRICAS_DIFERIDAS.items())},
        "metrics": [spec.a_json() for spec in catalogo()],
    })


def _buscar(metric_id: str) -> MetricaRegistrada:
    if metric_id in REGISTRO:
        return REGISTRO[metric_id]
    if metric_id in METRICAS_DIFERIDAS:
        raise MetricaAplazada(METRICAS_DIFERIDAS[metric_id])
    raise MetricaDesconocida("metrica_desconocida", valor=repr(metric_id))


# ---------------------------------------------------------------------------
# Aptitud: qué hace falta, y qué falta
# ---------------------------------------------------------------------------

def _lo_que_trae(muestra: Muestra) -> dict[str, bool]:
    return {
        "y_true": True,
        "classes": muestra.classes is not None,
        "probabilities": muestra.probabilities is not None,
        "scores": muestra.scores is not None,
        "labels": muestra.predictions is not None,
        "positive_label": muestra.positive_label is not None,
        "weights": muestra.weights is not None,
        "split_role": True,
        "resampling_unit": muestra.units is not None,
    }


def _rechazo_estructural(registrada: MetricaRegistrada, muestra: Muestra,
                         umbral: float | None = None) -> EntradaNoMedible | None:
    """Lo que hace IMPOSIBLE medir esa métrica sobre esa muestra, con su motivo.

    Es la mitad «error» de la frontera: no devuelve indefinición porque no es
    que los datos salgan degenerados, es que se está pidiendo otra cosa.
    """
    if muestra.task not in registrada.tareas:
        return EntradaNoMedible("metrica_de_otra_tarea", campo=registrada.metric_id,
                                opciones=list(registrada.tareas), valor=muestra.task)
    if muestra.weights is not None and not registrada.admite_pesos:
        return EntradaNoMedible("pesos_no_admitidos", campo=registrada.metric_id)
    if (registrada.necesita_umbral and umbral is None
            and muestra.predictions is None and muestra.probabilities is None):
        return EntradaNoMedible("umbral_por_omision_en_puntuacion")

    trae = _lo_que_trae(muestra)
    for requisito in registrada.spec.requires:
        grupo = registrada.grupo_de(requisito)
        if any(trae[x] for x in grupo):
            continue
        return _por_que_falta(registrada, muestra, grupo)
    return None


def _por_que_falta(registrada: MetricaRegistrada, muestra: Muestra,
                   grupo: tuple[str, ...]) -> EntradaNoMedible:
    metric_id = registrada.metric_id
    faltantes = set(grupo)
    if faltantes == {"positive_label"}:
        return EntradaNoMedible("binaria_sin_clase_positiva")
    if faltantes == {"classes"}:
        return EntradaNoMedible("clasificacion_sin_clases")
    if faltantes == {"probabilities"}:
        # El caso del criterio 3: unos logits ordenan y no son probabilidades.
        if muestra.scores is not None:
            return EntradaNoMedible("no_son_probabilidades", campo=metric_id)
        if muestra.predictions is not None:
            return EntradaNoMedible("etiquetas_duras_no_ordenan", campo=metric_id)
    if faltantes == {"scores", "probabilities"} and muestra.predictions is not None:
        return EntradaNoMedible("etiquetas_duras_no_ordenan", campo=metric_id)
    return EntradaNoMedible("muestra_sin_salida")


def _registro_trae(registro: PredictionRecord, requisito: str) -> bool:
    return {
        "y_true": registro.y_true is not None,
        "classes": registro.classes is not None,
        "probabilities": registro.probabilities is not None,
        "scores": registro.scores is not None,
        "labels": registro.label is not None,
        "split_role": registro.role is not None,
        "resampling_unit": registro.resampling_unit is not None,
        "weights": registro.weight is not None,
    }.get(requisito, False)


def aptitud(metric_id: str, registro: PredictionRecord, *,
            positive_label: str | None = None) -> Aptitud:
    """¿Sirve ESE registro de predicción para recomputar ESA métrica?

    Delega en `PredictionRecord.apto_para_recomputar` —la fórmula de la aptitud
    es del 104-C0 y no se duplica— y solo ensancha con los grupos alternativos:
    un registro que trae la distribución completa y no `scores` SÍ vale para
    AUROC, porque una probabilidad es una puntuación ordenada.

    `positive_label` hace falta porque la ficha del catálogo es GENÉRICA: lleva
    el marcador `CLASE_POSITIVA_DE_LA_MUESTRA` en vez de una clase concreta, así
    que sin decir cuál es la positiva la respuesta es —correctamente— que falta.
    Elegirla por orden alfabético sería justo lo que el 104-C0 prohíbe.
    """
    registrada = _buscar(metric_id)
    spec = registrada.spec
    if positive_label is not None and spec.positive_label == CLASE_POSITIVA_DE_LA_MUESTRA:
        spec = replace(spec, positive_label=positive_label)
    base = registro.apto_para_recomputar(spec)
    if base.apto:
        return base
    faltan = [f for f in base.faltan
              if not any(_registro_trae(registro, x) for x in registrada.grupo_de(f))]
    if not faltan:
        return Aptitud(apto=True, faltan=(), motivo=None)
    return Aptitud(apto=False, faltan=tuple(faltan),
                   motivo=motivo("no_vale_para_recomputar", campo=metric_id,
                                 valor=", ".join(faltan)))


def metricas_aplicables(muestra: Muestra, *,
                        umbral: float | None = None) -> tuple[str, ...]:
    """Los IDs que ESTA muestra puede sostener, en el orden del catálogo.

    El umbral entra en la cuenta: con puntuaciones sin normalizar y sin umbral
    declarado, las métricas que salen de la matriz de confusión NO son
    aplicables — decir que lo son y reventar al calcularlas sería peor.
    """
    return tuple(m.metric_id for m in _CATALOGO
                 if _rechazo_estructural(m, muestra, umbral) is None)


# ---------------------------------------------------------------------------
# Calcular
# ---------------------------------------------------------------------------

def calcular(metric_id: str, muestra: Muestra, *,
             umbral: float | None = None) -> ValorDeMetrica:
    """Un número, o la razón de que no lo haya. Nunca un cero inventado.

    Levanta excepción si la muestra no puede sostener la métrica (cableado) y
    devuelve `ValorDeMetrica` con `undefined_reason` si son los datos los que no
    dan para más (una sola clase, cero filas, denominador vacío).
    """
    registrada = _buscar(metric_id)
    rechazo = _rechazo_estructural(registrada, muestra, umbral)
    if rechazo is not None:
        raise rechazo
    if muestra.n == 0:
        resultado: float | Indefinida = Indefinida(
            motivo("sin_observaciones", campo=metric_id))
    else:
        resultado = registrada.formula(muestra, umbral)
    comun = {"metric_id": metric_id,
             "formula_version": registrada.spec.formula_version,
             "n_observations": muestra.n, "n_units": muestra.n_unidades}
    if isinstance(resultado, Indefinida):
        return ValorDeMetrica(undefined_reason=resultado.motivo, **comun)
    return ValorDeMetrica(value=float(resultado), **comun)


# ---------------------------------------------------------------------------
# Ordenar: dirección o valor ideal, nunca «mayor mejor» para todo
# ---------------------------------------------------------------------------

def direccion_de(metric_id: str) -> str | None:
    """`higher_is_better`, `lower_is_better` o `None` si la métrica tiene ideal."""
    return _buscar(metric_id).spec.direction


def distancia_al_ideal(metric_id: str, valor: float) -> float:
    """Cuánto se aleja del valor —o del rango— ideal. Solo para las que lo tienen."""
    spec = _buscar(metric_id).spec
    if spec.ideal_value is not None:
        return abs(valor - spec.ideal_value)
    if spec.ideal_range is not None:
        bajo, alto = spec.ideal_range
        return 0.0 if bajo <= valor <= alto else min(abs(valor - bajo), abs(valor - alto))
    raise EntradaNoMedible("metrica_de_otra_tarea", campo=metric_id,
                           opciones="ideal_value/ideal_range", valor="direction")


def es_mejor(metric_id: str, a: float | None, b: float | None) -> bool:
    """Si `a` es ESTRICTAMENTE mejor que `b` para esa métrica.

    Y aquí está el invariante 4 hecho código: si la métrica no tiene dirección,
    se ordena por distancia al ideal. Un intercepto de +0,2 y otro de -0,2 son
    igual de malos; «mayor mejor» coronaría al primero.

    Un valor indefinido no participa: una métrica que no se pudo calcular no es
    la peor, es que no se sabe.
    """
    if a is None or b is None:
        raise EntradaNoMedible("comparar_sin_valor", campo=metric_id)
    direccion = direccion_de(metric_id)
    if direccion == "higher_is_better":
        return a > b
    if direccion == "lower_is_better":
        return a < b
    return distancia_al_ideal(metric_id, a) < distancia_al_ideal(metric_id, b)


def ordenar_por(metric_id: str, valores: Iterable[float]) -> list[float]:
    """Los valores de mejor a peor según ESA métrica."""
    direccion = direccion_de(metric_id)
    if direccion == "higher_is_better":
        return sorted(valores, reverse=True)
    if direccion == "lower_is_better":
        return sorted(valores)
    return sorted(valores, key=lambda v: distancia_al_ideal(metric_id, v))


# ---------------------------------------------------------------------------
# El informe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InformeMetrico:
    """El informe ampliado: los números Y las reglas con las que se midieron.

    Documento NUEVO, con `report_version` y digest propios. Los campos históricos
    de `dense_evaluator` siguen donde están y valen lo mismo; este informe no los
    sustituye, los enmarca — y declara lo que a ellos les falta: el orden de
    clases, la clase positiva, el umbral, la regla de decisión, el clipping, si
    hubo pesos y con qué catálogo se midió.

    `not_applicable` no es decoración: una métrica que no se midió aparece
    DICIENDO por qué, en vez de faltar en silencio.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.metric_report"

    task: str
    n_observations: int
    catalog_digest: str
    metrics: tuple[ValorDeMetrica, ...] = ()
    classes: tuple[str, ...] | None = None
    positive_label: str | None = None
    threshold: float | None = None
    decision_rule: str | None = None
    decision_scale: str | None = None
    score_rule: str | None = None
    log_loss_clip: float = CLIP_LOG_LOSS
    weighted: bool = False
    n_units: int | None = None
    confusion_matrix: MatrizDeConfusion | None = None
    not_applicable: tuple[tuple[str, dict[str, str]], ...] = ()
    report_version: str = INFORME_VERSION

    def valor(self, metric_id: str) -> ValorDeMetrica | None:
        for m in self.metrics:
            if m.metric_id == metric_id:
                return m
        return None

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "task": self.task,
            "classes": list(self.classes) if self.classes else None,
            "positive_label": self.positive_label,
            "threshold": self.threshold,
            "decision_rule": self.decision_rule,
            "decision_scale": self.decision_scale,
            "score_rule": self.score_rule,
            "log_loss_clip": self.log_loss_clip,
            "weighted": self.weighted,
            "n_observations": self.n_observations,
            "n_units": self.n_units,
            "catalog_digest": self.catalog_digest,
            "metrics": [m.a_json() for m in self.metrics],
            "confusion_matrix": (self.confusion_matrix.a_json()
                                 if self.confusion_matrix else None),
            "not_applicable": [{"metric_id": mid, "reason": dict(razon)}
                               for mid, razon in self.not_applicable],
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), None)

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def evaluar(muestra: Muestra, metricas: Sequence[str] | None = None, *,
            umbral: float | None = None, incluir_matriz: bool = True) -> InformeMetrico:
    """Mide lo que se pida —o todo lo que la muestra sostenga— y lo enmarca.

    Con `metricas` EXPLÍCITAS, una que la muestra no pueda sostener levanta
    excepción: quien la nombró se ha equivocado y tiene que enterarse. En modo
    automático (`metricas=None`) se miden las aplicables y las demás quedan
    listadas en `not_applicable` con su motivo, que no es lo mismo que faltar.
    """
    explicitas = metricas is not None
    # En automático se recorre el catálogo DE ESTA TAREA entero, no solo lo
    # aplicable: lo que no se pueda medir tiene que aparecer diciendo por qué.
    # Las métricas de otra tarea sí se omiten — que una binaria no tenga RMSE no
    # es información, es ruido.
    ids = tuple(metricas) if explicitas else tuple(
        m.metric_id for m in _CATALOGO if muestra.task in m.tareas)

    valores: list[ValorDeMetrica] = []
    no_aplicables: list[tuple[str, dict[str, str]]] = []
    for metric_id in ids:
        registrada = _buscar(metric_id)
        rechazo = _rechazo_estructural(registrada, muestra, umbral)
        if rechazo is not None:
            if explicitas:
                raise rechazo
            no_aplicables.append((metric_id, rechazo.bilingue))
            continue
        valores.append(calcular(metric_id, muestra, umbral=umbral))

    matriz = None
    if incluir_matriz and muestra.n and muestra.task in TAREAS_DE_CLASIFICACION:
        try:
            matriz = matriz_de_confusion(muestra, umbral=umbral)
        except ErrorDeEstudio as rechazo_de_la_matriz:
            # No se traga: se DECLARA. La matriz no es un `MetricSpec`, así que
            # su motivo se publica en la misma lista que el de las métricas.
            no_aplicables.append(("confusion_matrix", rechazo_de_la_matriz.bilingue))

    return InformeMetrico(
        task=muestra.task, n_observations=muestra.n, n_units=muestra.n_unidades,
        catalog_digest=digest_del_catalogo(), metrics=tuple(valores),
        classes=muestra.classes, positive_label=muestra.positive_label,
        threshold=matriz.threshold if matriz else None,
        decision_rule=matriz.decision_rule if matriz else None,
        decision_scale=muestra.escala_de_decision, score_rule=muestra.score_rule,
        weighted=muestra.weights is not None, confusion_matrix=matriz,
        not_applicable=tuple(no_aplicables))
