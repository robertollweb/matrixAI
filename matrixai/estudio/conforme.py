# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C4 — predicción conforme por partición: un conjunto de clases o un
intervalo por fila, con una cobertura prometida y MEDIDA.

**Qué promete, y qué no** (invariante 2 del 115, y va dentro de cada JSON):
- en PROMEDIO sobre las filas, el conjunto (o el intervalo) contiene la
  respuesta verdadera al menos una proporción `1 - alfa` de las veces;
- solo si las filas nuevas se parecen a las de calibración (intercambiables);
- NO fila a fila, y NO resiste la deriva.

**Y el tamaño va siempre al lado de la cobertura.** Un conjunto `{sí, no}` en
todas las filas cubre el 100 % y no dice nada; un intervalo de ±infinito,
igual. Por eso la medición devuelve el tamaño medio del conjunto (o la anchura
media del intervalo) y cuántos conjuntos salieron vacíos, en la misma respuesta.

**Cómo** (el método «LAC» de la literatura, el más sencillo que da la garantía):
- clasificación: la puntuación de una fila es `1 − p(clase verdadera)`, con la
  distribución COMPLETA de probabilidades (una etiqueta dura no ordena nada);
- regresión: la puntuación es `|y − ŷ|`;
- el cuantil es la `k`-ésima puntuación de calibración, con
  `k = ⌈(n + 1)(1 − alfa)⌉`. El `+ 1` es lo que convierte «casi» en garantía
  con muestras finitas. Si `k > n`, con esas filas **no se puede** garantizar
  `1 − alfa`, y se dice con cuántas haría falta — nunca se recorta el cuantil
  para que salga algo.
- el conjunto de una fila son las clases con `1 − p ≤ cuantil`; puede salir
  VACÍO (ninguna clase es suficientemente probable), y eso se cuenta, no se
  rellena.

**Estado.** Fichero nuevo, sin cablear: nace con la pasada de Fase 0 del 113
viva, y `matrixai/estudio/__init__.py` no se toca con ella en marcha. Lo usarán
el estudio (calibrando en la partición de calibración) y el paquete (C6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from matrixai.estudio.metricas import Muestra
from matrixai.training.diagnostico import intervalo_wilson

__all__ = ["CalibracionConforme", "ConformeNoAplicable", "CoberturaMedida", "GARANTIA",
           "calibrar", "conjunto_de_clases", "intervalo_de_prediccion", "medir_cobertura"]

GARANTIA = {
    "es": "Cobertura en promedio sobre las filas, no fila a fila, y solo si los casos nuevos se "
          "parecen a los de calibración: no resiste la deriva.",
    "en": "Coverage on average over rows, not row by row, and only if new cases resemble the "
          "calibration ones: it does not withstand drift.",
}

_CLASIFICACION = ("binary_classification", "multiclass_classification")


class ConformeNoAplicable(ValueError):
    """La muestra no sostiene este método. Lleva su motivo en los dos idiomas:
    lo que redacta el núcleo se traduce en el núcleo."""

    def __init__(self, es: str, en: str) -> None:
        super().__init__(es)
        self.motivo = {"es": es, "en": en}


def _exigir_muestra(muestra: Muestra) -> None:
    if muestra.weights is not None:
        raise ConformeNoAplicable(
            "la muestra lleva pesos, y la garantía conforme de este módulo es para filas sin peso",
            "the sample carries weights, and this module's conformal guarantee is for unweighted rows")
    if muestra.task in _CLASIFICACION:
        if muestra.probabilities is None or muestra.classes is None:
            raise ConformeNoAplicable(
                "hace falta la distribución completa de probabilidades por fila: una etiqueta "
                "dura no ordena nada dentro de su clase",
                "the full probability distribution per row is needed: a hard label ranks "
                "nothing within its own class")
    elif muestra.task == "regression":
        if muestra.predictions is None:
            raise ConformeNoAplicable("hace falta el valor predicho por fila",
                                      "the predicted value per row is needed")
    else:
        raise ConformeNoAplicable(f"tarea {muestra.task!r} no admitida",
                                  f"task {muestra.task!r} not supported")


def _puntuaciones(muestra: Muestra) -> list[float]:
    if muestra.task == "regression":
        return [abs(float(y) - float(p)) for y, p in zip(muestra.y_true, muestra.predictions)]
    clases = list(muestra.classes)
    return [1.0 - float(fila[clases.index(y)])
            for y, fila in zip(muestra.y_true, muestra.probabilities)]


def _filas_necesarias(alfa: float) -> int:
    """El menor `n` con `⌈(n + 1)(1 − alfa)⌉ ≤ n`."""
    n = 1
    while math.ceil((n + 1) * (1 - alfa) - 1e-12) > n:
        n += 1
    return n


@dataclass(frozen=True)
class CalibracionConforme:
    tarea: str
    alfa: float
    n_calibracion: int
    #: `None` cuando con `n_calibracion` filas no se puede garantizar `1 − alfa`.
    cuantil: float | None
    clases: tuple[str, ...] | None = None
    motivo: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alfa < 1.0:
            raise ValueError(f"alfa tiene que estar entre 0 y 1, no {self.alfa!r}")
        if (self.cuantil is None) == (self.motivo is None):
            raise ValueError("un cuantil O su motivo, exactamente uno")

    @property
    def cobertura_nominal(self) -> float:
        return 1.0 - self.alfa

    def a_json(self) -> dict[str, Any]:
        return {"tarea": self.tarea, "alfa": self.alfa, "n_calibracion": self.n_calibracion,
                "cuantil": self.cuantil, "clases": list(self.clases) if self.clases else None,
                "motivo": self.motivo, "puntuacion": ("residuo_absoluto" if self.tarea == "regression"
                                                      else "1_menos_p_de_la_verdadera"),
                "garantia": GARANTIA}

    @classmethod
    def desde_json(cls, datos: Mapping[str, Any]) -> "CalibracionConforme":
        return cls(tarea=datos["tarea"], alfa=float(datos["alfa"]),
                   n_calibracion=int(datos["n_calibracion"]),
                   cuantil=(None if datos.get("cuantil") is None else float(datos["cuantil"])),
                   clases=tuple(datos["clases"]) if datos.get("clases") else None,
                   motivo=datos.get("motivo"))


def calibrar(muestra_calibracion: Muestra, *, alfa: float) -> CalibracionConforme:
    """El cuantil, SOLO con la partición de calibración (nunca con test)."""
    _exigir_muestra(muestra_calibracion)
    clases = tuple(muestra_calibracion.classes) if muestra_calibracion.task != "regression" else None
    puntuaciones = sorted(_puntuaciones(muestra_calibracion))
    n = len(puntuaciones)
    k = math.ceil((n + 1) * (1 - alfa) - 1e-12)
    if n == 0 or k > n:
        necesarias = _filas_necesarias(alfa)
        return CalibracionConforme(
            tarea=muestra_calibracion.task, alfa=alfa, n_calibracion=n, cuantil=None, clases=clases,
            motivo={"es": f"con {n} filas de calibración no se puede garantizar una cobertura del "
                          f"{1 - alfa:.0%}: harían falta al menos {necesarias}",
                    "en": f"with {n} calibration rows a {1 - alfa:.0%} coverage cannot be "
                          f"guaranteed: at least {necesarias} would be needed"})
    return CalibracionConforme(tarea=muestra_calibracion.task, alfa=alfa, n_calibracion=n,
                               cuantil=puntuaciones[k - 1], clases=clases)


def _exigir_cuantil(calibracion: CalibracionConforme) -> float:
    if calibracion.cuantil is None:
        raise ConformeNoAplicable(calibracion.motivo["es"], calibracion.motivo["en"])
    return calibracion.cuantil


def conjunto_de_clases(probabilidades: Sequence[float],
                       calibracion: CalibracionConforme) -> tuple[str, ...]:
    """Las clases de UNA fila con `1 − p ≤ cuantil`, en el orden de `clases`.
    Puede salir vacío, y es una respuesta: ninguna clase llega."""
    cuantil = _exigir_cuantil(calibracion)
    return tuple(c for c, p in zip(calibracion.clases, probabilidades)
                 if 1.0 - float(p) <= cuantil)


def intervalo_de_prediccion(prediccion: float,
                            calibracion: CalibracionConforme) -> tuple[float, float]:
    cuantil = _exigir_cuantil(calibracion)
    return float(prediccion) - cuantil, float(prediccion) + cuantil


@dataclass(frozen=True)
class CoberturaMedida:
    """Lo que pasó en test, con el tamaño al lado de la cobertura."""

    cobertura_nominal: float
    n: int
    cubiertas: int
    #: Intervalo de Wilson al 95 % de la cobertura OBSERVADA.
    intervalo_95: tuple[float, float] | None
    #: Tamaño medio del conjunto (clasificación) o anchura media (regresión).
    tamano_medio: float | None
    vacios: int = 0

    @property
    def cobertura_observada(self) -> float | None:
        return self.cubiertas / self.n if self.n else None

    @property
    def compatible_con_la_nominal(self) -> bool | None:
        """¿La nominal cae dentro del intervalo de la observada? `None` sin filas."""
        if self.intervalo_95 is None:
            return None
        bajo, alto = self.intervalo_95
        return bajo <= self.cobertura_nominal <= alto

    def a_json(self) -> dict[str, Any]:
        return {"cobertura_nominal": self.cobertura_nominal, "n": self.n,
                "cubiertas": self.cubiertas, "cobertura_observada": self.cobertura_observada,
                "intervalo_95": list(self.intervalo_95) if self.intervalo_95 else None,
                "compatible_con_la_nominal": self.compatible_con_la_nominal,
                "tamano_medio": self.tamano_medio, "vacios": self.vacios, "garantia": GARANTIA}


def medir_cobertura(muestra_test: Muestra, calibracion: CalibracionConforme) -> CoberturaMedida:
    """La cobertura en filas que la calibración NO vio, y cuánto midió cada
    conjunto o intervalo para conseguirla."""
    _exigir_muestra(muestra_test)
    if muestra_test.task != calibracion.tarea:
        raise ConformeNoAplicable(
            f"la calibración es de {calibracion.tarea} y la muestra, de {muestra_test.task}",
            f"the calibration is for {calibracion.tarea} and the sample, for {muestra_test.task}")
    _exigir_cuantil(calibracion)
    cubiertas = 0
    tamanos: list[float] = []
    vacios = 0
    if muestra_test.task == "regression":
        for y, p in zip(muestra_test.y_true, muestra_test.predictions):
            bajo, alto = intervalo_de_prediccion(p, calibracion)
            cubiertas += bajo <= float(y) <= alto
            tamanos.append(alto - bajo)
    else:
        if tuple(muestra_test.classes) != calibracion.clases:
            raise ConformeNoAplicable(
                "las clases de la muestra no son las de la calibración, o no van en el mismo orden",
                "the sample's classes are not the calibration's, or not in the same order")
        for y, fila in zip(muestra_test.y_true, muestra_test.probabilities):
            conjunto = conjunto_de_clases(fila, calibracion)
            cubiertas += y in conjunto
            tamanos.append(float(len(conjunto)))
            vacios += not conjunto
    n = len(tamanos)
    wilson = intervalo_wilson(cubiertas, n)
    intervalo_95 = None if wilson is None else (max(0.0, wilson[0] - wilson[1]),
                                                min(1.0, wilson[0] + wilson[1]))
    return CoberturaMedida(cobertura_nominal=calibracion.cobertura_nominal, n=n,
                           cubiertas=cubiertas, intervalo_95=intervalo_95,
                           tamano_medio=(sum(tamanos) / n if n else None), vacios=vacios)
