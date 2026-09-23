# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C3 — bandas de decisión: positiva, revisión y negativa, con el acierto
de cada una MEDIDO en test.

**Qué añade al umbral de siempre.** El umbral del estudio (104-C4) parte las
filas en dos. Aquí se eligen DOS umbrales sobre la misma escala calibrada, y lo
que cae entre ellos no se decide solo: se manda a revisión. Es la «opción de
rechazo» de toda la vida, elegida con el mismo criterio que el umbral de
siempre —el coste realizado sobre la partición de calibración, nunca sobre
test— con un tercer coste: el de revisar un caso.

**Los tres costes los DECLARA alguien.** Esta función no tiene valores por
omisión para ellos a propósito: el coste de mandar un caso a una persona es una
preferencia de quien usa el modelo, y un número puesto aquí sería una
preferencia inventada. Si revisar sale más caro que equivocarse, la banda de
revisión sale vacía y el resultado es exactamente el umbral de siempre.

**Las fronteras**, compatibles con el umbral de siempre (positiva si `p ≥ t`):
- positiva si `p ≥ umbral_alto`;
- negativa si `p < umbral_bajo`;
- revisión si `umbral_bajo ≤ p < umbral_alto`.
Con los dos umbrales iguales no hay revisión y la regla ES la de siempre.

**Una banda pintada sin medir es peor que no tenerla** (invariante 1 del 115).
El acierto de cada banda lleva su intervalo de Wilson al 95 %, y solo se da por
medido si ese intervalo no es más ancho que ±`SEMIANCHO_MAXIMO`; si no, sale
con su motivo y no se pinta. El acierto de la banda positiva es cuántos de sus
casos son positivos de verdad (el VPP); el de la negativa, cuántos son
negativos (el VPN); de la de revisión se da cuántos positivos había dentro,
porque ahí no se decidió nada que acertar.

**Estado.** Fichero nuevo, sin cablear (la pasada de Fase 0 del 113 congela el
núcleo). Sin restricciones obligatorias todavía: si el problema declara alguna,
se rechaza con su motivo en vez de elegir bandas que podrían incumplirla.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from matrixai.estudio.esquemas import Restriccion
from matrixai.estudio.metricas import Muestra
from matrixai.training.diagnostico import intervalo_wilson

__all__ = ["BANDAS", "BandasNoAplicables", "MedidaDeBanda", "PoliticaDeBandas",
           "SEMIANCHO_MAXIMO", "elegir_bandas", "medir_bandas"]

BANDAS = ("positiva", "revision", "negativa")

#: El acierto de una banda se da por medido si su intervalo de Wilson al 95 %
#: no pasa de ±10 puntos. Con un acierto del 95 % eso pide unas 24 filas (medido con la misma función de Wilson; con 20, ±11,4); con
#: uno del 70 %, unas 78. Por debajo, la cifra existe pero no sirve para
#: decidir, y pintarla la haría parecer que sí.
SEMIANCHO_MAXIMO = 0.10


class BandasNoAplicables(ValueError):
    def __init__(self, es: str, en: str) -> None:
        super().__init__(es)
        self.motivo = {"es": es, "en": en}


@dataclass(frozen=True)
class PoliticaDeBandas:
    umbral_bajo: float
    umbral_alto: float
    positive_label: str
    negative_label: str
    coste_falso_positivo: float
    coste_falso_negativo: float
    coste_de_revision: float

    def __post_init__(self) -> None:
        if self.umbral_bajo > self.umbral_alto:
            raise ValueError("el umbral bajo no puede superar al alto")

    def banda_de(self, probabilidad: float) -> str:
        if probabilidad >= self.umbral_alto:
            return "positiva"
        if probabilidad < self.umbral_bajo:
            return "negativa"
        return "revision"

    def a_json(self) -> dict[str, Any]:
        return {"umbral_bajo": self.umbral_bajo, "umbral_alto": self.umbral_alto,
                "positive_label": self.positive_label, "negative_label": self.negative_label,
                "coste_falso_positivo": self.coste_falso_positivo,
                "coste_falso_negativo": self.coste_falso_negativo,
                "coste_de_revision": self.coste_de_revision}

    @classmethod
    def desde_json(cls, datos: Mapping[str, Any]) -> "PoliticaDeBandas":
        return cls(**{k: datos[k] for k in (
            "umbral_bajo", "umbral_alto", "positive_label", "negative_label",
            "coste_falso_positivo", "coste_falso_negativo", "coste_de_revision")})


def _probabilidades(muestra: Muestra) -> tuple[float, ...]:
    if muestra.task != "binary_classification":
        raise BandasNoAplicables(f"las bandas son, de momento, solo para binaria (esta es {muestra.task})",
                                 f"bands are binary-only for now (this is {muestra.task})")
    probabilidades = muestra.probabilidad_del_positivo
    if probabilidades is None:
        raise BandasNoAplicables(
            "hace falta la probabilidad calibrada del positivo por fila",
            "the calibrated probability of the positive class per row is needed")
    return probabilidades


def elegir_bandas(muestra_calibracion: Muestra, *, coste_falso_positivo: float,
                  coste_falso_negativo: float, coste_de_revision: float,
                  restricciones: Sequence[Restriccion] = (),
                  umbral: float | None = None) -> PoliticaDeBandas:
    """Los dos umbrales que minimizan el coste REALIZADO en calibración:
    `(FP·c_fp + FN·c_fn + revisados·c_rev) / n`.

    El coste se separa en una parte que solo depende del umbral bajo (los
    positivos que caen por debajo, menos lo que ahorra no revisarlos) y otra
    que solo depende del alto; con el mínimo acumulado de la primera se
    recorren todos los pares con `bajo ≤ alto` en `O(n log n)`. En empate de
    coste gana el par con MENOS casos a revisión, y después el primero.

    **`umbral`: las bandas no contradicen al umbral de siempre (2026-09-23).** El
    umbral (`elegir_umbral`) y las bandas se eligen por separado sobre la misma
    muestra y los mismos costes, y DESEMPATAN distinto: medido en una auditoría,
    umbral 0,737 y bandas [0,717, 0,717], así que una fila con p = 0,727 salía
    «no» por el umbral y «positiva» por la banda (2 de 120 filas; con revisar a
    0,5, 6 de 120). Con `umbral` dado, solo se consideran pares con
    `bajo ≤ umbral ≤ alto` —así nada que el umbral llame negativo cae en la banda
    positiva, ni al revés— y el propio umbral entra como candidato, de modo que
    sin zona de revisión la política ES el umbral (`bajo = alto = umbral`). Sin
    `umbral`, lo de antes."""
    if any(r.obligatoria for r in restricciones):
        raise BandasNoAplicables(
            "el problema declara restricciones obligatorias, y las bandas todavía no saben "
            "cumplirlas: se usa el umbral de siempre",
            "the problem declares mandatory constraints, and bands cannot honour them yet: the "
            "usual threshold is used")
    for nombre, valor in (("coste_falso_positivo", coste_falso_positivo),
                          ("coste_falso_negativo", coste_falso_negativo),
                          ("coste_de_revision", coste_de_revision)):
        if not valor >= 0:
            raise ValueError(f"{nombre} tiene que ser un número ≥ 0, no {valor!r}")
    probabilidades = _probabilidades(muestra_calibracion)
    positiva = muestra_calibracion.positive_label
    negativa = next(c for c in muestra_calibracion.classes if c != positiva)
    es_positivo = [y == positiva for y in muestra_calibracion.y_true]
    n = len(probabilidades)
    if n == 0:
        raise BandasNoAplicables("no hay filas de calibración", "there are no calibration rows")

    candidatos = sorted(set(probabilidades) | {0.0, 1.0}
                        | ({float(umbral)} if umbral is not None else set()))
    # Con umbral: `bajo` solo entre los candidatos ≤ umbral y `alto` solo entre los ≥ umbral.
    indice_del_umbral = candidatos.index(float(umbral)) if umbral is not None else None
    # Para cada candidato t: cuántas filas (y cuántas positivas) quedan por DEBAJO.
    pares = sorted(zip(probabilidades, es_positivo))
    debajo: list[int] = []
    positivas_debajo: list[int] = []
    i = contadas = positivas = 0
    for t in candidatos:
        while i < n and pares[i][0] < t:
            contadas += 1
            positivas += pares[i][1]
            i += 1
        debajo.append(contadas)
        positivas_debajo.append(positivas)
    total_positivas = sum(es_positivo)
    total_negativas = n - total_positivas

    # f(bajo) = FN·c_fn − (filas por debajo)·c_rev ; g(alto) = FP·c_fp − (filas por encima)·c_rev
    def f(j: int) -> float:
        return positivas_debajo[j] * coste_falso_negativo - debajo[j] * coste_de_revision

    def g(j: int) -> float:
        encima = n - debajo[j]
        negativas_encima = total_negativas - (debajo[j] - positivas_debajo[j])
        return negativas_encima * coste_falso_positivo - encima * coste_de_revision

    mejor: tuple[float, int, int, int] | None = None  # (coste, revisados, bajo, alto)
    mejor_bajo = 0
    for alto in range(len(candidatos)):
        puede_ser_bajo = indice_del_umbral is None or alto <= indice_del_umbral
        if puede_ser_bajo and (f(alto) < f(mejor_bajo)
                               or (f(alto) == f(mejor_bajo) and debajo[alto] > debajo[mejor_bajo])):
            mejor_bajo = alto
        if indice_del_umbral is not None and alto < indice_del_umbral:
            continue
        coste = f(mejor_bajo) + g(alto) + n * coste_de_revision
        revisados = debajo[alto] - debajo[mejor_bajo]
        clave = (coste, revisados, mejor_bajo, alto)
        if mejor is None or clave[:2] < mejor[:2]:
            mejor = clave
    _, _, bajo, alto = mejor
    return PoliticaDeBandas(umbral_bajo=candidatos[bajo], umbral_alto=candidatos[alto],
                            positive_label=positiva, negative_label=negativa,
                            coste_falso_positivo=coste_falso_positivo,
                            coste_falso_negativo=coste_falso_negativo,
                            coste_de_revision=coste_de_revision)


@dataclass(frozen=True)
class MedidaDeBanda:
    banda: str
    n: int
    #: Qué parte de las filas de test cayó en esta banda.
    cobertura: float | None
    #: Positiva: positivos de verdad; negativa: negativos; revisión: positivos dentro.
    aciertos: int
    proporcion: float | None
    intervalo_95: tuple[float, float] | None
    medida: bool
    motivo: dict[str, str] | None

    def a_json(self) -> dict[str, Any]:
        return {"banda": self.banda, "n": self.n, "cobertura": self.cobertura,
                "aciertos": self.aciertos, "proporcion": self.proporcion,
                "intervalo_95": list(self.intervalo_95) if self.intervalo_95 else None,
                "medida": self.medida, "motivo": self.motivo}


def medir_bandas(muestra_test: Muestra, politica: PoliticaDeBandas) -> tuple[MedidaDeBanda, ...]:
    """Cada banda en test, con su intervalo, y si se puede pintar."""
    probabilidades = _probabilidades(muestra_test)
    n_total = len(probabilidades)
    por_banda: dict[str, list[bool]] = {b: [] for b in BANDAS}
    for p, y in zip(probabilidades, muestra_test.y_true):
        por_banda[politica.banda_de(p)].append(y == politica.positive_label)
    medidas = []
    for banda in BANDAS:
        positivos = por_banda[banda]
        n = len(positivos)
        aciertos = (n - sum(positivos)) if banda == "negativa" else sum(positivos)
        wilson = intervalo_wilson(aciertos, n)
        intervalo = None if wilson is None else (max(0.0, wilson[0] - wilson[1]),
                                                 min(1.0, wilson[0] + wilson[1]))
        medida = wilson is not None and wilson[1] <= SEMIANCHO_MAXIMO
        motivo = None if medida else {
            "es": f"con {n} filas de test en esta banda su acierto no se conoce con un margen de "
                  f"±{SEMIANCHO_MAXIMO:.0%}: no se pinta",
            "en": f"with {n} test rows in this band its accuracy is not known within "
                  f"±{SEMIANCHO_MAXIMO:.0%}: it is not shown"}
        medidas.append(MedidaDeBanda(
            banda=banda, n=n, cobertura=(n / n_total if n_total else None), aciertos=aciertos,
            proporcion=(aciertos / n if n else None), intervalo_95=intervalo, medida=medida,
            motivo=motivo))
    return tuple(medidas)
