# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C3 — calibración binaria: la curva de fiabilidad, y el recalibrado
logístico que la resume en dos números (o uno).

QUÉ ES ESTO Y QUÉ NO ES. `calibration_in_the_large` (105-C1) ya existe y es
la MEDIA de probabilidad predicha menos la prevalencia observada — un
resumen de un solo número, sin ajustar nada. Este corte es otra cosa: AJUSTA
un modelo, `logit(P(Y=1)) = a + b·logit(p)`, sobre el mismo `p` que el motor
ya predijo, para responder «si tuviera que corregir esta probabilidad con
una recta en la escala logit, ¿qué recta sería?». `a=0, b=1` es la
calibración perfecta; desviarse dice EN QUÉ SENTIDO se desvía (b≠1: mal
calibrado en pendiente —sobre/sub-confiado según el signo—; a≠0 con b=1:
sesgo constante, la «calibración en grande» hecha modelo en vez de resumen).

STDLIB PURO, A PROPÓSITO. El ajuste es Newton-Raphson de 2 parámetros (o 1,
en el caso `intercepto_en_grande`) sobre la log-verosimilitud binomial — un
problema pequeño y bien condicionado la mayoría de las veces, que no
necesita `scipy.optimize`. Convergencia, separación y datos insuficientes se
MIDEN aquí, con su propio criterio, no se toman prestados de otra
herramienta (invariante 9 del 105).

LA SEPARACIÓN, Y POR QUÉ SE DETECTA EN VEZ DE DEVOLVER UN NÚMERO ENORME. Con
separación perfecta o casi perfecta, la verosimilitud logística no tiene
máximo finito: `b` crece sin límite y el hessiano se degenera (su
determinante tiende a cero, porque `sigmoid·(1-sigmoid)` se acerca a cero
en las predicciones que ya saturan a 0/1). Devolver el último `b` de un
Newton que diverge sería justo lo que el criterio de terminado prohíbe:
«separación... no devuelve conclusiones falsas». Se detecta por dos vías —
`|a|`/`|b|` superando un techo grande, o un hessiano casi singular— y se
declara `separacion_detectada=True` con el ajuste sin usar, nunca con un
número que parece preciso y no lo es.

QUÉ SE EVALÚA CON QUÉ. El criterio de terminado pide literalmente «datos
usados para evaluar calibración no ajustan ese calibrador»: `ajustar_*`
recibe la muestra de AJUSTE (desarrollo/cross-fitting) y `aplicar_*`/
`curva_de_fiabilidad` se llaman aparte sobre la muestra de EVALUACIÓN — este
módulo no impone ESE reparto (es del llamante, con `RegistroDeAccesos` del
104-C0 por debajo), pero tampoco lo esconde: `RecalibracionLogistica` no
tiene ningún método que evalúe sobre los mismos datos con los que se ajustó.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.metricas import Muestra
from matrixai.estudio.textos import motivo
from matrixai.estudio.validacion import digest_canonico, exigir_entero, exigir_real
from matrixai.estudio.vocabulario import exigir_opcion

__all__ = [
    "BinDeFiabilidad",
    "CurvaDeFiabilidad",
    "METODOS_DE_BINS",
    "METODOS_DE_RECALIBRACION",
    "RecalibracionLogistica",
    "aplicar_recalibracion",
    "ajustar_recalibracion_logistica",
    "curva_de_fiabilidad",
]

#: Cerrado: un tercer método de "hacer bins" a mitad de programa cambiaría
#: el ECE de todo el mundo sin que nadie lo pidiera.
METODOS_DE_BINS = ("ancho_igual", "frecuencia_igual")

#: «Declarar cuál se usa» (texto literal del corte): completo (a y b) o
#: intercepto-en-grande (b FIJO a 1, solo se estima a) — dos preguntas
#: distintas, nunca el mismo número con dos nombres.
METODOS_DE_RECALIBRACION = ("logistico_completo", "intercepto_en_grande")

#: Límite de |a|/|b| para declarar separación en vez de devolver un número
#: que "parece" preciso. Medido: con separación perfecta (200 puntos, y=1
#: syss z>0) el propio Newton ya supera esto en 9 iteraciones — no hace
#: falta un techo mayor para cazar el caso real, y uno menor arriesgaría
#: cortar un ajuste legítimo pero con b grande de verdad.
TECHO_COEFICIENTE = 50.0

#: Igual que `CLIP_LOG_LOSS` de 105-C1 pero declarado aparte a propósito:
#: aquí se recorta ANTES de tomar logit (evita ±inf), allí se recorta un
#: log() — son dos operaciones distintas que comparten el motivo (una
#: probabilidad de 0 o 1 exactos no es representable establemente), no la
#: misma constante reutilizada por casualidad.
CLIP_LOGIT = 1e-10


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    recortada = min(max(p, CLIP_LOGIT), 1.0 - CLIP_LOGIT)
    return math.log(recortada / (1.0 - recortada))


@dataclass(frozen=True)
class RecalibracionLogistica:
    """`logit(P(Y=1)) = a + b·logit(p)`. `a`/`b` son `None` exactamente
    cuando `undefined_reason` no lo es — nunca un cero de relleno."""

    metodo: str
    n_observaciones: int
    a: float | None = None
    b: float | None = None
    convergio: bool = False
    iteraciones: int = 0
    separacion_detectada: bool = False
    undefined_reason: dict[str, str] | None = None

    def __post_init__(self) -> None:
        exigir_opcion(self.metodo, "metodo", METODOS_DE_RECALIBRACION)
        exigir_entero(self.n_observaciones, "n_observaciones", minimo=0)
        exigir_entero(self.iteraciones, "iteraciones", minimo=0)
        if (self.a is None) != (self.b is None) and self.undefined_reason is None:
            # el completo puede tener los dos o ninguno; el de intercepto
            # siempre trae b=1.0 fijo cuando hay ajuste, así que un desajuste
            # entre a y b presentes solo puede pasar sin motivo si algo se
            # construyó a mano de forma incoherente.
            raise EsquemaInvalido("falta_campo", campo="a/b")
        if self.a is None and self.undefined_reason is None:
            raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo="recalibracion")
        if self.a is not None and self.undefined_reason is not None:
            raise EsquemaInvalido("metrica_con_valor_y_motivo", campo="recalibracion")

    def a_json(self) -> dict[str, Any]:
        return {
            "metodo": self.metodo, "n_observaciones": self.n_observaciones,
            "a": self.a, "b": self.b, "convergio": self.convergio,
            "iteraciones": self.iteraciones,
            "separacion_detectada": self.separacion_detectada,
            "undefined_reason": dict(self.undefined_reason) if self.undefined_reason else None,
        }

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def _newton_logistico(zs: Sequence[float], ys: Sequence[float], *, fijar_b: float | None,
                      max_iter: int = 100, tol: float = 1e-10) -> tuple[float | None, float | None, int, bool, bool]:
    """Devuelve `(a, b, iteraciones, convergio, separacion)`. Con `fijar_b`
    dado, el Newton es de 1 parámetro (`a` solo); si no, de 2 (`a` y `b`)."""
    a = 0.0
    b = fijar_b if fijar_b is not None else 1.0
    for it in range(1, max_iter + 1):
        ga = gb = haa = hab = hbb = 0.0
        for z, y in zip(zs, ys):
            s = _sigmoid(a + b * z)
            w = s * (1.0 - s)
            ga += (y - s)
            haa -= w
            if fijar_b is None:
                gb += z * (y - s)
                hab -= z * w
                hbb -= z * z * w

        if fijar_b is not None:
            if abs(haa) < 1e-12:
                return a, b, it, False, True
            da = ga / haa
            a_nuevo = a - da
            if abs(a_nuevo - a) < tol:
                return a_nuevo, b, it, True, False
            a = a_nuevo
        else:
            det = haa * hbb - hab * hab
            if abs(det) < 1e-9:
                return a, b, it, False, True
            da = (hbb * ga - hab * gb) / det
            db = (-hab * ga + haa * gb) / det
            a_nuevo, b_nuevo = a - da, b - db
            if abs(a_nuevo - a) < tol and abs(b_nuevo - b) < tol:
                return a_nuevo, b_nuevo, it, True, False
            a, b = a_nuevo, b_nuevo

        if abs(a) > TECHO_COEFICIENTE or abs(b) > TECHO_COEFICIENTE:
            return a, b, it, False, True
    return a, b, max_iter, False, False


def ajustar_recalibracion_logistica(muestra: Muestra, *, metodo: str = "logistico_completo",
                                    max_iter: int = 100, tol: float = 1e-10) -> RecalibracionLogistica:
    """Ajusta sobre `muestra` — se espera que sea la de DESARROLLO/
    cross-fitting, nunca el test reservado (el llamante lo garantiza, este
    módulo no le da a `RecalibracionLogistica` ningún método para evaluarse
    a sí misma sobre los mismos datos)."""
    exigir_opcion(metodo, "metodo", METODOS_DE_RECALIBRACION)
    probabilidades = muestra.probabilidad_del_positivo
    if probabilidades is None:
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=len(muestra.y_true),
            undefined_reason=motivo("clasificacion_sin_clases"))
    n = len(probabilidades)
    if n < 2:
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n,
            undefined_reason=motivo("calibracion_datos_insuficientes", campo="recalibracion_logistica"))

    ys = [1.0 if y == muestra.positive_label else 0.0 for y in muestra.y_true]
    if len(set(ys)) < 2:
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n,
            undefined_reason=motivo("calibracion_una_sola_clase", campo="recalibracion_logistica"))
    zs = [_logit(p) for p in probabilidades]

    fijar_b = 1.0 if metodo == "intercepto_en_grande" else None
    a, b, iteraciones, convergio, separacion = _newton_logistico(
        zs, ys, fijar_b=fijar_b, max_iter=max_iter, tol=tol)

    if separacion:
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n, iteraciones=iteraciones,
            convergio=False, separacion_detectada=True,
            undefined_reason=motivo("calibracion_separacion_detectada",
                                    campo="recalibracion_logistica",
                                    opciones=TECHO_COEFICIENTE, valor=iteraciones))
    if not convergio:
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n, iteraciones=iteraciones,
            convergio=False, separacion_detectada=False,
            undefined_reason=motivo("calibracion_no_convergio",
                                    campo="recalibracion_logistica", valor=iteraciones))
    return RecalibracionLogistica(metodo=metodo, n_observaciones=n, a=a, b=b,
                                  convergio=True, iteraciones=iteraciones,
                                  separacion_detectada=False)


def aplicar_recalibracion(recalibracion: RecalibracionLogistica,
                          probabilidades: Sequence[float]) -> tuple[float, ...]:
    """`p' = sigmoid(a + b·logit(p))`, sobre datos NUEVOS (nunca los que
    ajustaron `recalibracion` — eso lo garantiza el llamante)."""
    if recalibracion.a is None or recalibracion.b is None:
        raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo="recalibracion")
    a, b = recalibracion.a, recalibracion.b
    return tuple(_sigmoid(a + b * _logit(p)) for p in probabilidades)


@dataclass(frozen=True)
class BinDeFiabilidad:
    """Un cubo de la curva: cuántas observaciones cayeron en él, cuántos
    eventos reales tuvo, y la probabilidad media predicha frente a la tasa
    observada — las dos cifras que un diagrama de fiabilidad enfrenta."""

    limite_inferior: float
    limite_superior: float
    n: int
    eventos: int
    probabilidad_media_predicha: float
    tasa_observada: float

    def a_json(self) -> dict[str, Any]:
        return {"limite_inferior": self.limite_inferior, "limite_superior": self.limite_superior,
                "n": self.n, "eventos": self.eventos,
                "probabilidad_media_predicha": self.probabilidad_media_predicha,
                "tasa_observada": self.tasa_observada}


@dataclass(frozen=True)
class CurvaDeFiabilidad:
    """La curva entera. `ece` es `None` exactamente cuando `undefined_reason`
    no lo es (cero observaciones, o ningún bin con datos)."""

    bins: tuple[BinDeFiabilidad, ...]
    metodo_de_bins: str
    n_bins_solicitados: int
    ece: float | None = None
    undefined_reason: dict[str, str] | None = None

    def __post_init__(self) -> None:
        exigir_opcion(self.metodo_de_bins, "metodo_de_bins", METODOS_DE_BINS)
        exigir_entero(self.n_bins_solicitados, "n_bins_solicitados", minimo=1)
        if (self.ece is None) == (self.undefined_reason is None):
            raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo="ece")

    def a_json(self) -> dict[str, Any]:
        return {"bins": [b.a_json() for b in self.bins], "metodo_de_bins": self.metodo_de_bins,
                "n_bins_solicitados": self.n_bins_solicitados, "ece": self.ece,
                "undefined_reason": dict(self.undefined_reason) if self.undefined_reason else None}


def _limites_ancho_igual(n_bins: int) -> list[tuple[float, float]]:
    paso = 1.0 / n_bins
    return [(i * paso, (i + 1) * paso) for i in range(n_bins)]


def _limites_frecuencia_igual(probabilidades: Sequence[float], n_bins: int) -> list[tuple[float, float]]:
    ordenadas = sorted(probabilidades)
    n = len(ordenadas)
    cortes = [ordenadas[min(n - 1, round(i * n / n_bins))] for i in range(1, n_bins)]
    limites = [0.0] + cortes + [1.0]
    return [(limites[i], limites[i + 1]) for i in range(n_bins)]


def curva_de_fiabilidad(muestra: Muestra, *, n_bins: int = 10,
                        metodo_de_bins: str = "ancho_igual") -> CurvaDeFiabilidad:
    """`ece` depende del método de bins (texto literal del corte) — por
    eso `metodo_de_bins` viaja SIEMPRE en el resultado, nunca implícito."""
    exigir_entero(n_bins, "n_bins", minimo=1)
    exigir_opcion(metodo_de_bins, "metodo_de_bins", METODOS_DE_BINS)
    probabilidades = muestra.probabilidad_del_positivo
    if probabilidades is None or len(probabilidades) == 0:
        return CurvaDeFiabilidad(bins=(), metodo_de_bins=metodo_de_bins, n_bins_solicitados=n_bins,
                                 undefined_reason=motivo("clasificacion_sin_clases"))

    if metodo_de_bins == "ancho_igual":
        limites = _limites_ancho_igual(n_bins)
    else:
        limites = _limites_frecuencia_igual(probabilidades, n_bins)

    aciertos = [1.0 if y == muestra.positive_label else 0.0 for y in muestra.y_true]
    bins: list[BinDeFiabilidad] = []
    n_total = len(probabilidades)
    ece_acumulado = 0.0
    for idx, (lo, hi) in enumerate(limites):
        es_ultimo = idx == len(limites) - 1
        en_bin = [i for i, p in enumerate(probabilidades)
                 if (lo <= p < hi) or (es_ultimo and p == hi)]
        if not en_bin:
            bins.append(BinDeFiabilidad(lo, hi, 0, 0, 0.0, 0.0))
            continue
        n_bin = len(en_bin)
        eventos_bin = sum(int(aciertos[i]) for i in en_bin)
        media_predicha = sum(probabilidades[i] for i in en_bin) / n_bin
        tasa = eventos_bin / n_bin
        bins.append(BinDeFiabilidad(lo, hi, n_bin, eventos_bin, media_predicha, tasa))
        ece_acumulado += (n_bin / n_total) * abs(media_predicha - tasa)

    return CurvaDeFiabilidad(bins=tuple(bins), metodo_de_bins=metodo_de_bins,
                             n_bins_solicitados=n_bins, ece=ece_acumulado)
