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
separación completa o cuasi completa, la verosimilitud logística no tiene
máximo finito: `b` crece sin límite. Devolver el último `b` de un Newton que
diverge sería justo lo que el criterio de terminado prohíbe: «separación...
no devuelve conclusiones falsas». Se declara `separacion_detectada=True` con
el ajuste sin usar, nunca con un número que parece preciso y no lo es.

141 (28-10): LA SEPARACIÓN SE MIRA EN LOS DATOS, NO EN EL CAMINO DEL NEWTON.
Hasta el 141 se declaraba cuando `|a|`/`|b|` superaba 50 a mitad de
iteraciones, o con un hessiano casi singular. Un Newton SIN amortiguar que
parte de `a=0, b=1` con logits extremos (−12 a +7, los de un buen
clasificador) da un primer paso enorme y pasa de 50 en 2–3 iteraciones con
datos que NO son separables: medido sobre las predicciones fuera de pliegue
de climate-model-simulation-crashes, 3 de 6 subconjuntos de 215 filas salían
«separación» y `LogisticRegression(C=1e8)` daba a≈0,4, b≈0,56. El estudio
entregaba sin recalibrar cuando había un ajuste finito. Ahora: (1) con `a` y
`b` libres, el máximo existe si y solo si las puntuaciones de las dos clases
SE SOLAPAN (Albert y Anderson, 1984: en una dimensión con intercepto, no
existe si `min(z | y=1) >= max(z | y=0)` o al revés) — eso se comprueba
sobre los `z` antes de iterar; con `b` fijo el máximo de `a` siempre existe
si hay las dos clases; (2) el Newton va AMORTIGUADO: un paso que empeora la
verosimilitud se parte a la mitad hasta que no la empeora, así que con
máximo finito converge desde cualquier arranque (la log-verosimilitud es
cóncava); (3) «no convergió» sigue existiendo con su motivo. Y una muestra
con TODAS las puntuaciones iguales no es separación: es una recta sin
pendiente identificable, y lleva su propio motivo.

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
from matrixai.estudio.textos import PorIdioma, motivo
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

#: Cuántas veces se parte a la mitad un paso de Newton que empeora la
#: verosimilitud antes de rendirse (2**-60 ≈ 1e-18: por debajo de eso el paso
#: ya no mueve un `float`).
MAX_MITADES = 60

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


def _log1p_exp(x: float) -> float:
    """`log(1 + e^x)` sin desbordar en ninguno de los dos extremos."""
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _evaluar(zs: Sequence[float], ys: Sequence[float], a: float, b: float, libre_b: bool
             ) -> tuple[float, float, float, float, float, float]:
    """Log-verosimilitud, gradiente y hessiano en `(a, b)`, en UNA pasada:
    `(ll, ga, gb, haa, hab, hbb)`. `ll_i = y·η − log(1+e^η)`, con `η = a + b·z`."""
    ll = ga = gb = haa = hab = hbb = 0.0
    for z, y in zip(zs, ys):
        eta = a + b * z
        ll += y * eta - _log1p_exp(eta)
        s = _sigmoid(eta)
        w = s * (1.0 - s)
        ga += (y - s)
        haa -= w
        if libre_b:
            gb += z * (y - s)
            hab -= z * w
            hbb -= z * z * w
    return ll, ga, gb, haa, hab, hbb


def _separacion_en_los_datos(zs: Sequence[float], ys: Sequence[float]) -> str | None:
    """`"completa"`, `"cuasi_completa"` o `None`, MIRANDO LOS DATOS. En una
    dimensión con intercepto, el máximo de `a + b·z` no existe si y solo si
    hay un corte con todas las de una clase a un lado y todas las de la otra
    al otro (Albert y Anderson, 1984): `min(z|y=1) >= max(z|y=0)` o
    `max(z|y=1) <= min(z|y=0)`. Estricto es completa; con empate en el corte,
    cuasi completa. El llamante descarta antes una sola clase y el `z`
    constante (que cumpliría la igualdad sin ser separación)."""
    unos = [z for z, y in zip(zs, ys) if y == 1.0]
    ceros = [z for z, y in zip(zs, ys) if y != 1.0]
    for abajo, arriba in ((ceros, unos), (unos, ceros)):
        if min(arriba) > max(abajo):
            return "completa"
        if min(arriba) >= max(abajo):
            return "cuasi_completa"
    return None


def _newton_logistico(zs: Sequence[float], ys: Sequence[float], *, fijar_b: float | None,
                      max_iter: int = 100, tol: float = 1e-10) -> tuple[float | None, float | None, int, bool, bool]:
    """Devuelve `(a, b, iteraciones, convergio, separacion)`. Con `fijar_b`
    dado, el Newton es de 1 parámetro (`a` solo); si no, de 2 (`a` y `b`).

    141: AMORTIGUADO. Cada paso de Newton se parte a la mitad mientras empeore
    la log-verosimilitud (hasta `MAX_MITADES` veces); si el hessiano no es
    definido negativo en el punto (pesos `s·(1-s)` que se anulan), el paso es
    el del gradiente, también con búsqueda en línea. La separación se decide
    sobre los datos antes de iterar (`_separacion_en_los_datos`), nunca por el
    tamaño de un coeficiente a mitad de camino."""
    libre_b = fijar_b is None
    if libre_b and _separacion_en_los_datos(zs, ys) is not None:
        return None, None, 0, False, True
    a = 0.0
    b = fijar_b if fijar_b is not None else 1.0
    ll, ga, gb, haa, hab, hbb = _evaluar(zs, ys, a, b, libre_b)
    for it in range(1, max_iter + 1):
        if libre_b:
            det = haa * hbb - hab * hab
            if haa < 0.0 and det > 0.0:
                da = -(hbb * ga - hab * gb) / det
                db = -(-hab * ga + haa * gb) / det
            else:
                escala = 1.0 / max(1.0, math.hypot(ga, gb))
                da, db = ga * escala, gb * escala
        else:
            da = -ga / haa if haa < 0.0 else ga / max(1.0, abs(ga))
            db = 0.0
        if max(abs(da), abs(db)) < tol:
            return a + da, b + db, it, True, False
        # la mejora que el paso promete (decremento de Newton, o |g|² si es el del gradiente)
        prometida = ga * da + gb * db
        holgura = 1e-13 * (1.0 + abs(ll))
        t = 1.0
        for _ in range(MAX_MITADES):
            a_n, b_n = a + t * da, b + t * db
            nuevo = _evaluar(zs, ys, a_n, b_n, libre_b)
            if math.isfinite(nuevo[0]) and nuevo[0] >= ll - holgura:
                break
            t *= 0.5
        else:
            # ningún trozo del paso deja de empeorarla: o ya estamos en el máximo (lo que el paso
            # prometía cabe en el ruido de redondeo), o el ajuste no se ha estabilizado.
            if prometida <= holgura:
                return a, b, it, True, False
            return a, b, it, False, False
        a, b = a_n, b_n
        ll, ga, gb, haa, hab, hbb = nuevo
    return a, b, max_iter, False, False


_ROTULO_DE_SEPARACION = {
    "completa": PorIdioma(es="completa", en="complete"),
    "cuasi_completa": PorIdioma(es="cuasi completa (empate en el corte)", en="quasi-complete (tie at the cut)"),
}


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
    if fijar_b is None and min(zs) == max(zs):
        # 141: no es separación (no hay corte que deje una clase a cada lado): con un solo `z`
        # solo se identifica `a + b·z`, no `a` y `b` por separado.
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n,
            undefined_reason=motivo("calibracion_puntuacion_constante",
                                    campo="recalibracion_logistica"))
    a, b, iteraciones, convergio, separacion = _newton_logistico(
        zs, ys, fijar_b=fijar_b, max_iter=max_iter, tol=tol)

    if separacion:
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n, iteraciones=iteraciones,
            convergio=False, separacion_detectada=True,
            undefined_reason=motivo("calibracion_separacion_detectada",
                                    campo="recalibracion_logistica",
                                    opciones=_ROTULO_DE_SEPARACION[_separacion_en_los_datos(zs, ys)]))
    if not convergio:
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n, iteraciones=iteraciones,
            convergio=False, separacion_detectada=False,
            undefined_reason=motivo("calibracion_no_convergio",
                                    campo="recalibracion_logistica", valor=iteraciones))
    if b < 0:
        # Decisión de Roberto (28-10, A): una pendiente NEGATIVA invierte la puntuación (el modelo ordena al revés en estas
        # filas). Aceptarla le daría la vuelta en silencio a un modelo que predice peor que el azar, y la guarda del campeón
        # dejaría de verlo. Se declara sin ajustar, con su motivo.
        return RecalibracionLogistica(
            metodo=metodo, n_observaciones=n, iteraciones=iteraciones,
            convergio=True, separacion_detectada=False,
            undefined_reason=motivo("calibracion_pendiente_negativa", campo="recalibracion_logistica"))
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
