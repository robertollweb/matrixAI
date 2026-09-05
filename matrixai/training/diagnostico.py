# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los detectores del diagnóstico, con su semántica correcta — contrato 103, C2.

QUÉ ENTREGA. Sobre un problema YA CONFIRMADO (`ProblemSpec` del 103-C1), mide
los datos y separa **tres cosas que no son la misma** — es la frase literal del
contrato, y el corte entero es que no se mezclen nunca:

* **Errores estructurales** (`Bloqueo`, reutilizado de `objetivo.py`): impiden
  una recomendación válida y NO se levantan aceptándolos. Tres, y solo tres:
  un predictor que reproduce el objetivo fila a fila (`objetivo_duplicado`),
  una entrada declarada disponible SOLO después del desenlace
  (`variables_disponibles_tras_el_desenlace`, leyendo lo que YA declaró el
  104-C0 en `ProblemSpec.predictor_availability` — nunca lo infiere: el
  invariante 5 lo prohíbe), y el identificador de unidad/grupo cruzado a la
  vez como entrada (`cruce_de_unidades`).
* **Sospechas** (`Sospecha`, nueva aquí): exigen contexto — se pueden aceptar
  con motivo, autor y aviso persistente, y una alta correlación **por sí sola
  no demuestra fuga** (es el error que este corte viene a corregir, medido con
  el caso del salario del contrato 71 más abajo). Cuatro: igualdad de valores
  entre dos entradas sin explicar (`igualdad_de_valores`), asociación muy alta
  con el objetivo (`asociacion_muy_alta`, Pearson/Spearman), proxy probable
  (`proxy_por_nmi` / `proxy_por_pureza`) e identificador probable
  (`identificador_probable`).
* **Límites de estimación** (`Limite`, nueva aquí): NO son fugas, acotan la
  conclusión — `falta_de_eventos`, `tamano_efectivo_insuficiente`,
  `precision_insuficiente` y `ausencia_de_linaje` (ésta última SIEMPRE
  presente: ningún detector de este core ve cómo se calculó una columna
  externa, y eso se declara como límite, no como ausencia de problema —
  criterio de terminado #4).

POR QUÉ PEARSON/SPEARMAN/NMI/PUREZA SON SEÑALES, NO PRUEBAS. El contrato lo
dice literal y aquí se aplica en el CÓDIGO, no solo en la prosa:

* Una correlación de Pearson 0,95 entre `last_year_salary` y `salary` —medida
  más abajo con un dataset sintético fijado por semilla, r ≈ 0,955— es
  EXACTAMENTE el caso legítimo del contrato 71: el sueldo anterior predice
  bien el siguiente. `asociacion_muy_alta` la marca como SOSPECHA (nunca
  `Bloqueo`) para que se pueda conservar documentando cuándo está disponible.
* La NMI tiene SESGO POR CARDINALIDAD, y está medido en este mismo módulo (ver
  `nmi_categorica`): un identificador de cardinalidad ≈ n contra un objetivo
  ALEATORIO da NMI ≈ 0,22 con el método sin corregir — más alta que una
  categoría con una asociación REAL fuerte pero minoritaria (NMI ≈ 0,09-0,15
  en los escenarios medidos aquí). Agrupar antes las categorías con menos de
  `SOPORTE_MINIMO_PUREZA` casos borra ese sesgo (NMI del identificador cae a
  0,0 tras agrupar, medido).
* La PUREZA por categoría, sin soporte mínimo, es la trampa exacta del
  criterio de terminado #3: sobre un CSV de 300 filas y 40 categorías SIN
  relación real con el objetivo, el 98 % de las repeticiones (196/200,
  medido con `random.seed` fijado) encuentra una categoría "pura" (≥ 0,85)
  por puro azar de muestra pequeña. Con soporte mínimo 20 esa tasa cae a
  0/200. Por eso `pureza_categorica` exige soporte mínimo Y evidencia FUERA DE
  MUESTRA (una partición INTERNA por paridad de índice — nunca el test
  reservado del protocolo 104-C0, que aquí ni siquiera existe todavía: el
  diagnóstico ocurre antes de que el 103-C4 construya ningún `SplitPlan`).

LOS UMBRALES, Y SU MEDIDA (invariante 9: «llevan medida y justificación», «no
se transfieren de otra herramienta si su estadístico es distinto» — así que
nada aquí viene citado de fuera, todo se mide DENTRO de este módulo o de su
prueba):

* `UMBRAL_ASOCIACION_ALTA = 0.9` — Pearson/Spearman. Bracket medido: 0,955
  (salario legítimo del 71) por encima, 0,46 (correlación moderada típica) muy
  por debajo. Se combina con `_r_critico(n)`, el umbral de significatividad
  estadística DE LA PROPIA correlación de Pearson vía la transformación de
  Fisher (`z = atanh(r)·√(n-3)`, con `z` fijo para α = 0,01 dos colas —
  2,5758293035489004, el cuantil de la normal que no hace falta invertir en
  producción porque α está fijado aquí) — para no acusar una correlación alta
  que a la vez es estadísticamente compatible con el azar en muestras
  pequeñas.
* `N_MINIMO_ASOCIACION = 30` — por debajo, el error estándar de Fisher
  (1/√(n-3)) es demasiado grande para que el número signifique algo estable.
* `SOPORTE_MINIMO_PUREZA = 20` y `UMBRAL_PUREZA_ALTA = 0.85` — medidos arriba:
  sin soporte mínimo, falsos positivos en el 98 % de las repeticiones; con él,
  0 %. `UMBRAL_PUREZA_ALTA_FUERA_DE_MUESTRA = 0.75` es más laxo a propósito:
  cada mitad interna tiene la mitad de filas, así que exigir el mismo 0,85 en
  cada una penalizaría dos veces el mismo ruido de muestra.
* `UMBRAL_NMI_ALTA = 0.3` — medido: una categoría que determina el objetivo al
  100 % pero solo en 1 de 4 subgrupos (75 % de las filas sin señal) llega a
  NMI ≈ 0,15; una dependencia total (proxy perfecto o identidad) llega a 1,0.
  El umbral separa los dos sin rozar ninguno.
* `UMBRAL_EVENTOS_MINIMOS = 10` — no es una regla de otra herramienta: para un
  RECUENTO de eventos, el error relativo de muestreo es del orden de
  1/√eventos (razonamiento de Poisson, no un umbral prestado), y en 10 eso ya
  es ≈ 32 %.
* `RATIO_MINIMO_FILAS_POR_PREDICTOR = 10` — grados de libertad: con menos de
  diez filas por predictor declarado, la estimación queda dominada por el
  ruido de la muestra concreta.
* `UMBRAL_MARGEN_RELATIVO_WILSON = 0.3` — sobre el intervalo de Wilson al 95 %
  (`z = 1.959963984540054`, el cuantil de la normal al 97,5 % — DISTINTO del
  usado para `asociacion_muy_alta` a propósito: aquí se describe la
  incertidumbre de una cifra ya publicada, allí se decide si acusar una
  entrada, y las dos preguntas no comparten nivel de exigencia). Medido: con
  10 eventos el margen relativo es ≈ 0,60-0,65 con independencia de `n`; con
  100 eventos cae a ≈ 0,19.

LO QUE SE MIDE SOBRE EL CSV ENTERO, Y SU COSTE. Estas señales son O(filas) por
par predictor-objetivo, y O(predictores²) para `igualdad_de_valores` entre
pares de entradas. Medido sobre un CSV sintético de 50.000 filas × 30 columnas
(ver la prueba de coste): recorrer todos los detectores sobre TODAS las filas
se queda por debajo del segundo; aun así, `MUESTREO_MAXIMO_FILAS` existe para
no dejarlo sin techo, y cuando actúa lo DECLARA en `Diagnostico.muestreado` —
nunca en silencio.

STDLIB PURO. Pearson, Spearman (por rangos, con empates promediados),
información mutua y su normalización se escriben aquí a mano —
`dependencies = []` no se toca—. `numpy`/`scipy`/`scikit-learn` verifican la
paridad de las fórmulas SOLO en `tests/test_c103_c2_detectores.py`, con
`skipUnless`.

LO QUE NO USA ESTE MÓDULO, dicho para que no se dé por hecho: la verdad de la
prueba reservada. Ningún detector de aquí recibe filas de rol `test` ni
`external_test` — el diagnóstico corre en desarrollo, sobre el CSV que el
usuario ha traído, antes de que exista ningún `SplitPlan` (eso es el 103-C4).
La partición «fuera de muestra» de `pureza_categorica` es interna a esta
función y no pasa por `RegistroDeAccesos`: no hace falta, porque no toca nada
reservado — es la misma idea que el propio desarrollo aplica dentro de sí
mismo antes de mirar ningún test.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from matrixai.estudio import ProblemSpec
from matrixai.training.diagnostico_textos import motivo
from matrixai.training.objetivo import Bloqueo

__all__ = [
    "Diagnostico", "Limite", "MUESTREO_MAXIMO_FILAS", "N_MINIMO_ASOCIACION",
    "RATIO_MINIMO_FILAS_POR_PREDICTOR", "SOPORTE_MINIMO_PUREZA", "Sospecha",
    "UMBRAL_ASOCIACION_ALTA", "UMBRAL_EVENTOS_MINIMOS",
    "UMBRAL_MARGEN_RELATIVO_WILSON", "UMBRAL_NMI_ALTA", "UMBRAL_PUREZA_ALTA",
    "UMBRAL_PUREZA_ALTA_FUERA_DE_MUESTRA", "asociacion", "asociacion_muy_alta",
    "ausencia_de_linaje", "cruce_de_unidades", "diagnosticar_csv",
    "falta_de_eventos", "identificador_probable", "igualdad_de_valores",
    "intervalo_wilson", "nmi_categorica", "objetivo_duplicado", "pearson",
    "precision_insuficiente", "proxy_por_nmi", "proxy_por_pureza",
    "pureza_categorica", "spearman", "tamano_efectivo_insuficiente",
    "variables_disponibles_tras_el_desenlace",
]

# ---------------------------------------------------------------------------
# Umbrales — cada uno con su medida en el docstring de arriba
# ---------------------------------------------------------------------------

UMBRAL_ASOCIACION_ALTA = 0.9
N_MINIMO_ASOCIACION = 30
#: scipy.stats.norm.ppf(0.995) — el cuantil de la normal para α = 0,01 a dos
#: colas. Se cita el valor y no se invierte la normal en producción porque α
#: está FIJO: no hace falta una función general de cuantiles para un solo punto.
_Z_ALPHA_001 = 2.5758293035489004
SOPORTE_MINIMO_PUREZA = 20
UMBRAL_PUREZA_ALTA = 0.85
UMBRAL_PUREZA_ALTA_FUERA_DE_MUESTRA = 0.75
UMBRAL_NMI_ALTA = 0.3
UMBRAL_EVENTOS_MINIMOS = 10
RATIO_MINIMO_FILAS_POR_PREDICTOR = 10
#: scipy.stats.norm.ppf(0.975) — el cuantil al 95 % de confianza, para
#: DESCRIBIR una cifra ya publicada. Deliberadamente distinto de `_Z_ALPHA_001`
#: (99 %, para decidir si SOSPECHAR de una entrada): describir y acusar no
#: comparten nivel de exigencia.
_Z_95 = 1.959963984540054
UMBRAL_MARGEN_RELATIVO_WILSON = 0.3
#: Filas por encima de las cuales las señales de este módulo se calculan sobre
#: una MUESTRA, no el CSV entero — medido y declarado, nunca en silencio
#: (ver la prueba de coste). 50.000 es diez veces el techo de fila del perfil
#: `hosted` (`matrixai.limits`) y ya cubre con margen los CSV reales que este
#: producto analiza sin invocar Studio descargable sin techo.
MUESTREO_MAXIMO_FILAS = 50_000


# ---------------------------------------------------------------------------
# Las tres formas de resultado — Bloqueo se reutiliza de objetivo.py (103-C1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sospecha:
    """Algo que EXIGE CONTEXTO (103, invariante 3). No bloquea: se acepta con
    motivo, autor y aviso persistente — la aceptación misma es del 103-C5, aquí
    solo se detecta y se mide."""

    clave: str
    campo: str | None
    motivo: dict[str, str]
    medida: dict[str, Any] = field(default_factory=dict)

    def a_json(self) -> dict[str, Any]:
        return {"clave": self.clave, "campo": self.campo,
                "motivo": dict(self.motivo), "medida": dict(self.medida)}


@dataclass(frozen=True)
class Limite:
    """Un LÍMITE de estimación (103, invariante 4): no es una fuga, acota la
    conclusión o deja una métrica indefinida. Aceptar el riesgo no cambia su
    estado científico — declararlo tampoco lo hace desaparecer."""

    clave: str
    campo: str | None
    motivo: dict[str, str]
    medida: dict[str, Any] = field(default_factory=dict)

    def a_json(self) -> dict[str, Any]:
        return {"clave": self.clave, "campo": self.campo,
                "motivo": dict(self.motivo), "medida": dict(self.medida)}


@dataclass(frozen=True)
class Diagnostico:
    """Las tres listas, SEPARADAS — es el corte entero. El esquema versionado
    completo (con `ProblemSpec`, políticas, decisiones y `SplitPlan`) es del
    103-C5; esto es lo que los detectores de este corte producen."""

    bloqueos: tuple[Bloqueo, ...] = ()
    sospechas: tuple[Sospecha, ...] = ()
    limites: tuple[Limite, ...] = ()
    muestreado: bool = False
    filas_medidas: int = 0
    filas_totales: int = 0

    @property
    def impide_recomendacion(self) -> bool:
        """Solo los errores ESTRUCTURALES impiden recomendar (invariante 2).
        Ni las sospechas ni los límites lo hacen — convertirlos en bloqueo es
        justo lo que el 71 midió que acusaba a una columna legítima."""
        return bool(self.bloqueos)

    def a_json(self) -> dict[str, Any]:
        return {
            "impide_recomendacion": self.impide_recomendacion,
            "bloqueos": [b.a_json() for b in self.bloqueos],
            "sospechas": [s.a_json() for s in self.sospechas],
            "limites": [l.a_json() for l in self.limites],
            "muestreado": self.muestreado,
            "filas_medidas": self.filas_medidas,
            "filas_totales": self.filas_totales,
        }


# ---------------------------------------------------------------------------
# Primitivas estadísticas — stdlib puro
# ---------------------------------------------------------------------------

def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Correlación de Pearson. `None` si `n < 2` o si alguna serie es
    constante (varianza cero: el coeficiente no está definido, y `0.0` sería
    afirmar «sin asociación» sobre algo que en realidad no se pudo calcular —
    un valor ausente no es un cero)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _rangos(xs: Sequence[float]) -> list[float]:
    """Rango 1-based de cada valor, con EMPATES promediados (el rango medio
    del grupo) — es lo que hace que Spearman de por rangos coincida con
    `scipy.stats.spearmanr`, que también promedia empates."""
    orden = sorted(range(len(xs)), key=lambda i: xs[i])
    rangos = [0.0] * len(xs)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and xs[orden[j + 1]] == xs[orden[i]]:
            j += 1
        rango_medio = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rangos[orden[k]] = rango_medio
        i = j + 1
    return rangos


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Correlación de Spearman: Pearson calculado sobre los RANGOS. Mide
    asociación MONÓTONA, no solo lineal — una `x` y una `y = x**3` dan
    Spearman 1,0 y Pearson bastante menos, y la prueba de paridad lo fija."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(_rangos(list(xs)), _rangos(list(ys)))


def _r_critico(n: int, z: float = _Z_ALPHA_001) -> float | None:
    """El |r| a partir del cual una correlación de Pearson es estadísticamente
    distinguible de cero, vía la transformación de Fisher: bajo H0 (rho=0),
    `atanh(r)·√(n-3)` es aproximadamente normal estándar, así que el `r`
    crítico es `tanh(z/√(n-3))`. Es una propiedad DE LA PROPIA correlación de
    Pearson —no un umbral prestado de otra herramienta (invariante 9)— y se
    usa también como aproximación para Spearman a falta de una fórmula propia
    en `stdlib` (aproximación asintótica habitual, declarada aquí para que
    quien audite sepa que lo es)."""
    if n <= 3:
        return None
    return math.tanh(z / math.sqrt(n - 3))


def asociacion(xs: Sequence[Any], ys: Sequence[Any], *, metodo: str = "pearson") -> dict[str, Any] | None:
    """Pearson o Spearman, alineando y quitando pares con algún lado sin
    número. `None` si quedan menos de `N_MINIMO_ASOCIACION` pares — por debajo,
    el error estándar de Fisher es demasiado grande para que el número
    signifique algo (ver docstring del módulo)."""
    pares: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        try:
            pares.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    n = len(pares)
    if n < N_MINIMO_ASOCIACION:
        return None
    valores_x = [p[0] for p in pares]
    valores_y = [p[1] for p in pares]
    r = pearson(valores_x, valores_y) if metodo == "pearson" else spearman(valores_x, valores_y)
    if r is None:
        return None
    critico = _r_critico(n)
    return {
        "metodo": metodo, "valor": round(r, 6), "n": n,
        "r_critico_1pct": round(critico, 6) if critico is not None else None,
    }


def _entropia(conteos: Mapping[Any, int], n: int) -> float:
    """Entropía de Shannon en BITS (`log2`). `0.0` para una única categoría —
    no hay incertidumbre que medir."""
    if n <= 0:
        return 0.0
    total = 0.0
    for c in conteos.values():
        if c <= 0:
            continue
        p = c / n
        total -= p * math.log2(p)
    return total


def _informacion_mutua(pares: Sequence[tuple[Any, Any]]) -> tuple[float, float, float]:
    """`(I(X;Y), H(X), H(Y))` en bits, sobre los pares dados."""
    n = len(pares)
    conteo_x: Counter = Counter(a for a, _ in pares)
    conteo_y: Counter = Counter(b for _, b in pares)
    conteo_xy: Counter = Counter(pares)
    hx = _entropia(conteo_x, n)
    hy = _entropia(conteo_y, n)
    mi = 0.0
    for (a, b), c in conteo_xy.items():
        pxy = c / n
        px = conteo_x[a] / n
        py = conteo_y[b] / n
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log2(pxy / (px * py))
    return mi, hx, hy


def _colapsar_categorias_raras(valores: Sequence[Any], soporte_minimo: int) -> tuple[list[Any], bool]:
    """Las categorías con menos de `soporte_minimo` casos se funden en
    `"__otras__"` ANTES de medir. Es lo que borra el sesgo por cardinalidad de
    la NMI: medido en el docstring del módulo, un identificador de
    cardinalidad ≈ n cae de NMI ≈ 0,22 a NMI = 0,0 tras esta operación, porque
    todas sus categorías (soporte 1) se funden en una sola."""
    conteo = Counter(valores)
    colapsado = False
    salida = []
    for v in valores:
        if conteo[v] < soporte_minimo:
            salida.append("__otras__")
            colapsado = True
        else:
            salida.append(v)
    return salida, colapsado


def nmi_categorica(valores_a: Sequence[Any], valores_b: Sequence[Any], *,
                   soporte_minimo: int = SOPORTE_MINIMO_PUREZA) -> dict[str, Any]:
    """Información mutua NORMALIZADA entre dos variables categóricas.

    **Método de normalización, declarado (invariante 9):** media aritmética —
    `NMI = 2·I(X;Y) / (H(X) + H(Y))`, en bits. Es el mismo que
    `sklearn.metrics.normalized_mutual_info_score(average_method="arithmetic")`
    (la prueba de paridad lo contrasta), y se declara el nombre para que quien
    audite sepa que existen otras normalizaciones (mínimo, máximo, geométrica)
    que dan números distintos sobre los MISMOS datos — mezclar umbrales de dos
    normalizaciones sería mezclar dos estadísticos distintos con el mismo
    nombre.

    **Sesgo por cardinalidad, declarado y mitigado:** antes de medir, las
    categorías con menos de `soporte_minimo` casos se agrupan en una sola
    (`_colapsar_categorias_raras`). Sin esto, una columna de cardinalidad alta
    (un identificador) infla la NMI por pura combinatoria — medido en el
    docstring del módulo. `categorias_colapsadas` dice si esto ha ocurrido, y
    `cardinalidad_x`/`cardinalidad_y` son las que quedan DESPUÉS de colapsar.
    """
    n = len(valores_a)
    a_colapsado, colapso_a = _colapsar_categorias_raras(valores_a, soporte_minimo)
    b_colapsado, colapso_b = _colapsar_categorias_raras(valores_b, soporte_minimo)
    pares = list(zip(a_colapsado, b_colapsado))
    mi, hx, hy = _informacion_mutua(pares)
    denominador = (hx + hy) / 2.0
    valor = (mi / denominador) if denominador > 0 else 0.0
    return {
        "valor": round(valor, 6),
        "metodo": "arithmetic: 2*I(X;Y)/(H(X)+H(Y)), en bits (log2)",
        "n": n,
        "soporte_minimo": soporte_minimo,
        "categorias_colapsadas": colapso_a or colapso_b,
        "cardinalidad_x": len(set(a_colapsado)),
        "cardinalidad_y": len(set(b_colapsado)),
    }


def intervalo_wilson(eventos: int, n: int, *, z: float = _Z_95) -> tuple[float, float] | None:
    """Centro y SEMIANCHURA del intervalo de Wilson para una proporción —
    stdlib puro (`math.sqrt`), sin necesitar la beta incompleta de una
    distribución t. `None` si `n <= 0` o `eventos` no está en `[0, n]`."""
    if n <= 0 or eventos < 0 or eventos > n:
        return None
    phat = eventos / n
    denom = 1.0 + z * z / n
    centro = (phat + z * z / (2 * n)) / denom
    semiancho = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centro, semiancho


def _mayoria(conteo: Counter) -> tuple[Any, int]:
    return max(conteo.items(), key=lambda kv: kv[1])


def _purezas_simples(categorias: Sequence[Any], objetivos: Sequence[Any],
                     soporte_minimo: int) -> dict[Any, float]:
    grupos: dict[Any, Counter] = {}
    for c, o in zip(categorias, objetivos):
        grupos.setdefault(c, Counter())[o] += 1
    salida: dict[Any, float] = {}
    for categoria, conteo in grupos.items():
        total = sum(conteo.values())
        if total < soporte_minimo:
            continue
        _, mayoria = _mayoria(conteo)
        salida[categoria] = mayoria / total
    return salida


def pureza_categorica(valores_categoria: Sequence[Any], valores_objetivo: Sequence[Any], *,
                      soporte_minimo: int = SOPORTE_MINIMO_PUREZA) -> dict[str, Any]:
    """Cuánto predice cada categoría al objetivo, DENTRO Y FUERA de muestra.

    **Por qué esto y no solo `max(conteo) / total` (criterio de terminado
    #3).** Medido en el docstring del módulo: sobre un CSV de 300 filas y 40
    categorías SIN relación real con el objetivo, mirar solo la pureza
    in-sample con soporte mínimo 1 encuentra una categoría "pura" (≥ 0,85) en
    el 98 % de las repeticiones. Un identificador de alta cardinalidad
    —cada categoría con una sola fila— es el caso extremo: pureza 1,0
    SIEMPRE, porque cada bucket "acierta" trivialmente prediciéndose a sí
    mismo.

    Dos defensas, las dos necesarias (medidas por separado):

    1. **Soporte mínimo.** Una categoría con menos de `soporte_minimo` filas
       no se evalúa — con eso solo, la tasa de falsos positivos del escenario
       de arriba cae de 98 % a 0 % (medido). El identificador de cardinalidad
       alta queda automáticamente SIN categorías evaluables: no es que se
       excluya a propósito, es que ninguna alcanza el soporte.
    2. **Evidencia fuera de muestra.** Con MUCHAS categorías (cardinalidad
       alta pero soporte suficiente en cada una — el caso adversarial de
       comparaciones múltiples), el soporte mínimo solo no basta: medido con
       1.000 categorías y ~22 filas cada una, la tasa de falsos positivos
       in-sample sube a un 32 % según crece la cardinalidad. Exigir que la
       MISMA categoría tenga pureza alta en dos mitades INTERNAS de los datos
       (partición por paridad de índice — nunca el test reservado del
       protocolo, que aquí no existe todavía) la reduce a un 11 % en ese
       mismo escenario extremo. **Esto es una reducción, no una eliminación**:
       con cardinalidad suficientemente alta el riesgo de falso positivo por
       comparaciones múltiples no desaparece del todo, y eso se deja escrito
       aquí en vez de prometer una garantía que la medida no sostiene.
    """
    n = len(valores_categoria)
    evaluadas: dict[Any, dict[str, Any]] = {}
    grupos: dict[Any, Counter] = {}
    for c, o in zip(valores_categoria, valores_objetivo):
        grupos.setdefault(c, Counter())[o] += 1
    for categoria, conteo in grupos.items():
        total = sum(conteo.values())
        if total < soporte_minimo:
            continue
        _, mayoria = _mayoria(conteo)
        evaluadas[categoria] = {"pureza": round(mayoria / total, 6), "n": total}

    mitad_a_cat: list[Any] = []
    mitad_a_obj: list[Any] = []
    mitad_b_cat: list[Any] = []
    mitad_b_obj: list[Any] = []
    for i, (c, o) in enumerate(zip(valores_categoria, valores_objetivo)):
        if i % 2 == 0:
            mitad_a_cat.append(c)
            mitad_a_obj.append(o)
        else:
            mitad_b_cat.append(c)
            mitad_b_obj.append(o)
    soporte_mitad = max(1, soporte_minimo // 2)
    purezas_a = _purezas_simples(mitad_a_cat, mitad_a_obj, soporte_mitad)
    purezas_b = _purezas_simples(mitad_b_cat, mitad_b_obj, soporte_mitad)
    confirmadas = tuple(sorted(
        (str(c) for c in evaluadas
         if c in purezas_a and c in purezas_b
         and purezas_a[c] >= UMBRAL_PUREZA_ALTA_FUERA_DE_MUESTRA
         and purezas_b[c] >= UMBRAL_PUREZA_ALTA_FUERA_DE_MUESTRA),
        key=str))

    return {
        "n": n, "soporte_minimo": soporte_minimo,
        "categorias_evaluadas": evaluadas,
        "confirmadas_fuera_de_muestra": confirmadas,
    }


# ---------------------------------------------------------------------------
# Comparación exacta de valores — para los duplicados (bloqueo y sospecha)
# ---------------------------------------------------------------------------
#
# `igualdad_de_valores` se llama sobre TODOS los pares de predictores
# (O(k²)): medido sobre 50.000 filas × 29 predictores, la versión que repetía
# `_is_null`/`strip`/`float()` dentro del cruce tardaba 33,6 s, y un
# `cProfile` señaló la causa exacta — `float()` fallando con `ValueError` en
# cada comparación no numérica, dentro de un bucle de 21,75 millones de
# llamadas: las excepciones de Python no son gratis. `_preparar_columna`
# hace ese trabajo UNA VEZ por columna (parseo, nulo, numérico-o-texto) y
# `_comparar_preparadas` cruza dos columnas ya preparadas con comparaciones
# directas — la MISMA semántica (hay una prueba que lo contrasta contra
# `_coincidencia_exacta`), 30-40× más rápido en el mismo banco.

def _preparar_columna(valores: Sequence[str]) -> tuple[list[bool], list[bool], list[Any]]:
    """`(es_nulo, es_numero, normalizado)`, UNA pasada por columna."""
    from matrixai.training.dataset_analysis import _is_null  # noqa: PLC0415

    n = len(valores)
    es_nulo = [False] * n
    es_numero = [False] * n
    normalizado: list[Any] = [None] * n
    for i, v in enumerate(valores):
        if _is_null(v):
            es_nulo[i] = True
            continue
        s = str(v).strip()
        try:
            normalizado[i] = float(s)
            es_numero[i] = True
        except ValueError:
            normalizado[i] = s
    return es_nulo, es_numero, normalizado


def _comparar_preparadas(
    prep_a: tuple[list[bool], list[bool], list[Any]],
    prep_b: tuple[list[bool], list[bool], list[Any]],
) -> tuple[int, int]:
    """`(coincidencias, filas_comparadas)` entre dos columnas YA preparadas.

    Un nulo en cualquiera de los dos lados no cuenta ni a favor ni en contra
    — un valor ausente no es un cero y no puede confirmar ni descartar una
    igualdad. Dos tipos distintos (uno numérico, el otro texto) en la misma
    fila nunca coinciden: si de verdad fueran el mismo valor, los dos habrían
    parseado igual.
    """
    nulos_a, num_a, val_a = prep_a
    nulos_b, num_b, val_b = prep_b
    coincidencias = 0
    comparadas = 0
    for i in range(len(val_a)):
        if nulos_a[i] or nulos_b[i]:
            continue
        comparadas += 1
        if num_a[i] and num_b[i]:
            if abs(val_a[i] - val_b[i]) < 1e-9:
                coincidencias += 1
        elif not num_a[i] and not num_b[i] and val_a[i] == val_b[i]:
            coincidencias += 1
    return coincidencias, comparadas


def _coincidencia_exacta(valores_a: Sequence[str], valores_b: Sequence[str]) -> tuple[int, int]:
    """`(coincidencias, filas_comparadas)` — prepara y compara UN par. Para
    cruzar MUCHAS columnas entre sí, preparar cada una con `_preparar_columna`
    una sola vez y llamar a `_comparar_preparadas` por cada cruce (lo que hace
    `diagnosticar_csv`) evita repetir el parseo O(k) veces por columna."""
    return _comparar_preparadas(_preparar_columna(valores_a), _preparar_columna(valores_b))


# ---------------------------------------------------------------------------
# Errores estructurales — Bloqueo (invariante 2, no se levantan aceptándolos)
# ---------------------------------------------------------------------------

def objetivo_duplicado(valores_objetivo: Sequence[str], valores_predictor: Sequence[str], *,
                       objetivo: str, predictor: str, minimo_filas: int = 20) -> Bloqueo | None:
    """`Bloqueo` si `predictor` reproduce `objetivo` fila a fila.

    Es DATOS, no nombre: a diferencia de `objetivo.pistas_por_nombre` (103-C1,
    que mira los NOMBRES y nunca bloquea), esto compara VALORES y solo se
    confirma con coincidencia TOTAL sobre al menos `minimo_filas` filas
    comparables — una coincidencia parcial, por pequeña que sea la diferencia,
    es la asociación alta de `asociacion_muy_alta`, no una identidad.
    """
    coincidencias, comparadas = _coincidencia_exacta(valores_objetivo, valores_predictor)
    if comparadas < minimo_filas or coincidencias != comparadas:
        return None
    return Bloqueo(
        clave="objetivo_duplicado_confirmado", campo=predictor,
        motivo=motivo("objetivo_duplicado_confirmado", campo=repr(predictor),
                      valor=repr(objetivo), opciones=comparadas))


def variables_disponibles_tras_el_desenlace(problema: ProblemSpec) -> tuple[Bloqueo, ...]:
    """`Bloqueo` por cada predictor DECLARADO `after_outcome`.

    Lee `problema.predictor_availability` tal cual — nunca infiere
    disponibilidad por nombre, orden del CSV o correlación (invariante 5: eso
    está prohibido). Si nadie ha declarado nada, `predictor_availability` está
    vacío y esta función no bloquea nada — la ausencia de declaración no es
    una declaración de disponibilidad.
    """
    return tuple(
        Bloqueo(clave="variable_disponible_tras_el_desenlace", campo=predictor,
                motivo=motivo("variable_disponible_tras_el_desenlace", campo=repr(predictor)))
        for predictor, cuando in problema.predictor_availability.items()
        if cuando == "after_outcome")


def cruce_de_unidades(predictores: Sequence[str], columna_unidad: str | None, *,
                      unidad_de_observacion: str | None = None) -> Bloqueo | None:
    """`Bloqueo` si la columna que identifica la UNIDAD de observación
    (paciente, cliente, centro…) se declara TAMBIÉN como entrada.

    Simétrico a `objetivo_entre_las_entradas` del 103-C1, pero para el otro
    identificador que este programa protege (invariante 8: «IDs de unidad/
    grupo se conservan como metadatos de partición aunque no entren como
    predictores»): cruzar los dos papeles deja que el modelo memorice de qué
    sujeto es cada fila, y además —cuando el 103-C4 construya el `SplitPlan`—
    esa misma columna es la que tiene que separar sujetos entre desarrollo y
    prueba; si además viaja como entrada, la identidad del sujeto se cuela por
    la puerta de atrás aunque la partición en sí esté bien hecha.

    `columna_unidad` es OPCIONAL a propósito: en la ruta prompt, o en un CSV
    transversal sin grupos, puede no haber ninguna columna así, y no hay nada
    que cruzar. No se INFIERE cuál es (el invariante 5 lo prohibiría igual):
    quien confirma el diseño la declara, igual que `unit_id_field` en
    `SplitPlan` (104-C0) — incluso es el mismo nombre de campo a propósito,
    para que el 103-C4 pueda reutilizar la misma declaración.
    """
    if columna_unidad is None or columna_unidad not in predictores:
        return None
    return Bloqueo(
        clave="cruce_de_unidades_prohibido", campo=columna_unidad,
        motivo=motivo("cruce_de_unidades_prohibido", campo=repr(columna_unidad),
                      valor=unidad_de_observacion or columna_unidad))


# ---------------------------------------------------------------------------
# Sospechas — exigen contexto, nunca bloquean (invariante 3)
# ---------------------------------------------------------------------------

def igualdad_de_valores(valores_a: Sequence[str], valores_b: Sequence[str], *,
                        columna_a: str, columna_b: str, minimo_filas: int = 20) -> Sospecha | None:
    """`Sospecha` si DOS ENTRADAS (nunca el objetivo — eso es
    `objetivo_duplicado`, y es un `Bloqueo`) coinciden fila a fila.

    Es SOSPECHA y no bloqueo a propósito: dos columnas iguales pueden ser dos
    nombres de la misma medida (celsius y centígrados), un cálculo derivado
    legítimo, o una fuga por una tercera variable — el core no puede saber
    cuál sin que alguien lo diga, y bloquear aquí sería inventar un motivo que
    nadie ha confirmado.
    """
    coincidencias, comparadas = _coincidencia_exacta(valores_a, valores_b)
    if comparadas < minimo_filas or coincidencias != comparadas:
        return None
    return Sospecha(
        clave="igualdad_de_valores_sin_explicar", campo=columna_a,
        motivo=motivo("igualdad_de_valores_sin_explicar", campo=repr(columna_a),
                      valor=repr(columna_b), opciones=comparadas),
        medida={"columna_a": columna_a, "columna_b": columna_b, "filas_comparadas": comparadas})


def asociacion_muy_alta(valores_x: Sequence[Any], valores_y: Sequence[Any], *,
                        columna: str, metodo: str = "pearson") -> Sospecha | None:
    """`Sospecha` cuando `|r| >= UMBRAL_ASOCIACION_ALTA` Y es estadísticamente
    significativo (`_r_critico`). Las dos condiciones son necesarias: sin la
    segunda, un `r` de 0,9 calculado sobre 4 filas pasaría el umbral de
    magnitud siendo puro ruido de muestra pequeña.

    **NUNCA es un `Bloqueo`.** Es la corrección literal que este corte existe
    para hacer: una correlación alta —incluida la 0,955 del salario del 71,
    medida en la prueba de este módulo— no demuestra fuga por sí sola.
    """
    medida = asociacion(valores_x, valores_y, metodo=metodo)
    if medida is None:
        return None
    r = medida["valor"]
    critico = medida["r_critico_1pct"]
    if abs(r) < UMBRAL_ASOCIACION_ALTA:
        return None
    if critico is not None and abs(r) < critico:
        return None
    return Sospecha(
        clave="asociacion_muy_alta", campo=columna,
        motivo=motivo("asociacion_muy_alta", campo=repr(columna), valor=round(r, 4),
                      opciones=metodo, n=medida["n"]),
        medida=medida)


def proxy_por_nmi(valores_categoria: Sequence[Any], valores_objetivo: Sequence[Any], *,
                  columna: str, soporte_minimo: int = SOPORTE_MINIMO_PUREZA) -> Sospecha | None:
    """`Sospecha` cuando la NMI (ya corregida por soporte mínimo — ver
    `nmi_categorica`) supera `UMBRAL_NMI_ALTA`."""
    medida = nmi_categorica(valores_categoria, valores_objetivo, soporte_minimo=soporte_minimo)
    if medida["valor"] < UMBRAL_NMI_ALTA:
        return None
    return Sospecha(
        clave="proxy_probable", campo=columna,
        motivo=motivo("proxy_probable", campo=repr(columna), valor=medida["valor"],
                      opciones="NMI aritmética", n=medida["n"], minimo=soporte_minimo),
        medida=medida)


def proxy_por_pureza(valores_categoria: Sequence[Any], valores_objetivo: Sequence[Any], *,
                     columna: str, soporte_minimo: int = SOPORTE_MINIMO_PUREZA) -> Sospecha | None:
    """`Sospecha` cuando alguna categoría tiene pureza alta Y CONFIRMADA fuera
    de muestra (criterio de terminado #3: nunca solo por la pureza in-sample).
    """
    medida = pureza_categorica(valores_categoria, valores_objetivo, soporte_minimo=soporte_minimo)
    candidatas = [c for c, info in medida["categorias_evaluadas"].items()
                 if info["pureza"] >= UMBRAL_PUREZA_ALTA and str(c) in medida["confirmadas_fuera_de_muestra"]]
    if not candidatas:
        return None
    categoria = sorted(candidatas, key=str)[0]
    detalle = medida["categorias_evaluadas"][categoria]
    return Sospecha(
        clave="categoria_predictiva_por_pureza", campo=columna,
        motivo=motivo("categoria_predictiva_por_pureza", campo=repr(columna),
                      valor=repr(categoria), opciones=detalle["pureza"], n=detalle["n"]),
        medida={**medida, "categoria": categoria})


def identificador_probable(columna: str, *, tipo_columna: str | None = None,
                           cardinalidad: int | None = None) -> Sospecha | None:
    """`Sospecha` si la columna PARECE un identificador — por tipo (lo que ya
    calculó `dataset_analysis._analyze_column` sobre los VALORES, contrato 57)
    o por nombre (`parece_identificador`, contrato 71, para cuando no hay CSV
    que analizar). NUNCA bloquea, ni se quita solo: un identificador puede
    llevar información real (un código que agrupa familias), y contrato 71 ya
    dejó escrito que se avisa, no se quita.
    """
    from matrixai.training.dense_generator import parece_identificador  # noqa: PLC0415

    por_tipo = tipo_columna == "identifier"
    por_nombre = parece_identificador(columna)
    if not (por_tipo or por_nombre):
        return None
    evidencia = "tipo de columna" if por_tipo else "nombre"
    return Sospecha(
        clave="identificador_probable", campo=columna,
        motivo=motivo("identificador_probable", campo=repr(columna), valor=evidencia),
        medida={"evidencia": evidencia, "tipo_columna": tipo_columna, "cardinalidad": cardinalidad})


# ---------------------------------------------------------------------------
# Límites de estimación — NO son fugas, acotan la conclusión (invariante 4)
# ---------------------------------------------------------------------------

def falta_de_eventos(valores_objetivo: Sequence[str], *, objetivo: str) -> tuple[Limite, ...]:
    """`Limite` por cada clase con menos de `UMBRAL_EVENTOS_MINIMOS` casos.

    Una clase con CERO casos no es esto: es `objetivo_con_una_sola_clase`
    (103-C1, un `Bloqueo` — no hay nada que aprender de una clase ausente) o,
    si la clase se DECLARÓ pero no aparece, `clases_declaradas_que_no_estan`.
    Aquí es POCOS casos, no ninguno: la clase SÍ está, pero cualquier métrica
    que dependa de ella tiene un error de muestreo demasiado grande para
    leerse como una cifra precisa.
    """
    from matrixai.training.dataset_analysis import _is_null  # noqa: PLC0415

    conteo = Counter(str(v).strip() for v in valores_objetivo if not _is_null(v))
    return tuple(
        Limite(clave="falta_de_eventos", campo=objetivo,
               motivo=motivo("falta_de_eventos", campo=repr(objetivo), valor=cuenta,
                             opciones=repr(clase)),
               medida={"clase": clase, "casos": cuenta, "umbral": UMBRAL_EVENTOS_MINIMOS})
        for clase, cuenta in sorted(conteo.items())
        if 0 < cuenta < UMBRAL_EVENTOS_MINIMOS)


def tamano_efectivo_insuficiente(filas_utiles: int, predictores: Sequence[str]) -> Limite | None:
    """`Limite` si hay menos de `RATIO_MINIMO_FILAS_POR_PREDICTOR` filas útiles
    por cada predictor declarado — grados de libertad, no un umbral prestado."""
    n_predictores = max(1, len(predictores))
    ratio = filas_utiles / n_predictores
    if ratio >= RATIO_MINIMO_FILAS_POR_PREDICTOR:
        return None
    return Limite(
        clave="tamano_efectivo_insuficiente", campo=None,
        motivo=motivo("tamano_efectivo_insuficiente", campo=filas_utiles,
                      valor=len(predictores), opciones=round(ratio, 2)),
        medida={"filas_utiles": filas_utiles, "predictores": len(predictores),
                "ratio": round(ratio, 4), "umbral": RATIO_MINIMO_FILAS_POR_PREDICTOR})


def precision_insuficiente(eventos: int, n: int, *, campo: str) -> Limite | None:
    """`Limite` si el margen del intervalo de Wilson al 95 % es demasiado
    grande RELATIVO a la propia proporción estimada (no en términos absolutos:
    un margen de 0,01 sobre una proporción de 0,001 es enorme en relativo y
    minúsculo en absoluto, y es el relativo el que dice si la cifra sirve)."""
    intervalo = intervalo_wilson(eventos, n)
    if intervalo is None:
        return None
    _, semiancho = intervalo
    proporcion = eventos / n
    if proporcion <= 0:
        return None
    margen_relativo = semiancho / proporcion
    if margen_relativo < UMBRAL_MARGEN_RELATIVO_WILSON:
        return None
    return Limite(
        clave="precision_insuficiente", campo=campo,
        motivo=motivo("precision_insuficiente", campo=repr(campo), valor=round(proporcion, 4),
                      n=n, opciones=f"±{round(semiancho, 4)}"),
        medida={"eventos": eventos, "n": n, "proporcion": round(proporcion, 6),
                "semiancho_wilson_95": round(semiancho, 6),
                "margen_relativo": round(margen_relativo, 4),
                "umbral_margen_relativo": UMBRAL_MARGEN_RELATIVO_WILSON})


def ausencia_de_linaje(predictores: Sequence[str]) -> Limite:
    """El `Limite` que SIEMPRE está presente (criterio de terminado #4): la
    ausencia de información de linaje se MUESTRA como limitación, nunca como
    ausencia de problema. Ningún detector de este módulo ve cómo se calculó
    una columna externa — si viene del objetivo, de otra unidad, o de un
    momento posterior al declarado —, y eso se declara aquí en vez de dejarlo
    implícito en que ningún otro detector haya saltado.
    """
    columnas = tuple(sorted(set(str(p) for p in predictores)))
    return Limite(
        clave="ausencia_de_linaje", campo=None,
        motivo=motivo("ausencia_de_linaje", campo=", ".join(columnas) or "—",
                      valor=len(columnas)),
        medida={"columnas": list(columnas)})


# ---------------------------------------------------------------------------
# El orquestador — un sitio que corre TODOS los detectores sobre un CSV
# ---------------------------------------------------------------------------

def _muestrear(filas: Sequence[Mapping[str, Any]], techo: int) -> list[Mapping[str, Any]]:
    """Una muestra determinista y UNIFORME por posición (zancada fija), no las
    primeras `techo` filas: un CSV ordenado por el objetivo dejaría la muestra
    entera del lado de una sola clase, igual que le pasa al corte de
    entrenamiento del 103-C1 — aquí no hace falta imitar ESE corte (el
    diagnóstico no entrena), pero sí evitar el mismo sesgo."""
    total = len(filas)
    if total <= techo or techo <= 0:
        return list(filas)
    zancada = total / techo
    indices = sorted({int(i * zancada) for i in range(techo)})
    return [filas[i] for i in indices]


def diagnosticar_csv(
    csv_text: str,
    problema: ProblemSpec,
    *,
    analisis: Mapping[str, Any] | None = None,
    filas: Sequence[Mapping[str, Any]] | None = None,
    columna_unidad: str | None = None,
    muestreo_maximo: int = MUESTREO_MAXIMO_FILAS,
) -> Diagnostico:
    """Corre TODOS los detectores de este corte sobre un CSV y un problema ya
    confirmado, y devuelve las tres listas separadas.

    `analisis`/`filas` son el resultado de `analyze_dataset_csv`/`_read_rows`,
    para quien ya los tiene (mismo motivo que en `objetivo.confirmar_desde_csv`:
    no repetir una pasada entera sobre un CSV que puede tener cientos de miles
    de filas).

    LO QUE ESTE ORQUESTADOR DEJA FUERA A PROPÓSITO, y por qué: la asociación
    numérica (Pearson/Spearman) solo se calcula para regresión y para la
    clase positiva de una clasificación BINARIA — una correlación ordinal
    entre un predictor numérico y una clase de una MULTICLASE sin orden no
    tiene un signo que interpretar, y fabricar uno sería inventar una
    magnitud que el problema no tiene. Esa comparación multiclase-numérica
    queda como hueco declarado, no resuelto en silencio; sigue disponible
    `proxy_por_nmi`/`proxy_por_pureza`, que no asumen orden.
    """
    from matrixai.training.dataset_analysis import _is_null, analyze_dataset_csv  # noqa: PLC0415
    from matrixai.training.dataset_project import _read_rows  # noqa: PLC0415

    if analisis is None:
        analisis = analyze_dataset_csv(csv_text)
    if filas is None:
        filas = _read_rows(csv_text)

    columnas_info: dict[str, Any] = dict(analisis.get("columns") or {})
    objetivo = problema.target
    predictores = [p for p in problema.predictors if p != objetivo]

    filas_totales = len(filas)
    muestreado = filas_totales > muestreo_maximo
    filas_medidas = _muestrear(filas, muestreo_maximo) if muestreado else list(filas)

    valores_objetivo = [str(row.get(objetivo) or "") for row in filas_medidas]
    valores_por_columna: dict[str, list[str]] = {
        p: [str(row.get(p) or "") for row in filas_medidas] for p in predictores}

    bloqueos: list[Bloqueo] = list(variables_disponibles_tras_el_desenlace(problema))
    cruce = cruce_de_unidades(predictores, columna_unidad,
                              unidad_de_observacion=problema.observation_unit)
    if cruce is not None:
        bloqueos.append(cruce)
    for predictor in predictores:
        b = objetivo_duplicado(valores_objetivo, valores_por_columna[predictor],
                               objetivo=objetivo, predictor=predictor)
        if b is not None:
            bloqueos.append(b)

    sospechas: list[Sospecha] = []
    for i, col_a in enumerate(predictores):
        for col_b in predictores[i + 1:]:
            s = igualdad_de_valores(valores_por_columna[col_a], valores_por_columna[col_b],
                                    columna_a=col_a, columna_b=col_b)
            if s is not None:
                sospechas.append(s)

    es_binaria = problema.task == "binary_classification" and problema.positive_label is not None
    for predictor in predictores:
        info = columnas_info.get(predictor) or {}
        tipo = info.get("type")
        cardinalidad = info.get("cardinality")

        idn = identificador_probable(predictor, tipo_columna=tipo, cardinalidad=cardinalidad)
        if idn is not None:
            sospechas.append(idn)

        if tipo in ("integer", "number"):
            if problema.task == "regression":
                s = asociacion_muy_alta(valores_por_columna[predictor], valores_objetivo,
                                        columna=predictor, metodo="pearson")
                if s is not None:
                    sospechas.append(s)
            elif es_binaria:
                indicador = [1.0 if v.strip() == problema.positive_label else 0.0
                            for v in valores_objetivo]
                s = asociacion_muy_alta(valores_por_columna[predictor], indicador,
                                        columna=predictor, metodo="pearson")
                if s is not None:
                    sospechas.append(s)
        elif problema.task != "regression":
            s = proxy_por_nmi(valores_por_columna[predictor], valores_objetivo, columna=predictor)
            if s is not None:
                sospechas.append(s)
            s = proxy_por_pureza(valores_por_columna[predictor], valores_objetivo, columna=predictor)
            if s is not None:
                sospechas.append(s)

    limites: list[Limite] = [ausencia_de_linaje(predictores)]
    if problema.task != "regression":
        limites.extend(falta_de_eventos(valores_objetivo, objetivo=objetivo))
    t = tamano_efectivo_insuficiente(len(filas_medidas), predictores)
    if t is not None:
        limites.append(t)
    if es_binaria:
        eventos = sum(1 for v in valores_objetivo
                      if not _is_null(v) and v.strip() == problema.positive_label)
        p = precision_insuficiente(eventos, len(valores_objetivo), campo=objetivo)
        if p is not None:
            limites.append(p)

    return Diagnostico(
        bloqueos=tuple(bloqueos), sospechas=tuple(sospechas), limites=tuple(limites),
        muestreado=muestreado, filas_medidas=len(filas_medidas), filas_totales=filas_totales)
