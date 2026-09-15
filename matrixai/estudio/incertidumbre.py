# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""La incertidumbre según el DISEÑO y el ESTIMANDO — 105-C2.

`metricas.py` (105-C1) da un número, o la razón de que no lo haya. Este módulo
da lo segundo que pide el contrato: **cuánto podría variar ese número** si se
repitiera la medición sobre otra muestra de la misma población, con el MISMO
modelo ya fijado. Es el invariante 6 del 105 hecho código: el IC de un modelo
fijo, la variación de volver a entrenar y la tolerancia numérica NO son la
misma cosa, y aquí solo se calcula la primera — las otras dos esperan a C3/C4
y a 104-C0, respectivamente.

QUÉ RESUELVE, LITERAL DEL CONTRATO: bootstrap **emparejado por observaciones**
—se remuestrean FILAS ENTERAS, nunca la verdad y la predicción por separado,
que es justo lo que «emparejado» significa aquí—, con semilla y número de
remuestras registrados en el resultado; remuestreo **por unidad** cuando el
diseño es de grupos, no por fila; bloques para el diseño temporal; y **nunca
IID en silencio** cuando el método no está soportado.

LAS CUATRO DECISIONES DE DISEÑO, elegidas y justificadas aquí porque el
contrato pide exactamente eso y no un menú de opciones que alguien pueda
activar sin pensarlo:

1. **IID en clasificación se estratifica por la etiqueta observada.** Cada
   remuestra conserva el recuento EXACTO de cada clase de la muestra
   original —no la proporción esperada, el recuento exacto—, porque AUROC,
   AP y el trapecio (105-C1) se declaran indefinidos sin las dos clases
   presentes, y sin estratificar, una remuestra de una población con eventos
   raros podría omitir por puro azar la única fila positiva. El intervalo
   resultante sería un artefacto del remuestreo, no una descripción del
   estimando. **IID en regresión no estratifica**: no hay clases discretas
   que preservar.
2. **Una remuestra que sale degenerada para ESA métrica —una sola clase para
   AUROC/AP, varianza nula para R², un denominador vacío para PPV— se
   EXCLUYE del percentil**, no se rellena con nada. Si la proporción de
   remuestras válidas cae por debajo de `PROPORCION_MINIMA_VALIDA`, el
   intervalo ENTERO se declara indefinido: un percentil calculado sobre un
   puñado de remuestras válidas describiría el ruido del remuestreo, no el
   estimando, y eso es exactamente lo que el criterio de terminado pide
   poder ver en una muestra de eventos raros.
3. **Grupos remuestrea UNIDADES completas, con reemplazo** —cada unidad
   elegida aporta TODAS sus filas—, y no se estratifica por clase: la unidad
   YA es la estratificación del diseño (es lo que hay que preservar entero
   para no romper la dependencia dentro de cada paciente/sujeto), y forzar
   además un balance de clases entre unidades que traen clases mezcladas
   rompería precisamente la correlación que el diseño por grupos existe para
   respetar.
4. **Temporal usa bloques circulares** (block bootstrap, Politis-Romano):
   se cortan tramos contiguos de longitud fija y se pegan hasta completar el
   tamaño original, envolviendo al llegar al final. La longitud por omisión
   es la heurística de cordura `n**(1/3)` —no el algoritmo de selección
   automática de Politis-White, que exige estimar el espectro y este core no
   lo hace—, y se puede declarar explícitamente. **Precondición que este
   módulo NO puede comprobar**: se asume que las filas de la muestra ya
   vienen en orden cronológico, que es el mismo supuesto que hace
   `SplitPlan.time_column` en el 104-C0; ni `Muestra` ni `PredictionRecord`
   llevan un campo de tiempo, así que reordenar aquí sería fabricar un dato
   que el core no tiene.

QUÉ NO ESTÁ SOPORTADO, Y SE DICE POR QUÉ (nunca IID en silencio):
`groups_and_time` no tiene aquí un método propio —combinar bloques temporales
con dependencia de grupo pide una implementación distinta, no una mezcla
apresurada de las otras dos— y pedirlo devuelve un intervalo NO DISPONIBLE con
su motivo, igual que pedir una métrica que no está en el catálogo de C1.

LAS REPETICIONES Y LOS PLIEGUES NO SON OBSERVACIONES NUEVAS. Si una muestra
sale de predicciones fuera de pliegue (OOF) con varias repeticiones de
validación cruzada, la fila de cada repetición sigue siendo LA MISMA
observación de base, y tratarlas como independientes en un IID inflaría el
tamaño efectivo de muestra. Por eso, si `Muestra.units` declara menos unidades
que filas, este módulo **rechaza el diseño IID con excepción** —es un error de
cableado de quien pide el intervalo, no un dato degenerado— y exige el diseño
`groups`, remuestreando por unidad: con `resampling_unit` puesto al id de la
observación base en cada repetición (104-C0 ya lo permite), remuestrear por
unidad respeta la dependencia entre repeticiones sin inventar nada.

`estimando` hace explícito el resto de esa misma regla: un intervalo sobre
predicciones OOF **agrupadas** (todas las filas de todos los pliegues en una
sola muestra, que es justo lo que `Muestra.desde_registros` ya construye) no
es lo mismo que uno sobre el **promedio de las métricas por pliegue**, y son
estimandos distintos que no se equiparan. Este módulo implementa el primero
—`oof_pooled_estimate`— reutilizando el mismo mecanismo de remuestreo por
unidad que ya sirve al diseño de grupos; el segundo, `oof_mean_of_folds`,
necesitaría resamplear PLIEGUES y promediar SUS métricas, un mecanismo
distinto que este corte no implementa y que por eso se declara fuera en vez
de calcularlo a medias (ver el informe del corte).

LA EVALUACIÓN DEL PROCESO COMPLETO NO ES ESTO. Lo que hay aquí es un bootstrap
de **predicciones ya guardadas**, de un modelo YA AJUSTADO: no vuelve a
entrenar nada, así que no puede decir cuánto variaría el resultado si se
repitiera el AJUSTE con otra muestra de desarrollo —esa variación de
entrenamiento («training_variability», 105 invariante 6) exige el diseño de
104-C0 y queda fuera de este módulo a propósito.

EL COSTE. Antes de tocar el número de filas —que el contrato prohíbe reducir
en silencio— se evita el trabajo redundante que sí se puede evitar: cada
remuestra reutiliza la fórmula YA registrada en C1 (`REGISTRO[...].formula`,
público) sobre una `Muestra` reconstruida SIN volver a pasar por
`__post_init__`. Es seguro: cada comprobación de `__post_init__` es una
propiedad POR FILA (pertenece a `classes`, está en `[0,1]`, suma 1) o una
propiedad agregada que no cambia al repetir filas ya válidas (`classes`,
`positive_label`, `task`). Repetir con reemplazo filas que ya pasaron esa
comprobación una vez no puede producir una fila inválida, así que
revalidar en cada una de las `n_resamples` iteraciones sería pagar otra vez
un coste que la primera medición ya pagó. Los tiempos medidos (1.000
remuestras sobre 1.000 / 10.000 / 100.000 filas) están en el informe del
corte, no en el código. Un modo más rápido es simplemente pedir menos
`n_resamples` — explícito en la llamada y declarado en el resultado
(`n_resamples`), nunca una submuestra de filas decidida en silencio por
tardar más de diez segundos.

Y sigue siendo **stdlib puro**: `math`, `random`, `time`, `dataclasses`. Nada
de `numpy` aquí; las referencias (si las hay) viven en las pruebas.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.metricas import REGISTRO, EntradaNoMedible, Indefinida, Muestra, calcular
from matrixai.estudio.esquemas import ValorDeMetrica
from matrixai.estudio.textos import motivo
from matrixai.estudio.validacion import (
    exigir_entero,
    exigir_entero_o_nulo,
    exigir_motivo_bilingue,
    exigir_real,
    exigir_real_o_nulo,
    exigir_texto,
)
from matrixai.estudio.vocabulario import TAREAS_DE_CLASIFICACION, TIPOS_DE_PARTICION, exigir_opcion

__all__ = [
    "DISENOS_SOPORTADOS", "ESTIMANDOS_DE_INTERVALO", "METODOS_DE_REMUESTREO",
    "PROPORCION_MINIMA_VALIDA", "UNIDADES_DE_REMUESTREO",
    "Intervalo", "intervalo", "medir",
]

#: Los diseños del 103-C4/104-C0 que este módulo SABE remuestrear. Es un
#: subconjunto propio de `TIPOS_DE_PARTICION` —no una lista nueva: reutilizar
#: el vocabulario compartido es lo que evita que este corte declare sus
#: propios nombres de diseño y acabe divergiendo de 104-C0—. `groups_and_time`
#: está en `TIPOS_DE_PARTICION` y NO está aquí a propósito: pedirlo cae en la
#: rama «método no soportado», nunca en un IID silencioso.
DISENOS_SOPORTADOS = ("iid", "groups", "temporal")

#: Qué representa el número sobre el que se construye el intervalo.
#: `fixed_model_on_population` es el mismo token que ya usa `MetricSpec` del
#: 104-C0 para toda fórmula del catálogo — un modelo ya ajustado, sobre una
#: muestra que se toma representativa de la población. `oof_pooled_estimate`
#: es el caso de predicciones fuera de pliegue AGRUPADAS en una sola muestra
#: (lo que construye `Muestra.desde_registros`): no hay un único modelo fijo
#: —cada pliegue entrenó el suyo— así que llamarlo `fixed_model_on_population`
#: sería atribuirle a un solo modelo una variabilidad que en realidad mezcla
#: varios. `oof_mean_of_folds` NO está aquí: promediar la métrica por pliegue
#: pide resamplear pliegues, no filas ni unidades, y ese mecanismo no está
#: implementado — pedirlo se rechaza con motivo en vez de calcularse a medias.
ESTIMANDOS_DE_INTERVALO = ("fixed_model_on_population", "oof_pooled_estimate")

#: Qué se resamplea. Viaja en el resultado para que quien lo lea no tenga que
#: adivinar si el remuestreo respetó la unidad declarada.
UNIDADES_DE_REMUESTREO = ("row", "unit", "block", "none")

#: El nombre del método, para quien audite sin abrir el código. `None` cuando
#: el intervalo no está disponible: un método que no corrió no tiene nombre.
METODOS_DE_REMUESTREO = (
    "iid_paired_bootstrap_stratified",  # IID + clasificación
    "iid_paired_bootstrap",             # IID + regresión
    "cluster_bootstrap_by_unit",        # grupos (y OOF agrupado)
    "circular_block_bootstrap",         # temporal
)

#: Bajo esta fracción de remuestras válidas, el intervalo se declara
#: indefinido en vez de publicarse. **Medido, no adivinado**: con un diseño de
#: grupos y una sola unidad portando el único evento raro, la fracción de
#: remuestras que lo omiten por azar converge a `1/e ≈ 0,368` cuando crece el
#: número de unidades — nunca supera esa cota con un solo evento raro—, así
#: que un umbral por debajo de 0,632 no cazaría ese caso. `0,75` dista de la
#: cota (margen para que la ley de los grandes números no dé un falso
#: positivo con pocas remuestras) y sigue exigiendo una mayoría clara de
#: remuestras utilizables antes de confiar en el percentil.
PROPORCION_MINIMA_VALIDA = 0.75


@dataclass(frozen=True)
class Intervalo:
    """El intervalo de confianza por remuestreo, con el diseño DECLARADO.

    Mismo patrón que `ValorDeMetrica` (105-C1): **límites o motivo, exactamente
    uno**. Un intervalo que no se pudo construir con este diseño no vale
    `(0, 0)` ni el propio punto estimado repetido dos veces: vale ausente, con
    su motivo — un valor ausente no es un cero, tampoco cuando el valor
    ausente es un intervalo.
    """

    metric_id: str
    design: str
    estimand: str
    resampling_unit: str
    seed: int
    n_resamples: int
    level: float
    method: str | None = None
    n_resamples_used: int | None = None
    n_resamples_degenerate: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    block_length: int | None = None
    elapsed_seconds: float | None = None
    undefined_reason: dict[str, str] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.metric_id, "metric_id")
        exigir_opcion(self.design, "design", TIPOS_DE_PARTICION)
        exigir_opcion(self.estimand, "estimand", ESTIMANDOS_DE_INTERVALO)
        exigir_opcion(self.resampling_unit, "resampling_unit", UNIDADES_DE_REMUESTREO)
        if self.method is not None:
            exigir_opcion(self.method, "method", METODOS_DE_REMUESTREO)
        exigir_entero(self.seed, "seed")
        exigir_entero(self.n_resamples, "n_resamples", minimo=1)
        exigir_real(self.level, "level", minimo=0.0, maximo=1.0)
        exigir_entero_o_nulo(self.n_resamples_used, "n_resamples_used", minimo=0)
        exigir_entero_o_nulo(self.n_resamples_degenerate, "n_resamples_degenerate", minimo=0)
        exigir_entero_o_nulo(self.block_length, "block_length", minimo=1)
        exigir_real_o_nulo(self.elapsed_seconds, "elapsed_seconds", minimo=0.0)
        exigir_real_o_nulo(self.ci_low, "ci_low")
        exigir_real_o_nulo(self.ci_high, "ci_high")

        if (self.ci_low is None) != (self.ci_high is None):
            raise EsquemaInvalido("intervalo_a_medias",
                                  valor=repr((self.ci_low, self.ci_high)))
        tiene_intervalo = self.ci_low is not None
        if self.undefined_reason is not None:
            object.__setattr__(self, "undefined_reason",
                               exigir_motivo_bilingue(self.undefined_reason,
                                                      "undefined_reason"))
        if tiene_intervalo == (self.undefined_reason is not None):
            clave = ("intervalo_con_valor_y_motivo" if tiene_intervalo
                     else "intervalo_sin_valor_ni_motivo")
            raise EsquemaInvalido(clave, campo=self.metric_id)
        if tiene_intervalo and self.ci_low > self.ci_high:  # type: ignore[operator]
            raise EsquemaInvalido("intervalo_al_reves",
                                  valor=repr((self.ci_low, self.ci_high)))

    @property
    def disponible(self) -> bool:
        return self.ci_low is not None

    def a_json(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id, "design": self.design,
            "estimand": self.estimand, "method": self.method,
            "resampling_unit": self.resampling_unit, "seed": self.seed,
            "n_resamples": self.n_resamples,
            "n_resamples_used": self.n_resamples_used,
            "n_resamples_degenerate": self.n_resamples_degenerate,
            "level": self.level, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "block_length": self.block_length,
            "elapsed_seconds": self.elapsed_seconds,
            "undefined_reason": (dict(self.undefined_reason)
                                 if self.undefined_reason else None),
        }

    @classmethod
    def desde_json(cls, payload: Any) -> "Intervalo":
        """El inverso de `a_json`. No lleva sobre (`schema`/`schema_version`):
        este documento solo vive EMBEBIDO en `ValorDeMetrica.uncertainty`, que
        ya viaja dentro del sobre de quien lo contiene."""
        return cls(**payload)


# ---------------------------------------------------------------------------
# Generadores de índices: uno por diseño, cada uno con su propia unidad
# ---------------------------------------------------------------------------

def _indices_iid(muestra: Muestra, azar: random.Random) -> list[int]:
    """Bootstrap emparejado por observaciones. Clasificación: ESTRATIFICADO por
    la etiqueta observada — ver la decisión 1 del módulo."""
    if muestra.task in TAREAS_DE_CLASIFICACION:
        filas_por_clase: dict[Any, list[int]] = {}
        for i, y in enumerate(muestra.y_true):
            filas_por_clase.setdefault(y, []).append(i)
        indices: list[int] = []
        for filas in filas_por_clase.values():
            indices.extend(azar.choices(filas, k=len(filas)))
        return indices
    return azar.choices(range(muestra.n), k=muestra.n)


def _indices_groups(muestra: Muestra, azar: random.Random) -> list[int]:
    """Cluster bootstrap: se eligen UNIDADES con reemplazo y cada una aporta
    TODAS sus filas — ver la decisión 3 del módulo."""
    filas_por_unidad: dict[str, list[int]] = {}
    for i, unidad in enumerate(muestra.units or ()):
        filas_por_unidad.setdefault(unidad, []).append(i)
    unidades = list(filas_por_unidad)
    indices: list[int] = []
    for unidad in azar.choices(unidades, k=len(unidades)):
        indices.extend(filas_por_unidad[unidad])
    return indices


def _bloques_circulares(n: int, longitud: int, azar: random.Random) -> list[int]:
    """Block bootstrap circular: tramos contiguos de `longitud`, envolviendo al
    llegar al final, hasta completar `n` — ver la decisión 4 del módulo."""
    indices: list[int] = []
    while len(indices) < n:
        inicio = azar.randrange(n)
        indices.extend((inicio + paso) % n for paso in range(longitud))
    return indices[:n]


def _longitud_de_bloque_por_omision(n: int) -> int:
    """`n**(1/3)`, la heurística de cordura — no la selección automática de
    Politis-White, que exige estimar el espectro. Documentada en la decisión 4."""
    return max(2, min(n, round(n ** (1.0 / 3.0))))


# ---------------------------------------------------------------------------
# El recorte SIN revalidar: por qué es seguro, en el docstring del módulo
# ---------------------------------------------------------------------------

_CAMPOS_TAL_CUAL = ("task", "classes", "positive_label", "score_rule")
_CAMPOS_POR_FILA = ("y_true", "probabilities", "scores", "predictions", "weights", "units")


def _recortar(muestra: Muestra, indices: Sequence[int]) -> Muestra:
    """La `Muestra` de esas filas —con repetidos—, sin pasar por `__post_init__`.

    Ver la sección «EL COSTE» del docstring del módulo: repetir con reemplazo
    filas que ya validaron una vez no puede producir una fila inválida.
    """
    recorte = object.__new__(Muestra)
    for campo in _CAMPOS_TAL_CUAL:
        object.__setattr__(recorte, campo, getattr(muestra, campo))
    for campo in _CAMPOS_POR_FILA:
        valor = getattr(muestra, campo)
        object.__setattr__(recorte, campo,
                           tuple(valor[i] for i in indices) if valor is not None else None)
    return recorte


# ---------------------------------------------------------------------------
# El percentil: interpolación lineal, el método por omisión de `numpy.percentile`
# ---------------------------------------------------------------------------

def _percentil(ordenados: Sequence[float], proporcion: float) -> float:
    n = len(ordenados)
    if n == 1:
        return ordenados[0]
    posicion = proporcion * (n - 1)
    bajo = math.floor(posicion)
    alto = math.ceil(posicion)
    if bajo == alto:
        return ordenados[bajo]
    fraccion = posicion - bajo
    return ordenados[bajo] + (ordenados[alto] - ordenados[bajo]) * fraccion


# ---------------------------------------------------------------------------
# La función pública
# ---------------------------------------------------------------------------

def intervalo(metric_id: str, muestra: Muestra, *, diseno: str, estimando: str,
              semilla: int, remuestras: int = 1000, nivel: float = 0.95,
              umbral: float | None = None, longitud_de_bloque: int | None = None,
              valor: ValorDeMetrica | None = None,
              formula: Callable[[Muestra, float | None],
                                "float | Indefinida"] | None = None) -> Intervalo:
    """El intervalo de confianza de `metric_id` sobre `muestra`, según `diseno`.

    `diseno` y `estimando` son OBLIGATORIOS y sin valor por omisión: elegir por
    quien llama, no adivinar aquí, es precisamente lo que impide que un IID se
    cuele en silencio donde el diseño real es de grupos o temporal.

    Devuelve `Intervalo` SIEMPRE — nunca levanta por datos degenerados; sí
    levanta `EntradaNoMedible` cuando la propia PETICIÓN está mal cableada
    (IID sobre una muestra con unidades repetidas, grupos sin unidad
    declarada), que es la misma frontera error/indefinición que traza C1.

    `formula` es la puerta para una cifra DERIVADA que no está —ni debe
    estar— en el catálogo de C1 porque necesita un parámetro que el catálogo
    no conoce: el PPV/NPV transportado a una prevalencia declarada (109-C2)
    es el caso que la abrió. Sin ella, cada cifra derivada se traería su
    propio bootstrap y acabaríamos con dos remuestreos que divergen; con
    ella, **el remuestreo se decide en un solo sitio**, que es este. Va
    siempre acompañada de `valor` —el punto ya calculado con esa misma
    fórmula—: pedir un intervalo derivado sin decir de qué punto es sería
    dejar que `calcular()` buscase en el catálogo un `metric_id` que por
    definición no está, y el rechazo llegaría con un motivo que no explica
    nada. `formula` recibe `(muestra, umbral)` y devuelve un número o
    `Indefinida`, exactamente como las del registro.
    """
    if formula is not None and valor is None:
        raise EntradaNoMedible("intervalo_derivado_sin_punto", campo=metric_id)
    exigir_opcion(diseno, "diseno", TIPOS_DE_PARTICION)
    exigir_opcion(estimando, "estimando", ESTIMANDOS_DE_INTERVALO)
    exigir_entero(semilla, "semilla")
    exigir_entero(remuestras, "remuestras", minimo=1)
    if not (0.0 < nivel < 1.0):
        raise EsquemaInvalido("fuera_de_rango", campo="nivel", minimo=0.0, maximo=1.0,
                              valor=repr(nivel))
    if longitud_de_bloque is not None:
        exigir_entero(longitud_de_bloque, "longitud_de_bloque", minimo=1)

    punto = valor if valor is not None else calcular(metric_id, muestra, umbral=umbral)
    comun: dict[str, Any] = dict(metric_id=metric_id, design=diseno, estimand=estimando,
                                 seed=semilla, n_resamples=remuestras, level=nivel)

    # El punto ya sale indefinido (una sola clase, cero filas...): no hay nada
    # que remuestrear, y el motivo es EL MISMO que ya dio C1 — no se inventa
    # uno nuevo para la misma ausencia.
    if punto.value is None:
        return Intervalo(**comun, resampling_unit="none",
                         undefined_reason=punto.undefined_reason)

    if diseno not in DISENOS_SOPORTADOS:
        return Intervalo(**comun, resampling_unit="none",
                         undefined_reason=motivo("diseno_no_soportado", valor=diseno))

    generador: Callable[[], list[int]]
    longitud_bloque_declarada: int | None = None

    if diseno == "iid":
        if muestra.n_unidades is not None and muestra.n_unidades < muestra.n:
            raise EntradaNoMedible("unidad_repetida_en_diseno_iid",
                                   valor=muestra.n_unidades, opciones=muestra.n)
        if muestra.n < 2:
            return Intervalo(**comun, resampling_unit="row",
                             undefined_reason=motivo("observaciones_insuficientes"))
        unidad = "row"
        metodo = ("iid_paired_bootstrap_stratified"
                  if muestra.task in TAREAS_DE_CLASIFICACION else "iid_paired_bootstrap")
        rng = random.Random(semilla)
        generador = lambda: _indices_iid(muestra, rng)  # noqa: E731

    elif diseno == "groups":
        if muestra.units is None:
            raise EntradaNoMedible("remuestreo_de_grupos_sin_unidad", campo=metric_id)
        if muestra.n_unidades < 2:  # type: ignore[operator]
            return Intervalo(**comun, resampling_unit="unit",
                             undefined_reason=motivo("unidades_insuficientes",
                                                     valor=muestra.n_unidades))
        unidad = "unit"
        metodo = "cluster_bootstrap_by_unit"
        rng = random.Random(semilla)
        generador = lambda: _indices_groups(muestra, rng)  # noqa: E731

    else:  # temporal
        if muestra.n < 4:
            return Intervalo(**comun, resampling_unit="block",
                             undefined_reason=motivo("observaciones_insuficientes"))
        longitud_bloque_declarada = longitud_de_bloque or _longitud_de_bloque_por_omision(
            muestra.n)
        longitud_bloque_declarada = min(longitud_bloque_declarada, muestra.n)
        unidad = "block"
        metodo = "circular_block_bootstrap"
        rng = random.Random(semilla)
        generador = lambda: _bloques_circulares(  # noqa: E731
            muestra.n, longitud_bloque_declarada, rng)

    regla = formula if formula is not None else REGISTRO[metric_id].formula
    valores: list[float] = []
    degeneradas = 0
    empieza = time.perf_counter()
    for _ in range(remuestras):
        recorte = _recortar(muestra, generador())
        resultado = regla(recorte, umbral)
        if isinstance(resultado, Indefinida):
            degeneradas += 1
        else:
            valores.append(float(resultado))
    transcurrido = time.perf_counter() - empieza

    validas = len(valores)
    if remuestras == 0 or validas / remuestras < PROPORCION_MINIMA_VALIDA:
        return Intervalo(**comun, resampling_unit=unidad, method=metodo,
                         n_resamples_used=validas, n_resamples_degenerate=degeneradas,
                         block_length=longitud_bloque_declarada,
                         elapsed_seconds=transcurrido,
                         undefined_reason=motivo("remuestras_degeneradas",
                                                 valor=degeneradas, opciones=remuestras))

    valores.sort()
    cola = (1.0 - nivel) / 2.0
    return Intervalo(**comun, resampling_unit=unidad, method=metodo,
                     n_resamples_used=validas, n_resamples_degenerate=degeneradas,
                     block_length=longitud_bloque_declarada, elapsed_seconds=transcurrido,
                     ci_low=_percentil(valores, cola), ci_high=_percentil(valores, 1.0 - cola))


def medir(metric_id: str, muestra: Muestra, *, diseno: str, estimando: str, semilla: int,
          remuestras: int = 1000, nivel: float = 0.95, umbral: float | None = None,
          longitud_de_bloque: int | None = None) -> ValorDeMetrica:
    """El número de C1 Y su incertidumbre de C2, en un solo `ValorDeMetrica`.

    Conveniencia para quien no necesita las dos llamadas por separado. Calcula
    el punto UNA vez y se lo pasa a `intervalo()` para no medir la misma
    fórmula dos veces sobre la muestra completa.
    """
    from dataclasses import replace as _replace  # noqa: PLC0415

    punto = calcular(metric_id, muestra, umbral=umbral)
    ic = intervalo(metric_id, muestra, diseno=diseno, estimando=estimando, semilla=semilla,
                   remuestras=remuestras, nivel=nivel, umbral=umbral,
                   longitud_de_bloque=longitud_de_bloque, valor=punto)
    return _replace(punto, uncertainty=ic.a_json())
