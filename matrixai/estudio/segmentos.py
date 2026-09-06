# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C4 — segmentos, errores e importancia: comparar un recorte de la
muestra contra el resto sin fingir que un tamaño cualquiera basta para
concluir, y medir cuánto se apoya un modelo en cada variable permutándola.

DOS PREGUNTAS DISTINTAS, DOS FUNCIONES. `analizar_segmento()` responde «¿este
recorte de filas rinde peor o mejor que el resto, y hay soporte para decirlo?»
reutilizando el registro de 105-C1 (`calcular`, `es_mejor` — la dirección de
cada métrica NO se reimplementa aquí, sería el invariante «dos sitios
declarando lo mismo» roto a la primera línea). `importancia_por_permutacion()`
responde «¿cuánto empeora la métrica si esta variable se vuelve ruido?»
barajando una columna (o un grupo de columnas correlacionadas, cuando se
declaran juntas) y volviendo a puntuar con la función que el LLAMANTE aporta
— este módulo no entrena ni predice nada, eso es de `matrixai-engines`.

SOPORTE, NO SIGNIFICANCIA. El criterio de terminado pide que «una categoría
con 30 filas y un evento reciba cautela»: 30 filas puede ser suficiente
volumen y aun así un solo evento de la clase minoritaria es demasiado poco
para que la cifra del segmento signifique nada. `MINIMO_FILAS_POR_SEGMENTO` y
`MINIMO_EVENTOS_POR_SEGMENTO` son los dos umbrales, y hace falta pasar los
DOS para que `soporte_suficiente` sea verdadero. Esto NO es una prueba de
hipótesis (eso, con su corrección de multiplicidad, es 105-C5 — comparar
segmento contra resto con una diferencia emparejada de verdad); aquí solo se
mide "hay tela para cortar" antes de dejar salir una conclusión.

EXPLORATORIO SE MARCA, NO SE ESCONDE. `predefinido=False` es el caso normal
de un hallazgo que salió de mirar los datos (invariante 9 del contrato); el
campo viaja siempre, nunca implícito.

LA PERMUTACIÓN, Y SU LÍMITE CONOCIDO. Permutar una columna correlacionada con
otra no aísla su aportación: el modelo puede seguir explicando la misma señal
a través de la variable gemela, y la caída medida sale MENOR que si esa
variable fuera la única fuente de esa señal. Esto no es un error del método,
es una propiedad conocida de la importancia por permutación — el criterio de
terminado pide mostrarla («alcance y límites»), no ocultarla ni "arreglarla".

QUÉ NO HACE ESTE MÓDULO. No decide qué filas forman un segmento (eso es una
decisión del llamante, con o sin variable clínica de por medio) ni impide
seleccionar variables mirando el mismo test que luego se usa para validar —
esa disciplina («no usar el test para elegir variables y conservar la misma
etiqueta de validación») es de quien orquesta el estudio con el registro de
accesos del 104-C0, no de esta función pura.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from matrixai.estudio.incertidumbre import _recortar
from matrixai.estudio.metricas import (
    EntradaNoMedible,
    EsquemaInvalido,
    Muestra,
    ValorDeMetrica,
    calcular,
    direccion_de,
    distancia_al_ideal,
    es_mejor,
    especificacion,
)
from matrixai.estudio.textos import motivo
from matrixai.estudio.validacion import digest_canonico, exigir_entero

__all__ = [
    "MINIMO_EVENTOS_POR_SEGMENTO",
    "MINIMO_FILAS_POR_SEGMENTO",
    "AnalisisDeSegmento",
    "ImportanciaDeVariable",
    "analizar_segmento",
    "importancia_por_permutacion",
]

#: Medido con el caso literal del criterio de terminado: una categoría de 30
#: filas y UN evento tiene que recibir cautela. 30 filas sin más NO alcanza
#: por sí solo si la clase minoritaria casi no aparece — hacen falta los dos
#: umbrales, no uno solo.
MINIMO_FILAS_POR_SEGMENTO = 30

#: Heurística de "eventos por variable" de la literatura de modelos clínicos
#: (TRIPOD/PROBAST la citan como referencia de cautela, no como regla dura) —
#: aquí se usa igual de informalmente: menos de 10 filas de la clase
#: minoritaria en un segmento no sostiene ninguna conclusión propia.
MINIMO_EVENTOS_POR_SEGMENTO = 10


@dataclass(frozen=True)
class AnalisisDeSegmento:
    """Segmento contra resto, en la métrica declarada. `diferencia`/
    `peor_que_el_resto` son `None` exactamente cuando alguna de las dos
    mitades salió indefinida — nunca un cero o un `False` de relleno."""

    segmento_id: str
    predefinido: bool
    metric_id: str
    n: int
    eventos: int | None
    metrica_segmento: ValorDeMetrica
    metrica_resto: ValorDeMetrica
    diferencia: float | None
    peor_que_el_resto: bool | None
    soporte_suficiente: bool
    motivo_de_cautela: dict[str, str] | None

    def __post_init__(self) -> None:
        exigir_entero(self.n, "n", minimo=0)
        if self.eventos is not None:
            exigir_entero(self.eventos, "eventos", minimo=0)
        if (self.diferencia is None) != (self.peor_que_el_resto is None):
            raise EsquemaInvalido("falta_campo", campo="diferencia/peor_que_el_resto")
        if self.soporte_suficiente and self.motivo_de_cautela is not None:
            raise EsquemaInvalido("metrica_con_valor_y_motivo", campo="soporte_del_segmento")
        if not self.soporte_suficiente and self.motivo_de_cautela is None:
            raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo="soporte_del_segmento")

    @property
    def es_exploratorio(self) -> bool:
        return not self.predefinido

    def a_json(self) -> dict[str, Any]:
        return {
            "segmento_id": self.segmento_id, "predefinido": self.predefinido,
            "metric_id": self.metric_id, "n": self.n, "eventos": self.eventos,
            "metrica_segmento": self.metrica_segmento.a_json(),
            "metrica_resto": self.metrica_resto.a_json(),
            "diferencia": self.diferencia, "peor_que_el_resto": self.peor_que_el_resto,
            "soporte_suficiente": self.soporte_suficiente,
            "motivo_de_cautela": dict(self.motivo_de_cautela) if self.motivo_de_cautela else None,
        }

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def analizar_segmento(metric_id: str, muestra: Muestra, indices_segmento: Sequence[int], *,
                      segmento_id: str, predefinido: bool, umbral: float | None = None,
                      minimo_filas: int = MINIMO_FILAS_POR_SEGMENTO,
                      minimo_eventos: int = MINIMO_EVENTOS_POR_SEGMENTO) -> AnalisisDeSegmento:
    """`indices_segmento` son posiciones dentro de `muestra` (0-índice). El
    resto es el complemento exacto — no hay tercera categoría de filas."""
    especificacion(metric_id)
    exigir_entero(minimo_filas, "minimo_filas", minimo=1)
    exigir_entero(minimo_eventos, "minimo_eventos", minimo=1)

    indices_segmento = list(indices_segmento)
    if len(set(indices_segmento)) != len(indices_segmento):
        raise EntradaNoMedible("segmento_indices_repetidos", campo=segmento_id)
    for i in indices_segmento:
        if i < 0 or i >= muestra.n:
            raise EntradaNoMedible("segmento_indice_fuera_de_rango",
                                   campo=segmento_id, valor=i, opciones=muestra.n)

    conjunto = set(indices_segmento)
    indices_resto = [i for i in range(muestra.n) if i not in conjunto]

    segmento = _recortar(muestra, indices_segmento)
    resto = _recortar(muestra, indices_resto)

    metrica_segmento = calcular(metric_id, segmento, umbral=umbral)
    metrica_resto = calcular(metric_id, resto, umbral=umbral)

    n = len(indices_segmento)
    eventos = None
    if muestra.task == "binary_classification":
        eventos = sum(1 for i in indices_segmento if muestra.y_true[i] == muestra.positive_label)

    diferencia = None
    peor_que_el_resto = None
    if metrica_segmento.value is not None and metrica_resto.value is not None:
        diferencia = metrica_segmento.value - metrica_resto.value
        peor_que_el_resto = es_mejor(metric_id, metrica_resto.value, metrica_segmento.value)

    razon_de_cautela = None
    if n < minimo_filas:
        razon_de_cautela = motivo("segmento_soporte_insuficiente", campo=segmento_id,
                                  valor=n, opciones=minimo_filas)
    elif eventos is not None and min(eventos, n - eventos) < minimo_eventos:
        razon_de_cautela = motivo("segmento_pocos_eventos", campo=segmento_id,
                                  valor=min(eventos, n - eventos), opciones=minimo_eventos)

    return AnalisisDeSegmento(
        segmento_id=segmento_id, predefinido=predefinido, metric_id=metric_id,
        n=n, eventos=eventos, metrica_segmento=metrica_segmento, metrica_resto=metrica_resto,
        diferencia=diferencia, peor_que_el_resto=peor_que_el_resto,
        soporte_suficiente=razon_de_cautela is None, motivo_de_cautela=razon_de_cautela)


@dataclass(frozen=True)
class ImportanciaDeVariable:
    """Cuánto empeora `metric_id` cuando esta variable (o grupo de variables
    correlacionadas, declaradas juntas) se vuelve ruido. `caida_media`
    positiva es la variable importando: la métrica empeoró al perderla."""

    variable: str
    n_repeticiones: int
    caida_media: float | None = None
    desviacion: float | None = None
    caidas: tuple[float, ...] = ()
    undefined_reason: dict[str, str] | None = None

    def __post_init__(self) -> None:
        exigir_entero(self.n_repeticiones, "n_repeticiones", minimo=1)
        if self.caida_media is None and self.undefined_reason is None:
            raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo="importancia")
        if self.caida_media is not None and self.undefined_reason is not None:
            raise EsquemaInvalido("metrica_con_valor_y_motivo", campo="importancia")
        if self.caida_media is not None and len(self.caidas) != self.n_repeticiones:
            raise EsquemaInvalido("filas_desalineadas", campo="caidas",
                                  valor=len(self.caidas), opciones=self.n_repeticiones)

    def a_json(self) -> dict[str, Any]:
        return {
            "variable": self.variable, "n_repeticiones": self.n_repeticiones,
            "caida_media": self.caida_media, "desviacion": self.desviacion,
            "caidas": list(self.caidas),
            "undefined_reason": dict(self.undefined_reason) if self.undefined_reason else None,
        }

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def _caida(metric_id: str, base: float, permutada: float) -> float:
    """Positiva = peor tras permutar, con independencia de si la métrica
    mejora subiendo, bajando, o acercándose a un ideal — mismo signo siempre,
    para no obligar a quien lea `caida_media` a conocer la dirección de cada
    métrica por su cuenta."""
    direccion = direccion_de(metric_id)
    if direccion == "higher_is_better":
        return base - permutada
    if direccion == "lower_is_better":
        return permutada - base
    return distancia_al_ideal(metric_id, permutada) - distancia_al_ideal(metric_id, base)


def importancia_por_permutacion(metric_id: str, filas: Sequence[Mapping[str, Any]],
                                funcion_de_puntuacion: Callable[[Sequence[Mapping[str, Any]]], Muestra],
                                *, variables: Mapping[str, Sequence[str]],
                                n_repeticiones: int = 5, semilla: int,
                                umbral: float | None = None) -> tuple[ImportanciaDeVariable, ...]:
    """`variables` mapea un NOMBRE lógico a una o más columnas físicas que se
    barajan JUNTAS con la misma permutación de filas — así una categórica
    con varias columnas one-hot (o dos variables que el llamante sabe
    correlacionadas y quiere tratar como una) se permuta como una unidad, no
    columna a columna (eso rompería la correlación en vez de simularla
    ausente).

    `funcion_de_puntuacion` no la ejecuta este módulo con conocimiento de
    ningún motor: la aporta el llamante (un `Motor` de `matrixai-engines` ya
    ajustado, normalmente), y aquí solo se la invoca con `filas` sin permutar
    y con cada versión permutada, para medir la diferencia."""
    especificacion(metric_id)
    exigir_entero(n_repeticiones, "n_repeticiones", minimo=1)
    if not variables:
        raise EsquemaInvalido("falta_campo", campo="variables")

    filas = [dict(fila) for fila in filas]
    n = len(filas)
    base = calcular(metric_id, funcion_de_puntuacion(filas), umbral=umbral)
    if base.value is None:
        motivo_base = motivo("importancia_metrica_base_indefinida", campo=metric_id)
        return tuple(ImportanciaDeVariable(variable=nombre, n_repeticiones=n_repeticiones,
                                           undefined_reason=motivo_base)
                    for nombre in variables)

    resultados = []
    for nombre, columnas in variables.items():
        caidas: list[float] = []
        indefinida = False
        for repeticion in range(n_repeticiones):
            azar = random.Random(f"{semilla}:{nombre}:{repeticion}")
            orden = list(range(n))
            azar.shuffle(orden)
            permutadas = [dict(fila) for fila in filas]
            for columna in columnas:
                valores_barajados = [filas[i][columna] for i in orden]
                for i, fila in enumerate(permutadas):
                    fila[columna] = valores_barajados[i]
            metrica_permutada = calcular(metric_id, funcion_de_puntuacion(permutadas), umbral=umbral)
            if metrica_permutada.value is None:
                indefinida = True
                break
            caidas.append(_caida(metric_id, base.value, metrica_permutada.value))
        if indefinida:
            resultados.append(ImportanciaDeVariable(
                variable=nombre, n_repeticiones=n_repeticiones,
                undefined_reason=motivo("importancia_metrica_permutada_indefinida", campo=nombre)))
            continue
        caida_media = statistics.mean(caidas)
        desviacion = statistics.stdev(caidas) if len(caidas) > 1 else 0.0
        resultados.append(ImportanciaDeVariable(
            variable=nombre, n_repeticiones=n_repeticiones, caida_media=caida_media,
            desviacion=desviacion, caidas=tuple(caidas)))
    return tuple(resultados)
