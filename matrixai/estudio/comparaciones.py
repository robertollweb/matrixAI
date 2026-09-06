# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C5 — comparaciones y equivalencia: si un candidato rinde de verdad
distinto a un baseline (o al proceso actual del cliente, tratado igual: es
otro candidato más), con una diferencia EMPAREJADA — no comparando dos
intervalos individuales por separado, que es precisamente el error que este
corte existe para no cometer.

POR QUÉ EMPAREJADA Y NO DOS INTERVALOS SUELTOS. Dos modelos evaluados sobre
las MISMAS filas suelen acertar y fallar en las mismas filas difíciles — sus
errores están correlacionados. Remuestrear cada uno por separado y mirar si
sus intervalos se solapan IGNORA esa correlación y es sistemáticamente
conservador: puede declarar «inconcluso» un caso donde la diferencia, medida
emparejada (mismos índices de remuestreo para los dos a la vez, resta
DESPUÉS), es clara. El criterio de terminado pide exactamente demostrar esto
con un caso: intervalos individuales solapados, diferencia emparejada
concluyente.

REUTILIZADO DE 105-C2, NO REESCRITO. Los tres generadores de remuestreo
(`_indices_iid`, `_indices_groups`, `_bloques_circulares`), el recorte barato
(`_recortar`) y el percentil (`_percentil`) son los MISMOS que usa
`intervalo()` — la única diferencia real es que aquí se recortan DOS muestras
con los mismos índices en cada iteración, y se resta antes de guardar, en vez
de guardar un valor solo. El resultado de la incertidumbre se devuelve como
el mismo `Intervalo` de 105-C2 (sus límites no están acotados a [0,1]: una
diferencia puede ser negativa igual que cualquier otro real).

ALCANCE DECLARADO: SOLO MÉTRICAS CON DIRECCIÓN. Una métrica de valor ideal
(pendiente/intercepto de calibración) no tiene un lado que favorezca al
candidato con un solo signo — decidir qué significa "mejor" ahí exigiría su
propia regla, que este corte no construye. Pedir la comparación con una de
esas métricas es un error de cableado (`EntradaNoMedible`), no un veredicto.

CINCO VEREDICTOS, Y LA PRIORIDAD ENTRE ELLOS CUANDO SE SOLAPAN. `mejora` e
`inferioridad` exigen que el intervalo de la diferencia EXCLUYA el cero hacia
un lado. `equivalencia_practica` exige que el intervalo entero quepa dentro
de `margen_equivalencia` en las dos direcciones — y se comprueba ANTES que
`mejora`/`inferioridad`: una diferencia estadísticamente clara pero
prácticamente insignificante (intervalo estrecho, dentro del margen, sin
tocar el cero) es equivalencia práctica, no una mejora que nadie notaría.
`inconcluso` es lo que queda cuando ninguna de las anteriores aplica —
incluida la propia métrica saliendo indefinida en cualquiera de las dos
muestras, o el remuestreo degenerado por debajo de `PROPORCION_MINIMA_VALIDA`
(105-C2). `incomparable` es aparte, y NUNCA compite con los demás: candidato
y baseline no describen las mismas filas, o declaran protocolos distintos —
ahí no hay nada que remuestrear, y no se intenta.

QUÉ NO HACE ESTE MÓDULO. No corrige por multiplicidad cuando el llamante
compara muchos candidatos o segmentos a la vez (el invariante 9 del contrato
pide DECLARARLA, no impone un método — igual que 105-C4 dejó fuera la
corrección de multiplicidad entre segmentos, con el mismo razonamiento: sin
un caso de uso real que fije el método, construir uno sería inventar una
respuesta). Tampoco decide qué modelo se despliega ante un empate o un
inconcluso («la elección operativa del más sencillo no cambia el veredicto
científico», texto literal del contrato) — eso es una política del llamante,
el veredicto que este módulo da no cambia por ella.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from matrixai.estudio.incertidumbre import (
    DISENOS_SOPORTADOS,
    ESTIMANDOS_DE_INTERVALO,
    PROPORCION_MINIMA_VALIDA,
    Intervalo,
    _bloques_circulares,
    _indices_groups,
    _indices_iid,
    _longitud_de_bloque_por_omision,
    _percentil,
    _recortar,
)
from matrixai.estudio.metricas import (
    REGISTRO,
    EntradaNoMedible,
    EsquemaInvalido,
    Indefinida,
    Muestra,
    calcular,
    direccion_de,
    especificacion,
)
from matrixai.estudio.textos import motivo
from matrixai.estudio.validacion import (
    exigir_entero,
    exigir_real,
    exigir_real_o_nulo,
    exigir_texto_o_nulo,
)
from matrixai.estudio.vocabulario import TIPOS_DE_PARTICION, exigir_opcion

__all__ = ["VEREDICTOS_DE_COMPARACION", "ComparacionEmparejada", "comparar_candidatos"]

#: `incomparable` no compite con los otros cuatro: es una categoría aparte,
#: para cuando la pregunta ("¿cuál rinde mejor?") ni siquiera se puede
#: plantear con estos dos conjuntos de filas.
VEREDICTOS_DE_COMPARACION = ("mejora", "inferioridad", "equivalencia_practica",
                            "inconcluso", "incomparable")


@dataclass(frozen=True)
class ComparacionEmparejada:
    """El veredicto de comparar `candidato` contra `baseline` en `metric_id`.

    `intervalo` es `None` exactamente cuando `veredicto="incomparable"` — ahí
    no hubo remuestreo porque no hay nada emparejable. `diferencia_puntual`
    puede faltar incluso con un `intervalo` presente (si la propia métrica
    salió indefinida en alguna mitad, el intervalo lo declara y no hay punto
    que dar), pero nunca al revés: un intervalo con límites numéricos siempre
    trae su punto.
    """

    metric_id: str
    diseno: str
    estimando: str
    veredicto: str
    margen_equivalencia: float | None
    diferencia_puntual: float | None
    intervalo: Intervalo | None
    undefined_reason: dict[str, str] | None = None

    def __post_init__(self) -> None:
        exigir_opcion(self.diseno, "diseno", TIPOS_DE_PARTICION)
        exigir_opcion(self.estimando, "estimando", ESTIMANDOS_DE_INTERVALO)
        exigir_opcion(self.veredicto, "veredicto", VEREDICTOS_DE_COMPARACION)
        exigir_real_o_nulo(self.margen_equivalencia, "margen_equivalencia", minimo=0.0)
        exigir_real_o_nulo(self.diferencia_puntual, "diferencia_puntual")

        incomparable = self.veredicto == "incomparable"
        if incomparable and self.undefined_reason is None:
            raise EsquemaInvalido("metrica_sin_valor_ni_motivo", campo="comparacion")
        if not incomparable and self.undefined_reason is not None:
            raise EsquemaInvalido("metrica_con_valor_y_motivo", campo="comparacion")
        if incomparable and self.intervalo is not None:
            raise EsquemaInvalido("falta_campo", campo="intervalo")
        if not incomparable and self.intervalo is None:
            raise EsquemaInvalido("falta_campo", campo="intervalo")
        if (self.diferencia_puntual is None and self.intervalo is not None
                and self.intervalo.disponible):
            raise EsquemaInvalido("falta_campo", campo="diferencia_puntual")

    def a_json(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id, "diseno": self.diseno, "estimando": self.estimando,
            "veredicto": self.veredicto, "margen_equivalencia": self.margen_equivalencia,
            "diferencia_puntual": self.diferencia_puntual,
            "intervalo": self.intervalo.a_json() if self.intervalo is not None else None,
            "undefined_reason": dict(self.undefined_reason) if self.undefined_reason else None,
        }


def _incomparable(metric_id: str, diseno: str, estimando: str,
                  margen_equivalencia: float | None, clave: str, **campos: object) -> ComparacionEmparejada:
    return ComparacionEmparejada(
        metric_id=metric_id, diseno=diseno, estimando=estimando, veredicto="incomparable",
        margen_equivalencia=margen_equivalencia, diferencia_puntual=None, intervalo=None,
        undefined_reason=motivo(clave, **campos))


def _generador_de_indices(diseno: str, muestra: Muestra, rng, longitud_de_bloque: int | None):
    """Mismo dispatch que `intervalo()` (105-C2) — devuelve `(generador,
    unidad, metodo, longitud_bloque_declarada)`, o `None` si la muestra no
    sostiene ESE diseño (el motivo ya lo compone el llamante)."""
    if diseno == "iid":
        if muestra.n < 2:
            return None
        unidad = "row"
        metodo = "iid_paired_bootstrap"
        return (lambda: _indices_iid(muestra, rng)), unidad, metodo, None
    if diseno == "groups":
        if muestra.units is None:
            raise EntradaNoMedible("remuestreo_de_grupos_sin_unidad", campo="comparacion")
        if muestra.n_unidades is None or muestra.n_unidades < 2:
            return None
        unidad = "unit"
        metodo = "cluster_bootstrap_by_unit"
        return (lambda: _indices_groups(muestra, rng)), unidad, metodo, None
    # temporal
    if muestra.n < 4:
        return None
    longitud = min(longitud_de_bloque or _longitud_de_bloque_por_omision(muestra.n), muestra.n)
    unidad = "block"
    metodo = "circular_block_bootstrap"
    return (lambda: _bloques_circulares(muestra.n, longitud, rng)), unidad, metodo, longitud


def comparar_candidatos(metric_id: str, candidato: Muestra, baseline: Muestra, *,
                        diseno: str, estimando: str, semilla: int,
                        margen_equivalencia: float | None = None,
                        remuestras: int = 1000, nivel: float = 0.95,
                        umbral: float | None = None, longitud_de_bloque: int | None = None,
                        protocolo_candidato: str | None = None,
                        protocolo_baseline: str | None = None) -> ComparacionEmparejada:
    """`candidato` y `baseline` tienen que describir las MISMAS filas (mismo
    `y_true`, en el mismo orden) — si no, el veredicto es `incomparable`, no
    una excepción: comparar candidatos sin un baseline compatible es una
    situación normal en un estudio con varios, no un error de quien llama.

    `protocolo_candidato`/`protocolo_baseline` son opcionales (p.ej. el
    digest de `SplitPlan` de cada uno); si se dan los DOS y no coinciden,
    también es `incomparable` — declarar el protocolo no es obligatorio,
    pero si se declara y no cuadra, se usa."""
    especificacion(metric_id)
    exigir_opcion(diseno, "diseno", TIPOS_DE_PARTICION)
    exigir_opcion(estimando, "estimando", ESTIMANDOS_DE_INTERVALO)
    exigir_entero(semilla, "semilla")
    exigir_entero(remuestras, "remuestras", minimo=1)
    if not (0.0 < nivel < 1.0):
        raise EsquemaInvalido("fuera_de_rango", campo="nivel", minimo=0.0, maximo=1.0,
                              valor=repr(nivel))
    if margen_equivalencia is not None:
        exigir_real(margen_equivalencia, "margen_equivalencia", minimo=0.0)
    exigir_texto_o_nulo(protocolo_candidato, "protocolo_candidato")
    exigir_texto_o_nulo(protocolo_baseline, "protocolo_baseline")

    direccion = direccion_de(metric_id)
    if direccion is None:
        raise EntradaNoMedible("comparacion_metrica_sin_direccion", campo=metric_id)

    if candidato.n != baseline.n or candidato.y_true != baseline.y_true:
        return _incomparable(metric_id, diseno, estimando, margen_equivalencia,
                             "comparacion_incomparable_filas",
                             valor=candidato.n, opciones=baseline.n)
    if (protocolo_candidato is not None and protocolo_baseline is not None
            and protocolo_candidato != protocolo_baseline):
        return _incomparable(metric_id, diseno, estimando, margen_equivalencia,
                             "comparacion_incomparable_protocolo",
                             valor=protocolo_candidato, opciones=protocolo_baseline)

    comun: dict[str, Any] = dict(metric_id=metric_id, design=diseno, estimand=estimando,
                                 seed=semilla, n_resamples=remuestras, level=nivel)

    metrica_candidato = calcular(metric_id, candidato, umbral=umbral)
    metrica_baseline = calcular(metric_id, baseline, umbral=umbral)
    if metrica_candidato.value is None or metrica_baseline.value is None:
        indefinido = metrica_candidato.undefined_reason or metrica_baseline.undefined_reason
        intervalo = Intervalo(**comun, resampling_unit="none", undefined_reason=indefinido)
        return ComparacionEmparejada(metric_id=metric_id, diseno=diseno, estimando=estimando,
                                     veredicto="inconcluso", margen_equivalencia=margen_equivalencia,
                                     diferencia_puntual=None, intervalo=intervalo)

    diferencia_puntual = metrica_candidato.value - metrica_baseline.value

    if diseno not in DISENOS_SOPORTADOS:
        intervalo = Intervalo(**comun, resampling_unit="none",
                              undefined_reason=motivo("diseno_no_soportado", valor=diseno))
        return ComparacionEmparejada(metric_id=metric_id, diseno=diseno, estimando=estimando,
                                     veredicto="inconcluso", margen_equivalencia=margen_equivalencia,
                                     diferencia_puntual=diferencia_puntual, intervalo=intervalo)

    rng = random.Random(semilla)
    seleccion = _generador_de_indices(diseno, candidato, rng, longitud_de_bloque)
    if seleccion is None:
        unidad = {"iid": "row", "groups": "unit", "temporal": "block"}[diseno]
        intervalo = Intervalo(**comun, resampling_unit=unidad,
                              undefined_reason=motivo("observaciones_insuficientes"))
        return ComparacionEmparejada(metric_id=metric_id, diseno=diseno, estimando=estimando,
                                     veredicto="inconcluso", margen_equivalencia=margen_equivalencia,
                                     diferencia_puntual=diferencia_puntual, intervalo=intervalo)
    generador, unidad, metodo, longitud_bloque_declarada = seleccion

    registrada = REGISTRO[metric_id]
    diferencias: list[float] = []
    degeneradas = 0
    empieza = time.perf_counter()
    for _ in range(remuestras):
        indices = generador()
        recorte_candidato = _recortar(candidato, indices)
        recorte_baseline = _recortar(baseline, indices)
        valor_candidato = registrada.formula(recorte_candidato, umbral)
        valor_baseline = registrada.formula(recorte_baseline, umbral)
        if isinstance(valor_candidato, Indefinida) or isinstance(valor_baseline, Indefinida):
            degeneradas += 1
        else:
            diferencias.append(float(valor_candidato) - float(valor_baseline))
    transcurrido = time.perf_counter() - empieza

    validas = len(diferencias)
    if remuestras == 0 or validas / remuestras < PROPORCION_MINIMA_VALIDA:
        intervalo = Intervalo(**comun, resampling_unit=unidad, method=metodo,
                              n_resamples_used=validas, n_resamples_degenerate=degeneradas,
                              block_length=longitud_bloque_declarada, elapsed_seconds=transcurrido,
                              undefined_reason=motivo("remuestras_degeneradas",
                                                      valor=degeneradas, opciones=remuestras))
        return ComparacionEmparejada(metric_id=metric_id, diseno=diseno, estimando=estimando,
                                     veredicto="inconcluso", margen_equivalencia=margen_equivalencia,
                                     diferencia_puntual=diferencia_puntual, intervalo=intervalo)

    diferencias.sort()
    cola = (1.0 - nivel) / 2.0
    ci_low = _percentil(diferencias, cola)
    ci_high = _percentil(diferencias, 1.0 - cola)
    intervalo = Intervalo(**comun, resampling_unit=unidad, method=metodo,
                          n_resamples_used=validas, n_resamples_degenerate=degeneradas,
                          block_length=longitud_bloque_declarada, elapsed_seconds=transcurrido,
                          ci_low=ci_low, ci_high=ci_high)

    # `ventaja_*` normaliza la dirección: positivo siempre significa "a favor
    # del candidato", con independencia de si la métrica es mayor-mejor o
    # menor-mejor. Multiplicar invierte el orden cuando el signo es -1.
    signo = 1.0 if direccion == "higher_is_better" else -1.0
    ventaja_baja = min(signo * ci_low, signo * ci_high)
    ventaja_alta = max(signo * ci_low, signo * ci_high)

    if (margen_equivalencia is not None
            and abs(ventaja_baja) <= margen_equivalencia and abs(ventaja_alta) <= margen_equivalencia):
        veredicto = "equivalencia_practica"
    elif ventaja_baja > 0:
        veredicto = "mejora"
    elif ventaja_alta < 0:
        veredicto = "inferioridad"
    else:
        veredicto = "inconcluso"

    return ComparacionEmparejada(metric_id=metric_id, diseno=diseno, estimando=estimando,
                                 veredicto=veredicto, margen_equivalencia=margen_equivalencia,
                                 diferencia_puntual=diferencia_puntual, intervalo=intervalo)
