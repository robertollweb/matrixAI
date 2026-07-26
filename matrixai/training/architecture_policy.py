# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 64 — política de arquitectura para el generador denso.

Por qué existe. Hasta ahora el tamaño de la red que sale del flujo desde-datos
era un accidente: `_default_hidden_layers` tenía tres escalones por dimensión de
entrada y el tercero (`128-64-32`, 12.194 parámetros) absorbía todo lo que
tuviera más de 10 columnas. Un dataset de 3.651 filas y otro de 94.833 recibían
exactamente la misma red, y no había forma de explicar por qué esa y no otra.

Este módulo no existe para hacer redes más grandes. Existe para que el tamaño
**se justifique, quepa en un presupuesto explícito y quede registrado**.

La regla, decidida con Roberto el 2026-07-26 (opción "la entrada manda, las filas
solo ponen techo"):

    ancho1 = clamp(potencia_de_2(4 · d), 32, 512)
    capas  = 2 si d ≤ 8 ; 3 si d ≤ 64 ; 4 si d > 64   (+1 si k > 8 clases)
    taper  = mitad por capa, suelo `_MIN_WIDTH` (16)
    techo  = mín(max_params del perfil, filas/10)

El suelo de la PRIMERA capa es 32 y no 16: ver `_MIN_FIRST_WIDTH`, que documenta
por qué (un problema de una sola entrada perdía capacidad y el R² caía).

`d` es la dimensión EFECTIVA de entrada (después de one-hot y de las columnas de
lag), no el número de columnas del CSV: es la que de verdad determina cuántos
pesos entran en la primera capa.

Las filas no agrandan la red — invariante 1 del contrato: "el tamaño del dataset
puede ampliar el presupuesto razonable de capacidad, pero la arquitectura se
justifica por dimensión de entrada, tarea, clases, señal observada y recursos".
Solo la encogen si el presupuesto no da para la que pedía la entrada. El criterio
`filas/10` es la heurística clásica de muestras por parámetro: con menos de diez
ejemplos por peso, la red memoriza en vez de aprender.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matrixai import limits as _limits

# Suelo de ancho. Coincide con `_MIN_RELU_WIDTH` del generador denso a propósito:
# una capa ReLU más estrecha que esto pierde tanta señal que el generador ya la
# ensanchaba por su cuenta. Aquí se respeta desde el principio para que la
# decisión registrada sea la de verdad y no una que el generador corrige después.
_MIN_WIDTH = 16
# Suelo de la PRIMERA capa, y suelo hasta el que puede encoger el techo de datos.
#
# No es cosmético: con `4·d` y suelo 16, un problema de una sola entrada (el CSV
# centígrados→Kelvin, d=1) recibía 16-16 donde el generador anterior daba 32-16,
# y el R² caía de 0,99 a 0,94 — la red dejaba de poder representar la relación.
# Una capa de entrada necesita anchura suficiente para separar el espacio
# aunque la entrada sea diminuta, y encoger por debajo de aquí cambia un riesgo
# de sobreajuste (corregible con más datos) por una incapacidad de aprender
# (que no se corrige con nada).
_MIN_FIRST_WIDTH = 32
# Techo de ancho de la política. No es un límite duro (un override del usuario
# puede pedir más, ver C3): es hasta dónde llega la regla por sí sola.
_MAX_POLICY_WIDTH = 512
# Muestras por parámetro por debajo de las cuales se considera que la red no está
# justificada por los datos.
_SAMPLES_PER_PARAM = 10


@dataclass(frozen=True)
class ArchitectureDecision:
    """Qué red, por qué, y qué se descartó por el camino (C4).

    Se construye SIEMPRE, también cuando gana un override del usuario o el LLM:
    "la decisión se registra o no existe" (invariante 3).
    """

    hidden_layers: list[tuple[int, str]]
    params: int
    source: str                       # 'policy' | 'user_override' | 'llm' | 'default'
    inputs: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    limits_applied: list[str] = field(default_factory=list)
    rationale: str = ""
    # Avisos para la persona, no diagnóstico interno: hoy solo uno, cuando el
    # dataset es demasiado pequeño para su propia dimensión de entrada.
    warnings: list[str] = field(default_factory=list)
    # Presente SOLO cuando el tope DURO `max_params` no se puede satisfacer ni
    # con la red mínima. No es un aviso: es un rechazo, porque `max_params` y
    # `_MIN_WIDTH` no pueden cumplirse a la vez y el límite duro manda
    # (invariante 2). Quien reciba esto no debe generar nada.
    limit_error: dict[str, Any] | None = None
    # REAUDITORÍA [MEDIA] — estimación de recursos de la red elegida, a título
    # INFORMATIVO. El contrato pedía "pasar toda arquitectura por el estimador",
    # pero `estimate_model_resources` está definido como orientativo y sin
    # umbral: tratarlo como límite duro sería inventarse un tope que nadie ha
    # fijado. El límite duro es `max_params`; esto acompaña a la decisión para
    # poder responder "¿cuánta memoria va a necesitar esto?" sin entrenar.
    resource_estimate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": [[u, a] for u, a in self.hidden_layers],
            "params": self.params,
            "source": self.source,
            "inputs": dict(self.inputs),
            "budget": dict(self.budget),
            "candidates": [dict(c) for c in self.candidates],
            "limits_applied": list(self.limits_applied),
            "rationale": self.rationale,
            "warnings": list(self.warnings),
            **({"limit_error": dict(self.limit_error)} if self.limit_error else {}),
            **({"resource_estimate": dict(self.resource_estimate)}
               if self.resource_estimate else {}),
        }


def miles(n: int) -> str:
    """Separador de miles en español (`3.474`).

    Existe como función porque encadenar `.replace(",", ".")` sobre un f-string
    completo también sustituye las comas que NO son separadores: el aviso del
    presupuesto salía como "512-512-512. 532.481 parámetros".
    """
    return f"{int(n):,}".replace(",", ".")


def param_count(input_dim: int, hidden_layers: list[tuple[int, str]],
                output_units: int) -> int:
    """Pesos + sesgos de una pila densa. Aritmética pura, sin construir nada.

    Se cuenta aquí y no con `estimate_model_resources` porque hay que poder
    evaluar CANDIDATAS antes de existir ningún `.mxai` que parsear.
    """
    total = 0
    prev = max(0, int(input_dim))
    for units, _ in hidden_layers:
        total += prev * units + units
        prev = units
    return total + prev * output_units + output_units


def effective_input_dim(
    columns: int,
    *,
    one_hot_widths: dict[str, int] | None = None,
    lag_columns: int = 0,
) -> int:
    """Dimensión que de verdad entra en la primera capa.

    Un CSV con 5 columnas de las que una es una categórica de 8 valores no entra
    con 5 pesos por neurona sino con 12: la categórica ocupa 8 y desaparece la
    original. Dimensionar por columnas del CSV subestimaría la red justo en los
    datasets donde más importa.
    """
    total = max(0, int(columns)) + max(0, int(lag_columns))
    for ancho in (one_hot_widths or {}).values():
        total += max(0, int(ancho)) - 1     # la columna original se sustituye
    return max(1, total)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _taper(width: int, depth: int) -> list[tuple[int, str]]:
    capas: list[tuple[int, str]] = []
    actual = width
    for _ in range(depth):
        capas.append((max(_MIN_WIDTH, actual), "relu"))
        actual = max(_MIN_WIDTH, actual // 2)
    return capas


def _depth_for(input_dim: int, output_units: int) -> int:
    profundidad = 2 if input_dim <= 8 else (3 if input_dim <= 64 else 4)
    # Más clases = frontera de decisión más compleja; una capa más de mezcla.
    if output_units > 8:
        profundidad += 1
    return profundidad


def budget_for(rows: int) -> dict[str, Any]:
    """Presupuesto de parámetros: el DURO del perfil y el blando de los datos."""
    duro = _limits.get_limit("max_params")
    por_filas = (rows // _SAMPLES_PER_PARAM) if rows and rows > 0 else None
    efectivo = duro
    if por_filas is not None:
        efectivo = por_filas if duro is None else min(duro, por_filas)
    return {
        "max_params": duro,
        "rows": int(rows or 0),
        "samples_per_param": _SAMPLES_PER_PARAM,
        "rows_ceiling": por_filas,
        "effective": efectivo,
    }


def propose(
    *,
    input_dim: int,
    output_units: int,
    task: str = "",
    rows: int = 0,
) -> ArchitectureDecision:
    """La arquitectura que propone la política, ya dentro del presupuesto.

    Determinista: mismas entradas → misma salida, sin azar ni estado global más
    allá del perfil de límites (invariante 4).
    """
    d = max(1, int(input_dim))
    k = max(1, int(output_units))
    presupuesto = budget_for(rows)
    techo = presupuesto["effective"]

    ancho = min(_MAX_POLICY_WIDTH, max(_MIN_FIRST_WIDTH, _next_pow2(4 * d)))
    profundidad = _depth_for(d, k)

    candidatas: list[dict[str, Any]] = []
    limites: list[str] = []
    capas = _taper(ancho, profundidad)
    params = param_count(d, capas, k)
    candidatas.append({"hidden_layers": [[u, a] for u, a in capas],
                       "params": params, "accepted": techo is None or params <= techo,
                       "reason": "regla: ancho por dimensión efectiva"})

    # Si no cabe, se ESTRECHA (no se recorta profundidad): la profundidad viene
    # de la complejidad de la tarea y quitarla cambia lo que la red puede
    # representar; el ancho solo cambia cuánto.
    while techo is not None and params > techo and ancho > _MIN_FIRST_WIDTH:
        ancho = max(_MIN_FIRST_WIDTH, ancho // 2)
        capas = _taper(ancho, profundidad)
        params = param_count(d, capas, k)
        candidatas.append({"hidden_layers": [[u, a] for u, a in capas],
                           "params": params, "accepted": params <= techo,
                           "reason": "estrechada para caber en el presupuesto"})
        if "budget" not in limites:
            limites.append("budget")

    avisos: list[str] = []
    error_tope: dict[str, Any] | None = None
    duro = presupuesto["max_params"]

    # REAUDITORÍA [ALTO] — hay que distinguir QUÉ techo aprieta, porque la
    # respuesta correcta es distinta:
    #
    #   · el techo de DATOS (filas/10) es una heurística contra el sobreajuste:
    #     se acepta la red mínima y se avisa, porque negarse dejaría a la persona
    #     sin ninguna salida por un riesgo que más datos corrigen;
    #   · `max_params` es un límite DURO del perfil (invariante 2): si ni la red
    #     mínima cabe, no se puede satisfacer a la vez que `_MIN_WIDTH`, y
    #     entregar igualmente una red que lo incumple sería saltárselo en
    #     silencio. Se RECHAZA.
    #
    # Antes ambos casos caían en la misma rama y `max_params=100` producía una
    # red de 1.250 parámetros sin error, sin límite anotado y sin aviso.
    if duro is not None and params > duro:
        limites.append("max_params_unsatisfiable")
        error_tope = _limits.limit_error("max_params", params)
    elif techo is not None and params > techo:
        # Solo aprieta el techo de datos: se acepta la mínima y se deja constancia.
        limites.append("min_width_floor")
        if presupuesto["rows_ceiling"] is not None:
            # El techo que aprieta es el de los DATOS, no el del perfil: cambiar
            # de perfil no arreglaría nada y decir "súbelo en Ajustes" sería
            # desviar a la persona. Lo que falta son filas.
            avisos.append(
                f"El dataset ({miles(presupuesto['rows'])} filas) es pequeño para "
                f"{d} entradas: la red más pequeña razonable ({miles(params)} "
                "parámetros) ya supera los diez ejemplos por parámetro. Puede "
                "memorizar en vez de aprender; con más datos, o con menos "
                "columnas de entrada, el modelo sería más fiable."
            )

    motivo = (
        f"dimensión efectiva de entrada {d} → ancho {capas[0][0]}; "
        f"{profundidad} capas por dimensión {'y clases ' if k > 8 else ''}"
        f"({k} salida{'s' if k != 1 else ''}); {miles(params)} parámetros"
    )
    if presupuesto["rows_ceiling"] is not None:
        motivo += (f"; techo por datos {miles(presupuesto['rows_ceiling'])} "
                   f"({miles(presupuesto['rows'])} filas / {_SAMPLES_PER_PARAM} "
                   "por parámetro)")

    return ArchitectureDecision(
        hidden_layers=capas,
        params=params,
        source="policy",
        inputs={"input_dim": d, "output_units": k, "task": task or "",
                "rows": int(rows or 0)},
        budget=presupuesto,
        candidates=candidatas,
        limits_applied=limites,
        rationale=motivo,
        warnings=avisos,
        limit_error=error_tope,
    )


def enforce_budget(
    hidden_layers: list[tuple[int, str]],
    *,
    input_dim: int,
    output_units: int,
) -> tuple[list[tuple[int, str]], dict | None]:
    """Aplica el límite DURO `max_params` a una arquitectura venga de donde venga.

    Invariante 2: "ninguna fuente puede saltarse los límites duros". Un override
    del Modo experto o una propuesta del LLM pasan por aquí igual que la política.

    Devuelve `(capas, error_de_tope_o_None)`. No encoge nada por su cuenta: si el
    usuario pidió explícitamente una red que no cabe, se le dice cuál es el tope
    en vez de darle en silencio otra cosa distinta de la que pidió.
    """
    params = param_count(input_dim, hidden_layers, output_units)
    tope = _limits.get_limit("max_params")
    if tope is not None and params > tope:
        return hidden_layers, _limits.limit_error("max_params", params)
    return hidden_layers, None
