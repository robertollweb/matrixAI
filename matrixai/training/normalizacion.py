# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Normalizar por los rangos que el MODELO declara — hallazgo 13.

POR QUÉ EXISTE, medido dos veces el 2026-08-25 desde el CLI:

* el caso clínico bajaba de **0,968 a 0,687** de exactitud solo por escribir
  `edad: Scalar[18, 100]` en vez de `edad: Scalar`;
* y el Kelvin **no convergía de ninguna manera** (val loss 17.000-37.000) sobre
  un problema cuya solución exacta es `y = x + 273,15`.

La causa no era el modelo: era que el `.mxai` **declara su dominio** y el
entrenador del CLI no lo miraba, así que la red recibía 18-100 junto a 0-1. Es
la misma familia que el contrato 61 —«la ruta Studio ya normaliza; el colapso
era deuda de paridad»— y la decisión de Roberto (2026-08-25) fue **paridad**.

LA REGLA QUE NO SE PUEDE ROMPER: si se normaliza al ENTRENAR, la inferencia
tiene que normalizar **igual**. Por eso esto devuelve los rangos que aplicó, y
quien exporta los mete en el `inference_spec`. Normalizar solo en un lado
produce un paquete que parece bueno y predice mal, que es peor que no
normalizar.
"""

from __future__ import annotations

from typing import Any

__all__ = ["rango_declarado_del_objetivo", "rangos_declarados_del_vector",
           "normalizar_filas"]


def rangos_declarados_del_vector(vector: Any) -> dict[str, tuple[float, float]]:
    """Los dominios declarados del vector, por nombre de campo.

    Lo que no declara rango **no entra**: no se supone `[0, 1]` aquí. Suponerlo
    escalaría una columna que quizá ya viene en su escala, y una suposición
    silenciosa en el camino de los datos es exactamente lo que este hallazgo
    vino a quitar.
    """
    dominios: dict[str, tuple[float, float]] = {}
    for nombre, tipo in (getattr(vector, "field_types", None) or {}).items():
        rango = getattr(tipo, "range", None)
        minimo, maximo = getattr(rango, "minimum", None), getattr(rango, "maximum", None)
        if isinstance(minimo, (int, float)) and isinstance(maximo, (int, float)):
            if maximo > minimo:
                dominios[str(nombre)] = (float(minimo), float(maximo))
    return dominios


def normalizar_filas(
    filas: list[list[float]],
    campos: list[str],
    dominios: dict[str, tuple[float, float]],
) -> tuple[list[list[float]], dict[str, tuple[float, float]]]:
    """Escala cada columna a [0, 1] con su dominio declarado.

    Devuelve `(filas, rangos_aplicados)`. Los rangos aplicados son los que hay
    que meter en el `inference_spec` para que quien prediga haga lo mismo — se
    devuelven en vez de darse por sabidos justamente para que no se olviden.

    Una columna sin dominio declarado **se deja como está**, y eso también se
    ve en lo devuelto: lo que no está ahí, no se tocó.
    """
    if not dominios:
        return filas, {}
    aplicados: dict[str, tuple[float, float]] = {}
    indices: list[tuple[int, float, float]] = []
    for i, campo in enumerate(campos):
        rango = dominios.get(campo)
        if rango is None:
            continue
        indices.append((i, rango[0], rango[1]))
        aplicados[campo] = rango
    if not indices:
        return filas, {}
    salida: list[list[float]] = []
    for fila in filas:
        nueva = list(fila)
        for i, minimo, maximo in indices:
            if i < len(nueva):
                nueva[i] = (float(nueva[i]) - minimo) / (maximo - minimo)
        salida.append(nueva)
    return salida, aplicados


def rango_declarado_del_objetivo(training: Any) -> tuple[float, float] | None:
    """El dominio que el `.mxtrain` declara para el OBJETIVO, si lo declara.

    HACE FALTA Y NO ES OPCIONAL, y esto salió midiendo: `predict.py` del paquete
    **desnormaliza la salida** con ese rango (contrato 59), o sea que espera que
    el modelo emita un valor en [0, 1]. Si se normalizan las entradas y no el
    objetivo, el entrenamiento va a peor —medido con el Kelvin: divergía a
    tasas a las que antes no divergía— y encima el paquete predice en otra
    escala.

    **Media normalización es peor que ninguna.** Solo se devuelve para
    objetivos continuos: en clasificación la etiqueta es un índice de clase y
    escalarla sería destruirla.
    """
    objetivo = getattr(getattr(training, "dataset", None), "target", None)
    tipo = getattr(objetivo, "type", None)
    nombre = str(getattr(tipo, "name", "") or "")
    if nombre.startswith(("Label", "ProbabilityMap")):
        return None
    rango = getattr(tipo, "range", None)
    minimo, maximo = getattr(rango, "minimum", None), getattr(rango, "maximum", None)
    if isinstance(minimo, (int, float)) and isinstance(maximo, (int, float)) and maximo > minimo:
        return float(minimo), float(maximo)
    return None
