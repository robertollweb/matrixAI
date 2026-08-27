# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Cómo se alimenta un modelo ONNX — **en un solo sitio**.

POR QUÉ EXISTE. La 2ª auditoría externa (2026-08-25) lo dijo con una frase que
vale más que el arreglo: **«la garantía de C1 no alcanza C2»**. El ejecutor del
motor (87-C1) ya comprobaba tipos y formas, y `matrixai attest` (87-C2) tenía
**su propio camino**, con los dos defectos que el ejecutor ya no tenía:

* filtraba los ejes dinámicos y confundía `[1, "features"]` con **una** entrada,
  así que rechazaba un modelo válido;
* y convertía el CSV a `float` siempre, así que un `int64` aceptaba `3.7`, ONNX
  Runtime lo truncaba **en silencio** y salía un `accuracy: 1.0` sobre un dato
  que nadie escribió.

Dos sitios decidiendo cómo se alimenta un modelo acaban divergiendo, y aquí ya
habían divergido: uno rechazaba lo que el otro aceptaba y al revés. Esto es el
único sitio.
"""

from __future__ import annotations

from typing import Any

__all__ = ["EntradaOnnxIncompatible", "TIPOS_ONNX", "caracteristicas_declaradas",
           "tensor_para"]

#: Los tipos de tensor que sabemos alimentar, y con qué constructor. La lista es
#: CERRADA: un tipo que no está se dice, en vez de convertirlo a float y esperar.
TIPOS_ONNX = {
    "tensor(float)": float, "tensor(double)": float,
    "tensor(int64)": int, "tensor(int32)": int,
}


class EntradaOnnxIncompatible(ValueError):
    """Lo que se le iba a dar al modelo no encaja con lo que el modelo declara."""


def caracteristicas_declaradas(meta: Any) -> tuple[int | None, str]:
    """Cuántas características declara la entrada, y si se ha podido saber.

    **El eje del lote no es una característica.** La versión anterior se quedaba
    con los ejes que son enteros —en `[1, "features"]`, solo el `1`— y comparaba
    ése con el número de columnas: rechazaba modelos válidos diciendo que
    esperaban una entrada.

    Devuelve `(None, "no declarada")` cuando hay ejes dinámicos: no se puede
    contrastar, y decir «comprobado» igualmente sería afirmar por omisión.
    """
    forma = list(getattr(meta, "shape", []) or [])
    caracteristicas = forma[1:] if len(forma) > 1 else forma
    if not caracteristicas or not all(isinstance(d, int) for d in caracteristicas):
        return None, "no declarada"
    esperado = 1
    for d in caracteristicas:
        esperado *= d
    return esperado, "declarada"


def tensor_para(meta: Any, vector: list, contexto: str) -> list:
    """El vector con el TIPO que el modelo declara, o un error que lo dice.

    Convertir a entero un `3.7` es perder el dato: ONNX Runtime lo acepta y lo
    trunca sin avisar. Aquí se rechaza, porque el número que salga de ahí
    acabaría en un recibo como si fuera el que se dio.
    """
    tipo = str(getattr(meta, "type", "") or "").strip().lower()
    constructor = TIPOS_ONNX.get(tipo)
    if constructor is None:
        raise EntradaOnnxIncompatible(
            f"{contexto}: el modelo declara una entrada de tipo {tipo!r}, y solo se "
            f"sabe alimentar {sorted(TIPOS_ONNX)}: convertirla a float y confiar sería "
            f"adivinar")
    if constructor is int:
        enteros = []
        for valor in vector:
            try:
                numero = float(valor)
            except (TypeError, ValueError):
                raise EntradaOnnxIncompatible(
                    f"{contexto}: el modelo espera enteros y recibió {valor!r}") from None
            if numero != int(numero):
                raise EntradaOnnxIncompatible(
                    f"{contexto}: el modelo espera enteros ({tipo}) y recibió {numero}: "
                    f"alimentarlo igual lo truncaría en silencio a {int(numero)}, y ese "
                    f"número acabaría en el recibo como si fuera el que se dio")
            enteros.append(int(numero))
        return [enteros]
    return [[float(v) for v in vector]]
