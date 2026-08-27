# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que le falta a un paquete, EN CASTELLANO — y por qué no es una traducción.

DECISIÓN DE ROBERTO (2026-08-26), la salida intermedia de las tres que había:

    El original **se cita** —esos bytes los cubre el `manifest_sha256` del
    paquete y traducirlos sería ponerle en la boca algo que no dijo— y **al
    lado** va esto, marcado como NUESTRO. Es la única de las tres que no miente
    en ninguna dirección: ni le atribuye al paquete palabras que no escribió, ni
    deja media pantalla en un idioma y un párrafo largo en otro.

Y LA PARTE FINA, que es lo que hace que esto no sea una traducción: **no se
traduce la frase, se vuelve a componer el hecho**. El manifiesto declara
`missing: ["recipe", "seed_dataset"]` —una lista de claves, no prosa— y de ahí
sale tanto el inglés del paquete como el castellano de aquí. Traducir la cadena
sería adivinar; componer desde las mismas claves es decir lo mismo otra vez.

Por eso, si un paquete trae una clave que este catálogo no conoce, **se dice**
en vez de inventar: `motivos_que_faltan` devuelve solo lo que sabe decir y
`claves_sin_traducir` enumera el resto. Media traducción presentada como
completa sería justo el problema que esta decisión evita.
"""

from __future__ import annotations

__all__ = ["CLAVES", "claves_sin_traducir", "hay_traduccion_para",
           "motivo_no_reproducible", "motivos_que_faltan"]

#: Las mismas claves de `_MISSING_REASONS`, dichas en castellano.
#:
#: Se escriben aquí y no junto a las inglesas para que salte a la vista cuando
#: una se quede sin pareja: dos diccionarios uno al lado del otro invitan a
#: editar uno y olvidar el otro, y hay una prueba que compara sus claves.
CLAVES: dict[str, str] = {
    "run_provenance": (
        "con este modelo no viaja ninguna captura autorizada del run, así que el "
        "paquete no puede demostrar su relación con los pesos que lleva: todo lo "
        "demás que declara lo puso quien exportó, y podría describir otro dataset "
        "u otro contrato de entrenamiento"
    ),
    "weights_source": (
        "este paquete no dice si los pesos que lleva vienen de un entrenamiento, y "
        "«no consta» no es «entrenado»: un paquete cuyos pesos pueden ser la "
        "inicialización al azar no se puede presentar como el modelo que describe"
    ),
    "weights_untrained": (
        "los pesos de este paquete son la inicialización al azar, no el resultado "
        "del entrenamiento que describe — exportar antes de que el run terminara, o "
        "después de borrar los pesos, produce exactamente esto: el modelo se puede "
        "rehacer reentrenando, pero el que viaja no es ése"
    ),
    "warm_start": (
        "este run arrancó de unos pesos que ya existían (arranque en caliente) y "
        "esos pesos iniciales no viajan aquí, así que reentrenar desde el contrato, "
        "los datos y las semillas de este paquete no tiene por qué dar estos números "
        "— medido con torch: 1,102227 desde cero contra 1,094253 reanudado, con la "
        "misma captura byte a byte"
    ),
    "warm_start_undecided": (
        "a este run se le dieron pesos de los que partir y nunca llegó a declarar si "
        "el entrenador los usó, así que no se puede distinguir de uno que arrancó "
        "desde cero — y esos dos no dan los mismos números"
    ),
    "warm_start_unknown": (
        "la captura del run no dice de dónde partieron los pesos, así que nada de lo "
        "que hay aquí descarta que este modelo se reentrenara encima de otro cuyos "
        "pesos iniciales no viajan en el paquete"
    ),
    "training": (
        "en este paquete no viaja ningún `.mxtrain`, así que el modelo no se puede "
        "reentrenar"
    ),
    "recipe": (
        "este modelo no tiene receta de datos, así que su dataset no se puede "
        "regenerar ni comparar; por qué no la hay no consta en el paquete"
    ),
    "dataset_sha256": (
        "no se conoce el sha256 esperado del dataset, así que un dataset regenerado "
        "no se puede comparar contra nada"
    ),
    "dataset_rows": "no se conoce el número de filas esperado",
    "seed_dataset": "no se conoce la semilla de generación del dataset",
    "seed_init": (
        "no se conoce la semilla de inicialización de los pesos, así que un modelo "
        "reentrenado no tiene por qué dar los mismos números"
    ),
    "backend": (
        "no se declara el motor de entrenamiento, y la biblioteca estándar y torch no "
        "dan los mismos números"
    ),
    "device": "no se declara el dispositivo de entrenamiento",
    "metrics": (
        "ninguna métrica publicada trae lo que hace falta para compararla (la "
        "partición, el sha256 del dataset sobre el que se midió, una dirección y una "
        "tolerancia)"
    ),
}


def hay_traduccion_para(clave: str) -> bool:
    return clave in CLAVES


def claves_sin_traducir(faltan: list[str]) -> list[str]:
    """Las que este catálogo NO sabe decir. Se enumeran en vez de callarlas."""
    return [c for c in faltan if c not in CLAVES]


def motivos_que_faltan(faltan: list[str]) -> list[str]:
    """Cada clave conocida, en castellano y en el orden en que llegó."""
    return [CLAVES[c] for c in faltan if c in CLAVES]


def motivo_no_reproducible(faltan: list[str]) -> str | None:
    """La frase entera, o `None` si no se sabe decir ninguna de las claves.

    `None` y no una frase vacía: quien lo pinte tiene que poder distinguir «no
    hay nada que traducir» de «hay algo y no lo sé decir», y en el segundo caso
    enseñar solo el original.
    """
    partes = motivos_que_faltan(faltan)
    if not partes:
        return None
    return "No es reproducible: " + "; ".join(partes) + "."
