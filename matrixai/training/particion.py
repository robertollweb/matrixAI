# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Cómo se parten los ejemplos — **en un solo sitio** (contrato 101-C0).

POR QUÉ EXISTE. Hasta hoy cada entrenador partía por su cuenta y los tres lo
hacían distinto, y ninguno hacía lo que el `.mxtrain` declaraba:

* el denso stdlib **ignora `training.dataset.split` por completo** salvo
  `mode=temporal`: corta por un 0,8 fijo, en el orden en que llegan las filas, y
  **no mira la semilla** (`dense_trainer.py`, comentario del contrato 57-C3);
* el compuesto/torch, igual;
* y solo el entrenador FUNCTION baraja con semilla.

Así que el generador escribe `SPLIT train=0.8 validation=0.2 seed=42`, y ese
`seed=42` **no manda**. Medido el 2026-09-04: dos entrenamientos con semillas
distintas dan la misma partición.

Y hay algo peor que la semilla: **no existe un conjunto de PRUEBA**. La misma
partición de validación elige la mejor época *y* publica la métrica, así que el
número que se enseña está elegido sobre los datos con los que se eligió. Eso no
es una estimación de nada, y es lo que el 101 viene a arreglar.

CÓMO SE ARREGLA SIN ROMPER LO DE ANTES. No se cambia el comportamiento de
ningún proyecto existente: quien manda es una **versión de protocolo declarada
en el propio `.mxtrain`** (`SPLIT … protocol=2`). Sin ella, la partición es la
de siempre, byte a byte — que es lo que permite reproducir un proyecto antiguo.
Con ella, se honra lo declarado: semilla, modo y un tercer tramo de prueba que
no toca nadie hasta el final.

*(Mismo criterio que ya usó el 57-C3 al añadir `mode`: lo nuevo se declara, y lo
que no lo declara no se entera.)*
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["PROTOCOLO_SEPARACION", "Particion", "particion_legada", "particion_declarada",
           "particion_para"]

#: La única versión de protocolo que cambia cómo se parte. Se declara en el
#: `.mxtrain` (`SPLIT … protocol=2`) y es lo que distingue «reproducir un
#: proyecto antiguo» de «hacer un estudio nuevo».
PROTOCOLO_SEPARACION = "2"


@dataclass(frozen=True)
class Particion:
    """Qué fila va a dónde, y **cómo se decidió**.

    Se guardan los ÍNDICES y no las filas: el criterio de cierre del corte pide
    que «el mismo manifiesto reproduzca los índices de partición», y para
    comprobar eso hace falta poder compararlos. Con las filas dentro no se
    puede: dos particiones distintas de los mismos datos se ven iguales si solo
    se mira el contenido.
    """

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...] = ()
    #: `"legado"` o `PROTOCOLO_SEPARACION`. Va al manifiesto: un número medido
    #: sobre una partición legada y otro sobre una con prueba reservada no son
    #: comparables, y quien lo lea tiene que poder saberlo.
    protocolo: str = "legado"
    modo: str = "secuencial"
    semilla: int | None = None

    def como_dict(self) -> dict[str, Any]:
        datos: dict[str, Any] = {
            "protocol": self.protocolo,
            "mode": self.modo,
            "n_train": len(self.train),
            "n_validation": len(self.validation),
        }
        if self.test:
            datos["n_test"] = len(self.test)
        if self.semilla is not None:
            datos["seed"] = self.semilla
        return datos

    def sin_solape(self) -> bool:
        """¿De verdad no comparten ni una fila?

        No es paranoia: es el fallo que el 104-C0 impide con su registro de
        accesos, aquí un piso más abajo. Comprobar conjuntos y no hashes,
        porque dos listas distintas pueden tener las mismas filas.
        """
        t, v, p = set(self.train), set(self.validation), set(self.test)
        return not (t & v) and not (t & p) and not (v & p)


def particion_legada(n: int) -> Particion:
    """La de siempre: 0,8 fijo, secuencial, sin mirar la semilla.

    **No se toca.** Es lo que reproduce un proyecto antiguo, y por eso está
    escrita aparte y con su nombre: cuando alguien la lea dentro de tres meses
    tiene que ver que es deliberada, no un descuido que sobrevivió.
    """
    corte = max(1, int(n * 0.8)) if n > 1 else n
    return Particion(train=tuple(range(corte)), validation=tuple(range(corte, n)))


def particion_temporal(n: int, train_ratio: float) -> Particion:
    """El último tramo, en el orden que llega, es la validación (57-C3).

    Nunca baraja aunque haya semilla: en una serie temporal, barajar es fuga.
    """
    corte = max(1, min(n - 1, int(n * train_ratio))) if n > 1 else n
    return Particion(train=tuple(range(corte)), validation=tuple(range(corte, n)),
                     modo="temporal")


def particion_declarada(n: int, *, train: float, validation: float,
                        test: float | None = None, seed: int | None = None,
                        modo: str = "random") -> Particion:
    """Lo que el `.mxtrain` declara, honrado de verdad.

    Con `modo="temporal"` los tres tramos van en orden —train, validación,
    prueba— y no se baraja. Con `modo="random"` se baraja **con la semilla
    declarada**, y sin semilla se deja el orden de llegada: barajar con una
    semilla que nadie escribió haría irreproducible el resultado, y esta casa
    no hace eso.
    """
    indices = list(range(n))
    if modo == "temporal":
        corte_train = max(1, min(n - 1, int(n * train))) if n > 1 else n
        corte_val = min(n, corte_train + int(n * validation))
    else:
        if seed is not None:
            random.Random(seed).shuffle(indices)
        corte_train = max(1, int(n * train)) if n > 1 else n
        corte_val = min(n, corte_train + max(1 if test else 0, int(n * validation)))
    return Particion(
        train=tuple(indices[:corte_train]),
        validation=tuple(indices[corte_train:corte_val]),
        test=tuple(indices[corte_val:]) if test else (),
        protocolo=PROTOCOLO_SEPARACION, modo=modo, semilla=seed)


def particion_para(n: int, split_spec: Any) -> Particion:
    """La partición que toca, mirando lo que el `.mxtrain` declara.

    **Este `if` es todo el contrato de compatibilidad**: sin `protocol=2`
    declarado, se devuelve exactamente lo que se devolvía antes de este corte.
    """
    if split_spec is None:
        return particion_legada(n)
    if getattr(split_spec, "protocol", None) == PROTOCOLO_SEPARACION:
        return particion_declarada(
            n, train=split_spec.train, validation=split_spec.validation,
            test=getattr(split_spec, "test", None), seed=split_spec.seed,
            modo=split_spec.mode)
    if split_spec.mode == "temporal":
        return particion_temporal(n, split_spec.train)
    return particion_legada(n)


def reparte(ejemplos: Sequence[Any], particion: Particion) -> tuple[list, list, list]:
    """Los ejemplos en tres listas, en el orden que dice la partición."""
    return ([ejemplos[i] for i in particion.train],
            [ejemplos[i] for i in particion.validation],
            [ejemplos[i] for i in particion.test])
