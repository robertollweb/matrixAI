# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Cómo `generate-dataset` parte el CSV generado en train y eval — **en un sitio**.

POR QUÉ EXISTE. El generador devuelve UN csv con todas las filas; el comando lo
parte en `<stem>-synthetic-train.csv` y `<stem>-synthetic-eval.csv`, y el
`.mxtrain` cita el de TRAIN. `matrixai verify` escribía el csv ENTERO donde el
contrato cita el de train, así que reentrenaba con 300 filas un run que se había
hecho con 240: las métricas salían parecidas y distintas, y R3 daba `FAIL`
acusando al paquete de algo que hacía el verificador.

Medido el 2026-08-26 con el paquete kelvin de la galería: publicado
`mae 6.505213034913027e-17`, `verify --retrain` daba `6.499430623326438e-17` —y
rehacerlo A MANO con `generate-dataset` + `train` devolvía el publicado, hasta el
último dígito—. La tolerancia de esa métrica está MEDIDA en 0,0, así que la
diferencia no era ruido: era otro entrenamiento.

Partir en dos sitios con la misma regla escrita dos veces es la forma más
silenciosa de que dejen de coincidir. Aquí está la regla, y los dos la llaman.
"""

from __future__ import annotations

__all__ = ["NOMBRE_EVAL", "NOMBRE_TRAIN", "corte_train_eval", "nombres_del_dataset"]

#: Los sufijos que escribe `generate-dataset`. Reconocerlos es lo que permite a
#: `verify` saber que un `SOURCE …-synthetic-train.csv` pide la PARTE de train.
NOMBRE_TRAIN = "-synthetic-train.csv"
NOMBRE_EVAL = "-synthetic-eval.csv"


def corte_train_eval(total: int) -> int:
    """Cuántas filas van a train. El resto van a eval.

    80/20 por posición, con la salvaguarda de que **ninguna de las dos partes
    se queda vacía**: con 2 filas salen 1 y 1, y no 2 y 0.
    """
    train = max(1, int(total * 0.8))
    if total - train < 1:
        train -= 1
    return train


def nombres_del_dataset(fuente: str) -> tuple[str, str] | None:
    """`(ruta_train, ruta_eval)` si `fuente` es un train de `generate-dataset`.

    Devuelve `None` cuando no lo es —un CSV propio, con otro nombre—: ahí no hay
    convención que seguir y suponerla escribiría un fichero que nadie pidió.
    """
    if not fuente or not fuente.endswith(NOMBRE_TRAIN):
        return None
    return fuente, fuente[: -len(NOMBRE_TRAIN)] + NOMBRE_EVAL
