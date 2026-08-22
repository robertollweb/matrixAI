"""La identidad de cada métrica: hacia dónde es mejor, y cómo se agrega.

El §5 bis del contrato 82 pide `direction` y `aggregation` por métrica, y
**R3 no puede comparar sin `direction`**: «se ha movido 0,03» no dice si
mejoró o empeoró, y una tolerancia se aplica igual en los dos sentidos
solo porque nadie miró.

Estos dos datos **no son del run**: son del NOMBRE de la métrica. Por eso
viven aquí, en un solo sitio, y no donde se exporta — dos sitios
declarando lo mismo acaban divergiendo, y aquí divergir significaría
publicar dos paquetes que dicen cosas distintas de la misma métrica.

**Lo que NO hace: adivinar.** Una métrica que no está en el catálogo
devuelve `None`, y el manifiesto la publica sin dirección **diciendo que
le falta**. Deducir que «todo lo que acaba en `_loss` baja» acertaría
casi siempre, y el casi es el problema: una dirección equivocada
convierte una mejora en un `FAIL` de R3.

La agregación se declara **solo donde está medida**, leyendo cómo se
calcula (`dense_evaluator.py`). Donde no consta va a `None` y se ve:
inventarla sería firmar una afirmación sin respaldo, que es la misma
línea que ya se sigue con las tolerancias.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AGREGACION", "ALCANCE_DE_LA_TOLERANCIA", "DIRECCION",
           "TOLERANCIA_MEDIDA", "identidad_de_metrica", "tolerancia_medida"]

#: LA TOLERANCIA, **MEDIDA** (2026-08-20, decisión de Roberto).
#:
#: El §5 bis del contrato 82 dice que sale de repetir en la matriz de
#: entornos soportada. Aquí se repitió en UNO: `SupervisedTrainer` 5 veces
#: y `DenseSupervisedTrainer` 3, con la misma semilla y en la misma
#: máquina. **Rango 0.000e+00 en las dos**: los valores salieron idénticos
#: hasta el último dígito, no parecidos.
#:
#: Por eso `0.0` aquí NO es el cero inventado que este módulo llevaba
#: rechazando: es el resultado de medirlo. La diferencia entre los dos
#: ceros es exactamente el alcance que va escrito al lado.
TOLERANCIA_MEDIDA = 0.0

#: Y DÓNDE VALE. Sin esto, un `PASS` de R3 se leería como «reproduce igual
#: en cualquier sitio» cuando lo medido es «reproduce igual AQUÍ». Quien
#: verifique en otro entorno tiene que poder saber que esta tolerancia no
#: se midió para él — y el verificador lo comprueba antes de comparar.
ALCANCE_DE_LA_TOLERANCIA = "same_environment_same_seed"


def tolerancia_medida(nombre: str) -> dict[str, Any] | None:
    """`{"tolerance_abs", "tolerance_scope"}`, o `None` si la métrica no consta.

    Se ata al catálogo a propósito: declarar una tolerancia para una
    métrica cuya repetibilidad nadie ha medido sería el cero inventado
    otra vez, solo que con más letra pequeña.
    """
    if str(nombre or "").strip() not in DIRECCION:
        return None
    return {"tolerance_abs": TOLERANCIA_MEDIDA,
            "tolerance_scope": ALCANCE_DE_LA_TOLERANCIA}

#: Hacia dónde es mejor. Cerrado a propósito.
DIRECCION: dict[str, str] = {
    "accuracy": "higher_is_better",
    "macro_f1": "higher_is_better",
    "r2": "higher_is_better",
    "mae": "lower_is_better",
    "rmse": "lower_is_better",
    "final_train_loss": "lower_is_better",
    "best_validation_loss": "lower_is_better",
}

#: Cómo se agrega, MEDIDO en `dense_evaluator.py`:
#: `compute_mae` divide entre n (media), `compute_rmse` es la raíz de la
#: media de cuadrados, `compute_accuracy` es aciertos/total y `r2` se
#: calcula sobre el conjunto entero. `macro_f1` promedia por clase sin
#: pesarlas — eso es `macro`, y es lo que dice su nombre.
#:
#: Las dos pérdidas NO están aquí: cómo las promedia cada entrenador no se
#: ha medido, y ponerles «mean» porque lo normal es eso sería justo lo que
#: este módulo dice que no hace.
AGREGACION: dict[str, str] = {
    "accuracy": "global",
    "macro_f1": "macro",
    "r2": "global",
    "mae": "mean",
    "rmse": "root_mean_square",
}


def identidad_de_metrica(nombre: str) -> dict[str, Any] | None:
    """`{"direction", "aggregation"}` de una métrica, o `None` si no consta.

    `aggregation` puede venir a `None` con `direction` puesta: son dos
    datos distintos y uno puede saberse sin el otro. Devolver el par
    entero a `None` por no saber la agregación perdería una dirección que
    sí está medida — y `direction` es la que R3 necesita.
    """
    clave = str(nombre or "").strip()
    if clave not in DIRECCION:
        return None
    return {"direction": DIRECCION[clave], "aggregation": AGREGACION.get(clave)}
