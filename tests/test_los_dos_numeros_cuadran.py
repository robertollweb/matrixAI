"""La exactitud y la matriz de confusión miden LO MISMO.

Lo destapó el manual público: en la captura de un entrenamiento la
exactitud decía 50,0 % y su propia matriz de confusión sumaba 55,5 %
—57+54 aciertos de 200— que además coincidía clavado con el Macro F1 de
al lado. Cualquiera con un lápiz suma la matriz y le sale otro número.

La causa no era un cálculo mal hecho: eran DOS evaluaciones distintas.
`accuracy` salía de la partición de VALIDACIÓN (datos que el modelo no
vio) y la matriz de una evaluación posterior que puntúa el dataset
ENTERO, filas de entrenamiento incluidas. Por eso la matriz sumaba 200
cuando la validación eran 40, y por eso el número inflado era además el
más halagüeño.

El código lo sabía: `playground.py` decía «confirmado que difieren en la
práctica; no se cambia la fuente para no alterar el valor que ya ve un
usuario». Se cambia ahora, y en la dirección honesta: manda la
validación, que es la que no ha visto los datos.
"""
from matrixai.training.dense_evaluator import _binary_metrics


def _cuadran(metricas: dict) -> None:
    """La matriz y la exactitud tienen que contar los mismos casos."""
    cm = metricas["confusion_matrix"]
    total = sum(sum(fila.values()) for fila in cm.values())
    aciertos = sum(cm[c].get(c, 0) for c in cm)
    assert total > 0
    assert abs(metricas["accuracy"] - aciertos / total) < 1e-9, (
        f"la matriz suma {aciertos}/{total} = {aciertos / total:.4f} "
        f"y la exactitud dice {metricas['accuracy']:.4f}"
    )


def test_la_matriz_y_la_exactitud_cuentan_lo_mismo():
    # Un caso con aciertos y fallos por los dos lados: si sólo hubiera
    # aciertos, cualquier par de números coincidiría por casualidad.
    predicciones = [[0.9], [0.8], [0.2], [0.1], [0.7], [0.3]]
    objetivos = [[1.0], [0.0], [0.0], [1.0], [1.0], [0.0]]
    _cuadran(_binary_metrics(predicciones, objetivos, []))


def test_tambien_cuando_falla_todo():
    # El caso extremo: 0 % de exactitud y una matriz con la diagonal a
    # cero. Un aserto que sólo mirara «parecido» lo dejaría pasar.
    m = _binary_metrics([[0.9], [0.1]], [[0.0], [1.0]], [])
    assert m["accuracy"] == 0.0
    _cuadran(m)


def test_el_resultado_del_entrenAMIENTO_lleva_la_matriz_de_VALIDACION():
    """Y no sólo la exactitud: era el hueco por el que entraba la otra.

    Se comprueba sobre el CONTRATO del tipo —que la clave existe y se
    serializa— porque entrenar de verdad aquí ataría esta prueba a un
    dataset y a un tiempo de entrenamiento que no son lo que mide.
    """
    from matrixai.training.spec import TrainingRunResult

    r = TrainingRunResult(
        run_id="r1", output_dir="/tmp", best_epoch=1, best_validation_loss=0.1,
        final_train_loss=0.1, final_validation_loss=0.1, accuracy=0.75,
        artifacts={},
        validation_metrics={"rows": 4, "accuracy": 0.75, "confusion_matrix": {}, "macro_f1": 0.7,
                            "labels": ["negative", "positive"], "precision": {}, "recall": {}, "f1": {}},
    )
    d = r.to_dict()
    assert d["validation_metrics"]["accuracy"] == d["accuracy"], (
        "la exactitud del resultado y la de sus métricas de validación "
        "son la MISMA medida: si se separan, vuelve el defecto"
    )
    # Y sin métricas, la clave NO aparece: «no las hay» no es «son cero».
    sin = TrainingRunResult(
        run_id="r2", output_dir="/tmp", best_epoch=1, best_validation_loss=0.1,
        final_train_loss=0.1, final_validation_loss=0.1, accuracy=0.0, artifacts={},
    )
    assert "validation_metrics" not in sin.to_dict()


def test_la_pantalla_lee_la_evaluacion_de_VALIDACION_y_no_la_inflada():
    """El cableado, que es donde estaba el defecto.

    Las pruebas de arriba miden el cálculo y el tipo, y las dos seguían
    en verde con el cableado revertido — o sea que no cubrían el
    arreglo. Ésta sí: da las DOS fuentes a la vez, con valores
    distinguibles, y exige que gane la de validación.
    """
    from matrixai.playground import _metricas_de_clasificacion

    validacion = {
        "confusion_matrix": {"positive": {"positive": 15, "negative": 5},
                             "negative": {"positive": 5, "negative": 15}},
        "macro_f1": 0.75,
        "labels": ["negative", "positive"],
        "precision": {"positive": 0.75, "negative": 0.75},
        "recall": {"positive": 0.75, "negative": 0.75},
        "f1": {"positive": 0.75, "negative": 0.75},
    }
    # La otra evaluación, la del dataset entero: mismos nombres, cifras
    # DISTINTAS y más altas — que es exactamente cómo se veía el defecto.
    inflada = {
        "confusion_matrix": {"positive": {"positive": 95, "negative": 5},
                             "negative": {"positive": 5, "negative": 95}},
        "macro_f1": 0.95,
        "labels": ["negative", "positive"],
        "per_label": {"positive": {"precision": 0.95, "recall": 0.95, "f1": 0.95}},
    }

    r = _metricas_de_clasificacion({"validation_metrics": validacion}, inflada)
    assert r["macro_f1"] == 0.75, "manda la validación, no la evaluación del dataset entero"
    total = sum(sum(f.values()) for f in r["confusion_matrix"].values())
    assert total == 40, f"la matriz tiene que ser la de validación (40 filas), y suma {total}"

    # Y un run ANTERIOR, que no trae la de validación: se cae a la otra
    # en vez de quedarse sin matriz. Se degrada, no se rompe.
    viejo = _metricas_de_clasificacion({}, inflada)
    assert viejo["macro_f1"] == 0.95
    assert viejo["confusion_matrix"] == inflada["confusion_matrix"]
