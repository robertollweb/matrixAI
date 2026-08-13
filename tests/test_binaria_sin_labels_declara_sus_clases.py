# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Una binaria SIN bloque LABELS también tiene clases, y el core las dice.

Medido conduciendo el producto por el API el 2026-08-13 (auditoría pasada 5):

    POST /api/studio/generate  {"prompt": "clasificar si un correo es spam o no…"}
    POST /api/train-start      → job
    GET  /api/train-status/<job>
      labels           : []
      per_label        : {}
      confusion_matrix : {"positive": {...}, "negative": {...}}

El mismo prompt en MULTICLASE («crítica, media o baja») sí las decía. La
diferencia: `_labels_from_spec` devuelve `[]` cuando no hay bloque LABELS, y
`_binary_metrics` inventa «positive»/«negative» para poder construir la matriz
— pero esa invención no volvía a salir.

Por qué importa, y no es cosmético: quien pinta la matriz ordena los ejes por
`labels`. Con `[]` no hay ejes, y la matriz —que venía llena— no se enseña.
`macro_precision`/`macro_recall` sí se calculaban de esos mismos
precision/recall, así que la respuesta se contradecía a sí misma: una macro de
un `per_label` vacío.
"""
from __future__ import annotations

from matrixai.training.dense_evaluator import (
    DenseEvaluationResult,
    _binary_metrics,
    _multiclass_metrics,
    effective_labels,
)

_PREDS = [[0.9], [0.8], [0.2], [0.1], [0.6], [0.3]]
_TGTS = [[1.0], [0.0], [0.0], [1.0], [1.0], [0.0]]


def _binaria_sin_labels() -> DenseEvaluationResult:
    m = _binary_metrics(_PREDS, _TGTS, labels=[])
    return DenseEvaluationResult(
        rows=len(_PREDS),
        loss=0.5,
        loss_fn="binary_cross_entropy",
        accuracy=m["accuracy"],
        confusion_matrix=m["confusion_matrix"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        macro_f1=m["macro_f1"],
    )


class TestClasesEfectivas:
    def test_binaria_sin_labels_devuelve_las_dos_clases_que_uso(self) -> None:
        result = _binaria_sin_labels()
        # La matriz vino con dos clases dentro: las etiquetas no pueden ir vacías.
        assert set(result.confusion_matrix) == {"positive", "negative"}
        assert effective_labels([], result) == ["negative", "positive"]
        assert effective_labels(None, result) == ["negative", "positive"]

    def test_el_orden_es_el_de_lectura_de_una_matriz_binaria(self) -> None:
        # `[negativa, positiva]`, el orden con el que se calcularon
        # precision/recall — no el de inserción de la matriz.
        assert effective_labels([], _binaria_sin_labels()) == ["negative", "positive"]

    def test_las_declaradas_MANDAN_sobre_las_inventadas(self) -> None:
        result = _binaria_sin_labels()
        assert effective_labels(["no_spam", "spam"], result) == ["no_spam", "spam"]

    def test_en_regresion_no_hay_clases_que_nombrar(self) -> None:
        result = DenseEvaluationResult(rows=4, loss=0.1, loss_fn="mse", mae=0.2, rmse=0.3, r2=0.9)
        assert effective_labels([], result) == []
        assert effective_labels(None, result) == []

    def test_multiclase_declarada_no_cambia(self) -> None:
        labels = ["critica", "media", "baja"]
        preds = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]]
        tgts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        m = _multiclass_metrics(preds, tgts, labels)
        result = DenseEvaluationResult(
            rows=3, loss=0.1, loss_fn="cross_entropy",
            accuracy=m["accuracy"], confusion_matrix=m["confusion_matrix"],
            precision=m["precision"], recall=m["recall"], f1=m["f1"],
            macro_f1=m["macro_f1"],
        )
        assert effective_labels(labels, result) == labels


class TestLaRespuestaNoSeContradice:
    """`per_label` y `labels` salen POBLADOS por la misma puerta por la que
    ya salían `macro_precision`/`macro_recall`."""

    def test_el_report_torch_gpu_declara_clases_y_per_label(self) -> None:
        from matrixai.playground import _eval_report_from_dense_result

        report = _eval_report_from_dense_result(_binaria_sin_labels(), [])
        assert report["labels"] == ["negative", "positive"]
        assert set(report["per_label"]) == {"negative", "positive"}
        assert set(report["confusion_matrix"]) == set(report["labels"])
        for clase in report["labels"]:
            assert set(report["per_label"][clase]) == {"precision", "recall", "f1"}

    def test_en_regresion_el_report_torch_sigue_sin_clases(self) -> None:
        from matrixai.playground import _eval_report_from_dense_result

        result = DenseEvaluationResult(rows=4, loss=0.1, loss_fn="mse", mae=0.2, rmse=0.3, r2=0.9)
        report = _eval_report_from_dense_result(result, [])
        assert report["labels"] is None
        assert report["per_label"] is None
        assert report["confusion_matrix"] is None
