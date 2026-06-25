# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 C8 — Evaluación de redes densas: mae/rmse/r2 (regresión) y accuracy (clasificación)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from matrixai.forward.dense_forward import dense_forward
from matrixai.parameters.store import ParameterSet
from matrixai.training.dense_backprop import compute_loss


@dataclass(frozen=True)
class DenseEvaluationResult:
    rows: int
    loss: float
    loss_fn: str
    # Regression metrics (set when loss_fn == "mse")
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    # Classification metrics (set when loss_fn in {"cross_entropy", "binary_cross_entropy"})
    accuracy: float = 0.0
    labels: list[str] = field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    # Per-class and macro metrics
    precision: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)
    f1: dict[str, float] = field(default_factory=dict)
    macro_f1: float = 0.0

    def is_regression(self) -> bool:
        return self.loss_fn == "mse"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "rows": self.rows,
            "loss": self.loss,
            "loss_fn": self.loss_fn,
        }
        if self.is_regression():
            data["mae"] = self.mae
            data["rmse"] = self.rmse
            data["r2"] = self.r2
        else:
            data["accuracy"] = self.accuracy
            data["macro_f1"] = self.macro_f1
            if self.labels:
                data["labels"] = list(self.labels)
            if self.confusion_matrix:
                data["confusion_matrix"] = {
                    k: dict(v) for k, v in self.confusion_matrix.items()
                }
            if self.precision:
                data["precision"] = dict(self.precision)
            if self.recall:
                data["recall"] = dict(self.recall)
            if self.f1:
                data["f1"] = dict(self.f1)
        return data


def evaluate_dense_network(
    network: Any,
    parameter_set: ParameterSet,
    examples: list[tuple[list[float], list[float]]],
    loss_fn: str,
    labels: list[str] | None = None,
) -> DenseEvaluationResult:
    """Evaluate a dense network over a list of (input, target) examples.

    Args:
        examples: list of (input_vector, target_vector) pairs.
            - regression: target_vector = [float]
            - binary_cross_entropy: target_vector = [float]  (0.0 or 1.0)
            - cross_entropy: target_vector = one-hot vector
        labels: class label strings for classification (same order as output units).
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    predictions: list[list[float]] = []
    targets: list[list[float]] = []
    for input_vec, target in examples:
        predictions.append(dense_forward(network, parameter_set, input_vec))
        targets.append(target)

    return result_from_predictions(predictions, targets, loss_fn, labels)


def result_from_predictions(
    predictions: list[list[float]],
    targets: list[list[float]],
    loss_fn: str,
    labels: list[str] | None = None,
) -> DenseEvaluationResult:
    """Build a DenseEvaluationResult from precomputed predictions (the network's
    probability outputs) and targets.

    Shared by the stdlib evaluator (per-row `dense_forward`) and the torch/GPU
    evaluator (batched forward) so both compute IDENTICAL metrics — only the forward
    pass differs. M14 (GPU end-to-end: la evaluación no debe ir por CPU fila a fila).
    """
    if not predictions:
        raise ValueError("predictions must be non-empty")
    total_loss = sum(compute_loss(loss_fn, p, t) for p, t in zip(predictions, targets))
    avg_loss = total_loss / len(predictions)
    rows = len(predictions)

    if loss_fn == "mse":
        return DenseEvaluationResult(
            rows=rows, loss=avg_loss, loss_fn=loss_fn,
            **_regression_metrics(predictions, targets),
        )
    elif loss_fn == "cross_entropy":
        return DenseEvaluationResult(
            rows=rows, loss=avg_loss, loss_fn=loss_fn, labels=list(labels or []),
            **_multiclass_metrics(predictions, targets, labels or []),
        )
    elif loss_fn == "binary_cross_entropy":
        return DenseEvaluationResult(
            rows=rows, loss=avg_loss, loss_fn=loss_fn, labels=list(labels or []),
            **_binary_metrics(predictions, targets, labels or []),
        )
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn!r}")


# ---------------------------------------------------------------------------
# Individual metric functions (also importable directly)
# ---------------------------------------------------------------------------

def compute_mae(predictions: list[float], targets: list[float]) -> float:
    n = len(predictions)
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / n


def compute_rmse(predictions: list[float], targets: list[float]) -> float:
    n = len(predictions)
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / n
    return math.sqrt(mse)


def compute_r2(predictions: list[float], targets: list[float]) -> float:
    n = len(targets)
    y_mean = sum(targets) / n
    ss_tot = sum((t - y_mean) ** 2 for t in targets)
    ss_res = sum((p - t) ** 2 for p, t in zip(predictions, targets))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def compute_accuracy(
    predictions: list[list[float]],
    targets: list[list[float]],
    threshold: float = 0.5,
) -> float:
    correct = 0
    for pred, tgt in zip(predictions, targets):
        if len(pred) == 1:
            pred_class = 1 if pred[0] >= threshold else 0
            tgt_class = 1 if tgt[0] >= threshold else 0
            correct += int(pred_class == tgt_class)
        else:
            correct += int(_argmax(pred) == _argmax(tgt))
    return correct / len(predictions) if predictions else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _regression_metrics(
    predictions: list[list[float]],
    targets: list[list[float]],
) -> dict[str, float]:
    flat_pred = [p[0] for p in predictions]
    flat_tgt = [t[0] for t in targets]
    return {
        "mae": compute_mae(flat_pred, flat_tgt),
        "rmse": compute_rmse(flat_pred, flat_tgt),
        "r2": compute_r2(flat_pred, flat_tgt),
    }


def _multiclass_metrics(
    predictions: list[list[float]],
    targets: list[list[float]],
    labels: list[str],
) -> dict[str, Any]:
    n = len(predictions)
    correct = 0
    cm: dict[str, dict[str, int]] = {}
    if labels:
        cm = {actual: {pred: 0 for pred in labels} for actual in labels}
    for pred, tgt in zip(predictions, targets):
        pred_idx = _argmax(pred)
        tgt_idx = _argmax(tgt)
        if pred_idx == tgt_idx:
            correct += 1
        if labels and tgt_idx < len(labels) and pred_idx < len(labels):
            actual_label = labels[tgt_idx]
            pred_label = labels[pred_idx]
            cm.setdefault(actual_label, {pred_label: 0})
            cm[actual_label][pred_label] = cm[actual_label].get(pred_label, 0) + 1
    precision, recall, f1, macro_f1 = _precision_recall_f1(cm, labels)
    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }


def _binary_metrics(
    predictions: list[list[float]],
    targets: list[list[float]],
    labels: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    n = len(predictions)
    correct = 0
    pos_label = labels[1] if len(labels) >= 2 else "positive"
    neg_label = labels[0] if len(labels) >= 2 else "negative"
    bin_labels = [neg_label, pos_label]
    cm: dict[str, dict[str, int]] = {
        pos_label: {pos_label: 0, neg_label: 0},
        neg_label: {pos_label: 0, neg_label: 0},
    }
    for pred, tgt in zip(predictions, targets):
        pred_class = pos_label if pred[0] >= threshold else neg_label
        tgt_class = pos_label if tgt[0] >= threshold else neg_label
        cm[tgt_class][pred_class] = cm[tgt_class].get(pred_class, 0) + 1
        if pred_class == tgt_class:
            correct += 1
    precision, recall, f1, macro_f1 = _precision_recall_f1(cm, bin_labels)
    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }


def _precision_recall_f1(
    cm: dict[str, dict[str, int]],
    labels: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    for label in labels:
        tp = cm.get(label, {}).get(label, 0)
        fp = sum(cm.get(actual, {}).get(label, 0) for actual in labels if actual != label)
        fn = sum(cm.get(label, {}).get(pred, 0) for pred in labels if pred != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision[label] = round(p, 6)
        recall[label] = round(r, 6)
        f1[label] = round(f, 6)
    macro_f1 = sum(f1.values()) / len(f1) if f1 else 0.0
    return precision, recall, f1, round(macro_f1, 6)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])
