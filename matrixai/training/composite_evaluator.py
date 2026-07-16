# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P19 C8 — Evaluación de redes compuestas: dropout off, métricas heredadas de P18."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from matrixai.forward.composite_forward import composite_forward
from matrixai.parameters.store import ParameterSet
from matrixai.training.dense_evaluator import (
    DenseEvaluationResult,
    result_from_predictions,
)


def evaluate_composite_network(
    network: Any,
    parameter_set: ParameterSet,
    examples: list[tuple[dict[str, Any], list[float]]],
    loss_fn: str,
    labels: list[str] | None = None,
) -> DenseEvaluationResult:
    """Evaluate a composite network over a list of (input_dict, target) examples.

    Always runs with training=False so Dropout layers act as identity.

    Args:
        examples: list of (input_data_dict, target_vector) pairs.
            - regression: target_vector = [float]
            - binary_cross_entropy: target_vector = [float]  (0.0 or 1.0)
            - cross_entropy: target_vector = one-hot vector
        labels: class label strings for classification (same order as output units).
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    predictions: list[list[float]] = []
    targets: list[list[float]] = []
    for input_data, target in examples:
        predictions.append(composite_forward(network, parameter_set, input_data, training=False))
        targets.append(target)

    return result_from_predictions(predictions, targets, loss_fn, labels)


def composite_examples_from_csv(
    path: str | Path,
    input_columns: list[str],
    target_column: str,
    labels: list[str] | None = None,
    embedding_specs: dict[str, int] | None = None,
) -> list[tuple[dict[str, Any], list[float]]]:
    """Load a CSV file as composite (input_dict, target_vector) pairs.

    Integer and float columns are stored as float in input_dict; the embedding
    lookup in composite_forward casts to int internally.

    Args:
        input_columns: CSV column names to include in input_dict.
        target_column: CSV column used as label / regression target.
        labels: class label strings for classification. If None, regression mode:
                target_vector = [float(target_value)].
        embedding_specs: optional {column: vocab_size} mapping. When provided,
            each categorical column is validated to be an integer in [0, vocab).
    """
    is_regression = not labels
    label_index = {lbl: i for i, lbl in enumerate(labels or [])}
    emb_vocab: dict[str, int] = embedding_specs or {}

    examples: list[tuple[dict[str, Any], list[float]]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        for row_num, row in enumerate(csv.DictReader(fh), start=2):
            input_dict: dict[str, Any] = {}
            for col in input_columns:
                val = float(row[col])
                if col in emb_vocab:
                    idx = int(round(val))
                    vocab = emb_vocab[col]
                    if idx < 0 or idx >= vocab:
                        raise ValueError(
                            f"Row {row_num}, column '{col}': "
                            f"index {idx} out of range [0, {vocab})"
                        )
                    input_dict[col] = float(idx)
                else:
                    input_dict[col] = val
            target_str = str(row[target_column])
            if is_regression:
                target_vec = [float(target_str)]
            else:
                n = len(labels)  # type: ignore[arg-type]
                idx = label_index.get(target_str, 0)
                target_vec = [1.0 if i == idx else 0.0 for i in range(n)]
            examples.append((input_dict, target_vec))
    return examples
