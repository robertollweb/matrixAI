# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from matrixai.compiler import torch_backend_metadata
from matrixai.parameters import (
    validate_parameter_set,
    parameter_set_to_torch_tensors,
    TensorParameterBridgeError
)
from matrixai.parser import parse_file
from matrixai.training.data import CSVDataAdapter
from matrixai.training.spec import TrainingSpec, EvaluationResult
from matrixai.training.torch_trainer import (
    TorchSupervisedTrainer,
    _build_backend_runtime,
    _import_torch,
)
from matrixai.training.trainer import (
    _resolve_path,
    _binary_target,
    _classification_metrics,
    _macro_average,
    _clip_probability,
)


class TorchSupervisedEvaluator:
    def evaluate(
        self,
        training: TrainingSpec,
        parameter_set: Any,
        data_path: str | Path | None = None,
        base_path: str | Path = ".",
    ) -> EvaluationResult:
        base = Path(base_path)
        model_path = _resolve_path(training.model, base)
        if model_path is None:
            raise ValueError(f"MODEL not found: {training.model}")

        dataset_source = data_path if data_path is not None else training.dataset.source
        dataset_path = _resolve_path(str(dataset_source), base)
        if dataset_path is None:
            raise ValueError(f"DATASET source not found: {dataset_source}")

        program = parse_file(model_path)
        validation = validate_parameter_set(program, parameter_set)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))

        trainer = TorchSupervisedTrainer()
        vector = trainer._training_vector(program, training)
        classifier = trainer._classifier(program, training)
        objective = trainer._objective(training, classifier)
        if objective == "mse_regression":
            raise NotImplementedError(
                "Torch backend for mse regression is gated for P17.1; use backend='stdlib'"
            )
        labels = trainer._labels(training)
        adapter = CSVDataAdapter(dataset_path, vector.name, vector.fields, training.dataset.target.name, labels)
        examples = adapter.examples()
        if not examples:
            raise ValueError("Evaluation dataset must contain at least one row")

        torch = _import_torch()
        device = training.backend.device if training.backend else "cpu"

        try:
            tensors = parameter_set_to_torch_tensors(parameter_set)
        except TensorParameterBridgeError as exc:
            raise ValueError(str(exc)) from exc

        weights = tensors["W1"].detach().to(device)
        bias = tensors["b1"].detach().to(device)

        metrics = _device_aware_metrics(examples, labels, weights, bias, objective, torch, device)

        backend_metadata = torch_backend_metadata(device=device)
        backend_metadata["evaluator"] = "TorchSupervisedEvaluator"
        backend_metadata["loss"] = training.loss.type
        backend_runtime = _build_backend_runtime(device, training)

        return EvaluationResult(
            model=str(model_path),
            model_hash=parameter_set.model_hash,
            parameter_schema_hash=parameter_set.parameter_schema_hash,
            parameter_set_id=parameter_set.parameter_set_id,
            dataset=str(dataset_path),
            dataset_fingerprint=adapter.fingerprint(),
            dataset_schema=adapter.schema().to_dict(),
            rows=len(examples),
            loss=metrics["loss"],
            accuracy=metrics["accuracy"],
            labels=labels,
            confusion_matrix=metrics["confusion_matrix"],
            per_label=metrics["per_label"],
            macro_precision=metrics["macro_precision"],
            macro_recall=metrics["macro_recall"],
            macro_f1=metrics["macro_f1"],
            backend=backend_metadata,
            backend_runtime=backend_runtime,
        )


def _device_aware_metrics(
    examples: list,
    labels: list[str],
    weights: Any,
    bias: Any,
    objective: str,
    torch: Any,
    device: str,
) -> dict[str, Any]:
    """Compute evaluation metrics with the forward pass running on `device`."""
    vectors = [[float(v) for v in example.vector] for example in examples]
    input_tensor = torch.tensor(vectors, dtype=torch.float32, device=device)

    if objective == "softmax_cross_entropy":
        logits = input_tensor.matmul(weights.t()) + bias
        probs_cpu = torch.softmax(logits, dim=1).detach().cpu().tolist()

        loss = 0.0
        correct = 0
        confusion_matrix = {a: {p: 0 for p in labels} for a in labels}
        for i, example in enumerate(examples):
            row_probs = {labels[j]: probs_cpu[i][j] for j in range(len(labels))}
            loss -= math.log(max(row_probs.get(str(example.label), 1e-12), 1e-12))
            predicted = max(row_probs, key=row_probs.get)
            if example.label not in confusion_matrix:
                confusion_matrix[example.label] = {lbl: 0 for lbl in labels}
            confusion_matrix[example.label][predicted] += 1
            if predicted == str(example.label):
                correct += 1

    elif objective == "sigmoid_binary_cross_entropy":
        logits = input_tensor.matmul(weights.reshape(-1)) + bias.reshape(())
        probs_cpu = torch.sigmoid(logits).detach().cpu().tolist()

        negative_label, positive_label = labels
        loss = 0.0
        correct = 0
        confusion_matrix = {a: {p: 0 for p in labels} for a in labels}
        for i, example in enumerate(examples):
            probability = probs_cpu[i]
            target_value = _binary_target(str(example.label), labels)
            clipped = _clip_probability(probability)
            loss -= target_value * math.log(clipped) + (1.0 - target_value) * math.log(1.0 - clipped)
            predicted = positive_label if probability >= 0.5 else negative_label
            actual = str(example.label)
            if actual not in confusion_matrix:
                confusion_matrix[actual] = {lbl: 0 for lbl in labels}
            confusion_matrix[actual][predicted] += 1
            if predicted == actual:
                correct += 1

    else:
        raise ValueError(f"TorchSupervisedEvaluator: unsupported objective {objective!r}")

    per_label = _classification_metrics(confusion_matrix, labels)
    return {
        "loss": loss / len(examples),
        "accuracy": correct / len(examples),
        "confusion_matrix": confusion_matrix,
        "per_label": per_label,
        "macro_precision": _macro_average(per_label, "precision"),
        "macro_recall": _macro_average(per_label, "recall"),
        "macro_f1": _macro_average(per_label, "f1"),
    }
