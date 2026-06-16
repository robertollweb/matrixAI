# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from matrixai.compiler import BackendContractAnalyzer
from matrixai.ir import FunctionSpec, MatrixAIProgram, VectorSpec
from matrixai.parameters import (
    ParameterSet,
    build_initial_parameter_set,
    validate_parameter_set,
    write_parameter_set,
)
from matrixai.parser import parse_file
from matrixai.training.data import CSVDataAdapter, MatrixAIBatch, SupervisedExample, dataset_fingerprint
from matrixai.training.spec import EvaluationResult, TrainingRunResult, TrainingSpec
from matrixai.training.verifier import TrainingVerifier


TrainingExample = SupervisedExample
_PROBABILITY_BINARY_LABELS = ["negative", "positive"]


class SupervisedTrainer:
    def train(
        self,
        training: TrainingSpec,
        output_dir: str | Path,
        base_path: str | Path = ".",
        training_path: str | Path | None = None,
        epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TrainingRunResult:
        base = Path(base_path)
        output = Path(output_dir)
        report = TrainingVerifier().verify(training, base_path=base)
        if not report.ok:
            raise ValueError("; ".join(report.errors))
        model_path = Path(report.model_path)
        dataset_path = Path(report.dataset_path)
        program = parse_file(model_path)
        vector = self._training_vector(program, training)
        classifier = self._classifier(program, training)
        objective = self._objective(training, classifier)
        labels = self._labels(training)
        adapter = CSVDataAdapter(dataset_path, vector.name, vector.fields, training.dataset.target.name, labels)

        initial = build_initial_parameter_set(program, parameter_set_id=f"{output.name}_initial")
        weights = _copy_parameter_values(initial.parameters["W1"]["values"])
        bias = _copy_parameter_values(initial.parameters["b1"]["values"])

        examples = adapter.examples()
        train_examples, validation_examples = self._split_examples(examples, training)
        if not train_examples or not validation_examples:
            raise ValueError("Training and validation splits must both contain rows")

        epochs = training.run.epochs if training.run else 1
        batch_size = training.dataset.batch.size if training.dataset.batch else len(train_examples)
        shuffle_batches = bool(training.dataset.batch and training.dataset.batch.shuffle)
        learning_rate = training.optimizer.learning_rate
        best_weights = _copy_parameter_values(weights)
        best_bias = _copy_parameter_values(bias)
        best_epoch = 0
        best_validation_loss = float("inf")
        stale_epochs = 0
        epoch_trace: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            epoch_examples = list(train_examples)
            if shuffle_batches:
                random.Random((training.dataset.split.seed or 0) + epoch).shuffle(epoch_examples)
            for batch in _batches(epoch_examples, batch_size):
                gradients = self._gradients(
                    adapter.batch_from_examples(batch),
                    labels,
                    weights,
                    bias,
                    vector.name,
                    training.dataset.target.name,
                    objective,
                )
                weights, bias = _apply_gradients(weights, bias, gradients, learning_rate)

            train_metrics = self._metrics(train_examples, labels, weights, bias, objective)
            validation_metrics = self._metrics(validation_examples, labels, weights, bias, objective)
            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "accuracy": validation_metrics["accuracy"],
            }
            epoch_trace.append(entry)
            if epoch_callback is not None:
                epoch_callback(entry)
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                best_weights = _copy_parameter_values(weights)
                best_bias = _copy_parameter_values(bias)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if training.run and training.run.early_stop_patience is not None:
                if stale_epochs >= training.run.early_stop_patience:
                    break

        final_train = self._metrics(train_examples, labels, weights, bias, objective)
        final_validation = self._metrics(validation_examples, labels, weights, bias, objective)
        best_validation = self._metrics(validation_examples, labels, best_weights, best_bias, objective)
        final = self._parameter_set(
            program,
            initial,
            f"{output.name}_final",
            weights,
            bias,
            _parameter_metrics(final_validation),
        )
        best = self._parameter_set(
            program,
            initial,
            f"{output.name}_best",
            best_weights,
            best_bias,
            _parameter_metrics(best_validation),
        )
        validation = validate_parameter_set(program, best)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))

        output.mkdir(parents=True, exist_ok=True)
        artifacts = self._write_artifacts(
            output=output,
            program=program,
            model_path=model_path,
            training=training,
            training_path=Path(training_path) if training_path is not None else None,
            validation_report=report.to_dict(),
            initial=initial,
            best=best,
            final=final,
            epochs=epoch_trace,
            train_rows=len(train_examples),
            validation_rows=len(validation_examples),
            dataset_path=dataset_path,
            dataset_fingerprint=adapter.fingerprint(),
            split_trace=_split_trace(training, train_examples, validation_examples),
            classifier=classifier,
            best_epoch=best_epoch,
        )
        return TrainingRunResult(
            run_id=output.name,
            output_dir=str(output),
            best_epoch=best_epoch,
            best_validation_loss=best_validation["loss"],
            final_train_loss=final_train["loss"],
            final_validation_loss=final_validation["loss"],
            accuracy=best_validation["accuracy"],
            artifacts=artifacts,
        )

    def _training_vector(self, program: MatrixAIProgram, training: TrainingSpec) -> VectorSpec:
        for vector in program.vectors:
            if vector.name == training.dataset.input.vector:
                return vector
        raise ValueError(f"Training vector not found: {training.dataset.input.vector}")

    def _classifier(self, program: MatrixAIProgram, training: TrainingSpec) -> FunctionSpec:
        expected_kind = _expected_semantic_kind(training.loss.type)
        for function in program.functions:
            if function.output == training.loss.prediction and function.semantic.kind == expected_kind:
                return function
        raise ValueError(
            f"P4 supervised trainer requires {expected_kind} prediction for {training.loss.type}: "
            f"{training.loss.prediction}"
        )

    def _objective(self, training: TrainingSpec, classifier: FunctionSpec) -> str:
        if training.loss.type == "cross_entropy" and classifier.semantic.kind == "softmax_linear":
            return "softmax_cross_entropy"
        if training.loss.type == "binary_cross_entropy" and classifier.semantic.kind == "sigmoid_linear":
            return "sigmoid_binary_cross_entropy"
        if training.loss.type == "mse" and classifier.semantic.kind == "linear_regression":
            return "mse_regression"
        raise ValueError(
            f"Unsupported P4 objective: {classifier.semantic.kind} + {training.loss.type}"
        )

    def _labels(self, training: TrainingSpec) -> list[str]:
        if training.loss.type == "mse":
            return []
        target_type = training.dataset.target.type
        if training.loss.type == "binary_cross_entropy" and target_type.name == "Probability":
            return list(_PROBABILITY_BINARY_LABELS)
        labels = target_type.parameters.get("args", [])
        if not labels:
            raise ValueError("P4 supervised trainer requires Label[...] target values unless BCE target is Probability")
        values = [str(label) for label in labels]
        if training.loss.type == "binary_cross_entropy" and len(values) != 2:
            raise ValueError("binary_cross_entropy requires exactly two Label[...] target values")
        return values

    def _load_examples(self, path: Path, vector: VectorSpec, target: str) -> list[TrainingExample]:
        return CSVDataAdapter(path, vector.name, vector.fields, target).examples()

    def _split_examples(
        self, examples: list[TrainingExample], training: TrainingSpec
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        indices = list(range(len(examples)))
        split = training.dataset.split
        if split and split.seed is not None:
            random.Random(split.seed).shuffle(indices)
        train_ratio = split.train if split else 0.8
        train_count = max(1, min(len(examples) - 1, int(len(examples) * train_ratio)))
        train_indices = set(indices[:train_count])
        train = [example for index, example in enumerate(examples) if index in train_indices]
        validation = [example for index, example in enumerate(examples) if index not in train_indices]
        return train, validation

    def _gradients(
        self,
        batch: MatrixAIBatch,
        labels: list[str],
        weights: Any,
        bias: Any,
        input_vector: str,
        target: str,
        objective: str,
    ) -> dict[str, Any]:
        vectors = batch.inputs[input_vector]
        targets = batch.targets[target]
        if objective == "mse_regression":
            return _mse_regression_gradients(vectors, targets, weights, bias)
        if objective == "sigmoid_binary_cross_entropy":
            return _sigmoid_binary_gradients(vectors, targets, labels, weights, bias)

        weight_grad = [[0.0 for _ in row] for row in weights]
        bias_grad = [0.0 for _ in bias]
        scale = 1.0 / len(vectors)
        for vector, label_value in zip(vectors, targets):
            probabilities = _softmax(_logits(vector, weights, bias, labels))
            for label_index, label in enumerate(labels):
                target_value = 1.0 if label_value == label else 0.0
                delta = (probabilities[label] - target_value) * scale
                for column_index, value in enumerate(vector):
                    weight_grad[label_index][column_index] += delta * value
                bias_grad[label_index] += delta
        return {"weights": weight_grad, "bias": bias_grad}

    def _metrics(
        self,
        examples: list[TrainingExample],
        labels: list[str],
        weights: Any,
        bias: Any,
        objective: str,
    ) -> dict[str, Any]:
        if objective == "mse_regression":
            return _mse_regression_metrics(examples, weights, bias)
        if objective == "sigmoid_binary_cross_entropy":
            return _sigmoid_binary_metrics(examples, labels, weights, bias)

        loss = 0.0
        correct = 0
        confusion_matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
        for example in examples:
            probabilities = _softmax(_logits(example.vector, weights, bias, labels))
            loss -= math.log(max(probabilities[example.label], 1e-12))
            predicted = max(probabilities, key=probabilities.get)
            if example.label not in confusion_matrix:
                confusion_matrix[example.label] = {label: 0 for label in labels}
            confusion_matrix[example.label][predicted] += 1
            if predicted == example.label:
                correct += 1
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

    def _parameter_set(
        self,
        program: MatrixAIProgram,
        initial: ParameterSet,
        parameter_set_id: str,
        weights: Any,
        bias: Any,
        metrics: dict[str, Any],
    ) -> ParameterSet:
        data = initial.to_dict()
        data["parameter_set_id"] = parameter_set_id
        data["source"] = "trained"
        data["metrics"] = metrics
        data["parameters"]["W1"]["values"] = _rounded_values(weights)
        data["parameters"]["b1"]["values"] = _rounded_values(bias)
        parameter_set = ParameterSet.from_dict(data)
        validation = validate_parameter_set(program, parameter_set)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))
        return parameter_set

    def _write_artifacts(
        self,
        output: Path,
        program: MatrixAIProgram,
        model_path: Path,
        training: TrainingSpec,
        training_path: Path | None,
        validation_report: dict[str, Any],
        initial: ParameterSet,
        best: ParameterSet,
        final: ParameterSet,
        epochs: list[dict[str, Any]],
        train_rows: int,
        validation_rows: int,
        dataset_path: Path,
        dataset_fingerprint: str,
        split_trace: dict[str, Any],
        classifier: FunctionSpec,
        best_epoch: int,
        backend_target: str = "differentiable_python",
        backend_metadata: dict[str, Any] | None = None,
        backend_runtime: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        model_snapshot = output / "model_snapshot.mxai"
        train_config = output / "train_config.mxtrain"
        shutil.copyfile(model_path, model_snapshot)
        if training_path is not None:
            shutil.copyfile(training_path, train_config)
        else:
            train_config.write_text(json.dumps(training.to_dict(), indent=2), encoding="utf-8")

        write_parameter_set(output / "params.initial.json", initial)
        write_parameter_set(output / "params.best.json", best)
        write_parameter_set(output / "params.final.json", final)
        task_kind = "regression" if classifier.semantic.kind == "linear_regression" else "classification"
        metrics_out: dict[str, Any] = {
            "best_epoch": best_epoch,
            "best_validation_loss": best.metrics.get("validation_loss"),
            "accuracy": best.metrics.get("accuracy"),
            "macro_f1": best.metrics.get("macro_f1"),
            "epochs": epochs,
        }
        if "mae" in best.metrics:
            metrics_out["mae"] = best.metrics.get("mae")
            metrics_out["rmse"] = best.metrics.get("rmse")
            metrics_out["r2"] = best.metrics.get("r2")
        _write_json(output / "metrics.json", metrics_out)
        _write_json(output / "validation_report.json", validation_report)
        backend_report = BackendContractAnalyzer(target=backend_target).analyze(program).to_dict()
        trace = {
            "run_id": output.name,
            "model": str(model_path),
            "model_hash": best.model_hash,
            "parameter_schema_hash": best.parameter_schema_hash,
            "task_kind": task_kind,
            "dataset": {
                "name": training.dataset.name,
                "source": str(dataset_path),
                "fingerprint": dataset_fingerprint,
                "rows_train": train_rows,
                "rows_validation": validation_rows,
                "split": split_trace,
            },
            "parameters_updated": list(training.optimizer.update),
            "prediction": classifier.output,
            "loss": training.loss.type,
            "optimizer": training.optimizer.type,
            "epochs": epochs,
            "selected_parameter_set": "params.best.json",
            "backend_report": backend_report,
        }
        if backend_metadata is not None:
            trace["backend"] = backend_metadata
        if backend_runtime is not None:
            trace["backend_runtime"] = backend_runtime
        _write_json(output / "training_trace.json", trace)
        manifest = {
            "run_id": output.name,
            "model_snapshot": "model_snapshot.mxai",
            "train_config": "train_config.mxtrain",
            "params_initial": "params.initial.json",
            "params_best": "params.best.json",
            "params_final": "params.final.json",
            "metrics": "metrics.json",
            "training_trace": "training_trace.json",
            "validation_report": "validation_report.json",
        }
        _write_json(output / "run_manifest.json", manifest)
        return {key: str(output / value) for key, value in manifest.items() if key != "run_id"}


class SupervisedEvaluator:
    def evaluate(
        self,
        training: TrainingSpec,
        parameter_set: ParameterSet,
        data_path: str | Path | None = None,
        base_path: str | Path = ".",
    ) -> EvaluationResult:
        base = Path(base_path)
        model_path = _resolve_path(training.model, base)
        if model_path is None:
            raise ValueError(f"MODEL not found: {training.model}")
        dataset_path = _resolve_path(str(data_path) if data_path is not None else training.dataset.source, base)
        if dataset_path is None:
            raise ValueError(f"DATASET source not found: {data_path or training.dataset.source}")

        program = parse_file(model_path)
        validation = validate_parameter_set(program, parameter_set)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))
        trainer = SupervisedTrainer()
        vector = trainer._training_vector(program, training)
        classifier = trainer._classifier(program, training)
        objective = trainer._objective(training, classifier)
        labels = trainer._labels(training)
        adapter = CSVDataAdapter(dataset_path, vector.name, vector.fields, training.dataset.target.name, labels)
        examples = adapter.examples()
        if not examples:
            raise ValueError("Evaluation dataset must contain at least one row")
        weights = _copy_parameter_values(parameter_set.parameters["W1"]["values"])
        bias = _copy_parameter_values(parameter_set.parameters["b1"]["values"])
        metrics = trainer._metrics(examples, labels, weights, bias, objective)
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
            accuracy=metrics.get("accuracy", 0.0),
            labels=labels,
            confusion_matrix=metrics.get("confusion_matrix", {}),
            per_label=metrics.get("per_label", {}),
            macro_precision=metrics.get("macro_precision", 0.0),
            macro_recall=metrics.get("macro_recall", 0.0),
            macro_f1=metrics.get("macro_f1", 0.0),
            mae=metrics.get("mae", 0.0),
            rmse=metrics.get("rmse", 0.0),
            r2=metrics.get("r2", 0.0),
        )


def _logits(vector: list[float], weights: list[list[float]], bias: list[float], labels: list[str]) -> dict[str, float]:
    return {
        label: sum(value * weight for value, weight in zip(vector, weights[index])) + bias[index]
        for index, label in enumerate(labels)
    }


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    max_logit = max(logits.values())
    exps = {label: math.exp(value - max_logit) for label, value in logits.items()}
    total = sum(exps.values())
    return {label: value / total for label, value in exps.items()}


def _expected_semantic_kind(loss_type: str) -> str:
    if loss_type == "cross_entropy":
        return "softmax_linear"
    if loss_type == "binary_cross_entropy":
        return "sigmoid_linear"
    if loss_type == "mse":
        return "linear_regression"
    raise ValueError(f"LOSS type not supported by P4 supervised trainer: {loss_type}")


def _mse_regression_gradients(
    vectors: list[list[float]],
    targets: list[str],
    weights: Any,
    bias: Any,
) -> dict[str, Any]:
    weight_values = _copy_vector(weights)
    weight_grad = [0.0 for _ in weight_values]
    bias_grad = 0.0
    scale = 1.0 / len(vectors)
    for vector, target_str in zip(vectors, targets):
        y_hat = _dot(vector, weight_values) + _bias_value(bias)
        y = float(target_str)
        delta = 2.0 * (y_hat - y) * scale
        for i, xi in enumerate(vector):
            weight_grad[i] += delta * xi
        bias_grad += delta
    return {"weights": weight_grad, "bias": bias_grad}


def _mse_regression_metrics(
    examples: list[TrainingExample],
    weights: Any,
    bias: Any,
) -> dict[str, Any]:
    weight_values = _copy_vector(weights)
    y_values: list[float] = []
    y_hat_values: list[float] = []
    for example in examples:
        y_hat = _dot(example.vector, weight_values) + _bias_value(bias)
        y = float(example.label)
        y_values.append(y)
        y_hat_values.append(y_hat)
    n = len(examples)
    squared_errors = [(yh - y) ** 2 for yh, y in zip(y_hat_values, y_values)]
    mse = sum(squared_errors) / n
    mae = sum(abs(yh - y) for yh, y in zip(y_hat_values, y_values)) / n
    rmse = math.sqrt(mse)
    y_mean = sum(y_values) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    ss_res = sum(squared_errors)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {
        "loss": mse,
        "accuracy": 0.0,
        "confusion_matrix": {},
        "per_label": {},
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def _sigmoid_binary_gradients(
    vectors: list[list[float]],
    targets: list[str],
    labels: list[str],
    weights: Any,
    bias: Any,
) -> dict[str, Any]:
    weight_values = _copy_vector(weights)
    weight_grad = [0.0 for _ in weight_values]
    bias_grad = 0.0
    scale = 1.0 / len(vectors)
    for vector, label_value in zip(vectors, targets):
        probability = _sigmoid(_dot(vector, weight_values) + _bias_value(bias))
        target_value = _binary_target(label_value, labels)
        delta = (probability - target_value) * scale
        for column_index, value in enumerate(vector):
            weight_grad[column_index] += delta * value
        bias_grad += delta
    return {"weights": weight_grad, "bias": bias_grad}


def _sigmoid_binary_metrics(
    examples: list[TrainingExample],
    labels: list[str],
    weights: Any,
    bias: Any,
) -> dict[str, Any]:
    weight_values = _copy_vector(weights)
    negative_label, positive_label = labels
    loss = 0.0
    correct = 0
    confusion_matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for example in examples:
        probability = _sigmoid(_dot(example.vector, weight_values) + _bias_value(bias))
        target_value = _binary_target(example.label, labels)
        clipped = _clip_probability(probability)
        loss -= target_value * math.log(clipped) + (1.0 - target_value) * math.log(1.0 - clipped)
        predicted = positive_label if probability >= 0.5 else negative_label
        actual = _binary_actual_label(target_value, labels)
        confusion_matrix[actual][predicted] += 1
        if predicted == actual:
            correct += 1
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


def _binary_target(label: str, labels: list[str]) -> float:
    if label == labels[0]:
        return 0.0
    if label == labels[1]:
        return 1.0
    if labels == _PROBABILITY_BINARY_LABELS:
        try:
            value = float(label)
        except ValueError:
            value = -1.0
        if 0.0 <= value <= 1.0:
            return value
    raise ValueError(f"binary_cross_entropy target {label!r} must be one of {labels}")


def _binary_actual_label(target_value: float, labels: list[str]) -> str:
    return labels[1] if target_value >= 0.5 else labels[0]


def _apply_gradients(weights: Any, bias: Any, gradients: dict[str, Any], learning_rate: float) -> tuple[Any, Any]:
    weight_grad = gradients["weights"]
    bias_grad = gradients["bias"]
    if _is_matrix(weights):
        next_weights = [
            [float(value) - learning_rate * float(weight_grad[row_index][column_index])
             for column_index, value in enumerate(row)]
            for row_index, row in enumerate(weights)
        ]
    elif isinstance(weights, list):
        next_weights = [
            float(value) - learning_rate * float(weight_grad[index])
            for index, value in enumerate(weights)
        ]
    else:
        next_weights = float(weights) - learning_rate * float(weight_grad)

    if isinstance(bias, list):
        next_bias = [float(value) - learning_rate * float(bias_grad[index]) for index, value in enumerate(bias)]
    else:
        next_bias = float(bias) - learning_rate * float(bias_grad)
    return next_weights, next_bias


def _is_matrix(values: Any) -> bool:
    return isinstance(values, list) and bool(values) and isinstance(values[0], list)


def _dot(vector: list[float], weights: list[float]) -> float:
    return sum(float(value) * float(weight) for value, weight in zip(vector, weights))


def _bias_value(bias: Any) -> float:
    if isinstance(bias, list):
        return float(bias[0] if bias else 0.0)
    return float(bias)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-12, max(1e-12, value))


def _batches(examples: list[TrainingExample], size: int) -> list[list[TrainingExample]]:
    return [examples[index:index + size] for index in range(0, len(examples), max(1, size))]


def _copy_matrix(values: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in values]


def _copy_vector(values: Any) -> list[float]:
    if isinstance(values, list):
        return [float(value) for value in values]
    return [float(values)]


def _copy_parameter_values(values: Any) -> Any:
    if isinstance(values, list):
        return [_copy_parameter_values(value) for value in values]
    return float(values)


def _rounded_matrix(values: list[list[float]]) -> list[list[float]]:
    return [[round(value, 10) for value in row] for row in values]


def _rounded_vector(values: list[float]) -> list[float]:
    return [round(value, 10) for value in values]


def _rounded_values(values: Any) -> Any:
    if isinstance(values, list):
        return [_rounded_values(value) for value in values]
    return round(float(values), 10)


def _classification_metrics(
    confusion_matrix: dict[str, dict[str, int]], labels: list[str]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = confusion_matrix.get(label, {}).get(label, 0)
        false_positive = sum(
            confusion_matrix.get(actual, {}).get(label, 0) for actual in labels if actual != label
        )
        false_negative = sum(
            confusion_matrix.get(label, {}).get(predicted, 0) for predicted in labels if predicted != label
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion_matrix.get(label, {}).values()),
        }
    return metrics


def _macro_average(per_label: dict[str, dict[str, float]], metric: str) -> float:
    if not per_label:
        return 0.0
    return sum(values[metric] for values in per_label.values()) / len(per_label)


def _parameter_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "validation_loss": metrics["loss"],
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_precision": metrics.get("macro_precision", 0.0),
        "macro_recall": metrics.get("macro_recall", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
    }
    if "mae" in metrics:
        result["mae"] = metrics["mae"]
        result["rmse"] = metrics["rmse"]
        result["r2"] = metrics["r2"]
    return result


def _split_trace(
    training: TrainingSpec,
    train_examples: list[TrainingExample],
    validation_examples: list[TrainingExample],
) -> dict[str, Any]:
    split = training.dataset.split
    config: dict[str, Any] = {
        "train": split.train if split is not None else 0.8,
        "validation": split.validation if split is not None else 0.2,
    }
    if split is not None and split.seed is not None:
        config["seed"] = split.seed
    payload = {
        "config": config,
        "train": _split_partition_trace(train_examples),
        "validation": _split_partition_trace(validation_examples),
    }
    payload["fingerprint"] = _stable_fingerprint(
        "split",
        {
            "config": config,
            "train": payload["train"]["row_hashes"],
            "validation": payload["validation"]["row_hashes"],
        },
    )
    return payload


def _split_partition_trace(examples: list[TrainingExample]) -> dict[str, Any]:
    row_hashes = [example.row_hash for example in examples]
    return {
        "rows": len(examples),
        "row_indices": [example.row_index for example in examples],
        "row_hashes": row_hashes,
        "fingerprint": _stable_fingerprint("split_part", row_hashes),
    }


def _stable_fingerprint(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(value: str, base: Path) -> Path | None:
    direct = Path(value)
    candidates = [direct]
    if not direct.is_absolute():
        candidates.append(base / value)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Hierarchical UPDATE pattern matching
# ---------------------------------------------------------------------------

def match_update_patterns(patterns: list[str], param_keys: list[str]) -> list[str]:
    """Return param keys matching any UPDATE pattern.

    Patterns:
      - "*"          → every key
      - "exact.path" → only that key
      - "prefix.*"   → any key equal to prefix or starting with prefix + "."
    """
    matched: list[str] = []
    for key in param_keys:
        for pattern in patterns:
            if pattern == "*" or pattern == key:
                matched.append(key)
                break
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if key == prefix or key.startswith(prefix + "."):
                    matched.append(key)
                    break
    return matched


# ---------------------------------------------------------------------------
# Nested-list flatten / unflatten for numerical gradient computation
# ---------------------------------------------------------------------------

def _flatten_nested(val: Any) -> tuple[list[float], list[int]]:
    """Return (flat_values, shape) for a nested list of floats."""
    if isinstance(val, (int, float)):
        return [float(val)], []
    if not isinstance(val, list) or not val:
        return [], [0] if isinstance(val, list) else []
    if not isinstance(val[0], list):
        return [float(v) for v in val], [len(val)]
    sub_flat: list[float] = []
    sub_shape: list[int] = []
    for row in val:
        rf, rs = _flatten_nested(row)
        sub_flat.extend(rf)
        if not sub_shape:
            sub_shape = rs
    return sub_flat, [len(val)] + (sub_shape or [])


def _unflatten_nested(flat: list[float], shape: list[int]) -> Any:
    """Reconstruct a nested list from flat values and shape."""
    if not shape:
        return flat[0] if flat else 0.0
    if len(shape) == 1:
        return list(flat[: shape[0]])
    row_size = 1
    for s in shape[1:]:
        row_size *= s
    return [
        _unflatten_nested(flat[i * row_size: (i + 1) * row_size], shape[1:])
        for i in range(shape[0])
    ]


# ---------------------------------------------------------------------------
# Generic numerical gradient trainer for layer-based models (P11+)
# ---------------------------------------------------------------------------

def _compile_and_exec(source: str) -> dict[str, Any]:
    """Execute compiled module source and return its namespace."""
    ns: dict[str, Any] = {}
    exec(compile(source, "<matrixai_compiled>", "exec"), ns)  # noqa: S102
    return ns


def _build_input_data(example: Any, vector_name: str = "", vector_fields: list[str] | None = None) -> dict[str, Any]:
    """Build the input_data dict expected by run(). Handles both dict-vector and list-vector examples."""
    vname = getattr(example, "vector_name", None) or vector_name
    vec = example.vector
    if isinstance(vec, list) and vector_fields is not None:
        vec = {f: v for f, v in zip(vector_fields, vec)}
    return {vname: vec}


def _softmax_list(values: list[float]) -> list[float]:
    max_v = max(values) if values else 0.0
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _generic_cross_entropy_loss(
    state: dict[str, Any],
    target_label: str,
    prediction_key: str,
    labels: list[str] | None = None,
) -> float:
    """Cross-entropy loss from compiled module state.

    When labels is provided and pred is a list[float], applies softmax then uses
    labels.index(target_label) to pick the right probability (layer_call output).
    Falls back to integer-index lookup for backward compat when labels is None.
    """
    if labels is not None:
        probabilities = _generic_prediction_probabilities(state, prediction_key, labels)
        return -math.log(max(probabilities.get(target_label, 1e-12), 1e-12))

    pred = state.get(prediction_key)
    if isinstance(pred, dict):
        prob = pred.get(target_label, 1e-12)
    elif isinstance(pred, list):
        try:
            idx = int(target_label)
            prob = pred[idx] if idx < len(pred) else 1e-12
        except (ValueError, TypeError):
            prob = 1e-12
    else:
        try:
            prob = float(pred)
        except (TypeError, ValueError):
            prob = 1e-12
    return -math.log(max(float(prob), 1e-12))


def _generic_prediction_probabilities(
    state: dict[str, Any],
    prediction_key: str,
    labels: list[str],
) -> dict[str, float]:
    """Return label probabilities for generic layer outputs.

    P11 layer_call classifiers expose raw logits as list[float], while older
    generic tests may still expose dict probabilities or scalar binary scores.
    """
    pred = state.get(prediction_key)
    if isinstance(pred, dict):
        values: dict[str, float] = {}
        for label in labels:
            try:
                values[label] = float(pred.get(label, 0.0))
            except (TypeError, ValueError):
                values[label] = 0.0
        total = sum(values.values())
        if total > 0.0 and all(value >= 0.0 for value in values.values()):
            return {label: values[label] / total for label in labels}
        logits = [values[label] for label in labels]
        probs = _softmax_list(logits)
        return {label: probs[index] if index < len(probs) else 0.0 for index, label in enumerate(labels)}

    if isinstance(pred, list):
        logits = []
        for value in pred:
            try:
                logits.append(float(value))
            except (TypeError, ValueError):
                logits.append(0.0)
        probs = _softmax_list(logits)
        return {label: probs[index] if index < len(probs) else 0.0 for index, label in enumerate(labels)}

    try:
        positive = _clip_probability(float(pred))
    except (TypeError, ValueError):
        positive = 1e-12
    if len(labels) >= 2:
        return {labels[0]: 1.0 - positive, labels[1]: positive}
    if labels:
        return {labels[0]: positive}
    return {}


def _generic_metrics(
    run_fn: Any,
    examples: list[Any],
    prediction_key: str,
    params: dict[str, Any],
    labels: list[str] | None = None,
    vector_name: str = "",
    vector_fields: list[str] | None = None,
) -> dict[str, Any]:
    metric_labels = list(labels or sorted({str(example.label) for example in examples}))
    if not examples:
        per_label = _classification_metrics(
            {label: {predicted: 0 for predicted in metric_labels} for label in metric_labels},
            metric_labels,
        )
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "confusion_matrix": {label: {predicted: 0 for predicted in metric_labels} for label in metric_labels},
            "per_label": per_label,
            "macro_precision": _macro_average(per_label, "precision"),
            "macro_recall": _macro_average(per_label, "recall"),
            "macro_f1": _macro_average(per_label, "f1"),
        }

    loss = 0.0
    correct = 0
    confusion_matrix = {actual: {predicted: 0 for predicted in metric_labels} for actual in metric_labels}
    for example in examples:
        input_data = _build_input_data(example, vector_name, vector_fields)
        result = run_fn(input_data, params)
        probabilities = _generic_prediction_probabilities(result["state"], prediction_key, metric_labels)
        loss -= math.log(max(probabilities.get(example.label, 1e-12), 1e-12))
        predicted = max(probabilities, key=probabilities.get)
        if example.label not in confusion_matrix:
            confusion_matrix[example.label] = {label: 0 for label in metric_labels}
        confusion_matrix[example.label][predicted] = confusion_matrix[example.label].get(predicted, 0) + 1
        if predicted == example.label:
            correct += 1

    per_label = _classification_metrics(confusion_matrix, metric_labels)
    return {
        "loss": loss / len(examples),
        "accuracy": correct / len(examples),
        "confusion_matrix": confusion_matrix,
        "per_label": per_label,
        "macro_precision": _macro_average(per_label, "precision"),
        "macro_recall": _macro_average(per_label, "recall"),
        "macro_f1": _macro_average(per_label, "f1"),
    }


def _generic_parameter_set_from_runtime_params(
    program: MatrixAIProgram,
    initial: ParameterSet,
    parameter_set_id: str,
    runtime_params: dict[str, Any],
    metrics: dict[str, Any],
) -> ParameterSet:
    data = initial.to_dict()
    data["parameter_set_id"] = parameter_set_id
    data["source"] = "trained"
    data["metrics"] = dict(metrics)
    for key, parameter in data["parameters"].items():
        if key in runtime_params:
            parameter["values"] = _rounded_values(runtime_params[key])
    parameter_set = ParameterSet.from_dict(data)
    validation = validate_parameter_set(program, parameter_set)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    return parameter_set


def _numerical_gradient_for_param(
    run_fn: Any,
    runtime_params: dict[str, Any],
    param_key: str,
    input_data: dict[str, Any],
    target_label: str,
    prediction_key: str,
    eps: float = 1e-4,
    labels: list[str] | None = None,
) -> Any:
    """Finite-difference gradient for one parameter (all elements)."""
    val = runtime_params.get(param_key)
    if val is None:
        return None
    flat, shape = _flatten_nested(val)
    grad_flat: list[float] = []
    for i in range(len(flat)):
        flat_plus = flat.copy()
        flat_plus[i] += eps
        flat_minus = flat.copy()
        flat_minus[i] -= eps
        params_plus = {**runtime_params, param_key: _unflatten_nested(flat_plus, shape)}
        params_minus = {**runtime_params, param_key: _unflatten_nested(flat_minus, shape)}
        res_plus = run_fn(input_data, params_plus)
        res_minus = run_fn(input_data, params_minus)
        l_plus = _generic_cross_entropy_loss(res_plus["state"], target_label, prediction_key, labels=labels)
        l_minus = _generic_cross_entropy_loss(res_minus["state"], target_label, prediction_key, labels=labels)
        grad_flat.append((l_plus - l_minus) / (2 * eps))
    return _unflatten_nested(grad_flat, shape)


def _apply_sgd_to_param(val: Any, grad: Any, lr: float) -> Any:
    """Apply SGD update: val - lr * grad (element-wise for nested lists)."""
    if isinstance(val, list) and isinstance(grad, list):
        return [_apply_sgd_to_param(v, g, lr) for v, g in zip(val, grad)]
    return float(val) - lr * float(grad)


class GenericSupervisedTrainer:
    """Numerical-gradient trainer for layer-based programs (P11+).

    Uses finite differences on the compiled differentiable_python forward pass.
    Suitable for small models and toy datasets — O(2*P) forward passes per step.
    """

    _EPS: float = 1e-4

    def train(
        self,
        program: MatrixAIProgram,
        training: TrainingSpec,
        examples: list[Any],
        validation_examples: list[Any],
        prediction_key: str,
        target_key: str,
        update_patterns: list[str],
        epochs: int = 10,
        learning_rate: float = 0.01,
        epoch_callback: Callable[[dict[str, Any]], None] | None = None,
        labels: list[str] | None = None,
        vector_name: str = "",
        vector_fields: list[str] | None = None,
        parameter_set_prefix: str | None = None,
    ) -> dict[str, Any]:
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler

        source = DifferentiablePythonCompiler().compile(program)
        ns = _compile_and_exec(source)
        run_fn = ns["run"]

        initial_params = build_initial_parameter_set(program)
        runtime_params = initial_params.runtime_parameters()
        all_keys = list(initial_params.parameters.keys())
        trainable_keys = match_update_patterns(update_patterns, all_keys)
        if not trainable_keys:
            raise ValueError(
                f"No parameters matched UPDATE patterns {update_patterns}. "
                f"Available: {all_keys}"
            )

        best_params = deepcopy(runtime_params)
        best_loss = float("inf")
        epoch_trace: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for example in examples:
                input_data = _build_input_data(example, vector_name, vector_fields)
                result = run_fn(input_data, runtime_params)
                loss = _generic_cross_entropy_loss(result["state"], example.label, prediction_key, labels=labels)
                epoch_loss += loss
                for key in trainable_keys:
                    grad = _numerical_gradient_for_param(
                        run_fn, runtime_params, key, _build_input_data(example, vector_name, vector_fields),
                        example.label, prediction_key, eps=self._EPS, labels=labels,
                    )
                    if grad is not None:
                        runtime_params = {
                            **runtime_params,
                            key: _apply_sgd_to_param(runtime_params[key], grad, learning_rate),
                        }

            val_loss = self._eval_loss(run_fn, validation_examples, prediction_key, runtime_params, labels=labels, vector_name=vector_name, vector_fields=vector_fields)
            entry = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(len(examples), 1),
                "validation_loss": val_loss,
            }
            epoch_trace.append(entry)
            if epoch_callback:
                epoch_callback(entry)
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = deepcopy(runtime_params)

        final_metrics = _generic_metrics(
            run_fn,
            validation_examples,
            prediction_key,
            runtime_params,
            labels=labels,
            vector_name=vector_name,
            vector_fields=vector_fields,
        )
        best_metrics = _generic_metrics(
            run_fn,
            validation_examples,
            prediction_key,
            best_params,
            labels=labels,
            vector_name=vector_name,
            vector_fields=vector_fields,
        )
        prefix = parameter_set_prefix or f"{program.project}_generic"
        final_parameter_set = _generic_parameter_set_from_runtime_params(
            program,
            initial_params,
            f"{prefix}_final",
            runtime_params,
            final_metrics,
        )
        best_parameter_set = _generic_parameter_set_from_runtime_params(
            program,
            initial_params,
            f"{prefix}_best",
            best_params,
            best_metrics,
        )

        return {
            "final_params": runtime_params,
            "best_params": best_params,
            "final_parameter_set": final_parameter_set,
            "best_parameter_set": best_parameter_set,
            "trainable_keys": trainable_keys,
            "epoch_trace": epoch_trace,
        }

    def _eval_loss(
        self,
        run_fn: Any,
        examples: list[Any],
        prediction_key: str,
        params: dict[str, Any],
        labels: list[str] | None = None,
        vector_name: str = "",
        vector_fields: list[str] | None = None,
    ) -> float:
        if not examples:
            return 0.0
        total = 0.0
        for example in examples:
            input_data = _build_input_data(example, vector_name, vector_fields)
            result = run_fn(input_data, params)
            total += _generic_cross_entropy_loss(result["state"], example.label, prediction_key, labels=labels)
        return total / len(examples)


class GenericSupervisedEvaluator:
    """EvaluationResult producer for generic layer_call classifiers (P11+)."""

    def evaluate(
        self,
        training: TrainingSpec,
        parameter_set: ParameterSet,
        data_path: str | Path | None = None,
        base_path: str | Path = ".",
    ) -> EvaluationResult:
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler

        base = Path(base_path)
        model_path = _resolve_path(training.model, base)
        if model_path is None:
            raise ValueError(f"MODEL not found: {training.model}")
        dataset_path = _resolve_path(str(data_path) if data_path is not None else training.dataset.source, base)
        if dataset_path is None:
            raise ValueError(f"DATASET source not found: {data_path or training.dataset.source}")

        program = parse_file(model_path)
        validation = validate_parameter_set(program, parameter_set)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))

        labels = SupervisedTrainer()._labels(training)
        adapter = CSVDataAdapter(
            dataset_path,
            training.dataset.input.vector,
            training.dataset.input.columns,
            training.dataset.target.name,
            labels,
        )
        examples = adapter.examples()
        if not examples:
            raise ValueError("Evaluation dataset must contain at least one row")

        source = DifferentiablePythonCompiler().compile(program)
        ns = _compile_and_exec(source)
        run_fn = ns["run"]
        metrics = _generic_metrics(
            run_fn,
            examples,
            training.loss.prediction,
            parameter_set.runtime_parameters(),
            labels=labels,
            vector_name=training.dataset.input.vector,
            vector_fields=training.dataset.input.columns,
        )
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
            backend={
                "target": "differentiable_python",
                "evaluator": "GenericSupervisedEvaluator",
                "loss": training.loss.type,
                "prediction": training.loss.prediction,
            },
        )
