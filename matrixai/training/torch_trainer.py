# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import random
from importlib import import_module
from pathlib import Path
from typing import Any

from matrixai.compiler import torch_backend_metadata
from matrixai.parameters import (
    TensorParameterBridgeError,
    build_initial_parameter_set,
    parameter_set_to_torch_tensors,
    validate_parameter_set,
)
from matrixai.parameters.tensor_bridge import torch_tensors_to_parameter_set
from matrixai.parser import parse_file
from matrixai.training.data import CSVDataAdapter, MatrixAIBatch
from matrixai.training.spec import TrainingRunResult, TrainingSpec
from matrixai.training.trainer import (
    SupervisedTrainer,
    _batches,
    _binary_target,
    _copy_parameter_values,
    _parameter_metrics,
    _rounded_values,
    _split_trace,
)
from matrixai.training.verifier import TrainingVerifier


class TorchSupervisedTrainer(SupervisedTrainer):
    def train(
        self,
        training: TrainingSpec,
        output_dir: str | Path,
        base_path: str | Path = ".",
        training_path: str | Path | None = None,
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
        if objective == "mse_regression":
            raise NotImplementedError(
                "Torch backend for mse regression is gated for P17.1; use backend='stdlib'"
            )
        labels = self._labels(training)
        adapter = CSVDataAdapter(dataset_path, vector.name, vector.fields, training.dataset.target.name, labels)

        torch = _import_torch()
        device = training.backend.device if training.backend else "cpu"
        initial = build_initial_parameter_set(program, parameter_set_id=f"{output.name}_initial")
        try:
            initial_tensors = parameter_set_to_torch_tensors(initial)
        except TensorParameterBridgeError as exc:
            raise ValueError(str(exc)) from exc
        weights = initial_tensors["W1"].detach().clone().to(device).requires_grad_(True)
        bias = initial_tensors["b1"].detach().clone().to(device).requires_grad_(True)
        optimizer = torch.optim.SGD([weights, bias], lr=training.optimizer.learning_rate)

        examples = adapter.examples()
        train_examples, validation_examples = self._split_examples(examples, training)
        if not train_examples or not validation_examples:
            raise ValueError("Training and validation splits must both contain rows")

        epochs = training.run.epochs if training.run else 1
        batch_size = training.dataset.batch.size if training.dataset.batch else len(train_examples)
        shuffle_batches = bool(training.dataset.batch and training.dataset.batch.shuffle)
        best_weights = weights.detach().clone()
        best_bias = bias.detach().clone()
        best_epoch = 0
        best_validation_loss = float("inf")
        stale_epochs = 0
        epoch_trace: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            epoch_examples = list(train_examples)
            if shuffle_batches:
                random.Random((training.dataset.split.seed or 0) + epoch).shuffle(epoch_examples)
            for batch in _batches(epoch_examples, batch_size):
                matrix_batch = adapter.batch_from_examples(batch)
                input_tensor, target_tensor = _batch_tensors(
                    matrix_batch,
                    vector.name,
                    training.dataset.target.name,
                    objective,
                    labels,
                    torch,
                    device,
                )
                optimizer.zero_grad()
                loss = _torch_loss(input_tensor, target_tensor, weights, bias, objective, torch)
                loss.backward()
                optimizer.step()

            train_metrics = self._metrics(
                train_examples,
                labels,
                _tensor_values(weights),
                _tensor_values(bias),
                objective,
            )
            validation_metrics = self._metrics(
                validation_examples,
                labels,
                _tensor_values(weights),
                _tensor_values(bias),
                objective,
            )
            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "accuracy": validation_metrics["accuracy"],
                "backend": "torch",
            }
            epoch_trace.append(entry)
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                best_weights = weights.detach().clone()
                best_bias = bias.detach().clone()
                stale_epochs = 0
            else:
                stale_epochs += 1
            if training.run and training.run.early_stop_patience is not None:
                if stale_epochs >= training.run.early_stop_patience:
                    break

        final_train = self._metrics(train_examples, labels, _tensor_values(weights), _tensor_values(bias), objective)
        final_validation = self._metrics(
            validation_examples,
            labels,
            _tensor_values(weights),
            _tensor_values(bias),
            objective,
        )
        best_validation = self._metrics(
            validation_examples,
            labels,
            _tensor_values(best_weights),
            _tensor_values(best_bias),
            objective,
        )
        final = _parameter_set_from_tensors(
            program,
            initial,
            f"{output.name}_final",
            weights,
            bias,
            {**_parameter_metrics(final_validation), "backend": "torch"},
        )
        best = _parameter_set_from_tensors(
            program,
            initial,
            f"{output.name}_best",
            best_weights,
            best_bias,
            {**_parameter_metrics(best_validation), "backend": "torch"},
        )
        validation = validate_parameter_set(program, best)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))

        output.mkdir(parents=True, exist_ok=True)
        backend_metadata = torch_backend_metadata(device=device)
        backend_metadata["trainer"] = "TorchSupervisedTrainer"
        backend_metadata["loss"] = training.loss.type
        backend_runtime = _build_backend_runtime(device, training)
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
            backend_target="torch",
            backend_metadata=backend_metadata,
            backend_runtime=backend_runtime,
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


def _build_backend_runtime(device: str, training: "TrainingSpec") -> dict[str, Any]:
    from matrixai.parameters.tensor_bridge import torch_device_info
    info = torch_device_info()
    runtime: dict[str, Any] = {
        "target": "torch",
        "device": device,
        "torch_version": info.get("torch_version"),
        "device_name": info.get("device_name"),
    }
    if info.get("cuda_available"):
        runtime["cuda_version"] = info.get("cuda_version")
    split = getattr(training.dataset, "split", None)
    runtime["seed"] = split.seed if split else None
    return runtime


def _import_torch():
    try:
        return import_module("torch")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"TorchSupervisedTrainer requires optional dependency PyTorch: {exc}") from exc


def _batch_tensors(
    batch: MatrixAIBatch,
    input_vector: str,
    target: str,
    objective: str,
    labels: list[str],
    torch,
    device: str = "cpu",
) -> tuple[Any, Any]:
    input_tensor = torch.tensor(batch.inputs[input_vector], dtype=torch.float32).to(device)
    targets = batch.targets[target]
    if objective == "softmax_cross_entropy":
        label_to_index = {label: index for index, label in enumerate(labels)}
        target_tensor = torch.tensor([label_to_index[str(value)] for value in targets], dtype=torch.long).to(device)
        return input_tensor, target_tensor
    target_values = [_binary_target(str(value), labels) for value in targets]
    return input_tensor, torch.tensor(target_values, dtype=torch.float32).to(device)


def _torch_loss(input_tensor, target_tensor, weights, bias, objective: str, torch):
    if objective == "softmax_cross_entropy":
        logits = input_tensor.matmul(weights.t()) + bias
        return torch.nn.functional.cross_entropy(logits, target_tensor)
    logits = input_tensor.matmul(weights.reshape(-1)) + bias.reshape(())
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target_tensor)


def _parameter_set_from_tensors(
    program,
    initial,
    parameter_set_id: str,
    weights,
    bias,
    metrics: dict[str, Any],
):
    parameter_set = torch_tensors_to_parameter_set(
        initial,
        {"W1": weights.detach().cpu(), "b1": bias.detach().cpu()},
        parameter_set_id=parameter_set_id,
        source="trained_torch",
        metrics=metrics,
    )
    data = parameter_set.to_dict()
    for parameter in data["parameters"].values():
        parameter["values"] = _rounded_values(parameter["values"])
    parameter_set = type(parameter_set).from_dict(data)
    validation = validate_parameter_set(program, parameter_set)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    return parameter_set


def _tensor_values(tensor) -> Any:
    return _copy_parameter_values(tensor.detach().cpu().tolist())
