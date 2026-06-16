# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matrixai.compiler import BackendContractAnalyzer
from matrixai.ir import MatrixAIProgram
from matrixai.parser import parse_file
from matrixai.training.differentiability import DifferentiabilityVerifier
from matrixai.training.spec import TrainingSpec
from matrixai.types import TypeSpec, semantic_kind_output_type, type_is_compatible


@dataclass(frozen=True)
class TrainingVerificationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_path: str = ""
    dataset_path: str = ""
    trainable_parameters: list[dict[str, Any]] = field(default_factory=list)
    differentiability: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "trainable_parameters": list(self.trainable_parameters),
            "differentiability": dict(self.differentiability),
        }

    def summary(self) -> str:
        status = "ok" if self.ok else "blocked"
        lines = [f"Training validation: {status}"]
        if self.model_path:
            lines.append(f"Model: {self.model_path}")
        if self.dataset_path:
            lines.append(f"Dataset: {self.dataset_path}")
        if self.trainable_parameters:
            lines.append("Trainable parameters:")
            for parameter in self.trainable_parameters:
                shape = parameter.get("shape", [])
                lines.append(f"- {parameter['function']}.{parameter['name']} shape={shape}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        for error in self.errors:
            lines.append(f"Error: {error}")
        if self.differentiability:
            prediction_node = self.differentiability.get("prediction_node")
            if prediction_node:
                lines.append(f"Differentiability prediction node: {prediction_node}")
        return "\n".join(lines)


class TrainingVerifier:
    def verify(self, training: TrainingSpec, base_path: str | Path = ".") -> TrainingVerificationResult:
        base = Path(base_path)
        errors: list[str] = []
        warnings: list[str] = []
        model_path = self._resolve_path(training.model, base)
        dataset_path = self._resolve_path(training.dataset.source, base)

        program: MatrixAIProgram | None = None
        if model_path is None:
            errors.append(f"MODEL not found: {training.model}")
        else:
            try:
                program = parse_file(model_path)
            except (OSError, ValueError) as exc:
                errors.append(f"MODEL invalid: {exc}")

        if training.dataset.source_kind != "csv":
            errors.append(f"DATASET source kind not supported by P4 MVP: {training.dataset.source_kind}")
        if dataset_path is None:
            errors.append(f"DATASET source not found: {training.dataset.source}")

        trainable_parameters: list[dict[str, Any]] = []
        differentiability: dict[str, Any] = {}
        if program is not None:
            errors.extend(self._verify_program_contract(training, program))
            backend_report = BackendContractAnalyzer().analyze(program)
            if not backend_report.ok:
                blocked = ", ".join(node.node for node in backend_report.unsupported_nodes)
                errors.append(f"MODEL is not portable to differentiable backend: {blocked}")
            trainable_parameters = [parameter.to_dict() for parameter in backend_report.trainable_parameters]
            errors.extend(self._verify_updates(training, trainable_parameters))
            differentiability_report = DifferentiabilityVerifier().verify(training, program, backend_report)
            differentiability = differentiability_report.to_dict()
            errors.extend(differentiability_report.errors)
            warnings.extend(differentiability_report.warnings)

        if program is not None and dataset_path is not None:
            errors.extend(self._verify_dataset(training, program, dataset_path))

        return TrainingVerificationResult(
            errors=errors,
            warnings=warnings,
            model_path=str(model_path) if model_path is not None else "",
            dataset_path=str(dataset_path) if dataset_path is not None else "",
            trainable_parameters=trainable_parameters,
            differentiability=differentiability,
        )

    def _resolve_path(self, value: str, base: Path) -> Path | None:
        direct = Path(value)
        candidates = [direct]
        if not direct.is_absolute():
            candidates.append(base / value)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _verify_program_contract(self, training: TrainingSpec, program: MatrixAIProgram) -> list[str]:
        errors: list[str] = []
        vector = next((item for item in program.vectors if item.name == training.dataset.input.vector), None)
        if vector is None:
            errors.append(f"INPUT vector not found in MODEL: {training.dataset.input.vector}")
        elif training.dataset.input.columns != vector.fields:
            errors.append(
                f"INPUT columns for {vector.name} must match VECTOR fields {vector.fields}, got {training.dataset.input.columns}"
            )

        prediction_function = self._prediction_function(program, training.loss.prediction)
        prediction_type = self._symbol_type(program, training.loss.prediction)
        if prediction_type is None:
            errors.append(f"LOSS prediction not found in MODEL: {training.loss.prediction}")
        if training.loss.target != training.dataset.target.name:
            errors.append(
                f"LOSS target {training.loss.target} does not match DATASET target {training.dataset.target.name}"
            )
        _is_regression = training.loss.type == "mse"
        _regression_target_names = {"Scalar", "Integer"}

        if training.loss.type == "cross_entropy":
            expected_prediction = TypeSpec("ProbabilityMap")
            expected_target = TypeSpec("Label")
            if prediction_type is not None and not type_is_compatible(prediction_type, expected_prediction):
                errors.append(
                    f"cross_entropy expects ProbabilityMap prediction, got {prediction_type.name}"
                )
            if training.dataset.target.type.name in _regression_target_names:
                errors.append(
                    f"cross_entropy expects Label target, got {training.dataset.target.type.name} — "
                    f"use loss type 'mse' for continuous targets"
                )
            elif not type_is_compatible(training.dataset.target.type, expected_target):
                errors.append(
                    f"cross_entropy expects Label target, got {training.dataset.target.type.name}"
                )
            _cross_entropy_kinds = {"softmax_linear", "layer_call"}
            if prediction_function is not None and prediction_function.semantic.kind not in _cross_entropy_kinds:
                errors.append("cross_entropy requires softmax_linear or layer_call prediction for supervised training")
        elif training.loss.type == "binary_cross_entropy":
            expected_prediction = TypeSpec("Probability")
            if prediction_type is not None and not type_is_compatible(prediction_type, expected_prediction):
                errors.append(
                    f"binary_cross_entropy expects Probability prediction, got {prediction_type.name}"
                )
            target_type = training.dataset.target.type
            if target_type.name in _regression_target_names:
                errors.append(
                    f"binary_cross_entropy expects Label[...] or Probability target, got {target_type.name} — "
                    f"use loss type 'mse' for continuous targets"
                )
            elif target_type.name == "Label":
                labels = target_type.parameters.get("args", [])
                if len(labels) != 2:
                    errors.append("binary_cross_entropy expects exactly two Label[...] target values")
            elif target_type.name != "Probability":
                errors.append(
                    f"binary_cross_entropy expects Label[...] or Probability target, got {target_type.name}"
                )
            _bce_kinds = {"sigmoid_linear", "layer_call"}
            if prediction_function is not None and prediction_function.semantic.kind not in _bce_kinds:
                errors.append("binary_cross_entropy requires sigmoid_linear or layer_call prediction for supervised training")
        elif training.loss.type == "mse":
            if training.dataset.target.type.name not in _regression_target_names:
                errors.append(
                    f"mse expects Scalar or Integer target, got {training.dataset.target.type.name} — "
                    f"use loss type 'cross_entropy' or 'binary_cross_entropy' for label targets"
                )
            _mse_kinds = {"linear_regression", "layer_call"}
            if prediction_function is not None and prediction_function.semantic.kind not in _mse_kinds:
                errors.append(
                    "mse requires linear_regression or layer_call prediction for supervised training"
                )
        else:
            errors.append(f"LOSS type not supported: {training.loss.type}")

        _classification_metric_types = {"accuracy"}
        _regression_metric_types = {"mae", "rmse", "r2"}
        for metric in training.metrics:
            if metric.prediction != training.loss.prediction:
                errors.append(f"METRIC {metric.name} prediction must match LOSS prediction")
            if metric.target != training.dataset.target.name:
                errors.append(f"METRIC {metric.name} target must match DATASET target")
            if _is_regression:
                if metric.type not in _regression_metric_types:
                    errors.append(
                        f"METRIC {metric.name} type {metric.type!r} not valid for mse loss — "
                        f"use one of: {sorted(_regression_metric_types)}"
                    )
            else:
                if metric.type in _regression_metric_types:
                    errors.append(
                        f"METRIC {metric.name} type {metric.type!r} requires mse loss"
                    )
                elif metric.type not in _classification_metric_types:
                    errors.append(f"METRIC type not supported: {metric.type}")

        if training.optimizer.type != "sgd":
            errors.append(f"OPTIMIZER type not supported by P4 MVP: {training.optimizer.type}")
        if training.optimizer.learning_rate <= 0:
            errors.append("OPTIMIZER LEARNING_RATE must be greater than 0")
        if training.run is not None and training.run.epochs <= 0:
            errors.append("RUN EPOCHS must be greater than 0")
        return errors

    def _symbol_type(self, program: MatrixAIProgram, symbol: str) -> TypeSpec | None:
        for function in program.functions:
            if function.output == symbol:
                if function.output_type is not None:
                    return function.output_type
                return semantic_kind_output_type(function.semantic.kind, function.semantic.parameters)
        for distribution in program.distributions:
            if distribution.variable == symbol or distribution.name == symbol:
                return TypeSpec(distribution.distribution_type)
        for network in getattr(program, "networks", []):
            if network.output == symbol or network.name == symbol:
                base_type = network.output_type_str.split("[")[0].strip()
                return TypeSpec(base_type)
        return None

    def _prediction_function(self, program: MatrixAIProgram, symbol: str):
        for function in program.functions:
            if function.output == symbol or function.name == symbol:
                return function
        return None

    def _verify_updates(self, training: TrainingSpec, trainable_parameters: list[dict[str, Any]]) -> list[str]:
        from matrixai.training.trainer import match_update_patterns
        errors: list[str] = []
        all_keys: list[str] = []
        seen: set[str] = set()
        for parameter in trainable_parameters:
            for key in (parameter["name"], f"{parameter['function']}.{parameter['name']}"):
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)
        for pattern in training.optimizer.update:
            if not match_update_patterns([pattern], all_keys):
                errors.append(f"UPDATE parameter is not trainable in MODEL: {pattern}")
        return errors

    def _verify_dataset(
        self, training: TrainingSpec, program: MatrixAIProgram, dataset_path: Path
    ) -> list[str]:
        errors: list[str] = []
        vector = next((item for item in program.vectors if item.name == training.dataset.input.vector), None)
        if vector is None:
            return errors
        try:
            with dataset_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            return [f"DATASET cannot be read: {exc}"]
        if not rows:
            return ["DATASET must contain at least one row"]
        fieldnames = set(rows[0].keys())
        required = set(training.dataset.input.columns + [training.dataset.target.name])
        missing = sorted(required - fieldnames)
        if missing:
            errors.append(f"DATASET missing columns: {', '.join(missing)}")
            return errors
        target_type = training.dataset.target.type
        labels = set(target_type.parameters.get("args", []))
        _is_continuous_target = target_type.name in {"Scalar", "Integer"}
        for row_index, row in enumerate(rows, start=2):
            for field in vector.fields:
                value = row.get(field, "")
                if value == "":
                    errors.append(f"DATASET row {row_index} field {field} is empty")
                    continue
                try:
                    number = float(value)
                except ValueError:
                    errors.append(f"DATASET row {row_index} field {field} must be numeric, got {value!r}")
                    continue
                type_spec = vector.field_types.get(field)
                if type_spec is not None and type_spec.range is not None and not type_spec.range.contains(number):
                    errors.append(
                        f"DATASET row {row_index} field {field} expects {type_spec.name} range "
                        f"[{type_spec.range.minimum}, {type_spec.range.maximum}], got {number}"
                    )
            target_value = row.get(training.dataset.target.name, "")
            if target_value == "":
                errors.append(f"DATASET row {row_index} target {training.dataset.target.name} is empty")
                continue
            if _is_continuous_target:
                try:
                    number = float(target_value)
                except ValueError:
                    errors.append(
                        f"DATASET row {row_index} target {training.dataset.target.name} must be numeric, "
                        f"got {target_value!r}"
                    )
                    continue
                if target_type.range is not None and not target_type.range.contains(number):
                    errors.append(
                        f"DATASET row {row_index} target {training.dataset.target.name} expects "
                        f"{target_type.name} range [{target_type.range.minimum}, {target_type.range.maximum}], "
                        f"got {number}"
                    )
            elif target_type.name == "Probability":
                try:
                    probability = float(target_value)
                except ValueError:
                    errors.append(
                        f"DATASET row {row_index} target {training.dataset.target.name} must be numeric, "
                        f"got {target_value!r}"
                    )
                    continue
                if target_type.range is not None and not target_type.range.contains(probability):
                    errors.append(
                        f"DATASET row {row_index} target {training.dataset.target.name} expects "
                        f"{target_type.name} range [{target_type.range.minimum}, {target_type.range.maximum}], "
                        f"got {probability}"
                    )
            elif labels and target_value not in labels:
                errors.append(
                    f"DATASET row {row_index} target {training.dataset.target.name} must be one of "
                    f"{sorted(labels)}, got {target_value!r}"
                )
        return errors