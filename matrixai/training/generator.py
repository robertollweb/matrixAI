# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from matrixai.compiler import BackendContractAnalyzer
from matrixai.ir import FunctionSpec, MatrixAIProgram, VectorSpec
from matrixai.parser import parse_file


@dataclass(frozen=True)
class TrainingGenerationResult:
    prompt: str
    model: str
    training_text: str
    dataset_template_text: str
    dataset_source: str
    dataset_name: str
    input_vector: str
    input_columns: list[str]
    target_name: str
    labels: list[str]
    prediction: str
    loss_type: str
    optimizer: str = "sgd"
    learning_rate: float = 0.05
    epochs: int = 20
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = ["Training generation: ok"]
        lines.append(f"Model: {self.model}")
        lines.append(f"Dataset source: {self.dataset_source}")
        lines.append(f"Input: {self.input_vector} columns={self.input_columns}")
        lines.append(f"Target: {self.target_name} labels={self.labels}")
        lines.append(f"Loss: {self.loss_type}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        return "\n".join(lines)


class TrainingPromptGenerator:
    _LABEL_RE = re.compile(
        r"\b(?:labels?|etiquetas|clases|categorias|categories)\s*(?::|=|son|are)?\s*(?P<labels>[^.;\n]+)",
        re.IGNORECASE,
    )
    _TARGET_RE = re.compile(r"\b(?:target|objetivo)\s*(?::|=)?\s*(?P<name>[A-Za-z_][\w]*)", re.IGNORECASE)
    _DATASET_RE = re.compile(r"\b(?:dataset|datos)\s*(?::|=)?\s*(?P<name>[A-Za-z_][\w]*)", re.IGNORECASE)

    def generate(
        self,
        prompt: str,
        model_path: str | Path,
        dataset_source: str | None = None,
        dataset_name: str | None = None,
        target_name: str | None = None,
        labels: list[str] | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        split_train: float = 0.8,
        split_seed: int = 42,
        model_reference: str | None = None,
        target_scalar_range: tuple[float, float] | None = None,
    ) -> TrainingGenerationResult:
        clean_prompt = " ".join(prompt.strip().split())
        if not clean_prompt:
            raise ValueError("TrainingPromptGenerator requires a non-empty prompt")

        model = model_reference or str(model_path)
        program = parse_file(model_path)
        vector, function = self._select_trainable_function(program)
        objective = self._objective(function)
        resolved_labels = list(labels or self._extract_labels(clean_prompt) or self._default_labels(objective, clean_prompt))
        expected_label_count = self._expected_label_count(program, function, objective)
        if expected_label_count is not None and len(resolved_labels) != expected_label_count:
            raise ValueError(
                f"{objective} generation requires {expected_label_count} labels for {function.name}, "
                f"got {len(resolved_labels)}"
            )
        if objective == "binary_cross_entropy" and len(resolved_labels) != 2:
            raise ValueError("binary_cross_entropy generation requires exactly two labels")
        if objective == "cross_entropy" and len(resolved_labels) < 2:
            raise ValueError("cross_entropy generation requires at least two labels")

        if objective == "mse":
            resolved_target = target_name or function.output
        else:
            resolved_target = target_name or self._extract_name(clean_prompt, self._TARGET_RE) or self._default_target(objective)
        resolved_dataset = dataset_name or self._extract_name(clean_prompt, self._DATASET_RE) or f"{program.project}TrainingSet"
        resolved_source = dataset_source or f"{Path(model).with_suffix('').name}.train.csv"
        if objective == "mse":
            resolved_epochs = epochs or 50
            resolved_learning_rate = learning_rate if learning_rate is not None else 0.001
            resolved_batch_size = batch_size or 4
        elif objective == "binary_cross_entropy":
            resolved_epochs = epochs or 30
            resolved_learning_rate = learning_rate if learning_rate is not None else 0.8
            resolved_batch_size = batch_size or 2
        else:
            resolved_epochs = epochs or 20
            resolved_learning_rate = learning_rate if learning_rate is not None else 0.05
            resolved_batch_size = batch_size or min(4, max(1, len(resolved_labels)))
        validation = round(1.0 - split_train, 10)

        parameter_names = self._parameter_names(program, function)
        training_text = self._training_text(
            model=model,
            dataset_name=resolved_dataset,
            dataset_source=resolved_source,
            vector=vector,
            target_name=resolved_target,
            labels=resolved_labels,
            loss_type=objective,
            prediction=function.output,
            learning_rate=resolved_learning_rate,
            update=parameter_names,
            batch_size=resolved_batch_size,
            split_train=split_train,
            split_validation=validation,
            split_seed=split_seed,
            epochs=resolved_epochs,
            target_scalar_range=target_scalar_range,
        )
        dataset_template = self._dataset_template(vector, resolved_target, resolved_labels, target_scalar_range=target_scalar_range)
        assumptions = [
            f"Selected trainable function {function.name} ({function.semantic.kind}) from P3 backend contract",
            f"Selected loss {objective} from function semantic kind",
            "Generated dataset rows are placeholders for schema validation, not production data",
        ]
        if objective == "mse":
            assumptions.append(
                "linear_regression pipeline: loss=mse, target=Scalar, metrics=mae/rmse/r2 — "
                "tune LEARNING_RATE to dataset scale (default 0.001 may diverge for large targets)"
            )
        warnings: list[str] = []
        if objective == "binary_cross_entropy":
            warnings.append(f"binary_cross_entropy treats label {resolved_labels[1]!r} as the positive class")
        return TrainingGenerationResult(
            prompt=clean_prompt,
            model=model,
            training_text=training_text,
            dataset_template_text=dataset_template,
            dataset_source=resolved_source,
            dataset_name=resolved_dataset,
            input_vector=vector.name,
            input_columns=list(vector.fields),
            target_name=resolved_target,
            labels=resolved_labels,
            prediction=function.output,
            loss_type=objective,
            learning_rate=resolved_learning_rate,
            epochs=resolved_epochs,
            assumptions=assumptions,
            warnings=warnings,
        )

    def _select_trainable_function(self, program: MatrixAIProgram) -> tuple[VectorSpec, FunctionSpec]:
        backend_report = BackendContractAnalyzer().analyze(program)
        if not backend_report.ok:
            blocked = ", ".join([node.node for node in backend_report.unsupported_nodes] + backend_report.parameter_errors)
            raise ValueError(f"MODEL is not portable to differentiable backend: {blocked}")
        vector_map = {vector.name: vector for vector in program.vectors}
        trainable_functions = {parameter.function for parameter in backend_report.trainable_parameters}
        for function in program.functions:
            if function.name not in trainable_functions:
                continue
            if function.semantic.kind not in {"softmax_linear", "sigmoid_linear", "linear_regression"}:
                continue
            if not function.semantic.inputs:
                continue
            vector = vector_map.get(function.semantic.inputs[0])
            if vector is not None:
                return vector, function
        raise ValueError("MODEL has no supported trainable softmax_linear, sigmoid_linear or linear_regression function")

    def _objective(self, function: FunctionSpec) -> str:
        if function.semantic.kind == "softmax_linear":
            return "cross_entropy"
        if function.semantic.kind == "sigmoid_linear":
            return "binary_cross_entropy"
        if function.semantic.kind == "linear_regression":
            return "mse"
        raise ValueError(f"Unsupported trainable function kind: {function.semantic.kind}")

    def _parameter_names(self, program: MatrixAIProgram, function: FunctionSpec) -> list[str]:
        report = BackendContractAnalyzer().analyze(program)
        return [parameter.name for parameter in report.trainable_parameters if parameter.function == function.name]

    def _expected_label_count(
        self,
        program: MatrixAIProgram,
        function: FunctionSpec,
        objective: str,
    ) -> int | None:
        if objective == "mse":
            return None
        if objective == "binary_cross_entropy":
            return 2
        report = BackendContractAnalyzer().analyze(program)
        for parameter in report.trainable_parameters:
            if parameter.function == function.name and parameter.role == "weights" and parameter.shape:
                if len(parameter.shape) == 2:
                    return int(parameter.shape[0])
        return None

    def _extract_labels(self, prompt: str) -> list[str]:
        match = self._LABEL_RE.search(prompt)
        if not match:
            return []
        raw = match.group("labels")
        raw = re.split(r"\b(?:con|with|para|for|dataset|datos|target|objetivo)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        parts = re.split(r",|/|\b(?:y|and)\b", raw)
        labels = [_identifier(part) for part in parts]
        return [label for label in labels if label]

    def _extract_name(self, prompt: str, pattern: re.Pattern[str]) -> str:
        match = pattern.search(prompt)
        return _identifier(match.group("name")) if match else ""

    def _default_labels(self, objective: str, prompt: str) -> list[str]:
        if objective == "mse":
            return []
        normalized = _normalize_text(prompt)
        if objective == "binary_cross_entropy":
            for words, labels in [
                (["riesgo", "caida", "caída"], ["bajo", "alto"]),
                (["risk", "fall"], ["low", "high"]),
                (["spam"], ["legit", "spam"]),
            ]:
                if any(w in normalized for w in words):
                    return labels
            return ["low", "high"]
        if any(word in normalized for word in ["email", "correo", "support", "sales"]):
            return ["support", "sales", "operations"]
        if any(word in normalized for word in ["temperatura", "temperature", "kelvin", "celsius", "calor", "frio", "convertir"]):
            return ["baja", "media", "alta"]
        if any(word in normalized for word in ["riesgo", "risk", "fall", "caida", "caída"]):
            return ["bajo", "moderado", "alto"]
        if any(word in normalized for word in ["farmacia", "pharmacy", "medicamento", "dispensacion"]):
            return ["correcto", "revisar", "bloquear"]
        return ["clase_a", "clase_b", "clase_c"]

    def _default_target(self, objective: str) -> str:
        if objective == "mse":
            return "predicted_value"
        return "risk_label" if objective == "binary_cross_entropy" else "label"

    def _training_text(
        self,
        model: str,
        dataset_name: str,
        dataset_source: str,
        vector: VectorSpec,
        target_name: str,
        labels: list[str],
        loss_type: str,
        prediction: str,
        learning_rate: float,
        update: list[str],
        batch_size: int,
        split_train: float,
        split_validation: float,
        split_seed: int,
        epochs: int,
        target_scalar_range: tuple[float, float] | None = None,
    ) -> str:
        is_regression = loss_type == "mse"
        lines = [
            f"MODEL {model}",
            "",
            f"DATASET {dataset_name}",
            f"  SOURCE csv(\"{dataset_source}\")",
            f"  INPUT {vector.name} FROM COLUMNS [",
        ]
        for index, column in enumerate(vector.fields):
            comma = "," if index < len(vector.fields) - 1 else ""
            lines.append(f"    {column}{comma}")
        lines.append("  ]")
        if is_regression:
            if target_scalar_range is not None:
                lo, hi = target_scalar_range
                lines.append(f"  TARGET {target_name}: Scalar[{_decimal(lo)}, {_decimal(hi)}]")
            else:
                lines.append(f"  TARGET {target_name}: Scalar")
        else:
            lines.append(f"  TARGET {target_name}: Label[{', '.join(labels)}]")
        lines.extend(
            [
                f"  SPLIT train={_decimal(split_train)} validation={_decimal(split_validation)} seed={split_seed}",
                f"  BATCH size={batch_size} shuffle=true",
                "END",
                "",
                "LOSS GeneratedLoss",
                f"  TYPE {loss_type}",
                f"  PREDICTION {prediction}",
                f"  TARGET {target_name}",
                "END",
                "",
                "OPTIMIZER GeneratedOptimizer",
                "  TYPE sgd",
                f"  LEARNING_RATE {_decimal(learning_rate)}",
                f"  UPDATE {', '.join(update)}",
                "END",
                "",
            ]
        )
        if is_regression:
            lines.extend([
                "METRIC GeneratedMetric",
                "  TYPE mae",
                f"  PREDICTION {prediction}",
                f"  TARGET {target_name}",
                "END",
            ])
        else:
            lines.extend([
                "METRIC Accuracy",
                "  TYPE accuracy",
                f"  PREDICTION {prediction}",
                f"  TARGET {target_name}",
                "END",
            ])
        lines.extend([
            "",
            "RUN",
            f"  EPOCHS {epochs}",
            "  SAVE_BEST true",
            "END",
        ])
        return "\n".join(lines) + "\n"

    def _dataset_template(
        self,
        vector: VectorSpec,
        target_name: str,
        labels: list[str],
        target_scalar_range: tuple[float, float] | None = None,
    ) -> str:
        handle = io.StringIO()
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([*vector.fields, target_name])
        is_regression = not labels
        rows = 4 if is_regression else max(2, len(labels))
        target_lo, target_hi = target_scalar_range or (0.0, 1.0)
        for row_index in range(rows):
            ratio = 0.0 if rows == 1 else row_index / (rows - 1)
            row_values: list[str] = []
            for fld in vector.fields:
                type_spec = vector.field_types.get(fld)
                if type_spec and type_spec.range and type_spec.range.minimum is not None and type_spec.range.maximum is not None:
                    lo, hi = type_spec.range.minimum, type_spec.range.maximum
                    row_values.append(_decimal(lo + ratio * (hi - lo)))
                else:
                    row_values.append(_decimal(ratio))
            if is_regression:
                target_value = target_lo + ratio * (target_hi - target_lo)
                writer.writerow(row_values + [_decimal(target_value)])
            else:
                writer.writerow(row_values + [labels[row_index % len(labels)]])
        return handle.getvalue()


def _identifier(value: str) -> str:
    candidate = _normalize_text(value).strip().replace("-", "_").replace(" ", "_")
    candidate = re.sub(r"[^a-z0-9_]", "", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        return ""
    if candidate[0].isdigit():
        candidate = f"label_{candidate}"
    return candidate


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _decimal(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"