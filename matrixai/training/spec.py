# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matrixai.types import TypeSpec


@dataclass(frozen=True)
class DatasetInputSpec:
    vector: str
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"vector": self.vector, "columns": list(self.columns)}


@dataclass(frozen=True)
class DatasetTargetSpec:
    name: str
    type: TypeSpec

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type.to_dict()}


@dataclass(frozen=True)
class DatasetSplitSpec:
    train: float
    validation: float
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"train": self.train, "validation": self.validation}
        if self.seed is not None:
            data["seed"] = self.seed
        return data


@dataclass(frozen=True)
class DatasetBatchSpec:
    size: int
    shuffle: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"size": self.size, "shuffle": self.shuffle}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_kind: str
    source: str
    input: DatasetInputSpec
    target: DatasetTargetSpec
    split: DatasetSplitSpec | None = None
    batch: DatasetBatchSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "source_kind": self.source_kind,
            "source": self.source,
            "input": self.input.to_dict(),
            "target": self.target.to_dict(),
        }
        if self.split is not None:
            data["split"] = self.split.to_dict()
        if self.batch is not None:
            data["batch"] = self.batch.to_dict()
        return data


@dataclass(frozen=True)
class LossSpec:
    name: str
    type: str
    prediction: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "prediction": self.prediction,
            "target": self.target,
        }


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    type: str
    learning_rate: float
    update: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "learning_rate": self.learning_rate,
            "update": list(self.update),
        }


@dataclass(frozen=True)
class MetricSpec:
    name: str
    type: str
    prediction: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "prediction": self.prediction,
            "target": self.target,
        }


@dataclass(frozen=True)
class RunSpec:
    epochs: int
    early_stop_patience: int | None = None
    early_stop_metric: str | None = None
    save_best: bool = True

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"epochs": self.epochs, "save_best": self.save_best}
        if self.early_stop_patience is not None:
            data["early_stop_patience"] = self.early_stop_patience
        if self.early_stop_metric is not None:
            data["early_stop_metric"] = self.early_stop_metric
        return data


_VALID_TARGETS = frozenset({"stdlib", "torch"})
_VALID_DEVICES = frozenset({"cpu", "cuda", "mps"})


@dataclass(frozen=True)
class BackendSpec:
    target: str = "stdlib"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.target not in _VALID_TARGETS:
            raise ValueError(f"BackendSpec: invalid target {self.target!r}. Must be one of: {sorted(_VALID_TARGETS)}")
        if self.device not in _VALID_DEVICES:
            raise ValueError(f"BackendSpec: invalid device {self.device!r}. Must be one of: {sorted(_VALID_DEVICES)}")
        if self.target == "stdlib" and self.device != "cpu":
            raise ValueError(
                f"BackendSpec: target='stdlib' only supports device='cpu', got device={self.device!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "device": self.device}


@dataclass(frozen=True)
class TrainingSpec:
    model: str
    dataset: DatasetSpec
    loss: LossSpec
    optimizer: OptimizerSpec
    metrics: list[MetricSpec] = field(default_factory=list)
    run: RunSpec | None = None
    backend: BackendSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "model": self.model,
            "dataset": self.dataset.to_dict(),
            "loss": self.loss.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }
        if self.run is not None:
            data["run"] = self.run.to_dict()
        if self.backend is not None:
            data["backend"] = self.backend.to_dict()
        return data


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    output_dir: str
    best_epoch: int
    best_validation_loss: float
    final_train_loss: float
    final_validation_loss: float
    accuracy: float
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_train_loss": self.final_train_loss,
            "final_validation_loss": self.final_validation_loss,
            "accuracy": self.accuracy,
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    model_hash: str
    parameter_schema_hash: str
    parameter_set_id: str
    dataset: str
    dataset_fingerprint: str
    dataset_schema: dict[str, Any]
    rows: int
    loss: float
    accuracy: float
    labels: list[str]
    confusion_matrix: dict[str, dict[str, int]]
    per_label: dict[str, dict[str, float]]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    backend: dict[str, Any] = field(default_factory=dict)
    backend_runtime: dict[str, Any] = field(default_factory=dict)
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0

    def is_regression(self) -> bool:
        return not self.labels

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model": self.model,
            "model_hash": self.model_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
            "parameter_set_id": self.parameter_set_id,
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_schema": self.dataset_schema,
            "rows": self.rows,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "labels": list(self.labels),
            "confusion_matrix": self.confusion_matrix,
            "per_label": self.per_label,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }
        if self.is_regression():
            data["mae"] = self.mae
            data["rmse"] = self.rmse
            data["r2"] = self.r2
        if self.backend:
            data["backend"] = dict(self.backend)
        if self.backend_runtime:
            data["backend_runtime"] = dict(self.backend_runtime)
        return data
