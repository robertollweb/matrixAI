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
    # BIBLIOTECA_PROYECTOS_INTELIGENTES C3: "random" (default) es el
    # comportamiento de SIEMPRE, byte-idéntico — barajado con `seed` si se
    # declara, secuencial si no. "temporal" NUNCA baraja (aunque haya
    # seed): el último tramo, en el orden que llega, es SIEMPRE la
    # validación — para series temporales, donde barajar sería fuga
    # (invariante 6 del contrato 57).
    mode: str = "random"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"train": self.train, "validation": self.validation}
        if self.seed is not None:
            data["seed"] = self.seed
        # Auditoría [MEDIA]: solo se anota si es "temporal" — "random" es el
        # default de SIEMPRE (pre-C3), así que la serialización de un split
        # sin `mode` declarado queda BYTE-IDÉNTICA a antes de C3 (mismo
        # criterio que `seed` arriba).
        if self.mode == "temporal":
            data["mode"] = self.mode
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
    #: El ESFUERZO real del run: con cuántos ejemplos por paso se
    #: actualizaron los pesos, cuántos pasos tuvo cada época y cuántas
    #: actualizaciones salieron en total. `None` si el entrenador no lo
    #: declara. Ver `esfuerzo_de_entrenamiento`.
    effort: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_train_loss": self.final_train_loss,
            "final_validation_loss": self.final_validation_loss,
            "accuracy": self.accuracy,
            "artifacts": dict(self.artifacts),
        }
        if self.effort is not None:
            d["effort"] = dict(self.effort)
        return d


def esfuerzo_de_entrenamiento(
    filas_de_entrenamiento: int, lote: int, epocas: int,
) -> dict[str, int]:
    """El esfuerzo REAL de un run: pasos por época y actualizaciones.

    La época NO es una unidad comparable entre máquinas; la actualización
    de pesos sí. Medido el 2026-08-09 con un millón de filas:

    | camino                        | lote   | pasos por época |
    |-------------------------------|--------|-----------------|
    | stdlib (por defecto sin GPU)  |      1 |       1.000.000 |
    | torch en CPU                  |      8 |         125.000 |
    | torch en CUDA                 | 16.384 |              62 |

    Dieciséis mil veces entre el camino por defecto y el de GPU. Quien
    entrena «50 épocas» en su portátil y luego «50 épocas» en Colab NO
    está repitiendo el experimento, aunque las dos pantallas digan 50.

    Vive aquí —y no en cada entrenador— porque los tres tienen que contar
    lo mismo: dos sitios calculando esto con reglas distintas es tener uno
    de los dos mal, y nadie sabría cuál.
    """
    lote_seguro = max(1, int(lote or 1))
    filas = max(0, int(filas_de_entrenamiento or 0))
    # Techo, no división entera: la última hornada cuenta aunque no llene
    # el lote. Con 51 filas y lote 8 son SIETE pasos, no seis.
    pasos = -(-filas // lote_seguro) if filas else 0
    return {
        "effective_batch_size": lote_seguro,
        "train_rows": filas,
        "steps_per_epoch": pasos,
        "weight_updates": pasos * max(0, int(epocas or 0)),
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
    # Anexo 58.1 C1: criterio EXPLÍCITO opcional, más fiable que `not labels`
    # — una clasificación BINARIA sin nombres de clase declarados (Probability
    # + binary_cross_entropy, sin bloque LABELS) también tiene `labels=[]`, así
    # que el criterio heurístico la confundía con regresión (reproducido con
    # `test_binary_classification_has_metrics`: is_regression() daba True para
    # un modelo de clasificación real). Los constructores que YA conocen el
    # loss_fn (dense/composite/torch/legado lineal) lo pasan aquí; los que no
    # lo pasan (código antiguo, tests que construyen el dataclass a mano)
    # caen al criterio heredado — retrocompatible, ningún comportamiento
    # previo cambia para quien no pase este campo.
    loss_fn: str = ""

    def is_regression(self) -> bool:
        if self.loss_fn:
            return self.loss_fn == "mse"
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
        }
        # Anexo 58.1 C1: espejo de `DenseEvaluationResult.to_dict()`
        # (dense_evaluator.py) — antes SIEMPRE incluía accuracy/labels/
        # confusion_matrix/per_label/macro_precision/macro_recall/macro_f1
        # con sus valores por defecto (0.0/[]/{}), incluso en regresión, así
        # que un consumidor que solo hiciera `.get("macro_f1")` nunca veía
        # `None` — veía 0.0, indistinguible de "el modelo no aprendió nada".
        # Un `0.0` de una tarea que no aplica NUNCA debe llegar aguas abajo.
        if self.is_regression():
            data["mae"] = self.mae
            data["rmse"] = self.rmse
            data["r2"] = self.r2
        else:
            data["accuracy"] = self.accuracy
            data["labels"] = list(self.labels)
            data["confusion_matrix"] = self.confusion_matrix
            data["per_label"] = self.per_label
            data["macro_precision"] = self.macro_precision
            data["macro_recall"] = self.macro_recall
            data["macro_f1"] = self.macro_f1
        if self.backend:
            data["backend"] = dict(self.backend)
        if self.backend_runtime:
            data["backend_runtime"] = dict(self.backend_runtime)
        return data
