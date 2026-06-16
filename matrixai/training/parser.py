# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import re
from pathlib import Path

from matrixai.training.spec import (
    BackendSpec,
    DatasetBatchSpec,
    DatasetInputSpec,
    DatasetSpec,
    DatasetSplitSpec,
    DatasetTargetSpec,
    LossSpec,
    MetricSpec,
    OptimizerSpec,
    RunSpec,
    TrainingSpec,
)
from matrixai.types import parse_type_spec


class MatrixAITrainingParseError(ValueError):
    pass


_SOURCE_RE = re.compile(r'^SOURCE\s+(?P<kind>[A-Za-z_][\w]*)\("(?P<source>.+)"\)$')
_INPUT_RE = re.compile(r"^INPUT\s+(?P<vector>[A-Za-z_][\w]*)\s+FROM\s+COLUMNS\s+(?P<columns>.+)$")
_TARGET_RE = re.compile(r"^TARGET\s+(?P<name>[A-Za-z_][\w]*)\s*:\s*(?P<type>.+)$")
_SPLIT_RE = re.compile(
    r"^SPLIT\s+train=(?P<train>[0-9.]+)\s+validation=(?P<validation>[0-9.]+)(?:\s+seed=(?P<seed>\d+))?$"
)
_BATCH_RE = re.compile(r"^BATCH\s+size=(?P<size>\d+)(?:\s+shuffle=(?P<shuffle>true|false))?$")
_TYPE_RE = re.compile(r"^TYPE\s+(?P<type>[A-Za-z_][\w]*)$")
_PREDICTION_RE = re.compile(r"^PREDICTION\s+(?P<prediction>[A-Za-z_][\w.]*)$")
_LEARNING_RATE_RE = re.compile(r"^LEARNING_RATE\s+(?P<learning_rate>[0-9.]+)$")
_UPDATE_RE = re.compile(r"^UPDATE\s+(?P<update>.+)$")
_EPOCHS_RE = re.compile(r"^EPOCHS\s+(?P<epochs>\d+)$")
_EARLY_STOP_RE = re.compile(
    r"^EARLY_STOP\s+patience=(?P<patience>\d+)\s+metric=(?P<metric>[A-Za-z_][\w.]*)$"
)
_SAVE_BEST_RE = re.compile(r"^SAVE_BEST\s+(?P<save_best>true|false)$")
_BACKEND_TARGET_RE = re.compile(r"^TARGET\s+(?P<target>stdlib|torch)$")
_BACKEND_DEVICE_RE = re.compile(r"^DEVICE\s+(?P<device>cpu|cuda|mps)$")


def parse_training_file(path: str | Path) -> TrainingSpec:
    return parse_training_text(Path(path).read_text(encoding="utf-8"))


def parse_training_text(text: str) -> TrainingSpec:
    lines = _clean_lines(text)
    if not lines:
        raise MatrixAITrainingParseError("Empty MatrixAI training document")

    model = ""
    dataset: DatasetSpec | None = None
    loss: LossSpec | None = None
    optimizer: OptimizerSpec | None = None
    metrics: list[MetricSpec] = []
    run: RunSpec | None = None
    backend: BackendSpec | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        keyword = line.split(maxsplit=1)[0]

        if keyword == "MODEL":
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise MatrixAITrainingParseError("MODEL requires a path")
            model = parts[1].strip()
            index += 1
            continue

        if keyword == "DATASET":
            block, index = _read_block(lines, index)
            dataset = _parse_dataset(block)
            continue

        if keyword == "LOSS":
            block, index = _read_block(lines, index)
            loss = _parse_loss(block)
            continue

        if keyword == "OPTIMIZER":
            block, index = _read_block(lines, index)
            optimizer = _parse_optimizer(block)
            continue

        if keyword == "METRIC":
            block, index = _read_block(lines, index)
            metrics.append(_parse_metric(block))
            continue

        if keyword == "RUN":
            block, index = _read_block(lines, index)
            run = _parse_run(block)
            continue

        if keyword == "BACKEND":
            block, index = _read_block(lines, index)
            backend = _parse_backend(block)
            continue

        raise MatrixAITrainingParseError(f"Unknown training block: {line}")

    if not model:
        raise MatrixAITrainingParseError("Missing MODEL declaration")
    if dataset is None:
        raise MatrixAITrainingParseError("Missing DATASET block")
    if loss is None:
        raise MatrixAITrainingParseError("Missing LOSS block")
    if optimizer is None:
        raise MatrixAITrainingParseError("Missing OPTIMIZER block")

    return TrainingSpec(
        model=model,
        dataset=dataset,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        run=run,
        backend=backend,
    )


def _clean_lines(text: str) -> list[str]:
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cleaned.append(line)
    return cleaned


def _read_block(lines: list[str], start: int) -> tuple[list[str], int]:
    block = [lines[start]]
    index = start + 1
    while index < len(lines):
        block.append(lines[index])
        if lines[index] == "END":
            return block, index + 1
        index += 1
    raise MatrixAITrainingParseError(f"Block '{lines[start]}' is missing END")


def _parse_dataset(block: list[str]) -> DatasetSpec:
    parts = block[0].split(maxsplit=1)
    if len(parts) != 2:
        raise MatrixAITrainingParseError("DATASET requires a name")

    source_kind = ""
    source = ""
    input_spec: DatasetInputSpec | None = None
    target: DatasetTargetSpec | None = None
    split: DatasetSplitSpec | None = None
    batch: DatasetBatchSpec | None = None

    body = block[1:-1]
    index = 0
    while index < len(body):
        line = body[index]
        if line.startswith("SOURCE "):
            match = _SOURCE_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid SOURCE declaration: {line}")
            source_kind = match.group("kind")
            source = match.group("source")
            index += 1
            continue
        if line.startswith("INPUT "):
            input_spec, index = _parse_input(body, index)
            continue
        if line.startswith("TARGET "):
            match = _TARGET_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid TARGET declaration: {line}")
            try:
                target = DatasetTargetSpec(match.group("name"), parse_type_spec(match.group("type")))
            except ValueError as exc:
                raise MatrixAITrainingParseError(f"Invalid TARGET type: {exc}") from exc
            index += 1
            continue
        if line.startswith("SPLIT "):
            match = _SPLIT_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid SPLIT declaration: {line}")
            split = DatasetSplitSpec(
                train=float(match.group("train")),
                validation=float(match.group("validation")),
                seed=int(match.group("seed")) if match.group("seed") else None,
            )
            index += 1
            continue
        if line.startswith("BATCH "):
            match = _BATCH_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid BATCH declaration: {line}")
            batch = DatasetBatchSpec(
                size=int(match.group("size")),
                shuffle=match.group("shuffle") == "true",
            )
            index += 1
            continue
        raise MatrixAITrainingParseError(f"Unknown DATASET line: {line}")

    if not source_kind or not source:
        raise MatrixAITrainingParseError(f"DATASET {parts[1]} missing SOURCE")
    if input_spec is None:
        raise MatrixAITrainingParseError(f"DATASET {parts[1]} missing INPUT")
    if target is None:
        raise MatrixAITrainingParseError(f"DATASET {parts[1]} missing TARGET")

    return DatasetSpec(
        name=parts[1],
        source_kind=source_kind,
        source=source,
        input=input_spec,
        target=target,
        split=split,
        batch=batch,
    )


def _parse_input(body: list[str], start: int) -> tuple[DatasetInputSpec, int]:
    line = body[start]
    match = _INPUT_RE.match(line)
    if not match:
        raise MatrixAITrainingParseError(f"Invalid INPUT declaration: {line}")
    columns_text = match.group("columns")
    consumed = start + 1
    while "[" in columns_text and "]" not in columns_text and consumed < len(body):
        columns_text += " " + body[consumed]
        consumed += 1
    columns = _parse_columns(columns_text)
    if not columns:
        raise MatrixAITrainingParseError(f"INPUT {match.group('vector')} requires columns")
    return DatasetInputSpec(vector=match.group("vector"), columns=columns), consumed


def _parse_columns(text: str) -> list[str]:
    if "[" not in text or "]" not in text:
        raise MatrixAITrainingParseError(f"Column list must use brackets: {text}")
    payload = text[text.index("[") + 1:text.rindex("]")]
    return [item.strip() for item in payload.split(",") if item.strip()]


def _parse_loss(block: list[str]) -> LossSpec:
    parts = block[0].split(maxsplit=1)
    if len(parts) != 2:
        raise MatrixAITrainingParseError("LOSS requires a name")
    values = _parse_common_prediction_block(block[1:-1], "LOSS")
    return LossSpec(
        name=parts[1],
        type=values["type"],
        prediction=values["prediction"],
        target=values["target"],
    )


def _parse_metric(block: list[str]) -> MetricSpec:
    parts = block[0].split(maxsplit=1)
    if len(parts) != 2:
        raise MatrixAITrainingParseError("METRIC requires a name")
    values = _parse_common_prediction_block(block[1:-1], "METRIC")
    return MetricSpec(
        name=parts[1],
        type=values["type"],
        prediction=values["prediction"],
        target=values["target"],
    )


def _parse_common_prediction_block(lines: list[str], block_name: str) -> dict[str, str]:
    values = {"type": "", "prediction": "", "target": ""}
    for line in lines:
        if line.startswith("TYPE "):
            match = _TYPE_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid {block_name} TYPE: {line}")
            values["type"] = match.group("type")
            continue
        if line.startswith("PREDICTION "):
            match = _PREDICTION_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid {block_name} PREDICTION: {line}")
            values["prediction"] = match.group("prediction")
            continue
        if line.startswith("TARGET "):
            values["target"] = line.split(maxsplit=1)[1].strip()
            continue
        raise MatrixAITrainingParseError(f"Unknown {block_name} line: {line}")
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise MatrixAITrainingParseError(f"{block_name} missing {', '.join(missing)}")
    return values


def _parse_optimizer(block: list[str]) -> OptimizerSpec:
    parts = block[0].split(maxsplit=1)
    if len(parts) != 2:
        raise MatrixAITrainingParseError("OPTIMIZER requires a name")
    optimizer_type = ""
    learning_rate: float | None = None
    update: list[str] = []
    for line in block[1:-1]:
        if line.startswith("TYPE "):
            match = _TYPE_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid OPTIMIZER TYPE: {line}")
            optimizer_type = match.group("type")
            continue
        if line.startswith("LEARNING_RATE "):
            match = _LEARNING_RATE_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid LEARNING_RATE: {line}")
            learning_rate = float(match.group("learning_rate"))
            continue
        if line.startswith("UPDATE "):
            match = _UPDATE_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid UPDATE: {line}")
            update = [item.strip() for item in match.group("update").split(",") if item.strip()]
            continue
        raise MatrixAITrainingParseError(f"Unknown OPTIMIZER line: {line}")
    if not optimizer_type:
        raise MatrixAITrainingParseError("OPTIMIZER missing TYPE")
    if learning_rate is None:
        raise MatrixAITrainingParseError("OPTIMIZER missing LEARNING_RATE")
    if not update:
        raise MatrixAITrainingParseError("OPTIMIZER missing UPDATE")
    return OptimizerSpec(parts[1], optimizer_type, learning_rate, update)


def _parse_run(block: list[str]) -> RunSpec:
    epochs: int | None = None
    early_stop_patience: int | None = None
    early_stop_metric: str | None = None
    save_best = True
    for line in block[1:-1]:
        if line.startswith("EPOCHS "):
            match = _EPOCHS_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid EPOCHS: {line}")
            epochs = int(match.group("epochs"))
            continue
        if line.startswith("EARLY_STOP "):
            match = _EARLY_STOP_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid EARLY_STOP: {line}")
            early_stop_patience = int(match.group("patience"))
            early_stop_metric = match.group("metric")
            continue
        if line.startswith("SAVE_BEST "):
            match = _SAVE_BEST_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid SAVE_BEST: {line}")
            save_best = match.group("save_best") == "true"
            continue
        raise MatrixAITrainingParseError(f"Unknown RUN line: {line}")
    if epochs is None:
        raise MatrixAITrainingParseError("RUN missing EPOCHS")
    return RunSpec(
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        early_stop_metric=early_stop_metric,
        save_best=save_best,
    )


def _parse_backend(block: list[str]) -> BackendSpec:
    target = "stdlib"
    device = "cpu"
    for line in block[1:-1]:
        if line.startswith("TARGET "):
            match = _BACKEND_TARGET_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid BACKEND TARGET: {line!r}. Must be 'stdlib' or 'torch'.")
            target = match.group("target")
            continue
        if line.startswith("DEVICE "):
            match = _BACKEND_DEVICE_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(f"Invalid BACKEND DEVICE: {line!r}. Must be 'cpu', 'cuda' or 'mps'.")
            device = match.group("device")
            continue
        raise MatrixAITrainingParseError(f"Unknown BACKEND line: {line!r}")
    try:
        return BackendSpec(target=target, device=device)
    except ValueError as exc:
        raise MatrixAITrainingParseError(str(exc)) from exc