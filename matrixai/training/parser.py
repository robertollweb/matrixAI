# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Parser del lenguaje `.mxtrain` (`parse_training_file`/`parse_training_text`).

Documenta aquí el bloque OPTIMIZER porque es el único sitio del repositorio
que declara su gramática completa (`docs/` solo enseña un ejemplo en el
tutorial): el bloque `OPTIMIZER <nombre> ... END` admite, en cualquier
orden, estas líneas:

    TYPE <identificador>       -- obligatoria. Aceptado por el PARSER: cualquier
                                   identificador (`sgd`, `adam`, `adamw`, o uno
                                   que no exista todavía). El parser NO decide
                                   qué optimizadores hay de verdad -- eso lo
                                   hace `TrainingVerifier` (`verifier.py`,
                                   lista cerrada) y, más abajo, cada entrenador
                                   (torch instancia `sgd`/`adam`/`adamw`; el
                                   camino stdlib solo sabe `sgd`).
    LEARNING_RATE <real>       -- obligatoria.
    UPDATE <patrón, patrón...> -- obligatoria (solo la usa el camino FUNCTION/
                                   lineal; las redes NETWORK la ignoran).
    WEIGHT_DECAY <real >= 0>   -- OPCIONAL (CONTRATO 118-C3b). El regularizador
                                   L2 que aplican los tres entrenadores torch
                                   (`torch_trainer.py`, `dense_torch_trainer.py`,
                                   `composite_torch_trainer.py`) sobre CUALQUIER
                                   TYPE declarado (sgd/adam/adamw admiten
                                   `weight_decay` en `torch.optim`). El camino
                                   SIN torch (`trainer.py`/`dense_trainer.py`)
                                   no lo aplica: si el texto lo declara, esos
                                   entrenadores se NIEGAN con su motivo en vez
                                   de entrenar ignorándolo.
    SCHEDULE cosine             -- OPCIONAL (CONTRATO 118-C3b). El ÚNICO valor
                                   admitido es `cosine`
                                   (`torch.optim.lr_scheduler.CosineAnnealingLR`,
                                   `T_max` = las épocas MÁXIMAS declaradas en
                                   RUN, un paso por época). Cualquier otro
                                   valor es un error de parseo -- no hay más
                                   programas de tasa en la gramática. Mismo
                                   camino SIN torch que arriba: se niega, no
                                   lo ignora.

Sin `WEIGHT_DECAY` ni `SCHEDULE` declarados, el `OptimizerSpec` resultante es
BYTE-IDÉNTICO al de antes de 118-C3b (`weight_decay=0.0`, `schedule=None`, y
`to_dict()` no añade esas claves) -- ver `test_118_c3b_optimizador.py`.
"""

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
from matrixai.training.particion import PROTOCOLO_SEPARACION as _PROTOCOLO_SEPARACION
from matrixai.types import parse_type_spec


class MatrixAITrainingParseError(ValueError):
    pass


_SOURCE_RE = re.compile(r'^SOURCE\s+(?P<kind>[A-Za-z_][\w]*)\("(?P<source>.+)"\)$')
_INPUT_RE = re.compile(r"^INPUT\s+(?P<vector>[A-Za-z_][\w]*)\s+FROM\s+COLUMNS\s+(?P<columns>.+)$")
_TARGET_RE = re.compile(r"^TARGET\s+(?P<name>[A-Za-z_][\w]*)\s*:\s*(?P<type>.+)$")
_SPLIT_RE = re.compile(
    # CONTRATO 101-C0: `test=` y `protocol=` son opcionales y van AL FINAL, en
    # ese orden, para que una declaración de antes case exactamente igual que
    # antes — el patrón no cambia para nadie que no los escriba.
    r"^SPLIT\s+train=(?P<train>[0-9.]+)\s+validation=(?P<validation>[0-9.]+)"
    r"(?:\s+test=(?P<test>[0-9.]+))?"
    r"(?:\s+seed=(?P<seed>\d+))?(?:\s+mode=(?P<mode>temporal|random))?"
    r"(?:\s+protocol=(?P<protocol>[0-9]+))?$"
)
_BATCH_RE = re.compile(r"^BATCH\s+size=(?P<size>\d+)(?:\s+shuffle=(?P<shuffle>true|false))?$")
_TYPE_RE = re.compile(r"^TYPE\s+(?P<type>[A-Za-z_][\w]*)$")
_PREDICTION_RE = re.compile(r"^PREDICTION\s+(?P<prediction>[A-Za-z_][\w.]*)$")
_LEARNING_RATE_RE = re.compile(r"^LEARNING_RATE\s+(?P<learning_rate>[0-9.]+)$")
_UPDATE_RE = re.compile(r"^UPDATE\s+(?P<update>.+)$")
# CONTRATO 118-C3b. Igual que `_LEARNING_RATE_RE`, sin signo: la clase de
# caracteres `[0-9.]+` ya impone "real >= 0" a nivel de patrón, así que un
# valor negativo ("WEIGHT_DECAY -0.1") no casa y cae en la línea DESCONOCIDA
# de más abajo, con el mismo mensaje que cualquier otra línea mal formada.
_WEIGHT_DECAY_RE = re.compile(r"^WEIGHT_DECAY\s+(?P<weight_decay>[0-9.]+)$")
# El valor puede ser cualquier token (no solo un identificador): así un
# `SCHEDULE 0.1` o `SCHEDULE "cosine"` mal escrito cae en la MISMA rama de
# "solo se admite cosine" en vez de en la genérica "Unknown OPTIMIZER line".
_SCHEDULE_RE = re.compile(r"^SCHEDULE\s+(?P<schedule>\S+)$")
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
            train_ratio = float(match.group("train"))
            validation_ratio = float(match.group("validation"))
            seed = int(match.group("seed")) if match.group("seed") else None
            mode = match.group("mode") or "random"
            test_ratio = float(match.group("test")) if match.group("test") else None
            protocol = match.group("protocol") or None
            # BIBLIOTECA_PROYECTOS_INTELIGENTES C3 (auditoría [MEDIA]): antes
            # se aceptaba cualquier train/validation (0.9+0.9, train=0,
            # train=1...) — los trainers solo usan `train` para el corte y
            # ajustan el resultado en silencio; `validation` era meramente
            # informativo, así que una declaración incoherente NUNCA se
            # notaba. Vocabulario cerrado también en los VALORES.
            if not (0.0 < train_ratio < 1.0):
                raise MatrixAITrainingParseError(
                    f"Invalid SPLIT declaration: train={train_ratio} debe estar "
                    f"estrictamente entre 0 y 1: {line}"
                )
            if not (0.0 < validation_ratio < 1.0):
                raise MatrixAITrainingParseError(
                    f"Invalid SPLIT declaration: validation={validation_ratio} debe "
                    f"estar estrictamente entre 0 y 1: {line}"
                )
            # CONTRATO 101-C0: con tres tramos, los tres suman 1,0. El mensaje
            # nombra los que hay, no los que debería haber.
            if test_ratio is not None and not (0.0 < test_ratio < 1.0):
                raise MatrixAITrainingParseError(
                    f"Invalid SPLIT declaration: test={test_ratio} debe estar "
                    f"estrictamente entre 0 y 1: {line}"
                )
            suma = train_ratio + validation_ratio + (test_ratio or 0.0)
            if abs(suma - 1.0) > 1e-6:
                partes = (f"train={train_ratio} + validation={validation_ratio}"
                          + (f" + test={test_ratio}" if test_ratio is not None else ""))
                raise MatrixAITrainingParseError(
                    f"Invalid SPLIT declaration: {partes} debe sumar 1.0: {line}"
                )
            # UN TRAMO DE PRUEBA QUE NADIE HONRA ES PEOR QUE NO TENERLO. Sin
            # `protocol=2` los entrenadores parten como siempre —0,8 fijo,
            # secuencial— y ese `test=` se quedaría escrito sin efecto: alguien
            # leería su `.mxtrain`, vería una prueba reservada y creería que el
            # número sale de ahí. Se rechaza en vez de ignorarlo en silencio,
            # igual que se hace arriba con `mode=temporal seed=`.
            if test_ratio is not None and protocol != _PROTOCOLO_SEPARACION:
                raise MatrixAITrainingParseError(
                    f"Invalid SPLIT declaration: test={test_ratio} necesita "
                    f"protocol={_PROTOCOLO_SEPARACION}; sin él los entrenadores parten "
                    f"como siempre y el tramo de prueba se quedaría escrito sin "
                    f"efecto: {line}"
                )
            if protocol is not None and protocol != _PROTOCOLO_SEPARACION:
                raise MatrixAITrainingParseError(
                    f"Invalid SPLIT declaration: protocol={protocol} no existe; el "
                    f"único que cambia cómo se parte es {_PROTOCOLO_SEPARACION}: {line}"
                )
            # mode=temporal nunca baraja (invariante 12 del contrato 57: "sin
            # barajar") — un seed ahí no tendría ningún efecto; declararlo de
            # todos modos es casi siempre una confusión del usuario ("¿por
            # qué mi seed no cambia nada?"), así que se rechaza en vez de
            # aceptarlo e ignorarlo en silencio.
            if mode == "temporal" and seed is not None:
                raise MatrixAITrainingParseError(
                    f"Invalid SPLIT declaration: mode=temporal no admite seed "
                    f"(nunca baraja, el seed no tendría efecto): {line}"
                )
            split = DatasetSplitSpec(
                train=train_ratio, validation=validation_ratio, seed=seed, mode=mode,
                test=test_ratio, protocol=protocol,
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
    # CONTRATO 118-C3b: por omisión (nada declarado) esto reproduce byte a
    # byte el `OptimizerSpec` de antes de la enmienda.
    weight_decay = 0.0
    schedule: str | None = None
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
        if line.startswith("WEIGHT_DECAY "):
            match = _WEIGHT_DECAY_RE.match(line)
            if not match:
                raise MatrixAITrainingParseError(
                    f"Invalid WEIGHT_DECAY: {line} — debe ser un real >= 0: {line}"
                )
            weight_decay = float(match.group("weight_decay"))
            continue
        if line.startswith("SCHEDULE "):
            match = _SCHEDULE_RE.match(line)
            if not match or match.group("schedule") != "cosine":
                raise MatrixAITrainingParseError(
                    f"Invalid SCHEDULE: {line!r}. El único programa de tasa "
                    f"admitido es 'cosine'."
                )
            schedule = match.group("schedule")
            continue
        raise MatrixAITrainingParseError(f"Unknown OPTIMIZER line: {line}")
    if not optimizer_type:
        raise MatrixAITrainingParseError("OPTIMIZER missing TYPE")
    if learning_rate is None:
        raise MatrixAITrainingParseError("OPTIMIZER missing LEARNING_RATE")
    if not update:
        raise MatrixAITrainingParseError("OPTIMIZER missing UPDATE")
    return OptimizerSpec(
        parts[1], optimizer_type, learning_rate, update,
        weight_decay=weight_decay, schedule=schedule,
    )


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