# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetSchema:
    source_kind: str
    source: str
    input_vector: str
    input_columns: list[str]
    target: str
    labels: list[str]
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source": self.source,
            "input_vector": self.input_vector,
            "input_columns": list(self.input_columns),
            "target": self.target,
            "labels": list(self.labels),
            "rows": self.rows,
        }


@dataclass(frozen=True)
class SupervisedExample:
    vector: list[float]
    label: str
    row_index: int
    row_hash: str
    target_value: float | None = None


@dataclass(frozen=True)
class MatrixAIBatch:
    inputs: dict[str, list[list[float]]]
    targets: dict[str, list[str]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs,
            "targets": self.targets,
            "metadata": dict(self.metadata),
        }


class DataAdapter:
    def schema(self) -> DatasetSchema:
        raise NotImplementedError

    def fingerprint(self) -> str:
        raise NotImplementedError

    def examples(self) -> list[SupervisedExample]:
        raise NotImplementedError

    def iter_batches(
        self,
        batch_size: int,
        indices: Iterable[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        raise NotImplementedError


class CSVDataAdapter(DataAdapter):
    def __init__(
        self,
        path: str | Path,
        input_vector: str,
        input_columns: list[str],
        target: str,
        labels: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.input_vector = input_vector
        self.input_columns = list(input_columns)
        self.target = target
        self.labels = list(labels or [])
        self._examples: list[SupervisedExample] | None = None

    def schema(self) -> DatasetSchema:
        return DatasetSchema(
            source_kind="csv",
            source=str(self.path),
            input_vector=self.input_vector,
            input_columns=list(self.input_columns),
            target=self.target,
            labels=list(self.labels),
            rows=len(self.examples()),
        )

    def fingerprint(self) -> str:
        return dataset_fingerprint(self.path)

    def examples(self) -> list[SupervisedExample]:
        if self._examples is None:
            self._examples = self._load_examples()
        return list(self._examples)

    def iter_batches(
        self,
        batch_size: int,
        indices: Iterable[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        examples = self.examples()
        selected = list(indices) if indices is not None else list(range(len(examples)))
        if shuffle:
            random.Random(seed).shuffle(selected)
        size = max(1, batch_size)
        for offset in range(0, len(selected), size):
            yield self.batch_from_examples(examples[index] for index in selected[offset:offset + size])

    def batch_from_examples(self, examples: Iterable[SupervisedExample]) -> MatrixAIBatch:
        batch_examples = list(examples)
        return MatrixAIBatch(
            inputs={self.input_vector: [example.vector for example in batch_examples]},
            targets={self.target: [example.label for example in batch_examples]},
            metadata={
                "source": str(self.path),
                "dataset_fingerprint": self.fingerprint(),
                "row_indices": [example.row_index for example in batch_examples],
                "row_hashes": [example.row_hash for example in batch_examples],
            },
        )

    def _encode_row(self, row: dict[str, str]) -> list[float]:
        return [float(row[column]) for column in self.input_columns]

    def _load_examples(self) -> list[SupervisedExample]:
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        is_regression = not self.labels
        examples: list[SupervisedExample] = []
        for row_offset, row in enumerate(rows, start=2):
            label_str = str(row[self.target])
            target_value: float | None = None
            if is_regression:
                try:
                    target_value = float(label_str)
                except (ValueError, TypeError):
                    pass
            examples.append(
                SupervisedExample(
                    vector=self._encode_row(row),
                    label=label_str,
                    row_index=row_offset,
                    row_hash=_row_hash(row_offset, row),
                    target_value=target_value,
                )
            )
        return examples


class CSVTextDataAdapter(CSVDataAdapter):
    """SECUENCIAS_PRODUCTO C3 — como `CSVDataAdapter` pero la ÚNICA columna
    es texto CRUDO, no floats: tokeniza cada fila en el boundary de carga
    (invariante 1 del contrato — el CSV guarda texto, el trainer ve ids;
    nunca se le piden ids al usuario). Solo sustituye la codificación de
    fila (`_encode_row`); `schema()`/`fingerprint()`/`iter_batches()` se
    heredan intactos — `SupervisedExample.vector` sigue siendo `list[float]`
    (los ids), así que el resto del pipeline (`_examples_to_xy`, `int(v) for
    v in x`) no cambia.

    `tokenizer` es cualquier objeto con `.encode(text: str) -> list[int]`
    (duck-typed a propósito — este módulo es genérico, no depende de
    `matrixai.text`; el caller pasa un `ByteTokenizer`)."""

    def __init__(
        self,
        path: str | Path,
        input_vector: str,
        text_column: str,
        target: str,
        tokenizer: Any,
        labels: list[str] | None = None,
    ) -> None:
        super().__init__(path, input_vector, [text_column], target, labels)
        self.text_column = text_column
        self.tokenizer = tokenizer

    def _encode_row(self, row: dict[str, str]) -> list[float]:
        ids = self.tokenizer.encode(str(row[self.text_column]))
        return [float(i) for i in ids]


class InMemoryDataAdapter(DataAdapter):
    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        input_vector: str,
        input_columns: list[str],
        target: str,
        labels: list[str] | None = None,
        source: str = "memory",
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.input_vector = input_vector
        self.input_columns = list(input_columns)
        self.target = target
        self.labels = list(labels or [])
        self.source = source
        self._examples: list[SupervisedExample] | None = None
        self._fingerprint: str | None = None

    def schema(self) -> DatasetSchema:
        return DatasetSchema(
            source_kind="memory",
            source=self.source,
            input_vector=self.input_vector,
            input_columns=list(self.input_columns),
            target=self.target,
            labels=list(self.labels),
            rows=len(self.examples()),
        )

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            payload = {
                "source_kind": "memory",
                "input_vector": self.input_vector,
                "input_columns": self.input_columns,
                "target": self.target,
                "labels": self.labels,
                "rows": self.rows,
            }
            self._fingerprint = _fingerprint_payload("data", payload)
        return self._fingerprint

    def examples(self) -> list[SupervisedExample]:
        if self._examples is None:
            self._examples = self._load_examples()
        return list(self._examples)

    def iter_batches(
        self,
        batch_size: int,
        indices: Iterable[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        examples = self.examples()
        selected = list(indices) if indices is not None else list(range(len(examples)))
        if shuffle:
            random.Random(seed).shuffle(selected)
        size = max(1, batch_size)
        for offset in range(0, len(selected), size):
            yield self.batch_from_examples(examples[index] for index in selected[offset:offset + size])

    def batch_from_examples(self, examples: Iterable[SupervisedExample]) -> MatrixAIBatch:
        batch_examples = list(examples)
        return MatrixAIBatch(
            inputs={self.input_vector: [example.vector for example in batch_examples]},
            targets={self.target: [example.label for example in batch_examples]},
            metadata={
                "source": self.source,
                "dataset_fingerprint": self.fingerprint(),
                "row_indices": [example.row_index for example in batch_examples],
                "row_hashes": [example.row_hash for example in batch_examples],
            },
        )

    def _load_examples(self) -> list[SupervisedExample]:
        is_regression = not self.labels
        examples: list[SupervisedExample] = []
        for row_index, row in enumerate(self.rows, start=1):
            label_str = str(row[self.target])
            target_value: float | None = None
            if is_regression:
                try:
                    target_value = float(label_str)
                except (ValueError, TypeError):
                    pass
            examples.append(
                SupervisedExample(
                    vector=[float(row[column]) for column in self.input_columns],
                    label=label_str,
                    row_index=row_index,
                    row_hash=_row_hash(row_index, row),
                    target_value=target_value,
                )
            )
        return examples


def dataset_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return "data_" + digest


def _fingerprint_payload(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _row_hash(row_index: int, row: dict[str, Any]) -> str:
    payload = {"row_index": row_index, "row": row}
    return _fingerprint_payload("row", payload)


# ---------------------------------------------------------------------------
# BIBLIOTECA_PROYECTOS_INTELIGENTES — normalización de CSV real de entrada
# ---------------------------------------------------------------------------
# Sugerencias de la autoauditoría de C1 (analyze_dataset_csv): un CSV REAL
# (subido a mano, exportado de Excel) llega con variaciones que un CSV
# generado internamente (sintético) nunca tiene. `normalize_csv_text` es el
# punto ÚNICO por el que debe pasar cualquier `csv_text` que entra desde
# FUERA del producto — se llama en los 3 sitios donde eso ocurre
# (`_validate_training_csv`, `_run_playground_training`,
# `_submit_training_job` en playground.py, y `analyze_dataset_csv` de C1) —
# para que las tres cosas (validar, entrenar, analizar) vean SIEMPRE el
# mismo texto normalizado, nunca tres normalizaciones distintas.

def strip_csv_bom(csv_text: str) -> str:
    """Quita el BOM UTF-8 (``\\ufeff``) que antepone Excel al exportar CSV.

    Sin esto, la primera columna se llama ``"\\ufefffecha"`` y nada casa
    aguas abajo (cabecera esperada, esquema, target) — el fallo es opaco:
    "falta la columna fecha" con un CSV que, a simple vista, la tiene.
    """
    return csv_text.removeprefix("﻿")


def normalize_csv_delimiter(csv_text: str) -> str:
    """Si la cabecera usa ``;`` (Excel europeo) y no ``,``, reescribe a
    coma — el resto del producto (parser de columnas, `TrainingVerifier`,
    `analyze_dataset_csv`) exige coma.

    Señal deliberadamente conservadora (cero comas Y al menos un `;` en la
    CABECERA cruda, antes de parsear nada): un CSV de verdad delimitado por
    comas casi nunca tiene cero comas en la cabecera con 2+ columnas, así
    que el riesgo de falso positivo es mínimo. La reescritura usa el módulo
    `csv` (no `str.replace`) para que un valor que YA contenga una coma
    quede correctamente entrecomillado en la salida.

    Límite documentado (NO resuelto aquí — llevaría a normalizar también el
    separador DECIMAL, un problema distinto): un CSV europeo que además usa
    la coma como separador decimal (``"12,5"``) sigue sin tipar numérico
    tras esta reescritura — la celda queda entrecomillada intacta
    (``"12,5"``) y `analyze_dataset_csv` la verá como texto. El usuario
    puede corregir el tipo en el editor (invariante 8).
    """
    if not csv_text.strip():
        return csv_text
    header_line = csv_text.splitlines()[0]
    if "," in header_line or ";" not in header_line:
        return csv_text
    # Auditoría de las sugerencias (H-B): la reescritura lee de un StringIO,
    # NUNCA de `splitlines()` — partir en líneas ANTES de parsear rompe los
    # campos entrecomillados que contienen saltos de línea, y `csv.reader`
    # los re-unía SIN el salto ("linea1\nlinea2" → "linea1linea2": corrupción
    # silenciosa de datos). Con el stream intacto, el valor multilínea
    # sobrevive el round-trip entero (re-entrecomillado en la salida).
    reader = csv.reader(io.StringIO(csv_text), delimiter=";")
    out = io.StringIO()
    writer = csv.writer(out, delimiter=",", lineterminator="\n")
    writer.writerows(reader)
    return out.getvalue()


def normalize_csv_text(csv_text: str) -> str:
    """BOM + delimitador — la única función que un entry point externo debe
    llamar (orden fijo: BOM antes que delimitador, para que el heurístico de
    la cabecera no se confunda con el BOM pegado al primer nombre)."""
    return normalize_csv_delimiter(strip_csv_bom(csv_text))
