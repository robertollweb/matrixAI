#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Standalone inference wrapper for a MatrixAI downloadable model.

This file ships inside an exported MatrixAI bundle as ``predict.py``. It has
**zero dependency on MatrixAI or the Studio** — only ``numpy`` and
``onnxruntime``. Feed RAW, human-readable values; the normalization, categorical
encoding and label mapping that the model was trained with are applied here.

    from predict import MatrixAIModel
    model = MatrixAIModel()                       # loads inference_spec.json next to this file
    model.predict({"edad": 65, "especialidad": "UCI"})
    # -> {"NO": 0.23, "SI": 0.77}

    python predict.py --input example_input.json

The contract between this script and the model is ``inference_spec.json``:
For tabular models, ``input_order`` is the exact float32 column order and
``fields`` says how raw values map onto it.  Sequence models instead declare a
``token_ids`` input and an optional padding mask.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

_TRUE = {"true", "1", "yes", "si", "sí", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


class MatrixAIModelError(Exception):
    """Raised on a malformed input record or a spec/model mismatch."""


class MatrixAIModel:
    """Load a MatrixAI bundle and run predictions from raw records."""

    def __init__(self, spec_path: str | Path | None = None, *, check_hash: bool = True) -> None:
        here = Path(__file__).resolve().parent
        spec_path = Path(spec_path) if spec_path is not None else here / "inference_spec.json"
        if not spec_path.is_absolute():
            spec_path = (here / spec_path).resolve()
        if not spec_path.exists():
            raise MatrixAIModelError(f"inference_spec.json not found at {spec_path}")
        self.spec: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
        self.input_order: list[str] = list(self.spec["input_order"])
        self.input_index = {col: i for i, col in enumerate(self.input_order)}
        self.fields: dict[str, Any] = self.spec.get("fields", {})
        self.output: dict[str, Any] = self.spec.get("output", {})
        sequence_inputs = {
            name: entry for name, entry in self.spec.get("input", {}).items()
            if entry.get("encoding") == "token_ids"
        }
        if len(sequence_inputs) > 1:
            raise MatrixAIModelError("predict.py supports exactly one token_ids input.")
        self.sequence_name: str | None = next(iter(sequence_inputs), None)
        self.sequence_entry: dict[str, Any] | None = (
            sequence_inputs.get(self.sequence_name) if self.sequence_name else None
        )
        self.input_name: str = self.spec.get("input_name") or self.spec.get("onnx_input")
        if not self.input_name:
            raise MatrixAIModelError("inference_spec.json does not declare input_name.")
        self.mask_input: str | None = self.spec.get("mask_input")
        if self.sequence_name and not self.mask_input:
            raise MatrixAIModelError("token_ids spec does not declare mask_input.")

        onnx_path = spec_path.parent / self.spec.get("onnx_file", "model.onnx")
        if not onnx_path.exists():
            raise MatrixAIModelError(f"ONNX model not found at {onnx_path}")
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if check_hash:
            self._verify_hash()

    # -- public API --------------------------------------------------------

    def predict(self, record: Any, *, return_meta: bool = False):
        if self.sequence_name:
            ids, mask, meta = self._encode_sequence(record)
            raw = self.session.run(None, {
                self.input_name: np.asarray([ids], dtype=np.int64),
                self.mask_input: np.asarray([mask], dtype=np.float32),
            })[0]
            result = self._decode(np.asarray(raw)[0])
            return (result, meta) if return_meta else result
        vector, meta = self._encode(record)
        x = np.asarray([vector], dtype=np.float32)
        raw = self.session.run(None, {self.input_name: x})[0]
        result = self._decode(np.asarray(raw)[0])
        return (result, meta) if return_meta else result

    def predict_batch(self, records: list[Any], *, return_meta: bool = False):
        if self.sequence_name:
            encoded = [self._encode_sequence(r) for r in records]
            ids = np.asarray([row[0] for row in encoded], dtype=np.int64)
            masks = np.asarray([row[1] for row in encoded], dtype=np.float32)
            raw = self.session.run(None, {
                self.input_name: ids,
                self.mask_input: masks,
            })[0]
            raw = np.asarray(raw)
            results = [self._decode(raw[i]) for i in range(len(records))]
            if return_meta:
                return results, [row[2] for row in encoded]
            return results
        encoded = [self._encode(r) for r in records]
        x = np.asarray([vec for vec, _ in encoded], dtype=np.float32)
        raw = self.session.run(None, {self.input_name: x})[0]
        raw = np.asarray(raw)
        results = [self._decode(raw[i]) for i in range(len(records))]
        if return_meta:
            return results, [meta for _, meta in encoded]
        return results

    # -- encoding ----------------------------------------------------------

    def _encode_sequence(self, record: Any) -> tuple[list[int], list[float], dict[str, Any]]:
        """Validate one token-id row and construct its ONNX padding mask."""
        assert self.sequence_name is not None and self.sequence_entry is not None
        meta: dict[str, Any] = {"spec_version": self.spec.get("spec_version"),
                                "warnings": [], "clipped": []}
        if isinstance(record, list):
            raw_ids = record
            raw_mask = None
        elif isinstance(record, dict):
            if self.sequence_name not in record:
                raise MatrixAIModelError(
                    f"Missing required sequence field {self.sequence_name!r}."
                )
            raw_ids = record[self.sequence_name]
            raw_mask = record.get(self.mask_input)
            allowed = {self.sequence_name, self.mask_input}
            unknown = sorted(k for k in record if k not in allowed)
            if unknown:
                meta["warnings"].append(f"ignored unknown fields: {unknown}")
        else:
            raise MatrixAIModelError(
                "Token input must be a list of integers or an object containing that list."
            )

        length = int(self.sequence_entry["length"])
        vocab_size = int(self.sequence_entry["vocab_size"])
        if not isinstance(raw_ids, list) or len(raw_ids) != length:
            got = len(raw_ids) if isinstance(raw_ids, list) else type(raw_ids).__name__
            raise MatrixAIModelError(
                f"Field {self.sequence_name!r}: expected a list of {length} token ids, got {got}."
            )
        ids: list[int] = []
        for i, value in enumerate(raw_ids):
            if isinstance(value, bool) or not isinstance(value, int):
                raise MatrixAIModelError(
                    f"Field {self.sequence_name!r}[{i}]: expected an integer token id, got {value!r}."
                )
            if not 0 <= value < vocab_size:
                raise MatrixAIModelError(
                    f"Field {self.sequence_name!r}[{i}]: token id {value} out of range "
                    f"[0, {vocab_size - 1}]."
                )
            ids.append(value)

        if raw_mask is None:
            mask = [1.0] * length
        else:
            if not isinstance(raw_mask, list) or len(raw_mask) != length:
                got = len(raw_mask) if isinstance(raw_mask, list) else type(raw_mask).__name__
                raise MatrixAIModelError(
                    f"Field {self.mask_input!r}: expected a list of {length} mask values, got {got}."
                )
            mask = []
            for i, value in enumerate(raw_mask):
                if isinstance(value, bool):
                    mask.append(1.0 if value else 0.0)
                elif isinstance(value, (int, float)) and value in (0, 1):
                    mask.append(float(value))
                else:
                    raise MatrixAIModelError(
                        f"Field {self.mask_input!r}[{i}]: mask values must be boolean or 0/1, "
                        f"got {value!r}."
                    )
        if not any(mask):
            raise MatrixAIModelError("Mask must keep at least one real token.")
        if self.spec.get("pool_kind") == "cls" and mask[0] != 1.0:
            raise MatrixAIModelError("POOL cls requires mask[0] = 1 (a real token).")
        return ids, mask, meta

    def _encode(self, record: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
        if not isinstance(record, dict):
            raise MatrixAIModelError("Input record must be a dict of raw field values.")
        meta: dict[str, Any] = {"spec_version": self.spec.get("spec_version"),
                                "warnings": [], "clipped": []}
        unknown = [k for k in record if k not in self.fields]
        if unknown:
            meta["warnings"].append(f"ignored unknown fields: {sorted(unknown)}")

        vector = [0.0] * len(self.input_order)
        for field, entry in self.fields.items():
            enc = entry["encoding"]
            if enc in ("scalar", "scalar01"):
                self._encode_scalar(field, entry, record, vector, meta)
            elif enc == "one_hot":
                self._encode_one_hot(field, entry, record, vector)
            elif enc == "embedding_index":
                self._encode_embedding(field, entry, record, vector)
            else:
                raise MatrixAIModelError(f"Field {field!r}: unknown encoding {enc!r}.")
        return vector, meta

    def _require(self, field: str, record: dict[str, Any]) -> Any:
        if field not in record:
            raise MatrixAIModelError(f"Missing required field {field!r}.")
        return record[field]

    def _encode_scalar(self, field, entry, record, vector, meta) -> None:
        value = self._require(field, record)
        field_type = entry.get("type")
        if field_type == "boolean":
            number = _parse_bool(field, value)
        elif field_type == "integer":
            number = float(_parse_int(field, value))  # reject non-integers (3.7 -> error)
        else:
            number = _parse_number(field, value)
        if entry["encoding"] == "scalar":
            lo, hi = entry["range"]
            span = (hi - lo) or 1.0
            normalized = (number - lo) / span
        else:  # scalar01
            normalized = number
        clipped = min(1.0, max(0.0, normalized))
        if clipped != normalized:
            meta["clipped"].append({"field": field, "raw_value": value,
                                    "normalized_value": clipped})
        vector[self.input_index[field]] = clipped

    def _encode_one_hot(self, field, entry, record, vector) -> None:
        value = str(self._require(field, record))
        valid = {v["raw"]: v["column"] for v in entry["values"]}
        if value not in valid:
            raise MatrixAIModelError(
                f"Field {field!r}: unknown category {value!r}. "
                f"Valid categories: {sorted(valid)}."
            )
        vector[self.input_index[valid[value]]] = 1.0

    def _encode_embedding(self, field, entry, record, vector) -> None:
        value = self._require(field, record)
        column = entry.get("column", field)
        if "vocab" in entry:
            vocab = entry["vocab"]
            text = str(value)
            if text not in vocab:
                raise MatrixAIModelError(
                    f"Field {field!r}: unknown category {text!r}. "
                    f"Valid categories: {vocab}."
                )
            index = vocab.index(text)
        else:
            index = _parse_int(field, value)
            size = int(entry["vocab_size"])
            if not 0 <= index < size:
                raise MatrixAIModelError(
                    f"Field {field!r}: embedding index {index} out of range [0, {size - 1}]."
                )
        vector[self.input_index[column]] = float(index)

    # -- decoding ----------------------------------------------------------

    def _decode(self, raw: Any):
        values = np.asarray(raw, dtype=np.float64).ravel()
        kind = self.output.get("kind")
        labels = self.output.get("labels") or []
        if kind == "classification":
            return {label: float(values[i]) for i, label in enumerate(labels)}
        if kind == "binary_classification":
            p = float(values[0])
            return {labels[0]: 1.0 - p, labels[1]: p}
        if kind == "regression":
            return float(values[0])
        return {"values": [float(v) for v in values]}

    # -- integrity ---------------------------------------------------------

    def _verify_hash(self) -> None:
        meta = self.session.get_modelmeta()
        embedded = dict(getattr(meta, "custom_metadata_map", {}) or {})
        for key in ("model_hash", "parameter_schema_hash"):
            want = self.spec.get(key)
            if not want:
                continue  # the spec doesn't declare it: nothing to check against
            got = embedded.get(f"matrixai_{key}")
            if not got:
                raise MatrixAIModelError(
                    f"model.onnx does not carry the embedded {key} that "
                    f"inference_spec.json expects ({want!r}). This does not look like the "
                    "model this spec describes; predictions could be wrong. Pass "
                    "check_hash=False to override."
                )
            if want != got:
                raise MatrixAIModelError(
                    f"Spec/model mismatch on {key}: inference_spec.json says {want!r} but "
                    f"model.onnx says {got!r}. This spec does not belong to this model; "
                    "predictions would be wrong. Pass check_hash=False to override."
                )


# -- value parsers ---------------------------------------------------------

def _parse_number(field: str, value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise MatrixAIModelError(f"Field {field!r}: expected a number, got {value!r}.")
    if math.isnan(number) or math.isinf(number):
        raise MatrixAIModelError(f"Field {field!r}: value must be finite, got {value!r}.")
    return number


def _parse_int(field: str, value: Any) -> int:
    number = _parse_number(field, value)
    if not float(number).is_integer():
        raise MatrixAIModelError(f"Field {field!r}: expected an integer, got {value!r}.")
    return int(number)


def _parse_bool(field: str, value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return float(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return 1.0
    if text in _FALSE:
        return 0.0
    raise MatrixAIModelError(
        f"Field {field!r}: expected a boolean (true/false, 1/0, yes/no, si/no), got {value!r}."
    )


# -- CLI --------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MatrixAI downloadable model on raw input.")
    parser.add_argument("--input", required=True, help="JSON file: one record (object) or a list of records.")
    parser.add_argument("--spec", default=None, help="Path to inference_spec.json (default: next to this file).")
    parser.add_argument("--meta", action="store_true", help="Include inference metadata in the output.")
    parser.add_argument("--no-check-hash", action="store_true", help="Skip the spec/model hash check.")
    args = parser.parse_args(argv)

    model = MatrixAIModel(args.spec, check_hash=not args.no_check_hash)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    try:
        # For a token_ids model a JSON array of integers is one raw sequence;
        # batches are arrays of objects/arrays.  Tabular semantics stay intact.
        is_single_sequence = (
            model.sequence_name is not None
            and isinstance(payload, list)
            and all(isinstance(v, int) and not isinstance(v, bool) for v in payload)
        )
        if isinstance(payload, list) and not is_single_sequence:
            out = model.predict_batch(payload, return_meta=args.meta)
        else:
            out = model.predict(payload, return_meta=args.meta)
    except MatrixAIModelError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1
    if args.meta:
        result, meta = out
        print(json.dumps({"prediction": result, "meta": meta}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
