# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from matrixai.training.data import dataset_fingerprint


DATASET_MANIFEST_VERSION = "matrixai.dataset_manifest.v1"
SYNTHETIC_GENERATOR_VERSION = "matrixai.synthetic.v1"


@dataclass(frozen=True)
class GeneratorSpec:
    version: str
    seed: int
    mode: str
    rows: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratorSpec:
        if not isinstance(data, dict):
            raise ValueError("Dataset manifest generator must be an object")
        return cls(
            version=str(data.get("version") or SYNTHETIC_GENERATOR_VERSION),
            seed=int(data.get("seed") or 0),
            mode=str(data.get("mode") or "random"),
            rows=int(data.get("rows") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "mode": self.mode,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class DatasetManifestEntry:
    role: str
    source: str
    version: str = ""
    fingerprint: str = ""
    sha256: str = ""
    rows: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifestEntry:
        if not isinstance(data, dict):
            raise ValueError("Dataset manifest entries must be objects")
        role = str(data.get("role") or "").strip().lower()
        source = str(data.get("source") or "").strip()
        if not role:
            raise ValueError("Dataset manifest entry requires role")
        if not source:
            raise ValueError(f"Dataset manifest entry {role} requires source")
        rows = data.get("rows")
        return cls(
            role=role,
            source=source,
            version=str(data.get("version") or ""),
            fingerprint=str(data.get("fingerprint") or ""),
            sha256=str(data.get("sha256") or ""),
            rows=int(rows) if rows is not None else None,
        )

    def resolved_path(self, base_path: str | Path) -> Path:
        path = Path(self.source)
        if path.is_absolute():
            return path
        return Path(base_path) / path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["rows"] is None:
            data.pop("rows")
        if not data["version"]:
            data.pop("version")
        if not data["fingerprint"]:
            data.pop("fingerprint")
        if not data["sha256"]:
            data.pop("sha256")
        return data


@dataclass(frozen=True)
class DatasetManifestSplitPartition:
    role: str
    dataset: str
    rows: int | None = None
    fingerprint: str = ""
    row_indices: list[int] = field(default_factory=list)
    row_hashes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifestSplitPartition:
        if not isinstance(data, dict):
            raise ValueError("Dataset manifest split partitions must be objects")
        role = str(data.get("role") or "").strip().lower()
        if not role:
            raise ValueError("Dataset manifest split partition requires role")
        dataset = str(data.get("dataset") or role).strip().lower()
        if not dataset:
            raise ValueError(f"Dataset manifest split partition {role} requires dataset")
        rows = data.get("rows")
        return cls(
            role=role,
            dataset=dataset,
            rows=int(rows) if rows is not None else None,
            fingerprint=str(data.get("fingerprint") or ""),
            row_indices=_int_list(data.get("row_indices") or []),
            row_hashes=[str(item) for item in data.get("row_hashes") or []],
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "dataset": self.dataset,
        }
        if self.rows is not None:
            data["rows"] = self.rows
        if self.fingerprint:
            data["fingerprint"] = self.fingerprint
        if self.row_indices:
            data["row_indices"] = list(self.row_indices)
        if self.row_hashes:
            data["row_hashes"] = list(self.row_hashes)
        return data


@dataclass(frozen=True)
class DatasetManifestSplit:
    name: str
    version: str
    strategy: str
    partitions: list[DatasetManifestSplitPartition]
    fold: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifestSplit:
        if not isinstance(data, dict):
            raise ValueError("Dataset manifest splits must be objects")
        raw_partitions = data.get("partitions")
        if not isinstance(raw_partitions, list) or not raw_partitions:
            raise ValueError("Dataset manifest split requires a non-empty partitions list")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Dataset manifest split metadata must be an object")
        fold = data.get("fold")
        return cls(
            name=str(data.get("name") or "").strip(),
            version=str(data.get("version") or "").strip(),
            strategy=str(data.get("strategy") or "").strip(),
            partitions=[DatasetManifestSplitPartition.from_dict(item) for item in raw_partitions],
            fold=int(fold) if fold is not None else None,
            metadata=dict(metadata),
        )

    def partition_for_role(self, *roles: str) -> DatasetManifestSplitPartition:
        allowed = {role.strip().lower() for role in roles}
        for partition in self.partitions:
            if partition.role in allowed:
                return partition
        raise ValueError(f"Dataset manifest split {self.name} missing role: {' or '.join(roles)}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "strategy": self.strategy,
            "partitions": [partition.to_dict() for partition in self.partitions],
        }
        if self.fold is not None:
            data["fold"] = self.fold
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class DatasetManifest:
    version: str
    name: str
    datasets: list[DatasetManifestEntry]
    splits: list[DatasetManifestSplit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    origin: str | None = None
    generator: GeneratorSpec | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        if not isinstance(data, dict):
            raise ValueError("Dataset manifest must be a JSON object")
        raw_datasets = data.get("datasets")
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError("Dataset manifest requires a non-empty datasets list")
        raw_splits = data.get("splits") or []
        if not isinstance(raw_splits, list):
            raise ValueError("Dataset manifest splits must be a list")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Dataset manifest metadata must be an object")
        raw_origin = data.get("origin")
        origin = str(raw_origin).strip() or None if raw_origin is not None else None
        raw_generator = data.get("generator")
        generator = GeneratorSpec.from_dict(raw_generator) if isinstance(raw_generator, dict) else None
        return cls(
            version=str(data.get("version") or ""),
            name=str(data.get("name") or ""),
            datasets=[DatasetManifestEntry.from_dict(item) for item in raw_datasets],
            splits=[DatasetManifestSplit.from_dict(item) for item in raw_splits],
            metadata=dict(metadata),
            origin=origin,
            generator=generator,
        )

    def dataset_for_role(self, *roles: str) -> DatasetManifestEntry:
        allowed = {role.strip().lower() for role in roles}
        for dataset in self.datasets:
            if dataset.role in allowed:
                return dataset
        raise ValueError(f"Dataset manifest missing role: {' or '.join(roles)}")

    def split_by_name(self, name: str | None = None) -> DatasetManifestSplit | None:
        if not self.splits:
            if name:
                raise ValueError(f"Dataset manifest split not found: {name}")
            return None
        if name:
            for split in self.splits:
                if split.name == name:
                    return split
            raise ValueError(f"Dataset manifest split not found: {name}")
        if len(self.splits) == 1:
            return self.splits[0]
        raise ValueError("Dataset manifest has multiple splits; dataset_split is required")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.version,
            "name": self.name,
        }
        if self.origin:
            data["origin"] = self.origin
        if self.generator:
            data["generator"] = self.generator.to_dict()
        data["datasets"] = [dataset.to_dict() for dataset in self.datasets]
        if self.splits:
            data["splits"] = [split.to_dict() for split in self.splits]
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class DatasetManifestVerificationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    splits: list[dict[str, Any]] = field(default_factory=list)
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "datasets": list(self.datasets),
            "splits": list(self.splits),
        }
        if self.is_synthetic:
            result["is_synthetic"] = True
        return result


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset manifest is not valid JSON: {exc}") from exc
    return DatasetManifest.from_dict(data)


def verify_dataset_manifest(
    manifest: DatasetManifest,
    base_path: str | Path = ".",
    selected_split: str | None = None,
) -> DatasetManifestVerificationResult:
    base = Path(base_path)
    errors: list[str] = []
    warnings: list[str] = []
    datasets: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    is_synthetic = manifest.origin == "synthetic"
    if is_synthetic:
        warnings.append(
            "Dataset manifest origin is synthetic; data is not auditable as a production dataset"
        )
    if manifest.origin is not None and manifest.origin != "synthetic":
        errors.append(
            f"Dataset manifest origin must be 'synthetic' or absent, got {manifest.origin!r}"
        )
    if manifest.version != DATASET_MANIFEST_VERSION:
        errors.append(
            f"Dataset manifest version must be {DATASET_MANIFEST_VERSION}, got {manifest.version!r}"
        )

    if selected_split:
        try:
            manifest.split_by_name(selected_split)
        except ValueError as exc:
            errors.append(str(exc))

    roles = [dataset.role for dataset in manifest.datasets]
    if len(roles) != len(set(roles)):
        errors.append("Dataset manifest dataset roles must be unique")
    if roles.count("train") != 1:
        errors.append("Dataset manifest requires exactly one train dataset")
    if sum(1 for role in roles if role in {"evaluation", "eval", "test"}) != 1:
        errors.append("Dataset manifest requires exactly one evaluation dataset")

    for dataset in manifest.datasets:
        path = dataset.resolved_path(base)
        entry = {
            "role": dataset.role,
            "source": dataset.source,
            "resolved_path": str(path),
            "version": dataset.version,
        }
        if not path.exists():
            errors.append(f"Dataset manifest {dataset.role} source not found: {dataset.source}")
            datasets.append(entry)
            continue
        actual_sha256 = _sha256(path)
        actual_fingerprint = dataset_fingerprint(path)
        actual_rows = _csv_rows(path)
        entry.update(
            {
                "sha256": actual_sha256,
                "fingerprint": actual_fingerprint,
                "rows": actual_rows,
            }
        )
        if dataset.sha256 and dataset.sha256 != actual_sha256:
            errors.append(
                f"Dataset manifest {dataset.role} sha256 mismatch: expected {dataset.sha256}, got {actual_sha256}"
            )
        if dataset.fingerprint and dataset.fingerprint != actual_fingerprint:
            errors.append(
                f"Dataset manifest {dataset.role} fingerprint mismatch: "
                f"expected {dataset.fingerprint}, got {actual_fingerprint}"
            )
        if dataset.rows is not None and dataset.rows != actual_rows:
            errors.append(
                f"Dataset manifest {dataset.role} row count mismatch: expected {dataset.rows}, got {actual_rows}"
            )
        if not dataset.sha256 and not dataset.fingerprint:
            warnings.append(f"Dataset manifest {dataset.role} has no hash pin")
        datasets.append(entry)

    dataset_by_role = {dataset.role: dataset for dataset in manifest.datasets}
    split_names: set[str] = set()
    for split in manifest.splits:
        split_entry: dict[str, Any] = {
            "name": split.name,
            "version": split.version,
            "strategy": split.strategy,
            "partitions": [],
        }
        if split.fold is not None:
            split_entry["fold"] = split.fold
        if not split.name:
            errors.append("Dataset manifest split requires name")
        elif split.name in split_names:
            errors.append(f"Dataset manifest split name duplicated: {split.name}")
        split_names.add(split.name)
        if not split.version:
            errors.append(f"Dataset manifest split {split.name or '<unnamed>'} requires version")
        partition_roles = [partition.role for partition in split.partitions]
        if partition_roles.count("train") != 1:
            errors.append(f"Dataset manifest split {split.name or '<unnamed>'} requires exactly one train partition")
        if sum(1 for role in partition_roles if role in {"evaluation", "eval", "test"}) != 1:
            errors.append(f"Dataset manifest split {split.name or '<unnamed>'} requires exactly one evaluation partition")
        if len(partition_roles) != len(set(partition_roles)):
            errors.append(f"Dataset manifest split {split.name or '<unnamed>'} partition roles must be unique")

        for partition in split.partitions:
            partition_entry: dict[str, Any] = {
                "role": partition.role,
                "dataset": partition.dataset,
            }
            dataset = dataset_by_role.get(partition.dataset)
            if dataset is None:
                errors.append(
                    f"Dataset manifest split {split.name or '<unnamed>'} partition {partition.role} "
                    f"references unknown dataset role: {partition.dataset}"
                )
                split_entry["partitions"].append(partition_entry)
                continue
            path = dataset.resolved_path(base)
            partition_entry["source"] = dataset.source
            if not path.exists():
                split_entry["partitions"].append(partition_entry)
                continue
            try:
                metadata = _csv_partition(path, partition.row_indices)
            except ValueError as exc:
                errors.append(str(exc))
                split_entry["partitions"].append(partition_entry)
                continue
            partition_entry.update(metadata)
            if partition.rows is not None and partition.rows != metadata["rows"]:
                errors.append(
                    f"Dataset manifest split {split.name or '<unnamed>'} partition {partition.role} "
                    f"row count mismatch: expected {partition.rows}, got {metadata['rows']}"
                )
            if partition.fingerprint and partition.fingerprint != metadata["fingerprint"]:
                errors.append(
                    f"Dataset manifest split {split.name or '<unnamed>'} partition {partition.role} "
                    f"fingerprint mismatch: expected {partition.fingerprint}, got {metadata['fingerprint']}"
                )
            if partition.row_hashes and partition.row_hashes != metadata["row_hashes"]:
                errors.append(
                    f"Dataset manifest split {split.name or '<unnamed>'} partition {partition.role} row_hashes mismatch"
                )
            if not partition.fingerprint and not partition.row_hashes:
                warnings.append(
                    f"Dataset manifest split {split.name or '<unnamed>'} partition {partition.role} has no row hash pin"
                )
            split_entry["partitions"].append(partition_entry)
        splits.append(split_entry)

    return DatasetManifestVerificationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        datasets=datasets,
        splits=splits,
        is_synthetic=is_synthetic,
    )


def build_synthetic_manifest(
    name: str,
    seed: int,
    mode: str,
    rows: int,
    datasets: list[DatasetManifestEntry],
    splits: list[DatasetManifestSplit] | None = None,
) -> DatasetManifest:
    return DatasetManifest(
        version=DATASET_MANIFEST_VERSION,
        name=name,
        origin="synthetic",
        generator=GeneratorSpec(
            version=SYNTHETIC_GENERATOR_VERSION,
            seed=seed,
            mode=mode,
            rows=rows,
        ),
        datasets=datasets,
        splits=splits or [],
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows(path: Path) -> int:
    return len(_csv_row_metadata(path))


def _csv_partition(path: Path, row_indices: list[int]) -> dict[str, Any]:
    row_metadata = _csv_row_metadata(path)
    selected_indices = row_indices or [row["row_index"] for row in row_metadata]
    row_by_index = {row["row_index"]: row for row in row_metadata}
    missing = [index for index in selected_indices if index not in row_by_index]
    if missing:
        raise ValueError(f"Dataset manifest split row indices not found in {path}: {missing}")
    row_hashes = [row_by_index[index]["row_hash"] for index in selected_indices]
    return {
        "rows": len(selected_indices),
        "row_indices": list(selected_indices),
        "row_hashes": row_hashes,
        "fingerprint": _stable_fingerprint("split_part", row_hashes),
    }


def _csv_row_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "row_index": row_index,
            "row_hash": _stable_fingerprint("row", {"row_index": row_index, "row": row}),
        }
        for row_index, row in enumerate(rows, start=2)
    ]


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("Dataset manifest row_indices must be a list")
    return [int(item) for item in value]


def _stable_fingerprint(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"