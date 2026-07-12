# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""ModelRegistry — append-only local versioned model registry."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrixai.registry.entry_hash import compute_entry_hash
from matrixai.registry.layout import RegistryLayout
from matrixai import __version__ as _MATRIXAI_VERSION
from matrixai.registry.schema import MATRIXAI_REGISTRY_SCHEMA_VERSION, RegistryEntry
from matrixai.registry.signing import (
    build_signature_record,
    get_signing_key,
    verify_entry_signature,
)


class ModelRegistryError(Exception):
    pass


class DuplicateEntryError(ModelRegistryError):
    pass


class EntryNotFoundError(ModelRegistryError):
    pass


class VerificationError(ModelRegistryError):
    pass


class ModelRegistry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.layout = RegistryLayout(self.path)
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        self.layout.entries_dir.mkdir(parents=True, exist_ok=True)
        self.layout.tags_dir.mkdir(parents=True, exist_ok=True)
        if not self.layout.index_path.exists():
            self._save_index([])

    # ── push ──────────────────────────────────────────────────────────────────

    def push(self, entry: RegistryEntry) -> None:
        if not entry.evaluation_report_hash:
            raise ModelRegistryError(
                f"Cannot push {entry.name}@{entry.version}: evaluation_report_hash is required"
            )
        index = self._load_index()
        for item in index:
            if item["name"] == entry.name and item["version"] == entry.version:
                raise DuplicateEntryError(
                    f"{entry.name}@{entry.version} already exists (registry is append-only)"
                )
        entry_dir = self.layout.entry_dir(entry.name, entry.version)
        entry_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = self.layout.entry_file(entry.name, entry.version, "manifest")
        manifest_path.write_text(json.dumps(entry.to_manifest(), indent=2))

        signing_key = get_signing_key(registry_path=self.path)
        if signing_key:
            sig_record = build_signature_record(entry.entry_hash, signing_key)
            sig_path = self.layout.entry_file(entry.name, entry.version, "signature")
            sig_path.write_text(json.dumps(sig_record, indent=2))

        index.append({
            "name": entry.name,
            "version": entry.version,
            "entry_hash": entry.entry_hash,
        })
        self._save_index(index)

    # ── get ───────────────────────────────────────────────────────────────────

    def get(self, name: str, version: str) -> RegistryEntry:
        resolved = self._resolve_version(name, version)
        manifest_path = self.layout.entry_file(name, resolved, "manifest")
        if not manifest_path.exists():
            raise EntryNotFoundError(f"{name}@{version} not found in registry")
        data = json.loads(manifest_path.read_text())
        return RegistryEntry.from_manifest(data)

    # ── list ──────────────────────────────────────────────────────────────────

    def list(self, filters: dict[str, Any] | None = None) -> list[RegistryEntry]:
        results = []
        for item in self._load_index():
            if filters and "name" in filters and filters["name"] != item["name"]:
                continue
            try:
                results.append(self.get(item["name"], item["version"]))
            except EntryNotFoundError:
                pass
        return results

    # ── tag ───────────────────────────────────────────────────────────────────

    def tag(self, name: str, version: str, tag_name: str) -> None:
        self.get(name, version)  # validates entry exists
        tag_path = self.layout.tag_path(name, tag_name)
        tag_path.parent.mkdir(parents=True, exist_ok=True)
        tag_path.write_text(json.dumps({"name": name, "version": version}))

    # ── verify ────────────────────────────────────────────────────────────────

    def verify(self, name: str, version: str) -> bool:
        from matrixai.registry.entry_hash import sha256_bytes

        resolved = self._resolve_version(name, version)
        entry = self.get(name, resolved)
        expected_hash = compute_entry_hash(
            name=entry.name,
            version=entry.version,
            model_hash=entry.model_hash,
            parameter_schema_hash=entry.parameter_schema_hash,
            parameter_set_id=entry.parameter_set_id,
            training_trace_hash=entry.training_trace_hash,
            evaluation_report_hash=entry.evaluation_report_hash,
            matrixai_version=entry.matrixai_version,
            params_content_hash=entry.params_content_hash,
        )
        if entry.entry_hash != expected_hash:
            raise VerificationError(
                f"entry_hash mismatch for {name}@{resolved}: stored {entry.entry_hash!r}"
            )

        # Re-hash stored artifact files and compare against manifest values
        _null_hash = "sha256:" + "0" * 64
        entry_dir = self.layout.entry_dir(name, resolved)
        for filename, stored_hash in [
            ("model.mxai", entry.model_hash if entry.model_hash != _null_hash else ""),
            ("params.json", entry.params_content_hash),
            ("training_trace.json", entry.training_trace_hash),
            ("evaluation_report.json", entry.evaluation_report_hash),
        ]:
            if not stored_hash:
                continue
            file_path = entry_dir / filename
            if not file_path.exists():
                continue
            actual_hash = sha256_bytes(file_path.read_bytes())
            if actual_hash != stored_hash:
                raise VerificationError(
                    f"{filename} content hash mismatch for {name}@{resolved}"
                )

        # TRANSFORMER C6: PESOS_GRANDES entries carry weights.mxw instead of
        # params.json — re-hash it in streaming (mxw_body_content_hash) rather
        # than sha256_bytes(full file), which would defeat the whole point of
        # not materializing a multi-GiB model in RAM to verify it.
        mxw_path = entry_dir / "weights.mxw"
        if entry.params_content_hash and mxw_path.exists():
            from matrixai.parameters.binary_store import mxw_body_content_hash
            actual_hash = "sha256:" + mxw_body_content_hash(mxw_path)
            if actual_hash != entry.params_content_hash:
                raise VerificationError(
                    f"weights.mxw content hash mismatch for {name}@{resolved}"
                )

        sig_path = self.layout.entry_file(name, resolved, "signature")
        if sig_path.exists():
            signing_key = get_signing_key(registry_path=self.path)
            if signing_key:
                sig_record = json.loads(sig_path.read_text())
                signature = sig_record.get("signature", "")
                if not verify_entry_signature(entry.entry_hash, signature, signing_key):
                    raise VerificationError(
                        f"signature verification failed for {name}@{resolved}"
                    )

        # Version compatibility check — warn when major versions differ.
        try:
            entry_major = int(entry.matrixai_version.split(".")[0])
            current_major = int(_MATRIXAI_VERSION.split(".")[0])
            if entry_major != current_major:
                import warnings
                warnings.warn(
                    f"{name}@{resolved} was created with matrixai_version "
                    f"'{entry.matrixai_version}'; running version is "
                    f"'{_MATRIXAI_VERSION}'. "
                    "Major version mismatch — see VERSIONING.md for compatibility guarantees.",
                    UserWarning,
                    stacklevel=2,
                )
        except (ValueError, IndexError, AttributeError):
            pass

        return True

    # ── push_run_dir ──────────────────────────────────────────────────────────

    def push_run_dir(
        self,
        run_dir: Path | str,
        name: str,
        version: str,
        *,
        interpretability_level: str = "full",
        input_type: dict | None = None,
        output_type: dict | None = None,
        metrics: dict | None = None,
    ) -> RegistryEntry:
        """Register a model from a training run directory.

        Expects evaluation_report.json (required) plus optional model.mxai,
        a JSON parameter set (params.json / params.best.json /
        parameter_set.json) OR weights.mxw (TRANSFORMER C6 / PESOS_GRANDES:
        weights materialized above the threshold), and training_trace.json.
        Artifacts are copied into the registry entry directory and their
        content is hashed to build the immutable entry_hash chain.
        """
        from matrixai.registry.entry_hash import sha256_bytes

        run_dir = Path(run_dir)

        # Required: evaluation report
        eval_path = run_dir / "evaluation_report.json"
        if not eval_path.exists():
            raise ModelRegistryError(
                f"push_run_dir: evaluation_report.json not found in {run_dir}"
            )
        evaluation_report_hash = sha256_bytes(eval_path.read_bytes())

        # Optional: model source
        model_path: Path | None = next(run_dir.glob("*.mxai"), None)
        model_hash = sha256_bytes(model_path.read_bytes()) if model_path else "sha256:" + "0" * 64

        # Optional: parameter set (prefer params.best.json, fallback params.json,
        # then parameter_set.json — TRANSFORMER C6: dense_trainer.py and
        # transformer_trainer.py both write this modern name via
        # write_parameter_set, same keys as params.best.json).
        params_path: Path | None = None
        for candidate in ("params.best.json", "params.json", "parameter_set.json"):
            if (run_dir / candidate).exists():
                params_path = run_dir / candidate
                break
        # TRANSFORMER C6 / PESOS_GRANDES: above the materialization threshold
        # the trainer persists weights.mxw instead (no JSON parameter set at
        # all) — its header already carries parameter_schema_hash and a
        # content_hash of the weight bytes, computed once at write time; the
        # registro P21 hash must cover those weights too, not fall back to
        # the null placeholder.
        mxw_path = run_dir / "weights.mxw"
        if params_path:
            ps_raw = json.loads(params_path.read_text())
            parameter_set_id = str(ps_raw.get("parameter_set_id", f"{name}_{version}_ps"))
            parameter_schema_hash = str(ps_raw.get("parameter_schema_hash", "sha256:" + "0" * 64))
            ps_metrics: dict = dict(ps_raw.get("metrics", {}))
            params_content_hash = sha256_bytes(params_path.read_bytes())
        elif mxw_path.exists():
            from matrixai.parameters.binary_store import read_mxw_header
            header = read_mxw_header(mxw_path)
            parameter_set_id = f"{name}_{version}_ps"
            parameter_schema_hash = str(header.get("parameter_schema_hash", "sha256:" + "0" * 64))
            ps_metrics = {}
            params_content_hash = "sha256:" + str(header.get("content_hash", "0" * 64))
        else:
            parameter_set_id = f"{name}_{version}_ps"
            parameter_schema_hash = "sha256:" + "0" * 64
            ps_metrics = {}
            params_content_hash = ""

        # Optional: training trace
        trace_path = run_dir / "training_trace.json"
        training_trace_hash = sha256_bytes(trace_path.read_bytes()) if trace_path.exists() else ""

        # Build entry
        eh = compute_entry_hash(
            name=name, version=version,
            model_hash=model_hash,
            parameter_schema_hash=parameter_schema_hash,
            parameter_set_id=parameter_set_id,
            training_trace_hash=training_trace_hash,
            evaluation_report_hash=evaluation_report_hash,
            matrixai_version=_MATRIXAI_VERSION,
            params_content_hash=params_content_hash,
        )
        entry = RegistryEntry(
            name=name, version=version, entry_hash=eh,
            model_hash=model_hash,
            parameter_schema_hash=parameter_schema_hash,
            parameter_set_id=parameter_set_id,
            input_type=input_type or {},
            output_type=output_type or {},
            metrics=metrics if metrics is not None else ps_metrics,
            matrixai_version=_MATRIXAI_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            training_dataset_fingerprint="",
            interpretability_level=interpretability_level,
            training_trace_hash=training_trace_hash,
            evaluation_report_hash=evaluation_report_hash,
            params_content_hash=params_content_hash,
        )

        # push() writes manifest + signature + updates index
        self.push(entry)

        # Copy artifact files into the entry directory
        entry_dir = self.layout.entry_dir(name, version)
        shutil.copy2(eval_path, entry_dir / "evaluation_report.json")
        if model_path:
            shutil.copy2(model_path, entry_dir / "model.mxai")
        if params_path:
            shutil.copy2(params_path, entry_dir / "params.json")
        elif mxw_path.exists():
            shutil.copy2(mxw_path, entry_dir / "weights.mxw")
        if trace_path.exists():
            shutil.copy2(trace_path, entry_dir / "training_trace.json")

        return entry

    # ── pull ──────────────────────────────────────────────────────────────────

    def pull(self, name: str, version: str, target_registry: "ModelRegistry") -> None:
        resolved = self._resolve_version(name, version)
        entry = self.get(name, resolved)
        target_registry.push(entry)
        # Copy content artifacts. "manifest" is already written by push().
        # "signature" is excluded: target.push() re-signs with the target's own key.
        for key in ("model", "params", "weights_mxw", "training_trace", "evaluation_report"):
            src = self.layout.entry_file(name, resolved, key)
            if src.exists():
                dst = target_registry.layout.entry_file(entry.name, entry.version, key)
                shutil.copy2(src, dst)

    # ── internal ──────────────────────────────────────────────────────────────

    def _resolve_version(self, name: str, version: str) -> str:
        tag_path = self.layout.tag_path(name, version)
        if tag_path.exists():
            data = json.loads(tag_path.read_text())
            return data["version"]
        return version

    def _load_index(self) -> list[dict]:
        if not self.layout.index_path.exists():
            return []
        data = json.loads(self.layout.index_path.read_text())
        return data.get("entries", [])

    def _save_index(self, entries: list[dict]) -> None:
        self.layout.index_path.write_text(
            json.dumps(
                {"version": MATRIXAI_REGISTRY_SCHEMA_VERSION, "entries": entries},
                indent=2,
            )
        )
