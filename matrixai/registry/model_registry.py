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

        # Re-hash stored artifact files and compare against manifest values.
        #
        # Auditoría C6 ronda 2 [ALTA-2]: para las entradas CON CUSTODIA de
        # pesos (params_content_hash no vacío — las que push_run_dir crea
        # desde C6), un hash declarado cuyo artefacto NO está almacenado es
        # un error, no un "continue": borrar el fichero del registro dejaba
        # verify() en verde. Las entradas legacy (push() de metadata pura,
        # sin params_content_hash) conservan su semántica histórica
        # (verificar-si-presente).
        _null_hash = "sha256:" + "0" * 64
        entry_dir = self.layout.entry_dir(name, resolved)
        custodial = bool(entry.params_content_hash)
        mxw_path = entry_dir / "weights.mxw"
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
                # params.json y weights.mxw son alternativas: el hash de
                # params_content_hash lo cubre el bloque mxw de abajo.
                if filename == "params.json":
                    if mxw_path.exists():
                        continue
                    if custodial:
                        raise VerificationError(
                            f"weights are declared by {name}@{resolved} "
                            f"(params_content_hash {stored_hash[:20]}…) but "
                            "neither params.json nor weights.mxw is stored in "
                            "the registry entry — artifact deleted or never "
                            "stored"
                        )
                    continue
                if custodial:
                    raise VerificationError(
                        f"{filename} is declared by {name}@{resolved} "
                        f"(hash {stored_hash[:20]}…) but is missing from the "
                        "registry entry — artifact deleted or never stored"
                    )
                continue
            actual_hash = sha256_bytes(file_path.read_bytes())
            if actual_hash != stored_hash:
                raise VerificationError(
                    f"{filename} content hash mismatch for {name}@{resolved}"
                )

        # TRANSFORMER C6: PESOS_GRANDES entries carry weights.mxw instead of
        # params.json — re-hash it in streaming rather than
        # sha256_bytes(full file in RAM). Ronda 2 [ALTA-3]: el hash cubre el
        # fichero COMPLETO (la cabecera manda sobre la interpretación del
        # cuerpo) y la semántica de la cabecera se contrasta con la entrada
        # (schema_hash) — un rename de path en la cabecera ya no pasa.
        if entry.params_content_hash and mxw_path.exists():
            from matrixai.parameters.binary_store import (
                mxw_file_content_hash,
                validate_mxw_file,
                MxwError,
            )
            actual_hash = "sha256:" + mxw_file_content_hash(mxw_path)
            if actual_hash != entry.params_content_hash:
                raise VerificationError(
                    f"weights.mxw content hash mismatch for {name}@{resolved}"
                )
            if entry.model_hash == _null_hash:
                raise VerificationError(
                    f"weights.mxw is custodial but model_hash is null for "
                    f"{name}@{resolved}"
                )
            model_path = entry_dir / "model.mxai"
            if not model_path.exists():
                raise VerificationError(
                    f"weights.mxw is custodial but model.mxai is missing for "
                    f"{name}@{resolved}"
                )
            # Reauditoría C6 ronda 3 [ALTA-3]: el hash de custodia detecta
            # cambios posteriores al push, pero un fichero que YA llegaba con
            # body != header.content_hash quedaba registrado y verify=True.
            try:
                header, _body_start = validate_mxw_file(mxw_path)
            except MxwError as exc:
                raise VerificationError(
                    f"weights.mxw is internally invalid for {name}@{resolved}: {exc}"
                ) from exc
            header_schema = str(header.get("parameter_schema_hash", ""))
            header_model = str(header.get("model_hash", ""))
            if not header_schema:
                raise VerificationError(
                    f"weights.mxw header has no parameter_schema_hash for {name}@{resolved}"
                )
            if header_schema != entry.parameter_schema_hash:
                raise VerificationError(
                    f"weights.mxw header parameter_schema_hash "
                    f"({header_schema!r}) does not match the registry entry "
                    f"({entry.parameter_schema_hash!r}) for {name}@{resolved}"
                )
            if not header_model:
                raise VerificationError(
                    f"weights.mxw header has no model_hash for {name}@{resolved}"
                )
            from matrixai.parameters.store import program_hash as _program_hash
            from matrixai.parser import parse_file as _parse_file
            try:
                stored_program_hash = _program_hash(_parse_file(model_path))
            except Exception as exc:  # noqa: BLE001
                raise VerificationError(
                    f"stored model.mxai cannot be parsed for {name}@{resolved}: {exc}"
                ) from exc
            if header_model != stored_program_hash:
                raise VerificationError(
                    f"weights.mxw header model_hash ({header_model!r}) does not "
                    f"match stored model.mxai ({stored_program_hash!r}) for "
                    f"{name}@{resolved}"
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
        weights_model_hash = ""
        modern_weights = bool(
            (params_path is not None and params_path.name == "parameter_set.json")
            or (params_path is None and mxw_path.exists())
        )
        if params_path:
            try:
                ps_raw = json.loads(params_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ModelRegistryError(
                    f"push_run_dir: invalid {params_path.name}: {exc}"
                ) from exc
            if params_path.name == "parameter_set.json":
                from matrixai.parameters.store import ParameterSet
                try:
                    modern_ps = ParameterSet.from_dict(ps_raw)
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise ModelRegistryError(
                        f"push_run_dir: invalid parameter_set.json: {exc}"
                    ) from exc
                parameter_set_id = modern_ps.parameter_set_id
                parameter_schema_hash = modern_ps.parameter_schema_hash
                ps_metrics = dict(modern_ps.metrics)
                weights_model_hash = modern_ps.model_hash
            else:
                parameter_set_id = str(
                    ps_raw.get("parameter_set_id", f"{name}_{version}_ps")
                )
                parameter_schema_hash = str(
                    ps_raw.get("parameter_schema_hash", "sha256:" + "0" * 64)
                )
                ps_metrics = dict(ps_raw.get("metrics", {}))
            params_content_hash = sha256_bytes(params_path.read_bytes())
        elif mxw_path.exists():
            from matrixai.parameters.binary_store import (
                mxw_file_content_hash,
                validate_mxw_file,
                MxwError,
            )
            try:
                header, _body_start = validate_mxw_file(mxw_path)
            except MxwError as exc:
                raise ModelRegistryError(
                    f"push_run_dir: weights.mxw is invalid or corrupt: {exc}"
                ) from exc
            parameter_set_id = f"{name}_{version}_ps"
            parameter_schema_hash = str(header.get("parameter_schema_hash", "sha256:" + "0" * 64))
            ps_metrics = {}
            # Auditoría C6 ronda 2 [ALTA-3]: hash del fichero COMPLETO
            # (cabecera incluida) — el content_hash de la cabecera solo cubre
            # el cuerpo, y la cabecera (paths/shapes/offsets) determina cómo
            # se interpretan esos bytes.
            params_content_hash = "sha256:" + mxw_file_content_hash(mxw_path)
            weights_model_hash = str(header.get("model_hash", ""))

        # Auditoría C6 [MEDIA-2] + ronda 2 [ALTA-4]: los pesos deben pertenecer
        # al modelo del run. Los hashes por-fichero de siempre detectan tamper
        # de CADA artefacto, pero nada ligaba pesos↔modelo. Solo aplica a los
        # caminos que C6 abre (parameter_set.json / weights.mxw) — las
        # entradas params.best.json/params.json preexistentes conservan su
        # semántica histórica intacta. Ronda 2: sin `.mxai` en el run el
        # cross-check se SALTABA en silencio (el ciclo CLI real no dejaba el
        # modelo en el run dir — el trainer ahora lo copia) → ahora el .mxai
        # es OBLIGATORIO en estos caminos, no opcional.
        # Reauditoría C6 ronda 3 [ALTA-2]: la obligatoriedad depende del TIPO
        # de artefacto moderno, no de que el hash aportado sea truthy. Antes
        # bastaba borrar model_hash del JSON/header para saltarse también el
        # model.mxai y registrar una entrada que verify() aceptaba.
        _null_hash = "sha256:" + "0" * 64
        if modern_weights:
            if (
                not weights_model_hash.strip()
                or weights_model_hash in (_null_hash, "None")
            ):
                raise ModelRegistryError(
                    "push_run_dir: a modern entry (parameter_set.json / "
                    "weights.mxw) requires a non-empty model_hash binding the "
                    "trained weights to model.mxai"
                )
            if (
                not parameter_schema_hash.strip()
                or parameter_schema_hash in (_null_hash, "None")
            ):
                raise ModelRegistryError(
                    "push_run_dir: a modern entry (parameter_set.json / "
                    "weights.mxw) requires a non-empty parameter_schema_hash"
                )
            if model_path is None:
                raise ModelRegistryError(
                    "push_run_dir: a modern entry (parameter_set.json / "
                    "weights.mxw) requires the trained model's .mxai in the "
                    "run directory to cross-check weights against model — "
                    "the trainer copies it as model.mxai since TRANSFORMER C6"
                )
            from matrixai.parameters.store import program_hash as _program_hash
            from matrixai.parser import parse_file as _parse_file
            try:
                program_digest = _program_hash(_parse_file(model_path))
            except Exception as exc:  # noqa: BLE001 — un .mxai ilegible en este camino es un run corrupto
                raise ModelRegistryError(
                    f"push_run_dir: cannot parse {model_path.name} to cross-check "
                    f"the trained weights' model_hash: {exc}"
                ) from exc
            if weights_model_hash != program_digest:
                raise ModelRegistryError(
                    f"push_run_dir: trained weights belong to a different model "
                    f"(weights declare model_hash {weights_model_hash!r} but "
                    f"{model_path.name} hashes to {program_digest!r})"
                )
        if not (params_path or mxw_path.exists()):
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
