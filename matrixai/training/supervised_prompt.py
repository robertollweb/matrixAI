# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import csv
import re
import json
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from matrixai.agents import PromptSupervisor
from matrixai.parameters import load_parameter_set
from matrixai.training.dataset_manifest import load_dataset_manifest, verify_dataset_manifest
from matrixai.training.generator import TrainingGenerationResult, TrainingPromptGenerator
from matrixai.training.parser import parse_training_file, parse_training_text
from matrixai.training.spec import EvaluationResult, TrainingRunResult
from matrixai.training.trainer import SupervisedEvaluator, SupervisedTrainer
from matrixai.training.verifier import TrainingVerifier


@dataclass(frozen=True)
class SupervisedPromptGenerationResult:
    prompt: str
    project: str
    output_dir: str
    model_path: str
    training_path: str
    dataset_path: str
    semantic_text: str
    mxai_text: str
    training_text: str
    dataset_template_text: str
    training_generation: TrainingGenerationResult
    supervision: dict[str, Any]
    training_verification: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = ["Supervised prompt generation: ok"]
        lines.append(f"Project: {self.project}")
        lines.append(f"Model: {self.model_path}")
        lines.append(f"Training: {self.training_path}")
        lines.append(f"Dataset template: {self.dataset_path}")
        lines.append(f"Loss: {self.training_generation.loss_type}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SupervisedPromptRunResult:
    prompt: str
    project: str
    output_dir: str
    model_path: str
    training_path: str
    train_dataset_path: str
    evaluation_dataset_path: str
    dataset_template_path: str
    run_dir: str
    evaluation_report_path: str
    manifest_path: str
    generation: SupervisedPromptGenerationResult
    training: TrainingRunResult
    evaluation: EvaluationResult
    training_verification: dict[str, Any]
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    dataset_split: dict[str, Any] = field(default_factory=dict)
    dataset_manifest_verification: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    synthetic_origin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = ["Supervised prompt run: ok"]
        lines.append(f"Project: {self.project}")
        lines.append(f"Model: {self.model_path}")
        lines.append(f"Training spec: {self.training_path}")
        lines.append(f"Run: {self.run_dir}")
        lines.append(f"Best validation loss: {self.training.best_validation_loss:.6f}")
        lines.append(f"Evaluation accuracy: {self.evaluation.accuracy:.6f}")
        lines.append(f"Evaluation report: {self.evaluation_report_path}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        return "\n".join(lines)


class SupervisedPromptGenerator:
    def generate(
        self,
        prompt: str,
        output_dir: str | Path,
        *,
        artifact_stem: str | None = None,
        dataset_name: str | None = None,
        target_name: str | None = None,
        labels: list[str] | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
    ) -> SupervisedPromptGenerationResult:
        clean_prompt = " ".join(prompt.strip().split())
        if not clean_prompt:
            raise ValueError("SupervisedPromptGenerator requires a non-empty prompt")

        supervision_report = PromptSupervisor().supervise_prompt(clean_prompt)
        if not supervision_report.accepted or not supervision_report.mxai:
            raise ValueError(_supervision_error(supervision_report.to_dict()))

        project = _project_name(supervision_report.to_dict())
        stem = _artifact_stem(artifact_stem or project or "matrixai-supervised")
        output = Path(output_dir)
        model_path = output / f"{stem}.mxai"
        training_path = output / f"{stem}.supervised.mxtrain"
        dataset_path = output / f"{stem}.train.csv"

        output.mkdir(parents=True, exist_ok=True)
        model_path.write_text(supervision_report.mxai, encoding="utf-8")

        training_generation = TrainingPromptGenerator().generate(
            clean_prompt,
            model_path,
            dataset_source=dataset_path.name,
            dataset_name=dataset_name,
            target_name=target_name,
            labels=labels,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            model_reference=model_path.name,
        )
        dataset_path.write_text(training_generation.dataset_template_text, encoding="utf-8")
        training_path.write_text(training_generation.training_text, encoding="utf-8")

        training_spec = parse_training_text(training_generation.training_text)
        verification = TrainingVerifier().verify(training_spec, base_path=output)
        if not verification.ok:
            errors = "; ".join(verification.errors)
            raise ValueError(f"Generated supervised training package is invalid: {errors}")

        supervision_payload = supervision_report.to_dict()
        supervision_payload["compiled_python"] = ""
        warnings = [*_supervision_warnings(supervision_payload), *training_generation.warnings, *verification.warnings]
        assumptions = [
            "PromptSupervisor accepted the prompt-generated semantic and mxai artifacts",
            "TrainingPromptGenerator generated the supervised contract against the written mxai model",
            "TrainingVerifier accepted the generated mxai, mxtrain and CSV template as one package",
        ]
        return SupervisedPromptGenerationResult(
            prompt=clean_prompt,
            project=project,
            output_dir=str(output),
            model_path=str(model_path),
            training_path=str(training_path),
            dataset_path=str(dataset_path),
            semantic_text=supervision_report.semantic_text,
            mxai_text=supervision_report.mxai,
            training_text=training_generation.training_text,
            dataset_template_text=training_generation.dataset_template_text,
            training_generation=training_generation,
            supervision=supervision_payload,
            training_verification=verification.to_dict(),
            assumptions=assumptions,
            warnings=warnings,
        )


class SupervisedPromptRunner:
    def run(
        self,
        prompt: str,
        output_dir: str | Path,
        *,
        train_data: str | Path | None = None,
        evaluation_data: str | Path | None = None,
        dataset_manifest: str | Path | None = None,
        dataset_split: str | None = None,
        artifact_stem: str | None = None,
        dataset_name: str | None = None,
        target_name: str | None = None,
        labels: list[str] | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        run_name: str = "run",
    ) -> SupervisedPromptRunResult:
        dataset_manifest_payload: dict[str, Any] = {}
        dataset_split_payload: dict[str, Any] = {}
        dataset_manifest_verification: dict[str, Any] = {}
        train_row_indices: list[int] | None = None
        evaluation_row_indices: list[int] | None = None
        is_synthetic = False
        if dataset_manifest is not None:
            if train_data is not None or evaluation_data is not None:
                raise ValueError("Use dataset_manifest or train_data/evaluation_data, not both")
            manifest_path = Path(dataset_manifest)
            manifest = load_dataset_manifest(manifest_path)
            selected_split = manifest.split_by_name(dataset_split)
            verification = verify_dataset_manifest(
                manifest,
                base_path=manifest_path.parent,
                selected_split=dataset_split,
            )
            dataset_manifest_payload = manifest.to_dict()
            dataset_manifest_verification = verification.to_dict()
            is_synthetic = verification.is_synthetic
            if not verification.ok:
                raise ValueError("Dataset manifest invalid: " + "; ".join(verification.errors))
            if selected_split is not None:
                dataset_split_payload = selected_split.to_dict()
                train_partition = selected_split.partition_for_role("train")
                evaluation_partition = selected_split.partition_for_role("evaluation", "eval", "test")
                train_data = manifest.dataset_for_role(train_partition.dataset).resolved_path(manifest_path.parent)
                evaluation_data = manifest.dataset_for_role(evaluation_partition.dataset).resolved_path(
                    manifest_path.parent
                )
                train_row_indices = train_partition.row_indices or None
                evaluation_row_indices = evaluation_partition.row_indices or None
            else:
                train_data = manifest.dataset_for_role("train").resolved_path(manifest_path.parent)
                evaluation_data = manifest.dataset_for_role("evaluation", "eval", "test").resolved_path(
                    manifest_path.parent
                )
        if train_data is None or evaluation_data is None:
            raise ValueError("train_data and evaluation_data are required unless dataset_manifest is provided")

        generation = SupervisedPromptGenerator().generate(
            prompt,
            output_dir,
            artifact_stem=artifact_stem,
            dataset_name=dataset_name,
            target_name=target_name,
            labels=labels,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        output = Path(generation.output_dir)
        model_path = Path(generation.model_path)
        training_path = Path(generation.training_path)
        train_dataset_path = Path(generation.dataset_path)
        stem = model_path.stem
        dataset_template_path = output / f"{stem}.template.csv"
        evaluation_dataset_path = output / f"{stem}.test.csv"
        run_dir = output / run_name
        evaluation_report_path = run_dir / "evaluation_report.json"
        manifest_path = output / "end_to_end_manifest.json"

        dataset_template_path.write_text(generation.dataset_template_text, encoding="utf-8")
        self._copy_dataset(train_data, train_dataset_path, "train_data", row_indices=train_row_indices)
        self._copy_dataset(
            evaluation_data,
            evaluation_dataset_path,
            "evaluation_data",
            row_indices=evaluation_row_indices,
        )

        training_spec = parse_training_file(training_path)
        verification = TrainingVerifier().verify(training_spec, base_path=output)
        if not verification.ok:
            errors = "; ".join(verification.errors)
            raise ValueError(f"Generated supervised training package is invalid with train_data: {errors}")

        training = SupervisedTrainer().train(
            training_spec,
            output_dir=run_dir,
            base_path=output,
            training_path=training_path,
        )
        if is_synthetic:
            trace_path = run_dir / "training_trace.json"
            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
            trace_data["dataset"]["synthetic_origin"] = True
            _write_json(trace_path, trace_data)
        parameter_set = load_parameter_set(run_dir / "params.best.json")
        evaluation = SupervisedEvaluator().evaluate(
            training_spec,
            parameter_set=parameter_set,
            data_path=evaluation_dataset_path.name,
            base_path=output,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_dict = evaluation.to_dict()
        if is_synthetic:
            eval_dict["synthetic_origin"] = True
        _write_json(evaluation_report_path, eval_dict)

        assumptions = [
            "The generated dataset template was preserved before copying the real training CSV",
            "The supplied train CSV was validated against the generated mxtrain schema before training",
            "The supplied evaluation CSV was evaluated with the best ParameterSet from the generated run",
        ]
        if dataset_split_payload:
            assumptions.append("The selected dataset split was verified before materializing train/evaluation CSVs")
        result = SupervisedPromptRunResult(
            prompt=generation.prompt,
            project=generation.project,
            output_dir=str(output),
            model_path=str(model_path),
            training_path=str(training_path),
            train_dataset_path=str(train_dataset_path),
            evaluation_dataset_path=str(evaluation_dataset_path),
            dataset_template_path=str(dataset_template_path),
            run_dir=str(run_dir),
            evaluation_report_path=str(evaluation_report_path),
            manifest_path=str(manifest_path),
            generation=generation,
            training=training,
            evaluation=evaluation,
            training_verification=verification.to_dict(),
            dataset_manifest=dataset_manifest_payload,
            dataset_split=dataset_split_payload,
            dataset_manifest_verification=dataset_manifest_verification,
            assumptions=assumptions,
            warnings=list(generation.warnings)
            + list(verification.warnings)
            + list(dataset_manifest_verification.get("warnings", [])),
            synthetic_origin=is_synthetic,
        )
        _write_json(manifest_path, result.to_dict())
        return result

    def _copy_dataset(
        self,
        source: str | Path,
        destination: Path,
        name: str,
        row_indices: list[int] | None = None,
    ) -> None:
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"{name} not found: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not row_indices:
            shutil.copyfile(source_path, destination)
            return
        with source_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            fieldnames = reader.fieldnames or []
            rows_by_index = {row_index: row for row_index, row in enumerate(reader, start=2)}
        missing = [index for index in row_indices if index not in rows_by_index]
        if missing:
            raise ValueError(f"{name} row indices not found: {missing}")
        with destination.open("w", encoding="utf-8", newline="") as destination_handle:
            writer = csv.DictWriter(destination_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row_index in row_indices:
                writer.writerow(rows_by_index[row_index])


def _project_name(supervision: dict[str, Any]) -> str:
    plan = supervision.get("plan") or {}
    if isinstance(plan, dict) and plan.get("project"):
        return str(plan["project"])
    program = supervision.get("program") or {}
    if isinstance(program, dict) and program.get("project"):
        return str(program["project"])
    return "MatrixAISupervised"


def _supervision_error(supervision: dict[str, Any]) -> str:
    errors: list[str] = []
    for check in supervision.get("checks", []):
        if not isinstance(check, dict) or check.get("ok"):
            continue
        check_errors = check.get("errors") or []
        if check_errors:
            errors.extend(f"{check.get('name', 'check')}: {error}" for error in check_errors)
        else:
            errors.append(f"{check.get('name', 'check')}: failed")
    detail = "; ".join(errors) if errors else "prompt was rejected"
    return f"Prompt supervision rejected: {detail}"


def _supervision_warnings(supervision: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for check in supervision.get("checks", []):
        if not isinstance(check, dict):
            continue
        for warning in check.get("warnings") or []:
            warnings.append(f"{check.get('name', 'check')}: {warning}")
    return warnings


def _artifact_stem(value: str) -> str:
    spaced = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", value.strip())
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", spaced)
    normalized = unicodedata.normalize("NFKD", spaced)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    stem = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()
    return stem or "matrixai-supervised"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")