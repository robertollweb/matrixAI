# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 audit fix — DenseSupervisedTrainer and DenseSupervisedEvaluator for CLI integration."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from matrixai.parameters import (
    ParameterSet,
    build_network_parameter_set,
    write_parameter_set,
)
from matrixai.parameters.store import program_hash
from matrixai.parser import parse_file
from matrixai.training.data import CSVDataAdapter, SupervisedExample, dataset_fingerprint
from matrixai.training.dense_backprop import dense_train_step, compute_loss
from matrixai.training.dense_evaluator import DenseEvaluationResult, evaluate_dense_network
from matrixai.training.spec import EvaluationResult, TrainingRunResult, TrainingSpec


def _resolve_path(value: str, base: Path) -> Path | None:
    direct = Path(value)
    for candidate in ([direct] if direct.is_absolute() else [direct, base / value]):
        if candidate.exists():
            return candidate
    return None


def _labels_from_spec(training: TrainingSpec) -> list[str]:
    """Extract class labels from training spec target type, if any."""
    try:
        target_type = training.dataset.target.type
        args = target_type.parameters.get("args", [])
        return [str(a) for a in args] if args else []
    except AttributeError:
        return []


def _examples_to_xy(
    examples: list[SupervisedExample],
    loss_fn: str,
    labels: list[str],
) -> list[tuple[list[float], list[float]]]:
    """Convert SupervisedExample list to (input, target) float pairs."""
    result: list[tuple[list[float], list[float]]] = []
    for ex in examples:
        x = ex.vector
        if loss_fn == "mse":
            y_val = ex.target_value if ex.target_value is not None else float(ex.label)
            result.append((x, [y_val]))
        elif loss_fn == "binary_cross_entropy":
            if ex.target_value is not None:
                result.append((x, [ex.target_value]))
            else:
                pos_label = labels[1] if len(labels) >= 2 else "positive"
                result.append((x, [1.0 if ex.label == pos_label else 0.0]))
        elif loss_fn == "cross_entropy":
            if labels:
                one_hot = [1.0 if ex.label == lbl else 0.0 for lbl in labels]
            else:
                try:
                    y_val = float(ex.label)
                    one_hot = [y_val]
                except (ValueError, TypeError):
                    one_hot = [0.0]
            result.append((x, one_hot))
        else:
            y_val = ex.target_value if ex.target_value is not None else 0.0
            result.append((x, [y_val]))
    return result


class DenseSupervisedTrainer:
    """Train a dense neural network defined as a NETWORK block in .mxai."""

    def train(
        self,
        training: TrainingSpec,
        output_dir: str | None = None,
        base_path: Path | None = None,
        training_path: Path | None = None,
        epoch_callback: Any | None = None,
        seed: int = 42,
    ) -> TrainingRunResult:
        base_path = base_path or Path(".")
        output_dir = output_dir or "output"
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_path = _resolve_path(training.model, base_path)
        if model_path is None:
            raise FileNotFoundError(f"Model file not found: {training.model}")

        program = parse_file(model_path)
        if not program.networks:
            raise ValueError(f"No NETWORK blocks found in {training.model}")

        net = program.networks[0]
        loss_fn = training.loss.type if training.loss else "mse"
        lr = training.optimizer.learning_rate if training.optimizer else 0.01
        # Auditoría C4 [ALTA-1]: este trainer stdlib SOLO implementa sgd — un
        # TYPE distinto falla cerrado, nunca se sustituye en silencio.
        opt_type = training.optimizer.type if training.optimizer else "sgd"
        if opt_type != "sgd":
            raise ValueError(
                f"OPTIMIZER TYPE {opt_type!r} is not implemented by the stdlib "
                f"dense trainer (sgd only) — use --backend torch"
            )
        epochs = training.run.epochs if training.run else 50
        # Early stopping: the only metric this trainer computes is validation_loss,
        # so any declared metric maps to it. best_ps already implements save_best.
        patience = training.run.early_stop_patience if training.run else None

        labels = _labels_from_spec(training)
        mhash = program_hash(program)

        # Run type-checking to resolve layer shapes (input_shape/output_shape)
        from matrixai.types import check_network_types
        vector_map = {v.name: v for v in program.vectors}
        type_result = check_network_types(net, vector_map)
        resolved_layers = type_result.resolved_layers if type_result.resolved_layers else net.layers

        dataset_source = training.dataset.source if training.dataset else ""
        data_path = _resolve_path(dataset_source, base_path) if dataset_source else None
        examples = self._load_examples(program, net, training, data_path)

        # BIBLIOTECA_PROYECTOS_INTELIGENTES C3: este camino YA era secuencial
        # (nunca baraja, ratio 0.8 fijo, ignora `training.dataset.split` por
        # completo) — mode ausente/"random" se deja INTACTO (byte-idéntico,
        # invariante del corte). mode=temporal es lo único nuevo: usa el
        # ratio DECLARADO en vez del 0.8 fijo, mismo mecanismo secuencial
        # (el último tramo, en el orden que llega, es la validación).
        split_spec = training.dataset.split
        if split_spec is not None and split_spec.mode == "temporal":
            train_ratio = split_spec.train
            split = max(1, min(len(examples) - 1, int(len(examples) * train_ratio))) \
                if len(examples) > 1 else len(examples)
        else:
            split = max(1, int(len(examples) * 0.8)) if len(examples) > 1 else len(examples)
        train_ex = examples[:split]
        val_ex = examples[split:]

        ps = build_network_parameter_set(net, resolved_layers, mhash, seed=seed)

        best_ps = ps
        best_val_loss = float("inf")
        best_epoch = 1
        train_loss = 0.0

        from matrixai.forward.dense_forward import dense_forward

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for x, y in train_ex:
                ps, loss = dense_train_step(net, ps, x, y, loss_fn, learning_rate=lr)
                epoch_loss += loss
            train_loss = epoch_loss / len(train_ex) if train_ex else 0.0

            val_loss = 0.0
            for x, y in val_ex:
                pred = dense_forward(net, ps, x)
                val_loss += compute_loss(loss_fn, pred, y)
            val_loss = val_loss / len(val_ex) if val_ex else train_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ps = ps
                best_epoch = epoch

            if epoch_callback is not None:
                epoch_callback({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "validation_loss": round(val_loss, 6),
                    "accuracy": None,
                })

            if patience is not None and (epoch - best_epoch) >= patience:
                break

        accuracy = 0.0
        if val_ex:
            eval_result = evaluate_dense_network(net, best_ps, val_ex, loss_fn, labels=labels or None)
            if eval_result.is_regression():
                accuracy = max(0.0, min(1.0, eval_result.r2))
            else:
                accuracy = eval_result.accuracy

        run_id = str(uuid.uuid4())[:8]
        ps_path = out / "parameter_set.json"
        write_parameter_set(ps_path, best_ps)

        trace_path = out / "training_trace.json"
        trace_path.write_text(
            json.dumps({
                "run_id": run_id,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "network": net.name,
                # BIBLIOTECA C2 (auditoría): sin esta clave,
                # `_collect_training_result` (playground.py) caía SIEMPRE al
                # default "classification" — cualquier NETWORK de regresión
                # entrenado por el camino stdlib (DenseSupervisedTrainer, el
                # fallback cuando use_torch=False) reportaba task_kind
                # incorrecto, aunque el entrenamiento en sí fuera correcto.
                # Mismo criterio que ya usa el camino torch (is_reg = loss_fn
                # == "mse").
                "task_kind": "regression" if loss_fn == "mse" else "classification",
            }, indent=2),
            encoding="utf-8",
        )

        return TrainingRunResult(
            run_id=run_id,
            output_dir=str(out),
            best_epoch=best_epoch,
            best_validation_loss=best_val_loss,
            final_train_loss=train_loss,
            final_validation_loss=best_val_loss,
            accuracy=accuracy,
            artifacts={
                "parameter_set": str(ps_path),
                "training_trace": str(trace_path),
            },
        )

    def _load_examples(
        self,
        program: Any,
        net: Any,
        training: TrainingSpec,
        data_path: Path | None,
    ) -> list[tuple[list[float], list[float]]]:
        if data_path is None or not data_path.exists():
            return []
        vector_map = {v.name: v for v in program.vectors}
        vector = vector_map.get(net.input)
        if vector is None:
            return []
        loss_fn = training.loss.type if training.loss else "mse"
        labels = _labels_from_spec(training)
        target_col = training.dataset.target.name
        adapter = CSVDataAdapter(
            data_path,
            vector.name,
            list(vector.fields),
            target_col,
            labels if labels else None,
        )
        return _examples_to_xy(adapter.examples(), loss_fn, labels)


class DenseSupervisedEvaluator:
    """Evaluate a dense neural network with a loaded ParameterSet."""

    def evaluate(
        self,
        training: TrainingSpec,
        parameter_set: ParameterSet,
        data_path: str | None = None,
        base_path: Path | None = None,
    ) -> EvaluationResult:
        base_path = base_path or Path(".")

        model_path = _resolve_path(training.model, base_path)
        if model_path is None:
            raise FileNotFoundError(f"Model file not found: {training.model}")

        program = parse_file(model_path)
        if not program.networks:
            raise ValueError(f"No NETWORK blocks found in {training.model}")

        net = program.networks[0]
        loss_fn = training.loss.type if training.loss else "mse"
        labels = _labels_from_spec(training)

        dataset_source = training.dataset.source if training.dataset else ""
        if data_path:
            resolved_data = _resolve_path(data_path, base_path)
        else:
            resolved_data = _resolve_path(dataset_source, base_path) if dataset_source else None

        examples = self._load_examples(program, net, training, resolved_data)

        if not examples:
            dense_result = DenseEvaluationResult(rows=0, loss=0.0, loss_fn=loss_fn)
        else:
            dense_result = evaluate_dense_network(
                net, parameter_set, examples, loss_fn, labels=labels or None
            )

        mhash = program_hash(program)
        data_fp = dataset_fingerprint(resolved_data) if resolved_data and resolved_data.exists() else ""

        per_label: dict[str, dict[str, float]] = {}
        if labels and dense_result.precision:
            for lbl in labels:
                per_label[lbl] = {
                    "precision": dense_result.precision.get(lbl, 0.0),
                    "recall": dense_result.recall.get(lbl, 0.0),
                    "f1": dense_result.f1.get(lbl, 0.0),
                }

        macro_p = sum(dense_result.precision.values()) / len(dense_result.precision) if dense_result.precision else 0.0
        macro_r = sum(dense_result.recall.values()) / len(dense_result.recall) if dense_result.recall else 0.0

        return EvaluationResult(
            model=training.model,
            model_hash=mhash,
            parameter_schema_hash=parameter_set.parameter_schema_hash,
            parameter_set_id=parameter_set.parameter_set_id,
            dataset=str(resolved_data or ""),
            dataset_fingerprint=data_fp,
            dataset_schema={},
            rows=dense_result.rows,
            loss=dense_result.loss,
            accuracy=dense_result.accuracy,
            labels=list(labels),
            confusion_matrix=dense_result.confusion_matrix,
            per_label=per_label,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=dense_result.macro_f1,
            mae=dense_result.mae,
            rmse=dense_result.rmse,
            r2=dense_result.r2,
        )

    def _load_examples(
        self,
        program: Any,
        net: Any,
        training: TrainingSpec,
        data_path: Path | None,
    ) -> list[tuple[list[float], list[float]]]:
        if data_path is None or not data_path.exists():
            return []
        vector_map = {v.name: v for v in program.vectors}
        vector = vector_map.get(net.input)
        if vector is None:
            return []
        loss_fn = training.loss.type if training.loss else "mse"
        labels = _labels_from_spec(training)
        target_col = training.dataset.target.name
        adapter = CSVDataAdapter(
            data_path,
            vector.name,
            list(vector.fields),
            target_col,
            labels if labels else None,
        )
        return _examples_to_xy(adapter.examples(), loss_fn, labels)
