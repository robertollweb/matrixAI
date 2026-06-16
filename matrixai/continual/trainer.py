# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P22 C5 — IncrementalTrainer: warm-start fine-tuning from a base ParameterSet."""
from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from matrixai.continual.dataset import ContinualDataset
from matrixai.ir.continual import ContinualPolicySpec, EarlyStopSpec
from matrixai.parameters.store import ParameterSet
from matrixai.training.data import SupervisedExample


# ── result dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IncrementalTrainingResult:
    candidate_parameter_set: ParameterSet
    parent_parameter_set_id: str
    epochs_run: int
    best_epoch: int
    best_validation_loss: float
    epoch_trace: list[dict[str, Any]] = field(default_factory=list, hash=False)
    stopped_early: bool = False


# ── trainer ───────────────────────────────────────────────────────────────────

class IncrementalTrainer:
    """Warm-start fine-tuning from a base ParameterSet on a ContinualDataset.

    Two modes are supported:

    **P4 mode** (auto-detected when ``program`` is None and the ParameterSet
    contains ``W1`` + ``b1`` keys and ``labels`` is provided): uses analytically
    derived softmax or sigmoid gradients.  Suitable for all P4/flat-weight
    classifiers and regression models.

    **Generic mode** (when ``program`` is provided): compiles the model with the
    ``differentiable_python`` backend and uses finite-difference numerical
    gradients via ``GenericSupervisedTrainer``.  Works for any compiled model
    (P11+, dense networks, composites).

    In both modes the learning rate applied during fine-tuning is::

        effective_lr = base_learning_rate * policy.training.learning_rate_factor

    Early stopping is controlled by ``policy.training.early_stop`` (an
    ``EarlyStopSpec`` with ``patience`` and ``metric``).  If not set the
    trainer runs for exactly ``policy.training.max_epochs`` epochs.

    The returned ``IncrementalTrainingResult.candidate_parameter_set`` stores
    ``parent_parameter_set_id`` inside its ``metrics`` dict so the lineage is
    preserved without modifying the ``ParameterSet`` schema.
    """

    def __init__(
        self,
        policy: ContinualPolicySpec,
        base_parameter_set: ParameterSet,
        continual_dataset: ContinualDataset,
        *,
        program: Any = None,
        labels: list[str] | None = None,
        prediction_key: str | None = None,
        base_learning_rate: float = 0.01,
        validation_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        _SUPPORTED_METHODS = {"incremental_finetune", "replay_buffer"}
        method = policy.training.method
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"IncrementalTrainer: unsupported TRAINING METHOD {method!r}. "
                f"Supported in P22 MVP: {sorted(_SUPPORTED_METHODS)}. "
                "'full_retrain' is reserved for a future version."
            )

        self._policy = policy
        self._base = base_parameter_set
        self._dataset = continual_dataset
        self._program = program
        self._labels = labels
        self._prediction_key = prediction_key
        self._base_lr = base_learning_rate
        self._val_fraction = validation_fraction
        self._seed = seed

    # ── public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> IncrementalTrainingResult:
        """Execute incremental fine-tuning and return the result."""
        examples = self._dataset.examples()
        weights = self._dataset.weights()

        if not examples:
            raise ValueError("ContinualDataset has no examples for incremental training")

        train_ex, train_w, val_ex = self._split(examples, weights)

        if self._program is not None:
            return self._run_generic(train_ex, train_w, val_ex, epoch_callback)

        if self._is_p4_mode():
            return self._run_p4(train_ex, train_w, val_ex, epoch_callback)

        raise ValueError(
            "IncrementalTrainer: no program provided and parameters are not P4-style "
            "(W1 + b1 keys with labels). Pass program=<MatrixAIProgram> to use "
            "numerical-gradient mode."
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    @property
    def _effective_lr(self) -> float:
        return self._base_lr * self._policy.training.learning_rate_factor

    @property
    def _max_epochs(self) -> int:
        return self._policy.training.max_epochs

    @property
    def _early_stop(self) -> EarlyStopSpec | None:
        return self._policy.training.early_stop

    def _split(
        self,
        examples: list[SupervisedExample],
        weights: list[float],
    ) -> tuple[list[SupervisedExample], list[float], list[SupervisedExample]]:
        n = len(examples)
        val_count = max(1, int(n * self._val_fraction)) if n > 1 else 0
        train_count = n - val_count

        rng = random.Random(self._seed)
        indices = list(range(n))
        rng.shuffle(indices)

        train_idx = indices[:train_count]
        val_idx = indices[train_count:]
        return (
            [examples[i] for i in train_idx],
            [weights[i] for i in train_idx],
            [examples[i] for i in val_idx],
        )

    def _is_p4_mode(self) -> bool:
        keys = set(self._base.parameters.keys())
        return "W1" in keys and "b1" in keys

    def _p4_objective(self) -> str:
        w1_values = self._base.parameters["W1"]["values"]
        if _is_matrix(w1_values):
            return "softmax_cross_entropy"
        labels = self._labels or []
        if len(labels) == 2:
            return "sigmoid_binary_cross_entropy"
        return "mse_regression"

    # ── P4 analytical training ─────────────────────────────────────────────────

    def _run_p4(
        self,
        train_ex: list[SupervisedExample],
        train_w: list[float],
        val_ex: list[SupervisedExample],
        epoch_callback: Callable[[dict[str, Any]], None] | None,
    ) -> IncrementalTrainingResult:
        objective = self._p4_objective()
        lr = self._effective_lr
        labels = self._labels or []

        weights_param: Any = _deep_copy_values(self._base.parameters["W1"]["values"])
        bias_param: Any = _deep_copy_values(self._base.parameters["b1"]["values"])

        best_weights = _deep_copy_values(weights_param)
        best_bias = _deep_copy_values(bias_param)
        best_metric_val = float("inf")  # always track loss for best-epoch selection
        best_epoch = 0
        stale_epochs = 0
        epoch_trace: list[dict[str, Any]] = []
        stopped_early = False

        monitor_metric = (self._early_stop.metric if self._early_stop else "loss")

        for epoch in range(1, self._max_epochs + 1):
            # Weighted gradient step
            weights_param, bias_param = self._p4_gradient_step(
                train_ex, train_w, labels, weights_param, bias_param, objective, lr
            )

            # Evaluate
            val_loss, val_acc = _p4_eval(val_ex or train_ex, labels, weights_param, bias_param, objective)
            train_loss, train_acc = _p4_eval(train_ex, labels, weights_param, bias_param, objective)

            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "accuracy": val_acc,
            }
            epoch_trace.append(entry)
            if epoch_callback is not None:
                epoch_callback(entry)

            # Early-stopping metric value (lower = better for loss, higher = better for accuracy)
            if monitor_metric == "accuracy":
                current = -val_acc  # negate so "lower is better" still applies
            else:
                current = val_loss

            if current < best_metric_val:
                best_metric_val = current
                best_epoch = epoch
                best_weights = _deep_copy_values(weights_param)
                best_bias = _deep_copy_values(bias_param)
                stale_epochs = 0
            else:
                stale_epochs += 1

            if self._early_stop is not None and stale_epochs >= self._early_stop.patience:
                stopped_early = True
                break

        best_val_loss = epoch_trace[best_epoch - 1]["validation_loss"] if epoch_trace else float("inf")
        best_val_acc = epoch_trace[best_epoch - 1]["accuracy"] if epoch_trace else 0.0

        candidate = self._build_candidate_p4(
            best_weights, best_bias, best_val_loss, best_val_acc,
            len(epoch_trace), stopped_early,
        )
        return IncrementalTrainingResult(
            candidate_parameter_set=candidate,
            parent_parameter_set_id=self._base.parameter_set_id,
            epochs_run=len(epoch_trace),
            best_epoch=best_epoch,
            best_validation_loss=best_val_loss,
            epoch_trace=epoch_trace,
            stopped_early=stopped_early,
        )

    def _p4_gradient_step(
        self,
        examples: list[SupervisedExample],
        weights: list[float],
        labels: list[str],
        w_param: Any,
        b_param: Any,
        objective: str,
        lr: float,
    ) -> tuple[Any, Any]:
        if not examples:
            return w_param, b_param

        total_w = sum(weights) or 1.0
        if objective == "softmax_cross_entropy":
            dW, db = _softmax_weighted_grad(examples, weights, labels, w_param, b_param, total_w)
        elif objective == "sigmoid_binary_cross_entropy":
            dW, db = _sigmoid_weighted_grad(examples, weights, labels, w_param, b_param, total_w)
        else:
            dW, db = _mse_weighted_grad(examples, weights, w_param, b_param, total_w)

        new_w = _sgd_update(w_param, dW, lr)
        new_b = _sgd_update(b_param, db, lr)
        return new_w, new_b

    def _build_candidate_p4(
        self,
        weights: Any,
        bias: Any,
        val_loss: float,
        val_acc: float,
        epochs_run: int,
        stopped_early: bool,
    ) -> ParameterSet:
        data = self._base.to_dict()
        data["parameter_set_id"] = f"{self._base.parameter_set_id}_incremental"
        data["source"] = "incremental_finetune"
        data["parameters"]["W1"]["values"] = _rounded_values(weights)
        data["parameters"]["b1"]["values"] = _rounded_values(bias)
        data["metrics"] = {
            "parent_parameter_set_id": self._base.parameter_set_id,
            "validation_loss": val_loss,
            "accuracy": val_acc,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "dataset_fingerprint": self._dataset.fingerprint(),
        }
        return ParameterSet.from_dict(data)

    # ── generic (program-based) training ──────────────────────────────────────

    def _run_generic(
        self,
        train_ex: list[SupervisedExample],
        train_w: list[float],
        val_ex: list[SupervisedExample],
        epoch_callback: Callable[[dict[str, Any]], None] | None,
    ) -> IncrementalTrainingResult:
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        from matrixai.training.trainer import (
            _compile_and_exec,
            _generic_cross_entropy_loss,
            _generic_metrics,
            _generic_parameter_set_from_runtime_params,
            _build_input_data,
            _apply_sgd_to_param,
            _numerical_gradient_for_param,
            match_update_patterns,
        )
        from matrixai.parameters.store import build_initial_parameter_set, separate_parameters

        program = self._program
        source = DifferentiablePythonCompiler().compile(program)
        ns = _compile_and_exec(source)
        run_fn = ns["run"]

        labels = self._labels
        prediction_key = self._prediction_key or ""

        # Start from base parameter VALUES (warm start)
        runtime_params: dict[str, Any] = {}
        for key, entry in self._base.parameters.items():
            runtime_params[key] = deepcopy(entry.get("values"))

        all_keys = list(self._base.parameters.keys())
        trainable_keys = [k for k in all_keys if not k.startswith("__frozen")]

        # Respect FROZEN params from composite programs
        _, frozen_keys = separate_parameters(self._base, program)
        trainable_keys = [k for k in all_keys if k not in frozen_keys]

        best_params = deepcopy(runtime_params)
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        epoch_trace: list[dict[str, Any]] = []
        stopped_early = False
        lr = self._effective_lr
        monitor_metric = (self._early_stop.metric if self._early_stop else "loss")

        for epoch in range(1, self._max_epochs + 1):
            epoch_loss = 0.0
            for example, w in zip(train_ex, train_w):
                input_data = _build_input_data(example)
                result = run_fn(input_data, runtime_params)
                loss = _generic_cross_entropy_loss(result["state"], example.label, prediction_key, labels=labels)
                epoch_loss += loss * w
                for key in trainable_keys:
                    grad = _numerical_gradient_for_param(
                        run_fn, runtime_params, key, input_data,
                        example.label, prediction_key, labels=labels,
                    )
                    if grad is not None:
                        runtime_params = {
                            **runtime_params,
                            key: _apply_sgd_to_param(runtime_params[key], grad, lr * w),
                        }

            eval_examples = val_ex or train_ex
            val_loss = sum(
                _generic_cross_entropy_loss(
                    run_fn(_build_input_data(e), runtime_params)["state"],
                    e.label, prediction_key, labels=labels,
                )
                for e in eval_examples
            ) / max(len(eval_examples), 1)

            val_metrics = _generic_metrics(
                run_fn, eval_examples, prediction_key, runtime_params,
                labels=labels,
            )
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(len(train_ex), 1),
                "validation_loss": val_loss,
                "accuracy": val_metrics.get("accuracy", 0.0),
            }
            epoch_trace.append(entry)
            if epoch_callback:
                epoch_callback(entry)

            current = -val_metrics.get("accuracy", 0.0) if monitor_metric == "accuracy" else val_loss
            if current < best_loss:
                best_loss = current
                best_epoch = epoch
                best_params = deepcopy(runtime_params)
                stale_epochs = 0
            else:
                stale_epochs += 1

            if self._early_stop is not None and stale_epochs >= self._early_stop.patience:
                stopped_early = True
                break

        best_val_loss = epoch_trace[best_epoch - 1]["validation_loss"] if epoch_trace else float("inf")
        best_val_acc = epoch_trace[best_epoch - 1]["accuracy"] if epoch_trace else 0.0

        # Reconstruct candidate ParameterSet from best runtime params
        data = self._base.to_dict()
        data["parameter_set_id"] = f"{self._base.parameter_set_id}_incremental"
        data["source"] = "incremental_finetune"
        for key in trainable_keys:
            if key in best_params and key in data["parameters"]:
                data["parameters"][key]["values"] = _rounded_values(best_params[key])
        data["metrics"] = {
            "parent_parameter_set_id": self._base.parameter_set_id,
            "validation_loss": best_val_loss,
            "accuracy": best_val_acc,
            "epochs_run": len(epoch_trace),
            "stopped_early": stopped_early,
            "dataset_fingerprint": self._dataset.fingerprint(),
        }
        candidate = ParameterSet.from_dict(data)

        return IncrementalTrainingResult(
            candidate_parameter_set=candidate,
            parent_parameter_set_id=self._base.parameter_set_id,
            epochs_run=len(epoch_trace),
            best_epoch=best_epoch,
            best_validation_loss=best_val_loss,
            epoch_trace=epoch_trace,
            stopped_early=stopped_early,
        )


# ── P4 gradient helpers ────────────────────────────────────────────────────────

def _is_matrix(values: Any) -> bool:
    return isinstance(values, list) and bool(values) and isinstance(values[0], list)


def _deep_copy_values(values: Any) -> Any:
    if isinstance(values, list):
        return [_deep_copy_values(v) for v in values]
    return float(values)


def _rounded_values(values: Any) -> Any:
    if isinstance(values, list):
        return [_rounded_values(v) for v in values]
    return round(float(values), 10)


def _logits_softmax(vector: list[float], W: list[list[float]], b: list[float]) -> list[float]:
    return [
        sum(v * w for v, w in zip(vector, W[k])) + b[k]
        for k in range(len(b))
    ]


def _softmax(logits: list[float]) -> list[float]:
    max_v = max(logits)
    exps = [math.exp(x - max_v) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dot(v: list[float], w: list[float]) -> float:
    return sum(a * b for a, b in zip(v, w))


def _softmax_weighted_grad(
    examples: list[SupervisedExample],
    sample_weights: list[float],
    labels: list[str],
    W: list[list[float]],
    b: list[float],
    total_w: float,
) -> tuple[list[list[float]], list[float]]:
    n_labels = len(labels)
    n_feat = len(examples[0].vector) if examples else 0
    dW = [[0.0] * n_feat for _ in range(n_labels)]
    db = [0.0] * n_labels
    for example, w in zip(examples, sample_weights):
        logits = _logits_softmax(example.vector, W, b)
        probs = _softmax(logits)
        for k, lbl in enumerate(labels):
            target = 1.0 if example.label == lbl else 0.0
            delta = (probs[k] - target) * w
            for j, xj in enumerate(example.vector):
                dW[k][j] += delta * xj
            db[k] += delta
    dW = [[g / total_w for g in row] for row in dW]
    db = [g / total_w for g in db]
    return dW, db


def _sigmoid_weighted_grad(
    examples: list[SupervisedExample],
    sample_weights: list[float],
    labels: list[str],
    W: list[float],
    b: Any,
    total_w: float,
) -> tuple[list[float], float]:
    pos_label = labels[1] if len(labels) >= 2 else ""
    n_feat = len(examples[0].vector) if examples else 0
    dW = [0.0] * n_feat
    db = 0.0
    b_val = float(b[0]) if isinstance(b, list) else float(b)
    for example, w in zip(examples, sample_weights):
        score = _dot(example.vector, W) + b_val
        p = _sigmoid(score)
        target = 1.0 if example.label == pos_label else 0.0
        delta = (p - target) * w
        for j, xj in enumerate(example.vector):
            dW[j] += delta * xj
        db += delta
    dW = [g / total_w for g in dW]
    db = db / total_w
    return dW, db


def _mse_weighted_grad(
    examples: list[SupervisedExample],
    sample_weights: list[float],
    W: Any,
    b: Any,
    total_w: float,
) -> tuple[Any, Any]:
    w_vals = [float(v) for v in W] if isinstance(W, list) else [float(W)]
    b_val = float(b[0]) if isinstance(b, list) else float(b)
    n_feat = len(examples[0].vector) if examples else len(w_vals)
    dW = [0.0] * n_feat
    db = 0.0
    for example, sw in zip(examples, sample_weights):
        y_hat = _dot(example.vector, w_vals) + b_val
        try:
            y = float(example.label)
        except (ValueError, TypeError):
            y = example.target_value if example.target_value is not None else 0.0
        delta = 2.0 * (y_hat - y) * sw
        for j, xj in enumerate(example.vector):
            dW[j] += delta * xj
        db += delta
    dW = [g / total_w for g in dW]
    db = db / total_w
    return dW, db


def _sgd_update(param: Any, grad: Any, lr: float) -> Any:
    if isinstance(param, list) and isinstance(grad, list):
        return [_sgd_update(p, g, lr) for p, g in zip(param, grad)]
    return float(param) - lr * float(grad)


def _p4_eval(
    examples: list[SupervisedExample],
    labels: list[str],
    W: Any,
    b: Any,
    objective: str,
) -> tuple[float, float]:
    if not examples:
        return 0.0, 0.0

    if objective == "softmax_cross_entropy":
        loss = 0.0
        correct = 0
        for ex in examples:
            logits = _logits_softmax(ex.vector, W, b)
            probs = _softmax(logits)
            target_idx = labels.index(ex.label) if ex.label in labels else 0
            loss -= math.log(max(probs[target_idx], 1e-12))
            predicted_idx = probs.index(max(probs))
            if labels[predicted_idx] == ex.label:
                correct += 1
        return loss / len(examples), correct / len(examples)

    if objective == "sigmoid_binary_cross_entropy":
        pos_label = labels[1] if len(labels) >= 2 else ""
        b_val = float(b[0]) if isinstance(b, list) else float(b)
        loss = 0.0
        correct = 0
        for ex in examples:
            score = _dot(ex.vector, W) + b_val
            p = _sigmoid(score)
            target = 1.0 if ex.label == pos_label else 0.0
            p_clip = min(1.0 - 1e-12, max(1e-12, p))
            loss -= target * math.log(p_clip) + (1.0 - target) * math.log(1.0 - p_clip)
            predicted = pos_label if p >= 0.5 else (labels[0] if labels else "")
            if predicted == ex.label:
                correct += 1
        return loss / len(examples), correct / len(examples)

    # mse_regression
    w_vals = [float(v) for v in W] if isinstance(W, list) else [float(W)]
    b_val = float(b[0]) if isinstance(b, list) else float(b)
    loss = 0.0
    for ex in examples:
        y_hat = _dot(ex.vector, w_vals) + b_val
        try:
            y = float(ex.label)
        except (ValueError, TypeError):
            y = ex.target_value if ex.target_value is not None else 0.0
        loss += (y_hat - y) ** 2
    return loss / len(examples), 0.0
