# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P22 C6 — ApprovalGate: holdout evaluation and regression guard before promoting a candidate."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from matrixai.ir.continual import ContinualPolicySpec
from matrixai.parameters.store import ParameterSet
from matrixai.training.data import SupervisedExample


# ── value objects ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HoldoutMetrics:
    """Evaluation metrics for one ParameterSet on the holdout set."""
    loss: float
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_label: dict[str, dict[str, float]] = field(default_factory=dict, hash=False)
    samples: int = 0

    def get(self, metric: str) -> float:
        lookup: dict[str, float] = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
        }
        if metric not in lookup:
            raise ValueError(f"ApprovalGate: unsupported metric {metric!r}; available: {sorted(lookup)}")
        return lookup[metric]


@dataclass(frozen=True)
class RegressionGuardResult:
    passed: bool
    metric: str
    baseline_value: float
    candidate_value: float
    must_improve_by: float
    actual_delta: float
    per_label_violations: dict[str, float] = field(default_factory=dict, hash=False)
    reasons: list[str] = field(default_factory=list, hash=False)


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    policy_hash: str
    candidate_parameter_set_id: str
    parent_parameter_set_id: str
    created_at: str
    expires_at: str | None
    approval_token: str
    channel: str | None
    status: str = "pending"          # pending | approved | rejected
    decided_at: str | None = None
    decided_by: str | None = None
    decision_token: str | None = None


@dataclass(frozen=True)
class ApprovalGateReport:
    policy_hash: str
    status: str          # "automatic_pass" | "pending_human" | "rejected"
    candidate_parameter_set_id: str
    baseline_parameter_set_id: str
    holdout_samples: int
    baseline_metrics: HoldoutMetrics
    candidate_metrics: HoldoutMetrics
    regression_guard: RegressionGuardResult
    pending_approval: PendingApproval | None
    evaluated_at: str
    rejection_reasons: list[str] = field(default_factory=list, hash=False)

    @property
    def passed(self) -> bool:
        return self.status in ("automatic_pass", "pending_human")


# ── gate ──────────────────────────────────────────────────────────────────────

class ApprovalGate:
    """Evaluates a candidate ParameterSet against a baseline on a holdout set.

    The gate applies the ``REGRESSION_GUARD`` declared in the policy:

    - The candidate's metric value must be at least
      ``baseline_value + regression_guard.must_improve_by``.
    - For each label, per-label recall must not fall below
      ``baseline_per_label_recall - regression_guard.max_degradation_per_label``.

    Gate outcome:
    - ``rejected``: regression guard fails (promotion is blocked).
    - ``automatic_pass``: guard passes and ``HUMAN_APPROVAL false``.
    - ``pending_human``: guard passes but ``HUMAN_APPROVAL true``; a
      :class:`PendingApproval` token is created.
    """

    def __init__(
        self,
        policy: ContinualPolicySpec,
        candidate_set: ParameterSet,
        baseline_set: ParameterSet,
        holdout: list[SupervisedExample],
        *,
        labels: list[str] | None = None,
        signing_key: str | None = None,
        now: datetime | None = None,
    ) -> None:
        self._policy = policy
        self._candidate = candidate_set
        self._baseline = baseline_set
        self._holdout = holdout
        self._labels = labels or []
        self._signing_key = signing_key
        self._now = now or datetime.now(tz=timezone.utc)

    # ── public API ─────────────────────────────────────────────────────────────

    def evaluate(self) -> ApprovalGateReport:
        objective = self._detect_objective()
        labels = self._labels

        baseline_metrics = _evaluate_parameter_set(self._baseline, self._holdout, labels, objective)
        candidate_metrics = _evaluate_parameter_set(self._candidate, self._holdout, labels, objective)

        guard_spec = self._policy.approval_gate.regression_guard
        guard_result = _check_regression_guard(guard_spec, baseline_metrics, candidate_metrics)

        reasons = list(guard_result.reasons)

        if not guard_result.passed:
            return self._build_report("rejected", baseline_metrics, candidate_metrics, guard_result, None, reasons)

        if self._policy.approval_gate.human_approval:
            pending = self._create_pending_approval()
            return self._build_report("pending_human", baseline_metrics, candidate_metrics, guard_result, pending, [])

        return self._build_report("automatic_pass", baseline_metrics, candidate_metrics, guard_result, None, [])

    # ── helpers ────────────────────────────────────────────────────────────────

    def _detect_objective(self) -> str:
        params = self._candidate.parameters
        if "W1" not in params or "b1" not in params:
            raise ValueError(
                f"ApprovalGate: ParameterSet {self._candidate.parameter_set_id!r} "
                "has no W1/b1 parameters (non-P4 generic model). "
                "Evaluation is not supported in the P22 MVP — "
                "only P4-style classifiers/regressors with W1+b1 can be evaluated on the holdout."
            )
        w1 = params["W1"]["values"]
        if _is_matrix(w1):
            return "softmax_cross_entropy"
        if len(self._labels) == 2:
            return "sigmoid_binary_cross_entropy"
        return "mse_regression"

    def _build_report(
        self,
        status: str,
        baseline_metrics: HoldoutMetrics,
        candidate_metrics: HoldoutMetrics,
        guard: RegressionGuardResult,
        pending: PendingApproval | None,
        reasons: list[str],
    ) -> ApprovalGateReport:
        return ApprovalGateReport(
            policy_hash=self._policy.policy_hash,
            status=status,
            candidate_parameter_set_id=self._candidate.parameter_set_id,
            baseline_parameter_set_id=self._baseline.parameter_set_id,
            holdout_samples=len(self._holdout),
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            regression_guard=guard,
            pending_approval=pending,
            evaluated_at=self._now.isoformat(),
            rejection_reasons=reasons,
        )

    def _create_pending_approval(self) -> PendingApproval:
        now_str = self._now.isoformat()
        gate_spec = self._policy.approval_gate
        timeout_h = gate_spec.approval_timeout_hours
        expires = (self._now + timedelta(hours=timeout_h)).isoformat() if timeout_h else None

        token = _make_approval_token(
            self._policy.policy_hash,
            self._candidate.parameter_set_id,
            now_str,
            self._signing_key,
            expires_at=expires,
        )
        parent_id = self._candidate.metrics.get("parent_parameter_set_id", self._baseline.parameter_set_id)
        return PendingApproval(
            approval_id=f"apr-{token[:12]}",
            policy_hash=self._policy.policy_hash,
            candidate_parameter_set_id=self._candidate.parameter_set_id,
            parent_parameter_set_id=parent_id,
            created_at=now_str,
            expires_at=expires,
            approval_token=token,
            channel=gate_spec.approval_channel,
        )


# ── regression guard logic ─────────────────────────────────────────────────────

def _check_regression_guard(
    guard_spec: Any,
    baseline: HoldoutMetrics,
    candidate: HoldoutMetrics,
) -> RegressionGuardResult:
    metric = guard_spec.metric
    must_improve = guard_spec.must_improve_by
    max_degradation = guard_spec.max_degradation_per_label

    try:
        base_val = baseline.get(metric)
        cand_val = candidate.get(metric)
    except ValueError as exc:
        return RegressionGuardResult(
            passed=False,
            metric=metric,
            baseline_value=0.0,
            candidate_value=0.0,
            must_improve_by=must_improve,
            actual_delta=0.0,
            reasons=[str(exc)],
        )

    # For loss: lower is better → negate so "delta >= must_improve" still works
    if metric == "loss":
        actual_delta = base_val - cand_val   # positive if candidate improved
    else:
        actual_delta = cand_val - base_val   # positive if candidate improved

    reasons: list[str] = []
    if actual_delta < must_improve:
        reasons.append(
            f"REGRESSION_GUARD: {metric} delta {actual_delta:+.6f} < must_improve_by {must_improve}"
        )

    # Per-label recall degradation check (only for classification)
    violations: dict[str, float] = {}
    if baseline.per_label and candidate.per_label:
        for label, base_label_metrics in baseline.per_label.items():
            cand_label_metrics = candidate.per_label.get(label, {})
            base_recall = base_label_metrics.get("recall", 0.0)
            cand_recall = cand_label_metrics.get("recall", 0.0)
            degradation = base_recall - cand_recall
            if degradation > max_degradation:
                violations[label] = degradation
                reasons.append(
                    f"REGRESSION_GUARD: label '{label}' recall degraded by {degradation:.4f} "
                    f"(max_degradation_per_label={max_degradation})"
                )

    passed = not reasons
    return RegressionGuardResult(
        passed=passed,
        metric=metric,
        baseline_value=base_val,
        candidate_value=cand_val,
        must_improve_by=must_improve,
        actual_delta=actual_delta,
        per_label_violations=violations,
        reasons=reasons,
    )


# ── parameter set evaluation helpers ──────────────────────────────────────────

def _evaluate_parameter_set(
    ps: ParameterSet,
    holdout: list[SupervisedExample],
    labels: list[str],
    objective: str,
) -> HoldoutMetrics:
    if not holdout:
        return HoldoutMetrics(loss=0.0, accuracy=0.0, macro_f1=0.0,
                               macro_precision=0.0, macro_recall=0.0,
                               per_label={}, samples=0)

    W = _copy_values(ps.parameters["W1"]["values"]) if "W1" in ps.parameters else None
    b = _copy_values(ps.parameters["b1"]["values"]) if "b1" in ps.parameters else None

    if objective == "softmax_cross_entropy" and W is not None and b is not None:
        return _softmax_holdout_metrics(holdout, labels, W, b)
    if objective == "sigmoid_binary_cross_entropy" and W is not None and b is not None:
        return _sigmoid_holdout_metrics(holdout, labels, W, b)
    if objective == "mse_regression" and W is not None and b is not None:
        return _mse_holdout_metrics(holdout, W, b)

    raise ValueError(
        f"ApprovalGate: ParameterSet {ps.parameter_set_id!r} has no W1/b1 parameters "
        "(non-P4 generic model). Evaluation is not supported in the P22 MVP — "
        "only P4-style classifiers/regressors with W1+b1 can be evaluated on the holdout. "
        "Provide a ParameterSet with W1 and b1 parameters."
    )


def _softmax_holdout_metrics(
    holdout: list[SupervisedExample],
    labels: list[str],
    W: list[list[float]],
    b: list[float],
) -> HoldoutMetrics:
    loss = 0.0
    confusion = {lbl: {p: 0 for p in labels} for lbl in labels}
    for ex in holdout:
        logits = [sum(v * W[k][j] for j, v in enumerate(ex.vector)) + b[k]
                  for k in range(len(labels))]
        probs = _softmax(logits)
        true_idx = labels.index(ex.label) if ex.label in labels else 0
        loss -= math.log(max(probs[true_idx], 1e-12))
        pred_idx = probs.index(max(probs))
        actual = ex.label if ex.label in labels else labels[0]
        predicted = labels[pred_idx]
        if actual not in confusion:
            confusion[actual] = {p: 0 for p in labels}
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1

    per_label = _classification_metrics(confusion, labels)
    accuracy = sum(confusion[l].get(l, 0) for l in labels if l in confusion) / len(holdout)
    macro_p = _macro(per_label, "precision")
    macro_r = _macro(per_label, "recall")
    macro_f = _macro(per_label, "f1")
    return HoldoutMetrics(
        loss=loss / len(holdout),
        accuracy=accuracy,
        macro_f1=macro_f,
        macro_precision=macro_p,
        macro_recall=macro_r,
        per_label=per_label,
        samples=len(holdout),
    )


def _sigmoid_holdout_metrics(
    holdout: list[SupervisedExample],
    labels: list[str],
    W: list[float],
    b: Any,
) -> HoldoutMetrics:
    pos_label = labels[1] if len(labels) >= 2 else ""
    neg_label = labels[0] if labels else ""
    b_val = float(b[0]) if isinstance(b, list) else float(b)
    confusion = {neg_label: {neg_label: 0, pos_label: 0},
                 pos_label: {neg_label: 0, pos_label: 0}}
    loss = 0.0
    for ex in holdout:
        score = sum(v * w for v, w in zip(ex.vector, W)) + b_val
        p = _sigmoid(score)
        target = 1.0 if ex.label == pos_label else 0.0
        p_clip = min(1 - 1e-12, max(1e-12, p))
        loss -= target * math.log(p_clip) + (1 - target) * math.log(1 - p_clip)
        predicted = pos_label if p >= 0.5 else neg_label
        actual = ex.label if ex.label in (neg_label, pos_label) else neg_label
        confusion[actual][predicted] += 1

    per_label = _classification_metrics(confusion, [neg_label, pos_label] if labels else [])
    n = len(holdout)
    accuracy = (confusion[neg_label][neg_label] + confusion[pos_label][pos_label]) / n if n > 0 else 0.0
    macro_p = _macro(per_label, "precision")
    macro_r = _macro(per_label, "recall")
    macro_f = _macro(per_label, "f1")
    return HoldoutMetrics(
        loss=loss / n,
        accuracy=accuracy,
        macro_f1=macro_f,
        macro_precision=macro_p,
        macro_recall=macro_r,
        per_label=per_label,
        samples=n,
    )


def _mse_holdout_metrics(
    holdout: list[SupervisedExample],
    W: Any,
    b: Any,
) -> HoldoutMetrics:
    w_vals = [float(v) for v in W] if isinstance(W, list) else [float(W)]
    b_val = float(b[0]) if isinstance(b, list) else float(b)
    total_loss = 0.0
    for ex in holdout:
        y_hat = sum(v * w for v, w in zip(ex.vector, w_vals)) + b_val
        try:
            y = float(ex.label)
        except (ValueError, TypeError):
            y = ex.target_value if ex.target_value is not None else 0.0
        total_loss += (y_hat - y) ** 2
    return HoldoutMetrics(
        loss=total_loss / len(holdout),
        accuracy=0.0,
        macro_f1=0.0,
        macro_precision=0.0,
        macro_recall=0.0,
        per_label={},
        samples=len(holdout),
    )


# ── math / utility helpers ─────────────────────────────────────────────────────

def _is_matrix(values: Any) -> bool:
    return isinstance(values, list) and bool(values) and isinstance(values[0], list)


def _copy_values(values: Any) -> Any:
    if isinstance(values, list):
        return [_copy_values(v) for v in values]
    return float(values)


def _softmax(logits: list[float]) -> list[float]:
    max_v = max(logits)
    exps = [math.exp(x - max_v) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _classification_metrics(
    confusion: dict[str, dict[str, int]], labels: list[str]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(a, {}).get(label, 0) for a in labels if a != label)
        fn = sum(confusion.get(label, {}).get(p, 0) for p in labels if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion.get(label, {}).values())
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": float(support)}
    return metrics


def _macro(per_label: dict[str, dict[str, float]], key: str) -> float:
    if not per_label:
        return 0.0
    return sum(v[key] for v in per_label.values()) / len(per_label)


def _make_approval_token(
    policy_hash: str,
    candidate_id: str,
    created_at: str,
    signing_key: str | None,
    *,
    expires_at: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "policy_hash": policy_hash,
            "candidate_id": candidate_id,
            "created_at": created_at,
            "expires_at": expires_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if signing_key:
        key_bytes = bytes.fromhex(signing_key) if len(signing_key) == 64 else signing_key.encode()
        digest = hmac.new(key_bytes, payload, hashlib.sha256).hexdigest()
        return f"hmac-sha256:{digest}"
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


def approve_pending_approval(
    pending: PendingApproval,
    *,
    decided_by: str,
    signing_key: str | None = None,
    decided_at: datetime | None = None,
) -> PendingApproval:
    """Return a new PendingApproval marked as human-approved."""
    who = decided_by.strip()
    if not who:
        raise ValueError("decided_by is required to approve a PendingApproval")
    when = decided_at or datetime.now(tz=timezone.utc)
    decided_at_s = when.isoformat()
    token = _make_decision_token(
        pending.approval_token,
        "approved",
        who,
        decided_at_s,
        signing_key,
    )
    return replace(
        pending,
        status="approved",
        decided_at=decided_at_s,
        decided_by=who,
        decision_token=token,
    )


def _make_decision_token(
    approval_token: str,
    status: str,
    decided_by: str,
    decided_at: str,
    signing_key: str | None,
) -> str:
    payload = json.dumps(
        {
            "approval_token": approval_token,
            "status": status,
            "decided_by": decided_by,
            "decided_at": decided_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if signing_key:
        key_bytes = bytes.fromhex(signing_key) if len(signing_key) == 64 else signing_key.encode()
        digest = hmac.new(key_bytes, payload, hashlib.sha256).hexdigest()
        return f"hmac-sha256:{digest}"
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"
