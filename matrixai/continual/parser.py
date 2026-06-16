# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import json
import re
from typing import Any

from matrixai.ir.continual import (
    ApprovalGateSpec,
    ContinualAuditSpec,
    ContinualPolicySpec,
    ContinualRollbackSpec,
    ContinualTrainingSpec,
    ConceptDriftSpec,
    DatasetMixSpec,
    DriftDetectionSpec,
    EarlyStopSpec,
    FeatureDriftMethodSpec,
    GroundTruthSpec,
    RecencyDecaySpec,
    RegressionGuardSpec,
    UpdateTriggerSpec,
    VALID_DRIFT_METHODS,
    compute_policy_hash,
)


class MxcontinualParseError(ValueError):
    pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_lines(source: str) -> list[str]:
    out = []
    for raw in source.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _bool(val: str, key: str) -> bool:
    v = val.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise MxcontinualParseError(f"{key}: expected true/false, got {val!r}")


def _int(val: str, key: str) -> int:
    try:
        return int(val.strip())
    except ValueError:
        raise MxcontinualParseError(f"{key}: expected integer, got {val!r}")


def _float(val: str, key: str) -> float:
    try:
        return float(val.strip())
    except ValueError:
        raise MxcontinualParseError(f"{key}: expected float, got {val!r}")


def _parse_list(raw: str, key: str) -> list[str]:
    """Parse '[a, b, c]' → ['a','b','c']."""
    raw = raw.strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise MxcontinualParseError(f"{key}: expected [...] list, got {raw!r}")
    inner = raw[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip('"').strip("'") for item in inner.split(",")]


def _extract_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """
    Starting right after an opening keyword line at index `start-1`,
    collect lines until the matching END (depth-aware). Returns (block_lines, next_idx).
    """
    block: list[str] = []
    depth = 1
    i = start
    while i < len(lines):
        line = lines[i]
        kw = line.split(maxsplit=1)[0]
        # Any line whose first token is a known opener increments depth
        if kw in _BLOCK_OPENERS:
            depth += 1
        elif line == "END":
            depth -= 1
            if depth == 0:
                return block, i + 1
        if line != "END" or depth > 0:
            if not (line == "END" and depth == 0):
                block.append(line)
        i += 1
    raise MxcontinualParseError("Unexpected end of input — missing END")


# All keywords that open a sub-block (need depth tracking)
_BLOCK_OPENERS = frozenset({
    "GROUND_TRUTH",
    "DRIFT_DETECTION",
    "CONCEPT_DRIFT",
    "UPDATE_TRIGGER",
    "TRAINING",
    "APPROVAL_GATE",
    "ROLLBACK",
    "AUDIT",
    "METHODS",
    "DATASET_MIX",
    "REGRESSION_GUARD",
    "ANY_OF",
    "NOTIFY_SCOPE",
    # NOTE: EARLY_STOP is inline syntax (patience=N metric=M), NOT a block
})


def _kv(line: str) -> tuple[str, str]:
    """Split 'KEY rest' into ('KEY', 'rest'). rest may be empty string."""
    parts = line.split(maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


# ── sub-block parsers ─────────────────────────────────────────────────────────

def _parse_scope_block(lines: list[str]) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    for line in lines:
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", line)
        if m:
            raw = m.group(2).strip()
            if raw.startswith("[") or raw.startswith("{"):
                try:
                    scope[m.group(1)] = json.loads(raw)
                    continue
                except json.JSONDecodeError:
                    pass
            if raw.lower() == "true":
                scope[m.group(1)] = True
            elif raw.lower() == "false":
                scope[m.group(1)] = False
            else:
                scope[m.group(1)] = raw.strip('"').strip("'")
    return scope


def _parse_ground_truth(lines: list[str]) -> GroundTruthSpec:
    window_days: int | None = None
    required_field: str | None = None
    label_type: str | None = None
    sources: list[str] = ["api", "cli"]
    file_watch_path: str | None = None

    i = 0
    while i < len(lines):
        k, v = _kv(lines[i])
        if k == "WINDOW_DAYS":
            window_days = _int(v, "WINDOW_DAYS")
        elif k == "REQUIRED_FIELD":
            required_field = v.strip()
        elif k == "LABEL_TYPE":
            label_type = v.strip()
        elif k == "SOURCES":
            sources = _parse_list(v, "SOURCES")
        elif k == "FILE_WATCH_PATH":
            file_watch_path = v.strip()
        i += 1

    if window_days is None:
        raise MxcontinualParseError("GROUND_TRUTH: WINDOW_DAYS is required")
    if not required_field:
        raise MxcontinualParseError("GROUND_TRUTH: REQUIRED_FIELD is required")
    if window_days <= 0:
        raise MxcontinualParseError("GROUND_TRUTH: WINDOW_DAYS must be positive")

    return GroundTruthSpec(
        window_days=window_days,
        required_field=required_field,
        label_type=label_type,
        sources=tuple(sources),
        file_watch_path=file_watch_path,
    )


def _parse_methods_block(lines: list[str]) -> dict[str, FeatureDriftMethodSpec]:
    methods: dict[str, FeatureDriftMethodSpec] = {}
    # Each line: "<feature>: <method> threshold=<float>"
    for line in lines:
        m = re.match(r"^(\w+):\s+(\w+)\s+threshold=([0-9.]+)$", line)
        if not m:
            raise MxcontinualParseError(
                f"METHODS: invalid line {line!r}; expected 'feature: method threshold=N'"
            )
        feature, method, threshold = m.group(1), m.group(2), m.group(3)
        if method not in VALID_DRIFT_METHODS:
            raise MxcontinualParseError(
                f"METHODS: unknown drift method {method!r}; "
                f"valid: {sorted(VALID_DRIFT_METHODS)}"
            )
        methods[feature] = FeatureDriftMethodSpec(
            method=method,
            threshold=float(threshold),
        )
    return methods


def _parse_drift_detection(lines: list[str]) -> DriftDetectionSpec:
    features: list[str] = []
    methods: dict[str, FeatureDriftMethodSpec] = {}
    min_samples: int = 100
    check_frequency: str = "daily"
    reference_dataset: str = "base_training"

    i = 0
    while i < len(lines):
        k, v = _kv(lines[i])
        if k == "FEATURES":
            features = _parse_list(v, "FEATURES")
        elif k == "METHODS":
            block, i = _extract_block(lines, i + 1)
            methods = _parse_methods_block(block)
            continue
        elif k == "MIN_SAMPLES":
            min_samples = _int(v, "MIN_SAMPLES")
        elif k == "CHECK_FREQUENCY":
            check_frequency = v.strip()
        elif k == "REFERENCE_DATASET":
            reference_dataset = v.strip()
        i += 1

    if not features:
        raise MxcontinualParseError("DRIFT_DETECTION: FEATURES is required")
    if min_samples <= 0:
        raise MxcontinualParseError("DRIFT_DETECTION: MIN_SAMPLES must be positive")
    if min_samples < 50:
        pass  # warning (not fatal per spec)

    # Validate declared methods reference declared features
    for feat in methods:
        if feat not in features:
            raise MxcontinualParseError(
                f"DRIFT_DETECTION: method defined for feature {feat!r} "
                f"which is not in FEATURES list"
            )

    return DriftDetectionSpec(
        features=tuple(features),
        methods=methods,
        min_samples=min_samples,
        check_frequency=check_frequency,
        reference_dataset=reference_dataset,
    )


def _parse_concept_drift(lines: list[str]) -> ConceptDriftSpec:
    prediction_metric: str | None = None
    reference_value: float | None = None
    threshold_degradation: float | None = None
    min_samples_with_label: int = 100

    for line in lines:
        k, v = _kv(line)
        if k == "PREDICTION_METRIC":
            prediction_metric = v.strip()
        elif k == "REFERENCE_VALUE":
            reference_value = _float(v, "REFERENCE_VALUE")
        elif k == "THRESHOLD_DEGRADATION":
            threshold_degradation = _float(v, "THRESHOLD_DEGRADATION")
        elif k == "MIN_SAMPLES_WITH_LABEL":
            min_samples_with_label = _int(v, "MIN_SAMPLES_WITH_LABEL")

    if not prediction_metric:
        raise MxcontinualParseError("CONCEPT_DRIFT: PREDICTION_METRIC is required")
    if reference_value is None:
        raise MxcontinualParseError("CONCEPT_DRIFT: REFERENCE_VALUE is required")
    if threshold_degradation is None:
        raise MxcontinualParseError("CONCEPT_DRIFT: THRESHOLD_DEGRADATION is required")
    if threshold_degradation <= 0:
        raise MxcontinualParseError(
            f"CONCEPT_DRIFT: THRESHOLD_DEGRADATION must be positive, got {threshold_degradation}"
        )
    if threshold_degradation >= reference_value:
        raise MxcontinualParseError(
            f"CONCEPT_DRIFT: THRESHOLD_DEGRADATION ({threshold_degradation}) must be less than "
            f"REFERENCE_VALUE ({reference_value}); the detector would never trigger"
        )

    return ConceptDriftSpec(
        prediction_metric=prediction_metric,
        reference_value=reference_value,
        threshold_degradation=threshold_degradation,
        min_samples_with_label=min_samples_with_label,
    )


def _parse_update_trigger(lines: list[str]) -> UpdateTriggerSpec:
    triggers: list[str] = []
    min_new_samples: int = 100
    min_ground_truth_ratio: float = 0.5
    cooldown_days: int = 1

    i = 0
    while i < len(lines):
        k, v = _kv(lines[i])
        if k == "ANY_OF":
            block, i = _extract_block(lines, i + 1)
            for bline in block:
                bk, bv = _kv(bline)
                if bk == "SCHEDULED":
                    triggers.append(f"scheduled_{bv.strip()}")
                elif bk in ("DRIFT_THRESHOLD_EXCEEDED", "CONCEPT_DRIFT_DETECTED"):
                    triggers.append(bk.lower())
            continue
        elif k == "MIN_NEW_SAMPLES":
            min_new_samples = _int(v, "MIN_NEW_SAMPLES")
        elif k == "MIN_GROUND_TRUTH_RATIO":
            min_ground_truth_ratio = _float(v, "MIN_GROUND_TRUTH_RATIO")
        elif k == "COOLDOWN_DAYS":
            cooldown_days = _int(v, "COOLDOWN_DAYS")
        elif k in ("DRIFT_THRESHOLD_EXCEEDED", "CONCEPT_DRIFT_DETECTED"):
            triggers.append(k.lower())
        elif k == "SCHEDULED":
            triggers.append(f"scheduled_{v.strip()}")
        i += 1

    if min_new_samples <= 0:
        raise MxcontinualParseError("UPDATE_TRIGGER: MIN_NEW_SAMPLES must be positive")
    if not (0.0 < min_ground_truth_ratio <= 1.0):
        raise MxcontinualParseError(
            "UPDATE_TRIGGER: MIN_GROUND_TRUTH_RATIO must be in (0, 1]"
        )

    return UpdateTriggerSpec(
        triggers=tuple(triggers),
        min_new_samples=min_new_samples,
        min_ground_truth_ratio=min_ground_truth_ratio,
        cooldown_days=cooldown_days,
    )


def _parse_recency_decay(value: str) -> RecencyDecaySpec:
    """Parse 'exponential half_life_days=30' or 'linear'."""
    parts = value.strip().split()
    method = parts[0].lower()
    if method not in ("exponential", "linear"):
        raise MxcontinualParseError(
            f"RECENCY_DECAY: unknown method {method!r}; expected exponential or linear"
        )
    half_life_days: int | None = None
    for part in parts[1:]:
        if part.startswith("half_life_days="):
            half_life_days = int(part.split("=", 1)[1])
    return RecencyDecaySpec(method=method, half_life_days=half_life_days)


def _parse_dataset_mix(lines: list[str]) -> DatasetMixSpec:
    base_weight: float | None = None
    production_weight: float | None = None
    recency_decay: RecencyDecaySpec = RecencyDecaySpec(method="linear", half_life_days=None)

    for line in lines:
        k, v = _kv(line)
        if k == "BASE_WEIGHT":
            base_weight = _float(v, "BASE_WEIGHT")
        elif k == "PRODUCTION_WEIGHT":
            production_weight = _float(v, "PRODUCTION_WEIGHT")
        elif k == "RECENCY_DECAY":
            recency_decay = _parse_recency_decay(v)

    if base_weight is None:
        raise MxcontinualParseError("DATASET_MIX: BASE_WEIGHT is required")
    if production_weight is None:
        raise MxcontinualParseError("DATASET_MIX: PRODUCTION_WEIGHT is required")

    total = round(base_weight + production_weight, 6)
    if abs(total - 1.0) > 1e-5:
        raise MxcontinualParseError(
            f"DATASET_MIX: BASE_WEIGHT + PRODUCTION_WEIGHT must sum to 1.0, got {total}"
        )

    return DatasetMixSpec(
        base_weight=base_weight,
        production_weight=production_weight,
        recency_decay=recency_decay,
    )


def _parse_early_stop(value: str) -> EarlyStopSpec:
    """Parse 'patience=3 metric=validation_loss'."""
    patience: int | None = None
    metric: str | None = None
    for part in value.strip().split():
        if part.startswith("patience="):
            patience = int(part.split("=", 1)[1])
        elif part.startswith("metric="):
            metric = part.split("=", 1)[1]
    if patience is None or metric is None:
        raise MxcontinualParseError(
            f"EARLY_STOP: expected 'patience=N metric=M', got {value!r}"
        )
    return EarlyStopSpec(patience=patience, metric=metric)


_VALID_TRAINING_METHODS = frozenset({"incremental_finetune", "replay_buffer", "full_retrain"})


def _parse_training(lines: list[str]) -> ContinualTrainingSpec:
    method: str = "incremental_finetune"
    learning_rate_factor: float | None = None
    max_epochs: int = 20
    early_stop: EarlyStopSpec | None = None
    dataset_mix: DatasetMixSpec | None = None
    seed_inheritance: str = "deterministic_from_parent"

    i = 0
    while i < len(lines):
        k, v = _kv(lines[i])
        if k == "METHOD":
            method = v.strip()
        elif k == "LEARNING_RATE_FACTOR":
            learning_rate_factor = _float(v, "LEARNING_RATE_FACTOR")
        elif k == "MAX_EPOCHS":
            max_epochs = _int(v, "MAX_EPOCHS")
        elif k == "EARLY_STOP":
            early_stop = _parse_early_stop(v)
        elif k == "DATASET_MIX":
            block, i = _extract_block(lines, i + 1)
            dataset_mix = _parse_dataset_mix(block)
            continue
        elif k == "SEED_INHERITANCE":
            seed_inheritance = v.strip()
        i += 1

    if method not in _VALID_TRAINING_METHODS:
        raise MxcontinualParseError(
            f"TRAINING: unknown method {method!r}; valid: {sorted(_VALID_TRAINING_METHODS)}"
        )
    if learning_rate_factor is None:
        raise MxcontinualParseError("TRAINING: LEARNING_RATE_FACTOR is required")
    if not (0.01 <= learning_rate_factor <= 1.0):
        raise MxcontinualParseError(
            f"TRAINING: LEARNING_RATE_FACTOR must be in [0.01, 1.0], got {learning_rate_factor}"
        )
    if dataset_mix is None:
        raise MxcontinualParseError("TRAINING: DATASET_MIX block is required")

    return ContinualTrainingSpec(
        method=method,
        learning_rate_factor=learning_rate_factor,
        max_epochs=max_epochs,
        early_stop=early_stop,
        dataset_mix=dataset_mix,
        seed_inheritance=seed_inheritance,
    )


def _parse_regression_guard(lines: list[str]) -> RegressionGuardSpec:
    metric: str | None = None
    must_improve_by: float = 0.0
    max_degradation_per_label: float = 0.05

    for line in lines:
        k, v = _kv(line)
        if k == "METRIC":
            metric = v.strip()
        elif k == "MUST_IMPROVE_BY":
            must_improve_by = _float(v, "MUST_IMPROVE_BY")
        elif k == "MAX_DEGRADATION_PER_LABEL":
            max_degradation_per_label = _float(v, "MAX_DEGRADATION_PER_LABEL")
        elif k == "MIN_DELTA":  # alias used in simplified syntax
            must_improve_by = _float(v, "MIN_DELTA")
        elif k == "MAX_DEGRADATION":  # alias
            max_degradation_per_label = _float(v, "MAX_DEGRADATION")

    if not metric:
        raise MxcontinualParseError("REGRESSION_GUARD: METRIC is required")

    return RegressionGuardSpec(
        metric=metric,
        must_improve_by=must_improve_by,
        max_degradation_per_label=max_degradation_per_label,
    )


def _parse_approval_gate(lines: list[str]) -> ApprovalGateSpec:
    holdout_fraction: float | None = None
    holdout_source: str = "production_recent"
    regression_guard: RegressionGuardSpec | None = None
    human_approval: bool = True
    approval_channel: str | None = None
    approval_timeout_hours: int | None = None

    i = 0
    while i < len(lines):
        k, v = _kv(lines[i])
        if k == "HOLDOUT_FRACTION":
            holdout_fraction = _float(v, "HOLDOUT_FRACTION")
        elif k == "HOLDOUT_SOURCE":
            holdout_source = v.strip()
        elif k == "REGRESSION_GUARD":
            block, i = _extract_block(lines, i + 1)
            regression_guard = _parse_regression_guard(block)
            continue
        elif k == "HUMAN_APPROVAL":
            human_approval = _bool(v, "HUMAN_APPROVAL")
        elif k == "APPROVAL_CHANNEL":
            approval_channel = v.strip()
        elif k == "APPROVAL_TIMEOUT_HOURS":
            approval_timeout_hours = _int(v, "APPROVAL_TIMEOUT_HOURS")
        i += 1

    if holdout_fraction is None:
        raise MxcontinualParseError("APPROVAL_GATE: HOLDOUT_FRACTION is required")
    if not (0.1 <= holdout_fraction <= 0.5):
        raise MxcontinualParseError(
            f"APPROVAL_GATE: HOLDOUT_FRACTION must be in [0.1, 0.5], got {holdout_fraction}"
        )
    if regression_guard is None:
        raise MxcontinualParseError("APPROVAL_GATE: REGRESSION_GUARD block is required")

    return ApprovalGateSpec(
        holdout_fraction=holdout_fraction,
        holdout_source=holdout_source,
        regression_guard=regression_guard,
        human_approval=human_approval,
        approval_channel=approval_channel,
        approval_timeout_hours=approval_timeout_hours,
    )


def _parse_rollback(lines: list[str]) -> ContinualRollbackSpec:
    auto_trigger: bool = False
    metric: str | None = None
    sliding_window_hours: int = 24
    degradation_threshold: float | None = None
    min_samples_in_window: int = 50
    notify_capability: str | None = None
    notify_scope: dict[str, Any] = {}

    i = 0
    while i < len(lines):
        k, v = _kv(lines[i])
        if k == "AUTO_TRIGGER":
            auto_trigger = _bool(v, "AUTO_TRIGGER")
        elif k == "METRIC":
            metric = v.strip()
        elif k == "SLIDING_WINDOW_HOURS":
            sliding_window_hours = _int(v, "SLIDING_WINDOW_HOURS")
        elif k == "DEGRADATION_THRESHOLD":
            degradation_threshold = _float(v, "DEGRADATION_THRESHOLD")
        elif k == "MIN_SAMPLES_IN_WINDOW":
            min_samples_in_window = _int(v, "MIN_SAMPLES_IN_WINDOW")
        elif k == "NOTIFY_CAPABILITY":
            notify_capability = v.strip()
        elif k == "NOTIFY_SCOPE":
            block, i = _extract_block(lines, i + 1)
            notify_scope = _parse_scope_block(block)
            continue
        i += 1

    if not metric:
        raise MxcontinualParseError("ROLLBACK: METRIC is required")
    if degradation_threshold is None:
        raise MxcontinualParseError("ROLLBACK: DEGRADATION_THRESHOLD is required")
    if not (0.001 <= degradation_threshold <= 0.5):
        raise MxcontinualParseError(
            f"ROLLBACK: DEGRADATION_THRESHOLD must be in [0.001, 0.5], "
            f"got {degradation_threshold}"
        )

    return ContinualRollbackSpec(
        auto_trigger=auto_trigger,
        metric=metric,
        sliding_window_hours=sliding_window_hours,
        degradation_threshold=degradation_threshold,
        min_samples_in_window=min_samples_in_window,
        notify_capability=notify_capability,
        notify_scope=notify_scope,
    )


def _parse_audit(lines: list[str]) -> ContinualAuditSpec:
    persist_drift_reports: bool = True
    persist_update_traces: bool = True
    emit_refinement_hint: bool = False
    refinement_drift_persistence_days: int = 14
    signature_required: bool = False

    for line in lines:
        k, v = _kv(line)
        if k == "PERSIST_DRIFT_REPORTS":
            persist_drift_reports = _bool(v, "PERSIST_DRIFT_REPORTS")
        elif k == "PERSIST_UPDATE_TRACES":
            persist_update_traces = _bool(v, "PERSIST_UPDATE_TRACES")
        elif k == "EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT":
            emit_refinement_hint = _bool(v, "EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT")
        elif k == "REFINEMENT_DRIFT_PERSISTENCE_DAYS":
            refinement_drift_persistence_days = _int(v, "REFINEMENT_DRIFT_PERSISTENCE_DAYS")
        elif k == "SIGNATURE_REQUIRED":
            signature_required = _bool(v, "SIGNATURE_REQUIRED")

    return ContinualAuditSpec(
        persist_drift_reports=persist_drift_reports,
        persist_update_traces=persist_update_traces,
        emit_refinement_hint_on_sustained_drift=emit_refinement_hint,
        refinement_drift_persistence_days=refinement_drift_persistence_days,
        signature_required=signature_required,
    )


# ── top-level parser ──────────────────────────────────────────────────────────

_POLICY_RE = re.compile(r"^CONTINUAL_POLICY\s+(\S+)$")
_REQUIRED_BLOCKS = {"GROUND_TRUTH", "DRIFT_DETECTION", "UPDATE_TRIGGER", "TRAINING",
                    "APPROVAL_GATE", "ROLLBACK", "AUDIT"}


def parse_mxcontinual(source: str) -> ContinualPolicySpec:
    """Parse a .mxcontinual source string into ContinualPolicySpec."""
    lines = _clean_lines(source)
    if not lines:
        raise MxcontinualParseError("Empty .mxcontinual source")

    # Find top-level CONTINUAL_POLICY block
    m = _POLICY_RE.match(lines[0])
    if not m:
        raise MxcontinualParseError(
            f"Expected 'CONTINUAL_POLICY <name>', got {lines[0]!r}"
        )
    name = m.group(1)

    # Collect body lines until matching END
    body, _ = _extract_block(lines, 1)

    # Parse top-level key-value pairs and sub-blocks
    target_model: str | None = None
    base_parameter_set: str | None = None
    registry_name: str | None = None
    base_version: str | None = None

    ground_truth: GroundTruthSpec | None = None
    drift_detection: DriftDetectionSpec | None = None
    concept_drift: ConceptDriftSpec | None = None
    update_trigger: UpdateTriggerSpec | None = None
    training: ContinualTrainingSpec | None = None
    approval_gate: ApprovalGateSpec | None = None
    rollback: ContinualRollbackSpec | None = None
    audit: ContinualAuditSpec | None = None

    i = 0
    while i < len(body):
        k, v = _kv(body[i])
        if k == "TARGET_MODEL":
            target_model = v.strip()
        elif k == "BASE_PARAMETER_SET":
            base_parameter_set = v.strip()
        elif k == "REGISTRY_NAME":
            registry_name = v.strip()
        elif k == "BASE_VERSION":
            base_version = v.strip()
        elif k == "GROUND_TRUTH":
            block, i = _extract_block(body, i + 1)
            ground_truth = _parse_ground_truth(block)
            continue
        elif k == "DRIFT_DETECTION":
            block, i = _extract_block(body, i + 1)
            drift_detection = _parse_drift_detection(block)
            continue
        elif k == "CONCEPT_DRIFT":
            block, i = _extract_block(body, i + 1)
            concept_drift = _parse_concept_drift(block)
            continue
        elif k == "UPDATE_TRIGGER":
            block, i = _extract_block(body, i + 1)
            update_trigger = _parse_update_trigger(block)
            continue
        elif k == "TRAINING":
            block, i = _extract_block(body, i + 1)
            training = _parse_training(block)
            continue
        elif k == "APPROVAL_GATE":
            block, i = _extract_block(body, i + 1)
            approval_gate = _parse_approval_gate(block)
            continue
        elif k == "ROLLBACK":
            block, i = _extract_block(body, i + 1)
            rollback = _parse_rollback(block)
            continue
        elif k == "AUDIT":
            block, i = _extract_block(body, i + 1)
            audit = _parse_audit(block)
            continue
        i += 1

    # Validate required fields
    if not target_model:
        raise MxcontinualParseError("CONTINUAL_POLICY: TARGET_MODEL is required")
    if not base_parameter_set:
        raise MxcontinualParseError("CONTINUAL_POLICY: BASE_PARAMETER_SET is required")
    if ground_truth is None:
        raise MxcontinualParseError("CONTINUAL_POLICY: GROUND_TRUTH block is required")
    if drift_detection is None:
        raise MxcontinualParseError("CONTINUAL_POLICY: DRIFT_DETECTION block is required")
    if update_trigger is None:
        raise MxcontinualParseError("CONTINUAL_POLICY: UPDATE_TRIGGER block is required")
    if training is None:
        raise MxcontinualParseError("CONTINUAL_POLICY: TRAINING block is required")
    if approval_gate is None:
        raise MxcontinualParseError("CONTINUAL_POLICY: APPROVAL_GATE block is required")
    if rollback is None:
        raise MxcontinualParseError("CONTINUAL_POLICY: ROLLBACK block is required")
    if audit is None:
        raise MxcontinualParseError("CONTINUAL_POLICY: AUDIT block is required")

    if registry_name and not base_version:
        raise MxcontinualParseError(
            "CONTINUAL_POLICY: BASE_VERSION is required when REGISTRY_NAME is declared"
        )

    # Assemble without hash first, then compute and attach
    spec_no_hash = ContinualPolicySpec(
        name=name,
        target_model=target_model,
        base_parameter_set=base_parameter_set,
        registry_name=registry_name,
        base_version=base_version,
        ground_truth=ground_truth,
        drift_detection=drift_detection,
        concept_drift=concept_drift,
        update_trigger=update_trigger,
        training=training,
        approval_gate=approval_gate,
        rollback=rollback,
        audit=audit,
        policy_hash="",
    )
    ph = compute_policy_hash(spec_no_hash)
    # Return with hash — rebuild via dataclass replace pattern
    import dataclasses
    return dataclasses.replace(spec_no_hash, policy_hash=ph)
