# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P22 C8 — ProductionMonitor: sliding-window online metrics with degradation detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from matrixai.ir.continual import ContinualPolicySpec


# ── value objects ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OnlineObservation:
    """A single production observation enriched with ground truth."""
    trace_id: str
    prediction: str
    ground_truth: str
    observed_at: str          # ISO8601 UTC
    parameter_set_id: str
    correct: bool             # prediction == ground_truth


@dataclass(frozen=True)
class WindowMetrics:
    """Accuracy and per-label metrics computed over a sliding window."""
    accuracy: float
    samples: int
    window_hours: int
    window_start: str         # ISO8601 UTC — earliest accepted timestamp
    window_end: str           # ISO8601 UTC — now
    per_label: dict[str, dict[str, float]] = field(default_factory=dict, hash=False)
    parameter_set_ids: list[str] = field(default_factory=list, hash=False)
    enough_samples: bool = False
    degradation_detected: bool = False
    reference_accuracy: float | None = None
    actual_degradation: float = 0.0


# ── monitor ───────────────────────────────────────────────────────────────────

class ProductionMonitor:
    """Records online (prediction + ground_truth) observations and computes
    sliding-window metrics for degradation detection.

    Configured via the ``ROLLBACK`` block of a :class:`ContinualPolicySpec`:

    - ``SLIDING_WINDOW_HOURS``: width of the rolling window.
    - ``MIN_SAMPLES_IN_WINDOW``: minimum observations before declaring
      degradation.
    - ``DEGRADATION_THRESHOLD``: how many accuracy points below the reference
      trigger a degradation event.
    - ``METRIC``: metric to monitor (currently ``accuracy``; extendable).

    Usage::

        monitor = ProductionMonitor(policy, reference_accuracy=0.91)
        monitor.record("spam", "spam", trace_id="t1", observed_at=ts1)
        monitor.record("ham",  "spam", trace_id="t2", observed_at=ts2)
        metrics = monitor.window_metrics(now=datetime.now(tz=timezone.utc))
        if metrics.degradation_detected:
            ...  # trigger rollback
    """

    def __init__(
        self,
        policy: ContinualPolicySpec,
        *,
        reference_accuracy: float | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self._policy = policy
        self._reference = reference_accuracy
        self._labels = list(labels or [])
        self._observations: list[OnlineObservation] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def record(
        self,
        prediction: str,
        ground_truth: str,
        *,
        trace_id: str = "",
        observed_at: datetime | str | None = None,
        parameter_set_id: str = "",
    ) -> OnlineObservation:
        """Record a new (prediction, ground_truth) pair and return the observation."""
        ts = _to_utc_str(observed_at) if observed_at is not None else _now_utc_str()
        obs = OnlineObservation(
            trace_id=trace_id,
            prediction=prediction,
            ground_truth=ground_truth,
            observed_at=ts,
            parameter_set_id=parameter_set_id,
            correct=(prediction == ground_truth),
        )
        self._observations.append(obs)
        return obs

    def window_metrics(self, now: datetime | None = None) -> WindowMetrics:
        """Compute metrics over observations within the sliding window ending at *now*."""
        now_dt = _coerce_utc(now) if now is not None else datetime.now(tz=timezone.utc)
        rb = self._policy.rollback
        window_hours = rb.sliding_window_hours
        min_samples = rb.min_samples_in_window
        threshold = rb.degradation_threshold

        window_start_dt = now_dt - timedelta(hours=window_hours)
        in_window = [
            obs for obs in self._observations
            if _parse_utc(obs.observed_at) >= window_start_dt
        ]

        n = len(in_window)
        accuracy = sum(1 for o in in_window if o.correct) / n if n > 0 else 0.0
        per_label = _per_label_metrics(in_window, self._labels)
        ps_ids = _deduplicated([o.parameter_set_id for o in in_window if o.parameter_set_id])
        enough = n >= min_samples

        # Degradation detection
        actual_deg = 0.0
        degradation = False
        if self._reference is not None and enough:
            actual_deg = self._reference - accuracy
            degradation = actual_deg > threshold

        return WindowMetrics(
            accuracy=accuracy,
            samples=n,
            window_hours=window_hours,
            window_start=window_start_dt.isoformat(),
            window_end=now_dt.isoformat(),
            per_label=per_label,
            parameter_set_ids=ps_ids,
            enough_samples=enough,
            degradation_detected=degradation,
            reference_accuracy=self._reference,
            actual_degradation=actual_deg,
        )

    def all_observations(self) -> list[OnlineObservation]:
        """Return all recorded observations in insertion order."""
        return list(self._observations)

    def observations_in_window(self, now: datetime | None = None) -> list[OnlineObservation]:
        """Return observations that fall within the current sliding window."""
        now_dt = _coerce_utc(now) if now is not None else datetime.now(tz=timezone.utc)
        window_start = now_dt - timedelta(hours=self._policy.rollback.sliding_window_hours)
        return [o for o in self._observations if _parse_utc(o.observed_at) >= window_start]

    def clear(self) -> None:
        """Remove all recorded observations (useful for testing or reset)."""
        self._observations.clear()

    def set_reference_accuracy(self, value: float) -> None:
        """Update the reference accuracy used for degradation comparisons."""
        self._reference = value


# ── helpers ────────────────────────────────────────────────────────────────────

def _now_utc_str() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_utc_str(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    return _coerce_utc(value).isoformat()


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _per_label_metrics(
    observations: list[OnlineObservation],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    if not observations or not labels:
        return {}

    # Build confusion per label
    all_labels = labels or sorted({o.ground_truth for o in observations})
    confusion: dict[str, dict[str, int]] = {l: {p: 0 for p in all_labels} for l in all_labels}
    for obs in observations:
        actual = obs.ground_truth
        pred = obs.prediction
        if actual not in confusion:
            confusion[actual] = {p: 0 for p in all_labels}
        confusion[actual][pred] = confusion[actual].get(pred, 0) + 1

    metrics: dict[str, dict[str, float]] = {}
    for label in all_labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(a, {}).get(label, 0) for a in all_labels if a != label)
        fn = sum(confusion.get(label, {}).get(p, 0) for p in all_labels if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion.get(label, {}).values())
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return metrics


def _deduplicated(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
