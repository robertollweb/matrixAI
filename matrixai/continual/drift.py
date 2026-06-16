# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from matrixai.ir.continual import ContinualPolicySpec, VALID_DRIFT_METHODS


# ── statistical functions ─────────────────────────────────────────────────────

def compute_psi(
    reference: list[float],
    observed: list[float],
    bins: int = 10,
) -> float:
    """
    Population Stability Index.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    Uses the reference distribution to define bin edges.
    """
    if not reference or not observed:
        return 0.0
    min_val = min(reference)
    max_val = max(reference)
    if min_val == max_val:
        return 0.0

    span = max_val - min_val
    n_ref = len(reference)
    n_obs = len(observed)

    def _bin_counts(data: list[float]) -> list[int]:
        counts = [0] * bins
        for v in data:
            # Clamp both sides: values below min_val map to bin 0 (not negative),
            # values above max_val map to the last bin.
            idx = max(0, min(int((v - min_val) / span * bins), bins - 1))
            counts[idx] += 1
        return counts

    ref_counts = _bin_counts(reference)
    obs_counts = _bin_counts(observed)

    psi = 0.0
    for rc, oc in zip(ref_counts, obs_counts):
        ref_pct = max(rc / n_ref, 1e-10)
        obs_pct = max(oc / n_obs, 1e-10)
        psi += (obs_pct - ref_pct) * math.log(obs_pct / ref_pct)
    return psi


def compute_ks_statistic(
    reference: list[float],
    observed: list[float],
) -> float:
    """
    Kolmogorov-Smirnov statistic: max |CDF_ref(x) - CDF_obs(x)|.

    Returns a value in [0, 1].
    """
    if not reference or not observed:
        return 0.0

    n_ref = len(reference)
    n_obs = len(observed)
    ref_sorted = sorted(reference)
    obs_sorted = sorted(observed)

    # Evaluate both CDFs at every observed value from either distribution.
    all_vals = sorted(set(ref_sorted + obs_sorted))
    max_diff = 0.0
    ri = oi = 0
    cdf_ref = cdf_obs = 0.0

    for val in all_vals:
        while ri < n_ref and ref_sorted[ri] <= val:
            ri += 1
        while oi < n_obs and obs_sorted[oi] <= val:
            oi += 1
        cdf_ref = ri / n_ref
        cdf_obs = oi / n_obs
        max_diff = max(max_diff, abs(cdf_ref - cdf_obs))

    return max_diff


def compute_chi_square(
    reference_counts: dict[str, int],
    observed_counts: dict[str, int],
) -> float:
    """
    Chi-square statistic for categorical distributions.

    χ² = Σ (observed - expected)² / expected
    where expected is derived from the reference proportion × observed total.
    """
    categories = set(reference_counts) | set(observed_counts)
    n_ref = sum(reference_counts.values())
    n_obs = sum(observed_counts.values())

    if n_ref == 0 or n_obs == 0:
        return 0.0

    chi2 = 0.0
    for cat in categories:
        expected = (reference_counts.get(cat, 0) / n_ref) * n_obs
        actual = observed_counts.get(cat, 0)
        if expected > 0:
            chi2 += (actual - expected) ** 2 / expected
    return chi2


MAX_CHI_SQUARE_CATEGORIES = 50


def compute_chi_square_from_samples(
    reference: list[float],
    observed: list[float],
) -> float:
    """Chi-square treating float values as categorical codes (int-rounded).

    Raises ValueError when the reference data yields more than
    MAX_CHI_SQUARE_CATEGORIES unique rounded values — a sign that the feature
    is continuous and should use 'psi' or 'ks' instead.
    """
    ref_counts: dict[str, int] = {}
    for v in reference:
        k = str(round(v))
        ref_counts[k] = ref_counts.get(k, 0) + 1

    if len(ref_counts) > MAX_CHI_SQUARE_CATEGORIES:
        raise ValueError(
            f"chi_square: reference data has {len(ref_counts)} unique rounded values "
            f"(limit {MAX_CHI_SQUARE_CATEGORIES}). "
            "Continuous features should declare 'psi' or 'ks' instead."
        )

    obs_counts: dict[str, int] = {}
    for v in observed:
        k = str(round(v))
        obs_counts[k] = obs_counts.get(k, 0) + 1
    return compute_chi_square(ref_counts, obs_counts)


def compute_js_divergence(
    reference: list[float],
    observed: list[float],
    bins: int = 20,
) -> float:
    """
    Jensen-Shannon divergence (log2 base), bounded in [0, 1].

    Uses histogram discretization of continuous data.
    """
    if not reference or not observed:
        return 0.0

    min_val = min(min(reference), min(observed))
    max_val = max(max(reference), max(observed))
    if min_val == max_val:
        return 0.0

    span = max_val - min_val

    def _to_probs(data: list[float]) -> list[float]:
        counts = [0] * bins
        for v in data:
            idx = max(0, min(int((v - min_val) / span * bins), bins - 1))
            counts[idx] += 1
        total = len(data)
        return [c / total for c in counts]

    p = _to_probs(reference)
    q = _to_probs(observed)
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

    def _kl(dist: list[float], mix: list[float]) -> float:
        total = 0.0
        for di, mi in zip(dist, mix):
            if di > 0.0 and mi > 0.0:
                total += di * math.log2(di / mi)
        return total

    return min(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 1.0)


def compute_wasserstein(
    reference: list[float],
    observed: list[float],
) -> float:
    """
    Wasserstein-1 distance (Earth Mover's Distance) approximated via CDFs.

    W₁ ≈ Σ |CDF_ref(x) - CDF_obs(x)| × Δx
    Handles small samples robustly.
    """
    if not reference or not observed:
        return 0.0

    ref_sorted = sorted(reference)
    obs_sorted = sorted(observed)
    n_ref = len(ref_sorted)
    n_obs = len(obs_sorted)

    all_vals = sorted(set(ref_sorted + obs_sorted))
    if len(all_vals) < 2:
        return 0.0

    emd = 0.0
    ri = oi = 0

    for i in range(len(all_vals) - 1):
        val = all_vals[i]
        next_val = all_vals[i + 1]
        while ri < n_ref and ref_sorted[ri] <= val:
            ri += 1
        while oi < n_obs and obs_sorted[oi] <= val:
            oi += 1
        cdf_ref = ri / n_ref
        cdf_obs = oi / n_obs
        emd += abs(cdf_ref - cdf_obs) * (next_val - val)

    return emd


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    feature: str
    method: str
    observed_value: float
    threshold: float
    drift_detected: bool
    samples_used: int
    enough_samples: bool
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class DriftReport:
    policy_hash: str
    checked_at: str                         # ISO8601 UTC
    features_checked: list[str]
    results: dict[str, FeatureDriftResult]  # feature → result
    drift_detected: bool                    # True if ≥1 feature over threshold AND enough samples
    enough_samples: bool                    # True if ≥1 feature meets MIN_SAMPLES
    total_production_samples: int


# ── DriftDetector ─────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Computes feature drift between a reference distribution and production samples.

    Usage::

        detector = DriftDetector(policy)
        report = detector.run_check(reference_data, production_data)

    `reference_data` and `production_data` are ``dict[feature_name, list[float]]``.
    """

    def __init__(self, policy: ContinualPolicySpec) -> None:
        self._policy = policy

    def run_check(
        self,
        reference_data: dict[str, list[float]],
        production_data: dict[str, list[float]],
        now: datetime | None = None,
    ) -> DriftReport:
        """
        Run drift checks for all declared features.

        Returns a `DriftReport` with per-feature results and an overall verdict.
        `drift_detected` is True only when enough_samples is met AND at least one
        feature exceeds its threshold.
        """
        now = now or datetime.now(tz=timezone.utc)
        dd = self._policy.drift_detection

        # Summary count for reporting; per-feature gating happens below.
        production_counts = [
            len(production_data.get(f, []))
            for f in dd.features
            if f in production_data
        ]
        total_production = max(production_counts) if production_counts else 0
        enough_samples = any(count >= dd.min_samples for count in production_counts)

        results: dict[str, FeatureDriftResult] = {}

        for feature in dd.features:
            method_spec = dd.methods.get(feature)
            if method_spec is None:
                # Feature declared but no method — skip gracefully
                samples_used = len(production_data.get(feature, []))
                results[feature] = FeatureDriftResult(
                    feature=feature,
                    method="none",
                    observed_value=0.0,
                    threshold=0.0,
                    drift_detected=False,
                    samples_used=samples_used,
                    enough_samples=samples_used >= dd.min_samples,
                    skipped=True,
                    skip_reason="no method declared for feature",
                )
                continue

            ref = reference_data.get(feature, [])
            obs = production_data.get(feature, [])

            if not ref:
                feature_enough = len(obs) >= dd.min_samples
                results[feature] = FeatureDriftResult(
                    feature=feature,
                    method=method_spec.method,
                    observed_value=0.0,
                    threshold=method_spec.threshold,
                    drift_detected=False,
                    samples_used=len(obs),
                    enough_samples=feature_enough,
                    skipped=True,
                    skip_reason="no reference data for feature",
                )
                continue

            if not obs:
                results[feature] = FeatureDriftResult(
                    feature=feature,
                    method=method_spec.method,
                    observed_value=0.0,
                    threshold=method_spec.threshold,
                    drift_detected=False,
                    samples_used=0,
                    enough_samples=False,
                    skipped=True,
                    skip_reason="no production data for feature",
                )
                continue

            val = self._compute_metric(method_spec.method, ref, obs)
            # Per-feature sample guard: each feature must independently meet MIN_SAMPLES.
            # Using the global max would enable drift on a feature with 1 sample if another
            # feature happens to have enough samples.
            feature_enough = len(obs) >= dd.min_samples
            drift_detected = feature_enough and (val > method_spec.threshold)

            results[feature] = FeatureDriftResult(
                feature=feature,
                method=method_spec.method,
                observed_value=val,
                threshold=method_spec.threshold,
                drift_detected=drift_detected,
                samples_used=len(obs),
                enough_samples=feature_enough,
            )

        any_drift = any(r.drift_detected for r in results.values())

        return DriftReport(
            policy_hash=self._policy.policy_hash,
            checked_at=now.isoformat(),
            features_checked=list(dd.features),
            results=results,
            drift_detected=any_drift,
            enough_samples=enough_samples,
            total_production_samples=total_production,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _compute_metric(
        self,
        method: str,
        reference: list[float],
        observed: list[float],
    ) -> float:
        if method == "psi":
            return compute_psi(reference, observed)
        if method == "ks":
            return compute_ks_statistic(reference, observed)
        if method == "chi_square":
            return compute_chi_square_from_samples(reference, observed)
        if method == "js":
            return compute_js_divergence(reference, observed)
        if method == "wasserstein":
            return compute_wasserstein(reference, observed)
        raise ValueError(f"Unknown drift method {method!r}")


# ── ConceptDriftDetector ──────────────────────────────────────────────────────

@dataclass
class ConceptDriftReport:
    prediction_metric: str
    current_value: float
    reference_value: float
    threshold_degradation: float
    alert_threshold: float        # = reference_value - threshold_degradation
    concept_drift_detected: bool
    enough_labeled_samples: bool
    labeled_samples: int
    checked_at: str               # ISO8601 UTC


class ConceptDriftDetector:
    """
    Detects concept drift by comparing a current prediction metric value
    against the baseline declared in ConceptDriftSpec.

    Drift is flagged when:
        current_metric_value < reference_value - threshold_degradation
        AND labeled_sample_count >= min_samples_with_label

    **Assumes "higher is better"** — drift means the metric *fell* below the
    alert threshold.  Suitable for accuracy, f1, precision, recall, r2.
    Not applicable to loss-type metrics where lower values are better
    (e.g. mse, cross_entropy); for those, invert the metric or use a
    dedicated loss-drift detector.

    Usage::

        detector = ConceptDriftDetector(policy)
        report = detector.run_check(current_accuracy, labeled_count)
    """

    def __init__(self, policy: ContinualPolicySpec) -> None:
        if policy.concept_drift is None:
            raise ValueError(
                "ConceptDriftDetector requires a CONCEPT_DRIFT block in the policy"
            )
        self._policy = policy
        self._spec = policy.concept_drift

    def run_check(
        self,
        current_metric_value: float,
        labeled_sample_count: int,
        now: datetime | None = None,
    ) -> ConceptDriftReport:
        """
        Check whether the prediction metric has degraded beyond the declared threshold.

        Args:
            current_metric_value: Observed value of the prediction metric.
            labeled_sample_count: Number of samples with ground-truth labels.
            now: Override current UTC time (for testing).

        Returns:
            ConceptDriftReport.  ``concept_drift_detected`` is True only when
            ``labeled_sample_count >= min_samples_with_label`` AND the metric
            has fallen below ``reference_value - threshold_degradation``.
        """
        now = now or datetime.now(tz=timezone.utc)
        spec = self._spec
        alert_threshold = spec.reference_value - spec.threshold_degradation
        enough = labeled_sample_count >= spec.min_samples_with_label
        detected = enough and (current_metric_value < alert_threshold)

        return ConceptDriftReport(
            prediction_metric=spec.prediction_metric,
            current_value=current_metric_value,
            reference_value=spec.reference_value,
            threshold_degradation=spec.threshold_degradation,
            alert_threshold=alert_threshold,
            concept_drift_detected=detected,
            enough_labeled_samples=enough,
            labeled_samples=labeled_sample_count,
            checked_at=now.isoformat(),
        )
