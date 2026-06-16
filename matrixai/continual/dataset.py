# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

from matrixai.ir.continual import DatasetMixSpec
from matrixai.training.data import SupervisedExample


class ContinualDataset:
    """
    Mixes base training examples with production examples using temporal recency
    decay applied to each production sample.

    - ``examples()`` returns base + production examples with non-zero decay weight,
      compatible with ``SupervisedTrainer.train()``.
    - ``weights()`` returns a parallel list of importance weights for weighted
      sampling.
    - ``fingerprint()`` is a deterministic hash of all inputs.
    """

    def __init__(
        self,
        base_examples: list[SupervisedExample],
        production_examples: list[SupervisedExample],
        production_timestamps: list[datetime],
        mix_spec: DatasetMixSpec,
        base_fingerprint: str,
        reference_time: datetime | None = None,
        window_days: int = 30,
    ) -> None:
        if len(production_examples) != len(production_timestamps):
            raise ValueError(
                f"production_examples ({len(production_examples)}) and "
                f"production_timestamps ({len(production_timestamps)}) must have the same length"
            )
        self._base = list(base_examples)
        self._production = list(production_examples)
        self._timestamps = list(production_timestamps)
        self._mix = mix_spec
        self._base_fingerprint = base_fingerprint
        self._reference_time = (
            reference_time
            if reference_time is not None
            else datetime.now(tz=timezone.utc)
        )
        self._window_days = window_days

    # ── public ────────────────────────────────────────────────────────────────

    def examples(self) -> list[SupervisedExample]:
        """Returns base + production examples that have non-zero decay weight."""
        result = list(self._base)
        if not self._production or self._mix.production_weight == 0.0:
            return result
        decay = self._decay_weights()
        for ex, w in zip(self._production, decay):
            if w > 0.0:
                result.append(ex)
        return result

    def weights(self) -> list[float]:
        """Importance weight for each example returned by ``examples()``."""
        n_base = max(len(self._base), 1)
        base_w = self._mix.base_weight / n_base
        result: list[float] = [base_w] * len(self._base)

        if not self._production or self._mix.production_weight == 0.0:
            return result

        decay = self._decay_weights()
        total_decay = sum(d for d in decay if d > 0.0)
        if total_decay == 0.0:
            return result

        for w in decay:
            if w > 0.0:
                result.append(self._mix.production_weight * w / total_decay)

        return result

    def fingerprint(self) -> str:
        decay_spec = self._mix.recency_decay
        decay_info: dict = {"method": decay_spec.method}
        if decay_spec.method == "exponential":
            decay_info["half_life_days"] = decay_spec.half_life_days
        else:
            decay_info["window_days"] = self._window_days

        payload = {
            "base": self._base_fingerprint,
            "production_hashes": [e.row_hash for e in self._production],
            # ISO timestamps determine which production examples have non-zero decay weight
            "production_timestamps": [_to_utc(ts).isoformat() for ts in self._timestamps],
            "mix": {
                "base_weight": self._mix.base_weight,
                "production_weight": self._mix.production_weight,
                "recency_decay": decay_info,
            },
            # Day-granularity reference date: captures decay calculation epoch without
            # making the fingerprint change every second when reference_time=None
            "reference_date": _to_utc(self._reference_time).date().isoformat(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"continual_{digest}"

    def base_count(self) -> int:
        return len(self._base)

    def production_count(self) -> int:
        return len(self._production)

    # ── private ───────────────────────────────────────────────────────────────

    def _decay_weights(self) -> list[float]:
        method = self._mix.recency_decay.method
        ref = _to_utc(self._reference_time)
        result: list[float] = []
        for ts in self._timestamps:
            age_days = max(0.0, (ref - _to_utc(ts)).total_seconds() / 86400.0)
            if method == "exponential":
                half_life = float(self._mix.recency_decay.half_life_days or 30)
                w = math.exp(-math.log(2) * age_days / half_life)
            else:  # linear
                w = max(0.0, 1.0 - age_days / self._window_days)
            result.append(w)
        return result


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
