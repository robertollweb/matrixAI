# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Block B — normalisation and probability transforms."""
from __future__ import annotations

import math as _math


def normalize(x: float) -> float:
    """Clip *x* to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def clip(x: float, lo: float, hi: float) -> float:
    """Clip *x* to [lo, hi]."""
    return max(float(lo), min(float(hi), float(x)))


def scale(
    x: float,
    old_min: float,
    old_max: float,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> float:
    """Linearly rescale *x* from [old_min, old_max] to [new_min, new_max]."""
    x, old_min, old_max = float(x), float(old_min), float(old_max)
    span = old_max - old_min
    if span == 0.0:
        return float(new_min)
    return float(new_min) + (x - old_min) / span * (float(new_max) - float(new_min))


def softmax(values) -> list[float]:
    """Stable softmax over an iterable of values."""
    vals = [float(v) for v in values]
    m = max(vals)
    exps = [_math.exp(v - m) for v in vals]
    total = sum(exps)
    return [e / total for e in exps]


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + _math.exp(-x))
    ex = _math.exp(x)
    return ex / (1.0 + ex)
