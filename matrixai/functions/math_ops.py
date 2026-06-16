# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Block A — basic mathematical operations.

All functions accept plain Python numbers (int or float).
``min_``, ``max_``, ``mean``, ``sum_`` also accept a single iterable.
"""
from __future__ import annotations

import math as _math


def add(a: float, b: float) -> float:
    return float(a) + float(b)


def sub(a: float, b: float) -> float:
    return float(a) - float(b)


def mul(a: float, b: float) -> float:
    return float(a) * float(b)


def div(a: float, b: float) -> float:
    b = float(b)
    if b == 0.0:
        raise ZeroDivisionError("Division by zero in MatrixAI div()")
    return float(a) / b


def pow_(base: float, exp: float) -> float:
    return float(base) ** float(exp)


def sqrt(x: float) -> float:
    return _math.sqrt(float(x))


def abs_(x: float) -> float:
    return abs(float(x))


def min_(*args) -> float:
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return float(min(args[0]))
    return float(min(args))


def max_(*args) -> float:
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return float(max(args[0]))
    return float(max(args))


def mean(*args) -> float:
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        vals = list(args[0])
    else:
        vals = list(args)
    if not vals:
        raise ValueError("mean() requires at least one value")
    return sum(float(v) for v in vals) / len(vals)


def sum_(*args) -> float:
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return float(sum(args[0]))
    return float(sum(args))
