# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Blocks C, D, E — scoring, cost and selection functions.

Scoring functions (Block C) and cost functions (Block D) accept either:
  - A plain float / int.
  - A dict with a matching key (e.g. ``{"relevance": 0.9, ...}``).

Selection functions (Block E) operate on lists of items.
"""
from __future__ import annotations

from typing import Callable


# ---------------------------------------------------------------------------
# Block C — Scoring
# ---------------------------------------------------------------------------

def _field(x, key: str, default: float = 0.0) -> float:
    if isinstance(x, dict):
        return float(x.get(key, default))
    return float(x)


def relevance(x) -> float:
    return _field(x, "relevance")


def coherence(x) -> float:
    return _field(x, "coherence")


def confidence(x) -> float:
    return _field(x, "confidence")


def novelty(x) -> float:
    return _field(x, "novelty")


def safety(x) -> float:
    return _field(x, "safety", default=1.0)


def quality(x) -> float:
    return _field(x, "quality")


# ---------------------------------------------------------------------------
# Block D — Cost
# ---------------------------------------------------------------------------

def cost(x) -> float:
    return _field(x, "cost")


def latency(x) -> float:
    return _field(x, "latency")


def token_cost(x) -> float:
    return _field(x, "token_cost")


# ---------------------------------------------------------------------------
# Block E — Selection
# ---------------------------------------------------------------------------

def argmax(items, score_fn: Callable | None = None) -> object:
    """Return the item in *items* with the highest score.

    If *score_fn* is omitted, items are compared directly.
    """
    if score_fn is None:
        return max(items)
    return max(items, key=score_fn)


def topk(items, score_fn: Callable, k: int) -> list:
    """Return the top-*k* items ranked by *score_fn* (descending)."""
    return sorted(items, key=score_fn, reverse=True)[: int(k)]


def threshold(items, score_fn: Callable, min_score: float) -> list:
    """Keep items whose score is >= *min_score*."""
    return [item for item in items if score_fn(item) >= float(min_score)]


def rank(items, score_fn: Callable) -> list:
    """Return items sorted by *score_fn* descending."""
    return sorted(items, key=score_fn, reverse=True)
