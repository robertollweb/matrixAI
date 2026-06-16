# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""MatrixAI functions — default library for the mathematical core.

Blocks
------
A — math_ops:   add, sub, mul, div, pow, sqrt, abs, min, max, mean, sum
B — transforms: normalize, clip, scale, softmax, sigmoid
C — scoring:    relevance, coherence, confidence, novelty, safety, quality
D — cost:       cost, latency, token_cost
E — selection:  argmax, topk, threshold, rank

Usage::

    from matrixai.functions import build_default_registry

    registry = build_default_registry()
    registry.register("my_fn", lambda x: x * 2)
"""
from .math_ops import abs_, add, div, max_, mean, min_, mul, pow_, sqrt, sub, sum_
from .scoring import (
    argmax,
    coherence,
    confidence,
    cost,
    latency,
    novelty,
    quality,
    rank,
    relevance,
    safety,
    threshold,
    token_cost,
    topk,
)
from .transforms import clip, normalize, scale, sigmoid, softmax


def build_default_registry():
    """Return a :class:`~matrixai.core.FunctionRegistry` pre-loaded with all built-in functions."""
    from matrixai.core import FunctionRegistry  # imported here to avoid circular import

    reg = FunctionRegistry()
    reg.register_many(
        {
            # Block A
            "add": add,
            "sub": sub,
            "mul": mul,
            "div": div,
            "pow": pow_,
            "sqrt": sqrt,
            "abs": abs_,
            "min": min_,
            "max": max_,
            "mean": mean,
            "sum": sum_,
            # Block B
            "normalize": normalize,
            "clip": clip,
            "scale": scale,
            "softmax": softmax,
            "sigmoid": sigmoid,
            # Block C
            "relevance": relevance,
            "coherence": coherence,
            "confidence": confidence,
            "novelty": novelty,
            "safety": safety,
            "quality": quality,
            # Block D
            "cost": cost,
            "latency": latency,
            "token_cost": token_cost,
            # Block E
            "argmax": argmax,
            "topk": topk,
            "threshold": threshold,
            "rank": rank,
        }
    )
    return reg


__all__ = [
    "abs_",
    "add",
    "argmax",
    "build_default_registry",
    "clip",
    "coherence",
    "confidence",
    "cost",
    "div",
    "latency",
    "max_",
    "mean",
    "min_",
    "mul",
    "normalize",
    "novelty",
    "pow_",
    "quality",
    "rank",
    "relevance",
    "safety",
    "scale",
    "sigmoid",
    "softmax",
    "sqrt",
    "sub",
    "sum_",
    "threshold",
    "token_cost",
    "topk",
]
