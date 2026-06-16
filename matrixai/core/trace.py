# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Trace data structures for MatrixAI expression evaluation.

Every call to :meth:`~matrixai.core.evaluator.Evaluator.eval_definition` or
:meth:`~matrixai.core.evaluator.Evaluator.call` returns an
:class:`EvalTrace` that records:

- The function/variable name (``node``).
- The expression as a string (``expression``).
- The input environment (``inputs``).
- A flat list of :class:`EvalStep` entries — one per function call or
  sub-expression evaluated during the traversal.
- The final scalar output (``output``).

Example JSON output::

    {
      "node": "score",
      "expression": "((0.6 * relevance(x)) + (0.4 * coherence(x)))",
      "inputs": {"x": {"relevance": 0.9, "coherence": 0.8}},
      "steps": [
        {"op": "relevance", "args": [...], "result": 0.9},
        {"op": "coherence", "args": [...], "result": 0.8},
        {"op": "+",         "args": [0.54, 0.32], "result": 0.86}
      ],
      "output": 0.86
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalStep:
    """One recorded computation step."""

    op: str
    args: list = field(default_factory=list)
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "args": self.args, "result": self.result}


@dataclass
class EvalTrace:
    """Full trace of a single top-level definition evaluation."""

    node: str
    expression: str
    inputs: dict = field(default_factory=dict)
    steps: list[EvalStep] = field(default_factory=list)
    output: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "expression": self.expression,
            "inputs": _json_safe(self.inputs),
            "steps": [s.to_dict() for s in self.steps],
            "output": self.output,
        }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
