# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""MatrixAI expression evaluator.

The :class:`Evaluator` traverses an AST and computes a scalar result.  It
records an :class:`~matrixai.core.trace.EvalTrace` for every top-level call so
that every step is auditable.

Resolution order for :class:`~matrixai.core.ast_nodes.CallNode`
----------------------------------------------------------------
1. User-defined functions (``AssignNode`` registered via :meth:`define`).
2. Callables in the :class:`~matrixai.core.registry.FunctionRegistry`.

Both paths record a step in the current trace.
"""
from __future__ import annotations

from typing import Any

from .ast_nodes import AssignNode, BinaryOpNode, CallNode, NumberNode, VarNode
from .registry import FunctionRegistry
from .trace import EvalStep, EvalTrace


class EvaluationError(Exception):
    pass


class Evaluator:
    def __init__(self, registry: FunctionRegistry) -> None:
        self._registry = registry
        self._definitions: dict[str, AssignNode] = {}
        self._trace: EvalTrace | None = None

    # ------------------------------------------------------------------ define

    def define(self, node: AssignNode) -> None:
        """Register a user-defined function/constant from an :class:`AssignNode`."""
        self._definitions[node.name] = node

    def define_all(self, nodes: list[AssignNode]) -> None:
        for node in nodes:
            self.define(node)

    # ------------------------------------------------------------------ public

    def eval_definition(
        self, name: str, env: dict[str, Any]
    ) -> tuple[float, EvalTrace]:
        """Evaluate definition *name* with a flat environment *env*.

        If the definition has parameters they are bound from *env* by name.
        Returns ``(result, trace)``.
        """
        if name not in self._definitions:
            raise EvaluationError(f"Undefined: {name!r}")
        defn = self._definitions[name]

        # If params are declared, build local env by binding params from env
        local_env = dict(env)
        if defn.params:
            # If there's a single param and it's not directly in env,
            # pass the whole env dict as the param value.
            if len(defn.params) == 1 and defn.params[0] not in env:
                local_env[defn.params[0]] = env

        trace = EvalTrace(
            node=name,
            expression=str(defn.expr),
            inputs=_safe_inputs(local_env),
        )
        self._trace = trace
        result = self._eval(defn.expr, local_env)
        result = float(max(result.values())) if isinstance(result, dict) else float(result)
        trace.output = result
        self._trace = None
        return result, trace

    def call(
        self, name: str, args: list[Any], env: dict[str, Any] | None = None
    ) -> tuple[float, EvalTrace]:
        """Call a defined function by name with positional *args*.

        Returns ``(result, trace)``.
        """
        env = env or {}
        if name not in self._definitions:
            raise EvaluationError(f"Undefined function: {name!r}")
        defn = self._definitions[name]
        if len(args) != len(defn.params):
            raise EvaluationError(
                f"Function {name!r} expects {len(defn.params)} args, got {len(args)}"
            )
        local_env = dict(env)
        for param, val in zip(defn.params, args):
            local_env[param] = val

        trace = EvalTrace(
            node=name,
            expression=str(defn.expr),
            inputs=_safe_inputs(local_env),
        )
        self._trace = trace
        result = self._eval(defn.expr, local_env)
        result = float(max(result.values())) if isinstance(result, dict) else float(result)
        trace.output = result
        self._trace = None
        return result, trace

    # ------------------------------------------------------------------ internal

    def _eval(self, node: Any, env: dict[str, Any]) -> Any:
        """Evaluate *node*.  Returns the raw Python value (dict, float, list …).

        Arithmetic operators coerce operands to float explicitly, so non-scalar
        VarNode values (e.g. a dict passed as a function argument) are kept
        intact until they reach a function that knows how to handle them.
        """
        if isinstance(node, NumberNode):
            return node.value

        if isinstance(node, VarNode):
            if node.name in env:
                return env[node.name]  # raw — let the caller coerce
            # dotted name: Confidence.max
            parts = node.name.split(".", 1)
            if len(parts) == 2 and parts[0] in env:
                obj = env[parts[0]]
                value = obj[parts[1]] if isinstance(obj, dict) else getattr(obj, parts[1])
                return float(value)
            raise EvaluationError(f"Undefined variable: {node.name!r}")

        if isinstance(node, BinaryOpNode):
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            # Coerce to float for arithmetic (dicts become max of their values)
            left = float(max(left.values())) if isinstance(left, dict) else float(left)
            right = float(max(right.values())) if isinstance(right, dict) else float(right)
            result = _apply_op(node.op, left, right)
            if self._trace is not None:
                self._trace.steps.append(
                    EvalStep(op=node.op, args=[left, right], result=result)
                )
            return result

        if isinstance(node, CallNode):
            args = [self._eval(arg, env) for arg in node.args]
            name = node.name

            # 1. User-defined function
            if name in self._definitions:
                defn = self._definitions[name]
                local_env = dict(env)
                for param, val in zip(defn.params, args):
                    local_env[param] = val
                result = self._eval(defn.expr, local_env)
                if self._trace is not None:
                    self._trace.steps.append(
                        EvalStep(op=name, args=list(args), result=result)
                    )
                return result

            # 2. Registry function
            fn = self._registry.get(name)
            result = fn(*args)
            if self._trace is not None:
                self._trace.steps.append(
                    EvalStep(op=name, args=list(args), result=result)
                )
            return float(result)

        raise EvaluationError(f"Unknown AST node type: {type(node).__name__}")


# ------------------------------------------------------------------ helpers

def _apply_op(op: str, left: float, right: float) -> float:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        if right == 0.0:
            raise EvaluationError("Division by zero")
        return left / right
    raise EvaluationError(f"Unknown operator: {op!r}")


def _safe_inputs(env: dict) -> dict:
    """Return env with non-scalar values represented as their type name."""
    result = {}
    for k, v in env.items():
        if isinstance(v, (int, float, bool)):
            result[k] = float(v)
        elif isinstance(v, dict):
            result[k] = {str(ik): float(iv) for ik, iv in v.items() if isinstance(iv, (int, float))}
        else:
            result[k] = repr(v)
    return result
