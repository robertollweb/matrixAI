# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""AST node definitions for the MatrixAI mini-language.

Each node represents one structural element of a MatrixAI expression or
statement.  All nodes expose:
  - ``eval(env)``    — evaluate to a Python float given a variable environment.
  - ``to_dict()``    — serialise to a JSON-safe dict.
  - ``__str__()``    — human-readable infix representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matrixai.types import TypeSpec


@dataclass
class NumberNode:
    """Numeric literal: ``0.6``, ``3.14``, ``100``."""

    value: float

    def eval(self, env: dict[str, Any]) -> float:  # noqa: ARG002
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "number", "value": self.value}

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class VarNode:
    """Variable reference: ``x``, ``score``, ``Confidence.max``."""

    name: str

    def eval(self, env: dict[str, Any]) -> float:
        parts = self.name.split(".", 1)
        value: Any = env.get(parts[0])
        if value is None:
            raise KeyError(f"Undefined variable: {self.name!r}")
        if len(parts) == 2:
            value = value[parts[1]] if isinstance(value, dict) else getattr(value, parts[1])
        if isinstance(value, dict):
            return float(max(value.values()))
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "var", "name": self.name}

    def __str__(self) -> str:
        return self.name


@dataclass
class BinaryOpNode:
    """Binary operation: ``left op right``  (op ∈ {+, -, *, /})."""

    op: str
    left: Any
    right: Any

    def eval(self, env: dict[str, Any]) -> float:
        l = self.left.eval(env)
        r = self.right.eval(env)
        if self.op == "+":
            return l + r
        if self.op == "-":
            return l - r
        if self.op == "*":
            return l * r
        if self.op == "/":
            if r == 0.0:
                raise ZeroDivisionError("Division by zero in MatrixAI expression")
            return l / r
        raise ValueError(f"Unknown operator: {self.op!r}")

    def to_dict(self) -> dict[str, Any]:
        _OP_TYPE = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
        return {
            "type": _OP_TYPE.get(self.op, self.op),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class CallNode:
    """Function call: ``relevance(x)``, ``normalize(score(x))``.

    Arguments are positional AST nodes.  Resolution order (in the evaluator):
    1. User-defined functions (``AssignNode`` definitions).
    2. Registry (builtin/registered callables).
    """

    name: str
    args: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "call",
            "name": self.name,
            "args": [a.to_dict() for a in self.args],
        }

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


@dataclass
class AssignNode:
    """Assignment / function definition.

    Examples::

        score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)
        decision = argmax(candidates)
    """

    name: str
    params: list  # list[str]
    expr: Any
    param_types: dict[str, TypeSpec] = field(default_factory=dict)
    return_type: TypeSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "assign",
            "name": self.name,
            "params": list(self.params),
            "expr": self.expr.to_dict(),
        }
        if self.param_types:
            data["param_types"] = {
                name: spec.to_dict() for name, spec in self.param_types.items()
            }
        if self.return_type is not None:
            data["return_type"] = self.return_type.to_dict()
        return data

    def __str__(self) -> str:
        params = []
        for param in self.params:
            if param in self.param_types:
                params.append(f"{param}: {self.param_types[param].name}")
            else:
                params.append(param)
        return_type = f" -> {self.return_type.name}" if self.return_type else ""
        if self.params:
            return f"{self.name}({', '.join(params)}){return_type} = {self.expr}"
        return f"{self.name}{return_type} = {self.expr}"
