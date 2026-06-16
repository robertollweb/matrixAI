# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import math
import re as _re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

class ExprNode(ABC):
    """Base class for symbolic expression AST nodes."""

    @abstractmethod
    def eval(self, env: dict[str, Any]) -> float:
        """Evaluate the expression given a variable environment."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable infix representation."""


@dataclass
class LiteralNode(ExprNode):
    """A numeric constant: 0.7, 1.0, etc."""

    value: float

    def eval(self, env: dict[str, Any]) -> float:
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "literal", "value": self.value}

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class VarNode(ExprNode):
    """A variable reference.  Name may be dotted: ``Confidence.max``."""

    name: str

    def eval(self, env: dict[str, Any]) -> float:
        parts = self.name.split(".")
        value: Any = env.get(parts[0])
        for part in parts[1:]:
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)
        if isinstance(value, dict):
            # dict node result — return max value (e.g. Categorical distribution)
            return float(max(value.values()))
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "var", "name": self.name}

    def __str__(self) -> str:
        return self.name


@dataclass
class CallNode(ExprNode):
    """A function call: ``normalize(x)``, ``sigmoid(x)``, ``relevance(x)``.

    If *func* is not a registered builtin the evaluator looks up *func* in the
    environment as a pre-computed value (a scoring oracle result already stored
    in state), ignoring the argument list.  This lets symbolic expressions like
    ``0.7 * relevance(x) + 0.3 * coherence(x)`` resolve correctly when
    ``state["relevance"]`` and ``state["coherence"]`` are available.
    """

    func: str
    args: list[ExprNode] = field(default_factory=list)

    def eval(self, env: dict[str, Any]) -> float:
        evaluated_args = [arg.eval(env) for arg in self.args]
        builtin = _BUILTIN_FUNCTIONS.get(self.func)
        if builtin is not None:
            return float(builtin(*evaluated_args))
        # Not a builtin — look up as a pre-computed state value
        value = env.get(self.func)
        if value is None:
            return 0.0
        if callable(value):
            return float(value(*evaluated_args))
        if isinstance(value, dict):
            return float(max(value.values()))
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "call", "func": self.func, "args": [a.to_dict() for a in self.args]}

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.func}({args_str})"


@dataclass
class BinOpNode(ExprNode):
    """A binary operation: left op right (op ∈ {+, -, *, /})."""

    op: str
    left: ExprNode
    right: ExprNode

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
                raise ZeroDivisionError("Division by zero in symbolic expression")
            return l / r
        raise ValueError(f"Unknown operator: {self.op!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "binop",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class WeightedSumNode(ExprNode):
    """Decomposed weighted sum: w1·f1 + w2·f2 + …

    This is a derived node produced by :func:`extract_weighted_sum`.  It
    carries the same semantics as the equivalent ``BinOpNode`` tree but in a
    flat, inspectable form that is easier to serialise and audit.
    """

    terms: list[tuple[float, ExprNode]] = field(default_factory=list)

    def eval(self, env: dict[str, Any]) -> float:
        return sum(w * expr.eval(env) for w, expr in self.terms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "weighted_sum",
            "terms": [{"weight": w, "expr": expr.to_dict()} for w, expr in self.terms],
        }

    def __str__(self) -> str:
        parts = []
        for w, expr in self.terms:
            if w == 1.0:
                parts.append(str(expr))
            else:
                parts.append(f"{w} * {expr}")
        return " + ".join(parts)


# ---------------------------------------------------------------------------
# Built-in scalar functions available inside expressions
# ---------------------------------------------------------------------------

def _normalize(x: float) -> float:
    """Clip to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _scale(x: float, x_min: float, x_max: float) -> float:
    x, x_min, x_max = float(x), float(x_min), float(x_max)
    if x_max == x_min:
        return 0.0
    return _clip((x - x_min) / (x_max - x_min))


def _sigmoid_product(*values: float) -> float:
    result = 1.0
    for value in values:
        result *= float(value)
    return result


def _sigmoid_or(*values: float) -> float:
    if len(values) == 1:
        return float(values[0])
    inactive = 1.0
    for value in values:
        inactive *= 1.0 - float(value)
    return 1.0 - inactive


_BUILTIN_FUNCTIONS: dict[str, Any] = {
    "normalize": _normalize,
    "sigmoid": _sigmoid,
    "clip": _clip,
    "scale": _scale,
    "abs": abs,
    "max": max,
    "min": min,
    "sigmoid_product": _sigmoid_product,
    "sigmoid_or": _sigmoid_or,
}

# Public set of known function names (for validation)
KNOWN_FUNCTIONS: frozenset[str] = frozenset(_BUILTIN_FUNCTIONS) | frozenset({
    "embed", "score", "aggregate", "select", "softmax",
    "relevance", "coherence", "confidence",
    # P10 tensor primitives
    "dot", "matmul", "relu", "gelu", "layer_norm", "residual",
    # P10 attention and embedding primitives
    "embedding_lookup", "positional_encoding", "attention", "mean_pooling", "cls_pooling",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_vars(node: ExprNode) -> list[str]:
    """Return all variable names referenced in the expression (depth-first)."""
    if isinstance(node, VarNode):
        return [node.name]
    if isinstance(node, LiteralNode):
        return []
    if isinstance(node, CallNode):
        result: list[str] = []
        for arg in node.args:
            result.extend(collect_vars(arg))
        return result
    if isinstance(node, BinOpNode):
        return collect_vars(node.left) + collect_vars(node.right)
    if isinstance(node, WeightedSumNode):
        result = []
        for _, expr in node.terms:
            result.extend(collect_vars(expr))
        return result
    return []


def extract_weighted_sum(node: ExprNode) -> WeightedSumNode | None:
    """Try to decompose an expression tree into a :class:`WeightedSumNode`.

    Recognises patterns like ``w·f + w·g + …`` where *w* is a literal and
    *f*, *g* are arbitrary sub-expressions.

    Returns ``None`` if the tree does not match.
    """
    terms: list[tuple[float, ExprNode]] = []

    def _collect(n: ExprNode) -> bool:
        if isinstance(n, BinOpNode) and n.op == "+":
            return _collect(n.left) and _collect(n.right)
        if isinstance(n, BinOpNode) and n.op == "*":
            if isinstance(n.left, LiteralNode):
                terms.append((n.left.value, n.right))
                return True
            if isinstance(n.right, LiteralNode):
                terms.append((n.right.value, n.left))
                return True
        if isinstance(n, (VarNode, CallNode)):
            terms.append((1.0, n))
            return True
        return False

    if _collect(node) and len(terms) >= 2:
        return WeightedSumNode(terms=terms)
    return None


# ---------------------------------------------------------------------------
# Parser: recursive-descent for arithmetic + function calls
# ---------------------------------------------------------------------------
# Grammar:
#   expr    := term (('+' | '-') term)*
#   term    := factor (('*' | '/') factor)*
#   factor  := '-' factor | '(' expr ')' | call | number | var
#   call    := name '(' arglist ')'
#   arglist := ε | expr (',' expr)*
#   name    := [A-Za-z_][A-Za-z0-9_.]*
#   number  := [0-9]+('.'[0-9]+)?

_TOKEN_RE = _re.compile(
    r"\s*(?:"
    r"(?P<NUMBER>[0-9]+(?:\.[0-9]+)?)"
    r"|(?P<NAME>[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<OP>[+\-*/])"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<COMMA>,)"
    r")\s*"
)


class _Parser:
    def __init__(self, text: str) -> None:
        self._tokens: list[tuple[str, str]] = [
            (m.lastgroup, m.group(m.lastgroup))  # type: ignore[arg-type]
            for m in _TOKEN_RE.finditer(text.strip())
            if m.lastgroup is not None
        ]
        self._pos = 0

    def _peek(self) -> tuple[str, str] | None:
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _consume(self) -> tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> str:
        tok_kind, tok_val = self._consume()
        if tok_kind != kind:
            raise ValueError(f"Expected {kind}, got {tok_kind}={tok_val!r}")
        return tok_val

    def parse(self) -> ExprNode:
        node = self._parse_expr()
        if self._peek() is not None:
            _, remaining = self._tokens[self._pos]
            raise ValueError(f"Unexpected token: {remaining!r}")
        return node

    def _parse_expr(self) -> ExprNode:
        left = self._parse_term()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in ("+", "-"):
                break
            _, op = self._consume()
            right = self._parse_term()
            left = BinOpNode(op, left, right)
        return left

    def _parse_term(self) -> ExprNode:
        left = self._parse_factor()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in ("*", "/"):
                break
            _, op = self._consume()
            right = self._parse_factor()
            left = BinOpNode(op, left, right)
        return left

    def _parse_factor(self) -> ExprNode:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression in factor")
        kind, val = tok

        # Unary minus
        if kind == "OP" and val == "-":
            self._consume()
            factor = self._parse_factor()
            return BinOpNode("*", LiteralNode(-1.0), factor)

        if kind == "LPAREN":
            self._consume()
            node = self._parse_expr()
            self._expect("RPAREN")
            return node

        if kind == "NUMBER":
            self._consume()
            return LiteralNode(float(val))

        if kind == "NAME":
            self._consume()
            # Function call?
            if self._peek() == ("LPAREN", "("):
                self._consume()  # consume (
                args: list[ExprNode] = []
                if self._peek() != ("RPAREN", ")"):
                    args.append(self._parse_expr())
                    while self._peek() == ("COMMA", ","):
                        self._consume()
                        args.append(self._parse_expr())
                self._expect("RPAREN")
                return CallNode(func=val, args=args)
            return VarNode(name=val)

        raise ValueError(f"Unexpected token in expression: {kind}={val!r}")


def parse_expr(text: str) -> ExprNode:
    """Parse an arithmetic expression string into an :class:`ExprNode` AST.

    Supports: numeric literals, variable references (dotted names), function
    calls, and the binary operators +, -, *, /.

    Examples::

        parse_expr("0.7 * relevance(x) + 0.3 * coherence(x)")
        parse_expr("normalize(raw_score)")
        parse_expr("sigmoid(20 * (x - 0.5))")
        parse_expr("scale(logit, -5.0, 5.0)")
    """
    return _Parser(text.strip()).parse()
