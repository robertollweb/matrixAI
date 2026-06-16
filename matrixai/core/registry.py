# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Function registry for the MatrixAI evaluator.

Keeps a map of name → callable.  The evaluator looks up functions here when
it encounters a :class:`~matrixai.core.ast_nodes.CallNode` whose name is not a
user-defined function (``AssignNode``).
"""
from __future__ import annotations

from typing import Callable


class FunctionRegistry:
    """Registry of callable functions available during expression evaluation."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register *fn* under *name*."""
        self._functions[name] = fn

    def register_many(self, mapping: dict[str, Callable]) -> None:
        """Register multiple functions at once from a dict."""
        self._functions.update(mapping)

    def get(self, name: str) -> Callable:
        """Return the callable for *name*, or raise :class:`KeyError`."""
        if name not in self._functions:
            raise KeyError(f"Function not found in registry: {name!r}")
        return self._functions[name]

    def has(self, name: str) -> bool:
        return name in self._functions

    def names(self) -> list[str]:
        return sorted(self._functions)

    def __repr__(self) -> str:
        return f"FunctionRegistry({self.names()})"
