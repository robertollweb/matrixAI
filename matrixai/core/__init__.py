# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""MatrixAI core — mathematical expression engine.

Public API
----------

Parsing::

    from matrixai.core import parse
    stmts = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")

Evaluation::

    from matrixai.core import Evaluator, FunctionRegistry
    from matrixai.functions import build_default_registry

    reg = build_default_registry()
    ev  = Evaluator(reg)
    ev.define_all(stmts)
    result, trace = ev.eval_definition("score", {"relevance": 0.9, "coherence": 0.8})

Graph::

    from matrixai.core import ast_to_graph, graph_to_text
    g = ast_to_graph(stmts[0])
    print(graph_to_text(g))
"""
from .ast_nodes import AssignNode, BinaryOpNode, CallNode, NumberNode, VarNode
from .evaluator import EvaluationError, Evaluator
from .graph import ast_to_graph, graph_to_text
from .lexer import LexError, Token, tokenize
from .parser import ParseError, parse
from .registry import FunctionRegistry
from .trace import EvalStep, EvalTrace
from matrixai.types import RangeSpec, TypeCheckResult, TypeSpec, check_mx_types, parse_type_spec

__all__ = [
    "AssignNode",
    "BinaryOpNode",
    "CallNode",
    "EvalStep",
    "EvalTrace",
    "EvaluationError",
    "Evaluator",
    "FunctionRegistry",
    "LexError",
    "NumberNode",
    "ParseError",
    "RangeSpec",
    "Token",
    "TypeCheckResult",
    "TypeSpec",
    "VarNode",
    "ast_to_graph",
    "check_mx_types",
    "graph_to_text",
    "parse",
    "parse_type_spec",
    "tokenize",
]
