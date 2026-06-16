# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Convert a MatrixAI AST into a computation graph (nodes + directed edges).

Usage::

    from matrixai.core.graph import ast_to_graph

    assign = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")[0]
    g = ast_to_graph(assign)
    # g["nodes"] — list of node dicts
    # g["edges"] — list of [source_id, target_id] pairs

Node types
----------
``input``   — function parameter (e.g. ``x``)
``const``   — numeric literal
``var``     — variable reference
``call``    — function call
``op``      — binary operation (add / sub / mul / div)
``output``  — result of the whole assignment
"""
from __future__ import annotations

from typing import Any

from .ast_nodes import AssignNode, BinaryOpNode, CallNode, NumberNode, VarNode


_OP_TYPE = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


def ast_to_graph(assign: AssignNode) -> dict[str, Any]:
    """Return ``{"nodes": [...], "edges": [...]}`` for *assign*."""
    nodes: list[dict] = []
    edges: list[list[str]] = []
    _counter = [0]

    def _new_id(prefix: str) -> str:
        _counter[0] += 1
        return f"{prefix}_{_counter[0]}"

    def _visit(node: Any) -> str:
        if isinstance(node, NumberNode):
            nid = _new_id("const")
            nodes.append({"id": nid, "type": "const", "value": node.value})
            return nid

        if isinstance(node, VarNode):
            # Reuse the input node if the name matches a param
            if node.name in param_ids:
                return param_ids[node.name]
            nid = _new_id("var")
            nodes.append({"id": nid, "type": "var", "name": node.name})
            return nid

        if isinstance(node, BinaryOpNode):
            op_name = _OP_TYPE.get(node.op, node.op)
            nid = _new_id(op_name)
            nodes.append({"id": nid, "type": "op", "op": op_name})
            left_id = _visit(node.left)
            right_id = _visit(node.right)
            edges.append([left_id, nid])
            edges.append([right_id, nid])
            return nid

        if isinstance(node, CallNode):
            nid = _new_id(node.name)
            nodes.append({"id": nid, "type": "call", "function": node.name})
            for arg in node.args:
                arg_id = _visit(arg)
                edges.append([arg_id, nid])
            return nid

        raise ValueError(f"Unknown node type: {type(node).__name__}")

    # Parameter input nodes
    param_ids: dict[str, str] = {}
    for param in assign.params:
        nid = _new_id(f"input_{param}")
        nodes.append({"id": nid, "type": "input", "name": param})
        param_ids[param] = nid

    expr_id = _visit(assign.expr)

    # Output node
    out_id = _new_id(assign.name)
    nodes.append({"id": out_id, "type": "output", "name": assign.name})
    edges.append([expr_id, out_id])

    return {"nodes": nodes, "edges": edges}


def graph_to_text(graph: dict[str, Any]) -> str:
    """Return a human-readable summary of the computation graph."""
    lines = ["Nodes:"]
    for n in graph["nodes"]:
        kind = n["type"]
        if kind == "input":
            lines.append(f"  [{n['id']}] input  {n['name']}")
        elif kind == "output":
            lines.append(f"  [{n['id']}] output {n['name']}")
        elif kind == "const":
            lines.append(f"  [{n['id']}] const  {n['value']}")
        elif kind == "var":
            lines.append(f"  [{n['id']}] var    {n['name']}")
        elif kind == "call":
            lines.append(f"  [{n['id']}] call   {n['function']}()")
        elif kind == "op":
            lines.append(f"  [{n['id']}] op     {n['op']}")
    lines.append("Edges:")
    for src, dst in graph["edges"]:
        lines.append(f"  {src} --> {dst}")
    return "\n".join(lines)
