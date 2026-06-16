# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Recursive-descent parser for the MatrixAI mini-language.

Grammar
-------
::

    program       := statement*
    statement     := assignment

    assignment    := IDENT "(" params? ")" return_type? "=" expression
                   | IDENT return_type? "=" expression

    params        := typed_param ("," typed_param)*
    typed_param   := IDENT (":" type_expr)?
    return_type   := "->" type_expr

    expression    := term (("+" | "-") term)*
    term          := factor (("*" | "/") factor)*
    factor        := "-" factor
                   | NUMBER
                   | IDENT
                   | function_call
                   | "(" expression ")"

    function_call := IDENT "(" arguments? ")"
    arguments     := expression ("," expression)*

Usage::

    from matrixai.core.parser import parse

    stmts = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")
"""
from __future__ import annotations

from .ast_nodes import AssignNode, BinaryOpNode, CallNode, NumberNode, VarNode
from .lexer import Token, tokenize
from matrixai.types import parse_type_spec


class ParseError(Exception):
    pass


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------ helpers

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise ParseError(
                f"Expected {kind!r} but got {tok.kind!r} ({tok.value!r}) "
                f"at line {tok.line}:{tok.col}"
            )
        if value is not None and tok.value != value:
            raise ParseError(
                f"Expected {value!r} but got {tok.value!r} "
                f"at line {tok.line}:{tok.col}"
            )
        return self._advance()

    def _match_op(self, *ops: str) -> Token | None:
        tok = self._peek()
        if tok.kind == "OP" and tok.value in ops:
            return self._advance()
        return None

    # ------------------------------------------------------------------ grammar

    def parse_program(self) -> list[AssignNode]:
        stmts: list[AssignNode] = []
        while self._peek().kind != "EOF":
            stmts.append(self._parse_statement())
        return stmts

    def _parse_statement(self) -> AssignNode:
        name_tok = self._expect("IDENT")
        name = name_tok.value

        params: list[str] = []
        param_types = {}
        if self._peek().kind == "LPAREN":
            self._advance()  # consume "("
            if self._peek().kind == "IDENT":
                param, param_type = self._parse_typed_param()
                params.append(param)
                if param_type is not None:
                    param_types[param] = param_type
                while self._peek().kind == "COMMA":
                    self._advance()
                    param, param_type = self._parse_typed_param()
                    params.append(param)
                    if param_type is not None:
                        param_types[param] = param_type
            self._expect("RPAREN")

        return_type = None
        if self._peek().kind == "ARROW":
            self._advance()
            return_type = self._parse_type_until({"EQUALS"})

        self._expect("EQUALS")
        expr = self._parse_expression()
        return AssignNode(
            name=name,
            params=params,
            expr=expr,
            param_types=param_types,
            return_type=return_type,
        )

    def _parse_typed_param(self):
        name = self._expect("IDENT").value
        if self._peek().kind != "COLON":
            return name, None
        self._advance()
        return name, self._parse_type_until({"COMMA", "RPAREN"})

    def _parse_type_until(self, stop_kinds: set[str]):
        pieces: list[str] = []
        bracket_depth = 0
        while True:
            tok = self._peek()
            if tok.kind == "EOF":
                break
            if bracket_depth == 0 and tok.kind in stop_kinds:
                break
            if tok.kind == "LBRACKET":
                bracket_depth += 1
            elif tok.kind == "RBRACKET":
                bracket_depth -= 1
            pieces.append(self._advance().value)
        try:
            return parse_type_spec("".join(pieces))
        except ValueError as exc:
            raise ParseError(str(exc)) from exc

    def _parse_expression(self):
        node = self._parse_term()
        while True:
            op = self._match_op("+", "-")
            if op is None:
                break
            right = self._parse_term()
            node = BinaryOpNode(op=op.value, left=node, right=right)
        return node

    def _parse_term(self):
        node = self._parse_factor()
        while True:
            op = self._match_op("*", "/")
            if op is None:
                break
            right = self._parse_factor()
            node = BinaryOpNode(op=op.value, left=node, right=right)
        return node

    def _parse_factor(self):
        tok = self._peek()

        # Unary minus
        if tok.kind == "OP" and tok.value == "-":
            self._advance()
            operand = self._parse_factor()
            return BinaryOpNode(op="*", left=NumberNode(-1.0), right=operand)

        # Parenthesized expression
        if tok.kind == "LPAREN":
            self._advance()
            node = self._parse_expression()
            self._expect("RPAREN")
            return node

        # Number literal
        if tok.kind == "NUMBER":
            self._advance()
            return NumberNode(float(tok.value))

        # IDENT → variable or function call
        if tok.kind == "IDENT":
            name = self._advance().value
            if self._peek().kind == "LPAREN":
                # function call
                self._advance()  # consume "("
                args = []
                if self._peek().kind != "RPAREN":
                    args.append(self._parse_expression())
                    while self._peek().kind == "COMMA":
                        self._advance()
                        args.append(self._parse_expression())
                self._expect("RPAREN")
                return CallNode(name=name, args=args)
            return VarNode(name=name)

        raise ParseError(
            f"Unexpected token {tok.kind!r} ({tok.value!r}) "
            f"at line {tok.line}:{tok.col}"
        )


def parse(source: str) -> list[AssignNode]:
    """Parse *source* into a list of :class:`~matrixai.core.ast_nodes.AssignNode`."""
    tokens = tokenize(source)
    return _Parser(tokens).parse_program()
