# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Lexer for the MatrixAI mini-language (.mx files).

Token kinds
-----------
  NUMBER    — float literal: ``0.6``, ``3.14``, ``1e-3``
  IDENT     — identifier: ``score``, ``relevance``, ``Confidence.max``
    OP        — operator: ``+  -  *  /``
    ARROW     — ``->``
    COLON     — ``:``
    LBRACKET  — ``[``
    RBRACKET  — ``]``
  LPAREN    — ``(``
  RPAREN    — ``)``
  COMMA     — ``,``
  EQUALS    — ``=``
  EOF       — end of input

Whitespace and ``#``-prefixed comments are silently discarded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_PATTERNS = [
    ("NUMBER", r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"),
    ("IDENT",  r"[a-zA-Z_][a-zA-Z0-9_.]*"),
    ("ARROW",  r"->"),
    ("OP",     r"[+\-*/]"),
    ("COLON",  r":"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA",  r","),
    ("EQUALS", r"="),
    ("NL",     r"\n"),
    ("SKIP",   r"[ \t\r]+|#[^\n]*"),
]

_MASTER = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_PATTERNS)
)


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r}, {self.line}:{self.col})"


class LexError(Exception):
    pass


def tokenize(source: str) -> list[Token]:
    """Return a list of :class:`Token` objects for *source*, ending with EOF."""
    tokens: list[Token] = []
    line = 1
    line_start = 0
    pos = 0
    while pos < len(source):
        m = _MASTER.match(source, pos)
        if m is None:
            col = pos - line_start + 1
            raise LexError(
                f"Unexpected character {source[pos]!r} at line {line}:{col}"
            )
        kind = m.lastgroup
        value = m.group()
        col = m.start() - line_start + 1
        if kind == "NL":
            line += 1
            line_start = m.end()
        elif kind != "SKIP":
            tokens.append(Token(kind, value, line, col))
        pos = m.end()
    tokens.append(Token("EOF", "", line, len(source) - line_start + 1))
    return tokens
