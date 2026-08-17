# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""M8 v2 — Domain-rule DSL + deterministic evaluator (LLM as domain simulator).

The LLM proposes, in a single call, the domain logic that maps features → class as a
small, bounded set of threshold rules. This module parses that textual form into a
safe structure and evaluates it deterministically over sampled rows — no `eval`, no
per-row LLM call. Determinista = SUELO, LLM = TECHO: the LLM only proposes the logic;
Python executes it reproducibly. Invalid/empty rules are rejected so the caller can
fall back to the toy `coherent` mode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Comparison operators, longest-first so ">=" is matched before ">".
_OPS = ("<=", ">=", "==", "<", ">", "=")
_OP_FUNCS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: abs(a - b) < 1e-9,
    "=": lambda a, b: abs(a - b) < 1e-9,
}
_CONDITION_RE = re.compile(
    r"^\s*(?P<feature>[A-Za-z_]\w*)\s*(?P<op><=|>=|==|<|>|=)\s*(?P<value>-?\d+(?:\.\d+)?)\s*$"
)


class DomainRulesError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: float

    def matches(self, row: dict[str, float]) -> bool:
        if self.feature not in row:
            return False
        try:
            return _OP_FUNCS[self.op](float(row[self.feature]), self.value)
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True)
class Clause:
    conditions: tuple[Condition, ...]
    combiner: str  # "AND" | "OR"

    def matches(self, row: dict[str, float]) -> bool:
        if not self.conditions:
            return False
        results = (c.matches(row) for c in self.conditions)
        return all(results) if self.combiner == "AND" else any(results)


@dataclass(frozen=True)
class Rule:
    label: str
    clause: Clause


@dataclass(frozen=True)
class DomainRules:
    """Ordered threshold rules + a default label (first match wins)."""
    rules: tuple[Rule, ...]
    default_label: str
    features: tuple[str, ...] = field(default_factory=tuple)
    #: Contrato 80-C1 — RUIDO DECLARADO, y por qué existe.
    #:
    #: Una regla determinista sin ruido produce un dataset perfectamente
    #: separable: el modelo entrena hasta el 100 % de exactitud y ese
    #: número no se sostiene delante de nadie técnico —ya hubo que
    #: retirar una captura del tour por exactamente eso—. Con `RUIDO: 0.1`
    #: una de cada diez filas contradice la regla, como en la vida.
    #:
    #: Es una PROPORCIÓN [0,1], no una desviación: se elige así porque se
    #: explica en una frase («una de cada diez filas no sigue la regla»)
    #: y porque vale igual para dos clases que para cinco.
    noise: float = 0.0

    def label_for(self, row: dict[str, float]) -> str:
        for rule in self.rules:
            if rule.clause.matches(row):
                return rule.label
        return self.default_label

    def referenced_features(self) -> set[str]:
        return {c.feature for rule in self.rules for c in rule.clause.conditions}

    def to_text(self) -> str:
        """Human-readable form (for audit). Mirrors the parsed textual DSL."""
        def fmt_val(v: float) -> str:
            return str(int(v)) if float(v).is_integer() else str(v)
        lines = []
        for rule in self.rules:
            conds = f" {rule.clause.combiner} ".join(
                f"{c.feature} {c.op} {fmt_val(c.value)}" for c in rule.clause.conditions
            )
            lines.append(f"{rule.label}: {conds}")
        if self.default_label:
            lines.append(f"DEFAULT: {self.default_label}")
        # El ruido se ESCRIBE en el texto auditable: si no sale aquí, quien
        # lea la receta no puede reproducir el dataset.
        if self.noise > 0:
            lines.append(f"RUIDO: {self.noise}")
        return "\n".join(lines)

    def normalized(self, ranges: dict[str, tuple[float, float]]) -> DomainRules:
        """Map domain-scale thresholds to [0,1] using each feature's range.

        The LLM expresses thresholds in domain units (e.g. `indice_charlson > 15`);
        the generator samples and labels in normalized [0,1] space. Features without
        a known range keep their value (assumed already in [0,1])."""
        def norm(cond: Condition) -> Condition:
            rng = ranges.get(cond.feature)
            if not rng:
                return cond
            lo, hi = rng
            if hi == lo:
                return cond
            # No clamping: an out-of-range threshold (e.g. edad > 200 with range
            # [0,120] → 1.67) must stay out of [0,1] so the condition is correctly
            # never-true (or always-true), not silently pulled to the boundary.
            v = (cond.value - lo) / (hi - lo)
            return Condition(cond.feature, cond.op, v)

        new_rules = tuple(
            Rule(r.label, Clause(tuple(norm(c) for c in r.clause.conditions), r.clause.combiner))
            for r in self.rules
        )
        # El ruido NO se normaliza: es una proporción de filas, no un umbral.
        return DomainRules(new_rules, self.default_label, self.features, self.noise)

    def validate(self, allowed_features: list[str], allowed_labels: list[str]) -> list[str]:
        """Return a list of problems; empty means the rules are usable."""
        errors: list[str] = []
        feats, labels = set(allowed_features), set(allowed_labels)
        if not self.rules:
            errors.append("no rules parsed")
        unknown_feats = sorted(self.referenced_features() - feats)
        if unknown_feats:
            errors.append(f"unknown features: {', '.join(unknown_feats)}")
        rule_labels = {r.label for r in self.rules} | {self.default_label}
        unknown_labels = sorted(rule_labels - labels)
        if unknown_labels:
            errors.append(f"unknown labels: {', '.join(unknown_labels)}")
        if self.default_label not in labels:
            errors.append("default label not in declared labels")
        # The rules should be able to discriminate (at least 2 labels reachable).
        if len(rule_labels & labels) < 2:
            errors.append("rules cover fewer than 2 declared labels")
        return errors


def _parse_clause(body: str) -> Clause | None:
    """Parse 'a > 1 AND b < 2' (single combiner). Returns None if malformed."""
    # Split on AND / OR (word-bounded, case-insensitive). A clause uses one combiner.
    has_and = re.search(r"\bAND\b", body, re.IGNORECASE) is not None
    has_or = re.search(r"\bOR\b", body, re.IGNORECASE) is not None
    if has_and and has_or:
        return None  # mixed combiners not supported in v2 (ambiguous precedence)
    combiner = "OR" if has_or else "AND"
    parts = re.split(r"\bAND\b|\bOR\b", body, flags=re.IGNORECASE)
    conditions: list[Condition] = []
    for part in parts:
        if not part.strip():
            continue
        m = _CONDITION_RE.match(part)
        if not m:
            return None
        conditions.append(Condition(m.group("feature"), m.group("op"), float(m.group("value"))))
    if not conditions:
        return None
    return Clause(tuple(conditions), combiner)


def parse_domain_rules(text: str) -> DomainRules:
    """Parse the textual domain-rule form. Tolerant: skips lines it cannot parse.

        CRITICO: indice_charlson > 15 AND creatinina > 4
        ALTO: ingresos_previos_12m > 5 OR num_diagnosticos > 10
        DEFAULT: BAJO

    Returns a (possibly empty/invalid) DomainRules — the caller validates."""
    rules: list[Rule] = []
    default_label = ""
    noise = 0.0
    for raw in (text or "").splitlines():
        line = raw.strip()
        # Contrato 80-C1: una receta la escribe una PERSONA, así que se
        # admiten comentarios. Sin esto, explicar una regla obliga a
        # explicarla fuera del fichero — y ahí es donde se pierde.
        if line.startswith("#"):
            continue
        if not line or ":" not in line:
            continue
        head, _, body = line.partition(":")
        head, body = head.strip(), body.strip()
        if not head or not body:
            continue
        if head.upper() == "DEFAULT":
            default_label = body.split()[0] if body.split() else ""
            continue
        # Contrato 80-C1: `RUIDO: 0.1` (o `NOISE: 0.1`). Fuera de [0,1] se
        # ignora en vez de reventar: el resto de este parser es tolerante a
        # propósito —lo escribe una persona— y una receta con un ruido
        # imposible sigue siendo una receta útil.
        if head.upper() in ("RUIDO", "NOISE"):
            try:
                v = float(body.split()[0])
            except (ValueError, IndexError):
                continue
            if 0.0 <= v <= 1.0:
                noise = v
            continue
        # Contrato 80-C1: un objetivo binario (`Probability`) tiene por
        # etiquetas «1» y «0», que no son identificadores. Se admiten los
        # dos: `ALTO` y `1`.
        if not re.match(r"^([A-Za-z_]\w*|\d+)$", head):
            continue  # ni identificador ni número: no es una etiqueta
        clause = _parse_clause(body)
        if clause is not None:
            rules.append(Rule(head, clause))
    return DomainRules(tuple(rules), default_label, (), noise)
