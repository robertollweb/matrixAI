# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Input representation: discrete rules expressed as plain text
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscreteRule:
    """A classic if/else rule in its original discrete form.

    Examples
    --------
    ``if risk > 0.8 then alert``
    ``if confidence >= 0.95 then reply``
    ``if score > 0.7 and urgency > 0.5 then dispatch``
    """
    raw: str
    conditions: list[dict[str, Any]] = field(default_factory=list)
    action: str = ""


# ---------------------------------------------------------------------------
# Output representation: continuous MatrixAI expression
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContinuousTranslation:
    """A discrete rule translated into a continuous sigmoid approximation.

    Attributes
    ----------
    original_rule : str
        The raw discrete rule that was translated.
    expression : str
        The continuous MatrixAI expression.
    expression_kind : str
        One of ``sigmoid_threshold``, ``sigmoid_and``, ``sigmoid_or``,
        ``softmax_linear``.
    inputs : list[str]
        Variable names that feed the expression.
    parameters : dict[str, Any]
        Numeric parameters (scale k, threshold τ, etc.).
    explanation : str
        Human-readable explanation of why the translation is correct.
    """
    original_rule: str
    expression: str
    expression_kind: str
    inputs: list[str]
    parameters: dict[str, Any]
    explanation: str


@dataclass(frozen=True)
class MathematicalReport:
    translations: list[ContinuousTranslation] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def all_resolved(self) -> bool:
        return not self.unresolved

    def summary(self) -> str:
        lines: list[str] = []
        if self.translations:
            lines.append(f"MathematicalAgent: {len(self.translations)} translation(s)")
            for t in self.translations:
                lines.append(f"\n  Original : {t.original_rule}")
                lines.append(f"  Continuous: {t.expression}")
                lines.append(f"  Kind      : {t.expression_kind}")
                lines.append(f"  Why       : {t.explanation}")
        if self.unresolved:
            lines.append(f"\nMathematicalAgent: {len(self.unresolved)} unresolved rule(s)")
            for r in self.unresolved:
                lines.append(f"  UNRESOLVED: {r}")
        if not self.translations and not self.unresolved:
            lines.append("MathematicalAgent: no rules to translate.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MathematicalAgent:
    """Translates discrete if/else rules into continuous MatrixAI expressions.

    The core idea (from the whitepaper):

        Traditional:   if x > t → action()
        MatrixAI:      σ(k · (x − t))

    The sigmoid σ with a large scale k approximates the Heaviside step function
    while remaining differentiable everywhere.  A scale of k=20 gives a sharp
    transition (±0.3 of the threshold → 99% confidence), k=50 gives an even
    sharper one suitable for classification boundaries.

    The agent also handles:
    - AND conditions   →  σ(k·(x₁−τ₁)) · σ(k·(x₂−τ₂))  (product of sigmoids)
    - OR conditions    →  1 − (1−σ(k·(x₁−τ₁))) · (1−σ(k·(x₂−τ₂)))
    - Multi-class      →  softmax([x₁, x₂, …])
    """

    # Matches: if <var> <op> <threshold>  (threshold may be negative, e.g. -273.15)
    _SIMPLE_RE = re.compile(
        r"^if\s+(?P<var>[A-Za-z_][\w.]*)\s*(?P<op>>|>=|<|<=)\s*(?P<threshold>-?[0-9.]+)"
        r"(?:\s+then\s+(?P<action>\w+))?$",
        re.IGNORECASE,
    )

    # Matches: if <var1> <op1> <t1> and <var2> <op2> <t2>
    _AND_RE = re.compile(
        r"^if\s+(?P<var1>[A-Za-z_][\w.]*)\s*(?P<op1>>|>=|<|<=)\s*(?P<t1>-?[0-9.]+)"
        r"\s+and\s+(?P<var2>[A-Za-z_][\w.]*)\s*(?P<op2>>|>=|<|<=)\s*(?P<t2>-?[0-9.]+)"
        r"(?:\s+then\s+(?P<action>\w+))?$",
        re.IGNORECASE,
    )

    # Matches: if <var1> <op1> <t1> or <var2> <op2> <t2>
    _OR_RE = re.compile(
        r"^if\s+(?P<var1>[A-Za-z_][\w.]*)\s*(?P<op1>>|>=|<|<=)\s*(?P<t1>-?[0-9.]+)"
        r"\s+or\s+(?P<var2>[A-Za-z_][\w.]*)\s*(?P<op2>>|>=|<|<=)\s*(?P<t2>-?[0-9.]+)"
        r"(?:\s+then\s+(?P<action>\w+))?$",
        re.IGNORECASE,
    )

    # Matches: classify <var> into <label1>, <label2>, ...
    _CLASSIFY_RE = re.compile(
        r"^classify\s+(?P<var>[A-Za-z_][\w.]*)\s+into\s+(?P<labels>.+)$",
        re.IGNORECASE,
    )

    # Default scale k for sigmoid approximation
    _DEFAULT_SCALE = 20

    def translate(self, rules: list[str]) -> MathematicalReport:
        """Translate a list of discrete rule strings into continuous expressions."""
        translations: list[ContinuousTranslation] = []
        unresolved: list[str] = []

        for raw in rules:
            rule = raw.strip()
            result = (
                self._try_and(rule)
                or self._try_or(rule)
                or self._try_simple(rule)
                or self._try_classify(rule)
                or self._try_aggregate(rule)
                or self._try_normalize(rule)
                or self._try_select(rule)
                or self._try_assign(rule)
            )
            if result is not None:
                translations.append(result)
            else:
                unresolved.append(rule)

        return MathematicalReport(translations=translations, unresolved=unresolved)

    def translate_text(self, text: str) -> MathematicalReport:
        """Translate rules from a multi-line string (one rule per line)."""
        rules = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return self.translate(rules)

    # ------------------------------------------------------------------
    # Internal translators
    # ------------------------------------------------------------------

    def _try_simple(self, rule: str) -> ContinuousTranslation | None:
        m = self._SIMPLE_RE.match(rule)
        if not m:
            return None
        var = m.group("var")
        op = m.group("op")
        threshold = float(m.group("threshold"))
        action = m.group("action") or "activate"
        k = self._DEFAULT_SCALE

        # Invert sign for < / <= operators
        if op in ("<", "<="):
            expr = f"sigmoid({k} * ({threshold} - {var}))"
            explanation = (
                f"'{var} {op} {threshold}' becomes σ({k}·({threshold}−{var})). "
                f"When {var} is well below {threshold} the output approaches 1; "
                f"when above it approaches 0. Scale k={k} gives a sharp transition."
            )
        else:
            expr = f"sigmoid({k} * ({var} - {threshold}))"
            explanation = (
                f"'{var} {op} {threshold}' becomes σ({k}·({var}−{threshold})). "
                f"When {var} is well above {threshold} the output approaches 1; "
                f"when below it approaches 0. Scale k={k} gives a sharp transition."
            )

        return ContinuousTranslation(
            original_rule=rule,
            expression=expr,
            expression_kind="sigmoid_threshold",
            inputs=[var],
            parameters={"scale": k, "threshold": threshold, "operator": op, "action": action},
            explanation=explanation,
        )

    def _try_and(self, rule: str) -> ContinuousTranslation | None:
        m = self._AND_RE.match(rule)
        if not m:
            return None
        var1, op1, t1 = m.group("var1"), m.group("op1"), float(m.group("t1"))
        var2, op2, t2 = m.group("var2"), m.group("op2"), float(m.group("t2"))
        action = m.group("action") or "activate"
        k = self._DEFAULT_SCALE

        e1 = self._sigmoid_term(var1, op1, t1, k)
        e2 = self._sigmoid_term(var2, op2, t2, k)
        expr = f"sigmoid_product({e1}, {e2})"

        return ContinuousTranslation(
            original_rule=rule,
            expression=expr,
            expression_kind="sigmoid_and",
            inputs=[var1, var2],
            parameters={
                "scale": k,
                "condition_1": {"var": var1, "op": op1, "threshold": t1},
                "condition_2": {"var": var2, "op": op2, "threshold": t2},
                "action": action,
            },
            explanation=(
                f"AND of two conditions is approximated as the product of two sigmoids: "
                f"σ₁·σ₂. Both must be close to 1 for the product to exceed the action threshold. "
                f"Scale k={k} on each factor."
            ),
        )

    def _try_or(self, rule: str) -> ContinuousTranslation | None:
        m = self._OR_RE.match(rule)
        if not m:
            return None
        var1, op1, t1 = m.group("var1"), m.group("op1"), float(m.group("t1"))
        var2, op2, t2 = m.group("var2"), m.group("op2"), float(m.group("t2"))
        action = m.group("action") or "activate"
        k = self._DEFAULT_SCALE

        e1 = self._sigmoid_term(var1, op1, t1, k)
        e2 = self._sigmoid_term(var2, op2, t2, k)
        expr = f"sigmoid_or(1 - (1 - {e1}) * (1 - {e2}))"

        return ContinuousTranslation(
            original_rule=rule,
            expression=expr,
            expression_kind="sigmoid_or",
            inputs=[var1, var2],
            parameters={
                "scale": k,
                "condition_1": {"var": var1, "op": op1, "threshold": t1},
                "condition_2": {"var": var2, "op": op2, "threshold": t2},
                "action": action,
            },
            explanation=(
                f"OR of two conditions uses the probabilistic OR formula: "
                f"1−(1−σ₁)·(1−σ₂). At least one must be active for the result to approach 1. "
                f"Scale k={k} on each factor."
            ),
        )

    def _try_classify(self, rule: str) -> ContinuousTranslation | None:
        m = self._CLASSIFY_RE.match(rule)
        if not m:
            return None
        var = m.group("var")
        labels = [label.strip() for label in m.group("labels").split(",") if label.strip()]
        if len(labels) < 2:
            return None

        label_list = ", ".join(labels)
        expr = f"softmax([{label_list}])"

        return ContinuousTranslation(
            original_rule=rule,
            expression=expr,
            expression_kind="softmax_linear",
            inputs=[var],
            parameters={"labels": labels, "num_classes": len(labels)},
            explanation=(
                f"Multi-class classification over [{label_list}] uses softmax, which produces "
                f"a valid probability distribution summing to 1. "
                f"The class with the highest probability becomes the predicted label."
            ),
        )

    def _sigmoid_term(self, var: str, op: str, threshold: float, k: int) -> str:
        if op in ("<", "<="):
            return f"sigmoid({k} * ({threshold} - {var}))"
        return f"sigmoid({k} * ({var} - {threshold}))"

    # ------------------------------------------------------------------
    # P1 — Symbolic expressions, aggregation, normalisation, selection
    # ------------------------------------------------------------------

    # Matches: output = expr
    _ASSIGN_RE = re.compile(r"^(?P<output>[A-Za-z_]\w*)\s*=\s*(?P<expr>.+)$")

    # Matches: aggregate v1, v2[, ...] using max|min|mean|softmax|vote
    _AGGREGATE_RE = re.compile(
        r"^aggregate\s+(?P<inputs>[A-Za-z_][\w,\s]*)\s+using\s+(?P<method>max|min|mean|softmax|vote)$",
        re.IGNORECASE,
    )

    # Matches: normalize var [to [lo, hi]]
    _NORMALIZE_RE = re.compile(
        r"^normalize\s+(?P<var>[A-Za-z_]\w*)"
        r"(?:\s+to\s+\[(?P<lo>[0-9.]+),\s*(?P<hi>[0-9.]+)\])?$",
        re.IGNORECASE,
    )

    # Matches: select best from <candidates> by <score>
    _SELECT_RE = re.compile(
        r"^select\s+best\s+from\s+(?P<candidates>[A-Za-z_]\w*)\s+by\s+(?P<score>[A-Za-z_]\w*)$",
        re.IGNORECASE,
    )

    def build_symbolic(self, expr_str: str) -> ContinuousTranslation:
        """Parse an arbitrary symbolic expression string into a ContinuousTranslation.

        Recognises weighted sums (``w1*f1 + w2*f2``) and labels them
        ``symbolic_weighted_sum``.  All other parseable expressions become
        ``symbolic_expr``.

        Raises ``ValueError`` if the expression cannot be parsed.
        """
        from matrixai.ir.expr import parse_expr, extract_weighted_sum, collect_vars

        clean = expr_str.strip()
        try:
            node = parse_expr(clean)
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Cannot parse symbolic expression: {clean!r} — {exc}"
            ) from exc

        weighted = extract_weighted_sum(node)
        if weighted is not None:
            inputs = list(
                dict.fromkeys(v for _, sub in weighted.terms for v in collect_vars(sub))
            )
            terms = [{"weight": w, "expr": str(sub)} for w, sub in weighted.terms]
            return ContinuousTranslation(
                original_rule=clean,
                expression=str(weighted),
                expression_kind="symbolic_weighted_sum",
                inputs=inputs,
                parameters={"terms": terms, "ast": weighted.to_dict()},
                explanation=f"Weighted combination: {str(weighted)}",
            )

        inputs = list(dict.fromkeys(collect_vars(node)))
        return ContinuousTranslation(
            original_rule=clean,
            expression=str(node),
            expression_kind="symbolic_expr",
            inputs=inputs,
            parameters={"ast": node.to_dict()},
            explanation=f"Symbolic expression: {str(node)}",
        )

    def _try_assign(self, rule: str) -> ContinuousTranslation | None:
        m = self._ASSIGN_RE.match(rule)
        if not m:
            return None
        output = m.group("output")
        expr_str = m.group("expr").strip()
        try:
            translation = self.build_symbolic(expr_str)
        except ValueError:
            return None
        return ContinuousTranslation(
            original_rule=rule,
            expression=f"{output} = {translation.expression}",
            expression_kind=translation.expression_kind,
            inputs=translation.inputs,
            parameters={**translation.parameters, "output": output},
            explanation=(
                f"Assignment '{output}' bound to symbolic expression: {translation.expression}"
            ),
        )

    def _try_aggregate(self, rule: str) -> ContinuousTranslation | None:
        m = self._AGGREGATE_RE.match(rule)
        if not m:
            return None
        inputs_str = m.group("inputs")
        method = m.group("method").lower()
        inputs = [v.strip() for v in inputs_str.split(",") if v.strip()]
        kind = f"aggregate_{method}"

        if method == "max":
            expr = f"max({', '.join(inputs)})"
            explanation = f"Aggregate: maximum of {inputs}"
        elif method == "min":
            expr = f"min({', '.join(inputs)})"
            explanation = f"Aggregate: minimum of {inputs}"
        elif method == "mean":
            n = len(inputs)
            expr = f"({' + '.join(inputs)}) / {n}"
            explanation = f"Aggregate: arithmetic mean of {inputs}"
        elif method == "softmax":
            expr = f"softmax([{', '.join(inputs)}])"
            explanation = f"Aggregate: softmax distribution over {inputs}"
        elif method == "vote":
            expr = f"vote({', '.join(inputs)})"
            explanation = (
                f"Aggregate: majority vote — fraction of inputs exceeding 0.5"
            )
        else:
            return None

        return ContinuousTranslation(
            original_rule=rule,
            expression=expr,
            expression_kind=kind,
            inputs=inputs,
            parameters={"inputs": inputs, "method": method},
            explanation=explanation,
        )

    def _try_normalize(self, rule: str) -> ContinuousTranslation | None:
        m = self._NORMALIZE_RE.match(rule)
        if not m:
            return None
        var = m.group("var")
        lo = float(m.group("lo")) if m.group("lo") is not None else 0.0
        hi = float(m.group("hi")) if m.group("hi") is not None else 1.0

        if lo == 0.0 and hi == 1.0:
            expr = f"normalize({var})"
            explanation = f"Clip '{var}' to [0, 1]"
        else:
            expr = f"scale({var}, {lo}, {hi})"
            explanation = f"Scale '{var}' from [{lo}, {hi}] to [0, 1]"

        return ContinuousTranslation(
            original_rule=rule,
            expression=expr,
            expression_kind="normalize",
            inputs=[var],
            parameters={"var": var, "lo": lo, "hi": hi},
            explanation=explanation,
        )

    def _try_select(self, rule: str) -> ContinuousTranslation | None:
        m = self._SELECT_RE.match(rule)
        if not m:
            return None
        candidates = m.group("candidates")
        score = m.group("score")
        return ContinuousTranslation(
            original_rule=rule,
            expression=f"argmax({score})",
            expression_kind="select_argmax",
            inputs=[candidates, score],
            parameters={"candidates": candidates, "score_input": score},
            explanation=(
                f"Select the candidate from '{candidates}' with the highest '{score}'"
            ),
        )
