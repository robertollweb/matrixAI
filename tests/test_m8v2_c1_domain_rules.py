"""M8 v2 — C1: DSL de reglas de dominio + evaluador determinista.

Reglas umbral AND/OR parseadas a una estructura segura y evaluadas sin `eval`. La
normalización mapea umbrales de escala-dominio a [0,1]. La validación rechaza reglas
inválidas para que el llamador caiga a `coherent` (fallback)."""
from __future__ import annotations

from matrixai.training.domain_rules import (
    Clause,
    Condition,
    DomainRules,
    Rule,
    parse_domain_rules,
)


def test_parse_basic_rules():
    dr = parse_domain_rules(
        "CRITICO: indice_charlson > 15 AND creatinina > 4\n"
        "ALTO: ingresos_previos > 5 OR num_diagnosticos > 10\n"
        "DEFAULT: BAJO\n"
    )
    assert dr.default_label == "BAJO"
    assert len(dr.rules) == 2
    assert dr.rules[0].label == "CRITICO"
    assert dr.rules[0].clause.combiner == "AND"
    assert dr.rules[1].clause.combiner == "OR"
    assert dr.referenced_features() == {"indice_charlson", "creatinina", "ingresos_previos", "num_diagnosticos"}


def test_first_match_wins_then_default():
    dr = parse_domain_rules(
        "HIGH: a > 0.8\n"
        "MID: a > 0.4\n"
        "DEFAULT: LOW\n"
    )
    assert dr.label_for({"a": 0.9}) == "HIGH"
    assert dr.label_for({"a": 0.5}) == "MID"
    assert dr.label_for({"a": 0.1}) == "LOW"


def test_and_or_semantics():
    and_clause = Clause((Condition("a", ">", 0.5), Condition("b", ">", 0.5)), "AND")
    or_clause = Clause((Condition("a", ">", 0.5), Condition("b", ">", 0.5)), "OR")
    assert and_clause.matches({"a": 0.6, "b": 0.6}) is True
    assert and_clause.matches({"a": 0.6, "b": 0.4}) is False
    assert or_clause.matches({"a": 0.6, "b": 0.4}) is True
    assert or_clause.matches({"a": 0.1, "b": 0.1}) is False


def test_operators():
    for op, val, expect in [(">", 0.4, True), ("<", 0.4, False), (">=", 0.5, True),
                             ("<=", 0.5, True), ("==", 0.5, True)]:
        c = Condition("a", op, val)
        assert c.matches({"a": 0.5}) is expect, (op, val)


def test_missing_feature_does_not_match():
    assert Condition("missing", ">", 0.0).matches({"a": 1.0}) is False


def test_mixed_combiner_rule_is_skipped():
    dr = parse_domain_rules("X: a > 1 AND b > 1 OR c > 1\nDEFAULT: Y\n")
    assert len(dr.rules) == 0  # ambiguous mixed AND/OR → dropped
    assert dr.default_label == "Y"


def test_normalized_maps_domain_thresholds_to_unit():
    dr = parse_domain_rules("HIGH: charlson > 15\nDEFAULT: LOW\n")
    norm = dr.normalized({"charlson": (0.0, 20.0)})
    # 15/20 = 0.75
    assert abs(norm.rules[0].clause.conditions[0].value - 0.75) < 1e-9
    # a normalized row at 0.8 (=16 domain) is HIGH; 0.7 (=14) is LOW
    assert norm.label_for({"charlson": 0.8}) == "HIGH"
    assert norm.label_for({"charlson": 0.7}) == "LOW"


def test_normalized_does_not_clamp_out_of_range_thresholds():
    # edad > 200 with range [0,120] → 1.67 (NOT clamped to 1.0); the condition is
    # then correctly never-true for normalized rows in [0,1].
    dr = parse_domain_rules("HIGH: edad > 200\nDEFAULT: LOW\n").normalized({"edad": (0, 120)})
    v = dr.rules[0].clause.conditions[0].value
    assert v > 1.0
    assert dr.label_for({"edad": 1.0}) == "LOW"  # max normalized value still not HIGH
    # an always-true lower bound also survives without clamping
    dr2 = parse_domain_rules("HIGH: edad > -50\nDEFAULT: LOW\n").normalized({"edad": (0, 120)})
    assert dr2.rules[0].clause.conditions[0].value < 0.0
    assert dr2.label_for({"edad": 0.0}) == "HIGH"


def test_normalized_keeps_value_when_no_range():
    dr = parse_domain_rules("HIGH: a > 0.5\nDEFAULT: LOW\n")
    norm = dr.normalized({})  # no range → unchanged
    assert norm.rules[0].clause.conditions[0].value == 0.5


def test_validate_ok():
    dr = parse_domain_rules("HIGH: a > 0.5\nMID: b > 0.3\nDEFAULT: LOW\n")
    assert dr.validate(["a", "b", "c"], ["HIGH", "MID", "LOW"]) == []


def test_validate_unknown_feature_and_label():
    dr = parse_domain_rules("HIGH: ghost > 0.5\nDEFAULT: LOW\n")
    errs = dr.validate(["a", "b"], ["HIGH", "LOW"])
    assert any("unknown features" in e for e in errs)


def test_validate_rejects_empty():
    assert "no rules parsed" in parse_domain_rules("DEFAULT: LOW\n").validate(["a"], ["LOW"])


def test_validate_requires_two_reachable_labels():
    # Only HIGH + default LOW reachable is fine; a single-label rule set is not.
    dr = parse_domain_rules("HIGH: a > 0.5\nDEFAULT: HIGH\n")
    errs = dr.validate(["a"], ["HIGH", "LOW"])
    assert any("fewer than 2" in e for e in errs)
