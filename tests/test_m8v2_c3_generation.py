"""M8 v2 — C3: reglas de dominio cableadas en la generación.

El generador etiqueta con el evaluador determinista (no el runtime de juguete) cuando
recibe domain_rules; `_generate_synthetic_dataset` las pide al LLM, valida, normaliza y
expone `label_origin=synthetic_domain` + reglas + aviso. Sin LLM / reglas inválidas →
fallback a coherent.
"""
from __future__ import annotations

import collections
import csv as csvmod
import io
from unittest.mock import patch

import pytest

from matrixai.parser import parse_text
from matrixai.training.parser import parse_training_text
from matrixai.training.synthetic import SyntheticDataGenerator
from matrixai.training.domain_rules import parse_domain_rules
from matrixai.playground import _generate_synthetic_dataset

MXAI = """PROJECT Reingreso
VECTOR Patient[3]
  indice_charlson: Scalar
  creatinina: Scalar
  edad: Scalar
END
NETWORK Net
  INPUT Patient
  LAYER Dense units=8 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT riesgo: ProbabilityMap[BAJO, MEDIO, ALTO]
END
GRAPH
  Patient -> Net
END
"""
TRAIN = """MODEL Reingreso.mxai
DATASET D
  SOURCE csv("d.csv")
  INPUT Patient FROM COLUMNS [indice_charlson, creatinina, edad]
  TARGET riesgo: Label[BAJO, MEDIO, ALTO]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END
LOSS L
  TYPE cross_entropy
  PREDICTION Net
  TARGET riesgo
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END
"""
RULES = "ALTO: indice_charlson > 0.75 OR creatinina > 0.6\nMEDIO: edad > 0.6\nDEFAULT: BAJO\n"


class _FakeProvider:
    def complete(self, system, user):  # noqa: ARG002
        # domain-scale rules; _generate normalizes with the field ranges below
        return ("ALTO: indice_charlson > 15 OR creatinina > 9\n"
                "MEDIO: edad > 72\n"
                "DEFAULT: BAJO\n")


def test_generator_uses_domain_rules_not_runtime():
    prog = parse_text(MXAI)
    tr = parse_training_text(TRAIN)
    rules = parse_domain_rules(RULES)  # already in [0,1]
    gen = SyntheticDataGenerator(prog, tr, seed=7, rows=40, mode="coherent",
                                 domain_rules=rules)
    adapter = gen.generate()
    assert gen.domain_rules_used == 40
    # every labelled row obeys the rule (deterministic, not toy/random)
    for row in adapter.rows:
        if row["indice_charlson"] > 0.75 or row["creatinina"] > 0.6:
            assert row["riesgo"] == "ALTO"


def test_domain_rules_skew_not_rebalanced():
    # A rule that makes ALTO rare must stay rare (no rebalancing override).
    prog = parse_text(MXAI)
    tr = parse_training_text(TRAIN)
    rules = parse_domain_rules("ALTO: indice_charlson > 0.95 AND creatinina > 0.95\nDEFAULT: BAJO\n")
    gen = SyntheticDataGenerator(prog, tr, seed=3, rows=60, mode="coherent", domain_rules=rules)
    adapter = gen.generate()
    dist = collections.Counter(r["riesgo"] for r in adapter.rows)
    assert dist["ALTO"] < dist["BAJO"]  # skew preserved


def test_generate_synthetic_dataset_domain_origin_with_llm():
    with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}), \
         patch("matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
               return_value=_FakeProvider()), \
         patch("matrixai.playground._llm_field_ranges",
               return_value={"indice_charlson": (0, 20), "creatinina": (0, 15), "edad": (0, 120)}):
        r = _generate_synthetic_dataset(MXAI, TRAIN, 50, 7, "coherent", use_llm=True)
    assert r["label_origin"] == "synthetic_domain"
    assert "ALTO:" in r["domain_rules"] and "DEFAULT: BAJO" in r["domain_rules"]
    assert "dominio" in r["domain_notice"].lower()
    # learnable signal: a clearly high-charlson row is ALTO
    rows = list(csvmod.DictReader(io.StringIO(r["csv_text"])))
    highs = [x for x in rows if float(x["indice_charlson"]) > 16]
    assert highs and all(x["riesgo"] == "ALTO" for x in highs)


def test_fallback_to_coherent_without_llm():
    # use_llm False → no domain rules → toy coherent, no domain fields
    r = _generate_synthetic_dataset(MXAI, TRAIN, 30, 7, "coherent", use_llm=False)
    assert r["label_origin"] == "synthetic_coherent"
    assert "domain_rules" not in r


def test_warns_when_declared_classes_are_missing():
    # Rules that only ever yield BAJO or ALTO (MEDIO unreachable) → 2 of 3 classes.
    # Not degenerate (≥2 present) → no fallback, but a missing-classes warning.
    class _PartialProvider:
        def complete(self, system, user):  # noqa: ARG002
            return "ALTO: indice_charlson > 0.95\nDEFAULT: BAJO\n"
    with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}), \
         patch("matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
               return_value=_PartialProvider()), \
         patch("matrixai.playground._llm_field_ranges",
               return_value={"indice_charlson": (0, 1), "creatinina": (0, 1), "edad": (0, 1)}):
        r = _generate_synthetic_dataset(MXAI, TRAIN, 60, 7, "coherent", use_llm=True)
    assert r["label_origin"] == "synthetic_domain"  # not degenerate
    assert "MEDIO" in r["missing_labels"]
    assert "MEDIO" in r["missing_classes_warning"]


def test_no_missing_warning_when_all_classes_present():
    # Coherent rebalanced data should cover all classes → no missing-classes warning.
    r = _generate_synthetic_dataset(MXAI, TRAIN, 60, 7, "coherent", use_llm=False)
    assert not r.get("missing_labels")
    assert "missing_classes_warning" not in r


def test_fallback_when_domain_rules_collapse_to_one_class():
    # Syntactically valid rules, but an impossible threshold (charlson > 500) means
    # every row falls to DEFAULT → single class. This must fall back to coherent and
    # warn, not ship a useless single-class dataset as "synthetic_domain".
    class _CollapseProvider:
        def complete(self, system, user):  # noqa: ARG002
            return "ALTO: indice_charlson > 500\nMEDIO: creatinina > 500\nDEFAULT: BAJO\n"
    with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}), \
         patch("matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
               return_value=_CollapseProvider()), \
         patch("matrixai.playground._llm_field_ranges",
               return_value={"indice_charlson": (0, 20), "creatinina": (0, 15), "edad": (0, 120)}):
        r = _generate_synthetic_dataset(MXAI, TRAIN, 40, 7, "coherent", use_llm=True)
    assert r["label_origin"] == "synthetic_coherent"  # fell back
    assert "domain_rules" not in r
    assert "domain_degenerate_warning" in r


def test_fallback_when_llm_rules_invalid():
    class _BadProvider:
        def complete(self, system, user):  # noqa: ARG002
            return "sorry, I cannot do that"  # unparseable → invalid
    with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}), \
         patch("matrixai.playground.ChatCompletionsLLMProposalProvider.from_env",
               return_value=_BadProvider()), \
         patch("matrixai.playground._llm_field_ranges", return_value={}):
        r = _generate_synthetic_dataset(MXAI, TRAIN, 30, 7, "coherent", use_llm=True)
    assert r["label_origin"] == "synthetic_coherent"
    assert "domain_rules" not in r
