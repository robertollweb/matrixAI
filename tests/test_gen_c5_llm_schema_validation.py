"""GENERACIÓN tipos del prompt — C5: la propuesta LLM honra los tipos.

La salida del LLM (input_fields, labels, categorical_fields) se valida contra el
parser/política común de C1 ANTES de materializarse:
(a) si el LLM contradice un tipo explícito del prompt → gana el prompt (invariante 1);
(b) si el LLM omite metadata que el prompt declaró → se rellena desde las FieldSpec;
(c) si el LLM produce nombres/tipos inválidos → se normalizan o rechazan con warning,
    nunca pasan crudos al generador (un nombre crudo rompía el parser .mxai);
(d) sin LLM, el path determinista honra los mismos tipos (paridad).

Además alinea el umbral de embedding del caller/LLM (`vocab > 5` → `> _ONEHOT_MAX`)
y cierra el diferido de C2: una Categorical de ALTA cardinalidad declarada en el
prompt se rutea al path composite (embedding con vocab humano persistido).
"""
from __future__ import annotations

import unittest

from matrixai.parser import parse_text
from matrixai.training.dense_generator import DenseNetworkGenerator, _ONEHOT_MAX
from matrixai.training.composite_generator import CompositeNetworkGenerator


_TYPED_PROMPT = """clasificar clientes
FEATURES:
  edad: Scalar en [18, 95]
  activo: Boolean
  producto_id: Categorical[A, B, C]
SALIDA: y: ProbabilityMap[X, Y]
"""


class LlmContradictionLosesToPromptTest(unittest.TestCase):
    """(a) el LLM contradice un tipo explícito → gana el prompt (invariante 1)."""

    def test_llm_categorical_over_explicit_scalar_is_ignored(self):
        # Before C5: edad became Integer[0,49] embedding while field_ranges still
        # said (18,95) — an invariant-6 contradiction inside one result.
        r = CompositeNetworkGenerator().generate(_TYPED_PROMPT, categorical_fields={"edad": 50})
        emb = {e["field"] for e in r.embeddings}
        self.assertNotIn("edad", emb)
        self.assertEqual(parse_text(r.mxai_text).vectors[0].field_types["edad"].name, "Scalar")
        self.assertEqual(r.field_ranges["edad"], (18.0, 95.0))
        self.assertTrue(any("invariante 1" in w and "edad" in w for w in r.warnings), r.warnings)

    def test_llm_categorical_over_explicit_boolean_is_ignored(self):
        r = CompositeNetworkGenerator().generate(_TYPED_PROMPT, categorical_fields={"activo": 30})
        self.assertNotIn("activo", {e["field"] for e in r.embeddings})
        self.assertEqual(r.field_types["activo"], "boolean")
        self.assertTrue(any("invariante 1" in w and "activo" in w for w in r.warnings))

    def test_prompt_categorical_vocab_beats_llm_count(self):
        # already guarded since the C3 audit; kept here as the (a) family closer
        r = CompositeNetworkGenerator().generate(_TYPED_PROMPT, categorical_fields={"producto_id": 500})
        emb = {e["field"]: e["vocab"] for e in r.embeddings}
        self.assertEqual(emb.get("producto_id"), 3)
        self.assertEqual(r.field_categories["producto_id"], ["A", "B", "C"])


class LlmOmissionFilledFromPromptTest(unittest.TestCase):
    """(b) el LLM omite metadata declarada → se rellena desde las FieldSpec."""

    def test_llm_input_fields_missing_declared_fields_still_typed(self):
        # the LLM proposed only "edad" — the declared boolean and categorical
        # survive with their full metadata anyway.
        r = DenseNetworkGenerator().generate(_TYPED_PROMPT, input_fields=["edad"])
        self.assertEqual(r.field_ranges["edad"], (18.0, 95.0))
        self.assertEqual(r.field_types["activo"], "boolean")
        self.assertEqual(r.field_categories["producto_id"], ["A", "B", "C"])


class LlmInvalidNamesNormalizedTest(unittest.TestCase):
    """(c) nombres inválidos del LLM → normalizados/descartados, nunca crudos."""

    def test_invalid_field_names_are_sanitized_and_mxai_parses(self):
        # Before C5: "customer age" went verbatim into the VECTOR block and
        # raised MatrixAIParseError downstream.
        r = DenseNetworkGenerator().generate(
            "clasificar", input_fields=["customer age", "saldo-medio", "edad"]
        )
        fields = parse_text(r.mxai_text).vectors[0].fields
        self.assertEqual(fields, ["customer_age", "saldo_medio", "edad"])
        self.assertTrue(any("customer age" in w and "normalizado" in w for w in r.warnings))

    def test_unrecoverable_name_is_dropped_with_warning(self):
        r = DenseNetworkGenerator().generate(
            "clasificar", input_fields=["123", "edad", "saldo"]
        )
        self.assertNotIn("123", r.mxai_text)
        self.assertIn("edad", parse_text(r.mxai_text).vectors[0].fields)
        self.assertTrue(any("descartado" in w for w in r.warnings))

    def test_parse_dense_schema_rejects_invalid_categorical_entries(self):
        from matrixai.playground import _parse_dense_schema
        r = _parse_dense_schema(
            "FIELDS: a, b\nCATEGORICALS: ok_field:50, low:1, bad:abc, neg:-3\n"
        )
        self.assertEqual(r.get("categorical_fields"), {"ok_field": 50})


class ThresholdAlignedWithOnehotMaxTest(unittest.TestCase):
    """Umbral caller/LLM alineado: vocab ≤ _ONEHOT_MAX es territorio one-hot."""

    def test_vocab_at_threshold_is_dropped_with_warning(self):
        r = CompositeNetworkGenerator().generate(
            "clasificar x y con red", input_fields=["saldo", "region"],
            categorical_fields={"region": _ONEHOT_MAX},
        )
        self.assertEqual(r.embeddings, [])
        self.assertTrue(any("territorio one-hot" in w for w in r.warnings), r.warnings)

    def test_vocab_above_threshold_is_embedding(self):
        r = CompositeNetworkGenerator().generate(
            "clasificar x y con red", input_fields=["saldo", "producto"],
            categorical_fields={"producto": _ONEHOT_MAX + 1},
        )
        emb = {e["field"]: e["vocab"] for e in r.embeddings}
        self.assertEqual(emb.get("producto"), _ONEHOT_MAX + 1)

    def test_prompt_declared_low_card_in_composite_still_embedding(self):
        # policy per-path (C2 clarification): a PROMPT-declared categorical in the
        # composite path is ALWAYS an embedding, regardless of cardinality — the
        # threshold only gates count-only caller/LLM proposals.
        r = CompositeNetworkGenerator().generate(
            "clasificar con bloques residuales\nFEATURES:\n"
            "  categoria: Categorical[A, B, C]\nSALIDA: y: ProbabilityMap[X, Y]",
            force_residual=True,
        )
        self.assertEqual({e["field"]: e["vocab"] for e in r.embeddings}, {"categoria": 3})


_HIGHCARD_VALUES = ", ".join(f"V{i}" for i in range(_ONEHOT_MAX + 3))
_HIGHCARD_PROMPT = (
    "clasificar maquinas\nFEATURES:\n"
    "  temp: Scalar en [0, 100]\n"
    f"  modelo: Categorical[{_HIGHCARD_VALUES}]\n"
    "SALIDA: y: ProbabilityMap[OK, KO]"
)


class HighCardPromptCategoricalRoutingTest(unittest.TestCase):
    """Diferido de C2, cerrado en C5: Categorical de alta cardinalidad declarada
    en el prompt → path composite → EMBEDDING con vocab humano persistido."""

    def test_dispatch_routes_highcard_prompt_to_composite_embedding(self):
        from matrixai.playground import analyze_playground_request
        res = analyze_playground_request(
            {"mode": "prompt", "prompt": _HIGHCARD_PROMPT, "use_llm": False}
        )
        self.assertTrue(res.get("ok"), res.get("error"))
        self.assertEqual(res.get("supervision_source"), "composite_generator")
        self.assertIn("EMBEDDING", res["mxai"])
        # human vocab persisted -> the export will give vocab:[...] not vocab_size
        self.assertEqual(len(res["field_categories"]["modelo"]), _ONEHOT_MAX + 3)
        self.assertEqual(res["field_ranges"]["temp"], (0.0, 100.0))

    def test_direct_dense_call_warns_and_leaves_scalar(self):
        r = DenseNetworkGenerator().generate(_HIGHCARD_PROMPT)
        self.assertNotIn("EMBEDDING", r.mxai_text)
        self.assertEqual(r.field_categories, {})
        self.assertTrue(any("composite" in w and "modelo" in w for w in r.warnings), r.warnings)


class DeterministicParityTest(unittest.TestCase):
    """(d) sin LLM, el determinista honra los mismos tipos: un eco del LLM con la
    misma información no cambia la metadata resultante."""

    def test_echoed_llm_kwargs_produce_same_type_metadata(self):
        base = DenseNetworkGenerator().generate(_TYPED_PROMPT)
        echoed = DenseNetworkGenerator().generate(
            _TYPED_PROMPT,
            input_fields=["edad", "activo", "producto_id"],
            labels=["x", "y"],
            network_name=base.network_name,
        )
        self.assertEqual(base.field_ranges, echoed.field_ranges)
        self.assertEqual(base.field_types, echoed.field_types)
        self.assertEqual(base.field_categories, echoed.field_categories)
        self.assertEqual(base.labels, echoed.labels)
        self.assertEqual(base.output_type, echoed.output_type)

    def test_dispatch_without_llm_same_as_generator(self):
        from matrixai.playground import analyze_playground_request
        res = analyze_playground_request(
            {"mode": "prompt", "prompt": _TYPED_PROMPT, "use_llm": False}
        )
        self.assertTrue(res.get("ok"), res.get("error"))
        gen = DenseNetworkGenerator().generate(_TYPED_PROMPT)
        self.assertEqual(res["field_ranges"], gen.field_ranges)
        self.assertEqual(res["field_types"], gen.field_types)
        self.assertEqual(res["field_categories"], gen.field_categories)
