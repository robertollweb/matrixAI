"""GENERACIÓN tipos del prompt — C3 (audit): el generador COMPOSITE honra los tipos.

Antes, CompositeNetworkGenerator usaba _extract_fields(clean) directamente (→ feature_1..4
con prompts tipados) y su resultado no llevaba field_ranges/field_types. Ahora usa la misma
resolución compartida (resolve_prompt_fields, invariante 5) y persiste la metadata; las
categóricas declaradas se materializan como EMBEDDING con vocab humano.
"""
from __future__ import annotations

import unittest

from matrixai.training.composite_generator import CompositeNetworkGenerator
from matrixai.parser import parse_text


_RESIDUAL_PROMPT = """clasificar bajo medio alto con bloques residuales complejo no lineal
FEATURES:
  edad: Scalar en [18, 95]
  saldo: Scalar en [0, 500000]
  activo: Boolean
  cod_region: Integer[0, 10]
  temp: Scalar en [0, 100]
  presion: Scalar en [0, 10]
SALIDA: y: ProbabilityMap[BAJO, MEDIO, ALTO]
"""


class CompositeHonorsPromptTypesTest(unittest.TestCase):
    def setUp(self):
        self.r = CompositeNetworkGenerator().generate(_RESIDUAL_PROMPT, force_residual=True)

    def test_clean_field_names_not_placeholders(self):
        # regression: used to emit feature_1..feature_4 for typed prompts
        fields = parse_text(self.r.mxai_text).vectors[0].fields
        self.assertIn("edad", fields)
        self.assertIn("activo", fields)
        self.assertNotIn("feature_1", fields)

    def test_field_ranges_persisted(self):
        self.assertEqual(self.r.field_ranges["edad"], (18.0, 95.0))
        self.assertEqual(self.r.field_ranges["saldo"], (0.0, 500000.0))
        self.assertEqual(self.r.field_ranges["cod_region"], (0.0, 10.0))

    def test_field_types_persisted(self):
        self.assertEqual(self.r.field_types, {"activo": "boolean", "cod_region": "integer"})

    def test_mxai_vector_stays_bare_scalar(self):
        prog = parse_text(self.r.mxai_text)
        self.assertIsNone(prog.vectors[0].field_types["edad"].range)

    def test_is_composite_with_residual_block(self):
        self.assertTrue(self.r.is_composite)
        self.assertTrue(self.r.blocks)


class CompositeCategoricalBecomesEmbeddingTest(unittest.TestCase):
    def test_declared_categorical_is_embedding_with_human_vocab(self):
        r = CompositeNetworkGenerator().generate(
            "clasificar con bloques residuales\nFEATURES:\n"
            "  precio: Scalar en [0, 1000]\n"
            "  categoria: Categorical[A, B, C, D, E, F, G, H]\n"
            "SALIDA: y: ProbabilityMap[X, Y]",
            force_residual=True,
        )
        # the human vocab is persisted for the export (vocab: [...], not vocab_size)
        self.assertEqual(r.field_categories, {"categoria": list("ABCDEFGH")})
        emb = {e["field"]: e["vocab"] for e in r.embeddings}
        self.assertEqual(emb.get("categoria"), 8)

    def test_prompt_categorical_wins_over_llm_categorical_fields(self):
        # invariant 1: a categorical declared in the prompt wins over categorical_fields.
        r = CompositeNetworkGenerator().generate(
            "clasificar con bloques residuales\nFEATURES:\n"
            "  categoria: Categorical[A, B, C]\n"
            "SALIDA: y: ProbabilityMap[X, Y]",
            categorical_fields={"categoria": 50}, force_residual=True,
        )
        self.assertEqual(r.field_categories["categoria"], ["A", "B", "C"])
        emb = {e["field"]: e["vocab"] for e in r.embeddings}
        self.assertEqual(emb.get("categoria"), 3)  # prompt's 3, not the LLM's 50


class DispatchSurfacesCompositeMetadataTest(unittest.TestCase):
    def test_analyze_playground_request_surfaces_composite_metadata(self):
        from matrixai.playground import analyze_playground_request
        res = analyze_playground_request(
            {"mode": "prompt", "prompt": _RESIDUAL_PROMPT, "use_llm": False}
        )
        self.assertTrue(res.get("ok"), res.get("error"))
        self.assertEqual(res.get("field_types"), {"activo": "boolean", "cod_region": "integer"})
        self.assertEqual(res.get("field_ranges", {}).get("edad"), (18.0, 95.0))
