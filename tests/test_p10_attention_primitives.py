"""P10 Cut 5 — Attention and embedding primitives: embedding_lookup, positional_encoding,
attention, mean_pooling, cls_pooling."""
from __future__ import annotations

import unittest

from matrixai.compiler import BackendContractAnalyzer
from matrixai.parser import parse_text


def _make_program(expr: str, extra_vectors: str = "") -> str:
    return (
        "PROJECT T\n"
        "VECTOR Input[4]\n  a : Scalar\n  b : Scalar\n  c : Scalar\n  d : Scalar\nEND\n"
        "VECTOR Ids[2]\n  i0 : Scalar\n  i1 : Scalar\nEND\n"
        + extra_vectors
        + f"FUNCTION F\n  result = {expr}\nEND\n"
        "GRAPH\n  Input -> F\nEND\n"
    )


class AttentionPrimitiveParseTest(unittest.TestCase):
    def test_embedding_lookup_kind(self) -> None:
        p = parse_text(
            _make_program(
                "embedding_lookup(Table, Ids)",
                "VECTOR Table[4]\n  e0 : Scalar\n  e1 : Scalar\n  e2 : Scalar\n  e3 : Scalar\nEND\n",
            )
        )
        self.assertEqual(p.functions[0].semantic.kind, "embedding_lookup")

    def test_embedding_lookup_inputs(self) -> None:
        p = parse_text(
            _make_program(
                "embedding_lookup(Table, Ids)",
                "VECTOR Table[4]\n  e0 : Scalar\n  e1 : Scalar\n  e2 : Scalar\n  e3 : Scalar\nEND\n",
            )
        )
        fn = p.functions[0]
        self.assertIn("Table", fn.semantic.inputs)
        self.assertIn("Ids", fn.semantic.inputs)

    def test_positional_encoding_kind(self) -> None:
        p = parse_text(
            _make_program(
                "positional_encoding(Input, Ids)",
            )
        )
        self.assertEqual(p.functions[0].semantic.kind, "positional_encoding")

    def test_attention_kind(self) -> None:
        p = parse_text(
            _make_program(
                "attention(Input, Input, Input)",
            )
        )
        self.assertEqual(p.functions[0].semantic.kind, "attention")

    def test_attention_inputs(self) -> None:
        p = parse_text(
            _make_program(
                "attention(Input, Input, Input)",
            )
        )
        fn = p.functions[0]
        self.assertIn("Input", fn.semantic.inputs)

    def test_mean_pooling_kind(self) -> None:
        p = parse_text(_make_program("mean_pooling(Input, Ids)"))
        self.assertEqual(p.functions[0].semantic.kind, "mean_pooling")

    def test_cls_pooling_kind(self) -> None:
        p = parse_text(_make_program("cls_pooling(Input)"))
        self.assertEqual(p.functions[0].semantic.kind, "cls_pooling")

    def test_all_primitives_have_ast(self) -> None:
        for expr in ["embedding_lookup(Input, Ids)", "attention(Input, Input, Input)",
                     "mean_pooling(Input, Ids)", "cls_pooling(Input)"]:
            p = parse_text(_make_program(expr))
            self.assertIn("ast", p.functions[0].semantic.parameters, expr)


class AttentionPrimitiveBackendContractTest(unittest.TestCase):
    def _analyze(self, expr: str, extra_vectors: str = ""):
        program = parse_text(_make_program(expr, extra_vectors))
        return BackendContractAnalyzer().analyze(program)

    def test_embedding_lookup_supported_not_differentiable(self) -> None:
        report = self._analyze(
            "embedding_lookup(Table, Ids)",
            "VECTOR Table[4]\n  e0 : Scalar\n  e1 : Scalar\n  e2 : Scalar\n  e3 : Scalar\nEND\n",
        )
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)
        self.assertFalse(fn_node.differentiable)

    def test_attention_supported_not_differentiable(self) -> None:
        report = self._analyze("attention(Input, Input, Input)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)
        self.assertFalse(fn_node.differentiable)

    def test_mean_pooling_supported_differentiable(self) -> None:
        report = self._analyze("mean_pooling(Input, Ids)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)
        self.assertTrue(fn_node.differentiable)

    def test_cls_pooling_supported_differentiable(self) -> None:
        report = self._analyze("cls_pooling(Input)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)
        self.assertTrue(fn_node.differentiable)

    def test_positional_encoding_supported_differentiable(self) -> None:
        report = self._analyze("positional_encoding(Input, Ids)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)
        self.assertTrue(fn_node.differentiable)

    def test_embedding_lookup_kind_in_node(self) -> None:
        report = self._analyze(
            "embedding_lookup(Table, Ids)",
            "VECTOR Table[4]\n  e0 : Scalar\n  e1 : Scalar\n  e2 : Scalar\n  e3 : Scalar\nEND\n",
        )
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertEqual(fn_node.kind, "embedding_lookup")

    def test_attention_kind_in_node(self) -> None:
        report = self._analyze("attention(Input, Input, Input)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertEqual(fn_node.kind, "attention")


if __name__ == "__main__":
    unittest.main()
