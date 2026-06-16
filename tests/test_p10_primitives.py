"""P10 Cut 4 — Tensor primitives: dot, matmul, relu, gelu, layer_norm, residual."""
from __future__ import annotations

import unittest

from matrixai.compiler import BackendContractAnalyzer
from matrixai.parser import parse_text


def _make_program(expr: str, extra_vectors: str = "") -> str:
    return (
        "PROJECT T\n"
        "VECTOR A[2]\n  x : Scalar\n  y : Scalar\nEND\n"
        "VECTOR B[2]\n  p : Scalar\n  q : Scalar\nEND\n"
        + extra_vectors
        + f"FUNCTION F\n  result = {expr}\nEND\n"
        "GRAPH\n  A -> F\nEND\n"
    )


class TensorPrimitiveParseTest(unittest.TestCase):
    def test_dot_kind(self) -> None:
        p = parse_text(_make_program("dot(A, B)"))
        self.assertEqual(p.functions[0].semantic.kind, "dot")

    def test_dot_inputs(self) -> None:
        p = parse_text(_make_program("dot(A, B)"))
        self.assertIn("A", p.functions[0].semantic.inputs)
        self.assertIn("B", p.functions[0].semantic.inputs)

    def test_matmul_kind(self) -> None:
        p = parse_text(_make_program("matmul(A, B)"))
        self.assertEqual(p.functions[0].semantic.kind, "matmul")

    def test_relu_kind(self) -> None:
        p = parse_text(_make_program("relu(A)"))
        self.assertEqual(p.functions[0].semantic.kind, "relu")

    def test_relu_inputs(self) -> None:
        p = parse_text(_make_program("relu(A)"))
        self.assertIn("A", p.functions[0].semantic.inputs)

    def test_gelu_kind(self) -> None:
        p = parse_text(_make_program("gelu(A)"))
        self.assertEqual(p.functions[0].semantic.kind, "gelu")

    def test_layer_norm_kind(self) -> None:
        p = parse_text(
            _make_program(
                "layer_norm(A, gain, bias, eps)",
                "VECTOR gain[2]\n  g0 : Scalar\n  g1 : Scalar\nEND\n"
                "VECTOR bias[2]\n  b0 : Scalar\n  b1 : Scalar\nEND\n"
                "VECTOR eps[1]\n  e : Scalar\nEND\n",
            )
        )
        self.assertEqual(p.functions[0].semantic.kind, "layer_norm")

    def test_layer_norm_inputs(self) -> None:
        p = parse_text(
            _make_program(
                "layer_norm(A, gain, bias, eps)",
                "VECTOR gain[2]\n  g0 : Scalar\n  g1 : Scalar\nEND\n"
                "VECTOR bias[2]\n  b0 : Scalar\n  b1 : Scalar\nEND\n"
                "VECTOR eps[1]\n  e : Scalar\nEND\n",
            )
        )
        fn = p.functions[0]
        self.assertIn("A", fn.semantic.inputs)
        self.assertIn("gain", fn.semantic.inputs)

    def test_residual_kind(self) -> None:
        p = parse_text(_make_program("residual(A, B)"))
        self.assertEqual(p.functions[0].semantic.kind, "residual")

    def test_residual_inputs(self) -> None:
        p = parse_text(_make_program("residual(A, B)"))
        fn = p.functions[0]
        self.assertIn("A", fn.semantic.inputs)
        self.assertIn("B", fn.semantic.inputs)

    def test_primitive_has_ast_in_parameters(self) -> None:
        p = parse_text(_make_program("relu(A)"))
        self.assertIn("ast", p.functions[0].semantic.parameters)


class TensorPrimitiveBackendContractTest(unittest.TestCase):
    def _analyze(self, expr: str, extra_vectors: str = ""):
        program = parse_text(_make_program(expr, extra_vectors))
        return BackendContractAnalyzer().analyze(program)

    def test_dot_is_supported(self) -> None:
        report = self._analyze("dot(A, B)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)
        self.assertTrue(fn_node.differentiable)

    def test_matmul_is_supported(self) -> None:
        report = self._analyze("matmul(A, B)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)

    def test_relu_is_supported(self) -> None:
        report = self._analyze("relu(A)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)
        self.assertTrue(fn_node.differentiable)

    def test_gelu_is_supported(self) -> None:
        report = self._analyze("gelu(A)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)

    def test_layer_norm_is_supported(self) -> None:
        report = self._analyze(
            "layer_norm(A, gain, bias, eps)",
            "VECTOR gain[2]\n  g0 : Scalar\n  g1 : Scalar\nEND\n"
            "VECTOR bias[2]\n  b0 : Scalar\n  b1 : Scalar\nEND\n"
            "VECTOR eps[1]\n  e : Scalar\nEND\n",
        )
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)

    def test_residual_is_supported(self) -> None:
        report = self._analyze("residual(A, B)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertTrue(fn_node.supported)

    def test_dot_kind_in_node_report(self) -> None:
        report = self._analyze("dot(A, B)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertEqual(fn_node.kind, "dot")

    def test_relu_kind_in_node_report(self) -> None:
        report = self._analyze("relu(A)")
        fn_node = next(n for n in report.nodes if n.node == "F")
        self.assertEqual(fn_node.kind, "relu")


class TensorPrimitiveBackwardCompatTest(unittest.TestCase):
    """Existing programs must not be affected by adding tensor primitives."""

    def test_softmax_linear_unchanged(self) -> None:
        from matrixai.types import check_program_types
        program = parse_text(
            "PROJECT E\nVECTOR Input[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Tensor[3, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * Input + b)\nEND\n"
            "GRAPH\n  Input -> F\nEND\n"
        )
        result = check_program_types(program)
        self.assertEqual(result.errors, [])

    def test_symbolic_expr_unchanged(self) -> None:
        program = parse_text(
            "PROJECT S\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "FUNCTION F\n  result = x * 2.0\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        self.assertEqual(program.functions[0].semantic.kind, "symbolic_expr")


if __name__ == "__main__":
    unittest.main()
