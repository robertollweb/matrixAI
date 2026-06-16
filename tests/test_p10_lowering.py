"""P10 Cut 7 — Lowering to differentiable_python: tensor primitives and layer_call."""
from __future__ import annotations

import types
import unittest

from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
from matrixai.parser import parse_text


def _compile_and_run(program_text: str, input_data: dict, params: dict | None = None):
    program = parse_text(program_text)
    source = DifferentiablePythonCompiler().compile(program)
    module = types.ModuleType("_p10_test")
    exec(source, module.__dict__)  # noqa: S102
    return module.run(input_data, params)


_DOT_PROGRAM = """\
PROJECT DotTest
VECTOR A[2]
  x : Scalar
  y : Scalar
END
FUNCTION F
  result = dot(A, A)
END
GRAPH
  A -> F
END
"""

_RELU_PROGRAM = """\
PROJECT ReluTest
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
FUNCTION F
  result = relu(Input)
END
GRAPH
  Input -> F
END
"""

_GELU_PROGRAM = """\
PROJECT GeluTest
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
FUNCTION F
  result = gelu(Input)
END
GRAPH
  Input -> F
END
"""

_RESIDUAL_PROGRAM = """\
PROJECT ResidualTest
VECTOR A[2]
  x : Scalar
  y : Scalar
END
FUNCTION F
  result = residual(A, A)
END
GRAPH
  A -> F
END
"""

_LAYER_CALL_PROGRAM = """\
PROJECT LayerTest
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
LAYER Identity
  PARAM W Tensor[2, 2]
END
FUNCTION F
  result = call_layer(Identity, Input)
END
GRAPH
  Input -> F
END
"""


class TensorPrimitiveLoweringTest(unittest.TestCase):
    def test_dot_compiles(self) -> None:
        source = DifferentiablePythonCompiler().compile(parse_text(_DOT_PROGRAM))
        self.assertIn("_tensor_dot", source)

    def test_dot_runs(self) -> None:
        result = _compile_and_run(_DOT_PROGRAM, {"A": {"x": 3.0, "y": 4.0}})
        self.assertIn("F", result["state"])
        # dot([3, 4], [3, 4]) = 9 + 16 = 25
        self.assertAlmostEqual(float(result["state"]["F"]), 25.0)

    def test_relu_compiles(self) -> None:
        source = DifferentiablePythonCompiler().compile(parse_text(_RELU_PROGRAM))
        self.assertIn("_tensor_relu", source)

    def test_relu_runs_positive(self) -> None:
        result = _compile_and_run(_RELU_PROGRAM, {"Input": {"x": 2.0, "y": 3.0}})
        out = result["state"]["F"]
        self.assertEqual(out, [2.0, 3.0])

    def test_relu_runs_negative(self) -> None:
        result = _compile_and_run(_RELU_PROGRAM, {"Input": {"x": -1.0, "y": 0.5}})
        out = result["state"]["F"]
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[1], 0.5)

    def test_gelu_compiles(self) -> None:
        source = DifferentiablePythonCompiler().compile(parse_text(_GELU_PROGRAM))
        self.assertIn("_tensor_gelu", source)

    def test_gelu_runs(self) -> None:
        result = _compile_and_run(_GELU_PROGRAM, {"Input": {"x": 1.0, "y": -1.0}})
        out = result["state"]["F"]
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        # GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        self.assertGreater(out[0], 0.8)
        self.assertLess(out[1], 0.0)

    def test_residual_compiles(self) -> None:
        source = DifferentiablePythonCompiler().compile(parse_text(_RESIDUAL_PROGRAM))
        self.assertIn("_tensor_residual", source)

    def test_residual_runs(self) -> None:
        result = _compile_and_run(_RESIDUAL_PROGRAM, {"A": {"x": 1.0, "y": 2.0}})
        out = result["state"]["F"]
        # residual([1, 2], [1, 2]) = [2, 4]
        self.assertAlmostEqual(out[0], 2.0)
        self.assertAlmostEqual(out[1], 4.0)

    def test_layer_call_compiles(self) -> None:
        source = DifferentiablePythonCompiler().compile(parse_text(_LAYER_CALL_PROGRAM))
        self.assertIn("_layer_call_passthrough", source)

    def test_layer_call_runs(self) -> None:
        result = _compile_and_run(_LAYER_CALL_PROGRAM, {"Input": {"x": 1.0, "y": 2.0}})
        self.assertIn("F", result["state"])

    def test_layer_call_produces_list(self) -> None:
        result = _compile_and_run(_LAYER_CALL_PROGRAM, {"Input": {"x": 3.0, "y": 4.0}})
        out = result["state"]["F"]
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)


class LoweringBackwardCompatTest(unittest.TestCase):
    """Existing programs must still compile and run correctly."""

    def test_softmax_linear_unchanged(self) -> None:
        text = (
            "PROJECT E\nVECTOR Input[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Tensor[3, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * Input + b)\nEND\n"
            "GRAPH\n  Input -> F\nEND\n"
        )
        program = parse_text(text)
        source = DifferentiablePythonCompiler().compile(program)
        module = types.ModuleType("_test_compat")
        exec(source, module.__dict__)  # noqa: S102
        result = module.run({"Input": {"a": 1.0, "b": 2.0}})
        self.assertIn("F", result["state"])

    def test_symbolic_expr_unchanged(self) -> None:
        from matrixai.compiler.python_backend import PythonBackendCompiler
        text = (
            "PROJECT S\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "FUNCTION F\n  result = x * 2.0\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        program = parse_text(text)
        source = PythonBackendCompiler().compile(program)
        module = types.ModuleType("_test_sym")
        exec(source, module.__dict__)  # noqa: S102
        result = module.run({"V": {"x": 3.0}})
        self.assertAlmostEqual(float(result["state"]["F"]), 6.0)


if __name__ == "__main__":
    unittest.main()
