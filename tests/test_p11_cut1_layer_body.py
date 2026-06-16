"""P11 Cut 1 — LAYER body execution + hierarchical UPDATE patterns.

Tests for:
- LayerBodyOp IR parsed from LAYER body lines
- DifferentiablePythonCompiler generates _layer_exec_<name> functions
- Executed LAYER body returns correct output (not passthrough)
- match_update_patterns handles exact paths, wildcards, and *
- GenericSupervisedTrainer resolves trainable keys from patterns
"""
from __future__ import annotations

import unittest

from matrixai.ir import LayerBodyOp, LayerSpec
from matrixai.parser import parse_text
from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
from matrixai.training.trainer import match_update_patterns


# ---------------------------------------------------------------------------
# Helper programs
# ---------------------------------------------------------------------------

_PROG_LAYER_BODY = """\
PROJECT layer_body_test
LAYER Scale(Tensor[4]) -> Tensor[4]
  PARAM W Tensor[4, 4]
  result = matmul(input, W)
END
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(Scale, X)
END
GRAPH
  X -> F
END
"""

_PROG_BODY_CHAIN = """\
PROJECT chain_body_test
LAYER Chain(Tensor[4]) -> Tensor[4]
  PARAM W Tensor[4, 4]
  hidden = matmul(input, W)
  result = relu(hidden)
END
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(Chain, X)
END
GRAPH
  X -> F
END
"""

_PROG_NO_BODY = """\
PROJECT no_body_test
LAYER Empty
  PARAM W Tensor[4, 4]
END
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(Empty, X)
END
GRAPH
  X -> F
END
"""


# ---------------------------------------------------------------------------
# IR: LayerBodyOp parsed from body lines
# ---------------------------------------------------------------------------

class TestLayerBodyOp(unittest.TestCase):

    def test_single_body_op_parsed(self):
        program = parse_text(_PROG_LAYER_BODY)
        layer = next(l for l in program.layers if l.name == "Scale")
        self.assertEqual(len(layer.body_ops), 1)
        op = layer.body_ops[0]
        self.assertIsInstance(op, LayerBodyOp)
        self.assertEqual(op.output, "result")
        self.assertEqual(op.kind, "matmul")
        self.assertEqual(op.args, ("input", "W"))

    def test_chain_body_ops_parsed(self):
        program = parse_text(_PROG_BODY_CHAIN)
        layer = next(l for l in program.layers if l.name == "Chain")
        self.assertEqual(len(layer.body_ops), 2)
        self.assertEqual(layer.body_ops[0].output, "hidden")
        self.assertEqual(layer.body_ops[0].kind, "matmul")
        self.assertEqual(layer.body_ops[1].output, "result")
        self.assertEqual(layer.body_ops[1].kind, "relu")

    def test_no_body_ops_when_only_params(self):
        program = parse_text(_PROG_NO_BODY)
        layer = next(l for l in program.layers if l.name == "Empty")
        self.assertEqual(len(layer.body_ops), 0)

    def test_body_ops_in_ir_to_dict(self):
        program = parse_text(_PROG_LAYER_BODY)
        d = program.to_dict()
        layers = d.get("layers", [])
        scale = next(l for l in layers if l["name"] == "Scale")
        self.assertIn("body_ops", scale)
        self.assertEqual(scale["body_ops"][0]["output"], "result")
        self.assertEqual(scale["body_ops"][0]["kind"], "matmul")


# ---------------------------------------------------------------------------
# Compiler: _layer_exec_<name> generated when body_ops present
# ---------------------------------------------------------------------------

class TestLayerExecutorGenerated(unittest.TestCase):

    def test_executor_function_in_compiled_source(self):
        program = parse_text(_PROG_LAYER_BODY)
        source = DifferentiablePythonCompiler().compile(program)
        self.assertIn("def _layer_exec_Scale(", source)

    def test_passthrough_when_no_body_ops(self):
        program = parse_text(_PROG_NO_BODY)
        source = DifferentiablePythonCompiler().compile(program)
        # No executor generated; call_layer uses passthrough
        self.assertNotIn("def _layer_exec_Empty(", source)
        self.assertIn("_layer_call_passthrough", source)

    def test_chain_executor_contains_relu(self):
        program = parse_text(_PROG_BODY_CHAIN)
        source = DifferentiablePythonCompiler().compile(program)
        self.assertIn("def _layer_exec_Chain(", source)
        self.assertIn("_tensor_relu", source)
        self.assertIn("_tensor_matmul", source)

    def test_body_binds_layer_param(self):
        program = parse_text(_PROG_LAYER_BODY)
        source = DifferentiablePythonCompiler().compile(program)
        self.assertIn("Scale.W", source)

    def test_param_binding_uses_is_not_none_not_or(self):
        # Generated code must not use truthiness fallback — 0.0 would silently
        # drop a zero-initialized scalar param back to the unqualified name.
        program = parse_text(_PROG_LAYER_BODY)
        source = DifferentiablePythonCompiler().compile(program)
        self.assertIn("is not None", source)
        self.assertNotIn("parameters.get('Scale.W') or", source)

    def test_param_binding_falsy_qualified_key_not_replaced_by_fallback(self):
        # Regression: if Scale.W is falsy-but-not-None (e.g. 0.0 for a scalar
        # bias), the executor must NOT silently fall back to the short name.
        # Call _layer_exec_Scale directly (bypasses run() parameter validation)
        # so we can pass an intentionally malformed falsy value.
        program = parse_text(_PROG_LAYER_BODY)
        source = DifferentiablePythonCompiler().compile(program)
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
        identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        exec_fn = ns["_layer_exec_Scale"]
        state = {"X": [1.0, 2.0, 3.0, 4.0]}
        # Scale.W=0.0 (falsy but not None);  W=identity (fallback)
        output = exec_fn("X", state, {"Scale.W": 0.0, "W": identity})
        # With old 'or': 0.0 or identity → uses identity → [1,2,3,4]  (BUG)
        # With 'is not None': 0.0 is not None → uses 0.0 → matmul returns []  (FIX)
        self.assertNotEqual(output, [1.0, 2.0, 3.0, 4.0], "must not fall back to unqualified 'W'")
        self.assertEqual(output, [], "Scale.W=0.0 is not None → executor uses it, matmul(vec, scalar) = []")


# ---------------------------------------------------------------------------
# Execution: compiled module runs layer body, not passthrough
# ---------------------------------------------------------------------------

class TestLayerBodyExecution(unittest.TestCase):

    _FIELDS = ["x1", "x2", "x3", "x4"]

    def _input(self, vec):
        return {"X": {f: v for f, v in zip(self._FIELDS, vec)}}

    def _run(self, prog_text, input_vec):
        program = parse_text(prog_text)
        source = DifferentiablePythonCompiler().compile(program)
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
        identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        result = ns["run"](self._input(input_vec), {"Scale.W": identity})
        return result["state"].get("F")

    def test_matmul_body_returns_transformed_input(self):
        vec = [1.0, 2.0, 3.0, 4.0]
        output = self._run(_PROG_LAYER_BODY, vec)
        # With identity W, matmul(x, I) = x
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 4)
        for got, expected in zip(output, vec):
            self.assertAlmostEqual(got, expected, places=6)

    def test_chain_body_applies_relu_after_matmul(self):
        program = parse_text(_PROG_BODY_CHAIN)
        source = DifferentiablePythonCompiler().compile(program)
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
        # W = -I: matmul(x, -I) = -x; relu(-x) = 0 for positive x
        neg_identity = [[-1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        vec = [1.0, 2.0, 3.0, 4.0]
        result = ns["run"](self._input(vec), {"Chain.W": neg_identity})
        output = result["state"].get("F")
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 4)
        for val in output:
            self.assertAlmostEqual(float(val), 0.0, places=9, msg="relu(matmul(x, -I)) must be zero")

    def test_matmul_with_non_identity_w(self):
        # Verify matmul actually transforms the input, not just passes it through
        program = parse_text(_PROG_LAYER_BODY)
        source = DifferentiablePythonCompiler().compile(program)
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
        # W doubles all values: W[i][i] = 2, off-diagonal = 0
        double_diag = [[2.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        vec = [1.0, 2.0, 3.0, 4.0]
        result = ns["run"](self._input(vec), {"Scale.W": double_diag})
        output = result["state"].get("F")
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 4)
        for got, expected in zip(output, [2.0, 4.0, 6.0, 8.0]):
            self.assertAlmostEqual(got, expected, places=9)


# ---------------------------------------------------------------------------
# match_update_patterns
# ---------------------------------------------------------------------------

class TestMatchUpdatePatterns(unittest.TestCase):

    _KEYS = [
        "W1", "b1",
        "encoder.block_0.attention.Wq",
        "encoder.block_0.attention.Wk",
        "encoder.block_0.feed_forward.W1",
        "classifier.W",
        "classifier.b",
    ]

    def test_wildcard_star_matches_all(self):
        result = match_update_patterns(["*"], self._KEYS)
        self.assertEqual(sorted(result), sorted(self._KEYS))

    def test_exact_path_matches_one(self):
        result = match_update_patterns(["classifier.W"], self._KEYS)
        self.assertEqual(result, ["classifier.W"])

    def test_prefix_wildcard_matches_subtree(self):
        result = match_update_patterns(["encoder.*"], self._KEYS)
        self.assertIn("encoder.block_0.attention.Wq", result)
        self.assertIn("encoder.block_0.attention.Wk", result)
        self.assertIn("encoder.block_0.feed_forward.W1", result)
        self.assertNotIn("classifier.W", result)
        self.assertNotIn("W1", result)

    def test_multiple_patterns_union(self):
        result = match_update_patterns(["classifier.*", "W1"], self._KEYS)
        self.assertIn("W1", result)
        self.assertIn("classifier.W", result)
        self.assertIn("classifier.b", result)
        self.assertNotIn("encoder.block_0.attention.Wq", result)

    def test_no_match_returns_empty(self):
        result = match_update_patterns(["nonexistent.*"], self._KEYS)
        self.assertEqual(result, [])

    def test_exact_encoder_prefix_matches(self):
        # "encoder.*" should match "encoder" exactly AND "encoder.*" prefix
        keys = ["encoder", "encoder.W", "not_encoder"]
        result = match_update_patterns(["encoder.*"], keys)
        self.assertIn("encoder", result)
        self.assertIn("encoder.W", result)
        self.assertNotIn("not_encoder", result)


if __name__ == "__main__":
    unittest.main()
