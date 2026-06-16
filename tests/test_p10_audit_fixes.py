"""Tests for the 5 audit fixes applied after P10 initial implementation.

Fix 1a: Tensor shape validation in validate_value_against_type
Fix 1b: _function_output_shape for new primitives
Fix 2:  call_layer validates that the referenced LAYER exists
Fix 3:  layer_manifest appears in BackendContractReport.to_dict()
Fix 4:  Record { field: type } syntax parses correctly
"""
import unittest

from matrixai.types import parse_type_spec, validate_value_against_type, TypeSpec
from matrixai.compiler.backend_contract import BackendContractAnalyzer
from matrixai.parser import parse_text


# ---------------------------------------------------------------------------
# Fix 1a: Tensor shape validation
# ---------------------------------------------------------------------------

class TestTensorShapeValidation(unittest.TestCase):

    def test_correct_shape_passes(self):
        spec = parse_type_spec("Tensor[2, 3]")
        value = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        errors = validate_value_against_type("x", value, spec)
        self.assertEqual(errors, [])

    def test_wrong_shape_reports_error(self):
        spec = parse_type_spec("Tensor[2, 3]")
        value = [[1.0, 2.0], [3.0, 4.0]]  # shape [2,2], declared [2,3]
        errors = validate_value_against_type("x", value, spec)
        self.assertTrue(any("shape" in e for e in errors), errors)

    def test_wrong_rank_reports_error(self):
        spec = parse_type_spec("Tensor[4]")
        value = [1.0, 2.0, 3.0]  # length 3, declared 4
        errors = validate_value_against_type("x", value, spec)
        self.assertTrue(len(errors) > 0, errors)

    def test_tensor_without_shape_accepts_any_list(self):
        spec = parse_type_spec("Tensor")
        value = [[1.0, 2.0], [3.0, 4.0]]
        errors = validate_value_against_type("x", value, spec)
        self.assertEqual(errors, [])

    def test_non_list_rejected(self):
        spec = parse_type_spec("Tensor[3]")
        errors = validate_value_against_type("x", 42.0, spec)
        self.assertTrue(len(errors) > 0)

    def test_3d_tensor_shape(self):
        spec = parse_type_spec("Tensor[2, 2, 2]")
        value = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        errors = validate_value_against_type("x", value, spec)
        self.assertEqual(errors, [])


# ---------------------------------------------------------------------------
# Fix 1b: _function_output_shape for new primitives
# ---------------------------------------------------------------------------

_PROG_DOT = """\
PROJECT dot_test
VECTOR A[4]
  a1 : Scalar
  a2 : Scalar
  a3 : Scalar
  a4 : Scalar
END
FUNCTION D
  result = dot(A, A)
END
GRAPH
  A -> D
END
"""

_PROG_RELU = """\
PROJECT relu_test
VECTOR A[4]
  a1 : Scalar
  a2 : Scalar
  a3 : Scalar
  a4 : Scalar
END
FUNCTION R
  result = relu(A)
END
GRAPH
  A -> R
END
"""


class TestFunctionOutputShapeNewPrimitives(unittest.TestCase):

    def _analyze(self, text):
        program = parse_text(text)
        analyzer = BackendContractAnalyzer()
        return analyzer.analyze(program)

    def test_dot_reported_as_supported(self):
        report = self._analyze(_PROG_DOT)
        node = next((n for n in report.nodes if n.node == "D"), None)
        self.assertIsNotNone(node)
        self.assertTrue(node.supported, node.reason)

    def test_relu_reported_as_supported(self):
        report = self._analyze(_PROG_RELU)
        node = next((n for n in report.nodes if n.node == "R"), None)
        self.assertIsNotNone(node)
        self.assertTrue(node.supported, node.reason)


# ---------------------------------------------------------------------------
# Fix 2: call_layer validates layer existence
# ---------------------------------------------------------------------------

_PROG_MISSING_LAYER = """\
PROJECT missing_layer
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(NotThere, X)
END
GRAPH
  X -> F
END
"""

_PROG_PRESENT_LAYER = """\
PROJECT present_layer
LAYER linear(Tensor[4]) -> Tensor[4]
  PARAM W Tensor[4, 4]
END
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(linear, X)
END
GRAPH
  X -> F
END
"""


class TestCallLayerValidatesExistence(unittest.TestCase):

    def _analyze(self, text):
        program = parse_text(text)
        analyzer = BackendContractAnalyzer()
        return analyzer.analyze(program)

    def test_missing_layer_not_supported(self):
        report = self._analyze(_PROG_MISSING_LAYER)
        node = next((n for n in report.nodes if n.node == "F"), None)
        self.assertIsNotNone(node)
        self.assertFalse(node.supported, "call_layer to undefined LAYER must not be supported")
        self.assertIn("NotThere", node.reason)

    def test_present_layer_supported(self):
        report = self._analyze(_PROG_PRESENT_LAYER)
        node = next((n for n in report.nodes if n.node == "F"), None)
        self.assertIsNotNone(node)
        self.assertTrue(node.supported, node.reason)


# ---------------------------------------------------------------------------
# Fix 3: layer_manifest in to_dict()
# ---------------------------------------------------------------------------

_PROG_WITH_LAYER = """\
PROJECT lm_dict_test
LAYER proj(Tensor[4]) -> Tensor[4]
  PARAM W Tensor[4, 4]
END
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(proj, X)
END
GRAPH
  X -> F
END
"""

_PROG_NO_LAYERS = """\
PROJECT no_layers
VECTOR A[2]
  a1 : Scalar
  a2 : Scalar
END
PARAM W Tensor[3, 2]
  TRAINABLE true
END
PARAM b Tensor[3]
  TRAINABLE true
END
FUNCTION S
  result = softmax(W * A + b)
END
GRAPH
  A -> S
END
"""


class TestLayerManifestInToDict(unittest.TestCase):

    def _analyze(self, text):
        program = parse_text(text)
        analyzer = BackendContractAnalyzer()
        return analyzer.analyze(program)

    def test_layer_manifest_present_in_to_dict(self):
        report = self._analyze(_PROG_WITH_LAYER)
        data = report.to_dict()
        self.assertIn("layer_manifest", data, "layer_manifest must appear in to_dict() when layers exist")

    def test_layer_manifest_has_correct_layer(self):
        report = self._analyze(_PROG_WITH_LAYER)
        data = report.to_dict()
        lm = data["layer_manifest"]
        self.assertTrue(any(entry.get("layer") == "proj" for entry in lm), lm)

    def test_no_layers_no_manifest_key(self):
        report = self._analyze(_PROG_NO_LAYERS)
        data = report.to_dict()
        self.assertNotIn("layer_manifest", data)


# ---------------------------------------------------------------------------
# Fix 4: Record { field: type } syntax
# ---------------------------------------------------------------------------

_PROG_RECORD_PARAM = """\
PROJECT record_param
PARAM schema Record { x: Scalar, y: Scalar }
END
VECTOR A[2]
  a1 : Scalar
  a2 : Scalar
END
PARAM W Tensor[3, 2]
  TRAINABLE true
END
PARAM b Tensor[3]
  TRAINABLE true
END
FUNCTION S
  result = softmax(W * A + b)
END
GRAPH
  A -> S
END
"""


class TestRecordTypeSyntax(unittest.TestCase):

    def test_empty_record(self):
        spec = parse_type_spec("Record {}")
        self.assertEqual(spec.name, "Record")
        self.assertEqual(spec.parameters.get("fields"), {})

    def test_single_scalar_field(self):
        spec = parse_type_spec("Record { age: Integer }")
        fields = spec.parameters["fields"]
        self.assertIn("age", fields)
        self.assertEqual(fields["age"]["name"], "Integer")

    def test_multiple_fields(self):
        spec = parse_type_spec("Record { name: String, score: Probability }")
        fields = spec.parameters["fields"]
        self.assertIn("name", fields)
        self.assertIn("score", fields)
        self.assertEqual(fields["name"]["name"], "String")
        self.assertEqual(fields["score"]["name"], "Probability")

    def test_tensor_field(self):
        spec = parse_type_spec("Record { features: Tensor[4, 8] }")
        fields = spec.parameters["fields"]
        self.assertIn("features", fields)
        tensor_spec = fields["features"]
        self.assertEqual(tensor_spec["name"], "Tensor")
        self.assertEqual(tensor_spec["parameters"]["shape"], [4, 8])

    def test_record_with_tensor_comma_inside_brackets(self):
        # Comma inside Tensor[4, 8] must not split field definitions
        spec = parse_type_spec("Record { a: Tensor[4, 8], b: Integer }")
        fields = spec.parameters["fields"]
        self.assertIn("a", fields)
        self.assertIn("b", fields)
        self.assertEqual(len(fields), 2)

    def test_record_in_param_parse(self):
        program = parse_text(_PROG_RECORD_PARAM)
        param = next((p for p in program.parameters if p.name == "schema"), None)
        self.assertIsNotNone(param)
        self.assertEqual(param.type_spec.name, "Record")
        fields = param.type_spec.parameters.get("fields", {})
        self.assertIn("x", fields)
        self.assertIn("y", fields)


# ---------------------------------------------------------------------------
# Shape compatibility between operands (post-audit improvement)
# ---------------------------------------------------------------------------

_PROG_RESIDUAL_MISMATCH = """\
PROJECT residual_mismatch
VECTOR A[4]
  a1 : Scalar
  a2 : Scalar
  a3 : Scalar
  a4 : Scalar
END
VECTOR B[3]
  b1 : Scalar
  b2 : Scalar
  b3 : Scalar
END
FUNCTION R
  result = residual(A, B)
END
GRAPH
  A -> R
  B -> R
END
"""

_PROG_RESIDUAL_OK = """\
PROJECT residual_ok
VECTOR A[4]
  a1 : Scalar
  a2 : Scalar
  a3 : Scalar
  a4 : Scalar
END
VECTOR B[4]
  b1 : Scalar
  b2 : Scalar
  b3 : Scalar
  b4 : Scalar
END
FUNCTION R
  result = residual(A, B)
END
GRAPH
  A -> R
  B -> R
END
"""

_PROG_DOT_MISMATCH = """\
PROJECT dot_mismatch
VECTOR A[4]
  a1 : Scalar
  a2 : Scalar
  a3 : Scalar
  a4 : Scalar
END
VECTOR B[3]
  b1 : Scalar
  b2 : Scalar
  b3 : Scalar
END
FUNCTION D
  result = dot(A, B)
END
GRAPH
  A -> D
  B -> D
END
"""

_PROG_ATTENTION_MASK = """\
PROJECT attention_with_mask
VECTOR Q[4]
  q1 : Scalar
  q2 : Scalar
  q3 : Scalar
  q4 : Scalar
END
VECTOR K[4]
  k1 : Scalar
  k2 : Scalar
  k3 : Scalar
  k4 : Scalar
END
VECTOR V[4]
  v1 : Scalar
  v2 : Scalar
  v3 : Scalar
  v4 : Scalar
END
VECTOR M[4]
  m1 : Scalar
  m2 : Scalar
  m3 : Scalar
  m4 : Scalar
END
FUNCTION Attn
  result = attention(Q, K, V, M)
END
GRAPH
  Q -> Attn
  K -> Attn
  V -> Attn
  M -> Attn
END
"""

_PROG_ATTENTION_NO_MASK = """\
PROJECT attention_no_mask
VECTOR Q[4]
  q1 : Scalar
  q2 : Scalar
  q3 : Scalar
  q4 : Scalar
END
VECTOR K[4]
  k1 : Scalar
  k2 : Scalar
  k3 : Scalar
  k4 : Scalar
END
VECTOR V[4]
  v1 : Scalar
  v2 : Scalar
  v3 : Scalar
  v4 : Scalar
END
FUNCTION Attn
  result = attention(Q, K, V)
END
GRAPH
  Q -> Attn
  K -> Attn
  V -> Attn
END
"""

_PROG_ATTENTION_DIM_MISMATCH = """\
PROJECT attention_dim_mismatch
VECTOR Q[4]
  q1 : Scalar
  q2 : Scalar
  q3 : Scalar
  q4 : Scalar
END
VECTOR K[3]
  k1 : Scalar
  k2 : Scalar
  k3 : Scalar
END
VECTOR V[4]
  v1 : Scalar
  v2 : Scalar
  v3 : Scalar
  v4 : Scalar
END
FUNCTION Attn
  result = attention(Q, K, V)
END
GRAPH
  Q -> Attn
  K -> Attn
  V -> Attn
END
"""


class TestOperandShapeCompatibility(unittest.TestCase):

    def _analyze(self, text):
        program = parse_text(text)
        analyzer = BackendContractAnalyzer()
        return analyzer.analyze(program)

    def test_residual_shape_mismatch_blocked(self):
        report = self._analyze(_PROG_RESIDUAL_MISMATCH)
        node = next(n for n in report.nodes if n.node == "R")
        self.assertFalse(node.supported, "residual with different-size inputs must be blocked")
        self.assertIn("mismatch", node.reason)

    def test_residual_same_shape_ok(self):
        report = self._analyze(_PROG_RESIDUAL_OK)
        node = next(n for n in report.nodes if n.node == "R")
        self.assertTrue(node.supported, node.reason)

    def test_dot_shape_mismatch_blocked(self):
        report = self._analyze(_PROG_DOT_MISMATCH)
        node = next(n for n in report.nodes if n.node == "D")
        self.assertFalse(node.supported, "dot with different-size inputs must be blocked")
        self.assertIn("mismatch", node.reason)

    def test_attention_with_mask_supported(self):
        report = self._analyze(_PROG_ATTENTION_MASK)
        node = next(n for n in report.nodes if n.node == "Attn")
        self.assertTrue(node.supported, node.reason)

    def test_attention_without_mask_supported(self):
        report = self._analyze(_PROG_ATTENTION_NO_MASK)
        node = next(n for n in report.nodes if n.node == "Attn")
        self.assertTrue(node.supported, node.reason)

    def test_attention_q_k_dim_mismatch_blocked(self):
        report = self._analyze(_PROG_ATTENTION_DIM_MISMATCH)
        node = next(n for n in report.nodes if n.node == "Attn")
        self.assertFalse(node.supported, "attention with Q/K dim mismatch must be blocked")
        self.assertIn("dim", node.reason)


if __name__ == "__main__":
    unittest.main()
