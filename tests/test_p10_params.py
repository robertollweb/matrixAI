"""P10 Cut 3 — Hierarchical ParameterSet with layer-qualified paths."""
from __future__ import annotations

import unittest

from matrixai.compiler import BackendContractAnalyzer
from matrixai.parser import parse_text
from matrixai.parameters.store import (
    build_initial_parameter_set,
    validate_parameter_set,
)


_ATTENTION_PROGRAM = """\
PROJECT AttentionTest
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
LAYER Attention
  PARAM Wq Tensor[4, 4]
  PARAM Wk Tensor[4, 4]
  PARAM Wv Tensor[4, 4]
END
FUNCTION F
  result = call_layer(Attention, Input)
END
GRAPH
  Input -> F
END
"""

_FLAT_PROGRAM = """\
PROJECT FlatTest
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
PARAM W Tensor[3, 2]
  TRAINABLE true
END
PARAM b Tensor[3]
  TRAINABLE true
END
FUNCTION F
  result = softmax(W * Input + b)
END
GRAPH
  Input -> F
END
"""


class HierarchicalManifestTest(unittest.TestCase):
    def _manifest(self, program_text: str) -> list[dict]:
        program = parse_text(program_text)
        return BackendContractAnalyzer().analyze(program).parameter_manifest

    def test_layer_params_appear_in_manifest(self) -> None:
        manifest = self._manifest(_ATTENTION_PROGRAM)
        names = [m["name"] for m in manifest]
        self.assertIn("Wq", names)
        self.assertIn("Wk", names)
        self.assertIn("Wv", names)

    def test_layer_params_have_hierarchical_path(self) -> None:
        manifest = self._manifest(_ATTENTION_PROGRAM)
        paths = {m["name"]: m["path"] for m in manifest}
        self.assertEqual(paths["Wq"], "Attention.Wq")
        self.assertEqual(paths["Wk"], "Attention.Wk")
        self.assertEqual(paths["Wv"], "Attention.Wv")

    def test_flat_params_path_equals_name(self) -> None:
        manifest = self._manifest(_FLAT_PROGRAM)
        for m in manifest:
            self.assertEqual(m["path"], m["name"])

    def test_layer_params_have_function_field(self) -> None:
        manifest = self._manifest(_ATTENTION_PROGRAM)
        for m in manifest:
            self.assertEqual(m["function"], "Attention")

    def test_layer_params_shapes(self) -> None:
        manifest = self._manifest(_ATTENTION_PROGRAM)
        for m in manifest:
            self.assertEqual(m["shape"], [4, 4])


class HierarchicalParameterSetTest(unittest.TestCase):
    def test_layer_params_stored_with_path_key(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        self.assertIn("Attention.Wq", ps.parameters)
        self.assertIn("Attention.Wk", ps.parameters)
        self.assertIn("Attention.Wv", ps.parameters)

    def test_flat_params_stored_with_name_key(self) -> None:
        program = parse_text(_FLAT_PROGRAM)
        ps = build_initial_parameter_set(program)
        self.assertIn("W", ps.parameters)
        self.assertIn("b", ps.parameters)

    def test_layer_param_entry_has_shape(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        self.assertEqual(ps.parameters["Attention.Wq"]["shape"], [4, 4])

    def test_layer_param_entry_has_is_layer_flag(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        self.assertTrue(ps.parameters["Attention.Wq"].get("is_layer"))

    def test_flat_param_no_is_layer_flag(self) -> None:
        program = parse_text(_FLAT_PROGRAM)
        ps = build_initial_parameter_set(program)
        self.assertNotIn("is_layer", ps.parameters.get("W", {}))

    def test_layer_params_have_initial_values(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        values = ps.parameters["Attention.Wq"]["values"]
        self.assertIsNotNone(values)
        self.assertIsInstance(values, list)
        self.assertEqual(len(values), 4)
        self.assertEqual(len(values[0]), 4)


class HierarchicalRuntimeParametersTest(unittest.TestCase):
    def test_layer_path_key_in_runtime_params(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        rt = ps.runtime_parameters()
        self.assertIn("Attention.Wq", rt)
        self.assertIn("Attention.Wk", rt)
        self.assertIn("Attention.Wv", rt)

    def test_flat_params_still_expose_function_dot_name(self) -> None:
        program = parse_text(_FLAT_PROGRAM)
        ps = build_initial_parameter_set(program)
        rt = ps.runtime_parameters()
        self.assertIn("W", rt)
        self.assertIn("b", rt)
        # flat params also exposed as function.name for backward compat
        self.assertIn("F.W", rt)
        self.assertIn("F.b", rt)

    def test_layer_path_not_doubled(self) -> None:
        """Attention.Wq must not appear as Attention.Attention.Wq."""
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        rt = ps.runtime_parameters()
        self.assertNotIn("Attention.Attention.Wq", rt)


class ValidateHierarchicalParameterSetTest(unittest.TestCase):
    def test_valid_layer_parameter_set(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        result = validate_parameter_set(program, ps)
        self.assertTrue(result.ok, result.errors)

    def test_valid_flat_parameter_set_unchanged(self) -> None:
        program = parse_text(_FLAT_PROGRAM)
        ps = build_initial_parameter_set(program)
        result = validate_parameter_set(program, ps)
        self.assertTrue(result.ok, result.errors)

    def test_missing_layer_param_detected(self) -> None:
        from matrixai.parameters.store import ParameterSet
        program = parse_text(_ATTENTION_PROGRAM)
        ps = build_initial_parameter_set(program)
        # Remove one hierarchical param
        trimmed = {k: v for k, v in ps.parameters.items() if k != "Attention.Wq"}
        ps2 = ParameterSet(
            parameter_set_id=ps.parameter_set_id,
            model_hash=ps.model_hash,
            parameter_schema_hash=ps.parameter_schema_hash,
            parameters=trimmed,
        )
        result = validate_parameter_set(program, ps2)
        self.assertFalse(result.ok)
        self.assertTrue(any("Attention.Wq" in e for e in result.errors))


class BackwardCompatFlatParamsTest(unittest.TestCase):
    """Existing flat ParameterSet programs must continue to work identically."""

    def test_flat_ps_roundtrip(self) -> None:
        import json
        program = parse_text(_FLAT_PROGRAM)
        ps = build_initial_parameter_set(program)
        d = ps.to_dict()
        from matrixai.parameters.store import ParameterSet
        ps2 = ParameterSet.from_dict(d)
        result = validate_parameter_set(program, ps2)
        self.assertTrue(result.ok, result.errors)

    def test_flat_param_keys_unchanged(self) -> None:
        program = parse_text(_FLAT_PROGRAM)
        ps = build_initial_parameter_set(program)
        self.assertIn("W", ps.parameters)
        self.assertIn("b", ps.parameters)
        self.assertNotIn("F.W", ps.parameters)


if __name__ == "__main__":
    unittest.main()
