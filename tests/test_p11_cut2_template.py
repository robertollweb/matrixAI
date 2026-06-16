"""P11 Cut 2 — Canonical transformer-classifier-vector.mxai template.

Tests for:
- Template parses without errors
- Layer structure: names, parameter counts, body op counts
- All expected parameter paths present in BackendContractAnalyzer report
- BackendContractAnalyzer reports ok=True (no unsupported nodes, no param errors)
- DifferentiablePythonCompiler generates _layer_exec_ functions for all three layers
- build_initial_parameter_set succeeds and returns expected hierarchical keys
- Graph topology is correct: Input -> AttnBlock -> FfnBlock -> Logits
"""
from __future__ import annotations

import unittest
from pathlib import Path

from matrixai.parser import parse_file
from matrixai.compiler.backend_contract import BackendContractAnalyzer
from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
from matrixai.parameters.store import build_initial_parameter_set

_TEMPLATE = Path(__file__).parent.parent / "examples" / "transformer-classifier-vector.mxai"

_EXPECTED_PARAMS = [
    "encoder_attn.Wq",
    "encoder_attn.Wk",
    "encoder_attn.Wv",
    "encoder_attn.Wo",
    "encoder_attn.gain",
    "encoder_attn.bias",
    "encoder_ffn.W1",
    "encoder_ffn.b1",
    "encoder_ffn.W2",
    "encoder_ffn.b2",
    "encoder_ffn.gain",
    "encoder_ffn.bias",
    "classifier.W",
    "classifier.b",
]


def _load():
    return parse_file(str(_TEMPLATE))


class TestTemplateParses(unittest.TestCase):

    def test_template_file_exists(self):
        self.assertTrue(_TEMPLATE.exists(), f"Template not found: {_TEMPLATE}")

    def test_project_name(self):
        program = _load()
        self.assertEqual(program.project, "transformer_classifier")

    def test_three_layers_declared(self):
        program = _load()
        names = [l.name for l in program.layers]
        self.assertIn("encoder_attn", names)
        self.assertIn("encoder_ffn", names)
        self.assertIn("classifier", names)
        self.assertEqual(len(names), 3)

    def test_one_vector_declared(self):
        program = _load()
        self.assertEqual(len(program.vectors), 1)
        self.assertEqual(program.vectors[0].name, "Input")
        self.assertEqual(program.vectors[0].size, 8)

    def test_three_functions_declared(self):
        program = _load()
        names = [f.name for f in program.functions]
        self.assertIn("AttnBlock", names)
        self.assertIn("FfnBlock", names)
        self.assertIn("Logits", names)

    def test_graph_topology(self):
        program = _load()
        nodes = program.graph.nodes
        self.assertEqual(nodes, ["Input", "AttnBlock", "FfnBlock", "Logits"])

    def test_graph_edges(self):
        program = _load()
        edges = program.graph.edges
        self.assertIn(("Input", "AttnBlock"), edges)
        self.assertIn(("AttnBlock", "FfnBlock"), edges)
        self.assertIn(("FfnBlock", "Logits"), edges)


class TestLayerStructure(unittest.TestCase):

    def _layer(self, name: str):
        program = _load()
        return next(l for l in program.layers if l.name == name)

    def test_encoder_attn_param_count(self):
        layer = self._layer("encoder_attn")
        self.assertEqual(len(layer.params), 6)

    def test_encoder_attn_param_names(self):
        layer = self._layer("encoder_attn")
        names = {p.name for p in layer.params}
        self.assertSetEqual(names, {"Wq", "Wk", "Wv", "Wo", "gain", "bias"})

    def test_encoder_attn_body_op_count(self):
        layer = self._layer("encoder_attn")
        self.assertEqual(len(layer.body_ops), 7)

    def test_encoder_attn_body_ops_include_attention(self):
        layer = self._layer("encoder_attn")
        kinds = [op.kind for op in layer.body_ops]
        self.assertIn("attention", kinds)
        self.assertIn("matmul", kinds)
        self.assertIn("residual", kinds)
        self.assertIn("layer_norm", kinds)

    def test_encoder_ffn_param_count(self):
        layer = self._layer("encoder_ffn")
        self.assertEqual(len(layer.params), 6)

    def test_encoder_ffn_param_names(self):
        layer = self._layer("encoder_ffn")
        names = {p.name for p in layer.params}
        self.assertSetEqual(names, {"W1", "b1", "W2", "b2", "gain", "bias"})

    def test_encoder_ffn_body_op_count(self):
        layer = self._layer("encoder_ffn")
        self.assertEqual(len(layer.body_ops), 7)

    def test_encoder_ffn_body_ops_include_gelu(self):
        layer = self._layer("encoder_ffn")
        kinds = [op.kind for op in layer.body_ops]
        self.assertIn("gelu", kinds)
        self.assertIn("matmul", kinds)
        self.assertIn("residual", kinds)
        self.assertIn("layer_norm", kinds)

    def test_classifier_param_count(self):
        layer = self._layer("classifier")
        self.assertEqual(len(layer.params), 2)

    def test_classifier_param_names(self):
        layer = self._layer("classifier")
        names = {p.name for p in layer.params}
        self.assertSetEqual(names, {"W", "b"})

    def test_classifier_body_op_count(self):
        layer = self._layer("classifier")
        self.assertEqual(len(layer.body_ops), 2)


class TestBackendContract(unittest.TestCase):

    def setUp(self):
        program = _load()
        self._report = BackendContractAnalyzer().analyze(program)

    def test_report_ok(self):
        self.assertTrue(self._report.ok)

    def test_no_unsupported_nodes(self):
        self.assertEqual(len(self._report.unsupported_nodes), 0)

    def test_no_parameter_errors(self):
        self.assertEqual(len(self._report.parameter_errors), 0)

    def test_trainable_parameter_count(self):
        self.assertEqual(len(self._report.trainable_parameters), len(_EXPECTED_PARAMS))

    def test_all_expected_paths_present(self):
        paths = {p.path for p in self._report.trainable_parameters}
        for expected in _EXPECTED_PARAMS:
            self.assertIn(expected, paths, f"Missing parameter path: {expected}")

    def test_matrix_shapes(self):
        shape_map = {p.path: p.shape for p in self._report.trainable_parameters}
        self.assertEqual(shape_map["encoder_attn.Wq"], (8, 8))
        self.assertEqual(shape_map["encoder_ffn.W1"], (8, 32))
        self.assertEqual(shape_map["encoder_ffn.W2"], (32, 8))
        self.assertEqual(shape_map["classifier.W"], (8, 2))

    def test_vector_shapes(self):
        shape_map = {p.path: p.shape for p in self._report.trainable_parameters}
        self.assertEqual(shape_map["encoder_attn.gain"], (8,))
        self.assertEqual(shape_map["encoder_attn.bias"], (8,))
        self.assertEqual(shape_map["encoder_ffn.b1"], (32,))
        self.assertEqual(shape_map["encoder_ffn.b2"], (8,))
        self.assertEqual(shape_map["classifier.b"], (2,))

    def test_trainable_field_in_manifest_dict(self):
        for p in self._report.trainable_parameters:
            md = p.to_manifest_dict()
            self.assertIn("trainable", md, f"'trainable' missing from manifest for {p.path}")
            self.assertTrue(md["trainable"], f"trainable should be True for {p.path}")

    def test_bias_params_have_bias_role(self):
        role_map = {p.path: p.role for p in self._report.trainable_parameters}
        for path in ["encoder_attn.bias", "encoder_ffn.b1", "encoder_ffn.b2", "classifier.b"]:
            self.assertEqual(role_map[path], "bias", f"{path} should have role='bias'")

    def test_gain_params_have_gain_role(self):
        role_map = {p.path: p.role for p in self._report.trainable_parameters}
        self.assertEqual(role_map["encoder_attn.gain"], "gain")
        self.assertEqual(role_map["encoder_ffn.gain"], "gain")


class TestCompilation(unittest.TestCase):

    def setUp(self):
        program = _load()
        self._source = DifferentiablePythonCompiler().compile(program)

    def test_source_non_empty(self):
        self.assertGreater(len(self._source), 0)

    def test_layer_exec_encoder_attn_generated(self):
        self.assertIn("def _layer_exec_encoder_attn(", self._source)

    def test_layer_exec_encoder_ffn_generated(self):
        self.assertIn("def _layer_exec_encoder_ffn(", self._source)

    def test_layer_exec_classifier_generated(self):
        self.assertIn("def _layer_exec_classifier(", self._source)

    def test_all_param_keys_in_source(self):
        for path in _EXPECTED_PARAMS:
            self.assertIn(path, self._source, f"Parameter key {path!r} not in compiled source")

    def test_is_not_none_sentinel_used(self):
        self.assertIn("is not None", self._source)

    def test_source_executes_without_error(self):
        ns: dict = {}
        exec(compile(self._source, "<test>", "exec"), ns)  # noqa: S102
        self.assertIn("run", ns)

    def test_forward_pass_returns_logits_length_2_and_finite(self):
        program = _load()
        source = DifferentiablePythonCompiler().compile(program)
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
        params = {k: v["values"] for k, v in build_initial_parameter_set(program).parameters.items()}
        input_data = {"Input": {f"x{i}": float(i + 1) * 0.1 for i in range(8)}}
        result = ns["run"](input_data, params)
        logits = result["state"].get("logits")
        self.assertIsInstance(logits, list, "logits should be a list")
        self.assertEqual(len(logits), 2, "logits should have length 2")
        for v in logits:
            self.assertFalse(v != v, f"logit {v} is NaN")  # NaN != NaN
            self.assertLess(abs(v), 1e9, f"logit {v} is not finite")


class TestParameterSet(unittest.TestCase):

    def setUp(self):
        program = _load()
        self._ps = build_initial_parameter_set(program, parameter_set_id="test_p11_cut2")

    def test_parameter_set_id(self):
        self.assertEqual(self._ps.parameter_set_id, "test_p11_cut2")

    def test_all_expected_keys_present(self):
        for path in _EXPECTED_PARAMS:
            self.assertIn(path, self._ps.parameters, f"Missing key: {path}")

    def test_total_key_count(self):
        self.assertEqual(len(self._ps.parameters), len(_EXPECTED_PARAMS))

    def test_matrix_values_have_correct_shape(self):
        wq = self._ps.parameters["encoder_attn.Wq"]["values"]
        self.assertIsNotNone(wq)
        self.assertEqual(len(wq), 8)
        self.assertEqual(len(wq[0]), 8)

    def test_w1_has_shape_8x32(self):
        w1 = self._ps.parameters["encoder_ffn.W1"]["values"]
        self.assertIsNotNone(w1)
        self.assertEqual(len(w1), 8)
        self.assertEqual(len(w1[0]), 32)

    def test_classifier_w_has_shape_8x2(self):
        cw = self._ps.parameters["classifier.W"]["values"]
        self.assertIsNotNone(cw)
        self.assertEqual(len(cw), 8)
        self.assertEqual(len(cw[0]), 2)

    def test_bias_vectors_are_lists(self):
        bias_paths = ["encoder_attn.bias", "encoder_ffn.b1", "encoder_ffn.b2", "encoder_ffn.bias", "classifier.b"]
        for path in bias_paths:
            v = self._ps.parameters[path]["values"]
            self.assertIsInstance(v, list, f"{path} values should be list")

    def test_bias_values_are_zeros(self):
        bias_paths = ["encoder_attn.bias", "encoder_ffn.b1", "encoder_ffn.b2", "encoder_ffn.bias", "classifier.b"]
        for path in bias_paths:
            v = self._ps.parameters[path]["values"]
            self.assertTrue(all(x == 0.0 for x in v), f"{path} should be initialized to zeros")

    def test_gain_values_are_ones(self):
        for path in ["encoder_attn.gain", "encoder_ffn.gain"]:
            v = self._ps.parameters[path]["values"]
            self.assertTrue(all(x == 1.0 for x in v), f"{path} should be initialized to ones")

    def test_model_hash_non_empty(self):
        self.assertGreater(len(self._ps.model_hash), 0)

    def test_parameter_schema_hash_non_empty(self):
        self.assertGreater(len(self._ps.parameter_schema_hash), 0)


if __name__ == "__main__":
    unittest.main()
