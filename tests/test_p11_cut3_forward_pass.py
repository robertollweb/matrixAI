"""P11 Cut 3 — Audited forward pass for transformer-classifier-vector.mxai.

Tests for:
- Layer boundary shapes: Input=8, AttnBlock=8, FfnBlock=8, Logits=2
- State completeness: all expected state keys present after run()
- Trace: 4 entries, one per graph node, correct types and output_refs
- Value properties: all logits finite, no NaN, deterministic, input-sensitive
- Result metadata: autodiff_plan.ready, 14 trainable params, parameter_manifest
- Torch backend equivalence (skipped when PyTorch not available)
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from matrixai.parser import parse_file
from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
from matrixai.parameters.store import build_initial_parameter_set

_TEMPLATE = Path(__file__).parent.parent / "examples" / "transformer-classifier-vector.mxai"

_INPUT_A = {"Input": {f"x{i}": float(i + 1) * 0.1 for i in range(8)}}
_INPUT_B = {"Input": {f"x{i}": float(8 - i) * 0.1 for i in range(8)}}


def _setup():
    program = parse_file(str(_TEMPLATE))
    source = DifferentiablePythonCompiler().compile(program)
    ns: dict = {}
    exec(compile(source, "<forward_pass>", "exec"), ns)  # noqa: S102
    ps = build_initial_parameter_set(program)
    params = {k: v["values"] for k, v in ps.parameters.items()}
    return ns["run"], params


_run, _params = _setup()


class TestForwardPassShapes(unittest.TestCase):

    def setUp(self):
        self._result = _run(_INPUT_A, _params)
        self._state = self._result["state"]

    def test_input_vector_shape(self):
        self.assertEqual(len(self._state["Input"]), 8)

    def test_attn_block_output_shape(self):
        self.assertEqual(len(self._state["AttnBlock"]), 8)

    def test_ffn_block_output_shape(self):
        self.assertEqual(len(self._state["FfnBlock"]), 8)

    def test_logits_output_shape(self):
        self.assertEqual(len(self._state["Logits"]), 2)

    def test_attn_block_alias_matches(self):
        # state["AttnBlock"] and state["attn_block"] must be equal
        self.assertEqual(self._state["AttnBlock"], self._state["attn_block"])

    def test_ffn_block_alias_matches(self):
        self.assertEqual(self._state["FfnBlock"], self._state["ffn_block"])

    def test_logits_alias_matches(self):
        self.assertEqual(self._state["Logits"], self._state["logits"])


class TestForwardPassState(unittest.TestCase):

    def setUp(self):
        self._state = _run(_INPUT_A, _params)["state"]

    def test_all_graph_nodes_in_state(self):
        for key in ("Input", "AttnBlock", "FfnBlock", "Logits"):
            self.assertIn(key, self._state, f"state missing key: {key}")

    def test_all_output_refs_in_state(self):
        for key in ("attn_block", "ffn_block", "logits"):
            self.assertIn(key, self._state, f"state missing output_ref: {key}")

    def test_scalar_fields_in_state(self):
        for i in range(8):
            self.assertIn(f"x{i}", self._state)

    def test_no_extra_junk_keys(self):
        expected = {
            "Input", "AttnBlock", "FfnBlock", "Logits",
            "attn_block", "ffn_block", "logits",
            *[f"x{i}" for i in range(8)],
        }
        unexpected = set(self._state.keys()) - expected
        self.assertEqual(unexpected, set(), f"Unexpected state keys: {unexpected}")


class TestForwardPassTrace(unittest.TestCase):

    def setUp(self):
        self._trace = _run(_INPUT_A, _params)["trace"]

    def test_trace_has_four_entries(self):
        self.assertEqual(len(self._trace), 4)

    def test_trace_node_order(self):
        nodes = [t["node"] for t in self._trace]
        self.assertEqual(nodes, ["Input", "AttnBlock", "FfnBlock", "Logits"])

    def test_trace_node_types(self):
        type_map = {t["node"]: t["node_type"] for t in self._trace}
        self.assertEqual(type_map["Input"], "vector")
        self.assertEqual(type_map["AttnBlock"], "function")
        self.assertEqual(type_map["FfnBlock"], "function")
        self.assertEqual(type_map["Logits"], "function")

    def test_trace_output_refs(self):
        ref_map = {t["node"]: t.get("output_ref") for t in self._trace}
        self.assertEqual(ref_map["Input"], "Input")
        self.assertEqual(ref_map["AttnBlock"], "attn_block")
        self.assertEqual(ref_map["FfnBlock"], "ffn_block")
        self.assertEqual(ref_map["Logits"], "logits")


class TestForwardPassValues(unittest.TestCase):

    def test_logits_are_finite(self):
        result = _run(_INPUT_A, _params)
        for v in result["state"]["logits"]:
            self.assertTrue(math.isfinite(v), f"logit {v!r} is not finite")

    def test_intermediate_outputs_are_finite(self):
        state = _run(_INPUT_A, _params)["state"]
        for key in ("attn_block", "ffn_block"):
            for v in state[key]:
                self.assertTrue(math.isfinite(v), f"{key}[{v!r}] is not finite")

    def test_forward_pass_is_deterministic(self):
        r1 = _run(_INPUT_A, _params)["state"]["logits"]
        r2 = _run(_INPUT_A, _params)["state"]["logits"]
        self.assertEqual(r1, r2)

    def test_different_input_produces_different_logits(self):
        logits_a = _run(_INPUT_A, _params)["state"]["logits"]
        logits_b = _run(_INPUT_B, _params)["state"]["logits"]
        self.assertNotEqual(logits_a, logits_b)

    def test_different_params_produce_different_logits(self):
        # Perturb one weight and verify output changes
        import copy
        params2 = copy.deepcopy(_params)
        params2["classifier.W"][0][0] += 1.0
        logits1 = _run(_INPUT_A, _params)["state"]["logits"]
        logits2 = _run(_INPUT_A, params2)["state"]["logits"]
        self.assertNotEqual(logits1, logits2)


class TestForwardPassMetadata(unittest.TestCase):

    def setUp(self):
        self._result = _run(_INPUT_A, _params)

    def test_autodiff_plan_ready(self):
        self.assertTrue(self._result["autodiff_plan"]["ready"])

    def test_trainable_parameters_count(self):
        self.assertEqual(len(self._result["trainable_parameters"]), 14)

    def test_parameter_manifest_count(self):
        self.assertEqual(len(self._result["parameter_manifest"]), 14)

    def test_parameter_manifest_has_trainable_field(self):
        for entry in self._result["parameter_manifest"]:
            self.assertIn("trainable", entry, f"'trainable' missing from manifest entry {entry.get('path')}")
            self.assertTrue(entry["trainable"])

    def test_tensor_shapes_includes_input(self):
        shapes = self._result["tensor_shapes"]
        self.assertIn("Input", shapes)
        self.assertEqual(shapes["Input"], [8])


class TestTorchBackendContract(unittest.TestCase):
    """Verify the torch backend contract accepts the transformer template — no PyTorch required."""

    def test_torch_contract_ok(self):
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        program = parse_file(str(_TEMPLATE))
        report = BackendContractAnalyzer(target="torch").analyze(program)
        self.assertTrue(report.ok, f"torch contract not ok: {report.unsupported_nodes}")

    def test_torch_contract_layer_call_nodes_supported(self):
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        program = parse_file(str(_TEMPLATE))
        report = BackendContractAnalyzer(target="torch").analyze(program)
        layer_call_nodes = [n for n in report.nodes if n.kind == "layer_call"]
        self.assertEqual(len(layer_call_nodes), 3)
        for node in layer_call_nodes:
            self.assertTrue(node.supported, f"{node.node} should be supported")
            self.assertTrue(node.differentiable, f"{node.node} should be differentiable")

    def test_differentiable_python_contract_still_ok(self):
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        program = parse_file(str(_TEMPLATE))
        report = BackendContractAnalyzer(target="differentiable_python").analyze(program)
        self.assertTrue(report.ok)


try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch not installed")
class TestForwardPassTorchEquivalence(unittest.TestCase):
    """Verify differentiable_python and torch backends produce equivalent logits."""

    def test_logits_match_within_tolerance(self):
        from matrixai.compiler.torch_forward import TorchForwardRunner

        program = parse_file(str(_TEMPLATE))
        ps = build_initial_parameter_set(program)

        params = {k: v["values"] for k, v in ps.parameters.items()}
        py_logits = _run(_INPUT_A, params)["state"]["logits"]

        runner = TorchForwardRunner()
        torch_result = runner.run(program, _INPUT_A, ps)
        torch_logits = torch_result["state"]["logits"]

        self.assertEqual(len(py_logits), len(torch_logits))
        for a, b in zip(py_logits, torch_logits):
            self.assertAlmostEqual(float(a), float(b), places=4,
                                   msg=f"differentiable_python={a} vs torch={b}")

    def test_torch_forward_state_has_expected_shapes(self):
        from matrixai.compiler.torch_forward import TorchForwardRunner

        program = parse_file(str(_TEMPLATE))
        ps = build_initial_parameter_set(program)
        runner = TorchForwardRunner()
        result = runner.run(program, _INPUT_A, ps)
        state = result["state"]

        self.assertEqual(len(state["attn_block"]), 8)
        self.assertEqual(len(state["ffn_block"]), 8)
        self.assertEqual(len(state["logits"]), 2)

    def test_unknown_body_op_raises_torch_forward_error(self):
        import torch
        from matrixai.compiler.torch_forward import TorchForwardRunner, TorchForwardError
        from matrixai.ir import LayerBodyOp

        runner = TorchForwardRunner()
        unknown_op = LayerBodyOp(output="out", kind="frobnicate", args=("x",))
        local_state = {"x": torch.tensor([1.0, 2.0])}
        with self.assertRaises(TorchForwardError):
            runner._execute_body_op_torch(unknown_op, local_state, torch)

    def test_deferred_body_op_raises_torch_forward_error(self):
        import torch
        from matrixai.compiler.torch_forward import TorchForwardRunner, TorchForwardError
        from matrixai.ir import LayerBodyOp

        runner = TorchForwardRunner()
        # embedding_lookup, mean_pooling, cls_pooling, positional_encoding are now
        # implemented in P11.5 — only truly unknown ops should raise
        unknown_op = LayerBodyOp(output="out", kind="unknown_deferred_op", args=("x",))
        local_state = {"x": torch.tensor([1.0, 2.0])}
        with self.assertRaises(TorchForwardError):
            runner._execute_body_op_torch(unknown_op, local_state, torch)

    def test_attention_with_mask_applies_mask_to_score(self):
        import torch
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.ir import LayerBodyOp

        runner = TorchForwardRunner()
        q = torch.tensor([1.0, 0.0])
        k = torch.tensor([1.0, 0.0])
        v = torch.tensor([2.0, 3.0])
        # no mask
        op_no_mask = LayerBodyOp(output="out", kind="attention", args=("q", "k", "v"))
        local_state = {"q": q, "k": k, "v": v}
        result_no_mask = runner._execute_body_op_torch(op_no_mask, local_state, torch)
        # large negative mask — should suppress attention weight toward 0
        neg_mask = torch.tensor([-100.0, -100.0])
        op_masked = LayerBodyOp(output="out", kind="attention", args=("q", "k", "v", "mask"))
        local_state_masked = {"q": q, "k": k, "v": v, "mask": neg_mask}
        result_masked = runner._execute_body_op_torch(op_masked, local_state_masked, torch)
        # masked weight → sigmoid(score + sum(-100,-100)) ≈ 0 → output near zero
        self.assertLess(result_masked[0].item(), result_no_mask[0].item(),
                        "negative mask should reduce attention output")


if __name__ == "__main__":
    unittest.main()
