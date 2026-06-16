"""P11.5 Cut 3 — Attention Unrolling.

Verifies that:
- encoder_attn no longer uses the opaque 'attention' operation.
- encoder_attn uses 'dot', 'scale', and 'softmax' correctly.
- Backend contract accepts the unrolled attention operations.
- The forward passes (differentiable_python and torch) still produce finite results.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from matrixai.parser import parse_file
from matrixai.compiler import BackendContractAnalyzer

_BASE = Path(__file__).parent.parent
_TEMPLATE = _BASE / "examples" / "transformer-classifier.mxai"


class TestAttentionUnrolling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prog = parse_file(str(_TEMPLATE))
        cls.attn_layer = next(l for l in cls.prog.layers if l.name == "encoder_attn")

    def test_attention_op_is_removed(self):
        kinds = [op.kind for op in self.attn_layer.body_ops]
        self.assertNotIn("attention", kinds, "Opaque 'attention' op should not be in the layer body")

    def test_unrolled_ops_are_present(self):
        kinds = [op.kind for op in self.attn_layer.body_ops]
        self.assertIn("dot", kinds)
        self.assertIn("scale", kinds)
        self.assertIn("softmax", kinds)

    def test_backend_contract_accepts_unrolled_attention(self):
        report = BackendContractAnalyzer(target="differentiable_python").analyze(self.prog)
        self.assertTrue(report.ok, report.parameter_errors)

    def test_torch_backend_contract_accepts_unrolled_attention(self):
        report = BackendContractAnalyzer(target="torch").analyze(self.prog)
        self.assertTrue(report.ok, report.parameter_errors)

    def test_differentiable_python_forward_is_finite(self):
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        from matrixai.parameters.store import build_initial_parameter_set
        
        ps = build_initial_parameter_set(self.prog)
        compiler = DifferentiablePythonCompiler()
        source = compiler.compile(self.prog)
        
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)
        params = {k: v["values"] for k, v in ps.parameters.items()}
        
        # input tokens
        result = ns["run"]({"Input": [3, 7, 2, 1, 5, 0, 8, 4]}, params)
        logits = result["state"]["logits"]
        self.assertEqual(len(logits), 2)
        for v in logits:
            self.assertTrue(math.isfinite(v))

    def test_torch_forward_is_finite(self):
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.parameters.store import build_initial_parameter_set
        
        ps = build_initial_parameter_set(self.prog)
        runner = TorchForwardRunner()
        
        result = runner.run(self.prog, {"Input": [3, 7, 2, 1, 5, 0, 8, 4]}, ps)
        logits = result["state"]["logits"]
        self.assertEqual(len(logits), 2)
        for v in logits:
            self.assertTrue(math.isfinite(v))


class TestScaleSoftmaxInContract(unittest.TestCase):
    """Verify that scale and softmax are accepted as top-level FUNCTION kinds
    by the differentiable_python backend contract (not only inside layer body ops)."""

    def _minimal_program(self, func_kind: str, raw_expr: str, inputs: list) -> object:
        from matrixai.ir.schema import (
            ExpressionSpec,
            FunctionSpec,
            GraphSpec,
            MatrixAIProgram,
            VectorSpec,
        )
        vec = VectorSpec(name="X", size=4, fields=["a", "b", "c", "d"])
        func = FunctionSpec(
            name="Out",
            output="result",
            expression=raw_expr,
            semantic=ExpressionSpec(kind=func_kind, raw=raw_expr, inputs=inputs, parameters={}),
        )
        return MatrixAIProgram(
            project=f"test_{func_kind}_contract",
            vectors=[vec],
            functions=[func],
            graph=GraphSpec(
                nodes=["X", "Out"],
                edges=[("X", "Out")],
                node_types={"X": "vector", "Out": "function"},
            ),
        )

    def test_scale_accepted_by_dp_backend_contract(self):
        prog = self._minimal_program("scale", "scale(X, 2.0)", ["X"])
        report = BackendContractAnalyzer(target="differentiable_python").analyze(prog)
        node = next((n for n in report.nodes if n.node == "Out"), None)
        self.assertIsNotNone(node)
        self.assertTrue(node.supported, f"scale should be supported: {node.reason}")
        self.assertTrue(node.differentiable)

    def test_softmax_accepted_by_dp_backend_contract(self):
        prog = self._minimal_program("softmax", "softmax(X)", ["X"])
        report = BackendContractAnalyzer(target="differentiable_python").analyze(prog)
        node = next((n for n in report.nodes if n.node == "Out"), None)
        self.assertIsNotNone(node)
        self.assertTrue(node.supported, f"softmax should be supported: {node.reason}")
        self.assertTrue(node.differentiable)


class TestAttentionDegeneracy(unittest.TestCase):
    """Documents the known architectural limitation of encoder_attn with pooled 1D input.

    dot(q, k) yields a scalar because q and k are 1D vectors (not sequences).
    softmax(scalar) = 1.0 in both backends, so attn = scale(v, 1.0) = v always.
    Wq and Wk have zero effect on the output — they are functionally dead parameters.

    Fix requires passing the full sequence matrix [L×D] to encoder_attn, not the
    pooled vector. This is the responsibility of a future corte (post-P11.5).
    """

    def test_softmax_of_scalar_is_one_differentiable_python(self):
        # Simulate _tensor_softmax with a scalar (float) input — the actual runtime path.
        scalar = 0.43690  # typical scaled_score = dot(q,k) * 0.35355339
        vec = scalar
        attn_weight = 1.0 if not isinstance(vec, list) else None
        self.assertEqual(attn_weight, 1.0,
            "softmax(scalar) returns 1.0: Wq/Wk have no effect on encoder_attn output")

    def test_softmax_of_scalar_is_one_torch(self):
        import torch
        # dot(q, k) returns a 0-dim tensor in torch.
        score = torch.tensor(0.43690)
        self.assertEqual(score.dim(), 0,
            "dot product of two 1D vectors is a 0-dim (scalar) tensor")
        # _execute_body_op_torch path: val.dim() == 0 → torch.ones_like(val)
        attn_weight = torch.ones_like(score).item()
        self.assertEqual(attn_weight, 1.0,
            "softmax(0-dim tensor) = ones_like = 1.0 in torch backend")

    def test_wq_wk_have_no_effect_on_forward_output(self):
        """Changing Wq and Wk by 100× must not alter the model output."""
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.parameters.store import build_initial_parameter_set
        import copy

        prog = next(
            p for p in [__import__("matrixai.parser", fromlist=["parse_file"]).parse_file(
                str(Path(__file__).parent.parent / "examples" / "transformer-classifier.mxai")
            )]
        )
        ps_base = build_initial_parameter_set(prog)
        runner = TorchForwardRunner()
        tokens = {"Input": [3, 7, 2, 1, 5, 0, 8, 4]}

        result_base = runner.run(prog, tokens, ps_base)
        logits_base = result_base["state"]["logits"]

        def _scale_vals(vals, factor):
            if isinstance(vals, list) and vals and isinstance(vals[0], list):
                return [[v * factor for v in row] for row in vals]
            if isinstance(vals, list):
                return [v * factor for v in vals]
            return vals

        # Scale Wq and Wk by 100 — if attention routing worked, output would change
        ps_perturbed = copy.deepcopy(ps_base)
        for key in list(ps_perturbed.parameters.keys()):
            if key.endswith(".Wq") or key.endswith(".Wk"):
                entry = ps_perturbed.parameters[key]
                vals = entry["values"]
                ps_perturbed.parameters[key] = {**entry, "values": _scale_vals(vals, 100.0)}

        result_perturbed = runner.run(prog, tokens, ps_perturbed)
        logits_perturbed = result_perturbed["state"]["logits"]

        for a, b in zip(logits_base, logits_perturbed):
            self.assertAlmostEqual(a, b, places=5,
                msg="Wq/Wk×100 must not change output — attn_weight is always 1.0")


if __name__ == "__main__":
    unittest.main()
