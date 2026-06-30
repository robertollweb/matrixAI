"""P15 Corte 4: ONNX export of P10 primitives (matmul, residual, layer_norm, gelu, attention).

Tests cover:
- Export transformer-classifier-vector.mxai to ONNX
- onnx.checker.check_model passes
- onnxruntime can load and run the model
- Output shape is [N, 2] (raw logits)
- Equivalence vs differentiable_python (atol=1e-5, rtol=1e-4)
- OnnxEquivalenceValidator handles layer_call pipeline
- All layer primitives present: matmul, residual, layer_norm, gelu, attention
- Exported functions list includes AttnBlock, FfnBlock, Logits
- Skipped functions is empty (all layer_call)
- model_hash embedded in ONNX metadata
- Edge bundle for transformer works end-to-end
- Unsupported primitive raises OnnxExportError with primitive name
- layer_call kind now in _SUPPORTED_KINDS
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_TRANSFORMER_MXAI = _BASE / "examples" / "transformer-classifier-vector.mxai"
_EMAIL_MXAI = _BASE / "examples" / "email-agent.mxai"


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    from importlib import util
    return util.find_spec("onnxruntime") is not None


def _make_transformer():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_TRANSFORMER_MXAI)
    return prog, build_initial_parameter_set(prog)


def _write_params(ps, path):
    from matrixai.parameters import write_parameter_set
    write_parameter_set(str(path), ps)


# ---------------------------------------------------------------------------
# Export basics
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestTransformerOnnxExport(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_transformer()
        self.td = tempfile.mkdtemp()
        self.onnx_path = Path(self.td) / "model.onnx"
        from matrixai.export import export_onnx
        self.result = export_onnx(self.prog, self.ps, self.onnx_path)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_onnx_file_exists(self):
        self.assertTrue(self.onnx_path.exists())

    def test_onnx_checker_passes(self):
        import onnx
        model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(model)

    def test_input_name_and_shape(self):
        self.assertEqual(self.result.input_name, "Input")
        self.assertEqual(self.result.input_shape, [-1, 8])

    def test_output_name_and_shape(self):
        self.assertEqual(self.result.output_name, "classifier.result")
        self.assertEqual(self.result.output_shape, [-1, 2])

    def test_opset_version(self):
        self.assertEqual(self.result.opset_version, 17)

    def test_exported_functions(self):
        self.assertEqual(self.result.exported_functions, ["AttnBlock", "FfnBlock", "Logits"])

    def test_skipped_functions_empty(self):
        self.assertEqual(self.result.skipped_functions, [])

    def test_model_hash_in_result(self):
        self.assertEqual(self.result.model_hash, self.ps.model_hash)

    def test_parameter_schema_hash_in_result(self):
        self.assertEqual(self.result.parameter_schema_hash, self.ps.parameter_schema_hash)

    def test_onnx_metadata_model_hash(self):
        import onnx
        model = onnx.load(str(self.onnx_path))
        props = {p.key: p.value for p in model.metadata_props}
        self.assertEqual(props.get("matrixai_model_hash"), self.ps.model_hash)

    def test_onnx_metadata_kind(self):
        import onnx
        model = onnx.load(str(self.onnx_path))
        props = {p.key: p.value for p in model.metadata_props}
        self.assertEqual(props.get("matrixai_kind"), "layer_call")

    def test_to_dict_keys(self):
        d = self.result.to_dict()
        expected = {
            "output_path", "opset_version", "model_hash", "parameter_set_id",
            "parameter_schema_hash", "input_name", "input_shape", "output_name",
            "output_shape", "exported_functions", "skipped_functions", "labels",
        }
        self.assertEqual(set(d.keys()), expected)


# ---------------------------------------------------------------------------
# Inference via onnxruntime
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestTransformerOnnxInference(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.prog, self.ps = _make_transformer()
        self.td = tempfile.mkdtemp()
        self.onnx_path = Path(self.td) / "model.onnx"
        from matrixai.export import export_onnx
        export_onnx(self.prog, self.ps, self.onnx_path)
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(self.onnx_path))
        self.np = np

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_batch_size_1(self):
        x = self.np.random.randn(1, 8).astype(self.np.float32)
        out = self.sess.run(None, {"Input": x})[0]
        self.assertEqual(out.shape, (1, 2))

    def test_batch_size_4(self):
        x = self.np.random.randn(4, 8).astype(self.np.float32)
        out = self.sess.run(None, {"Input": x})[0]
        self.assertEqual(out.shape, (4, 2))

    def test_output_is_float32(self):
        x = self.np.random.randn(1, 8).astype(self.np.float32)
        out = self.sess.run(None, {"Input": x})[0]
        self.assertEqual(out.dtype, self.np.float32)

    def test_deterministic_output(self):
        x = self.np.ones((1, 8), dtype=self.np.float32)
        out1 = self.sess.run(None, {"Input": x})[0]
        out2 = self.sess.run(None, {"Input": x})[0]
        self.np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Equivalence vs differentiable_python
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestTransformerEquivalence(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.prog, self.ps = _make_transformer()
        self.td = tempfile.mkdtemp()
        self.onnx_path = Path(self.td) / "model.onnx"
        from matrixai.export import export_onnx
        export_onnx(self.prog, self.ps, self.onnx_path)
        self.np = np

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_equivalence_validator_passes(self):
        from matrixai.export import validate_onnx_equivalence
        eq = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, n_samples=20)
        self.assertTrue(eq.passed, f"Equivalence failed: max_abs_diff={eq.max_abs_diff:.2e}")

    def test_max_abs_diff_within_atol(self):
        from matrixai.export import validate_onnx_equivalence
        eq = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, n_samples=20)
        self.assertLessEqual(eq.max_abs_diff, 1e-5)

    def test_n_outputs_per_sample(self):
        from matrixai.export import validate_onnx_equivalence
        eq = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, n_samples=5)
        self.assertEqual(eq.n_outputs_per_sample, 2)

    def test_manual_sample_equivalence(self):
        from matrixai.compiler import DifferentiablePythonCompiler
        import onnxruntime as ort

        src = DifferentiablePythonCompiler().compile(self.prog)
        ns: dict = {}
        exec(src, ns)
        sess = ort.InferenceSession(str(self.onnx_path))

        rng = self.np.random.default_rng(0)
        params = self.ps.runtime_parameters()
        vector = self.prog.vectors[0]
        max_diff = 0.0

        for _ in range(10):
            row = rng.random((vector.size,), dtype=self.np.float64).astype(self.np.float32)
            inp = {vector.name: {f: float(v) for f, v in zip(vector.fields, row)}}
            dp_out = ns["run"](inp, params)["state"]["Logits"]
            ort_out = sess.run(None, {"Input": row.reshape(1, -1)})[0].reshape(-1)
            for a, b in zip(dp_out, ort_out):
                max_diff = max(max_diff, abs(float(a) - float(b)))

        self.assertLessEqual(max_diff, 1e-5)


# ---------------------------------------------------------------------------
# OnnxEquivalenceValidator handles layer_call
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestEquivalenceValidatorLayerCall(unittest.TestCase):
    def test_validate_returns_result(self):
        from matrixai.export import OnnxEquivalenceValidator, export_onnx
        prog, ps = _make_transformer()
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            export_onnx(prog, ps, onnx_path)
            eq = OnnxEquivalenceValidator().validate(prog, ps, onnx_path, n_samples=5)
            self.assertTrue(eq.passed)
            self.assertIsInstance(eq.max_abs_diff, float)


# ---------------------------------------------------------------------------
# Primitive coverage: verify graph nodes contain expected ONNX ops
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestPrimitiveCoverage(unittest.TestCase):
    def _load_op_types(self):
        import onnx
        prog, ps = _make_transformer()
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            from matrixai.export import export_onnx
            export_onnx(prog, ps, onnx_path)
            model = onnx.load(str(onnx_path))
            return {node.op_type for node in model.graph.node}

    def test_matmul_present(self):
        self.assertIn("MatMul", self._load_op_types())

    def test_add_present(self):
        self.assertIn("Add", self._load_op_types())

    def test_layer_norm_present(self):
        self.assertIn("LayerNormalization", self._load_op_types())

    def test_tanh_present(self):
        self.assertIn("Tanh", self._load_op_types())

    def test_sigmoid_present(self):
        self.assertIn("Sigmoid", self._load_op_types())

    def test_reduce_sum_present(self):
        self.assertIn("ReduceSum", self._load_op_types())


# ---------------------------------------------------------------------------
# layer_call in _SUPPORTED_KINDS
# ---------------------------------------------------------------------------

class TestSupportedKinds(unittest.TestCase):
    def test_layer_call_in_supported_kinds(self):
        from matrixai.export.onnx_exporter import _SUPPORTED_KINDS
        self.assertIn("layer_call", _SUPPORTED_KINDS)

    def test_softmax_linear_still_supported(self):
        from matrixai.export.onnx_exporter import _SUPPORTED_KINDS
        self.assertIn("softmax_linear", _SUPPORTED_KINDS)

    def test_sigmoid_linear_still_supported(self):
        from matrixai.export.onnx_exporter import _SUPPORTED_KINDS
        self.assertIn("sigmoid_linear", _SUPPORTED_KINDS)


# ---------------------------------------------------------------------------
# Unsupported primitive raises explicit error
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestUnsupportedPrimitive(unittest.TestCase):
    def test_unsupported_op_kind_raises_with_name(self):
        """A layer with an unrecognized primitive should raise OnnxExportError naming it."""
        from matrixai.export import OnnxExportError
        from matrixai.export.onnx_exporter import _build_layer_nodes
        import importlib, numpy as np

        onnx = importlib.import_module("onnx")
        numpy_helper = importlib.import_module("onnx.numpy_helper")
        helper = importlib.import_module("onnx.helper")
        TensorProto = onnx.TensorProto

        # Build a mock layer with an unsupported op kind
        class MockOp:
            output = "result"
            kind = "unsupported_op_xyz"
            args = ("input",)

        class MockParam:
            name = "W"
            type_spec = None

        class MockLayer:
            name = "mock_layer"
            params = []
            body_ops = [MockOp()]

        prog, ps = _make_transformer()
        with self.assertRaises(OnnxExportError) as ctx:
            _build_layer_nodes(
                MockLayer(), "mock_layer", "Input", [-1, 8],
                ps, np, numpy_helper, helper, TensorProto,
            )
        self.assertIn("unsupported_op_xyz", str(ctx.exception))

    def test_hash_mismatch_raises(self):
        from matrixai.export import OnnxExportError, export_onnx
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set

        prog_a = parse_file(_TRANSFORMER_MXAI)
        prog_b = parse_file(_EMAIL_MXAI)
        ps_b = build_initial_parameter_set(prog_b)

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(OnnxExportError) as ctx:
                export_onnx(prog_a, ps_b, Path(td) / "model.onnx")
            self.assertIn("model_hash", str(ctx.exception))


# ---------------------------------------------------------------------------
# Edge bundle for transformer
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestTransformerEdgeBundle(unittest.TestCase):
    def test_bundle_created_with_expected_files(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            bundle_dir = Path(td) / "bundle"
            result = create_edge_bundle(
                prog, ps,
                mxai_path=str(_TRANSFORMER_MXAI),
                params_path=str(params_path),
                outdir=str(bundle_dir),
                validate=True,
            )
            # This transformer is a 2-class softmax with no declared labels, so the
            # inference_spec is (correctly) skipped: base P15 bundle, 6 files.
            expected = {"README.md", "export_manifest.json", "model.mxai",
                        "model.onnx", "model_manifest.json", "params.best.json"}
            self.assertEqual(set(result.files), expected)
            self.assertIsNotNone(result.inference_spec_skipped_reason)

    def test_bundle_equivalence_passed(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            result = create_edge_bundle(
                prog, ps,
                mxai_path=str(_TRANSFORMER_MXAI),
                params_path=str(params_path),
                outdir=str(Path(td) / "bundle"),
                validate=True,
            )
            self.assertTrue(result.equivalence_passed)

    def test_bundle_model_manifest_project(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            bundle_dir = Path(td) / "bundle"
            create_edge_bundle(
                prog, ps,
                mxai_path=str(_TRANSFORMER_MXAI),
                params_path=str(params_path),
                outdir=str(bundle_dir),
                validate=True,
            )
            manifest = json.loads((bundle_dir / "model_manifest.json").read_text())
            self.assertEqual(manifest["project"], "transformer_classifier")

    def test_bundle_export_manifest_format(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            bundle_dir = Path(td) / "bundle"
            create_edge_bundle(
                prog, ps,
                mxai_path=str(_TRANSFORMER_MXAI),
                params_path=str(params_path),
                outdir=str(bundle_dir),
                validate=True,
            )
            em = json.loads((bundle_dir / "export_manifest.json").read_text())
            self.assertEqual(em["format"], "onnx")
            self.assertEqual(em["format_version"], 17)
            self.assertIsNotNone(em["equivalence_check"])
            self.assertTrue(em["equivalence_check"]["passed"])


# ---------------------------------------------------------------------------
# Existing softmax/sigmoid models unaffected
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestExistingModelsUnaffected(unittest.TestCase):
    def test_email_agent_still_exports(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.export import export_onnx, validate_onnx_equivalence
        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            result = export_onnx(prog, ps, onnx_path)
            self.assertEqual(result.exported_functions, ["Classifier"])
            eq = validate_onnx_equivalence(prog, ps, onnx_path, n_samples=5)
            self.assertTrue(eq.passed)

    def test_fall_risk_still_exports(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.export import export_onnx, validate_onnx_equivalence
        prog = parse_file(_BASE / "examples" / "fall-risk.mxai")
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            result = export_onnx(prog, ps, onnx_path)
            eq = validate_onnx_equivalence(prog, ps, onnx_path, n_samples=5)
            self.assertTrue(eq.passed)


if __name__ == "__main__":
    unittest.main()
