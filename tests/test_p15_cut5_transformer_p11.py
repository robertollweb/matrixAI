"""P15 Corte 5: ONNX export of P11 transformer with SEQUENCE input.

Tests cover:
- Export transformer-classifier.mxai (SEQUENCE input, int64) to ONNX
- onnx.checker.check_model passes
- ONNX input type is INT64 (not FLOAT)
- onnxruntime inference: output shape [N, 2]
- Equivalence vs differentiable_python (atol=1e-5, rtol=1e-4)
- New primitives present: Gather (embedding_lookup), ReduceMean (mean_pooling),
  ReduceSum (dot), Mul (scale), Softmax (softmax)
- exported_functions includes Embed, AttnBlock, FfnBlock, Logits
- OnnxEquivalenceValidator handles SEQUENCE model
- Edge bundle for SEQUENCE transformer
- Existing VECTOR models (email-agent, fall-risk, transformer-classifier-vector) unaffected
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_TRANSFORMER_SEQ_MXAI = _BASE / "examples" / "transformer-classifier.mxai"
_TRANSFORMER_VEC_MXAI = _BASE / "examples" / "transformer-classifier-vector.mxai"
_EMAIL_MXAI = _BASE / "examples" / "email-agent.mxai"


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    from importlib import util
    return util.find_spec("onnxruntime") is not None


def _make_seq_transformer():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_TRANSFORMER_SEQ_MXAI)
    return prog, build_initial_parameter_set(prog)


def _write_params(ps, path):
    from matrixai.parameters import write_parameter_set
    write_parameter_set(str(path), ps)


# ---------------------------------------------------------------------------
# Export basics for SEQUENCE transformer
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestSequenceTransformerExport(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_seq_transformer()
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

    def test_input_is_int64(self):
        import onnx
        model = onnx.load(str(self.onnx_path))
        inp = model.graph.input[0]
        self.assertEqual(inp.type.tensor_type.elem_type, onnx.TensorProto.INT64)

    def test_input_name_and_shape(self):
        self.assertEqual(self.result.input_name, "Input")
        self.assertEqual(self.result.input_shape, [-1, 8])

    def test_output_name_and_shape(self):
        self.assertEqual(self.result.output_name, "classifier.result")
        self.assertEqual(self.result.output_shape, [-1, 2])

    def test_exported_functions(self):
        self.assertEqual(self.result.exported_functions, ["Embed", "AttnBlock", "FfnBlock", "Logits"])

    def test_skipped_functions_empty(self):
        self.assertEqual(self.result.skipped_functions, [])

    def test_opset_version(self):
        self.assertEqual(self.result.opset_version, 17)

    def test_model_hash_in_result(self):
        self.assertEqual(self.result.model_hash, self.ps.model_hash)

    def test_onnx_metadata_kind(self):
        import onnx
        model = onnx.load(str(self.onnx_path))
        props = {p.key: p.value for p in model.metadata_props}
        self.assertEqual(props.get("matrixai_kind"), "layer_call")


# ---------------------------------------------------------------------------
# ONNX op type coverage
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestSequencePrimitiveCoverage(unittest.TestCase):
    def _load_op_types(self):
        import onnx
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            from matrixai.export import export_onnx
            export_onnx(prog, ps, onnx_path)
            model = onnx.load(str(onnx_path))
            return {node.op_type for node in model.graph.node}

    def test_gather_present(self):
        self.assertIn("Gather", self._load_op_types())

    def test_reduce_mean_present(self):
        self.assertIn("ReduceMean", self._load_op_types())

    def test_reduce_sum_present(self):
        self.assertIn("ReduceSum", self._load_op_types())

    def test_softmax_present(self):
        self.assertIn("Softmax", self._load_op_types())

    def test_layer_normalization_present(self):
        self.assertIn("LayerNormalization", self._load_op_types())

    def test_matmul_present(self):
        self.assertIn("MatMul", self._load_op_types())


# ---------------------------------------------------------------------------
# Inference via onnxruntime
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestSequenceTransformerInference(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.prog, self.ps = _make_seq_transformer()
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
        ids = self.np.array([[10, 5, 23, 7, 15, 3, 28, 1]], dtype=self.np.int64)
        out = self.sess.run(None, {"Input": ids})[0]
        self.assertEqual(out.shape, (1, 2))

    def test_batch_size_3(self):
        ids = self.np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                              [8, 7, 6, 5, 4, 3, 2, 1],
                              [0, 31, 15, 16, 7, 24, 11, 3]], dtype=self.np.int64)
        out = self.sess.run(None, {"Input": ids})[0]
        self.assertEqual(out.shape, (3, 2))

    def test_output_is_float32(self):
        ids = self.np.array([[0] * 8], dtype=self.np.int64)
        out = self.sess.run(None, {"Input": ids})[0]
        self.assertEqual(out.dtype, self.np.float32)

    def test_ids_in_range_respected(self):
        ids_low = self.np.array([[0] * 8], dtype=self.np.int64)
        ids_high = self.np.array([[31] * 8], dtype=self.np.int64)
        out_low = self.sess.run(None, {"Input": ids_low})[0]
        out_high = self.sess.run(None, {"Input": ids_high})[0]
        self.assertEqual(out_low.shape, (1, 2))
        self.assertEqual(out_high.shape, (1, 2))

    def test_deterministic_output(self):
        ids = self.np.array([[5, 10, 15, 20, 25, 30, 0, 1]], dtype=self.np.int64)
        out1 = self.sess.run(None, {"Input": ids})[0]
        out2 = self.sess.run(None, {"Input": ids})[0]
        self.np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Equivalence vs differentiable_python
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestSequenceTransformerEquivalence(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.prog, self.ps = _make_seq_transformer()
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
        self.assertTrue(eq.passed, f"max_abs_diff={eq.max_abs_diff:.2e}")

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

        rng = self.np.random.default_rng(7)
        params = self.ps.runtime_parameters()
        seq = self.prog.sequences[0]
        max_diff = 0.0

        for _ in range(10):
            ids = rng.integers(0, seq.vocab_size, size=(seq.length,))
            dp_out = ns["run"]({seq.name: ids.tolist()}, params)["state"]["Logits"]
            ort_out = sess.run(None, {"Input": ids.reshape(1, -1).astype(self.np.int64)})[0].reshape(-1)
            for a, b in zip(dp_out, ort_out):
                max_diff = max(max_diff, abs(float(a) - float(b)))

        self.assertLessEqual(max_diff, 1e-5)


# ---------------------------------------------------------------------------
# OnnxEquivalenceValidator handles SEQUENCE model
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestEquivalenceValidatorSequence(unittest.TestCase):
    def test_validate_returns_passing_result(self):
        from matrixai.export import OnnxEquivalenceValidator, export_onnx
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            export_onnx(prog, ps, onnx_path)
            eq = OnnxEquivalenceValidator().validate(prog, ps, onnx_path, n_samples=5)
            self.assertTrue(eq.passed)
            self.assertIsInstance(eq.max_abs_diff, float)
            self.assertEqual(eq.n_outputs_per_sample, 2)


# ---------------------------------------------------------------------------
# Edge bundle for SEQUENCE transformer
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestSequenceTransformerEdgeBundle(unittest.TestCase):
    def test_bundle_created(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            bundle_dir = Path(td) / "bundle"
            result = create_edge_bundle(
                prog, ps,
                mxai_path=str(_TRANSFORMER_SEQ_MXAI),
                params_path=str(params_path),
                outdir=str(bundle_dir),
                validate=True,
            )
            expected = {"README.md", "export_manifest.json", "model.mxai",
                        "model.onnx", "model_manifest.json", "params.best.json"}
            self.assertEqual(set(result.files), expected)
            self.assertTrue(result.equivalence_passed)

    def test_bundle_input_shape_in_manifest(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            bundle_dir = Path(td) / "bundle"
            create_edge_bundle(
                prog, ps,
                mxai_path=str(_TRANSFORMER_SEQ_MXAI),
                params_path=str(params_path),
                outdir=str(bundle_dir),
                validate=True,
            )
            em = json.loads((bundle_dir / "export_manifest.json").read_text())
            self.assertEqual(em["input_shape"], [-1, 8])
            self.assertEqual(em["output_shape"], [-1, 2])
            self.assertTrue(em["equivalence_check"]["passed"])


# ---------------------------------------------------------------------------
# Existing VECTOR models unaffected by Corte 5 changes
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestVectorModelsUnaffected(unittest.TestCase):
    def test_transformer_vector_still_works(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.export import export_onnx, validate_onnx_equivalence
        prog = parse_file(_TRANSFORMER_VEC_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            result = export_onnx(prog, ps, onnx_path)
            self.assertEqual(result.input_shape, [-1, 8])
            eq = validate_onnx_equivalence(prog, ps, onnx_path, n_samples=5)
            self.assertTrue(eq.passed)

    def test_email_agent_still_works(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.export import export_onnx, validate_onnx_equivalence
        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            export_onnx(prog, ps, onnx_path)
            eq = validate_onnx_equivalence(prog, ps, onnx_path, n_samples=5)
            self.assertTrue(eq.passed)


if __name__ == "__main__":
    unittest.main()
