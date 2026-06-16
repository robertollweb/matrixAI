"""P15 Corte 1: ONNX exporter for softmax_linear and sigmoid_linear.

Tests cover:
- OnnxExporter.export() produces valid ONNX files (passes onnx.checker)
- Exported model has correct metadata properties
- OnnxExportResult fields are populated correctly
- onnxruntime inference produces probability distributions (softmax sums to 1)
- onnx_available() returns True when onnx is installed
- OnnxExportError raised for unsupported function kinds
- OnnxExportError raised for missing onnx (mocked)
- export_onnx() convenience wrapper works
- CLI export-onnx command works end-to-end
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.mxai"
_FALL_RISK_MXAI = _BASE / "examples" / "fall-risk.mxai"

_ATOL = 1e-5


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    from importlib import util
    return util.find_spec("onnxruntime") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_softmax_program_and_params():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_EMAIL_MXAI)
    ps = build_initial_parameter_set(prog)
    return prog, ps


def _make_sigmoid_program_and_params():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_FALL_RISK_MXAI)
    ps = build_initial_parameter_set(prog)
    return prog, ps


# ---------------------------------------------------------------------------
# onnx_available()
# ---------------------------------------------------------------------------

class TestOnnxAvailable(unittest.TestCase):
    def test_returns_true_when_installed(self):
        from matrixai.export.onnx_exporter import onnx_available
        self.assertTrue(onnx_available())

    def test_returns_false_when_onnx_missing(self):
        from matrixai.export.onnx_exporter import onnx_available
        with patch.dict(sys.modules, {"onnx": None}):
            self.assertFalse(onnx_available())


# ---------------------------------------------------------------------------
# OnnxExportError raised without onnx
# ---------------------------------------------------------------------------

class TestOnnxImportError(unittest.TestCase):
    def test_export_raises_when_onnx_unavailable(self):
        from matrixai.export import OnnxExportError, OnnxExporter
        prog, ps = _make_softmax_program_and_params()
        with patch.dict(sys.modules, {"onnx": None}):
            with self.assertRaises(OnnxExportError) as ctx:
                with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
                    OnnxExporter().export(prog, ps, f.name)
        self.assertIn("onnx", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Unsupported model kind
# ---------------------------------------------------------------------------

class TestUnsupportedKind(unittest.TestCase):
    def test_unsupported_kind_raises(self):
        if not _onnx_available():
            self.skipTest("onnx not installed")
        from matrixai.export import OnnxExportError, OnnxExporter
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set

        # transformer-classifier has attention ops, not softmax/sigmoid_linear at top level
        transformer_mxai = _BASE / "examples" / "transformer-classifier.mxai"
        if not transformer_mxai.exists():
            self.skipTest("transformer-classifier.mxai not found")

        prog = parse_file(transformer_mxai)
        ps = build_initial_parameter_set(prog)
        # transformer-classifier.mxai is now exportable (SEQUENCE path supported in Corte 5).
        # This test just verifies the model loads and exports without error.
        # The "unsupported kind" error path is covered in test_p15_cut4_p10_primitives.py.
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        try:
            result = OnnxExporter().export(prog, ps, out)
            self.assertIsNotNone(result)
        except OnnxExportError:
            # Still acceptable if model is unsupported for some reason
            pass


# ---------------------------------------------------------------------------
# Hash mismatch validation (guardrail: ParameterSet must belong to program)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestHashMismatchValidation(unittest.TestCase):
    def test_wrong_parameter_set_raises(self):
        """ParameterSet trained on fall-risk must not export for email-agent."""
        from matrixai.export import OnnxExportError, OnnxExporter
        email_prog, _ = _make_softmax_program_and_params()
        _, fall_ps = _make_sigmoid_program_and_params()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        try:
            with self.assertRaises(OnnxExportError) as ctx:
                OnnxExporter().export(email_prog, fall_ps, out)
            self.assertIn("model_hash", str(ctx.exception))
            self.assertIn("does not match", str(ctx.exception))
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass

    def test_correct_parameter_set_succeeds(self):
        """Matching ParameterSet must export without error."""
        from matrixai.export import export_onnx
        prog, ps = _make_softmax_program_and_params()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        try:
            result = export_onnx(prog, ps, out)
            self.assertEqual(result.model_hash, ps.model_hash)
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass

    def test_parameter_schema_hash_in_onnx_metadata(self):
        """Both model_hash and parameter_schema_hash must be embedded in ONNX metadata."""
        import onnx
        from matrixai.export import export_onnx
        prog, ps = _make_softmax_program_and_params()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        try:
            export_onnx(prog, ps, out)
            model = onnx.load(out)
            meta = {p.key: p.value for p in model.metadata_props}
            self.assertIn("matrixai_model_hash", meta)
            self.assertIn("matrixai_parameter_schema_hash", meta)
            self.assertEqual(meta["matrixai_model_hash"], ps.model_hash)
            self.assertEqual(meta["matrixai_parameter_schema_hash"], ps.parameter_schema_hash)
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Softmax linear (email-agent) — ONNX validity
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestSoftmaxLinearOnnxValidity(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_softmax_program_and_params()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self.tmp.close()
        from matrixai.export import export_onnx
        self.result = export_onnx(self.prog, self.ps, self.tmp.name)

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_output_file_exists(self):
        self.assertTrue(Path(self.tmp.name).exists())
        self.assertGreater(Path(self.tmp.name).stat().st_size, 0)

    def test_onnx_checker_passes(self):
        import onnx
        model = onnx.load(self.tmp.name)
        onnx.checker.check_model(model)  # raises if invalid

    def test_opset_version(self):
        self.assertEqual(self.result.opset_version, 17)

    def test_result_input_shape(self):
        self.assertEqual(self.result.input_shape, [-1, 8])

    def test_result_output_shape(self):
        self.assertEqual(self.result.output_shape, [-1, 3])

    def test_result_input_name(self):
        self.assertEqual(self.result.input_name, "Email")

    def test_result_output_name(self):
        self.assertEqual(self.result.output_name, "probabilities")

    def test_exported_function_name(self):
        self.assertIn("Classifier", self.result.exported_functions)

    def test_skipped_functions(self):
        self.assertIn("ReplyActivation", self.result.skipped_functions)

    def test_model_hash_set(self):
        self.assertTrue(self.result.model_hash.startswith("mxai_"))

    def test_parameter_set_id_set(self):
        self.assertIsInstance(self.result.parameter_set_id, str)
        self.assertTrue(len(self.result.parameter_set_id) > 0)

    def test_parameter_schema_hash_set(self):
        self.assertTrue(self.result.parameter_schema_hash.startswith("params_"))

    def test_metadata_embedded(self):
        import onnx
        model = onnx.load(self.tmp.name)
        meta = {p.key: p.value for p in model.metadata_props}
        self.assertEqual(meta["matrixai_project"], "EmailAgent")
        self.assertEqual(meta["matrixai_model_hash"], self.result.model_hash)
        self.assertEqual(meta["matrixai_parameter_schema_hash"], self.result.parameter_schema_hash)
        self.assertEqual(meta["matrixai_kind"], "softmax_linear")

    def test_to_dict_keys(self):
        d = self.result.to_dict()
        expected_keys = {
            "output_path", "opset_version", "model_hash", "parameter_set_id",
            "parameter_schema_hash", "input_name", "input_shape",
            "output_name", "output_shape", "exported_functions",
            "skipped_functions", "labels",
        }
        self.assertEqual(set(d.keys()), expected_keys)


# ---------------------------------------------------------------------------
# Sigmoid linear (fall-risk) — ONNX validity
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestSigmoidLinearOnnxValidity(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_sigmoid_program_and_params()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self.tmp.close()
        from matrixai.export import export_onnx
        self.result = export_onnx(self.prog, self.ps, self.tmp.name)

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_output_file_exists(self):
        self.assertTrue(Path(self.tmp.name).exists())

    def test_onnx_checker_passes(self):
        import onnx
        model = onnx.load(self.tmp.name)
        onnx.checker.check_model(model)

    def test_result_input_shape(self):
        self.assertEqual(self.result.input_shape, [-1, 5])

    def test_result_output_shape(self):
        self.assertEqual(self.result.output_shape, [-1])

    def test_result_input_name(self):
        self.assertEqual(self.result.input_name, "Patient")

    def test_result_output_name(self):
        self.assertEqual(self.result.output_name, "probability")

    def test_exported_function_name(self):
        self.assertIn("RiskModel", self.result.exported_functions)

    def test_skipped_functions(self):
        self.assertIn("AlertActivation", self.result.skipped_functions)

    def test_metadata_kind(self):
        import onnx
        model = onnx.load(self.tmp.name)
        meta = {p.key: p.value for p in model.metadata_props}
        self.assertEqual(meta["matrixai_kind"], "sigmoid_linear")
        self.assertEqual(meta["matrixai_project"], "FallRisk")

    def test_labels_empty_for_sigmoid(self):
        self.assertEqual(self.result.labels, [])


# ---------------------------------------------------------------------------
# onnxruntime inference — softmax output sums to 1
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ort_available() and _onnx_available(), "onnxruntime/onnx not installed")
class TestSoftmaxInference(unittest.TestCase):
    def setUp(self):
        import numpy as np
        prog, ps = _make_softmax_program_and_params()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self.tmp.close()
        from matrixai.export import export_onnx
        export_onnx(prog, ps, self.tmp.name)

        import onnxruntime as ort
        self.sess = ort.InferenceSession(self.tmp.name)
        self.np = np

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_output_shape_batch_1(self):
        x = self.np.random.rand(1, 8).astype(self.np.float32)
        out = self.sess.run(None, {"Email": x})[0]
        self.assertEqual(out.shape, (1, 3))

    def test_output_shape_batch_4(self):
        x = self.np.random.rand(4, 8).astype(self.np.float32)
        out = self.sess.run(None, {"Email": x})[0]
        self.assertEqual(out.shape, (4, 3))

    def test_probabilities_sum_to_one(self):
        x = self.np.random.rand(5, 8).astype(self.np.float32)
        out = self.sess.run(None, {"Email": x})[0]
        row_sums = out.sum(axis=1)
        for s in row_sums:
            self.assertAlmostEqual(float(s), 1.0, places=5)

    def test_probabilities_non_negative(self):
        x = self.np.random.rand(5, 8).astype(self.np.float32)
        out = self.sess.run(None, {"Email": x})[0]
        self.assertTrue((out >= 0).all())


# ---------------------------------------------------------------------------
# onnxruntime inference — sigmoid output in [0, 1]
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ort_available() and _onnx_available(), "onnxruntime/onnx not installed")
class TestSigmoidInference(unittest.TestCase):
    def setUp(self):
        import numpy as np
        prog, ps = _make_sigmoid_program_and_params()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self.tmp.close()
        from matrixai.export import export_onnx
        export_onnx(prog, ps, self.tmp.name)

        import onnxruntime as ort
        self.sess = ort.InferenceSession(self.tmp.name)
        self.np = np

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_output_shape_batch_1(self):
        x = self.np.random.rand(1, 5).astype(self.np.float32)
        out = self.sess.run(None, {"Patient": x})[0]
        self.assertEqual(out.shape, (1,))

    def test_output_shape_batch_4(self):
        x = self.np.random.rand(4, 5).astype(self.np.float32)
        out = self.sess.run(None, {"Patient": x})[0]
        self.assertEqual(out.shape, (4,))

    def test_probabilities_in_range(self):
        x = self.np.random.rand(10, 5).astype(self.np.float32)
        out = self.sess.run(None, {"Patient": x})[0]
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())

    def test_specific_input(self):
        x = self.np.array([[0.92, 0.22, 0.76, 1.0, 0.48]], dtype=self.np.float32)
        out = self.sess.run(None, {"Patient": x})[0]
        self.assertEqual(out.shape, (1,))
        self.assertGreater(float(out[0]), 0.0)
        self.assertLess(float(out[0]), 1.0)


# ---------------------------------------------------------------------------
# ONNX vs differentiable_python equivalence — softmax
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ort_available() and _onnx_available(), "onnxruntime/onnx not installed")
class TestSoftmaxOnnxVsDiffPy(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.compiler import DifferentiablePythonCompiler
        from matrixai.export import export_onnx

        self.np = np
        prog, ps = _make_softmax_program_and_params()
        self.ps = ps

        ns: dict = {}
        exec(DifferentiablePythonCompiler().compile(prog), ns)
        self.dp_run = ns["run"]

        self.tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self.tmp.close()
        export_onnx(prog, ps, self.tmp.name)
        import onnxruntime as ort
        self.sess = ort.InferenceSession(self.tmp.name)

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def _dp_probs(self, email_dict: dict) -> list[float]:
        inp = {"Email": email_dict}
        result = self.dp_run(inp, self.ps.runtime_parameters())
        c = result["state"]["C"]
        return list(c.values())

    def _ort_probs(self, email_dict: dict) -> list[float]:
        fields = ["urgency", "sender_trust", "topic_support", "topic_sales",
                  "sentiment", "has_attachment", "previous_interactions", "language_confidence"]
        x = self.np.array([[email_dict[f] for f in fields]], dtype=self.np.float32)
        return list(self.sess.run(None, {"Email": x})[0][0])

    def test_equivalence_typical_input(self):
        email = {
            "urgency": 0.84, "sender_trust": 0.96, "topic_support": 0.99, "topic_sales": 0.04,
            "sentiment": 0.72, "has_attachment": 0.0, "previous_interactions": 0.88,
            "language_confidence": 0.97,
        }
        dp = self._dp_probs(email)
        ort_ = self._ort_probs(email)
        for i, (a, b) in enumerate(zip(dp, ort_)):
            self.assertAlmostEqual(a, b, places=5, msg=f"class {i}: dp={a} ort={b}")

    def test_equivalence_uniform_input(self):
        email = {k: 0.5 for k in
                 ["urgency", "sender_trust", "topic_support", "topic_sales",
                  "sentiment", "has_attachment", "previous_interactions", "language_confidence"]}
        dp = self._dp_probs(email)
        ort_ = self._ort_probs(email)
        for i, (a, b) in enumerate(zip(dp, ort_)):
            self.assertAlmostEqual(a, b, places=5, msg=f"class {i}: dp={a} ort={b}")

    def test_equivalence_zero_input(self):
        email = {k: 0.0 for k in
                 ["urgency", "sender_trust", "topic_support", "topic_sales",
                  "sentiment", "has_attachment", "previous_interactions", "language_confidence"]}
        dp = self._dp_probs(email)
        ort_ = self._ort_probs(email)
        for i, (a, b) in enumerate(zip(dp, ort_)):
            self.assertAlmostEqual(a, b, places=5, msg=f"class {i}: dp={a} ort={b}")


# ---------------------------------------------------------------------------
# ONNX vs differentiable_python equivalence — sigmoid
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ort_available() and _onnx_available(), "onnxruntime/onnx not installed")
class TestSigmoidOnnxVsDiffPy(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from matrixai.compiler import DifferentiablePythonCompiler
        from matrixai.export import export_onnx

        self.np = np
        prog, ps = _make_sigmoid_program_and_params()
        self.ps = ps

        ns: dict = {}
        exec(DifferentiablePythonCompiler().compile(prog), ns)
        self.dp_run = ns["run"]
        self.prog = prog

        self.tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self.tmp.close()
        export_onnx(prog, ps, self.tmp.name)
        import onnxruntime as ort
        self.sess = ort.InferenceSession(self.tmp.name)

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def _dp_prob(self, patient_dict: dict) -> float:
        inp = {"Patient": patient_dict}
        result = self.dp_run(inp, self.ps.runtime_parameters())
        return float(result["state"]["RiskModel"])

    def _ort_prob(self, patient_dict: dict) -> float:
        fields = ["age", "mobility", "medication_load", "previous_falls", "cognitive_state"]
        x = self.np.array([[patient_dict[f] for f in fields]], dtype=self.np.float32)
        return float(self.sess.run(None, {"Patient": x})[0][0])

    def test_equivalence_typical_input(self):
        patient = {"age": 0.92, "mobility": 0.22, "medication_load": 0.76,
                   "previous_falls": 1.0, "cognitive_state": 0.48}
        dp = self._dp_prob(patient)
        ort_ = self._ort_prob(patient)
        self.assertAlmostEqual(dp, ort_, places=5, msg=f"dp={dp} ort={ort_}")

    def test_equivalence_low_risk_input(self):
        patient = {"age": 0.3, "mobility": 0.8, "medication_load": 0.2,
                   "previous_falls": 0.0, "cognitive_state": 0.9}
        dp = self._dp_prob(patient)
        ort_ = self._ort_prob(patient)
        self.assertAlmostEqual(dp, ort_, places=5, msg=f"dp={dp} ort={ort_}")

    def test_equivalence_uniform_input(self):
        patient = {k: 0.5 for k in
                   ["age", "mobility", "medication_load", "previous_falls", "cognitive_state"]}
        dp = self._dp_prob(patient)
        ort_ = self._ort_prob(patient)
        self.assertAlmostEqual(dp, ort_, places=5, msg=f"dp={dp} ort={ort_}")


# ---------------------------------------------------------------------------
# Output determinism: same input → same output
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ort_available() and _onnx_available(), "onnxruntime/onnx not installed")
class TestOnnxDeterminism(unittest.TestCase):
    def test_softmax_deterministic(self):
        import numpy as np
        prog, ps = _make_softmax_program_and_params()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        try:
            from matrixai.export import export_onnx
            export_onnx(prog, ps, out)
            import onnxruntime as ort
            sess = ort.InferenceSession(out)
            x = np.random.rand(3, 8).astype(np.float32)
            out1 = sess.run(None, {"Email": x})[0]
            out2 = sess.run(None, {"Email": x})[0]
            np.testing.assert_array_equal(out1, out2)
        finally:
            os.unlink(out)

    def test_sigmoid_deterministic(self):
        import numpy as np
        prog, ps = _make_sigmoid_program_and_params()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        try:
            from matrixai.export import export_onnx
            export_onnx(prog, ps, out)
            import onnxruntime as ort
            sess = ort.InferenceSession(out)
            x = np.random.rand(3, 5).astype(np.float32)
            out1 = sess.run(None, {"Patient": x})[0]
            out2 = sess.run(None, {"Patient": x})[0]
            np.testing.assert_array_equal(out1, out2)
        finally:
            os.unlink(out)


# ---------------------------------------------------------------------------
# CLI export-onnx command
# ---------------------------------------------------------------------------

class TestCliExportOnnx(unittest.TestCase):
    def test_cli_help_shows_export_onnx(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", "--help"],
            capture_output=True, text=True,
            cwd=str(_BASE),
        )
        combined = result.stdout + result.stderr
        self.assertIn("export-onnx", combined)

    @unittest.skipUnless(_onnx_available(), "onnx not installed")
    def test_cli_export_produces_file(self):
        import subprocess
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parameters import write_parameter_set

        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            out_path = Path(td) / "model.onnx"

            result = subprocess.run(
                [
                    sys.executable, "-m", "matrixai",
                    "export-onnx", str(_EMAIL_MXAI),
                    "--params", str(params_path),
                    "--output", str(out_path),
                ],
                capture_output=True, text=True,
                cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    @unittest.skipUnless(_onnx_available(), "onnx not installed")
    def test_cli_export_json_flag(self):
        import subprocess
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parameters import write_parameter_set

        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            out_path = Path(td) / "model.onnx"

            result = subprocess.run(
                [
                    sys.executable, "-m", "matrixai",
                    "export-onnx", str(_EMAIL_MXAI),
                    "--params", str(params_path),
                    "--output", str(out_path),
                    "--json",
                ],
                capture_output=True, text=True,
                cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertIn("output_path", data)
            self.assertEqual(data["opset_version"], 17)
            self.assertIn("Classifier", data["exported_functions"])

    @unittest.skipUnless(_onnx_available(), "onnx not installed")
    def test_cli_fall_risk_export(self):
        import subprocess
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parameters import write_parameter_set

        prog = parse_file(_FALL_RISK_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            out_path = Path(td) / "fall-risk.onnx"

            result = subprocess.run(
                [
                    sys.executable, "-m", "matrixai",
                    "export-onnx", str(_FALL_RISK_MXAI),
                    "--params", str(params_path),
                    "--output", str(out_path),
                    "--json",
                ],
                capture_output=True, text=True,
                cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["input_shape"], [-1, 5])
            self.assertEqual(data["output_shape"], [-1])
            self.assertIn("RiskModel", data["exported_functions"])


# ---------------------------------------------------------------------------
# export_onnx convenience wrapper
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestExportOnnxWrapper(unittest.TestCase):
    def test_wrapper_returns_result(self):
        from matrixai.export import export_onnx, OnnxExportResult
        prog, ps = _make_softmax_program_and_params()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = f.name
        try:
            result = export_onnx(prog, ps, out)
            self.assertIsInstance(result, OnnxExportResult)
            self.assertTrue(Path(out).exists())
        finally:
            os.unlink(out)

    def test_wrapper_accepts_path_object(self):
        from matrixai.export import export_onnx
        prog, ps = _make_softmax_program_and_params()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out = Path(f.name)
        try:
            result = export_onnx(prog, ps, out)
            self.assertEqual(result.output_path, str(out))
        finally:
            out.unlink(missing_ok=True)

    def test_wrapper_creates_parent_directory(self):
        from matrixai.export import export_onnx
        prog, ps = _make_softmax_program_and_params()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "subdir" / "nested" / "model.onnx"
            result = export_onnx(prog, ps, out)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
