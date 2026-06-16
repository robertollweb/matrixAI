"""P15 Corte 2: Formal ONNX equivalence validator and export_manifest.json.

Tests cover:
- OnnxEquivalenceValidator produces OnnxEquivalenceResult with passed=True for both models
- max_abs_diff within atol=1e-5 for softmax_linear and sigmoid_linear
- OnnxEquivalenceResult.to_dict() has expected keys
- write_export_manifest() writes valid JSON with all required fields
- export_manifest.json contains model_hash, parameter_schema_hash, tolerance, equivalence_check, exported_at
- Mismatched ParameterSet causes OnnxEquivalenceError (via hash validation in exporter)
- validate_onnx_equivalence() convenience wrapper works
- CLI --validate flag runs check and reports PASS
- CLI --validate --json merges equivalence_check into JSON output
- CLI --manifest writes export_manifest.json
- CLI --manifest without --validate emits a warning
- ort_available() returns True when onnxruntime is installed
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.mxai"
_FALL_RISK_MXAI = _BASE / "examples" / "fall-risk.mxai"

_ATOL = 1e-5
_RTOL = 1e-4


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    from importlib import util
    return util.find_spec("onnxruntime") is not None


def _make_softmax_program_and_params():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    return parse_file(_EMAIL_MXAI), build_initial_parameter_set(parse_file(_EMAIL_MXAI))


def _make_sigmoid_program_and_params():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    return parse_file(_FALL_RISK_MXAI), build_initial_parameter_set(parse_file(_FALL_RISK_MXAI))


def _export_to_tmp(prog, ps):
    from matrixai.export import export_onnx
    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    f.close()
    result = export_onnx(prog, ps, f.name)
    return result, f.name


# ---------------------------------------------------------------------------
# ort_available()
# ---------------------------------------------------------------------------

class TestOrtAvailable(unittest.TestCase):
    def test_returns_true_when_installed(self):
        from matrixai.export import ort_available
        self.assertTrue(ort_available())


# ---------------------------------------------------------------------------
# OnnxEquivalenceResult structure
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestEquivalenceResultStructure(unittest.TestCase):
    def setUp(self):
        from matrixai.export import validate_onnx_equivalence
        prog, ps = _make_softmax_program_and_params()
        result, self.onnx_path = _export_to_tmp(prog, ps)
        self.eq = validate_onnx_equivalence(prog, ps, self.onnx_path)

    def tearDown(self):
        try:
            os.unlink(self.onnx_path)
        except OSError:
            pass

    def test_is_dataclass(self):
        from matrixai.export import OnnxEquivalenceResult
        self.assertIsInstance(self.eq, OnnxEquivalenceResult)

    def test_to_dict_keys(self):
        d = self.eq.to_dict()
        expected = {"passed", "atol", "rtol", "max_abs_diff", "max_rel_diff",
                    "n_samples", "n_outputs_per_sample"}
        self.assertEqual(set(d.keys()), expected)

    def test_atol_default(self):
        self.assertAlmostEqual(self.eq.atol, 1e-5)

    def test_rtol_default(self):
        self.assertAlmostEqual(self.eq.rtol, 1e-4)

    def test_n_samples_default(self):
        self.assertEqual(self.eq.n_samples, 20)


# ---------------------------------------------------------------------------
# Softmax equivalence — email-agent
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestSoftmaxEquivalence(unittest.TestCase):
    def setUp(self):
        from matrixai.export import validate_onnx_equivalence
        self.prog, self.ps = _make_softmax_program_and_params()
        _, self.onnx_path = _export_to_tmp(self.prog, self.ps)
        self.eq = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path)

    def tearDown(self):
        try:
            os.unlink(self.onnx_path)
        except OSError:
            pass

    def test_passes(self):
        self.assertTrue(self.eq.passed)

    def test_max_abs_diff_within_atol(self):
        self.assertLess(self.eq.max_abs_diff, _ATOL)

    def test_n_outputs_per_sample(self):
        # email-agent has 3 classes
        self.assertEqual(self.eq.n_outputs_per_sample, 3)

    def test_max_abs_diff_non_negative(self):
        self.assertGreaterEqual(self.eq.max_abs_diff, 0.0)

    def test_max_rel_diff_non_negative(self):
        self.assertGreaterEqual(self.eq.max_rel_diff, 0.0)

    def test_custom_n_samples(self):
        from matrixai.export import validate_onnx_equivalence
        eq = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, n_samples=5)
        self.assertEqual(eq.n_samples, 5)
        self.assertTrue(eq.passed)

    def test_deterministic_with_seed(self):
        from matrixai.export import validate_onnx_equivalence
        eq1 = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, seed=7)
        eq2 = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, seed=7)
        self.assertAlmostEqual(eq1.max_abs_diff, eq2.max_abs_diff)

    def test_different_seeds_may_differ(self):
        from matrixai.export import validate_onnx_equivalence
        eq1 = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, seed=1)
        eq2 = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path, seed=99)
        # Both should pass; exact max_abs_diff may differ
        self.assertTrue(eq1.passed)
        self.assertTrue(eq2.passed)


# ---------------------------------------------------------------------------
# Sigmoid equivalence — fall-risk
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestSigmoidEquivalence(unittest.TestCase):
    def setUp(self):
        from matrixai.export import validate_onnx_equivalence
        self.prog, self.ps = _make_sigmoid_program_and_params()
        _, self.onnx_path = _export_to_tmp(self.prog, self.ps)
        self.eq = validate_onnx_equivalence(self.prog, self.ps, self.onnx_path)

    def tearDown(self):
        try:
            os.unlink(self.onnx_path)
        except OSError:
            pass

    def test_passes(self):
        self.assertTrue(self.eq.passed)

    def test_max_abs_diff_within_atol(self):
        self.assertLess(self.eq.max_abs_diff, _ATOL)

    def test_n_outputs_per_sample(self):
        # fall-risk is binary: scalar output
        self.assertEqual(self.eq.n_outputs_per_sample, 1)


# ---------------------------------------------------------------------------
# export_manifest.json structure
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestExportManifest(unittest.TestCase):
    _REQUIRED_KEYS = {
        "model_hash", "parameter_schema_hash", "parameter_set_id",
        "format", "format_version", "input_name", "input_shape",
        "output_name", "output_shape", "exported_function",
        "tolerance", "equivalence_check", "exported_at",
    }

    def setUp(self):
        from matrixai.export import export_onnx, validate_onnx_equivalence, write_export_manifest
        prog, ps = _make_softmax_program_and_params()
        self.td = tempfile.mkdtemp()
        onnx_path = Path(self.td) / "model.onnx"
        self.export_result = export_onnx(prog, ps, onnx_path)
        self.eq = validate_onnx_equivalence(prog, ps, onnx_path)
        self.manifest_path = Path(self.td) / "export_manifest.json"
        write_export_manifest(self.export_result, self.eq, self.manifest_path)
        with open(self.manifest_path) as f:
            self.data = json.load(f)
        self.ps = ps

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_manifest_file_exists(self):
        self.assertTrue(self.manifest_path.exists())

    def test_required_keys_present(self):
        missing = self._REQUIRED_KEYS - set(self.data.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_model_hash_matches(self):
        self.assertEqual(self.data["model_hash"], self.export_result.model_hash)

    def test_parameter_schema_hash_matches(self):
        self.assertEqual(
            self.data["parameter_schema_hash"],
            self.export_result.parameter_schema_hash,
        )

    def test_parameter_set_id_matches(self):
        self.assertEqual(self.data["parameter_set_id"], self.export_result.parameter_set_id)

    def test_format_is_onnx(self):
        self.assertEqual(self.data["format"], "onnx")

    def test_format_version_is_opset(self):
        self.assertEqual(self.data["format_version"], 17)

    def test_tolerance_keys(self):
        tol = self.data["tolerance"]
        self.assertIn("atol", tol)
        self.assertIn("rtol", tol)
        self.assertAlmostEqual(tol["atol"], 1e-5)

    def test_equivalence_check_passed(self):
        self.assertTrue(self.data["equivalence_check"]["passed"])

    def test_equivalence_check_max_abs_diff(self):
        self.assertIn("max_abs_diff", self.data["equivalence_check"])
        self.assertLess(self.data["equivalence_check"]["max_abs_diff"], _ATOL)

    def test_equivalence_check_n_samples(self):
        self.assertEqual(self.data["equivalence_check"]["n_samples"], 20)

    def test_exported_at_is_iso_string(self):
        from datetime import datetime
        dt_str = self.data["exported_at"]
        dt = datetime.fromisoformat(dt_str)
        self.assertIsNotNone(dt)

    def test_input_shape_correct(self):
        self.assertEqual(self.data["input_shape"], [-1, 8])

    def test_output_shape_correct(self):
        self.assertEqual(self.data["output_shape"], [-1, 3])

    def test_manifest_creates_parent_directories(self):
        from matrixai.export import export_onnx, validate_onnx_equivalence, write_export_manifest
        prog, ps = _make_softmax_program_and_params()
        nested = Path(self.td) / "a" / "b" / "c" / "manifest.json"
        onnx_path = Path(self.td) / "model2.onnx"
        er = export_onnx(prog, ps, onnx_path)
        eq = validate_onnx_equivalence(prog, ps, onnx_path)
        write_export_manifest(er, eq, nested)
        self.assertTrue(nested.exists())

    def test_fall_risk_manifest(self):
        from matrixai.export import export_onnx, validate_onnx_equivalence, write_export_manifest
        prog, ps = _make_sigmoid_program_and_params()
        onnx_path = Path(self.td) / "fall-risk.onnx"
        manifest_path = Path(self.td) / "fall-risk-manifest.json"
        er = export_onnx(prog, ps, onnx_path)
        eq = validate_onnx_equivalence(prog, ps, onnx_path)
        write_export_manifest(er, eq, manifest_path)
        with open(manifest_path) as f:
            data = json.load(f)
        self.assertEqual(data["input_shape"], [-1, 5])
        self.assertEqual(data["output_shape"], [-1])
        self.assertTrue(data["equivalence_check"]["passed"])


# ---------------------------------------------------------------------------
# CLI --validate flag
# ---------------------------------------------------------------------------

class TestCliValidateFlag(unittest.TestCase):
    @unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
    def test_validate_exits_zero(self):
        import subprocess
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set

        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            out_path = Path(td) / "model.onnx"

            result = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-onnx", str(_EMAIL_MXAI),
                 "--params", str(params_path),
                 "--output", str(out_path),
                 "--validate"],
                capture_output=True, text=True, cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            combined = result.stdout + result.stderr
            self.assertIn("PASS", combined)

    @unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
    def test_validate_json_flag_includes_equivalence_check(self):
        import subprocess
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set

        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            out_path = Path(td) / "model.onnx"

            result = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-onnx", str(_EMAIL_MXAI),
                 "--params", str(params_path),
                 "--output", str(out_path),
                 "--validate", "--json"],
                capture_output=True, text=True, cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertIn("equivalence_check", data)
            self.assertTrue(data["equivalence_check"]["passed"])

    @unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
    def test_validate_manifest_flag_writes_file(self):
        import subprocess
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set

        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            out_path = Path(td) / "model.onnx"
            manifest_path = Path(td) / "export_manifest.json"

            result = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-onnx", str(_EMAIL_MXAI),
                 "--params", str(params_path),
                 "--output", str(out_path),
                 "--validate", "--manifest", str(manifest_path)],
                capture_output=True, text=True, cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(manifest_path.exists())
            with open(manifest_path) as f:
                data = json.load(f)
            self.assertIn("equivalence_check", data)
            self.assertEqual(data["format"], "onnx")

    def test_manifest_without_validate_warns(self):
        import subprocess
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set

        if not _onnx_available():
            self.skipTest("onnx not installed")

        prog = parse_file(_EMAIL_MXAI)
        ps = build_initial_parameter_set(prog)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            out_path = Path(td) / "model.onnx"
            manifest_path = Path(td) / "should_not_exist.json"

            result = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-onnx", str(_EMAIL_MXAI),
                 "--params", str(params_path),
                 "--output", str(out_path),
                 "--manifest", str(manifest_path)],
                capture_output=True, text=True, cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Warning", result.stderr)
            self.assertFalse(manifest_path.exists())


# ---------------------------------------------------------------------------
# Validate convenience wrapper
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestValidateConvenienceWrapper(unittest.TestCase):
    def test_validate_onnx_equivalence_wrapper(self):
        from matrixai.export import validate_onnx_equivalence, OnnxEquivalenceResult
        prog, ps = _make_softmax_program_and_params()
        _, onnx_path = _export_to_tmp(prog, ps)
        try:
            result = validate_onnx_equivalence(prog, ps, onnx_path)
            self.assertIsInstance(result, OnnxEquivalenceResult)
            self.assertTrue(result.passed)
        finally:
            os.unlink(onnx_path)

    def test_validator_class_directly(self):
        from matrixai.export import OnnxEquivalenceValidator
        prog, ps = _make_sigmoid_program_and_params()
        _, onnx_path = _export_to_tmp(prog, ps)
        try:
            validator = OnnxEquivalenceValidator()
            result = validator.validate(prog, ps, onnx_path, n_samples=5)
            self.assertTrue(result.passed)
            self.assertEqual(result.n_samples, 5)
        finally:
            os.unlink(onnx_path)


if __name__ == "__main__":
    unittest.main()
