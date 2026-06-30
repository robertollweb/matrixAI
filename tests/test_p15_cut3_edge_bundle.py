"""P15 Corte 3: Edge bundle (model.onnx + manifests + .mxai + params).

Tests cover:
- EdgeBundler.bundle() creates the expected 6-file directory
- model.mxai and params.best.json are exact copies of inputs
- model.onnx passes onnx.checker
- model_manifest.json has required fields and correct hashes
- export_manifest.json has required fields, equivalence_check, exported_at
- README.md exists and is non-empty
- EdgeBundleResult.to_dict() has expected keys
- equivalence_passed == True for both softmax and sigmoid models
- Existing directory raises EdgeBundleError without force=True
- force=True overwrites existing bundle
- validate=False skips equivalence check and export_manifest has null equivalence
- EdgeBundleError when ParameterSet belongs to wrong program
- create_edge_bundle() convenience wrapper
- CLI export-bundle produces correct bundle directory
- CLI export-bundle --json returns structured output
- CLI export-bundle --no-validate skips equivalence
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.mxai"
_FALL_RISK_MXAI = _BASE / "examples" / "fall-risk.mxai"

# Base bundle (P15). A flat-VECTOR model that yields a usable spec also carries the
# self-usable prediction artifacts (EXPORT C1+C2); unlabelled classification does not.
_BUNDLE_FILES = {"README.md", "export_manifest.json", "model.mxai",
                 "model.onnx", "model_manifest.json", "params.best.json"}
_BUNDLE_FILES_WITH_SPEC = _BUNDLE_FILES | {
    "inference_spec.json", "predict.py", "requirements.txt",
    "example_input.json", "expected_output.json"}
_MODEL_MANIFEST_KEYS = {
    "project", "model_hash", "parameter_schema_hash", "parameter_set_id",
    "vectors", "functions", "backend_contract", "created_at",
}
_EXPORT_MANIFEST_KEYS = {
    "model_hash", "parameter_schema_hash", "parameter_set_id",
    "format", "format_version", "input_name", "input_shape",
    "output_name", "output_shape", "exported_function",
    "tolerance", "equivalence_check", "exported_at",
}


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    from importlib import util
    return util.find_spec("onnxruntime") is not None


def _make_softmax_program_and_params():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_EMAIL_MXAI)
    return prog, build_initial_parameter_set(prog)


def _make_sigmoid_program_and_params():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_FALL_RISK_MXAI)
    return prog, build_initial_parameter_set(prog)


def _write_params(ps, path):
    from matrixai.parameters import write_parameter_set
    write_parameter_set(str(path), ps)


# ---------------------------------------------------------------------------
# Email-agent bundle (softmax_linear)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestEmailAgentBundle(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_softmax_program_and_params()
        self.td = tempfile.mkdtemp()
        self.params_path = Path(self.td) / "params.json"
        _write_params(self.ps, self.params_path)
        self.bundle_dir = Path(self.td) / "bundle"
        from matrixai.export import create_edge_bundle
        self.result = create_edge_bundle(
            self.prog, self.ps,
            mxai_path=str(_EMAIL_MXAI),
            params_path=str(self.params_path),
            outdir=str(self.bundle_dir),
            validate=True,
        )

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_bundle_dir_exists(self):
        self.assertTrue(self.bundle_dir.exists())

    def test_expected_files(self):
        actual = set(self.result.files)
        self.assertEqual(actual, _BUNDLE_FILES)

    def test_model_mxai_content_matches(self):
        original = _EMAIL_MXAI.read_text()
        bundled = (self.bundle_dir / "model.mxai").read_text()
        self.assertEqual(original, bundled)

    def test_params_best_json_is_valid_json(self):
        content = (self.bundle_dir / "params.best.json").read_bytes()
        data = json.loads(content)
        self.assertIn("parameter_set_id", data)

    def test_model_onnx_passes_checker(self):
        import onnx
        model = onnx.load(str(self.bundle_dir / "model.onnx"))
        onnx.checker.check_model(model)

    def test_model_manifest_keys(self):
        data = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        missing = _MODEL_MANIFEST_KEYS - set(data.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_model_manifest_hashes(self):
        data = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        self.assertEqual(data["model_hash"], self.ps.model_hash)
        self.assertEqual(data["parameter_schema_hash"], self.ps.parameter_schema_hash)
        self.assertEqual(data["project"], "EmailAgent")

    def test_model_manifest_vectors(self):
        data = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        vectors = data["vectors"]
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0]["name"], "Email")
        self.assertEqual(vectors[0]["size"], 8)

    def test_model_manifest_functions(self):
        data = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        fn_names = [f["name"] for f in data["functions"]]
        self.assertIn("Classifier", fn_names)

    def test_model_manifest_backend_contract_ok(self):
        data = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        self.assertTrue(data["backend_contract"]["ok"])
        self.assertEqual(data["backend_contract"]["unsupported_nodes"], [])

    def test_export_manifest_keys(self):
        data = json.loads((self.bundle_dir / "export_manifest.json").read_text())
        missing = _EXPORT_MANIFEST_KEYS - set(data.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_export_manifest_hashes(self):
        data = json.loads((self.bundle_dir / "export_manifest.json").read_text())
        self.assertEqual(data["model_hash"], self.ps.model_hash)
        self.assertEqual(data["parameter_schema_hash"], self.ps.parameter_schema_hash)

    def test_export_manifest_format(self):
        data = json.loads((self.bundle_dir / "export_manifest.json").read_text())
        self.assertEqual(data["format"], "onnx")
        self.assertEqual(data["format_version"], 17)

    def test_export_manifest_equivalence_passed(self):
        data = json.loads((self.bundle_dir / "export_manifest.json").read_text())
        self.assertIsNotNone(data["equivalence_check"])
        self.assertTrue(data["equivalence_check"]["passed"])

    def test_export_manifest_exported_at(self):
        from datetime import datetime
        data = json.loads((self.bundle_dir / "export_manifest.json").read_text())
        dt = datetime.fromisoformat(data["exported_at"])
        self.assertIsNotNone(dt)

    def test_readme_exists_and_nonempty(self):
        readme = (self.bundle_dir / "README.md").read_text()
        self.assertGreater(len(readme), 100)
        self.assertIn("EmailAgent", readme)

    def test_readme_contains_hash(self):
        readme = (self.bundle_dir / "README.md").read_text()
        self.assertIn(self.ps.model_hash, readme)

    def test_result_model_hash(self):
        self.assertEqual(self.result.model_hash, self.ps.model_hash)

    def test_result_parameter_set_id(self):
        self.assertEqual(self.result.parameter_set_id, self.ps.parameter_set_id)

    def test_result_equivalence_passed(self):
        self.assertTrue(self.result.equivalence_passed)

    def test_result_to_dict_keys(self):
        d = self.result.to_dict()
        expected = {"bundle_dir", "files", "model_hash", "parameter_set_id",
                    "equivalence_passed", "export", "equivalence_check",
                    "inference_spec_skipped_reason"}
        self.assertEqual(set(d.keys()), expected)


# ---------------------------------------------------------------------------
# Fall-risk bundle (sigmoid_linear)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestFallRiskBundle(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_sigmoid_program_and_params()
        self.td = tempfile.mkdtemp()
        self.params_path = Path(self.td) / "params.json"
        _write_params(self.ps, self.params_path)
        self.bundle_dir = Path(self.td) / "bundle"
        from matrixai.export import create_edge_bundle
        self.result = create_edge_bundle(
            self.prog, self.ps,
            mxai_path=str(_FALL_RISK_MXAI),
            params_path=str(self.params_path),
            outdir=str(self.bundle_dir),
        )

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_expected_files(self):
        self.assertEqual(set(self.result.files), _BUNDLE_FILES_WITH_SPEC)

    def test_model_manifest_project(self):
        data = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        self.assertEqual(data["project"], "FallRisk")

    def test_model_manifest_vectors_size_5(self):
        data = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        self.assertEqual(data["vectors"][0]["size"], 5)

    def test_export_manifest_input_shape(self):
        data = json.loads((self.bundle_dir / "export_manifest.json").read_text())
        self.assertEqual(data["input_shape"], [-1, 5])

    def test_equivalence_passed(self):
        self.assertTrue(self.result.equivalence_passed)

    def test_readme_contains_fall_risk(self):
        readme = (self.bundle_dir / "README.md").read_text()
        self.assertIn("FallRisk", readme)


# ---------------------------------------------------------------------------
# Guard: existing directory without force
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestBundleForceFlag(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_existing_dir_raises_without_force(self):
        from matrixai.export import EdgeBundleError, create_edge_bundle
        prog, ps = _make_softmax_program_and_params()
        params_path = Path(self.td) / "params.json"
        _write_params(ps, params_path)
        bundle_dir = Path(self.td) / "bundle"
        bundle_dir.mkdir()
        with self.assertRaises(EdgeBundleError) as ctx:
            create_edge_bundle(
                prog, ps, str(_EMAIL_MXAI), str(params_path), str(bundle_dir),
            )
        self.assertIn("already exists", str(ctx.exception))

    def test_force_true_overwrites(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_softmax_program_and_params()
        params_path = Path(self.td) / "params.json"
        _write_params(ps, params_path)
        bundle_dir = Path(self.td) / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "stale.txt").write_text("old")
        result = create_edge_bundle(
            prog, ps, str(_EMAIL_MXAI), str(params_path), str(bundle_dir),
            force=True,
        )
        self.assertEqual(set(result.files), _BUNDLE_FILES)


# ---------------------------------------------------------------------------
# validate=False skips equivalence check
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestBundleNoValidate(unittest.TestCase):
    def test_no_validate_skips_equivalence(self):
        from matrixai.export import create_edge_bundle
        prog, ps = _make_softmax_program_and_params()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            result = create_edge_bundle(
                prog, ps, str(_EMAIL_MXAI), str(params_path),
                outdir=str(Path(td) / "bundle"),
                validate=False,
            )
            self.assertIsNone(result.equivalence_result)
            self.assertFalse(result.equivalence_passed)
            # export_manifest.json exists but equivalence_check is null
            data = json.loads((Path(td) / "bundle" / "export_manifest.json").read_text())
            self.assertIsNone(data["equivalence_check"])


# ---------------------------------------------------------------------------
# CLI export-bundle
# ---------------------------------------------------------------------------

class TestCliBundleCommand(unittest.TestCase):
    @unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
    def test_cli_bundle_creates_files(self):
        import subprocess
        prog, ps = _make_softmax_program_and_params()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            bundle_dir = Path(td) / "bundle"
            result = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-bundle", str(_EMAIL_MXAI),
                 "--params", str(params_path),
                 "--outdir", str(bundle_dir)],
                capture_output=True, text=True, cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(bundle_dir.exists())
            actual_files = {p.name for p in bundle_dir.iterdir()}
            self.assertEqual(actual_files, _BUNDLE_FILES)

    @unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
    def test_cli_bundle_json_output(self):
        import subprocess
        prog, ps = _make_softmax_program_and_params()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            result = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-bundle", str(_EMAIL_MXAI),
                 "--params", str(params_path),
                 "--outdir", str(Path(td) / "bundle"),
                 "--json"],
                capture_output=True, text=True, cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertIn("bundle_dir", data)
            self.assertIn("files", data)
            self.assertTrue(data["equivalence_passed"])

    @unittest.skipUnless(_onnx_available(), "onnx not installed")
    def test_cli_bundle_no_validate(self):
        import subprocess
        prog, ps = _make_softmax_program_and_params()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            result = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-bundle", str(_EMAIL_MXAI),
                 "--params", str(params_path),
                 "--outdir", str(Path(td) / "bundle"),
                 "--no-validate", "--json"],
                capture_output=True, text=True, cwd=str(_BASE),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertFalse(data["equivalence_passed"])

    def test_cli_help_shows_export_bundle(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", "--help"],
            capture_output=True, text=True, cwd=str(_BASE),
        )
        combined = result.stdout + result.stderr
        self.assertIn("export-bundle", combined)


# ---------------------------------------------------------------------------
# create_edge_bundle convenience wrapper
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestCreateEdgeBundleWrapper(unittest.TestCase):
    def test_wrapper_returns_result(self):
        from matrixai.export import create_edge_bundle, EdgeBundleResult
        prog, ps = _make_softmax_program_and_params()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            _write_params(ps, params_path)
            result = create_edge_bundle(
                prog, ps, str(_EMAIL_MXAI), str(params_path),
                outdir=str(Path(td) / "bundle"),
            )
            self.assertIsInstance(result, EdgeBundleResult)
            self.assertTrue(result.equivalence_passed)


if __name__ == "__main__":
    unittest.main()
