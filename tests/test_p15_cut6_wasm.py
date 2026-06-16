"""P15 Corte 6: WASM bundle export via ONNX Runtime Web.

Tests cover:
- Bundle creation: 3 files (model.onnx, wasm_manifest.json, predict.js)
- wasm_manifest.json fields: format, hashes, equivalence_check, tolerance, input/output shapes
- predict.js: exists, contains ORT Web CDN reference, correct input/output names
- predict.js SEQUENCE model: BigInt64Array / int64
- predict.js VECTOR model: Float32Array / float32
- model.onnx validity (onnx.checker passes)
- WasmExportResult fields and to_dict()
- WasmExporter.export() with validate=False (skips ORT requirement)
- force=True overwrites existing directory
- force=False raises on existing directory
- export_wasm() convenience function
- CLI export-wasm command
- Hash mismatch raises WasmExportError
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


def _make_vec_transformer():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_TRANSFORMER_VEC_MXAI)
    return prog, build_initial_parameter_set(prog)


def _make_email_agent():
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(_EMAIL_MXAI)
    return prog, build_initial_parameter_set(prog)


# ---------------------------------------------------------------------------
# WasmExporter basic creation — SEQUENCE model
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestWasmBundleSequence(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_seq_transformer()
        self.td = tempfile.mkdtemp()
        self.bundle_dir = Path(self.td) / "wasm_bundle"
        from matrixai.export import WasmExporter
        self.result = WasmExporter().export(self.prog, self.ps, self.bundle_dir)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_bundle_dir_created(self):
        self.assertTrue(self.bundle_dir.exists())

    def test_three_files_present(self):
        self.assertEqual(set(self.result.files), {"model.onnx", "wasm_manifest.json", "predict.js"})

    def test_model_onnx_exists(self):
        self.assertTrue((self.bundle_dir / "model.onnx").exists())

    def test_wasm_manifest_exists(self):
        self.assertTrue((self.bundle_dir / "wasm_manifest.json").exists())

    def test_predict_js_exists(self):
        self.assertTrue((self.bundle_dir / "predict.js").exists())

    def test_wasm_runtime_field(self):
        self.assertEqual(self.result.wasm_runtime, "onnxruntime-web")

    def test_model_hash(self):
        self.assertEqual(self.result.model_hash, self.ps.model_hash)

    def test_parameter_set_id(self):
        self.assertEqual(self.result.parameter_set_id, self.ps.parameter_set_id)

    def test_parameter_schema_hash(self):
        self.assertEqual(self.result.parameter_schema_hash, self.ps.parameter_schema_hash)

    def test_input_shape(self):
        self.assertEqual(self.result.input_shape, [-1, 8])

    def test_output_shape(self):
        self.assertEqual(self.result.output_shape, [-1, 2])

    def test_opset_version(self):
        self.assertEqual(self.result.opset_version, 17)

    def test_equivalence_passed(self):
        self.assertTrue(self.result.equivalence_passed)

    def test_equivalence_result_not_none(self):
        self.assertIsNotNone(self.result.equivalence_result)


# ---------------------------------------------------------------------------
# wasm_manifest.json fields — SEQUENCE model
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestWasmManifestSequence(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_seq_transformer()
        self.td = tempfile.mkdtemp()
        self.bundle_dir = Path(self.td) / "bundle"
        from matrixai.export import WasmExporter
        self.result = WasmExporter().export(self.prog, self.ps, self.bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "wasm_manifest.json").read_text())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_format_field(self):
        self.assertEqual(self.manifest["format"], "wasm-onnxruntime-web")

    def test_wasm_runtime_field(self):
        self.assertEqual(self.manifest["wasm_runtime"], "onnxruntime-web")

    def test_ort_web_min_version(self):
        self.assertEqual(self.manifest["ort_web_min_version"], "1.14")

    def test_model_hash(self):
        self.assertEqual(self.manifest["model_hash"], self.ps.model_hash)

    def test_parameter_schema_hash(self):
        self.assertEqual(self.manifest["parameter_schema_hash"], self.ps.parameter_schema_hash)

    def test_parameter_set_id(self):
        self.assertEqual(self.manifest["parameter_set_id"], self.ps.parameter_set_id)

    def test_input_name(self):
        self.assertEqual(self.manifest["input_name"], "Input")

    def test_input_shape(self):
        self.assertEqual(self.manifest["input_shape"], [-1, 8])

    def test_output_name(self):
        self.assertEqual(self.manifest["output_name"], "classifier.result")

    def test_output_shape(self):
        self.assertEqual(self.manifest["output_shape"], [-1, 2])

    def test_onnx_opset(self):
        self.assertEqual(self.manifest["onnx_opset"], 17)

    def test_equivalence_check_passed(self):
        self.assertTrue(self.manifest["equivalence_check"]["passed"])

    def test_tolerance_present(self):
        self.assertIsNotNone(self.manifest["tolerance"])
        self.assertIn("atol", self.manifest["tolerance"])
        self.assertIn("rtol", self.manifest["tolerance"])

    def test_exported_at_present(self):
        self.assertIn("exported_at", self.manifest)
        self.assertTrue(self.manifest["exported_at"])


# ---------------------------------------------------------------------------
# predict.js content — SEQUENCE model (int64 / BigInt64Array)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestPredictJsSequence(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_seq_transformer()
        self.td = tempfile.mkdtemp()
        self.bundle_dir = Path(self.td) / "bundle"
        from matrixai.export import WasmExporter
        WasmExporter().export(self.prog, self.ps, self.bundle_dir)
        self.js = (self.bundle_dir / "predict.js").read_text()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_ort_cdn_reference(self):
        self.assertIn("onnxruntime-web", self.js)

    def test_predict_function_present(self):
        self.assertIn("async function predict", self.js)

    def test_input_name_present(self):
        self.assertIn("Input", self.js)

    def test_output_name_present(self):
        self.assertIn("classifier.result", self.js)

    def test_model_hash_present(self):
        self.assertIn(self.ps.model_hash, self.js)

    def test_int64_dtype(self):
        self.assertIn("int64", self.js)

    def test_bigint64array(self):
        self.assertIn("BigInt64Array", self.js)

    def test_no_float32array(self):
        self.assertNotIn("Float32Array", self.js)

    def test_model_onnx_reference(self):
        self.assertIn("./model.onnx", self.js)


# ---------------------------------------------------------------------------
# predict.js content — VECTOR model (float32 / Float32Array)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestPredictJsVector(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_vec_transformer()
        self.td = tempfile.mkdtemp()
        self.bundle_dir = Path(self.td) / "bundle"
        from matrixai.export import WasmExporter
        WasmExporter().export(self.prog, self.ps, self.bundle_dir)
        self.js = (self.bundle_dir / "predict.js").read_text()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_float32_dtype(self):
        self.assertIn("float32", self.js)

    def test_float32array(self):
        self.assertIn("Float32Array", self.js)

    def test_no_bigint64array(self):
        self.assertNotIn("BigInt64Array", self.js)


# ---------------------------------------------------------------------------
# model.onnx validity
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestWasmModelOnnxValid(unittest.TestCase):
    def test_onnx_checker_passes_sequence(self):
        import onnx
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            WasmExporter().export(prog, ps, bundle_dir)
            model = onnx.load(str(bundle_dir / "model.onnx"))
            onnx.checker.check_model(model)

    def test_onnx_checker_passes_vector(self):
        import onnx
        prog, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            WasmExporter().export(prog, ps, bundle_dir)
            model = onnx.load(str(bundle_dir / "model.onnx"))
            onnx.checker.check_model(model)


# ---------------------------------------------------------------------------
# WasmExportResult.to_dict()
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestWasmExportResultToDict(unittest.TestCase):
    def setUp(self):
        self.prog, self.ps = _make_seq_transformer()
        self.td = tempfile.mkdtemp()
        self.bundle_dir = Path(self.td) / "bundle"
        from matrixai.export import WasmExporter
        self.result = WasmExporter().export(self.prog, self.ps, self.bundle_dir)
        self.d = self.result.to_dict()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_bundle_dir_key(self):
        self.assertIn("bundle_dir", self.d)

    def test_files_key(self):
        self.assertIn("files", self.d)
        self.assertEqual(set(self.d["files"]), {"model.onnx", "wasm_manifest.json", "predict.js"})

    def test_model_hash_key(self):
        self.assertIn("model_hash", self.d)

    def test_wasm_runtime_key(self):
        self.assertIn("wasm_runtime", self.d)

    def test_equivalence_passed_key(self):
        self.assertIn("equivalence_passed", self.d)
        self.assertTrue(self.d["equivalence_passed"])

    def test_equivalence_check_key(self):
        self.assertIn("equivalence_check", self.d)

    def test_input_output_shapes(self):
        self.assertEqual(self.d["input_shape"], [-1, 8])
        self.assertEqual(self.d["output_shape"], [-1, 2])


# ---------------------------------------------------------------------------
# validate=False path (skips ORT requirement)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestWasmNoValidate(unittest.TestCase):
    def test_export_without_validate(self):
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            self.assertIsNone(result.equivalence_result)
            self.assertFalse(result.equivalence_passed)
            self.assertEqual(set(result.files), {"model.onnx", "wasm_manifest.json", "predict.js"})

    def test_manifest_no_tolerance_when_no_validate(self):
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            WasmExporter().export(prog, ps, bundle_dir, validate=False)
            manifest = json.loads((bundle_dir / "wasm_manifest.json").read_text())
            self.assertIsNone(manifest["tolerance"])
            self.assertIsNone(manifest["equivalence_check"])


# ---------------------------------------------------------------------------
# force flag
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestWasmForceFlag(unittest.TestCase):
    def test_force_false_raises_on_existing_dir(self):
        from matrixai.export import WasmExporter, WasmExportError
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            bundle_dir.mkdir()
            with self.assertRaises(WasmExportError):
                WasmExporter().export(prog, ps, bundle_dir, validate=False, force=False)

    def test_force_true_overwrites(self):
        from matrixai.export import WasmExporter
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            bundle_dir.mkdir()
            (bundle_dir / "stale.txt").write_text("old")
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False, force=True)
            self.assertNotIn("stale.txt", result.files)
            self.assertEqual(set(result.files), {"model.onnx", "wasm_manifest.json", "predict.js"})


# ---------------------------------------------------------------------------
# export_wasm() convenience function
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestExportWasmFunction(unittest.TestCase):
    def test_export_wasm_returns_result(self):
        from matrixai.export import export_wasm
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            result = export_wasm(prog, ps, bundle_dir)
            self.assertEqual(set(result.files), {"model.onnx", "wasm_manifest.json", "predict.js"})
            self.assertTrue(result.equivalence_passed)

    def test_export_wasm_vector_model(self):
        from matrixai.export import export_wasm
        prog, ps = _make_email_agent()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            result = export_wasm(prog, ps, bundle_dir)
            self.assertTrue(result.equivalence_passed)
            self.assertEqual(result.wasm_runtime, "onnxruntime-web")


# ---------------------------------------------------------------------------
# CLI export-wasm
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestCliExportWasm(unittest.TestCase):
    def _write_params(self, ps, path):
        from matrixai.parameters import write_parameter_set
        write_parameter_set(str(path), ps)

    def test_cli_export_wasm_exits_zero(self):
        import subprocess, sys
        prog_obj, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            self._write_params(ps, params_path)
            bundle_dir = Path(td) / "wasm_bundle"
            r = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-wasm", str(_TRANSFORMER_VEC_MXAI),
                 "--params", str(params_path),
                 "--outdir", str(bundle_dir),
                 "--no-validate"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("WASM bundle OK", r.stdout)

    def test_cli_export_wasm_json_flag(self):
        import subprocess, sys
        prog_obj, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            self._write_params(ps, params_path)
            bundle_dir = Path(td) / "wasm_bundle"
            r = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-wasm", str(_TRANSFORMER_VEC_MXAI),
                 "--params", str(params_path),
                 "--outdir", str(bundle_dir),
                 "--no-validate", "--json"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(r.stdout)
            self.assertIn("bundle_dir", data)
            self.assertIn("files", data)

    def test_cli_force_flag(self):
        import subprocess, sys
        prog_obj, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            self._write_params(ps, params_path)
            bundle_dir = Path(td) / "wasm_bundle"
            bundle_dir.mkdir()
            r = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-wasm", str(_TRANSFORMER_VEC_MXAI),
                 "--params", str(params_path),
                 "--outdir", str(bundle_dir),
                 "--no-validate", "--force"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_cli_no_force_existing_dir_fails(self):
        import subprocess, sys
        prog_obj, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            self._write_params(ps, params_path)
            bundle_dir = Path(td) / "wasm_bundle"
            bundle_dir.mkdir()
            r = subprocess.run(
                [sys.executable, "-m", "matrixai",
                 "export-wasm", str(_TRANSFORMER_VEC_MXAI),
                 "--params", str(params_path),
                 "--outdir", str(bundle_dir),
                 "--no-validate"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r.returncode, 0)


# ---------------------------------------------------------------------------
# Hash mismatch raises WasmExportError
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestWasmHashMismatch(unittest.TestCase):
    def test_hash_mismatch_raises(self):
        from matrixai.export import WasmExportError, WasmExporter
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file

        prog1 = parse_file(_TRANSFORMER_SEQ_MXAI)
        prog2 = parse_file(_TRANSFORMER_VEC_MXAI)
        ps_wrong = build_initial_parameter_set(prog2)

        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            with self.assertRaises((WasmExportError, Exception)):
                WasmExporter().export(prog1, ps_wrong, bundle_dir, validate=False)


# ---------------------------------------------------------------------------
# Node.js (onnxruntime-node) validation — same ONNX runtime as onnxruntime-web
# ---------------------------------------------------------------------------

def _node_available() -> bool:
    import shutil
    return shutil.which("node") is not None


def _ort_node_available() -> bool:
    import subprocess, sys
    r = subprocess.run(
        ["node", "-e", "require('onnxruntime-node'); process.exit(0)"],
        capture_output=True, cwd=str(_BASE),
    )
    return r.returncode == 0


_WASM_VALIDATE_JS = _BASE / "scripts" / "wasm_validate.js"


@unittest.skipUnless(_onnx_available(), "onnx not installed")
@unittest.skipUnless(_node_available(), "node not installed")
class TestWasmNodeJsValidation(unittest.TestCase):
    """Validate WASM bundle with onnxruntime-node (same runtime as onnxruntime-web)."""

    def _run_node_validation(self, bundle_dir: Path, result) -> dict:
        import subprocess, json as _json
        is_seq = bool(result.input_shape and result.input_name == "Input" and
                      any(d < 0 or d > 1 for d in result.input_shape))
        # Detect dtype from wasm_manifest
        manifest = _json.loads((bundle_dir / "wasm_manifest.json").read_text())
        input_name = manifest["input_name"]

        # Build sample input
        from matrixai.parser import parse_file
        prog = parse_file(bundle_dir / "model.mxai") if (bundle_dir / "model.mxai").exists() else None

        if "int64" in (bundle_dir / "predict.js").read_text():
            dtype = "int64"
            seq_len = result.input_shape[-1] if result.input_shape else 8
            input_data = [list(range(seq_len))]
        else:
            dtype = "float32"
            vec_size = result.input_shape[-1] if result.input_shape else 8
            input_data = [[0.1 * (i + 1) for i in range(vec_size)]]

        spec = {
            "input_name": input_name,
            "input_dtype": dtype,
            "input_data": input_data,
            "expected_output_shape": result.output_shape,
        }
        spec_path = bundle_dir / "node_spec.json"
        import json as _j
        spec_path.write_text(_j.dumps(spec))

        r = subprocess.run(
            ["node", str(_WASM_VALIDATE_JS), str(bundle_dir / "model.onnx"), str(spec_path)],
            capture_output=True, text=True, cwd=str(_BASE),
        )
        self.assertEqual(r.returncode, 0, f"Node.js validation failed:\n{r.stderr}")
        return _j.loads(r.stdout)

    @unittest.skipUnless(_ort_node_available(), "onnxruntime-node not installed")
    def test_node_validates_vector_model(self):
        prog, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            # Copy mxai for spec detection
            import shutil
            shutil.copy2(str(_TRANSFORMER_VEC_MXAI), str(bundle_dir / "model.mxai"))
            data = self._run_node_validation(bundle_dir, result)
            self.assertTrue(data["ok"])
            self.assertEqual(data["n_samples"], 1)

    @unittest.skipUnless(_ort_node_available(), "onnxruntime-node not installed")
    def test_node_validates_sequence_model(self):
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            import shutil
            shutil.copy2(str(_TRANSFORMER_SEQ_MXAI), str(bundle_dir / "model.mxai"))
            data = self._run_node_validation(bundle_dir, result)
            self.assertTrue(data["ok"])
            self.assertEqual(data["n_samples"], 1)

    @unittest.skipUnless(_ort_node_available(), "onnxruntime-node not installed")
    def test_node_output_shape_matches_manifest(self):
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            import json as _j
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            import shutil
            shutil.copy2(str(_TRANSFORMER_SEQ_MXAI), str(bundle_dir / "model.mxai"))
            data = self._run_node_validation(bundle_dir, result)
            manifest = _j.loads((bundle_dir / "wasm_manifest.json").read_text())
            out_len = len(data["results"][0]["data"])
            self.assertEqual(out_len, manifest["output_shape"][-1])


# ---------------------------------------------------------------------------
# onnxruntime-web WASM backend — real browser runtime in Node.js
# ---------------------------------------------------------------------------

def _ort_web_available() -> bool:
    import subprocess
    r = subprocess.run(
        ["node", "-e", "require('onnxruntime-web'); process.exit(0)"],
        capture_output=True, cwd=str(_BASE),
    )
    return r.returncode == 0


_WASM_VALIDATE_WEB_JS = _BASE / "scripts" / "wasm_validate_web.js"


@unittest.skipUnless(_onnx_available(), "onnx not installed")
@unittest.skipUnless(_node_available(), "node not installed")
class TestWasmOrtWebValidation(unittest.TestCase):
    """Validate WASM bundle with onnxruntime-web WASM execution provider.

    This is the contractually required equivalence check: the same runtime
    (onnxruntime-web + WASM backend) that a browser would use to load the bundle.
    """

    def _run_web_validation(self, bundle_dir: Path, result, input_data, input_dtype: str) -> dict:
        import subprocess, json as _j
        manifest = _j.loads((bundle_dir / "wasm_manifest.json").read_text())
        spec = {
            "input_name": manifest["input_name"],
            "input_dtype": input_dtype,
            "input_data": input_data,
            "expected_output_shape": manifest["output_shape"],
        }
        spec_path = bundle_dir / "web_spec.json"
        spec_path.write_text(_j.dumps(spec))
        r = subprocess.run(
            ["node", str(_WASM_VALIDATE_WEB_JS),
             str(bundle_dir / "model.onnx"), str(spec_path)],
            capture_output=True, text=True, cwd=str(_BASE),
        )
        self.assertEqual(r.returncode, 0,
                         f"onnxruntime-web WASM failed:\n{r.stderr}")
        return _j.loads(r.stdout)

    @unittest.skipUnless(_ort_web_available(), "onnxruntime-web not installed")
    def test_ortWeb_wasm_backend_vector_model(self):
        prog, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            vec_size = result.input_shape[-1]
            data = self._run_web_validation(
                bundle_dir, result,
                input_data=[[0.1 * (i + 1) for i in range(vec_size)]],
                input_dtype="float32",
            )
            self.assertTrue(data["ok"])
            self.assertEqual(data["backend"], "wasm")
            self.assertEqual(data["runtime"], "onnxruntime-web")
            self.assertEqual(data["n_samples"], 1)

    @unittest.skipUnless(_ort_web_available(), "onnxruntime-web not installed")
    def test_ortWeb_wasm_backend_sequence_model(self):
        prog, ps = _make_seq_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            seq_len = result.input_shape[-1]
            data = self._run_web_validation(
                bundle_dir, result,
                input_data=[list(range(seq_len))],
                input_dtype="int64",
            )
            self.assertTrue(data["ok"])
            self.assertEqual(data["backend"], "wasm")
            self.assertEqual(data["n_samples"], 1)

    @unittest.skipUnless(_ort_web_available(), "onnxruntime-web not installed")
    def test_ortWeb_wasm_output_shape_correct(self):
        """Output shape from onnxruntime-web WASM matches wasm_manifest.json."""
        import json as _j
        prog, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            vec_size = result.input_shape[-1]
            data = self._run_web_validation(
                bundle_dir, result,
                input_data=[[0.5] * vec_size],
                input_dtype="float32",
            )
            manifest = _j.loads((bundle_dir / "wasm_manifest.json").read_text())
            self.assertEqual(len(data["results"][0]["data"]),
                             manifest["output_shape"][-1])

    @unittest.skipUnless(_ort_web_available(), "onnxruntime-web not installed")
    def test_ortWeb_wasm_deterministic(self):
        """Two runs with same input produce identical output (WASM is deterministic)."""
        prog, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            vec_size = result.input_shape[-1]
            inp = [[0.3 * (i + 1) for i in range(vec_size)]]
            d1 = self._run_web_validation(bundle_dir, result, inp, "float32")
            d2 = self._run_web_validation(bundle_dir, result, inp, "float32")
            self.assertEqual(d1["results"][0]["data"], d2["results"][0]["data"])

    @unittest.skipUnless(_ort_web_available(), "onnxruntime-web not installed")
    @unittest.skipUnless(_ort_available(), "onnxruntime not installed")
    def test_ortWeb_wasm_matches_python_ort(self):
        """onnxruntime-web WASM output matches onnxruntime Python/CPU within atol=1e-4."""
        import json as _j
        import numpy as np
        import onnxruntime as ort
        prog, ps = _make_vec_transformer()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "bundle"
            from matrixai.export import WasmExporter
            result = WasmExporter().export(prog, ps, bundle_dir, validate=False)
            vec_size = result.input_shape[-1]
            inp_list = [0.1 * (i + 1) for i in range(vec_size)]

            # onnxruntime-web WASM
            web_data = self._run_web_validation(
                bundle_dir, result, [inp_list], "float32"
            )
            web_out = web_data["results"][0]["data"]

            # onnxruntime Python CPU
            sess = ort.InferenceSession(str(bundle_dir / "model.onnx"))
            py_out = sess.run(None, {
                result.input_name: np.array([inp_list], dtype=np.float32)
            })[0].flatten().tolist()

            max_diff = max(abs(a - b) for a, b in zip(web_out, py_out))
            self.assertLessEqual(max_diff, 1e-4,
                f"WASM vs Python ORT diff={max_diff:.2e} exceeds 1e-4")


if __name__ == "__main__":
    unittest.main()
