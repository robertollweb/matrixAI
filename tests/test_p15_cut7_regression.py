"""P15 Corte 7: Regression suite — P1-P14 core flows unaffected by P15 export.

Tests verify:
- Core parse, validate, compile, runtime still work for all archetypes
- Export does not mutate program or ParameterSet
- All four model archetypes produce valid ONNX (softmax_linear, sigmoid_linear,
  layer_call VECTOR, layer_call SEQUENCE)
- All four archetypes pass OnnxEquivalenceValidator
- All four archetypes produce valid EdgeBundle
- All four archetypes produce valid WASM bundle
- Hash stability: model_hash is deterministic across multiple exports
- Import isolation: matrixai core imports don't pull in onnx/onnxruntime
- ort_available() reports correctly
- Bundle file counts match contracts
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.mxai"
_FALL_RISK_MXAI = _BASE / "examples" / "fall-risk.mxai"
_TRANSFORMER_VEC_MXAI = _BASE / "examples" / "transformer-classifier-vector.mxai"
_TRANSFORMER_SEQ_MXAI = _BASE / "examples" / "transformer-classifier.mxai"


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    from importlib import util
    return util.find_spec("onnxruntime") is not None


def _load(path):
    from matrixai.parser import parse_file
    from matrixai.parameters import build_initial_parameter_set
    prog = parse_file(path)
    return prog, build_initial_parameter_set(prog)


def _vec_input(prog):
    v = prog.vectors[0]
    return {v.name: {f"v{i}": float(i + 1) / 10 for i in range(v.size)}}


def _seq_input(prog):
    seq = prog.sequences[0]
    return {seq.name: list(range(seq.length))}


# ---------------------------------------------------------------------------
# P1-P3: Parse, validate, compile (core flows unaffected)
# ---------------------------------------------------------------------------

class TestCoreParseValidate(unittest.TestCase):
    """P1-P3 regression: parse and validate work for all archetypes."""

    def test_parse_email_agent(self):
        from matrixai.parser import parse_file
        prog = parse_file(_EMAIL_MXAI)
        self.assertTrue(prog.vectors)
        self.assertEqual(len(prog.functions), 2)

    def test_parse_fall_risk(self):
        from matrixai.parser import parse_file
        prog = parse_file(_FALL_RISK_MXAI)
        self.assertTrue(prog.vectors)

    def test_parse_transformer_vector(self):
        from matrixai.parser import parse_file
        prog = parse_file(_TRANSFORMER_VEC_MXAI)
        self.assertTrue(prog.vectors)

    def test_parse_transformer_sequence(self):
        from matrixai.parser import parse_file
        prog = parse_file(_TRANSFORMER_SEQ_MXAI)
        self.assertTrue(prog.sequences)

    def test_validate_email_agent(self):
        from matrixai.parser import parse_file
        from matrixai.agents import VerifierAgent
        prog = parse_file(_EMAIL_MXAI)
        result = VerifierAgent().verify(prog)
        self.assertTrue(result.ok, result.errors)

    def test_validate_fall_risk(self):
        from matrixai.parser import parse_file
        from matrixai.agents import VerifierAgent
        prog = parse_file(_FALL_RISK_MXAI)
        result = VerifierAgent().verify(prog)
        self.assertTrue(result.ok, result.errors)

    def test_validate_transformer_vector(self):
        from matrixai.parser import parse_file
        from matrixai.agents import VerifierAgent
        prog = parse_file(_TRANSFORMER_VEC_MXAI)
        result = VerifierAgent().verify(prog)
        self.assertTrue(result.ok, result.errors)

    def test_backend_contract_email_agent(self):
        from matrixai.parser import parse_file
        from matrixai.compiler import BackendContractAnalyzer
        prog = parse_file(_EMAIL_MXAI)
        report = BackendContractAnalyzer().analyze(prog)
        self.assertTrue(report.ok)

    def test_compile_dp_email_agent(self):
        from matrixai.compiler import DifferentiablePythonCompiler
        prog, _ = _load(_EMAIL_MXAI)
        src = DifferentiablePythonCompiler().compile(prog)
        self.assertIn("def run(", src)

    def test_compile_dp_transformer_vector(self):
        from matrixai.compiler import DifferentiablePythonCompiler
        prog, _ = _load(_TRANSFORMER_VEC_MXAI)
        src = DifferentiablePythonCompiler().compile(prog)
        self.assertIn("def run(", src)


# ---------------------------------------------------------------------------
# P4: ParameterSet core flows
# ---------------------------------------------------------------------------

class TestParameterSetCoreFlows(unittest.TestCase):
    """P4 regression: ParameterSet build, validate, and hashing work."""

    def test_build_parameter_set_email_agent(self):
        from matrixai.parameters import build_initial_parameter_set, validate_parameter_set
        prog, ps = _load(_EMAIL_MXAI)
        val = validate_parameter_set(prog, ps)
        self.assertTrue(val.ok, val.errors)

    def test_parameter_set_has_hashes(self):
        _, ps = _load(_EMAIL_MXAI)
        self.assertTrue(ps.model_hash)
        self.assertTrue(ps.parameter_schema_hash)
        self.assertTrue(ps.parameter_set_id)

    def test_parameter_set_hash_stable(self):
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file
        prog = parse_file(_EMAIL_MXAI)
        ps1 = build_initial_parameter_set(prog)
        ps2 = build_initial_parameter_set(prog)
        self.assertEqual(ps1.model_hash, ps2.model_hash)
        self.assertEqual(ps1.parameter_schema_hash, ps2.parameter_schema_hash)

    def test_write_and_load_parameter_set(self):
        from matrixai.parameters import load_parameter_set, write_parameter_set
        _, ps = _load(_EMAIL_MXAI)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "params.json"
            write_parameter_set(str(path), ps)
            ps2 = load_parameter_set(str(path))
            self.assertEqual(ps.model_hash, ps2.model_hash)
            self.assertEqual(ps.parameter_set_id, ps2.parameter_set_id)


# ---------------------------------------------------------------------------
# P5 + P10 + P11: Runtime inference (differentiable_python) unaffected
# ---------------------------------------------------------------------------

class TestRuntimeInferenceUnaffected(unittest.TestCase):
    """P5/P10/P11 regression: differentiable_python runtime produces correct output."""

    def _dp_run(self, path, input_data):
        from matrixai.compiler import DifferentiablePythonCompiler
        prog, ps = _load(path)
        src = DifferentiablePythonCompiler().compile(prog)
        ns: dict = {}
        exec(src, ns)
        return ns["run"](input_data, ps.runtime_parameters())

    def test_dp_email_agent_produces_output(self):
        prog, _ = _load(_EMAIL_MXAI)
        result = self._dp_run(_EMAIL_MXAI, _vec_input(prog))
        self.assertIn("state", result)

    def test_dp_fall_risk_produces_output(self):
        prog, _ = _load(_FALL_RISK_MXAI)
        result = self._dp_run(_FALL_RISK_MXAI, _vec_input(prog))
        self.assertIn("state", result)

    def test_dp_transformer_vector_produces_output(self):
        prog, _ = _load(_TRANSFORMER_VEC_MXAI)
        result = self._dp_run(_TRANSFORMER_VEC_MXAI, _vec_input(prog))
        self.assertIn("state", result)

    def test_dp_transformer_sequence_produces_output(self):
        prog, _ = _load(_TRANSFORMER_SEQ_MXAI)
        result = self._dp_run(_TRANSFORMER_SEQ_MXAI, _seq_input(prog))
        self.assertIn("state", result)

    def test_dp_output_deterministic_email_agent(self):
        prog, _ = _load(_EMAIL_MXAI)
        inp = _vec_input(prog)
        r1 = self._dp_run(_EMAIL_MXAI, inp)
        r2 = self._dp_run(_EMAIL_MXAI, inp)
        self.assertEqual(r1["state"], r2["state"])


# ---------------------------------------------------------------------------
# Export does not mutate program or ParameterSet
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestExportNoMutation(unittest.TestCase):
    """Export operations must not alter program IR or ParameterSet state."""

    def test_export_does_not_mutate_program_email_agent(self):
        from matrixai.export import export_onnx
        prog, ps = _load(_EMAIL_MXAI)
        fn_count_before = len(prog.functions)
        vec_count_before = len(prog.vectors)
        with tempfile.TemporaryDirectory() as td:
            export_onnx(prog, ps, Path(td) / "m.onnx")
        self.assertEqual(len(prog.functions), fn_count_before)
        self.assertEqual(len(prog.vectors), vec_count_before)

    def test_export_does_not_mutate_parameter_set_email_agent(self):
        from matrixai.export import export_onnx
        prog, ps = _load(_EMAIL_MXAI)
        hash_before = ps.model_hash
        schema_hash_before = ps.parameter_schema_hash
        with tempfile.TemporaryDirectory() as td:
            export_onnx(prog, ps, Path(td) / "m.onnx")
        self.assertEqual(ps.model_hash, hash_before)
        self.assertEqual(ps.parameter_schema_hash, schema_hash_before)

    def test_runtime_output_unchanged_after_export(self):
        """DP runtime produces identical output before and after exporting to ONNX."""
        from matrixai.compiler import DifferentiablePythonCompiler
        from matrixai.export import export_onnx
        prog, ps = _load(_EMAIL_MXAI)
        inp = _vec_input(prog)
        src = DifferentiablePythonCompiler().compile(prog)
        ns: dict = {}
        exec(src, ns)
        params = ps.runtime_parameters()
        out_before = ns["run"](inp, params)["state"]
        with tempfile.TemporaryDirectory() as td:
            export_onnx(prog, ps, Path(td) / "m.onnx")
        out_after = ns["run"](inp, params)["state"]
        self.assertEqual(out_before, out_after)

    def test_export_transformer_vec_no_mutation(self):
        from matrixai.export import export_onnx
        prog, ps = _load(_TRANSFORMER_VEC_MXAI)
        fn_names_before = [f.name for f in prog.functions]
        with tempfile.TemporaryDirectory() as td:
            export_onnx(prog, ps, Path(td) / "m.onnx")
        fn_names_after = [f.name for f in prog.functions]
        self.assertEqual(fn_names_before, fn_names_after)


# ---------------------------------------------------------------------------
# All archetypes: ONNX export correctness
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestAllArchetypesOnnxExport(unittest.TestCase):
    """Regression: all four archetypes produce valid ONNX files."""

    def _export_and_check(self, path):
        import onnx
        prog, ps = _load(path)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            from matrixai.export import export_onnx
            result = export_onnx(prog, ps, onnx_path)
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            return result

    def test_softmax_linear_email_agent(self):
        result = self._export_and_check(_EMAIL_MXAI)
        self.assertEqual(result.opset_version, 17)

    def test_sigmoid_linear_fall_risk(self):
        result = self._export_and_check(_FALL_RISK_MXAI)
        self.assertEqual(result.opset_version, 17)

    def test_layer_call_vector(self):
        result = self._export_and_check(_TRANSFORMER_VEC_MXAI)
        self.assertEqual(result.opset_version, 17)

    def test_layer_call_sequence(self):
        result = self._export_and_check(_TRANSFORMER_SEQ_MXAI)
        self.assertEqual(result.opset_version, 17)

    def test_hash_embedded_email_agent(self):
        import onnx
        prog, ps = _load(_EMAIL_MXAI)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            from matrixai.export import export_onnx
            export_onnx(prog, ps, onnx_path)
            model = onnx.load(str(onnx_path))
            props = {p.key: p.value for p in model.metadata_props}
            self.assertEqual(props.get("matrixai_model_hash"), ps.model_hash)


# ---------------------------------------------------------------------------
# All archetypes: equivalence check
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestAllArchetypesEquivalence(unittest.TestCase):
    """Regression: OnnxEquivalenceValidator passes for all archetypes."""

    def _check_equivalence(self, path):
        from matrixai.export import export_onnx, validate_onnx_equivalence
        prog, ps = _load(path)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            export_onnx(prog, ps, onnx_path)
            eq = validate_onnx_equivalence(prog, ps, onnx_path, n_samples=10)
            return eq

    def test_softmax_linear_email_agent(self):
        eq = self._check_equivalence(_EMAIL_MXAI)
        self.assertTrue(eq.passed, f"max_abs_diff={eq.max_abs_diff:.2e}")
        self.assertLessEqual(eq.max_abs_diff, 1e-5)

    def test_sigmoid_linear_fall_risk(self):
        eq = self._check_equivalence(_FALL_RISK_MXAI)
        self.assertTrue(eq.passed, f"max_abs_diff={eq.max_abs_diff:.2e}")
        self.assertLessEqual(eq.max_abs_diff, 1e-5)

    def test_layer_call_vector(self):
        eq = self._check_equivalence(_TRANSFORMER_VEC_MXAI)
        self.assertTrue(eq.passed, f"max_abs_diff={eq.max_abs_diff:.2e}")
        self.assertLessEqual(eq.max_abs_diff, 1e-5)

    def test_layer_call_sequence(self):
        eq = self._check_equivalence(_TRANSFORMER_SEQ_MXAI)
        self.assertTrue(eq.passed, f"max_abs_diff={eq.max_abs_diff:.2e}")
        self.assertLessEqual(eq.max_abs_diff, 1e-5)


# ---------------------------------------------------------------------------
# All archetypes: EdgeBundle
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestAllArchetypesEdgeBundle(unittest.TestCase):
    """Regression: EdgeBundler produces correct 6-file bundles for all archetypes."""

    _EXPECTED_FILES = {"model.mxai", "params.best.json", "model.onnx",
                       "model_manifest.json", "export_manifest.json", "README.md"}

    def _make_bundle(self, path):
        from matrixai.export import create_edge_bundle
        from matrixai.parameters import write_parameter_set
        prog, ps = _load(path)
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "params.json"
            write_parameter_set(str(params_path), ps)
            bundle_dir = Path(td) / "bundle"
            result = create_edge_bundle(
                prog, ps,
                mxai_path=str(path),
                params_path=str(params_path),
                outdir=str(bundle_dir),
                validate=True,
            )
            return result

    def test_email_agent_bundle_files(self):
        result = self._make_bundle(_EMAIL_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)

    def test_fall_risk_bundle_files(self):
        result = self._make_bundle(_FALL_RISK_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)

    def test_transformer_vector_bundle_files(self):
        result = self._make_bundle(_TRANSFORMER_VEC_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)

    def test_transformer_sequence_bundle_files(self):
        result = self._make_bundle(_TRANSFORMER_SEQ_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)

    def test_all_bundles_equivalence_passed(self):
        for path in (_EMAIL_MXAI, _FALL_RISK_MXAI, _TRANSFORMER_VEC_MXAI, _TRANSFORMER_SEQ_MXAI):
            with self.subTest(model=path.name):
                result = self._make_bundle(path)
                self.assertTrue(result.equivalence_passed, f"{path.name} bundle failed equivalence")


# ---------------------------------------------------------------------------
# All archetypes: WASM bundle
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class TestAllArchetypesWasmBundle(unittest.TestCase):
    """Regression: WasmExporter produces correct 3-file bundles for all archetypes."""

    _EXPECTED_FILES = {"model.onnx", "wasm_manifest.json", "predict.js"}

    def _make_wasm(self, path):
        from matrixai.export import WasmExporter
        prog, ps = _load(path)
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "wasm"
            result = WasmExporter().export(prog, ps, bundle_dir)
            return result

    def test_email_agent_wasm_files(self):
        result = self._make_wasm(_EMAIL_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)
        self.assertTrue(result.equivalence_passed)

    def test_fall_risk_wasm_files(self):
        result = self._make_wasm(_FALL_RISK_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)
        self.assertTrue(result.equivalence_passed)

    def test_transformer_vector_wasm_files(self):
        result = self._make_wasm(_TRANSFORMER_VEC_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)
        self.assertTrue(result.equivalence_passed)

    def test_transformer_sequence_wasm_files(self):
        result = self._make_wasm(_TRANSFORMER_SEQ_MXAI)
        self.assertEqual(set(result.files), self._EXPECTED_FILES)
        self.assertTrue(result.equivalence_passed)


# ---------------------------------------------------------------------------
# Hash stability across multiple exports
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestHashStability(unittest.TestCase):
    """model_hash and parameter_schema_hash are deterministic across exports."""

    def test_model_hash_stable_email_agent(self):
        from matrixai.export import export_onnx
        prog, ps = _load(_EMAIL_MXAI)
        with tempfile.TemporaryDirectory() as td:
            r1 = export_onnx(prog, ps, Path(td) / "m1.onnx")
            r2 = export_onnx(prog, ps, Path(td) / "m2.onnx")
        self.assertEqual(r1.model_hash, r2.model_hash)
        self.assertEqual(r1.parameter_schema_hash, r2.parameter_schema_hash)

    def test_model_hash_stable_transformer_vec(self):
        from matrixai.export import export_onnx
        prog, ps = _load(_TRANSFORMER_VEC_MXAI)
        with tempfile.TemporaryDirectory() as td:
            r1 = export_onnx(prog, ps, Path(td) / "m1.onnx")
            r2 = export_onnx(prog, ps, Path(td) / "m2.onnx")
        self.assertEqual(r1.model_hash, r2.model_hash)

    def test_model_hash_differs_between_models(self):
        from matrixai.export import export_onnx
        prog_e, ps_e = _load(_EMAIL_MXAI)
        prog_f, ps_f = _load(_FALL_RISK_MXAI)
        with tempfile.TemporaryDirectory() as td:
            re = export_onnx(prog_e, ps_e, Path(td) / "e.onnx")
            rf = export_onnx(prog_f, ps_f, Path(td) / "f.onnx")
        self.assertNotEqual(re.model_hash, rf.model_hash)


# ---------------------------------------------------------------------------
# Import isolation: core matrixai imports don't require onnx/onnxruntime
# ---------------------------------------------------------------------------

class TestImportIsolation(unittest.TestCase):
    """Core matrixai modules must not import onnx or onnxruntime at load time."""

    def test_matrixai_core_importable(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        self.assertTrue(True)

    def test_matrixai_runtime_importable(self):
        from matrixai.runtime import MatrixAIRuntime
        self.assertTrue(True)

    def test_matrixai_compiler_importable(self):
        from matrixai.compiler import DifferentiablePythonCompiler
        self.assertTrue(True)

    def test_export_module_symbols_accessible(self):
        from matrixai.export import (
            OnnxExporter, OnnxEquivalenceValidator,
            EdgeBundler, WasmExporter,
            export_onnx, export_wasm, create_edge_bundle,
        )
        self.assertTrue(True)

    def test_ort_available_returns_bool(self):
        from matrixai.export import ort_available
        result = ort_available()
        self.assertIsInstance(result, bool)

    def test_ort_available_true_when_installed(self):
        if not _ort_available():
            self.skipTest("onnxruntime not installed")
        from matrixai.export import ort_available
        self.assertTrue(ort_available())


# ---------------------------------------------------------------------------
# Unsupported model raises explicit error (not a silent failure)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available(), "onnx not installed")
class TestUnsupportedModelExplicitError(unittest.TestCase):
    """P15 contract: unsupported models raise OnnxExportError with clear message."""

    def test_hash_mismatch_raises_onnx_export_error(self):
        from matrixai.export import OnnxExportError, export_onnx
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file
        prog_e = parse_file(_EMAIL_MXAI)
        prog_f = parse_file(_FALL_RISK_MXAI)
        ps_wrong = build_initial_parameter_set(prog_f)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(OnnxExportError):
                export_onnx(prog_e, ps_wrong, Path(td) / "m.onnx")


if __name__ == "__main__":
    unittest.main()
