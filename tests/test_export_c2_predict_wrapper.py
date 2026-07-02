"""EXPORT Modelo descargable — Corte 2: predict.py standalone + artefactos usables.

Cobertura:
- predict.py es importable y NO depende de matrixai (solo numpy + onnxruntime)
- codificación: scalar (norm+clip), scalar01, boolean, one_hot, embedding (vocab/vocab_size)
- errores legibles: clase desconocida, campo faltante, NaN/no-numérico, embedding fuera de rango
- decodificación: classification / binary_classification / regression / raw_vector
- verificación de hash spec vs model.onnx
- CLI --input
- bundle real (con spec) trae predict.py + requirements.txt + example/expected; round-trip OK

La codificación se prueba con un ONNX identidad + output raw_vector: la salida ES el vector
normalizado, así que predict() devuelve exactamente lo que se construye para el grafo.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path

import matrixai.export.predict_template as pt


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    return util.find_spec("onnxruntime") is not None


def _identity_onnx(path: Path, n: int, meta: dict | None = None) -> None:
    import onnx
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, n])
    node = helper.make_node("Identity", ["features"], ["out"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    for k, v in (meta or {}).items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(model, str(path))


def _spec(input_order, fields, output=None, **extra):
    spec = {
        "spec_version": 1,
        "model_hash": "mxai_test",
        "parameter_schema_hash": "params_test",
        "onnx_file": "model.onnx",
        "input_name": "features",
        "input_shape": [-1, len(input_order)],
        "input_order": list(input_order),
        "fields": fields,
        "output": output or {"kind": "raw_vector", "shape": [-1, len(input_order)]},
    }
    spec.update(extra)
    return spec


@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class _ModelHarness(unittest.TestCase):
    def _model(self, spec, *, onnx_meta=None, check_hash=False):
        self.td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.td, True)
        d = Path(self.td)
        (d / "inference_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        _identity_onnx(d / "model.onnx", len(spec["input_order"]), onnx_meta)
        return pt.MatrixAIModel(str(d / "inference_spec.json"), check_hash=check_hash)


class StaticTemplateTest(unittest.TestCase):
    def test_template_has_no_matrixai_imports(self):
        src = Path(pt.__file__).read_text(encoding="utf-8")
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                self.assertNotIn("matrixai", s, f"predict.py must not import matrixai: {s!r}")

    def test_only_numpy_and_onnxruntime_third_party(self):
        src = Path(pt.__file__).read_text(encoding="utf-8")
        self.assertIn("import numpy", src)
        self.assertIn("import onnxruntime", src)


class ScalarEncodingTest(_ModelHarness):
    def test_scalar_normalized(self):
        m = self._model(_spec(["edad"], {"edad": {"encoding": "scalar", "range": [0, 120]}}))
        self.assertAlmostEqual(m.predict({"edad": 60})["values"][0], 0.5, places=6)

    def test_scalar_clipped_and_reported(self):
        m = self._model(_spec(["edad"], {"edad": {"encoding": "scalar", "range": [0, 120]}}))
        out, meta = m.predict({"edad": 240}, return_meta=True)
        self.assertEqual(out["values"][0], 1.0)
        self.assertEqual(meta["clipped"][0]["field"], "edad")

    def test_scalar01_passthrough(self):
        m = self._model(_spec(["x"], {"x": {"encoding": "scalar01"}}))
        self.assertAlmostEqual(m.predict({"x": 0.3})["values"][0], 0.3, places=6)

    def test_boolean_accepts_many_forms(self):
        m = self._model(_spec(["b"], {"b": {"encoding": "scalar01", "type": "boolean"}}))
        for truthy in (True, 1, "si", "true", "yes"):
            self.assertEqual(m.predict({"b": truthy})["values"][0], 1.0)
        for falsy in (False, 0, "no", "false"):
            self.assertEqual(m.predict({"b": falsy})["values"][0], 0.0)

    def test_integer_scalar_rejects_non_integer(self):
        # GEN C3 finding: a scalar with type "integer" must reject a fractional value
        # instead of silently normalizing it (3.7 in [0,10] -> 0.37).
        m = self._model(_spec(["cod"], {"cod": {"encoding": "scalar", "range": [0, 10],
                                                 "type": "integer"}}))
        self.assertAlmostEqual(m.predict({"cod": 3})["values"][0], 0.3, places=6)
        with self.assertRaises(pt.MatrixAIModelError):
            m.predict({"cod": 3.7})

    def test_non_numeric_scalar_raises(self):
        m = self._model(_spec(["x"], {"x": {"encoding": "scalar01"}}))
        with self.assertRaises(pt.MatrixAIModelError):
            m.predict({"x": "abc"})

    def test_nan_scalar_raises(self):
        m = self._model(_spec(["x"], {"x": {"encoding": "scalar01"}}))
        with self.assertRaises(pt.MatrixAIModelError):
            m.predict({"x": float("nan")})

    def test_missing_field_raises(self):
        m = self._model(_spec(["x"], {"x": {"encoding": "scalar01"}}))
        with self.assertRaises(pt.MatrixAIModelError):
            m.predict({})


class OneHotEncodingTest(_ModelHarness):
    def _m(self):
        fields = {"esp": {"encoding": "one_hot", "values": [
            {"raw": "A", "column": "esp__a"}, {"raw": "B", "column": "esp__b"}]}}
        return self._model(_spec(["esp__a", "esp__b"], fields))

    def test_sets_chosen_column(self):
        self.assertEqual(self._m().predict({"esp": "B"})["values"], [0.0, 1.0])

    def test_unknown_category_raises_with_valid_list(self):
        with self.assertRaises(pt.MatrixAIModelError) as ctx:
            self._m().predict({"esp": "Z"})
        self.assertIn("A", str(ctx.exception))


class EmbeddingEncodingTest(_ModelHarness):
    def test_vocab_index(self):
        fields = {"g": {"encoding": "embedding_index", "column": "g",
                        "vocab": ["a", "b", "c"]}}
        m = self._model(_spec(["g"], fields))
        self.assertEqual(m.predict({"g": "b"})["values"], [1.0])

    def test_vocab_unknown_raises(self):
        fields = {"g": {"encoding": "embedding_index", "column": "g", "vocab": ["a", "b"]}}
        with self.assertRaises(pt.MatrixAIModelError):
            self._model(_spec(["g"], fields)).predict({"g": "z"})

    def test_vocab_size_integer_index(self):
        fields = {"code": {"encoding": "embedding_index", "column": "code", "vocab_size": 50}}
        m = self._model(_spec(["code"], fields))
        self.assertEqual(m.predict({"code": 7})["values"], [7.0])

    def test_vocab_size_out_of_range_raises(self):
        fields = {"code": {"encoding": "embedding_index", "column": "code", "vocab_size": 50}}
        with self.assertRaises(pt.MatrixAIModelError):
            self._model(_spec(["code"], fields)).predict({"code": 100})

    def test_vocab_size_non_integer_raises(self):
        fields = {"code": {"encoding": "embedding_index", "column": "code", "vocab_size": 50}}
        with self.assertRaises(pt.MatrixAIModelError):
            self._model(_spec(["code"], fields)).predict({"code": 2.5})


class DecodeTest(_ModelHarness):
    def test_classification_maps_labels(self):
        out = {"kind": "classification", "labels": ["NO", "SI"], "shape": [-1, 2]}
        m = self._model(_spec(["a", "b"], {"a": {"encoding": "scalar01"},
                                           "b": {"encoding": "scalar01"}}, output=out))
        result = m.predict({"a": 0.2, "b": 0.8})
        self.assertEqual(set(result), {"NO", "SI"})
        self.assertAlmostEqual(result["SI"], 0.8, places=6)

    def test_binary_classification_two_way(self):
        out = {"kind": "binary_classification", "labels": ["NO", "SI"], "shape": [-1]}
        m = self._model(_spec(["p"], {"p": {"encoding": "scalar01"}}, output=out))
        result = m.predict({"p": 0.7})
        self.assertAlmostEqual(result["SI"], 0.7, places=6)
        self.assertAlmostEqual(result["NO"], 0.3, places=6)

    def test_regression_scalar(self):
        out = {"kind": "regression", "shape": [-1]}
        m = self._model(_spec(["v"], {"v": {"encoding": "scalar01"}}, output=out))
        self.assertAlmostEqual(m.predict({"v": 0.42}), 0.42, places=6)


class HashCheckTest(_ModelHarness):
    def test_hash_mismatch_raises(self):
        spec = _spec(["x"], {"x": {"encoding": "scalar01"}})
        with self.assertRaises(pt.MatrixAIModelError):
            self._model(spec, onnx_meta={"matrixai_model_hash": "WRONG"}, check_hash=True)

    def test_missing_metadata_raises(self):
        # a model.onnx swapped for one without MatrixAI provenance must not pass silently.
        spec = _spec(["x"], {"x": {"encoding": "scalar01"}})
        with self.assertRaises(pt.MatrixAIModelError):
            self._model(spec, onnx_meta=None, check_hash=True)

    def test_missing_metadata_allowed_when_check_disabled(self):
        spec = _spec(["x"], {"x": {"encoding": "scalar01"}})
        m = self._model(spec, onnx_meta=None, check_hash=False)
        self.assertAlmostEqual(m.predict({"x": 0.3})["values"][0], 0.3, places=6)

    def test_hash_match_ok(self):
        spec = _spec(["x"], {"x": {"encoding": "scalar01"}})
        m = self._model(spec, onnx_meta={"matrixai_model_hash": "mxai_test",
                                         "matrixai_parameter_schema_hash": "params_test"},
                        check_hash=True)
        self.assertIsNotNone(m)


class BatchAndCliTest(_ModelHarness):
    def test_predict_batch(self):
        m = self._model(_spec(["x"], {"x": {"encoding": "scalar01"}},
                              output={"kind": "regression", "shape": [-1]}))
        out = m.predict_batch([{"x": 0.1}, {"x": 0.9}])
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0], 0.1, places=5)
        self.assertAlmostEqual(out[1], 0.9, places=5)

    def test_cli_prints_prediction(self):
        import io
        import contextlib
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(d), True)
        spec = _spec(["x"], {"x": {"encoding": "scalar01"}},
                     output={"kind": "regression", "shape": [-1]})
        (d / "inference_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        _identity_onnx(d / "model.onnx", 1)
        (d / "ex.json").write_text(json.dumps({"x": 0.5}), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = pt._main(["--input", str(d / "ex.json"),
                           "--spec", str(d / "inference_spec.json"), "--no-check-hash"])
        self.assertEqual(rc, 0)
        self.assertAlmostEqual(float(buf.getvalue().strip()), 0.5, places=6)


# ---------------------------------------------------------------------------
# Integration: a real bundle ships the C2 artifacts and the downloaded predict.py
# reproduces expected_output.json.
# ---------------------------------------------------------------------------

@unittest.skipUnless(_onnx_available() and _ort_available(), "onnx/onnxruntime not installed")
class IntegrationC2Test(unittest.TestCase):
    _USABLE = {"model.mxai", "params.best.json", "model.onnx", "model_manifest.json",
               "export_manifest.json", "README.md", "inference_spec.json", "predict.py",
               "requirements.txt", "example_input.json", "expected_output.json"}

    def setUp(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        base = Path(__file__).parent.parent
        self.mxai = base / "examples" / "fall-risk.mxai"
        self.prog = parse_file(self.mxai)
        self.ps = build_initial_parameter_set(self.prog)
        self.td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.td, True)
        self.params = Path(self.td) / "params.json"
        write_parameter_set(str(self.params), self.ps)

    def _bundle(self):
        from matrixai.export import create_edge_bundle
        return create_edge_bundle(self.prog, self.ps, mxai_path=str(self.mxai),
                                  params_path=str(self.params),
                                  outdir=str(Path(self.td) / "b"), validate=False)

    def test_bundle_ships_usable_artifacts(self):
        result = self._bundle()
        self.assertEqual(set(result.files), self._USABLE)

    def test_downloaded_predict_reproduces_expected_output(self):
        result = self._bundle()
        bd = Path(result.bundle_dir)
        spec = util.spec_from_file_location("predict_dl", str(bd / "predict.py"))
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))
        example = json.loads((bd / "example_input.json").read_text())
        expected = json.loads((bd / "expected_output.json").read_text())
        self.assertEqual(model.predict(example), expected)

    def test_usable_bundle_requires_onnxruntime(self):
        # Without onnxruntime the smoke test (expected_output.json) cannot be produced,
        # so a usable bundle must fail loudly rather than ship half-built.
        import unittest.mock
        from matrixai.export import create_edge_bundle, EdgeBundleError
        with unittest.mock.patch("matrixai.export.bundle.ort_available", return_value=False):
            with self.assertRaises(EdgeBundleError):
                create_edge_bundle(self.prog, self.ps, mxai_path=str(self.mxai),
                                   params_path=str(self.params),
                                   outdir=str(Path(self.td) / "b_noort"), validate=False)

    def test_classification_bundle_with_labels(self):
        # email-agent: 3-class softmax; pass labels so it becomes a usable classifier.
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        from matrixai.export import create_edge_bundle
        base = Path(__file__).parent.parent
        mxai = base / "examples" / "email-agent.mxai"
        prog = parse_file(mxai)
        ps = build_initial_parameter_set(prog)
        params = Path(self.td) / "email_params.json"
        write_parameter_set(str(params), ps)
        result = create_edge_bundle(prog, ps, mxai_path=str(mxai), params_path=str(params),
                                    outdir=str(Path(self.td) / "email_b"), validate=False,
                                    labels=["support", "sales", "other"])
        self.assertIsNone(result.inference_spec_skipped_reason)
        bd = Path(result.bundle_dir)
        expected = json.loads((bd / "expected_output.json").read_text())
        self.assertEqual(set(expected), {"support", "sales", "other"})
