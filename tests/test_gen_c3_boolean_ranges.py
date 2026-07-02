"""GENERACIÓN tipos del prompt — Corte 3: Boolean + rangos escalares.

El generador determinista honra `Boolean` y `Scalar en [min,max]`/`Integer[min,max]`
declarados en el prompt, persistiéndolos como metadata (field_types/field_ranges) —
NUNCA en el tipo del VECTOR del .mxai, porque el pipeline de entrenamiento normaliza a
[0,1] y un rango crudo en el VECTOR haría que el verificador rechazara el CSV
normalizado (ver GENERACION_TIPOS_PROMPT_CONTRACT.md, tabla "fuente canónica").
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path

from matrixai.training.dense_generator import DenseNetworkGenerator
from matrixai.parser import parse_text
from matrixai.parameters import build_initial_parameter_set
from matrixai.types import RangeSpec


_PROMPT = """PROYECTO: AbandonoClientesBanca
MODO: clasificación binaria.
ARQUITECTURA: red densa.
FEATURES:
  edad: Scalar en [18, 95]
  saldo_medio: Scalar en [0, 500000]
  tiene_hipoteca: Boolean
  cod_region: Integer[0, 10]
SALIDA: resultado: ProbabilityMap[PERMANECE, ABANDONA]
"""


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None and util.find_spec("onnxruntime") is not None


class BooleanAndRangeMetadataTest(unittest.TestCase):
    def setUp(self):
        self.r = DenseNetworkGenerator().generate(_PROMPT)

    def test_field_ranges_populated(self):
        self.assertEqual(self.r.field_ranges, {
            "edad": (18.0, 95.0),
            "saldo_medio": (0.0, 500000.0),
            "cod_region": (0.0, 10.0),
        })

    def test_field_types_populated(self):
        self.assertEqual(self.r.field_types, {
            "tiene_hipoteca": "boolean",
            "cod_region": "integer",
        })

    def test_mxai_vector_stays_bare_scalar_no_range(self):
        # Critical: the .mxai VECTOR type must NOT carry the raw range/type, or the
        # training verifier would reject the [0,1]-normalized CSV against it.
        prog = parse_text(self.r.mxai_text)
        for name in ("edad", "saldo_medio", "tiene_hipoteca", "cod_region"):
            t = prog.vectors[0].field_types[name]
            self.assertEqual(t.name, "Scalar", name)
            self.assertIsNone(t.range, name)

    def test_normalized_csv_value_passes_verifier_range_check(self):
        # regression guard for the trap: if the range WERE written into the VECTOR
        # type, a normalized value like 0.5 would fail RangeSpec.contains([18,95]).
        prog = parse_text(self.r.mxai_text)
        t = prog.vectors[0].field_types["edad"]
        self.assertIsNone(t.range)  # no range means no verifier rejection possible
        # sanity: an explicit raw range WOULD reject a normalized value (the trap)
        self.assertFalse(RangeSpec(18.0, 95.0).contains(0.5))

    def test_parses_and_params_build(self):
        prog = parse_text(self.r.mxai_text)
        ps = build_initial_parameter_set(prog)
        self.assertTrue(ps.parameters)

    def test_input_dim_unaffected_by_boolean_or_range(self):
        # boolean/ranged scalars are NOT expanded (only categoricals are) — one column each
        self.assertEqual(self.r.input_dim, 4)


class DispatchSurfacesC3MetadataTest(unittest.TestCase):
    def test_analyze_playground_request_surfaces_field_ranges_and_types(self):
        from matrixai.playground import analyze_playground_request
        res = analyze_playground_request({"mode": "prompt", "prompt": _PROMPT, "use_llm": False})
        self.assertTrue(res.get("ok"), res.get("error"))
        self.assertEqual(res.get("field_ranges"), {
            "edad": (18.0, 95.0), "saldo_medio": (0.0, 500000.0), "cod_region": (0.0, 10.0),
        })
        self.assertEqual(res.get("field_types"), {"tiene_hipoteca": "boolean", "cod_region": "integer"})


class MultilineNoBracketRegressionTest(unittest.TestCase):
    """Regression: a bracket-less type (Boolean) on one line used to eat the newline
    boundary needed by the NEXT field, silently dropping it (found in C3 audit)."""

    def test_boolean_does_not_swallow_next_field(self):
        prompt = (
            "FEATURES:\n"
            "  activo: Boolean\n"
            "  edad: Integer[0, 120]\n"
        )
        r = DenseNetworkGenerator().generate(
            "clasificar red densa\n" + prompt + "SALIDA: y: ProbabilityMap[A, B]"
        )
        self.assertEqual(r.field_types, {"activo": "boolean", "edad": "integer"})
        self.assertEqual(r.field_ranges, {"edad": (0.0, 120.0)})

    def test_plain_scalar_does_not_swallow_next_field(self):
        prompt = (
            "FEATURES:\n"
            "  saldo: Scalar\n"
            "  edad: Integer[0, 120]\n"
        )
        r = DenseNetworkGenerator().generate(
            "clasificar red densa\n" + prompt + "SALIDA: y: ProbabilityMap[A, B]"
        )
        self.assertEqual(r.field_ranges, {"edad": (0.0, 120.0)})


class MixedWithCategoricalTest(unittest.TestCase):
    def test_all_four_kinds_together(self):
        prompt = (
            "clasificar red densa\nFEATURES:\n"
            "  edad: Scalar en [18, 95]\n"
            "  tiene_hipoteca: Boolean\n"
            "  cod_region: Integer[0, 10]\n"
            "  segmento: Categorical[PARTICULAR, PYME, PREMIUM]\n"
            "SALIDA: y: ProbabilityMap[A, B]"
        )
        r = DenseNetworkGenerator().generate(prompt)
        self.assertEqual(r.field_ranges, {"edad": (18.0, 95.0), "cod_region": (0.0, 10.0)})
        self.assertEqual(r.field_types, {"tiene_hipoteca": "boolean", "cod_region": "integer"})
        self.assertEqual(r.field_categories, {"segmento": ["PARTICULAR", "PYME", "PREMIUM"]})
        prog = parse_text(r.mxai_text)
        self.assertIn("segmento__particular", prog.vectors[0].fields)
        self.assertIn("edad", prog.vectors[0].fields)
        self.assertIn("tiene_hipoteca", prog.vectors[0].fields)


@unittest.skipUnless(_onnx_available(), "onnx/onnxruntime required")
class ExportAcceptsBooleanAndRangeTest(unittest.TestCase):
    """End-to-end: field_ranges/field_types from the generator flow straight into a
    self-usable bundle whose predict.py normalizes ranges and accepts true/false."""

    def test_downloaded_predict_normalizes_and_accepts_boolean(self):
        from matrixai.parameters import write_parameter_set
        from matrixai.export import create_edge_bundle

        r = DenseNetworkGenerator().generate(_PROMPT)
        prog = parse_text(r.mxai_text)
        ps = build_initial_parameter_set(prog)
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(td), True)
        (td / "m.mxai").write_text(r.mxai_text, encoding="utf-8")
        write_parameter_set(str(td / "p.json"), ps)

        result = create_edge_bundle(
            prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
            outdir=str(td / "b"), validate=False,
            field_ranges=r.field_ranges,
            field_types=r.field_types,
            labels=["PERMANECE", "ABANDONA"],
        )
        self.assertIsNone(result.inference_spec_skipped_reason)
        bd = Path(result.bundle_dir)
        spec = json.loads((bd / "inference_spec.json").read_text())
        self.assertEqual(spec["fields"]["edad"], {"encoding": "scalar", "range": [18, 95]})
        self.assertEqual(spec["fields"]["tiene_hipoteca"]["type"], "boolean")
        self.assertEqual(spec["fields"]["cod_region"]["type"], "integer")

        m = util.spec_from_file_location("pred_dl", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))
        out = model.predict({"edad": 56, "saldo_medio": 12000, "tiene_hipoteca": "si",
                             "cod_region": 3})
        self.assertEqual(set(out), {"PERMANECE", "ABANDONA"})
