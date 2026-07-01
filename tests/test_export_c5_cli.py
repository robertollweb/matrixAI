"""EXPORT Modelo descargable — Corte 5: CLI export-bundle + sidecar de metadata.

`matrixai export-bundle` acepta `--inference-metadata sidecar.json` (field_ranges,
field_categories, field_types, labels, example_input) para producir un bundle
auto-usable desde línea de comandos, sin pasar por el Studio. Los labels también
fluyen del ProbabilityMap del .mxai. La verificación end-to-end en venv limpio
(predict.py con solo numpy+onnxruntime reproduce expected_output.json) se hace a
mano en el servidor (ver contrato); aquí se cubre el camino del CLI.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from importlib import util
from pathlib import Path

_BASE = Path(__file__).parent.parent

_SCALAR_MXAI = """PROJECT ScalarRange
VECTOR Patient[3]
  edad: Scalar
  imc: Scalar
  tension: Scalar
END
NETWORK Net
  INPUT Patient
  LAYER Dense units=6 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[BAJO, ALTO]
END
GRAPH
  Patient -> Net
END
"""


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None and util.find_spec("onnxruntime") is not None


def _run_cli(*cli_args: str):
    proc = subprocess.run(
        [sys.executable, "-m", "matrixai", "export-bundle", *cli_args],
        capture_output=True, text=True, cwd=str(_BASE),
    )
    return proc.returncode, proc.stdout, proc.stderr


@unittest.skipUnless(_onnx_available(), "onnx/onnxruntime not installed")
class ExportBundleCliTest(unittest.TestCase):
    def setUp(self):
        from matrixai.parser import parse_text
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        self.td = Path(tempfile.mkdtemp())
        self.mxai = self.td / "m.mxai"
        self.mxai.write_text(_SCALAR_MXAI, encoding="utf-8")
        ps = build_initial_parameter_set(parse_text(_SCALAR_MXAI))
        self.params = self.td / "p.json"
        write_parameter_set(str(self.params), ps)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def _files(self, outdir: Path) -> set[str]:
        return {p.name for p in outdir.iterdir() if p.is_file()}

    def test_sidecar_produces_usable_bundle(self):
        meta = self.td / "meta.json"
        meta.write_text(json.dumps({
            "field_ranges": {"edad": [0, 120], "imc": [10, 70], "tension": [0, 250]},
            "example_input": {"edad": 60, "imc": 40, "tension": 125},
        }), encoding="utf-8")
        out = self.td / "bundle"
        rc, stdout, stderr = _run_cli(
            str(self.mxai), "--params", str(self.params), "--outdir", str(out),
            "--no-validate", "--inference-metadata", str(meta),
        )
        self.assertEqual(rc, 0, stderr)
        self.assertIn("Self-usable: yes", stdout)
        files = self._files(out)
        self.assertIn("predict.py", files)
        self.assertIn("inference_spec.json", files)
        self.assertIn("expected_output.json", files)
        # the sidecar ranges made it into the spec
        spec = json.loads((out / "inference_spec.json").read_text())
        self.assertEqual(spec["fields"]["edad"], {"encoding": "scalar", "range": [0, 120]})
        # the sidecar example_input is the bundle's example
        self.assertEqual(json.loads((out / "example_input.json").read_text()),
                         {"edad": 60, "imc": 40, "tension": 125})

    def test_labels_flow_from_mxai_without_sidecar(self):
        # No sidecar at all: ProbabilityMap[BAJO, ALTO] in the .mxai still yields a
        # usable, labelled bundle (scalars fall back to scalar01).
        out = self.td / "bundle_nosidecar"
        rc, stdout, stderr = _run_cli(
            str(self.mxai), "--params", str(self.params), "--outdir", str(out), "--no-validate",
        )
        self.assertEqual(rc, 0, stderr)
        self.assertIn("Self-usable: yes", stdout)
        spec = json.loads((out / "inference_spec.json").read_text())
        self.assertEqual(spec["output"]["labels"], ["BAJO", "ALTO"])
        self.assertEqual(spec["fields"]["edad"], {"encoding": "scalar01"})

    def test_malformed_json_sidecar_errors_cleanly(self):
        bad = self.td / "bad.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        out = self.td / "bundle_bad"
        rc, stdout, stderr = _run_cli(
            str(self.mxai), "--params", str(self.params), "--outdir", str(out),
            "--no-validate", "--inference-metadata", str(bad),
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("Bundle error", stderr)
        self.assertFalse(out.exists(), "no bundle should be written on a bad sidecar")

    def test_semantically_invalid_sidecar_errors(self):
        # A syntactically-valid JSON that is semantically wrong must NOT degrade
        # silently into a partial/usable bundle (it would normalize the input wrong).
        payload = {"field_ranges": {"edad": [120, 0]}}  # min >= max
        meta = self.td / "sembad.json"
        meta.write_text(json.dumps(payload), encoding="utf-8")
        out = self.td / "bundle_sembad"
        rc, stdout, stderr = _run_cli(
            str(self.mxai), "--params", str(self.params), "--outdir", str(out),
            "--no-validate", "--inference-metadata", str(meta),
        )
        self.assertNotEqual(rc, 0, stdout)
        self.assertIn("Bundle error", stderr)
        self.assertFalse(out.exists())

    def test_json_output_reports_skip_reason_field(self):
        out = self.td / "bundle_json"
        rc, stdout, stderr = _run_cli(
            str(self.mxai), "--params", str(self.params), "--outdir", str(out),
            "--no-validate", "--json",
        )
        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        # a labelled scalar model is self-usable -> reason is null in the JSON result
        self.assertIn("inference_spec_skipped_reason", payload)
        self.assertIsNone(payload["inference_spec_skipped_reason"])


class LoadInferenceMetadataTest(unittest.TestCase):
    """Strict validation of the sidecar (fast, no subprocess)."""

    def _load(self, payload):
        from matrixai.cli import _load_inference_metadata
        p = Path(tempfile.mktemp(suffix=".json"))
        p.write_text(json.dumps(payload), encoding="utf-8")
        try:
            return _load_inference_metadata(str(p))
        finally:
            p.unlink(missing_ok=True)

    def test_valid_sidecar_loads(self):
        out = self._load({
            "field_ranges": {"edad": [0, 120]},
            "field_categories": {"color": ["red", "green"]},
            "field_types": {"edad": "integer"},
            "labels": ["NO", "SI"],
            "example_input": {"edad": 60},
        })
        self.assertEqual(out["field_ranges"], {"edad": (0.0, 120.0)})
        self.assertEqual(out["field_categories"], {"color": ["red", "green"]})
        self.assertEqual(out["field_types"], {"edad": "integer"})

    def test_range_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self._load({"field_ranges": {"edad": [0]}})

    def test_range_min_not_less_than_max_raises(self):
        with self.assertRaises(ValueError):
            self._load({"field_ranges": {"edad": [120, 0]}})

    def test_range_non_finite_raises(self):
        with self.assertRaises(ValueError):
            self._load({"field_ranges": {"edad": [0, float("inf")]}})

    def test_category_string_not_list_raises(self):
        # {"color": "red"} must NOT become ['r','e','d']
        with self.assertRaises(ValueError):
            self._load({"field_categories": {"color": "red"}})

    def test_empty_category_list_raises(self):
        with self.assertRaises(ValueError):
            self._load({"field_categories": {"color": []}})

    def test_unknown_field_type_raises(self):
        with self.assertRaises(ValueError):
            self._load({"field_types": {"edad": "banana"}})

    def test_non_string_labels_raise(self):
        with self.assertRaises(ValueError):
            self._load({"labels": [1, 2]})

    def test_wrong_top_level_type_raises(self):
        with self.assertRaises(ValueError):
            self._load(["not", "an", "object"])
