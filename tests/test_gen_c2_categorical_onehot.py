"""GENERACIÓN tipos del prompt — Corte 2: categórica declarada → one-hot.

El generador determinista materializa `Categorical[A, B, C]` (baja cardinalidad) como
columnas one-hot reusando expand_categoricals (reescribe .mxai + training_text), persiste
`field_categories`, y el dispatch lo surfacea. Sin LLM ni editor de esquema.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path

from matrixai.training.dense_generator import DenseNetworkGenerator, _ONEHOT_MAX
from matrixai.training.categorical import _build_group_names
from matrixai.parser import parse_text
from matrixai.parameters import build_initial_parameter_set


_PROMPT = """PROYECTO: EstadoMaquina
MODO: clasificación multiclase.
ARQUITECTURA: red densa moderada.
FEATURES:
  temperatura: Scalar en [0, 150]
  vibracion: Scalar en [0, 50]
  tipo_maquina: Categorical[TORNO, FRESADORA, PRENSA, COMPRESOR]
SALIDA: estado: ProbabilityMap[OK, DESGASTE, FALLO]
"""


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None and util.find_spec("onnxruntime") is not None


class DeterministicOneHotTest(unittest.TestCase):
    def setUp(self):
        self.r = DenseNetworkGenerator().generate(_PROMPT)

    def test_field_categories_populated(self):
        self.assertEqual(self.r.field_categories,
                         {"tipo_maquina": ["TORNO", "FRESADORA", "PRENSA", "COMPRESOR"]})

    def test_vector_has_onehot_columns(self):
        cols = _build_group_names("tipo_maquina", ["TORNO", "FRESADORA", "PRENSA", "COMPRESOR"])
        prog = parse_text(self.r.mxai_text)
        fields = prog.vectors[0].fields
        for c in cols:
            self.assertIn(c, fields)
        self.assertNotIn("tipo_maquina", fields)  # original column replaced
        self.assertEqual(prog.vectors[0].size, len(fields))

    def test_input_dim_matches_expanded(self):
        # 2 scalars + 4 one-hot columns
        self.assertEqual(self.r.input_dim, 6)

    def test_training_text_uses_expanded_columns(self):
        self.assertIn("tipo_maquina__torno", self.r.training_text)
        self.assertNotIn("FROM COLUMNS [temperatura, vibracion, tipo_maquina]", self.r.training_text)

    def test_dataset_template_header_expanded(self):
        header = self.r.dataset_template_text.splitlines()[0]
        self.assertIn("tipo_maquina__compresor", header)
        self.assertNotIn(",tipo_maquina,", "," + header + ",")

    def test_parses_and_params_build(self):
        prog = parse_text(self.r.mxai_text)
        ps = build_initial_parameter_set(prog)  # must not raise
        self.assertTrue(ps.parameters)

    def test_output_labels_preserved(self):
        # multiclass keeps a labelled ProbabilityMap (case is normalized by the
        # deterministic generator; C4 owns label handling — here we just ensure the
        # one-hot expansion didn't clobber the 3-class labelled output).
        self.assertIn("ProbabilityMap[", self.r.mxai_text)
        self.assertEqual(self.r.output_units, 3)
        self.assertEqual([l.lower() for l in self.r.labels], ["ok", "desgaste", "fallo"])


class ThresholdTest(unittest.TestCase):
    def test_high_cardinality_not_onehot_by_dense(self):
        # > _ONEHOT_MAX values: the dense generator leaves it scalar (embedding path is
        # the composite/LLM route, C5); it must NOT explode into dozens of columns here.
        values = ", ".join(f"C{i}" for i in range(_ONEHOT_MAX + 5))
        prompt = f"clasificar multiclase\nFEATURES:\n  cod: Categorical[{values}]\n  x: Scalar en [0,1]\nSALIDA: y: ProbabilityMap[A, B]"
        r = DenseNetworkGenerator().generate(prompt)
        self.assertEqual(r.field_categories, {})  # not materialized as one-hot
        prog = parse_text(r.mxai_text)
        self.assertIn("cod", prog.vectors[0].fields)  # stays a single (scalar) column

    def test_low_cardinality_boundary_onehot(self):
        values = ", ".join(f"C{i}" for i in range(_ONEHOT_MAX))  # exactly the max
        prompt = f"clasificar multiclase\nFEATURES:\n  cod: Categorical[{values}]\n  x: Scalar en [0,1]\nSALIDA: y: ProbabilityMap[A, B]"
        r = DenseNetworkGenerator().generate(prompt)
        self.assertIn("cod", r.field_categories)
        self.assertEqual(len(r.field_categories["cod"]), _ONEHOT_MAX)


class InvariantPromptWinsTest(unittest.TestCase):
    """Invariant 1: an explicit Categorical in the prompt must survive even when a
    caller (LLM path) passes its own input_fields — and mixed prompts must not
    produce spurious/mangled fields (segmento_categorical, scalar)."""

    def test_input_fields_do_not_drop_prompt_categorical(self):
        r = DenseNetworkGenerator().generate(
            "clasificar red densa\nFEATURES:\n  segmento: Categorical[VIP, Normal, Riesgo]\n"
            "SALIDA: y: ProbabilityMap[A, B]",
            input_fields=["edad"],
        )
        # the explicit categorical is materialized as one-hot despite input_fields
        self.assertEqual(r.field_categories, {"segmento": ["VIP", "Normal", "Riesgo"]})
        fields = parse_text(r.mxai_text).vectors[0].fields
        self.assertIn("edad", fields)
        self.assertIn("segmento__vip", fields)

    def test_mixed_prompt_has_no_spurious_or_mangled_fields(self):
        # regression: _extract_fields used to mangle "segmento: Categorical[...]" into
        # "segmento_categorical" and pick up "scalar" as a field. Stripping typed
        # declarations first prevents that.
        r = DenseNetworkGenerator().generate(
            "clasificar red densa\nFEATURES: edad: Scalar en [0,100], "
            "segmento: Categorical[A, B, C]\nSALIDA: y: ProbabilityMap[X, Y]"
        )
        fields = parse_text(r.mxai_text).vectors[0].fields
        self.assertNotIn("segmento_categorical", fields)
        self.assertNotIn("scalar", fields)
        self.assertNotIn("segmento", fields)  # replaced by one-hot columns
        self.assertEqual(set(fields), {"edad", "segmento__a", "segmento__b", "segmento__c"})


class DispatchSurfacesFieldCategoriesTest(unittest.TestCase):
    def test_analyze_playground_request_surfaces_field_categories(self):
        from matrixai.playground import analyze_playground_request
        res = analyze_playground_request({"mode": "prompt", "prompt": _PROMPT, "use_llm": False})
        self.assertTrue(res.get("ok"), res.get("error"))
        self.assertEqual(res.get("field_categories"),
                         {"tipo_maquina": ["TORNO", "FRESADORA", "PRENSA", "COMPRESOR"]})


@unittest.skipUnless(_onnx_available(), "onnx/onnxruntime required")
class ExportAcceptsRawStringTest(unittest.TestCase):
    """End-to-end: a prompt-declared categorical → one-hot → exported bundle whose
    predict.py accepts the raw human name (the whole point of C2)."""

    def test_downloaded_predict_accepts_raw_category(self):
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
            field_ranges={"temperatura": (0, 150), "vibracion": (0, 50)},
            field_categories=r.field_categories,   # <- straight from the generator
            labels=["OK", "DESGASTE", "FALLO"],
        )
        self.assertIsNone(result.inference_spec_skipped_reason)
        bd = Path(result.bundle_dir)
        spec = json.loads((bd / "inference_spec.json").read_text())
        tm = spec["fields"]["tipo_maquina"]
        self.assertEqual(tm["encoding"], "one_hot")
        mapping = {v["raw"]: v["column"] for v in tm["values"]}
        self.assertEqual(mapping["TORNO"], "tipo_maquina__torno")

        # the downloaded predict.py accepts the RAW human name
        m = util.spec_from_file_location("pred_dl", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))
        out = model.predict({"temperatura": 75, "vibracion": 10, "tipo_maquina": "TORNO"})
        self.assertEqual(set(out), {"OK", "DESGASTE", "FALLO"})
