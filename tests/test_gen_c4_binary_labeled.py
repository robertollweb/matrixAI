"""GENERACIÓN tipos del prompt — C4: salida binaria etiquetada.

Si el prompt declara `ProbabilityMap[neg, pos]` / `Label[neg, pos]` EXPLÍCITO con
exactamente 2 labels, el generador produce softmax de 2 clases (no sigmoid de 1
unidad): mismo camino que multiclase con n=2, con `loss=cross_entropy` y TARGET de
dataset `Label[neg, pos]`. El orden de labels declarado se conserva. Un prompt
binario SIN ProbabilityMap explícito sigue usando sigmoid/Probability (retrocompat).

También corrige un bug real descubierto al implementar esto: la detección de tarea
solo miraba la palabra "binario" (masculino) y no "binaria" (femenino, la forma más
común en español: "clasificación binaria"), lo que hacía que ese prompt EXACTO cayera
por defecto a multiclase con 3 labels inventados ("class_a/b/c") en vez de sigmoid.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path

from matrixai.training.dense_generator import DenseNetworkGenerator
from matrixai.training.composite_generator import CompositeNetworkGenerator
from matrixai.parser import parse_text


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None and util.find_spec("onnxruntime") is not None


class ExplicitTwoLabelBracketIsSoftmaxTest(unittest.TestCase):
    def setUp(self):
        self.r = DenseNetworkGenerator().generate(
            "modelo de abandono de clientes\nSALIDA: y: ProbabilityMap[NO, SI]"
        )

    def test_output_is_two_class_softmax_not_sigmoid(self):
        self.assertEqual(self.r.output_activation, "softmax")
        self.assertEqual(self.r.output_units, 2)
        self.assertEqual(self.r.output_type, "ProbabilityMap[no, si]")

    def test_loss_is_cross_entropy(self):
        self.assertEqual(self.r.loss_type, "cross_entropy")

    def test_label_order_preserved(self):
        self.assertEqual(self.r.labels, ["no", "si"])

    def test_dataset_target_is_label_not_probability(self):
        self.assertIn("TARGET predicted_class: Label[no, si]", self.r.training_text)
        self.assertNotIn("TARGET predicted_class: Probability", self.r.training_text)
        self.assertIn("TYPE cross_entropy", self.r.training_text)

    def test_dummy_row_uses_first_label_not_zero(self):
        header, row = self.r.dataset_template_text.strip().splitlines()
        self.assertEqual(row.split(",")[-1], "no")

    def test_mxai_declares_probabilitymap_two_units(self):
        prog = parse_text(self.r.mxai_text)
        self.assertIn("ProbabilityMap[no, si]", self.r.mxai_text)
        self.assertEqual(prog.networks[0].layers[-1].units, 2)


class LabelOrderIsDeclarationOrderTest(unittest.TestCase):
    def test_reversed_declaration_order_is_kept(self):
        r = DenseNetworkGenerator().generate("modelo\nSALIDA: y: ProbabilityMap[SI, NO]")
        self.assertEqual(r.labels, ["si", "no"])
        self.assertEqual(r.output_type, "ProbabilityMap[si, no]")


class BinaryWithoutBracketStaysSigmoidTest(unittest.TestCase):
    """Retrocompat (contract's own example): a bare binary prompt with NO explicit
    ProbabilityMap/Label bracket must keep the 1-unit sigmoid path unchanged."""

    def test_clasificacion_binaria_a_secas_is_sigmoid(self):
        r = DenseNetworkGenerator().generate(
            "clasificación binaria de clientes con edad, saldo, activo"
        )
        self.assertEqual(r.output_activation, "sigmoid")
        self.assertEqual(r.output_units, 1)
        self.assertEqual(r.output_type, "Probability")
        self.assertEqual(r.loss_type, "binary_cross_entropy")

    def test_existing_binary_keyword_prompt_unaffected(self):
        r = DenseNetworkGenerator().generate("detectar fraude")
        self.assertEqual(r.output_activation, "sigmoid")
        self.assertEqual(r.output_units, 1)
        self.assertEqual(r.loss_type, "binary_cross_entropy")

    def test_english_binary_classification_bare_is_sigmoid(self):
        r = DenseNetworkGenerator().generate("binary classification of transactions")
        self.assertEqual(r.output_activation, "sigmoid")
        self.assertEqual(r.output_units, 1)


class MulticlassThreeLabelsUnaffectedTest(unittest.TestCase):
    def test_three_label_bracket_still_multiclass(self):
        r = DenseNetworkGenerator().generate(
            "clasificar en tres niveles\nSALIDA: y: ProbabilityMap[BAJO, MEDIO, ALTO]"
        )
        self.assertEqual(r.output_activation, "softmax")
        self.assertEqual(r.output_units, 3)
        self.assertEqual(r.labels, ["bajo", "medio", "alto"])


class CompositeHonorsSameC4PolicyTest(unittest.TestCase):
    """Invariant 5: the composite generator must apply the exact same policy."""

    def test_composite_explicit_two_label_bracket_is_softmax(self):
        r = CompositeNetworkGenerator().generate(
            "clasificar con bloques residuales\nFEATURES:\n"
            "  precio: Scalar en [0, 1000]\n"
            "SALIDA: y: ProbabilityMap[NO, SI]",
            force_residual=True,
        )
        self.assertEqual(r.output_activation, "softmax")
        self.assertEqual(r.output_units, 2)
        self.assertEqual(r.labels, ["no", "si"])
        self.assertEqual(r.loss_type, "cross_entropy")
        self.assertIn("TARGET predicted_class: Label[no, si]", r.training_text)

    def test_composite_bare_binary_stays_sigmoid(self):
        r = CompositeNetworkGenerator().generate(
            "clasificación binaria con bloques residuales\nFEATURES:\n"
            "  precio: Scalar en [0, 1000]\n",
            force_residual=True,
        )
        self.assertEqual(r.output_activation, "sigmoid")
        self.assertEqual(r.output_units, 1)


_DISPATCH_PROMPT = """PROYECTO: AbandonoClientesBanca
MODO: clasificación binaria.
ARQUITECTURA: red densa.
FEATURES:
  edad: Scalar en [18, 95]
  saldo_medio: Scalar en [0, 500000]
SALIDA: resultado: ProbabilityMap[PERMANECE, ABANDONA]
"""


class AuditPromptBracketWinsOverCallerLabelsTest(unittest.TestCase):
    """C4 audit [ALTA]: caller/LLM ``labels=`` must NOT override an explicit
    ProbabilityMap[...]/Label[...] bracket in the prompt (invariant 1). Before the
    fix, labels=['x','y'] turned ProbabilityMap[NO,SI] into a sigmoid with x/y."""

    def test_two_caller_labels_do_not_make_it_sigmoid(self):
        r = DenseNetworkGenerator().generate(
            "modelo\nSALIDA: y: ProbabilityMap[NO, SI]", labels=["x", "y"]
        )
        self.assertEqual(r.output_activation, "softmax")
        self.assertEqual(r.output_units, 2)
        self.assertEqual(r.labels, ["no", "si"])
        self.assertTrue(any("invariante 1" in w for w in r.warnings), r.warnings)

    def test_three_caller_labels_do_not_widen_the_output(self):
        r = DenseNetworkGenerator().generate(
            "modelo\nSALIDA: y: ProbabilityMap[NO, SI]", labels=["x", "y", "z"]
        )
        self.assertEqual(r.output_units, 2)
        self.assertEqual(r.labels, ["no", "si"])

    def test_matching_caller_labels_produce_no_spurious_warning(self):
        # the Studio/LLM echoing the same labels (any case) is not a conflict
        r = DenseNetworkGenerator().generate(
            "modelo\nSALIDA: y: ProbabilityMap[NO, SI]", labels=["NO", "SI"]
        )
        self.assertEqual(r.labels, ["no", "si"])
        self.assertEqual(r.warnings, [])

    def test_composite_same_precedence(self):
        r = CompositeNetworkGenerator().generate(
            "con bloques residuales\nFEATURES:\n  precio: Scalar en [0, 1000]\n"
            "SALIDA: y: ProbabilityMap[NO, SI]",
            labels=["x", "y"], force_residual=True,
        )
        self.assertEqual(r.output_activation, "softmax")
        self.assertEqual(r.output_units, 2)
        self.assertEqual(r.labels, ["no", "si"])
        self.assertTrue(any("invariante 1" in w for w in r.warnings), r.warnings)


class AuditBracketForcesClassificationTaskTest(unittest.TestCase):
    """C4 audit [ALTA/MEDIA]: an explicit bracket with >=2 labels forces the
    classification path even with NO classification keyword — before the fix,
    'modelo ... ProbabilityMap[A,B,C]' extracted the labels but still emitted
    linear/Scalar/mse (regression default), an incoherent result."""

    def test_bracket_without_any_task_keyword(self):
        r = DenseNetworkGenerator().generate("modelo\nSALIDA: y: ProbabilityMap[A, B, C]")
        self.assertEqual(r.output_activation, "softmax")
        self.assertEqual(r.output_units, 3)
        self.assertEqual(r.loss_type, "cross_entropy")
        self.assertEqual(r.labels, ["a", "b", "c"])

    def test_bracket_beats_regression_keywords(self):
        r = DenseNetworkGenerator().generate(
            "predecir precio de la vivienda\nSALIDA: y: ProbabilityMap[A, B, C]"
        )
        self.assertEqual(r.output_activation, "softmax")
        self.assertEqual(r.output_units, 3)
        self.assertEqual(r.output_type, "ProbabilityMap[a, b, c]")

    def test_label_bracket_also_forces_classification(self):
        r = DenseNetworkGenerator().generate("modelo\nSALIDA: y: Label[A, B, C]")
        self.assertEqual(r.output_activation, "softmax")
        self.assertEqual(r.output_units, 3)

    def test_regression_without_bracket_unchanged(self):
        r = DenseNetworkGenerator().generate(
            "predecir precio de la vivienda con superficie, habitaciones"
        )
        self.assertEqual(r.output_activation, "linear")
        self.assertEqual(r.output_type, "Scalar")
        self.assertEqual(r.loss_type, "mse")


class AuditMaxLabelsCapWarnsTest(unittest.TestCase):
    """C4 audit [MEDIA/BAJA]: the max_labels limit still applies to a declared
    bracket, but with an explicit warning — never a silent truncation."""

    def setUp(self):
        import os
        os.environ["MATRIXAI_MAX_LABELS"] = "2"
        self.addCleanup(os.environ.pop, "MATRIXAI_MAX_LABELS", None)

    def test_capped_bracket_emits_warning(self):
        r = DenseNetworkGenerator().generate("clasificar\nSALIDA: y: ProbabilityMap[A, B, C]")
        self.assertEqual(r.output_units, 2)
        self.assertEqual(r.labels, ["a", "b"])
        self.assertTrue(any("max_labels=2" in w and "recorta" in w for w in r.warnings),
                        r.warnings)

    def test_bracket_within_cap_no_warning(self):
        r = DenseNetworkGenerator().generate("clasificar\nSALIDA: y: ProbabilityMap[A, B]")
        self.assertEqual(r.labels, ["a", "b"])
        self.assertEqual(r.warnings, [])


class DispatchSurfacesC4Test(unittest.TestCase):
    def test_analyze_playground_request_generates_two_class_softmax(self):
        from matrixai.playground import analyze_playground_request
        res = analyze_playground_request({
            "mode": "prompt", "prompt": _DISPATCH_PROMPT, "use_llm": False,
        })
        self.assertTrue(res.get("ok"), res.get("error"))
        self.assertIn("ProbabilityMap[permanece, abandona]", res["mxai"])
        self.assertIn("LAYER Dense units=2 activation=softmax", res["mxai"])
        self.assertIn("TYPE cross_entropy", res["training_text"])


@unittest.skipUnless(_onnx_available(), "onnx/onnxruntime required")
class ExportAcceptsTwoClassSoftmaxTest(unittest.TestCase):
    """End-to-end: an explicit 2-label bracket produces a real 2-unit softmax
    network whose exported bundle decodes as {NO: p, SI: p} (classification kind,
    not binary_classification) — the export layer is shape-driven and needs no C4
    changes of its own, only a correctly-shaped generated network."""

    def test_downloaded_predict_gives_both_named_probabilities(self):
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        from matrixai.export import create_edge_bundle

        r = DenseNetworkGenerator().generate(
            "modelo de abandono\nFEATURES:\n"
            "  edad: Scalar en [18, 95]\n"
            "  saldo: Scalar en [0, 500000]\n"
            "SALIDA: y: ProbabilityMap[NO, SI]"
        )
        prog = parse_text(r.mxai_text)
        ps = build_initial_parameter_set(prog)
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(td), True)
        (td / "m.mxai").write_text(r.mxai_text, encoding="utf-8")
        write_parameter_set(str(td / "p.json"), ps)

        result = create_edge_bundle(
            prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
            outdir=str(td / "b"), validate=False,
            field_ranges=r.field_ranges, field_types=r.field_types, labels=r.labels,
        )
        self.assertIsNone(result.inference_spec_skipped_reason)
        bd = Path(result.bundle_dir)
        spec = json.loads((bd / "inference_spec.json").read_text())
        self.assertEqual(spec["output"]["kind"], "classification")
        self.assertEqual(spec["output"]["labels"], ["no", "si"])

        m = util.spec_from_file_location("pred_dl", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))
        out = model.predict({"edad": 56, "saldo": 12000})
        self.assertEqual(set(out), {"no", "si"})
        self.assertAlmostEqual(sum(out.values()), 1.0, places=5)
