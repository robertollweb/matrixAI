"""EXPORT Modelo descargable — Corte 1: inference_spec.json.

El "tokenizer" del bundle: describe cómo un registro crudo se convierte en el
vector float32 normalizado que espera el ONNX. Estos tests cubren la lógica de
codificación en aislamiento (sin onnx) más un test de integración por
create_edge_bundle.

Cobertura:
- input_order == vector.fields (orden exacto del grafo)
- escalar con rango (arg field_ranges y rango del .mxai), scalar01 de fallback
- one-hot: mapeo raw->column autoritativo, incluido acentos/espacios
- embedding_index: con vocab humano y con vocab_size puro
- hashes + shapes + metadata presentes
- output: classification / binary_classification / regression, labels override y desde .mxai
- field_types (S2 boolean/integer) anotado
- guard de SEQUENCE / multi-VECTOR
- integración: inference_spec.json cae en el bundle con input_order coherente
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from matrixai.ir.schema import MatrixAIProgram, VectorSpec, NetworkSpec, EmbeddingSpec, SequenceSpec
from matrixai.types import TypeSpec, RangeSpec
from matrixai.parameters.store import ParameterSet
from matrixai.export.onnx_exporter import OnnxExportResult
from matrixai.export.inference_spec import build_inference_spec, InferenceSpecError
from matrixai.training.categorical import _build_group_names


def _ps() -> ParameterSet:
    return ParameterSet(
        parameter_set_id="PS_test",
        model_hash="mxai_deadbeef",
        parameter_schema_hash="params_cafe",
        parameters={},
    )


def _export_result(*, input_name="features", input_shape=None, output_name="out",
                   output_shape=None, labels=None) -> OnnxExportResult:
    # Default output is a single regression value: tests that don't care about the
    # output contract build a spec without needing labels (multiclass now requires them).
    return OnnxExportResult(
        output_path="model.onnx",
        opset_version=17,
        model_hash="mxai_deadbeef",
        parameter_set_id="PS_test",
        parameter_schema_hash="params_cafe",
        input_name=input_name,
        input_shape=input_shape if input_shape is not None else [-1, 1],
        output_name=output_name,
        output_shape=output_shape if output_shape is not None else [-1],
        exported_functions=["net"],
        labels=labels or [],
    )


def _vector(fields, field_types=None) -> VectorSpec:
    return VectorSpec(name="V", size=len(fields), fields=list(fields),
                      field_types=field_types or {})


def _program(vector, networks=None, sequences=None) -> MatrixAIProgram:
    return MatrixAIProgram(
        project="Test",
        vectors=[vector],
        networks=networks or [],
        sequences=sequences or [],
    )


class InputOrderAndMetadataTest(unittest.TestCase):
    def test_input_order_equals_vector_fields(self):
        vec = _vector(["a", "b", "c"])
        spec = build_inference_spec(_program(vec), _ps(),
                                    _export_result(input_shape=[-1, 3]))
        self.assertEqual(spec["input_order"], ["a", "b", "c"])

    def test_hashes_shapes_and_metadata_present(self):
        vec = _vector(["a"])
        spec = build_inference_spec(_program(vec), _ps(),
                                    _export_result(input_shape=[-1, 1]))
        self.assertEqual(spec["spec_version"], 1)
        self.assertEqual(spec["model_hash"], "mxai_deadbeef")
        self.assertEqual(spec["parameter_schema_hash"], "params_cafe")
        self.assertEqual(spec["parameter_set_id"], "PS_test")
        self.assertEqual(spec["onnx_opset"], 17)
        self.assertEqual(spec["onnx_file"], "model.onnx")
        self.assertEqual(spec["input_name"], "features")
        self.assertEqual(spec["input_shape"], [-1, 1])
        self.assertIn("created_at", spec)
        self.assertIn("matrixai_version", spec)


class ScalarEncodingTest(unittest.TestCase):
    def test_scalar_range_from_field_ranges_arg(self):
        vec = _vector(["edad"])
        spec = build_inference_spec(
            _program(vec), _ps(), _export_result(input_shape=[-1, 1]),
            field_ranges={"edad": (0.0, 120.0)},
        )
        self.assertEqual(spec["fields"]["edad"], {"encoding": "scalar", "range": [0, 120]})

    def test_scalar_range_from_mxai_field_type(self):
        vec = _vector(["imc"], {"imc": TypeSpec("Scalar", range=RangeSpec(10.0, 70.0))})
        spec = build_inference_spec(_program(vec), _ps(), _export_result(input_shape=[-1, 1]))
        self.assertEqual(spec["fields"]["imc"], {"encoding": "scalar", "range": [10, 70]})

    def test_scalar01_fallback_when_no_range(self):
        vec = _vector(["x"])
        spec = build_inference_spec(_program(vec), _ps(), _export_result(input_shape=[-1, 1]))
        self.assertEqual(spec["fields"]["x"], {"encoding": "scalar01"})

    def test_field_ranges_arg_wins_over_mxai_range(self):
        vec = _vector(["t"], {"t": TypeSpec("Scalar", range=RangeSpec(0.0, 1.0))})
        spec = build_inference_spec(
            _program(vec), _ps(), _export_result(input_shape=[-1, 1]),
            field_ranges={"t": (30.0, 45.0)},
        )
        self.assertEqual(spec["fields"]["t"]["range"], [30, 45])


class OneHotEncodingTest(unittest.TestCase):
    def setUp(self):
        self.values = ["CARDIOLOGÍA", "UCI", "Médico de Familia"]
        self.cols = _build_group_names("especialidad", self.values)
        # input_order = expanded one-hot columns + one plain scalar
        self.vec = _vector(["edad"] + self.cols)

    def test_one_hot_group_detected(self):
        spec = build_inference_spec(
            _program(self.vec), _ps(),
            _export_result(input_shape=[-1, len(self.vec.fields)]),
            field_categories={"especialidad": self.values},
        )
        entry = spec["fields"]["especialidad"]
        self.assertEqual(entry["encoding"], "one_hot")
        self.assertEqual([v["raw"] for v in entry["values"]], self.values)

    def test_raw_to_column_mapping_is_authoritative(self):
        spec = build_inference_spec(
            _program(self.vec), _ps(),
            _export_result(input_shape=[-1, len(self.vec.fields)]),
            field_categories={"especialidad": self.values},
        )
        mapping = {v["raw"]: v["column"] for v in spec["fields"]["especialidad"]["values"]}
        # accents/spaces must resolve to the exact training-time column names
        self.assertEqual(mapping["CARDIOLOGÍA"], "especialidad__cardiologia")
        self.assertEqual(mapping["UCI"], "especialidad__uci")
        self.assertEqual(mapping["Médico de Familia"], "especialidad__medico_de_familia")

    def test_one_hot_columns_not_listed_as_scalars(self):
        spec = build_inference_spec(
            _program(self.vec), _ps(),
            _export_result(input_shape=[-1, len(self.vec.fields)]),
            field_categories={"especialidad": self.values},
        )
        # the expanded columns must NOT appear as standalone scalar fields
        for col in self.cols:
            self.assertNotIn(col, spec["fields"])
        # the plain scalar is still there
        self.assertIn("edad", spec["fields"])


class EmbeddingEncodingTest(unittest.TestCase):
    def _program_with_embedding(self, source, vocab):
        net = NetworkSpec(
            name="N", input="V", layers=[], output="out",
            output_type_str="ProbabilityMap[NO, SI]", kind="composite_network",
            embeddings=[EmbeddingSpec(name="e", source=source, vocab=vocab, dim=4)],
        )
        return _program(_vector(["edad", source]), networks=[net])

    def test_embedding_with_human_vocab(self):
        prog = self._program_with_embedding("grupo", 3)
        spec = build_inference_spec(
            prog, _ps(), _export_result(input_shape=[-1, 2]),
            field_categories={"grupo": ["cardio", "respiratorio", "trauma"]},
        )
        entry = spec["fields"]["grupo"]
        self.assertEqual(entry["encoding"], "embedding_index")
        self.assertEqual(entry["column"], "grupo")
        self.assertEqual(entry["vocab"], ["cardio", "respiratorio", "trauma"])
        self.assertNotIn("vocab_size", entry)

    def test_embedding_with_vocab_size_only(self):
        prog = self._program_with_embedding("codigo", 50)
        spec = build_inference_spec(prog, _ps(), _export_result(input_shape=[-1, 2]))
        entry = spec["fields"]["codigo"]
        self.assertEqual(entry["encoding"], "embedding_index")
        self.assertEqual(entry["vocab_size"], 50)
        self.assertNotIn("vocab", entry)


class OutputContractTest(unittest.TestCase):
    def test_classification_labels_from_arg(self):
        vec = _vector(["a"])
        spec = build_inference_spec(
            _program(vec), _ps(), _export_result(output_shape=[-1, 2]),
            labels=["NO", "SI"],
        )
        self.assertEqual(spec["output"]["kind"], "classification")
        self.assertEqual(spec["output"]["labels"], ["NO", "SI"])

    def test_classification_labels_from_program_probmap(self):
        net = NetworkSpec(name="N", input="V", layers=[], output="out",
                          output_type_str="ProbabilityMap[BAJO, ALTO]", kind="dense_network")
        spec = build_inference_spec(
            _program(_vector(["a"]), networks=[net]), _ps(),
            _export_result(output_shape=[-1, 2]),
        )
        self.assertEqual(spec["output"]["labels"], ["BAJO", "ALTO"])

    def test_multiclass_without_real_labels_raises(self):
        # export labels positional ("0","1","2") are NOT real labels -> must fail,
        # not invent labels for a downloadable model.
        vec = _vector(["a"])
        with self.assertRaises(InferenceSpecError):
            build_inference_spec(
                _program(vec), _ps(),
                _export_result(output_shape=[-1, 3], labels=["0", "1", "2"]),
            )

    def test_multiclass_label_count_mismatch_raises(self):
        vec = _vector(["a"])
        with self.assertRaises(InferenceSpecError):
            build_inference_spec(
                _program(vec), _ps(), _export_result(output_shape=[-1, 3]),
                labels=["only", "two"],
            )

    def test_multiclass_with_real_export_labels_ok(self):
        vec = _vector(["a"])
        spec = build_inference_spec(
            _program(vec), _ps(),
            _export_result(output_shape=[-1, 2], labels=["SPAM", "HAM"]),
        )
        self.assertEqual(spec["output"]["kind"], "classification")
        self.assertEqual(spec["output"]["labels"], ["SPAM", "HAM"])

    def test_binary_classification_single_output_two_labels(self):
        vec = _vector(["a"])
        spec = build_inference_spec(
            _program(vec), _ps(), _export_result(output_shape=[-1]),
            labels=["NO", "SI"],
        )
        self.assertEqual(spec["output"]["kind"], "binary_classification")
        self.assertEqual(spec["output"]["labels"], ["NO", "SI"])

    def test_regression_single_output_no_labels(self):
        vec = _vector(["a"])
        spec = build_inference_spec(_program(vec), _ps(), _export_result(output_shape=[-1]))
        self.assertEqual(spec["output"]["kind"], "regression")


class FieldTypesTest(unittest.TestCase):
    def test_field_type_annotation_from_arg(self):
        vec = _vector(["activo"])
        spec = build_inference_spec(
            _program(vec), _ps(), _export_result(input_shape=[-1, 1]),
            field_types={"activo": "boolean"},
        )
        self.assertEqual(spec["fields"]["activo"]["type"], "boolean")

    def test_field_type_derived_from_mxai(self):
        # Boolean/Integer semantics survive even when the Studio doesn't pass field_types.
        vec = _vector(["activo", "edad", "x"], {
            "activo": TypeSpec("Boolean"),
            "edad": TypeSpec("Integer"),
        })
        spec = build_inference_spec(_program(vec), _ps(),
                                    _export_result(input_shape=[-1, 3]))
        self.assertEqual(spec["fields"]["activo"]["type"], "boolean")
        self.assertEqual(spec["fields"]["edad"]["type"], "integer")
        # plain Scalar carries no type annotation
        self.assertNotIn("type", spec["fields"]["x"])

    def test_field_type_arg_wins_over_mxai(self):
        vec = _vector(["f"], {"f": TypeSpec("Integer")})
        spec = build_inference_spec(_program(vec), _ps(),
                                    _export_result(input_shape=[-1, 1]),
                                    field_types={"f": "number"})
        self.assertEqual(spec["fields"]["f"]["type"], "number")


class EmbeddingVocabFromMxaiTest(unittest.TestCase):
    def test_embedding_vocab_from_type_args(self):
        # human vocab declared in the .mxai (Categorical[A, B, C]) is used even without
        # field_categories from the Studio.
        net = NetworkSpec(name="N", input="V", layers=[], output="out",
                          output_type_str="", kind="composite_network",
                          embeddings=[EmbeddingSpec(name="e", source="grupo", vocab=3, dim=4)])
        vec = _vector(["edad", "grupo"],
                      {"grupo": TypeSpec("Categorical", parameters={"args": ["cardio", "resp", "trauma"]})})
        spec = build_inference_spec(_program(vec, networks=[net]), _ps(),
                                    _export_result(output_shape=[-1]))
        entry = spec["fields"]["grupo"]
        self.assertEqual(entry["encoding"], "embedding_index")
        self.assertEqual(entry["vocab"], ["cardio", "resp", "trauma"])
        self.assertNotIn("vocab_size", entry)


class ExampleInputValidationTest(unittest.TestCase):
    def test_valid_example_input_accepted(self):
        vec = _vector(["edad", "x"])
        # should not raise
        build_inference_spec(_program(vec), _ps(), _export_result(input_shape=[-1, 2]),
                             example_input={"edad": 65})

    def test_unknown_field_in_example_input_raises(self):
        vec = _vector(["edad", "x"])
        with self.assertRaises(InferenceSpecError):
            build_inference_spec(_program(vec), _ps(), _export_result(input_shape=[-1, 2]),
                                 example_input={"no_existe": 1})


class ScopeGuardTest(unittest.TestCase):
    def test_sequence_input_rejected(self):
        prog = MatrixAIProgram(project="Seq", vectors=[_vector(["a"])],
                               sequences=[SequenceSpec(name="S", length=8, vocab_size=100)])
        with self.assertRaises(InferenceSpecError):
            build_inference_spec(prog, _ps(), _export_result())

    def test_no_vector_rejected(self):
        prog = MatrixAIProgram(project="None", vectors=[])
        with self.assertRaises(InferenceSpecError):
            build_inference_spec(prog, _ps(), _export_result())

    def test_multi_vector_rejected(self):
        prog = MatrixAIProgram(project="Multi",
                               vectors=[_vector(["a"]), _vector(["b"])])
        with self.assertRaises(InferenceSpecError):
            build_inference_spec(prog, _ps(), _export_result())


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


@unittest.skipUnless(_onnx_available(), "onnx not installed")
class IntegrationBundleTest(unittest.TestCase):
    """inference_spec.json must land in the bundle with a coherent input_order."""

    def setUp(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        base = Path(__file__).parent.parent
        self.mxai = base / "examples" / "email-agent.mxai"
        self.prog = parse_file(self.mxai)
        self.ps = build_initial_parameter_set(self.prog)
        self.td = tempfile.mkdtemp()
        self.params_path = Path(self.td) / "params.json"
        write_parameter_set(str(self.params_path), self.ps)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _bundle(self, *, labels=None, outdir="bundle"):
        from matrixai.export import create_edge_bundle
        return create_edge_bundle(
            self.prog, self.ps,
            mxai_path=str(self.mxai), params_path=str(self.params_path),
            outdir=str(Path(self.td) / outdir), validate=False,
            labels=labels,
        )

    def test_inference_spec_in_bundle_with_labels(self):
        # email-agent is a 3-class softmax with no declared labels; the Studio would
        # pass them. With labels, the bundle gets a usable inference_spec.json.
        result = self._bundle(labels=["support", "sales", "other"])
        self.assertIn("inference_spec.json", result.files)
        self.assertIsNone(result.inference_spec_skipped_reason)
        spec = json.loads((Path(result.bundle_dir) / "inference_spec.json").read_text())
        self.assertEqual(len(spec["input_order"]), spec["input_shape"][-1])
        self.assertEqual(spec["model_hash"], self.ps.model_hash)
        self.assertEqual(spec["output"]["labels"], ["support", "sales", "other"])

    def test_unlabelled_classification_skips_spec_observably(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._bundle(outdir="bundle_nolabels")  # no labels
        self.assertNotIn("inference_spec.json", result.files)
        self.assertIsNotNone(result.inference_spec_skipped_reason)
        self.assertIn("labels", result.inference_spec_skipped_reason.lower())
        # observable via to_dict too
        self.assertIn("inference_spec_skipped_reason", result.to_dict())
