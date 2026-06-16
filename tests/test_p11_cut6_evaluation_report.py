"""P11 Cut 6 — audited evaluation_report.json for layer_call classifiers."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.parameters import validate_parameter_set, write_parameter_set
from matrixai.parser import parse_file
from matrixai.training.data import CSVDataAdapter
from matrixai.training.parser import parse_training_file
from matrixai.training.spec import EvaluationResult
from matrixai.training.trainer import GenericSupervisedEvaluator, GenericSupervisedTrainer

_BASE = Path(__file__).parent.parent
_MODEL = _BASE / "examples" / "transformer-classifier-vector.mxai"
_TRAIN_SPEC = _BASE / "examples" / "transformer-classifier-vector.mxtrain"
_CSV = _BASE / "examples" / "transformer-classifier-vector.train.csv"

_FIELDS = [f"x{i}" for i in range(8)]
_LABELS = ["class_a", "class_b"]
_VECTOR_NAME = "Input"


class TestP11Cut6EvaluationReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = parse_file(str(_MODEL))
        cls.spec = parse_training_file(str(_TRAIN_SPEC))
        cls.adapter = CSVDataAdapter(_CSV, _VECTOR_NAME, _FIELDS, "label", _LABELS)
        examples = cls.adapter.examples()

        training_result = GenericSupervisedTrainer().train(
            program=cls.program,
            training=cls.spec,
            examples=examples[:4],
            validation_examples=examples[4:6],
            prediction_key="logits",
            target_key="label",
            update_patterns=["classifier.*"],
            epochs=3,
            learning_rate=0.05,
            labels=_LABELS,
            vector_name=_VECTOR_NAME,
            vector_fields=_FIELDS,
            parameter_set_prefix="transformer_classifier_cut6",
        )
        cls.parameter_set = training_result["best_parameter_set"]
        cls.evaluation = GenericSupervisedEvaluator().evaluate(
            cls.spec,
            parameter_set=cls.parameter_set,
            data_path=str(_CSV),
            base_path=str(_BASE),
        )
        cls.payload = cls.evaluation.to_dict()

    def test_trainer_returns_valid_parameter_sets(self):
        self.assertEqual(self.parameter_set.parameter_set_id, "transformer_classifier_cut6_best")
        validation = validate_parameter_set(self.program, self.parameter_set)
        self.assertTrue(validation.ok, validation.errors)

    def test_evaluation_result_shape_matches_p4_contract(self):
        self.assertIsInstance(self.evaluation, EvaluationResult)
        for key in [
            "accuracy",
            "backend",
            "confusion_matrix",
            "dataset_fingerprint",
            "dataset_schema",
            "loss",
            "macro_f1",
            "model_hash",
            "parameter_schema_hash",
            "parameter_set_id",
            "per_label",
        ]:
            self.assertIn(key, self.payload)

    def test_backend_metadata_identifies_generic_evaluator(self):
        self.assertEqual(self.payload["backend"]["target"], "differentiable_python")
        self.assertEqual(self.payload["backend"]["evaluator"], "GenericSupervisedEvaluator")
        self.assertEqual(self.payload["backend"]["loss"], "cross_entropy")
        self.assertEqual(self.payload["backend"]["prediction"], "logits")

    def test_dataset_fingerprint_and_schema_are_preserved(self):
        self.assertEqual(self.payload["dataset_fingerprint"], self.adapter.fingerprint())
        self.assertEqual(self.payload["dataset_schema"]["input_vector"], _VECTOR_NAME)
        self.assertEqual(self.payload["dataset_schema"]["input_columns"], _FIELDS)
        self.assertEqual(self.payload["dataset_schema"]["target"], "label")
        self.assertEqual(self.payload["dataset_schema"]["rows"], 20)

    def test_metrics_are_finite_and_bounded(self):
        self.assertTrue(math.isfinite(self.payload["loss"]))
        self.assertGreaterEqual(self.payload["accuracy"], 0.0)
        self.assertLessEqual(self.payload["accuracy"], 1.0)
        self.assertGreaterEqual(self.payload["macro_f1"], 0.0)
        self.assertLessEqual(self.payload["macro_f1"], 1.0)

    def test_confusion_matrix_covers_all_rows(self):
        self.assertEqual(self.payload["labels"], _LABELS)
        self.assertEqual(set(self.payload["confusion_matrix"]), set(_LABELS))
        total = sum(sum(row.values()) for row in self.payload["confusion_matrix"].values())
        self.assertEqual(total, self.payload["rows"])
        self.assertEqual(total, 20)

    def test_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "evaluation_report.json"
            report_path.write_text(
                json.dumps(self.payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["parameter_set_id"], self.parameter_set.parameter_set_id)
        self.assertIn("confusion_matrix", loaded)

    def test_cli_evaluate_writes_generic_evaluation_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            params_path = Path(tmp_dir) / "params.best.json"
            report_path = Path(tmp_dir) / "evaluation_report.json"
            write_parameter_set(params_path, self.parameter_set)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "evaluate",
                    "examples/transformer-classifier-vector.mxai",
                    "--training",
                    "examples/transformer-classifier-vector.mxtrain",
                    "--params",
                    str(params_path),
                    "--data",
                    "examples/transformer-classifier-vector.train.csv",
                    "--output",
                    str(report_path),
                    "--json",
                ],
                cwd=_BASE,
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["backend"]["evaluator"], "GenericSupervisedEvaluator")
        self.assertEqual(report["parameter_set_id"], payload["parameter_set_id"])
        self.assertEqual(report["dataset_fingerprint"], self.adapter.fingerprint())


if __name__ == "__main__":
    unittest.main()
