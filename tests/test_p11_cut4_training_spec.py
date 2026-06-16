"""P11 Cut 4 — Hierarchical UPDATE .mxtrain spec, validated by TrainingVerifier and DifferentiabilityVerifier.

Tests for:
- Parsing transformer-classifier-vector.mxtrain without errors
- TrainingVerifier: ok, 14 trainable params, glob UPDATE patterns match
- DifferentiabilityVerifier: ok, prediction_node resolved
- Dataset: 20 rows, correct columns, valid label values
- Optimizer: SGD, LR=0.01, UPDATE encoder_attn.*, encoder_ffn.*, classifier.*
- Verifier extends P4 restriction: layer_call accepted for cross_entropy loss
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from matrixai.training.parser import parse_training_file
from matrixai.training.verifier import TrainingVerifier
from matrixai.training.differentiability import DifferentiabilityVerifier
from matrixai.training.trainer import match_update_patterns
from matrixai.parser import parse_file
from matrixai.compiler import BackendContractAnalyzer

_TRAIN_SPEC = Path(__file__).parent.parent / "examples" / "transformer-classifier-vector.mxtrain"
_MODEL_PATH = Path(__file__).parent.parent / "examples" / "transformer-classifier-vector.mxai"
_BASE = Path(__file__).parent.parent


def _spec():
    return parse_training_file(str(_TRAIN_SPEC))


def _result():
    return TrainingVerifier().verify(_spec(), base_path=str(_BASE))


class TestTrainingSpecParses(unittest.TestCase):

    def test_file_exists(self):
        self.assertTrue(_TRAIN_SPEC.exists())

    def test_parses_without_exception(self):
        spec = _spec()
        self.assertIsNotNone(spec)

    def test_model_field(self):
        self.assertIn("transformer-classifier-vector.mxai", _spec().model)

    def test_dataset_name(self):
        self.assertEqual(_spec().dataset.name, "TransformerTrainingSet")

    def test_dataset_source_kind(self):
        self.assertEqual(_spec().dataset.source_kind, "csv")

    def test_dataset_input_vector(self):
        self.assertEqual(_spec().dataset.input.vector, "Input")

    def test_dataset_input_columns(self):
        self.assertEqual(_spec().dataset.input.columns, [f"x{i}" for i in range(8)])

    def test_dataset_target_name(self):
        self.assertEqual(_spec().dataset.target.name, "label")

    def test_dataset_target_labels(self):
        labels = _spec().dataset.target.type.parameters.get("args", [])
        self.assertIn("class_a", labels)
        self.assertIn("class_b", labels)

    def test_loss_type(self):
        self.assertEqual(_spec().loss.type, "cross_entropy")

    def test_loss_prediction(self):
        self.assertEqual(_spec().loss.prediction, "logits")

    def test_loss_target(self):
        self.assertEqual(_spec().loss.target, "label")

    def test_optimizer_type(self):
        self.assertEqual(_spec().optimizer.type, "sgd")

    def test_optimizer_learning_rate(self):
        self.assertAlmostEqual(_spec().optimizer.learning_rate, 0.01)

    def test_optimizer_update_patterns(self):
        self.assertEqual(_spec().optimizer.update, ["encoder_attn.*", "encoder_ffn.*", "classifier.*"])

    def test_run_epochs(self):
        self.assertEqual(_spec().run.epochs, 10)

    def test_metric_type(self):
        self.assertEqual(_spec().metrics[0].type, "accuracy")


class TestTrainingVerifier(unittest.TestCase):

    def setUp(self):
        self._result = _result()

    def test_verifier_ok(self):
        self.assertTrue(self._result.ok, f"errors: {self._result.errors}")

    def test_no_errors(self):
        self.assertEqual(self._result.errors, [])

    def test_model_path_resolved(self):
        self.assertTrue(self._result.model_path.endswith(".mxai"))

    def test_dataset_path_resolved(self):
        self.assertTrue(self._result.dataset_path.endswith(".csv"))

    def test_trainable_params_count(self):
        self.assertEqual(len(self._result.trainable_parameters), 14)

    def test_encoder_attn_params_present(self):
        functions = {p["function"] for p in self._result.trainable_parameters}
        self.assertIn("encoder_attn", functions)

    def test_encoder_ffn_params_present(self):
        functions = {p["function"] for p in self._result.trainable_parameters}
        self.assertIn("encoder_ffn", functions)

    def test_classifier_params_present(self):
        functions = {p["function"] for p in self._result.trainable_parameters}
        self.assertIn("classifier", functions)

    def test_differentiability_ok(self):
        self.assertTrue(self._result.differentiability.get("ok"), f"diff errors: {self._result.differentiability.get('errors')}")

    def test_differentiability_prediction_node(self):
        self.assertEqual(self._result.differentiability.get("prediction_node"), "Logits")


class TestUpdatePatterns(unittest.TestCase):
    """Verify UPDATE glob patterns match the correct trainable parameters."""

    def setUp(self):
        result = _result()
        all_keys: list[str] = []
        seen: set[str] = set()
        for param in result.trainable_parameters:
            for key in (param["name"], f"{param['function']}.{param['name']}"):
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)
        self._all_keys = all_keys

    def test_encoder_attn_glob_matches_six_params(self):
        matched = match_update_patterns(["encoder_attn.*"], self._all_keys)
        self.assertEqual(len(matched), 6, f"matched: {matched}")

    def test_encoder_ffn_glob_matches_six_params(self):
        matched = match_update_patterns(["encoder_ffn.*"], self._all_keys)
        self.assertEqual(len(matched), 6, f"matched: {matched}")

    def test_classifier_glob_matches_two_params(self):
        matched = match_update_patterns(["classifier.*"], self._all_keys)
        self.assertEqual(len(matched), 2, f"matched: {matched}")

    def test_all_patterns_together_match_fourteen_keys(self):
        patterns = ["encoder_attn.*", "encoder_ffn.*", "classifier.*"]
        matched = match_update_patterns(patterns, self._all_keys)
        self.assertEqual(len(matched), 14, f"matched: {matched}")

    def test_wildcard_matches_all_fourteen(self):
        matched = match_update_patterns(["*"], self._all_keys)
        layer_keys = [k for k in matched if "." in k]
        self.assertEqual(len(layer_keys), 14)

    def test_encoder_attn_glob_includes_bias_and_gain(self):
        matched = match_update_patterns(["encoder_attn.*"], self._all_keys)
        self.assertIn("encoder_attn.bias", matched)
        self.assertIn("encoder_attn.gain", matched)

    def test_encoder_ffn_glob_includes_b1_and_b2(self):
        matched = match_update_patterns(["encoder_ffn.*"], self._all_keys)
        self.assertIn("encoder_ffn.b1", matched)
        self.assertIn("encoder_ffn.b2", matched)

    def test_classifier_glob_includes_b(self):
        matched = match_update_patterns(["classifier.*"], self._all_keys)
        self.assertIn("classifier.b", matched)


class TestDatasetFile(unittest.TestCase):
    """Verify the training CSV is structurally correct."""

    def setUp(self):
        import csv
        csv_path = _BASE / "examples" / "transformer-classifier-vector.train.csv"
        with csv_path.open(newline="") as f:
            self._rows = list(csv.DictReader(f))

    def test_row_count(self):
        self.assertEqual(len(self._rows), 20)

    def test_has_all_input_columns(self):
        for col in [f"x{i}" for i in range(8)]:
            self.assertIn(col, self._rows[0], f"missing column {col}")

    def test_has_label_column(self):
        self.assertIn("label", self._rows[0])

    def test_all_input_values_are_numeric(self):
        for row_idx, row in enumerate(self._rows, start=2):
            for col in [f"x{i}" for i in range(8)]:
                try:
                    v = float(row[col])
                    self.assertTrue(math.isfinite(v))
                except ValueError:
                    self.fail(f"row {row_idx} {col} is not numeric: {row[col]!r}")

    def test_all_labels_are_valid(self):
        valid = {"class_a", "class_b"}
        for row_idx, row in enumerate(self._rows, start=2):
            self.assertIn(row["label"], valid, f"row {row_idx} has invalid label {row['label']!r}")

    def test_both_classes_represented(self):
        labels = {row["label"] for row in self._rows}
        self.assertEqual(labels, {"class_a", "class_b"})


class TestLayerCallAcceptedForCrossEntropy(unittest.TestCase):
    """Regression: verifier must not reject layer_call prediction for cross_entropy."""

    def test_verifier_does_not_error_on_layer_call_loss(self):
        result = _result()
        for error in result.errors:
            self.assertNotIn("softmax_linear", error,
                             f"verifier still enforces old P4 softmax_linear restriction: {error}")

    def test_verifier_accepts_layer_call_prediction_kind(self):
        from matrixai.training.verifier import TrainingVerifier
        program = parse_file(str(_MODEL_PATH))
        prediction_fn = None
        for fn in program.functions:
            if fn.output == "logits":
                prediction_fn = fn
                break
        self.assertIsNotNone(prediction_fn)
        self.assertEqual(prediction_fn.semantic.kind, "layer_call")
        result = _result()
        self.assertTrue(result.ok)

    def test_old_softmax_linear_models_still_accepted(self):
        from matrixai.training.parser import parse_training_file as ptf
        email_train = _BASE / "examples" / "email-agent.supervised.mxtrain"
        if not email_train.exists():
            self.skipTest("email-agent.supervised.mxtrain not present")
        spec = ptf(str(email_train))
        result = TrainingVerifier().verify(spec, base_path=str(_BASE))
        self.assertTrue(result.ok, f"email-agent training broke: {result.errors}")


if __name__ == "__main__":
    unittest.main()
