"""P11 Cut 5 — Supervised training over small deterministic dataset.

Tests for:
- DifferentiabilityVerifier: glob UPDATE patterns now traced (14 paths, no "no paths" warning)
- Layer-to-node mapping: encoder_attn→AttnBlock, encoder_ffn→FfnBlock, classifier→Logits
- Cross-entropy over raw logits: softmax applied, loss finite and label-sensitive
- GenericSupervisedTrainer: train runs, loss decreases, params change, correct keys
- Backward compat: existing P4 softmax_linear models unaffected
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from pathlib import Path

from matrixai.parser import parse_file
from matrixai.training.parser import parse_training_file
from matrixai.training.data import CSVDataAdapter
from matrixai.training.differentiability import DifferentiabilityVerifier
from matrixai.training.trainer import (
    GenericSupervisedTrainer,
    _generic_cross_entropy_loss,
    _softmax_list,
)
from matrixai.training.verifier import TrainingVerifier
from matrixai.compiler import BackendContractAnalyzer

_BASE = Path(__file__).parent.parent
_MODEL = _BASE / "examples" / "transformer-classifier-vector.mxai"
_TRAIN_SPEC = _BASE / "examples" / "transformer-classifier-vector.mxtrain"
_CSV = _BASE / "examples" / "transformer-classifier-vector.train.csv"

_FIELDS = [f"x{i}" for i in range(8)]
_LABELS = ["class_a", "class_b"]
_VECTOR_NAME = "Input"


def _program():
    return parse_file(str(_MODEL))


def _spec():
    return parse_training_file(str(_TRAIN_SPEC))


def _adapter():
    return CSVDataAdapter(_CSV, _VECTOR_NAME, _FIELDS, "label", _LABELS)


def _verifier_result():
    return TrainingVerifier().verify(_spec(), base_path=str(_BASE))


# ---------------------------------------------------------------------------
# DifferentiabilityVerifier with glob UPDATE patterns
# ---------------------------------------------------------------------------

class TestDifferentiabilityVerifierGlob(unittest.TestCase):

    def setUp(self):
        spec = _spec()
        program = _program()
        report = BackendContractAnalyzer().analyze(program)
        self._diff = DifferentiabilityVerifier().verify(spec, program, report)

    def test_ok_no_errors(self):
        self.assertTrue(self._diff.ok, f"errors: {self._diff.errors}")

    def test_no_paths_warning_absent(self):
        for w in self._diff.warnings:
            self.assertNotIn("no differentiability paths", w)

    def test_fourteen_paths_verified(self):
        self.assertEqual(len(self._diff.parameter_paths), 14,
                         f"paths: {list(self._diff.parameter_paths.keys())}")

    def test_encoder_attn_paths_go_through_attn_block(self):
        path = self._diff.parameter_paths.get("encoder_attn.Wq")
        self.assertIsNotNone(path)
        self.assertIn("AttnBlock", path)
        self.assertIn("Logits", path)

    def test_encoder_ffn_paths_go_through_ffn_block(self):
        path = self._diff.parameter_paths.get("encoder_ffn.W1")
        self.assertIsNotNone(path)
        self.assertIn("FfnBlock", path)
        self.assertIn("Logits", path)

    def test_classifier_paths_reach_logits_directly(self):
        path = self._diff.parameter_paths.get("classifier.W")
        self.assertIsNotNone(path)
        self.assertEqual(path, ["Logits"])

    def test_layer_to_node_map_correct(self):
        verifier = DifferentiabilityVerifier()
        mapping = verifier._layer_to_node_map(_program())
        self.assertEqual(mapping.get("encoder_attn"), "AttnBlock")
        self.assertEqual(mapping.get("encoder_ffn"), "FfnBlock")
        self.assertEqual(mapping.get("classifier"), "Logits")

    def test_all_paths_only_contain_graph_nodes(self):
        graph_nodes = {"Input", "AttnBlock", "FfnBlock", "Logits"}
        for key, path in self._diff.parameter_paths.items():
            for node in path:
                self.assertIn(node, graph_nodes, f"{key} path contains non-graph node {node!r}")


# ---------------------------------------------------------------------------
# Cross-entropy over raw logits (layer_call list output)
# ---------------------------------------------------------------------------

class TestCrossEntropyOverLogits(unittest.TestCase):

    def test_loss_is_finite_for_list_prediction(self):
        state = {"logits": [0.3, 0.7]}
        loss = _generic_cross_entropy_loss(state, "class_a", "logits", labels=_LABELS)
        self.assertTrue(math.isfinite(loss))

    def test_loss_not_max_sentinel_when_labels_provided(self):
        state = {"logits": [0.3, 0.7]}
        loss = _generic_cross_entropy_loss(state, "class_a", "logits", labels=_LABELS)
        # Without labels, target_label="class_a" can't be int, so prob=1e-12 → loss≈27.6
        self.assertLess(loss, 5.0, "loss should not be the sentinel 1e-12 value")

    def test_correct_class_has_lower_loss(self):
        # logit[1] > logit[0] → class_b more likely
        state = {"logits": [0.1, 0.9]}
        loss_a = _generic_cross_entropy_loss(state, "class_a", "logits", labels=_LABELS)
        loss_b = _generic_cross_entropy_loss(state, "class_b", "logits", labels=_LABELS)
        self.assertLess(loss_b, loss_a)

    def test_softmax_applied_probabilities_sum_to_one(self):
        probs = _softmax_list([0.3, 0.7])
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_backward_compat_int_string_target_no_labels(self):
        state = {"F": [0.1, 0.9]}
        loss = _generic_cross_entropy_loss(state, "1", "F")
        expected = -math.log(0.9)
        self.assertAlmostEqual(loss, expected, places=6)

    def test_backward_compat_dict_prediction(self):
        state = {"probs": {"class_a": 0.3, "class_b": 0.7}}
        loss = _generic_cross_entropy_loss(state, "class_b", "probs")
        expected = -math.log(0.7)
        self.assertAlmostEqual(loss, expected, places=6)


# ---------------------------------------------------------------------------
# GenericSupervisedTrainer on transformer dataset
# ---------------------------------------------------------------------------

class TestGenericTrainerOnTransformer(unittest.TestCase):
    """Train only classifier.* (18 scalars) for speed — O(2*18*4) = 144 forward passes."""

    @classmethod
    def setUpClass(cls):
        program = _program()
        spec = _spec()
        all_ex = _adapter().examples()
        cls._examples = all_ex[:4]
        cls._val_examples = all_ex[4:6]
        cls._labels = _LABELS

        trainer = GenericSupervisedTrainer()
        cls._result = trainer.train(
            program=program, training=spec,
            examples=cls._examples, validation_examples=cls._val_examples,
            prediction_key="logits", target_key="label",
            update_patterns=["classifier.*"],
            epochs=3, learning_rate=0.05,
            labels=cls._labels,
            vector_name=_VECTOR_NAME, vector_fields=_FIELDS,
        )

    def test_runs_without_exception(self):
        self.assertIn("epoch_trace", self._result)

    def test_trainable_keys_are_classifier_params(self):
        self.assertEqual(sorted(self._result["trainable_keys"]), ["classifier.W", "classifier.b"])

    def test_epoch_trace_has_three_entries(self):
        self.assertEqual(len(self._result["epoch_trace"]), 3)

    def test_train_loss_decreases_over_epochs(self):
        trace = self._result["epoch_trace"]
        self.assertLess(trace[-1]["train_loss"], trace[0]["train_loss"],
                        f"loss did not decrease: {[t['train_loss'] for t in trace]}")

    def test_final_loss_is_finite(self):
        for entry in self._result["epoch_trace"]:
            self.assertTrue(math.isfinite(entry["train_loss"]))
            self.assertTrue(math.isfinite(entry["validation_loss"]))

    def test_params_changed_after_training(self):
        from matrixai.parameters.store import build_initial_parameter_set
        initial = build_initial_parameter_set(_program())
        initial_W = initial.parameters["classifier.W"]["values"]
        final_W = self._result["final_params"]["classifier.W"]
        self.assertNotEqual(initial_W, final_W, "classifier.W should have changed")

    def test_best_params_present(self):
        self.assertIn("classifier.W", self._result["best_params"])
        self.assertIn("classifier.b", self._result["best_params"])


# ---------------------------------------------------------------------------
# Backward compat: Cut 4 verifier still passes, P4 models unaffected
# ---------------------------------------------------------------------------

class TestBackwardCompatAfterFixes(unittest.TestCase):

    def test_cut4_verifier_still_ok(self):
        result = _verifier_result()
        self.assertTrue(result.ok, f"errors: {result.errors}")

    def test_cut4_diff_now_has_14_paths(self):
        result = _verifier_result()
        self.assertEqual(len(result.differentiability.get("parameter_paths", {})), 14)

    def test_cut4_no_paths_warning_gone(self):
        result = _verifier_result()
        for w in result.differentiability.get("warnings", []):
            self.assertNotIn("no differentiability paths", w)

    def test_email_agent_p4_model_unaffected(self):
        email_train = _BASE / "examples" / "email-agent.supervised.mxtrain"
        if not email_train.exists():
            self.skipTest("email-agent.supervised.mxtrain not present")
        spec = parse_training_file(str(email_train))
        result = TrainingVerifier().verify(spec, base_path=str(_BASE))
        self.assertTrue(result.ok, f"email-agent training broke: {result.errors}")


if __name__ == "__main__":
    unittest.main()
