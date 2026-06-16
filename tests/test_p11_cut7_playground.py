"""P11 Cut 7 — playground integration for layer_call (transformer) models."""
from __future__ import annotations

import time
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_MXAI = (_BASE / "examples" / "transformer-classifier-vector.mxai").read_text(encoding="utf-8")
_TRAINING = (_BASE / "examples" / "transformer-classifier-vector.mxtrain").read_text(encoding="utf-8")
_CSV = (_BASE / "examples" / "transformer-classifier-vector.train.csv").read_text(encoding="utf-8")


class TestP11Cut7PlaygroundIntegration(unittest.TestCase):
    # Run training once for all tests to keep the suite fast.
    @classmethod
    def setUpClass(cls):
        from matrixai.playground import _run_playground_training
        cls.result = _run_playground_training(_MXAI, _TRAINING, _CSV, epochs_override=2)

    # ------------------------------------------------------------------
    # EXAMPLES registry and helper detection
    # ------------------------------------------------------------------

    def test_transformer_classifier_in_examples_dict(self):
        from matrixai.playground import EXAMPLES
        self.assertIn("transformer-classifier", EXAMPLES)
        entry = EXAMPLES["transformer-classifier"]
        self.assertIn("model", entry)
        self.assertIn("training", entry)
        self.assertIn("transformer-classifier-vector.mxai", entry["model"])

    def test_transformer_classifier_in_project_example_index(self):
        from matrixai.playground import PROJECT_EXAMPLE_INDEX
        self.assertIn("transformer_classifier", PROJECT_EXAMPLE_INDEX)
        self.assertEqual(PROJECT_EXAMPLE_INDEX["transformer_classifier"], "transformer-classifier")

    def test_get_prediction_kind_layer_call_for_transformer(self):
        from matrixai.playground import _get_prediction_kind
        kind = _get_prediction_kind(_MXAI, _TRAINING)
        self.assertEqual(kind, "layer_call")

    def test_get_prediction_kind_softmax_linear_for_fall_risk(self):
        from matrixai.playground import _get_prediction_kind
        mxai = (_BASE / "examples" / "fall-risk.typed.mxai").read_text(encoding="utf-8")
        train = (_BASE / "examples" / "fall-risk.supervised.mxtrain").read_text(encoding="utf-8")
        kind = _get_prediction_kind(mxai, train)
        self.assertNotEqual(kind, "layer_call")

    def test_get_prediction_kind_returns_empty_on_invalid_input(self):
        from matrixai.playground import _get_prediction_kind
        self.assertEqual(_get_prediction_kind("not valid", "not valid"), "")

    # ------------------------------------------------------------------
    # Synchronous training result shape
    # ------------------------------------------------------------------

    def test_run_playground_training_generic_returns_ok(self):
        self.assertTrue(self.result.get("ok"), self.result.get("error"))

    def test_run_playground_training_generic_has_two_epochs(self):
        epochs = self.result.get("epochs", [])
        self.assertEqual(len(epochs), 2)

    def test_run_playground_training_generic_epoch_has_losses(self):
        epoch = self.result["epochs"][0]
        self.assertIn("train_loss", epoch)
        self.assertIn("validation_loss", epoch)

    def test_run_playground_training_generic_best_epoch_positive(self):
        self.assertGreater(self.result["best_epoch"], 0)

    def test_run_playground_training_generic_params_best_has_id(self):
        pb = self.result.get("params_best", {})
        self.assertIn("parameter_set_id", pb)

    def test_run_playground_training_generic_params_best_has_transformer_keys(self):
        pb = self.result.get("params_best", {})
        params = pb.get("parameters", {})
        # At least one hierarchical key from the transformer model must be present
        has_transformer_key = any(
            "." in k for k in params
        )
        self.assertTrue(has_transformer_key, f"No hierarchical key found in: {list(params.keys())[:5]}")

    def test_run_playground_training_generic_has_evaluation_report(self):
        er = self.result.get("evaluation_report")
        self.assertIsNotNone(er)
        self.assertIn("accuracy", er)

    def test_run_playground_training_generic_backend_stdlib(self):
        self.assertEqual(self.result.get("backend"), "stdlib")

    def test_run_playground_training_generic_has_metrics_field(self):
        self.assertIn("metrics", self.result)
        self.assertIn("epochs", self.result["metrics"])

    # ------------------------------------------------------------------
    # Async submit path (layer_call routing in _submit_training_job)
    # ------------------------------------------------------------------

    def test_submit_training_job_layer_call_returns_job_id(self):
        from matrixai.playground import _submit_training_job
        r = _submit_training_job(_MXAI, _TRAINING, _CSV, epochs_override=2)
        self.assertTrue(r.get("ok"), r.get("error"))
        self.assertIn("job_id", r)

    def test_submit_training_job_layer_call_completes(self):
        from matrixai.playground import _submit_training_job, _get_job_status
        r = _submit_training_job(_MXAI, _TRAINING, _CSV, epochs_override=2)
        self.assertTrue(r.get("ok"), r.get("error"))
        job_id = r["job_id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            status = _get_job_status(job_id)
            if status["status"] in ("done", "error", "cancelled", "timeout"):
                break
            time.sleep(0.5)
        status = _get_job_status(job_id)
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertIn("params_best", status)

    # ------------------------------------------------------------------
    # P4/P5 backward compat: fall-risk still uses SupervisedTrainer path
    # ------------------------------------------------------------------

    def test_fall_risk_training_unaffected(self):
        from matrixai.playground import _run_playground_training, _generate_training_from_mxai
        mxai = (_BASE / "examples" / "fall-risk.typed.mxai").read_text(encoding="utf-8")
        csv = (_BASE / "examples" / "fall-risk.train.csv").read_text(encoding="utf-8")
        gen = _generate_training_from_mxai(mxai)
        self.assertTrue(gen.get("ok"), gen.get("error"))
        r = _run_playground_training(mxai, gen["training_text"], csv, epochs_override=2)
        self.assertTrue(r.get("ok"), r.get("error"))
        self.assertEqual(len(r["epochs"]), 2)


if __name__ == "__main__":
    unittest.main()
