from __future__ import annotations

import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
KELVIN_MODEL = ROOT / "examples/celsius_to_kelvin.mxai"


def _gen(prompt: str = "predecir valor continuo", **kwargs):
    from matrixai.training.generator import TrainingPromptGenerator
    return TrainingPromptGenerator().generate(prompt, model_path=KELVIN_MODEL, **kwargs)


class TestGeneratorRegressionBasic(unittest.TestCase):

    def test_generates_mse_loss(self):
        result = _gen()
        self.assertEqual(result.loss_type, "mse")
        self.assertIn("TYPE mse", result.training_text)

    def test_labels_empty_for_regression(self):
        result = _gen()
        self.assertEqual(result.labels, [])

    def test_target_defaults_to_function_output(self):
        result = _gen()
        self.assertEqual(result.target_name, "predicted_kelvin")

    def test_target_name_override(self):
        result = _gen(target_name="kelvin_out")
        self.assertEqual(result.target_name, "kelvin_out")

    def test_generates_scalar_target_no_range(self):
        result = _gen()
        self.assertIn("TARGET predicted_kelvin: Scalar", result.training_text)
        self.assertNotIn("Label[", result.training_text)

    def test_generates_scalar_target_with_range(self):
        result = _gen(target_scalar_range=(0.0, 1000.0))
        self.assertIn("TARGET predicted_kelvin: Scalar[0, 1000]", result.training_text)

    def test_generates_mae_metric_not_accuracy(self):
        result = _gen()
        self.assertIn("TYPE mae", result.training_text)
        self.assertNotIn("TYPE accuracy", result.training_text)

    def test_no_label_bracket_in_target(self):
        result = _gen()
        self.assertNotIn("Label[", result.training_text)

    def test_assumptions_mention_linear_regression(self):
        result = _gen()
        combined = " ".join(result.assumptions)
        self.assertIn("linear_regression", combined)

    def test_epochs_default_fifty(self):
        result = _gen()
        self.assertEqual(result.epochs, 50)
        self.assertIn("EPOCHS 50", result.training_text)

    def test_learning_rate_default(self):
        result = _gen()
        self.assertAlmostEqual(result.learning_rate, 0.001)
        self.assertIn("LEARNING_RATE 0.001", result.training_text)

    def test_dataset_template_numeric_target(self):
        result = _gen()
        lines = [l for l in result.dataset_template_text.strip().splitlines() if l]
        # First row is header
        header = lines[0].split(",")
        self.assertIn("predicted_kelvin", header)
        # Data rows should have numeric targets
        for line in lines[1:]:
            values = line.split(",")
            target_str = values[-1].strip()
            float(target_str)  # must be parseable as float

    def test_dataset_template_uses_scalar_range(self):
        result = _gen(target_scalar_range=(0.0, 100.0))
        lines = [l for l in result.dataset_template_text.strip().splitlines() if l]
        target_values = [float(line.split(",")[-1]) for line in lines[1:]]
        self.assertAlmostEqual(min(target_values), 0.0, places=5)
        self.assertAlmostEqual(max(target_values), 100.0, places=5)

    def test_prediction_matches_function_output(self):
        result = _gen()
        self.assertEqual(result.prediction, "predicted_kelvin")

    def test_generated_mxtrain_is_parseable_and_verifiable(self):
        from matrixai.training.parser import parse_training_text
        from matrixai.training.verifier import TrainingVerifier
        result = _gen(
            dataset_source="examples/celsius_to_kelvin.train.csv",
            target_scalar_range=(0.0, 1000.0),
        )
        spec = parse_training_text(result.training_text)
        report = TrainingVerifier().verify(spec, base_path=str(ROOT))
        self.assertTrue(report.ok, msg=report.errors)

    def test_classification_still_works(self):
        if not (ROOT / "examples/email-agent.mxai").exists():
            self.skipTest("email-agent.mxai not available")
        from matrixai.training.generator import TrainingPromptGenerator
        result = TrainingPromptGenerator().generate(
            "clasificar emails",
            model_path=ROOT / "examples/email-agent.mxai",
            labels=["support", "sales", "operations"],
        )
        self.assertEqual(result.loss_type, "cross_entropy")
        self.assertIn("TYPE accuracy", result.training_text)
        self.assertIn("Label[", result.training_text)


if __name__ == "__main__":
    unittest.main()
