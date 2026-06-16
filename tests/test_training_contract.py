from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.parser import parse_file
from matrixai.training import DifferentiabilityVerifier, TrainingVerifier, parse_training_file, parse_training_text


ROOT = Path(__file__).resolve().parents[1]


class MatrixAITrainingContractTest(unittest.TestCase):
    def test_mxtrain_parser_accepts_email_training_spec(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")

        self.assertEqual(spec.model, "examples/email-agent.typed.mxai")
        self.assertEqual(spec.dataset.name, "EmailTrainingSet")
        self.assertEqual(spec.dataset.source_kind, "csv")
        self.assertEqual(spec.dataset.input.vector, "Email")
        self.assertEqual(len(spec.dataset.input.columns), 8)
        self.assertEqual(spec.dataset.target.name, "label")
        self.assertEqual(spec.dataset.target.type.name, "Label")
        self.assertEqual(spec.dataset.target.type.parameters["args"], ["support", "sales", "operations"])
        self.assertEqual(spec.loss.type, "cross_entropy")
        self.assertEqual(spec.loss.prediction, "C")
        self.assertEqual(spec.optimizer.type, "sgd")
        self.assertEqual(spec.optimizer.update, ["W1", "b1"])
        self.assertEqual(spec.metrics[0].type, "accuracy")
        self.assertEqual(spec.run.epochs, 20)

    def test_mxtrain_parser_accepts_fall_risk_binary_training_spec(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.supervised.mxtrain")

        self.assertEqual(spec.model, "examples/fall-risk.typed.mxai")
        self.assertEqual(spec.dataset.name, "FallRiskTrainingSet")
        self.assertEqual(spec.dataset.source_kind, "csv")
        self.assertEqual(spec.dataset.input.vector, "Patient")
        self.assertEqual(len(spec.dataset.input.columns), 5)
        self.assertEqual(spec.dataset.target.name, "risk_label")
        self.assertEqual(spec.dataset.target.type.name, "Label")
        self.assertEqual(spec.dataset.target.type.parameters["args"], ["low", "high"])
        self.assertEqual(spec.loss.type, "binary_cross_entropy")
        self.assertEqual(spec.loss.prediction, "R")
        self.assertEqual(spec.optimizer.type, "sgd")
        self.assertEqual(spec.optimizer.update, ["W1", "b1"])
        self.assertEqual(spec.run.epochs, 30)

    def test_mxtrain_parser_accepts_fall_risk_probability_target_spec(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.probability.mxtrain")

        self.assertEqual(spec.model, "examples/fall-risk.typed.mxai")
        self.assertEqual(spec.dataset.name, "FallRiskProbabilityTrainingSet")
        self.assertEqual(spec.dataset.source_kind, "csv")
        self.assertEqual(spec.dataset.input.vector, "Patient")
        self.assertEqual(spec.dataset.target.name, "risk_probability")
        self.assertEqual(spec.dataset.target.type.name, "Probability")
        self.assertEqual(spec.loss.type, "binary_cross_entropy")
        self.assertEqual(spec.loss.target, "risk_probability")

    def test_validate_training_accepts_email_training_spec(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")

        report = TrainingVerifier().verify(spec, base_path=ROOT)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(
            report.trainable_parameters,
            [
                {"function": "Classifier", "name": "W1", "role": "weights", "shape": [3, 8]},
                {"function": "Classifier", "name": "b1", "role": "bias", "shape": [3]},
            ],
        )
        self.assertTrue(report.differentiability["ok"])
        self.assertEqual(report.differentiability["prediction_node"], "Classifier")
        self.assertEqual(report.differentiability["parameter_paths"]["W1"], ["Classifier"])
        self.assertEqual(report.differentiability["parameter_paths"]["b1"], ["Classifier"])

    def test_validate_training_accepts_fall_risk_binary_training_spec(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.supervised.mxtrain")

        report = TrainingVerifier().verify(spec, base_path=ROOT)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(
            report.trainable_parameters,
            [
                {"function": "RiskModel", "name": "W1", "role": "weights", "shape": [5]},
                {"function": "RiskModel", "name": "b1", "role": "bias", "shape": []},
            ],
        )
        self.assertTrue(report.differentiability["ok"])
        self.assertEqual(report.differentiability["prediction_node"], "RiskModel")
        self.assertEqual(report.differentiability["parameter_paths"], {"W1": ["RiskModel"], "b1": ["RiskModel"]})

    def test_validate_training_accepts_fall_risk_probability_target_spec(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.probability.mxtrain")

        report = TrainingVerifier().verify(spec, base_path=ROOT)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(
            report.trainable_parameters,
            [
                {"function": "RiskModel", "name": "W1", "role": "weights", "shape": [5]},
                {"function": "RiskModel", "name": "b1", "role": "bias", "shape": []},
            ],
        )
        self.assertTrue(report.differentiability["ok"])

    def test_binary_cross_entropy_requires_two_label_values(self) -> None:
        text = (ROOT / "examples" / "fall-risk.supervised.mxtrain").read_text(encoding="utf-8")
        spec = parse_training_text(text.replace("Label[low, high]", "Label[low, medium, high]"))

        report = TrainingVerifier().verify(spec, base_path=ROOT)

        self.assertFalse(report.ok)
        self.assertIn("binary_cross_entropy expects exactly two Label[...] target values", report.errors)

    def test_binary_cross_entropy_rejects_probability_target_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset = tmp / "bad_probability.csv"
            dataset.write_text(
                "age,mobility,medication_load,previous_falls,cognitive_state,risk_probability\n"
                "0.88,0.82,0.78,1.0,0.76,1.2\n",
                encoding="utf-8",
            )
            text = (ROOT / "examples" / "fall-risk.probability.mxtrain").read_text(encoding="utf-8")
            text = text.replace("examples/fall-risk.typed.mxai", str(ROOT / "examples" / "fall-risk.typed.mxai"))
            text = text.replace("examples/fall-risk.probability.train.csv", str(dataset))
            spec = parse_training_text(text)

            report = TrainingVerifier().verify(spec, base_path=tmp)

        self.assertFalse(report.ok)
        self.assertTrue(any("target risk_probability expects Probability range" in error for error in report.errors))

    def test_differentiability_verifier_reports_parameter_paths(self) -> None:
        training = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")

        report = DifferentiabilityVerifier().verify(training, program)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.prediction_node, "Classifier")
        self.assertEqual(report.parameter_paths, {"W1": ["Classifier"], "b1": ["Classifier"]})

    def test_loss_target_missing_fails(self) -> None:
        text = (ROOT / "examples" / "email-agent.supervised.mxtrain").read_text(encoding="utf-8")
        spec = parse_training_text(text.replace("TARGET label\nEND\n\nOPTIMIZER", "TARGET should_reply\nEND\n\nOPTIMIZER"))

        report = TrainingVerifier().verify(spec, base_path=ROOT)

        self.assertFalse(report.ok)
        self.assertIn("LOSS target should_reply does not match DATASET target label", report.errors)

    def test_update_unknown_parameter_fails(self) -> None:
        text = (ROOT / "examples" / "email-agent.supervised.mxtrain").read_text(encoding="utf-8")
        spec = parse_training_text(text.replace("UPDATE W1, b1", "UPDATE W2, b1"))

        report = TrainingVerifier().verify(spec, base_path=ROOT)

        self.assertFalse(report.ok)
        self.assertIn("UPDATE parameter is not trainable in MODEL: W2", report.errors)

    def test_non_differentiable_training_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            model = tmp / "discrete-choice.mxai"
            model.write_text(
                """PROJECT DiscreteChoice

VECTOR Email[8]
  urgency: Score
  sender_trust: Score
  topic_support: Probability
  topic_sales: Probability
  sentiment: Score
  has_attachment: Probability
  previous_interactions: Score
  language_confidence: Confidence
END

FUNCTION Classifier
  C: ProbabilityMap = softmax(W1 * Email + b1)
END

FUNCTION Choice
  choice = argmax(C)
END

GRAPH
  Email -> Classifier -> Choice
END

AUDIT
  EXPLAIN Email -> Classifier -> Choice
END
""",
                encoding="utf-8",
            )
            text = (ROOT / "examples" / "email-agent.supervised.mxtrain").read_text(encoding="utf-8")
            text = text.replace("examples/email-agent.typed.mxai", str(model))
            text = text.replace("examples/email-agent.train.csv", str(ROOT / "examples" / "email-agent.train.csv"))
            text = text.replace("PREDICTION C", "PREDICTION choice")
            spec = parse_training_text(text)

            report = TrainingVerifier().verify(spec, base_path=tmp)

        self.assertFalse(report.ok)
        self.assertTrue(any("LOSS prediction node is not differentiable: Choice" in error for error in report.errors))
        self.assertTrue(
            any("Differentiability path for W1 is blocked by unsupported node Choice" in error for error in report.errors)
        )

    def test_probability_range_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset = tmp / "bad.csv"
            dataset.write_text(
                "urgency,sender_trust,topic_support,topic_sales,sentiment,has_attachment,previous_interactions,language_confidence,label\n"
                "1.5,0.8,1.0,0.0,0.7,0.0,0.6,0.98,support\n",
                encoding="utf-8",
            )
            text = (ROOT / "examples" / "email-agent.supervised.mxtrain").read_text(encoding="utf-8")
            text = text.replace("examples/email-agent.typed.mxai", str(ROOT / "examples" / "email-agent.typed.mxai"))
            text = text.replace("examples/email-agent.train.csv", str(dataset))
            spec = parse_training_text(text)

            report = TrainingVerifier().verify(spec, base_path=tmp)

        self.assertFalse(report.ok)
        self.assertTrue(any("field urgency expects Score range" in error for error in report.errors))

    def test_label_domain_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset = tmp / "bad_label.csv"
            dataset.write_text(
                "urgency,sender_trust,topic_support,topic_sales,sentiment,has_attachment,previous_interactions,language_confidence,label\n"
                "0.5,0.8,1.0,0.0,0.7,0.0,0.6,0.98,legal\n",
                encoding="utf-8",
            )
            text = (ROOT / "examples" / "email-agent.supervised.mxtrain").read_text(encoding="utf-8")
            text = text.replace("examples/email-agent.typed.mxai", str(ROOT / "examples" / "email-agent.typed.mxai"))
            text = text.replace("examples/email-agent.train.csv", str(dataset))
            spec = parse_training_text(text)

            report = TrainingVerifier().verify(spec, base_path=tmp)

        self.assertFalse(report.ok)
        self.assertTrue(any("target label must be one of" in error for error in report.errors))

    def test_cli_validate_training_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "validate-training",
                "examples/email-agent.supervised.mxtrain",
                "--json",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["training"]["loss"]["type"], "cross_entropy")
        self.assertEqual(payload["trainable_parameters"][0]["shape"], [3, 8])


if __name__ == "__main__":
    unittest.main()
