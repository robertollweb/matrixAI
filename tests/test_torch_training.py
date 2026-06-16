from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.parameters import build_initial_parameter_set, load_parameter_set, torch_available, validate_parameter_set
from matrixai.parser import parse_file
from matrixai.training import TorchSupervisedTrainer, parse_training_file


ROOT = Path(__file__).resolve().parents[1]


class MatrixAITorchTrainingTest(unittest.TestCase):
    def test_cli_train_torch_requires_optional_dependency_when_absent(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_torch_missing"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "train",
                    "examples/email-agent.typed.mxai",
                    "--training",
                    "examples/email-agent.supervised.mxtrain",
                    "--output",
                    str(output_dir),
                    "--backend",
                    "torch",
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("optional dependency PyTorch", result.stderr)

    def test_cli_evaluate_torch_requires_optional_dependency_when_absent(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment")
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        with tempfile.TemporaryDirectory() as tmp_dir:
            parameter_path = Path(tmp_dir) / "params.initial.json"
            parameter_path.write_text(
                json.dumps(build_initial_parameter_set(program).to_dict()),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "evaluate",
                    "examples/email-agent.typed.mxai",
                    "--training",
                    "examples/email-agent.supervised.mxtrain",
                    "--params",
                    str(parameter_path),
                    "--data",
                    "examples/email-agent.test.csv",
                    "--backend",
                    "torch",
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("optional dependency PyTorch", result.stderr)

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_torch_train_softmax_linear_writes_p4_artifacts(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_torch_001"
            result = TorchSupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))
            initial = load_parameter_set(output_dir / "params.initial.json")
            best = load_parameter_set(output_dir / "params.best.json")
            final = load_parameter_set(output_dir / "params.final.json")

        self.assertEqual(result.run_id, "email_torch_001")
        self.assertEqual(trace["backend"]["target"], "torch")
        self.assertEqual(trace["backend"]["trainer"], "TorchSupervisedTrainer")
        self.assertEqual(trace["backend_report"]["target"], "torch")
        self.assertEqual(trace["backend_report"]["backend"]["execution"], "training_minimal")
        self.assertEqual(best.metrics["backend"], "torch")
        self.assertEqual(best.source, "trained_torch")
        self.assertTrue(validate_parameter_set(program, best).ok)
        self.assertEqual(final.parameters["W1"]["shape"], [3, 8])
        self.assertNotEqual(initial.parameters["W1"]["values"], final.parameters["W1"]["values"])
        self.assertIn("training_trace", result.artifacts)

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_torch_train_sigmoid_linear_binary_cross_entropy_writes_valid_params(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.supervised.mxtrain")
        program = parse_file(ROOT / "examples" / "fall-risk.typed.mxai")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "fall_risk_torch_001"
            result = TorchSupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "fall-risk.supervised.mxtrain",
            )
            trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))
            best = load_parameter_set(output_dir / "params.best.json")
            final = load_parameter_set(output_dir / "params.final.json")

        self.assertEqual(result.run_id, "fall_risk_torch_001")
        self.assertEqual(trace["loss"], "binary_cross_entropy")
        self.assertEqual(trace["backend"]["loss"], "binary_cross_entropy")
        self.assertTrue(validate_parameter_set(program, best).ok)
        self.assertEqual(final.parameters["W1"]["shape"], [5])
        self.assertEqual(final.parameters["b1"]["shape"], [])
        self.assertIsInstance(final.parameters["b1"]["values"], float)

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_torch_train_sigmoid_linear_probability_target_writes_valid_params(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.probability.mxtrain")
        program = parse_file(ROOT / "examples" / "fall-risk.typed.mxai")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "fall_risk_probability_torch_001"
            TorchSupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "fall-risk.probability.mxtrain",
            )
            trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))
            best = load_parameter_set(output_dir / "params.best.json")

        self.assertEqual(trace["backend"]["target"], "torch")
        self.assertEqual(trace["loss"], "binary_cross_entropy")
        self.assertTrue(validate_parameter_set(program, best).ok)


    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_evaluate_torch_backend_matches_stdlib(self):
        from matrixai.training.torch_evaluator import TorchSupervisedEvaluator
        from matrixai.training.trainer import SupervisedEvaluator
        from matrixai.training.spec import EvaluationResult
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.training.parser import parse_training_file

        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        training = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        parameter_set = build_initial_parameter_set(program, "email_torch_eval_initial")

        torch_result = TorchSupervisedEvaluator().evaluate(
            training,
            parameter_set,
            data_path=ROOT / "examples" / "email-agent.test.csv",
            base_path=ROOT,
        )
        stdlib_result = SupervisedEvaluator().evaluate(
            training,
            parameter_set,
            data_path=ROOT / "examples" / "email-agent.test.csv",
            base_path=ROOT,
        )

        self.assertIsInstance(torch_result, EvaluationResult)
        self.assertEqual(torch_result.rows, stdlib_result.rows)
        self.assertAlmostEqual(torch_result.loss, stdlib_result.loss, places=5)
        self.assertEqual(torch_result.accuracy, stdlib_result.accuracy)
        self.assertEqual(torch_result.backend["target"], "torch")
        self.assertEqual(torch_result.backend["evaluator"], "TorchSupervisedEvaluator")

if __name__ == "__main__":
    unittest.main()
