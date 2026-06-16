from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from matrixai.compiler import DifferentiablePythonCompiler
from matrixai.compiler.torch_forward import TorchForwardError, TorchForwardRunner
from matrixai.parameters import build_initial_parameter_set, torch_available
from matrixai.parser import parse_file


ROOT = Path(__file__).resolve().parents[1]


class MatrixAITorchForwardTest(unittest.TestCase):
    def _differentiable_python_result(self, model_name: str, input_name: str):
        program = parse_file(ROOT / "examples" / model_name)
        parameter_set = build_initial_parameter_set(program)
        input_data = json.loads((ROOT / "examples" / input_name).read_text(encoding="utf-8"))
        namespace: dict[str, object] = {}
        exec(DifferentiablePythonCompiler().compile(program), namespace)
        result = namespace["run"](input_data, parameter_set.runtime_parameters())
        return program, input_data, parameter_set, result

    def test_torch_forward_blocks_deferred_continuous_program_before_torch_import(self) -> None:
        program = parse_file(ROOT / "examples" / "continuous-scoring.typed.mxai")

        with self.assertRaisesRegex(TorchForwardError, "not portable to torch forward"):
            TorchForwardRunner().run(program, {})

    def test_cli_backend_run_torch_requires_optional_dependency_when_absent(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "backend-run",
                "examples/email-agent.typed.mxai",
                "--input",
                "examples/email-sample.json",
                "--target",
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
    def test_torch_forward_email_matches_differentiable_python_initial_parameters(self) -> None:
        program, input_data, parameter_set, expected = self._differentiable_python_result(
            "email-agent.typed.mxai",
            "email-sample.json",
        )

        actual = TorchForwardRunner().run(program, input_data, parameter_set)

        self.assertEqual(actual["target"], "torch")
        self.assertEqual(actual["parameter_set_id"], parameter_set.parameter_set_id)
        for label, expected_value in expected["state"]["C"].items():
            self.assertAlmostEqual(actual["state"]["C"][label], expected_value, places=6)
        self.assertEqual(actual["state"]["Confidence"]["label"], expected["state"]["Confidence"]["label"])
        self.assertAlmostEqual(actual["state"]["ReplyActivation"], expected["state"]["ReplyActivation"], places=6)
        self.assertEqual(actual["actions"][0]["activated"], expected["actions"][0]["activated"])
        self.assertIn("ReplyActivation", {boundary["node"] for boundary in actual["runtime_boundaries"]})

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_torch_forward_fall_risk_matches_differentiable_python_initial_parameters(self) -> None:
        program, input_data, parameter_set, expected = self._differentiable_python_result(
            "fall-risk.typed.mxai",
            "fall-risk-sample.json",
        )

        actual = TorchForwardRunner().run(program, input_data, parameter_set)

        self.assertEqual(actual["target"], "torch")
        self.assertAlmostEqual(actual["state"]["R"], expected["state"]["R"], places=6)
        self.assertAlmostEqual(actual["state"]["Risk"]["mean"], expected["state"]["Risk"]["mean"], places=6)
        self.assertAlmostEqual(actual["state"]["AlertActivation"], expected["state"]["AlertActivation"], places=6)
        self.assertEqual(actual["actions"][0]["activated"], expected["actions"][0]["activated"])

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_cli_backend_run_torch_outputs_forward_report(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "backend-run",
                "examples/email-agent.typed.mxai",
                "--input",
                "examples/email-sample.json",
                "--target",
                "torch",
                "--parameters",
                "initial",
                "--json",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "torch")
        self.assertEqual(payload["backend"]["target"], "torch")
        self.assertIn("C", payload["state"])


if __name__ == "__main__":
    unittest.main()
