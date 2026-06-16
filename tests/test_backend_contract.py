from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.compiler import BackendContractAnalyzer, DifferentiablePythonCompiler
from matrixai.parser import parse_file, parse_text
from matrixai.runtime import MatrixAIRuntime


ROOT = Path(__file__).resolve().parents[1]


class MatrixAIBackendContractTest(unittest.TestCase):
    def _load_generated_module(self, path: Path):
        spec = importlib.util.spec_from_file_location("compiled_differentiable_program", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_typed_email_example_fits_differentiable_backend_contract(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")

        report = BackendContractAnalyzer().analyze(program)

        self.assertTrue(report.ok, report.to_dict())
        self.assertIn("Email", [node.node for node in report.differentiable_nodes])
        self.assertIn("Classifier", [node.node for node in report.differentiable_nodes])
        self.assertIn("ReplyActivation", [node.node for node in report.differentiable_nodes])
        self.assertEqual(
            {(param.function, param.name, param.role, param.shape) for param in report.trainable_parameters},
            {("Classifier", "W1", "weights", (3, 8)), ("Classifier", "b1", "bias", (3,))},
        )
        self.assertEqual(report.tensor_shapes["Email"], [8])
        self.assertEqual(report.tensor_shapes["Classifier"], [3])
        self.assertEqual(report.tensor_shapes["ReplyActivation"], [])
        self.assertEqual(report.type_constraints["Email"]["fields"]["urgency"]["name"], "Score")
        self.assertEqual(report.type_constraints["Email"]["fields"]["urgency"]["range"]["min"], 0.0)
        self.assertEqual(report.type_constraints["ReplyActivation"]["output"]["type"]["name"], "ActionSignal")
        self.assertEqual(report.parameter_manifest[0]["dtype"], "float32")
        self.assertEqual(report.parameter_manifest[0]["initializer"], "deterministic_uniform")
        self.assertEqual(report.parameter_manifest[0]["initial_value"][0][0], -0.04583333)
        self.assertEqual(report.parameter_manifest[1]["initializer"], "zeros")
        self.assertEqual(report.parameter_manifest[1]["initial_value"], [0.0, 0.0, 0.0])
        self.assertTrue(report.autodiff_plan["ready"])
        self.assertEqual(report.autodiff_plan["parameterized_nodes"], ["Classifier"])
        self.assertEqual(report.autodiff_plan["runtime_boundaries"], ["Confidence", "DraftReply"])
        action = next(node for node in report.nodes if node.node == "DraftReply")
        self.assertTrue(action.supported)
        self.assertFalse(action.differentiable)

        def test_explicit_param_declarations_match_inferred_manifest(self) -> None:
                program = parse_text(
                        """PROJECT ExplicitParams

PARAM W1 Tensor[3, 2]
    TRAINABLE true
    INIT deterministic_uniform
END

PARAM b1 Vector[3]
    TRAINABLE true
    INIT zeros
END

VECTOR Email[2]
    urgency: Score
    sender_trust: Score
END

FUNCTION Classifier
    C: ProbabilityMap = softmax(W1 * Email + b1)
END

GRAPH
    Email -> Classifier
END

AUDIT
    EXPLAIN Email -> Classifier
END
"""
                )

                report = BackendContractAnalyzer().analyze(program)

                self.assertTrue(report.ok, report.to_dict())
                self.assertEqual(program.parameters[0].type_spec.parameters["shape"], [3, 2])
                self.assertEqual(program.to_dict()["parameters"][0]["type"]["parameters"]["shape"], [3, 2])
                self.assertEqual(report.parameter_errors, [])
                self.assertEqual(report.parameter_manifest[0]["shape"], [3, 2])
                self.assertEqual(report.parameter_manifest[1]["shape"], [3])

        def test_explicit_param_mismatch_blocks_backend_contract(self) -> None:
                program = parse_text(
                        """PROJECT BadExplicitParams

PARAM W1 Tensor[2, 2]
    TRAINABLE true
    INIT zeros
END

VECTOR Email[2]
    urgency: Score
    sender_trust: Score
END

FUNCTION Classifier
    C: ProbabilityMap = softmax(W1 * Email + b1)
END

GRAPH
    Email -> Classifier
END

AUDIT
    EXPLAIN Email -> Classifier
END
"""
                )

                report = BackendContractAnalyzer().analyze(program)

                self.assertFalse(report.ok)
                self.assertIn("Missing explicit PARAM for inferred trainable parameter: b1", report.parameter_errors)
                self.assertIn("Explicit PARAM W1 expected shape [3, 2], got [2, 2]", report.parameter_errors)
                self.assertIn(
                        "Explicit PARAM W1 expected initializer deterministic_uniform, got zeros",
                        report.parameter_errors,
                )

    def test_sigmoid_linear_manifest_uses_scalar_output_shape(self) -> None:
        program = parse_file(ROOT / "examples" / "fall-risk.typed.mxai")

        report = BackendContractAnalyzer().analyze(program)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.tensor_shapes["Patient"], [5])
        self.assertEqual(report.tensor_shapes["RiskModel"], [])
        self.assertEqual(
            [param.to_dict() for param in report.trainable_parameters],
            [
                {"function": "RiskModel", "name": "W1", "role": "weights", "shape": [5]},
                {"function": "RiskModel", "name": "b1", "role": "bias", "shape": []},
            ],
        )
        self.assertEqual(report.parameter_manifest[0]["initial_value"], [-0.03, -0.01, 0.01, 0.03, 0.05])
        self.assertEqual(report.parameter_manifest[1]["initial_value"], 0.0)

    def test_torch_backend_report_accepts_email_classifier(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")

        report = BackendContractAnalyzer(target="torch").analyze(program)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.target, "torch")
        self.assertEqual(report.backend["target"], "torch")
        self.assertEqual(report.backend["device"], "cpu")
        self.assertEqual(report.backend["dtype"], "float32")
        self.assertIn(report.backend["torch_available"], {True, False})
        self.assertIn("Classifier", [node.node for node in report.differentiable_nodes])
        self.assertNotIn("ReplyActivation", [node.node for node in report.differentiable_nodes])
        self.assertEqual(report.tensor_shapes["Classifier"], [3])
        self.assertEqual(report.autodiff_plan["target"], "torch")
        self.assertEqual(report.autodiff_plan["runtime_boundaries"], ["Confidence", "ReplyActivation", "DraftReply"])
        self.assertIn("P5 tensor subset", report.autodiff_plan["notes"][0])
        self.assertEqual(
            [param.to_dict() for param in report.trainable_parameters],
            [
                {"function": "Classifier", "name": "W1", "role": "weights", "shape": [3, 8]},
                {"function": "Classifier", "name": "b1", "role": "bias", "shape": [3]},
            ],
        )

    def test_torch_backend_report_accepts_fall_risk_binary_classifier(self) -> None:
        program = parse_file(ROOT / "examples" / "fall-risk.typed.mxai")

        report = BackendContractAnalyzer(target="torch").analyze(program)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.tensor_shapes["Patient"], [5])
        self.assertEqual(report.tensor_shapes["RiskModel"], [])
        self.assertEqual(report.autodiff_plan["parameterized_nodes"], ["RiskModel"])
        self.assertEqual(
            [param.to_dict() for param in report.trainable_parameters],
            [
                {"function": "RiskModel", "name": "W1", "role": "weights", "shape": [5]},
                {"function": "RiskModel", "name": "b1", "role": "bias", "shape": []},
            ],
        )

    def test_torch_backend_report_blocks_deferred_symbolic_function(self) -> None:
        program = parse_file(ROOT / "examples" / "continuous-scoring.typed.mxai")

        report = BackendContractAnalyzer(target="torch").analyze(program)

        self.assertFalse(report.ok)
        self.assertEqual(report.target, "torch")
        self.assertIn("Quality", [node.node for node in report.unsupported_nodes])
        quality = next(node for node in report.unsupported_nodes if node.node == "Quality")
        self.assertEqual(quality.kind, "normalize")
        self.assertIn("deferred", quality.reason)

    def test_discrete_argmax_blocks_differentiable_backend_contract(self) -> None:
        program = parse_text(
            """PROJECT P1Select

VECTOR Email[8]
  urgency
  sender_trust
  topic_support
  topic_sales
  sentiment
  has_attachment
  previous_interactions
  language_confidence
END

FUNCTION Scores
  scores = softmax(W1 * Email + b1)
END

FUNCTION Choice
  choice = argmax(scores)
END

GRAPH
  Email -> Scores -> Choice
END

AUDIT
  EXPLAIN Email -> Scores -> Choice
END
"""
        )

        report = BackendContractAnalyzer().analyze(program)

        self.assertFalse(report.ok)
        self.assertEqual(report.unsupported_nodes[0].node, "Choice")
        self.assertEqual(report.unsupported_nodes[0].kind, "select_argmax")
        self.assertIn("discrete", report.unsupported_nodes[0].reason)
        self.assertFalse(report.autodiff_plan["ready"])
        self.assertEqual(report.autodiff_plan["blocked_nodes"], ["Choice"])

    def test_torch_backend_report_blocks_discrete_argmax(self) -> None:
        program = parse_text(
            """PROJECT P1Select

VECTOR Email[8]
  urgency
  sender_trust
  topic_support
  topic_sales
  sentiment
  has_attachment
  previous_interactions
  language_confidence
END

FUNCTION Scores
  scores = softmax(W1 * Email + b1)
END

FUNCTION Choice
  choice = argmax(scores)
END

GRAPH
  Email -> Scores -> Choice
END

AUDIT
  EXPLAIN Email -> Scores -> Choice
END
"""
        )

        report = BackendContractAnalyzer(target="torch").analyze(program)

        self.assertFalse(report.ok)
        self.assertEqual(report.unsupported_nodes[0].node, "Choice")
        self.assertEqual(report.unsupported_nodes[0].kind, "select_argmax")
        self.assertIn("discrete", report.unsupported_nodes[0].reason)
        self.assertFalse(report.autodiff_plan["ready"])
        self.assertEqual(report.autodiff_plan["blocked_nodes"], ["Choice"])

    def test_cli_backend_report_json_returns_trainable_manifest(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "backend-report",
                "examples/email-agent.typed.mxai",
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
        self.assertEqual(payload["target"], "differentiable_python")
        self.assertIn("Classifier", payload["differentiable_nodes"])
        self.assertEqual(
            payload["trainable_parameters"],
            [
                {"function": "Classifier", "name": "W1", "role": "weights", "shape": [3, 8]},
                {"function": "Classifier", "name": "b1", "role": "bias", "shape": [3]},
            ],
        )
        self.assertEqual(payload["tensor_shapes"]["Email"], [8])
        self.assertEqual(payload["tensor_shapes"]["ReplyActivation"], [])
        self.assertEqual(payload["type_constraints"]["Email"]["fields"]["language_confidence"]["name"], "Confidence")
        self.assertEqual(payload["type_constraints"]["ReplyActivation"]["output"]["type"]["range"]["max"], 1.0)
        self.assertEqual(payload["parameter_manifest"][0]["shape"], [3, 8])
        self.assertEqual(payload["parameter_manifest"][0]["dtype"], "float32")
        self.assertEqual(payload["parameter_manifest"][1]["initial_value"], [0.0, 0.0, 0.0])
        self.assertEqual(payload["autodiff_plan"]["status"], "metadata_only")
        self.assertEqual(payload["autodiff_plan"]["parameterized_nodes"], ["Classifier"])

    def test_cli_backend_report_torch_json_returns_optional_backend_metadata(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "backend-report",
                "examples/email-agent.typed.mxai",
                "--target",
                "torch",
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
        self.assertEqual(payload["target"], "torch")
        self.assertEqual(payload["backend"]["target"], "torch")
        self.assertEqual(payload["backend"]["execution"], "training_minimal")
        self.assertIn(payload["backend"]["torch_available"], {True, False})
        self.assertEqual(payload["autodiff_plan"]["runtime_boundaries"], ["Confidence", "ReplyActivation", "DraftReply"])

    def test_cli_backend_parameters_torch_target_outputs_manifest(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "backend-parameters",
                "examples/email-agent.typed.mxai",
                "--target",
                "torch",
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
        self.assertEqual(payload["parameter_manifest"][0]["shape"], [3, 8])
        self.assertEqual(payload["parameter_set"]["parameters"]["b1"]["shape"], [3])

    def test_differentiable_python_compiler_runs_typed_email_boundaries(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))

        source = DifferentiablePythonCompiler().compile(program)

        self.assertIn("TARGET = 'differentiable_python'", source)
        self.assertIn("TYPE_CONSTRAINTS =", source)
        self.assertIn("def initial_parameters()", source)
        self.assertIn("def validate_parameters(parameters: dict[str, Any]) -> list[str]", source)
        self.assertIn("def run(input_data: dict[str, Any], parameters: dict[str, Any] | None = None)", source)
        self.assertNotIn("from matrixai", source)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_differentiable_email.py"
            output_path.write_text(source, encoding="utf-8")
            module = self._load_generated_module(output_path)

            generated = module.run(input_data)

        runtime = MatrixAIRuntime().run(program, input_data)
        boundary_nodes = {boundary["node"] for boundary in generated["runtime_boundaries"]}

        self.assertEqual(generated["target"], "differentiable_python")
        self.assertAlmostEqual(
            generated["state"]["ReplyActivation"],
            runtime["state"]["ReplyActivation"],
        )
        self.assertEqual(generated["actions"], runtime["actions"])
        self.assertEqual(boundary_nodes, {"Confidence", "DraftReply"})
        self.assertEqual(generated["tensor_shapes"]["Classifier"], [3])
        self.assertEqual(generated["tensor_shapes"]["ReplyActivation"], [])
        self.assertEqual(generated["type_constraints"]["Email"]["fields"]["sender_trust"]["name"], "Score")
        self.assertEqual(generated["type_constraints"]["Classifier"]["output"]["type"]["name"], "ProbabilityMap")
        self.assertEqual(generated["parameter_manifest"][0]["initializer"], "deterministic_uniform")
        self.assertEqual(generated["parameter_manifest"][0]["initial_value"][-1][-1], 0.05)
        self.assertEqual(generated["autodiff_plan"]["runtime_boundaries"], ["Confidence", "DraftReply"])
        self.assertEqual(generated["autodiff_plan"]["status"], "metadata_only")
        self.assertEqual(
            generated["trainable_parameters"],
            [
                {"function": "Classifier", "name": "W1", "role": "weights", "shape": [3, 8]},
                {"function": "Classifier", "name": "b1", "role": "bias", "shape": [3]},
            ],
        )

    def test_differentiable_python_runs_continuous_p1_typed_example(self) -> None:
        program = parse_file(ROOT / "examples" / "continuous-scoring.typed.mxai")
        input_data = json.loads(
            (ROOT / "examples" / "continuous-scoring-sample.json").read_text(encoding="utf-8")
        )

        report = BackendContractAnalyzer().analyze(program)
        source = DifferentiablePythonCompiler().compile(program)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(
            [function.semantic.kind for function in program.functions],
            ["symbolic_weighted_sum", "normalize", "aggregate_mean"],
        )
        self.assertEqual(report.tensor_shapes["Candidate"], [4])
        self.assertEqual(report.tensor_shapes["FinalScore"], [])
        self.assertEqual(report.tensor_shapes["Quality"], [])
        self.assertEqual(report.tensor_shapes["Readiness"], [])
        self.assertEqual(report.type_constraints["Candidate"]["fields"]["raw_quality"]["range"]["max"], 10.0)
        self.assertEqual(report.type_constraints["Readiness"]["output"]["type"]["name"], "Score")
        self.assertEqual(report.trainable_parameters, [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_continuous_scoring.py"
            output_path.write_text(source, encoding="utf-8")
            module = self._load_generated_module(output_path)
            generated = module.run(input_data)

        runtime = MatrixAIRuntime().run(program, input_data)

        self.assertEqual(generated["state"], runtime["state"])
        self.assertEqual(generated["runtime_boundaries"], [])
        self.assertEqual(generated["trainable_parameters"], [])
        self.assertAlmostEqual(generated["state"]["final_score"], 0.74, places=6)
        self.assertAlmostEqual(generated["state"]["quality"], 0.5, places=6)
        self.assertAlmostEqual(generated["state"]["readiness"], (0.74 + 0.5 + 0.9) / 3, places=6)

    def test_differentiable_python_run_accepts_initial_parameters(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_differentiable_email.py"
            output_path.write_text(DifferentiablePythonCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)

            default_result = module.run(input_data)
            parameterized_result = module.run(input_data, module.initial_parameters())

        self.assertIn("Classifier.W1", module.initial_parameters())
        self.assertNotEqual(default_result["state"]["C"], parameterized_result["state"]["C"])
        self.assertEqual(set(default_result["state"]["C"]), set(parameterized_result["state"]["C"]))
        self.assertEqual(parameterized_result["target"], "differentiable_python")

    def test_differentiable_python_rejects_invalid_parameter_shape(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_differentiable_email.py"
            output_path.write_text(DifferentiablePythonCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)

            parameters = module.initial_parameters()
            parameters["W1"] = [[0.0 for _ in range(8)]]

            self.assertEqual(
                module.validate_parameters(parameters),
                ["Parameter W1 expected shape [3, 8], got [1, 8]"],
            )
            with self.assertRaisesRegex(ValueError, r"Parameter W1 expected shape \[3, 8\]"):
                module.run(input_data, parameters)

    def test_differentiable_python_reports_ragged_parameter_errors(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_differentiable_email.py"
            output_path.write_text(DifferentiablePythonCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)

            errors = module.validate_parameters({"Classifier.W1": [[0.0], [0.0, 0.1], [0.0]]})

        self.assertEqual(errors, ["Parameter Classifier.W1 invalid: Parameter contains ragged values"])

    def test_differentiable_python_compiler_rejects_discrete_argmax(self) -> None:
        program = parse_text(
            """PROJECT P1Select

VECTOR Email[8]
  urgency
  sender_trust
  topic_support
  topic_sales
  sentiment
  has_attachment
  previous_interactions
  language_confidence
END

FUNCTION Scores
  scores = softmax(W1 * Email + b1)
END

FUNCTION Choice
  choice = argmax(scores)
END

GRAPH
  Email -> Scores -> Choice
END

AUDIT
  EXPLAIN Email -> Scores -> Choice
END
"""
        )

        with self.assertRaisesRegex(ValueError, "not portable"):
            DifferentiablePythonCompiler().compile(program)

    def test_cli_compile_differentiable_python_outputs_module(self) -> None:
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_differentiable_email.py"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "compile",
                    "examples/email-agent.typed.mxai",
                    "--target",
                    "differentiable-python",
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            module = self._load_generated_module(output_path)
            generated = module.run(input_data)

        self.assertEqual(generated["target"], "differentiable_python")
        self.assertTrue(generated["backend_report"]["ok"])
        self.assertEqual(generated["tensor_shapes"]["Email"], [8])
        self.assertEqual(generated["type_constraints"]["Email"]["fields"]["urgency"]["range"]["max"], 1.0)
        self.assertEqual(generated["parameter_manifest"][0]["name"], "W1")
        self.assertEqual(generated["runtime_boundaries"][0]["node"], "Confidence")

    def test_cli_backend_run_supports_initial_parameters(self) -> None:
        base_cmd = [
            sys.executable,
            "-m",
            "matrixai",
            "backend-run",
            "examples/email-agent.typed.mxai",
            "--input",
            "examples/email-sample.json",
            "--json",
        ]
        default_result = subprocess.run(
            base_cmd,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        initial_result = subprocess.run(
            [*base_cmd, "--parameters", "initial"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(default_result.returncode, 0, default_result.stderr)
        self.assertEqual(initial_result.returncode, 0, initial_result.stderr)
        default_payload = json.loads(default_result.stdout)
        initial_payload = json.loads(initial_result.stdout)

        self.assertEqual(default_payload["target"], "differentiable_python")
        self.assertEqual(initial_payload["target"], "differentiable_python")
        self.assertNotEqual(default_payload["state"]["C"], initial_payload["state"]["C"])
        self.assertEqual(initial_payload["parameter_manifest"][0]["name"], "W1")
        self.assertEqual(initial_payload["runtime_boundaries"][0]["node"], "Confidence")

    def test_cli_backend_run_rejects_invalid_parameter_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parameter_path = Path(tmp_dir) / "bad-parameters.json"
            parameter_path.write_text(
                json.dumps({"W1": [[0.0 for _ in range(8)]], "b1": [0.0, 0.0, 0.0]}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "backend-run",
                    "examples/email-agent.typed.mxai",
                    "--input",
                    "examples/email-sample.json",
                    "--parameters",
                    str(parameter_path),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Parameter W1 expected shape [3, 8]", result.stderr)

    def test_cli_backend_parameters_outputs_initial_parameters(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "backend-parameters",
                "examples/email-agent.typed.mxai",
                "--json",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "differentiable_python")
        self.assertEqual(payload["project"], "EmailAgentTyped")
        self.assertEqual(payload["parameter_manifest"][0]["shape"], [3, 8])
        self.assertIn("W1", payload["parameters"])
        self.assertIn("Classifier.W1", payload["parameters"])

    def test_cli_backend_parameters_validates_parameter_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parameter_path = Path(tmp_dir) / "bad-parameters.json"
            parameter_path.write_text(json.dumps({"W1": [[0.0]]}), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "backend-parameters",
                    "examples/email-agent.typed.mxai",
                    "--validate",
                    str(parameter_path),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["errors"], ["Parameter W1 expected shape [3, 8], got [1, 1]"])


if __name__ == "__main__":
    unittest.main()
