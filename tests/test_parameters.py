from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.parameters import (
    ParameterSet,
    ParameterStore,
    TensorParameterBridge,
    TensorParameterBridgeError,
    build_initial_parameter_set,
    load_parameter_set,
    parameter_set_to_torch_tensors,
    torch_available,
    torch_tensors_to_parameter_set,
    validate_parameter_set,
    validate_parameter_set_for_torch,
)
from matrixai.parser import parse_file


ROOT = Path(__file__).resolve().parents[1]


class MatrixAIParameterStoreTest(unittest.TestCase):
    def test_build_initial_parameter_set_from_backend_manifest(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")

        parameter_set = build_initial_parameter_set(program, parameter_set_id="email_initial")

        self.assertEqual(parameter_set.parameter_set_id, "email_initial")
        self.assertTrue(parameter_set.model_hash.startswith("mxai_"))
        self.assertTrue(parameter_set.parameter_schema_hash.startswith("params_"))
        self.assertEqual(parameter_set.parameters["W1"]["type"], "Tensor[3,8]")
        self.assertEqual(parameter_set.parameters["W1"]["shape"], [3, 8])
        self.assertEqual(parameter_set.parameters["b1"]["type"], "Vector[3]")
        self.assertEqual(parameter_set.parameters["b1"]["values"], [0.0, 0.0, 0.0])
        self.assertIn("Classifier.W1", parameter_set.runtime_parameters())

    def test_parameter_store_roundtrip(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        parameter_set = build_initial_parameter_set(program)

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ParameterStore(tmp_dir)
            path = store.write("params.initial.json", parameter_set)
            loaded = store.load("params.initial.json")

        self.assertEqual(path.name, "params.initial.json")
        self.assertEqual(loaded.to_dict(), parameter_set.to_dict())
        self.assertTrue(validate_parameter_set(program, loaded).ok)

    def test_wrong_parameter_schema_fails(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        data = build_initial_parameter_set(program).to_dict()
        data["parameter_schema_hash"] = "params_wrong"

        report = validate_parameter_set(program, ParameterSet.from_dict(data))

        self.assertFalse(report.ok)
        self.assertTrue(any("parameter_schema_hash mismatch" in error for error in report.errors))

    def test_wrong_parameter_value_shape_fails(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        data = build_initial_parameter_set(program).to_dict()
        data["parameters"]["W1"]["values"] = [[0.0 for _ in range(8)]]

        report = validate_parameter_set(program, ParameterSet.from_dict(data))

        self.assertFalse(report.ok)
        self.assertIn("Parameter W1 expected values shape [3, 8], got [1, 8]", report.errors)

    def test_cli_init_parameters_writes_versioned_parameter_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "params.initial.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "init-parameters",
                    "examples/email-agent.typed.mxai",
                    "--output",
                    str(output_path),
                    "--parameter-set-id",
                    "email_classifier_initial",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["parameter_set_id"], "email_classifier_initial")
        self.assertEqual(payload["parameters"]["W1"]["shape"], [3, 8])
        self.assertEqual(payload["parameters"]["b1"]["values"], [0.0, 0.0, 0.0])

    def test_cli_validate_parameters_accepts_parameter_set(self) -> None:
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
                    "validate-parameters",
                    "examples/email-agent.typed.mxai",
                    "--params",
                    str(parameter_path),
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
        self.assertTrue(payload["model_hash"].startswith("mxai_"))
        self.assertTrue(payload["parameter_schema_hash"].startswith("params_"))

    def test_cli_backend_parameters_torch_outputs_parameter_set_without_torch_dependency(self) -> None:
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
        self.assertIn(payload["backend"]["torch_available"], {True, False})
        self.assertEqual(payload["parameter_set"]["parameters"]["W1"]["shape"], [3, 8])
        self.assertIn("Classifier.W1", payload["parameters"])

    def test_cli_backend_parameters_torch_validates_parameter_set_without_torch_dependency(self) -> None:
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
                    "backend-parameters",
                    "examples/email-agent.typed.mxai",
                    "--target",
                    "torch",
                    "--validate",
                    str(parameter_path),
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
        self.assertEqual(payload["parameter_set_id"], f"{program.project}_initial")

    def test_tensor_bridge_validates_parameter_set_before_torch_import(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        data = build_initial_parameter_set(program).to_dict()
        data["parameters"]["W1"]["values"] = [[0.0 for _ in range(8)]]
        data["parameters"]["b1"]["dtype"] = "float64"

        errors = validate_parameter_set_for_torch(ParameterSet.from_dict(data))

        self.assertIn("Parameter W1 expected values shape [3, 8], got [1, 8]", errors)
        self.assertIn("Parameter b1 expected dtype float32, got float64", errors)

    def test_tensor_bridge_requires_optional_torch_dependency_when_absent(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment")
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        parameter_set = build_initial_parameter_set(program)

        with self.assertRaisesRegex(TensorParameterBridgeError, "optional dependency PyTorch"):
            parameter_set_to_torch_tensors(parameter_set)

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_tensor_bridge_materializes_torch_tensors(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        parameter_set = build_initial_parameter_set(program)

        tensors = TensorParameterBridge().to_torch_tensors(parameter_set)

        self.assertEqual(list(tensors["W1"].shape), [3, 8])
        self.assertEqual(str(tensors["W1"].dtype), "torch.float32")
        self.assertIs(tensors["W1"], tensors["Classifier.W1"])
        self.assertEqual(list(tensors["b1"].shape), [3])

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_tensor_bridge_roundtrip_preserves_hashes(self) -> None:
        import torch

        program = parse_file(ROOT / "examples" / "fall-risk.typed.mxai")
        parameter_set = build_initial_parameter_set(program)
        tensors = parameter_set_to_torch_tensors(parameter_set)
        updated_tensors = {
            "W1": tensors["W1"] + torch.ones_like(tensors["W1"]),
            "b1": tensors["b1"] + torch.tensor(0.25, dtype=torch.float32),
        }

        updated = torch_tensors_to_parameter_set(
            parameter_set,
            updated_tensors,
            parameter_set_id="fall_risk_torch_bridge",
            metrics={"backend": "torch"},
        )

        self.assertEqual(updated.parameter_set_id, "fall_risk_torch_bridge")
        self.assertEqual(updated.model_hash, parameter_set.model_hash)
        self.assertEqual(updated.parameter_schema_hash, parameter_set.parameter_schema_hash)
        self.assertEqual(updated.source, "torch")
        self.assertEqual(updated.metrics, {"backend": "torch"})
        self.assertTrue(validate_parameter_set(program, updated).ok)

    @unittest.skipUnless(torch_available(), "PyTorch optional dependency is not installed")
    def test_tensor_bridge_rejects_torch_tensor_shape_mismatch(self) -> None:
        import torch

        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        parameter_set = build_initial_parameter_set(program)

        with self.assertRaisesRegex(TensorParameterBridgeError, r"Parameter W1 expected tensor shape \[3, 8\]"):
            torch_tensors_to_parameter_set(
                parameter_set,
                {
                    "W1": torch.zeros((1, 8), dtype=torch.float32),
                    "b1": torch.zeros((3,), dtype=torch.float32),
                },
            )


if __name__ == "__main__":
    unittest.main()
