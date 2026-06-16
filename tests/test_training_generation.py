from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from matrixai.agents import ChatCompletionsLLMProposalProvider
from matrixai.training import (
    DATASET_MANIFEST_VERSION,
    SupervisedPromptGenerator,
    SupervisedPromptRunner,
    TrainingPromptGenerator,
    TrainingVerifier,
    load_dataset_manifest,
    parse_training_text,
    verify_dataset_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class MatrixAITrainingGenerationTest(unittest.TestCase):
    def test_generate_fall_risk_binary_training_contract_from_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = Path(tmp_dir) / "fall.generated.csv"
            result = TrainingPromptGenerator().generate(
                "Genera entrenamiento para riesgo de caida con etiquetas low, high",
                ROOT / "examples" / "fall-risk.typed.mxai",
                dataset_source=str(dataset),
            )
            dataset.write_text(result.dataset_template_text, encoding="utf-8")
            spec = parse_training_text(result.training_text)
            report = TrainingVerifier().verify(spec, base_path=ROOT)

        self.assertEqual(spec.model, str(ROOT / "examples" / "fall-risk.typed.mxai"))
        self.assertEqual(spec.loss.type, "binary_cross_entropy")
        self.assertEqual(spec.loss.prediction, "R")
        self.assertEqual(spec.dataset.input.vector, "Patient")
        self.assertEqual(spec.dataset.target.name, "risk_label")
        self.assertEqual(spec.dataset.target.type.parameters["args"], ["low", "high"])
        self.assertIn("binary_cross_entropy treats label 'high' as the positive class", result.warnings)
        self.assertEqual(
            result.dataset_template_text.splitlines()[0],
            "age,mobility,medication_load,previous_falls,cognitive_state,risk_label",
        )
        self.assertTrue(report.ok, report.to_dict())

    def test_generate_email_softmax_training_contract_from_prompt(self) -> None:
        result = TrainingPromptGenerator().generate(
            "Genera training de correos con labels support, sales, operations",
            ROOT / "examples" / "email-agent.typed.mxai",
            dataset_source="examples/email-generated.csv",
        )
        spec = parse_training_text(result.training_text)

        self.assertEqual(spec.loss.type, "cross_entropy")
        self.assertEqual(spec.loss.prediction, "C")
        self.assertEqual(spec.dataset.target.name, "label")
        self.assertEqual(spec.dataset.target.type.parameters["args"], ["support", "sales", "operations"])
        self.assertEqual(spec.optimizer.update, ["W1", "b1"])
        self.assertEqual(result.dataset_template_text.splitlines()[0].split(",")[-1], "label")
        self.assertEqual(len(result.dataset_template_text.splitlines()), 4)

    def test_generate_training_rejects_wrong_softmax_label_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "cross_entropy generation requires 3 labels"):
            TrainingPromptGenerator().generate(
                "Genera training de correos con labels support, sales",
                ROOT / "examples" / "email-agent.typed.mxai",
            )

    def test_cli_generate_training_writes_mxtrain_and_dataset_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "fall.generated.mxtrain"
            dataset = Path(tmp_dir) / "fall.generated.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "generate-training",
                    "examples/fall-risk.typed.mxai",
                    "Genera entrenamiento para riesgo de caida con etiquetas low, high",
                    "--output",
                    str(output),
                    "--dataset-output",
                    str(dataset),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            output_exists = output.exists()
            dataset_exists = dataset.exists()
            output_text = output.read_text(encoding="utf-8") if output_exists else ""
            dataset_header = dataset.read_text(encoding="utf-8").splitlines()[0] if dataset_exists else ""

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["loss_type"], "binary_cross_entropy")
        self.assertEqual(payload["written"]["training"], str(output))
        self.assertEqual(payload["written"]["dataset_template"], str(dataset))
        self.assertTrue(output_exists)
        self.assertTrue(dataset_exists)
        self.assertIn("TYPE binary_cross_entropy", output_text)
        self.assertIn("risk_label", dataset_header)

    def test_generate_supervised_prompt_package_writes_valid_relative_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch.object(
                ChatCompletionsLLMProposalProvider,
                "from_env",
                side_effect=ValueError("no_api_key"),
            ):
                result = SupervisedPromptGenerator().generate(
                    "Crear un sistema de riesgo de caida para pacientes con etiquetas low, high",
                    tmp_dir,
                )
            model_path = Path(result.model_path)
            training_path = Path(result.training_path)
            dataset_path = Path(result.dataset_path)
            spec = parse_training_text(training_path.read_text(encoding="utf-8"))
            report = TrainingVerifier().verify(spec, base_path=tmp_dir)
            model_exists = model_path.exists()
            training_exists = training_path.exists()
            dataset_exists = dataset_path.exists()

        self.assertEqual(result.project, "FallRisk")
        self.assertEqual(model_path.name, "fall-risk.mxai")
        self.assertEqual(training_path.name, "fall-risk.supervised.mxtrain")
        self.assertEqual(dataset_path.name, "fall-risk.train.csv")
        self.assertEqual(spec.model, "fall-risk.mxai")
        self.assertEqual(spec.dataset.source, "fall-risk.train.csv")
        self.assertEqual(spec.loss.type, "binary_cross_entropy")
        self.assertTrue(model_exists)
        self.assertTrue(training_exists)
        self.assertTrue(dataset_exists)
        self.assertTrue(report.ok, report.to_dict())

    def test_cli_generate_supervised_writes_valid_classification_package(self) -> None:
        _no_llm_env = {**os.environ, "MATRIXAI_LLM_API_KEY": ""}
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "generate-supervised",
                    "Crear un sistema que clasifique correos con etiquetas support, sales, operations",
                    "--output-dir",
                    tmp_dir,
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=_no_llm_env,
            )
            payload = json.loads(result.stdout)
            training_path = Path(payload["training_path"])
            dataset_path = Path(payload["dataset_path"])
            validation = subprocess.run(
                [sys.executable, "-m", "matrixai", "validate-training", str(training_path), "--json"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=_no_llm_env,
            )
            validation_payload = json.loads(validation.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["project"], "EmailAgent")
        self.assertEqual(payload["training_generation"]["loss_type"], "cross_entropy")
        self.assertEqual(payload["training_generation"]["labels"], ["support", "sales", "operations"])
        self.assertEqual(training_path.name, "email-agent.supervised.mxtrain")
        self.assertEqual(dataset_path.name, "email-agent.train.csv")
        self.assertEqual(validation.returncode, 0, validation.stderr)
        self.assertTrue(validation_payload["ok"], validation_payload)

    def test_run_supervised_prompt_with_real_fall_risk_datasets_trains_and_evaluates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch.object(
                ChatCompletionsLLMProposalProvider,
                "from_env",
                side_effect=ValueError("no_api_key"),
            ):
                result = SupervisedPromptRunner().run(
                    "Crear un sistema de riesgo de caida para pacientes con etiquetas low, high",
                    tmp_dir,
                    train_data=ROOT / "examples" / "fall-risk.train.csv",
                    evaluation_data=ROOT / "examples" / "fall-risk.test.csv",
                )
            manifest_path = Path(result.manifest_path)
            evaluation_report_path = Path(result.evaluation_report_path)
            train_header = Path(result.train_dataset_path).read_text(encoding="utf-8").splitlines()[0]
            template_header = Path(result.dataset_template_path).read_text(encoding="utf-8").splitlines()[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evaluation_report = json.loads(evaluation_report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.project, "FallRisk")
        self.assertEqual(result.training.run_id, "run")
        self.assertGreater(result.training.best_epoch, 0)
        self.assertEqual(result.evaluation.rows, 4)
        self.assertEqual(result.evaluation.labels, ["low", "high"])
        self.assertGreaterEqual(result.evaluation.accuracy, 0.0)
        self.assertLessEqual(result.evaluation.accuracy, 1.0)
        self.assertEqual(train_header, "age,mobility,medication_load,previous_falls,cognitive_state,risk_label")
        self.assertEqual(template_header, train_header)
        self.assertEqual(manifest["training"]["run_id"], "run")
        self.assertEqual(evaluation_report["rows"], 4)

    def test_run_supervised_prompt_accepts_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch.object(
                ChatCompletionsLLMProposalProvider,
                "from_env",
                side_effect=ValueError("no_api_key"),
            ):
                result = SupervisedPromptRunner().run(
                    "Crear un sistema de riesgo de caida para pacientes con etiquetas low, high",
                    tmp_dir,
                    dataset_manifest=ROOT / "examples" / "fall-risk.dataset-manifest.json",
                )
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(result.project, "FallRisk")
        self.assertEqual(result.evaluation.rows, 4)
        self.assertEqual(result.dataset_manifest["name"], "fall-risk-label-datasets-v1")
        self.assertTrue(result.dataset_manifest_verification["ok"])
        self.assertEqual(manifest["dataset_manifest"]["name"], "fall-risk-label-datasets-v1")
        self.assertEqual(manifest["dataset_manifest_verification"]["datasets"][0]["rows"], 8)

    def test_dataset_manifest_verifies_versioned_split(self) -> None:
        manifest_path = ROOT / "examples" / "fall-risk.dataset-manifest.json"
        report = verify_dataset_manifest(load_dataset_manifest(manifest_path), base_path=manifest_path.parent)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.splits[0]["name"], "holdout-v1")
        self.assertEqual(report.splits[0]["partitions"][0]["row_indices"], [2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(report.splits[0]["partitions"][0]["fingerprint"], "split_part_bd849fe68d83ea1f")

    def test_run_supervised_prompt_accepts_dataset_manifest_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch.object(
                ChatCompletionsLLMProposalProvider,
                "from_env",
                side_effect=ValueError("no_api_key"),
            ):
                result = SupervisedPromptRunner().run(
                    "Crear un sistema de riesgo de caida para pacientes con etiquetas low, high",
                    tmp_dir,
                    dataset_manifest=ROOT / "examples" / "fall-risk.dataset-manifest.json",
                    dataset_split="holdout-v1",
                )
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(result.project, "FallRisk")
        self.assertEqual(result.dataset_split["name"], "holdout-v1")
        self.assertTrue(result.dataset_manifest_verification["ok"])
        self.assertEqual(result.evaluation.rows, 4)
        self.assertEqual(manifest["dataset_split"]["partitions"][0]["fingerprint"], "split_part_bd849fe68d83ea1f")

    def test_dataset_manifest_rejects_split_fingerprint_mismatch(self) -> None:
        data = json.loads((ROOT / "examples" / "fall-risk.dataset-manifest.json").read_text(encoding="utf-8"))
        data["splits"][0]["partitions"][0]["fingerprint"] = "split_part_wrong"
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "bad-split-manifest.json"
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            report = verify_dataset_manifest(load_dataset_manifest(manifest_path), base_path=ROOT / "examples")

        self.assertFalse(report.ok)
        self.assertIn("fingerprint mismatch", "; ".join(report.errors))

    def test_dataset_manifest_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "bad-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": DATASET_MANIFEST_VERSION,
                        "name": "bad-fall-risk",
                        "datasets": [
                            {
                                "role": "train",
                                "source": str(ROOT / "examples" / "fall-risk.train.csv"),
                                "sha256": "0" * 64,
                            },
                            {
                                "role": "evaluation",
                                "source": str(ROOT / "examples" / "fall-risk.test.csv"),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = verify_dataset_manifest(load_dataset_manifest(manifest_path), base_path=manifest_path.parent)

        self.assertFalse(report.ok)
        self.assertIn("sha256 mismatch", "; ".join(report.errors))

    def test_cli_train_supervised_generates_trains_and_evaluates_email_package(self) -> None:
        _no_llm_env = {**os.environ, "MATRIXAI_LLM_API_KEY": ""}
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "train-supervised",
                    "Crear un sistema que clasifique correos con etiquetas support, sales, operations",
                    "--output-dir",
                    tmp_dir,
                    "--train-data",
                    "examples/email-agent.train.csv",
                    "--eval-data",
                    "examples/email-agent.test.csv",
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=_no_llm_env,
            )
            payload = json.loads(result.stdout)
            manifest_exists = Path(payload["manifest_path"]).exists()
            evaluation_report_exists = Path(payload["evaluation_report_path"]).exists()
            params_best_exists = Path(payload["training"]["artifacts"]["params_best"]).exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["project"], "EmailAgent")
        self.assertEqual(payload["generation"]["training_generation"]["loss_type"], "cross_entropy")
        self.assertEqual(payload["evaluation"]["rows"], 3)
        self.assertEqual(payload["evaluation"]["labels"], ["support", "sales", "operations"])
        self.assertGreaterEqual(payload["evaluation"]["accuracy"], 0.0)
        self.assertTrue(manifest_exists)
        self.assertTrue(evaluation_report_exists)
        self.assertTrue(params_best_exists)

    def test_cli_train_supervised_accepts_dataset_manifest(self) -> None:
        _no_llm_env = {**os.environ, "MATRIXAI_LLM_API_KEY": ""}
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "train-supervised",
                    "Crear un sistema de riesgo de caida para pacientes con etiquetas low, high",
                    "--output-dir",
                    tmp_dir,
                    "--dataset-manifest",
                    "examples/fall-risk.dataset-manifest.json",
                    "--dataset-split",
                    "holdout-v1",
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=_no_llm_env,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["project"], "FallRisk")
        self.assertEqual(payload["dataset_manifest"]["name"], "fall-risk-label-datasets-v1")
        self.assertEqual(payload["dataset_split"]["name"], "holdout-v1")
        self.assertTrue(payload["dataset_manifest_verification"]["ok"])
        self.assertEqual(payload["evaluation"]["rows"], 4)


if __name__ == "__main__":
    unittest.main()