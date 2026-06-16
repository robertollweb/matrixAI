from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.compiler import DifferentiablePythonCompiler
from matrixai.parameters import load_parameter_set, validate_parameter_set
from matrixai.parser import parse_file
from matrixai.runtime import MatrixAIRuntime
from matrixai.training import (
    CSVDataAdapter,
    InMemoryDataAdapter,
    MatrixAIBatch,
    SupervisedEvaluator,
    SupervisedTrainer,
    dataset_fingerprint,
    parse_training_file,
)


ROOT = Path(__file__).resolve().parents[1]
P4_SNAPSHOT_DIR = ROOT / "tests" / "snapshots" / "p4"


def _p4_snapshot(name: str):
    return json.loads((P4_SNAPSHOT_DIR / name).read_text(encoding="utf-8"))


def _normalized_training_trace(trace: dict) -> dict:
    normalized = dict(trace)
    normalized["model"] = "<MODEL>"
    normalized["dataset"] = dict(trace["dataset"])
    normalized["dataset"]["source"] = "<DATASET>"
    normalized.pop("backend_report", None)
    return normalized


class MatrixAITrainingPipelineTest(unittest.TestCase):
    def _load_generated_module(self, path: Path):
        spec = importlib.util.spec_from_file_location("compiled_p4_program", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_csv_data_adapter_emits_matrixai_batches_with_metadata(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        adapter = CSVDataAdapter(
            ROOT / spec.dataset.source,
            spec.dataset.input.vector,
            spec.dataset.input.columns,
            spec.dataset.target.name,
            spec.dataset.target.type.parameters["args"],
        )

        batch = next(adapter.iter_batches(2))
        schema = adapter.schema()

        self.assertIsInstance(batch, MatrixAIBatch)
        self.assertEqual(schema.rows, 6)
        self.assertEqual(schema.input_vector, "Email")
        self.assertEqual(schema.labels, ["support", "sales", "operations"])
        self.assertEqual(len(batch.inputs["Email"]), 2)
        self.assertEqual(batch.targets["label"], ["support", "sales"])
        self.assertEqual(batch.metadata["dataset_fingerprint"], dataset_fingerprint(ROOT / spec.dataset.source))
        self.assertEqual(batch.metadata["row_indices"], [2, 3])
        self.assertTrue(all(value.startswith("row_") for value in batch.metadata["row_hashes"]))

    def test_in_memory_data_adapter_emits_reproducible_batches(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        rows = [
            {
                "urgency": 0.9,
                "sender_trust": 0.7,
                "topic_support": 1.0,
                "topic_sales": 0.0,
                "sentiment": 0.6,
                "has_attachment": 1.0,
                "previous_interactions": 0.4,
                "language_confidence": 0.95,
                "label": "support",
            },
            {
                "urgency": 0.2,
                "sender_trust": 0.9,
                "topic_support": 0.0,
                "topic_sales": 1.0,
                "sentiment": 0.8,
                "has_attachment": 0.0,
                "previous_interactions": 0.6,
                "language_confidence": 0.98,
                "label": "sales",
            },
            {
                "urgency": 0.5,
                "sender_trust": 0.6,
                "topic_support": 0.0,
                "topic_sales": 0.0,
                "sentiment": 0.4,
                "has_attachment": 1.0,
                "previous_interactions": 0.3,
                "language_confidence": 0.9,
                "label": "operations",
            },
        ]
        adapter = InMemoryDataAdapter(
            rows,
            spec.dataset.input.vector,
            spec.dataset.input.columns,
            spec.dataset.target.name,
            spec.dataset.target.type.parameters["args"],
            source="generated://email-memory",
        )
        same_adapter = InMemoryDataAdapter(
            rows,
            spec.dataset.input.vector,
            spec.dataset.input.columns,
            spec.dataset.target.name,
            spec.dataset.target.type.parameters["args"],
            source="generated://email-memory",
        )

        batch = next(adapter.iter_batches(2, indices=[0, 2]))
        schema = adapter.schema()

        self.assertIsInstance(batch, MatrixAIBatch)
        self.assertEqual(schema.source_kind, "memory")
        self.assertEqual(schema.source, "generated://email-memory")
        self.assertEqual(schema.rows, 3)
        self.assertEqual(adapter.fingerprint(), same_adapter.fingerprint())
        self.assertTrue(adapter.fingerprint().startswith("data_"))
        self.assertEqual(batch.targets["label"], ["support", "operations"])
        self.assertEqual(batch.metadata["source"], "generated://email-memory")
        self.assertEqual(batch.metadata["dataset_fingerprint"], adapter.fingerprint())
        self.assertEqual(batch.metadata["row_indices"], [1, 3])
        self.assertTrue(all(value.startswith("row_") for value in batch.metadata["row_hashes"]))

    def test_train_softmax_linear_reduces_loss_and_writes_artifacts(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            result = SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))
            initial = load_parameter_set(output_dir / "params.initial.json")
            best = load_parameter_set(output_dir / "params.best.json")
            final = load_parameter_set(output_dir / "params.final.json")

        self.assertEqual(result.run_id, "email_classifier_001")
        self.assertEqual(Path(result.artifacts["params_best"]).name, "params.best.json")
        self.assertGreater(len(trace["epochs"]), 1)
        self.assertLess(trace["epochs"][-1]["train_loss"], trace["epochs"][0]["train_loss"])
        self.assertEqual(trace["selected_parameter_set"], "params.best.json")
        self.assertEqual(
            trace["dataset"]["fingerprint"],
            dataset_fingerprint(ROOT / "examples" / "email-agent.train.csv"),
        )
        split = trace["dataset"]["split"]
        self.assertEqual(split["config"], {"seed": 42, "train": 0.75, "validation": 0.25})
        self.assertEqual(split["train"]["rows"], trace["dataset"]["rows_train"])
        self.assertEqual(split["validation"]["rows"], trace["dataset"]["rows_validation"])
        self.assertEqual(split["train"]["row_indices"], [3, 4, 5, 6])
        self.assertEqual(split["validation"]["row_indices"], [2, 7])
        self.assertTrue(split["fingerprint"].startswith("split_"))
        self.assertTrue(split["train"]["fingerprint"].startswith("split_part_"))
        self.assertTrue(all(value.startswith("row_") for value in split["train"]["row_hashes"]))
        self.assertEqual(initial.parameters["W1"]["shape"], [3, 8])
        self.assertEqual(best.parameters["W1"]["shape"], [3, 8])
        self.assertEqual(final.parameters["b1"]["shape"], [3])
        self.assertNotEqual(initial.parameters["W1"]["values"], final.parameters["W1"]["values"])

    def test_train_sigmoid_linear_binary_cross_entropy_reduces_loss(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.supervised.mxtrain")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "fall_risk_001"
            result = SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "fall-risk.supervised.mxtrain",
            )
            trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))
            initial = load_parameter_set(output_dir / "params.initial.json")
            best = load_parameter_set(output_dir / "params.best.json")
            final = load_parameter_set(output_dir / "params.final.json")

        self.assertEqual(result.run_id, "fall_risk_001")
        self.assertEqual(trace["loss"], "binary_cross_entropy")
        self.assertEqual(trace["prediction"], "R")
        self.assertGreater(len(trace["epochs"]), 1)
        self.assertLess(trace["epochs"][-1]["train_loss"], trace["epochs"][0]["train_loss"])
        self.assertEqual(trace["dataset"]["fingerprint"], dataset_fingerprint(ROOT / "examples" / "fall-risk.train.csv"))
        self.assertEqual(initial.parameters["W1"]["shape"], [5])
        self.assertEqual(best.parameters["W1"]["shape"], [5])
        self.assertEqual(final.parameters["b1"]["shape"], [])
        self.assertIsInstance(final.parameters["b1"]["values"], float)
        self.assertNotEqual(initial.parameters["W1"]["values"], final.parameters["W1"]["values"])

    def test_train_sigmoid_linear_binary_cross_entropy_probability_target(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.probability.mxtrain")
        test_dataset = ROOT / "examples" / "fall-risk.probability.test.csv"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "fall_risk_probability_001"
            result = SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "fall-risk.probability.mxtrain",
            )
            trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))
            final = load_parameter_set(output_dir / "params.final.json")
            evaluation = SupervisedEvaluator().evaluate(
                spec,
                parameter_set=load_parameter_set(output_dir / "params.best.json"),
                data_path=test_dataset,
                base_path=ROOT,
            )

        self.assertEqual(result.run_id, "fall_risk_probability_001")
        self.assertEqual(trace["loss"], "binary_cross_entropy")
        self.assertEqual(trace["dataset"]["fingerprint"], dataset_fingerprint(ROOT / "examples" / "fall-risk.probability.train.csv"))
        self.assertGreater(len(trace["epochs"]), 1)
        self.assertLess(trace["epochs"][-1]["train_loss"], trace["epochs"][0]["train_loss"])
        self.assertEqual(final.parameters["W1"]["shape"], [5])
        self.assertEqual(evaluation.rows, 4)
        self.assertEqual(evaluation.labels, ["negative", "positive"])
        self.assertEqual(set(evaluation.confusion_matrix), {"negative", "positive"})
        self.assertGreaterEqual(evaluation.accuracy, 0.0)
        self.assertLessEqual(evaluation.accuracy, 1.0)

    def test_evaluate_fall_risk_binary_classifier_reports_metrics(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.supervised.mxtrain")
        test_dataset = ROOT / "examples" / "fall-risk.test.csv"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "fall_risk_001"
            SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "fall-risk.supervised.mxtrain",
            )
            result = SupervisedEvaluator().evaluate(
                spec,
                parameter_set=load_parameter_set(output_dir / "params.best.json"),
                data_path=test_dataset,
                base_path=ROOT,
            )

        self.assertEqual(result.rows, 4)
        self.assertEqual(result.labels, ["low", "high"])
        self.assertEqual(result.dataset_fingerprint, dataset_fingerprint(test_dataset))
        self.assertEqual(result.confusion_matrix["low"]["low"], 2)
        self.assertEqual(result.confusion_matrix["high"]["high"], 2)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)

    def test_training_trace_includes_split_fingerprint(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_output = Path(tmp_dir) / "email_classifier_001"
            second_output = Path(tmp_dir) / "email_classifier_002"
            SupervisedTrainer().train(
                spec,
                output_dir=first_output,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            SupervisedTrainer().train(
                spec,
                output_dir=second_output,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            first_trace = json.loads((first_output / "training_trace.json").read_text(encoding="utf-8"))
            second_trace = json.loads((second_output / "training_trace.json").read_text(encoding="utf-8"))

        split = first_trace["dataset"]["split"]
        self.assertEqual(split, second_trace["dataset"]["split"])
        self.assertEqual(split["fingerprint"], "split_276e95b71e747e92")
        self.assertEqual(set(split["train"]), {"rows", "row_indices", "row_hashes", "fingerprint"})
        self.assertEqual(set(split["validation"]), {"rows", "row_indices", "row_hashes", "fingerprint"})
        self.assertEqual(split["train"]["row_indices"], [3, 4, 5, 6])
        self.assertEqual(split["validation"]["row_indices"], [2, 7])
        self.assertTrue(all(value.startswith("row_") for value in split["validation"]["row_hashes"]))

    def test_trained_params_validate_and_run_with_runtime(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            best_path = output_dir / "params.best.json"
            best = load_parameter_set(best_path)
            validation = validate_parameter_set(program, best)
            run_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrixai",
                    "run",
                    "examples/email-agent.typed.mxai",
                    "--input",
                    "examples/email-sample.json",
                    "--params",
                    str(best_path),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(run_result.stdout)

        self.assertTrue(validation.ok, validation.to_dict())
        self.assertEqual(run_result.returncode, 0, run_result.stderr)
        self.assertEqual(set(payload["state"]["C"]), {"support", "sales", "operations"})
        self.assertIn("audit", payload)

    def test_evaluate_trained_params_reports_metrics(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            result = SupervisedEvaluator().evaluate(
                spec,
                parameter_set=load_parameter_set(output_dir / "params.best.json"),
                base_path=ROOT,
            )

        self.assertEqual(result.rows, 6)
        self.assertGreater(result.loss, 0.0)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)
        self.assertEqual(set(result.confusion_matrix), {"support", "sales", "operations"})
        self.assertEqual(result.per_label["support"]["support"], 2)
        self.assertGreaterEqual(result.macro_f1, 0.0)
        self.assertLessEqual(result.macro_f1, 1.0)
        self.assertEqual(result.dataset_fingerprint, dataset_fingerprint(ROOT / "examples" / "email-agent.train.csv"))

    def test_evaluate_can_use_separate_test_dataset(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        test_dataset = ROOT / "examples" / "email-agent.test.csv"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            result = SupervisedEvaluator().evaluate(
                spec,
                parameter_set=load_parameter_set(output_dir / "params.best.json"),
                data_path=test_dataset,
                base_path=ROOT,
            )

        self.assertEqual(result.rows, 3)
        self.assertEqual(result.dataset_fingerprint, dataset_fingerprint(test_dataset))
        self.assertEqual(result.dataset_schema["rows"], 3)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)

    def test_trained_parameter_set_matches_differentiable_compiler_runtime(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            best = load_parameter_set(output_dir / "params.best.json")
            compiled_path = Path(tmp_dir) / "compiled_email_agent.py"
            compiled_path.write_text(DifferentiablePythonCompiler().compile(program), encoding="utf-8")
            compiled = self._load_generated_module(compiled_path)
            runtime_result = MatrixAIRuntime().run(program, input_data, parameters=best.runtime_parameters())
            compiled_result = compiled.run(input_data, best.to_dict())

        self.assertEqual(set(compiled_result["state"]["C"]), set(runtime_result["state"]["C"]))
        for label, probability in runtime_result["state"]["C"].items():
            self.assertAlmostEqual(compiled_result["state"]["C"][label], probability)
        self.assertEqual(compiled_result["state"]["Confidence"], runtime_result["state"]["Confidence"])
        self.assertAlmostEqual(compiled_result["state"]["ReplyActivation"], runtime_result["state"]["ReplyActivation"])
        self.assertEqual(compiled_result["actions"], runtime_result["actions"])

    def test_fall_risk_trained_params_runtime_matches_compiled(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "fall-risk.supervised.mxtrain")
        program = parse_file(ROOT / "examples" / "fall-risk.typed.mxai")
        input_data = json.loads((ROOT / "examples" / "fall-risk-sample.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "fall_risk_001"
            SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "fall-risk.supervised.mxtrain",
            )
            best = load_parameter_set(output_dir / "params.best.json")
            compiled_path = Path(tmp_dir) / "compiled_fall_risk.py"
            compiled_path.write_text(DifferentiablePythonCompiler().compile(program), encoding="utf-8")
            compiled = self._load_generated_module(compiled_path)
            runtime_result = MatrixAIRuntime().run(program, input_data, parameters=best.runtime_parameters())
            compiled_result = compiled.run(input_data, best.to_dict())

        self.assertAlmostEqual(compiled_result["state"]["R"], runtime_result["state"]["R"])
        self.assertAlmostEqual(compiled_result["state"]["Risk"]["mean"], runtime_result["state"]["Risk"]["mean"])
        self.assertAlmostEqual(compiled_result["state"]["AlertActivation"], runtime_result["state"]["AlertActivation"])
        self.assertEqual(compiled_result["actions"], runtime_result["actions"])

    def test_cli_train_writes_parameter_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
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
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            artifacts = {path.name for path in output_dir.iterdir()}

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["run_id"], "email_classifier_001")
        self.assertIn("params.best.json", artifacts)
        self.assertIn("metrics.json", artifacts)
        self.assertIn("training_trace.json", artifacts)
        self.assertGreater(payload["best_epoch"], 0)

    def test_cli_evaluate_uses_parameter_set_and_training_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            train_result = subprocess.run(
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
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
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
                    str(output_dir / "params.best.json"),
                    "--data",
                    "examples/email-agent.test.csv",
                    "--output",
                    str(output_dir / "evaluation_report.json"),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            report = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))

        self.assertEqual(train_result.returncode, 0, train_result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["rows"], 3)
        self.assertEqual(payload["dataset_fingerprint"], dataset_fingerprint(ROOT / "examples" / "email-agent.test.csv"))
        self.assertIn("accuracy", payload)
        self.assertEqual(report["parameter_set_id"], payload["parameter_set_id"])
        self.assertEqual(report["dataset_schema"]["rows"], 3)
        self.assertIn("confusion_matrix", report)

    def test_cli_backend_run_accepts_trained_parameter_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            train_result = subprocess.run(
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
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
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
                    str(output_dir / "params.best.json"),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(train_result.returncode, 0, train_result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["target"], "differentiable_python")
        self.assertEqual(set(payload["state"]["C"]), {"support", "sales", "operations"})

    def test_cli_train_rejects_mismatched_model_argument(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "train",
                "examples/fall-risk.typed.mxai",
                "--training",
                "examples/email-agent.supervised.mxtrain",
                "--output",
                "/tmp/matrixai-should-not-write",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("does not match command model", result.stderr)

    def test_training_trace_snapshot(self) -> None:
        spec = parse_training_file(ROOT / "examples" / "email-agent.supervised.mxtrain")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "email_classifier_001"
            SupervisedTrainer().train(
                spec,
                output_dir=output_dir,
                base_path=ROOT,
                training_path=ROOT / "examples" / "email-agent.supervised.mxtrain",
            )
            trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))

        self.assertEqual(
            _normalized_training_trace(trace),
            _p4_snapshot("training_trace_email_classifier.json"),
        )


if __name__ == "__main__":
    unittest.main()
