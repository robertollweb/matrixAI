"""P12 Cut 7 — regression guard: P1-P11 contracts unaffected by P12 changes.

Scope: each test targets a specific P12 change site and verifies the prior-phase
contract at that site is preserved.  NOT a re-run of existing tests; targeted
smoke-tests of the shared code paths P12 extended.

P12 change sites tested:
  - training/dataset_manifest.py  (DatasetManifest: new optional fields origin/generator;
                                    DatasetManifestVerificationResult: new is_synthetic field)
  - training/supervised_prompt.py  (SupervisedPromptRunResult: new synthetic_origin field;
                                     run() patching logic for training_trace/eval_report)
  - training/synthetic.py  (SyntheticDataGenerator label lookup fix for real parser output)
  - cli.py  (new generate-dataset subcommand; existing subcommands unaffected)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.typed.mxai"
_EMAIL_TRAIN = _BASE / "examples" / "email-agent.supervised.mxtrain"
_EMAIL_TRAIN_CSV = _BASE / "examples" / "email-agent.train.csv"
_EMAIL_TEST_CSV = _BASE / "examples" / "email-agent.test.csv"
_FALL_MANIFEST = _BASE / "examples" / "fall-risk.dataset-manifest.json"


# ---------------------------------------------------------------------------
# dataset_manifest.py — backwards compat for real manifests
# ---------------------------------------------------------------------------

class TestP12RegressionManifestCompat(unittest.TestCase):
    """P12 added origin/generator to DatasetManifest; real manifests must be unchanged."""

    def test_manifest_without_origin_round_trips_cleanly(self):
        from matrixai.training.dataset_manifest import DatasetManifest, DATASET_MANIFEST_VERSION, DatasetManifestEntry
        manifest = DatasetManifest(
            version=DATASET_MANIFEST_VERSION,
            name="real-project",
            datasets=[
                DatasetManifestEntry(role="train", source="train.csv", sha256="abc", rows=100),
                DatasetManifestEntry(role="evaluation", source="eval.csv", sha256="def", rows=20),
            ],
        )
        d = manifest.to_dict()
        self.assertNotIn("origin", d)
        self.assertNotIn("generator", d)

        restored = DatasetManifest.from_dict(d)
        self.assertIsNone(restored.origin)
        self.assertIsNone(restored.generator)

    def test_verify_real_manifest_is_not_synthetic(self):
        from matrixai.training.dataset_manifest import load_dataset_manifest, verify_dataset_manifest
        manifest = load_dataset_manifest(_FALL_MANIFEST)
        result = verify_dataset_manifest(manifest, base_path=_FALL_MANIFEST.parent)
        self.assertFalse(result.is_synthetic)
        for w in result.warnings:
            self.assertNotIn("synthetic", w.lower())

    def test_verify_real_manifest_to_dict_no_is_synthetic_key(self):
        from matrixai.training.dataset_manifest import load_dataset_manifest, verify_dataset_manifest
        manifest = load_dataset_manifest(_FALL_MANIFEST)
        result = verify_dataset_manifest(manifest, base_path=_FALL_MANIFEST.parent)
        d = result.to_dict()
        self.assertNotIn("is_synthetic", d)

    def test_load_real_manifest_preserves_all_existing_fields(self):
        from matrixai.training.dataset_manifest import load_dataset_manifest
        manifest = load_dataset_manifest(_FALL_MANIFEST)
        self.assertEqual(manifest.version, "matrixai.dataset_manifest.v1")
        self.assertTrue(len(manifest.datasets) >= 1)
        self.assertTrue(len(manifest.splits) >= 1)
        self.assertIsNone(manifest.origin)
        self.assertIsNone(manifest.generator)


# ---------------------------------------------------------------------------
# supervised_prompt.py — existing train flows unaffected
# ---------------------------------------------------------------------------

class TestP12RegressionSupervisedPrompt(unittest.TestCase):
    """P12 added synthetic_origin to SupervisedPromptRunResult; non-synthetic runs unaffected."""

    def test_run_result_has_synthetic_origin_false_with_direct_csvs(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-m", "matrixai", "train-supervised",
                 "Crear un sistema que clasifique correos con etiquetas support, sales, operations",
                 "--output-dir", tmp,
                 "--train-data", str(_EMAIL_TRAIN_CSV),
                 "--eval-data", str(_EMAIL_TEST_CSV),
                 "--json"],
                cwd=_BASE, capture_output=True, text=True,
                env={**os.environ, "MATRIXAI_LLM_API_KEY": ""},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)

        self.assertFalse(payload.get("synthetic_origin", False),
                         "direct-CSV run must not have synthetic_origin=True")

    def test_training_trace_has_no_synthetic_origin_with_direct_csvs(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-m", "matrixai", "train-supervised",
                 "Crear un sistema que clasifique correos con etiquetas support, sales, operations",
                 "--output-dir", tmp,
                 "--train-data", str(_EMAIL_TRAIN_CSV),
                 "--eval-data", str(_EMAIL_TEST_CSV),
                 "--json"],
                cwd=_BASE, capture_output=True, text=True,
                env={**os.environ, "MATRIXAI_LLM_API_KEY": ""},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            run_dir = Path(payload["run_dir"])
            trace = json.loads((run_dir / "training_trace.json").read_text(encoding="utf-8"))

        self.assertFalse(trace["dataset"].get("synthetic_origin", False),
                         "training_trace.dataset must not have synthetic_origin for real data")

    def test_evaluation_report_has_no_synthetic_origin_with_direct_csvs(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-m", "matrixai", "train-supervised",
                 "Crear un sistema que clasifique correos con etiquetas support, sales, operations",
                 "--output-dir", tmp,
                 "--train-data", str(_EMAIL_TRAIN_CSV),
                 "--eval-data", str(_EMAIL_TEST_CSV),
                 "--json"],
                cwd=_BASE, capture_output=True, text=True,
                env={**os.environ, "MATRIXAI_LLM_API_KEY": ""},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            eval_report = json.loads(
                Path(payload["evaluation_report_path"]).read_text(encoding="utf-8")
            )

        self.assertFalse(eval_report.get("synthetic_origin", False),
                         "evaluation_report must not have synthetic_origin for real data")


# ---------------------------------------------------------------------------
# synthetic.py — label lookup fix does not break unit tests using 'labels' key
# ---------------------------------------------------------------------------

class TestP12RegressionSyntheticLabelLookup(unittest.TestCase):
    """Fix: generator now checks parameters['args'] before parameters['labels'].
    Old behaviour for explicit 'labels' key must still work."""

    def _make_training(self, labels_key: str):
        from matrixai.ir.schema import MatrixAIProgram, VectorSpec
        from matrixai.training.spec import (
            DatasetInputSpec, DatasetSpec, DatasetTargetSpec,
            LossSpec, OptimizerSpec, TrainingSpec,
        )
        from matrixai.types import TypeSpec
        program = MatrixAIProgram(
            project="test",
            vectors=[VectorSpec(
                name="V", size=1, fields=["x"],
                field_types={"x": TypeSpec(name="Probability")},
            )],
        )
        training = TrainingSpec(
            model="test.mxai",
            dataset=DatasetSpec(
                name="ds", source_kind="synthetic", source="synthetic",
                input=DatasetInputSpec(vector="V", columns=["x"]),
                target=DatasetTargetSpec(
                    name="label",
                    type=TypeSpec(name="Label", parameters={labels_key: ["A", "B"]}),
                ),
            ),
            loss=LossSpec(name="l", type="CrossEntropy", prediction="logits", target="label"),
            optimizer=OptimizerSpec(name="o", type="SGD", learning_rate=0.01, update=[]),
        )
        return program, training

    def test_labels_key_still_works(self):
        from matrixai.training.synthetic import SyntheticDataGenerator
        program, training = self._make_training("labels")
        adapter = SyntheticDataGenerator(program, training, seed=1, rows=5).generate()
        self.assertEqual(len(adapter.examples()), 5)

    def test_args_key_works_as_real_parser_produces(self):
        from matrixai.training.synthetic import SyntheticDataGenerator
        program, training = self._make_training("args")
        adapter = SyntheticDataGenerator(program, training, seed=1, rows=5).generate()
        self.assertEqual(len(adapter.examples()), 5)


# ---------------------------------------------------------------------------
# cli.py — existing commands unaffected by new generate-dataset subcommand
# ---------------------------------------------------------------------------

class TestP12RegressionCLICommands(unittest.TestCase):
    """New generate-dataset subcommand must not shadow or break existing commands."""

    def _help(self, *cmd: str) -> int:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", *cmd, "--help"],
            cwd=_BASE, capture_output=True, text=True,
        )
        return result.returncode

    def test_validate_training_help(self):
        self.assertEqual(self._help("validate-training"), 0)

    def test_generate_training_help(self):
        self.assertEqual(self._help("generate-training"), 0)

    def test_train_supervised_help(self):
        self.assertEqual(self._help("train-supervised"), 0)

    def test_train_help(self):
        self.assertEqual(self._help("train"), 0)

    def test_evaluate_help(self):
        self.assertEqual(self._help("evaluate"), 0)

    def test_generate_dataset_help(self):
        self.assertEqual(self._help("generate-dataset"), 0)


if __name__ == "__main__":
    unittest.main()
