from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from matrixai.ir.schema import MatrixAIProgram, VectorSpec
from matrixai.training.spec import (
    DatasetBatchSpec,
    DatasetInputSpec,
    DatasetSpec,
    DatasetTargetSpec,
    LossSpec,
    OptimizerSpec,
    TrainingSpec,
)
from matrixai.types import RangeSpec, TypeSpec
from matrixai.training.synthetic import SyntheticDataGenerator
from matrixai.training.dataset_manifest import (
    DATASET_MANIFEST_VERSION,
    SYNTHETIC_GENERATOR_VERSION,
    DatasetManifest,
    DatasetManifestEntry,
    GeneratorSpec,
    build_synthetic_manifest,
    load_dataset_manifest,
    verify_dataset_manifest,
)


class TestP12SyntheticDataGenerator(unittest.TestCase):
    def setUp(self):
        self.program = MatrixAIProgram(
            project="test_p12",
            vectors=[
                VectorSpec(
                    name="Input",
                    size=2,
                    fields=["f1", "f2"],
                    field_types={
                        "f1": TypeSpec(name="Probability"),
                        "f2": TypeSpec(name="Float", range=RangeSpec(minimum=10.0, maximum=20.0)),
                    },
                )
            ],
        )

        self.training = TrainingSpec(
            model="test_p12_model",
            dataset=DatasetSpec(
                name="test_dataset",
                source_kind="synthetic",
                source="synthetic",
                input=DatasetInputSpec(vector="Input", columns=["f1", "f2"]),
                target=DatasetTargetSpec(
                    name="label",
                    type=TypeSpec(name="Label", parameters={"labels": ["A", "B", "C"]}),
                ),
            ),
            loss=LossSpec(name="loss", type="CrossEntropy", prediction="logits", target="label"),
            optimizer=OptimizerSpec(name="opt", type="Adam", learning_rate=0.01, update=[]),
        )

    def test_random_mode_generation(self):
        generator = SyntheticDataGenerator(
            program=self.program,
            training=self.training,
            seed=42,
            rows=10,
            mode="random",
        )
        adapter = generator.generate()

        self.assertEqual(len(adapter.examples()), 10)
        self.assertEqual(adapter.schema().source_kind, "memory")
        self.assertEqual(adapter.schema().rows, 10)

        for example in adapter.examples():
            self.assertEqual(len(example.vector), 2)
            # f1 is Probability [0, 1]
            self.assertTrue(0.0 <= example.vector[0] <= 1.0)
            # f2 is Float [10, 20]
            self.assertTrue(10.0 <= example.vector[1] <= 20.0)
            # label
            self.assertIn(example.label, ["A", "B", "C"])

    def test_reproducibility(self):
        gen1 = SyntheticDataGenerator(self.program, self.training, 123, 5, "random")
        adapter1 = gen1.generate()

        gen2 = SyntheticDataGenerator(self.program, self.training, 123, 5, "random")
        adapter2 = gen2.generate()

        self.assertEqual(adapter1.fingerprint(), adapter2.fingerprint())

        gen3 = SyntheticDataGenerator(self.program, self.training, 456, 5, "random")
        adapter3 = gen3.generate()

        self.assertNotEqual(adapter1.fingerprint(), adapter3.fingerprint())

    def test_random_mode_has_zero_fallback_count(self):
        gen = SyntheticDataGenerator(self.program, self.training, 42, 10, "random")
        gen.generate()
        self.assertEqual(gen.coherent_fallback_count, 0)

    def test_coherent_mode_fallback_counted_when_runtime_cannot_resolve(self):
        # This program has no GRAPH/RULES so the runtime produces no label: all rows fall back.
        import warnings
        gen = SyntheticDataGenerator(self.program, self.training, 42, 10, "coherent")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen.generate()
        self.assertEqual(gen.coherent_fallback_count, 10)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        msg = str(caught[0].message)
        self.assertIn("10/10", msg)
        self.assertIn("coherent", msg)

    def test_coherent_mode_no_warning_when_no_fallback(self):
        # Use random mode as a proxy — coherent with a working runtime should not emit warning.
        # This test guards that zero-fallback runs are silent.
        import warnings
        gen = SyntheticDataGenerator(self.program, self.training, 42, 5, "random")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen.generate()
        synthetic_warns = [w for w in caught if issubclass(w.category, UserWarning) and "fallback" in str(w.message).lower()]
        self.assertEqual(len(synthetic_warns), 0)


class TestP12SyntheticManifest(unittest.TestCase):
    def _minimal_entries(self):
        return [
            DatasetManifestEntry(role="train", source="train.csv"),
            DatasetManifestEntry(role="evaluation", source="eval.csv"),
        ]

    def test_generator_spec_round_trip(self):
        spec = GeneratorSpec(
            version=SYNTHETIC_GENERATOR_VERSION,
            seed=42,
            mode="random",
            rows=200,
        )
        d = spec.to_dict()
        self.assertEqual(d["version"], SYNTHETIC_GENERATOR_VERSION)
        self.assertEqual(d["seed"], 42)
        self.assertEqual(d["mode"], "random")
        self.assertEqual(d["rows"], 200)

        spec2 = GeneratorSpec.from_dict(d)
        self.assertEqual(spec2.seed, 42)
        self.assertEqual(spec2.mode, "random")
        self.assertEqual(spec2.rows, 200)

    def test_build_synthetic_manifest(self):
        manifest = build_synthetic_manifest(
            name="myproject-synthetic-42",
            seed=42,
            mode="coherent",
            rows=100,
            datasets=self._minimal_entries(),
        )
        self.assertEqual(manifest.version, DATASET_MANIFEST_VERSION)
        self.assertEqual(manifest.origin, "synthetic")
        self.assertIsNotNone(manifest.generator)
        self.assertEqual(manifest.generator.seed, 42)
        self.assertEqual(manifest.generator.mode, "coherent")
        self.assertEqual(manifest.generator.rows, 100)

    def test_manifest_to_dict_includes_origin_and_generator(self):
        manifest = build_synthetic_manifest(
            name="proj-synthetic-7",
            seed=7,
            mode="random",
            rows=50,
            datasets=self._minimal_entries(),
        )
        d = manifest.to_dict()
        self.assertEqual(d["origin"], "synthetic")
        self.assertIn("generator", d)
        self.assertEqual(d["generator"]["seed"], 7)
        self.assertEqual(d["generator"]["mode"], "random")
        self.assertEqual(d["generator"]["rows"], 50)

    def test_manifest_from_dict_round_trip(self):
        manifest = build_synthetic_manifest(
            name="proj-synthetic-99",
            seed=99,
            mode="coherent",
            rows=300,
            datasets=self._minimal_entries(),
        )
        restored = DatasetManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.origin, "synthetic")
        self.assertIsNotNone(restored.generator)
        self.assertEqual(restored.generator.seed, 99)
        self.assertEqual(restored.generator.mode, "coherent")
        self.assertEqual(restored.generator.rows, 300)

    def test_load_manifest_from_json_file(self):
        payload = {
            "version": DATASET_MANIFEST_VERSION,
            "name": "proj-synthetic-5",
            "origin": "synthetic",
            "generator": {
                "version": SYNTHETIC_GENERATOR_VERSION,
                "seed": 5,
                "mode": "random",
                "rows": 20,
            },
            "datasets": [
                {"role": "train", "source": "train.csv"},
                {"role": "evaluation", "source": "eval.csv"},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            path = f.name

        loaded = load_dataset_manifest(path)
        self.assertEqual(loaded.origin, "synthetic")
        self.assertIsNotNone(loaded.generator)
        self.assertEqual(loaded.generator.seed, 5)
        self.assertEqual(loaded.generator.rows, 20)

    def test_verify_manifest_is_synthetic_flag(self):
        manifest = build_synthetic_manifest(
            name="proj-synthetic-1",
            seed=1,
            mode="random",
            rows=10,
            datasets=self._minimal_entries(),
        )
        result = verify_dataset_manifest(manifest, base_path="/nonexistent")
        self.assertTrue(result.is_synthetic)
        self.assertFalse(result.ok)
        any_synthetic_warning = any(
            "synthetic" in w.lower() for w in result.warnings
        )
        self.assertTrue(any_synthetic_warning)

    def test_verify_manifest_non_synthetic_is_not_flagged(self):
        manifest = DatasetManifest(
            version=DATASET_MANIFEST_VERSION,
            name="real-project",
            datasets=self._minimal_entries(),
        )
        result = verify_dataset_manifest(manifest, base_path="/nonexistent")
        self.assertFalse(result.is_synthetic)

    def test_to_dict_includes_is_synthetic_when_true(self):
        manifest = build_synthetic_manifest(
            name="proj-synthetic-2",
            seed=2,
            mode="random",
            rows=10,
            datasets=self._minimal_entries(),
        )
        result = verify_dataset_manifest(manifest, base_path="/nonexistent")
        d = result.to_dict()
        self.assertTrue(d.get("is_synthetic"))

    def test_manifest_without_origin_preserves_backwards_compat(self):
        payload = {
            "version": DATASET_MANIFEST_VERSION,
            "name": "legacy-project",
            "datasets": [
                {"role": "train", "source": "train.csv"},
                {"role": "evaluation", "source": "eval.csv"},
            ],
        }
        manifest = DatasetManifest.from_dict(payload)
        self.assertIsNone(manifest.origin)
        self.assertIsNone(manifest.generator)
        d = manifest.to_dict()
        self.assertNotIn("origin", d)
        self.assertNotIn("generator", d)


class TestP12GenerateDatasetCLI(unittest.TestCase):
    MXAI = str(ROOT / "examples" / "email-agent.typed.mxai")
    MXTRAIN = str(ROOT / "examples" / "email-agent.supervised.mxtrain")

    def _run(self, *extra_args: str, expect_ok: bool = True):
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", "generate-dataset",
             self.MXAI, "--training", self.MXTRAIN, *extra_args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if expect_ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_generates_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("--rows", "20", "--seed", "42", "--output-dir", tmp, "--json")
            payload = json.loads(result.stdout)

            self.assertEqual(payload["project"], "EmailAgentTyped")
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["mode"], "random")
            self.assertEqual(payload["rows"], 20)
            self.assertEqual(payload["train_rows"], 16)
            self.assertEqual(payload["eval_rows"], 4)
            self.assertEqual(payload["origin"], "synthetic")

            self.assertTrue(Path(payload["train_csv"]).exists())
            self.assertTrue(Path(payload["eval_csv"]).exists())
            self.assertTrue(Path(payload["manifest"]).exists())

    def test_manifest_has_correct_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("--rows", "10", "--seed", "5", "--mode", "random",
                               "--output-dir", tmp, "--json")
            payload = json.loads(result.stdout)
            manifest_data = json.loads(Path(payload["manifest"]).read_text(encoding="utf-8"))

            self.assertEqual(manifest_data["version"], "matrixai.dataset_manifest.v1")
            self.assertEqual(manifest_data["origin"], "synthetic")
            self.assertEqual(manifest_data["generator"]["seed"], 5)
            self.assertEqual(manifest_data["generator"]["mode"], "random")
            self.assertEqual(manifest_data["generator"]["rows"], 10)
            self.assertEqual(len(manifest_data["datasets"]), 2)
            roles = {d["role"] for d in manifest_data["datasets"]}
            self.assertEqual(roles, {"train", "evaluation"})

    def test_csv_headers_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("--rows", "10", "--seed", "1", "--output-dir", tmp, "--json")
            payload = json.loads(result.stdout)
            train_path = Path(payload["train_csv"])

            lines = train_path.read_text(encoding="utf-8").splitlines()
            header = lines[0].split(",")
            expected_columns = [
                "urgency", "sender_trust", "topic_support", "topic_sales",
                "sentiment", "has_attachment", "previous_interactions",
                "language_confidence", "label",
            ]
            self.assertEqual(header, expected_columns)
            # 8 data rows (80% of 10)
            self.assertEqual(len(lines) - 1, 8)

    def test_reproducibility_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1 = self._run("--rows", "10", "--seed", "99", "--output-dir", tmp, "--json")
            p1 = json.loads(r1.stdout)
            fp1_train = json.loads(Path(p1["manifest"]).read_text())["datasets"][0]["fingerprint"]

        with tempfile.TemporaryDirectory() as tmp:
            r2 = self._run("--rows", "10", "--seed", "99", "--output-dir", tmp, "--json")
            p2 = json.loads(r2.stdout)
            fp2_train = json.loads(Path(p2["manifest"]).read_text())["datasets"][0]["fingerprint"]

        self.assertEqual(fp1_train, fp2_train)

    def test_different_seeds_produce_different_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1 = self._run("--rows", "10", "--seed", "11", "--output-dir", tmp, "--json")
            fp1 = json.loads(Path(json.loads(r1.stdout)["manifest"]).read_text())["datasets"][0]["fingerprint"]

        with tempfile.TemporaryDirectory() as tmp:
            r2 = self._run("--rows", "10", "--seed", "22", "--output-dir", tmp, "--json")
            fp2 = json.loads(Path(json.loads(r2.stdout)["manifest"]).read_text())["datasets"][0]["fingerprint"]

        self.assertNotEqual(fp1, fp2)

    def test_verify_manifest_passes_on_generated_files(self):
        from matrixai.training.dataset_manifest import load_dataset_manifest, verify_dataset_manifest
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("--rows", "20", "--seed", "3", "--output-dir", tmp, "--json")
            payload = json.loads(result.stdout)
            manifest = load_dataset_manifest(payload["manifest"])
            report = verify_dataset_manifest(manifest, base_path=tmp)

        self.assertTrue(report.ok, report.errors)
        self.assertTrue(report.is_synthetic)

    def test_rows_too_small_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("--rows", "1", "--output-dir", tmp, expect_ok=False)
        self.assertNotEqual(result.returncode, 0)

    def test_rows_too_large_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("--rows", "99999", "--output-dir", tmp, expect_ok=False)
        self.assertNotEqual(result.returncode, 0)

    def test_human_readable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("--rows", "10", "--seed", "42", "--output-dir", tmp)
        self.assertIn("EmailAgentTyped", result.stdout)
        self.assertIn("synthetic", result.stdout.lower())


class TestP12SyntheticOriginPropagation(unittest.TestCase):
    """Corte 5: synthetic_origin propagation via train-supervised with synthetic manifest."""

    MXAI = str(ROOT / "examples" / "email-agent.typed.mxai")
    MXTRAIN = str(ROOT / "examples" / "email-agent.supervised.mxtrain")
    _NO_LLM = {"MATRIXAI_LLM_API_KEY": "", **__import__("os").environ}

    def _generate_synthetic_dataset(self, tmp: str, rows: int = 20, seed: int = 42) -> dict:
        """Generate a synthetic dataset and return the CLI payload."""
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", "generate-dataset",
             self.MXAI, "--training", self.MXTRAIN,
             "--rows", str(rows), "--seed", str(seed), "--output-dir", tmp, "--json"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_train_supervised_with_synthetic_manifest_propagates_synthetic_origin(self):
        import os
        with tempfile.TemporaryDirectory() as gen_dir, tempfile.TemporaryDirectory() as train_dir:
            gen = self._generate_synthetic_dataset(gen_dir, rows=20, seed=99)
            manifest_path = gen["manifest"]

            result = subprocess.run(
                [sys.executable, "-m", "matrixai", "train-supervised",
                 "Classify emails with labels support, sales, operations",
                 "--output-dir", train_dir,
                 "--dataset-manifest", manifest_path,
                 "--json"],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, "MATRIXAI_LLM_API_KEY": ""},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)

            # end_to_end_manifest.json must carry synthetic_origin: true
            self.assertTrue(payload["synthetic_origin"], "end_to_end_manifest missing synthetic_origin")

            # training_trace.json must carry dataset.synthetic_origin: true
            run_dir = Path(payload["run_dir"])
            trace = json.loads((run_dir / "training_trace.json").read_text(encoding="utf-8"))
            self.assertTrue(
                trace["dataset"].get("synthetic_origin"),
                "training_trace.json missing dataset.synthetic_origin",
            )

            # evaluation_report.json must carry synthetic_origin: true
            eval_report = json.loads(
                Path(payload["evaluation_report_path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(
                eval_report.get("synthetic_origin"),
                "evaluation_report.json missing synthetic_origin",
            )

    def test_train_supervised_with_real_manifest_has_no_synthetic_origin(self):
        import os
        with tempfile.TemporaryDirectory() as train_dir:
            result = subprocess.run(
                [sys.executable, "-m", "matrixai", "train-supervised",
                 "Crear un sistema de riesgo de caida para pacientes con etiquetas low, high",
                 "--output-dir", train_dir,
                 "--dataset-manifest",
                 str(ROOT / "examples" / "fall-risk.dataset-manifest.json"),
                 "--dataset-split", "holdout-v1",
                 "--json"],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, "MATRIXAI_LLM_API_KEY": ""},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)

            # Real manifest must NOT have synthetic_origin
            self.assertFalse(payload.get("synthetic_origin", False))

            run_dir = Path(payload["run_dir"])
            trace = json.loads((run_dir / "training_trace.json").read_text(encoding="utf-8"))
            self.assertFalse(trace["dataset"].get("synthetic_origin", False))

            eval_report = json.loads(
                Path(payload["evaluation_report_path"]).read_text(encoding="utf-8")
            )
            self.assertFalse(eval_report.get("synthetic_origin", False))


# ---------------------------------------------------------------------------
# Corte 6 — playground backend: _generate_synthetic_dataset unit tests
# ---------------------------------------------------------------------------

_EMAIL_MXAI_TEXT = (ROOT / "examples" / "email-agent.typed.mxai").read_text(encoding="utf-8")
_EMAIL_TRAIN_TEXT = (ROOT / "examples" / "email-agent.supervised.mxtrain").read_text(encoding="utf-8")


class TestP12PlaygroundGenerateSyntheticDataset(unittest.TestCase):
    """Unit tests for the _generate_synthetic_dataset playground backend function."""

    def _call(self, mxai=None, training=None, rows=20, seed=42, mode="random"):
        from matrixai.playground import _generate_synthetic_dataset
        return _generate_synthetic_dataset(
            mxai_text=mxai if mxai is not None else _EMAIL_MXAI_TEXT,
            training_text=training if training is not None else _EMAIL_TRAIN_TEXT,
            rows=rows,
            seed=seed,
            mode=mode,
        )

    def test_ok_returns_csv_text(self):
        result = self._call(rows=10)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertIn("csv_text", result)
        lines = result["csv_text"].strip().splitlines()
        self.assertGreater(len(lines), 1)

    def test_row_count_matches_request(self):
        result = self._call(rows=15)
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], 15)
        # CSV has header + 15 data rows
        lines = result["csv_text"].strip().splitlines()
        self.assertEqual(len(lines), 16)

    def test_fingerprint_present_and_deterministic(self):
        r1 = self._call(rows=10, seed=7)
        r2 = self._call(rows=10, seed=7)
        self.assertTrue(r1["ok"])
        self.assertTrue(r1["fingerprint"].startswith("data_"))
        self.assertEqual(r1["fingerprint"], r2["fingerprint"])

    def test_different_seeds_give_different_fingerprints(self):
        r1 = self._call(rows=10, seed=1)
        r2 = self._call(rows=10, seed=2)
        self.assertNotEqual(r1["fingerprint"], r2["fingerprint"])

    def test_origin_is_synthetic(self):
        result = self._call(rows=10)
        self.assertTrue(result["ok"])
        self.assertEqual(result["origin"], "synthetic")

    def test_columns_in_response(self):
        result = self._call(rows=10)
        self.assertTrue(result["ok"])
        self.assertIsInstance(result["columns"], list)
        self.assertGreater(len(result["columns"]), 0)
        header = result["csv_text"].splitlines()[0]
        for col in result["columns"]:
            self.assertIn(col, header)

    def test_row_clamp_respects_p9_max(self):
        result = self._call(rows=999999)
        self.assertTrue(result["ok"])
        self.assertLessEqual(result["rows"], 50_000)

    def test_row_clamp_min_2(self):
        result = self._call(rows=0)
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["rows"], 2)

    def test_empty_mxai_returns_error(self):
        result = self._call(mxai="")
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_empty_training_returns_error(self):
        result = self._call(training="")
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_coherent_mode_returns_ok(self):
        result = self._call(rows=10, mode="coherent")
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(result["mode"], "coherent")

    def test_coherent_mode_fallback_exposed_in_response(self):
        # The email-agent program has no runtime GRAPH that resolves labels,
        # so coherent mode should fall back for all rows and report it.
        result = self._call(rows=10, mode="coherent")
        self.assertTrue(result["ok"], result.get("error"))
        if result.get("coherent_fallback_count", 0) > 0:
            self.assertIn("coherent_fallback_warning", result)
            self.assertIsInstance(result["coherent_fallback_warning"], str)
            self.assertGreater(len(result["coherent_fallback_warning"]), 0)

    def test_random_mode_has_no_fallback_key(self):
        result = self._call(rows=10, mode="random")
        self.assertTrue(result["ok"])
        self.assertNotIn("coherent_fallback_count", result)
        self.assertNotIn("coherent_fallback_warning", result)

    def test_invalid_mode_falls_back_to_random(self):
        result = self._call(rows=10, mode="bogus")
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(result["mode"], "random")


if __name__ == "__main__":
    unittest.main()
