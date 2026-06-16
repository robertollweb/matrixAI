from __future__ import annotations

import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
KELVIN_MODEL_PATH = ROOT / "examples/celsius_to_kelvin.mxai"

KELVIN_MXTRAIN = """
    MODEL examples/celsius_to_kelvin.mxai

    DATASET CelsiusToKelvinSet
      SOURCE csv("examples/celsius_to_kelvin.train.csv")
      INPUT Reading FROM COLUMNS [celsius]
      TARGET predicted_kelvin: Scalar[0, 1000]
      SPLIT train=0.8 validation=0.2 seed=42
      BATCH size=4 shuffle=true
    END

    LOSS KelvinLoss
      TYPE mse
      PREDICTION predicted_kelvin
      TARGET predicted_kelvin
    END

    OPTIMIZER KelvinOptimizer
      TYPE sgd
      LEARNING_RATE 0.001
      UPDATE W1, b1
    END

    METRIC KelvinMAE
      TYPE mae
      PREDICTION predicted_kelvin
      TARGET predicted_kelvin
    END

    RUN
      EPOCHS 5
      SAVE_BEST true
    END
"""


def _make_generator(mode: str = "random", rows: int = 10, seed: int = 42):
    from matrixai.parser import parse_file
    from matrixai.training.parser import parse_training_text
    from matrixai.training.synthetic import SyntheticDataGenerator
    program = parse_file(KELVIN_MODEL_PATH)
    spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
    return SyntheticDataGenerator(program, spec, seed=seed, rows=rows, mode=mode)


class TestSyntheticRegressionRandom(unittest.TestCase):

    def test_random_returns_adapter(self):
        gen = _make_generator("random")
        adapter = gen.generate()
        self.assertIsNotNone(adapter)

    def test_random_labels_empty(self):
        gen = _make_generator("random")
        adapter = gen.generate()
        self.assertEqual(adapter.labels, [])

    def test_random_correct_row_count(self):
        gen = _make_generator("random", rows=8)
        adapter = gen.generate()
        self.assertEqual(len(adapter.examples()), 8)

    def test_random_target_is_numeric(self):
        gen = _make_generator("random", rows=20)
        adapter = gen.generate()
        for example in adapter.examples():
            float(example.label)  # must be parseable as float

    def test_random_target_within_scalar_range(self):
        gen = _make_generator("random", rows=50)
        adapter = gen.generate()
        for example in adapter.examples():
            val = float(example.label)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1000.0)

    def test_random_target_value_field_set(self):
        gen = _make_generator("random", rows=5)
        adapter = gen.generate()
        for example in adapter.examples():
            self.assertIsNotNone(example.target_value)
            self.assertAlmostEqual(example.target_value, float(example.label), places=6)

    def test_random_input_vector_present(self):
        gen = _make_generator("random", rows=5)
        adapter = gen.generate()
        for example in adapter.examples():
            self.assertEqual(len(example.vector), 1)  # celsius: Scalar → 1 feature

    def test_random_reproducible_with_same_seed(self):
        gen1 = _make_generator("random", seed=99)
        gen2 = _make_generator("random", seed=99)
        vals1 = [float(e.label) for e in gen1.generate().examples()]
        vals2 = [float(e.label) for e in gen2.generate().examples()]
        self.assertEqual(vals1, vals2)

    def test_random_different_with_different_seed(self):
        gen1 = _make_generator("random", seed=1)
        gen2 = _make_generator("random", seed=2)
        vals1 = [float(e.label) for e in gen1.generate().examples()]
        vals2 = [float(e.label) for e in gen2.generate().examples()]
        self.assertNotEqual(vals1, vals2)


class TestSyntheticRegressionCoherent(unittest.TestCase):

    def test_coherent_returns_adapter(self):
        gen = _make_generator("coherent", rows=4)
        adapter = gen.generate()
        self.assertIsNotNone(adapter)

    def test_coherent_labels_empty(self):
        gen = _make_generator("coherent", rows=4)
        adapter = gen.generate()
        self.assertEqual(adapter.labels, [])

    def test_coherent_target_is_numeric(self):
        gen = _make_generator("coherent", rows=4)
        adapter = gen.generate()
        for example in adapter.examples():
            float(example.label)  # must be parseable

    def test_coherent_target_value_is_float(self):
        gen = _make_generator("coherent", rows=4)
        adapter = gen.generate()
        for example in adapter.examples():
            self.assertIsNotNone(example.target_value)
            self.assertIsInstance(example.target_value, float)


class TestSyntheticRegressionRangeHelper(unittest.TestCase):

    def test_regression_range_uses_scalar_bounds(self):
        from matrixai.parser import parse_file
        from matrixai.training.parser import parse_training_text
        from matrixai.training.synthetic import SyntheticDataGenerator
        program = parse_file(KELVIN_MODEL_PATH)
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        gen = SyntheticDataGenerator(program, spec, seed=0, rows=1, mode="random")
        target_type = spec.dataset.target.type
        lo, hi = gen._regression_range(target_type)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1000.0)

    def test_regression_range_defaults_when_no_range(self):
        from matrixai.parser import parse_file
        from matrixai.training.parser import parse_training_text
        from matrixai.training.synthetic import SyntheticDataGenerator
        from matrixai.types import TypeSpec
        program = parse_file(KELVIN_MODEL_PATH)
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        gen = SyntheticDataGenerator(program, spec, seed=0, rows=1)
        bare_scalar = TypeSpec(name="Scalar", parameters={})
        lo, hi = gen._regression_range(bare_scalar)
        self.assertEqual(lo, -1.0)
        self.assertEqual(hi, 1.0)

    def test_classification_spec_still_raises_without_labels(self):
        import dataclasses
        from matrixai.parser import parse_file
        from matrixai.training.parser import parse_training_text
        from matrixai.training.spec import DatasetTargetSpec
        from matrixai.training.synthetic import SyntheticDataGenerator
        from matrixai.types import TypeSpec
        program = parse_file(KELVIN_MODEL_PATH)
        spec = parse_training_text(textwrap.dedent("""
            MODEL examples/celsius_to_kelvin.mxai
            DATASET D
              SOURCE csv("examples/celsius_to_kelvin.train.csv")
              INPUT Reading FROM COLUMNS [celsius]
              TARGET label: Label[a, b]
              SPLIT train=0.8 validation=0.2 seed=0
              BATCH size=4 shuffle=false
            END
            LOSS L
              TYPE cross_entropy
              PREDICTION predicted_kelvin
              TARGET label
            END
            OPTIMIZER O
              TYPE sgd
              LEARNING_RATE 0.01
              UPDATE W1, b1
            END
            RUN
              EPOCHS 1
              SAVE_BEST false
            END
        """).strip())
        # Simulate Label[] with no args (classification without labels → should raise)
        bad_target = DatasetTargetSpec(name="label", type=TypeSpec(name="Label", parameters={}))
        bad_dataset = dataclasses.replace(spec.dataset, target=bad_target)
        bad_spec = dataclasses.replace(spec, dataset=bad_dataset)
        gen = SyntheticDataGenerator(program, bad_spec, seed=0, rows=1)
        with self.assertRaises(ValueError):
            gen.generate()


if __name__ == "__main__":
    unittest.main()
