from __future__ import annotations

import textwrap
import unittest
from pathlib import Path

from matrixai.training.parser import parse_training_text
from matrixai.training.verifier import TrainingVerifier

ROOT = Path(__file__).parent.parent


def _verify(mxtrain: str, base: str = str(ROOT)) -> object:
    spec = parse_training_text(textwrap.dedent(mxtrain).strip())
    return TrainingVerifier().verify(spec, base_path=base)


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
      EPOCHS 50
      SAVE_BEST true
    END
"""

EMAIL_MXTRAIN = """
    MODEL examples/email-agent.mxai

    DATASET EmailSet
      SOURCE csv("examples/email-agent.train.csv")
      INPUT EmailFeatures FROM COLUMNS [urgency, sender_trust, topic_support, topic_sales, sentiment, has_attachment, previous_interactions, language_confidence]
      TARGET label: Label[support, sales, operations]
      SPLIT train=0.8 validation=0.2 seed=42
      BATCH size=4 shuffle=true
    END

    LOSS EmailLoss
      TYPE cross_entropy
      PREDICTION email_category
      TARGET label
    END

    OPTIMIZER EmailOpt
      TYPE sgd
      LEARNING_RATE 0.05
      UPDATE W1, b1
    END

    METRIC EmailAccuracy
      TYPE accuracy
      PREDICTION email_category
      TARGET label
    END

    RUN
      EPOCHS 20
      SAVE_BEST true
    END
"""


class TestVerifierMSEAccepts(unittest.TestCase):

    def test_kelvin_mxtrain_verifies_ok(self):
        r = _verify(KELVIN_MXTRAIN)
        self.assertTrue(r.ok, msg=r.errors)

    def test_kelvin_trainable_parameters_found(self):
        r = _verify(KELVIN_MXTRAIN)
        self.assertTrue(r.ok)
        names = {p["name"] for p in r.trainable_parameters}
        self.assertIn("W1", names)
        self.assertIn("b1", names)

    def test_kelvin_differentiability_ok(self):
        r = _verify(KELVIN_MXTRAIN)
        self.assertTrue(r.ok)
        self.assertTrue(r.differentiability.get("ok"), msg=r.differentiability)

    def test_mae_metric_accepted_for_mse(self):
        r = _verify(KELVIN_MXTRAIN)
        self.assertTrue(r.ok, msg=r.errors)

    def test_no_errors_in_dataset_validation(self):
        r = _verify(KELVIN_MXTRAIN)
        self.assertFalse(any("DATASET row" in e for e in r.errors), msg=r.errors)


class TestVerifierMSERejects(unittest.TestCase):

    def test_mse_with_label_target_rejected(self):
        bad = KELVIN_MXTRAIN.replace(
            "TARGET predicted_kelvin: Scalar[0, 1000]",
            "TARGET predicted_kelvin: Label[bajo, medio, alto]",
        )
        r = _verify(bad)
        self.assertFalse(r.ok)
        self.assertTrue(any("mse expects Scalar or Integer" in e for e in r.errors), msg=r.errors)

    def test_mse_with_accuracy_metric_rejected(self):
        bad = KELVIN_MXTRAIN.replace("TYPE mae", "TYPE accuracy")
        r = _verify(bad)
        self.assertFalse(r.ok)
        self.assertTrue(any("not valid for mse loss" in e for e in r.errors), msg=r.errors)

    def test_mse_with_rmse_metric_accepted(self):
        good = KELVIN_MXTRAIN.replace("TYPE mae", "TYPE rmse")
        r = _verify(good)
        self.assertTrue(r.ok, msg=r.errors)

    def test_mse_with_r2_metric_accepted(self):
        good = KELVIN_MXTRAIN.replace("TYPE mae", "TYPE r2")
        r = _verify(good)
        self.assertTrue(r.ok, msg=r.errors)


class TestVerifierClassificationRejectsScalarTarget(unittest.TestCase):

    def test_cross_entropy_with_scalar_target_rejected(self):
        bad = """
            MODEL examples/celsius_to_kelvin.mxai

            DATASET D
              SOURCE csv("examples/celsius_to_kelvin.train.csv")
              INPUT Reading FROM COLUMNS [celsius]
              TARGET predicted_kelvin: Scalar[0, 1000]
              SPLIT train=0.8 validation=0.2 seed=42
              BATCH size=4 shuffle=true
            END

            LOSS L
              TYPE cross_entropy
              PREDICTION predicted_kelvin
              TARGET predicted_kelvin
            END

            OPTIMIZER O
              TYPE sgd
              LEARNING_RATE 0.05
              UPDATE W1, b1
            END

            RUN
              EPOCHS 10
              SAVE_BEST true
            END
        """
        r = _verify(bad)
        self.assertFalse(r.ok)
        self.assertTrue(
            any("cross_entropy expects Label" in e for e in r.errors),
            msg=r.errors,
        )

    def test_binary_cross_entropy_with_scalar_target_rejected(self):
        bad = """
            MODEL examples/celsius_to_kelvin.mxai

            DATASET D
              SOURCE csv("examples/celsius_to_kelvin.train.csv")
              INPUT Reading FROM COLUMNS [celsius]
              TARGET predicted_kelvin: Scalar[0, 1000]
              SPLIT train=0.8 validation=0.2 seed=42
              BATCH size=4 shuffle=true
            END

            LOSS L
              TYPE binary_cross_entropy
              PREDICTION predicted_kelvin
              TARGET predicted_kelvin
            END

            OPTIMIZER O
              TYPE sgd
              LEARNING_RATE 0.05
              UPDATE W1, b1
            END

            RUN
              EPOCHS 10
              SAVE_BEST true
            END
        """
        r = _verify(bad)
        self.assertFalse(r.ok)
        self.assertTrue(
            any("binary_cross_entropy expects Label" in e for e in r.errors),
            msg=r.errors,
        )

    def test_classification_with_regression_metric_rejected(self):
        if not Path(ROOT / "examples/email-agent.mxai").exists():
            self.skipTest("email-agent.mxai not available")
        bad = EMAIL_MXTRAIN.replace("TYPE accuracy", "TYPE mae")
        r = _verify(bad)
        self.assertFalse(r.ok)
        self.assertTrue(any("requires mse loss" in e for e in r.errors), msg=r.errors)


def _verify_with_csv(mxai_path: str, csv_content: str, mxtrain_template: str) -> object:
    """Verify using an in-memory CSV written to a unique temp path."""
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "model.mxai").write_text(Path(ROOT / mxai_path).read_text())
        (tmp / "data.csv").write_text(csv_content)
        mxtrain = mxtrain_template.replace(
            f"MODEL examples/celsius_to_kelvin.mxai", "MODEL model.mxai"
        ).replace(
            f'SOURCE csv("examples/celsius_to_kelvin.train.csv")', 'SOURCE csv("data.csv")'
        )
        spec = parse_training_text(textwrap.dedent(mxtrain).strip())
        from matrixai.training.verifier import TrainingVerifier
        return TrainingVerifier().verify(spec, base_path=tmpdir)


class TestVerifierDatasetContinuousTarget(unittest.TestCase):

    def test_dataset_numeric_target_accepted(self):
        r = _verify(KELVIN_MXTRAIN)
        self.assertFalse(any("must be one of" in e for e in r.errors), msg=r.errors)

    def test_dataset_non_numeric_target_rejected(self):
        csv_content = "celsius,predicted_kelvin\n0.0,not_a_number\n20.0,also_bad\n"
        r = _verify_with_csv("examples/celsius_to_kelvin.mxai", csv_content, KELVIN_MXTRAIN)
        self.assertFalse(r.ok)
        self.assertTrue(any("must be numeric" in e for e in r.errors), msg=r.errors)

    def test_dataset_out_of_range_target_rejected(self):
        csv_content = "celsius,predicted_kelvin\n0.0,9999.0\n20.0,9999.0\n"
        r = _verify_with_csv("examples/celsius_to_kelvin.mxai", csv_content, KELVIN_MXTRAIN)
        self.assertFalse(r.ok)
        self.assertTrue(any("range" in e for e in r.errors), msg=r.errors)


class TestEvaluationResultRegression(unittest.TestCase):

    def test_is_regression_when_no_labels(self):
        from matrixai.training.spec import EvaluationResult
        result = EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s",
            parameter_set_id="p", dataset="d", dataset_fingerprint="f",
            dataset_schema={}, rows=10, loss=0.01, accuracy=0.0,
            labels=[], confusion_matrix={}, per_label={},
            macro_precision=0.0, macro_recall=0.0, macro_f1=0.0,
            mae=0.05, rmse=0.1, r2=0.99,
        )
        self.assertTrue(result.is_regression())
        d = result.to_dict()
        self.assertIn("mae", d)
        self.assertIn("rmse", d)
        self.assertIn("r2", d)
        self.assertAlmostEqual(d["mae"], 0.05)

    def test_is_not_regression_when_labels(self):
        from matrixai.training.spec import EvaluationResult
        result = EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s",
            parameter_set_id="p", dataset="d", dataset_fingerprint="f",
            dataset_schema={}, rows=10, loss=0.5, accuracy=0.9,
            labels=["a", "b"], confusion_matrix={}, per_label={},
            macro_precision=0.9, macro_recall=0.9, macro_f1=0.9,
        )
        self.assertFalse(result.is_regression())
        d = result.to_dict()
        self.assertNotIn("mae", d)
        self.assertNotIn("rmse", d)
        self.assertNotIn("r2", d)


if __name__ == "__main__":
    unittest.main()
