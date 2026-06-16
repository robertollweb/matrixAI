from __future__ import annotations

import json
import math
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent

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


def _parse_kelvin_spec():
    from matrixai.training.parser import parse_training_text
    return parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())


class TestExpectedSemanticKind(unittest.TestCase):

    def test_mse_returns_linear_regression(self):
        from matrixai.training.trainer import _expected_semantic_kind
        self.assertEqual(_expected_semantic_kind("mse"), "linear_regression")

    def test_cross_entropy_still_softmax(self):
        from matrixai.training.trainer import _expected_semantic_kind
        self.assertEqual(_expected_semantic_kind("cross_entropy"), "softmax_linear")

    def test_binary_cross_entropy_still_sigmoid(self):
        from matrixai.training.trainer import _expected_semantic_kind
        self.assertEqual(_expected_semantic_kind("binary_cross_entropy"), "sigmoid_linear")


class TestLabelsForMSE(unittest.TestCase):

    def test_labels_empty_for_mse(self):
        from matrixai.training.trainer import SupervisedTrainer
        spec = _parse_kelvin_spec()
        trainer = SupervisedTrainer()
        labels = trainer._labels(spec)
        self.assertEqual(labels, [])

    def test_labels_non_empty_for_classification(self):
        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedTrainer
        if not (ROOT / "examples/email-agent.mxai").exists():
            self.skipTest("email-agent.mxai not available")
        mxtrain = textwrap.dedent("""
            MODEL examples/email-agent.mxai
            DATASET D
              SOURCE csv("examples/email-agent.train.csv")
              INPUT EmailFeatures FROM COLUMNS [urgency]
              TARGET label: Label[support, sales]
              SPLIT train=0.8 validation=0.2 seed=0
              BATCH size=4 shuffle=false
            END
            LOSS L
              TYPE cross_entropy
              PREDICTION email_category
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
        """).strip()
        spec = parse_training_text(mxtrain)
        labels = SupervisedTrainer()._labels(spec)
        self.assertIn("support", labels)


class TestMSERegressionGradients(unittest.TestCase):

    def test_gradient_direction_weight(self):
        from matrixai.training.trainer import _mse_regression_gradients
        # y_hat = dot([2.0], [0.5]) + 0.0 = 1.0, y = 3.0
        # delta = 2*(1.0 - 3.0)/1 = -4.0
        # weight_grad[0] = -4.0 * 2.0 = -8.0 (negative means weight should increase)
        grads = _mse_regression_gradients([[2.0]], ["3.0"], [0.5], 0.0)
        self.assertLess(grads["weights"][0], 0.0, "gradient should be negative when prediction < target")

    def test_gradient_direction_bias(self):
        from matrixai.training.trainer import _mse_regression_gradients
        grads = _mse_regression_gradients([[1.0]], ["5.0"], [0.0], 0.0)
        # y_hat=0, y=5, delta=2*(0-5)/1=-10, bias_grad=-10
        self.assertLess(grads["bias"], 0.0, "bias gradient should be negative when prediction < target")

    def test_gradient_values_exact(self):
        from matrixai.training.trainer import _mse_regression_gradients
        # Single sample: x=[1.0], w=[0.5], b=0.0, y=2.0
        # y_hat = 0.5, delta = 2*(0.5-2.0)/1 = -3.0
        # weight_grad = -3.0 * 1.0 = -3.0, bias_grad = -3.0
        grads = _mse_regression_gradients([[1.0]], ["2.0"], [0.5], 0.0)
        self.assertAlmostEqual(grads["weights"][0], -3.0, places=10)
        self.assertAlmostEqual(grads["bias"], -3.0, places=10)

    def test_gradient_batch_averaged(self):
        from matrixai.training.trainer import _mse_regression_gradients
        # Two samples, same x=[1.0], targets 2.0 and 4.0, w=[1.0], b=0.0
        # sample1: y_hat=1, y=2, delta1 = 2*(1-2)/2 = -1.0, wg1=-1.0
        # sample2: y_hat=1, y=4, delta2 = 2*(1-4)/2 = -3.0, wg2=-3.0
        # total wg = -4.0
        grads = _mse_regression_gradients([[1.0], [1.0]], ["2.0", "4.0"], [1.0], 0.0)
        self.assertAlmostEqual(grads["weights"][0], -4.0, places=10)

    def test_gradient_zero_error(self):
        from matrixai.training.trainer import _mse_regression_gradients
        # Perfect prediction: y_hat == y → gradient == 0
        grads = _mse_regression_gradients([[1.0]], ["1.0"], [1.0], 0.0)
        self.assertAlmostEqual(grads["weights"][0], 0.0, places=10)
        self.assertAlmostEqual(grads["bias"], 0.0, places=10)


class TestMSERegressionMetrics(unittest.TestCase):

    def _make_examples(self, xs, ys):
        from matrixai.training.data import SupervisedExample
        return [
            SupervisedExample(vector=[x], label=str(y), row_index=i, row_hash=f"h{i}")
            for i, (x, y) in enumerate(zip(xs, ys))
        ]

    def test_mse_perfect_prediction(self):
        from matrixai.training.trainer import _mse_regression_metrics
        # w=[1.0], b=273.15 → y_hat = x + 273.15, which matches y
        examples = self._make_examples([-40.0, 0.0, 20.0], [233.15, 273.15, 293.15])
        metrics = _mse_regression_metrics(examples, [1.0], 273.15)
        self.assertAlmostEqual(metrics["loss"], 0.0, places=5)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=5)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=5)
        self.assertAlmostEqual(metrics["r2"], 1.0, places=5)

    def test_mse_known_error(self):
        from matrixai.training.trainer import _mse_regression_metrics
        # w=[0], b=0 → y_hat=0, y=2.0 → error=4.0, MAE=2.0, RMSE=2.0
        examples = self._make_examples([1.0], [2.0])
        metrics = _mse_regression_metrics(examples, [0.0], 0.0)
        self.assertAlmostEqual(metrics["loss"], 4.0, places=10)
        self.assertAlmostEqual(metrics["mae"], 2.0, places=10)
        self.assertAlmostEqual(metrics["rmse"], 2.0, places=10)

    def test_r2_below_one_for_imperfect(self):
        from matrixai.training.trainer import _mse_regression_metrics
        examples = self._make_examples([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        # Prediction w=[0], b=2 → y_hat=2 for all → not perfect
        metrics = _mse_regression_metrics(examples, [0.0], 2.0)
        self.assertLess(metrics["r2"], 1.0)

    def test_no_confusion_matrix(self):
        from matrixai.training.trainer import _mse_regression_metrics
        examples = self._make_examples([1.0], [1.0])
        metrics = _mse_regression_metrics(examples, [1.0], 0.0)
        self.assertEqual(metrics["confusion_matrix"], {})
        self.assertEqual(metrics["per_label"], {})

    def test_accuracy_zero(self):
        from matrixai.training.trainer import _mse_regression_metrics
        examples = self._make_examples([1.0], [1.0])
        metrics = _mse_regression_metrics(examples, [1.0], 0.0)
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_mae_is_mean_abs_error(self):
        from matrixai.training.trainer import _mse_regression_metrics
        # w=[1], b=0: y_hat=x, y targets [2,4], x=[1,3]
        # errors: |1-2|=1, |3-4|=1 → MAE=1.0, MSE=1.0
        examples = self._make_examples([1.0, 3.0], [2.0, 4.0])
        metrics = _mse_regression_metrics(examples, [1.0], 0.0)
        self.assertAlmostEqual(metrics["mae"], 1.0, places=10)
        self.assertAlmostEqual(metrics["loss"], 1.0, places=10)


class TestSupervisedTrainerRegressionIntegration(unittest.TestCase):

    def test_train_runs_without_error(self):
        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedTrainer
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SupervisedTrainer().train(spec, output_dir=tmpdir + "/run", base_path=str(ROOT))
        self.assertIsNotNone(result)

    def test_task_kind_regression_in_trace(self):
        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedTrainer
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = tmpdir + "/run"
            SupervisedTrainer().train(spec, output_dir=run_dir, base_path=str(ROOT))
            trace = json.loads(Path(run_dir + "/training_trace.json").read_text())
        self.assertEqual(trace["task_kind"], "regression")

    def test_mae_in_metrics_json(self):
        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedTrainer
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = tmpdir + "/run"
            SupervisedTrainer().train(spec, output_dir=run_dir, base_path=str(ROOT))
            metrics = json.loads(Path(run_dir + "/metrics.json").read_text())
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

    def test_canonical_celsius_example_converges(self):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.trainer import SupervisedTrainer
        spec = parse_training_file(ROOT / "examples/celsius_to_kelvin.mxtrain")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            SupervisedTrainer().train(spec, output_dir=run_dir, base_path=str(ROOT))
            metrics = json.loads((run_dir / "metrics.json").read_text())
            params = json.loads((run_dir / "params.best.json").read_text())
        self.assertLess(metrics["mae"], 0.01)
        self.assertAlmostEqual(params["parameters"]["W1"]["values"][0], 1.0, places=3)
        self.assertAlmostEqual(params["parameters"]["b1"]["values"], 273.15, places=2)

    def test_torch_regression_is_explicitly_gated(self):
        from matrixai.training.parser import parse_training_text
        from matrixai.training.torch_trainer import TorchSupervisedTrainer
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(NotImplementedError) as ctx:
                TorchSupervisedTrainer().train(spec, output_dir=tmpdir + "/run", base_path=str(ROOT))
        self.assertIn("P17.1", str(ctx.exception))

    def test_loss_decreases_with_training(self):
        import csv as csv_mod
        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedTrainer

        # Use normalized values (y in [0,1]) so LR=0.01 is stable
        xs = [i / 20.0 for i in range(20)]
        rows = [(x, 0.5 * x + 0.1) for x in xs]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            with csv_path.open("w", newline="") as fh:
                w = csv_mod.writer(fh)
                w.writerow(["celsius", "predicted_kelvin"])
                w.writerows(rows)
            model_path = Path(tmpdir) / "model.mxai"
            model_path.write_text((ROOT / "examples/celsius_to_kelvin.mxai").read_text())
            mxtrain = textwrap.dedent("""
                MODEL model.mxai
                DATASET D
                  SOURCE csv("data.csv")
                  INPUT Reading FROM COLUMNS [celsius]
                  TARGET predicted_kelvin: Scalar[0, 1]
                  SPLIT train=0.8 validation=0.2 seed=42
                  BATCH size=4 shuffle=false
                END
                LOSS L
                  TYPE mse
                  PREDICTION predicted_kelvin
                  TARGET predicted_kelvin
                END
                OPTIMIZER O
                  TYPE sgd
                  LEARNING_RATE 0.01
                  UPDATE W1, b1
                END
                RUN
                  EPOCHS 100
                  SAVE_BEST true
                END
            """).strip()
            spec = parse_training_text(mxtrain)
            run_dir = Path(tmpdir) / "run"
            SupervisedTrainer().train(spec, output_dir=str(run_dir), base_path=tmpdir)
            trace = json.loads((run_dir / "training_trace.json").read_text())

        first_loss = trace["epochs"][0]["validation_loss"]
        last_loss = trace["epochs"][-1]["validation_loss"]
        self.assertLess(last_loss, first_loss, "validation loss should decrease over 100 epochs")


class TestSupervisedEvaluatorRegression(unittest.TestCase):

    def test_evaluator_returns_mae_rmse_r2(self):
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file
        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedEvaluator
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        program = parse_file(ROOT / "examples/celsius_to_kelvin.mxai")
        params = build_initial_parameter_set(program)
        result = SupervisedEvaluator().evaluate(spec, params, base_path=str(ROOT))
        self.assertTrue(result.is_regression())
        d = result.to_dict()
        self.assertIn("mae", d)
        self.assertIn("rmse", d)
        self.assertIn("r2", d)

    def test_evaluator_perfect_weights_gives_low_mae(self):
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file
        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedEvaluator
        spec = parse_training_text(textwrap.dedent(KELVIN_MXTRAIN).strip())
        program = parse_file(ROOT / "examples/celsius_to_kelvin.mxai")
        params = build_initial_parameter_set(program)
        # Override to perfect weights
        data = params.to_dict()
        data["parameters"]["W1"]["values"] = [1.0]
        data["parameters"]["b1"]["values"] = 273.15
        from matrixai.parameters import ParameterSet
        perfect_params = ParameterSet.from_dict(data)
        result = SupervisedEvaluator().evaluate(spec, perfect_params, base_path=str(ROOT))
        self.assertAlmostEqual(result.mae, 0.0, places=3)
        self.assertAlmostEqual(result.r2, 1.0, places=3)


class TestSupervisedExampleTargetValue(unittest.TestCase):

    def test_csv_adapter_sets_target_value_for_regression(self):
        from matrixai.training.data import CSVDataAdapter
        adapter = CSVDataAdapter(
            ROOT / "examples/celsius_to_kelvin.train.csv",
            "Reading",
            ["celsius"],
            "predicted_kelvin",
            labels=[],
        )
        examples = adapter.examples()
        self.assertTrue(all(e.target_value is not None for e in examples))
        # First row: celsius=-40, predicted_kelvin=233.15
        ex = examples[0]
        self.assertAlmostEqual(ex.target_value, float(ex.label), places=10)

    def test_csv_adapter_no_target_value_for_classification(self):
        from matrixai.training.data import CSVDataAdapter
        if not (ROOT / "examples/email-agent.train.csv").exists():
            self.skipTest("email-agent.train.csv not available")
        adapter = CSVDataAdapter(
            ROOT / "examples/email-agent.train.csv",
            "EmailFeatures",
            ["urgency"],
            "label",
            labels=["support", "sales", "operations"],
        )
        examples = adapter.examples()
        self.assertTrue(all(e.target_value is None for e in examples))


if __name__ == "__main__":
    unittest.main()
