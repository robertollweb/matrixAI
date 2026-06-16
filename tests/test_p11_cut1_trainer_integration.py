"""P11 Cut 1 — Integration: TrainingVerifier UPDATE patterns + GenericSupervisedTrainer.

Tests for:
- TrainingVerifier._verify_updates accepts *, exact, and prefix.* patterns
- TrainingVerifier._verify_updates rejects patterns that match no parameter
- GenericSupervisedTrainer resolves trainable keys from update_patterns
- GenericSupervisedTrainer returns expected output shape and epoch_trace
- GenericSupervisedTrainer reduces loss on a simple deterministic task
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Any

from matrixai.parser import parse_text
from matrixai.training.spec import (
    DatasetInputSpec,
    DatasetSplitSpec,
    DatasetTargetSpec,
    DatasetSpec,
    LossSpec,
    OptimizerSpec,
    TrainingSpec,
)
from matrixai.training.trainer import GenericSupervisedTrainer, match_update_patterns
from matrixai.training.verifier import TrainingVerifier
from matrixai.types import TypeSpec


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_PROG_CHAIN = """\
PROJECT chain_trainer_test
LAYER Chain(Tensor[4]) -> Tensor[4]
  PARAM W Tensor[4, 4]
  hidden = matmul(input, W)
  result = relu(hidden)
END
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(Chain, X)
END
GRAPH
  X -> F
END
"""

_PROG_TWO_LAYERS = """\
PROJECT two_layer_test
LAYER Enc(Tensor[4]) -> Tensor[4]
  PARAM W Tensor[4, 4]
  result = matmul(input, W)
END
LAYER Head(Tensor[4]) -> Tensor[4]
  PARAM W Tensor[4, 4]
  result = matmul(input, W)
END
VECTOR X[4]
  x1 : Scalar
  x2 : Scalar
  x3 : Scalar
  x4 : Scalar
END
FUNCTION F
  result = call_layer(Enc, X)
END
GRAPH
  X -> F
END
"""


# Minimal example type for GenericSupervisedTrainer (vector as field dict)
@dataclass
class _LayerExample:
    vector_name: str
    vector: dict[str, float]
    label: str


def _make_optimizer(update: list[str]) -> OptimizerSpec:
    return OptimizerSpec(name="opt", type="sgd", learning_rate=0.01, update=update)


def _make_training(update: list[str]) -> TrainingSpec:
    return TrainingSpec(
        model="dummy.mxai",
        dataset=DatasetSpec(
            name="dummy",
            source_kind="csv",
            source="dummy.csv",
            input=DatasetInputSpec(vector="X", columns=["x1"]),
            target=DatasetTargetSpec(name="label", type=TypeSpec("Label")),
        ),
        loss=LossSpec(name="L", type="cross_entropy", prediction="F", target="label"),
        optimizer=_make_optimizer(update),
    )


# ---------------------------------------------------------------------------
# TrainingVerifier UPDATE pattern integration
# ---------------------------------------------------------------------------

class TestVerifierUpdatePatterns(unittest.TestCase):

    _PARAMS_SCALE = [
        {"function": "Scale", "name": "W", "role": "weights", "shape": [4, 4]},
        {"function": "Scale", "name": "b", "role": "bias", "shape": [4]},
    ]

    _PARAMS_TWO_LAYERS = [
        {"function": "Encoder", "name": "Wq", "role": "weights", "shape": [4, 4]},
        {"function": "Encoder", "name": "Wk", "role": "weights", "shape": [4, 4]},
        {"function": "Classifier", "name": "W", "role": "weights", "shape": [4, 2]},
    ]

    def _errors(self, update: list[str], params: list[dict[str, Any]]) -> list[str]:
        return TrainingVerifier()._verify_updates(_make_training(update), params)

    def test_wildcard_star_accepted_for_all_params(self):
        errors = self._errors(["*"], self._PARAMS_SCALE)
        self.assertEqual(errors, [])

    def test_exact_hierarchical_path_accepted(self):
        errors = self._errors(["Scale.W"], self._PARAMS_SCALE)
        self.assertEqual(errors, [])

    def test_exact_flat_name_accepted(self):
        # Flat names (e.g. "W", "b") are still valid for backward compat
        errors = self._errors(["W"], self._PARAMS_SCALE)
        self.assertEqual(errors, [])

    def test_prefix_wildcard_accepted_when_params_match(self):
        errors = self._errors(["Encoder.*"], self._PARAMS_TWO_LAYERS)
        self.assertEqual(errors, [])

    def test_prefix_wildcard_matches_all_layers_of_prefix(self):
        errors = self._errors(["Encoder.*", "Classifier.*"], self._PARAMS_TWO_LAYERS)
        self.assertEqual(errors, [])

    def test_unknown_prefix_rejected(self):
        errors = self._errors(["encoder.*"], self._PARAMS_SCALE)
        self.assertIn("UPDATE parameter is not trainable in MODEL: encoder.*", errors)

    def test_unknown_exact_rejected(self):
        errors = self._errors(["Scale.nonexistent"], self._PARAMS_SCALE)
        self.assertIn("UPDATE parameter is not trainable in MODEL: Scale.nonexistent", errors)

    def test_mix_valid_and_invalid_reports_only_invalid(self):
        errors = self._errors(["Scale.W", "Scale.nonexistent"], self._PARAMS_SCALE)
        self.assertEqual(len(errors), 1)
        self.assertIn("Scale.nonexistent", errors[0])

    def test_original_error_message_preserved(self):
        # Regression: exact error message format unchanged (used by tests/P4 contract)
        errors = self._errors(["W2"], self._PARAMS_SCALE)
        self.assertEqual(errors, ["UPDATE parameter is not trainable in MODEL: W2"])


# ---------------------------------------------------------------------------
# GenericSupervisedTrainer
# ---------------------------------------------------------------------------

class TestGenericSupervisedTrainer(unittest.TestCase):

    _FIELDS = ["x1", "x2", "x3", "x4"]

    def _program(self):
        return parse_text(_PROG_CHAIN)

    def _example(self, vec: list[float], label: str) -> _LayerExample:
        return _LayerExample(
            vector_name="X",
            vector={f: v for f, v in zip(self._FIELDS, vec)},
            label=label,
        )

    def _train(self, *, patterns: list[str], epochs: int = 3, lr: float = 0.001,
               examples=None, val_examples=None):
        program = self._program()
        if examples is None:
            # input [0,0,1,0] with initial W gives relu[3] = 0.025 > 0 → differentiable
            examples = [self._example([0.0, 0.0, 1.0, 0.0], "3")]
        if val_examples is None:
            val_examples = examples
        trainer = GenericSupervisedTrainer()
        return trainer.train(
            program=program,
            training=None,  # unused in method body
            examples=examples,
            validation_examples=val_examples,
            prediction_key="F",
            target_key="label",
            update_patterns=patterns,
            epochs=epochs,
            learning_rate=lr,
        )

    def test_trainable_keys_with_wildcard_star(self):
        result = self._train(patterns=["*"])
        self.assertEqual(result["trainable_keys"], ["Chain.W"])

    def test_trainable_keys_with_layer_prefix(self):
        result = self._train(patterns=["Chain.*"])
        self.assertEqual(result["trainable_keys"], ["Chain.W"])

    def test_trainable_keys_with_exact_path(self):
        result = self._train(patterns=["Chain.W"])
        self.assertEqual(result["trainable_keys"], ["Chain.W"])

    def test_no_matching_pattern_raises(self):
        program = self._program()
        trainer = GenericSupervisedTrainer()
        with self.assertRaises(ValueError) as ctx:
            trainer.train(
                program=program,
                training=None,
                examples=[self._example([0.0, 0.0, 1.0, 0.0], "3")],
                validation_examples=[self._example([0.0, 0.0, 1.0, 0.0], "3")],
                prediction_key="F",
                target_key="label",
                update_patterns=["nonexistent.*"],
                epochs=1,
                learning_rate=0.001,
            )
        self.assertIn("No parameters matched", str(ctx.exception))

    def test_epoch_trace_has_correct_length(self):
        result = self._train(patterns=["*"], epochs=4)
        self.assertEqual(len(result["epoch_trace"]), 4)

    def test_epoch_trace_entry_keys(self):
        result = self._train(patterns=["*"], epochs=2)
        for entry in result["epoch_trace"]:
            self.assertIn("epoch", entry)
            self.assertIn("train_loss", entry)
            self.assertIn("validation_loss", entry)

    def test_epoch_numbers_are_sequential(self):
        result = self._train(patterns=["*"], epochs=3)
        epochs = [e["epoch"] for e in result["epoch_trace"]]
        self.assertEqual(epochs, [1, 2, 3])

    def test_returns_best_and_final_params(self):
        result = self._train(patterns=["*"], epochs=2)
        self.assertIn("best_params", result)
        self.assertIn("final_params", result)
        self.assertIn("Chain.W", result["best_params"])
        self.assertIn("Chain.W", result["final_params"])

    def test_best_params_is_matrix(self):
        result = self._train(patterns=["*"], epochs=2)
        w = result["best_params"]["Chain.W"]
        self.assertIsInstance(w, list)
        self.assertIsInstance(w[0], list)
        self.assertEqual(len(w), 4)
        self.assertEqual(len(w[0]), 4)

    def test_loss_is_non_negative(self):
        result = self._train(patterns=["*"], epochs=3)
        for entry in result["epoch_trace"]:
            self.assertGreaterEqual(entry["train_loss"], 0.0)
            self.assertGreaterEqual(entry["validation_loss"], 0.0)

    def test_loss_decreases_on_deterministic_task(self):
        # Input [0,0,1,0], target "3": initial W[2][3]=0.025 gives loss ≈ 3.69.
        # Gradient w.r.t. W[2][3] is -1/W[2][3] ≈ -40 → SGD increases W[2][3] → loss decreases.
        result = self._train(patterns=["*"], epochs=5, lr=0.001)
        losses = [e["train_loss"] for e in result["epoch_trace"]]
        self.assertLess(losses[-1], losses[0], "loss should decrease over 5 epochs on this task")

    def test_epoch_callback_called_each_epoch(self):
        calls: list[dict] = []
        self._train(patterns=["*"], epochs=3, lr=0.001,
                    examples=[self._example([0.0, 0.0, 1.0, 0.0], "3")],
                    val_examples=[self._example([0.0, 0.0, 1.0, 0.0], "3")])
        # Re-run with callback
        program = self._program()
        GenericSupervisedTrainer().train(
            program=program,
            training=None,
            examples=[self._example([0.0, 0.0, 1.0, 0.0], "3")],
            validation_examples=[self._example([0.0, 0.0, 1.0, 0.0], "3")],
            prediction_key="F",
            target_key="label",
            update_patterns=["*"],
            epochs=3,
            learning_rate=0.001,
            epoch_callback=calls.append,
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual([c["epoch"] for c in calls], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
