"""P11 Cut 8 — regression guard: P1-P10 contracts unaffected by P11 changes.

Scope: each test targets a specific P11 change site and verifies the prior-phase
contract at that site is preserved.  This is NOT a re-run of existing test files;
it is a targeted smoke-test of the shared code paths P11 extended.

P11 change sites tested:
  - training/verifier.py  (_cross_entropy_kinds / _bce_kinds extension)
  - training/differentiability.py  (glob expansion + _layer_to_node_map)
  - training/trainer.py  (_generic_cross_entropy_loss labels=None, new classes)
  - compiler/differentiable_python.py  (LAYER body execution, passthrough)
  - playground.py  (prediction_kind routing)
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.typed.mxai"
_EMAIL_TRAIN = _BASE / "examples" / "email-agent.supervised.mxtrain"
_FALL_MXAI = _BASE / "examples" / "fall-risk.typed.mxai"
_FALL_TRAIN = _BASE / "examples" / "fall-risk.supervised.mxtrain"
_FALL_CSV = _BASE / "examples" / "fall-risk.train.csv"


# ---------------------------------------------------------------------------
# P4 — TrainingVerifier contract (verifier.py change site)
# ---------------------------------------------------------------------------

class TestP11RegressionVerifier(unittest.TestCase):
    """P11 extended _cross_entropy_kinds; P4 softmax_linear must still pass."""

    def _verify(self, mxai_path: Path, train_path: Path):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.verifier import TrainingVerifier
        training = parse_training_file(str(train_path))
        return TrainingVerifier().verify(training, base_path=_BASE)

    def test_email_agent_training_verifier_still_ok(self):
        report = self._verify(_EMAIL_MXAI, _EMAIL_TRAIN)
        self.assertTrue(report.ok, report.errors)

    def test_fall_risk_training_verifier_still_ok(self):
        report = self._verify(_FALL_MXAI, _FALL_TRAIN)
        self.assertTrue(report.ok, report.errors)

    def test_cross_entropy_still_accepts_softmax_linear(self):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.verifier import TrainingVerifier
        training = parse_training_file(str(_EMAIL_TRAIN))
        report = TrainingVerifier().verify(training, base_path=_BASE)
        # Verifier must not complain about softmax_linear being invalid
        self.assertFalse(
            any("softmax_linear" in e for e in report.errors),
            report.errors,
        )

    def test_binary_cross_entropy_still_accepts_sigmoid_linear(self):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.verifier import TrainingVerifier
        training = parse_training_file(str(_FALL_TRAIN))
        report = TrainingVerifier().verify(training, base_path=_BASE)
        self.assertFalse(
            any("sigmoid_linear" in e for e in report.errors),
            report.errors,
        )


# ---------------------------------------------------------------------------
# P4 — DifferentiabilityVerifier contract (differentiability.py change site)
# ---------------------------------------------------------------------------

class TestP11RegressionDifferentiability(unittest.TestCase):
    """P11 added glob expansion and _layer_to_node_map; P4 exact-path params must still verify."""

    def _diff_verify(self, mxai_path: Path, train_path: Path):
        from matrixai.parser import parse_file
        from matrixai.training.differentiability import DifferentiabilityVerifier
        from matrixai.training.parser import parse_training_file
        program = parse_file(str(mxai_path))
        training = parse_training_file(str(train_path))
        return DifferentiabilityVerifier().verify(training, program)

    def test_email_agent_differentiability_ok(self):
        result = self._diff_verify(_EMAIL_MXAI, _EMAIL_TRAIN)
        self.assertTrue(result.ok, result.errors)

    def test_fall_risk_differentiability_ok(self):
        result = self._diff_verify(_FALL_MXAI, _FALL_TRAIN)
        self.assertTrue(result.ok, result.errors)

    def test_email_agent_parameter_paths_still_verified(self):
        result = self._diff_verify(_EMAIL_MXAI, _EMAIL_TRAIN)
        self.assertGreater(len(result.parameter_paths), 0)

    def test_no_spurious_warning_for_p4_exact_paths(self):
        result = self._diff_verify(_EMAIL_MXAI, _EMAIL_TRAIN)
        # P11 changed warning text only for glob patterns; P4 exact paths must not warn
        no_paths_warnings = [w for w in result.warnings if "no differentiability paths" in w.lower()]
        self.assertEqual(no_paths_warnings, [])


# ---------------------------------------------------------------------------
# P4 — SupervisedTrainer + SupervisedEvaluator (trainer.py change site)
# ---------------------------------------------------------------------------

class TestP11RegressionSupervisedTrainer(unittest.TestCase):
    """P11 added GenericSupervisedTrainer; SupervisedTrainer must be unchanged."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from matrixai.training.parser import parse_training_file
        from matrixai.training.trainer import SupervisedTrainer, SupervisedEvaluator
        from matrixai.parameters.store import ParameterSet

        training = parse_training_file(str(_FALL_TRAIN))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "run"
            result = SupervisedTrainer().train(training, output_dir=out, base_path=_BASE)
            rd = result.to_dict()
            best_path = rd.get("artifacts", {}).get("params_best")
            cls.train_ok = rd.get("ok", True)
            cls.best_epoch = rd.get("best_epoch", 0)
            cls.evaluation = None
            if best_path and Path(best_path).exists():
                from matrixai.parameters import load_parameter_set
                ps = load_parameter_set(Path(best_path))
                ev = SupervisedEvaluator().evaluate(training, ps, base_path=_BASE)
                cls.evaluation = ev.to_dict()

    def test_supervised_trainer_runs_without_error(self):
        self.assertTrue(self.train_ok)

    def test_supervised_trainer_has_best_epoch(self):
        self.assertGreater(self.best_epoch, 0)

    def test_supervised_evaluator_produces_accuracy(self):
        self.assertIsNotNone(self.evaluation)
        acc = self.evaluation.get("accuracy", -1.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_generic_cross_entropy_loss_without_labels_unchanged(self):
        """_generic_cross_entropy_loss(labels=None) must behave as before P11 for dict predictions."""
        from matrixai.training.trainer import _generic_cross_entropy_loss
        # Dict prediction path: P4 models produce {label: prob} dicts
        state = {"pred": {"class_a": 0.7, "class_b": 0.3}}
        loss = _generic_cross_entropy_loss(state, "class_a", "pred", labels=None)
        self.assertTrue(math.isfinite(loss), loss)
        self.assertAlmostEqual(loss, -math.log(0.7), places=6)


# ---------------------------------------------------------------------------
# P5 — torch backend (compiler change site)
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch not installed")
class TestP11RegressionTorchBackend(unittest.TestCase):
    """P11 added layer_call support to torch; P4/P5 softmax_linear forward must be unchanged."""

    def test_torch_forward_runner_ok_for_email_agent(self):
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file
        program = parse_file(str(_EMAIL_MXAI))
        ps = build_initial_parameter_set(program)
        runner = TorchForwardRunner()
        result = runner.run(
            program,
            {"Email": {"urgency": 0.8, "sender_trust": 0.9, "topic_support": 0.7,
                       "topic_sales": 0.1, "sentiment": 0.6, "has_attachment": 0.0,
                       "previous_interactions": 0.8, "language_confidence": 0.95}},
            ps,
        )
        # email-agent final output is 'Confidence' (Categorical), not urgency score
        self.assertIn("Confidence", result["state"])

    def test_torch_backend_contract_ok_for_email_agent(self):
        from matrixai.compiler import BackendContractAnalyzer
        from matrixai.parser import parse_file
        program = parse_file(str(_EMAIL_MXAI))
        report = BackendContractAnalyzer(target="torch").analyze(program)
        self.assertTrue(report.ok, report.parameter_errors)


# ---------------------------------------------------------------------------
# P9/P10 — playground routing and LAYER passthrough (playground + compiler)
# ---------------------------------------------------------------------------

class TestP11RegressionPlaygroundRouting(unittest.TestCase):
    """P11 added layer_call routing; P4/P5 models must still take the SupervisedTrainer path."""

    def test_get_prediction_kind_email_agent_not_layer_call(self):
        from matrixai.playground import _get_prediction_kind
        mxai = _EMAIL_MXAI.read_text(encoding="utf-8")
        train = _EMAIL_TRAIN.read_text(encoding="utf-8")
        kind = _get_prediction_kind(mxai, train)
        self.assertNotEqual(kind, "layer_call")

    def test_get_prediction_kind_fall_risk_not_layer_call(self):
        from matrixai.playground import _get_prediction_kind
        mxai = _FALL_MXAI.read_text(encoding="utf-8")
        train = _FALL_TRAIN.read_text(encoding="utf-8")
        kind = _get_prediction_kind(mxai, train)
        self.assertNotEqual(kind, "layer_call")

    def test_fall_risk_playground_training_uses_supervised_path(self):
        """Fall-risk goes through SupervisedTrainer (not GenericSupervisedTrainer)."""
        from matrixai.playground import _run_playground_training, _generate_training_from_mxai
        mxai = _FALL_MXAI.read_text(encoding="utf-8")
        csv = _FALL_CSV.read_text(encoding="utf-8")
        gen = _generate_training_from_mxai(mxai)
        self.assertTrue(gen.get("ok"), gen.get("error"))
        r = _run_playground_training(mxai, gen["training_text"], csv, epochs_override=2)
        self.assertTrue(r.get("ok"), r.get("error"))
        self.assertEqual(len(r["epochs"]), 2)


class TestP11RegressionP10LayerPassthrough(unittest.TestCase):
    """P11 added _layer_exec_<Name>; P10 LAYERs without body ops must still use passthrough."""

    def test_p10_layer_without_body_ops_compiles_with_passthrough(self):
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        from matrixai.parser import parse_text
        # Minimal P10-style model with a LAYER that has no body ops
        src = """
PROJECT regression_p10_passthrough

VECTOR X[2]
  x0: Float
  x1: Float
END

LAYER Scale(Tensor[2]) -> Tensor[2]
  PARAM W Tensor[2]
END

FUNCTION Scaled
  result = call_layer(Scale, X)
END

GRAPH
  X -> Scaled
END
"""
        program = parse_text(src)
        source = DifferentiablePythonCompiler().compile(program)
        # A LAYER without body ops must NOT generate _layer_exec_Scale
        self.assertNotIn("_layer_exec_Scale", source)
        # The source must be executable and produce a functioning run()
        ns: dict = {}
        exec(compile(source, "<regression_p10>", "exec"), ns)  # noqa: S102
        self.assertIn("run", ns)
        # Execute run() with a concrete input and verify passthrough identity:
        # call_layer(Scale, X) with no body ops returns the input vector unchanged.
        # _resolve_vector converts the input dict to a list of float values.
        x_input = {"x0": 1.0, "x1": 2.0}
        result = ns["run"]({"X": x_input}, {})
        state = result["state"]
        self.assertIn("Scaled", state, "Output node 'Scaled' missing from state after run()")
        # passthrough: resolved vector is [x0, x1] in field order
        passthrough_out = state.get("result", state["Scaled"])
        self.assertEqual(list(passthrough_out), [1.0, 2.0])

    def test_p10_layer_with_body_ops_still_generates_executor(self):
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        from matrixai.parser import parse_text
        # Use the transformer model which has LAYER body ops
        transformer_src = (_BASE / "examples" / "transformer-classifier-vector.mxai").read_text(encoding="utf-8")
        program = parse_text(transformer_src)
        source = DifferentiablePythonCompiler().compile(program)
        self.assertIn("_layer_exec_encoder_attn", source)
        self.assertIn("_layer_exec_encoder_ffn", source)
        self.assertIn("_layer_exec_classifier", source)


if __name__ == "__main__":
    unittest.main()
