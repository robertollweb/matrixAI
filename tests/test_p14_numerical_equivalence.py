"""P14 Corte 5: Numerical equivalence between differentiable_python and torch cpu backend.

Tolerance: atol=1e-5, rtol=1e-4 for float32 arithmetic.
Both backends are fed identical parameter sets (via ParameterSet.runtime_parameters())
so any residual difference is purely floating-point rounding.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Any

_BASE = Path(__file__).parent.parent

_ATOL = 1e-5
_RTOL = 1e-4

_EMAIL_MXAI = _BASE / "examples" / "email-agent.typed.mxai"
_FALL_RISK_MXAI = _BASE / "examples" / "fall-risk.typed.mxai"

_EMAIL_INPUT = {
    "Email": {
        "urgency": 0.84,
        "sender_trust": 0.96,
        "topic_support": 0.99,
        "topic_sales": 0.04,
        "sentiment": 0.72,
        "has_attachment": 0.0,
        "previous_interactions": 0.88,
        "language_confidence": 0.97,
    }
}

_FALL_RISK_INPUT = {
    "Patient": {
        "age": 0.92,
        "mobility": 0.22,
        "medication_load": 0.76,
        "previous_falls": 1.0,
        "cognitive_state": 0.48,
    }
}

_EMAIL_INPUTS = [
    _EMAIL_INPUT,
    {"Email": {"urgency": 0.1, "sender_trust": 0.2, "topic_support": 0.3, "topic_sales": 0.8, "sentiment": 0.4, "has_attachment": 1.0, "previous_interactions": 0.1, "language_confidence": 0.6}},
    {"Email": {"urgency": 0.5, "sender_trust": 0.5, "topic_support": 0.5, "topic_sales": 0.5, "sentiment": 0.5, "has_attachment": 0.5, "previous_interactions": 0.5, "language_confidence": 0.5}},
]

_FALL_RISK_INPUTS = [
    _FALL_RISK_INPUT,
    {"Patient": {"age": 0.3, "mobility": 0.8, "medication_load": 0.2, "previous_falls": 0.0, "cognitive_state": 0.9}},
    {"Patient": {"age": 0.5, "mobility": 0.5, "medication_load": 0.5, "previous_falls": 0.5, "cognitive_state": 0.5}},
]


def _isclose(a: float, b: float, atol: float = _ATOL, rtol: float = _RTOL) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _compare_values(dp_val: Any, torch_val: Any, path: str) -> list[str]:
    """Recursively compare state values; return list of mismatch descriptions."""
    errors: list[str] = []
    if isinstance(dp_val, float) and isinstance(torch_val, (float, int)):
        if not _isclose(dp_val, float(torch_val)):
            errors.append(f"{path}: dp={dp_val!r} torch={torch_val!r} diff={abs(dp_val - float(torch_val)):.2e}")
    elif isinstance(dp_val, list) and isinstance(torch_val, list):
        for i, (d, t) in enumerate(zip(dp_val, torch_val)):
            errors.extend(_compare_values(d, t, f"{path}[{i}]"))
    elif isinstance(dp_val, dict) and isinstance(torch_val, dict):
        for k in dp_val:
            if k in torch_val:
                errors.extend(_compare_values(dp_val[k], torch_val[k], f"{path}.{k}"))
    return errors


def _run_both(mxai_path: Path, input_data: dict) -> tuple[dict, dict]:
    from matrixai.compiler import DifferentiablePythonCompiler
    from matrixai.compiler.torch_forward import TorchForwardRunner
    from matrixai.parameters import build_initial_parameter_set
    from matrixai.parser import parse_file

    program = parse_file(mxai_path)
    params = build_initial_parameter_set(program)

    ns: dict = {}
    exec(DifferentiablePythonCompiler().compile(program), ns)
    dp_result = ns["run"](input_data, params.runtime_parameters())

    torch_result = TorchForwardRunner(device="cpu").run(program, input_data, params)

    return dp_result, torch_result


# ---------------------------------------------------------------------------
# Email-agent model equivalence
# ---------------------------------------------------------------------------

class TestEmailAgentEquivalence(unittest.TestCase):
    def test_classifier_output_matches(self):
        dp, tor = _run_both(_EMAIL_MXAI, _EMAIL_INPUT)
        errors = _compare_values(dp["state"]["Classifier"], tor["state"]["Classifier"], "Classifier")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_reply_activation_matches(self):
        dp, tor = _run_both(_EMAIL_MXAI, _EMAIL_INPUT)
        errors = _compare_values(dp["state"]["ReplyActivation"], tor["state"]["ReplyActivation"], "ReplyActivation")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_action_value_matches(self):
        dp, tor = _run_both(_EMAIL_MXAI, _EMAIL_INPUT)
        dp_val = dp["actions"][0]["value"]
        tor_val = tor["actions"][0]["value"]
        self.assertTrue(
            _isclose(dp_val, tor_val),
            f"Action value mismatch: dp={dp_val!r} torch={tor_val!r} diff={abs(dp_val - tor_val):.2e}",
        )

    def test_action_activated_agrees(self):
        dp, tor = _run_both(_EMAIL_MXAI, _EMAIL_INPUT)
        self.assertEqual(dp["actions"][0]["activated"], tor["actions"][0]["activated"])

    def test_all_float_state_keys_match(self):
        dp, tor = _run_both(_EMAIL_MXAI, _EMAIL_INPUT)
        errors: list[str] = []
        for key, dp_val in dp["state"].items():
            if isinstance(dp_val, float) and key in tor["state"]:
                errors.extend(_compare_values(dp_val, tor["state"][key], key))
        self.assertEqual(errors, [], "\n".join(errors))

    def test_multiple_inputs_agree(self):
        for i, inp in enumerate(_EMAIL_INPUTS):
            dp, tor = _run_both(_EMAIL_MXAI, inp)
            errors = _compare_values(dp["state"]["Classifier"], tor["state"]["Classifier"], f"input[{i}].Classifier")
            self.assertEqual(errors, [], f"Input {i}: " + "\n".join(errors))


# ---------------------------------------------------------------------------
# Fall-risk model equivalence
# ---------------------------------------------------------------------------

class TestFallRiskEquivalence(unittest.TestCase):
    def test_risk_model_output_matches(self):
        dp, tor = _run_both(_FALL_RISK_MXAI, _FALL_RISK_INPUT)
        errors = _compare_values(dp["state"]["RiskModel"], tor["state"]["RiskModel"], "RiskModel")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_alert_activation_matches(self):
        dp, tor = _run_both(_FALL_RISK_MXAI, _FALL_RISK_INPUT)
        errors = _compare_values(dp["state"]["AlertActivation"], tor["state"]["AlertActivation"], "AlertActivation")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_action_value_matches(self):
        dp, tor = _run_both(_FALL_RISK_MXAI, _FALL_RISK_INPUT)
        dp_val = dp["actions"][0]["value"]
        tor_val = tor["actions"][0]["value"]
        self.assertTrue(
            _isclose(dp_val, tor_val),
            f"Action value mismatch: dp={dp_val!r} torch={tor_val!r} diff={abs(dp_val - tor_val):.2e}",
        )

    def test_all_float_state_keys_match(self):
        dp, tor = _run_both(_FALL_RISK_MXAI, _FALL_RISK_INPUT)
        errors: list[str] = []
        for key, dp_val in dp["state"].items():
            if isinstance(dp_val, float) and key in tor["state"]:
                errors.extend(_compare_values(dp_val, tor["state"][key], key))
        self.assertEqual(errors, [], "\n".join(errors))

    def test_multiple_inputs_agree(self):
        for i, inp in enumerate(_FALL_RISK_INPUTS):
            dp, tor = _run_both(_FALL_RISK_MXAI, inp)
            errors = _compare_values(dp["state"]["RiskModel"], tor["state"]["RiskModel"], f"input[{i}].RiskModel")
            self.assertEqual(errors, [], f"Input {i}: " + "\n".join(errors))


# ---------------------------------------------------------------------------
# Tolerance boundary: documented atol/rtol values
# ---------------------------------------------------------------------------

class TestToleranceContract(unittest.TestCase):
    def test_documented_atol_value(self):
        self.assertEqual(_ATOL, 1e-5)

    def test_documented_rtol_value(self):
        self.assertEqual(_RTOL, 1e-4)

    def test_max_observed_diff_within_atol(self):
        """Spot-check: worst observed difference stays well below atol."""
        dp, tor = _run_both(_EMAIL_MXAI, _EMAIL_INPUT)
        probs_dp = dp["state"]["Classifier"]
        probs_tor = tor["state"]["Classifier"]
        max_diff = max(
            abs(probs_dp[label] - probs_tor[label]) for label in probs_dp if label in probs_tor
        )
        self.assertLess(max_diff, _ATOL, f"Max diff {max_diff:.2e} exceeds atol {_ATOL:.2e}")

    def test_max_observed_diff_fall_risk_within_atol(self):
        dp, tor = _run_both(_FALL_RISK_MXAI, _FALL_RISK_INPUT)
        diff = abs(dp["state"]["RiskModel"] - tor["state"]["RiskModel"])
        self.assertLess(diff, _ATOL, f"RiskModel diff {diff:.2e} exceeds atol {_ATOL:.2e}")


# ---------------------------------------------------------------------------
# CUDA/MPS equivalence (skipped when hardware absent)
# ---------------------------------------------------------------------------

class TestCudaEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if "cuda" not in torch_device_info()["available_devices"]:
            raise unittest.SkipTest("CUDA not available")

    def test_cuda_matches_cpu_within_tolerance(self):
        from matrixai.compiler import DifferentiablePythonCompiler
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file
        program = parse_file(_EMAIL_MXAI)
        params = build_initial_parameter_set(program)
        ns: dict = {}
        exec(DifferentiablePythonCompiler().compile(program), ns)
        dp_result = ns["run"](_EMAIL_INPUT, params.runtime_parameters())
        cuda_result = TorchForwardRunner(device="cuda").run(program, _EMAIL_INPUT, params)
        errors = _compare_values(
            dp_result["state"]["Classifier"],
            cuda_result["state"]["Classifier"],
            "Classifier",
        )
        self.assertEqual(errors, [], "\n".join(errors))


class TestMpsEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if "mps" not in torch_device_info()["available_devices"]:
            raise unittest.SkipTest("MPS not available")

    def test_mps_matches_cpu_within_tolerance(self):
        from matrixai.compiler import DifferentiablePythonCompiler
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.parser import parse_file
        program = parse_file(_EMAIL_MXAI)
        params = build_initial_parameter_set(program)
        ns: dict = {}
        exec(DifferentiablePythonCompiler().compile(program), ns)
        dp_result = ns["run"](_EMAIL_INPUT, params.runtime_parameters())
        mps_result = TorchForwardRunner(device="mps").run(program, _EMAIL_INPUT, params)
        errors = _compare_values(
            dp_result["state"]["Classifier"],
            mps_result["state"]["Classifier"],
            "Classifier",
        )
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
