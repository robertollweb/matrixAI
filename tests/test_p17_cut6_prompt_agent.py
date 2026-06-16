from __future__ import annotations

import unittest
import unittest.mock

from matrixai.agents.prompt import PromptAgent


def _synth(prompt: str):
    return PromptAgent().synthesize(prompt)


_VALID_CLASSIFICATION_SEMANTIC = """\
PROJECT PriceClassifier
INTENT classify house prices
MODE classification
ENTITY House

FIELDS House
  meters
  rooms
  area
END

GOAL classify_incoming_email
CONSTRAINT confidence > 0.80
ACTION_THRESHOLD 0.90

ACTION SendAlert
  POLICY simulate_only
  CALL simulated.alerts.send
END"""


def _fake_transport(semantic_text: str):
    def transport(url, headers, payload_bytes, timeout):
        return {
            "choices": [{"message": {"content": semantic_text}}],
            "usage": {"total_tokens": 100},
        }

    return transport


class TestPromptAgentRegressionTemplateSelection(unittest.TestCase):

    def test_precio_selects_regression(self):
        r = _synth("predecir precio de vivienda")
        self.assertEqual(r.inferred_template, "generic_regression")

    def test_price_selects_regression(self):
        r = _synth("predict price of a house")
        self.assertEqual(r.inferred_template, "generic_regression")

    def test_predecir_selects_regression(self):
        r = _synth("predecir el tiempo esperado de entrega")
        self.assertEqual(r.inferred_template, "generic_regression")

    def test_celsius_selects_regression(self):
        r = _synth("convert celsius to kelvin as trained model")
        self.assertEqual(r.inferred_template, "generic_regression")

    def test_kelvin_selects_regression(self):
        r = _synth("predecir kelvin desde grados celsius")
        self.assertEqual(r.inferred_template, "generic_regression")

    def test_estim_selects_regression(self):
        r = _synth("estimacion de consumo energetico")
        self.assertEqual(r.inferred_template, "generic_regression")

    def test_regres_selects_regression(self):
        r = _synth("modelo de regresion lineal para ventas")
        self.assertEqual(r.inferred_template, "generic_regression")

    def test_classify_still_selects_classification(self):
        r = _synth("clasificar documentos por categoria")
        self.assertNotEqual(r.inferred_template, "generic_regression")
        self.assertEqual(r.inferred_template, "generic_classification")

    def test_fall_risk_still_selects_fall_risk(self):
        r = _synth("patient fall risk prediction")
        self.assertEqual(r.inferred_template, "fall_risk")

    def test_email_not_regression(self):
        r = _synth("classify incoming email replies")
        self.assertNotEqual(r.inferred_template, "generic_regression")


class TestPromptAgentRegressionOutput(unittest.TestCase):

    def test_mode_is_regression(self):
        r = _synth("predecir precio de casa")
        self.assertEqual(r.inferred_mode, "regression")

    def test_semantic_text_has_mode_regression(self):
        r = _synth("predecir precio de casa")
        self.assertIn("MODE regression", r.semantic_text)

    def test_semantic_text_has_mse_loss(self):
        r = _synth("predecir precio de casa")
        self.assertIn("LOSS mse", r.semantic_text)

    def test_semantic_text_has_mae_metric(self):
        r = _synth("predecir precio de casa")
        self.assertIn("METRIC mae", r.semantic_text)

    def test_semantic_text_has_output_scalar(self):
        r = _synth("predecir precio de casa")
        self.assertIn("OUTPUT", r.semantic_text)
        self.assertIn("Scalar", r.semantic_text)

    def test_semantic_text_no_action_simulate_only(self):
        r = _synth("predecir precio de casa")
        self.assertNotIn("simulate_only", r.semantic_text)

    def test_extracted_rules_empty_for_regression(self):
        r = _synth("predecir precio de casa si area > 100")
        self.assertEqual(r.extracted_rules, [])

    def test_assumptions_mention_trainable_package(self):
        r = _synth("predecir precio de casa")
        combined = " ".join(r.assumptions)
        self.assertIn("trainable", combined.lower())

    def test_assumptions_mention_mse(self):
        r = _synth("predecir precio de casa")
        combined = " ".join(r.assumptions)
        self.assertIn("mse", combined.lower())


class TestPromptAgentExactFormulaDistinction(unittest.TestCase):

    def test_celsius_kelvin_notes_deterministic_formula(self):
        r = _synth("convert celsius to kelvin")
        combined = " ".join(r.assumptions)
        self.assertTrue(
            any("formula" in a.lower() or "deterministic" in a.lower() for a in r.assumptions),
            msg=f"Expected formula/deterministic note in assumptions: {r.assumptions}",
        )

    def test_price_prediction_no_formula_note(self):
        r = _synth("predecir precio de casa con regresion lineal")
        combined = " ".join(r.assumptions).lower()
        self.assertNotIn("deterministic formula", combined)

    def test_convertir_notes_formula(self):
        r = _synth("convertir temperatura en grados a kelvin")
        combined = " ".join(r.assumptions).lower()
        self.assertIn("trainable", combined)

    def test_centigrados_kelvin_uses_auditable_domain_names(self):
        r = _synth("Convertir Grados Centigrados a Grados Kelvin")
        self.assertIn("PROJECT CelsiusToKelvin", r.semantic_text)
        self.assertIn("ENTITY Reading", r.semantic_text)
        self.assertIn("  celsius", r.semantic_text)
        self.assertIn("OUTPUT predicted_kelvin: Scalar", r.semantic_text)


class TestPromptRegressionEndToEnd(unittest.TestCase):

    def test_prompt_supervisor_accepts_deterministic_regression_semantic(self):
        from matrixai.agents.prompt_supervisor import PromptSupervisor
        synthesis = _synth("predecir precio casa con campos metros, habitaciones y zona")
        report = PromptSupervisor().supervise_semantic(
            prompt=synthesis.prompt,
            semantic_text=synthesis.semantic_text,
            source="PromptAgent",
            synthesis=synthesis,
        )
        self.assertTrue(report.accepted, [(c.name, c.ok, c.errors) for c in report.checks])

    def test_prompt_regression_generates_linear_mxai(self):
        from matrixai.agents.prompt_supervisor import PromptSupervisor
        synthesis = _synth("predecir precio casa con campos metros, habitaciones y zona")
        report = PromptSupervisor().supervise_semantic(
            prompt=synthesis.prompt,
            semantic_text=synthesis.semantic_text,
            source="PromptAgent",
            synthesis=synthesis,
        )
        self.assertIn("linear(W1 * Item + b1)", report.mxai)
        self.assertIn("predicted_value: Scalar", report.mxai)

    def test_prompt_regression_mxai_is_trainable_shape(self):
        from matrixai.agents.prompt_supervisor import PromptSupervisor
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        from matrixai.parser import parse_text
        synthesis = _synth("predecir precio casa con campos metros, habitaciones y zona")
        report = PromptSupervisor().supervise_semantic(
            prompt=synthesis.prompt,
            semantic_text=synthesis.semantic_text,
            source="PromptAgent",
            synthesis=synthesis,
        )
        backend_report = BackendContractAnalyzer().analyze(parse_text(report.mxai))
        self.assertTrue(backend_report.ok, backend_report.parameter_errors)
        weights = next(p for p in backend_report.trainable_parameters if p.name == "W1")
        self.assertEqual(weights.shape, (3,))

    def test_supervise_prompt_falls_back_when_llm_classifies_continuous_prompt(self):
        from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider
        from matrixai.agents.prompt_supervisor import PromptSupervisor

        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="chat-test",
            transport=_fake_transport(_VALID_CLASSIFICATION_SEMANTIC),
            max_retries=1,
            retry_delay_s=0.0,
        )

        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            return_value=provider,
        ):
            report = PromptSupervisor().supervise_prompt(
                "predecir precio casa con campos metros, habitaciones y zona"
            )

        self.assertTrue(report.accepted, [(c.name, c.ok, c.errors) for c in report.checks])
        self.assertEqual(report.supervision_source, "deterministic")
        self.assertEqual(report.fallback_reason, "llm_non_regression_for_continuous_prompt")
        self.assertIn("linear(W1 * Item + b1)", report.mxai)


if __name__ == "__main__":
    unittest.main()
