from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matrixai.agents import (
    ArchitectAgent,
    LLMBudgetExceededError,
    LLMCallTrace,
    LLMHTTPError,
    LLMProposal,
    LLMProposalAgent,
    ChatCompletionsLLMProposalProvider,
    PlannerVerifier,
    PromptAgent,
    PromptSupervisor,
    SemanticPlan,
    VerifierAgent,
)
from matrixai.agents.mathematical import MathematicalAgent
from matrixai.agents.optimizer import OptimizerAgent
from matrixai.compiler import PythonBackendCompiler
from matrixai.parser import parse_file, parse_text
from matrixai.runtime import MatrixAIRuntime


ROOT = Path(__file__).resolve().parents[1]


class _FirstUnsafeThenSafeProvider:
    provider_name = "test-llm"
    model_name = "first-unsafe-then-safe"

    def propose(self, prompt: str) -> tuple[list[LLMProposal], list[LLMCallTrace]]:
        unsafe = """PROJECT UnsafeEmail
INTENT Draft replies directly.
MODE classification
ENTITY Email

FIELDS Email
  urgency
  sender_trust
END

GOAL minimize_false_replies
CONSTRAINT confidence > 0.95
ACTION_THRESHOLD 0.90

ACTION DraftReply
  POLICY execute
  CALL email.draft
END
"""
        safe = PromptAgent().synthesize(prompt).semantic_text
        return [
            LLMProposal(
                candidate_id="candidate-1",
                provider=self.provider_name,
                model=self.model_name,
                prompt=prompt,
                semantic_text=unsafe,
                raw_output=unsafe,
            ),
            LLMProposal(
                candidate_id="candidate-2",
                provider=self.provider_name,
                model=self.model_name,
                prompt=prompt,
                semantic_text=safe,
                raw_output=safe,
            ),
        ], []


class _AlwaysUnsafeProvider:
    provider_name = "test-llm"
    model_name = "always-unsafe"

    def propose(self, prompt: str) -> tuple[list[LLMProposal], list[LLMCallTrace]]:
        semantic_text = """PROJECT UnsafeEmail
INTENT Draft replies directly.
MODE classification
ENTITY Email

FIELDS Email
  urgency
  sender_trust
END

GOAL minimize_false_replies
CONSTRAINT confidence > 0.95
ACTION_THRESHOLD 0.90

ACTION DraftReply
  POLICY execute
  CALL email.draft
END
"""
        return [
            LLMProposal(
                candidate_id="candidate-1",
                provider=self.provider_name,
                model=self.model_name,
                prompt=prompt,
                semantic_text=semantic_text,
                raw_output=semantic_text,
            )
        ], []


class _RetryingTransport:
    """Transport that fails the first N attempts with the given status, then succeeds."""

    def __init__(self, fail_status: int, fail_times: int, semantic_text: str) -> None:
        self.fail_status = fail_status
        self.fail_times = fail_times
        self.semantic_text = semantic_text
        self.call_count = 0

    def __call__(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> dict:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise LLMHTTPError(self.fail_status, f"Simulated {self.fail_status}")
        return {
            "choices": [{"message": {"content": self.semantic_text}}],
            "usage": {"total_tokens": 100},
        }


class _AlwaysErrorTransport:
    """Transport that always fails with the given status."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.call_count = 0

    def __call__(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> dict:
        self.call_count += 1
        raise LLMHTTPError(self.status, f"Simulated {self.status}")


class _TokenUsageTransport:
    """Transport that returns a fixed token_usage in the response."""

    def __init__(self, semantic_text: str, total_tokens: int) -> None:
        self.semantic_text = semantic_text
        self.total_tokens = total_tokens

    def __call__(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> dict:
        return {
            "choices": [{"message": {"content": self.semantic_text}}],
            "usage": {"total_tokens": self.total_tokens},
        }


class _RecordingChatCompletionsTransport:
    def __init__(self, semantic_text: str) -> None:
        self.semantic_text = semantic_text
        self.calls: list[tuple[str, dict[str, str], dict, float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> dict:
        self.calls.append((url, headers, json.loads(payload.decode("utf-8")), timeout))
        return {
            "choices": [
                {
                    "message": {
                        "content": f"```semantic\n{self.semantic_text}\n```",
                    }
                }
            ]
        }


class MatrixAIMVPTest(unittest.TestCase):
    def _load_generated_module(self, path: Path):
        spec = importlib.util.spec_from_file_location("compiled_matrixai_program", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_planner_verifier_accepts_valid_plan(self) -> None:
        semantic_text = (ROOT / "examples" / "email-agent.semantic").read_text()
        plan = ArchitectAgent().plan_from_text(semantic_text)

        result = PlannerVerifier().verify(plan)

        self.assertTrue(result.ok, result.errors)

    def test_planner_verifier_rejects_unsafe_action(self) -> None:
        semantic_text = (ROOT / "examples" / "email-agent.semantic").read_text()
        plan_data = ArchitectAgent().plan_from_text(semantic_text).to_dict()
        plan_data["actions"][0]["policy"] = "execute"
        plan_data["actions"][0]["call"] = "email.draft"
        bad_plan = SemanticPlan(**plan_data)

        result = PlannerVerifier().verify(bad_plan)

        self.assertFalse(result.ok)
        self.assertTrue(any("simulate_only" in error for error in result.errors))
        self.assertTrue(any("simulated.*" in error for error in result.errors))

    def test_architect_generates_valid_email_agent(self) -> None:
        semantic_text = (ROOT / "examples" / "email-agent.semantic").read_text()

        architect = ArchitectAgent()
        plan = architect.plan_from_text(semantic_text)
        mxai_text = architect.to_mxai(plan)
        program = parse_text(mxai_text)
        result = VerifierAgent().verify(program)

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(plan.mode, "classification")
        self.assertEqual(plan.graph["nodes"][0], "Email")
        self.assertIn("PROJECT EmailAgent", mxai_text)
        self.assertIn("VECTOR Email[8]", mxai_text)
        self.assertEqual(program.functions[1].semantic.parameters["threshold"], 0.95)
        self.assertEqual(program.actions[0].condition.threshold, 0.9)
        self.assertEqual(plan.mathematical_translations[0]["expression_kind"], "sigmoid_threshold")
        self.assertIn("sigmoid(20 * (Confidence.max - 0.95))", mxai_text)

    def test_prompt_agent_generates_supervised_email_agent(self) -> None:
        prompt = (
            "Crear un sistema que clasifique correos entrantes y prepare una respuesta "
            "solo si la confianza supera el 95%, reduciendo falsas respuestas."
        )

        synthesis = PromptAgent().synthesize(prompt)
        plan = ArchitectAgent().plan_from_text(synthesis.semantic_text)
        mxai_text = ArchitectAgent().to_mxai(plan)
        program = parse_text(mxai_text)

        self.assertIn("PROJECT EmailAgent", synthesis.semantic_text)
        self.assertIn("RULES", synthesis.semantic_text)
        self.assertIn("PromptAgent", synthesis.agent_chain)
        self.assertIn("MathematicalAgent", synthesis.agent_chain)
        self.assertTrue(PlannerVerifier().verify(plan).ok)
        self.assertTrue(VerifierAgent().verify(program).ok)
        self.assertEqual(plan.mode, "classification")
        self.assertEqual(plan.mathematical_translations[0]["expression_kind"], "sigmoid_threshold")
        self.assertIn("DraftReply", mxai_text)

    def test_prompt_agent_extracts_fields_rules_and_lineage(self) -> None:
        prompt = (
            "Proyecto ClaimRisk. Entidad Claim. "
            "Campos: severity, amount, customer_history, fraud_score. "
            "Si severity > 0.7 y fraud_score > 0.8 entonces Notify. "
            "Prioriza seguridad."
        )

        synthesis = PromptAgent().synthesize(prompt)
        plan = ArchitectAgent().plan_from_text(synthesis.semantic_text)

        self.assertEqual(synthesis.inferred_template, "generic_risk")
        self.assertEqual(synthesis.inferred_entity, "Claim")
        self.assertEqual(
            synthesis.selected_fields,
            ["severity", "amount", "customer_history", "fraud_score"],
        )
        self.assertIn(
            "if severity > 0.7 and fraud_score > 0.8 then Notify",
            synthesis.extracted_rules,
        )
        self.assertTrue(any(step["event"] == "structure_extracted" for step in synthesis.trace))
        self.assertTrue(any(step["event"] == "rules_extracted" for step in synthesis.trace))
        self.assertTrue(PlannerVerifier().verify(plan).ok)
        self.assertIn("sigmoid_and", {t["expression_kind"] for t in plan.mathematical_translations})
        self.assertTrue(any(item["used_in_graph"] for item in plan.lineage))
        self.assertTrue(
            any(
                item["expression_kind"] == "sigmoid_and" and item["used_in_graph"]
                for item in plan.lineage
            )
        )
        self.assertIn("AlertActivationRule2", plan.graph["nodes"])
        self.assertTrue(
            any(function["expression"] == "max(AlertActivationBase, AlertActivationRule2)"
                for function in plan.functions)
        )

    def test_prompt_rules_are_executable_action_evidence(self) -> None:
        prompt = (
            "Proyecto ClaimRisk. Entidad Claim. "
            "Campos: severity, amount, customer_history, fraud_score. "
            "Si severity > 0.7 y fraud_score > 0.8 entonces Notify. "
            "Prioriza seguridad."
        )
        synthesis = PromptAgent().synthesize(prompt)
        plan = ArchitectAgent().plan_from_text(synthesis.semantic_text)
        mxai_text = ArchitectAgent().to_mxai(plan)
        program = parse_text(mxai_text)
        input_data = {
            "Claim": {
                "severity": 0.95,
                "amount": 0.1,
                "customer_history": 0.1,
                "fraud_score": 0.95,
            }
        }

        runtime = MatrixAIRuntime().run(program, input_data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_claim_rules.py"
            output_path.write_text(PythonBackendCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)
            generated = module.run(input_data)

        self.assertIn("AlertActivationBase", runtime["state"])
        self.assertIn("AlertActivationRule2", runtime["state"])
        self.assertAlmostEqual(
            runtime["state"]["AlertActivation"],
            max(runtime["state"]["AlertActivationBase"], runtime["state"]["AlertActivationRule2"]),
            places=6,
        )
        self.assertEqual(generated["state"], runtime["state"])
        self.assertEqual(generated["actions"], runtime["actions"])

    def test_prompt_supervisor_accepts_prompt_pipeline(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text()

        with patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            report = PromptSupervisor().supervise_prompt(prompt)

        self.assertTrue(report.accepted, report.summary())
        self.assertEqual(report.source, "PromptAgent")
        self.assertIsNotNone(report.plan)
        self.assertIn("PROJECT ClaimRisk", report.mxai)
        self.assertIn("def run(input_data", report.compiled_python)
        self.assertTrue(all(check.ok for check in report.checks))
        check_names = {check.name for check in report.checks}
        self.assertIn("planner_verifier", check_names)
        self.assertIn("safety_agent", check_names)
        self.assertIn("python_compiler", check_names)

    def test_prompt_supervisor_rejects_unsafe_semantic_proposal(self) -> None:
        semantic_text = """PROJECT UnsafeEmail
INTENT Draft replies directly.
MODE classification
ENTITY Email

FIELDS Email
  urgency
  sender_trust
END

GOAL minimize_false_replies
CONSTRAINT confidence > 0.95
ACTION_THRESHOLD 0.90

ACTION DraftReply
  POLICY execute
  CALL email.draft
END
"""

        report = PromptSupervisor().supervise_semantic(
            prompt="Draft email replies directly",
            semantic_text=semantic_text,
            source="llm_proposal",
        )

        self.assertFalse(report.accepted)
        planner_check = next(check for check in report.checks if check.name == "planner_verifier")
        self.assertFalse(planner_check.ok)
        self.assertTrue(any("simulate_only" in error for error in planner_check.errors))
        self.assertTrue(any("simulated.*" in error for error in planner_check.errors))

    def test_llm_proposal_agent_accepts_deterministic_provider(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text()

        decision = LLMProposalAgent().propose_and_supervise(prompt)

        self.assertTrue(decision.accepted, decision.summary())
        self.assertEqual(decision.selected_proposal_id, "candidate-1")
        self.assertEqual(len(decision.reports), 1)
        self.assertTrue(decision.reports[0].accepted)
        self.assertEqual(decision.reports[0].source, "llm:deterministic-mvp:candidate-1")
        self.assertIn("PROJECT ClaimRisk", decision.reports[0].mxai)

    def test_llm_proposal_agent_tries_next_candidate_after_rejection(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text()

        decision = LLMProposalAgent(
            provider=_FirstUnsafeThenSafeProvider()
        ).propose_and_supervise(prompt)

        self.assertTrue(decision.accepted, decision.summary())
        self.assertEqual(decision.selected_proposal_id, "candidate-2")
        self.assertEqual(len(decision.reports), 2)
        self.assertFalse(decision.reports[0].accepted)
        self.assertTrue(decision.reports[1].accepted)

    def test_llm_proposal_agent_rejects_when_no_candidate_passes(self) -> None:
        decision = LLMProposalAgent(provider=_AlwaysUnsafeProvider()).propose_and_supervise(
            "Draft email replies directly"
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.selected_proposal_id, "")
        self.assertEqual(len(decision.reports), 1)
        self.assertFalse(decision.reports[0].accepted)

    def test_chat_completions_provider_generates_supervised_candidate(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text()
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _RecordingChatCompletionsTransport(semantic_text)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            endpoint="https://llm.example/v1/chat/completions",
            provider_name="test-chat-completions-compatible",
            candidates=1,
            transport=transport,
        )

        decision = LLMProposalAgent(provider=provider).propose_and_supervise(prompt)

        self.assertTrue(decision.accepted, decision.summary())
        self.assertEqual(decision.selected_proposal_id, "candidate-1")
        self.assertEqual(decision.reports[0].source, "llm:test-chat-completions-compatible:candidate-1")
        self.assertEqual(len(transport.calls), 1)
        url, headers, payload, timeout = transport.calls[0]
        self.assertEqual(url, "https://llm.example/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(timeout, 30.0)
        self.assertIn("MatrixAI .semantic proposals only", payload["messages"][0]["content"])

    def test_chat_completions_provider_from_env_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_env_file = str(Path(tmpdir) / "missing.env")
            env = {"MATRIXAI_LLM_ENV_FILE": missing_env_file}
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    ChatCompletionsLLMProposalProvider.from_env()

    def test_chat_completions_provider_rejects_dotenv_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("MATRIXAI_LLM_API_KEY=xxxxxx\n", encoding="utf-8")
            with patch.dict(os.environ, {"MATRIXAI_LLM_ENV_FILE": str(env_file)}, clear=True):
                with self.assertRaises(ValueError):
                    ChatCompletionsLLMProposalProvider.from_env()

    def test_chat_completions_provider_loads_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "MATRIXAI_LLM_API_KEY=file-key\n"
                "MATRIXAI_LLM_MODEL=file-model\n"
                "MATRIXAI_LLM_CANDIDATES=2\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MATRIXAI_LLM_ENV_FILE": str(env_file)},
                clear=True,
            ):
                provider = ChatCompletionsLLMProposalProvider.from_env()

        self.assertEqual(provider.api_key, "file-key")
        self.assertEqual(provider.model_name, "file-model")
        self.assertEqual(provider.candidates, 2)

    def test_architect_generates_valid_risk_agent(self) -> None:
        semantic_text = (ROOT / "examples" / "fall-risk.semantic").read_text()

        architect = ArchitectAgent()
        plan = architect.plan_from_text(semantic_text)
        mxai_text = architect.to_mxai(plan)
        program = parse_text(mxai_text)
        result = VerifierAgent().verify(program)

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(plan.mode, "risk")
        self.assertIn("FUNCTION RiskModel", mxai_text)
        self.assertIn("Risk ~ Normal(R, uncertainty(Patient))", mxai_text)
        self.assertEqual(program.functions[0].semantic.kind, "sigmoid_linear")
        self.assertEqual(program.distributions[0].distribution_type, "Normal")

    def test_email_agent_parses_and_validates(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.mxai")
        result = VerifierAgent().verify(program)

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(program.project, "EmailAgent")
        self.assertEqual(program.vectors[0].name, "Email")
        self.assertEqual(program.graph.nodes[-1], "DraftReply")
        self.assertEqual(program.graph.node_types["Classifier"], "function")
        self.assertEqual(program.functions[0].semantic.kind, "softmax_linear")
        self.assertEqual(program.functions[1].semantic.kind, "sigmoid_threshold")
        self.assertEqual(program.actions[0].policy, "simulate_only")
        self.assertEqual(program.actions[0].condition.source, "ReplyActivation")

    def test_email_agent_runs_simulated_action(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text())

        result = MatrixAIRuntime().run(program, input_data)

        self.assertEqual(len(result["actions"]), 1)
        self.assertTrue(result["actions"][0]["activated"])
        self.assertEqual(result["actions"][0]["call"], "simulated.email.draft")
        self.assertEqual(result["trace"][0]["step"], 1)
        self.assertEqual(result["trace"][1]["expression_kind"], "softmax_linear")
        self.assertEqual(result["trace"][-1]["policy"], "simulate_only")


    def test_goal_translator_derives_rules_from_goals(self) -> None:
        semantic_text = (ROOT / "examples" / "email-agent.semantic").read_text()
        plan = ArchitectAgent().plan_from_text(semantic_text)

        # email-agent has goals: classify_incoming_email, minimize_false_replies
        self.assertGreater(len(plan.verification_rules), 0)
        checks = {rule["check"] for rule in plan.verification_rules}
        self.assertIn("action_threshold_min", checks)
        self.assertIn("distribution_required", checks)

    def test_planner_verifier_fails_goal_rule_violation(self) -> None:
        semantic_text = (ROOT / "examples" / "email-agent.semantic").read_text()
        plan_data = ArchitectAgent().plan_from_text(semantic_text).to_dict()
        # Lower the action threshold below what minimize_false_replies requires (>= 0.85)
        plan_data["actions"][0]["when"] = "ReplyActivation > 0.50"
        from matrixai.agents.architect import SemanticPlan
        bad_plan = SemanticPlan(**plan_data)

        result = PlannerVerifier().verify(bad_plan)

        self.assertFalse(result.ok)
        self.assertTrue(
            any("minimize_false_replies" in error for error in result.errors),
            result.errors,
        )

    def test_optimizer_suggests_merge_for_email_agent(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.mxai")

        report = OptimizerAgent().analyze(program)

        kinds = {s.kind for s in report.suggestions}
        self.assertIn("merge_linear_activation", kinds)
        merge = next(s for s in report.suggestions if s.kind == "merge_linear_activation")
        self.assertEqual(len(merge.nodes), 2)

    def test_optimizer_no_suggestions_on_minimal_graph(self) -> None:
        from matrixai.ir.schema import (
            MatrixAIProgram, VectorSpec, FunctionSpec, ExpressionSpec,
            GraphSpec, ActionSpec, ActionConditionSpec, AuditSpec,
        )
        program = MatrixAIProgram(
            project="Minimal",
            vectors=[VectorSpec(name="X", size=2, fields=["a", "b"])],
            functions=[
                FunctionSpec(
                    name="F",
                    output="y",
                    expression="sigmoid(W1 * X + b1)",
                    semantic=ExpressionSpec(raw="sigmoid(W1 * X + b1)", kind="sigmoid_linear", inputs=["X"]),
                )
            ],
            distributions=[],
            graph=GraphSpec(nodes=["X", "F"], edges=[("X", "F")], node_types={"X": "vector", "F": "function"}),
            actions=[],
            audit=AuditSpec(explain=["X", "F"]),
        )

        report = OptimizerAgent().analyze(program)

        # No merge (no consecutive sigmoid+threshold), no isolated, no fan-in, no complexity issues
        kinds = {s.kind for s in report.suggestions}
        self.assertNotIn("merge_linear_activation", kinds)
        self.assertNotIn("prune_isolated_nodes", kinds)

    # ------------------------------------------------------------------
    # Python compiler tests
    # ------------------------------------------------------------------

    def test_python_compiler_generated_email_matches_runtime(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text())

        source = PythonBackendCompiler().compile(program)

        self.assertIn("def run(input_data", source)
        self.assertNotIn("from matrixai", source)
        self.assertNotIn("GRAPH_NODES", source)
        self.assertNotIn("VECTORS =", source)
        self.assertNotIn("FUNCTIONS =", source)
        self.assertNotIn("DISTRIBUTIONS =", source)
        self.assertNotIn("ACTIONS =", source)
        self.assertNotIn("for node in", source)
        self.assertIn("# VECTOR Email", source)
        self.assertIn("# FUNCTION Classifier", source)
        self.assertIn("Classifier = _email_classifier(state['Email'])", source)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_email_agent.py"
            output_path.write_text(source, encoding="utf-8")
            module = self._load_generated_module(output_path)

            generated = module.run(input_data)
            runtime = MatrixAIRuntime().run(program, input_data)

        self.assertEqual(generated["actions"], runtime["actions"])
        self.assertEqual(len(generated["trace"]), len(runtime["trace"]))
        self.assertAlmostEqual(
            generated["state"]["ReplyActivation"],
            runtime["state"]["ReplyActivation"],
        )

    def test_python_compiler_generated_pharmacy_matches_runtime(self) -> None:
        program = parse_file(ROOT / "examples" / "pharmacy-dispense.mxai")
        input_data = json.loads(
            (ROOT / "examples" / "pharmacy-dispense-sample.json").read_text()
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_pharmacy.py"
            output_path.write_text(PythonBackendCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)

            generated = module.run(input_data)
            runtime = MatrixAIRuntime().run(program, input_data)

        self.assertEqual(generated["actions"], runtime["actions"])
        self.assertTrue(generated["actions"][0]["activated"])

    # ------------------------------------------------------------------
    # MathematicalAgent tests
    # ------------------------------------------------------------------

    def test_mathematical_agent_translates_simple_rule(self) -> None:
        report = MathematicalAgent().translate(["if risk > 0.8 then alert"])

        self.assertTrue(report.all_resolved)
        self.assertEqual(len(report.translations), 1)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "sigmoid_threshold")
        self.assertIn("sigmoid", t.expression)
        self.assertIn("0.8", t.expression)
        self.assertIn("risk", t.inputs)
        self.assertEqual(t.parameters["threshold"], 0.8)

    def test_mathematical_agent_translates_and_rule(self) -> None:
        report = MathematicalAgent().translate(
            ["if age > 0.8 and mobility < 0.3 then notify"]
        )

        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "sigmoid_and")
        self.assertIn("age", t.inputs)
        self.assertIn("mobility", t.inputs)
        self.assertIn("sigmoid_product", t.expression)

    def test_mathematical_agent_translates_or_rule(self) -> None:
        report = MathematicalAgent().translate(
            ["if urgency > 0.9 or priority > 0.85 then dispatch"]
        )

        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "sigmoid_or")
        self.assertIn("sigmoid_or", t.expression)

    def test_mathematical_agent_translates_classify_rule(self) -> None:
        report = MathematicalAgent().translate(
            ["classify Email into support, sales, operations"]
        )

        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "softmax_linear")
        self.assertEqual(t.parameters["num_classes"], 3)
        self.assertIn("softmax", t.expression)

    def test_mathematical_agent_marks_unresolved(self) -> None:
        report = MathematicalAgent().translate(["do something weird with x"])

        self.assertFalse(report.all_resolved)
        self.assertEqual(len(report.unresolved), 1)
        self.assertEqual(len(report.translations), 0)

    def test_mathematical_agent_translates_negative_threshold_simple(self) -> None:
        # Regression: LLM-generated rules like "if temp >= -273.15" must not be
        # left unresolved due to the leading minus sign.
        report = MathematicalAgent().translate(
            ["if grados_centigrados >= -273.15 then calcular_kelvin"]
        )
        self.assertTrue(report.all_resolved)
        self.assertEqual(len(report.unresolved), 0)
        self.assertEqual(len(report.translations), 1)

    def test_mathematical_agent_translates_negative_threshold_and(self) -> None:
        report = MathematicalAgent().translate(
            ["if temperatura >= -273.15 and temperatura <= 1000.0 then valid"]
        )
        self.assertTrue(report.all_resolved)
        self.assertEqual(len(report.unresolved), 0)

    def test_mathematical_agent_translates_negative_threshold_or(self) -> None:
        report = MathematicalAgent().translate(
            ["if x < -10.5 or y >= -0.5 then activate"]
        )
        self.assertTrue(report.all_resolved)
        self.assertEqual(len(report.unresolved), 0)

    def test_mathematical_agent_positive_threshold_still_works(self) -> None:
        report = MathematicalAgent().translate(["if score > 0.7 then send"])
        self.assertTrue(report.all_resolved)
        self.assertEqual(len(report.unresolved), 0)

    def test_mathematical_agent_mixed_batch(self) -> None:
        rules = [
            "if risk > 0.8 then alert",
            "if confidence >= 0.95 then reply",
            "if score > 0.7 and urgency > 0.5 then dispatch",
            "classify Email into support, sales, spam",
            "unknown rule that cannot be parsed",
        ]
        report = MathematicalAgent().translate(rules)

        self.assertEqual(len(report.translations), 4)
        self.assertEqual(len(report.unresolved), 1)

    # ------------------------------------------------------------------
    # P0 — LLM trazabilidad, retries y control de coste
    # ------------------------------------------------------------------

    def test_llm_call_trace_recorded_on_success(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text()
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _RecordingChatCompletionsTransport(semantic_text)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            endpoint="https://llm.example/v1/chat/completions",
            candidates=1,
            retry_delay_s=0.0,
            transport=transport,
        )

        _, call_traces = provider.propose(prompt)

        self.assertEqual(len(call_traces), 1)
        trace = call_traces[0]
        self.assertEqual(trace.http_status, 200)
        self.assertEqual(trace.attempt, 1)
        self.assertGreaterEqual(trace.latency_ms, 0.0)
        self.assertEqual(trace.error, "")
        self.assertRegex(trace.prompt_hash, r"^[0-9a-f]{12}$")

    def test_llm_call_trace_prompt_hash_is_stable(self) -> None:
        prompt = "Classify emails with high confidence"
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _RecordingChatCompletionsTransport(semantic_text)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            retry_delay_s=0.0,
            transport=transport,
        )

        _, traces1 = provider.propose(prompt)
        # rebuild transport (RecordingChatCompletionsTransport is reusable)
        _, traces2 = provider.propose(prompt)

        self.assertEqual(traces1[0].prompt_hash, traces2[0].prompt_hash)

    def test_llm_call_trace_token_usage_captured(self) -> None:
        prompt = "Classify emails"
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _TokenUsageTransport(semantic_text, total_tokens=42)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            retry_delay_s=0.0,
            transport=transport,
        )

        _, call_traces = provider.propose(prompt)

        self.assertEqual(call_traces[0].token_usage, 42)

    def test_llm_call_trace_recorded_on_http_error(self) -> None:
        transport = _AlwaysErrorTransport(status=400)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            max_retries=1,
            retry_delay_s=0.0,
            transport=transport,
        )

        with self.assertRaises(LLMHTTPError) as ctx:
            provider.propose("some prompt")

        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(transport.call_count, 1)  # 400 is not retried

    def test_chat_completions_provider_retries_on_429(self) -> None:
        prompt = "Classify emails"
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _RetryingTransport(fail_status=429, fail_times=1, semantic_text=semantic_text)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            max_retries=3,
            retry_delay_s=0.0,
            transport=transport,
        )

        proposals, call_traces = provider.propose(prompt)

        self.assertEqual(transport.call_count, 2)
        self.assertEqual(len(call_traces), 2)
        self.assertEqual(call_traces[0].http_status, 429)
        self.assertNotEqual(call_traces[0].error, "")
        self.assertEqual(call_traces[1].http_status, 200)
        self.assertEqual(call_traces[1].error, "")
        self.assertGreater(len(proposals), 0)

    def test_chat_completions_provider_retries_on_5xx(self) -> None:
        prompt = "Classify emails"
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _RetryingTransport(fail_status=503, fail_times=2, semantic_text=semantic_text)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            max_retries=3,
            retry_delay_s=0.0,
            transport=transport,
        )

        proposals, call_traces = provider.propose(prompt)

        self.assertEqual(transport.call_count, 3)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(call_traces[-1].http_status, 200)

    def test_chat_completions_provider_no_retry_on_400(self) -> None:
        transport = _AlwaysErrorTransport(status=400)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            max_retries=3,
            retry_delay_s=0.0,
            transport=transport,
        )

        with self.assertRaises(LLMHTTPError) as ctx:
            provider.propose("some prompt")

        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(transport.call_count, 1)

    def test_chat_completions_provider_retries_exhausted(self) -> None:
        transport = _AlwaysErrorTransport(status=429)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            max_retries=3,
            retry_delay_s=0.0,
            transport=transport,
        )

        with self.assertRaises(LLMHTTPError) as ctx:
            provider.propose("some prompt")

        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(transport.call_count, 3)

    def test_chat_completions_provider_no_choices_raises(self) -> None:
        def empty_transport(url, headers, payload, timeout):
            return {"choices": [], "usage": {"total_tokens": 5}}

        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            retry_delay_s=0.0,
            transport=empty_transport,
        )

        with self.assertRaises(RuntimeError, msg="no semantic proposals"):
            provider.propose("some prompt")

    def test_chat_completions_provider_candidate_without_project_skipped(self) -> None:
        def bad_transport(url, headers, payload, timeout):
            return {
                "choices": [{"message": {"content": "This is not a MatrixAI spec."}}],
                "usage": {"total_tokens": 10},
            }

        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            retry_delay_s=0.0,
            transport=bad_transport,
        )

        with self.assertRaises(RuntimeError):
            provider.propose("some prompt")

    def test_chat_completions_provider_budget_exceeded(self) -> None:
        prompt = "Classify emails"
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _TokenUsageTransport(semantic_text, total_tokens=200)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            token_budget=100,
            retry_delay_s=0.0,
            transport=transport,
        )

        with self.assertRaises(LLMBudgetExceededError) as ctx:
            provider.propose(prompt)

        self.assertEqual(ctx.exception.used, 200)
        self.assertEqual(ctx.exception.budget, 100)

    def test_propose_and_supervise_exhaust_all_evaluates_all_candidates(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text()

        decision = LLMProposalAgent(
            provider=_FirstUnsafeThenSafeProvider()
        ).propose_and_supervise(prompt, exhaust_all=True)

        # Both candidates evaluated even though first is rejected
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.selected_proposal_id, "candidate-2")
        self.assertEqual(len(decision.reports), 2)
        self.assertFalse(decision.reports[0].accepted)
        self.assertTrue(decision.reports[1].accepted)

    def test_llm_proposal_batch_call_traces_in_json(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text()
        semantic_text = PromptAgent().synthesize(prompt).semantic_text
        transport = _TokenUsageTransport(semantic_text, total_tokens=55)
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="test-model",
            retry_delay_s=0.0,
            transport=transport,
        )
        agent = LLMProposalAgent(provider=provider)

        batch = agent.propose(prompt)
        batch_dict = batch.to_dict()

        self.assertIn("call_traces", batch_dict)
        self.assertEqual(len(batch_dict["call_traces"]), 1)
        ct = batch_dict["call_traces"][0]
        self.assertEqual(ct["http_status"], 200)
        self.assertEqual(ct["token_usage"], 55)
        self.assertEqual(ct["error"], "")
        self.assertRegex(ct["prompt_hash"], r"^[0-9a-f]{12}$")

    # ------------------------------------------------------------------
    # P1 — Symbolic expression AST
    # ------------------------------------------------------------------

    def test_expr_literal_eval(self) -> None:
        from matrixai.ir.expr import LiteralNode
        self.assertAlmostEqual(LiteralNode(0.7).eval({}), 0.7)

    def test_expr_var_eval(self) -> None:
        from matrixai.ir.expr import VarNode
        self.assertAlmostEqual(VarNode("x").eval({"x": 0.5}), 0.5)

    def test_expr_var_dotted_eval(self) -> None:
        from matrixai.ir.expr import VarNode
        self.assertAlmostEqual(VarNode("Conf.max").eval({"Conf": {"max": 0.9}}), 0.9)

    def test_expr_binop_add_mul(self) -> None:
        from matrixai.ir.expr import BinOpNode, LiteralNode, VarNode
        node = BinOpNode(
            "+",
            BinOpNode("*", LiteralNode(0.7), VarNode("a")),
            BinOpNode("*", LiteralNode(0.3), VarNode("b")),
        )
        self.assertAlmostEqual(node.eval({"a": 0.8, "b": 0.6}), 0.7 * 0.8 + 0.3 * 0.6)

    def test_expr_call_normalize(self) -> None:
        from matrixai.ir.expr import parse_expr
        self.assertAlmostEqual(parse_expr("normalize(1.5)").eval({}), 1.0)
        self.assertAlmostEqual(parse_expr("normalize(-0.5)").eval({}), 0.0)
        self.assertAlmostEqual(parse_expr("normalize(0.7)").eval({}), 0.7)

    def test_expr_call_sigmoid(self) -> None:
        from matrixai.ir.expr import parse_expr
        self.assertAlmostEqual(parse_expr("sigmoid(0.0)").eval({}), 0.5, places=5)

    def test_expr_call_clip_and_scale(self) -> None:
        from matrixai.ir.expr import parse_expr
        self.assertAlmostEqual(parse_expr("clip(x, 0.0, 1.0)").eval({"x": -0.5}), 0.0)
        self.assertAlmostEqual(parse_expr("scale(x, 0.0, 10.0)").eval({"x": 5.0}), 0.5)

    def test_parse_expr_weighted_sum_structure(self) -> None:
        from matrixai.ir.expr import parse_expr, BinOpNode
        node = parse_expr("0.7 * relevance + 0.3 * coherence")
        self.assertIsInstance(node, BinOpNode)
        self.assertEqual(node.op, "+")

    def test_extract_weighted_sum_two_terms(self) -> None:
        from matrixai.ir.expr import parse_expr, extract_weighted_sum, WeightedSumNode
        node = parse_expr("0.7 * relevance + 0.3 * coherence")
        ws = extract_weighted_sum(node)
        self.assertIsNotNone(ws)
        self.assertIsInstance(ws, WeightedSumNode)
        self.assertEqual(len(ws.terms), 2)
        weights = [w for w, _ in ws.terms]
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_extract_weighted_sum_eval(self) -> None:
        from matrixai.ir.expr import parse_expr, extract_weighted_sum
        ws = extract_weighted_sum(parse_expr("0.7 * a + 0.3 * b"))
        self.assertIsNotNone(ws)
        self.assertAlmostEqual(ws.eval({"a": 1.0, "b": 0.0}), 0.7)

    def test_collect_vars(self) -> None:
        from matrixai.ir.expr import parse_expr, collect_vars
        node = parse_expr("0.7 * relevance(x) + 0.3 * coherence(y)")
        vars_ = collect_vars(node)
        self.assertIn("x", vars_)
        self.assertIn("y", vars_)

    def test_expr_to_dict_round_trip(self) -> None:
        from matrixai.ir.expr import parse_expr
        node = parse_expr("0.5 * a + 0.5 * b")
        d = node.to_dict()
        self.assertEqual(d["type"], "binop")
        self.assertEqual(d["op"], "+")

    def test_expr_division_by_zero_raises(self) -> None:
        from matrixai.ir.expr import parse_expr
        with self.assertRaises(ZeroDivisionError):
            parse_expr("x / y").eval({"x": 1.0, "y": 0.0})

    def test_parse_expr_unknown_function_resolves_from_env(self) -> None:
        from matrixai.ir.expr import parse_expr
        # Non-builtin call → resolved from env as pre-computed oracle value
        result = parse_expr("0.7 * relevance(x) + 0.3 * coherence(x)").eval(
            {"x": 0.5, "relevance": 0.9, "coherence": 0.7}
        )
        self.assertAlmostEqual(result, 0.7 * 0.9 + 0.3 * 0.7)

    # ------------------------------------------------------------------
    # P1 — MathematicalAgent extensions
    # ------------------------------------------------------------------

    def test_mathematical_agent_build_symbolic_weighted_sum(self) -> None:
        t = MathematicalAgent().build_symbolic("0.7 * relevance(x) + 0.3 * coherence(x)")
        self.assertEqual(t.expression_kind, "symbolic_weighted_sum")
        self.assertEqual(len(t.parameters["terms"]), 2)
        self.assertAlmostEqual(t.parameters["terms"][0]["weight"], 0.7)

    def test_mathematical_agent_build_symbolic_expr(self) -> None:
        t = MathematicalAgent().build_symbolic("sigmoid(20 * (x - 0.5))")
        self.assertEqual(t.expression_kind, "symbolic_expr")
        self.assertIn("x", t.inputs)

    def test_mathematical_agent_build_symbolic_invalid(self) -> None:
        with self.assertRaises(ValueError):
            MathematicalAgent().build_symbolic("@@@bad expr")

    def test_mathematical_agent_translate_assign(self) -> None:
        report = MathematicalAgent().translate(
            ["final_score = 0.7 * relevance + 0.3 * coherence"]
        )
        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "symbolic_weighted_sum")
        self.assertEqual(t.parameters["output"], "final_score")

    def test_mathematical_agent_translate_aggregate_max(self) -> None:
        report = MathematicalAgent().translate(["aggregate s1, s2, s3 using max"])
        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "aggregate_max")
        self.assertEqual(t.parameters["inputs"], ["s1", "s2", "s3"])

    def test_mathematical_agent_translate_aggregate_mean(self) -> None:
        report = MathematicalAgent().translate(["aggregate s1, s2 using mean"])
        self.assertTrue(report.all_resolved)
        self.assertEqual(report.translations[0].expression_kind, "aggregate_mean")

    def test_mathematical_agent_translate_aggregate_softmax(self) -> None:
        report = MathematicalAgent().translate(["aggregate a, b, c using softmax"])
        self.assertTrue(report.all_resolved)
        self.assertEqual(report.translations[0].expression_kind, "aggregate_softmax")

    def test_mathematical_agent_translate_aggregate_vote(self) -> None:
        report = MathematicalAgent().translate(["aggregate p1, p2, p3 using vote"])
        self.assertTrue(report.all_resolved)
        self.assertEqual(report.translations[0].expression_kind, "aggregate_vote")

    def test_mathematical_agent_translate_normalize(self) -> None:
        report = MathematicalAgent().translate(["normalize raw_score"])
        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "normalize")
        self.assertEqual(t.parameters["var"], "raw_score")
        self.assertAlmostEqual(t.parameters["lo"], 0.0)
        self.assertAlmostEqual(t.parameters["hi"], 1.0)

    def test_mathematical_agent_translate_normalize_with_range(self) -> None:
        report = MathematicalAgent().translate(["normalize raw_score to [0, 100]"])
        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertAlmostEqual(t.parameters["hi"], 100.0)

    def test_mathematical_agent_translate_select(self) -> None:
        report = MathematicalAgent().translate(["select best from candidates by score"])
        self.assertTrue(report.all_resolved)
        t = report.translations[0]
        self.assertEqual(t.expression_kind, "select_argmax")
        self.assertEqual(t.parameters["candidates"], "candidates")
        self.assertEqual(t.parameters["score_input"], "score")

    # ------------------------------------------------------------------
    # P1 — Runtime handlers for new kinds
    # ------------------------------------------------------------------

    def _make_simple_program(
        self, func_name: str, kind: str, raw: str, inputs: list,
        parameters: dict, fields: list, output_var: str = "result"
    ):
        from matrixai.ir.schema import (
            MatrixAIProgram, VectorSpec, FunctionSpec, ExpressionSpec,
            GraphSpec, AuditSpec,
        )
        return MatrixAIProgram(
            project="Test",
            vectors=[VectorSpec(name="X", size=len(fields), fields=fields)],
            functions=[
                FunctionSpec(
                    name=func_name,
                    output=output_var,
                    expression=raw,
                    semantic=ExpressionSpec(
                        raw=raw, kind=kind, inputs=inputs, parameters=parameters,
                    ),
                )
            ],
            distributions=[],
            graph=GraphSpec(
                nodes=["X", func_name],
                edges=[("X", func_name)],
                node_types={"X": "vector", func_name: "function"},
            ),
            actions=[],
            audit=AuditSpec(explain=["X", func_name]),
        )

    def test_runtime_evaluates_aggregate_max(self) -> None:
        program = self._make_simple_program(
            func_name="BestScore",
            kind="aggregate_max",
            raw="max(s1, s2)",
            inputs=["s1", "s2"],
            parameters={"inputs": ["s1", "s2"], "method": "max"},
            fields=["s1", "s2"],
            output_var="best",
        )
        result = MatrixAIRuntime().run(program, {"X": {"s1": 0.6, "s2": 0.9}})
        self.assertAlmostEqual(result["state"]["best"], 0.9)

    def test_runtime_evaluates_aggregate_mean(self) -> None:
        program = self._make_simple_program(
            func_name="MeanScore",
            kind="aggregate_mean",
            raw="(s1 + s2) / 2",
            inputs=["s1", "s2"],
            parameters={"inputs": ["s1", "s2"], "method": "mean"},
            fields=["s1", "s2"],
            output_var="mean_val",
        )
        result = MatrixAIRuntime().run(program, {"X": {"s1": 0.4, "s2": 0.8}})
        self.assertAlmostEqual(result["state"]["mean_val"], 0.6)

    def test_runtime_evaluates_normalize(self) -> None:
        program = self._make_simple_program(
            func_name="Normed",
            kind="normalize",
            raw="normalize(raw)",
            inputs=["raw"],
            parameters={"var": "raw", "lo": 0.0, "hi": 10.0},
            fields=["raw"],
            output_var="normed",
        )
        result = MatrixAIRuntime().run(program, {"X": {"raw": 5.0}})
        self.assertAlmostEqual(result["state"]["normed"], 0.5)

    def test_runtime_evaluates_symbolic_weighted_sum(self) -> None:
        program = self._make_simple_program(
            func_name="FinalScore",
            kind="symbolic_weighted_sum",
            raw="0.7 * relevance + 0.3 * coherence",
            inputs=["relevance", "coherence"],
            parameters={},
            fields=["relevance", "coherence"],
            output_var="final_score",
        )
        result = MatrixAIRuntime().run(
            program, {"X": {"relevance": 0.8, "coherence": 0.6}}
        )
        self.assertAlmostEqual(
            result["state"]["final_score"], 0.7 * 0.8 + 0.3 * 0.6, places=5
        )

    def test_runtime_vector_fields_exposed_in_state(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text())
        result = MatrixAIRuntime().run(program, input_data)
        # Individual field names from the Email vector should be in state
        self.assertIn("urgency", result["state"])
        self.assertIn("sender_trust", result["state"])

    def test_p1_mxai_symbolic_normalize_aggregate_runtime_matches_compiled(self) -> None:
        mxai_text = """PROJECT P1Expressions

VECTOR X[4]
  relevance
  coherence
  raw
  spare
END

FUNCTION FinalScore
  final_score = 0.7 * relevance + 0.3 * coherence
END

FUNCTION Normed
  normed = scale(raw, 0, 10)
END

FUNCTION BestScore
  best = max(final_score, normed)
END

FUNCTION MeanScore
  mean_val = mean(final_score, normed)
END

GRAPH
  X -> FinalScore -> Normed -> BestScore -> MeanScore
END

AUDIT
  EXPLAIN X -> FinalScore -> Normed -> BestScore -> MeanScore
END
"""
        program = parse_text(mxai_text)
        kinds = [function.semantic.kind for function in program.functions]
        self.assertEqual(
            kinds,
            ["symbolic_weighted_sum", "normalize", "aggregate_max", "aggregate_mean"],
        )
        input_data = {"X": {"relevance": 0.8, "coherence": 0.6, "raw": 5.0, "spare": 0.0}}

        runtime = MatrixAIRuntime().run(program, input_data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_p1.py"
            output_path.write_text(PythonBackendCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)
            generated = module.run(input_data)

        self.assertAlmostEqual(runtime["state"]["final_score"], 0.74, places=6)
        self.assertAlmostEqual(runtime["state"]["normed"], 0.5, places=6)
        self.assertAlmostEqual(runtime["state"]["best"], 0.74, places=6)
        self.assertAlmostEqual(runtime["state"]["mean_val"], 0.62, places=6)
        self.assertEqual(generated["state"], runtime["state"])
        self.assertEqual(generated["trace"], runtime["trace"])

    def test_p1_mxai_select_argmax_runtime_matches_compiled(self) -> None:
        mxai_text = """PROJECT P1Select

VECTOR Email[8]
  urgency
  sender_trust
  topic_support
  topic_sales
  sentiment
  has_attachment
  previous_interactions
  language_confidence
END

FUNCTION Scores
  scores = softmax(W1 * Email + b1)
END

FUNCTION Choice
  choice = argmax(scores)
END

GRAPH
  Email -> Scores -> Choice
END

AUDIT
  EXPLAIN Email -> Scores -> Choice
END
"""
        program = parse_text(mxai_text)
        self.assertEqual(program.functions[1].semantic.kind, "select_argmax")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text())

        runtime = MatrixAIRuntime().run(program, input_data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_select.py"
            output_path.write_text(PythonBackendCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)
            generated = module.run(input_data)

        self.assertEqual(runtime["state"]["choice"], generated["state"]["choice"])
        self.assertIn(runtime["state"]["choice"], runtime["state"]["scores"])

    def test_unknown_function_semantics_fail_explicitly(self) -> None:
        from matrixai.ir.schema import MatrixAIProgram, VectorSpec, FunctionSpec, ExpressionSpec
        from matrixai.ir.schema import GraphSpec, AuditSpec

        program = MatrixAIProgram(
            project="UnknownTest",
            vectors=[VectorSpec(name="X", size=1, fields=["x"])],
            functions=[
                FunctionSpec(
                    name="Broken",
                    output="y",
                    expression="not parseable @@@",
                    semantic=ExpressionSpec(raw="not parseable @@@", kind="unknown"),
                )
            ],
            graph=GraphSpec(
                nodes=["X", "Broken"],
                edges=[("X", "Broken")],
                node_types={"X": "vector", "Broken": "function"},
            ),
            audit=AuditSpec(explain=["X", "Broken"]),
        )

        with self.assertRaisesRegex(ValueError, "Unsupported function semantic kind"):
            MatrixAIRuntime().run(program, {"X": {"x": 1.0}})
        with self.assertRaisesRegex(ValueError, "Unsupported function semantic kind"):
            PythonBackendCompiler().compile(program)


if __name__ == "__main__":
    unittest.main()
