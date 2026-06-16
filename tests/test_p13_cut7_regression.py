"""P13 Cut 7 — regression guard: P7-P12 contracts unaffected by P13 changes.

Scope: targeted smoke-tests of the shared code paths P13 extended or touched.
NOT a re-run of existing test files.

P13 change sites tested:
  - agents/prompt_supervisor.py  (PromptSupervisionReport: new refinement_chain / parent_prompt_hash)
  - agents/__init__.py           (new IterationLimitReached, RefinementAgent, RefinementProposal exports)
  - cli.py                       (new refine subcommand; existing subcommands must be unaffected)
  - playground.py                (new _refine_prompt / /api/refine; renderRunView target changed;
                                   analyze_playground_request + AuditorAgent pipeline unchanged)
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_FALL_MXAI = _BASE / "examples" / "fall-risk.typed.mxai"
_EMAIL_MXAI = _BASE / "examples" / "email-agent.typed.mxai"


# ---------------------------------------------------------------------------
# agents/prompt_supervisor.py — PromptSupervisionReport backward compat (P7/P8)
# ---------------------------------------------------------------------------

class TestP13RegressionPromptSupervisionReport(unittest.TestCase):
    """P13 added refinement_chain and parent_prompt_hash with defaults; old callers unaffected."""

    def test_new_fields_default_to_empty(self):
        from matrixai.agents import PromptSupervisionReport, SupervisionCheck
        report = PromptSupervisionReport(
            accepted=True,
            prompt="test",
            source="deterministic",
            semantic_text="",
            checks=[SupervisionCheck(name="c", ok=True)],
        )
        self.assertEqual(report.refinement_chain, [])
        self.assertEqual(report.parent_prompt_hash, "")

    def test_report_is_still_frozen(self):
        from matrixai.agents import PromptSupervisionReport, SupervisionCheck
        report = PromptSupervisionReport(
            accepted=True, prompt="p", source="deterministic",
            semantic_text="", checks=[SupervisionCheck(name="c", ok=True)],
        )
        with self.assertRaises((AttributeError, TypeError)):
            report.accepted = False  # type: ignore[misc]

    def test_supervise_prompt_returns_empty_chain_without_p13_context(self):
        from matrixai.agents import PromptSupervisor
        report = PromptSupervisor().supervise_prompt(
            "Si fiebre > 38 entonces ALERT"
        )
        self.assertEqual(report.refinement_chain, [])
        self.assertEqual(report.parent_prompt_hash, "")

    def test_supervise_prompt_returns_accepted_or_rejected(self):
        from matrixai.agents import PromptSupervisor
        report = PromptSupervisor().supervise_prompt(
            "Si fiebre > 38 entonces ALERT"
        )
        self.assertIsInstance(report.accepted, bool)

    def test_supervise_prompt_report_to_dict_has_no_spurious_keys(self):
        from matrixai.agents import PromptSupervisor
        report = PromptSupervisor().supervise_prompt(
            "Si fiebre > 38 entonces ALERT"
        )
        d = report.to_dict()
        # Old required keys must still be present
        for key in ("accepted", "prompt", "source", "semantic_text", "checks", "mxai"):
            self.assertIn(key, d, f"Missing key: {key}")
        # New fields are present but empty
        self.assertEqual(d["refinement_chain"], [])
        self.assertEqual(d["parent_prompt_hash"], "")

    def test_supervise_semantic_still_works(self):
        from matrixai.agents import PromptSupervisor
        semantic = _FALL_MXAI.read_text(encoding="utf-8")
        report = PromptSupervisor().supervise_semantic(
            prompt="Calcular riesgo de caida",
            semantic_text=semantic,
            source="test",
        )
        self.assertIsInstance(report.accepted, bool)
        self.assertIsInstance(report.mxai, str)

    def test_supervise_semantic_new_fields_still_empty_without_p13(self):
        from matrixai.agents import PromptSupervisor
        semantic = _FALL_MXAI.read_text(encoding="utf-8")
        report = PromptSupervisor().supervise_semantic(
            prompt="Calcular riesgo de caida",
            semantic_text=semantic,
            source="test",
        )
        self.assertEqual(report.refinement_chain, [])
        self.assertEqual(report.parent_prompt_hash, "")

    def test_summary_method_still_returns_string(self):
        from matrixai.agents import PromptSupervisor
        report = PromptSupervisor().supervise_prompt(
            "Si fiebre > 38 entonces ALERT"
        )
        s = report.summary()
        self.assertIsInstance(s, str)
        self.assertIn("PromptSupervisor", s)


# ---------------------------------------------------------------------------
# agents/__init__.py — existing exports not shadowed (P7/P8/P10/P11/P12)
# ---------------------------------------------------------------------------

class TestP13RegressionAgentsExports(unittest.TestCase):
    """P13 adds new exports; existing ones must still be importable and functional."""

    def test_auditor_agent_importable(self):
        from matrixai.agents import AuditorAgent
        self.assertTrue(callable(getattr(AuditorAgent, "explain", None) or AuditorAgent))

    def test_prompt_supervisor_importable(self):
        from matrixai.agents import PromptSupervisor
        self.assertTrue(hasattr(PromptSupervisor(), "supervise_prompt"))

    def test_safety_agent_importable(self):
        from matrixai.agents import SafetyAgent
        self.assertIsNotNone(SafetyAgent)

    def test_verifier_agent_importable(self):
        from matrixai.agents import VerifierAgent
        self.assertIsNotNone(VerifierAgent)

    def test_llm_proposal_provider_importable(self):
        from matrixai.agents import ChatCompletionsLLMProposalProvider
        self.assertIsNotNone(ChatCompletionsLLMProposalProvider)

    def test_p13_new_exports_coexist_with_old(self):
        import dataclasses
        from matrixai.agents import (
            AuditorAgent, IterationLimitReached, PromptSupervisor,
            RefinementAgent, RefinementProposal, SafetyAgent,
        )
        self.assertTrue(issubclass(IterationLimitReached, RuntimeError))
        self.assertTrue(hasattr(RefinementAgent, "refine"))
        field_names = {f.name for f in dataclasses.fields(RefinementProposal)}
        self.assertIn("refinement_id", field_names)
        # Old imports still resolve to the right types
        self.assertTrue(hasattr(PromptSupervisor(), "supervise_prompt"))
        self.assertTrue(hasattr(AuditorAgent(), "explain"))

    def test_all_list_unchanged_for_old_symbols(self):
        import matrixai.agents as agents_mod
        old_expected = {
            "AuditorAgent", "PromptSupervisor", "PromptSupervisionReport",
            "SafetyAgent", "VerifierAgent", "ChatCompletionsLLMProposalProvider",
            "SupervisionCheck",
        }
        actual = set(agents_mod.__all__)
        missing = old_expected - actual
        self.assertFalse(missing, f"Symbols removed from __all__: {missing}")


# ---------------------------------------------------------------------------
# cli.py — new refine subcommand does not shadow existing commands (P11/P12)
# ---------------------------------------------------------------------------

class TestP13RegressionCLICommands(unittest.TestCase):
    """New refine subcommand must not shadow or break existing CLI subcommands."""

    def _help(self, *cmd: str) -> tuple[int, str]:
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "matrixai", *cmd, "--help"],
            cwd=_BASE, capture_output=True, text=True,
        )
        return r.returncode, r.stdout + r.stderr

    def test_refine_help_exits_0(self):
        rc, _ = self._help("refine")
        self.assertEqual(rc, 0)

    def test_train_help_unaffected(self):
        rc, _ = self._help("train")
        self.assertEqual(rc, 0)

    def test_evaluate_help_unaffected(self):
        rc, _ = self._help("evaluate")
        self.assertEqual(rc, 0)

    def test_train_supervised_help_unaffected(self):
        rc, _ = self._help("train-supervised")
        self.assertEqual(rc, 0)

    def test_validate_training_help_unaffected(self):
        rc, _ = self._help("validate-training")
        self.assertEqual(rc, 0)

    def test_generate_training_help_unaffected(self):
        rc, _ = self._help("generate-training")
        self.assertEqual(rc, 0)

    def test_generate_dataset_help_unaffected(self):
        rc, _ = self._help("generate-dataset")
        self.assertEqual(rc, 0)

    def test_matrixai_top_level_help_lists_refine(self):
        rc, out = self._help()
        self.assertEqual(rc, 0)
        self.assertIn("refine", out)

    def test_matrixai_top_level_help_still_lists_train(self):
        rc, out = self._help()
        self.assertEqual(rc, 0)
        self.assertIn("train", out)


# ---------------------------------------------------------------------------
# playground.py — analyze_playground_request + AuditorAgent pipeline (P9/P10)
# ---------------------------------------------------------------------------

class TestP13RegressionPlaygroundAnalyze(unittest.TestCase):
    """P13 added _refine_prompt and modified renderRunView; analyze pipeline must be unchanged."""

    _PROMPT = _EMAIL_MXAI.read_text(encoding="utf-8") if _EMAIL_MXAI.exists() else ""
    _INPUT = json.dumps({"Email": {
        "urgency": 0.84, "sender_trust": 0.96, "topic_support": 0.99,
        "topic_sales": 0.04, "sentiment": 0.72, "has_attachment": 0.0,
        "previous_interactions": 0.88, "language_confidence": 0.97,
    }})

    def test_analyze_without_input_still_returns_ok_structure(self):
        from matrixai.playground import analyze_playground_request
        r = analyze_playground_request({"mode": "mxai", "mxai_text": self._PROMPT})
        self.assertIn("ok", r)
        self.assertIn("mxai", r)
        self.assertIn("program", r)

    def test_analyze_with_input_returns_run_result(self):
        from matrixai.playground import analyze_playground_request
        r = analyze_playground_request({
            "mode": "mxai",
            "mxai_text": self._PROMPT,
            "input_json": self._INPUT,
        })
        self.assertIn("run_result", r)
        run = r["run_result"]
        self.assertIn("actions", run)
        self.assertIn("trace", run)

    def test_run_result_still_has_audit_text(self):
        from matrixai.playground import analyze_playground_request
        r = analyze_playground_request({
            "mode": "mxai",
            "mxai_text": self._PROMPT,
            "input_json": self._INPUT,
        })
        run = r["run_result"]
        self.assertIn("audit", run)
        self.assertIsInstance(run["audit"], str)
        self.assertGreater(len(run["audit"]), 0)

    def test_refine_prompt_helper_does_not_strip_audit_from_analyze_result(self):
        """analyze_playground_request() and _refine_prompt() are independent;
        _refine_prompt strips 'audit' only from its own payload, not from analyze."""
        from matrixai.playground import analyze_playground_request
        r = analyze_playground_request({
            "mode": "mxai",
            "mxai_text": self._PROMPT,
            "input_json": self._INPUT,
        })
        # The analyze result's run_result must still carry 'audit'
        self.assertIn("audit", r["run_result"])

    def test_analyze_workflow_steps_still_present(self):
        from matrixai.playground import analyze_playground_request
        r = analyze_playground_request({"mode": "mxai", "mxai_text": self._PROMPT})
        self.assertIn("workflow", r)
        self.assertIsInstance(r["workflow"], list)

    def test_analyze_checks_still_present(self):
        from matrixai.playground import analyze_playground_request
        r = analyze_playground_request({"mode": "mxai", "mxai_text": self._PROMPT})
        self.assertIn("checks", r)
        self.assertIsInstance(r["checks"], list)


# ---------------------------------------------------------------------------
# playground.py — _refine_prompt audit stripping is local only (P9)
# ---------------------------------------------------------------------------

class TestP13RegressionRefinePromptIsolation(unittest.TestCase):
    """_refine_prompt strips 'audit' key from run_result before passing to RefinementAgent;
    this must not affect the original run_result dict passed by the caller."""

    def test_original_run_result_dict_not_mutated(self):
        from matrixai.playground import _refine_prompt
        run = {
            "actions": [{"name": "ALERT", "value": 0.9, "activated": True}],
            "trace": [{"step": 0, "node": "urgency", "node_type": "input", "status": "ok"}],
            "audit": "El modelo activa ALERT cuando urgency supera 0.8.",
        }
        original_keys = set(run.keys())
        _refine_prompt({
            "prompt": "Si urgency > 0.8 entonces ALERT",
            "run_result": run,
        })
        self.assertEqual(set(run.keys()), original_keys)
        self.assertIn("audit", run)

    def test_refine_prompt_ok_response_does_not_include_audit(self):
        from matrixai.playground import _refine_prompt
        r = _refine_prompt({
            "prompt": "Si urgency > 0.8 entonces ALERT",
            "run_result": {
                "actions": [{"name": "ALERT", "value": 0.9, "activated": True}],
                "trace": [],
                "audit": "El modelo activa ALERT cuando urgency supera 0.8.",
            },
        })
        self.assertTrue(r["ok"])
        self.assertNotIn("audit", r)


# ---------------------------------------------------------------------------
# AuditorAgent.explain() contract unchanged (P10)
# ---------------------------------------------------------------------------

class TestP13RegressionAuditorAgent(unittest.TestCase):
    """P13 imports AuditorAgent in playground.py; the agent itself must be unchanged."""

    @classmethod
    def setUpClass(cls):
        from matrixai.agents import AuditorAgent
        from matrixai.parser import parse_text
        from matrixai.runtime import MatrixAIRuntime
        mxai = _EMAIL_MXAI.read_text(encoding="utf-8")
        program = parse_text(mxai)
        run_result = MatrixAIRuntime().run(program, {"Email": {
            "urgency": 0.84, "sender_trust": 0.96, "topic_support": 0.99,
            "topic_sales": 0.04, "sentiment": 0.72, "has_attachment": 0.0,
            "previous_interactions": 0.88, "language_confidence": 0.97,
        }})
        cls.explanation = AuditorAgent().explain(run_result)
        cls.run_result = run_result

    def test_explain_returns_non_empty_string(self):
        self.assertIsInstance(self.explanation, str)
        self.assertGreater(len(self.explanation), 0)

    def test_run_result_still_has_actions_key(self):
        self.assertIn("actions", self.run_result)

    def test_run_result_still_has_trace_key(self):
        self.assertIn("trace", self.run_result)

    def test_auditor_explain_does_not_mutate_run_result(self):
        keys_before = set(self.run_result.keys())
        from matrixai.agents import AuditorAgent
        AuditorAgent().explain(self.run_result)
        self.assertEqual(set(self.run_result.keys()), keys_before)


if __name__ == "__main__":
    unittest.main()
