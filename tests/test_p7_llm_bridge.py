from __future__ import annotations

import unittest
import unittest.mock
from typing import Any

from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider
from matrixai.agents.prompt_supervisor import PromptSupervisor
from matrixai.agents.safety import SafetyAgent
from matrixai.parser import parse_text
from matrixai.playground import DEFAULT_PROMPT, analyze_playground_request


# ---------------------------------------------------------------------------
# Minimal valid .semantic that passes the full supervision chain
# (architect_plan -> planner_verifier -> math -> parser -> verifier ->
#  safety_agent -> python_compiler)
# ---------------------------------------------------------------------------
_VALID_SEMANTIC = """\
PROJECT TestClassifier
INTENT classify test signals above 80% confidence
MODE classification
ENTITY Signal

FIELDS Signal
  score
  confidence_level
END

GOAL classify_incoming_email
CONSTRAINT confidence > 0.80
ACTION_THRESHOLD 0.90

ACTION SendAlert
  POLICY simulate_only
  CALL simulated.alerts.send
END"""

# ---------------------------------------------------------------------------
# .semantic whose ACTION carries POLICY execute + non-simulated call.
# The supervision chain must reject this regardless of which check catches it.
# ---------------------------------------------------------------------------
_UNSAFE_SEMANTIC = """\
PROJECT UnsafeAgent
INTENT execute external commands
MODE classification
ENTITY Signal

FIELDS Signal
  score
  threat_level
END

GOAL classify_incoming_email
CONSTRAINT confidence > 0.80
ACTION_THRESHOLD 0.70

ACTION ExecAction
  POLICY execute
  CALL real.external.system
END"""

# ---------------------------------------------------------------------------
# .mxai with POLICY execute that can be fed directly to parse_text so that
# SafetyAgent is the terminal check in the chain (not planner_verifier).
# ---------------------------------------------------------------------------
_UNSAFE_MXAI = """\
PROJECT UnsafeEmail

VECTOR Email[2]
  urgency
  sender_trust
END

FUNCTION ReplyActivation
  A = sigmoid(20 * (urgency - 0.5))
END

GRAPH
  Email -> ReplyActivation -> DraftReply
END

ACTION DraftReply
  WHEN ReplyActivation > 0.90
  POLICY execute
  CALL real.email.send
END

AUDIT
  EXPLAIN Email -> ReplyActivation -> DraftReply
END
"""


def _make_fake_transport(semantic_text: str):
    """Return a transport callable that wraps *semantic_text* in an
    chat-completions-compatible chat-completions response. No real HTTP request is made."""

    def transport(
        url: str, headers: dict[str, str], payload_bytes: bytes, timeout: float
    ) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": semantic_text}}],
            "usage": {"total_tokens": 120},
        }

    return transport


class TestP7LLMBridge(unittest.TestCase):
    # ------------------------------------------------------------------
    # Test 1 – transport fake
    # ------------------------------------------------------------------
    def test_fake_transport_llm_branch_accepted(self) -> None:
        """transport fake: supervise_prompt uses the LLM branch when from_env
        returns a provider backed by a fake transport; supervision_source is
        'llm' and call_traces_summary records the call."""
        provider = ChatCompletionsLLMProposalProvider(
            api_key="test-key",
            model="chat-test",
            transport=_make_fake_transport(_VALID_SEMANTIC),
            max_retries=1,
            retry_delay_s=0.0,
        )

        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            return_value=provider,
        ):
            report = PromptSupervisor().supervise_prompt("classify test signals")

        self.assertEqual(report.supervision_source, "llm")
        self.assertEqual(report.llm_provider, "chat-completions-compatible")
        self.assertEqual(report.llm_model, "chat-test")
        self.assertTrue(report.accepted)
        self.assertEqual(len(report.call_traces_summary), 1)
        trace = report.call_traces_summary[0]
        self.assertEqual(trace["http_status"], 200)
        self.assertEqual(trace["token_usage"], 120)
        self.assertEqual(trace["error"], "")
        self.assertEqual(trace["attempt"], 1)

    # ------------------------------------------------------------------
    # Test 2 – rama determinista
    # ------------------------------------------------------------------
    def test_deterministic_branch_when_no_api_key(self) -> None:
        """rama determinista: supervise_prompt falls back to the deterministic
        path when from_env raises ValueError (no API key configured).
        supervision_source must be 'deterministic' and fallback_reason must
        record the cause."""
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            report = PromptSupervisor().supervise_prompt(
                "Classify emails above 95% confidence"
            )

        self.assertEqual(report.supervision_source, "deterministic")
        self.assertEqual(report.fallback_reason, "no_api_key")
        self.assertTrue(report.accepted)
        self.assertEqual(report.llm_provider, "")
        self.assertEqual(report.call_traces_summary, [])

    # ------------------------------------------------------------------
    # Test 3 – SafetyAgent bloqueando LLM
    # ------------------------------------------------------------------
    def test_safety_agent_blocks_unsafe_llm_proposal(self) -> None:
        """SafetyAgent bloqueando LLM: a fake transport that returns a semantic
        with POLICY execute is rejected by the supervision chain (accepted=False).

        Two complementary assertions:
        1. supervise_semantic rejects the unsafe semantic – the planner_verifier
           check is ok=False because POLICY execute and non-simulated calls
           violate the MVP guardrails at the planning stage.
        2. SafetyAgent directly blocks a parsed .mxai carrying POLICY execute,
           confirming that even if an unsafe program somehow reached the parser
           step, SafetyAgent would still catch it.
        """
        # --- Part 1: full chain rejects POLICY execute via supervise_semantic ---
        report = PromptSupervisor().supervise_semantic(
            prompt="execute external action",
            semantic_text=_UNSAFE_SEMANTIC,
            source="llm:fake-provider",
        )

        self.assertFalse(report.accepted)
        check_ok = {c.name: c.ok for c in report.checks}
        self.assertIn("planner_verifier", check_ok)
        self.assertFalse(check_ok["planner_verifier"])

        # --- Part 2: SafetyAgent specifically blocks a parsed program with POLICY execute ---
        program = parse_text(_UNSAFE_MXAI)
        safety_errors = SafetyAgent().review(program)

        self.assertTrue(len(safety_errors) > 0)
        combined = " ".join(safety_errors).lower()
        self.assertIn("execute", combined)

    # ------------------------------------------------------------------
    # Test 4 – playground llm_mode
    # ------------------------------------------------------------------
    def test_playground_exposes_llm_mode_field(self) -> None:
        """playground llm_mode: analyze_playground_request (mode=prompt) returns
        an 'llm_mode' dict computed by the backend.  When no LLM provider is
        configured the dict must have active=False and model=''."""
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            result = analyze_playground_request(
                {"mode": "prompt", "prompt": DEFAULT_PROMPT}
            )

        self.assertIn("llm_mode", result)
        llm_mode = result["llm_mode"]
        self.assertIsInstance(llm_mode, dict)
        self.assertIn("active", llm_mode)
        self.assertFalse(llm_mode["active"])
        self.assertEqual(llm_mode.get("model", ""), "")

    # ------------------------------------------------------------------
    # Test 5 – playground supervision_source
    # ------------------------------------------------------------------
    def test_playground_supervision_source_field(self) -> None:
        """playground supervision_source: the playground result carries a
        'supervision_source' field.  It must be 'deterministic' when no LLM
        provider is configured (mode=prompt), and must also be present for
        mode=semantic."""
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            prompt_result = analyze_playground_request(
                {"mode": "prompt", "prompt": DEFAULT_PROMPT}
            )

        self.assertIn("supervision_source", prompt_result)
        self.assertEqual(prompt_result["supervision_source"], "deterministic")

        # mode=semantic does not call from_env; supervision_source still present
        semantic_result = analyze_playground_request(
            {
                "mode": "semantic",
                "semantic_text": _VALID_SEMANTIC,
                "prompt": "classify test signals",
            }
        )
        self.assertIn("supervision_source", semantic_result)
        self.assertEqual(semantic_result["supervision_source"], "deterministic")


if __name__ == "__main__":
    unittest.main()
