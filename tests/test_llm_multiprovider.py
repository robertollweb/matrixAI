"""Multi-provider LLM support.

Covers the Anthropic Messages API adapter (wire format, headers, parsing,
``complete``, retries, token budget) and the ``from_env`` routing that sends
OpenAI / DeepSeek / Gemini through the chat-completions provider with the right
default endpoint, and Anthropic through the dedicated adapter.

All transports are injected; no real HTTP request is made.
"""

from __future__ import annotations

import json
import os
import unittest
import unittest.mock
from typing import Any

from matrixai.agents.llm_adapters import AnthropicLLMProposalProvider
from matrixai.agents.llm_proposal import (
    ChatCompletionsLLMProposalProvider,
    LLMBudgetExceededError,
    LLMHTTPError,
)

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


def _anthropic_messages_response(text: str, *, input_tokens: int = 30, output_tokens: int = 70):
    """Build a transport returning a Messages-API-shaped response.

    Also captures the last (url, headers, payload) the provider sent so tests
    can assert on the wire format.
    """
    captured: dict[str, Any] = {}

    def transport(url: str, headers: dict[str, str], payload_bytes: bytes, timeout: float) -> dict[str, Any]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json.loads(payload_bytes.decode("utf-8"))
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

    transport.captured = captured  # type: ignore[attr-defined]
    return transport


class TestAnthropicAdapterWireFormat(unittest.TestCase):
    def test_propose_uses_messages_endpoint_and_headers(self) -> None:
        transport = _anthropic_messages_response(_VALID_SEMANTIC)
        provider = AnthropicLLMProposalProvider(
            api_key="sk-ant-test", model="claude-opus-4-8",
            transport=transport, max_retries=1, retry_delay_s=0.0,
        )
        proposals, traces = provider.propose("classify signals")

        cap = transport.captured
        self.assertEqual(cap["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(cap["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(cap["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", cap["headers"])
        # system is top-level, not a message; max_tokens required; no temperature.
        self.assertIn("system", cap["payload"])
        self.assertIn("max_tokens", cap["payload"])
        self.assertNotIn("temperature", cap["payload"])
        self.assertEqual(cap["payload"]["messages"], [{"role": "user", "content": "classify signals"}])

        self.assertEqual(len(proposals), 1)
        self.assertTrue(proposals[0].semantic_text.startswith("PROJECT "))
        # token usage = input + output
        self.assertEqual(traces[0].token_usage, 100)

    def test_complete_extracts_content_text(self) -> None:
        transport = _anthropic_messages_response("FIELDS: age, income\nNAME: Scorer")
        provider = AnthropicLLMProposalProvider(
            api_key="sk-ant-test", model="claude-opus-4-8", transport=transport,
        )
        out = provider.complete("system instructions", "design a model")
        self.assertIn("FIELDS: age, income", out)
        self.assertEqual(transport.captured["payload"]["system"], "system instructions")

    def test_multi_block_content_is_joined(self) -> None:
        def transport(url, headers, payload_bytes, timeout):
            return {"content": [
                {"type": "text", "text": "PROJECT A"},
                {"type": "text", "text": "INTENT x"},
            ], "usage": {"input_tokens": 1, "output_tokens": 1}}

        provider = AnthropicLLMProposalProvider(
            api_key="k", model="claude-opus-4-8", transport=transport, max_retries=1,
        )
        out = provider.complete("s", "u")
        self.assertEqual(out, "PROJECT A\nINTENT x")

    def test_empty_api_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            AnthropicLLMProposalProvider(api_key="", model="claude-opus-4-8")

    def test_no_semantic_text_raises(self) -> None:
        transport = _anthropic_messages_response("no project marker here")
        provider = AnthropicLLMProposalProvider(
            api_key="k", model="claude-opus-4-8", transport=transport, max_retries=1,
        )
        with self.assertRaises(RuntimeError):
            provider.propose("x")

    def test_token_budget_enforced(self) -> None:
        transport = _anthropic_messages_response(_VALID_SEMANTIC, input_tokens=600, output_tokens=600)
        provider = AnthropicLLMProposalProvider(
            api_key="k", model="claude-opus-4-8", transport=transport,
            max_retries=1, token_budget=500,
        )
        with self.assertRaises(LLMBudgetExceededError):
            provider.propose("x")

    def test_retries_on_retryable_status(self) -> None:
        calls = {"n": 0}

        def transport(url, headers, payload_bytes, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMHTTPError(529, "overloaded")
            return {"content": [{"type": "text", "text": _VALID_SEMANTIC}],
                    "usage": {"input_tokens": 1, "output_tokens": 1}}

        provider = AnthropicLLMProposalProvider(
            api_key="k", model="claude-opus-4-8", transport=transport,
            max_retries=3, retry_delay_s=0.0,
        )
        proposals, traces = provider.propose("x")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(len(traces), 2)  # one failed + one ok


class TestProviderRouting(unittest.TestCase):
    """from_env dispatches to the right provider/endpoint by provider name."""

    def _from_env_with(self, env: dict[str, str]):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("MATRIXAI_LLM")}
        clean.update(env)
        # MATRIXAI_LLM_ENV_FILE points at a non-existent file so no .env is read.
        clean["MATRIXAI_LLM_ENV_FILE"] = "/nonexistent/.env"
        with unittest.mock.patch.dict(os.environ, clean, clear=True):
            return ChatCompletionsLLMProposalProvider.from_env()

    def test_anthropic_routes_to_adapter(self) -> None:
        p = self._from_env_with({
            "MATRIXAI_LLM_PROVIDER_NAME": "anthropic",
            "MATRIXAI_LLM_API_KEY": "sk-ant-x",
            "MATRIXAI_LLM_MODEL": "claude-opus-4-8",
        })
        self.assertIsInstance(p, AnthropicLLMProposalProvider)
        self.assertEqual(p.endpoint, "https://api.anthropic.com/v1/messages")

    def test_anthropic_ignores_leftover_openai_endpoint(self) -> None:
        p = self._from_env_with({
            "MATRIXAI_LLM_PROVIDER_NAME": "claude",
            "MATRIXAI_LLM_API_KEY": "sk-ant-x",
            "MATRIXAI_LLM_MODEL": "claude-opus-4-8",
            "MATRIXAI_LLM_ENDPOINT": "https://api.openai.com/v1/chat/completions",
        })
        self.assertIsInstance(p, AnthropicLLMProposalProvider)
        self.assertEqual(p.endpoint, "https://api.anthropic.com/v1/messages")

    def test_deepseek_routes_to_chat_completions_with_default_endpoint(self) -> None:
        p = self._from_env_with({
            "MATRIXAI_LLM_PROVIDER_NAME": "deepseek",
            "MATRIXAI_LLM_API_KEY": "sk-ds-x",
            "MATRIXAI_LLM_MODEL": "deepseek-chat",
        })
        self.assertIsInstance(p, ChatCompletionsLLMProposalProvider)
        self.assertEqual(p.endpoint, "https://api.deepseek.com/chat/completions")

    def test_gemini_routes_to_openai_compatible_endpoint(self) -> None:
        p = self._from_env_with({
            "MATRIXAI_LLM_PROVIDER_NAME": "gemini",
            "MATRIXAI_LLM_API_KEY": "g-x",
            "MATRIXAI_LLM_MODEL": "gemini-2.0-flash",
        })
        self.assertIsInstance(p, ChatCompletionsLLMProposalProvider)
        self.assertEqual(
            p.endpoint,
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        )

    def test_openai_default_endpoint(self) -> None:
        p = self._from_env_with({
            "MATRIXAI_LLM_PROVIDER_NAME": "openai",
            "MATRIXAI_LLM_API_KEY": "sk-x",
            "MATRIXAI_LLM_MODEL": "gpt-4o-mini",
        })
        self.assertIsInstance(p, ChatCompletionsLLMProposalProvider)
        self.assertEqual(p.endpoint, "https://api.openai.com/v1/chat/completions")

    def test_explicit_endpoint_overrides_default(self) -> None:
        p = self._from_env_with({
            "MATRIXAI_LLM_PROVIDER_NAME": "openai",
            "MATRIXAI_LLM_API_KEY": "sk-x",
            "MATRIXAI_LLM_MODEL": "gpt-4o-mini",
            "MATRIXAI_LLM_ENDPOINT": "https://my-proxy.local/v1/chat/completions",
        })
        self.assertEqual(p.endpoint, "https://my-proxy.local/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
