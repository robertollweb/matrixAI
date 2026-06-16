# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Provider-specific LLM adapters that do NOT share the OpenAI chat-completions
wire format.

Only Anthropic (Claude) needs a dedicated adapter: it uses the Messages API
(`POST /v1/messages`) with `x-api-key` + `anthropic-version` headers, a top-level
`system` field, a required `max_tokens`, and a `content[]` response — none of
which match the chat-completions shape.

DeepSeek and Google Gemini are reachable through OpenAI-compatible endpoints, so
they reuse :class:`ChatCompletionsLLMProposalProvider` directly (wired in
``ChatCompletionsLLMProposalProvider.from_env``) — no adapter required.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .llm_proposal import (
    LLMBudgetExceededError,
    LLMCallTrace,
    LLMHTTPError,
    LLMHTTPTransport,
    LLMProposal,
    _hash_prompt,
    _urlopen_json_transport,
)

# Default Messages API version pin (see docs: required header).
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicLLMProposalProvider:
    """Claude provider using the Anthropic Messages API wire format.

    Mirrors the public surface of
    :class:`~matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider`
    (``propose`` + ``complete`` + the ``provider_name`` / ``model_name`` /
    ``endpoint`` / ``api_key`` attributes that ``_detect_llm_mode`` and the
    Studio settings UI read), so the rest of the codebase is provider-agnostic.

    Wire-format differences from chat-completions, all handled here:

    * endpoint ``https://api.anthropic.com/v1/messages``
    * auth via ``x-api-key`` (not ``Authorization: Bearer``) + the required
      ``anthropic-version`` header
    * the system prompt is a top-level ``system`` field, not a message
    * ``max_tokens`` is required
    * the assistant text lives in ``content[].text`` (blocks of ``type==text``)
    * ``temperature`` is **not** sent — recent Claude models (opus-4-8/4.7,
      fable-5) reject sampling parameters with a 400.
    """

    provider_name = "anthropic"
    model_name = "claude-opus-4-8"
    default_endpoint = "https://api.anthropic.com/v1/messages"

    _FENCE_RE = re.compile(
        r"^```(?:semantic|text|matrixai)?\s*(?P<body>.*?)\s*```$", re.DOTALL
    )
    _RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504, 529})

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str | None = None,
        provider_name: str = "anthropic",
        temperature: float = 0.0,
        candidates: int = 1,
        timeout: float = 30.0,
        max_retries: int = 3,
        token_budget: int = 0,
        retry_delay_s: float = 1.0,
        max_tokens: int = 4096,
        anthropic_version: str = _ANTHROPIC_VERSION,
        transport: LLMHTTPTransport | None = None,
    ) -> None:
        if not api_key or api_key == "xxxxxx":
            raise ValueError("AnthropicLLMProposalProvider requires an API key")
        if not model:
            raise ValueError("AnthropicLLMProposalProvider requires a model")
        if max_retries < 1:
            raise ValueError("AnthropicLLMProposalProvider requires max_retries >= 1")
        if max_tokens < 1:
            raise ValueError("AnthropicLLMProposalProvider requires max_tokens >= 1")
        self.api_key = api_key
        self.model_name = model
        self.endpoint = endpoint or self.default_endpoint
        self.provider_name = provider_name
        # temperature is retained for interface parity but intentionally not sent.
        self.temperature = temperature
        self.candidates = candidates
        self.timeout = timeout
        self.max_retries = max_retries
        self.token_budget = token_budget
        self.retry_delay_s = retry_delay_s
        self.max_tokens = max_tokens
        self.anthropic_version = anthropic_version
        self.transport = transport or _urlopen_json_transport

    @classmethod
    def from_env(cls, **kwargs: Any) -> "AnthropicLLMProposalProvider":
        # Reuse the chat-completions env loader so the "only override if the key
        # was not set in the real environment" semantics stay consistent.
        from .llm_proposal import ChatCompletionsLLMProposalProvider as _CC

        env = _CC._env_with_dotenv()

        api_key = kwargs.pop("api_key", None) or env.get("MATRIXAI_LLM_API_KEY", "")
        model = kwargs.pop("model", None) or env.get("MATRIXAI_LLM_MODEL", cls.model_name)
        provider_name = env.get("MATRIXAI_LLM_PROVIDER_NAME", "anthropic")
        max_tokens = int(env.get("MATRIXAI_LLM_MAX_TOKENS", "4096"))
        anthropic_version = env.get("MATRIXAI_LLM_ANTHROPIC_VERSION", _ANTHROPIC_VERSION)

        # Honour an explicit endpoint only when it is not a leftover chat-completions
        # URL (a common case: the user switches provider but keeps the OpenAI endpoint
        # configured). Otherwise fall back to the Anthropic Messages endpoint so we
        # never POST a Messages payload to an OpenAI URL.
        endpoint = env.get("MATRIXAI_LLM_ENDPOINT", "").strip()
        if not endpoint or "chat/completions" in endpoint or endpoint == _CC.default_endpoint:
            endpoint = cls.default_endpoint

        return cls(
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            provider_name=provider_name,
            max_tokens=max_tokens,
            anthropic_version=anthropic_version,
            **kwargs,
        )

    # -- wire helpers --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        # No `temperature`: recent Claude models reject sampling params (400).
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

    @staticmethod
    def _content_text(response: dict[str, Any]) -> str:
        """Extract the assistant text from a Messages API response.

        The response carries ``content`` as a list of typed blocks; we join the
        text of every ``type == "text"`` block.
        """
        content = response.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part).strip()
        return ""

    @staticmethod
    def _total_tokens(response: dict[str, Any]) -> int | None:
        usage = response.get("usage") or {}
        inp = usage.get("input_tokens")
        out = usage.get("output_tokens")
        if inp is None and out is None:
            return None
        return int(inp or 0) + int(out or 0)

    def _extract_semantic_text(self, text: str) -> str:
        stripped = text.strip()
        match = self._FENCE_RE.match(stripped)
        if match:
            stripped = match.group("body").strip()
        project_index = stripped.find("PROJECT ")
        if project_index < 0:
            return ""
        return stripped[project_index:]

    def _system_prompt(self) -> str:
        return """You generate MatrixAI .semantic proposals only.
Return plain text, no Markdown, no JSON, no explanations.
Use this format:
PROJECT <Name>
INTENT <one sentence>
MODE classification|risk
ENTITY <Entity>

FIELDS <Entity>
  <field_one>
  <field_two>
END

GOAL <goal_name>
CONSTRAINT <confidence|risk> > <0-1 threshold>
ACTION_THRESHOLD <0-1 threshold>

RULES
  if <metric> > <threshold> then <ActionName>
END

ACTION <ActionName>
  POLICY simulate_only
  CALL simulated.<domain>.<action>
END

Never use POLICY execute. Never use calls outside simulated.*."""

    # -- public API ----------------------------------------------------------

    def propose(self, prompt: str) -> tuple[list[LLMProposal], list[LLMCallTrace]]:
        payload_bytes = json.dumps(self._payload(self._system_prompt(), prompt)).encode("utf-8")
        headers = self._headers()
        prompt_hash = _hash_prompt(prompt)
        call_traces: list[LLMCallTrace] = []
        total_tokens_used = 0
        last_exc: BaseException | None = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                response = self.transport(self.endpoint, headers, payload_bytes, self.timeout)
            except LLMHTTPError as exc:
                latency_ms = (time.monotonic() - t0) * 1000.0
                call_traces.append(LLMCallTrace(
                    provider=self.provider_name, model=self.model_name, endpoint=self.endpoint,
                    prompt_hash=prompt_hash, attempt=attempt, latency_ms=latency_ms,
                    http_status=exc.status, token_usage=None, error=str(exc),
                ))
                last_exc = exc
                if exc.status in self._RETRYABLE_STATUSES and attempt < self.max_retries:
                    if self.retry_delay_s > 0:
                        time.sleep(self.retry_delay_s)
                    continue
                raise
            except RuntimeError as exc:
                latency_ms = (time.monotonic() - t0) * 1000.0
                call_traces.append(LLMCallTrace(
                    provider=self.provider_name, model=self.model_name, endpoint=self.endpoint,
                    prompt_hash=prompt_hash, attempt=attempt, latency_ms=latency_ms,
                    http_status=None, token_usage=None, error=str(exc),
                ))
                raise

            latency_ms = (time.monotonic() - t0) * 1000.0
            token_usage = self._total_tokens(response)
            if token_usage is not None:
                total_tokens_used += token_usage

            call_traces.append(LLMCallTrace(
                provider=self.provider_name, model=self.model_name, endpoint=self.endpoint,
                prompt_hash=prompt_hash, attempt=attempt, latency_ms=latency_ms,
                http_status=200, token_usage=token_usage, error="",
            ))

            if self.token_budget > 0 and total_tokens_used > self.token_budget:
                raise LLMBudgetExceededError(total_tokens_used, self.token_budget)

            raw_output = self._content_text(response)
            semantic_text = self._extract_semantic_text(raw_output)
            if not semantic_text:
                raise RuntimeError("LLM provider returned no semantic proposals")
            proposals = [LLMProposal(
                candidate_id="candidate-1",
                provider=self.provider_name,
                model=self.model_name,
                prompt=prompt,
                semantic_text=semantic_text,
                raw_output=raw_output,
                notes=["Generated by Anthropic Messages API provider"],
            )]
            return proposals, call_traces

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM provider: all retry attempts failed")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Single-turn completion with a custom system prompt. No retries."""
        payload_bytes = json.dumps(self._payload(system_prompt, user_prompt)).encode("utf-8")
        response = self.transport(self.endpoint, self._headers(), payload_bytes, self.timeout)
        text = self._content_text(response)
        if not text:
            raise RuntimeError("LLM returned no content")
        return text
