# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from matrixai.agents.prompt import PromptAgent
from matrixai.agents.prompt_supervisor import PromptSupervisionReport, PromptSupervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_prompt(prompt: str) -> str:
    """Return first 12 hex chars of the SHA-256 of the normalised prompt."""
    normalised = " ".join(prompt.strip().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMHTTPError(RuntimeError):
    """Raised by LLM transports on non-2xx HTTP responses."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"LLM provider returned HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class LLMBudgetExceededError(RuntimeError):
    """Raised when accumulated token usage exceeds the configured budget."""

    def __init__(self, used: int, budget: int) -> None:
        super().__init__(f"Token budget exceeded: used {used}, budget {budget}")
        self.used = used
        self.budget = budget


# ---------------------------------------------------------------------------
# Call trace
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMCallTrace:
    """Audit record for a single HTTP attempt to an LLM provider."""

    provider: str
    model: str
    endpoint: str
    prompt_hash: str
    attempt: int
    latency_ms: float
    http_status: int | None  # None for transport-level errors
    token_usage: int | None  # total_tokens from response, or None
    error: str               # empty string on success

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LLMProposal:
    candidate_id: str
    provider: str
    model: str
    prompt: str
    semantic_text: str
    confidence: float | None = None
    raw_output: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LLMProposalBatch:
    prompt: str
    provider: str
    model: str
    proposals: list[LLMProposal]
    trace: list[dict] = field(default_factory=list)
    call_traces: list[LLMCallTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LLMProposalDecision:
    accepted: bool
    prompt: str
    batch: LLMProposalBatch
    reports: list[PromptSupervisionReport]
    selected_proposal_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def accepted_report(self) -> PromptSupervisionReport | None:
        for proposal, report in zip(self.batch.proposals, self.reports):
            if proposal.candidate_id == self.selected_proposal_id:
                return report
        return None

    def summary(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [
            f"LLMProposalAgent: {status}",
            f"Provider: {self.batch.provider}",
            f"Model: {self.batch.model}",
            f"Candidates: {len(self.reports)}",
        ]
        if self.selected_proposal_id:
            lines.append(f"Selected: {self.selected_proposal_id}")
        for proposal, report in zip(self.batch.proposals, self.reports):
            mark = "accepted" if report.accepted else "rejected"
            lines.append(f"- {proposal.candidate_id}: {mark}")
            if not report.accepted:
                failed_checks = [check.name for check in report.checks if not check.ok]
                if failed_checks:
                    lines.append(f"  Failed checks: {', '.join(failed_checks)}")
        return "\n".join(lines)


class LLMProposalProvider(Protocol):
    provider_name: str
    model_name: str

    def propose(self, prompt: str) -> tuple[list[LLMProposal], list[LLMCallTrace]]:
        ...


LLMHTTPTransport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]


def _urlopen_json_transport(
    url: str, headers: dict[str, str], payload: bytes, timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMHTTPError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM provider request failed: {exc.reason}") from exc
    return json.loads(body)


class DeterministicLLMProposalProvider:
    """Local stand-in for an external LLM provider.

    It produces the same kind of `.semantic` artifact an LLM adapter will later
    return, while keeping the MVP reproducible and offline.
    """

    provider_name = "deterministic-mvp"
    model_name = "prompt-agent-proposer-v0"

    def propose(self, prompt: str) -> tuple[list[LLMProposal], list[LLMCallTrace]]:
        synthesis = PromptAgent().synthesize(prompt)
        proposals = [
            LLMProposal(
                candidate_id="candidate-1",
                provider=self.provider_name,
                model=self.model_name,
                prompt=synthesis.prompt,
                semantic_text=synthesis.semantic_text,
                confidence=0.70,
                raw_output=synthesis.semantic_text,
                notes=[
                    "Generated by deterministic MVP provider",
                    "Must pass PromptSupervisor before acceptance",
                ],
            )
        ]
        return proposals, []


class ChatCompletionsLLMProposalProvider:
    """External LLM provider using a chat-completions compatible wire format."""

    default_endpoint = "https://provider.example/v1/chat/completions"
    # Default endpoints for OpenAI-compatible providers (used when
    # MATRIXAI_LLM_ENDPOINT is not set). DeepSeek and Gemini both expose an
    # OpenAI chat-completions compatible surface, so no dedicated adapter is needed.
    _OPENAI_COMPAT_ENDPOINTS: dict[str, str] = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "chat-completions-compatible": "https://api.openai.com/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "palm": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    }
    _FENCE_RE = re.compile(r"^```(?:semantic|text|matrixai)?\s*(?P<body>.*?)\s*```$", re.DOTALL)

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = default_endpoint,
        provider_name: str = "chat-completions-compatible",
        temperature: float = 0.0,
        candidates: int = 1,
        timeout: float = 30.0,
        max_retries: int = 3,
        token_budget: int = 0,
        retry_delay_s: float = 1.0,
        transport: LLMHTTPTransport | None = None,
    ) -> None:
        if not api_key or api_key == "xxxxxx":
            raise ValueError("ChatCompletionsLLMProposalProvider requires an API key")
        if not model:
            raise ValueError("ChatCompletionsLLMProposalProvider requires a model")
        if candidates < 1:
            raise ValueError("ChatCompletionsLLMProposalProvider requires at least one candidate")
        if max_retries < 1:
            raise ValueError("ChatCompletionsLLMProposalProvider requires max_retries >= 1")
        self.api_key = api_key
        self.endpoint = endpoint
        self.provider_name = provider_name
        self.model_name = model
        self.temperature = temperature
        self.candidates = candidates
        self.timeout = timeout
        self.max_retries = max_retries
        self.token_budget = token_budget
        self.retry_delay_s = retry_delay_s
        self.transport = transport or _urlopen_json_transport

    @classmethod
    def from_env(cls) -> ChatCompletionsLLMProposalProvider:
        # Factory: create provider instance based on provider name
        env = cls._env_with_dotenv()
        provider_name = env.get("MATRIXAI_LLM_PROVIDER_NAME", "chat-completions-compatible")

        # Common params
        api_key = env.get("MATRIXAI_LLM_API_KEY", "")
        model = env.get("MATRIXAI_LLM_MODEL", "external-model-id")
        temperature = float(env.get("MATRIXAI_LLM_TEMPERATURE", "0"))
        candidates = int(env.get("MATRIXAI_LLM_CANDIDATES", "1"))
        timeout = float(env.get("MATRIXAI_LLM_TIMEOUT", "30"))
        max_retries = int(env.get("MATRIXAI_LLM_MAX_RETRIES", "3"))
        token_budget = int(env.get("MATRIXAI_LLM_TOKEN_BUDGET", "0"))
        retry_delay_s = float(env.get("MATRIXAI_LLM_RETRY_DELAY", "1.0"))

        # Anthropic / Claude — dedicated Messages API adapter (different wire format)
        if provider_name in ("anthropic", "claude"):
            from matrixai.agents.llm_adapters import AnthropicLLMProposalProvider

            return AnthropicLLMProposalProvider.from_env(
                api_key=api_key,
                model=model,
                temperature=temperature,
                candidates=candidates,
                timeout=timeout,
                max_retries=max_retries,
                token_budget=token_budget,
                retry_delay_s=retry_delay_s,
            )

        # OpenAI-compatible providers: OpenAI, DeepSeek, and Google Gemini
        # (via its OpenAI-compatible endpoint) all speak the chat-completions
        # wire format, so they reuse this provider directly. The per-provider
        # default endpoint is used when MATRIXAI_LLM_ENDPOINT is not set, so the
        # user only needs to pick a provider + model + key.
        endpoint = env.get("MATRIXAI_LLM_ENDPOINT", "").strip()
        if not endpoint:
            endpoint = cls._OPENAI_COMPAT_ENDPOINTS.get(provider_name, cls.default_endpoint)
        return cls(
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            provider_name=provider_name,
            temperature=temperature,
            candidates=candidates,
            timeout=timeout,
            max_retries=max_retries,
            token_budget=token_budget,
            retry_delay_s=retry_delay_s,
        )

    @classmethod
    def _env_with_dotenv(cls) -> dict[str, str]:
        env = dict(os.environ)
        env_file = Path(env.get("MATRIXAI_LLM_ENV_FILE", ".env"))
        if not env_file.exists():
            return env
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Only set from .env if the key was never explicitly set in the real environment.
            # Respects test overrides (KEY="") and runtime env vars; Docker images should
            # not declare empty-placeholder ENV lines for these keys.
            if key not in os.environ:
                env[key] = value
        return env

    _RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

    def propose(self, prompt: str) -> tuple[list[LLMProposal], list[LLMCallTrace]]:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "n": self.candidates,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        prompt_hash = _hash_prompt(prompt)
        call_traces: list[LLMCallTrace] = []
        total_tokens_used = 0
        last_exc: BaseException | None = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            http_status: int | None = None
            error_msg = ""
            response: dict[str, Any] = {}

            try:
                response = self.transport(self.endpoint, headers, payload_bytes, self.timeout)
                http_status = 200
            except LLMHTTPError as exc:
                http_status = exc.status
                error_msg = str(exc)
                latency_ms = (time.monotonic() - t0) * 1000.0
                call_traces.append(LLMCallTrace(
                    provider=self.provider_name,
                    model=self.model_name,
                    endpoint=self.endpoint,
                    prompt_hash=prompt_hash,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    http_status=http_status,
                    token_usage=None,
                    error=error_msg,
                ))
                last_exc = exc
                if exc.status in self._RETRYABLE_STATUSES and attempt < self.max_retries:
                    if self.retry_delay_s > 0:
                        time.sleep(self.retry_delay_s)
                    continue
                raise
            except RuntimeError as exc:
                error_msg = str(exc)
                latency_ms = (time.monotonic() - t0) * 1000.0
                call_traces.append(LLMCallTrace(
                    provider=self.provider_name,
                    model=self.model_name,
                    endpoint=self.endpoint,
                    prompt_hash=prompt_hash,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    http_status=None,
                    token_usage=None,
                    error=error_msg,
                ))
                raise

            latency_ms = (time.monotonic() - t0) * 1000.0
            usage = response.get("usage") or {}
            token_usage: int | None = usage.get("total_tokens")
            if token_usage is not None:
                total_tokens_used += token_usage

            call_traces.append(LLMCallTrace(
                provider=self.provider_name,
                model=self.model_name,
                endpoint=self.endpoint,
                prompt_hash=prompt_hash,
                attempt=attempt,
                latency_ms=latency_ms,
                http_status=http_status,
                token_usage=token_usage,
                error="",
            ))

            if self.token_budget > 0 and total_tokens_used > self.token_budget:
                raise LLMBudgetExceededError(total_tokens_used, self.token_budget)

            proposals: list[LLMProposal] = []
            for index, choice in enumerate(response.get("choices", []), start=1):
                raw_output = self._choice_text(choice)
                semantic_text = self._extract_semantic_text(raw_output)
                if not semantic_text:
                    continue
                proposals.append(
                    LLMProposal(
                        candidate_id=f"candidate-{index}",
                        provider=self.provider_name,
                        model=self.model_name,
                        prompt=prompt,
                        semantic_text=semantic_text,
                        raw_output=raw_output,
                        notes=["Generated by external chat-completions-compatible provider"],
                    )
                )
            if not proposals:
                raise RuntimeError("LLM provider returned no semantic proposals")
            return proposals, call_traces

        # Should not be reached, but satisfies type checker
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM provider: all retry attempts failed")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Single-turn completion with a custom system prompt. No retries."""
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "n": 1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.transport(self.endpoint, headers, json.dumps(payload).encode("utf-8"), self.timeout)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LLM returned no choices")
        return self._choice_text(choices[0])

    def _choice_text(self, choice: dict[str, Any]) -> str:
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in parts if part).strip()
        return ""

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


class LLMProposalAgent:
    """Generate candidate semantic artifacts and delegate acceptance to PromptSupervisor."""

    def __init__(
        self,
        provider: LLMProposalProvider | None = None,
        supervisor: PromptSupervisor | None = None,
    ) -> None:
        self.provider = provider or DeterministicLLMProposalProvider()
        self.supervisor = supervisor or PromptSupervisor()

    def propose(self, prompt: str) -> LLMProposalBatch:
        clean_prompt = " ".join(prompt.strip().split())
        if not clean_prompt:
            raise ValueError("LLMProposalAgent requires a non-empty prompt")
        proposals, call_traces = self.provider.propose(clean_prompt)
        return LLMProposalBatch(
            prompt=clean_prompt,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            proposals=proposals,
            trace=[
                {
                    "event": "llm_proposals_generated",
                    "provider": self.provider.provider_name,
                    "model": self.provider.model_name,
                    "candidates": len(proposals),
                }
            ],
            call_traces=call_traces,
        )

    def propose_and_supervise(
        self, prompt: str, *, max_candidates: int | None = None, exhaust_all: bool = False
    ) -> LLMProposalDecision:
        batch = self.propose(prompt)
        proposals = batch.proposals
        if max_candidates is not None:
            proposals = proposals[:max_candidates]
        reports: list[PromptSupervisionReport] = []
        selected_proposal_id = ""

        for proposal in proposals:
            report = self.supervisor.supervise_semantic(
                prompt=batch.prompt,
                semantic_text=proposal.semantic_text,
                source=f"llm:{proposal.provider}:{proposal.candidate_id}",
            )
            reports.append(report)
            if report.accepted and not selected_proposal_id:
                selected_proposal_id = proposal.candidate_id
                if not exhaust_all:
                    break

        return LLMProposalDecision(
            accepted=bool(selected_proposal_id),
            prompt=batch.prompt,
            batch=batch,
            reports=reports,
            selected_proposal_id=selected_proposal_id,
        )
