#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure package import works when running from scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider


def mask(s: str) -> str:
    if not s:
        return "(empty)"
    if len(s) <= 8:
        return s[0] + "..." + s[-1]
    return s[:4] + "..." + s[-4:]


def main() -> int:
    provider = ChatCompletionsLLMProposalProvider.from_env()
    print(f"Provider: {provider.provider_name}")
    print(f"Model: {provider.model_name}")
    print(f"Endpoint: {provider.endpoint}")
    print(f"API key: {mask(provider.api_key)}")

    try:
        # Basic connectivity / sanity call: single-turn completion
        out = provider.complete("System: connectivity check", "Hola, por favor responde brevemente:")
        print("LLM returned (truncated):")
        print(out[:1000])
        return 0
    except Exception as exc:
        print("LLM call failed:", type(exc).__name__, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
