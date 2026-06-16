#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matrixai.agents.prompt_supervisor import PromptSupervisor


def mask_key(s: str) -> str:
    if not s:
        return "(empty)"
    if len(s) <= 8:
        return s[0] + "..." + s[-1]
    return s[:4] + "..." + s[-4:]


def main() -> int:
    sup = PromptSupervisor()
    prompt = "Escribe un proyecto de ejemplo: nombre proyecto prueba"
    report = sup.supervise_prompt(prompt)
    print("supervision_source:", report.supervision_source)
    print("fallback_reason:", report.fallback_reason)
    print("llm_provider:", report.llm_provider)
    print("llm_model:", report.llm_model)
    print("call_traces_summary:")
    for t in report.call_traces_summary:
        print(" ", t)
    print("accepted:", report.accepted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
