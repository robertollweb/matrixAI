from __future__ import annotations

import json
import unittest
import unittest.mock
from pathlib import Path

from matrixai.agents import ChatCompletionsLLMProposalProvider, PromptSupervisionReport, PromptSupervisor


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "tests" / "snapshots" / "p2"


def _snapshot(name: str) -> dict:
    return json.loads((SNAPSHOT_DIR / name).read_text(encoding="utf-8"))


def _compact_supervision(report: PromptSupervisionReport) -> dict:
    plan = report.plan or {}
    return {
        "accepted": report.accepted,
        "source": report.source,
        "project": plan.get("project", ""),
        "mode": plan.get("mode", ""),
        "vector": plan.get("vector", {}),
        "graph_nodes": plan.get("graph", {}).get("nodes", []),
        "actions": [action.get("name") for action in plan.get("actions", [])],
        "translation_kinds": [
            translation.get("expression_kind")
            for translation in plan.get("mathematical_translations", [])
        ],
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "errors": check.errors,
                "warnings": check.warnings,
            }
            for check in report.checks
        ],
    }


class MatrixAISupervisionSnapshotTest(unittest.TestCase):
    maxDiff = None

    def test_prompt_claim_risk_accepted_snapshot(self) -> None:
        prompt = (ROOT / "examples" / "claim-risk.prompt.txt").read_text(encoding="utf-8")

        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            report = PromptSupervisor().supervise_prompt(prompt)

        self.assertEqual(
            _compact_supervision(report),
            _snapshot("prompt_claim_risk_accepted.json"),
        )

    def test_proposal_email_accepted_snapshot(self) -> None:
        prompt = (ROOT / "examples" / "email-agent.prompt.txt").read_text(encoding="utf-8")
        semantic_text = (ROOT / "examples" / "email-agent.semantic").read_text(encoding="utf-8")

        report = PromptSupervisor().supervise_semantic(
            prompt=prompt,
            semantic_text=semantic_text,
            source="accepted_proposal",
        )

        self.assertEqual(
            _compact_supervision(report),
            _snapshot("proposal_email_accepted.json"),
        )

    def test_proposal_email_rejected_snapshot(self) -> None:
        prompt = (ROOT / "examples" / "email-agent.prompt.txt").read_text(encoding="utf-8")
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
            prompt=prompt,
            semantic_text=semantic_text,
            source="rejected_proposal",
        )

        self.assertEqual(
            _compact_supervision(report),
            _snapshot("proposal_email_rejected.json"),
        )


if __name__ == "__main__":
    unittest.main()
