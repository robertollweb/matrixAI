from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from matrixai.agents import SafetyAgent
from matrixai.parser import parse_file, parse_text
from matrixai.sandbox import SandboxPolicy, capabilities_for_call


ROOT = Path(__file__).resolve().parents[1]


class MatrixAISandboxTest(unittest.TestCase):
    def test_capabilities_for_simulated_domain_calls(self) -> None:
        self.assertEqual(
            capabilities_for_call("simulated.email.draft"),
            ["simulated", "email"],
        )
        self.assertEqual(
            capabilities_for_call("simulated.pharmacy.dispense"),
            ["simulated", "pharmacy"],
        )
        self.assertEqual(
            capabilities_for_call("simulated.nurse_station.alert"),
            ["simulated", "notification"],
        )

    def test_sandbox_allows_mvp_simulated_action(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.mxai")

        report = SandboxPolicy.mvp_simulate_only().review(program)

        self.assertTrue(report.ok)
        self.assertEqual(report.decisions[0].action, "DraftReply")
        self.assertIn("simulated", report.decisions[0].capabilities)
        self.assertEqual(report.messages(), [])

    def test_sandbox_blocks_external_action(self) -> None:
        program = parse_text(
            """PROJECT UnsafeEmail

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
  CALL email.draft
END

AUDIT
  EXPLAIN Email -> ReplyActivation -> DraftReply
END
"""
        )

        report = SandboxPolicy.mvp_simulate_only().review(program)
        warnings = SafetyAgent().review(program)

        self.assertFalse(report.ok)
        self.assertFalse(report.decisions[0].allowed)
        self.assertIn("external", report.decisions[0].capabilities)
        self.assertTrue(any("simulate_only" in message for message in warnings))
        self.assertTrue(any("simulated actions" in message for message in warnings))

    def test_cli_permissions_json_reports_capabilities(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", "permissions", "examples/email-agent.mxai", "--json"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["policy"], "mvp_simulate_only")
        self.assertEqual(payload["decisions"][0]["action"], "DraftReply")
        self.assertIn("email", payload["decisions"][0]["capabilities"])


if __name__ == "__main__":
    unittest.main()
