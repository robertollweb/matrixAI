from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.parser import parse_file
from matrixai.tooling import diagnose_runtime_compiler, format_source, graph_source, lint_source


ROOT = Path(__file__).resolve().parents[1]


class MatrixAILanguageToolingTest(unittest.TestCase):
    def test_format_semantic_canonicalizes_blocks(self) -> None:
        source = """PROJECT EmailAgent
INTENT Classify incoming emails.
MODE classification
ENTITY Email
FIELDS Email
urgency
sender_trust
END
GOAL minimize_false_replies
CONSTRAINT confidence > 0.95
ACTION_THRESHOLD 0.9
ACTION DraftReply
CALL simulated.email.draft
POLICY simulate_only
END
"""

        formatted = format_source(source, "semantic")

        self.assertIn("FIELDS Email\n  urgency\n  sender_trust\nEND", formatted)
        self.assertIn("ACTION_THRESHOLD 0.90", formatted)
        self.assertIn("ACTION DraftReply\n  POLICY simulate_only\n  CALL simulated.email.draft\nEND", formatted)

    def test_lint_semantic_reports_planner_errors(self) -> None:
        source = """PROJECT UnsafeEmail
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

        report = lint_source(source, "semantic")

        self.assertFalse(report.ok)
        self.assertTrue(any("simulate_only" in item.message for item in report.diagnostics))
        self.assertTrue(any(item.source == "planner" for item in report.diagnostics))

    def test_format_mxai_preserves_p2_type_annotations(self) -> None:
        source = """PROJECT TypedScores
VECTOR X[2]
raw: Score[0,10]
confidence: Probability
END
FUNCTION Normed
normed: Probability = scale(raw, 0, 10)
END
GRAPH
X -> Normed
END
AUDIT
EXPLAIN X -> Normed
END
"""

        formatted = format_source(source, "mxai")

        self.assertIn("  raw: Score[0, 10]", formatted)
        self.assertIn("  confidence: Probability", formatted)
        self.assertIn("  normed: Probability = scale(raw, 0, 10)", formatted)

    def test_format_mxai_preserves_explicit_params(self) -> None:
        source = """PROJECT ExplicitParams
VECTOR X[2]
a: Score
b: Score
END
PARAM W1 Tensor[3,2]
INIT deterministic_uniform
TRAINABLE true
END
FUNCTION Score
scores = softmax(W1 * X + b1)
END
GRAPH
X -> Score
END
AUDIT
EXPLAIN X -> Score
END
"""

        formatted = format_source(source, "mxai")

        self.assertIn("PARAM W1 Tensor[3, 2]", formatted)
        self.assertIn("  TRAINABLE true", formatted)
        self.assertIn("  INIT deterministic_uniform", formatted)

    def test_lint_mxai_reports_verifier_errors(self) -> None:
        source = """PROJECT Broken

VECTOR X[2]
  a
  b
END

FUNCTION Score
  score = sigmoid(W1 * X + b1)
END

GRAPH
  X -> Missing -> Score
END

AUDIT
  EXPLAIN X -> Missing -> Score
END
"""

        report = lint_source(source, "mxai")

        self.assertFalse(report.ok)
        self.assertTrue(any("undeclared node 'Missing'" in item.message for item in report.diagnostics))

    def test_cli_lint_json_reports_ok_for_example(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "matrixai", "lint", "examples/email-agent.mxai", "--json"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["language"], "mxai")

    def test_cli_format_check_detects_unformatted_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "unformatted.mxai"
            path.write_text(
                """PROJECT Demo
VECTOR X[2]
a
b
END
FUNCTION Score
score = sigmoid(W1 * X + b1)
END
GRAPH
X -> Score
END
AUDIT
EXPLAIN X -> Score
END
""",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "-m", "matrixai", "format", str(path), "--check"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Would reformat", result.stdout)

    def test_graph_mermaid_renders_nodes_and_edges(self) -> None:
        source = (ROOT / "examples" / "email-agent.mxai").read_text(encoding="utf-8")

        graph = graph_source(source, "mxai", output_format="mermaid")

        self.assertIn("flowchart LR", graph)
        self.assertIn('Email["Email\\nvector"]', graph)
        self.assertIn("Email --> Classifier", graph)

    def test_graph_json_supports_semantic_source(self) -> None:
        source = (ROOT / "examples" / "email-agent.semantic").read_text(encoding="utf-8")

        payload = json.loads(graph_source(source, "semantic", output_format="json"))

        self.assertEqual(payload["project"], "EmailAgent")
        self.assertEqual(payload["nodes"][0], {"id": "Email", "type": "vector"})

    def test_runtime_compiler_diagnostic_matches_example(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))

        report = diagnose_runtime_compiler(program, input_data)

        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.runtime_result["actions"], report.compiled_result["actions"])

    def test_cli_diagnose_json_reports_match(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrixai",
                "diagnose",
                "examples/email-agent.mxai",
                "--input",
                "examples/email-sample.json",
                "--json",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["project"], "EmailAgent")


if __name__ == "__main__":
    unittest.main()
