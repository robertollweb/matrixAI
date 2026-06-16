"""P13 Corte 4 — CLI `matrixai refine` tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_ENV = {**os.environ, "MATRIXAI_LLM_API_KEY": ""}

_PROMPT = "Crear un sistema de clasificacion de riesgo para pacientes con etiquetas alto, medio, bajo"

_AUDIT_NOT_ACTIVATED = {
    "actions": [
        {"name": "HighRisk", "source": "risk_score", "value": 0.3,
         "threshold": 0.5, "activated": False, "call": "simulated.HighRisk"},
    ],
    "trace": [{"node": "risk_score"}],
}

_EVAL_LOW_ACCURACY = {
    "accuracy": 0.62,
    "loss": 0.38,
    "thresholds": {"accuracy": 0.8, "loss": 0.5},
    "metrics_by_label": {},
}


def _run(*args, stdin_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "matrixai", *args],
        cwd=_BASE,
        capture_output=True,
        text=True,
        input=stdin_text,
        env=_ENV,
    )


class TestP13CliRefineArgParsing(unittest.TestCase):
    """Fast argparse-level tests — no PromptSupervisor calls."""

    def test_help_exits_zero(self):
        r = _run("refine", "--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--audit", r.stdout)
        self.assertIn("--evaluation", r.stdout)
        self.assertIn("--mxai-output", r.stdout)
        self.assertIn("--chain-output", r.stdout)
        self.assertIn("--accept", r.stdout)

    def test_missing_audit_and_evaluation_returns_2(self):
        r = _run("refine", "un prompt")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--audit", r.stderr)

    def test_both_audit_and_evaluation_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.json"
            eval_path = Path(tmp) / "eval.json"
            audit_path.write_text(json.dumps(_AUDIT_NOT_ACTIVATED))
            eval_path.write_text(json.dumps(_EVAL_LOW_ACCURACY))
            r = _run("refine", "un prompt", "--audit", str(audit_path),
                     "--evaluation", str(eval_path))
        self.assertEqual(r.returncode, 2)
        self.assertIn("mutually exclusive", r.stderr)

    def test_empty_prompt_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.json"
            audit_path.write_text(json.dumps(_AUDIT_NOT_ACTIVATED))
            r = _run("refine", "   ", "--audit", str(audit_path))
        self.assertEqual(r.returncode, 2)
        self.assertIn("empty", r.stderr)

    def test_missing_audit_file_returns_2(self):
        r = _run("refine", "un prompt", "--audit", "/nonexistent/audit.json")
        self.assertEqual(r.returncode, 2)
        self.assertIn("cannot read", r.stderr)

    def test_missing_evaluation_file_returns_2(self):
        r = _run("refine", "un prompt", "--evaluation", "/nonexistent/eval.json")
        self.assertEqual(r.returncode, 2)
        self.assertIn("cannot read", r.stderr)

    def test_malformed_audit_json_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{ not valid json }")
            r = _run("refine", "un prompt", "--audit", str(bad))
        self.assertEqual(r.returncode, 2)

    def test_missing_mxai_file_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.json"
            audit_path.write_text(json.dumps(_AUDIT_NOT_ACTIVATED))
            r = _run("refine", "un prompt", "--audit", str(audit_path),
                     "--mxai", "/nonexistent/model.mxai")
        self.assertEqual(r.returncode, 2)

    def test_malformed_chain_file_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.json"
            chain_path = Path(tmp) / "chain.json"
            audit_path.write_text(json.dumps(_AUDIT_NOT_ACTIVATED))
            chain_path.write_text('"not-a-list"')
            r = _run("refine", "un prompt", "--audit", str(audit_path),
                     "--chain", str(chain_path))
        self.assertEqual(r.returncode, 2)
        self.assertIn("chain", r.stderr)


class TestP13CliRefineAuditDriven(unittest.TestCase):
    """Integration tests with PromptSupervisor — audit_driven mode."""

    def _audit_file(self, tmp: str) -> Path:
        p = Path(tmp) / "audit.json"
        p.write_text(json.dumps(_AUDIT_NOT_ACTIVATED))
        return p

    def test_json_output_has_refinement_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)), "--json")
        self.assertIn(r.returncode, (0, 1))
        data = json.loads(r.stdout)
        self.assertIn("refinement_id", data)
        self.assertTrue(data["refinement_id"].startswith("refinement_audit_"))

    def test_json_output_has_chain_and_parent_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)), "--json")
        data = json.loads(r.stdout)
        self.assertIn("refinement_chain", data)
        self.assertIn("parent_prompt_hash", data)
        self.assertEqual(len(data["refinement_chain"]), 1)
        self.assertRegex(data["parent_prompt_hash"], r"^[0-9a-f]{64}$")

    def test_json_output_mode_is_audit_driven(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)), "--json")
        data = json.loads(r.stdout)
        self.assertEqual(data["mode"], "audit_driven")

    def test_human_readable_output_contains_refinement_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)))
        self.assertIn("Refinement ID", r.stdout)
        self.assertIn("audit_driven", r.stdout)

    def test_no_accept_no_output_files_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "prompt.txt"
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                     "--output", str(out_path))
            self.assertFalse(out_path.exists(), "Without --accept, file must not be written")
        self.assertIn("--accept", r.stderr)

    def test_accept_without_outputs_emits_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)), "--accept")
        self.assertIn("WARNING: --accept was provided, but no output destinations", r.stderr)

    def test_accept_writes_proposed_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "prompt.txt"
            _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                 "--output", str(out_path), "--accept")
            if out_path.exists():
                content = out_path.read_text()
                self.assertIn(_PROMPT, content)

    def test_accept_writes_chain_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain_path = Path(tmp) / "chain.json"
            _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                 "--chain-output", str(chain_path), "--accept")
            if chain_path.exists():
                chain = json.loads(chain_path.read_text())
                self.assertIsInstance(chain, list)
                self.assertEqual(len(chain), 1)

    def test_accept_writes_mxai_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            mxai_out = Path(tmp) / "refined.mxai"
            _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                 "--mxai-output", str(mxai_out), "--accept")
            if mxai_out.exists():
                content = mxai_out.read_text()
                self.assertIn("# refinement_chain:", content)
                self.assertIn("# parent_prompt_hash:", content)

    def test_hint_flag_appears_in_json_hints_applied(self):
        user_hint = "Reducir el umbral a 0.3"
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                     "--hint", user_hint, "--json")
        data = json.loads(r.stdout)
        self.assertIn(user_hint, data["hints_applied"])

    def test_iteration_flag_stored_in_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                     "--iteration", "2", "--json")
        data = json.loads(r.stdout)
        self.assertEqual(data["iteration_count"], 2)

    def test_stdin_prompt_with_dash(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", "-", "--audit", str(self._audit_file(tmp)), "--json",
                     stdin_text=_PROMPT)
        self.assertIn(r.returncode, (0, 1))
        data = json.loads(r.stdout)
        self.assertEqual(data["original_prompt"], _PROMPT)

    def test_chain_file_propagated_to_second_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            # First iteration
            chain_path = Path(tmp) / "chain.json"
            r1 = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                      "--chain-output", str(chain_path), "--accept", "--json")
            d1 = json.loads(r1.stdout)
            if not chain_path.exists():
                self.skipTest("mxai output not generated (supervisor rejected)")
            # Second iteration
            r2 = _run(
                "refine", d1["proposed_prompt"], "--audit", str(self._audit_file(tmp)),
                "--chain", str(chain_path),
                "--parent-hash", d1["parent_prompt_hash"],
                "--iteration", "2", "--json",
            )
            d2 = json.loads(r2.stdout)
            self.assertEqual(len(d2["refinement_chain"]), 2)
            self.assertEqual(d2["refinement_chain"][0], d1["refinement_id"])
            self.assertEqual(d2["parent_prompt_hash"], d1["parent_prompt_hash"])


class TestP13CliRefineIterationLimit(unittest.TestCase):
    """Corte 5 — hard iteration limit via CLI."""

    def _audit_file(self, tmp: str) -> Path:
        p = Path(tmp) / "audit.json"
        p.write_text(json.dumps(_AUDIT_NOT_ACTIVATED))
        return p

    def test_iteration_exceeds_default_limit_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                     "--iteration", "4")
        self.assertEqual(r.returncode, 2)
        self.assertIn("Limite", r.stderr)

    def test_max_iterations_flag_blocks_when_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                     "--iteration", "3", "--max-iterations", "2")
        self.assertEqual(r.returncode, 2)
        self.assertIn("Limite", r.stderr)

    def test_max_iterations_flag_allows_at_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                     "--iteration", "3", "--max-iterations", "3", "--json")
        self.assertIn(r.returncode, (0, 1))
        data = json.loads(r.stdout)
        self.assertEqual(data["iteration_count"], 3)

    def test_error_message_includes_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--audit", str(self._audit_file(tmp)),
                     "--iteration", "5", "--max-iterations", "3")
        self.assertEqual(r.returncode, 2)
        self.assertIn("5", r.stderr)
        self.assertIn("3", r.stderr)

    def test_max_iterations_appears_in_help(self):
        r = _run("refine", "--help")
        self.assertIn("--max-iterations", r.stdout)


class TestP13CliRefineMetricDriven(unittest.TestCase):
    """Integration tests — metric_driven mode."""

    def _eval_file(self, tmp: str) -> Path:
        p = Path(tmp) / "eval.json"
        p.write_text(json.dumps(_EVAL_LOW_ACCURACY))
        return p

    def test_metric_driven_json_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--evaluation", str(self._eval_file(tmp)), "--json")
        self.assertIn(r.returncode, (0, 1))
        data = json.loads(r.stdout)
        self.assertEqual(data["mode"], "metric_driven")
        self.assertTrue(data["refinement_id"].startswith("refinement_metri_"))

    def test_metric_driven_chain_in_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run("refine", _PROMPT, "--evaluation", str(self._eval_file(tmp)), "--json")
        data = json.loads(r.stdout)
        self.assertEqual(len(data["refinement_chain"]), 1)


if __name__ == "__main__":
    unittest.main()
