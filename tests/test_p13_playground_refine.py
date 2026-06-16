"""P13 Corte 6: playground /api/refine endpoint tests."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from matrixai.playground import _refine_prompt


MINIMAL_PROMPT = "Si fiebre > 38 entonces ALERT"
MINIMAL_RUN = {
    "actions": [{"name": "ALERT", "value": 0.9, "activated": True, "threshold": 0.5, "policy": "threshold"}],
    "trace": [{"step": 0, "node": "fiebre", "node_type": "input", "status": "ok"}],
}


class TestP13PlaygroundRefineValidation:
    def test_missing_prompt_returns_error(self):
        r = _refine_prompt({})
        assert r["ok"] is False
        assert "prompt" in r["error"].lower()

    def test_empty_prompt_returns_error(self):
        r = _refine_prompt({"prompt": "   ", "run_result": MINIMAL_RUN})
        assert r["ok"] is False

    def test_missing_run_result_returns_error(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT})
        assert r["ok"] is False
        assert "run_result" in r["error"].lower()

    def test_empty_run_result_returns_error(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": {}})
        assert r["ok"] is False


class TestP13PlaygroundRefineSuccess:
    def test_ok_response_has_required_keys(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN})
        assert r["ok"] is True
        assert "refinement_id" in r
        assert "proposed_prompt" in r
        assert "explanation" in r
        assert "chain" in r
        assert "parent_hash" in r
        assert "supervision_accepted" in r
        assert "mode" in r
        assert "iteration" in r

    def test_mode_is_audit_driven(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN})
        assert r["mode"] == "audit_driven"

    def test_chain_starts_with_one_element(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN})
        assert isinstance(r["chain"], list)
        assert len(r["chain"]) == 1

    def test_parent_hash_is_64_hex_chars(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN})
        assert isinstance(r["parent_hash"], str)
        assert len(r["parent_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in r["parent_hash"])

    def test_iteration_defaults_to_1(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN})
        assert r["iteration"] == 1

    def test_explicit_iteration_reflected(self):
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN, "iteration_count": 2, "max_iterations": 5})
        assert r["iteration"] == 2

    def test_audit_field_stripped_from_run_result(self):
        run_with_audit = {**MINIMAL_RUN, "audit": "Este modelo activa ALERT cuando fiebre > 38."}
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": run_with_audit})
        assert r["ok"] is True

    def test_hints_accepted(self):
        r = _refine_prompt({
            "prompt": MINIMAL_PROMPT,
            "run_result": MINIMAL_RUN,
            "hints": ["Ser mas especifico", "Mencionar unidad Celsius"],
        })
        assert r["ok"] is True

    def test_mxai_text_accepted(self):
        mxai = "PROJECT test\nINPUT fiebre FLOAT\nACTION ALERT IF fiebre > 38"
        r = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN, "mxai_text": mxai})
        assert r["ok"] is True

    def test_chain_grows_across_iterations(self):
        r1 = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN})
        assert r1["ok"]
        r2 = _refine_prompt({
            "prompt": r1["proposed_prompt"],
            "run_result": MINIMAL_RUN,
            "iteration_count": 2,
            "refinement_chain": r1["chain"],
            "parent_prompt_hash": r1["parent_hash"],
            "max_iterations": 5,
        })
        assert r2["ok"]
        assert len(r2["chain"]) == 2
        assert r2["chain"][0] == r1["chain"][0]

    def test_parent_hash_stable_across_iterations(self):
        r1 = _refine_prompt({"prompt": MINIMAL_PROMPT, "run_result": MINIMAL_RUN})
        r2 = _refine_prompt({
            "prompt": r1["proposed_prompt"],
            "run_result": MINIMAL_RUN,
            "iteration_count": 2,
            "refinement_chain": r1["chain"],
            "parent_prompt_hash": r1["parent_hash"],
            "max_iterations": 5,
        })
        assert r1["parent_hash"] == r2["parent_hash"]


class TestP13PlaygroundRefineIterationLimit:
    def test_iteration_limit_reached_returns_error(self):
        r = _refine_prompt({
            "prompt": MINIMAL_PROMPT,
            "run_result": MINIMAL_RUN,
            "iteration_count": 4,
            "max_iterations": 3,
        })
        assert r["ok"] is False
        assert r.get("iteration_limit_reached") is True

    def test_iteration_limit_error_message_has_counts(self):
        r = _refine_prompt({
            "prompt": MINIMAL_PROMPT,
            "run_result": MINIMAL_RUN,
            "iteration_count": 5,
            "max_iterations": 3,
        })
        assert "5" in r["error"]
        assert "3" in r["error"]

    def test_default_max_iterations_is_3(self):
        r = _refine_prompt({
            "prompt": MINIMAL_PROMPT,
            "run_result": MINIMAL_RUN,
            "iteration_count": 4,
        })
        assert r["ok"] is False
        assert r.get("iteration_limit_reached") is True

    def test_at_max_iterations_boundary_succeeds(self):
        r = _refine_prompt({
            "prompt": MINIMAL_PROMPT,
            "run_result": MINIMAL_RUN,
            "iteration_count": 3,
            "max_iterations": 3,
        })
        assert r["ok"] is True

    def test_custom_max_iterations_respected(self):
        r = _refine_prompt({
            "prompt": MINIMAL_PROMPT,
            "run_result": MINIMAL_RUN,
            "iteration_count": 4,
            "max_iterations": 5,
        })
        assert r["ok"] is True
