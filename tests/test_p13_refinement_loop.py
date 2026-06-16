"""P13 Refinement Loop — test suite.

Corte 1: RefinementAgent con modo audit_driven.
  - Contrato: refine() devuelve RefinementProposal válido.
  - refinement_id determinista respecto a los inputs.
  - Análisis de auditoria: desde AuditReport estructurado en vez de regex.
  - proposed_prompt incorpora los hints derivados del audit con <SystemFeedback>.
  - PromptSupervisor siempre se invoca y su resultado queda en supervision_report.
  - Validaciones de entradas (prompt vacío, modo inválido, audit ausente).
  - metric_driven levanta NotImplementedError (pendiente Corte 2).
"""
from __future__ import annotations

import unittest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUDIT_NOT_ACTIVATED = {
    "actions": [
        {"name": "HighRisk", "source": "risk_score", "value": 0.3, "threshold": 0.5, "activated": False, "call": "simulated.HighRisk"}
    ],
    "trace": [{"node": "risk_score"}, {"node": "risk_level"}]
}

_AUDIT_ACTIVATED = {
    "actions": [
        {"name": "SendAlert", "source": "alert_score", "value": 0.8, "threshold": 0.5, "activated": True, "call": "simulated.SendAlert"}
    ],
    "trace": [{"node": "alert_score"}]
}

_AUDIT_NO_ACTIONS = {
    "actions": [],
    "trace": []
}

_PROMPT = "Crear un sistema de clasificacion de riesgo para pacientes con etiquetas alto, medio, bajo"

_MXAI_MOCK = """
risk_score = 0.3 * factor_a
"""

# ---------------------------------------------------------------------------
# Corte 1: RefinementAgent — audit_driven
# ---------------------------------------------------------------------------

class TestP13RefinementAgentAuditDriven(unittest.TestCase):

    def _agent(self):
        from matrixai.agents.refinement import RefinementAgent
        return RefinementAgent()

    # --- Happy path ----------------------------------------------------------

    def test_refine_returns_proposal(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mode, "audit_driven")
        self.assertEqual(proposal.original_prompt, _PROMPT)
        self.assertEqual(proposal.iteration_count, 1)

    def test_proposal_has_refinement_id(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertTrue(proposal.refinement_id.startswith("refinement_audit_"))
        self.assertGreater(len(proposal.refinement_id), 10)

    def test_refinement_id_is_deterministic(self):
        a = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        b = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertEqual(a.refinement_id, b.refinement_id)

    def test_refinement_id_differs_for_different_prompts(self):
        a = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        b = self._agent().refine("Otro prompt completamente distinto", audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertNotEqual(a.refinement_id, b.refinement_id)

    def test_proposed_prompt_differs_from_original(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertNotEqual(proposal.proposed_prompt, _PROMPT)
        self.assertIn(_PROMPT, proposal.proposed_prompt)

    def test_proposed_prompt_includes_action_name_from_audit(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIn("HighRisk", proposal.proposed_prompt)

    def test_hints_applied_not_empty(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIsInstance(proposal.hints_applied, list)
        self.assertGreater(len(proposal.hints_applied), 0)

    def test_explanation_references_audit(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIn("Auditoria", proposal.explanation)
        self.assertIn("0/1 acciones activadas", proposal.explanation)

    def test_supervision_report_is_dict_with_accepted_key(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIsInstance(proposal.supervision_report, dict)
        self.assertIn("accepted", proposal.supervision_report)

    def test_supervision_accepted_matches_report(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertEqual(proposal.supervision_accepted, proposal.supervision_report["accepted"])

    def test_to_dict_is_serializable(self):
        import json
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        d = proposal.to_dict()
        # Should be JSON-serializable (supervision_report may have nested dicts)
        dumped = json.dumps(d, default=str)
        self.assertIn("refinement_id", dumped)

    # --- Audit pattern: action activated ------------------------------------

    def test_activated_action_detected_in_audit(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_ACTIVATED, mode="audit_driven")
        self.assertIn("SendAlert", proposal.proposed_prompt)

    # --- Audit pattern: no actions ------------------------------------------

    def test_no_actions_audit_adds_generic_hint(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NO_ACTIONS, mode="audit_driven")
        text = " ".join(proposal.hints_applied)
        self.assertTrue(
            any("accion" in h.lower() or "condicion" in h.lower() for h in proposal.hints_applied)
        )

    # --- Context mxai injection ---------------------------------------------

    def test_mxai_context_is_injected(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mxai=_MXAI_MOCK, mode="audit_driven")
        self.assertTrue(any("risk_score = 0.3 * factor_a" in h for h in proposal.hints_applied))

    # --- Iteration count ----------------------------------------------------

    def test_iteration_count_adds_warning_if_high(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven",
                                        iteration_count=4, max_iterations=5)
        self.assertEqual(proposal.iteration_count, 4)
        self.assertIn("ADVERTENCIA", proposal.proposed_prompt)

    def test_xml_tags_in_prompt(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIn("<SystemFeedback>", proposal.proposed_prompt)
        self.assertIn("</SystemFeedback>", proposal.proposed_prompt)

    # --- User hints ---------------------------------------------------------

    def test_user_hints_are_appended(self):
        user_hint = "Reducir el umbral de clasificacion a 0.3"
        proposal = self._agent().refine(
            _PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven",
            hints=[user_hint],
        )
        self.assertIn(user_hint, proposal.hints_applied)
        self.assertIn(user_hint, proposal.proposed_prompt)

    # --- Default mode -------------------------------------------------------

    def test_default_mode_is_audit_driven(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED)
        self.assertEqual(proposal.mode, "audit_driven")

    # --- Validation errors --------------------------------------------------

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            self._agent().refine("", audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")

    def test_whitespace_prompt_raises(self):
        with self.assertRaises(ValueError):
            self._agent().refine("   ", audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")

    def test_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="unknown_mode")

    def test_audit_driven_without_audit_raises(self):
        with self.assertRaises(ValueError):
            self._agent().refine(_PROMPT, mode="audit_driven")

    def test_metric_driven_without_evaluation_raises(self):
        with self.assertRaises(ValueError):
            self._agent().refine(_PROMPT, mode="metric_driven")

    # --- Import from agents package -----------------------------------------

    def test_importable_from_agents_package(self):
        from matrixai.agents import RefinementAgent, RefinementProposal
        self.assertTrue(callable(RefinementAgent))
        self.assertTrue(callable(RefinementProposal))


# ---------------------------------------------------------------------------
# Corte 5: Límite de iteraciones
# ---------------------------------------------------------------------------

class TestP13IterationLimit(unittest.TestCase):

    def _agent(self):
        from matrixai.agents.refinement import RefinementAgent
        return RefinementAgent()

    # --- Default limit (3) ------------------------------------------------

    def test_default_max_iterations_is_3(self):
        from matrixai.agents.refinement import RefinementAgent
        self.assertEqual(RefinementAgent.DEFAULT_MAX_ITERATIONS, 3)

    def test_iteration_at_boundary_does_not_raise(self):
        """iteration_count == max_iterations debe ser aceptado."""
        from matrixai.agents.refinement import IterationLimitReached
        audit = {"actions": [], "trace": []}
        try:
            self._agent().refine(
                _PROMPT, audit=audit, mode="audit_driven",
                iteration_count=3, max_iterations=3,
            )
        except IterationLimitReached:
            self.fail("iteration_count == max_iterations should not raise IterationLimitReached")

    def test_iteration_exceeds_default_limit_raises(self):
        from matrixai.agents.refinement import IterationLimitReached
        audit = {"actions": [], "trace": []}
        with self.assertRaises(IterationLimitReached):
            self._agent().refine(
                _PROMPT, audit=audit, mode="audit_driven",
                iteration_count=4,
            )

    def test_error_message_includes_current_and_max(self):
        from matrixai.agents.refinement import IterationLimitReached
        audit = {"actions": [], "trace": []}
        with self.assertRaises(IterationLimitReached) as ctx:
            self._agent().refine(
                _PROMPT, audit=audit, mode="audit_driven",
                iteration_count=5, max_iterations=3,
            )
        msg = str(ctx.exception)
        self.assertIn("5", msg)
        self.assertIn("3", msg)

    # --- Custom limit -----------------------------------------------------

    def test_custom_limit_blocks_at_threshold(self):
        from matrixai.agents.refinement import IterationLimitReached
        audit = {"actions": [], "trace": []}
        with self.assertRaises(IterationLimitReached):
            self._agent().refine(
                _PROMPT, audit=audit, mode="audit_driven",
                iteration_count=3, max_iterations=2,
            )

    def test_custom_limit_allows_below_threshold(self):
        from matrixai.agents.refinement import IterationLimitReached
        audit = {"actions": [], "trace": []}
        try:
            self._agent().refine(
                _PROMPT, audit=audit, mode="audit_driven",
                iteration_count=2, max_iterations=5,
            )
        except IterationLimitReached:
            self.fail("iteration_count < max_iterations should not raise")

    def test_limit_applies_to_metric_driven_too(self):
        from matrixai.agents.refinement import IterationLimitReached
        with self.assertRaises(IterationLimitReached):
            self._agent().refine(
                _PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven",
                iteration_count=4, max_iterations=3,
            )

    # --- Exception type ---------------------------------------------------

    def test_exception_is_runtime_error_subclass(self):
        from matrixai.agents.refinement import IterationLimitReached
        self.assertTrue(issubclass(IterationLimitReached, RuntimeError))

    def test_importable_from_agents_package(self):
        from matrixai.agents import IterationLimitReached
        self.assertTrue(issubclass(IterationLimitReached, RuntimeError))


# ---------------------------------------------------------------------------
# Corte 2: RefinementAgent — metric_driven
# ---------------------------------------------------------------------------

_EVAL_LOW_ACCURACY = {
    "accuracy": 0.62,
    "loss": 0.38,
    "thresholds": {"accuracy": 0.8, "loss": 0.5},
    "metrics_by_label": {},
}

_EVAL_HIGH_LOSS = {
    "accuracy": 0.85,
    "loss": 0.72,
    "thresholds": {"accuracy": 0.8, "loss": 0.5},
    "metrics_by_label": {},
}

_EVAL_LOW_F1_LABEL = {
    "accuracy": 0.85,
    "loss": 0.3,
    "thresholds": {"accuracy": 0.8, "loss": 0.5, "f1": 0.7},
    "metrics_by_label": {
        "alto": {"f1": 0.55},
        "bajo": {"f1": 0.9},
    },
}

_EVAL_ALL_OK = {
    "accuracy": 0.92,
    "loss": 0.2,
    "thresholds": {"accuracy": 0.8, "loss": 0.5},
    "metrics_by_label": {},
}


class TestP13RefinementAgentMetricDriven(unittest.TestCase):

    def _agent(self):
        from matrixai.agents.refinement import RefinementAgent
        return RefinementAgent()

    # --- Happy path ----------------------------------------------------------

    def test_metric_driven_returns_proposal(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mode, "metric_driven")
        self.assertEqual(proposal.original_prompt, _PROMPT)

    def test_metric_driven_refinement_id_format(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertTrue(proposal.refinement_id.startswith("refinement_metri_"))
        self.assertGreater(len(proposal.refinement_id), 10)

    def test_metric_driven_id_is_deterministic(self):
        a = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        b = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertEqual(a.refinement_id, b.refinement_id)

    def test_metric_driven_proposed_prompt_contains_original(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertIn(_PROMPT, proposal.proposed_prompt)

    def test_metric_driven_low_accuracy_hint_in_hints(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertTrue(any("precision" in h.lower() or "accuracy" in h.lower() or "0.62" in h for h in proposal.hints_applied))

    def test_metric_driven_high_loss_hint_in_hints(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_HIGH_LOSS, mode="metric_driven")
        self.assertTrue(any("perdida" in h.lower() or "loss" in h.lower() or "0.72" in h for h in proposal.hints_applied))

    def test_metric_driven_low_f1_label_hint_in_hints(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_F1_LABEL, mode="metric_driven")
        self.assertTrue(any("alto" in h for h in proposal.hints_applied))

    def test_metric_driven_all_ok_still_returns_proposal(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_ALL_OK, mode="metric_driven")
        self.assertIsInstance(proposal, type(proposal))
        self.assertGreater(len(proposal.hints_applied), 0)

    def test_metric_driven_explanation_references_accuracy(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertIn("evaluacion", proposal.explanation.lower())
        self.assertIn("0.62", proposal.explanation)

    def test_metric_driven_has_system_feedback_tags(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertIn("<SystemFeedback>", proposal.proposed_prompt)
        self.assertIn("</SystemFeedback>", proposal.proposed_prompt)

    def test_metric_driven_supervision_report_present(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertIsInstance(proposal.supervision_report, dict)
        self.assertIn("accepted", proposal.supervision_report)

    def test_metric_driven_supervision_accepted_matches_report(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertEqual(proposal.supervision_accepted, proposal.supervision_report["accepted"])

    def test_metric_driven_user_hints_appended(self):
        user_hint = "Aumentar el dataset a 500 ejemplos por clase"
        proposal = self._agent().refine(
            _PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven",
            hints=[user_hint],
        )
        self.assertIn(user_hint, proposal.hints_applied)
        self.assertIn(user_hint, proposal.proposed_prompt)

    def test_metric_driven_iteration_count_stored(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven", iteration_count=2)
        self.assertEqual(proposal.iteration_count, 2)

    def test_metric_driven_high_iteration_adds_warning(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven",
                                        iteration_count=4, max_iterations=5)
        self.assertIn("ADVERTENCIA", proposal.proposed_prompt)

    def test_metric_driven_to_dict_serializable(self):
        import json
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        dumped = json.dumps(proposal.to_dict(), default=str)
        self.assertIn("refinement_id", dumped)


# ---------------------------------------------------------------------------
# Corte 3: Trazabilidad de iteraciones
# ---------------------------------------------------------------------------

class TestP13Traceability(unittest.TestCase):

    def _agent(self):
        from matrixai.agents.refinement import RefinementAgent
        return RefinementAgent()

    # --- Single iteration chain -------------------------------------------

    def test_single_iteration_chain_has_one_element(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertEqual(len(proposal.refinement_chain), 1)
        self.assertEqual(proposal.refinement_chain[0], proposal.refinement_id)

    def test_single_iteration_parent_hash_is_sha256_of_prompt(self):
        import hashlib
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        expected = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
        self.assertEqual(proposal.parent_prompt_hash, expected)

    def test_parent_hash_is_64_hex_chars(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertRegex(proposal.parent_prompt_hash, r"^[0-9a-f]{64}$")

    # --- Multi-iteration chain --------------------------------------------

    def test_chain_grows_across_iterations(self):
        agent = self._agent()
        first = agent.refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven", iteration_count=1)
        second = agent.refine(
            first.proposed_prompt,
            audit=_AUDIT_NOT_ACTIVATED,
            mode="audit_driven",
            iteration_count=2,
            refinement_chain=first.refinement_chain,
            parent_prompt_hash=first.parent_prompt_hash,
        )
        self.assertEqual(len(second.refinement_chain), 2)
        self.assertEqual(second.refinement_chain[0], first.refinement_id)
        self.assertEqual(second.refinement_chain[1], second.refinement_id)

    def test_parent_hash_stable_across_iterations(self):
        agent = self._agent()
        first = agent.refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        second = agent.refine(
            first.proposed_prompt,
            audit=_AUDIT_NOT_ACTIVATED,
            mode="audit_driven",
            refinement_chain=first.refinement_chain,
            parent_prompt_hash=first.parent_prompt_hash,
        )
        self.assertEqual(first.parent_prompt_hash, second.parent_prompt_hash)

    def test_explicit_parent_hash_is_preserved(self):
        custom_hash = "a" * 64
        proposal = self._agent().refine(
            _PROMPT,
            audit=_AUDIT_NOT_ACTIVATED,
            mode="audit_driven",
            parent_prompt_hash=custom_hash,
        )
        self.assertEqual(proposal.parent_prompt_hash, custom_hash)

    def test_three_iteration_chain_length(self):
        agent = self._agent()
        p1 = agent.refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven", iteration_count=1)
        p2 = agent.refine(
            p1.proposed_prompt, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven",
            iteration_count=2, refinement_chain=p1.refinement_chain, parent_prompt_hash=p1.parent_prompt_hash,
        )
        p3 = agent.refine(
            p2.proposed_prompt, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven",
            iteration_count=3, refinement_chain=p2.refinement_chain, parent_prompt_hash=p2.parent_prompt_hash,
        )
        self.assertEqual(len(p3.refinement_chain), 3)
        self.assertEqual(p3.refinement_chain, [p1.refinement_id, p2.refinement_id, p3.refinement_id])

    # --- Supervision report carries chain --------------------------------

    def test_supervision_report_has_refinement_chain(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIn("refinement_chain", proposal.supervision_report)
        self.assertEqual(proposal.supervision_report["refinement_chain"], proposal.refinement_chain)

    def test_supervision_report_has_parent_prompt_hash(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        self.assertIn("parent_prompt_hash", proposal.supervision_report)
        self.assertEqual(proposal.supervision_report["parent_prompt_hash"], proposal.parent_prompt_hash)

    # --- metric_driven also carries chain --------------------------------

    def test_metric_driven_chain_present(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertEqual(len(proposal.refinement_chain), 1)
        self.assertEqual(proposal.refinement_chain[0], proposal.refinement_id)

    def test_metric_driven_parent_hash_present(self):
        proposal = self._agent().refine(_PROMPT, evaluation=_EVAL_LOW_ACCURACY, mode="metric_driven")
        self.assertRegex(proposal.parent_prompt_hash, r"^[0-9a-f]{64}$")

    # --- Serialization ---------------------------------------------------

    def test_chain_survives_to_dict(self):
        import json
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        d = proposal.to_dict()
        self.assertIn("refinement_chain", d)
        self.assertIn("parent_prompt_hash", d)
        dumped = json.dumps(d)
        self.assertIn("refinement_chain", dumped)

    # --- _hash_prompt helper ---------------------------------------------

    def test_hash_prompt_helper(self):
        import hashlib
        from matrixai.agents.refinement import _hash_prompt
        expected = hashlib.sha256("test".encode("utf-8")).hexdigest()
        self.assertEqual(_hash_prompt("test"), expected)

    def test_hash_prompt_deterministic(self):
        from matrixai.agents.refinement import _hash_prompt
        self.assertEqual(_hash_prompt("abc"), _hash_prompt("abc"))

    def test_hash_prompt_differs_for_different_prompts(self):
        from matrixai.agents.refinement import _hash_prompt
        self.assertNotEqual(_hash_prompt("abc"), _hash_prompt("xyz"))

    # --- Edge cases: empty inputs ------------------------------------------

    def test_empty_list_refinement_chain_equals_none(self):
        """refinement_chain=[] debe comportarse igual que refinement_chain=None."""
        agent = self._agent()
        p_none = agent.refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        p_empty = agent.refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven", refinement_chain=[])
        self.assertEqual(len(p_none.refinement_chain), 1)
        self.assertEqual(len(p_empty.refinement_chain), 1)
        self.assertEqual(p_none.refinement_id, p_empty.refinement_id)

    def test_empty_string_parent_hash_computes_from_prompt(self):
        """parent_prompt_hash='' debe calcular el hash del prompt, igual que None."""
        import hashlib
        agent = self._agent()
        p = agent.refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven", parent_prompt_hash="")
        expected = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
        self.assertEqual(p.parent_prompt_hash, expected)

    # --- .mxai metadata embedding ------------------------------------------

    def test_mxai_has_refinement_chain_comment(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        mxai = proposal.supervision_report.get("mxai", "")
        if mxai:
            self.assertIn("# refinement_chain:", mxai)
            self.assertIn(proposal.refinement_id, mxai)

    def test_mxai_has_parent_prompt_hash_comment(self):
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        mxai = proposal.supervision_report.get("mxai", "")
        if mxai:
            self.assertIn("# parent_prompt_hash:", mxai)
            self.assertIn(proposal.parent_prompt_hash, mxai)

    def test_mxai_metadata_roundtrip(self):
        from matrixai.agents.refinement import _parse_refinement_metadata
        proposal = self._agent().refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven")
        mxai = proposal.supervision_report.get("mxai", "")
        if mxai:
            chain, phash = _parse_refinement_metadata(mxai)
            self.assertEqual(chain, proposal.refinement_chain)
            self.assertEqual(phash, proposal.parent_prompt_hash)

    def test_mxai_metadata_chain_grows_across_iterations(self):
        from matrixai.agents.refinement import _parse_refinement_metadata
        agent = self._agent()
        p1 = agent.refine(_PROMPT, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven", iteration_count=1)
        p2 = agent.refine(
            p1.proposed_prompt, audit=_AUDIT_NOT_ACTIVATED, mode="audit_driven",
            iteration_count=2, refinement_chain=p1.refinement_chain,
            parent_prompt_hash=p1.parent_prompt_hash,
        )
        mxai2 = p2.supervision_report.get("mxai", "")
        if mxai2:
            chain, _ = _parse_refinement_metadata(mxai2)
            self.assertEqual(len(chain), 2)
            self.assertEqual(chain[0], p1.refinement_id)
            self.assertEqual(chain[1], p2.refinement_id)

    def test_parse_refinement_metadata_no_metadata(self):
        from matrixai.agents.refinement import _parse_refinement_metadata
        chain, phash = _parse_refinement_metadata("PROJECT Simple\n\nVECTOR X[2]\nEND\n")
        self.assertEqual(chain, [])
        self.assertEqual(phash, "")

    def test_embed_refinement_metadata_empty_mxai(self):
        from matrixai.agents.refinement import _embed_refinement_metadata
        result = _embed_refinement_metadata("", ["id1"], "hash1")
        self.assertEqual(result, "")

    def test_embed_refinement_metadata_roundtrip(self):
        from matrixai.agents.refinement import _embed_refinement_metadata, _parse_refinement_metadata
        mxai = "PROJECT X\nVECTOR V[1]\n  v\nEND\n"
        chain = ["refinement_audit_abc", "refinement_audit_def"]
        phash = "a" * 64
        embedded = _embed_refinement_metadata(mxai, chain, phash)
        parsed_chain, parsed_hash = _parse_refinement_metadata(embedded)
        self.assertEqual(parsed_chain, chain)
        self.assertEqual(parsed_hash, phash)


# ---------------------------------------------------------------------------
# Helpers: _build_metric_explanation
# ---------------------------------------------------------------------------

class TestP13MetricHelpers(unittest.TestCase):

    def test_build_metric_explanation_accuracy(self):
        from matrixai.agents.refinement import _build_metric_explanation
        exp = _build_metric_explanation({"accuracy": 0.62, "loss": 0.38}, ["hint uno"])
        self.assertIn("0.6200", exp)
        self.assertIn("hint uno", exp)

    def test_build_metric_explanation_loss_only(self):
        from matrixai.agents.refinement import _build_metric_explanation
        exp = _build_metric_explanation({"loss": 0.7}, [])
        self.assertIn("0.7000", exp)

    def test_build_metric_explanation_empty_eval(self):
        from matrixai.agents.refinement import _build_metric_explanation
        exp = _build_metric_explanation({}, [])
        self.assertIn("evaluacion", exp.lower())


# ---------------------------------------------------------------------------
# Helpers module-level functions
# ---------------------------------------------------------------------------

class TestP13RefinementHelpers(unittest.TestCase):

    def test_make_refinement_id_format(self):
        from matrixai.agents.refinement import _make_refinement_id
        rid = _make_refinement_id("prompt A", "prompt A + hint", "audit_driven")
        self.assertRegex(rid, r"^refinement_audit_[0-9a-f]{12}$")

    def test_make_refinement_id_deterministic(self):
        from matrixai.agents.refinement import _make_refinement_id
        self.assertEqual(
            _make_refinement_id("p", "q", "audit_driven"),
            _make_refinement_id("p", "q", "audit_driven"),
        )

    def test_build_refined_prompt_no_hints(self):
        from matrixai.agents.refinement import _build_refined_prompt
        self.assertEqual(_build_refined_prompt("Mi prompt", []), "Mi prompt")

    def test_build_refined_prompt_with_hints(self):
        from matrixai.agents.refinement import _build_refined_prompt
        result = _build_refined_prompt("Mi prompt", ["hint A", "hint B"])
        self.assertIn("Mi prompt", result)
        self.assertIn("hint A", result)
        self.assertIn("hint B", result)
        self.assertIn("<SystemFeedback>", result)

    def test_build_explanation_contains_audit_preview(self):
        from matrixai.agents.refinement import _build_explanation
        exp = _build_explanation({"actions": [{"activated": True}]}, ["hint uno"])
        self.assertIn("1/1", exp)
        self.assertIn("hint uno", exp)

    def test_build_explanation_no_actions(self):
        from matrixai.agents.refinement import _build_explanation
        exp = _build_explanation({"actions": []}, [])
        self.assertIn("No se activo", exp)


if __name__ == "__main__":
    unittest.main()
