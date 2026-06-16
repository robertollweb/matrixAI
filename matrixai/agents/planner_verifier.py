# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from matrixai.agents.architect import SemanticPlan


@dataclass(frozen=True)
class PlanVerificationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class PlannerVerifier:
    _CONDITION_RE = re.compile(
        r"^(?P<source>[A-Za-z_][\w.]*)\s*(?P<operator>>|>=|<|<=)\s*(?P<threshold>[0-9.]+)$"
    )
    _IDENTIFIER_RE = re.compile(r"^[A-Za-z_][\w]*$")

    def verify(self, plan: SemanticPlan) -> PlanVerificationResult:
        errors: list[str] = []
        warnings: list[str] = []

        self._verify_project(plan, errors)
        self._verify_mode(plan, errors)
        self._verify_vector(plan, errors)
        self._verify_goals(plan, warnings)
        self._verify_functions(plan, errors)
        self._verify_distributions(plan, errors, warnings)
        self._verify_graph(plan, errors)
        self._verify_actions(plan, errors, warnings)
        self._verify_audit(plan, errors)
        self._verify_goal_rules(plan, errors, warnings)
        self._verify_mathematical_translations(plan, warnings)

        return PlanVerificationResult(errors=errors, warnings=warnings)

    def _verify_project(self, plan: SemanticPlan, errors: list[str]) -> None:
        if not plan.project or not self._IDENTIFIER_RE.match(plan.project):
            errors.append("PLAN project must be a valid identifier")

    def _verify_mode(self, plan: SemanticPlan, errors: list[str]) -> None:
        if plan.mode not in {"classification", "risk", "regression"}:
            errors.append("PLAN mode must be classification, risk or regression")

    def _verify_vector(self, plan: SemanticPlan, errors: list[str]) -> None:
        name = plan.vector.get("name", "")
        fields = plan.vector.get("fields", [])
        if not name or not self._IDENTIFIER_RE.match(name):
            errors.append("PLAN vector requires a valid name")
        min_fields = 1 if plan.mode == "regression" else 2
        if not isinstance(fields, list) or len(fields) < min_fields:
            errors.append(f"PLAN vector requires at least {min_fields} field(s)")
            return
        if len(fields) != len(set(fields)):
            errors.append("PLAN vector fields must be unique")
        invalid_fields = [field for field in fields if not self._IDENTIFIER_RE.match(str(field))]
        if invalid_fields:
            errors.append(f"PLAN vector has invalid fields: {', '.join(invalid_fields)}")

    def _verify_goals(self, plan: SemanticPlan, warnings: list[str]) -> None:
        if not plan.goals:
            warnings.append("PLAN has no goals; agent supervision will be weaker")

    def _verify_functions(self, plan: SemanticPlan, errors: list[str]) -> None:
        names = [function.get("name", "") for function in plan.functions]
        if not names:
            errors.append("PLAN requires at least one function")
        if len(names) != len(set(names)):
            errors.append("PLAN function names must be unique")
        for function in plan.functions:
            if not function.get("name") or not function.get("output") or not function.get("expression"):
                errors.append("PLAN functions require name, output and expression")

    def _verify_distributions(
        self, plan: SemanticPlan, errors: list[str], warnings: list[str]
    ) -> None:
        names = [distribution.get("name", "") for distribution in plan.distributions]
        raws = [distribution.get("raw", "") for distribution in plan.distributions]
        if plan.mode == "regression":
            if names:
                warnings.append("regression PLAN does not require distributions")
            return
        if not names:
            errors.append("PLAN requires at least one distribution")
            return
        if plan.mode == "classification" and not any("Categorical" in raw for raw in raws):
            errors.append("classification PLAN requires a Categorical distribution")
        if plan.mode == "risk" and not any("Normal" in raw for raw in raws):
            errors.append("risk PLAN requires a Normal distribution")
        if plan.mode == "risk" and "Risk" not in names:
            warnings.append("risk PLAN should expose a Risk distribution")

    def _verify_graph(self, plan: SemanticPlan, errors: list[str]) -> None:
        nodes = plan.graph.get("nodes", [])
        edges = plan.graph.get("edges", [])
        if not isinstance(nodes, list) or len(nodes) < 2:
            errors.append("PLAN graph requires at least two nodes")
            return
        node_set = set(nodes)
        for edge in edges:
            if len(edge) != 2:
                errors.append("PLAN graph edges must have source and target")
                continue
            source, target = edge
            if source not in node_set or target not in node_set:
                errors.append(f"PLAN graph edge references unknown node: {source} -> {target}")

    def _verify_actions(
        self, plan: SemanticPlan, errors: list[str], warnings: list[str]
    ) -> None:
        graph_nodes = set(plan.graph.get("nodes", []))
        if plan.mode == "regression":
            if plan.actions:
                warnings.append("regression PLAN should not expose simulated actions")
            return
        if not plan.actions:
            errors.append("PLAN requires at least one action")
            return

        for action in plan.actions:
            name = action.get("name", "")
            call = action.get("call", "")
            policy = action.get("policy", "")
            condition = action.get("when", "")

            if name not in graph_nodes:
                errors.append(f"PLAN action {name} is not present in graph")
            if policy != "simulate_only":
                errors.append(f"PLAN action {name} must use policy simulate_only in MVP")
            if not call.startswith("simulated."):
                errors.append(f"PLAN action {name} must call a simulated.* target in MVP")

            condition_match = self._CONDITION_RE.match(condition)
            if not condition_match:
                errors.append(f"PLAN action {name} has unsupported condition: {condition}")
                continue
            source = condition_match.group("source").split(".", 1)[0]
            threshold = float(condition_match.group("threshold"))
            if source not in graph_nodes:
                errors.append(f"PLAN action {name} condition source is not in graph: {source}")
            if not 0.0 <= threshold <= 1.0:
                errors.append(f"PLAN action {name} threshold must be between 0 and 1")

            if plan.mode == "risk" and not name.startswith(("Notify", "Alert")):
                warnings.append("risk PLAN action should be notification-oriented")

    def _verify_audit(self, plan: SemanticPlan, errors: list[str]) -> None:
        graph_nodes = set(plan.graph.get("nodes", []))
        audit_path = plan.audit_path
        if not audit_path:
            errors.append("PLAN requires an audit_path")
            return
        missing = [node for node in audit_path if node not in graph_nodes]
        if missing:
            errors.append(f"PLAN audit_path references unknown nodes: {', '.join(missing)}")
        vector_name = plan.vector.get("name")
        action_names = {action.get("name") for action in plan.actions}
        if audit_path[0] != vector_name:
            errors.append("PLAN audit_path must start at the vector node")
        if plan.mode == "regression":
            function_names = {function.get("name") for function in plan.functions}
            if audit_path[-1] not in function_names:
                errors.append("regression PLAN audit_path must end at a function node")
        elif audit_path[-1] not in action_names:
            errors.append("PLAN audit_path must end at an action node")

    def _verify_goal_rules(
        self, plan: SemanticPlan, errors: list[str], warnings: list[str]
    ) -> None:
        for rule in plan.verification_rules:
            check = rule.get("check")
            parameter = rule.get("parameter")
            goal = rule.get("goal")
            description = rule.get("description", "")

            if check == "action_threshold_min":
                for action in plan.actions:
                    m = self._CONDITION_RE.match(action.get("when", ""))
                    if m and float(m.group("threshold")) < float(parameter):
                        errors.append(
                            f"GOAL {goal}: {description} "
                            f"(got {m.group('threshold')}, required >= {parameter})"
                        )

            elif check == "action_threshold_max":
                for action in plan.actions:
                    m = self._CONDITION_RE.match(action.get("when", ""))
                    if m and float(m.group("threshold")) > float(parameter):
                        warnings.append(
                            f"GOAL {goal}: {description} "
                            f"(got {m.group('threshold')}, suggested <= {parameter})"
                        )

            elif check == "graph_nodes_max":
                node_count = len(plan.graph.get("nodes", []))
                if node_count > int(parameter):
                    warnings.append(
                        f"GOAL {goal}: {description} "
                        f"(got {node_count} nodes, suggested <= {parameter})"
                    )

            elif check == "distribution_required":
                raws = [d.get("raw", "") for d in plan.distributions]
                if not any(str(parameter) in raw for raw in raws):
                    errors.append(
                        f"GOAL {goal}: {description} "
                        f"(no {parameter} distribution found in plan)"
                    )

    def _verify_mathematical_translations(
        self, plan: SemanticPlan, warnings: list[str]
    ) -> None:
        unresolved = getattr(plan, "mathematical_unresolved", [])
        if unresolved:
            warnings.append(
                "PLAN has unresolved mathematical rule(s): " + ", ".join(unresolved)
            )
