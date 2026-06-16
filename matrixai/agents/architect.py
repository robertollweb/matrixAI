# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai.agents.mathematical import MathematicalAgent
from matrixai.ir.schema import VerificationRule


class ArchitectSpecError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticAction:
    name: str
    call: str
    policy: str = "simulate_only"


@dataclass(frozen=True)
class SemanticSpec:
    project: str
    intent: str
    mode: str
    entity: str
    fields: list[str]
    goals: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    output_name: str = ""
    output_type: str = "Scalar"
    loss: str = ""
    metric: str = ""
    confidence_threshold: float = 0.95
    action_threshold: float = 0.9
    action: SemanticAction = field(
        default_factory=lambda: SemanticAction(name="DraftReply", call="simulated.email.draft")
    )


@dataclass(frozen=True)
class SemanticPlan:
    project: str
    mode: str
    intent: str
    goals: list[str]
    vector: dict[str, Any]
    parameters: list[dict[str, Any]]
    functions: list[dict[str, Any]]
    distributions: list[dict[str, Any]]
    graph: dict[str, Any]
    actions: list[dict[str, Any]]
    audit_path: list[str]
    verification_rules: list[dict[str, Any]] = field(default_factory=list)
    mathematical_translations: list[dict[str, Any]] = field(default_factory=list)
    mathematical_unresolved: list[str] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GoalTranslator:
    """Converts GOAL declarations into VerificationRules."""

    _RULES: dict[str, VerificationRule] = {
        "minimize_false_alerts": VerificationRule(
            goal="minimize_false_alerts",
            description="action threshold must be >= 0.85 to reduce false alerts",
            check="action_threshold_min",
            parameter=0.85,
        ),
        "minimize_false_replies": VerificationRule(
            goal="minimize_false_replies",
            description="action threshold must be >= 0.85 to reduce false replies",
            check="action_threshold_min",
            parameter=0.85,
        ),
        "maximize_precision": VerificationRule(
            goal="maximize_precision",
            description="action threshold must be >= 0.85 to maximize precision",
            check="action_threshold_min",
            parameter=0.85,
        ),
        "maximize_safety": VerificationRule(
            goal="maximize_safety",
            description="action threshold must be >= 0.90 to maximize safety",
            check="action_threshold_min",
            parameter=0.90,
        ),
        "maximize_recall": VerificationRule(
            goal="maximize_recall",
            description="action threshold should be <= 0.75 to maximize recall",
            check="action_threshold_max",
            parameter=0.75,
        ),
        "minimize_latency": VerificationRule(
            goal="minimize_latency",
            description="graph must have at most 6 nodes to minimize latency",
            check="graph_nodes_max",
            parameter=6,
        ),
        "minimize_fall_incidents": VerificationRule(
            goal="minimize_fall_incidents",
            description="risk mode requires Normal distribution for uncertainty",
            check="distribution_required",
            parameter="Normal",
        ),
        "classify_incoming_email": VerificationRule(
            goal="classify_incoming_email",
            description="classification mode requires Categorical distribution",
            check="distribution_required",
            parameter="Categorical",
        ),
    }

    def translate(self, goals: list[str]) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        for goal in goals:
            key = goal.strip().lower().replace(" ", "_")
            rule = self._RULES.get(key)
            if rule is not None:
                rules.append(asdict(rule))
        return rules


class ArchitectAgent:
    def from_text(self, text: str) -> str:
        plan = self.plan_from_text(text)
        return self.to_mxai(plan)

    def plan_from_text(self, text: str) -> SemanticPlan:
        spec = self.parse_semantic_spec(text)
        return self.to_plan(spec)

    def parse_semantic_spec(self, text: str) -> SemanticSpec:
        lines = self._clean_lines(text)
        if not lines:
            raise ArchitectSpecError("Empty semantic spec")

        project = ""
        intent = ""
        mode = ""
        entity = ""
        fields: list[str] = []
        goals: list[str] = []
        rules: list[str] = []
        output_name = ""
        output_type = "Scalar"
        loss = ""
        metric = ""
        confidence_threshold = 0.95
        action_threshold = 0.9
        action_name = ""
        action_call = ""
        action_policy = "simulate_only"

        index = 0
        while index < len(lines):
            line = lines[index]
            keyword = line.split(maxsplit=1)[0]

            if keyword == "PROJECT":
                project = self._value_after_keyword(line, "PROJECT")
                index += 1
                continue

            if keyword == "INTENT":
                intent = self._value_after_keyword(line, "INTENT")
                index += 1
                continue

            if keyword == "MODE":
                mode = self._value_after_keyword(line, "MODE")
                index += 1
                continue

            if keyword == "ENTITY":
                entity = self._value_after_keyword(line, "ENTITY")
                index += 1
                continue

            if keyword == "FIELDS":
                block, index = self._read_block(lines, index)
                fields = block[1:-1]
                continue

            if keyword == "GOAL":
                goals.append(self._value_after_keyword(line, "GOAL"))
                index += 1
                continue

            if keyword == "RULE":
                rules.append(self._value_after_keyword(line, "RULE"))
                index += 1
                continue

            if keyword == "RULES":
                block, index = self._read_block(lines, index)
                rules.extend(block[1:-1])
                continue

            if keyword == "OUTPUT":
                output_name, output_type = self._parse_output_line(
                    self._value_after_keyword(line, "OUTPUT")
                )
                index += 1
                continue

            if keyword == "LOSS":
                loss = self._value_after_keyword(line, "LOSS").lower()
                index += 1
                continue

            if keyword == "METRIC":
                metric = self._value_after_keyword(line, "METRIC").lower()
                index += 1
                continue

            if keyword == "CONSTRAINT":
                confidence_threshold = self._parse_confidence_threshold(
                    self._value_after_keyword(line, "CONSTRAINT")
                )
                index += 1
                continue

            if keyword == "ACTION_THRESHOLD":
                action_threshold = float(self._value_after_keyword(line, "ACTION_THRESHOLD"))
                index += 1
                continue

            if keyword == "ACTION":
                block, index = self._read_block(lines, index)
                action_name, action_call, action_policy = self._parse_action_block(block)
                continue

            raise ArchitectSpecError(f"Unknown semantic spec line: {line}")

        if not project:
            raise ArchitectSpecError("Semantic spec requires PROJECT")
        if not entity:
            raise ArchitectSpecError("Semantic spec requires ENTITY")
        if not fields:
            raise ArchitectSpecError("Semantic spec requires FIELDS")
        resolved_mode = self._resolve_mode(mode, intent, action_name)
        if resolved_mode == "regression":
            if not output_name:
                raise ArchitectSpecError("Regression semantic spec requires OUTPUT")
            if loss and loss != "mse":
                raise ArchitectSpecError("Regression semantic spec supports only LOSS mse")
            if metric and metric not in {"mae", "rmse", "r2"}:
                raise ArchitectSpecError("Regression semantic spec supports METRIC mae, rmse or r2")
        elif not action_name or not action_call:
            raise ArchitectSpecError("Semantic spec requires ACTION with CALL")

        return SemanticSpec(
            project=project,
            intent=intent,
            mode=resolved_mode,
            entity=entity,
            fields=fields,
            goals=goals,
            rules=rules,
            output_name=output_name,
            output_type=output_type,
            loss=loss,
            metric=metric,
            confidence_threshold=confidence_threshold,
            action_threshold=action_threshold,
            action=SemanticAction(name=action_name, call=action_call, policy=action_policy),
        )

    def to_plan(self, spec: SemanticSpec) -> SemanticPlan:
        if spec.mode == "classification":
            return self._classification_plan(spec)
        if spec.mode == "risk":
            return self._risk_plan(spec)
        if spec.mode == "regression":
            return self._regression_plan(spec)
        raise ArchitectSpecError(f"Unsupported MODE: {spec.mode}")

    def to_mxai(self, plan: SemanticPlan) -> str:
        vector_fields = "\n".join(f"  {field}" for field in plan.vector["fields"])
        parameter_blocks = "\n\n".join(
            f"PARAM {parameter['name']} {parameter['type']}\nEND"
            for parameter in plan.parameters
        )
        function_blocks = "\n\n".join(
            f"FUNCTION {function['name']}\n  {function['output']} = {function['expression']}\nEND"
            for function in plan.functions
        )
        distribution_blocks = "\n\n".join(
            f"DISTRIBUTION {distribution['name']}\n  {distribution['raw']}\nEND"
            for distribution in plan.distributions
        )
        graph_chain = " -> ".join(plan.graph["nodes"])
        action_blocks = "\n\n".join(
            f"ACTION {action['name']}\n"
            f"  WHEN {action['when']}\n"
            f"  POLICY {action['policy']}\n"
            f"  CALL {action['call']}\n"
            "END"
            for action in plan.actions
        )
        audit_chain = " -> ".join(plan.audit_path)
        parts = [
            f"""PROJECT {plan.project}

VECTOR {plan.vector['name']}[{len(plan.vector['fields'])}]
{vector_fields}
END""",
        ]
        if parameter_blocks:
            parts.append(parameter_blocks)
        if function_blocks:
            parts.append(function_blocks)
        if distribution_blocks:
            parts.append(distribution_blocks)
        parts.append(f"""GRAPH
  {graph_chain}
END""")
        if action_blocks:
            parts.append(action_blocks)
        parts.append(f"""AUDIT
  EXPLAIN {audit_chain}
END""")
        return "\n\n".join(parts).strip() + "\n"

    def _classification_plan(self, spec: SemanticSpec) -> SemanticPlan:
        activation_name = self._activation_name(spec.action.name)
        math_translations, unresolved = self._activation_from_rule(
            source="Confidence.max",
            threshold=spec.confidence_threshold,
            action_name=spec.action.name,
            extra_rules=spec.rules,
        )
        activation_functions, activation_nodes, lineage = self._activation_graph(
            activation_name, math_translations
        )
        nodes = [spec.entity, "Classifier", "Confidence", *activation_nodes, spec.action.name]
        return SemanticPlan(
            project=spec.project,
            mode=spec.mode,
            intent=spec.intent,
            goals=spec.goals,
            vector={"name": spec.entity, "fields": spec.fields},
            parameters=[],
            functions=[
                {
                    "name": "Classifier",
                    "output": "C",
                    "expression": f"softmax(W1 * {spec.entity} + b1)",
                },
                *activation_functions,
            ],
            distributions=[
                {"name": "Confidence", "raw": "Confidence ~ Categorical(C)"},
            ],
            graph={"nodes": nodes, "edges": list(zip(nodes, nodes[1:]))},
            actions=[
                {
                    "name": spec.action.name,
                    "when": f"{activation_name} > {spec.action_threshold:.2f}",
                    "policy": spec.action.policy,
                    "call": spec.action.call,
                }
            ],
            audit_path=nodes,
            verification_rules=GoalTranslator().translate(spec.goals),
            mathematical_translations=math_translations,
            mathematical_unresolved=unresolved,
            lineage=lineage,
        )

    def _risk_plan(self, spec: SemanticSpec) -> SemanticPlan:
        activation_name = self._activation_name(spec.action.name)
        math_translations, unresolved = self._activation_from_rule(
            source="Risk.mean",
            threshold=spec.confidence_threshold,
            action_name=spec.action.name,
            extra_rules=spec.rules,
        )
        activation_functions, activation_nodes, lineage = self._activation_graph(
            activation_name, math_translations
        )
        nodes = [spec.entity, "RiskModel", "Risk", *activation_nodes, spec.action.name]
        return SemanticPlan(
            project=spec.project,
            mode=spec.mode,
            intent=spec.intent,
            goals=spec.goals,
            vector={"name": spec.entity, "fields": spec.fields},
            parameters=[],
            functions=[
                {
                    "name": "RiskModel",
                    "output": "R",
                    "expression": f"sigmoid(W1 * {spec.entity} + b1)",
                },
                *activation_functions,
            ],
            distributions=[
                {"name": "Risk", "raw": f"Risk ~ Normal(R, uncertainty({spec.entity}))"},
            ],
            graph={"nodes": nodes, "edges": list(zip(nodes, nodes[1:]))},
            actions=[
                {
                    "name": spec.action.name,
                    "when": f"{activation_name} > {spec.action_threshold:.2f}",
                    "policy": spec.action.policy,
                    "call": spec.action.call,
                }
            ],
            audit_path=nodes,
            verification_rules=GoalTranslator().translate(spec.goals),
            mathematical_translations=math_translations,
            mathematical_unresolved=unresolved,
            lineage=lineage,
        )

    def _regression_plan(self, spec: SemanticSpec) -> SemanticPlan:
        output_name = spec.output_name or "predicted_value"
        function_name = self._regression_function_name(output_name)
        nodes = [spec.entity, function_name]
        return SemanticPlan(
            project=spec.project,
            mode=spec.mode,
            intent=spec.intent,
            goals=spec.goals,
            vector={"name": spec.entity, "fields": spec.fields},
            parameters=[
                {"name": "W1", "type": f"Vector[{len(spec.fields)}]"},
                {"name": "b1", "type": "Scalar"},
            ],
            functions=[
                {
                    "name": function_name,
                    "output": f"{output_name}: {spec.output_type or 'Scalar'}",
                    "expression": f"linear(W1 * {spec.entity} + b1)",
                },
            ],
            distributions=[],
            graph={"nodes": nodes, "edges": list(zip(nodes, nodes[1:]))},
            actions=[],
            audit_path=nodes,
            verification_rules=GoalTranslator().translate(spec.goals),
            mathematical_translations=[],
            mathematical_unresolved=[],
            lineage=[
                {
                    "agent": "ArchitectAgent",
                    "source": "semantic_regression_output",
                    "output": output_name,
                    "expression_kind": "linear_regression",
                    "used_in_graph": True,
                }
            ],
        )

    def _lineage(
        self,
        translations: list[dict[str, Any]],
        node_by_index: dict[int, str],
        output_by_index: dict[int, str],
    ) -> list[dict[str, Any]]:
        lineage: list[dict[str, Any]] = []
        for index, translation in enumerate(translations):
            used = index in node_by_index
            lineage.append(
                {
                    "agent": "MathematicalAgent",
                    "source": "semantic_rule",
                    "rule": translation["original_rule"],
                    "expression": translation["expression"],
                    "expression_kind": translation["expression_kind"],
                    "node": node_by_index.get(index),
                    "output": output_by_index.get(index),
                    "used_in_graph": used,
                }
            )
        return lineage

    def _activation_graph(
        self, activation_name: str, translations: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        evidence_indexes = [
            index
            for index, translation in enumerate(translations)
            if self._is_action_evidence(translation)
        ]
        if not evidence_indexes:
            raise ArchitectSpecError("No mathematical rule can be used as action evidence")

        if len(evidence_indexes) == 1:
            index = evidence_indexes[0]
            expression = self._node_expression(translations[index])
            functions = [{"name": activation_name, "output": "A", "expression": expression}]
            node_by_index = {index: activation_name}
            output_by_index = {index: "A"}
            return functions, [activation_name], self._lineage(
                translations, node_by_index, output_by_index
            )

        functions: list[dict[str, Any]] = []
        activation_nodes: list[str] = []
        node_by_index: dict[int, str] = {}
        output_by_index: dict[int, str] = {}

        for order, index in enumerate(evidence_indexes):
            node_name = f"{activation_name}Base" if order == 0 else f"{activation_name}Rule{order + 1}"
            expression = self._node_expression(translations[index])
            functions.append({"name": node_name, "output": node_name, "expression": expression})
            activation_nodes.append(node_name)
            node_by_index[index] = node_name
            output_by_index[index] = node_name

        functions.append(
            {
                "name": activation_name,
                "output": "A",
                "expression": f"max({', '.join(activation_nodes)})",
            }
        )
        activation_nodes.append(activation_name)
        return functions, activation_nodes, self._lineage(
            translations, node_by_index, output_by_index
        )

    def _is_action_evidence(self, translation: dict[str, Any]) -> bool:
        return translation.get("expression_kind") not in {"softmax_linear", "select_argmax"}

    def _node_expression(self, translation: dict[str, Any]) -> str:
        expression = str(translation["expression"])
        if " = " in expression and translation.get("expression_kind") in {
            "symbolic_expr",
            "symbolic_weighted_sum",
        }:
            return expression.split("=", 1)[1].strip()
        return expression

    def _activation_from_rule(
        self,
        *,
        source: str,
        threshold: float,
        action_name: str,
        extra_rules: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        generated_rule = f"if {source} > {threshold:.2f} then {action_name}"
        rules = list(dict.fromkeys([generated_rule, *extra_rules]))
        report = MathematicalAgent().translate(rules)
        if not report.translations:
            raise ArchitectSpecError(f"Could not mathematize activation rule: {generated_rule}")
        return (
            [asdict(translation) for translation in report.translations],
            report.unresolved,
        )

    def _clean_lines(self, text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines

    def _value_after_keyword(self, line: str, keyword: str) -> str:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ArchitectSpecError(f"{keyword} requires a value")
        return parts[1].strip()

    def _read_block(self, lines: list[str], start: int) -> tuple[list[str], int]:
        block = [lines[start]]
        index = start + 1
        while index < len(lines):
            block.append(lines[index])
            if lines[index] == "END":
                return block, index + 1
            index += 1
        raise ArchitectSpecError(f"Block '{lines[start]}' is missing END")

    def _parse_confidence_threshold(self, constraint: str) -> float:
        parts = constraint.split()
        if len(parts) == 3 and parts[0] in {"confidence", "risk"} and parts[1] in {">", ">="}:
            return float(parts[2])
        raise ArchitectSpecError("Supported constraint format: confidence > 0.95 or risk > 0.80")

    def _parse_output_line(self, value: str) -> tuple[str, str]:
        name, sep, type_name = value.partition(":")
        output_name = name.strip()
        output_type = type_name.strip() if sep else "Scalar"
        if not output_name:
            raise ArchitectSpecError("OUTPUT requires a name")
        if output_type not in {"Scalar", "Integer"}:
            raise ArchitectSpecError("Regression OUTPUT must be Scalar or Integer")
        return output_name, output_type

    def _parse_action_block(self, block: list[str]) -> tuple[str, str, str]:
        action_name = self._value_after_keyword(block[0], "ACTION")
        call = ""
        policy = "simulate_only"
        for line in block[1:-1]:
            if line.startswith("CALL "):
                call = self._value_after_keyword(line, "CALL")
            elif line.startswith("POLICY "):
                policy = self._value_after_keyword(line, "POLICY")
        return action_name, call, policy

    def _activation_name(self, action_name: str) -> str:
        if action_name.startswith("Draft") and len(action_name) > len("Draft"):
            return f"{action_name.removeprefix('Draft')}Activation"
        if action_name.startswith("Notify"):
            return "AlertActivation"
        return f"{action_name}Activation"

    def _regression_function_name(self, output_name: str) -> str:
        words = [part for part in output_name.split("_") if part]
        stem = "".join(word[:1].upper() + word[1:] for word in words) or "Value"
        if stem.lower().startswith("predicted"):
            return f"{stem}Model"
        return f"{stem}Prediction"

    def _resolve_mode(self, mode: str, intent: str, action_name: str) -> str:
        normalized = mode.lower().strip()
        if normalized in {"classification", "risk", "regression"}:
            return normalized
        if normalized:
            raise ArchitectSpecError("MODE must be classification, risk or regression")
        lower_intent = intent.lower()
        if any(token in lower_intent for token in ("regression", "regresion", "predecir", "predict")):
            return "regression"
        if "risk" in lower_intent or "riesgo" in lower_intent or action_name.startswith("Notify"):
            return "risk"
        return "classification"
