# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptSynthesis:
    prompt: str
    semantic_text: str
    inferred_template: str = ""
    inferred_mode: str = ""
    inferred_entity: str = ""
    selected_fields: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    agent_chain: list[str] = field(default_factory=list)
    extracted_rules: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class PromptAgent:
    """Turns a human prompt into the structured semantic layer.

    This MVP agent is deterministic: it does not call an external LLM. Its job is
    to make the future prompt-first interface explicit while keeping the current
    test suite reproducible.
    """

    _TEMPLATES = {
        "email": {
            "project": "EmailAgent",
            "mode": "classification",
            "entity": "Email",
            "fields": [
                "urgency",
                "sender_trust",
                "topic_support",
                "topic_sales",
                "sentiment",
                "has_attachment",
                "previous_interactions",
                "language_confidence",
            ],
            "goals": ["classify_incoming_email", "minimize_false_replies"],
            "constraint_name": "confidence",
            "constraint_source": "Confidence.max",
            "constraint_threshold": 0.95,
            "action_name": "DraftReply",
            "action_call": "simulated.email.draft",
            "action_threshold": 0.90,
        },
        "pharmacy": {
            "project": "PharmacyDispense",
            "mode": "risk",
            "entity": "Order",
            "fields": [
                "urgency",
                "medication_risk",
                "stock_level",
                "patient_priority",
                "interaction_risk",
                "time_limit",
            ],
            "goals": ["maximize_safety", "minimize_latency"],
            "constraint_name": "risk",
            "constraint_source": "Risk.mean",
            "constraint_threshold": 0.75,
            "action_name": "Dispense",
            "action_call": "simulated.pharmacy.dispense",
            "action_threshold": 0.90,
        },
        "fall_risk": {
            "project": "FallRisk",
            "mode": "risk",
            "entity": "Patient",
            "fields": [
                "age",
                "mobility",
                "medication_load",
                "previous_falls",
                "cognitive_state",
            ],
            "goals": ["minimize_fall_incidents", "minimize_false_alerts"],
            "constraint_name": "risk",
            "constraint_source": "Risk.mean",
            "constraint_threshold": 0.80,
            "action_name": "Notify",
            "action_call": "simulated.nurse_station.alert",
            "action_threshold": 0.90,
        },
        "generic_risk": {
            "project": "PromptGeneratedRiskAgent",
            "mode": "risk",
            "entity": "Signal",
            "fields": ["severity", "confidence", "history", "context"],
            "goals": ["maximize_safety"],
            "constraint_name": "risk",
            "constraint_source": "Risk.mean",
            "constraint_threshold": 0.80,
            "action_name": "Notify",
            "action_call": "simulated.notification.alert",
            "action_threshold": 0.90,
        },
        "generic_classification": {
            "project": "PromptGeneratedClassifier",
            "mode": "classification",
            "entity": "Item",
            "fields": ["priority", "trust", "topic_a", "topic_b", "confidence"],
            "goals": ["maximize_precision"],
            "constraint_name": "confidence",
            "constraint_source": "Confidence.max",
            "constraint_threshold": 0.90,
            "action_name": "DraftAction",
            "action_call": "simulated.workflow.draft",
            "action_threshold": 0.90,
        },
        "generic_regression": {
            "project": "PromptGeneratedRegressor",
            "mode": "regression",
            "entity": "Item",
            "fields": ["input_1"],
            "goals": ["minimize_prediction_error"],
            "output_name": "predicted_value",
        },
    }

    _REGRESSION_KEYWORDS = [
        "precio", "price",
        "predecir",
        "estim", "regres",
        "kelvin", "celsius", "centigrad",
        "convertir", "convert",
        "temperatura", "temperature",
        "consumo", "duracion",
    ]

    _EXACT_FORMULA_HINTS = [
        "celsius", "centigrad", "kelvin", "convertir", "convert", "formula", "formula exacta",
    ]

    _FIELD_HINT_RE = re.compile(
        r"(?:campos|fields|variables|features)\s*(?::|=|son)?\s*(?P<fields>[^.;\n]+)",
        re.IGNORECASE,
    )
    _PROJECT_RE = re.compile(r"\b(?:project|proyecto)\s*[:=]?\s*(?P<name>[A-Za-z_][\w]*)", re.IGNORECASE)
    _ENTITY_RE = re.compile(r"\b(?:entity|entidad)\s*[:=]?\s*(?P<name>[A-Za-z_][\w]*)", re.IGNORECASE)
    _RULE_RE = re.compile(
        r"\b(?:if|si)\s+(?P<rule>.*?(?:then|entonces)\s+[A-Za-z_][\w]*)",
        re.IGNORECASE,
    )
    _COMPARISON_RE = re.compile(
        r"[A-Za-z_][\w.]*\s*(?:>=|<=|>|<)\s*[0-9]+(?:\.[0-9]+)?"
    )
    _COMPARISON_DETAIL_RE = re.compile(
        r"(?P<var>[A-Za-z_][\w.]*)\s*(?P<op>>=|<=|>|<)\s*(?P<num>[0-9]+(?:\.[0-9]+)?)"
    )

    def synthesize(self, prompt: str) -> PromptSynthesis:
        clean_prompt = " ".join(prompt.strip().split())
        if not clean_prompt:
            raise ValueError("PromptAgent requires a non-empty prompt")

        template_name = self._select_template(clean_prompt)
        template = self._TEMPLATES[template_name]
        trace: list[dict[str, Any]] = []
        self._append_trace(trace, "prompt_received", characters=len(clean_prompt))
        self._append_trace(trace, "template_selected", template=template_name)

        resolved = dict(template)
        resolved["project"] = self._extract_named_identifier(
            clean_prompt, self._PROJECT_RE, template["project"], title_case=True
        )
        resolved["entity"] = self._extract_named_identifier(
            clean_prompt, self._ENTITY_RE, template["entity"], title_case=True
        )
        resolved["fields"] = self._extract_fields(clean_prompt) or list(template["fields"])
        if template_name == "generic_regression":
            self._specialize_regression_template(clean_prompt, resolved)
        self._append_trace(
            trace,
            "structure_extracted",
            project=resolved["project"],
            entity=resolved["entity"],
            fields=resolved["fields"],
        )

        is_regression = resolved.get("mode") == "regression"
        goals = self._infer_goals(clean_prompt, resolved)
        self._append_trace(trace, "goals_inferred", goals=goals)

        if is_regression:
            rules: list[str] = []
            normalized_lower = _normalize_ascii(clean_prompt).lower()
            is_exact_formula = self._contains_any(normalized_lower, self._EXACT_FORMULA_HINTS)
            semantic_text = self._semantic_text_regression(
                template=resolved,
                intent=clean_prompt,
                goals=goals,
            )
            self._append_trace(trace, "semantic_emitted", lines=len(semantic_text.splitlines()))
            assumptions = [
                f"PromptAgent inferred template={template_name}",
                "PromptAgent inferred mode=regression",
                f"PromptAgent inferred entity={resolved['entity']}",
                "linear_regression pipeline: loss=mse, target=Scalar, metrics=mae/rmse/r2",
                "generate-supervised and train-supervised produce a trainable package (not an algebraic solver)",
            ]
            if is_exact_formula:
                assumptions.append(
                    "This prompt implies a deterministic formula — the pipeline still generates a trainable package"
                )
            if resolved["project"] == "CelsiusToKelvin":
                assumptions.append(
                    "Celsius/Kelvin intent mapped to Reading.celsius -> predicted_kelvin for auditable training"
                )
            return PromptSynthesis(
                prompt=clean_prompt,
                semantic_text=semantic_text,
                inferred_template=template_name,
                inferred_mode="regression",
                inferred_entity=resolved["entity"],
                selected_fields=resolved["fields"],
                goals=goals,
                assumptions=assumptions,
                agent_chain=[
                    "PromptAgent",
                    "ArchitectAgent",
                    "MathematicalAgent",
                    "PlannerVerifier",
                    "VerifierAgent",
                ],
                extracted_rules=[],
                trace=trace,
            )

        constraint_threshold = self._extract_constraint_threshold(clean_prompt, template)
        action_threshold = self._extract_action_threshold(clean_prompt, template)
        default_rule = (
            f"if {resolved['constraint_source']} > "
            f"{constraint_threshold:.2f} then {resolved['action_name']}"
        )
        rules = list(
            dict.fromkeys([default_rule, *self._extract_rules(clean_prompt, resolved["action_name"])])
        )
        self._append_trace(trace, "rules_extracted", rules=rules)

        semantic_text = self._semantic_text(
            template=resolved,
            intent=clean_prompt,
            goals=goals,
            constraint_threshold=constraint_threshold,
            action_threshold=action_threshold,
            rules=rules,
        )
        self._append_trace(trace, "semantic_emitted", lines=len(semantic_text.splitlines()))
        assumptions = [
            f"PromptAgent inferred template={template_name}",
            f"PromptAgent inferred mode={resolved['mode']}",
            f"PromptAgent inferred entity={resolved['entity']}",
            "SafetyAgent boundary: generated action is simulate_only",
            "MathematicalAgent will translate the extracted rule into a continuous activation",
        ]
        return PromptSynthesis(
            prompt=clean_prompt,
            semantic_text=semantic_text,
            inferred_template=template_name,
            inferred_mode=resolved["mode"],
            inferred_entity=resolved["entity"],
            selected_fields=resolved["fields"],
            goals=goals,
            assumptions=assumptions,
            agent_chain=[
                "PromptAgent",
                "ArchitectAgent",
                "MathematicalAgent",
                "PlannerVerifier",
                "VerifierAgent",
                "SafetyAgent",
            ],
            extracted_rules=rules,
            trace=trace,
        )

    def _select_template(self, prompt: str) -> str:
        text = _normalize_ascii(prompt).lower()
        if self._contains_any(text, ["email", "correo", "correos", "reply", "respuesta"]):
            return "email"
        if self._contains_any(
            text,
            ["pharmacy", "farmacia", "dispens", "medication", "medicamento", "pedido"],
        ):
            return "pharmacy"
        if self._contains_any(text, ["fall", "caida", "patient", "paciente", "nurse"]):
            return "fall_risk"
        if self._contains_any(text, self._REGRESSION_KEYWORDS):
            return "generic_regression"
        if self._contains_any(text, ["classify", "clasifica", "clasificar", "categoriza"]):
            return "generic_classification"
        return "generic_risk"

    def _specialize_regression_template(self, prompt: str, resolved: dict[str, Any]) -> None:
        text = _normalize_ascii(prompt).lower()
        if "kelvin" not in text:
            return
        if not self._contains_any(text, ["celsius", "centigrad"]):
            return
        resolved["project"] = "CelsiusToKelvin"
        resolved["entity"] = "Reading"
        resolved["fields"] = ["celsius"]
        resolved["output_name"] = "predicted_kelvin"

    def _infer_goals(self, prompt: str, template: dict) -> list[str]:
        text = prompt.lower()
        goals = list(template["goals"])
        if self._contains_any(text, ["seguridad", "safety", "seguro"]):
            goals.append("maximize_safety")
        if self._contains_any(text, ["latencia", "rapido", "rapida", "latency"]):
            goals.append("minimize_latency")
        if self._contains_any(text, ["falsas alarmas", "false alerts"]):
            goals.append("minimize_false_alerts")
        if self._contains_any(text, ["falsas respuestas", "false replies"]):
            goals.append("minimize_false_replies")
        if self._contains_any(text, ["precision", "preciso", "precisa"]):
            goals.append("maximize_precision")
        return list(dict.fromkeys(goals))

    def _extract_constraint_threshold(self, prompt: str, template: dict) -> float:
        text = prompt.lower()
        names = [template["constraint_name"]]
        if template["constraint_name"] == "confidence":
            names.extend(["confianza", "confidence"])
        else:
            names.extend(["riesgo", "risk"])
        return self._extract_threshold_after_names(text, names, template["constraint_threshold"])

    def _extract_action_threshold(self, prompt: str, template: dict) -> float:
        text = prompt.lower()
        return self._extract_threshold_after_names(
            text,
            ["accion", "action", "activation", "activacion"],
            template["action_threshold"],
        )

    def _extract_threshold_after_names(
        self, text: str, names: list[str], default: float
    ) -> float:
        name_pattern = "|".join(re.escape(name) for name in names)
        pattern = re.compile(rf"(?:{name_pattern})[^0-9%]*(?P<number>[0-9]+(?:\.[0-9]+)?%?)")
        match = pattern.search(text)
        if not match:
            return default
        value = match.group("number")
        if value.endswith("%"):
            return float(value[:-1]) / 100.0
        number = float(value)
        if number > 1.0 and number <= 100.0:
            return number / 100.0
        return number

    def _extract_named_identifier(
        self,
        prompt: str,
        pattern: re.Pattern[str],
        default: str,
        *,
        title_case: bool = False,
    ) -> str:
        match = pattern.search(prompt)
        if not match:
            return default
        return self._identifier(match.group("name"), default=default, title_case=title_case)

    def _extract_fields(self, prompt: str) -> list[str]:
        match = self._FIELD_HINT_RE.search(prompt)
        if not match:
            return []
        raw_fields = match.group("fields")
        parts = re.split(r",|;|\s+y\s+|\s+and\s+", raw_fields, flags=re.IGNORECASE)
        fields = [self._identifier(part, default="field") for part in parts]
        fields = [field for field in fields if field and field != "field"]
        return list(dict.fromkeys(fields)) if len(fields) >= 2 else []

    def _extract_rules(self, prompt: str, action_name: str) -> list[str]:
        rules: list[str] = []
        for match in self._RULE_RE.finditer(prompt):
            normalized = self._normalize_rule(match.group("rule"), action_name)
            if normalized:
                rules.append(normalized)
        return list(dict.fromkeys(rules))

    def _normalize_rule(self, rule: str, action_name: str) -> str:
        text = self._ascii(rule).strip()
        text = re.sub(r"\bentonces\b", "then", text, flags=re.IGNORECASE)
        text = re.sub(r"\by\b", "and", text, flags=re.IGNORECASE)
        text = re.sub(r"\bo\b", "or", text, flags=re.IGNORECASE)
        text = self._normalize_percentage_literals(text)
        text = self._normalize_spanish_comparisons(text)
        text = re.sub(r"\b(?:la|el|las|los|un|una)\s+(?=[A-Za-z_])", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        comparisons = list(self._COMPARISON_DETAIL_RE.finditer(text))
        if not comparisons:
            return ""
        action = self._extract_rule_action(text, action_name)
        first = comparisons[0]
        conditions = [self._condition_text(first)]
        if len(comparisons) > 1:
            second = comparisons[1]
            between = text[first.end() : second.start()].lower()
            connector = "or" if " or " in f" {between} " else "and"
            conditions.append(connector)
            conditions.append(self._condition_text(second))
        return f"if {' '.join(conditions)} then {action}"

    def _condition_text(self, match: re.Match[str]) -> str:
        return f"{match.group('var')} {match.group('op')} {match.group('num')}"

    def _extract_rule_action(self, text: str, action_name: str) -> str:
        action_match = re.search(r"\bthen\s+(?P<action>[A-Za-z_][\w]*)", text, re.IGNORECASE)
        if not action_match:
            return action_name
        return self._identifier(action_match.group("action"), default=action_name, title_case=True)

    def _normalize_percentage_literals(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            value = float(match.group("number")) / 100.0
            return f"{value:.2f}"

        return re.sub(r"(?P<number>[0-9]+(?:\.[0-9]+)?)%", replace, text)

    def _normalize_spanish_comparisons(self, text: str) -> str:
        variable = r"(?P<var>[A-Za-z_][\w.]*)"
        number = r"(?P<num>[0-9]+(?:\.[0-9]+)?)"
        greater = re.compile(
            rf"{variable}\s+(?:supera|mayor\s+que|por\s+encima\s+de)\s+(?:el\s+)?{number}",
            re.IGNORECASE,
        )
        lower = re.compile(
            rf"{variable}\s+(?:menor\s+que|por\s+debajo\s+de)\s+(?:el\s+)?{number}",
            re.IGNORECASE,
        )
        text = greater.sub(lambda m: f"{m.group('var')} > {m.group('num')}", text)
        return lower.sub(lambda m: f"{m.group('var')} < {m.group('num')}", text)

    def _semantic_text_regression(
        self,
        *,
        template: dict,
        intent: str,
        goals: list[str],
    ) -> str:
        fields = "\n".join(f"  {fld}" for fld in template["fields"])
        goals_text = "\n".join(f"GOAL {goal}" for goal in goals)
        output_name = template.get("output_name", "predicted_value")
        return f"""PROJECT {template['project']}
INTENT {intent}
MODE regression
ENTITY {template['entity']}

FIELDS {template['entity']}
{fields}
END

{goals_text}
OUTPUT {output_name}: Scalar
LOSS mse
METRIC mae
""".strip() + "\n"

    def _semantic_text(
        self,
        *,
        template: dict,
        intent: str,
        goals: list[str],
        constraint_threshold: float,
        action_threshold: float,
        rules: list[str],
    ) -> str:
        fields = "\n".join(f"  {field}" for field in template["fields"])
        goals_text = "\n".join(f"GOAL {goal}" for goal in goals)
        rules_text = "\n".join(f"  {rule}" for rule in rules)
        return f"""PROJECT {template['project']}
INTENT {intent}
MODE {template['mode']}
ENTITY {template['entity']}

FIELDS {template['entity']}
{fields}
END

{goals_text}
CONSTRAINT {template['constraint_name']} > {constraint_threshold:.2f}
ACTION_THRESHOLD {action_threshold:.2f}

RULES
{rules_text}
END

ACTION {template['action_name']}
  POLICY simulate_only
  CALL {template['action_call']}
END
""".strip() + "\n"

    def _contains_any(self, text: str, needles: list[str]) -> bool:
        return any(needle in text for needle in needles)

    def _identifier(self, value: str, *, default: str, title_case: bool = False) -> str:
        text = self._ascii(value).strip()
        text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            return default
        if title_case:
            text = text[:1].upper() + text[1:]
        else:
            text = text.lower()
        if text[0].isdigit():
            text = f"{default}_{text}"
        return text

    def _ascii(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return normalized.encode("ascii", "ignore").decode("ascii")

    def _append_trace(self, trace: list[dict[str, Any]], event: str, **data: Any) -> None:
        trace.append({"step": len(trace) + 1, "agent": "PromptAgent", "event": event, **data})


def _normalize_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")
