# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import math
from typing import Any

from matrixai.ir import DistributionSpec, FunctionSpec, MatrixAIProgram, VectorSpec
from matrixai.types import validate_value_against_type


class MatrixAIRuntime:
    def run(
        self,
        program: MatrixAIProgram,
        input_data: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = parameters or {}
        state: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []

        vectors = {vector.name: vector for vector in program.vectors}
        functions = {function.name: function for function in program.functions}
        distributions = {distribution.name: distribution for distribution in program.distributions}
        action_specs = {action.name: action for action in program.actions}
        networks_map = {net.name: net for net in getattr(program, "networks", [])}

        for node in program.graph.nodes:
            if node in vectors:
                value = self._load_vector(vectors[node], input_data)
                state[node] = value
                # Also expose individual field values for use in symbolic expressions
                for fname, fval in zip(vectors[node].fields, value):
                    state.setdefault(fname, fval)
                trace.append(
                    self._trace_step(
                        trace,
                        node=node,
                        node_type="vector",
                        value=value,
                        output_ref=node,
                    )
                )
            elif node in functions:
                value = self._evaluate_function(functions[node], state, parameters)
                state[node] = value
                state[functions[node].output] = value
                trace.append(
                    self._trace_step(
                        trace,
                        node=node,
                        node_type="function",
                        value=value,
                        output_ref=functions[node].output,
                        expression_kind=functions[node].semantic.kind,
                        inputs=functions[node].semantic.inputs,
                    )
                )
            elif node in distributions:
                value = self._evaluate_distribution(distributions[node], state)
                state[node] = value
                state[distributions[node].variable] = value
                trace.append(
                    self._trace_step(
                        trace,
                        node=node,
                        node_type="distribution",
                        value=value,
                        output_ref=distributions[node].variable,
                        distribution_type=distributions[node].distribution_type,
                    )
                )
            elif node in networks_map:
                net = networks_map[node]
                # M2-C1: dispatch by network kind. Composite networks (P19:
                # embeddings/blocks/residual/layernorm/dropout) need the named
                # field dict and composite_forward; dense networks use the flat
                # vector and dense_forward as before.
                if getattr(net, "kind", "dense_network") == "composite_network":
                    vec = vectors.get(net.input)
                    input_vec = state.get(net.input, [])
                    input_data = (
                        {f: input_vec[i] for i, f in enumerate(vec.fields) if i < len(input_vec)}
                        if vec else {}
                    )
                    value = self._execute_composite_network(net, input_data, parameters)
                    node_type = "composite_network"
                else:
                    input_vec = state.get(net.input, [])
                    value = self._execute_dense_network(net, input_vec, parameters)
                    node_type = "dense_network"
                state[node] = value
                state[net.output] = value
                trace.append(
                    self._trace_step(
                        trace,
                        node=node,
                        node_type=node_type,
                        value=value,
                        output_ref=net.output,
                        network_name=net.name,
                    )
                )
            elif node in action_specs:
                action_result = self._evaluate_action(action_specs[node], state)
                actions.append(action_result)
                trace.append(
                    self._trace_step(
                        trace,
                        node=node,
                        node_type="action",
                        value=action_result,
                        output_ref=node,
                        policy=action_specs[node].policy,
                    )
                )

        return {"state": state, "trace": trace, "actions": actions}

    def _load_vector(self, vector: VectorSpec, input_data: dict[str, Any]) -> list[float]:
        source = input_data.get(vector.name, input_data)
        values = []
        for field in vector.fields:
            value = float(source.get(field, 0.0))
            errors = validate_value_against_type(
                f"{vector.name}.{field}", value, vector.field_types.get(field)
            )
            if errors:
                raise ValueError("; ".join(errors))
            values.append(value)
        if len(values) != vector.size:
            raise ValueError(f"Vector {vector.name} expected {vector.size} values")
        return values

    def _evaluate_function(
        self,
        function: FunctionSpec,
        state: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> Any:
        parameters = parameters or {}
        expression_kind = function.semantic.kind
        if expression_kind == "softmax_linear":
            vector = self._first_vector(state)
            weights = self._get_parameter(parameters, function, "weights")
            bias = self._get_parameter(parameters, function, "bias")
            if weights is not None and bias is not None:
                labels = function.semantic.parameters.get("labels") or ["support", "sales", "operations"]
                return self._parameterized_softmax_linear(vector, weights, bias, labels)
            return self._email_classifier(vector)
        if expression_kind == "sigmoid_threshold":
            return self._evaluate_sigmoid_threshold(function, state)
        if expression_kind == "sigmoid_linear":
            vector = self._first_vector(state)
            weights = self._get_parameter(parameters, function, "weights")
            bias = self._get_parameter(parameters, function, "bias")
            if weights is not None and bias is not None:
                return self._sigmoid(self._dot(vector, weights) + self._bias_value(bias))
            return self._sigmoid(6.0 * (self._linear_score(vector) - 0.5))
        if expression_kind == "linear_regression":
            vector = self._first_vector(state)
            weights = self._get_parameter(parameters, function, "weights")
            bias = self._get_parameter(parameters, function, "bias")
            if weights is not None and bias is not None:
                return self._dot(vector, weights) + self._bias_value(bias)
            return self._linear_score(vector)
        # P1: symbolic expression kinds
        if expression_kind in ("symbolic_expr", "symbolic_weighted_sum"):
            return self._evaluate_symbolic(function.semantic.raw, state)
        # P1: aggregate kinds
        if expression_kind.startswith("aggregate_"):
            return self._evaluate_aggregate(function.semantic, state)
        # P1: normalisation
        if expression_kind == "normalize":
            return self._evaluate_normalize(function.semantic, state)
        # P1: argmax selection
        if expression_kind == "select_argmax":
            return self._evaluate_select_argmax(function.semantic, state)
        raise ValueError(
            f"Unsupported function semantic kind {expression_kind!r} "
            f"for FUNCTION {function.name}: {function.expression}"
        )

    def _evaluate_distribution(self, distribution: DistributionSpec, state: dict[str, Any]) -> dict[str, Any]:
        if distribution.distribution_type == "Categorical":
            source = state.get(distribution.source)
            if not isinstance(source, dict):
                raise ValueError(f"Categorical source {distribution.source} was not produced")
            label, probability = max(source.items(), key=lambda item: item[1])
            return {
                "type": "Categorical",
                "probabilities": source,
                "label": label,
                "max": probability,
            }

        if distribution.distribution_type == "Normal":
            source_name = distribution.source.split(",", 1)[0].strip()
            mean = self._resolve_numeric(source_name, state)
            return {"type": "Normal", "mean": mean, "sigma": 0.05}

        raise ValueError(f"Unsupported distribution: {distribution.distribution_type}")

    def _evaluate_action(self, action: Any, state: dict[str, Any]) -> dict[str, Any]:
        source = action.condition.source
        threshold = action.condition.threshold
        value = self._resolve_numeric(source, state)
        activated = self._compare(value, action.condition.operator, threshold)
        return {
            "name": action.name,
            "call": action.call,
            "source": source,
            "operator": action.condition.operator,
            "value": value,
            "threshold": threshold,
            "activated": activated,
            "policy": action.policy,
            "simulated": True,
        }

    def _evaluate_sigmoid_threshold(self, function: FunctionSpec, state: dict[str, Any]) -> float:
        source = function.semantic.inputs[0]
        scale = float(function.semantic.parameters["scale"])
        threshold = float(function.semantic.parameters["threshold"])
        return self._sigmoid(scale * (self._resolve_numeric(source, state) - threshold))

    def _compare(self, value: float, operator: str, threshold: float) -> bool:
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<":
            return value < threshold
        if operator == "<=":
            return value <= threshold
        raise ValueError(f"Unsupported operator: {operator}")

    def _trace_step(self, trace: list[dict[str, Any]], **data: Any) -> dict[str, Any]:
        return {"step": len(trace) + 1, "status": "ok", **data}

    def _email_classifier(self, vector: list[float]) -> dict[str, float]:
        padded = vector + [0.0] * max(0, 8 - len(vector))
        urgency, sender_trust, topic_support, topic_sales = padded[:4]
        sentiment, has_attachment, previous_interactions, language_confidence = padded[4:8]

        logits = {
            "support": 6.0 * (topic_support + 0.45 * sender_trust + 0.25 * previous_interactions + 0.2 * language_confidence),
            "sales": 6.0 * (topic_sales + 0.2 * sender_trust + 0.15 * sentiment),
            "operations": 6.0 * (0.5 * urgency + 0.25 * has_attachment + 0.2 * (1.0 - sentiment)),
        }
        return self._softmax(logits)

    def _linear_score(self, vector: list[float]) -> float:
        if not vector:
            return 0.0
        weighted_sum = sum((index + 1) * value for index, value in enumerate(vector))
        max_sum = sum(range(1, len(vector) + 1))
        return max(0.0, min(1.0, weighted_sum / max_sum))

    def _first_vector(self, state: dict[str, Any]) -> list[float]:
        for value in state.values():
            if isinstance(value, list):
                return value
        raise ValueError("No vector available in runtime state")

    def _resolve_numeric(self, path: str, state: dict[str, Any]) -> float:
        parts = path.split(".")
        value: Any = state.get(parts[0])
        for part in parts[1:]:
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)
        if isinstance(value, dict):
            if "max" in value:
                return float(value["max"])
            if "mean" in value:
                return float(value["mean"])
        return float(value)

    def _softmax(self, logits: dict[str, float]) -> dict[str, float]:
        max_logit = max(logits.values())
        exps = {key: math.exp(value - max_logit) for key, value in logits.items()}
        total = sum(exps.values())
        return {key: value / total for key, value in exps.items()}

    def _get_parameter(self, parameters: dict[str, Any], function: FunctionSpec, role: str) -> Any:
        name = function.semantic.parameters.get(role)
        if not name:
            return None
        qualified = f"{function.name}.{name}"
        if qualified in parameters:
            return parameters[qualified]
        return parameters.get(name)

    def _parameterized_softmax_linear(
        self,
        vector: list[float],
        weights: Any,
        bias: Any,
        labels: list[str],
    ) -> dict[str, float]:
        logits: dict[str, float] = {}
        for index, label in enumerate(labels):
            row = weights[index] if isinstance(weights, list) and index < len(weights) else []
            bias_value = bias[index] if isinstance(bias, list) and index < len(bias) else 0.0
            logits[str(label)] = self._dot(vector, row) + float(bias_value)
        return self._softmax(logits)

    def _dot(self, vector: list[float], weights: Any) -> float:
        if weights is None:
            return 0.0
        return sum(float(value) * float(weight) for value, weight in zip(vector, weights))

    def _bias_value(self, bias: Any) -> float:
        if isinstance(bias, list):
            return float(bias[0] if bias else 0.0)
        return float(bias)

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    # ------------------------------------------------------------------
    # P1 — Handlers for symbolic, aggregate, normalise, select kinds
    # ------------------------------------------------------------------

    def _evaluate_symbolic(self, expr_str: str, state: dict[str, Any]) -> float:
        from matrixai.ir.expr import parse_expr
        node = parse_expr(expr_str)
        return float(node.eval(state))

    def _evaluate_aggregate(
        self, semantic: Any, state: dict[str, Any]
    ) -> float:
        inputs: list[str] = semantic.parameters.get("inputs", list(semantic.inputs))
        method: str = semantic.parameters.get("method", "max")
        values = [self._resolve_numeric(inp, state) for inp in inputs]
        if not values:
            return 0.0
        if method == "max":
            return max(values)
        if method == "min":
            return min(values)
        if method == "mean":
            return sum(values) / len(values)
        if method == "softmax":
            max_v = max(values)
            exps = [math.exp(v - max_v) for v in values]
            total = sum(exps)
            return max(e / total for e in exps)
        if method == "vote":
            # Fraction of inputs exceeding 0.5
            return float(sum(1 for v in values if v > 0.5)) / len(values)
        raise ValueError(f"Unsupported aggregate method: {method}")

    def _evaluate_normalize(
        self, semantic: Any, state: dict[str, Any]
    ) -> float:
        var: str = semantic.parameters.get(
            "var", semantic.inputs[0] if semantic.inputs else ""
        )
        lo = float(semantic.parameters.get("lo", 0.0))
        hi = float(semantic.parameters.get("hi", 1.0))
        value = self._resolve_numeric(var, state)
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    def _evaluate_select_argmax(
        self, semantic: Any, state: dict[str, Any]
    ) -> Any:
        score_input: str = semantic.parameters.get("score_input", "")
        score_value = state.get(score_input)
        if isinstance(score_value, dict):
            return max(score_value, key=lambda k: score_value[k])
        return score_value

    def _execute_dense_network(
        self, network: Any, input_vec: list[float], parameters: dict[str, Any]
    ) -> list[float]:
        from matrixai.forward.dense_forward import dense_forward
        from matrixai.parameters.store import ParameterSet

        # Normalize flat param values (list) to ParameterSet entry format {"values": ...}
        net_params: dict[str, Any] = {}
        for key, val in parameters.items():
            if isinstance(val, dict) and "values" in val:
                net_params[key] = val
            elif isinstance(val, list):
                net_params[key] = {"values": val, "shape": [], "dtype": "float32"}
            else:
                net_params[key] = {"values": val, "shape": [], "dtype": "float32"}

        ps = ParameterSet(
            parameter_set_id="runtime",
            model_hash="",
            parameter_schema_hash="",
            source="runtime",
            parameters=net_params,
        )
        return dense_forward(network, ps, input_vec)

    def _execute_composite_network(
        self, network: Any, input_data: dict[str, Any], parameters: dict[str, Any]
    ) -> list[float]:
        """M2-C1: run a P19 composite network (embeddings/blocks/residual)."""
        from matrixai.forward.composite_forward import composite_forward
        from matrixai.parameters.store import ParameterSet

        net_params: dict[str, Any] = {}
        for key, val in parameters.items():
            if isinstance(val, dict) and "values" in val:
                net_params[key] = val
            else:
                # Preserve 2D embedding tables and 1D weights alike
                net_params[key] = {"values": val, "shape": [], "dtype": "float32"}

        ps = ParameterSet(
            parameter_set_id="runtime",
            model_hash="",
            parameter_schema_hash="",
            source="runtime",
            parameters=net_params,
        )
        # Inference: training=False so dropout is a no-op
        return composite_forward(network, ps, input_data, training=False)