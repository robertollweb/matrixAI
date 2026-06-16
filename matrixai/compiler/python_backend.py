# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import keyword
import re
import textwrap

from matrixai.ir.expr import BinOpNode, CallNode, ExprNode, LiteralNode, VarNode, parse_expr
from matrixai.ir import ActionSpec, DistributionSpec, FunctionSpec, MatrixAIProgram, VectorSpec


class PythonBackendCompiler:
    """Compile MatrixAI IR into a standalone Python module.

    The generated module exposes ``run(input_data)`` and emits direct Python
    statements for every graph node. Small math/runtime helpers are still
    generated, but the graph itself is no longer interpreted from embedded IR
    constants.
    """

    _IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]")

    def compile(self, program: MatrixAIProgram) -> str:
        vectors = {vector.name: vector for vector in program.vectors}
        functions = {function.name: function for function in program.functions}
        distributions = {distribution.name: distribution for distribution in program.distributions}
        actions = {action.name: action for action in program.actions}

        run_lines = [
            "def run(input_data: dict[str, Any]) -> dict[str, Any]:",
            "    state: dict[str, Any] = {}",
            "    trace: list[dict[str, Any]] = []",
            "    actions: list[dict[str, Any]] = []",
            "",
        ]

        for node in program.graph.nodes:
            if node in vectors:
                run_lines.extend(self._compile_vector(node, vectors[node]))
            elif node in functions:
                run_lines.extend(self._compile_function(node, functions[node]))
            elif node in distributions:
                run_lines.extend(self._compile_distribution(node, distributions[node]))
            elif node in actions:
                run_lines.extend(self._compile_action(node, actions[node]))

        run_lines.extend(
            [
                "    return {'state': state, 'trace': trace, 'actions': actions}",
                "",
            ]
        )

        source = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import math",
                "from typing import Any",
                "",
                "",
                f"PROJECT = {program.project!r}",
                "",
                *run_lines,
                self._helpers(),
            ]
        )
        return source.rstrip() + "\n"

    def _compile_vector(self, node: str, vector: VectorSpec) -> list[str]:
        value_name = self._name(node)
        fields = repr(vector.fields)
        field_types = {
            field: spec.to_dict() for field, spec in vector.field_types.items()
        }
        return [
            f"    # VECTOR {vector.name}",
            f"    {value_name}_source = input_data.get({vector.name!r}, input_data)",
            f"    {value_name}_fields = {fields}",
            f"    {value_name}_field_types = {field_types!r}",
            f"    {value_name} = [float({value_name}_source.get(field, 0.0)) for field in {value_name}_fields]",
            f"    if len({value_name}) != {vector.size!r}:",
            f"        raise ValueError('Vector {vector.name} expected {vector.size} values')",
            f"    for field, value in zip({value_name}_fields, {value_name}):",
            f"        _validate_type_value(f'{vector.name}.{{field}}', value, {value_name}_field_types.get(field))",
            f"    state[{node!r}] = {value_name}",
            f"    for field, value in zip({value_name}_fields, {value_name}):",
            "        state.setdefault(field, value)",
            "    trace.append(_trace_step(trace, "
            f"node={node!r}, node_type='vector', value={value_name}, output_ref={node!r}))",
            "",
        ]

    def _compile_function(self, node: str, function: FunctionSpec) -> list[str]:
        value_name = self._name(node)
        semantic = function.semantic
        expression = self._function_expression(function)
        return [
            f"    # FUNCTION {function.name}",
            f"    {value_name} = {expression}",
            f"    state[{node!r}] = {value_name}",
            f"    state[{function.output!r}] = {value_name}",
            "    trace.append(_trace_step(trace, "
            f"node={node!r}, node_type='function', value={value_name}, "
            f"output_ref={function.output!r}, expression_kind={semantic.kind!r}, "
            f"inputs={semantic.inputs!r}))",
            "",
        ]

    def _function_expression(self, function: FunctionSpec) -> str:
        semantic = function.semantic
        if semantic.kind == "softmax_linear":
            vector_expr = self._vector_expression(semantic.inputs)
            return f"_email_classifier({vector_expr})"
        if semantic.kind == "sigmoid_linear":
            vector_expr = self._vector_expression(semantic.inputs)
            return f"_sigmoid(6.0 * (_linear_score({vector_expr}) - 0.5))"
        if semantic.kind == "sigmoid_threshold":
            source = semantic.inputs[0]
            scale = float(semantic.parameters["scale"])
            threshold = float(semantic.parameters["threshold"])
            return f"_sigmoid({scale!r} * (_resolve_numeric({source!r}, state) - {threshold!r}))"
        if semantic.kind in {"symbolic_expr", "symbolic_weighted_sum"}:
            return self._compile_symbolic_expression(semantic.raw)
        if semantic.kind.startswith("aggregate_"):
            method = semantic.parameters.get("method", semantic.kind.removeprefix("aggregate_"))
            inputs = semantic.parameters.get("inputs", semantic.inputs)
            return f"_aggregate({inputs!r}, {method!r}, state)"
        if semantic.kind == "normalize":
            var = semantic.parameters.get("var", semantic.inputs[0] if semantic.inputs else "")
            lo = float(semantic.parameters.get("lo", 0.0))
            hi = float(semantic.parameters.get("hi", 1.0))
            return f"_normalize_range({var!r}, {lo!r}, {hi!r}, state)"
        if semantic.kind == "linear_regression":
            vector_expr = self._vector_expression(semantic.inputs)
            return f"_linear_score({vector_expr})"
        if semantic.kind == "select_argmax":
            score_input = semantic.parameters.get(
                "score_input", semantic.inputs[0] if semantic.inputs else ""
            )
            return f"_select_argmax({score_input!r}, state)"
        raise ValueError(
            f"Unsupported function semantic kind {semantic.kind!r} "
            f"for FUNCTION {function.name}: {function.expression}"
        )

    def _compile_symbolic_expression(self, expression: str) -> str:
        return self._compile_expr_node(parse_expr(expression))

    def _compile_expr_node(self, node: ExprNode) -> str:
        if isinstance(node, LiteralNode):
            return repr(float(node.value))
        if isinstance(node, VarNode):
            return f"_resolve_numeric({node.name!r}, state)"
        if isinstance(node, BinOpNode):
            left = self._compile_expr_node(node.left)
            right = self._compile_expr_node(node.right)
            return f"({left} {node.op} {right})"
        if isinstance(node, CallNode):
            args = [self._compile_expr_node(arg) for arg in node.args]
            if node.func == "normalize" and len(args) == 1:
                return f"_normalize({args[0]})"
            if node.func == "sigmoid" and len(args) == 1:
                return f"_sigmoid({args[0]})"
            if node.func == "clip" and 1 <= len(args) <= 3:
                return f"_clip({', '.join(args)})"
            if node.func == "scale" and len(args) == 3:
                return f"_scale({args[0]}, {args[1]}, {args[2]})"
            if node.func in {"abs", "max", "min"}:
                return f"{node.func}({', '.join(args)})"
            if node.func == "sigmoid_product":
                return f"_sigmoid_product({', '.join(args)})"
            if node.func == "sigmoid_or":
                return f"_sigmoid_or({', '.join(args)})"
            return f"_resolve_symbolic_call({node.func!r}, [{', '.join(args)}], state)"
        raise ValueError(f"Unsupported symbolic AST node: {type(node).__name__}")

    def _compile_distribution(self, node: str, distribution: DistributionSpec) -> list[str]:
        value_name = self._name(node)
        if distribution.distribution_type == "Categorical":
            lines = [
                f"    # DISTRIBUTION {distribution.name}",
                f"    {value_name}_source = state.get({distribution.source!r})",
                f"    if not isinstance({value_name}_source, dict):",
                f"        raise ValueError('Categorical source {distribution.source} was not produced')",
                f"    {value_name}_label, {value_name}_probability = max({value_name}_source.items(), key=lambda item: item[1])",
                f"    {value_name} = {{'type': 'Categorical', 'probabilities': {value_name}_source, "
                f"'label': {value_name}_label, 'max': {value_name}_probability}}",
            ]
        elif distribution.distribution_type == "Normal":
            source_name = distribution.source.split(",", 1)[0].strip()
            lines = [
                f"    # DISTRIBUTION {distribution.name}",
                f"    {value_name}_mean = _resolve_numeric({source_name!r}, state)",
                f"    {value_name} = {{'type': 'Normal', 'mean': {value_name}_mean, 'sigma': 0.05}}",
            ]
        else:
            lines = [
                f"    raise ValueError('Unsupported distribution: {distribution.distribution_type}')"
            ]

        lines.extend(
            [
                f"    state[{node!r}] = {value_name}",
                f"    state[{distribution.variable!r}] = {value_name}",
                "    trace.append(_trace_step(trace, "
                f"node={node!r}, node_type='distribution', value={value_name}, "
                f"output_ref={distribution.variable!r}, distribution_type={distribution.distribution_type!r}))",
                "",
            ]
        )
        return lines

    def _compile_action(self, node: str, action: ActionSpec) -> list[str]:
        value_name = self._name(node)
        condition = action.condition
        return [
            f"    # ACTION {action.name}",
            f"    {value_name}_value = _resolve_numeric({condition.source!r}, state)",
            f"    {value_name}_activated = _compare({value_name}_value, {condition.operator!r}, {condition.threshold!r})",
            f"    {value_name} = {{",
            f"        'name': {action.name!r},",
            f"        'call': {action.call!r},",
            f"        'source': {condition.source!r},",
            f"        'operator': {condition.operator!r},",
            f"        'value': {value_name}_value,",
            f"        'threshold': {condition.threshold!r},",
            f"        'activated': {value_name}_activated,",
            f"        'policy': {action.policy!r},",
            "        'simulated': True,",
            "    }",
            f"    actions.append({value_name})",
            "    trace.append(_trace_step(trace, "
            f"node={node!r}, node_type='action', value={value_name}, "
            f"output_ref={node!r}, policy={action.policy!r}))",
            "",
        ]

    def _vector_expression(self, inputs: list[str]) -> str:
        if not inputs:
            return "_first_vector(state)"
        return f"state[{inputs[0]!r}]"

    def _name(self, value: str) -> str:
        name = self._IDENTIFIER_RE.sub("_", value)
        if not name or name[0].isdigit() or keyword.iskeyword(name):
            name = f"node_{name}"
        return name

    def _helpers(self) -> str:
        return textwrap.dedent(
            '''
            def _compare(value: float, operator: str, threshold: float) -> bool:
                if operator == ">":
                    return value > threshold
                if operator == ">=":
                    return value >= threshold
                if operator == "<":
                    return value < threshold
                if operator == "<=":
                    return value <= threshold
                raise ValueError(f"Unsupported operator: {operator}")


            def _trace_step(trace: list[dict[str, Any]], **data: Any) -> dict[str, Any]:
                return {"step": len(trace) + 1, "status": "ok", **data}


            def _email_classifier(vector: list[float]) -> dict[str, float]:
                padded = vector + [0.0] * max(0, 8 - len(vector))
                urgency, sender_trust, topic_support, topic_sales = padded[:4]
                sentiment, has_attachment, previous_interactions, language_confidence = padded[4:8]

                logits = {
                    "support": 6.0 * (
                        topic_support
                        + 0.45 * sender_trust
                        + 0.25 * previous_interactions
                        + 0.2 * language_confidence
                    ),
                    "sales": 6.0 * (topic_sales + 0.2 * sender_trust + 0.15 * sentiment),
                    "operations": 6.0 * (
                        0.5 * urgency + 0.25 * has_attachment + 0.2 * (1.0 - sentiment)
                    ),
                }
                return _softmax(logits)


            def _linear_score(vector: list[float]) -> float:
                if not vector:
                    return 0.0
                weighted_sum = sum((index + 1) * value for index, value in enumerate(vector))
                max_sum = sum(range(1, len(vector) + 1))
                return max(0.0, min(1.0, weighted_sum / max_sum))


            def _first_vector(state: dict[str, Any]) -> list[float]:
                for value in state.values():
                    if isinstance(value, list):
                        return value
                raise ValueError("No vector available in runtime state")


            def _resolve_numeric(path: str, state: dict[str, Any]) -> float:
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


            def _resolve_symbolic_call(name: str, args: list[float], state: dict[str, Any]) -> float:
                value = state.get(name)
                if value is None:
                    return 0.0
                if callable(value):
                    return float(value(*args))
                if isinstance(value, dict):
                    return float(max(value.values()))
                return float(value)


            def _validate_type_value(name: str, value: Any, spec: dict[str, Any] | None) -> None:
                if not spec:
                    return
                range_spec = spec.get("range")
                if range_spec:
                    number = float(value)
                    minimum = range_spec.get("min")
                    maximum = range_spec.get("max")
                    if minimum is not None and number < float(minimum):
                        raise ValueError(f"{name}={number} outside {spec['name']} range")
                    if maximum is not None and number > float(maximum):
                        raise ValueError(f"{name}={number} outside {spec['name']} range")


            def _normalize(value: float) -> float:
                return max(0.0, min(1.0, float(value)))


            def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
                return max(float(lo), min(float(hi), float(value)))


            def _scale(value: float, lo: float, hi: float) -> float:
                if hi == lo:
                    return 0.0
                return _clip((float(value) - float(lo)) / (float(hi) - float(lo)))


            def _normalize_range(path: str, lo: float, hi: float, state: dict[str, Any]) -> float:
                return _scale(_resolve_numeric(path, state), lo, hi)


            def _aggregate(inputs: list[str], method: str, state: dict[str, Any]) -> float:
                values = [_resolve_numeric(inp, state) for inp in inputs]
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
                    return float(sum(1 for v in values if v > 0.5)) / len(values)
                raise ValueError(f"Unsupported aggregate method: {method}")


            def _select_argmax(score_input: str, state: dict[str, Any]) -> Any:
                score_value = state.get(score_input)
                if isinstance(score_value, dict):
                    return max(score_value, key=lambda key: score_value[key])
                return score_value


            def _sigmoid_product(*values: float) -> float:
                result = 1.0
                for value in values:
                    result *= float(value)
                return result


            def _sigmoid_or(*values: float) -> float:
                if len(values) == 1:
                    return float(values[0])
                inactive = 1.0
                for value in values:
                    inactive *= 1.0 - float(value)
                return 1.0 - inactive


            def _softmax(logits: dict[str, float]) -> dict[str, float]:
                max_logit = max(logits.values())
                exps = {key: math.exp(value - max_logit) for key, value in logits.items()}
                total = sum(exps.values())
                return {key: value / total for key, value in exps.items()}


            def _sigmoid(value: float) -> float:
                return 1.0 / (1.0 + math.exp(-value))
            '''
        ).strip()
