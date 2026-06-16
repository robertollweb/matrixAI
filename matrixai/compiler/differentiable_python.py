# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from typing import Any

from matrixai.compiler.backend_contract import BackendContractAnalyzer, BackendContractReport
from matrixai.compiler.python_backend import PythonBackendCompiler
from matrixai.ir import ActionSpec, DistributionSpec, MatrixAIProgram


class DifferentiablePythonCompiler(PythonBackendCompiler):
    """Compile the P3 differentiable-python subset into a standalone module.

    The generated module executes the supported continuous graph while marking
    distributions and simulated actions as runtime boundaries. It does not train
    parameters, perform autodiff, or execute real side effects.
    """

    target = "differentiable_python"

    def compile(self, program: MatrixAIProgram) -> str:
        report = BackendContractAnalyzer(target=self.target).analyze(program)
        if not report.ok:
            issues = [node.node for node in report.unsupported_nodes] + list(report.parameter_errors)
            blocked = ", ".join(issues)
            raise ValueError(
                f"Program {program.project} is not portable to {self.target}: {blocked}"
            )

        self._layer_map = {layer.name: layer for layer in program.layers}
        vectors = {vector.name: vector for vector in program.vectors}
        sequences = {seq.name: seq for seq in program.sequences}
        functions = {function.name: function for function in program.functions}
        distributions = {distribution.name: distribution for distribution in program.distributions}
        actions = {action.name: action for action in program.actions}

        run_lines = [
            "def initial_parameters() -> dict[str, Any]:",
            "    values: dict[str, Any] = {}",
            "    for parameter in PARAMETER_MANIFEST:",
            "        if 'initial_value' not in parameter:",
            "            continue",
            "        values[parameter['name']] = parameter['initial_value']",
            "        values[f\"{parameter['function']}.{parameter['name']}\"] = parameter['initial_value']",
            "    return values",
            "",
            "",
            "def validate_parameters(parameters: dict[str, Any]) -> list[str]:",
            "    return _parameter_validation_errors(_normalize_parameters(parameters))",
            "",
            "",
            "def run(input_data: dict[str, Any], parameters: dict[str, Any] | None = None) -> dict[str, Any]:",
            "    parameters = _normalize_parameters(parameters or {})",
            "    _validate_parameters(parameters)",
            "    state: dict[str, Any] = {}",
            "    trace: list[dict[str, Any]] = []",
            "    actions: list[dict[str, Any]] = []",
            "    runtime_boundaries: list[dict[str, Any]] = []",
            "",
        ]

        for node in program.graph.nodes:
            if node in vectors:
                run_lines.extend(self._compile_vector(node, vectors[node]))
            elif node in sequences:
                run_lines.extend(self._compile_sequence(node, sequences[node]))
            elif node in functions:
                run_lines.extend(self._compile_function(node, functions[node]))
            elif node in distributions:
                run_lines.extend(self._compile_distribution_boundary(node, distributions[node]))
            elif node in actions:
                run_lines.extend(self._compile_action_boundary(node, actions[node]))

        run_lines.extend(
            [
                "    return {",
                "        'target': TARGET,",
                "        'state': state,",
                "        'trace': trace,",
                "        'actions': actions,",
                "        'runtime_boundaries': runtime_boundaries,",
                "        'tensor_shapes': TENSOR_SHAPES,",
                "        'type_constraints': TYPE_CONSTRAINTS,",
                "        'trainable_parameters': TRAINABLE_PARAMETERS,",
                "        'parameter_manifest': PARAMETER_MANIFEST,",
                "        'autodiff_plan': AUTODIFF_PLAN,",
                "        'backend_report': BACKEND_REPORT,",
                "    }",
                "",
            ]
        )

        report_dict = report.to_dict()
        boundary_nodes = [
            node.to_dict() for node in report.nodes if node.supported and not node.differentiable
        ]
        source = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import math",
                "from typing import Any",
                "",
                "",
                f"TARGET = {self.target!r}",
                f"PROJECT = {program.project!r}",
                f"BACKEND_REPORT = {report_dict!r}",
                f"TENSOR_SHAPES = {report_dict['tensor_shapes']!r}",
                f"TYPE_CONSTRAINTS = {report_dict['type_constraints']!r}",
                f"TRAINABLE_PARAMETERS = {report_dict['trainable_parameters']!r}",
                f"PARAMETER_MANIFEST = {report_dict['parameter_manifest']!r}",
                f"AUTODIFF_PLAN = {report_dict['autodiff_plan']!r}",
                f"RUNTIME_BOUNDARY_NODES = {boundary_nodes!r}",
                "",
                *run_lines,
                self._helpers(),
                self._compile_layer_executors(),
            ]
        )
        return source.rstrip() + "\n"

    def _function_expression(self, function) -> str:
        semantic = function.semantic
        if semantic.kind == "softmax_linear":
            vector_expr = self._vector_expression(semantic.inputs)
            labels = semantic.parameters.get("labels") or ["support", "sales", "operations"]
            weights = semantic.parameters.get("weights", "")
            bias = semantic.parameters.get("bias", "")
            return (
                f"_parameterized_softmax_linear({vector_expr}, "
                f"_get_parameter(parameters, {weights!r}, {function.name!r}), "
                f"_get_parameter(parameters, {bias!r}, {function.name!r}), {labels!r})"
            )
        if semantic.kind == "sigmoid_linear":
            vector_expr = self._vector_expression(semantic.inputs)
            weights = semantic.parameters.get("weights", "")
            bias = semantic.parameters.get("bias", "")
            return (
                f"_parameterized_sigmoid_linear({vector_expr}, "
                f"_get_parameter(parameters, {weights!r}, {function.name!r}), "
                f"_get_parameter(parameters, {bias!r}, {function.name!r}))"
            )
        if semantic.kind == "linear_regression":
            vector_expr = self._vector_expression(semantic.inputs)
            weights = semantic.parameters.get("weights", "")
            bias = semantic.parameters.get("bias", "")
            return (
                f"_parameterized_linear_regression({vector_expr}, "
                f"_get_parameter(parameters, {weights!r}, {function.name!r}), "
                f"_get_parameter(parameters, {bias!r}, {function.name!r}))"
            )
        if semantic.kind == "layer_call":
            layer_name = semantic.parameters.get("layer", "")
            input_name = semantic.inputs[0] if semantic.inputs else ""
            layer = getattr(self, "_layer_map", {}).get(layer_name)
            if layer and layer.body_ops:
                return f"_layer_exec_{layer_name}({input_name!r}, state, parameters)"
            return f"_layer_call_passthrough({input_name!r}, {layer_name!r}, state, parameters)"
        if semantic.kind == "dot":
            a, b = (semantic.inputs + ["", ""])[:2]
            return f"_tensor_dot({a!r}, {b!r}, state)"
        if semantic.kind == "matmul":
            a, b = (semantic.inputs + ["", ""])[:2]
            return f"_tensor_matmul({a!r}, {b!r}, state)"
        if semantic.kind == "relu":
            x = semantic.inputs[0] if semantic.inputs else ""
            return f"_tensor_relu({x!r}, state)"
        if semantic.kind == "gelu":
            x = semantic.inputs[0] if semantic.inputs else ""
            return f"_tensor_gelu({x!r}, state)"
        if semantic.kind == "layer_norm":
            inputs = semantic.inputs
            x = inputs[0] if len(inputs) > 0 else ""
            gain = inputs[1] if len(inputs) > 1 else ""
            bias_var = inputs[2] if len(inputs) > 2 else ""
            eps_var = inputs[3] if len(inputs) > 3 else ""
            return f"_tensor_layer_norm({x!r}, {gain!r}, {bias_var!r}, {eps_var!r}, state)"
        if semantic.kind == "residual":
            a, b = (semantic.inputs + ["", ""])[:2]
            return f"_tensor_residual({a!r}, {b!r}, state)"
        if semantic.kind == "mean_pooling":
            x, mask = (semantic.inputs + ["", ""])[:2]
            return f"_tensor_mean_pooling({x!r}, {mask!r}, state)"
        if semantic.kind == "cls_pooling":
            x = semantic.inputs[0] if semantic.inputs else ""
            return f"_tensor_cls_pooling({x!r}, state)"
        if semantic.kind == "positional_encoding":
            x, pos = (semantic.inputs + ["", ""])[:2]
            return f"_tensor_positional_encoding({x!r}, {pos!r}, state)"
        if semantic.kind == "embedding_lookup":
            table, ids = (semantic.inputs + ["", ""])[:2]
            return f"_tensor_embedding_lookup({table!r}, {ids!r}, state)"
        if semantic.kind == "attention":
            q = semantic.inputs[0] if len(semantic.inputs) > 0 else ""
            k = semantic.inputs[1] if len(semantic.inputs) > 1 else ""
            v = semantic.inputs[2] if len(semantic.inputs) > 2 else ""
            mask = semantic.inputs[3] if len(semantic.inputs) > 3 else ""
            return f"_tensor_attention({q!r}, {k!r}, {v!r}, {mask!r}, state)"
        return super()._function_expression(function)

    # ------------------------------------------------------------------ #
    # LAYER body executor generation                                       #
    # ------------------------------------------------------------------ #

    def _body_op_expr(self, op: Any) -> str:
        """Translate a LayerBodyOp into a Python expression using local_state."""
        args = list(op.args)

        def a(i: int, default: str = "") -> str:
            return args[i] if i < len(args) else default

        k = op.kind
        if k == "matmul":
            return f"_tensor_matmul({a(0)!r}, {a(1)!r}, local_state)"
        if k == "scale":
            return f"_tensor_scale({a(0)!r}, {a(1, '1.0')!r}, local_state)"
        if k == "softmax":
            return f"_tensor_softmax({a(0)!r}, local_state)"
        if k == "dot":
            return f"_tensor_dot({a(0)!r}, {a(1)!r}, local_state)"
        if k == "relu":
            return f"_tensor_relu({a(0)!r}, local_state)"
        if k == "gelu":
            return f"_tensor_gelu({a(0)!r}, local_state)"
        if k == "layer_norm":
            # layer_norm(x, gain, bias[, eps])
            eps = a(3, "1e-5")
            try:
                eps_val = float(eps)
            except ValueError:
                eps_val = 1e-5
            return f"_tensor_layer_norm({a(0)!r}, {a(1)!r}, {a(2)!r}, {eps_val!r}, local_state)"
        if k == "residual":
            return f"_tensor_residual({a(0)!r}, {a(1)!r}, local_state)"
        if k == "attention":
            return f"_tensor_attention({a(0)!r}, {a(1)!r}, {a(2)!r}, {a(3)!r}, local_state)"
        if k == "mean_pooling":
            return f"_tensor_mean_pooling({a(0)!r}, {a(1)!r}, local_state)"
        if k == "cls_pooling":
            return f"_tensor_cls_pooling({a(0)!r}, local_state)"
        if k == "positional_encoding":
            return f"_tensor_positional_encoding({a(0)!r}, {a(1)!r}, local_state)"
        if k == "embedding_lookup":
            return f"_tensor_embedding_lookup({a(0)!r}, {a(1)!r}, local_state)"
        # fallback: identity of first arg
        return f"_resolve_vector({a(0)!r}, local_state)"

    def _compile_layer_executors(self) -> str:
        """Generate one _layer_exec_<name> function per LAYER that has body_ops."""
        layer_map = getattr(self, "_layer_map", {})
        blocks: list[str] = []
        for layer in layer_map.values():
            if not layer.body_ops:
                continue
            lines: list[str] = [
                f"def _layer_exec_{layer.name}(input_name: str, state: dict, parameters: dict) -> Any:",
                "    local_state = {**state}",
                "    local_state['input'] = _resolve_vector(input_name, state)",
            ]
            # bind each layer param into local_state
            for param in layer.params:
                pname = param.name
                qualified_key = layer.name + '.' + pname
                lines.append(f"    _v = parameters.get({qualified_key!r})")
                lines.append(
                    f"    local_state[{pname!r}] = "
                    f"_v if _v is not None else parameters.get({pname!r})"
                )
            # execute body ops
            for op in layer.body_ops:
                expr = self._body_op_expr(op)
                lines.append(f"    local_state[{op.output!r}] = {expr}")
            # return last output or 'result' if present
            last_output = layer.body_ops[-1].output if layer.body_ops else ""
            lines.append(
                f"    return local_state.get('result', "
                f"local_state.get({last_output!r}, _resolve_vector(input_name, state)))"
            )
            blocks.append("\n".join(lines))
        return "\n\n\n" + "\n\n\n".join(blocks) if blocks else ""

    def _compile_sequence(self, node: str, seq) -> list[str]:
        value_name = self._name(node)
        return [
            f"    # SEQUENCE {seq.name}",
            f"    {value_name}_raw = input_data.get({seq.name!r}, input_data)",
            f"    if isinstance({value_name}_raw, list):",
            f"        {value_name} = [int(t) % {seq.vocab_size} for t in {value_name}_raw]",
            f"    else:",
            f"        {value_name} = [int({value_name}_raw.get(f't{{i}}', 0)) % {seq.vocab_size} for i in range({seq.length})]",
            f"    if len({value_name}) != {seq.length!r}:",
            f"        raise ValueError('SEQUENCE {seq.name} expected {seq.length} tokens')",
            f"    state[{node!r}] = {value_name}",
            "    trace.append(_trace_step(trace, "
            f"node={node!r}, node_type='sequence', value={value_name}, output_ref={node!r}))",
            "",
        ]

    def _compile_distribution_boundary(
        self, node: str, distribution: DistributionSpec
    ) -> list[str]:
        lines = super()._compile_distribution(node, distribution)
        value_name = self._name(node)
        lines.insert(
            -1,
            "    runtime_boundaries.append({"
            f"'node': {node!r}, 'node_type': 'distribution', "
            f"'kind': {distribution.distribution_type!r}, "
            f"'output_ref': {distribution.variable!r}, 'value': {value_name}}})",
        )
        return lines

    def _compile_action_boundary(self, node: str, action: ActionSpec) -> list[str]:
        lines = super()._compile_action(node, action)
        value_name = self._name(node)
        lines.insert(
            -1,
            "    runtime_boundaries.append({"
            f"'node': {node!r}, 'node_type': 'action', "
            f"'kind': {action.policy!r}, 'call': {action.call!r}, "
            f"'value': {value_name}}})",
        )
        return lines

    def _helpers(self) -> str:
        return super()._helpers() + "\n\n" + "\n".join(
            [
                "def _get_parameter(parameters: dict[str, Any], name: str, function: str) -> Any:",
                "    if not name:",
                "        return None",
                "    qualified = f'{function}.{name}'",
                "    if qualified in parameters:",
                "        return parameters[qualified]",
                "    return parameters.get(name)",
                "",
                "",
                "def _validate_parameters(parameters: dict[str, Any]) -> None:",
                "    errors = _parameter_validation_errors(_normalize_parameters(parameters))",
                "    if errors:",
                "        raise ValueError('; '.join(errors))",
                "",
                "",
                "def _normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:",
                "    if not isinstance(parameters, dict):",
                "        return {}",
                "    payload = parameters.get('parameters')",
                "    if not isinstance(payload, dict):",
                "        return parameters",
                "    normalized: dict[str, Any] = {}",
                "    for name, parameter in payload.items():",
                "        if isinstance(parameter, dict):",
                "            value = parameter.get('values')",
                "            function = parameter.get('function')",
                "        else:",
                "            value = parameter",
                "            function = None",
                "        normalized[name] = value",
                "        if function:",
                "            normalized[f'{function}.{name}'] = value",
                "    return normalized",
                "",
                "",
                "def _parameter_validation_errors(parameters: dict[str, Any]) -> list[str]:",
                "    errors: list[str] = []",
                "    for parameter in PARAMETER_MANIFEST:",
                "        expected_shape = parameter.get('shape')",
                "        if expected_shape is None:",
                "            continue",
                "        name = parameter['name']",
                "        qualified = f\"{parameter['function']}.{name}\"",
                "        for key in (name, qualified):",
                "            if key in parameters:",
                "                error = _parameter_shape_error(key, parameters[key], expected_shape)",
                "                if error:",
                "                    errors.append(error)",
                "    return errors",
                "",
                "",
                "def _parameter_shape_error(name: str, value: Any, expected_shape: list[int]) -> str:",
                "    try:",
                "        actual_shape = _parameter_shape(value)",
                "    except ValueError as exc:",
                "        return f'Parameter {name} invalid: {exc}'",
                "    if actual_shape != list(expected_shape):",
                "        return f'Parameter {name} expected shape {list(expected_shape)}, got {actual_shape}'",
                "    return ''",
                "",
                "",
                "def _parameter_shape(value: Any) -> list[int]:",
                "    if isinstance(value, list):",
                "        if not value:",
                "            return [0]",
                "        first_shape = _parameter_shape(value[0])",
                "        for item in value[1:]:",
                "            if _parameter_shape(item) != first_shape:",
                "                raise ValueError('Parameter contains ragged values')",
                "        return [len(value)] + first_shape",
                "    try:",
                "        float(value)",
                "    except (TypeError, ValueError) as exc:",
                "        raise ValueError(f'Parameter contains non-numeric value {value!r}') from exc",
                "    return []",
                "",
                "",
                "def _parameterized_softmax_linear(vector: list[float], weights: Any, bias: Any, labels: list[str]) -> dict[str, float]:",
                "    if weights is None or bias is None:",
                "        return _email_classifier(vector)",
                "    logits: dict[str, float] = {}",
                "    for index, label in enumerate(labels):",
                "        row = weights[index] if index < len(weights) else []",
                "        bias_value = bias[index] if isinstance(bias, list) and index < len(bias) else 0.0",
                "        logits[label] = _dot(vector, row) + float(bias_value)",
                "    return _softmax(logits)",
                "",
                "",
                "def _parameterized_sigmoid_linear(vector: list[float], weights: Any, bias: Any) -> float:",
                "    if weights is None or bias is None:",
                "        return _sigmoid(6.0 * (_linear_score(vector) - 0.5))",
                "    bias_value = float(bias if not isinstance(bias, list) else (bias[0] if bias else 0.0))",
                "    return _sigmoid(_dot(vector, weights) + bias_value)",
                "",
                "",
                "def _parameterized_linear_regression(vector: list[float], weights: Any, bias: Any) -> float:",
                "    if weights is None or bias is None:",
                "        return sum(vector) / max(1, len(vector))",
                "    bias_value = float(bias if not isinstance(bias, list) else (bias[0] if bias else 0.0))",
                "    return _dot(vector, weights) + bias_value",
                "",
                "",
                "def _dot(vector: list[float], weights: Any) -> float:",
                "    if weights is None:",
                "        return 0.0",
                "    return sum(float(value) * float(weight) for value, weight in zip(vector, weights))",
                "",
                "",
                "def _resolve_vector(name: str, state: dict) -> list[float]:",
                "    v = state.get(name, 0.0)",
                "    if isinstance(v, list):",
                "        return [float(x) for x in v]",
                "    if isinstance(v, dict):",
                "        return [float(x) for x in v.values()]",
                "    return [float(v)]",
                "",
                "",
                "def _tensor_dot(a: str, b: str, state: dict) -> float:",
                "    va = _resolve_vector(a, state)",
                "    vb = _resolve_vector(b, state)",
                "    return sum(x * y for x, y in zip(va, vb))",
                "",
                "",
                "def _tensor_scale(x: str, factor_raw, state: dict) -> list:",
                "    vec = state.get(x, [])",
                "    try:",
                "        factor = float(factor_raw)",
                "    except ValueError:",
                "        factor = float(state.get(factor_raw, 1.0))",
                "    if not isinstance(vec, list) or not vec:",
                "        return float(vec) * factor if vec else 0.0 * factor",
                "    if isinstance(vec[0], list):",
                "        return [[v * factor for v in row] for row in vec]",
                "    return [v * factor for v in vec]",
                "",
                "",
                "def _tensor_softmax(x: str, state: dict) -> list:",
                "    vec = state.get(x, [])",
                "    if not isinstance(vec, list) or not vec:",
                "        return 1.0",
                "    if isinstance(vec[0], list):",
                "        res = []",
                "        for row in vec:",
                "            m = max(row) if row else 0.0",
                "            exps = [math.exp(v - m) for v in row]",
                "            total = sum(exps) if sum(exps) > 0 else 1.0",
                "            res.append([e / total for e in exps])",
                "        return res",
                "    m = max(vec) if vec else 0.0",
                "    exps = [math.exp(v - m) for v in vec]",
                "    total = sum(exps) if sum(exps) > 0 else 1.0",
                "    return [e / total for e in exps]",
                "",
                "",
                "def _tensor_matmul(a: str, b: str, state: dict) -> list:",
                "    va = state.get(a, [])",
                "    vb = state.get(b, [])",
                "    if not isinstance(va, list) or not isinstance(vb, list):",
                "        return []",
                "    if not va or not isinstance(va[0], list):",
                "        if vb and isinstance(vb[0], list):",
                "            n = min(len(va), len(vb))",
                "            cols_b = len(vb[0]) if vb else 0",
                "            return [sum(float(va[k]) * float(vb[k][j]) for k in range(n)) for j in range(cols_b)]",
                "        return _resolve_vector(a, state)",
                "    rows_a = len(va)",
                "    cols_a = len(va[0]) if va else 0",
                "    if not vb or not isinstance(vb[0], list):",
                "        vec_b = _resolve_vector(b, state)",
                "        return [sum(va[i][k] * vec_b[k] for k in range(min(cols_a, len(vec_b)))) for i in range(rows_a)]",
                "    cols_b = len(vb[0]) if vb else 0",
                "    return [[sum(va[i][k] * vb[k][j] for k in range(min(cols_a, len(vb)))) for j in range(cols_b)] for i in range(rows_a)]",
                "",
                "",
                "def _tensor_relu(x: str, state: dict) -> list[float]:",
                "    return [max(0.0, v) for v in _resolve_vector(x, state)]",
                "",
                "",
                "def _tensor_gelu(x: str, state: dict) -> list[float]:",
                "    def _gelu_scalar(v: float) -> float:",
                "        return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))",
                "    return [_gelu_scalar(v) for v in _resolve_vector(x, state)]",
                "",
                "",
                "def _tensor_layer_norm(x: str, gain: str, bias: str, eps: str, state: dict) -> list[float]:",
                "    vec = _resolve_vector(x, state)",
                "    if not vec:",
                "        return vec",
                "    mean = sum(vec) / len(vec)",
                "    var = sum((v - mean) ** 2 for v in vec) / len(vec)",
                "    eps_val = float(state.get(eps, 1e-5)) if eps else 1e-5",
                "    if isinstance(eps_val, list):",
                "        eps_val = float(eps_val[0]) if eps_val else 1e-5",
                "    std = math.sqrt(var + eps_val)",
                "    normalized = [(v - mean) / std for v in vec]",
                "    g = _resolve_vector(gain, state) if gain else [1.0] * len(vec)",
                "    bi = _resolve_vector(bias, state) if bias else [0.0] * len(vec)",
                "    return [n * gi + bi_v for n, gi, bi_v in zip(normalized, g, bi)]",
                "",
                "",
                "def _tensor_residual(a: str, b: str, state: dict) -> list[float]:",
                "    va = _resolve_vector(a, state)",
                "    vb = _resolve_vector(b, state)",
                "    return [x + y for x, y in zip(va, vb)]",
                "",
                "",
                "def _tensor_mean_pooling(x: str, mask: str, state: dict) -> list[float]:",
                "    val = state.get(x)",
                "    if isinstance(val, list) and val and isinstance(val[0], list):",
                "        embed_dim = len(val[0])",
                "        n = len(val)",
                "        return [sum(row[j] for row in val) / n for j in range(embed_dim)]",
                "    vec = _resolve_vector(x, state)",
                "    if not vec:",
                "        return vec",
                "    mask_vec = _resolve_vector(mask, state) if mask else [1.0] * len(vec)",
                "    total = sum(mask_vec)",
                "    if total == 0.0:",
                "        return [0.0] * len(vec)",
                "    return [v * m / total for v, m in zip(vec, mask_vec)]",
                "",
                "",
                "def _tensor_cls_pooling(x: str, state: dict) -> float:",
                "    vec = _resolve_vector(x, state)",
                "    return vec[0] if vec else 0.0",
                "",
                "",
                "def _tensor_positional_encoding(x: str, pos: str, state: dict) -> list[float]:",
                "    vec = _resolve_vector(x, state)",
                "    pos_vals = _resolve_vector(pos, state) if pos else list(range(len(vec)))",
                "    dim = len(vec)",
                "    result = []",
                "    for i, (v, p) in enumerate(zip(vec, pos_vals)):",
                "        angle = float(p) / (10000 ** ((2 * (i // 2)) / max(dim, 1)))",
                "        enc = math.sin(angle) if i % 2 == 0 else math.cos(angle)",
                "        result.append(v + enc)",
                "    return result",
                "",
                "",
                "def _tensor_embedding_lookup(table: str, ids: str, state: dict) -> list:",
                "    tbl = state.get(table, [])",
                "    id_vec = _resolve_vector(ids, state)",
                "    if not isinstance(tbl, list) or not tbl:",
                "        return id_vec",
                "    return [tbl[int(i) % len(tbl)] if isinstance(tbl[0], list) else float(tbl[int(i) % len(tbl)]) for i in id_vec]",
                "",
                "",
                "def _tensor_attention(q: str, k: str, v: str, mask: str, state: dict) -> list[float]:",
                "    vq = _resolve_vector(q, state)",
                "    vk = _resolve_vector(k, state)",
                "    vv = _resolve_vector(v, state)",
                "    vmask = _resolve_vector(mask, state) if mask else None",
                "    if not vq or not vk:",
                "        return vv",
                "    scale = math.sqrt(max(len(vq), 1))",
                "    score = sum(qi * ki for qi, ki in zip(vq, vk)) / scale",
                "    if vmask:",
                "        score += sum(mi for mi in vmask)",
                "    weight = 1.0 / (1.0 + math.exp(-score))",
                "    return [weight * vi for vi in vv]",
                "",
                "",
                "def _layer_call_passthrough(input_name: str, layer_name: str, state: dict, parameters: dict) -> Any:",
                "    # P10 LayerCall lowering is manifest/namespace only: validates layer existence and",
                "    # records parameters in the manifest, but does not execute the layer body.",
                "    # Full body execution (composing layer operations) is P11 scope.",
                "    vec = _resolve_vector(input_name, state)",
                "    return vec",
            ]
        )
