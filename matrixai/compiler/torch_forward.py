# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from matrixai.compiler.backend_contract import BackendContractAnalyzer
from matrixai.ir import ActionSpec, DistributionSpec, FunctionSpec, MatrixAIProgram, SequenceSpec, VectorSpec
from matrixai.parameters.store import ParameterSet, build_initial_parameter_set, validate_parameter_set
from matrixai.parameters.tensor_bridge import TensorParameterBridgeError, parameter_set_to_torch_tensors
from matrixai.types import validate_value_against_type


class TorchForwardError(ValueError):
    pass


@dataclass(frozen=True)
class TorchForwardRunner:
    device: str = "cpu"
    dtype: str = "float32"

    def run(
        self,
        program: MatrixAIProgram,
        input_data: dict[str, Any],
        parameter_set: ParameterSet | None = None,
    ) -> dict[str, Any]:
        self._validate_config()
        report = BackendContractAnalyzer(target="torch").analyze(program)
        # Auditoría C4 [ALTA-3]: este runner ejecuta el mundo LAYER/FUNCTION y
        # no tiene rama para NETWORKs — gatea por la capacidad forward_ok de
        # cada nodo además del agregado (para nodos legacy forward_ok ==
        # supported: idéntico). Sin esto, un programa con una red cuyo
        # entrenamiento sí está soportado devolvería "éxito" OMITIENDO la red
        # y su salida en silencio.
        forward_blocked = [node.node for node in report.nodes if not node.forward_ok]
        if not report.ok or forward_blocked:
            issues = sorted(
                set(
                    [node.node for node in report.unsupported_nodes]
                    + forward_blocked
                )
            ) + list(report.parameter_errors)
            raise TorchForwardError(
                f"Program {program.project} is not portable to torch forward: {', '.join(issues)}"
            )

        parameters = parameter_set or build_initial_parameter_set(program)
        validation = validate_parameter_set(program, parameters)
        if not validation.ok:
            raise TorchForwardError("; ".join(validation.errors))

        try:
            parameter_tensors = parameter_set_to_torch_tensors(parameters)
        except TensorParameterBridgeError as exc:
            raise TorchForwardError(str(exc)) from exc
        if self.device != "cpu":
            parameter_tensors = {k: v.to(self.device) for k, v in parameter_tensors.items()}
        torch = import_module("torch")

        state: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        runtime_boundaries: list[dict[str, Any]] = []

        vectors = {vector.name: vector for vector in program.vectors}
        sequences = {seq.name: seq for seq in program.sequences}
        functions = {function.name: function for function in program.functions}
        distributions = {distribution.name: distribution for distribution in program.distributions}
        action_specs = {action.name: action for action in program.actions}
        layer_map = {layer.name: layer for layer in program.layers}

        for node in program.graph.nodes:
            if node in vectors:
                value = self._load_vector(vectors[node], input_data)
                state[node] = value
                for field, field_value in zip(vectors[node].fields, value):
                    state.setdefault(field, field_value)
                trace.append(
                    self._trace_step(trace, node=node, node_type="vector", value=value, output_ref=node)
                )
            elif node in sequences:
                value = self._load_sequence(sequences[node], input_data)
                state[node] = value
                trace.append(
                    self._trace_step(trace, node=node, node_type="sequence", value=value, output_ref=node)
                )
            elif node in functions:
                value, boundary = self._evaluate_function(
                    functions[node], state, parameter_tensors, torch, layer_map
                )
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
                if boundary:
                    runtime_boundaries.append(
                        {
                            "node": node,
                            "node_type": "function",
                            "kind": functions[node].semantic.kind,
                            "output_ref": functions[node].output,
                            "value": value,
                        }
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
                runtime_boundaries.append(
                    {
                        "node": node,
                        "node_type": "distribution",
                        "kind": distributions[node].distribution_type,
                        "output_ref": distributions[node].variable,
                        "value": value,
                    }
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
                runtime_boundaries.append(
                    {
                        "node": node,
                        "node_type": "action",
                        "kind": action_specs[node].policy,
                        "call": action_specs[node].call,
                        "value": action_result,
                    }
                )

        report_dict = report.to_dict()
        return {
            "target": "torch",
            "project": program.project,
            "parameter_set_id": parameters.parameter_set_id,
            "state": state,
            "trace": trace,
            "actions": actions,
            "runtime_boundaries": runtime_boundaries,
            "tensor_shapes": report_dict["tensor_shapes"],
            "type_constraints": report_dict["type_constraints"],
            "trainable_parameters": report_dict["trainable_parameters"],
            "parameter_manifest": report_dict["parameter_manifest"],
            "autodiff_plan": report_dict["autodiff_plan"],
            "backend_report": report_dict,
            "backend": report_dict["backend"],
        }

    def _load_sequence(self, seq: SequenceSpec, input_data: dict[str, Any]) -> list[int]:
        source = input_data.get(seq.name, input_data)
        if isinstance(source, list):
            tokens = [int(t) % seq.vocab_size for t in source]
        elif isinstance(source, dict):
            tokens = [int(source.get(f"t{i}", 0)) % seq.vocab_size for i in range(seq.length)]
        else:
            tokens = [0] * seq.length
        if len(tokens) != seq.length:
            raise TorchForwardError(
                f"SEQUENCE {seq.name} expected {seq.length} tokens, got {len(tokens)}"
            )
        return tokens

    def _load_vector(self, vector: VectorSpec, input_data: dict[str, Any]) -> list[float]:
        source = input_data.get(vector.name, input_data)
        values: list[float] = []
        for field in vector.fields:
            value = float(source.get(field, 0.0))
            errors = validate_value_against_type(
                f"{vector.name}.{field}", value, vector.field_types.get(field)
            )
            if errors:
                raise TorchForwardError("; ".join(errors))
            values.append(value)
        if len(values) != vector.size:
            raise TorchForwardError(f"Vector {vector.name} expected {vector.size} values")
        return values

    def _evaluate_function(
        self,
        function: FunctionSpec,
        state: dict[str, Any],
        parameter_tensors: dict[str, Any],
        torch,
        layer_map: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        kind = function.semantic.kind
        if kind == "layer_call":
            return self._evaluate_layer_call(function, state, parameter_tensors, torch, layer_map or {}), False
        if kind == "softmax_linear":
            vector = self._vector_tensor(function, state, torch)
            weights = self._parameter_tensor(parameter_tensors, function, "weights")
            bias = self._parameter_tensor(parameter_tensors, function, "bias")
            labels = [
                str(label)
                for label in (function.semantic.parameters.get("labels") or ["support", "sales", "operations"])
            ]
            logits = torch.matmul(weights, vector) + bias
            probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
            return {label: float(probabilities[index]) for index, label in enumerate(labels)}, False
        if kind == "sigmoid_linear":
            vector = self._vector_tensor(function, state, torch)
            weights = self._parameter_tensor(parameter_tensors, function, "weights")
            bias = self._parameter_tensor(parameter_tensors, function, "bias")
            logit = torch.dot(weights.reshape(-1), vector.reshape(-1)) + bias.reshape(())
            return float(torch.sigmoid(logit).detach().cpu().item()), False
        if kind == "sigmoid_threshold":
            source = function.semantic.inputs[0]
            scale = float(function.semantic.parameters["scale"])
            threshold = float(function.semantic.parameters["threshold"])
            value = self._sigmoid(scale * (self._resolve_numeric(source, state) - threshold))
            return value, True
        raise TorchForwardError(f"Unsupported torch forward function kind {kind!r} for FUNCTION {function.name}")

    def _evaluate_layer_call(
        self,
        function: FunctionSpec,
        state: dict[str, Any],
        parameter_tensors: dict[str, Any],
        torch,
        layer_map: dict[str, Any],
    ) -> list[float]:
        layer_name = function.semantic.parameters.get("layer", "")
        layer = layer_map.get(layer_name)
        if layer is None:
            raise TorchForwardError(f"layer_call references undefined LAYER: {layer_name!r}")

        input_name = function.semantic.inputs[0] if function.semantic.inputs else ""
        raw_input = state.get(input_name, [])

        local_state: dict[str, Any] = {}
        local_state["input"] = self._to_tensor(raw_input, torch)

        for param in layer.params:
            qualified_key = f"{layer_name}.{param.name}"
            if qualified_key in parameter_tensors:
                local_state[param.name] = parameter_tensors[qualified_key]
            elif param.name in parameter_tensors:
                local_state[param.name] = parameter_tensors[param.name]

        for op in layer.body_ops:
            local_state[op.output] = self._execute_body_op_torch(op, local_state, torch)

        last_output = layer.body_ops[-1].output if layer.body_ops else ""
        result = local_state.get("result", local_state.get(last_output))
        if result is None:
            return []
        if hasattr(result, "tolist"):
            return result.tolist()
        return list(result)

    def _execute_body_op_torch(self, op: Any, local_state: dict[str, Any], torch) -> Any:
        args = op.args

        def t(i: int) -> Any:
            name = args[i] if i < len(args) else ""
            return self._to_tensor(local_state.get(name), torch)

        k = op.kind
        if k == "matmul":
            return torch.matmul(t(0), t(1))
        if k == "scale":
            factor_name_or_val = args[1] if len(args) > 1 else "1.0"
            try:
                factor = float(factor_name_or_val)
            except ValueError:
                factor = float(self._to_tensor(local_state.get(factor_name_or_val), torch).item())
            return t(0) * factor
        if k == "softmax":
            val = t(0)
            if val.dim() == 0:
                return torch.ones_like(val)
            return torch.softmax(val, dim=-1)
        if k == "dot":
            return torch.dot(t(0).reshape(-1), t(1).reshape(-1))
        if k == "relu":
            return torch.relu(t(0))
        if k == "gelu":
            return torch.nn.functional.gelu(t(0))
        if k == "residual":
            a, b = t(0), t(1)
            return a + b
        if k == "layer_norm":
            x = t(0)
            gain = t(1)
            bias_v = t(2)
            eps = float(args[3]) if len(args) > 3 else 1e-5
            return torch.layer_norm(x, [x.shape[-1]], weight=gain, bias=bias_v, eps=eps)
        if k == "attention":
            vq, vk, vv = t(0), t(1), t(2)
            scale = torch.sqrt(torch.tensor(float(max(vq.shape[-1], 1)), device=self.device))
            score = torch.dot(vq.reshape(-1), vk.reshape(-1)) / scale
            if len(args) > 3 and args[3]:
                mask_t = self._to_tensor(local_state.get(args[3]), torch)
                if mask_t.numel() > 0:
                    score = score + mask_t.sum()
            weight = torch.sigmoid(score)
            return weight * vv
        if k == "embedding_lookup":
            table_val = local_state.get(args[0]) if args else None
            ids_val = local_state.get(args[1]) if len(args) > 1 else None
            table_t = self._to_tensor(table_val, torch)
            if table_t.dim() < 2:
                return table_t
            if ids_val is None:
                return table_t
            ids_t = self._to_tensor(ids_val, torch)
            ids_long = ids_t.long() % table_t.shape[0]
            return table_t[ids_long]
        if k == "mean_pooling":
            x = t(0)
            if x.dim() > 1:
                return x.float().mean(dim=0)
            return x
        if k == "cls_pooling":
            x = t(0)
            if x.dim() > 1:
                return x[0]
            return x
        if k == "positional_encoding":
            x = t(0)
            seq_len = x.shape[0] if x.dim() > 1 else 1
            embed_dim = x.shape[-1] if x.dim() > 1 else x.shape[0]
            positions = torch.arange(seq_len, dtype=torch.float32, device=self.device).unsqueeze(1)
            dim_idx = torch.arange(embed_dim, dtype=torch.float32, device=self.device).unsqueeze(0)
            angles = positions / (10000 ** (2 * (dim_idx // 2) / embed_dim))
            encoding = torch.where(dim_idx % 2 == 0, torch.sin(angles), torch.cos(angles))
            return (x.float() + encoding) if x.dim() > 1 else x.float()
        raise TorchForwardError(f"Unsupported torch body op: {k!r}")

    def _to_tensor(self, value: Any, torch) -> Any:
        if value is None:
            return torch.tensor([], dtype=self._torch_dtype(torch), device=self.device)
        if hasattr(value, "dtype"):
            return value
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return torch.tensor(value, dtype=self._torch_dtype(torch), device=self.device)
            return torch.tensor(value, dtype=self._torch_dtype(torch), device=self.device)
        return torch.tensor(float(value), dtype=self._torch_dtype(torch), device=self.device)

    def _vector_tensor(self, function: FunctionSpec, state: dict[str, Any], torch) -> Any:
        vector_value: Any = None
        if function.semantic.inputs:
            vector_value = state.get(function.semantic.inputs[0])
        if not isinstance(vector_value, list):
            for value in state.values():
                if isinstance(value, list):
                    vector_value = value
                    break
        if not isinstance(vector_value, list):
            raise TorchForwardError(f"FUNCTION {function.name} has no tensorizable VECTOR input")
        return torch.tensor(vector_value, dtype=self._torch_dtype(torch), device=self.device)

    def _parameter_tensor(self, parameter_tensors: dict[str, Any], function: FunctionSpec, role: str) -> Any:
        name = function.semantic.parameters.get(role)
        if not name:
            raise TorchForwardError(f"FUNCTION {function.name} missing {role} parameter")
        qualified = f"{function.name}.{name}"
        if qualified in parameter_tensors:
            return parameter_tensors[qualified]
        if name in parameter_tensors:
            return parameter_tensors[name]
        raise TorchForwardError(f"Missing tensor for parameter {qualified}")

    def _evaluate_distribution(self, distribution: DistributionSpec, state: dict[str, Any]) -> dict[str, Any]:
        if distribution.distribution_type == "Categorical":
            source = state.get(distribution.source)
            if not isinstance(source, dict):
                raise TorchForwardError(f"Categorical source {distribution.source} was not produced")
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
        raise TorchForwardError(f"Unsupported distribution: {distribution.distribution_type}")

    def _evaluate_action(self, action: ActionSpec, state: dict[str, Any]) -> dict[str, Any]:
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

    def _compare(self, value: float, operator: str, threshold: float) -> bool:
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<":
            return value < threshold
        if operator == "<=":
            return value <= threshold
        raise TorchForwardError(f"Unsupported operator: {operator}")

    def _trace_step(self, trace: list[dict[str, Any]], **data: Any) -> dict[str, Any]:
        return {"step": len(trace) + 1, "status": "ok", **data}

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def _validate_config(self) -> None:
        if self.dtype != "float32":
            raise TorchForwardError("Torch forward supports only dtype='float32'")
        if self.device != "cpu":
            from matrixai.parameters.tensor_bridge import torch_device_info
            info = torch_device_info()
            if self.device not in info["available_devices"]:
                avail = ", ".join(info["available_devices"])
                raise TorchForwardError(
                    f"Device {self.device!r} is not available in this environment. Available: {avail}"
                )

    def _torch_dtype(self, torch):
        if self.dtype == "float32":
            return torch.float32
        raise TorchForwardError(f"Unsupported torch forward dtype {self.dtype!r}")