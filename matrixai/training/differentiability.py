# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matrixai.compiler import BackendContractAnalyzer, BackendContractReport
from matrixai.ir import MatrixAIProgram
from matrixai.training.spec import TrainingSpec


@dataclass(frozen=True)
class DifferentiabilityVerificationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prediction_node: str = ""
    parameter_paths: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "prediction_node": self.prediction_node,
            "parameter_paths": {key: list(value) for key, value in self.parameter_paths.items()},
        }


class DifferentiabilityVerifier:
    def verify(
        self,
        training: TrainingSpec,
        program: MatrixAIProgram,
        backend_report: BackendContractReport | None = None,
    ) -> DifferentiabilityVerificationResult:
        report = backend_report or BackendContractAnalyzer().analyze(program)
        prediction_node = self._prediction_node(program, training.loss.prediction)
        errors: list[str] = []
        warnings: list[str] = []
        parameter_paths: dict[str, list[str]] = {}

        if not prediction_node:
            return DifferentiabilityVerificationResult(
                errors=[f"LOSS prediction has no differentiable graph node: {training.loss.prediction}"],
                warnings=warnings,
            )

        from matrixai.training.trainer import match_update_patterns

        node_reports = {node.node: node for node in report.nodes}
        trainable_by_name = self._trainable_parameter_map(report)
        graph = self._adjacency(program)
        layer_to_node = self._layer_to_node_map(program)
        all_param_keys = list(trainable_by_name.keys())

        prediction_report = node_reports.get(prediction_node)
        if prediction_report is None:
            errors.append(f"LOSS prediction node is not present in backend report: {prediction_node}")
        elif not prediction_report.differentiable:
            errors.append(f"LOSS prediction node is not differentiable: {prediction_node}")

        seen_params: set[tuple[str, str]] = set()
        for update in training.optimizer.update:
            matched_keys = match_update_patterns([update], all_param_keys)
            if not matched_keys:
                continue
            for matched_key in matched_keys:
                parameter = trainable_by_name.get(matched_key)
                if parameter is None:
                    continue
                param_id = (str(parameter["function"]), str(parameter["name"]))
                if param_id in seen_params:
                    continue
                seen_params.add(param_id)
                layer_or_fn = str(parameter["function"])
                function_node = layer_to_node.get(layer_or_fn, layer_or_fn)
                path = self._find_path(graph, function_node, prediction_node)
                if not path:
                    errors.append(
                        f"UPDATE parameter {matched_key} is not connected to LOSS prediction {training.loss.prediction}"
                    )
                    continue
                parameter_paths[matched_key] = path
                for node in path:
                    node_report = node_reports.get(node)
                    if node_report is None:
                        errors.append(f"Differentiability path contains unknown node: {node}")
                        continue
                    if not node_report.supported:
                        errors.append(
                            f"Differentiability path for {matched_key} is blocked by unsupported node {node}: {node_report.reason}"
                        )
                    elif not node_report.differentiable:
                        errors.append(
                            f"Differentiability path for {matched_key} crosses runtime boundary {node}: {node_report.reason}"
                        )

        if not parameter_paths and not errors:
            warnings.append("no differentiability paths were verified")

        return DifferentiabilityVerificationResult(
            errors=errors,
            warnings=warnings,
            prediction_node=prediction_node,
            parameter_paths=parameter_paths,
        )

    def _prediction_node(self, program: MatrixAIProgram, prediction: str) -> str:
        for function in program.functions:
            if function.output == prediction or function.name == prediction:
                return function.name
        for distribution in program.distributions:
            if distribution.variable == prediction or distribution.name == prediction:
                return distribution.name
        for network in getattr(program, "networks", []):
            if network.output == prediction or network.name == prediction:
                return network.name
        return ""

    def _trainable_parameter_map(self, report: BackendContractReport) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for parameter in report.trainable_parameters:
            payload = parameter.to_dict()
            name = str(payload["name"])
            function = str(payload["function"])
            result[name] = payload
            result[f"{function}.{name}"] = payload
        return result

    def _layer_to_node_map(self, program: MatrixAIProgram) -> dict[str, str]:
        """Map LAYER names to the FUNCTION graph node that calls them via layer_call."""
        result: dict[str, str] = {}
        for fn in program.functions:
            if fn.semantic.kind == "layer_call":
                layer_name = fn.semantic.parameters.get("layer", "")
                if layer_name:
                    result[layer_name] = fn.name
        return result

    def _adjacency(self, program: MatrixAIProgram) -> dict[str, list[str]]:
        graph = {node: [] for node in program.graph.nodes}
        for left, right in program.graph.edges:
            graph.setdefault(left, []).append(right)
            graph.setdefault(right, [])
        return graph

    def _find_path(self, graph: dict[str, list[str]], start: str, target: str) -> list[str]:
        if start == target:
            return [start]
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for next_node in graph.get(node, []):
                if next_node in path:
                    continue
                next_path = path + [next_node]
                if next_node == target:
                    return next_path
                stack.append((next_node, next_path))
        return []