# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field

from matrixai.ir import MatrixAIProgram
from matrixai.types import check_program_types


@dataclass(frozen=True)
class VerificationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class VerifierAgent:
    def verify(self, program: MatrixAIProgram) -> VerificationResult:
        errors: list[str] = []
        warnings: list[str] = []

        declared_nodes = self._declared_nodes(program)
        graph_nodes = set(program.graph.nodes)

        if not program.graph.nodes:
            errors.append("GRAPH is empty")

        for node in graph_nodes:
            if node not in declared_nodes:
                errors.append(f"GRAPH references undeclared node '{node}'")

        for action in program.actions:
            if action.name not in graph_nodes:
                errors.append(f"ACTION {action.name} is not present in GRAPH")
            if action.condition.operator not in {">", ">=", "<", "<="}:
                errors.append(f"ACTION {action.name} has unsupported operator")
            if action.condition.source.split(".", 1)[0] not in declared_nodes:
                errors.append(
                    f"ACTION {action.name} condition references undeclared source "
                    f"'{action.condition.source}'"
                )
            if not action.policy:
                errors.append(f"ACTION {action.name} requires a policy")

        for distribution in program.distributions:
            if not distribution.source:
                errors.append(f"DISTRIBUTION {distribution.name} has no source")

        if program.audit.explain:
            missing = [node for node in program.audit.explain if node not in graph_nodes]
            if missing:
                errors.append(f"AUDIT references nodes outside GRAPH: {', '.join(missing)}")
        else:
            warnings.append("AUDIT has no EXPLAIN path")

        if self._has_cycle(program):
            errors.append("GRAPH has a cycle; MVP supports acyclic graphs only")

        type_result = check_program_types(program)
        errors.extend(type_result.errors)
        warnings.extend(type_result.warnings)

        return VerificationResult(errors=errors, warnings=warnings)

    def _declared_nodes(self, program: MatrixAIProgram) -> set[str]:
        nodes = {vector.name for vector in program.vectors}
        # TRANSFORMER C6: SEQUENCE es un input de GRAPH tan válido como VECTOR
        # (BLOCK TRANSFORMER, contrato 51) — faltaba desde C1 y solo lo
        # detectó el cierre duro (mx export-bundle sobre un .mxai real con
        # GRAPH Texto -> N, antes rechazado como "undeclared node").
        nodes.update(sequence.name for sequence in getattr(program, "sequences", []))
        nodes.update(function.name for function in program.functions)
        nodes.update(distribution.name for distribution in program.distributions)
        nodes.update(action.name for action in program.actions)
        nodes.update(network.name for network in getattr(program, "networks", []))
        return nodes

    def _has_cycle(self, program: MatrixAIProgram) -> bool:
        edges_by_source: dict[str, list[str]] = {}
        for source, target in program.graph.edges:
            edges_by_source.setdefault(source, []).append(target)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for next_node in edges_by_source.get(node, []):
                if visit(next_node):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in program.graph.nodes)