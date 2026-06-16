# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field

from matrixai.ir import MatrixAIProgram


@dataclass(frozen=True)
class OptimizationSuggestion:
    kind: str
    description: str
    nodes: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass(frozen=True)
class OptimizationReport:
    suggestions: list[OptimizationSuggestion] = field(default_factory=list)

    @property
    def has_suggestions(self) -> bool:
        return bool(self.suggestions)

    def summary(self) -> str:
        if not self.suggestions:
            return "OptimizerAgent: no suggestions. Graph looks clean."
        lines = [f"OptimizerAgent: {len(self.suggestions)} suggestion(s)"]
        for s in self.suggestions:
            nodes_str = f" [{', '.join(s.nodes)}]" if s.nodes else ""
            detail_str = f" — {s.detail}" if s.detail else ""
            lines.append(f"  SUGGEST {s.kind}{nodes_str}: {s.description}{detail_str}")
        return "\n".join(lines)


class OptimizerAgent:
    """Analyzes a MatrixAI IR graph and emits optimization suggestions.

    In MVP mode the agent only suggests; it never modifies the graph.
    """

    # Maximum number of nodes before suggesting graph simplification
    _NODE_WARN_THRESHOLD = 7
    # Minimum number of edges from a node before suggesting merge
    _FAN_IN_THRESHOLD = 3

    def analyze(self, program: MatrixAIProgram) -> OptimizationReport:
        suggestions: list[OptimizationSuggestion] = []

        self._check_merge_linear_activation(program, suggestions)
        self._check_cache_embedding(program, suggestions)
        self._check_prune_isolated_nodes(program, suggestions)
        self._check_graph_complexity(program, suggestions)
        self._check_unknown_expressions(program, suggestions)
        self._check_fan_in_nodes(program, suggestions)

        return OptimizationReport(suggestions=suggestions)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_merge_linear_activation(
        self, program: MatrixAIProgram, suggestions: list[OptimizationSuggestion]
    ) -> None:
        """Suggest merging a linear function immediately followed by a sigmoid threshold."""
        function_names = [f.name for f in program.functions]
        for i, fn in enumerate(program.functions):
            if fn.semantic.kind == "softmax_linear" and i + 1 < len(program.functions):
                next_fn = program.functions[i + 1]
                if next_fn.semantic.kind == "sigmoid_threshold":
                    suggestions.append(
                        OptimizationSuggestion(
                            kind="merge_linear_activation",
                            description=(
                                "consecutive softmax + sigmoid_threshold can be fused "
                                "into a single operation to reduce graph steps"
                            ),
                            nodes=[fn.name, next_fn.name],
                            detail=f"fuse {fn.name} → {next_fn.name}",
                        )
                    )
            if fn.semantic.kind == "sigmoid_linear" and i + 1 < len(program.functions):
                next_fn = program.functions[i + 1]
                if next_fn.semantic.kind == "sigmoid_threshold":
                    suggestions.append(
                        OptimizationSuggestion(
                            kind="merge_linear_activation",
                            description=(
                                "consecutive sigmoid_linear + sigmoid_threshold can be fused "
                                "into a single operation to reduce graph steps"
                            ),
                            nodes=[fn.name, next_fn.name],
                            detail=f"fuse {fn.name} → {next_fn.name}",
                        )
                    )

    def _check_cache_embedding(
        self, program: MatrixAIProgram, suggestions: list[OptimizationSuggestion]
    ) -> None:
        """Suggest caching the input vector when it fans out to multiple functions."""
        vector_names = {v.name for v in program.vectors}
        edge_counts: dict[str, int] = {}
        for src, _ in program.graph.edges:
            if src in vector_names:
                edge_counts[src] = edge_counts.get(src, 0) + 1

        for name, count in edge_counts.items():
            if count >= 2:
                suggestions.append(
                    OptimizationSuggestion(
                        kind="cache_embedding",
                        description=(
                            f"vector {name} feeds {count} downstream nodes; "
                            "caching its representation avoids repeated computation"
                        ),
                        nodes=[name],
                        detail=f"fan-out={count}",
                    )
                )

    def _check_prune_isolated_nodes(
        self, program: MatrixAIProgram, suggestions: list[OptimizationSuggestion]
    ) -> None:
        """Detect declared nodes that are not part of the graph."""
        graph_node_set = set(program.graph.nodes)
        declared: list[str] = (
            [v.name for v in program.vectors]
            + [f.name for f in program.functions]
            + [d.name for d in program.distributions]
            + [a.name for a in program.actions]
        )
        isolated = [n for n in declared if n not in graph_node_set]
        if isolated:
            suggestions.append(
                OptimizationSuggestion(
                    kind="prune_isolated_nodes",
                    description=(
                        "declared nodes not referenced in GRAPH; "
                        "removing them reduces IR size and parse time"
                    ),
                    nodes=isolated,
                    detail=f"{len(isolated)} isolated node(s)",
                )
            )

    def _check_graph_complexity(
        self, program: MatrixAIProgram, suggestions: list[OptimizationSuggestion]
    ) -> None:
        """Warn when graph depth exceeds threshold."""
        count = len(program.graph.nodes)
        if count > self._NODE_WARN_THRESHOLD:
            suggestions.append(
                OptimizationSuggestion(
                    kind="simplify_graph",
                    description=(
                        f"graph has {count} nodes which exceeds the recommended maximum "
                        f"of {self._NODE_WARN_THRESHOLD} for low-latency inference"
                    ),
                    nodes=program.graph.nodes,
                    detail=f"node_count={count}",
                )
            )

    def _check_unknown_expressions(
        self, program: MatrixAIProgram, suggestions: list[OptimizationSuggestion]
    ) -> None:
        """Flag functions with unknown semantic kind; they cannot be optimized."""
        for fn in program.functions:
            if fn.semantic.kind == "unknown":
                suggestions.append(
                    OptimizationSuggestion(
                        kind="annotate_expression",
                        description=(
                            f"function {fn.name} has unknown expression semantics; "
                            "annotating it enables fusion and quantization passes"
                        ),
                        nodes=[fn.name],
                        detail=f"expression='{fn.expression}'",
                    )
                )

    def _check_fan_in_nodes(
        self, program: MatrixAIProgram, suggestions: list[OptimizationSuggestion]
    ) -> None:
        """Suggest reviewing nodes with high in-degree (multiple inputs)."""
        in_degree: dict[str, int] = {}
        for _, tgt in program.graph.edges:
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

        for node, degree in in_degree.items():
            if degree >= self._FAN_IN_THRESHOLD:
                suggestions.append(
                    OptimizationSuggestion(
                        kind="review_fan_in",
                        description=(
                            f"node {node} has {degree} incoming edges; "
                            "consider merging upstream nodes or using a shared representation"
                        ),
                        nodes=[node],
                        detail=f"in_degree={degree}",
                    )
                )
