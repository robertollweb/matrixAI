# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matrixai.agents import (
    ArchitectAgent,
    ArchitectSpecError,
    PlannerVerifier,
    SafetyAgent,
    VerifierAgent,
)
from matrixai.compiler import PythonBackendCompiler
from matrixai.ir import MatrixAIProgram
from matrixai.parser import MatrixAIParseError, parse_text
from matrixai.runtime import MatrixAIRuntime
from matrixai.types import TypeSpec, parse_type_spec


@dataclass(frozen=True)
class LanguageDiagnostic:
    severity: str
    message: str
    source: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
        }
        if self.line is not None:
            data["line"] = self.line
        return data


@dataclass(frozen=True)
class LanguageToolReport:
    path: str
    language: str
    diagnostics: list[LanguageDiagnostic] = field(default_factory=list)
    formatted: bool = True

    @property
    def ok(self) -> bool:
        return not any(diagnostic.severity == "error" for diagnostic in self.diagnostics)

    @property
    def has_warnings(self) -> bool:
        return any(diagnostic.severity == "warning" for diagnostic in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "language": self.language,
            "formatted": self.formatted,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class RuntimeCompilerReport:
    project: str
    ok: bool
    mismatches: list[str]
    runtime_result: dict[str, Any]
    compiled_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project": self.project,
            "mismatches": list(self.mismatches),
            "runtime_result": self.runtime_result,
            "compiled_result": self.compiled_result,
        }


def detect_language(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".semantic":
        return "semantic"
    if suffix == ".mxai":
        return "mxai"
    raise ValueError("Language tooling supports .semantic and .mxai files")


def lint_path(path: str | Path) -> LanguageToolReport:
    source_path = Path(path)
    return lint_source(
        source_path.read_text(encoding="utf-8"),
        detect_language(source_path),
        path=str(source_path),
    )


def lint_source(text: str, language: str, path: str = "<memory>") -> LanguageToolReport:
    if language == "semantic":
        return _lint_semantic(text, path)
    if language == "mxai":
        return _lint_mxai(text, path)
    raise ValueError("Language tooling supports semantic and mxai sources")


def format_path(path: str | Path) -> str:
    source_path = Path(path)
    return format_source(source_path.read_text(encoding="utf-8"), detect_language(source_path))


def format_source(text: str, language: str) -> str:
    if language == "semantic":
        return format_semantic(text)
    if language == "mxai":
        return format_mxai(text)
    raise ValueError("Language tooling supports semantic and mxai sources")


def graph_path(path: str | Path, output_format: str = "mermaid") -> str:
    source_path = Path(path)
    program = _program_from_source(
        source_path.read_text(encoding="utf-8"),
        detect_language(source_path),
    )
    return graph_program(program, output_format)


def graph_source(text: str, language: str, output_format: str = "mermaid") -> str:
    return graph_program(_program_from_source(text, language), output_format)


def graph_program(program: MatrixAIProgram, output_format: str = "mermaid") -> str:
    normalized = output_format.lower()
    if normalized == "mermaid":
        return _graph_mermaid(program)
    if normalized == "dot":
        return _graph_dot(program)
    if normalized == "json":
        import json

        return json.dumps(_graph_json(program), indent=2, ensure_ascii=False) + "\n"
    raise ValueError("Graph format must be mermaid, dot or json")


def diagnose_runtime_compiler(
    program: MatrixAIProgram,
    input_data: dict[str, Any],
    tolerance: float = 1e-9,
) -> RuntimeCompilerReport:
    runtime_result = MatrixAIRuntime().run(program, input_data)
    namespace: dict[str, Any] = {}
    exec(PythonBackendCompiler().compile(program), namespace)
    compiled_result = namespace["run"](input_data)
    mismatches: list[str] = []
    _compare_values(runtime_result, compiled_result, "result", tolerance, mismatches)
    return RuntimeCompilerReport(
        project=program.project,
        ok=not mismatches,
        mismatches=mismatches,
        runtime_result=runtime_result,
        compiled_result=compiled_result,
    )


def format_semantic(text: str) -> str:
    spec = ArchitectAgent().parse_semantic_spec(text)
    lines: list[str] = [f"PROJECT {spec.project}"]
    if spec.intent:
        lines.append(f"INTENT {spec.intent}")
    lines.extend([f"MODE {spec.mode}", f"ENTITY {spec.entity}", ""])

    lines.append(f"FIELDS {spec.entity}")
    lines.extend(f"  {field}" for field in spec.fields)
    lines.extend(["END", ""])

    lines.extend(f"GOAL {goal}" for goal in spec.goals)
    if spec.rules:
        if spec.goals:
            lines.append("")
        lines.append("RULES")
        lines.extend(f"  {rule}" for rule in spec.rules)
        lines.append("END")
    if spec.goals or spec.rules:
        lines.append("")

    constraint_source = "confidence" if spec.mode == "classification" else "risk"
    lines.append(f"CONSTRAINT {constraint_source} > {_format_decimal(spec.confidence_threshold)}")
    lines.append(f"ACTION_THRESHOLD {_format_decimal(spec.action_threshold)}")
    lines.extend(["", f"ACTION {spec.action.name}"])
    lines.append(f"  POLICY {spec.action.policy}")
    lines.append(f"  CALL {spec.action.call}")
    lines.append("END")
    return "\n".join(lines).rstrip() + "\n"


def format_mxai(text: str) -> str:
    program = parse_text(text)
    blocks: list[str] = [f"PROJECT {program.project}"]

    for vector in program.vectors:
        lines = [f"VECTOR {vector.name}[{vector.size}]"]
        for field_name in vector.fields:
            field_type = vector.field_types.get(field_name)
            if field_type is None:
                lines.append(f"  {field_name}")
            else:
                lines.append(f"  {field_name}: {_type_to_source(field_type)}")
        lines.append("END")
        blocks.append("\n".join(lines))

    for parameter in program.parameters:
        lines = [f"PARAM {parameter.name} {_type_to_source(parameter.type_spec)}"]
        lines.append(f"  TRAINABLE {str(parameter.trainable).lower()}")
        if parameter.initializer:
            lines.append(f"  INIT {parameter.initializer}")
        lines.append("END")
        blocks.append("\n".join(lines))

    for function in program.functions:
        output = function.output
        if function.output_type is not None:
            output = f"{output}: {_type_to_source(function.output_type)}"
        blocks.append(
            "\n".join(
                [
                    f"FUNCTION {function.name}",
                    f"  {output} = {function.expression}",
                    "END",
                ]
            )
        )

    for distribution in program.distributions:
        blocks.append(
            "\n".join(
                [
                    f"DISTRIBUTION {distribution.name}",
                    f"  {distribution.raw}",
                    "END",
                ]
            )
        )

    graph_lines = ["GRAPH"]
    graph_lines.extend(f"  {line}" for line in _graph_lines(program))
    graph_lines.append("END")
    blocks.append("\n".join(graph_lines))

    for action in program.actions:
        blocks.append(
            "\n".join(
                [
                    f"ACTION {action.name}",
                    f"  WHEN {action.condition.source} {action.condition.operator} {_format_decimal(action.condition.threshold)}",
                    f"  POLICY {action.policy}",
                    f"  CALL {action.call}",
                    "END",
                ]
            )
        )

    audit_lines = ["AUDIT"]
    if program.audit.explain:
        audit_lines.append(f"  EXPLAIN {' -> '.join(program.audit.explain)}")
    audit_lines.append("END")
    blocks.append("\n".join(audit_lines))

    return "\n\n".join(blocks).rstrip() + "\n"


def _program_from_source(text: str, language: str) -> MatrixAIProgram:
    if language == "mxai":
        return parse_text(text)
    if language == "semantic":
        architect = ArchitectAgent()
        return parse_text(architect.to_mxai(architect.plan_from_text(text)))
    raise ValueError("Language tooling supports semantic and mxai sources")


def _lint_semantic(text: str, path: str) -> LanguageToolReport:
    diagnostics: list[LanguageDiagnostic] = []
    try:
        architect = ArchitectAgent()
        plan = architect.plan_from_text(text)
        plan_result = PlannerVerifier().verify(plan)
        _extend_diagnostics(diagnostics, plan_result.errors, "planner", "error", text)
        _extend_diagnostics(diagnostics, plan_result.warnings, "planner", "warning", text)

        mxai_text = architect.to_mxai(plan)
        program = parse_text(mxai_text)
        verifier_result = VerifierAgent().verify(program)
        _extend_diagnostics(diagnostics, verifier_result.errors, "verifier", "error", mxai_text)
        _extend_diagnostics(diagnostics, verifier_result.warnings, "verifier", "warning", mxai_text)
        _extend_diagnostics(diagnostics, SafetyAgent().review(program), "safety", "warning", mxai_text)

        formatted = format_semantic(text)
    except (ArchitectSpecError, MatrixAIParseError, ValueError) as exc:
        return LanguageToolReport(
            path=path,
            language="semantic",
            formatted=False,
            diagnostics=[LanguageDiagnostic("error", str(exc), "parser")],
        )

    is_formatted = formatted == _ensure_trailing_newline(text)
    if not is_formatted:
        diagnostics.append(
            LanguageDiagnostic("info", "Document differs from canonical formatter output", "format")
        )
    return LanguageToolReport(path=path, language="semantic", diagnostics=diagnostics, formatted=is_formatted)


def _lint_mxai(text: str, path: str) -> LanguageToolReport:
    diagnostics: list[LanguageDiagnostic] = []
    try:
        program = parse_text(text)
        verifier_result = VerifierAgent().verify(program)
        _extend_diagnostics(diagnostics, verifier_result.errors, "verifier", "error", text)
        _extend_diagnostics(diagnostics, verifier_result.warnings, "verifier", "warning", text)
        _extend_diagnostics(diagnostics, SafetyAgent().review(program), "safety", "warning", text)
        formatted = format_mxai(text)
    except (MatrixAIParseError, ValueError) as exc:
        return LanguageToolReport(
            path=path,
            language="mxai",
            formatted=False,
            diagnostics=[LanguageDiagnostic("error", str(exc), "parser")],
        )

    is_formatted = formatted == _ensure_trailing_newline(text)
    if not is_formatted:
        diagnostics.append(
            LanguageDiagnostic("info", "Document differs from canonical formatter output", "format")
        )
    return LanguageToolReport(path=path, language="mxai", diagnostics=diagnostics, formatted=is_formatted)


def _extend_diagnostics(
    diagnostics: list[LanguageDiagnostic],
    messages: list[str],
    source: str,
    severity: str,
    text: str,
) -> None:
    for message in messages:
        diagnostics.append(
            LanguageDiagnostic(
                severity=severity,
                source=source,
                message=message,
                line=_best_effort_line(text, message),
            )
        )


def _best_effort_line(text: str, message: str) -> int | None:
    tokens = [part.strip("'.,:()") for part in message.split() if len(part.strip("'.,:()")) > 2]
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if any(token and token in stripped for token in tokens):
            return line_number
    return None


def _graph_lines(program: MatrixAIProgram) -> list[str]:
    chain_edges = list(zip(program.graph.nodes, program.graph.nodes[1:]))
    if program.graph.edges == chain_edges and program.graph.nodes:
        return [" -> ".join(program.graph.nodes)]
    return [f"{source} -> {target}" for source, target in program.graph.edges]


def _graph_json(program: MatrixAIProgram) -> dict[str, Any]:
    return {
        "project": program.project,
        "nodes": [
            {"id": node, "type": program.graph.node_types.get(node, "unknown")}
            for node in program.graph.nodes
        ],
        "edges": [
            {"source": source, "target": target}
            for source, target in program.graph.edges
        ],
    }


def _graph_mermaid(program: MatrixAIProgram) -> str:
    lines = ["flowchart LR"]
    for node in program.graph.nodes:
        node_id = _graph_node_id(node)
        node_type = program.graph.node_types.get(node, "unknown")
        lines.append(f"  {node_id}[\"{_escape_graph_label(node)}\\n{node_type}\"]")
    for source, target in program.graph.edges:
        lines.append(f"  {_graph_node_id(source)} --> {_graph_node_id(target)}")
    return "\n".join(lines) + "\n"


def _graph_dot(program: MatrixAIProgram) -> str:
    lines = ["digraph MatrixAI {", "  rankdir=LR;"]
    for node in program.graph.nodes:
        node_type = program.graph.node_types.get(node, "unknown")
        label = f"{node}\n{node_type}"
        lines.append(f"  {_quote_dot(node)} [label={_quote_dot(label)}];")
    for source, target in program.graph.edges:
        lines.append(f"  {_quote_dot(source)} -> {_quote_dot(target)};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _graph_node_id(node: str) -> str:
    sanitized = "".join(char if char.isalnum() or char == "_" else "_" for char in node)
    if sanitized and sanitized[0].isdigit():
        return f"n_{sanitized}"
    return sanitized or "node"


def _escape_graph_label(value: str) -> str:
    return value.replace('"', '\\"')


def _quote_dot(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _compare_values(
    runtime_value: Any,
    compiled_value: Any,
    path: str,
    tolerance: float,
    mismatches: list[str],
) -> None:
    if isinstance(runtime_value, (int, float)) and isinstance(compiled_value, (int, float)):
        if abs(float(runtime_value) - float(compiled_value)) > tolerance:
            mismatches.append(f"{path}: runtime={runtime_value!r} compiled={compiled_value!r}")
        return

    if isinstance(runtime_value, dict) and isinstance(compiled_value, dict):
        runtime_keys = set(runtime_value)
        compiled_keys = set(compiled_value)
        for missing in sorted(runtime_keys - compiled_keys):
            mismatches.append(f"{path}.{missing}: missing from compiled result")
        for extra in sorted(compiled_keys - runtime_keys):
            mismatches.append(f"{path}.{extra}: extra in compiled result")
        for key in sorted(runtime_keys & compiled_keys):
            _compare_values(runtime_value[key], compiled_value[key], f"{path}.{key}", tolerance, mismatches)
        return

    if isinstance(runtime_value, list) and isinstance(compiled_value, list):
        if len(runtime_value) != len(compiled_value):
            mismatches.append(f"{path}: runtime length {len(runtime_value)} compiled length {len(compiled_value)}")
            return
        for index, (runtime_item, compiled_item) in enumerate(zip(runtime_value, compiled_value)):
            _compare_values(runtime_item, compiled_item, f"{path}[{index}]", tolerance, mismatches)
        return

    if runtime_value != compiled_value:
        mismatches.append(f"{path}: runtime={runtime_value!r} compiled={compiled_value!r}")


def _type_to_source(spec: TypeSpec) -> str:
    shape = spec.parameters.get("shape")
    if shape is not None and spec.name == "Tensor":
        return f"{spec.name}[{', '.join(str(item) for item in shape)}]"

    dim = spec.parameters.get("dim")
    if dim is not None and spec.name in {"Vector", "Embedding"}:
        return f"{spec.name}[{dim}]"

    args = spec.parameters.get("args")
    if args:
        return f"{spec.name}[{', '.join(str(arg) for arg in args)}]"

    if spec.range is not None and not _is_default_range(spec):
        return f"{spec.name}[{_format_decimal(spec.range.minimum)}, {_format_decimal(spec.range.maximum)}]"
    return spec.name


def _is_default_range(spec: TypeSpec) -> bool:
    try:
        default = parse_type_spec(spec.name)
    except ValueError:
        return False
    return default.range == spec.range


def _format_decimal(value: float | int | None) -> str:
    if value is None:
        return "inf"
    number = float(value)
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    if "." in text and len(text.split(".", 1)[1]) == 1:
        return f"{text}0"
    return text


def _ensure_trailing_newline(text: str) -> str:
    return text.rstrip() + "\n"
