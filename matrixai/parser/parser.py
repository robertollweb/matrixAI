# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import re
from pathlib import Path

from matrixai.ir import (
    ActionInputParam,
    ActionSpec,
    ActionConditionSpec,
    AuditSpec,
    DistributionSpec,
    ExpressionSpec,
    FunctionSpec,
    GraphSpec,
    LayerBodyOp,
    LayerSpec,
    MatrixAIProgram,
    ParameterSpec,
    SequenceSpec,
    VectorSpec,
)
from matrixai.ir.schema import ImportSpec
from matrixai.ir.schema import (
    BlockSpec,
    ConcatSpec,
    CompositeLayerSpec,
    DenseLayerSpec,
    EmbeddingSpec,
    NetworkSpec,
    TransformerBlockSpec,
    _COMPOSITE_ACTIVATIONS,
    _DENSE_ACTIVATIONS,
    _POOL_KINDS,
    _TRANSFORMER_ACTIVATIONS,
    _TRANSFORMER_POS_KINDS,
)
from matrixai.types import parse_type_spec


class MatrixAIParseError(ValueError):
    pass


_IMPORT_RE = re.compile(
    r"^IMPORT\s+(?P<alias>[A-Za-z_]\w*)\s+FROM\s+registry\s+"
    r"(?P<reg_name>[a-z][a-z0-9_]*)@(?P<version>[A-Za-z0-9_][A-Za-z0-9_.]*)"
    r"\s+(?P<mode>FROZEN|TRAINABLE)$"
)
_VALID_IMPORT_MODES = {"FROZEN", "TRAINABLE"}

_VECTOR_RE = re.compile(r"^VECTOR\s+(?P<name>[A-Za-z_][\w]*)\[(?P<size>\d+)\]$")
_SEQUENCE_HDR_RE = re.compile(r"^SEQUENCE\s+(?P<name>[A-Za-z_][\w]*)$")
_PARAM_RE = re.compile(r"^PARAM\s+(?P<name>[A-Za-z_][\w]*)\s+(?P<type>.+)$")
_FIELD_RE = re.compile(r"^(?P<name>[A-Za-z_][\w]*)(?:\s*:\s*(?P<type>.+))?$")
_ASSIGNMENT_RE = re.compile(
    r"^(?P<output>[A-Za-z_][\w]*)(?:\s*:\s*(?P<type>[^=]+?))?\s*=\s*(?P<expression>.+)$"
)
_DISTRIBUTION_RE = re.compile(
    r"^(?P<variable>[A-Za-z_][\w]*)\s*~\s*(?P<type>[A-Za-z_][\w]*)\((?P<source>.*)\)$"
)
_LINEAR_RE = re.compile(
    r"^(?P<activation>softmax|sigmoid)\(\s*(?P<weights>[A-Za-z_][\w]*)\s*\*\s*"
    r"(?P<input>[A-Za-z_][\w]*)\s*\+\s*(?P<bias>[A-Za-z_][\w]*)\s*\)$"
)
_LINEAR_REGRESSION_RE = re.compile(
    r"^linear\(\s*(?P<weights>[A-Za-z_][\w]*)\s*\*\s*"
    r"(?P<input>[A-Za-z_][\w]*)\s*\+\s*(?P<bias>[A-Za-z_][\w]*)\s*\)$"
)
_SIGMOID_THRESHOLD_RE = re.compile(
    r"^sigmoid\(\s*(?P<scale>[0-9.]+)\s*\*\s*\(\s*"
    r"(?P<source>[A-Za-z_][\w.]*)\s*-\s*(?P<threshold>[0-9.]+)\s*\)\s*\)$"
)
_CONDITION_RE = re.compile(
    r"^(?P<source>[A-Za-z_][\w.]*)\s*(?P<operator>>|>=|<|<=)\s*(?P<threshold>[0-9.]+)$"
)
_NETWORK_HDR_RE = re.compile(r"^NETWORK\s+(?P<name>[A-Za-z_][\w]*)$")
_DENSE_LAYER_RE = re.compile(r"^LAYER\s+Dense\s+units=(?P<units>\d+)\s+activation=(?P<activation>\w+)$")
_NETWORK_INPUT_RE = re.compile(r"^INPUT\s+(?P<name>[A-Za-z_][\w]*)$")
_NETWORK_OUTPUT_RE = re.compile(r"^OUTPUT\s+(?P<name>[A-Za-z_][\w]*)\s*:\s*(?P<type>.+)$")
# P19 composite network patterns
_EMBEDDING_RE = re.compile(
    r"^EMBEDDING\s+(?P<name>[A-Za-z_]\w*)\s+FROM\s+(?P<source>[A-Za-z_]\w*)"
    r"\s+VOCAB\s+(?P<vocab>\d+)\s+DIM\s+(?P<dim>\d+)$"
)
# TRANSFORMER_BLOQUE C1: EMBEDDING without VOCAB — only valid FROM a SEQUENCE
# (VOCAB is inherited from the sequence; typecheck validates the source kind).
_EMBEDDING_SEQ_RE = re.compile(
    r"^EMBEDDING\s+(?P<name>[A-Za-z_]\w*)\s+FROM\s+(?P<source>[A-Za-z_]\w*)"
    r"\s+DIM\s+(?P<dim>\d+)$"
)
_TRANSFORMER_HDR_RE = re.compile(r"^BLOCK\s+(?P<name>[A-Za-z_]\w*)\s+TRANSFORMER$")
_CONCAT_RE = re.compile(
    r"^CONCAT\s+\[(?P<sources>[^\]]+)\]\s*->\s*(?P<name>[A-Za-z_]\w*)$"
)
_BLOCK_HDR_RE = re.compile(r"^BLOCK\s+(?P<name>[A-Za-z_]\w*)$")
_RESIDUAL_RE = re.compile(r"^RESIDUAL\s+FROM\s+(?P<source>\w+)$")
_POOL_RE = re.compile(r"^POOL\s+(?P<kind>mean|max|cls)$")
_DROPOUT_RE = re.compile(r"^LAYER\s+Dropout\s+rate=(?P<rate>[0-9]*\.?[0-9]+)$")
_LAYERNORM_RE = re.compile(r"^LAYER\s+LayerNorm$")
_ACTIVATION_RE = re.compile(r"^LAYER\s+Activation\s+kind=(?P<kind>\w+)$")
_RESHAPE_RE = re.compile(r"^LAYER\s+Reshape\s+target=\[(?P<dims>[^\]]+)\]$")
_LAYER_HDR_RE = re.compile(
    r"^LAYER\s+(?P<name>[A-Za-z_][\w]*)"
    r"(?:\s*\(\s*(?P<input_type>[^)]+)\s*\))?"
    r"(?:\s*->\s*(?P<output_type>.+))?$"
)
_CALL_LAYER_RE = re.compile(
    r"^call_layer\(\s*(?P<layer>[A-Za-z_][\w]*)\s*,\s*(?P<input>[A-Za-z_][\w]*)\s*\)$"
)
_BODY_OP_RE = re.compile(
    r"^(?P<output>[A-Za-z_][\w]*)\s*=\s*(?P<kind>[A-Za-z_][\w]*)\s*\((?P<args>[^)]*)\)$"
)


def parse_file(path: str | Path) -> MatrixAIProgram:
    return parse_text(Path(path).read_text(encoding="utf-8"))


def parse_text(text: str) -> MatrixAIProgram:
    lines = _clean_lines(text)
    if not lines:
        raise MatrixAIParseError("Empty MatrixAI document")

    project = ""
    imports: list[ImportSpec] = []
    vectors: list[VectorSpec] = []
    sequences: list[SequenceSpec] = []
    parameters: list[ParameterSpec] = []
    functions: list[FunctionSpec] = []
    layers: list[LayerSpec] = []
    distributions: list[DistributionSpec] = []
    networks: list[NetworkSpec] = []
    graph = GraphSpec()
    actions: list[ActionSpec] = []
    audit = AuditSpec()

    index = 0
    while index < len(lines):
        line = lines[index]
        keyword = line.split(maxsplit=1)[0]

        if keyword == "PROJECT":
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise MatrixAIParseError("PROJECT requires a name")
            project = parts[1].strip()
            index += 1
            continue

        if keyword == "IMPORT":
            # Single-line statement — no END block.
            imports.append(_parse_import(line))
            index += 1
            continue

        if keyword == "VECTOR":
            block, index = _read_block(lines, index)
            vectors.append(_parse_vector(block))
            continue

        if keyword == "SEQUENCE":
            block, index = _read_block(lines, index)
            sequences.append(_parse_sequence(block))
            continue

        if keyword == "PARAM":
            block, index = _read_block(lines, index)
            parameters.append(_parse_parameter(block))
            continue

        if keyword == "FUNCTION":
            block, index = _read_block(lines, index)
            functions.append(_parse_function(block))
            continue

        if keyword == "NETWORK":
            block, index = _read_block(lines, index)
            networks.append(_parse_network(block))
            continue

        if keyword == "LAYER":
            block, index = _read_block(lines, index)
            layers.append(_parse_layer(block))
            continue

        if keyword == "DISTRIBUTION":
            block, index = _read_block(lines, index)
            distributions.append(_parse_distribution(block))
            continue

        if keyword == "GRAPH":
            block, index = _read_block(lines, index)
            graph = _parse_graph(block)
            continue

        if keyword == "ACTION":
            block, index = _read_block(lines, index)
            actions.append(_parse_action(block))
            continue

        if keyword == "AUDIT":
            block, index = _read_block(lines, index)
            audit = _parse_audit(block)
            continue

        raise MatrixAIParseError(f"Unknown block: {line}")

    if not project:
        raise MatrixAIParseError("Missing PROJECT block")

    # Validate IMPORT aliases: uniqueness and no collision with other symbols.
    if imports:
        seen_aliases: set[str] = set()
        for imp in imports:
            if imp.alias in seen_aliases:
                raise MatrixAIParseError(
                    f"IMPORT alias {imp.alias!r} is declared more than once"
                )
            seen_aliases.add(imp.alias)
        defined_names = (
            {v.name for v in vectors}
            | {n.name for n in networks}
            | {a.name for a in actions}
            | {f.name for f in functions}
            | {d.name for d in distributions}
            | {s.name for s in sequences}
        )
        for imp in imports:
            if imp.alias in defined_names:
                raise MatrixAIParseError(
                    f"IMPORT alias {imp.alias!r} collides with an existing symbol"
                )

    graph = _attach_node_types(graph, vectors, sequences, functions, distributions, actions, networks, imports)

    return MatrixAIProgram(
        project=project,
        imports=imports,
        vectors=vectors,
        sequences=sequences,
        parameters=parameters,
        functions=functions,
        layers=layers,
        distributions=distributions,
        networks=networks,
        graph=graph,
        actions=actions,
        audit=audit,
    )


def _clean_lines(text: str) -> list[str]:
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cleaned.append(line)
    return cleaned


def _read_block(lines: list[str], start: int) -> tuple[list[str], int]:
    block = [lines[start]]
    depth = 1
    index = start + 1
    while index < len(lines):
        line = lines[index]
        block.append(line)
        kw = line.split(maxsplit=1)[0] if line.strip() else ""
        if kw == "BLOCK":
            depth += 1
        elif line.strip() == "END":
            depth -= 1
            if depth == 0:
                return block, index + 1
        index += 1
    raise MatrixAIParseError(f"Block '{lines[start]}' is missing END")


def _parse_vector(block: list[str]) -> VectorSpec:
    match = _VECTOR_RE.match(block[0])
    if not match:
        raise MatrixAIParseError(f"Invalid VECTOR declaration: {block[0]}")
    fields: list[str] = []
    field_types = {}
    for line in block[1:-1]:
        if not line:
            continue
        field_match = _FIELD_RE.match(line)
        if not field_match:
            raise MatrixAIParseError(f"Invalid VECTOR field declaration: {line}")
        field_name = field_match.group("name")
        fields.append(field_name)
        type_text = field_match.group("type")
        if type_text:
            try:
                field_types[field_name] = parse_type_spec(type_text)
            except ValueError as exc:
                raise MatrixAIParseError(f"Invalid type for VECTOR field {field_name}: {exc}") from exc
    size = int(match.group("size"))
    if len(fields) != size:
        raise MatrixAIParseError(
            f"VECTOR {match.group('name')} declares size {size} but has {len(fields)} fields"
        )
    return VectorSpec(name=match.group("name"), size=size, fields=fields, field_types=field_types)


def _parse_sequence(block: list[str]) -> SequenceSpec:
    match = _SEQUENCE_HDR_RE.match(block[0])
    if not match:
        raise MatrixAIParseError(f"Invalid SEQUENCE declaration: {block[0]}")
    name = match.group("name")
    length = 0
    vocab_size = 0
    for line in block[1:-1]:
        if not line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "length":
            length = int(value)
        elif key == "vocab_size":
            vocab_size = int(value)
        else:
            raise MatrixAIParseError(f"Unknown SEQUENCE attribute: {key}")
    if length <= 0:
        raise MatrixAIParseError(f"SEQUENCE {name} must declare length > 0")
    if vocab_size <= 0:
        raise MatrixAIParseError(f"SEQUENCE {name} must declare vocab_size > 0")
    return SequenceSpec(name=name, length=length, vocab_size=vocab_size)


def _parse_parameter(block: list[str]) -> ParameterSpec:
    match = _PARAM_RE.match(block[0])
    if not match:
        raise MatrixAIParseError(f"Invalid PARAM declaration: {block[0]}")
    try:
        type_spec = parse_type_spec(match.group("type"))
    except ValueError as exc:
        raise MatrixAIParseError(f"Invalid type for PARAM {match.group('name')}: {exc}") from exc

    trainable = True
    initializer = ""
    for line in block[1:-1]:
        if line.startswith("TRAINABLE "):
            value = line.removeprefix("TRAINABLE ").strip().lower()
            if value not in {"true", "false"}:
                raise MatrixAIParseError(f"PARAM {match.group('name')} TRAINABLE must be true or false")
            trainable = value == "true"
        elif line.startswith("INIT "):
            initializer = line.removeprefix("INIT ").strip()
            if not initializer:
                raise MatrixAIParseError(f"PARAM {match.group('name')} INIT requires a value")
        elif line:
            raise MatrixAIParseError(f"Unsupported PARAM line: {line}")

    return ParameterSpec(
        name=match.group("name"),
        type_spec=type_spec,
        trainable=trainable,
        initializer=initializer,
    )


def _parse_function(block: list[str]) -> FunctionSpec:
    parts = block[0].split(maxsplit=1)
    if len(parts) != 2:
        raise MatrixAIParseError("FUNCTION requires a name")
    assignments = [_ASSIGNMENT_RE.match(line) for line in block[1:-1]]
    assignments = [match for match in assignments if match]
    if len(assignments) != 1:
        raise MatrixAIParseError(f"FUNCTION {parts[1]} requires exactly one assignment")
    match = assignments[0]
    output_type = None
    if match.group("type"):
        try:
            output_type = parse_type_spec(match.group("type"))
        except ValueError as exc:
            raise MatrixAIParseError(f"Invalid type for FUNCTION {parts[1]} output: {exc}") from exc
    return FunctionSpec(
        name=parts[1],
        output=match.group("output"),
        expression=match.group("expression"),
        semantic=_parse_expression(match.group("expression")),
        output_type=output_type,
    )


def _parse_distribution(block: list[str]) -> DistributionSpec:
    parts = block[0].split(maxsplit=1)
    if len(parts) != 2:
        raise MatrixAIParseError("DISTRIBUTION requires a name")
    body = " ".join(block[1:-1])
    match = _DISTRIBUTION_RE.match(body)
    if not match:
        raise MatrixAIParseError(f"Invalid DISTRIBUTION body: {body}")
    return DistributionSpec(
        name=parts[1],
        variable=match.group("variable"),
        distribution_type=match.group("type"),
        source=match.group("source").strip(),
        raw=body,
    )


def _parse_graph(block: list[str]) -> GraphSpec:
    edges: list[tuple[str, str]] = []
    nodes: list[str] = []
    for line in block[1:-1]:
        chain = [part.strip() for part in line.split("->") if part.strip()]
        if len(chain) < 2:
            raise MatrixAIParseError(f"GRAPH line requires at least two nodes: {line}")
        for node in chain:
            if node not in nodes:
                nodes.append(node)
        edges.extend((left, right) for left, right in zip(chain, chain[1:]))
    return GraphSpec(nodes=nodes, edges=edges)


_VALID_POLICIES = {"simulate_only", "real_with_audit", "execute"}
_INPUT_PARAM_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>\S+)$")


def _parse_input_params(raw: str) -> tuple[ActionInputParam, ...]:
    params = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        m = _INPUT_PARAM_RE.match(part)
        if not m:
            raise MatrixAIParseError(f"Invalid INPUT param: {part!r}")
        params.append(ActionInputParam(name=m.group("name"), type=m.group("type")))
    return tuple(params)


def _parse_action(block: list[str]) -> ActionSpec:
    parts = block[0].split(maxsplit=1)
    if len(parts) != 2:
        raise MatrixAIParseError("ACTION requires a name")
    name = parts[1]
    when = ""
    call = ""
    policy = "simulate_only"
    target = ""
    input_params: tuple[ActionInputParam, ...] = ()
    for line in block[1:-1]:
        if line.startswith("WHEN "):
            when = line.removeprefix("WHEN ").strip()
        elif line.startswith("CALL "):
            call = line.removeprefix("CALL ").strip()
        elif line.startswith("POLICY "):
            policy = line.removeprefix("POLICY ").strip()
            if policy not in _VALID_POLICIES:
                raise MatrixAIParseError(
                    f"ACTION {name} unknown POLICY {policy!r}; valid: {sorted(_VALID_POLICIES)}"
                )
        elif line.startswith("TARGET "):
            target = line.removeprefix("TARGET ").strip()
        elif line.startswith("CONDITION "):
            when = line.removeprefix("CONDITION ").strip()
        elif line.startswith("INPUT "):
            input_params = _parse_input_params(line.removeprefix("INPUT ").strip())
    # validation: need (WHEN+CALL) or (TARGET+CONDITION)
    if target:
        if not when:
            raise MatrixAIParseError(f"ACTION {name} with TARGET requires CONDITION")
    else:
        if not when or not call:
            raise MatrixAIParseError(f"ACTION {name} requires WHEN and CALL")
    if policy == "real_with_audit" and not target:
        raise MatrixAIParseError(f"ACTION {name} with POLICY real_with_audit requires TARGET")
    return ActionSpec(
        name=name,
        when=when,
        call=call,
        condition=_parse_condition(when),
        policy=policy,
        target=target,
        input_params=input_params,
    )


def _parse_import(line: str) -> ImportSpec:
    """Parse a single-line IMPORT statement."""
    m = _IMPORT_RE.match(line)
    if not m:
        # Produce a detailed error for common mistakes.
        parts = line.split()
        if len(parts) < 6:
            raise MatrixAIParseError(
                f"IMPORT syntax: IMPORT <alias> FROM registry <name>@<version> FROZEN|TRAINABLE — "
                f"got: {line!r}"
            )
        if len(parts) >= 6 and parts[5] not in _VALID_IMPORT_MODES:
            raise MatrixAIParseError(
                f"IMPORT mode must be FROZEN or TRAINABLE, got {parts[5]!r}"
            )
        if "@" not in (parts[4] if len(parts) > 4 else ""):
            raise MatrixAIParseError(
                f"IMPORT requires name@version (e.g. text_encoder@v1), got {line!r}"
            )
        raise MatrixAIParseError(f"Invalid IMPORT syntax: {line!r}")
    return ImportSpec(
        alias=m.group("alias"),
        registry_name=m.group("reg_name"),
        version=m.group("version"),
        mode=m.group("mode"),
    )


def _parse_audit(block: list[str]) -> AuditSpec:
    explain: list[str] = []
    for line in block[1:-1]:
        if line.startswith("EXPLAIN "):
            explain = [part.strip() for part in line.removeprefix("EXPLAIN ").split("->")]
    return AuditSpec(explain=[node for node in explain if node])


def _parse_inline_param(line: str) -> ParameterSpec:
    """Parse a single-line PARAM inside a LAYER block: PARAM name type [TRAINABLE bool] [INIT val]."""
    m = re.match(r"^PARAM\s+([A-Za-z_][\w]*)\s+(.+)$", line)
    if not m:
        raise MatrixAIParseError(f"Invalid inline PARAM: {line}")
    name = m.group(1)
    rest = m.group(2).strip()

    trainable = True
    initializer = ""

    trainable_m = re.search(r"\s+TRAINABLE\s+(true|false)(?:\s+|$)", rest)
    if trainable_m:
        trainable = trainable_m.group(1) == "true"
        rest = (rest[: trainable_m.start()] + rest[trainable_m.end() :]).strip()

    init_m = re.search(r"\s+INIT\s+(\S+)(?:\s+|$)", rest)
    if init_m:
        initializer = init_m.group(1)
        rest = (rest[: init_m.start()] + rest[init_m.end() :]).strip()

    try:
        type_spec = parse_type_spec(rest)
    except ValueError as exc:
        raise MatrixAIParseError(f"Invalid type for inline PARAM {name}: {exc}") from exc
    return ParameterSpec(name=name, type_spec=type_spec, trainable=trainable, initializer=initializer)


def _parse_layer(block: list[str]) -> LayerSpec:
    match = _LAYER_HDR_RE.match(block[0])
    if not match:
        raise MatrixAIParseError(f"Invalid LAYER declaration: {block[0]}")

    name = match.group("name")
    input_type = None
    output_type = None

    if match.group("input_type"):
        try:
            input_type = parse_type_spec(match.group("input_type").strip())
        except ValueError as exc:
            raise MatrixAIParseError(f"Invalid input type for LAYER {name}: {exc}") from exc

    if match.group("output_type"):
        try:
            output_type = parse_type_spec(match.group("output_type").strip())
        except ValueError as exc:
            raise MatrixAIParseError(f"Invalid output type for LAYER {name}: {exc}") from exc

    parameters: list[ParameterSpec] = []
    body_lines: list[str] = []
    body_ops: list[LayerBodyOp] = []

    for line in block[1:-1]:
        if line.startswith("PARAM "):
            parameters.append(_parse_inline_param(line))
        else:
            body_lines.append(line)
            m = _BODY_OP_RE.match(line.strip())
            if m:
                raw_args = m.group("args")
                args = tuple(a.strip() for a in raw_args.split(",") if a.strip())
                body_ops.append(LayerBodyOp(output=m.group("output"), kind=m.group("kind"), args=args))

    return LayerSpec(
        name=name,
        params=parameters,
        input_type=input_type,
        output_type=output_type,
        body=" ; ".join(body_lines),
        body_ops=tuple(body_ops),
    )


def _parse_expression(expression: str) -> ExpressionSpec:
    call_layer_m = _CALL_LAYER_RE.match(expression)
    if call_layer_m:
        return ExpressionSpec(
            raw=expression,
            kind="layer_call",
            inputs=[call_layer_m.group("input")],
            parameters={
                "layer": call_layer_m.group("layer"),
                "input": call_layer_m.group("input"),
            },
        )

    linear_regression = _LINEAR_REGRESSION_RE.match(expression)
    if linear_regression:
        return ExpressionSpec(
            raw=expression,
            kind="linear_regression",
            inputs=[linear_regression.group("input")],
            parameters={
                "weights": linear_regression.group("weights"),
                "bias": linear_regression.group("bias"),
            },
        )

    linear = _LINEAR_RE.match(expression)
    if linear:
        return ExpressionSpec(
            raw=expression,
            kind=f"{linear.group('activation')}_linear",
            inputs=[linear.group("input")],
            parameters={"weights": linear.group("weights"), "bias": linear.group("bias")},
        )

    sigmoid_threshold = _SIGMOID_THRESHOLD_RE.match(expression)
    if sigmoid_threshold:
        return ExpressionSpec(
            raw=expression,
            kind="sigmoid_threshold",
            inputs=[sigmoid_threshold.group("source")],
            parameters={
                "scale": float(sigmoid_threshold.group("scale")),
                "threshold": float(sigmoid_threshold.group("threshold")),
            },
        )

    symbolic = _parse_symbolic_expression(expression)
    if symbolic is not None:
        return symbolic

    return ExpressionSpec(raw=expression, kind="unknown")


def _parse_symbolic_expression(expression: str) -> ExpressionSpec | None:
    from matrixai.ir.expr import CallNode, VarNode
    from matrixai.ir.expr import collect_vars, extract_weighted_sum, parse_expr

    try:
        node = parse_expr(expression)
    except ValueError:
        return None

    if isinstance(node, CallNode):
        tensor_prim = _tensor_primitive_expression_spec(expression, node)
        if tensor_prim is not None:
            return tensor_prim

        aggregate = _aggregate_expression_spec(expression, node)
        if aggregate is not None:
            return aggregate

        normalize = _normalize_expression_spec(expression, node)
        if normalize is not None:
            return normalize

        if node.func == "argmax" and len(node.args) == 1 and isinstance(node.args[0], VarNode):
            score_input = node.args[0].name
            return ExpressionSpec(
                raw=expression,
                kind="select_argmax",
                inputs=[score_input],
                parameters={"score_input": score_input},
            )

    weighted = extract_weighted_sum(node)
    if weighted is not None:
        inputs = list(dict.fromkeys(collect_vars(weighted)))
        terms = [{"weight": weight, "expr": str(term)} for weight, term in weighted.terms]
        return ExpressionSpec(
            raw=expression,
            kind="symbolic_weighted_sum",
            inputs=inputs,
            parameters={"terms": terms, "ast": weighted.to_dict()},
        )

    return ExpressionSpec(
        raw=expression,
        kind="symbolic_expr",
        inputs=list(dict.fromkeys(collect_vars(node))),
        parameters={"ast": node.to_dict()},
    )


_TENSOR_PRIMITIVES = frozenset({
    "dot", "matmul", "relu", "gelu", "layer_norm", "residual",
    "embedding_lookup", "positional_encoding", "attention", "mean_pooling", "cls_pooling",
    "scale", "softmax"
})


def _tensor_primitive_expression_spec(expression: str, node) -> ExpressionSpec | None:
    from matrixai.ir.expr import VarNode
    if node.func not in _TENSOR_PRIMITIVES:
        return None
    if node.func == "scale" and len(node.args) == 3:
        return None
    inputs = [arg.name for arg in node.args if isinstance(arg, VarNode)]
    return ExpressionSpec(
        raw=expression,
        kind=node.func,
        inputs=inputs,
        parameters={"ast": node.to_dict()},
    )


def _aggregate_expression_spec(expression: str, node) -> ExpressionSpec | None:
    from matrixai.ir.expr import VarNode

    if node.func not in {"max", "min", "mean", "softmax", "vote"}:
        return None
    if not node.args or not all(isinstance(arg, VarNode) for arg in node.args):
        return None
    inputs = [arg.name for arg in node.args]
    return ExpressionSpec(
        raw=expression,
        kind=f"aggregate_{node.func}",
        inputs=inputs,
        parameters={"inputs": inputs, "method": node.func, "ast": node.to_dict()},
    )


def _normalize_expression_spec(expression: str, node) -> ExpressionSpec | None:
    from matrixai.ir.expr import LiteralNode, VarNode

    if node.func == "normalize" and len(node.args) == 1 and isinstance(node.args[0], VarNode):
        return ExpressionSpec(
            raw=expression,
            kind="normalize",
            inputs=[node.args[0].name],
            parameters={"var": node.args[0].name, "lo": 0.0, "hi": 1.0, "ast": node.to_dict()},
        )

    if (
        node.func == "scale"
        and len(node.args) == 3
        and isinstance(node.args[0], VarNode)
        and isinstance(node.args[1], LiteralNode)
        and isinstance(node.args[2], LiteralNode)
    ):
        return ExpressionSpec(
            raw=expression,
            kind="normalize",
            inputs=[node.args[0].name],
            parameters={
                "var": node.args[0].name,
                "lo": node.args[1].value,
                "hi": node.args[2].value,
                "ast": node.to_dict(),
            },
        )

    return None


def _collect_block_body(body: list[str], start: int, net_name: str) -> tuple[list[str], int]:
    """Collect lines between start and matching END for a BLOCK sub-block."""
    lines_out: list[str] = []
    depth = 1
    i = start
    while i < len(body):
        line = body[i]
        kw = line.split(maxsplit=1)[0] if line.strip() else ""
        if kw == "BLOCK":
            depth += 1
        elif line.strip() == "END":
            depth -= 1
            if depth == 0:
                return lines_out, i + 1
        if depth > 0:
            lines_out.append(line)
        i += 1
    raise MatrixAIParseError(f"NETWORK {net_name}: BLOCK is missing END")


def _parse_composite_layer(line: str, index: int, net_name: str, block_name: str = "") -> CompositeLayerSpec:
    """Parse a single layer line inside a NETWORK or BLOCK into a CompositeLayerSpec."""
    ctx = f"NETWORK {net_name}" + (f" BLOCK {block_name}" if block_name else "")

    m = _DENSE_LAYER_RE.match(line)
    if m:
        units = int(m.group("units"))
        if units <= 0:
            raise MatrixAIParseError(f"{ctx}: LAYER Dense units must be > 0, got {units}")
        activation = m.group("activation")
        if activation not in _DENSE_ACTIVATIONS:
            raise MatrixAIParseError(
                f"{ctx}: LAYER Dense unknown activation '{activation}'. "
                f"Allowed: {', '.join(sorted(_DENSE_ACTIVATIONS))}"
            )
        return CompositeLayerSpec(index=index, layer_type="Dense", units=units, activation=activation)

    if _LAYERNORM_RE.match(line):
        return CompositeLayerSpec(index=index, layer_type="LayerNorm")

    m = _DROPOUT_RE.match(line)
    if m:
        rate = float(m.group("rate"))
        if rate <= 0.0 or rate >= 1.0:
            raise MatrixAIParseError(f"{ctx}: LAYER Dropout rate must be in (0, 1), got {rate}")
        return CompositeLayerSpec(index=index, layer_type="Dropout", rate=rate)

    m = _ACTIVATION_RE.match(line)
    if m:
        kind = m.group("kind")
        if kind not in _COMPOSITE_ACTIVATIONS:
            raise MatrixAIParseError(
                f"{ctx}: LAYER Activation kind '{kind}' not allowed. "
                f"Allowed: {', '.join(sorted(_COMPOSITE_ACTIVATIONS))}"
            )
        return CompositeLayerSpec(index=index, layer_type="Activation", activation_kind=kind)

    m = _RESHAPE_RE.match(line)
    if m:
        try:
            dims = [int(d.strip()) for d in m.group("dims").split(",") if d.strip()]
        except ValueError:
            dims = []
        if not dims or any(d <= 0 for d in dims):
            raise MatrixAIParseError(f"{ctx}: LAYER Reshape target dims must be positive integers")
        return CompositeLayerSpec(index=index, layer_type="Reshape", target_shape=dims)

    m = _POOL_RE.match(line)
    if m:
        return CompositeLayerSpec(index=index, layer_type="Pool", pool_kind=m.group("kind"))

    layer_prefix = re.match(r"^LAYER\s+(\w+)", line)
    if layer_prefix:
        layer_type = layer_prefix.group(1)
        if layer_type == "Dense":
            raise MatrixAIParseError(f"{ctx}: invalid LAYER Dense syntax: {line}")
        raise MatrixAIParseError(
            f"{ctx}: unsupported layer type '{layer_type}'. "
            "Allowed: Dense, LayerNorm, Dropout, Activation, Pool, Reshape."
        )
    return None  # type: ignore  # caller handles None


def _parse_block(block_name: str, lines: list[str], net_name: str) -> BlockSpec:
    """Parse the body lines of a BLOCK...END sub-block."""
    comp_layers: list[CompositeLayerSpec] = []
    residual_from: str = ""

    for line in lines:
        if _BLOCK_HDR_RE.match(line):
            raise MatrixAIParseError(
                f"NETWORK {net_name} BLOCK {block_name}: nested blocks are not allowed"
            )

        m = _RESIDUAL_RE.match(line)
        if m:
            residual_from = m.group("source")
            continue

        layer = _parse_composite_layer(line, len(comp_layers) + 1, net_name, block_name)
        if layer is not None:
            comp_layers.append(layer)
            continue

        raise MatrixAIParseError(
            f"NETWORK {net_name} BLOCK {block_name}: unexpected line: {line}"
        )

    if not comp_layers:
        raise MatrixAIParseError(f"NETWORK {net_name}: BLOCK {block_name} is empty")

    return BlockSpec(name=block_name, layers=comp_layers, residual_from=residual_from)


_TRANSFORMER_KEYS = ("LAYERS", "HEADS", "FF", "DROPOUT", "ACTIVATION", "POS")


def _parse_transformer_block(
    block_name: str, lines: list[str], net_name: str
) -> TransformerBlockSpec:
    """Parse the key-value body of a BLOCK <name> TRANSFORMER ... END.

    dim/length/vocab are NOT declared here (inherited from EMBEDDING/SEQUENCE);
    HEADS-divides-dim and FF=4*dim resolution happen at typecheck, where the
    inherited dim is known.
    """
    ctx = f"NETWORK {net_name} BLOCK {block_name} TRANSFORMER"
    seen: dict[str, str] = {}

    for line in lines:
        parts = line.split(maxsplit=1)
        key = parts[0]
        if key not in _TRANSFORMER_KEYS:
            raise MatrixAIParseError(
                f"{ctx}: unknown key '{key}' — valid keys: {', '.join(_TRANSFORMER_KEYS)}"
            )
        if len(parts) != 2:
            raise MatrixAIParseError(f"{ctx}: {key} requires a value")
        if key in seen:
            raise MatrixAIParseError(f"{ctx}: duplicate key '{key}'")
        seen[key] = parts[1].strip()

    if "LAYERS" not in seen:
        raise MatrixAIParseError(f"{ctx}: LAYERS is required (number of encoder layers, >= 1)")

    def _int_value(key: str, default: int) -> int:
        raw = seen.get(key)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise MatrixAIParseError(f"{ctx}: {key} must be an integer, got '{raw}'") from None
        if value < 1:
            raise MatrixAIParseError(f"{ctx}: {key} must be >= 1, got {value}")
        return value

    layers = _int_value("LAYERS", 0)
    heads = _int_value("HEADS", 4)
    ff = _int_value("FF", 0)  # 0 = default 4*dim, resolved at typecheck

    dropout = 0.0
    if "DROPOUT" in seen:
        try:
            dropout = float(seen["DROPOUT"])
        except ValueError:
            raise MatrixAIParseError(
                f"{ctx}: DROPOUT must be a number in [0, 1), got '{seen['DROPOUT']}'"
            ) from None
        if not (0.0 <= dropout < 1.0):
            raise MatrixAIParseError(f"{ctx}: DROPOUT must be in [0, 1), got {dropout}")

    activation = seen.get("ACTIVATION", "gelu").lower()
    if activation not in _TRANSFORMER_ACTIVATIONS:
        raise MatrixAIParseError(
            f"{ctx}: ACTIVATION must be one of {sorted(_TRANSFORMER_ACTIVATIONS)}, "
            f"got '{seen['ACTIVATION']}'"
        )

    pos = seen.get("POS", "sinusoidal").lower()
    if pos not in _TRANSFORMER_POS_KINDS:
        raise MatrixAIParseError(
            f"{ctx}: POS must be one of {sorted(_TRANSFORMER_POS_KINDS)}, got '{seen['POS']}'"
        )

    return TransformerBlockSpec(
        name=block_name,
        layers=layers,
        heads=heads,
        ff=ff,
        dropout=dropout,
        activation=activation,
        pos=pos,
    )


def _parse_network(block: list[str]) -> NetworkSpec:
    match = _NETWORK_HDR_RE.match(block[0])
    if not match:
        raise MatrixAIParseError(f"Invalid NETWORK declaration: {block[0]}")
    name = match.group("name")

    input_name: str | None = None
    output_name: str | None = None
    output_type_str: str | None = None
    embeddings: list[EmbeddingSpec] = []
    concats: list[ConcatSpec] = []
    blocks: list[BlockSpec] = []
    transformer_blocks: list[TransformerBlockSpec] = []
    all_top_layers: list[CompositeLayerSpec] = []  # all top-level layers (Dense or P19)
    is_composite = False
    input_count = 0
    # Audit finding MEDIA (2026-07-10): `position` (count of preceding top_layers)
    # cannot order an EMBEDDING against a BLOCK TRANSFORMER when no top_layer sits
    # between them (both would read 0). This global counter, incremented once per
    # EMBEDDING/transformer block in textual order, gives them a comparable value.
    seq_order = 0

    body = block[1:-1]
    i = 0
    while i < len(body):
        line = body[i]

        m = _NETWORK_INPUT_RE.match(line)
        if m:
            input_count += 1
            if input_count > 1:
                raise MatrixAIParseError(f"NETWORK {name} must have exactly one INPUT")
            input_name = m.group("name")
            i += 1
            continue

        m = _NETWORK_OUTPUT_RE.match(line)
        if m:
            output_name = m.group("name")
            output_type_str = m.group("type").strip()
            i += 1
            continue

        m = _EMBEDDING_RE.match(line)
        if m:
            vocab = int(m.group("vocab"))
            dim = int(m.group("dim"))
            emb_name = m.group("name")
            if vocab <= 0:
                raise MatrixAIParseError(
                    f"NETWORK {name} EMBEDDING {emb_name}: VOCAB must be > 0, got {vocab}"
                )
            if dim <= 0:
                raise MatrixAIParseError(
                    f"NETWORK {name} EMBEDDING {emb_name}: DIM must be > 0, got {dim}"
                )
            seq_order += 1
            embeddings.append(EmbeddingSpec(
                name=emb_name, source=m.group("source"), vocab=vocab, dim=dim, parse_order=seq_order
            ))
            is_composite = True
            i += 1
            continue

        # TRANSFORMER_BLOQUE C1: EMBEDDING without VOCAB (inherit from SEQUENCE).
        # vocab=0 is the "inherit" sentinel; typecheck resolves it against the
        # source SEQUENCE and rejects tabular sources (those still require VOCAB).
        m = _EMBEDDING_SEQ_RE.match(line)
        if m:
            dim = int(m.group("dim"))
            emb_name = m.group("name")
            if dim <= 0:
                raise MatrixAIParseError(
                    f"NETWORK {name} EMBEDDING {emb_name}: DIM must be > 0, got {dim}"
                )
            seq_order += 1
            embeddings.append(EmbeddingSpec(
                name=emb_name, source=m.group("source"), vocab=0, dim=dim, parse_order=seq_order
            ))
            is_composite = True
            i += 1
            continue

        m = _TRANSFORMER_HDR_RE.match(line)
        if m:
            from dataclasses import replace as _dc_replace
            tb_name = m.group("name")
            tb_lines, i = _collect_block_body(body, i + 1, name)
            parsed_tb = _parse_transformer_block(tb_name, tb_lines, name)
            seq_order += 1
            transformer_blocks.append(_dc_replace(
                parsed_tb, position=len(all_top_layers), parse_order=seq_order
            ))
            is_composite = True
            continue

        m = _CONCAT_RE.match(line)
        if m:
            sources = [s.strip() for s in m.group("sources").split(",") if s.strip()]
            if not sources:
                raise MatrixAIParseError(f"NETWORK {name} CONCAT requires at least one source")
            concats.append(ConcatSpec(name=m.group("name"), sources=sources))
            is_composite = True
            i += 1
            continue

        m = _BLOCK_HDR_RE.match(line)
        if m:
            from dataclasses import replace as _dc_replace
            block_name = m.group("name")
            block_lines, i = _collect_block_body(body, i + 1, name)
            parsed_block = _parse_block(block_name, block_lines, name)
            blocks.append(_dc_replace(parsed_block, position=len(all_top_layers)))
            is_composite = True
            continue

        # POOL at top level (not inside block)
        m = _POOL_RE.match(line)
        if m:
            idx = len(all_top_layers) + 1
            all_top_layers.append(CompositeLayerSpec(index=idx, layer_type="Pool", pool_kind=m.group("kind")))
            is_composite = True
            i += 1
            continue

        layer = _parse_composite_layer(line, len(all_top_layers) + 1, name)
        if layer is not None:
            if layer.layer_type != "Dense":
                is_composite = True
            all_top_layers.append(layer)
            i += 1
            continue

        raise MatrixAIParseError(f"NETWORK {name} unexpected line: {line}")

    if input_name is None:
        raise MatrixAIParseError(f"NETWORK {name} is missing INPUT")
    has_content = all_top_layers or blocks or embeddings or transformer_blocks
    if not has_content:
        raise MatrixAIParseError(f"NETWORK {name} must have at least one LAYER or block")
    if output_name is None or output_type_str is None:
        raise MatrixAIParseError(f"NETWORK {name} is missing OUTPUT")

    if is_composite:
        return NetworkSpec(
            name=name,
            input=input_name,
            layers=[],
            output=output_name,
            output_type_str=output_type_str,
            kind="composite_network",
            embeddings=embeddings,
            concats=concats,
            blocks=blocks,
            top_layers=all_top_layers,
            transformer_blocks=transformer_blocks,
        )
    # P18 dense_network: convert CompositeLayerSpec back to DenseLayerSpec
    dense_layers = [
        DenseLayerSpec(index=cl.index, units=cl.units, activation=cl.activation)
        for cl in all_top_layers
    ]
    return NetworkSpec(
        name=name,
        input=input_name,
        layers=dense_layers,
        output=output_name,
        output_type_str=output_type_str,
        kind="dense_network",
    )


def _parse_condition(condition: str) -> ActionConditionSpec:
    match = _CONDITION_RE.match(condition)
    if not match:
        raise MatrixAIParseError(f"Unsupported action condition: {condition}")
    return ActionConditionSpec(
        raw=condition,
        source=match.group("source"),
        operator=match.group("operator"),
        threshold=float(match.group("threshold")),
    )


def _attach_node_types(
    graph: GraphSpec,
    vectors: list[VectorSpec],
    sequences: list[SequenceSpec],
    functions: list[FunctionSpec],
    distributions: list[DistributionSpec],
    actions: list[ActionSpec],
    networks: list[NetworkSpec] | None = None,
    imports: list[ImportSpec] | None = None,
) -> GraphSpec:
    node_types: dict[str, str] = {}
    node_types.update({vector.name: "vector" for vector in vectors})
    node_types.update({sequence.name: "sequence" for sequence in sequences})
    node_types.update({function.name: "function" for function in functions})
    node_types.update({distribution.name: "distribution" for distribution in distributions})
    node_types.update({action.name: "action" for action in actions})
    if networks:
        node_types.update({network.name: network.kind for network in networks})
    if imports:
        node_types.update({imp.alias: "composite_model" for imp in imports})
    return GraphSpec(
        nodes=graph.nodes,
        edges=graph.edges,
        node_types={node: node_types.get(node, "unknown") for node in graph.nodes},
    )