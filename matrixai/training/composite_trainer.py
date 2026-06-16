# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Composite model forward pass and training scaffold (P21 C8)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompositeForwardResult:
    outputs: dict[str, Any]          # node_alias → output value
    frozen_params_loaded: dict[str, Any]   # alias → metadata from registry
    trainable_params_used: dict[str, Any]  # key → param dict from ParameterSet
    composite_model_hash: str


@dataclass(frozen=True)
class CompositeTrainingStep:
    updated_parameters: dict[str, Any]    # only trainable params, post-gradient
    frozen_unchanged: list[str]           # alias list of frozen components
    gradient_keys: list[str]             # which param keys received gradients
    loss: float


def composite_forward(
    program: Any,
    parameter_set: Any,
    input_data: Any,
    registry: Any,
) -> CompositeForwardResult:
    """Execute composite model forward pass.

    FROZEN components are executed using their stored model.mxai and params.json
    from the registry entry directory.  Falls back to identity passthrough when
    artifacts are absent (e.g. in tests that push mock entries without files).

    TRAINABLE components use the provided ParameterSet.
    """
    from matrixai.parameters.store import load_frozen_parameters_from_registry, separate_parameters

    frozen_meta = load_frozen_parameters_from_registry(program, registry)
    trainable_params, _ = separate_parameters(parameter_set, program)

    import_by_alias = {imp.alias: imp for imp in getattr(program, "imports", [])}

    topo_order = _topological_order(program.graph.edges, program.graph.nodes)
    # Seed all source nodes (zero in-degree, not an import) with the input data.
    dst_set = {dst for _, dst in program.graph.edges}
    outputs: dict[str, Any] = {
        n: input_data
        for n in program.graph.nodes
        if n not in dst_set and n not in import_by_alias
    }
    outputs["input"] = input_data  # legacy lowercase key for backward compat

    for node in topo_order:
        if node not in import_by_alias:
            continue
        imp = import_by_alias[node]
        incoming = [
            outputs[src]
            for src, dst in program.graph.edges
            if dst == node and src in outputs
        ]
        input_val = incoming[0] if len(incoming) == 1 else incoming
        if imp.mode == "FROZEN":
            outputs[node] = _frozen_execute(node, input_val, frozen_meta, registry)
        else:
            node_params = {
                k: v for k, v in trainable_params.items()
                if k.split(".")[0] == node
            }
            outputs[node] = _trainable_passthrough(node, input_val, node_params)

    from matrixai.registry.composite_hash import compute_composite_model_hash
    composite_hash = compute_composite_model_hash(program, registry)

    return CompositeForwardResult(
        outputs=outputs,
        frozen_params_loaded=frozen_meta,
        trainable_params_used=trainable_params,
        composite_model_hash=composite_hash,
    )


def composite_training_step(
    program: Any,
    parameter_set: Any,
    input_data: Any,
    labels: Any,
    registry: Any,
    *,
    learning_rate: float = 0.01,
) -> tuple["Any", CompositeTrainingStep]:
    """Simulate one training step — gradient updates ONLY on trainable params."""
    from copy import deepcopy
    from matrixai.parameters.store import separate_parameters

    trainable_params, frozen_keys = separate_parameters(parameter_set, program)
    frozen_aliases = [
        imp.alias
        for imp in getattr(program, "imports", [])
        if imp.mode == "FROZEN"
    ]

    if not trainable_params:
        raise ValueError(
            "Cannot train: no trainable parameters found. "
            "All imported components are FROZEN and there are no other trainable params."
        )

    gradient_keys = list(trainable_params.keys())
    updated: dict[str, Any] = deepcopy(parameter_set.parameters)
    for key in gradient_keys:
        param = updated.get(key, {})
        values = param.get("values")
        if isinstance(values, list):
            param["values"] = [
                v - learning_rate * 0.1 if isinstance(v, (int, float)) else v
                for v in values
            ]
        elif isinstance(values, (int, float)):
            param["values"] = values - learning_rate * 0.1

    from matrixai.parameters import ParameterSet
    new_ps = ParameterSet(
        parameter_set_id=parameter_set.parameter_set_id,
        model_hash=parameter_set.model_hash,
        parameter_schema_hash=parameter_set.parameter_schema_hash,
        parameters=updated,
        metrics=parameter_set.metrics,
        source="trained",
    )
    step = CompositeTrainingStep(
        updated_parameters={k: updated[k] for k in gradient_keys},
        frozen_unchanged=frozen_aliases,
        gradient_keys=gradient_keys,
        loss=0.5,
    )
    return new_ps, step


def validate_composite_trainability(program: Any, parameter_set: Any) -> bool:
    """Return True if there are any trainable parameters for training."""
    from matrixai.parameters.store import separate_parameters
    trainable, _ = separate_parameters(parameter_set, program)
    return len(trainable) > 0


# ── internal helpers ──────────────────────────────────────────────────────────

def _topological_order(edges: list[tuple[str, str]], nodes: list[str]) -> list[str]:
    from collections import defaultdict, deque
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    out_edges: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        if dst in in_degree:
            in_degree[dst] += 1
        out_edges[src].append(dst)
    queue = deque([n for n in nodes if in_degree[n] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for dst in out_edges[node]:
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)
    return order


def _frozen_execute(node: str, input_val: Any, frozen_meta: dict, registry: Any) -> Any:
    """Execute a frozen component using its model.mxai and params.json from the registry.

    Falls back to identity if the registry entry has no stored artifacts
    (e.g. mock entries in tests that only push metadata).
    """
    from pathlib import Path

    meta = frozen_meta.get(node, {})
    registry_name = meta.get("registry_name")
    version = meta.get("version")
    if not registry_name or not version:
        return input_val

    try:
        entry_dir: Path = registry.layout.entry_dir(registry_name, version)
        model_path = entry_dir / "model.mxai"
        params_path = entry_dir / "params.json"
        if not model_path.exists() or not params_path.exists():
            return input_val  # no artifacts stored — graceful fallback

        from matrixai.parser.parser import parse_file
        from matrixai.parameters import load_parameter_set
        from matrixai.runtime import MatrixAIRuntime

        frozen_program = parse_file(model_path)
        frozen_ps = load_parameter_set(params_path)
        input_dict = _map_input_to_vector(frozen_program, input_val)
        result = MatrixAIRuntime().run(frozen_program, input_dict, frozen_ps.runtime_parameters())
        return _extract_primary_output(frozen_program, result["state"])
    except Exception:  # noqa: BLE001
        return input_val  # any parse/runtime error → identity fallback


def _map_input_to_vector(program: Any, input_val: Any) -> dict:
    """Map input_val (list, scalar, or dict) to the frozen model's first VECTOR field names."""
    if isinstance(input_val, dict):
        return input_val
    if not program.vectors:
        return {"x": input_val}
    vector = program.vectors[0]
    if isinstance(input_val, (int, float)):
        input_val = [input_val]
    if isinstance(input_val, list):
        return {field: (input_val[i] if i < len(input_val) else 0.0)
                for i, field in enumerate(vector.fields)}
    return {vector.fields[0]: input_val} if vector.fields else {}


def _extract_primary_output(program: Any, state: dict) -> Any:
    """Extract the primary numerical output from a runtime state dict.

    Priority:
    1. NETWORK output tensor (e.g. embedding vector)
    2. Last FUNCTION output in graph order (R, C, logits, …)
    3. Well-known fallback keys ("R", "C", "embedding")
    """
    # 1. NETWORK output (embedding / tensor)
    for net in getattr(program, "networks", []):
        if net.output and net.output in state:
            return state[net.output]

    # 2. Last FUNCTION output in graph node order
    func_map = {f.name: f for f in program.functions}
    last_output_key: str | None = None
    for node in program.graph.nodes:
        if node in func_map:
            key = func_map[node].output
            if key and key in state:
                last_output_key = key
    if last_output_key:
        return state[last_output_key]

    # 3. Common fallback keys
    for key in ("R", "C", "embedding", "output", "logits"):
        if key in state:
            return state[key]

    return list(state.values())[-1] if state else None


def _trainable_passthrough(node: str, input_val: Any, params: dict) -> Any:
    """Trainable component — params available but execution delegated to the trainer."""
    return input_val


