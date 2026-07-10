# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from importlib import metadata, util
from typing import Any

from matrixai import __version__ as _MATRIXAI_VERSION
from matrixai.ir import FunctionSpec, LayerSpec, MatrixAIProgram, ParameterSpec, SequenceSpec, VectorSpec
from matrixai.sandbox import SandboxPolicy
from matrixai.types import is_numeric_type


_CONTINUOUS_KINDS = {
    "softmax_linear",
    "sigmoid_linear",
    "linear_regression",
    "sigmoid_threshold",
    "symbolic_expr",
    "symbolic_weighted_sum",
    "normalize",
    "aggregate_max",
    "aggregate_min",
    "aggregate_mean",
    "aggregate_softmax",
    "layer_call",
    "dot",
    "matmul",
    "relu",
    "gelu",
    "layer_norm",
    "residual",
    "mean_pooling",
    "cls_pooling",
    "positional_encoding",
    "scale",
    "softmax",
}
_NON_CONTINUOUS_KINDS = {"aggregate_vote", "select_argmax"}
# These kinds are supported but introduce a runtime boundary (discrete → continuous)
_RUNTIME_BOUNDARY_KINDS = {"embedding_lookup", "attention"}
_TARGET_ALIASES = {
    "differentiable-python": "differentiable_python",
    "differentiable_python": "differentiable_python",
    "torch": "torch",
}
_TORCH_TRAINABLE_KINDS = {"softmax_linear", "sigmoid_linear"}
_TORCH_RUNTIME_BOUNDARY_KINDS = {"sigmoid_threshold"}


def torch_backend_metadata(device: str = "cpu") -> dict[str, Any]:
    available = util.find_spec("torch") is not None
    data: dict[str, Any] = {
        "target": "torch",
        "device": device,
        "dtype": "float32",
        "torch_available": available,
        "execution": "training_minimal",
    }
    if available:
        try:
            data["torch_version"] = metadata.version("torch")
        except metadata.PackageNotFoundError:
            data["torch_version"] = "unknown"
    return data


def _zeros(shape: tuple[int, ...]) -> Any:
    if not shape:
        return 0.0
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    if len(shape) == 2:
        rows, columns = shape
        return [[0.0 for _ in range(columns)] for _ in range(rows)]
    return {"shape": list(shape), "fill": 0.0}


def _ones(shape: tuple[int, ...]) -> Any:
    if not shape:
        return 1.0
    if len(shape) == 1:
        return [1.0 for _ in range(shape[0])]
    if len(shape) == 2:
        rows, columns = shape
        return [[1.0 for _ in range(columns)] for _ in range(rows)]
    return {"shape": list(shape), "fill": 1.0}


_BIAS_NAMES = {"b", "bias"}
_GAIN_NAMES = {"gain", "scale", "gamma"}
_DENSE_INITIALIZER_FOR_ACTIVATION: dict[str, str] = {
    "relu": "he_normal",
    "linear": "xavier_normal",
    "sigmoid": "xavier_normal",
    "softmax": "xavier_normal",
    "tanh": "xavier_normal",
}


def _infer_layer_param_role(name: str) -> str:
    """Infer role for a LAYER parameter from its name."""
    if name in _BIAS_NAMES or (name.startswith("b") and name[1:].isdigit()):
        return "bias"
    if name in _GAIN_NAMES:
        return "gain"
    return "weights"


def _deterministic_uniform(shape: tuple[int, ...]) -> Any:
    if not shape:
        return 0.0
    total = 1
    for size in shape:
        total *= max(1, size)
    scale = 0.1

    def value(index: int) -> float:
        centered = ((index + 1) / total) - 0.5
        return round(centered * scale, 8)

    if len(shape) == 1:
        return [value(index) for index in range(shape[0])]
    if len(shape) == 2:
        rows, columns = shape
        return [
            [value(row * columns + column) for column in range(columns)]
            for row in range(rows)
        ]
    return {"shape": list(shape), "initializer": "deterministic_uniform"}


def _he_normal(shape: tuple[int, ...]) -> Any:
    """He normal init (std=sqrt(2/fan_in), deterministic seed derived from shape)."""
    if not shape:
        return 0.0
    fan_in = shape[-1] if len(shape) >= 2 else shape[0]
    std = math.sqrt(2.0 / max(1, fan_in))
    rng = random.Random(sum(s * (i + 1) for i, s in enumerate(shape)))
    if len(shape) == 1:
        return [round(rng.gauss(0.0, std), 8) for _ in range(shape[0])]
    if len(shape) == 2:
        rows, cols = shape
        return [[round(rng.gauss(0.0, std), 8) for _ in range(cols)] for _ in range(rows)]
    return {"shape": list(shape), "initializer": "he_normal"}


def _xavier_normal(shape: tuple[int, ...]) -> Any:
    """Xavier normal init (std=sqrt(2/(fan_in+fan_out)), deterministic seed)."""
    if not shape:
        return 0.0
    fan_in = shape[-1] if len(shape) >= 2 else shape[0]
    fan_out = shape[0]
    std = math.sqrt(2.0 / max(1, fan_in + fan_out))
    rng = random.Random(sum(s * (i + 1) * 31 for i, s in enumerate(shape)))
    if len(shape) == 1:
        return [round(rng.gauss(0.0, std), 8) for _ in range(shape[0])]
    if len(shape) == 2:
        rows, cols = shape
        return [[round(rng.gauss(0.0, std), 8) for _ in range(cols)] for _ in range(rows)]
    return {"shape": list(shape), "initializer": "xavier_normal"}


@dataclass(frozen=True)
class BackendNodeReport:
    node: str
    node_type: str
    supported: bool
    differentiable: bool
    reason: str = ""
    kind: str = ""
    output_shape: tuple[int, ...] | None = None
    type_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "node": self.node,
            "node_type": self.node_type,
            "supported": self.supported,
            "differentiable": self.differentiable,
        }
        if self.kind:
            data["kind"] = self.kind
        if self.output_shape is not None:
            data["output_shape"] = list(self.output_shape)
        if self.type_constraints:
            data["type_constraints"] = self.type_constraints
        if self.reason:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True)
class TrainableParameter:
    function: str
    name: str
    role: str
    shape: tuple[int, ...] | None = None
    path: str = ""  # non-empty for hierarchical layer params; empty = use name as flat key
    initializer_override: str = ""  # when set, replaces the default initializer logic

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "function": self.function,
            "name": self.name,
            "role": self.role,
        }
        if self.shape is not None:
            data["shape"] = list(self.shape)
        return data

    # Cap for materializing `initial_value` in the manifest. The manifest is a
    # CONTRACT DESCRIPTION (name/shape/initializer), not a weights container:
    # generating every value in pure Python is O(params) CPU and RAM — a
    # 16x16384 net (~4B params) took ~1h of CPU and >16 GB just to build a
    # manifest nobody renders (the playground card and the Studio only use the
    # metadata columns). 65_536 = one 256x256 layer, so every default-tapered
    # net (max width 256) keeps full values and stays byte-identical; only
    # explicitly big prompt-built nets get metadata-only entries. The compiled
    # differentiable-python `initial_parameters()` already skips entries
    # without `initial_value` (and embedding billions of literals in generated
    # source was never viable anyway).
    MANIFEST_MAX_ELEMENTS = 65_536

    def to_manifest_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data["dtype"] = "float32"
        data["initializer"] = self.initializer
        data["trainable"] = True
        data["path"] = self.path or self.name
        if self.shape is not None:
            elements = math.prod(self.shape)
            if elements <= self.MANIFEST_MAX_ELEMENTS:
                data["initial_value"] = self.initial_value
            else:
                data["initial_value_omitted"] = {
                    "elements": elements,
                    "reason": (
                        f"tensor exceeds the manifest cap ({self.MANIFEST_MAX_ELEMENTS} "
                        "elements); initialize from 'initializer' + 'shape'"
                    ),
                }
        return data

    @property
    def initializer(self) -> str:
        if self.initializer_override:
            return self.initializer_override
        if self.role == "bias":
            return "zeros"
        if self.role == "gain":
            return "ones"
        return "deterministic_uniform"

    @property
    def initial_value(self) -> Any:
        if self.shape is None:
            return None
        if self.role == "bias":
            return _zeros(self.shape)
        if self.role == "gain":
            return _ones(self.shape)
        if self.initializer_override == "he_normal":
            return _he_normal(self.shape)
        if self.initializer_override == "xavier_normal":
            return _xavier_normal(self.shape)
        return _deterministic_uniform(self.shape)


@dataclass(frozen=True)
class BackendContractReport:
    target: str
    project: str
    nodes: list[BackendNodeReport] = field(default_factory=list)
    trainable_parameters: list[TrainableParameter] = field(default_factory=list)
    parameter_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    backend: dict[str, Any] = field(default_factory=dict)
    layer_manifest: list[dict[str, Any]] = field(default_factory=list)
    component_manifest: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported_nodes and not self.parameter_errors

    @property
    def unsupported_nodes(self) -> list[BackendNodeReport]:
        return [node for node in self.nodes if not node.supported]

    @property
    def differentiable_nodes(self) -> list[BackendNodeReport]:
        return [node for node in self.nodes if node.differentiable]

    @property
    def tensor_shapes(self) -> dict[str, list[int]]:
        return {
            node.node: list(node.output_shape)
            for node in self.nodes
            if node.output_shape is not None
        }

    @property
    def type_constraints(self) -> dict[str, dict[str, Any]]:
        return {
            node.node: node.type_constraints
            for node in self.nodes
            if node.type_constraints
        }

    @property
    def parameter_manifest(self) -> list[dict[str, Any]]:
        return [param.to_manifest_dict() for param in self.trainable_parameters]

    @property
    def autodiff_plan(self) -> dict[str, Any]:
        parameterized_nodes = sorted({param.function for param in self.trainable_parameters})
        return {
            "ready": self.ok,
            "target": self.target,
            "differentiable_nodes": [node.node for node in self.differentiable_nodes],
            "parameterized_nodes": parameterized_nodes,
            "trainable_parameters": [param.to_dict() for param in self.trainable_parameters],
            "runtime_boundaries": [
                node.node for node in self.nodes if node.supported and not node.differentiable
            ],
            "blocked_nodes": [node.node for node in self.unsupported_nodes],
            "status": "metadata_only",
            "notes": [_autodiff_note_for_target(self.target)],
        }

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ok": self.ok,
            "target": self.target,
            "project": self.project,
            "nodes": [node.to_dict() for node in self.nodes],
            "unsupported_nodes": [node.to_dict() for node in self.unsupported_nodes],
            "differentiable_nodes": [node.node for node in self.differentiable_nodes],
            "tensor_shapes": self.tensor_shapes,
            "type_constraints": self.type_constraints,
            "trainable_parameters": [param.to_dict() for param in self.trainable_parameters],
            "parameter_manifest": self.parameter_manifest,
            "parameter_errors": list(self.parameter_errors),
            "autodiff_plan": self.autodiff_plan,
            "warnings": list(self.warnings),
            "backend": dict(self.backend),
        }
        if self.layer_manifest:
            data["layer_manifest"] = self.layer_manifest
        return data

    def summary(self) -> str:
        status = "portable" if self.ok else "blocked"
        lines = [f"Backend target: {self.target}", f"Project: {self.project}", f"Status: {status}"]
        if self.backend:
            lines.append("Backend metadata:")
            for key, value in self.backend.items():
                lines.append(f"- {key}: {value}")
        if self.trainable_parameters:
            lines.append("Trainable parameters:")
            for parameter in self.trainable_parameters:
                shape = f" shape={list(parameter.shape)}" if parameter.shape is not None else ""
                lines.append(f"- {parameter.function}.{parameter.name} ({parameter.role}){shape}")
        else:
            lines.append("Trainable parameters: none")
        for node in self.nodes:
            mark = "ok" if node.supported else "blocked"
            differentiable = "differentiable" if node.differentiable else "boundary"
            suffix = f" [{node.kind}]" if node.kind else ""
            shape = f", shape={list(node.output_shape)}" if node.output_shape is not None else ""
            lines.append(f"- {node.node} ({node.node_type}{suffix}): {mark}, {differentiable}{shape}")
            if node.reason:
                lines.append(f"  {node.reason}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        for error in self.parameter_errors:
            lines.append(f"Parameter error: {error}")
        return "\n".join(lines)


class BackendContractAnalyzer:
    """Analyze whether a MatrixAI program fits a future differentiable backend subset."""

    def __init__(self, target: str = "differentiable_python") -> None:
        try:
            self.target = _TARGET_ALIASES[target]
        except KeyError as exc:
            allowed = ", ".join(sorted(_TARGET_ALIASES))
            raise ValueError(f"Unsupported backend contract target {target!r}. Expected one of: {allowed}") from exc

    def analyze(self, program: MatrixAIProgram, *, registry: Any = None) -> BackendContractReport:
        vector_map = {vector.name: vector for vector in program.vectors}
        sequence_map = {seq.name: seq for seq in program.sequences}
        function_map = {function.name: function for function in program.functions}
        distribution_map = {distribution.name: distribution for distribution in program.distributions}
        action_map = {action.name: action for action in program.actions}
        layer_map = {layer.name: layer for layer in program.layers}
        param_map = {p.name: p for p in program.parameters}
        network_map = {net.name: net for net in getattr(program, "networks", [])}
        import_map = {imp.alias: imp for imp in getattr(program, "imports", [])}

        nodes: list[BackendNodeReport] = []
        trainable_parameters: list[TrainableParameter] = []
        warnings: list[str] = []
        has_dense_network = False
        has_composite_network = False
        very_reduced = False

        for node_name in program.graph.nodes:
            if node_name in vector_map:
                nodes.append(self._analyze_vector(vector_map[node_name]))
            elif node_name in sequence_map:
                nodes.append(self._analyze_sequence(sequence_map[node_name]))
            elif node_name in network_map:
                net = network_map[node_name]
                if getattr(net, "kind", "dense_network") == "composite_network":
                    from matrixai.types import check_composite_network_types
                    type_result = check_composite_network_types(net, vector_map, sequence_map)
                    nodes.append(self._analyze_composite_network(net, type_result))
                    trainable_parameters.extend(
                        self._composite_network_trainable_parameters(net, type_result)
                    )
                    has_composite_network = True
                    if sum(1 for b in net.blocks if b.residual_from) >= 2:
                        very_reduced = True
                    if any(e.dim > 16 for e in net.embeddings):
                        very_reduced = True
                else:
                    nodes.append(self._analyze_dense_network(net, vector_map))
                    trainable_parameters.extend(self._dense_network_trainable_parameters(net, vector_map))
                    has_dense_network = True
            elif node_name in function_map:
                function = function_map[node_name]
                nodes.append(self._analyze_function(function, vector_map, layer_map, param_map))
                if function.semantic.kind == "layer_call":
                    trainable_parameters.extend(self._layer_trainable_parameters(function, layer_map))
                else:
                    trainable_parameters.extend(self._trainable_parameters(function, vector_map))
            elif node_name in distribution_map:
                nodes.append(
                    BackendNodeReport(
                        node=node_name,
                        node_type="distribution",
                        supported=True,
                        differentiable=False,
                        kind=distribution_map[node_name].distribution_type,
                        reason="probabilistic distribution is a runtime boundary for this backend contract",
                    )
                )
            elif node_name in action_map:
                action = action_map[node_name]
                decision = SandboxPolicy.mvp_simulate_only().review_action(action)
                nodes.append(
                    BackendNodeReport(
                        node=node_name,
                        node_type="action",
                        supported=decision.allowed,
                        differentiable=False,
                        kind=action.policy,
                        reason="; ".join(decision.reasons) or "action remains simulated and outside differentiable backend",
                    )
                )
            elif node_name in import_map and registry is not None:
                imp = import_map[node_name]
                nodes.append(
                    self._analyze_imported_component(node_name, imp, registry, program)
                )
            elif node_name in import_map:
                nodes.append(
                    BackendNodeReport(
                        node=node_name,
                        node_type="composite_model",
                        supported=True,
                        differentiable=False,
                        kind="imported_component",
                        reason="imported component — registry not provided; type checking skipped",
                    )
                )
            else:
                nodes.append(
                    BackendNodeReport(
                        node=node_name,
                        node_type="unknown",
                        supported=False,
                        differentiable=False,
                        reason="node is not declared in the IR",
                    )
                )

        if has_dense_network:
            warnings.append(
                "dense network nodes have interpretability_level=reduced: "
                "internal activations are not individually auditable"
            )
        if has_composite_network:
            if very_reduced:
                warnings.append(
                    "composite network has interpretability_level=very_reduced: "
                    "≥2 residual blocks or embedding dim>16 — internal representations are non-inspectable"
                )
            else:
                warnings.append(
                    "composite network has interpretability_level=reduced: "
                    "embedding learned representations are not individually auditable"
                )
        if not trainable_parameters:
            warnings.append("no trainable parameters discovered for the selected backend subset")
        parameter_errors = self._explicit_parameter_errors(program.parameters, trainable_parameters)
        layer_manifest = self._build_layer_manifest(program.layers)
        for net in network_map.values():
            if getattr(net, "kind", "dense_network") == "composite_network":
                from matrixai.types import check_composite_network_types
                # Audit finding ALTA-3 (2026-07-10): this call was missing sequence_map,
                # so a SEQUENCE-input network re-typechecked here without knowing its
                # input was ever a valid SEQUENCE — always failing INPUT resolution.
                type_result = check_composite_network_types(net, vector_map, sequence_map)
                layer_manifest.extend(self._build_composite_network_layer_manifest(net, type_result))
            else:
                layer_manifest.extend(self._build_network_layer_manifest(net, vector_map))

        component_manifest = self._build_component_manifest(
            getattr(program, "imports", []), registry
        )
        # Emit warnings for imported components
        for entry_info in component_manifest:
            if entry_info.get("error"):
                continue
            alias = entry_info.get("alias", "?")
            interp = entry_info.get("interpretability_level", "full")
            if interp != "full":
                warnings.append(
                    f"imported component {alias!r} has interpretability_level={interp}: "
                    f"internal representations may not be individually auditable"
                )
            comp_version = entry_info.get("matrixai_version", "")
            if comp_version and comp_version != _MATRIXAI_VERSION:
                warnings.append(
                    f"imported component {alias!r} was built with matrixai {comp_version}, "
                    f"current version is {_MATRIXAI_VERSION}"
                )

        return BackendContractReport(
            target=self.target,
            project=program.project,
            nodes=nodes,
            trainable_parameters=trainable_parameters,
            parameter_errors=parameter_errors,
            warnings=warnings,
            backend=self._backend_metadata(),
            layer_manifest=layer_manifest,
            component_manifest=component_manifest,
        )

    # ── composite model (imported components) ─────────────────────────────────

    def _analyze_imported_component(
        self, node_name: str, imp: Any, registry: Any, program: Any
    ) -> BackendNodeReport:
        from matrixai.registry.model_registry import EntryNotFoundError
        try:
            entry = registry.get(imp.registry_name, imp.version)
        except EntryNotFoundError:
            return BackendNodeReport(
                node=node_name,
                node_type="composite_model",
                supported=False,
                differentiable=False,
                kind="imported_component",
                reason=f"registry entry {imp.registry_name}@{imp.version} not found",
            )

        # Check if intermediate node (has both in- and out-edges)
        out_nodes = {src for src, dst in program.graph.edges}
        is_intermediate = node_name in out_nodes and any(
            dst == node_name for src, dst in program.graph.edges
        )

        blockers = list(getattr(entry, "blockers", []))
        if is_intermediate and "real_with_audit_action" in blockers:
            return BackendNodeReport(
                node=node_name,
                node_type="composite_model",
                supported=False,
                differentiable=False,
                kind="imported_component",
                reason=f"{node_name} is an intermediate component with real_with_audit action — not allowed",
            )

        reason_parts = []
        if entry.interpretability_level != "full":
            reason_parts.append(
                f"interpretability_level={entry.interpretability_level}"
            )
        if blockers:
            reason_parts.append(f"blockers: {', '.join(blockers)}")

        return BackendNodeReport(
            node=node_name,
            node_type="composite_model",
            supported=not blockers,
            differentiable=False,
            kind=f"imported_{imp.mode.lower()}",
            reason="; ".join(reason_parts) if reason_parts else "",
        )

    def _build_component_manifest(self, imports: list[Any], registry: Any) -> list[dict[str, Any]]:
        if not imports or registry is None:
            return []
        manifest = []
        for imp in imports:
            try:
                from matrixai.registry.model_registry import EntryNotFoundError
                entry = registry.get(imp.registry_name, imp.version)
                manifest.append({
                    "alias": imp.alias,
                    "registry_name": imp.registry_name,
                    "version": imp.version,
                    "mode": imp.mode,
                    "entry_hash": entry.entry_hash,
                    "interpretability_level": entry.interpretability_level,
                    "metrics": dict(entry.metrics),
                    "matrixai_version": entry.matrixai_version,
                    "blockers": list(getattr(entry, "blockers", [])),
                })
            except Exception:  # noqa: BLE001
                manifest.append({
                    "alias": imp.alias,
                    "registry_name": imp.registry_name,
                    "version": imp.version,
                    "mode": imp.mode,
                    "error": "registry entry not found",
                })
        return manifest

    # ── existing helpers ───────────────────────────────────────────────────────

    def _analyze_vector(self, vector: VectorSpec) -> BackendNodeReport:
        incompatible = []
        type_constraints = self._vector_type_constraints(vector)
        for field_name in vector.fields:
            field_type = vector.field_types.get(field_name)
            if field_type is not None and not is_numeric_type(field_type):
                incompatible.append(f"{field_name}: {field_type.name}")
        if incompatible:
            return BackendNodeReport(
                node=vector.name,
                node_type="vector",
                supported=False,
                differentiable=False,
                output_shape=(vector.size,),
                type_constraints=type_constraints,
                reason="non-numeric vector field(s): " + ", ".join(incompatible),
            )
        return BackendNodeReport(
            node=vector.name,
            node_type="vector",
            supported=True,
            differentiable=True,
            output_shape=(vector.size,),
            type_constraints=type_constraints,
            reason="numeric vector input",
        )

    def _analyze_sequence(self, seq: SequenceSpec) -> BackendNodeReport:
        return BackendNodeReport(
            node=seq.name,
            node_type="sequence",
            supported=True,
            differentiable=False,
            output_shape=(seq.length,),
            reason=f"discrete token sequence (length={seq.length}, vocab_size={seq.vocab_size})",
        )

    def _analyze_dense_network(self, network: Any, vector_map: dict[str, Any]) -> BackendNodeReport:
        output_shape: tuple[int, ...] | None = None
        if network.layers:
            output_shape = (network.layers[-1].units,)
        return BackendNodeReport(
            node=network.name,
            node_type="dense_network",
            supported=True,
            differentiable=True,
            kind="dense_network",
            output_shape=output_shape,
            reason="dense network is a fully differentiable parameterized subgraph",
        )

    def _analyze_composite_network(self, network: Any, type_result: Any) -> BackendNodeReport:
        # Audit finding ALTA-3 (2026-07-10): a network with a BLOCK TRANSFORMER was
        # unconditionally reported supported=True/differentiable=True — but the
        # block's forward/parameter lowering doesn't exist yet (TRANSFORMER_BLOQUE
        # C2+). Fail closed until then instead of claiming a fully differentiable
        # subgraph that CLI/estimators/ParameterSet construction would then build
        # incompletely (missing the block's own weights).
        if getattr(network, "transformer_blocks", []):
            return BackendNodeReport(
                node=network.name,
                node_type="composite_network",
                supported=False,
                differentiable=False,
                kind="composite_network",
                reason=(
                    "composite network contains a BLOCK TRANSFORMER — forward and "
                    "parameter lowering for it are not implemented yet "
                    "(TRANSFORMER_BLOQUE C2+); marked unsupported until then"
                ),
            )
        output_shape: tuple[int, ...] | None = None
        for layer in reversed(getattr(type_result, "resolved_layers", [])):
            if layer.layer_type == "Dense":
                output_shape = (layer.units,)
                break
        if output_shape is None:
            for block in reversed(getattr(type_result, "resolved_blocks", [])):
                for layer in reversed(block.layers):
                    if layer.layer_type == "Dense":
                        output_shape = (layer.units,)
                        break
                if output_shape is not None:
                    break
        return BackendNodeReport(
            node=network.name,
            node_type="composite_network",
            supported=True,
            differentiable=True,
            kind="composite_network",
            output_shape=output_shape,
            reason="composite network is a fully differentiable parameterized subgraph",
        )

    def _composite_network_trainable_parameters(
        self, network: Any, type_result: Any
    ) -> list[TrainableParameter]:
        from matrixai.parameters.network_params import composite_network_parameter_manifest
        # Audit finding ALTA-3: mirrors _analyze_composite_network's fail-closed —
        # no trainable-parameter manifest until the block's own lowering exists.
        if getattr(network, "transformer_blocks", []):
            return []
        if not type_result.ok:
            return []
        manifest = composite_network_parameter_manifest(network.name, network, type_result)
        _role_map = {
            "weights": "weights",
            "bias": "bias",
            "gamma": "gain",
            "beta": "bias",
            "embedding_table": "weights",
        }
        network_prefix = network.name + "."
        result: list[TrainableParameter] = []
        for entry in manifest:
            full_path = entry["path"]
            param_name = full_path[len(network_prefix):] if full_path.startswith(network_prefix) else full_path
            result.append(TrainableParameter(
                function=network.name,
                name=param_name,
                role=_role_map.get(entry["role"], "weights"),
                shape=tuple(entry["shape"]),
                path=full_path,
                initializer_override=entry["initializer"],
            ))
        return result

    def _build_composite_network_layer_manifest(
        self, network: Any, type_result: Any
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        # Audit finding ALTA-3 (2026-07-10): a network with a BLOCK TRANSFORMER has
        # no parameter lowering yet — showing only the embedding (with its vocab
        # now correctly resolved, never the raw 0 sentinel) would still misrepresent
        # the architecture as fully described. Report the block explicitly as
        # pending instead of silently omitting it or leaking a shape=[0, dim] entry.
        for tb in getattr(network, "transformer_blocks", []):
            entries.append({
                "layer": f"{network.name}.{tb.name}",
                "network": network.name,
                "layer_type": "TransformerBlock",
                "block_name": tb.name,
                "differentiable": False,
                "reason": (
                    "forward and parameter lowering not implemented yet "
                    "(TRANSFORMER_BLOQUE C2+)"
                ),
            })
            return entries

        # Embedding entries — vocab resolved (never the raw inherit-from-SEQUENCE
        # sentinel 0; ALTA-3): use type_result.resolved_embeddings when available.
        embeddings = getattr(type_result, "resolved_embeddings", None) or getattr(network, "embeddings", [])
        for emb in embeddings:
            entries.append({
                "layer": f"{network.name}.embedding.{emb.name}",
                "network": network.name,
                "layer_type": "Embedding",
                "embedding_name": emb.name,
                "source": emb.source,
                "vocab": emb.vocab,
                "dim": emb.dim,
                "differentiable": True,
                "trainable_param_count": 1,
                "parameters": [{
                    "path": f"{network.name}.{emb.name}.table",
                    "name": f"{emb.name}.table",
                    "shape": [emb.vocab, emb.dim],
                    "dtype": "float32",
                    "initializer": "xavier_normal",
                    "trainable": True,
                }],
            })

        # Interleave top_layers and blocks by textual position
        body_items: list[tuple[int, str, Any]] = []
        for layer in getattr(type_result, "resolved_layers", []):
            body_items.append((layer.index * 2, "layer", layer))
        for block in getattr(type_result, "resolved_blocks", []):
            pos = getattr(block, "position", 0)
            body_items.append((pos * 2 + 1, "block", block))
        body_items.sort(key=lambda x: x[0])

        for _, kind, spec in body_items:
            if kind == "layer":
                entries.append(self._composite_layer_entry(network.name, network.name, spec))
            else:
                block = spec
                block_prefix = f"{network.name}.{block.name}"
                sub_layers = [
                    self._composite_layer_entry(network.name, block_prefix, lyr)
                    for lyr in block.layers
                ]
                entries.append({
                    "layer": f"{network.name}.{block.name}",
                    "network": network.name,
                    "block_name": block.name,
                    "layer_type": "Block",
                    "residual_from": block.residual_from,
                    "input_shape": list(block.input_shape),
                    "output_shape": list(block.output_shape),
                    "differentiable": True,
                    "layers": sub_layers,
                })

        return entries

    def _composite_layer_entry(
        self, network_name: str, prefix: str, layer: Any
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "layer": f"{prefix}.L{layer.index}",
            "network": network_name,
            "layer_index": layer.index,
            "layer_type": layer.layer_type,
            "input_shape": list(layer.input_shape),
            "output_shape": list(layer.output_shape),
            "differentiable": True,
        }
        if layer.layer_type == "Dense":
            in_dim = layer.input_shape[-1] if layer.input_shape else 0
            units = layer.units
            initializer = _DENSE_INITIALIZER_FOR_ACTIVATION.get(layer.activation, "xavier_normal")
            entry["units"] = units
            entry["activation"] = layer.activation
            entry["trainable_param_count"] = 2
            entry["parameters"] = [
                {
                    "path": f"{prefix}.L{layer.index}.W",
                    "name": f"L{layer.index}.W",
                    "shape": [units, in_dim],
                    "dtype": "float32",
                    "initializer": initializer,
                    "trainable": True,
                },
                {
                    "path": f"{prefix}.L{layer.index}.b",
                    "name": f"L{layer.index}.b",
                    "shape": [units],
                    "dtype": "float32",
                    "initializer": "zeros",
                    "trainable": True,
                },
            ]
        elif layer.layer_type == "LayerNorm":
            features = layer.input_shape[-1] if layer.input_shape else 0
            entry["trainable_param_count"] = 2
            entry["parameters"] = [
                {
                    "path": f"{prefix}.L{layer.index}.gamma",
                    "name": f"L{layer.index}.gamma",
                    "shape": [features],
                    "dtype": "float32",
                    "initializer": "ones",
                    "trainable": True,
                },
                {
                    "path": f"{prefix}.L{layer.index}.beta",
                    "name": f"L{layer.index}.beta",
                    "shape": [features],
                    "dtype": "float32",
                    "initializer": "zeros",
                    "trainable": True,
                },
            ]
        elif layer.layer_type == "Dropout":
            entry["rate"] = layer.rate
            entry["dropout_active"] = True
            entry["trainable_param_count"] = 0
            entry["parameters"] = []
        else:
            entry["trainable_param_count"] = 0
            entry["parameters"] = []
        return entry

    def _dense_network_trainable_parameters(
        self, network: Any, vector_map: dict[str, Any]
    ) -> list[TrainableParameter]:
        result: list[TrainableParameter] = []
        input_vector = vector_map.get(network.input)
        input_dim: int | None = input_vector.size if input_vector is not None else None
        for layer in network.layers:
            units = layer.units
            w_shape: tuple[int, ...] | None = (units, input_dim) if input_dim is not None else None
            b_shape: tuple[int, ...] = (units,)
            result.append(TrainableParameter(
                function=network.name,
                name=f"W{layer.index}",
                role="weights",
                shape=w_shape,
                path=f"{network.name}.W{layer.index}",
                initializer_override=_DENSE_INITIALIZER_FOR_ACTIVATION.get(layer.activation, "xavier_normal"),
            ))
            result.append(TrainableParameter(
                function=network.name,
                name=f"b{layer.index}",
                role="bias",
                shape=b_shape,
                path=f"{network.name}.b{layer.index}",
            ))
            input_dim = units
        return result

    def _build_network_layer_manifest(self, network: Any, vector_map: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        input_vector = vector_map.get(network.input)
        input_dim: int | None = input_vector.size if input_vector is not None else None
        for layer in network.layers:
            units = layer.units
            w_shape = [units, input_dim] if input_dim is not None else []
            b_shape = [units]
            initializer = _DENSE_INITIALIZER_FOR_ACTIVATION.get(layer.activation, "xavier_normal")
            entries.append({
                "layer": f"{network.name}.layer{layer.index}",
                "network": network.name,
                "layer_index": layer.index,
                "activation": layer.activation,
                "trainable_param_count": 2,
                "parameters": [
                    {
                        "path": f"{network.name}.W{layer.index}",
                        "name": f"W{layer.index}",
                        "shape": w_shape,
                        "dtype": "float32",
                        "initializer": initializer,
                        "trainable": True,
                    },
                    {
                        "path": f"{network.name}.b{layer.index}",
                        "name": f"b{layer.index}",
                        "shape": b_shape,
                        "dtype": "float32",
                        "initializer": "zeros",
                        "trainable": True,
                    },
                ],
            })
            input_dim = units
        return entries

    def _analyze_function(
        self,
        function: FunctionSpec,
        vector_map: dict[str, VectorSpec],
        layer_map: dict[str, LayerSpec] | None = None,
        param_map: dict[str, ParameterSpec] | None = None,
    ) -> BackendNodeReport:
        kind = function.semantic.kind
        if kind == "layer_call":
            layer_name = function.semantic.parameters.get("layer", "")
            if not layer_map or layer_name not in layer_map:
                return BackendNodeReport(
                    node=function.name,
                    node_type="function",
                    supported=False,
                    differentiable=False,
                    kind=kind,
                    reason=f"call_layer references undefined LAYER: {layer_name!r}",
                )
        shape_error = self._check_operand_shapes(function, vector_map, param_map or {})
        if shape_error:
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=False,
                differentiable=False,
                kind=kind,
                reason=shape_error,
            )
        output_shape = self._function_output_shape(function, vector_map)
        if self.target == "torch":
            return self._analyze_torch_function(function, vector_map, output_shape)
        if kind in _RUNTIME_BOUNDARY_KINDS:
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=True,
                differentiable=False,
                kind=kind,
                output_shape=output_shape,
                type_constraints=self._function_type_constraints(function),
                reason=f"{kind} introduces a runtime boundary (discrete/external input)",
            )
        if kind in _CONTINUOUS_KINDS:
            reason = "continuous function kind supported by backend contract"
            if kind in {"aggregate_max", "aggregate_min", "normalize"}:
                reason = "piecewise continuous function kind supported with boundary caveats"
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=True,
                differentiable=True,
                kind=kind,
                output_shape=output_shape,
                type_constraints=self._function_type_constraints(function),
                reason=reason,
            )
        if kind in _NON_CONTINUOUS_KINDS:
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=False,
                differentiable=False,
                kind=kind,
                output_shape=output_shape,
                type_constraints=self._function_type_constraints(function),
                reason="discrete function kind is outside the continuous backend subset",
            )
        return BackendNodeReport(
            node=function.name,
            node_type="function",
            supported=False,
            differentiable=False,
            kind=kind,
            output_shape=output_shape,
            type_constraints=self._function_type_constraints(function),
            reason="function kind is not in the backend contract",
        )

    def _analyze_torch_function(
        self,
        function: FunctionSpec,
        vector_map: dict[str, VectorSpec],
        output_shape: tuple[int, ...] | None,
    ) -> BackendNodeReport:
        kind = function.semantic.kind
        type_constraints = self._function_type_constraints(function)
        if kind == "linear_regression":
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=False,
                differentiable=False,
                kind=kind,
                output_shape=output_shape,
                type_constraints=type_constraints,
                reason="torch regression training is gated for P17.1; use stdlib for mse regression",
            )
        if kind in _TORCH_TRAINABLE_KINDS:
            if self._input_dim(function, vector_map) is None:
                return BackendNodeReport(
                    node=function.name,
                    node_type="function",
                    supported=False,
                    differentiable=False,
                    kind=kind,
                    output_shape=output_shape,
                    type_constraints=type_constraints,
                    reason="torch target requires a fixed numeric VECTOR input shape",
                )
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=True,
                differentiable=True,
                kind=kind,
                output_shape=output_shape,
                type_constraints=type_constraints,
                reason="P5 torch tensor subset supports this trainable linear function",
            )
        if kind in _TORCH_RUNTIME_BOUNDARY_KINDS:
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=True,
                differentiable=False,
                kind=kind,
                output_shape=output_shape,
                type_constraints=type_constraints,
                reason="runtime activation boundary outside the P5 torch trainable subset",
            )
        if kind in _NON_CONTINUOUS_KINDS:
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=False,
                differentiable=False,
                kind=kind,
                output_shape=output_shape,
                type_constraints=type_constraints,
                reason="discrete function kind is outside the P5 torch tensor subset",
            )
        if kind == "layer_call":
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=True,
                differentiable=True,
                kind=kind,
                output_shape=output_shape,
                type_constraints=type_constraints,
                reason="layer_call with body ops supported in P11 torch forward",
            )
        if kind in _CONTINUOUS_KINDS:
            return BackendNodeReport(
                node=function.name,
                node_type="function",
                supported=False,
                differentiable=False,
                kind=kind,
                output_shape=output_shape,
                type_constraints=type_constraints,
                reason="function kind is deferred outside the P5 torch MVP tensor subset",
            )
        return BackendNodeReport(
            node=function.name,
            node_type="function",
            supported=False,
            differentiable=False,
            kind=kind,
            output_shape=output_shape,
            type_constraints=type_constraints,
            reason="function kind is not in the P5 torch backend contract",
        )

    def _backend_metadata(self) -> dict[str, Any]:
        if self.target == "torch":
            return torch_backend_metadata()
        return {}

    def _vector_type_constraints(self, vector: VectorSpec) -> dict[str, Any]:
        if not vector.field_types:
            return {}
        return {
            "fields": {
                field: spec.to_dict()
                for field, spec in vector.field_types.items()
            }
        }

    def _function_type_constraints(self, function: FunctionSpec) -> dict[str, Any]:
        if function.output_type is None:
            return {}
        return {
            "output": {
                "name": function.output,
                "type": function.output_type.to_dict(),
            }
        }

    def _trainable_parameters(
        self, function: FunctionSpec, vector_map: dict[str, VectorSpec]
    ) -> list[TrainableParameter]:
        kind = function.semantic.kind
        parameters = function.semantic.parameters
        if kind in {"softmax_linear", "sigmoid_linear", "linear_regression"}:
            result: list[TrainableParameter] = []
            weights = parameters.get("weights")
            bias = parameters.get("bias")
            input_dim = self._input_dim(function, vector_map)
            output_shape = self._function_output_shape(function, vector_map)
            if weights:
                result.append(
                    TrainableParameter(
                        function=function.name,
                        name=str(weights),
                        role="weights",
                        shape=self._weight_shape(kind, input_dim, output_shape),
                    )
                )
            if bias:
                result.append(
                    TrainableParameter(
                        function=function.name,
                        name=str(bias),
                        role="bias",
                        shape=output_shape,  # () for linear_regression and sigmoid_linear
                    )
                )
            return result
        return []

    def _operand_shape(
        self,
        name: str,
        vector_map: dict[str, VectorSpec],
        param_map: dict[str, ParameterSpec],
    ) -> tuple[int, ...] | None:
        """Return the known static shape of a named variable, or None if unknown."""
        if name in vector_map:
            return (vector_map[name].size,)
        if name in param_map:
            spec = param_map[name].type_spec
            if spec is not None:
                shape = spec.parameters.get("shape")
                if shape:
                    return tuple(int(d) for d in shape)
                dim = spec.parameters.get("dim")
                if dim is not None:
                    return (int(dim),)
        return None

    def _check_operand_shapes(
        self,
        function: FunctionSpec,
        vector_map: dict[str, VectorSpec],
        param_map: dict[str, ParameterSpec],
    ) -> str | None:
        """Return an error string if operand shapes are statically incompatible, else None."""
        kind = function.semantic.kind
        inputs = function.semantic.inputs

        if kind in {"dot", "residual"} and len(inputs) >= 2:
            s0 = self._operand_shape(inputs[0], vector_map, param_map)
            s1 = self._operand_shape(inputs[1], vector_map, param_map)
            if s0 is not None and s1 is not None and s0 != s1:
                return (
                    f"{kind}: operand shape mismatch — "
                    f"{inputs[0]}{list(s0)} vs {inputs[1]}{list(s1)}"
                )

        if kind == "matmul" and len(inputs) >= 2:
            s0 = self._operand_shape(inputs[0], vector_map, param_map)
            s1 = self._operand_shape(inputs[1], vector_map, param_map)
            if s0 is not None and s1 is not None:
                inner_a = s0[-1] if s0 else None
                inner_b = s1[-2] if len(s1) >= 2 else (s1[0] if s1 else None)
                if inner_a is not None and inner_b is not None and inner_a != inner_b:
                    return (
                        f"matmul: inner dimension mismatch — "
                        f"{inputs[0]}{list(s0)} @ {inputs[1]}{list(s1)}"
                    )

        if kind == "layer_norm" and len(inputs) >= 3:
            s0 = self._operand_shape(inputs[0], vector_map, param_map)
            s1 = self._operand_shape(inputs[1], vector_map, param_map)
            s2 = self._operand_shape(inputs[2], vector_map, param_map)
            if s0 is not None and s1 is not None and s0 != s1:
                return f"layer_norm: gain shape {list(s1)} != input shape {list(s0)}"
            if s0 is not None and s2 is not None and s0 != s2:
                return f"layer_norm: bias shape {list(s2)} != input shape {list(s0)}"

        if kind == "attention" and len(inputs) >= 2:
            s0 = self._operand_shape(inputs[0], vector_map, param_map)
            s1 = self._operand_shape(inputs[1], vector_map, param_map)
            if s0 is not None and s1 is not None and s0[-1] != s1[-1]:
                return (
                    f"attention: Q last dim {s0[-1]} != K last dim {s1[-1]} — "
                    f"{inputs[0]}{list(s0)} vs {inputs[1]}{list(s1)}"
                )

        return None

    def _input_dim(self, function: FunctionSpec, vector_map: dict[str, VectorSpec]) -> int | None:
        if not function.semantic.inputs:
            return None
        vector = vector_map.get(function.semantic.inputs[0])
        return vector.size if vector is not None else None

    def _function_output_shape(
        self, function: FunctionSpec, vector_map: dict[str, VectorSpec]
    ) -> tuple[int, ...] | None:
        kind = function.semantic.kind
        if kind == "softmax_linear":
            return (self._softmax_class_count(function),)
        if kind == "linear_regression":
            return ()
        if kind in {
            "sigmoid_linear",
            "sigmoid_threshold",
            "symbolic_expr",
            "symbolic_weighted_sum",
            "normalize",
            "aggregate_max",
            "aggregate_min",
            "aggregate_mean",
            "aggregate_softmax",
            "aggregate_vote",
        }:
            return ()
        if kind == "dot":
            return ()  # scalar
        if kind in {"relu", "gelu", "layer_norm", "residual", "mean_pooling", "positional_encoding"}:
            # same shape as first input vector
            first = next((v for v in function.semantic.inputs if v in vector_map), None)
            if first is not None:
                return (vector_map[first].size,)
            return None
        if kind == "cls_pooling":
            return ()  # single scalar element
        # matmul, embedding_lookup, attention, layer_call: shape not statically determined
        return None

    def _softmax_class_count(self, function: FunctionSpec) -> int:
        labels = function.semantic.parameters.get("labels")
        if isinstance(labels, (list, tuple)) and labels:
            return len(labels)
        classes = function.semantic.parameters.get("classes")
        if isinstance(classes, int) and classes > 0:
            return classes
        return 3

    def _weight_shape(
        self,
        kind: str,
        input_dim: int | None,
        output_shape: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if input_dim is None:
            return None
        if kind == "softmax_linear" and output_shape:
            return (output_shape[0], input_dim)
        return (input_dim,)

    def _layer_trainable_parameters(
        self, function: FunctionSpec, layer_map: dict[str, LayerSpec]
    ) -> list[TrainableParameter]:
        layer_name = function.semantic.parameters.get("layer", "")
        layer = layer_map.get(layer_name)
        if layer is None:
            return []
        result: list[TrainableParameter] = []
        for p in layer.params:
            if not p.trainable:
                continue
            shape = None
            type_shape = p.type_spec.parameters.get("shape")
            if type_shape is not None:
                shape = tuple(int(d) for d in type_shape)
            elif p.type_spec.name in {"Vector", "Embedding"}:
                dim = p.type_spec.parameters.get("dim")
                if dim is not None:
                    shape = (int(dim),)
            hierarchical_path = f"{layer_name}.{p.name}"
            result.append(
                TrainableParameter(
                    function=layer_name,
                    name=p.name,
                    role=_infer_layer_param_role(p.name),
                    shape=shape,
                    path=hierarchical_path,
                )
            )
        return result

    def _build_layer_manifest(self, layers: list[LayerSpec]) -> list[dict[str, Any]]:
        result = []
        for layer in layers:
            trainable_params = [p for p in layer.params if p.trainable]
            params_info = []
            for p in trainable_params:
                shape = _shape_from_parameter_type(p.type_spec) or []
                params_info.append({
                    "path": f"{layer.name}.{p.name}",
                    "name": p.name,
                    "shape": shape,
                    "dtype": "float32",
                    "initializer": p.initializer or "deterministic_uniform",
                    "trainable": p.trainable,
                })
            entry: dict[str, Any] = {
                "layer": layer.name,
                "trainable_param_count": len(trainable_params),
                "parameters": params_info,
            }
            if layer.input_type is not None:
                entry["input_type"] = layer.input_type.to_dict()
            if layer.output_type is not None:
                entry["output_type"] = layer.output_type.to_dict()
            result.append(entry)
        return result

    def _explicit_parameter_errors(
        self,
        explicit_parameters: list[ParameterSpec],
        inferred_parameters: list[TrainableParameter],
    ) -> list[str]:
        if not explicit_parameters:
            return []
        errors: list[str] = []
        explicit = {parameter.name: parameter for parameter in explicit_parameters}
        # Layer params (with explicit path set) are declared in LayerSpec, not in program.parameters
        inferred = {parameter.name: parameter for parameter in inferred_parameters if not parameter.path}

        for name in sorted(set(explicit) - set(inferred)):
            errors.append(f"Explicit PARAM {name} is not inferred as trainable by P3")
        for name in sorted(set(inferred) - set(explicit)):
            errors.append(f"Missing explicit PARAM for inferred trainable parameter: {name}")

        for name in sorted(set(explicit) & set(inferred)):
            declared = explicit[name]
            expected = inferred[name]
            if not declared.trainable:
                errors.append(f"Explicit PARAM {name} must be TRAINABLE true for inferred trainable parameter")
            expected_shape = list(expected.shape or [])
            actual_shape = _shape_from_parameter_type(declared.type_spec)
            if actual_shape is None:
                errors.append(f"Explicit PARAM {name} type {declared.type_spec.name} does not declare a parameter shape")
            elif actual_shape != expected_shape:
                errors.append(f"Explicit PARAM {name} expected shape {expected_shape}, got {actual_shape}")
            if declared.initializer and declared.initializer != expected.initializer:
                errors.append(
                    f"Explicit PARAM {name} expected initializer {expected.initializer}, got {declared.initializer}"
                )
        return errors


def _analyze_layer_param_usage(layer: Any) -> dict[str, bool]:
    """Return {param_name: bool} indicating whether each param is live on the path to result.

    A param is considered live if its name appears as an argument of any body_op whose output
    is reachable (directly or transitively) from the final result output via backward BFS.
    Params that only appear in disconnected branches (unreachable from result) are marked False.
    """
    body_ops = list(getattr(layer, "body_ops", ()) or ())
    raw_params = getattr(layer, "params", None)
    if raw_params is None:
        raw_params = getattr(layer, "parameters", [])
    params = list(raw_params or [])

    if not body_ops:
        return {p.name: False for p in params}

    # Identify the result output: op explicitly named "result", or last body_op as fallback
    result_output = body_ops[-1].output
    for op in body_ops:
        if op.output == "result":
            result_output = "result"
            break

    # Backward BFS: collect all intermediate variable names reachable from result_output
    reachable: set[str] = {result_output}
    changed = True
    while changed:
        changed = False
        for op in body_ops:
            if op.output in reachable:
                for arg in op.args:
                    if arg not in reachable:
                        reachable.add(arg)
                        changed = True

    # A param is live if its name appears as an arg of any op whose output is reachable
    param_names: set[str] = {p.name for p in params}
    used: dict[str, bool] = {p.name: False for p in params}
    for op in body_ops:
        if op.output in reachable:
            for arg in op.args:
                if arg in param_names:
                    used[arg] = True

    return used


def _shape_from_parameter_type(type_spec) -> list[int] | None:
    if type_spec.name == "Scalar":
        return []
    if type_spec.name in {"Vector", "Embedding"}:
        dim = type_spec.parameters.get("dim")
        return [int(dim)] if dim is not None else None
    if type_spec.name == "Tensor":
        shape = type_spec.parameters.get("shape")
        return [int(item) for item in shape] if shape is not None else None
    return None


def _autodiff_note_for_target(target: str) -> str:
    if target == "torch":
        return "torch target supports the P5 tensor subset for forward, training and evaluation when PyTorch is installed"
    return "autodiff execution is not implemented by the differentiable_python target"
