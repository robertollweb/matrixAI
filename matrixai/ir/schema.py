# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai.types import TypeSpec


@dataclass(frozen=True)
class ExpressionSpec:
    raw: str
    kind: str
    inputs: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSpec:
    name: str
    size: int
    fields: list[str]
    field_types: dict[str, TypeSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    length: int
    vocab_size: int


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    output: str
    expression: str
    semantic: ExpressionSpec
    output_type: TypeSpec | None = None


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type_spec: TypeSpec
    trainable: bool = True
    initializer: str = ""


@dataclass(frozen=True)
class LayerBodyOp:
    """A single named assignment inside a LAYER body: output = kind(arg0, arg1, ...)."""
    output: str
    kind: str
    args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    params: list[ParameterSpec] = field(default_factory=list)
    input_type: TypeSpec | None = None
    output_type: TypeSpec | None = None
    body: str = ""
    body_ops: tuple[LayerBodyOp, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    variable: str
    distribution_type: str
    source: str
    raw: str


@dataclass(frozen=True)
class GraphSpec:
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    node_types: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionConditionSpec:
    raw: str
    source: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class ActionInputParam:
    name: str
    type: str


@dataclass(frozen=True)
class ActionSpec:
    name: str
    when: str
    call: str
    condition: ActionConditionSpec
    policy: str = "simulate_only"
    target: str = ""
    input_params: tuple[ActionInputParam, ...] = ()


@dataclass(frozen=True)
class VerificationRule:
    goal: str
    description: str
    check: str
    parameter: Any


@dataclass(frozen=True)
class AuditSpec:
    explain: list[str] = field(default_factory=list)


_DENSE_ACTIVATIONS = frozenset({"linear", "relu", "sigmoid", "softmax", "tanh"})
_COMPOSITE_ACTIVATIONS = frozenset({"linear", "relu", "sigmoid", "softmax", "tanh", "gelu"})
_POOL_KINDS = frozenset({"mean", "max", "cls"})


@dataclass(frozen=True)
class DenseLayerSpec:
    index: int
    units: int
    activation: str
    input_shape: list[int] = field(default_factory=list)
    output_shape: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# P19 composite network dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    source: str
    vocab: int    # 0 = inherit from the source SEQUENCE (typecheck resolves/validates)
    dim: int


@dataclass(frozen=True)
class ConcatSpec:
    name: str
    sources: list[str]


@dataclass(frozen=True)
class CompositeLayerSpec:
    """A layer inside a BLOCK or top-level in a composite network."""
    index: int
    layer_type: str          # Dense | LayerNorm | Dropout | Activation | Pool | Reshape
    units: int = 0           # Dense
    activation: str = ""     # Dense
    rate: float = 0.0        # Dropout
    pool_kind: str = ""      # Pool
    activation_kind: str = ""  # Activation (standalone)
    target_shape: list[int] = field(default_factory=list)  # Reshape
    input_shape: list[int] = field(default_factory=list)
    output_shape: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class BlockSpec:
    name: str
    layers: list[CompositeLayerSpec]
    residual_from: str = ""   # "" = no residual; "PREVIOUS" = block input; else named tensor
    input_shape: list[int] = field(default_factory=list)
    output_shape: list[int] = field(default_factory=list)
    position: int = 0         # textual position: how many top_layers precede this block


_TRANSFORMER_ACTIVATIONS = frozenset({"gelu", "relu"})
_TRANSFORMER_POS_KINDS = frozenset({"sinusoidal", "learned"})


@dataclass(frozen=True)
class TransformerBlockSpec:
    """BLOCK <name> TRANSFORMER inside a composite network (TRANSFORMER_BLOQUE C1).

    dim is NEVER declared here: it is inherited from the DIM of the EMBEDDING
    that feeds the block; length/vocab come from the SEQUENCE. ff == 0 means
    "use the default 4*dim" — both are resolved by the composite typecheck
    into resolved_dim / resolved_ff.
    """
    name: str
    layers: int                    # required, >= 1
    heads: int = 4                 # must divide the inherited dim
    ff: int = 0                    # 0 → 4*dim (resolved at typecheck)
    dropout: float = 0.0           # [0, 1); train-only, identity in eval (P19 semantics)
    activation: str = "gelu"       # gelu | relu (FFN)
    pos: str = "sinusoidal"        # sinusoidal (not trainable) | learned (table [L, dim])
    position: int = 0              # textual position: how many top_layers precede this block
    input_shape: list[int] = field(default_factory=list)    # [L, dim] (typecheck)
    output_shape: list[int] = field(default_factory=list)   # [L, dim] (typecheck)
    resolved_dim: int = 0          # inherited from the feeding EMBEDDING (typecheck)
    resolved_ff: int = 0           # ff or 4*dim (typecheck)


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    input: str
    layers: list[DenseLayerSpec]    # P18 dense-only (kind=dense_network)
    output: str
    output_type_str: str
    kind: str = "dense_network"
    # P19 extensions (empty for P18 networks)
    embeddings: list[EmbeddingSpec] = field(default_factory=list)
    concats: list[ConcatSpec] = field(default_factory=list)
    blocks: list[BlockSpec] = field(default_factory=list)
    top_layers: list[CompositeLayerSpec] = field(default_factory=list)
    # TRANSFORMER_BLOQUE C1 (max one per network in v1; typecheck enforces)
    transformer_blocks: list[TransformerBlockSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ImportSpec:
    """Declared import of a versioned model from the local registry."""
    alias: str            # local name used in GRAPH / NETWORK nodes
    registry_name: str    # name of the entry in the registry
    version: str          # "v1", "v2.1", "latest", or any tag
    mode: str             # "FROZEN" | "TRAINABLE"
    resolved_entry_hash: str = ""  # sha256:... filled by CompositeModelResolver (C4)
    resolved_at: str = ""          # ISO8601 UTC, filled by C4


@dataclass(frozen=True)
class MatrixAIProgram:
    project: str
    vectors: list[VectorSpec] = field(default_factory=list)
    sequences: list[SequenceSpec] = field(default_factory=list)
    parameters: list[ParameterSpec] = field(default_factory=list)
    functions: list[FunctionSpec] = field(default_factory=list)
    layers: list[LayerSpec] = field(default_factory=list)
    distributions: list[DistributionSpec] = field(default_factory=list)
    graph: GraphSpec = field(default_factory=GraphSpec)
    actions: list[ActionSpec] = field(default_factory=list)
    audit: AuditSpec = field(default_factory=AuditSpec)
    networks: list[NetworkSpec] = field(default_factory=list)
    imports: list[ImportSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "project": self.project,
            "vectors": [self._vector_to_dict(vector) for vector in self.vectors],
            "functions": [self._function_to_dict(function) for function in self.functions],
            "distributions": [asdict(distribution) for distribution in self.distributions],
            "graph": {
                "nodes": list(self.graph.nodes),
                "edges": [list(edge) for edge in self.graph.edges],
                "node_types": dict(self.graph.node_types),
            },
            "actions": [self._action_to_dict(action) for action in self.actions],
            "audit": asdict(self.audit),
        }
        if self.sequences:
            data["sequences"] = [{"name": s.name, "length": s.length, "vocab_size": s.vocab_size} for s in self.sequences]
        if self.parameters:
            data["parameters"] = [self._parameter_to_dict(parameter) for parameter in self.parameters]
        if self.layers:
            data["layers"] = [self._layer_to_dict(layer) for layer in self.layers]
        if self.networks:
            data["networks"] = [self._network_to_dict(network) for network in self.networks]
        if self.imports:
            data["imports"] = [
                {
                    "alias": imp.alias,
                    "registry_name": imp.registry_name,
                    "version": imp.version,
                    "mode": imp.mode,
                    "resolved_entry_hash": imp.resolved_entry_hash,
                    "resolved_at": imp.resolved_at,
                }
                for imp in self.imports
            ]
        return data

    def _network_to_dict(self, network: "NetworkSpec") -> dict[str, Any]:
        if network.kind == "dense_network":
            return {
                "name": network.name,
                "kind": network.kind,
                "input": network.input,
                "layers": [
                    {
                        "index": layer.index,
                        "type": "Dense",
                        "units": layer.units,
                        "activation": layer.activation,
                        "input_shape": list(layer.input_shape),
                        "output_shape": list(layer.output_shape),
                        "parameters": {
                            "weights": f"{network.name}.W{layer.index}",
                            "bias": f"{network.name}.b{layer.index}",
                        },
                    }
                    for layer in network.layers
                ],
                "output": network.output,
                "output_type": network.output_type_str,
            }
        # composite_network
        d: dict[str, Any] = {
            "name": network.name,
            "kind": network.kind,
            "input": network.input,
        }
        if network.embeddings:
            d["embeddings"] = [
                {
                    "name": e.name,
                    "source": e.source,
                    "vocab": e.vocab,
                    "dim": e.dim,
                    "parameter": f"{network.name}.{e.name}.table",
                    "output_shape": [e.dim],
                }
                for e in network.embeddings
            ]
        if network.concats:
            d["concats"] = [
                {"name": c.name, "sources": list(c.sources)}
                for c in network.concats
            ]
        if network.blocks:
            d["blocks"] = [self._block_to_dict(network.name, b) for b in network.blocks]
        if network.transformer_blocks:
            d["transformer_blocks"] = [
                {
                    "name": tb.name,
                    "layers": tb.layers,
                    "heads": tb.heads,
                    "ff": tb.ff,
                    "dropout": tb.dropout,
                    "activation": tb.activation,
                    "pos": tb.pos,
                    "input_shape": list(tb.input_shape),
                    "output_shape": list(tb.output_shape),
                    "resolved_dim": tb.resolved_dim,
                    "resolved_ff": tb.resolved_ff,
                }
                for tb in network.transformer_blocks
            ]
        if network.top_layers:
            d["layers"] = [
                self._composite_layer_to_dict(network.name, layer)
                for layer in network.top_layers
            ]
        d["output"] = network.output
        d["output_type"] = network.output_type_str
        return d

    def _block_to_dict(self, net_name: str, block: "BlockSpec") -> dict[str, Any]:
        prefix = f"{net_name}.{block.name}"
        return {
            "name": block.name,
            "layers": [self._composite_layer_to_dict(prefix, layer) for layer in block.layers],
            "residual_from": block.residual_from,
            "input_shape": list(block.input_shape),
            "output_shape": list(block.output_shape),
        }

    def _composite_layer_to_dict(self, prefix: str, layer: "CompositeLayerSpec") -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": layer.index,
            "type": layer.layer_type,
            "input_shape": list(layer.input_shape),
            "output_shape": list(layer.output_shape),
        }
        if layer.layer_type == "Dense":
            d["units"] = layer.units
            d["activation"] = layer.activation
            d["parameters"] = {
                "weights": f"{prefix}.L{layer.index}.W",
                "bias": f"{prefix}.L{layer.index}.b",
            }
        elif layer.layer_type == "LayerNorm":
            d["parameters"] = {
                "gamma": f"{prefix}.L{layer.index}.gamma",
                "beta": f"{prefix}.L{layer.index}.beta",
            }
        elif layer.layer_type == "Dropout":
            d["rate"] = layer.rate
            d["parameters"] = {}
        elif layer.layer_type == "Activation":
            d["kind"] = layer.activation_kind
            d["parameters"] = {}
        elif layer.layer_type == "Pool":
            d["kind"] = layer.pool_kind
            d["parameters"] = {}
        elif layer.layer_type == "Reshape":
            d["target_shape"] = list(layer.target_shape)
            d["parameters"] = {}
        return d

    def _action_to_dict(self, action: "ActionSpec") -> dict[str, Any]:
        d = asdict(action)
        # asdict preserves tuple fields as tuples; normalize input_params to list for JSON consistency
        d["input_params"] = list(d.get("input_params", ()))
        return d

    def _vector_to_dict(self, vector: VectorSpec) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": vector.name,
            "size": vector.size,
            "fields": list(vector.fields),
        }
        if vector.field_types:
            data["field_types"] = {
                field: spec.to_dict() for field, spec in vector.field_types.items()
            }
        return data

    def _function_to_dict(self, function: FunctionSpec) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": function.name,
            "output": function.output,
            "expression": function.expression,
            "semantic": asdict(function.semantic),
        }
        if function.output_type is not None:
            data["output_type"] = function.output_type.to_dict()
        return data

    def _layer_to_dict(self, layer: LayerSpec) -> dict[str, Any]:
        data: dict[str, Any] = {"name": layer.name}
        if layer.params:
            data["parameters"] = [self._parameter_to_dict(p) for p in layer.params]
        if layer.input_type is not None:
            data["input_type"] = layer.input_type.to_dict()
        if layer.output_type is not None:
            data["output_type"] = layer.output_type.to_dict()
        if layer.body:
            data["body"] = layer.body
        if layer.body_ops:
            data["body_ops"] = [
                {"output": op.output, "kind": op.kind, "args": list(op.args)}
                for op in layer.body_ops
            ]
        return data

    def _parameter_to_dict(self, parameter: ParameterSpec) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": parameter.name,
            "type": parameter.type_spec.to_dict(),
            "trainable": parameter.trainable,
        }
        if parameter.initializer:
            data["initializer"] = parameter.initializer
        return data


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_interleaved_body(
    network: Any,
    *,
    reverse: bool = False,
) -> list[tuple[int, str, Any]]:
    """Return (sort_key, kind, spec) triples for top_layers and blocks in textual order.

    sort_key uses layer.index * 2 for top-level layers and block.position * 2 + 1 for
    blocks, preserving the original source order across both sequences.
    kind is "layer" or "block".
    """
    items: list[tuple[int, str, Any]] = []
    for layer in getattr(network, "top_layers", []):
        items.append((layer.index * 2, "layer", layer))
    for block in getattr(network, "blocks", []):
        items.append((getattr(block, "position", 0) * 2 + 1, "block", block))
    items.sort(key=lambda x: x[0], reverse=reverse)
    return items
