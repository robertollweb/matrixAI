# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RangeSpec:
    minimum: float | None = None
    maximum: float | None = None
    inclusive_min: bool = True
    inclusive_max: bool = True

    def contains(self, value: float) -> bool:
        number = float(value)
        if self.minimum is not None:
            if self.inclusive_min and number < self.minimum:
                return False
            if not self.inclusive_min and number <= self.minimum:
                return False
        if self.maximum is not None:
            if self.inclusive_max and number > self.maximum:
                return False
            if not self.inclusive_max and number >= self.maximum:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "inclusive_min": self.inclusive_min,
            "inclusive_max": self.inclusive_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RangeSpec | None:
        if not data:
            return None
        return cls(
            minimum=data.get("min"),
            maximum=data.get("max"),
            inclusive_min=bool(data.get("inclusive_min", True)),
            inclusive_max=bool(data.get("inclusive_max", True)),
        )


@dataclass(frozen=True)
class TypeSpec:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    range: RangeSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name}
        if self.parameters:
            data["parameters"] = dict(self.parameters)
        if self.range is not None:
            data["range"] = self.range.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TypeSpec | None:
        if not data:
            return None
        return cls(
            name=str(data["name"]),
            parameters=dict(data.get("parameters", {})),
            range=RangeSpec.from_dict(data.get("range")),
        )


@dataclass(frozen=True)
class FunctionSignature:
    name: str
    arg_types: tuple[TypeSpec, ...]
    return_type: TypeSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": [arg.to_dict() for arg in self.arg_types],
            "returns": self.return_type.to_dict(),
        }


@dataclass(frozen=True)
class TypeCheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    symbols: dict[str, TypeSpec] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "symbols": {name: spec.to_dict() for name, spec in self.symbols.items()},
        }


SCALAR = TypeSpec("Scalar")
ANY = TypeSpec("Any")
INTEGER = TypeSpec("Integer")
BOOLEAN = TypeSpec("Boolean")
STRING = TypeSpec("String")
LABEL = TypeSpec("Label")
PROBABILITY = TypeSpec("Probability", range=RangeSpec(0.0, 1.0))
SCORE = TypeSpec("Score", range=RangeSpec(0.0, 1.0))
CONFIDENCE = TypeSpec("Confidence", range=RangeSpec(0.0, 1.0))
RISK = TypeSpec("Risk", range=RangeSpec(0.0, 1.0))
LOGIT = TypeSpec("Logit")
TOKEN_COST = TypeSpec("TokenCost", range=RangeSpec(0.0, None))
LATENCY = TypeSpec("Latency", range=RangeSpec(0.0, None))
ACTION_SIGNAL = TypeSpec("ActionSignal", range=RangeSpec(0.0, 1.0))
PROBABILITY_MAP = TypeSpec("ProbabilityMap")
CATEGORICAL = TypeSpec("Categorical")
NORMAL = TypeSpec("Normal")
RECORD = TypeSpec("Record")
SEQUENCE = TypeSpec("Sequence")
ACTION_RESULT = TypeSpec("ActionResult")

_TYPE_ALIASES = {
    "Number": "Scalar",
    "Float": "Scalar",
    "float": "Scalar",
    "number": "Scalar",
    "Int": "Integer",
    "int": "Integer",
    "Bool": "Boolean",
    "bool": "Boolean",
    "Str": "String",
    "str": "String",
    "Distribution.Categorical": "Categorical",
    "Distribution.Normal": "Normal",
}

_TYPE_DEFAULTS = {
    "Any": ANY,
    "Scalar": SCALAR,
    "Integer": INTEGER,
    "Boolean": BOOLEAN,
    "String": STRING,
    "Label": LABEL,
    "Probability": PROBABILITY,
    "Score": SCORE,
    "Confidence": CONFIDENCE,
    "Risk": RISK,
    "Logit": LOGIT,
    "TokenCost": TOKEN_COST,
    "Latency": LATENCY,
    "ActionSignal": ACTION_SIGNAL,
    "ProbabilityMap": PROBABILITY_MAP,
    "Categorical": CATEGORICAL,
    "Normal": NORMAL,
    "Record": RECORD,
    "ActionResult": ACTION_RESULT,
    "Vector": TypeSpec("Vector"),
    "Embedding": TypeSpec("Embedding"),
    "Tensor": TypeSpec("Tensor"),
    "List": TypeSpec("List"),
    "Map": TypeSpec("Map"),
    "Sequence": SEQUENCE,
}

_NUMERIC_TYPES = {
    "Scalar",
    "Integer",
    "Probability",
    "Score",
    "Confidence",
    "Risk",
    "Logit",
    "TokenCost",
    "Latency",
    "ActionSignal",
}
_STRUCTURED_TYPES = {"Vector", "Embedding", "Tensor", "Record", "List", "Map", "ProbabilityMap", "Sequence"}
_DISTRIBUTION_TYPES = {"Categorical", "Normal"}
_STRING_TYPES = {"String", "Label"}

NATIVE_FUNCTION_SIGNATURES = {
    "normalize": FunctionSignature("normalize", (SCALAR,), PROBABILITY),
    "sigmoid": FunctionSignature("sigmoid", (SCALAR,), PROBABILITY),
    "sigmoid_product": FunctionSignature("sigmoid_product", (PROBABILITY,), PROBABILITY),
    "sigmoid_or": FunctionSignature("sigmoid_or", (PROBABILITY,), PROBABILITY),
    "clip": FunctionSignature("clip", (SCALAR, SCALAR, SCALAR), SCALAR),
    "scale": FunctionSignature("scale", (SCALAR, SCALAR, SCALAR), PROBABILITY),
    "abs": FunctionSignature("abs", (SCALAR,), SCALAR),
    "max": FunctionSignature("max", (SCALAR,), SCALAR),
    "min": FunctionSignature("min", (SCALAR,), SCALAR),
    "mean": FunctionSignature("mean", (SCALAR,), SCALAR),
    "softmax": FunctionSignature("softmax", (SCALAR,), PROBABILITY_MAP),
    "argmax": FunctionSignature("argmax", (PROBABILITY_MAP,), LABEL),
    "relevance": FunctionSignature("relevance", (RECORD,), SCORE),
    "coherence": FunctionSignature("coherence", (RECORD,), SCORE),
    "confidence": FunctionSignature("confidence", (RECORD,), CONFIDENCE),
    "novelty": FunctionSignature("novelty", (RECORD,), SCORE),
    "safety": FunctionSignature("safety", (RECORD,), SCORE),
    "quality": FunctionSignature("quality", (RECORD,), SCORE),
    "cost": FunctionSignature("cost", (RECORD,), TOKEN_COST),
    "latency": FunctionSignature("latency", (RECORD,), LATENCY),
    "token_cost": FunctionSignature("token_cost", (RECORD,), TOKEN_COST),
}


def canonical_type_name(name: str) -> str:
    stripped = name.strip()
    return _TYPE_ALIASES.get(stripped, stripped)


def known_type_names() -> list[str]:
    return sorted(_TYPE_DEFAULTS)


def parse_type_spec(text: str) -> TypeSpec:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty type annotation")
    # Handle Record { field: type, ... } syntax before the bracket check
    if raw.startswith("Record") and "{" in raw:
        return _parse_record_spec(raw)
    if "[" in raw:
        if not raw.endswith("]"):
            raise ValueError(f"Invalid type annotation: {text}")
        name, payload = raw.split("[", 1)
        payload = payload[:-1].strip()
    else:
        name, payload = raw, ""

    canonical = canonical_type_name(name)
    if canonical not in _TYPE_DEFAULTS:
        raise ValueError(f"Unknown MatrixAI type: {canonical}")

    base = _TYPE_DEFAULTS[canonical]
    parameters = dict(base.parameters)
    range_spec = base.range

    if payload:
        parts = _split_top_level(payload)
        if canonical == "Vector" and len(parts) == 1 and _is_number(parts[0]):
            parameters["dim"] = int(parts[0])
        elif canonical == "Embedding":
            if len(parts) == 1 and _is_number(parts[0]):
                parameters["dim"] = int(parts[0])
            elif len(parts) == 2 and all(_is_number(p) for p in parts):
                parameters["vocab"] = int(parts[0])
                parameters["dim"] = int(parts[1])
            else:
                parameters["args"] = parts
        elif canonical == "Tensor" and parts and all(_is_number(p) for p in parts):
            parameters["shape"] = [int(p) for p in parts]
        elif canonical == "Sequence" and len(parts) == 2 and _is_number(parts[1]):
            parameters["element_type"] = parse_type_spec(parts[0]).to_dict()
            parameters["length"] = int(parts[1])
        elif canonical == "List" and len(parts) == 1:
            parameters["element_type"] = parse_type_spec(parts[0]).to_dict()
        elif canonical == "Map" and len(parts) == 2:
            parameters["key_type"] = parse_type_spec(parts[0]).to_dict()
            parameters["value_type"] = parse_type_spec(parts[1]).to_dict()
        elif len(parts) == 2 and _is_number(parts[0]) and _is_number(parts[1]):
            range_spec = RangeSpec(float(parts[0]), float(parts[1]))
        else:
            parameters["args"] = parts

    return TypeSpec(name=canonical, parameters=parameters, range=range_spec)


def type_from_dict(data: dict[str, Any] | None) -> TypeSpec | None:
    return TypeSpec.from_dict(data)


def tensor_shape(spec: TypeSpec) -> tuple[int, ...] | None:
    """Return declared shape tuple for Tensor/Embedding types, or None if undeclared."""
    if spec.name == "Tensor":
        shape = spec.parameters.get("shape")
        if shape:
            return tuple(int(d) for d in shape)
    elif spec.name == "Embedding":
        vocab = spec.parameters.get("vocab")
        dim = spec.parameters.get("dim")
        if vocab is not None and dim is not None:
            return (int(vocab), int(dim))
        if dim is not None:
            return (int(dim),)
    return None


def embedding_dims(spec: TypeSpec) -> tuple[int, int] | None:
    """Return (vocab, dim) for Embedding[vocab, dim], or None."""
    if spec.name != "Embedding":
        return None
    vocab = spec.parameters.get("vocab")
    dim = spec.parameters.get("dim")
    if vocab is not None and dim is not None:
        return (int(vocab), int(dim))
    return None


def type_family(spec: TypeSpec | None) -> str:
    if spec is None:
        return "unknown"
    name = canonical_type_name(spec.name)
    if name == "Any":
        return "any"
    if name in _NUMERIC_TYPES:
        return "numeric"
    if name in _STRUCTURED_TYPES:
        return "structured"
    if name in _DISTRIBUTION_TYPES:
        return "distribution"
    if name in _STRING_TYPES:
        return "string"
    if name == "ActionResult":
        return "action"
    return "unknown"


def is_numeric_type(spec: TypeSpec | None) -> bool:
    return type_family(spec) == "numeric"


def type_is_compatible(actual: TypeSpec, expected: TypeSpec) -> bool:
    actual_family = type_family(actual)
    expected_family = type_family(expected)
    if expected_family == "any" or actual_family == "any":
        return True
    if actual.name == expected.name:
        return True
    if actual_family == expected_family == "numeric":
        return True
    if actual_family == expected_family == "structured":
        return True
    return False


def validate_value_against_type(name: str, value: Any, spec: TypeSpec | None) -> list[str]:
    if spec is None or spec.name == "Any":
        return []
    errors: list[str] = []
    family = type_family(spec)
    if family == "numeric":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return [f"{name} expected {spec.name}, got {type(value).__name__}"]
        if spec.range and not spec.range.contains(number):
            errors.append(f"{name}={number} outside {spec.name} range {format_range(spec.range)}")
    elif spec.name in {"Vector", "Embedding", "Tensor"}:
        if not isinstance(value, list):
            errors.append(f"{name} expected {spec.name}, got {type(value).__name__}")
        else:
            dim = spec.parameters.get("dim")
            if dim is not None and len(value) != int(dim):
                errors.append(f"{name} expected {spec.name}[{dim}], got length {len(value)}")
            if spec.name == "Tensor":
                declared_shape = spec.parameters.get("shape")
                if declared_shape is not None:
                    actual_shape = _list_shape(value)
                    expected_shape = [int(d) for d in declared_shape]
                    if actual_shape != expected_shape:
                        errors.append(
                            f"{name}: Tensor shape {actual_shape} != declared shape {expected_shape}"
                        )
    elif spec.name == "List":
        if not isinstance(value, list):
            errors.append(f"{name} expected List, got {type(value).__name__}")
        elif "element_type" in spec.parameters:
            element_spec = TypeSpec.from_dict(spec.parameters["element_type"])
            if element_spec:
                for i, item in enumerate(value):
                    errors.extend(validate_value_against_type(f"{name}[{i}]", item, element_spec))
    elif spec.name == "Map":
        if not isinstance(value, dict):
            errors.append(f"{name} expected Map, got {type(value).__name__}")
    elif spec.name == "Sequence":
        if not isinstance(value, list):
            errors.append(f"{name} expected Sequence, got {type(value).__name__}")
        else:
            length = spec.parameters.get("length")
            if length is not None and len(value) > int(length):
                errors.append(f"{name} Sequence max length {length}, got {len(value)}")
    elif spec.name == "ProbabilityMap":
        if not isinstance(value, dict):
            errors.append(f"{name} expected ProbabilityMap, got {type(value).__name__}")
        else:
            for key, item in value.items():
                try:
                    probability = float(item)
                except (TypeError, ValueError):
                    errors.append(f"{name}.{key} expected probability, got {type(item).__name__}")
                    continue
                if not PROBABILITY.range.contains(probability):  # type: ignore[union-attr]
                    errors.append(f"{name}.{key}={probability} outside Probability range [0, 1]")
    return errors


def format_range(range_spec: RangeSpec) -> str:
    left = "[" if range_spec.inclusive_min else "("
    right = "]" if range_spec.inclusive_max else ")"
    minimum = "-inf" if range_spec.minimum is None else _format_number(range_spec.minimum)
    maximum = "inf" if range_spec.maximum is None else _format_number(range_spec.maximum)
    return f"{left}{minimum}, {maximum}{right}"


def semantic_kind_output_type(kind: str, parameters: dict[str, Any] | None = None) -> TypeSpec:
    parameters = parameters or {}
    if kind == "softmax_linear":
        return PROBABILITY_MAP
    if kind == "linear_regression":
        return SCALAR
    if kind in {"sigmoid_linear", "sigmoid_threshold", "normalize"}:
        return PROBABILITY
    if kind in {"symbolic_expr", "symbolic_weighted_sum"}:
        return SCALAR
    if kind.startswith("aggregate_"):
        method = parameters.get("method", kind.removeprefix("aggregate_"))
        if method in {"softmax", "vote"}:
            return PROBABILITY
        return SCALAR
    if kind == "select_argmax":
        return LABEL
    if kind == "layer_call":
        return ANY
    if kind == "dot":
        return SCALAR
    if kind in {"matmul", "relu", "gelu", "layer_norm", "residual"}:
        return ANY
    if kind in {"embedding_lookup", "attention", "positional_encoding",
                "mean_pooling", "cls_pooling"}:
        return ANY
    return ANY


def distribution_output_type(distribution_type: str) -> TypeSpec:
    canonical = canonical_type_name(distribution_type)
    if canonical == "Categorical":
        return CATEGORICAL
    if canonical == "Normal":
        return NORMAL
    return ANY


@dataclass(frozen=True)
class NetworkTypeResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_type: "TypeSpec | None" = None
    resolved_layers: list[Any] = field(default_factory=list)
    # P19 extras (empty for P18 dense_network results)
    named_shapes: dict[str, list[int]] = field(default_factory=dict)
    resolved_blocks: list[Any] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "output_type": self.output_type.to_dict() if self.output_type else None,
        }


# activation → required output type name (for final layer validation)
_ACTIVATION_OUTPUT_TYPE: dict[str, str] = {
    "linear": "Scalar",
    "sigmoid": "Probability",
    "softmax": "ProbabilityMap",
}

# activations that cannot be the final layer for Probability/ProbabilityMap outputs
_NON_PROB_FINAL_ACTIVATIONS = frozenset({"relu", "tanh"})


def check_network_types(network: Any, vectors_by_name: dict[str, Any]) -> NetworkTypeResult:
    """Shape inference and type checking for a NetworkSpec. Dispatches by network.kind."""
    if getattr(network, "kind", "dense_network") == "composite_network":
        return check_composite_network_types(network, vectors_by_name)
    from matrixai.ir.schema import DenseLayerSpec

    errors: list[str] = []
    warnings: list[str] = []

    vector = vectors_by_name.get(network.input)
    if vector is None:
        return NetworkTypeResult(
            errors=[f"NETWORK {network.name}: INPUT '{network.input}' is not a declared VECTOR"],
        )

    # Check that all vector fields are numeric
    field_types = getattr(vector, "field_types", {}) or {}
    for fname in vector.fields:
        ftype = field_types.get(fname, SCALAR)
        if not is_numeric_type(ftype):
            errors.append(
                f"NETWORK {network.name}: input vector field '{fname}' has non-numeric type "
                f"'{ftype.name}' — dense networks require all-numeric inputs"
            )

    if errors:
        return NetworkTypeResult(errors=errors)

    # Parse declared output type
    try:
        declared_output_type = parse_type_spec(network.output_type_str)
    except ValueError as exc:
        return NetworkTypeResult(
            errors=[f"NETWORK {network.name}: invalid output type '{network.output_type_str}': {exc}"]
        )

    # Shape inference
    resolved: list[DenseLayerSpec] = []
    input_dim = vector.size
    for layer in network.layers:
        in_shape = [input_dim]
        out_shape = [layer.units]
        resolved.append(DenseLayerSpec(
            index=layer.index,
            units=layer.units,
            activation=layer.activation,
            input_shape=in_shape,
            output_shape=out_shape,
        ))
        input_dim = layer.units

    # Validate final layer vs declared output type
    final = network.layers[-1]
    final_activation = final.activation

    if declared_output_type.name == "Scalar":
        if final_activation not in ("linear",):
            errors.append(
                f"NETWORK {network.name}: output type Scalar requires final activation 'linear', "
                f"got '{final_activation}'"
            )
        if final.units != 1:
            errors.append(
                f"NETWORK {network.name}: Scalar output requires units=1 in final layer, "
                f"got units={final.units}"
            )

    elif declared_output_type.name == "Probability":
        if final_activation not in ("sigmoid",):
            errors.append(
                f"NETWORK {network.name}: output type Probability requires final activation 'sigmoid', "
                f"got '{final_activation}'"
            )
        if final.units != 1:
            errors.append(
                f"NETWORK {network.name}: Probability output requires units=1 in final layer, "
                f"got units={final.units}"
            )

    elif declared_output_type.name == "ProbabilityMap":
        if final_activation not in ("softmax",):
            errors.append(
                f"NETWORK {network.name}: output type ProbabilityMap requires final activation 'softmax', "
                f"got '{final_activation}'"
            )
        if final.units < 2:
            errors.append(
                f"NETWORK {network.name}: softmax output requires units >= 2, "
                f"got units={final.units}"
            )

    # Reject relu/tanh as final activation for probability outputs
    if final_activation in _NON_PROB_FINAL_ACTIVATIONS and declared_output_type.name in (
        "Probability", "ProbabilityMap"
    ):
        errors.append(
            f"NETWORK {network.name}: activation '{final_activation}' cannot be used as final "
            f"activation for output type '{declared_output_type.name}'"
        )

    warnings.append(
        f"NETWORK {network.name}: interpretability_level=reduced — "
        "dense neural network with hidden layers"
    )

    inferred_output = declared_output_type if not errors else None
    return NetworkTypeResult(
        errors=errors,
        warnings=warnings,
        output_type=inferred_output,
        resolved_layers=resolved,
    )


def _infer_composite_layer_shape(
    layer: Any, in_shape: list[int], net_name: str
) -> tuple[list[int], list[str]]:
    """Return (output_shape, errors) for a single CompositeLayerSpec."""
    errors: list[str] = []
    if layer.layer_type == "Dense":
        return [layer.units], errors
    elif layer.layer_type in ("LayerNorm", "Dropout", "Activation", "Pool"):
        return list(in_shape), errors
    elif layer.layer_type == "Reshape":
        target = list(layer.target_shape)
        if target:
            in_product = 1
            for d in in_shape:
                in_product *= d
            out_product = 1
            for d in target:
                out_product *= d
            if in_product != out_product:
                errors.append(
                    f"NETWORK {net_name}: LAYER Reshape — product of input dims {in_product} "
                    f"!= product of target dims {out_product}"
                )
            return target, errors
        return list(in_shape), errors
    return list(in_shape), errors


def _validate_composite_output_type(
    net_name: str,
    declared_type: Any,
    activation: str,
    units: int,
    errors: list[str],
) -> None:
    """Validate final Dense layer matches declared output type (reused from dense checker)."""
    if declared_type.name == "Scalar":
        if activation != "linear":
            errors.append(
                f"NETWORK {net_name}: output type Scalar requires final activation 'linear', "
                f"got '{activation}'"
            )
        if units != 1:
            errors.append(
                f"NETWORK {net_name}: Scalar output requires units=1 in final layer, "
                f"got units={units}"
            )
    elif declared_type.name == "Probability":
        if activation != "sigmoid":
            errors.append(
                f"NETWORK {net_name}: output type Probability requires final activation 'sigmoid', "
                f"got '{activation}'"
            )
        if units != 1:
            errors.append(
                f"NETWORK {net_name}: Probability output requires units=1 in final layer, "
                f"got units={units}"
            )
    elif declared_type.name == "ProbabilityMap":
        if activation != "softmax":
            errors.append(
                f"NETWORK {net_name}: output type ProbabilityMap requires final activation 'softmax', "
                f"got '{activation}'"
            )
        if units < 2:
            errors.append(
                f"NETWORK {net_name}: softmax output requires units >= 2, "
                f"got units={units}"
            )


def check_composite_network_types(
    network: Any, vectors_by_name: dict[str, Any]
) -> NetworkTypeResult:
    """Shape inference and type checking for a composite_network (P19)."""
    import dataclasses as _dc
    from matrixai.ir.schema import CompositeLayerSpec, BlockSpec

    errors: list[str] = []
    warnings: list[str] = []

    vector = vectors_by_name.get(network.input)
    if vector is None:
        return NetworkTypeResult(
            errors=[f"NETWORK {network.name}: INPUT '{network.input}' is not a declared VECTOR"]
        )

    # named_shapes: tensor name → shape list
    named_shapes: dict[str, list[int]] = {}
    for fname in getattr(vector, "fields", []):
        named_shapes[fname] = [1]

    current_shape: list[int] = [vector.size]

    # Process embeddings (define named shapes, don't change current_shape)
    for emb in network.embeddings:
        named_shapes[emb.name] = [emb.dim]

    # Process concats (update named_shapes and current_shape)
    for concat in network.concats:
        total_dim = 0
        concat_ok = True
        for src in concat.sources:
            if src not in named_shapes:
                errors.append(
                    f"NETWORK {network.name}: CONCAT '{concat.name}': undeclared source '{src}'"
                )
                concat_ok = False
            else:
                total_dim += named_shapes[src][-1]
        if concat_ok:
            named_shapes[concat.name] = [total_dim]
            current_shape = [total_dim]

    # Build interleaved body ops: sort top_layers and blocks by textual position
    # top_layers: key = layer.index * 2 (even → after layer at position index-1)
    # blocks: key = block.position * 2 + 1 (odd → after position top_layers)
    body_items: list[tuple[int, str, Any]] = []
    for layer in network.top_layers:
        body_items.append((layer.index * 2, "layer", layer))
    for block in network.blocks:
        pos = getattr(block, "position", 0)
        body_items.append((pos * 2 + 1, "block", block))
    body_items.sort(key=lambda x: x[0])

    resolved_top_layers: list[CompositeLayerSpec] = []
    resolved_blocks: list[BlockSpec] = []

    for _, kind, spec in body_items:
        if kind == "layer":
            in_shape = list(current_shape)
            out_shape, layer_errors = _infer_composite_layer_shape(spec, in_shape, network.name)
            errors.extend(layer_errors)
            resolved_top_layers.append(_dc.replace(
                spec,
                input_shape=in_shape,
                output_shape=out_shape,
            ))
            current_shape = out_shape
        else:
            block = spec
            block_input_shape = list(current_shape)
            block_current = list(current_shape)
            resolved_block_layers: list[CompositeLayerSpec] = []

            for layer in block.layers:
                in_shape = list(block_current)
                out_shape, layer_errors = _infer_composite_layer_shape(layer, in_shape, network.name)
                errors.extend(layer_errors)
                resolved_block_layers.append(_dc.replace(
                    layer,
                    input_shape=in_shape,
                    output_shape=out_shape,
                ))
                block_current = out_shape

            # Residual shape check
            if block.residual_from:
                if block.residual_from == "PREVIOUS":
                    residual_source = block_input_shape
                else:
                    residual_source = named_shapes.get(block.residual_from)
                    if residual_source is None:
                        errors.append(
                            f"NETWORK {network.name}: BLOCK '{block.name}': RESIDUAL FROM "
                            f"'{block.residual_from}' references undeclared tensor"
                        )

                if residual_source is not None and block_current != residual_source:
                    errors.append(
                        f"NETWORK {network.name}: BLOCK '{block.name}': RESIDUAL shape mismatch — "
                        f"block output {block_current} != residual source {residual_source}"
                    )

            current_shape = block_current
            resolved_blocks.append(_dc.replace(
                block,
                layers=resolved_block_layers,
                input_shape=block_input_shape,
                output_shape=list(block_current),
            ))

    # Validate output type
    try:
        declared_output_type = parse_type_spec(network.output_type_str)
    except ValueError as exc:
        errors.append(f"NETWORK {network.name}: invalid output type '{network.output_type_str}': {exc}")
        return NetworkTypeResult(errors=errors, named_shapes=named_shapes)

    # Find final Dense layer for output type validation
    final_dense = None
    for layer in reversed(resolved_top_layers):
        if layer.layer_type == "Dense":
            final_dense = layer
            break
    if final_dense is None and resolved_blocks:
        for layer in reversed(resolved_blocks[-1].layers):
            if layer.layer_type == "Dense":
                final_dense = layer
                break

    if final_dense is not None:
        _validate_composite_output_type(
            network.name, declared_output_type,
            final_dense.activation, final_dense.units, errors
        )

    warnings.append(
        f"NETWORK {network.name}: interpretability_level=reduced — "
        "composite network with embeddings/blocks"
    )

    inferred_output = declared_output_type if not errors else None
    return NetworkTypeResult(
        errors=errors,
        warnings=warnings,
        output_type=inferred_output,
        resolved_layers=resolved_top_layers,
        named_shapes=named_shapes,
        resolved_blocks=resolved_blocks,
    )


def check_program_types(program: Any) -> TypeCheckResult:
    checker = _ProgramTypeChecker(program)
    return checker.check()


def check_mx_types(statements: list[Any]) -> TypeCheckResult:
    checker = _MXTypeChecker(statements)
    return checker.check()


class _ProgramTypeChecker:
    def __init__(self, program: Any) -> None:
        self.program = program
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.symbols: dict[str, TypeSpec] = {}

    def check(self) -> TypeCheckResult:
        self._check_parameters()

        for vector in self.program.vectors:
            self._register_vector(vector)

        functions = {function.name: function for function in self.program.functions}
        distributions = {distribution.name: distribution for distribution in self.program.distributions}
        actions = {action.name: action for action in self.program.actions}

        for node in self.program.graph.nodes:
            if node in functions:
                self._register_function(functions[node])
            elif node in distributions:
                self._register_distribution(distributions[node])
            elif node in actions:
                self._register_action(actions[node])

        for function in self.program.functions:
            if function.name not in self.symbols:
                self._register_function(function)
        for distribution in self.program.distributions:
            if distribution.name not in self.symbols:
                self._register_distribution(distribution)
        for action in self.program.actions:
            if action.name not in self.symbols:
                self._register_action(action)

        vectors_by_name = {v.name: v for v in self.program.vectors}
        for network in getattr(self.program, "networks", []):
            self._register_network(network, vectors_by_name)

        return TypeCheckResult(self.errors, self.warnings, self.symbols)

    def _register_network(self, network: Any, vectors_by_name: dict[str, Any]) -> None:
        result = check_network_types(network, vectors_by_name)
        self.errors.extend(result.errors)
        self.warnings.extend(result.warnings)
        if result.output_type is not None:
            self.symbols[network.name] = result.output_type
            self.symbols[network.output] = result.output_type

    def _check_parameters(self) -> None:
        for param in getattr(self.program, "parameters", []):
            spec = getattr(param, "type_spec", None)
            if spec is None:
                continue
            if spec.name in {"Tensor", "Embedding"} and tensor_shape(spec) is None:
                self.warnings.append(
                    f"PARAM {param.name} has type {spec.name} without declared shape"
                )

    def _register_vector(self, vector: Any) -> None:
        self.symbols[vector.name] = TypeSpec("Vector", parameters={"dim": vector.size})
        field_types = getattr(vector, "field_types", {}) or {}
        for field in vector.fields:
            field_type = field_types.get(field, SCALAR)
            self.symbols.setdefault(field, field_type)

    def _register_function(self, function: Any) -> None:
        semantic = function.semantic
        if semantic.kind == "unknown":
            self.errors.append(f"FUNCTION {function.name} has unknown expression semantics")
            return
        for source in semantic.inputs:
            root = source.split(".", 1)[0]
            if root not in self.symbols:
                self.errors.append(f"FUNCTION {function.name} references unknown symbol '{source}'")
        inferred = semantic_kind_output_type(semantic.kind, semantic.parameters)
        declared = getattr(function, "output_type", None)
        if declared is not None and not type_is_compatible(inferred, declared):
            self.errors.append(
                f"FUNCTION {function.name} declares {declared.name} but expression infers {inferred.name}"
            )
        elif declared is not None and declared.range and inferred.range is None and is_numeric_type(inferred):
            self.warnings.append(
                f"FUNCTION {function.name} declares bounded {declared.name}; static range cannot be proven"
            )
        output_type = declared or inferred
        self.symbols[function.name] = output_type
        self.symbols[function.output] = output_type

    def _register_distribution(self, distribution: Any) -> None:
        root = distribution.source.split(",", 1)[0].strip().split(".", 1)[0]
        source_type = self.symbols.get(root)
        if source_type is None:
            self.errors.append(f"DISTRIBUTION {distribution.name} references unknown source '{distribution.source}'")
        elif distribution.distribution_type == "Categorical" and source_type.name not in {"ProbabilityMap", "Any"}:
            self.errors.append(
                f"DISTRIBUTION {distribution.name} expects ProbabilityMap source, got {source_type.name}"
            )
        elif distribution.distribution_type == "Normal" and not is_numeric_type(source_type):
            self.errors.append(
                f"DISTRIBUTION {distribution.name} expects numeric source, got {source_type.name}"
            )
        output_type = distribution_output_type(distribution.distribution_type)
        self.symbols[distribution.name] = output_type
        self.symbols[distribution.variable] = output_type

    def _register_action(self, action: Any) -> None:
        root = action.condition.source.split(".", 1)[0]
        source_type = self.symbols.get(root)
        if source_type is None:
            self.errors.append(
                f"ACTION {action.name} condition references unknown typed source '{action.condition.source}'"
            )
        elif source_type.range and not source_type.range.contains(action.condition.threshold):
            self.errors.append(
                f"ACTION {action.name} threshold {action.condition.threshold} outside "
                f"{source_type.name} range {format_range(source_type.range)}"
            )
        self.symbols[action.name] = ACTION_RESULT


class _MXTypeChecker:
    def __init__(self, statements: list[Any]) -> None:
        self.statements = statements
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.symbols: dict[str, TypeSpec] = {}
        self.definitions = {stmt.name: stmt for stmt in statements}

    def check(self) -> TypeCheckResult:
        for stmt in self.statements:
            inferred = self._infer_expr(stmt.expr, self._param_env(stmt))
            declared = getattr(stmt, "return_type", None)
            if declared is not None:
                if not type_is_compatible(inferred, declared):
                    self.errors.append(
                        f"{stmt.name} declares {declared.name} but expression infers {inferred.name}"
                    )
                elif declared.range and inferred.range is None and is_numeric_type(inferred):
                    self.warnings.append(
                        f"{stmt.name} declares bounded {declared.name}; static range cannot be proven"
                    )
                self.symbols[stmt.name] = declared
            else:
                self.symbols[stmt.name] = inferred
        return TypeCheckResult(self.errors, self.warnings, self.symbols)

    def _param_env(self, stmt: Any) -> dict[str, TypeSpec]:
        env: dict[str, TypeSpec] = {}
        param_types = getattr(stmt, "param_types", {}) or {}
        for param in stmt.params:
            env[param] = param_types.get(param, ANY)
        return env

    def _infer_expr(self, node: Any, env: dict[str, TypeSpec]) -> TypeSpec:
        node_type = type(node).__name__
        if node_type in {"NumberNode", "LiteralNode"}:
            return SCALAR
        if node_type == "VarNode":
            return env.get(node.name, self.symbols.get(node.name, ANY))
        if node_type == "BinaryOpNode":
            left = self._infer_expr(node.left, env)
            right = self._infer_expr(node.right, env)
            if not is_numeric_type(left) and left.name != "Any":
                self.errors.append(f"Operator {node.op} expected numeric left operand, got {left.name}")
            if not is_numeric_type(right) and right.name != "Any":
                self.errors.append(f"Operator {node.op} expected numeric right operand, got {right.name}")
            return SCALAR
        if node_type == "CallNode":
            if node.name in self.definitions:
                declared = getattr(self.definitions[node.name], "return_type", None)
                return declared or self.symbols.get(node.name, SCALAR)
            signature = NATIVE_FUNCTION_SIGNATURES.get(node.name)
            if signature is None:
                self.warnings.append(f"No native signature for function {node.name}")
                return ANY
            return signature.return_type
        return ANY


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(value)


def _list_shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    return [len(value)] + _list_shape(value[0])


def _split_top_level(text: str) -> list[str]:
    """Split by commas only at bracket-depth 0, so Tensor[4,8] stays intact."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def _parse_record_spec(raw: str) -> TypeSpec:
    """Parse 'Record { field: type, ... }' into TypeSpec(name='Record', parameters={'fields': {...}})."""
    brace_start = raw.index("{")
    brace_end = raw.rindex("}")
    inner = raw[brace_start + 1:brace_end].strip()
    fields: dict[str, Any] = {}
    if inner:
        for part in _split_top_level(inner):
            if ":" not in part:
                raise ValueError(f"Invalid Record field definition: {part!r}")
            colon_idx = part.index(":")
            field_name = part[:colon_idx].strip()
            field_type_str = part[colon_idx + 1:].strip()
            fields[field_name] = parse_type_spec(field_type_str).to_dict()
    return TypeSpec(name="Record", parameters={"fields": fields})


# ---------------------------------------------------------------------------
# P21 composite type checking
# ---------------------------------------------------------------------------

class TypeCompatibilityError(Exception):
    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


def _composite_types_compatible(src: dict[str, Any], dst: dict[str, Any]) -> bool:
    """Return True if src output dict-type is compatible with dst input dict-type."""
    src_kind = src.get("kind", "")
    dst_kind = dst.get("kind", "")
    if not src_kind or not dst_kind:
        return True
    if src_kind == "Any" or dst_kind == "Any":
        return True
    if src_kind != dst_kind:
        return False
    if src_kind == "Tensor":
        src_shape = src.get("shape", [])
        dst_shape = dst.get("shape", [])
        if src_shape and dst_shape and list(src_shape) != list(dst_shape):
            return False
    return True


def _node_output_type(node: str, program: Any, entries: dict[str, Any]) -> dict[str, Any] | None:
    if node in entries:
        return entries[node].output_type or {}
    for v in getattr(program, "vectors", []):
        if v.name == node:
            return {"kind": "VECTOR", "name": v.name}
    for net in getattr(program, "networks", []):
        if net.name == node:
            return {"kind": net.output_type_str}
    return None


def _node_input_type(node: str, entries: dict[str, Any]) -> dict[str, Any] | None:
    if node in entries:
        return entries[node].input_type or {}
    return None


def check_composite_program_types(
    program: Any,
    registry: Any,
    *,
    allow_mutable_tags: bool = False,
) -> "TypeCheckResult":
    """Verify type compatibility across imported-component edges in program.graph.

    By default mutable tags (e.g. @latest) are rejected — pass
    allow_mutable_tags=True to allow them (equivalent to --allow-mutable-imports
    on the CLI).
    """
    import re
    _PINNED = re.compile(r"^v\d+(\.\d+)*$")

    errors: list[str] = []
    warnings: list[str] = []

    entries: dict[str, Any] = {}
    for imp in getattr(program, "imports", []):
        # Enforce mutable-tag policy before touching the registry.
        if not _PINNED.match(imp.version):
            if not allow_mutable_tags:
                errors.append(
                    f"Import {imp.alias!r}: mutable tag {imp.version!r} rejected; "
                    "use --allow-mutable-imports to permit mutable version references"
                )
                continue
            else:
                warnings.append(
                    f"Import {imp.alias!r}: mutable tag {imp.version!r} — "
                    "resolution may change between runs"
                )
        try:
            entries[imp.alias] = registry.get(imp.registry_name, imp.version)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Cannot resolve {imp.alias!r} for type checking: {exc}")

    for src, dst in program.graph.edges:
        src_out = _node_output_type(src, program, entries)
        dst_in = _node_input_type(dst, entries)
        if src_out is None or dst_in is None:
            continue
        if not dst_in:
            continue
        if not _composite_types_compatible(src_out, dst_in):
            errors.append(
                f"Type mismatch on edge {src!r} → {dst!r}: "
                f"output {src_out} incompatible with input {dst_in}"
            )

    if errors:
        raise TypeCompatibilityError(
            f"{len(errors)} type compatibility error(s) in composite program", errors
        )
    return TypeCheckResult(errors=[], warnings=warnings)
