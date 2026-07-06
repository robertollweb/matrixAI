# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import import_module, util
from pathlib import Path
from typing import Any

from matrixai.ir import MatrixAIProgram, VectorSpec, FunctionSpec
from matrixai.parameters import validate_parameter_set
from matrixai.parameters.store import ParameterSet, program_hash


class OnnxExportError(ValueError):
    pass


_SUPPORTED_KINDS = frozenset({"softmax_linear", "sigmoid_linear", "layer_call"})
_OPSET_VERSION = 17


def onnx_size_limit_error(program: Any) -> str | None:
    """PESOS_GRANDES C6 — mensaje de error si los pesos estimados del programa
    NO caben en un fichero ONNX (protobuf rechaza serializar mensajes >2 GiB;
    ONNX guarda los pesos EN LÍNEA). `None` si caben — o si el propio
    estimador falla (fail-open, invariante 6: la estimación es orientativa y
    nunca debe convertir un export válido en un error por un fallo SUYO; sin
    el chequeo, un modelo realmente grande fallará igualmente más adelante,
    solo que con el error críptico de protobuf).

    Compartido por `OnnxExporter.export` (lanza `OnnxExportError`) y por el
    Studio (`_studio_export` lo consulta ANTES de materializar los pesos — un
    modelo de varios GiB no debe pagar minutos de `.tolist()` para morir
    aquí igualmente). El mensaje NO sugiere bundle/wasm como alternativa:
    ambos empaquetan un ONNX interno (`bundle.py`/`wasm_exporter.py` llaman a
    `OnnxExporter().export`), así que comparten exactamente este límite.
    """
    try:
        from matrixai.resources import estimate_model_resources, ONNX_PROTOBUF_LIMIT_GIB
        estimate = estimate_model_resources(program)
    except Exception:  # noqa: BLE001 — fail-open: el guardrail nunca rompe un export válido
        return None
    if estimate.weights_gib <= ONNX_PROTOBUF_LIMIT_GIB:
        return None
    return (
        f"Este modelo tiene ~{estimate.weights_gib:.2f} GiB de pesos "
        f"({estimate.param_count:,} parámetros) — supera el límite de "
        f"~{ONNX_PROTOBUF_LIMIT_GIB:.1f} GiB de un fichero ONNX (el formato "
        "guarda los pesos en línea en el protobuf, y por encima de ese tamaño "
        "necesitaría 'external data', fuera de alcance hoy). Los formatos "
        "wasm y bundle empaquetan un ONNX interno, así que comparten este "
        "límite. El modelo sigue siendo usable dentro del Studio: guárdalo "
        "(binario o json) para inferir y reentrenar con él."
    )


@dataclass(frozen=True)
class OnnxExportResult:
    output_path: str
    opset_version: int
    model_hash: str
    parameter_set_id: str
    parameter_schema_hash: str
    input_name: str
    input_shape: list[int]
    output_name: str
    output_shape: list[int]
    exported_functions: list[str]
    skipped_functions: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "opset_version": self.opset_version,
            "model_hash": self.model_hash,
            "parameter_set_id": self.parameter_set_id,
            "parameter_schema_hash": self.parameter_schema_hash,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "output_name": self.output_name,
            "output_shape": self.output_shape,
            "exported_functions": self.exported_functions,
            "skipped_functions": self.skipped_functions,
            "labels": self.labels,
        }


class OnnxExporter:
    def export(
        self,
        program: MatrixAIProgram,
        parameter_set: ParameterSet,
        output_path: str | Path,
    ) -> OnnxExportResult:
        onnx, numpy_helper, helper, TensorProto = _import_onnx()
        np = _import_numpy()

        output_path = Path(output_path)

        # PESOS_GRANDES C6: ONNX guarda los pesos EN LÍNEA en el protobuf — por
        # encima de ~2 GiB, protobuf rechaza serializar el mensaje (falla a
        # mitad de export con un ValueError críptico, o peor, produce un
        # fichero incompleto). Frenar AQUÍ, antes de construir el grafo, da un
        # error claro y accionable. El chequeo (estimación desde el manifest,
        # O(#tensores), sin materializar valores) vive en
        # `onnx_size_limit_error` — compartido con el Studio, que lo consulta
        # aún antes (sin pagar la materialización de los pesos).
        size_error = onnx_size_limit_error(program)
        if size_error is not None:
            raise OnnxExportError(size_error)

        # Guardrail: ParameterSet must belong to this program (hash)
        expected_hash = program_hash(program)
        if parameter_set.model_hash != expected_hash:
            raise OnnxExportError(
                f"ParameterSet model_hash {parameter_set.model_hash!r} does not match "
                f"program hash {expected_hash!r} for {program.project!r}. "
                "Export refused: ParameterSet was not trained on this .mxai."
            )

        # Guardrail: ParameterSet shapes and schema must be consistent.
        val = validate_export_parameter_set(program, parameter_set)
        if not val.ok:
            raise OnnxExportError(
                f"ParameterSet validation failed for {program.project!r}: "
                f"{'; '.join(val.errors)}"
            )

        # Classify functions by kind
        layer_call_fns = [f for f in program.functions if f.semantic.kind == "layer_call"]
        simple_fns = [f for f in program.functions
                      if f.semantic.kind in ("softmax_linear", "sigmoid_linear")]
        skipped = [f.name for f in program.functions if f.semantic.kind not in _SUPPORTED_KINDS]
        dense_nets = [n for n in program.networks if getattr(n, "kind", "") == "dense_network"]
        composite_nets = [n for n in program.networks if getattr(n, "kind", "") == "composite_network"]

        if layer_call_fns:
            # Input size for result: from VECTOR or SEQUENCE spec
            if program.vectors:
                input_dim = program.vectors[0].size
            elif program.sequences:
                input_dim = program.sequences[0].length
            else:
                raise OnnxExportError(
                    f"No VECTOR or SEQUENCE input for layer_call in {program.project!r}"
                )
            nodes, initializers, x_info, y_info, labels, out_shape = _build_layer_call_pipeline(
                layer_call_fns, program, parameter_set, np, numpy_helper, helper, TensorProto
            )
            exported_names = [f.name for f in layer_call_fns]
            kind = "layer_call"
        elif simple_fns:
            fn = simple_fns[0]
            vector = _find_vector(fn, program)
            if vector is None:
                raise OnnxExportError(
                    f"FUNCTION {fn.name} has no VECTOR input resolvable in program {program.project!r}"
                )
            input_dim = vector.size
            kind = fn.semantic.kind
            if kind == "softmax_linear":
                nodes, initializers, x_info, y_info, labels, out_shape = _build_softmax_linear(
                    fn, vector, parameter_set, np, numpy_helper, helper, TensorProto
                )
            else:
                nodes, initializers, x_info, y_info, labels, out_shape = _build_sigmoid_linear(
                    fn, vector, parameter_set, np, numpy_helper, helper, TensorProto
                )
            exported_names = [fn.name]
        elif dense_nets:
            # DenseNetworkGenerator produces NETWORK blocks with no FUNCTION declarations.
            # Build the ONNX graph directly from DenseLayerSpec + Gemm nodes.
            network = dense_nets[0]
            if not program.vectors:
                raise OnnxExportError(f"No VECTOR input for dense network {network.name!r}")
            input_dim = program.vectors[0].size
            nodes, initializers, x_info, y_info, out_shape = _build_dense_network_pipeline(
                network, program, parameter_set, np, numpy_helper, helper, TensorProto
            )
            exported_names = [network.name]
            labels = []
            kind = "dense_network"
            skipped = [n.name for n in dense_nets[1:]]
        elif composite_nets:
            # M2 v2 — composite_network (P19 blocks/residual/LayerNorm/Dropout/embeddings)
            # produced by CompositeNetworkGenerator. Lower the composite forward to ONNX.
            network = composite_nets[0]
            if not program.vectors:
                raise OnnxExportError(f"No VECTOR input for composite network {network.name!r}")
            input_dim = program.vectors[0].size
            nodes, initializers, x_info, y_info, out_shape = _build_composite_network_pipeline(
                network, program, parameter_set, np, numpy_helper, helper, TensorProto
            )
            exported_names = [network.name]
            labels = []
            kind = "composite_network"
            skipped = [n.name for n in composite_nets[1:]]
        else:
            kinds = {f.semantic.kind for f in program.functions}
            raise OnnxExportError(
                f"No exportable functions found in {program.project!r}. "
                f"Supported: {sorted(_SUPPORTED_KINDS)}. Found kinds: {sorted(kinds)}"
            )

        graph = helper.make_graph(
            nodes,
            name=f"{program.project}_graph",
            inputs=[x_info],
            outputs=[y_info],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
        model.doc_string = (
            f"MatrixAI export of {program.project!r} "
            f"({', '.join(exported_names)}, {kind})"
        )
        model.ir_version = 10

        # Embed MatrixAI metadata as model properties (all hashes must be validatable)
        _set_meta(model, "matrixai_project", program.project)
        _set_meta(model, "matrixai_model_hash", parameter_set.model_hash)
        _set_meta(model, "matrixai_parameter_set_id", parameter_set.parameter_set_id)
        _set_meta(model, "matrixai_parameter_schema_hash", parameter_set.parameter_schema_hash)
        _set_meta(model, "matrixai_kind", kind)
        if labels:
            _set_meta(model, "matrixai_labels", ",".join(labels))

        onnx.checker.check_model(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_path))

        in_shape = [-1, input_dim]
        return OnnxExportResult(
            output_path=str(output_path),
            opset_version=_OPSET_VERSION,
            model_hash=parameter_set.model_hash,
            parameter_set_id=parameter_set.parameter_set_id,
            parameter_schema_hash=parameter_set.parameter_schema_hash,
            input_name=x_info.name,
            input_shape=in_shape,
            output_name=y_info.name,
            output_shape=out_shape,
            exported_functions=exported_names,
            skipped_functions=skipped,
            labels=labels,
        )


def validate_export_parameter_set(program, parameter_set):
    """Validate a ParameterSet for export, dispatching composite networks (P19) to
    their dedicated validator. The generic BackendContractAnalyzer-based validator
    only knows dense/function programs and would reject composite parameter schemas."""
    composite = [n for n in program.networks if getattr(n, "kind", "") == "composite_network"]
    if composite:
        from matrixai.types import check_composite_network_types
        from matrixai.parameters.network_params import validate_composite_network_parameter_set
        net = composite[0]
        vector_map = {v.name: v for v in program.vectors}
        type_result = check_composite_network_types(net, vector_map)
        return validate_composite_network_parameter_set(
            net, type_result, parameter_set, program_hash(program)
        )
    return validate_parameter_set(program, parameter_set)


def export_onnx(
    program: MatrixAIProgram,
    parameter_set: ParameterSet,
    output_path: str | Path,
) -> OnnxExportResult:
    return OnnxExporter().export(program, parameter_set, output_path)


def onnx_available() -> bool:
    return util.find_spec("onnx") is not None


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _build_softmax_linear(fn, vector, parameter_set, np, numpy_helper, helper, TensorProto):
    """Gemm(X, W1, b1, transB=1) → Softmax → probs [N, n_classes]."""
    w1_param = fn.semantic.parameters.get("weights", "W1")
    b1_param = fn.semantic.parameters.get("bias", "b1")

    w1_values = np.array(
        _get_param_values(parameter_set, w1_param, fn.name), dtype=np.float32
    )
    b1_values = np.array(
        _get_param_values(parameter_set, b1_param, fn.name), dtype=np.float32
    )

    if w1_values.ndim != 2:
        raise OnnxExportError(
            f"softmax_linear {fn.name}: W1 must be 2-D [n_classes, input_dim], "
            f"got shape {list(w1_values.shape)}"
        )
    n_classes, input_dim = w1_values.shape

    labels = [str(lbl) for lbl in (fn.semantic.parameters.get("labels") or [])]
    if not labels:
        labels = [str(i) for i in range(n_classes)]

    w1_init = numpy_helper.from_array(w1_values, name="W1")
    b1_init = numpy_helper.from_array(b1_values, name="b1")

    gemm = helper.make_node(
        "Gemm",
        inputs=[vector.name, "W1", "b1"],
        outputs=["logits"],
        transB=1,
        alpha=1.0,
        beta=1.0,
    )
    softmax = helper.make_node("Softmax", inputs=["logits"], outputs=["probabilities"], axis=1)

    x_info = helper.make_tensor_value_info(vector.name, TensorProto.FLOAT, [-1, input_dim])
    y_info = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [-1, n_classes])

    return [gemm, softmax], [w1_init, b1_init], x_info, y_info, labels, [-1, n_classes]


def _build_sigmoid_linear(fn, vector, parameter_set, np, numpy_helper, helper, TensorProto):
    """MatMul(X, W1_col) + b1 → Sigmoid → score [N]."""
    w1_param = fn.semantic.parameters.get("weights", "W1")
    b1_param = fn.semantic.parameters.get("bias", "b1")

    w1_values = np.array(
        _get_param_values(parameter_set, w1_param, fn.name), dtype=np.float32
    ).reshape(-1)  # ensure 1-D
    b1_raw = _get_param_values(parameter_set, b1_param, fn.name)
    b1_scalar = float(b1_raw) if not isinstance(b1_raw, list) else float(b1_raw[0])

    input_dim = w1_values.shape[0]

    # W1 reshaped to column [input_dim, 1] for MatMul → [N, 1]
    w1_col = w1_values.reshape(input_dim, 1)
    b1_arr = np.array([b1_scalar], dtype=np.float32)

    w1_init = numpy_helper.from_array(w1_col, name="W1_col")
    b1_init = numpy_helper.from_array(b1_arr, name="b1_val")
    squeeze_axes_init = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="squeeze_axes"
    )

    matmul = helper.make_node("MatMul", inputs=[vector.name, "W1_col"], outputs=["raw_2d"])
    add = helper.make_node("Add", inputs=["raw_2d", "b1_val"], outputs=["logit_2d"])
    sigmoid = helper.make_node("Sigmoid", inputs=["logit_2d"], outputs=["score_2d"])
    squeeze = helper.make_node("Squeeze", inputs=["score_2d", "squeeze_axes"], outputs=["probability"])

    x_info = helper.make_tensor_value_info(vector.name, TensorProto.FLOAT, [-1, input_dim])
    y_info = helper.make_tensor_value_info("probability", TensorProto.FLOAT, [-1])

    labels: list[str] = []
    return (
        [matmul, add, sigmoid, squeeze],
        [w1_init, b1_init, squeeze_axes_init],
        x_info, y_info, labels, [-1],
    )


# ---------------------------------------------------------------------------
# Layer-call pipeline builders
# ---------------------------------------------------------------------------

def _build_layer_call_pipeline(layer_call_fns, program, parameter_set, np, numpy_helper, helper, TensorProto):
    """Build ONNX nodes for a sequential pipeline of layer_call functions."""
    layers = {layer.name: layer for layer in program.layers}

    # Determine input: VECTOR (float32) or SEQUENCE (int64)
    if program.vectors:
        inp_spec = program.vectors[0]
        current_input = inp_spec.name
        current_shape: list = [-1, inp_spec.size]
        x_info = helper.make_tensor_value_info(inp_spec.name, TensorProto.FLOAT, current_shape)
    elif program.sequences:
        inp_spec = program.sequences[0]
        current_input = inp_spec.name
        current_shape = [-1, inp_spec.length]
        x_info = helper.make_tensor_value_info(inp_spec.name, TensorProto.INT64, current_shape)
    else:
        raise OnnxExportError("No VECTOR or SEQUENCE input found in program")

    all_nodes: list = []
    all_inits: list = []

    for fn in layer_call_fns:
        layer_name = fn.semantic.parameters["layer"]
        layer = layers[layer_name]
        nodes, inits, output_tensor, output_shape = _build_layer_nodes(
            layer, layer_name, current_input, current_shape,
            parameter_set, np, numpy_helper, helper, TensorProto,
        )
        all_nodes.extend(nodes)
        all_inits.extend(inits)
        current_input = output_tensor
        current_shape = output_shape

    y_info = helper.make_tensor_value_info(current_input, TensorProto.FLOAT, current_shape)
    return all_nodes, all_inits, x_info, y_info, [], current_shape


def _build_layer_nodes(layer, layer_name, input_tensor, input_shape, parameter_set, np, numpy_helper, helper, TensorProto):
    """Build ONNX nodes for one layer. Returns (nodes, initializers, output_tensor, output_shape)."""
    nodes: list = []
    initializers: list = []

    # Map layer-local variable names → ONNX tensor names
    local_to_onnx: dict[str, str] = {"input": input_tensor}
    # Track shapes for primitives that need them (e.g., attention dim)
    shapes: dict[str, list] = {"input": input_shape}

    # Load parameter initializers
    for param_spec in layer.params:
        pname = param_spec.name
        onnx_name = f"{layer_name}.{pname}"
        vals = parameter_set.parameters[f"{layer_name}.{pname}"]["values"]
        arr = np.array(vals, dtype=np.float32)
        initializers.append(numpy_helper.from_array(arr, name=onnx_name))
        local_to_onnx[pname] = onnx_name
        shapes[pname] = list(arr.shape)

    def resolve(var: str) -> str:
        if var in local_to_onnx:
            return local_to_onnx[var]
        raise OnnxExportError(f"Variable {var!r} not defined in layer {layer_name!r}")

    for op in layer.body_ops:
        out_onnx = f"{layer_name}.{op.output}"
        local_to_onnx[op.output] = out_onnx

        if op.kind == "matmul":
            a, b = op.args
            nodes.append(helper.make_node("MatMul", inputs=[resolve(a), resolve(b)], outputs=[out_onnx]))
            b_shape = shapes.get(b, [])
            shapes[op.output] = [-1, b_shape[-1]] if len(b_shape) >= 2 else [-1]

        elif op.kind == "residual":
            a, b = op.args
            nodes.append(helper.make_node("Add", inputs=[resolve(a), resolve(b)], outputs=[out_onnx]))
            shapes[op.output] = shapes.get(a, [-1])

        elif op.kind == "layer_norm":
            x, gain, bias_var = op.args
            nodes.append(helper.make_node(
                "LayerNormalization",
                inputs=[resolve(x), resolve(gain), resolve(bias_var)],
                outputs=[out_onnx],
                axis=-1,
                epsilon=1e-5,
            ))
            shapes[op.output] = shapes.get(x, [-1])

        elif op.kind == "gelu":
            (x,) = op.args
            gelu_nodes, gelu_inits = _add_gelu_nodes(resolve(x), out_onnx, layer_name, helper, numpy_helper, np)
            nodes.extend(gelu_nodes)
            initializers.extend(gelu_inits)
            shapes[op.output] = shapes.get(x, [-1])

        elif op.kind == "attention":
            q, k, v = op.args
            q_shape = shapes.get(q, [-1, 8])
            d = q_shape[-1] if isinstance(q_shape[-1], int) and q_shape[-1] > 0 else 8
            attn_nodes, attn_inits = _add_attention_nodes(
                resolve(q), resolve(k), resolve(v), out_onnx, layer_name, d,
                helper, numpy_helper, np,
            )
            nodes.extend(attn_nodes)
            initializers.extend(attn_inits)
            shapes[op.output] = shapes.get(v, [-1])

        elif op.kind == "embedding_lookup":
            table, ids = op.args
            nodes.append(helper.make_node("Gather", inputs=[resolve(table), resolve(ids)], outputs=[out_onnx], axis=0))
            table_shape = shapes.get(table, [])
            ids_shape = shapes.get(ids, [-1])
            seq_len = ids_shape[-1] if len(ids_shape) >= 2 else -1
            embed_dim = table_shape[-1] if table_shape else -1
            shapes[op.output] = [-1, seq_len, embed_dim]

        elif op.kind == "mean_pooling":
            (x,) = op.args
            # ReduceMean over axis 1 (seq_len), opset 17 uses attribute
            nodes.append(helper.make_node("ReduceMean", inputs=[resolve(x)], outputs=[out_onnx], axes=[1], keepdims=0))
            x_shape = shapes.get(x, [-1, -1, -1])
            embed_dim = x_shape[-1] if len(x_shape) >= 3 else -1
            shapes[op.output] = [-1, embed_dim]

        elif op.kind == "dot":
            a, b = op.args
            qk_name = f"{layer_name}.{op.output}_qk"
            axes_name = f"{layer_name}.{op.output}_axes"
            initializers.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name))
            nodes.append(helper.make_node("Mul", inputs=[resolve(a), resolve(b)], outputs=[qk_name]))
            nodes.append(helper.make_node("ReduceSum", inputs=[qk_name, axes_name], outputs=[out_onnx], keepdims=1))
            shapes[op.output] = [-1, 1]

        elif op.kind == "scale":
            x, factor = op.args
            if factor in local_to_onnx:
                nodes.append(helper.make_node("Mul", inputs=[resolve(x), resolve(factor)], outputs=[out_onnx]))
            else:
                const_name = f"{layer_name}.{op.output}_scale_c"
                initializers.append(numpy_helper.from_array(np.array(float(factor), dtype=np.float32), name=const_name))
                nodes.append(helper.make_node("Mul", inputs=[resolve(x), const_name], outputs=[out_onnx]))
            shapes[op.output] = shapes.get(x, [-1])

        elif op.kind == "softmax":
            (x,) = op.args
            nodes.append(helper.make_node("Softmax", inputs=[resolve(x)], outputs=[out_onnx], axis=-1))
            shapes[op.output] = shapes.get(x, [-1])

        else:
            raise OnnxExportError(
                f"Unsupported primitive {op.kind!r} in layer {layer_name!r}. "
                f"Supported: matmul, residual, layer_norm, gelu, attention, "
                f"embedding_lookup, mean_pooling, dot, scale, softmax"
            )

    output_tensor = f"{layer_name}.result"
    output_shape = shapes.get("result", [-1])
    return nodes, initializers, output_tensor, output_shape


def _build_dense_network_pipeline(network, program, parameter_set, np, numpy_helper, helper, TensorProto):
    """Build ONNX nodes for a NETWORK block produced by DenseNetworkGenerator.

    Parameters are stored as {network.name}.W{i} / {network.name}.b{i} (1-based layer index).
    Each Dense layer becomes: Gemm(X, W, b, transB=1) → activation.
    """
    vec = program.vectors[0]
    net = network.name
    nodes = []
    initializers = []
    current = vec.name
    current_shape = [-1, vec.size]

    x_info = helper.make_tensor_value_info(vec.name, TensorProto.FLOAT, current_shape)

    for layer in network.layers:
        i = layer.index  # 1-based
        w_key = f"{net}.W{i}"
        b_key = f"{net}.b{i}"
        if w_key not in parameter_set.parameters:
            raise OnnxExportError(f"Dense-network parameter {w_key!r} not found in ParameterSet")
        if b_key not in parameter_set.parameters:
            raise OnnxExportError(f"Dense-network parameter {b_key!r} not found in ParameterSet")

        W = np.array(parameter_set.parameters[w_key]["values"], dtype=np.float32)  # (units_out, units_in)
        b = np.array(parameter_set.parameters[b_key]["values"], dtype=np.float32)  # (units_out,)

        w_name = f"{net}_W{i}"
        b_name = f"{net}_b{i}"
        pre_act = f"{net}_pre{i}"
        post_act = f"{net}_out{i}"

        initializers.append(numpy_helper.from_array(W, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))

        # Gemm: Y = alpha * X * W^T + beta * b  (transB=1 transposes W from units_out×units_in)
        nodes.append(helper.make_node(
            "Gemm",
            inputs=[current, w_name, b_name],
            outputs=[pre_act],
            transB=1,
            alpha=1.0,
            beta=1.0,
        ))

        act = layer.activation.lower()
        if act == "relu":
            nodes.append(helper.make_node("Relu", inputs=[pre_act], outputs=[post_act]))
        elif act == "sigmoid":
            nodes.append(helper.make_node("Sigmoid", inputs=[pre_act], outputs=[post_act]))
        elif act == "tanh":
            nodes.append(helper.make_node("Tanh", inputs=[pre_act], outputs=[post_act]))
        elif act == "softmax":
            nodes.append(helper.make_node("Softmax", inputs=[pre_act], outputs=[post_act], axis=1))
        else:  # linear / identity
            nodes.append(helper.make_node("Identity", inputs=[pre_act], outputs=[post_act]))

        current = post_act
        current_shape = [-1, layer.units]

    y_info = helper.make_tensor_value_info(current, TensorProto.FLOAT, current_shape)
    return nodes, initializers, x_info, y_info, current_shape


def _emit_dense_activation(act, pre_act, post_act, helper):
    """Emit the ONNX activation node for a Dense/Activation layer. Softmax over the
    last axis (composite tensors are [batch, dim], so axis=1)."""
    act = (act or "linear").lower()
    if act == "relu":
        return helper.make_node("Relu", inputs=[pre_act], outputs=[post_act])
    if act == "sigmoid":
        return helper.make_node("Sigmoid", inputs=[pre_act], outputs=[post_act])
    if act == "tanh":
        return helper.make_node("Tanh", inputs=[pre_act], outputs=[post_act])
    if act == "softmax":
        return helper.make_node("Softmax", inputs=[pre_act], outputs=[post_act], axis=1)
    # linear / identity
    return helper.make_node("Identity", inputs=[pre_act], outputs=[post_act])


def _build_composite_layer_onnx_nodes(layer, prefix, tag, current, current_dim,
                                      parameter_set, np, numpy_helper, helper):
    """Lower one CompositeLayerSpec to ONNX nodes. Mirrors composite_forward's
    _forward_composite_layer (inference: Dropout/Pool/Reshape are identity).

    Returns (nodes, initializers, output_tensor, output_dim).
    """
    nodes: list = []
    initializers: list = []
    lt = layer.layer_type
    pfx = f"{prefix}.L{layer.index}"          # parameter-store key prefix
    name = f"{tag}_L{layer.index}"            # ONNX tensor-name prefix (dots are awkward)

    if lt == "Dense":
        w_key = f"{pfx}.W"
        b_key = f"{pfx}.b"
        if w_key not in parameter_set.parameters:
            raise OnnxExportError(f"Composite parameter {w_key!r} not found in ParameterSet")
        if b_key not in parameter_set.parameters:
            raise OnnxExportError(f"Composite parameter {b_key!r} not found in ParameterSet")
        W = np.array(parameter_set.parameters[w_key]["values"], dtype=np.float32)  # (out, in)
        b = np.array(parameter_set.parameters[b_key]["values"], dtype=np.float32)  # (out,)
        w_name, b_name = f"{name}_W", f"{name}_b"
        pre_act, post_act = f"{name}_pre", f"{name}_out"
        initializers.append(numpy_helper.from_array(W, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))
        nodes.append(helper.make_node(
            "Gemm", inputs=[current, w_name, b_name], outputs=[pre_act],
            transB=1, alpha=1.0, beta=1.0,
        ))
        nodes.append(_emit_dense_activation(layer.activation, pre_act, post_act, helper))
        return nodes, initializers, post_act, int(W.shape[0])

    if lt == "LayerNorm":
        gamma_key = f"{pfx}.gamma"
        beta_key = f"{pfx}.beta"
        if gamma_key not in parameter_set.parameters:
            raise OnnxExportError(f"Composite parameter {gamma_key!r} not found in ParameterSet")
        if beta_key not in parameter_set.parameters:
            raise OnnxExportError(f"Composite parameter {beta_key!r} not found in ParameterSet")
        gamma = np.array(parameter_set.parameters[gamma_key]["values"], dtype=np.float32)
        beta = np.array(parameter_set.parameters[beta_key]["values"], dtype=np.float32)
        g_name, bt_name, out = f"{name}_gamma", f"{name}_beta", f"{name}_ln"
        initializers.append(numpy_helper.from_array(gamma, name=g_name))
        initializers.append(numpy_helper.from_array(beta, name=bt_name))
        nodes.append(helper.make_node(
            "LayerNormalization", inputs=[current, g_name, bt_name], outputs=[out],
            axis=-1, epsilon=1e-5,
        ))
        return nodes, initializers, out, current_dim

    if lt == "Activation":
        out = f"{name}_act"
        nodes.append(_emit_dense_activation(getattr(layer, "activation_kind", "relu"),
                                            current, out, helper))
        return nodes, initializers, out, current_dim

    if lt in ("Dropout", "Pool", "Reshape"):
        # Inference-time identity (composite_forward training=False): pass the tensor
        # through with no node, exactly like the stdlib forward.
        return [], [], current, current_dim

    raise OnnxExportError(
        f"Unsupported composite layer type {lt!r} in {prefix!r}. "
        f"Supported: Dense, LayerNorm, Dropout, Activation, Pool, Reshape"
    )


def _build_composite_network_pipeline(network, program, parameter_set, np, numpy_helper, helper, TensorProto):
    """Build ONNX nodes for a composite_network (P19) produced by CompositeNetworkGenerator.

    Mirrors composite_forward: VECTOR input → embeddings (Gather) → concats (Concat) →
    interleaved top_layers/blocks (residual connections via Add).
    """
    from matrixai.ir.schema import get_interleaved_body

    vec = program.vectors[0]
    nodes: list = []
    initializers: list = []
    x_info = helper.make_tensor_value_info(vec.name, TensorProto.FLOAT, [-1, vec.size])
    tag = network.name

    # named tensors: name -> (onnx_tensor_name, dim). Mirrors composite_forward.
    named: dict[str, tuple[str, int]] = {}
    field_index = {f: i for i, f in enumerate(vec.fields)}

    def _field_column(name: str) -> tuple[str, int]:
        """Slice one input field as a [batch, 1] column (cached)."""
        if name in named:
            return named[name]
        if name not in field_index:
            raise OnnxExportError(f"Composite network {tag!r}: field {name!r} not in VECTOR")
        idx_init = f"{tag}_idx_{name}"
        initializers.append(numpy_helper.from_array(np.array([field_index[name]], dtype=np.int64), name=idx_init))
        out = f"{tag}_col_{name}"
        nodes.append(helper.make_node("Gather", inputs=[vec.name, idx_init], outputs=[out], axis=1))
        named[name] = (out, 1)
        return named[name]

    # 1. Embeddings: round the source field index, Cast to int64, Gather a table row.
    for emb in getattr(network, "embeddings", []):
        table_key = f"{tag}.{emb.name}.table"
        if table_key not in parameter_set.parameters:
            raise OnnxExportError(f"Composite parameter {table_key!r} not found in ParameterSet")
        table = np.array(parameter_set.parameters[table_key]["values"], dtype=np.float32)  # (vocab, dim)
        dim = int(table.shape[1])
        col, _ = _field_column(emb.source)
        rounded = f"{tag}_{emb.name}_round"
        idx_i64 = f"{tag}_{emb.name}_i64"
        idx_flat = f"{tag}_{emb.name}_flat"
        table_name = f"{tag}_{emb.name}_table"
        emb_out = f"{tag}_{emb.name}_emb"
        flat_shape = f"{tag}_{emb.name}_flatshape"
        initializers.append(numpy_helper.from_array(table, name=table_name))
        initializers.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), name=flat_shape))
        # int(round(x)) — ONNX Round is round-half-to-even, matching Python round().
        nodes.append(helper.make_node("Round", inputs=[col], outputs=[rounded]))
        nodes.append(helper.make_node("Cast", inputs=[rounded], outputs=[idx_i64], to=TensorProto.INT64))
        nodes.append(helper.make_node("Reshape", inputs=[idx_i64, flat_shape], outputs=[idx_flat]))
        nodes.append(helper.make_node("Gather", inputs=[table_name, idx_flat], outputs=[emb_out], axis=0))
        named[emb.name] = (emb_out, dim)

    # 2. Concats: combine named tensors along the feature axis.
    for concat in getattr(network, "concats", []):
        parts = [_field_column(s) if s not in named else named[s] for s in concat.sources]
        concat_out = f"{tag}_concat_{concat.name}"
        nodes.append(helper.make_node(
            "Concat", inputs=[t for t, _ in parts], outputs=[concat_out], axis=1))
        named[concat.name] = (concat_out, sum(d for _, d in parts))

    # 3. Initial current vector (mirrors composite_forward step 4).
    concats = getattr(network, "concats", [])
    if concats:
        current, current_dim = named[concats[-1].name]
    else:
        # No concats: the flat input VECTOR is the feature vector (embeddings, if any,
        # would be unused — the generator never emits that shape).
        current = vec.name
        current_dim = vec.size

    for _, kind, spec in get_interleaved_body(network):
        if kind == "layer":
            n, inits, current, current_dim = _build_composite_layer_onnx_nodes(
                spec, network.name, network.name, current, current_dim,
                parameter_set, np, numpy_helper, helper,
            )
            nodes.extend(n)
            initializers.extend(inits)
        else:
            block = spec
            residual_from = getattr(block, "residual_from", "")
            block_input = current
            block_dim = current_dim
            block_prefix = f"{network.name}.{block.name}"
            block_tag = f"{network.name}_{block.name}"
            for layer in block.layers:
                n, inits, current, current_dim = _build_composite_layer_onnx_nodes(
                    layer, block_prefix, block_tag, current, current_dim,
                    parameter_set, np, numpy_helper, helper,
                )
                nodes.extend(n)
                initializers.extend(inits)
            if residual_from:
                if residual_from == "PREVIOUS":
                    res_tensor, res_dim = block_input, block_dim
                elif residual_from in named:
                    res_tensor, res_dim = named[residual_from]
                else:
                    raise OnnxExportError(
                        f"Block {block.name!r}: RESIDUAL FROM {residual_from!r} not a known "
                        f"tensor (expected PREVIOUS or a named embedding/concat/field)"
                    )
                if res_dim != current_dim:
                    raise OnnxExportError(
                        f"Block {block.name!r}: RESIDUAL shape mismatch "
                        f"(residual={res_dim}, block_output={current_dim})"
                    )
                res_out = f"{block_tag}_residual"
                nodes.append(helper.make_node(
                    "Add", inputs=[res_tensor, current], outputs=[res_out]))
                current = res_out

    out_shape = [-1, current_dim]
    y_info = helper.make_tensor_value_info(current, TensorProto.FLOAT, out_shape)
    return nodes, initializers, x_info, y_info, out_shape


def _add_gelu_nodes(x_tensor, output_tensor, layer_name, helper, numpy_helper, np):
    """GELU tanh approx: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))"""
    pfx = f"{layer_name}._gelu_"
    coeff_name = f"{pfx}coeff"
    factor_name = f"{pfx}factor"
    half_name = f"{pfx}half"
    one_name = f"{pfx}one"
    x_sq = f"{pfx}x_sq"
    x_cu = f"{pfx}x_cu"
    scaled_cu = f"{pfx}scaled_cu"
    inner = f"{pfx}inner"
    pre_tanh = f"{pfx}pre_tanh"
    tanh_out = f"{pfx}tanh_out"
    one_plus = f"{pfx}one_plus"
    half_x = f"{pfx}half_x"

    initializers = [
        numpy_helper.from_array(np.array(0.044715, dtype=np.float32), name=coeff_name),
        numpy_helper.from_array(np.array(math.sqrt(2.0 / math.pi), dtype=np.float32), name=factor_name),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name=half_name),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name),
    ]
    nodes = [
        helper.make_node("Mul", inputs=[x_tensor, x_tensor], outputs=[x_sq]),
        helper.make_node("Mul", inputs=[x_sq, x_tensor], outputs=[x_cu]),
        helper.make_node("Mul", inputs=[x_cu, coeff_name], outputs=[scaled_cu]),
        helper.make_node("Add", inputs=[x_tensor, scaled_cu], outputs=[inner]),
        helper.make_node("Mul", inputs=[inner, factor_name], outputs=[pre_tanh]),
        helper.make_node("Tanh", inputs=[pre_tanh], outputs=[tanh_out]),
        helper.make_node("Add", inputs=[one_name, tanh_out], outputs=[one_plus]),
        helper.make_node("Mul", inputs=[x_tensor, half_name], outputs=[half_x]),
        helper.make_node("Mul", inputs=[half_x, one_plus], outputs=[output_tensor]),
    ]
    return nodes, initializers


def _add_attention_nodes(q_tensor, k_tensor, v_tensor, output_tensor, layer_name, d, helper, numpy_helper, np):
    """Sigmoid dot-product attention: weight=sigmoid(dot(q,k)/sqrt(d)), output=weight*v"""
    pfx = f"{layer_name}._attn_"
    scale_name = f"{pfx}scale"
    axes_name = f"{pfx}axes"
    qk_elem = f"{pfx}qk_elem"
    score_sum = f"{pfx}score_sum"
    score_scaled = f"{pfx}score_scaled"
    weight = f"{pfx}weight"

    scale_val = float(1.0 / math.sqrt(max(d, 1)))
    initializers = [
        numpy_helper.from_array(np.array(scale_val, dtype=np.float32), name=scale_name),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name),
    ]
    nodes = [
        helper.make_node("Mul", inputs=[q_tensor, k_tensor], outputs=[qk_elem]),
        helper.make_node("ReduceSum", inputs=[qk_elem, axes_name], outputs=[score_sum], keepdims=1),
        helper.make_node("Mul", inputs=[score_sum, scale_name], outputs=[score_scaled]),
        helper.make_node("Sigmoid", inputs=[score_scaled], outputs=[weight]),
        helper.make_node("Mul", inputs=[weight, v_tensor], outputs=[output_tensor]),
    ]
    return nodes, initializers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_vector(fn: FunctionSpec, program: MatrixAIProgram) -> VectorSpec | None:
    vectors = {v.name: v for v in program.vectors}
    for name in fn.semantic.inputs:
        if name in vectors:
            return vectors[name]
    # Fallback: first vector in program
    return program.vectors[0] if program.vectors else None


def _get_param_values(parameter_set: ParameterSet, param_name: str, fn_name: str) -> Any:
    """Lookup parameter values by bare name or qualified fn.name."""
    if param_name in parameter_set.parameters:
        return parameter_set.parameters[param_name]["values"]
    qualified = f"{fn_name}.{param_name}"
    if qualified in parameter_set.parameters:
        return parameter_set.parameters[qualified]["values"]
    raise OnnxExportError(
        f"Parameter {param_name!r} (or {qualified!r}) not found in ParameterSet"
    )


def _set_meta(model: Any, key: str, value: str) -> None:
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def _import_onnx():
    if not onnx_available():
        raise OnnxExportError(
            "ONNX/WASM export requires optional dependencies not installed.\n"
            "  From source:  pip install -e \".[export]\"  (run from the matrixAI repo)\n"
            "  From PyPI:    pip install \"matrixai-core[export]\"  (once published)\n"
            "  Installs:     onnx, onnxruntime, numpy"
        )
    try:
        onnx = import_module("onnx")
        numpy_helper = import_module("onnx.numpy_helper")
        helper = import_module("onnx.helper")
        TensorProto = onnx.TensorProto
        return onnx, numpy_helper, helper, TensorProto
    except Exception as exc:
        raise OnnxExportError(f"Unable to import onnx: {exc}") from exc


def _import_numpy():
    try:
        return import_module("numpy")
    except Exception as exc:
        raise OnnxExportError(f"Unable to import numpy: {exc}") from exc
