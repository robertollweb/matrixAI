# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""ONNX equivalence validator: compares onnxruntime outputs vs differentiable_python."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module, util
from pathlib import Path
from typing import Any

from matrixai.ir import MatrixAIProgram
from matrixai.parameters.store import ParameterSet

_SUPPORTED_KINDS = frozenset({"softmax_linear", "sigmoid_linear", "layer_call"})
_DEFAULT_ATOL = 1e-5
_DEFAULT_RTOL = 1e-4
_DEFAULT_N_SAMPLES = 20


class OnnxEquivalenceError(ValueError):
    pass


@dataclass(frozen=True)
class OnnxEquivalenceResult:
    passed: bool
    atol: float
    rtol: float
    max_abs_diff: float
    max_rel_diff: float
    n_samples: int
    n_outputs_per_sample: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "atol": self.atol,
            "rtol": self.rtol,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "n_samples": self.n_samples,
            "n_outputs_per_sample": self.n_outputs_per_sample,
        }


class OnnxEquivalenceValidator:
    """Validates that an ONNX model produces outputs equivalent to differentiable_python."""

    DEFAULT_ATOL = _DEFAULT_ATOL
    DEFAULT_RTOL = _DEFAULT_RTOL
    DEFAULT_N_SAMPLES = _DEFAULT_N_SAMPLES

    def validate(
        self,
        program: MatrixAIProgram,
        parameter_set: ParameterSet,
        onnx_path: str | Path,
        *,
        atol: float = _DEFAULT_ATOL,
        rtol: float = _DEFAULT_RTOL,
        n_samples: int = _DEFAULT_N_SAMPLES,
        seed: int = 42,
    ) -> OnnxEquivalenceResult:
        np = _import_numpy()
        sess = _import_ort_session(onnx_path)
        rng = np.random.default_rng(seed)
        max_abs = 0.0
        max_rel = 0.0
        n_out = 0
        all_close = True

        # Dense-network programs have no FUNCTION declarations; use numpy forward pass.
        dense_nets = [n for n in program.networks if getattr(n, "kind", "") == "dense_network"]
        is_dense_only = bool(dense_nets) and not program.functions
        # Composite networks (P19) use the stdlib composite_forward as reference.
        composite_nets = [n for n in program.networks if getattr(n, "kind", "") == "composite_network"]

        if composite_nets:
            network = composite_nets[0]
            if not program.vectors:
                raise OnnxEquivalenceError(f"No VECTOR input for composite network {network.name!r}")
            vector = program.vectors[0]
            inputs = rng.random((n_samples, vector.size), dtype=np.float64).astype(np.float32)
            for row in inputs:
                ref_out = _run_composite_forward(network, parameter_set, vector, row)
                ort_out = _run_ort(sess, vector, row, np)
                if len(ref_out) != len(ort_out):
                    raise OnnxEquivalenceError(
                        f"Output length mismatch: composite_forward={len(ref_out)}, ort={len(ort_out)}"
                    )
                n_out = len(ref_out)
                for a, b in zip(ref_out, ort_out):
                    a_f, b_f = float(a), float(b)
                    abs_diff = abs(a_f - b_f)
                    rel_diff = abs_diff / max(abs(b_f), 1e-10)
                    if abs_diff > max_abs:
                        max_abs = abs_diff
                    if rel_diff > max_rel:
                        max_rel = rel_diff
                    if abs_diff > atol + rtol * abs(b_f):
                        all_close = False
        elif is_dense_only:
            network = dense_nets[0]
            if not program.vectors:
                raise OnnxEquivalenceError(f"No VECTOR input for dense network {network.name!r}")
            vector = program.vectors[0]
            inputs = rng.random((n_samples, vector.size), dtype=np.float64).astype(np.float32)
            for row in inputs:
                np_out = _run_numpy_dense(network, parameter_set, row, np)
                ort_out = _run_ort(sess, vector, row, np)
                if len(np_out) != len(ort_out):
                    raise OnnxEquivalenceError(
                        f"Output length mismatch: numpy={len(np_out)}, ort={len(ort_out)}"
                    )
                n_out = len(np_out)
                for a, b in zip(np_out, ort_out):
                    a_f, b_f = float(a), float(b)
                    abs_diff = abs(a_f - b_f)
                    rel_diff = abs_diff / max(abs(b_f), 1e-10)
                    if abs_diff > max_abs:
                        max_abs = abs_diff
                    if rel_diff > max_rel:
                        max_rel = rel_diff
                    if abs_diff > atol + rtol * abs(b_f):
                        all_close = False
        else:
            ns = _build_dp_module(program)
            fn = _find_exportable_function(program)
            vector = _find_vector(fn, program)
            sequence = program.sequences[0] if program.sequences else None

            if vector is None and sequence is None:
                raise OnnxEquivalenceError(
                    f"No VECTOR or SEQUENCE input for {fn.name} in {program.project!r}"
                )

            params = parameter_set.runtime_parameters()

            if sequence is not None:
                inputs = rng.integers(0, sequence.vocab_size, size=(n_samples, sequence.length))
                for row in inputs:
                    dp_out = _run_dp_sequence(ns, fn, sequence, row, params)
                    ort_out = _run_ort_sequence(sess, sequence, row, np)
                    if len(dp_out) != len(ort_out):
                        raise OnnxEquivalenceError(
                            f"Output length mismatch: dp={len(dp_out)}, ort={len(ort_out)}"
                        )
                    n_out = len(dp_out)
                    for a, b in zip(dp_out, ort_out):
                        a_f, b_f = float(a), float(b)
                        abs_diff = abs(a_f - b_f)
                        rel_diff = abs_diff / max(abs(b_f), 1e-10)
                        if abs_diff > max_abs:
                            max_abs = abs_diff
                        if rel_diff > max_rel:
                            max_rel = rel_diff
                        if abs_diff > atol + rtol * abs(b_f):
                            all_close = False
            else:
                inputs = rng.random((n_samples, vector.size), dtype=np.float64).astype(np.float32)
                for row in inputs:
                    dp_out = _run_dp(ns, fn, vector, row, params, np)
                    ort_out = _run_ort(sess, vector, row, np)
                    if len(dp_out) != len(ort_out):
                        raise OnnxEquivalenceError(
                            f"Output length mismatch: dp={len(dp_out)}, ort={len(ort_out)}"
                        )
                    n_out = len(dp_out)
                    for a, b in zip(dp_out, ort_out):
                        a_f, b_f = float(a), float(b)
                        abs_diff = abs(a_f - b_f)
                        rel_diff = abs_diff / max(abs(b_f), 1e-10)
                        if abs_diff > max_abs:
                            max_abs = abs_diff
                        if rel_diff > max_rel:
                            max_rel = rel_diff
                        if abs_diff > atol + rtol * abs(b_f):
                            all_close = False

        passed = all_close

        return OnnxEquivalenceResult(
            passed=passed,
            atol=atol,
            rtol=rtol,
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
            n_samples=n_samples,
            n_outputs_per_sample=n_out,
        )


def write_export_manifest(
    export_result: Any,
    equivalence_result: OnnxEquivalenceResult,
    manifest_path: str | Path,
) -> None:
    """Write export_manifest.json with hashes, format, tolerance and equivalence result."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_hash": export_result.model_hash,
        "parameter_schema_hash": export_result.parameter_schema_hash,
        "parameter_set_id": export_result.parameter_set_id,
        "format": "onnx",
        "format_version": export_result.opset_version,
        "input_name": export_result.input_name,
        "input_shape": export_result.input_shape,
        "output_name": export_result.output_name,
        "output_shape": export_result.output_shape,
        "exported_function": export_result.exported_functions[0] if export_result.exported_functions else None,
        "tolerance": {
            "atol": equivalence_result.atol,
            "rtol": equivalence_result.rtol,
        },
        "equivalence_check": equivalence_result.to_dict(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def validate_onnx_equivalence(
    program: MatrixAIProgram,
    parameter_set: ParameterSet,
    onnx_path: str | Path,
    *,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
    n_samples: int = _DEFAULT_N_SAMPLES,
    seed: int = 42,
) -> OnnxEquivalenceResult:
    return OnnxEquivalenceValidator().validate(
        program, parameter_set, onnx_path,
        atol=atol, rtol=rtol, n_samples=n_samples, seed=seed,
    )


def ort_available() -> bool:
    return util.find_spec("onnxruntime") is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_exportable_function(program: MatrixAIProgram):
    # For layer_call pipelines, the last function holds the final output in state
    layer_call_fns = [f for f in program.functions if f.semantic.kind == "layer_call"]
    if layer_call_fns:
        return layer_call_fns[-1]
    for fn in program.functions:
        if fn.semantic.kind in _SUPPORTED_KINDS:
            return fn
    kinds = {f.semantic.kind for f in program.functions}
    raise OnnxEquivalenceError(
        f"No exportable function in {program.project!r}. Found: {sorted(kinds)}"
    )


def _find_vector(fn, program: MatrixAIProgram):
    vectors = {v.name: v for v in program.vectors}
    for name in fn.semantic.inputs:
        if name in vectors:
            return vectors[name]
    return program.vectors[0] if program.vectors else None


def _build_dp_module(program: MatrixAIProgram) -> dict:
    from matrixai.compiler import DifferentiablePythonCompiler
    ns: dict = {}
    exec(DifferentiablePythonCompiler().compile(program), ns)
    return ns


def _run_dp(ns: dict, fn, vector, row, params: dict, np) -> list[float]:
    inp = {vector.name: {field: float(v) for field, v in zip(vector.fields, row)}}
    result = ns["run"](inp, params)
    val = result["state"][fn.name]
    if isinstance(val, dict):
        return [float(v) for v in val.values()]
    if isinstance(val, list):
        return [float(v) for v in val]
    return [float(val)]


def _run_ort(sess, vector, row, np) -> list[float]:
    x = row.reshape(1, -1).astype(np.float32)
    out = sess.run(None, {vector.name: x})[0]
    return [float(v) for v in out.reshape(-1)]


def _run_dp_sequence(ns: dict, fn, sequence, ids, params: dict) -> list[float]:
    inp = {sequence.name: [int(i) for i in ids]}
    result = ns["run"](inp, params)
    val = result["state"][fn.name]
    if isinstance(val, list):
        return [float(v) for v in val]
    if isinstance(val, dict):
        return [float(v) for v in val.values()]
    return [float(val)]


def _run_ort_sequence(sess, sequence, ids, np) -> list[float]:
    x = ids.reshape(1, -1).astype(np.int64)
    out = sess.run(None, {sequence.name: x})[0]
    return [float(v) for v in out.reshape(-1)]


def _import_numpy():
    try:
        return import_module("numpy")
    except Exception as exc:
        raise OnnxEquivalenceError(f"numpy not available: {exc}") from exc


def _run_numpy_dense(network, parameter_set: ParameterSet, row, np) -> list[float]:
    """Numpy forward pass for a DenseNetworkGenerator NETWORK block.

    Mirrors the Gemm→activation chain built by _build_dense_network_pipeline in onnx_exporter.
    Parameters are keyed as {network.name}.W{i} / {network.name}.b{i} (1-based).
    """
    net = network.name
    x = np.array(row, dtype=np.float32).reshape(1, -1)
    for layer in network.layers:
        i = layer.index
        W = np.array(parameter_set.parameters[f"{net}.W{i}"]["values"], dtype=np.float32)
        b = np.array(parameter_set.parameters[f"{net}.b{i}"]["values"], dtype=np.float32)
        x = x @ W.T + b  # Gemm with transB=1 → (1, units_out)
        act = layer.activation.lower()
        if act == "relu":
            x = np.maximum(0.0, x)
        elif act == "sigmoid":
            x = 1.0 / (1.0 + np.exp(-x))
        elif act == "tanh":
            x = np.tanh(x)
        elif act == "softmax":
            e = np.exp(x - x.max(axis=1, keepdims=True))
            x = e / e.sum(axis=1, keepdims=True)
        # linear / identity: no-op
    return x.flatten().tolist()


def _run_composite_forward(network, parameter_set: ParameterSet, vector, row) -> list[float]:
    """Reference forward for a composite_network (P19) — mirrors what the ONNX graph
    computes (composite_forward at inference: Dropout/Pool/Reshape are identity)."""
    from matrixai.forward.composite_forward import composite_forward
    input_data = {field: float(v) for field, v in zip(vector.fields, row)}
    return [float(v) for v in composite_forward(network, parameter_set, input_data, training=False)]


def _import_ort_session(onnx_path: str | Path):
    if not ort_available():
        raise OnnxEquivalenceError(
            "Equivalence validation requires 'onnxruntime'. "
            "Install with: pip install onnxruntime"
        )
    try:
        ort = import_module("onnxruntime")
        return ort.InferenceSession(str(onnx_path))
    except Exception as exc:
        raise OnnxEquivalenceError(f"Failed to load ONNX model for validation: {exc}") from exc
