"""P21 C5 — Type system: check_composite_program_types."""
from __future__ import annotations

import pytest

from matrixai.ir.schema import GraphSpec, ImportSpec, MatrixAIProgram
from matrixai.registry import ModelRegistry, RegistryEntry, compute_entry_hash
from matrixai.types import TypeCompatibilityError, check_composite_program_types

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE = dict(
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version="0.21.0",
)


def _make_entry(name: str, version: str, input_type: dict, output_type: dict, **overrides) -> RegistryEntry:
    kw = {**_BASE, **overrides, "name": name, "version": version}
    hash_kw = {k: v for k, v in kw.items() if k in {
        "name", "version", "model_hash", "parameter_schema_hash",
        "parameter_set_id", "training_trace_hash", "evaluation_report_hash", "matrixai_version"
    }}
    eh = compute_entry_hash(**hash_kw)
    return RegistryEntry(
        name=name,
        version=version,
        entry_hash=eh,
        model_hash=kw["model_hash"],
        parameter_schema_hash=kw["parameter_schema_hash"],
        parameter_set_id=kw["parameter_set_id"],
        input_type=input_type,
        output_type=output_type,
        metrics={},
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
    )


def _reg_with(*entries: RegistryEntry, tmp_path) -> ModelRegistry:
    reg = ModelRegistry(tmp_path)
    for entry in entries:
        reg.push(entry)
    return reg


def _imp(alias: str, name: str, version: str = "v1") -> ImportSpec:
    return ImportSpec(alias=alias, registry_name=name, version=version, mode="FROZEN")


def _prog(imports: list[ImportSpec], edges: list[tuple[str, str]]) -> MatrixAIProgram:
    nodes = list({n for edge in edges for n in edge})
    return MatrixAIProgram(
        project="Test",
        imports=imports,
        graph=GraphSpec(nodes=nodes, edges=edges),
    )


# ── compatible ────────────────────────────────────────────────────────────────

def test_type_check_accepts_compatible_components(tmp_path):
    enc = _make_entry("enc", "v1",
                      input_type={"kind": "VECTOR"},
                      output_type={"kind": "Tensor", "shape": [128]})
    sent = _make_entry("sent", "v1",
                       input_type={"kind": "Tensor", "shape": [128]},
                       output_type={"kind": "Probability"},
                       model_hash="sha256:" + "f" * 64)
    reg = _reg_with(enc, sent, tmp_path=tmp_path)
    program = _prog([_imp("Enc", "enc"), _imp("Sent", "sent")],
                    [("Enc", "Sent")])
    result = check_composite_program_types(program, reg)
    assert result.errors == []


def test_type_check_empty_imports_ok(tmp_path):
    reg = ModelRegistry(tmp_path)
    program = _prog([], [])
    result = check_composite_program_types(program, reg)
    assert result.errors == []


def test_type_check_no_edges_between_imports_ok(tmp_path):
    enc = _make_entry("enc", "v1",
                      input_type={"kind": "VECTOR"},
                      output_type={"kind": "Tensor", "shape": [128]})
    reg = _reg_with(enc, tmp_path=tmp_path)
    program = _prog([_imp("Enc", "enc")], [])
    result = check_composite_program_types(program, reg)
    assert result.errors == []


# ── shape mismatch ────────────────────────────────────────────────────────────

def test_type_check_rejects_shape_mismatch(tmp_path):
    enc = _make_entry("enc", "v1",
                      input_type={"kind": "VECTOR"},
                      output_type={"kind": "Tensor", "shape": [128]})
    sent = _make_entry("sent", "v1",
                       input_type={"kind": "Tensor", "shape": [64]},
                       output_type={"kind": "Probability"},
                       model_hash="sha256:" + "f" * 64)
    reg = _reg_with(enc, sent, tmp_path=tmp_path)
    program = _prog([_imp("Enc", "enc"), _imp("Sent", "sent")],
                    [("Enc", "Sent")])
    with pytest.raises(TypeCompatibilityError) as exc_info:
        check_composite_program_types(program, reg)
    detail = " ".join(exc_info.value.errors)
    assert "Enc" in detail or "shape" in detail.lower()


def test_type_check_handles_tensor_dim_mismatch(tmp_path):
    a = _make_entry("a", "v1",
                    input_type={"kind": "VECTOR"},
                    output_type={"kind": "Tensor", "shape": [256]})
    b = _make_entry("b", "v1",
                    input_type={"kind": "Tensor", "shape": [512]},
                    output_type={"kind": "Label"},
                    model_hash="sha256:" + "f" * 64)
    reg = _reg_with(a, b, tmp_path=tmp_path)
    program = _prog([_imp("A", "a"), _imp("B", "b")], [("A", "B")])
    with pytest.raises(TypeCompatibilityError):
        check_composite_program_types(program, reg)


# ── kind mismatch ─────────────────────────────────────────────────────────────

def test_type_check_rejects_kind_mismatch(tmp_path):
    a = _make_entry("a", "v1",
                    input_type={"kind": "VECTOR"},
                    output_type={"kind": "Probability"})
    b = _make_entry("b", "v1",
                    input_type={"kind": "Tensor", "shape": [128]},
                    output_type={"kind": "Label"},
                    model_hash="sha256:" + "f" * 64)
    reg = _reg_with(a, b, tmp_path=tmp_path)
    program = _prog([_imp("A", "a"), _imp("B", "b")], [("A", "B")])
    with pytest.raises(TypeCompatibilityError):
        check_composite_program_types(program, reg)


# ── chain ─────────────────────────────────────────────────────────────────────

def test_type_check_propagates_through_chain_of_components(tmp_path):
    a = _make_entry("a", "v1",
                    input_type={"kind": "VECTOR"},
                    output_type={"kind": "Tensor", "shape": [128]})
    b = _make_entry("b", "v1",
                    input_type={"kind": "Tensor", "shape": [128]},
                    output_type={"kind": "Tensor", "shape": [64]},
                    model_hash="sha256:" + "f" * 64)
    c = _make_entry("c", "v1",
                    input_type={"kind": "Tensor", "shape": [64]},
                    output_type={"kind": "Probability"},
                    model_hash="sha256:" + "e" * 64)
    reg = _reg_with(a, b, c, tmp_path=tmp_path)
    program = _prog(
        [_imp("A", "a"), _imp("B", "b"), _imp("C", "c")],
        [("A", "B"), ("B", "C")],
    )
    result = check_composite_program_types(program, reg)
    assert result.errors == []


def test_type_check_detects_error_in_chain(tmp_path):
    a = _make_entry("a", "v1",
                    input_type={"kind": "VECTOR"},
                    output_type={"kind": "Tensor", "shape": [128]})
    b = _make_entry("b", "v1",
                    input_type={"kind": "Tensor", "shape": [128]},
                    output_type={"kind": "Tensor", "shape": [64]},
                    model_hash="sha256:" + "f" * 64)
    c = _make_entry("c", "v1",
                    input_type={"kind": "Tensor", "shape": [256]},  # mismatch!
                    output_type={"kind": "Probability"},
                    model_hash="sha256:" + "e" * 64)
    reg = _reg_with(a, b, c, tmp_path=tmp_path)
    program = _prog(
        [_imp("A", "a"), _imp("B", "b"), _imp("C", "c")],
        [("A", "B"), ("B", "C")],
    )
    with pytest.raises(TypeCompatibilityError):
        check_composite_program_types(program, reg)


# ── unknown / missing types ───────────────────────────────────────────────────

def test_type_check_rejects_unknown_output_type(tmp_path):
    a = _make_entry("a", "v1",
                    input_type={"kind": "VECTOR"},
                    output_type={"kind": "WeirdType"})
    b = _make_entry("b", "v1",
                    input_type={"kind": "Tensor", "shape": [64]},
                    output_type={"kind": "Probability"},
                    model_hash="sha256:" + "f" * 64)
    reg = _reg_with(a, b, tmp_path=tmp_path)
    program = _prog([_imp("A", "a"), _imp("B", "b")], [("A", "B")])
    with pytest.raises(TypeCompatibilityError):
        check_composite_program_types(program, reg)


def test_type_check_skips_edge_when_dst_has_no_input_type(tmp_path):
    a = _make_entry("a", "v1",
                    input_type={"kind": "VECTOR"},
                    output_type={"kind": "Tensor", "shape": [128]})
    reg = _reg_with(a, tmp_path=tmp_path)
    # Edge to a non-imported node (e.g., VECTOR or output sink)
    program = _prog([_imp("A", "a")], [("A", "Output")])
    result = check_composite_program_types(program, reg)
    assert result.errors == []
