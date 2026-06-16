"""P21 C7 — Hierarchical ParameterSet: namespaced paths, frozen separation."""
from __future__ import annotations

import pytest

from matrixai.ir.schema import GraphSpec, ImportSpec, MatrixAIProgram, VectorSpec
from matrixai.parameters import (
    ParameterSet,
    composite_parameter_schema_hash,
    load_frozen_parameters_from_registry,
    parameter_schema_hash,
    separate_parameters,
)
from matrixai.registry import ModelRegistry, RegistryEntry, compute_entry_hash

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE = dict(
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version="0.21.0",
)


def _make_entry(name: str, version: str = "v1", **overrides) -> RegistryEntry:
    kw = {**_BASE, **overrides, "name": name, "version": version}
    hash_kw = {k: v for k, v in kw.items() if k in _BASE}
    hash_kw["name"] = name
    hash_kw["version"] = version
    eh = compute_entry_hash(**hash_kw)
    return RegistryEntry(
        name=name, version=version, entry_hash=eh,
        model_hash=kw["model_hash"],
        parameter_schema_hash=kw["parameter_schema_hash"],
        parameter_set_id=kw["parameter_set_id"],
        input_type={"kind": "VECTOR"},
        output_type={"kind": "Tensor", "shape": [128]},
        metrics={},
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
    )


def _program_with_imports(*imports: ImportSpec) -> MatrixAIProgram:
    return MatrixAIProgram(
        project="Test",
        imports=list(imports),
        graph=GraphSpec(nodes=[], edges=[]),
    )


def _frozen_imp(alias: str, name: str) -> ImportSpec:
    return ImportSpec(alias=alias, registry_name=name, version="v1", mode="FROZEN")


def _trainable_imp(alias: str, name: str) -> ImportSpec:
    return ImportSpec(alias=alias, registry_name=name, version="v1", mode="TRAINABLE")


# ── namespaced paths ──────────────────────────────────────────────────────────

def test_parameter_set_supports_namespaced_paths():
    ps = ParameterSet(
        parameter_set_id="test",
        model_hash="mxai_abc",
        parameter_schema_hash="params_xyz",
        parameters={
            "TextEncoder.W1": {"values": [1.0, 2.0], "shape": [2]},
            "Router.W1": {"values": [3.0, 4.0], "shape": [2]},
        },
    )
    assert "TextEncoder.W1" in ps.parameters
    assert "Router.W1" in ps.parameters
    assert ps.parameters["TextEncoder.W1"]["values"] == [1.0, 2.0]


def test_parameter_set_deep_namespaced_paths():
    ps = ParameterSet(
        parameter_set_id="test",
        model_hash="h",
        parameter_schema_hash="s",
        parameters={
            "TextEncoder.layer1.W1": {"values": [[1.0]], "shape": [1, 1]},
            "TextEncoder.layer1.b1": {"values": [0.0], "shape": [1]},
        },
    )
    assert "TextEncoder.layer1.W1" in ps.parameters
    assert "TextEncoder.layer1.b1" in ps.parameters


def test_parameter_set_mixed_namespaced_and_flat():
    ps = ParameterSet(
        parameter_set_id="test",
        model_hash="h",
        parameter_schema_hash="s",
        parameters={
            "Router.W1": {"values": [1.0], "shape": [1]},
            "flat_param": {"values": 0.5},
        },
    )
    assert "Router.W1" in ps.parameters
    assert "flat_param" in ps.parameters


# ── schema hash excludes frozen ───────────────────────────────────────────────

def test_parameter_schema_hash_excludes_frozen_components():
    full_manifest = [
        {"function": "TextEncoder", "name": "W1", "role": "weights", "shape": [4, 4], "dtype": "float32", "initializer": "xavier_normal"},
        {"function": "TextEncoder", "name": "b1", "role": "bias", "shape": [4], "dtype": "float32", "initializer": "zeros"},
        {"function": "Router", "name": "W1", "role": "weights", "shape": [4, 2], "dtype": "float32", "initializer": "xavier_normal"},
    ]
    frozen_aliases = frozenset({"TextEncoder"})
    h_composite = composite_parameter_schema_hash(full_manifest, frozen_aliases)
    h_all = parameter_schema_hash(full_manifest)
    assert h_composite != h_all


def test_composite_schema_hash_with_no_frozen_equals_full():
    manifest = [
        {"function": "Router", "name": "W1", "role": "weights", "shape": [4, 2], "dtype": "float32", "initializer": "xavier_normal"},
    ]
    assert composite_parameter_schema_hash(manifest) == parameter_schema_hash(manifest)


def test_composite_schema_hash_same_for_same_trainable_params():
    manifest = [
        {"function": "TextEncoder", "name": "W1", "role": "weights", "shape": [4, 4], "dtype": "float32", "initializer": "xavier_normal"},
        {"function": "Router", "name": "W1", "role": "weights", "shape": [4, 2], "dtype": "float32", "initializer": "xavier_normal"},
    ]
    frozen = frozenset({"TextEncoder"})
    h1 = composite_parameter_schema_hash(manifest, frozen)
    h2 = composite_parameter_schema_hash(manifest, frozen)
    assert h1 == h2


# ── load frozen from registry ─────────────────────────────────────────────────

def test_parameter_set_loads_frozen_from_registry(tmp_path):
    reg = ModelRegistry(tmp_path)
    enc_entry = _make_entry("enc")
    reg.push(enc_entry)
    program = _program_with_imports(_frozen_imp("TextEncoder", "enc"))
    frozen = load_frozen_parameters_from_registry(program, reg)
    assert "TextEncoder" in frozen
    info = frozen["TextEncoder"]
    assert info["registry_name"] == "enc"
    assert info["parameter_set_id"] == enc_entry.parameter_set_id
    assert info["mode"] == "FROZEN"


def test_load_frozen_skips_trainable_imports(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("router"))
    program = _program_with_imports(_trainable_imp("Router", "router"))
    frozen = load_frozen_parameters_from_registry(program, reg)
    assert "Router" not in frozen


def test_load_frozen_returns_empty_for_no_imports(tmp_path):
    reg = ModelRegistry(tmp_path)
    program = _program_with_imports()
    assert load_frozen_parameters_from_registry(program, reg) == {}


def test_load_frozen_handles_multiple_frozen(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("bert", model_hash="sha256:" + "f" * 64))
    program = _program_with_imports(
        _frozen_imp("Enc", "enc"),
        _frozen_imp("Bert", "bert"),
    )
    frozen = load_frozen_parameters_from_registry(program, reg)
    assert "Enc" in frozen
    assert "Bert" in frozen


# ── separate trainable and frozen ─────────────────────────────────────────────

def test_parameter_set_separates_trainable_and_frozen():
    ps = ParameterSet(
        parameter_set_id="test",
        model_hash="h",
        parameter_schema_hash="s",
        parameters={
            "TextEncoder.W1": {"values": [1.0]},
            "Router.W1": {"values": [2.0]},
            "flat_param": {"values": 0.1},
        },
    )
    program = _program_with_imports(
        _frozen_imp("TextEncoder", "enc"),
        _trainable_imp("Router", "router"),
    )
    trainable, frozen_keys = separate_parameters(ps, program)
    assert "Router.W1" in trainable
    assert "flat_param" in trainable
    assert "TextEncoder.W1" not in trainable
    assert "TextEncoder.W1" in frozen_keys


def test_separate_parameters_all_trainable():
    ps = ParameterSet(
        parameter_set_id="t",
        model_hash="h",
        parameter_schema_hash="s",
        parameters={"Router.W1": {"values": [1.0]}},
    )
    program = _program_with_imports(_trainable_imp("Router", "router"))
    trainable, frozen_keys = separate_parameters(ps, program)
    assert "Router.W1" in trainable
    assert frozen_keys == []


def test_separate_parameters_no_imports():
    ps = ParameterSet(
        parameter_set_id="t",
        model_hash="h",
        parameter_schema_hash="s",
        parameters={"W1": {"values": [1.0]}},
    )
    program = _program_with_imports()
    trainable, frozen_keys = separate_parameters(ps, program)
    assert "W1" in trainable
    assert frozen_keys == []
