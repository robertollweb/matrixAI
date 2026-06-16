"""P21 C6 — BackendContractAnalyzer: composite_model node type."""
from __future__ import annotations

import pytest

from matrixai import __version__ as _MATRIXAI_VERSION
from matrixai.compiler import BackendContractAnalyzer
from matrixai.ir.schema import GraphSpec, ImportSpec, MatrixAIProgram, VectorSpec
from matrixai.registry import ModelRegistry, RegistryEntry, compute_entry_hash

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE = dict(
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version=_MATRIXAI_VERSION,
)


def _make_entry(name: str, version: str = "v1", *,
                interpretability_level: str = "full",
                blockers: list[str] | None = None,
                matrixai_version: str = _MATRIXAI_VERSION,
                **overrides) -> RegistryEntry:
    kw = {**_BASE, **overrides, "name": name, "version": version,
          "matrixai_version": matrixai_version}
    hash_kw = {k: v for k, v in kw.items() if k in {
        "name", "version", "model_hash", "parameter_schema_hash",
        "parameter_set_id", "training_trace_hash", "evaluation_report_hash", "matrixai_version"
    }}
    eh = compute_entry_hash(**hash_kw)
    return RegistryEntry(
        name=name, version=version, entry_hash=eh,
        model_hash=kw["model_hash"],
        parameter_schema_hash=kw["parameter_schema_hash"],
        parameter_set_id=kw["parameter_set_id"],
        input_type={"kind": "VECTOR"},
        output_type={"kind": "Tensor", "shape": [128]},
        metrics={"accuracy": 0.91},
        matrixai_version=matrixai_version,
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level=interpretability_level,
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
        blockers=blockers or [],
    )


def _simple_program(alias: str, registry_name: str) -> MatrixAIProgram:
    return MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias=alias, registry_name=registry_name, version="v1", mode="FROZEN")],
        graph=GraphSpec(nodes=["Input", alias], edges=[("Input", alias)]),
        vectors=[VectorSpec(name="Input", size=1, fields=["x"])],
    )


def _chain_program(alias_a: str, alias_b: str,
                   reg_a: str, reg_b: str) -> MatrixAIProgram:
    return MatrixAIProgram(
        project="Test",
        imports=[
            ImportSpec(alias=alias_a, registry_name=reg_a, version="v1", mode="FROZEN"),
            ImportSpec(alias=alias_b, registry_name=reg_b, version="v1", mode="FROZEN"),
        ],
        graph=GraphSpec(
            nodes=["Input", alias_a, alias_b],
            edges=[("Input", alias_a), (alias_a, alias_b)],
        ),
        vectors=[VectorSpec(name="Input", size=1, fields=["x"])],
    )


# ── composite_model nodes ─────────────────────────────────────────────────────

def test_analyzer_detects_composite_model_nodes(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    enc_node = next(n for n in report.nodes if n.node == "Enc")
    assert enc_node.node_type == "composite_model"


def test_analyzer_composite_node_is_supported(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    enc_node = next(n for n in report.nodes if n.node == "Enc")
    assert enc_node.supported is True


def test_analyzer_composite_node_kind_reflects_mode(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="TRAINABLE")],
        graph=GraphSpec(nodes=["Enc"], edges=[]),
    )
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    enc_node = next(n for n in report.nodes if n.node == "Enc")
    assert "trainable" in enc_node.kind


# ── component_manifest ────────────────────────────────────────────────────────

def test_analyzer_builds_component_manifest(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    assert len(report.component_manifest) == 1
    entry_info = report.component_manifest[0]
    assert entry_info["alias"] == "Enc"
    assert entry_info["registry_name"] == "enc"
    assert entry_info["mode"] == "FROZEN"
    assert entry_info["interpretability_level"] == "full"


def test_analyzer_component_manifest_includes_entry_hash(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry("enc")
    reg.push(entry)
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    assert report.component_manifest[0]["entry_hash"] == entry.entry_hash


def test_analyzer_component_manifest_empty_without_registry(tmp_path):
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program)
    assert report.component_manifest == []


# ── interpretability warnings ─────────────────────────────────────────────────

def test_analyzer_warns_about_interpretability_reduced(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc", interpretability_level="reduced",
                         model_hash="sha256:" + "f" * 64))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    assert any("interpretability_level=reduced" in w for w in report.warnings)


def test_analyzer_no_interp_warning_for_full(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    assert not any("interpretability_level" in w for w in report.warnings)


# ── blockers ──────────────────────────────────────────────────────────────────

def test_analyzer_propagates_blockers_from_imported_component(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc", blockers=["custom_blocker"],
                         model_hash="sha256:" + "f" * 64))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    enc_node = next(n for n in report.nodes if n.node == "Enc")
    assert enc_node.supported is False
    assert "custom_blocker" in enc_node.reason


def test_analyzer_rejects_action_in_intermediate_component(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc", blockers=["real_with_audit_action"]))
    reg.push(_make_entry("sent", model_hash="sha256:" + "f" * 64))
    # enc is intermediate: Input → enc → sent
    program = _chain_program("Enc", "Sent", "enc", "sent")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    enc_node = next(n for n in report.nodes if n.node == "Enc")
    assert enc_node.supported is False
    assert "real_with_audit" in enc_node.reason


# ── matrixai version mismatch ─────────────────────────────────────────────────

def test_analyzer_reports_matrixai_version_mismatch(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc", matrixai_version="0.20.0",
                         model_hash="sha256:" + "f" * 64))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    assert any("0.20.0" in w for w in report.warnings)


def test_analyzer_no_version_warning_for_current_version(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _simple_program("Enc", "enc")
    report = BackendContractAnalyzer().analyze(program, registry=reg)
    assert not any(_MATRIXAI_VERSION in w and "version" in w for w in report.warnings)
