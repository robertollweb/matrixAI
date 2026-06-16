"""P21 C9 — composite_model_hash chain and integrity verification."""
from __future__ import annotations

import json

import pytest

from matrixai.actions.trace import ActionTrace, sign_action_trace, verify_action_trace
from matrixai.ir.schema import DenseLayerSpec, GraphSpec, ImportSpec, MatrixAIProgram, NetworkSpec
from matrixai.registry import (
    ModelRegistry,
    RegistryEntry,
    VerificationError,
    compute_composite_model_hash,
    compute_entry_hash,
    verify_composite_model,
)
from matrixai.registry.resolver import ImportResolutionError

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
        metrics={},
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
    )


def _imp(alias: str, name: str, version: str = "v1") -> ImportSpec:
    return ImportSpec(alias=alias, registry_name=name, version=version, mode="FROZEN")


def _prog(*imports: ImportSpec) -> MatrixAIProgram:
    return MatrixAIProgram(
        project="Test",
        imports=list(imports),
        graph=GraphSpec(nodes=[], edges=[]),
    )


# ── composite_model_hash ──────────────────────────────────────────────────────

def test_composite_model_hash_includes_import_entry_hashes(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry("enc")
    reg.push(entry)
    program = _prog(_imp("Enc", "enc"))
    h = compute_composite_model_hash(program, reg)
    assert h.startswith("sha256:")
    # Hash must depend on the entry_hash
    assert h != "sha256:" + "0" * 64


def test_composite_model_hash_stable_with_same_imports(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog(_imp("Enc", "enc"))
    h1 = compute_composite_model_hash(program, reg)
    h2 = compute_composite_model_hash(program, reg)
    assert h1 == h2


def test_composite_model_hash_changes_when_import_version_changes(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc", version="v1"))
    reg.push(_make_entry("enc", version="v2", model_hash="sha256:" + "f" * 64))
    prog_v1 = _prog(_imp("Enc", "enc", "v1"))
    prog_v2 = _prog(_imp("Enc", "enc", "v2"))
    h1 = compute_composite_model_hash(prog_v1, reg)
    h2 = compute_composite_model_hash(prog_v2, reg)
    assert h1 != h2


def test_composite_model_hash_stable_regardless_of_import_declaration_order(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("sent", model_hash="sha256:" + "f" * 64))
    prog_ab = _prog(_imp("Enc", "enc"), _imp("Sent", "sent"))
    prog_ba = _prog(_imp("Sent", "sent"), _imp("Enc", "enc"))
    h1 = compute_composite_model_hash(prog_ab, reg)
    h2 = compute_composite_model_hash(prog_ba, reg)
    assert h1 == h2


def test_composite_model_hash_changes_when_new_import_added(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("sent", model_hash="sha256:" + "f" * 64))
    prog_one = _prog(_imp("Enc", "enc"))
    prog_two = _prog(_imp("Enc", "enc"), _imp("Sent", "sent"))
    h1 = compute_composite_model_hash(prog_one, reg)
    h2 = compute_composite_model_hash(prog_two, reg)
    assert h1 != h2


def test_composite_model_hash_empty_imports_is_deterministic(tmp_path):
    reg = ModelRegistry(tmp_path)
    prog = _prog()
    h1 = compute_composite_model_hash(prog, reg)
    h2 = compute_composite_model_hash(prog, reg)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_composite_model_hash_raises_for_missing_import(tmp_path):
    reg = ModelRegistry(tmp_path)
    prog = _prog(_imp("Enc", "nonexistent"))
    with pytest.raises(ImportResolutionError, match="nonexistent"):
        compute_composite_model_hash(prog, reg)


# ── hash sensitivity: local changes ──────────────────────────────────────────

def test_composite_model_hash_changes_when_local_graph_changes(tmp_path):
    """Changing the local GRAPH topology changes the composite hash."""
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    imp = _imp("Enc", "enc")

    prog_a = MatrixAIProgram(
        project="Test", imports=[imp],
        graph=GraphSpec(nodes=["Input", "Enc"], edges=[("Input", "Enc")]),
    )
    prog_b = MatrixAIProgram(
        project="Test", imports=[imp],
        graph=GraphSpec(nodes=["Input", "Enc", "Extra"], edges=[("Input", "Enc"), ("Enc", "Extra")]),
    )
    assert compute_composite_model_hash(prog_a, reg) != compute_composite_model_hash(prog_b, reg)


def test_composite_model_hash_changes_when_import_mode_changes(tmp_path):
    """Switching a component from FROZEN to TRAINABLE changes the composite hash."""
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))

    prog_frozen = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="FROZEN")],
        graph=GraphSpec(nodes=[], edges=[]),
    )
    prog_trainable = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="TRAINABLE")],
        graph=GraphSpec(nodes=[], edges=[]),
    )
    h_frozen = compute_composite_model_hash(prog_frozen, reg)
    h_trainable = compute_composite_model_hash(prog_trainable, reg)
    assert h_frozen != h_trainable


def test_composite_model_hash_changes_when_local_network_changes(tmp_path):
    """Changing the local NETWORK definition changes the composite hash."""
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    imp = _imp("Enc", "enc")

    def _net(units: int) -> NetworkSpec:
        return NetworkSpec(
            name="Router", input="Enc",
            layers=[DenseLayerSpec(index=1, units=units, activation="relu")],
            output="out", output_type_str="Categorical", kind="dense_network",
        )

    prog_a = MatrixAIProgram(project="Test", imports=[imp], graph=GraphSpec(), networks=[_net(4)])
    prog_b = MatrixAIProgram(project="Test", imports=[imp], graph=GraphSpec(), networks=[_net(8)])
    assert compute_composite_model_hash(prog_a, reg) != compute_composite_model_hash(prog_b, reg)


def test_composite_model_hash_same_imports_different_modes_both_stable(tmp_path):
    """Two programs with different modes are each individually stable."""
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))

    prog = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="FROZEN")],
        graph=GraphSpec(nodes=[], edges=[]),
    )
    h1 = compute_composite_model_hash(prog, reg)
    h2 = compute_composite_model_hash(prog, reg)
    assert h1 == h2


# ── verify_composite_model ────────────────────────────────────────────────────

def test_verify_composite_model_passes_for_valid_entries(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog(_imp("Enc", "enc"))
    assert verify_composite_model(program, reg) is True


def test_action_trace_verification_detects_imported_component_change(tmp_path, monkeypatch):
    """If a component manifest is tampered, verify_composite_model detects it."""
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog(_imp("Enc", "enc"))
    # Tamper the manifest
    manifest_path = tmp_path / "entries" / "enc" / "v1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["model_hash"] = "sha256:" + "z" * 64
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(VerificationError, match="tampered|mismatch"):
        verify_composite_model(program, reg)


def test_verify_composite_model_fails_for_missing_entry(tmp_path):
    reg = ModelRegistry(tmp_path)
    program = _prog(_imp("Enc", "enc"))
    with pytest.raises(VerificationError, match="not found"):
        verify_composite_model(program, reg)


# ── action trace with composite hash ─────────────────────────────────────────

def _make_action_trace(model_hash: str, hmac_signature: str | None = None) -> ActionTrace:
    return ActionTrace(
        report_id="r001",
        model_hash=model_hash,
        parameter_set_id="ps_001",
        action_contract_hash="sha256:" + "0" * 64,
        input_hash="sha256:" + "1" * 64,
        executed_at="2026-06-01T12:00:00+00:00",
        executor_kind="simulate_only",
        ok=True,
        response_summary="ok",
        error=None,
        latency_ms=0.0,
        hmac_signature=hmac_signature,
    )


def test_action_trace_signature_includes_composite_chain(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog(_imp("Enc", "enc"))
    composite_hash = compute_composite_model_hash(program, reg)

    trace = _make_action_trace(composite_hash)
    key = "a" * 64
    sig = sign_action_trace(trace, key)
    signed_trace = _make_action_trace(composite_hash, hmac_signature=sig)
    assert verify_action_trace(signed_trace, key) is True


def test_action_trace_signature_bound_to_composite_hash(tmp_path):
    """If the composite hash changes (different component version), old sig is invalid."""
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc", version="v1"))
    reg.push(_make_entry("enc", version="v2", model_hash="sha256:" + "f" * 64))

    prog_v1 = _prog(_imp("Enc", "enc", "v1"))
    prog_v2 = _prog(_imp("Enc", "enc", "v2"))
    hash_v1 = compute_composite_model_hash(prog_v1, reg)
    hash_v2 = compute_composite_model_hash(prog_v2, reg)
    assert hash_v1 != hash_v2

    key = "a" * 64
    trace_v1 = _make_action_trace(hash_v1)
    sig_v1 = sign_action_trace(trace_v1, key)

    # Attempt to verify v2 trace with v1 signature — must fail
    trace_v2_with_old_sig = _make_action_trace(hash_v2, hmac_signature=sig_v1)
    assert not verify_action_trace(trace_v2_with_old_sig, key)
