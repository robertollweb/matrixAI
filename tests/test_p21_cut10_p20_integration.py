"""P21 C10 — P20 integration: action contracts with composite models."""
from __future__ import annotations

import pytest

from matrixai.actions.composite_integration import (
    CompositeAuditResult,
    composite_dry_run,
    get_component_chain,
    validate_composite_action_contract,
)
from matrixai.actions.trace import ActionTrace, sign_action_trace, verify_action_trace
from matrixai.ir.schema import GraphSpec, ImportSpec, MatrixAIProgram
from matrixai.registry import (
    ModelRegistry,
    RegistryEntry,
    compute_composite_model_hash,
    compute_entry_hash,
)

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE = dict(
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version="0.21.0",
)


def _make_entry(name: str, version: str = "v1", *,
                blockers: list[str] | None = None, **overrides) -> RegistryEntry:
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
        blockers=blockers or [],
    )


def _imp(alias: str, name: str, version: str = "v1") -> ImportSpec:
    return ImportSpec(alias=alias, registry_name=name, version=version, mode="FROZEN")


def _chain_prog(alias_a: str, alias_b: str,
                reg_a: str = "enc", reg_b: str = "router") -> MatrixAIProgram:
    """A → B chain: A is intermediate, B is terminal."""
    return MatrixAIProgram(
        project="Test",
        imports=[_imp(alias_a, reg_a), _imp(alias_b, reg_b)],
        graph=GraphSpec(
            nodes=["Input", alias_a, alias_b],
            edges=[("Input", alias_a), (alias_a, alias_b)],
        ),
    )


def _single_prog(alias: str, reg: str = "enc") -> MatrixAIProgram:
    return MatrixAIProgram(
        project="Test",
        imports=[_imp(alias, reg)],
        graph=GraphSpec(nodes=["Input", alias], edges=[("Input", alias)]),
    )


# ── action contract attaches to terminal ──────────────────────────────────────

def test_action_contract_attaches_to_composite_terminal(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64))
    program = _chain_prog("Enc", "Router")
    result = validate_composite_action_contract(program, reg)
    chain = result.component_chain
    terminal = next(e for e in chain if e.alias == "Router")
    intermediate = next(e for e in chain if e.alias == "Enc")
    assert terminal.is_terminal is True
    assert intermediate.is_terminal is False


def test_intermediate_component_cannot_have_real_action(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc", blockers=["real_with_audit_action"]))
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64))
    program = _chain_prog("Enc", "Router")
    result = validate_composite_action_contract(program, reg)
    assert not result.ok
    assert any("real_with_audit" in e for e in result.errors)


def test_terminal_with_real_action_is_allowed(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64,
                          blockers=["real_with_audit_action"]))
    program = _chain_prog("Enc", "Router")
    result = validate_composite_action_contract(program, reg)
    # Terminal can have real actions — no error
    assert result.ok


def test_validate_composite_returns_composite_hash(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _single_prog("Enc")
    result = validate_composite_action_contract(program, reg)
    assert result.composite_model_hash.startswith("sha256:")


# ── action trace with composite hash ─────────────────────────────────────────

def test_action_trace_firms_composite_model_hash(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _single_prog("Enc")
    composite_hash = compute_composite_model_hash(program, reg)

    trace = ActionTrace(
        report_id="r001",
        model_hash=composite_hash,
        parameter_set_id="ps_001",
        action_contract_hash="sha256:" + "0" * 64,
        input_hash="sha256:" + "1" * 64,
        executed_at="2026-06-01T12:00:00+00:00",
        executor_kind="simulate_only",
        ok=True,
        response_summary="ok",
        error=None,
        latency_ms=0.0,
        hmac_signature=None,
    )
    key = "a" * 64
    sig = sign_action_trace(trace, key)
    trace_signed = ActionTrace(
        report_id=trace.report_id,
        model_hash=trace.model_hash,
        parameter_set_id=trace.parameter_set_id,
        action_contract_hash=trace.action_contract_hash,
        input_hash=trace.input_hash,
        executed_at=trace.executed_at,
        executor_kind=trace.executor_kind,
        ok=trace.ok,
        response_summary=trace.response_summary,
        error=trace.error,
        latency_ms=trace.latency_ms,
        hmac_signature=sig,
    )
    assert verify_action_trace(trace_signed, key) is True
    assert trace_signed.model_hash == composite_hash


# ── audit shows full chain ────────────────────────────────────────────────────

def test_audit_action_shows_full_component_chain(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64))
    program = _chain_prog("Enc", "Router")
    chain = get_component_chain(program, reg)
    aliases = [e.alias for e in chain]
    assert "Enc" in aliases
    assert "Router" in aliases
    for e in chain:
        assert e.entry_hash.startswith("sha256:")


def test_component_chain_includes_mode_info(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _single_prog("Enc")
    chain = get_component_chain(program, reg)
    assert chain[0].mode == "FROZEN"


# ── composite dry run ─────────────────────────────────────────────────────────

def test_composite_dry_run_resolves_all_imports(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _single_prog("Enc")
    result = composite_dry_run(program, reg, input_data={"x": 1.0})
    assert result["ok"] is True
    assert len(result["resolved_imports"]) == 1
    imp = result["resolved_imports"][0]
    assert imp["alias"] == "Enc"
    assert imp["resolved_entry_hash"].startswith("sha256:")


def test_composite_dry_run_fails_for_missing_import(tmp_path):
    reg = ModelRegistry(tmp_path)
    program = _single_prog("Missing")
    result = composite_dry_run(program, reg, input_data={})
    assert not result["ok"]
    assert any("Missing" in e or "nonexistent" in e.lower() or "enc" in e.lower()
               for e in result["errors"])


def test_composite_dry_run_includes_composite_hash(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _single_prog("Enc")
    result = composite_dry_run(program, reg, input_data={})
    assert result["composite_model_hash"].startswith("sha256:")
