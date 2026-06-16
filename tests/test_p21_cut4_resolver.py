"""P21 C4 — CompositeModelResolver: import resolution against the registry."""
from __future__ import annotations

import pytest

from matrixai.ir.schema import ImportSpec, MatrixAIProgram
from matrixai.registry import (
    CompositeModelResolver,
    ImportResolutionError,
    ModelRegistry,
    RegistryEntry,
    compute_entry_hash,
)

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE = dict(
    name="text_encoder",
    version="v1",
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version="0.21.0",
)


def _make_entry(**overrides) -> RegistryEntry:
    kw = {**_BASE, **overrides}
    eh = compute_entry_hash(**{k: v for k, v in kw.items() if k in _BASE})
    return RegistryEntry(
        name=kw["name"],
        version=kw["version"],
        entry_hash=eh,
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


def _make_program(*imports: ImportSpec) -> MatrixAIProgram:
    return MatrixAIProgram(
        project="Test",
        imports=list(imports),
    )


def _imp(alias: str, name: str, version: str, mode: str = "FROZEN") -> ImportSpec:
    return ImportSpec(alias=alias, registry_name=name, version=version, mode=mode)


# ── resolve existing ──────────────────────────────────────────────────────────

def test_resolver_resolves_existing_import(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "text_encoder", "v1"))
    hydrated = resolver.resolve(program)
    assert len(hydrated) == 1
    assert hydrated[0].alias == "Enc"
    assert hydrated[0].registry_name == "text_encoder"
    assert hydrated[0].version == "v1"


def test_resolver_records_resolved_entry_hash(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry()
    reg.push(entry)
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "text_encoder", "v1"))
    hydrated = resolver.resolve(program)
    assert hydrated[0].resolved_entry_hash == entry.entry_hash
    assert hydrated[0].resolved_entry_hash.startswith("sha256:")


def test_resolver_sets_resolved_at_timestamp(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    resolver = CompositeModelResolver(reg)
    hydrated = resolver.resolve(_make_program(_imp("Enc", "text_encoder", "v1")))
    assert hydrated[0].resolved_at != ""
    assert "T" in hydrated[0].resolved_at  # ISO8601 format


def test_resolver_preserves_mode(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    resolver = CompositeModelResolver(reg)
    hydrated = resolver.resolve(_make_program(_imp("Enc", "text_encoder", "v1", "TRAINABLE")))
    assert hydrated[0].mode == "TRAINABLE"


def test_resolver_resolves_multiple_imports(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry(name="text_encoder", version="v1"))
    reg.push(_make_entry(name="sentiment", version="v1", model_hash="sha256:" + "f" * 64))
    resolver = CompositeModelResolver(reg)
    program = _make_program(
        _imp("Enc", "text_encoder", "v1"),
        _imp("Sent", "sentiment", "v1"),
    )
    hydrated = resolver.resolve(program)
    assert len(hydrated) == 2
    aliases = {h.alias for h in hydrated}
    assert aliases == {"Enc", "Sent"}


# ── missing entry ─────────────────────────────────────────────────────────────

def test_resolver_fails_on_missing_entry(tmp_path):
    reg = ModelRegistry(tmp_path)
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "nonexistent", "v1"))
    with pytest.raises(ImportResolutionError, match="nonexistent@v1"):
        resolver.resolve(program)


def test_resolver_fails_on_missing_version(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "text_encoder", "v99"))
    with pytest.raises(ImportResolutionError, match="v99"):
        resolver.resolve(program)


# ── mutable tags ──────────────────────────────────────────────────────────────

def test_resolver_warns_on_mutable_tag_without_flag(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    reg.tag("text_encoder", "v1", "latest")
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "text_encoder", "latest"))
    with pytest.raises(ImportResolutionError, match="mutable"):
        resolver.resolve(program)  # allow_mutable_tags defaults to False


def test_resolver_resolves_tag_alias(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry()
    reg.push(entry)
    reg.tag("text_encoder", "v1", "latest")
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "text_encoder", "latest"))
    with pytest.warns(UserWarning, match="mutable"):
        hydrated = resolver.resolve(program, allow_mutable_tags=True)
    assert hydrated[0].resolved_entry_hash == entry.entry_hash


def test_resolver_allows_mutable_tag_with_flag(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    reg.tag("text_encoder", "v1", "prod")
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "text_encoder", "prod"))
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        hydrated = resolver.resolve(program, allow_mutable_tags=True)
    assert len(hydrated) == 1


# ── caching ───────────────────────────────────────────────────────────────────

def test_resolver_caches_resolved_components(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    call_count = 0
    original_get = reg.get

    def counting_get(name, version):
        nonlocal call_count
        call_count += 1
        return original_get(name, version)

    reg.get = counting_get  # type: ignore[method-assign]
    resolver = CompositeModelResolver(reg)
    program = _make_program(_imp("Enc", "text_encoder", "v1"))
    resolver.resolve(program)
    resolver.resolve(program)
    assert call_count == 1  # cached after first call


def test_resolver_empty_program_returns_empty(tmp_path):
    reg = ModelRegistry(tmp_path)
    resolver = CompositeModelResolver(reg)
    program = _make_program()
    assert resolver.resolve(program) == []
