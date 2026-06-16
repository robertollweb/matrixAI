"""P21 C3 — ModelRegistry API: push, get, list, tag, verify, pull."""
from __future__ import annotations

import json

import pytest

from matrixai import __version__ as _MATRIXAI_VERSION
from matrixai.registry import (
    DuplicateEntryError,
    EntryNotFoundError,
    MATRIXAI_REGISTRY_SCHEMA_VERSION,
    ModelRegistry,
    ModelRegistryError,
    RegistryEntry,
    VerificationError,
    compute_entry_hash,
)

# ── shared helpers ────────────────────────────────────────────────────────────

_BASE_KWARGS = dict(
    name="text_encoder",
    version="v1",
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version=_MATRIXAI_VERSION,
)


def _make_entry(**overrides) -> RegistryEntry:
    kw = {**_BASE_KWARGS, **overrides}
    eh = compute_entry_hash(**{k: v for k, v in kw.items() if k in _BASE_KWARGS})
    return RegistryEntry(
        name=kw["name"],
        version=kw["version"],
        entry_hash=eh,
        model_hash=kw["model_hash"],
        parameter_schema_hash=kw["parameter_schema_hash"],
        parameter_set_id=kw["parameter_set_id"],
        input_type={"name": "RawText", "kind": "VECTOR"},
        output_type={"name": "embedding", "kind": "Tensor", "shape": [128]},
        metrics={"loss": 0.23, "accuracy": 0.91},
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="sha256:" + "e" * 64,
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
    )


def _make_entry_no_eval(**overrides) -> RegistryEntry:
    kw = {**_BASE_KWARGS, **overrides, "evaluation_report_hash": ""}
    eh = compute_entry_hash(**{k: v for k, v in kw.items() if k in _BASE_KWARGS})
    return RegistryEntry(
        name=kw["name"],
        version=kw["version"],
        entry_hash=eh,
        model_hash=kw["model_hash"],
        parameter_schema_hash=kw["parameter_schema_hash"],
        parameter_set_id=kw["parameter_set_id"],
        input_type={},
        output_type={},
        metrics={},
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash="",
    )


# ── structure ─────────────────────────────────────────────────────────────────

def test_registry_initializes_directory_structure(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    assert (tmp_path / "reg" / "entries").is_dir()
    assert (tmp_path / "reg" / "tags").is_dir()
    assert (tmp_path / "reg" / "registry.json").exists()


def test_registry_index_file_has_correct_format(tmp_path):
    ModelRegistry(tmp_path)
    data = json.loads((tmp_path / "registry.json").read_text())
    assert "entries" in data
    assert data["version"] == MATRIXAI_REGISTRY_SCHEMA_VERSION
    assert data["entries"] == []


# ── push ──────────────────────────────────────────────────────────────────────

def test_registry_push_writes_manifest(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry()
    reg.push(entry)
    manifest_path = tmp_path / "entries" / "text_encoder" / "v1" / "manifest.json"
    assert manifest_path.exists()


def test_registry_push_requires_evaluation_report(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry_no_eval()
    with pytest.raises(ModelRegistryError, match="evaluation_report_hash"):
        reg.push(entry)


def test_registry_push_fails_on_duplicate_version(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry()
    reg.push(entry)
    with pytest.raises(DuplicateEntryError):
        reg.push(entry)


def test_registry_is_append_only(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry()
    reg.push(entry)
    with pytest.raises(DuplicateEntryError, match="append-only"):
        reg.push(entry)


def test_registry_push_updates_index(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    data = json.loads((tmp_path / "registry.json").read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["name"] == "text_encoder"


def test_registry_push_signs_with_env_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIXAI_REGISTRY_SIGNING_KEY", "a" * 64)
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    sig_path = tmp_path / "entries" / "text_encoder" / "v1" / "signature.json"
    assert sig_path.exists()
    sig_data = json.loads(sig_path.read_text())
    assert sig_data["signature"].startswith("hmac-sha256:")


# ── get ───────────────────────────────────────────────────────────────────────

def test_registry_get_returns_full_entry(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry()
    reg.push(entry)
    fetched = reg.get("text_encoder", "v1")
    assert fetched == entry


def test_registry_get_raises_for_missing_entry(tmp_path):
    reg = ModelRegistry(tmp_path)
    with pytest.raises(EntryNotFoundError):
        reg.get("nonexistent", "v1")


# ── list ──────────────────────────────────────────────────────────────────────

def test_registry_list_empty_when_no_entries(tmp_path):
    reg = ModelRegistry(tmp_path)
    assert reg.list() == []


def test_registry_list_returns_all_entries(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry(name="text_encoder", version="v1"))
    reg.push(_make_entry(name="sentiment", version="v1",
                         model_hash="sha256:" + "f" * 64))
    entries = reg.list()
    assert len(entries) == 2


def test_registry_list_filters_by_name(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry(name="text_encoder", version="v1"))
    reg.push(_make_entry(name="sentiment", version="v1",
                         model_hash="sha256:" + "f" * 64))
    entries = reg.list(filters={"name": "text_encoder"})
    assert len(entries) == 1
    assert entries[0].name == "text_encoder"


# ── tag ───────────────────────────────────────────────────────────────────────

def test_registry_tag_creates_alias(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    reg.tag("text_encoder", "v1", "latest")
    tag_path = tmp_path / "tags" / "text_encoder" / "latest"
    assert tag_path.exists()


def test_registry_tag_moves_existing_alias(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry(version="v1"))
    reg.push(_make_entry(version="v2", model_hash="sha256:" + "f" * 64))
    reg.tag("text_encoder", "v1", "latest")
    reg.tag("text_encoder", "v2", "latest")
    tag_path = tmp_path / "tags" / "text_encoder" / "latest"
    data = json.loads(tag_path.read_text())
    assert data["version"] == "v2"


def test_registry_get_resolves_tag_to_version(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry()
    reg.push(entry)
    reg.tag("text_encoder", "v1", "latest")
    fetched = reg.get("text_encoder", "latest")
    assert fetched.version == "v1"
    assert fetched == entry


def test_registry_tag_points_to_correct_version(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry(version="v1"))
    reg.push(_make_entry(version="v2", model_hash="sha256:" + "f" * 64))
    reg.tag("text_encoder", "v2", "prod")
    fetched = reg.get("text_encoder", "prod")
    assert fetched.version == "v2"


# ── verify ────────────────────────────────────────────────────────────────────

def test_registry_verify_passes_for_valid_entry(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    assert reg.verify("text_encoder", "v1") is True


def test_registry_verify_rejects_tampered_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    manifest_path = tmp_path / "entries" / "text_encoder" / "v1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["model_hash"] = "sha256:" + "z" * 64
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(VerificationError):
        reg.verify("text_encoder", "v1")


def test_registry_verify_rejects_wrong_signature(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIXAI_REGISTRY_SIGNING_KEY", "a" * 64)
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry())
    sig_path = tmp_path / "entries" / "text_encoder" / "v1" / "signature.json"
    sig_data = json.loads(sig_path.read_text())
    sig_data["signature"] = "hmac-sha256:" + "0" * 64
    sig_path.write_text(json.dumps(sig_data))
    with pytest.raises(VerificationError, match="signature"):
        reg.verify("text_encoder", "v1")


# ── pull ──────────────────────────────────────────────────────────────────────

def test_registry_pull_copies_entry_between_registries(tmp_path):
    src = ModelRegistry(tmp_path / "src")
    dst = ModelRegistry(tmp_path / "dst")
    entry = _make_entry()
    src.push(entry)
    src.pull("text_encoder", "v1", dst)
    fetched = dst.get("text_encoder", "v1")
    assert fetched == entry


def test_registry_pull_copies_signature_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIXAI_REGISTRY_SIGNING_KEY", "a" * 64)
    src = ModelRegistry(tmp_path / "src")
    dst = ModelRegistry(tmp_path / "dst")
    src.push(_make_entry())
    src.pull("text_encoder", "v1", dst)
    sig_path = tmp_path / "dst" / "entries" / "text_encoder" / "v1" / "signature.json"
    assert sig_path.exists()


# ── verify: file tamper detection ─────────────────────────────────────────────

def _make_run_dir(tmp_path, name="classifier", version="v1", *, with_model=True, with_params=True, with_trace=True):
    """Build a minimal run directory for push_run_dir."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9, "loss": 0.1}))
    if with_model:
        (run_dir / f"{name}.mxai").write_bytes(b"fake-model-weights-v1")
    if with_params:
        (run_dir / "params.best.json").write_text(json.dumps({
            "parameter_set_id": f"{name}_{version}_ps",
            "parameter_schema_hash": "sha256:" + "b" * 64,
            "metrics": {"loss": 0.1, "accuracy": 0.9},
        }))
    if with_trace:
        (run_dir / "training_trace.json").write_text(json.dumps({"epochs": 10}))
    return run_dir


def test_verify_detects_tampered_params_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _make_run_dir(tmp_path)
    reg.push_run_dir(run_dir, "classifier", "v1")

    params_stored = tmp_path / "reg" / "entries" / "classifier" / "v1" / "params.json"
    data = json.loads(params_stored.read_text())
    data["metrics"]["accuracy"] = 0.999  # tamper: change a weight/metric
    params_stored.write_text(json.dumps(data))

    with pytest.raises(VerificationError, match="params.json"):
        reg.verify("classifier", "v1")


def test_verify_detects_tampered_model_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _make_run_dir(tmp_path)
    reg.push_run_dir(run_dir, "classifier", "v1")

    model_stored = tmp_path / "reg" / "entries" / "classifier" / "v1" / "model.mxai"
    model_stored.write_bytes(model_stored.read_bytes() + b"# TAMPERED")

    with pytest.raises(VerificationError, match="model.mxai"):
        reg.verify("classifier", "v1")


def test_verify_detects_tampered_evaluation_report(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _make_run_dir(tmp_path)
    reg.push_run_dir(run_dir, "classifier", "v1")

    eval_stored = tmp_path / "reg" / "entries" / "classifier" / "v1" / "evaluation_report.json"
    data = json.loads(eval_stored.read_text())
    data["accuracy"] = 0.999
    eval_stored.write_text(json.dumps(data))

    with pytest.raises(VerificationError, match="evaluation_report.json"):
        reg.verify("classifier", "v1")


def test_verify_detects_tampered_training_trace(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _make_run_dir(tmp_path)
    reg.push_run_dir(run_dir, "classifier", "v1")

    trace_stored = tmp_path / "reg" / "entries" / "classifier" / "v1" / "training_trace.json"
    data = json.loads(trace_stored.read_text())
    data["epochs"] = 999
    trace_stored.write_text(json.dumps(data))

    with pytest.raises(VerificationError, match="training_trace.json"):
        reg.verify("classifier", "v1")


def test_verify_passes_for_untampered_push_run_dir_entry(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _make_run_dir(tmp_path)
    reg.push_run_dir(run_dir, "classifier", "v1")
    assert reg.verify("classifier", "v1") is True


def test_push_run_dir_records_product_matrixai_version(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _make_run_dir(tmp_path)
    entry = reg.push_run_dir(run_dir, "classifier", "v1")
    assert entry.matrixai_version == _MATRIXAI_VERSION
