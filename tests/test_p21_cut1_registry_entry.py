"""P21 C1 — RegistryEntry, entry_hash determinista, firma HMAC, RegistryLayout."""
from __future__ import annotations

import stat
import tempfile
from pathlib import Path

import pytest

from matrixai.registry import (
    RegistryEntry,
    RegistryEntryError,
    RegistryLayout,
    build_signature_record,
    compute_entry_hash,
    get_signing_key,
    sha256_bytes,
    sha256_str,
    sign_entry_hash,
    verify_entry_signature,
)

# ── shared fixtures ───────────────────────────────────────────────────────────

_KEY = "a" * 64

_HASH_KWARGS = dict(
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
    kw = {**_HASH_KWARGS, **overrides}
    eh = compute_entry_hash(**kw)
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


# ── entry_hash ────────────────────────────────────────────────────────────────

def test_entry_hash_is_deterministic():
    assert compute_entry_hash(**_HASH_KWARGS) == compute_entry_hash(**_HASH_KWARGS)


def test_entry_hash_is_sha256_prefixed():
    assert compute_entry_hash(**_HASH_KWARGS).startswith("sha256:")


def test_entry_hash_changes_when_model_hash_changes():
    h1 = compute_entry_hash(**_HASH_KWARGS)
    h2 = compute_entry_hash(**{**_HASH_KWARGS, "model_hash": "sha256:" + "f" * 64})
    assert h1 != h2


def test_entry_hash_changes_when_evaluation_report_changes():
    h1 = compute_entry_hash(**_HASH_KWARGS)
    h2 = compute_entry_hash(**{**_HASH_KWARGS, "evaluation_report_hash": "sha256:" + "z" * 64})
    assert h1 != h2


def test_entry_hash_changes_when_training_trace_changes():
    h1 = compute_entry_hash(**_HASH_KWARGS)
    h2 = compute_entry_hash(**{**_HASH_KWARGS, "training_trace_hash": "sha256:" + "z" * 64})
    assert h1 != h2


def test_entry_hash_changes_when_version_changes():
    h1 = compute_entry_hash(**_HASH_KWARGS)
    h2 = compute_entry_hash(**{**_HASH_KWARGS, "version": "v2"})
    assert h1 != h2


def test_entry_hash_changes_when_name_changes():
    h1 = compute_entry_hash(**_HASH_KWARGS)
    h2 = compute_entry_hash(**{**_HASH_KWARGS, "name": "other_encoder"})
    assert h1 != h2


def test_entry_hash_stable_regardless_of_kwargs_order():
    """Canonical JSON sorts keys — kwarg order must not affect the result."""
    h1 = compute_entry_hash(**_HASH_KWARGS)
    h2 = compute_entry_hash(**dict(reversed(list(_HASH_KWARGS.items()))))
    assert h1 == h2


def test_sha256_str_and_sha256_bytes_agree():
    text = "hello registry"
    assert sha256_str(text) == sha256_bytes(text.encode("utf-8"))


# ── RegistryEntry ─────────────────────────────────────────────────────────────

def test_registry_entry_is_frozen():
    entry = _make_entry()
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        entry.name = "changed"  # type: ignore[misc]


def test_registry_entry_error_is_exception():
    with pytest.raises(RegistryEntryError):
        raise RegistryEntryError("corrupt entry")


def test_registry_entry_to_manifest_roundtrip():
    entry = _make_entry()
    restored = RegistryEntry.from_manifest(entry.to_manifest())
    assert restored == entry


def test_registry_entry_manifest_contains_required_keys():
    manifest = _make_entry().to_manifest()
    for key in ("name", "version", "entry_hash", "model_hash", "parameter_schema_hash",
                "parameter_set_id", "metrics", "matrixai_version", "created_at",
                "training_trace_hash", "evaluation_report_hash"):
        assert key in manifest, f"manifest missing {key!r}"


def test_registry_entry_from_manifest_defaults_optional_fields():
    minimal = {
        "name": "enc",
        "version": "v1",
        "entry_hash": "sha256:" + "0" * 64,
        "model_hash": "sha256:" + "0" * 64,
        "parameter_schema_hash": "sha256:" + "0" * 64,
        "parameter_set_id": "ps_x",
        "input_type": {},
        "output_type": {},
        "matrixai_version": "0.21.0",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    entry = RegistryEntry.from_manifest(minimal)
    assert entry.interpretability_level == "full"
    assert entry.training_trace_hash == ""
    assert entry.evaluation_report_hash == ""
    assert entry.training_dataset_fingerprint == ""


# ── signing ───────────────────────────────────────────────────────────────────

def test_sign_returns_hmac_sha256_prefix():
    h = compute_entry_hash(**_HASH_KWARGS)
    assert sign_entry_hash(h, _KEY).startswith("hmac-sha256:")


def test_verify_passes_for_valid_signature():
    h = compute_entry_hash(**_HASH_KWARGS)
    sig = sign_entry_hash(h, _KEY)
    assert verify_entry_signature(h, sig, _KEY) is True


def test_verify_fails_for_wrong_key():
    h = compute_entry_hash(**_HASH_KWARGS)
    sig = sign_entry_hash(h, _KEY)
    assert verify_entry_signature(h, sig, "b" * 64) is False


def test_verify_rejects_tampered_entry_hash():
    h = compute_entry_hash(**_HASH_KWARGS)
    sig = sign_entry_hash(h, _KEY)
    assert verify_entry_signature("sha256:" + "0" * 64, sig, _KEY) is False


def test_verify_rejects_empty_signature():
    h = compute_entry_hash(**_HASH_KWARGS)
    assert verify_entry_signature(h, "", _KEY) is False


def test_build_signature_record_shape():
    h = compute_entry_hash(**_HASH_KWARGS)
    rec = build_signature_record(h, _KEY)
    assert rec["entry_hash"] == h
    assert rec["signature"].startswith("hmac-sha256:")
    assert "signed_at" in rec
    assert rec["signing_key_fingerprint"].startswith("sha256:")


def test_sign_verify_roundtrip_is_consistent():
    h = compute_entry_hash(**_HASH_KWARGS)
    sig = sign_entry_hash(h, _KEY)
    assert verify_entry_signature(h, sig, _KEY)
    assert not verify_entry_signature(h, sig + "x", _KEY)


# ── key management ────────────────────────────────────────────────────────────

def test_env_key_takes_precedence_over_local(monkeypatch, tmp_path):
    monkeypatch.setenv("MATRIXAI_REGISTRY_SIGNING_KEY", "env_key_value")
    assert get_signing_key(registry_path=tmp_path) == "env_key_value"


def test_local_key_file_creates_with_restricted_permissions(monkeypatch, tmp_path):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    get_signing_key(registry_path=tmp_path)
    key_path = tmp_path / ".registry_signing_key"
    assert key_path.exists()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600


def test_local_key_file_reuses_existing(monkeypatch, tmp_path):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    k1 = get_signing_key(registry_path=tmp_path)
    k2 = get_signing_key(registry_path=tmp_path)
    assert k1 == k2


def test_get_signing_key_returns_empty_when_no_path_and_no_env(monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    assert get_signing_key(registry_path=None) == ""


# ── RegistryLayout ────────────────────────────────────────────────────────────

def test_registry_layout_index_path(tmp_path):
    layout = RegistryLayout(tmp_path)
    assert layout.index_path == tmp_path / "registry.json"


def test_registry_layout_entries_dir(tmp_path):
    layout = RegistryLayout(tmp_path)
    assert layout.entries_dir == tmp_path / "entries"


def test_registry_layout_tags_dir(tmp_path):
    layout = RegistryLayout(tmp_path)
    assert layout.tags_dir == tmp_path / "tags"


def test_registry_layout_entry_dir(tmp_path):
    layout = RegistryLayout(tmp_path)
    assert layout.entry_dir("text_encoder", "v1") == tmp_path / "entries" / "text_encoder" / "v1"


def test_registry_layout_entry_file_manifest(tmp_path):
    layout = RegistryLayout(tmp_path)
    path = layout.entry_file("enc", "v1", "manifest")
    assert path == tmp_path / "entries" / "enc" / "v1" / "manifest.json"


def test_registry_layout_entry_file_model(tmp_path):
    layout = RegistryLayout(tmp_path)
    assert layout.entry_file("enc", "v1", "model").name == "model.mxai"


def test_registry_layout_entry_file_unknown_raises(tmp_path):
    layout = RegistryLayout(tmp_path)
    with pytest.raises(KeyError):
        layout.entry_file("enc", "v1", "nonexistent_key")


def test_registry_layout_tag_path(tmp_path):
    layout = RegistryLayout(tmp_path)
    assert layout.tag_path("enc", "latest") == tmp_path / "tags" / "enc" / "latest"
