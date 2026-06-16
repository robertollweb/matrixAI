"""PR4-C4 — Signing key rotation tests.

Covers:
- KeyStore: record, retire, find_by_fingerprint, keys_for_purpose, list_entries
- KeyStore: persistence (save/load round-trip)
- KeyStore: try_verify_action with current key and historical keys
- KeyStore: try_verify_registry_entry with fingerprint lookup and fallback
- verify_action_trace_with_keystore (trace.py helper)
- verify_entry_signature_with_keystore (registry/signing.py helper)
- CLI: matrixai keys rotate records key and marks it rotated
- CLI: matrixai keys list shows history
- Guarantee: old ActionTrace verifies after key rotation
- Guarantee: old registry entry verifies after key rotation
- Guarantee: lost key makes trace unverifiable (no false positive)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from matrixai.signing.keystore import KeyStore, KeyEntry, key_fingerprint


OLD_KEY = "a" * 64      # 64 hex chars = 32 bytes
NEW_KEY = "b" * 64
REG_KEY = "c" * 64


# ---------------------------------------------------------------------------
# key_fingerprint
# ---------------------------------------------------------------------------

class TestKeyFingerprint:
    def test_returns_sha256_prefix(self):
        fp = key_fingerprint(OLD_KEY)
        assert fp.startswith("sha256:")

    def test_length_is_7_plus_16(self):
        fp = key_fingerprint(OLD_KEY)
        assert len(fp) == 7 + 16  # "sha256:" + 16 hex chars

    def test_different_keys_different_fingerprints(self):
        assert key_fingerprint(OLD_KEY) != key_fingerprint(NEW_KEY)

    def test_same_key_same_fingerprint(self):
        assert key_fingerprint(OLD_KEY) == key_fingerprint(OLD_KEY)


# ---------------------------------------------------------------------------
# KeyStore — in-memory operations
# ---------------------------------------------------------------------------

class TestKeyStoreRecord:
    @pytest.fixture()
    def store(self, tmp_path):
        return KeyStore.load(tmp_path / "key_history.json")

    def test_record_adds_entry(self, store):
        store.record(OLD_KEY, "action")
        assert len(store.list_entries()) == 1

    def test_record_returns_fingerprint(self, store):
        fp = store.record(OLD_KEY, "action")
        assert fp == key_fingerprint(OLD_KEY)

    def test_record_is_idempotent(self, store):
        store.record(OLD_KEY, "action")
        store.record(OLD_KEY, "action")
        assert len(store.list_entries()) == 1

    def test_record_entry_is_active(self, store):
        store.record(OLD_KEY, "action")
        entry = store.list_entries()[0]
        assert entry.is_active
        assert entry.rotated_at is None

    def test_record_sets_purpose(self, store):
        store.record(OLD_KEY, "registry")
        assert store.list_entries()[0].purpose == "registry"


class TestKeyStoreRetire:
    @pytest.fixture()
    def store(self, tmp_path):
        return KeyStore.load(tmp_path / "key_history.json")

    def test_retire_records_key(self, store):
        store.retire(OLD_KEY, "action")
        assert len(store.list_entries()) == 1

    def test_retire_sets_rotated_at(self, store):
        store.retire(OLD_KEY, "action")
        entry = store.list_entries()[0]
        assert entry.rotated_at is not None
        assert not entry.is_active

    def test_retire_already_recorded_key(self, store):
        store.record(OLD_KEY, "action")
        store.retire(OLD_KEY, "action")
        assert len(store.list_entries()) == 1
        assert not store.list_entries()[0].is_active

    def test_retire_returns_fingerprint(self, store):
        fp = store.retire(OLD_KEY, "action")
        assert fp == key_fingerprint(OLD_KEY)


class TestKeyStoreLookup:
    @pytest.fixture()
    def store(self, tmp_path):
        s = KeyStore.load(tmp_path / "key_history.json")
        s.record(OLD_KEY, "action")
        s.record(NEW_KEY, "action")
        s.record(REG_KEY, "registry")
        return s

    def test_find_by_fingerprint(self, store):
        fp = key_fingerprint(OLD_KEY)
        entry = store.find_by_fingerprint(fp)
        assert entry is not None
        assert entry.key == OLD_KEY

    def test_find_by_fingerprint_missing(self, store):
        assert store.find_by_fingerprint("sha256:nonexistent") is None

    def test_keys_for_purpose_action(self, store):
        keys = store.keys_for_purpose("action")
        assert OLD_KEY in keys
        assert NEW_KEY in keys
        assert REG_KEY not in keys

    def test_keys_for_purpose_registry(self, store):
        keys = store.keys_for_purpose("registry")
        assert REG_KEY in keys
        assert OLD_KEY not in keys


# ---------------------------------------------------------------------------
# KeyStore — persistence
# ---------------------------------------------------------------------------

class TestKeyStorePersistence:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "history.json"
        store = KeyStore.load(path)
        store.record(OLD_KEY, "action")
        store.retire(NEW_KEY, "registry")

        reloaded = KeyStore.load(path)
        entries = reloaded.list_entries()
        assert len(entries) == 2
        assert any(e.key == OLD_KEY for e in entries)
        assert any(e.key == NEW_KEY and e.rotated_at is not None for e in entries)

    def test_history_file_permissions(self, tmp_path):
        path = tmp_path / "history.json"
        store = KeyStore.load(path)
        store.record(OLD_KEY, "action")
        mode = oct(path.stat().st_mode)[-3:]
        assert mode == "600"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        store = KeyStore.load(tmp_path / "nonexistent.json")
        assert store.list_entries() == []

    def test_json_is_valid(self, tmp_path):
        path = tmp_path / "history.json"
        store = KeyStore.load(path)
        store.record(OLD_KEY, "action")
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert data[0]["fingerprint"].startswith("sha256:")


# ---------------------------------------------------------------------------
# ActionTrace verification with keystore
# ---------------------------------------------------------------------------

class TestVerifyActionTraceWithKeystore:
    @pytest.fixture()
    def signed_trace(self):
        from matrixai.actions.trace import ActionTrace, sign_action_trace
        trace = ActionTrace(
            report_id="rpt-001",
            model_hash="mxai_abc",
            parameter_set_id="ps-001",
            action_contract_hash="contract-hash",
            input_hash="input-hash",
            executed_at="2026-05-27T12:00:00+00:00",
            executor_kind="ActionExecutor",
            ok=True,
            response_summary="ok",
            error=None,
            latency_ms=1.5,
            hmac_signature=None,
        )
        trace.hmac_signature = sign_action_trace(trace, OLD_KEY)
        return trace

    def test_verifies_with_current_key(self, signed_trace, tmp_path):
        store = KeyStore.load(tmp_path / "h.json")
        assert store.try_verify_action(signed_trace, OLD_KEY) is True

    def test_fails_with_wrong_key(self, signed_trace, tmp_path):
        store = KeyStore.load(tmp_path / "h.json")
        assert store.try_verify_action(signed_trace, NEW_KEY) is False

    def test_verifies_with_historical_key_after_rotation(self, signed_trace, tmp_path):
        store = KeyStore.load(tmp_path / "h.json")
        # OLD_KEY was the signing key; rotate to NEW_KEY
        store.retire(OLD_KEY, "action")
        store.record(NEW_KEY, "action")
        # Now current key is NEW_KEY but trace was signed with OLD_KEY
        assert store.try_verify_action(signed_trace, NEW_KEY) is True

    def test_fails_when_key_not_in_history(self, signed_trace, tmp_path):
        store = KeyStore.load(tmp_path / "h.json")
        # Neither current key nor history contains OLD_KEY
        store.record(NEW_KEY, "action")
        assert store.try_verify_action(signed_trace, NEW_KEY) is False

    def test_verify_action_trace_with_keystore_helper(self, signed_trace, tmp_path):
        from matrixai.actions.trace import verify_action_trace_with_keystore
        store = KeyStore.load(tmp_path / "h.json")
        store.retire(OLD_KEY, "action")
        assert verify_action_trace_with_keystore(signed_trace, NEW_KEY, store) is True

    def test_unsigned_trace_returns_false(self, tmp_path):
        from matrixai.actions.trace import ActionTrace
        trace = ActionTrace(
            report_id="r", model_hash="m", parameter_set_id="p",
            action_contract_hash="c", input_hash="i",
            executed_at="2026-01-01T00:00:00+00:00",
            executor_kind="ActionExecutor", ok=True, response_summary="",
            error=None, latency_ms=0.0, hmac_signature=None,
        )
        store = KeyStore.load(tmp_path / "h.json")
        store.record(OLD_KEY, "action")
        assert store.try_verify_action(trace, OLD_KEY) is False


# ---------------------------------------------------------------------------
# Registry entry verification with keystore
# ---------------------------------------------------------------------------

class TestVerifyRegistryEntryWithKeystore:
    @pytest.fixture()
    def signed_entry(self):
        from matrixai.registry.signing import sign_entry_hash, signing_key_fingerprint, build_signature_record
        entry_hash = "sha256:aaabbbccc"
        rec = build_signature_record(entry_hash, REG_KEY)
        return entry_hash, rec["signature"], rec["signing_key_fingerprint"]

    def test_verifies_with_current_key(self, signed_entry, tmp_path):
        entry_hash, signature, fingerprint = signed_entry
        store = KeyStore.load(tmp_path / "h.json")
        assert store.try_verify_registry_entry(entry_hash, signature, fingerprint, REG_KEY) is True

    def test_verifies_via_fingerprint_lookup(self, signed_entry, tmp_path):
        entry_hash, signature, fingerprint = signed_entry
        store = KeyStore.load(tmp_path / "h.json")
        store.retire(REG_KEY, "registry")
        # Current key is now different; fingerprint lookup finds REG_KEY in history
        assert store.try_verify_registry_entry(entry_hash, signature, fingerprint, NEW_KEY) is True

    def test_fails_with_wrong_key_and_empty_store(self, signed_entry, tmp_path):
        entry_hash, signature, fingerprint = signed_entry
        store = KeyStore.load(tmp_path / "h.json")
        assert store.try_verify_registry_entry(entry_hash, signature, fingerprint, NEW_KEY) is False

    def test_verify_entry_signature_with_keystore_helper(self, signed_entry, tmp_path):
        from matrixai.registry.signing import verify_entry_signature_with_keystore
        entry_hash, signature, fingerprint = signed_entry
        store = KeyStore.load(tmp_path / "h.json")
        store.retire(REG_KEY, "registry")
        assert verify_entry_signature_with_keystore(
            entry_hash, signature, fingerprint, NEW_KEY, store
        ) is True


# ---------------------------------------------------------------------------
# CLI: matrixai keys rotate
# ---------------------------------------------------------------------------

class TestCLIKeysRotate:
    def _run(self, *args, env=None):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "keys", *args],
            capture_output=True, text=True, env={**os.environ, **(env or {})},
        )
        return result

    def test_rotate_records_key_in_history(self, tmp_path):
        history = tmp_path / "reg" / ".matrixai_key_history.json"
        r = self._run(
            "rotate", "--purpose", "action",
            "--registry-path", str(tmp_path / "reg"),
            env={"MATRIXAI_ACTION_SIGNING_KEY": OLD_KEY},
        )
        assert r.returncode == 0, r.stderr
        assert history.exists()
        data = json.loads(history.read_text())
        assert any(e["key"] == OLD_KEY for e in data)

    def test_rotate_marks_key_as_retired(self, tmp_path):
        r = self._run(
            "rotate", "--purpose", "action",
            "--registry-path", str(tmp_path / "reg"),
            env={"MATRIXAI_ACTION_SIGNING_KEY": OLD_KEY},
        )
        assert r.returncode == 0
        data = json.loads((tmp_path / "reg" / ".matrixai_key_history.json").read_text())
        entry = next(e for e in data if e["key"] == OLD_KEY)
        assert entry["rotated_at"] is not None

    def test_rotate_prints_next_steps(self, tmp_path):
        r = self._run(
            "rotate", "--purpose", "action",
            "--registry-path", str(tmp_path / "reg"),
            env={"MATRIXAI_ACTION_SIGNING_KEY": OLD_KEY},
        )
        assert "openssl rand -hex 32" in r.stdout
        assert "MATRIXAI_ACTION_SIGNING_KEY" in r.stdout

    def test_rotate_without_key_fails(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "MATRIXAI_ACTION_SIGNING_KEY"}
        r = self._run(
            "rotate", "--purpose", "action",
            "--registry-path", str(tmp_path / "reg"),
            env=env,
        )
        assert r.returncode == 1

    def test_rotate_explicit_key_flag(self, tmp_path):
        r = self._run(
            "rotate", "--purpose", "action",
            "--key", OLD_KEY,
            "--registry-path", str(tmp_path / "reg"),
        )
        assert r.returncode == 0
        data = json.loads((tmp_path / "reg" / ".matrixai_key_history.json").read_text())
        assert any(e["key"] == OLD_KEY for e in data)


# ---------------------------------------------------------------------------
# CLI: matrixai keys list
# ---------------------------------------------------------------------------

class TestCLIKeysList:
    def _run(self, *args, env=None):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "keys", *args],
            capture_output=True, text=True, env={**os.environ, **(env or {})},
        )
        return result

    def _setup_history(self, tmp_path):
        reg = tmp_path / "reg"
        reg.mkdir()
        store = KeyStore.load(reg / ".matrixai_key_history.json")
        store.retire(OLD_KEY, "action")
        store.record(NEW_KEY, "action")
        return reg

    def test_list_shows_fingerprints(self, tmp_path):
        reg = self._setup_history(tmp_path)
        r = self._run("list", "--registry-path", str(reg))
        assert r.returncode == 0
        assert key_fingerprint(OLD_KEY) in r.stdout
        assert key_fingerprint(NEW_KEY) in r.stdout

    def test_list_shows_retired_status(self, tmp_path):
        reg = self._setup_history(tmp_path)
        r = self._run("list", "--registry-path", str(reg))
        assert "retired" in r.stdout
        assert "active" in r.stdout

    def test_list_json_output(self, tmp_path):
        reg = self._setup_history(tmp_path)
        r = self._run("list", "--registry-path", str(reg), "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_list_empty_store(self, tmp_path):
        reg = tmp_path / "reg"
        reg.mkdir()
        r = self._run("list", "--registry-path", str(reg))
        assert r.returncode == 0
        assert "No keys recorded" in r.stdout


# ---------------------------------------------------------------------------
# default_path
# ---------------------------------------------------------------------------

class TestKeyStoreDefaultPath:
    def test_uses_env_var_when_set(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_history.json"
        monkeypatch.setenv("MATRIXAI_KEY_HISTORY_PATH", str(custom))
        path = KeyStore.default_path()
        assert path == custom

    def test_uses_registry_path_when_given(self, tmp_path):
        path = KeyStore.default_path(tmp_path / "registry")
        assert path == tmp_path / "registry" / ".matrixai_key_history.json"
