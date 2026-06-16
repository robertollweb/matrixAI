# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Key history store for signing-key rotation (PR4-C4).

Maintains a JSON file that records every signing key ever used, indexed by
purpose ('action' or 'registry').  When a key is rotated, the retiring entry
is stamped with ``rotated_at``; the key value is kept so that old
signatures can still be verified.

Threat model (inherited from P21): integrity on a trusted host.  An attacker
with filesystem read access could read key values; that is out of scope.  The
store protects against *accidental* signature invalidity after rotation, not
against a malicious operator who controls the disk.

Typical usage
-------------
    store = KeyStore.load(registry_path / ".matrixai_key_history.json")

    # On rotation: retire the old key before switching to the new one
    store.retire_current(old_key, purpose="action")

    # Verify an old ActionTrace whose signing key is unknown
    verified = store.try_verify_action(trace, current_key)

    # Verify a registry entry whose fingerprint is known
    verified = store.try_verify_registry(entry_hash, signature, fingerprint, current_key)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from matrixai.registry.signing import signing_key_fingerprint as key_fingerprint  # noqa: F401


_DEFAULT_FILENAME = ".matrixai_key_history.json"
_ENV_PATH_VAR = "MATRIXAI_KEY_HISTORY_PATH"


@dataclass
class KeyEntry:
    fingerprint: str   # sha256:<16hex>
    key: str           # raw key value (hex or passphrase)
    purpose: str       # "action" | "registry"
    added_at: str      # ISO8601 UTC — when this key was first recorded
    rotated_at: str | None  # ISO8601 UTC — when retired; None = still active

    @property
    def is_active(self) -> bool:
        return self.rotated_at is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KeyEntry":
        return cls(
            fingerprint=d["fingerprint"],
            key=d["key"],
            purpose=d["purpose"],
            added_at=d["added_at"],
            rotated_at=d.get("rotated_at"),
        )


class KeyStore:
    """JSON-backed history of signing keys, keyed by fingerprint."""

    def __init__(self, path: Path, entries: list[KeyEntry] | None = None):
        self.path = path
        self._entries: list[KeyEntry] = entries or []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "KeyStore":
        """Load from *path*, creating an empty store if the file does not exist."""
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            entries = [KeyEntry.from_dict(e) for e in raw]
        else:
            entries = []
        return cls(path=path, entries=entries)

    @classmethod
    def default_path(cls, registry_path: Path | None = None) -> Path:
        """Return the key history path from env var or registry dir."""
        env = os.environ.get(_ENV_PATH_VAR)
        if env:
            return Path(env)
        if registry_path is not None:
            return registry_path / _DEFAULT_FILENAME
        return Path(_DEFAULT_FILENAME)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def record(self, key: str, purpose: str) -> str:
        """Add *key* to the history if not already present. Returns fingerprint.

        If the key is already recorded under a different purpose, the purpose
        is updated so keys_for_purpose() finds it under the new role.
        """
        fp = key_fingerprint(key)
        for e in self._entries:
            if e.fingerprint == fp:
                if e.purpose != purpose:
                    e.purpose = purpose
                    self._save()
                return fp
        self._entries.append(KeyEntry(
            fingerprint=fp,
            key=key,
            purpose=purpose,
            added_at=_now_iso(),
            rotated_at=None,
        ))
        self._save()
        return fp

    def retire(self, key: str, purpose: str) -> str:
        """Record *key* and mark it as rotated-at-now. Returns fingerprint.

        Call this *before* switching to the new key so the old key is
        preserved for historical verification.
        """
        fp = self.record(key, purpose)
        for e in self._entries:
            if e.fingerprint == fp and e.rotated_at is None:
                e.rotated_at = _now_iso()
                self._save()
                break
        return fp

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def find_by_fingerprint(self, fingerprint: str) -> KeyEntry | None:
        for e in self._entries:
            if e.fingerprint == fingerprint:
                return e
        return None

    def keys_for_purpose(self, purpose: str) -> list[str]:
        """All key values for *purpose* (active + retired), oldest first."""
        return [e.key for e in self._entries if e.purpose == purpose]

    def list_entries(self) -> list[KeyEntry]:
        return list(self._entries)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def try_verify_action(self, trace: object, current_key: str | None) -> bool:
        """Try to verify *trace* with *current_key* then every stored action key.

        Returns True on first successful verification.  Returns False if no key
        matches or if the trace has no signature.
        """
        from matrixai.actions.trace import verify_action_trace

        candidates: list[str] = []
        if current_key:
            candidates.append(current_key)
        candidates += [k for k in self.keys_for_purpose("action") if k not in candidates]

        for key in candidates:
            if verify_action_trace(trace, key):
                return True
        return False

    def try_verify_registry_entry(
        self,
        entry_hash: str,
        signature: str,
        fingerprint: str,
        current_key: str | None,
    ) -> bool:
        """Verify a registry entry signature.

        Tries to look up the key by *fingerprint* first (efficient O(n)).
        Falls back to trying all registry keys if fingerprint is missing or
        not found in the store.
        """
        from matrixai.registry.signing import verify_entry_signature

        # Fast path: fingerprint-directed lookup.
        # If HMAC fails despite a fingerprint match (e.g. truncation collision),
        # fall through to the slow path rather than returning False immediately.
        if fingerprint:
            entry = self.find_by_fingerprint(fingerprint)
            if entry and verify_entry_signature(entry_hash, signature, entry.key):
                return True

        # Slow path: try current key then all stored registry keys
        candidates: list[str] = []
        if current_key:
            candidates.append(current_key)
        candidates += [k for k in self.keys_for_purpose("registry") if k not in candidates]

        for key in candidates:
            if verify_entry_signature(entry_hash, signature, key):
                return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([e.to_dict() for e in self._entries], indent=2),
            encoding="utf-8",
        )
        self.path.chmod(0o600)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
