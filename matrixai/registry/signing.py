# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import hashlib
import hmac as _hmac
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

_ENV_KEY_VAR = "MATRIXAI_REGISTRY_SIGNING_KEY"
_LOCAL_KEY_FILE = ".registry_signing_key"
_HMAC_PREFIX = "hmac-sha256:"


def get_signing_key(registry_path: Path | None = None) -> str:
    """Return the active signing key.

    Priority:
    1. ``MATRIXAI_REGISTRY_SIGNING_KEY`` environment variable.
    2. Local key file inside ``registry_path`` (created on first use, perms 0600).
    3. Empty string when ``registry_path`` is also None (unsigned mode).
    """
    env_key = os.environ.get(_ENV_KEY_VAR)
    if env_key:
        return env_key
    if registry_path is not None:
        return _get_or_create_local_key(registry_path)
    return ""


def _get_or_create_local_key(registry_path: Path) -> str:
    key_path = registry_path / _LOCAL_KEY_FILE
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    key_path.write_text(key, encoding="utf-8")
    key_path.chmod(0o600)
    return key


def sign_entry_hash(entry_hash: str, signing_key: str) -> str:
    """Return 'hmac-sha256:<hex>' HMAC of entry_hash under signing_key."""
    key_bytes = signing_key.encode("utf-8")
    msg = entry_hash.encode("utf-8")
    digest = _hmac.new(key_bytes, msg, hashlib.sha256).hexdigest()
    return f"{_HMAC_PREFIX}{digest}"


def verify_entry_signature(entry_hash: str, signature: str, signing_key: str) -> bool:
    """Return True iff signature matches HMAC(signing_key, entry_hash). Constant-time."""
    if not signature:
        return False
    expected = sign_entry_hash(entry_hash, signing_key)
    return _hmac.compare_digest(expected, signature)


def signing_key_fingerprint(key: str) -> str:
    """Return a short 'sha256:<16hex>' fingerprint of the signing key."""
    return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def build_signature_record(entry_hash: str, signing_key: str) -> dict:
    """Return the dict that goes into ``signature.json``."""
    return {
        "entry_hash": entry_hash,
        "signature": sign_entry_hash(entry_hash, signing_key),
        "signed_at": datetime.now(tz=timezone.utc).isoformat(),
        "signing_key_fingerprint": signing_key_fingerprint(signing_key),
    }


def verify_entry_signature_with_keystore(
    entry_hash: str,
    signature: str,
    fingerprint: str,
    current_key: str | None,
    keystore: "KeyStore",
) -> bool:
    """Verify a registry entry against *current_key* and all historical registry keys.

    Uses the stored *fingerprint* for an efficient O(n) lookup when possible.
    Falls back to trying all known registry keys if the fingerprint is not in
    the store (e.g. the store is new or the entry predates rotation tracking).
    """
    return keystore.try_verify_registry_entry(entry_hash, signature, fingerprint, current_key)


if TYPE_CHECKING:
    from matrixai.signing.keystore import KeyStore
