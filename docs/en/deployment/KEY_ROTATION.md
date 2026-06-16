# PR4-C4 — Signing Key Management and Rotation

> **Español:** [docs/es/deployment/KEY_ROTATION.md](../../es/deployment/KEY_ROTATION.md)

MatrixAI uses two independent HMAC-SHA256 signing keys:

| Key | Environment variable | Protects |
|---|---|---|
| Action key | `MATRIXAI_ACTION_SIGNING_KEY` | `ActionTrace` signatures (P20) |
| Registry key | `MATRIXAI_REGISTRY_SIGNING_KEY` | Registry entry signatures (P21) |

**Critical guarantee:** rotating a key never invalidates traces or registry entries signed with the previous key. The key history file preserves all retired keys so historical verification remains possible indefinitely.

---

## Generating a secure key

```bash
openssl rand -hex 32
# e.g.: 4a7f3b2c9e1d6f8a...  (64 hex characters = 256 bits)
```

The minimum accepted length is 32 bytes (64 hex characters). Shorter values are accepted by the runtime but are not recommended for production.

---

## Initial setup

Set both keys in `.env` before first deployment:

```
MATRIXAI_ACTION_SIGNING_KEY=<openssl rand -hex 32>
MATRIXAI_REGISTRY_SIGNING_KEY=<openssl rand -hex 32>
```

If `MATRIXAI_REGISTRY_SIGNING_KEY` is not set, the registry generates a local key per container (stored as `matrixai_registry/.registry_signing_key`, permissions 0600). This local key is not persistent across container replacements — set the env var explicitly for production.

---

## Rotating a key

Key rotation is a three-step procedure:

### Step 1 — Record the retiring key in the history file

```bash
# Rotate the action signing key
matrixai keys rotate \
  --purpose action \
  --registry-path matrixai_registry

# Rotate the registry signing key
matrixai keys rotate \
  --purpose registry \
  --registry-path matrixai_registry
```

This stamps the current key with `rotated_at` in the history file (`matrixai_registry/.matrixai_key_history.json`). The key value is preserved for historical verification.

**Do this before changing the env var.** If you change the env var first and then record the old key, you must pass `--key <old-key-value>` explicitly.

### Step 2 — Set the new key

```bash
# Generate a new key
NEW_KEY=$(openssl rand -hex 32)
echo $NEW_KEY  # copy this value

# Update .env
# MATRIXAI_ACTION_SIGNING_KEY=<new value>
```

### Step 3 — Restart the server

```bash
docker compose restart
```

New signatures use the new key immediately. Historical traces remain verifiable.

---

## Verifying historical signatures

After rotation, old traces can still be verified two ways:

**1. Explicit key (when you know which key signed the trace):**

```bash
matrixai audit-action \
  --trace trace.json \
  --signing-key <old-key-value>
```

**2. Via key store (automatic fallback through all known keys):**

```python
from matrixai.signing.keystore import KeyStore
from matrixai.actions.trace import verify_action_trace_with_keystore

store = KeyStore.load(Path("matrixai_registry/.matrixai_key_history.json"))
verified = store.try_verify_action(trace, current_key)
```

For registry entries (which store a `signing_key_fingerprint`), the key store looks up the key by fingerprint directly:

```python
verified = store.try_verify_registry_entry(
    entry_hash, signature, fingerprint, current_key
)
```

---

## Listing the key history

```bash
matrixai keys list --registry-path matrixai_registry

# Key history: matrixai_registry/.matrixai_key_history.json
# Fingerprint              Purpose     Status      Added                        Rotated
# sha256:e861b2eab679927c  action      retired     2026-05-27T12:00:00+00:00    2026-05-27T13:00:00+00:00
# sha256:4a7f3b2c9e1d6f8a  action      active      2026-05-27T13:00:00+00:00    -

# JSON output
matrixai keys list --registry-path matrixai_registry --json
```

---

## What happens if a key is lost or compromised

### Key lost (not compromised)

If the key is lost but not compromised:

- Existing traces signed with that key **can no longer be verified**. The signature is intact but the verification key is gone.
- New signatures use the current key.
- **Mitigation:** Keep the key history file backed up alongside the registry. Both live in `matrixai_registry/` by default.

What is preserved even without the lost key:
- The `entry_hash` and `model_hash` chain — these are SHA256 hashes of the artifacts, not of the key. The integrity of the artifact contents can still be checked by recomputing the hash.
- The tamper detection via `registry verify` — this rehashes stored files against manifest hashes, independent of the signing key.

### Key compromised

If a signing key is compromised:

1. **Rotate immediately** (Step 1-3 above).
2. The compromised key is still in the history file. An attacker with the key could forge signatures — but **only for new forged artifacts**; they cannot retroactively change the `entry_hash` chain (which depends on artifact content, not the key).
3. Consider the compromised key "untrustworthy for new signatures" — existing traces signed before the compromise are unaffected in terms of content integrity.
4. If the history file is also at risk, delete the compromised key entry from it manually (edit the JSON and remove the entry with the compromised fingerprint).

---

## Key history file format

The history file is a JSON array at `matrixai_registry/.matrixai_key_history.json` (permissions 0600):

```json
[
  {
    "fingerprint": "sha256:e861b2eab679927c",
    "key": "abc123def456...",
    "purpose": "action",
    "added_at": "2026-05-27T12:00:00+00:00",
    "rotated_at": "2026-05-27T13:00:00+00:00"
  },
  {
    "fingerprint": "sha256:4a7f3b2c9e1d6f8a",
    "key": "newkeyvalue...",
    "purpose": "action",
    "added_at": "2026-05-27T13:00:00+00:00",
    "rotated_at": null
  }
]
```

`rotated_at: null` means the key is currently active. A non-null `rotated_at` means the key has been retired but is retained for historical verification.

Override the default path with `MATRIXAI_KEY_HISTORY_PATH=/path/to/key_history.json`.

---

## Security notes

- The history file stores key values in plaintext. Protect it as you would protect any secret: filesystem permissions 0600, excluded from version control, backed up encrypted.
- The threat model is integrity on a trusted host. An attacker with filesystem read access could read historical keys. If that threat is relevant, use a secrets manager (Vault, AWS Secrets Manager) and pass keys via env var only — never write them to the history file in that configuration.
- Fingerprints (`sha256:<16hex>`) are short digests of the key, safe to log and display. They do not reveal the key value.
