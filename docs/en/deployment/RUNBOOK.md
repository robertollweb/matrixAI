# PR4-C3 — Operational Runbook

> **Español:** [docs/es/deployment/RUNBOOK.md](../../es/deployment/RUNBOOK.md)

This runbook covers the six failure scenarios an operator may encounter running MatrixAI in production. Each scenario follows the same structure: **Symptom → Diagnosis → Action → Verification**. The operator should be able to resolve any scenario using only this document and the MatrixAI CLI, without contacting the author.

**Prerequisites:** `matrixai` CLI available; access to the registry directory and `.mxcontinual` policy file if applicable.

---

## Scenario 1 — Drift detected

### Symptom

One or more of:
- `/metrics` shows `matrixai_drift_degradation_detected{...} 1`
- `/metrics` shows `matrixai_drift_actual_degradation{...} > 0`
- The continual monitoring log reports a rollback trigger
- Accuracy in the sliding window has dropped below the configured threshold

```bash
curl -s http://localhost:8000/metrics | grep drift
# matrixai_drift_degradation_detected{project="AlertMonitor"} 1
# matrixai_drift_actual_degradation{project="AlertMonitor"} 0.12
```

### Diagnosis

The `ProductionMonitor` detected that accuracy in the current sliding window has fallen more than `DEGRADATION_THRESHOLD` below the reference accuracy. This triggers the drift policy condition.

Check the continual policy status to confirm the current and baseline versions:

```bash
matrixai continual status policy/alert_monitor.mxcontinual \
  --registry-dir registry/
# Registry       : alert-monitor
# Current version: v1.1  (ps=run_20260527_best)
# Base version   : v1.0  (ps=train_out_best)
# Rollback config: threshold=0.1  window=1h  min_samples=100
# Metrics (training):
#   accuracy: 0.91
# Drift status   : see GET /metrics (server must run with --continual-policy)
```

For live drift data (window accuracy, degradation detected), query the running server:

```bash
curl -s http://localhost:8000/metrics | grep matrixai_drift
# matrixai_drift_degradation_detected{project="AlertMonitor"} 1
# matrixai_drift_window_accuracy{project="AlertMonitor"} 0.71
# matrixai_drift_actual_degradation{project="AlertMonitor"} 0.12
```

### Action

> **Note:** The MatrixAI server does not execute rollbacks automatically. Rollback is always triggered manually via the CLI after the operator confirms drift in `/metrics`. An automated rollback pipeline can be built by scripting this command in response to the Prometheus alert.

1. Dry-run first to confirm what will be rolled back:

```bash
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/ \
  --dry-run
# Would roll back alert-monitor from v1.1 → v1.0
# from_ps: run_20260527_best  →  to_ps: train_out_best
```

2. Execute the rollback (requires `MATRIXAI_CONTINUAL_SIGNING_KEY` set, or pass `--signing-key`):

```bash
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/ \
  --json
```

3. Restart the server pointing to the rolled-back version:

```bash
# Stop the running container
docker compose stop

# Update .env to point to the rolled-back params if needed
# Then restart
docker compose up -d
```

### Verification

```bash
# Confirm rollback event was recorded
matrixai continual status policy/alert_monitor.mxcontinual --registry-dir registry/
# Registry       : alert-monitor
# Current version: v1.0  (ps=train_out_best)
# Last rollback  : v1.1 → v1.0  (manual)
#   executed_at  : 2026-05-28T12:34:56+00:00
#   event        : rollback_20260528T123456...

# Confirm /metrics no longer shows degradation
curl -s http://localhost:8000/metrics | grep drift_degradation_detected
# matrixai_drift_degradation_detected{...} 0
```

---

## Scenario 2 — Rollback triggered: verify it completed correctly

### Symptom

A rollback was triggered (automatically or manually). The operator needs to confirm it completed without errors and the correct parameter set is now active.

### Diagnosis

Rollback events are signed `RollbackEvent` objects persisted in the continual audit log. An incomplete rollback leaves the server on the old version.

Check the current version matches the rollback target:

```bash
matrixai continual status policy/alert_monitor.mxcontinual \
  --registry-dir registry/
# Current version: v1.0  (ps=train_out_best)   ← rolled back from v1.1
```

### Action

If `Current version` matches the rollback target, no action is needed — skip to Verification.

If the version has not changed or the command returns an error, see **Scenario 3**.

### Verification

```bash
# 1. Confirm the active registry version matches the rollback target
matrixai registry show alert-monitor@v1.0 --registry-path registry/
# parameter_set_id: train_out_best  ← matches expected target

# 2. Verify the rollback event signature
matrixai continual status policy/alert_monitor.mxcontinual \
  --registry-dir registry/ --json | python3 -m json.tool | grep signature
# "signature": "hmac-sha256:..."  ← non-empty means signed

# 3. Confirm the serving server loaded the correct params
curl -s -H "Authorization: Bearer $MATRIXAI_API_KEY" \
  http://localhost:8000/health | python3 -m json.tool
```

---

## Scenario 3 — Rollback failed

### Symptom

```
RollbackResult.ok = false
Error: rollback_failed — no previous version found
```
or
```
Error: rollback_failed — registry entry not found: alert-monitor@v0.9
```
or the rollback command exits non-zero.

### Diagnosis

Possible causes:

| Cause | Indicator |
|---|---|
| No previous version in registry | `registry list` shows only one version |
| Target version was deleted | `registry show alert-monitor@v0.9` fails |
| Policy base version doesn't match registry | `base_version` in policy not found |
| Signing key not set | `Error: MATRIXAI_CONTINUAL_SIGNING_KEY not set` |

```bash
matrixai registry list --registry-path registry/ --json
# Check: is there a rollback target version?
```

### Action

**Case A — No previous version available:**

The model has no earlier trained version in the registry. Options:

1. Re-train on stable historical data and push a new version:
```bash
matrixai train model.mxai --training train.mxtrain --output runs/recovery
matrixai registry push runs/recovery/ \
  --name alert-monitor --version v0.9 --registry-path registry/
```
2. Then re-run rollback.

**Case B — Signing key missing:**

```bash
export MATRIXAI_CONTINUAL_SIGNING_KEY=$(openssl rand -hex 32)
# Or load from your secrets manager
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/
```

**Case C — Policy base_version mismatch:**

Edit the `.mxcontinual` policy to set `BASE_VERSION` to an existing registry version, then retry.

### Verification

```bash
matrixai continual rollback policy/alert_monitor.mxcontinual \
  --registry-dir registry/ --dry-run
# Must print: "Would roll back ... → v<target>" without error
```

---

## Scenario 4 — Registry corrupted

### Symptom

```bash
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# FAIL: params.json content hash mismatch for alert-monitor@v1.0
```
or
```
FAIL: entry_hash mismatch
FAIL: signature invalid
```

### Diagnosis

The registry is append-only and cryptographically verified. Any mismatch means one of:

| Error | Cause |
|---|---|
| `params.json content hash mismatch` | `params.json` was edited directly on disk |
| `entry_hash mismatch` | `manifest.json` was modified after signing |
| `signature invalid` | Signing key rotated without re-signing, or file tampered |
| `model.mxai hash mismatch` | Model file replaced after registration |

Run full verification across all entries:

```bash
for entry in $(matrixai registry list --registry-path registry/ --json \
  | python3 -c "import sys,json; [print(e['name']+'@'+e['version']) for e in json.load(sys.stdin)]")
do
  matrixai registry verify --registry-path registry/ "$entry"
done
```

### Action

**Do not modify registry files manually** — this voids the audit chain.

**Recovery from backup (preferred):**

```bash
# Stop the server
docker compose stop

# Restore registry from the last known-good backup
cp -r /backup/registry-YYYYMMDD/ registry/

# Re-verify
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# OK: alert-monitor@v1.0 integrity verified
```

**Recovery by re-registering (if no backup):**

```bash
# Re-push the entry from the original training run artifacts
matrixai registry push runs/v1.0/ \
  --name alert-monitor \
  --version v1.0 \
  --registry-path registry/

# The existing corrupted entry must be removed first — stop the server,
# delete entries/alert-monitor/v1.0/, remove its index entry from
# registry/registry.json, then re-push.
```

### Verification

```bash
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# OK: alert-monitor@v1.0 integrity verified

# Restart the server
docker compose up -d
curl -s http://localhost:8000/health | python3 -m json.tool
# "status": "ok"
```

---

## Scenario 5 — Signing key lost or compromised

### Symptom

One of:
- The server fails to sign ActionTraces (`MATRIXAI_ACTION_SIGNING_KEY not set`)
- `matrixai audit-action` returns `signature_valid: false` for recent traces
- The key was accidentally deleted or the container was restarted without a persistent key

### Diagnosis

**Lost key (deleted or not persisted):**
Recent ActionTraces signed with the lost key cannot be re-verified. Traces signed before the loss are unaffected if the key was previously recorded with `matrixai keys rotate`.

**Compromised key (leaked):**
An attacker with the key could forge ActionTrace signatures. Rotate immediately.

Check the key history:

```bash
matrixai keys list --registry-path registry/
# fingerprint  purpose   added_at                     rotated_at
# sha256:ab12  action    2026-05-01T10:00:00+00:00    —         (active)
# sha256:cd34  action    2026-04-01T09:00:00+00:00    2026-05-01T10:00:00+00:00
```

### Action

**Step 1 — Retire the old/compromised key** (`keys rotate` records the old key as retired):

```bash
# Pass the CURRENT (old) key value — this retires it and adds it to history
matrixai keys rotate --purpose action \
  --key <current-key-hex> \
  --registry-path registry/
# Key retired and recorded in registry/.matrixai_key_history.json
#   Purpose    : action
#   Fingerprint: sha256:ab12...

# If the key is truly lost (value unknown), skip this step.
# The old key can't be added to history, but the new key will work for new traces.
```

**Step 2 — Generate and set the new key** (do NOT call `keys rotate` on the new key):

```bash
NEW_KEY=$(openssl rand -hex 32)
echo "MATRIXAI_ACTION_SIGNING_KEY=$NEW_KEY" >> .env
# The new key will be recorded in history automatically when it is rotated in the future.
```

**Step 3 — Restart the server:**

```bash
docker compose stop
docker compose up -d
```

**Step 4 — Assess impact:**

```bash
# Identify traces that may be unverifiable
# (those signed with the lost key, if key was never recorded)
matrixai audit-action trace_2026-05-27.json \
  --signing-key $NEW_KEY
# If signature_valid: false, this trace cannot be re-verified with the new key.
# It must be re-audited against the old key if the old key is recovered.
```

### Verification

```bash
# Confirm new key is active
matrixai keys list --registry-path registry/
# sha256:<new_fp>  action  2026-05-28T...  — (active)

# Confirm new traces are signed correctly
matrixai audit-action new_trace.json
# signature_valid: true
```

**What is preserved after key loss:**
- All traces signed before the loss remain on disk — their content is intact.
- Signature verification of those traces requires the old key. If the old key was recorded in key history before the loss, `matrixai audit-action` will find it automatically.
- The audit chain (what was decided, when, and what the model state was) is preserved regardless.

---

## Scenario 6 — Tamper detection in production

### Symptom

One of:
- `matrixai registry verify` returns `FAIL: entry_hash mismatch` or `FAIL: signature invalid`
- `matrixai audit-action` returns `signature_valid: false` for a trace you did not modify
- The server logs an unexpected signing or verification error
- A parameter set was loaded but its `model_hash` doesn't match the current model

### Diagnosis

MatrixAI's tamper detection works at three levels:

| Level | What it protects | How to check |
|---|---|---|
| Registry entries | `params.json`, `model.mxai`, `manifest.json` | `matrixai registry verify` |
| ActionTraces | Trace content (prediction, action, timestamp) | `matrixai audit-action <trace>` |
| ParameterSet | Model identity via `model_hash` | `matrixai validate-parameters` |

```bash
# Check registry integrity
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# FAIL: entry_hash mismatch ← content was changed after signing

# Check parameter set integrity
matrixai validate-parameters model.mxai --params params.json
# model_hash mismatch: expected mxai_20d8ce..., got mxai_61d33f...
```

### Action

**Do not serve a tampered model.** Stop the server first.

```bash
docker compose stop
```

**Identify the scope:**

```bash
# Verify all registry entries
for entry in $(matrixai registry list --registry-path registry/ --json \
  | python3 -c "import sys,json; [print(e['name']+'@'+e['version']) for e in json.load(sys.stdin)]")
do
  matrixai registry verify --registry-path registry/ "$entry"
done

# Check ActionTrace chain for recent traces
matrixai audit-action trace_latest.json
```

**Restore from backup:**

```bash
cp -r /backup/registry-YYYYMMDD/ registry/
cp /backup/params-YYYYMMDD.json params.json
```

**If no backup is available:**
Re-train the model from the original dataset and re-register. The tampered artifacts must be discarded — do not attempt to patch individual fields.

### Verification

```bash
# All entries must pass
matrixai registry verify --registry-path registry/ alert-monitor@v1.0
# OK: alert-monitor@v1.0 integrity verified

matrixai validate-parameters model.mxai --params params.json
# ParameterSet validation: ok

# Restart the server
docker compose up -d
curl -s http://localhost:8000/health | python3 -m json.tool
# "status": "ok"
```

**Record the incident.** Document what was tampered, when it was detected, and what was restored. If this is a production system, treat it as a security incident and notify your incident response process.

---

## Quick reference

| Symptom | Scenario | First command |
|---|---|---|
| `drift_degradation_detected 1` | 1 — Drift | `matrixai continual status policy.mxcontinual` |
| Need to confirm rollback completed | 2 — Verify rollback | `matrixai continual status policy.mxcontinual --json` |
| `rollback_failed` in logs | 3 — Rollback failed | `matrixai registry list --registry-path registry/` |
| `registry verify` FAIL | 4 — Registry corrupted | Full verify loop + restore from backup |
| Traces not signing / `signature_valid: false` | 5 — Key lost | `matrixai keys list --registry-path registry/` |
| `entry_hash mismatch` / unexpected verify failures | 6 — Tamper | Stop server + `matrixai registry verify` all entries |

---

## Related guides

| Guide | Contents |
|---|---|
| [Deployment](DEPLOYMENT.md) | Pack and deploy with Docker |
| [Observability](OBSERVABILITY.md) | Prometheus `/metrics`, drift gauges |
| [Key Rotation](KEY_ROTATION.md) | Rotate signing keys without invalidating history |
