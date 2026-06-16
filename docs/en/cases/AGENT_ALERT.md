# Case 4 — Auditable Automated Agent: Real Actions with Control

> **Español:** [docs/es/cases/AGENT_ALERT.md](../../es/cases/AGENT_ALERT.md)

**Industry:** IT Operations / Infrastructure Monitoring  
**Difficulty:** Advanced  
**Runtime:** ~15 seconds

---

## The problem

An infrastructure monitoring system classifies events by severity and should automatically trigger an email alert when the score exceeds a threshold. Today this is done manually, introducing latency in critical incidents.

The automation is blocked by a compliance question: how do you prove, six months later, exactly what the system did, with which model, on which input, at what moment — and that no one altered the record?

Three unsolved problems:
- How do you prevent a model from acting without prior validation?
- How do you cryptographically sign each executed action so that any tampering is detectable?
- How do you reverse a wrong action when automation makes a mistake?

Without answers to all three, no regulated organization will trust an automated action with real consequences.

---

## The solution

MatrixAI combines a trained AlertModel with the P20 real action framework:

- **Dry-run**: before any action, `DryRunSimulator` validates scope, rate limits, input types, and rollback availability. If any check fails, the action is blocked.
- **Signed execution**: `ActionExecutor` executes the action with a signing key; every execution produces an `ActionTrace` signed with HMAC-SHA256.
- **Tamper detection**: verifying the trace against the original signing key detects any modification of the record.
- **Rollback**: `RollbackManager` executes the declared `send_correction` contract — the reversal is itself a traced and signed action.

Every action is preceded by a simulation. Every execution is signed. Every trace is verifiable.

---

## Run it yourself

From the `matrixAI` root directory:

```bash
python3 examples/agent-alert/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/agent-alert/run_case.py
```

No external dependencies. No API keys. Email is mocked (no real mail is sent).

### Expected output

```
════════════════════════════════════════════════════════════════════════
PR2-C4 — AGENTE OPERATIVO CON ACCIÓN AUDITADA
  Dry-run  ·  Ejecución firmada  ·  ActionTrace  ·  Rollback
════════════════════════════════════════════════════════════════════════

[PASO 1] Generando dataset de entrenamiento…
  train: 30 filas  |  test: 6 filas

[PASO 2] Entrenando AlertModel…
  Modelo registrado: alert-monitor@v1.0
  entry_hash       : sha256:f22ba8600…
  Accuracy entreno : 100.0%
  Exactitud test   : 83%  (baseline: 67%)

[PASO 3] Clasificando eventos de infraestructura…
  CRIT-001  servidor-db-01   0.616  AVISO   SÍ ←
  CRIT-002  red-core-02      0.629  AVISO   SÍ ←
  WARN-001  servidor-app-05  0.283  INFO    no
  INFO-001  cron-backup      0.037  INFO    no

[PASO 4] Flujo P20 para evento CRIT-001 (score=0.616)
  4.1  DRY-RUN
       scope_ok: OK  |  rate_limit: OK  |  input_types: OK  |  rollback_ok: OK
       resultado: OK

  4.2  EJECUCIÓN FIRMADA
       ok: OK  |  latencia: 0.0 ms
       respuesta: 250 OK (simulado — no se envió email real)

  4.3  ACTION TRACE
       hmac_sig: hmac-sha256:4e0df1f7d9b06a792f43…
       verificación HMAC: OK

[PASO 5] Guardarrailes de seguridad
  5.1  Sin signing_key → Bloqueado correctamente: ActionExecutorError
  5.2  Recipient no autorizado → dry_run.ok: FALLO  scope_ok: FALLO
  5.3  ActionTrace manipulado → verificación HMAC: FALLO (esperado: FALLO)
       Integridad protegida correctamente

[PASO 6] Rollback — corrección post-envío
       attempted: True  |  ok: OK
       rollback_contract: send_correction

[PASO 7] Audit Trail completo
  Emails registrados: 2
  ActionTrace firmado: report_id, model_hash, parameter_set_id, action_contract, hmac_signature

RESULTADO FINAL
  Exactitud clasificación: 83%
  Acciones disparadas: 2 (sobre 2 críticos esperados)
  No-críticos correctamente ignorados: 2 / 2
  Flujo demostrado: dry-run OK → ejecución firmada → ActionTrace HMAC verificada → rollback ejecutado
  Guardarrailes verificados: sin clave, scope incorrecto, tamper detección
```

---

## The result

### Model metric

| Component | Role | Accuracy |
|---|---|---|
| AlertModel | severity + source_trust + is_business_hours → alert probability | 83% on 6 test events |
| Baseline (majority class) | — | 67% |

### Value metric

**Every automated action is preceded by a simulation, cryptographically signed, verifiable, and reversible.**

For any historical action:
- The `ActionTrace` records `model_hash`, `parameter_set_id`, `action_contract_hash`, `input_hash`, and `executed_at`.
- HMAC-SHA256 signature detects any alteration of the trace in milliseconds.
- The `rollback` contract (`send_correction`) can reverse a wrong action, and the reversal is itself a signed trace.
- Three guardrails are enforced: no signing key → blocked; unauthorized recipient → dry-run rejects; tampered trace → HMAC verification fails.

This directly solves the automation compliance problem: the system can act automatically without losing the ability to prove *exactly what it did* and *why*.

---

## Architecture

```
Infrastructure event
  severity, source_trust, is_business_hours
          │
          ▼
┌─────────────────────────────┐
│  AlertModel                 │
│  (registered, entry_hash)   │
│  sigmoid → alert_score      │
└─────────────────────────────┘
          │
          ▼  score ≥ 0.60 threshold
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  P20 Action Framework                               │
│                                                     │
│  DryRunSimulator                                    │
│    scope_ok: recipient in allowed_recipients?       │
│    rate_limit_ok: within configured limits?         │
│    input_types_ok: types match contract schema?     │
│    rollback_ok: rollback contract declared?         │
│                                                     │
│  ActionExecutor (signing_key required)              │
│    → email_fn(smtp_host, port, user, pass,          │
│               recipient, subject, body)             │
│    → ActionTrace (HMAC-SHA256 signed)               │
│                                                     │
│  RollbackManager                                    │
│    → executes send_correction contract              │
│    → produces second signed ActionTrace             │
└─────────────────────────────────────────────────────┘
```

Action contract (`alert_notifier.mxact`) declares:
```
ACTION_CONTRACT TriggerAlert
  ACTION email_send
  SCOPE allowed_recipients ["ops@example.com"]
  DRY_RUN required
  ROLLBACK send_correction
  SIGNATURE_REQUIRED true
END
```

---

## Limits

- The email mock does not send real mail. For production: pass real SMTP credentials via `MATRIXAI_SMTP_*` environment variables.
- The alert threshold (0.60) is illustrative. In production it should be calibrated against the actual false-positive cost of the environment.
- The signing key in the demo (`cafebabe` × 8) is for demonstration only. In production, use a secret managed by a key management system.
- Rollback executes end-to-end: `RollbackManager` reuses the dry-run internally within its default 5-minute validity window. In production, an operator-controlled explicit dry-run can be required before rollback.
- The registry is local-first. Production deployments would use managed registry infrastructure (paid tier).
- This case demonstrates the control framework. Responsibility for configuring guardrails appropriately for the specific operational environment lies with the operator.

---

## What is free and what is paid

| Layer | Status |
|---|---|
| AlertModel training and registration | **Core — free** |
| Dry-run simulation before every action | **Core — free** |
| HMAC-signed ActionTrace for every execution | **Core — free** |
| Rollback contract execution | **Core — free** |
| Tamper detection for all registered models | **Core — free** |
| Managed registry with retention and access control | Paid tier |
| Human-in-the-loop approval workflows | Paid tier |
| Production guardrail configuration and support | Paid tier |
| Audit report generation for compliance review | Paid tier |

---

## Files

```
examples/agent-alert/
  alert_model_train.mxai   — AlertModel: SystemMetrics[3] → sigmoid → alert probability
  alert_monitor.mxai       — Full project with ACTION TriggerAlert and GRAPH
  alert_notifier.mxact     — ACTION_CONTRACT: email_send, DRY_RUN required, ROLLBACK, SIGNATURE
  data/
    train.csv              — 30 events (10 critical + 10 warning + 10 info), auto-generated
    test.csv               — 6 held-out events (2 per class)
  run_case.py              — End-to-end 7-step demo: train, score, dry-run, execute, trace, guardrails, rollback
```
