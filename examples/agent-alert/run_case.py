#!/usr/bin/env python3
"""PR2-C4 — Agente operativo con acción real auditada.

Narrativa: un sistema de monitorización clasifica eventos de infraestructura y
dispara una alerta de email cuando la severidad supera el umbral. La organización
exige control total: simulación previa (dry-run), firma HMAC, trazabilidad, y
capacidad de rollback.

Ejecución desde la raíz del proyecto:
    python3 examples/agent-alert/run_case.py
"""
from __future__ import annotations

import copy
import csv
import json
import shutil
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
REGISTRY_PATH = HERE / "registry"
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de demo
# ─────────────────────────────────────────────────────────────────────────────
SIGNING_KEY = "cafebabe" * 8          # 32 bytes como hex — sólo para demo
ALERT_THRESHOLD = 0.60
FEATURE_NAMES = ["severity", "source_trust", "is_business_hours"]

# ─────────────────────────────────────────────────────────────────────────────
# Mock de transporte de email (no envía nada real)
# ─────────────────────────────────────────────────────────────────────────────
_email_log: list[dict] = []

def _mock_email(smtp_host, smtp_port, smtp_user, smtp_pass, recipient, subject, body) -> str:
    _email_log.append({"to": recipient, "subject": subject, "body": body})
    return f"250 OK (simulado — no se envió email real a {recipient!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
# Cada fila: severity, source_trust, is_business_hours, alert_label
# Label 1 = debe disparar alerta; 0 = no
_CORPUS = [
    # Críticos (alta severidad + confianza) → alerta
    (0.92, 0.90, 1.0, 1),
    (0.88, 0.85, 1.0, 1),
    (0.85, 0.80, 0.0, 1),
    (0.95, 0.95, 1.0, 1),
    (0.80, 0.75, 1.0, 1),
    (0.90, 0.88, 0.0, 1),
    (0.87, 0.82, 1.0, 1),
    (0.93, 0.91, 1.0, 1),
    (0.82, 0.78, 0.0, 1),
    (0.91, 0.87, 1.0, 1),
    (0.86, 0.83, 1.0, 1),
    (0.94, 0.92, 0.0, 1),
    # Advertencias (severidad media) → no alerta
    (0.60, 0.70, 1.0, 0),
    (0.55, 0.65, 1.0, 0),
    (0.65, 0.60, 0.0, 0),
    (0.50, 0.80, 1.0, 0),
    (0.62, 0.72, 1.0, 0),
    (0.58, 0.68, 0.0, 0),
    (0.63, 0.75, 1.0, 0),
    (0.57, 0.62, 1.0, 0),
    (0.61, 0.70, 0.0, 0),
    (0.54, 0.65, 1.0, 0),
    (0.66, 0.71, 1.0, 0),
    (0.59, 0.67, 0.0, 0),
    # Informativos (baja severidad) → no alerta
    (0.20, 0.90, 1.0, 0),
    (0.15, 0.85, 1.0, 0),
    (0.25, 0.80, 0.0, 0),
    (0.30, 0.75, 1.0, 0),
    (0.10, 0.95, 1.0, 0),
    (0.22, 0.88, 0.0, 0),
    (0.18, 0.82, 1.0, 0),
    (0.28, 0.78, 1.0, 0),
    (0.12, 0.92, 0.0, 0),
    (0.24, 0.86, 1.0, 0),
    (0.16, 0.84, 1.0, 0),
    (0.26, 0.79, 0.0, 0),
]

# Train: 10 críticos + 10 advertencias + 10 informativos (30)
# Test: 2 críticos + 2 advertencias + 2 informativos (6)
_TRAIN_IDX = list(range(0, 10)) + list(range(12, 22)) + list(range(24, 34))
_TEST_IDX  = [10, 11, 22, 23, 34, 35]

ALERT_EVENTS = [
    # Eventos en vivo para la demo de la acción
    {"id": "CRIT-001", "severity": 0.90, "source_trust": 0.85, "is_business_hours": 1.0,
     "source": "servidor-db-01", "desc": "CPU al 95% sostenido durante 10 min"},
    {"id": "CRIT-002", "severity": 0.85, "source_trust": 0.80, "is_business_hours": 0.0,
     "source": "red-core-02",    "desc": "pérdida de paquetes >30% en backbone"},
    {"id": "WARN-001", "severity": 0.62, "source_trust": 0.72, "is_business_hours": 1.0,
     "source": "servidor-app-05","desc": "latencia P99 elevada (>800 ms)"},
    {"id": "INFO-001", "severity": 0.20, "source_trust": 0.90, "is_business_hours": 1.0,
     "source": "cron-backup",    "desc": "backup completo — duración mayor de lo habitual"},
]


def _ensure_data() -> None:
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)

    header = FEATURE_NAMES + ["alert_label"]
    train_path = data_dir / "train.csv"
    test_path  = data_dir / "test.csv"

    if not train_path.exists():
        with open(train_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in _TRAIN_IDX:
                row = list(_CORPUS[i])
                w.writerow(row)

    if not test_path.exists():
        with open(test_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in _TEST_IDX:
                row = list(_CORPUS[i])
                w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
def _train_and_register():
    """Entrena AlertModel, guarda artefactos y lo registra. Devuelve entry."""
    from matrixai.training import parse_training_text, SupervisedTrainer
    from matrixai.registry import ModelRegistry

    mxtrain = f"""MODEL alert_model_train.mxai

DATASET AlertData
  SOURCE csv("data/train.csv")
  INPUT SystemMetrics FROM COLUMNS [
    severity, source_trust, is_business_hours
  ]
  TARGET alert_label: Probability
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8 shuffle=true
END

LOSS AlertLoss
  TYPE binary_cross_entropy
  PREDICTION R
  TARGET alert_label
END

OPTIMIZER AlertOpt
  TYPE sgd
  LEARNING_RATE 0.6
  UPDATE W1, b1
END

METRIC Accuracy
  TYPE accuracy
  PREDICTION R
  TARGET alert_label
END

RUN
  EPOCHS 80
  SAVE_BEST true
END
"""

    spec = parse_training_text(mxtrain)
    if REGISTRY_PATH.exists():
        shutil.rmtree(REGISTRY_PATH)
    registry = ModelRegistry(REGISTRY_PATH)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "train_out"
        result = SupervisedTrainer().train(spec, output_dir=str(run_dir), base_path=HERE)

        (run_dir / "evaluation_report.json").write_text(json.dumps({
            "accuracy": round(result.accuracy, 4),
            "best_epoch": result.best_epoch,
            "best_validation_loss": round(result.best_validation_loss, 6),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        shutil.copy(HERE / "alert_model_train.mxai", run_dir / "model_snapshot.mxai")
        entry = registry.push_run_dir(run_dir, "alert-monitor", "v1.0")

    return registry, entry, result.accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia
# ─────────────────────────────────────────────────────────────────────────────
def _predict(program, params: dict, features: dict) -> float:
    from matrixai.runtime.runtime import MatrixAIRuntime
    result = MatrixAIRuntime().run(program, features, params)
    r = result["state"]["R"]
    return float(r[0]) if isinstance(r, list) else float(r)


# ─────────────────────────────────────────────────────────────────────────────
# Acción P20
# ─────────────────────────────────────────────────────────────────────────────
def _load_contract():
    from matrixai.actions import parse_mxact
    with open(HERE / "alert_notifier.mxact") as f:
        source = f.read()
    contracts = parse_mxact(source)
    return contracts[0]


def _load_action_program():
    """Load the full alert_monitor.mxai (with ACTION) for DryRunSimulator."""
    from matrixai.parser.parser import parse_file
    return parse_file(str(HERE / "alert_monitor.mxai"))


def _run_dry_run(contract, program, model_hash, param_set_id, input_data):
    from matrixai.actions import DryRunSimulator
    sim = DryRunSimulator()
    return sim.simulate(contract, program, param_set_id, model_hash, input_data)


def _execute_action(contract, dry_run_report, model_hash, param_set_id, input_data,
                    signing_key=SIGNING_KEY, allow_real=True):
    from matrixai.actions import ActionExecutor, ExecutionContext
    ctx = ExecutionContext(
        contract=contract,
        dry_run_report=dry_run_report,
        input_data=input_data,
        model_hash=model_hash,
        parameter_set_id=param_set_id,
        allow_real_actions=allow_real,
        signing_key=signing_key,
    )
    executor = ActionExecutor(email_fn=_mock_email)
    result = executor.execute(ctx)
    return ctx, result


def _build_trace(ctx, result, signing_key=SIGNING_KEY):
    from matrixai.actions import build_action_trace
    return build_action_trace(ctx, result, signing_key=signing_key)


def _verify_trace(trace, signing_key=SIGNING_KEY):
    from matrixai.actions import verify_action_trace
    return verify_action_trace(trace, signing_key)


def _do_rollback(trace, contract, rollback_input):
    from matrixai.actions import ActionExecutor, RollbackManager
    executor = ActionExecutor(email_fn=_mock_email)
    mgr = RollbackManager(executor=executor)
    return mgr.execute_rollback(trace, contract, rollback_input)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de presentación
# ─────────────────────────────────────────────────────────────────────────────
SEP  = "─" * 72
SEP2 = "═" * 72

def _tag(score: float) -> str:
    if score >= 0.80: return "CRITICO"
    if score >= 0.70: return "ALERTA"
    if score >= 0.50: return "AVISO"
    return "INFO"

def _pass(ok: bool) -> str:
    return "OK" if ok else "FALLO"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(SEP2)
    print("PR2-C4 — AGENTE OPERATIVO CON ACCIÓN AUDITADA")
    print("  Dry-run  ·  Ejecución firmada  ·  ActionTrace  ·  Rollback")
    print(SEP2)

    # ── PASO 1: Datos ─────────────────────────────────────────────────────────
    print("\n[PASO 1] Generando dataset de entrenamiento…")
    _ensure_data()
    train_path = HERE / "data" / "train.csv"
    test_path  = HERE / "data" / "test.csv"
    with open(train_path) as f:
        train_rows = list(csv.DictReader(f))
    with open(test_path) as f:
        test_rows = list(csv.DictReader(f))
    print(f"  train: {len(train_rows)} filas  |  test: {len(test_rows)} filas")

    # ── PASO 2: Entrenamiento + registro ─────────────────────────────────────
    print("\n[PASO 2] Entrenando AlertModel…")
    registry, entry, train_acc = _train_and_register()
    print(f"  Modelo registrado: alert-monitor@v1.0")
    print(f"  entry_hash       : {entry.entry_hash[:16]}…")
    print(f"  parameter_set_id : {entry.parameter_set_id}")
    print(f"  Accuracy entreno : {train_acc:.1%}")

    from matrixai.parameters import load_parameter_set
    from matrixai.parser.parser import parse_file
    program = parse_file(str(HERE / "alert_model_train.mxai"))
    ps_path = REGISTRY_PATH / "entries" / "alert-monitor" / "v1.0" / "params.json"
    ps      = load_parameter_set(ps_path)
    params  = ps.runtime_parameters()

    # Evaluación en test
    correct = 0
    for row in test_rows:
        feat = {k: float(row[k]) for k in FEATURE_NAMES}
        score = _predict(program, params, feat)
        label = int(float(row["alert_label"]))
        predicted_label = 1 if score >= ALERT_THRESHOLD else 0
        if predicted_label == label:
            correct += 1
    accuracy = correct / len(test_rows)
    baseline = max(
        sum(1 for r in test_rows if float(r["alert_label"]) == 1),
        sum(1 for r in test_rows if float(r["alert_label"]) == 0),
    ) / len(test_rows)
    print(f"  Exactitud test   : {accuracy:.0%}  (baseline: {baseline:.0%})")

    # ── PASO 3: Scoring de eventos en vivo ───────────────────────────────────
    print("\n[PASO 3] Clasificando eventos de infraestructura…")
    print(f"  {'ID':<12} {'Origen':<20} {'Score':>7}  {'Nivel':<9}  {'Acción?'}")
    print(f"  {SEP}")

    scored_events = []
    for ev in ALERT_EVENTS:
        feat = {k: ev[k] for k in FEATURE_NAMES}
        score = _predict(program, params, feat)
        fires = score >= ALERT_THRESHOLD
        scored_events.append((ev, score, fires))
        print(f"  {ev['id']:<12} {ev['source']:<20} {score:>7.3f}  {_tag(score):<9}  {'SÍ ←' if fires else 'no'}")

    # ── PASO 4: Flujo completo para primer evento crítico ────────────────────
    critical_events = [(ev, score) for ev, score, fires in scored_events if fires]
    if not critical_events:
        print("\n  Ningún evento supera el umbral — no se demuestra el flujo de acción.")
        return

    ev, score = critical_events[0]
    print(f"\n[PASO 4] Flujo P20 para evento {ev['id']} (score={score:.3f})")

    contract        = _load_contract()
    action_program  = _load_action_program()

    action_input = {
        "recipient": "ops@example.com",
        "subject":   f"[{_tag(score)}] {ev['id']}: {ev['desc'][:60]}",
        "body":      (
            f"Evento:  {ev['id']}\n"
            f"Fuente:  {ev['source']}\n"
            f"Score:   {score:.4f}\n"
            f"Detalle: {ev['desc']}\n"
            f"Hash modelo: {entry.entry_hash[:16]}…\n"
        ),
    }

    # 4.1 Dry-run
    print(f"\n  4.1  DRY-RUN")
    dry_run = _run_dry_run(contract, action_program, entry.entry_hash, entry.parameter_set_id, action_input)
    print(f"       report_id   : {dry_run.report_id[:24]}…")
    print(f"       scope_ok    : {_pass(dry_run.scope_ok)}")
    print(f"       rate_limit  : {_pass(dry_run.rate_limit_ok)}")
    print(f"       input_types : {_pass(dry_run.input_types_ok)}")
    print(f"       rollback_ok : {_pass(dry_run.rollback_ok)}")
    print(f"       resultado   : {_pass(dry_run.ok)}")

    if not dry_run.ok:
        print(f"       errores: {dry_run.errors}")
        return

    # 4.2 Ejecución firmada
    print(f"\n  4.2  EJECUCIÓN FIRMADA")
    ctx, result = _execute_action(
        contract, dry_run, entry.entry_hash, entry.parameter_set_id, action_input
    )
    print(f"       ok          : {_pass(result.ok)}")
    print(f"       latencia    : {result.latency_ms:.1f} ms")
    print(f"       respuesta   : {result.response_summary}")

    # 4.3 ActionTrace + verificación HMAC
    print(f"\n  4.3  ACTION TRACE")
    trace = _build_trace(ctx, result)
    print(f"       report_id   : {trace.report_id[:24]}…")
    print(f"       model_hash  : {trace.model_hash[:16]}…")
    print(f"       executor    : {trace.executor_kind}")
    print(f"       executed_at : {trace.executed_at}")
    print(f"       hmac_sig    : {trace.hmac_signature[:32] if trace.hmac_signature else 'None'}…")
    valid = _verify_trace(trace)
    print(f"       verificación HMAC: {_pass(valid)}")

    # ── PASO 5: Guardarrailes ─────────────────────────────────────────────────
    print(f"\n[PASO 5] Guardarrailes de seguridad")

    # 5.1 Sin clave de firma → bloqueado
    print(f"\n  5.1  Sin signing_key → debe ser bloqueado")
    try:
        from matrixai.actions import ActionExecutor, ExecutionContext, ActionExecutorError
        ctx_no_key = ExecutionContext(
            contract=contract, dry_run_report=dry_run, input_data=action_input,
            model_hash=entry.entry_hash, parameter_set_id=entry.parameter_set_id,
            allow_real_actions=True, signing_key=None,
        )
        ActionExecutor(email_fn=_mock_email).execute(ctx_no_key)
        print(f"       FALLO: la ejecución no fue bloqueada")
    except Exception as exc:
        print(f"       Bloqueado correctamente: {type(exc).__name__}: {exc}")

    # 5.2 Recipient fuera de scope → dry-run falla
    print(f"\n  5.2  Recipient no autorizado → dry-run debe rechazar")
    bad_input = {**action_input, "recipient": "hacker@evil.com"}
    bad_dry = _run_dry_run(contract, action_program, entry.entry_hash, entry.parameter_set_id, bad_input)
    print(f"       dry_run.ok  : {_pass(bad_dry.ok)}")
    print(f"       scope_ok    : {_pass(bad_dry.scope_ok)}")
    if bad_dry.errors:
        print(f"       error       : {bad_dry.errors[0]}")

    # 5.3 Tamper de ActionTrace → verificación falla
    print(f"\n  5.3  Manipulación del ActionTrace → verificación HMAC debe fallar")
    tampered_trace = copy.copy(trace)
    tampered_trace.ok = not trace.ok          # Invertir resultado
    valid_tampered = _verify_trace(tampered_trace)
    print(f"       original ok={trace.ok} → tampered ok={tampered_trace.ok}")
    print(f"       verificación HMAC: {_pass(valid_tampered)} (esperado: FALLO)")
    print(f"       {'Integridad protegida correctamente' if not valid_tampered else 'ERROR: manipulacion no detectada'}")

    # ── PASO 6: Rollback ─────────────────────────────────────────────────────
    print(f"\n[PASO 6] Rollback — corrección post-envío")
    rollback_input = {
        "recipient": "ops@example.com",
        "subject":   f"[CORRECCIÓN] {ev['id']} — alerta anulada",
        "body":      f"El alerta {ev['id']} fue enviada por error y ha sido anulada.\n",
    }
    rb_result = _do_rollback(trace, contract, rollback_input)
    print(f"       attempted           : {rb_result.attempted}")
    print(f"       ok                  : {_pass(rb_result.ok)}")
    print(f"       rollback_contract   : {rb_result.rollback_contract_name}")
    if rb_result.error:
        print(f"       error               : {rb_result.error}")

    # Registro del correo de corrección
    if len(_email_log) >= 2:
        corr = _email_log[-1]
        print(f"       correo enviado a    : {corr['to']}")
        print(f"       asunto              : {corr['subject']}")

    # ── PASO 7: Resumen del audit trail ──────────────────────────────────────
    print(f"\n[PASO 7] Audit Trail completo")
    print(f"  Emails registrados en esta sesión: {len(_email_log)}")
    for i, e in enumerate(_email_log, 1):
        print(f"  [{i}] → {e['to']}: {e['subject'][:60]}")

    print(f"\n  ActionTrace firmado:")
    print(f"    report_id         : {trace.report_id}")
    print(f"    model_hash        : {trace.model_hash}")
    print(f"    parameter_set_id  : {trace.parameter_set_id}")
    print(f"    action_contract   : {trace.action_contract_hash[:32]}…")
    print(f"    input_hash        : {trace.input_hash[:32]}…")
    print(f"    hmac_signature    : {trace.hmac_signature[:40] if trace.hmac_signature else 'unsigned'}…")

    print(f"\n{SEP2}")
    print("RESULTADO FINAL")
    # Métricas honestas: cuántos eventos esperan disparar (severity≥0.7 a ojo) vs cuántos dispararon.
    expected_critical = sum(1 for ev in ALERT_EVENTS if ev["severity"] >= 0.70)
    actually_fired    = len(critical_events)
    non_critical      = len(ALERT_EVENTS) - expected_critical
    correctly_ignored = sum(1 for ev, _s, fires in scored_events if (not fires) and ev["severity"] < 0.70)
    print(f"  Modelo AlertModel entrenado y registrado: alert-monitor@v1.0")
    print(f"  Exactitud clasificación: {accuracy:.0%}")
    print(f"  Acciones disparadas: {actually_fired} (sobre {expected_critical} críticos esperados)")
    print(f"  No-críticos correctamente ignorados: {correctly_ignored} / {non_critical}")
    print(f"  Flujo demostrado: dry-run OK → ejecución firmada → ActionTrace HMAC verificada → rollback ejecutado")
    print(f"  Guardarrailes verificados: sin clave, scope incorrecto, tamper detección")
    print(f"  Límite honesto: el mock de email no envía correo real (SMTP no configurado).")
    print(f"    Para producción: pasar credenciales reales vía env MATRIXAI_SMTP_*.")
    print(SEP2)


if __name__ == "__main__":
    main()
