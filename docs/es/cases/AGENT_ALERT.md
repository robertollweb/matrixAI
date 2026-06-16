# Caso 4 — Agente Automatizado Auditable: Acciones Reales con Control

> **English:** [docs/en/cases/AGENT_ALERT.md](../../en/cases/AGENT_ALERT.md)

**Industria:** Operaciones de TI / Monitorización de Infraestructura  
**Dificultad:** Avanzado  
**Tiempo de ejecución:** ~15 segundos

---

## El problema

Un sistema de monitorización de infraestructura clasifica eventos por severidad y debería disparar automáticamente una alerta de email cuando la puntuación supera el umbral. Hoy esto se hace manualmente, introduciendo latencia en los incidentes críticos.

La automatización está bloqueada por una cuestión de cumplimiento: ¿cómo se demuestra, seis meses después, exactamente qué hizo el sistema, con qué modelo, sobre qué input, en qué momento — y que nadie alteró el registro?

Tres problemas sin resolver:
- ¿Cómo se evita que un modelo actúe sin validación previa?
- ¿Cómo se firma criptográficamente cada acción ejecutada para que cualquier manipulación sea detectable?
- ¿Cómo se revierte una acción errónea cuando la automatización se equivoca?

Sin respuesta a los tres, ninguna organización regulada confiará en una acción automatizada con consecuencias reales.

---

## La solución

MatrixAI combina un AlertModel entrenado con el framework de acción real P20:

- **Dry-run**: antes de cualquier acción, `DryRunSimulator` valida scope, límites de tasa, tipos de input y disponibilidad de rollback. Si alguna verificación falla, la acción queda bloqueada.
- **Ejecución firmada**: `ActionExecutor` ejecuta la acción con una clave de firma; cada ejecución produce un `ActionTrace` firmado con HMAC-SHA256.
- **Detección de manipulación**: verificar la traza contra la clave de firma original detecta cualquier modificación del registro.
- **Rollback**: `RollbackManager` ejecuta el contrato `send_correction` declarado — la reversión es en sí misma una acción trazada y firmada.

Toda acción va precedida de una simulación. Toda ejecución está firmada. Toda traza es verificable.

---

## Ejecútalo tú mismo

Desde el directorio raíz de `matrixAI`:

```bash
python3 examples/agent-alert/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/agent-alert/run_case.py
```

Sin dependencias externas. Sin API keys. El email es simulado (no se envía correo real).

### Salida esperada

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

## El resultado

### Métrica de modelo

| Componente | Rol | Accuracy |
|---|---|---|
| AlertModel | severidad + confianza_fuente + horario_laboral → probabilidad de alerta | 83% en 6 eventos de test |
| Baseline (clase mayoritaria) | — | 67 % |

### Métrica de valor

**Cada acción automatizada va precedida de una simulación, está firmada criptográficamente, es verificable y reversible.**

Para cualquier acción histórica:
- El `ActionTrace` registra `model_hash`, `parameter_set_id`, `action_contract_hash`, `input_hash` y `executed_at`.
- La firma HMAC-SHA256 detecta cualquier alteración de la traza en milisegundos.
- El contrato de rollback (`send_correction`) puede revertir una acción errónea, y la reversión es en sí misma una traza firmada.
- Tres guardarrailes son obligatorios: sin clave de firma → bloqueado; destinatario no autorizado → el dry-run rechaza; traza manipulada → falla la verificación HMAC.

Esto resuelve directamente el problema de cumplimiento de la automatización: el sistema puede actuar automáticamente sin perder la capacidad de demostrar *exactamente qué hizo* y *por qué*.

---

## Arquitectura

```
Evento de infraestructura
  severity, source_trust, is_business_hours
          │
          ▼
┌─────────────────────────────┐
│  AlertModel                 │
│  (registrado, entry_hash)   │
│  sigmoid → alert_score      │
└─────────────────────────────┘
          │
          ▼  score ≥ 0.60 umbral
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  Framework de Acción P20                            │
│                                                     │
│  DryRunSimulator                                    │
│    scope_ok: ¿destinatario en allowed_recipients?   │
│    rate_limit_ok: ¿dentro de los límites?           │
│    input_types_ok: ¿tipos coinciden con el schema?  │
│    rollback_ok: ¿contrato de rollback declarado?    │
│                                                     │
│  ActionExecutor (signing_key requerido)             │
│    → email_fn(smtp_host, port, user, pass,          │
│               destinatario, asunto, cuerpo)         │
│    → ActionTrace (firmado con HMAC-SHA256)          │
│                                                     │
│  RollbackManager                                    │
│    → ejecuta contrato send_correction               │
│    → produce segundo ActionTrace firmado            │
└─────────────────────────────────────────────────────┘
```

El contrato de acción (`alert_notifier.mxact`) declara:
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

## Límites

- El mock de email no envía correo real. Para producción: pasar credenciales SMTP reales vía variables de entorno `MATRIXAI_SMTP_*`.
- El umbral de alerta (0.60) es ilustrativo. En producción debe calibrarse contra el coste real de falsos positivos en el entorno concreto.
- La clave de firma del demo (`cafebabe` × 8) es solo para la demostración. En producción, usar un secreto gestionado por un sistema de gestión de claves.
- El rollback se ejecuta de principio a fin: `RollbackManager` reutiliza el dry-run internamente dentro de su ventana de validez por defecto de 5 minutos. En producción, puede exigirse un dry-run explícito controlado por el operador antes del rollback.
- El registry es local-first. Para producción se operaría sobre infraestructura gestionada (nivel de pago).
- Este caso demuestra el framework de control. La responsabilidad de configurar los guardarrailes adecuadamente para el entorno operativo concreto recae en el operador.

---

## Qué es gratis y qué se paga

| Capa | Estado |
|---|---|
| Entrenamiento y registro del AlertModel | **Core — gratuito** |
| Simulación dry-run antes de cada acción | **Core — gratuito** |
| ActionTrace firmado con HMAC por cada ejecución | **Core — gratuito** |
| Ejecución del contrato de rollback | **Core — gratuito** |
| Detección de manipulación en modelos registrados | **Core — gratuito** |
| Registry gestionado con retención y control de acceso | Nivel de pago |
| Flujos de aprobación human-in-the-loop | Nivel de pago |
| Configuración y soporte de guardarrailes en producción | Nivel de pago |
| Generación de informes de auditoría para revisión de cumplimiento | Nivel de pago |

---

## Archivos

```
examples/agent-alert/
  alert_model_train.mxai   — AlertModel: SystemMetrics[3] → sigmoid → probabilidad de alerta
  alert_monitor.mxai       — Proyecto completo con ACTION TriggerAlert y GRAPH
  alert_notifier.mxact     — ACTION_CONTRACT: email_send, DRY_RUN requerido, ROLLBACK, SIGNATURE
  data/
    train.csv              — 30 eventos (10 críticos + 10 advertencias + 10 informativos), autogenerado
    test.csv               — 6 eventos de test (2 por clase)
  run_case.py              — Demo extremo a extremo de 7 pasos
```
