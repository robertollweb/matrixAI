# MatrixAI — Casos de Uso: Guía para Evaluadores

> **English:** [docs/en/USE_CASES.md](../en/USE_CASES.md)

Esta guía es para alguien que evalúa si MatrixAI resuelve un problema real.

Cada caso a continuación es un problema de industria que MatrixAI resuelve de principio a fin. Puedes ejecutar cada caso en menos de 30 segundos con un solo comando. Sin API keys. Sin configuración. Datos incluidos.

---

## Antes de empezar

Desde la raíz del proyecto, comprueba que MatrixAI es importable:

```bash
python3 -c "import matrixai; print('OK')"
```

Si falla, añade el path:

```bash
export PYTHONPATH=$(pwd)
```

---

## Los cuatro casos

| # | Industria | Problema | Métrica clave | Tiempo |
|---|---|---|---|---|
| 1 | [Financiero](#caso-1--financiero-scoring-de-crédito) | Demostrar qué modelo exacto aprobó un préstamo | 93,3% vs 73,3% baseline | ~10 s |
| 2 | [SaaS / Soporte](#caso-2--saas-routing-de-tickets) | Enrutar tickets automáticamente sin perder trazabilidad | 100% vs 33,3% baseline | ~20 s |
| 3 | [Salud](#caso-3--salud-riesgo-clínico) | Decisiones de riesgo con explicación defendible | 100% vs 58,3% baseline | ~10 s |
| 4 | [Operaciones TI](#caso-4--operaciones-ti-agente-automatizado) | Acciones automáticas con audit trail, firma y rollback | 83% vs 67% baseline | ~15 s |

---

## Caso 1 — Financiero: Scoring de Crédito

**El problema.** Una fintech concede microcréditos y debe demostrar ante un regulador, para cualquier decisión histórica, qué versión exacta del modelo aprobó o rechazó la solicitud — y que el registro no fue alterado retroactivamente.

**Qué hace MatrixAI.** Entrena un modelo de scoring de crédito, lo registra con un `entry_hash` firmado, y vincula cada decisión de scoring a esa entrada exacta. Simula una actualización de modelo v1.0→v1.1 y demuestra que las decisiones antiguas siguen trazando a v1.0. Demuestra la detección de manipulación.

```bash
python3 examples/credit-scoring/run_case.py
```

**Qué mirar:**
- El `entry_hash` que se imprime tras el registro: es la huella criptográfica del modelo.
- El paso de audit trail: dado cualquier decisión, el sistema recupera la versión exacta del modelo que la produjo.
- El paso de tamper al final: modificar el `params.json` registrado lanza `VerificationError` en milisegundos.

**Runbook completo:** [docs/es/cases/CREDIT_SCORING.md](cases/CREDIT_SCORING.md)

---

## Caso 2 — SaaS: Routing de Tickets

**El problema.** Un equipo de soporte enruta tickets manualmente a colas de facturación, técnico y ventas. A 1000 tickets/día, el routing manual no escala. Cuando un ticket se enruta mal seis meses después, no hay forma de saber qué versión del modelo lo causó.

**Qué hace MatrixAI.** Entrena un pipeline compuesto de dos etapas: un TextEmbedder (BoW → Dense(8) → señal de routing) y un RouteClassifier (señal → 3 categorías). Ambos se registran de forma independiente. Un único `composite_model_hash` vincula cada decisión de routing a la versión exacta de *ambos* componentes simultáneamente.

```bash
python3 examples/text-routing/run_case.py
```

**Qué mirar:**
- Paso 3: 9 tickets de test enrutados con 100% de accuracy vs 33,3% baseline.
- Paso 4: el audit trail de un ticket — el `composite_model_hash` se descompone mostrando tanto `feature_extractor@v1` como `route_classifier@v1`.
- Paso 5: manipular el TextEmbedder lanza `VerificationError` aunque el RouteClassifier no haya sido tocado.

**Runbook completo:** [docs/es/cases/TEXT_ROUTING.md](cases/TEXT_ROUTING.md)

---

## Caso 3 — Salud: Riesgo Clínico

**El problema.** Un sistema de apoyo a la decisión hospitalaria estima el riesgo de caída. Un modelo que solo dice "riesgo 0.87" no es aceptable en la práctica clínica: el personal necesita saber *por qué*, y esa explicación debe conservarse para una posible revisión.

**Qué hace MatrixAI.** Entrena un clasificador de riesgo de caída sobre 5 features clínicas (edad, movilidad, carga de medicación, caídas previas, estado cognitivo). Para cada evaluación de paciente, calcula la contribución lineal exacta de cada feature (`W1[i] × x_i`) — atribución matemáticamente precisa para esta clase de modelo. Cada decisión registra el `entry_hash` de la versión exacta del modelo que la produjo.

```bash
python3 examples/clinical-risk/run_case.py
```

**Qué mirar:**
- Paso 1: pesos aprendidos — `caídas_previas W=+2.12` es el feature de mayor impacto.
- Paso 2: nivel de riesgo de cada paciente (BAJO/MEDIO/ALTO/ALERTA) con el factor principal nombrado.
- Paso 3: desglose completo de contribuciones para 3 pacientes representativos, con la suma `Σ contribuciones + bias`.
- Paso 5: tamper detection — modificar el peso de edad (`W1[0]`) lanza `VerificationError`.

> **Importante:** este caso ilustra capacidad técnica. No constituye validación clínica. Un sistema clínico real requiere validación médica y aprobación regulatoria.

**Runbook completo:** [docs/es/cases/CLINICAL_RISK.md](cases/CLINICAL_RISK.md)

---

## Caso 4 — Operaciones TI: Agente Automatizado

**El problema.** Un sistema de monitorización de infraestructura debería alertar automáticamente ante eventos críticos. Pero la organización exige demostrar exactamente qué se envió, a quién, con qué modelo, en qué momento — y poder revertir una acción errónea.

**Qué hace MatrixAI.** Entrena un AlertModel y ejecuta cada acción disparada a través del framework P20: simulación dry-run obligatoria, ejecución firmada con HMAC, `ActionTrace` verificable y un contrato de rollback declarado. Se demuestran tres guardarrailes: sin clave de firma → bloqueado; destinatario no autorizado → dry-run rechaza; traza manipulada → falla la verificación HMAC.

```bash
python3 examples/agent-alert/run_case.py
```

**Qué mirar:**
- Paso 3: 4 eventos clasificados — 2 disparan la alerta, 2 se ignoran correctamente.
- Paso 4.1: el dry-run verifica `scope_ok`, `rate_limit_ok`, `input_types_ok`, `rollback_ok` — todo pasa.
- Paso 4.3: `ActionTrace` con firma HMAC; `verificación HMAC: OK`.
- Paso 5: los tres guardarrailes bloquean o rechazan según lo esperado.
- Paso 6: el rollback ejecuta un correo de corrección — en sí mismo una acción trazada.

> **Nota:** el transporte de email es simulado. No se envía correo real. Para producción, pasar credenciales SMTP reales vía variables de entorno `MATRIXAI_SMTP_*`.

**Runbook completo:** [docs/es/cases/AGENT_ALERT.md](cases/AGENT_ALERT.md)

---

## Qué es gratis y qué se paga

La misma frontera aplica a los cuatro casos:

| Qué es gratuito | Qué se paga |
|---|---|
| Entrenamiento y registro de cualquier modelo en local | Registry gestionado con retención, backup y control de acceso |
| `entry_hash` firmado y detección de manipulación | API de producción con SLA |
| Pipelines compuestos y `composite_model_hash` | Generación de informes de auditoría para cumplimiento |
| Log de auditoría de decisiones por decisión | Integraciones enterprise (SMTP, HL7/FHIR, Slack, etc.) |
| Dry-run, ejecución firmada HMAC, rollback | Flujos de aprobación human-in-the-loop y soporte |
| Los cuatro casos de uso anteriores | Aprendizaje continuo y monitorización de drift en producción |

El patrón: **el sistema es gratuito en local. El nivel de pago es operarlo a escala de producción, con las garantías de fiabilidad, retención y cumplimiento que las industrias reguladas exigen.**

---

## Siguientes pasos

- Para entender la arquitectura de un caso concreto: leer el runbook completo en `docs/es/cases/`.
- Para ver el formato común de todos los runbooks: [docs/en/RUNBOOK_TEMPLATE.md](../en/RUNBOOK_TEMPLATE.md).
- Para empezar con tu propio modelo: [docs/es/QUICKSTART.md](QUICKSTART.md).
- Para ver el roadmap técnico completo: [documentacion/OBJETIVO_FINAL_Y_ROADMAP.md](../../documentacion/OBJETIVO_FINAL_Y_ROADMAP.md).
