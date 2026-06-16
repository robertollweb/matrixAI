# Caso 2 — Pipeline de routing de texto: trazabilidad de modelos compuestos

> **English:** [docs/en/cases/TEXT_ROUTING.md](../../en/cases/TEXT_ROUTING.md)

**Industria:** SaaS / Soporte al cliente  
**Dificultad:** Intermedio  
**Tiempo de ejecución:** ~20 segundos

---

## El problema

Una empresa SaaS enruta los tickets de soporte entrantes a tres colas —billing, technical y sales— usando un pipeline de ML compuesto por dos modelos entrenados de forma independiente: un TextEmbedder (BoW → embedding denso → señal de routing) y un RouteClassifier.

Cuando un ticket se enruta mal y un cliente escala seis meses después, la empresa necesita saber:
- ¿Qué versión exacta del TextEmbedder estaba activa en ese momento?
- ¿Qué versión exacta del RouteClassifier estaba activa?
- ¿Ha sido modificado alguno desde entonces?

Con dos modelos separados, la traza de auditoría se fragmenta en dos registros, dos almacenes de parámetros y dos logs de despliegue. El riesgo: un cambio de modelo considerado "menor" queda sin documentar y se vuelve inverificable.

---

## La solución

MatrixAI representa el pipeline de dos etapas como un único **modelo compuesto**. Cada componente se registra de forma independiente con su propio `entry_hash`. El modelo compuesto produce un único `composite_model_hash` derivado determinísticamente de los hashes de ambos componentes.

Para cada decisión de routing:
- El `composite_model_hash` vincula la decisión a la versión exacta de **ambos** componentes simultáneamente.
- Actualizar cualquier componente —incluso un ajuste "menor" de parámetros— cambia el hash compuesto automáticamente.
- `registry.verify()` prueba que ningún componente fue alterado tras el registro.

---

## Ejecútalo tú mismo

Desde el directorio raíz de `matrixAI`:

```bash
python3 examples/text-routing/run_case.py
```

**Windows (PowerShell):**
```powershell
python examples/text-routing/run_case.py
```

Sin dependencias externas. Sin API keys. Dataset incluido.

### Salida esperada

```
MatrixAI — PR2-C2: Text Routing Pipeline with Composite Traceability
====================================================================

── Step 1 — Train & register TextEmbedder ──
  Registered: feature_extractor@v1
  entry_hash: sha256:18baae3b26e25519d...
  Architecture: TicketBOW[30] → Dense(8, relu) → Dense(1, sigmoid)

── Step 2 — Train & register RouteClassifier ──
  Registered: route_classifier@v1
  entry_hash: sha256:f628823298aefc2d6...
  RouteClassifier: routing_signal[1] → Dense(3, softmax)

── Step 3 — Route 9 test tickets ──
  Tickets routed: 9
  Pipeline accuracy:  100.0%
  Baseline (clase mayoritaria): 33.3%
  Improvement: +66.7%
  Composite hash: sha256:9a0f529045084d72...

── Step 4 — Audit trail for TKT-0002 ──
  Decision: BILLING (confidence=0.9183)
  Ground truth: BILLING
  Composite hash: sha256:9a0f5290...
  TextEmbedder:    feature_extractor@v1 (sha256:18baae3b...)
  RouteClassifier: route_classifier@v1  (sha256:f6288232...)
  verify('feature_extractor', 'v1') → True  + intact
  verify('route_classifier', 'v1') → True   + intact

── Step 5 — Tamper detection ──
  Tamper detected — VerificationError: params.json content hash mismatch for feature_extractor@v1
  + Cryptographic chain caught modification of TextEmbedder
```

---

## El resultado

### Métrica de modelo

| Componente | Rol | Accuracy |
|---|---|---|
| TextEmbedder | TicketBOW[30] → Dense(8) → señal de routing | N/A (cabeza de regresión) |
| RouteClassifier | signal → billing/technical/sales | 100% sobre señal FE |
| **Pipeline compuesto** | texto crudo → categoría enrutada | **100% en 9 tickets de test** |
| Baseline (clase mayoritaria) | — | 33,3 % |

En texto real de producción con jerga fuera de vocabulario se espera 80–90%; el valor sostenible es la trazabilidad firmada, no la accuracy sobre un corpus controlado.

### Métrica de valor

**Cada decisión de routing es trazable a la versión exacta de ambos componentes del pipeline.**

Para cualquier decisión histórica:
- El `composite_model_hash` identifica qué versión del TextEmbedder Y qué versión del RouteClassifier la produjeron.
- `registry.verify()` prueba que ningún modelo fue modificado tras el registro.
- Actualizar el TextEmbedder a v2 cambia el hash compuesto automáticamente —sin tracking manual.
- Cualquier manipulación retroactiva de cualquier componente se detecta en milisegundos.

Esto elimina directamente la ambigüedad "¿qué modelo causó este enrutamiento incorrecto?" en pipelines multi-modelo —un problema que de otro modo requiere cruzar logs de despliegue, historial de git y backups de parámetros.

---

## Arquitectura

```
Texto crudo del ticket
       │
       ▼  vocabulario fijo de 30 palabras de dominio (account, charge, crash, server, upgrade, pricing…)
       ▼  TicketBOW[30] (presencia binaria por palabra)
┌─────────────────────┐
│  TextEmbedder       │  BoW[30] → Dense(8, relu) → Dense(1, sigmoid)
│  (FROZEN, registry) │  → routing_signal ∈ (0,1)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  RouteClassifier    │  signal → [p_billing, p_technical, p_sales]
│  (FROZEN, registry) │  argmax → categoría
└─────────────────────┘
       │
       ▼
  decisión de routing
```

El hash compuesto es:
```
composite_model_hash = SHA256({
  "own_program_hash": "<hash de text_routing_pipeline.mxai>",
  "imports": [
    {"alias": "TextEmbedder",    "entry_hash": "<fe_entry_hash>"},
    {"alias": "RouteClassifier", "entry_hash": "<rc_entry_hash>"}
  ]
})
```

---

## Límites

- El dataset es sintético con un vocabulario controlado de 30 palabras. La accuracy real sobre texto del cliente depende de cuánto de la jerga del dominio esté cubierto por el vocabulario; para dominios nuevos o idiomas distintos del inglés, el vocabulario debe reentrenarse sobre datos del cliente.
- BoW captura presencia de palabras — no orden, ni negación, ni contexto. Para esos casos, sustituye la Etapa 1 por un encoder más expresivo (transformer, fastText) manteniendo el mismo patrón compuesto.
- Ambos componentes son FROZEN en el compuesto. La API `composite_training_step` permite componentes TRAINABLE para ajustar la segunda etapa sin reentrenar la primera.
- El registry es local. Para producción se operaría sobre infraestructura gestionada (nivel de pago).
- El pipeline no incluye serving HTTP en este corte (HTTP es P6, disponible en el sistema, pero no demostrado aquí).

---

## Qué es gratis y qué se paga

| Capa | Estado |
|---|---|
| Entrenamiento y registro del TextEmbedder | **Core — gratuito** |
| Entrenamiento y registro del RouteClassifier | **Core — gratuito** |
| Pipeline compuesto con `composite_model_hash` | **Core — gratuito** |
| Log de decisiones con hash compuesto por decisión | **Core — gratuito** |
| Detección de manipulación en todos los componentes registrados | **Core — gratuito** |
| Registry gestionado con retención y control de acceso | Nivel de pago |
| API de routing en producción con SLA | Nivel de pago |
| Generación de informes de auditoría para cumplimiento enterprise | Nivel de pago |

---

## Archivos

```
examples/text-routing/
  feature_extractor.mxai       — Etapa 1: TextEmbedder NETWORK sobre TicketBOW[30]
  route_classifier.mxai        — Etapa 2: routing signal → 3 categorías (softmax)
  ticket_router.mxai           — Router standalone de referencia
  text_routing_pipeline.mxai   — Compuesto: TextEmbedder + RouteClassifier (ambos FROZEN)
  ticket_router.mxtrain        — Especificación de entrenamiento para router standalone
  data/
    train.csv                  — 45 tickets sintéticos (15 por categoría), texto crudo + 30 columnas bow_*
    test.csv                   — 9 tickets de test (3 por categoría)
  run_case.py                  — Script de demo extremo a extremo (autogenera los CSV en el primer run)
```
