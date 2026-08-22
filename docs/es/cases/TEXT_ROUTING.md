# Caso 2 — Pipeline de routing de texto: trazabilidad de modelos compuestos

> **English:** [docs/en/cases/TEXT_ROUTING.md](../../en/cases/TEXT_ROUTING.md)

**Industria:** SaaS / Soporte al cliente  
**Dificultad:** Intermedio  
**Tiempo de ejecución:** ~1 segundo (medido: 0,79 s / 1,11 s / 0,79 s en tres pasadas)

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
MatrixAI — PR2-C2: Text Routing with BoW Text Embedding + Composite Traceability
================================================================================

  Vocabulary (30 words): account, amount, api, cancel, charge, connection, contract, crash, demo, discount ...
  Architecture: ticket text → BoW[30] → Dense(8,relu) → Dense(1,sigmoid)
                                                      → Dense(3,softmax) → category

────────────────────────────────────────────────────────────────
  Step 1 — Train TextEmbedder: BoW[30] → Dense(8,relu) → Dense(1,sigmoid)
────────────────────────────────────────────────────────────────
  This is the learned text embedding: maps raw ticket words to a routing signal.
  Registered: feature_extractor@v1
  entry_hash: sha256:57ac18f47f1286f7bd3d7...
  Embedding signal range: billing~0.1  technical~0.5  sales~0.9

────────────────────────────────────────────────────────────────
  Step 2 — Text embedding demo: raw ticket → BoW → routing signal
────────────────────────────────────────────────────────────────

  Text:    "need help with my invoice the charge amount is wrong on my accoun"
  BoW:     [account, amount, charge, help, invoice, need]
  Signal:  0.0643  (expected category: billing)

  Text:    "api is crashing network connection is very slow please fix this i"
  BoW:     [api, connection, fix, issue, network, slow]
  Signal:  0.4597  (expected category: technical)

  Text:    "want to upgrade to enterprise plan please send pricing informatio"
  BoW:     [enterprise, plan, pricing, upgrade]
  Signal:  0.8859  (expected category: sales)

────────────────────────────────────────────────────────────────
  Step 3 — Train RouteClassifier: routing_signal → billing/technical/sales
────────────────────────────────────────────────────────────────
  Registered: route_classifier@v1
  entry_hash: sha256:f8c6c2abe6a9549608644...
  RouteClassifier accuracy on signals: 100.0%

────────────────────────────────────────────────────────────────
  Step 4 — Route 9 test tickets through composite pipeline (text → category)
────────────────────────────────────────────────────────────────
  Tickets routed:            9
  Pipeline accuracy:         100.0%
  Baseline (majority class): 33.3%
  Improvement:               +66.7%
  Composite hash: sha256:1a31cd2c438a01ff38762...
  (covers both fe_entry_hash and rc_entry_hash)

  [+] TKT-0001  BILLING     conf=0.62  [cancel my plan need a refund for the amount char]
  [+] TKT-0002  BILLING     conf=0.92  [my account was charged wrong amount please check]
  [+] TKT-0003  BILLING     conf=0.95  [please help me fix the charge on my account the ]
  [+] TKT-0004  TECHNICAL   conf=0.88  [app is crashing slow server connection and api e]
  [+] TKT-0005  TECHNICAL   conf=0.88  [connection timeout on server when login the api ]
  [+] TKT-0006  TECHNICAL   conf=0.88  [help me fix the server error the network connect]
  [+] TKT-0007  SALES       conf=0.92  [team wants to upgrade to enterprise plan pricing]
  [+] TKT-0008  SALES       conf=0.92  [enterprise demo and pricing for upgrade and cont]
  [+] TKT-0009  SALES       conf=0.93  [want features of enterprise plan for contract up]

────────────────────────────────────────────────────────────────
  Step 5 — Audit trail: exact components for TKT-0002
────────────────────────────────────────────────────────────────
  Ticket:         TKT-0002
  Text:           "my account was charged wrong amount please check my inv..."
  Decision:       BILLING (confidence=0.9183)
  Ground truth:   BILLING

  Composite hash:  sha256:1a31cd2c438a01ff3876204b2...
  TextEmbedder:    feature_extractor@v1  (sha256:57ac18f47f1286f7b...)
  RouteClassifier: route_classifier@v1   (sha256:f8c6c2abe6a954960...)

  verify('feature_extractor', 'v1') → True  + text embedding intact
  verify('route_classifier', 'v1')  → True  + route classifier intact

────────────────────────────────────────────────────────────────
  Step 6 — Tamper detection: modifying TextEmbedder breaks pipeline
────────────────────────────────────────────────────────────────
  Tamper detected — VerificationError: params.json content hash mismatch for feature_extractor@v1
  + Cryptographic chain caught modification of TextEmbedder
  + Restored: the entry verifies again

────────────────────────────────────────────────────────────────
  Summary
────────────────────────────────────────────────────────────────
  Registry entries: 2
    feature_extractor@v1  entry_hash=sha256:57ac18f47f1286f7b...
    route_classifier@v1  entry_hash=sha256:f8c6c2abe6a954960...

  Value delivered:
    + Raw ticket TEXT is the input — not hardcoded numeric features
    + 30-word vocabulary maps ticket words to a BoW vector
    + TextEmbedder (Dense 8x1) learns which words signal each routing queue
    + Composite pipeline routes text with accuracy 100.0% vs 33.3% baseline
    + Every routing decision traceable to exact TextEmbedder + RouteClassifier entry_hash
    + Upgrading either component changes composite_hash automatically
    + Tampering with any component is cryptographically detected

  Business value: 100.0% of tickets correctly routed without human triaging.
  At 1000 tickets/day: 1000 auto-routed, 0 need manual review.
```

> **Los hashes que ves arriba NO te van a coincidir, y eso es correcto.**
> Medido el 2026-08-20 sobre tres ejecuciones seguidas: las únicas líneas
> que cambian entre una y otra son las **dieciséis** que llevan un
> `sha256:`. El motivo está en el manifiesto de cada entrada, que guarda un
> `created_at` con la hora de reloj; el `entry_hash` cubre el manifiesto,
> así que registrar dos veces el mismo modelo da dos hashes distintos —y el
> hash compuesto, que deriva de los dos, cambia con ellos.
>
> Lo que **sí** te tiene que coincidir, carácter a carácter, es todo lo
> demás: la señal de cada texto (`0.0643`, `0.4597`, `0.8859`), las nueve
> decisiones con su confianza, el 100,0 % del pipeline frente al 33,3 % del
> baseline, y —dentro del manifiesto— el `model_hash` y el
> `params_content_hash`, que son hashes de CONTENIDO y no llevan hora. El
> entrenamiento es determinista: si esas cifras te salen distintas, ahí sí
> hay algo que mirar.


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
    train.csv                  — 36 tickets sintéticos (12 por categoría), texto crudo + 30 columnas bow_*
    test.csv                   — 9 tickets de test (3 por categoría)
  run_case.py                  — Script de demo extremo a extremo
  registry/                    — El registry P21 del ejemplo, VERSIONADO en el repo
    entries/feature_extractor/v1/   — manifest, params, firma, traza, informe
    entries/route_classifier/v1/    — lo mismo para la segunda etapa
    registry.json                   — índice del registry
```

**Ojo con `registry/`:** está en el repo, pero `run_case.py` lo **borra
entero y lo vuelve a crear** en cada ejecución (`shutil.rmtree` antes de
registrar). Correr el caso te deja el árbol sucio en esos dieciocho
ficheros, y el diff que verás será casi todo `created_at` y hashes. Los CSV
de `data/` sí van versionados y **no** se regeneran: el código sabe
generarlos, pero solo entra por ahí si faltan, y en el repo no faltan.

