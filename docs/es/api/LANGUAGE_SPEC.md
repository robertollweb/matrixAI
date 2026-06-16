# Especificación del lenguaje MatrixAI v1.0

> **Nota de desarrollo:** Este documento fue elaborado como entregable anticipado del corte PR5-C6
> (API REST completa y Studio desacoplado), durante la sesión de cierre de PR4 (2026-05-28).

**Fecha:** 2026-05-28  
**Estado:** Actual — refleja el estado de la versión v1.0 del runtime, el compilador, el stack de entrenamiento y el sistema de aprendizaje continuo.  
**Sustituye a:** `documentacion/MATRIXAI_LANGUAGE_SPEC_V0_1.md` (v0.1, 2026-04-28)

---

## Introducción

MatrixAI es un lenguaje para describir modelos de IA como grafos computacionales auditables. Un modelo no es una caja negra: es un programa con entradas explícitas, transformaciones explícitas, salidas explícitas y una traza de auditoría explícita. Cada decisión del modelo es trazable hasta un nodo con nombre en el grafo.

El ecosistema usa cuatro formatos de fichero, cada uno con un rol distinto:

| Formato | Extensión | Rol |
|---------|-----------|-----|
| Programa de modelo | `.mxai` | Declara el modelo: entradas, grafo de cómputo, salidas, acciones, auditoría |
| Especificación de entrenamiento | `.mxtrain` | Declara cómo entrenar el modelo: datos, épocas, pérdida, métricas |
| Contrato de acción | `.mxact` | Declara los límites legales de las acciones reales que puede desencadenar el modelo |
| Política continual | `.mxcontinual` | Declara detección de drift, disparadores de reentrenamiento y reglas de rollback |

Este documento especifica la sintaxis y la semántica de los cuatro formatos.

---

## Reglas léxicas comunes

### Identificadores

Los identificadores válidos siguen este patrón:

```
[A-Za-z_][A-Za-z0-9_]*
```

Los identificadores distinguen mayúsculas de minúsculas. `Risk`, `risk` y `RISK` son nombres distintos.

### Comentarios

Las líneas cuyo primer carácter no blanco es `#` se ignoran.

### Líneas en blanco

Las líneas en blanco se ignoran en cualquier posición.

### Bloques

Los bloques multilínea terminan con `END`.

### Números

Los literales numéricos son decimales no negativos: `0.75`, `0.90`, `50`, `8`. Los umbrales para acciones y verificación deben estar en `[0.0, 1.0]`.

### Cadenas de texto

Las cadenas se encierran entre comillas dobles: `"ops@example.com"`, `"correction"`.

---

## `.mxai` — Programa de modelo

### Bloques permitidos

Un programa `.mxai` puede contener estos bloques, en cualquier orden:

```
PROJECT
VECTOR
SEQUENCE
LAYER
FUNCTION
DISTRIBUTION
GRAPH
ACTION
AUDIT
IMPORT
```

`PROJECT` y `GRAPH` son obligatorios. El resto depende de la arquitectura del modelo.

---

### PROJECT

```
PROJECT <Identifier>
```

Declara el nombre del programa. Debe aparecer exactamente una vez.

```
PROJECT CreditScoring
```

---

### VECTOR

Declara un vector de entrada con nombre y un número fijo de campos numéricos.

```
VECTOR <Identifier>[<size>]
  <nombre_campo>[: <Tipo>]
  ...
END
```

Reglas:
- `size` debe coincidir exactamente con el número de campos declarados.
- Los nombres de campos deben ser identificadores válidos.
- Las anotaciones de tipo son opcionales. Los campos sin anotar son `Any` por defecto.
- Los campos ausentes en la entrada JSON se cargan como `0.0`.

Formatos de entrada aceptados en tiempo de ejecución (ambos son válidos):

```json
{ "income_score": 0.8, "debt_ratio": 0.3 }
```
```json
{ "Application": { "income_score": 0.8, "debt_ratio": 0.3 } }
```

Ejemplo:

```
VECTOR Application[5]
  income_score: Score
  credit_history: Score
  debt_ratio: Risk
  employment_years: Score
  loan_amount_ratio: Risk
END
```

---

### SEQUENCE

Declara una entrada de secuencia con nombre, para arquitecturas de tipo transformer.

```
SEQUENCE <Identifier>
  length = <entero>
  vocab_size = <entero>
END
```

Ejemplo:

```
SEQUENCE Input
  length = 8
  vocab_size = 32
END
```

---

### LAYER

Declara una capa de red neuronal con nombre, entradas y salidas tipadas, parámetros entrenables y operaciones. Se usa para construir arquitecturas multicapa y transformers.

```
LAYER <nombre>(<TipoEntrada>) -> <TipoSalida>
  PARAM <nombre_param> <Tipo>
  ...
  <salida> = <operacion>(...)
  ...
  result = <salida_final>
END
```

La última asignación en un bloque `LAYER` es implícitamente la salida. Las capas se invocan desde bloques `FUNCTION` usando `call_layer`.

Operaciones disponibles dentro de `LAYER`:

| Operación | Firma | Descripción |
|-----------|-------|-------------|
| `matmul` | `(Tensor, Tensor)` | Multiplicación de matrices |
| `dot` | `(Tensor, Tensor)` | Producto escalar |
| `scale` | `(Tensor, escalar)` | Escalado elemento a elemento |
| `residual` | `(Tensor, Tensor)` | Suma elemento a elemento (conexión residual) |
| `softmax` | `(Tensor)` | Normalización softmax |
| `gelu` | `(Tensor)` | Activación GELU |
| `layer_norm` | `(Tensor, gain, bias)` | Normalización de capa |
| `embedding_lookup` | `(embed_table, input)` | Búsqueda de embedding de token |
| `mean_pooling` | `(Tensor)` | Media sobre la dimensión de secuencia |

Ejemplo (bloque de atención transformer):

```
LAYER encoder_attn(Tensor[8]) -> Tensor[8]
  PARAM Wq Tensor[8, 8]
  PARAM Wk Tensor[8, 8]
  PARAM Wv Tensor[8, 8]
  PARAM Wo Tensor[8, 8]
  PARAM gain Tensor[8]
  PARAM bias Tensor[8]
  q = matmul(input, Wq)
  k = matmul(input, Wk)
  v = matmul(input, Wv)
  score = dot(q, k)
  scaled_score = scale(score, 0.35355339)
  attn_weight = softmax(scaled_score)
  attn = scale(v, attn_weight)
  proj = matmul(attn, Wo)
  res = residual(input, proj)
  result = layer_norm(res, gain, bias)
END
```

---

### FUNCTION

Declara un nodo de cómputo con nombre que evalúa una única expresión.

```
FUNCTION <Identifier>
  <salida>[: <Tipo>] = <expresión>
END
```

Cada bloque `FUNCTION` contiene exactamente una asignación. El estado del nodo se guarda en `state[NombreFuncion]`. La variable de salida también se guarda en `state[salida]`.

#### Expresiones soportadas

**sigmoid_linear** — clasificación binaria o puntuación de riesgo:

```
Y = sigmoid(W1 * X + b1)
Y: Risk = sigmoid(W1 * X + b1)
```

**softmax_linear** — clasificación multiclase:

```
Y = softmax(W1 * X + b1)
Y: ProbabilityMap = softmax(W1 * X + b1)
```

**linear_regression** — salida de regresión continua:

```
Y: Scalar = linear(W1 * X + b1)
```

**sigmoid_threshold** — puerta de activación desde una ruta de distribución:

```
Y = sigmoid(20 * (Risk.mean - 0.8))
Y: ActionSignal = sigmoid(50 * (Confidence.max - 0.95))
```

`Source.path` se resuelve contra las claves de distribución (`.mean`, `.max`, etc.) o valores directos de nodo.

**call_layer** — invocar un bloque `LAYER`:

```
embedded = call_layer(encoder_embed, Input)
```

Las expresiones que no coinciden con ninguno de los patrones anteriores se aceptan como `kind = unknown` con un aviso del `VerifierAgent`. Los programas de producción no deben depender de `unknown`.

---

### DISTRIBUTION

Declara una salida probabilística con nombre calculada a partir de un nodo `FUNCTION`.

```
DISTRIBUTION <Identifier>
  <variable> ~ <TipoDistribucion>(<fuente>)
END
```

Tipos de distribución soportados:

| Tipo | Forma | Claves disponibles |
|------|-------|--------------------|
| `Categorical` | `Confidence ~ Categorical(C)` | `.probabilities`, `.label`, `.max` |
| `Normal` | `Risk ~ Normal(R, uncertainty(Vector))` | `.mean`, `.sigma` |

El nodo de distribución se guarda en `state[NombreDistribucion]`. La variable también se guarda en `state[variable]`.

`Categorical` requiere que la fuente resuelva a un mapa de probabilidades (salida de `softmax`). `Normal` toma el primer argumento como media. `sigma` es `0.05` por defecto a menos que se especifique `uncertainty(Vector)`.

---

### GRAPH

Declara el orden de ejecución como un grafo acíclico dirigido.

```
GRAPH
  A -> B -> C
  A -> D -> E
END
```

Reglas:
- Cada línea debe contener al menos dos nodos separados por `->`.
- Se permiten varias líneas (para rutas ramificadas).
- Los nodos se ejecutan en el orden en que aparecen por primera vez en todas las cadenas.
- Los ciclos son rechazados por `VerifierAgent`.
- Todos los nodos referenciados deben estar declarados como `VECTOR`, `SEQUENCE`, `LAYER`, `FUNCTION`, `DISTRIBUTION` o `ACTION`.

---

### ACTION

Declara una acción discreta que el modelo puede desencadenar según una condición en tiempo de ejecución.

Se soportan dos políticas:

#### simulate_only (modo seguro)

```
ACTION <Identifier>
  WHEN <fuente> <operador> <umbral>
  POLICY simulate_only
  CALL simulated.<dominio>.<operacion>
END
```

El runtime evalúa la condición pero nunca ejecuta llamadas externas. Devuelve un resultado simulado. Se usa en desarrollo, pruebas y despliegues en sandbox.

#### real_with_audit (modo producción)

```
ACTION <Identifier>
  TARGET <capability>
  POLICY real_with_audit
  CONDITION <fuente> > <umbral>
  INPUT <param>: <Tipo>[, ...]
END
```

Ejecuta una llamada externa real cuando se cumple la condición. Requiere un contrato `.mxact` cargado al servir el modelo (`matrixai serve --contract`). Cada ejecución se firma, audita y registra.

Operadores de condición: `>`, `>=`, `<`, `<=`.

---

### AUDIT

Declara la ruta de traza de auditoría que el runtime sigue en cada ejecución.

```
AUDIT
  EXPLAIN <nodo> -> <nodo> -> ... -> <nodo>
END
```

Reglas:
- La ruta debe referenciar nodos declarados en `GRAPH`.
- En programas de clasificación y riesgo, la ruta debe empezar en el vector de entrada y terminar en una acción.
- `VerifierAgent` emite un aviso si falta `AUDIT`.

---

### IMPORT

Importa un modelo del registro como nodo con nombre. El modelo importado actúa como nodo de cómputo en el `GRAPH`.

```
IMPORT <Identifier> FROM registry <nombre>@<version> FROZEN
```

- `FROZEN` indica que los pesos del modelo importado están fijos y no se reentrenan.
- El nodo importado puede usarse en `GRAPH` y `AUDIT` igual que cualquier nodo local.

Ejemplo:

```
IMPORT FeatureExtractor FROM registry feature_extractor@v1 FROZEN
IMPORT RouteClassifier FROM registry route_classifier@v1 FROZEN

VECTOR TicketBOW[30]
  bow_account: Score
  ...
END

GRAPH
  TicketBOW -> FeatureExtractor -> RouteClassifier
END
```

---

## Sistema de tipos

Las anotaciones de tipo son opcionales. Los nodos y campos sin anotar son `Any` por defecto. El verificador de tipos (`matrixai typecheck`) valida las anotaciones donde están presentes.

### Tipos escalares

| Tipo | Rango | Uso |
|------|-------|-----|
| `Any` | — | Por defecto, sin restricción |
| `Scalar` | ℝ | Salida de regresión continua |
| `Integer` | ℤ | Conteo discreto o índice de clase |
| `Boolean` | {0, 1} | Indicador binario |
| `String` | — | Valor textual |

### Tipos nativos de IA

| Tipo | Rango | Uso |
|------|-------|-----|
| `Probability` | [0, 1] | Valor de probabilidad único |
| `Score` | [0, 1] | Característica de entrada normalizada |
| `Risk` | [0, 1] | Magnitud de riesgo |
| `Confidence` | [0, 1] | Confianza del modelo |
| `Logit` | ℝ | Puntuación bruta pre-softmax |
| `ProbabilityMap` | [0,1]ⁿ, suma=1 | Distribución de probabilidad multiclase |
| `Categorical` | — | Salida de distribución categórica |
| `Normal` | — | Salida de distribución normal |
| `Label` | — | Cadena de etiqueta de clase |
| `ActionSignal` | [0, 1] | Valor de activación de puerta de acción |

### Tipos tensor

| Tipo | Uso |
|------|-----|
| `Tensor[n]` | Tensor 1D de dimensión n |
| `Tensor[n, m]` | Tensor 2D |
| `Embedding[n]` | Vector de embedding de dimensión n |

### Sintaxis de anotación

En campos de `VECTOR`:

```
nombre_campo: Score
```

En salidas de `FUNCTION`:

```
R: Risk = sigmoid(W1 * Patient + b1)
```

En parámetros de `LAYER`:

```
PARAM Wq Tensor[8, 8]
```

---

## Modelo de ejecución

El runtime ejecuta los nodos del `GRAPH` en orden de declaración.

Para cada nodo:

1. **VECTOR / SEQUENCE** — carga valores numéricos desde `input_data`. Los campos ausentes se inicializan a `0.0`.
2. **FUNCTION** — evalúa la expresión declarada con el estado actual.
3. **DISTRIBUTION** — calcula la distribución a partir de la salida de función referenciada.
4. **ACTION** — evalúa la condición; ejecuta (real o simulada) si se cumple.
5. Todos los nodos añaden un paso a `trace`.

Salida del runtime:

```json
{
  "state": { "NombreNodo": <valor>, ... },
  "trace": [
    { "step": 1, "node": "Patient", "node_type": "vector", "status": "ok", "value": [...] },
    ...
  ],
  "actions": [
    { "name": "Notify", "activated": true, "simulated": true, ... }
  ]
}
```

Campos del resultado de acción:

```json
{
  "name": "DraftReply",
  "call": "simulated.email.draft",
  "source": "ReplyActivation",
  "operator": ">",
  "value": 0.99,
  "threshold": 0.90,
  "activated": true,
  "policy": "simulate_only",
  "simulated": true
}
```

---

## Niveles de conformidad

| Nivel | Requisito |
|-------|-----------|
| **1 — Parseable** | El documento se analiza en IR sin error |
| **2 — Verificable** | Sin errores de `VerifierAgent`; sin avisos críticos de `SafetyAgent` |
| **3 — Ejecutable** | El runtime puede ejecutar el grafo completo y devolver `state`, `trace`, `actions` |
| **4 — Compilable** | El compilador Python genera un módulo autónomo cuyo `run(input_data)` coincide con el runtime para todos los tipos de expresión soportados |

Un programa de producción v1.0 debe alcanzar el Nivel 4.

---

## `.mxact` — Contrato de acción

Define los límites legales dentro de los cuales puede ejecutarse una acción real. Se carga al servir con `matrixai serve --contract <fichero.mxact>`.

### Estructura

```
ACTION_CONTRACT <Identifier>
  CAPABILITY <nombre_capability>
  SCOPE
    <clave> = <valor>
    ...
  END
  DRY_RUN <required|optional|disabled>
  ROLLBACK <nombre_rollback>
  SANDBOX <required|not_required>
  HUMAN_APPROVAL <true|false>
  RATE_LIMIT per_minute=<n> per_hour=<n>
  SIGNATURE_REQUIRED <true|false>
END

ROLLBACK <nombre_rollback>
  CAPABILITY <nombre_capability>
  SCOPE
    <clave> = <valor>
    ...
  END
END
```

### Campos

| Campo | Descripción |
|-------|-------------|
| `CAPABILITY` | Capacidad requerida por nombre (p.ej. `email_send`, `database_write`) |
| `SCOPE` | Restricciones clave-valor sobre la acción (destinatarios permitidos, dominios, límites de tamaño, etc.) |
| `DRY_RUN` | Si se requiere un dry-run antes de la ejecución real |
| `ROLLBACK` | Nombre del procedimiento de rollback para deshacer la acción si es necesario |
| `SANDBOX` | Si la ejecución debe realizarse en sandbox |
| `HUMAN_APPROVAL` | Si un humano debe aprobar antes de la ejecución |
| `RATE_LIMIT` | Máximo de ejecuciones por ventana de tiempo |
| `SIGNATURE_REQUIRED` | Si el ActionTrace debe estar firmado con HMAC |

### Ejemplo

```
ACTION_CONTRACT TriggerAlert
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com", "alerts@example.com"]
    allowed_domains    = ["example.com"]
    max_subject_length = 200
    max_body_length    = 5000
  END
  DRY_RUN required
  ROLLBACK send_correction
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED true
END

ROLLBACK send_correction
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com", "alerts@example.com"]
    template = "correction"
  END
END
```

---

## `.mxcontinual` — Política de aprendizaje continuo

Define las reglas de detección de drift, disparadores de reentrenamiento y condiciones de rollback para un modelo en producción. Se usa con `matrixai serve --continual-policy` y el grupo de comandos `matrixai continual`.

### Estructura

```
CONTINUAL_POLICY <Identifier>
  TARGET_MODEL <ruta_al_.mxai>
  BASE_PARAMETER_SET <ruta_a_params.json>
  REGISTRY_NAME <nombre>
  BASE_VERSION <version>

  GROUND_TRUTH
    WINDOW_DAYS <n>
    REQUIRED_FIELD <nombre_campo>
  END

  DRIFT_DETECTION
    FEATURES [<campo>, ...]
    METHODS
      <campo>: <metodo> threshold=<valor>
      ...
    END
    MIN_SAMPLES <n>
    CHECK_FREQUENCY <daily|hourly|manual>
    REFERENCE_DATASET <nombre>
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES <n>
    MIN_GROUND_TRUTH_RATIO <valor>
    COOLDOWN_DAYS <n>
  END

  TRAINING
    METHOD <incremental_finetune|full_retrain>
    LEARNING_RATE_FACTOR <valor>
    MAX_EPOCHS <n>
    DATASET_MIX
      BASE_WEIGHT <valor>
      PRODUCTION_WEIGHT <valor>
      RECENCY_DECAY <linear|exponential|none>
    END
  END
END
```

### Métodos de detección de drift

| Método | Descripción |
|--------|-------------|
| `ks` | Test de Kolmogorov-Smirnov sobre la distribución de características |
| `psi` | Índice de Estabilidad de Población (Population Stability Index) |

### Ejemplo

```
CONTINUAL_POLICY CreditScoringContinual
  TARGET_MODEL examples/credit-scoring/credit_scoring.mxai
  BASE_PARAMETER_SET examples/credit-scoring/registry/entries/credit-scoring/v1.0/params.json
  REGISTRY_NAME credit-scoring
  BASE_VERSION v1.0

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
  END

  DRIFT_DETECTION
    FEATURES [income_score, credit_history, debt_ratio, employment_years, loan_amount_ratio]
    METHODS
      income_score: ks threshold=0.15
      credit_history: ks threshold=0.15
      debt_ratio: psi threshold=0.20
    END
    MIN_SAMPLES 5
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES 20
    MIN_GROUND_TRUTH_RATIO 0.5
    COOLDOWN_DAYS 1
  END

  TRAINING
    METHOD incremental_finetune
    LEARNING_RATE_FACTOR 0.1
    MAX_EPOCHS 10
    DATASET_MIX
      BASE_WEIGHT 0.6
      PRODUCTION_WEIGHT 0.4
      RECENCY_DECAY linear
    END
  END
END
```

---

## Ejemplos completos

### Clasificación de riesgo con acción (simulate_only)

```
PROJECT FallRisk

VECTOR Patient[5]
  age: Score
  mobility: Score
  medication_load: Risk
  previous_falls: Probability
  cognitive_state: Score
END

FUNCTION RiskModel
  R: Risk = sigmoid(W1 * Patient + b1)
END

FUNCTION AlertActivation
  A: ActionSignal = sigmoid(20 * (Risk.mean - 0.8))
END

DISTRIBUTION Risk
  Risk ~ Normal(R, uncertainty(Patient))
END

GRAPH
  Patient -> RiskModel -> Risk -> AlertActivation -> Notify
END

ACTION Notify
  WHEN AlertActivation > 0.90
  POLICY simulate_only
  CALL simulated.nurse_station.alert
END

AUDIT
  EXPLAIN Patient -> RiskModel -> Risk -> AlertActivation -> Notify
END
```

### Clasificación multiclase

```
PROJECT EmailAgent

VECTOR Email[8]
  urgency: Score
  sender_trust: Score
  topic_support: Probability
  topic_sales: Probability
  sentiment: Score
  has_attachment: Probability
  previous_interactions: Score
  language_confidence: Confidence
END

FUNCTION Classifier
  C: ProbabilityMap = softmax(W1 * Email + b1)
END

DISTRIBUTION Confidence
  Confidence ~ Categorical(C)
END

FUNCTION ReplyActivation
  A: ActionSignal = sigmoid(50 * (Confidence.max - 0.95))
END

GRAPH
  Email -> Classifier -> Confidence -> ReplyActivation -> DraftReply
END

ACTION DraftReply
  WHEN ReplyActivation > 0.90
  POLICY simulate_only
  CALL simulated.email.draft
END

AUDIT
  EXPLAIN Email -> Classifier -> Confidence -> ReplyActivation -> DraftReply
END
```

### Regresión lineal

```
PROJECT HousePricing

VECTOR Property[4]
  area: Score
  rooms: Score
  location_score: Score
  age_years: Score
END

FUNCTION PriceModel
  predicted_price: Scalar = linear(W1 * Property + b1)
END

GRAPH
  Property -> PriceModel
END

AUDIT
  EXPLAIN Property -> PriceModel
END
```

### Clasificador basado en transformer

```
PROJECT TransformerClassifier

SEQUENCE Input
  length = 8
  vocab_size = 32
END

LAYER encoder_embed(Tensor[8]) -> Tensor[8]
  PARAM embed_table Tensor[32, 8]
  PARAM gain Tensor[8]
  PARAM bias Tensor[8]
  embedded = embedding_lookup(embed_table, input)
  pooled = mean_pooling(embedded)
  result = layer_norm(pooled, gain, bias)
END

LAYER encoder_attn(Tensor[8]) -> Tensor[8]
  PARAM Wq Tensor[8, 8]
  PARAM Wk Tensor[8, 8]
  PARAM Wv Tensor[8, 8]
  PARAM Wo Tensor[8, 8]
  PARAM gain Tensor[8]
  PARAM bias Tensor[8]
  q = matmul(input, Wq)
  k = matmul(input, Wk)
  v = matmul(input, Wv)
  score = dot(q, k)
  scaled_score = scale(score, 0.35355339)
  attn_weight = softmax(scaled_score)
  attn = scale(v, attn_weight)
  proj = matmul(attn, Wo)
  res = residual(input, proj)
  result = layer_norm(res, gain, bias)
END

LAYER classifier(Tensor[8]) -> Tensor[2]
  PARAM W Tensor[8, 2]
  PARAM b Tensor[2]
  proj = matmul(input, W)
  result = residual(proj, b)
END

FUNCTION Embed
  embedded = call_layer(encoder_embed, Input)
END

FUNCTION AttnBlock
  attn_block = call_layer(encoder_attn, embedded)
END

FUNCTION Logits
  logits = call_layer(classifier, attn_block)
END

GRAPH
  Input -> Embed -> AttnBlock -> Logits
END
```

---

## Lo que v1.0 no incluye

Lo siguiente está fuera del alcance de v1.0 y puede aparecer en versiones futuras:

- Grafos dinámicos o ciclos controlados
- Ejecución asíncrona o distribuida
- Optimización automática del grafo (reescritura estructural)
- Funciones de pérdida personalizadas definidas en `.mxai`
- Cabezas de regresión con múltiples salidas
- Aprendizaje en línea dentro de un único proceso servidor (usar `.mxcontinual` para aprendizaje continuo offline)
- Precedencia a nivel de parser para expresiones matemáticas arbitrarias (usar `LAYER` para arquitecturas complejas)
