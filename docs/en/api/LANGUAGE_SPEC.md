# MatrixAI Language Specification v1.0

**Date:** 2026-05-28  
**Status:** Current — reflects the v1.0 release state of the runtime, compiler, training stack and continual learning system.  
**Supersedes:** `documentacion/MATRIXAI_LANGUAGE_SPEC_V0_1.md` (v0.1, 2026-04-28)

---

## Overview

MatrixAI is a language for describing AI models as auditable computation graphs. A model is not a black box: it is a program with explicit inputs, explicit transformations, explicit outputs and an explicit audit trail. Every decision the model makes is traceable to a named node in the graph.

The ecosystem uses four file formats, each with a distinct role:

| Format | Extension | Role |
|--------|-----------|------|
| Model program | `.mxai` | Declares the model: inputs, computation graph, outputs, actions, audit |
| Training spec | `.mxtrain` | Declares how to train the model: data, epochs, loss, metrics |
| Action contract | `.mxact` | Declares the legal boundaries of real actions the model can trigger |
| Continual policy | `.mxcontinual` | Declares drift detection, retraining triggers and rollback rules |

This document specifies the syntax and semantics of all four formats.

---

## Common lexical rules

### Identifiers

Valid identifiers follow this pattern:

```
[A-Za-z_][A-Za-z0-9_]*
```

Identifiers are case-sensitive. `Risk`, `risk` and `RISK` are different names.

### Comments

Lines whose first non-whitespace character is `#` are ignored.

### Blank lines

Blank lines are ignored everywhere.

### Blocks

Multi-line blocks end with `END`.

### Numbers

Numeric literals are non-negative decimals: `0.75`, `0.90`, `50`, `8`. Thresholds for actions and verification must be in `[0.0, 1.0]`.

### String literals

Strings are enclosed in double quotes: `"ops@example.com"`, `"correction"`.

---

## `.mxai` — Model program

### Allowed blocks

A `.mxai` program may contain these blocks, in any order:

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

`PROJECT` and `GRAPH` are required. Everything else depends on the model architecture.

---

### PROJECT

```
PROJECT <Identifier>
```

Declares the program name. Must appear exactly once.

```
PROJECT CreditScoring
```

---

### VECTOR

Declares a named input vector with a fixed number of numeric fields.

```
VECTOR <Identifier>[<size>]
  <field_name>[: <Type>]
  ...
END
```

Rules:
- `size` must equal the number of declared fields exactly.
- Field names must be valid identifiers.
- Type annotations are optional. Unannotated fields default to `Any`.
- Missing fields in JSON input are loaded as `0.0`.

Input formats accepted at runtime (either is valid):

```json
{ "income_score": 0.8, "debt_ratio": 0.3 }
```
```json
{ "Application": { "income_score": 0.8, "debt_ratio": 0.3 } }
```

Example:

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

Declares a named sequence input, used by transformer-style architectures.

```
SEQUENCE <Identifier>
  length = <integer>
  vocab_size = <integer>
END
```

Example:

```
SEQUENCE Input
  length = 8
  vocab_size = 32
END
```

---

### LAYER

Declares a named neural network layer with typed inputs, typed outputs, learnable parameters and operations. Used to build multi-layer and transformer architectures.

```
LAYER <name>(<InputType>) -> <OutputType>
  PARAM <param_name> <Type>
  ...
  <output> = <operation>(...)
  ...
  result = <final_output>
END
```

The last assignment in a `LAYER` block is implicitly the output. Layers are called from `FUNCTION` blocks using `call_layer`.

Available operations inside `LAYER`:

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `matmul` | `(Tensor, Tensor)` | Matrix multiplication |
| `dot` | `(Tensor, Tensor)` | Dot product |
| `scale` | `(Tensor, scalar)` | Element-wise scaling |
| `residual` | `(Tensor, Tensor)` | Element-wise addition (residual connection) |
| `softmax` | `(Tensor)` | Softmax normalization |
| `gelu` | `(Tensor)` | GELU activation |
| `layer_norm` | `(Tensor, gain, bias)` | Layer normalization |
| `embedding_lookup` | `(embed_table, input)` | Token embedding lookup |
| `mean_pooling` | `(Tensor)` | Mean over sequence dimension |

Example (transformer encoder block):

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

Declares a named computation node that evaluates one expression.

```
FUNCTION <Identifier>
  <output>[: <Type>] = <expression>
END
```

Each `FUNCTION` block contains exactly one assignment. The node state is stored as `state[FunctionName]`. The output variable is also stored as `state[output]`.

#### Supported expressions

**sigmoid_linear** — binary classification or risk scoring:

```
Y = sigmoid(W1 * X + b1)
Y: Risk = sigmoid(W1 * X + b1)
```

**softmax_linear** — multi-class classification:

```
Y = softmax(W1 * X + b1)
Y: ProbabilityMap = softmax(W1 * X + b1)
```

**linear_regression** — continuous regression output:

```
Y: Scalar = linear(W1 * X + b1)
```

**sigmoid_threshold** — activation gate from a distribution path:

```
Y = sigmoid(20 * (Risk.mean - 0.8))
Y: ActionSignal = sigmoid(50 * (Confidence.max - 0.95))
```

`Source.path` resolves against distribution keys (`.mean`, `.max`, etc.) or direct node values.

**call_layer** — invoke a `LAYER` block:

```
embedded = call_layer(encoder_embed, Input)
```

Expressions that do not match any of the above are accepted as `kind = unknown` with a `VerifierAgent` warning. Programs for production should not rely on `unknown`.

---

### DISTRIBUTION

Declares a named probabilistic output computed from a `FUNCTION` node.

```
DISTRIBUTION <Identifier>
  <variable> ~ <DistributionType>(<source>)
END
```

Supported distribution types:

| Type | Form | Keys available |
|------|------|----------------|
| `Categorical` | `Confidence ~ Categorical(C)` | `.probabilities`, `.label`, `.max` |
| `Normal` | `Risk ~ Normal(R, uncertainty(Vector))` | `.mean`, `.sigma` |

The distribution node is stored as `state[DistributionName]`. The variable is also stored as `state[variable]`.

`Categorical` requires the source to resolve to a probability map (output of `softmax`). `Normal` takes the first argument as the mean. `sigma` defaults to `0.05` unless `uncertainty(Vector)` is specified.

---

### GRAPH

Declares the execution order as a directed acyclic graph.

```
GRAPH
  A -> B -> C
  A -> D -> E
END
```

Rules:
- Each line must contain at least two nodes separated by `->`.
- Multiple lines are allowed (for branching paths).
- Nodes execute in the order they first appear across all chains.
- Cycles are rejected by `VerifierAgent`.
- Every referenced node must be declared as `VECTOR`, `SEQUENCE`, `LAYER`, `FUNCTION`, `DISTRIBUTION` or `ACTION`.

---

### ACTION

Declares a discrete action that the model may trigger based on a runtime condition.

Two policies are supported:

#### simulate_only (safe mode)

```
ACTION <Identifier>
  WHEN <source> <operator> <threshold>
  POLICY simulate_only
  CALL simulated.<domain>.<operation>
END
```

The runtime evaluates the condition but never executes external calls. Returns a simulated result. Used in development, testing and sandboxed deployments.

#### real_with_audit (production mode)

```
ACTION <Identifier>
  TARGET <capability>
  POLICY real_with_audit
  CONDITION <source> > <threshold>
  INPUT <param>: <Type>[, ...]
END
```

Executes a real external call when the condition is met. Requires a `.mxact` contract to be loaded at serve time (`matrixai serve --contract`). Every execution is signed, audited and recorded.

Condition operators: `>`, `>=`, `<`, `<=`.

---

### AUDIT

Declares the audit trail path that the runtime traces for each execution.

```
AUDIT
  EXPLAIN <node> -> <node> -> ... -> <node>
END
```

Rules:
- The path must reference nodes declared in `GRAPH`.
- In classification and risk programs, the path should start at the input vector and end at an action.
- `VerifierAgent` emits a warning if `AUDIT` is absent.

---

### IMPORT

Imports a model from the registry as a named node. The imported model acts as a computation node in the `GRAPH`.

```
IMPORT <Identifier> FROM registry <name>@<version> FROZEN
```

- `FROZEN` means the imported model's weights are fixed and not retrained.
- The imported node can be used in `GRAPH` and `AUDIT` like any local node.

Example:

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

## Type system

Type annotations are optional. Unannotated nodes and fields default to `Any`. The type checker (`matrixai typecheck`) validates annotations where present.

### Scalar types

| Type | Range | Use |
|------|-------|-----|
| `Any` | — | Default, no constraint |
| `Scalar` | ℝ | Continuous regression output |
| `Integer` | ℤ | Discrete count or class index |
| `Boolean` | {0, 1} | Binary flag |
| `String` | — | Text value |

### AI-native types

| Type | Range | Use |
|------|-------|-----|
| `Probability` | [0, 1] | Single probability value |
| `Score` | [0, 1] | Normalized input feature |
| `Risk` | [0, 1] | Risk magnitude |
| `Confidence` | [0, 1] | Model confidence |
| `Logit` | ℝ | Pre-softmax raw score |
| `ProbabilityMap` | [0,1]ⁿ, sum=1 | Multi-class probability distribution |
| `Categorical` | — | Categorical distribution output |
| `Normal` | — | Normal distribution output |
| `Label` | — | Class label string |
| `ActionSignal` | [0, 1] | Action activation gate value |

### Tensor types

| Type | Use |
|------|-----|
| `Tensor[n]` | 1D tensor of dimension n |
| `Tensor[n, m]` | 2D tensor |
| `Embedding[n]` | Embedding vector of dimension n |

### Annotation syntax

In `VECTOR` fields:

```
field_name: Score
```

In `FUNCTION` outputs:

```
R: Risk = sigmoid(W1 * Patient + b1)
```

In `LAYER` parameters:

```
PARAM Wq Tensor[8, 8]
```

---

## Execution model

The runtime executes `GRAPH` nodes in declaration order.

For each node:

1. **VECTOR / SEQUENCE** — load numeric values from `input_data`. Missing fields default to `0.0`.
2. **FUNCTION** — evaluate the declared expression with current state.
3. **DISTRIBUTION** — compute the distribution from the referenced function output.
4. **ACTION** — evaluate the condition; execute (real or simulated) if met.
5. All nodes append a step to `trace`.

Runtime output:

```json
{
  "state": { "NodeName": <value>, ... },
  "trace": [
    { "step": 1, "node": "Patient", "node_type": "vector", "status": "ok", "value": [...] },
    ...
  ],
  "actions": [
    { "name": "Notify", "activated": true, "simulated": true, ... }
  ]
}
```

Action result fields:

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

## Verification levels

| Level | Requirement |
|-------|-------------|
| **1 — Parseable** | The document parses into IR without error |
| **2 — Verifiable** | No `VerifierAgent` errors; no critical `SafetyAgent` warnings |
| **3 — Executable** | The runtime can execute the full graph and return `state`, `trace`, `actions` |
| **4 — Compilable** | The Python compiler generates a standalone module whose `run(input_data)` matches the runtime for all supported expression kinds |

A v1.0 production program should reach Level 4.

---

## `.mxact` — Action contract

Defines the legal boundaries within which a real action may execute. Loaded at serve time with `matrixai serve --contract <file.mxact>`.

### Structure

```
ACTION_CONTRACT <Identifier>
  CAPABILITY <capability_name>
  SCOPE
    <key> = <value>
    ...
  END
  DRY_RUN <required|optional|disabled>
  ROLLBACK <rollback_name>
  SANDBOX <required|not_required>
  HUMAN_APPROVAL <true|false>
  RATE_LIMIT per_minute=<n> per_hour=<n>
  SIGNATURE_REQUIRED <true|false>
END

ROLLBACK <rollback_name>
  CAPABILITY <capability_name>
  SCOPE
    <key> = <value>
    ...
  END
END
```

### Fields

| Field | Description |
|-------|-------------|
| `CAPABILITY` | Named capability required (e.g. `email_send`, `database_write`) |
| `SCOPE` | Key-value restrictions on the action (allowed recipients, domains, size limits, etc.) |
| `DRY_RUN` | Whether a dry-run is required before real execution |
| `ROLLBACK` | Name of the rollback procedure to undo the action if needed |
| `SANDBOX` | Whether execution must be sandboxed |
| `HUMAN_APPROVAL` | Whether a human must approve before execution |
| `RATE_LIMIT` | Maximum executions per time window |
| `SIGNATURE_REQUIRED` | Whether the ActionTrace must be HMAC-signed |

### Example

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

## `.mxcontinual` — Continual learning policy

Defines drift detection rules, retraining triggers and rollback conditions for a model in production. Used with `matrixai serve --continual-policy` and the `matrixai continual` command group.

### Structure

```
CONTINUAL_POLICY <Identifier>
  TARGET_MODEL <path_to_.mxai>
  BASE_PARAMETER_SET <path_to_params.json>
  REGISTRY_NAME <name>
  BASE_VERSION <version>

  GROUND_TRUTH
    WINDOW_DAYS <n>
    REQUIRED_FIELD <field_name>
  END

  DRIFT_DETECTION
    FEATURES [<field>, ...]
    METHODS
      <field>: <method> threshold=<value>
      ...
    END
    MIN_SAMPLES <n>
    CHECK_FREQUENCY <daily|hourly|manual>
    REFERENCE_DATASET <name>
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES <n>
    MIN_GROUND_TRUTH_RATIO <value>
    COOLDOWN_DAYS <n>
  END

  TRAINING
    METHOD <incremental_finetune|full_retrain>
    LEARNING_RATE_FACTOR <value>
    MAX_EPOCHS <n>
    DATASET_MIX
      BASE_WEIGHT <value>
      PRODUCTION_WEIGHT <value>
      RECENCY_DECAY <linear|exponential|none>
    END
  END
END
```

### Drift detection methods

| Method | Description |
|--------|-------------|
| `ks` | Kolmogorov-Smirnov test on feature distribution |
| `psi` | Population Stability Index |

### Example

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

## Complete examples

### Risk classification with action (simulate_only)

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

### Multi-class classification

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

### Linear regression

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

### Transformer-based classifier

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

### Composite pipeline with IMPORT

```
PROJECT TextRoutingPipeline

IMPORT FeatureExtractor FROM registry feature_extractor@v1 FROZEN
IMPORT RouteClassifier FROM registry route_classifier@v1 FROZEN

VECTOR TicketBOW[30]
  bow_account: Score
  bow_amount: Score
  bow_api: Score
  bow_error: Score
  bow_help: Score
  bow_invoice: Score
  bow_payment: Score
  bow_plan: Score
  bow_problem: Score
  bow_refund: Score
  bow_reset: Score
  bow_server: Score
  bow_slow: Score
  bow_timeout: Score
  bow_upgrade: Score
  bow_account: Score
  bow_cancel: Score
  bow_charge: Score
  bow_connection: Score
  bow_contract: Score
  bow_crash: Score
  bow_demo: Score
  bow_discount: Score
  bow_enterprise: Score
  bow_features: Score
  bow_fix: Score
  bow_issue: Score
  bow_login: Score
  bow_need: Score
  bow_network: Score
END

GRAPH
  TicketBOW -> FeatureExtractor -> RouteClassifier
END

AUDIT
  EXPLAIN TicketBOW -> FeatureExtractor -> RouteClassifier
END
```

---

## What v1.0 does not include

The following are out of scope for v1.0 and may appear in future versions:

- Dynamic graphs or controlled cycles
- Asynchronous or distributed execution
- Automatic graph optimization (structural rewriting)
- Custom loss functions defined in `.mxai`
- Multi-output regression heads
- Online learning within a single serve process (use `.mxcontinual` for offline continual learning)
- Parser-level precedence for arbitrary mathematical expressions (use `LAYER` for complex architectures)
