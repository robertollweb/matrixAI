# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Transformer blocks** — `BLOCK <name> TRANSFORMER` (multi-head attention,
  feed-forward, layer norm, configurable positional encoding) over a
  `SEQUENCE` input, with a byte-level tokenizer for text classification.
  Trained end-to-end on the torch/GPU backend, evaluated and exported
  through the same CLI cycle as dense and composite networks (`train`,
  `evaluate`, `export-onnx`, `export-bundle`). See
  `examples/transformer-classifier.mxai`.
- **Model generation from real data** — `generate_project_from_dataset` /
  `generate_temporal_project_from_dataset` (`matrixai.training.dataset_project`)
  build a runnable `.mxai` + `.mxtrain` pair directly from a CSV instead of a
  prompt: column type and range inference, one-hot categoricals, temporal
  columns, target-column candidate detection (classification or regression),
  and generated feature/target name mapping recorded in a provenance trail.
  This is the engine behind MatrixAI Studio's "Create from data" flow.

### Fixed

- **Regression targets in any scale now converge** — the regression target
  was trained on raw values while features were normalized to `[0, 1]`;
  above roughly `O(100)` this caused gradient explosion and a dead-ReLU
  collapse to the output bias (constant prediction). The target is now
  normalized with the same range mechanism as the features, and MAE/RMSE
  are rescaled back to the original units for reporting. R² is unaffected
  (scale-invariant). Confirmed on a Celsius→Kelvin dataset: R² −0.0001 → 0.999993.

---

## [1.2.0] — 2026-07-08

Feature release: self-usable model export bundles, typed prompt fields, and
first-class support for large models (billions of parameters) across the whole
train → save → infer → export cycle. Validated end-to-end on real hardware with
a 2.95B-parameter dense model (A100 80 GB).

### Added

- **Self-usable export bundles (HuggingFace-style)** — the edge bundle is now a
  complete, standalone package: `model.onnx`, `predict.py`, `inference_spec.json`,
  `requirements.txt`, an example input and its expected output. `predict.py`
  accepts **raw human values** (e.g. `"TORNO"`, `edad=54`) and applies the same
  normalization / one-hot encoding the model was trained with, so predictions
  match the source environment exactly — no MatrixAI installation required, only
  `onnxruntime`. CLI: `matrixai export-bundle --inference-metadata`.
- **Typed prompt fields** — declare feature types directly in the prompt
  (`FEATURES: edad: Scalar en [18, 95]`, `Integer[1, 10]`, `Boolean`,
  `Categorical[...]`, output `ProbabilityMap[NO, SI]`). Declared types and ranges
  are honoured end-to-end: deterministic generator, LLM proposals (validated
  against the prompt types — a proposal cannot override them), synthetic data,
  training and export metadata. Categorical fields expand to one-hot; an explicit
  two-label `ProbabilityMap` produces a 2-class softmax head.
- **Large models: binary weights format `.mxw`** — JSON header + raw float32
  blobs, SHA-256 content hash with tamper detection, atomic writes. Weights format
  is user-selectable (`json` | `binary`); binary is the default above 50M
  parameters (`MATRIXAI_TORCH_NATIVE_MIN_PARAMS`).
- **Large models: resource estimator** — params / VRAM / RAM / disk / time are
  estimated per weights format *before* training or saving
  (`estimate_model_resources`), from the parameter manifest in O(#tensors).
- **Large models: torch end-to-end** — above the threshold, training keeps the
  weights as tensors (never converted to Python lists), evaluation and the
  collapse probe run in torch (GPU when available), inference on a saved model
  does a single torch forward from the `.mxw`, and training can **resume** from
  saved binary weights.
- **Large models: ONNX external-data export** — above the ~2 GiB protobuf limit
  the exporter switches to the standard external-data layout (`model.onnx` +
  `model.onnx.data`) instead of failing. Tensor blobs are **streamed** straight
  from the `.mxw` into an uncompressed ZIP (constant memory, content hash
  re-verified while copying), and large exports are delivered as a **streamed
  download** instead of inline base64. WASM export remains limited to models that
  fit in the browser.

### Fixed

- Generating a very large model from a prompt no longer takes ~1 hour: the
  backend-contract manifest reports metadata instead of materializing every
  initial weight above 65k elements per tensor (4B params: ~1h → <1s).
- Streamed ZIP entries over 4 GiB no longer fail with `File size too large, try
  using force_zip64` (ZIP64 headers are always written for the weights entry).
- A `Boolean` prompt field can no longer receive a numeric range from an LLM
  proposal.
- Resuming training with a learning rate that never improves returns the
  starting weights instead of the worst epoch.
- Export with both a live training job and a saved model present no longer
  fails with "save the model first" for large models.

---

## [1.1.1] — 2026-06-25

Release de correcciones tras validación GPU real (Colab + RTX 2000 Ada). El camino denso
ahora usa la GPU de verdad de extremo a extremo; varios bugs de generación desde prompt y
de cancelación/serialización quedaron resueltos.

### Fixed

- **Generación de modelo desde prompt (red densa)** — prompts en lenguaje natural con
  rangos y cabeceras (`vibracion_axial [0-50]`, `FEATURES NUMÉRICAS (24), …:`) ya no meten
  prosa como campos (24 campos limpios); la profundidad se detecta con "Dense" en medio
  (`12 capas Dense ocultas` → 12 capas, antes 4 por defecto); las etiquetas salen de
  `ProbabilityMap[...]`/`Label[...]` (antes `class_a/b/c`); y "red densa pura" / "SOLO
  capas Dense" fuerza red densa en vez de enrutar a composite por la palabra "profunda".
- **Stop / cancelación** — el entrenamiento torch para entre lotes y **libera la VRAM al
  instante** (antes la traza de la excepción retenía los tensores GPU; la GPU se quedaba
  ocupada tras Stop). Cubre dense y composite.
- **GPU infrautilizada** — en CUDA el trainer denso ignora el `BATCH size=8` autogenerado y
  usa un batch grande para llenar la GPU (tunable por `MATRIXAI_GPU_BATCH`, default 16384).
- **"Se queda pensando" al acabar** — la prueba de colapso (M7) corre por torch/GPU en vez
  de 4 forwards en Python (O(params)); el resultado de entrenamiento ya no arrastra los
  pesos completos (se leen aparte para guardar/exportar), evitando respuestas enormes.

### Added

- **`MATRIXAI_GPU_BATCH`** — tamaño de batch por defecto en CUDA (16384). Bájalo si una red
  muy grande da OOM en una GPU pequeña.

### Changed

- **Generación de dataset sintético: nunca ejecuta el modelo para etiquetar** — los valores
  salen de los rangos (LLM/"Sugerir rangos") y las etiquetas de reglas de dominio del LLM o,
  si no hay, aleatorias. Se retiró el etiquetado por runtime/torch de la red sin entrenar
  (colgaba con redes grandes). Aplica a web, playground API y CLI.
- **Límites de filas configurables por perfil en toda la superficie** — frontend sin tope
  artificial; el CLI `generate-dataset` respeta `MATRIXAI_LIMITS_PROFILE`/`MATRIXAI_MAX_ROWS`;
  aviso cuando el perfil recorta las filas pedidas.

---

## [1.1.0] — 2026-06-18

### Added

- **ONNX / WASM export for composite networks (P19)** — residual blocks, LayerNorm,
  Dropout (identity at inference), native embeddings and concat now lower to ONNX
  (and WASM by delegation), with onnxruntime↔reference equivalence validation. The
  edge bundle exports them too. Dense export is unchanged.
- **LLM as domain simulator** — the schema designer can propose bounded threshold
  rules (`feature OP value`, AND/OR) that a deterministic evaluator applies to label
  synthetic data with plausible, learnable signal instead of a toy mapping. Falls back
  to coherent labelling when no rules are usable; collapsed/missing-class datasets are
  flagged honestly.
- **Native embeddings from prompt** — the schema can declare high-cardinality
  categoricals (`field:vocab`); the composite generator emits `EMBEDDING` + `CONCAT`
  for them, with integer-index synthetic data.

### Changed

- Training epochs and `early_stop` from the prompt are honoured by the composite
  generator too (not only dense). The epoch sanity ceiling is 1000.
- The training wall-clock budget (`MATRIXAI_TRAIN_TIMEOUT`) can be disabled with `0`
  (train to completion); the default stays 300s.

### Fixed

- Prompt label extraction no longer swallows trailing prose (e.g. `niveles BAJO MEDIO
  ALTO con una red…`) and now parses space-separated labels.
- Sequence detection aligned across the generator and the playground so a tabular
  composite (e.g. with embeddings) is not mislabelled as a sequence.

---

## [1.0.0] — 2026-05-30

First stable public release.

### Language and runtime

- **`.mxai` language** — complete type system: scalars, vectors, matrices, embeddings,
  probability distributions and composite types. Node types: `VECTOR`, `FUNCTION`,
  `DENSE`, `TRANSFORMER`, `ACTION`, `EMBEDDING`. Explicit input/output declarations,
  named parameters, fully auditable computation graph.
- **`.mxtrain` training spec** — supervised training with SGD, Adam and L-BFGS optimisers;
  loss functions: binary cross-entropy, MSE, categorical cross-entropy. Configurable
  epochs, batch size and learning rate. Reproducible training with deterministic seeding.
- **`.mxact` action contracts** — HMAC-signed action traces with dry-run, approval gates,
  rollback and tamper detection. Real-action execution governed by
  `MATRIXAI_ALLOW_REAL_ACTIONS` environment variable.
- **`.mxcontinual` policies** — drift detection, automatic versioning, rollback triggers
  and configurable retraining windows.

### CLI

Complete command surface:

| Command | Description |
|---------|-------------|
| `matrixai init` | Scaffold a new project from a template |
| `matrixai train` | Train a model from a `.mxtrain` spec |
| `matrixai run` | Run inference on a trained model |
| `matrixai serve` | Serve a model over HTTP with auth and rate limiting |
| `matrixai prompt` | Generate a `.mxai` program from a natural-language prompt |
| `matrixai playground` | Launch the local prompt-to-runtime playground |
| `matrixai pack` | Package a model for Docker deployment |
| `matrixai export` | Export a model to ONNX or WASM |
| `matrixai registry` | Push, pull, verify and inspect model registry entries |
| `matrixai keys` | Rotate and list signing keys |
| `matrixai continual` | Manage continual learning policies |

### HTTP server

- REST API: `/predict`, `/execute-action`, `/feedback`, `/health`, `/metrics`
- Prometheus metrics endpoint: request counts, latency, drift gauges
- API key authentication (`X-API-Key` header and `Authorization: Bearer`)
- Configurable CORS origins and rate limiting (sliding window per IP)
- OpenAPI spec at `/docs`

### Model registry

- Versioned, signed entries with HMAC verification
- `matrixai_version` field on every entry for compatibility tracking
- Tamper detection across registry, traces and parameter files
- Signing key rotation with historical verification by fingerprint

### Training and learning

- Classification, risk scoring and regression pipelines
- Dense networks, transformer encoders, composite architectures
- GPU training via optional PyTorch backend (`pip install matrixai-core[torch]`)
- Synthetic data generation for rapid prototyping
- Continual learning: drift detection, rollback and automatic retraining

### Export and deployment

- ONNX export with equivalence validation (`pip install matrixai-core[export]`)
- WASM bundles for browser and edge deployment
- `matrixai pack --docker` generates a production-ready Dockerfile and Compose file

### Distribution

- PyPI package: `pip install matrixai-core`
- SHA-256 checksums published with each release
- PyPI Trusted Publishing — no manual credentials in CI

### Project templates

- `matrixai init --template classification` — binary classification with a ready-to-train
  dataset, model and training spec included

### License

AGPL-3.0-only. See `LICENSE`.

---

## [Unreleased]

Nothing yet.

[1.0.0]: https://github.com/robertollweb/matrixAI/releases/tag/v1.0.0
[Unreleased]: https://github.com/robertollweb/matrixAI/compare/v1.0.0...HEAD
