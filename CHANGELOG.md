# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.7.1] — 2026-09-06

`attest` read the first ONNX output and thresholded it at 0.5 — right by
accident on a binary model, wrong on anything else.

### Fixed
- **`matrixai attest` now chooses the ONNX output by what it means, not by
  its position.** A 3-class logistic regression on iris (accuracy 0.9733,
  fixed model and dataset, versioned as a test fixture) was attested at
  0.6667: `attest` took `session.run(None, ...)[0]`, the first tensor a
  standard converter emits, and thresholded it at 0.5 — the right column by
  accident in binary models (`label` comes first), the wrong one with three
  classes. The new `onnx_salida.py` picks the output by declared task, name,
  type and the **semantics of the graph itself** (a `Sigmoid` node declares
  a probability; a bare `MatMul` declares nothing, and an ambiguous output
  is refused with a request for an explicit map instead of a guess).
  `argmax` now returns the real label — from a ZipMap's keys, from declared
  classes, or deduced from the target column and said in the receipt —
  never a bare index. Text and non-consecutive classes are supported. The
  receipt gains `models[].output_spec`: which output was read, what it
  means, and where the classes came from. New `--output-name`,
  `--output-kind` and `--classes` flags for the ambiguous case.

---

## [1.7.0] — 2026-08-27

The receipt reaches the product, the envelope follows the spec, the package
can finally prove that it reproduces — and MatrixAI stops needing to have
trained the model in order to say something verifiable about it.

### Added
- **`matrixai attest <model.onnx> --data <eval.csv>`** — a receipt for a
  model **MatrixAI did not train**. It ties the number to the digests of
  the model and of the evaluation data, and records the environment it
  claims to attest (ORT version, providers, python, platform). What makes
  it worth anything is what it *refuses* to claim: the receipt carries
  `evidence.does_not_attest` and `models[].provenance: "external"`, and
  says in plain words that it does not attest how the model was trained
  nor on what data — a copied model produces exactly the same receipt as
  your own. Its ceiling is **A1**, and that is written in the receipt, not
  only in the docs.
- **An `onnx` executor in the pipeline engine.** Declared input types are
  checked against `meta.type` and what would be lost is **rejected** rather
  than silently coerced; a dynamic axis is reported as *not checked*
  instead of wrongly checked.
- **`matrixai bom <package>` → CycloneDX 1.6 ML-BOM.** Validated against
  the official schema, which ships in `tests/data/` so the test checks it
  every day and not only the day it was downloaded. It is **deterministic**
  — the serial number comes from `manifest_sha256`, not `uuid4()`, because
  a BOM that changes on every run can be neither compared nor signed — and
  `--missing` enumerates what it cannot say.
- **`matrixai attest --in-toto`** emits the receipt as an in-toto Statement
  (`application/vnd.in-toto+json`, `subject[]` from the digests we already
  had, a `predicateType` versioned by URL). **Without `--key` it is
  refused**: an unsigned Statement does not say whose it is. The native
  receipt is not withdrawn — both are emitted, and which is which is said.
- **`matrixai attest --sigstore`.** It does **not** fall back to HMAC in
  silence: it says which of the two things is missing — the library or the
  identity, two problems with two solutions — and writes no receipt at
  all rather than half a bundle. And its limit is written and tested:
  **Sigstore does not raise the assurance level**. A0–A4 describe what was
  *checked*, not how strong the signature is; if signing better raised the
  level, A2 would stop meaning «there is reproducible evidence».
- **`matrixai report <package> --tripod [--locale es|en]`** — the TRIPOD+AI
  checklist a clinical-prediction journal asks for, filled from what
  `reproduce.json` already captures. It **invents no box**: it enumerates
  the ones it cannot fill (absent data, calibration, fairness, and what
  belongs to the author), and the SYNTHETIC-data warning goes at the very
  top.
- **`matrixai generate-dataset --recipe`** — the data recipe was reachable
  from `export-bundle` and from the product, but not from the CLI, so the
  first-contact path could not use it. `resolver_receta` was **extracted**
  to the core so the CLI and `playground.py` resolve it the same way. In
  the CLI an unreadable recipe **fails** (exit 1); in the Studio it warns.
- **`matrixai verify --locale {es,en}`** and `verify_package(..., locale=)`.
  The verifier used to write every reason in English only, so half the
  screen showed one language and half the other. What the core writes is
  translated in the core. What the *package* wrote about itself is
  **quoted, not translated**: putting words in its mouth would change
  bytes that its own `manifest_sha256` covers.
- **`effective_mode` in the synthetic dataset result.** `mode` is what was
  *requested*; a `coherent` run with no domain rules degrades to random,
  so a manifest that declared the request would describe a dataset that
  was never generated that way.
- **The generation mode travels in the run capture** (`mode`), so
  `generation.mode` stops being `null` in every package the product
  builds. It is **measured, not declared**: the mode that ships is the one
  that regenerates the CSV byte for byte.

### Changed
- **BREAKING — the DSSE envelope of a `.mxreceipt` now carries its payload
  in base64**, as the specification requires (it used to store the
  canonical JSON in clear text, which no third-party DSSE implementation
  accepts). **Receipts issued by earlier versions are rejected**, with a
  reason that says exactly that and tells you to re-issue them: reading
  them silently would make a non-conforming envelope pass as conforming.
  No receipts existed outside development, which is why this is the
  cheapest moment in the life of the format to fix it.
- An unreadable envelope is now level **A0** instead of A1. It used to
  fall through to «it carries a signature», which is the reassuring
  half-truth this project exists to remove.
- **Training now normalises inputs by the ranges the contract declares**,
  and the target range travels to the `inference_spec`. Both halves or
  neither: normalising only the inputs made a model that used to converge
  diverge instead. Measured driving the product — a clinical case went
  from accuracy 0.687 to 0.969, and the Kelvin example from *not
  converging* to a validation loss of 0.000000 and 273.150001 K for 0 °C.
- **A registry entry published without its `model.mxai` is marked, not
  hidden.** Publishing metrics alone has legitimate uses, so it is still
  accepted; but `RegistryEntry.es_ejecutable()` **deduces it from the
  `model_hash`** rather than adding a field — the `entry_hash` covers the
  identity fields, and adding one would break the chain of everything
  already published. The executor now says why it cannot run it, and what
  the entry *is* good for, instead of raising a bare `FileNotFoundError`.
- **A `.mxreceipt` reason catalogue in Spanish** (`export/reproduce_textos.py`).
  The phrase is not translated — the fact is **recomposed** from the same
  keys the manifest declares (`missing: ["recipe", …]`), because
  translating the string would be guessing. A key the catalogue cannot say
  is **enumerated, not silently dropped**, and if it can say none it
  returns `None` rather than an empty sentence.

### Fixed
- **An accuracy of 1.000000 that had compared nothing.** With
  `Label[0, 1]` and `ProbabilityMap[0, 1]` read as a *range* instead of two
  classes, the target loaded as a one-element vector, and `argmax` of one
  element is always 0 — so every row «matched». On 2,189 days of real
  rainfall observations the trainer printed a perfect 100 %. The evaluator
  now **refuses** mismatched shapes and one-element vectors, naming the
  likely cause and the fix (`name them: Label[no, si]`); the same model on
  the same data then gives a believable 0.7626. A bad accuracy still comes
  out bad — the fix cannot be that everything now passes.
- **A diverging training run came out as a Python traceback.**
  `matrixai train` on the `.mxtrain` that `matrixai generate-training`
  itself produces raised `OverflowError: Numerical result out of range`
  with a stack dump. It is the first thing that happens to anyone trying a
  regression from the CLI; it now says what diverged and what to change.
- **`matrixai run` did not print the prediction.** `_print_run_report`
  printed `project`, `actions` and `audit` and never touched `state`, so
  any pure classifier or regressor — anything without discrete actions —
  ran and showed no result. It prints it now, and a classifier names its
  classes.
- **A recipe condition that can never fire is now reported**, instead of
  being dropped in silence while the dataset came out 50 % accurate and
  the warning pointed elsewhere.
- **The weights file no longer has a different name depending on which
  trainer wrote it** (`params.best.json` vs `parameter_set.json`), which
  made a package unreadable to the path that did not write it.
- **The Kelvin example of the repository can be exported to a package**;
  `linear_regression` was not exportable, so the example that best teaches
  the flow was the one that could not complete it.
- **Three of the four `verify` stages were unreachable for any package the
  product built.** Without `generation.mode`, R1 could not compare; and
  since `training` is `INCOMPARABLE` when R1 did not pass, and R3 depends
  on `training`, only the manifest integrity check could ever run. The
  field existed in the core since the reproducible-package work; nothing
  ever filled it in.

---

## [1.6.0] — 2026-08-23

Pipelines that leave a receipt, packages that can prove they reproduce,
and a composite that finally passes its own verifier.

### Added
- **Verifiable pipelines** (`matrixai.pipelines`). A pipeline is a JSON
  policy — deterministic, fail-closed, and it *names the rule that
  fired* — plus an engine that resolves every component **by digest**,
  verifies it **before** running it, and never fills in what is missing.
  What did not start is said out loud instead of being silently skipped.
- **Decision receipts** (`.mxreceipt`, DSSE-signed). Assurance levels
  A0–A4 are **deduced from what was actually checked**, never declared,
  and the independent verifier reports what it could *not* verify as
  prominently as what it could.
- **`matrixai replay`** — re-run a receipt, with `--compare-reference`
  and `--receipt-out`. The comparison **names the stages that differ**
  («something changed» forces you to open both receipts side by side)
  and always says whether the two environments were the same one:
  matching inside the same environment proves repeatability, not
  reproducibility.
- **`matrixai receipt inspect | verify | compare`**. `inspect` verifies
  nothing and says so.
- **`matrixai verify`** and a reproducible export (`export/reproduce.py`,
  `export/verify.py`): the bundle now carries what it needs to reproduce
  itself, and `verify` runs four stages and reports each one. A run in a
  different environment returns `INCOMPARABLE` — it neither accuses nor
  approves for free.
- **A dependency and licence inventory** in the reproduction receipt. A
  licence that cannot be established is reported as `null` and counted
  apart: inventing one would be worse than not having it, because
  somebody would use it to decide.
- **Sandboxed reproduction** that **fails closed** when there is nothing
  to isolate with, and reproduces *inside* the isolation.
- **A Hugging Face Space template** (`export/space.py`) that shows the
  result of `verify` **before** inviting anyone to try the model.
- **Supervised continual learning** (`continual/supervision.py`,
  `continual/policy_view.py`): accepting a suggestion creates a
  **candidate**, never a deployment. Promoting stays a separate, human
  act.
- **A reference case** (`matrixai.reference.readmission`) that runs the
  whole thing end to end on real registry models.
- **Interface types when publishing** (`registry/interface_types.py`).
  `registry push` used to write `input_type: {}` / `output_type: {}`, so
  the composite type check **could never fail**: an impossible connection
  returned `Typecheck OK`. Types are now derived from the model itself,
  and what cannot be determined without ambiguity is published **without
  types** and shown as such — an invented type is worse than none.

### Fixed
- **`validate` and `lint` rejected every composite program.** The
  verifier did not count `IMPORT` aliases as declared nodes, so the two
  official composite examples shipped in this repository failed their own
  verifier. Same omission its own comment documented for `SEQUENCE`.
- **The two ends of the composite type check spoke different
  vocabularies.** The consumer's input was published normalised
  (`{kind: "VECTOR", size: n}`) and the producer's output raw — the
  annotation string as written in the `.mxai`. `"Vector[1]"` did not even
  match a `VECTOR`, so **no pair published by `registry push` could ever
  fit**. Outputs are now described in the same vocabulary; the rule
  itself is unchanged, and a `Score` still does not fit a `VECTOR[1]`.
- **Sizes are compared for vectors**, symmetrically to tensor shapes. A
  `VECTOR[2]` used to fit a `VECTOR[30]`: harmless while manifests were
  empty, a reassuring half-truth once they carry the size.
- **The provenance of a run outranks whatever the screen says.**

## [1.5.0] — 2026-08-18

Generated data that a model can actually learn from — and a prompt in
Spanish that builds a network again.

### Added
- **The data recipe.** Synthetic datasets had no relationship between
  inputs and target: measured, the correlation was 0.049 against a
  chance threshold of 0.098 for 200 rows. Whoever trained on them saw
  50 % accuracy and a collapse warning, and concluded the product does
  not work. A recipe is a small text a **person** writes —
  `1: debt > 60000 OR income < 20000`, `DEFAULT: 0`, `NOISE: 0.1`,
  `BALANCE: 1=0.3` — parsed and evaluated deterministically, with no LLM
  in the loop: that was the only way a freshly installed Studio with no
  API key could produce a learnable dataset at all.
- **Recipes for continuous targets** (`usage = 0.05*sqm + 1.2*people +
  5`, plus its noise). A `Scalar` target used to be filled with
  `uniform(lo, hi)` — pure noise, maximum correlation 0.054 — so a
  regression model could not learn anything.
- **The recipe travels with the model** (`data_recipe`) and ships inside
  the exported bundle as `data_recipe.txt`: the dataset that trained a
  model can be rebuilt by whoever receives it. Same recipe and seed give
  the same fingerprint, which is the sha256 of the CSV.
- **A requested class split** (`BALANCE: 1=0.3, 0=0.7`), and an explicit
  warning when the rule and the ranges cannot deliver it.

### Fixed
- **A prompt written in Spanish builds a NETWORK again.** The task-verb
  list held bare infinitives without accents (`predecir`, `clasificar`),
  matched as substrings against the raw text. Nobody writes that way in
  Spanish: people write `predice`, `clasifica`, `predicción`. Those
  prompts fell through to the generic agent — no `NETWORK`, no `LAYER`,
  invented fields — so the product looked broken **only in Spanish**,
  including for the examples the interface itself suggests.
- **`BALANCE` was ignored for binary targets**, which is the most common
  case: the labels of a `Probability` target are not in `target_labels`,
  and rows store `1.0`, which never matched `"1"`. A line written by a
  person was dropped without a word.
- **The degeneracy warning says whose rule it was.** When a recipe does
  not discriminate the core falls back to random labels — correct — but
  it blamed «domain rules proposed by the LLM» even when a person wrote
  them, and did not say the thresholds go in the data's own units.
- **Declared noise is the noise you get.** The alternative label could be
  the current one, so `NOISE: 0.1` produced 6.5 % of contradicting rows.

---

## [1.4.3] — 2026-08-14

Two numbers on the same screen that did not add up.

### Fixed
- **Accuracy and the confusion matrix now measure the same thing.** A
  training result reported accuracy from the **validation** split while
  the confusion matrix, macro-F1 and per-class figures came from a
  *later* evaluation that scores the **whole dataset**, training rows
  included. Summing the matrix therefore gave a different — and higher —
  percentage than the accuracy shown beside it. The code already knew:
  «confirmed they differ in practice; the source is left alone so as not
  to change a value the user already sees».

  The obvious fix would have been the wrong one. The inflated figure is
  the one computed over data the model was trained on; aligning accuracy
  to it would have turned a visible contradiction into a coherent lie.
  Validation wins instead. `DenseSupervisedTrainer` already computed that
  matrix and threw it away, keeping only the accuracy; it now travels in
  `TrainingRunResult.validation_metrics`, and the reported classification
  metrics come from there. The whole-dataset report remains as a
  **fallback** for runs stored before this change: it degrades, it does
  not break.

---

## [1.4.2] — 2026-08-13

A binary classifier now names the classes it actually used, and the
downloadable bundle stops shipping its own build artefacts.

### Added
- `binarize` dataset operation: turns a continuous measurement into a
  class with an **explicit** threshold, so a numeric column can become a
  yes/no target without leaving the declared pipeline. There is no
  default threshold — a default would silently decide what counts as
  "yes". An empty dataset stays empty and the column offset is
  inherited.

### Changed
- Thousands separators are written in the reader's language. The number
  was already measured correctly; only its presentation moved.

### Fixed
- **A binary classification without a `LABELS` block used to report
  `labels: []` and an empty `per_label`**, while still computing
  `macro_precision`/`macro_recall` from the very per-class values it had
  dropped — a response that contradicted itself. The evaluator now
  returns the labels it actually used (`negative`/`positive`), at both
  sites that declared the same rule (stdlib and torch). Consumers that
  fall back with `??` never recovered from this, because `??` does not
  fall back on an empty array: the confusion matrix, full of data, was
  never drawn.
- A constant target is rejected where it is detected, with the reason
  written out. A column with a single distinct value gives the network
  nothing to learn, and training used to run to completion reporting a
  near-zero loss for a model that can only repeat that constant.
- A corrupt CSV is rejected with its reason instead of entering
  silently.
- The downloadable model bundle no longer ships the `__pycache__` left
  by its own packaging smoke test — bytecode carrying the build
  machine's interpreter version, present in the zip but absent from the
  manifest. The large-weights streaming path already filtered it and the
  normal path did not, so the same model shipped different files
  depending on its size.

### Notes
- `_propose_margin` is deliberately unchanged: clamping the proposed
  range margin at zero breaks Kelvin training ("Numerical result out of
  range"). The reason is written next to the code so the next reader
  does not "fix" it again.

---

## [1.4.1] — 2026-08-12

Data a template generates must have something to learn; a prompt must
declare its ranges; and a training job belongs to whoever launched it.

### Added

- **Derived columns in the synthetic provider.** It could only generate
  INDEPENDENT columns, so a template's target was noise — there was
  nothing to learn. Three new column types: `linear` (a value derived
  from another column), `threshold` (a label by bands) and `seasonal` (a
  series with memory). Measured by training: tabular classification
  0.438 → 0.950; next-day consumption R² −0.027 → 0.689; synthetic
  series R² −0.078 → 0.916.

### Fixed

- **Field ranges on the PROMPT route.** Only the CSV route declared
  `field_ranges`, so the core did not normalise —the family of contract
  61— and the test screen had no scale to draw. They are now completed
  with the same mechanism already used to invent the data, without
  touching what the prompt does declare and without inventing a domain
  for generic names like `feature_1`.
- **A training job belongs to whoever launched it.** The job registry is
  global and the routes only looked at the `job_id`, so on a shared
  deployment anyone could read or cancel someone else's training by
  guessing an identifier. The owner is an opaque string, and `None`
  still means «nobody's» — which is what the downloadable Studio runs,
  where the machine belongs to whoever uses it.

---

## [1.4.0] — 2026-08-11

The task a model solves is decided by the QUESTION, not by a verb; an
identifier is not a feature; and what the core writes, the core
translates.

### Added

- **Task detection from the question** (contract 70). «Predict which
  customers will churn» is a classification even though «predict» reads
  like a regression verb. The inferred task now reaches the pipeline and
  is stated in the trace, and the assumption is only flagged when it was
  a DEFAULT — flagging a decision the prompt made explicitly taught
  people to ignore the warnings.
- **Identifiers are not features** (contract 71, first half). A column
  that looks like an id gets a warning explaining the consequence: the
  model memorises which row is which instead of learning from its
  traits, so it scores well on data it has seen and is useless for
  anyone new.
- **GPU path validated without a GPU**: prompt-built models train
  through both backends, with the effort declared in STEPS — the unit
  that is comparable between machines, unlike epochs (the same «50
  epochs» is 125,000 steps on CPU and 62 on GPU).

### Fixed

- **A multiclass with fewer than two classes is not a multiclass.** One
  in six generations produced a one-unit softmax that the core's own
  verifier rejected. Now the sentence decides: a yes/no question builds
  a binary model, and example classes are used only when nothing else is
  available — saying they are examples, not inventing yours.
- **JSON has no NaN, and we were writing it.** A NaN or an infinity in a
  metric produced a body that no strict parser accepts. They now travel
  as `null` — never as 0: an absent value is not a zero.
- **What the core writes is translated in the core.** Four class
  warnings were hard-coded in Spanish, so an English UI showed a screen
  in two languages. Translating them in the interface would have meant
  two versions of what the product says about its own decision.

---

## [1.3.1] — 2026-07-25

Patch release: normalization coherence across the REST endpoints. The
synchronous training route and the parameter-run route did not share the
normalization boundary the async route has used since 1.2.0, so a REST client
that trained on a **domain-scale** CSV (e.g. pressure ~1000 next to a signal
~1) trained in a different space than the one it later ran inference in. The
Studio product path (`/api/train-start`) was never affected.

### Fixed

- **`/api/train` normalizes like `/api/train-start`** — the synchronous
  training endpoint now accepts `field_ranges` and `target_range` and applies
  them at the same single boundary (CSV level, before dispatch, so all three
  trainer paths — dense, composite, transformer — only ever see `[0, 1]`).
  Without them a domain-scale CSV reached the trainer raw and the network
  collapsed to the majority class (measured on a real 2189-row dataset:
  accuracy 0.646 with a fully degenerate confusion matrix, versus 0.765 /
  macro-F1 0.720 and both classes predicted through the normalized path).
  Boolean, one-hot and embedding-index columns carry no range and are left
  untouched.
- **`target_range` reaches the network trainers on the synchronous route** —
  it was accepted but never threaded through, so regression MAE/RMSE came back
  in normalized space and `target_range` was not echoed. Confirmed on a
  Celsius→Kelvin dataset: R² −0.0001 / MAE 25.0 → R² 0.99999 / MAE 0.037 K.
- **`/api/run-with-params` train/serve coherence** — the endpoint ran the input
  raw. Passing `field_ranges` now applies the same `(v − min) / (max − min)`
  clamped to `[0, 1]` used at training time, so a domain-scale input predicts
  what the model actually learned (without ranges the input is still taken as
  already-normalized slider space, as the Studio sends it). Sending domain
  values with no ranges could flip the predicted class.
- **CSV validation failures now carry a readable message** —
  `_validate_training_csv` returned the detail only in the `errors` list and
  left `error` empty, so callers surfaced a mute "load failed" with no cause.
  It now summarizes the first error plus the remaining count, keeping the full
  list.

### Documentation

- `docs/{en,es}/api/REST_API.md`: input scale (`field_ranges`) and output scale
  (`target_range`) documented for `/api/train` and `/api/train-start`, plus the
  input-space contract of `/api/run-with-params` with a domain-scale example.

---

## [1.3.0] — 2026-07-21

Feature release: transformer blocks for text classification, model
generation directly from a real dataset (no prompt required), pluggable
external data providers, and two correctness fixes for regression training
(target-scale normalization and a torch/GPU-specific weight-initialization
bug) discovered and closed against real datasets.

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
- **External data providers** (`matrixai.training.data_provider`) — a
  pluggable registry for pulling real datasets from third-party APIs
  instead of a manual CSV upload, with license acceptance tracking and
  SSRF-hardened fetching (fixed host allowlist, redirect validation,
  DNS-rebinding protection). Ships an Open-Meteo provider (historical
  weather / marine data) as the reference implementation.

### Fixed

- **Regression targets in any scale now converge** — the regression target
  was trained on raw values while features were normalized to `[0, 1]`;
  above roughly `O(100)` this caused gradient explosion and a dead-ReLU
  collapse to the output bias (constant prediction). The target is now
  normalized with the same range mechanism as the features, and MAE/RMSE
  are rescaled back to the original units for reporting. R² is unaffected
  (scale-invariant). Confirmed on a Celsius→Kelvin dataset: R² −0.0001 → 0.999993.
- **Regression on the torch/GPU backend now matches the CPU backend** — a
  dense network built without pre-materialized weights (the common case)
  kept PyTorch's default `nn.Linear` initialization (`kaiming_uniform_` with
  `a=√5`, tuned for LeakyReLU, plus a non-zero random bias), which collapsed
  regression with few input features on GPU (R² as low as −35) while the
  identical model trained fine on the stdlib/CPU backend. Both backends now
  initialize with the same scheme (`he_normal`/`xavier_normal` + zero bias).
  Reproducible without a GPU (torch on CPU already showed the collapse).

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
