# MatrixAI CLI Reference

MatrixAI is invoked as `python -m matrixai <command> [options]` or `matrixai <command> [options]` after installation.

## Common flags

Most commands accept these flags:

| Flag | Description |
|------|-------------|
| `--json` | Print output as machine-readable JSON instead of human-readable text |
| `--output, -o <path>` | Write output to a file instead of stdout |

Registry commands default to `matrixai_registry/` in the current directory. Override with `--registry-path`.

---

## Command groups

- [Project](#project)
- [Model development](#model-development)
- [Analysis and diagnostics](#analysis-and-diagnostics)
- [Compilation and backends](#compilation-and-backends)
- [Parameters](#parameters)
- [Training](#training)
- [Serving](#serving)
- [Export and packaging](#export-and-packaging)
- [Actions](#actions)
- [Registry](#registry)
- [Key management](#key-management)
- [Continual learning](#continual-learning)
- [Refinement](#refinement)

---

## Project

### matrixai init

Create a new MatrixAI project from a template.

```
matrixai init <project_name> [--template <name>] [--output-dir <dir>] [--list-templates]
```

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `project_name` | — | Name of the project to create |
| `--template` | `classification` | Starter template to use |
| `--output-dir` | `.` | Directory where the project folder is created |
| `--list-templates` | — | Print available templates and exit |

Creates a project folder with a starter `.mxai`, `.mxtrain`, sample CSV and quickstart instructions.

---

## Model development

### matrixai prompt

Generate a `.mxai` program from a natural-language prompt.

```
matrixai prompt <prompt...> [-o <file>] [--semantic] [--json]
```

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `prompt` | — | Description of the model. Use `-` to read from stdin |
| `-o, --output` | stdout | Write generated `.mxai` to this file |
| `--semantic` | — | Also print the intermediate semantic spec |
| `--json` | — | Print full synthesis result as JSON |

---

### matrixai propose

Generate candidate semantic proposals via LLM and supervise them interactively.

```
matrixai propose <prompt...> [--max-candidates <n>] [--provider <name>] [--json] [--include-compiled]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-candidates` | — | Maximum candidates to present for supervision |
| `--provider` | `deterministic` | Proposal backend: `deterministic` or `chat-completions-compatible` |
| `--json` | — | Print decision as JSON |
| `--include-compiled` | — | Include compiled Python source in JSON output |

---

### matrixai supervise-prompt

Supervise a prompt-generated or proposed semantic artifact.

```
matrixai supervise-prompt <prompt...> [--proposal <file>] [--json] [--include-compiled]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--proposal` | — | `.semantic` proposal file to supervise instead of synthesising from the prompt |
| `--json` | — | Print report as JSON |
| `--include-compiled` | — | Include compiled Python source in JSON output |

---

### matrixai architect

Generate a `.mxai` file from a semantic spec file.

```
matrixai architect <file> [-o <output>] [--json]
```

---

### matrixai validate

Validate the structure and semantics of a `.mxai` file.

```
matrixai validate <file>
```

Exits with code 0 on success, non-zero on error.

---

### matrixai validate-plan

Validate a semantic spec before generating `.mxai` from it.

```
matrixai validate-plan <file>
```

---

### matrixai lint

Lint `.semantic` or `.mxai` files and report diagnostics.

```
matrixai lint <file> [--json] [--strict]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | — | Print diagnostics as JSON |
| `--strict` | — | Exit non-zero if any warnings are present |

---

### matrixai format

Print or write canonical formatting for `.semantic` or `.mxai` files.

```
matrixai format <file> [--check | --write]
```

| Flag | Description |
|------|-------------|
| `--check` | Exit non-zero if formatting would change the file (CI-friendly) |
| `--write` | Format the file in place |

Without flags, prints formatted output to stdout.

---

### matrixai typecheck

Infer and validate MatrixAI types for `.mx` or `.mxai` files.

```
matrixai typecheck <file> [--json] [--registry-path <dir>] [--allow-mutable-imports]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--registry-path` | `matrixai_registry` | Registry path for resolving `IMPORT` declarations |
| `--allow-mutable-imports` | — | Allow `@latest` and other mutable tags in `IMPORT` |
| `--json` | — | Print type report as JSON |

---

### matrixai parse

Parse a `.mxai` file and print its JSON intermediate representation.

```
matrixai parse <file>
```

---

## Analysis and diagnostics

### matrixai graph

Render the computation graph of a `.semantic` or `.mxai` file.

```
matrixai graph <file> [--format <fmt>] [-o <file>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `mermaid` | Output format: `mermaid`, `dot`, or `json` |
| `-o, --output` | stdout | Write graph to file |

---

### matrixai diagnose

Compare interpreted runtime and compiled Python output for the same input.

```
matrixai diagnose <file> --input <json_file> [--json] [--tolerance <float>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | JSON input file |
| `--tolerance` | `1e-9` | Numeric comparison tolerance |
| `--json` | — | Print diagnostic report as JSON |

---

### matrixai optimize

Analyse a `.mxai` file and suggest optimizations.

```
matrixai optimize <file> [--json]
```

---

### matrixai permissions

Review the sandbox permissions required by the actions in a `.mxai` file.

```
matrixai permissions <file> [--json]
```

---

### matrixai mathematize

Translate discrete if/else rules into continuous MatrixAI expressions.

```
matrixai mathematize <file> [--json]
```

`<file>` is a text file with one rule per line, or `-` to read from stdin.

---

## Compilation and backends

### matrixai compile

Compile a `.mxai` file to an executable backend.

```
matrixai compile <file> [--target <backend>] [-o <file>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `python` | Compilation target: `python` or `differentiable-python` |
| `-o, --output` | stdout | Write compiled output to file |

---

### matrixai eval

Evaluate a `.mx` mathematical expression file.

```
matrixai eval <file> [--input <json>] [--call <fn>] [--json] [--trace] [--graph]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | — | JSON input data: file path or inline JSON string |
| `--call` | all | Call only this specific function |
| `--trace` | — | Include evaluation trace in output |
| `--graph` | — | Print computation graph |
| `--json` | — | Print output as JSON |

---

### matrixai backend-report

Report portability of a `.mxai` file to a differentiable backend.

```
matrixai backend-report <file> [--target <backend>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `differentiable_python` | Backend: `differentiable_python` or `torch` |
| `--json` | — | Print report as JSON |

---

### matrixai backend-run

Run a `.mxai` file through a differentiable backend.

```
matrixai backend-run <file> --input <json_file> [--target <backend>] [--parameters <json>] [--device <dev>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | JSON input file |
| `--target` | `differentiable-python` | Backend: `differentiable-python` or `torch` |
| `--parameters` | — | ParameterSet JSON file, or `initial` to use generated defaults |
| `--device` | — | Compute device: `cpu`, `cuda`, `mps` (cuda/mps require `--target torch`) |
| `--json` | — | Print result as JSON |

---

### matrixai backend-parameters

Inspect or validate differentiable backend parameters for a `.mxai` file.

```
matrixai backend-parameters <file> [--target <backend>] [--validate <json>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `differentiable-python` | Backend target |
| `--validate` | — | ParameterSet JSON file to validate against the model |
| `--json` | — | Print parameter report as JSON |

---

## Parameters

### matrixai init-parameters

Create a versioned ParameterSet with default values from a `.mxai` file.

```
matrixai init-parameters <file> [-o <file>] [--parameter-set-id <id>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | stdout | Write ParameterSet JSON to file |
| `--parameter-set-id` | auto | Identifier for the generated ParameterSet |
| `--json` | — | Print ParameterSet as JSON |

---

### matrixai validate-parameters

Validate a ParameterSet JSON file against a `.mxai` program.

```
matrixai validate-parameters <file> --params <json_file> [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--params` | required | ParameterSet JSON file to validate |
| `--json` | — | Print validation report as JSON |

---

## Training

### matrixai validate-training

Validate a `.mxtrain` supervised training spec.

```
matrixai validate-training <file> [--json]
```

---

### matrixai generate-training

Generate a `.mxtrain` spec and CSV dataset template from a `.mxai` file and a prompt.

```
matrixai generate-training <mxai_file> <prompt...> [-o <file>] [--dataset-output <file>] [--epochs <n>] [--learning-rate <lr>] [--batch-size <n>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | stdout | Write `.mxtrain` to file |
| `--dataset-output` | — | Write CSV template to file |
| `--dataset-source` | — | Dataset source path to embed in the spec |
| `--dataset-name` | — | Override `DATASET` block name |
| `--target-name` | — | Override `TARGET` column name |
| `--labels` | — | Comma-separated label list |
| `--epochs` | — | Override `RUN EPOCHS` |
| `--learning-rate` | — | Override SGD learning rate |
| `--batch-size` | — | Override batch size |
| `--json` | — | Print generation result as JSON |

---

### matrixai generate-supervised

Generate `.mxai`, `.mxtrain` and a CSV template from a single prompt.

```
matrixai generate-supervised <prompt...> -o <output_dir> [--stem <name>] [--epochs <n>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir` | required | Directory for generated artifacts |
| `--stem` | auto | Override filename stem for generated files |
| `--dataset-name` | — | Override `DATASET` block name |
| `--target-name` | — | Override `TARGET` column name |
| `--labels` | — | Comma-separated label list |
| `--epochs` | — | Override `RUN EPOCHS` |
| `--json` | — | Print generation result as JSON |

---

### matrixai generate-dataset

Generate a reproducible synthetic dataset from a `.mxai` + `.mxtrain` spec.

```
matrixai generate-dataset <mxai_file> --training <mxtrain_file> [-o <output_dir>] [--rows <n>] [--seed <n>] [--mode <mode>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--training` | required | `.mxtrain` spec file |
| `--rows` | `200` | Total rows (range: 2–50,000) |
| `--seed` | `42` | Random seed for reproducibility |
| `--mode` | `random` | `random` or `coherent` (coherent is semantics-consistent) |
| `-o, --output-dir` | `.` | Output directory for CSVs and manifest |
| `--stem` | auto | Filename stem |
| `--json` | — | Print generation result as JSON |

---

### matrixai train

Train a supervised model and write MatrixAI training artifacts.

```
matrixai train <mxai_file> --training <mxtrain_file> -o <output_dir> [--backend <name>] [--device <dev>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--training` | required | `.mxtrain` spec file |
| `-o, --output` | required | Output run directory |
| `--backend` | from spec | `stdlib` or `torch` |
| `--device` | from spec | `cpu`, `cuda`, or `mps` |
| `--json` | — | Print training result as JSON |

---

### matrixai train-supervised

Full pipeline: generate `.mxai` + `.mxtrain` from a prompt, train, and evaluate.

```
matrixai train-supervised <prompt...> -o <output_dir> [--train-data <csv>] [--eval-data <csv>] [--dataset-manifest <json>] [--epochs <n>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir` | required | Output directory |
| `--train-data` | — | Training CSV file |
| `--eval-data` | — | Evaluation CSV file |
| `--dataset-manifest` | — | Versioned dataset manifest JSON |
| `--dataset-split` | — | Named split/fold in the manifest |
| `--stem` | auto | Artifact filename stem |
| `--run-name` | `run` | Run artifact directory name |
| `--epochs` | — | Override `RUN EPOCHS` |
| `--json` | — | Print end-to-end result as JSON |

---

### matrixai evaluate

Evaluate a trained ParameterSet on a supervised CSV dataset.

```
matrixai evaluate <mxai_file> --params <json_file> --training <mxtrain_file> [-o <report>] [--data <csv>] [--backend <name>] [--device <dev>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--params` | required | ParameterSet JSON (trained weights) |
| `--training` | required | `.mxtrain` spec file |
| `--data` | — | Optional CSV dataset override |
| `-o, --output` | — | Write `evaluation_report.json` to this path |
| `--backend` | — | `stdlib` or `torch` |
| `--device` | — | `cpu`, `cuda`, or `mps` |
| `--json` | — | Print evaluation result as JSON |

---

## Serving

### matrixai run

Run a `.mxai` file once with JSON input and print the result.

```
matrixai run <file> --input <json_file> [--params <json_file>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | JSON input file |
| `--params` | — | ParameterSet JSON file |
| `--json` | — | Print raw JSON result |

---

### matrixai serve

Serve a trained model and/or a model registry over HTTP (production server).

```
matrixai serve [<mxai_file>] [--params <json>] [--port <n>] [--host <addr>]
               [--api-key <key>] [--api-key-read <key>] [--registry <path>]
               [--contract <mxact>] [--allow-real-actions]
               [--continual-policy <policy>] [--reference-accuracy <float>]
               [--rate-limit <n>] [--cors-origin <origin>] [--backend <name>]
```

`<mxai_file>` is optional when `--registry` is provided.

| Flag | Default | Description |
|------|---------|-------------|
| `--params` | — | ParameterSet JSON file (trained weights) |
| `--port` | `8000` | HTTP port |
| `--host` | `127.0.0.1` | Bind address |
| `--backend` | `stdlib` | Execution backend: `stdlib` or `torch` |
| `--api-key` | auto-generated | Write API key — full access (env: `MATRIXAI_API_KEY`) |
| `--api-key-read` | — | Read-only API key — GET and predict endpoints only (env: `MATRIXAI_API_KEY_READ`) |
| `--registry` | — | Path to a registry directory — enables `/api/v1/registry/*` endpoints |
| `--contract` | — | `.mxact` file to enable `POST /api/v1/execute-action` |
| `--allow-real-actions` | off | Enable real action execution (required alongside `--contract`) |
| `--signing-key` | — | Hex HMAC key for ActionTrace signing (env: `MATRIXAI_ACTION_SIGNING_KEY`) |
| `--continual-policy` | — | `.mxcontinual` policy file to enable drift monitoring and `POST /api/v1/feedback` |
| `--reference-accuracy` | from params | Reference accuracy baseline for drift detection |
| `--rate-limit` | `60` | Max requests/minute per IP. `0` disables. (env: `MATRIXAI_RATE_LIMIT`) |
| `--cors-origin` | `*` | Allowed CORS origin. Repeatable. (env: `MATRIXAI_CORS_ORIGINS`) |

---

### matrixai playground / matrixai studio

Start the local model development environment in the browser.

```
matrixai playground [--host <addr>] [--port <n>] [--open]
matrixai studio    [--host <addr>] [--port <n>] [--open]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | HTTP port |
| `--open` | — | Open the URL in the default browser automatically |

`studio` and `playground` are aliases for the same command.

---

### matrixai pack

Package a `.mxai` model as a deployable artifact bundle, optionally with Docker support.

```
matrixai pack <mxai_file> [--params <json>] [--contract <mxact>] [--outdir <dir>] [--docker]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--params` | — | ParameterSet JSON file |
| `--contract` | — | `.mxact` contract file (included in the Docker image) |
| `--outdir` | `dist` | Output directory |
| `--docker` | — | Generate `Dockerfile`, `docker-compose.yml` and `.env.example` |

All generated files use UTF-8 encoding.

---

## Export and packaging

### matrixai export-onnx

Export a trained model to ONNX format.

```
matrixai export-onnx <mxai_file> --params <json> -o <output.onnx> [--validate] [--manifest <path>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--params` | required | ParameterSet JSON (trained weights) |
| `-o, --output` | required | Output `.onnx` file path |
| `--validate` | — | Run equivalence check against `onnxruntime` |
| `--manifest` | — | Write `export_manifest.json` here (requires `--validate`) |
| `--json` | — | Print export result as JSON |

---

### matrixai export-bundle

Create a self-contained edge bundle: `model.onnx` + manifests + README.

```
matrixai export-bundle <mxai_file> --params <json> --outdir <dir> \
  [--inference-metadata <json>] [--no-validate] [--force] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--params` | required | ParameterSet JSON |
| `--outdir` | required | Output directory |
| `--inference-metadata` | — | JSON sidecar with normalization metadata; makes the bundle self-usable (adds `predict.py`, `inference_spec.json`, `requirements.txt`, `example_input.json`, `expected_output.json`) |
| `--no-validate` | — | Skip equivalence check (not recommended for production) |
| `--force` | — | Overwrite existing bundle directory |
| `--json` | — | Print bundle result as JSON |

**Self-usable bundle.** With `--inference-metadata` the bundle ships a standalone
`predict.py` (only `numpy` + `onnxruntime`) that takes **raw** values and returns a
labelled prediction — normalization and category encoding are applied for you. Labels
also flow automatically from the model's `ProbabilityMap[...]`. The command prints
`Self-usable: yes/no` (and `inference_spec_skipped_reason` with `--json`).

The sidecar is strictly validated — a malformed key aborts with `Bundle error`
instead of silently producing a bundle that normalizes wrong:

```json
{
  "field_ranges":     {"age": [0, 120], "bmi": [10, 70]},
  "field_categories": {"color": ["red", "green", "blue"]},
  "field_types":      {"age": "integer", "active": "boolean"},
  "labels":           ["LOW", "HIGH"],
  "example_input":    {"age": 60, "bmi": 40, "color": "red"}
}
```

`field_ranges` values must be `[min, max]` finite numbers with `min < max`;
`field_categories` values must be non-empty lists of strings; `field_types` must be
one of `number` / `integer` / `boolean`.

**Using your downloaded model:**

```bash
cd my_bundle
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python predict.py --input example_input.json   # should reproduce expected_output.json
```

---

### matrixai export-wasm

Create an ONNX Runtime Web (WebAssembly) deployment bundle.

```
matrixai export-wasm <mxai_file> --params <json> --outdir <dir> [--no-validate] [--force] [--json]
```

Same flags as `export-bundle`. Output includes `model.onnx`, `predict.js` and WASM manifests for browser deployment.

---

## Actions

Actions require a `.mxact` contract file and the `MATRIXAI_ALLOW_REAL_ACTIONS=true` environment variable (or the `--allow-real-actions` flag) for real execution.

### matrixai validate-actions

Validate a `.mxact` contract against a `.mxai` program.

```
matrixai validate-actions <contract.mxact> <program.mxai> [--json]
```

---

### matrixai dry-run-action

Simulate a contract action without side effects and print the `DryRunReport`.

```
matrixai dry-run-action <contract.mxact> <program.mxai> --contract-name <name> [--input <json>] [--model-hash <hash>] [--param-set <id>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--contract-name` | required | Name of the `ACTION_CONTRACT` to simulate |
| `--input` | — | JSON object with `input_data`. Use `-` for stdin |
| `--model-hash` | `cli` | Model hash identifier |
| `--param-set` | `default` | Parameter set identifier |
| `--json` | — | Print report as JSON |

---

### matrixai execute-action

Execute a contract action with full side effects.

```
matrixai execute-action <contract.mxact> <program.mxai> --contract-name <name> --allow-real-actions [--input <json>] [--signing-key <hex>] [--json]
```

`--allow-real-actions` is required. Without it the command exits with an error.

| Flag | Default | Description |
|------|---------|-------------|
| `--contract-name` | required | Name of the `ACTION_CONTRACT` to execute |
| `--allow-real-actions` | off | Enables actual execution |
| `--input` | — | JSON object with `input_data`. Use `-` for stdin |
| `--model-hash` | — | Model hash identifier |
| `--param-set` | `default` | Parameter set identifier |
| `--signing-key` | — | Hex HMAC key (env: `MATRIXAI_ACTION_SIGNING_KEY`) |
| `--json` | — | Print result as JSON |

---

### matrixai audit-action

Verify the HMAC signature of an `ActionTrace` JSON file.

```
matrixai audit-action <trace_file> [--signing-key <hex>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--signing-key` | — | Hex HMAC key (env: `MATRIXAI_ACTION_SIGNING_KEY`) |
| `--json` | — | Print result as JSON |

---

## Registry

Registry entries are referenced as `name@version` (e.g. `credit-scoring@v1.0`).

### matrixai registry push

Register a trained model run into the registry.

```
matrixai registry push <run_dir> --name <name> --version <version> [--registry-path <dir>]
```

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `run_dir` | — | Path to training run directory |
| `--name` | required | Model name (lowercase) |
| `--version` | required | Version tag (e.g. `v1.0`) |
| `--registry-path` | `matrixai_registry` | Registry directory |

---

### matrixai registry pull

Copy a registry entry from one registry to another.

```
matrixai registry pull <name@version> --from <source_dir> --to <dest_dir>
```

---

### matrixai registry list

List all entries in the registry.

```
matrixai registry list [--name <name>] [--registry-path <dir>] [--json]
```

---

### matrixai registry show

Print the manifest for a specific registry entry.

```
matrixai registry show <name@version> [--registry-path <dir>] [--json]
```

---

### matrixai registry tag

Create or move a tag alias (e.g. `latest`, `prod`) to a specific version.

```
matrixai registry tag <name@version> <tag_name> [--registry-path <dir>]
```

---

### matrixai registry verify

Verify the integrity of a registry entry (checksums and completeness).

```
matrixai registry verify <name@version> [--registry-path <dir>]
```

Exits with code 0 if the entry is intact, non-zero otherwise.

---

### matrixai registry diff

Compare the manifests of two versions of a model.

```
matrixai registry diff <name@version_a> <name@version_b> [--registry-path <dir>]
```

---

## Key management

Signing keys are stored in `<registry-path>/.matrixai_key_history.json` by default.

### matrixai keys rotate

Retire the current signing key and record it in the key history.

```
matrixai keys rotate --purpose <purpose> [--key <value>] [--history-path <file>] [--registry-path <dir>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--purpose` | required | Which key to rotate: `action` or `registry` |
| `--key` | from env | Key value to retire |
| `--history-path` | `<registry>/.matrixai_key_history.json` | Key history file |
| `--registry-path` | `matrixai_registry` | Registry directory |

---

### matrixai keys list

List all recorded signing keys (active and retired) with status and fingerprints.

```
matrixai keys list [--history-path <file>] [--registry-path <dir>] [--json]
```

---

## Continual learning

Continual commands operate on a `.mxcontinual` policy file. They manage drift detection, version promotion and rollback in production.

### matrixai continual init

Validate and display a summary of a `.mxcontinual` policy.

```
matrixai continual init <policy.mxcontinual> [--json]
```

---

### matrixai continual ingest

Record a ground truth label for a production `ActionTrace` (used by the drift monitor).

```
matrixai continual ingest <policy.mxcontinual> --trace-id <id> --label <value> [--trace-file <json>] [--signing-key <hex>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--trace-id` | required | `ActionTrace` `report_id` to annotate |
| `--label` | required | Ground truth label or value |
| `--trace-file` | — | Path to `ActionTrace` JSON file |
| `--signing-key` | — | Hex HMAC key (env: `MATRIXAI_CONTINUAL_SIGNING_KEY`) |

---

### matrixai continual status

Show the current registry version, metrics and rollback history for a policy.

```
matrixai continual status <policy.mxcontinual> [--registry-dir <dir>] [--json]
```

---

### matrixai continual promote

Promote a candidate ParameterSet to production via the `ContinualVersioner`.

```
matrixai continual promote <policy.mxcontinual> --approval-report <json> --candidate-params <json> [--registry-dir <dir>] [--update-id <id>] [--human-approved] [--approved-by <identity>] [--signing-key <hex>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--approval-report` | required | `ApprovalGateReport` JSON file |
| `--candidate-params` | required | Candidate ParameterSet JSON file |
| `--registry-dir` | `matrixai_registry` | Registry directory |
| `--update-id` | auto | Continual update identifier |
| `--human-approved` | — | Record a human approval decision |
| `--approved-by` | `$USER` | Human approver identity |
| `--signing-key` | — | Hex HMAC key to verify `PendingApproval` token |

---

### matrixai continual rollback

Roll back to the previous registry version.

```
matrixai continual rollback <policy.mxcontinual> [--registry-dir <dir>] [--dry-run] [--signing-key <hex>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | — | Show what would be rolled back without executing |
| `--signing-key` | — | Hex HMAC key (env: `MATRIXAI_CONTINUAL_SIGNING_KEY`) |

---

### matrixai continual audit

Display audit configuration and optionally generate a drift-driven refinement hint.

```
matrixai continual audit <policy.mxcontinual> [--drift-report <json>] [--prompt <text>] [--drift-persistence-days <n>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--drift-report` | — | `DriftReport` JSON file for refinement hint |
| `--prompt` | — | Original prompt (required with `--drift-report`) |
| `--drift-persistence-days` | — | Days drift has been continuously observed (gates `REFINEMENT_DRIFT_PERSISTENCE_DAYS`) |

---

## Refinement

### matrixai refine

Propose a refined prompt based on an audit or evaluation result.

```
matrixai refine <prompt...> [--audit <json>] [--evaluation <json>] [--mxai <file>] [--hint <text>] [--iteration <n>] [--chain <json>] [--parent-hash <sha256>] [-o <file>] [--mxai-output <file>] [--chain-output <file>] [--accept] [--max-iterations <n>] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `prompt` | — | Original prompt text. Use `-` for stdin |
| `--audit` | — | Audit JSON file (activates `audit_driven` mode) |
| `--evaluation` | — | `evaluation_report.json` (activates `metric_driven` mode) |
| `--mxai` | — | Current `.mxai` file (adds model context to refinement) |
| `--hint` | — | Additional user hint. Repeatable |
| `--iteration` | `1` | Current iteration number |
| `--chain` | — | JSON file with prior `refinement_chain` (list of IDs) |
| `--parent-hash` | — | SHA-256 of the original root prompt |
| `-o, --output` | — | Write proposed prompt to file (requires `--accept`) |
| `--mxai-output` | — | Write generated `.mxai` to file (requires `--accept`) |
| `--chain-output` | — | Write updated `refinement_chain` JSON to file |
| `--accept` | — | Explicitly accept the proposal to enable writing output files |
| `--max-iterations` | default | Hard iteration limit. Exit code `2` if exceeded |
| `--json` | — | Print full `RefinementProposal` as JSON |

---

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `MATRIXAI_API_KEY` | `serve` | API key for HTTP authentication |
| `MATRIXAI_ACTION_SIGNING_KEY` | `serve`, `execute-action`, `audit-action` | Hex HMAC key for action trace signing |
| `MATRIXAI_CONTINUAL_SIGNING_KEY` | `continual ingest`, `continual rollback` | Hex HMAC key for continual learning events |
| `MATRIXAI_ALLOW_REAL_ACTIONS` | `serve` | Set to `true` to enable `/execute-action` |
| `MATRIXAI_RATE_LIMIT` | `serve` | Max requests/minute per IP (integer) |
| `MATRIXAI_CORS_ORIGINS` | `serve` | Comma-separated allowed CORS origins |

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (parse, validation, runtime) |
| `2` | Iteration limit exceeded (`refine`) |
