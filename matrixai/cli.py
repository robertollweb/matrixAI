# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from matrixai.agents import (
    ArchitectAgent,
    AuditorAgent,
    IterationLimitReached,
    LLMProposalAgent,
    MathematicalAgent,
    ChatCompletionsLLMProposalProvider,
    OptimizerAgent,
    PlannerVerifier,
    PromptAgent,
    PromptSupervisor,
    RefinementAgent,
    SafetyAgent,
    VerifierAgent,
)
from matrixai.compiler import (
    BackendContractAnalyzer,
    DifferentiablePythonCompiler,
    PythonBackendCompiler,
)
from matrixai.compiler.torch_forward import TorchForwardRunner
from matrixai.export import (
    EdgeBundleError,
    OnnxEquivalenceError,
    OnnxExportError,
    WasmExportError,
    create_edge_bundle,
    export_onnx,
    export_wasm,
    ort_available,
    validate_onnx_equivalence,
    write_export_manifest,
)
from matrixai.parameters import (
    build_initial_parameter_set,
    load_parameter_set,
    validate_parameter_set,
    validate_parameter_set_for_torch,
    write_parameter_set,
)
from matrixai.parser import parse_file
from matrixai.runtime import MatrixAIRuntime
from matrixai.training import (
    BackendSpec,
    GenericSupervisedEvaluator,
    MatrixAITrainingParseError,
    SupervisedEvaluator,
    SupervisedPromptGenerator,
    SupervisedPromptRunner,
    SupervisedTrainer,
    TorchSupervisedTrainer,
    TrainingPromptGenerator,
    TrainingVerifier,
    parse_training_file,
)
from matrixai.training.dense_trainer import DenseSupervisedEvaluator, DenseSupervisedTrainer


def main() -> int:
    parser = argparse.ArgumentParser(prog="matrixai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a .mxai file and print JSON IR")
    parse_parser.add_argument("file")

    prompt_parser = subparsers.add_parser(
        "prompt", help="Generate MatrixAI artifacts from a natural-language prompt"
    )
    prompt_parser.add_argument(
        "prompt",
        nargs="+",
        help="Natural-language prompt, or - to read the prompt from stdin",
    )
    prompt_parser.add_argument("--output", "-o")
    prompt_parser.add_argument("--semantic", action="store_true", help="Print semantic spec")
    prompt_parser.add_argument("--json", action="store_true", help="Print prompt synthesis JSON")

    propose_parser = subparsers.add_parser(
        "propose", help="Generate LLM-style semantic proposals and supervise them"
    )
    propose_parser.add_argument(
        "prompt",
        nargs="+",
        help="Natural-language prompt, or - to read the prompt from stdin",
    )
    propose_parser.add_argument(
        "--max-candidates",
        type=int,
        help="Maximum number of candidates to supervise",
    )
    propose_parser.add_argument(
        "--provider",
        choices=["deterministic", "chat-completions-compatible"],
        default="deterministic",
        help="Proposal provider to use",
    )
    propose_parser.add_argument("--json", action="store_true", help="Print proposal decision JSON")
    propose_parser.add_argument(
        "--include-compiled",
        action="store_true",
        help="Include compiled Python source in JSON output",
    )

    supervise_parser = subparsers.add_parser(
        "supervise-prompt", help="Supervise prompt-generated or proposed semantic artifacts"
    )
    supervise_parser.add_argument(
        "prompt",
        nargs="+",
        help="Natural-language prompt, or - to read the prompt from stdin",
    )
    supervise_parser.add_argument(
        "--proposal",
        help="Optional .semantic proposal to supervise instead of PromptAgent synthesis",
    )
    supervise_parser.add_argument("--json", action="store_true", help="Print report as JSON")
    supervise_parser.add_argument(
        "--include-compiled",
        action="store_true",
        help="Include compiled Python source in JSON output",
    )

    architect_parser = subparsers.add_parser(
        "architect", help="Generate a .mxai file from a semantic spec"
    )
    architect_parser.add_argument("file")
    architect_parser.add_argument("--output", "-o")
    architect_parser.add_argument("--json", action="store_true", help="Print semantic plan JSON")

    validate_plan_parser = subparsers.add_parser(
        "validate-plan", help="Validate a semantic spec before .mxai generation"
    )
    validate_plan_parser.add_argument("file")

    validate_parser = subparsers.add_parser("validate", help="Validate a .mxai file")
    validate_parser.add_argument("file")

    lint_parser = subparsers.add_parser(
        "lint", help="Lint .semantic or .mxai files with MatrixAI language diagnostics"
    )
    lint_parser.add_argument("file")
    lint_parser.add_argument("--json", action="store_true", help="Print diagnostics as JSON")
    lint_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when warnings are present",
    )

    format_parser = subparsers.add_parser(
        "format", help="Print or write canonical formatting for .semantic or .mxai files"
    )
    format_parser.add_argument("file")
    format_parser.add_argument("--check", action="store_true", help="Fail if formatting would change")
    format_parser.add_argument("--write", action="store_true", help="Write formatted output in place")

    graph_parser = subparsers.add_parser(
        "graph", help="Render a .semantic or .mxai computation graph"
    )
    graph_parser.add_argument("file")
    graph_parser.add_argument(
        "--format",
        choices=["mermaid", "dot", "json"],
        default="mermaid",
        help="Graph output format",
    )
    graph_parser.add_argument("--output", "-o", help="Write graph output to a file")

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Compare interpreted runtime and compiled Python output"
    )
    diagnose_parser.add_argument("file")
    diagnose_parser.add_argument("--input", required=True, help="JSON input file")
    diagnose_parser.add_argument("--json", action="store_true", help="Print diagnostic report as JSON")
    diagnose_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Numeric comparison tolerance",
    )

    typecheck_parser = subparsers.add_parser(
        "typecheck", help="Infer and validate MatrixAI types for .mx or .mxai files"
    )
    typecheck_parser.add_argument("file")
    typecheck_parser.add_argument("--json", action="store_true", help="Print type report as JSON")
    typecheck_parser.add_argument(
        "--registry-path", default="matrixai_registry", dest="registry_path",
        help="Path to local registry (used to resolve IMPORT declarations)"
    )
    typecheck_parser.add_argument(
        "--allow-mutable-imports", action="store_true", dest="allow_mutable_imports",
        help="Allow @latest and other mutable tags in IMPORT declarations"
    )

    permissions_parser = subparsers.add_parser(
        "permissions", help="Review sandbox permissions for MatrixAI actions"
    )
    permissions_parser.add_argument("file")
    permissions_parser.add_argument("--json", action="store_true", help="Print permission report as JSON")

    backend_parser = subparsers.add_parser(
        "backend-report", help="Report portability to a future differentiable backend subset"
    )
    backend_parser.add_argument("file")
    backend_parser.add_argument(
        "--target",
        default="differentiable_python",
        choices=["differentiable_python", "differentiable-python", "torch"],
        help="Backend contract target",
    )
    backend_parser.add_argument("--json", action="store_true", help="Print backend report as JSON")

    backend_run_parser = subparsers.add_parser(
        "backend-run", help="Run a .mxai file through a differentiable backend target"
    )
    backend_run_parser.add_argument("file")
    backend_run_parser.add_argument("--input", required=True, help="JSON input file")
    backend_run_parser.add_argument(
        "--target",
        default="differentiable-python",
        choices=["differentiable-python", "torch"],
        help="Backend execution target",
    )
    backend_run_parser.add_argument(
        "--parameters",
        help="JSON parameters file, or 'initial' to use the generated initial parameter manifest",
    )
    backend_run_parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device for --target torch (default: cpu)",
    )
    backend_run_parser.add_argument("--json", action="store_true", help="Print backend run result as JSON")

    backend_parameters_parser = subparsers.add_parser(
        "backend-parameters", help="Inspect or validate differentiable backend parameters"
    )
    backend_parameters_parser.add_argument("file")
    backend_parameters_parser.add_argument(
        "--target",
        default="differentiable-python",
        choices=["differentiable-python", "torch"],
        help="Backend execution target",
    )
    backend_parameters_parser.add_argument(
        "--validate",
        help="Optional JSON parameters file to validate against the generated manifest",
    )
    backend_parameters_parser.add_argument("--json", action="store_true", help="Print parameter report as JSON")

    init_parameters_parser = subparsers.add_parser(
        "init-parameters", help="Create a versioned MatrixAI ParameterSet from a .mxai file"
    )
    init_parameters_parser.add_argument("file")
    init_parameters_parser.add_argument("--output", "-o", help="Write ParameterSet JSON to this file")
    init_parameters_parser.add_argument(
        "--parameter-set-id",
        help="Optional identifier for the generated ParameterSet",
    )
    init_parameters_parser.add_argument("--json", action="store_true", help="Print ParameterSet as JSON")

    validate_parameters_parser = subparsers.add_parser(
        "validate-parameters", help="Validate a MatrixAI ParameterSet against a .mxai file"
    )
    validate_parameters_parser.add_argument("file")
    validate_parameters_parser.add_argument("--params", required=True, help="ParameterSet JSON file")
    validate_parameters_parser.add_argument("--json", action="store_true", help="Print validation report as JSON")

    validate_training_parser = subparsers.add_parser(
        "validate-training", help="Validate a MatrixAI .mxtrain supervised training spec"
    )
    validate_training_parser.add_argument("file")
    validate_training_parser.add_argument("--json", action="store_true", help="Print training report as JSON")

    generate_training_parser = subparsers.add_parser(
        "generate-training", help="Generate a controlled .mxtrain spec and CSV dataset template from a prompt"
    )
    generate_training_parser.add_argument("file", help=".mxai model file")
    generate_training_parser.add_argument(
        "prompt",
        nargs="+",
        help="Human training description, or - to read it from stdin",
    )
    generate_training_parser.add_argument("--output", "-o", help="Write generated .mxtrain to this file")
    generate_training_parser.add_argument("--dataset-output", help="Write generated CSV template to this file")
    generate_training_parser.add_argument(
        "--dataset-source",
        help="Dataset source path to embed in the .mxtrain; defaults to --dataset-output when provided",
    )
    generate_training_parser.add_argument("--dataset-name", help="Override DATASET block name")
    generate_training_parser.add_argument("--target-name", help="Override TARGET column name")
    generate_training_parser.add_argument("--labels", help="Comma-separated label list")
    generate_training_parser.add_argument("--epochs", type=int, help="Override RUN EPOCHS")
    generate_training_parser.add_argument("--learning-rate", type=float, help="Override SGD learning rate")
    generate_training_parser.add_argument("--batch-size", type=int, help="Override batch size")
    generate_training_parser.add_argument("--json", action="store_true", help="Print generation result as JSON")

    generate_supervised_parser = subparsers.add_parser(
        "generate-supervised",
        help="Generate supervised .mxai, .mxtrain and CSV template artifacts from a prompt",
    )
    generate_supervised_parser.add_argument(
        "prompt",
        nargs="+",
        help="Human supervised system description, or - to read it from stdin",
    )
    generate_supervised_parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where the generated .mxai, .mxtrain and CSV template are written",
    )
    generate_supervised_parser.add_argument("--stem", help="Override generated artifact filename stem")
    generate_supervised_parser.add_argument("--dataset-name", help="Override DATASET block name")
    generate_supervised_parser.add_argument("--target-name", help="Override TARGET column name")
    generate_supervised_parser.add_argument("--labels", help="Comma-separated label list")
    generate_supervised_parser.add_argument("--epochs", type=int, help="Override RUN EPOCHS")
    generate_supervised_parser.add_argument("--learning-rate", type=float, help="Override SGD learning rate")
    generate_supervised_parser.add_argument("--batch-size", type=int, help="Override batch size")
    generate_supervised_parser.add_argument("--json", action="store_true", help="Print generation result as JSON")

    train_supervised_parser = subparsers.add_parser(
        "train-supervised",
        help="Generate, train and evaluate a supervised package from a prompt and CSV datasets",
    )
    train_supervised_parser.add_argument(
        "prompt",
        nargs="+",
        help="Human supervised system description, or - to read it from stdin",
    )
    train_supervised_parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where the generated package and run artifacts are written",
    )
    train_supervised_parser.add_argument("--train-data", help="Compatible supervised training CSV")
    train_supervised_parser.add_argument("--eval-data", help="Compatible supervised evaluation CSV")
    train_supervised_parser.add_argument(
        "--dataset-manifest",
        help="Versioned dataset manifest JSON with train and evaluation CSV entries",
    )
    train_supervised_parser.add_argument(
        "--dataset-split",
        help="Named split/fold inside --dataset-manifest to use for train/evaluation partitions",
    )
    train_supervised_parser.add_argument("--stem", help="Override generated artifact filename stem")
    train_supervised_parser.add_argument("--run-name", default="run", help="Run artifact directory name")
    train_supervised_parser.add_argument("--dataset-name", help="Override DATASET block name")
    train_supervised_parser.add_argument("--target-name", help="Override TARGET column name")
    train_supervised_parser.add_argument("--labels", help="Comma-separated label list")
    train_supervised_parser.add_argument("--epochs", type=int, help="Override RUN EPOCHS")
    train_supervised_parser.add_argument("--learning-rate", type=float, help="Override SGD learning rate")
    train_supervised_parser.add_argument("--batch-size", type=int, help="Override batch size")
    train_supervised_parser.add_argument("--json", action="store_true", help="Print end-to-end result as JSON")

    generate_dataset_parser = subparsers.add_parser(
        "generate-dataset",
        help="Generate a reproducible synthetic dataset from a .mxai and .mxtrain",
    )
    generate_dataset_parser.add_argument("file", help=".mxai model file")
    generate_dataset_parser.add_argument("--training", required=True, help=".mxtrain training spec")
    generate_dataset_parser.add_argument("--rows", type=int, default=200, help="Total rows to generate (default: 200). Upper bound governed by the limits profile (MATRIXAI_LIMITS_PROFILE: equilibrado=50000, ilimitado=none; or MATRIXAI_MAX_ROWS)")
    generate_dataset_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    generate_dataset_parser.add_argument(
        "--mode",
        choices=["random", "coherent"],
        default="random",
        help="Sampling mode: random (uniform) or coherent (model-guided labels)",
    )
    generate_dataset_parser.add_argument(
        "--output-dir", "-o", default=".", help="Output directory for CSV files and manifest (default: current dir)"
    )
    generate_dataset_parser.add_argument("--stem", help="Filename stem for generated files (default: derived from project name)")
    generate_dataset_parser.add_argument("--json", action="store_true", help="Print generation result as JSON")

    train_parser = subparsers.add_parser(
        "train", help="Train the P4 supervised MVP and write MatrixAI training artifacts"
    )
    train_parser.add_argument("file", help=".mxai model file")
    train_parser.add_argument("--training", required=True, help=".mxtrain training spec")
    train_parser.add_argument("--output", "-o", required=True, help="Output run directory")
    train_parser.add_argument(
        "--backend",
        choices=["stdlib", "torch"],
        default=None,
        help="Training backend (overrides .mxtrain BACKEND TARGET; default: stdlib)",
    )
    train_parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device (overrides .mxtrain BACKEND DEVICE; default: cpu). Requires --backend torch for cuda/mps.",
    )
    train_parser.add_argument("--json", action="store_true", help="Print training result as JSON")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a MatrixAI ParameterSet on a supervised CSV dataset"
    )
    evaluate_parser.add_argument("file", help=".mxai model file")
    evaluate_parser.add_argument("--params", required=True, help="ParameterSet JSON file")
    evaluate_parser.add_argument("--training", required=True, help=".mxtrain training spec")
    evaluate_parser.add_argument("--data", help="Optional CSV dataset override")
    evaluate_parser.add_argument("--output", "-o", help="Optional evaluation_report.json output path")
    evaluate_parser.add_argument("--json", action="store_true", help="Print evaluation result as JSON")
    evaluate_parser.add_argument(
        "--backend",
        choices=["stdlib", "torch"],
        default=None,
        help="Backend to use for evaluation (overrides .mxtrain BACKEND TARGET; default: stdlib)",
    )
    evaluate_parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device (overrides .mxtrain BACKEND DEVICE; default: cpu). Requires --backend torch for cuda/mps.",
    )

    serve_parser = subparsers.add_parser("serve", help="Serve a .mxai model or a registry over HTTP")
    serve_parser.add_argument("file", nargs="?", default=None,
                              help=".mxai model file (optional when --registry is provided)")
    serve_parser.add_argument("--params", help="Optional MatrixAI ParameterSet JSON file")
    serve_parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    serve_parser.add_argument("--host", default="127.0.0.1", help="HTTP server host")
    serve_parser.add_argument("--backend", choices=["stdlib", "torch"], default="stdlib", help="Execution backend")
    serve_parser.add_argument("--api-key", help="Secret token for Bearer authentication (auto-generated if not provided)")
    serve_parser.add_argument("--contract", help="Path to a .mxact file to enable POST /execute-action")
    serve_parser.add_argument("--allow-real-actions", action="store_true", dest="allow_real_actions",
                              help="Enable real action execution via /execute-action endpoint")
    serve_parser.add_argument("--signing-key", dest="signing_key",
                              help="Hex signing key for ActionTrace HMAC (or MATRIXAI_ACTION_SIGNING_KEY)")
    serve_parser.add_argument("--continual-policy", dest="continual_policy", default=None,
                              help="Path to a .mxcontinual policy file to enable drift monitoring and POST /feedback")
    serve_parser.add_argument("--reference-accuracy", dest="reference_accuracy", type=float, default=None,
                              help="Reference accuracy for drift detection (default: read from --params metrics)")
    serve_parser.add_argument("--rate-limit", type=int, dest="rate_limit", default=None,
                              help="Max requests/minute per IP (default: 60; 0 disables; env: MATRIXAI_RATE_LIMIT)")
    serve_parser.add_argument("--cors-origin", action="append", dest="cors_origins", metavar="ORIGIN",
                              help="Allowed CORS origin (repeatable; default: *; env: MATRIXAI_CORS_ORIGINS)")
    serve_parser.add_argument("--registry", dest="registry", default=None, metavar="PATH",
                              help="Path to a MatrixAI registry directory — enables /api/v1/registry/* endpoints")
    serve_parser.add_argument("--api-key-read", dest="api_key_read", default=None, metavar="KEY",
                              help="Read-only API key for /api/v1/registry GET and predict endpoints (env: MATRIXAI_API_KEY_READ)")

    pack_parser = subparsers.add_parser("pack", help="Package a .mxai model into an artifact bundle or Docker container")
    pack_parser.add_argument("file", help=".mxai model file")
    pack_parser.add_argument("--params", help="Optional MatrixAI ParameterSet JSON file")
    pack_parser.add_argument("--contract", help="Optional .mxact contract file; included in Docker bundle and wired to serve --contract")
    pack_parser.add_argument("--outdir", default="dist", help="Output directory for the packaged artifacts")
    pack_parser.add_argument("--docker", action="store_true", help="Generate a Dockerfile and context")

    keys_parser = subparsers.add_parser("keys", help="Manage signing-key rotation (PR4-C4)")
    keys_sub = keys_parser.add_subparsers(dest="keys_command")

    keys_rotate_p = keys_sub.add_parser("rotate", help="Retire the current signing key and record it in key history")
    keys_rotate_p.add_argument(
        "--purpose", choices=["action", "registry"], required=True,
        help="Which key to rotate: 'action' (MATRIXAI_ACTION_SIGNING_KEY) or 'registry' (MATRIXAI_REGISTRY_SIGNING_KEY)",
    )
    keys_rotate_p.add_argument(
        "--key", dest="key_value",
        help="Key value to retire (default: read from env var for the chosen purpose)",
    )
    keys_rotate_p.add_argument(
        "--history-path",
        help="Path to key history JSON file (default: MATRIXAI_KEY_HISTORY_PATH or registry/.matrixai_key_history.json)",
    )
    keys_rotate_p.add_argument(
        "--registry-path", default="matrixai_registry",
        help="Registry directory used to locate the default history file (default: matrixai_registry)",
    )

    keys_list_p = keys_sub.add_parser("list", help="List all recorded signing keys and their status")
    keys_list_p.add_argument(
        "--history-path",
        help="Path to key history JSON file",
    )
    keys_list_p.add_argument(
        "--registry-path", default="matrixai_registry",
        help="Registry directory used to locate the default history file",
    )
    keys_list_p.add_argument("--json", action="store_true", help="Output as JSON")

    export_onnx_parser = subparsers.add_parser(
        "export-onnx", help="Export a trained MatrixAI model to ONNX format"
    )
    export_onnx_parser.add_argument("file", help=".mxai model file")
    export_onnx_parser.add_argument("--params", required=True, help="ParameterSet JSON file (trained weights)")
    export_onnx_parser.add_argument("--output", "-o", required=True, help="Output .onnx file path")
    export_onnx_parser.add_argument("--json", action="store_true", help="Print export result as JSON")
    export_onnx_parser.add_argument(
        "--validate", action="store_true",
        help="Run equivalence check (differentiable_python vs onnxruntime) after export",
    )
    export_onnx_parser.add_argument(
        "--manifest", metavar="PATH",
        help="Write export_manifest.json to this path (requires --validate)",
    )

    export_bundle_parser = subparsers.add_parser(
        "export-bundle", help="Create a self-contained edge bundle (model.onnx + manifests)"
    )
    export_bundle_parser.add_argument("file", help=".mxai model file")
    export_bundle_parser.add_argument("--params", required=True, help="ParameterSet JSON file (trained weights)")
    export_bundle_parser.add_argument("--outdir", required=True, help="Output directory for the bundle")
    export_bundle_parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip equivalence check (not recommended for production bundles)",
    )
    export_bundle_parser.add_argument("--force", action="store_true", help="Overwrite existing bundle directory")
    export_bundle_parser.add_argument(
        "--inference-metadata",
        help=(
            "JSON sidecar with normalization metadata so the bundle is self-usable "
            "(predict.py + inference_spec.json). Keys: field_ranges {col:[lo,hi]}, "
            "field_categories {col:[values]}, field_types {col:number|integer|boolean}, "
            "labels [..], example_input {..}. Labels also flow from the .mxai ProbabilityMap."
        ),
    )
    export_bundle_parser.add_argument("--json", action="store_true", help="Print bundle result as JSON")

    export_wasm_parser = subparsers.add_parser(
        "export-wasm", help="Create an ONNX Runtime Web (WASM) deployment bundle"
    )
    export_wasm_parser.add_argument("file", help=".mxai model file")
    export_wasm_parser.add_argument("--params", required=True, help="ParameterSet JSON file (trained weights)")
    export_wasm_parser.add_argument("--outdir", required=True, help="Output directory for the WASM bundle")
    export_wasm_parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip equivalence check (not recommended for production bundles)",
    )
    export_wasm_parser.add_argument("--force", action="store_true", help="Overwrite existing bundle directory")
    export_wasm_parser.add_argument("--json", action="store_true", help="Print bundle result as JSON")

    run_parser = subparsers.add_parser("run", help="Run a .mxai file with JSON input")
    run_parser.add_argument("file")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--params", help="Optional MatrixAI ParameterSet JSON file")
    run_parser.add_argument("--json", action="store_true", help="Print raw JSON result")

    optimize_parser = subparsers.add_parser(
        "optimize", help="Analyze a .mxai file and suggest optimizations"
    )
    optimize_parser.add_argument("file")
    optimize_parser.add_argument("--json", action="store_true", help="Print suggestions as JSON")

    mathematize_parser = subparsers.add_parser(
        "mathematize", help="Translate discrete if/else rules to continuous MatrixAI expressions"
    )
    mathematize_parser.add_argument(
        "file",
        help="Text file with one discrete rule per line (or - to read from stdin)",
    )
    mathematize_parser.add_argument("--json", action="store_true", help="Print output as JSON")

    compile_parser = subparsers.add_parser(
        "compile", help="Compile a .mxai file to an executable backend"
    )
    compile_parser.add_argument("file")
    compile_parser.add_argument(
        "--target",
        default="python",
        choices=["python", "differentiable-python"],
        help="Compilation target",
    )
    compile_parser.add_argument("--output", "-o", help="Output file path")

    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate a .mx mathematical expression file"
    )
    eval_parser.add_argument("file", help=".mx source file with MatrixAI expressions")
    eval_parser.add_argument(
        "--input",
        help="JSON input data: a file path or an inline JSON string",
    )
    eval_parser.add_argument(
        "--call",
        help="Call a specific defined function by name (default: evaluate all)",
    )
    eval_parser.add_argument("--json", action="store_true", help="Print output as JSON")
    eval_parser.add_argument("--trace", action="store_true", help="Include evaluation trace")
    eval_parser.add_argument("--graph", action="store_true", help="Print computation graph")

    playground_parser = subparsers.add_parser(
        "playground", help="Start the local MatrixAI prompt-to-runtime playground"
    )
    playground_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    playground_parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    playground_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the playground URL in the default browser",
    )

    refine_parser = subparsers.add_parser(
        "refine",
        help="Propose a refined prompt based on an audit or evaluation report",
    )
    refine_parser.add_argument(
        "prompt",
        nargs="+",
        help="Original prompt text, or - to read from stdin",
    )
    refine_parser.add_argument(
        "--audit",
        help="Path to an audit JSON file (activates audit_driven mode)",
    )
    refine_parser.add_argument(
        "--evaluation",
        help="Path to an evaluation_report.json (activates metric_driven mode)",
    )
    refine_parser.add_argument(
        "--mxai",
        help="Path to the current .mxai model file (adds context to audit hints)",
    )
    refine_parser.add_argument(
        "--hint",
        action="append",
        dest="hints",
        metavar="HINT",
        help="Additional user hint (can be repeated)",
    )
    refine_parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Current iteration number in the refinement chain (default: 1)",
    )
    refine_parser.add_argument(
        "--chain",
        help="Path to a JSON file containing the prior refinement_chain (list of IDs)",
    )
    refine_parser.add_argument(
        "--parent-hash",
        dest="parent_hash",
        help="SHA256 of the original root prompt (pass from previous iteration)",
    )
    refine_parser.add_argument(
        "--output", "-o",
        help="Write the proposed prompt to this file",
    )
    refine_parser.add_argument(
        "--mxai-output",
        help="Write the generated .mxai (with embedded refinement metadata) to this file",
    )
    refine_parser.add_argument(
        "--chain-output",
        help="Write the updated refinement_chain JSON to this file (for next iteration)",
    )
    refine_parser.add_argument(
        "--accept",
        action="store_true",
        help="Explicitly accept the proposal (required to write output files)",
    )
    refine_parser.add_argument(
        "--max-iterations",
        type=int,
        default=RefinementAgent.DEFAULT_MAX_ITERATIONS,
        dest="max_iterations",
        help=f"Hard iteration limit (default: {RefinementAgent.DEFAULT_MAX_ITERATIONS}). "
             "Stops with exit code 2 when iteration exceeds this value.",
    )
    refine_parser.add_argument(
        "--json",
        action="store_true",
        help="Print full RefinementProposal as JSON",
    )

    # ── P20 action commands ───────────────────────────────────────────────────

    validate_actions_parser = subparsers.add_parser(
        "validate-actions",
        help="Validate a .mxact contract file against a .mxai program",
    )
    validate_actions_parser.add_argument("contract", help="Path to the .mxact file")
    validate_actions_parser.add_argument("mxai", help="Path to the .mxai program file")
    validate_actions_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    dry_run_action_parser = subparsers.add_parser(
        "dry-run-action",
        help="Simulate a contract action (dry run) and print the DryRunReport",
    )
    dry_run_action_parser.add_argument("contract", help="Path to the .mxact file")
    dry_run_action_parser.add_argument("mxai", help="Path to the .mxai program file")
    dry_run_action_parser.add_argument("--contract-name", required=True, dest="contract_name",
                                       help="Name of the ACTION_CONTRACT to simulate")
    dry_run_action_parser.add_argument("--input", dest="input_json",
                                       help="JSON object with input_data (or - for stdin)")
    dry_run_action_parser.add_argument("--model-hash", default="cli", dest="model_hash")
    dry_run_action_parser.add_argument("--param-set", default="default", dest="param_set")
    dry_run_action_parser.add_argument("--json", action="store_true", help="Print report as JSON")

    execute_action_parser = subparsers.add_parser(
        "execute-action",
        help="Execute a contract action (requires --allow-real-actions)",
    )
    execute_action_parser.add_argument("contract", help="Path to the .mxact file")
    execute_action_parser.add_argument("mxai", help="Path to the .mxai program file")
    execute_action_parser.add_argument("--contract-name", required=True, dest="contract_name",
                                       help="Name of the ACTION_CONTRACT to execute")
    execute_action_parser.add_argument("--input", dest="input_json",
                                       help="JSON object with input_data (or - for stdin)")
    execute_action_parser.add_argument("--model-hash", default=None, dest="model_hash")
    execute_action_parser.add_argument("--param-set", default="default", dest="param_set")
    execute_action_parser.add_argument("--allow-real-actions", action="store_true",
                                       dest="allow_real_actions",
                                       help="Enable real action execution (off by default)")
    execute_action_parser.add_argument("--signing-key", dest="signing_key",
                                       help="Hex signing key (or set MATRIXAI_ACTION_SIGNING_KEY)")
    execute_action_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    audit_action_parser = subparsers.add_parser(
        "audit-action",
        help="Verify an ActionTrace HMAC signature",
    )
    audit_action_parser.add_argument("trace_file", help="Path to the ActionTrace JSON file")
    audit_action_parser.add_argument("--signing-key", dest="signing_key",
                                     help="Hex signing key (or set MATRIXAI_ACTION_SIGNING_KEY)")
    audit_action_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    registry_parser = subparsers.add_parser(
        "registry",
        help="Manage the local model registry",
    )
    registry_sub = registry_parser.add_subparsers(dest="registry_command", required=True)

    reg_push_p = registry_sub.add_parser("push", help="Push an entry to the registry")
    reg_push_p.add_argument("source_dir", help="Path to training run directory")
    reg_push_p.add_argument("--name", required=True, help="Model name (lowercase)")
    reg_push_p.add_argument("--version", required=True, dest="version_tag", help="Version tag (e.g. v1)")
    reg_push_p.add_argument("--registry-path", default="matrixai_registry", dest="registry_path")

    reg_pull_p = registry_sub.add_parser("pull", help="Pull an entry to a target registry")
    reg_pull_p.add_argument("entry", help="name@version to pull")
    reg_pull_p.add_argument("--from", dest="src_registry", default="matrixai_registry")
    reg_pull_p.add_argument("--to", dest="dst_registry", required=True)

    reg_list_p = registry_sub.add_parser("list", help="List registry entries")
    reg_list_p.add_argument("--name", default=None, help="Filter by name")
    reg_list_p.add_argument("--registry-path", default="matrixai_registry", dest="registry_path")
    reg_list_p.add_argument("--json", action="store_true", help="Output JSON")

    reg_show_p = registry_sub.add_parser("show", help="Show manifest for a registry entry")
    reg_show_p.add_argument("entry", help="name@version to show")
    reg_show_p.add_argument("--registry-path", default="matrixai_registry", dest="registry_path")
    reg_show_p.add_argument("--json", action="store_true", help="Output JSON")

    reg_tag_p = registry_sub.add_parser("tag", help="Create or move a tag alias")
    reg_tag_p.add_argument("entry", help="name@version to tag")
    reg_tag_p.add_argument("tag_name", help="Tag name (e.g. latest, prod)")
    reg_tag_p.add_argument("--registry-path", default="matrixai_registry", dest="registry_path")

    reg_verify_p = registry_sub.add_parser("verify", help="Verify integrity of a registry entry")
    reg_verify_p.add_argument("entry", help="name@version to verify")
    reg_verify_p.add_argument("--registry-path", default="matrixai_registry", dest="registry_path")

    reg_diff_p = registry_sub.add_parser("diff", help="Show differences between two versions")
    reg_diff_p.add_argument("entry_a", help="name@version_a")
    reg_diff_p.add_argument("entry_b", help="name@version_b")
    reg_diff_p.add_argument("--registry-path", default="matrixai_registry", dest="registry_path")

    # ── continual subcommand (P22) ─────────────────────────────────────────────
    continual_parser = subparsers.add_parser(
        "continual",
        help="Manage continual learning policies (.mxcontinual)",
    )
    continual_sub = continual_parser.add_subparsers(dest="continual_command", required=True)

    cont_ingest_p = continual_sub.add_parser(
        "ingest", help="Ingest ground truth for a production ActionTrace"
    )
    cont_ingest_p.add_argument("policy", help="Path to .mxcontinual policy file")
    cont_ingest_p.add_argument("--trace-id", dest="trace_id", required=True,
                               help="ActionTrace report_id (trace to annotate)")
    cont_ingest_p.add_argument("--label", dest="label", required=True,
                               help="Ground truth label or value")
    cont_ingest_p.add_argument("--trace-file", dest="trace_file", default=None,
                               help="Path to ActionTrace JSON file (to load the trace)")
    cont_ingest_p.add_argument("--signing-key", dest="signing_key", default=None,
                               help="Hex HMAC signing key (or set MATRIXAI_CONTINUAL_SIGNING_KEY)")
    cont_ingest_p.add_argument("--json", action="store_true", help="Output JSON")

    # continual init
    cont_init_p = continual_sub.add_parser(
        "init", help="Validate and display a .mxcontinual policy summary"
    )
    cont_init_p.add_argument("policy", help="Path to .mxcontinual policy file")
    cont_init_p.add_argument("--json", action="store_true", help="Output JSON")

    # continual status
    cont_status_p = continual_sub.add_parser(
        "status", help="Show current registry version for a continual policy"
    )
    cont_status_p.add_argument("policy", help="Path to .mxcontinual policy file")
    cont_status_p.add_argument("--registry-dir", dest="registry_dir", default="matrixai_registry",
                               help="Path to model registry directory")
    cont_status_p.add_argument("--json", action="store_true", help="Output JSON")

    # continual promote
    cont_promote_p = continual_sub.add_parser(
        "promote", help="Promote a candidate ParameterSet via ContinualVersioner"
    )
    cont_promote_p.add_argument("policy", help="Path to .mxcontinual policy file")
    cont_promote_p.add_argument("--approval-report", dest="approval_report", required=True,
                                help="Path to ApprovalGateReport JSON file")
    cont_promote_p.add_argument("--candidate-params", dest="candidate_params", required=True,
                                help="Path to candidate ParameterSet JSON file")
    cont_promote_p.add_argument("--registry-dir", dest="registry_dir", default="matrixai_registry",
                                help="Path to model registry directory")
    cont_promote_p.add_argument("--update-id", dest="update_id", default=None,
                                help="Continual update identifier (auto-generated if omitted)")
    cont_promote_p.add_argument("--human-approved", dest="human_approved", action="store_true",
                                help="Record a human approval decision for a pending candidate")
    cont_promote_p.add_argument("--approved-by", dest="approved_by", default=None,
                                help="Human approver identity for --human-approved")
    cont_promote_p.add_argument("--signing-key", dest="signing_key", default=None,
                                help="Hex HMAC key to verify PendingApproval token signature")
    cont_promote_p.add_argument("--json", action="store_true", help="Output JSON")

    # continual rollback
    cont_rollback_p = continual_sub.add_parser(
        "rollback", help="Execute a manual rollback to the previous registry version"
    )
    cont_rollback_p.add_argument("policy", help="Path to .mxcontinual policy file")
    cont_rollback_p.add_argument("--registry-dir", dest="registry_dir", default="matrixai_registry",
                                 help="Path to model registry directory")
    cont_rollback_p.add_argument("--dry-run", dest="dry_run", action="store_true",
                                 help="Show what would be rolled back without executing")
    cont_rollback_p.add_argument("--signing-key", dest="signing_key", default=None,
                                 help="Hex HMAC signing key for RollbackEvent signature")
    cont_rollback_p.add_argument("--json", action="store_true", help="Output JSON")

    # continual audit
    cont_audit_p = continual_sub.add_parser(
        "audit", help="Show audit configuration and optional drift refinement hint"
    )
    cont_audit_p.add_argument("policy", help="Path to .mxcontinual policy file")
    cont_audit_p.add_argument("--drift-report", dest="drift_report", default=None,
                              help="Path to DriftReport JSON file for refinement hint check")
    cont_audit_p.add_argument("--prompt", dest="prompt", default=None,
                              help="Original prompt text (required with --drift-report for hint)")
    cont_audit_p.add_argument("--drift-persistence-days", dest="drift_persistence_days",
                              type=int, default=None,
                              help="Days drift has been continuously observed (gates REFINEMENT_DRIFT_PERSISTENCE_DAYS)")
    cont_audit_p.add_argument("--json", action="store_true", help="Output JSON")

    # ── init: create new project from template ──────────────────────────────────
    init_p = subparsers.add_parser("init", help="Create a new MatrixAI project from a template")
    init_p.add_argument("project_name", help="Name of the project to create")
    init_p.add_argument("--template", dest="template", default="classification",
                        help="Template to use (default: classification)")
    init_p.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Directory where to create the project (default: current directory)")
    init_p.add_argument("--list-templates", dest="list_templates", action="store_true",
                        help="List available templates and exit")

    args = parser.parse_args()

    # ── init handler ───────────────────────────────────────────────────────────
    if args.command == "init":
        from matrixai.scaffolding import list_templates as list_avail_templates, scaffold_project, ScaffoldError

        if args.list_templates:
            templates = list_avail_templates()
            if templates:
                print("Available templates:")
                for tmpl in templates:
                    print(f"  - {tmpl}")
            else:
                print("No templates found.")
            return 0

        try:
            project_path = scaffold_project(
                args.project_name,
                args.template,
                output_dir=args.output_dir,
            )
            project_display_name = project_path.name
            print(f"✓ Project '{project_display_name}' created at {project_path}")
            model_path = project_path / f"{project_display_name}.mxai"
            training_path = project_path / f"{project_display_name}.mxtrain"
            output_path = project_path / "runs" / "v1"
            params_path = output_path / "params.best.json"
            input_path = project_path / "input" / "sample.json"
            print(f"\nNext steps:")
            print(f"  1. python3 -m matrixai train {model_path} --training {training_path} --output {output_path}")
            print(f"  2. python3 -m matrixai run {model_path} --params {params_path} --input {input_path} --json")
            print(f"\nSee {project_path / 'README.md'} for a full walkthrough.")
            return 0
        except ScaffoldError as e:
            print(str(e), file=sys.stderr)
            return 1

    if args.command == "parse":
        program = parse_file(args.file)
        print(json.dumps(program.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "prompt":
        prompt_text = sys.stdin.read() if args.prompt == ["-"] else " ".join(args.prompt)
        synthesis = PromptAgent().synthesize(prompt_text)
        architect = ArchitectAgent()
        plan = architect.plan_from_text(synthesis.semantic_text)
        plan_validation_code = _print_plan_validation(plan, stream=sys.stderr, emit_ok=False)
        if plan_validation_code != 0:
            return plan_validation_code
        mxai_text = architect.to_mxai(plan)
        if args.json:
            data = synthesis.to_dict()
            data["plan"] = plan.to_dict()
            data["mxai"] = mxai_text
            output = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        elif args.semantic:
            output = synthesis.semantic_text
        else:
            output = mxai_text
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Generated {args.output}")
        else:
            print(output, end="")
        return 0

    if args.command == "propose":
        prompt_text = sys.stdin.read() if args.prompt == ["-"] else " ".join(args.prompt)
        try:
            provider = None
            if args.provider == "chat-completions-compatible":
                provider = ChatCompletionsLLMProposalProvider.from_env()
            decision = LLMProposalAgent(provider=provider).propose_and_supervise(
                prompt_text, max_candidates=args.max_candidates
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Proposal provider error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            data = decision.to_dict()
            if not args.include_compiled:
                for report in data["reports"]:
                    report["compiled_python"] = ""
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(decision.summary())
        return 0 if decision.accepted else 1

    if args.command == "supervise-prompt":
        prompt_text = sys.stdin.read() if args.prompt == ["-"] else " ".join(args.prompt)
        supervisor = PromptSupervisor()
        if args.proposal:
            semantic_text = Path(args.proposal).read_text(encoding="utf-8")
            report = supervisor.supervise_semantic(
                prompt=prompt_text,
                semantic_text=semantic_text,
                source=f"proposal:{args.proposal}",
            )
        else:
            report = supervisor.supervise_prompt(prompt_text)
        if args.json:
            data = report.to_dict()
            if not args.include_compiled:
                data["compiled_python"] = ""
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(report.summary())
        return 0 if report.accepted else 1

    if args.command == "architect":
        semantic_text = Path(args.file).read_text(encoding="utf-8")
        architect = ArchitectAgent()
        plan = architect.plan_from_text(semantic_text)
        plan_validation_code = _print_plan_validation(plan, stream=sys.stderr, emit_ok=False)
        if plan_validation_code != 0:
            return plan_validation_code
        if args.json:
            plan_text = json.dumps(plan.to_dict(), indent=2, ensure_ascii=False)
            if args.output:
                Path(args.output).write_text(plan_text + "\n", encoding="utf-8")
                print(f"Generated {args.output}")
            else:
                print(plan_text)
            return 0
        mxai_text = architect.to_mxai(plan)
        if args.output:
            Path(args.output).write_text(mxai_text, encoding="utf-8")
            print(f"Generated {args.output}")
        else:
            print(mxai_text, end="")
        return 0

    if args.command == "validate-plan":
        semantic_text = Path(args.file).read_text(encoding="utf-8")
        plan = ArchitectAgent().plan_from_text(semantic_text)
        return _print_plan_validation(plan)

    if args.command == "validate":
        program = parse_file(args.file)
        return _print_validation(program)

    if args.command == "lint":
        return _cmd_lint(args)

    if args.command == "format":
        return _cmd_format(args)

    if args.command == "graph":
        return _cmd_graph(args)

    if args.command == "diagnose":
        return _cmd_diagnose(args)

    if args.command == "typecheck":
        return _cmd_typecheck(args)

    if args.command == "permissions":
        return _cmd_permissions(args)

    if args.command == "backend-report":
        return _cmd_backend_report(args)

    if args.command == "backend-run":
        return _cmd_backend_run(args)

    if args.command == "export-onnx":
        return _cmd_export_onnx(args)

    if args.command == "export-bundle":
        return _cmd_export_bundle(args)

    if args.command == "export-wasm":
        return _cmd_export_wasm(args)

    if args.command == "serve":
        return _cmd_serve(args)

    if args.command == "pack":
        return _cmd_pack(args)

    if args.command == "run":
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=args.json)
        if validation_code != 0:
            return validation_code
        input_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        runtime_parameters = None
        if args.params:
            try:
                parameter_set = load_parameter_set(args.params)
                parameter_validation = validate_parameter_set(program, parameter_set)
            except (OSError, ValueError) as exc:
                print(f"Parameter validation error: {exc}", file=sys.stderr)
                return 2
            if not parameter_validation.ok:
                for error in parameter_validation.errors:
                    print(f"Parameter error: {error}", file=sys.stderr)
                return 1
            runtime_parameters = parameter_set.runtime_parameters()
        result = MatrixAIRuntime().run(program, input_data, parameters=runtime_parameters)
        result["audit"] = AuditorAgent().explain(result)
        if args.json:
            print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
        else:
            _print_run_report(program.project, result)
        return 0

    if args.command == "backend-parameters":
        return _cmd_backend_parameters(args)

    if args.command == "init-parameters":
        return _cmd_init_parameters(args)

    if args.command == "validate-parameters":
        return _cmd_validate_parameters(args)

    if args.command == "validate-training":
        return _cmd_validate_training(args)

    if args.command == "generate-training":
        return _cmd_generate_training(args)

    if args.command == "generate-supervised":
        return _cmd_generate_supervised(args)

    if args.command == "train-supervised":
        return _cmd_train_supervised(args)

    if args.command == "generate-dataset":
        return _cmd_generate_dataset(args)

    if args.command == "train":
        return _cmd_train(args)

    if args.command == "evaluate":
        return _cmd_evaluate(args)

    if args.command == "optimize":
        program = parse_file(args.file)
        report = OptimizerAgent().analyze(program)
        if args.json:
            suggestions = [
                {"kind": s.kind, "description": s.description, "nodes": s.nodes, "detail": s.detail}
                for s in report.suggestions
            ]
            print(json.dumps({"project": program.project, "suggestions": suggestions}, indent=2, ensure_ascii=False))
        else:
            print(report.summary())
        return 0

    if args.command == "mathematize":
        if args.file == "-":
            text = sys.stdin.read()
        else:
            text = Path(args.file).read_text(encoding="utf-8")
        report = MathematicalAgent().translate_text(text)
        if args.json:
            data = {
                "translations": [
                    {
                        "original_rule": t.original_rule,
                        "expression": t.expression,
                        "expression_kind": t.expression_kind,
                        "inputs": t.inputs,
                        "parameters": t.parameters,
                        "explanation": t.explanation,
                    }
                    for t in report.translations
                ],
                "unresolved": report.unresolved,
            }
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(report.summary())
        return 0

    if args.command == "compile":
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=True)
        if validation_code != 0:
            return validation_code
        if args.target == "python":
            output = PythonBackendCompiler().compile(program)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"Compiled {program.project} -> {args.output}")
            else:
                print(output, end="")
            return 0
        if args.target == "differentiable-python":
            try:
                output = DifferentiablePythonCompiler().compile(program)
            except ValueError as exc:
                print(f"Compile error: {exc}", file=sys.stderr)
                return 1
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"Compiled {program.project} -> {args.output}")
            else:
                print(output, end="")
            return 0

    if args.command == "eval":
        return _cmd_eval(args)

    if args.command == "playground":
        from matrixai.playground import serve

        return serve(host=args.host, port=args.port, open_browser=args.open)

    if args.command == "refine":
        return _cmd_refine(args)

    if args.command == "validate-actions":
        return _cmd_validate_actions(args)

    if args.command == "dry-run-action":
        return _cmd_dry_run_action(args)

    if args.command == "execute-action":
        return _cmd_execute_action(args)

    if args.command == "audit-action":
        return _cmd_audit_action(args)

    if args.command == "registry":
        return _cmd_registry(args)

    if args.command == "continual":
        return _cmd_continual(args)

    if args.command == "keys":
        return _handle_keys(args)

    return 1


def _cmd_eval(args) -> int:
    from matrixai.core import Evaluator, ParseError, ast_to_graph, graph_to_text, parse
    from matrixai.functions import build_default_registry

    source = Path(args.file).read_text(encoding="utf-8")
    try:
        stmts = parse(source)
    except ParseError as exc:
        print(f"Parse error: {exc}", file=sys.stderr)
        return 2

    # Load input data
    env: dict = {}
    if args.input:
        raw = args.input.strip()
        if raw.startswith("{") or raw.startswith("["):
            env = json.loads(raw)
        else:
            env = json.loads(Path(raw).read_text(encoding="utf-8"))

    registry = build_default_registry()
    ev = Evaluator(registry)
    ev.define_all(stmts)

    results = []
    targets = [args.call] if args.call else [s.name for s in stmts]

    for name in targets:
        try:
            value, trace = ev.eval_definition(name, env)
        except Exception as exc:  # noqa: BLE001
            print(f"Evaluation error [{name}]: {exc}", file=sys.stderr)
            results.append({"name": name, "error": str(exc)})
            continue
        entry: dict = {"name": name, "value": value}
        if args.trace:
            entry["trace"] = trace.to_dict()
        if args.graph:
            assign = next(s for s in stmts if s.name == name)
            entry["graph"] = ast_to_graph(assign)
        results.append(entry)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            if "error" in r:
                print(f"{r['name']} = ERROR: {r['error']}")
            else:
                print(f"{r['name']} = {r['value']:.6g}")
            if args.trace and "trace" in r:
                for step in r["trace"]["steps"]:
                    print(f"  {step['op']}({', '.join(str(a) for a in step['args'])}) = {step['result']:.6g}")
            if args.graph and "graph" in r:
                assign = next(s for s in stmts if s.name == r["name"])
                from matrixai.core import ast_to_graph as _atg
                print(graph_to_text(ast_to_graph(assign)))

    return 0 if all("error" not in r for r in results) else 1


def _cmd_typecheck(args) -> int:
    from matrixai.core import parse
    from matrixai.types import check_mx_types, check_program_types

    path = Path(args.file)
    if path.suffix == ".mxai":
        program = parse_file(path)
        report = check_program_types(program)
        # If the program declares IMPORT blocks, also run composite type checking.
        if getattr(program, "imports", []) and not report.errors:
            from matrixai.registry import ModelRegistry
            from matrixai.types import check_composite_program_types
            registry_path = getattr(args, "registry_path", "matrixai_registry")
            reg = ModelRegistry(registry_path)
            try:
                allow_mutable = getattr(args, "allow_mutable_imports", False)
                composite_report = check_composite_program_types(
                    program, reg, allow_mutable_tags=allow_mutable
                )
                report = type(report)(
                    errors=report.errors + composite_report.errors,
                    warnings=report.warnings + composite_report.warnings,
                    symbols=report.symbols,
                )
            except Exception as exc:  # noqa: BLE001
                # Surface per-import errors directly when available (e.g. mutable tag policy).
                detail = getattr(exc, "errors", None)
                if detail:
                    extra_errors = list(detail)
                else:
                    extra_errors = [f"Composite typecheck failed: {exc}"]
                report = type(report)(
                    errors=report.errors + extra_errors,
                    warnings=report.warnings,
                    symbols=report.symbols,
                )
    elif path.suffix == ".mx":
        report = check_mx_types(parse(path.read_text(encoding="utf-8")))
    else:
        print("Typecheck supports .mx and .mxai files", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        if report.ok:
            print(f"Typecheck OK: {path.name}")
        for warning in report.warnings:
            print(f"Warning: {warning}")
        for error in report.errors:
            print(f"Error: {error}")
        if report.symbols:
            print("Symbols:")
            for name, spec in sorted(report.symbols.items()):
                print(f"  {name}: {spec.name}")
    return 0 if report.ok else 1


def _cmd_lint(args) -> int:
    from matrixai.tooling import lint_path

    try:
        report = lint_path(args.file)
    except (OSError, ValueError) as exc:
        print(f"Lint error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        if report.ok and not report.has_warnings:
            print(f"Lint OK: {Path(args.file).name}")
        for diagnostic in report.diagnostics:
            location = f":{diagnostic.line}" if diagnostic.line else ""
            print(
                f"{diagnostic.severity.upper()}[{diagnostic.source}]{location}: "
                f"{diagnostic.message}"
            )

    if not report.ok:
        return 1
    if args.strict and report.has_warnings:
        return 1
    return 0


def _cmd_permissions(args) -> int:
    from matrixai.sandbox import SandboxPolicy

    try:
        report = SandboxPolicy.mvp_simulate_only().review(parse_file(args.file))
    except (OSError, ValueError) as exc:
        print(f"Permission review error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0 if report.ok else 1


def _cmd_backend_report(args) -> int:
    try:
        report = BackendContractAnalyzer(target=args.target).analyze(parse_file(args.file))
    except (OSError, ValueError) as exc:
        print(f"Backend report error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0 if report.ok else 1


def _cmd_backend_run(args) -> int:
    try:
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=True)
        if validation_code != 0:
            return validation_code
        if args.target == "torch":
            input_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
            parameter_set = _load_torch_backend_parameter_set(program, args.parameters)
            device = getattr(args, "device", None) or "cpu"
            result = TorchForwardRunner(device=device).run(program, input_data, parameter_set)
            if args.json:
                print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
            else:
                _print_backend_run_report(program.project, result)
            return 0
        input_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        namespace: dict[str, object] = {}
        exec(DifferentiablePythonCompiler().compile(program), namespace)
        parameters = _load_backend_parameters(args.parameters, namespace)
        result = namespace["run"](input_data, parameters) if parameters is not None else namespace["run"](input_data)
    except OSError as exc:
        print(f"Backend run error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Backend run error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    else:
        _print_backend_run_report(program.project, result)
    return 0


def _load_torch_backend_parameter_set(program, parameters_arg):
    if not parameters_arg or parameters_arg == "initial":
        return build_initial_parameter_set(program, parameter_set_id=f"{program.project}_torch_initial")
    parameter_set = load_parameter_set(parameters_arg)
    validation = validate_parameter_set(program, parameter_set)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    return parameter_set


def _cmd_backend_parameters(args) -> int:
    try:
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=True)
        if validation_code != 0:
            return validation_code
        if args.target == "torch":
            report = BackendContractAnalyzer(target="torch").analyze(program)
            if not report.ok:
                issues = [node.node for node in report.unsupported_nodes] + list(report.parameter_errors)
                raise ValueError(f"Program {program.project} is not portable to torch: {', '.join(issues)}")
            if args.validate:
                parameter_set = load_parameter_set(args.validate)
                compatibility = validate_parameter_set(program, parameter_set)
                errors = list(compatibility.errors) + validate_parameter_set_for_torch(parameter_set)
                payload = {
                    "ok": not errors,
                    "target": "torch",
                    "project": program.project,
                    "parameter_set_id": parameter_set.parameter_set_id,
                    "errors": errors,
                    "parameter_manifest": report.parameter_manifest,
                    "backend": report.backend,
                }
                if args.json:
                    print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
                else:
                    if errors:
                        for error in errors:
                            print(f"Error: {error}")
                    else:
                        print(f"Parameters OK: {program.project}")
                return 0 if not errors else 1

            parameter_set = build_initial_parameter_set(
                program,
                parameter_set_id=f"{program.project}_torch_initial",
            )
            payload = {
                "target": "torch",
                "project": program.project,
                "parameters": parameter_set.runtime_parameters(),
                "parameter_set": parameter_set.to_dict(),
                "parameter_manifest": report.parameter_manifest,
                "backend": report.backend,
            }
            if args.json:
                print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
            else:
                _print_backend_parameters_report(payload)
            return 0
        namespace: dict[str, object] = {}
        exec(DifferentiablePythonCompiler().compile(program), namespace)
        if args.validate:
            parameters = json.loads(Path(args.validate).read_text(encoding="utf-8"))
            errors = namespace["validate_parameters"](parameters)
            payload = {
                "ok": not errors,
                "target": namespace["TARGET"],
                "project": program.project,
                "errors": errors,
                "parameter_manifest": namespace["PARAMETER_MANIFEST"],
            }
            if args.json:
                print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
            else:
                if errors:
                    for error in errors:
                        print(f"Error: {error}")
                else:
                    print(f"Parameters OK: {program.project}")
            return 0 if not errors else 1

        payload = {
            "target": namespace["TARGET"],
            "project": program.project,
            "parameters": namespace["initial_parameters"](),
            "parameter_manifest": namespace["PARAMETER_MANIFEST"],
        }
    except OSError as exc:
        print(f"Backend parameters error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Backend parameters error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
    else:
        _print_backend_parameters_report(payload)
    return 0


def _cmd_validate_training(args) -> int:
    path = Path(args.file)
    try:
        training = parse_training_file(path)
        report = TrainingVerifier().verify(training, base_path=path.parent)
    except OSError as exc:
        print(f"Training validation error: {exc}", file=sys.stderr)
        return 2
    except MatrixAITrainingParseError as exc:
        print(f"Training validation error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = report.to_dict()
        payload["training"] = training.to_dict()
        print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0 if report.ok else 1


def _cmd_generate_training(args) -> int:
    prompt_text = sys.stdin.read() if args.prompt == ["-"] else " ".join(args.prompt)
    dataset_source = args.dataset_source or args.dataset_output
    try:
        labels = _parse_cli_labels(args.labels)
        result = TrainingPromptGenerator().generate(
            prompt_text,
            args.file,
            dataset_source=dataset_source,
            dataset_name=args.dataset_name,
            target_name=args.target_name,
            labels=labels,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        written: dict[str, str] = {}
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.training_text, encoding="utf-8")
            written["training"] = str(output_path)
        if args.dataset_output:
            dataset_path = Path(args.dataset_output)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(result.dataset_template_text, encoding="utf-8")
            written["dataset_template"] = str(dataset_path)
    except OSError as exc:
        print(f"Training generation error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Training generation error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = result.to_dict()
        payload["written"] = written
        print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
    elif not args.output:
        print(result.training_text, end="")
    else:
        print(result.summary())
        for kind, path in written.items():
            print(f"Generated {kind}: {path}")
    return 0


def _cmd_generate_supervised(args) -> int:
    prompt_text = sys.stdin.read() if args.prompt == ["-"] else " ".join(args.prompt)
    try:
        labels = _parse_cli_labels(args.labels)
        result = SupervisedPromptGenerator().generate(
            prompt_text,
            args.output_dir,
            artifact_stem=args.stem,
            dataset_name=args.dataset_name,
            target_name=args.target_name,
            labels=labels,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    except OSError as exc:
        print(f"Supervised generation error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Supervised generation error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False))
    else:
        print(result.summary())
    return 0


def _resolve_backend_spec(
    cli_target: str | None,
    cli_device: str | None,
    training_backend: "BackendSpec | None",
) -> "BackendSpec":
    """Merge CLI flags, .mxtrain BACKEND block, and hard defaults.

    Priority: explicit CLI flag > .mxtrain BACKEND block > stdlib/cpu defaults.
    Raises ValueError (via BackendSpec) if the resolved combination is invalid.
    """
    target = cli_target or (training_backend.target if training_backend else "stdlib")
    device = cli_device or (training_backend.device if training_backend else "cpu")
    return BackendSpec(target=target, device=device)


def _validate_device(backend: str, device: str) -> str | None:
    """Return an error message if the device is not available, None if OK."""
    if device == "cpu":
        return None
    if backend != "torch":
        return f"--device {device!r} requires --backend torch"
    from matrixai.parameters import torch_device_info
    info = torch_device_info()
    if device not in info["available_devices"]:
        avail = ", ".join(info["available_devices"])
        return f"Device {device!r} is not available in this environment. Available: {avail}"
    return None


def _cmd_train_supervised(args) -> int:
    prompt_text = sys.stdin.read() if args.prompt == ["-"] else " ".join(args.prompt)
    try:
        labels = _parse_cli_labels(args.labels)
        result = SupervisedPromptRunner().run(
            prompt_text,
            args.output_dir,
            train_data=args.train_data,
            evaluation_data=args.eval_data,
            dataset_manifest=args.dataset_manifest,
            dataset_split=args.dataset_split,
            artifact_stem=args.stem,
            dataset_name=args.dataset_name,
            target_name=args.target_name,
            labels=labels,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            run_name=args.run_name,
        )
    except OSError as exc:
        print(f"Supervised training error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Supervised training error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False))
    else:
        print(result.summary())
    return 0


def _parse_cli_labels(value: str | None) -> list[str] | None:
    if value is None:
        return None
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels requires at least one label")
    return labels


def _cmd_train(args) -> int:
    training_path = Path(args.training)
    try:
        training = parse_training_file(training_path)
        _file_resolved = Path(args.file).resolve()
        _model_from_parent = (training_path.parent / training.model).resolve()
        _model_from_cwd = Path(training.model).resolve()
        if _file_resolved not in (_model_from_parent, _model_from_cwd):
            print(
                f"Training spec MODEL {training.model} does not match command model {args.file}",
                file=sys.stderr,
            )
            return 1
        resolved = _resolve_backend_spec(args.backend, args.device, training.backend)
        device_error = _validate_device(resolved.target, resolved.device)
        if device_error:
            print(f"Training error: {device_error}", file=sys.stderr)
            return 1
        training = dataclasses.replace(training, backend=resolved)
        if resolved.target == "torch":
            trainer: Any = TorchSupervisedTrainer()
        else:
            _prog = parse_file(args.file)
            trainer = DenseSupervisedTrainer() if getattr(_prog, "networks", []) else SupervisedTrainer()
        result = trainer.train(
            training,
            output_dir=args.output,
            base_path=training_path.parent,
            training_path=training_path,
        )
    except OSError as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 2
    except (MatrixAITrainingParseError, ValueError, NotImplementedError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False))
    else:
        print(f"Training OK: {result.run_id}")
        print(f"Best epoch: {result.best_epoch}")
        print(f"Best validation loss: {result.best_validation_loss:.6f}")
        print(f"Accuracy: {result.accuracy:.6f}")
        print(f"Artifacts: {result.output_dir}")
    return 0


def _cmd_evaluate(args) -> int:
    training_path = Path(args.training)
    try:
        training = parse_training_file(training_path)
        _file_resolved = Path(args.file).resolve()
        _model_from_parent = (training_path.parent / training.model).resolve()
        _model_from_cwd = Path(training.model).resolve()
        if _file_resolved not in (_model_from_parent, _model_from_cwd):
            print(
                f"Training spec MODEL {training.model} does not match command model {args.file}",
                file=sys.stderr,
            )
            return 1
        resolved = _resolve_backend_spec(
            getattr(args, "backend", None),
            getattr(args, "device", None),
            training.backend,
        )
        device_error = _validate_device(resolved.target, resolved.device)
        if device_error:
            print(f"Evaluation error: {device_error}", file=sys.stderr)
            return 1
        training = dataclasses.replace(training, backend=resolved)
        parameter_set = load_parameter_set(args.params)

        if resolved.target == "torch":
            from matrixai.training.torch_evaluator import TorchSupervisedEvaluator
            evaluator: Any = TorchSupervisedEvaluator()
        else:
            program = parse_file(args.file)
            if getattr(program, "networks", []):
                evaluator = DenseSupervisedEvaluator()
            else:
                prediction_kind = next(
                    (
                        function.semantic.kind
                        for function in program.functions
                        if function.output == training.loss.prediction
                    ),
                    "",
                )
                evaluator = (
                    GenericSupervisedEvaluator()
                    if prediction_kind == "layer_call"
                    else SupervisedEvaluator()
                )

        result = evaluator.evaluate(
            training,
            parameter_set=parameter_set,
            data_path=args.data,
            base_path=training_path.parent,
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except OSError as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    except (MatrixAITrainingParseError, ValueError, NotImplementedError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False))
    else:
        print(f"Evaluation OK: {Path(args.params).name}")
        print(f"Dataset: {result.dataset}")
        print(f"Fingerprint: {result.dataset_fingerprint}")
        print(f"Rows: {result.rows}")
        print(f"Loss: {result.loss:.6f}")
        print(f"Accuracy: {result.accuracy:.6f}")
        print(f"Macro F1: {result.macro_f1:.6f}")
        if args.output:
            print(f"Report: {args.output}")
    return 0


def _cmd_init_parameters(args) -> int:
    try:
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=True)
        if validation_code != 0:
            return validation_code
        parameter_set = build_initial_parameter_set(
            program,
            parameter_set_id=args.parameter_set_id,
        )
    except OSError as exc:
        print(f"Init parameters error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Init parameters error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        write_parameter_set(args.output, parameter_set)
        if not args.json:
            print(f"Generated {args.output}")
    if args.json or not args.output:
        print(json.dumps(_json_safe(parameter_set.to_dict()), indent=2, ensure_ascii=False))
    return 0


def _cmd_validate_parameters(args) -> int:
    try:
        program = parse_file(args.file)
        parameter_set = load_parameter_set(args.params)
        report = validate_parameter_set(program, parameter_set)
    except OSError as exc:
        print(f"Parameter validation error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Parameter validation error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = report.to_dict()
        payload["parameter_set_id"] = parameter_set.parameter_set_id
        print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0 if report.ok else 1


def _load_backend_parameters(value: str | None, namespace: dict[str, object]) -> dict[str, object] | None:
    if not value:
        return None
    if value == "initial":
        return namespace["initial_parameters"]()
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _cmd_format(args) -> int:
    from matrixai.tooling import format_path

    if args.check and args.write:
        print("format accepts --check or --write, not both", file=sys.stderr)
        return 2

    path = Path(args.file)
    try:
        formatted = format_path(path)
    except (OSError, ValueError) as exc:
        print(f"Format error: {exc}", file=sys.stderr)
        return 2

    current = path.read_text(encoding="utf-8").rstrip() + "\n"
    if args.check:
        if formatted != current:
            print(f"Would reformat: {path.name}")
            return 1
        print(f"Format OK: {path.name}")
        return 0

    if args.write:
        path.write_text(formatted, encoding="utf-8")
        print(f"Formatted {path}")
        return 0

    print(formatted, end="")
    return 0


def _cmd_graph(args) -> int:
    from matrixai.tooling import graph_path

    try:
        output = graph_path(args.file, output_format=args.format)
    except (OSError, ValueError) as exc:
        print(f"Graph error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Graph written to {args.output}")
    else:
        print(output, end="")
    return 0


def _cmd_diagnose(args) -> int:
    from matrixai.tooling import diagnose_runtime_compiler

    try:
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=args.json)
        if validation_code != 0:
            return validation_code
        input_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        report = diagnose_runtime_compiler(program, input_data, tolerance=args.tolerance)
    except (OSError, ValueError) as exc:
        print(f"Diagnostic error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        if report.ok:
            print(f"Diagnostic OK: runtime and compiled outputs match for {report.project}")
        else:
            print(f"Diagnostic mismatch: {report.project}")
            for mismatch in report.mismatches:
                print(f"  {mismatch}")
    return 0 if report.ok else 1


def _print_validation(program, quiet: bool = False) -> int:
    verifier_result = VerifierAgent().verify(program)
    safety_warnings = SafetyAgent().review(program)
    warnings = verifier_result.warnings + safety_warnings

    if not quiet:
        if verifier_result.ok:
            print(f"Validation OK: {program.project}")
        for warning in warnings:
            print(f"Warning: {warning}")
        for error in verifier_result.errors:
            print(f"Error: {error}")

    return 0 if verifier_result.ok and not safety_warnings else 1


def _print_plan_validation(
    plan, stream: TextIO = sys.stdout, emit_ok: bool = True
) -> int:
    result = PlannerVerifier().verify(plan)
    if emit_ok and result.ok:
        print(f"Plan validation OK: {plan.project}", file=stream)
    for warning in result.warnings:
        print(f"Warning: {warning}", file=stream)
    for error in result.errors:
        print(f"Error: {error}", file=stream)
    return 0 if result.ok else 1


def _print_run_report(project: str, result: dict) -> None:
    print(f"Project: {project}")
    for action in result["actions"]:
        status = "simulated" if action["activated"] else "skipped"
        print(
            f"Action: {action['name']} {status} "
            f"({action['source']}={action['value']:.4f}, threshold={action['threshold']:.4f})"
        )
    print("\nAudit:")
    print(result["audit"])


def _print_backend_run_report(project: str, result: dict) -> None:
    print(f"Project: {project}")
    print(f"Backend target: {result['target']}")
    print(f"State keys: {', '.join(sorted(result['state']))}")
    print(f"Runtime boundaries: {len(result['runtime_boundaries'])}")
    for boundary in result["runtime_boundaries"]:
        print(f"- {boundary['node']} ({boundary['node_type']})")
    print(f"Trainable parameters: {len(result['trainable_parameters'])}")


def _print_backend_parameters_report(payload: dict) -> None:
    print(f"Project: {payload['project']}")
    print(f"Target: {payload['target']}")
    print(f"Trainable parameters: {len(payload['parameter_manifest'])}")
    for parameter in payload["parameter_manifest"]:
        shape = parameter.get("shape", [])
        print(f"- {parameter['function']}.{parameter['name']} ({parameter['role']}) shape={shape}")


def _json_safe(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value
def _cmd_export_onnx(args) -> int:
    try:
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=True)
        if validation_code != 0:
            if not args.json:
                print(f"Export error: model {args.file} did not pass validation", file=sys.stderr)
            return validation_code
        parameter_set = load_parameter_set(args.params)
        parameter_validation = validate_parameter_set(program, parameter_set)
        if not parameter_validation.ok:
            for error in parameter_validation.errors:
                print(f"Parameter error: {error}", file=sys.stderr)
            return 1
        result = export_onnx(program, parameter_set, args.output)
    except OnnxExportError as exc:
        print(f"Export error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Export error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Export error: {exc}", file=sys.stderr)
        return 1

    if not args.validate:
        if args.json:
            print(json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False))
        else:
            print(f"Export OK: {result.output_path}")
            print(f"Format: ONNX opset {result.opset_version}")
            print(f"Input: {result.input_name} {result.input_shape}")
            print(f"Output: {result.output_name} {result.output_shape}")
            if result.labels:
                print(f"Labels: {result.labels}")
            if result.skipped_functions:
                print(f"Skipped (unsupported): {result.skipped_functions}")
    else:
        if not args.json:
            print(f"Export OK: {result.output_path}")

    if args.manifest and not args.validate:
        print("Warning: --manifest requires --validate; skipping manifest write.", file=sys.stderr)

    if args.validate:
        if not ort_available():
            print(
                "Equivalence check skipped: onnxruntime not installed. "
                "Run: pip install matrixai-core[export]",
                file=sys.stderr,
            )
            return 0
        try:
            eq = validate_onnx_equivalence(program, parameter_set, args.output)
        except OnnxEquivalenceError as exc:
            print(f"Equivalence check error: {exc}", file=sys.stderr)
            return 1

        if args.manifest:
            try:
                write_export_manifest(result, eq, args.manifest)
                if not args.json:
                    print(f"Manifest: {args.manifest}")
            except OSError as exc:
                print(f"Manifest write error: {exc}", file=sys.stderr)
                return 2

        if args.json:
            out = result.to_dict()
            out["equivalence_check"] = eq.to_dict()
            print(json.dumps(_json_safe(out), indent=2, ensure_ascii=False))
        else:
            status = "PASS" if eq.passed else "FAIL"
            print(
                f"Equivalence {status}: max_abs_diff={eq.max_abs_diff:.2e} "
                f"(atol={eq.atol:.0e}, n={eq.n_samples})"
            )

        if not eq.passed:
            print(
                f"Equivalence check FAILED: max_abs_diff={eq.max_abs_diff:.2e} "
                f"exceeds tolerance atol={eq.atol:.0e} + rtol={eq.rtol:.0e}",
                file=sys.stderr,
            )
            return 1

    return 0


_VALID_FIELD_TYPES = ("number", "integer", "boolean")


def _load_inference_metadata(path: str) -> dict:
    """Read + STRICTLY validate the --inference-metadata sidecar.

    Any present known key must be fully well-formed or a ValueError is raised.
    Silent coercion is dangerous here: a dropped range, a category string turned
    into a list of characters, or an unknown type would normalize the input wrong
    and yield a bundle that looks self-usable but predicts garbage.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("inference-metadata must be a JSON object")

    kwargs: dict = {}

    if "field_ranges" in raw:
        ranges = raw["field_ranges"]
        if not isinstance(ranges, dict):
            raise ValueError("field_ranges must be an object of {field: [min, max]}")
        out_ranges: dict[str, tuple[float, float]] = {}
        for key, pair in ranges.items():
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(f"field_ranges[{key!r}] must be a [min, max] pair")
            try:
                lo, hi = float(pair[0]), float(pair[1])
            except (TypeError, ValueError):
                raise ValueError(f"field_ranges[{key!r}] bounds must be numbers")
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"field_ranges[{key!r}] bounds must be finite")
            if not lo < hi:
                raise ValueError(f"field_ranges[{key!r}] requires min < max, got [{lo}, {hi}]")
            out_ranges[str(key)] = (lo, hi)
        kwargs["field_ranges"] = out_ranges

    if "field_categories" in raw:
        cats = raw["field_categories"]
        if not isinstance(cats, dict):
            raise ValueError("field_categories must be an object of {field: [values]}")
        out_cats: dict[str, list[str]] = {}
        for key, values in cats.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"field_categories[{key!r}] must be a non-empty list of strings")
            if not all(isinstance(x, str) for x in values):
                raise ValueError(f"field_categories[{key!r}] values must be strings")
            out_cats[str(key)] = list(values)
        kwargs["field_categories"] = out_cats

    if "field_types" in raw:
        types = raw["field_types"]
        if not isinstance(types, dict):
            raise ValueError("field_types must be an object of {field: type}")
        out_types: dict[str, str] = {}
        for key, value in types.items():
            if value not in _VALID_FIELD_TYPES:
                raise ValueError(
                    f"field_types[{key!r}] must be one of {list(_VALID_FIELD_TYPES)}, got {value!r}"
                )
            out_types[str(key)] = value
        kwargs["field_types"] = out_types

    if "labels" in raw:
        labels = raw["labels"]
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise ValueError("labels must be a list of strings")
        kwargs["labels"] = list(labels)

    if "example_input" in raw:
        example = raw["example_input"]
        if not isinstance(example, dict):
            raise ValueError("example_input must be an object of {field: value}")
        kwargs["example_input"] = dict(example)

    return kwargs


def _cmd_export_bundle(args) -> int:
    try:
        meta_kwargs: dict = {}
        if getattr(args, "inference_metadata", None):
            meta_kwargs = _load_inference_metadata(args.inference_metadata)
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=True)
        if validation_code != 0:
            print(f"Export error: model {args.file} did not pass validation", file=sys.stderr)
            return validation_code
        parameter_set = load_parameter_set(args.params)
        parameter_validation = validate_parameter_set(program, parameter_set)
        if not parameter_validation.ok:
            for error in parameter_validation.errors:
                print(f"Parameter error: {error}", file=sys.stderr)
            return 1
        result = create_edge_bundle(
            program, parameter_set,
            mxai_path=args.file,
            params_path=args.params,
            outdir=args.outdir,
            validate=not args.no_validate,
            force=args.force,
            **meta_kwargs,
        )
    except EdgeBundleError as exc:  # subclass of ValueError — must precede it
        print(f"Bundle error: {exc}", file=sys.stderr)
        return 1
    except OnnxExportError as exc:  # subclass of ValueError — must precede it
        print(f"Export error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        # malformed --inference-metadata sidecar (or another value error)
        print(f"Bundle error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Bundle error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False))
    else:
        print(f"Bundle OK: {result.bundle_dir}")
        print(f"Files: {', '.join(result.files)}")
        if result.equivalence_result is not None:
            status = "PASS" if result.equivalence_passed else "FAIL"
            print(
                f"Equivalence {status}: "
                f"max_abs_diff={result.equivalence_result.max_abs_diff:.2e}"
            )
        if result.inference_spec_skipped_reason is None:
            print("Self-usable: yes (predict.py + inference_spec.json included)")
        else:
            print(f"Self-usable: no — {result.inference_spec_skipped_reason}")
    return 0


def _cmd_export_wasm(args) -> int:
    try:
        program = parse_file(args.file)
        validation_code = _print_validation(program, quiet=True)
        if validation_code != 0:
            print(f"Export error: model {args.file} did not pass validation", file=sys.stderr)
            return validation_code
        parameter_set = load_parameter_set(args.params)
        parameter_validation = validate_parameter_set(program, parameter_set)
        if not parameter_validation.ok:
            for error in parameter_validation.errors:
                print(f"Parameter error: {error}", file=sys.stderr)
            return 1
        result = export_wasm(
            program, parameter_set,
            output_dir=args.outdir,
            validate=not args.no_validate,
            force=args.force,
        )
    except WasmExportError as exc:
        print(f"WASM export error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"WASM export error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_json_safe(result.to_dict()), indent=2, ensure_ascii=False))
    else:
        print(f"WASM bundle OK: {result.bundle_dir}")
        print(f"Files: {', '.join(result.files)}")
        if result.equivalence_result is not None:
            status = "PASS" if result.equivalence_passed else "FAIL"
            print(
                f"Equivalence {status}: "
                f"max_abs_diff={result.equivalence_result.max_abs_diff:.2e}"
            )
    return 0


def _cmd_serve(args) -> int:
    import os
    from matrixai.errors import error_serve_model_not_found, error_serve_params_not_found
    from matrixai.server import serve_model

    registry_path = Path(args.registry) if getattr(args, "registry", None) else None
    model_path = Path(args.file) if args.file else None
    params_path = Path(args.params) if args.params else None

    if model_path is None and registry_path is None:
        print("Error: provide a .mxai model file or --registry PATH (or both).", file=sys.stderr)
        return 2
    if model_path is not None and not model_path.exists():
        print(error_serve_model_not_found(str(model_path)), file=sys.stderr)
        return 2
    if params_path is not None and not params_path.exists():
        print(error_serve_params_not_found(str(params_path)), file=sys.stderr)
        return 2
    if registry_path is not None and not registry_path.exists():
        print(f"Error: registry path not found: {registry_path}", file=sys.stderr)
        return 2

    signing_key = getattr(args, "signing_key", None) or os.environ.get("MATRIXAI_ACTION_SIGNING_KEY")

    monitor = None
    continual_policy_path = getattr(args, "continual_policy", None)
    if continual_policy_path:
        from matrixai.continual.parser import parse_mxcontinual, MxcontinualParseError
        from matrixai.continual.monitor import ProductionMonitor
        try:
            policy_text = Path(continual_policy_path).read_text(encoding="utf-8")
            policy = parse_mxcontinual(policy_text)
        except (OSError, MxcontinualParseError) as exc:
            print(f"Error loading continual policy: {exc}", file=sys.stderr)
            return 2

        # Determine reference accuracy: explicit flag > params file metrics > None
        reference_accuracy: float | None = getattr(args, "reference_accuracy", None)
        if reference_accuracy is None and params_path and params_path.exists():
            try:
                import json as _json
                ps_data = _json.loads(params_path.read_text(encoding="utf-8"))
                reference_accuracy = ps_data.get("metrics", {}).get("accuracy")
            except Exception:
                pass

        monitor = ProductionMonitor(policy, reference_accuracy=reference_accuracy, labels=[])
        ref_str = f"{reference_accuracy:.4f}" if reference_accuracy is not None else "not set (degradation detection disabled)"
        print(f"Continual monitoring active (policy: {policy.name}, reference_accuracy: {ref_str})")

    return serve_model(
        file_path=model_path,
        params_path=params_path,
        host=args.host,
        port=args.port,
        backend=args.backend,
        api_key=args.api_key,
        contract_path=Path(args.contract) if getattr(args, "contract", None) else None,
        allow_real_actions=getattr(args, "allow_real_actions", False),
        signing_key=signing_key,
        rate_limit=getattr(args, "rate_limit", None),
        cors_origins=getattr(args, "cors_origins", None),
        monitor=monitor,
        api_key_read=getattr(args, "api_key_read", None),
        registry_path=registry_path,
    )

def _cmd_pack(args) -> int:
    from matrixai.pack import pack_model
    return pack_model(
        file_path=Path(args.file),
        params_path=Path(args.params) if args.params else None,
        contract_path=Path(args.contract) if getattr(args, "contract", None) else None,
        docker=args.docker,
        outdir=Path(args.outdir),
    )


def _cmd_generate_dataset(args) -> int:
    from matrixai.training.dataset_manifest import (
        DatasetManifestEntry,
        build_synthetic_manifest,
    )
    from matrixai.training.data import dataset_fingerprint
    from matrixai.training.synthetic import SyntheticDataGenerator

    rows_requested = args.rows
    if rows_requested < 2:
        print("--rows must be at least 2 (need at least 1 train + 1 eval row)", file=sys.stderr)
        return 1
    # M12: el tope de filas lo gobierna el perfil de límites (env/MATRIXAI_LIMITS_PROFILE),
    # no un hard cap. En 'ilimitado' no hay tope; en 'equilibrado' (default) son 50000.
    from matrixai import limits as _limits
    if _limits.exceeds(rows_requested, "max_rows"):
        print(f"--rows exceeds the configured maximum of {_limits.get_limit('max_rows')} "
              f"(raise it with MATRIXAI_LIMITS_PROFILE=ilimitado or MATRIXAI_MAX_ROWS)", file=sys.stderr)
        return 1

    try:
        program = parse_file(args.file)
    except Exception as exc:
        print(f"Error parsing .mxai: {exc}", file=sys.stderr)
        return 1

    try:
        training = parse_training_file(Path(args.training))
    except MatrixAITrainingParseError as exc:
        print(f"Error parsing .mxtrain: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem or program.project.lower().replace(" ", "_").replace("-", "_")

    try:
        generator = SyntheticDataGenerator(
            program=program,
            training=training,
            seed=args.seed,
            rows=rows_requested,
            mode=args.mode,
        )
        adapter = generator.generate()
    except ValueError as exc:
        print(f"Generation error: {exc}", file=sys.stderr)
        return 1

    if generator.coherent_fallback_count > 0:
        print(
            f"Warning: {generator.coherent_fallback_count}/{rows_requested} rows fell back to "
            f"random labels in coherent mode (runtime did not produce a valid label).",
            file=sys.stderr,
        )

    all_rows = adapter.rows
    train_count = max(1, int(len(all_rows) * 0.8))
    eval_count = len(all_rows) - train_count
    if eval_count < 1:
        train_count -= 1
        eval_count = 1

    schema = adapter.schema()
    columns = list(schema.input_columns) + [schema.target]

    train_path = output_dir / f"{stem}-synthetic-train.csv"
    eval_path = output_dir / f"{stem}-synthetic-eval.csv"
    manifest_path = output_dir / f"{stem}-synthetic-manifest.json"

    _write_rows_csv(train_path, columns, all_rows[:train_count])
    _write_rows_csv(eval_path, columns, all_rows[train_count:])

    today = datetime.date.today().isoformat()
    manifest = build_synthetic_manifest(
        name=f"{program.project}-synthetic-{args.seed}",
        seed=args.seed,
        mode=args.mode,
        rows=rows_requested,
        datasets=[
            DatasetManifestEntry(
                role="train",
                source=train_path.name,
                version=today,
                fingerprint=dataset_fingerprint(train_path),
                sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
                rows=train_count,
            ),
            DatasetManifestEntry(
                role="evaluation",
                source=eval_path.name,
                version=today,
                fingerprint=dataset_fingerprint(eval_path),
                sha256=hashlib.sha256(eval_path.read_bytes()).hexdigest(),
                rows=eval_count,
            ),
        ],
    )

    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = {
        "project": program.project,
        "seed": args.seed,
        "mode": args.mode,
        "rows": rows_requested,
        "train_rows": train_count,
        "eval_rows": eval_count,
        "train_csv": str(train_path),
        "eval_csv": str(eval_path),
        "manifest": str(manifest_path),
        "origin": "synthetic",
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Project:  {program.project}")
        print(f"Seed:     {args.seed}  mode: {args.mode}  rows: {rows_requested}")
        print(f"Train:    {train_path}  ({train_count} rows)")
        print(f"Eval:     {eval_path}  ({eval_count} rows)")
        print(f"Manifest: {manifest_path}")

    return 0


def _write_rows_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in columns})


def _cmd_refine(args) -> int:
    # --- Resolve prompt ---------------------------------------------------
    raw_parts = args.prompt
    if len(raw_parts) == 1 and raw_parts[0] == "-":
        prompt_text = sys.stdin.read()
    else:
        prompt_text = " ".join(raw_parts)
    prompt_text = prompt_text.strip()
    if not prompt_text:
        print("refine: prompt is empty", file=sys.stderr)
        return 2

    # --- Resolve mode from inputs -----------------------------------------
    has_audit = bool(args.audit)
    has_eval = bool(args.evaluation)
    if has_audit and has_eval:
        print("refine: --audit and --evaluation are mutually exclusive", file=sys.stderr)
        return 2
    if not has_audit and not has_eval:
        print("refine: one of --audit or --evaluation is required", file=sys.stderr)
        return 2
    mode = "audit_driven" if has_audit else "metric_driven"

    # --- Load audit or evaluation dict ------------------------------------
    audit: dict | None = None
    evaluation: dict | None = None
    try:
        if has_audit:
            audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        else:
            evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        src = args.audit or args.evaluation
        print(f"refine: cannot read {src}: {exc}", file=sys.stderr)
        return 2

    # --- Optional .mxai context -------------------------------------------
    mxai_text: str | None = None
    if args.mxai:
        try:
            mxai_text = Path(args.mxai).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"refine: cannot read --mxai {args.mxai}: {exc}", file=sys.stderr)
            return 2

    # --- Optional prior chain ---------------------------------------------
    prior_chain: list[str] | None = None
    if args.chain:
        try:
            prior_chain = json.loads(Path(args.chain).read_text(encoding="utf-8"))
            if not isinstance(prior_chain, list):
                raise ValueError("chain file must contain a JSON array")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"refine: cannot read --chain {args.chain}: {exc}", file=sys.stderr)
            return 2

    # --- Run refinement ---------------------------------------------------
    try:
        proposal = RefinementAgent().refine(
            prompt_text,
            audit=audit,
            evaluation=evaluation,
            mxai=mxai_text,
            hints=args.hints or [],
            mode=mode,
            iteration_count=args.iteration,
            refinement_chain=prior_chain,
            parent_prompt_hash=args.parent_hash,
            max_iterations=args.max_iterations,
        )
    except IterationLimitReached as exc:
        print(f"refine: {exc}", file=sys.stderr)
        return 2
    except (ValueError, NotImplementedError) as exc:
        print(f"refine: {exc}", file=sys.stderr)
        return 2

    # --- Output: JSON mode ------------------------------------------------
    if args.json:
        print(json.dumps(proposal.to_dict(), indent=2, ensure_ascii=False, default=str))
        return 0 if proposal.supervision_accepted else 1

    # --- Output: human-readable -------------------------------------------
    status = "ACCEPTED" if proposal.supervision_accepted else "REJECTED"
    print(f"Refinement ID : {proposal.refinement_id}")
    print(f"Mode          : {proposal.mode}")
    print(f"Iteration     : {proposal.iteration_count}")
    print(f"Supervision   : {status}")
    print(f"Chain length  : {len(proposal.refinement_chain)}")
    print(f"Parent hash   : {proposal.parent_prompt_hash[:16]}...")
    print()
    print("--- Proposed prompt ---")
    print(proposal.proposed_prompt)
    print()
    print("--- Explanation ---")
    print(proposal.explanation)

    if not proposal.supervision_accepted:
        print("\nWARNING: proposal was rejected by PromptSupervisor.", file=sys.stderr)

    if not args.accept:
        print(
            "\nNo output files written. Re-run with --accept to write output files.",
            file=sys.stderr,
        )
        return 0 if proposal.supervision_accepted else 1

    # --- Write output files (only with --accept) --------------------------
    if args.accept and not (args.output or args.mxai_output or args.chain_output):
        print("WARNING: --accept was provided, but no output destinations (--output, --mxai-output, --chain-output) were specified.", file=sys.stderr)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(proposal.proposed_prompt, encoding="utf-8")
        print(f"Proposed prompt written to: {out_path}")

    if args.mxai_output:
        mxai_out = proposal.supervision_report.get("mxai", "")
        if mxai_out:
            mx_path = Path(args.mxai_output)
            mx_path.parent.mkdir(parents=True, exist_ok=True)
            mx_path.write_text(mxai_out, encoding="utf-8")
            print(f"Generated .mxai written to: {args.mxai_output}")
        else:
            print("WARNING: no .mxai generated (supervision may have rejected early)", file=sys.stderr)

    if args.chain_output:
        chain_path = Path(args.chain_output)
        chain_path.parent.mkdir(parents=True, exist_ok=True)
        chain_path.write_text(
            json.dumps(proposal.refinement_chain, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Refinement chain written to: {args.chain_output}")

    return 0 if proposal.supervision_accepted else 1


# ── P20 action CLI commands ───────────────────────────────────────────────────

def _load_input_json(input_arg: str | None) -> dict:
    if not input_arg:
        return {}
    if input_arg == "-":
        return json.loads(sys.stdin.read())
    return json.loads(input_arg)


def _cmd_validate_actions(args) -> int:
    from matrixai.actions import parse_mxact, validate_action_contract

    try:
        source = Path(args.contract).read_text(encoding="utf-8")
        contracts = parse_mxact(source)
    except Exception as exc:
        print(f"validate-actions: failed to parse contract: {exc}", file=sys.stderr)
        return 1

    program = parse_file(args.mxai)
    all_ok = True
    results = []
    for contract in contracts:
        result = validate_action_contract(contract, program)
        results.append({
            "contract": contract.name,
            "ok": result.ok,
            "errors": result.errors,
        })
        if not result.ok:
            all_ok = False

    if args.json:
        print(json.dumps({"contracts": results, "all_ok": all_ok}, indent=2))
    else:
        for r in results:
            status = "OK" if r["ok"] else "FAIL"
            print(f"  [{status}] {r['contract']}")
            for err in r["errors"]:
                print(f"         {err}")
        print(f"\n{'All contracts valid.' if all_ok else 'Validation errors found.'}")

    return 0 if all_ok else 1


def _cmd_dry_run_action(args) -> int:
    from matrixai.actions import DryRunSimulator, parse_mxact

    try:
        source = Path(args.contract).read_text(encoding="utf-8")
        contracts = parse_mxact(source)
    except Exception as exc:
        print(f"dry-run-action: failed to parse contract: {exc}", file=sys.stderr)
        return 1

    matches = [c for c in contracts if c.name == args.contract_name]
    if not matches:
        names = [c.name for c in contracts]
        print(f"dry-run-action: contract {args.contract_name!r} not found. Available: {names}",
              file=sys.stderr)
        return 1
    contract = matches[0]

    program = parse_file(args.mxai)

    from matrixai.actions import validate_action_contract as _validate_contract
    validation = _validate_contract(contract, program)
    if not validation.ok:
        print(f"dry-run-action: contract validation failed — {validation.errors}",
              file=sys.stderr)
        return 1

    input_data = _load_input_json(args.input_json)

    sim = DryRunSimulator()
    report = sim.simulate(
        contract, program,
        args.param_set, args.model_hash,
        input_data,
    )

    if args.json:
        print(json.dumps({
            "report_id": report.report_id,
            "ok": report.ok,
            "errors": report.errors,
            "scope_ok": report.scope_ok,
            "rate_limit_ok": report.rate_limit_ok,
            "input_types_ok": report.input_types_ok,
            "rollback_ok": report.rollback_ok,
            "action_contract_hash": report.action_contract_hash,
            "input_hash": report.input_hash,
            "executed_at": report.executed_at,
            "valid_until": report.valid_until,
        }, indent=2))
    else:
        status = "OK" if report.ok else "FAIL"
        print(f"DryRunReport [{status}]  contract: {contract.name}")
        print(f"  report_id:  {report.report_id}")
        print(f"  hash:       {report.action_contract_hash}")
        print(f"  valid_until:{report.valid_until}")
        if report.errors:
            for err in report.errors:
                print(f"  ERROR: {err}")

    return 0 if report.ok else 1


def _cmd_execute_action(args) -> int:
    import os
    from matrixai.actions import (
        ActionExecutor, DryRunSimulator, ExecutionContext,
        SandboxedActionExecutor, parse_mxact, validate_action_contract,
    )
    from matrixai.actions.dryrun import RateTracker
    from matrixai.actions.schema import HIGH_RISK_CAPABILITIES
    from matrixai.actions.trace import build_action_trace

    try:
        source = Path(args.contract).read_text(encoding="utf-8")
        contracts = parse_mxact(source)
    except Exception as exc:
        print(f"execute-action: failed to parse contract: {exc}", file=sys.stderr)
        return 1

    matches = [c for c in contracts if c.name == args.contract_name]
    if not matches:
        names = [c.name for c in contracts]
        print(f"execute-action: contract {args.contract_name!r} not found. Available: {names}",
              file=sys.stderr)
        return 1
    contract = matches[0]

    program = parse_file(args.mxai)

    validation = validate_action_contract(contract, program)
    if not validation.ok:
        print(f"execute-action: contract validation failed — {validation.errors}",
              file=sys.stderr)
        return 1

    if not args.param_set:
        print("execute-action: --param-set is required for real action execution",
              file=sys.stderr)
        return 1
    param_set_path = Path(args.param_set)
    if param_set_path.suffix in (".json", ".params") and not param_set_path.exists():
        print(f"execute-action: param-set file not found: {args.param_set}", file=sys.stderr)
        return 1
    # Default identity from CLI args; overridden by certified ParameterSet when available.
    # Fall back to "cli" when --model-hash is not supplied and no file-based PS is loaded.
    effective_model_hash = args.model_hash or "cli"
    effective_param_set_id = args.param_set

    if param_set_path.suffix in (".json", ".params") and param_set_path.exists():
        from matrixai.parameters import load_parameter_set, validate_parameter_set
        try:
            parameter_set = load_parameter_set(param_set_path)
            compat = validate_parameter_set(program, parameter_set)
            if not compat.ok:
                print(f"execute-action: parameter set validation failed — {compat.errors}",
                      file=sys.stderr)
                return 1
        except Exception as exc:
            print(f"execute-action: failed to load param-set: {exc}", file=sys.stderr)
            return 1
        # Derive model_hash and parameter_set_id from the certified artifact.
        effective_model_hash = parameter_set.model_hash
        effective_param_set_id = parameter_set.parameter_set_id
        if args.model_hash and args.model_hash != effective_model_hash:
            print(
                f"execute-action: --model-hash {args.model_hash!r} mismatches "
                f"ParameterSet model_hash {effective_model_hash!r}",
                file=sys.stderr,
            )
            return 1

    input_data = _load_input_json(args.input_json)

    signing_key = args.signing_key or os.environ.get("MATRIXAI_ACTION_SIGNING_KEY")
    rate_tracker = RateTracker()

    sim = DryRunSimulator()
    report = sim.simulate(contract, program, effective_param_set_id, effective_model_hash,
                          input_data, rate_tracker=rate_tracker)
    if not report.ok:
        print(f"execute-action: dry-run failed — {report.errors}", file=sys.stderr)
        return 1

    ctx = ExecutionContext(
        contract=contract,
        dry_run_report=report,
        input_data=input_data,
        model_hash=effective_model_hash,
        parameter_set_id=effective_param_set_id,
        allow_real_actions=args.allow_real_actions,
        signing_key=signing_key,
    )

    try:
        if contract.capability in HIGH_RISK_CAPABILITIES:
            executor = SandboxedActionExecutor()
        else:
            executor = ActionExecutor()
        result = executor.execute(ctx)
    except Exception as exc:
        print(f"execute-action: {exc}", file=sys.stderr)
        return 1

    trace = build_action_trace(ctx, result, signing_key=signing_key)

    if args.json:
        print(json.dumps({
            "report_id": trace.report_id,
            "model_hash": trace.model_hash,
            "parameter_set_id": trace.parameter_set_id,
            "action_contract_hash": trace.action_contract_hash,
            "input_hash": trace.input_hash,
            "executed_at": trace.executed_at,
            "executor_kind": trace.executor_kind,
            "ok": trace.ok,
            "response_summary": trace.response_summary,
            "error": trace.error,
            "latency_ms": trace.latency_ms,
            "hmac_signature": trace.hmac_signature,
        }, indent=2))
    else:
        status = "OK" if result.ok else "FAIL"
        print(f"ActionResult [{status}]  executor: {result.executor_kind}")
        if result.response_summary:
            print(f"  response: {result.response_summary}")
        if result.error:
            print(f"  error:    {result.error}")
        print(f"  latency:  {result.latency_ms:.1f} ms")
        print(f"  trace_id: {trace.report_id}")
        if trace.hmac_signature:
            print(f"  hmac:     {trace.hmac_signature[:32]}...")

    return 0 if result.ok else 1


def _cmd_audit_action(args) -> int:
    import os
    from matrixai.actions.trace import ActionTrace, verify_action_trace

    try:
        data = json.loads(Path(args.trace_file).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"audit-action: failed to read trace: {exc}", file=sys.stderr)
        return 1

    trace = ActionTrace(
        report_id=data.get("report_id", ""),
        model_hash=data.get("model_hash", ""),
        parameter_set_id=data.get("parameter_set_id", ""),
        action_contract_hash=data.get("action_contract_hash", ""),
        input_hash=data.get("input_hash", ""),
        executed_at=data.get("executed_at", ""),
        executor_kind=data.get("executor_kind", ""),
        ok=data.get("ok", False),
        response_summary=data.get("response_summary", ""),
        error=data.get("error"),
        latency_ms=data.get("latency_ms", 0.0),
        hmac_signature=data.get("hmac_signature"),
    )

    signing_key = args.signing_key or os.environ.get("MATRIXAI_ACTION_SIGNING_KEY")

    if not signing_key:
        verified = None
        note = "no signing key — signature not verified"
    else:
        verified = verify_action_trace(trace, signing_key)
        note = "signature valid" if verified else "SIGNATURE MISMATCH"

    if args.json:
        print(json.dumps({
            "report_id": trace.report_id,
            "ok": trace.ok,
            "executor_kind": trace.executor_kind,
            "executed_at": trace.executed_at,
            "hmac_signature": trace.hmac_signature,
            "signature_verified": verified,
            "note": note,
        }, indent=2))
    else:
        print(f"ActionTrace  report_id: {trace.report_id}")
        print(f"  executor:  {trace.executor_kind}   ok: {trace.ok}")
        print(f"  executed:  {trace.executed_at}")
        print(f"  signature: {note}")

    if verified is False:
        return 1
    return 0


def _cmd_registry(args) -> int:
    from matrixai.registry import (
        DuplicateEntryError,
        EntryNotFoundError,
        ModelRegistry,
        ModelRegistryError,
        VerificationError,
    )

    registry_path = getattr(args, "registry_path", "matrixai_registry")
    reg = ModelRegistry(registry_path)

    sub = args.registry_command

    if sub == "list":
        filters = {"name": args.name} if args.name else None
        entries = reg.list(filters=filters)
        if args.json:
            print(json.dumps([e.to_manifest() for e in entries], indent=2))
        else:
            if not entries:
                print("(no entries)")
            for e in entries:
                print(f"{e.name}@{e.version}  interp={e.interpretability_level}  hash={e.entry_hash[:20]}...")
        return 0

    if sub == "show":
        name, version = _parse_entry_ref(args.entry)
        try:
            entry = reg.get(name, version)
        except EntryNotFoundError as exc:
            print(f"registry show: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(entry.to_manifest(), indent=2))
        else:
            m = entry.to_manifest()
            for k, v in m.items():
                print(f"  {k}: {v}")
        return 0

    if sub == "tag":
        name, version = _parse_entry_ref(args.entry)
        try:
            reg.tag(name, version, args.tag_name)
            print(f"Tagged {name}@{version} as {args.tag_name!r}")
        except EntryNotFoundError as exc:
            print(f"registry tag: {exc}", file=sys.stderr)
            return 1
        return 0

    if sub == "verify":
        name, version = _parse_entry_ref(args.entry)
        try:
            reg.verify(name, version)
            print(f"OK: {name}@{version} integrity verified; compatibility policy checked")
        except (VerificationError, EntryNotFoundError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        return 0

    if sub == "pull":
        name, version = _parse_entry_ref(args.entry)
        src = ModelRegistry(args.src_registry)
        dst = ModelRegistry(args.dst_registry)
        try:
            src.pull(name, version, dst)
            print(f"Pulled {name}@{version} → {args.dst_registry}")
        except (EntryNotFoundError, DuplicateEntryError, ModelRegistryError) as exc:
            print(f"registry pull: {exc}", file=sys.stderr)
            return 1
        return 0

    if sub == "diff":
        name_a, ver_a = _parse_entry_ref(args.entry_a)
        name_b, ver_b = _parse_entry_ref(args.entry_b)
        try:
            a = reg.get(name_a, ver_a)
            b = reg.get(name_b, ver_b)
        except EntryNotFoundError as exc:
            print(f"registry diff: {exc}", file=sys.stderr)
            return 1
        ma, mb = a.to_manifest(), b.to_manifest()
        changed = False
        for key in sorted(set(ma) | set(mb)):
            va, vb = ma.get(key), mb.get(key)
            if va != vb:
                print(f"  {key}:")
                print(f"    - {va}")
                print(f"    + {vb}")
                changed = True
        if not changed:
            print("(no differences)")
        return 0

    if sub == "push":
        try:
            entry = reg.push_run_dir(
                args.source_dir, args.name, args.version_tag,
            )
            print(f"Pushed {entry.name}@{entry.version}  entry_hash={entry.entry_hash[:20]}...")
        except (ModelRegistryError, DuplicateEntryError) as exc:
            print(f"registry push: {exc}", file=sys.stderr)
            return 1
        return 0

    print(f"registry: unknown subcommand {sub!r}", file=sys.stderr)
    return 1


def _cmd_continual(args) -> int:
    import dataclasses
    from matrixai.continual import parse_mxcontinual, MxcontinualParseError
    from matrixai.continual.collector import CollectorError, ProductionDataCollector
    from matrixai.actions.trace import ActionTrace

    sub = args.continual_command

    # ── ingest ────────────────────────────────────────────────────────────────
    if sub == "ingest":
        policy_text = Path(args.policy).read_text(encoding="utf-8")
        try:
            policy = parse_mxcontinual(policy_text)
        except MxcontinualParseError as exc:
            print(f"continual ingest: policy parse error: {exc}", file=sys.stderr)
            return 2

        signing_key = args.signing_key or os.environ.get("MATRIXAI_CONTINUAL_SIGNING_KEY")
        collector = ProductionDataCollector(policy, signing_key=signing_key)

        if args.trace_file:
            try:
                trace_dict = json.loads(Path(args.trace_file).read_text())
                trace = ActionTrace(**{
                    f.name: trace_dict[f.name]
                    for f in dataclasses.fields(ActionTrace)
                    if f.name in trace_dict
                })
                collector.register_trace(trace)
            except (KeyError, TypeError, OSError) as exc:
                print(f"continual ingest: cannot load trace: {exc}", file=sys.stderr)
                return 2

        try:
            sample = collector.ingest_by_id(args.trace_id, args.label)
        except CollectorError as exc:
            print(f"continual ingest: {exc}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(dataclasses.asdict(sample), indent=2))
        else:
            print(f"Ingested sample {sample.sample_id} for trace {sample.trace_id!r}")
            print(f"  ground_truth : {sample.ground_truth}")
            print(f"  source       : {sample.source}")
            print(f"  signed       : {sample.signed}")
        return 0

    # ── init ──────────────────────────────────────────────────────────────────
    if sub == "init":
        policy_text = Path(args.policy).read_text(encoding="utf-8")
        try:
            policy = parse_mxcontinual(policy_text)
        except MxcontinualParseError as exc:
            print(f"continual init: policy parse error: {exc}", file=sys.stderr)
            return 2

        from matrixai.continual import canonical_dict
        from matrixai.continual.policy_view import build_continual_policy_view

        view = build_continual_policy_view(policy)
        if args.json:
            print(json.dumps(dataclasses.asdict(view), indent=2))
        else:
            print(f"Policy       : {view.name}")
            print(f"Target model : {view.target_model}")
            print(f"Registry     : {view.registry_name or '(none)'}")
            print(f"Base version : {view.base_version or '(none)'}")
            print(f"Policy hash  : {view.policy_hash}")
            print(f"Rollback     : auto={view.rollback_auto_trigger}  "
                  f"metric={view.rollback_metric}  "
                  f"threshold={view.rollback_degradation_threshold}")
            print(f"Audit        : emit_hint={view.audit_emit_refinement_hint}  "
                  f"signature_required={view.audit_signature_required}")
        return 0

    # ── status ────────────────────────────────────────────────────────────────
    if sub == "status":
        from matrixai.registry import ModelRegistry
        from matrixai.registry.model_registry import EntryNotFoundError

        policy_text = Path(args.policy).read_text(encoding="utf-8")
        try:
            policy = parse_mxcontinual(policy_text)
        except MxcontinualParseError as exc:
            print(f"continual status: policy parse error: {exc}", file=sys.stderr)
            return 2

        registry_name = policy.registry_name
        if not registry_name:
            print("continual status: policy has no REGISTRY_NAME", file=sys.stderr)
            return 1

        reg = ModelRegistry(args.registry_dir)
        try:
            entry = reg.get(registry_name, "current")
        except EntryNotFoundError:
            print(f"continual status: no 'current' entry for {registry_name!r}", file=sys.stderr)
            return 1

        # Load persisted rollback event if available
        last_rollback: dict | None = None
        event_path = Path(args.registry_dir) / f".{registry_name}_last_rollback.json"
        if event_path.exists():
            try:
                last_rollback = json.loads(event_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass

        # Try to fetch the base version entry for comparison
        base_entry = None
        base_version = getattr(policy, "base_version", None)
        if base_version:
            try:
                base_entry = reg.get(registry_name, base_version)
            except Exception:  # noqa: BLE001
                pass

        rollback = getattr(policy, "rollback", None)
        rollback_cfg = {
            "degradation_threshold": getattr(rollback, "degradation_threshold", None),
            "sliding_window_hours": getattr(rollback, "sliding_window_hours", None),
            "min_samples_in_window": getattr(rollback, "min_samples_in_window", None),
        } if rollback else {}

        info = {
            "registry_name": entry.name,
            "current_version": entry.version,
            "current_parameter_set_id": entry.parameter_set_id,
            "entry_hash": entry.entry_hash,
            "metrics": entry.metrics,
            "base_version": base_version,
            "base_parameter_set_id": base_entry.parameter_set_id if base_entry else None,
            "rollback_config": rollback_cfg,
            "last_rollback": last_rollback,
            "drift_status": "see GET /metrics (requires server running with --continual-policy)",
        }
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"Registry       : {entry.name}")
            print(f"Current version: {entry.version}  (ps={entry.parameter_set_id})")
            if base_entry:
                print(f"Base version   : {base_version}  (ps={base_entry.parameter_set_id})")
            elif base_version:
                print(f"Base version   : {base_version}  (not found in registry)")
            print(f"Entry hash     : {entry.entry_hash[:20]}...")
            if rollback_cfg.get("degradation_threshold") is not None:
                print(f"Rollback config: threshold={rollback_cfg['degradation_threshold']}  "
                      f"window={rollback_cfg['sliding_window_hours']}h  "
                      f"min_samples={rollback_cfg['min_samples_in_window']}")
            if entry.metrics:
                print("Metrics (training):")
                for k, v in entry.metrics.items():
                    print(f"  {k}: {v}")
            if last_rollback:
                print(f"Last rollback  : {last_rollback.get('from_version', '?')} → "
                      f"{last_rollback.get('to_version', '?')}  "
                      f"({last_rollback.get('trigger_reason', '?')})")
                executed_at = last_rollback.get("rolled_back_at", "")
                if executed_at:
                    print(f"  executed_at  : {executed_at}")
                rollback_id = last_rollback.get("rollback_id", "")
                if rollback_id:
                    print(f"  event        : {rollback_id[:24]}...")
            else:
                print("Last rollback  : none recorded")
            print("Drift status   : see GET /metrics (server must run with --continual-policy)")
        return 0

    # ── promote ───────────────────────────────────────────────────────────────
    if sub == "promote":
        from matrixai.registry import ModelRegistry
        from matrixai.continual.versioning import ContinualVersioner, ContinualVersioningError
        from matrixai.continual.approval import (
            ApprovalGateReport, HoldoutMetrics, RegressionGuardResult, PendingApproval,
        )
        from matrixai.parameters.store import ParameterSet

        policy_text = Path(args.policy).read_text(encoding="utf-8")
        try:
            policy = parse_mxcontinual(policy_text)
        except MxcontinualParseError as exc:
            print(f"continual promote: policy parse error: {exc}", file=sys.stderr)
            return 2

        try:
            report_dict = json.loads(Path(args.approval_report).read_text())
            report = _approval_report_from_dict(report_dict)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"continual promote: cannot load approval report: {exc}", file=sys.stderr)
            return 2

        signing_key = getattr(args, "signing_key", None)
        if signing_key and report.pending_approval is not None:
            token = report.pending_approval.approval_token
            if token.startswith("hmac-sha256:"):
                from matrixai.continual.approval import _make_approval_token
                expected = _make_approval_token(
                    report.policy_hash,
                    report.candidate_parameter_set_id,
                    report.pending_approval.created_at,
                    signing_key,
                    expires_at=report.pending_approval.expires_at,
                )
                if expected != token:
                    print("continual promote: approval token HMAC verification failed — "
                          "report may have been tampered with", file=sys.stderr)
                    return 1

        if getattr(args, "human_approved", False) and report.pending_approval is not None:
            from matrixai.continual.approval import approve_pending_approval
            approver = getattr(args, "approved_by", None) or os.environ.get("USER") or "cli"
            try:
                approved = approve_pending_approval(
                    report.pending_approval,
                    decided_by=approver,
                    signing_key=signing_key,
                )
            except ValueError as exc:
                print(f"continual promote: {exc}", file=sys.stderr)
                return 2
            report = dataclasses.replace(report, pending_approval=approved)

        try:
            ps_dict = json.loads(Path(args.candidate_params).read_text())
            candidate = ParameterSet.from_dict(ps_dict)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"continual promote: cannot load candidate params: {exc}", file=sys.stderr)
            return 2

        reg = ModelRegistry(args.registry_dir)
        try:
            result = ContinualVersioner(
                policy, reg, report, candidate,
                continual_update_id=args.update_id,
                approval_signing_key=signing_key,
            ).promote(human_approved=getattr(args, "human_approved", False))
        except ContinualVersioningError as exc:
            print(f"continual promote: {exc}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(dataclasses.asdict(result), indent=2))
        else:
            print(f"Promoted {result.registry_name}@{result.new_version}")
            print(f"  previous version : {result.previous_version}")
            print(f"  entry hash       : {result.entry_hash[:20]}...")
            print(f"  update id        : {result.continual_update_id}")
        return 0

    # ── rollback ──────────────────────────────────────────────────────────────
    if sub == "rollback":
        from matrixai.registry import ModelRegistry
        from matrixai.registry.model_registry import EntryNotFoundError
        from matrixai.continual.rollback import RollbackManager, _find_version_by_ps_id

        policy_text = Path(args.policy).read_text(encoding="utf-8")
        try:
            policy = parse_mxcontinual(policy_text)
        except MxcontinualParseError as exc:
            print(f"continual rollback: policy parse error: {exc}", file=sys.stderr)
            return 2

        registry_name = policy.registry_name
        if not registry_name:
            print("continual rollback: policy has no REGISTRY_NAME", file=sys.stderr)
            return 1

        reg = ModelRegistry(args.registry_dir)
        try:
            from_entry = reg.get(registry_name, "current")
        except EntryNotFoundError:
            print(f"continual rollback: no 'current' entry for {registry_name!r}", file=sys.stderr)
            return 1

        parent_ps_id = from_entry.metrics.get("parent_parameter_set_id", "")
        if not parent_ps_id:
            print("continual rollback: current entry has no parent_parameter_set_id", file=sys.stderr)
            return 1

        to_version = _find_version_by_ps_id(reg, registry_name, parent_ps_id)
        if to_version is None:
            print(f"continual rollback: parent parameter set {parent_ps_id!r} not found in registry",
                  file=sys.stderr)
            return 1

        if args.dry_run:
            info = {
                "dry_run": True,
                "from_version": from_entry.version,
                "to_version": to_version,
                "from_parameter_set_id": from_entry.parameter_set_id,
                "to_parameter_set_id": parent_ps_id,
            }
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                print(f"[dry-run] Would rollback {registry_name}")
                print(f"  from : {from_entry.version}  (ps={from_entry.parameter_set_id})")
                print(f"  to   : {to_version}  (ps={parent_ps_id})")
            return 0

        signing_key = getattr(args, "signing_key", None) or os.environ.get("MATRIXAI_CONTINUAL_SIGNING_KEY")
        # Use a minimal monitor (no live data — manual rollback)
        from matrixai.continual.monitor import ProductionMonitor
        monitor = ProductionMonitor(policy, reference_accuracy=None, labels=[])
        manager = RollbackManager(policy, monitor, reg, signing_key=signing_key)
        event = manager.execute(
            from_parameter_set_id=from_entry.parameter_set_id,
            to_parameter_set_id=parent_ps_id,
            from_version=from_entry.version,
            to_version=to_version,
            trigger_reason="manual",
        )
        # Persist event so `continual status` can display it
        event_dict = dataclasses.asdict(event)
        event_path = Path(args.registry_dir) / f".{registry_name}_last_rollback.json"
        try:
            event_path.write_text(json.dumps(event_dict, indent=2), encoding="utf-8")
        except OSError:
            pass  # non-fatal — rollback succeeded even if we can't persist the event

        if args.json:
            print(json.dumps(event_dict, indent=2))
        else:
            print(f"Rolled back {registry_name}")
            print(f"  from : {event.from_version}  (ps={event.from_parameter_set_id})")
            print(f"  to   : {event.to_version}  (ps={event.to_parameter_set_id})")
            print(f"  event: {event.rollback_id}")
            print(f"  sig  : {event.signature}")
            print(f"  saved: {event_path}")
        return 0

    # ── audit ─────────────────────────────────────────────────────────────────
    if sub == "audit":
        policy_text = Path(args.policy).read_text(encoding="utf-8")
        try:
            policy = parse_mxcontinual(policy_text)
        except MxcontinualParseError as exc:
            print(f"continual audit: policy parse error: {exc}", file=sys.stderr)
            return 2

        audit = policy.audit
        info: dict = {
            "persist_drift_reports": audit.persist_drift_reports,
            "persist_update_traces": audit.persist_update_traces,
            "emit_refinement_hint_on_sustained_drift": audit.emit_refinement_hint_on_sustained_drift,
            "refinement_drift_persistence_days": audit.refinement_drift_persistence_days,
            "signature_required": audit.signature_required,
            "refinement_proposal": None,
        }

        if args.drift_report:
            try:
                dr_dict = json.loads(Path(args.drift_report).read_text())
                drift_report = _drift_report_from_dict(dr_dict)
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"continual audit: cannot load drift report: {exc}", file=sys.stderr)
                return 2

            prompt = getattr(args, "prompt", None) or ""
            if audit.emit_refinement_hint_on_sustained_drift and drift_report.drift_detected and prompt:
                from matrixai.continual.refinement_bridge import DriftRefinementBridge
                persistence_days = getattr(args, "drift_persistence_days", None)
                proposal = DriftRefinementBridge(policy, prompt=prompt).maybe_refine(
                    drift_report, drift_persistence_days=persistence_days,
                )
                if proposal is not None:
                    info["refinement_proposal"] = proposal.to_dict()

        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"persist_drift_reports              : {audit.persist_drift_reports}")
            print(f"persist_update_traces              : {audit.persist_update_traces}")
            print(f"emit_refinement_hint               : {audit.emit_refinement_hint_on_sustained_drift}")
            print(f"refinement_drift_persistence_days  : {audit.refinement_drift_persistence_days}")
            print(f"signature_required                 : {audit.signature_required}")
            if info["refinement_proposal"]:
                print("refinement_proposal                : (present)")
            else:
                print("refinement_proposal                : (none)")
        return 0

    print(f"continual: unknown subcommand {sub!r}", file=sys.stderr)
    return 1


def _handle_keys(args) -> int:
    from matrixai.signing.keystore import KeyStore

    _ENV_FOR_PURPOSE = {
        "action": "MATRIXAI_ACTION_SIGNING_KEY",
        "registry": "MATRIXAI_REGISTRY_SIGNING_KEY",
    }

    sub = getattr(args, "keys_command", None)
    if not sub:
        print("Usage: matrixai keys <rotate|list>", file=sys.stderr)
        return 1

    history_path = (
        Path(args.history_path) if args.history_path
        else KeyStore.default_path(Path(args.registry_path))
    )

    if sub == "rotate":
        purpose = args.purpose
        env_var = _ENV_FOR_PURPOSE[purpose]
        key = args.key_value or os.environ.get(env_var)
        if not key:
            print(
                f"Error: no key provided. Set {env_var} or pass --key.",
                file=sys.stderr,
            )
            return 1

        store = KeyStore.load(history_path)
        fp = store.retire(key, purpose)
        print(f"Key retired and recorded in {history_path}")
        print(f"  Purpose    : {purpose}")
        print(f"  Fingerprint: {fp}")
        print()
        print("Next steps:")
        print(f"  1. Generate new key  : openssl rand -hex 32")
        print(f"  2. Set env var       : export {env_var}=<new-key>")
        print(f"  3. Restart server    : docker compose restart  (or equivalent)")
        print()
        print("Historical verification is preserved: old signatures can still be")
        print(f"verified with: matrixai audit-action --signing-key <old-key>")
        print(f"or automatically via: matrixai keys list (shows all stored keys)")
        return 0

    if sub == "list":
        store = KeyStore.load(history_path)
        entries = store.list_entries()
        if getattr(args, "json", False):
            print(json.dumps([e.to_dict() for e in entries], indent=2))
            return 0
        if not entries:
            print(f"No keys recorded in {history_path}")
            return 0
        print(f"Key history: {history_path}")
        print(f"{'Fingerprint':<22}  {'Purpose':<10}  {'Status':<10}  {'Added':<27}  Rotated")
        print("-" * 100)
        for e in entries:
            status = "active" if e.is_active else "retired"
            rotated = e.rotated_at or "-"
            print(f"{e.fingerprint:<22}  {e.purpose:<10}  {status:<10}  {e.added_at:<27}  {rotated}")
        return 0

    print(f"keys: unknown subcommand {sub!r}", file=sys.stderr)
    return 1


def _approval_report_from_dict(d: dict) -> "Any":
    """Reconstruct ApprovalGateReport from a dataclasses.asdict() dict."""
    from matrixai.continual.approval import (
        ApprovalGateReport, HoldoutMetrics, RegressionGuardResult, PendingApproval,
    )

    def _hm(h: dict) -> HoldoutMetrics:
        return HoldoutMetrics(
            loss=h["loss"], accuracy=h["accuracy"], macro_f1=h["macro_f1"],
            macro_precision=h["macro_precision"], macro_recall=h["macro_recall"],
            per_label=h.get("per_label", {}), samples=h["samples"],
        )

    def _rg(r: dict) -> RegressionGuardResult:
        return RegressionGuardResult(
            passed=r["passed"], metric=r["metric"],
            baseline_value=r["baseline_value"], candidate_value=r["candidate_value"],
            must_improve_by=r["must_improve_by"], actual_delta=r["actual_delta"],
            per_label_violations=r.get("per_label_violations", {}),
            reasons=r.get("reasons", []),
        )

    def _pa(p: dict | None) -> "PendingApproval | None":
        if p is None:
            return None
        return PendingApproval(
            approval_id=p["approval_id"], policy_hash=p["policy_hash"],
            candidate_parameter_set_id=p["candidate_parameter_set_id"],
            parent_parameter_set_id=p["parent_parameter_set_id"],
            created_at=p["created_at"], expires_at=p.get("expires_at"),
            approval_token=p["approval_token"], channel=p.get("channel", ""),
            status=p.get("status", "pending"),
            decided_at=p.get("decided_at"),
            decided_by=p.get("decided_by"),
            decision_token=p.get("decision_token"),
        )

    return ApprovalGateReport(
        policy_hash=d["policy_hash"],
        status=d["status"],
        candidate_parameter_set_id=d["candidate_parameter_set_id"],
        baseline_parameter_set_id=d["baseline_parameter_set_id"],
        holdout_samples=d["holdout_samples"],
        baseline_metrics=_hm(d["baseline_metrics"]),
        candidate_metrics=_hm(d["candidate_metrics"]),
        regression_guard=_rg(d["regression_guard"]),
        pending_approval=_pa(d.get("pending_approval")),
        evaluated_at=d["evaluated_at"],
        rejection_reasons=d.get("rejection_reasons", []),
    )


def _drift_report_from_dict(d: dict) -> "Any":
    """Reconstruct DriftReport from a dataclasses.asdict() dict."""
    from matrixai.continual.drift import DriftReport, FeatureDriftResult
    results = {
        feat: FeatureDriftResult(
            feature=r["feature"], method=r["method"],
            observed_value=r["observed_value"], threshold=r["threshold"],
            drift_detected=r["drift_detected"], samples_used=r["samples_used"],
            enough_samples=r["enough_samples"], skipped=r.get("skipped", False),
            skip_reason=r.get("skip_reason"),
        )
        for feat, r in d.get("results", {}).items()
    }
    return DriftReport(
        policy_hash=d["policy_hash"],
        checked_at=d["checked_at"],
        features_checked=d["features_checked"],
        results=results,
        drift_detected=d["drift_detected"],
        enough_samples=d["enough_samples"],
        total_production_samples=d["total_production_samples"],
    )


def _parse_entry_ref(ref: str) -> tuple[str, str]:
    """Parse 'name@version' → (name, version)."""
    if "@" not in ref:
        raise SystemExit(f"Invalid entry reference {ref!r}: expected name@version")
    name, version = ref.split("@", 1)
    return name, version


if __name__ == "__main__":
    raise SystemExit(main())
