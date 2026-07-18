# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Stable public API of the MatrixAI playground engine.

This module is the supported surface for building a playground or a product on top
of the MatrixAI core. It re-exports, with stable public names, the request handlers
and helpers that today live (with private ``_`` names) inside :mod:`matrixai.playground`.

Why it exists
-------------
The technical playground (open source, served at ``/expert``) and any product built
on top of the core share the same engine: prompt analysis, dataset generation,
CSV validation, training (sync + async jobs), schema coercion and LLM helpers.
Importing those through this module — instead of reaching into ``matrixai.playground``
private names — gives downstream code (e.g. a separate Studio backend) a stable
contract that survives internal refactors of ``playground.py``.

Guarantee
---------
Everything here is part of the AGPL-3.0 core. Names listed in ``__all__`` are stable;
the underlying private functions in ``playground.py`` may be renamed or moved without
breaking importers of this module.
"""

from __future__ import annotations

from matrixai.training.dataset_analysis import (
    analyze_dataset_csv,
    DatasetAnalysisError,
)
from matrixai.training.dataset_project import (
    generate_project_from_dataset,
    generate_temporal_project_from_dataset,
    DatasetProjectError,
    _force_temporal_split as force_temporal_split,
    _extract_seed as extract_seed,
    _read_rows as read_csv_rows,
    _rows_to_csv_text as rows_to_csv_text,
)
from matrixai.training.data_provider import (
    get_default_registry,
    get_default_acceptance_store,
    DataProviderError,
    LicenseAcceptance,
)
from matrixai.training.dataset_pipeline import (
    # BIBLIOTECA_PROYECTOS_INTELIGENTES C6 (reauditoría): la orquestación de
    # plantillas se movió a matrixai_studio (decisión de alcance 1 del
    # contrato — "Biblioteca... va a studio-backend/SPA, nada en el core
    # público") pero sigue necesitando el motor de pipeline C3 tal cual —
    # reexportado aquí para que studio-backend nunca importe
    # matrixai.training.* directamente (mismo criterio que el resto de este
    # módulo).
    run_pipeline,
    check_anti_leakage,
    validate_pipeline_output,
    PipelineError,
)
from matrixai.playground import (
    # Already-public engine entry points
    analyze_playground_request,
    serve,
    # Shared model/visual view used by both the technical playground and products
    _visual_model as visual_model,
    # Shared module state: repo root (bundled examples) and the training job
    # registry. A product building on the core must use the SAME registry object
    # (not a copy) so async jobs submitted via submit_training_job are visible.
    PROJECT_ROOT,
    _training_jobs as training_jobs,
    # HTTP handler factory (build a server reusing the core router)
    _handler_class as handler_class,
    # Training-text + dataset generation
    _generate_training_from_mxai as generate_training_from_mxai,
    _generate_synthetic_dataset as generate_synthetic_dataset,
    _suggest_field_ranges as suggest_field_ranges,
    _validate_training_csv as validate_training_csv,
    _csv_template as csv_template,
    # Training — synchronous and async jobs (shared job registry lives in the core)
    _run_playground_training as run_playground_training,
    _submit_training_job as submit_training_job,
    _get_job_status as get_job_status,
    _cancel_job as cancel_job,
    # Execution / refinement
    _playground_run_with_params as playground_run_with_params,
    _refine_prompt as refine_prompt,
    # Field schema coercion (ranges / types / categoricals / identifiers)
    _coerce_field_ranges as coerce_field_ranges,
    _coerce_field_types as coerce_field_types,
    _coerce_field_categories as coerce_field_categories,
    _coerce_field_identifiers as coerce_field_identifiers,
    _normalize_csv_with_ranges as normalize_csv_with_ranges,
    # Pipeline view helpers
    _build_pipeline_stages as build_pipeline_stages,
    _build_artifacts as build_artifacts,
    # LLM-assisted helpers
    _dense_llm_schema as dense_llm_schema,
    _llm_field_ranges as llm_field_ranges,
    _resolve_llm_config_path as resolve_llm_config_path,
    _detect_llm_mode as detect_llm_mode,
    # Misc utilities
    _safe_float as safe_float,
)

__all__ = [
    # BIBLIOTECA_PROYECTOS_INTELIGENTES C1/C2/C4 — flujo datos-primero
    "analyze_dataset_csv",
    "DatasetAnalysisError",
    "generate_project_from_dataset",
    "generate_temporal_project_from_dataset",
    "DatasetProjectError",
    "get_default_registry",
    "get_default_acceptance_store",
    "DataProviderError",
    "LicenseAcceptance",
    "force_temporal_split",
    "extract_seed",
    "read_csv_rows",
    "rows_to_csv_text",
    "run_pipeline",
    "check_anti_leakage",
    "validate_pipeline_output",
    "PipelineError",
    "analyze_playground_request",
    "serve",
    "handler_class",
    "visual_model",
    "PROJECT_ROOT",
    "training_jobs",
    "generate_training_from_mxai",
    "generate_synthetic_dataset",
    "suggest_field_ranges",
    "validate_training_csv",
    "csv_template",
    "run_playground_training",
    "submit_training_job",
    "get_job_status",
    "cancel_job",
    "playground_run_with_params",
    "refine_prompt",
    "coerce_field_ranges",
    "coerce_field_types",
    "coerce_field_categories",
    "coerce_field_identifiers",
    "normalize_csv_with_ranges",
    "build_pipeline_stages",
    "build_artifacts",
    "dense_llm_schema",
    "llm_field_ranges",
    "resolve_llm_config_path",
    "detect_llm_mode",
    "safe_float",
]
