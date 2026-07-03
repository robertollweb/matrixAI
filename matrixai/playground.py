# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import base64
import csv
import io
import json
from datetime import datetime, timezone
import os
import re
import shutil
import tempfile
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from matrixai.agents import AuditorAgent, IterationLimitReached, ChatCompletionsLLMProposalProvider, PromptSupervisor, RefinementAgent, SafetyAgent, VerifierAgent
from matrixai.compiler import BackendContractAnalyzer, PythonBackendCompiler
from matrixai.parameters.store import ParameterSet
from matrixai.parser import parse_text
from matrixai.runtime import MatrixAIRuntime
from matrixai.sandbox import SandboxPolicy
from matrixai.tooling import diagnose_runtime_compiler, graph_program
from matrixai.training.dataset_manifest import DatasetManifest, verify_dataset_manifest
from matrixai.training.generator import TrainingPromptGenerator
from matrixai.training.parser import parse_training_text
from matrixai.training.dense_generator import _ONEHOT_MAX
from matrixai.training.dense_trainer import DenseSupervisedEvaluator, DenseSupervisedTrainer
from matrixai.training.trainer import GenericSupervisedEvaluator, GenericSupervisedTrainer, SupervisedEvaluator, SupervisedTrainer
from matrixai.training.verifier import TrainingVerifier
from matrixai.types import check_program_types


DEFAULT_PROMPT = (
    "Crear un sistema que clasifique correos entrantes y prepare una respuesta "
    "solo si la confianza supera el 95%"
)

DEFAULT_INPUT = {
    "Email": {
        "urgency": 0.84,
        "sender_trust": 0.96,
        "topic_support": 0.99,
        "topic_sales": 0.04,
        "sentiment": 0.72,
        "has_attachment": 0.0,
        "previous_interactions": 0.88,
        "language_confidence": 0.97,
    }
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "email-agent": {
        "label": "Email agent",
        "model": "examples/email-agent.typed.mxai",
        "training": "examples/email-agent.supervised.mxtrain",
        "input": "examples/email-sample.json",
        "manifest": "",
        "evaluation": "examples/email-agent.evaluation_report.json",
    },
    "fall-risk": {
        "label": "Fall risk",
        "model": "examples/fall-risk.typed.mxai",
        "training": "examples/fall-risk.supervised.mxtrain",
        "input": "examples/fall-risk-sample.json",
        "manifest": "examples/fall-risk.dataset-manifest.json",
        "evaluation": "examples/fall-risk.evaluation_report.json",
    },
    "celsius-to-kelvin": {
        "label": "Celsius to Kelvin",
        "model": "examples/celsius_to_kelvin.mxai",
        "training": "examples/celsius_to_kelvin.mxtrain",
        "input": "",
        "manifest": "",
        "evaluation": "",
    },
    "transformer-classifier": {
        "label": "Transformer classifier",
        "model": "examples/transformer-classifier-vector.mxai",
        "training": "examples/transformer-classifier-vector.mxtrain",
        "input": "examples/transformer-classifier-sample.json",
        "manifest": "",
        "evaluation": "",
    },
    "transformer-seq": {
        "label": "Transformer sequence classifier",
        "model": "examples/transformer-classifier.mxai",
        "training": "examples/transformer-classifier.mxtrain",
        "input": "examples/transformer-classifier-sample.json",
        "manifest": "",
        "evaluation": "",
    },
}
PROJECT_EXAMPLE_INDEX = {
    "EmailAgentTyped": "email-agent",
    "FallRiskAgentTyped": "fall-risk",
    "CelsiusToKelvin": "celsius-to-kelvin",
    "transformer_classifier": "transformer-classifier",
    "transformer_seq": "transformer-seq",
}


_SUPERVISION_STAGES = [
    ("architect_plan", "Architect Plan"),
    ("planner_verifier", "Planner Verifier"),
    ("mathematical_rules_resolved", "Reglas Matemáticas"),
    ("parser", "Parser"),
    ("verifier_agent", "Verifier Agent"),
    ("safety_agent", "Safety Agent"),
    ("python_compiler", "Python Compiler"),
]

# Keywords that signal the user explicitly wants a dense neural network.
_NEURAL_INTENT_KEYWORDS = [
    # Spanish
    "neuronal", "red neuronal", "red densa", "red profunda",
    "capas ocultas", "capas densas", "arquitectura neuronal",
    # English
    "neural network", "neural net", "dense network", "deep learning",
    "hidden layer", "dense layer", "neural architecture",
    # Common abbreviations
    "mlp", "feedforward", "feed-forward",
]

# Common predictive/classification task verbs that imply a learned model.
# These expand the router beyond explicit neural vocabulary so that natural
# business prompts ("detectar reingresos", "clasificar tickets") also produce
# a real network architecture.
_NEURAL_TASK_KEYWORDS = [
    # Spanish
    "detectar", "clasificar", "clasificacion", "predecir", "prediccion",
    "estimar", "estimacion", "identificar", "pronosticar", "pronostico",
    "evaluar riesgo", "calcular riesgo", "riesgo de", "probabilidad de",
    "analizar", "score", "puntaje",
    # English
    "detect", "classify", "classification", "predict", "prediction",
    "estimate", "estimation", "identify", "forecast",
    "risk of", "probability of", "risk score",
]

# Workflow/rules prompts: these should stay in PromptSupervisor even if they
# also contain task verbs from _NEURAL_TASK_KEYWORDS.
_WORKFLOW_KEYWORDS = [
    "workflow", "flujo de trabajo", "motor de reglas", "reglas de negocio",
    "automatizar", "automatizacion", "automatización",
    "trigger", "pipeline de decisiones", "orquestar", "orchestrat",
    "si entonces", "if then", "cuando ocurra", "on event",
]

_MXAI_STAGES = [
    ("parser", "Parser"),
    ("verifier_agent", "Verifier Agent"),
    ("safety_agent", "Safety Agent"),
    ("typecheck", "Type Check"),
    ("python_compiler", "Python Compiler"),
]

# Pipeline stages for the DenseNetworkGenerator path.
# Makes the neural routing explicit: prompt was classified, network generated,
# then verified by the same agents as the mxai path.
_DENSE_STAGES = [
    ("prompt_router",     "Prompt Router"),
    ("dense_generator",   "Dense Network Generator"),
    ("parser",            "Parser"),
    ("verifier_agent",    "Verifier Agent"),
    ("safety_agent",      "Safety Agent"),
    ("typecheck",         "Type Check"),
    ("backend_contract",  "Backend Contract"),
    ("python_compiler",   "Python Compiler"),
    ("training_verifier", "Training Verifier"),
]


def _dense_pipeline_stages(result: dict[str, Any], generator_name: str = "dense_generator") -> list[dict[str, Any]]:
    """Build pipeline stages for the network generator path (dense or composite).

    - Adds synthetic Prompt Router and Network Generator stages (ok). The
      generator stage name is "dense_generator" or, for M2 composites,
      "composite_generator".
    - Removes the interpretability warning from Verifier Agent — it is
      expected for dense networks and already reported by Type Check.
      This avoids showing the same notice twice.
    - Synthesizes Backend Contract and Training Verifier from artifacts.
    """
    # Synthetic stages: routing decision + network generation both succeeded
    router_check = {"name": "prompt_router", "ok": True, "errors": [], "warnings": []}
    dng_check    = {"name": generator_name, "ok": True, "errors": [], "warnings": []}

    # Existing verifier/parser/safety/typecheck/compiler checks
    filtered: list[dict[str, Any]] = []
    for c in result.get("checks", []):
        if not isinstance(c, dict):
            continue
        if c.get("name") == "verifier_agent":
            # Remove interpretability notice — it is already in Type Check
            c = dict(c)
            c["warnings"] = [
                w for w in (c.get("warnings") or [])
                if "interpretability_level" not in str(w)
            ]
        filtered.append(c)

    # Backend Contract — synthesized from artifacts
    bc = result.get("backend_contract") or {}
    bc_unsupported = [
        f"{n.get('node', '?')}: {n.get('reason', 'unsupported')}"
        for n in (bc.get("unsupported_nodes") or [])
    ]
    bc_check = {
        "name": "backend_contract",
        "ok": bool(bc.get("ok", True)) and not bc_unsupported,
        "errors": bc_unsupported,
        "warnings": [],
    }

    # Training Verifier — synthesized from training_artifacts.verification
    ta = result.get("training_artifacts") or {}
    tv = ta.get("verification") or {}
    tv_warnings = list(tv.get("warnings") or []) + [
        w for w in (ta.get("warnings") or [])
        if w not in list(tv.get("warnings") or [])
    ]
    tv_check = {
        "name": "training_verifier",
        "ok": bool(tv.get("ok", ta.get("ok", True))),
        "errors": list(tv.get("errors") or []),
        "warnings": tv_warnings,
    }

    all_checks = [router_check, dng_check] + filtered + [bc_check, tv_check]
    # M2-C3: the generator stage name varies (dense vs composite); swap it into
    # the fixed stage order so it isn't filtered out.
    gen_label = "Composite Network Generator" if generator_name == "composite_generator" else "Dense Network Generator"
    stage_order = [
        (generator_name, gen_label) if name == "dense_generator" else (name, label)
        for name, label in _DENSE_STAGES
    ]
    return _build_pipeline_stages(all_checks, stage_order)


def _build_pipeline_stages(
    checks: list[dict[str, Any]],
    stage_order: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    checks_by_name = {c["name"]: c for c in checks if isinstance(c, dict)}
    stages: list[dict[str, Any]] = []
    failed = False
    for name, label in stage_order:
        if name in checks_by_name:
            check = checks_by_name[name]
            ok = bool(check.get("ok"))
            errors = list(check.get("errors") or [])
            warnings = list(check.get("warnings") or [])
            if not ok:
                status = "fail"
                failed = True
            elif warnings:
                status = "warning"
            else:
                status = "ok"
        elif failed:
            status = "skipped"
            errors, warnings = [], []
        else:
            status = "pending"
            errors, warnings = [], []
        stages.append({"name": name, "label": label, "status": status, "errors": errors, "warnings": warnings})
    return stages


def _build_artifacts(
    mode: str,
    source: str,
    semantic_text: str,
    mxai_text: str,
    compiled_python: str,
    supervision_source: str,
    llm_provider: str,
    llm_model: str,
) -> dict[str, Any]:
    if supervision_source == "llm" and llm_provider:
        semantic_source = f"LLM:{llm_provider}:{llm_model}"
    elif mode == "prompt":
        semantic_source = "PromptAgent"
    elif mode == "semantic":
        semantic_source = "playground:semantic"
    else:
        semantic_source = source or mode

    mxai_source = f"derived_from_{mode}" if mode in ("prompt", "semantic") else "playground:mxai"

    result: dict[str, Any] = {}
    if semantic_text:
        result["semantic"] = {"text": semantic_text, "source": semantic_source}
    if mxai_text:
        result["mxai"] = {"text": mxai_text, "source": mxai_source}
    if compiled_python:
        result["compiled_python"] = {"text": compiled_python, "source": "compiled"}
    return result


# P9 operational limits — M12: ahora configurables en runtime (hosted/env/perfil) vía
# `matrixai.limits`. Estas constantes son los DEFAULTS (perfil "equilibrado") y se
# conservan para compatibilidad/visualización; el código de runtime llama a
# `limits.cap()/exceeds()/get_limit()` para respetar la configuración del usuario.
from matrixai import limits as _limits  # noqa: E402
_P9_MAX_CSV_BYTES = _limits._EQUILIBRADO["max_csv_bytes"]
_P9_MAX_ROWS = _limits._EQUILIBRADO["max_rows"]
_P9_MAX_EPOCHS = _limits._EQUILIBRADO["max_epochs"]
# Wall-clock training budget, configurable via MATRIXAI_TRAIN_TIMEOUT.
# Default 300s protects a SHARED hosted playground. **Set to 0 to disable** (no
# limit) — the downloadable Studio sets 0 because the machine is the user's and a
# large model may legitimately train for hours/days; the Cancel button is the
# real user control. <=0 means "train to completion".
_P9_TRAIN_TIMEOUT = int(os.environ.get("MATRIXAI_TRAIN_TIMEOUT", "300"))


def _train_join_timeout() -> float | None:
    """None (block until done) when the wall-clock budget is disabled (<=0)."""
    return _P9_TRAIN_TIMEOUT if _P9_TRAIN_TIMEOUT and _P9_TRAIN_TIMEOUT > 0 else None

# P9 async job store
class _TrainingCancelled(Exception):
    pass

_training_jobs: dict[str, dict[str, Any]] = {}
_MAX_JOBS = 20


def _diag(msg: str) -> None:
    """Log de diagnóstico gateado por MATRIXAI_DEBUG.

    - Stdout: solo cuando MATRIXAI_DEBUG=1 (evita ruido en usos de librería).
    - Fichero: siempre (silencioso; útil para debug post-hoc con `!tail` en Colab).
    En Colab el fichero cae en /content/matrixai_diag.log (o MATRIXAI_DIAG_LOG).
    Best-effort: nunca falla.
    """
    line = f"[matrixai] {msg}"
    if os.environ.get("MATRIXAI_DEBUG") == "1":
        print(line, flush=True)
    try:
        import time as _t
        path = os.environ.get("MATRIXAI_DIAG_LOG") or (
            "/content/matrixai_diag.log" if os.path.isdir("/content") else "/tmp/matrixai_diag.log"
        )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{_t.strftime('%H:%M:%S')} {line}\n")
    except Exception:  # noqa: BLE001
        pass


def _mxai_project_stem(mxai_text: str) -> str:
    m = re.search(r"^\s*PROJECT\s+(\S+)", mxai_text, re.MULTILINE)
    return m.group(1).lower() if m else "model"


def _is_celsius_kelvin_program(program: Any) -> bool:
    project = str(getattr(program, "project", "")).lower()
    fields = {field.lower() for vector in getattr(program, "vectors", []) for field in vector.fields}
    outputs = {function.output.lower() for function in getattr(program, "functions", [])}
    if project == "celsiustokelvin":
        return True
    return "celsius" in fields and any("kelvin" in output for output in outputs)


def _example_training_package(example_id: str) -> dict[str, Any]:
    config = EXAMPLES.get(example_id)
    if not config or not config.get("training"):
        return {"ok": False, "error": f"No training example registered for {example_id}"}
    training_text = (PROJECT_ROOT / config["training"]).read_text(encoding="utf-8")
    dataset_template_text = _dataset_text_from_training_text(training_text)
    return {
        "ok": True,
        "training_text": training_text,
        "dataset_template_text": dataset_template_text,
        "warnings": [],
        "source": "example",
    }


def _dataset_text_from_training_text(training_text: str) -> str:
    try:
        training = parse_training_text(training_text)
    except Exception:  # noqa: BLE001
        return ""
    candidates = [PROJECT_ROOT / training.dataset.source, PROJECT_ROOT / Path(training.dataset.source).name]
    for path in candidates:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    return ""


def _get_prediction_kind(mxai_text: str, training_text: str) -> str:
    """Return semantic.kind of the prediction function defined by the training loss, or '' on error.

    Matches by output ref first (e.g. PREDICTION logits), then by function name (e.g. PREDICTION Logits),
    mirroring the same dual lookup that TrainingVerifier._prediction_function uses.
    """
    try:
        program = parse_text(mxai_text)
        training = parse_training_text(training_text)
        prediction = training.loss.prediction
        fn = next(
            (f for f in program.functions if f.output == prediction),
            None,
        ) or next(
            (f for f in program.functions if f.name == prediction),
            None,
        )
        if fn is not None:
            return fn.semantic.kind
        # No matching FUNCTION — check for NETWORK blocks (dense model)
        if program.networks:
            return "network_call"
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _generate_training_from_mxai(mxai_text: str, prompt: str = "") -> dict[str, Any]:
    if not mxai_text.strip():
        return {"ok": False, "error": "mxai_text es obligatorio"}
    stem = _mxai_project_stem(mxai_text)
    try:
        program = parse_text(mxai_text)
        if _is_celsius_kelvin_program(program):
            package = _example_training_package("celsius-to-kelvin")
            if package.get("ok"):
                package["warnings"] = [
                    *list(package.get("warnings") or []),
                    "Celsius/Kelvin usa el contrato canónico P17 con datos celsius -> predicted_kelvin.",
                ]
                return package

        clean_prompt = " ".join((prompt or f"supervisado para {stem}").strip().split())
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mxai_path = tmp / f"{stem}.mxai"
            mxai_path.write_text(mxai_text, encoding="utf-8")
            dataset_path = tmp / f"{stem}.train.csv"
            gen = TrainingPromptGenerator().generate(
                prompt=clean_prompt,
                model_path=mxai_path,
                dataset_source=dataset_path.name,
                model_reference=mxai_path.name,
            )
            return {
                "ok": True,
                "training_text": gen.training_text,
                "dataset_template_text": gen.dataset_template_text,
                "warnings": list(gen.warnings),
                "source": "generated",
            }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _generate_synthetic_dataset(
    mxai_text: str,
    training_text: str,
    rows: int,
    seed: int,
    mode: str,
    use_llm: bool = False,
    field_ranges_override: dict[str, tuple[float, float]] | None = None,
    field_types: dict[str, str] | None = None,
    field_categories: dict[str, list[str]] | None = None,
    field_identifiers: list[str] | None = None,
) -> dict[str, Any]:
    if not mxai_text.strip() or not training_text.strip():
        return {"ok": False, "error": "mxai_text y training_text son obligatorios"}
    requested_rows = int(rows)
    rows = max(2, _limits.cap(requested_rows, "max_rows"))
    # Si el perfil de límites recortó las filas pedidas, avisamos (no silenciosamente):
    # el frontend ya no topa, así que el cap solo lo aplica el backend según el perfil.
    rows_capped_warning = ""
    if requested_rows > rows:
        _eff = _limits.get_limit("max_rows")
        rows_capped_warning = (
            f"Pediste {requested_rows} filas pero el perfil de límites las recortó a {_eff}. "
            f"Cambia el perfil a 'ilimitado' en Ajustes → Límites para generar más."
        )
    if mode not in ("random", "coherent"):
        mode = "random"
    # M9: aviso accionable ante .mxai incoherente (elipsis / dim≠campos / columnas≠VECTOR)
    # antes de que el parser falle con un error técnico opaco.
    from matrixai.training.coherence import check_dataset_coherence
    _coh = check_dataset_coherence(mxai_text, training_text)
    if not _coh.ok:
        return {"ok": False, "error": _coh.error_es, "error_en": _coh.error_en,
                "error_kind": "coherence"}
    try:
        from matrixai.parser import parse_text
        from matrixai.training.parser import parse_training_text
        from matrixai.training.synthetic import SyntheticDataGenerator
        from matrixai.training.categorical import expand_categoricals, exclude_identifiers
        import io

        # S2-C4: drop identifier columns (patient_id …) from model + training
        # first — they carry no signal and must never be features.
        mxai_text, training_text, excluded_ids = exclude_identifiers(
            mxai_text, training_text, field_identifiers,
        )
        # GEN C6: a categorical already materialized as an EMBEDDING (its column
        # is an embedding source — an Integer index in the VECTOR) must NOT be
        # one-hot expanded: the model consumes ONE index column, and expanding it
        # would corrupt the dataset (15 one-hot columns for a 1-column input).
        # Its human vocab still travels (echoed below) for inference/export.
        _pre = parse_text(mxai_text)
        _emb_sources: set[str] = {
            e.source for n in _pre.networks for e in getattr(n, "embeddings", [])
        }
        _expandable = {c: v for c, v in (field_categories or {}).items()
                       if c not in _emb_sources}
        # S2-C2: expand declared categoricals to one-hot BEFORE parsing, so the
        # whole pipeline (program, generator, returned model) sees the expanded
        # VECTOR/columns. With no categoricals this is a no-op.
        expansion = expand_categoricals(mxai_text, training_text, _expandable)
        mxai_text = expansion.mxai_text
        training_text = expansion.training_text
        one_hot_members = set(expansion.members.keys())
        model_changed = bool(expansion.groups) or bool(excluded_ids)

        program = parse_text(mxai_text)
        training = parse_training_text(training_text)

        # GEN C6: a model whose categoricals were expanded AT GENERATION TIME
        # (C2: prompt-declared one-hot) arrives with the expanded columns already
        # in the VECTOR and field_categories as metadata — expand_categoricals is
        # a no-op then (the raw column no longer exists). Reconstruct the one-hot
        # groups from the metadata so the sampler still emits exactly one 1 per
        # group (instead of independent random scalars) and the echoed
        # field_categories/one_hot_groups don't silently drop the metadata.
        one_hot_groups = dict(expansion.groups)
        if field_categories:
            from matrixai.training.categorical import _build_group_names  # noqa: PLC0415
            _cols = set(training.dataset.input.columns)
            for _cat, _values in field_categories.items():
                if _cat in one_hot_groups or _cat in _cols or not _values:
                    continue
                _members = _build_group_names(_cat, list(_values))
                if _members and all(m in _cols for m in _members):
                    one_hot_groups[_cat] = _members
                    one_hot_members.update(_members)

        # M6: precedence user > LLM > default. The LLM only fills the gaps the
        # user left open; if the override covers every input column it is never
        # called (no wasted latency on regenerate-with-edited-ranges).
        # S2-C2 / GEN C6: one-hot members are 0/1, never ranged/typed.
        # Embedding sources are integer lookup indices (e.g. 0..14), not domain
        # scalars; LLM/user ranges must never normalize or re-sample them.
        input_columns = list(training.dataset.input.columns)
        rangeable = [
            c for c in input_columns
            if c not in one_hot_members and c not in _emb_sources
        ]
        user_ranges = {
            col: rng for col, rng in (field_ranges_override or {}).items()
            if col in rangeable
        }
        field_ranges: dict[str, tuple[float, float]] = {}
        llm_ranges_used = False
        gaps = [col for col in rangeable if col not in user_ranges]
        if use_llm and gaps and _detect_llm_mode().get("active", False):
            context = re.search(r"^\s*PROJECT\s+(\S+)", mxai_text, re.MULTILINE)
            context_str = context.group(1) if context else ""
            llm_ranges = _llm_field_ranges(gaps, context_str)
            field_ranges = {col: rng for col, rng in llm_ranges.items() if col in gaps}
            llm_ranges_used = bool(field_ranges)
        field_ranges.update(user_ranges)

        # S2: declared types only apply to real (non one-hot) input columns
        types = {col: t for col, t in (field_types or {}).items() if col in rangeable}

        # M8 v2: LLM as domain simulator — propose feature→class threshold rules ONCE,
        # normalize to [0,1], and let the generator label rows deterministically with
        # plausible signal. Only for multiclass classification in coherent mode with the
        # LLM active; invalid/absent rules → fall back to the toy coherent labelling.
        domain_rules = None
        domain_rules_text = ""
        target_type = training.dataset.target.type
        dr_labels = list(target_type.parameters.get("labels")
                         or target_type.parameters.get("args", []))
        is_multiclass = (target_type.name not in {"Scalar", "Integer"}
                         and target_type.name.lower() != "probability"
                         and len(dr_labels) >= 2)
        if (mode == "coherent" and use_llm and is_multiclass
                and _detect_llm_mode().get("active", False)):
            project = re.search(r"^\s*PROJECT\s+(\S+)", mxai_text, re.MULTILINE)
            context = project.group(1) if project else ""
            dr = _llm_domain_rules(context, rangeable, dr_labels)
            if dr is not None and not dr.validate(rangeable, dr_labels):
                domain_rules = dr.normalized(field_ranges)  # eval in [0,1] space
                domain_rules_text = dr.to_text()             # domain scale, for audit

        # Opción A: la generación NUNCA instancia ni ejecuta el modelo para etiquetar.
        # Una red sin entrenar (init aleatorio) es un etiquetador arbitrario y, en redes
        # grandes, construirla y ejecutarla cuelga (p.ej. 24×8192 ≈ minutos/GB). Los
        # VALORES de las columnas se muestrean igual de los rangos (LLM/"Sugerir rangos");
        # las ETIQUETAS usan reglas de dominio del LLM si las hay, y si no, random. Así
        # 'coherent' sin reglas pasa a "valores realistas + etiquetas aleatorias" en vez
        # de etiquetas derivadas de un forward de la red.
        effective_mode = "random" if (mode == "coherent" and domain_rules is None) else mode

        generator = SyntheticDataGenerator(
            program=program,
            training=training,
            seed=seed,
            rows=rows,
            mode=effective_mode,
            field_ranges=field_ranges if field_ranges else None,
            # M5: human-readable CSV (salary 35000, age 72). Training normalizes
            # back at the boundary using the field_ranges returned below.
            domain_scale=True,
            field_types=types or None,
            one_hot_groups=one_hot_groups or None,
            domain_rules=domain_rules,
        )
        adapter = generator.generate()

        # M8 v2: if the domain rules collapsed to a single class on the sampled data
        # (bad thresholds / missing ranges / unhit branches), the signal is useless.
        # Fall back to coherent labelling honestly instead of shipping it as "domain".
        domain_degenerate_warning = ""
        if domain_rules is not None and generator.domain_rules_degenerate:
            domain_degenerate_warning = (
                "Las reglas de dominio propuestas por el LLM no discriminaron sobre los "
                "datos generados (casi todas las filas cayeron en una sola clase); se usaron "
                "etiquetas aleatorias. Ajusta el prompt o sube datos reales."
            )
            domain_rules = None
            domain_rules_text = ""
            # Opción A: sin reglas válidas no se ejecuta la red → etiquetas aleatorias.
            effective_mode = "random" if mode == "coherent" else mode
            generator = SyntheticDataGenerator(
                program=program,
                training=training,
                seed=seed,
                rows=rows,
                mode=effective_mode,
                field_ranges=field_ranges if field_ranges else None,
                domain_scale=True,
                field_types=types or None,
                one_hot_groups=one_hot_groups or None,
                domain_rules=None,
            )
            adapter = generator.generate()

        schema = adapter.schema()
        columns = list(schema.input_columns) + [schema.target]

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns)
        writer.writeheader()
        for row in adapter.rows:
            writer.writerow({col: row[col] for col in columns})
        csv_text = buf.getvalue()

        if _limits.exceeds(len(csv_text.encode()), "max_csv_bytes"):
            _lim = _limits.get_limit("max_csv_bytes")
            return {"ok": False, "error": f"Dataset sintetico supera el límite de {_lim // 1000} KB"}

        result: dict[str, Any] = {
            "ok": True,
            "csv_text": csv_text,
            "rows": rows,
            "seed": seed,
            # Echo the requested mode; `label_origin` below carries what the labels
            # actually are (synthetic_random when coherent degraded under option A).
            "mode": mode,
            "fingerprint": adapter.fingerprint(),
            "columns": columns,
            "labels": list(schema.labels),
            "origin": "synthetic",
            "llm_ranges_used": llm_ranges_used,
            # M5: domain ranges per column — the client must echo these to
            # /api/train-start (normalization)
            # (persistence for inference/export with the same scale).
            "field_ranges": {k: [v[0], v[1]] for k, v in field_ranges.items()},
            # S2: declared types actually applied to this dataset
            "field_types": types,
            # S2-C2: categoricals that were expanded to one-hot — here or at
            # generation time (GEN C6: pre-expanded models keep their metadata) —
            # plus EMBEDDING categoricals (their human vocab must keep travelling
            # to train/save/export, invariante 4). The client must use the
            # returned (expanded) model for train/save and render a single
            # selector per category at inference time.
            "field_categories": {col: list(vals) for col, vals in (field_categories or {}).items()
                                 if col in one_hot_groups or col in _emb_sources},
            "one_hot_groups": {col: list(members) for col, members in one_hot_groups.items()},
            # S2-C4: identifier columns dropped from the model (no predictive signal)
            "excluded_identifiers": excluded_ids,
        }
        if rows_capped_warning:
            result["rows_capped_warning"] = rows_capped_warning
        # M8-C1: be honest about what the labels represent. Synthetic labels are
        # learnable but a toy; random-mode classification has NO input→output
        # relationship (labels independent of features) → the model can only
        # predict the prior (collapse). Real signal needs an uploaded CSV.
        is_classification = bool(schema.labels)
        if generator.domain_rules_used:
            # M8 v2: labels followed LLM-proposed domain rules (plausible, not toy).
            result["label_origin"] = "synthetic_domain"
            result["domain_rules"] = domain_rules_text
            result["domain_notice"] = (
                "Etiquetas generadas por reglas de dominio propuestas por el LLM "
                "(sintéticas pero plausibles, no de juguete). Sube datos reales para "
                "producción."
            )
        else:
            result["label_origin"] = (
                "synthetic_random" if effective_mode == "random" else "synthetic_coherent"
            )
        if domain_degenerate_warning:
            result["domain_degenerate_warning"] = domain_degenerate_warning
        # M8 v2: declared classes absent from the generated data → the model can't
        # learn them and macro F1 will be low. Honest warning (not a block: ≥2 are
        # present; the single-class case is already handled by the degeneracy fallback).
        if is_classification and getattr(generator, "missing_labels", None):
            result["missing_labels"] = list(generator.missing_labels)
            _present = [l for l in schema.labels if l not in generator.missing_labels]
            result["missing_classes_warning"] = (
                f"El dataset solo contiene {len(_present)} de {len(schema.labels)} clases "
                f"(faltan: {', '.join(generator.missing_labels)}). El modelo no aprenderá "
                f"las clases ausentes y el F1 macro será bajo. Ajusta el prompt/reglas o "
                f"sube datos reales."
            )
        if is_classification and effective_mode == "random":
            result["signal_warning"] = (
                "Datos sintéticos aleatorios: la salida no depende de la entrada, "
                "así que el modelo no puede aprender nada (colapsará al predictor "
                "constante). Para señal real, activa las reglas de dominio del LLM "
                "o sube datos reales."
            )
        if model_changed:
            # The VECTOR/columns changed (one-hot and/or excluded ids) — the
            # client must train/save with this model.
            result["mxai_text"] = mxai_text
            result["training_text"] = training_text
        if generator.coherent_fallback_count > 0:
            result["coherent_fallback_count"] = generator.coherent_fallback_count
            result["coherent_fallback_warning"] = (
                f"{generator.coherent_fallback_count}/{rows} filas usaron labels aleatorias "
                f"porque el runtime no produjo una label válida en modo coherent."
            )
        return result
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _suggest_field_ranges(mxai_text: str, training_text: str) -> dict[str, Any]:
    """M6 — input columns + LLM-suggested domain ranges for the range editor.

    Returns the exact input column list (so the client never has to parse the
    CSV template) and a suggestion per column when the LLM is active. Columns
    without a suggestion are returned without entry: the editor shows them
    empty and generation falls back to [0,1].
    """
    if not mxai_text.strip() or not training_text.strip():
        return {"ok": False, "error": "mxai_text y training_text son obligatorios"}
    try:
        from matrixai.parser import parse_text
        from matrixai.training.parser import parse_training_text

        parse_text(mxai_text)
        training = parse_training_text(training_text)
        input_columns = list(training.dataset.input.columns)

        field_ranges: dict[str, tuple[float, float]] = {}
        llm_ranges_used = False
        if _detect_llm_mode().get("active", False):
            context = re.search(r"^\s*PROJECT\s+(\S+)", mxai_text, re.MULTILINE)
            context_str = context.group(1) if context else ""
            llm_ranges = _llm_field_ranges(input_columns, context_str)
            field_ranges = {col: rng for col, rng in llm_ranges.items() if col in input_columns}
            llm_ranges_used = bool(field_ranges)

        return {
            "ok": True,
            "columns": input_columns,
            "field_ranges": {k: [v[0], v[1]] for k, v in field_ranges.items()},
            "llm_ranges_used": llm_ranges_used,
            # S2: heuristic type pre-fill for the schema editor (user overrides)
            "field_types": _suggest_field_types(input_columns),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _expected_csv_columns(training: Any) -> tuple[list[str], str]:
    """M16 — columnas que un CSV subido debe CONTENER: features de INPUT FROM COLUMNS + la
    columna target (su nombre semántico del TARGET, p. ej. `estado`). Se leen por nombre:
    el orden no importa y las columnas extra se ignoran (solo se exige que estén presentes)."""
    input_columns = list(training.dataset.input.columns)
    target_column = training.dataset.target.name
    return input_columns, target_column


def _csv_template(mxai_text: str, training_text: str) -> dict[str, Any]:
    """M16 — plantilla exacta del CSV de entrenamiento esperado: nombres de columnas
    (features + target), clases (si clasificación) y un CSV de ejemplo descargable, para
    que el usuario sepa qué subir (nombres y formato) sin adivinar."""
    if not mxai_text.strip() or not training_text.strip():
        return {"ok": False, "error": "mxai_text y training_text son obligatorios"}
    try:
        from matrixai.parser import parse_text
        from matrixai.training.parser import parse_training_text
        from matrixai.training.dense_trainer import _labels_from_spec

        parse_text(mxai_text)
        training = parse_training_text(training_text)
        input_columns, target_column = _expected_csv_columns(training)
        labels = _labels_from_spec(training)
        columns = input_columns + [target_column]

        # 2 filas de ejemplo: inputs en 0.5 (mitad de [0,1]); target = primera clase, o
        # 0.0 si es regresión. Solo ilustra el FORMATO y los nombres de columna.
        example_target = labels[0] if labels else "0.0"
        example_row = ",".join(["0.5"] * len(input_columns) + [str(example_target)])
        template_csv = ",".join(columns) + "\n" + example_row + "\n" + example_row + "\n"

        return {
            "ok": True,
            "columns": columns,
            "input_columns": input_columns,
            "target_column": target_column,
            "labels": list(labels),
            "is_classification": bool(labels),
            "template_csv": template_csv,
            "note": (
                "El CSV debe contener estas columnas (mismos nombres); el orden no importa y "
                f"las columnas extra se ignoran. La columna objetivo es '{target_column}'. Los "
                "valores numéricos van en escala de dominio (los rangos guardados); las clases, "
                "como texto."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _validate_training_csv(
    mxai_text: str,
    training_text: str,
    csv_text: str,
    field_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    if not mxai_text.strip() or not training_text.strip() or not csv_text.strip():
        return {"ok": False, "error": "mxai_text, training_text y csv_text son obligatorios"}
    if _limits.exceeds(len(csv_text.encode()), "max_csv_bytes"):
        _lim = _limits.get_limit("max_csv_bytes")
        return {"ok": False, "error": f"CSV supera el límite de {_lim // 1000} KB"}
    # M5: domain-scale CSV must be normalized before the TrainingVerifier checks
    # values against the DSL type ranges (Scalar [0,1]).
    if field_ranges:
        csv_text = _normalize_csv_with_ranges(csv_text, field_ranges)
    try:
        training = parse_training_text(training_text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"training_text inválido: {exc}"}

    # M16 — chequeo de cabecera con mensaje accionable ANTES del verificador técnico:
    # si faltan columnas esperadas, decimos exactamente cuáles (y cuáles llegaron).
    input_columns, target_column = _expected_csv_columns(training)
    expected_columns = input_columns + [target_column]
    header_line = csv_text.splitlines()[0] if csv_text.strip() else ""
    found_columns = [c.strip() for c in header_line.split(",") if c.strip()]
    missing = [c for c in expected_columns if c not in found_columns]
    if missing:
        return {
            "ok": False,
            "expected_columns": expected_columns,
            "found_columns": found_columns,
            "missing_columns": missing,
            "target_column": target_column,
            "error": (
                f"El CSV no tiene las columnas esperadas. Faltan: {', '.join(missing)}. "
                f"Esperadas: {', '.join(expected_columns)} (la columna objetivo es "
                f"'{target_column}'). Encontradas: {', '.join(found_columns) or '(ninguna)'}. "
                f"Descarga la plantilla para ver el formato exacto."
            ),
            "error_en": (
                f"The CSV is missing expected columns. Missing: {', '.join(missing)}. "
                f"Expected: {', '.join(expected_columns)} (target column is '{target_column}'). "
                f"Found: {', '.join(found_columns) or '(none)'}. Download the template for the "
                f"exact format."
            ),
            "error_kind": "csv_columns",
        }

    # Columnas extra (presentes en el CSV pero no esperadas): no bloquean —se leen por
    # nombre, así que las extra se ignoran— pero avisamos para que sea transparente.
    extra_columns = [c for c in found_columns if c not in expected_columns]

    stem = _mxai_project_stem(mxai_text)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / f"{stem}.mxai").write_text(mxai_text, encoding="utf-8")
            # Write the mxai also under the name the spec references
            model_name = Path(training.model).name
            (tmp / model_name).write_text(mxai_text, encoding="utf-8")
            csv_name = Path(training.dataset.source).name
            csv_path = tmp / csv_name
            csv_path.write_text(csv_text, encoding="utf-8")
            report = TrainingVerifier().verify(training, base_path=tmp)
            if not report.ok:
                return {"ok": False, "errors": list(report.errors), "warnings": list(report.warnings)}
            # Count rows
            rows = 0
            with csv_path.open(newline="", encoding="utf-8") as fh:
                rows = sum(1 for _ in csv.reader(fh)) - 1  # subtract header
            if _limits.exceeds(rows, "max_rows"):
                return {"ok": False, "error": f"CSV tiene {rows} filas, máximo {_limits.get_limit('max_rows')}"}
            warnings_out = list(report.warnings)
            # M5-C3: uploaded CSV without ranges but with domain-scale values on
            # untyped fields — the verifier cannot catch it (no declared range)
            # and training would silently consume out-of-slider-space features.
            if not field_ranges:
                scale_warning = _detect_domain_scale_csv(csv_text, training)
                if scale_warning:
                    warnings_out.append(scale_warning)
            if extra_columns:
                warnings_out.append(
                    f"El CSV tiene columnas extra que se ignorarán: {', '.join(extra_columns)}."
                )
            return {"ok": True, "rows": rows, "warnings": warnings_out,
                    "expected_columns": expected_columns,
                    "extra_columns": extra_columns}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _detect_domain_scale_csv(csv_text: str, training: Any, sample_rows: int = 200) -> str | None:
    """M5-C3 — heuristic: input columns with values well outside [0,1] in a CSV
    that arrived WITHOUT field_ranges. Returns a human warning or None."""
    try:
        input_columns = set(training.dataset.input.columns)
    except Exception:  # noqa: BLE001
        return None
    suspects: set[str] = set()
    reader = csv.DictReader(io.StringIO(csv_text))
    for i, row in enumerate(reader):
        if i >= sample_rows:
            break
        for col in input_columns:
            try:
                value = float(row.get(col, ""))
            except (TypeError, ValueError):
                continue
            if value > 1.5 or value < -0.5:
                suspects.add(col)
    if not suspects:
        return None
    cols = ", ".join(sorted(suspects))
    return (
        f"Las columnas [{cols}] tienen valores fuera de [0, 1] y no se han "
        "proporcionado rangos: el entrenamiento las usará tal cual y la "
        "inferencia (sliders 0-1) no coincidirá con esa escala. Usa el dataset "
        "sintético generado, o normaliza el CSV a [0, 1]."
    )


def _apply_epoch_cap(training: Any, epochs_override: int | None) -> int | None:
    if epochs_override is not None:
        return _limits.cap(int(epochs_override), "max_epochs")
    if training.run and _limits.exceeds(training.run.epochs, "max_epochs"):
        return _limits.get_limit("max_epochs")
    return None


def _build_spec_with_epochs(training: Any, epochs_override: int | None) -> Any:
    if epochs_override is None:
        return training
    import dataclasses as _dc
    from matrixai.training.spec import RunSpec
    new_run = _dc.replace(training.run, epochs=epochs_override) if training.run else RunSpec(epochs=epochs_override)
    return _dc.replace(training, run=new_run)


def _collect_training_result(tmp: Path, run_result: Any, spec: Any) -> dict[str, Any]:
    """Read artifact files and run evaluation while temp dir is still alive."""
    rd = run_result.to_dict()
    artifacts = rd.get("artifacts", {})

    def _read_json(key: str) -> Any:
        path = artifacts.get(key)
        if path and Path(path).exists():
            try:
                return json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None
        return None

    metrics = _read_json("metrics") or {}
    # SupervisedTrainer writes "params_best"; DenseSupervisedTrainer writes "parameter_set"
    params_best_data = _read_json("params_best") or _read_json("parameter_set")
    training_trace = _read_json("training_trace")
    backend_raw = (training_trace or {}).get("backend_report", {}).get("target", "stdlib")
    backend = "torch" if backend_raw == "torch" else "stdlib"

    evaluation_report = None
    if params_best_data:
        try:
            from matrixai.parameters.store import ParameterSet as _PS
            ps = _PS.from_dict(params_best_data)
        except Exception:  # noqa: BLE001
            ps = None
        if ps is not None:
            # M3: the workflow evaluator can't handle NETWORK (dense) models —
            # fall back to the dense evaluator, which already computes
            # macro_f1/confusion_matrix (this was the gap: the metrics existed
            # but never reached the training result for Studio models).
            for evaluator in (SupervisedEvaluator(), DenseSupervisedEvaluator()):
                try:
                    evaluation_report = evaluator.evaluate(spec, ps, base_path=tmp).to_dict()
                    break
                except Exception:  # noqa: BLE001
                    continue

    task_kind = (training_trace or {}).get("task_kind", "classification")
    return {
        "ok": True,
        "task_kind": task_kind,
        "run_id": rd.get("run_id", ""),
        "best_epoch": rd.get("best_epoch", 0),
        "best_validation_loss": rd.get("best_validation_loss"),
        "final_train_loss": rd.get("final_train_loss"),
        "accuracy": rd.get("accuracy"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        # M3: classification metrics flattened for the client (also inside
        # evaluation_report; absent/None for regression)
        "macro_f1": (evaluation_report or {}).get("macro_f1"),
        "confusion_matrix": (evaluation_report or {}).get("confusion_matrix"),
        "labels": (evaluation_report or {}).get("labels"),
        "per_label": (evaluation_report or {}).get("per_label"),
        "backend": backend,
        "epochs": metrics.get("epochs", []),
        "params_best": params_best_data,
        "metrics": metrics,
        "training_trace": training_trace,
        "evaluation_report": evaluation_report,
    }


def _select_train_backend() -> tuple[bool, str]:
    """GPU-C3 — (use_torch, device) for Studio training.

    Policy via MATRIXAI_TRAIN_BACKEND: 'auto' (default) → torch on CUDA when
    available, else stdlib; 'torch' → force the torch path (cuda if present, else
    cpu — useful to test the torch path on a CPU box); 'stdlib' → never torch.
    The downloadable Studio 'just works': GPU accelerates, CPU-only falls back to
    stdlib with no config and no regression."""
    mode = os.environ.get("MATRIXAI_TRAIN_BACKEND", "auto").strip().lower()
    if mode == "stdlib":
        return (False, "cpu")
    try:
        from matrixai.parameters.tensor_bridge import torch_available
        if not torch_available():
            return (False, "cpu")
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return (False, "cpu")
    if mode == "torch":
        return (True, "cuda" if cuda else "cpu")
    return (cuda, "cuda" if cuda else "cpu")  # auto


def _eval_report_from_dense_result(dense_result: Any, labels: list[str] | None) -> dict[str, Any]:
    """M14 — construye el report (mismas claves que EvaluationResult.to_dict consumidas
    aguas abajo) a partir de un DenseEvaluationResult del evaluador torch/GPU."""
    per_label: dict[str, dict[str, float]] = {}
    if labels and dense_result.precision:
        for lbl in labels:
            per_label[lbl] = {
                "precision": dense_result.precision.get(lbl, 0.0),
                "recall": dense_result.recall.get(lbl, 0.0),
                "f1": dense_result.f1.get(lbl, 0.0),
            }
    return {
        "rows": dense_result.rows,
        "loss": dense_result.loss,
        "accuracy": dense_result.accuracy,
        "macro_f1": dense_result.macro_f1,
        "confusion_matrix": dense_result.confusion_matrix,
        "labels": list(labels or []),
        "per_label": per_label,
        "mae": dense_result.mae,
        "rmse": dense_result.rmse,
        "r2": dense_result.r2,
    }


def _dense_torch_train_result(
    mxai_text: str,
    training: Any,
    spec: Any,
    csv_text: str,
    device: str,
    seed: int,
    epoch_callback: Any,
    cancel_check: Any = None,
) -> dict[str, Any]:
    """GPU-C3 — train a dense_network with the torch trainer and build the SAME
    result shape as the stdlib path (so snapshot/infer/export/M3 metrics are
    unaffected). Evaluation uses evaluate_dense_network_torch (batched, chunked)."""
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.training.dense_trainer import (
        _labels_from_spec, _examples_to_xy,
    )
    from matrixai.training.data import CSVDataAdapter
    from matrixai.training.dense_torch_trainer import (
        train_dense_network_torch, evaluate_dense_network_torch,
    )

    program = parse_text(mxai_text)
    if not program.networks:
        return {"ok": False, "error": "El modelo no define ninguna NETWORK"}
    net = program.networks[0]
    vector_map = {v.name: v for v in program.vectors}
    vector = vector_map.get(net.input)
    if vector is None:
        return {"ok": False, "error": f"VECTOR de entrada {net.input!r} no encontrado"}
    type_result = check_network_types(net, vector_map)
    resolved_layers = type_result.resolved_layers if type_result.resolved_layers else net.layers

    loss_fn = training.loss.type if training.loss else "mse"
    lr = training.optimizer.learning_rate if training.optimizer else 0.01
    epochs = spec.run.epochs if spec.run else (training.run.epochs if training.run else 50)
    patience = spec.run.early_stop_patience if spec.run else None
    # Honor explicit BATCH size from the training spec (overrides the trainer default).
    batch_size: int | None = (
        training.dataset.batch.size if (training.dataset and training.dataset.batch) else None
    )
    labels = _labels_from_spec(training)
    epoch_trace: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / Path(training.model).name).write_text(mxai_text, encoding="utf-8")
        data_path = tmp / Path(training.dataset.source).name
        data_path.write_text(csv_text, encoding="utf-8")
        adapter = CSVDataAdapter(
            data_path, vector.name, list(vector.fields),
            training.dataset.target.name, labels if labels else None,
        )
        examples = _examples_to_xy(adapter.examples(), loss_fn, labels)
        if not examples:
            return {"ok": False, "error": "El dataset no produjo ejemplos válidos"}

        # M15(a): plantilla de estructura (sin pesos en Python); el módulo torch usa su
        # init nativo (Kaiming), sembrado por torch.manual_seed(seed) en el trainer.
        ps = build_network_parameter_set(net, resolved_layers, program_hash(program),
                                         seed=seed, with_values=False)

        def _torch_cb(e: dict[str, Any]) -> None:
            entry = {"epoch": e["epoch"], "train_loss": round(e["loss"], 6),
                     "validation_loss": round(e["val_loss"], 6), "accuracy": None}
            epoch_trace.append(entry)
            if epoch_callback is not None:
                epoch_callback(entry)  # may raise _TrainingCancelled

        # Log ANTES de entrenar: confirma backend (cuda/cpu) y que el preprocesado de
        # las filas terminó y arranca el bucle (si no aparece, está aún cargando el CSV).
        _diag(f"inicia entrenamiento: device={device}, ejemplos={len(examples)}, "
              f"epochs={epochs}, batch spec={batch_size}")
        tr = train_dense_network_torch(
            net, ps, examples, loss_fn, lr=lr, epochs=epochs,
            early_stop=(patience, "validation_loss") if patience else None,
            device=device, seed=seed, batch_size=batch_size,
            epoch_callback=_torch_cb, cancel_check=cancel_check,
        )
        # Visible en Colab: batch del spec (autogenerado = 8) vs el efectivo en GPU.
        _diag(f"entrenamiento OK en {device}: batch spec={batch_size} → efectivo={tr.get('batch_size')} "
              f"(ejemplos={len(examples)}, VRAM pico={tr.get('peak_vram_gb')} GB)")
        best_ps = tr["best_params"]
        # M14 — evaluación BATCHED en torch/GPU (no fila a fila en CPU). Reusa los
        # `examples` ya cargados; sin re-leer el CSV. Con datasets grandes + redes
        # anchas evita que el run se cuelgue evaluando en CPU tras entrenar en GPU.
        evaluation_warning: str | None = None
        try:
            dense_result = evaluate_dense_network_torch(
                net, best_ps, examples, loss_fn, labels=labels or None, device=device,
                cancel_check=cancel_check,
            )
            evaluation_report = _eval_report_from_dense_result(dense_result, labels)
            evaluation_backend = device
        except _TrainingCancelled:
            raise
        except Exception as _eval_exc:  # noqa: BLE001
            evaluation_report = None
            evaluation_backend = "failed"
            evaluation_warning = f"eval torch failed, metrics unavailable: {_eval_exc}"

    # M7 en GPU: prueba de colapso por torch (4 forwards instantáneos) en vez del runtime
    # Python de `_probe_model_collapse` (O(params)×4 → minutos y GBs de RAM con redes
    # grandes; era el "se queda pensando" al acabar). Se adjunta aquí para que
    # `_attach_collapse_check` detecte que ya está probado y NO repita la versión lenta.
    _collapse = None
    try:
        from matrixai.training.dense_torch_trainer import probe_collapse_torch
        _collapse = probe_collapse_torch(net, best_ps, len(vector.fields), device=device)
    except Exception:  # noqa: BLE001
        _collapse = None

    is_reg = loss_fn == "mse"
    er = evaluation_report or {}
    result = {
        "ok": True,
        "task_kind": "regression" if is_reg else "classification",
        "run_id": uuid.uuid4().hex[:8],
        "best_epoch": tr["best_epoch"],
        "best_validation_loss": tr["best_val_loss"],
        "final_train_loss": tr["train_loss"],
        "accuracy": None if is_reg else er.get("accuracy"),
        "macro_f1": er.get("macro_f1"),
        "confusion_matrix": er.get("confusion_matrix"),
        "labels": er.get("labels"),
        "per_label": er.get("per_label"),
        "mae": er.get("mae"),
        "rmse": er.get("rmse"),
        "r2": er.get("r2"),
        "backend": device,
        "evaluation_backend": evaluation_backend,
        "evaluation_warning": evaluation_warning,
        "effective_batch_size": tr.get("batch_size"),
        "epochs": epoch_trace,
        "params_best": best_ps.to_dict(),
        "metrics": {"epochs": epoch_trace},
        "training_trace": {"backend_report": {"target": device},
                           "task_kind": "regression" if is_reg else "classification"},
        "evaluation_report": evaluation_report,
    }
    if _collapse is not None:
        result["model_collapsed"] = bool(_collapse["collapsed"])
        if _collapse["collapsed"]:
            result["collapse_constant_output"] = _collapse.get("constant_output")
            result["collapse_warning"] = (
                "El modelo entrenado produce la misma salida para cualquier entrada "
                "(predictor constante). Suele deberse a una red demasiado profunda o a "
                "un cuello de botella ReLU antes de la capa de salida: simplifica la "
                "arquitectura y reentrena."
            )
    return result


def _run_playground_dense_training(
    mxai_text: str,
    training_text: str,
    csv_text: str,
    epochs_override: int | None = None,
    epoch_callback: Any = None,
    seed: int = 42,
    cancel_check: Any = None,
) -> dict[str, Any]:
    """Synchronous training for NETWORK (dense) models using DenseSupervisedTrainer."""
    try:
        training = parse_training_text(training_text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"training_text inválido: {exc}"}

    epochs_override = _apply_epoch_cap(training, epochs_override)
    spec = _build_spec_with_epochs(training, epochs_override)

    # GPU-C3: when a GPU/torch backend is selected, train with the torch trainer
    # (batched — this is where GPU helps); otherwise fall through to stdlib.
    use_torch, device = _select_train_backend()
    # Diagnóstico clave: por qué se usa (o no) la GPU. Si use_torch=False con una GPU
    # presente, el entrenamiento va por stdlib/CPU (lento, GPU ociosa). Reporta el motivo.
    _ta = _tcuda = "?"
    try:
        from matrixai.parameters.tensor_bridge import torch_available as _tav
        _ta = _tav()
        if _ta:
            import torch as _t
            _tcuda = bool(_t.cuda.is_available())
    except Exception as _e:  # noqa: BLE001
        _ta = f"error:{_e}"
    _diag(f"backend dense seleccionado: use_torch={use_torch}, device={device} | "
          f"MATRIXAI_TRAIN_BACKEND={os.environ.get('MATRIXAI_TRAIN_BACKEND','auto')}, "
          f"torch_available={_ta}, cuda_available={_tcuda}")
    if use_torch:
        try:
            return _dense_torch_train_result(mxai_text, training, spec, csv_text,
                                             device, seed, epoch_callback, cancel_check)
        except _TrainingCancelled:
            raise
        except Exception as exc:  # noqa: BLE001  — never let GPU break training: fall back
            import logging as _log
            _log.getLogger(__name__).warning("torch dense training failed, falling back to stdlib: %s", exc)
            # Visible en Colab: si esto aparece, la GPU NO se está usando (entrena en CPU,
            # lentísimo con muchas filas). El motivo (p.ej. CUDA OOM) va en el mensaje.
            _diag(f"⚠️ el entrenamiento torch/GPU falló → fallback a CPU (stdlib): {exc}")

    # Collect epoch data locally; also forward to caller's callback (async path).
    epoch_trace: list[dict[str, Any]] = []
    _diag("entrenando por camino STDLIB (CPU) — la GPU NO se usa aquí; "
          "construye los pesos en Python (~20s para 12×2048) y entrena en CPU (lento con muchas filas)")

    def _internal_cb(entry: dict[str, Any]) -> None:
        epoch_trace.append(entry)
        if epoch_callback is not None:
            epoch_callback(entry)

    result_holder: dict[str, Any] = {}
    error_holder: list[str] = []

    def _do_train(tmp: Path) -> None:
        try:
            (tmp / Path(training.model).name).write_text(mxai_text, encoding="utf-8")
            (tmp / Path(training.dataset.source).name).write_text(csv_text, encoding="utf-8")
            training_path = tmp / "training.mxtrain"
            training_path.write_text(training_text, encoding="utf-8")
            run_dir = tmp / "run"
            run_result = DenseSupervisedTrainer().train(
                spec, output_dir=run_dir, base_path=tmp, training_path=training_path,
                epoch_callback=_internal_cb, seed=seed,
            )
            result_holder["run"] = (run_result, tmp, spec)
        except Exception as exc:  # noqa: BLE001
            error_holder.append(str(exc))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t = threading.Thread(target=_do_train, args=(tmp,), daemon=True)
        t.start()
        t.join(timeout=_train_join_timeout())
        if t.is_alive():
            return {"ok": False, "error": f"Entrenamiento superó el límite de {_P9_TRAIN_TIMEOUT}s"}
        if error_holder:
            return {"ok": False, "error": error_holder[0]}
        if "run" not in result_holder:
            return {"ok": False, "error": "Entrenamiento no produjo resultado"}
        run_result, tmp2, spec2 = result_holder["run"]
        result = _collect_training_result(tmp2, run_result, spec2)
        # DenseSupervisedTrainer doesn't write metrics.json; patch epochs from local callback.
        result["epochs"] = epoch_trace
        return result


def _network_is_composite(mxai_text: str) -> bool:
    """M2-C2: True when the first NETWORK block is a P19 composite network."""
    try:
        program = parse_text(mxai_text)
        nets = getattr(program, "networks", []) or []
        return bool(nets) and getattr(nets[0], "kind", "dense_network") == "composite_network"
    except Exception:  # noqa: BLE001
        return False


def _run_playground_composite_training(
    mxai_text: str,
    training_text: str,
    csv_text: str,
    epochs_override: int | None = None,
    epoch_callback: Any = None,
    seed: int = 42,
    cancel_check: Any = None,
) -> dict[str, Any]:
    """M2-C2 — Synchronous training for composite (P19) NETWORK models.

    Mirrors the dense path but uses the composite primitives: builds the
    composite parameter set, trains with composite_train_step over epochs
    (stdlib, no torch), tracks best by validation loss, and evaluates with
    evaluate_composite_network. Returns the same result shape the rest of the
    pipeline (snapshot/infer/export, M3 metrics, M7 collapse probe) expects.
    """
    try:
        from matrixai.types import check_composite_network_types
        from matrixai.parameters.network_params import build_composite_network_parameter_set
        from matrixai.training.composite_backprop import composite_train_step
        from matrixai.forward.composite_forward import composite_forward
        from matrixai.training.composite_evaluator import evaluate_composite_network
        from matrixai.training.dense_backprop import compute_loss
        from matrixai.training.dense_trainer import _labels_from_spec, _examples_to_xy
        from matrixai.training.data import CSVDataAdapter
        from matrixai.parameters.store import program_hash

        program = parse_text(mxai_text)
        training = parse_training_text(training_text)
        epochs_override = _apply_epoch_cap(training, epochs_override)
        if not program.networks:
            return {"ok": False, "error": "El modelo no define ninguna NETWORK"}
        net = program.networks[0]
        vector_map = {v.name: v for v in program.vectors}
        vector = vector_map.get(net.input)
        if vector is None:
            return {"ok": False, "error": f"VECTOR de entrada {net.input!r} no encontrado"}

        type_result = check_composite_network_types(net, vector_map)
        if not type_result.ok:
            return {"ok": False, "error": "; ".join(type_result.errors)}

        loss_fn = training.loss.type if training.loss else "mse"
        lr = training.optimizer.learning_rate if training.optimizer else 0.01
        epochs = epochs_override if epochs_override is not None else (training.run.epochs if training.run else 50)
        patience = training.run.early_stop_patience if training.run else None
        batch_size: int | None = (
            training.dataset.batch.size if (training.dataset and training.dataset.batch) else None
        )
        labels = _labels_from_spec(training)

        # Load examples as (input_dict, target): composite_forward needs named fields
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / Path(training.dataset.source).name
            data_path.write_text(csv_text, encoding="utf-8")
            adapter = CSVDataAdapter(
                data_path, vector.name, list(vector.fields),
                training.dataset.target.name, labels if labels else None,
            )
            xy = _examples_to_xy(adapter.examples(), loss_fn, labels)
        examples = [(dict(zip(vector.fields, x)), y) for x, y in xy]
        if not examples:
            return {"ok": False, "error": "El dataset no produjo ejemplos válidos"}

        split = training.dataset.split
        train_ratio = split.train if split else 0.8
        n_train = max(1, int(len(examples) * train_ratio))
        train_ex = examples[:n_train]
        val_ex = examples[n_train:] or examples[-1:]

        mhash = program_hash(program)
        epoch_trace: list[dict[str, Any]] = []
        use_torch, device = _select_train_backend()
        # M15(f): en torch, plantilla sin pesos (init nativo, sembrado por manual_seed);
        # en stdlib, valores reales (composite_train_step los necesita).
        ps = build_composite_network_parameter_set(
            net, type_result, model_hash_str=mhash, seed=seed, with_values=not use_torch,
        )

        effective_batch_size: int | None = None
        peak_vram_gb: float | None = None

        if use_torch:
            # GPU-C3: torch backend (CUDA when available). Evaluation/result identical.
            from matrixai.training.composite_torch_trainer import train_composite_network_torch

            def _torch_cb(e: dict[str, Any]) -> None:
                entry = {"epoch": e["epoch"], "train_loss": round(e["loss"], 6),
                         "validation_loss": round(e["val_loss"], 6), "accuracy": None}
                epoch_trace.append(entry)
                if epoch_callback is not None:
                    epoch_callback(entry)  # may raise _TrainingCancelled

            tr = train_composite_network_torch(
                net, ps, examples, loss_fn, lr=lr, epochs=epochs,
                early_stop=(patience, "validation_loss") if patience else None,
                device=device, seed=seed, batch_size=batch_size,
                epoch_callback=_torch_cb, cancel_check=cancel_check,
            )
            best_ps = tr["best_params"]
            best_epoch = tr["best_epoch"]
            best_val_loss = tr["best_val_loss"]
            final_train_loss = tr["train_loss"]
            effective_batch_size = tr.get("effective_batch_size")
            peak_vram_gb = tr.get("peak_vram_gb")
            _diag(f"entrenamiento composite OK en {device}: batch spec={batch_size} -> efectivo={effective_batch_size} "
                  f"(ejemplos={len(examples)}, VRAM pico={peak_vram_gb} GB)")
            backend_label = device  # "cuda" | "cpu"
        else:
            best_ps = ps
            best_val_loss = float("inf")
            best_epoch = 1
            final_train_loss = 0.0
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                for x, y in train_ex:
                    ps, loss = composite_train_step(net, ps, x, y, loss_fn, learning_rate=lr, training=True)
                    epoch_loss += loss
                final_train_loss = epoch_loss / len(train_ex) if train_ex else 0.0

                val_loss = 0.0
                for x, y in val_ex:
                    pred = composite_forward(net, ps, x, training=False)
                    val_loss += compute_loss(loss_fn, pred, y)
                val_loss = val_loss / len(val_ex) if val_ex else final_train_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ps = ps
                    best_epoch = epoch

                entry = {
                    "epoch": epoch,
                    "train_loss": round(final_train_loss, 6),
                    "validation_loss": round(val_loss, 6),
                    "accuracy": None,
                }
                epoch_trace.append(entry)
                if epoch_callback is not None:
                    epoch_callback(entry)  # may raise _TrainingCancelled on cancel/timeout

                if patience is not None and (epoch - best_epoch) >= patience:
                    break
            backend_label = "stdlib"

        # M14 — si se entrenó en torch/GPU, evaluar también por torch (no CPU por muestra).
        composite_eval_warning: str | None = None
        if use_torch:
            from matrixai.training.composite_torch_trainer import evaluate_composite_network_torch
            try:
                ev = evaluate_composite_network_torch(
                    net, best_ps, val_ex, loss_fn, labels=labels or None, device=device,
                    cancel_check=cancel_check,
                )
                composite_eval_backend = device
            except _TrainingCancelled:
                raise
            except Exception as _eval_exc:  # noqa: BLE001
                ev = evaluate_composite_network(net, best_ps, val_ex, loss_fn, labels=labels or None)
                composite_eval_backend = "stdlib_fallback"
                composite_eval_warning = f"eval torch failed, fell back to stdlib: {_eval_exc}"
        else:
            ev = evaluate_composite_network(net, best_ps, val_ex, loss_fn, labels=labels or None)
            composite_eval_backend = "stdlib"
        evaluation_report = ev.to_dict()
        is_reg = ev.is_regression()

        return {
            "ok": True,
            "task_kind": "regression" if is_reg else "classification",
            "run_id": uuid.uuid4().hex[:8],
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "accuracy": None if is_reg else evaluation_report.get("accuracy"),
            "macro_f1": evaluation_report.get("macro_f1"),
            "confusion_matrix": evaluation_report.get("confusion_matrix"),
            "labels": evaluation_report.get("labels"),
            "per_label": evaluation_report.get("per_label"),
            "mae": evaluation_report.get("mae"),
            "rmse": evaluation_report.get("rmse"),
            "r2": evaluation_report.get("r2"),
            "backend": backend_label,
            "evaluation_backend": composite_eval_backend,
            "evaluation_warning": composite_eval_warning,
            "effective_batch_size": effective_batch_size,
            "peak_vram_gb": peak_vram_gb,
            "epochs": epoch_trace,
            "params_best": best_ps.to_dict(),
            "metrics": {"epochs": epoch_trace},
            "training_trace": {
                "backend_report": {
                    "target": backend_label,
                    "effective_batch_size": effective_batch_size,
                    "peak_vram_gb": peak_vram_gb,
                },
                "task_kind": "regression" if is_reg else "classification",
            },
            "evaluation_report": evaluation_report,
            "network_kind": "composite_network",
        }
    except _TrainingCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _run_playground_training(
    mxai_text: str,
    training_text: str,
    csv_text: str,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Synchronous training — used by tests and /api/train endpoint."""
    prediction_kind = _get_prediction_kind(mxai_text, training_text)

    if prediction_kind == "layer_call":
        validation = _validate_training_csv(mxai_text, training_text, csv_text)
        if not validation.get("ok"):
            return {"ok": False, "error": validation.get("error") or str(validation.get("errors", "validation failed"))}
        result_holder: dict[str, Any] = {}
        error_holder: list[str] = []

        def _do_generic(tmp: Path) -> None:  # noqa: F841  (tmp unused but mirrors legacy signature)
            try:
                result_holder["run"] = _run_playground_generic_training(
                    mxai_text, training_text, csv_text, epochs_override
                )
            except Exception as exc:  # noqa: BLE001
                error_holder.append(str(exc))

        t = threading.Thread(target=_do_generic, args=(Path("."),), daemon=True)
        t.start()
        t.join(timeout=_train_join_timeout())
        if t.is_alive():
            return {"ok": False, "error": f"Entrenamiento superó el límite de {_P9_TRAIN_TIMEOUT}s"}
        if error_holder:
            return {"ok": False, "error": error_holder[0]}
        return result_holder.get("run", {"ok": False, "error": "Entrenamiento no produjo resultado"})

    if prediction_kind == "network_call":
        validation = _validate_training_csv(mxai_text, training_text, csv_text)
        if not validation.get("ok"):
            return {"ok": False, "error": validation.get("error") or str(validation.get("errors", "validation failed"))}
        # M2-C2: route composite (P19) networks to the composite trainer
        if _network_is_composite(mxai_text):
            return _run_playground_composite_training(mxai_text, training_text, csv_text, epochs_override)
        return _run_playground_dense_training(mxai_text, training_text, csv_text, epochs_override)

    validation = _validate_training_csv(mxai_text, training_text, csv_text)
    if not validation.get("ok"):
        return {"ok": False, "error": validation.get("error") or str(validation.get("errors", "validation failed"))}
    try:
        training = parse_training_text(training_text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"training_text inválido: {exc}"}

    epochs_override = _apply_epoch_cap(training, epochs_override)
    spec = _build_spec_with_epochs(training, epochs_override)

    result_holder: dict[str, Any] = {}
    error_holder: list[str] = []

    def _do_train(tmp: Path) -> None:
        try:
            (tmp / Path(training.model).name).write_text(mxai_text, encoding="utf-8")
            (tmp / Path(training.dataset.source).name).write_text(csv_text, encoding="utf-8")
            training_path = tmp / "training.mxtrain"
            training_path.write_text(training_text, encoding="utf-8")
            run_dir = tmp / "run"
            run_result = SupervisedTrainer().train(spec, output_dir=run_dir, base_path=tmp, training_path=training_path)
            result_holder["run"] = (run_result, tmp, spec)
        except Exception as exc:  # noqa: BLE001
            error_holder.append(str(exc))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t = threading.Thread(target=_do_train, args=(tmp,), daemon=True)
        t.start()
        t.join(timeout=_train_join_timeout())
        if t.is_alive():
            return {"ok": False, "error": f"Entrenamiento superó el límite de {_P9_TRAIN_TIMEOUT}s"}
        if error_holder:
            return {"ok": False, "error": error_holder[0]}
        if "run" not in result_holder:
            return {"ok": False, "error": "Entrenamiento no produjo resultado"}
        run_result, tmp2, spec2 = result_holder["run"]
        return _collect_training_result(tmp2, run_result, spec2)


def _run_playground_generic_training(
    mxai_text: str,
    training_text: str,
    csv_text: str,
    epochs_override: int | None = None,
    epoch_callback: Any = None,
) -> dict[str, Any]:
    """Training for layer_call (P11+) models using GenericSupervisedTrainer."""
    from matrixai.training.data import CSVDataAdapter

    try:
        training = parse_training_text(training_text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"training_text inválido: {exc}"}

    epochs = _limits.cap(
        int(epochs_override) if epochs_override is not None else (training.run.epochs if training.run else 10),
        "max_epochs",
    )

    try:
        program = parse_text(mxai_text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"mxai_text inválido: {exc}"}

    vector_name = training.dataset.input.vector
    vector_fields = list(training.dataset.input.columns)
    target_name = training.dataset.target.name
    labels_raw = training.dataset.target.type.parameters.get("args", [])
    labels = [str(l) for l in labels_raw] if labels_raw else None
    prediction_key = training.loss.prediction
    update_patterns = list(training.optimizer.update)
    learning_rate = training.optimizer.learning_rate or 0.01
    stem = _mxai_project_stem(mxai_text)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write files under both the bare name and the relative subdirectory path
        # so that _resolve_path(training.model, tmp) can find them.
        model_rel = Path(training.model)
        (tmp / model_rel.name).write_text(mxai_text, encoding="utf-8")
        if str(model_rel.parent) not in (".", ""):
            dest = tmp / model_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(mxai_text, encoding="utf-8")

        csv_rel = Path(training.dataset.source)
        csv_dest = tmp / csv_rel.name
        csv_dest.write_text(csv_text, encoding="utf-8")
        if str(csv_rel.parent) not in (".", ""):
            dest2 = tmp / csv_rel
            dest2.parent.mkdir(parents=True, exist_ok=True)
            dest2.write_text(csv_text, encoding="utf-8")

        try:
            adapter = CSVDataAdapter(csv_dest, vector_name, vector_fields, target_name, labels)
            all_examples = adapter.examples()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"CSV inválido: {exc}"}

        split = training.dataset.split
        train_ratio = split.train if split else 0.8
        n_train = max(1, int(len(all_examples) * train_ratio))
        train_examples = all_examples[:n_train]
        val_examples = all_examples[n_train:] or all_examples[-1:]

        try:
            result = GenericSupervisedTrainer().train(
                program=program,
                training=training,
                examples=train_examples,
                validation_examples=val_examples,
                prediction_key=prediction_key,
                target_key=target_name,
                update_patterns=update_patterns,
                epochs=epochs,
                learning_rate=learning_rate,
                epoch_callback=epoch_callback,
                labels=labels,
                vector_name=vector_name,
                vector_fields=vector_fields,
                parameter_set_prefix=stem,
            )
        except _TrainingCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

        best_ps = result["best_parameter_set"]
        epoch_trace = result.get("epoch_trace", [])

        best_epoch = 0
        best_val_loss = None
        for entry in epoch_trace:
            if best_val_loss is None or entry["validation_loss"] < best_val_loss:
                best_val_loss = entry["validation_loss"]
                best_epoch = entry["epoch"]
        final_train_loss = epoch_trace[-1]["train_loss"] if epoch_trace else None

        evaluation_report = None
        try:
            ev = GenericSupervisedEvaluator().evaluate(
                training,
                parameter_set=best_ps,
                data_path=str(csv_dest),
                base_path=str(tmp),
            )
            evaluation_report = ev.to_dict()
        except Exception:  # noqa: BLE001
            pass

        _is_reg = evaluation_report and not evaluation_report.get("labels")
        return {
            "ok": True,
            "task_kind": "regression" if _is_reg else "classification",
            "run_id": uuid.uuid4().hex[:8],
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "accuracy": (best_ps.metrics or {}).get("accuracy"),
            "mae": (evaluation_report or {}).get("mae"),
            "rmse": (evaluation_report or {}).get("rmse"),
            "r2": (evaluation_report or {}).get("r2"),
            # M3: classification metrics flattened for the client
            "macro_f1": (evaluation_report or {}).get("macro_f1"),
            "confusion_matrix": (evaluation_report or {}).get("confusion_matrix"),
            "labels": (evaluation_report or {}).get("labels"),
            "per_label": (evaluation_report or {}).get("per_label"),
            "backend": "stdlib",
            "epochs": epoch_trace,
            "params_best": best_ps.to_dict(),
            "metrics": {"epochs": epoch_trace},
            "training_trace": {"backend_report": {"target": "stdlib"}},
            "evaluation_report": evaluation_report,
        }


def _coerce_field_ranges(raw: Any) -> dict[str, tuple[float, float]]:
    """Validate a {col: [lo, hi]} payload into usable ranges (lo < hi)."""
    ranges: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return ranges
    for key, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
        if lo < hi:
            ranges[str(key)] = (lo, hi)
    return ranges


_S2_FIELD_TYPES = ("number", "integer", "boolean")


def _coerce_field_types(raw: Any) -> dict[str, str]:
    """S2 — validate a {col: type} payload; unknown types are dropped."""
    types: dict[str, str] = {}
    if not isinstance(raw, dict):
        return types
    for key, value in raw.items():
        t = str(value or "").strip().lower()
        if t in _S2_FIELD_TYPES:
            types[str(key)] = t
    return types


_S2_MAX_VOCAB = 100  # one-hot above this is impractical; reject the column


def _coerce_field_categories(raw: Any) -> dict[str, list[str]]:
    """S2-C2 — validate a {col: [values]} payload for one-hot expansion.

    Keeps columns with >= 2 distinct non-empty string values (order preserved,
    duplicates dropped), capped at _S2_MAX_VOCAB. Everything else is dropped.
    """
    out: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return out
    for key, values in raw.items():
        if not isinstance(values, (list, tuple)):
            continue
        seen: set[str] = set()
        clean: list[str] = []
        for v in values:
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                clean.append(s)
        if 2 <= len(clean) <= _S2_MAX_VOCAB:
            out[str(key)] = clean
    return out


def _coerce_field_identifiers(raw: Any) -> list[str]:
    """S2-C4 — validate a list of identifier column names to exclude."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in raw:
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _suggest_field_types(columns: list[str]) -> dict[str, str]:
    """S2 — heuristic type suggestion from column names (user can override).

    Only suggests non-default types; columns not listed are "number".
    """
    suggestions: dict[str, str] = {}
    bool_prefixes = ("is_", "has_", "es_", "tiene_", "flag_")
    int_markers = ("num_", "n_", "count", "_count", "numero_", "número_")
    for col in columns:
        low = col.lower()
        if low.startswith(bool_prefixes) or low.endswith(("_flag", "_bool")):
            suggestions[col] = "boolean"
        elif low.startswith(int_markers) or low.endswith(int_markers):
            suggestions[col] = "integer"
    return suggestions


def _normalize_csv_with_ranges(csv_text: str, field_ranges: dict[str, tuple[float, float]]) -> str:
    """M5 — training boundary: map domain-scale columns (salary 35000) back to
    [0,1] with the SAME ranges used to generate/display the dataset.

    Columns without a range, non-numeric cells and the target are left as-is.
    Values are clamped to [0,1] so a hand-edited cell slightly out of range
    cannot push training outside the slider space.
    """
    if not field_ranges:
        return csv_text
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return csv_text
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(reader.fieldnames))
    writer.writeheader()
    for row in reader:
        for col, (lo, hi) in field_ranges.items():
            if col not in row:
                continue
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                continue
            norm = (value - lo) / (hi - lo)
            row[col] = str(round(min(1.0, max(0.0, norm)), 6))
        writer.writerow(row)
    return out.getvalue()


_COLLAPSE_TOLERANCE = 1e-6


def _probe_model_collapse(mxai_text: str, params_best: dict[str, Any]) -> dict[str, Any] | None:
    """M7 — detect a trained model that became a constant predictor.

    Runs 4 deterministic probes over the normalized input space (zeros, ones,
    two seeded randoms) and compares the network outputs. A collapsed model
    (e.g. dying ReLU in a bottleneck layer) returns softmax(bias) for every
    input — the class prior — and looks "trained OK" without this check.

    Best-effort: returns None when the model cannot be probed (no networks,
    invalid params); a training result must never fail because of the probe.
    """
    import random  # noqa: PLC0415

    try:
        program = parse_text(mxai_text)
        networks = getattr(program, "networks", []) or []
        if not networks or not program.vectors:
            return None
        vector = program.vectors[0]
        fields = list(vector.fields)
        if not fields:
            return None
        runtime_params = ParameterSet.from_dict(params_best).runtime_parameters()

        rng = random.Random(7)
        probes = [
            {f: 0.0 for f in fields},
            {f: 1.0 for f in fields},
            {f: round(rng.random(), 6) for f in fields},
            {f: round(rng.random(), 6) for f in fields},
        ]
        outputs: list[list[float]] = []
        runtime = MatrixAIRuntime()
        for values in probes:
            run_result = runtime.run(program, {vector.name: values}, parameters=runtime_params)
            state = run_result.get("state", {})
            flat: list[float] = []
            for net in networks:
                out = state.get(net.output)
                if isinstance(out, (list, tuple)):
                    flat.extend(_safe_float(v) for v in out)
                elif out is not None:
                    flat.append(_safe_float(out))
            if not flat:
                return None
            outputs.append(flat)
        first = outputs[0]
        if any(len(o) != len(first) for o in outputs[1:]):
            return None
        collapsed = all(
            abs(value - first[i]) <= _COLLAPSE_TOLERANCE
            for probe_out in outputs[1:]
            for i, value in enumerate(probe_out)
        )
        result: dict[str, Any] = {"collapsed": collapsed}
        if collapsed:
            result["constant_output"] = [round(v, 6) for v in first]
        return result
    except Exception:  # noqa: BLE001
        return None


def _attach_collapse_check(result: Any, mxai_text: str) -> Any:
    """M7 — annotate a successful training result with the collapse probe."""
    if not isinstance(result, dict) or not result.get("ok") or not result.get("params_best"):
        return result
    # GPU: el camino torch ya hizo la prueba por torch (instantánea); no repetir la
    # versión por runtime Python (O(params)×4 → minutos/GBs con redes grandes).
    if "model_collapsed" in result:
        return result
    probe = _probe_model_collapse(mxai_text, result["params_best"])
    if probe is None:
        return result
    result["model_collapsed"] = probe["collapsed"]
    if probe["collapsed"]:
        result["collapse_constant_output"] = probe.get("constant_output")
        result["collapse_warning"] = (
            "El modelo entrenado produce la misma salida para cualquier entrada "
            "(predictor constante). Suele deberse a una red demasiado profunda o a "
            "un cuello de botella ReLU antes de la capa de salida: simplifica la "
            "arquitectura y reentrena."
        )
    return result


def _submit_training_job(
    mxai_text: str,
    training_text: str,
    csv_text: str,
    epochs_override: int | None = None,
    field_ranges: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Start async training job. Returns {ok, job_id} immediately.

    M8-A3: `seed` drives parameter initialization; a different seed is the
    actionable retry for a collapsed (dying-ReLU) model.
    """
    # Enforce 1 concurrent run (contract P9 §Límites operativos)
    if any(j["status"] == "running" for j in _training_jobs.values()):
        return {"ok": False, "error": "Ya hay un entrenamiento en curso. Espera a que termine o pulsa Detener."}

    # M5: domain-scale CSV → normalized BEFORE validation, so the validator and
    # the three trainer paths only ever see slider-space [0,1] values.
    if field_ranges:
        csv_text = _normalize_csv_with_ranges(csv_text, field_ranges)

    validation = _validate_training_csv(mxai_text, training_text, csv_text)
    if not validation.get("ok"):
        return {"ok": False, "error": validation.get("error") or str(validation.get("errors", "validation failed"))}
    try:
        training = parse_training_text(training_text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"training_text inválido: {exc}"}

    prediction_kind = _get_prediction_kind(mxai_text, training_text)
    _diag(f"job: prediction_kind={prediction_kind!r} → "
          f"{'GPU (red densa)' if prediction_kind == 'network_call' else 'CPU (camino genérico/stdlib)'}")
    epochs_override = _apply_epoch_cap(training, epochs_override)
    spec = _build_spec_with_epochs(training, epochs_override)

    job_id = uuid.uuid4().hex[:8]
    cancel_event = threading.Event()
    job: dict[str, Any] = {"status": "running", "epochs": [], "result": None, "error": None}
    _training_jobs[job_id] = job
    # Keep job store bounded
    if len(_training_jobs) > _MAX_JOBS:
        oldest = next(iter(_training_jobs))
        del _training_jobs[oldest]

    def _run() -> None:
        def epoch_cb(entry: dict[str, Any]) -> None:
            if cancel_event.is_set():
                raise _TrainingCancelled()
            job["epochs"].append(entry)

        def cancel_check() -> None:
            if cancel_event.is_set():
                raise _TrainingCancelled()

        # Watchdog: enforce the wall-clock budget on the async path (same as sync).
        # Disabled when the budget is <=0 (downloadable Studio): the job runs to
        # completion and the user's Cancel button is the control.
        def _on_timeout() -> None:
            cancel_event.set()
            job["_timed_out"] = True

        watchdog = (
            threading.Timer(_P9_TRAIN_TIMEOUT, _on_timeout)
            if _P9_TRAIN_TIMEOUT and _P9_TRAIN_TIMEOUT > 0 else None
        )
        if watchdog is not None:
            watchdog.daemon = True
            watchdog.start()
        try:
            if prediction_kind == "layer_call":
                result = _run_playground_generic_training(
                    mxai_text, training_text, csv_text, epochs_override, epoch_callback=epoch_cb,
                )
                if cancel_event.is_set():
                    job["status"] = "timeout" if job.get("_timed_out") else "cancelled"
                    return
                job["result"] = _attach_collapse_check(result, mxai_text)
                job["status"] = "done" if result.get("ok") else "error"
                if not result.get("ok"):
                    job["error"] = result.get("error")
            elif prediction_kind == "network_call":
                # M2-C2: composite (P19) networks use the composite trainer
                _net_trainer = (
                    _run_playground_composite_training
                    if _network_is_composite(mxai_text)
                    else _run_playground_dense_training
                )
                result = _net_trainer(
                    mxai_text, training_text, csv_text, epochs_override, epoch_callback=epoch_cb,
                    seed=seed, cancel_check=cancel_check,
                )
                if cancel_event.is_set():
                    job["status"] = "timeout" if job.get("_timed_out") else "cancelled"
                    return
                job["result"] = _attach_collapse_check(result, mxai_text)
                job["status"] = "done" if result.get("ok") else "error"
                if not result.get("ok"):
                    job["error"] = result.get("error")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    (tmp / Path(training.model).name).write_text(mxai_text, encoding="utf-8")
                    (tmp / Path(training.dataset.source).name).write_text(csv_text, encoding="utf-8")
                    training_path = tmp / "training.mxtrain"
                    training_path.write_text(training_text, encoding="utf-8")
                    run_dir = tmp / "run"
                    run_result = SupervisedTrainer().train(
                        spec, output_dir=run_dir, base_path=tmp,
                        training_path=training_path, epoch_callback=epoch_cb,
                    )
                    result = _collect_training_result(tmp, run_result, spec)
                # Guard: worker may finish after a cancel/timeout signal
                if cancel_event.is_set():
                    job["status"] = "timeout" if job.get("_timed_out") else "cancelled"
                    return
                job["result"] = _attach_collapse_check(result, mxai_text)
                job["status"] = "done"
        except _TrainingCancelled:
            job["status"] = "timeout" if job.get("_timed_out") else "cancelled"
        except Exception as exc:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            if watchdog is not None:
                watchdog.cancel()
            # Release GPU VRAM back to the OS so Colab/nvidia-smi reflect the stop immediately.
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    import gc as _gc
                    _gc.collect()
                    _torch.cuda.empty_cache()
                    _free_gb = _torch.cuda.memory_reserved() / 1e9
                    _diag(f"worker {job_id} terminó (status={job['status']}); "
                          f"VRAM reservada tras limpiar: {_free_gb:.2f} GB")
            except Exception:  # noqa: BLE001
                pass

    job["cancel_event"] = cancel_event
    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job_id": job_id}


def _get_job_status(job_id: str) -> dict[str, Any]:
    job = _training_jobs.get(job_id)
    if job is None:
        return {"ok": False, "error": f"job {job_id!r} no encontrado"}
    result = {
        "ok": True,
        "job_id": job_id,
        "status": job["status"],
        "epochs": list(job["epochs"]),
        "error": job.get("error"),
    }
    if job["status"] == "done" and job.get("result"):
        result.update(job["result"])
    return result


def _cancel_job(job_id: str) -> dict[str, Any]:
    job = _training_jobs.get(job_id)
    if job is None:
        # Visible en la consola/celda de Colab: prueba que el POST llegó pero el job
        # no está (id equivocado / registro distinto). Útil para diagnosticar Stop.
        _diag(f"cancel: job {job_id!r} NO encontrado")
        return {"ok": False, "error": f"job {job_id!r} no encontrado"}
    if "cancel_event" in job:
        job["cancel_event"].set()
    # Only force-set if the worker hasn't already finished
    if job["status"] == "running":
        job["status"] = "cancelled"
    _diag(f"cancel: job {job_id} → cancel_event.set(), status={job['status']}")
    return {"ok": True, "job_id": job_id, "status": job["status"]}


def _playground_run_with_params(mxai_text: str, params_json: str, input_json: str) -> dict[str, Any]:
    if not mxai_text.strip():
        return {"ok": False, "error": "mxai_text es obligatorio"}
    if not params_json.strip():
        return {"ok": False, "error": "params_json es obligatorio"}
    if not input_json.strip():
        return {"ok": False, "error": "input_json es obligatorio"}
    try:
        params_data = json.loads(params_json)
        parameter_set = ParameterSet.from_dict(params_data)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"params_json inválido: {exc}"}
    try:
        input_data = json.loads(input_json)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"input_json inválido: {exc}"}
    try:
        from matrixai.parser import parse_text as _parse_text
        program = _parse_text(mxai_text)
        runtime_params = parameter_set.runtime_parameters()
        result = MatrixAIRuntime().run(program, input_data, parameters=runtime_params)
        return {"ok": True, "result": result.to_dict() if hasattr(result, "to_dict") else str(result)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _refine_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "Se requiere un prompt."}
    run_result = payload.get("run_result") or {}
    if not run_result:
        return {"ok": False, "error": "Se requiere un run_result con acciones y traza."}
    mxai_text = str(payload.get("mxai_text") or "").strip() or None
    hints: list[str] = [str(h) for h in (payload.get("hints") or []) if str(h).strip()]
    iteration_count = int(payload.get("iteration_count") or 1)
    refinement_chain: list[str] = [str(x) for x in (payload.get("refinement_chain") or [])]
    parent_prompt_hash = str(payload.get("parent_prompt_hash") or "").strip() or None
    max_iterations = int(payload.get("max_iterations") or RefinementAgent.DEFAULT_MAX_ITERATIONS)
    audit = {k: v for k, v in run_result.items() if k != "audit"}
    try:
        proposal = RefinementAgent().refine(
            prompt,
            mxai=mxai_text,
            audit=audit,
            hints=hints or None,
            mode="audit_driven",
            iteration_count=iteration_count,
            refinement_chain=refinement_chain or None,
            parent_prompt_hash=parent_prompt_hash,
            max_iterations=max_iterations,
        )
        return {
            "ok": True,
            "refinement_id": proposal.refinement_id,
            "mode": proposal.mode,
            "iteration": proposal.iteration_count,
            "supervision_accepted": proposal.supervision_accepted,
            "chain": proposal.refinement_chain,
            "parent_hash": proposal.parent_prompt_hash,
            "proposed_prompt": proposal.proposed_prompt,
            "explanation": proposal.explanation,
        }
    except IterationLimitReached as exc:
        return {"ok": False, "error": str(exc), "iteration_limit_reached": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
















# ---------------------------------------------------------------------------
# PRD-3 — Persistencia de modelos del Studio ("Mis modelos")
# ---------------------------------------------------------------------------



















# ---------------------------------------------------------------------------
# PRD-2 — Inferencia post-entrenamiento ("Probar modelo")
# ---------------------------------------------------------------------------





def _resolve_llm_config_path() -> Path:
    """Resolve the LLM config file path.

    Priority: MATRIXAI_LLM_ENV_FILE > MATRIXAI_LLM_CONFIG_PATH > /config/llm.env (Docker)
    Fallback when /config is absent (dev): ~/.config/matrixai/llm.env
    """
    explicit = os.environ.get("MATRIXAI_LLM_ENV_FILE") or os.environ.get("MATRIXAI_LLM_CONFIG_PATH")
    if explicit:
        return Path(explicit)
    docker_path = Path("/config/llm.env")
    if docker_path.parent.exists():
        return docker_path
    return Path.home() / ".config" / "matrixai" / "llm.env"


_LLM_CONFIG_PATH = _resolve_llm_config_path()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default










def _detect_llm_mode() -> dict[str, Any]:
    try:
        p = ChatCompletionsLLMProposalProvider.from_env()
        return {"active": True, "model": p.model_name, "provider": p.provider_name}
    except Exception:  # noqa: BLE001
        return {"active": False, "model": "", "provider": ""}


_DENSE_SCHEMA_SYSTEM = (
    "You are a schema designer for supervised machine learning models.\n"
    "Analyse the problem first — its difficulty, number of features, and whether it "
    "is genuinely non-linear — then design a SOUND architecture. Respond with "
    "exactly these lines — no extra text:\n\n"
    "FIELDS: comma-separated input feature names (snake_case, domain-specific; "
    "include EVERY input feature the user explicitly lists, up to 32 — only invent 4-8 when none are given)\n"
    "LABELS: comma-separated class names (for classification, use the user's class names verbatim; omit line for regression)\n"
    "NAME: PascalCase model class name (no spaces)\n"
    "ENTITY: PascalCase entity name being scored (no spaces, e.g. LoanApplicant)\n"
    "ARCHITECTURE: 'dense' or 'residual'. Use 'dense' for typical tabular tasks; "
    "choose 'residual' ONLY for genuinely deep/complex non-linear problems. NEVER put "
    "a narrow ReLU layer right before the output.\n"
    "LAYERS: comma-separated hidden layer sizes (2-12 integers; honour any architecture the user explicitly requests, e.g. 64, 64, 64, 32)\n"
    "CATEGORICALS: comma-separated 'field:vocab' for HIGH-cardinality categorical "
    f"features (more than {_ONEHOT_MAX} distinct values) that have NO natural order "
    "(e.g. product_id:5000, diagnosis_code:1200). vocab is the approximate number of "
    "distinct values. Omit the line when there are no such features; do NOT list "
    "ordered/numeric features here, nor fields the user already declared with an "
    "explicit type (Categorical[...], Boolean, Scalar[...]) — those are honored "
    "from the prompt directly.\n"
    "RATIONALE: one short sentence justifying the architecture choice\n\n"
    "Use precise domain vocabulary. No explanations beyond the RATIONALE line."
)


def _parse_dense_schema(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("FIELDS:"):
            fields = [f.strip() for f in line[7:].split(",") if f.strip()]
            if fields:
                result["input_fields"] = fields[:32]
        elif line.startswith("LABELS:"):
            labels = [lb.strip() for lb in line[7:].split(",") if lb.strip()]
            if labels:
                result["labels"] = labels
        elif line.startswith("NAME:"):
            name = re.sub(r"\s+", "", line[5:].strip())
            if name:
                result["network_name"] = name
        elif line.startswith("ENTITY:"):
            entity = re.sub(r"\s+", "", line[7:].strip())
            if entity:
                result["input_name"] = entity
        elif line.startswith("LAYERS:"):
            try:
                sizes = [int(s.strip()) for s in line[7:].split(",") if s.strip()]
                sizes = [s for s in sizes if s > 0]
                if sizes:
                    result["hidden_layers"] = [(s, "relu") for s in sizes[:12]]
            except ValueError:
                pass
        elif line.startswith("ARCHITECTURE:"):
            # M8-B1: the LLM's architecture choice (dense vs residual)
            val = line[len("ARCHITECTURE:"):].strip().lower()
            if val:
                result["architecture"] = "residual" if "resid" in val else "dense"
        elif line.startswith("CATEGORICALS:"):
            # M2 v2 C5: high-cardinality categoricals the LLM proposes for native
            # EMBEDDING. Parse 'field:vocab' pairs; ignore malformed entries.
            cats: dict[str, int] = {}
            for item in line[len("CATEGORICALS:"):].split(","):
                m = re.match(r"^\s*(\w+)\s*:\s*(\d+)\s*$", item)
                if m:
                    vocab = int(m.group(2))
                    if vocab >= 2:
                        cats[m.group(1)] = vocab
            if cats:
                result["categorical_fields"] = cats
        elif line.startswith("RATIONALE:"):
            rationale = line[len("RATIONALE:"):].strip()
            if rationale:
                result["rationale"] = rationale[:240]
    return result


def _dense_llm_schema(prompt: str) -> dict[str, Any]:
    try:
        provider = ChatCompletionsLLMProposalProvider.from_env()
        text = provider.complete(_DENSE_SCHEMA_SYSTEM, prompt)
        return _parse_dense_schema(text)
    except Exception as exc:  # noqa: BLE001
        return {"_llm_warning": f"LLM schema extraction failed: {exc}"}


_DATASET_RANGES_SYSTEM = (
    "You are a data scientist creating realistic synthetic training datasets.\n"
    "Given a list of input feature names for a machine learning model, suggest realistic numeric value ranges.\n"
    "Reply with exactly one line per field in this format:\n\n"
    "FIELD_NAME: min max\n\n"
    "Rules:\n"
    "- min and max must be plain numbers (integers or decimals), e.g. age: 18 90\n"
    "- Use domain-appropriate ranges (credit_score: 300 850, age: 18 90, income: 20000 200000)\n"
    "- For ratio/probability fields (rate, ratio, pct): 0.0 1.0\n"
    "- For boolean-like flags (has_X, is_X): 0 1\n"
    "- Only output FIELD_NAME: min max lines, nothing else. No explanations."
)


def _parse_field_ranges(text: str) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(\w+)\s*:\s*([-\d.]+)\s+([-\d.]+)\s*$", line)
        if m:
            try:
                lo, hi = float(m.group(2)), float(m.group(3))
                if lo < hi:
                    result[m.group(1)] = (lo, hi)
            except ValueError:
                pass
    return result


def _llm_field_ranges(fields: list[str], context: str = "") -> dict[str, tuple[float, float]]:
    try:
        provider = ChatCompletionsLLMProposalProvider.from_env()
        user = f"Model context: {context}\nFields: {', '.join(fields)}" if context else f"Fields: {', '.join(fields)}"
        text = provider.complete(_DATASET_RANGES_SYSTEM, user)
        return _parse_field_ranges(text)
    except Exception:  # noqa: BLE001
        return {}


# M8 v2 — LLM as domain simulator: propose the feature→class logic ONCE as bounded
# threshold rules. A deterministic evaluator (domain_rules.py) applies them to every
# sampled row → synthetic data with plausible, learnable signal instead of toy.
_DOMAIN_RULES_SYSTEM = (
    "You are a domain expert defining how input features determine the output class "
    "in a realistic dataset. Given the problem, the input features and the class "
    "labels, write deterministic threshold rules a real expert would use.\n\n"
    "Respond with one rule per line, the most severe/specific class first, in EXACTLY "
    "this format — no other text:\n\n"
    "CLASS_LABEL: feature OP value [AND|OR feature OP value ...]\n"
    "DEFAULT: CLASS_LABEL\n\n"
    "Rules:\n"
    "- OP is one of < <= > >= ==. 'value' is a plain number in the feature's natural "
    "domain scale (e.g. age 65, charlson_index 15, creatinine 4).\n"
    "- Use ONLY the given feature names and class labels, verbatim.\n"
    "- A single line uses either AND or OR, never both.\n"
    "- Order rules from the most severe / least common class to the least; the first "
    "matching rule wins.\n"
    "- End with DEFAULT: <the baseline / most common class>.\n"
    "- Encode genuine domain knowledge so labels are plausible, not arbitrary."
)


def _llm_domain_rules(prompt: str, features: list[str], labels: list[str]):
    """Ask the LLM for domain threshold rules; return a parsed DomainRules or None.

    Returns None on any failure (no provider, error) — the caller validates and falls
    back to the toy `coherent` labelling."""
    from matrixai.training.domain_rules import parse_domain_rules  # noqa: PLC0415
    try:
        provider = ChatCompletionsLLMProposalProvider.from_env()
        user = (
            f"Problem: {prompt}\n"
            f"Features: {', '.join(features)}\n"
            f"Classes: {', '.join(labels)}"
        )
        text = provider.complete(_DOMAIN_RULES_SYSTEM, user)
        return parse_domain_rules(text)
    except Exception:  # noqa: BLE001
        return None


def _is_neural_prompt(prompt: str) -> bool:
    """Return True when the prompt requests a predictive/classification model.

    Precedence:
    1. Workflow/rules intent → False (these stay in PromptSupervisor).
    2. Explicit neural vocabulary → True.
    3. Common predictive/classification task verbs → True.
    """
    text = prompt.lower()
    if any(kw in text for kw in _WORKFLOW_KEYWORDS):
        return False
    if any(kw in text for kw in _NEURAL_INTENT_KEYWORDS):
        return True
    return any(kw in text for kw in _NEURAL_TASK_KEYWORDS)


# M2-C3 — composite/sequence/unsupported detection in the prompt.
# Conservative: only route to composite on EXPLICIT architecture hints, so
# ordinary neural prompts keep the dense path unchanged.
_COMPOSITE_HINTS = (
    "residual", "layernorm", "layer norm", "dropout", "bloque", "block",
    "complejo", "complex", "no lineal", "no-lineal", "nonlinear", "non-linear",
    "expresiv", "expressive", "profund", "deep",
)
_SEQUENCE_HINTS = (
    "secuencia", "sequence", "serie temporal", "series temporales", "time series",
    "temporal", "recurren",
)
_UNSUPPORTED_OPS = {
    "atención": "attention", "attention": "attention", "transformer": "transformer",
    "rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "batchnorm": "BatchNorm",
    "batch norm": "BatchNorm", "convolu": "convolution", "conv ": "convolution",
}


_FORCE_DENSE_HINTS = (
    "densa pura", "densas puras", "red densa pura", "dense pura", "pure dense",
    "solo dense", "solo densa", "solo densas", "solo capas dense", "only dense",
    "only dense layers", "solo capas densas", "puramente densa",
)


def _prompt_forces_dense(prompt: str) -> bool:
    """True cuando el prompt pide EXPLÍCITAMENTE una red densa pura ("SOLO capas Dense",
    "red densa pura"). Anula los hints débiles de composite (p.ej. "profunda"/"deep", que
    el usuario usa para "muchas capas", no para una arquitectura compleja)."""
    text = prompt.lower()
    return any(h in text for h in _FORCE_DENSE_HINTS)


def _prompt_wants_composite(prompt: str) -> bool:
    text = prompt.lower()
    if _prompt_forces_dense(prompt):
        return False
    return any(h in text for h in _COMPOSITE_HINTS)


def _prompt_is_sequence(prompt: str) -> bool:
    text = prompt.lower()
    return any(h in text for h in _SEQUENCE_HINTS)


def _prompt_unsupported_ops(prompt: str) -> list[str]:
    text = prompt.lower()
    found: list[str] = []
    for needle, label in _UNSUPPORTED_OPS.items():
        if needle in text and label not in found:
            found.append(label)
    return found


def analyze_playground_request(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode") or "prompt").strip().lower()
    prompt = str(payload.get("prompt") or DEFAULT_PROMPT)
    input_text = str(payload.get("input_json") or "").strip()
    training_text = str(payload.get("training_text") or "").strip()
    manifest_text = str(payload.get("manifest_text") or "").strip()
    evaluation_report_text = str(payload.get("evaluation_report_text") or "").strip()

    if mode == "prompt":
        use_llm = bool(payload.get("use_llm", False))
        if _is_neural_prompt(prompt):
            # Route to a network generator when the user requests a neural network.
            # Produces real NETWORK blocks in the .mxai → program.networks populated →
            # network_visual.networks non-empty → Architecture tab visible in Studio.
            # M2-C3: explicit composite hints (residual/layernorm/dropout/complex) use
            # CompositeNetworkGenerator; otherwise the dense path (unchanged).
            from matrixai.training.dense_generator import (  # noqa: PLC0415
                DenseNetworkGenerator,
                DenseNetworkGeneratorError,
            )
            llm_schema = _dense_llm_schema(prompt) if use_llm else {}
            llm_warning = llm_schema.pop("_llm_warning", None)
            # M8-B1: the LLM may propose the architecture type + a rationale.
            # Pop them out so they don't reach the generators (not kwargs).
            llm_architecture = llm_schema.pop("architecture", None)
            llm_rationale = llm_schema.pop("rationale", None)
            # M2 v2 C5: the LLM may declare high-cardinality categoricals for native
            # EMBEDDING. They route to the composite generator (embeddings need it).
            llm_categoricals = llm_schema.pop("categorical_fields", None) or {}
            llm_kwargs: dict[str, Any] = llm_schema

            # M2-C3: robustness notices (the downloadable Studio gets unknown prompts)
            gen_warnings: list[str] = []
            # GEN C5: the LLM threshold is aligned with _ONEHOT_MAX. A low-vocab
            # proposal is one-hot territory (a bare count has no values to one-hot),
            # so it must not drag the prompt to the composite path — filter it here
            # (routing) with a visible warning; the generator applies the same
            # policy for direct callers (invariant 5).
            _low_vocab = {f: v for f, v in llm_categoricals.items() if v <= _ONEHOT_MAX}
            if _low_vocab:
                llm_categoricals = {f: v for f, v in llm_categoricals.items()
                                    if v > _ONEHOT_MAX}
                gen_warnings.append(
                    "Categóricas propuestas por el LLM ignoradas (vocab ≤ "
                    f"{_ONEHOT_MAX}, territorio one-hot): "
                    + ", ".join(f"{f} ({v})" for f, v in _low_vocab.items())
                    + ". Decláralas en el prompt como Categorical[valores...] para "
                    "one-hot con valores humanos."
                )
            is_seq = _prompt_is_sequence(prompt)
            if is_seq:
                gen_warnings.append(
                    "Secuencias/series temporales no soportadas aún en el Studio; "
                    "se genera un modelo tabular."
                )
            unsupported = _prompt_unsupported_ops(prompt)
            if unsupported:
                gen_warnings.append(
                    "Operaciones no soportadas (se omiten): " + ", ".join(unsupported) + "."
                )
            # GEN C5 (diferido de C2): a PROMPT-declared categorical beyond one-hot
            # territory (> _ONEHOT_MAX values) needs the embedding path — the dense
            # generator would leave it scalar. Route it to the composite generator,
            # which materializes it as EMBEDDING with the human vocab persisted.
            from matrixai.generation import parse_field_specs  # noqa: PLC0415
            _prompt_highcard = any(
                f.kind == "categorical" and f.values and len(f.values) > _ONEHOT_MAX
                for f in parse_field_specs(prompt).fields
            )
            # Composite when the prompt hints at it OR the LLM (M8-B1) proposes
            # 'residual' OR the LLM (M2 v2 C5) declares categoricals for EMBEDDING
            # OR the prompt declares a high-cardinality categorical (GEN C5);
            # never for sequence prompts (flat-CSV pipeline, v1).
            want_composite = (
                _prompt_wants_composite(prompt)
                or llm_architecture == "residual"
                or bool(llm_categoricals)
                or _prompt_highcard
            ) and not is_seq
            try:
                if want_composite:
                    from matrixai.training.composite_generator import (  # noqa: PLC0415
                        CompositeNetworkGenerator,
                    )
                    # M2 v2 C5: LLM-declared high-cardinality categoricals become native
                    # EMBEDDING blocks (EMBEDDING+CONCAT for vocab > _ONEHOT_MAX, GEN C5).
                    # Small/ordered features stay scalar; one-hot remains the editor path.
                    # force_residual ensures a residual block when the prompt asks for one.
                    comp_kwargs = {
                        k: v for k, v in llm_kwargs.items()
                        if k in ("input_fields", "labels", "network_name", "input_name")
                    }
                    # A categorical must be a real VECTOR field, or its EMBEDDING would
                    # reference a column that does not exist. When the LLM gave explicit
                    # fields, fold any missing categorical into them; otherwise (fields
                    # invented by the generator) drop the orphaned categoricals.
                    if llm_categoricals:
                        if comp_kwargs.get("input_fields"):
                            fields_list = list(comp_kwargs["input_fields"])
                            for cf in llm_categoricals:
                                if cf not in fields_list:
                                    fields_list.append(cf)
                            comp_kwargs["input_fields"] = fields_list
                        else:
                            llm_categoricals = {}
                    gen = CompositeNetworkGenerator().generate(
                        prompt,
                        categorical_fields=llm_categoricals,
                        force_residual=_prompt_wants_composite(prompt) or llm_architecture == "residual",
                        **comp_kwargs,
                    )
                    gen_source = "composite_generator"
                    llm_used = bool(comp_kwargs) or bool(llm_categoricals)
                else:
                    gen = DenseNetworkGenerator().generate(prompt, **llm_kwargs)
                    gen_source = "dense_generator"
                    llm_used = bool(llm_kwargs)
                result = _result_from_mxai(
                    "prompt",
                    gen.mxai_text,
                    input_text,
                    gen.training_text,
                    manifest_text,
                    evaluation_report_text,
                    training_source_override="generated",
                    dataset_template_text=gen.dataset_template_text,
                )
                result["supervision_source"] = gen_source
                result["prompt"] = prompt
                result["semantic_text"] = ""
                result["pipeline_stages"] = _dense_pipeline_stages(result, generator_name=gen_source)
                result["llm_schema_used"] = llm_used
                # GEN C2: categoricals the generator materialized as one-hot from the
                # prompt ({campo: [valores humanos]}). Source of truth for the Studio
                # (no schema editor needed) and the export's field_categories.
                result["field_categories"] = dict(getattr(gen, "field_categories", {}) or {})
                # GEN C3: scalar ranges + boolean/integer types declared in the prompt.
                # Metadata only (never written into the .mxai VECTOR type — training
                # data is normalized to [0,1]). Source of truth for the Studio/export.
                result["field_ranges"] = dict(getattr(gen, "field_ranges", {}) or {})
                result["field_types"] = dict(getattr(gen, "field_types", {}) or {})
                # M8-B1: record who chose the architecture + the LLM's rationale,
                # for auditability. The deterministic sanitizer (A1) still governs.
                _emitted_embeddings = bool(getattr(gen, "embeddings", []))
                # GEN C6: label the architecture ACTUALLY generated, not the routing
                # intent. Routing may send a prompt to the composite generator and
                # still get a plain dense network back (e.g. every proposed
                # categorical was rejected by validation) — showing "composite
                # (embeddings)" for an all-Scalar dense model was the Colab debt.
                _emitted_blocks = bool(getattr(gen, "blocks", []))
                arch_kind = (
                    "residual" if _emitted_blocks
                    else "composite" if (getattr(gen, "is_composite", False) or _emitted_embeddings)
                    else "dense"
                )
                result["architecture_decision"] = {
                    "kind": arch_kind,
                    # GEN C5 audit: "prompt_types" = routed composite because the
                    # PROMPT declared a high-cardinality categorical (no LLM, no
                    # residual hints) — before, this showed up as "default".
                    "source": ("llm" if (llm_architecture or llm_categoricals)
                               else "prompt_hints" if _prompt_wants_composite(prompt)
                               else "prompt_types" if _prompt_highcard
                               else "default"),
                    "rationale": llm_rationale or "",
                }
                if llm_rationale:
                    gen_warnings.append(f"Arquitectura ({result['architecture_decision']['kind']}, "
                                        f"propuesta por el LLM): {llm_rationale}")
                # M2 v2 C5 / GEN C5 audit: surface the native embeddings WITH their
                # real origin — an embedding from a prompt-declared Categorical[...]
                # must not be attributed to the LLM (it fires with use_llm=False).
                if _emitted_embeddings:
                    _gen_cats = getattr(gen, "field_categories", {}) or {}
                    _emb_fields = ", ".join(
                        f"{e['field']} (vocab {e['vocab']}, dim {e['dim']}, "
                        + ("declarada en el prompt" if e["field"] in _gen_cats
                           else "propuesta por el LLM" if e["field"] in llm_categoricals
                           else "auto-detectada por heurística") + ")"
                        for e in gen.embeddings
                    )
                    gen_warnings.append(
                        "Categóricas con EMBEDDING nativo: " + _emb_fields + "."
                    )
                notes = list(gen_warnings)
                # M8-A1: the generator's own warnings include the architecture
                # sanitizer notes (widened ReLU bottlenecks) — surface them.
                notes.extend(getattr(gen, "warnings", []) or [])
                if llm_warning:
                    notes.append(llm_warning)
                if notes:
                    for stage in result["pipeline_stages"]:
                        if stage.get("name") in ("dense_generator", "composite_generator"):
                            stage.setdefault("warnings", []).extend(notes)
                            stage["status"] = "warning"
                            break
            except DenseNetworkGeneratorError:
                # Fall back to PromptSupervisor if the generator cannot handle the prompt.
                report = PromptSupervisor().supervise_prompt(prompt, force_deterministic=not use_llm)
                result = _result_from_supervision(
                    mode, report.to_dict(), input_text, "", manifest_text, evaluation_report_text,
                )
                result["llm_schema_used"] = False
        else:
            report = PromptSupervisor().supervise_prompt(prompt, force_deterministic=not use_llm)
            result = _result_from_supervision(
                mode,
                report.to_dict(),
                input_text,
                "",  # prompt mode: don't carry over stale training_text from previous model
                manifest_text,
                evaluation_report_text,
            )
            result["llm_schema_used"] = False
        result["llm_mode"] = _detect_llm_mode()
        # In prompt mode a fresh model is generated — clear any stale training/mxai artefacts
        # so the JS overwrites the textareas rather than echoing stale Email/Risk content.
        result.setdefault("training_text", "")
        result.setdefault("mxai", result.get("mxai") or "")
    elif mode == "semantic":
        semantic_text = str(payload.get("semantic_text") or "")
        report = PromptSupervisor().supervise_semantic(
            prompt=prompt,
            semantic_text=semantic_text,
            source="playground:semantic",
        )
        result = _result_from_supervision(
            mode,
            report.to_dict(),
            input_text,
            training_text,
            manifest_text,
            evaluation_report_text,
        )
    elif mode == "mxai":
        mxai_text = str(payload.get("mxai_text") or "")
        result = _result_from_mxai(
            mode,
            mxai_text,
            input_text,
            training_text,
            manifest_text,
            evaluation_report_text,
        )
    else:
        result = {
            "ok": False,
            "mode": mode,
            "error": "mode must be prompt, semantic or mxai",
        }

    result.setdefault("prompt", prompt)
    result.setdefault("input_json", input_text)
    result.setdefault("training_text", training_text)
    result.setdefault("dataset_template_text", "")
    result.setdefault("manifest_text", manifest_text)
    result.setdefault("evaluation_report_text", evaluation_report_text)
    return result


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> int:
    # The technical playground is the open-source community surface and has no
    # license gating. The commercial Studio runs its own server with its own
    # LicenseGuard (in the separate product).
    server = ThreadingHTTPServer((host, port), _handler_class())
    url = f"http://{host}:{server.server_port}"
    print(f"MatrixAI playground running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMatrixAI playground stopped")
    finally:
        server.server_close()
    return 0


def _result_from_supervision(
    mode: str,
    report: dict[str, Any],
    input_text: str,
    training_text: str,
    manifest_text: str,
    evaluation_report_text: str,
) -> dict[str, Any]:
    mxai_text = str(report.get("mxai") or "")
    result: dict[str, Any] = {
        "ok": bool(report.get("accepted")),
        "mode": mode,
        "accepted": bool(report.get("accepted")),
        "prompt": report.get("prompt", ""),
        "source": report.get("source", ""),
        "semantic_text": report.get("semantic_text", ""),
        "checks": report.get("checks", []),
        "plan": report.get("plan"),
        "mxai": mxai_text,
        "program": report.get("program"),
        "supervision_source": report.get("supervision_source", "deterministic"),
        "llm_provider": report.get("llm_provider", ""),
        "llm_model": report.get("llm_model", ""),
        "call_traces_summary": report.get("call_traces_summary", []),
        "fallback_reason": report.get("fallback_reason", ""),
    }
    supervision_source = result.get("supervision_source", "deterministic")
    llm_provider = result.get("llm_provider", "")
    llm_model = result.get("llm_model", "")

    if mxai_text:
        supervision_checks = list(result["checks"])
        prog_artifacts = _program_artifacts(
            mxai_text,
            input_text,
            training_text,
            manifest_text,
            evaluation_report_text,
            generation_prompt=str(report.get("prompt") or ""),
        )
        program_ok = bool(prog_artifacts.get("ok"))
        prog_artifacts["checks"] = supervision_checks + list(prog_artifacts.get("checks", []))
        prog_artifacts["ok"] = bool(report.get("accepted")) and program_ok and _checks_ok(prog_artifacts["checks"])
        result.update(prog_artifacts)

    all_checks = list(result.get("checks", []))
    result["pipeline_stages"] = _build_pipeline_stages(all_checks, _SUPERVISION_STAGES)
    result["artifacts"] = _build_artifacts(
        mode=mode,
        source=result.get("source", ""),
        semantic_text=result.get("semantic_text", ""),
        mxai_text=result.get("mxai", ""),
        compiled_python=result.get("compiled_python", ""),
        supervision_source=supervision_source,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    return result


def _result_from_mxai(
    mode: str,
    mxai_text: str,
    input_text: str,
    training_text: str,
    manifest_text: str,
    evaluation_report_text: str,
    *,
    training_source_override: str = "",
    dataset_template_text: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "mode": mode,
        "accepted": False,
        "mxai": mxai_text,
        "checks": [],
    }
    try:
        result.update(
            _program_artifacts(
                mxai_text,
                input_text,
                training_text,
                manifest_text,
                evaluation_report_text,
                training_source_override=training_source_override,
                dataset_template_text_override=dataset_template_text,
            )
        )
    except Exception as exc:  # noqa: BLE001
        result["checks"].append({"name": "parser", "ok": False, "errors": [str(exc)], "warnings": []})
        result["error"] = str(exc)
        return result
    result["ok"] = bool(result.get("ok")) and _checks_ok(result.get("checks", []))
    result["accepted"] = result["ok"]
    result["pipeline_stages"] = _build_pipeline_stages(result.get("checks", []), _MXAI_STAGES)
    result["artifacts"] = _build_artifacts(
        mode=mode,
        source="playground:mxai",
        semantic_text="",
        mxai_text=mxai_text,
        compiled_python=result.get("compiled_python", ""),
        supervision_source="",
        llm_provider="",
        llm_model="",
    )
    return result


def _program_artifacts(
    mxai_text: str,
    input_text: str,
    training_text: str = "",
    manifest_text: str = "",
    evaluation_report_text: str = "",
    generation_prompt: str = "",
    *,
    training_source_override: str = "",
    dataset_template_text_override: str = "",
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    program = parse_text(mxai_text)
    checks.append({"name": "parser", "ok": True, "errors": [], "warnings": []})

    verifier_result = VerifierAgent().verify(program)
    checks.append(
        {
            "name": "verifier_agent",
            "ok": verifier_result.ok,
            "errors": verifier_result.errors,
            "warnings": verifier_result.warnings,
        }
    )

    safety_warnings = SafetyAgent().review(program)
    sandbox_report = SandboxPolicy.mvp_simulate_only().review(program)
    checks.append(
        {
            "name": "safety_agent",
            "ok": not safety_warnings,
            "errors": safety_warnings,
            "warnings": [],
        }
    )

    type_result = check_program_types(program)
    checks.append(
        {
            "name": "typecheck",
            "ok": type_result.ok,
            "errors": type_result.errors,
            "warnings": type_result.warnings,
        }
    )

    compiled_python = ""
    try:
        compiled_python = PythonBackendCompiler().compile(program)
        checks.append({"name": "python_compiler", "ok": True, "errors": [], "warnings": []})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "python_compiler", "ok": False, "errors": [str(exc)], "warnings": []})

    # P3/P8 backend contract analysis — trainability and parameter manifest
    backend_contract: dict[str, Any] = {}
    try:
        bc_report = BackendContractAnalyzer().analyze(program)
        backend_contract = bc_report.to_dict()
        checks.append({
            "name": "backend_contract",
            "ok": bc_report.ok,
            "errors": bc_report.parameter_errors,
            "warnings": bc_report.warnings,
        })
    except Exception as exc:  # noqa: BLE001
        backend_contract = {"error": str(exc)}

    dataset_template_text = dataset_template_text_override
    training_source = training_source_override or ("provided" if training_text else "none")
    generated_training_error = ""
    if not training_text:
        inferred_training = _infer_example_text(program.project, "training")
        if inferred_training:
            training_text = inferred_training
            training_source = "inferred"
            dataset_template_text = _dataset_text_from_training_text(training_text)
        else:
            generated = _generate_training_from_mxai(
                mxai_text,
                prompt=generation_prompt or f"supervisado para {program.project}",
            )
            if generated.get("ok"):
                training_text = str(generated.get("training_text") or "")
                dataset_template_text = str(generated.get("dataset_template_text") or "")
                training_source = "generated"
            else:
                generated_training_error = str(generated.get("error") or "")
    if training_text and not dataset_template_text:
        dataset_template_text = _dataset_text_from_training_text(training_text)
    if not manifest_text:
        manifest_text = _infer_example_text(program.project, "manifest")

    training_artifacts = _training_artifacts(
        training_text,
        manifest_text,
        training_source,
        mxai_text=mxai_text,
        dataset_template_text=dataset_template_text,
    )
    if generated_training_error and not training_artifacts.get("available"):
        training_artifacts["generation_error"] = generated_training_error
    evaluation_artifacts = _evaluation_artifacts(evaluation_report_text)

    artifacts: dict[str, Any] = {
        "checks": checks,
        "program": program.to_dict(),
        "graph_mermaid": graph_program(program, "mermaid"),
        "typecheck": type_result.to_dict(),
        "compiled_python": compiled_python,
        "compiled_python_bytes": len(compiled_python.encode("utf-8")),
        "backend_contract": backend_contract,
        "training_text": training_text,
        "dataset_template_text": dataset_template_text,
        "manifest_text": manifest_text,
        "evaluation_report_text": evaluation_report_text,
        "training_artifacts": training_artifacts,
        "evaluation_artifacts": evaluation_artifacts,
        "visual_model": _visual_model(
            program,
            sandbox_report.to_dict(),
            training_artifacts,
            evaluation_artifacts,
            backend_contract,
        ),
    }

    if input_text:
        input_data = json.loads(input_text)
        run_result = MatrixAIRuntime().run(program, input_data)
        run_result["audit"] = AuditorAgent().explain(run_result)
        diagnose = diagnose_runtime_compiler(program, input_data)
        artifacts["run_result"] = run_result
        artifacts["diagnose"] = diagnose.to_dict()
        checks.append(
            {
                "name": "runtime_compiler_diagnose",
                "ok": diagnose.ok,
                "errors": diagnose.mismatches,
                "warnings": [],
            }
        )

    artifacts["ok"] = (
        _checks_ok(checks)
        and (not training_artifacts.get("available") or bool(training_artifacts.get("ok")))
        and (not evaluation_artifacts.get("available") or bool(evaluation_artifacts.get("ok")))
    )
    artifacts["workflow"] = _workflow_steps(
        program,
        checks,
        bool(input_text),
        training_artifacts,
        evaluation_artifacts,
    )
    # P21 — Composición panel
    artifacts["composition"] = _composition_panel(program)
    return artifacts


def _checks_ok(checks: list[dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in checks)


def _composition_panel(program: Any) -> dict[str, Any]:
    """Return composition metadata for the Studio 'Composición' panel."""
    imports = getattr(program, "imports", [])
    if not imports:
        return {"has_imports": False, "components": []}
    return {
        "has_imports": True,
        "components": [
            {
                "alias": imp.alias,
                "registry_name": imp.registry_name,
                "version": imp.version,
                "mode": imp.mode,
                "evaluation_report_link": f"matrixai_registry/entries/{imp.registry_name}/{imp.version}/evaluation_report.json",
            }
            for imp in imports
        ],
    }


def _serialize_network(net: Any, total_input_fields: int = 0) -> dict[str, Any]:
    """Serialize a NetworkSpec into a frontend-ready architecture dict."""
    if net.kind == "dense_network":
        layers_info = []
        resolved = (net.layers[0].input_shape[0]
                    if net.layers and net.layers[0].input_shape
                    and net.layers[0].input_shape[0] > 0 else 0)
        dim = resolved if resolved > 0 else total_input_fields
        current_dim = dim
        total_params = 0
        arch_parts = [f"Input[{dim}]"]
        for layer in net.layers:
            p = current_dim * layer.units + layer.units if current_dim else 0
            total_params += p
            layers_info.append({
                "index": layer.index,
                "layer_type": "Dense",
                "units": layer.units,
                "activation": layer.activation,
                "params": p,
            })
            arch_parts.append(f"Dense({layer.units},{layer.activation})")
            current_dim = layer.units
        output_label = net.output_type_str.split("[")[0] if net.output_type_str else "Output"
        arch_parts.append(output_label)
        return {
            "name": net.name, "kind": net.kind,
            "architecture_text": " → ".join(arch_parts),
            "input_dim": dim, "total_params": total_params,
            "output_type": net.output_type_str,
            "layers": layers_info, "embeddings": [], "blocks": [],
        }
    elif net.kind == "composite_network":
        top_layers = getattr(net, "top_layers", [])
        blocks = getattr(net, "blocks", [])
        embeddings = getattr(net, "embeddings", [])
        emb_info = [
            {"name": e.name, "vocab": e.vocab, "dim": e.dim, "params": e.vocab * e.dim}
            for e in embeddings
        ]
        total_params = sum(e["params"] for e in emb_info)
        layers_info = [
            {
                "index": layer.index,
                "layer_type": layer.layer_type,
                "units": getattr(layer, "units", 0),
                "activation": (getattr(layer, "activation", "")
                               or getattr(layer, "activation_kind", "")),
                "params": 0,
            }
            for layer in top_layers
        ]
        blocks_info = [
            {
                "name": blk.name,
                "layers": [
                    {"type": bl.layer_type, "units": getattr(bl, "units", 0), "params": 0}
                    for bl in blk.layers
                ],
                "residual_from": blk.residual_from,
                "params": 0,
            }
            for blk in blocks
        ]
        # input_dim: "sequence" for embedding models; numeric dim otherwise.
        # net.input is the entity name ("Tokens", "Input"), not a dimension — only use it in arch text.
        if embeddings:
            input_dim: Any = "sequence"
        else:
            first_dense = next(
                (l for l in top_layers if getattr(l, "layer_type", "") == "Dense"
                 and l.input_shape and l.input_shape[0] > 0), None
            )
            input_dim = first_dense.input_shape[0] if first_dense else total_input_fields or 0

        parts = [f"Input[{net.input}]"]
        for e in emb_info:
            parts.append(f"Emb({e['vocab']},{e['dim']})")
        for layer in top_layers:
            if layer.layer_type == "Dense":
                parts.append(f"Dense({layer.units},{layer.activation})")
            elif layer.layer_type == "LayerNorm":
                parts.append("LN")
            else:
                parts.append(layer.layer_type)
        for blk in blocks:
            inner = "+".join(bl.layer_type for bl in blk.layers[:2])
            res = "+Res" if blk.residual_from else ""
            parts.append(f"Block[{inner}]{res}")
        output_label = net.output_type_str.split("[")[0] if net.output_type_str else "Output"
        parts.append(output_label)
        return {
            "name": net.name, "kind": net.kind,
            "architecture_text": " → ".join(parts),
            "input_dim": input_dim, "total_params": total_params,
            "output_type": net.output_type_str,
            "layers": layers_info, "embeddings": emb_info, "blocks": blocks_info,
        }
    return {
        "name": net.name, "kind": net.kind, "architecture_text": net.name,
        "input_dim": 0, "total_params": 0, "output_type": "",
        "layers": [], "embeddings": [], "blocks": [],
    }


def _visual_model(
    program: Any,
    sandbox_report: dict[str, Any],
    training_artifacts: dict[str, Any] | None = None,
    evaluation_artifacts: dict[str, Any] | None = None,
    backend_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = []
    for vector in program.vectors:
        for field in vector.fields:
            type_spec = vector.field_types.get(field)
            inputs.append(
                {
                    "vector": vector.name,
                    "field": field,
                    "type": type_spec.name if type_spec else "Number",
                    "range": _range_label(type_spec),
                }
            )

    calculations = [
        {
            "name": function.name,
            "output": function.output,
            "kind": function.semantic.kind,
            "expression": function.expression,
            "inputs": list(function.semantic.inputs),
        }
        for function in program.functions
    ]

    distributions = [
        {
            "name": distribution.name,
            "type": distribution.distribution_type,
            "source": distribution.source,
        }
        for distribution in program.distributions
    ]

    actions = []
    decisions_by_action = {
        decision["action"]: decision for decision in sandbox_report.get("decisions", [])
    }
    for action in program.actions:
        decision = decisions_by_action.get(action.name, {})
        actions.append(
            {
                "name": action.name,
                "when": action.when,
                "call": action.call,
                "policy": action.policy,
                "allowed": bool(decision.get("allowed", False)),
                "simulated": action.call.startswith("simulated."),
                "capabilities": list(decision.get("capabilities", [])),
                "reasons": list(decision.get("reasons", [])),
            }
        )

    return {
        "project": program.project,
        "overview": {
            "vectors": len(program.vectors),
            "inputs": len(inputs),
            "functions": len(program.functions),
            "distributions": len(program.distributions),
            "actions": len(program.actions),
            "graph_nodes": len(program.graph.nodes),
            "graph_edges": len(program.graph.edges),
        },
        "inputs": inputs,
        "calculations": calculations,
        "distributions": distributions,
        "actions": actions,
        "networks": [
            _serialize_network(n, sum(len(list(v.fields)) for v in program.vectors))
            for n in getattr(program, "networks", [])
        ],
        "security": sandbox_report,
        "training": training_artifacts or {"available": False},
        "evaluation": evaluation_artifacts or {"available": False},
        "backend_contract": backend_contract or {},
        "serving": {
            "command": (
                f"python3 -m matrixai serve examples/{program.project}.mxai"
                " --host 127.0.0.1 --port 8000 --api-key prueba-local"
            ),
            "docs_url": "http://localhost:8000/docs",
            "openapi_url": "http://localhost:8000/openapi.json",
            "health_url": "http://localhost:8000/health",
            "predict_url": "http://localhost:8000/predict",
            "auth": "Authorization: Bearer prueba-local",
            "curl": (
                "curl -X POST http://localhost:8000/predict "
                "-H 'Authorization: Bearer prueba-local' "
                "-H 'Content-Type: application/json' "
                "-d '{\"Email\": {\"urgency\": 0.8}}'"
            ),
        },
    }


def _range_label(type_spec: Any | None) -> str:
    if type_spec is None or getattr(type_spec, "range", None) is None:
        return ""
    range_spec = type_spec.range
    minimum = "-inf" if range_spec.minimum is None else range_spec.minimum
    maximum = "+inf" if range_spec.maximum is None else range_spec.maximum
    left = "[" if range_spec.inclusive_min else "("
    right = "]" if range_spec.inclusive_max else ")"
    return f"{left}{minimum}, {maximum}{right}"


def _workflow_steps(
    program: Any,
    checks: list[dict[str, Any]],
    has_run: bool,
    training_artifacts: dict[str, Any],
    evaluation_artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    check_map = {check["name"]: check for check in checks}
    parser_ok = bool(check_map.get("parser", {}).get("ok"))
    verifier_ok = bool(check_map.get("verifier_agent", {}).get("ok"))
    type_ok = bool(check_map.get("typecheck", {}).get("ok"))
    safety_ok = bool(check_map.get("safety_agent", {}).get("ok"))
    compiler_ok = bool(check_map.get("python_compiler", {}).get("ok"))
    run_ok = bool(check_map.get("runtime_compiler_diagnose", {}).get("ok")) if has_run else False
    training_available = bool(training_artifacts.get("available"))
    training_ok = bool(training_artifacts.get("ok"))
    evaluation_available = bool(evaluation_artifacts.get("available"))
    evaluation_ok = bool(evaluation_artifacts.get("ok"))

    def status(ok: bool, warnings: list[Any] | None = None) -> str:
        if not ok:
            return "blocked"
        return "warning" if warnings else "valid"

    return [
        {
            "id": "model",
            "label": "Modelo",
            "status": status(parser_ok and verifier_ok, check_map.get("verifier_agent", {}).get("warnings", [])),
            "summary": f"{program.project}: {len(program.graph.nodes)} nodos y {len(program.graph.edges)} conexiones.",
        },
        {
            "id": "types",
            "label": "Entradas",
            "status": status(type_ok, check_map.get("typecheck", {}).get("warnings", [])),
            "summary": f"{sum(len(vector.fields) for vector in program.vectors)} campos tipados en {len(program.vectors)} vector(es).",
        },
        {
            "id": "security",
            "label": "Seguridad",
            "status": status(safety_ok),
            "summary": "Acciones revisadas bajo politica simulate_only.",
        },
        {
            "id": "training",
            "label": "Entrenamiento",
            "status": (
                status(training_ok, training_artifacts.get("warnings", []))
                if training_available
                else "pending"
            ),
            "summary": (
                (
                    "Contrato .mxtrain y dataset validados, sin manifiesto versionado."
                    if training_artifacts.get("warnings")
                    else "Contrato .mxtrain y dataset validados."
                )
                if training_available and training_ok
                else (
                    "Contrato .mxtrain o manifiesto de dataset bloqueado."
                    if training_available
                    else "Carga o infiere un .mxtrain para ver artefactos P4."
                )
            ),
        },
        {
            "id": "evaluation",
            "label": "Evaluacion",
            "status": status(evaluation_ok) if evaluation_available else "pending",
            "summary": (
                "Reporte de evaluacion disponible con metricas y matriz de confusion."
                if evaluation_available and evaluation_ok
                else "Carga un evaluation_report.json para revisar metricas finales."
            ),
        },
        {
            "id": "runtime",
            "label": "Ejecucion",
            "status": status(run_ok) if has_run else "pending",
            "summary": "Runtime comparado con compiler." if has_run else "Introduce Input JSON para ejecutar y comparar.",
        },
        {
            "id": "serving",
            "label": "API",
            "status": status(compiler_ok),
            "summary": "Listo para servir por HTTP con Bearer Auth si el modelo valida.",
        },
    ]


def _training_artifacts(
    training_text: str,
    manifest_text: str = "",
    source: str = "provided",
    *,
    mxai_text: str = "",
    dataset_template_text: str = "",
) -> dict[str, Any]:
    if not training_text:
        return {
            "available": False,
            "ok": False,
            "source": "none",
            "message": "No .mxtrain contract was provided or inferred for this model.",
        }

    try:
        training = parse_training_text(training_text)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": True,
            "ok": False,
            "source": source,
            "errors": [str(exc)],
            "training_text": training_text,
        }

    if source == "generated" and mxai_text and dataset_template_text:
        verification = _verify_training_text_package(training, mxai_text, dataset_template_text)
        dataset_summary = _dataset_summary_from_text(dataset_template_text, training.dataset.source)
    else:
        verification = TrainingVerifier().verify(training, PROJECT_ROOT).to_dict()
        dataset_summary = _dataset_summary(verification.get("dataset_path") or "")
    manifest_summary = _manifest_summary(manifest_text)
    warnings = []
    if source == "generated":
        warnings.append("Contrato .mxtrain y plantilla CSV generados desde el modelo actual; valida datos reales antes de entrenar.")
    if not manifest_summary:
        warnings.append(
            "Dataset CSV validated, but no dataset manifest was provided for hashes and splits."
        )
    return {
        "available": True,
        "ok": bool(verification.get("ok")) and (not manifest_summary or bool(manifest_summary.get("ok"))),
        "source": source,
        "warnings": warnings,
        "training_text": training_text,
        "dataset_template_text": dataset_template_text,
        "spec": training.to_dict(),
        "verification": verification,
        "dataset": dataset_summary,
        "manifest": manifest_summary,
    }


def _verify_training_text_package(training: Any, mxai_text: str, csv_text: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model_rel = Path(training.model)
        (tmp / model_rel.name).write_text(mxai_text, encoding="utf-8")
        if str(model_rel.parent) not in (".", ""):
            model_dest = tmp / model_rel
            model_dest.parent.mkdir(parents=True, exist_ok=True)
            model_dest.write_text(mxai_text, encoding="utf-8")

        csv_rel = Path(training.dataset.source)
        (tmp / csv_rel.name).write_text(csv_text, encoding="utf-8")
        if str(csv_rel.parent) not in (".", ""):
            csv_dest = tmp / csv_rel
            csv_dest.parent.mkdir(parents=True, exist_ok=True)
            csv_dest.write_text(csv_text, encoding="utf-8")

        return TrainingVerifier().verify(training, base_path=tmp).to_dict()


def _dataset_summary_from_text(csv_text: str, path_text: str = "") -> dict[str, Any]:
    if not csv_text:
        return {"available": False, "path": path_text}
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    except csv.Error as exc:
        return {"available": False, "path": path_text, "error": str(exc)}
    return {
        "available": True,
        "path": path_text,
        "columns": columns,
        "rows": len(rows),
        "preview": rows[:3],
    }


def _evaluation_artifacts(evaluation_report_text: str) -> dict[str, Any]:
    if not evaluation_report_text:
        return {
            "available": False,
            "ok": False,
            "message": "No evaluation_report.json was provided for this model.",
        }
    try:
        report = json.loads(evaluation_report_text)
    except json.JSONDecodeError as exc:
        return {
            "available": True,
            "ok": False,
            "errors": [f"Invalid evaluation_report.json: {exc}"],
            "evaluation_report_text": evaluation_report_text,
        }

    required = ["rows", "loss", "accuracy", "labels", "confusion_matrix", "per_label"]
    missing = [key for key in required if key not in report]
    backend = report.get("backend") or {"target": "stdlib"}
    return {
        "available": True,
        "ok": not missing,
        "errors": [f"Missing evaluation fields: {', '.join(missing)}"] if missing else [],
        "evaluation_report_text": evaluation_report_text,
        "report": report,
        "backend": backend,
        "report_path": report.get("report_path") or "evaluation_report.json",
    }


def _dataset_summary(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {"available": False}
    path = Path(path_text)
    if not path.exists():
        return {"available": False, "path": path_text}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    return {
        "available": True,
        "path": str(path),
        "columns": columns,
        "rows": len(rows),
        "preview": rows[:3],
    }


def _manifest_summary(manifest_text: str) -> dict[str, Any]:
    if not manifest_text:
        return {}
    try:
        data = json.loads(manifest_text)
        manifest = DatasetManifest.from_dict(data)
        verification = verify_dataset_manifest(manifest, PROJECT_ROOT).to_dict()
        if not verification.get("ok"):
            examples_verification = verify_dataset_manifest(manifest, PROJECT_ROOT / "examples").to_dict()
            if examples_verification.get("ok"):
                verification = examples_verification
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "ok": False, "errors": [str(exc)]}
    return {
        "available": True,
        "ok": bool(verification.get("ok")),
        "manifest": manifest.to_dict(),
        "verification": verification,
    }


def _infer_example_text(project: str, artifact: str) -> str:
    example_id = PROJECT_EXAMPLE_INDEX.get(project)
    if not example_id:
        return ""
    path_text = EXAMPLES[example_id].get(artifact) or ""
    if not path_text:
        return ""
    path = PROJECT_ROOT / path_text
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _example_payload(example_id: str) -> dict[str, Any]:
    try:
        config = EXAMPLES[example_id]
    except KeyError as exc:
        raise ValueError(f"Unknown example: {example_id}") from exc

    payload = {
        "id": example_id,
        "label": config["label"],
        "mode": "mxai",
        "mxai_text": (PROJECT_ROOT / config["model"]).read_text(encoding="utf-8"),
        "training_text": (PROJECT_ROOT / config["training"]).read_text(encoding="utf-8"),
        "input_json": (PROJECT_ROOT / config["input"]).read_text(encoding="utf-8") if config.get("input") else "",
        "manifest_text": "",
        "evaluation_report_text": "",
    }
    if config.get("manifest"):
        payload["manifest_text"] = (PROJECT_ROOT / config["manifest"]).read_text(encoding="utf-8")
    if config.get("evaluation"):
        payload["evaluation_report_text"] = (PROJECT_ROOT / config["evaluation"]).read_text(encoding="utf-8")
    return payload


# Endpoints that must keep responding even when the license is invalid, so the
# Docker healthcheck (and basic liveness probes) keep working. The technical
# playground is the open-source community surface; the commercial Studio (with
# its own /api/studio gating) lives in the separate product.
_LICENSE_EXEMPT_PATHS = {"/health"}


def _handler_class(guard: Any = None):
    class PlaygroundHandler(BaseHTTPRequestHandler):
        server_version = "MatrixAIPlayground/0.1"
        license_guard = guard

        def _license_blocks(self) -> bool:
            """Return True (and emit a 503) if the license is invalid for this path.

            Healthcheck/liveness paths are always allowed. Any other path is
            blocked with HTTP 503 when the guard reports the license invalid.
            """
            if self.license_guard is None:
                return False
            # Strip query string for the exemption check.
            path = self.path.split("?", 1)[0]
            if path in _LICENSE_EXEMPT_PATHS:
                return False
            if self.license_guard.is_valid():
                return False
            self._send_json(
                {
                    "error": "license_invalid",
                    "message": "License validation failed. Please restart with start.sh.",
                },
                status=503,
            )
            return True

        def do_GET(self) -> None:  # noqa: N802
            if self._license_blocks():
                return
            if self.path in {"/", "/studio", "/studio/", "/expert", "/expert/", "/index.html"}:
                # The technical playground (community). The commercial Studio is a
                # separate product (matrixaistudio) and is not served from here.
                self._send_text(_INDEX_HTML, "text/html; charset=utf-8")
                return
            if self.path == "/api/defaults":
                self._send_json(
                    {
                        "prompt": DEFAULT_PROMPT,
                        "input_json": json.dumps(DEFAULT_INPUT, indent=2, ensure_ascii=False),
                        "examples": [
                            {"id": key, "label": value["label"]}
                            for key, value in EXAMPLES.items()
                        ],
                    }
                )
                return
            if self.path.startswith("/api/example/"):
                example_id = self.path.rsplit("/", 1)[-1]
                try:
                    self._send_json(_example_payload(example_id))
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=404)
                return
            if self.path.startswith("/api/train-status/"):
                job_id = self.path.rsplit("/", 1)[-1]
                result = _get_job_status(job_id)
                self._send_json(result, status=200 if result.get("ok") else 404)
                return
            self.send_error(404, "Not found")

        def _origin_is_local(self) -> bool:
            """Return True if Origin is absent or bound to localhost/127.0.0.1.

            Browsers always send Origin on cross-origin requests. A missing Origin
            means a same-origin load or a direct tool call (curl) — both safe.
            A present non-local Origin means a cross-origin browser request, which
            could be a CSRF attempt trying to reach the local LLM config endpoint.
            """
            origin = self.headers.get("Origin", "")
            if not origin:
                return True
            return bool(re.match(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$", origin))

        def do_POST(self) -> None:  # noqa: N802
            if self._license_blocks():
                return
            if not self._origin_is_local():
                self._send_json({"ok": False, "error": "forbidden"}, status=403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw) if raw else {}
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"JSON inválido: {exc}"}, status=400)
                return
            if self.path == "/api/analyze":
                try:
                    result = analyze_playground_request(payload)
                    self._send_json(result, status=200 if result.get("ok") else 422)
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)}, status=500)
            elif self.path == "/api/generate-training":
                result = _generate_training_from_mxai(str(payload.get("mxai_text") or ""))
                self._send_json(result, status=200 if result.get("ok") else 422)
            elif self.path == "/api/generate-dataset":
                result = _generate_synthetic_dataset(
                    str(payload.get("mxai_text") or ""),
                    str(payload.get("training_text") or ""),
                    int(payload.get("rows") or 200),
                    int(payload.get("seed") or 42),
                    str(payload.get("mode") or "random"),
                    bool(payload.get("use_llm", False)),
                    field_ranges_override=_coerce_field_ranges(payload.get("field_ranges")) or None,
                    field_types=_coerce_field_types(payload.get("field_types")) or None,
                    field_categories=_coerce_field_categories(payload.get("field_categories")) or None,
                    field_identifiers=_coerce_field_identifiers(payload.get("field_identifiers")) or None,
                )
                self._send_json(result, status=200 if result.get("ok") else 422)
            elif self.path == "/api/suggest-ranges":
                result = _suggest_field_ranges(
                    str(payload.get("mxai_text") or ""),
                    str(payload.get("training_text") or ""),
                )
                self._send_json(result, status=200 if result.get("ok") else 422)
            elif self.path == "/api/validate-csv":
                result = _validate_training_csv(
                    str(payload.get("mxai_text") or ""),
                    str(payload.get("training_text") or ""),
                    str(payload.get("csv_text") or ""),
                    field_ranges=_coerce_field_ranges(payload.get("field_ranges")) or None,
                )
                self._send_json(result, status=200 if result.get("ok") else 422)
            elif self.path == "/api/train":
                epochs_raw = payload.get("epochs_override")
                epochs_override = int(epochs_raw) if epochs_raw is not None else None
                result = _run_playground_training(
                    str(payload.get("mxai_text") or ""),
                    str(payload.get("training_text") or ""),
                    str(payload.get("csv_text") or ""),
                    epochs_override,
                )
                self._send_json(result, status=200 if result.get("ok") else 422)
            elif self.path == "/api/run-with-params":
                result = _playground_run_with_params(
                    str(payload.get("mxai_text") or ""),
                    str(payload.get("params_json") or ""),
                    str(payload.get("input_json") or ""),
                )
                self._send_json(result, status=200 if result.get("ok") else 422)
            elif self.path == "/api/train-start":
                epochs_raw = payload.get("epochs_override")
                epochs_override = int(epochs_raw) if epochs_raw is not None else None
                result = _submit_training_job(
                    str(payload.get("mxai_text") or ""),
                    str(payload.get("training_text") or ""),
                    str(payload.get("csv_text") or ""),
                    epochs_override,
                    field_ranges=_coerce_field_ranges(payload.get("field_ranges")) or None,
                    seed=int(payload.get("seed") or 42),
                )
                self._send_json(result, status=200 if result.get("ok") else 422)
            elif self.path == "/api/train-cancel":
                result = _cancel_job(str(payload.get("job_id") or ""))
                self._send_json(result, status=200 if result.get("ok") else 404)
            elif self.path == "/api/refine":
                result = _refine_prompt(payload)
                self._send_json(result, status=200 if result.get("ok") else 422)
            else:
                self.send_error(404, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PlaygroundHandler


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MatrixAI Playground</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --panel-soft: #eef3f1;
      --text: #17202a;
      --muted: #5f6f7a;
      --line: #d6dee2;
      --accent: #176b87;
      --accent-strong: #0d4d63;
      --ok: #1d7a46;
      --bad: #b42318;
      --warn: #a15c00;
      --ink: #263238;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 750; letter-spacing: 0; }
    .subtitle { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: 320px minmax(360px, 440px) minmax(420px, 1fr);
      min-height: calc(100vh - 58px);
    }
    section { padding: 14px; }
    .rail { border-right: 1px solid var(--line); background: var(--panel-soft); }
    .workbench { border-right: 1px solid var(--line); background: #fbfcfd; }
    .stack { display: grid; gap: 12px; }
    h2 { margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }
    label { display: block; margin: 0 0 6px; color: var(--muted); font-weight: 600; }
    textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    textarea { min-height: 120px; padding: 10px; resize: vertical; }
    select { padding: 8px; font-family: inherit; }
    button {
      border: 0;
      border-radius: 6px;
      padding: 9px 12px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: 0.65; cursor: wait; }
    .secondary {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
    }
    .secondary:hover { background: #edf2f4; }
    .workflow { display: grid; gap: 8px; }
    .step {
      display: grid;
      grid-template-columns: 24px 1fr;
      gap: 8px;
      align-items: start;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }
    .dot {
      width: 12px;
      height: 12px;
      margin-top: 4px;
      border-radius: 50%;
      background: var(--muted);
    }
    .step.valid .dot { background: var(--ok); }
    .step.blocked .dot { background: var(--bad); }
    .step.pending .dot { background: var(--warn); }
    .step.warning .dot { background: var(--warn); }
    .step-title { font-weight: 750; }
    .step-summary { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .demo-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0 12px; }
    .guide { display: grid; gap: 8px; }
    .guide-step {
      width: 100%;
      display: grid;
      grid-template-columns: 12px 1fr;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      text-align: left;
      font-weight: 700;
    }
    .guide-step:hover { background: #edf2f4; }
    .guide-step .dot { margin: 0; }
    .guide-step.valid .dot { background: var(--ok); }
    .guide-step.blocked .dot { background: var(--bad); }
    .guide-step.pending .dot, .guide-step.warning .dot { background: var(--warn); }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: var(--panel);
      min-height: 58px;
    }
    .metric-value { display: block; font-size: 20px; font-weight: 800; }
    .metric-label { display: block; color: var(--muted); font-size: 12px; }
    .table {
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    .table th, .table td {
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      font-size: 12px;
    }
    .table th { color: var(--muted); background: #f8fafb; font-weight: 750; }
    .table tr:last-child td { border-bottom: 0; }
    .callout {
      border-left: 4px solid var(--accent);
      background: var(--panel);
      padding: 10px;
      border-radius: 6px;
    }
    .callout.warn { border-left-color: var(--warn); }
    .callout.bad { border-left-color: var(--bad); }
    .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
    .tab {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      padding: 7px 10px;
      border-radius: 6px;
    }
    .tab.active { border-color: var(--accent); color: var(--accent-strong); font-weight: 700; }
    .status { font-weight: 700; }
    .status.ok { color: var(--ok); }
    .status.bad { color: var(--bad); }
    .status.warn { color: var(--warn); }
    .status.running { color: var(--accent-strong); }
    .header-status { text-align: right; min-width: 190px; }
    .status-detail { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .action-note {
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      padding: 8px 10px;
      background: var(--panel);
      color: var(--muted);
      font-size: 12px;
    }
    .action-note.ok { border-left-color: var(--ok); color: var(--text); }
    .action-note.bad { border-left-color: var(--bad); color: var(--text); }
    .action-note.running { border-left-color: var(--accent); color: var(--text); }
    .diagnostics-list { margin: 6px 0 0; padding-left: 18px; color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    pre {
      margin: 0;
      overflow: auto;
      max-height: 72vh;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .checks { display: grid; gap: 8px; }
    .check {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: var(--panel);
    }
    .check.ok { border-left: 4px solid var(--ok); }
    .check.bad { border-left: 4px solid var(--bad); }
    .check .name { font-weight: 700; }
    .check .detail { color: var(--muted); margin-top: 4px; }
    .pipeline { display: flex; flex-direction: column; gap: 4px; }
    .pipeline-stage { display: flex; align-items: flex-start; gap: 10px; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel); }
    .pipeline-stage.ok { border-left: 4px solid var(--ok); }
    .pipeline-stage.fail { border-left: 4px solid var(--bad); background: #fff5f5; }
    .pipeline-stage.warning { border-left: 4px solid #f0b429; }
    .pipeline-stage.skipped { opacity: 0.45; }
    .pipeline-stage.pending { opacity: 0.55; border-style: dashed; }
    .ps-num { font-size: 11px; color: var(--muted); min-width: 18px; padding-top: 2px; }
    .ps-label { font-weight: 700; font-size: 13px; }
    .ps-status { font-size: 11px; margin-left: auto; padding: 2px 6px; border-radius: 4px; background: var(--panel-soft); }
    .ps-status.ok { background: #e6f4ea; color: #1a6b38; }
    .ps-status.fail { background: #fde8e8; color: #b91c1c; }
    .ps-status.warning { background: #fef9e7; color: #7a5c00; }
    .ps-status.skipped { background: var(--panel-soft); color: var(--muted); }
    .ps-detail { font-size: 12px; color: var(--muted); margin-top: 3px; white-space: pre-wrap; }
    .train-section { margin-top: 16px; }
    .train-section h3 { font-size: 13px; margin: 0 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
    .epoch-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
    .epoch-table th, .epoch-table td { padding: 4px 8px; border: 1px solid var(--line); text-align: right; }
    .epoch-table th { background: var(--panel-soft); text-align: center; }
    .epoch-table tr.best { background: #e6f4ea; }
    .download-btn { font-size: 12px; padding: 4px 10px; margin: 3px 4px 3px 0; border: 1px solid var(--line); border-radius: 4px; background: var(--panel-soft); cursor: pointer; }
    .download-btn:hover { background: var(--line); }
    .hidden { display: none; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 300px 1fr; }
      .rail { grid-column: 1 / -1; border-right: 0; border-bottom: 1px solid var(--line); }
      .workflow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; }
      .rail, .workbench { border-right: 0; border-bottom: 1px solid var(--line); }
      .workflow, .summary-grid { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>MatrixAI Workbench</h1>
      <div class="subtitle">P9 training loop: prompt → .mxai → entrenar con CSV → params descargables → predicción, sin salir del navegador.</div>
    </div>
    <div class="header-status">
      <div id="status" class="status">Ready</div>
      <div id="statusDetail" class="status-detail">Sin analisis todavia.</div>
    </div>
  </header>
  <main>
    <section class="rail">
      <h2>Recorrido</h2>
      <div id="workflow" class="workflow"></div>
      <h2 style="margin-top:16px">Demo guiada</h2>
      <div class="demo-actions">
        <button class="secondary" data-demo-example="email-agent" title="Carga el ejemplo EmailAgent (clasificacion de correos) y actualiza todos los paneles para recorrerlo paso a paso.">Email</button>
        <button class="secondary" data-demo-example="fall-risk" title="Carga el ejemplo FallRisk (riesgo de caida hospitalaria) y actualiza todos los paneles para recorrerlo paso a paso.">Riesgo</button>
      </div>
      <div id="demoGuide" class="guide"></div>
      <div class="callout warn" style="margin-top:12px">
        Las acciones reales siguen bloqueadas. La vista de seguridad muestra que se simula y por que.
      </div>
    </section>
    <section class="workbench">
      <div class="stack">
        <div>
          <label for="mode">Entrada</label>
          <select id="mode">
            <option value="prompt">Prompt &mdash; genera el modelo desde lenguaje natural</option>
            <option value="semantic">.semantic &mdash; valida y refina la propuesta intermedia</option>
            <option value="mxai">.mxai &mdash; analiza un programa ya compilado</option>
          </select>
          <div id="modeHint" style="margin-top:6px;font-size:12px;color:var(--muted);padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel-soft);line-height:1.5;"></div>
          <div id="llmModeIndicator" style="display:none;margin-top:6px;font-size:12px;padding:7px 10px;border-radius:6px;line-height:1.5;"></div>
        </div>
        <div id="promptField">
          <label for="prompt">Prompt</label>
          <p style="margin:0 0 4px;font-size:12px;color:var(--muted);line-height:1.45;">Describe en lenguaje natural lo que debe hacer el modelo. El sistema genera el .semantic y el .mxai automáticamente.</p>
          <textarea id="prompt"></textarea>
          <div id="llmToggleRow" style="display:none;margin-top:6px;">
            <label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:6px;">
              <input type="checkbox" id="useLlm" style="margin:0;">
              Usar LLM externo para generar el modelo (requiere <code>MATRIXAI_LLM_API_KEY</code>)
            </label>
          </div>
        </div>
        <div id="semanticField">
          <label for="semantic">.semantic</label>
          <p style="margin:0 0 4px;font-size:12px;color:var(--muted);line-height:1.45;">Representación intermedia legible del modelo: entidades, campos, distribuciones y acciones. Se genera desde el prompt y es editable antes de reanalizar en modo <em>.semantic</em>.</p>
          <textarea id="semantic" placeholder="La propuesta .semantic generada aparece aqui. Puedes editarla y cambiar a modo .semantic para reanalizar."></textarea>
        </div>
        <div id="mxaiField">
          <label for="mxai">.mxai — programa compilado</label>
          <p style="margin:0 0 4px;font-size:12px;color:var(--muted);line-height:1.45;">Programa MatrixAI compilado listo para ejecutar. Contiene vectores, funciones, distribuciones y acciones en formato ejecutable.</p>
          <textarea id="mxai" placeholder="Pega un programa .mxai o carga un ejemplo para analizarlo directamente."></textarea>
        </div>
        <div>
          <label for="training">.mxtrain — contrato de entrenamiento</label>
          <p style="margin:0 0 4px;font-size:12px;color:var(--muted);line-height:1.45;">Define hiperparámetros, dataset y épocas. Necesario para lanzar el entrenamiento y obtener parámetros W/b optimizados.</p>
          <textarea id="training" placeholder="Opcional: pega un contrato .mxtrain o carga un ejemplo real"></textarea>
        </div>
        <div>
          <label for="manifest">Dataset manifest</label>
          <p style="margin:0 0 4px;font-size:12px;color:var(--muted);line-height:1.45;">Manifiesto versionado del dataset (hashes SHA-256, splits train/val/test). Garantiza reproducibilidad entre ejecuciones.</p>
          <textarea id="manifest" placeholder="Opcional: manifiesto versionado de dataset"></textarea>
        </div>
        <div>
          <label for="evaluationReport">evaluation_report.json</label>
          <p style="margin:0 0 4px;font-size:12px;color:var(--muted);line-height:1.45;">Reporte de métricas post-entrenamiento (accuracy, MAE, RMSE, R²…). Generado por <code>matrixai evaluate</code> o por el entrenamiento integrado.</p>
          <textarea id="evaluationReport" placeholder="Opcional: pega un reporte de evaluacion generado por matrixai evaluate"></textarea>
        </div>
        <div>
          <label for="input">Input JSON — caso de prueba</label>
          <p style="margin:0 0 4px;font-size:12px;color:var(--muted);line-height:1.45;">Ejemplo de entrada para ejecutar el modelo con los parámetros entrenados y ver la predicción y el recorrido de auditoría.</p>
          <textarea id="input"></textarea>
        </div>
        <button id="run">Analizar</button>
        <div id="analysisNote" class="action-note">Pulsa Analizar para recalcular los paneles con los datos actuales.</div>
        <button id="fillEmail" class="secondary" title="Atajo rapido: rellena los campos de la columna central con los archivos del ejemplo EmailAgent. Equivale a pulsar el boton Email de la Demo guiada.">&#8615; Rellenar con EmailAgent</button>
        <button id="fillFallRisk" class="secondary" title="Atajo rapido: rellena los campos de la columna central con los archivos del ejemplo FallRisk. Equivale a pulsar el boton Riesgo de la Demo guiada.">&#8615; Rellenar con FallRisk</button>
      </div>
    </section>
    <section>
      <div class="stack" style="margin-bottom:12px">
        <div id="overview" class="summary-grid"></div>
        <div id="humanSummary" class="callout"></div>
        <div id="diagnosticsSummary" class="callout hidden"></div>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="modelView">Modelo</button>
        <button class="tab" data-tab="networkView">Red</button>
        <button class="tab" data-tab="dataView">Datos</button>
        <button class="tab" data-tab="evaluationView">Evaluacion</button>
        <button class="tab" data-tab="securityView">Seguridad</button>
        <button class="tab" data-tab="apiView">API</button>
        <button class="tab" data-tab="pipelineView">Pipeline</button>
        <button class="tab" data-tab="checks">Validaciones</button>
        <button class="tab" data-tab="semanticPanel">Semantico</button>
        <button class="tab" data-tab="mxaiPanel">MXAI</button>
        <button class="tab" data-tab="irOut">IR</button>
        <button class="tab" data-tab="graphOut">Grafo</button>
        <button class="tab" data-tab="runOut">Ejecucion</button>
        <button class="tab" data-tab="diagnoseOut">Diagnostico</button>
      </div>
      <div id="modelView" class="panel stack"></div>
      <div id="networkView" class="panel hidden" style="overflow-x:auto;padding:8px 0;"></div>
      <div id="dataView" class="panel stack hidden"></div>
      <div id="evaluationView" class="panel stack hidden"></div>
      <div id="securityView" class="panel stack hidden"></div>
      <div id="apiView" class="panel stack hidden"></div>
      <div id="pipelineView" class="panel stack hidden"></div>
      <div id="checks" class="panel checks hidden"></div>
      <div id="semanticPanel" class="panel stack hidden">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
          <div id="semanticSourceBadge" style="font-size:11px;padding:4px 8px;border-radius:4px;background:var(--panel-soft);border:1px solid var(--line);"></div>
          <button id="reanalizeSemanticBtn" class="secondary" style="display:none;font-size:12px;padding:3px 10px;" title="Enviar el texto .semantic editado al backend como mode=semantic">&#9654; Reanalizar como .semantic</button>
          <button class="secondary" style="font-size:12px;padding:3px 10px;" onclick="copyToClipboard(byId('semanticOut').textContent, this)">Copiar</button>
        </div>
        <pre id="semanticOut" style="margin:0;white-space:pre-wrap;"></pre>
      </div>
      <div id="mxaiPanel" class="panel stack hidden">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
          <div id="mxaiSourceBadge" style="font-size:11px;padding:4px 8px;border-radius:4px;background:var(--panel-soft);border:1px solid var(--line);"></div>
          <button class="secondary" style="font-size:12px;padding:3px 10px;" onclick="copyToClipboard(byId('mxaiOut').textContent, this)">Copiar</button>
          <button class="secondary" id="editAsMxaiBtn" style="font-size:12px;padding:3px 10px;" onclick="editAsMxai()">Editar como .mxai</button>
        </div>
        <pre id="mxaiOut" style="margin:0;white-space:pre-wrap;"></pre>
      </div>
      <pre id="irOut" class="panel hidden"></pre>
      <div id="graphOut" class="panel hidden" style="overflow:auto;max-height:72vh;padding:12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);"></div>
      <div id="runOut" class="panel stack hidden">
        <div id="runDataInner"></div>
        <div id="refinePanel" style="display:none;margin-top:16px;border-top:1px solid var(--line);padding-top:12px;">
          <h2 style="margin:0 0 10px;">Refinar prompt <span style="font-size:11px;font-weight:400;color:var(--muted);">P13</span></h2>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
            <label style="font-size:12px;color:var(--muted);">Iteracion:</label>
            <input id="refineIteration" type="number" min="1" value="1" style="width:50px;font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:4px;">
            <label style="font-size:12px;color:var(--muted);">Max iter.:</label>
            <input id="refineMaxIter" type="number" min="1" value="3" style="width:50px;font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:4px;">
            <button class="secondary" id="refineBtn" onclick="refinePrompt()">Refinar prompt</button>
          </div>
          <textarea id="refineHints" rows="2" placeholder="Hints adicionales (uno por linea, opcional)..." style="width:100%;font-family:monospace;font-size:12px;padding:6px;border:1px solid var(--line);border-radius:4px;resize:vertical;margin-bottom:8px;box-sizing:border-box;"></textarea>
          <div id="refineResult" style="display:none;"></div>
        </div>
      </div>
      <div id="diagnoseOut" class="panel stack hidden"></div>
    </section>
  </main>
  <script>
    const byId = (id) => document.getElementById(id);
    const state = { last: null, lastSignature: '', pendingModeNote: '' };
    const refineState = { lastRunResult: null, chain: [], parentHash: '' };

    async function loadDefaults() {
      const response = await fetch('/api/defaults');
      const data = await response.json();
      byId('prompt').value = data.prompt;
      byId('input').value = data.input_json;
    }

    function showTab(id) {
      document.querySelectorAll('.panel').forEach((el) => el.classList.add('hidden'));
      document.querySelectorAll('.tab').forEach((el) => el.classList.remove('active'));
      byId(id).classList.remove('hidden');
      document.querySelector(`[data-tab="${id}"]`).classList.add('active');
      if (id === 'graphOut' && state.last && state.last.graph_mermaid) {
        renderGraphView(state.last.graph_mermaid);
      }
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[char]));
    }

    function table(headers, rows) {
      if (!rows.length) return '<div class="callout">Sin elementos declarados.</div>';
      const head = headers.map((header) => `<th>${escapeHtml(header.label)}</th>`).join('');
      const body = rows.map((row) => `<tr>${headers.map((header) => `<td>${escapeHtml(row[header.key])}</td>`).join('')}</tr>`).join('');
      return `<table class="table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }

    mermaid.initialize({ startOnLoad: false, theme: 'neutral' });

    function copyToClipboard(text, btn) {
      navigator.clipboard.writeText(text || '').then(() => {
        const orig = btn.textContent;
        btn.textContent = '✓ Copiado';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      }).catch(() => { btn.textContent = '✗ Error'; });
    }

    // P8: switch to mxai mode and copy current displayed .mxai into the input textarea
    function editAsMxai() {
      const content = byId('mxaiOut').textContent.trim();
      if (!content) return;
      byId('mxai').value = content;
      byId('mode').value = 'mxai';
      updateModeVisibility();
      byId('mxai').focus();
    }

    // P9 training state
    const trainState = { paramsBest: null, runId: null, mxaiText: null, jobId: null, pollTimer: null };

    async function generateTraining() {
      const mxai = byId('mxai').value.trim();
      if (!mxai) { alert('Carga o genera un .mxai primero.'); return; }
      byId('generateTrainingBtn').disabled = true;
      byId('generateTrainingBtn').textContent = 'Generando...';
      try {
        const res = await fetch('/api/generate-training', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mxai_text: mxai }),
        });
        const data = await res.json();
        if (data.ok) {
          byId('training').value = data.training_text;
          byId('csvPaste').value = data.dataset_template_text;
          byId('trainPanel').classList.remove('hidden');
          byId('csvValidStatus').textContent = 'CSV: plantilla generada, pega tus datos o sube un fichero.';
          byId('csvValidStatus').className = 'callout warn';
          byId('trainBtn').disabled = true;
        } else {
          byId('trainFeedback').textContent = 'Error generando entrenamiento: ' + (data.error || '');
          byId('trainFeedback').className = 'callout bad';
        }
      } finally {
        byId('generateTrainingBtn').disabled = false;
        byId('generateTrainingBtn').textContent = 'Generar entrenamiento';
      }
    }

    async function generateSyntheticDataset() {
      const mxai = byId('mxai').value.trim();
      const training = byId('training').value.trim();
      if (!mxai || !training) { alert('Carga un .mxai y un .mxtrain primero.'); return; }
      const rows = parseInt(byId('syntheticRows').value) || 200;
      const seed = parseInt(byId('syntheticSeed').value) || 42;
      const mode = byId('syntheticMode').value;
      byId('generateDatasetBtn').disabled = true;
      byId('generateDatasetBtn').textContent = 'Generando...';
      byId('syntheticStatus').style.display = 'none';
      try {
        const res = await fetch('/api/generate-dataset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mxai_text: mxai, training_text: training, rows, seed, mode }),
        });
        const data = await res.json();
        if (data.ok) {
          byId('csvPaste').value = data.csv_text;
          byId('csvValidStatus').textContent = `Dataset sintético: ${data.rows} filas · seed=${data.seed} · ${data.mode} · fingerprint ${data.fingerprint}`;
          byId('csvValidStatus').className = data.coherent_fallback_count ? 'callout warn' : 'callout ok';
          byId('csvValidStatus').style.display = '';
          byId('trainBtn').disabled = false;
          byId('trainPanel').classList.remove('hidden');
          const fallbackMsg = data.coherent_fallback_warning ? ` ⚠ ${data.coherent_fallback_warning}` : '';
          // M12: el backend recorta las filas según el perfil de límites y lo avisa.
          const cappedMsg = data.rows_capped_warning ? ` ⚠ ${data.rows_capped_warning}` : '';
          const signalMsg = data.signal_warning ? ` ⚠ ${data.signal_warning}` : '';
          byId('syntheticStatus').textContent = `Dataset sintético generado: ${data.rows} filas, seed=${data.seed}, mode=${data.mode}. Fingerprint: ${data.fingerprint}. Columnas: ${data.columns?.join(', ')}.${fallbackMsg}${cappedMsg}${signalMsg}`;
          byId('syntheticStatus').className = (data.coherent_fallback_count || data.rows_capped_warning || data.signal_warning) ? 'callout warn' : 'callout ok';
          byId('syntheticStatus').style.display = '';
        } else {
          byId('syntheticStatus').textContent = 'Error generando dataset: ' + (data.error || 'error desconocido');
          byId('syntheticStatus').className = 'callout bad';
          byId('syntheticStatus').style.display = '';
        }
      } catch (e) {
        byId('syntheticStatus').textContent = 'Error de red: ' + e.message;
        byId('syntheticStatus').className = 'callout bad';
        byId('syntheticStatus').style.display = '';
      } finally {
        byId('generateDatasetBtn').disabled = false;
        byId('generateDatasetBtn').textContent = 'Generar dataset sintético';
      }
    }

    async function validateCsv() {
      const mxai = byId('mxai').value.trim();
      const training = byId('training').value.trim();
      const csv = byId('csvPaste').value.trim();
      if (!mxai || !training || !csv) { byId('csvValidStatus').textContent = 'Necesitas .mxai, .mxtrain y CSV.'; byId('csvValidStatus').className = 'callout warn'; return; }
      byId('validateCsvBtn').disabled = true;
      byId('csvValidStatus').textContent = 'Validando...';
      byId('csvValidStatus').className = 'callout';
      try {
        const res = await fetch('/api/validate-csv', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mxai_text: mxai, training_text: training, csv_text: csv }),
        });
        const data = await res.json();
        if (data.ok) {
          byId('csvValidStatus').textContent = `CSV válido: ${data.rows} filas${data.warnings?.length ? ' — ' + data.warnings.join('; ') : ''}.`;
          byId('csvValidStatus').className = data.warnings?.length ? 'callout warn' : 'callout ok';
          byId('trainBtn').disabled = false;
        } else {
          const msg = data.error || (data.errors || []).join('\\n');
          byId('csvValidStatus').textContent = 'CSV inválido: ' + msg;
          byId('csvValidStatus').className = 'callout bad';
          byId('trainBtn').disabled = true;
        }
      } finally {
        byId('validateCsvBtn').disabled = false;
      }
    }

    function stopPolling() {
      if (trainState.pollTimer) { clearInterval(trainState.pollTimer); trainState.pollTimer = null; }
    }

    async function cancelTraining() {
      stopPolling();
      if (!trainState.jobId) return;
      await fetch('/api/train-cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_id: trainState.jobId }) });
      byId('trainFeedback').textContent = 'Entrenamiento cancelado.';
      byId('trainFeedback').className = 'callout warn';
      byId('trainBtn').disabled = false;
      byId('trainBtn').textContent = 'Entrenar';
      byId('stopTrainBtn').classList.add('hidden');
    }

    async function runTraining() {
      const mxai = byId('mxai').value.trim();
      const training = byId('training').value.trim();
      const csv = byId('csvPaste').value.trim();
      const epochsVal = byId('epochsOverride').value;
      const epochs = epochsVal ? parseInt(epochsVal, 10) : null;
      byId('trainBtn').disabled = true;
      byId('trainBtn').textContent = 'Iniciando...';
      byId('stopTrainBtn').classList.remove('hidden');
      byId('epochProgress').innerHTML = '';
      byId('trainFeedback').textContent = '';
      byId('artifactsPanel').classList.add('hidden');
      byId('predPanel').classList.add('hidden');
      trainState.paramsBest = null;
      trainState.jobId = null;
      stopPolling();
      try {
        const res = await fetch('/api/train-start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mxai_text: mxai, training_text: training, csv_text: csv, epochs_override: epochs }),
        });
        const data = await res.json();
        if (!data.ok) {
          byId('trainFeedback').textContent = 'Error al iniciar: ' + (data.error || '');
          byId('trainFeedback').className = 'callout bad';
          byId('trainBtn').disabled = false;
          byId('trainBtn').textContent = 'Entrenar';
          byId('stopTrainBtn').classList.add('hidden');
          return;
        }
        trainState.jobId = data.job_id;
        byId('trainBtn').textContent = 'Entrenando...';
        trainState.pollTimer = setInterval(async () => {
          const s = await fetch(`/api/train-status/${trainState.jobId}`).then((r) => r.json());
          renderEpochTable(s.epochs || [], s.best_epoch, s.task_kind);
          if (s.status === 'done') {
            stopPolling();
            trainState.paramsBest = s.params_best;
            trainState.mxaiText = mxai;
            trainState.taskKind = s.task_kind || 'classification';
            const _isReg = trainState.taskKind === 'regression';
            const _metricTxt = _isReg ? `mae ${(s.mae ?? 0).toFixed(4)}` : `accuracy ${(s.accuracy ?? 0).toFixed(4)}`;
            byId('trainFeedback').textContent = `Entrenamiento OK — epoch ${s.best_epoch}, ${_metricTxt}, val_loss ${(s.best_validation_loss ?? 0).toFixed(4)}, backend: ${s.backend || 'stdlib'}`;
            byId('trainFeedback').className = 'callout ok';
            byId('predPanel').classList.remove('hidden');
            renderArtifactDownloads(s);
            byId('trainBtn').disabled = false;
            byId('trainBtn').textContent = 'Entrenar';
            byId('stopTrainBtn').classList.add('hidden');
            const _trainStep = byId('guideStep_training');
            if (_trainStep) _trainStep.className = 'guide-step valid';
            const _wfTrainStep = byId('workflowStep_training');
            if (_wfTrainStep) _wfTrainStep.className = 'step valid';
            renderNetworkView(window._lastVisualModel || null, s.params_best || null);
          } else if (s.status === 'error') {
            stopPolling();
            byId('trainFeedback').textContent = 'Error: ' + (s.error || '');
            byId('trainFeedback').className = 'callout bad';
            byId('trainBtn').disabled = false;
            byId('trainBtn').textContent = 'Entrenar';
            byId('stopTrainBtn').classList.add('hidden');
          } else if (s.status === 'cancelled') {
            stopPolling();
            byId('trainFeedback').textContent = 'Entrenamiento cancelado.';
            byId('trainFeedback').className = 'callout warn';
            byId('trainBtn').disabled = false;
            byId('trainBtn').textContent = 'Entrenar';
            byId('stopTrainBtn').classList.add('hidden');
          } else if (s.status === 'timeout') {
            stopPolling();
            byId('trainFeedback').textContent = 'Entrenamiento superó el límite de 30s y fue detenido automáticamente.';
            byId('trainFeedback').className = 'callout bad';
            byId('trainBtn').disabled = false;
            byId('trainBtn').textContent = 'Entrenar';
            byId('stopTrainBtn').classList.add('hidden');
          }
        }, 800);
      } catch (err) {
        byId('trainFeedback').textContent = 'Error de red: ' + err.message;
        byId('trainFeedback').className = 'callout bad';
        byId('trainBtn').disabled = false;
        byId('trainBtn').textContent = 'Entrenar';
        byId('stopTrainBtn').classList.add('hidden');
      }
    }

    function renderEpochTable(epochs, bestEpoch, taskKind) {
      if (!epochs || !epochs.length) return;
      const isReg = taskKind === 'regression';
      const header = isReg
        ? '<tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th></tr>'
        : '<tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Accuracy</th></tr>';
      const rows = epochs.map((e) => {
        const isBest = e.epoch === bestEpoch;
        const extra = isReg ? '' : `<td>${(e.accuracy ?? 0).toFixed(4)}</td>`;
        return `<tr class="${isBest ? 'best' : ''}"><td>${e.epoch}</td><td>${(e.train_loss ?? 0).toFixed(4)}</td><td>${(e.validation_loss ?? 0).toFixed(4)}</td>${extra}</tr>`;
      }).join('');
      byId('epochProgress').innerHTML = `<table class="epoch-table"><thead>${header}</thead><tbody>${rows}</tbody></table>`;
    }

    function renderArtifactDownloads(data) {
      const items = [];
      if (data.params_best) items.push(['params.best.json', JSON.stringify(data.params_best, null, 2)]);
      if (data.metrics) items.push(['metrics.json', JSON.stringify(data.metrics, null, 2)]);
      if (data.training_trace) items.push(['training_trace.json', JSON.stringify(data.training_trace, null, 2)]);
      if (data.evaluation_report) items.push(['evaluation_report.json', JSON.stringify(data.evaluation_report, null, 2)]);
      byId('artifactsPanel').innerHTML = '<div class="train-section"><h3>Artefactos descargables</h3>' +
        items.map(([name, content]) => {
          const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
          return `<a class="download-btn" href="${url}" download="${name}">↓ ${name}</a>`;
        }).join('') + '</div>';
      byId('artifactsPanel').classList.remove('hidden');
    }

    async function runPrediction() {
      if (!trainState.paramsBest) { alert('Entrena primero para obtener parámetros.'); return; }
      const mxai = trainState.mxaiText || byId('mxai').value.trim();
      const inputJson = byId('predInput').value.trim() || byId('input').value.trim();
      byId('predBtn').disabled = true;
      byId('predBtn').textContent = 'Ejecutando...';
      byId('predResult').textContent = '';
      try {
        const res = await fetch('/api/run-with-params', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mxai_text: mxai, params_json: JSON.stringify(trainState.paramsBest), input_json: inputJson }),
        });
        const data = await res.json();
        byId('predResult').textContent = data.ok ? JSON.stringify(data.result, null, 2) : 'Error: ' + (data.error || '');
        byId('predResult').className = data.ok ? '' : 'callout bad';
      } finally {
        byId('predBtn').disabled = false;
        byId('predBtn').textContent = 'Probar predicción';
      }
    }

    function handleCsvFile(event) {
      const file = event.target.files[0];
      if (!file) return;
      if (file.size > 50000000) { alert('Fichero CSV supera el límite de 50 MB'); return; }
      const reader = new FileReader();
      reader.onload = (e) => { byId('csvPaste').value = e.target.result; byId('trainBtn').disabled = true; byId('csvValidStatus').textContent = 'CSV cargado. Pulsa Validar CSV para verificarlo.'; byId('csvValidStatus').className = 'callout warn'; };
      reader.readAsText(file);
    }

    const MODE_HINTS = {
      prompt: '\u25b6 Escribe en lenguaje natural lo que debe hacer el modelo. El sistema genera autom\u00e1ticamente el .semantic y el .mxai por ti. \u2014 Objetivo final del proyecto: todo el pipeline desde un solo prompt.',
      semantic: '\u25b6 Etapa intermedia del pipeline. Edita o pega la propuesta .semantic generada por el Prompt para revisarla antes de compilar a .mxai.',
      mxai: '\u25b6 Programa .mxai ya compilado listo para analizar, ejecutar y servir. Los ejemplos de la Demo guiada usan este modo porque son modelos acabados.',
    };

    let _prevMode = null;
    function resetPanels() {
      // Clear sidebar
      byId('workflow').innerHTML = '';
      byId('demoGuide').innerHTML = '';
      byId('overview').innerHTML = '';
      byId('humanSummary').textContent = 'Pulsa Analizar para ver el recorrido del modelo.';
      byId('diagnosticsSummary').classList.add('hidden');
      byId('analysisNote').textContent = 'Pulsa Analizar para recalcular los paneles con los datos actuales.';
      byId('analysisNote').className = 'action-note';
      byId('status').textContent = 'Ready';
      byId('status').className = 'status';
      byId('statusDetail').textContent = '';
      window._lastVisualModel = null;
      // Clear right panels
      byId('modelView').innerHTML = '';
      byId('networkView').innerHTML = '';
      byId('dataView').innerHTML = '';
      byId('evaluationView').innerHTML = '';
      byId('securityView').innerHTML = '';
      byId('apiView').innerHTML = '';
      byId('pipelineView').innerHTML = '';
      byId('checks').innerHTML = '';
      byId('semanticOut').textContent = '';
      byId('mxaiOut').textContent = '';
      byId('irOut').textContent = '';
      byId('graphOut').innerHTML = '';
      const runInner = byId('runDataInner');
      if (runInner) runInner.innerHTML = '';
      const refinePanel = byId('refinePanel');
      if (refinePanel) refinePanel.style.display = 'none';
      const refineResult = byId('refineResult');
      if (refineResult) refineResult.style.display = 'none';
      byId('diagnoseOut').innerHTML = '';
      renderNetworkView(null, null);
      renderOverview(null);
    }

    function updateModeVisibility() {
      const mode = byId('mode').value;
      byId('promptField').style.display = (mode === 'mxai') ? 'none' : '';
      byId('semanticField').style.display = (mode === 'semantic') ? '' : 'none';
      byId('mxaiField').style.display = (mode === 'mxai') ? '' : 'none';
      byId('llmToggleRow').style.display = (mode === 'prompt') ? '' : 'none';
      byId('modeHint').textContent = MODE_HINTS[mode] || '';
      if (_prevMode !== null && _prevMode !== mode && mode === 'prompt') {
        byId('mxai').value = '';
        byId('semantic').value = '';
        byId('training').value = '';
        byId('manifest').value = '';
        byId('evaluationReport').value = '';
        byId('input').value = '';
        resetPanels();
      }
      _prevMode = mode;
    }

    function renderGraphView(code) {
      const el = byId('graphOut');
      el.innerHTML = '';
      if (!code) return;
      const wrapper = document.createElement('div');
      wrapper.className = 'mermaid';
      wrapper.textContent = code;
      el.appendChild(wrapper);
      mermaid.run({ nodes: [wrapper] }).catch(() => {
        el.innerHTML = '';
        const pre = document.createElement('pre');
        pre.style.cssText = 'margin:0;overflow:auto;font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;';
        pre.textContent = code;
        el.appendChild(pre);
      });
    }

    function renderPipelineView(stages, artifacts) {
      const el = byId('pipelineView');
      if (!stages || !stages.length) {
        el.innerHTML = '<div class="callout warn">Sin datos de pipeline — analiza un prompt o .semantic para ver las etapas.</div>';
        return;
      }
      const STATUS_LABEL = { ok: 'ok', fail: 'FALLO', warning: 'aviso', skipped: 'omitida', pending: 'pendiente' };
      let html = '<div class="pipeline">' + stages.map((s, i) => {
        const detail = [...(s.errors || []), ...(s.warnings || [])].map(escapeHtml).join('\\n');
        return `<div class="pipeline-stage ${escapeHtml(s.status)}">
          <div class="ps-num">${i + 1}</div>
          <div style="flex:1">
            <div class="ps-label">${escapeHtml(s.label || s.name)}</div>
            ${detail ? `<div class="ps-detail">${detail}</div>` : ''}
          </div>
          <div class="ps-status ${escapeHtml(s.status)}">${STATUS_LABEL[s.status] || s.status}</div>
        </div>`;
      }).join('') + '</div>';
      const cp = artifacts && artifacts.compiled_python;
      if (cp && cp.text) {
        html += `<div style="margin-top:14px;border-top:1px solid var(--line);padding-top:12px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <strong style="font-size:13px;">compiled_python</strong>
            <span style="font-size:11px;color:var(--muted);">${escapeHtml(cp.source || '')}</span>
            <button class="secondary" style="font-size:12px;padding:3px 10px;margin-left:auto;" id="copyCompiledBtn" onclick="copyToClipboard(byId('compiledPythonPre').textContent, this)">Copiar</button>
          </div>
          <pre id="compiledPythonPre" style="margin:0;white-space:pre-wrap;max-height:300px;overflow:auto;">${escapeHtml(cp.text)}</pre>
        </div>`;
      }
      el.innerHTML = html;
    }

    function renderArtifactBadges(artifacts, mode) {
      const sem = artifacts && artifacts.semantic;
      const mx = artifacts && artifacts.mxai;
      const semBadge = byId('semanticSourceBadge');
      const mxBadge = byId('mxaiSourceBadge');
      const reanalizeBtn = byId('reanalizeSemanticBtn');
      if (semBadge) semBadge.textContent = sem ? `Origen: ${sem.source}` : '';
      if (mxBadge) mxBadge.textContent = mx ? `Origen: ${mx.source}` : '';
      if (reanalizeBtn) reanalizeBtn.style.display = (sem && mode === 'prompt') ? '' : 'none';
    }

    function renderRunView(run) {
      const el = byId('runDataInner');
      if (!run) {
        el.innerHTML = '<div class="callout warn">Introduce un Input JSON y pulsa Analizar para ver la ejecucion.</div>';
        byId('refinePanel').style.display = 'none';
        return;
      }
      const actions = (run.actions || []);
      const trace = (run.trace || []);
      const auditHtml = run.audit ? `<div class="callout" style="margin-bottom:8px">${escapeHtml(run.audit)}</div>` : '';
      const actionsHtml = table(
        [{ key: 'name', label: 'Accion' }, { key: 'valueStr', label: 'Valor' }, { key: 'threshold', label: 'Umbral' }, { key: 'activatedStr', label: 'Activada' }, { key: 'policy', label: 'Politica' }],
        actions.map((a) => ({ ...a, valueStr: typeof a.value === 'number' ? a.value.toFixed(4) : String(a.value ?? ''), activatedStr: a.activated ? 'Si' : 'No', threshold: a.threshold != null ? a.threshold : '' }))
      );
      const traceHtml = table(
        [{ key: 'step', label: 'Paso' }, { key: 'node', label: 'Nodo' }, { key: 'node_type', label: 'Tipo' }, { key: 'status', label: 'Estado' }],
        trace.map((t) => ({ ...t, status: t.status === 'ok' ? 'ok' : t.status }))
      );
      el.innerHTML = auditHtml + '<h2>Acciones</h2>' + actionsHtml + '<h2>Traza</h2>' + traceHtml;
      byId('refinePanel').style.display = '';
    }

    function renderDiagnoseView(diag) {
      const el = byId('diagnoseOut');
      if (!diag) { el.innerHTML = '<div class="callout warn">Sin datos de diagnostico todavia.</div>'; return; }
      const okCss = diag.ok ? 'ok' : 'bad';
      const badge = `<div class="callout ${okCss}" style="margin-bottom:8px"><strong>${diag.ok ? 'Runtime y compiler coinciden.' : 'Se detectaron discrepancias.'}</strong>${diag.project ? ` Proyecto: ${escapeHtml(diag.project)}` : ''}</div>`;
      const mismatches = (diag.mismatches || []);
      const mismatchHtml = mismatches.length
        ? '<h2>Discrepancias</h2>' + table([{ key: 'text', label: 'Detalle' }], mismatches.map((m) => ({ text: typeof m === 'string' ? m : JSON.stringify(m) })))
        : '';
      el.innerHTML = badge + mismatchHtml;
    }

    function renderWorkflow(data) {
      byId('workflow').innerHTML = (data.workflow || []).map((step) => `
        <div id="workflowStep_${escapeHtml(step.id)}" class="step ${escapeHtml(step.status)}">
          <div class="dot"></div>
          <div>
            <div class="step-title">${escapeHtml(step.label)}</div>
            <div class="step-summary">${escapeHtml(step.summary)}</div>
          </div>
        </div>
      `).join('');
    }

    function renderDemoGuide(data) {
      const workflow = Object.fromEntries((data.workflow || []).map((step) => [step.id, step]));
      const steps = [
        { id: 'model', label: 'Modelo', tab: 'modelView', scroll: '' },
        { id: 'types', label: 'Entradas', tab: 'dataView', scroll: '' },
        { id: 'training', label: 'Entrenamiento', tab: 'dataView', scroll: 'trainingSection' },
        { id: 'evaluation', label: 'Evaluacion', tab: 'evaluationView', scroll: '' },
        { id: 'runtime', label: 'Ejecucion', tab: 'runOut', scroll: '' },
        { id: 'serving', label: 'API', tab: 'apiView', scroll: '' },
      ];
      byId('demoGuide').innerHTML = steps.map((step) => {
        const status = workflow[step.id]?.status || 'pending';
        return `
          <button id="guideStep_${escapeHtml(step.id)}" class="guide-step ${escapeHtml(status)}" data-demo-tab="${escapeHtml(step.tab)}" data-demo-scroll="${escapeHtml(step.scroll)}">
            <span class="dot"></span>
            <span>${escapeHtml(step.label)}</span>
          </button>
        `;
      }).join('');
    }

    function renderOverview(visual) {
      const overview = visual?.overview || {};
      const metrics = [
        ['inputs', 'Campos'],
        ['functions', 'Calculos'],
        ['actions', 'Acciones'],
      ];
      byId('overview').innerHTML = metrics.map(([key, label]) => `
        <div class="metric">
          <span class="metric-value">${escapeHtml(overview[key] ?? 0)}</span>
          <span class="metric-label">${escapeHtml(label)}</span>
        </div>
      `).join('');
      byId('humanSummary').textContent = visual
        ? `${visual.project}: ${overview.graph_nodes || 0} nodos, ${overview.graph_edges || 0} conexiones y ${overview.actions || 0} accion(es) revisadas.`
        : 'Ejecuta el analisis para ver el recorrido del modelo.';
    }

    function renderDiagnostics(data) {
      const blocked = (data.workflow || []).filter((step) => step.status === 'blocked');
      const warnings = (data.workflow || []).filter((step) => step.status === 'warning');
      const pending = (data.workflow || []).filter((step) => step.status === 'pending');
      const detail = byId('diagnosticsSummary');
      if (!blocked.length && !warnings.length && !pending.length) {
        detail.className = 'callout hidden';
        detail.innerHTML = '';
        return;
      }
      const primary = blocked.length
        ? 'Bloqueos detectados'
        : warnings.length
          ? 'Avisos detectados'
          : 'Pasos pendientes';
      const css = blocked.length ? 'bad' : 'warn';
      const rows = [...blocked, ...warnings, ...pending].map((step) => `
        <li><strong>${escapeHtml(step.label)}</strong>: ${escapeHtml(step.summary)}</li>
      `).join('');
      detail.className = `callout ${css}`;
      detail.innerHTML = `<strong>${primary}</strong><ul class="diagnostics-list">${rows}</ul>`;
    }

    function hasWorkflowWarnings(data) {
      return (data.workflow || []).some((step) => step.status === 'warning');
    }

    function tabForWorkflowStep(stepId) {
      return {
        model: 'modelView',
        types: 'dataView',
        training: 'dataView',
        evaluation: 'evaluationView',
        security: 'securityView',
        runtime: 'runOut',
        serving: 'apiView',
      }[stepId] || 'checks';
    }

    function focusFirstBlockedView(data) {
      const failedStage = (data.pipeline_stages || []).find((s) => s.status === 'fail');
      if (failedStage) { showTab('pipelineView'); return; }
      const blocked = (data.workflow || []).find((step) => step.status === 'blocked');
      if (blocked) showTab(tabForWorkflowStep(blocked.id));
    }

    function escSvg(s) {
      return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    function renderNetworkView(visual, params) {
      const el = byId('networkView');
      if (!visual || !(visual.inputs || []).length) {
        el.innerHTML = '<div class="callout warn" style="margin:16px">Analiza un modelo para ver la arquitectura de red.</div>';
        return;
      }

      const inputNodes = visual.inputs;
      const calculations = visual.calculations || [];
      const mainCalc = calculations[0] || {};
      const kind = mainCalc.kind || '';

      // Parse PARAM W1[rows,cols] — rows=n_outputs, cols=n_inputs
      const mxaiText = (state.last && state.last.mxai) || byId('mxai')?.value || '';
      const paramDims = {};
      const paramRe = /PARAM\\s+(\\w+)\\s*\\[(\\d+)(?:\\s*,\\s*(\\d+))?\\]/g;
      let pm;
      while ((pm = paramRe.exec(mxaiText)) !== null) {
        paramDims[pm[1]] = { rows: parseInt(pm[2], 10), cols: pm[3] ? parseInt(pm[3], 10) : 1 };
      }

      const w1Dim = paramDims['W1'];
      const numIn = inputNodes.length;

      // W stored as [numOut][numIn] (classification) or 1-D [numIn] (regression).
      // Infer numOut from trained W when available — avoids relying on PARAM declarations
      // that may be absent in pre-compiled .mxai files (e.g. email-agent.mxai).
      const paramVals = (params || trainState.paramsBest || {}).parameters || {};
      const Wraw = (paramVals['W1']?.values) ?? null;
      let W = null, hasWeights = false, numOut;
      if (Wraw && Array.isArray(Wraw)) {
        if (Array.isArray(Wraw[0])) {
          // 2D [n_outputs][n_inputs] — trust shape directly
          numOut = Wraw.length;
          hasWeights = Wraw[0].length === numIn;
          if (hasWeights) W = Wraw;
        } else {
          // 1D [n_inputs] — regression single output
          numOut = 1;
          hasWeights = Wraw.length === numIn;
          if (hasWeights) W = [Wraw];
        }
      } else {
        // No trained params yet — infer from PARAM declaration or kind fallback
        numOut = w1Dim ? w1Dim.rows : (kind === 'softmax_linear' ? 2 : 1);
      }
      const getW = (i, j) => hasWeights ? W[j][i] : 0;
      const wFlat = hasWeights ? W.flat() : [];
      const wMax = wFlat.length ? Math.max(...wFlat.map(Math.abs), 0.0001) : 1;

      const actLabel = kind === 'softmax_linear' ? 'softmax'
        : kind === 'sigmoid_threshold' ? 'sigmoid'
        : kind === 'linear_regression' ? 'linear'
        : (kind.split('_')[0] || 'f');

      // 3-column layout: inputs | activation neurons | outputs
      const rIn = 20, rNeuron = 18, rOut = 16;
      const pad = 52;
      const rowHIn = Math.max(44, Math.min(64, 340 / numIn));
      const rowHOut = Math.max(44, Math.min(64, 340 / numOut));
      const totalInH = numIn * rowHIn;
      const totalOutH = numOut * rowHOut;
      const svgH = Math.max(totalInH, totalOutH) + pad * 2;
      const colIn = 130, colNeuron = 370, colOut = 590, svgW = 730;

      const inStartY = pad + (svgH - pad * 2 - totalInH) / 2;
      const outStartY = pad + (svgH - pad * 2 - totalOutH) / 2;
      const inY = (i) => inStartY + i * rowHIn + rowHIn / 2;
      const neuronY = (j) => outStartY + j * rowHOut + rowHOut / 2;
      const outY = (j) => outStartY + j * rowHOut + rowHOut / 2;

      let svg = `<svg viewBox="0 0 ${svgW} ${svgH}" style="width:100%;max-width:${svgW}px;display:block;" xmlns="http://www.w3.org/2000/svg">`;
      svg += `<rect width="${svgW}" height="${svgH}" fill="#f8fafc" rx="8"/>`;

      svg += `<text x="${colIn}" y="24" text-anchor="middle" font-size="11" fill="#888" font-family="ui-sans-serif,sans-serif">Capa de entrada</text>`;
      svg += `<text x="${colNeuron}" y="24" text-anchor="middle" font-size="11" fill="#555" font-family="ui-sans-serif,sans-serif" font-style="italic">${escSvg(actLabel)}</text>`;
      svg += `<text x="${colOut}" y="24" text-anchor="middle" font-size="11" fill="#888" font-family="ui-sans-serif,sans-serif">Capa de salida</text>`;

      for (let i = 0; i < numIn; i++) {
        for (let j = 0; j < numOut; j++) {
          let strokeColor, strokeW;
          if (hasWeights) {
            const w = getW(i, j);
            const abs = Math.abs(w) / wMax;
            strokeW = Math.max(0.5, abs * 5.5);
            const alpha = Math.max(0.07, abs * 0.82);
            strokeColor = w >= 0 ? `rgba(37,99,235,${alpha.toFixed(2)})` : `rgba(220,38,38,${alpha.toFixed(2)})`;
          } else {
            strokeW = 0.7;
            strokeColor = 'rgba(160,160,160,0.15)';
          }
          svg += `<line x1="${colIn + rIn}" y1="${inY(i).toFixed(1)}" x2="${colNeuron - rNeuron}" y2="${neuronY(j).toFixed(1)}" stroke="${strokeColor}" stroke-width="${strokeW.toFixed(2)}"/>`;
        }
      }

      for (let j = 0; j < numOut; j++) {
        svg += `<line x1="${colNeuron + rNeuron}" y1="${neuronY(j).toFixed(1)}" x2="${colOut - rOut}" y2="${outY(j).toFixed(1)}" stroke="rgba(100,100,100,0.28)" stroke-width="1.2"/>`;
      }

      for (let i = 0; i < numIn; i++) {
        const node = inputNodes[i];
        const cx = colIn, cy = inY(i);
        svg += `<circle cx="${cx}" cy="${cy.toFixed(1)}" r="${rIn}" fill="#e8f0fe" stroke="#4285f4" stroke-width="1.5"/>`;
        const lbl = node.field.length > 9 ? node.field.slice(0, 8) + '…' : node.field;
        svg += `<text x="${cx}" y="${(cy + 4).toFixed(1)}" text-anchor="middle" font-size="9" fill="#1a56db" font-family="ui-monospace,monospace">${escSvg(lbl)}</text>`;
        svg += `<text x="${cx - rIn - 6}" y="${(cy + 4).toFixed(1)}" text-anchor="end" font-size="10" fill="#555" font-family="ui-sans-serif,sans-serif">${escSvg(node.vector)}</text>`;
      }

      for (let j = 0; j < numOut; j++) {
        const cx = colNeuron, cy = neuronY(j);
        svg += `<circle cx="${cx}" cy="${cy.toFixed(1)}" r="${rNeuron}" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>`;
        svg += `<text x="${cx}" y="${(cy + 3).toFixed(1)}" text-anchor="middle" font-size="8" fill="#15803d" font-family="ui-monospace,monospace">${escSvg(actLabel.slice(0, 7))}</text>`;
      }

      for (let j = 0; j < numOut; j++) {
        const cx = colOut, cy = outY(j);
        const lbl = numOut === 1 ? (mainCalc.output || 'ŷ') : `C${j}`;
        const sublbl = numOut === 1 ? (mainCalc.output || 'salida') : `clase ${j}`;
        svg += `<circle cx="${cx}" cy="${cy.toFixed(1)}" r="${rOut}" fill="#fef3e2" stroke="#f59e0b" stroke-width="1.5"/>`;
        svg += `<text x="${cx}" y="${(cy + 4).toFixed(1)}" text-anchor="middle" font-size="9" fill="#92400e" font-family="ui-monospace,monospace">${escSvg(lbl)}</text>`;
        svg += `<text x="${cx + rOut + 6}" y="${(cy + 4).toFixed(1)}" text-anchor="start" font-size="10" fill="#555" font-family="ui-sans-serif,sans-serif">${escSvg(sublbl)}</text>`;
      }

      const ly = svgH - 16;
      if (hasWeights) {
        svg += `<circle cx="12" cy="${ly}" r="5" fill="rgba(37,99,235,0.75)"/>`;
        svg += `<text x="20" y="${ly + 4}" font-size="10" fill="#555" font-family="ui-sans-serif,sans-serif">peso positivo</text>`;
        svg += `<circle cx="108" cy="${ly}" r="5" fill="rgba(220,38,38,0.75)"/>`;
        svg += `<text x="116" y="${ly + 4}" font-size="10" fill="#555" font-family="ui-sans-serif,sans-serif">peso negativo</text>`;
        svg += `<text x="${svgW - 6}" y="${ly + 4}" text-anchor="end" font-size="10" fill="#888" font-family="ui-sans-serif,sans-serif">grosor = |W|</text>`;
      } else {
        svg += `<text x="${svgW / 2}" y="${ly + 4}" text-anchor="middle" font-size="10" fill="#bbb" font-family="ui-sans-serif,sans-serif">Entrena para ver los pesos en las conexiones</text>`;
      }

      svg += '</svg>';

      const totalParams = numIn * numOut + numOut;
      const statusSpan = hasWeights
        ? '<span style="color:var(--ok);">✓ pesos entrenados</span>'
        : '<span style="color:var(--warn);">(entrena para ver pesos)</span>';
      const statsHtml = `<div style="margin-top:10px;font-size:12px;color:var(--muted);padding:0 4px;">
        <strong>${numIn}</strong> entradas &times; <strong>${numOut}</strong> neuronas
        = <strong>${numIn * numOut}</strong> pesos W + <strong>${numOut}</strong> bias b
        = <strong>${totalParams}</strong> parámetros ${statusSpan}.
      </div>`;

      el.innerHTML = `<h2 style="margin-bottom:12px">Arquitectura de red</h2>${svg}${statsHtml}`;
    }

    function renderHumanViews(visual) {
      if (!visual) return;
      const training = visual.training || {};
      const evaluation = visual.evaluation || {};
      const bc = visual.backend_contract || {};
      const spec = training.spec || {};
      const dataset = training.dataset || {};
      const verification = training.verification || {};
      const manifest = training.manifest || {};
      const evaluationReport = evaluation.report || {};
      const target = spec.dataset?.target || {};
      const split = spec.dataset?.split || {};
      const batch = spec.dataset?.batch || {};
      const trainingStatusLabel = training.ok
        ? ((training.warnings || []).length ? 'OK con aviso' : 'OK')
        : 'Bloqueado';
      const bcParams = bc.parameter_manifest || [];
      const bcErrors = bc.parameter_errors || [];
      const bcWarnings = bc.warnings || [];
      const bcOk = bc.ok;
      const bcStatusClass = bcErrors.length ? 'bad' : (bcWarnings.length ? 'warn' : 'ok');
      const bcStatusLabel = bcErrors.length ? 'No portable' : (bcWarnings.length ? 'Portable con avisos' : 'Portable');
      byId('modelView').innerHTML = `
        <h2>Modelo entendible</h2>
        ${table(
          [
            { key: 'name', label: 'Funcion' },
            { key: 'kind', label: 'Tipo' },
            { key: 'output', label: 'Salida' },
            { key: 'expression', label: 'Calculo' },
          ],
          visual.calculations || []
        )}
        <h2>Distribuciones</h2>
        ${table(
          [
            { key: 'name', label: 'Nombre' },
            { key: 'type', label: 'Tipo' },
            { key: 'source', label: 'Fuente' },
          ],
          visual.distributions || []
        )}
      `;
      byId('dataView').innerHTML = `
        <h2>Entradas tipadas</h2>
        ${table(
          [
            { key: 'vector', label: 'Vector' },
            { key: 'field', label: 'Campo' },
            { key: 'type', label: 'Tipo' },
            { key: 'range', label: 'Rango' },
          ],
          visual.inputs || []
        )}
        ${bc.target !== undefined ? `
        <h2>Diagnóstico de backend P3</h2>
        <div class="summary-grid">
          <div class="metric"><span class="metric-value">${escapeHtml(bc.target || 'stdlib')}</span><span class="metric-label">Target</span></div>
          <div class="metric"><span class="metric-value">${bcParams.length}</span><span class="metric-label">Parámetros entrenables</span></div>
          <div class="metric"><span class="metric-value">${bcStatusLabel}</span><span class="metric-label">Portabilidad</span></div>
        </div>
        ${bcParams.length ? `
          <h3 style="margin:10px 0 4px;">Manifiesto de parámetros</h3>
          ${table(
            [
              { key: 'name', label: 'Parámetro' },
              { key: 'function', label: 'Función' },
              { key: 'shapeLabel', label: 'Shape' },
              { key: 'trainable', label: 'Entrenable' },
            ],
            bcParams.map((p) => ({ ...p, shapeLabel: JSON.stringify(p.shape || []), trainable: p.trainable ? 'Sí' : 'No' }))
          )}
        ` : ''}
        ${bcErrors.length ? `<div class="callout bad">${escapeHtml(bcErrors.join(' | '))}</div>` : ''}
        ${bcWarnings.length ? `<div class="callout warn">${escapeHtml(bcWarnings.join(' | '))}</div>` : ''}
        ` : ''}
        <h2 id="trainingSection">Contrato de entrenamiento</h2>
        ${training.available ? `
          <div class="summary-grid">
            <div class="metric"><span class="metric-value">${escapeHtml(trainingStatusLabel)}</span><span class="metric-label">Validacion P4</span></div>
            <div class="metric"><span class="metric-value">${escapeHtml(dataset.rows ?? 0)}</span><span class="metric-label">Filas CSV</span></div>
            <div class="metric"><span class="metric-value">${escapeHtml((verification.trainable_parameters || []).length)}</span><span class="metric-label">Parametros</span></div>
          </div>
          ${table(
            [
              { key: 'item', label: 'Artefacto' },
              { key: 'value', label: 'Valor' },
            ],
            [
              { item: 'Modelo', value: spec.model || '' },
              { item: 'Dataset', value: spec.dataset?.source || '' },
              { item: 'Target', value: `${target.name || ''} ${target.type?.name || ''}`.trim() },
              { item: 'Loss', value: `${spec.loss?.type || ''} -> ${spec.loss?.prediction || ''}` },
              { item: 'Optimizer', value: `${spec.optimizer?.type || ''} lr=${spec.optimizer?.learning_rate ?? ''}` },
              { item: 'Update', value: (spec.optimizer?.update || []).join(', ') },
              { item: 'Split', value: split.train ? `train=${split.train} validation=${split.validation} seed=${split.seed ?? ''}` : '' },
              { item: 'Batch', value: batch.size ? `size=${batch.size} shuffle=${batch.shuffle}` : '' },
            ]
          )}
          <h2>Columnas CSV</h2>
          <pre>${escapeHtml((dataset.columns || []).join(', '))}</pre>
          <h2>Parametros entrenables</h2>
          ${table(
            [
              { key: 'name', label: 'Parametro' },
              { key: 'function', label: 'Funcion' },
              { key: 'shapeLabel', label: 'Shape' },
            ],
            (verification.trainable_parameters || []).map((parameter) => ({
              ...parameter,
              shapeLabel: JSON.stringify(parameter.shape || []),
            }))
          )}
          ${manifest.available ? `
            <h2>Manifiesto versionado</h2>
            <div class="callout ${manifest.ok ? '' : 'bad'}">${manifest.ok ? 'Hashes, filas y splits validados.' : escapeHtml(`El manifiesto tiene bloqueos: ${(manifest.verification?.errors || []).join(' | ')}`)}</div>
            <pre>${escapeHtml(JSON.stringify(manifest.verification || null, null, 2))}</pre>
          ` : `
            <h2>Manifiesto versionado</h2>
            <div class="callout warn">No hay manifiesto cargado. El CSV del .mxtrain existe y se valida, pero faltan hashes/splits versionados en esta demo.</div>
          `}
          ${training.ok ? '' : `<div class="callout bad">${escapeHtml([...(verification.errors || []), ...(training.errors || [])].join('\\n') || 'Contrato de entrenamiento no validado.')}</div>`}
        ` : `
          <div class="callout warn">${escapeHtml(training.message || 'Carga un .mxtrain para ver dataset, loss, parametros y trazas P4.')}</div>
        `}
        <div class="train-section" style="margin-top:20px;border-top:1px solid var(--line);padding-top:16px;">
          <h2 style="margin:0 0 10px;">Entrenar desde el playground <span style="font-size:11px;font-weight:400;color:var(--muted);">límites configurables por perfil (M12); el backend gobierna y avisa si recorta</span></h2>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px;">
            <button id="generateTrainingBtn" class="secondary" onclick="generateTraining()">Generar entrenamiento</button>
            <button id="generateDatasetBtn" class="secondary" onclick="generateSyntheticDataset()" title="Genera un dataset sintético reproducible desde el esquema del modelo (P12)">Generar dataset sintético</button>
            <span style="font-size:11px;color:var(--muted);">Filas:</span>
            <input id="syntheticRows" type="number" min="2" value="200" list="rowPresets" title="Sin tope en el frontend: lo gobierna el perfil de límites del backend (M12). Presets sugeridos hasta 1.000.000" style="width:84px;font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:4px;">
            <datalist id="rowPresets"><option value="200"><option value="1000"><option value="5000"><option value="20000"><option value="50000"><option value="100000"><option value="250000"><option value="500000"><option value="1000000"></datalist>
            <span style="font-size:11px;color:var(--muted);">Seed:</span>
            <input id="syntheticSeed" type="number" min="0" value="42" style="width:60px;font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:4px;">
            <select id="syntheticMode" style="font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:4px;">
              <option value="random">random</option>
              <option value="coherent">coherent</option>
            </select>
          </div>
          <div id="syntheticStatus" style="margin:0 0 8px;display:none;"></div>
          <div id="trainPanel">
            <h3>Dataset CSV</h3>
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
              <input type="file" id="csvFileInput" accept=".csv" style="font-size:12px;" onchange="handleCsvFile(event)">
            </div>
            <textarea id="csvPaste" rows="6" placeholder="Pega aquí el CSV de entrenamiento o usa el fichero de arriba..." style="width:100%;font-family:monospace;font-size:12px;padding:6px;border:1px solid var(--line);border-radius:4px;resize:vertical;"></textarea>
            <div style="margin:6px 0;display:flex;gap:8px;align-items:center;">
              <button id="validateCsvBtn" class="secondary" onclick="validateCsv()">Validar CSV</button>
              <span style="font-size:12px;color:var(--muted);">Épocas override (opcional):</span>
              <input id="epochsOverride" type="number" min="1" placeholder="auto" title="Sin tope en el frontend: lo gobierna el perfil de límites del backend (M12)" style="width:70px;font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:4px;">
              <button id="trainBtn" class="primary" onclick="runTraining()" disabled>Entrenar</button>
              <button id="stopTrainBtn" class="secondary hidden" onclick="cancelTraining()">Detener</button>
            </div>
            <div id="csvValidStatus" class="callout warn" style="margin:6px 0;display:none;"></div>
            <div id="trainFeedback" style="margin:6px 0;"></div>
            <div id="epochProgress" style="margin:8px 0;max-height:220px;overflow:auto;"></div>
            <div id="artifactsPanel" class="hidden"></div>
            <div id="predPanel" class="hidden train-section" style="border-top:1px solid var(--line);padding-top:12px;">
              <h3>Probar predicción con parámetros entrenados</h3>
              <textarea id="predInput" rows="4" placeholder="Input JSON (usa el de la columna central si está vacío)..." style="width:100%;font-family:monospace;font-size:12px;padding:6px;border:1px solid var(--line);border-radius:4px;resize:vertical;margin-bottom:6px;"></textarea>
              <button id="predBtn" class="primary" onclick="runPrediction()">Probar predicción</button>
              <pre id="predResult" style="margin-top:8px;font-size:12px;white-space:pre-wrap;"></pre>
            </div>
          </div>
        </div>
      `;
      // Show csvValidStatus only after first interaction
      const csvStatus = byId('csvValidStatus');
      if (csvStatus) csvStatus.style.display = '';
      const _isRegReport = evaluation.available && evaluationReport.mae !== undefined && !(evaluationReport.labels || []).length;
      byId('evaluationView').innerHTML = `
        <h2>Evaluacion</h2>
        ${evaluation.available ? `
          <div class="summary-grid">
            ${_isRegReport ? `
              <div class="metric"><span class="metric-value">${escapeHtml(evaluationReport.mae ?? '')}</span><span class="metric-label">MAE</span></div>
              <div class="metric"><span class="metric-value">${escapeHtml(evaluationReport.rmse ?? '')}</span><span class="metric-label">RMSE</span></div>
              <div class="metric"><span class="metric-value">${escapeHtml(evaluationReport.r2 ?? '')}</span><span class="metric-label">R²</span></div>
            ` : `
              <div class="metric"><span class="metric-value">${escapeHtml(evaluationReport.accuracy ?? '')}</span><span class="metric-label">Accuracy</span></div>
              <div class="metric"><span class="metric-value">${escapeHtml(evaluationReport.loss ?? '')}</span><span class="metric-label">Loss</span></div>
              <div class="metric"><span class="metric-value">${escapeHtml(evaluationReport.macro_f1 ?? '')}</span><span class="metric-label">Macro F1</span></div>
            `}
          </div>
          ${table(
            [
              { key: 'item', label: 'Artefacto' },
              { key: 'value', label: 'Valor' },
            ],
            [
              { item: 'Reporte', value: evaluation.report_path || 'evaluation_report.json' },
              { item: 'Backend', value: evaluation.backend?.target || evaluationReport.backend?.target || 'stdlib' },
              { item: 'Dataset', value: evaluationReport.dataset || '' },
              { item: 'Filas', value: evaluationReport.rows ?? '' },
              { item: 'ParameterSet', value: evaluationReport.parameter_set_id || '' },
              ...(_isRegReport ? [] : [{ item: 'Labels', value: (evaluationReport.labels || []).join(', ') }]),
            ]
          )}
          ${_isRegReport ? `
            <h2>Metricas de regresion</h2>
            <pre>${escapeHtml(JSON.stringify({ mae: evaluationReport.mae, rmse: evaluationReport.rmse, r2: evaluationReport.r2, loss: evaluationReport.loss }, null, 2))}</pre>
          ` : `
            <h2>Matriz de confusion</h2>
            <pre>${escapeHtml(JSON.stringify(evaluationReport.confusion_matrix || null, null, 2))}</pre>
            <h2>Metricas por label</h2>
            <pre>${escapeHtml(JSON.stringify(evaluationReport.per_label || null, null, 2))}</pre>
          `}
          ${evaluation.ok ? '' : `<div class="callout bad">${escapeHtml((evaluation.errors || []).join('\\n') || 'Reporte de evaluacion incompleto.')}</div>`}
        ` : `
          <div class="callout warn">${escapeHtml(evaluation.message || 'Carga un evaluation_report.json para ver metricas de evaluacion.')}</div>
          <pre>${escapeHtml('python3 -m matrixai evaluate model.mxai --training train_config.mxtrain --params params.best.json --data evaluation.csv --output evaluation_report.json')}</pre>
        `}
      `;
      byId('securityView').innerHTML = `
        <h2>Seguridad y simulacion</h2>
        ${table(
          [
            { key: 'name', label: 'Accion' },
            { key: 'when', label: 'Condicion' },
            { key: 'call', label: 'Llamada' },
            { key: 'policy', label: 'Politica' },
            { key: 'state', label: 'Estado' },
          ],
          (visual.actions || []).map((action) => ({
            ...action,
            state: action.allowed && action.simulated ? 'Simulada permitida' : 'Bloqueada',
          }))
        )}
        <pre>${JSON.stringify(visual.security || null, null, 2)}</pre>
      `;
      byId('apiView').innerHTML = `
        <h2>Servidor de prediccion (puerto 8000)</h2>
        <div class="callout" style="margin-bottom:14px;">
          <strong>Este Workbench corre en el puerto 8765</strong> — es solo la herramienta de analisis y diseno.<br>
          El <strong>servidor de prediccion</strong> es un proceso separado que expone el modelo como una API REST en el <strong>puerto 8000</strong>.
          Arrancalo cuando el modelo este listo para recibir peticiones reales.
        </div>
        <h3 style="margin:0 0 8px;">Arrancar el servidor</h3>
        <pre style="margin-bottom:16px;">${escapeHtml(visual.serving.command)}</pre>
        <h3 style="margin:0 0 8px;">Endpoints disponibles</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px;">
          <thead><tr style="background:var(--panel-soft);">
            <th style="text-align:left;padding:6px 10px;border:1px solid var(--line);">Endpoint</th>
            <th style="text-align:left;padding:6px 10px;border:1px solid var(--line);">URL</th>
            <th style="text-align:left;padding:6px 10px;border:1px solid var(--line);">Descripcion</th>
          </tr></thead>
          <tbody>
            <tr><td style="padding:5px 10px;border:1px solid var(--line);font-family:monospace;">GET /health</td>
                <td style="padding:5px 10px;border:1px solid var(--line);color:var(--muted);">${escapeHtml(visual.serving.health_url)}</td>
                <td style="padding:5px 10px;border:1px solid var(--line);">Comprueba que el servidor esta vivo. Sin autenticacion.</td></tr>
            <tr><td style="padding:5px 10px;border:1px solid var(--line);font-family:monospace;">POST /predict</td>
                <td style="padding:5px 10px;border:1px solid var(--line);color:var(--muted);">${escapeHtml(visual.serving.predict_url)}</td>
                <td style="padding:5px 10px;border:1px solid var(--line);">Ejecuta el modelo con el JSON de entrada. Requiere Bearer Auth.</td></tr>
            <tr><td style="padding:5px 10px;border:1px solid var(--line);font-family:monospace;">GET /docs</td>
                <td style="padding:5px 10px;border:1px solid var(--line);color:var(--muted);">${escapeHtml(visual.serving.docs_url)}</td>
                <td style="padding:5px 10px;border:1px solid var(--line);">Swagger UI interactivo. Prueba los endpoints desde el navegador.</td></tr>
            <tr><td style="padding:5px 10px;border:1px solid var(--line);font-family:monospace;">GET /openapi.json</td>
                <td style="padding:5px 10px;border:1px solid var(--line);color:var(--muted);">${escapeHtml(visual.serving.openapi_url)}</td>
                <td style="padding:5px 10px;border:1px solid var(--line);">Esquema OpenAPI 3.x. Importalo en Postman, Insomnia o cualquier cliente.</td></tr>
          </tbody>
        </table>
        <h3 style="margin:0 0 8px;">Autenticacion</h3>
        <div class="callout" style="margin-bottom:14px;font-family:monospace;font-size:13px;">${escapeHtml(visual.serving.auth)}</div>
        <h3 style="margin:0 0 8px;">Ejemplo con curl</h3>
        <pre>${escapeHtml(visual.serving.curl)}</pre>
      `;
    }

    function render(data) {
      const previousSignature = state.lastSignature;
      const nextSignature = JSON.stringify(data);
      const unchanged = Boolean(previousSignature && previousSignature === nextSignature);
      const hasWarnings = hasWorkflowWarnings(data);
      state.last = data;
      state.lastSignature = nextSignature;
      byId('status').textContent = data.ok
        ? (hasWarnings ? 'Accepted with warnings' : 'Accepted')
        : 'Needs attention';
      byId('status').className = `status ${data.ok ? (hasWarnings ? 'warn' : 'ok') : 'bad'}`;
      const time = new Date().toLocaleTimeString();
      byId('statusDetail').textContent = `Ultimo analisis: ${time}`;
      byId('analysisNote').className = `action-note ${data.ok ? (hasWarnings ? 'warn' : 'ok') : 'bad'}`;
      let noteText = unchanged
        ? `Analisis repetido a las ${time}: no hay cambios visibles en el resultado.`
        : data.ok
          ? (
              hasWarnings
                ? `Analisis completado a las ${time}: paneles actualizados con avisos.`
                : `Analisis completado a las ${time}: paneles actualizados y checks aceptados.`
            )
          : `Analisis completado a las ${time}: revisa Recorrido y Checks para ver bloqueos.`;
      if (state.pendingModeNote) { noteText += ' — ' + state.pendingModeNote; state.pendingModeNote = ''; }
      byId('analysisNote').textContent = noteText;
      renderWorkflow(data);
      renderDemoGuide(data);
      renderOverview(data.visual_model);
      renderDiagnostics(data);
      renderHumanViews(data.visual_model);
      focusFirstBlockedView(data);
      byId('checks').innerHTML = '';
      (data.checks || []).forEach((check) => {
        const item = document.createElement('div');
        item.className = `check ${check.ok ? 'ok' : 'bad'}`;
        const errors = (check.errors || []).join('\\n');
        const warnings = (check.warnings || []).join('\\n');
        item.innerHTML = `<div class="name">${check.name}</div>`;
        if (errors || warnings) {
          const detail = document.createElement('div');
          detail.className = 'detail';
          detail.textContent = [errors, warnings].filter(Boolean).join('\\n');
          item.appendChild(detail);
        }
        byId('checks').appendChild(item);
      });
      renderPipelineView(data.pipeline_stages || [], data.artifacts || null);
      renderArtifactBadges(data.artifacts || null, data.mode || 'prompt');
      byId('semanticOut').textContent = data.semantic_text || '';
      byId('mxaiOut').textContent = data.mxai || '';
      byId('irOut').textContent = JSON.stringify(data.program || null, null, 2);
      renderGraphView(data.graph_mermaid || '');
      refineState.lastRunResult = data.run_result || null;
      refineState.chain = [];
      refineState.parentHash = '';
      byId('refineResult').style.display = 'none';
      byId('refineIteration').value = '1';
      renderRunView(data.run_result || null);
      renderDiagnoseView(data.diagnose || null);
      window._lastVisualModel = data.visual_model || null;
      renderNetworkView(data.visual_model || null, null);
      // LLM mode indicator
      const llmMode = data.llm_mode;
      if (llmMode) {
        const el = byId('llmModeIndicator');
        el.style.display = '';
        if (llmMode.active) {
          el.style.cssText = 'display:block;margin-top:6px;font-size:12px;padding:7px 10px;border-radius:6px;line-height:1.5;background:#e6f4ea;border:1px solid #a8d5b5;color:#1a6b38;';
          el.textContent = `● LLM activo: ${llmMode.provider} (${llmMode.model})`;
        } else {
          el.style.cssText = 'display:block;margin-top:6px;font-size:12px;padding:7px 10px;border-radius:6px;line-height:1.5;background:#fef9e7;border:1px solid #f0d78c;color:#7a5c00;';
          el.textContent = '○ Modo determinístico — configura MATRIXAI_LLM_API_KEY para dominio abierto';
        }
      }
      // Show editable .semantic field after prompt-mode analysis
      if (data.mode === 'prompt' && data.semantic_text) {
        byId('semanticField').style.display = '';
      }
      // supervision_source badge in checks area
      if (data.supervision_source) {
        const src = data.supervision_source === 'llm'
          ? `✓ Propuesta generada por LLM (${data.llm_model || data.llm_provider || 'externo'})`
          : '↺ Propuesta generada por agente determinístico';
        const item = document.createElement('div');
        item.className = `check ${data.supervision_source === 'llm' ? 'ok' : ''}`;
        item.style.cssText = 'opacity:0.8;font-style:italic;';
        item.innerHTML = `<div class="name">supervision_source: ${data.supervision_source}</div><div class="detail">${src}</div>`;
        byId('checks').prepend(item);
      }
    }

    function renderDelta(original, proposed) {
      const origLines = original.split('\\n');
      const propLines = proposed.split('\\n');
      const maxLen = Math.max(origLines.length, propLines.length);
      let html = '<table style="width:100%;font-size:12px;font-family:monospace;border-collapse:collapse;">';
      html += '<tr><th style="padding:4px 8px;text-align:left;background:var(--panel-soft);border-bottom:1px solid var(--line);">Original</th>';
      html += '<th style="padding:4px 8px;text-align:left;background:var(--panel-soft);border-bottom:1px solid var(--line);">Propuesto</th></tr>';
      for (let i = 0; i < maxLen; i++) {
        const a = origLines[i] ?? '';
        const b = propLines[i] ?? '';
        const changed = a !== b;
        const style = changed ? 'background:#fff3cd;' : '';
        html += `<tr style="${style}"><td style="padding:3px 8px;border-bottom:1px solid var(--line);vertical-align:top;">${escapeHtml(a)}</td>`;
        html += `<td style="padding:3px 8px;border-bottom:1px solid var(--line);vertical-align:top;">${escapeHtml(b)}</td></tr>`;
      }
      html += '</table>';
      return html;
    }

    function applyRefinedPrompt(proposed, refinementId) {
      byId('prompt').value = proposed;
      const iter = parseInt(byId('refineIteration').value || '1', 10);
      byId('refineIteration').value = String(iter + 1);
      const chainLen = refineState.chain.length;
      const hashPfx = refineState.parentHash ? refineState.parentHash.slice(0, 8) : '';
      byId('refineResult').innerHTML += `<div class="callout ok" style="margin-top:6px;font-size:12px;">Prompt aplicado. Cadena: ${chainLen} entradas. Hash raiz: ${escapeHtml(hashPfx)}... Pulsa Analizar para regenerar.</div>`;
    }

    function showRefineResult(data) {
      const el = byId('refineResult');
      el.style.display = '';
      if (!data.ok) {
        const limitMsg = data.iteration_limit_reached
          ? `<strong>Limite de iteraciones alcanzado.</strong> ${escapeHtml(data.error)}`
          : escapeHtml(data.error || 'Error desconocido.');
        el.innerHTML = `<div class="callout bad" style="margin-bottom:8px;">${limitMsg}</div>`;
        return;
      }
      const acceptedCss = data.supervision_accepted ? 'ok' : 'warn';
      const acceptedLabel = data.supervision_accepted ? 'Supervision: ACEPTADO' : 'Supervision: RECHAZADO';
      const chainLen = (data.chain || []).length;
      const hashPfx = data.parent_hash ? data.parent_hash.slice(0, 8) : '';
      const header = `<div class="callout ${acceptedCss}" style="margin-bottom:8px;">
        <strong>${acceptedLabel}</strong> &nbsp;|&nbsp; ID: ${escapeHtml(data.refinement_id)} &nbsp;|&nbsp;
        Modo: ${escapeHtml(data.mode)} &nbsp;|&nbsp; Iter: ${data.iteration} &nbsp;|&nbsp;
        Cadena: ${chainLen} &nbsp;|&nbsp; Hash raiz: ${escapeHtml(hashPfx)}...
      </div>`;
      const explanation = data.explanation
        ? `<div style="font-size:12px;margin-bottom:8px;padding:6px 8px;background:var(--panel-soft);border:1px solid var(--line);border-radius:4px;white-space:pre-wrap;">${escapeHtml(data.explanation)}</div>`
        : '';
      const delta = renderDelta(byId('prompt').value, data.proposed_prompt || '');
      const applyBtn = data.supervision_accepted
        ? `<button class="primary" style="font-size:12px;margin-top:8px;" onclick="applyRefinedPrompt(${JSON.stringify(data.proposed_prompt)}, ${JSON.stringify(data.refinement_id)})">Aplicar prompt propuesto</button>`
        : '';
      el.innerHTML = header + explanation + '<h3 style="margin:8px 0 4px;font-size:13px;">Delta</h3>' + delta + applyBtn;
    }

    async function refinePrompt() {
      const run = refineState.lastRunResult;
      if (!run) { alert('Ejecuta un analisis con Input JSON primero.'); return; }
      const prompt = byId('prompt').value.trim();
      if (!prompt) { alert('El prompt esta vacio.'); return; }
      const hintsRaw = byId('refineHints').value.trim();
      const hints = hintsRaw ? hintsRaw.split('\\n').map((h) => h.trim()).filter(Boolean) : [];
      const iterationCount = parseInt(byId('refineIteration').value || '1', 10);
      const maxIter = parseInt(byId('refineMaxIter').value || '3', 10);
      const btn = byId('refineBtn');
      btn.disabled = true;
      btn.textContent = 'Refinando...';
      byId('refineResult').style.display = 'none';
      try {
        const res = await fetch('/api/refine', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            mxai_text: byId('mxaiOut').textContent.trim() || byId('mxai').value.trim(),
            run_result: run,
            hints,
            iteration_count: iterationCount,
            refinement_chain: refineState.chain,
            parent_prompt_hash: refineState.parentHash,
            max_iterations: maxIter,
          }),
        });
        const data = await res.json();
        if (data.ok) {
          refineState.chain = data.chain || [];
          refineState.parentHash = data.parent_hash || '';
        }
        showRefineResult(data);
      } catch (err) {
        byId('refineResult').style.display = '';
        byId('refineResult').innerHTML = `<div class="callout bad">Error de red: ${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Refinar prompt';
      }
    }

    async function analyze() {
      byId('status').textContent = 'Running';
      byId('status').className = 'status running';
      byId('statusDetail').textContent = 'Recalculando paneles...';
      byId('analysisNote').className = 'action-note running';
      byId('analysisNote').textContent = 'Analizando los datos actuales de la columna central.';
      byId('run').disabled = true;
      const payload = {
        mode: byId('mode').value,
        prompt: byId('prompt').value,
        semantic_text: byId('semantic').value,
        mxai_text: byId('mxai').value,
        training_text: byId('training').value,
        manifest_text: byId('manifest').value,
        evaluation_report_text: byId('evaluationReport').value,
        input_json: byId('input').value,
        use_llm: byId('useLlm')?.checked ?? false,
      };
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        render(data);
        const _promptMode = byId('mode').value === 'prompt';
        if (data.mxai && (_promptMode || !byId('mxai').value.trim())) byId('mxai').value = data.mxai;
        if (data.semantic_text && (_promptMode || !byId('semantic').value.trim())) byId('semantic').value = data.semantic_text;
        if (_promptMode) byId('training').value = data.training_text || '';
        else if (data.training_text && !byId('training').value.trim()) byId('training').value = data.training_text;
        const csvBox = byId('csvPaste');
        if (csvBox && data.dataset_template_text && (_promptMode || !csvBox.value.trim())) {
          csvBox.value = data.dataset_template_text;
          const csvStatus = byId('csvValidStatus');
          if (csvStatus) {
            csvStatus.textContent = 'CSV: plantilla generada desde el modelo actual. Valídala antes de entrenar.';
            csvStatus.className = 'callout warn';
            csvStatus.style.display = '';
          }
          const trainPanel = byId('trainPanel');
          if (trainPanel) trainPanel.classList.remove('hidden');
          const trainBtn = byId('trainBtn');
          if (trainBtn) trainBtn.disabled = true;
        }
        if (data.manifest_text && !byId('manifest').value.trim()) byId('manifest').value = data.manifest_text;
        if (data.evaluation_report_text && !byId('evaluationReport').value.trim()) byId('evaluationReport').value = data.evaluation_report_text;
      } catch (error) {
        const time = new Date().toLocaleTimeString();
        byId('status').textContent = 'Needs attention';
        byId('status').className = 'status bad';
        byId('statusDetail').textContent = `Error de analisis: ${time}`;
        byId('analysisNote').className = 'action-note bad';
        byId('analysisNote').textContent = `No se pudo ejecutar el analisis: ${error.message || error}`;
      } finally {
        byId('run').disabled = false;
      }
    }

    async function loadExample(exampleId) {
      byId('status').textContent = 'Loading example';
      byId('status').className = 'status';
      const response = await fetch(`/api/example/${exampleId}`);
      const data = await response.json();
      const prevMode = byId('mode').value;
      const nextMode = data.mode || 'mxai';
      byId('mode').value = nextMode;
      updateModeVisibility();
      if (prevMode !== nextMode) {
        state.pendingModeNote = `Entrada cambiada automaticamente a \u201c${nextMode}\u201d porque el ejemplo usa un archivo .mxai precompilado.`;
      }
      byId('prompt').value = '';
      byId('semantic').value = '';
      byId('mxai').value = data.mxai_text || '';
      byId('training').value = data.training_text || '';
      byId('manifest').value = data.manifest_text || '';
      byId('evaluationReport').value = data.evaluation_report_text || '';
      byId('input').value = data.input_json || '';
      analyze();
    }

    document.querySelectorAll('.tab').forEach((tab) => {
      tab.addEventListener('click', () => showTab(tab.dataset.tab));
    });
    document.addEventListener('click', (event) => {
      const exampleButton = event.target.closest('[data-demo-example]');
      if (exampleButton) loadExample(exampleButton.dataset.demoExample);
      const tabButton = event.target.closest('[data-demo-tab]');
      if (tabButton) {
        showTab(tabButton.dataset.demoTab);
        const scrollId = tabButton.dataset.demoScroll;
        if (scrollId) {
          const target = document.getElementById(scrollId);
          if (target) setTimeout(() => target.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
        }
      }
    });
    byId('run').addEventListener('click', analyze);
    byId('fillEmail').addEventListener('click', () => loadExample('email-agent'));
    byId('fillFallRisk').addEventListener('click', () => loadExample('fall-risk'));
    byId('mode').addEventListener('change', updateModeVisibility);
    byId('reanalizeSemanticBtn').addEventListener('click', () => {
      const semText = byId('semantic').value.trim();
      if (!semText) return;
      byId('mode').value = 'semantic';
      updateModeVisibility();
      analyze();
    });
    loadDefaults().then(() => { updateModeVisibility(); });
  </script>
</body>
</html>
"""

