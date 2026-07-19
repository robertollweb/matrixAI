# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Contrato 58 (BIBLIOTECA_MEJORAS_USO_REAL) C5 — interpretación LLM
OPT-IN de la intención local (C4): un canal COMPLETAMENTE separado del
prompt tipado que alimenta al generador. El LLM nunca ve el CSV ni sus
filas — solo `llm_context` (nombres/tipos/rangos/categorías/target/tarea,
ya decididos, + la intención normalizada) — y solo puede proponer la FORMA
de una red densa (tamaños de capas ocultas), nunca features/tipos/rangos/
vocabularios/target/pipeline (eso ya está fijado quando este módulo se
invoca).

La propuesta pasa SIEMPRE por `sanitize_hidden_layers` (M8-A1, el mismo
saneador que ya protege cualquier arquitectura densa venga de donde venga)
antes de viajar como hint hacia el generador real."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider
from matrixai.training.dense_generator import DenseNetworkGenerator, sanitize_hidden_layers

_MAX_LAYERS = DenseNetworkGenerator._MAX_EXPLICIT_DEPTH
_MAX_WIDTH = DenseNetworkGenerator._MAX_EXPLICIT_WIDTH

_INTENT_ARCHITECTURE_SYSTEM = (
    "You are a neural network architect. You are given a FIXED input schema "
    "(feature names, types, ranges, categories, target and task — already "
    "decided, you cannot change them) and a short free-text note describing "
    "what the user wants the model to prioritize. Propose ONLY the hidden-"
    "layer sizes for a dense feed-forward network. Respond with EXACTLY "
    "these lines, no extra text:\n\n"
    "LAYERS: comma-separated hidden layer sizes (2-12 integers, each "
    "between 8 and 16384)\n"
    "RATIONALE: one short sentence justifying the choice\n\n"
    "Rules:\n"
    "- A genuinely simple problem (few features, a clear linear signal) "
    "deserves FEWER, narrower layers.\n"
    "- A genuinely complex problem (the note describes nuanced trade-offs, "
    "many interacting factors) can justify MORE/WIDER layers.\n"
    "- You cannot change features, types, ranges, categories, or the "
    "target — only the network shape."
)


class IntentArchitectureError(Exception):
    """Fallo interpretando la intención vía LLM. `retryable=False` significa
    "no hay LLM configurado" (reintentar no ayudaría sin cambiar la
    configuración) — cualquier otro fallo (red, HTTP, timeout, respuesta sin
    LAYERS interpretable) es `retryable=True` (decisión E: nunca hay
    fallback silencioso — un fallo SIEMPRE se informa, nunca se ignora)."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class IntentArchitectureProposal:
    hidden_layers: list[tuple[int, str]]
    rationale: str | None
    raw_text: str
    provider: str
    model: str
    sanitizer_adjusted: bool


def build_llm_context(
    *, features: list[dict], task: str, target_column: str, user_intent: str,
) -> str:
    """Texto ESTRUCTURADO enviado al LLM — nunca filas, nunca el CSV. Cada
    entrada de `features` es `{"name", "type", "range"?, "categories"?}`."""
    lines = [
        f"User note: {user_intent}",
        f"Task: {task}",
        f"Target: {target_column}",
        "Features:",
    ]
    for f in features:
        detail = f["type"]
        if f.get("range") is not None:
            lo, hi = f["range"]
            detail += f" [{lo}, {hi}]"
        if f.get("categories"):
            detail += " (categories: " + ", ".join(str(c) for c in f["categories"]) + ")"
        lines.append(f"  - {f['name']}: {detail}")
    return "\n".join(lines)


def _parse_layers(text: str) -> tuple[list[tuple[int, str]] | None, str | None]:
    """Solo consume `LAYERS:`/`RATIONALE:` — cualquier otra línea (el LLM
    podría alucinar FIELDS/LABELS pese a las instrucciones) se ignora en
    silencio, nunca se aplica (invariante: el esquema no puede cambiar)."""
    hidden_layers: list[tuple[int, str]] | None = None
    rationale: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("LAYERS:"):
            try:
                sizes = [int(s.strip()) for s in line[len("LAYERS:"):].split(",") if s.strip()]
            except ValueError:
                continue
            sizes = [max(1, min(s, _MAX_WIDTH)) for s in sizes if s > 0][:_MAX_LAYERS]
            if sizes:
                hidden_layers = [(s, "relu") for s in sizes]
        elif line.startswith("RATIONALE:"):
            text_val = line[len("RATIONALE:"):].strip()
            if text_val:
                rationale = text_val[:240]
    return hidden_layers, rationale


def propose_intent_architecture(llm_context: str) -> IntentArchitectureProposal:
    """Llama al LLM configurado (mismo mecanismo que el resto del core,
    `ChatCompletionsLLMProposalProvider`), interpreta SOLO `LAYERS`/
    `RATIONALE`, y sanea el resultado (M8-A1) antes de devolverlo. Lanza
    `IntentArchitectureError` en cualquier fallo — nunca devuelve un
    resultado "vacío" en silencio."""
    try:
        provider = ChatCompletionsLLMProposalProvider.from_env()
    except ValueError as exc:
        raise IntentArchitectureError(f"No hay ningún LLM configurado: {exc}", retryable=False) from exc

    try:
        raw_text = provider.complete(_INTENT_ARCHITECTURE_SYSTEM, llm_context)
    except Exception as exc:  # noqa: BLE001 — cualquier fallo de transporte es recuperable
        raise IntentArchitectureError(f"La llamada al LLM falló: {exc}", retryable=True) from exc

    hidden_layers, rationale = _parse_layers(raw_text)
    if not hidden_layers:
        raise IntentArchitectureError(
            "El LLM no propuso una arquitectura interpretable (sin línea LAYERS válida).",
            retryable=True,
        )

    sanitized, notes = sanitize_hidden_layers(hidden_layers)
    return IntentArchitectureProposal(
        hidden_layers=sanitized,
        rationale=rationale,
        raw_text=raw_text,
        provider=provider.provider_name,
        model=provider.model_name,
        sanitizer_adjusted=bool(notes),
    )


def proposal_sha256(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
