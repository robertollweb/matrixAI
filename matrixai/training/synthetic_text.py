# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""SECUENCIAS_PRODUCTO C3 — generación sintética de texto (decisión 5 del
contrato, 3 orígenes de etiqueta):

- **random**: texto de palabras neutras SIN relación con la etiqueta
  (asignada también al azar) — sin señal, declarado como tal
  (`label_origin="synthetic_random"`, honestidad M8-C1).
- **template**: plantillas deterministas con un TOKEN-SEÑAL por clase — el
  "suelo sin LLM" (decisión 5): siempre produce una tarea de clasificación
  real y aprendible, sin depender de ningún proveedor externo
  (`label_origin="synthetic_template"`).
- **LLM (M8v2)**: el LLM redacta N ejemplos de texto por clase en una única
  llamada; se validan (cobertura mínima por clase) y, si fallan o no hay
  proveedor, se cae al determinista de plantillas — nunca al azar (la
  plantilla YA es señal real; degradar a random tirarla seria peor que no
  usar LLM en absoluto). `label_origin="synthetic_llm_examples"`.

Nunca importa `matrixai.text`/tokenizador: produce pares (texto, etiqueta)
crudos — la tokenización ocurre en el boundary de train
(`CSVTextDataAdapter`), no aquí (invariante 2 del contrato).
"""
from __future__ import annotations

import random
import re

_RANDOM_WORDS = [
    "lorem", "ipsum", "dato", "elemento", "valor", "muestra", "registro",
    "articulo", "nota", "entrada", "bloque", "fragmento", "seccion", "unidad",
    "informe", "detalle", "resumen", "consulta",
]

_TEMPLATE_FILLERS = [
    "el producto", "el servicio", "la experiencia", "el pedido", "la entrega",
    "la atencion", "el resultado", "el proceso", "la respuesta", "el equipo",
]

_LLM_TEXT_SYSTEM = (
    "You write short example texts for a text classification dataset. For "
    "EACH class given, write exactly N distinct example sentences a human "
    "would recognize as belonging to that class. Output STRICTLY as lines "
    "in the form 'CLASS: example text' — one per line, nothing else (no "
    "headers, no numbering, no explanations). Keep each example under 200 "
    "characters. Never repeat an example."
)

_LLM_LINE_RE = re.compile(r"^\s*(?P<label>[^:\n]{1,64}):\s*(?P<text>.+?)\s*$")


def generate_random_examples(n: int, labels: list[str], seed: int) -> list[tuple[str, str]]:
    """`n` ejemplos con texto de palabras aleatorias sin relación con la
    etiqueta, también asignada al azar."""
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for _ in range(n):
        length = rng.randint(4, 12)
        text = " ".join(rng.choice(_RANDOM_WORDS) for _ in range(length))
        examples.append((text, rng.choice(labels)))
    return examples


def _signal_token(label: str) -> str:
    """Token-señal determinista y único por clase — misma clase produce
    SIEMPRE el mismo token (en cualquier proceso/máquina), distinto entre
    clases distintas, sin depender de `hash()` (no determinista entre
    procesos por PYTHONHASHSEED)."""
    from matrixai.training.dense_generator import _identifier
    ident = _identifier(label) or "clase"
    return f"senal_{ident}"


def generate_template_examples(n: int, labels: list[str], seed: int) -> list[tuple[str, str]]:
    """Plantillas deterministas: relleno neutro + el token-señal de la clase
    en una posición aleatoria dentro de la frase. Cobertura uniforme entre
    clases (ronda round-robin), orden barajado con el mismo seed."""
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for i in range(n):
        label = labels[i % len(labels)]
        token = _signal_token(label)
        filler = [rng.choice(_TEMPLATE_FILLERS) for _ in range(rng.randint(2, 5))]
        filler.insert(rng.randint(0, len(filler)), token)
        examples.append((" ".join(filler), label))
    rng.shuffle(examples)
    return examples


def parse_llm_text_examples(raw: str, labels: list[str]) -> dict[str, list[str]]:
    """Parsea 'CLASE: texto' por línea. Solo sobreviven líneas cuya CLASE
    (normalizada, sin distinguir mayúsculas/espacios) coincide EXACTAMENTE
    con una de las `labels` declaradas — cabeceras/numeración/explicaciones
    que el LLM haya colado se descartan en vez de intentar interpretarlas."""
    by_label: dict[str, list[str]] = {label: [] for label in labels}
    label_lookup = {label.strip().lower(): label for label in labels}
    for line in raw.splitlines():
        m = _LLM_LINE_RE.match(line)
        if not m:
            continue
        canonical = label_lookup.get(m.group("label").strip().lower())
        if canonical is None:
            continue
        text = m.group("text").strip()
        if text and text not in by_label[canonical]:
            by_label[canonical].append(text)
    return by_label


def validate_llm_text_examples(
    by_label: dict[str, list[str]], labels: list[str], min_per_class: int = 2,
) -> list[str]:
    """Lista de problemas (vacía = usable) — mismo patrón que
    `DomainRules.validate`."""
    missing = [l for l in labels if len(by_label.get(l, [])) < min_per_class]
    if missing:
        return [
            f"insufficient examples for classes: {', '.join(missing)} "
            f"(need >= {min_per_class} each)"
        ]
    return []


def llm_text_examples(prompt: str, labels: list[str], n_per_class: int) -> dict[str, list[str]] | None:
    """Pide al LLM N ejemplos de texto por clase en una única llamada;
    devuelve `None` ante CUALQUIER fallo (sin proveedor, error de red,
    respuesta vacía) — el caller valida y cae al determinista de plantillas
    (mismo patrón que `_llm_domain_rules`, M8v2)."""
    from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider
    try:
        provider = ChatCompletionsLLMProposalProvider.from_env()
        user = (
            f"Problem: {prompt}\n"
            f"Classes: {', '.join(labels)}\n"
            f"Examples per class: {n_per_class}"
        )
        text = provider.complete(_LLM_TEXT_SYSTEM, user)
        return parse_llm_text_examples(text, labels)
    except Exception:  # noqa: BLE001
        return None


def _rows_from_by_label(
    by_label: dict[str, list[str]], n: int, labels: list[str], seed: int,
) -> list[tuple[str, str]]:
    """`n` filas cubriendo las clases round-robin, reciclando el pool de
    ejemplos por clase si `n` pide más filas de las que el LLM redactó."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    counters = {label: 0 for label in labels}
    for i in range(n):
        label = labels[i % len(labels)]
        pool = by_label[label]
        rows.append((pool[counters[label] % len(pool)], label))
        counters[label] += 1
    rng.shuffle(rows)
    return rows


def generate_text_examples(
    mode: str, prompt: str, n: int, labels: list[str], seed: int, use_llm: bool = False,
) -> tuple[list[tuple[str, str]], str]:
    """Punto de entrada único de los 3 orígenes (decisión 5 del contrato).
    `mode == "random"` → sin señal; cualquier otro `mode` (típicamente
    `"coherent"`, igual que el generador tabular) → plantillas deterministas,
    o LLM si `use_llm=True` y la respuesta valida, con fallback a plantillas
    — NUNCA a random (la plantilla ya es señal real).

    Devuelve `(filas, label_origin)`."""
    if not labels:
        raise ValueError("generate_text_examples requires at least one label")
    if mode == "random":
        return generate_random_examples(n, labels, seed), "synthetic_random"
    if use_llm:
        n_per_class = max(1, -(-n // len(labels)))  # ceil division
        by_label = llm_text_examples(prompt, labels, n_per_class)
        if by_label is not None and not validate_llm_text_examples(by_label, labels):
            return _rows_from_by_label(by_label, n, labels, seed), "synthetic_llm_examples"
    return generate_template_examples(n, labels, seed), "synthetic_template"
