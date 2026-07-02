# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Shared parser for explicit field-type declarations in a generation prompt.

Single source of truth for turning declarations like

    edad: Scalar en [18, 95]
    segmento: Categorical[PARTICULAR, PYME, PREMIUM, BANCA_PRIVADA]
    tiene_hipoteca: Boolean

into typed :class:`FieldSpec`. Consumed by the deterministic generator, the
composite generator, the LLM dispatch and (via metadata) the export, so all
paths honour the SAME declared types — see GENERACION_TIPOS_PROMPT_CONTRACT.md
(C1) and its "Gramática del prompt" section.

C1 scope: parsing + normalization + validation ONLY. Materializing a
categorical into one-hot/embedding, a boolean into a typed column, etc. is the
generator's job (C2–C5). This module never touches the .mxai.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    """One field's declared type. ``kind`` ∈ {scalar, categorical, boolean}.

    - categorical: ``values`` holds the ordered, de-duplicated HUMAN vocabulary
      (spaces/accents preserved); the canonical human input stays this field.
    - scalar: ``range`` holds ``(min, max)`` when declared and valid; ``integer``
      is True when declared as Integer.
    """
    name: str
    kind: str
    values: tuple[str, ...] | None = None
    range: tuple[float, float] | None = None
    integer: bool = False


@dataclass(frozen=True)
class FieldSpecParse:
    """Result of parsing a prompt: the typed fields plus any normalization notices."""
    fields: list[FieldSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_name(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields}


# Type keyword → canonical family. Case-insensitive (see regex flag).
_TYPE_ALIASES = {
    "scalar": "scalar", "number": "scalar", "numeric": "scalar", "float": "scalar",
    "integer": "integer", "int": "integer",
    "categorical": "categorical", "category": "categorical", "categorica": "categorical",
    "categórica": "categorical",
    "boolean": "boolean", "bool": "boolean", "booleano": "boolean",
}

# Matches "<name>: <Type>" optionally followed by "en/in/de" and a "[...]" payload.
# The NAME is captured WHOLE up to the ':' (delimited by a field boundary — start
# of line, newline, ',', ';' or a preceding ':'), so accents/spaces/hyphens are
# kept and sanitized afterwards instead of being truncated mid-token. The type
# keyword is what disqualifies non-field "word: value" lines (PROYECTO:, DOMINIO:,
# ...) and the output (ProbabilityMap): only these keywords count as a field type.
_FIELD_ENTRY_RE = re.compile(
    r"(?:^|[\n,;:])"                       # field boundary
    r"(?P<name>[^\n,;:]+?)"                # full name up to ':' (accents/spaces/hyphens ok)
    r"\s*:\s*"
    r"(?P<type>categorical|categorica|categórica|category|boolean|booleano|bool|"
    r"scalar|numeric|number|float|integer|int)\b"
    r"(?:\s+(?:en|in|de))?\s*"
    r"(?:\[(?P<args>[^\]]*)\])?",
    re.IGNORECASE | re.MULTILINE,
)


def strip_field_specs(prompt: str) -> str:
    """Remove explicit ``name: <Type>[...]`` declarations, leaving a ``;`` separator.

    Lets a legacy bare-name extractor (dense_generator._extract_fields) see ONLY the
    untyped fields, instead of mangling typed declarations into garbage names like
    ``segmento_categorical`` / ``scalar``. Used by C2's typed+bare field merge.

    Replaced with a comma (not ';'): the legacy extractor splits on ',' but STOPS its
    capture at ';', so a ',' keeps the surrounding bare fields visible.
    """
    return _FIELD_ENTRY_RE.sub(", ", prompt or "")


def parse_field_specs(prompt: str) -> FieldSpecParse:
    """Parse the EXPLICIT field-type declarations from ``prompt``.

    Only declarations of the form ``name: <Type>[...]`` are returned; bare field
    names (no ``: Type``) are out of scope here (the legacy extractor treats them
    as scalars). Invalid declarations are NORMALIZED, never accepted raw:
    a malformed/inverted scalar range is dropped (plain scalar) and a categorical
    with <2 values is downgraded to scalar, each with a warning.
    """
    specs: list[FieldSpec] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for m in _FIELD_ENTRY_RE.finditer(prompt or ""):
        name = _sanitize_name(m.group("name"))
        if not name or name in seen:
            continue
        canonical = _TYPE_ALIASES.get(m.group("type").lower().replace("á", "a"), "scalar")
        args = (m.group("args") or "").strip()

        if canonical == "categorical":
            values = _dedup(args.split(","))
            if len(values) < 2:
                warnings.append(
                    f"campo {name!r}: Categorical con menos de 2 valores; se trata como escalar"
                )
                specs.append(FieldSpec(name=name, kind="scalar"))
            else:
                specs.append(FieldSpec(name=name, kind="categorical", values=tuple(values)))
        elif canonical == "boolean":
            specs.append(FieldSpec(name=name, kind="boolean"))
        else:  # scalar or integer
            rng = _parse_range(args, name, warnings)
            specs.append(FieldSpec(name=name, kind="scalar", range=rng,
                                   integer=(canonical == "integer")))
        seen.add(name)

    return FieldSpecParse(fields=specs, warnings=warnings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(value: str) -> str:
    """Identifier-safe field name (accents stripped, lowercased). '' if unusable."""
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text if text and not text[0].isdigit() else ""


def _dedup(raw_values: list[str]) -> list[str]:
    """Trim, drop empties, de-duplicate preserving order. Human form is kept."""
    seen: set[str] = set()
    out: list[str] = []
    for v in raw_values:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _parse_range(args: str, name: str, warnings: list[str]) -> tuple[float, float] | None:
    """Parse a ``[min, max]`` scalar range. Invalid → None (+ warning), never raw."""
    if not args:
        return None
    parts = [p.strip() for p in args.split(",")]
    if len(parts) != 2:
        warnings.append(f"campo {name!r}: rango escalar mal formado [{args}]; se ignora el rango")
        return None
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        warnings.append(f"campo {name!r}: límites de rango no numéricos [{args}]; se ignora")
        return None
    if not (lo < hi):
        warnings.append(
            f"campo {name!r}: rango invertido o degenerado [{lo}, {hi}]; se ignora el rango"
        )
        return None
    return (lo, hi)
