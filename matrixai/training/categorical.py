# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""S2-C2 — One-hot expansion of categorical features.

A categorical field (``tipo_ingreso`` with values urgente/programado/traslado)
is expanded to one Scalar column per value (``tipo_ingreso__urgente`` …). This
avoids the false ordinality of a single Scalar (where traslado=0.66 would sit
"between" programado and urgente). A Dense layer over one-hot input is
mathematically a trainable embedding, so no native EMBEDDING op is needed.

The expansion rewrites the model VECTOR (fields + arity) and the training
``FROM COLUMNS`` list, keeping every other line untouched. It is a pure text
transform so it can run on any saved/generated model and travels through the
same generate → train → snapshot → export plumbing as field_ranges/field_types.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Separator between a categorical column and its value in the expanded name.
# Double underscore is unlikely to collide with normal column names and lets
# the inference UI split the group back out ("tipo_ingreso__urgente").
ONE_HOT_SEP = "__"


@dataclass
class ExpansionResult:
    mxai_text: str
    training_text: str
    # original categorical column -> ordered expanded column names
    groups: dict[str, list[str]] = field(default_factory=dict)
    # expanded column name -> (original column, value) for the inference UI
    members: dict[str, tuple[str, str]] = field(default_factory=dict)


def _sanitize_value(value: str) -> str:
    """Turn an arbitrary category value into an identifier-safe suffix."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def _build_group_names(col: str, values: list[str]) -> list[str]:
    """Expanded column names for one categorical column, collisions disambiguated."""
    names: list[str] = []
    seen: set[str] = set()
    for i, value in enumerate(values):
        suffix = _sanitize_value(value) or f"v{i}"
        name = f"{col}{ONE_HOT_SEP}{suffix}"
        if name in seen:  # two values sanitize to the same token
            name = f"{name}_{i}"
        seen.add(name)
        names.append(name)
    return names


def _rewrite_vector_block(mxai_text: str, groups: dict[str, list[str]]) -> str:
    """Replace each categorical field line with its one-hot lines and fix arity."""
    lines = mxai_text.splitlines()
    out: list[str] = []
    in_vector = False
    vector_header_idx: int | None = None
    field_count = 0

    field_re = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*:\s*(.+?)\s*$")

    for line in lines:
        stripped = line.strip()
        if not in_vector and stripped.upper().startswith("VECTOR "):
            in_vector = True
            vector_header_idx = len(out)
            out.append(line)
            field_count = 0
            continue
        if in_vector:
            if stripped.upper() == "END":
                # Patch the arity [N] on the header now that we know field_count
                if vector_header_idx is not None:
                    out[vector_header_idx] = re.sub(
                        r"\[\s*\d+\s*\]", f"[{field_count}]", out[vector_header_idx], count=1,
                    )
                in_vector = False
                vector_header_idx = None
                out.append(line)
                continue
            m = field_re.match(line)
            if m:
                indent, name, _type = m.groups()
                if name in groups:
                    for expanded in groups[name]:
                        out.append(f"{indent}{expanded}: Scalar")
                        field_count += 1
                else:
                    out.append(line)
                    field_count += 1
                continue
            # Non-field line inside the block (blank/comment): keep as-is
            out.append(line)
            continue
        out.append(line)

    return "\n".join(out) + ("\n" if mxai_text.endswith("\n") else "")


def _rewrite_from_columns(training_text: str, groups: dict[str, list[str]]) -> str:
    """Replace categorical columns inside FROM COLUMNS [...] with their members."""
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        cols = [c.strip() for c in inner.split(",") if c.strip()]
        expanded: list[str] = []
        for c in cols:
            expanded.extend(groups.get(c, [c]))
        return "FROM COLUMNS [" + ", ".join(expanded) + "]"

    return re.sub(r"FROM COLUMNS\s*\[([^\]]*)\]", repl, training_text)


def expand_categoricals(
    mxai_text: str,
    training_text: str,
    field_categories: dict[str, list[str]] | None,
) -> ExpansionResult:
    """Expand declared categorical columns to one-hot in model + training text.

    field_categories maps an existing VECTOR column to its ordered value list
    (>= 2 values). Columns not present in the VECTOR are ignored. Returns the
    rewritten texts plus the group/member maps the generator and inference UI
    need. With no categoricals the inputs are returned unchanged.
    """
    if not field_categories:
        return ExpansionResult(mxai_text=mxai_text, training_text=training_text)

    # Only expand columns that actually exist in the model VECTOR.
    vector_fields = _vector_field_names(mxai_text)
    groups: dict[str, list[str]] = {}
    for col, values in field_categories.items():
        vals = [str(v) for v in (values or []) if str(v).strip()]
        if col in vector_fields and len(vals) >= 2:
            groups[col] = _build_group_names(col, vals)

    if not groups:
        return ExpansionResult(mxai_text=mxai_text, training_text=training_text)

    new_mxai = _rewrite_vector_block(mxai_text, groups)
    new_training = _rewrite_from_columns(training_text, groups)
    members: dict[str, tuple[str, str]] = {}
    for col, values in field_categories.items():
        if col in groups:
            for name, value in zip(groups[col], [str(v) for v in values if str(v).strip()]):
                members[name] = (col, value)

    return ExpansionResult(
        mxai_text=new_mxai,
        training_text=new_training,
        groups=groups,
        members=members,
    )


def _remove_vector_fields(mxai_text: str, drop: set[str]) -> str:
    """Drop the given field lines from the VECTOR block and fix arity."""
    lines = mxai_text.splitlines()
    out: list[str] = []
    in_vector = False
    header_idx: int | None = None
    field_count = 0
    field_re = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*:\s*(.+?)\s*$")
    for line in lines:
        stripped = line.strip()
        if not in_vector and stripped.upper().startswith("VECTOR "):
            in_vector = True
            header_idx = len(out)
            out.append(line)
            field_count = 0
            continue
        if in_vector:
            if stripped.upper() == "END":
                if header_idx is not None:
                    out[header_idx] = re.sub(
                        r"\[\s*\d+\s*\]", f"[{field_count}]", out[header_idx], count=1,
                    )
                in_vector = False
                header_idx = None
                out.append(line)
                continue
            m = field_re.match(line)
            if m:
                name = m.group(2)
                if name in drop:
                    continue  # remove this feature
                field_count += 1
                out.append(line)
                continue
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if mxai_text.endswith("\n") else "")


def _remove_from_columns(training_text: str, drop: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        cols = [c.strip() for c in match.group(1).split(",") if c.strip()]
        kept = [c for c in cols if c not in drop]
        return "FROM COLUMNS [" + ", ".join(kept) + "]"
    return re.sub(r"FROM COLUMNS\s*\[([^\]]*)\]", repl, training_text)


def exclude_identifiers(
    mxai_text: str,
    training_text: str,
    identifiers: list[str] | None,
) -> tuple[str, str, list[str]]:
    """S2-C4 — drop identifier columns (patient_id …) from model + training.

    Identifiers carry no predictive signal and invite memorization, so they
    must not be training features. Removes them from the VECTOR (fields +
    arity) and the FROM COLUMNS list. Refuses to remove the last remaining
    feature (a model needs at least one input). Returns the rewritten texts and
    the list of columns actually excluded.
    """
    existing = _vector_field_names(mxai_text)
    ids = [c for c in (identifiers or []) if c in existing]
    if not ids:
        return mxai_text, training_text, []
    # never strip every feature — keep at least one input column
    if len(existing) - len(set(ids)) < 1:
        return mxai_text, training_text, []
    drop = set(ids)
    return (
        _remove_vector_fields(mxai_text, drop),
        _remove_from_columns(training_text, drop),
        ids,
    )


def _vector_field_names(mxai_text: str) -> set[str]:
    names: set[str] = set()
    in_vector = False
    field_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*.+$")
    for line in mxai_text.splitlines():
        stripped = line.strip()
        if not in_vector and stripped.upper().startswith("VECTOR "):
            in_vector = True
            continue
        if in_vector:
            if stripped.upper() == "END":
                break
            m = field_re.match(line)
            if m:
                names.add(m.group(1))
    return names
