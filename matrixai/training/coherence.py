# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""M9 — Validación de coherencia VECTOR ↔ campos ↔ columnas del dataset.

Cuando el `.mxai` (a menudo emitido por el LLM arquitecto) es inconsistente — un VECTOR
declarado con N dimensiones pero con menos campos enumerados (elipsis `…`, "crea las
columnas", o lista truncada), o `INPUT FROM COLUMNS` que no casa con los campos del VECTOR —
la generación de dataset falla con un error técnico opaco ("VECTOR In declares size 100 but
has 3 fields"). Este módulo detecta esos desajustes ANTES de generar y devuelve un mensaje
accionable y bilingüe para que el usuario corrija el prompt.

Es un escaneo estructural ligero (regex), independiente del parser estricto, para poder
explicar el problema incluso en los casos en que el parser sólo lanzaría una excepción.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_VECTOR_HDR_RE = re.compile(r"^\s*VECTOR\s+(?P<name>[A-Za-z_]\w*)\[(?P<size>\d+)\]\s*$")
_FIELD_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)(?:\s*:\s*(?P<type>.+))?\s*$")
_END_RE = re.compile(r"^\s*END\s*$")
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")


@dataclass(frozen=True)
class CoherenceResult:
    ok: bool
    error_es: str | None = None
    error_en: str | None = None


@dataclass(frozen=True)
class _Vector:
    name: str
    size: int
    fields: list[str]
    has_ellipsis: bool


def _scan_vectors(mxai_text: str) -> list[_Vector]:
    lines = mxai_text.splitlines()
    vectors: list[_Vector] = []
    i = 0
    while i < len(lines):
        m = _VECTOR_HDR_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name, size = m.group("name"), int(m.group("size"))
        fields: list[str] = []
        has_ellipsis = False
        i += 1
        while i < len(lines) and not _END_RE.match(lines[i]):
            line = lines[i]
            if line.strip():
                if _ELLIPSIS_RE.search(line):
                    has_ellipsis = True
                elif _FIELD_RE.match(line):
                    fields.append(_FIELD_RE.match(line).group("name"))
            i += 1
        vectors.append(_Vector(name, size, fields, has_ellipsis))
        i += 1
    return vectors


def check_dataset_coherence(mxai_text: str, training_text: str) -> CoherenceResult:
    """Detecta desajustes dimensión↔campos. Devuelve el primer problema (el más accionable)
    como mensaje es/en, o ok=True si todo cuadra (o no hay datos suficientes para juzgar —
    en cuyo caso el parser/pipeline normal sigue su curso).

    Sólo marca como error los casos que el parser estricto YA rechaza (elipsis / dimensión ≠
    nº de campos), pero con un mensaje accionable y bilingüe en vez del error técnico. No
    impone reglas nuevas (p. ej. columnas extra en INPUT, que el pipeline tolera)."""
    vectors = _scan_vectors(mxai_text)

    # 1) Elipsis en los campos de un VECTOR → casi seguro lista truncada.
    for v in vectors:
        if v.has_ellipsis:
            return CoherenceResult(
                False,
                error_es=(
                    f"El VECTOR {v.name} usa '…' en su lista de campos. Enumera cada feature "
                    f"explícitamente (p. ej. sensor_001, sensor_002, …, sensor_{v.size:03d}); "
                    f"el modelo no puede inferir los nombres a partir de la elipsis."
                ),
                error_en=(
                    f"VECTOR {v.name} uses '…' in its field list. Enumerate every feature "
                    f"explicitly (e.g. sensor_001, sensor_002, …, sensor_{v.size:03d}); the "
                    f"model cannot infer the names from the ellipsis."
                ),
            )

    # 2) Dimensión declarada ≠ nº de campos enumerados.
    for v in vectors:
        if len(v.fields) != v.size:
            return CoherenceResult(
                False,
                error_es=(
                    f"El VECTOR {v.name} declara {v.size} entradas pero solo se han enumerado "
                    f"{len(v.fields)} campos. Enumera cada feature explícitamente (sin '…') o "
                    f"ajusta la dimensión del vector a {len(v.fields)}."
                ),
                error_en=(
                    f"VECTOR {v.name} declares {v.size} inputs but only {len(v.fields)} fields "
                    f"are enumerated. List every feature explicitly (no '…') or set the vector "
                    f"dimension to {len(v.fields)}."
                ),
            )

    return CoherenceResult(True)
