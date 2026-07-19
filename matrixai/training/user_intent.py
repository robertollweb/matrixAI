# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Contrato 58 (BIBLIOTECA_MEJORAS_USO_REAL) C4 — intención local del
usuario en el flujo "desde datos": texto libre que describe qué se espera
del modelo, persistido en la procedencia como metadata auditable.

Invariante central (decisión D del contrato, confirmada por una auditoría
propia de este mismo corte): la intención NUNCA entra al prompt tipado que
alimenta al generador determinista. Un prefijo ingenuo como
`"PROYECTO: <texto>"` NO basta para aislarla — aunque el parser de campos
(`prompt_field_specs.py`) ignore esa línea, los detectores GLOBALES de
`analyze_playground_request` (p.ej. las palabras "residual"/"deep"/
"workflow"/"temporal") siguen leyendo el texto completo del prompt y pueden
cambiar de generador o de supervisor sin que el usuario lo pida. Por eso
`generate_project_from_dataset` mantiene la intención en un canal
COMPLETAMENTE separado (la procedencia), nunca concatenada al prompt
sintetizado."""
from __future__ import annotations

import unicodedata

USER_INTENT_MAX_CHARS = 1000


class UserIntentError(Exception):
    """`user_intent` inválido tras normalizar — mensaje siempre accionable."""


def normalize_user_intent(value: str | None) -> str | None:
    """Normaliza la intención local del usuario:

    1. Unicode NFC (para que el límite de longitud sea estable frente a
       formas compuestas/descompuestas del mismo texto).
    2. Colapsa CUALQUIER whitespace Unicode (incluidos saltos de línea —
       una intención multilínea se aplana a una sola línea con espacios
       simples) a un único espacio ASCII.
    3. Quita cualquier carácter de control restante (categoría Unicode
       `Cc` — NUL, ESC, DEL...; los que eran whitespace ya se colapsaron
       en el paso 2).
    4. Cadena vacía o compuesta solo por whitespace/controles → `None`
       (equivalente a "sin intención").
    5. Valida el límite de longitud DESPUÉS de normalizar — supera el
       límite → `UserIntentError` (mensaje accionable, nunca se trunca en
       silencio: el usuario decide qué recortar, no nosotros)."""
    if value is None:
        return None
    text = unicodedata.normalize("NFC", value)
    text = " ".join(text.split())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    if not text:
        return None
    if len(text) > USER_INTENT_MAX_CHARS:
        raise UserIntentError(
            f"La intención supera el límite de {USER_INTENT_MAX_CHARS} "
            f"caracteres tras normalizar (tiene {len(text)})."
        )
    return text
