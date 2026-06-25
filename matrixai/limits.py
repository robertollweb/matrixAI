# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""M12 — Límites operativos configurables del playground.

Los topes (filas, épocas, tamaño CSV, profundidad, nº de clases) protegen el playground
HOSTED compartido (anti-DoS). El Studio DESCARGABLE corre en la máquina del usuario, así
que debe poder subirlos o quitarlos (su máquina, su responsabilidad) — misma filosofía que
`MATRIXAI_TRAIN_TIMEOUT=0` (ver project_training_limits).

Resolución por límite, en este orden:
  1. `MATRIXAI_HOSTED=1` → topes DUROS del perfil "equilibrado"; se ignora cualquier
     override (anti-DoS). "Sin límite" NO existe en hosted.
  2. Override por-límite por env (`MATRIXAI_MAX_ROWS`, `MATRIXAI_MAX_EPOCHS`, ...): un
     entero positivo (tope) o `0`/`none`/`unlimited` (sin tope).
  3. Perfil `MATRIXAI_LIMITS_PROFILE` = equilibrado (default) | avanzado | ilimitado.
  4. Default = equilibrado (los valores de hoy).

Un límite `None` significa "sin tope" (el código lo trata como ilimitado). `_MIN_RELU_WIDTH`
NO se gestiona aquí: es una corrección de sanidad (la ReLU muere por debajo de 16), no un
tope de capacidad.
"""
from __future__ import annotations

import os

# Perfil "equilibrado" = valores de hoy (defaults). Cada valor: int (tope) o None (sin tope).
_EQUILIBRADO: dict[str, int | None] = {
    "max_rows": 50_000,
    "max_epochs": 1_000,
    "max_csv_bytes": 50_000_000,
    "max_depth": 12,
    "max_labels": 12,
}
# Perfil "avanzado": máquina potente; topes altos pero aún con red de seguridad anti-typo.
_AVANZADO: dict[str, int | None] = {
    "max_rows": 1_000_000,
    "max_epochs": 100_000,
    "max_csv_bytes": 1_000_000_000,
    "max_depth": 128,
    "max_labels": 128,
}
# Perfil "ilimitado": sin topes. SOLO descargable (hosted nunca lo ofrece).
_ILIMITADO: dict[str, int | None] = {k: None for k in _EQUILIBRADO}

_PROFILES: dict[str, dict[str, int | None]] = {
    "equilibrado": _EQUILIBRADO,
    "avanzado": _AVANZADO,
    "ilimitado": _ILIMITADO,
}

_ENV_BY_KEY = {
    "max_rows": "MATRIXAI_MAX_ROWS",
    "max_epochs": "MATRIXAI_MAX_EPOCHS",
    "max_csv_bytes": "MATRIXAI_MAX_CSV_BYTES",
    "max_depth": "MATRIXAI_MAX_DEPTH",
    "max_labels": "MATRIXAI_MAX_LABELS",
}

_UNLIMITED_TOKENS = {"0", "none", "unlimited", "ilimitado", "sin", "off"}

_INVALID = object()  # centinela: override que no parsea (se ignora, cae al perfil)


def is_hosted() -> bool:
    """True si corre como playground hosted compartido (topes duros, anti-DoS)."""
    return os.environ.get("MATRIXAI_HOSTED", "0") == "1"


def _profile_name() -> str:
    name = os.environ.get("MATRIXAI_LIMITS_PROFILE", "equilibrado").strip().lower()
    return name if name in _PROFILES else "equilibrado"


def _parse_override(raw: str):
    """int (tope), None (sin tope) o _INVALID si no parsea."""
    token = raw.strip().lower()
    if token in _UNLIMITED_TOKENS:
        return None
    try:
        value = int(token)
    except ValueError:
        return _INVALID
    return None if value <= 0 else value


def get_limit(key: str) -> int | None:
    """Tope efectivo (int) o None (sin tope) para `key`, aplicando hosted/env/perfil."""
    if key not in _EQUILIBRADO:
        raise KeyError(f"unknown limit {key!r}")
    # 1. Hosted: topes duros del perfil equilibrado; ignora overrides (anti-DoS).
    if is_hosted():
        return _EQUILIBRADO[key]
    # 2. Override por-límite por env.
    raw = os.environ.get(_ENV_BY_KEY[key])
    if raw is not None:
        parsed = _parse_override(raw)
        if parsed is not _INVALID:
            return parsed  # int o None
    # 3/4. Perfil (equilibrado por defecto).
    return _PROFILES[_profile_name()][key]


def cap(value: int, key: str) -> int:
    """Aplica el tope `key` a `value`: `value` si el tope es None, si no `min(value, tope)`."""
    limit = get_limit(key)
    return value if limit is None else min(value, limit)


def exceeds(value: int, key: str) -> bool:
    """True si `value` supera el tope `key` (False si no hay tope)."""
    limit = get_limit(key)
    return limit is not None and value > limit


def limits_snapshot() -> dict:
    """Estado de los límites para el endpoint /config y la UI (M12 Corte UI)."""
    hosted = is_hosted()
    return {
        "hosted": hosted,
        "profile": "equilibrado" if hosted else _profile_name(),
        "limits": {k: get_limit(k) for k in _EQUILIBRADO},
        "profiles_available": ["equilibrado", "avanzado"] + ([] if hosted else ["ilimitado"]),
    }
