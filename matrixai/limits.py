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
    # SECUENCIAS_PRODUCTO C2 (auditoría [ALTA]): Text[L] sin tope permite pedir
    # una SEQUENCE arbitrariamente grande — la atención del bloque transformer
    # escala O(L²) y el .mxtrain/CSV generados escalan O(L) por fila (varias
    # veces el texto: mxai, training_text, dataset_template_text).
    "max_sequence_length": 512,
    # CONTRATO 64 C1 — presupuesto de PARÁMETROS de la red. Sin esto no se podía
    # prometer "topes del perfil" para el tamaño del modelo: el resto de topes
    # acota los datos y el entrenamiento, nunca la capacidad. 2.000.000 son unos
    # 8 MB de pesos en float32 y entrenan en CPU en minutos, que es el listón del
    # perfil por defecto.
    "max_params": 2_000_000,
}
# Perfil "avanzado": máquina potente; topes altos pero aún con red de seguridad anti-typo.
_AVANZADO: dict[str, int | None] = {
    "max_rows": 1_000_000,
    "max_epochs": 100_000,
    "max_csv_bytes": 1_000_000_000,
    "max_depth": 128,
    "max_labels": 128,
    "max_sequence_length": 8_192,
    # ~200 MB de pesos: razonable en una GPU de portátil y en Colab, inviable en
    # CPU — que es exactamente la diferencia que separa este perfil del anterior.
    "max_params": 50_000_000,
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
    "max_sequence_length": "MATRIXAI_MAX_SEQUENCE_LENGTH",
    "max_params": "MATRIXAI_MAX_PARAMS",
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


# ── Error estructurado de tope superado (CONTRATO 62 C1) ─────────────────────
#
# Antes, cada sitio que comprobaba un tope componía su propia frase ("CSV tiene
# N filas, máximo M", "Dataset sintetico supera el límite de N KB"), y quien la
# recibía solo podía reconocerla parseando texto. Peor: `dataset_project` la
# envolvía en "esto indica un hueco en la preparación del CSV, no un problema de
# tus datos", que para un tope es FALSO — la causa es el perfil y la solución es
# cambiarlo. Ahora el core emite datos; el texto humano es solo respaldo y la
# localización ES/EN vive en la SPA, que es quien sabe de idiomas.

LIMIT_EXCEEDED = "limit_exceeded"

# Unidades por límite, para que el consumidor formatee sin adivinar.
_LIMIT_UNITS = {
    "max_rows": "rows",
    "max_epochs": "epochs",
    "max_csv_bytes": "bytes",
    "max_depth": "layers",
    "max_labels": "labels",
    "max_sequence_length": "chars",
    "max_params": "params",
}


def limit_error(key: str, actual: int) -> dict:
    """Payload estructurado de un tope superado, listo para `{"ok": False, **payload}`.

    `configurable` es exactamente la condición con la que `get_limit` ignora los
    overrides: en hosted los topes son duros (anti-DoS) y ofrecer "cámbialo en
    Ajustes" sería mentir. `error` es texto de respaldo en español para quien no
    entienda la forma estructurada (CLI, logs, clientes viejos).
    """
    maximum = get_limit(key)
    hosted = is_hosted()
    return {
        "error_kind": LIMIT_EXCEEDED,
        "limit_key": key,
        "unit": _LIMIT_UNITS.get(key, "units"),
        "actual": actual,
        "maximum": maximum,
        "profile": "equilibrado" if hosted else _profile_name(),
        "configurable": not hosted,
        "error": _limit_error_text(key, actual, maximum, hosted),
    }


# Etiqueta en español SOLO para el texto de respaldo; `unit` (en inglés) es el
# token estable que consume la SPA para localizar. No se mezclan.
_LIMIT_UNITS_ES = {
    "max_rows": "filas",
    "max_epochs": "épocas",
    "max_depth": "capas",
    "max_labels": "etiquetas",
    "max_sequence_length": "caracteres",
    "max_params": "parámetros",
}


def human_bytes(n: int) -> str:
    """Tamaño legible con la unidad ADECUADA a la magnitud.

    Formatear siempre en MB daba "El CSV ocupa 0.0 MB y el máximo es 0 MB" para
    un tope pequeño — un mensaje inútil justo cuando más falta hace entenderlo.
    """
    if n < 1_000:
        return f"{n} B"
    if n < 1_000_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n / 1_000_000:.1f} MB"


def _limit_error_text(key: str, actual: int, maximum: int | None, hosted: bool) -> str:
    if key == "max_csv_bytes":
        actual_h = human_bytes(actual)
        max_h = "sin límite" if maximum is None else human_bytes(maximum)
        cuerpo = f"El CSV ocupa {actual_h} y el máximo es {max_h}"
    elif key == "max_params":
        # El sujeto no es el dataset sino la RED: decir "el dataset tiene
        # 3.400.000 parámetros" no significaría nada.
        max_h = "sin límite" if maximum is None else f"{maximum:,}".replace(",", ".")
        cuerpo = (f"La red tiene {actual:,}".replace(",", ".")
                  + f" parámetros y el máximo es {max_h}")
    else:
        unit = _LIMIT_UNITS_ES.get(key, "unidades")
        max_h = "sin límite" if maximum is None else f"{maximum}"
        cuerpo = f"El dataset tiene {actual} {unit} y el máximo es {max_h}"
    if hosted:
        return f"{cuerpo}. Es un límite del servicio compartido y no se puede subir."
    return (f"{cuerpo}. Puedes subirlo en Ajustes → Límites "
            f"(perfil «avanzado» o «ilimitado»).")


def is_limit_error(payload: object) -> bool:
    """True si `payload` es (o contiene) un error estructurado de tope."""
    return isinstance(payload, dict) and payload.get("error_kind") == LIMIT_EXCEEDED


def limits_snapshot() -> dict:
    """Estado de los límites para el endpoint /config y la UI (M12 Corte UI)."""
    hosted = is_hosted()
    return {
        "hosted": hosted,
        "profile": "equilibrado" if hosted else _profile_name(),
        "limits": {k: get_limit(k) for k in _EQUILIBRADO},
        "profiles_available": ["equilibrado", "avanzado"] + ([] if hosted else ["ilimitado"]),
    }
