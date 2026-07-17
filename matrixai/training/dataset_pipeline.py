# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""BIBLIOTECA_PROYECTOS_INTELIGENTES C3 — pipeline declarativo de preparación
de datos + anti-fuga temporal (contrato 57).

Vocabulario CERRADO de 8 operaciones (mismo espíritu que el parser de
`SPLIT`/`DATASET` en `training/parser.py`: un `dict` por operación, whitelist
estricta, `PipelineError` accionable ante cualquier cosa no reconocida —
nunca un pipeline a medias, invariante 7 del contrato):

  sort_temporal(column)                    — ordena por una columna fecha, ascendente
  drop_duplicates()                        — elimina filas EXACTAMENTE duplicadas
  missing_values(strategy, columns=None)   — "drop" o "interpolate" (solo numéricas)
  rename(mapping)                          — {columna_actual: columna_nueva}
  cast(column, to)                         — "number"/"integer"/"string"
  lag_window(column, k)                    — añade {column}_lag1..{column}_lagK (SIEMPRE hacia atrás, k>=1)
  shift_target(column, horizon, as=None)   — añade {column}_target_h{h} (SIEMPRE hacia delante, horizon>=1)
  drop_columns(columns)                    — elimina columnas

Sirve a AMBOS flujos del contrato (A: datos-primero, B: biblioteca de
plantillas) — quien declara la SECUENCIA de operaciones es la capa de
arriba (C4: UI del flujo A: columna temporal + ventana + horizonte; el
flujo B: la plantilla); este módulo solo es el MOTOR que las ejecuta.

Anti-fuga (invariante 6): "ninguna columna con información posterior a t,
sobre columnas DESPLAZADAS, no por nombre" — `check_anti_leakage` no mira
el NOMBRE de ninguna columna (un nombre inocuo no salva de una fuga real,
y una columna con nombre alarmante puede ser inocente); mira el HISTORIAL
REAL de operaciones: cualquier columna producida por `shift_target` queda
marcada como "desplazada al futuro" — si esa columna concreta aparece
listada como FEATURE, es fuga, se detecte como se detecte cómo se llame.
`lag_window` (siempre k>=1, hacia atrás) nunca puede producir una fuga por
construcción — usar el valor de AYER para predecir HOY es autorregresión
normal, no fuga.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from matrixai.training.dataset_analysis import _detect_date_format, _is_null

_OPS = frozenset({
    "sort_temporal", "drop_duplicates", "missing_values", "rename",
    "cast", "lag_window", "shift_target", "drop_columns",
})
_CAST_TYPES = frozenset({"number", "integer", "string"})
_MISSING_STRATEGIES = frozenset({"drop", "interpolate"})


class PipelineError(ValueError):
    """Operación/parámetro inválido, o dato que no encaja en el pipeline
    declarado — error accionable (invariante 7 del contrato 57): nunca un
    pipeline a medias."""


@dataclass(frozen=True)
class PipelineStepResult:
    """Procedencia de UN paso — orden (posición en `steps`), parámetros,
    filas antes/después (invariante 3 del contrato: "pipeline en orden")."""
    operation: str
    params: dict[str, Any]
    rows_before: int
    rows_after: int
    columns_before: list[str]
    columns_after: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "params": dict(self.params),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_before": list(self.columns_before),
            "columns_after": list(self.columns_after),
        }


@dataclass(frozen=True)
class PipelineResult:
    rows: list[dict[str, str]]
    steps: list[PipelineStepResult] = field(default_factory=list)
    # Columnas producidas por `shift_target` — nunca por nombre, por
    # HISTORIAL (ver docstring del módulo). `check_anti_leakage` las usa.
    forward_shifted_columns: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [dict(r) for r in self.rows],
            "steps": [s.to_dict() for s in self.steps],
            "forward_shifted_columns": sorted(self.forward_shifted_columns),
        }


def run_pipeline(
    rows: list[dict[str, str]], operations: list[dict[str, Any]]
) -> PipelineResult:
    """Ejecuta las `operations` (vocabulario cerrado, ver docstring del
    módulo) EN ORDEN sobre `rows`. Nunca muta la lista/dicts de entrada del
    caller (copia defensiva) — devuelve un `PipelineResult` nuevo con la
    procedencia completa de cada paso."""
    if not rows:
        raise PipelineError("El pipeline necesita al menos una fila de entrada.")
    current: list[dict[str, str]] = [dict(row) for row in rows]
    columns_order = list(rows[0].keys())
    forward_shifted: set[str] = set()
    steps: list[PipelineStepResult] = []

    for i, op_decl in enumerate(operations):
        if not isinstance(op_decl, dict) or "op" not in op_decl:
            raise PipelineError(f"Paso {i}: cada operación necesita una clave 'op'.")
        op = op_decl["op"]
        if op not in _OPS:
            raise PipelineError(
                f"Paso {i}: operación desconocida {op!r}. Vocabulario cerrado: "
                f"{sorted(_OPS)}."
            )
        params = {k: v for k, v in op_decl.items() if k != "op"}
        rows_before, cols_before = len(current), list(columns_order)

        if op == "sort_temporal":
            _op_sort_temporal(current, params, columns_order)
        elif op == "drop_duplicates":
            current = _op_drop_duplicates(current, columns_order)
        elif op == "missing_values":
            current = _op_missing_values(current, params, columns_order)
        elif op == "rename":
            columns_order = _op_rename(current, params, columns_order)
        elif op == "cast":
            _op_cast(current, params, columns_order)
        elif op == "lag_window":
            columns_order = _op_lag_window(current, params, columns_order)
        elif op == "shift_target":
            columns_order = _op_shift_target(current, params, columns_order, forward_shifted)
        elif op == "drop_columns":
            columns_order = _op_drop_columns(current, params, columns_order)

        steps.append(PipelineStepResult(
            operation=op, params=params,
            rows_before=rows_before, rows_after=len(current),
            columns_before=cols_before, columns_after=list(columns_order),
        ))

    return PipelineResult(rows=current, steps=steps, forward_shifted_columns=frozenset(forward_shifted))


def check_anti_leakage(result: PipelineResult, feature_columns: list[str]) -> list[str]:
    """Invariante 6: ninguna FEATURE puede ser una columna que `shift_target`
    desplazó al futuro — detectado por HISTORIAL de operaciones (`result.
    forward_shifted_columns`), nunca por el nombre de la columna."""
    leaking = sorted(set(feature_columns) & result.forward_shifted_columns)
    return [
        f"La columna {col!r} se generó con shift_target (valor DESPLAZADO al "
        "futuro respecto a su fila) y no puede usarse como feature — fuga de "
        "información temporal (invariante 6 del contrato 57)."
        for col in leaking
    ]


def validate_pipeline_output(
    rows: list[dict[str, str]], *, target_column: str, min_rows: int = 2,
) -> list[str]:
    """Validación final del contrato ("min_rows, tipos, nulos residuales,
    target presente"): `tipos` ya lo garantiza `cast` (falla cerrado al
    aplicarse, no hace falta re-chequear); aquí quedan las tres
    comprobaciones que solo se pueden hacer sobre el resultado YA
    completo del pipeline."""
    if len(rows) < min_rows:
        return [f"El pipeline deja {len(rows)} fila(s) — hacen falta al menos {min_rows}."]
    if target_column not in rows[0]:
        return [f"La columna objetivo {target_column!r} no existe tras el pipeline."]
    residual_nulls = sum(1 for row in rows if _is_null(row.get(target_column)))
    if residual_nulls:
        return [
            f"{residual_nulls} fila(s) se quedan sin valor en la columna "
            f"objetivo {target_column!r} tras el pipeline (¿falta un "
            "missing_values(strategy=drop) al final para limpiar los bordes "
            "de un lag_window/shift_target?)."
        ]
    return []


# ---------------------------------------------------------------------------
# Operaciones
# ---------------------------------------------------------------------------

def _require_column(col: Any, columns_order: list[str], op: str, field_name: str = "column") -> str:
    if not isinstance(col, str) or col not in columns_order:
        raise PipelineError(f"{op}: {field_name} {col!r} no existe. Columnas: {columns_order}.")
    return col


def _op_sort_temporal(rows: list[dict[str, str]], params: dict[str, Any], columns_order: list[str]) -> None:
    column = _require_column(params.get("column"), columns_order, "sort_temporal")
    values = [row.get(column) for row in rows if not _is_null(row.get(column))]
    # `_detect_date_format` (dataset_analysis.py, reutilizado — misma
    # detección que usa C1) devuelve el PRIMER formato de la lista sin
    # validar nada cuando `values` está vacía — guardia explícita aquí:
    # una columna temporal enteramente nula no tiene NADA que ordenar.
    fmt = _detect_date_format(values) if values else None
    if fmt is None:
        raise PipelineError(
            f"sort_temporal: la columna {column!r} no tiene un formato de fecha "
            "reconocible en todas sus filas."
        )

    def _key(row: dict[str, str]) -> datetime:
        v = row.get(column)
        return datetime.min if _is_null(v) else datetime.strptime(v, fmt)

    rows.sort(key=_key)  # estable: filas con la misma fecha conservan su orden relativo


def _op_drop_duplicates(rows: list[dict[str, str]], columns_order: list[str]) -> list[dict[str, str]]:
    seen: set[tuple[str | None, ...]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = tuple(row.get(c) for c in columns_order)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _op_missing_values(
    rows: list[dict[str, str]], params: dict[str, Any], columns_order: list[str]
) -> list[dict[str, str]]:
    strategy = params.get("strategy")
    if strategy not in _MISSING_STRATEGIES:
        raise PipelineError(
            f"missing_values: strategy debe ser {sorted(_MISSING_STRATEGIES)} "
            f"(recibido {strategy!r})."
        )
    columns = params.get("columns") or list(columns_order)
    if not isinstance(columns, list):
        raise PipelineError("missing_values: columns debe ser una lista de nombres de columna.")
    for c in columns:
        _require_column(c, columns_order, "missing_values", "columns")
    if strategy == "drop":
        return [row for row in rows if not any(_is_null(row.get(c)) for c in columns)]
    for c in columns:
        _interpolate_column(rows, c)
    return rows


def _interpolate_column(rows: list[dict[str, str]], column: str) -> None:
    n = len(rows)
    values: list[float | None] = []
    for row in rows:
        raw = row.get(column)
        if _is_null(raw):
            values.append(None)
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            raise PipelineError(
                f"missing_values(interpolate): el valor {raw!r} de la columna "
                f"{column!r} no es numérico — interpolate solo aplica a columnas "
                "numéricas."
            )
    i = 0
    while i < n:
        if values[i] is not None:
            i += 1
            continue
        j = i
        while j < n and values[j] is None:
            j += 1
        prev_idx, next_idx = i - 1, j if j < n else None
        if prev_idx >= 0 and next_idx is not None:
            prev_v, next_v = values[prev_idx], values[next_idx]
            span = next_idx - prev_idx
            for k in range(i, j):
                values[k] = prev_v + (next_v - prev_v) * (k - prev_idx) / span
        # borde inicial/final sin vecino conocido en un lado: queda vacío
        # (un missing_values(strategy=drop) posterior lo limpia si hace falta)
        i = j
    for row, v in zip(rows, values):
        if v is not None:
            row[column] = _fmt_num(v)


def _op_rename(
    rows: list[dict[str, str]], params: dict[str, Any], columns_order: list[str]
) -> list[str]:
    mapping = params.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise PipelineError("rename: mapping debe ser un objeto {columna_actual: columna_nueva} no vacío.")
    for old in mapping:
        _require_column(old, columns_order, "rename", "mapping")
    new_names = list(mapping.values())
    if len(set(new_names)) != len(new_names):
        raise PipelineError("rename: hay nombres nuevos duplicados en mapping.")
    kept = [c for c in columns_order if c not in mapping]
    collision = set(new_names) & set(kept)
    if collision:
        raise PipelineError(
            f"rename: el nuevo nombre {sorted(collision)[0]!r} choca con una "
            "columna existente que no se está renombrando."
        )
    for row in rows:
        for old, new in mapping.items():
            row[new] = row.pop(old)
    return [mapping.get(c, c) for c in columns_order]


def _op_cast(rows: list[dict[str, str]], params: dict[str, Any], columns_order: list[str]) -> None:
    column = _require_column(params.get("column"), columns_order, "cast")
    to = params.get("to")
    if to not in _CAST_TYPES:
        raise PipelineError(f"cast: to debe ser {sorted(_CAST_TYPES)} (recibido {to!r}).")
    if to == "string":
        return
    for row in rows:
        raw = row.get(column)
        if _is_null(raw):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise PipelineError(f"cast: el valor {raw!r} de la columna {column!r} no es numérico.")
        if to == "integer":
            if not value.is_integer():
                raise PipelineError(
                    f"cast: el valor {raw!r} de la columna {column!r} no es un entero exacto."
                )
            row[column] = str(int(value))
        else:
            row[column] = _fmt_num(value)


def _op_lag_window(
    rows: list[dict[str, str]], params: dict[str, Any], columns_order: list[str]
) -> list[str]:
    column = _require_column(params.get("column"), columns_order, "lag_window")
    k = params.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise PipelineError(
            f"lag_window: k debe ser un entero >= 1 (recibido {k!r}) — un lag "
            "es SIEMPRE hacia atrás (invariante 6: nunca hacia delante)."
        )
    new_cols = [f"{column}_lag{i}" for i in range(1, k + 1)]
    for c in new_cols:
        if c in columns_order:
            raise PipelineError(f"lag_window: la columna {c!r} ya existe.")
    for idx, row in enumerate(rows):
        for i in range(1, k + 1):
            src_idx = idx - i
            row[f"{column}_lag{i}"] = rows[src_idx][column] if src_idx >= 0 else ""
    return columns_order + new_cols


def _op_shift_target(
    rows: list[dict[str, str]], params: dict[str, Any], columns_order: list[str],
    forward_shifted: set[str],
) -> list[str]:
    column = _require_column(params.get("column"), columns_order, "shift_target")
    horizon = params.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise PipelineError(
            f"shift_target: horizon debe ser un entero >= 1 (recibido {horizon!r}) "
            "— el target es SIEMPRE un valor futuro respecto a la fila."
        )
    as_name = params.get("as") or f"{column}_target_h{horizon}"
    if not isinstance(as_name, str) or not as_name:
        raise PipelineError("shift_target: 'as' debe ser un nombre de columna no vacío.")
    if as_name in columns_order:
        raise PipelineError(f"shift_target: la columna {as_name!r} ya existe.")
    n = len(rows)
    for idx, row in enumerate(rows):
        src_idx = idx + horizon
        row[as_name] = rows[src_idx][column] if src_idx < n else ""
    forward_shifted.add(as_name)
    return columns_order + [as_name]


def _op_drop_columns(
    rows: list[dict[str, str]], params: dict[str, Any], columns_order: list[str]
) -> list[str]:
    columns = params.get("columns")
    if not isinstance(columns, list) or not columns:
        raise PipelineError("drop_columns: columns debe ser una lista no vacía.")
    for c in columns:
        _require_column(c, columns_order, "drop_columns", "columns")
    for row in rows:
        for c in columns:
            row.pop(c, None)
    return [c for c in columns_order if c not in columns]


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
