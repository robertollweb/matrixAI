# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""BIBLIOTECA_PROYECTOS_INTELIGENTES C2 — modelo desde los datos.

Cierra el flujo A (datos-primero, ver contrato 57): del esquema FINAL que
produjo C1 (`analyze_dataset_csv`, corregido a mano por el usuario si
aplica — invariante 8, "el usuario manda sobre la inferencia") + una
columna objetivo elegida → sintetiza el prompt TIPADO canónico de GEN
(decisión 4 del contrato) → lo entrega a `analyze_playground_request` (el
MISMO dispatcher que usa el flujo prompt-primero de siempre) → devuelve
`mxai + training_text + esquema S2`, listo para el flujo de entrenamiento
EXISTENTE. Cero caminos paralelos de generación (invariante 4 del
contrato): los 6 invariantes de GEN aplican tal cual.

Por qué hace falta preparar el CSV (y no solo pasar el crudo tal cual)
------------------------------------------------------------------------
Verificado empíricamente contra el generador real (no asumido):

1. **La columna objetivo SIEMPRE se renombra.** GEN nombra el target
   `predicted_class` (clasificación) / `predicted_value` (regresión) —
   `_output_config` en `dense_generator.py` — sin importar qué nombre se
   escriba en el prompt. La columna real del usuario ("resultado",
   "tiempo"...) nunca coincide con eso por casualidad.
2. **Las etiquetas de clasificación se normalizan a minúsculas
   SIEMPRE.** `resolve_task_and_labels` extrae las etiquetas de
   `ProbabilityMap[...]`/`Label[...]` vía `_identifier()` (minúsculas, solo
   alfanumérico+guion_bajo) — escribir `ProbabilityMap[Lluvia, Nublado,
   Sol]` en el prompt emite `ProbabilityMap[lluvia, nublado, sol]` en el
   `.mxai`. La validación de fila (`TrainingVerifier`) compara el valor
   CRUDO de la columna objetivo contra esas etiquetas con `==` exacta — así
   que si el CSV real trae "Lluvia" (con mayúscula) y no se normaliza
   también el VALOR, la fila se rechaza aunque signifique lo mismo. Se
   normaliza con la MISMA función (`_identifier`) que usa GEN por dentro,
   así que coincide por construcción, nunca por reimplementar la regla.
   (El kwarg estructurado `labels=[...]` SÍ preserva mayúsculas — pero
   `analyze_playground_request` no lo expone en su payload; solo el texto
   del prompt llega, verificado.)
3. **Las categóricas se expanden a one-hot.** El CSV de entrenamiento
   esperado por un modelo con `Categorical[...]` no lleva la columna cruda
   — lleva una columna binaria por valor (`col__valor`, S2-C2), verificado
   con `_csv_template`. Se usa `_build_group_names` (categorical.py), la
   MISMA función que `expand_categoricals` usa para nombrar esas columnas
   en el `.mxai`, así que los nombres coinciden por construcción.
4. **Las booleanas van como 0/1, no "si"/"no".** El CSV de entrenamiento
   trata `Boolean` como un Scalar más (`_csv_template` propone 0.5 de
   ejemplo) — la conversión de tokens humanos ("si"/"no"/"true"/"false") a
   0/1 solo existe en `predict.py` (inferencia), nunca en el CSV de
   entrenamiento.

Los rangos numéricos NO se tocan aquí — igual que el flujo de subida de
HOY, viajan como `field_ranges` y `_normalize_csv_with_ranges` (M5) los
lleva a [0,1] en el boundary de entrenamiento existente.
"""
from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime, timezone
from typing import Any

from matrixai.training.dataset_analysis import (
    _BOOL_FALSE,
    _BOOL_TRUE,
    _is_null,
    analyze_dataset_csv,
)
from matrixai.training.categorical import _build_group_names
from matrixai.training.dense_generator import _identifier

# Tipos de columna que nunca son una FEATURE ni un target válido — igual
# que C1 los excluye de target_candidates, aquí se excluyen del prompt
# sintetizado por completo (nunca aparecen en FEATURES ni en FROM COLUMNS).
_NEVER_FEATURE_TYPES = {"identifier", "unknown"}
# `date` tampoco es una FEATURE utilizable todavía (v1): una fecha cruda no
# es un `Scalar`/`Categorical` — el pipeline de ventanas/desplazamiento que
# la haría utilizable es C3 (aún no construido). Se excluye igual que
# identifier/unknown; el usuario la ve en `temporal_columns` (C1) mientras
# tanto.
_NOT_YET_USABLE_FEATURE_TYPES = _NEVER_FEATURE_TYPES | {"date"}

_CLASSIFICATION_TARGET_TYPES = {"boolean", "categorical"}
_REGRESSION_TARGET_TYPES = {"number", "integer"}


class DatasetProjectError(ValueError):
    """Esquema/target inválido para generar un proyecto — error accionable
    (invariante 7 del contrato 57): nunca un proyecto a medias."""


def generate_project_from_dataset(
    csv_text: str,
    target_column: str,
    *,
    column_type_overrides: dict[str, str] | None = None,
    column_range_overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Genera un proyecto MatrixAI completo A PARTIR de datos reales.

    `column_type_overrides`/`column_range_overrides` son las correcciones
    del usuario sobre el esquema que C1 infirió (invariante 8 — SIEMPRE
    ganan sobre lo inferido). Devuelve un dict con el MISMO shape de campos
    que `analyze_playground_request` (`ok`, `mxai`, `training_text`,
    `field_ranges`, `field_types`, `field_categories`, ...) más:
      - `csv_text`: el CSV PREPARADO (target renombrado/normalizado,
        categóricas expandidas, booleanas a 0/1) — listo para
        `/api/validate-csv`/`/api/train-start`, el flujo existente.
      - `provenance`: procedencia del flujo A (invariante 3 del contrato).

    Lanza `DatasetAnalysisError` si el CSV es ilegible (delegado a C1) o
    `DatasetProjectError` si el target/esquema no permite generar un
    modelo (columna inexistente, tipo no soportado, etiquetas
    ambiguas tras normalizar, target constante...).
    """
    analysis = analyze_dataset_csv(csv_text)
    schema_inferred = analysis["columns"]

    columns: dict[str, dict[str, Any]] = {
        col: dict(info) for col, info in schema_inferred.items()
    }
    for col, new_type in (column_type_overrides or {}).items():
        if col not in columns:
            raise DatasetProjectError(
                f"column_type_overrides referencia la columna {col!r}, que no "
                f"existe en el CSV. Columnas: {analysis['column_order']}."
            )
        columns[col]["type"] = new_type
    for col, rng in (column_range_overrides or {}).items():
        if col not in columns:
            raise DatasetProjectError(
                f"column_range_overrides referencia la columna {col!r}, que no "
                f"existe en el CSV. Columnas: {analysis['column_order']}."
            )
        columns[col]["proposed_range"] = list(rng)

    if target_column not in columns:
        raise DatasetProjectError(
            f"La columna objetivo {target_column!r} no existe en el CSV. "
            f"Columnas: {analysis['column_order']}."
        )
    target_type = columns[target_column]["type"]
    if target_type in _CLASSIFICATION_TARGET_TYPES:
        task = "classification"
    elif target_type in _REGRESSION_TARGET_TYPES:
        task = "regression"
    else:
        raise DatasetProjectError(
            f"La columna objetivo {target_column!r} es de tipo {target_type!r} "
            "— no es un target válido (identificador/fecha/columna vacía no se "
            "pueden predecir; corrige el tipo en column_type_overrides si C1 "
            "se equivocó)."
        )

    feature_columns = [
        col for col in analysis["column_order"]
        if col != target_column and columns[col]["type"] not in _NOT_YET_USABLE_FEATURE_TYPES
    ]
    if not feature_columns:
        raise DatasetProjectError(
            "Ninguna columna es utilizable como feature (todas son el target, "
            "identificadores, fechas o columnas vacías) — no hay nada con lo "
            "que entrenar."
        )

    # Reescanea el CSV crudo para el target (valores REALES, con su case
    # original — C1 no guarda vocabulario para 'boolean' y lo trunca para
    # categóricas de cardinalidad alta) y para las categóricas de alta
    # cardinalidad (vocabulario completo, no la muestra de C1 — ver
    # docstring de C1 sobre `vocabulary_sample`).
    rows = _read_rows(csv_text)
    target_values_raw = _distinct_non_null(rows, target_column)
    if len(target_values_raw) < 2 and task == "classification":
        raise DatasetProjectError(
            f"La columna objetivo {target_column!r} tiene menos de 2 valores "
            f"distintos ({target_values_raw}) — no hay nada que clasificar."
        )

    feature_lines: list[str] = []
    for col in feature_columns:
        info = columns[col]
        col_type = info["type"]
        if col_type == "boolean":
            feature_lines.append(f"  {col}: Boolean")
        elif col_type == "integer":
            lo, hi = _range_for(info, col)
            feature_lines.append(f"  {col}: Integer[{_fmt_num(lo)}, {_fmt_num(hi)}]")
        elif col_type == "number":
            lo, hi = _range_for(info, col)
            feature_lines.append(f"  {col}: Scalar en [{_fmt_num(lo)}, {_fmt_num(hi)}]")
        elif col_type == "categorical":
            values = _distinct_non_null(rows, col)
            if len(values) < 2:
                # Cardinalidad<2 tras corregir el tipo a mano — no aporta
                # señal; se excluye en vez de fallar todo el proyecto.
                continue
            escaped = ", ".join(values)
            feature_lines.append(f"  {col}: Categorical[{escaped}]")
        else:
            raise DatasetProjectError(
                f"Tipo de columna {col_type!r} en {col!r} no soportado como "
                "feature todavía."
            )
    if not feature_lines:
        raise DatasetProjectError(
            "Ninguna columna quedó utilizable como feature tras excluir "
            "categóricas con menos de 2 valores."
        )

    target_labels_normalized: list[str] | None = None
    if task == "classification":
        target_labels_normalized = _normalize_labels(target_values_raw, target_column)
        prompt = (
            "clasificar\nFEATURES:\n" + "\n".join(feature_lines) +
            f"\nSALIDA: {_safe_field_name(target_column)}: ProbabilityMap"
            f"[{', '.join(target_labels_normalized)}]\n"
        )
    else:
        prompt = (
            "predecir\nFEATURES:\n" + "\n".join(feature_lines) +
            f"\nSALIDA: {_safe_field_name(target_column)}\n"
        )

    from matrixai.playground import analyze_playground_request
    res = analyze_playground_request({"mode": "prompt", "prompt": prompt, "use_llm": False})
    if not res.get("ok"):
        raise DatasetProjectError(
            f"El prompt sintetizado desde el esquema no generó un modelo válido: "
            f"{res.get('error') or res}"
        )

    prepared_csv = _prepare_training_csv(
        rows, feature_columns, columns, target_column, task,
    )

    provenance = _build_provenance(
        csv_text=csv_text,
        prepared_csv=prepared_csv,
        schema_inferred=schema_inferred,
        schema_final=columns,
        target_column=target_column,
        task=task,
        prompt=prompt,
        column_type_overrides=column_type_overrides or {},
        column_range_overrides=column_range_overrides or {},
    )

    result = dict(res)
    result["csv_text"] = prepared_csv
    result["provenance"] = provenance
    # field_ranges/field_types/field_categories YA vienen en `res` (extraídos
    # del prompt sintetizado por analyze_playground_request) — no se
    # duplican aquí, se devuelven tal cual llegaron.
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_rows(csv_text: str) -> list[dict[str, str]]:
    from matrixai.training.data import normalize_csv_text
    normalized = normalize_csv_text(csv_text)
    return list(csv.DictReader(io.StringIO(normalized)))


def _distinct_non_null(rows: list[dict[str, str]], col: str) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        value = row.get(col)
        if _is_null(value):
            continue
        v = value.strip()
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


def _range_for(info: dict[str, Any], col: str) -> tuple[float, float]:
    rng = info.get("proposed_range") or info.get("observed_range")
    if rng is None:
        raise DatasetProjectError(f"La columna {col!r} no tiene un rango numérico calculable.")
    return float(rng[0]), float(rng[1])


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _safe_field_name(name: str) -> str:
    """Nombre de campo seguro para el prompt (invariante: el VALOR real de
    esta etiqueta se descarta siempre — GEN emite `predicted_class`/
    `predicted_value` pase lo que pase, verificado — pero el prompt debe
    parsear, así que se sanea igual que un identificador de campo."""
    safe = _identifier(name)
    return safe or "objetivo"


def _normalize_labels(raw_values: list[str], target_column: str) -> list[str]:
    """`_identifier(v)` por cada valor — la MISMA normalización que GEN
    aplica a `ProbabilityMap[...]` por dentro (ver docstring del módulo,
    punto 2). Detecta colisiones (dos valores crudos DISTINTOS que
    normalizan igual, p.ej. "Sí" y "SI") y valores vacíos tras normalizar —
    ambos casos son ambigüedad real, error accionable en vez de un
    entrenamiento que silenciosamente confunde dos clases."""
    normalized: dict[str, str] = {}
    for raw in raw_values:
        norm = _identifier(raw)
        if not norm:
            raise DatasetProjectError(
                f"El valor {raw!r} de la columna objetivo {target_column!r} "
                "queda vacío tras normalizar (solo tenía símbolos/espacios) — "
                "no se puede declarar como etiqueta."
            )
        if norm in normalized and normalized[norm] != raw:
            raise DatasetProjectError(
                f"Los valores {normalized[norm]!r} y {raw!r} de la columna "
                f"objetivo {target_column!r} son la misma etiqueta tras "
                f"normalizar ({norm!r}) — no se pueden distinguir. Unifica "
                "el texto de esas filas antes de generar el modelo."
            )
        normalized[norm] = raw
    return sorted(normalized.keys())


def _prepare_training_csv(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    columns: dict[str, dict[str, Any]],
    target_column: str,
    task: str,
) -> str:
    # Grupos one-hot + el mapa valor_crudo->columna, calculados UNA VEZ (no
    # por fila — recalcular _distinct_non_null dentro del bucle de filas es
    # O(filas²) y, peor, podría ver un vocabulario distinto por fila si el
    # cálculo no fuera puramente determinista).
    onehot_columns: dict[str, dict[str, str]] = {}  # col -> {valor_crudo: columna_onehot}
    header: list[str] = []
    for col in feature_columns:
        if columns[col]["type"] == "categorical":
            values = _distinct_non_null(rows, col)
            if len(values) < 2:
                continue
            names = _build_group_names(col, values)
            onehot_columns[col] = dict(zip(values, names))
            header.extend(names)
        else:
            header.append(col)
    target_header = "predicted_class" if task == "classification" else "predicted_value"
    header.append(target_header)

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header)
    writer.writeheader()
    for row in rows:
        target_raw = row.get(target_column)
        if _is_null(target_raw):
            continue  # sin target no hay fila que entrenar (nunca se inventa uno)
        prepared: dict[str, str] = {}
        for col in feature_columns:
            info = columns[col]
            if info["type"] == "categorical":
                value_to_column = onehot_columns.get(col)
                if value_to_column is None:
                    continue
                for onehot_col in value_to_column.values():
                    prepared[onehot_col] = "0"
                raw = row.get(col)
                raw = raw.strip() if raw is not None else raw
                if raw in value_to_column:
                    prepared[value_to_column[raw]] = "1"
            elif info["type"] == "boolean":
                raw = (row.get(col) or "").strip().lower()
                if raw in _BOOL_TRUE:
                    prepared[col] = "1"
                elif raw in _BOOL_FALSE:
                    prepared[col] = "0"
                else:
                    prepared[col] = row.get(col, "")  # deja que el verificador existente lo rechace
            else:
                prepared[col] = row.get(col, "")
        prepared[target_header] = (
            _identifier(target_raw.strip()) if task == "classification" else target_raw.strip()
        )
        writer.writerow(prepared)
    return out.getvalue()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_provenance(
    *,
    csv_text: str,
    prepared_csv: str,
    schema_inferred: dict[str, Any],
    schema_final: dict[str, Any],
    target_column: str,
    task: str,
    prompt: str,
    column_type_overrides: dict[str, str],
    column_range_overrides: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    from matrixai.export.inference_spec import _matrixai_version

    operations: list[str] = [f"rename_target_column:{target_column}->predicted_{'class' if task == 'classification' else 'value'}"]
    if task == "classification":
        operations.append("normalize_target_labels")
    if any(c["type"] == "categorical" for c in schema_final.values()):
        operations.append("expand_categoricals_onehot")
    if any(c["type"] == "boolean" for c in schema_final.values()):
        operations.append("normalize_boolean_features")

    return {
        "source": "user_upload",
        "raw_csv_sha256": _sha256_text(csv_text),
        "prepared_csv_sha256": _sha256_text(prepared_csv),
        "schema_inferred": schema_inferred,
        "schema_final": schema_final,
        "target_column": target_column,
        "task": task,
        "column_type_overrides": column_type_overrides,
        "column_range_overrides": {k: list(v) for k, v in column_range_overrides.items()},
        "synthesized_prompt": prompt,
        "operations": operations,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrixai_version": _matrixai_version(),
    }
