# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""BIBLIOTECA_PROYECTOS_INTELIGENTES C1 — inferir un esquema desde datos REALES.

`analyze_dataset_csv(csv_text)` es la mitad "datos-primero" del contrato 57: en
vez de generar un modelo desde un prompt y luego un dataset sintético que lo
respete (flujo de hoy), aquí el usuario ya tiene un CSV real (subido a mano, o
descargado por un proveedor del flujo B) y el Studio necesita ADIVINAR su
esquema — tipo por columna, rangos, categorías, columna objetivo, columna
temporal — para poder ofrecerlo como punto de partida editable (C4 reutiliza
el editor S2; invariante 8: el usuario manda sobre la inferencia).

Determinista, stdlib puro, sin red, sin UI — la salida es un dict JSON-
serializable listo para que C2 lo convierta en el prompt tipado de GEN y para
que C4 lo pinte en el editor de esquema.

Diseño deliberado (documentado porque toda heurística de datos reales tiene
casos ambiguos — el usuario siempre puede corregir en el editor, invariante 8):
  - El TIPO de una columna se infiere de sus VALORES, nunca de su nombre (a
    diferencia de `_suggest_field_types`, heurística de nombre que sigue
    existiendo para el editor manual). Orden de comprobación: fecha → boolean
    → numérico (entero/decimal) → identificador (alta unicidad) → categórica
    (todo lo demás). El orden importa: una columna de fechas únicas no debe
    caer en "identificador", y "0"/"1" puros se leen como boolean antes que
    como entero (mismos tokens que `predict_template.py`).
  - "categórica" cubre TANTO baja como alta cardinalidad — la cardinalidad
    viaja en el resultado para que C2 decida one-hot vs embedding
    (`_ONEHOT_MAX`, el mismo umbral que ya usan los generadores), esto no es
    una decisión de tipo.
  - Candidatos a TARGET puntúan 3 señales (posición última columna, nombre
    típico, cardinalidad "clasificable") y se devuelven TODOS los candidatos
    viables (excluidas identificador/fecha) ordenados — la UI de C4 propone,
    el usuario elige.
"""
from __future__ import annotations

import csv
import io
from typing import Any

from matrixai import limits as _limits
from matrixai.training.dense_generator import _ONEHOT_MAX

# Tokens boolean — mismo vocabulario que `matrixai/export/predict_template.py`
# (_TRUE/_FALSE), para que "lo que el usuario ve como booleano en un CSV" sea
# consistente con "lo que predict.py acepta como booleano" en el otro extremo
# del ciclo. Duplicado deliberadamente: predict_template.py viaja standalone
# dentro de cada bundle exportado (cero dependencia de matrixai) y no debe
# importarse desde aquí.
_BOOL_TRUE = {"true", "verdadero", "si", "sí", "yes", "y", "t", "1"}
_BOOL_FALSE = {"false", "falso", "no", "n", "f", "0"}

# Marcadores de nulo habituales en CSVs reales (case-insensitive).
_NULL_TOKENS = {"", "na", "n/a", "null", "nan", "none", "-", "?"}

# Formatos de fecha/hora probados en orden — el primero que casa el 100% de
# los valores no vacíos de la columna gana. ISO primero (inequívoco); luego
# día/mes (convención habitual fuera de EEUU, coherente con el resto del
# producto en español) antes que mes/día. Los formatos CON hora existen
# porque los proveedores reales los emiten así (auditoría C1: Open-Meteo
# hourly devuelve "2024-01-01T00:00" — sin ellos, el timestamp del ejemplo
# canónico del mar caía a "identifier" por unicidad y el modo serie temporal
# nunca se ofrecía).
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
)

# Identificador: unicidad casi total Y suficientes filas para que la señal
# sea significativa (con pocas filas, "todo distinto" es habitual y no dice
# nada — ver test de bordes).
_IDENTIFIER_UNIQUE_RATIO = 0.98
_IDENTIFIER_MIN_ROWS = 10

# Margen del rango PROPUESTO sobre el rango OBSERVADO (10% del span; con
# span 0 — todas las filas igual valor — se usa un margen absoluto mínimo).
_RANGE_MARGIN_FRACTION = 0.1
_RANGE_MARGIN_MIN_ABS = 1.0

# Nombres típicos de columna objetivo (comparación exacta, minúsculas, tras
# strip — es una señal más entre tres, no la única, así que no hace falta
# heurística de substring).
_TARGET_NAME_HINTS = {
    "target", "label", "class", "clase", "resultado", "objetivo", "etiqueta",
    "salida", "output", "result", "outcome", "y", "prediction", "prediccion",
}


class DatasetAnalysisError(ValueError):
    """CSV ilegible o vacío — error accionable (invariante 7 del contrato 57)."""


def analyze_dataset_csv(csv_text: str) -> dict[str, Any]:
    """Analiza un CSV real y propone un esquema — SUGERENCIA, no decisión.

    Devuelve un dict con: `columns` (tipo/nulos/rango u vocabulario u
    unicidad por columna, según tipo), `column_order`, `duplicate_rows`,
    `target_candidates` (ordenados, con tarea sugerida y motivo), y
    `temporal_columns` (columnas tipo fecha, para el modo serie temporal de
    C4). Lanza `DatasetAnalysisError` si el CSV está vacío o es ilegible.
    """
    if not csv_text or not csv_text.strip():
        raise DatasetAnalysisError("El CSV está vacío.")

    # Auditoría de las sugerencias: el límite de tamaño se comprueba sobre el
    # texto CRUDO, antes de pagar la reescritura O(n) de la normalización —
    # un payload gigante debe rechazarse ANTES de costar CPU/memoria (mismo
    # orden que `_normalize_external_csv` en playground.py).
    size = len(csv_text.encode("utf-8"))
    if _limits.exceeds(size, "max_csv_bytes"):
        limit = _limits.get_limit("max_csv_bytes")
        raise DatasetAnalysisError(
            f"El CSV supera el límite de tamaño ({limit // 1_000_000} MB)."
        )

    # Autoauditoría C1 (sugerencias implementadas): BOM UTF-8 de Excel y
    # delimitador ';' (Excel europeo) — mismo helper compartido que usan
    # `_validate_training_csv`/`_run_playground_training`/
    # `_submit_training_job` en playground.py, para que validar/entrenar/
    # analizar vean SIEMPRE el mismo texto normalizado (ver docstring de
    # `normalize_csv_text`).
    from matrixai.training.data import normalize_csv_text
    csv_text = normalize_csv_text(csv_text)

    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise DatasetAnalysisError("El CSV no tiene fila de cabecera.")
        rows = list(reader)
    except csv.Error as exc:
        raise DatasetAnalysisError(f"El CSV es ilegible: {exc}") from exc

    columns = [str(c) for c in fieldnames if c is not None]
    if not columns:
        raise DatasetAnalysisError("El CSV no tiene columnas.")
    # Auditoría C1: una cabecera con nombres duplicados o vacíos no puede
    # convertirse en un VECTOR (los campos del mxai son únicos y con nombre)
    # y además corrompe el análisis en silencio — DictReader se queda con el
    # ÚLTIMO valor de cada nombre repetido, así que la columna duplicada se
    # analizaría a medias y saldría DOS veces en target_candidates. Error
    # accionable aquí, nunca un esquema a medias (invariante 7).
    if any(not c.strip() for c in columns):
        raise DatasetAnalysisError(
            "La cabecera del CSV tiene columnas sin nombre (¿coma de más?). "
            "Pon nombre a todas las columnas."
        )
    duplicated = sorted({c for c in columns if columns.count(c) > 1})
    if duplicated:
        raise DatasetAnalysisError(
            f"La cabecera del CSV repite nombres de columna: {duplicated}. "
            "Renombra las columnas duplicadas."
        )
    if not rows:
        raise DatasetAnalysisError("El CSV no tiene filas de datos.")

    rows_total = len(rows)
    rows_capped_warning: str | None = None
    max_rows = _limits.get_limit("max_rows")
    if max_rows is not None and rows_total > max_rows:
        rows_capped_warning = (
            f"El CSV tiene {rows_total} filas; el análisis usa solo las "
            f"primeras {max_rows} (perfil de límites actual)."
        )
        rows = rows[:max_rows]
    rows_analyzed = len(rows)

    duplicate_rows = _count_duplicate_rows(rows, columns)

    column_infos: dict[str, dict[str, Any]] = {}
    for col in columns:
        raw_values = [row.get(col) for row in rows]
        column_infos[col] = _analyze_column(raw_values, rows_analyzed)

    target_candidates = _rank_target_candidates(columns, column_infos)
    temporal_columns = [c for c in columns if column_infos[c]["type"] == "date"]

    result: dict[str, Any] = {
        "ok": True,
        "rows_total": rows_total,
        "rows_analyzed": rows_analyzed,
        "duplicate_rows": duplicate_rows,
        "column_order": columns,
        "columns": column_infos,
        "target_candidates": target_candidates,
        "temporal_columns": temporal_columns,
    }
    if rows_capped_warning:
        result["rows_capped_warning"] = rows_capped_warning
    return result


# ---------------------------------------------------------------------------
# Por columna
# ---------------------------------------------------------------------------

def _is_null(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in _NULL_TOKENS


def _analyze_column(raw_values: list[str | None], rows_analyzed: int) -> dict[str, Any]:
    non_null = [v.strip() for v in raw_values if not _is_null(v)]
    null_count = rows_analyzed - len(non_null)
    info: dict[str, Any] = {
        "null_count": null_count,
        "null_ratio": round(null_count / rows_analyzed, 4) if rows_analyzed else 0.0,
    }

    if not non_null:
        info["type"] = "unknown"
        info["cardinality"] = 0
        return info

    distinct = sorted(set(non_null))
    cardinality = len(distinct)
    unique_ratio = cardinality / len(non_null)

    date_format = _detect_date_format(non_null)
    if date_format is not None:
        info["type"] = "date"
        info["date_format"] = date_format
        info["cardinality"] = cardinality
        return info

    if _is_boolean_column(non_null):
        info["type"] = "boolean"
        info["cardinality"] = cardinality
        return info

    is_identifier_candidate = (
        len(non_null) >= _IDENTIFIER_MIN_ROWS
        and unique_ratio >= _IDENTIFIER_UNIQUE_RATIO
    )

    # Autoauditoría C1 (sugerencia implementada): un código con cero(s) a la
    # izquierda ("08001", "007") NUNCA es un valor de dominio numérico — es
    # un código (postal, de empleado…) donde el cero es información, no
    # relleno. `float`/`int` lo aceptarían igual (`int("007") == 7`) y
    # normalizarlo perdería el cero para siempre. Si ALGÚN valor de la
    # columna tiene esta forma, la columna entera se trata como no-numérica
    # (identificador si es casi-única, si no categórica) — una columna
    # inconsistente (algunas filas con cero, otras sin él) es señal de que
    # el cero se perdió en algún punto de la exportación, no de que sea
    # opcional; el usuario puede corregir el tipo en el editor
    # (invariante 8).
    numeric_kind = (
        None if any(_has_significant_leading_zero(v) for v in non_null)
        else _numeric_kind(non_null)
    )
    # Un entero casi-todo-distinto (1,2,3,...,N — el clásico id secuencial)
    # es identificador, no un valor de dominio — pero un DECIMAL nunca lo es
    # por esta vía sola (una medida continua es normal que salga casi única
    # incluso sin ser un id; ver test_numeric_looking_strings_are_numeric).
    if numeric_kind == "integer" and is_identifier_candidate:
        info["type"] = "identifier"
        info["cardinality"] = cardinality
        info["unique_ratio"] = round(unique_ratio, 4)
        return info
    if numeric_kind is not None:
        values = [float(v) for v in non_null]
        lo, hi = min(values), max(values)
        info["type"] = numeric_kind
        info["cardinality"] = cardinality
        info["observed_range"] = _round_range([lo, hi], numeric_kind)
        info["proposed_range"] = _round_range(_propose_margin(lo, hi), numeric_kind)
        return info

    if is_identifier_candidate:
        info["type"] = "identifier"
        info["cardinality"] = cardinality
        info["unique_ratio"] = round(unique_ratio, 4)
        return info

    info["type"] = "categorical"
    info["cardinality"] = cardinality
    # Auditoría C1 (alineación con el contrato: "vocabulario si categórica
    # (cardinalidad BAJA)"): el vocabulario completo solo viaja en territorio
    # one-hot (≤ _ONEHOT_MAX, el umbral existente) — una categórica de 600
    # ciudades metía 600 entradas en la respuesta del análisis sin tope. Por
    # encima va una MUESTRA (mismo tamaño que el umbral, sin inventar otro) y
    # el flag de truncado; C2 puede re-derivar el vocabulario completo del
    # propio CSV cuando el camino embedding lo necesite.
    if cardinality <= _ONEHOT_MAX:
        info["vocabulary"] = distinct
    else:
        info["vocabulary_sample"] = distinct[:_ONEHOT_MAX]
        info["vocabulary_truncated"] = True
    return info


def _detect_date_format(values: list[str]) -> str | None:
    from datetime import datetime

    for fmt in _DATE_FORMATS:
        try:
            for v in values:
                datetime.strptime(v, fmt)
            return fmt
        except ValueError:
            continue
    return None


def _is_boolean_column(values: list[str]) -> bool:
    tokens = {v.strip().lower() for v in values}
    return tokens.issubset(_BOOL_TRUE | _BOOL_FALSE) and len(tokens) <= 2


def _has_significant_leading_zero(value: str) -> bool:
    """"08001"/"007" → True (código con cero significativo); "0"/"0.5"/"8001"
    → False. Deliberadamente estricto (solo dígitos tras el 0 inicial, sin
    signo ni punto decimal) para no atrapar "0.5" ni "-08" como falsos
    positivos de una columna que SÍ es numérica de verdad."""
    return len(value) > 1 and value[0] == "0" and value[1:].isdigit()


def _numeric_kind(values: list[str]) -> str | None:
    """`"integer"` si TODOS los valores parsean como entero, `"number"` si
    parsean como float pero no todos como entero, `None` si alguno no es
    numérico (incluye NaN/Infinity textual — no son un rango físico usable)."""
    all_int = True
    for v in values:
        try:
            f = float(v)
        except ValueError:
            return None
        if f != f or f in (float("inf"), float("-inf")):  # NaN / Infinity
            return None
        if not f.is_integer():
            all_int = False
    return "integer" if all_int else "number"


def _propose_margin(lo: float, hi: float) -> list[float]:
    span = hi - lo
    margin = span * _RANGE_MARGIN_FRACTION if span > 0 else _RANGE_MARGIN_MIN_ABS
    return [lo - margin, hi + margin]


def _round_range(range_pair: list[float], numeric_kind: str) -> list[float | int]:
    if numeric_kind == "integer":
        import math
        return [int(math.floor(range_pair[0])), int(math.ceil(range_pair[1]))]
    return [round(range_pair[0], 4), round(range_pair[1], 4)]


# ---------------------------------------------------------------------------
# Duplicados
# ---------------------------------------------------------------------------

def _count_duplicate_rows(rows: list[dict[str, Any]], columns: list[str]) -> int:
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    for row in rows:
        key = tuple(row.get(c) for c in columns)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


# ---------------------------------------------------------------------------
# Candidatos a target
# ---------------------------------------------------------------------------

def _rank_target_candidates(
    columns: list[str], column_infos: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    last_col = columns[-1] if columns else None

    for col in columns:
        info = column_infos[col]
        col_type = info["type"]
        if col_type in ("identifier", "unknown"):
            continue  # nunca tiene sentido predecir un id o una columna vacía
        # Auditoría C1: una columna CONSTANTE (cardinalidad < 2) tampoco es
        # un target — "predecir" un valor que nunca cambia no entrena nada,
        # y antes una constante llamada "y" llegaba a proponerse la PRIMERA
        # (última columna + nombre típico) con el motivo "valores numéricos
        # continuos"… siendo una constante.
        if info.get("cardinality", 0) < 2:
            continue

        score = 0.0
        reasons: list[str] = []
        # Contrato 58 C3 — códigos ESTRUCTURADOS y versionables (un `code` +
        # parámetros propios cuando aplica) para que el SPA los traduzca
        # es/en sin parsear el texto en español de `reasons` (que se
        # conserva tal cual, retrocompatible con quien ya lo consumía).
        reason_codes: list[dict[str, Any]] = []

        if col == last_col:
            score += 1.0
            reasons.append("es la última columna del CSV")
            reason_codes.append({"code": "last_column"})

        if col.strip().lower() in _TARGET_NAME_HINTS:
            score += 2.0
            reasons.append("nombre típico de columna objetivo")
            reason_codes.append({"code": "typical_target_name"})

        cardinality = info.get("cardinality", 0)
        few_categories = 2 <= cardinality <= _ONEHOT_MAX
        if col_type == "date":
            task = None  # una fecha no es un target razonable, pero no se excluye
        elif col_type in ("boolean", "categorical"):
            # Categórica/boolean SIEMPRE sugiere clasificación (es lo que es,
            # con independencia de cuántas clases tenga) — el bono de
            # puntuación y el motivo "pocas categorías" solo aplican cuando
            # de verdad son pocas (mismo umbral que decide one-hot en C2).
            task = "classification"
            if few_categories:
                score += 1.0
                reasons.append(f"pocas categorías distintas ({cardinality})")
                reason_codes.append({"code": "low_cardinality", "cardinality": cardinality})
        elif col_type in ("integer", "number") and few_categories:
            task = "classification"
            score += 1.0
            reasons.append(f"pocas categorías distintas ({cardinality})")
            reason_codes.append({"code": "low_cardinality", "cardinality": cardinality})
        else:
            task = "regression"
            score += 0.5
            reasons.append("valores numéricos continuos")
            reason_codes.append({"code": "continuous_numeric"})

        if task is None:
            continue

        candidates.append({
            "column": col,
            "task": task,
            "score": round(score, 2),
            "reasons": reasons,
            "reason_codes": reason_codes,
        })

    candidates.sort(key=lambda c: (-c["score"], columns.index(c["column"])))
    return candidates
