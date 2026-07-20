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
3. **Las categóricas de cardinalidad baja se expanden a one-hot.** El CSV
   de entrenamiento esperado por un modelo con `Categorical[...]` que GEN
   resolvió como one-hot no lleva la columna cruda — lleva una columna
   binaria por valor (`col__valor`, S2-C2), verificado con `_csv_template`.
   Se usa `_build_group_names` (categorical.py), la MISMA función que
   `expand_categoricals` usa para nombrar esas columnas en el `.mxai`, así
   que los nombres coinciden por construcción.
4. **Las booleanas van como 0/1, no "si"/"no".** El CSV de entrenamiento
   trata `Boolean` como un Scalar más (`_csv_template` propone 0.5 de
   ejemplo) — la conversión de tokens humanos ("si"/"no"/"true"/"false") a
   0/1 solo existe en `predict.py` (inferencia), nunca en el CSV de
   entrenamiento.
5. **El nombre de cada FEATURE debe coincidir con lo que GEN sanea.**
   `parse_field_specs` acepta cualquier texto como nombre de campo en el
   prompt (acentos/espacios/símbolos) pero lo sanea con
   `_sanitize_name` — idéntica normalización a `_identifier` (verificado
   comparando ambas funciones) — antes de escribirlo en el VECTOR. Si el
   CSV preparado usara el nombre crudo de la columna ("customer age") y el
   VECTOR generado usa el saneado ("customer_age"), `/api/validate-csv`
   rechaza el proyecto siempre que la cabecera tenga espacios, guiones,
   acentos o símbolos (auditoría C2 [ALTA], reproducido). Aquí se sanea
   con `_identifier` (la misma regla) ANTES de escribir el prompt y el CSV,
   con detección de colisión si dos columnas crudas distintas saneasen
   igual.
6. **Una categórica de alta cardinalidad NO se expande a one-hot.** GEN
   enruta cualquier `Categorical[...]` con más de `_ONEHOT_MAX` valores al
   generador composite con EMBEDDING nativo — la columna sigue siendo UNA
   sola en el VECTOR (no N columnas one-hot) y el CSV de entrenamiento
   espera el ÍNDICE del valor en el vocabulario (verificado empíricamente:
   `_validate_training_csv` exige "campo X debe ser numérico" para la
   columna fuente de un EMBEDDING). Expandir siempre a one-hot sin mirar
   la cardinalidad (auditoría C2 [ALTA]) produce un CSV con columnas que el
   modelo generado ni siquiera declara.
7. **Una etiqueta de clasificación que empieza por dígito no queda
   vacía.** `_identifier` rechaza cualquier token que empiece por número
   (un identificador de Python tampoco puede) — un target booleano
   CANÓNICO 0/1 (el que C1 reconoce a propósito) o una etiqueta como "24h"
   normalizaban a cadena vacía y `generate_project_from_dataset` fallaba
   siempre con esos datasets (auditoría C2 [ALTA], reproducido). Se
   reintenta con el prefijo `class_` (mismo criterio que usaría cualquier
   generador de identificadores) — solo un valor SIN ningún carácter
   alfanumérico ("###") sigue siendo un error real.
8. **Un valor categórico con ',', ']' o salto de línea rompería el
   corchete del prompt.** `Categorical[...]` se parsea partiendo por comas
   sin escape (`args.split(",")` en `prompt_field_specs.py`) — un valor
   real "red,blue" se leería como DOS categorías distintas mientras el CSV
   preparado seguiría tratándolo como un único valor, produciendo un
   desalineamiento silencioso entre modelo y CSV (auditoría C2 [ALTA],
   reproducido). Se detecta ANTES de sintetizar el prompt y se rechaza con
   un error accionable (invariante 7) — GEN no tiene mecanismo de escape
   para este vocabulario, así que no hay forma segura de "arreglarlo" en
   silencio.

Los rangos numéricos NO se tocan aquí — igual que el flujo de subida de
HOY, viajan como `field_ranges` y `_normalize_csv_with_ranges` (M5) los
lleva a [0,1] en el boundary de entrenamiento existente.

El CSV preparado se valida CONTRA el modelo generado (`_validate_training_
csv`, el mismo flujo `/api/validate-csv` de siempre) antes de responder
`ok` (auditoría C2 [MEDIA] — contrato §C2: "el CSV real queda... validado
contra el modelo generado"). Nunca se devuelve un proyecto "aparentemente
correcto" que falla después, en silencio, al entrenar.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from matrixai.training.dataset_analysis import (
    _BOOL_FALSE,
    _BOOL_TRUE,
    _has_significant_leading_zero,
    _is_null,
    _numeric_kind,
    _propose_margin,
    _round_range,
    analyze_dataset_csv,
)
from matrixai.training.categorical import _build_group_names
from matrixai.training.dense_generator import _identifier, _ONEHOT_MAX
from matrixai.training.user_intent import UserIntentError, normalize_user_intent
from matrixai.training.intent_llm import (
    IntentArchitectureError,
    build_llm_context,
    propose_intent_architecture,
    proposal_sha256,
)

# Tipos de columna que nunca son una FEATURE ni un target válido — igual
# que C1 los excluye de target_candidates, aquí se excluyen del prompt
# sintetizado por completo (nunca aparecen en FEATURES ni en FROM COLUMNS).
_NEVER_FEATURE_TYPES = {"identifier", "unknown"}
# `date` tampoco es una FEATURE utilizable directamente: una fecha cruda no
# es un `Scalar`/`Categorical`. El pipeline de ventanas/desplazamiento que la
# hace utilizable es C3 (`dataset_pipeline.py`) — `sort_temporal` la consume
# para ordenar pero la columna cruda sigue sin ser feature; lo que SÍ se
# vuelve feature son las columnas `_lag*` que produce `lag_window` sobre
# OTRAS columnas (ver `generate_temporal_project_from_dataset`, C4). Se
# excluye igual que identifier/unknown; el usuario la ve en
# `temporal_columns` (C1) mientras tanto.
_NOT_YET_USABLE_FEATURE_TYPES = _NEVER_FEATURE_TYPES | {"date"}

_CLASSIFICATION_TARGET_TYPES = {"boolean", "categorical"}
_REGRESSION_TARGET_TYPES = {"number", "integer"}

# Auditoría C2 [ALTA]: caracteres que romperían el parseo de
# `Categorical[v1, v2, ...]` si aparecieran DENTRO de un valor — ',' es el
# separador (sin escape posible, verificado en prompt_field_specs.py),
# ']' cierra el corchete, '\n'/'\r' rompen el límite de línea del parser.
_UNSAFE_CATEGORY_CHARS = (",", "]", "\n", "\r")


class DatasetProjectError(ValueError):
    """Esquema/target inválido para generar un proyecto — error accionable
    (invariante 7 del contrato 57): nunca un proyecto a medias."""


@dataclass
class _PreparedCSV:
    text: str
    rows_dropped_null_target: int
    operations: list[str] = field(default_factory=list)


def generate_project_from_dataset(
    csv_text: str,
    target_column: str,
    *,
    column_type_overrides: dict[str, str] | None = None,
    column_range_overrides: dict[str, tuple[float, float]] | None = None,
    column_category_overrides: dict[str, list[str]] | None = None,
    user_intent: str | None = None,
    use_intent_llm: bool = False,
) -> dict[str, Any]:
    """Genera un proyecto MatrixAI completo A PARTIR de datos reales.

    Los overrides de tipo/rango/vocabulario son las correcciones
    del usuario sobre el esquema que C1 infirió (invariante 8 — SIEMPRE
    ganan sobre lo inferido). Devuelve un dict con el MISMO shape de campos
    que `analyze_playground_request` (`ok`, `mxai`, `training_text`,
    `field_ranges`, `field_types`, `field_categories`, ...) más:
      - `csv_text`: el CSV PREPARADO (target renombrado/normalizado,
        categóricas expandidas a one-hot o indexadas para embedding,
        booleanas a 0/1) — YA VALIDADO contra el modelo generado, listo
        para `/api/train-start`, el flujo existente.
      - `provenance`: procedencia del flujo A (invariante 3 del contrato),
        incluida `provenance["user_intent"]` (Contrato 58 C4) — la
        intención LOCAL normalizada (`user_intent.py`), que NUNCA entra al
        prompt tipado/generador (ver docstring de ese módulo). `None` si no
        se declaró intención o quedó vacía tras normalizar.
      - `provenance["intent_llm"]` (Contrato 58 C5) — bloque de auditoría de
        la interpretación LLM opt-in (`intent_llm.py`): `None` si no hay
        intención; si la hay, `{requested, used, provider, model,
        proposal_sha256, sanitizer_result, fallback}` — `requested=used=
        false` si `use_intent_llm=False` (el caso por defecto). El LLM SOLO
        puede proponer la forma de la red (tamaños de capa); nunca toca
        features/tipos/rangos/categorías/target/pipeline.

    Lanza `DatasetAnalysisError` si el CSV es ilegible (delegado a C1),
    `DatasetProjectError` si el target/esquema no permite generar un
    modelo (columna inexistente, tipo no soportado, etiquetas/nombres de
    columna ambiguos tras normalizar, target constante, valor categórico
    que rompería el prompt...), si `user_intent` no es válido tras
    normalizar (envuelve `UserIntentError`), si `use_intent_llm=True` sin
    intención declarada, o si la llamada LLM falla (envuelve
    `IntentArchitectureError` — el atributo `.retryable` de la excepción
    resultante distingue "sin LLM configurado" de un fallo transitorio,
    para que el caller pueda ofrecer reintentar). Todos comparten
    `DatasetProjectError` como tipo — un solo tipo que el caller tiene que
    capturar.
    """
    try:
        normalized_intent = normalize_user_intent(user_intent)
    except UserIntentError as exc:
        raise DatasetProjectError(str(exc)) from exc
    if use_intent_llm and normalized_intent is None:
        raise DatasetProjectError(
            "use_intent_llm=true requiere una intención declarada (user_intent está vacío)."
        )

    analysis = analyze_dataset_csv(csv_text)
    schema_inferred = analysis["columns"]
    # CONTRATO 59 C2: se necesitan los valores CRUDOS antes de aplicar los
    # overrides (para recalcular rango si el usuario corrige el tipo a
    # number/integer, hallazgo 3) — antes se leía más abajo, solo para
    # target/features; movido aquí, un único `_read_rows`, reutilizado en
    # todo el resto de la función (nunca se reasigna, es de solo lectura).
    rows = _read_rows(csv_text)

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
        # CONTRATO 59 C2 [hallazgo 3]: antes, corregir el tipo a number/
        # integer sobre una columna sin rango calculado (identifier/
        # unknown/categórica no lo calculan) dejaba `_range_for` sin nada
        # que usar más adelante ("no tiene un rango numérico calculable") —
        # el usuario tenía que ADEMÁS adivinar el rango a mano en
        # column_range_overrides. Se recalcula aquí desde los valores
        # crudos, mismo cálculo que C1 aplicaría si hubiera visto la
        # columna así desde el principio; `column_range_overrides` (más
        # abajo) sigue pudiendo sobreescribirlo si el usuario lo declara
        # explícitamente — invariante 8, el usuario manda.
        if new_type in ("number", "integer") and not (
            columns[col].get("proposed_range") or columns[col].get("observed_range")
        ):
            recomputed = _numeric_range_from_raw_values([row.get(col) for row in rows])
            if recomputed is not None:
                _, (rng_lo, rng_hi) = recomputed
                columns[col]["observed_range"] = [rng_lo, rng_hi]
                columns[col]["proposed_range"] = [rng_lo, rng_hi]
    for col, rng in (column_range_overrides or {}).items():
        if col not in columns:
            raise DatasetProjectError(
                f"column_range_overrides referencia la columna {col!r}, que no "
                f"existe en el CSV. Columnas: {analysis['column_order']}."
            )
        # Auditoría C2 [MEDIA, reauditoría]: hasta aquí solo se validaba la
        # FORMA del override ([min, max] numérico, en el endpoint Studio) —
        # un rango invertido ([10, 0]) o no finito (NaN/inf) pasaba de largo
        # y GEN lo descartaba por su cuenta más adelante SIN avisar (un
        # rango inválido se degrada a "sin rango declarado", verificado),
        # así que la columna se quedaba sin field_ranges pese a `ok: true`.
        # Se valida aquí, en el núcleo — protege a CUALQUIER caller, no solo
        # al endpoint Studio.
        lo, hi = float(rng[0]), float(rng[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise DatasetProjectError(
                f"column_range_overrides[{col!r}] = {list(rng)!r} no es un "
                "rango finito — usa valores numéricos reales (nada de "
                "NaN/infinito)."
            )
        if lo >= hi:
            raise DatasetProjectError(
                f"column_range_overrides[{col!r}] = {list(rng)!r} tiene el "
                "mínimo mayor o igual que el máximo — corrige el rango."
            )
        columns[col]["proposed_range"] = [lo, hi]

    if target_column not in columns:
        raise DatasetProjectError(
            f"La columna objetivo {target_column!r} no existe en el CSV. "
            f"Columnas: {analysis['column_order']}."
        )
    target_type = columns[target_column]["type"]
    target_range: tuple[float, float] | None = None
    if target_type in _CLASSIFICATION_TARGET_TYPES:
        task = "classification"
    elif target_type in _REGRESSION_TARGET_TYPES:
        task = "regression"
        # CONTRATO 59 C1: el target de regresión se entrena normalizado a
        # [0,1] con el MISMO mecanismo que ya usan las features (rango
        # observado + margen) — sin esto, un target en escala de dominio
        # (p.ej. 273-372 Kelvin) hace explotar el MSE con los defaults de
        # entrenamiento y la red colapsa a predecir la media (ver
        # 59_REGRESION_QUE_APRENDE_CONTRACT.md, "Base verificada", punto 1).
        target_range = _range_for(columns[target_column], target_column)
    else:
        raise DatasetProjectError(
            f"La columna objetivo {target_column!r} es de tipo {target_type!r} "
            "— no es un target válido (identificador/fecha/columna vacía no se "
            "pueden predecir; corrige el tipo en column_type_overrides si C1 "
            "se equivocó)."
        )
    # GEN nombra el target SIEMPRE así, sea cual sea la columna real (ver
    # punto 1 del docstring) — se calcula aquí, temprano, porque también
    # hace falta para detectar una FEATURE que colisione con ese nombre
    # reservado (ver `_normalize_feature_names`).
    target_header = "predicted_class" if task == "classification" else "predicted_value"

    feature_columns = [
        col for col in analysis["column_order"]
        if col != target_column and columns[col]["type"] not in _NOT_YET_USABLE_FEATURE_TYPES
    ]
    # CONTRATO 59 C2 (hallazgo de auditoría): declarada FUERA del `if` de
    # abajo para que exista (vacía) también cuando la reconsideración ni se
    # dispara — se pasa siempre a `_build_provenance` más abajo.
    reconsidered_columns: list[str] = []
    if not feature_columns:
        # CONTRATO 59 C2 [decisión C]: antes de abortar, reconsiderar los
        # identificadores NUMÉRICOS (un entero casi-único como
        # "centigrados" 0..99 es un id secuencial para la heurística de C1,
        # pero SÍ es una medida de dominio real) — nunca se reconsidera un
        # identificador de texto (UUID-like: `_numeric_range_from_raw_
        # values` devuelve `None` para esos, se deja excluido) ni una fecha/
        # columna vacía. Si hay OTRAS features reales disponibles,
        # `feature_columns` ya no está vacío y este bloque ni se ejecuta —
        # un identificador con features reales al lado se sigue excluyendo
        # como siempre (test de cierre del corte).
        for col in analysis["column_order"]:
            if col == target_column or columns[col]["type"] != "identifier":
                continue
            recomputed = _numeric_range_from_raw_values([row.get(col) for row in rows])
            if recomputed is None:
                continue
            numeric_kind, (rng_lo, rng_hi) = recomputed
            columns[col]["type"] = numeric_kind
            columns[col]["observed_range"] = [rng_lo, rng_hi]
            columns[col]["proposed_range"] = [rng_lo, rng_hi]
            reconsidered_columns.append(col)
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

    # Auditoría C2 [ALTA] (ver punto 5 del docstring): nombres de campo
    # saneados por adelantado y colisión detectada como error accionable —
    # incluida la colisión con el nombre reservado del target (auditoría
    # C2 [MEDIA, residual]).
    feature_safe_names = _normalize_feature_names(feature_columns, target_header)

    # `rows` (leído al principio de la función, CONTRATO 59 C2) tiene los
    # valores REALES, con su case original — C1 no guarda vocabulario para
    # 'boolean' y lo trunca para categóricas de cardinalidad alta; aquí se
    # usa para el target y para las categóricas de alta cardinalidad
    # (vocabulario completo, no la muestra de C1 — ver docstring de C1
    # sobre `vocabulary_sample`).
    target_values_raw = _distinct_non_null(rows, target_column)
    category_vocabularies: dict[str, list[str]] = {}
    for col, raw_values in (column_category_overrides or {}).items():
        if col not in columns:
            raise DatasetProjectError(
                f"column_category_overrides referencia la columna {col!r}, que no "
                f"existe en el CSV. Columnas: {analysis['column_order']}."
            )
        if columns[col]["type"] != "categorical":
            raise DatasetProjectError(
                f"column_category_overrides[{col!r}] solo se puede aplicar a "
                "una columna cuyo tipo final sea 'categorical'."
            )
        if not isinstance(raw_values, list):
            raise DatasetProjectError(
                f"column_category_overrides[{col!r}] debe ser una lista de valores."
            )
        if not all(isinstance(value, str) for value in raw_values):
            raise DatasetProjectError(
                f"column_category_overrides[{col!r}] solo admite valores de texto."
            )
        values = [value.strip() for value in raw_values]
        if len(values) < 2 or any(not value for value in values):
            raise DatasetProjectError(
                f"column_category_overrides[{col!r}] debe contener al menos "
                "2 valores no vacíos."
            )
        if len(set(values)) != len(values):
            raise DatasetProjectError(
                f"column_category_overrides[{col!r}] contiene valores duplicados."
            )
        _check_categorical_values_safe(values, col)
        observed = _distinct_non_null(rows, col)
        missing = [value for value in observed if value not in values]
        if missing:
            raise DatasetProjectError(
                f"column_category_overrides[{col!r}] omite valores presentes en "
                f"el CSV: {missing}. Añádelos al vocabulario o corrige los datos."
            )
        category_vocabularies[col] = values

    effective_target_values = category_vocabularies.get(target_column, target_values_raw)
    if len(effective_target_values) < 2 and task == "classification":
        raise DatasetProjectError(
            f"La columna objetivo {target_column!r} tiene menos de 2 valores "
            f"distintos ({effective_target_values}) — no hay nada que clasificar."
        )

    feature_lines: list[str] = []
    for col in feature_columns:
        info = columns[col]
        col_type = info["type"]
        safe_name = feature_safe_names[col]
        if col_type == "boolean":
            feature_lines.append(f"  {safe_name}: Boolean")
        elif col_type == "integer":
            lo, hi = _range_for(info, col)
            feature_lines.append(f"  {safe_name}: Integer[{_fmt_num(lo)}, {_fmt_num(hi)}]")
        elif col_type == "number":
            lo, hi = _range_for(info, col)
            feature_lines.append(f"  {safe_name}: Scalar en [{_fmt_num(lo)}, {_fmt_num(hi)}]")
        elif col_type == "categorical":
            values = category_vocabularies.get(col) or _distinct_non_null(rows, col)
            if len(values) < 2:
                # Cardinalidad<2 tras corregir el tipo a mano — no aporta
                # señal; se excluye en vez de fallar todo el proyecto.
                continue
            _check_categorical_values_safe(values, col)
            feature_lines.append(f"  {safe_name}: Categorical[{', '.join(values)}]")
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
    target_label_map: dict[str, str] | None = None
    if task == "classification":
        target_labels_normalized, target_label_map = _normalize_labels(
            effective_target_values, target_column
        )
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

    # Contrato 58 C5 — interpretación LLM OPT-IN de la intención (ver
    # intent_llm.py). Canal COMPLETAMENTE separado del prompt tipado de
    # arriba: el LLM recibe `llm_context` (esquema YA decidido + intención),
    # nunca el CSV/filas, y solo puede proponer la FORMA de la red — se
    # enhebra a `analyze_playground_request` como `architecture_hints`, un
    # payload NUEVO y aislado del mecanismo `use_llm=True` de siempre (ese
    # otro re-deriva FIELDS/LABELS del prompt — prohibido aquí, invariante
    # "el esquema no cambia").
    architecture_hints: dict[str, Any] = {}
    intent_llm: dict[str, Any] | None = None
    if normalized_intent is not None:
        if use_intent_llm:
            llm_features = [
                {
                    "name": col,
                    "type": columns[col]["type"],
                    "range": (
                        list(_range_for(columns[col], col))
                        if columns[col]["type"] in ("number", "integer") else None
                    ),
                    "categories": (
                        category_vocabularies.get(col) or _distinct_non_null(rows, col)
                        if columns[col]["type"] == "categorical" else None
                    ),
                }
                for col in feature_columns
            ]
            llm_context = build_llm_context(
                features=llm_features, task=task, target_column=target_column,
                user_intent=normalized_intent,
            )
            try:
                proposal = propose_intent_architecture(llm_context)
            except IntentArchitectureError as exc:
                err = DatasetProjectError(str(exc))
                err.retryable = exc.retryable  # type: ignore[attr-defined]
                raise err from exc
            architecture_hints["hidden_layers"] = proposal.hidden_layers
            intent_llm = {
                "requested": True,
                "used": True,
                "provider": proposal.provider,
                "model": proposal.model,
                "proposal_sha256": proposal_sha256(proposal.raw_text),
                "sanitizer_result": "adjusted" if proposal.sanitizer_adjusted else "accepted",
                "fallback": None,
            }
        else:
            intent_llm = {
                "requested": False, "used": False, "provider": None, "model": None,
                "proposal_sha256": None, "sanitizer_result": None, "fallback": None,
            }

    from matrixai.playground import analyze_playground_request, _validate_training_csv
    res = analyze_playground_request({
        "mode": "prompt", "prompt": prompt, "use_llm": False,
        **({"architecture_hints": architecture_hints} if architecture_hints else {}),
    })
    if not res.get("ok"):
        # Auditoría C2 [MEDIA, reauditoría]: `res` es el dict COMPLETO de
        # `analyze_playground_request` (mxai + AST + python compilado +
        # checks...) — volcarlo entero como mensaje de error producía un
        # DatasetProjectError de decenas de miles de caracteres, nada
        # accionable. Se extrae el motivo real de `checks` (cada uno trae su
        # propia lista de errores) y, si no hay ninguno, un mensaje corto en
        # vez del dict crudo.
        reason = res.get("error") or "; ".join(
            err for check in (res.get("checks") or []) for err in (check.get("errors") or [])
        ) or "el generador rechazó el prompt sintetizado sin detallar el motivo"
        raise DatasetProjectError(
            f"El prompt sintetizado desde el esquema no generó un modelo válido: {reason}"
        )

    # Auditoría C5 [ALTA]: `intent_llm["used"]` se marcaba `True` en cuanto
    # el LLM devolvía una propuesta interpretable, ANTES de saber si el
    # enrutamiento de `analyze_playground_request` iba a aplicarla de
    # verdad. `hidden_layers` (el único hint que viaja por este canal) solo
    # lo consume la rama DENSA — composite/transformer lo descartan al
    # filtrar sus kwargs (ver `playground.py`, `comp_kwargs`/`trans_kwargs`)
    # — así que un dataset con una categórica de alta cardinalidad (que
    # fuerza `want_composite`) generaba un modelo sin ninguno de los
    # tamaños propuestos, mientras la procedencia seguía afirmando
    # `used=true` y la SPA mostraba "interpretada por IA, afectó a la forma
    # de la red": procedencia falsa. `res["supervision_source"]` refleja el
    # generador REAL que produjo el `.mxai` — se corrige `used` a `False`
    # (con un `fallback` explicando el motivo) si la propuesta no llegó a
    # la única rama que la aplica.
    if (
        intent_llm is not None
        and architecture_hints.get("hidden_layers")
        and res.get("supervision_source") != "dense_generator"
    ):
        intent_llm["used"] = False
        intent_llm["fallback"] = (
            "El dataset requirió un generador distinto al denso (p.ej. una "
            "categórica de alta cardinalidad enruta a composite/embedding, "
            "o un campo Text al transformer); esa ruta no admite la forma "
            "de red propuesta por el LLM, así que se ignoró y se usó la "
            "arquitectura determinista de esa ruta."
        )

    prepared = _prepare_training_csv(
        rows, feature_columns, columns, feature_safe_names, target_column, task,
        target_label_map, target_header, category_vocabularies,
    )
    prepared_csv = prepared.text

    # Auditoría C2 [MEDIA]: el contrato exige validar el CSV preparado
    # CONTRA el modelo generado antes de responder — mismo flujo
    # `/api/validate-csv` de siempre. Si alguna de las transformaciones de
    # arriba tuviera un hueco no cazado por sus propios tests, esto lo
    # convierte en un error accionable AQUÍ, nunca en un proyecto
    # "aparentemente correcto" que falla después, en silencio, al entrenar.
    # CONTRATO 59 C1: la validación interna debe ver el CSV EXACTAMENTE como
    # lo verá el entrenamiento real — target incluido — o valida un CSV que
    # nunca se entrena de verdad (mismo espíritu que la auditoría C2 de
    # BIBLIOTECA_MEJORAS_USO_REAL que introdujo esta llamada).
    validate_ranges = dict(res.get("field_ranges") or {})
    if target_range is not None:
        validate_ranges[target_header] = target_range
    validation = _validate_training_csv(
        res["mxai"], res["training_text"], prepared_csv,
        field_ranges=validate_ranges or None,
    )
    if not validation.get("ok"):
        raise DatasetProjectError(
            "El CSV preparado no pasa la validación del modelo que acaba de "
            "generarse (esto indica un hueco en la preparación del CSV, no un "
            f"problema de tus datos): {validation.get('errors') or validation.get('error')}"
        )

    excluded_columns = [
        col for col in analysis["column_order"]
        if col != target_column and columns[col]["type"] in _NOT_YET_USABLE_FEATURE_TYPES
    ]

    provenance = _build_provenance(
        csv_text=csv_text,
        prepared_csv=prepared_csv,
        schema_inferred=schema_inferred,
        schema_final=columns,
        target_column=target_column,
        excluded_columns=excluded_columns,
        rows_dropped_null_target=prepared.rows_dropped_null_target,
        feature_operations=prepared.operations,
        feature_name_map=feature_safe_names,
        target_label_map=target_label_map,
        task=task,
        prompt=prompt,
        training_text=res.get("training_text") or "",
        column_type_overrides=column_type_overrides or {},
        column_range_overrides=column_range_overrides or {},
        column_category_overrides=column_category_overrides or {},
        user_intent=normalized_intent,
        intent_llm=intent_llm,
        target_range=target_range,
        reconsidered_identifier_columns=reconsidered_columns,
    )

    result = dict(res)
    result["csv_text"] = prepared_csv
    result["provenance"] = provenance
    # field_ranges/field_types/field_categories YA vienen en `res` (extraídos
    # del prompt sintetizado por analyze_playground_request) — no se
    # duplican aquí, se devuelven tal cual llegaron.
    # CONTRATO 59 C1: `target_range` es DELIBERADAMENTE una clave separada de
    # `field_ranges` (que es solo-features en todo el resto del producto —
    # sliders de entrada, export, SchemaEditor) — mezclarla ahí filtraría el
    # target como si fuera un campo de entrada editable/normalizable en la UI.
    # `None` para clasificación o cuando el target no tiene rango numérico.
    result["target_range"] = list(target_range) if target_range is not None else None
    result["target_header"] = target_header
    return result


def generate_temporal_project_from_dataset(
    csv_text: str,
    target_column: str,
    *,
    temporal_column: str,
    horizon: int,
    lag_window_columns: list[str] | None = None,
    lag_window_size: int | None = None,
    column_type_overrides: dict[str, str] | None = None,
    column_range_overrides: dict[str, tuple[float, float]] | None = None,
    column_category_overrides: dict[str, list[str]] | None = None,
    user_intent: str | None = None,
    use_intent_llm: bool = False,
) -> dict[str, Any]:
    """C4 — flujo A, caso serie temporal: "columna temporal + ventana +
    horizonte → operaciones de C3" (contrato 57). Envoltorio DELGADO
    alrededor de `generate_project_from_dataset` (cero caminos paralelos,
    invariante 4) — nunca reimplementa la generación, solo prepara el CSV
    antes de entregárselo:

      1. `run_pipeline` (C3, `dataset_pipeline.py`) sobre el CSV crudo:
         `sort_temporal(temporal_column)` → `shift_target(target_column,
         horizon)` → opcionalmente `lag_window(lag_window_columns,
         lag_window_size)` → `missing_values(strategy=drop)` (limpia a la
         vez los bordes del lag y la cola sin target futuro).
      2. El CSV resultante + el nombre del target DESPLAZADO
         (`{target_column}_target_h{horizon}`) se entregan tal cual a
         `generate_project_from_dataset` — el resto del flujo (esquema,
         prompt tipado, generación, validación) es EXACTAMENTE el mismo
         que el caso no temporal.
      3. Anti-fuga (invariante 6/13): tras generar, se verifica con
         `check_anti_leakage` que ninguna FEATURE del proyecto resultante
         tenga desplazamiento temporal positivo — defensa en profundidad,
         nunca debería dispararse si el propio pipeline se construyó bien
         arriba, pero un futuro cambio en cómo se arma `ops` no puede
         colar una fuga en silencio.

    `column_type_overrides`/`column_range_overrides` (invariante 8) se
    aplican DESPUÉS del pipeline — sus claves son los nombres de columna
    TRANSFORMADOS (p.ej. `altura_ola_lag1`, no `altura_ola`), porque son
    los únicos que el usuario ve en el esquema final editable. **Excepción
    deliberada, solo para `column_type_overrides`**: una corrección de
    TIPO declarada sobre `target_column` (el nombre CRUDO, el único que el
    editor conoce — el target desplazado `{target_column}_target_h
    {horizon}` no existe hasta que este mismo envoltorio lo crea) se
    PROPAGA también a esa clave desplazada, además de aplicarse tal cual a
    `target_column` (que sigue existiendo como FEATURE tras el shift) —
    sin esto, corregir el tipo del target crudo nunca alcanzaba al target
    REAL usado para entrenar. `column_range_overrides` NO se propaga: un
    target (temporal o no, regresión o clasificación) nunca declara rango
    en el prompt sintetizado (`SALIDA: nombre` sin `en [lo, hi]`) — los
    rangos solo importan para FEATURES, y `target_column` ya los recibe
    tal cual por seguir siendo una feature tras el shift.

    Lanza `DatasetProjectError` si el pipeline no puede construirse
    (columna temporal/objetivo inexistente, parámetros inválidos), si
    `validate_pipeline_output` (C3) rechaza el resultado (min_rows, target
    presente, nulos residuales en target/features, columnas numéricas que
    no lo son de verdad) o si no queda ninguna fila tras
    `missing_values(drop)` (ventana/horizonte demasiado grandes para el
    dataset)."""
    from matrixai.training.dataset_pipeline import (
        PipelineError,
        check_anti_leakage,
        run_pipeline,
        validate_pipeline_output,
    )

    # Auditoría C4 [ALTA]: procedencia del CSV REALMENTE subido por el
    # usuario (hash + esquema) — antes de que el pipeline lo transforme.
    # `generate_project_from_dataset` (más abajo) recibe el CSV YA
    # transformado y lo trata como "el crudo", así que sin esto
    # `raw_csv_sha256`/`schema_inferred` de la procedencia final
    # describían el CSV post-pipeline, no el que el usuario vio y corrigió
    # — imposible reconstruir "CSV subido → esquema editado → pipeline →
    # CSV final" (invariante 3 del contrato).
    original_analysis = analyze_dataset_csv(csv_text)
    original_csv_sha256 = _sha256_text(csv_text)

    rows = _read_rows(csv_text)
    ops: list[dict[str, Any]] = [{"op": "sort_temporal", "column": temporal_column}]
    ops.append({"op": "shift_target", "column": target_column, "horizon": horizon})
    effective_target = f"{target_column}_target_h{horizon}"
    if lag_window_columns:
        if not lag_window_size:
            raise DatasetProjectError(
                "lag_window_size es obligatorio si se declara lag_window_columns."
            )
        ops.append({
            "op": "lag_window", "columns": list(lag_window_columns), "window": lag_window_size,
        })
    ops.append({"op": "missing_values", "strategy": "drop"})

    # Reauditoría 2026-07-17 (ronda 2) [ALTA]: el editor de esquema del SPA
    # analiza el CSV CRUDO (antes del pipeline) — el único nombre de target
    # que conoce es `target_column`. Una corrección de TIPO declarada
    # sobre esa clave se propaga también a `effective_target` (el target
    # REAL, que este envoltorio crea) — el original SIGUE recibiendo su
    # propia entrada tal cual, porque sigue existiendo como FEATURE tras
    # el shift.
    type_overrides = dict(column_type_overrides or {})
    if target_column in (column_type_overrides or {}):
        type_overrides[effective_target] = column_type_overrides[target_column]
    category_overrides = dict(column_category_overrides or {})
    if target_column in category_overrides:
        category_overrides[effective_target] = category_overrides[target_column]
    # CONTRATO 59 C1: `column_range_overrides` SÍ se propaga ahora — antes el
    # comentario de este bloque decía que un target "nunca declara rango" y
    # propagarlo sería código muerto; eso dejó de ser cierto en cuanto
    # `generate_project_from_dataset` empezó a calcular el rango del target
    # de regresión para normalizarlo (ver más abajo, `_range_for`). Sin este
    # eco, un target desplazado cuyo tipo crudo se corrigió a mano a
    # "number" (p.ej. porque la heurística de identificador lo atrapó, como
    # cualquier otra columna casi-única) se queda sin rango calculable y
    # `generate_project_from_dataset` revienta con "no tiene un rango
    # numérico calculable" — mismo patrón que type/category arriba.
    range_overrides = dict(column_range_overrides or {})
    if target_column in (column_range_overrides or {}):
        range_overrides[effective_target] = column_range_overrides[target_column]

    try:
        pipeline_result = run_pipeline(rows, ops)
    except PipelineError as exc:
        raise DatasetProjectError(f"Serie temporal: {exc}") from exc

    # Reauditoría 2026-07-17 (ronda 2) [MEDIA]: C3 documenta `feature_
    # columns`/`expected_types` como opcionales A PROPÓSITO para que el
    # caller que conoce el esquema final los declare (ver 4ª pasada de C3)
    # — C4 es ese caller. `expected_types` solo puede declarar tipos de
    # `_CAST_TYPES` (number/integer/string), así que se limita a las
    # columnas cuyo tipo FINAL (esquema original de C1 + overrides ya
    # traducidos arriba) es number/integer — es lo único verificable aquí,
    # y atrapa un CSV que no es realmente numérico con un mensaje
    # específico antes de la generación, no el rechazo genérico de GEN más
    # abajo.
    expected_types: dict[str, str] = {}
    effective_schema_types: dict[str, str] = {}
    for col, info in original_analysis["columns"].items():
        effective_type = type_overrides.get(col, info["type"])
        effective_schema_types[col] = effective_type
        if effective_type not in ("number", "integer"):
            continue
        expected_types[col] = effective_type
        if col == target_column:
            # Mismo criterio de propagación que los overrides arriba: el
            # target desplazado comparte los valores (y por tanto el tipo)
            # del target crudo.
            expected_types[effective_target] = effective_type
    for col in lag_window_columns or []:
        source_type = effective_schema_types.get(col)
        for lag in range(1, (lag_window_size or 0) + 1):
            lag_col = f"{col}_lag{lag}"
            lag_type = type_overrides.get(lag_col, source_type)
            if lag_type in ("number", "integer"):
                expected_types[lag_col] = lag_type
    # Un override sobre cualquier nombre ya transformado también forma
    # parte del contrato de salida, aunque no sea una columna de lag.
    for col, overridden_type in type_overrides.items():
        if overridden_type in ("number", "integer"):
            expected_types[col] = overridden_type
    final_columns = pipeline_result.steps[-1].columns_after if pipeline_result.steps else []
    feature_columns = [c for c in final_columns if c != effective_target]

    # Auditoría C4 [MEDIA]: la validación final obligatoria de C3
    # ("min_rows, tipos, nulos residuales, target presente") no se estaba
    # ejecutando — solo se comprobaba "queda alguna fila". `missing_values
    # (drop)` ya deja el target y las features sin nulos por construcción,
    # así que lo que la comprobación de nulos añade de verdad es
    # `min_rows` (una sola fila sobreviviente pasaría "queda alguna fila"
    # pero no basta para entrenar nada); `expected_types` sí es un chequeo
    # nuevo genuino (arriba).
    validation_errors = validate_pipeline_output(
        pipeline_result.rows, target_column=effective_target,
        feature_columns=feature_columns, expected_types=expected_types,
    )
    if validation_errors:
        raise DatasetProjectError("Serie temporal: " + "; ".join(validation_errors))
    prepared_csv = _rows_to_csv_text(pipeline_result.rows)

    result = generate_project_from_dataset(
        prepared_csv, effective_target,
        column_type_overrides=type_overrides or None,
        column_range_overrides=range_overrides or None,
        column_category_overrides=category_overrides or None,
        # Contrato 58 C4/C5 — el camino temporal NUNCA ignora la intención (ni
        # su interpretación LLM) en silencio: se enhebran tal cual
        # (normalización/validación/llamada LLM ocurren dentro de
        # `generate_project_from_dataset`, un solo sitio).
        user_intent=user_intent,
        use_intent_llm=use_intent_llm,
    )

    feature_columns = list(result["provenance"]["feature_name_map"].keys())
    leaks = check_anti_leakage(pipeline_result, feature_columns)
    if leaks:
        raise DatasetProjectError("Serie temporal: " + "; ".join(leaks))

    # Auditoría C4 [ALTA]: GEN emite SIEMPRE `SPLIT train=X validation=Y
    # seed=42` — nunca `mode=temporal` (no es un parámetro que el
    # generador conozca). Sin esto, un proyecto "serie temporal" entrenaba
    # con split ALEATORIO, anulando la protección de C3 (invariante 6/13)
    # en el momento exacto en que más importa: el entrenamiento real. Se
    # reescribe la línea SPLIT ya generada (mismo ratio, sin seed —
    # mode=temporal no lo admite, ver parser.py) en vez de enseñarle a GEN
    # un concepto que no le pertenece (GEN no sabe nada de series
    # temporales; C3/C4 sí).
    result["training_text"] = _force_temporal_split(result.get("training_text") or "")

    # Reauditoría 2026-07-17 (ronda 2) [MEDIA]: `provenance["seed"]` se
    # había extraído del `training_text` ALEATORIO original (seed=42, el
    # que GEN siempre emite) ANTES de la reescritura de arriba — el
    # resultado finalmente entregado no tiene seed (mode=temporal lo
    # rechaza), pero la procedencia seguía "recordando" el seed viejo,
    # contradiciendo el propio training_text devuelto. Se re-extrae del
    # texto YA reescrito — mismo criterio que `_extract_seed` ya declara
    # en su docstring ("para que la procedencia nunca pueda divergir del
    # training_text real devuelto").
    prov_seed_source = result["training_text"]

    # Auditoría C4 [ALTA]: corrige la procedencia para que describa el
    # pipeline COMPLETO — el CSV que `generate_project_from_dataset` trató
    # como "crudo" es en realidad el resultado del pipeline C3; se
    # renombra a `post_pipeline_csv_sha256` y se restaura el hash/esquema
    # del CSV ORIGINAL en su lugar. Las operaciones de C3 se anteponen a
    # las de C2 en `operations` — son las que ocurrieron primero.
    prov = result["provenance"]
    prov["seed"] = _extract_seed(prov_seed_source)
    prov["post_pipeline_csv_sha256"] = prov["raw_csv_sha256"]
    prov["raw_csv_sha256"] = original_csv_sha256
    prov["schema_inferred"] = original_analysis["columns"]
    prov["operations"] = [s.operation for s in pipeline_result.steps] + prov["operations"]
    prov["temporal"] = {
        "temporal_column": temporal_column,
        "raw_target_column": target_column,
        "horizon": horizon,
        "lag_window_columns": list(lag_window_columns or []),
        "lag_window_size": lag_window_size,
        "pipeline_operations": [s.to_dict() for s in pipeline_result.steps],
    }
    return result


def _force_temporal_split(training_text: str) -> str:
    """Reescribe la línea `SPLIT` de un `training_text` ya generado para
    declarar `mode=temporal` (nunca baraja) en vez del `seed=...` que GEN
    siempre emite — ver el comentario en el punto de uso. Preserva la
    ratio train/validation declarada y la indentación original."""
    from matrixai.training.parser import _SPLIT_RE

    out_lines = []
    for line in training_text.split("\n"):
        stripped = line.strip()
        match = _SPLIT_RE.match(stripped)
        if match:
            leading_ws = line[: len(line) - len(line.lstrip())]
            out_lines.append(
                f"{leading_ws}SPLIT train={match.group('train')} "
                f"validation={match.group('validation')} mode=temporal"
            )
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_rows(csv_text: str) -> list[dict[str, str]]:
    from matrixai.training.data import normalize_csv_text
    normalized = normalize_csv_text(csv_text)
    return list(csv.DictReader(io.StringIO(normalized)))


def _rows_to_csv_text(rows: list[dict[str, str]]) -> str:
    """Inverso de `_read_rows` — serializa las filas YA transformadas por
    `run_pipeline` (C3) de vuelta a texto CSV para entregárselas a
    `generate_project_from_dataset` (C4, caso serie temporal)."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


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


def _numeric_range_from_raw_values(
    raw_values: list[str | None],
) -> tuple[str, tuple[float, float]] | None:
    """CONTRATO 59 C2 — recalcula tipo numérico + rango (con margen) de una
    columna a partir de sus valores CRUDOS del CSV: mismo cálculo EXACTO
    que `dataset_analysis._analyze_column` aplicaría si hubiera visto la
    columna así desde el principio (mismos `_numeric_kind` +
    `_propose_margin` + `_round_range`, importados de ese módulo — una
    sola fuente de verdad para "qué es un rango numérico válido").

    Se usa en dos sitios: cuando el usuario corrige el tipo a mano a
    number/integer sobre una columna que C1 clasificó sin rango
    (identifier/unknown/categórica), y cuando se reconsidera un
    identificador porque dejaría el proyecto sin features (decisión C).

    Devuelve `None` si los valores no son numéricos de verdad (nunca
    fuerza un identificador de texto tipo UUID, ni una columna vacía) —
    el caller decide qué hacer con ese `None` (mantener el tipo pedido sin
    rango sigue fallando más adelante en `_range_for`, con el mismo
    mensaje accionable de siempre)."""
    non_null = [v.strip() for v in raw_values if not _is_null(v)]
    if not non_null:
        return None
    numeric_kind = (
        None if any(_has_significant_leading_zero(v) for v in non_null)
        else _numeric_kind(non_null)
    )
    if numeric_kind is None:
        return None
    values = [float(v) for v in non_null]
    lo, hi = min(values), max(values)
    rng = _round_range(_propose_margin(lo, hi), numeric_kind)
    return numeric_kind, (float(rng[0]), float(rng[1]))


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _safe_field_name(name: str) -> str:
    """Nombre de campo seguro para el prompt (invariante: el VALOR real de
    esta etiqueta se descarta siempre — GEN emite `predicted_class`/
    `predicted_value` pase lo que pase, verificado — pero el prompt debe
    parsear, así que se sanea igual que un identificador de campo."""
    safe = _identifier(name)
    return safe or "objetivo"


# Auditoría C2 [MEDIA, residual]: centinela para distinguir, dentro de
# `_normalize_feature_names`, "colisiona con el nombre reservado del
# target" de "colisiona con otra columna real" — mensajes distintos,
# mismo mecanismo de detección.
_RESERVED_TARGET_SENTINEL = "\0target_header\0"


def _normalize_feature_names(feature_columns: list[str], target_header: str) -> dict[str, str]:
    """Nombre de columna cruda -> nombre de campo seguro para VECTOR/CSV.

    GEN sanea el nombre de cada FEATURE con `_sanitize_name`
    (`prompt_field_specs.py`) — idéntica a `_identifier` (NFKD, no-alnum a
    '_', minúsculas) — al construir el VECTOR. Si el CSV preparado usara el
    nombre CRUDO, el VECTOR generado y la cabecera del CSV divergirían en
    cuanto la columna tuviera espacios/acentos/símbolos (auditoría C2
    [ALTA]). Se aplica aquí la MISMA función, con colisión (dos columnas
    crudas distintas que sanean igual) como error accionable — GEN se
    quedaría con la primera y descartaría la segunda en silencio si no se
    detectara antes.

    También detecta que una FEATURE normalice al mismo nombre que
    `target_header` (`predicted_class`/`predicted_value`, SIEMPRE
    reservado para el target — ver punto 1 del docstring del módulo): sin
    este chequeo, el prompt sintetizado generaba un VECTOR con dos campos
    llamados igual (uno de entrada, uno de salida) y GEN lo rechazaba con
    un error interno de cientos de líneas, nada accionable (auditoría C2
    [MEDIA, residual])."""
    mapping: dict[str, str] = {}
    seen: dict[str, str] = {target_header: _RESERVED_TARGET_SENTINEL}
    for col in feature_columns:
        safe = _identifier(col)
        if not safe:
            raise DatasetProjectError(
                f"La columna {col!r} no tiene un nombre de campo válido tras "
                "normalizar (solo símbolos/espacios) — renómbrala en el CSV."
            )
        if safe in seen and seen[safe] != col:
            if seen[safe] == _RESERVED_TARGET_SENTINEL:
                raise DatasetProjectError(
                    f"La columna {col!r} normaliza a {safe!r}, el nombre "
                    "reservado que el modelo generado usa siempre para la "
                    "columna objetivo — renómbrala en el CSV de origen."
                )
            raise DatasetProjectError(
                f"Las columnas {seen[safe]!r} y {col!r} generan el mismo "
                f"nombre de campo tras normalizar ({safe!r}) — el modelo no "
                "puede distinguirlas. Renombra una de las dos columnas."
            )
        seen[safe] = col
        mapping[col] = safe
    return mapping


def _check_categorical_values_safe(values: list[str], col: str) -> None:
    """Auditoría C2 [ALTA] (ver punto 8 del docstring): un valor con ',',
    ']' o salto de línea rompería el corchete `Categorical[...]` — GEN lo
    parsea partiendo por comas sin escape, así que "red,blue" se leería
    como DOS categorías mientras el CSV lo trataría como una. Fallo cerrado
    en vez de sintetizar un prompt que generaría un modelo desalineado."""
    for v in values:
        if any(ch in v for ch in _UNSAFE_CATEGORY_CHARS):
            raise DatasetProjectError(
                f"El valor {v!r} de la columna categórica {col!r} contiene una "
                "coma, ']' o un salto de línea — el prompt tipado no puede "
                "representarlo sin ambigüedad. Limpia ese valor en el CSV de "
                "origen antes de generar el modelo."
            )


def _slug(raw: str) -> str:
    """Como `_identifier` pero SIN el veto de dígito inicial — solo se usa
    como base del prefijo `class_` cuando `_identifier` rechaza un valor
    por empezar con número (ver `_normalize_labels`)."""
    text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def _normalize_labels(
    raw_values: list[str], target_column: str
) -> tuple[list[str], dict[str, str]]:
    """`_identifier(v)` por cada valor — la MISMA normalización que GEN
    aplica a `ProbabilityMap[...]` por dentro (ver docstring del módulo,
    punto 2). Auditoría C2 [ALTA] (ver punto 7 del docstring): un valor que
    EMPIEZA por dígito ("0"/"1" — el booleano canónico que C1 reconoce a
    propósito, o "24h") queda vacío tras `_identifier` (los identificadores
    no pueden empezar por número) pero SÍ tiene contenido real — se
    reintenta con el prefijo `class_` en vez de descartarlo. Solo un valor
    SIN ningún carácter alfanumérico ("###") sigue siendo ambigüedad real,
    error accionable. También detecta colisiones (dos valores crudos
    DISTINTOS que normalizan igual, p.ej. "Sí" y "SI").
    Devuelve `(etiquetas_ordenadas, mapa_valor_crudo->etiqueta)` — el mapa
    se reutiliza en `_prepare_training_csv` para que cada fila del CSV use
    EXACTAMENTE la misma etiqueta que se escribió en el prompt, nunca una
    normalización recalculada por separado que podría divergir."""
    normalized: dict[str, str] = {}  # etiqueta -> primer valor crudo (para colisiones)
    raw_to_label: dict[str, str] = {}
    for raw in raw_values:
        norm = _identifier(raw)
        if not norm:
            slug = _slug(raw)
            norm = f"class_{slug}" if slug else ""
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
        raw_to_label[raw] = norm
    return sorted(normalized.keys()), raw_to_label


def _prepare_training_csv(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    columns: dict[str, dict[str, Any]],
    feature_safe_names: dict[str, str],
    target_column: str,
    task: str,
    target_label_map: dict[str, str] | None,
    target_header: str,
    category_vocabularies: dict[str, list[str]],
) -> _PreparedCSV:
    # Grupos one-hot/embedding + los mapas valor_crudo->columna o índice,
    # calculados UNA VEZ (no por fila — recalcular _distinct_non_null
    # dentro del bucle de filas es O(filas²) y, peor, podría ver un
    # vocabulario distinto por fila si el cálculo no fuera puramente
    # determinista).
    onehot_columns: dict[str, dict[str, str]] = {}  # col -> {valor_crudo: columna_onehot}
    embedding_columns: dict[str, dict[str, int]] = {}  # col -> {valor_crudo: índice}
    header: list[str] = []
    operations: list[str] = []
    for col in feature_columns:
        safe_name = feature_safe_names[col]
        col_type = columns[col]["type"]
        if col_type == "categorical":
            values = category_vocabularies.get(col) or _distinct_non_null(rows, col)
            if len(values) < 2:
                continue
            if len(values) > _ONEHOT_MAX:
                # Auditoría C2 [ALTA] (ver punto 6 del docstring): GEN
                # enrutó esta columna al composite con EMBEDDING nativo —
                # el CSV lleva el ÍNDICE del valor en el vocabulario (mismo
                # orden que se escribió en el prompt), NUNCA one-hot.
                embedding_columns[col] = {v: i for i, v in enumerate(values)}
                header.append(safe_name)
                if "embed_high_cardinality_categoricals" not in operations:
                    operations.append("embed_high_cardinality_categoricals")
            else:
                names = _build_group_names(safe_name, values)
                onehot_columns[col] = dict(zip(values, names))
                header.extend(names)
                if "expand_categoricals_onehot" not in operations:
                    operations.append("expand_categoricals_onehot")
        else:
            header.append(safe_name)
            if col_type == "boolean" and "normalize_boolean_features" not in operations:
                operations.append("normalize_boolean_features")
    header.append(target_header)

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header)
    writer.writeheader()
    rows_dropped = 0
    for row in rows:
        target_raw = row.get(target_column)
        if _is_null(target_raw):
            rows_dropped += 1
            continue  # sin target no hay fila que entrenar (nunca se inventa uno)
        prepared: dict[str, str] = {}
        for col in feature_columns:
            info = columns[col]
            safe_name = feature_safe_names[col]
            if info["type"] == "categorical":
                if col in onehot_columns:
                    value_to_column = onehot_columns[col]
                    for onehot_col in value_to_column.values():
                        prepared[onehot_col] = "0"
                    raw = row.get(col)
                    raw = raw.strip() if raw is not None else raw
                    if raw in value_to_column:
                        prepared[value_to_column[raw]] = "1"
                elif col in embedding_columns:
                    raw = row.get(col)
                    raw = raw.strip() if raw is not None else raw
                    idx = embedding_columns[col].get(raw)
                    prepared[safe_name] = str(idx) if idx is not None else ""
                # cardinalidad<2 -> columna excluida arriba, nada que escribir
            elif info["type"] == "boolean":
                raw = (row.get(col) or "").strip().lower()
                if raw in _BOOL_TRUE:
                    prepared[safe_name] = "1"
                elif raw in _BOOL_FALSE:
                    prepared[safe_name] = "0"
                else:
                    prepared[safe_name] = row.get(col, "")  # deja que el verificador existente lo rechace
            else:
                prepared[safe_name] = row.get(col, "")
        raw_target = target_raw.strip()
        if task == "classification":
            prepared[target_header] = (target_label_map or {}).get(raw_target, "")
        else:
            prepared[target_header] = raw_target
        writer.writerow(prepared)
    return _PreparedCSV(text=out.getvalue(), rows_dropped_null_target=rows_dropped, operations=operations)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_seed(training_text: str) -> int | None:
    """El seed de SPLIT que `training_text` ya declara (GEN lo fija, C2 no
    lo elige) — se re-extrae en vez de duplicar el literal para que la
    procedencia nunca pueda divergir del training_text real devuelto."""
    m = re.search(r"\bseed=(\d+)", training_text)
    return int(m.group(1)) if m else None


def _build_provenance(
    *,
    csv_text: str,
    prepared_csv: str,
    schema_inferred: dict[str, Any],
    schema_final: dict[str, Any],
    target_column: str,
    excluded_columns: list[str],
    rows_dropped_null_target: int,
    feature_operations: list[str],
    feature_name_map: dict[str, str],
    target_label_map: dict[str, str] | None,
    task: str,
    prompt: str,
    training_text: str,
    column_type_overrides: dict[str, str],
    column_range_overrides: dict[str, tuple[float, float]],
    column_category_overrides: dict[str, list[str]],
    user_intent: str | None = None,
    intent_llm: dict[str, Any] | None = None,
    target_range: tuple[float, float] | None = None,
    reconsidered_identifier_columns: list[str] | None = None,
) -> dict[str, Any]:
    from matrixai.export.inference_spec import _matrixai_version

    # Auditoría C2 [MEDIA]: `feature_operations` viene de lo que
    # `_prepare_training_csv` hizo REALMENTE con las FEATURES — antes se
    # miraba `schema_final` completo (incluido el target), así que un
    # target categórico (el caso normal de clasificación) declaraba
    # "expand_categoricals_onehot" aunque ninguna feature se hubiera
    # expandido.
    operations: list[str] = [
        f"rename_target_column:{target_column}->predicted_"
        f"{'class' if task == 'classification' else 'value'}"
    ]
    if task == "classification":
        operations.append("normalize_target_labels")
    # CONTRATO 59 C2 (hallazgo de auditoría): sin esto, un cambio de tipo
    # automático (sin `column_type_overrides` del usuario) era invisible en
    # la procedencia — la única otra vía para que el tipo cambiara.
    for col in (reconsidered_identifier_columns or []):
        operations.append(f"reconsidered_identifier_as_feature:{col}")
    operations.extend(feature_operations)

    return {
        "source": "user_upload",
        "raw_csv_sha256": _sha256_text(csv_text),
        "prepared_csv_sha256": _sha256_text(prepared_csv),
        "schema_inferred": schema_inferred,
        "schema_final": schema_final,
        "target_column": target_column,
        # Auditoría C2 [MEDIA]: qué se excluyó (identificador/fecha/vacía)
        # y cuántas filas se perdieron por target nulo — antes la
        # procedencia no dejaba rastro de ninguna de las dos cosas.
        "excluded_columns": excluded_columns,
        "rows_dropped_null_target": rows_dropped_null_target,
        # Auditoría C2 [MEDIA, reauditoría]: mapa cabecera_original ->
        # cabecera_normalizada y valor_crudo -> etiqueta (None en
        # regresión, donde no hay etiquetas) — la reversibilidad que la
        # auditoría pedía ya existía en memoria (`_normalize_feature_names`/
        # `_normalize_labels`) pero se descartaba al no viajar en la
        # procedencia devuelta.
        "feature_name_map": feature_name_map,
        "target_label_map": target_label_map,
        "task": task,
        "column_type_overrides": column_type_overrides,
        "column_range_overrides": {k: list(v) for k, v in column_range_overrides.items()},
        "column_category_overrides": column_category_overrides,
        "synthesized_prompt": prompt,
        "operations": operations,
        "seed": _extract_seed(training_text),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrixai_version": _matrixai_version(),
        # Contrato 58 C4 — intención LOCAL del usuario, ya normalizada. NUNCA
        # forma parte de `synthesized_prompt` (arriba) — ver user_intent.py.
        "user_intent": user_intent,
        # Contrato 58 C5 — auditoría de la interpretación LLM opt-in de esa
        # intención (None si no hay intención) — ver intent_llm.py.
        "intent_llm": intent_llm,
        # CONTRATO 59 C1: rango de dominio del target usado para normalizar
        # antes de entrenar y desnormalizar la predicción — None en
        # clasificación. Fuente auditable para `_studio_infer` (evita
        # confiar en un valor recalculado ad-hoc en otro punto del código).
        "target_range": list(target_range) if target_range is not None else None,
    }
