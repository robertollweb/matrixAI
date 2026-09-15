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
9. **Un hueco no se puede escribir tal cual.** Una celda vacía en una
   FEATURE hacía que el modelo recién generado rechazara su propio CSV
   ("DATASET row N field X is empty") y el proyecto entero moría —
   reproducido el 2026-09-13 con 12 filas sintéticas y dos huecos en una
   numérica, y con `KDDCup09_appetency`, donde en las categóricas el
   faltante llega como `?` (que `_is_null` considera nulo y por tanto
   fuera del vocabulario). Se aplica la política del NÚCLEO
   (`preparacion.py`, 103-C3), que este camino no usaba: numérica →
   MEDIANA más un indicador `{columna}_faltante`, para que el modelo pueda
   distinguir "valía eso" de "no había valor"; categórica → `__faltante__`
   como categoría propia, sin colapsarla con ninguna otra (antes: one-hot
   todo a cero, o un índice inexistente escrito como ""). Solo en las
   columnas que de verdad tienen huecos: un CSV limpio sale byte a byte
   igual que antes. El objetivo NUNCA se imputa — esas filas se descartan,
   como siempre. Lo imputado se declara en `provenance["missing_values"]`
   y la política queda CONGELADA en la receta (`preparation_spec
   ["missing_policy"]`): re-preparar no vuelve a calcular la mediana.

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
import json
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
from matrixai import limits as _limits
from matrixai.training.categorical import _build_group_names, embedding_source_columns
from matrixai.training.dense_generator import _identifier, _ONEHOT_MAX
# LA POLÍTICA DE FALTANTES DEL NÚCLEO, NO UNA SEGUNDA. `preparacion.py`
# (103-C3) ya declara qué se hace con un hueco: mediana —no media, que la
# mueve un extremo— más un indicador `{columna}__faltante` para que el
# modelo pueda distinguir «valía 0» de «no había valor», y para una
# categórica un TERCER estado propio (`__faltante__`) que no se colapsa ni
# con la categoría de referencia ni con una desconocida. Este camino no la
# usaba (hueco de CABLEADO nº 15): se ajusta y se aplica con las funciones
# del núcleo, y lo único que vive aquí es la traducción del CSV crudo —que
# es todo texto— a los valores tipados que `ajustar_preparacion` exige.
from matrixai.training.preparacion import (
    CATEGORIA_DESCONOCIDA,
    CATEGORIA_FALTANTE,
    PoliticaDePreparacion,
    ajustar_preparacion,
    nombre_de_indicador,
    transformar_fila,
)
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

# El aviso de la columna de TEXTO que se queda fuera. Lo redacta el core, así
# que se traduce en el core (misma regla que `_MARCOS` en playground.py).
#
# Dice las TRES cosas, porque las tres son la noticia: qué columna, que se ha
# quedado fuera del modelo, y por dónde SÍ se puede hacer lo que quería. Un
# aviso que solo dijera «excluida» dejaría a su dueño pensando que el
# producto no sabe leer texto — y sabe.
_AVISO_COLUMNA_TEXTO = {
    "es": lambda columnas, ejemplo: (
        f"La(s) columna(s) {columnas} contiene(n) TEXTO escrito por una persona y se "
        "ha(n) dejado fuera del modelo: el camino desde datos solo usa columnas "
        "numéricas, categóricas o booleanas. Para aprovechar ese texto, crea el modelo "
        f"desde una descripción declarando el campo así: «{ejemplo}: Text» — entrena con "
        "la columna tal cual, sin convertirla a números."),
    "en": lambda columnas, ejemplo: (
        f"Column(s) {columnas} hold(s) TEXT written by a person and were left out of the "
        "model: the from-data path only uses numeric, categorical or boolean columns. To "
        "use that text, create the model from a description declaring the field as "
        f"«{ejemplo}: Text» — it trains on the column as-is, without turning it into "
        "numbers."),
}

_CLASSIFICATION_TARGET_TYPES = {"boolean", "categorical"}
_REGRESSION_TARGET_TYPES = {"number", "integer"}

# Auditoría C2 [ALTA]: caracteres que romperían el parseo de
# `Categorical[v1, v2, ...]` si aparecieran DENTRO de un valor — ',' es el
# separador (sin escape posible, verificado en prompt_field_specs.py),
# ']' cierra el corchete, '\n'/'\r' rompen el límite de línea del parser.
_UNSAFE_CATEGORY_CHARS = (",", "]", "\n", "\r")


class DatasetProjectError(ValueError):
    """Esquema/target inválido para generar un proyecto — error accionable
    (invariante 7 del contrato 57): nunca un proyecto a medias.

    CONTRATO 62 C1: puede transportar el payload estructurado de un tope
    superado (`limits.limit_error`) en `.details`. Un tope NO es un fallo de
    esquema, así que el caller necesita distinguirlos sin parsear el texto.
    `.details` es `None` para el resto de errores de este tipo.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details


# CONTRATO 62 C3 — versión de la RECETA de preparación. Se sube cuando cambia
# lo que `_prepare_training_csv` produce para las mismas entradas (orden de
# columnas, codificación one-hot/embedding, normalización booleana…). No es la
# versión del paquete: dos versiones distintas de MatrixAI con la misma receta
# comparten este número, y ese es justo el punto — permite saber si un modelo
# guardado se puede reproducir byte a byte o solo aproximar.
PREPARATION_SPEC_VERSION = 1


@dataclass
class _PreparedCSV:
    text: str
    rows_dropped_null_target: int
    operations: list[str] = field(default_factory=list)
    # CONTRATO 62 C3 — vocabulario REALMENTE usado por columna categórica.
    # Sin congelarlo no hay reproducción posible: para una categórica sin
    # override, `_prepare_training_csv` lo calculaba al vuelo con
    # `_distinct_non_null`, así que un CSV con un valor de más (o de menos)
    # produciría otras columnas one-hot y otro VECTOR.
    effective_vocabularies: dict[str, list[str]] = field(default_factory=dict)
    # Nombres SAFE de las categóricas que este CSV escribió como ÍNDICE de
    # embedding (una columna) en vez de one-hot (N columnas). Se congela en la
    # receta por el mismo motivo que el vocabulario: la decisión la tomó el
    # MODELO generado, no una regla recalculable desde el CSV crudo, así que
    # re-preparar sin ella produciría otras columnas.
    embedding_source_names: list[str] = field(default_factory=list)
    # LO QUE PASÓ DE VERDAD, no lo que la política pedía: celdas realmente
    # rellenadas por columna (nombre SAFE) y celdas categóricas que se
    # escribieron como `__faltante__`. Se CUENTAN al escribir cada fila, no se
    # derivan de `proporcion_faltante`, porque una imputación es afirmar algo
    # que el dato no decía y la procedencia tiene que declarar lo ocurrido.
    imputed_numeric_cells: dict[str, int] = field(default_factory=dict)
    missing_category_cells: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LOS RECHAZOS DE «NINGUNA COLUMNA UTILIZABLE», EN LOS DOS IDIOMAS.
#
# Estaban en español fijo, y son de los pocos mensajes de este módulo que
# ve alguien de FUERA: es lo que recibe quien sube su primer CSV y no
# sale nada. Medido el 2026-08-14 con `locale=en`: la aplicación en
# inglés y el rechazo en español. Media aplicación traducida se ve peor
# que ninguna — y este es justo el momento en el que alguien decide si
# el producto le sirve.
#
# Se traduce AQUÍ porque lo redacta el core, misma regla que `_MARCOS`
# en `playground.py`. Los tres van juntos porque son el mismo momento
# —«no hay con qué entrenar»— y arreglar uno solo dejaría a su vecino
# contestando en otro idioma en la pantalla de al lado.
#
# El español sigue siendo el de por defecto: quien no pase `locale`
# recibe exactamente lo de antes, palabra por palabra.
_SIN_FEATURES: dict[str, dict[str, Any]] = {
    "es": {
        "constantes": lambda cols: (
            f"Ninguna columna es utilizable como feature: {cols} tienen un único valor en "
            "todo el CSV y no aportan información. Añade columnas que varíen, "
            "o consérvalas explícitamente declarando su rango de dominio."),
        "texto": lambda cols: (
            f"Ninguna columna es utilizable como feature: {cols} "
            "contiene(n) TEXTO escrito por una persona, y el camino desde datos "
            "todavía no construye modelos de texto (solo columnas numéricas, "
            "categóricas o booleanas). Para un modelo de texto, créalo desde una "
            f"descripción declarando el campo así: «{cols[0]}: Text» "
            "— entrena con esa columna tal cual, sin convertirla a números."),
        "nada": ("Ninguna columna es utilizable como feature (todas son el target, "
                 "identificadores, fechas o columnas vacías) — no hay nada con lo "
                 "que entrenar."),
    },
    "en": {
        "constantes": lambda cols: (
            f"No column can be used as a feature: {cols} hold a single value across "
            "the whole CSV and carry no information. Add columns that vary, or keep "
            "them explicitly by declaring their domain range."),
        "texto": lambda cols: (
            f"No column can be used as a feature: {cols} hold(s) TEXT written by a "
            "person, and the from-data path does not build text models yet (numeric, "
            "categorical or boolean columns only). For a text model, create it from a "
            f"description declaring the field like this: «{cols[0]}: Text» — it trains "
            "on that column as it is, without turning it into numbers."),
        "nada": ("No column can be used as a feature (they are all the target, "
                 "identifiers, dates or empty columns) — there is nothing to train "
                 "with."),
    },
}


def generate_project_from_dataset(
    csv_text: str,
    target_column: str,
    *,
    column_type_overrides: dict[str, str] | None = None,
    column_range_overrides: dict[str, tuple[float, float]] | None = None,
    column_category_overrides: dict[str, list[str]] | None = None,
    keep_constant_columns: list[str] | None = None,
    user_intent: str | None = None,
    use_intent_llm: bool = False,
    locale: str = "es",
    # CONTRATO 103 C1 — las respuestas a lo que hay que CONFIRMAR del problema.
    # Todas opcionales y todas `None` por defecto: sin ellas se genera igual que
    # siempre y `result['confirmacion']` enumera lo que falta; con ellas, el
    # problema queda confirmado y viaja como `ProblemSpec` (104-C0).
    unidad_de_observacion: str | None = None,
    clase_positiva: str | None = None,
    momento_de_prediccion: str | None = None,
    horizonte: Any | None = None,
    uso_previsto: str | None = None,
    # Cómo marca la AUSENCIA el origen de este CSV, si es que lo sabe. `None`
    # —el defecto, y el camino de un CSV subido a mano— deja la heurística
    # `_NULL_TOKENS` de siempre. Ver `dataset_analysis._is_null`: se declara
    # aquí y gobierna TODO el paso (análisis, vocabularios, política de
    # faltantes y escritura del CSV preparado), no solo el análisis — si solo
    # gobernara el análisis, la preparación seguiría escribiendo `__faltante__`
    # en celdas que sí traían dato, que es el defecto con otra cara.
    tokens_de_ausencia: set[str] | None = None,
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

    analysis = analyze_dataset_csv(csv_text, tokens_de_ausencia=tokens_de_ausencia)
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
            recomputed = _numeric_range_from_raw_values(
                [row.get(col) for row in rows], tokens_de_ausencia)
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

    # CONTRATO 62 C2 — una feature CONSTANTE (un solo valor distinto en todo
    # el CSV) no aporta información al modelo y además arrastra dos daños
    # concretos, ambos medidos con el dataset de lluvia de una sola ciudad:
    # `_propose_margin` le inventa un rango de ±1 (`lat` 43.46 → [42.46,
    # 44.46]), y con ese rango el panel de inferencia pinta un slider que solo
    # sirve para sacar al modelo de la distribución en la que se entrenó.
    # Se excluye por defecto y se dice cuál; conservarla es una decisión
    # explícita del usuario (`keep_constant_columns`).
    #
    # INVARIANTE 3 del contrato: esto es una regla de GENERACIÓN. Re-preparar
    # un modelo ya existente debe reproducir el esquema con el que nació —
    # `keep_constant_columns` es justo la palanca que permitirá a C3 replicar
    # un esquema anterior a este corte sin perder columnas.
    _keep_constant = {str(c) for c in (keep_constant_columns or [])}
    constant_columns = [
        col for col in analysis["column_order"]
        if col != target_column
        and columns[col]["type"] not in _NOT_YET_USABLE_FEATURE_TYPES
        and columns[col].get("constant")
    ]
    dropped_constant_columns = [c for c in constant_columns if c not in _keep_constant]
    kept_constant_columns = [c for c in constant_columns if c in _keep_constant]

    # Conservar una constante NUMÉRICA exige un rango de dominio real
    # (`min < max`) declarado por el usuario: sin él seguiríamos con el rango
    # inventado, que es exactamente lo que este corte viene a eliminar.
    for col in kept_constant_columns:
        _tipo = columns[col]["type"]
        if _tipo in ("number", "integer"):
            override = (column_range_overrides or {}).get(col)
            if override is None or not (float(override[0]) < float(override[1])):
                raise DatasetProjectError(
                    f"La columna {col!r} tiene un único valor en todo el CSV "
                    f"({columns[col].get('constant_value')!r}). Para conservarla como "
                    "feature hay que declarar su rango de dominio real (mínimo y "
                    "máximo, con mínimo < máximo); si no, quítala del modelo."
                )
        elif _tipo == "categorical":
            # AUDITORÍA C2 [ALTO]: conservar una categórica constante decía
            # "hecho" y luego la columna desaparecía del .mxai y del CSV
            # preparado, porque one-hot con UN solo valor no tiene sentido
            # (`len(values) < 2: continue`) — una columna siempre a 1 no es
            # una feature, es un sesgo. El equivalente al rango de dominio de
            # una numérica es aquí declarar el VOCABULARIO completo esperado:
            # con dos o más valores la columna entra de verdad y el modelo
            # puede aprender de ella cuando lleguen datos con variedad.
            vocab = (column_category_overrides or {}).get(col)
            if not vocab or len(vocab) < 2:
                raise DatasetProjectError(
                    f"La columna categórica {col!r} tiene un único valor en todo el "
                    f"CSV ({columns[col].get('constant_value')!r}). Para conservarla "
                    "hay que declarar su vocabulario completo (al menos dos valores "
                    "posibles); si no, quítala del modelo: una categórica de un solo "
                    "valor produce una columna constante que no aporta nada."
                )

    feature_columns = [
        col for col in analysis["column_order"]
        if col != target_column
        and columns[col]["type"] not in _NOT_YET_USABLE_FEATURE_TYPES
        and col not in dropped_constant_columns
    ]
    # Las columnas que C1 ha visto como TEXTO ESCRITO POR UNA PERSONA. Se
    # excluyen exactamente igual que antes —el camino desde datos no
    # construye modelos de texto: eso lo hace el generador de transformers
    # desde un prompt con `campo: Text`— pero llamarlas «identificador» era
    # falso, y era justo la columna que le importaba a quien subió el CSV.
    free_text_columns = [
        col for col in analysis["column_order"]
        if col != target_column and columns[col].get("looks_like_free_text")
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
            recomputed = _numeric_range_from_raw_values(
                [row.get(col) for row in rows], tokens_de_ausencia)
            if recomputed is None:
                continue
            numeric_kind, (rng_lo, rng_hi) = recomputed
            columns[col]["type"] = numeric_kind
            columns[col]["observed_range"] = [rng_lo, rng_hi]
            columns[col]["proposed_range"] = [rng_lo, rng_hi]
            reconsidered_columns.append(col)
        feature_columns = [
            col for col in analysis["column_order"]
            if col != target_column
            and columns[col]["type"] not in _NOT_YET_USABLE_FEATURE_TYPES
            and col not in dropped_constant_columns
        ]
    if not feature_columns:
        # CONTRATO 62 C2: si lo que dejó el proyecto sin features fueron las
        # constantes, hay que decirlo Y decir cómo conservarlas — el mensaje
        # genérico de abajo mandaría al usuario a buscar identificadores o
        # fechas que no existen.
        textos = _SIN_FEATURES.get(str(locale or "es").strip().lower(), _SIN_FEATURES["es"])
        if dropped_constant_columns:
            raise DatasetProjectError(textos["constantes"](sorted(dropped_constant_columns)))
        if free_text_columns:
            # Este CSV no está vacío de información: trae TEXTO. Mandar a su
            # dueño a buscar identificadores y fechas que no existen es la
            # media verdad tranquilizadora de siempre — y encima el producto
            # SÍ sabe hacer este modelo, solo que por la otra puerta.
            raise DatasetProjectError(textos["texto"](sorted(free_text_columns)))
        raise DatasetProjectError(textos["nada"])

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
    target_values_raw = _distinct_non_null(rows, target_column, tokens_de_ausencia)
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
        observed = _distinct_non_null(rows, col, tokens_de_ausencia)
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

    # ------------------------------------------------------------------
    # LOS FALTANTES, ANTES DE SINTETIZAR EL PROMPT (hueco de cableado nº 15).
    #
    # Hasta aquí, un hueco viajaba al CSV preparado tal cual —celda vacía— y
    # el modelo recién generado rechazaba su propio CSV («DATASET row N field
    # X is empty»), así que el dataset entero se perdía. `preparacion.py`
    # (103-C3) ya tenía la política del núcleo y este camino no la usaba.
    #
    # La imputación AFIRMA algo que el dato no decía, así que se hace con lo
    # que el núcleo decidió y se DECLARA:
    #   · numérica -> MEDIANA (no media: no la mueve un extremo) más un
    #     indicador `{columna}__faltante`. Sin el indicador la imputación
    #     miente en silencio: en una columna de 0 a 10, rellenar con la
    #     mediana sin marcarlo hace indistinguible «valía eso» de «no había
    #     valor» — el mismo fallo que el motor lineal cometió rellenando con
    #     cero, que en esa columna equivale a decir «era el mínimo».
    #   · categórica -> `__faltante__` como CATEGORÍA propia, el tercer
    #     estado que el núcleo exige no colapsar. Lo de antes sí lo
    #     colapsaba: one-hot todo a cero (indistinguible de una fila rota) o,
    #     en la rama de embedding, un índice inexistente escrito como "".
    #
    # Y SOLO DONDE HAY HUECOS: una columna sin faltantes no gana indicador ni
    # categoría, así que el CSV de un dataset limpio sale byte a byte igual
    # que antes de esto.
    missing_policy = _ajustar_politica_de_faltantes(
        rows, feature_columns, columns, feature_safe_names, target_column,
        tokens_de_ausencia,
    )
    for col in feature_columns:
        if columns[col]["type"] != "categorical":
            continue
        valores_reales = category_vocabularies.get(col) or _distinct_non_null(
            rows, col, tokens_de_ausencia)
        # AQUÍ IBA UNA GUARDA `len(valores_reales) < 2` PARA QUE EL CENTINELA NO
        # RESCATARA UNA COLUMNA QUE SE IBA A CAER. Se quitó porque NO PODÍA
        # PASAR: una categórica con un solo valor es `constant` y se excluye de
        # `feature_columns` antes de llegar aquí, y conservarla exige un
        # `column_category_overrides` de dos valores o más (la comprobación de
        # arriba lo rechaza si no). Se midió con los dos casos —columna de un
        # valor, y columna entera vacía forzada a `categorical` con un
        # override— y ninguno llega. Su sabotaje salía VERDE, que es justo
        # cómo se descubrió: una línea que dice impedir algo imposible cuenta
        # una historia falsa, y el banco de pruebas no tenía dientes para ella.
        # Lo que sí queda, con su prueba: una categórica constante con huecos
        # se sigue excluyendo, no la rescata el faltante.
        #
        # Los huecos que CUENTAN son los de las filas que se van a escribir:
        # una fila sin objetivo se descarta, y su hueco no puede añadir al
        # modelo una columna que luego nadie marca (una constante a 0 que
        # además afirmaría que este dataset tiene faltantes donde no los
        # tiene). Mismo criterio que usa la política numérica.
        if not any(_is_null(row.get(col), tokens_de_ausencia) for row in rows
                   if not _is_null(row.get(target_column), tokens_de_ausencia)):
            continue
        if CATEGORIA_FALTANTE in valores_reales:
            raise DatasetProjectError(
                f"La columna categórica {col!r} tiene valores ausentes Y un valor "
                f"literal {CATEGORIA_FALTANTE!r}, que es el nombre reservado para "
                "marcarlos. No se pueden distinguir: renombra ese valor en el CSV "
                "de origen antes de generar el modelo."
            )
        category_vocabularies[col] = [*valores_reales, CATEGORIA_FALTANTE]

    _reservar_codigo_de_desconocida(
        rows, feature_columns, columns, category_vocabularies, tokens_de_ausencia)

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
            values = category_vocabularies.get(col) or _distinct_non_null(
                rows, col, tokens_de_ausencia)
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
    # EL INDICADOR TIENE QUE VIAJAR HASTA EL MODELO, no solo hasta el CSV: si
    # no se declara aquí, el VECTOR generado no lo tiene y la validación
    # rechaza el CSV por traer una columna de más — imputar sin indicador
    # sería, además, rellenar en silencio. `Boolean` porque eso es: ¿faltaba?
    # sí/no (el generador lo materializa como un Scalar 0/1, medido).
    # Al FINAL de las features y en el mismo orden que `_prepare_training_csv`
    # escribe la cabecera.
    if missing_policy is not None:
        for indicador in _nombres_de_indicador(missing_policy).values():
            feature_lines.append(f"  {indicador}: Boolean")

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
                        category_vocabularies.get(col)
                        or _distinct_non_null(rows, col, tokens_de_ausencia)
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
        # El IDIOMA de los avisos del pipeline. No llegaba: esta rama los
        # pedía sin `locale`, así que un CSV subido con la aplicación en
        # inglés devolvía «El dataset (120 filas) es pequeño para 2
        # entradas…» en español. El dato ya existía en la otra puerta
        # (`/api/analyze` lo pide desde 2026-08-09) y aquí no se usaba.
        "locale": locale,
        # CONTRATO 64 C2 — las filas que de verdad van a ENTRENAR.
        #
        # REAUDITORÍA [MEDIA]: aquí iba `len(rows)`, que cuenta también las filas
        # sin target — y esas las descarta `_prepare_training_csv` más abajo. Con
        # 100 filas de las que 90 tienen el target vacío, el presupuesto se
        # calculaba sobre 100 y no sobre las 10 entrenables: además de falsear la
        # auditoría, podía elegir una red más ancha de la que los datos sostienen.
        # Se usa el MISMO criterio (`_is_null`) que la preparación real.
        "dataset_rows": sum(1 for r in rows
                            if not _is_null(r.get(target_column), tokens_de_ausencia)),
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
        # CONTRATO 64 C1 (reauditoría [ALTO]): un tope superado NO es "el prompt
        # sintetizado no generó un modelo válido" — envolverlo con esa frase
        # esconde la única acción que lo resuelve (subir el tope o el perfil).
        # Mismo criterio que el contrato 62 C1 aplicó a `max_rows` más abajo.
        if _limits.is_limit_error(res):
            _details = {k: v for k, v in res.items()
                        if k in ("error_kind", "limit_key", "unit", "actual",
                                 "maximum", "profile", "configurable")}
            raise DatasetProjectError(str(res.get("error")), details=_details)
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
        # LO QUE EL MODELO PIDE DE VERDAD. `res["mxai"]` ya es el modelo final
        # (VECTOR expandido incluido): qué categóricas consume como índice de
        # EMBEDDING se LEE de él, en vez de volver a deducirlo por cardinalidad
        # — que es lo que hacía divergir el CSV del modelo en cuanto había una
        # categórica por encima de `_ONEHOT_MAX` junto a otra por debajo.
        embedding_sources=embedding_source_columns(res.get("mxai") or ""),
        missing_policy=missing_policy,
        tokens_de_ausencia=tokens_de_ausencia,
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
        # CONTRATO 62 C1: un tope superado NO es un hueco de preparación. Antes
        # se envolvía con la frase de abajo, que afirma justo lo contrario de la
        # verdad y esconde la acción que lo resuelve (subir el perfil). Ahora se
        # propaga tal cual, con su payload estructurado.
        if _limits.is_limit_error(validation):
            _details = {k: v for k, v in validation.items() if k != "ok"}
            raise DatasetProjectError(str(validation.get("error")), details=_details)
        raise DatasetProjectError(
            "El CSV preparado no pasa la validación del modelo que acaba de "
            "generarse (esto indica un hueco en la preparación del CSV, no un "
            f"problema de tus datos): {validation.get('errors') or validation.get('error')}"
        )

    excluded_columns = [
        col for col in analysis["column_order"]
        if col != target_column and (
            columns[col]["type"] in _NOT_YET_USABLE_FEATURE_TYPES
            or col in dropped_constant_columns
        )
    ]
    # CONTRATO 62 C2 — motivo ESTRUCTURADO de cada exclusión, y si fue
    # automática o decisión del usuario. Antes `excluded_columns` era una lista
    # de nombres a secas: no se podía saber por qué faltaba una columna, ni
    # reproducir la decisión al re-preparar (C3). `keep_constant_columns`
    # queda registrado por el mismo motivo.
    excluded_column_reasons = {
        col: {
            "reason": ("constant_feature" if col in dropped_constant_columns
                       else f"unusable_type:{columns[col]['type']}"),
            "automatic": True,
            # El motivo REAL cuando el tipo dice `identifier` porque cada fila
            # es distinta, pero lo que hay dentro es prosa. La clave se añade
            # solo cuando es cierto: `reason` no cambia —quien lo lea seguirá
            # viendo el tipo con el que se excluyó— pero deja de ser toda la
            # historia. Un `False` aquí sería afirmar algo sobre columnas que
            # ni se han mirado con este criterio.
            **({"looks_like_free_text": True} if col in free_text_columns else {}),
        }
        for col in excluded_columns
    }
    # El aviso VISIBLE, por el mismo canal que el resto de avisos de
    # generación (la etapa del generador) — `excluded_columns` ya lo
    # registraba para quien programa contra la API, pero eso no es
    # decírselo a quien acaba de subir su CSV.
    _texto_excluido = [c for c in free_text_columns if c in excluded_columns]
    if _texto_excluido:
        from matrixai.playground import _anotar_avisos  # noqa: PLC0415
        _marco = _AVISO_COLUMNA_TEXTO.get(
            str(locale or "es").strip().lower(), _AVISO_COLUMNA_TEXTO["es"])
        _anotar_avisos(res, [_marco(", ".join(repr(c) for c in _texto_excluido),
                                   _texto_excluido[0])])

    # CONTRATO 62 C3 — la RECETA: todo lo que `_prepare_training_csv` necesita
    # para producir exactamente este mismo CSV preparado a partir del crudo.
    # Es la diferencia entre "describir" la preparación (lo que hacía
    # `operations`, cadenas legibles) y poder RE-EJECUTARLA.
    preparation_spec = {
        "version": PREPARATION_SPEC_VERSION,
        "feature_columns": list(feature_columns),
        "feature_name_map": dict(feature_safe_names),
        "column_types": {col: columns[col]["type"] for col in feature_columns},
        "category_vocabularies": dict(prepared.effective_vocabularies),
        # CONGELADO, no recalculable: qué categóricas fueron a índice de
        # embedding lo decidió el generador que produjo ESTE modelo. Una receta
        # SIN esta clave es anterior al arreglo y se re-prepara con el criterio
        # de entonces (`_prepare_v1`); una lista VACÍA significa "ninguna", que
        # no es lo mismo que "no se sabe".
        "embedding_columns": list(prepared.embedding_source_names),
        # CÓMO SE LEYÓ LA AUSENCIA, congelado. Sin esto la receta MENTIRÍA por
        # omisión: re-preparar volvería a aplicar la heurística `_NULL_TOKENS`
        # sobre un CSV que se preparó con una declaración, y un nivel legítimo
        # llamado «None» pasaría a ausente en la re-preparación aunque no lo
        # fuera al generar — el CSV re-preparado dejaría de ser el que entrenó
        # este modelo. Se OMITE la clave cuando no se declaró nada (AUSENTE no
        # es VACÍA, igual que `embedding_columns`): ausente pide la heurística
        # de entonces, y una lista vacía dice «declarado: aquí no falta nada».
        **({"tokens_de_ausencia": sorted(tokens_de_ausencia)}
           if tokens_de_ausencia is not None else {}),
        # LA POLÍTICA DE FALTANTES, CONGELADA. La mediana se ajustó sobre ESTAS
        # filas: re-preparar recalculándola daría otro número en cuanto el CSV
        # cambiara una fila, y el dataset dejaría de ser el que entrenó el
        # modelo. Es el mismo motivo por el que `preparacion.py` separa ajustar
        # de transformar — transformar NUNCA vuelve a mirar los datos.
        # `None` aquí significa «este dataset no tenía faltantes numéricos», y
        # se comporta igual que la ausencia de la clave en una receta anterior
        # a este arreglo: en ninguno de los dos casos se imputa nada. (La
        # asimetría con `embedding_columns` es a propósito: allí ausente y
        # vacía piden criterios DISTINTOS; aquí piden el mismo.) Lo que sí
        # distingue a una receta vieja es su `category_vocabularies`, que no
        # trae `__faltante__` y por eso reproduce el CSV de entonces.
        # SIN `limites`, y la clave se OMITE en vez de escribirla vacía: una
        # lista vacía afirmaría «esta preparación no encontró nada que
        # declarar», que sería falso. Los `Limite` son HALLAZGOS, no algo que
        # `transformar_fila` necesite para reproducir nada, y viajan una sola
        # vez en `provenance["missing_values"]["limits"]` — repetirlos aquí
        # costaba otros 88 KB en `KDDCup09_appetency` (medido). `desde_json`
        # ya trata la clave como opcional.
        "missing_policy": ({k: v for k, v in missing_policy.a_json().items()
                            if k != "limites"}
                           if missing_policy is not None else None),
        "target_column": target_column,
        "target_header": target_header,
        "task": task,
        "target_label_map": target_label_map,
        "kept_constant_columns": list(kept_constant_columns),
    }

    provenance = _build_provenance(
        preparation_spec=preparation_spec,
        csv_text=csv_text,
        prepared_csv=prepared_csv,
        schema_inferred=schema_inferred,
        schema_final=columns,
        target_column=target_column,
        excluded_columns=excluded_columns,
        excluded_column_reasons=excluded_column_reasons,
        kept_constant_columns=kept_constant_columns,
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
        missing_values=_declaracion_de_faltantes(missing_policy, prepared),
    )

    result = dict(res)
    result["csv_text"] = prepared_csv
    result["provenance"] = provenance
    # CONTRATO 103 C1 — EL PROBLEMA, CONFIRMADO ANTES DE ENTRENAR.
    #
    # Se PUBLICA, no se impone: generar el proyecto sigue haciendo exactamente lo
    # mismo que antes de este corte, y lo que aparece al lado es qué falta por
    # confirmar para que entrenar signifique algo — la unidad de observación, la
    # clase positiva de una binaria, el tipo de tarea cuando el objetivo es
    # numérico con pocos valores distintos— y qué lo impide: un objetivo que no
    # varía en las filas que DE VERDAD entrenan (que no es lo mismo que un
    # objetivo constante en el CSV entero).
    #
    # Se le pasan el `analysis` y las `rows` YA calculados. Releerlos costaría
    # tres pasadas más sobre un CSV que puede tener cientos de miles de filas
    # (medido en este producto: 0,24 s por pasada sobre 200.000 filas).
    result["confirmacion"] = _confirmacion_del_problema(
        csv_text=csv_text, analysis=analysis, rows=rows, target_column=target_column,
        feature_columns=list(feature_columns),
        unidad_de_observacion=unidad_de_observacion, clase_positiva=clase_positiva,
        momento_de_prediccion=momento_de_prediccion, horizonte=horizonte,
        uso_previsto=uso_previsto, tokens_de_ausencia=tokens_de_ausencia)
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
    keep_constant_columns: list[str] | None = None,
    user_intent: str | None = None,
    use_intent_llm: bool = False,
    locale: str = "es",
    # CONTRATO 103 C1 — las mismas respuestas que el camino no temporal. Un
    # envoltorio que se come un parámetro deja media aplicación sin confirmar,
    # que es lo que ya pasó aquí con el idioma.
    unidad_de_observacion: str | None = None,
    clase_positiva: str | None = None,
    momento_de_prediccion: str | None = None,
    horizonte: Any | None = None,
    uso_previsto: str | None = None,
    # Mismo motivo que los parámetros del 103-C1 de arriba: un envoltorio que
    # se come un parámetro deja media aplicación sin él. Aquí, además, el
    # envoltorio ANALIZA el CSV crudo por su cuenta (`original_analysis`) antes
    # de delegar, así que sin esto los dos análisis del mismo fichero usarían
    # criterios de ausencia distintos.
    tokens_de_ausencia: set[str] | None = None,
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
    original_analysis = analyze_dataset_csv(csv_text,
                                            tokens_de_ausencia=tokens_de_ausencia)
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
        tokens_de_ausencia=tokens_de_ausencia,
        column_type_overrides=type_overrides or None,
        column_range_overrides=range_overrides or None,
        column_category_overrides=category_overrides or None,
        # CONTRATO 62 C2: el camino temporal delega la decisión sobre features
        # constantes en el generador de siempre (invariante 4, cero caminos
        # paralelos) — se enhebra tal cual.
        keep_constant_columns=keep_constant_columns,
        # Contrato 58 C4/C5 — el camino temporal NUNCA ignora la intención (ni
        # su interpretación LLM) en silencio: se enhebran tal cual
        # (normalización/validación/llamada LLM ocurren dentro de
        # `generate_project_from_dataset`, un solo sitio).
        user_intent=user_intent,
        use_intent_llm=use_intent_llm,
        # El idioma también aquí: el camino temporal es un envoltorio, y un
        # envoltorio que se come un parámetro deja media aplicación traducida.
        locale=locale,
        # CONTRATO 103 C1 — y las respuestas de la confirmación, por lo mismo.
        unidad_de_observacion=unidad_de_observacion,
        clase_positiva=clase_positiva,
        momento_de_prediccion=momento_de_prediccion,
        horizonte=horizonte,
        uso_previsto=uso_previsto,
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
# CONTRATO 62 C3 — re-preparación de un CSV para un modelo YA generado
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityReport:
    """Qué pasó al re-preparar, en datos y no en prosa."""
    ok: bool
    same_raw_csv: bool
    prepared_matches: bool | None   # None = no comprobable (crudo distinto)
    spec_version: int | None
    legacy_adapter: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "same_raw_csv": self.same_raw_csv,
            "prepared_matches": self.prepared_matches,
            "spec_version": self.spec_version,
            "legacy_adapter": self.legacy_adapter,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass
class RePreparedDataset:
    csv_text: str
    provenance: dict[str, Any]
    compatibility: CompatibilityReport


_SPEC_REQUIRED_KEYS = (
    "feature_columns", "feature_name_map", "column_types",
    "target_column", "target_header", "task",
)


def _validate_preparation_spec(spec: dict[str, Any]) -> None:
    """AUDITORÍA C3 [MEDIO-ALTO]: una receta incompleta reventaba con
    `KeyError: 'feature_columns'` y el endpoint la convertía en un 500. Un
    dato corrupto o de otra versión es un fallo FUNCIONAL del usuario/modelo,
    no un error interno del servidor: se valida la forma entera y se responde
    con un mensaje accionable."""
    faltan = [k for k in _SPEC_REQUIRED_KEYS if not spec.get(k)]
    if faltan:
        raise DatasetProjectError(
            "La receta de preparación de este modelo está incompleta "
            f"(faltan: {sorted(faltan)}), así que no se puede reproducir su "
            "dataset. Regenera el proyecto desde el CSV; el modelo guardado "
            "sigue sirviendo para inferir y exportar."
        )
    if not isinstance(spec["feature_columns"], list):
        raise DatasetProjectError("La receta de preparación tiene `feature_columns` inválido.")
    if not isinstance(spec["feature_name_map"], dict):
        raise DatasetProjectError("La receta de preparación tiene `feature_name_map` inválido.")
    if not isinstance(spec["column_types"], dict):
        raise DatasetProjectError("La receta de preparación tiene `column_types` inválido.")
    sin_nombre = [c for c in spec["feature_columns"] if c not in spec["feature_name_map"]]
    if sin_nombre:
        raise DatasetProjectError(
            f"La receta de preparación no sabe cómo llamar a {sorted(sin_nombre)} "
            "en el modelo. Regenera el proyecto desde el CSV."
        )
    vocab = spec.get("category_vocabularies")
    if vocab is not None and not isinstance(vocab, dict):
        raise DatasetProjectError("La receta de preparación tiene `category_vocabularies` inválido.")
    # Igual que arriba: una receta corrupta (editada a mano, truncada al
    # guardar) no puede salir como un KeyError dentro de
    # `PoliticaDePreparacion.desde_json` — eso en el producto es un error
    # interno sin nada accionable. `None` es válido: significa «sin faltantes
    # numéricos», lo mismo que la ausencia de la clave.
    politica = spec.get("missing_policy")
    if politica is not None and (not isinstance(politica, dict)
                                 or not isinstance(politica.get("columnas"), list)):
        raise DatasetProjectError("La receta de preparación tiene `missing_policy` inválido.")


# AUDITORÍA C3 [MEDIO-ALTO]: la versión de receta era NOMINAL — ante una
# desconocida se avisaba y se ejecutaba igual el preparador actual, que es
# precisamente lo que la versión existe para impedir. El registro explícito
# hace que una versión sin preparador NO se ejecute como si fuera compatible.
def _prepare_v1(
    rows: list[dict[str, str]], spec: dict[str, Any]
) -> _PreparedCSV:
    columns_for_prepare = {
        col: {"type": spec["column_types"].get(col, "number")}
        for col in spec["feature_columns"]
    }
    return _prepare_training_csv(
        rows,
        list(spec["feature_columns"]),
        columns_for_prepare,
        dict(spec["feature_name_map"]),
        spec["target_column"],
        spec["task"],
        spec.get("target_label_map"),
        spec["target_header"],
        dict(spec.get("category_vocabularies") or {}),
        # AUSENTE no es VACÍA: una receta anterior a esta clave no sabe qué
        # columnas fueron a embedding, y forzar `set()` convertiría en one-hot
        # las que entonces se escribieron como índice — re-preparar dejaría de
        # reproducir el CSV que ese modelo entrenó. `None` pide el criterio de
        # entonces; una lista (aunque sea vacía) es una respuesta.
        embedding_sources=(set(spec["embedding_columns"])
                           if "embedding_columns" in spec else None),
        # Ver la clave homónima en `preparation_spec`: ausente = la heurística
        # de entonces, presente (aunque sea vacía) = lo que aquel origen
        # declaró.
        tokens_de_ausencia=(set(spec["tokens_de_ausencia"])
                            if "tokens_de_ausencia" in spec else None),
        # LA MEDIANA NO SE RECALCULA: se lee de la receta. Recalcularla sobre
        # el CSV que ahora se re-prepara daría otro número en cuanto cambiara
        # una fila, y `prepare_dataset_from_provenance` abortaría —con razón—
        # por no reproducir el dataset que entrenó el modelo. Ausente y `None`
        # piden lo mismo, y es correcto: una receta anterior a esta clave
        # NUNCA pudo tener faltantes numéricos (el modelo rechazaba su propio
        # CSV y no llegaba a guardarse), así que «no lo sé» y «no había» son
        # aquí el mismo caso. Lo que SÍ distingue a una receta vieja con
        # faltantes CATEGÓRICOS es su `category_vocabularies`, que no trae
        # `__faltante__`: sin él, la re-preparación repite el criterio de
        # entonces sin que haya que preguntarle nada más.
        missing_policy=(PoliticaDePreparacion.desde_json(spec["missing_policy"])
                        if spec.get("missing_policy") else None),
    )


PREPARERS = {1: _prepare_v1}


def _spec_from_legacy_provenance(provenance: dict[str, Any]) -> dict[str, Any] | None:
    """Adaptador para una procedencia anterior a la receta versionada.

    NO inventa nada: deriva la receta de lo que esa procedencia SÍ registraba
    (`feature_name_map`, `schema_final`, `target_label_map`, `task`…). El
    vocabulario categórico es el único hueco real —nunca se guardó— y se
    reconstruye del esquema (`vocabulary`) cuando está; si no, se deja fuera y
    se recalculará del CSV. Por eso el adaptador **solo vale si el hash del
    preparado coincide**: esa es su demostración de compatibilidad, no una
    promesa (invariante 8 del contrato).
    """
    feature_map = provenance.get("feature_name_map")
    schema = provenance.get("schema_final") or provenance.get("schema_inferred")
    task = provenance.get("task")
    target_column = provenance.get("target_column")
    if not feature_map or not schema or not task or not target_column:
        return None
    feature_columns = list(feature_map.keys())
    # El vocabulario NO se toma del esquema: `analyze_dataset_csv` lo guarda
    # ORDENADO alfabéticamente, mientras que la preparación real usaba
    # `_distinct_non_null` (orden de aparición en el CSV) para las categóricas
    # sin override. Derivarlo del esquema cambiaba el orden de las columnas
    # one-hot y el preparado dejaba de coincidir — lo cazó el propio chequeo de
    # hash. Lo único autoritativo que la procedencia legacy sí guarda son los
    # overrides explícitos del usuario; el resto se recalcula igual que
    # entonces, y la coincidencia del hash es la prueba de que salió bien.
    vocabularies = {
        col: list(values)
        for col, values in (provenance.get("column_category_overrides") or {}).items()
        if col in feature_columns
    }
    return {
        "version": None,  # legacy: sin versión declarada
        "feature_columns": feature_columns,
        "feature_name_map": dict(feature_map),
        "column_types": {
            col: schema[col]["type"] for col in feature_columns if col in schema
        },
        "category_vocabularies": vocabularies,
        "target_column": target_column,
        "target_header": ("predicted_class" if task == "classification"
                          else "predicted_value"),
        "task": task,
        "target_label_map": provenance.get("target_label_map"),
        "kept_constant_columns": list(provenance.get("kept_constant_columns") or []),
    }


def _replay_temporal_pipeline(
    rows: list[dict[str, str]], temporal: dict[str, Any]
) -> list[dict[str, str]]:
    """Reconstruye el CSV post-pipeline de un modelo temporal desde el bloque
    ESTRUCTURADO de la procedencia — nunca desde las cadenas de `operations`,
    que son descriptivas (invariante 2 del contrato).

    `pipeline_operations` guarda cada paso como `{"operation": ..., "params":
    {...}, ...}`; `run_pipeline` espera `{"op": ..., **params}`. La conversión
    es exacta porque `params` es literalmente el declarativo de entrada sin su
    clave `op` (ver `run_pipeline`).
    """
    from matrixai.training.dataset_pipeline import PipelineError, run_pipeline

    steps = temporal.get("pipeline_operations") or []
    if not steps:
        raise DatasetProjectError(
            "El modelo es de serie temporal pero su procedencia no guarda las "
            "operaciones del pipeline: no se puede reconstruir el dataset. "
            "Regenera el proyecto desde el CSV."
        )
    ops = [{"op": step["operation"], **(step.get("params") or {})} for step in steps]
    try:
        return run_pipeline(rows, ops).rows
    except PipelineError as exc:
        raise DatasetProjectError(f"Serie temporal (reconstrucción): {exc}") from exc


def prepare_dataset_from_provenance(
    raw_csv: str,
    provenance: dict[str, Any],
    *,
    expected_columns: list[str] | None = None,
    allow_incompatible_spec: bool = False,
) -> RePreparedDataset:
    """Prepara un CSV crudo EXACTAMENTE como se preparó al generar el modelo.

    Es la única vía correcta para reimportar el dataset de un modelo
    desde-datos (invariante 2 del contrato 62): reutiliza la implementación
    real de preparación con la receta congelada en la procedencia, en vez de
    imitarla reescribiendo texto —lo que hacía el parche de v1.4, que solo
    cubría target escalar/binario y se rompía con categóricas o temporal.

    NO regenera arquitectura ni entrenamiento: devuelve CSV preparado,
    procedencia nueva y un informe de compatibilidad estructurado.

    Política de verificación (invariante 4):
      - mismo crudo + misma versión de receta + hash preparado distinto → es un
        bug, se aborta;
      - mismo crudo + versión DISTINTA (o receta legacy) → no se aborta: se
        avisa nombrando lo que cambia y el caller decide;
      - crudo distinto → no se compara contra el hash antiguo; se comprueba
        compatibilidad de esquema y se emite procedencia nueva referenciando la
        anterior.
    """
    if not isinstance(provenance, dict):
        raise DatasetProjectError("La procedencia debe ser un objeto.")

    raw_sha = _sha256_text(raw_csv)
    same_raw = raw_sha == provenance.get("raw_csv_sha256")

    spec = provenance.get("preparation_spec") or None
    if spec is not None and not isinstance(spec, dict):
        raise DatasetProjectError("La receta de preparación de este modelo es inválida.")
    legacy = not spec
    if legacy:
        spec = _spec_from_legacy_provenance(provenance)
        if spec is None:
            raise DatasetProjectError(
                "La procedencia de este modelo no tiene receta de preparación ni "
                "los datos mínimos para deducirla. Regenera el proyecto desde el "
                "CSV; el modelo guardado sigue sirviendo para inferir y exportar."
            )

    _validate_preparation_spec(spec)

    warnings: list[str] = []
    errors: list[str] = []
    spec_version = spec.get("version")

    # La receta legacy no declara versión: se prepara con la v1, que es la que
    # existía cuando se guardó (y el hash lo demuestra). Una versión declarada
    # SIN preparador registrado no se ejecuta con el actual: sería justo lo que
    # el versionado viene a impedir.
    preparer = PREPARERS.get(1) if legacy else PREPARERS.get(spec_version)
    if preparer is None:
        # Conjuga las dos exigencias: el contrato pide no dejar inservible un
        # modelo por una actualización (invariante 4) y la auditoría pide no
        # ejecutar una versión desconocida COMO SI fuera compatible. Por
        # defecto se aborta con un mensaje accionable; con confirmación
        # explícita del caller se usa el preparador actual y el aviso queda
        # registrado en el informe.
        if not allow_incompatible_spec:
            raise DatasetProjectError(
                f"Este modelo se preparó con la receta v{spec_version}, que esta "
                f"versión de MatrixAI no sabe reproducir (conoce: "
                f"{sorted(PREPARERS)}). Actualiza MatrixAI, o confirma "
                "explícitamente que quieres prepararlo con la receta actual "
                "asumiendo que el resultado puede no ser equivalente. El modelo "
                "guardado sigue sirviendo para inferir y exportar.",
                details={"error_kind": "incompatible_preparation_spec",
                         "spec_version": spec_version,
                         "known_versions": sorted(PREPARERS)},
            )
        preparer = PREPARERS[PREPARATION_SPEC_VERSION]
        warnings.append(
            f"Este modelo se preparó con la receta v{spec_version}, desconocida "
            f"para esta versión: se ha usado la v{PREPARATION_SPEC_VERSION} bajo "
            "tu confirmación. El resultado puede no ser equivalente."
        )
    if legacy:
        warnings.append(
            "Este modelo se guardó antes de que la preparación fuese "
            "reproducible: se ha deducido su receta y solo se puede confiar en "
            "ella si el CSV preparado sale idéntico."
        )
    elif spec_version != PREPARATION_SPEC_VERSION:
        warnings.append(
            f"El modelo se preparó con la receta v{spec_version} y esta versión "
            f"usa la v{PREPARATION_SPEC_VERSION}: se reproduce con el preparador "
            f"v{spec_version} registrado, pero revisa el resultado antes de "
            "reentrenar."
        )

    rows = _read_rows(raw_csv)
    if not rows:
        raise DatasetProjectError("El CSV no tiene filas de datos.")

    temporal = provenance.get("temporal")
    if temporal:
        rows = _replay_temporal_pipeline(rows, temporal)
        if not rows:
            raise DatasetProjectError(
                "Tras reconstruir la serie temporal no queda ninguna fila."
            )

    present = set(rows[0].keys())
    feature_columns = list(spec["feature_columns"])
    target_column = spec["target_column"]

    # Política de compatibilidad del contrato.
    missing = [c for c in feature_columns if c not in present]
    if target_column not in present:
        missing.append(target_column)
    if missing:
        errors.append(
            f"Al CSV le faltan columnas que este modelo necesita: {sorted(missing)}."
        )
    extra = sorted(present - set(feature_columns) - {target_column})
    if extra:
        warnings.append(
            f"El CSV trae columnas que este modelo no usa y se ignoran: {extra}."
        )

    # Vocabulario congelado: un valor nuevo cambiaría el VECTOR (una columna
    # one-hot más), así que NUNCA se amplía en silencio.
    #
    # PERO `__desconocida__` NO ES UN VALOR NUEVO DEL USUARIO: es la marca que
    # el propio núcleo (`preparacion.transformar_fila`, 103-C3) pone donde
    # había un valor presente que el entrenamiento no vio. Rechazarla era el
    # núcleo sin reconocer su propio vocabulario, y costaba el dataset entero
    # —11 de los 15 pliegues de `Moneyball`, medido el 2026-09-14, por 1 a 3
    # filas de 246—. Se codifica como «ninguna de las conocidas» (el grupo
    # one-hot a 0) y se DECLARA, con su recuento, en vez de abortar.
    #
    # En EMBEDDING no hay «ninguna de las conocidas»: el CSV lleva el índice
    # del valor. Si la receta congelada no reservó el código (un modelo
    # anterior a `_reservar_codigo_de_desconocida`), la fila no se puede
    # representar y se sigue abortando — pero diciendo QUÉ falta, no
    # llamándolo «un valor que el modelo no conoce».
    #
    # AUSENTE no es VACÍA, igual que en `_prepare_v1`: una receta anterior a
    # `embedding_columns` no sabe qué columnas fueron a embedding, y el
    # preparador aplica entonces el criterio de aquel momento
    # (`len(vocab) > _ONEHOT_MAX`). Aquí se pregunta lo mismo, o el aviso
    # describiría una codificación que no es la que se va a escribir.
    embebidas_declaradas = (set(spec["embedding_columns"])
                            if "embedding_columns" in spec else None)
    # El MISMO criterio de ausencia con el que se va a re-preparar (`_prepare_
    # v1` lo lee de la misma clave). Con la heurística aquí y la declaración
    # allí, este informe contaría como «valores que el modelo no conoce» justo
    # los niveles que el preparador sí va a escribir.
    tokens_de_ausencia = (set(spec["tokens_de_ausencia"])
                          if "tokens_de_ausencia" in spec else None)

    def _va_por_embedding(col: str, vocab: list[str]) -> bool:
        if embebidas_declaradas is None:
            return len(vocab) > _ONEHOT_MAX
        return (spec.get("feature_name_map") or {}).get(col) in embebidas_declaradas

    # El recuento se lleva SIEMPRE, no solo cuando el vocabulario no trae el
    # código. En un modelo de embedding sí lo trae —lo reservó al generarse—,
    # así que el centinela no aparecería como «nuevo» y las filas que el
    # modelo no puede situar se colarían sin que nadie las contara. Salvar el
    # dataset y no decir cuántas filas se salvaron a ciegas sería media
    # verdad tranquilizadora.
    filas_desconocidas: dict[str, tuple[int, bool]] = {}
    for col, vocab in (spec.get("category_vocabularies") or {}).items():
        if col not in present:
            continue
        observed = _distinct_non_null(rows, col, tokens_de_ausencia)
        nuevos = [v for v in observed if v not in vocab]
        afectadas = sum(1 for row in rows
                        if (row.get(col) or "").strip() == CATEGORIA_DESCONOCIDA)
        por_embedding = _va_por_embedding(col, list(vocab))
        if afectadas:
            nuevos = [v for v in nuevos if v != CATEGORIA_DESCONOCIDA]
            if por_embedding and CATEGORIA_DESCONOCIDA not in vocab:
                errors.append(
                    f"La columna {col!r} trae categorías que el entrenamiento "
                    f"no vio ({afectadas} filas), y este modelo la consume "
                    "como EMBEDDING: su vocabulario no reservó un código "
                    f"{CATEGORIA_DESCONOCIDA!r} y un índice inexistente no se "
                    "puede escribir. Regenera el proyecto para que lo reserve."
                )
            else:
                filas_desconocidas[col] = (afectadas, por_embedding)
        if nuevos:
            errors.append(
                f"La columna {col!r} trae valores que el modelo no conoce: "
                f"{sorted(nuevos)}. El vocabulario quedó fijado al generar el "
                f"modelo ({sorted(vocab)}); para incorporarlos hay que regenerar "
                "el proyecto."
            )
    for col, (filas_afectadas, por_embedding) in sorted(filas_desconocidas.items()):
        como = ("el código reservado del vocabulario, cuyo vector no se "
                "entrenó con ningún ejemplo" if por_embedding else
                "«ninguna de las conocidas» (todo el grupo a 0), no como la "
                "categoría de referencia ni como un dato ausente")
        warnings.append(
            f"La columna {col!r} trae en {filas_afectadas} filas una categoría "
            f"que el entrenamiento nunca vio: se codifica como {como}."
        )

    if errors:
        report = CompatibilityReport(
            ok=False, same_raw_csv=same_raw, prepared_matches=None,
            spec_version=spec_version, legacy_adapter=legacy,
            warnings=warnings, errors=errors,
        )
        raise DatasetProjectError(" ".join(errors), details={"compatibility": report.to_dict()})

    prepared = preparer(rows, spec)

    prepared_sha = _sha256_text(prepared.text)
    prepared_matches: bool | None = None
    if same_raw:
        prepared_matches = prepared_sha == provenance.get("prepared_csv_sha256")
        if not prepared_matches:
            if not legacy and spec_version == PREPARATION_SPEC_VERSION:
                # Mismo crudo, misma receta, resultado distinto: no es un caso
                # de versionado, es un fallo de reproducibilidad.
                raise DatasetProjectError(
                    "La re-preparación del MISMO CSV no reproduce el dataset con "
                    "el que se entrenó este modelo. Es un fallo de "
                    "reproducibilidad, no un problema de tus datos: no se "
                    "reentrena con un dataset que no es el esperado."
                )
            warnings.append(
                "El CSV preparado no coincide byte a byte con el original "
                "(receta distinta o deducida): revisa el resultado antes de "
                "reentrenar."
            )

    if expected_columns is not None:
        header = prepared.text.split("\n", 1)[0].strip()
        got = [c.strip() for c in header.split(",")] if header else []
        if got != list(expected_columns):
            raise DatasetProjectError(
                "El CSV preparado no encaja con el modelo cargado. Esperaba las "
                f"columnas {list(expected_columns)} y salieron {got}."
            )

    new_provenance = dict(provenance)
    new_provenance["raw_csv_sha256"] = raw_sha
    new_provenance["prepared_csv_sha256"] = prepared_sha
    new_provenance["rows_dropped_null_target"] = prepared.rows_dropped_null_target
    new_provenance["reprepared_at"] = datetime.now(timezone.utc).isoformat()
    # REAUDITORÍA [ALTO]: la procedencia se copiaba y solo se cambiaban los
    # hashes, así que `schema_inferred` (rangos, cardinalidades, fechas) seguía
    # describiendo el CSV PADRE — para un CSV nuevo con rango [10,15] la
    # procedencia seguía diciendo [0,5]. No se debe SOBRESCRIBIR el esquema
    # contractual del modelo (es el congelado, y re-preparar no lo cambia:
    # invariante 3), pero sí hay que registrar lo OBSERVADO ahora, separado.
    try:
        new_provenance["observed_schema"] = analyze_dataset_csv(
            _rows_to_csv_text(rows),
            # El MISMO criterio con el que se acaba de re-preparar. Con la
            # heurística aquí, el esquema OBSERVADO que se registra describiría
            # un CSV distinto del que se escribió dos pasos más arriba.
            tokens_de_ausencia=tokens_de_ausencia)["columns"]
    except Exception:  # noqa: BLE001
        # Nunca bloquea la re-preparación: es trazabilidad, no corrección.
        new_provenance["observed_schema"] = None
    if not same_raw:
        # Crudo distinto: procedencia NUEVA que apunta a la anterior, nunca una
        # procedencia vieja que finge describir datos que ya no son los suyos.
        new_provenance["parent_provenance_sha256"] = _sha256_text(
            json.dumps(provenance, sort_keys=True, default=str))
        warnings.append(
            "El CSV no es el mismo con el que se generó el modelo: se ha "
            "comprobado que encaja con su esquema y se registra como origen nuevo."
        )

    report = CompatibilityReport(
        ok=True, same_raw_csv=same_raw, prepared_matches=prepared_matches,
        spec_version=spec_version, legacy_adapter=legacy,
        warnings=warnings, errors=[],
    )
    return RePreparedDataset(
        csv_text=prepared.text, provenance=new_provenance, compatibility=report)


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


def _distinct_non_null(rows: list[dict[str, str]], col: str,
                       tokens_de_ausencia: set[str] | None = None) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        value = row.get(col)
        if _is_null(value, tokens_de_ausencia):
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
    tokens_de_ausencia: set[str] | None = None,
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
    non_null = [v.strip() for v in raw_values if not _is_null(v, tokens_de_ausencia)]
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
            # EMPEZAR POR DÍGITO NO ES «NO TENER NOMBRE», y el mensaje de
            # abajo lo decía igual: un identificador no puede empezar por
            # número, así que `_identifier` devuelve vacío para `1stFlrSF` —
            # que tiene ocho letras dentro. Se rescata con el prefijo
            # `campo_`, exactamente como `_normalize_labels` rescata con
            # `class_` un valor que empieza por dígito, y por el mismo motivo:
            # no es un dato ambiguo, es la normalización tirándolo.
            #
            # Medido el 2026-09-14 sobre los 40 datasets del protocolo de
            # Fase 0: el único afectado es `house_prices_nominal`
            # (`1stFlrSF`, `2ndFlrSF`, `3SsnPorch`), y perdía **los 15
            # pliegues**, con los 7 motores, antes de entrenar nada. Nada de
            # lo que hoy funciona cambia de nombre: este camino solo se toma
            # donde antes se levantaba una excepción.
            slug = _slug(col)
            safe = f"campo_{slug}" if slug else ""
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


#: Los signos RELACIONALES, con el nombre que los sustituye. Son los únicos
#: caracteres no alfanuméricos que este módulo NOMBRA en vez de borrar, y la
#: lista no es de gusto: son los que **distinguen un valor de otro** en una
#: etiqueta de clase (`<=50K` frente a `>50K`, `<18` frente a `>=18`), mientras
#: que un espacio, un punto o un guion dentro de una palabra son SEPARADORES y
#: tienen que seguir dando `_` (ver `test_un_guion_EN_MEDIO_no_es_un_signo`).
#:
#: Las formas de dos caracteres van PRIMERO: con `<` antes que `<=`, el `<=`
#: de `<=50K` se comería por la izquierda y daría `lt_eq_50k` — que no está
#: mal, pero deja de coincidir con lo que escribe quien lee `le`.
_SIGNOS_RELACIONALES: tuple[tuple[str, str], ...] = (
    ("<=", "le"), (">=", "ge"), ("<>", "ne"), ("!=", "ne"),
    ("<", "lt"), (">", "gt"), ("=", "eq"),
)


def _nombrar_signos_relacionales(raw: str) -> str:
    """Sustituye cada signo relacional por su nombre, EN SU SITIO.

    Va en su propia función porque hacen falta los DOS caminos de
    `_normalize_labels`: `<=50K` cae en `_slug` (su contenido alfanumérico
    empieza por dígito) pero `a<b` lo resuelve `_identifier` sin pasar por
    ahí, y sin esto `a<b` y `a>b` seguirían dando los dos `a_b`. Una sola
    tabla y un solo sitio que la aplica: copiarla al otro camino es
    exactamente lo que acaba divergiendo.

    Es IDEMPOTENTE —lo que sale no tiene ya ningún relacional—, así que
    aplicarla dos veces (una aquí y otra dentro de `_slug`) no cambia nada.
    """
    for signo, nombre in _SIGNOS_RELACIONALES:
        raw = raw.replace(signo, f"_{nombre}_")
    return raw


def _reservar_codigo_de_desconocida(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    columns: dict[str, dict[str, Any]],
    category_vocabularies: dict[str, list[str]],
    tokens_de_ausencia: set[str] | None = None,
) -> None:
    """Reserva `__desconocida__` en el vocabulario **solo si el modelo va a ir
    por EMBEDDING**, que son los únicos que no pueden representarla de otra
    forma. Modifica `category_vocabularies` en sitio.

    EL PROBLEMA. `preparacion.transformar_fila` (103-C3) marca con
    `CATEGORIA_DESCONOCIDA` un valor PRESENTE que el train de ese pliegue no
    vio — el tercer estado que el núcleo declara no colapsar. Ese centinela
    llega luego a `prepare_dataset_from_provenance` como si fuera un valor
    nuevo del usuario, y el vocabulario congelado no lo trae: el dataset
    ENTERO se perdía. Medido el 2026-09-14 sobre `Moneyball` (columna `Team`,
    39 categorías, ~788 filas de train): **11 de los 15 pliegues** caídos, con
    1 a 3 filas afectadas de 246 de test. Los 4 que pasaban lo hacían por
    CASUALIDAD —la validación de ese pliegue traía el centinela, así que
    entraba al vocabulario como una categoría más—, que es peor que fallar:
    el mismo dato daba un resultado u otro según qué filas cayeran en
    validación.

    POR QUÉ SOLO EMBEDDING, Y NO SIEMPRE. En one-hot, un valor que el
    vocabulario no trae ya se escribe como **el grupo entero a 0** —«ninguna
    de las conocidas»— y eso es mejor que reservarle una columna: una columna
    que vale 0 en TODAS las filas de train no recibe gradiente, así que su
    peso se queda en la inicialización y al predecir metería un peso SIN
    ENTRENAR donde el todo-a-cero no mete nada. En embedding no hay
    «todo a cero»: el CSV lleva el ÍNDICE del valor, y un valor sin índice se
    escribía `""` y el modelo rechazaba su propia fila. Ahí sí hace falta un
    código reservado, aunque su vector tampoco se entrene: la alternativa es
    perder la fila.

    POR QUÉ NO CAMBIA EL ENRUTADO, que es la trampa de tocar esto. El
    generador manda el prompt ENTERO al camino composite (embedding) en
    cuanto UNA categórica pasa de `_ONEHOT_MAX`. Aquí solo se añade el código
    cuando alguna YA lo pasa **sin contarlo**, así que añadirlo no puede
    convertir en composite un modelo que no lo fuera: la decisión de
    enrutado es exactamente la misma con y sin esta función. Un modelo
    one-hot no gana ni una columna y su CSV preparado sale byte a byte igual
    que antes de esto.

    LO QUE SÍ MUEVE. Un dataset con una categórica de más de `_ONEHOT_MAX`
    valores cambia su vocabulario, su `.mxai` y su CSV preparado. De los 40
    del protocolo de Fase 0 son 7 (medido el 2026-09-14): `splice`,
    `KDDCup09_appetency`, `adult`, `okcupid-stem`, `Moneyball`,
    `house_prices_nominal` y `Allstate_Claims_Severity`.
    """
    categoricas = [col for col in feature_columns
                   if columns[col]["type"] == "categorical"]
    vocabularios = {
        col: list(category_vocabularies.get(col)
                  or _distinct_non_null(rows, col, tokens_de_ausencia))
        for col in categoricas
    }
    # Sin contar el código que se va a añadir: ver «POR QUÉ NO CAMBIA EL
    # ENRUTADO» en el docstring.
    if not any(len(valores) > _ONEHOT_MAX for valores in vocabularios.values()):
        return
    for col, valores in vocabularios.items():
        if len(valores) < 2:
            # Cardinalidad<2 -> la columna se excluye más abajo; darle un
            # código reservado la rescataría por la puerta de atrás.
            continue
        if CATEGORIA_DESCONOCIDA in valores:
            # YA ESTABA EN LOS DATOS, Y NO ES UN CHOQUE — a diferencia de
            # `__faltante__`, que sí levanta. Quien escribe esta cadena es
            # `transformar_fila`, y lo que dice es exactamente lo que el
            # código reservado significa: aquí había una categoría que el
            # train no vio. Reordenarla al final es lo único que hace falta,
            # para que el índice del código NO dependa de en qué fila
            # apareciera — que es como los 4 pliegues «buenos» de `Moneyball`
            # lo estaban resolviendo, por casualidad y en un sitio distinto
            # cada vez. Ver `test_el_codigo_reservado_va_SIEMPRE_al_final`.
            valores = [v for v in valores if v != CATEGORIA_DESCONOCIDA]
        category_vocabularies[col] = [*valores, CATEGORIA_DESCONOCIDA]


def _slug(raw: str) -> str:
    """Como `_identifier` pero SIN el veto de dígito inicial — la base de los
    dos rescates que este módulo hace cuando `_identifier` devuelve vacío por
    empezar con número: el prefijo `class_` de una ETIQUETA
    (`_normalize_labels`) y el prefijo `campo_` de un NOMBRE DE COLUMNA
    (`_normalize_feature_names`).

    **LOS SIGNOS RELACIONALES TAMBIÉN SE CONSERVAN** (`<=50K` → `le_50k`,
    `>50K` → `gt_50k`), y por el mismo motivo que el menos. `adult` —el
    dataset de renta de OpenML, y el único de los 30 de clasificación del
    protocolo con este choque, barrido el 2026-09-14— tiene exactamente esas
    dos clases: `<` y `>` caían en «cualquier símbolo», los dos valores daban
    `class_50k` y `_normalize_labels` los declaraba indistinguibles. La densa
    perdía el dataset ENTERO, igual que perdió `PhishingWebsites` por el menos.

    **Lo que este módulo NO hace, y por qué.** No nombra el resto de la
    puntuación: un espacio, un punto o un guion dentro de una palabra son
    SEPARADORES —`alto-riesgo` tiene que seguir dando `alto_riesgo`— y no hay
    ninguna medida detrás de inventarles un nombre. Un par que se diferencie
    solo en un signo que no está en `_SIGNOS_RELACIONALES` (`a+` frente a
    `a*`) sigue chocando, y `_normalize_labels` lo sigue diciendo con un error
    accionable en vez de fundir las dos clases en silencio: ver
    `test_un_par_que_choca_por_un_signo_NO_relacional_sigue_avisando`.

    **El SIGNO se conserva** (`-1` → `neg_1`), y no es un detalle de estilo.
    Antes el menos caía en la clase de «cualquier símbolo» y se convertía en
    `_`, que `.strip("_")` remataba: `-1` y `1` daban los dos `1`, y
    `_normalize_labels` los declaraba «la misma etiqueta tras normalizar».

    Costó 15 intentos y un dataset entero, medido: en la pasada exploratoria
    del 101-C3 (2026-09-07, diagnosticado el 09-12), `PhishingWebsites` tiene
    la columna objetivo `Result` con valores `-1` y `1` — la codificación más
    común que existe para un problema binario. La red densa perdió los 15
    intentos con `DatasetProjectError`, así que en ese dataset no compitió, y
    LightGBM figuraba como mejor **sin rival**. La regla de cierre del 101-C1
    leía esa victoria como buena.

    `-1` y `1` no son un dato ambiguo que alguien deba «unificar antes de
    generar el modelo», que es lo que el error pedía: son dos valores
    perfectamente distintos que la normalización estaba fundiendo. Un mensaje
    correcto sobre un diagnóstico equivocado sigue siendo un fallo.

    `-0,5` frente a `0,5` tenía exactamente el mismo problema, por el mismo
    motivo.
    """
    text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    # Solo un signo de VERDAD al principio, no un guion en medio de una
    # palabra: «alto-riesgo» tiene que seguir dando `alto_riesgo`.
    negativo = bool(re.match(r"^\s*-\s*[0-9.,]", text))
    # Los relacionales, ANTES del barrido de «cualquier símbolo» — que es
    # justo lo que los borraba. Se sustituyen EN SU SITIO, no como prefijo:
    # `50K+` y `<=50K` no dicen lo mismo y la etiqueta tiene que enseñarlo.
    text = _nombrar_signos_relacionales(text)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return f"neg_{text}" if (negativo and text) else text


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
        # LOS SIGNOS RELACIONALES, ANTES DE LAS DOS NORMALIZACIONES. Sin esto
        # `a<b` y `a>b` dan los dos `a_b` por el camino de `_identifier`, que
        # es el que NO pasa por `_slug`. Ver `_nombrar_signos_relacionales`.
        con_signos = _nombrar_signos_relacionales(raw)
        norm = _identifier(con_signos)
        if not norm:
            slug = _slug(con_signos)
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


def _nombres_de_indicador(politica: PoliticaDePreparacion) -> dict[str, str]:
    """Los indicadores que `politica` EMITE -> cómo se llaman en este camino.

    La clave es el nombre que `transformar_fila` devuelve; el valor, el nombre
    con el que se escribe en el CSV y se declara en el modelo.

    NO SE RECONSTRUYE EL SUFIJO: se pregunta a la política
    (`columnas_de_salida()` declara exactamente lo que `transformar_fila`
    escribe, columna + indicador, deduplicado) y los indicadores son lo que
    queda al quitar las columnas fuente.

    SÍ SE SANEA, y esa es la única libertad que se toma este camino. GEN
    sanea cada nombre de FEATURE con `_sanitize_name`, que colapsa `_+` en un
    solo `_`: un campo declarado `edad__faltante` en el prompt sale del
    generador como `edad_faltante` y el CSV dejaba de casar con el modelo
    («Faltan: edad_faltante», medido). Se aplica `_identifier` —la MISMA
    normalización que ya reciben todas las demás columnas de este CSV— para
    que el nombre coincida con el del modelo por construcción y no por suerte.
    """
    fuentes = {propuesta.columna for propuesta in politica.columnas}
    return {nombre: _identifier(nombre)
            for nombre in politica.columnas_de_salida() if nombre not in fuentes}


def _valor_numerico_o_none(valor: str | None,
                           tokens_de_ausencia: set[str] | None = None) -> float | None | str:
    """El valor tipado que `ajustar_preparacion` espera, desde texto de CSV.

    `None` si el CSV dice que no hay dato (`_is_null`, la MISMA definición que
    usa el resto de este camino para tipar columnas y para descartar filas sin
    target); el `float` si parsea; y el TEXTO CRUDO si no parsea.

    Ese último caso no es teórico y por eso no se convierte en una excepción:
    con `column_type_overrides` el usuario puede declarar numérica una columna
    con basura dentro, y re-preparar un CSV distinto con la misma receta puede
    traer un valor nuevo que no parsea. Inventar ahí un número sería imputar
    algo que el dato sí decía — mal; el texto viaja tal cual y el verificador
    del modelo lo rechaza con «must be numeric», que es la verdad.
    """
    if _is_null(valor, tokens_de_ausencia):
        return None
    try:
        return float(str(valor).strip())
    except (TypeError, ValueError):
        return str(valor).strip()


#: Clave con la que el objetivo viaja a `ajustar_preparacion`: un `\0` no
#: puede salir de una cabecera de CSV ni de `_identifier`, así que no choca
#: con ninguna columna real.
_SENTINELA_OBJETIVO = "\0objetivo\0"


def _declaracion_de_faltantes(
    politica: PoliticaDePreparacion | None, prepared: _PreparedCSV,
) -> dict[str, Any]:
    """Lo que la imputación AFIRMÓ, para la procedencia.

    Se declara lo que PASÓ —celdas contadas al escribir el CSV— y no lo que la
    política pedía: `proporcion_faltante` describe las filas que se miraron al
    ajustar, no las que se acabaron escribiendo, y confundirlas sería firmar
    un número que nadie ha medido.
    """
    por_columna: dict[str, Any] = {}
    indicadores = _nombres_de_indicador(politica) if politica is not None else {}
    for nombre, celdas in sorted(prepared.imputed_numeric_cells.items()):
        propuesta = politica.columna(nombre) if politica is not None else None
        por_columna[nombre] = {
            "cells": celdas,
            # El valor CON EL QUE se rellenó — sin esto, «se imputó» no es una
            # declaración auditable, es un aviso.
            "imputed_with": propuesta.mediana if propuesta is not None else None,
            "strategy": "median",
            "indicator_column": indicadores.get(nombre_de_indicador(nombre)),
        }
    return {
        # La política NO se repite aquí: vive en `preparation_spec`, que es
        # quien la necesita para reproducir. Repetirla costaba 116 KB en
        # `KDDCup09_appetency` (medido: 231 columnas, 160 imputadas) y duplicar
        # una declaración es exactamente lo que acaba divergiendo.
        "imputed_numeric": por_columna,
        "missing_category": {
            nombre: {"cells": celdas, "category": CATEGORIA_FALTANTE}
            for nombre, celdas in sorted(prepared.missing_category_cells.items())
        },
        "limits": [limite.a_json() for limite in (politica.limites if politica else ())],
    }


def _ajustar_politica_de_faltantes(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    columns: dict[str, dict[str, Any]],
    feature_safe_names: dict[str, str],
    target_column: str,
    tokens_de_ausencia: set[str] | None = None,
) -> PoliticaDePreparacion | None:
    """Ajusta la política de faltantes DEL NÚCLEO sobre las numéricas que de
    verdad tienen huecos. `None` si no hay ninguna: un CSV sin faltantes no
    debe ganar ni una columna ni cambiar de forma.

    SOLO LAS QUE TIENEN HUECOS, y esto es deliberado. `preparacion.py` emite
    un indicador para TODA numérica porque allí la política se ajusta por
    pliegue y una columna puede faltar en test sin faltar en train. Aquí el
    CSV preparado lo ve entero y la forma del modelo queda congelada en la
    receta, así que un indicador constante a 0 solo añadiría una entrada
    inútil a la red y cambiaría el CSV de todos los datasets que hoy
    funcionan. El precio, declarado: una columna que no tuvo ningún hueco al
    preparar no tiene dónde declarar uno más tarde — una fila nueva con ese
    hueco la sigue rechazando el verificador, que es preferible a rellenarla
    con una mediana que nunca se ajustó.

    Se ajusta con NOMBRES SAFE (los del modelo), no con los nombres crudos
    del CSV: lo que la política produce son columnas del CSV preparado.
    """
    candidatas: list[str] = []
    filas_tipadas: list[dict[str, Any]] = []
    # SOLO LAS FILAS QUE SE VAN A ESCRIBIR. Una fila sin objetivo la descarta
    # `_prepare_training_csv`, así que su hueco no puede decidir la forma del
    # modelo: añadiría un indicador que después nadie marca.
    con_objetivo = [row for row in rows
                    if not _is_null(row.get(target_column), tokens_de_ausencia)]
    for col in feature_columns:
        # NUMÉRICAS, NO BOOLEANAS, y es una decisión declarada, no un olvido.
        # Una `boolean` de este camino se escribe 0/1 pero su tipo declarado no
        # es ninguna de las dos ramas que el núcleo tiene: la mediana de 0/1
        # puede salir 0,5 —un valor que esa columna no tuvo nunca— y rellenar
        # con la más frecuente es justo lo que el núcleo NO hace con una
        # categórica. Inventar aquí una tercera política sería fabricar lo que
        # el núcleo no ha dicho. Una booleana con huecos sigue, por tanto,
        # rechazándose; `column_type_overrides={col: "categorical"}` la lleva a
        # la rama categórica, que sí tiene política (medido, y con prueba con
        # su nombre en `tests/test_c101_c5_faltantes_cableados.py`).
        if columns[col]["type"] not in ("number", "integer"):
            continue
        valores = [_valor_numerico_o_none(row.get(col), tokens_de_ausencia)
                   for row in con_objetivo]
        # Una columna solo entra si TIENE hueco, si le queda algún valor con
        # el que calcular la mediana, y si TODOS los presentes son números:
        # con un valor sin parsear, `ajustar_preparacion` la vería categórica
        # (no parsea texto, por diseño del núcleo) y le pondría centinelas de
        # texto en una columna que el modelo declara Scalar.
        if not any(v is None for v in valores):
            continue
        presentes = [v for v in valores if v is not None]
        if not presentes or not all(isinstance(v, float) for v in presentes):
            continue
        candidatas.append(col)

    if not candidatas:
        return None

    for row in rows:
        fila: dict[str, Any] = {
            # El objetivo viaja SOLO para que el núcleo excluya las filas sin
            # él (nunca se imputa un objetivo) — con la misma definición de
            # «no hay dato» que usa `_prepare_training_csv` al descartarlas,
            # para que la mediana se ajuste exactamente sobre las filas que
            # de verdad se escriben.
            #
            # BAJO UNA CLAVE QUE NADIE PUEDE TENER, no bajo su propio nombre:
            # las features viajan con su nombre SAFE y el objetivo con el
            # CRUDO, así que un objetivo llamado `edad` junto a una feature
            # `edad!` (que sanea a `edad`) compartirían clave y el valor de la
            # feature pisaría al del objetivo — la mediana se ajustaría sobre
            # otras filas. Mismo truco que `_RESERVED_TARGET_SENTINEL` usa
            # unas líneas más arriba para el otro choque de nombres.
            _SENTINELA_OBJETIVO: (
                None if _is_null(row.get(target_column), tokens_de_ausencia)
                else row.get(target_column)),
        }
        for col in candidatas:
            fila[feature_safe_names[col]] = _valor_numerico_o_none(
                row.get(col), tokens_de_ausencia)
        filas_tipadas.append(fila)

    return ajustar_preparacion(
        filas_tipadas,
        objetivo=_SENTINELA_OBJETIVO,
        columnas=[feature_safe_names[col] for col in candidatas],
        # La red densa/composite de MatrixAI no tiene tratamiento nativo ni de
        # faltantes ni de categóricas: son los dos booleanos de `Capacidades`
        # (102-C1) de `matrixai.dense.torch_cpu`, pasados como datos sueltos
        # porque `preparacion.py` es stdlib puro y no puede importar el motor.
        # Con `admite_faltantes=False` la política RELLENA (mediana) en vez de
        # dejar el hueco, que es justo lo que este camino necesita.
        admite_categoricas=False,
        admite_faltantes=False,
    )


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
    *,
    embedding_sources: set[str] | None = None,
    missing_policy: PoliticaDePreparacion | None = None,
    tokens_de_ausencia: set[str] | None = None,
) -> _PreparedCSV:
    # Grupos one-hot/embedding + los mapas valor_crudo->columna o índice,
    # calculados UNA VEZ (no por fila — recalcular _distinct_non_null
    # dentro del bucle de filas es O(filas²) y, peor, podría ver un
    # vocabulario distinto por fila si el cálculo no fuera puramente
    # determinista).
    onehot_columns: dict[str, dict[str, str]] = {}  # col -> {valor_crudo: columna_onehot}
    embedding_columns: dict[str, dict[str, int]] = {}  # col -> {valor_crudo: índice}
    effective_vocabularies: dict[str, list[str]] = {}  # CONTRATO 62 C3
    header: list[str] = []
    operations: list[str] = []
    for col in feature_columns:
        safe_name = feature_safe_names[col]
        col_type = columns[col]["type"]
        if col_type == "categorical":
            values = category_vocabularies.get(col) or _distinct_non_null(
                rows, col, tokens_de_ausencia)
            if len(values) < 2:
                continue
            effective_vocabularies[col] = list(values)
            # ¿ÍNDICE DE EMBEDDING O ONE-HOT? LO DICE EL MODELO, NO LA
            # CARDINALIDAD. `embedding_sources` son las columnas que el `.mxai`
            # recién generado consume como fuente de un EMBEDDING
            # (`embedding_source_columns`, categorical.py). Aquí había un
            # `len(values) > _ONEHOT_MAX` que REIMPLEMENTABA la decisión, y
            # divergía justo en el caso mixto: basta UNA categórica por encima
            # del umbral para que el enrutado mande el prompt ENTERO al
            # generador composite, que materializa como EMBEDDING TODAS las
            # declaradas — también las de 2 valores. El CSV las expandía a
            # one-hot igualmente y el proyecto moría en su propia validación
            # ("Faltan: var191, var194" con KDDCup09_appetency; reproducido el
            # 2026-09-13 con 14 filas sintéticas y cero faltantes).
            # Sin modelo a mano —re-preparar desde una receta anterior a
            # `embedding_columns`— se aplica el criterio VIEJO, que es
            # exactamente el que produjo aquel CSV: cambiarlo ahí reescribiría
            # el pasado.
            if embedding_sources is not None:
                va_por_embedding = safe_name in embedding_sources
            else:
                va_por_embedding = len(values) > _ONEHOT_MAX
            if va_por_embedding:
                # Ver punto 6 del docstring: el CSV lleva el ÍNDICE del valor
                # en el vocabulario (mismo orden que se escribió en el
                # prompt), NUNCA one-hot.
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

    # LOS INDICADORES, AL FINAL Y SOLO SI LOS HAY. Van después de todas las
    # features para que el CSV de un dataset SIN faltantes salga byte a byte
    # igual que antes de este arreglo (la cabecera anterior no se desplaza), y
    # en el MISMO orden en el que el prompt sintetizado los declara — el
    # VECTOR generado sigue ese orden y la validación compara posición a
    # posición.
    indicadores = _nombres_de_indicador(missing_policy) if missing_policy else {}
    if indicadores:
        ya_en_cabecera = set(header) | {target_header}
        colisiones = sorted(n for n in indicadores.values() if n in ya_en_cabecera)
        if colisiones:
            # NO SE PISA EN SILENCIO. Un `dict` de fila no repite claves: si el
            # indicador derivado de una numérica se llama igual que una columna
            # real (un CSV con `edad` y `edad__faltante`) o que una one-hot ya
            # generada, el CSV saldría con una columna de menos que la cabecera
            # y el desalineamiento no se vería hasta entrenar. Es el mismo
            # choque que `PoliticaDePreparacion.columnas_de_salida()` declara
            # sin resolver; aquí sí hay a quién decírselo.
            raise DatasetProjectError(
                "El indicador de valores ausentes que hace falta para "
                f"{', '.join(repr(n) for n in colisiones)} choca con una columna "
                "que este CSV ya produce. Renombra esa columna en el CSV de "
                "origen (el sufijo '__faltante' está reservado para marcar qué "
                "celdas se rellenaron) antes de generar el modelo."
            )
        if len(set(indicadores.values())) != len(indicadores):
            raise DatasetProjectError(
                "Dos columnas numéricas con valores ausentes necesitan el mismo "
                "indicador tras normalizar su nombre. Renombra una de ellas en "
                "el CSV de origen antes de generar el modelo."
            )
        header.extend(indicadores.values())
    header.append(target_header)

    # Las numéricas que la política rellena, indexadas por su nombre SAFE —
    # que es como se ajustó (ver `_ajustar_politica_de_faltantes`).
    imputadas: dict[str, str] = {  # safe_name -> columna cruda
        feature_safe_names[col]: col
        for col in feature_columns
        if missing_policy is not None
        and missing_policy.columna(feature_safe_names[col]) is not None
    }
    imputed_cells: dict[str, int] = {safe: 0 for safe in imputadas}
    missing_category_cells: dict[str, int] = {}

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header)
    writer.writeheader()
    rows_dropped = 0
    for row in rows:
        target_raw = row.get(target_column)
        if _is_null(target_raw, tokens_de_ausencia):
            rows_dropped += 1
            continue  # sin target no hay fila que entrenar (nunca se inventa uno)
        prepared: dict[str, str] = {}
        # LA IMPUTACIÓN LA DECIDE EL NÚCLEO, no este bucle: `transformar_fila`
        # aplica la política tal como quedó ajustada (mediana congelada) y
        # devuelve, junto al valor, su indicador 0.0/1.0. Aquí solo se traduce
        # el texto del CSV a valores tipados y el resultado de vuelta a texto.
        tipados = {safe: _valor_numerico_o_none(row.get(col), tokens_de_ausencia)
                   for safe, col in imputadas.items()}
        # Un valor PRESENTE que no parsea (`"?" ` no: eso es `_is_null`; esto es
        # basura real en una columna que el usuario declaró numérica a mano, o
        # un valor nuevo al re-preparar otro CSV con la misma receta) se aparta
        # ANTES: `transformar_fila` haría `float()` sobre él y reventaría con un
        # ValueError que no dice nada. Ver `_valor_numerico_o_none`.
        sin_parsear = {safe for safe, valor in tipados.items() if isinstance(valor, str)}
        transformada: dict[str, Any] = (
            transformar_fila({safe: (None if safe in sin_parsear else valor)
                              for safe, valor in tipados.items()}, missing_policy)
            if imputadas else {}
        )
        for col in feature_columns:
            info = columns[col]
            safe_name = feature_safe_names[col]
            if info["type"] == "categorical":
                raw = row.get(col)
                raw = raw.strip() if raw is not None else raw
                vocabulario = onehot_columns.get(col) or embedding_columns.get(col) or {}
                # EL TERCER ESTADO DEL NÚCLEO, cuando el vocabulario lo declara.
                # Un hueco en una categórica no es «ninguna de las categorías»
                # (one-hot todo a cero, indistinguible de una fila rota) ni un
                # índice inexistente (la rama de embedding escribía "" y el
                # modelo rechazaba su propio CSV): es `__faltante__`, una
                # categoría más. Que el vocabulario CONGELADO la traiga o no es
                # lo que distingue una receta nueva de una anterior a esto —
                # una vieja se sigue reproduciendo con el criterio de entonces.
                if _is_null(raw, tokens_de_ausencia) and CATEGORIA_FALTANTE in vocabulario:
                    raw = CATEGORIA_FALTANTE
                    missing_category_cells[safe_name] = missing_category_cells.get(safe_name, 0) + 1
                if col in onehot_columns:
                    value_to_column = onehot_columns[col]
                    for onehot_col in value_to_column.values():
                        prepared[onehot_col] = "0"
                    if raw in value_to_column:
                        prepared[value_to_column[raw]] = "1"
                elif col in embedding_columns:
                    idx = embedding_columns[col].get(raw)
                    prepared[safe_name] = str(idx) if idx is not None else ""
                # cardinalidad<2 -> columna excluida arriba, nada que escribir
            elif safe_name in imputadas:
                # NO SE IMPUTA LO QUE SÍ TENÍA DATO. Un valor ilegible es una
                # respuesta equivocada, no un hueco: rellenarlo con la mediana
                # y marcar el indicador a 1 sería declarar «no había valor»
                # cuando lo había. Viaja tal cual y el verificador del modelo lo
                # rechaza con «must be numeric», que es la verdad.
                clave_indicador = nombre_de_indicador(safe_name)
                falta = (safe_name not in sin_parsear
                         and transformada[clave_indicador] == 1.0)
                if falta:
                    prepared[safe_name] = _fmt_num(transformada[safe_name])
                    imputed_cells[safe_name] += 1
                else:
                    # El texto ORIGINAL, no el `float` re-formateado: una celda
                    # que no se tocó no puede cambiar ni un byte por el hecho de
                    # que otra fila tuviera un hueco ("20" no se vuelve "20.0").
                    prepared[safe_name] = row.get(col, "")
                prepared[indicadores[clave_indicador]] = "1" if falta else "0"
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
    if any(imputed_cells.values()):
        operations.append("impute_missing_numeric_median")
        operations.append("add_missing_value_indicator")
    if missing_category_cells:
        operations.append("encode_missing_category")
    return _PreparedCSV(
        text=out.getvalue(), rows_dropped_null_target=rows_dropped,
        operations=operations, effective_vocabularies=effective_vocabularies,
        embedding_source_names=[feature_safe_names[c] for c in embedding_columns],
        imputed_numeric_cells={k: v for k, v in imputed_cells.items() if v},
        missing_category_cells=dict(missing_category_cells),
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_seed(training_text: str) -> int | None:
    """El seed de SPLIT que `training_text` ya declara (GEN lo fija, C2 no
    lo elige) — se re-extrae en vez de duplicar el literal para que la
    procedencia nunca pueda divergir del training_text real devuelto."""
    m = re.search(r"\bseed=(\d+)", training_text)
    return int(m.group(1)) if m else None


def _confirmacion_del_problema(
    *, csv_text: str, analysis: dict[str, Any], rows: list[dict[str, str]],
    target_column: str,
    feature_columns: list[str], unidad_de_observacion: str | None,
    clase_positiva: str | None, momento_de_prediccion: str | None,
    horizonte: Any | None, uso_previsto: str | None,
    tokens_de_ausencia: set[str] | None = None,
) -> dict[str, Any]:
    """El documento del 103-C1 para este proyecto, en JSON.

    Vive aquí y no dentro de `objetivo.py` porque lo que traduce es lo de ESTE
    flujo: las features que el proyecto ha decidido de verdad (`feature_columns`,
    ya sin la columna objetivo, sin las constantes descartadas y sin las de tipo
    no usable) son las entradas del problema — pasarle todas las columnas del CSV
    declararía como predictores unas que este modelo no mira.

    Nunca lanza: un fallo confirmando el problema no puede tumbar una generación
    que ya ha terminado bien. Si algo va mal se dice, con el motivo, en vez de
    devolver un documento a medias que parezca una confirmación.
    """
    from matrixai.estudio import Horizonte  # noqa: PLC0415
    from matrixai.training.objetivo import confirmar_desde_csv  # noqa: PLC0415

    if isinstance(horizonte, dict):
        horizonte = Horizonte.desde_json(horizonte)
    try:
        return confirmar_desde_csv(
            csv_text, objetivo=target_column, entradas=feature_columns,
            filas=rows,
            unidad_de_observacion=unidad_de_observacion,
            clase_positiva=clase_positiva,
            momento_de_prediccion=momento_de_prediccion, horizonte=horizonte,
            uso_previsto=uso_previsto, analisis=analysis,
            tokens_de_ausencia=tokens_de_ausencia,
        ).a_json()
    except Exception as exc:  # noqa: BLE001
        return {"confirmado": False, "problema": None, "propuesta": {},
                "preguntas": [], "bloqueos": [], "pistas": [],
                "error": f"{type(exc).__name__}: {exc}"}


def _build_provenance(
    *,
    csv_text: str,
    prepared_csv: str,
    schema_inferred: dict[str, Any],
    schema_final: dict[str, Any],
    target_column: str,
    excluded_columns: list[str],
    rows_dropped_null_target: int,
    excluded_column_reasons: dict[str, dict[str, Any]] | None = None,
    kept_constant_columns: list[str] | None = None,
    preparation_spec: dict[str, Any] | None = None,
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
    missing_values: dict[str, Any] | None = None,
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
        "preparation_spec": preparation_spec or {},
        "excluded_columns": excluded_columns,
        # CONTRATO 62 C2 — por qué falta cada columna, y qué constantes se
        # conservaron a propósito (lo que C3 necesita para reproducir el
        # esquema con el que nació el modelo, invariante 3).
        "excluded_column_reasons": excluded_column_reasons or {},
        "kept_constant_columns": list(kept_constant_columns or []),
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
        # 101-C5 — QUÉ SE RELLENÓ Y CON QUÉ. (No confundir con el paso
        # `missing_values(drop)` del pipeline TEMPORAL, que vive en
        # `provenance["temporal"]["pipeline_operations"]` y ocurre ANTES, sobre
        # la serie: aquel DESCARTA filas, este declara lo que se rellenó en la
        # preparación de features.) Una imputación afirma algo que el
        # dato no decía, así que no basta con hacerla: queda declarada aquí,
        # celda a celda contada, con el valor usado y con los `Limite` que la
        # propia política encontró motivo para levantar (una columna con más
        # de la mitad de huecos, una muestra corta para ajustar). Siempre
        # presente: `imputed_numeric`/`missing_category` vacíos afirman «no se
        # rellenó nada», que no es lo mismo que no decir nada.
        "missing_values": missing_values or {
            "imputed_numeric": {}, "missing_category": {}, "limits": [],
        },
    }
