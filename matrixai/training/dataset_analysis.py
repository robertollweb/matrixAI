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
  - El TIPO de una columna se infiere de sus VALORES (a diferencia de
    `_suggest_field_types`, heurística de nombre que sigue existiendo para el
    editor manual). Orden de comprobación: fecha → boolean → numérico
    (entero/decimal) → identificador (alta unicidad) → categórica (todo lo
    demás). El orden importa: una columna de fechas únicas no debe caer en
    "identificador", y "0"/"1" puros se leen como boolean antes que como
    entero (mismos tokens que `predict_template.py`).
    ÚNICA excepción, y va dicha porque antes esta línea decía "nunca de su
    nombre": para separar un ENTERO casi-único que es un id de uno que es
    una medida, el nombre entra como segunda señal junto a la forma de los
    valores (ver `_IDENTIFIER_RUN_DENSITY`). Solo puede AÑADIR
    identificadores que la forma no ve, nunca quitar los que sí ve.
  - "categórica" cubre TANTO baja como alta cardinalidad — la cardinalidad
    viaja en el resultado para que C2 decida one-hot vs embedding
    (`_ONEHOT_MAX`, el mismo umbral que ya usan los generadores), esto no es
    una decisión de tipo.
  - Candidatos a TARGET puntúan 3 señales (posición última columna, nombre
    típico, cardinalidad "clasificable") y se devuelven TODOS los candidatos
    viables (excluidas identificador/fecha) ordenados — la UI de C4 propone,
    el usuario elige.

Lo que este módulo RECHAZA además de proponer (auditoría 2026-08-13, las dos
medidas conduciendo el producto):

  - Un fichero que **no es una tabla** (filas con más o menos campos que la
    cabecera, una comilla sin cerrar, bytes de control) no sale de aquí con
    un esquema inventado: `_structural_damage` lo dice con su fila y su
    evidencia. Antes entraba en silencio. Lo que SÍ sigue entrando es el
    texto entrecomillado con comas y saltos dentro, que es CSV legítimo.
  - Un **objetivo que no varía** no es un objetivo: `constant_target_error`
    da el motivo, para que quien reciba un dataset lo rechace ANTES de
    construir y entrenar un modelo que llega a pérdida 0 sin haber aprendido
    nada. Es una función aparte —y no parte de `analyze_dataset_csv`— porque
    el análisis es agnóstico del target: aquí solo se PROPONEN candidatos, y
    quién es el objetivo lo decide quien llama.
"""
from __future__ import annotations

import csv
import io
import re
import statistics
from typing import Any

from matrixai import limits as _limits
from matrixai.training.dense_generator import _ONEHOT_MAX, parece_identificador

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

# ENTERO casi-único: hace falta MÁS que la unicidad para llamarlo id.
#
# La intención escrita aquí siempre fue «el clásico id secuencial
# 1,2,3,...,N», pero la condición implementada era solo «entero + casi todo
# distinto», y eso se traga cualquier MEDIDA de dominio con rango ancho. Un
# `salario` declarado `Scalar en [15000, 120000]` sale entero (el redondeo a
# escala de dominio no deja decimales cuando el span es grande), 198 de 200
# valores distintos → `identifier` → `_NEVER_FEATURE_TYPES` lo saca del
# modelo EN SILENCIO. No es cosa del dato sintético: un CSV real de 300
# sueldos enteros se pierde igual (medido 2026-08-14).
#
# Lo que de verdad distingue a un id secuencial es que sus valores son un
# TRAMO: ocupan casi todos los enteros entre el mínimo y el máximo. Medido
# con los dos lados delante — densidad = valores distintos / (max - min + 1):
#
#   id 1..15          1.0000  |  salario [15000, 120000], 200 filas  0.0019
#   id 0..49          1.0000  |  empleado_id aleatorio de 8 cifras   0.0000
#   centigrados 0..99 1.0000  |
#   P1000..P1014      1.0000  |
#
# Los dos grupos no se rozan, así que el umbral va en medio y no en el borde
# de ninguno. Y como un id ESPARCIDO (un número de empleado aleatorio) no
# forma tramo, se conserva la otra señal que ya existe y que no mira los
# valores: el NOMBRE (`parece_identificador`, contrato 71). Las dos juntas
# cubren los dos lados — sin ellas, arreglar el sesgo del salario habría
# creado el contrario.
_IDENTIFIER_RUN_DENSITY = 0.5

# TEXTO LIBRE dentro de un identificador — lo que distingue una columna de
# reseñas de una de DNIs, medido con los dos lados delante (2026-08-14).
#
# Las dos comparten lo único que hoy se mira, `unique_ratio == 1.0`: cada
# fila es distinta. Por eso un CSV de 120 reseñas salía como `identifier` y
# la columna que le importaba al usuario se caía del modelo. Pero un texto
# escrito por una persona tiene forma propia, y ninguna de estas señales
# mira el NOMBRE de la columna (lección del contrato 71):
#
#   · varias PALABRAS separadas por espacios — un UUID, un DNI, un email,
#     una URL, una ruta o un SKU no tienen ninguno;
#   · palabras DE LETRAS, no trozos alfanuméricos — `SKU-000123-XZ` tiene
#     guiones, no vocabulario;
#   · valores LARGOS;
#   · y la que de verdad separa: el vocabulario SE REPITE entre filas («el»,
#     «pedido», «llegó»…). En una columna de identificadores cada token
#     aparece una sola vez, así que la proporción de palabras distintas se
#     va a ~1,0; en texto real cae por debajo de 0,1.
#
# Medido con 19 columnas de los dos lados: las 7 de texto (reseñas es/en,
# notas clínicas, descripciones, frases cortas) dan SÍ; las 12 de código
# (id secuencial, UUID, DNI, email, matrícula, nombre completo, SKU, código
# con guiones, URL, JSON, ruta de fichero, base64) dan NO. La única que
# cambia de lado es una DIRECCIÓN POSTAL, que también es prosa escrita por
# una persona y que se excluye igual — solo cambia lo que se dice de ella.
_TEXT_MIN_ROWS = 8
_TEXT_MIN_TOKENS = 4          # mediana de tokens separados por espacios
_TEXT_MIN_WORDS = 4           # mediana de palabras de LETRAS (>= 2 letras)
_TEXT_MIN_CHARS = 25          # mediana de longitud
_TEXT_MAX_VOCAB_RATIO = 0.5   # palabras distintas / palabras totales
_TEXT_SAMPLE = 200            # techo de valores mirados (coste acotado)
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# Margen del rango PROPUESTO sobre el rango OBSERVADO (10% del span; con
# span 0 — todas las filas igual valor — se usa un margen absoluto mínimo).
_RANGE_MARGIN_FRACTION = 0.1
_RANGE_MARGIN_MIN_ABS = 1.0

# Bytes de CONTROL que no pueden estar dentro de una celda de texto. Se dejan
# fuera el tabulador (raro pero es texto) y `\n`/`\r`, que se miran aparte
# porque su diagnóstico es OTRO (una celda partida por una comilla sin cerrar,
# no un fichero binario).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Cuántos daños de estructura se enseñan con nombre y apellidos. El resto se
# cuenta: una lista de 4.000 filas rotas no es más accionable que tres
# ejemplos y el total.
_MAX_DANOS_DETALLADOS = 3

# Nombres típicos de columna objetivo (comparación exacta, minúsculas, tras
# strip — es una señal más entre tres, no la única, así que no hace falta
# heurística de substring).
_TARGET_NAME_HINTS = {
    "target", "label", "class", "clase", "resultado", "objetivo", "etiqueta",
    "salida", "output", "result", "outcome", "y", "prediction", "prediccion",
}


class DatasetAnalysisError(ValueError):
    """CSV ilegible o vacío — error accionable (invariante 7 del contrato 57).

    CONTRATO 62 C1: igual que `DatasetProjectError`, puede transportar el
    payload estructurado de un tope superado en `.details`.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details


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
        # AUDITORÍA C1 [MEDIO]: esta era la ÚNICA de las cinco rutas del Studio
        # que seguía lanzando texto plano — y encima decía "0 MB" con un tope
        # pequeño, por dividir entre 1.000.000 con enteros. Ahora emite el
        # mismo payload estructurado que el resto (`limits.limit_error`), que
        # ya elige la unidad según la magnitud.
        raise DatasetAnalysisError(
            _limits.limit_error("max_csv_bytes", size)["error"],
            details=_limits.limit_error("max_csv_bytes", size),
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
        # La LÍNEA física de cada fila se guarda a la vez que la fila: es lo
        # que hay que decirle a alguien para que vaya a mirar el fichero, y
        # `list(reader)` la tiraba. `line_num` es la línea donde ACABA el
        # registro (una celda con saltos de línea ocupa varias).
        rows = []
        row_lines = []
        for row in reader:
            rows.append(row)
            row_lines.append(reader.line_num)
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
        row_lines = row_lines[:max_rows]
    rows_analyzed = len(rows)

    # Un fichero que NO es una tabla se rechaza aquí, antes de inventarle un
    # esquema (ver `_structural_damage`).
    message = _damage_message(_structural_damage(rows, row_lines, columns, csv_text))
    if message is not None:
        raise DatasetAnalysisError(message)

    duplicate_rows = _count_duplicate_rows(rows, columns)

    column_infos: dict[str, dict[str, Any]] = {}
    for col in columns:
        raw_values = [row.get(col) for row in rows]
        column_infos[col] = _analyze_column(raw_values, rows_analyzed, col)

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
# Estructura del fichero
# ---------------------------------------------------------------------------

def _damage_message(damage: tuple[int, list[str]]) -> str | None:
    """Los daños, en UN mensaje accionable. `None` si no hay ninguno."""
    total, examples = damage
    if total == 0:
        return None
    detail = "; ".join(examples)
    if total > len(examples):
        detail += f"; y {total - len(examples)} problema(s) más"
    return (
        f"El fichero no se puede leer como una tabla: {detail}. Corrige esas "
        "filas y vuelve a subirlo: leído así, no describiría tus datos."
    )


def structural_damage_error(csv_text: str) -> str | None:
    """El MISMO veredicto de estructura, para quien recibe un CSV por otra
    puerta que no pasa por `analyze_dataset_csv`.

    Existe porque hay dos puertas y solo una analizaba. Medido el
    2026-08-13: `prepare_dataset_from_provenance` (la que usa «traer mis
    datos» para reentrenar un modelo que ya existe) tragaba el fichero roto
    de la auditoría y devolvía `ok` con UNA fila preparada de las cinco del
    fichero, sin decir que se hubieran perdido cuatro. Entrenar con una
    quinta parte de los datos de alguien sin avisar es peor que rechazarlos.

    Devuelve el motivo o `None`. Nunca lanza: los errores de cabecera, CSV
    vacío o ilegible los sigue diciendo `analyze_dataset_csv` con su
    mensaje, y adelantarse aquí sería un segundo sitio diciendo lo mismo.
    """
    if not csv_text or not csv_text.strip():
        return None
    from matrixai.training.data import normalize_csv_text
    text = normalize_csv_text(csv_text)
    try:
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return None
        columns = [str(c) for c in reader.fieldnames if c is not None]
        rows = []
        row_lines = []
        for row in reader:
            rows.append(row)
            row_lines.append(reader.line_num)
    except csv.Error:
        return None
    return _damage_message(_structural_damage(rows, row_lines, columns, text))


def _structural_damage(
    rows: list[dict[str, Any]],
    row_lines: list[int],
    columns: list[str],
    csv_text: str,
) -> tuple[int, list[str]]:
    """CUÁNTOS daños de ESTRUCTURA tiene el fichero, y los primeros con su
    evidencia.

    Un CSV roto entraba sin decir palabra, y eso es peor que rechazarlo.
    Medido el 2026-08-13 contra el backend real con un fichero de seis
    líneas (una comilla sin cerrar, una fila de 2 campos, bytes binarios y
    una fila de 5): el análisis contestaba `ok` con «2 filas · 3 columnas»
    y un esquema donde `b` y `c` salían «constantes al 50 % de vacíos» —
    pero esas celdas NO estaban vacías en el fichero: es que sus filas
    nunca llegaron a leerse. Con eso, «Construir modelo» se encendía y
    fallaba al pulsarlo, con un error del core sobre algo que la pantalla
    ya sabía. Lo que se sabe se dice ANTES.

    Se RECHAZA en vez de avisar, por el mismo criterio con el que ya se
    rechaza una cabecera con nombres repetidos: no es una preferencia del
    usuario, es que el resultado del análisis no describiría su fichero.
    Los cuatro daños que se miran son hechos comprobables, no heurísticas:

    - **Sobran campos**: `DictReader` mete lo que no cabe bajo la clave
      `None` y nadie lo mira nunca — datos tirados en silencio.
    - **Faltan campos**: `DictReader` rellena con `restval` (`None`), que
      el análisis cuenta como celda vacía. Un campo que no está no es un
      campo vacío (un valor ausente no es un cero).
    - **Comilla sin cerrar**: una celda que ACABA en salto de línea. El
      fichero se terminó dentro de un campo entrecomillado, así que el
      salto que separaba dos filas se leyó como parte del valor.
      «Contiene un salto» no vale como criterio y se probó: un CSV real
      trae texto entrecomillado con comas y saltos DENTRO, y eso está
      soportado a propósito (`test_comas_y_saltos_entre_comillas` de
      `test_repreparacion_c62_c3`, que se puso rojo con la primera
      versión de esta regla). «ACABA en salto» distingue las dos cosas —
      medido con los cuatro casos: comilla sin cerrar, multilínea
      legítima intermedia, multilínea legítima al final, y el único falso
      positivo que queda (un valor que de verdad termina en `\\n` justo
      antes de su comilla de cierre).
    - **Celda binaria**: contiene bytes de control. Eso no es un CSV de
      texto.

    Coste: las dos primeras son O(1) por fila (`DictReader` rellena por la
    IZQUIERDA, así que a una fila corta le falta SIEMPRE la última
    columna). Las dos últimas recorrerían celda a celda, así que primero se
    mira el texto entero de una pasada en C: sin comillas no puede haber
    una celda partida, y sin bytes de control no hay ninguna celda binaria.
    MEDIDO sobre un CSV de 200.000 filas × 8 columnas (12,7 MB): 0,19 s por
    la vía rápida y 0,41 s cuando el fichero SÍ lleva comillas legítimas,
    sobre los 3,25 s que ya costaba el análisis entero.

    Se devuelve el TOTAL y solo los `_MAX_DANOS_DETALLADOS` primeros textos:
    un fichero de un millón de filas todas cortas generaba un millón de
    frases en memoria para enseñar tres. El total sigue siendo exacto — decir
    «y muchos más» cuando se sabe el número sería redondear a peor.
    """
    total = 0
    examples: list[str] = []

    def note(text: str) -> None:
        nonlocal total
        total += 1
        if len(examples) < _MAX_DANOS_DETALLADOS:
            examples.append(text)

    if not columns:
        return total, examples
    last_column = columns[-1]

    def where(i: int) -> str:
        if i < len(row_lines):
            return f"la fila de datos {i + 1} (línea {row_lines[i]})"
        return f"la fila de datos {i + 1}"

    for i, row in enumerate(rows):
        extra = row.get(None)
        if extra:
            note(
                f"{where(i)} tiene {len(columns) + len(extra)} campos y la cabecera "
                f"{len(columns)}: sobra(n) {list(extra)!r}, que se estaba(n) "
                "tirando en silencio"
            )
        if row.get(last_column, "") is None:
            missing = [c for c in columns if row.get(c, "") is None]
            note(
                f"{where(i)} se queda sin la(s) columna(s) {missing!r}: no están "
                "vacías, es que la fila tiene menos campos que la cabecera"
            )

    has_quotes = '"' in csv_text
    has_control = _CONTROL_RE.search(csv_text) is not None
    if has_quotes or has_control:
        for i, row in enumerate(rows):
            for col in columns:
                value = row.get(col)
                if not isinstance(value, str):
                    continue
                if has_quotes and value.endswith(("\n", "\r")):
                    note(
                        f"{where(i)}, columna {col!r}: la celda se ha tragado el "
                        "salto de línea que separaba las filas — hay una comilla "
                        "sin cerrar y el fichero se acaba dentro de esa celda"
                    )
                if has_control:
                    found = _CONTROL_RE.search(value)
                    if found is not None:
                        note(
                            f"{where(i)}, columna {col!r}: la celda contiene bytes de "
                            f"control ({found.group()!r}) — esto no es un CSV de texto"
                        )
    return total, examples


# ---------------------------------------------------------------------------
# Por columna
# ---------------------------------------------------------------------------

def _is_null(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in _NULL_TOKENS


def _integer_run_density(values: list[str]) -> float:
    """Cuánto TRAMO ocupan estos enteros: distintos / (max - min + 1).

    1.0 es `1,2,3,...,N` sin huecos (el id secuencial); una medida de
    dominio con rango ancho (un sueldo) se queda en milésimas. Devuelve la
    medida, no un veredicto — quien decide es `_analyze_column`, y quien
    audita tiene que poder ver el número con el que se afirmó.
    """
    enteros = [int(float(v)) for v in values]
    span = max(enteros) - min(enteros) + 1
    if span <= 0:
        return 1.0
    return len(set(enteros)) / span


def _analyze_column(
    raw_values: list[str | None], rows_analyzed: int, column_name: str = "",
) -> dict[str, Any]:
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
        if cardinality < 2:  # CONTRATO 62 C2 — ver el bloque numérico
            info["constant"] = True
            info["constant_value"] = distinct[0]
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
        # Casi-único NO basta (ver `_IDENTIFIER_RUN_DENSITY`): o los valores
        # forman un TRAMO —el id secuencial que esta rama siempre dijo
        # buscar— o lo dice el NOMBRE, que es la señal que cubre al id
        # esparcido. Si no es ninguna de las dos, es una medida de dominio y
        # sigue por la rama numérica de abajo con su rango.
        densidad = _integer_run_density(non_null)
        if densidad >= _IDENTIFIER_RUN_DENSITY or parece_identificador(column_name):
            info["type"] = "identifier"
            info["cardinality"] = cardinality
            info["unique_ratio"] = round(unique_ratio, 4)
            info["run_density"] = round(densidad, 4)
            return info
    if numeric_kind is not None:
        values = [float(v) for v in non_null]
        lo, hi = min(values), max(values)
        info["type"] = numeric_kind
        info["cardinality"] = cardinality
        info["observed_range"] = _round_range([lo, hi], numeric_kind)
        info["proposed_range"] = _round_range(_propose_margin(lo, hi), numeric_kind)
        # CONTRATO 62 C2: una columna con un solo valor distinto no aporta
        # NADA al modelo, y además `_propose_margin` le inventa un rango de
        # ±1 (`lat` constante 43.46 salía como [42.46, 44.46]) que en el panel
        # de inferencia se convierte en un slider que solo sirve para sacar al
        # modelo de su distribución. Se marca aquí; quien decide qué hacer con
        # ella es `dataset_project` (excluirla por defecto).
        if cardinality < 2:
            info["constant"] = True
            info["constant_value"] = distinct[0]
        return info

    if is_identifier_candidate:
        info["type"] = "identifier"
        info["cardinality"] = cardinality
        info["unique_ratio"] = round(unique_ratio, 4)
        # El TIPO no cambia, y es a propósito: `identifier` viaja por la API
        # de análisis a TRES interfaces —una de ellas la clásica, que no se
        # toca— y sus listas de tipos son cerradas (`SchemaRowType` y la
        # unión de `api/client.ts` en el Studio, `_VALID_COLUMN_TYPES` en el
        # backend, que RECHAZARÍA un tipo nuevo al validar una plantilla).
        # Inventar aquí un tipo `text` rompería a quien no puede arreglarse.
        # Lo que sí se puede es DECIR lo que se ve: una clave añadida que
        # quien no la conoce ignora, y que quien decide (dataset_project)
        # usa para no llamar «identificador» a una columna de reseñas.
        evidencia = _free_text_evidence(non_null)
        if evidencia is not None:
            info["looks_like_free_text"] = True
            info["free_text_evidence"] = evidencia
        return info

    info["type"] = "categorical"
    info["cardinality"] = cardinality
    if cardinality < 2:  # CONTRATO 62 C2 — ver el bloque numérico
        info["constant"] = True
        info["constant_value"] = distinct[0]
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


def _free_text_evidence(values: list[str]) -> dict[str, Any] | None:
    """La EVIDENCIA de que esta columna es texto escrito por una persona, o None.

    Devuelve los números con los que se afirma, no un `True` pelado: quien
    audita tiene que poder ver por qué se dijo, y un umbral sin su medida al
    lado no se puede discutir. No mira el nombre de la columna ni decide
    ningún tipo — solo describe la forma de los valores.

    Los cinco criterios son AND: basta que uno falle para que la columna
    siga siendo lo que era. Es deliberado que el sesgo caiga de ese lado —
    marcar un identificador como texto estropearía el caso que hoy funciona
    bien, mientras que no marcar un texto solo deja las cosas como estaban.
    """
    muestra = [v.strip() for v in values if v and v.strip()][:_TEXT_SAMPLE]
    if len(muestra) < _TEXT_MIN_ROWS:
        return None
    tokens = statistics.median([len(v.split()) for v in muestra])
    if tokens < _TEXT_MIN_TOKENS:
        return None
    palabras_por_valor = [len(_WORD_RE.findall(v)) for v in muestra]
    palabras = statistics.median(palabras_por_valor)
    if palabras < _TEXT_MIN_WORDS:
        return None
    caracteres = statistics.median([len(v) for v in muestra])
    if caracteres < _TEXT_MIN_CHARS:
        return None
    todas = [w.lower() for v in muestra for w in _WORD_RE.findall(v)]
    if not todas:
        return None
    reuso = len(set(todas)) / len(todas)
    if reuso > _TEXT_MAX_VOCAB_RATIO:
        return None
    return {
        "median_words": palabras,
        "median_chars": caracteres,
        "distinct_word_ratio": round(reuso, 4),
        "values_sampled": len(muestra),
    }


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
    """El rango PROPUESTO: el observado con un margen del 10 %.

    El margen existe para que un valor un poco fuera de lo visto no se
    recorte en inferencia.

    **Cruza el cero, y eso se ve raro en pantalla.** Con la lluvia de
    Santander (observado 0 → 71,4 mm) sale `[-7.14, 78.54]`, y la fase
    Prueba lo enseña como el dominio del campo: lluvia negativa. Igual
    las horas de precipitación (`[-3, 27]`, en un día de 24) y la
    nubosidad (`[-10, 110]`). Lo cazó una auditoría conduciendo la
    plantilla (2026-08-13).

    **Se INTENTÓ acotarlo aquí y se deshizo**: acotar el margen al cero
    rompió el entrenamiento compuesto de Kelvin con «Numerical result
    out of range». Este número no es un rótulo — la normalización del
    entrenamiento depende de él, y cambiarlo mueve la escala con la que
    aprende la red. Un arreglo cosmético que rompe el entrenamiento es
    peor que el defecto.

    Queda ABIERTO y anotado: el sitio correcto es la PANTALLA, que hoy
    llama «dominio esperado» a una envolvente, o la ficha de la
    plantilla, que sí sabe que una humedad no pasa del 100 %. El core no
    puede saberlo.
    """
    span = hi - lo
    margin = span * _RANGE_MARGIN_FRACTION if span > 0 else _RANGE_MARGIN_MIN_ABS
    return [lo - margin, hi + margin]


def _round_range(range_pair: list[float], numeric_kind: str) -> list[float | int]:
    if numeric_kind == "integer":
        import math
        return [int(math.floor(range_pair[0])), int(math.ceil(range_pair[1]))]
    return [round(range_pair[0], 4), round(range_pair[1], 4)]


# ---------------------------------------------------------------------------
# El objetivo
# ---------------------------------------------------------------------------

def constant_target_error(csv_text: str, target_column: str) -> str | None:
    """Por qué esta columna no puede ser el OBJETIVO por no variar, o `None`.

    El caso, medido conduciendo el producto el 2026-08-13: un CSV de 300
    filas con `y = 7` en todas, elegido como objetivo. El modelo se
    construye, entrena hasta `Pérdida 0.000` / `Error de validación
    0.000`, el raíl dice «Entrenamiento completado», la auditoría «6
    pasan» y el despliegue queda disponible. Nadie miente en ninguna de
    esas pantallas por separado, y el resultado es un modelo inservible
    con boletín de notas perfecto.

    **Se RECHAZA, no se avisa**, y por dos razones que ya están escritas
    en el propio producto:

    1. Es el MISMO criterio que ya se aplica a las features. Una columna
       constante se excluye por defecto (`constant` se marca en
       `_analyze_column`) y, si no queda ninguna otra, `dataset_project`
       aborta: «Ninguna columna es utilizable como feature: [...] tienen
       un único valor en todo el CSV y no aportan información». Que el
       mismo hecho —una columna que no varía— bloquee como feature y pase
       como objetivo era la incoherencia.
    2. El core YA rechaza el objetivo constante cuando la tarea es de
       CLASIFICACIÓN («no hay nada que clasificar»), sin escape ninguno.
       Lo que quedaba abierto era la puerta de REGRESIÓN, que es la que
       toma un objetivo numérico de cardinalidad 1. No hay nada que
       conservar aquí, así que tampoco hay palanca de «consérvalo igual»:
       a diferencia de una feature constante —que se puede querer en el
       esquema declarando su rango de dominio real— un objetivo que no
       varía no es un problema de aprendizaje difícil, es que no hay
       problema.

    Coste, MEDIDO sobre un CSV de 200.000 filas y 12,7 MB: 0,24 s, y casi
    todo se va en `normalize_csv_text` (una reescritura O(n) del texto que
    hace falta para que el `;` de Excel no esconda la columna). El recorrido
    de filas para en cuanto aparece el segundo valor distinto, así que en un
    dataset normal son dos o tres filas; la pasada completa solo se paga
    cuando la columna es de verdad constante, que es justo cuando se va a
    rechazar.

    Devuelve `None` —y no un error— si la columna no existe o si no tiene
    ningún valor: esos dos casos ya tienen su propio mensaje aguas abajo
    («no es un target válido»), y dos mensajes para lo mismo acaban
    divergiendo.

    Y devuelve `None` también en cuanto el fichero da señales de NO ser una
    tabla (una fila a la que le falta o le sobra un campo, una celda que se
    tragó el salto de línea, bytes de control). Sin eso, el CSV roto de la
    auditoría —seis líneas, una comilla sin cerrar— salía por aquí con «la
    columna 'c' tiene un único valor ('3')», que es una conclusión sacada de
    un fichero que no se ha podido leer entero: el diagnóstico correcto lo da
    `analyze_dataset_csv`, y taparlo con éste sería mandar a alguien a
    cambiar de columna cuando lo que tiene que arreglar es el fichero.
    """
    if not csv_text or not csv_text.strip() or not target_column:
        return None

    from matrixai.training.data import normalize_csv_text
    reader = csv.DictReader(io.StringIO(normalize_csv_text(csv_text)))
    try:
        if not reader.fieldnames or target_column not in reader.fieldnames:
            return None
        # Se mira la ÚLTIMA columna aunque el objetivo sea otra: `DictReader`
        # rellena por la izquierda, así que a una fila corta le falta siempre
        # la última — y un fichero con filas cortas no se diagnostica aquí ni
        # aunque el objetivo esté entre las columnas que sí llegaron.
        last_column = str(reader.fieldnames[-1])
        distinct: set[str] = set()
        for row in reader:
            if None in row:
                return None  # a esa fila le sobran campos: el fichero está roto
            if row.get(last_column, "") is None:
                return None  # a esa fila le FALTAN campos (no es una celda vacía)
            value = row.get(target_column)
            if value is None:
                return None
            if _is_null(value):
                continue
            if value.endswith(("\n", "\r")) or _CONTROL_RE.search(value):
                return None  # comilla sin cerrar o celda binaria: no es una tabla
            distinct.add(value.strip())
            if len(distinct) > 1:
                return None
    except csv.Error:
        # Un CSV ilegible no es cosa de esta comprobación: `analyze_dataset_
        # csv` lo dice con su propio error, y adelantarse aquí lo taparía.
        return None

    if len(distinct) != 1:
        return None
    only = next(iter(distinct))
    return (
        f"La columna objetivo {target_column!r} tiene un único valor en todo el "
        f"CSV ({only!r}): no hay nada que aprender. Un modelo entrenado contra un "
        "objetivo que no varía llega a pérdida 0 acertando siempre lo mismo, y ese "
        "0 se lee como un acierto perfecto. Elige una columna que varíe, o trae "
        "datos en los que el resultado cambie."
    )


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
            else:
                # Auditoría C3 [MEDIA]: una categórica/boolean con MUCHAS
                # categorías (por encima de _ONEHOT_MAX) que además no es la
                # última columna ni tiene nombre típico llegaba aquí con
                # `reasons=[]`/`reason_codes=[]` — score 0.0 pero SIN motivo,
                # contradiciendo la promesa contractual de un motivo por
                # tarjeta. Motivo no puntuable (no toca `score`, orden ni la
                # heurística previa): es categórica, así que sigue siendo
                # clasificación aunque tenga muchas clases.
                reasons.append(f"columna categórica ({cardinality} categorías)")
                reason_codes.append({"code": "categorical_target", "cardinality": cardinality})
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
