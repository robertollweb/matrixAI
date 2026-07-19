# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 C10 — DenseNetworkGenerator: genera NetworkSpec y textos .mxai/.mxtrain desde intención humana."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai import limits as _limits
from matrixai.generation import parse_field_specs, strip_field_specs
from matrixai.training.categorical import expand_categoricals

# GEN C2: a declared categorical with at most this many values becomes one-hot
# columns here (dense model); above it, it should be an embedding (composite path).
# Keeps one-hot column counts sane; aligns the old composite `vocab > 5`.
_ONEHOT_MAX = 12


class DenseNetworkGeneratorError(ValueError):
    pass


@dataclass(frozen=True)
class DenseNetworkGenerationResult:
    prompt: str
    network_name: str
    input_name: str
    input_dim: int
    output_type: str
    output_activation: str
    loss_type: str
    hidden_layers: list[tuple[int, str]]
    output_units: int
    labels: list[str]
    mxai_text: str
    training_text: str
    dataset_template_text: str
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # GEN C2: categoricals declared in the prompt that were materialized as one-hot
    # ({campo: [valores humanos ordenados]}). The canonical human input stays the
    # original field; the .mxai/training_text carry the expanded columns. Empty when
    # no categorical was declared. Source of truth for the export's field_categories.
    field_categories: dict[str, list[str]] = field(default_factory=dict)
    # GEN C3: scalar ranges declared in the prompt ({campo: [min, max]}). NOT written
    # into the .mxai VECTOR type (training data is normalized to [0,1]; a raw range
    # there would make the training verifier reject every normalized row). This is
    # metadata only — the canonical source for the Studio/export's field_ranges.
    field_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # GEN C3: declared semantic types ({campo: "boolean"|"integer"}) for fields whose
    # .mxai type stays a bare Scalar (same reasoning as field_ranges). Canonical source
    # for the Studio/export's field_types.
    field_types: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def network_spec(self) -> NetworkSpec:
        all_layers = list(self.hidden_layers) + [(self.output_units, self.output_activation)]
        specs: list[DenseLayerSpec] = []
        dim = self.input_dim
        for i, (units, activation) in enumerate(all_layers, start=1):
            specs.append(DenseLayerSpec(
                index=i,
                units=units,
                activation=activation,
                input_shape=[dim],
                output_shape=[units],
            ))
            dim = units
        output_name = _output_name(self.output_activation)
        return NetworkSpec(
            name=self.network_name,
            input=self.input_name,
            layers=specs,
            output=output_name,
            output_type_str=self.output_type,
        )


class DenseNetworkGenerator:
    _REGRESSION_KEYWORDS = [
        "precio", "price", "predecir", "estim", "regres",
        "temperatura", "temperature", "consumo", "duracion",
        "valor", "value", "cantidad", "amount",
    ]
    _BINARY_KEYWORDS = [
        "spam", "fraude", "fraud", "binario", "binary",
        "dos clases", "two classes", "positivo o negativo",
        "detec", "detect",
    ]
    _MULTICLASS_KEYWORDS = [
        "clasifica", "classify", "categoriza", "categor",
        "multiclase", "multiclass", "clases", "categorias",
    ]
    # Strong, unambiguous classification intent. These outrank regression keywords:
    # a clinical feature named "temperatura" must not flip an explicit classifier
    # ("clasificación multiclase") into a regressor.
    _STRONG_CLASSIFICATION_KEYWORDS = [
        "clasifica", "classify", "classification", "multiclase", "multiclass",
        "categoriza", "categorize",
    ]

    _FIELD_RE = re.compile(
        r"(?:campos|fields|variables|features|entradas|inputs)\s*(?::|=|son|are)?\s*(?P<fields>[^.;\n]+)",
        re.IGNORECASE,
    )
    _LABEL_RE = re.compile(
        r"(?:labels?|etiquetas|clases|categorias|categories|niveles?|levels?)\s*(?::|=|son|are)?\s*(?P<labels>[^.;\n]+)",
        re.IGNORECASE,
    )
    # Truncates the captured label region at descriptive connectors so trailing
    # prose (architecture/feature descriptions) is not parsed as class names.
    _LABEL_STOP_RE = re.compile(
        r"\s+(?:con|usando|mediante|para|seg[uú]n|a\s+partir\s+de|y\s+una|"
        r"with|using|from|based\s+on|"
        r"features?|caracter[ií]sticas?|variables?|columnas?|atributos?)\b.*$",
        re.IGNORECASE | re.DOTALL,
    )
    _NAME_RE = re.compile(
        r"\b(?:network|red|modelo|model)\s*(?:llamad[ao]|named|called)?\s*(?P<name>[A-Za-z_][\w]*)",
        re.IGNORECASE,
    )
    _ENTITY_RE = re.compile(
        r"\b(?:entidad|entity|entrada|input)\s*(?:llamad[ao]|named|called)?\s*(?P<name>[A-Za-z_][\w]*)",
        re.IGNORECASE,
    )
    # "12 capas", "12 capas Dense ocultas", "12 hidden layers", "12 layers". La palabra
    # "Dense"/"densas"/"ocultas" entre medias NO debe romper la detección (antes exigía
    # "capas ocultas" juntas → "capas Dense ocultas" no casaba y caía al default).
    _DEPTH_RE = re.compile(
        r"(\d+)\s*(?:capas|hidden\s+layers?|layers?)\b",
        re.IGNORECASE,
    )
    # M12: el tope de profundidad ahora es configurable en runtime (limits.cap(..,"max_depth"));
    # este valor es solo el default histórico (perfil equilibrado), conservado como referencia.
    _MAX_EXPLICIT_DEPTH = 12
    # M12 — ancho de capa desde el prompt ("2048 unidades", "units=2048"). Sin esto el
    # ancho lo fija el tapering (máx 256) y no se pueden pedir redes grandes (la GPU no
    # se carga). El tope es de cordura (evita typos tipo units=999999), no de capacidad:
    # la máquina del usuario manda (ver M12 en MEJORAS_FUTURAS).
    _WIDTH_RE = re.compile(
        r"units?\s*[:=]\s*(\d+)|(\d+)\s*(?:unidades|neuronas|units|de\s+ancho)",
        re.IGNORECASE,
    )
    _MAX_EXPLICIT_WIDTH = 16384
    _EPOCHS_RE = re.compile(
        r"(?:\bepochs?\b|\bepocas?\b)\s*[:=]?\s*(\d+)|(\d+)\s*(?:\bepochs?\b|\bepocas?\b)",
        re.IGNORECASE,
    )
    _EARLY_STOP_RE = re.compile(
        r"early[_\s-]?stop\w*\s+patience\s*[:=]?\s*(\d+)(?:\s+metric\s*[:=]?\s*([A-Za-z_][\w.]*))?",
        re.IGNORECASE,
    )
    _MAX_EPOCHS = 1000
    _DEFAULT_EPOCHS = 50

    def generate(
        self,
        prompt: str,
        *,
        input_fields: list[str] | None = None,
        labels: list[str] | None = None,
        network_name: str | None = None,
        input_name: str | None = None,
        hidden_layers: list[tuple[int, str]] | None = None,
    ) -> DenseNetworkGenerationResult:
        clean = " ".join(prompt.strip().split())
        if not clean:
            raise DenseNetworkGeneratorError("DenseNetworkGenerator requires a non-empty prompt")

        # GEN C4: task/label resolution shared with the composite generator (invariant
        # 5) — an explicit ProbabilityMap[...]/Label[...] bracket wins over caller
        # labels and task keywords; 2 declared labels mean 2-class softmax, never
        # the 1-unit sigmoid (see resolve_task_and_labels).
        task, resolved_labels, label_warnings = resolve_task_and_labels(self, clean, labels)
        # GEN C1/C2/C3: honor explicit field-type declarations from the prompt (shared
        # with the composite generator so both use the SAME policy — invariant 5).
        resolved_fields, specs_by_name, field_ranges, field_types, spec_warnings = \
            resolve_prompt_fields(self, prompt, input_fields)
        input_dim = len(resolved_fields)
        resolved_name = network_name or self._extract_name(clean) or _default_network_name(task)
        resolved_entity = input_name or self._extract_entity(clean) or "Input"

        output_activation, output_type, output_units, loss_type = _output_config(task, resolved_labels)
        resolved_hidden = hidden_layers or self._extract_hidden_layers(clean, input_dim)
        # M8-A1: sanitize whatever architecture we got (default / prompt / LLM)
        # so no source can emit a dying-ReLU bottleneck before the output.
        resolved_hidden, sanitizer_notes = sanitize_hidden_layers(resolved_hidden)
        epochs = self._extract_epochs(clean)
        early_stop = self._extract_early_stop(clean)

        mxai_text = _build_mxai_text(
            resolved_name, resolved_entity, resolved_fields,
            resolved_hidden, output_units, output_activation, output_type,
        )
        out_name = _output_name(output_activation)
        ds_target_type = _dataset_target_type(task, resolved_labels if task == "multiclass" else None)
        training_text = _build_training_text(
            resolved_name, resolved_entity, resolved_fields,
            out_name, ds_target_type, loss_type,
            epochs=epochs, early_stop=early_stop,
        )

        # GEN C2: materialize declared low-cardinality categoricals as one-hot. Reuse
        # expand_categoricals, which rewrites the .mxai VECTOR and the training_text
        # FROM COLUMNS together, so training/inference use the expanded columns while
        # the human canonical input stays the original field. High-cardinality
        # categoricals (> _ONEHOT_MAX) are left for the embedding/composite path.
        # field_ranges/field_types already resolved by resolve_prompt_fields (C3).
        field_categories: dict[str, list[str]] = {}
        categoricals = {
            name: list(specs_by_name[name].values or [])
            for name in resolved_fields
            if name in specs_by_name and specs_by_name[name].kind == "categorical"
            and 2 <= len(specs_by_name[name].values or []) <= _ONEHOT_MAX
        }
        # GEN C5: a declared categorical beyond one-hot territory needs the embedding
        # (composite) path — the playground dispatch routes it there. A DIRECT dense
        # call leaves it scalar; say so loudly instead of dropping the declaration
        # in silence (spec_warnings feeds result.warnings below).
        for name in resolved_fields:
            spec = specs_by_name.get(name)
            if (spec is not None and spec.kind == "categorical" and spec.values
                    and len(spec.values) > _ONEHOT_MAX):
                spec_warnings.append(
                    f"'{name}': Categorical de {len(spec.values)} valores "
                    f"(> {_ONEHOT_MAX}) requiere el path composite (embedding); "
                    "el generador denso la deja como escalar."
                )
        template_fields = resolved_fields
        if categoricals:
            expansion = expand_categoricals(mxai_text, training_text, categoricals)
            mxai_text = expansion.mxai_text
            training_text = expansion.training_text
            field_categories = {c: list(v) for c, v in categoricals.items()}
            template_fields = _expanded_field_order(resolved_fields, expansion.groups)
            input_dim = len(template_fields)

        header = template_fields + [out_name]
        # Binary target type is Probability (numeric) — dummy must be float, not a label string.
        # Multiclass target type is Label — dummy is the first label string.
        dummy_target = resolved_labels[0] if task == "multiclass" else "0.0"
        dummy_values = ["0.0"] * len(template_fields) + [dummy_target]
        dataset_template_text = ",".join(header) + "\n" + ",".join(dummy_values) + "\n"

        depth_note = f"depth from prompt ({len(resolved_hidden)} layers)" if self._DEPTH_RE.search(_norm(clean)) else "default depth"
        assumptions = [
            f"DenseNetworkGenerator inferred task={task}",
            f"input_dim={input_dim} from {len(resolved_fields)} fields",
            f"hidden architecture: {resolved_hidden} ({depth_note})",
            f"loss={loss_type}, output_activation={output_activation}",
            "Architecture is a heuristic — tune for production",
        ]
        warnings: list[str] = list(sanitizer_notes)
        warnings.extend(spec_warnings)  # GEN C1/C3: rango inválido, categórica <2, etc.
        warnings.extend(label_warnings)  # GEN C4: labels= ignorados / bracket recortado
        if task == "multiclass" and len(resolved_labels) < 2:
            warnings.append("multiclass task requires at least 2 labels — using defaults")

        return DenseNetworkGenerationResult(
            prompt=clean,
            network_name=resolved_name,
            input_name=resolved_entity,
            input_dim=input_dim,
            output_type=output_type,
            output_activation=output_activation,
            loss_type=loss_type,
            hidden_layers=resolved_hidden,
            output_units=output_units,
            labels=resolved_labels,
            mxai_text=mxai_text,
            training_text=training_text,
            dataset_template_text=dataset_template_text,
            assumptions=assumptions,
            warnings=warnings,
            field_categories=field_categories,
            field_ranges=field_ranges,
            field_types=field_types,
        )

    def _extract_hidden_layers(self, prompt: str, input_dim: int) -> list[tuple[int, str]]:
        norm = _norm(prompt)
        width = self._extract_width(norm)
        m = self._DEPTH_RE.search(norm)
        if m:
            n = _limits.cap(int(m.group(1)), "max_depth")
            # M12: ancho del prompt → capas uniformes de ese ancho; si no, tapering.
            if width is not None:
                return [(width, "relu")] * n
            return _hidden_layers_for_depth(n, input_dim)
        if width is not None:
            # Ancho explícito sin profundidad → profundidad por defecto con ese ancho.
            n = len(_default_hidden_layers(input_dim))
            return [(width, "relu")] * n
        return _default_hidden_layers(input_dim)

    def _extract_width(self, norm_prompt: str) -> int | None:
        m = self._WIDTH_RE.search(norm_prompt)
        if not m:
            return None
        raw = int(m.group(1) or m.group(2))
        if raw <= 0:
            return None
        return min(raw, self._MAX_EXPLICIT_WIDTH)

    def _extract_epochs(self, prompt: str) -> int:
        return extract_epochs_from_prompt(prompt)

    def _extract_early_stop(self, prompt: str) -> tuple[int, str] | None:
        return extract_early_stop_from_prompt(prompt)

    def _detect_task(self, prompt: str, labels: list[str] | None) -> str:
        # Regression keywords in the prompt take priority over LLM-supplied labels,
        # preventing an over-eager LLM from turning "predict price" into a classifier.
        # Exception: explicit classification vocabulary ("clasificación multiclase",
        # "classify") outranks regression keywords, which may match feature names.
        text = _norm(prompt).lower()
        if _any(text, self._STRONG_CLASSIFICATION_KEYWORDS):
            if labels is not None:
                return "binary" if len(labels) == 2 else "multiclass"
            # GEN C4 fix: "binari" (stem) catches both "binario" and the
            # feminine-agreement "binaria" ("clasificación binaria"), which the
            # exact word "binario" was silently missing — that miss used to fall
            # through to "multiclass" with 3 fake default labels for the
            # contract's own retrocompat example ("clasificación binaria" a secas).
            if _any(text, ["binari", "binary", "dos clases", "two classes"]):
                return "binary"
            return "multiclass"
        if _any(text, self._REGRESSION_KEYWORDS):
            return "regression"
        if labels is not None:
            if len(labels) == 2:
                return "binary"
            if len(labels) > 2:
                return "multiclass"
        if _any(text, self._BINARY_KEYWORDS):
            return "binary"
        if _any(text, self._MULTICLASS_KEYWORDS):
            return "multiclass"
        return "regression"

    def _extract_bracket_labels(self, prompt: str) -> list[str]:
        """Labels declared EXPLICITLY via `ProbabilityMap[...]`/`Label[...]` in the
        prompt — the most reliable source (avoids capturing prose like "(6 clases)").
        GEN C4: such a bracket is a declared OUTPUT TYPE, so it forces the softmax
        classification path and wins over caller labels (resolve_task_and_labels).
        Returns the FULL declared list, uncapped: the max_labels limit is the
        caller's job, because truncating an explicitly declared output must warn
        (resolve_task_and_labels), never happen silently."""
        mb = re.search(r"(?:ProbabilityMap|Label)\s*\[\s*(?P<labels>[^\]]+)\]", prompt, re.IGNORECASE)
        if not mb:
            return []
        parts = [p for p in re.split(r",|;", mb.group("labels")) if p.strip()]
        bracket = [_identifier(p) for p in parts if _identifier(p)]
        return bracket if len(bracket) >= 2 else []

    def _extract_labels(self, prompt: str) -> list[str]:
        bracket = self._extract_bracket_labels(prompt)
        if bracket:
            m_labels = _limits.get_limit("max_labels")
            return bracket if m_labels is None else bracket[:m_labels]
        m = self._LABEL_RE.search(prompt)
        if not m:
            return []
        raw = m.group("labels")
        # Drop trailing descriptive prose so it is not swallowed as labels, e.g.
        # "BAJO MEDIO ALTO con una red profunda…" → "BAJO MEDIO ALTO". The connectors
        # introduce architecture/feature descriptions, not class names.
        raw = self._LABEL_STOP_RE.sub("", raw).strip()
        parts = [p for p in re.split(r",|;|\s+y\s+|\s+and\s+|\s+o\s+|\s+or\s+", raw,
                                     flags=re.IGNORECASE) if p.strip()]
        # Space-separated short labels ("BAJO MEDIO ALTO") when no explicit separator
        # produced a list. Multi-word labels stay intact when comma/connector-separated.
        if len(parts) < 2:
            ws = raw.split()
            if len(ws) >= 2:
                parts = ws
        result = [_identifier(p) for p in parts if _identifier(p)]
        if len(result) < 2:
            return []
        m_labels = _limits.get_limit("max_labels")
        return result if m_labels is None else result[:m_labels]

    def _extract_fields(self, prompt: str, *, min_count: int = 2) -> list[str]:
        m = self._FIELD_RE.search(prompt)
        if not m:
            return []
        raw = m.group("fields")
        # Si tras la palabra clave queda una CABECERA con dos puntos antes de la lista
        # ("NUMÉRICAS (24), normalizables a su rango físico: vibracion_axial, ..."), la
        # lista real empieza tras el PRIMER ':' — lo anterior es prosa, no son campos.
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        # Quita rangos/anotaciones entre corchetes o paréntesis de cada campo
        # ("vibracion_axial [0-50]" → "vibracion_axial", "(24)" → "").
        raw = re.sub(r"[\[(][^\])]*[\])]", " ", raw)
        parts = re.split(r",|;|\s+y\s+|\s+and\s+", raw, flags=re.IGNORECASE)
        result = [_identifier(p) for p in parts if _identifier(p)]
        return result if len(result) >= min_count else []

    def _extract_name(self, prompt: str) -> str:
        m = self._NAME_RE.search(prompt)
        return _titlecase(m.group("name")) if m else ""

    def _extract_entity(self, prompt: str) -> str:
        m = self._ENTITY_RE.search(prompt)
        return _titlecase(m.group("name")) if m else ""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _detect_task_from_labels(labels: list[str] | None) -> str | None:
    if labels is None:
        return None
    return "binary" if len(labels) == 2 else "multiclass"


def _output_config(task: str, labels: list[str]) -> tuple[str, str, int, str]:
    if task == "regression":
        return "linear", "Scalar", 1, "mse"
    if task == "binary":
        return "sigmoid", "Probability", 1, "binary_cross_entropy"
    label_str = ", ".join(labels)
    return "softmax", f"ProbabilityMap[{label_str}]", len(labels), "cross_entropy"


def _dataset_target_type(task: str, labels: list[str] | None = None) -> str:
    """Return the TARGET type expected in the .mxtrain DATASET (CSV column type, not model output)."""
    if task == "regression":
        return "Scalar"
    if task == "binary":
        return "Probability"
    if labels:
        return f"Label[{', '.join(labels)}]"
    return "Label"


# M8-A1: minimum width for a ReLU hidden layer. A narrow ReLU layer (especially
# the one feeding the softmax, e.g. Dense(n_classes, relu) → Dense(n_classes,
# softmax)) is a dying-ReLU trap: its few units can all die during training,
# collapsing the model to a constant predictor. This floor matches the width the
# deterministic generator already uses by default, and is enforced on EVERY
# source (default / prompt / LLM) so no path can emit the bottleneck.
_MIN_RELU_WIDTH = 16


def sanitize_hidden_layers(
    hidden_layers: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[str]]:
    """M8-A1 — widen narrow ReLU hidden layers to avoid dying-ReLU bottlenecks.

    Returns (sanitized_layers, notes) where notes describes any change made, for
    auditability (surfaced in the pipeline). Non-ReLU layers are left untouched.
    """
    out: list[tuple[int, str]] = []
    notes: list[str] = []
    for i, (units, activation) in enumerate(hidden_layers):
        if activation == "relu" and units < _MIN_RELU_WIDTH:
            notes.append(
                f"capa oculta {i + 1}: ancho {units}→{_MIN_RELU_WIDTH} "
                f"(evita un cuello ReLU que colapsaría el modelo)"
            )
            units = _MIN_RELU_WIDTH
        out.append((units, activation))
    return out, notes


# Auditoría C5 [MEDIA]: `sanitize_hidden_layers` de arriba SOLO ensancha
# ReLU demasiado estrechas — no valida tipos ni acota profundidad/ancho, así
# que no basta como defensa para `architecture_hints`, un payload que
# `/api/analyze` acepta TAL CUAL del cliente (`playground.py`, canal C5).
# Reproducido exactamente: `architecture_hints="bad"` → `AttributeError`
# sin capturar (HTTP 500, no un error controlado); `hidden_layers=[("bad",
# "relu")]` → `TypeError` al comparar unidades; `hidden_layers=[(64,
# "relu")]*100` → aceptado sin más, 101 capas Dense generadas, muy por
# encima del tope de 12 capas / 16384 unidades que exige el propio C5. Las
# únicas activaciones que este generador produce para capas OCULTAS en
# cualquier camino (prompt/LLM/default) son "relu" — ver `_parse_layers`
# en `intent_llm.py` y `_default_hidden_layers`/`_hidden_layers_for_depth`
# aquí mismo — así que es el único valor aceptado desde la entrada pública.
_ALLOWED_HIDDEN_ACTIVATIONS = {"relu"}


def validate_architecture_hints(hints: Any) -> tuple[dict[str, Any], str | None]:
    """Valida `architecture_hints` (canal público C5) ANTES de que
    `analyze_playground_request` lo toque — a diferencia de
    `sanitize_hidden_layers` (que asume una forma ya correcta, propuesta
    por el propio core), esta función es la primera línea de defensa contra
    un payload arbitrario de `/api/analyze`.

    Devuelve `({}, None)` si `hints` está ausente/vacío, `(hints_limpios,
    None)` si es válido (unidades coeridas a `int` normal, nunca `bool`), o
    `({}, mensaje)` si no lo es — el caller debe tratar el segundo caso como
    un error controlado (`{"ok": False, "error": mensaje}`), nunca dejar
    que la excepción de más abajo llegue sin capturar al handler HTTP.

    Reauditoría C5 [BAJA]: `not hints` trataba CUALQUIER valor falsy como
    "ausente" — `""`, `[]`, `0`, `False` pasaban con `ok=true` en vez de
    rechazarse por tipo, contradiciendo la validación estricta de más abajo
    (que si exige `isinstance(hints, dict)`). Solo `None` y `{}` son
    "vacío" de verdad; cualquier otro valor falsy pero de tipo incorrecto
    debe rechazarse igual que un valor truthy del tipo incorrecto (p.ej.
    `"bad"`)."""
    if hints is None or hints == {}:
        return {}, None
    if not isinstance(hints, dict):
        return {}, "architecture_hints debe ser un objeto (diccionario)."
    unknown = set(hints) - {"hidden_layers"}
    if unknown:
        return {}, f"architecture_hints: claves no reconocidas {sorted(unknown)!r} (solo se admite 'hidden_layers')."
    if "hidden_layers" not in hints:
        return {}, None
    layers = hints["hidden_layers"]
    if not isinstance(layers, list) or not layers:
        return {}, "architecture_hints.hidden_layers debe ser una lista no vacía de [unidades, activación]."
    if len(layers) > DenseNetworkGenerator._MAX_EXPLICIT_DEPTH:
        return {}, (
            "architecture_hints.hidden_layers supera la profundidad máxima "
            f"({DenseNetworkGenerator._MAX_EXPLICIT_DEPTH} capas)."
        )
    cleaned: list[tuple[int, str]] = []
    for i, item in enumerate(layers):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return {}, f"architecture_hints.hidden_layers[{i}] debe ser un par [unidades, activación]."
        units, activation = item
        if isinstance(units, bool) or not isinstance(units, int):
            return {}, f"architecture_hints.hidden_layers[{i}]: unidades debe ser un entero."
        if not (1 <= units <= DenseNetworkGenerator._MAX_EXPLICIT_WIDTH):
            return {}, (
                f"architecture_hints.hidden_layers[{i}]: unidades fuera de rango "
                f"(1..{DenseNetworkGenerator._MAX_EXPLICIT_WIDTH})."
            )
        if activation not in _ALLOWED_HIDDEN_ACTIVATIONS:
            return {}, (
                f"architecture_hints.hidden_layers[{i}]: activación no permitida "
                f"{activation!r} (solo {sorted(_ALLOWED_HIDDEN_ACTIVATIONS)!r})."
            )
        cleaned.append((int(units), activation))
    return {"hidden_layers": cleaned}, None


def _default_hidden_layers(input_dim: int) -> list[tuple[int, str]]:
    if input_dim <= 4:
        return [(32, "relu"), (16, "relu")]
    if input_dim <= 10:
        return [(64, "relu"), (32, "relu"), (16, "relu")]
    return [(128, "relu"), (64, "relu"), (32, "relu")]


def _hidden_layers_for_depth(n: int, input_dim: int) -> list[tuple[int, str]]:
    """Generate exactly n hidden layers with a tapering unit schedule."""
    base = 64 if input_dim <= 4 else (128 if input_dim <= 10 else 256)
    return [(max(16, base >> (i // 2)), "relu") for i in range(n)]


def _default_labels(task: str) -> list[str]:
    if task == "binary":
        return ["negative", "positive"]
    if task == "multiclass":
        return ["class_a", "class_b", "class_c"]
    return []


def _default_fields() -> list[str]:
    return ["feature_1", "feature_2", "feature_3", "feature_4"]


def _expanded_field_order(fields: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Field order after one-hot expansion: each categorical field replaced, in
    place, by its ordered one-hot columns (from ExpansionResult.groups)."""
    out: list[str] = []
    for f in fields:
        out.extend(groups[f] if f in groups else [f])
    return out


def resolve_task_and_labels(dg, prompt, labels):
    """Task + label resolution shared by the dense AND composite generators
    (invariant 5: same policy in both paths). Returns (task, labels, warnings).

    GEN C4 (+audit): an explicit `ProbabilityMap[...]`/`Label[...]` bracket in the
    prompt with >=2 labels is a DECLARED OUTPUT TYPE and always wins (invariant 1):
    - over caller/LLM ``labels`` (they are ignored, with a warning if they differ);
    - over task keyword detection ("predecir precio ... ProbabilityMap[A,B,C]" is
      a classifier, not a regressor — before the audit the extracted labels were
      silently dropped and the output stayed linear/mse);
    - with EXACTLY 2 labels this means 2-class softmax + cross_entropy +
      Label[...] target, never the 1-unit sigmoid. Reusing "multiclass" for n=2
      is deliberate: `_output_config`/`_dataset_target_type` already handle any
      label count generically, so no separate "labeled binary" task is needed.
    The max_labels limit still applies to a declared bracket, but with an explicit
    warning — never a silent truncation of an output the user spelled out.

    Without such a bracket, resolution is unchanged (retrocompat): caller labels,
    then prose labels, then task keywords; a bare "clasificación binaria" prompt
    still gets the 1-unit sigmoid.
    """
    warnings: list[str] = []
    bracket = dg._extract_bracket_labels(prompt)
    if len(bracket) >= 2:
        m_labels = _limits.get_limit("max_labels")
        if m_labels is not None and len(bracket) > m_labels:
            warnings.append(
                f"El prompt declara {len(bracket)} labels explícitos pero el límite "
                f"max_labels={m_labels} los recorta a {bracket[:m_labels]} "
                f"(sube MATRIXAI_MAX_LABELS o el perfil de límites para conservarlos)."
            )
            bracket = bracket[:m_labels]
        if len(bracket) >= 2:
            if labels is not None and [_identifier(str(l)) for l in labels] != bracket:
                warnings.append(
                    f"labels={list(labels)} ignorados: el prompt declara explícitamente "
                    f"ProbabilityMap/Label{bracket} y el prompt gana (invariante 1)."
                )
            return "multiclass", bracket, warnings
    task = dg._detect_task(prompt, labels)
    resolved_labels = list(labels or dg._extract_labels(prompt) or _default_labels(task))
    return task, resolved_labels, warnings


def resolve_prompt_fields(dg, prompt, input_fields):
    """Field resolution + C3 metadata shared by the dense AND composite generators.

    Honors the prompt's explicit type declarations (invariants 1 & 5): the typed
    fields always survive (even when a caller passes input_fields), clean names come
    from parse_field_specs (not the mangling _extract_fields), and bare untyped
    fields are added from the prompt with the typed declarations stripped.

    GEN C5: caller/LLM ``input_fields`` are sanitized here — a name that is not a
    valid identifier ("customer age") would be written verbatim into the .mxai
    VECTOR and crash the parser downstream, so it is normalized with `_identifier`
    (or dropped if nothing survives), with a warning. Valid names pass verbatim.

    Returns ``(resolved_fields, specs_by_name, field_ranges, field_types, warnings)``.
    field_ranges/field_types are METADATA only (never written into the .mxai VECTOR;
    training data is [0,1]-normalized — see GENERACION_TIPOS_PROMPT_CONTRACT.md C3).
    """
    parsed = parse_field_specs(prompt)
    specs_by_name = parsed.by_name()
    warnings = list(parsed.warnings)
    typed_names = [f.name for f in parsed.fields]
    # SECUENCIAS_PRODUCTO C2 (auditoría [MEDIA]): collapsing ALL whitespace
    # (incl. newlines) destroyed the `\n` boundary _FIELD_RE relies on to stop
    # capturing ("variables: edad, ingreso\nOUTPUT clase: ProbabilityMap[...]"
    # swallowed the OUTPUT line into the field list and lost "edad, ingreso"
    # entirely). Normalize horizontal whitespace PER LINE, keep line breaks.
    bare_clean = "\n".join(" ".join(line.split()) for line in strip_field_specs(prompt).split("\n"))
    bare_names = [n for n in (dg._extract_fields(bare_clean) or []) if n not in specs_by_name]
    caller_fields: list[str] = []
    for raw in (input_fields or []):
        name = str(raw)
        if not _VALID_FIELD_NAME_RE.fullmatch(name):
            fixed = _identifier(name)
            if not fixed:
                warnings.append(
                    f"input_fields: nombre inválido {name!r} descartado "
                    "(no queda un identificador utilizable)."
                )
                continue
            warnings.append(
                f"input_fields: nombre inválido {name!r} normalizado a '{fixed}' "
                "(el nombre crudo rompería el .mxai)."
            )
            name = fixed
        caller_fields.append(name)
    resolved_fields: list[str] = []
    for n in caller_fields + typed_names + bare_names:
        if n not in resolved_fields:
            resolved_fields.append(n)
    if not resolved_fields:
        resolved_fields = list(dg._extract_fields(" ".join(prompt.split())) or _default_fields())

    field_ranges: dict[str, tuple[float, float]] = {
        name: specs_by_name[name].range
        for name in resolved_fields
        if name in specs_by_name and specs_by_name[name].kind == "scalar"
        and specs_by_name[name].range is not None
    }
    field_types: dict[str, str] = {}
    for name in resolved_fields:
        spec = specs_by_name.get(name)
        if spec is None:
            continue
        if spec.kind == "boolean":
            field_types[name] = "boolean"
        elif spec.kind == "scalar" and spec.integer:
            field_types[name] = "integer"
    return resolved_fields, specs_by_name, field_ranges, field_types, warnings


def _default_network_name(task: str) -> str:
    return {"regression": "Regressor", "binary": "BinaryClassifier", "multiclass": "Classifier"}[task]


def _output_name(activation: str) -> str:
    return {"linear": "predicted_value", "sigmoid": "predicted_prob", "softmax": "predicted_class"}.get(activation, "output")


def _build_mxai_text(
    network_name: str,
    input_name: str,
    fields: list[str],
    hidden_layers: list[tuple[int, str]],
    output_units: int,
    output_activation: str,
    output_type: str,
) -> str:
    field_lines = "\n".join(f"  {f}: Scalar" for f in fields)
    layer_lines = "\n".join(f"  LAYER Dense units={u} activation={a}" for u, a in hidden_layers)
    layer_lines += f"\n  LAYER Dense units={output_units} activation={output_activation}"
    out_name = _output_name(output_activation)
    return (
        f"PROJECT {network_name}Project\n\n"
        f"VECTOR {input_name}[{len(fields)}]\n{field_lines}\nEND\n\n"
        f"NETWORK {network_name}\n"
        f"  INPUT {input_name}\n"
        f"{layer_lines}\n"
        f"  OUTPUT {out_name}: {output_type}\n"
        f"END\n\n"
        f"GRAPH\n"
        f"  {input_name} -> {network_name}\n"
        f"END\n\n"
        f"AUDIT\n"
        f"  EXPLAIN {input_name} -> {network_name}\n"
        f"END\n"
    )


def extract_epochs_from_prompt(prompt: str) -> int:
    """EPOCHS from the prompt (`epochs=300`, `300 epocas`), capped at the sanity
    ceiling; default when absent. Shared by the dense and composite generators."""
    m = DenseNetworkGenerator._EPOCHS_RE.search(_norm(prompt))
    if m:
        n = int(m.group(1) or m.group(2))
        return max(1, min(n, DenseNetworkGenerator._MAX_EPOCHS))
    return DenseNetworkGenerator._DEFAULT_EPOCHS


def extract_early_stop_from_prompt(prompt: str) -> tuple[int, str] | None:
    """(patience, metric) from `early_stop patience=20 metric=validation_loss`."""
    m = DenseNetworkGenerator._EARLY_STOP_RE.search(_norm(prompt))
    if m:
        return (max(1, int(m.group(1))), m.group(2) or "validation_loss")
    return None


def _build_training_text(
    network_name: str,
    input_name: str,
    fields: list[str],
    output_name: str,
    dataset_target_type: str,
    loss_type: str,
    epochs: int = 50,
    early_stop: tuple[int, str] | None = None,
) -> str:
    field_list = "[" + ", ".join(fields) + "]"
    loss_name = f"{network_name}Loss"
    optimizer_name = f"{network_name}Optimizer"
    lines = [
        f"MODEL {network_name}Project.mxai",
        "",
        f"DATASET {network_name}TrainingSet",
        f'  SOURCE csv("{network_name.lower()}.train.csv")',
        f"  INPUT {input_name} FROM COLUMNS {field_list}",
        f"  TARGET {output_name}: {dataset_target_type}",
        "  SPLIT train=0.8 validation=0.2 seed=42",
        "  BATCH size=8",
        "END",
        "",
        f"LOSS {loss_name}",
        f"  TYPE {loss_type}",
        f"  PREDICTION {network_name}",
        f"  TARGET {output_name}",
        "END",
        "",
        f"OPTIMIZER {optimizer_name}",
        "  TYPE sgd",
        "  LEARNING_RATE 0.01",
        f"  UPDATE {network_name}.*",
        "END",
        "",
        "RUN",
        f"  EPOCHS {epochs}",
    ]
    if early_stop is not None:
        patience, metric = early_stop
        lines.append(f"  EARLY_STOP patience={patience} metric={metric}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _norm(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


# GEN C5: what the .mxai parser accepts as a VECTOR field name. Anything else
# written verbatim into the VECTOR block raises MatrixAIParseError downstream.
_VALID_FIELD_NAME_RE = re.compile(r"[A-Za-z_]\w*")


def _identifier(value: str) -> str:
    text = _norm(value).strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text if text and not text[0].isdigit() else ""


def _titlecase(value: str) -> str:
    text = _norm(value).strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:1].upper() + text[1:] if text else ""


def _any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)
