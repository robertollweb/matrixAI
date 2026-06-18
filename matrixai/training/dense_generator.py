# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 C10 — DenseNetworkGenerator: genera NetworkSpec y textos .mxai/.mxtrain desde intención humana."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai.ir.schema import DenseLayerSpec, NetworkSpec


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
    _DEPTH_RE = re.compile(
        r"(\d+)\s*(?:capas\s+(?:ocultas?|densas?)|hidden\s+layers?|layers?\s+ocultas?)",
        re.IGNORECASE,
    )
    _MAX_EXPLICIT_DEPTH = 12
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

        task = self._detect_task(clean, labels)
        resolved_labels = list(labels or self._extract_labels(clean) or _default_labels(task))
        resolved_fields = list(input_fields or self._extract_fields(clean) or _default_fields())
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
        header = resolved_fields + [out_name]
        # Binary target type is Probability (numeric) — dummy must be float, not a label string.
        # Multiclass target type is Label — dummy is the first label string.
        dummy_target = resolved_labels[0] if task == "multiclass" else "0.0"
        dummy_values = ["0.0"] * len(resolved_fields) + [dummy_target]
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
        )

    def _extract_hidden_layers(self, prompt: str, input_dim: int) -> list[tuple[int, str]]:
        m = self._DEPTH_RE.search(_norm(prompt))
        if m:
            n = min(int(m.group(1)), self._MAX_EXPLICIT_DEPTH)
            return _hidden_layers_for_depth(n, input_dim)
        return _default_hidden_layers(input_dim)

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
            if _any(text, ["binario", "binary", "dos clases", "two classes"]):
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

    def _extract_labels(self, prompt: str) -> list[str]:
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
        return result[:12] if len(result) >= 2 else []

    def _extract_fields(self, prompt: str) -> list[str]:
        m = self._FIELD_RE.search(prompt)
        if not m:
            return []
        raw = m.group("fields")
        parts = re.split(r",|;|\s+y\s+|\s+and\s+", raw, flags=re.IGNORECASE)
        result = [_identifier(p) for p in parts if _identifier(p)]
        return result if len(result) >= 2 else []

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
