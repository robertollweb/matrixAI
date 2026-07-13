# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""SECUENCIAS_PRODUCTO C2 — TransformerNetworkGenerator.

Convierte un campo `Text`/`Text[L]` declarado en el prompt (SECUENCIAS_
PRODUCTO_CONTRACT.md C1) en un composite network con SEQUENCE + EMBEDDING +
BLOCK TRANSFORMER (TRANSFORMER_BLOQUE_CONTRACT.md, contrato A) + POOL mean +
cabeza de clasificación/regresión — la misma gramática que el contrato A ya
audita y entrena, generada aquí a partir de la intención del prompt.

v1 (decisión 3 del contrato): UN campo Text por modelo, sin mezclar con
campos tabulares — cualquier otra combinación es un error accionable, nunca
una degradación silenciosa a tabular.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai import limits as _limits
from matrixai.generation import parse_field_specs, strip_field_specs
from matrixai.text.tokenizer import ByteTokenizer
from matrixai.training.dense_generator import (
    DenseNetworkGenerator,
    _default_network_name,
    _identifier,
    _norm,
    _output_config,
    _output_name,
    _titlecase,
    extract_early_stop_from_prompt,
    extract_epochs_from_prompt,
    resolve_task_and_labels,
)

# Decisión 6 del contrato: defaults del modelo de texto (~0.4M params, entrena
# en CPU modesta). FF nunca se emite explícito: el sentinel del contrato A
# (FF omitido -> 4*dim en typecheck) ya da 256 para DIM=64, y escala solo si
# el prompt sube DIM (M12/M17).
_DEFAULT_LENGTH = 64
_DEFAULT_DIM = 64
_DEFAULT_LAYERS = 2
_DEFAULT_HEADS = 4


class TransformerNetworkGeneratorError(ValueError):
    pass


@dataclass(frozen=True)
class TransformerNetworkGenerationResult:
    prompt: str
    network_name: str
    sequence_name: str
    field_name: str
    length: int
    dim: int
    layers: int
    heads: int
    output_type: str
    output_activation: str
    loss_type: str
    output_units: int
    labels: list[str]
    mxai_text: str
    training_text: str
    dataset_template_text: str
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # SECUENCIAS_PRODUCTO invariante 3: metadata canónica del campo de texto —
    # nunca se pierde en export/import (viaja en inference_spec.json en C5).
    field_types: dict[str, str] = field(default_factory=dict)
    field_seq: dict[str, dict[str, Any]] = field(default_factory=dict)
    # playground.analyze_playground_request lo lee vía getattr para
    # architecture_decision.kind — nunca "residual"/"composite"/"dense".
    is_transformer: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pick_heads(dim: int, preferred: int = _DEFAULT_HEADS) -> int:
    """HEADS debe dividir DIM (invariante del bloque, contrato A). `preferred`
    si divide exacto; si no, el mayor divisor de `dim` que no supere
    `preferred` (1 siempre divide, así que nunca falla)."""
    if dim % preferred == 0:
        return preferred
    for heads in range(preferred, 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class TransformerNetworkGenerator:
    def generate(
        self,
        prompt: str,
        *,
        input_fields: list[str] | None = None,
        labels: list[str] | None = None,
        network_name: str | None = None,
        input_name: str | None = None,
    ) -> TransformerNetworkGenerationResult:
        clean = " ".join(prompt.strip().split())
        if not clean:
            raise TransformerNetworkGeneratorError(
                "TransformerNetworkGenerator requires a non-empty prompt"
            )

        # parse_field_specs necesita las fronteras de línea CRUDAS del prompt
        # (un campo en su propia línea, sin ',' / ';' / ':' precedente, solo
        # delimita bien contra '\n') — igual que resolve_prompt_fields en
        # dense_generator.py, que parsea `prompt`, nunca `clean`.
        parsed = parse_field_specs(prompt)
        text_fields = [f for f in parsed.fields if f.kind == "text"]
        tabular_fields = [f for f in parsed.fields if f.kind != "text"]
        # Auditoría C2 [MEDIA]: una re-declaración contradictoria del mismo
        # campo ("resenas: Scalar" ... "resenas: Text" en otra línea) hacía que
        # el resultado dependiera de qué línea aparece PRIMERO — con Scalar
        # ganando, ni siquiera se sabía que el prompt pedía un campo Text
        # (mensaje genérico "no Text field"). parse_field_specs ya detecta el
        # conflicto en `warnings`; aquí se usa para dar un error específico
        # en vez de una ausencia sin explicar.
        _text_conflicts = [w for w in parsed.warnings if "'text'" in w and "declarado como" in w]
        if not text_fields:
            if _text_conflicts:
                raise TransformerNetworkGeneratorError(
                    "El prompt declara un campo como Text, pero una segunda "
                    f"declaración lo contradice y gana: {_text_conflicts[0]} — "
                    "declara ese campo como Text una sola vez."
                )
            raise TransformerNetworkGeneratorError(
                "TransformerNetworkGenerator requires a Text field declared in the prompt"
            )
        if len(text_fields) > 1:
            raise TransformerNetworkGeneratorError(
                "Solo se admite un campo Text por modelo (v1); declarados: "
                f"{[f.name for f in text_fields]}. Varios campos Text por modelo "
                "queda fuera de alcance de este contrato."
            )
        if tabular_fields:
            raise TransformerNetworkGeneratorError(
                f"Mezclar Text ({text_fields[0].name!r}) con campos tabulares "
                f"({[f.name for f in tabular_fields]!r}) en el mismo modelo no está "
                "soportado (v1, decisión 3 del contrato SECUENCIAS_PRODUCTO): usa un "
                "único campo Text por modelo, o quita el campo Text para un modelo "
                "tabular normal."
            )

        text_field = text_fields[0]
        _dg = DenseNetworkGenerator()

        # Auditoría C2 [MEDIA]: parse_field_specs solo ve declaraciones TIPADAS
        # ("campo: Tipo") — un prompt como "resenas: Text\nvariables: edad,
        # ingreso" mezclaba Text con campos tabulares BARE sin que la
        # comprobación de arriba lo detectara (silenciosamente ignorados).
        # Mismo mecanismo que resolve_prompt_fields (dense/composite): strip
        # las declaraciones tipadas y pasa el resto por el extractor legado.
        bare_clean = "\n".join(
            " ".join(line.split()) for line in strip_field_specs(prompt).split("\n")
        )
        bare_names = [
            n for n in (_dg._extract_fields(bare_clean) or [])
            if n != text_field.name
        ]
        if bare_names:
            raise TransformerNetworkGeneratorError(
                f"Mezclar Text ({text_field.name!r}) con campos tabulares sin tipo "
                f"declarado ({bare_names!r}) en el mismo modelo no está soportado "
                "(v1, decisión 3 del contrato SECUENCIAS_PRODUCTO): usa un único "
                "campo Text por modelo, o quita el campo Text para un modelo "
                "tabular normal."
            )

        # Text ganó (está en text_fields) pero una declaración posterior LO
        # contradice igualmente — mismo caso que arriba, con Text en vez de
        # Scalar como ganador; también debe ser error, no éxito silencioso.
        if _text_conflicts:
            raise TransformerNetworkGeneratorError(
                f"El campo Text {text_field.name!r} tiene declaraciones "
                f"contradictorias en el prompt: {_text_conflicts[0]} — declara "
                f"{text_field.name!r} como Text una sola vez."
            )

        warnings: list[str] = list(parsed.warnings)

        # Invariante GEN 1: el campo Text del prompt gana — un input_fields de
        # caller/LLM que lo contradiga se ignora, con aviso (nunca en silencio).
        if input_fields and list(input_fields) != [text_field.name]:
            warnings.append(
                f"input_fields {list(input_fields)!r} (LLM/caller) ignorado: el "
                f"campo Text {text_field.name!r} del prompt gana (invariante 1)."
            )

        length = text_field.length or _DEFAULT_LENGTH
        # Auditoría C2 [ALTA]: Text[L] no tenía tope — la atención del bloque
        # escala O(L²) y el .mxtrain/CSV generados escalan O(L) por fila
        # (varias veces el texto). Rechazo explícito, no recorte silencioso
        # (invariante GEN: nunca degradar un valor declarado sin decirlo) —
        # a diferencia de LAYERS/DIM (M12/M17), que sí se capan en silencio
        # porque son defaults del GENERADOR, no algo que el usuario tipeó
        # como parte del contrato de datos (Text[L] fija la forma del dataset).
        if _limits.exceeds(length, "max_sequence_length"):
            max_length = _limits.get_limit("max_sequence_length")
            raise TransformerNetworkGeneratorError(
                f"Text[{length}] supera el límite de longitud de secuencia "
                f"({max_length}) — la atención del bloque transformer escala "
                "cuadráticamente con L. Usa un valor menor, o sube el perfil de "
                "límites (MATRIXAI_LIMITS_PROFILE=avanzado; no disponible en hosted)."
            )

        task, resolved_labels, label_warnings = resolve_task_and_labels(_dg, clean, labels)
        warnings.extend(label_warnings)
        output_activation, output_type, output_units, loss_type = _output_config(
            task, resolved_labels
        )
        if task == "multiclass" and len(resolved_labels) < 2:
            warnings.append("multiclass task requires at least 2 labels — using defaults")

        resolved_name = network_name or _dg._extract_name(clean) or _default_network_name(task)
        resolved_seq_name = (
            input_name or _dg._extract_entity(clean) or _titlecase(text_field.name) or "Texto"
        )

        # M12/M17: el prompt puede pedir más capas ("3 capas") / más anchura
        # ("dim=256", "256 unidades") — mismos extractores que el generador
        # denso (invariante 5: misma política determinista en todo el core).
        norm = _norm(clean)
        dim = _DEFAULT_DIM
        width = _dg._extract_width(norm)
        if width is not None:
            dim = width
        layers = _DEFAULT_LAYERS
        depth_match = _dg._DEPTH_RE.search(norm)
        if depth_match:
            layers = _limits.cap(int(depth_match.group(1)), "max_depth")
        heads = _pick_heads(dim)

        vocab_size = ByteTokenizer.VOCAB_SIZE
        mxai_text = _build_transformer_mxai(
            resolved_name, resolved_seq_name, length, vocab_size,
            dim, layers, heads, output_units, output_activation, output_type,
        )
        out_name = _output_name(output_activation)
        dataset_target_type = _dataset_target_type_for(task, resolved_labels)
        epochs = extract_epochs_from_prompt(clean)
        early_stop = extract_early_stop_from_prompt(clean)
        training_text = _build_transformer_training_text(
            resolved_name, resolved_seq_name, length, out_name,
            dataset_target_type, loss_type, epochs, early_stop,
        )
        dataset_template_text = _build_dataset_template(resolved_seq_name, length, out_name, resolved_labels, task)

        field_types = {text_field.name: "text"}
        field_seq = {text_field.name: {"length": length, "tokenizer": "byte_v1"}}

        assumptions = [
            f"TransformerNetworkGenerator inferred task={task}",
            f"Text field {text_field.name!r}: length={length} (byte_v1 tokenizer, "
            f"vocab_size={vocab_size})",
            f"transformer: layers={layers}, heads={heads}, dim={dim}",
            f"loss={loss_type}, output_activation={output_activation}",
            "training_text usa columnas t0..t{L-1} pre-tokenizadas (convención del "
            "contrato A/C4) — el dataset con texto crudo + tokenización en el "
            "boundary de train llega en el corte C3.",
        ]

        return TransformerNetworkGenerationResult(
            prompt=clean,
            network_name=resolved_name,
            sequence_name=resolved_seq_name,
            field_name=text_field.name,
            length=length,
            dim=dim,
            layers=layers,
            heads=heads,
            output_type=output_type,
            output_activation=output_activation,
            loss_type=loss_type,
            output_units=output_units,
            labels=resolved_labels,
            mxai_text=mxai_text,
            training_text=training_text,
            dataset_template_text=dataset_template_text,
            assumptions=assumptions,
            warnings=warnings,
            field_types=field_types,
            field_seq=field_seq,
        )


# ---------------------------------------------------------------------------
# Pure builders
# ---------------------------------------------------------------------------

def _dataset_target_type_for(task: str, labels: list[str]) -> str:
    if task == "regression":
        return "Scalar"
    if task == "binary":
        return "Probability"
    return f"Label[{', '.join(labels)}]"


def _build_transformer_mxai(
    network_name: str,
    seq_name: str,
    length: int,
    vocab_size: int,
    dim: int,
    layers: int,
    heads: int,
    output_units: int,
    output_activation: str,
    output_type: str,
) -> str:
    out_name = _output_name(output_activation)
    return (
        f"PROJECT {network_name}Project\n\n"
        f"SEQUENCE {seq_name}\n"
        f"  length = {length}\n"
        f"  vocab_size = {vocab_size}\n"
        f"END\n\n"
        f"NETWORK {network_name}\n"
        f"  INPUT {seq_name}\n"
        f"  EMBEDDING tok FROM {seq_name} DIM {dim}\n"
        f"  BLOCK enc TRANSFORMER\n"
        f"    LAYERS {layers}\n"
        f"    HEADS {heads}\n"
        f"  END\n"
        f"  POOL mean\n"
        f"  LAYER Dense units={output_units} activation={output_activation}\n"
        f"  OUTPUT {out_name}: {output_type}\n"
        f"END\n\n"
        f"GRAPH\n"
        f"  {seq_name} -> {network_name}\n"
        f"END\n"
    )


def _build_transformer_training_text(
    network_name: str,
    seq_name: str,
    length: int,
    output_name: str,
    dataset_target_type: str,
    loss_type: str,
    epochs: int,
    early_stop: tuple[int, str] | None,
) -> str:
    columns = "[" + ", ".join(f"t{i}" for i in range(length)) + "]"
    loss_name = f"{network_name}Loss"
    optimizer_name = f"{network_name}Optimizer"
    lines = [
        f"MODEL {network_name}Project.mxai",
        "",
        f"DATASET {network_name}TrainingSet",
        f'  SOURCE csv("{network_name.lower()}.train.csv")',
        f"  INPUT {seq_name} FROM COLUMNS {columns}",
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
        # Decisión 5 del contrato A: los transformers reales no convergen bien
        # con SGD plano — adam por defecto (a diferencia del denso, que usa sgd).
        "  TYPE adam",
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


def _build_dataset_template(
    seq_name: str, length: int, output_name: str, labels: list[str], task: str,
) -> str:
    columns = [f"t{i}" for i in range(length)]
    dummy_target = labels[0] if task == "multiclass" else "0.0"
    header = ",".join(columns + [output_name])
    dummy_row = ",".join(["0"] * length + [dummy_target])
    return f"{header}\n{dummy_row}\n"
