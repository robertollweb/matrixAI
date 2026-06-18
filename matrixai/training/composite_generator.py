# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P19 C10 — CompositeNetworkGenerator: genera composite networks desde intención humana."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai.training.dense_generator import (
    DenseNetworkGenerator,
    _any,
    _build_training_text,
    extract_epochs_from_prompt,
    extract_early_stop_from_prompt,
    _default_fields,
    _default_hidden_layers,
    _default_labels,
    _default_network_name,
    _identifier,
    _norm,
    _output_config,
    _output_name,
    _titlecase,
    _dataset_target_type,
)


class CompositeNetworkGeneratorError(ValueError):
    pass


@dataclass(frozen=True)
class CompositeNetworkGenerationResult:
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
    is_composite: bool
    embeddings: list[dict]
    blocks: list[dict]
    is_sequence: bool
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompositeNetworkGenerator:
    _COMPLEX_KEYWORDS = [
        "complejo", "complex", "no lineal", "nonlinear",
        "expresiv", "expressive", "preciso", "precise",
    ]
    # Aligned with playground._SEQUENCE_HINTS so a tabular composite (e.g. with
    # embeddings) is NOT mislabelled as a sequence + given a spurious POOL. Bare
    # "serie"/"series"/"seq" were false positives (Spanish "una serie de…", etc.);
    # genuine sequences need explicit time-series wording.
    _SEQUENCE_KEYWORDS = [
        "secuencia", "sequence", "serie temporal", "series temporales",
        "time series", "temporal", "recurren",
    ]
    _CAT_KEYWORDS = [
        "categ", "tipo", "genre", "gender", "rol", "region",
        "clase", "estado", "status", "kind", "type",
    ]
    _DEFAULT_AUTO_VOCAB = 50

    def generate(
        self,
        prompt: str,
        *,
        input_fields: list[str] | None = None,
        labels: list[str] | None = None,
        network_name: str | None = None,
        input_name: str | None = None,
        categorical_fields: dict[str, int] | None = None,
        force_residual: bool = False,
        force_dense: bool = False,
    ) -> CompositeNetworkGenerationResult:
        clean = " ".join(prompt.strip().split())
        if not clean:
            raise CompositeNetworkGeneratorError(
                "CompositeNetworkGenerator requires a non-empty prompt"
            )

        _dg = DenseNetworkGenerator()
        task = _dg._detect_task(clean, labels)
        resolved_labels = list(labels or _dg._extract_labels(clean) or _default_labels(task))
        resolved_fields = list(input_fields or _dg._extract_fields(clean) or _default_fields())
        input_dim = len(resolved_fields)
        resolved_name = network_name or _dg._extract_name(clean) or _default_network_name(task)
        resolved_entity = input_name or _dg._extract_entity(clean) or "Input"

        output_activation, output_type, output_units, loss_type = _output_config(task, resolved_labels)
        resolved_hidden = _default_hidden_layers(input_dim)
        out_name = _output_name(output_activation)

        text = _norm(clean).lower()

        is_sequence = not force_dense and _any(text, self._SEQUENCE_KEYWORDS)
        has_complex_kw = _any(text, self._COMPLEX_KEYWORDS)
        wants_residual = not force_dense and (force_residual or (has_complex_kw and input_dim >= 6))

        cat_fields_dict: dict[str, int] = {}
        if not force_dense:
            if categorical_fields is not None:
                cat_fields_dict = {
                    f: v for f, v in categorical_fields.items() if v > 5
                }
            else:
                for f in resolved_fields:
                    if any(kw in f.lower() for kw in self._CAT_KEYWORDS):
                        cat_fields_dict[f] = self._DEFAULT_AUTO_VOCAB

        is_composite = bool(cat_fields_dict or wants_residual or is_sequence)

        emb_infos: list[dict] = []
        block_infos: list[dict] = []

        if is_composite:
            for f, vocab in cat_fields_dict.items():
                dim = min(8, math.ceil(math.sqrt(vocab)))
                base = f[: -3] if f.endswith("_id") else f
                emb_name = f"{base}_emb"
                emb_infos.append({"field": f, "vocab": vocab, "dim": dim, "emb_name": emb_name})

            if wants_residual:
                block_size = max(resolved_hidden[0][0] if resolved_hidden else 16, 16)
                block_infos.append({
                    "name": "res1",
                    "units": block_size,
                    "has_layernorm": True,
                    "dropout_rate": 0.2,
                })

            mxai_text = _build_composite_mxai(
                resolved_name, resolved_entity, resolved_fields,
                emb_infos, block_infos, is_sequence,
                resolved_hidden, output_units, output_activation, output_type,
                cat_fields_dict,
            )
        else:
            mxai_text = _build_dense_mxai(
                resolved_name, resolved_entity, resolved_fields,
                resolved_hidden, output_units, output_activation, output_type,
            )

        # Honour epochs / early_stop from the prompt (like the dense generator);
        # default when absent. The user controls their machine (downloadable Studio).
        training_text = _build_training_text(
            resolved_name, resolved_entity, resolved_fields,
            out_name, _dataset_target_type(task, resolved_labels if task == "multiclass" else None), loss_type,
            epochs=extract_epochs_from_prompt(clean),
            early_stop=extract_early_stop_from_prompt(clean),
        )
        # Include a dummy data row so the TrainingVerifier sees ≥1 row (mirrors the
        # dense generator). Categorical/embedding fields take an integer index (0);
        # scalars take 0.0; the target takes the first label (multiclass) or 0.0.
        dummy_target = resolved_labels[0] if task == "multiclass" else "0.0"
        dummy_values = ["0" if f in cat_fields_dict else "0.0" for f in resolved_fields] + [dummy_target]
        dataset_template_text = (
            ",".join(resolved_fields + [out_name]) + "\n" + ",".join(dummy_values) + "\n"
        )

        assumptions = [
            f"CompositeNetworkGenerator inferred task={task}",
            f"is_composite={is_composite}",
            f"input_dim={input_dim} from {len(resolved_fields)} fields",
            f"embeddings={len(emb_infos)}, blocks={len(block_infos)}",
            f"loss={loss_type}, output_activation={output_activation}",
            "Architecture is a heuristic — tune for production",
        ]
        warnings: list[str] = []
        if task == "multiclass" and len(resolved_labels) < 2:
            warnings.append("multiclass task requires at least 2 labels — using defaults")

        return CompositeNetworkGenerationResult(
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
            is_composite=is_composite,
            embeddings=emb_infos,
            blocks=block_infos,
            is_sequence=is_sequence,
            assumptions=assumptions,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# mxai text builders
# ---------------------------------------------------------------------------

def _build_composite_mxai(
    network_name: str,
    input_name: str,
    fields: list[str],
    emb_infos: list[dict],
    block_infos: list[dict],
    is_sequence: bool,
    hidden_layers: list[tuple[int, str]],
    output_units: int,
    output_activation: str,
    output_type: str,
    cat_fields: dict[str, int],
) -> str:
    scalar_fields = [f for f in fields if f not in cat_fields]

    field_lines: list[str] = []
    for f in fields:
        if f in cat_fields:
            # Embedding index range is 0..vocab-1 (inclusive); the table has `vocab` rows.
            field_lines.append(f"  {f}: Integer[0, {max(cat_fields[f] - 1, 0)}]")
        else:
            field_lines.append(f"  {f}: Scalar")

    net_lines: list[str] = [f"NETWORK {network_name}", f"  INPUT {input_name}"]

    emb_names: list[str] = []
    for emb in emb_infos:
        net_lines.append(
            f"  EMBEDDING {emb['emb_name']} FROM {emb['field']} VOCAB {emb['vocab']} DIM {emb['dim']}"
        )
        emb_names.append(emb["emb_name"])

    if emb_names:
        concat_items = emb_names + scalar_fields
        net_lines.append(f"  CONCAT [{', '.join(concat_items)}] -> features")

    if is_sequence:
        net_lines.append("  POOL mean")

    pre_block = hidden_layers[:1] if hidden_layers and block_infos else hidden_layers
    for units, activation in pre_block:
        net_lines.append(f"  LAYER Dense units={units} activation={activation}")

    for block in block_infos:
        net_lines.append(f"  BLOCK {block['name']}")
        net_lines.append(f"    LAYER Dense units={block['units']} activation=relu")
        if block["has_layernorm"]:
            net_lines.append("    LAYER LayerNorm")
        if block["dropout_rate"] > 0:
            net_lines.append(f"    LAYER Dropout rate={block['dropout_rate']}")
        net_lines.append("    RESIDUAL FROM PREVIOUS")
        net_lines.append("  END")

    if block_infos and len(hidden_layers) > 1:
        for units, activation in hidden_layers[1:]:
            net_lines.append(f"  LAYER Dense units={units} activation={activation}")

    net_lines.append(f"  LAYER Dense units={output_units} activation={output_activation}")
    out_name = _output_name(output_activation)
    net_lines.append(f"  OUTPUT {out_name}: {output_type}")
    net_lines.append("END")

    return (
        f"PROJECT {network_name}Project\n\n"
        f"VECTOR {input_name}[{len(fields)}]\n"
        + "\n".join(field_lines) + "\n"
        + "END\n\n"
        + "\n".join(net_lines) + "\n\n"
        + f"GRAPH\n"
        + f"  {input_name} -> {network_name}\n"
        + "END\n"
    )


def _build_dense_mxai(
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
        f"END\n"
    )
