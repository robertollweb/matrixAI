# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Build inference_spec.json — the "tokenizer" of an exported MatrixAI bundle.

Tabular ONNX graphs expect a flat float32 vector in `vector.fields` order;
transformer graphs expect fixed-length int64 token IDs plus a padding mask. A
downloaded model is unusable unless that encoding contract travels with it.

`build_inference_spec` derives that contract from the program + the ONNX export
result, combined with the optional Studio-side metadata (field_ranges,
field_categories, field_types, labels). The resulting dict is written verbatim
as inference_spec.json and consumed by the standalone predict.py (C2).

Scope: one flat VECTOR input, or one C5 transformer SEQUENCE input. Other
multi-input models still fail explicitly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from matrixai.ir import MatrixAIProgram
from matrixai.parameters.store import ParameterSet
from matrixai.export.onnx_exporter import OnnxExportResult
from matrixai.training.categorical import _build_group_names

SPEC_VERSION = 1
_NORMALIZE_NOTE = "Inputs are normalized internally by predict.py. Feed raw human values."


class InferenceSpecError(ValueError):
    """Raised when a program cannot be described by a single-VECTOR inference spec."""


def build_inference_spec(
    program: MatrixAIProgram,
    parameter_set: ParameterSet,
    export_result: OnnxExportResult,
    *,
    field_ranges: dict[str, tuple[float, float]] | None = None,
    field_categories: dict[str, list[str]] | None = None,
    field_types: dict[str, str] | None = None,
    labels: list[str] | None = None,
    example_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the inference_spec.json payload for an exported model.

    All metadata args are optional: when absent, encodings fall back to scalar
    ranges and Boolean/Integer types declared in the .mxai, then to ``scalar01``.
    ``field_categories`` and the network embeddings drive one-hot / embedding-index
    encodings; embedding vocab also falls back to the .mxai type args.

    ``example_input`` (a raw record), if given, is validated against the resolved
    fields so a malformed example fails at export time rather than for the consumer.
    """
    # TRANSFORMER C5: entrada SEQUENCE (token ids). El contrato B elevará esta
    # spec a texto crudo con el tokenizador embebido; en C5 el consumidor pasa
    # la lista [L] de enteros directamente.
    sequences = list(getattr(program, "sequences", []) or [])
    if sequences:
        return _build_sequence_inference_spec(
            program, parameter_set, export_result,
            labels=labels, example_input=example_input,
        )
    _guard_single_vector(program)
    vector = program.vectors[0]
    input_order = list(vector.fields)
    order_set = set(input_order)

    field_ranges = field_ranges or {}
    field_categories = field_categories or {}
    field_types = field_types or {}

    # Embedding source field -> vocab size, from composite NETWORK specs.
    embeddings: dict[str, int] = {}
    for net in program.networks:
        for emb in getattr(net, "embeddings", []):
            embeddings[emb.source] = emb.vocab

    fields: dict[str, Any] = {}
    consumed: set[str] = set()  # one-hot columns already described by their group

    # 1. One-hot groups: a categorical whose expanded columns live in input_order.
    #    Reuse the exact training-time column naming so raw->column is authoritative,
    #    never reconstructed by heuristics in the consumer.
    #    GEN C6 (invariante 6): the metadata must MATCH the model — a group that is
    #    partial, missing, or left as a raw scalar used to be swallowed silently
    #    (its columns fell back to scalar01, or some categories became unreachable),
    #    producing a bundle that contradicts what the consumer was told. Fail the
    #    spec loudly instead (the bundle machinery surfaces the reason).
    for group, values in field_categories.items():
        if group in embeddings:
            continue  # embedding-encoded categorical: handled (and checked) below
        values = list(values)
        if len(values) < 2:
            raise InferenceSpecError(
                f"field_categories for {group!r} declares {len(values)} value(s); a "
                "categorical needs at least 2. The metadata is malformed."
            )
        col_names = _build_group_names(group, values)
        present = [col for col in col_names if col in order_set]
        if not present:
            if group in order_set:
                raise InferenceSpecError(
                    f"field_categories declares {len(values)} categories for {group!r} "
                    "but the model kept it as a plain scalar column (neither one-hot "
                    "columns nor an embedding). Metadata and model contradict each "
                    "other; re-generate the dataset/model with these categories or "
                    "drop the category metadata."
                )
            raise InferenceSpecError(
                f"field_categories references {group!r} but the model has neither that "
                f"column nor its one-hot columns ({col_names[0]}, …). This metadata "
                "does not belong to this model."
            )
        if len(present) != len(col_names):
            missing = [c for c in col_names if c not in order_set]
            raise InferenceSpecError(
                f"field_categories for {group!r} declares {len(col_names)} one-hot "
                f"columns but the model is missing {missing}. Vocabulary and model "
                "disagree — some categories would be unreachable. Re-generate the "
                "dataset/model with the same category values."
            )
        pairs = [{"raw": raw, "column": col} for raw, col in zip(values, col_names)]
        entry: dict[str, Any] = {"encoding": "one_hot", "values": pairs}
        _annotate_type(entry, _resolve_field_type(group, field_types, vector))
        fields[group] = entry
        consumed.update(p["column"] for p in pairs)

    # 2. Remaining columns: embedding index, scalar with range, or scalar01.
    for col in input_order:
        if col in consumed or col in fields:
            continue
        if col in embeddings:
            entry = {"encoding": "embedding_index", "column": col}
            # Human vocab from the Studio metadata, else from the .mxai type args.
            declared = field_categories.get(col)
            if declared is not None and len(declared) != int(embeddings[col]):
                # GEN C6 (invariante 6): a vocab that disagrees with the embedding
                # table size would index out of range (or leave rows unreachable).
                raise InferenceSpecError(
                    f"field_categories for {col!r} has {len(declared)} values but the "
                    f"embedding table has {int(embeddings[col])} rows. Metadata and "
                    "model disagree; re-generate/re-train with the same vocabulary."
                )
            human_vocab = declared or _program_vocab(vector, col)
            if human_vocab:
                entry["vocab"] = list(human_vocab)
            else:
                entry["vocab_size"] = int(embeddings[col])
            _annotate_type(entry, _resolve_field_type(col, field_types, vector))
            fields[col] = entry
            continue
        rng = field_ranges.get(col) or _program_range(vector, col)
        type_name = _resolve_field_type(col, field_types, vector)
        if rng is not None:
            if type_name == "boolean":
                raise InferenceSpecError(
                    f"field_ranges for {col!r} contradicts boolean type metadata. "
                    "Booleans are encoded as 0/1 flags, not normalized over a scalar range."
                )
            entry = {"encoding": "scalar", "range": [_num(rng[0]), _num(rng[1])]}
        else:
            entry = {"encoding": "scalar01"}
        _annotate_type(entry, type_name)
        fields[col] = entry

    _validate_explicit_metadata(fields, field_ranges, field_types)

    spec: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrixai_version": _matrixai_version(),
        "model_hash": parameter_set.model_hash,
        "parameter_schema_hash": parameter_set.parameter_schema_hash,
        "parameter_set_id": parameter_set.parameter_set_id,
        "onnx_opset": export_result.opset_version,
        "onnx_file": "model.onnx",
        "input_name": export_result.input_name,
        "input_shape": list(export_result.input_shape),
        "input_order": input_order,
        "fields": fields,
        "output": _build_output(program, export_result, labels),
        "notes": _NORMALIZE_NOTE,
    }
    if example_input is not None:
        _validate_example_input(example_input, fields)
    return spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_explicit_metadata(
    fields: dict[str, Any],
    field_ranges: dict[str, tuple[float, float]],
    field_types: dict[str, str],
) -> None:
    """GEN C6 invariant 6: explicit metadata must describe this model.

    Categories already fail while their encodings are built. Ranges and semantic
    types need the same hard boundary: unknown metadata, ranges on categorical
    encodings, or boolean+range combinations would make predict.py normalize the
    wrong raw value while pretending the bundle is self-usable.
    """
    for name in field_ranges:
        entry = fields.get(name)
        if entry is None:
            raise InferenceSpecError(
                f"field_ranges references {name!r} but the model has no such raw "
                "input field. This metadata does not belong to this model."
            )
        if entry.get("encoding") != "scalar":
            raise InferenceSpecError(
                f"field_ranges for {name!r} targets a {entry.get('encoding')!r} "
                "field. Only scalar input fields can have numeric ranges."
            )
        if entry.get("type") == "boolean":
            raise InferenceSpecError(
                f"field_ranges for {name!r} contradicts boolean type metadata. "
                "Booleans are encoded as 0/1 flags, not normalized over a scalar range."
            )

    for name, type_name in field_types.items():
        entry = fields.get(name)
        if entry is None:
            raise InferenceSpecError(
                f"field_types references {name!r} but the model has no such raw "
                "input field. This metadata does not belong to this model."
            )
        if entry.get("encoding") not in {"scalar", "scalar01"}:
            raise InferenceSpecError(
                f"field_types[{name!r}]={type_name!r} targets a "
                f"{entry.get('encoding')!r} field. Semantic number/integer/boolean "
                "types only apply to scalar input fields."
            )
        if type_name == "boolean" and entry.get("encoding") == "scalar":
            raise InferenceSpecError(
                f"field_types[{name!r}]='boolean' contradicts a scalar range. "
                "Booleans are encoded as 0/1 flags, not normalized over a range."
            )


def build_example_input(spec: dict[str, Any]) -> dict[str, Any]:
    """A minimal, safe RAW example record compatible with the spec.

    Used to seed example_input.json / expected_output.json. Never contains real
    dataset rows: midpoints for scalars, first category for one-hot/embedding.
    """
    example: dict[str, Any] = {}
    # TRANSFORMER C5 audit: sequence specs do not have ``fields``.  Generate
    # the same raw record accepted by the bundled predict.py instead of the
    # empty object that made its packaging smoke-test meaningless.
    for name, entry in spec.get("input", {}).items():
        if entry.get("encoding") == "token_ids":
            example[name] = [0] * int(entry["length"])
    for field, entry in spec.get("fields", {}).items():
        enc = entry["encoding"]
        if enc == "scalar":
            lo, hi = entry["range"]
            mid = (float(lo) + float(hi)) / 2.0
            example[field] = int(round(mid)) if entry.get("type") == "integer" else mid
        elif enc == "scalar01":
            example[field] = False if entry.get("type") == "boolean" else 0.5
        elif enc == "one_hot":
            example[field] = entry["values"][0]["raw"]
        elif enc == "embedding_index":
            example[field] = entry["vocab"][0] if "vocab" in entry else 0
    return example


def _build_sequence_inference_spec(
    program: MatrixAIProgram,
    parameter_set: ParameterSet,
    export_result: "OnnxExportResult",
    *,
    labels: list[str] | None = None,
    example_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """TRANSFORMER C5 — spec de inferencia para entrada SEQUENCE.

    input: {"encoding": "token_ids", "length": L, "vocab_size": V} (literal del
    contrato 51 §C5). El grafo ONNX expone ademas una segunda entrada
    `<sequence>_mask` (1.0 real / 0.0 padding); mask_input la documenta y el
    consumidor puede pasar todo-unos si no hay padding. POOL cls exige que la
    posicion 0 sea real (invariante 1c) — anotado en la spec.
    """
    if len(program.sequences) > 1:
        raise InferenceSpecError(
            f"Model has {len(program.sequences)} SEQUENCE inputs; the transformer "
            "bundle supports exactly one."
        )
    seq = program.sequences[0]
    net = program.networks[0] if program.networks else None
    if net is None or not getattr(net, "transformer_blocks", []):
        raise InferenceSpecError(
            "SEQUENCE input without a BLOCK TRANSFORMER network has no defined "
            "inference path (TRANSFORMER_BLOQUE covers only the transformer)."
        )

    if example_input is not None:
        value = example_input.get(seq.name)
        if not isinstance(value, list) or len(value) != seq.length:
            raise InferenceSpecError(
                f"example_input[{seq.name!r}] must be a list of {seq.length} token ids"
            )
        bad = [v for v in value if not isinstance(v, int) or v < 0 or v >= seq.vocab_size]
        if bad:
            raise InferenceSpecError(
                f"example_input[{seq.name!r}] contains ids outside [0, {seq.vocab_size}): {bad[:3]}"
            )

    pool_kinds = [
        layer.pool_kind for layer in getattr(net, "top_layers", [])
        if layer.layer_type == "Pool"
    ]
    pool_kind = pool_kinds[0] if pool_kinds else None

    # TRANSFORMER C6: metadata de auditoría del bloque (§"C6 — Auditoría del
    # bloque"). layer_manifest ya trae la arquitectura resuelta calculada por
    # el mismo camino que exporta a ONNX (single source of truth) — un
    # typecheck sucio aquí sería un bug en un caller previo (export ya lo
    # habría rechazado), así que un KeyError es el fallo correcto.
    from matrixai.compiler import BackendContractAnalyzer
    from matrixai.parameters.network_params import transformer_block_export_metadata
    block_entry = next(
        e for e in BackendContractAnalyzer().analyze(program).layer_manifest
        if e.get("layer_type") == "TransformerBlock" and e.get("network") == net.name
    )
    transformer_block = transformer_block_export_metadata(block_entry)

    spec: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrixai_version": _matrixai_version(),
        "model_hash": parameter_set.model_hash,
        "parameter_schema_hash": parameter_set.parameter_schema_hash,
        "parameter_set_id": parameter_set.parameter_set_id,
        "onnx_opset": export_result.opset_version,
        "onnx_file": "model.onnx",
        "input": {
            seq.name: {
                "encoding": "token_ids",
                "length": seq.length,
                "vocab_size": seq.vocab_size,
            }
        },
        "input_order": [seq.name],
        # Keep the common envelope consumed by predict.py.  ``onnx_input`` is
        # retained as the explicit sequence alias for backwards compatibility
        # with the first C5 spec draft.
        "input_name": export_result.input_name,
        "input_shape": list(export_result.input_shape),
        "onnx_input": export_result.input_name,
        "mask_input": f"{export_result.input_name}_mask",
        "mask_shape": list(export_result.input_shape),
        "mask_semantics": "1.0 = real token, 0.0 = padding; all-ones if no padding",
        "pool_kind": pool_kind,
        "transformer_block": transformer_block,
        "output": _build_output(program, export_result, labels),
    }
    if "cls" in pool_kinds:
        spec["notes"] = "POOL cls: position 0 must be a real token (mask[0] = 1.0)"
    return spec


def _guard_single_vector(program: MatrixAIProgram) -> None:
    if getattr(program, "sequences", None):
        raise InferenceSpecError(
            "Model has a SEQUENCE input; predict.py v1 only supports a single flat "
            "VECTOR input. See MEJORAS_FUTURAS_CONTRACT.md (M2 v3) for sequences."
        )
    if not program.vectors:
        raise InferenceSpecError("Model has no VECTOR input; cannot build inference_spec.")
    if len(program.vectors) > 1:
        raise InferenceSpecError(
            f"Model has {len(program.vectors)} VECTOR inputs; predict.py v1 supports "
            "exactly one. Multi-input models are out of scope for this bundle."
        )


def _build_output(
    program: MatrixAIProgram,
    export_result: OnnxExportResult,
    labels: list[str] | None,
) -> dict[str, Any]:
    out_shape = list(export_result.output_shape)
    last = out_shape[-1] if out_shape and out_shape[-1] != -1 else None
    real = _resolve_real_labels(labels, program, export_result)

    output: dict[str, Any] = {"name": export_result.output_name, "shape": out_shape}
    if last is not None and last >= 2:
        # Multi-class softmax: a downloadable model is useless without real class
        # labels (the consumer gets [0.3, 0.7] and cannot tell which class is which).
        # Refuse to invent positional labels — fail in spec construction instead.
        if not real:
            raise InferenceSpecError(
                f"Classification output has {last} classes but no labels: none passed, "
                "none declared as ProbabilityMap[...] in the .mxai, and the export only "
                "carries positional placeholders. A downloadable model must name its "
                "classes; pass labels=[...] (the Studio does this automatically)."
            )
        if len(real) != last:
            raise InferenceSpecError(
                f"Classification output has {last} classes but {len(real)} labels were "
                f"provided: {real}. The label count must match the class count."
            )
        output["kind"] = "classification"
        output["labels"] = real
    elif out_shape == [-1] or last == 1:
        if len(real) == 2:
            output["kind"] = "binary_classification"
            output["labels"] = real
        else:
            # single value with no two-way labelling: a usable probability / score
            output["kind"] = "regression"
    else:
        output["kind"] = "raw_vector"
    return output


def _resolve_real_labels(
    labels: list[str] | None,
    program: MatrixAIProgram,
    export_result: OnnxExportResult,
) -> list[str]:
    """Real (semantic) output labels, or [] when only positional placeholders exist.

    Priority: explicit arg → ProbabilityMap[...] in the .mxai → export labels, but
    only if the export labels are not the exporter's positional fallback ("0".."n-1").
    """
    if labels:
        return [str(x) for x in labels]
    prog = _program_labels(program)
    if prog:
        return prog
    exported = [str(x) for x in (export_result.labels or [])]
    if exported and not _is_positional(exported):
        return exported
    return []


def _is_positional(labels: list[str]) -> bool:
    return labels == [str(i) for i in range(len(labels))]


def _program_labels(program: MatrixAIProgram) -> list[str]:
    """Extract output labels from a NETWORK ProbabilityMap[...] / Label[...] type."""
    for net in program.networks:
        labels = _parse_bracket_labels(getattr(net, "output_type_str", ""))
        if labels:
            return labels
    for fn in program.functions:
        spec = getattr(fn, "output_type", None)
        if spec is not None and spec.name in ("ProbabilityMap", "Label"):
            args = spec.parameters.get("args")
            if isinstance(args, list) and args:
                return [str(a) for a in args]
    return []


def _parse_bracket_labels(type_str: str) -> list[str]:
    if not type_str or "[" not in type_str:
        return []
    head, _, rest = type_str.partition("[")
    if head.strip() not in ("ProbabilityMap", "Label"):
        return []
    inner = rest.rsplit("]", 1)[0]
    return [tok.strip() for tok in inner.split(",") if tok.strip()]


def _program_range(vector: Any, field: str) -> tuple[float, float] | None:
    """Scalar [min, max] declared in the .mxai field type, if both bounds exist."""
    spec = vector.field_types.get(field)
    rng = getattr(spec, "range", None) if spec is not None else None
    if rng is None or rng.minimum is None or rng.maximum is None:
        return None
    return (float(rng.minimum), float(rng.maximum))


def _program_vocab(vector: Any, field: str) -> list[str] | None:
    """Human vocab declared in the .mxai field type args (e.g. Categorical[A, B, C])."""
    spec = vector.field_types.get(field)
    if spec is None:
        return None
    args = spec.parameters.get("args")
    if isinstance(args, list) and args and not all(_is_number(a) for a in args):
        return [str(a) for a in args]
    return None


# S2 semantic field types worth surfacing to the consumer (mirrors playground's set).
_KNOWN_FIELD_TYPES = ("number", "integer", "boolean")


def _resolve_field_type(field: str, field_types: dict[str, str], vector: Any) -> str | None:
    """Boolean/Integer/Number semantics: the explicit arg wins, else from the .mxai type."""
    if field in field_types:
        return field_types[field]
    spec = vector.field_types.get(field)
    if spec is not None:
        name = spec.name.lower()
        if name in _KNOWN_FIELD_TYPES:
            return name
    return None


def _annotate_type(entry: dict[str, Any], type_name: str | None) -> None:
    if type_name:
        entry["type"] = type_name


def _validate_example_input(example_input: Any, fields: dict[str, Any]) -> None:
    """A provided example record must only reference known high-level fields."""
    if not isinstance(example_input, dict):
        raise InferenceSpecError("example_input must be a dict of raw field values.")
    unknown = [k for k in example_input if k not in fields]
    if unknown:
        raise InferenceSpecError(
            f"example_input references unknown fields {sorted(unknown)}. "
            f"Valid fields: {sorted(fields)}."
        )


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _num(value: Any) -> float | int:
    f = float(value)
    return int(f) if f.is_integer() else f


def _matrixai_version() -> str | None:
    try:
        import matrixai
        return getattr(matrixai, "__version__", None)
    except Exception:  # noqa: BLE001
        return None
