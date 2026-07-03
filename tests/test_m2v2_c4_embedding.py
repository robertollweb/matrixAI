"""M2 v2 — C4: EMBEDDING nativo — dataset por índice entero + generador.

El generador emite `EMBEDDING ... / CONCAT [...]` y declara la categórica como
`Integer[0, vocab-1]`; el dataset sintético muestrea índices enteros válidos (no
floats). El export/entrenamiento/inferencia de embeddings se cubren en C1/C2.
"""
from __future__ import annotations

import csv as csvmod
import io

from matrixai.parser import parse_text
from matrixai.training.parser import parse_training_text
from matrixai.training.composite_generator import CompositeNetworkGenerator
from matrixai.training.synthetic import SyntheticDataGenerator


# GEN C5: caller/LLM categorical_fields below _ONEHOT_MAX (12) are one-hot
# territory and no longer become embeddings — these tests exercise the embedding
# MECHANICS, so they use a vocab above the aligned threshold.
def _gen(vocab=16):
    return CompositeNetworkGenerator().generate(
        "Clasificar bajo medio alto con bloques residuales",
        categorical_fields={"categoria": vocab},
        force_residual=True,
        input_fields=["categoria", "precio"],
    )


def test_generator_declares_index_range_zero_to_vocab_minus_one():
    gen = _gen(vocab=16)
    # Integer index range is 0..vocab-1 (the table has `vocab` rows).
    assert "categoria: Integer[0, 15]" in gen.mxai_text
    assert "EMBEDDING" in gen.mxai_text and "VOCAB 16" in gen.mxai_text
    assert "CONCAT [categoria_emb, precio] -> features" in gen.mxai_text


def test_synthetic_dataset_samples_integer_indices():
    gen = _gen(vocab=16)
    prog = parse_text(gen.mxai_text)
    tr = parse_training_text(gen.training_text)
    adapter = SyntheticDataGenerator(prog, tr, rows=60, seed=7, mode="coherent").generate()
    # Read the categorical column from the generated rows.
    field_type = prog.vectors[0].field_types["categoria"]
    assert field_type.name == "Integer"
    gen_sampler = SyntheticDataGenerator(prog, tr, rows=1, seed=3, mode="coherent")
    samples = [gen_sampler._sample_type(field_type, "categoria") for _ in range(200)]
    assert all(float(s).is_integer() for s in samples)
    assert all(0 <= s <= 15 for s in samples)
    assert len(set(samples)) > 1  # actually varies, not a constant


def test_composite_template_has_a_data_row():
    """The composite generator's dataset template must include ≥1 data row, or the
    generation-time TrainingVerifier hard-fails ('DATASET must contain at least one
    row') — the bug that blocked training of embedding/residual models."""
    gen = _gen(vocab=16)
    lines = [ln for ln in gen.dataset_template_text.splitlines() if ln.strip()]
    assert len(lines) >= 2, "template needs a header and at least one data row"
    header, row = lines[0].split(","), lines[1].split(",")
    assert len(row) == len(header)
    # the categorical (embedding) column carries an integer index, not a float/label
    cat_idx = header.index("categoria")
    assert float(row[cat_idx]).is_integer()


def test_embedding_index_not_normalized_in_dataset_csv():
    """The categorical index column keeps whole-number values in the generated CSV."""
    from matrixai.playground import _generate_synthetic_dataset
    gen = _gen(vocab=16)
    ds = _generate_synthetic_dataset(gen.mxai_text, gen.training_text, 40, 7, "coherent")
    csv_text = ds.get("csv") or ds.get("csv_text") or ""
    rows = list(csvmod.DictReader(io.StringIO(csv_text)))
    cats = [float(r["categoria"]) for r in rows]
    assert cats, "expected generated rows"
    assert all(c.is_integer() for c in cats)
    assert all(0 <= c <= 15 for c in cats)
