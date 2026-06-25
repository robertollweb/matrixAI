"""M9 — validación de coherencia VECTOR↔campos con aviso accionable bilingüe.

Cuando el .mxai es incoherente (elipsis en los campos, o dimensión declarada ≠ nº de
campos enumerados), `check_dataset_coherence` lo detecta antes de generar y devuelve un
mensaje claro (es/en) en vez del error técnico opaco del parser. No impone reglas nuevas
que el pipeline ya tolera (p. ej. columnas extra en INPUT).
"""
from __future__ import annotations

from matrixai.training.coherence import check_dataset_coherence

_TRAIN = (
    "MODEL P.mxai\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT In FROM COLUMNS [a, b]\n  TARGET y: Label[A, B]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=16\nEND\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET y\nEND\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.1\n  UPDATE Net.*\nEND\n"
)


def _mxai(vector_block: str) -> str:
    return (
        f"PROJECT P\n{vector_block}\n"
        "NETWORK Net\n  INPUT In\n  LAYER Dense units=4 activation=relu\n"
        "  LAYER Dense units=2 activation=softmax\n  OUTPUT y: ProbabilityMap[A, B]\nEND\n"
        "GRAPH\n  In -> Net\nEND\n"
    )


# ── casos OK ──────────────────────────────────────────────────────────────────────

def test_coherent_vector_passes():
    r = check_dataset_coherence(_mxai("VECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND"), _TRAIN)
    assert r.ok is True
    assert r.error_es is None and r.error_en is None


def test_extra_input_columns_tolerated():
    """El pipeline tolera columnas extra en INPUT; M9 NO debe convertirlo en error."""
    train_extra = _TRAIN.replace("[a, b]", "[a, b, c, d]")
    r = check_dataset_coherence(_mxai("VECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND"), train_extra)
    assert r.ok is True


# ── elipsis ─────────────────────────────────────────────────────────────────────────

def test_ellipsis_three_dots_detected():
    block = "VECTOR In[100]\n  sensor_001: Scalar\n  ...\n  sensor_100: Scalar\nEND"
    r = check_dataset_coherence(_mxai(block), _TRAIN)
    assert r.ok is False
    assert "…" in r.error_es or "elipsis" in r.error_es.lower()
    assert r.error_en and "ellipsis" in r.error_en.lower()


def test_ellipsis_unicode_detected():
    block = "VECTOR In[50]\n  f1: Scalar\n  …\n  f50: Scalar\nEND"
    r = check_dataset_coherence(_mxai(block), _TRAIN)
    assert r.ok is False
    assert "In" in r.error_es


# ── dimensión ≠ nº de campos ──────────────────────────────────────────────────────

def test_dimension_field_mismatch():
    block = "VECTOR In[100]\n  sensor_001: Scalar\n  sensor_002: Scalar\n  sensor_003: Scalar\nEND"
    r = check_dataset_coherence(_mxai(block), _TRAIN)
    assert r.ok is False
    assert "100" in r.error_es and "3" in r.error_es
    assert "100" in r.error_en and "3" in r.error_en


def test_message_is_bilingual_and_actionable():
    block = "VECTOR In[10]\n  a: Scalar\n  b: Scalar\nEND"
    r = check_dataset_coherence(_mxai(block), _TRAIN)
    assert r.ok is False
    # accionable: dice qué hacer
    assert "enumera" in r.error_es.lower() or "ajusta" in r.error_es.lower()
    assert "explicitly" in r.error_en.lower() or "dimension" in r.error_en.lower()


# ── integración con el generador del playground ───────────────────────────────────

def test_playground_generation_returns_coherence_error():
    from matrixai.playground import _generate_synthetic_dataset
    block = "VECTOR In[100]\n  sensor_001: Scalar\n  ...\n  sensor_100: Scalar\nEND"
    r = _generate_synthetic_dataset(_mxai(block), _TRAIN, rows=20, seed=1, mode="random")
    assert r["ok"] is False
    assert r["error_kind"] == "coherence"
    assert r["error"] and r["error_en"]


def test_playground_generation_ok_when_coherent():
    from matrixai.playground import _generate_synthetic_dataset
    mxai = _mxai("VECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND")
    r = _generate_synthetic_dataset(mxai, _TRAIN, rows=20, seed=1, mode="random")
    assert r["ok"] is True
    assert r.get("error_kind") is None
