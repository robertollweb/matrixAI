"""M12 Corte 1 — límites operativos configurables (matrixai.limits).

Verifica la resolución de topes (hosted/env/perfil/default), que el playground y el
generador denso los respetan en runtime, y que "sin límite" sólo existe en descargable.
"""
from __future__ import annotations

import importlib

import pytest

from matrixai import limits


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Cada test parte sin overrides ni hosted ni perfil."""
    for var in ("MATRIXAI_HOSTED", "MATRIXAI_LIMITS_PROFILE",
                "MATRIXAI_MAX_ROWS", "MATRIXAI_MAX_EPOCHS", "MATRIXAI_MAX_CSV_BYTES",
                "MATRIXAI_MAX_DEPTH", "MATRIXAI_MAX_LABELS"):
        monkeypatch.delenv(var, raising=False)
    yield


# ── resolución de topes ──────────────────────────────────────────────────────────

def test_defaults_are_equilibrado():
    assert limits.get_limit("max_rows") == 50_000
    assert limits.get_limit("max_epochs") == 1_000
    assert limits.get_limit("max_csv_bytes") == 50_000_000
    assert limits.get_limit("max_depth") == 12
    assert limits.get_limit("max_labels") == 12
    assert limits.is_hosted() is False


def test_unknown_limit_raises():
    with pytest.raises(KeyError):
        limits.get_limit("max_nope")


def test_profile_avanzado(monkeypatch):
    monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "avanzado")
    assert limits.get_limit("max_rows") == 1_000_000
    assert limits.get_limit("max_depth") == 128


def test_profile_ilimitado_gives_none(monkeypatch):
    monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "ilimitado")
    assert limits.get_limit("max_rows") is None
    assert limits.get_limit("max_epochs") is None


def test_bad_profile_falls_back_to_equilibrado(monkeypatch):
    monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "turbo-inexistente")
    assert limits.get_limit("max_rows") == 50_000


def test_per_limit_env_override(monkeypatch):
    monkeypatch.setenv("MATRIXAI_MAX_ROWS", "250000")
    assert limits.get_limit("max_rows") == 250_000
    # otros límites siguen en default
    assert limits.get_limit("max_epochs") == 1_000


@pytest.mark.parametrize("token", ["0", "none", "unlimited", "ilimitado", "off"])
def test_env_unlimited_tokens(monkeypatch, token):
    monkeypatch.setenv("MATRIXAI_MAX_ROWS", token)
    assert limits.get_limit("max_rows") is None


def test_env_invalid_falls_back_to_profile(monkeypatch):
    monkeypatch.setenv("MATRIXAI_MAX_ROWS", "abc")
    assert limits.get_limit("max_rows") == 50_000


def test_negative_env_means_unlimited(monkeypatch):
    monkeypatch.setenv("MATRIXAI_MAX_EPOCHS", "-5")
    assert limits.get_limit("max_epochs") is None


# ── hosted = topes duros, ignora overrides (anti-DoS) ──────────────────────────────

def test_hosted_ignores_env_and_profile(monkeypatch):
    monkeypatch.setenv("MATRIXAI_HOSTED", "1")
    monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "ilimitado")
    monkeypatch.setenv("MATRIXAI_MAX_ROWS", "9999999")
    assert limits.is_hosted() is True
    assert limits.get_limit("max_rows") == 50_000  # duro, ignora todo
    assert limits.get_limit("max_epochs") == 1_000


def test_hosted_never_offers_ilimitado(monkeypatch):
    monkeypatch.setenv("MATRIXAI_HOSTED", "1")
    snap = limits.limits_snapshot()
    assert "ilimitado" not in snap["profiles_available"]
    assert snap["hosted"] is True
    assert snap["profile"] == "equilibrado"


def test_snapshot_descargable_offers_ilimitado(monkeypatch):
    monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "avanzado")
    snap = limits.limits_snapshot()
    assert "ilimitado" in snap["profiles_available"]
    assert snap["profile"] == "avanzado"
    assert snap["limits"]["max_rows"] == 1_000_000


# ── cap / exceeds ──────────────────────────────────────────────────────────────────

def test_cap_and_exceeds_with_limit():
    assert limits.cap(60_000, "max_rows") == 50_000
    assert limits.cap(10, "max_rows") == 10
    assert limits.exceeds(60_000, "max_rows") is True
    assert limits.exceeds(10, "max_rows") is False


def test_cap_and_exceeds_unlimited(monkeypatch):
    monkeypatch.setenv("MATRIXAI_MAX_ROWS", "unlimited")
    assert limits.cap(10_000_000, "max_rows") == 10_000_000
    assert limits.exceeds(10_000_000, "max_rows") is False


# ── integración: playground y generador respetan los límites en runtime ────────────

def test_playground_generation_honours_row_override(monkeypatch):
    """Sin override topa a 50k; con override a 80k deja pasar más filas."""
    from matrixai.playground import _generate_synthetic_dataset
    mxai = ("PROJECT P\nVECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND\n"
            "NETWORK Net\n  INPUT In\n  LAYER Dense units=4 activation=relu\n"
            "  LAYER Dense units=2 activation=softmax\n  OUTPUT y: ProbabilityMap[A, B]\nEND\n"
            "GRAPH\n  In -> Net\nEND\n")
    mxtrain = ("MODEL P.mxai\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
               "  INPUT In FROM COLUMNS [a, b]\n  TARGET y: Label[A, B]\n"
               "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=16\nEND\n"
               "LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET y\nEND\n"
               "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.1\n  UPDATE Net.*\nEND\n")
    # default: 70000 pedidas → topa a 50000
    r = _generate_synthetic_dataset(mxai, mxtrain, rows=70000, seed=1, mode="random")
    assert r["ok"]
    assert r["rows"] == 50_000
    # override a 80000: 70000 pedidas pasan tal cual
    monkeypatch.setenv("MATRIXAI_MAX_ROWS", "80000")
    r2 = _generate_synthetic_dataset(mxai, mxtrain, rows=70000, seed=1, mode="random")
    assert r2["ok"]
    assert r2["rows"] == 70_000


def test_epoch_cap_honours_override(monkeypatch):
    from matrixai.playground import _apply_epoch_cap
    from matrixai.training.parser import parse_training_text
    train = parse_training_text(
        "MODEL P.mxai\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
        "  INPUT In FROM COLUMNS [a]\n  TARGET y: Label[A, B]\n"
        "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=16\nEND\n"
        "LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET y\nEND\n"
        "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.1\n  UPDATE Net.*\nEND\n"
    )
    assert _apply_epoch_cap(train, 5000) == 1_000  # default cap
    monkeypatch.setenv("MATRIXAI_MAX_EPOCHS", "20000")
    assert _apply_epoch_cap(train, 5000) == 5_000  # bajo el nuevo tope, pasa
    monkeypatch.setenv("MATRIXAI_MAX_EPOCHS", "0")  # sin límite
    assert _apply_epoch_cap(train, 5000) == 5_000


def test_generator_depth_honours_override(monkeypatch):
    from matrixai.training.dense_generator import DenseNetworkGenerator
    gen = DenseNetworkGenerator()
    # default: 40 capas pedidas → topa a 12
    hidden = gen._extract_hidden_layers("crea una red de 40 capas ocultas", input_dim=8)
    assert len(hidden) == 12
    monkeypatch.setenv("MATRIXAI_MAX_DEPTH", "64")
    hidden2 = gen._extract_hidden_layers("crea una red de 40 capas ocultas", input_dim=8)
    assert len(hidden2) == 40
