# SPDX-License-Identifier: AGPL-3.0-only
"""M15/M12 — batch efectivo en CUDA vs CPU + override por MATRIXAI_GPU_BATCH.

El training text autogenerado trae `BATCH size=8` (default del demo CPU stdlib). En GPU
honrarlo deja la VRAM casi vacía y el throughput por los suelos. `effective_batch_size`
ignora ese batch pequeño en CUDA (usa un default grande tunable) pero lo respeta en CPU.
"""
from __future__ import annotations

import pytest

from matrixai.training.dense_torch_trainer import effective_batch_size, _GPU_DEFAULT_BATCH


# ── CUDA: ignora el batch pequeño del spec, usa el default grande ─────────────
def test_cuda_ignores_small_spec_batch():
    # BATCH size=8 autogenerado → en CUDA se sube al default grande (capado por n_train).
    assert effective_batch_size("cuda", 8, n_train=1_000_000) == _GPU_DEFAULT_BATCH


def test_cuda_none_spec_uses_default():
    assert effective_batch_size("cuda", None, n_train=1_000_000) == _GPU_DEFAULT_BATCH


def test_cuda_respects_larger_spec():
    # Un usuario que pide un batch MAYOR que el default, manda.
    assert effective_batch_size("cuda", 65536, n_train=1_000_000) == 65536


def test_cuda_capped_by_n_train():
    # Nunca exceder el nº de ejemplos de entrenamiento.
    assert effective_batch_size("cuda", 8, n_train=100) == 100


def test_cuda_device_index_string():
    # 'cuda:0' también cuenta como CUDA.
    assert effective_batch_size("cuda:0", 8, n_train=1_000_000) == _GPU_DEFAULT_BATCH


# ── Override por env MATRIXAI_GPU_BATCH ───────────────────────────────────────
def test_env_override_lowers_default(monkeypatch):
    # Bajar el batch (p.ej. para no dar OOM en una T4 con red enorme).
    monkeypatch.setenv("MATRIXAI_GPU_BATCH", "4096")
    assert effective_batch_size("cuda", 8, n_train=1_000_000) == 4096


def test_env_override_below_spec_still_takes_max(monkeypatch):
    # max(env, spec): si el spec del usuario es mayor que el env, gana el spec.
    monkeypatch.setenv("MATRIXAI_GPU_BATCH", "1024")
    assert effective_batch_size("cuda", 8192, n_train=1_000_000) == 8192


def test_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MATRIXAI_GPU_BATCH", "0")  # 0 → cae al default
    assert effective_batch_size("cuda", 8, n_train=1_000_000) == _GPU_DEFAULT_BATCH


# ── CPU: respeta el batch del spec ────────────────────────────────────────────
def test_cpu_respects_spec_batch():
    assert effective_batch_size("cpu", 8, n_train=1_000_000) == 8


def test_cpu_none_spec_defaults_2048():
    assert effective_batch_size("cpu", None, n_train=1_000_000) == 2048


def test_cpu_capped_by_n_train():
    assert effective_batch_size("cpu", 8192, n_train=500) == 500


def test_cpu_env_does_not_apply(monkeypatch):
    # MATRIXAI_GPU_BATCH solo afecta a CUDA.
    monkeypatch.setenv("MATRIXAI_GPU_BATCH", "16384")
    assert effective_batch_size("cpu", 8, n_train=1_000_000) == 8
