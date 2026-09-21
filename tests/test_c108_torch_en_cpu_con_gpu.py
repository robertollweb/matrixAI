"""`MATRIXAI_TRAIN_BACKEND=torch_cpu`: el camino torch EN CPU aunque haya CUDA,
y SIN preguntar a CUDA (2026-09-21).

Lo abrió Roberto probando un estudio en el paquete GPU del Studio: la red densa
(`matrixai.dense.torch_cpu`) fallaba sus cinco intentos con «the core was about
to train on «cuda»» y el estudio no terminaba. El motor pedía CPU escondiendo la
GPU con `CUDA_VISIBLE_DEVICES=""`, pero el core ya había preguntado por CUDA en
ese proceso, y torch se queda con la PRIMERA respuesta. Con 'torch' y una GPU
presente no había forma de pedir CPU.

Por eso lo que se prueba no es solo el dispositivo que devuelve, sino que NO
PREGUNTA: preguntar es lo que fija la GPU para todo el proceso. Aquí no hay GPU;
se simula con `torch.cuda.is_available` parcheado, que es lo único que consultan
las dos políticas.
"""
from __future__ import annotations

from importlib import util

import pytest

pytestmark = pytest.mark.skipif(util.find_spec("torch") is None, reason="torch not installed")

POLITICAS = ("_select_train_backend", "_select_transformer_train_device")


def _politica(nombre):
    import matrixai.playground as pg
    return getattr(pg, nombre)


def _con_gpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _cuda_prohibida(monkeypatch):
    """Una GPU que ESTALLA si alguien le pregunta: en una máquina real, esa
    pregunta ya la habría dejado visible para todo el proceso."""
    import torch

    def pregunta():
        raise AssertionError("con torch_cpu nadie debe preguntar a CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", pregunta)


@pytest.mark.parametrize("nombre", POLITICAS)
def test_torch_cpu_con_gpu_presente_entrena_en_cpu_sin_preguntar(nombre, monkeypatch):
    _cuda_prohibida(monkeypatch)
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch_cpu")
    assert _politica(nombre)() == (True, "cpu")


@pytest.mark.parametrize("nombre", POLITICAS)
def test_torch_cpu_admite_mayusculas_y_espacios_como_los_demas_modos(nombre, monkeypatch):
    _cuda_prohibida(monkeypatch)
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "  Torch_CPU ")
    assert _politica(nombre)() == (True, "cpu")


@pytest.mark.parametrize("nombre", POLITICAS)
def test_control_torch_con_gpu_sigue_yendo_a_cuda(nombre, monkeypatch):
    """Control del instrumento, y lo que NO cambia: con 'torch' y GPU, CUDA.
    Si esto saliera 'cpu', la GPU simulada no estaría simulando nada."""
    _con_gpu(monkeypatch)
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
    assert _politica(nombre)() == (True, "cuda")


def test_control_auto_con_gpu_sigue_yendo_a_cuda(monkeypatch):
    """El paquete GPU entrena en CUDA por omisión: el modo nuevo no se lo quita
    a quien no lo pide."""
    _con_gpu(monkeypatch)
    monkeypatch.delenv("MATRIXAI_TRAIN_BACKEND", raising=False)
    assert _politica("_select_train_backend")() == (True, "cuda")


def test_torch_cpu_sin_torch_no_finge_torch(monkeypatch):
    """Sin torch instalado, 'torch_cpu' no puede devolver `use_torch=True`: sería
    el fallo de siempre, entrenar por otro camino declarándose torch."""
    import matrixai.parameters.tensor_bridge as tb
    monkeypatch.setattr(tb, "torch_available", lambda: False)
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch_cpu")
    for nombre in POLITICAS:
        assert _politica(nombre)() == (False, "cpu"), nombre


def test_el_nombre_del_modo_es_uno_solo():
    import matrixai.playground as pg
    assert pg.MODO_TORCH_EN_CPU == "torch_cpu"
