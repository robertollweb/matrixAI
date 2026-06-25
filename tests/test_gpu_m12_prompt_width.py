"""GPU C6 / M12 — el ancho de capa se toma del prompt (units=N / N unidades).

Antes el generador determinista ignoraba el ancho y aplicaba un tapering capado a 256,
así que no se podían pedir redes anchas (la GPU no se cargaba). Ahora el prompt fija el
ancho; sin ancho explícito se mantiene el tapering (comportamiento intacto).
"""
from __future__ import annotations

from matrixai.training.dense_generator import DenseNetworkGenerator

gen = DenseNetworkGenerator()


def test_width_from_prompt_unidades():
    r = gen.generate("clasificar riesgo con 12 capas ocultas de 2048 unidades", labels=["A", "B", "C"])
    widths = [u for u, _ in r.hidden_layers]
    assert len(r.hidden_layers) == 12
    assert all(w == 2048 for w in widths)


def test_width_from_prompt_units_equals():
    r = gen.generate("detectar fraude, 6 capas ocultas units=1024")
    assert len(r.hidden_layers) == 6
    assert all(u == 1024 for u, _ in r.hidden_layers)


def test_width_capped_at_sanity_max():
    r = gen.generate("clasificar con 3 capas ocultas de 999999 unidades", labels=["A", "B"])
    assert all(u == DenseNetworkGenerator._MAX_EXPLICIT_WIDTH for u, _ in r.hidden_layers)


def test_no_width_keeps_tapering():
    # Sin ancho explícito: tapering por defecto (no uniforme), comportamiento intacto.
    r = gen.generate("quiero una red con 6 capas ocultas para clasificar", labels=["A", "B", "C"])
    widths = [u for u, _ in r.hidden_layers]
    assert len(r.hidden_layers) == 6
    assert len(set(widths)) > 1  # tapering → anchos distintos, no uniforme


def test_width_without_depth_uses_default_depth():
    r = gen.generate("clasificar spam con capas de 512 unidades", labels=["A", "B"])
    assert r.hidden_layers  # genera algo
    assert all(u == 512 for u, _ in r.hidden_layers)


def test_irrelevant_number_not_taken_as_width():
    # "30 días" no debe interpretarse como ancho ni profundidad.
    r = gen.generate("predecir reingreso en 30 días", labels=["A", "B"])
    widths = [u for u, _ in r.hidden_layers]
    assert all(w <= 256 for w in widths)  # tapering por defecto, no 30
