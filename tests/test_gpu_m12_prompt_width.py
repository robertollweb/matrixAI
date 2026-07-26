"""GPU C6 / M12 — el ancho de capa se toma del prompt (units=N / N unidades).

Antes el generador determinista ignoraba el ancho y aplicaba un tapering capado a 256,
así que no se podían pedir redes anchas (la GPU no se cargaba). Ahora el prompt fija el
ancho; sin ancho explícito se mantiene el tapering (comportamiento intacto).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from matrixai.training.dense_generator import DenseNetworkGenerator

gen = DenseNetworkGenerator()


@contextmanager
def _perfil(nombre: str):
    """CONTRATO 64 C1 — estos tests piden redes de decenas de millones de
    parámetros, que es justo lo que el perfil `equilibrado` (2 M) existe para
    impedir. Pedirlas es legítimo: para eso están los perfiles. Lo que M12
    prometía —"el prompt fija el ancho"— sigue siendo cierto DENTRO del
    presupuesto de cada perfil; ver `test_ancho_explicito_respeta_el_tope` para
    el comportamiento en el perfil por defecto.
    """
    previo = os.environ.get("MATRIXAI_LIMITS_PROFILE")
    os.environ["MATRIXAI_LIMITS_PROFILE"] = nombre
    try:
        yield
    finally:
        if previo is None:
            os.environ.pop("MATRIXAI_LIMITS_PROFILE", None)
        else:
            os.environ["MATRIXAI_LIMITS_PROFILE"] = previo


def test_width_from_prompt_unidades():
    with _perfil("ilimitado"):
        r = gen.generate("clasificar riesgo con 12 capas ocultas de 2048 unidades", labels=["A", "B", "C"])
    widths = [u for u, _ in r.hidden_layers]
    assert len(r.hidden_layers) == 12
    assert all(w == 2048 for w in widths)


def test_width_from_prompt_units_equals():
    with _perfil("ilimitado"):
        r = gen.generate("detectar fraude, 6 capas ocultas units=1024")
    assert len(r.hidden_layers) == 6
    assert all(u == 1024 for u, _ in r.hidden_layers)


def test_width_capped_at_sanity_max():
    with _perfil("ilimitado"):
        r = gen.generate("clasificar con 3 capas ocultas de 999999 unidades", labels=["A", "B"])
    assert all(u == DenseNetworkGenerator._MAX_EXPLICIT_WIDTH for u, _ in r.hidden_layers)


# ── CONTRATO 64 C1/C3 — el presupuesto manda sobre el prompt ──────────────────

def test_ancho_explicito_respeta_el_tope_del_perfil():
    """Invariante 2: ninguna fuente se salta los límites duros, tampoco el
    prompt. En `equilibrado` una red de 12x2048 (~50 M parámetros) no cabe."""
    with _perfil("equilibrado"):
        r = gen.generate("clasificar riesgo con 12 capas ocultas de 2048 unidades",
                         labels=["A", "B", "C"])
    anchos = [u for u, _ in r.hidden_layers]
    assert len(r.hidden_layers) == 12, "la PROFUNDIDAD pedida se respeta"
    assert max(anchos) < 2048, "el ancho se estrecha para caber en el presupuesto"


def test_estrecharla_no_es_silencioso():
    """Dar en silencio algo distinto de lo que se pidió es peor que negarse."""
    with _perfil("equilibrado"):
        r = gen.generate("clasificar riesgo con 12 capas ocultas de 2048 unidades",
                         labels=["A", "B", "C"])
    aviso = "\n".join(r.warnings)
    assert "2048" in aviso and "tope" in aviso.lower()
    assert "Ajustes" in aviso, "hay que decir DÓNDE se sube el tope"


def test_en_avanzado_cabe_lo_que_en_equilibrado_no():
    """El perfil es la respuesta a "quiero una red grande", no un muro."""
    with _perfil("equilibrado"):
        pequena = gen.generate("clasificar, 4 capas ocultas units=512", labels=["A", "B"])
    with _perfil("avanzado"):
        grande = gen.generate("clasificar, 4 capas ocultas units=512", labels=["A", "B"])
    assert max(u for u, _ in grande.hidden_layers) >= max(u for u, _ in pequena.hidden_layers)
    assert max(u for u, _ in grande.hidden_layers) == 512


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
