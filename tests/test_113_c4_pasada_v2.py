# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""113-C4 — la pasada v2 no mide si la receta que ejecutaría la densa no es la
que el protocolo v2 registró (invariante 1 del contrato 113: «la receta se
fija ANTES de medir»)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
sys.path.insert(0, str(_FASE0))

import pasada_v2_113 as pasada  # noqa: E402


def test_con_la_receta_de_hoy_la_guarda_deja_medir():
    comprobacion = pasada.exigir_la_receta_registrada()
    assert comprobacion["registrada"] == comprobacion["ejecutada"]
    assert comprobacion["ejecutada"]["optimizador"] == "adam"


def test_si_la_densa_ejecutara_otra_tasa_no_se_mide(monkeypatch):
    from matrixai_engines.motores import densa
    monkeypatch.setattr(densa, "TASA_DE_APRENDIZAJE", 0.01)
    with pytest.raises(SystemExit) as info:
        pasada.exigir_la_receta_registrada()
    assert "NO es la registrada" in str(info.value)


def test_lee_el_protocolo_v2_y_escribe_en_su_propio_artefacto():
    assert pasada.RUTA_DEL_PROTOCOLO_V2.name == "protocolo_exploratorio_v2.json"
    assert pasada.RUTA_DE_SALIDA_V2.name == "pasada_v2_113_resultado.json"
    assert pasada.RUTA_DE_SALIDA_V2.name != "pasada_amplia_101_c5_resultado.json"
