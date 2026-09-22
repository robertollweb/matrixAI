# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El catálogo y el alcance salen del protocolo EN USO, no del v1 escrito a mano.

Bloqueante de la auditoría del 2026-09-22: la pasada v2 cambiaba `RUTA_DEL_PROTOCOLO`
y `_catalogo_registrado()` seguía leyendo `protocolo_exploratorio.json`; `kick` (que solo
está en la v2) se leyó sin catálogo y tomó `WarrantyCost` como objetivo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
sys.path.insert(0, str(_FASE0))

import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402

V1 = _FASE0 / "protocolo_exploratorio.json"
V2 = _FASE0 / "protocolo_exploratorio_v2.json"
KICK = 41162


def _ids(ruta):
    return {d["data_id"] for d in json.loads(ruta.read_text(encoding="utf-8"))["datasets"]}


def test_el_catalogo_es_el_del_protocolo_en_uso_y_cambia_con_el(monkeypatch):
    monkeypatch.setattr(c3, "_CATALOGO_POR_DATA_ID", None)
    monkeypatch.setattr(c3, "RUTA_DEL_PROTOCOLO", V2)
    catalogo = c3._catalogo_registrado()
    assert set(catalogo) == _ids(V2)
    assert catalogo[KICK]["columna_objetivo"] == "IsBadBuy"
    # Y si el protocolo en uso cambia, el catálogo NO se queda en el de antes.
    monkeypatch.setattr(c3, "RUTA_DEL_PROTOCOLO", V1)
    assert set(c3._catalogo_registrado()) == _ids(V1)
    assert KICK not in c3._catalogo_registrado()


def test_el_alcance_de_c3_lee_el_protocolo_en_uso(monkeypatch, tmp_path):
    monkeypatch.setattr(c3, "RUTA_DEL_PROTOCOLO", tmp_path / "no_existe.json")
    alcance = c3._alcance_y_veredicto([])
    assert "no_se_pudo_determinar_el_alcance" in alcance
    assert "no_existe.json" in alcance["no_se_pudo_determinar_el_alcance"]


def test_c5_no_guarda_una_copia_de_la_ruta():
    """Una copia tomada al importar seguía en la v1 cuando la v2 cambiaba la de C3."""
    assert not hasattr(c5, "RUTA_DEL_PROTOCOLO")
