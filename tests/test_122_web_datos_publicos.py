# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""122-WEB — el JSON público de TabM elegible en el modo experto (`tabm_122_c0.json`) es el que genera
`generar_tabm_122_c0.py` desde el resultado de 122-C0, BYTE A BYTE, y el generador PARA cuando la fuente no
cuadra (no se publica nada a medias). Mismo patrón que `test_120_web_datos_publicos.py`.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "datos_publicos_generar_tabm_122_c0", RAIZ / "benchmarks" / "datos_publicos" / "generar_tabm_122_c0.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_el_publicado_es_el_que_se_genera_hoy():
    assert gen.SALIDA.read_text(encoding="utf-8") == gen.serializar(gen.componer())


def test_copia_y_no_calcula_lo_que_dice_la_web():
    """Las cifras que enseña la web, contra el resultado crudo: 7 de 8, la excepción y el pliegue cortado."""
    datos = gen.componer()
    crudo = json.loads(gen.FUENTE.read_text(encoding="utf-8"))
    v = crudo["veredicto"]
    assert (datos["veredicto"], datos["cumplidos"], datos["de"]) == (v["veredicto"], v["cumplidos"], v["de"])
    assert [f["conjunto"] for f in datos["fuera_de_la_regla"]] == sorted(
        d["dataset"] for d in v["detalle"] if not d["cumple"])
    assert datos["pliegues_de_tabm"] == sum(1 for r in crudo["resultados"] if r["motor"] == "tabm")
    assert datos["pliegues_cortados_por_el_plazo"] == [
        {"conjunto": c["dataset"], "pliegue": c["pliegue"]} for c in v["tabm_con_el_plazo_cortado"]]


def _con(monkeypatch, mutar):
    """`componer()` leyendo una copia del resultado de 122-C0 mutada por `mutar`."""
    falso = copy.deepcopy(json.loads(gen.FUENTE.read_text(encoding="utf-8")))
    mutar(falso)
    leer = Path.read_text

    def _leer(self, *a, **k):
        if self == gen.FUENTE:
            return json.dumps(falso)
        return leer(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", _leer)


def _primer_tabm(d):
    return next(r for r in d["resultados"] if r["motor"] == "tabm")


@pytest.mark.parametrize("mutar", [
    lambda d: d["veredicto"].update(de=d["veredicto"]["de"] + 1),
    lambda d: d["veredicto"].update(cumplidos=d["veredicto"]["cumplidos"] + 1),
    lambda d: d["veredicto"].update(tabm_completa_todos=False),
    lambda d: _primer_tabm(d).update(estado="failed"),
    lambda d: d["veredicto"].update(tabm_con_el_plazo_cortado=[]),
], ids=["de-no-cuadra", "cumplidos-no-cuadra", "tabm-incompleta", "pliegue-fallido", "cortados-no-cuadran"])
def test_para_si_la_fuente_no_cuadra(monkeypatch, mutar):
    _con(monkeypatch, mutar)
    with pytest.raises(gen.NoCuadra):
        gen.componer()
