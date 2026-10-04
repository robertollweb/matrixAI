# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""La pasada de la palanca de las sí/no (`benchmarks/fase0/pasada_sino_b.py`) aplica la regla
REGISTRADA (`protocolo_sino_b.json`) y no otra: el protocolo cuadra con su generador y con su
digest, un protocolo tocado a mano para la pasada, y la regla decide en sus cuatro casos."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
if str(_FASE0) not in sys.path:
    sys.path.insert(0, str(_FASE0))

pasada = pytest.importorskip("pasada_sino_b")   # necesita la palanca en matrixai_engines
import generar_protocolo_sino_b as generador  # noqa: E402

DENSA = "matrixai.dense.torch_cpu"


def test_el_protocolo_escrito_es_el_que_compone_su_generador():
    assert generador.main.__module__  # el generador importa
    escrito = json.loads((_FASE0 / "protocolo_sino_b.json").read_text(encoding="utf-8"))
    assert escrito == generador.componer()


def test_un_protocolo_tocado_a_mano_PARA_la_pasada(tmp_path, monkeypatch):
    tocado = json.loads((_FASE0 / "protocolo_sino_b.json").read_text(encoding="utf-8"))
    tocado["conjuntos"] = ["sick"]
    ruta = tmp_path / "protocolo_sino_b.json"
    ruta.write_text(json.dumps(tocado), encoding="utf-8")
    monkeypatch.setattr(pasada, "RUTA_DEL_PROTOCOLO", ruta)
    with pytest.raises(SystemExit, match="no cuadra con su digest"):
        pasada.protocolo_sino_b()


def test_el_protocolo_bueno_se_lee():
    assert pasada.protocolo_sino_b()["conjuntos"] == ["sick", "house_prices_nominal"]


def _registros(valores: dict[tuple, float | None], metrica="auroc"):
    return [{"motor": DENSA, "repeticion": r, "pliegue": p,
             "estado": "completed" if v is not None else "failed", metrica: v}
            for (r, p), v in valores.items()]


def _pliegues(base: float, delta: float):
    return {(r, p): base + 0.001 * ((r * 5 + p) % 3) + delta for r in range(3) for p in range(5)}


def test_inferioridad_cuando_B_pierde_en_todos_los_pliegues():
    hoy = _pliegues(0.90, 0.0)
    b = {k: v - 0.02 for k, v in hoy.items()}
    v = pasada.veredicto_del_conjunto(dataset="x", metric_id="auroc",
                                      hoy=_registros(hoy), b=_registros(b))
    assert v["inferioridad"] and not v["mejora"] and v["n_pliegues_comunes"] == 15


def test_mejora_cuando_B_gana_en_todos_los_pliegues():
    hoy = _pliegues(0.90, 0.0)
    b = {k: v + 0.02 for k, v in hoy.items()}
    v = pasada.veredicto_del_conjunto(dataset="x", metric_id="auroc",
                                      hoy=_registros(hoy), b=_registros(b))
    assert v["mejora"] and not v["inferioridad"]


def test_ruido_un_intervalo_que_cruza_el_cero_no_es_ni_mejora_ni_inferioridad():
    """El caso que separa usar el extremo BAJO o el ALTO del intervalo: B gana en unos pliegues
    y pierde en otros. Con «inferioridad = bajo < 0» saldría inferioridad, y no lo es."""
    hoy = _pliegues(0.90, 0.0)
    b = {k: v + (0.01 if (k[0] * 5 + k[1]) % 2 else -0.01) for k, v in hoy.items()}
    v = pasada.veredicto_del_conjunto(dataset="x", metric_id="auroc",
                                      hoy=_registros(hoy), b=_registros(b))
    assert v["intervalo"]["bajo"] < 0.0 < v["intervalo"]["alto"]
    assert not v["inferioridad"] and not v["mejora"]


def test_B_que_deja_de_completar_se_cuenta_aunque_su_media_sea_mejor():
    hoy = _pliegues(0.90, 0.0)
    b = {k: v + 0.02 for k, v in hoy.items()}
    b[(1, 2)] = None
    v = pasada.veredicto_del_conjunto(dataset="x", metric_id="auroc",
                                      hoy=_registros(hoy), b=_registros(b))
    assert v["B_deja_de_completar"] == [{"repeticion": 1, "pliegue": 2}]


@pytest.mark.parametrize("caso,esperado", [
    ("limpio", True), ("inferioridad", False), ("deja_de_completar", False), ("falta", None)])
def test_la_regla_de_la_pasada(caso, esperado):
    bueno = {"dataset": "sick", "inferioridad": False, "mejora": False, "B_deja_de_completar": []}
    otro = dict(bueno, dataset="house_prices_nominal")
    if caso == "inferioridad":
        otro["inferioridad"] = True
    if caso == "deja_de_completar":
        otro["B_deja_de_completar"] = [{"repeticion": 0, "pliegue": 0}]
    medidos = [bueno] if caso == "falta" else [bueno, otro]
    v = pasada.veredicto_de_la_pasada(medidos, ["sick", "house_prices_nominal"])
    assert v["entra"] is esperado
