"""120-WEB — el JSON público de la red nueva en el Studio (`red_120_en_el_studio.json`) es el que
genera `generar_red_120.py` desde los resultados de C3′ y C3‴, BYTE A BYTE, y el generador PARA
cuando la fuente no cuadra (no se publica nada a medias)."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "datos_publicos_generar_red_120", RAIZ / "benchmarks" / "datos_publicos" / "generar_red_120.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_el_publicado_es_el_que_se_genera_hoy():
    assert gen.SALIDA.read_text(encoding="utf-8") == gen.serializar(gen.componer())


def test_copia_y_no_calcula_lo_que_sube_y_lo_que_queda_fuera():
    datos = gen.componer()
    cpu = json.loads((gen.FASE0 / "resultado_120_c3s2.json").read_text(encoding="utf-8"))
    sube = {s["conjunto"]: s["diferencia_en_puntos"] for s in datos["paquete_cpu"]["suben"]}
    assert sube == {k: c["diferencia"] for k, c in cpu["comparaciones"].items() if c["clase"] == "sube"}
    assert {f["conjunto"] for f in datos["paquete_cpu"]["fuera_por_memoria"]} == {
        "KDDCup09_appetency", "Allstate_Claims_Severity"}
    # El paquete con GPU se midió antes de la guarda: nada fuera por memoria.
    assert datos["paquete_gpu"]["fuera_por_memoria"] == []
    assert datos["motor"] == "matrixai.dense.tabm_cpu"


def _con(monkeypatch, nombre, mutar):
    """`componer()` leyendo una copia del resultado `nombre` mutada por `mutar`."""
    real = json.loads((gen.FASE0 / nombre).read_text(encoding="utf-8"))
    falso = copy.deepcopy(real)
    mutar(falso)
    leer = Path.read_text

    def _leer(self, *a, **k):
        if self.name == nombre:
            return json.dumps(falso)
        return leer(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", _leer)


@pytest.mark.parametrize("mutar", [
    lambda d: d["veredicto"].update(mejora=False),
    lambda d: d["veredicto"].update(suben=d["veredicto"]["suben"] + 1),
    lambda d: d["veredicto"].update(bajan=1),
    lambda d: d["veredicto"].update(dejan_de_completar=1),
], ids=["no-mejora", "suben-no-cuadra", "baja-uno", "deja-de-completar"])
def test_para_si_la_fuente_no_cuadra(monkeypatch, mutar):
    _con(monkeypatch, "resultado_120_c3s2.json", mutar)
    with pytest.raises(gen.NoCuadra):
        gen.componer()
