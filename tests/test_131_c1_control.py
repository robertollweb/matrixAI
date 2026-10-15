"""131 · C1 — la comprobación del CONTROL (`control_131_c1.py`): qué da por bueno y qué no. Sin estudios ni contenedores:
registros fabricados a partir del C5 del 129 (dresses-sales) y un CSV sintético que reparte el núcleo de verdad."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
spec = importlib.util.spec_from_file_location("control_131_c1_en_pruebas", _FASE0 / "control_131_c1.py")
C = importlib.util.module_from_spec(spec)
_antes = sys.dont_write_bytecode
try:
    spec.loader.exec_module(C)
finally:
    sys.dont_write_bytecode = _antes

N = "dresses-sales"
C5 = json.loads((_FASE0 / "c5_129_diez.json").read_text())["conjuntos"]
TEXTO = "x,y\n" + "".join(f"{i},1\n" for i in range(290)) + "".join(f"{i},2\n" for i in range(210))     # 58 + 42 = 100 de test


def _estado(digest, n=100):
    return {"seleccion": {"evaluacion_final": {"split_plan_digest": digest, "metrics": [{"n_observations": n}]}}}


def _mundo():
    s0 = {N: copy.deepcopy(C5[N])}
    s0[N]["semilla"] = None
    s1 = {N: copy.deepcopy(C5[N])}
    s1[N]["semilla"] = 1
    s1[N]["media_de_la_seleccion"]["motores"][0]["valor"] += 0.01
    s1[N]["test"]["metricas"]["specificity"] = 50 / 58
    return s0, s1, {N: _estado("A")}, {N: _estado("B")}, {N: _estado("A")}


def _comprobar(s0, s1, e0, e1, ec5):
    return C.comprobar(s0, s1, C5, e0, e1, ec5, lambda n: (TEXTO, "y", "2"))


def test_un_control_bueno_cuadra():
    ok, lin = _comprobar(*_mundo())
    assert ok, "\n".join(lin)
    assert not [x for x in lin if x.startswith("FALLA")] and len(lin) >= 10


def test_el_reparto_del_nucleo_da_el_mismo_tamano_y_las_mismas_clases_con_otra_semilla_y_otras_filas():
    p0, c0 = C.reparto_del_test(TEXTO, "y", 0)
    p1, c1 = C.reparto_del_test(TEXTO, "y", 1)
    assert len(p0) == len(p1) == 100 and c0 == c1 == {"1": 58, "2": 42} and p0 != p1
    assert C.reparto_del_test(TEXTO, "y", 0)[0] == p0                      # y la misma semilla repite


@pytest.mark.parametrize("que,cambia", [
    ("la semilla de siempre ya no es el C5", lambda m: m[0][N].update(campeon="otro")),
    ("la medida (i) lleva semilla", lambda m: m[0][N].update(semilla=0)),
    ("la semilla 1 no llegó al registro", lambda m: m[1][N].update(semilla=None)),
    ("con la semilla 1 no termina", lambda m: m[1][N].update(estado="failed")),
    ("el reparto no cambió (la semilla no viaja)", lambda m: m[3].update({N: _estado("A")})),
    ("el reparto de siempre no es el del C5", lambda m: m[4].update({N: _estado("Z")})),
    ("los pliegues de la búsqueda no cambiaron", lambda m: m[1][N].update(media_de_la_seleccion=copy.deepcopy(m[0][N]["media_de_la_seleccion"]))),
    ("el test cambió de tamaño", lambda m: m[3].update({N: _estado("B", n=90)})),
    ("el test de la semilla de siempre ya era de otro tamaño que el de la 1", lambda m: m[2].update({N: _estado("A", n=90)})),
    ("la especificidad no cuadra con el conteo", lambda m: m[1][N]["test"]["metricas"].update(specificity=0.8)),
    ("falta la medida con la semilla 1", lambda m: m[1].clear()),
])
def test_el_control_se_pone_rojo_si(que, cambia):
    m = _mundo()
    cambia(m)
    ok, lin = _comprobar(*m)
    assert not ok and [x for x in lin if x.startswith("FALLA")], que


def test_el_control_se_pone_rojo_si_el_reparto_del_nucleo_cambia_las_clases_del_test_con_la_semilla(monkeypatch):
    """Si algún día el reparto dejara de estratificar, la proporción de clases del test dependería de la semilla."""
    real = C.reparto_del_test

    def torcido(texto, objetivo, semilla, test_fraction=0.2):
        prueba, conteo = real(texto, objetivo, semilla, test_fraction)
        return (prueba, {"1": 57, "2": 43}) if semilla == 1 else (prueba, conteo)

    monkeypatch.setattr(C, "reparto_del_test", torcido)
    ok, lin = _comprobar(*_mundo())
    assert not ok and any("mismos conteos por clase" in x and x.startswith("FALLA") for x in lin)
