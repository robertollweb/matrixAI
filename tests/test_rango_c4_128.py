"""128-C4: el control de los conjuntos que paran por reloj — ¿cae el C4 dentro de la dispersión de la v2.12?"""
import importlib.util
from pathlib import Path

RUTA = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "rango_c4_128.py"
spec = importlib.util.spec_from_file_location("rango_c4_128_bajo_prueba", RUTA)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def x(test, campeon="lightgbm", completa=True):
    return {"campeon": campeon, "completa": completa, "test": test if completa else None}


def test_solapar_es_dispersion_y_quedar_por_debajo_de_todas_empeora():
    assert G.clasificar([x(67.3), x(67.7), x(67.5)], [x(67.27), x(67.6), x(67.4)])["clase"] == "dentro"
    assert G.clasificar([x(67.3), x(67.7)], [x(67.0), x(67.2), x(67.29)])["clase"] == "empeora"
    assert G.clasificar([x(67.3), x(67.7)], [x(67.8), x(68.0)])["clase"] == "mejora"
    assert G.clasificar([x(67.3)], [x(67.3)])["clase"] == "dentro"                          # iguales: dentro


def test_no_completar_en_alguna_pasada_no_es_dispersion():
    assert G.clasificar([x(67.3), x(67.7)], [x(67.5), x(None, completa=False)])["clase"] == "no_completa_en_alguna"
    assert G.clasificar([], [x(67.5)])["clase"] == "sin_pasadas"


def test_cifras_lee_el_test_del_campeon_con_la_funcion_del_120():
    rec = {"nombre": "h", "estado": "completed", "campeon": "sklearn.hgb", "tarea": "regression",
           "test": {"rol": "test", "evidencia": "independent_test", "metricas": {"r2": 0.6727}}}
    assert G.cifras([{"conjuntos": {"h": rec}}, {"conjuntos": {}}], "h") == [
        {"campeon": "sklearn.hgb", "completa": True, "test": 67.27}]
