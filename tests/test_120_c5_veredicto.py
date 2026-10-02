"""120-C5 · el veredicto de las correcciones de TabM (enmienda 3 del 119 v4), escrito ANTES de tener los datos: sobre el
resultado REAL de 119-C3 (la 1.1.0 congelada), una «1.2.0» fabricada moviendo sus cifras. Cargado por ruta, sin tocar
`sys.path`."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
_spec = importlib.util.spec_from_file_location("veredicto_120_c5_en_pruebas", _FASE0 / "veredicto_120_c5.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
VIEJO = json.loads((_FASE0 / "resultado_pasada_119_c3.json").read_text())


def _nuevo(mover=None, fallar=(), quitar=(), version="1.2.0+torch2.6.0"):
    """La «1.2.0»: los intentos de TabM de los 12 de la 1.1.0, con otra versión y la métrica movida por conjunto."""
    cierre = VIEJO["metrica_de_cierre_por_dataset"]
    nuevo = copy.deepcopy(VIEJO)
    nuevo["resultados"] = []
    for x in VIEJO["resultados"]:
        if x["motor"] != V.MOTOR or x["dataset"] not in V.CONJUNTOS or x["dataset"] in quitar:
            continue
        y = copy.deepcopy(x)
        y["engine_version"] = version
        if (x["dataset"], x["repeticion"], x["pliegue"]) in fallar:
            y["estado"] = "failed"
        elif mover and x["dataset"] in mover and y.get("estado") == "completed":
            y["metricas"][cierre[x["dataset"]]] += mover[x["dataset"]]
        nuevo["resultados"].append(y)
    return nuevo


def _primer_intento(nombre):
    return next((x["dataset"], x["repeticion"], x["pliegue"]) for x in VIEJO["resultados"]
                if x["motor"] == V.MOTOR and x["dataset"] == nombre and x["estado"] == "completed")


def test_los_12_son_los_de_la_enmienda_3_y_la_congelada_es_toda_1_1_0():
    enmienda = json.loads((_FASE0 / "protocolo_119_v4_enmienda_3.json").read_text())
    for n in V.CONJUNTOS:
        assert n in enmienda["que_se_remide"], n
    assert len(V.CONJUNTOS) == 12 and V.versiones(VIEJO) == {"1.1.0"}


def test_la_misma_red_con_otra_version_no_sube_ni_baja_nada_y_se_adopta():
    filas = V.comparar(VIEJO, _nuevo())
    assert {f["clase"] for f in filas} <= {"igual", "ninguno_completa"}
    assert V.veredicto(filas)["adopta_la_1_2_0"] is True


def test_una_bajada_de_2_puntos_o_dejar_de_completar_no_se_adopta():
    v = V.veredicto(V.comparar(VIEJO, _nuevo(mover={"sick": -0.0201})))
    assert (v["bajan_2"], v["adopta_la_1_2_0"]) == (1, False)
    filas = V.comparar(VIEJO, _nuevo(fallar={_primer_intento("kick")}))
    assert next(f for f in filas if f["nombre"] == "kick")["clase"] == "deja_de_completar"
    assert V.veredicto(filas)["adopta_la_1_2_0"] is False


def test_no_se_adopta_si_bajan_mas_de_los_que_suben():
    menos = {"sick": -0.015, "kick": -0.015, "Moneyball": +0.015}
    assert V.veredicto(V.comparar(VIEJO, _nuevo(mover=menos)))["adopta_la_1_2_0"] is False
    iguales = {"sick": -0.015, "Moneyball": +0.015}
    assert V.veredicto(V.comparar(VIEJO, _nuevo(mover=iguales)))["adopta_la_1_2_0"] is True


def test_menos_de_un_punto_no_cuenta():
    filas = V.comparar(VIEJO, _nuevo(mover={"sick": -0.0099, "kick": +0.0099}))
    assert {f["nombre"]: f["clase"] for f in filas if f["nombre"] in ("sick", "kick")} == {"sick": "igual",
                                                                                          "kick": "igual"}


def test_otros_conjuntos_o_otra_version_no_son_esta_medida():
    with pytest.raises(V.NoEsEstaMedida, match="los 12"):
        V.comparar(VIEJO, _nuevo(quitar=("sick",)))
    with pytest.raises(V.NoEsEstaMedida, match="versiones"):
        V.comparar(VIEJO, _nuevo(version="1.1.0+torch2.6.0"))
    with pytest.raises(V.NoEsEstaMedida, match="versiones"):
        V.comparar(_nuevo(), _nuevo())                     # la «congelada» no es de la 1.1.0


def test_importar_el_veredicto_no_toca_sys_path():
    antes = list(sys.path)
    spec = importlib.util.spec_from_file_location("otra_vez", _FASE0 / "veredicto_120_c5.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert sys.path == antes


def test_una_bajada_de_2_veta_aunque_suban_mas():
    v = V.veredicto(V.comparar(VIEJO, _nuevo(mover={"sick": -0.0201, "kick": +0.015, "Moneyball": +0.015})))
    assert (v["suben"], v["bajan"], v["bajan_2"]) == (2, 1, 1)
    assert v["adopta_la_1_2_0"] is False


def test_los_que_pararon_por_plazo_en_la_congelada_no_deciden_y_salen_del_dato():
    plazo = V.por_plazo_en_la_congelada(VIEJO)
    todos_por_plazo = {n for n, (p, t) in plazo.items() if t and p == t}
    assert todos_por_plazo == {"Allstate_Claims_Severity", "Internet-Advertisements", "KDDCup09_appetency", "connect-4"}
    # Una bajada de 3 puntos en uno de ellos no veta; en uno que decide, sí.
    v = V.veredicto(V.comparar(VIEJO, _nuevo(mover={"Internet-Advertisements": -0.03})))
    assert v["adopta_la_1_2_0"] is True and v["no_deciden_por_plazo"]["Internet-Advertisements"] == "baja_2"
    assert len(v["deciden"]) == 8
    assert V.veredicto(V.comparar(VIEJO, _nuevo(mover={"sick": -0.03})))["adopta_la_1_2_0"] is False
