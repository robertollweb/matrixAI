"""128-C4: el comparador de la integración contra la referencia de la v2.12, con la regla del §2 del contrato (bit a bit
sin TabM; con TabM, el suelo de ruido) y los dos grandes que tienen que terminar. Registros sintéticos con la forma de los
del instrumento del 120."""
import importlib.util
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "comparar_c4_128.py"
spec = importlib.util.spec_from_file_location("comparar_c4_128_bajo_prueba", RUTA)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def rec(nombre, campeon="lightgbm", test=0.9, medias=None, estado="completed", tarea="binary_classification"):
    medias = medias if medias is not None else {"lightgbm": 0.91, "sklearn.hgb": 0.90, "baseline": 0.5}
    return {"nombre": nombre, "estado": estado, "campeon": campeon if estado == "completed" else None, "tarea": tarea,
            "memoria_pico_mb": 1000.0,
            "test": {"rol": "test", "evidencia": "independent_test", "metricas": {"auroc": test}},
            "media_de_la_seleccion": {"metrica": "auroc",
                                      "motores": [{"motor": m, "valor": v, "compite": True} for m, v in medias.items()]}}


def doc(recs, suelo=0.0):
    return {"suelo_de_ruido": suelo, "conjuntos": {r["nombre"]: r for r in recs}}


GRANDES_NO = [rec("APSFailure", estado="sondeo_fallido"), rec("Allstate_Claims_Severity", estado="completed", campeon=None)]
GRANDES_SI = [rec("APSFailure"), rec("Allstate_Claims_Severity")]


def test_identico_y_los_grandes_terminando_CUMPLE_bit_a_bit():
    r = C.comparar(doc([rec("a"), *GRANDES_NO]), doc([rec("a"), *GRANDES_SI]), exacto=True)
    assert r["resultado"] == "CUMPLE" and r["codigo"] == 0
    assert r["empiezan_a_completar"] == ["APSFailure", "Allstate_Claims_Severity"]


def test_bit_a_bit_una_media_distinta_con_el_mismo_test_NO_CUMPLE():
    nuevo = rec("a", medias={"lightgbm": 0.91, "sklearn.hgb": 0.9000000001, "baseline": 0.5})
    r = C.comparar(doc([rec("a"), *GRANDES_NO]), doc([nuevo, *GRANDES_SI]), exacto=True)
    assert r["filas"][0]["clase"] == "distinto" and r["resultado"] == "NO CUMPLE"


def test_bit_a_bit_otro_campeon_con_el_mismo_test_NO_CUMPLE():
    nuevo = rec("a", campeon="sklearn.hgb")
    r = C.comparar(doc([rec("a"), *GRANDES_NO]), doc([nuevo, *GRANDES_SI]), exacto=True)
    assert r["filas"][0]["clase"] == "distinto" and r["resultado"] == "NO CUMPLE"


def test_los_grandes_sin_terminar_NO_CUMPLE_aunque_el_resto_sea_igual():
    r = C.comparar(doc([rec("a"), *GRANDES_NO]), doc([rec("a"), *GRANDES_NO]), exacto=True)
    assert r["resultado"] == "NO CUMPLE" and r["grandes_sin_terminar"] == ["APSFailure", "Allstate_Claims_Severity"]


def test_con_suelo_dentro_del_suelo_cuenta_como_igual_y_por_encima_baja():
    ref = doc([rec("a", test=0.900), rec("b", test=0.900), *GRANDES_NO], suelo=0.5)
    nuevo = doc([rec("a", test=0.896), rec("b", test=0.890), *GRANDES_SI], suelo=0.2)      # -0,4 y -1,0 puntos
    r = C.comparar(ref, nuevo, exacto=False)
    assert r["suelo"] == 0.5                                                                # el mayor de las dos
    assert [f["clase"] for f in r["filas"][:2]] == ["dentro_del_suelo", "baja"]
    assert r["resultado"] == "NO CUMPLE" and r["empeoran"] == ["b"]


def test_con_suelo_un_cambio_de_campeon_que_no_empeora_hay_que_explicarlo():
    ref = doc([rec("a", test=0.90), *GRANDES_NO], suelo=0.0)
    nuevo = doc([rec("a", campeon="matrixai.dense.tabm_cpu", test=0.92), *GRANDES_SI], suelo=0.0)
    r = C.comparar(ref, nuevo, exacto=False)
    assert r["filas"][0]["clase"] == "sube"
    assert r["resultado"] == "REVISAR" and r["codigo"] == 2 and r["cambios_de_campeon_por_explicar"] == ["a"]


def test_dejar_de_completar_o_faltar_NO_CUMPLE():
    ref = doc([rec("a"), rec("b"), *GRANDES_NO])
    r = C.comparar(ref, doc([rec("a", estado="failed"), *GRANDES_SI]), exacto=True)
    assert {f["nombre"]: f["clase"] for f in r["filas"]}["a"] == "deja_de_completar"
    assert {f["nombre"]: f["clase"] for f in r["filas"]}["b"] == "falta"
    assert r["resultado"] == "NO CUMPLE" and r["empeoran"] == ["a", "b"]


def test_sin_suelo_la_regla_del_producto_no_se_aplica():
    ref = doc([rec("a")]); del ref["suelo_de_ruido"]
    with pytest.raises(C.Incomparable):
        C.comparar(ref, doc([rec("a")]), exacto=False)


def test_un_completado_sin_test_es_un_instrumento_roto():
    roto = rec("a"); roto["test"] = {}
    with pytest.raises(C.Incomparable):
        C.comparar(doc([rec("a")]), doc([roto]), exacto=True)
