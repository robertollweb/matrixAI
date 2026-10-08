"""128 · C5 — el veredicto de la GPU (`benchmarks/fase0/veredicto_128_c5.py`), la regla E2 de Roberto: ninguno baja más
que el suelo de ruido medido repitiendo, ninguno deja de terminar, alguno mejora. Con registros FABRICADOS (aquí no
hay GPU). Cada prueba lleva su nombre porque cada una tiene su sabotaje."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"


def _cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre + "_en_pruebas", _FASE0 / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    antes = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = antes
    return mod


V = _cargar("veredicto_128_c5")
TABM = V.MOTOR_TABM
CONTROL = ["dresses-sales", "climate-model-simulation-crashes", "us_crime", "pc1"]
OTROS = ["micro-mass", "mfeat-factors", "wilt", "pendigits", "Internet-Advertisements", "house_16H",
         "KDDCup09_appetency", "APSFailure", "Allstate_Claims_Severity"]
TODOS = CONTROL + OTROS


def _rec(nombre, test=80.0, *, gpu=False, estado="completed", campeon="lightgbm", vram_extra=None, pared=60.0):
    tarea = "regression" if nombre in ("us_crime", "house_16H", "Allstate_Claims_Severity") else "binary_classification"
    metrica = V._v3.METRICA_DE_CIERRE[tarea]
    return {"nombre": nombre, "tarea": tarea, "estado": estado, "campeon": campeon if estado == "completed" else None,
            "test": {"rol": "test", "evidencia": "independent_test", "metricas": {metrica: test / 100.0}},
            "entrenamiento_del_entregado": {"dispositivo": "cuda" if gpu else None},
            "recursos": {"vram_extra_mb": vram_extra}, "pared_estudio_s": pared}


def _brazo(cifras, *, gpu=False, repeticion=None, **kw):
    conjuntos = {n: _rec(n, cifras.get(n, 80.0), gpu=gpu and n == "micro-mass", **kw) for n in TODOS}
    rep = {n: _rec(n, (repeticion or {}).get(n, cifras.get(n, 80.0))) for n in CONTROL}
    return {"conjuntos": conjuntos, "repeticion_de_control": rep}


def _doc(a, b):
    return {"brazos": {"A": a, "B": b}, "esperados": 13}


# --- la regla ---------------------------------------------------------------------------------------------------
def test_ENTRA_si_ninguno_baja_ninguno_deja_de_terminar_y_alguno_mejora():
    a = _brazo({})
    b = _brazo({"micro-mass": 83.0}, gpu=True)
    r = V.evaluar(_doc(a, b))
    v = r["veredicto"]
    assert v["entra"] is True, v
    assert (v["suben"], v["bajan"], v["dejan_de_completar"]) == (1, 0, 0)
    assert [c["clase"] for c in r["comparaciones"] if c["nombre"] == "wilt"] == ["igual"]


def test_NO_ENTRA_si_ninguno_mejora():
    v = V.evaluar(_doc(_brazo({}), _brazo({}, gpu=True)))["veredicto"]
    assert v["entra"] is False and "ninguno mejora" in v["motivos_para_no_entrar"]


def test_NO_ENTRA_si_UNO_baja_mas_que_el_suelo_aunque_otros_suban():
    a = _brazo({}, repeticion={"pc1": 80.4})                      # suelo de A = 0,4
    b = _brazo({"micro-mass": 90.0, "wilt": 79.0}, gpu=True)      # wilt baja 1,0 > 0,4
    r = V.evaluar(_doc(a, b))
    assert r["veredicto"]["suelo_de_ruido"] == pytest.approx(0.4)
    assert [c["nombre"] for c in r["comparaciones"] if c["clase"] == "baja"] == ["wilt"]
    assert r["veredicto"]["entra"] is False and r["veredicto"]["bajan"] == 1


def test_una_bajada_DENTRO_del_suelo_no_es_una_bajada():
    a = _brazo({}, repeticion={"pc1": 81.0})                      # suelo de A = 1,0
    b = _brazo({"micro-mass": 84.0, "wilt": 79.2}, gpu=True)      # baja 0,8 <= 1,0
    r = V.evaluar(_doc(a, b))
    assert {c["nombre"]: c["clase"] for c in r["comparaciones"]}["wilt"] == "igual"
    assert r["veredicto"]["entra"] is True


def test_el_suelo_es_el_MAYOR_de_los_dos_brazos():
    a = _brazo({}, repeticion={"pc1": 80.3})
    b = _brazo({"micro-mass": 84.0}, gpu=True, repeticion={"us_crime": 81.5})
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["suelos_por_brazo"] == pytest.approx({"A": 0.3, "B": 1.5}) and v["suelo_de_ruido"] == pytest.approx(1.5)


def test_una_mejora_DENTRO_del_suelo_tampoco_cuenta_como_mejora():
    a = _brazo({}, repeticion={"pc1": 81.0})
    b = _brazo({"micro-mass": 80.9}, gpu=True)
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["entra"] is False and v["suben"] == 0


def test_NO_ENTRA_si_un_conjunto_deja_de_terminar():
    a = _brazo({})
    b = _brazo({"micro-mass": 90.0}, gpu=True)
    b["conjuntos"]["APSFailure"] = _rec("APSFailure", estado="failed")
    r = V.evaluar(_doc(a, b))
    assert r["veredicto"]["dejan_de_completar"] == 1 and r["veredicto"]["entra"] is False
    assert V.corta([c for c in r["comparaciones"] if c["nombre"] == "APSFailure"]) == "APSFailure deja de completar"


def test_un_conjunto_que_el_brazo_A_no_terminaba_y_el_B_si_cuenta_como_mejora():
    a = _brazo({}); a["conjuntos"]["Allstate_Claims_Severity"] = _rec("Allstate_Claims_Severity", estado="failed")
    b = _brazo({}, gpu=True)
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["suben"] == 1 and v["entra"] is True


def test_NO_ENTRA_con_menos_conjuntos_de_los_esperados():
    a = _brazo({})
    b = _brazo({"micro-mass": 90.0}, gpu=True)
    for n in OTROS[3:]:
        del b["conjuntos"][n]
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["entra"] is False and any("de 13" in m for m in v["motivos_para_no_entrar"])


# --- el corte ----------------------------------------------------------------------------------------------------
def test_CORTA_en_el_primer_conjunto_que_baja_mas_que_el_suelo():
    base = V.comparar(_rec("wilt", 80), _rec("wilt", 80), 0.5)
    assert V.corta([base]) is None
    baja = V.comparar(_rec("wilt", 80), _rec("wilt", 78), 0.5)
    assert baja["clase"] == "baja" and "wilt baja -2.00" in V.corta([base, baja])
    assert V.corta([base, V.comparar(_rec("wilt", 80), _rec("wilt", 79.7), 0.5)]) is None      # dentro del suelo


def test_CORTA_si_deja_de_completar():
    c = V.comparar(_rec("wilt", 80), _rec("wilt", estado="failed"), 0.5)
    assert c["clase"] == "deja_de_completar" and V.corta([c]) == "wilt deja de completar"


# --- la presencia: que se midió la GPU ------------------------------------------------------------------------------
def test_NO_ENTRA_si_el_brazo_B_nunca_entreno_en_la_GPU_diga_lo_que_diga_la_tabla():
    a = _brazo({})
    b = _brazo({"micro-mass": 95.0}, gpu=False)
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["suben"] == 1 and v["entra"] is False
    assert any("no entrenó TabM en la GPU" in m for m in v["motivos_para_no_entrar"])


def test_la_VRAM_de_mas_vale_como_prueba_de_que_el_B_uso_la_GPU_pero_nunca_acusa_al_A():
    b = _brazo({"micro-mass": 95.0}, vram_extra=3000.0)             # sin dispositivo declarado, pero 3 GB de VRAM
    assert V.evaluar(_doc(_brazo({}), b))["veredicto"]["entra"] is True
    a = _brazo({}, vram_extra=3000.0)                               # otra cosa usaba la GPU durante el brazo A: no se acusa
    assert V.presencia(a["conjuntos"], _brazo({}, gpu=True)["conjuntos"]) is None


def test_NO_ENTRA_si_el_brazo_A_entreno_en_la_GPU():
    a = _brazo({}, gpu=True)
    assert "brazo A" in V.presencia(a["conjuntos"], _brazo({}, gpu=True)["conjuntos"])


def test_los_dispositivos_vistos_en_los_intentos_tambien_cuentan():
    rec = _rec("micro-mass"); rec["entrenamiento_del_entregado"] = None; rec["dispositivos_vistos"] = {"cuda": 3, "cpu": 1}
    assert V.entreno_tabm_en_gpu(rec) is True


# --- el suelo ------------------------------------------------------------------------------------------------------
def test_el_suelo_es_cero_si_se_repite_identico_y_la_mayor_diferencia_si_no():
    p = {n: _rec(n, 80.0) for n in CONTROL}
    assert V.suelo_de_ruido(p, {n: _rec(n, 80.0) for n in CONTROL}) == 0.0
    s = {n: _rec(n, 80.0 + (0.7 if n == "pc1" else 0.1)) for n in CONTROL}
    assert V.suelo_de_ruido(p, s) == pytest.approx(0.7)


def test_un_campeon_que_cambia_al_repetir_es_ruido_no_un_error():
    p = {n: _rec(n, 80.0) for n in CONTROL}
    s = {n: _rec(n, 80.0, campeon=TABM if n == "pc1" else "lightgbm") for n in CONTROL}
    assert V.suelo_de_ruido(p, s) == 0.0 and V.campeones_que_cambian(p, s) == ["pc1"]


def test_si_un_control_completa_en_una_repeticion_y_en_la_otra_no_es_incomparable():
    p = {n: _rec(n, 80.0) for n in CONTROL}
    s = dict(p); s["pc1"] = _rec("pc1", estado="failed")
    with pytest.raises(V.Incomparable):
        V.suelo_de_ruido(p, s)


def test_sin_repeticion_de_control_no_hay_suelo_y_no_se_inventa_uno():
    a = _brazo({}); del a["repeticion_de_control"]
    with pytest.raises(V.Incomparable):
        V.evaluar(_doc(a, _brazo({}, gpu=True)))
    assert V.evaluar(_doc(a, _brazo({"micro-mass": 90.0}, gpu=True)), suelo_dado=0.5)["veredicto"]["entra"] is True


def test_un_suelo_alto_se_avisa_pero_no_cambia_la_regla():
    a = _brazo({}, repeticion={"pc1": 85.0})
    v = V.evaluar(_doc(a, _brazo({"micro-mass": 90.0}, gpu=True)))["veredicto"]
    assert v["suelo_alto"] is True and v["entra"] is True


def test_la_cifra_es_la_del_test_y_un_estudio_sin_test_es_un_instrumento_roto():
    rec = _rec("wilt"); rec["test"] = None
    with pytest.raises(V.Incomparable):
        V.cifra_de_test(rec)


def test_el_CLI_lee_un_resultado_y_dice_si_entra(tmp_path, capsys):
    ruta = tmp_path / "r.json"
    ruta.write_text(json.dumps(_doc(_brazo({}), _brazo({"micro-mass": 90.0}, gpu=True))))
    assert V._main(["x", str(ruta)]) == 0
    assert "LA GPU ENTRA" in capsys.readouterr().out
    ruta.write_text(json.dumps(_doc(_brazo({}), _brazo({"wilt": 70.0}, gpu=True))))
    V._main(["x", str(ruta)])
    assert "LA GPU NO ENTRA" in capsys.readouterr().out


# --- 128 (reparación, I1): el modelo ENTREGADO --------------------------------------------------------------------
def test_I1_NO_ENTRA_si_TabM_gana_en_B_con_su_entregado_en_CPU_sin_decirlo_aunque_la_busqueda_usara_la_GPU():
    a = _brazo({})
    b = _brazo({"micro-mass": 83.0}, gpu=True)
    b["conjuntos"]["wilt"]["campeon"] = TABM
    b["conjuntos"]["wilt"]["dispositivos_vistos"] = {"cuda": 3}                    # la búsqueda, en la GPU
    b["conjuntos"]["wilt"]["entrenamiento_del_entregado"] = {"dispositivo": "cpu", "cayo_a_cpu": False}
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["entra"] is False and any("ENTREGADO" in m or "entregó TabM" in m for m in v["motivos_para_no_entrar"]), v


@pytest.mark.parametrize("entregado", [{"dispositivo": "cuda"}, {"dispositivo": "cpu", "cayo_a_cpu": True}])
def test_I1_con_el_entregado_de_TabM_en_la_GPU_o_su_caida_declarada_la_presencia_no_lo_impide(entregado):
    a = _brazo({})
    b = _brazo({"micro-mass": 83.0}, gpu=True)
    b["conjuntos"]["wilt"]["campeon"] = TABM
    b["conjuntos"]["wilt"]["entrenamiento_del_entregado"] = entregado
    assert V.evaluar(_doc(a, b))["veredicto"]["entra"] is True


def test_I1_un_campeon_que_no_es_TabM_no_tiene_entregado_de_TabM_que_mirar():
    assert V.entregado_de_tabm_en_cpu_sin_decirlo(_rec("wilt")) is False
    assert V.entregado_de_tabm_en_cpu_sin_decirlo(_rec("wilt", campeon=TABM)) is True
    assert V.entregado_de_tabm_en_cpu_sin_decirlo(_rec("wilt", campeon=TABM, estado="failed")) is False
