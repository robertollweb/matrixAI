"""128 · C5 — el veredicto de la GPU (`benchmarks/fase0/veredicto_128_c5.py`), la regla E2 de Roberto: ninguno baja más
que el suelo de ruido medido repitiendo, ninguno deja de terminar, alguno mejora — sustituida el 09-10 por la C5′ de
Roberto («banda de 1 punto + tiempo», al final de este fichero). Con registros FABRICADOS (aquí no hay GPU). Cada
prueba lleva su nombre porque cada una tiene su sabotaje."""
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
    assert (v["suben"], v["bajas_toleradas"], v["empeoran"], v["dejan_de_completar"]) == (1, 0, 0, 0)
    assert [c["clase"] for c in r["comparaciones"] if c["nombre"] == "wilt"] == ["igual"]


def test_NO_ENTRA_si_ninguno_mejora():
    v = V.evaluar(_doc(_brazo({}), _brazo({}, gpu=True)))["veredicto"]
    assert v["entra"] is False and any("no superan" in m for m in v["motivos_para_no_entrar"]), v


def test_NO_ENTRA_si_UNO_empeora_aunque_otros_suban():
    a = _brazo({}, repeticion={"pc1": 80.4})                      # suelo de A = 0,4
    b = _brazo({"micro-mass": 90.0, "mfeat-factors": 90.0, "wilt": 77.5}, gpu=True)      # wilt baja 2,5 >= 2
    r = V.evaluar(_doc(a, b))
    assert r["veredicto"]["suelo_de_ruido"] == pytest.approx(0.4)
    assert [c["nombre"] for c in r["comparaciones"] if c["clase"] == "empeora"] == ["wilt"]
    assert r["veredicto"]["entra"] is False and r["veredicto"]["empeoran"] == 1


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
def test_CORTA_en_el_primer_conjunto_que_empeora():
    base = V.comparar(_rec("wilt", 80), _rec("wilt", 80), 0.5)
    assert V.corta([base]) is None
    empeora = V.comparar(_rec("wilt", 80), _rec("wilt", 78), 0.5)
    assert empeora["clase"] == "empeora" and "wilt empeora -2.00" in V.corta([base, empeora])
    assert V.corta([base, V.comparar(_rec("wilt", 80), _rec("wilt", 79.7), 0.5)]) is None      # dentro de la banda
    assert V.corta([base, V.comparar(_rec("wilt", 80), _rec("wilt", 78.9), 0.5)]) is None      # baja tolerada: no corta sola


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


# --- 128 (2.ª auditoría, N3/N4): las caídas a CPU declaradas en B ------------------------------------------------------
def _b_con_mejora_caida(**extra):
    a = _brazo({})
    b = _brazo({"micro-mass": 83.0}, gpu=True, **extra)            # micro-mass mejora y su entregado SÍ está en la GPU
    return a, b


def _solo_mejora_en(b, nombre, cifra, entregado):
    """B sin ninguna mejora salvo `nombre` (con ese entregado). Quita la mejora de micro-mass."""
    b["conjuntos"]["micro-mass"] = _rec("micro-mass", 80.0, gpu=True)
    b["conjuntos"][nombre] = _rec(nombre, cifra, campeon=TABM)
    b["conjuntos"][nombre]["entrenamiento_del_entregado"] = entregado
    b["conjuntos"][nombre]["dispositivos_vistos"] = {"cuda": 3}


def test_N3_una_mejora_cuyo_ENTREGADO_cayo_a_CPU_no_cuenta_como_mejora_de_la_GPU_y_se_dice():
    a, b = _b_con_mejora_caida()
    _solo_mejora_en(b, "wilt", 90.0, {"dispositivo": "cpu", "cayo_a_cpu": True, "por_que_cayo": "fallo_de_la_gpu"})
    r = V.evaluar(_doc(a, b))
    v = r["veredicto"]
    assert v["entra"] is False and v["suben"] == 0, v
    assert any("no superan" in m and "cayó a CPU: no cuenta como GPU" in m and "wilt" in m
               for m in v["motivos_para_no_entrar"]), v["motivos_para_no_entrar"]
    assert v["mejoras_que_cayeron_a_cpu"] == ["wilt"]
    assert [c["nombre"] for c in v["caidas_a_cpu_en_b"]] == ["wilt"]
    assert v["caidas_a_cpu_en_b"][0]["por_que"] == "fallo_de_la_gpu"
    assert next(c for c in r["comparaciones"] if c["nombre"] == "wilt")["cayo_a_cpu_en_b"] is True


def test_N3_con_otra_mejora_que_SI_uso_la_GPU_entra_pero_la_caida_sigue_LISTADA():
    a, b = _b_con_mejora_caida()
    b["conjuntos"]["wilt"] = _rec("wilt", 90.0, campeon=TABM)
    b["conjuntos"]["wilt"]["entrenamiento_del_entregado"] = {"dispositivo": "cpu", "cayo_a_cpu": True}
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["entra"] is True and v["suben"] == 1, v                      # solo micro-mass es de la GPU
    assert [c["nombre"] for c in v["caidas_a_cpu_en_b"]] == ["wilt"]


def test_N3_el_CLI_escribe_cayo_a_CPU_no_cuenta_como_GPU(tmp_path, capsys):
    a, b = _b_con_mejora_caida()
    b["conjuntos"]["wilt"] = _rec("wilt", 90.0, campeon=TABM)
    b["conjuntos"]["wilt"]["entrenamiento_del_entregado"] = {"dispositivo": "cpu", "cayo_a_cpu": True,
                                                              "por_que_cayo": "fallo_de_la_gpu"}
    ruta = tmp_path / "r.json"
    ruta.write_text(json.dumps(_doc(a, b)))
    assert V._main(["x", str(ruta)]) == 0
    sal = capsys.readouterr().out
    assert "CAYÓ A CPU en el brazo B: wilt" in sal and "no cuenta como GPU" in sal, sal


def test_N4_la_RAM_pico_de_cada_brazo_va_al_veredicto_y_el_coste_no_medido_se_dice():
    a, b = _b_con_mejora_caida()
    a["conjuntos"]["wilt"]["recursos"]["ram_pico_mb"] = 1500.0
    b["conjuntos"]["wilt"]["recursos"]["ram_pico_mb"] = 2300.0
    b["conjuntos"]["pc1"]["recursos"]["ram_pico_mb"] = 1900.0
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["ram_pico_mb_por_brazo"] == {"A": 1500.0, "B": 2300.0}, v["ram_pico_mb_por_brazo"]
    assert "densa vea CUDA" in v["coste_no_medido_en_b"] and "NO medido" in v["coste_no_medido_en_b"]


# =====================================================================================================================
# 128-C5′ — la regla nueva de Roberto (09-10 13:40, «banda de 1 punto + tiempo»), escrita antes de construir
# =====================================================================================================================
def _par(d, *, suelo=0.0, pared_a=100.0, pared_b=100.0, nombre="wilt"):
    """Un conjunto con B − A = d puntos."""
    a, b = _rec(nombre, 80.0, pared=pared_a), _rec(nombre, 80.0 + d, pared=pared_b)
    return V.comparar(a, b, suelo)


@pytest.mark.parametrize("d,suelo,clase", [
    (0.0, 0.0, "igual"), (0.99, 0.0, "igual"), (-0.99, 0.0, "igual"), (-0.08, 0.0, "igual"),       # |d| < 1
    (1.0, 0.0, "sube"), (3.0, 0.0, "sube"),                                                          # d >= 1
    (-1.0, 0.0, "baja_tolerada"), (-1.59, 0.0, "baja_tolerada"), (-1.99, 0.0, "baja_tolerada"),    # -2 < d <= -1
    (-2.0, 0.0, "empeora"), (-2.5, 0.0, "empeora"), (-9.0, 0.0, "empeora"),                          # d <= -2
    (1.4, 1.5, "igual"), (1.5, 1.5, "sube"),                                                         # el suelo SUBE la banda
    (-1.4, 1.5, "igual"), (-1.5, 1.5, "baja_tolerada"), (-1.9, 1.5, "baja_tolerada"), (-2.0, 1.5, "empeora"),
    (-2.4, 2.5, "igual"), (-2.5, 2.5, "empeora"),                                                    # suelo > 2: ya no hay tolerada
])
def test_C5p_las_clases_por_diferencia_y_sus_bordes(d, suelo, clase):
    assert _par(d, suelo=suelo)["clase"] == clase


def test_C5p_con_un_suelo_mayor_de_2_no_existe_la_baja_tolerada():
    for d in (-2.6, -3.0, -2.5, -2.49, -2.0, -1.0):
        assert _par(d, suelo=2.5)["clase"] in ("igual", "empeora")


def test_C5p_mejora_de_tiempo_solo_si_el_B_tarda_la_mitad_o_menos_y_queda_igual():
    assert _par(-0.08, pared_a=355.7, pared_b=117.1)["clase"] == "mejora_de_tiempo"      # us_crime de hoy: 3x más rápido
    assert _par(0.0, pared_a=100.0, pared_b=50.0)["clase"] == "mejora_de_tiempo"         # justo la mitad
    assert _par(0.0, pared_a=100.0, pared_b=50.01)["clase"] == "igual"
    assert _par(0.5, pared_a=100.0, pared_b=10.0)["clase"] == "mejora_de_tiempo"
    assert _par(2.0, pared_a=100.0, pared_b=10.0)["clase"] == "sube"                      # una subida no cuenta dos veces
    assert _par(-1.5, pared_a=100.0, pared_b=10.0)["clase"] == "baja_tolerada"            # el tiempo no redime una baja
    assert _par(-2.5, pared_a=100.0, pared_b=10.0)["clase"] == "empeora"
    assert _par(0.0, pared_a=None, pared_b=10.0)["clase"] == "igual"                      # sin pared medida, no hay mejora
    assert _par(0.0, pared_a=100.0, pared_b=None)["clase"] == "igual"


def _con(b_cambios, a_cambios=None, *, gpu=True):
    """Dos brazos de 13 donde los conjuntos de `b_cambios` = (cifra, pared_b) y los de `a_cambios` = pared_a."""
    a = _brazo({})
    b = _brazo({n: c for n, (c, _) in b_cambios.items()}, gpu=gpu)
    for n, (_, pared) in b_cambios.items():
        b["conjuntos"][n]["pared_estudio_s"] = pared
    for n, pared in (a_cambios or {}).items():
        a["conjuntos"][n]["pared_estudio_s"] = pared
    return a, b


def test_C5p_ENTRA_si_las_mejoras_superan_a_las_bajas_toleradas():
    a, b = _con({"micro-mass": (83.0, 60.0), "wilt": (78.8, 60.0)})            # 1 sube, 1 baja tolerada (-1,2)
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert (v["suben"], v["mejoras_de_tiempo"], v["bajas_toleradas"]) == (1, 0, 1) and v["entra"] is False
    a, b = _con({"micro-mass": (83.0, 60.0), "mfeat-factors": (82.0, 60.0), "wilt": (78.8, 60.0)})
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert (v["suben"], v["bajas_toleradas"]) == (2, 1) and v["entra"] is True, v


def test_C5p_la_mejora_de_tiempo_cuenta_como_mejora():
    a, b = _con({"micro-mass": (80.0, 20.0), "wilt": (78.8, 60.0)}, {"micro-mass": 60.0})
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert (v["suben"], v["mejoras_de_tiempo"], v["bajas_toleradas"]) == (0, 1, 1) and v["entra"] is False
    a, b = _con({"micro-mass": (80.0, 20.0), "pendigits": (80.0, 20.0), "wilt": (78.8, 60.0)},
                {"micro-mass": 60.0, "pendigits": 60.0})
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert (v["suben"], v["mejoras_de_tiempo"], v["bajas_toleradas"]) == (0, 2, 1) and v["entra"] is True, v


def test_C5p_solo_mejoras_de_tiempo_entra_sin_ninguna_baja():
    a, b = _con({"micro-mass": (80.0, 20.0)}, {"micro-mass": 60.0})
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["entra"] is True and v["mejoras_de_tiempo"] == 1 and v["suben"] == 0


def test_C5p_un_empeora_bloquea_aunque_haya_muchas_mejoras():
    a, b = _con({"micro-mass": (90.0, 60.0), "mfeat-factors": (90.0, 60.0), "pendigits": (90.0, 60.0),
                 "wilt": (77.9, 60.0)})                                         # wilt: -2,1
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["entra"] is False and v["empeoran"] == 1 and v["suben"] == 3
    assert any("empeoran" in m for m in v["motivos_para_no_entrar"])


def test_C5p_la_baja_tolerada_sola_no_bloquea_y_sin_mejoras_no_entra_por_las_mejoras():
    a, b = _con({"wilt": (78.8, 60.0)})
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["empeoran"] == 0 and v["bajas_toleradas"] == 1 and v["entra"] is False
    assert any("no superan" in m for m in v["motivos_para_no_entrar"])


def test_C5p_una_mejora_de_tiempo_cuyo_entregado_cayo_a_CPU_no_cuenta_como_GPU():
    a, b = _con({"wilt": (80.0, 20.0)}, {"wilt": 60.0})
    b["conjuntos"]["wilt"]["campeon"] = TABM
    b["conjuntos"]["wilt"]["entrenamiento_del_entregado"] = {"dispositivo": "cpu", "cayo_a_cpu": True}
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["entra"] is False and v["mejoras_de_tiempo"] == 0 and v["mejoras_que_cayeron_a_cpu"] == ["wilt"]


# --- el corte (regla 9) ------------------------------------------------------------------------------------------
def test_C5p_el_corte_de_las_bajas_se_calcula_con_lo_que_falta():
    def cadena(n_bajas, n_mejoras):
        return ([_par(-1.5, nombre=f"b{i}") for i in range(n_bajas)]
                + [_par(2.0, nombre=f"m{i}") for i in range(n_mejoras)])
    # 5 bajas, 0 mejoras, faltan 3 de 8: 0 + 3 <= 5 -> corta
    assert "ni con los 3" in V.corta(cadena(5, 0), esperados=8)
    # 3 bajas, 0 mejoras, faltan 5 de 8: 0 + 5 > 3 -> sigue
    assert V.corta(cadena(3, 0), esperados=8) is None
    # el borde: 4 bajas, 0 mejoras, faltan 4 de 8: 0 + 4 <= 4 -> corta (con 4 mejoras empataría, y no basta: hay que SUPERAR)
    assert V.corta(cadena(4, 0), esperados=8) is not None
    # 4 bajas, 1 mejora, faltan 3 de 8: 1 + 3 <= 4 -> corta ; 4 bajas, 1 mejora, faltan 4 de 9: 5 > 4 -> sigue
    assert V.corta(cadena(4, 1), esperados=8) is not None
    assert V.corta(cadena(4, 1), esperados=9) is None
    # sin `esperados` solo se mira el último conjunto
    assert V.corta(cadena(5, 0)) is None


def test_C5p_el_corte_no_cuenta_como_mejora_la_de_un_entregado_caido_a_CPU():
    caida = _par(2.0, nombre="m0")
    caida["cayo_a_cpu_en_b"] = True
    cadena = [caida] + [_par(-1.5, nombre=f"b{i}") for i in range(3)]
    assert V.corta(cadena, esperados=7) is not None              # 3 bajas, 0 mejoras (la caída no cuenta), faltan 3


def test_C5p_el_corte_CORTA_en_un_empeora_y_en_dejar_de_completar():
    assert "empeora" in V.corta([_par(-2.0)])
    assert V.corta([_par(-2.0)], esperados=13) is not None


# --- con los 4 conjuntos medidos hoy en el portátil de Roberto (RTX 2000 Ada) -------------------------------------
HOY = Path(__file__).resolve().parent / "data" / "resultado_128_c5_portatil_4_de_control.json"


def test_C5p_con_los_4_medidos_hoy_NO_ENTRA_por_1_mejora_de_tiempo_frente_a_1_baja_tolerada():
    r = V.evaluar(json.loads(HOY.read_text()))
    clases = {c["nombre"]: c["clase"] for c in r["comparaciones"]}
    assert clases == {"dresses-sales": "igual", "climate-model-simulation-crashes": "igual",
                      "us_crime": "mejora_de_tiempo", "pc1": "baja_tolerada"}, clases
    d = {c["nombre"]: c["diferencia"] for c in r["comparaciones"]}
    assert d["dresses-sales"] == pytest.approx(-0.33, abs=0.005) and d["us_crime"] == pytest.approx(-0.08, abs=0.005)
    assert d["pc1"] == pytest.approx(-1.59, abs=0.005) and d["climate-model-simulation-crashes"] == 0.0
    v = r["veredicto"]
    assert (v["suben"], v["mejoras_de_tiempo"], v["bajas_toleradas"], v["empeoran"]) == (0, 1, 1, 0)
    assert v["suelo_de_ruido"] == 0.0 and v["entra"] is False
    assert any("no superan" in m for m in v["motivos_para_no_entrar"]), v["motivos_para_no_entrar"]
    # y la pasada de hoy NO se habría cortado en ninguno de los cuatro (solo uno es baja tolerada, faltaban 9)
    assert V.corta(r["comparaciones"], esperados=13) is None


def test_C5p_el_CLI_dice_las_clases_nuevas(tmp_path, capsys):
    V._main(["x", str(HOY)])
    sal = capsys.readouterr().out
    assert "mejora_de_tiempo" in sal and "baja_tolerada" in sal and "LA GPU NO ENTRA" in sal
