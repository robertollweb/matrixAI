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
            "recursos": {"vram_extra_mb": vram_extra}, "pared_estudio_s": pared,
            # 128-C5′ (H16): el CSV que midió; en los dos brazos el mismo (depende solo del nombre)
            "csv": {"filas": 100, "columnas": 5, "bytes": 1000, "sha256": f"sha-{nombre}"}}


def _huella(**cambios):
    """128-C5′ (H16): la huella de un brazo; los dos brazos de las pruebas traen la MISMA (compatibles)."""
    return {"imagen_id": "sha256:img", "sha256_medir_c5": "m" * 64, "sha256_veredicto": "v" * 64,
            "sha256_manifiesto": "f" * 64, "memoria": "16g", "cpus": "2", "entorno": {}, **cambios}


def _brazo(cifras, *, gpu=False, repeticion=None, **kw):
    conjuntos = {n: _rec(n, cifras.get(n, 80.0), gpu=gpu and n == "micro-mass", **kw) for n in TODOS}
    rep = {n: _rec(n, (repeticion or {}).get(n, cifras.get(n, 80.0))) for n in CONTROL}
    return {"conjuntos": conjuntos, "repeticion_de_control": rep, "huella": _huella()}


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


def test_H18_la_VRAM_de_mas_ya_NO_prueba_que_el_B_uso_la_GPU_se_ensena_como_auxiliar_y_nunca_acusa_al_A():
    b = _brazo({"micro-mass": 95.0}, vram_extra=3000.0)             # sin dispositivo declarado, pero 3 GB de VRAM
    v = V.evaluar(_doc(_brazo({}), b))["veredicto"]
    assert v["entra"] is False and v["instrumento_midio_bien"] is False, v
    assert any("evidencia auxiliar" in m and "no basta" in m for m in v["motivos_del_instrumento"]), v
    assert "micro-mass" in v["vram_auxiliar_en_b"] and v["vram_auxiliar_en_b"]["micro-mass"] == 3000.0
    a = _brazo({}, vram_extra=3000.0)                               # otra cosa usaba la GPU durante el brazo A: no se acusa
    assert V.presencia(a["conjuntos"], _brazo({}, gpu=True)["conjuntos"]) is None


def test_NO_ENTRA_si_el_brazo_A_entreno_en_la_GPU():
    a = _brazo({}, gpu=True)
    assert "brazo A" in V.presencia(a["conjuntos"], _brazo({}, gpu=True)["conjuntos"])


def test_los_dispositivos_de_los_intentos_de_TabM_tambien_cuentan_pero_los_vistos_en_todo_el_estado_no():
    rec = _rec("micro-mass"); rec["entrenamiento_del_entregado"] = None
    rec["dispositivos_de_tabm"] = {"cuda": 3, "cpu": 1}
    assert V.entreno_tabm_en_gpu(rec) is True
    # H18: `dispositivos_vistos` cuenta las declaraciones de TODO el estado, no de TabM: no prueba la GPU de TabM
    rec2 = _rec("micro-mass"); rec2["entrenamiento_del_entregado"] = None; rec2["dispositivos_vistos"] = {"cuda": 3}
    assert V.entreno_tabm_en_gpu(rec2) is False
    # ...pero para ACUSAR al brazo A basta cualquier `cuda` declarado en su estado
    assert V.declara_cuda_en_algun_sitio(rec2) is True
    assert "brazo A" in V.presencia({"micro-mass": rec2}, _brazo({}, gpu=True)["conjuntos"])


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
HOY = Path(__file__).resolve().parent / "data" / "resultado_128_c5_portatil_original_e2.json"


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


# =====================================================================================================================
# 128-C5′ — la revisión externa del 10-10 (H15, H16, H17, H18, R9.7, R9.8): el INSTRUMENTO. Cada prueba lleva su
# nombre porque cada una tiene su sabotaje (ver el informe `.agentes/128-gpu-h/informe.md`).
# =====================================================================================================================
ORIGINAL_E2 = Path(__file__).resolve().parent / "data" / "resultado_128_c5_portatil_original_e2.json"
NAN, INF = float("nan"), float("inf")


# --- H15: lo que no es un número no es una cifra ------------------------------------------------------------------
@pytest.mark.parametrize("malo", [NAN, INF, -INF, "no es un numero"])
def test_H15_una_cifra_de_test_no_finita_o_que_no_es_un_numero_es_INCOMPARABLE_no_igual(malo):
    rec = _rec("wilt")
    rec["test"]["metricas"]["auroc"] = malo
    with pytest.raises(V.Incomparable):
        V.cifra_de_test(rec)
    with pytest.raises(V.Incomparable):
        V.comparar(_rec("wilt", 80.0), rec, 0.0)                    # antes: NaN -> «igual»
    with pytest.raises(V.Incomparable):
        V.comparar(rec, _rec("wilt", 80.0), 0.0)


def test_H15_una_cifra_NaN_en_B_no_cuenta_como_mejora_de_tiempo_con_un_registro_entero(tmp_path):
    a = _brazo({})
    b = _brazo({"micro-mass": NAN}, gpu=True)
    b["conjuntos"]["micro-mass"]["pared_estudio_s"] = 1.0           # «1 s» frente a 60 s de A: antes, mejora_de_tiempo
    with pytest.raises(V.Incomparable):
        V.evaluar(_doc(a, b))


@pytest.mark.parametrize("d", [NAN, INF, -INF])
def test_H15_una_diferencia_no_finita_es_INCOMPARABLE(d):
    with pytest.raises(V.Incomparable):
        V.clase_por_diferencia(d, 0.0)


@pytest.mark.parametrize("suelo", [-0.5, -1.0, NAN, INF, None, "0"])
def test_H15_un_suelo_negativo_o_no_finito_es_INCOMPARABLE_en_todas_las_puertas(suelo):
    with pytest.raises(V.Incomparable):
        V.clase_por_diferencia(1.0, suelo)
    with pytest.raises(V.Incomparable):
        V.comparar(_rec("wilt", 80.0), _rec("wilt", 80.0), suelo)
    with pytest.raises(V.Incomparable):
        V.veredicto([], suelo, esperados=13)
    if suelo is not None:                                           # `None` en `suelo_dado` significa «no dado»
        with pytest.raises(V.Incomparable):
            V.evaluar(_doc(_brazo({}), _brazo({}, gpu=True)), suelo_dado=suelo)


@pytest.mark.parametrize("pared_a,pared_b", [(100.0, -1.0), (100.0, 0.0), (100.0, NAN), (100.0, INF), (100.0, None),
                                              (NAN, 1.0), (INF, 1.0), (0.0, 0.0), (-100.0, -200.0), (None, 1.0),
                                              ("100", 1.0)])
def test_H15_un_tiempo_no_finito_cero_negativo_o_ausente_NO_es_una_mejora_de_tiempo(pared_a, pared_b):
    assert V.mejora_de_tiempo(pared_a, pared_b) is False


def test_H15_lo_valido_sigue_valiendo_el_borde_de_la_mitad_exacta():
    assert V.mejora_de_tiempo(100.0, 50.0) is True and V.mejora_de_tiempo(355.7, 117.1) is True
    assert V.mejora_de_tiempo(100.0, 50.01) is False


@pytest.mark.parametrize("pared_a,pared_b,quien", [(100.0, -1.0, "B"), (-3.0, 10.0, "A")])
def test_H15_un_tiempo_NEGATIVO_es_un_instrumento_roto_INCOMPARABLE(pared_a, pared_b, quien):
    with pytest.raises(V.Incomparable) as e:
        V.comparar(_rec("wilt", 80.0, pared=pared_a), _rec("wilt", 80.0, pared=pared_b), 0.0)
    assert "negativo" in str(e.value) and quien in str(e.value)


# --- H16 (a): «no medido» no es «no completó» ----------------------------------------------------------------------
def test_H16a_sin_referencia_en_A_es_otra_clase_distinta_de_empieza_a_completar():
    sin = V.comparar(None, _rec("wilt", 80.0), 0.0)
    assert sin["clase"] == "sin_referencia" and sin["a"] is None and sin["b"] == pytest.approx(80.0)
    midio_y_fallo = V.comparar(_rec("wilt", estado="failed"), _rec("wilt", 80.0), 0.0)
    assert midio_y_fallo["clase"] == "empieza_a_completar"           # A MIDIÓ y no completó: esa sí es una mejora


def test_H16a_un_conjunto_sin_referencia_impide_entrar_lo_dice_y_no_cuenta_como_mejora():
    a = _brazo({})
    del a["conjuntos"]["Allstate_Claims_Severity"]
    b = _brazo({"micro-mass": 83.0}, gpu=True)                       # una mejora real: sin la ausencia, ENTRARÍA
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["sin_referencia"] == ["Allstate_Claims_Severity"] and v["entra"] is False
    assert v["suben"] == 1                                           # y la ausencia no suma como «A no completaba»
    assert v["instrumento_midio_bien"] is False
    assert any("sin referencia del brazo A" in m and "Allstate" in m for m in v["motivos_del_instrumento"]), v


def test_H16a_el_corte_para_en_un_conjunto_sin_referencia():
    c = V.comparar(None, _rec("wilt", 80.0), 0.0)
    assert "no tiene ese conjunto" in V.corta([c])


# --- H16 (b): los conjuntos son los del banco, por identidad -------------------------------------------------------
def test_H16b_el_BANCO_es_el_del_manifiesto_del_kit_en_su_orden():
    manifiesto = json.loads((_FASE0 / "kit_128_c5" / "datos" / "manifiesto.json").read_text())
    nombres = [c["nombre"] for c in sorted(manifiesto["conjuntos"], key=lambda x: x["orden"])]
    assert list(V.BANCO) == nombres
    assert list(V.CONTROL) == [c["nombre"] for c in sorted(manifiesto["conjuntos"], key=lambda x: x["orden"])
                               if c["es_control"]]


def test_H16b_un_nombre_inesperado_es_INCOMPARABLE_aunque_el_recuento_cuadre():
    a = _brazo({})
    b = _brazo({}, gpu=True)
    b["conjuntos"]["intruso"] = b["conjuntos"].pop("wilt")           # 13 conjuntos, pero uno NO es del banco
    b["conjuntos"]["intruso"]["nombre"] = "intruso"
    assert len(b["conjuntos"]) == 13
    with pytest.raises(V.Incomparable, match="intruso"):
        V.evaluar(_doc(a, b))
    a2 = _brazo({})
    a2["conjuntos"]["intruso"] = a2["conjuntos"].pop("wilt")
    with pytest.raises(V.Incomparable, match="intruso"):
        V.evaluar(_doc(a2, _brazo({}, gpu=True)))


def test_H16b_los_que_faltan_se_listan_por_nombre_y_el_esperado_del_resultado_tiene_que_ser_el_del_banco():
    a, b = _brazo({}), _brazo({"micro-mass": 83.0}, gpu=True)
    for n in ("wilt", "pendigits"):
        del b["conjuntos"][n]
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["faltan"] == ["wilt", "pendigits"] and any("faltan ['wilt', 'pendigits']" in m for m in v["motivos_para_no_entrar"])
    doc = _doc(a, _brazo({}, gpu=True)); doc["esperados"] = 12
    with pytest.raises(V.Incomparable, match="12"):
        V.evaluar(doc)


def test_H16b_una_repeticion_de_control_sin_su_medida_principal_es_INCOMPARABLE_no_un_KeyError():
    a, b = _brazo({}), _brazo({}, gpu=True)
    del b["conjuntos"]["pc1"]
    with pytest.raises(V.Incomparable, match="pc1"):
        V.evaluar(_doc(a, b))


def test_H16b_la_repeticion_de_control_tiene_que_ser_la_de_los_cuatro_de_control():
    a, b = _brazo({}), _brazo({}, gpu=True)
    b["repeticion_de_control"]["wilt"] = _rec("wilt")
    b["conjuntos"]["wilt"] = _rec("wilt")
    with pytest.raises(V.Incomparable, match="control"):
        V.evaluar(_doc(a, b))


# --- H16 (c): los dos brazos tienen que ser comparables ------------------------------------------------------------
@pytest.mark.parametrize("campo,valor", [("imagen_id", "sha256:otra"), ("sha256_medir_c5", "x" * 64),
                                         ("sha256_veredicto", "x" * 64), ("sha256_manifiesto", "x" * 64),
                                         ("memoria", "8g"), ("cpus", "4")])
def test_H16c_un_brazo_medido_con_otra_imagen_codigo_manifiesto_o_recursos_es_INCOMPARABLE(campo, valor):
    a, b = _brazo({}), _brazo({"micro-mass": 83.0}, gpu=True)
    b["huella"] = _huella(**{campo: valor})
    with pytest.raises(V.Incomparable) as e:
        V.evaluar(_doc(a, b))
    assert campo in str(e.value) and repr(valor) in str(e.value)


@pytest.mark.parametrize("campo", ["imagen_id", "sha256_medir_c5", "sha256_veredicto", "sha256_manifiesto", "memoria", "cpus"])
def test_H16c_lo_que_no_consta_en_la_huella_no_se_da_por_igual(campo):
    a, b = _brazo({}), _brazo({}, gpu=True)
    b["huella"] = {k: v for k, v in _huella().items() if k != campo}
    with pytest.raises(V.Incomparable, match=campo):
        V.evaluar(_doc(a, b))


def test_H16c_sin_huella_en_algun_brazo_no_hay_comparacion():
    a, b = _brazo({}), _brazo({}, gpu=True)
    del a["huella"]
    with pytest.raises(V.Incomparable, match="brazo A no trae su huella"):
        V.evaluar(_doc(a, b))


def test_H16c_el_entorno_solo_puede_diferir_en_lo_que_define_al_brazo():
    a, b = _brazo({}), _brazo({"micro-mass": 83.0}, gpu=True)
    a["huella"] = _huella(entorno={"MATRIXAI_TABM_DISPOSITIVO": "cpu"})
    b["huella"] = _huella(entorno={"MATRIXAI_TABM_DISPOSITIVO": "auto", "gpus": "all"})
    assert V.evaluar(_doc(a, b))["veredicto"]["entra"] is True       # la señal y la GPU: lo único que los distingue
    b["huella"] = _huella(entorno={"MATRIXAI_TABM_DISPOSITIVO": "auto", "gpus": "all", "OMP_NUM_THREADS": "8"})
    with pytest.raises(V.Incomparable, match="OMP_NUM_THREADS"):
        V.evaluar(_doc(a, b))


def test_H16c_el_CSV_de_un_conjunto_tiene_que_ser_el_mismo_en_los_dos_brazos_y_constar():
    a, b = _brazo({}), _brazo({}, gpu=True)
    b["conjuntos"]["wilt"]["csv"]["sha256"] = "otro"
    with pytest.raises(V.Incomparable, match="wilt.*CSV"):
        V.evaluar(_doc(a, b))
    a, b = _brazo({}), _brazo({}, gpu=True)
    del a["conjuntos"]["pc1"]["csv"]
    with pytest.raises(V.Incomparable, match="pc1.*CSV"):
        V.evaluar(_doc(a, b))


def test_H16c_con_el_JSON_real_las_dos_huellas_son_compatibles():
    doc = json.loads(ORIGINAL_E2.read_text())
    assert V.incompatibilidades_de_huellas(doc["brazos"]["A"]["huella"], doc["brazos"]["B"]["huella"]) == []


# --- H17: RAM y tiempo sobre los conjuntos comunes -------------------------------------------------------------------
def test_H17_la_RAM_se_resume_sobre_los_conjuntos_comunes_y_los_no_emparejados_van_aparte():
    a, b = _brazo({}), _brazo({"micro-mass": 83.0}, gpu=True)
    for n in TODOS:
        a["conjuntos"][n]["recursos"]["ram_pico_mb"] = 9000.0 if n == "APSFailure" else 1000.0
    for n in ("dresses-sales", "us_crime"):
        b["conjuntos"][n]["recursos"]["ram_pico_mb"] = 1800.0
    for n in TODOS:
        if n not in CONTROL and n != "micro-mass":
            del b["conjuntos"][n]
    b["conjuntos"]["micro-mass"]["recursos"]["ram_pico_mb"] = 1500.0
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["ram_pico_mb_por_brazo"] == {"A": 1000.0, "B": 1800.0}, v["ram_pico_mb_por_brazo"]   # NO 9000 frente a 1800
    r = v["resumen_emparejado"]
    assert r["conjuntos_comunes"] == CONTROL + ["micro-mass"]
    assert r["ram_pico_mb_por_conjunto"]["us_crime"] == {"A": 1000.0, "B": 1800.0}
    assert r["no_emparejados"]["solo_en_A"]["APSFailure"]["ram_pico_mb"] == 9000.0     # el 9000 sigue visible, APARTE
    assert "APSFailure" not in r["no_emparejados"]["solo_en_B"] and "APSFailure" not in r["ram_pico_mb_por_conjunto"]


def test_H17_la_pared_tambien_va_sobre_los_comunes_y_un_dato_ausente_saca_al_conjunto_de_la_suma():
    a, b = _brazo({}), _brazo({}, gpu=True)
    for n in TODOS:
        if n not in ("wilt", "pc1"):
            del b["conjuntos"][n]
    a["conjuntos"]["wilt"]["pared_estudio_s"] = 100.0; b["conjuntos"]["wilt"]["pared_estudio_s"] = 40.0
    a["conjuntos"]["pc1"]["pared_estudio_s"] = 300.0; b["conjuntos"]["pc1"]["pared_estudio_s"] = None
    r = V.resumen_emparejado(a["conjuntos"], b["conjuntos"])
    assert r["pared_total_s"] == {"A": 100.0, "B": 40.0}             # pc1 no tiene la pared de B: no se suma solo la de A
    assert r["no_emparejados"]["sin_pared_en_un_brazo"] == ["pc1"]


def test_H17_con_el_JSON_real_la_RAM_de_los_cuatro_comunes_es_la_que_calculo_la_revision():
    r = V.evaluar(json.loads(ORIGINAL_E2.read_text()))["veredicto"]
    assert r["ram_pico_mb_por_brazo"] == {"A": 1342.5, "B": 1786.8}      # y NO 8673,4 frente a 1786,8
    assert all(p["B"] > p["A"] for p in r["resumen_emparejado"]["ram_pico_mb_por_conjunto"].values())
    assert "APSFailure" in r["resumen_emparejado"]["no_emparejados"]["solo_en_A"]


# --- H18: la presencia de la GPU se demuestra con la declaración ------------------------------------------------------
def test_H18_la_declaracion_del_dispositivo_en_los_intentos_de_TabM_prueba_la_GPU_sin_VRAM():
    a = _brazo({})
    b = _brazo({"micro-mass": 83.0})
    b["conjuntos"]["micro-mass"]["dispositivos_de_tabm"] = {"cuda": 2}
    assert V.evaluar(_doc(a, b))["veredicto"]["entra"] is True


def test_H18_la_VRAM_sin_declaracion_no_basta_y_el_motivo_la_enseña_como_auxiliar():
    v = V.evaluar(_doc(_brazo({}), _brazo({"micro-mass": 95.0}, vram_extra=3000.0)))["veredicto"]
    assert v["entra"] is False and v["instrumento_midio_bien"] is False
    motivo = next(m for m in v["motivos_del_instrumento"] if "ninguna declaración de dispositivo" in m)
    assert "auxiliar" in motivo and "no basta" in motivo and "micro-mass" in motivo


def test_H18_el_comparar_enseña_la_VRAM_auxiliar_pero_tabm_en_gpu_es_solo_la_declaracion():
    rec = _rec("micro-mass", vram_extra=3000.0)
    c = V.comparar(_rec("micro-mass"), rec, 0.0)
    assert c["tabm_en_gpu_en_b"] is False and c["vram_auxiliar_en_b_mb"] == 3000.0
    assert V.vram_auxiliar(_rec("x", vram_extra=10.0)) is None and V.vram_auxiliar(_rec("x", vram_extra=NAN)) is None


# --- R9.7: el instrumento y el producto, por separado ------------------------------------------------------------------
def test_R97_un_empeora_es_de_la_regla_el_instrumento_midio_bien():
    v = V.evaluar(_doc(_brazo({}), _brazo({"micro-mass": 90.0, "wilt": 70.0}, gpu=True)))["veredicto"]
    assert (v["instrumento_midio_bien"], v["la_regla_se_cumple"], v["producto_entra"]) == (True, False, False)
    assert v["motivos_del_instrumento"] == [] and any("empeoran" in m for m in v["motivos_de_la_regla"])


def test_R97_un_fallo_de_presencia_es_del_instrumento_aunque_la_regla_se_cumpliera():
    v = V.evaluar(_doc(_brazo({}), _brazo({"micro-mass": 90.0}, gpu=False)))["veredicto"]
    assert (v["instrumento_midio_bien"], v["la_regla_se_cumple"], v["producto_entra"]) == (False, True, False)
    assert v["motivos_de_la_regla"] == [] and v["motivos_del_instrumento"]


def test_R97_entra_es_el_mismo_valor_que_producto_entra_y_solo_si_las_dos_cosas():
    v = V.evaluar(_doc(_brazo({}), _brazo({"micro-mass": 83.0}, gpu=True)))["veredicto"]
    assert (v["instrumento_midio_bien"], v["la_regla_se_cumple"], v["producto_entra"], v["entra"]) == (True, True, True, True)
    assert v["motivos_para_no_entrar"] == []


def test_R97_una_pasada_incompleta_que_no_cortaba_la_regla_es_del_instrumento_y_si_la_cortaba_es_de_la_regla():
    a = _brazo({})
    b = _brazo({"micro-mass": 83.0}, gpu=True)
    for n in TODOS[4:]:
        if n != "micro-mass":
            del b["conjuntos"][n]
    b["corte"] = {"tipo": "regla_9", "motivo": "lo cortó otra regla (la E2)", "tras": "micro-mass"}
    v = V.evaluar(_doc(a, b))["veredicto"]
    assert v["instrumento_midio_bien"] is False and any("no la cortó la regla C5′" in m for m in v["motivos_del_instrumento"])
    # ahora SÍ la corta esta regla: el último medido empeora
    b2 = _brazo({"micro-mass": 83.0, "mfeat-factors": 70.0}, gpu=True)
    for n in TODOS[6:]:
        del b2["conjuntos"][n]
    b2["corte"] = {"tipo": "regla_9", "motivo": "mfeat-factors empeora", "tras": "mfeat-factors"}
    v2 = V.evaluar(_doc(a, b2))["veredicto"]
    assert v2["instrumento_midio_bien"] is True and v2["producto_entra"] is False
    assert any("solo 6 de 13" in m for m in v2["motivos_de_la_regla"])


def test_R97_un_incomparable_deja_la_regla_sin_aplicar_no_en_falso():
    v = V.sin_veredicto_por_incomparable("otra imagen")
    assert v["instrumento_midio_bien"] is False and v["la_regla_se_cumple"] is None and v["producto_entra"] is False
    assert v["entra"] is False and v["motivos_para_no_entrar"] == ["incomparable: otra imagen"]


# --- R9.8: el original de E2, evaluado con C5′ como evaluación NUEVA ---------------------------------------------------
def test_R98_la_evaluacion_nueva_cita_al_original_con_su_hash_y_no_lo_toca():
    antes = ORIGINAL_E2.read_bytes()
    doc = json.loads(antes)
    r = V.evaluacion_nueva(doc, origen={"fichero": ORIGINAL_E2.name, "sha256": V._sha256_de(ORIGINAL_E2)})
    assert ORIGINAL_E2.read_bytes() == antes and doc == json.loads(antes)          # ni en disco ni en memoria
    e = r["evaluacion"]
    assert e["tipo"] == "evaluacion_nueva_de_un_resultado_ya_medido" and "C5′" in e["regla"]
    assert e["original"]["fichero"] == ORIGINAL_E2.name and e["original"]["sha256"] == V._sha256_de(ORIGINAL_E2)
    assert e["original"]["entra_segun_el_original"] is False and "ninguno baja" in (e["original"]["regla_del_original"] or "")
    assert e["sha256_veredicto_128_c5"] == V._sha256_de(V.__file__)
    clases = {c["nombre"]: c["clase"] for c in r["comparaciones"]}
    assert clases == {"dresses-sales": "igual", "climate-model-simulation-crashes": "igual",
                      "us_crime": "mejora_de_tiempo", "pc1": "baja_tolerada"}
    v = r["veredicto"]
    assert v["producto_entra"] is False and v["la_regla_se_cumple"] is False
    # el original se cortó con la E2 en el primer conjunto, y C5′ no lo habría cortado: el instrumento dice «incompleta»
    assert v["instrumento_midio_bien"] is False and any("no la cortó la regla C5′" in m for m in v["motivos_del_instrumento"])


def test_R98_si_los_brazos_no_son_comparables_la_evaluacion_nueva_lo_dice_en_vez_de_fallar():
    doc = json.loads(ORIGINAL_E2.read_text())
    doc["brazos"]["B"]["huella"]["imagen_id"] = "sha256:otra"
    r = V.evaluacion_nueva(doc, origen={"fichero": "x.json", "sha256": "0" * 64})
    assert r["veredicto"]["instrumento_midio_bien"] is False and r["comparaciones"] == []
    assert any("imagen_id" in m for m in r["veredicto"]["motivos_del_instrumento"])


def test_R98_el_CLI_escribe_la_evaluacion_en_un_fichero_NUEVO_y_se_niega_a_sobrescribir(tmp_path, capsys):
    copia = tmp_path / "resultado_128_c5.json"
    copia.write_bytes(ORIGINAL_E2.read_bytes())
    nueva = tmp_path / "evaluacion_c5p.json"
    assert V._main(["x", str(copia), "--evaluacion-nueva", str(nueva)]) == 0
    sal = capsys.readouterr().out
    assert "cita a resultado_128_c5.json" in sal and "INSTRUMENTO: NO midió bien" in sal and "LA GPU NO ENTRA" in sal
    escrito = json.loads(nueva.read_text())
    assert escrito["evaluacion"]["original"]["sha256"] == V._sha256_de(copia) == V._sha256_de(ORIGINAL_E2)
    assert copia.read_bytes() == ORIGINAL_E2.read_bytes()
    assert V._main(["x", str(copia), "--evaluacion-nueva", str(nueva)]) == 2        # ya existe
    assert V._main(["x", str(copia), "--evaluacion-nueva", str(copia)]) == 2        # es el original
    assert copia.read_bytes() == ORIGINAL_E2.read_bytes() and json.loads(nueva.read_text()) == escrito


# =====================================================================================================================
# 128, H10 de la revisión externa: la lectura de NO INFERIORIDAD, informativa y aparte. La regla C5′ no cambia.
# =====================================================================================================================
def test_H10_con_el_JSON_real_enseña_la_diferencia_por_conjunto_la_peor_la_media_y_cuantos_bajan_y_suben():
    r = V.evaluar(json.loads(ORIGINAL_E2.read_text()))
    ni = r["veredicto"]["no_inferioridad"]
    assert ni["decide"] is False and ni["emparejados"] == 4
    assert set(ni["por_conjunto"]) == {"dresses-sales", "climate-model-simulation-crashes", "us_crime", "pc1"}
    assert ni["por_conjunto"]["pc1"] == pytest.approx(-1.5858, abs=1e-3)
    assert ni["peor"]["conjunto"] == "pc1" and ni["peor"]["diferencia"] == pytest.approx(-1.5858, abs=1e-3)
    assert ni["media"] == pytest.approx(-0.49965, abs=1e-4)
    assert (ni["bajan"], ni["suben"], ni["iguales"]) == (3, 0, 1)


def test_H10_no_entra_en_la_decision_la_regla_C5p_da_lo_mismo_y_veredicto_no_la_lleva():
    r = V.evaluar(json.loads(ORIGINAL_E2.read_text()))
    v = r["veredicto"]
    assert v["entra"] is False and v["producto_entra"] is False and (v["suben"], v["mejoras_de_tiempo"], v["bajas_toleradas"]) == (0, 1, 1)
    # `veredicto()` (la decisión) no sabe de la no inferioridad: lo que decide no depende de ella
    decision = V.veredicto(r["comparaciones"], v["suelo_de_ruido"], esperados=13)
    assert "no_inferioridad" not in decision
    # y NO hay motivo de la regla ni del instrumento que cite la no inferioridad
    assert not any("inferioridad" in m for m in v["motivos_para_no_entrar"])


def test_H10_una_pasada_donde_todas_las_diferencias_suben_o_bajan_se_cuenta_bien():
    comps = [_par(+3.0, nombre="a"), _par(-1.5, nombre="b"), _par(0.0, nombre="c"), _par(+1.0, nombre="d")]
    ni = V.no_inferioridad(comps)
    assert ni["emparejados"] == 4 and ni["peor"] == {"conjunto": "b", "diferencia": pytest.approx(-1.5)}
    assert (ni["bajan"], ni["suben"], ni["iguales"]) == (1, 2, 1) and ni["media"] == pytest.approx(0.625)


def test_H10_sin_conjuntos_emparejados_todo_es_None_no_cero():
    sin_ref = V.comparar(None, _rec("wilt", 80.0), 0.0)             # sin referencia en A: no hay diferencia
    ni = V.no_inferioridad([sin_ref])
    assert ni["emparejados"] == 0 and ni["peor"] is None and ni["media"] is None and ni["bajan"] is None
    assert ni["suben"] is None and ni["iguales"] is None and ni["decide"] is False


def test_H10_el_CLI_la_dice_y_dice_que_no_decide(capsys):
    V._main(["x", str(ORIGINAL_E2)])
    sal = capsys.readouterr().out
    assert "NO INFERIORIDAD (informativa, no decide)" in sal and "peor pc1 -1.59" in sal and "LA GPU NO ENTRA" in sal
