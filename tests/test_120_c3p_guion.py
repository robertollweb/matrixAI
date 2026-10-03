"""120 · C3′ — el guion de la medida (`referencia_120_r0.py --r1gpu` / `--c3p`, enmienda 7).

Nada de esto arranca un contenedor ni genera un CSV: lo que tiene efectos fuera se sustituye por un fallo RUIDOSO
(02-10: un sabotaje de una guarda, con una prueba que confiaba en ella, arrancó un contenedor de verdad y reescribió
un artefacto). `correr` es un doble que entrega registros fabricados; el guion hace lo demás de verdad.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"


def _cargar(nombre: str, fichero: str):
    spec = importlib.util.spec_from_file_location(nombre, _FASE0 / fichero)
    mod = importlib.util.module_from_spec(spec)
    antes = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = antes
    return mod


G = _cargar("referencia_120_r0_c3p_en_pruebas", "referencia_120_r0.py")
V3 = G.veredicto_120_c3p
TABM, DENSA = V3.MOTOR_TABM, V3.MOTOR_DENSA
HUMO = list(G.CONTROL)                      # dresses-sales, climate, us_crime, pc1: los 4 primeros del protocolo
TAREA = {"us_crime": "regression"}


def _rec(nombre, *, test=0.80, campeon="lightgbm", motores=("lightgbm", "sklearn.hgb", "baseline", DENSA),
         estado="completed"):
    tarea = TAREA.get(nombre, "binary_classification")
    control = G.CONTROL.get(nombre, {"lightgbm": 0.5, "sklearn.hgb": 0.5})
    return {"nombre": nombre, "tarea": tarea, "estado": estado, "campeon": campeon if estado == "completed" else None,
            "media_de_la_seleccion": {"metrica": "auroc", "motores": [
                {"motor": m, "valor": v, "compite": True} for m, v in control.items()]},
            "test": {"rol": "test", "evidencia": "independent_test",
                     "metricas": {V3.METRICA_DE_CIERRE[tarea]: test}},
            "intentos": [{"motor": m} for m in motores], "pared_estudio_s": 60.0, "http": 200}


def _banco(*extra):
    return [{"nombre": n, "orden": i} for i, n in enumerate(HUMO + list(extra), 1)]


@pytest.fixture
def entorno(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "SALIDA", tmp_path / "salida.json")
    monkeypatch.setattr(G, "ESTADOS", tmp_path / "estados")
    (tmp_path / "estados").mkdir()
    for peligrosa in ("arrancar_contenedor", "cargar_generador", "comprobar_bytes", "estudiar"):
        monkeypatch.setattr(G, peligrosa, lambda *a, _n=peligrosa, **k: pytest.fail(f"ejecutó lo real: {_n}"))
    monkeypatch.setattr(G, "_contenedor", None)
    return tmp_path


def _datos():
    return {"procedencia": {}, "conjuntos": {}, "control": None, "corte": None, "comparaciones": {}}


def _correr_desde(datos, registros, repeticion=None):
    """Un doble de `correr`: el registro de cada conjunto sale de `registros` (o de `repeticion` al repetir)."""
    llamadas = []

    def correr(b, destino=None, reusar=True):
        n = b["nombre"]
        llamadas.append((n, destino is None))
        fuente = registros if destino is None or repeticion is None else repeticion
        (datos["conjuntos"] if destino is None else destino)[n] = fuente[n]
    return correr, llamadas


# ------------------------------------------------------------------ lo que se guarda de cada estudio
def test_registrar_guarda_la_evaluacion_del_test():
    est = {"estado": "completed", "seleccion": {"candidate_engine": "lightgbm", "evaluacion_final": {
        "schema": "matrixai.estudio.evaluation_result", "evaluated_role": "test", "evidence": "independent_test",
        "metrics": [{"metric_id": "auroc", "value": 0.91}, {"metric_id": "accuracy", "value": 0.8}]}}}
    rec = G.registrar("pc1", {"data_id": 1, "tarea": "binary_classification", "objetivo": "y"}, {}, est,
                      1.0, 200, None, ["0", "0", "0"], "x")
    assert rec["test"] == {"rol": "test", "evidencia": "independent_test",
                           "metricas": {"auroc": 0.91, "accuracy": 0.8}}
    assert V3.cifra_de_test(rec) == pytest.approx(91.0)
    assert G.registrar("pc1", {"data_id": 1, "tarea": "x", "objetivo": "y"}, {}, {"estado": "rechazado"},
                       1.0, 400, None, ["0"] * 3, "x")["test"] is None


# ------------------------------------------------------------------ las guardas de los argumentos
@pytest.mark.parametrize("args, motivo", [
    (["--c3p"], "van con --contra"),
    (["--r1gpu", "--c3p", "--contra", "X"], "una cada vez"),
    (["--r1gpu"], "--salida PROPIA"),                      # sin --salida, escribiría encima de R0
    (["--r1gpu", "--salida", "S", "--banco", "B"], "sin --banco"),
    (["--c3p", "--contra", "R1", "--salida", "S"], "suelo_de_ruido"),   # una R1 de C2 no es una R1-GPU
])
def test_las_guardas_de_C3p_paran_antes_de_tocar_nada(monkeypatch, entorno, args, motivo):
    r1 = entorno / "r1.json"
    r1.write_text((_FASE0 / "referencia_120_r1.json").read_text())
    args = [str(r1) if x == "R1" else str(entorno / "s.json") if x == "S" else x for x in args]
    monkeypatch.setattr(sys, "argv", ["referencia_120_r0.py", *args])
    monkeypatch.setattr(G, "guardar", lambda *a, **k: pytest.fail("la guarda dejó pasar: guardar"))
    with pytest.raises(SystemExit, match=motivo):
        G.main()


# ------------------------------------------------------------------ R1-GPU: el suelo de ruido
def test_r1gpu_repite_los_4_de_humo_tras_el_control_y_escribe_el_suelo(entorno):
    datos = _datos()
    regs = {n: _rec(n) for n in HUMO + ["wilt"]}
    rep = {n: _rec(n) for n in HUMO}
    rep["pc1"] = _rec("pc1", test=0.803)                  # 0,3 puntos al repetir
    correr, llamadas = _correr_desde(datos, regs, rep)
    G.correr_c3p("r1gpu", _banco("wilt"), datos, correr, None)
    assert datos["control"]["cuadra"] is True
    assert datos["suelo_de_ruido"] == pytest.approx(0.3)
    assert sorted(datos["repeticion_de_control"]) == sorted(HUMO)
    # El suelo se escribe ANTES de medir el resto: la repetición va justo tras el 4.º de humo.
    assert [n for n, _ in llamadas] == HUMO + HUMO + ["wilt"]
    assert datos["corte"] is None and "veredicto" not in datos
    assert json.loads(G.SALIDA.read_text())["suelo_de_ruido"] == pytest.approx(0.3)


def test_r1gpu_sin_torch_o_con_TabM_es_un_PARO(entorno):
    datos = _datos()
    regs = {n: _rec(n) for n in HUMO}
    regs["dresses-sales"] = _rec("dresses-sales", motores=("lightgbm", "baseline"))
    correr, _ = _correr_desde(datos, regs)
    with pytest.raises(SystemExit):
        G.correr_c3p("r1gpu", _banco(), datos, correr, None)
    assert datos["corte"]["tipo"] == "instrumento" and "torch" in datos["corte"]["motivo"]


def test_r1gpu_un_control_que_no_cuadra_para(entorno):
    datos = _datos()
    regs = {n: _rec(n) for n in HUMO}
    regs["pc1"]["media_de_la_seleccion"]["motores"][0]["valor"] += 0.01
    correr, _ = _correr_desde(datos, regs)
    with pytest.raises(SystemExit):
        G.correr_c3p("r1gpu", _banco(), datos, correr, None)
    assert datos["control"]["cuadra"] is False and "suelo_de_ruido" not in datos


# ------------------------------------------------------------------ C3′ contra R1-GPU
def _contra(suelo=0.0, **tests):
    return {"suelo_de_ruido": suelo, "control": {"cuadra": True},
            "conjuntos": {n: _rec(n, test=tests.get(n, 0.80)) for n in HUMO + ["wilt", "pendigits"]}}


def test_c3p_sin_TabM_en_un_estudio_es_un_PARO(entorno):
    datos = _datos()
    correr, _ = _correr_desde(datos, {n: _rec(n) for n in HUMO})
    with pytest.raises(SystemExit):
        G.correr_c3p("c3p", _banco(), datos, correr, _contra())
    assert "TabM" in datos["corte"]["motivo"] and datos["corte"]["tras"] == HUMO[0]


def test_c3p_la_paridad_para_si_sin_TabM_de_campeon_cambia_la_cifra(entorno):
    datos = _datos()
    regs = {n: _rec(n, motores=("lightgbm", DENSA, TABM)) for n in HUMO}
    regs["climate-model-simulation-crashes"]["test"]["metricas"]["auroc"] = 0.8001
    correr, _ = _correr_desde(datos, regs)
    with pytest.raises(SystemExit):
        G.correr_c3p("c3p", _banco(), datos, correr, _contra())
    assert "R1-GPU" in datos["corte"]["motivo"] and datos["corte"]["tras"].startswith("climate")


def test_c3p_corta_por_la_regla_9_y_da_el_veredicto(entorno):
    datos = _datos()
    dos = ("lightgbm", DENSA, TABM)
    regs = {n: _rec(n, motores=dos) for n in HUMO + ["wilt", "pendigits"]}
    regs["wilt"] = _rec("wilt", campeon=TABM, test=0.77, motores=dos)     # −3 puntos
    correr, llamadas = _correr_desde(datos, regs)
    G.correr_c3p("c3p", _banco("wilt", "pendigits"), datos, correr, _contra())
    assert datos["corte"]["tipo"] == "regla_9" and datos["corte"]["sin_medir"] == ["pendigits"]
    assert [n for n, _ in llamadas] == HUMO + ["wilt"]
    assert datos["veredicto"]["mejora"] is False and datos["veredicto"]["bajan_2"] == 1


def test_c3p_mejora_cuando_TabM_gana_y_sube(entorno):
    datos = _datos()
    dos = ("lightgbm", DENSA, TABM)
    regs = {n: _rec(n, motores=dos) for n in HUMO + ["wilt"]}
    regs["wilt"] = _rec("wilt", campeon=TABM, test=0.83, motores=dos)
    correr, _ = _correr_desde(datos, regs)
    G.correr_c3p("c3p", _banco("wilt"), datos, correr, _contra())
    assert datos["corte"] is None
    assert datos["veredicto"]["mejora"] is True and datos["veredicto"]["tabm_campeon_en"] == ["wilt"]
    assert datos["comparaciones"]["wilt"]["diferencia"] == pytest.approx(3.0)


# ------------------------------------------------------------------ R1-GPU: la misma imagen, con TabM apagado
def test_r1gpu_apaga_TabM_con_su_interruptor_y_c3p_no():
    assert G.entorno_extra_de("r1gpu") == {"MATRIXAI_ESTUDIO_SIN_TABM": "1"}
    assert G.entorno_extra_de("c3p") == {}
    assert G.entorno_extra_de(None) == {}


def test_el_contenedor_lleva_el_entorno_extra(monkeypatch):
    llamadas = []

    class _Hecho:
        returncode, stderr, stdout = 0, "", ""

    monkeypatch.setattr(G.subprocess, "run", lambda args, **k: llamadas.append(list(args)) or _Hecho())
    monkeypatch.setattr(G, "pedir", lambda *a, **k: {})
    monkeypatch.setattr(G.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(G.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(G, "_contenedor", None)
    monkeypatch.setattr(G, "ENTORNO_EXTRA", {"MATRIXAI_ESTUDIO_SIN_TABM": "1"})
    G.arrancar_contenedor(None)
    docker = llamadas[0]
    assert docker[:2] == ["docker", "run"]
    i = docker.index("MATRIXAI_ESTUDIO_SIN_TABM=1")
    assert docker[i - 1] == "-e" and docker[-1] == G.IMAGEN
    llamadas.clear()
    monkeypatch.setattr(G, "ENTORNO_EXTRA", {})
    G.arrancar_contenedor(None)
    assert not any("SIN_TABM" in x for x in llamadas[0])


# ------------------------------------------------------------------ re-auditoría de 120-C6: nada de política sin pedirla
def test_ninguna_pide_los_arboles_de_hoy_al_contenedor():
    assert G.entorno_extra_de(None, ninguna=True) == {"MATRIXAI_POLITICA_DE_ARBOLES": "ninguna"}
    assert G.entorno_extra_de("r1gpu", ninguna=True) == {"MATRIXAI_ESTUDIO_SIN_TABM": "1",
                                                          "MATRIXAI_POLITICA_DE_ARBOLES": "ninguna"}


def test_ninguna_y_politica_a_la_vez_paran(monkeypatch, entorno):
    r1 = entorno / "r1.json"
    r1.write_text((_FASE0 / "referencia_120_r1.json").read_text())
    monkeypatch.setattr(sys, "argv", ["referencia_120_r0.py", "--politica", "l2_0", "--contra", str(r1),
                                      "--ninguna", "--salida", str(entorno / "s.json")])
    monkeypatch.setattr(G, "guardar", lambda *a, **k: pytest.fail("la guarda dejó pasar: guardar"))
    with pytest.raises(SystemExit, match="una u otra"):
        G.main()


def test_c3p_un_arbol_que_declara_una_politica_sin_pedirla_es_un_PARO(entorno):
    """En una imagen con 120-C6 sin torch, los medianos llevarían la política de producción: medir eso como si
    fueran los árboles de hoy sería medir otra cosa sin decirlo."""
    datos = _datos()
    dos = ("lightgbm", DENSA, TABM)
    regs = {n: _rec(n, motores=dos) for n in HUMO}
    regs["dresses-sales"]["intentos"] = [{"motor": m} for m in dos]
    regs["dresses-sales"]["perfiles_declarados"] = [
        {"motor": "lightgbm", "candidato": "lightgbm-csv-p0-r0", "estado": "completed", "declara": True,
         "tamano": "mediano", "l2": 1.0, "aplicada": True}]
    correr, _ = _correr_desde(datos, regs)
    with pytest.raises(SystemExit):
        G.correr_c3p("c3p", _banco(), datos, correr, _contra())
    assert "SIN pedirla" in datos["corte"]["motivo"]


# ------------------------------------------------------------------ C3″ (enmienda 8): el paquete CPU
SOBRE_CPU = {"valor": "produccion", "activada_por": "produccion_densa_fuera", "densa_fuera": True,
             "motores_sin_politica": []}
_ARBOL_COMPLETADO = [{"motor": "lightgbm", "candidato": "lightgbm-csv-p0-r0", "estado": "completed", "declara": True,
                      "tamano": "pequeno", "l2": 1.0, "aplicada": False}]


@pytest.mark.parametrize("sobre, para", [
    (SOBRE_CPU, False),
    (None, True),                                                       # sin sobre
    ({**SOBRE_CPU, "valor": "c2b_l2_1", "activada_por": "variable_de_medida"}, True),   # una de medida
    ({**SOBRE_CPU, "activada_por": "produccion_sin_torch"}, True),      # la de sin torch
    ({**SOBRE_CPU, "densa_fuera": None}, True),
    ({**SOBRE_CPU, "motores_sin_politica": ["sklearn.hgb"]}, True),
])
def test_el_paquete_cpu_exige_el_sobre_de_produccion_con_la_densa_fuera(sobre, para):
    rec = _rec("x", motores=("lightgbm", "baseline"))
    rec["perfiles_declarados"] = _ARBOL_COMPLETADO
    rec["politica_de_arboles"] = sobre
    assert (G.sobre_del_paquete_cpu_o_motivo(rec) is not None) == para
    sin_arboles = dict(rec, perfiles_declarados=[])                    # nada que comprobar: no para
    assert G.sobre_del_paquete_cpu_o_motivo(sin_arboles) is None


def test_la_referencia_es_la_misma_imagen_y_con_el_entorno_de_su_modo():
    ok = {"procedencia": {"imagen_id": "sha256:a", "entorno_extra": G.entorno_extra_de("rcpu")}}
    assert G.contra_o_motivo(ok, "c3s", "sha256:a") is None
    assert "otra imagen" in G.contra_o_motivo(ok, "c3s", "sha256:b")
    assert "entorno" in G.contra_o_motivo(ok, "c3p", "sha256:a")        # una R-CPU no sirve a C3′
    assert G.contra_o_motivo(None, None, "sha256:a") is None


def test_ninguna_no_con_el_paquete_cpu(monkeypatch, entorno):
    monkeypatch.setattr(sys, "argv", ["referencia_120_r0.py", "--rcpu", "--ninguna", "--salida", str(entorno / "s.json")])
    monkeypatch.setattr(G, "guardar", lambda *a, **k: pytest.fail("la guarda dejó pasar: guardar"))
    with pytest.raises(SystemExit, match="no con --rcpu"):
        G.main()


def test_el_paquete_cpu_apaga_la_densa_y_rcpu_ademas_tabm():
    assert G.entorno_extra_de("rcpu") == {"MATRIXAI_ESTUDIO_SIN_DENSA": "1", "MATRIXAI_ESTUDIO_SIN_TABM": "1"}
    assert G.entorno_extra_de("c3s") == {"MATRIXAI_ESTUDIO_SIN_DENSA": "1"}


def test_presencia_en_el_paquete_cpu_sin_densa():
    sin_redes = _rec("x", motores=("lightgbm", "sklearn.hgb", "baseline"))
    assert V3.presencia(sin_redes, c3p=False, con_densa=False) is None                       # R-CPU
    assert "densa anterior" in V3.presencia(_rec("x"), c3p=False, con_densa=False)          # con densa: PARO
    con_tabm = _rec("x", motores=("lightgbm", "baseline", TABM))
    assert V3.presencia(con_tabm, c3p=True, con_densa=False) is None                         # C3″
    assert "TabM" in V3.presencia(sin_redes, c3p=True, con_densa=False)


def test_rcpu_fija_el_suelo_y_admite_la_politica_de_produccion(entorno):
    datos = _datos()
    sin_redes = ("lightgbm", "sklearn.hgb", "baseline")
    regs = {n: _rec(n, motores=sin_redes) for n in HUMO + ["wilt"]}
    # en un mediano del paquete CPU, la política de C6 SE ESPERA: no es un PARO
    regs["wilt"]["perfiles_declarados"] = [{"motor": "lightgbm", "candidato": "lightgbm-csv-p0-r0",
                                            "estado": "completed", "declara": True, "tamano": "mediano",
                                            "l2": 1.0, "aplicada": True}]
    regs["wilt"]["politica_de_arboles"] = dict(SOBRE_CPU)
    correr, _ = _correr_desde(datos, regs, {n: _rec(n, motores=sin_redes) for n in HUMO})
    G.correr_c3p("rcpu", _banco("wilt"), datos, correr, None)
    assert datos["corte"] is None and datos["suelo_de_ruido"] == 0.0


def test_c3s_con_la_densa_anterior_es_un_PARO(entorno):
    datos = _datos()
    regs = {n: _rec(n, motores=("lightgbm", "baseline", TABM, DENSA)) for n in HUMO}
    correr, _ = _correr_desde(datos, regs)
    contra = _contra()
    contra["conjuntos"] = {n: _rec(n, motores=("lightgbm", "baseline")) for n in HUMO}
    with pytest.raises(SystemExit):
        G.correr_c3p("c3s", _banco(), datos, correr, contra)
    assert "densa anterior" in datos["corte"]["motivo"]


@pytest.mark.parametrize("medida, modo_de_la_referencia", [("--c3s", "r1gpu"), ("--c3p", "rcpu"), ("--c3s", None)])
def test_cada_medida_exige_su_referencia(monkeypatch, entorno, medida, modo_de_la_referencia):
    ref = {"suelo_de_ruido": 0.0, "control": {"cuadra": True}, "conjuntos": {},
           "procedencia": {"modo": modo_de_la_referencia}}
    r = entorno / "ref.json"
    r.write_text(json.dumps(ref))
    monkeypatch.setattr(sys, "argv", ["referencia_120_r0.py", medida, "--contra", str(r),
                                      "--salida", str(entorno / "s.json")])
    monkeypatch.setattr(G, "guardar", lambda *a, **k: pytest.fail("la guarda dejó pasar: guardar"))
    with pytest.raises(SystemExit, match="no es una referencia"):
        G.main()


def test_rcpu_sin_el_sobre_de_produccion_es_un_PARO_en_el_flujo(entorno):
    """La función se prueba arriba; aquí, que el FLUJO de R-CPU la usa (sabotaje del supervisor, 03-10: quitar la
    llamada dejó todo verde)."""
    datos = _datos()
    sin_redes = ("lightgbm", "sklearn.hgb", "baseline")
    regs = {n: _rec(n, motores=sin_redes) for n in HUMO}
    regs["dresses-sales"]["perfiles_declarados"] = _ARBOL_COMPLETADO
    regs["dresses-sales"]["politica_de_arboles"] = None
    correr, _ = _correr_desde(datos, regs)
    with pytest.raises(SystemExit):
        G.correr_c3p("rcpu", _banco(), datos, correr, None)
    assert "sobre de la política de C6" in datos["corte"]["motivo"]
