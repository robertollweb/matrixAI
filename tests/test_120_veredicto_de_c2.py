"""120 · C2 contra R1: el VEREDICTO y cuándo se CORTA la pasada (enmienda 3), y la comprobación de que de verdad se
mide la política pedida. Sobre los registros REALES de R1 (`benchmarks/fase0/referencia_120_r1.json`): un C2 se
fabrica moviendo una cifra de R1, y R1 contra sí mismo no puede subir ni bajar nada.

Los dos módulos se cargan por RUTA (sin tocar `sys.path`), y la última prueba lo vigila.
"""
from __future__ import annotations

import copy
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
        sys.dont_write_bytecode = antes      # el guion lo pone a True para sí; no para el resto de la suite
    return mod


V = _cargar("veredicto_120_en_pruebas", "veredicto_120.py")
G = _cargar("referencia_120_r0_en_pruebas", "referencia_120_r0.py")
R1 = json.loads((_FASE0 / "referencia_120_r1.json").read_text())["conjuntos"]


def _c2(nombre: str, *, valor=None, delta=None, motor=None, linea_base=None, estado=None, tamano="pequeno"):
    """Un registro de C2 hecho desde el de R1: mueve la cifra del campeón (`valor` o `delta` en la métrica cruda)."""
    rec = copy.deepcopy(R1[nombre])
    m = rec.get("media_de_la_seleccion") or {}
    campeon = motor or rec.get("campeon")
    for x in m.get("motores", []):
        if x["motor"] == campeon and valor is not None:
            x["valor"] = valor
        if x["motor"] == campeon and delta is not None:
            x["valor"] = x["valor"] + delta
        if x["motor"] == "baseline" and linea_base is not None:
            x["valor"] = linea_base
    if estado is not None:
        rec["estado"] = estado
    rec["perfiles_declarados"] = [{"motor": "lightgbm", "candidato": "lightgbm-p0-r0", "estado": "completed",
                                   "declara": True, "tamano": tamano, "l2": 0.0}]
    rec["politica_de_arboles"] = _sobre()
    return rec


def _sobre(valor="l2_0", l2=0.0, densa_fuera=True, fijados=True):
    """`seleccion.politica_de_arboles` como lo escribe el Studio de C2 (leído de lo entrenado)."""
    del_campeon = {"l2": l2, "tamano": "pequeno", "paro": "fijado"}
    if fijados:
        del_campeon["arboles_fijados"] = {"arboles": 120, "de_donde": "mediana_de_arboles_efectivos_de_la_busqueda"}
    return {"variable": "MATRIXAI_POLITICA_DE_ARBOLES", "valor": valor, "l2": l2, "densa_fuera": densa_fuera,
            "campeon": {"motor": "lightgbm", "politica_por_tamano": del_campeon}}


# ── la cifra ─────────────────────────────────────────────────────────────────

def test_R1_contra_si_mismo_no_sube_ni_baja_nada():
    clases = {n: V.comparar(r, dict(r, perfiles_declarados=[]))["clase"] for n, r in R1.items()}
    assert set(clases.values()) <= {"igual", "ninguno_completa"}, clases
    # Los que R1 no termina con campeón: KDD (failed), APS (el servidor murió), Allstate (sin campeón).
    assert {n for n, c in clases.items() if c == "ninguno_completa"} == {
        "KDDCup09_appetency", "APSFailure", "Allstate_Claims_Severity"}


def test_la_cifra_de_regresion_es_R2_de_la_validacion_frente_a_la_linea_base_nunca_el_test():
    r = R1["us_crime"]
    rmse = {x["motor"]: x["valor"] for x in r["media_de_la_seleccion"]["motores"]}
    assert r["campeon"] == "lightgbm"
    assert V.puntos(r) == pytest.approx(100 * (1 - (rmse["lightgbm"] / rmse["baseline"]) ** 2))
    assert 60 < V.puntos(r) < 66                                   # ≈ 63,6: una R² de verdad, no un RMSE × 100
    # Y el test no entra: con otra `evaluacion_final` la cifra es la misma.
    otro = copy.deepcopy(r); otro["seleccion"] = {"evaluacion_final": {"metrics": [{"metric_id": "rmse", "value": 9}]}}
    assert V.puntos(otro) == V.puntos(r)


def test_en_clasificacion_la_cifra_es_la_media_del_campeon_por_cien():
    r = R1["pc1"]
    assert r["campeon"] == "lightgbm"
    assert V.puntos(r) == pytest.approx(100 * 0.8841100532666799)


def test_un_RMSE_que_baja_es_una_R2_que_sube():
    mejor = V.comparar(R1["house_16H"], _c2("house_16H", delta=-2000.0, tamano="mediano"))
    peor = V.comparar(R1["house_16H"], _c2("house_16H", delta=+2000.0, tamano="mediano"))
    assert mejor["diferencia"] > 0 > peor["diferencia"]


# ── las clases y el corte ────────────────────────────────────────────────────

def test_menos_de_un_punto_no_cuenta():
    assert V.comparar(R1["pc1"], _c2("pc1", delta=+0.0099))["clase"] == "igual"
    assert V.comparar(R1["pc1"], _c2("pc1", delta=-0.0099))["clase"] == "igual"
    assert V.comparar(R1["pc1"], _c2("pc1", delta=+0.0101))["clase"] == "sube"
    assert V.comparar(R1["pc1"], _c2("pc1", delta=-0.0101))["clase"] == "baja"


def test_una_bajada_de_2_puntos_corta_la_pasada_y_una_de_1_9_no():
    c = V.comparar(R1["pc1"], _c2("pc1", delta=-0.0201))
    assert c["clase"] == "baja_2" and "pc1 baja" in V.corta(c)
    c = V.comparar(R1["pc1"], _c2("pc1", delta=-0.019))
    assert c["clase"] == "baja" and V.corta(c) is None


def test_dejar_de_completar_corta_y_empezar_a_completar_cuenta_como_sube():
    c = V.comparar(R1["wilt"], _c2("wilt", estado="failed"))
    assert c["clase"] == "deja_de_completar" and "deja de completar" in V.corta(c)
    # KDD no completa en R1: si con C2 termina con campeón, es una mejora.
    kdd = _c2("pc1"); kdd["nombre"] = "KDDCup09_appetency"
    c = V.comparar(R1["KDDCup09_appetency"], kdd)
    assert c["clase"] == "empieza_a_completar" and V.corta(c) is None


def test_otra_linea_base_es_incomparable_y_para():
    with pytest.raises(V.Incomparable, match="línea base"):
        V.comparar(R1["pc1"], _c2("pc1", linea_base=0.51))
    with pytest.raises(V.Incomparable, match="línea base"):
        V.comparar(R1["us_crime"], _c2("us_crime", linea_base=0.2400))


def test_el_veredicto_es_por_tamano_y_un_tamano_sin_nada_se_queda_el_de_hoy():
    comps = [V.comparar(R1["pc1"], _c2("pc1", delta=+0.02, tamano="pequeno")),
             V.comparar(R1["dresses-sales"], _c2("dresses-sales", delta=+0.02, tamano="pequeno")),
             V.comparar(R1["climate-model-simulation-crashes"],
                        _c2("climate-model-simulation-crashes", delta=-0.012, tamano="pequeno")),
             V.comparar(R1["wilt"], _c2("wilt", delta=-0.012, tamano="mediano"))]
    v = V.veredicto(comps)["por_tamano"]
    assert v["pequeno"]["mejora"] is True and (v["pequeno"]["suben"], v["pequeno"]["bajan"]) == (2, 1)
    assert v["mediano"]["mejora"] is False and v["mediano"]["se_queda"] == "la configuración de hoy"
    assert v["grande"]["mejora"] is False and v["grande"]["conjuntos"] == []


def test_una_bajada_de_2_en_un_tamano_es_un_no_aunque_suban_mas():
    comps = [V.comparar(R1["pc1"], _c2("pc1", delta=+0.02)),
             V.comparar(R1["dresses-sales"], _c2("dresses-sales", delta=+0.02)),
             V.comparar(R1["us_crime"], _c2("us_crime", delta=+0.01))]   # RMSE +0,01 → R² baja > 2 puntos
    assert comps[-1]["clase"] == "baja_2"
    assert V.veredicto(comps)["por_tamano"]["pequeno"]["mejora"] is False


# ── la política que se mide ──────────────────────────────────────────────────

def _estado(*intentos):
    return {"estado": "completed", "leaderboard": [
        {"engine": m, "candidate": f"{m}-p{i}-r0", "estado": e, "wall_seconds": 1.0,
         "config_efectiva": {"hiperparametros": ({"politica_por_tamano": d} if d is not None else {})}}
        for i, (m, e, d) in enumerate(intentos)]}


def _rec(est, nombre="pc1", sobre=None, campeon="lightgbm"):
    return {"nombre": nombre, "estado": est["estado"], "campeon": campeon,
            "perfiles_declarados": G.perfiles_declarados(est), "politica_de_arboles": sobre}


def test_cada_intento_de_arboles_completado_tiene_que_declarar_la_politica_pedida():
    d0 = {"tamano": "pequeno", "l2": 0.0, "filas_de_train": 640}
    bien = _rec(_estado(("lightgbm", "completed", d0), ("sklearn.hgb", "completed", d0), ("baseline", "completed", None)),
                sobre=_sobre())
    assert G.comprobar_politica(bien, "l2_0") == (None, 2)
    motivo, _ = G.comprobar_politica(bien, "l2_1")
    assert motivo and "sin la política l2_1" in motivo
    # Un intento que no declara nada (el contenedor sin la variable) PARA.
    sin = _rec(_estado(("lightgbm", "completed", d0), ("sklearn.hgb", "completed", None)))
    motivo, _ = G.comprobar_politica(sin, "l2_0")
    assert motivo and "sklearn.hgb-p1-r0" in motivo


def test_sin_politica_ningun_intento_declara_ninguna():
    d0 = {"tamano": "pequeno", "l2": 0.0}
    assert G.comprobar_politica(_rec(_estado(("lightgbm", "completed", None))), None) == (None, 1)
    motivo, _ = G.comprobar_politica(_rec(_estado(("lightgbm", "completed", d0))), None)
    assert motivo and "SIN pedirla" in motivo


def test_un_conjunto_sin_arboles_completados_no_para_y_dice_cero():
    """Allstate en R1: ningún intento de árboles terminó. No hay nada que comprobar y no es un fallo del instrumento."""
    est = _estado(("lightgbm", "failed", None), ("sklearn.hgb", "failed", None), ("baseline", "completed", None))
    assert G.comprobar_politica(_rec(est, "Allstate_Claims_Severity"), "l2_0") == (None, 0)


def test_el_tamano_es_el_que_declara_el_estudio():
    d = lambda t: {"tamano": t, "l2": 0.0}
    rec = _rec(_estado(("lightgbm", "completed", d("mediano")), ("sklearn.hgb", "completed", d("mediano")),
                       ("lightgbm", "completed", d("pequeno"))))
    assert V.tamano_declarado(rec) == "mediano"


def test_importar_el_guion_no_toca_sys_path():
    antes = list(sys.path)
    _cargar("referencia_120_r0_otra_vez", "referencia_120_r0.py")
    assert sys.path == antes


def test_empezar_a_completar_es_una_mejora_en_el_veredicto_de_su_tamano():
    kdd = _c2("pc1", tamano="grande"); kdd["nombre"] = "KDDCup09_appetency"
    v = V.veredicto([V.comparar(R1["KDDCup09_appetency"], kdd)])["por_tamano"]["grande"]
    assert (v["suben"], v["bajan"], v["mejora"]) == (1, 0, True)


# ── el bucle entero, con el contenedor de mentira ────────────────────────────

def test_la_pasada_controla_sin_politica_rearranca_con_ella_y_corta_en_la_primera_bajada_de_2(monkeypatch, tmp_path):
    banco = sorted(json.loads((_FASE0 / "protocolo_120.json").read_text())["conjuntos"]["banco"],
                   key=lambda x: x["orden"])
    nombres = [b["nombre"] for b in banco]
    assert nombres[:4] == ["dresses-sales", "climate-model-simulation-crashes", "us_crime", "pc1"]
    arranques, medidos = [], []
    monkeypatch.setattr(G, "ESTADOS", tmp_path)
    monkeypatch.setattr(G, "POLITICA", "l2_0")
    monkeypatch.setattr(G, "guardar", lambda d: None)
    monkeypatch.setattr(G, "borrar_contenedor", lambda: None)
    monkeypatch.setattr(G, "dentro", lambda *a: "MATRIXAI_POLITICA_DE_ARBOLES l2_0")
    monkeypatch.setattr(G, "arrancar_contenedor", lambda politica=None: (arranques.append(politica), ("b", "c"))[1])
    datos = {"procedencia": {}, "conjuntos": {}, "comparaciones": {}, "corte": None, "control": None}

    def correr(b, destino=None, reusar=True):
        n = b["nombre"]
        if destino is not None:                     # el control: R1 tal cual, sin ninguna política declarada
            assert reusar is False
            destino[n] = dict(R1[n], nombre=n, perfiles_declarados=[
                {"motor": "lightgbm", "candidato": "x", "estado": "completed", "declara": False}])
            return
        medidos.append(n)
        datos["conjuntos"][n] = _c2(n, delta=(-0.03 if n == "pc1" else 0.0))   # pc1 baja 3 puntos

    G.correr_con_politica(banco, banco, datos, {"politica": None}, correr, R1)
    assert arranques == ["l2_0"]                     # el primero (sin política) lo arranca main; aquí, CON ella
    assert datos["control"]["cuadra"] is True and datos["control"]["sin_politica"] is True
    assert medidos == nombres[:4]                    # tras pc1 no se mide nada más
    assert datos["corte"]["tipo"] == "regla_9" and datos["corte"]["tras"] == "pc1"
    assert datos["corte"]["sin_medir"] == nombres[4:]
    assert datos["veredicto"]["por_tamano"]["pequeno"]["mejora"] is False


def test_un_control_sin_politica_que_no_cuadra_para_antes_de_medir_nada(monkeypatch, tmp_path):
    banco = sorted(json.loads((_FASE0 / "protocolo_120.json").read_text())["conjuntos"]["banco"],
                   key=lambda x: x["orden"])
    monkeypatch.setattr(G, "ESTADOS", tmp_path)
    monkeypatch.setattr(G, "POLITICA", "l2_0")
    monkeypatch.setattr(G, "guardar", lambda d: None)
    monkeypatch.setattr(G, "arrancar_contenedor", lambda politica=None: pytest.fail("no debía rearrancar"))
    datos = {"procedencia": {}, "conjuntos": {}, "comparaciones": {}, "corte": None, "control": None}

    def correr(b, destino=None, reusar=True):
        n = b["nombre"]
        assert destino is not None, "midió con la política sin pasar el control"
        rec = _c2(n, delta=0.0001)                    # una cifra que no es la de R1 a 4 decimales... en uno
        destino[n] = rec if n == "pc1" else dict(R1[n], nombre=n, perfiles_declarados=[])
        destino[n]["perfiles_declarados"] = [{"motor": "lightgbm", "candidato": "x", "estado": "completed",
                                              "declara": False}]

    with pytest.raises(SystemExit) as e:
        G.correr_con_politica(banco, banco, datos, {"politica": None}, correr, R1)
    assert e.value.code == 2 and datos["control"]["cuadra"] is False



@pytest.mark.parametrize("declarados", [
    [{"motor": "lightgbm", "candidato": "x", "estado": "completed", "declara": True, "l2": 0.0}],   # sin pedirla
    [{"motor": "lightgbm", "candidato": "x", "estado": "failed", "declara": False}],               # nada comprobado
])
def test_el_control_sin_politica_para_si_alguien_la_declara_o_si_no_hay_arboles_que_mirar(monkeypatch, tmp_path,
                                                                                            declarados):
    banco = sorted(json.loads((_FASE0 / "protocolo_120.json").read_text())["conjuntos"]["banco"],
                   key=lambda x: x["orden"])
    monkeypatch.setattr(G, "ESTADOS", tmp_path)
    monkeypatch.setattr(G, "POLITICA", "l2_0")
    monkeypatch.setattr(G, "guardar", lambda d: None)
    monkeypatch.setattr(G, "arrancar_contenedor", lambda politica=None: pytest.fail("no debía rearrancar"))
    datos = {"procedencia": {}, "conjuntos": {}, "comparaciones": {}, "corte": None, "control": None}

    def correr(b, destino=None, reusar=True):
        n = b["nombre"]
        destino[n] = dict(R1[n], nombre=n, perfiles_declarados=declarados if n == "us_crime" else [
            {"motor": "lightgbm", "candidato": "x", "estado": "completed", "declara": False}])

    with pytest.raises(SystemExit) as e:
        G.correr_con_politica(banco, banco, datos, {"politica": None}, correr, R1)
    assert e.value.code == 2 and datos["control"]["cuadra"] is False
    assert any("us_crime" in l and ("SIN pedirla" in l or " 0 intentos" in l) for l in datos["control"]["lineas"])


# ── el sobre de selección (lo que se entrenó, no lo que se pidió) ────────────

_D0 = {"tamano": "pequeno", "l2": 0.0}


def _con_sobre(sobre, campeon="lightgbm"):
    return _rec(_estado(("lightgbm", "completed", _D0), ("sklearn.hgb", "completed", _D0)), sobre=sobre,
                campeon=campeon)


def test_con_la_politica_el_sobre_tiene_que_decir_el_valor_pedido():
    assert G.comprobar_politica(_con_sobre(_sobre()), "l2_0") == (None, 2)
    for sobre in (None, _sobre(valor="l2_1")):
        motivo, _ = G.comprobar_politica(_con_sobre(sobre), "l2_0")
        assert motivo and "el sobre no declara l2_0" in motivo


def test_con_la_politica_la_densa_tiene_que_estar_fuera_enmienda_2():
    """Con torch la densa compite y cada árbol tiene 1,87 s: no es la condición que se mide."""
    for fuera in (False, None):
        motivo, _ = G.comprobar_politica(_con_sobre(_sobre(densa_fuera=fuera)), "l2_0")
        assert motivo and "la densa NO estaba fuera" in motivo


def test_si_gana_un_arbol_su_modelo_es_el_reajuste_con_los_arboles_fijados():
    motivo, _ = G.comprobar_politica(_con_sobre(_sobre(fijados=False)), "l2_0")
    assert motivo and "árboles fijados" in motivo
    motivo, _ = G.comprobar_politica(_con_sobre(_sobre(l2=1.0, valor="l2_0")), "l2_0")
    assert motivo and "árboles fijados" in motivo
    # Si gana la línea base, no hay reajuste de árbol que mirar.
    assert G.comprobar_politica(_con_sobre(_sobre(fijados=False), campeon="baseline"), "l2_0") == (None, 2)


def test_sin_la_politica_el_sobre_no_puede_llevarla():
    sin = _rec(_estado(("lightgbm", "completed", None)), sobre=_sobre())
    motivo, _ = G.comprobar_politica(sin, None)
    assert motivo and "el sobre declara una política SIN pedirla" in motivo
    assert G.comprobar_politica(_rec(_estado(("lightgbm", "completed", None))), None) == (None, 1)


def test_registrar_guarda_el_sobre_de_la_seleccion():
    est = _estado(("lightgbm", "completed", _D0))
    est["seleccion"] = {"candidate_engine": "lightgbm", "politica_de_arboles": _sobre()}
    rec = G.registrar("pc1", {"data_id": 1, "tarea": "binary_classification", "objetivo": "y"}, {}, est,
                      1.0, 200, None, ["0", "0", "0"], "x")
    assert rec["politica_de_arboles"] == _sobre() and rec["campeon"] == "lightgbm"


@pytest.mark.parametrize("salida", ["la_de_R0", "la_de_contra"])
def test_con_la_politica_no_se_escribe_encima_de_R0_ni_de_la_referencia(monkeypatch, tmp_path, salida):
    """Todo en `tmp_path`, y lo que tiene efectos fuera (el contenedor, el generador de CSV, escribir el JSON)
    sustituido por un fallo RUIDOSO: si la guarda se rompe, la prueba se pone roja al instante en vez de arrancar
    un contenedor y escribir encima del artefacto (02-10: un sabotaje de esta guarda, con la prueba vieja, arrancó
    un contenedor de verdad y reescribió referencia_120_r0.json; restaurado del commit)."""
    contra = tmp_path / "r1.json"
    contra.write_text((_FASE0 / "referencia_120_r1.json").read_text())
    r0 = tmp_path / "referencia_120_r0.json"
    argv = ["referencia_120_r0.py", "--politica", "l2_0", "--contra", str(contra)]
    if salida == "la_de_contra":
        argv += ["--salida", str(contra)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(G, "SALIDA", r0)                 # la omisión de --salida, aquí
    for peligrosa in ("arrancar_contenedor", "cargar_generador", "comprobar_bytes", "guardar", "borrar_contenedor"):
        monkeypatch.setattr(G, peligrosa, lambda *a, _n=peligrosa, **k: pytest.fail(f"la guarda dejó pasar: {_n}"))
    with pytest.raises(SystemExit, match="--salida PROPIA"):
        G.main()
