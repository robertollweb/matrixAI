"""129-C0: los mapas de solo lectura. Lo que se prueba SIN los datos grandes: clasificar por qué paró TabM, contar columnas
binarias/constantes con datos sintéticos, y el chequeo de la cola 85/15 con particiones sintéticas (la real del núcleo y el
reparto real del estudio). Cada pieza tiene su prueba CON DIENTES (se sabotea la pieza y se pone en rojo)."""
import importlib.util
import json
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "mapas_129_c0.py"
spec = importlib.util.spec_from_file_location("mapas_129_c0_bajo_prueba", RUTA)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


# ------------------------------------------------------------------------------------------------ por qué para TabM
def tabm(candidato, **ent):
    base = {"epocas_ejecutadas": 11, "epocas_completas": 11, "mejor_epoca": 11, "parado_por_plazo": False,
            "parado_por_paciencia": False, "ultima_epoca_parcial": False,
            "plazo_antes_de_la_primera_epoca_completa": False, "plazo_de_entrenamiento_segundos": 45.0}
    base.update(ent)
    return {"candidate": candidato, "estado": "completed", "wall_seconds": 44.0,
            "config_efectiva": {"entrenamiento_efectivo": base}}


def test_el_plazo_y_la_paciencia_se_distinguen_y_la_mejor_epoca_ultima_se_ve():
    a = M.por_que_de_un_intento(tabm("x", parado_por_plazo=True))
    assert (a["por_que"], a["epocas"], a["mejor_epoca"], a["mejor_es_la_ultima"]) == ("plazo", 11, 11, True)
    b = M.por_que_de_un_intento(tabm("x", parado_por_paciencia=True, epocas_ejecutadas=34, mejor_epoca=17))
    assert (b["por_que"], b["epocas"], b["mejor_epoca"], b["mejor_es_la_ultima"]) == ("paciencia", 34, 17, False)


def test_sin_ninguna_de_las_dos_es_maximo_de_epocas_y_con_las_dos_se_dice():
    assert M.por_que_de_un_intento(tabm("x"))["por_que"] == "maximo_de_epocas"
    assert M.por_que_de_un_intento(tabm("x", parado_por_plazo=True, parado_por_paciencia=True))["por_que"] == "plazo_y_paciencia"


def test_el_techo_externo_no_es_un_fallo_del_motor():
    fuera = {"candidate": "x", "estado": "failed",
             "motivo": {"es": "el intento superó 90.0 s sin terminar: el proceso se ha matado desde fuera, no es un fallo...",
                        "en": "the attempt exceeded 90.0 s ... killed from the outside"}}
    assert M.por_que_de_un_intento(fuera)["por_que"] == "techo_externo"
    propio = {"candidate": "x", "estado": "failed",
              "motivo": {"es": "el presupuesto se agotó antes de producir ningún candidato", "en": "the budget ran out"}}
    r = M.por_que_de_un_intento(propio)
    assert r["por_que"] == "sin_candidato" and r["estado"] == "failed"


def test_un_intento_que_no_existe_es_sin_candidato():
    assert M.por_que_de_un_intento(None)["por_que"] == "sin_candidato"


def test_solo_cuentan_los_intentos_de_tabm_de_la_repeticion_0():
    est = {"leaderboard": [tabm("lightgbm-csv-p0-r0"), tabm("matrixai.dense.tabm_cpu-csv-p2-r0"),
                           tabm("matrixai.dense.tabm_cpu-csv-p3-r1"), tabm("sklearn.hgb-csv-p1-r0")]}
    assert sorted(M.intentos_de_tabm(est)) == [2]


def escribir_estudio(carpeta, nombre, estudio):
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"{nombre}.json").write_text(json.dumps(estudio))


def test_ausencias_de_tabm_sin_confundirlas(tmp_path):
    arboles = [tabm("lightgbm-csv-p0-r0")]
    escribir_estudio(tmp_path, "sinres", {"estado": "sondeo_fallido"})
    escribir_estudio(tmp_path, "memoria", {"leaderboard": arboles, "motores_fuera_por_memoria": {
        "matrixai.dense.tabm_cpu": {"es": "TabM no compitió en este estudio: la memoria..."}}})
    escribir_estudio(tmp_path, "arboles", {"leaderboard": arboles, "motores_fuera_por_memoria": {}})
    escribir_estudio(tmp_path, "bueno", {"leaderboard": [tabm(f"matrixai.dense.tabm_cpu-csv-p{i}-r0", parado_por_plazo=True)
                                                         for i in range(5)]})
    t = M.tabla_de_una_pasada(tmp_path)
    assert {c: t[c][0].get("ausencia") for c in ("sinres", "memoria", "arboles")} == {
        "sinres": "estudio_sin_resultado", "memoria": "fuera_por_memoria", "arboles": "tabm_no_competia"}
    assert t["memoria"][0]["detalle"].startswith("TabM no compitió")
    assert all(t["bueno"][p]["por_que"] == "plazo" for p in range(5))


def test_una_pasada_sin_tabm_no_diluye_el_resumen():
    con = {p: M.por_que_de_un_intento(tabm("x", parado_por_paciencia=True, epocas_ejecutadas=20 + p, mejor_epoca=5))
           for p in range(5)}
    sin = {p: {"por_que": "sin_candidato", "ausencia": "tabm_no_competia"} for p in range(5)}
    r = M.resumir_pasadas({"a": con, "control_sin_tabm": sin})
    assert r["intentos"] == 5 and r["por_causa"] == {"paciencia": 5} and r["epocas"] == [20, 24]
    assert r["pasadas_sin_tabm_no_contadas"] == ["control_sin_tabm"]
    assert r["mejor_es_la_ultima"] == "0 de 5"


def test_el_resumen_ve_cuando_la_causa_cambia_entre_pasadas_del_mismo_pliegue():
    a = {0: M.por_que_de_un_intento(tabm("x", parado_por_plazo=True))}
    b = {0: M.por_que_de_un_intento(tabm("x", parado_por_paciencia=True))}
    assert M.resumir_pasadas({"a": a, "b": b})["pliegues_con_causa_distinta_entre_pasadas"] == [0]


# -------------------------------------------------------------------------------------------------- columnas de TabM
def test_constantes_binarias_e_indicadores_con_una_matriz_sintetica():
    np = pytest.importorskip("numpy")
    n = 40
    rng = np.random.default_rng(0)
    cols = {
        "continua": rng.normal(size=n),
        "const": np.full(n, 3.0),
        "binaria_01": (np.arange(n) % 2).astype(float),
        "binaria_otra": np.where(np.arange(n) % 3 == 0, 5.0, -2.0),
        "tres": (np.arange(n) % 3).astype(float),
        "x__faltante": np.zeros(n),                     # indicador constante (nunca falta)
        "y__faltante": (np.arange(n) % 5 == 0).astype(float),   # indicador binario
    }
    nombres = list(cols)
    r = M.clasificar_columnas(nombres, np.column_stack([cols[k] for k in nombres]))
    assert r["numericas"] == 7
    assert (r["constantes"], r["binarias"], r["binarias_01"]) == (2, 3, 2)
    assert r["faltante"] == 2 and (r["faltante_constantes"], r["faltante_binarias"]) == (1, 1)
    assert (r["no_faltante_constantes"], r["no_faltante_binarias"]) == (1, 2)
    assert r["afectadas_por_r6"] == 5 and r["proporcion_afectadas"] == round(5 / 7, 4)
    assert sorted(r["_afectadas"]) == ["binaria_01", "binaria_otra", "const", "x__faltante", "y__faltante"]


def test_clasificar_columnas_rechaza_una_matriz_que_no_cuadra():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        M.clasificar_columnas(["a", "b"], np.zeros((4, 3)))


def test_columnas_de_tabm_usa_la_preparacion_real_y_cuenta_sobre_train():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pt = pytest.importorskip("matrixai_engines.redes.preparacion_tabm")
    # 60 filas: `a` continua, `c` constante, `b` binaria 0/1, `a__faltante` el indicador (1 en 6 filas), `cat` categórica
    feats = [{"a": float(i) * 0.37, "c": 1.0, "b": float(i % 2), "a__faltante": 1.0 if i % 10 == 0 else 0.0,
              "cat": "u" if i % 2 else "v"} for i in range(60)]
    r = M.columnas_de_tabm(feats, ["a", "c", "b", "a__faltante", "cat"], d_embedding=24, semilla=0)
    assert r["predictores"] == 5 and r["filas_train"] == 60
    assert r["categoricas"] == len(pt.ajustar_preparacion(feats, ["a", "c", "b", "a__faltante", "cat"]).columnas_categoricas)
    assert r["numericas"] + r["categoricas"] == r["columnas_de_entrada"]
    assert r["constantes"] >= 1 and r["binarias"] >= 2 and r["faltante"] == 1
    assert r["afectadas_por_r6"] == r["constantes"] + r["binarias"]


def test_el_control_contra_el_estado_compara_nombres_y_dice_cuando_no_puede():
    ce = {"columnas_numericas": ["a", "b"], "columnas_categoricas": ["c"]}
    ok = M.comparar_con_el_estado(["b", "a"], ["c"], ce)
    assert ok["comparable"] and ok["numericas_coinciden"] and ok["categoricas_coinciden"]
    mal = M.comparar_con_el_estado(["a", "x"], ["c"], ce)
    assert mal["numericas_coinciden"] is False and mal["categoricas_coinciden"] is True
    omitido = M.comparar_con_el_estado(["a"], [], {"columnas_numericas": {"omitido": "tamano"}, "columnas_categoricas": []})
    assert omitido["comparable"] is False
    assert M.comparar_con_el_estado(["a"], [], None)["comparable"] is False


# --------------------------------------------------------------------------------------------- la cola de la parada temprana
def test_una_clase_ausente_con_esperado_ge_1_hace_la_cola_no_representativa():
    entrena = ["a"] * 80 + ["b"] * 17 + ["c"] * 3
    cola_buena = ["a"] * 12 + ["b"] * 3
    cola_mala = ["a"] * 12 + ["c"] * 3                       # le falta `b` (esperado 2,5 ≥ 1)
    bien = M.diagnostico_de_cola_clasificacion(entrena, cola_buena)
    mal = M.diagnostico_de_cola_clasificacion(entrena, cola_mala)
    assert bien["representativa"] is True and bien["ausentes_graves"] == []
    assert mal["representativa"] is False
    assert [a["clase"] for a in mal["ausentes_graves"]] == ["b"] and mal["ausentes_graves"][0]["esperado"] == 2.55


def test_una_clase_ausente_con_esperado_menor_que_1_se_dice_pero_no_es_grave():
    entrena = ["a"] * 95 + ["rara"] * 5                       # esperado de `rara` en una cola de 15: 0,75
    r = M.diagnostico_de_cola_clasificacion(entrena, ["a"] * 15)
    assert [x["clase"] for x in r["ausentes"]] == ["rara"] and r["ausentes_graves"] == []


def test_el_z_por_clase_ve_una_cola_desviada_aunque_no_falte_ninguna_clase():
    entrena = ["a"] * 500 + ["b"] * 500
    sesgada = ["a"] * 90 + ["b"] * 10                         # esperado 50/50 de 100: z enorme
    r = M.diagnostico_de_cola_clasificacion(entrena, sesgada)
    assert r["ausentes"] == [] and abs(r["z_max"]) > M.UMBRAL_Z and r["representativa"] is False
    ok = M.diagnostico_de_cola_clasificacion(entrena, ["a"] * 52 + ["b"] * 48)
    assert ok["representativa"] is True


def test_una_clase_nueva_en_la_cola_se_declara():
    r = M.diagnostico_de_cola_clasificacion(["a"] * 10, ["a"] * 3 + ["z"])
    assert r["clases_nuevas_en_la_cola"] == ["z"] and r["representativa"] is False


def test_regresion_media_y_dispersion_de_la_cola():
    y = [float(i) for i in range(1000)]
    ok = M.diagnostico_de_cola_regresion(y, y[::7][:150])     # una muestra repartida por todo el rango
    assert ok["representativa"] is True and abs(ok["z_media"]) < 1
    sesgada = M.diagnostico_de_cola_regresion(y, y[-150:])    # la cola son los valores más altos
    assert sesgada["representativa"] is False and sesgada["z_media"] > M.UMBRAL_Z
    uniforme = [float(i % 100) for i in range(1000)]            # media 49,5 y dispersión ~28,9
    corrida = M.diagnostico_de_cola_regresion(uniforme, [float(i % 100) + 8.0 for i in range(200)])   # igual de ancha, +8
    assert corrida["razon_std"] == pytest.approx(1.0, abs=0.01) and corrida["z_media"] > M.UMBRAL_Z
    assert corrida["representativa"] is False             # solo la media la delata: el z tiene que contar por sí solo
    estrecha = M.diagnostico_de_cola_regresion(y, [500.0 + (i % 5) for i in range(150)])   # media bien, dispersión ~0
    assert estrecha["representativa"] is False and estrecha["razon_std"] < M.RAZON_STD_MIN


def test_unidades_partidas_entre_ajuste_y_cola():
    r = M.unidades_partidas(["u1", "u1", "u2", "u3"], ["u3", "u4"])
    assert r["parte_una_unidad"] is True and r["unidades_partidas"] == 1 and r["ejemplos"] == ["u3"]
    limpio = M.unidades_partidas(["u1", "u2"], ["u3"])
    assert limpio["parte_una_unidad"] is False and limpio["unidades_partidas"] == 0


def test_solape_de_colas_agrupa_los_pliegues_con_la_misma_cola():
    base = [f"r{i}" for i in range(100)]
    colas = {0: base, 1: base[:-1] + ["otra"], 2: base, 3: base, 4: [f"s{i}" for i in range(100)]}
    s = M.solape_de_colas(colas)
    assert s["grupos_con_la_misma_cola"] == [[0, 1, 2, 3], [4]] and s["pliegues_con_cola_propia"] == 1
    assert s["solape_por_par"]["0-4"] == 0.0 and s["solape_por_par"]["0-1"] == 0.99


def colas(**por_conjunto):
    return {c: {"pliegues": {str(i): d for i, d in enumerate(ds)}} for c, ds in por_conjunto.items()}


def info(representativa=True, graves=(), unidades=None, tipo="clasificacion"):
    d = {"tipo": tipo, "representativa": representativa, "ausentes_graves": list(graves), "z_max": 4.2,
         "clase_del_z_max": "k", "z_media": 3.5, "razon_std": 1.0}
    i = {"diagnostico": d}
    if unidades is not None:
        i["unidades"] = unidades
    return i


def test_la_decision_de_h20_sin_defectos_y_con_ellos():
    limpio = M.decidir_sobre_h20(colas(a=[info(), info()], b=[info(tipo="regresion")]))
    assert limpio["la_cola_es_representativa_en_todos"] is True and limpio["pliegues_mirados"] == 3
    assert "sin efecto medible" in limpio["veredicto"]
    sucio = M.decidir_sobre_h20(colas(a=[info(), info(False, graves=[{"clase": "x"}])], b=[info(False)],
                                      c=[info(unidades={"parte_una_unidad": True, "unidades_partidas": 2})]))
    assert sucio["la_cola_es_representativa_en_todos"] is False
    assert sorted(sucio["conjuntos_con_defecto"]) == ["a", "b", "c"]
    assert list(sucio["conjuntos_con_defecto"]["a"]) == ["1"]
    assert "ausente" in sucio["conjuntos_con_defecto"]["a"]["1"][0]          # la ausencia grave se NOMBRA, no solo el z
    assert "z máximo" in sucio["conjuntos_con_defecto"]["b"]["0"][0]
    assert "parte 2 unidad" in sucio["conjuntos_con_defecto"]["c"]["0"][0]
    assert "NO representativas" in sucio["veredicto"]


def test_r6_umbral_del_10_por_ciento():
    r = M.conjuntos_sin_nada_que_ganar_en_r6({"poco": {"proporcion_afectadas_de_la_entrada": 0.09},
                                              "justo": {"proporcion_afectadas_de_la_entrada": 0.10},
                                              "micro-mass": {"proporcion_afectadas_de_la_entrada": 0.6},
                                              "sin": {"proporcion_afectadas_de_la_entrada": None}})
    assert r["por_debajo_del_umbral"] == ["poco"] and r["por_encima_del_umbral"] == ["justo", "micro-mass"]
    assert r["anchos"] == ["micro-mass"]


# ------------------------------------------------- con la PARTICIÓN REAL del núcleo y el REPARTO REAL del estudio
def particion_sintetica(n=300, raras=30, folds=5):
    """`filas` iid con una clase RARA de `raras` casos (≥ folds: se puede estratificar) y la mayoritaria."""
    pp = pytest.importorskip("matrixai.training.particion_por_diseno")
    filas = [{"row_id": f"r{i}", "y": ("rara" if i < raras else ("b" if i % 4 == 0 else "a"))} for i in range(n)]
    p = pp.proponer_particion(filas, plan_id="sintetico", observation_id_field="row_id", split_type="iid", seed=0,
                              test_fraction=0.2, folds=folds, repeats=1, unit_id_field=None, time_column=None, gap=None,
                              objetivo="y")
    assert p.es_viable
    return filas, p


def reparto_real():
    ej = pytest.importorskip("matrixai_studio.estudio_job")
    return ej._reparto_de_entrena


def test_con_la_particion_real_los_pliegues_comparten_la_cola_posicional():
    """Estructural, no de semilla: `entrena` es la concatenación de los cubos de los OTROS pliegues en orden, así que en cuatro
    de cinco pliegues el último cubo es el mismo y la cola 85/15 (los últimos) son las MISMAS filas."""
    reparto = reparto_real()
    filas, p = particion_sintetica()
    assert p.pliegues.folds == 5          # sin esto el reparto de abajo no es el de 5 pliegues
    colas_ids = {i: list(reparto(p.pliegues.pliegue_de(repeticion=0, pliegue=i))[1]) for i in range(5)}
    s = M.solape_de_colas(colas_ids)
    assert sorted(len(g) for g in s["grupos_con_la_misma_cola"]) == [1, 4]
    for i, cola in colas_ids.items():
        pl = p.pliegues.pliegue_de(repeticion=0, pliegue=i)
        ajuste, cola_real = reparto(pl)
        assert not set(ajuste) & set(cola_real) and len(ajuste) + len(cola_real) == len(pl.entrena)
        assert len(cola_real) == len(pl.entrena) - int(len(pl.entrena) * 0.85)


def test_la_cola_real_de_una_clase_con_un_caso_por_cubo_pierde_la_clase_en_algun_pliegue():
    pp = pytest.importorskip("matrixai.training.particion_por_diseno")
    reparto = reparto_real()
    perdidas = 0
    for semilla in range(30):
        filas = [{"row_id": f"r{i}", "y": ("rara" if i < 25 else "a")} for i in range(500)]
        p = pp.proponer_particion(filas, plan_id="s", observation_id_field="row_id", split_type="iid", seed=semilla,
                                  test_fraction=0.2, folds=5, repeats=1, unit_id_field=None, time_column=None, gap=None,
                                  objetivo="y")
        y = {f["row_id"]: f["y"] for f in filas}
        for i in range(5):
            pl = p.pliegues.pliegue_de(repeticion=0, pliegue=i)
            _, cola = reparto(pl)
            d = M.diagnostico_de_cola_clasificacion([y[x] for x in pl.entrena], [y[x] for x in cola])
            if any(a["clase"] == "rara" for a in d["ausentes"]):
                perdidas += 1
                # y el diagnóstico es coherente con el recuento directo
                assert not any(y[x] == "rara" for x in cola)
    assert perdidas > 0, "ninguna semilla produjo la pérdida: el chequeo de la cola no se ejerce"


# -------------------------------------------------------- el mapa 1 entero, la fuente del 85/15 y la procedencia
def estudio_tabm(**ent):
    return {"leaderboard": [tabm(f"matrixai.dense.tabm_cpu-csv-p{i}-r0", **ent) for i in range(5)]}


def test_el_mapa_entero_con_directorios_sinteticos(tmp_path):
    ref, c4 = tmp_path / "referencia_128_pcpu_v212_estados", tmp_path / "c4_128_pcpu_estados"
    escribir_estudio(ref, "cA", estudio_tabm(parado_por_plazo=True))
    escribir_estudio(c4, "cA", estudio_tabm(parado_por_paciencia=True, epocas_ejecutadas=30, mejor_epoca=12))
    escribir_estudio(ref, "cB", estudio_tabm(parado_por_paciencia=True))
    (tmp_path / "tabm_por_fases_128_r4.json").write_text(json.dumps({"conjuntos": {"cA:tabm_por_fases": {
        "csv": {"filas": 10}, "resumen_de_corridas": [{"rotulo": "plazo_amplio", "pared_s": 99.0, "epocas": {
            "n": 34, "entrenar_mediana_s": 3.4, "validar_mediana_s": 0.2}, "partes_del_ajuste": {"ajustar_s": 90.0, "preparar_s": 1.0}}],
        "medida": {"corridas": [{"rotulo": "plazo_amplio", "plazo_s": 1800.0, "estado": "completed", "filas": {"train": 5},
                                 "entrenamiento_efectivo": {"epocas_ejecutadas": 34, "mejor_epoca": 17,
                                                            "parado_por_paciencia": True, "parado_por_plazo": False}}]}}}}))
    m = M.mapa_por_que(tmp_path)
    assert set(m["pasadas_leidas"]) == {"referencia_v212", "integracion_c4"}
    a = m["conjuntos"]["cA"]
    assert a["resumen_de_la_referencia"]["por_causa"] == {"plazo": 5}
    assert a["resumen_de_todas_las_pasadas"]["por_causa"] == {"paciencia": 5, "plazo": 5}
    assert a["resumen_de_todas_las_pasadas"]["pliegues_con_causa_distinta_entre_pasadas"] == [0, 1, 2, 3, 4]
    assert a["referencia_v212"]["3"]["por_que"] == "plazo"
    assert m["conjuntos"]["cB"]["pasadas"].keys() == {"referencia_v212"}
    f = m["tabm_por_fases_128_r4"]["cA"]["corridas"][0]
    assert (f["por_que"], f["epocas"], f["mejor_epoca"], f["ajustar_s"]) == ("paciencia", 34, 17, 90.0)


def test_el_control_del_pliegue_0_lee_la_referencia(tmp_path):
    ref = tmp_path / "referencia_128_pcpu_v212_estados"
    e = tabm("matrixai.dense.tabm_cpu-csv-p0-r0", parado_por_plazo=True)
    e["config_efectiva"].update(columnas_numericas=["a", "b"], columnas_categoricas=[])
    escribir_estudio(ref, "c", {"leaderboard": [e]})
    ok = M.control_del_pliegue_0("c", {"_nombres_numericas": ["b", "a"], "_nombres_categoricas": []}, tmp_path)
    assert ok["comparable"] and ok["numericas_coinciden"]
    assert M.control_del_pliegue_0("otro", {}, tmp_path)["comparable"] is False


def test_la_fuente_del_85_15_se_localiza_en_el_estudio_real():
    ej = pytest.importorskip("matrixai_studio.estudio_job")
    import inspect
    r = M._la_fuente_dice_85(ej)
    lineas = inspect.getsource(ej).splitlines()
    assert "0.85" in lineas[r["linea_del_corte"] - 1]
    assert r["texto_del_corte"] == lineas[r["linea_del_corte"] - 1].strip()
    # 129-C3 (B1 mínimo): contra un Studio que corrige la cola, el producto ya NO es el posicional, y se dice
    corrige = "contexto" in inspect.signature(ej._reparto_de_entrena).parameters
    assert r["el_producto_corrige_la_cola"] is corrige and r["es_85_posicional"] is (not corrige)
    assert "posicional de B0" in r["mide"] and ("ya no es ese" in r["mide"]) is corrige
    fake = type("F", (), {"_reparto_de_entrena": staticmethod(lambda p: ([], []))})   # sin 0.85: lo dice
    assert M._la_fuente_dice_85(fake)["es_85_posicional"] is False


def _reparto_posicional_de_b0(pliegue):
    ids_train = list(pliegue.entrena)
    corte = int(len(ids_train) * 0.85)
    return ids_train[:corte], ids_train[corte:]


def _reparto_del_b1_minimo(pliegue, contexto=None):
    ids_train = list(pliegue.entrena)
    corte = int(len(ids_train) * 0.85)
    return ids_train[:corte], ids_train[corte:]


def test_la_guarda_del_85_15_no_dice_posicional_contra_un_studio_que_corrige_la_cola():
    """M3 de la auditoría del B1 mínimo: el corte 0.85 sigue en la fuente de los dos, pero solo el de B0 ES el reparto
    del producto; con `contexto` (el B1 mínimo) el mapa mide el posicional de B0 y tiene que decirlo."""
    b0 = M._la_fuente_dice_85(type("B0", (), {"_reparto_de_entrena": staticmethod(_reparto_posicional_de_b0)}))
    b1 = M._la_fuente_dice_85(type("B1", (), {"_reparto_de_entrena": staticmethod(_reparto_del_b1_minimo)}))
    assert b0["linea_del_corte"] and b1["linea_del_corte"]
    assert (b0["es_85_posicional"], b0["el_producto_corrige_la_cola"]) == (True, False)
    assert (b1["es_85_posicional"], b1["el_producto_corrige_la_cola"]) == (False, True)
    assert "ya no es ese" in b1["mide"] and "ya no es ese" not in b0["mide"]


def test_sha256_de_un_fichero(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert M.sha256_de(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
